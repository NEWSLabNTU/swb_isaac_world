"""`make_world`'s plan and the world file it writes (roadmap 076 A9); no Isaac."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swb_isaac_world import make_world as mw, usd_world  # noqa: E402


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.set = root / "carla-usd"
        self.env = {"SWB_CARLA_USD": str(self.set)}
        self.maps = root / "maps"
        for m in ("Town01", "Town10", "Town05"):
            (self.maps / m).mkdir(parents=True)

    def town(self, name, layer=True):
        d = self.set / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.usd").write_text("export")
        if layer:
            (d / f"{name}_physics.usda").write_text("layer")

    def plan(self, m, **kw):
        return mw.plan(str(self.maps / m), env=self.env, **kw)

    def test_the_town_from_the_set_keeps_its_layer(self):
        self.town("Town01")
        p = self.plan("Town01")
        self.assertEqual(p.town, "Town01")
        self.assertEqual(p.layer, str(self.set / "Town01" / "Town01_physics.usda"))
        self.assertFalse(p.write_layer)
        self.assertIsNone(p.export_to)
        self.assertEqual(p.out, str(self.maps / "Town01" / "isaac" / "world.yaml"))

    def test_town10_is_town10hd(self):
        self.town("Town10HD")
        self.assertEqual(self.plan("Town10").town, "Town10HD")

    def test_an_export_without_a_layer_gets_one_beside_it_or_in_the_state(self):
        self.town("Town05", layer=False)
        p = self.plan("Town05")
        self.assertTrue(p.write_layer)
        self.assertEqual(p.layer, str(self.set / "Town05" / "Town05_physics.usda"))
        state = Path(self.tmp.name) / "state"
        p = self.plan("Town05", state_dir=str(state))
        self.assertEqual(p.layer, str(state / "Town05" / "isaac" / "Town05_physics.usda"))
        self.assertEqual(p.export, str(self.set / "Town05" / "Town05.usd"))

    def test_physics_writes_the_layer_again(self):
        self.town("Town01")
        self.assertTrue(self.plan("Town01", physics=True).write_layer)

    def test_a_missing_town_is_refused_unless_exported(self):
        with self.assertRaises(mw.MakeWorldError) as e:
            self.plan("Town05")
        self.assertIn("--export", str(e.exception))
        p = self.plan("Town05", export=True)
        self.assertEqual(p.export_to, str(self.set / "Town05"))
        self.assertTrue(p.write_layer)

    def test_usd_names_the_stage(self):
        self.town("Town01")
        layer = self.set / "Town01" / "Town01_physics.usda"
        p = self.plan("Town05", usd="${SWB_CARLA_USD}/Town01/Town01_physics.usda")
        self.assertEqual((p.town, p.layer, p.write_layer), ("Town01", str(layer), False))
        p = self.plan("Town05", usd=str(self.set / "Town01" / "Town01.usd"))
        self.assertEqual((p.layer, p.write_layer), (str(layer), False))
        with self.assertRaises(mw.MakeWorldError):
            self.plan("Town05", usd=str(self.set / "Nope.usd"))

    def test_no_set_and_no_usd_is_refused(self):
        with self.assertRaises(mw.MakeWorldError) as e:
            mw.plan(str(self.maps / "Town01"), env={})
        self.assertIn("SWB_CARLA_USD", str(e.exception))
        with self.assertRaises(mw.MakeWorldError):
            mw.plan(str(self.maps / "Nowhere"), env=self.env)

    def test_dry_run(self):
        import io
        from contextlib import redirect_stdout
        from unittest import mock

        self.town("Town01")
        out = io.StringIO()
        with mock.patch.dict("os.environ", self.env), redirect_stdout(out):
            self.assertEqual(mw.main([str(self.maps / "Town01"), "--dry-run"]), 0)
        self.assertIn("layer: " + str(self.set / "Town01" / "Town01_physics.usda"),
                      out.getvalue())


class WorldFileText(unittest.TestCase):
    def test_a_stage_in_the_set_is_named_through_the_environment(self):
        env = {"SWB_CARLA_USD": "/data/carla-usd"}
        self.assertEqual(usd_world.usd_name("/data/carla-usd/Town01/Town01_physics.usda",
                                            "/maps/Town01", env),
                         "${SWB_CARLA_USD}/Town01/Town01_physics.usda")
        self.assertEqual(usd_world.usd_name("/maps/Town01/isaac/w.usd", "/maps/Town01", env),
                         "isaac/w.usd")
        self.assertEqual(usd_world.usd_name("/elsewhere/w.usd", "/maps/Town01", env),
                         "/elsewhere/w.usd")

    def test_the_text_reads_back_as_the_server_reads_it(self):
        import yaml

        from swb_isaac_world import names

        with tempfile.TemporaryDirectory() as d:
            wf = Path(d) / "Town01" / "isaac" / "world.yaml"
            wf.parent.mkdir(parents=True)
            wf.write_text(usd_world.world_yaml("${SWB_CARLA_USD}/Town01/T.usda",
                                               ["/World/env/Road"]))
            data = yaml.safe_load(wf.read_text())
            self.assertEqual(data["ground_paths"], ["/World/env/Road"])
            self.assertEqual(names.world_usd_of(str(wf), env={"SWB_CARLA_USD": "/s"}),
                             "/s/Town01/T.usda")


if __name__ == "__main__":
    unittest.main()
