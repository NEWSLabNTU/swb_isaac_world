"""Names, not paths, and the state directory (roadmap 076 A9); no Isaac."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swb_isaac_world import names, world_server  # noqa: E402

try:
    from pxr import Usd
except ImportError:  # outside Isaac's Python
    Usd = None

COURSE_USDA = '''#usda 1.0
(defaultPrim = "Course")
def Xform "Course" {
    def Xform "rail" (references = @./carla-usd/Props/rail.usda@) {
    }
}
'''


class Expand(unittest.TestCase):
    def test_variables_and_home(self):
        env = {"SWB_CARLA_USD": "/data/carla-usd", "HOME": "/home/u", "X": "x"}
        self.assertEqual(names.expand("${SWB_CARLA_USD}/Town01/T.usda", env),
                         "/data/carla-usd/Town01/T.usda")
        self.assertEqual(names.expand("$X/a", env), "x/a")
        self.assertEqual(names.expand("~/a", env), os.path.expanduser("~/a"))
        self.assertEqual(names.expand("plain/path", env), "plain/path")

    def test_an_unset_variable_is_refused_by_name(self):
        with self.assertRaises(names.UnresolvedName) as e:
            names.expand("${SWB_CARLA_USD}/Town01/T.usda", {})
        self.assertIn("SWB_CARLA_USD not set", str(e.exception))
        with self.assertRaises(names.UnresolvedName) as e:
            names.expand("$NOPE/a", {})
        self.assertIn("NOPE", str(e.exception))

    def test_the_older_host_setting_names_the_set(self):
        self.assertEqual(names.expand("${SWB_CARLA_USD}/a", {"CARLA_USD": "/old"}), "/old/a")
        self.assertEqual(names.carla_usd_root({"CARLA_USD": "/old"}), "/old")
        self.assertEqual(names.carla_usd_root({"CARLA_USD": "/old", "SWB_CARLA_USD": "/new"}),
                         "/new")


class WorldFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.set = self.root / "carla-usd"
        (self.set / "Town01").mkdir(parents=True)
        (self.set / "Town01" / "Town01_physics.usda").write_text("#usda 1.0\n")
        self.map = self.root / "maps" / "Town01"
        (self.map / "isaac").mkdir(parents=True)
        self.env = {"SWB_CARLA_USD": str(self.set)}

    def world(self, usd):
        (self.map / "isaac" / "world.yaml").write_text(f"usd: {usd}\nmap_origin: [0, 0, 0]\n")
        return str(self.map / "isaac" / "world.yaml")

    def test_a_world_files_usd_is_expanded(self):
        wf = self.world("${SWB_CARLA_USD}/Town01/Town01_physics.usda")
        self.assertEqual(names.world_usd_of(wf, env=self.env),
                         str(self.set / "Town01" / "Town01_physics.usda"))

    def test_a_relative_usd_is_the_map_directorys(self):
        wf = self.world("isaac/course.usdc")
        self.assertEqual(names.world_usd_of(wf, env=self.env),
                         str(self.map / "isaac" / "course.usdc"))

    def test_the_world_servers_preload_expands_it(self):
        self.world("${SWB_CARLA_USD}/Town01/Town01_physics.usda")
        clean = {k: v for k, v in os.environ.items() if k not in ("SWB_CARLA_USD", "CARLA_USD")}
        with mock.patch.dict(os.environ, dict(clean, SWB_CARLA_USD=str(self.set)), clear=True):
            self.assertEqual(world_server.preload_list(str(self.map)),
                             [str(self.set / "Town01" / "Town01_physics.usda")])
        with mock.patch.dict(os.environ, clean, clear=True):
            with self.assertRaises(SystemExit) as e:
                world_server.preload_list(str(self.map))
            self.assertIn("SWB_CARLA_USD", str(e.exception))

    def test_a_requested_world_the_bridge_joined_unexpanded_is_found_again(self):
        self.world("${SWB_CARLA_USD}/Town01/Town01_physics.usda")
        joined = str(self.map / "${SWB_CARLA_USD}" / "Town01" / "Town01_physics.usda")
        want = str(self.set / "Town01" / "Town01_physics.usda")
        self.assertEqual(names.requested_usd(joined, str(self.map), self.env), want)
        self.assertEqual(names.requested_usd("${SWB_CARLA_USD}/Town01/Town01_physics.usda",
                                             None, self.env), want)
        self.assertEqual(names.requested_usd("", str(self.map), self.env), "")

    def test_state_and_map_files(self):
        self.assertIsNone(names.state_for(None, self.map))
        state = names.state_for(str(self.root / "state"), str(self.map))
        self.assertEqual(state, str(self.root / "state" / "Town01"))
        self.assertEqual(names.find_file("surfaces.yaml", str(self.map), state),
                         str(self.map / "surfaces.yaml"))
        Path(state).mkdir(parents=True)
        (Path(state) / "surfaces.yaml").write_text("defaults: {}\n")
        self.assertEqual(names.find_file("surfaces.yaml", str(self.map), state),
                         str(Path(state) / "surfaces.yaml"))
        self.assertEqual(names.find_file("road_detail.yaml", str(self.map), state),
                         str(self.map / "road_detail.yaml"))

    def test_the_map_name_is_the_linked_directorys(self):
        cache = self.root / "cache" / "abc123" / "Town01"
        cache.mkdir(parents=True)
        link = self.root / "assets" / "maps" / "MyTown"
        link.parent.mkdir(parents=True)
        link.symlink_to(cache, target_is_directory=True)
        self.assertEqual(names.map_name(str(link)), "MyTown")


class CourseProps(unittest.TestCase):
    """A course world's props through a `carla-usd` link that a dataset cannot carry."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.set = root / "carla-usd"
        (self.set / "Props").mkdir(parents=True)
        (self.set / "Props" / "rail.usda").write_text(
            '#usda 1.0\n(defaultPrim = "Rail")\ndef Xform "Rail" {\n    def Cube "c" {\n    }\n}\n')
        self.isaac = root / "cache" / "haruna" / "isaac"
        self.isaac.mkdir(parents=True)
        self.usd = self.isaac / "haruna.usda"
        self.usd.write_text(COURSE_USDA)
        (self.isaac / "photo.png").write_bytes(b"png")
        self.state = root / "state" / "haruna"
        self.env = {"SWB_CARLA_USD": str(self.set), "XDG_CACHE_HOME": str(root / "xdg")}

    def test_the_link_is_seen_in_the_layer(self):
        self.assertTrue(names.uses_link(str(self.usd)))
        self.assertFalse(names.uses_link(str(self.usd), limit=10))
        other = self.isaac / "plain.usda"
        other.write_text('#usda 1.0\ndef Xform "W" {\n}\n')
        self.assertFalse(names.uses_link(str(other)))
        # the marker split over two reads is still found
        big = self.isaac / "big.usdc"
        big.write_bytes(b"x" * ((1 << 20) - 4) + b"./carla-usd/a.usd")
        self.assertTrue(names.uses_link(str(big)))

    def test_a_world_with_its_link_is_loaded_as_it_is(self):
        (self.isaac / "carla-usd").symlink_to(self.set, target_is_directory=True)
        self.assertEqual(names.staged_world(str(self.usd), str(self.state), self.env),
                         str(self.usd))

    def test_a_world_without_it_is_loaded_from_a_stand_in(self):
        os.chmod(self.isaac, 0o555)  # a read-only cache
        self.addCleanup(os.chmod, self.isaac, 0o755)
        got = names.staged_world(str(self.usd), str(self.state), self.env)
        self.assertEqual(got, str(self.state / "isaac" / "haruna.usda"))
        here = self.state / "isaac"
        self.assertEqual(os.readlink(here / "carla-usd"), str(self.set))
        self.assertEqual(os.readlink(here / "haruna.usda"), str(self.usd))
        self.assertEqual((here / "photo.png").read_bytes(), b"png")
        self.assertTrue((here / "carla-usd" / "Props" / "rail.usda").exists())
        self.assertFalse((self.isaac / "carla-usd").exists())  # the map is untouched
        # again: the same, and a moved set is followed
        self.assertEqual(names.staged_world(str(self.usd), str(self.state), self.env), got)
        moved = Path(self.tmp.name) / "carla-usd-2"
        self.set.rename(moved)
        names.staged_world(str(self.usd), str(self.state), dict(self.env, SWB_CARLA_USD=str(moved)))
        self.assertEqual(os.readlink(here / "carla-usd"), str(moved))

    def test_without_a_state_directory_the_stand_in_is_in_the_cache(self):
        got = names.staged_world(str(self.usd), None, self.env)
        self.assertTrue(got.startswith(str(Path(self.tmp.name) / "xdg" / "swb_isaac_world")))
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(got), "carla-usd",
                                                    "Props", "rail.usda")))

    @unittest.skipUnless(Usd, "pxr: run under Isaac's Python")
    def test_usd_finds_the_props_through_the_stand_in(self):
        self.assertFalse(Usd.Stage.Open(str(self.usd)).GetPrimAtPath("/Course/rail/c"))
        got = names.staged_world(str(self.usd), str(self.state), self.env)
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World/env").GetReferences().AddReference(got)
        self.assertTrue(stage.GetPrimAtPath("/World/env/rail/c"))

    def test_no_set_named_is_refused(self):
        with self.assertRaises(names.UnresolvedName) as e:
            names.staged_world(str(self.usd), str(self.state), {})
        self.assertIn("SWB_CARLA_USD", str(e.exception))


if __name__ == "__main__":
    unittest.main()
