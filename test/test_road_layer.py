"""The road detail in the world server (roadmap 066 R3), with a stand-in for aib's vehicle."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deps/autoware_isaac_bridge/src/aib"))

from swb_isaac_world import road_layer  # noqa: E402

DT = 1.0 / 60


def params(travel=0.2):
    def suspension():
        m = 1760 * 0.47 / 2, 1760 * 0.53 / 2
        w = 2 * math.pi * 1.6
        return tuple((s, s * w * w, 2 * 0.45 * math.sqrt(s * w * w * s)) for s in m), 9.81 / w / w
    return SimpleNamespace(suspension_travel=travel, wheel_radius=0.37, wheel_width=0.235,
                           wheel_base=3.005, wheel_tread=1.667, suspension=suspension)


class Parse(unittest.TestCase):
    def test_a_corridor(self):
        c, = road_layer.parse({"corridors": [{
            "centreline": [[0, 0, 10], [100, 0, 12]], "seed": 4, "roughness": {"cls": "D"},
            "section": {"crown": 0.0, "half_width": 1.5}, "ruts": {},
            "washboard": [{"u0": 10, "u1": 20}], "water_bars": {"at": [50]}}]})
        self.assertEqual(c.roughness.cls, "D")
        self.assertEqual(c.section.crown, 0.0)
        self.assertAlmostEqual(float(c.base_z(50.0)), 11.0)
        self.assertEqual(len(c.washboard), 1)

    def test_problems_are_listed(self):
        with self.assertRaises(road_layer.RoadDetailError) as e:
            road_layer.parse({"corridors": [{"centreline": [[0, 0]]},
                                            {"centreline": [[0, 0], [1, 0]], "ruts": {"x": 1},
                                             "roughness": {"cls": "Z"}, "bumps": 1}],
                              "other": 1})
        msg = str(e.exception)
        for part in ("unknown key(s) ['other']", "centreline must be", "ruts: unknown ['x']",
                     "class Z", "['bumps']"):
            self.assertIn(part, msg)


class Schema(unittest.TestCase):
    """road_detail.yaml's keys, documented and refused by name (roadmap 076 A9)."""

    def test_every_key_is_listed(self):
        keys = road_layer.keys()
        for k in ("corridors", "ground", "corridors[].centreline", "corridors[].taper",
                  "corridors[].washboard[].*", "corridors[].rocks.*"):
            self.assertIn(k, keys)
        self.assertEqual(keys["corridors[].washboard[].*"]["u0"], road_layer.REQUIRED)
        self.assertEqual(keys["corridors[].section.*"]["half_width"], 2.5)

    def test_washboard_and_shapes_are_refused_by_name(self):
        line = [[0, 0], [100, 0]]
        with self.assertRaises(road_layer.RoadDetailError) as e:
            road_layer.parse({"corridors": [
                {"centreline": line, "washboard": [{"u0": 1, "u1": 2, "wave": 3}, {"u0": 1}]},
                {"centreline": line, "ruts": [1, 2]},
                "nope"]})
        msg = str(e.exception)
        for part in ("washboard[0]: unknown ['wave']", "washboard[1]: needs u1",
                     "corridors[1].ruts: a mapping", "corridors[2]: not a mapping"):
            self.assertIn(part, msg)
        with self.assertRaises(road_layer.RoadDetailError):
            road_layer.parse([1, 2])

    def test_the_examples_parse(self):
        ex = ROOT / "swb_isaac_world" / "examples" / "road"
        files = sorted(ex.glob("*/road_detail.yaml"))
        self.assertGreaterEqual(len(files), 2)
        for f in files:
            self.assertTrue(road_layer.load(f), f)
        town, = road_layer.load(ex / "town01" / "road_detail.yaml")
        self.assertIsNotNone(town.rocks)
        self.assertEqual(road_layer.find(str(ex / "town01")),
                         str(ex / "town01" / "road_detail.yaml"))

    def test_check_and_profile(self):
        import contextlib
        import io

        ex = ROOT / "swb_isaac_world" / "examples" / "road" / "town01"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(road_layer.main(["check", str(ex)]), 0)
        self.assertIn("1 corridor(s)", out.getvalue())
        self.assertIn("washboard 1 zone(s)", out.getvalue())
        c, = road_layer.load(ex / "road_detail.yaml")
        u, v, h = road_layer.profile(c, length=5.0)
        self.assertEqual(h.shape, (len(v), len(u)))
        self.assertTrue(np.isfinite(h).all())
        self.assertLess(h.min(), -0.2)  # the ditches


class Layer(unittest.TestCase):
    def car(self, z0=0.0, grade=0.0, x=10.0):
        """A vehicle standing at rest on a base of `grade`, PhysX's frame raised (R2)."""
        p = params(0.2 + road_layer.PX_EXTRA)
        (_, _), sag = p.suspension()
        frame_z = p.wheel_radius + p.suspension_travel - sag
        pitch = -math.atan(grade)
        q = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
        hits = []
        for f, side in road_layer.ORDER:
            hx = x + (p.wheel_base if f else 0.0) * math.cos(pitch)
            hits.append(SimpleNamespace(on_ground=True, position=(hx, side * 0.83,
                                                                  z0 + grade * hx)))
        pos = (x, 0.0, z0 + grade * x + frame_z / math.cos(pitch) - frame_z)
        return SimpleNamespace(path="/ego", p=p, last=(pos, q, (0.0, 0.0, 0.0)), contacts=hits,
                               wheel_paths=[0] * 4, tire_angle=0.0)

    def test_no_detail_no_force_on_a_grade(self):
        """A flat corridor with no roughness on a 10 % base: the plane base, no push."""
        c = road_layer.parse({"corridors": [{"centreline": [[0, 0], [200, 0]],
                                             "roughness": {"gd0": 1e-14},
                                             "section": {"crown": 0.0}}]})
        forces = []
        layer = road_layer.RoadLayer(c, apply_force=lambda *a: forces.append(a), workers=0)
        v = self.car(z0=5.0, grade=0.1)
        for _ in range(30):
            forces.clear()
            layer.before_substep([v], DT)
        fz = [f[1][2] for f in forces]
        self.assertEqual(len(fz), 4)
        # what is left is the stand-in pose's own sag error through the tire's series spring
        # (k / k_tire of it): under 3 % of a corner's weight, and the same left and right
        self.assertLess(max(abs(f) for f in fz), 150.0, fz)
        self.assertAlmostEqual(fz[0], fz[1], delta=1.0)
        for i in range(4):
            self.assertAlmostEqual(layer.scale("/ego", i), 1.0, delta=0.05)
        layer.close()

    def test_a_bar_under_a_wheel_pushes_it_up(self):
        c = road_layer.parse({"corridors": [{"centreline": [[0, 0], [200, 0]],
                                             "roughness": {"gd0": 1e-14},
                                             "section": {"crown": 0.0},
                                             "water_bars": {"at": [10.0], "angle": 0.0,
                                                            "height": 0.05}}]})
        forces = []
        layer = road_layer.RoadLayer(c, apply_force=lambda *a: forces.append(a), workers=0)
        v = self.car(x=10.0 - 0.0)  # the rear wheels on the bar
        for _ in range(30):
            forces.clear()
            layer.before_substep([v], DT)
        fz = {i: f[1][2] for i, f in enumerate(forces)}
        self.assertGreater(min(fz[2], fz[3]), 1000.0)  # rear: tens of mm of extra compression
        self.assertLess(max(abs(fz[0]), abs(fz[1])), 100.0)
        layer.close()

    def test_a_rock_under_a_wheel_lifts_it_and_is_pushed(self):
        """The rocks with the layer as the world runs it: a stand-in stream (no USD) and one
        loose rock under the rear left wheel."""
        from swb_isaac_world import rocks as rk

        c = road_layer.parse({"corridors": [{"centreline": [[0, 0], [200, 0]],
                                             "roughness": {"gd0": 1e-14},
                                             "section": {"crown": 0.0},
                                             "rocks": {"carriageway": 0.0, "shoulder": 0.0,
                                                       "ditch": 0.0}}]})
        self.assertIsInstance(c[0].rocks, rk.Zones)
        forces = []
        layer = road_layer.RoadLayer(c, apply_force=lambda *a: forces.append(a), workers=0)
        self.assertIsNotNone(layer.rocks)
        rock = rk.make_rock(np.random.default_rng(1), 0.12, rough=0.0)
        layer.rocks.add([rock], [(10.0, 0.83, -rock.verts[:, 2].min())], [(1, 0, 0, 0)])
        pushed = []

        class Stream:  # the part of rock_layer.RockStream the car step uses
            def read_near(self, xy, road, dt):
                layer.rocks.seen[:] = True

            def push(self, forces_, speed, radius, road, dt, apply_force):
                pushed.append(forces_)
                return 1.0

        layer.rock_stream = Stream()
        v = self.car(x=10.0)  # the rear left wheel on the rock
        for _ in range(30):
            forces.clear()
            layer.before_substep([v], DT)
        fz = [f[1][2] for f in forces[:4]]
        self.assertGreater(fz[2], 500.0)  # rear left lifted by the 6 cm rock
        self.assertLess(abs(fz[0]), 150.0)
        self.assertTrue(pushed and 0 in pushed[-1])  # and the rock takes its reaction
        f = sum(ff for ff, _ in pushed[-1][0])
        self.assertLess(f[2], -1000.0)  # pressed down by the tire
        layer.close()


class MeshFaces(unittest.TestCase):
    """The render mesh's faces: only the corridors', road and verge apart (no Isaac)."""

    def test_faces_cover_the_corridor_only(self):
        from swb_isaac_world import road_detail as rd

        c = rd.Corridor(np.array([[0.0, 0.0], [400.0, 0.0]]), seed=1)
        d = rd.Detail([c])
        ms = road_layer.MeshStream(None, SimpleNamespace(detail=d, ox=0.0, oy=0.0))
        quads, road = ms.tile_faces(d.zones_tile(2, -1))
        n, s = d.n, d.spacing
        ys = -16.0 + (np.arange(n + 1) * s)[quads // (n + 1)].mean(axis=1)
        self.assertLess(ys.min(), -c.reach + 0.2)  # out to the reach
        self.assertGreater(ys.min(), -c.reach - 0.2)  # and no further
        edge = c.section.half_width + c.section.shoulder
        self.assertTrue((np.abs(ys[road]) <= edge + s).all())
        self.assertTrue((np.abs(ys[~road]) >= edge - s).all())
        self.assertGreater(road.sum(), 0)
        self.assertGreater((~road).sum(), 0)


class PlaneFit(unittest.TestCase):
    def test_the_closed_form_is_lstsq(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            n = int(rng.integers(3, 5))
            x = 300.0 + rng.uniform(-2, 2, n)  # world coordinates, far from the origin
            y = -150.0 + rng.uniform(-1, 1, n)
            z = 0.02 * x - 0.03 * y + rng.normal(0, 0.01, n)
            a = np.stack([x, y, np.ones(n)], axis=1)
            (ga, gb, _), *_ = np.linalg.lstsq(a, z, rcond=None)
            fa, fb = road_layer._fit_slope(x, y, z)
            self.assertAlmostEqual(fa, ga, delta=1e-7)
            self.assertAlmostEqual(fb, gb, delta=1e-7)

    def test_collinear_points_fall_back_to_lstsq(self):
        x = np.array([0.0, 1.0, 2.0])
        y = 2.0 * x
        z = np.array([0.0, 0.1, 0.2])
        a = np.stack([x, y, np.ones(3)], axis=1)
        (ga, gb, _), *_ = np.linalg.lstsq(a, z, rcond=None)
        fa, fb = road_layer._fit_slope(x, y, z)
        self.assertAlmostEqual(fa, ga, places=9)
        self.assertAlmostEqual(fb, gb, places=9)
