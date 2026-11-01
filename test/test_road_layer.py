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
