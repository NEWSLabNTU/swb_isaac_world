"""The car's anti-roll bars on a world without the road layer (roadmap 066 R5). Pure."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swb_isaac_world import road_contact as rc  # noqa: E402
from swb_isaac_world import roll_bars as rb  # noqa: E402


def params():
    """aib's VehicleParams, the parts the bars read: Model 3-like axles."""
    front, rear = (430.0, 30000.0, 3000.0), (450.0, 32000.0, 3200.0)
    return SimpleNamespace(suspension=lambda: ((front, rear), 0.14), wheel_radius=0.33,
                           suspension_travel=0.2, wheel_mass=20.0, wheel_base=2.875,
                           wheel_tread=1.58)


class Bars(unittest.TestCase):
    def test_level_car_has_no_bar_force(self):
        b = rb._Bars(params())
        self.assertTrue(np.allclose(rb.forces(b.k, np.full(4, 0.14)), 0.0))

    def test_roll_loads_the_compressed_side_and_resists_it(self):
        """Left wheels compressed 2 cm more than the right: the bars push the body up on the
        left and down on the right, each axle's pair equal and opposite, the front's larger."""
        b = rb._Bars(params())
        f = rb.forces(b.k, np.array([0.16, 0.14, 0.16, 0.14]))
        self.assertGreater(f[0], 0)
        self.assertAlmostEqual(f[0], -f[1])
        self.assertAlmostEqual(f[2], -f[3])
        self.assertGreater(f[0], f[2])  # BAR_FRONT of the bars' stiffness at the front
        bars = rc.anti_roll_bars((0, 1), (2, 3), 30000.0, 32000.0)
        self.assertAlmostEqual(f[0], bars[0][2] * 0.02)

    def test_jounce_from_the_geometry(self):
        x = rb.jounce([0.47, 0.47, 0.47, 0.6], [0.0, 0.01, 0.0, 0.0],
                      [True, True, True, False], 0.33, 0.2)
        self.assertTrue(np.allclose(x, [0.06, 0.07, 0.06, 0.0]))

    def test_qrot_is_aibs(self):
        import math
        h = math.radians(30) / 2
        q = (math.cos(h), 0.0, 0.0, math.sin(h))  # 30 deg about z
        r = rb.qrot(q, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(r[0], math.cos(math.radians(30)))
        self.assertAlmostEqual(r[1], math.sin(math.radians(30)))
        h = math.radians(10) / 2
        up = rb.qrot((math.cos(h), math.sin(h), 0.0, 0.0), (0.0, 0.0, 1.0))  # 10 deg roll
        self.assertAlmostEqual(up[1], -math.sin(math.radians(10)))

    def test_friction_follows_the_load_with_the_bar(self):
        s = rb.friction_scale([4000.0, 4000.0], [800.0, -800.0], 200.0)
        self.assertAlmostEqual(s[0], 5000.0 / 4200.0)
        self.assertAlmostEqual(s[1], 3400.0 / 4200.0)
        self.assertEqual(rb.friction_scale([0.0], [-5000.0], 200.0)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
