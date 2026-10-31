"""The tire's vertical contact (roadmap 066 R5); no Isaac needed."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swb_isaac_world import road_contact as rc  # noqa: E402

# aib's Model 3 front corner
SPRUNG, K, C, TRAVEL, SAG, R, WIDTH = 413.6, 41.8e3, 3.74e3, 0.20, 0.097, 0.37, 0.235
DT = 1.0 / 60


def flat(x, y):
    return np.zeros(np.broadcast(x, y).shape)


def step_up(h):
    return lambda x, y: np.where((x >= 10.0) & (x <= 10.2), h, 0.0) + 0.0 * y


def drive(height, speed=10.0, x_end=14.0):
    """A stand-in for PhysX's corner on a flat base: the sprung mass on the rig's spring and
    damper, 8 substeps per step, plus the model's force; (tire loads, body rise, model)."""
    tc = rc.TireContact([rc.Corner(SPRUNG, K, C, TRAVEL, R, WIDTH)])
    z_rest = R + TRAVEL - SAG
    z, vz, x, loads, rise = z_rest, 0.0, 5.0, [], 0.0
    while x < x_end:
        o = tc.step(DT, height, flat, np.array([[x, 0.0]]), np.array([[1.0, 0.0]]),
                    np.array([speed]), np.array([z]), np.array([vz]))
        for _ in range(8):
            j = TRAVEL + R - z
            fp = max(K * min(j, TRAVEL) - C * vz, 0.0) if j > 0 else 0.0
            vz += ((fp + o["dfz"][0]) / SPRUNG - rc.G) * DT / 8
            z += vz * DT / 8
        loads.extend(o["load_inner"][0])
        rise = max(rise, z - z_rest)
        x += speed * DT
    return np.array(loads), rise, o


class RoadContactTest(unittest.TestCase):
    def test_flat_road_adds_nothing(self):
        loads, rise, o = drive(flat)
        static = (SPRUNG + 42.0) * rc.G
        self.assertLess(abs(loads.mean() - static) / static, 1e-3)
        self.assertLess(abs(o["dfz"][0]), 5.0)
        self.assertLess(rise, 1e-4)

    def test_cleat_lifts_the_body_and_unloads_the_wheel(self):
        loads, rise, _ = drive(step_up(0.10))
        self.assertGreater(loads.max(), 15e3)  # tens of kN, not PhysX's 4.4
        self.assertEqual(loads.min(), 0.0)  # the wheel leaves the road behind the cleat
        self.assertGreater(rise, 0.05)

    def test_envelope_rises_ahead_of_a_step(self):
        tc = rc.TireContact([rc.Corner(SPRUNG, K, C, TRAVEL, R, WIDTH)])
        s, w, slope = tc.envelope(step_up(0.10), np.array([[9.0, 0.0]]),
                                  np.array([[1.0, 0.0]]), 2.0)
        first = s[np.nonzero(w[0] > 1e-3)[0][0]] + 9.0
        self.assertTrue(9.6 < first < 9.8)  # a 0.37 m disc meets a 10 cm step ~0.25 m early
        self.assertAlmostEqual(float(w[0].max()), 0.10, places=3)
        self.assertGreater(float(slope[0].max()), 0.0)

    def test_friction_follows_the_load(self):
        tc = rc.TireContact([rc.Corner(SPRUNG, K, C, TRAVEL, R, WIDTH)])
        px = np.array([SPRUNG * rc.G])
        self.assertAlmostEqual(float(tc.friction_scale(px + 42.0 * rc.G, px)[0]), 1.0)
        self.assertEqual(float(tc.friction_scale(np.zeros(1), px)[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
