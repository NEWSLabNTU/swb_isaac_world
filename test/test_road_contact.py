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


class CornerForces(unittest.TestCase):
    """R2: the damper, the anti-roll bars and PhysX's longer travel."""

    def test_damper_is_digressive_and_rebounds_harder(self):
        cn = rc.Corner(SPRUNG, K, C, TRAVEL, R, WIDTH)
        bump, reb = cn.damper(0.05), -cn.damper(-0.05)
        self.assertAlmostEqual((bump + reb) / 2, C * 0.05, delta=1.0)  # the rig's c on average
        self.assertAlmostEqual(reb / bump, 3.0, places=6)
        # beyond the knee the slope falls to damper_high of the low one
        slope = (cn.damper(1.0) - cn.damper(0.5)) / 0.5
        self.assertAlmostEqual(slope, cn.damper(0.1) / 0.1 * cn.damper_high, places=3)

    def test_longer_physx_travel_changes_nothing_at_rest(self):
        for px in (TRAVEL, 0.30):
            tc = rc.TireContact([rc.Corner(SPRUNG, K, C, TRAVEL, R, WIDTH, px_travel=px)])
            z = np.array([R + px - SAG])
            for _ in range(30):
                o = tc.step(DT, flat, flat, np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]),
                            np.array([0.0]), z, np.zeros(1))
            self.assertLess(abs(o["dfz"][0]), 5.0)
            self.assertAlmostEqual(o["jounce"][0], SAG, delta=1e-3)
            self.assertFalse(o["px_limit"][0])

    def test_bars_resist_roll_only(self):
        corners = [rc.Corner(SPRUNG, K, C, TRAVEL, R, WIDTH) for _ in range(2)]
        bars = rc.anti_roll_bars((0, 1), (0, 1), K, K)[:1]
        tc = rc.TireContact(corners, bars=bars)
        free = rc.TireContact(corners)
        xy = np.array([[0.0, 0.8], [0.0, -0.8]])
        hd = np.array([[1.0, 0.0], [1.0, 0.0]])
        za = np.full(2, R + TRAVEL - SAG)

        def settle(t, dz):
            t.reset()
            for _ in range(60):
                o = t.step(DT, flat, flat, xy, hd, np.zeros(2), za + dz, np.zeros(2))
            return o["dfz"]

        roll = np.array([-0.02, 0.02])  # left down (compressed), right up
        with_bars, without = settle(tc, roll), settle(free, roll)
        extra = with_bars - without
        self.assertGreater(extra[0], 0.0)  # pushes the compressed side up
        self.assertAlmostEqual(extra[0], -extra[1], delta=1.0)  # and the other down: no heave
        self.assertLess(np.abs(settle(tc, np.full(2, 0.02)) - settle(free, np.full(2, 0.02))).max(),
                        1.0)  # heave: the bar does nothing
