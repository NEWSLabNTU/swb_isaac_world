"""The rock stream's sleep rule (roadmap 066 R4, issue 031), with a stand-in for PhysX."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swb_isaac_world import rock_layer  # noqa: E402
from swb_isaac_world import rocks as rk  # noqa: E402


class PhysX:
    """Rock 0's pose as PhysX reports it, moved by the test."""
    def __init__(self):
        self.pos = (0.0, 0.0, 0.5)

    def get_rigidbody_transformation(self, path):
        return {"position": self.pos, "rotation": (0.0, 0.0, 0.0, 1.0)}


def stream():
    rocks = rk.RockSet([rk.scaled(0, 0.12)], np.array([[0.0, 0.0, 0.5]]), [(1, 0, 0, 0)])
    s = rock_layer.RockStream.__new__(rock_layer.RockStream)
    s.layer = SimpleNamespace(rocks=rocks)
    s.physx = PhysX()
    s.paths = ["/World/RoadRocks/r1"]
    s.vel, s.read_at, s.reads, s.settling = {}, {}, 0, {}
    return s


class Settle(unittest.TestCase):
    def test_a_flying_rock_does_not_sleep(self):
        s = stream()
        self.assertFalse(s._settled(0))  # the first look has nothing to compare with
        for z in (0.45, 0.35, 0.2, 0.05):  # falling, a frame apart
            s.physx.pos = (0.0, 0.0, z)
            self.assertFalse(s._settled(0))
        s.physx.pos = (0.0, 0.0, 0.052)  # come to rest
        self.assertTrue(s._settled(0))
        self.assertAlmostEqual(s.layer.rocks.pos[0, 2], 0.052)  # sleeps where PhysX has it
        self.assertEqual(s.settling, {})


class Wake(unittest.TestCase):
    """Which rocks wake for a car at the origin heading +x: on its tire tracks, or between them
    only when tall enough to reach its underbody."""

    def stream(self, rocks, pos):
        s = rock_layer.RockStream.__new__(rock_layer.RockStream)
        s.layer = SimpleNamespace(rocks=rk.RockSet(rocks, np.array(pos), [(1, 0, 0, 0)] * len(pos)))
        s.paths = [f"/r{i}" for i in range(len(pos))]
        s.win, s.loaded = {(0, 0): None}, {(0, 0): list(range(len(pos)))}
        s.tall = np.array([float(np.ptp(r.verts[:, 2])) for r in rocks])
        s.active, s.settling, s.prev = {}, {}, []
        s._sleep = lambda i, on: None
        return s

    def test_tracks_and_underbody(self):
        small, big = rk.scaled(0, 0.04), rk.scaled(0, 0.25)
        self.assertLess(np.ptp(small.verts[:, 2]), 0.08)
        self.assertGreater(np.ptp(big.verts[:, 2]), 0.09)
        pos = [(6.0, 0.79, 0.0), (6.0, -0.79, 0.0), (6.0, 0.0, 0.0), (6.0, 0.0, 0.0),
               (6.0, 1.6, 0.0), (30.0, 0.79, 0.0), (-5.0, 0.79, 0.0)]
        s = self.stream([small, small, small, big, small, small, small], pos)
        s._activate([(0.0, 0.0, 0.0, 10.0, 0.79, 0.12, 0.138)])
        self.assertEqual([bool(s.active.get(i)) for i in range(7)],
                         [True, True, False, True, False, False, False])

    def test_the_tracks_follow_the_turn(self):
        small = rk.scaled(0, 0.04)
        # turning left on a 20 m radius: 10 m ahead the left track is 2.5 m to the left
        y = 20.0 - math.sqrt(20.0 ** 2 - 10.0 ** 2)
        s = self.stream([small, small], [(10.0, y + 0.79, 0.0), (10.0, 0.79, 0.0)])
        s.prev = [(-1.0, 0.0, -1.0 / 20.0)]  # a metre back, 1/20 rad less: the turn
        s._activate([(0.0, 0.0, 0.0, 15.0, 0.79, 0.12, 0.138)])
        self.assertEqual([bool(s.active.get(i)) for i in range(2)], [True, False])


class Groups(unittest.TestCase):
    def test_a_frame_edits_a_group_once(self):
        s = rock_layer.RockStream.__new__(rock_layer.RockStream)
        s.pending = {}
        s._group("g", "/a", True)
        s._group("g", "/b", True)
        s._group("g", "/a", False)  # made and released in one frame: nothing to add
        s._group("g", "/c", False)
        self.assertEqual(s.pending, {"g": ({"/b"}, {"/a", "/c"})})


if __name__ == "__main__":
    unittest.main()


class Prewarm(unittest.TestCase):
    """The pool made at the road's load (issue 032's cost, R4): its sizes are the ones placement
    draws, and its count is what a window of the road holds."""

    def test_window_matches_placement(self):
        from swb_isaac_world import road_detail as rd

        c = rd.Corridor(np.array([[0.0, 0.0], [2000.0, 0.0]]), seed=5,
                        roughness=rd.Roughness(cls="C"), rocks=rk.Zones())
        d = rd.Detail([c])
        lam = rock_layer.expected_window(d)
        band = c.section.half_width + c.section.shoulder + rock_layer.BAND_MARGIN
        loose = []
        for i in range(int(2000 / d.tile)):
            for j in range(-2, 2):
                for r, p, _, dyn in rk.place(d, i, j):
                    if dyn and abs(p[1]) <= band:
                        loose.append((r, p[0]))
        sizes = {2 * x / (1 + rk.AXES[1]) for x in lam}
        for r, _ in loose:  # every loose rock's size is one the pool is made in
            self.assertIn(r.scale, sizes)
        window = 2 * rock_layer.ROCK_REACH + d.tile
        per = len(loose) / (2000.0 / window)
        self.assertAlmostEqual(per, sum(lam.values()), delta=0.2 * sum(lam.values()))
