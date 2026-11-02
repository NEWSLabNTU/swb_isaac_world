"""The rock stream's sleep rule (roadmap 066 R4, issue 031), with a stand-in for PhysX."""
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


if __name__ == "__main__":
    unittest.main()
