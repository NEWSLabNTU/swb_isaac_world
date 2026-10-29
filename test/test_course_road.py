"""The course's road ribbon (roadmap 058, docs/issues/024 of swb); no Isaac needed."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swb_isaac_world.course import geometry  # noqa: E402


def hairpin():
    """A 10 m bend through 180 deg, climbing 6 %, sampled every 0.5 m as the centreline is."""
    s = np.arange(0.0, math.pi * 10.0, 0.5)
    th = s / 10.0
    return {"x": 10.0 * np.sin(th), "y": 10.0 - 10.0 * np.cos(th), "z": 0.06 * s}


def worst_fold(points, faces):
    """The largest angle (deg) between the two triangles a quad is split into (0-2)."""
    a, b, c, d = (points[faces[:, k]] for k in range(4))
    n1, n2 = np.cross(b - a, c - a), np.cross(c - a, d - a)
    n1 /= np.linalg.norm(n1, axis=1)[:, None]
    n2 /= np.linalg.norm(n2, axis=1)[:, None]
    return float(np.degrees(np.arccos(np.clip((n1 * n2).sum(1), -1, 1))).max())


class RoadRibbon(unittest.TestCase):
    def test_a_climbing_hairpin_in_strips_is_nearly_flat_per_quad(self):
        c = hairpin()
        one = geometry.ribbon(c, 4.0, -4.0)
        strips = geometry.ribbon(c, 4.0, -4.0, strip=geometry.ROAD_STRIP)
        self.assertGreater(worst_fold(*one), 2.5)      # one 8 m quad across: folded
        self.assertLess(worst_fold(*strips), 0.7)

    def test_strips_tile_the_road_once(self):
        c = hairpin()
        pts, faces = geometry.ribbon(c, 4.0, -4.0, strip=0.5)
        n = len(c["x"])
        self.assertEqual(pts.shape, (17 * n, 3))
        self.assertEqual(faces.shape, (16 * (n - 1), 4))
        # every quad wound one way, as the single ribbon's (usd's _face_up turns them up)
        one, one_faces = geometry.ribbon(c, 4.0, -4.0)
        a, b, cc = (pts[faces[:, k]] for k in range(3))
        oa, ob, oc = (one[one_faces[:, k]] for k in range(3))
        self.assertEqual(set(np.sign(np.cross(b - a, cc - a)[:, 2])),
                         set(np.sign(np.cross(ob - oa, oc - oa)[:, 2])))
        # the edges are the single ribbon's
        np.testing.assert_allclose(pts[:n], one[:n])
        np.testing.assert_allclose(pts[-n:], one[n:])


if __name__ == "__main__":
    unittest.main()
