"""Rocks' hulls, their top surface and the tire's load on them (roadmap 066 R4); no Isaac."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swb_isaac_world import rocks as rk  # noqa: E402

def flat(x, y):
    return np.zeros(np.broadcast(x, y).shape)


class Hull(unittest.TestCase):
    def test_a_rock_is_its_size_and_mass(self):
        r = rk.make_rock(np.random.default_rng(1), 0.30)
        self.assertLessEqual(len(r.verts), rk.MAX_VERTS)
        ext = r.verts.max(axis=0) - r.verts.min(axis=0)
        self.assertAlmostEqual(max(ext[:2]) / 0.30, 1.0, delta=0.2)
        self.assertAlmostEqual(ext[2] / 0.15, 1.0, delta=0.25)  # short axis vertical
        # an ellipsoid of the same axes, within the noise and the hull's facets
        ell = 4 / 3 * math.pi * 0.15 * 0.105 * 0.075
        self.assertAlmostEqual(r.volume / ell, 0.9, delta=0.15)
        self.assertAlmostEqual(r.mass, r.volume * rk.DENSITY)
        np.testing.assert_allclose(np.abs(r.planes[:, :3]).max(axis=1) > 0, True)
        # every vertex inside every plane
        self.assertLess((r.verts @ r.planes[:, :3].T - r.planes[:, 3]).max(), 1e-9)

    def test_top_is_the_hulls_surface(self):
        r = rk.from_points(np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1)
                                     for z in (-0.05, 0.05)]))
        s = rk.RockSet([r], [(5.0, 1.0, 0.05)], [(1.0, 0.0, 0.0, 0.0)])
        t, i = s.top(np.array([5.0, 5.09, 5.2]), np.array([1.0, 0.95, 1.0]))
        np.testing.assert_allclose(t[:2], 0.10)
        self.assertEqual(list(i), [0, 0, -1])
        self.assertEqual(t[2], -np.inf)
        # tilted 30 deg about y: against a brute-force search up the vertical line
        q = (math.cos(math.radians(15)), 0.0, math.sin(math.radians(15)), 0.0)
        s.set_pose(0, (5.0, 1.0, 0.05), q)
        t, _ = s.top(np.array([5.0]), np.array([1.0]))
        pts = np.array([[x, 1.0, z] for x in [5.0] for z in np.linspace(-0.3, 0.4, 70001)])
        R = rk._rot(q)
        body = (pts - s.pos[0]) @ R  # world -> body
        inside = np.all(body @ r.planes[:, :3].T <= r.planes[:, 3] + 1e-12, axis=1)
        self.assertAlmostEqual(float(t[0]), float(pts[inside, 2].max()), delta=2e-5)

    def test_height_is_the_road_or_the_rock(self):
        r = rk.make_rock(np.random.default_rng(3), 0.2)
        s = rk.RockSet([r], [(0.0, 0.0, -r.verts[:, 2].min() - 0.02)], [(1, 0, 0, 0)])
        h = s.height(flat)
        x = np.linspace(-0.5, 0.5, 101)
        z = h(x, np.zeros_like(x))
        self.assertEqual(z[0], 0.0)
        self.assertAlmostEqual(z.max(), r.verts[:, 2].max() + s.pos[0, 2], delta=0.01)


class Support(unittest.TestCase):
    def test_a_rock_off_the_ground_holds_nothing_up(self):
        r = rk.make_rock(np.random.default_rng(2), 0.2)
        s = rk.RockSet([r], [(0.0, 0.0, -r.verts[:, 2].min())], [(1, 0, 0, 0)])
        self.assertTrue(s.rest(0, flat))
        self.assertGreater(s.top(np.array([0.0]), np.array([0.0]))[0][0], 0.05)
        s.set_pose(0, (0.0, 0.0, -r.verts[:, 2].min() + 0.2), (1, 0, 0, 0))  # thrown up
        self.assertFalse(s.rest(0, flat))
        self.assertEqual(s.top(np.array([0.0]), np.array([0.0]))[0][0], -np.inf)
        self.assertEqual(s.height(flat)(np.array([0.0]), np.array([0.0]))[0], 0.0)


class Shares(unittest.TestCase):
    def setUp(self):
        self.rock = rk.from_points(np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.15, 0.15)
                                             for z in (0.0, 0.15)]))

    def shares(self, rock_x, zu):
        s = rk.RockSet([self.rock], [(rock_x, 0.0, 0.0)], [(1, 0, 0, 0)])
        return s.shares(flat, np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]),
                        np.array([0.235]), np.array([0.33]), np.array([zu]))[0]

    def test_on_top_of_a_rock_it_takes_all(self):
        # the box's top is 0.075 (centred on its centroid): the wheel on it, 2 cm deflected
        res = self.shares(0.0, 0.33 + 0.075 - 0.02)
        self.assertEqual(len(res), 1)
        self.assertAlmostEqual(res[0][1], 1.0, places=6)

    def test_beside_a_rock_it_takes_nothing(self):
        self.assertEqual(self.shares(1.0, 0.33 - 0.02), [])

    def test_climbing_the_rock_splits_the_load(self):
        wedge = rk.from_points(np.array([[0.0, -0.15, 0.0], [0.0, 0.15, 0.0],
                                         [0.2, -0.15, 0.0], [0.2, 0.15, 0.0],
                                         [0.2, -0.15, 0.1], [0.2, 0.15, 0.1]]))  # rises ahead
        s = rk.RockSet([wedge], [(0.1, 0.0, 0.0)], [(1, 0, 0, 0)])
        z0 = -wedge.verts[:, 2].min()
        s.set_pose(0, (0.1 + 0.0, 0.0, z0), (1, 0, 0, 0))
        res = s.shares(flat, np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]),
                       np.array([0.235]), np.array([0.33]), np.array([0.33 - 0.01]))[0]
        self.assertEqual(len(res), 1)
        i, share, gu, gv, _ = res[0]
        self.assertGreater(share, 0.05)
        self.assertLess(share, 1.0)
        self.assertAlmostEqual(gu, 0.5, places=6)  # the wedge's face: 0.1 over 0.2
        self.assertAlmostEqual(gv, 0.0, places=6)
        forces, lat = rk.reactions([res], [4000.0], [(1.0, 0.0)])
        f = forces[0][0][0]
        self.assertAlmostEqual(f[2], -share * 4000.0)  # pressed down by its share
        self.assertAlmostEqual(f[0], -0.5 * f[2])  # and ahead, normal to its face
        np.testing.assert_allclose(lat, 0.0, atol=1e-9)

    def test_a_stone_at_the_edge_is_pushed_out(self):
        r = rk.make_rock(np.random.default_rng(5), 0.16, rough=0.0)
        s = rk.RockSet([r], [(0.0, 0.14, -r.verts[:, 2].min())], [(1, 0, 0, 0)])
        res = s.shares(flat, np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]),
                       np.array([0.235]), np.array([0.33]),
                       np.array([0.33 + r.height - 0.02]))[0]
        forces, lat = rk.reactions([res], [4000.0], [(1.0, 0.0)])
        f = forces[0][0][0]
        self.assertGreater(f[1], 0.0)  # the stone out to the left
        self.assertAlmostEqual(lat[0][1], -f[1])  # the car to the right, as much

    def test_the_push_is_no_faster_than_the_tire(self):
        f = [(np.array([3000.0, 0.0, -500.0]), np.zeros(3))]
        out, sc = rk.limit(f, 2.0, np.zeros(3), 4.0, 1 / 60)
        self.assertAlmostEqual(out[0][0][0] / 60 / 2.0, 4.0)  # 2 kg to 4 m/s in one step
        self.assertEqual(out[0][0][2], -500.0)
        out, sc = rk.limit(f, 2.0, np.array([5.0, 0.0, 0.0]), 4.0, 1 / 60)
        self.assertEqual(sc, 0.0)  # already faster than the tire: no more push
        out, sc = rk.limit(f, 200.0, np.zeros(3), 4.0, 1 / 60)
        self.assertEqual(sc, 1.0)


class Placement(unittest.TestCase):
    def detail(self, **kw):
        from swb_isaac_world import road_detail as rd

        c = rd.Corridor(np.array([[0.0, 0.0], [200.0, 0.0]]), seed=3,
                        roughness=rd.Roughness(cls="C"))
        return rd.Detail([c], extras="swb_isaac_world.rocks:place", **kw)

    def test_a_tile_places_the_same_rocks_again(self):
        a, b = rk.place(self.detail(), 3, 0), rk.place(self.detail(), 3, 0)
        self.assertGreater(len(a), 0)
        for (ra, pa, _, da), (rb, pb, _, db) in zip(a, b):
            np.testing.assert_array_equal(ra.verts, rb.verts)
            np.testing.assert_array_equal(pa, pb)
            self.assertEqual(da, db)
        self.assertTrue(all(r.mass <= rk.DYNAMIC_MAX_KG for r, _, _, d in a if d))

    def test_the_worker_places_them(self):
        here = rk.place(self.detail(), 3, 0)
        d = self.detail(workers=1, processes=True)
        try:
            d.prefetch({(3, 0)})
            d.tile_of(3, 0)
            got = d.extra(3, 0)
            self.assertEqual(len(got), len(here))
            np.testing.assert_allclose(got[0][1], here[0][1])
        finally:
            d.close()

    def test_static_rocks_merge(self):
        e = rk.place(self.detail(), 3, 0)
        v, f = rk.merged(e)
        self.assertEqual(len(v), sum(len(r.verts) for r, *_ in e))
        self.assertEqual(f.max(), len(v) - 1)


if __name__ == "__main__":
    unittest.main()
