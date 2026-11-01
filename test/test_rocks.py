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
                        roughness=rd.Roughness(cls="C"), rocks=rk.Zones())
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

    def test_no_loose_rock_below_the_base(self):
        from swb_isaac_world import road_detail as rd

        c = rd.Corridor(np.array([[0.0, 0.0], [200.0, 0.0]]), seed=3,
                        roughness=rd.Roughness(cls="C"),
                        rocks=rk.Zones(loose_below=rk.LOOSE_BELOW))
        d = rd.Detail([c])
        for i in range(0, 12):
            for j in (-1, 0):
                for r, p, _, dyn in rk.place(d, i, j):
                    if dyn:
                        h = float(d.height(np.array([p[0]]), np.array([p[1]]))[0])
                        self.assertGreaterEqual(h, -rk.LOOSE_BELOW)
        none = rd.Detail([rd.Corridor(np.array([[0.0, 0.0], [200.0, 0.0]]), seed=3)])
        self.assertEqual(rk.place(none, 3, 0), [])  # a corridor without rocks has none

    def test_the_carriageway_is_graded(self):
        from swb_isaac_world import road_detail as rd

        c = rd.Corridor(np.array([[0.0, 0.0], [400.0, 0.0]]), seed=4,
                        roughness=rd.Roughness(cls="C"),
                        rocks=rk.Zones(carriageway=0.03, shoulder=0.03, ditch=0.0))  # the same k
        d = rd.Detail([c])
        n = 0
        for i in range(0, 20):
            for j in (-1, 0):
                for r, p, _, dyn in rk.place(d, i, j):
                    _, v, _ = c.uv(np.array([p[0]]), np.array([p[1]]))
                    if c.rocks.zone(c.section, v[0]) != "carriageway":
                        continue  # the shoulder's are not graded
                    top = p[2] + r.verts[:, 2].max()
                    h = float(d.height(np.array([p[0]]), np.array([p[1]]))[0])
                    if dyn:
                        self.assertLessEqual(max(r.verts[:, 0].ptp(), r.verts[:, 1].ptp()),
                                             2 * 0.30 / 1.7 * 1.25)
                    else:
                        self.assertLessEqual(top - h, 0.05 + 1e-9)
                    n += 1
        self.assertGreater(n, 10)

    def test_static_rocks_merge(self):
        e = rk.place(self.detail(), 3, 0)
        v, f = rk.merged(e)
        self.assertEqual(len(v), sum(len(r.verts) for r, *_ in e))
        self.assertEqual(f.max(), len(v) - 1)


if __name__ == "__main__":
    unittest.main()


class Prototypes(unittest.TestCase):
    def test_a_scaled_prototype_is_one_rock_for_the_tire_and_the_mesh(self):
        r = rk.scaled(5, 0.4)
        self.assertEqual(r.proto, 5)
        self.assertLess((r.verts @ r.planes[:, :3].T - r.planes[:, 3]).max(), 1e-9)
        self.assertAlmostEqual(r.mass, rk.prototypes()[5].mass * 0.4 ** 3)
        yaw, p = 0.7, np.array([3.0, 1.0, 0.1])
        s = rk.RockSet([r], [p], [rk._yaw_quat(yaw)])
        v, f = rk.merged([(r, p, yaw, False)])
        # the merged mesh's highest point lies on the tire's top surface there
        k = int(np.argmax(v[:, 2]))
        t, i = s.top(np.array([v[k, 0]]), np.array([v[k, 1]]))
        self.assertEqual(int(i[0]), 0)
        self.assertAlmostEqual(float(t[0]), float(v[k, 2]), places=6)


class Index(unittest.TestCase):
    def test_near_is_the_full_scan(self):
        rng = np.random.default_rng(3)
        rocks = [rk.scaled(int(rng.integers(rk.PROTOTYPES)), float(rng.uniform(0.1, 1.5)))
                 for _ in range(400)]
        pos = np.c_[rng.uniform(-50, 50, 400), rng.uniform(-10, 10, 400), np.zeros(400)]
        s = rk.RockSet(rocks, pos, [(1, 0, 0, 0)] * 400)
        for _ in range(30):
            x, y = rng.uniform(-50, 50, 6), rng.uniform(-10, 10, 6)
            m = float(rng.uniform(0, 1))
            x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
            dx = np.maximum(np.maximum(x0 - pos[:, 0], pos[:, 0] - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - pos[:, 1], pos[:, 1] - y1), 0.0)
            want = np.nonzero(np.hypot(dx, dy) <= s.r + m)[0]
            np.testing.assert_array_equal(s.near(x, y, m), want)
        was = pos[0].copy()
        s.set_pose(0, (40.0, 8.0, 0.0), (1, 0, 0, 0))  # moved across cells
        self.assertIn(0, s.near(np.array([40.0]), np.array([8.0])))
        self.assertNotIn(0, s.near(np.array([was[0]]), np.array([was[1]])))
