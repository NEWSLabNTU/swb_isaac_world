"""The road's small-scale detail (roadmap 066 R3); no Isaac needed."""
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swb_isaac_world import road_detail as rd  # noqa: E402

LINE = np.array([[0.0, 0.0], [400.0, 0.0]])


class Generation(unittest.TestCase):
    def test_a_tile_made_again_is_the_same(self):
        a = rd.Detail([rd.Corridor(LINE, seed=7, potholes=rd.Potholes())])
        b = rd.Detail([rd.Corridor(LINE, seed=7, potholes=rd.Potholes())])
        b.tile_of(3, -1)  # another order of making
        np.testing.assert_array_equal(a.tile_of(2, 0), b.tile_of(2, 0))
        c = rd.Detail([rd.Corridor(LINE, seed=8)])
        self.assertGreater(np.abs(a.tile_of(2, 0) - c.tile_of(2, 0)).max(), 1e-3)

    def test_workers_make_the_same_tiles(self):
        def corridor():
            return rd.Corridor(LINE, seed=7, potholes=rd.Potholes(), ruts=rd.Ruts(),
                               water_bars=rd.WaterBars(at=(40.0,)))

        here = rd.Detail([corridor()])
        for kw in ({"workers": 1}, {"workers": 1, "processes": True}):
            d = rd.Detail([corridor()], **kw)
            try:
                d.prefetch({(2, 0), (2, -1)})
                for key in ((2, 0), (2, -1)):
                    np.testing.assert_array_equal(d.tile_of(*key), here.tile_of(*key))
                for _ in range(100):  # the done callbacks run just after the result
                    if len(d.made_ms) == 2 and not d.pending:
                        break
                    time.sleep(0.01)
                self.assertEqual(len(d.made_ms), 2)
            finally:
                d.close()

    def test_no_seam_between_tiles(self):
        d = rd.Detail([rd.Corridor(LINE, seed=1)])
        y = np.linspace(-2.0, 2.0, 81)
        edge = d.height(np.full_like(y, 32.0 - 1e-9), y) - d.height(np.full_like(y, 32.0), y)
        self.assertLess(np.abs(edge).max(), 1e-6)  # the shared row of samples
        # the slope across the seam is like the slope inside the tile
        xs = np.arange(28.0, 36.0, rd.SPACING)
        h = d.height(xs, np.full_like(xs, 0.8))
        jumps = np.abs(np.diff(h))
        at = np.argmin(np.abs(xs[:-1] - 32.0))
        self.assertLess(jumps[at - 1:at + 2].max(), 4 * np.median(jumps) + 1e-3)

    def test_spectrum_is_the_class(self):
        d = rd.Detail([rd.Corridor(np.array([[0.0, 0.0], [700.0, 0.0]]), seed=3,
                                   section=rd.Section(crown=0.0, ditch_depth=0.0, shoulder=3.0))])
        x = np.arange(0.0, 640.0, rd.SPACING)
        h = d.height(x, np.full_like(x, 0.8))
        n, g = rd.line_psd(h, rd.SPACING)
        rough = rd.Roughness()
        for lo, hi in ((0.1, 0.2), (0.4, 0.8), (1.0, 2.0), (2.5, 4.0)):
            s = (n >= lo) & (n < hi)
            self.assertAlmostEqual(np.mean(g[s]) / np.mean(rough.psd(n[s])), 1.0, delta=0.25)

    def test_iri_follows_the_class(self):
        out = {}
        for cls in "CD":
            d = rd.Detail([rd.Corridor(np.array([[0.0, 0.0], [700.0, 0.0]]), seed=5,
                                       roughness=rd.Roughness(cls=cls),
                                       section=rd.Section(crown=0.0, ditch_depth=0.0,
                                                          shoulder=3.0))])
            x = np.arange(0.0, 640.0, rd.SPACING)
            out[cls] = rd.iri(d.height(x, np.full_like(x, 0.8)), rd.SPACING)
        self.assertAlmostEqual(out["D"] / out["C"], 2.0, delta=0.3)  # 4 x the PSD, 2 x the IRI
        self.assertTrue(5.0 < out["C"] < 14.0, out)  # class C: about 9 m/km (analytic 8.7)

    def test_section(self):
        s = rd.Section()
        self.assertAlmostEqual(float(s.height(0.0)), 0.0)
        self.assertAlmostEqual(float(s.height(2.5)), -0.10)  # 4 % over 2.5 m
        bottom = float(s.height(s.half_width + s.shoulder + s.ditch_depth / s.foreslope + 0.3))
        self.assertAlmostEqual(bottom, -0.04 * 3.0 - 0.30)
        self.assertEqual(float(s.height(s.extent() + 0.1)), 0.0)
        self.assertEqual(float(rd.Section(crown=0.0).height(1.5)), 0.0)  # a level section

    def test_features_where_they_are_put(self):
        c = rd.Corridor(LINE, roughness=rd.Roughness(gd0=1e-12),
                        washboard=(rd.Washboard(100.0, 150.0, height=0.02),),
                        water_bars=rd.WaterBars(at=(300.0,)), ruts=rd.Ruts())
        d = rd.Detail([c])
        x = np.arange(110.0, 140.0, 0.01)
        h = d.height(x, np.zeros_like(x)) - c.ruts.height(0.0)
        self.assertAlmostEqual(h.max() - h.min(), 0.02, delta=0.002)
        self.assertAlmostEqual(float(d.height(300.0, 0.0) - c.ruts.height(0.0)), 0.06,
                               delta=0.005)
        self.assertLess(float(d.height(50.0, 0.8)), -0.02)  # the rut

    def test_render_mesh_agrees_with_the_height(self):
        """Question 3: a renderer splits each quad of samples into two triangles; the bilinear
        height the tire reads lies within 5 mm of them."""
        d = rd.Detail([rd.Corridor(LINE, seed=2, potholes=rd.Potholes(per_100m=30.0))])
        t = d.tile_of(1, 0)
        rng = np.random.default_rng(0)
        n = d.n
        i, j = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
        a, b = rng.random(20000), rng.random(20000)
        x = 16.0 + (i + a) * rd.SPACING
        y = (j + b) * rd.SPACING
        bil = d.height(x, y)
        h00, h10, h01, h11 = t[j, i], t[j, i + 1], t[j + 1, i], t[j + 1, i + 1]
        lower = a >= b  # the triangle (00, 10, 11) or (00, 11, 01)
        tri = np.where(lower, h00 + a * (h10 - h00) + b * (h11 - h10),
                       h00 + b * (h01 - h00) + a * (h11 - h01))
        self.assertLess(np.abs(bil - tri).max(), 0.005)


class Sections(unittest.TestCase):
    """A road made of sections, one corridor each, overlapping by their taper (the tour)."""

    @staticmethod
    def road(t, cls=("B", "D")):
        a = rd.Corridor(np.array([[0.0, 0.0], [100.0 + t / 2, 0.0]]), seed=1, taper=t,
                        roughness=rd.Roughness(cls=cls[0]))
        b = rd.Corridor(np.array([[100.0 - t / 2, 0.0], [200.0, 0.0]]), seed=2, taper=t,
                        roughness=rd.Roughness(cls=cls[1]))
        return rd.Detail([a, b])

    def test_the_section_stays_itself_through_a_join(self):
        flat = rd.Roughness(gd0=1e-30)
        a = rd.Corridor(np.array([[0.0, 0.0], [102.5, 0.0]]), taper=5.0, roughness=flat)
        b = rd.Corridor(np.array([[97.5, 0.0], [200.0, 0.0]]), taper=5.0, roughness=flat)
        d = rd.Detail([a, b])
        x = np.linspace(80.0, 120.0, 81)
        for y in (0.0, 1.5, 2.5, 3.5):
            got = d.height(x, np.full_like(x, y))
            np.testing.assert_allclose(got, a.section.height(np.full_like(x, y)), atol=1e-6)

    def test_no_step_where_sections_meet(self):
        d = self.road(5.0)
        x = np.arange(60.0, 140.0, rd.SPACING)
        h = d.height(x, np.zeros_like(x))
        dh = np.abs(np.diff(h))
        join = (x[1:] > 97.5) & (x[1:] < 102.5)
        # the join's steps are no bigger than the rough section's own
        self.assertLessEqual(dh[join].max(), dh[x[1:] > 102.5].max() * 1.5)

    def test_the_sections_keep_their_class(self):
        d = self.road(5.0)
        x = np.arange(20.0, 180.0, rd.SPACING)
        h = d.height(x, np.zeros_like(x))
        smooth, rough = h[x < 90.0].std(), h[x > 110.0].std()
        self.assertGreater(rough, 2.0 * smooth)


class Zones(unittest.TestCase):
    def test_road_verge_and_outside(self):
        c = rd.Corridor(LINE, seed=1)
        d = rd.Detail([c])
        z = d.zones_tile(2, -1)  # y from -16 to 0: the corridor's right half and beyond
        y = -16.0 + np.arange(d.n + 1) * d.spacing
        col = z[:, d.n // 2]
        road = c.section.half_width + c.section.shoulder
        np.testing.assert_array_equal(col[np.abs(y) <= road - 0.01], rd.Detail.ROAD)
        between = (np.abs(y) > road + 0.01) & (np.abs(y) < c.reach - 0.01)
        np.testing.assert_array_equal(col[between], rd.Detail.VERGE)
        np.testing.assert_array_equal(col[np.abs(y) > c.reach + 0.01], rd.Detail.OUTSIDE)


class Measures(unittest.TestCase):
    def test_wk_weighting(self):
        """ISO 2631-1's Wk: about 1 at 4-8 Hz, about 0.5 at 1 Hz and 16 Hz."""
        rate = 500.0
        t = np.arange(0, 20.0, 1 / rate)
        for f, gain in ((1.0, 0.48), (6.3, 1.05), (16.0, 0.77)):
            a = np.sin(2 * np.pi * f * t)
            w = rd.iso2631_wk(a, rate)
            self.assertAlmostEqual(w[1000:-1000].std() / a.std(), gain, delta=0.08)
