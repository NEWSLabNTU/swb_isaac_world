"""A closed course (a circuit, `road.closed: true`): the loop from OSM-like ways, its centreline,
ribbons and lanelets closing on themselves (roadmap 058, 060 M7); synthetic, no network."""
import math
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swb_isaac_world.course import centreline as cl  # noqa: E402
from swb_isaac_world.course import geometry, maps  # noqa: E402

ORIGIN = {"lat": 36.15, "lon": 139.92, "alt": 20.0}
R, STRAIGHT = 15.0, 100.0   # a stadium: two straights joined by two 180 deg bends


class FlatDem:
    def at(self, lat, lon):
        return 21.0


def stadium(frame, clockwise=True):
    """A stadium's points (x, y), counterclockwise from (0, -R) along the lower straight, about
    every 2 m, as one closed way's (lat, lon)."""
    pts = []
    for x in np.arange(0.0, STRAIGHT, 2.0):
        pts.append((x, -R))
    for a in np.arange(-90.0, 90.0, 10.0):
        pts.append((STRAIGHT + R * math.cos(math.radians(a)), R * math.sin(math.radians(a))))
    for x in np.arange(STRAIGHT, 0.0, -2.0):
        pts.append((x, R))
    for a in np.arange(90.0, 270.0, 10.0):
        pts.append((R * math.cos(math.radians(a)), R * math.sin(math.radians(a))))
    if clockwise:
        pts = pts[::-1]
    return [frame.latlon(x, y) for x, y in pts]


def way(latlon, oneway=True, first_id=1):
    ids = list(range(first_id, first_id + len(latlon)))
    return {"type": "way", "nodes": ids + ids[:1],
            "geometry": [{"lat": a, "lon": o} for a, o in latlon + latlon[:1]],
            "tags": {"highway": "raceway", "oneway": "yes"} if oneway else {"highway": "raceway"}}


def course(frame, start_xy):
    return {"name": "stadium", "origin": ORIGIN, "margin": 50.0,
            "road": {"closed": True, "from": list(frame.latlon(*start_xy)), "lanes": 2,
                     "lane_width": 6.0, "shoulder": 1.5}}


class ClosedCentreline(unittest.TestCase):
    def setUp(self):
        self.frame = cl.Frame(ORIGIN)
        self.ways = [way(stadium(self.frame))]
        self.c = cl.build(course(self.frame, (50.0, -R)), self.ways, FlatDem(), self.frame)

    def test_the_loop_closes_a_step_short(self):
        c = self.c
        self.assertTrue(bool(c["closed"]))
        step = c["s"][1] - c["s"][0]
        self.assertAlmostEqual(step, cl.STEP, delta=0.01)
        np.testing.assert_allclose(np.diff(c["s"]), step)
        # the last point a step before the first, not the first again
        gap = math.hypot(c["x"][0] - c["x"][-1], c["y"][0] - c["y"][-1])
        self.assertAlmostEqual(gap, step, delta=0.02)
        lap = c["s"][-1] + step
        self.assertAlmostEqual(lap, 2 * STRAIGHT + 2 * math.pi * R, delta=3.0)
        # the start on the straight nearest `from`
        self.assertAlmostEqual(c["x"][0], 50.0, delta=1.5)
        self.assertAlmostEqual(c["y"][0], -R, delta=0.2)

    def test_curvature_and_grade_wrap_at_the_seam(self):
        c = self.c
        # clockwise: one lap's turn is -2 pi, every bend a right one
        step = c["s"][1] - c["s"][0]
        self.assertAlmostEqual(float(np.sum(c["curvature"]) * step), -2 * math.pi, delta=0.01)
        # at the seam (a straight) as straight as mid straight, no end effects
        self.assertLess(abs(c["curvature"][0]), 1e-3)
        self.assertLess(abs(c["curvature"][-1]), 1e-3)
        self.assertAlmostEqual(float(-c["curvature"].min()), 1 / R, delta=0.2 / R)
        np.testing.assert_allclose(c["z"], 1.0, atol=1e-9)
        np.testing.assert_allclose(c["grade"], 0.0, atol=1e-9)

    def test_the_bends_are_hairpins_once_each_even_across_the_seam(self):
        self.assertEqual([h["direction"] for h in self.c["hairpins"]], ["right", "right"])
        # the loop started on a bend's apex: the bend is still one hairpin, not two halves
        c = cl.build(course(self.frame, (-R, 0.0)), self.ways, FlatDem(), self.frame)
        self.assertEqual(len(c["hairpins"]), 2)
        for h in c["hairpins"]:
            self.assertAlmostEqual(abs(h["turn_deg"]), 180.0, delta=25.0)
            self.assertAlmostEqual(h["radius"], R, delta=2.0)

    def test_the_loop_goes_the_ways_direction(self):
        ccw = [way(stadium(self.frame, clockwise=False))]
        c = cl.build(course(self.frame, (50.0, -R)), ccw, FlatDem(), self.frame)
        self.assertAlmostEqual(float(np.sum(c["curvature"]) * (c["s"][1] - c["s"][0])), 2 * math.pi,
                               delta=0.01)
        self.assertEqual([h["direction"] for h in c["hairpins"]], ["left", "left"])

    def test_a_shorter_loop_through_the_start_wins(self):
        # a two-way chord across the stadium's middle, from the start's straight to the other:
        # the shortest loop through the start uses it
        a = stadium(self.frame)
        pos = [self.frame.xy(*p) for p in a]
        lo = min(range(len(pos)), key=lambda i: math.dist(pos[i], (50.0, -R)))
        hi = min(range(len(pos)), key=lambda i: math.dist(pos[i], (50.0, R)))
        chord = {"type": "way", "nodes": [lo + 1, 10 ** 6, hi + 1],
                 "geometry": [{"lat": a[lo][0], "lon": a[lo][1]},
                              dict(zip(("lat", "lon"), self.frame.latlon(50.0, 0.0))),
                              {"lat": a[hi][0], "lon": a[hi][1]}],
                 "tags": {"highway": "raceway"}}
        c = cl.build(course(self.frame, (50.0, -R)), self.ways + [chord], FlatDem(), self.frame)
        self.assertLess(c["s"][-1], 0.7 * (2 * STRAIGHT + 2 * math.pi * R))


class ClosedRibbonsAndLanelets(unittest.TestCase):
    def setUp(self):
        self.frame = cl.Frame(ORIGIN)
        self.course = course(self.frame, (50.0, -R))
        self.c = cl.build(self.course, [way(stadium(self.frame))], FlatDem(), self.frame)

    def test_ribbons_close(self):
        c, n = self.c, len(self.c["x"])
        pts, faces = geometry.ribbon(c, 7.5, -7.5, strip=geometry.ROAD_STRIP)
        self.assertEqual(faces.shape, (30 * n, 4))
        # the last quad of a strip joins the last point to the first
        self.assertEqual(list(faces[n - 1]), [n - 1, 0, n, 2 * n - 1])
        # the closing quads as long as their neighbours (a step, more on a bend's outside)
        a, b = pts[faces[:, 0]], pts[faces[:, 1]]
        ln = np.linalg.norm(b - a, axis=1).reshape(30, n)
        np.testing.assert_allclose(ln[:, -1], ln[:, -2], rtol=0.05)
        # the open course's ribbon is unchanged: n - 1 quads per strip
        op = {k: c[k] for k in ("x", "y", "z")}
        self.assertEqual(geometry.ribbon(op, 7.5, -7.5)[1].shape, (n - 1, 4))

    def test_lanelets_close(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "lanelet2_map.osm")
            along, back = maps.lanelets(self.c, self.course, self.frame, out, 200.0)
            root = ET.parse(out).getroot()
        ways = {w.get("id"): [int(nd.get("ref")) for nd in w.findall("nd")] for w in root.findall("way")}
        rels = {r.get("id"): {m.get("role"): m.get("ref") for m in r.findall("member")}
                for r in root.findall("relation")}
        lap = self.c["s"][-1] + self.c["s"][1]
        self.assertEqual(len(along), math.ceil(lap / maps.CHUNK))
        for ids in (along, back):
            for role in ("left", "right"):
                bounds = [ways[rels[str(i)][role]] for i in ids]
                # each stretch starts where the one before ends, around the loop
                for p, q in zip(bounds, bounds[1:] + bounds[:1]):
                    if ids is along:
                        self.assertEqual(p[-1], q[0])
                    else:
                        self.assertEqual(p[0], q[-1])


if __name__ == "__main__":
    unittest.main()
