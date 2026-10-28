"""C2: the section's centreline in the map frame, its height profile and its hairpins.

The road's ways are joined on their shared nodes and the shortest path between the nodes
nearest the section's two ends is taken. It is resampled at 1 m and smoothed (Gaussian, 2 m),
which rounds OSM's polygon corners without moving a hairpin's apex by more than a few
decimetres, then resampled at `STEP`. Heights come from the DEM under each point, filtered
along the road (a 25 m median, then a 10 m Gaussian): the DEM's ground beside the road has
banks and walls, the road itself has no steps.
"""
import heapq
import math

import numpy as np

STEP = 0.5          # m between centreline points
SMOOTH_XY = 2.0     # m, sigma along the road
MEDIAN_Z = 25.0     # m, window
SMOOTH_Z = 10.0     # m, sigma
HAIRPIN_TURN = 120.0  # deg within HAIRPIN_SPAN
HAIRPIN_SPAN = 60.0   # m


class Frame:
    """The map frame: transverse Mercator at the course's origin, z above its altitude."""

    def __init__(self, origin):
        import pyproj

        self.lat0, self.lon0, self.alt0 = origin["lat"], origin["lon"], origin["alt"]
        self.proj = pyproj.Proj(proj="tmerc", lat_0=self.lat0, lon_0=self.lon0, k=1.0, x_0=0.0,
                                y_0=0.0, ellps="WGS84")

    def xy(self, lat, lon):
        return self.proj(lon, lat)

    def latlon(self, x, y):
        lon, lat = self.proj(x, y, inverse=True)
        return lat, lon


def _path(ways, frame, a, b):
    """The road's points from the node nearest a to the node nearest b (lat, lon pairs),
    shortest along the ways."""
    pos, adj = {}, {}
    for w in ways:
        for n, g in zip(w["nodes"], w["geometry"]):
            pos[n] = frame.xy(g["lat"], g["lon"])
        for u, v in zip(w["nodes"], w["nodes"][1:]):
            d = math.dist(pos[u], pos[v])
            adj.setdefault(u, []).append((v, d))
            if w.get("tags", {}).get("oneway") != "yes":
                adj.setdefault(v, []).append((u, d))
    near = lambda p: min(pos, key=lambda n: math.dist(pos[n], frame.xy(*p)))
    src, dst = near(a), near(b)
    dist, prev, todo = {src: 0.0}, {}, [(0.0, src)]
    while todo:
        d, u = heapq.heappop(todo)
        if u == dst:
            break
        if d > dist[u]:
            continue
        for v, w in adj.get(u, []):
            if d + w < dist.get(v, math.inf):
                dist[v], prev[v] = d + w, u
                heapq.heappush(todo, (d + w, v))
    if dst not in dist:
        raise RuntimeError("the section's ends are not connected along the road")
    nodes = [dst]
    while nodes[-1] != src:
        nodes.append(prev[nodes[-1]])
    return np.array([pos[n] for n in reversed(nodes)])


def _resample(p, step):
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def _gauss(v, sigma_pts):
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(v, sigma_pts, axis=0, mode="nearest")


def build(course, ways, dem, frame):
    """The centreline: a dict of arrays s, x, y, z (map frame), heading, curvature, grade, and
    the hairpins."""
    from scipy.ndimage import median_filter

    road = course["road"]
    p = _path(ways, frame, road["from"], road["to"])
    p = _resample(p, 1.0)
    p = _gauss(p, SMOOTH_XY / 1.0)
    p = _resample(p, STEP)
    s = np.arange(len(p)) * STEP
    raw_z = np.array([dem.at(*frame.latlon(x, y)) for x, y in p]) - frame.alt0
    if np.isnan(raw_z).any():
        good = ~np.isnan(raw_z)
        raw_z = np.interp(s, s[good], raw_z[good])
    z = median_filter(raw_z, size=int(MEDIAN_Z / STEP), mode="nearest")
    z = _gauss(z, SMOOTH_Z / STEP)
    d = np.gradient(p, STEP, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    kappa = np.gradient(heading, STEP)
    grade = np.gradient(z, STEP)
    # hairpins: the heading turns more than HAIRPIN_TURN within HAIRPIN_SPAN
    half = int(HAIRPIN_SPAN / 2 / STEP)
    turn = np.zeros(len(s))
    turn[half:-half] = np.degrees(heading[2 * half:] - heading[:-2 * half])
    hairpins = []
    i = 0
    while i < len(s):
        if abs(turn[i]) >= HAIRPIN_TURN:
            j = i
            while j < len(s) and abs(turn[j]) >= HAIRPIN_TURN:
                j += 1
            k = i + int(np.argmax(np.abs(kappa[i:j])))  # the apex: the tightest point
            a, b = max(0, k - half * 2), min(len(s) - 1, k + half * 2)
            hairpins.append({
                "s": round(float(s[k]), 1), "x": round(float(p[k, 0]), 2), "y": round(float(p[k, 1]), 2),
                "z": round(float(z[k]), 2), "radius": round(float(1.0 / abs(kappa[k])), 1),
                "turn_deg": round(float(np.degrees(heading[b] - heading[a])), 1),
                "direction": "left" if kappa[k] > 0 else "right"})
            i = j
        else:
            i += 1
    return {"s": s, "x": p[:, 0], "y": p[:, 1], "z": z, "z_dem": raw_z, "heading": heading,
            "curvature": kappa, "grade": grade, "hairpins": hairpins}
