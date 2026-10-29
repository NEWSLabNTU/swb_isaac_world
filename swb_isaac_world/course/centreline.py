"""C2: the section's centreline in the map frame, its height profile and its hairpins.

The road's ways are joined on their shared nodes and the shortest path between the nodes
nearest the section's two ends is taken. It is resampled at 1 m and smoothed (Gaussian, 2 m),
which rounds OSM's polygon corners without moving a hairpin's apex by more than a few
decimetres, then resampled at `STEP`. Heights come from the DEM under each point, filtered
along the road (a 25 m median, then a 10 m Gaussian): the DEM's ground beside the road has
banks and walls, the road itself has no steps.

A closed course (`road.closed: true`, a circuit) has one point, `from`: the loop is the shortest
way from the node nearest it around and back (along the ways' direction where they are one
way). Its filters wrap around the loop, it is resampled evenly (a step within a hair of `STEP`)
so the last point is a step before the first, the planner's convention for a closed track, and
its arrays carry `closed`.
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


def _path(ways, frame, a, b=None):
    """The road's points from the node nearest a to the node nearest b (lat, lon pairs),
    shortest along the ways; b None: the shortest loop from a's node back to it (its first
    point repeated last)."""
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
    src = near(a)
    start, dst = src, (src if b is None else near(b))
    if b is None:  # a loop: from a stand-in for src (OSM's ids are positive) to src itself
        start = -1
        adj[start] = adj.get(src, [])
    dist, prev, todo = {start: 0.0}, {}, [(0.0, start)]
    while todo:
        d, u = heapq.heappop(todo)
        if u == dst:
            break
        if d > dist[u]:
            continue
        for v, w in adj.get(u, []):
            if v == prev.get(u):  # not back the way it came (a loop's two-way way)
                continue
            if d + w < dist.get(v, math.inf):
                dist[v], prev[v] = d + w, u
                heapq.heappush(todo, (d + w, v))
    if dst not in dist:
        raise RuntimeError("the section's ends are not connected along the road")
    nodes = [dst]
    while nodes[-1] != start:
        nodes.append(prev[nodes[-1]])
    return np.array([pos[src if n == start else n] for n in reversed(nodes)])


def _resample(p, step, closed=False):
    """Points every `step` along p; closed: p's last point is its first again, the loop is
    resampled evenly at the step nearest `step` and the first point is not repeated."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    t = np.arange(0.0, s[-1], step) if not closed else np.arange(round(s[-1] / step)) * (s[-1] / round(s[-1] / step))
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def _gauss(v, sigma_pts, closed=False):
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(v, sigma_pts, axis=0, mode="wrap" if closed else "nearest")


def _close(p):
    """A loop's points with its first repeated last."""
    return np.concatenate([p, p[:1]])


def build(course, ways, dem, frame):
    """The centreline: a dict of arrays s, x, y, z (map frame), heading, curvature, grade, and
    the hairpins (and `closed`, True, for a closed course)."""
    from scipy.ndimage import median_filter

    road = course["road"]
    closed = bool(road.get("closed", False))
    p = _path(ways, frame, road["from"], None if closed else road["to"])
    p = _resample(p, 1.0, closed)
    p = _gauss(p, SMOOTH_XY / 1.0, closed)
    p = _resample(_close(p) if closed else p, STEP, closed)
    step = STEP
    if closed:
        step = float(np.sum(np.hypot(*np.diff(_close(p), axis=0).T)) / len(p))  # a hair off STEP
    s = np.arange(len(p)) * step
    mode = "wrap" if closed else "nearest"
    raw_z = np.array([dem.at(*frame.latlon(x, y)) for x, y in p]) - frame.alt0
    if np.isnan(raw_z).any():
        good = ~np.isnan(raw_z)
        raw_z = np.interp(s, s[good], raw_z[good], period=s[-1] + step if closed else None)
    z = median_filter(raw_z, size=int(MEDIAN_Z / step), mode=mode)
    z = _gauss(z, SMOOTH_Z / step, closed)
    if closed:  # central differences around the loop
        d = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * step)
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        dh = np.angle(np.exp(1j * (np.roll(heading, -1) - np.roll(heading, 1))))
        kappa = dh / (2 * step)
        grade = (np.roll(z, -1) - np.roll(z, 1)) / (2 * step)
        turns = float(np.sum(np.angle(np.exp(1j * np.diff(np.append(heading, heading[0]))))))
    else:
        d = np.gradient(p, step, axis=0)
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        kappa = np.gradient(heading, step)
        grade = np.gradient(z, step)
    # hairpins: the heading turns more than HAIRPIN_TURN within HAIRPIN_SPAN
    half = int(HAIRPIN_SPAN / 2 / step)
    n = len(s)
    turn = np.zeros(n)
    if closed:  # the heading continued around the loop, a lap's turn added each time
        at = lambda i: heading[np.mod(i, n)] + turns * np.floor_divide(i, n)  # noqa: E731
        turn = np.degrees(at(np.arange(n) + half) - at(np.arange(n) - half))
    else:
        at = lambda i: heading[min(max(i, 0), n - 1)]  # noqa: E731
        turn[half:-half] = np.degrees(heading[2 * half:] - heading[:-2 * half])
    # a closed course is scanned from a point outside any hairpin, so none is cut at the seam
    i0 = 0 if not closed or (np.abs(turn) >= HAIRPIN_TURN).all() else int(np.argmax(np.abs(turn) < HAIRPIN_TURN))
    order = (i0 + np.arange(n)) % n
    hairpins = []
    i = 0
    while i < n:
        if abs(turn[order[i]]) >= HAIRPIN_TURN:
            j = i
            while j < n and abs(turn[order[j]]) >= HAIRPIN_TURN:
                j += 1
            k = int(order[i + int(np.argmax(np.abs(kappa[order[i:j]])))])  # the apex: the tightest point
            a, b = k - half * 2, k + half * 2
            if not closed:
                a, b = max(0, a), min(n - 1, b)
            hairpins.append({
                "s": round(float(s[k]), 1), "x": round(float(p[k, 0]), 2), "y": round(float(p[k, 1]), 2),
                "z": round(float(z[k]), 2), "radius": round(float(1.0 / abs(kappa[k])), 1),
                "turn_deg": round(float(np.degrees(at(b) - at(a))), 1),
                "direction": "left" if kappa[k] > 0 else "right"})
            i = j
        else:
            i += 1
    if closed:
        hairpins.sort(key=lambda h: h["s"])
    out = {"s": s, "x": p[:, 0], "y": p[:, 1], "z": z, "z_dem": raw_z, "heading": heading,
           "curvature": kappa, "grade": grade, "hairpins": hairpins}
    if closed:
        out["closed"] = np.array(True)
    return out
