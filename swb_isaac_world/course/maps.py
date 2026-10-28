"""C7: Autoware's maps of a course, beside the Isaac world: `lanelet2_map.osm`,
`pointcloud_map.pcd`, `map_projector_info.yaml` and `map_config.yaml`.

The lanelets: one per lane per stretch of `CHUNK` m, both directions. Japan drives on the left, so
the lane along the section (s increasing) is the one left of the centreline and the other runs
back. Its bounds: the lanes' outer edges (white, solid) and the centre line (yellow, solid: no
passing on a mountain road), shared by both lanes, inverted for the one running back. Every
point carries `local_x`/`local_y` (the map frame) and `ele`, and a latitude and longitude from
the course's transverse Mercator, which `map_projector_info.yaml` names.

The point cloud: the triangles the lidar sees, sampled. The terrain grid's quads are split
from their first corner to their third, as the renderer splits them, so a point lies on the
surface the lidar's ray hits and not on a bilinear patch near it. Within `NEAR` m of the road
the sampling is `STEP_NEAR`, beyond it `STEP_FAR`; the road at `STEP_NEAR`. Written as a binary
PCD of x, y, z (float32), as Autoware's map loader reads, thinned to one point per `VOXEL`.
"""
import os

import numpy as np

CHUNK = 100.0       # m of road per lanelet
STEP_NEAR = 0.25    # m
STEP_FAR = 1.0      # m
NEAR = 120.0        # m from the centreline
VOXEL = 0.2         # m: the cloud keeps one point per voxel this size


def _sample_points(c, step):
    s = c["s"]
    t = np.arange(0.0, s[-1], step)
    return np.searchsorted(s, t).clip(0, len(s) - 1)


def lanelets(c, course, frame, out, speed_limit_kmh):
    """Write the Lanelet2 map; returns the lanelets' ids along the section and back."""
    from .geometry import _normals

    r = course["road"]
    edge = r["lanes"] * r["lane_width"] / 2
    x, y, z = c["x"], c["y"], c["z"]
    nrm = _normals(x, y)
    idx = _sample_points(c, 2.0)  # a bound point every 2 m
    idx = np.unique(np.concatenate([idx, [len(x) - 1]]))
    node_id = [0]
    nodes, ways, rels = [], [], []

    def node(px, py, pz):
        node_id[0] += 1
        lat, lon = frame.latlon(px, py)
        nodes.append((node_id[0], lat, lon, px, py, pz))
        return node_id[0]

    lines = {}
    for name, off in (("left", edge), ("centre", 0.0), ("right", -edge)):
        lines[name] = [node(x[i] + off * nrm[i, 0], y[i] + off * nrm[i, 1], z[i]) for i in idx]
    # stretches of CHUNK m, sharing their end points
    s = c["s"][idx]
    cuts = [0]
    for k in range(1, len(idx)):
        if s[k] - s[cuts[-1]] >= CHUNK and len(idx) - k > 10:
            cuts.append(k)
    cuts.append(len(idx) - 1)
    wid = [10 ** 6]

    def way(refs, tags):
        wid[0] += 1
        ways.append((wid[0], refs, tags))
        return wid[0]

    white = {"type": "line_thin", "subtype": "solid", "color": "white"}
    yellow = {"type": "line_thin", "subtype": "solid", "color": "yellow"}
    along, back = [], []
    rid = [2 * 10 ** 6]
    for a, b in zip(cuts, cuts[1:]):
        left = way(lines["left"][a:b + 1], white)
        centre = way(lines["centre"][a:b + 1], yellow)
        right = way(lines["right"][a:b + 1], white)
        right_back = way(list(reversed(lines["right"][a:b + 1])), white)
        centre_back = way(list(reversed(lines["centre"][a:b + 1])), yellow)
        tags = {"type": "lanelet", "subtype": "road", "location": "urban", "one_way": "yes",
                "speed_limit": f"{speed_limit_kmh:g}"}
        rid[0] += 1
        rels.append((rid[0], left, centre, tags))      # along s: on the left half
        along.append(rid[0])
        rid[0] += 1
        rels.append((rid[0], right_back, centre_back, tags))  # back: the right half, reversed
        back.append(rid[0])
    with open(out, "w") as f:
        f.write("<?xml version='1.0' encoding='UTF-8'?>\n<osm version=\"0.6\" generator=\"swb_isaac_world.course\">\n")
        for n, lat, lon, px, py, pz in nodes:
            f.write(f'  <node id="{n}" visible="true" version="1" lat="{lat:.9f}" lon="{lon:.9f}">\n'
                    f'    <tag k="local_x" v="{px:.4f}"/>\n    <tag k="local_y" v="{py:.4f}"/>\n'
                    f'    <tag k="ele" v="{pz:.4f}"/>\n  </node>\n')
        for w, refs, tags in ways:
            f.write(f'  <way id="{w}" visible="true" version="1">\n')
            for r_ in refs:
                f.write(f'    <nd ref="{r_}"/>\n')
            for k, v in tags.items():
                f.write(f'    <tag k="{k}" v="{v}"/>\n')
            f.write("  </way>\n")
        for r_, left, right, tags in rels:
            f.write(f'  <relation id="{r_}" visible="true" version="1">\n'
                    f'    <member type="way" ref="{left}" role="left"/>\n'
                    f'    <member type="way" ref="{right}" role="right"/>\n')
            for k, v in tags.items():
                f.write(f'    <tag k="{k}" v="{v}"/>\n')
            f.write("  </relation>\n")
        f.write("</osm>\n")
    return along, back


def _sample_terrain(m, c, rng):
    """Points on the terrain's triangles: STEP_NEAR near the road, STEP_FAR beyond."""
    from scipy.spatial import cKDTree

    ny, nx = (int(v) for v in m["terrain_shape"])
    P = m["terrain_points"].reshape(ny, nx, 3)
    x0, y0 = P[0, 0, 0], P[0, 0, 1]
    g = P[0, 1, 0] - x0
    Z = P[..., 2]
    out = []
    tree = cKDTree(np.stack([c["x"], c["y"]], axis=1))
    for step in (STEP_NEAR, STEP_FAR):
        xs = np.arange(x0, P[0, -1, 0] - 1e-6, step)
        ys = np.arange(y0, P[-1, 0, 1] - 1e-6, step)
        gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
        gx = gx + rng.uniform(-step / 2, step / 2, gx.shape)
        gy = gy + rng.uniform(-step / 2, step / 2, gy.shape)
        gx = gx.clip(x0, P[0, -1, 0] - 1e-6)
        gy = gy.clip(y0, P[-1, 0, 1] - 1e-6)
        d, _ = tree.query(np.stack([gx, gy], axis=1))
        keep = d <= NEAR if step == STEP_NEAR else d > NEAR
        gx, gy = gx[keep], gy[keep]
        i = ((gx - x0) / g).astype(int)
        j = ((gy - y0) / g).astype(int)
        fx, fy = (gx - x0) / g - i, (gy - y0) / g - j
        z00, z10, z11, z01 = Z[j, i], Z[j, i + 1], Z[j + 1, i + 1], Z[j + 1, i]
        # the quad (00, 10, 11, 01) split on 00-11: below the diagonal (fx >= fy) the triangle
        # 00-10-11, above it 00-11-01
        lower = fx >= fy
        z = np.where(lower, z00 + fx * (z10 - z00) + fy * (z11 - z10),
                     z00 + fy * (z01 - z00) + fx * (z11 - z01))
        out.append(np.stack([gx, gy, z], axis=1))
    return np.concatenate(out)


def _sample_ribbon(pts, faces, step, rng):
    """Points on a ribbon's quads (two triangles each, split 0-2), about one per step^2."""
    a, b, cc, d = (pts[faces[:, k]] for k in range(4))
    out = []
    for p0, p1, p2 in ((a, b, cc), (a, cc, d)):
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        n = rng.poisson(area / (step * step))
        tri = np.repeat(np.arange(len(p0)), n)
        u, v = rng.random(len(tri)), rng.random(len(tri))
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        out.append(p0[tri] + u[:, None] * (p1 - p0)[tri] + v[:, None] * (p2 - p0)[tri])
    return np.concatenate(out)


def point_cloud(m, c, out, extra=()):
    rng = np.random.default_rng(58)
    parts = [_sample_terrain(m, c, rng), _sample_ribbon(m["road_points"], m["road_faces"],
                                                        STEP_NEAR, rng)]
    parts += list(extra)
    pts = np.concatenate(parts)
    # one point per VOXEL cube (the trees' foliage alone would be millions of points)
    key = np.floor(pts / VOXEL).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    pts = pts[np.sort(first)].astype(np.float32)
    with open(out, "wb") as f:
        f.write(("# .PCD v0.7 - swb_isaac_world.course (roadmap 058)\nVERSION 0.7\n"
                 "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                 f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\n"
                 "DATA binary\n").encode())
        f.write(pts.tobytes())
    return len(pts)


def projector(frame, map_dir):
    with open(os.path.join(map_dir, "map_projector_info.yaml"), "w") as f:
        f.write("# Generated by swb_isaac_world.course (roadmap 058): a local transverse Mercator at the\n"
                "# course's origin; every lanelet point also carries local_x/local_y.\n"
                "projector_type: TransverseMercator\nvertical_datum: WGS84\nmap_origin:\n"
                f"  latitude: {frame.lat0}\n  longitude: {frame.lon0}\n  altitude: {frame.alt0}\n")
    with open(os.path.join(map_dir, "map_config.yaml"), "w") as f:
        f.write("/**:\n  ros__parameters:\n    map_origin:\n"
                f"      latitude: {frame.lat0}\n      longitude: {frame.lon0}\n      elevation: {frame.alt0}\n"
                "      roll: 0.0\n      pitch: 0.0\n      yaw: 0.0\n")


def build(c, course, frame, map_dir, speed_limit_kmh=60.0):
    m = np.load(os.path.join(map_dir, "course", "meshes.npz"))
    along, back = lanelets(c, course, frame, os.path.join(map_dir, "lanelet2_map.osm"),
                           speed_limit_kmh)
    extra = []
    props = os.path.join(map_dir, "course", "props_points.npy")
    if os.path.exists(props):  # the USD step's guardrails and trees (C5)
        extra.append(np.load(props))
    n = point_cloud(m, c, os.path.join(map_dir, "pointcloud_map.pcd"), extra)
    projector(frame, map_dir)
    return {"lanelets_along": along, "lanelets_back": back, "pcd_points": n}
