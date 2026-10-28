"""C3-C4: the terrain and the road as meshes in the map frame, written to `<work>/meshes.npz`
for the USD step (Isaac's Python) to read.

The terrain is a grid at `TERRAIN_STEP` over the section's box, its heights the DEM's. Under the
road and its shoulders it is cut or filled to the road's own height (`ROAD_SINK` below it), and
it blends back to the DEM over `BLEND` m beyond the shoulder, so the road sits in the hillside
as a real one does. Its UVs place the aerial-photo mosaic (`photo.jpg`), so the forest, the
road's scar and the buildings show where they are.

The road is a ribbon along the centreline at the lanes' and shoulders' width, its lines (the
centre line, yellow, and the two edge lines, white) thin ribbons just above it. Each mesh is
a list of quads.
"""
import numpy as np

TERRAIN_STEP = 2.0   # m
ROAD_SINK = 0.08     # m, the cut terrain below the road surface
BLEND = 8.0          # m beyond the shoulder, back to the DEM
LINE_HALF = 0.075    # m, half a painted line's width
LINE_LIFT = 0.006    # m, the lines above the road


def _normals(x, y):
    d = np.stack([np.gradient(x), np.gradient(y)], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.stack([-d[:, 1], d[:, 0]], axis=1)  # to the left of travel


def ribbon(c, left, right, lift=0.0):
    """Quads between offsets `left` and `right` (m, left of the centreline positive) along it:
    points (2n, 3) and faces (n-1, 4)."""
    x, y, z = c["x"], c["y"], c["z"]
    nrm = _normals(x, y)
    a = np.stack([x + left * nrm[:, 0], y + left * nrm[:, 1], z + lift], axis=1)
    b = np.stack([x + right * nrm[:, 0], y + right * nrm[:, 1], z + lift], axis=1)
    pts = np.concatenate([a, b])
    n = len(x)
    i = np.arange(n - 1)
    faces = np.stack([i, i + 1, n + i + 1, n + i], axis=1)  # counter-clockwise seen from above
    return pts, faces


def road(c, course):
    """The paved surface and its lines: a dict of (points, faces)."""
    r = course["road"]
    lanes, lane, shoulder = r["lanes"], r["lane_width"], r["shoulder"]
    half = lanes * lane / 2 + shoulder
    edge = lanes * lane / 2  # the lines sit at the lanes' outer edges
    return {
        "road": ribbon(c, half, -half),
        "line_centre": ribbon(c, LINE_HALF, -LINE_HALF, LINE_LIFT),
        "line_left": ribbon(c, edge, edge - 2 * LINE_HALF, LINE_LIFT),
        "line_right": ribbon(c, -edge + 2 * LINE_HALF, -edge, LINE_LIFT),
        "half_width": half,
    }


def terrain(c, course, dem, frame, photo_block, photo_zoom, box_xy):
    """The terrain grid over box_xy (x0, x1, y0, y1): points (N, 3), quads, UVs into the photo
    mosaic, and the grid's shape."""
    from scipy.spatial import cKDTree

    x0, x1, y0, y1 = box_xy
    xs = np.arange(x0, x1 + 1e-6, TERRAIN_STEP)
    ys = np.arange(y0, y1 + 1e-6, TERRAIN_STEP)
    gx, gy = np.meshgrid(xs, ys)
    lon, lat = frame.proj(gx.ravel(), gy.ravel(), inverse=True)
    z = np.array([dem.at(a, o) for a, o in zip(lat, lon)]) - frame.alt0
    if np.isnan(z).any():  # a hole in the DEM: the nearest height
        good = ~np.isnan(z)
        tree = cKDTree(np.stack([gx.ravel()[good], gy.ravel()[good]], axis=1))
        _, j = tree.query(np.stack([gx.ravel()[~good], gy.ravel()[~good]], axis=1))
        z[~good] = z[good][j]
    # cut and fill to the road
    half = road(c, course)["half_width"]
    tree = cKDTree(np.stack([c["x"], c["y"]], axis=1))
    d, j = tree.query(np.stack([gx.ravel(), gy.ravel()], axis=1))
    zr = c["z"][j] - ROAD_SINK
    w = np.clip((d - half) / BLEND, 0.0, 1.0)
    w = w * w * (3 - 2 * w)  # smoothstep
    z = np.where(d <= half, zr, (1 - w) * zr + w * z)
    # UVs: the photo mosaic's pixel / its size, v up
    tx = (lon + 180.0) / 360.0 * 2 ** photo_zoom
    ty = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * 2 ** photo_zoom
    px0, py0, nx, ny = photo_block
    u = (tx - px0) / nx
    v = 1.0 - (ty - py0) / ny
    pts = np.stack([gx.ravel(), gy.ravel(), z], axis=1)
    nc = len(xs)
    jj, ii = np.meshgrid(np.arange(len(ys) - 1), np.arange(nc - 1), indexing="ij")
    a = (jj * nc + ii).ravel()
    faces = np.stack([a, a + 1, a + nc + 1, a + nc], axis=1)
    return pts, faces, np.stack([u, v], axis=1), (len(ys), nc)


def build(c, course, dem, frame, photo_block, photo_zoom, out):
    """Write the meshes; returns a short summary."""
    m = course["margin"]
    box = (float(c["x"].min() - m), float(c["x"].max() + m),
           float(c["y"].min() - m), float(c["y"].max() + m))
    tp, tf, tuv, shape = terrain(c, course, dem, frame, photo_block, photo_zoom, box)
    rd = road(c, course)
    arrays = {"terrain_points": tp, "terrain_faces": tf, "terrain_uv": tuv,
              "terrain_shape": np.array(shape), "box": np.array(box),
              "half_width": np.array(rd["half_width"])}
    for k in ("road", "line_centre", "line_left", "line_right"):
        arrays[f"{k}_points"], arrays[f"{k}_faces"] = rd[k]
    np.savez_compressed(out, **arrays)
    return {"terrain_vertices": len(tp), "terrain_grid": list(shape), "box_m": [round(v, 1) for v in box],
            "road_quads": len(rd["road"][1]), "half_width_m": rd["half_width"],
            "terrain_z_range_m": [round(float(tp[:, 2].min() + frame.alt0), 1),
                                  round(float(tp[:, 2].max() + frame.alt0), 1)]}
