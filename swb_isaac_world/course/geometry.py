"""C3-C4: the terrain and the road as meshes in the map frame, written to `<work>/meshes.npz`
for the USD step (Isaac's Python) to read.

The terrain is a grid at `TERRAIN_STEP` over the section's box, its heights the DEM's. Under the
road and its shoulders it is cut or filled to the road's own height (`ROAD_SINK` below it), and
it blends back to the DEM over `BLEND` m beyond the shoulder, so the road sits in the hillside
as a real one does. Its UVs place the aerial-photo mosaic (`photo.jpg`), so the forest, the
road's scar and the buildings show where they are.

The road is a ribbon along the centreline at the lanes' and shoulders' width, its lines (the
centre line, yellow, and the two edge lines, white) thin ribbons just above it. Each mesh is
a list of quads. A closed course's ribbons close: their last quads join the last point to the
first.
"""
import numpy as np

TERRAIN_STEP = 2.0   # m
ROAD_SINK = 0.08     # m, the cut terrain below the road surface
ROAD_STRIP = 0.5    # m, the road's strips across (a wide quad on a climbing bend folds)
BLEND = 8.0          # m beyond the shoulder, back to the DEM
LINE_HALF = 0.075    # m, half a painted line's width
LINE_LIFT = 0.006    # m, the lines above the road


def is_closed(c):
    """Whether a centreline (centreline.build's dict or its npz) is a closed course's."""
    return "closed" in getattr(c, "files", c) and bool(c["closed"])


def _normals(x, y, closed=False):
    if closed:  # central differences around the loop
        d = np.stack([np.roll(x, -1) - np.roll(x, 1), np.roll(y, -1) - np.roll(y, 1)], axis=1)
    else:
        d = np.stack([np.gradient(x), np.gradient(y)], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.stack([-d[:, 1], d[:, 0]], axis=1)  # to the left of travel


def ribbon(c, left, right, lift=0.0, strip=None):
    """Quads between offsets `left` and `right` (m, left of the centreline positive) along it,
    in strips across of at most `strip` m (None: one): points (k+1 rows of n, 3), faces
    ((n-1) k, 4; n k closed), row by row from `left`.

    A section is level across, so on a bend that climbs each quad is twisted, and the two
    triangles a collider makes of it meet at an angle that grows with its width: 3.4 deg one
    8 m quad across Haruna's hairpins, a fold every half metre that a wheel's raycast feels
    (swb docs/issues/024); 0.6 deg in 0.5 m strips."""
    x, y, z = c["x"], c["y"], c["z"]
    closed = is_closed(c)
    nrm = _normals(x, y, closed)
    k = 1 if strip is None else max(1, int(np.ceil(abs(left - right) / strip)))
    rows = [np.stack([x + o * nrm[:, 0], y + o * nrm[:, 1], z + lift], axis=1)
            for o in np.linspace(left, right, k + 1)]
    pts = np.concatenate(rows)
    n = len(x)
    i = np.arange(n if closed else n - 1)
    i1 = (i + 1) % n
    faces = np.concatenate([np.stack([r * n + i, r * n + i1, (r + 1) * n + i1,
                                      (r + 1) * n + i], axis=1)  # one winding; usd._face_up turns it up
                            for r in range(k)])
    return pts, faces


def road(c, course):
    """The paved surface and its lines: a dict of (points, faces)."""
    r = course["road"]
    lanes, lane, shoulder = r["lanes"], r["lane_width"], r["shoulder"]
    half = lanes * lane / 2 + shoulder
    edge = lanes * lane / 2  # the lines sit at the lanes' outer edges
    return {
        "road": ribbon(c, half, -half, strip=ROAD_STRIP),
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
