"""C5: a course's props, rough: guardrails and trees from the CARLA Isaac set ($CARLA_USD).

Guardrails, one asset segment after another along the road's edge, where the ground beside the
shoulder falls more than `DROP` m below the road within `DROP_AT` m, and on the outside of every
bend tighter than `RAIL_RADIUS` m. Each segment has an invisible box collider; the asset's own
mesh has none.

Trees where the aerial photo is forest (green over red, not bright), at about the course's
`tree_spacing` between the road's shoulder plus `TREE_CLEAR` m and its `tree_band`, as a
PointInstancer of the course's tree assets, each turned and scaled at random: cheap to render,
no colliders (a car that leaves the road meets the guardrail first, or the hillside).

The props' surfaces within the point-cloud map's reach are sampled into
`<work>/props_points.npy` for `maps.point_cloud`, so NDT's map has the guardrails and trees the
lidar sees.
"""
import math

import numpy as np

DROP, DROP_AT = 1.0, 3.0
RAIL_RADIUS = 30.0
RAIL_OFFSET = 0.4   # m beyond the shoulder
TREE_CLEAR = 4.0
SAMPLE_DENSITY = 6.0  # points per m^2 of a prop's surface
MAP_REACH = 120.0


def _terrain_z(m):
    from scipy.interpolate import RegularGridInterpolator

    ny, nx = (int(v) for v in m["terrain_shape"])
    P = m["terrain_points"].reshape(ny, nx, 3)
    f = RegularGridInterpolator((P[:, 0, 1], P[0, :, 0]), P[..., 2], bounds_error=False,
                                fill_value=None)
    return lambda x, y: f(np.stack([np.asarray(y), np.asarray(x)], axis=-1))


def guardrails(c, m, half, seg_len):
    """(x, y, z, yaw, length) of each segment, from its start, both sides."""
    from .geometry import _normals

    tz = _terrain_z(m)
    x, y, z, k = c["x"], c["y"], c["z"], c["curvature"]
    nrm = _normals(x, y)
    step = int(round(seg_len / 0.5))
    out = []
    for side in (1, -1):
        off = side * (half + RAIL_OFFSET)
        probe = side * (half + DROP_AT)
        for i in range(0, len(x) - step, step):
            gz = float(tz(x[i] + probe * nrm[i, 0], y[i] + probe * nrm[i, 1]))
            outer = abs(k[i]) > 1.0 / RAIL_RADIUS and np.sign(k[i]) == -side
            if gz < z[i] - DROP or outer:
                px, py = x[i] + off * nrm[i, 0], y[i] + off * nrm[i, 1]
                qx, qy = x[i + step] + off * nrm[i + step, 0], y[i + step] + off * nrm[i + step, 1]
                yaw = math.atan2(qy - py, qx - px)
                out.append((px, py, float(z[i]), yaw, math.hypot(qx - px, qy - py)))
    return out


def trees(c, m, half, photo, spacing, band, n_kinds, rng):
    """(x, y, z, yaw, scale, kind) of each tree."""
    from PIL import Image
    from scipy.spatial import cKDTree

    tz = _terrain_z(m)
    ny, nx = (int(v) for v in m["terrain_shape"])
    P = m["terrain_points"].reshape(ny, nx, 3)
    uv = m["terrain_uv"].reshape(ny, nx, 2)
    x0, x1, y0, y1 = P[0, 0, 0], P[0, -1, 0], P[0, 0, 1], P[-1, 0, 1]
    xs = np.arange(x0 + spacing, x1 - spacing, spacing)
    ys = np.arange(y0 + spacing, y1 - spacing, spacing)
    gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
    gx = gx + rng.uniform(-0.4, 0.4, gx.shape) * spacing
    gy = gy + rng.uniform(-0.4, 0.4, gy.shape) * spacing
    d, _ = cKDTree(np.stack([c["x"], c["y"]], axis=1)).query(np.stack([gx, gy], axis=1))
    keep = (d > half + TREE_CLEAR) & (d < band)
    gx, gy = gx[keep], gy[keep]
    # the photo under each: the terrain's UV, interpolated on the grid
    g = P[0, 1, 0] - x0
    i = ((gx - x0) / g).astype(int).clip(0, nx - 1)
    j = ((gy - y0) / g).astype(int).clip(0, ny - 1)
    u, v = uv[j, i, 0], uv[j, i, 1]
    img = np.asarray(Image.open(photo).convert("RGB")).astype(float)
    H, W = img.shape[:2]
    pix = img[((1 - v) * H).astype(int).clip(0, H - 1), (u * W).astype(int).clip(0, W - 1)]
    r_, g_, b_ = pix[:, 0], pix[:, 1], pix[:, 2]
    forest = (g_ > 1.05 * r_) & (g_ > b_) & (pix.mean(axis=1) < 140)
    gx, gy = gx[forest], gy[forest]
    gz = tz(gx, gy)
    n = len(gx)
    return np.stack([gx, gy, gz, rng.uniform(0, 2 * math.pi, n), rng.uniform(0.8, 1.3, n),
                     rng.integers(0, n_kinds, n)], axis=1)


def asset_surface(path):
    """An asset's triangles in its own frame (metres, z up): arrays a, b, c of corners."""
    from pxr import Usd, UsdGeom

    st = Usd.Stage.Open(path)
    mpu = UsdGeom.GetStageMetersPerUnit(st) or 1.0
    cache = UsdGeom.XformCache()
    parts = []
    for prim in st.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get() or [], dtype=float)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=int)
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=int)
        if not len(pts) or not len(counts):
            continue
        M = np.array(cache.GetLocalToWorldTransform(prim))  # row vectors
        pts = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3] * mpu
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        tri = [(idx[o], idx[o + t], idx[o + t + 1]) for o, n in zip(starts, counts)
               for t in range(1, n - 1)]
        tri = np.array(tri, dtype=int)
        parts.append((pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def sample_surface(tri, density, rng):
    a, b, c = tri
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = rng.poisson(area * density)
    k = np.repeat(np.arange(len(a)), n)
    u, v = rng.random(len(k)), rng.random(len(k))
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    return a[k] + u[:, None] * (b - a)[k] + v[:, None] * (c - a)[k]


def place(pts, x, y, z, yaw, scale=1.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    p = pts * scale
    return np.stack([x + cy * p[:, 0] - sy * p[:, 1], y + sy * p[:, 0] + cy * p[:, 1], z + p[:, 2]], axis=1)
