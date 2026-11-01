"""Rocks on the road (roadmap 066 R4): one convex hull per rock, for PhysX and for the tire.
Pure numpy (scipy for the hull).

A rock is a seeded convex hull of at most 64 vertices: points on an ellipsoid with a natural
rock's axes (Zingg: intermediate/long about 0.7, short/long about 0.5), each pushed in or out
by up to `rough`. PhysX collides with that hull (a convex mesh: it is already one, so PhysX's
own hull of the points is the same), and the tire reads its top surface: the vertical line's
intersection with the hull's face planes, exact, no sampling. So collision and the wheel see
one rock, as R3 has one road.

PhysX's wheel query does not see rocks (they are in a collision group the ground query
filters out); the tire contact (`road_contact`) does, through `height`: the road, or the top
of a rock where one is higher. A rock under a wheel takes its share of the tire's force,
reversed (question 2: momentum is conserved): `shares` splits a tire's load between the road
and each rock as a bed of radial springs would, by how deep each presses into the tire.
"""
from dataclasses import dataclass
import math

import numpy as np

DENSITY = 2650.0  # kg/m^3, granite and basalt about 2.6-2.9; quartz sandstone 2.65
AXES = (1.0, 0.7, 0.5)  # long : intermediate : short (Zingg's mean for river and road gravel)
MAX_VERTS = 64  # PhysX's default hull vertex limit (and GPU contact's)
GRAD_MAX = 3.0  # a face's gradient under a tire at most (72 deg): the tread rounds an edge


@dataclass
class Rock:
    """A hull in its own frame, centred on its centroid: verts (k, 3), planes (f, 4) as
    n . x <= h (n unit, outward), volume m^3, mass kg (DENSITY), radius m (bounding sphere)."""

    verts: np.ndarray
    faces: np.ndarray  # (f, 3) vertex indices, outward winding
    planes: np.ndarray
    volume: float
    mass: float
    radius: float

    @property
    def height(self):
        return float(self.verts[:, 2].max() - self.verts[:, 2].min())


def _fibonacci(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def make_rock(rng, d, axes=AXES, rough=0.15, n=48, density=DENSITY):
    """A rock whose long axis is `d` m: the hull of `n` ellipsoid points (n <= MAX_VERTS), each
    scaled radially by 1 +- rough, the short axis vertical (a rock at rest lies flat)."""
    n = min(n, MAX_VERTS)
    a = np.array(axes) / axes[0] * d / 2
    p = _fibonacci(n) * a[None, :]
    p = p * (1.0 + rng.uniform(-rough, rough, size=(n, 1)))
    yaw = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(yaw), math.sin(yaw)
    p = p @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
    return from_points(p, density)


def from_points(p, density=DENSITY):
    """The hull of points p (k, 3), centred on its volume's centroid."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(p)
    verts = p[hull.vertices]
    remap = {int(v): i for i, v in enumerate(hull.vertices)}
    faces = np.array([[remap[int(v)] for v in f] for f in hull.simplices])
    # the centroid and volume by tetrahedra from an interior point
    o = verts.mean(axis=0)
    a, b, c = verts[faces[:, 0]] - o, verts[faces[:, 1]] - o, verts[faces[:, 2]] - o
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    # outward winding: the triple product positive from the interior point
    flip = vol6 < 0
    faces[flip] = faces[flip][:, ::-1]
    vol6 = np.abs(vol6)
    volume = float(vol6.sum() / 6.0)
    cen = o + ((a + b + c) / 4.0 * vol6[:, None]).sum(axis=0) / vol6.sum()
    verts = verts - cen
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    planes = np.concatenate([nrm, np.einsum("ij,ij->i", nrm, v0)[:, None]], axis=1)
    return Rock(verts, faces, planes, volume, volume * density,
                float(np.linalg.norm(verts, axis=1).max()))


def _rot(q):
    """Rotation matrix of a unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class RockSet:
    """Rocks with poses in the world: `rocks` (list of Rock), `pos` (N, 3) and `quat` (N, 4,
    w first) as PhysX has them. `top(x, y)` is the highest rock surface over each point
    (-inf where none), with the rock's index (-1)."""

    def __init__(self, rocks, pos, quat):
        self.rocks = list(rocks)
        self.pos = np.asarray(pos, dtype=float).reshape(-1, 3)
        self.quat = np.asarray(quat, dtype=float).reshape(-1, 4)
        self.r = np.array([r.radius for r in self.rocks])
        self._planes = [None] * len(self.rocks)

    def set_pose(self, i, pos, quat):
        self.pos[i], self.quat[i] = pos, quat
        self._planes[i] = None

    def planes(self, i):
        """Rock i's face planes in the world: n . X <= h."""
        p = self._planes[i]
        if p is None:
            pl = self.rocks[i].planes
            n = pl[:, :3] @ _rot(self.quat[i]).T
            p = self._planes[i] = np.concatenate(
                [n, (pl[:, 3] + n @ self.pos[i])[:, None]], axis=1)
        return p

    def near(self, x, y, margin=0.0):
        """The rocks whose bounding circle comes within `margin` of any point."""
        if not self.rocks:
            return np.zeros(0, dtype=int)
        x, y = np.ravel(x), np.ravel(y)
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        dx = np.maximum(np.maximum(x0 - self.pos[:, 0], self.pos[:, 0] - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - self.pos[:, 1], self.pos[:, 1] - y1), 0.0)
        return np.nonzero(np.hypot(dx, dy) <= self.r + margin)[0]

    def top(self, x, y, rocks=None, normals=False):
        """The highest rock surface over each point and its rock; with `normals`, also the
        face's outward normal there (..., 3), world."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        out = np.full(x.shape, -np.inf)
        idx = np.full(x.shape, -1)
        nrm = np.zeros(x.shape + (3,))
        fx, fy, fo, fi = x.reshape(-1), y.reshape(-1), out.reshape(-1), idx.reshape(-1)
        fn = nrm.reshape(-1, 3)
        for i in self.near(fx, fy) if rocks is None else rocks:
            sel = np.nonzero(np.hypot(fx - self.pos[i, 0], fy - self.pos[i, 1]) <= self.r[i])[0]
            if not sel.size:
                continue
            pl = self.planes(i)
            nz = pl[:, 2]
            rhs = pl[None, :, 3] - pl[None, :, 0] * fx[sel, None] - pl[None, :, 1] * fy[sel, None]
            up, dn, side = nz > 1e-9, nz < -1e-9, np.abs(nz) <= 1e-9
            if up.any():
                zu = rhs[:, up] / nz[up]
                face = np.argmin(zu, axis=1)
                zt = zu[np.arange(sel.size), face]
            else:
                zt = np.full(sel.size, np.inf)
            zb = np.max(rhs[:, dn] / nz[dn], axis=1) if dn.any() else np.full(sel.size, -np.inf)
            ok = zt >= zb
            if side.any():
                ok &= np.all(rhs[:, side] >= 0, axis=1)
            better = ok & (zt > fo[sel])
            fo[sel[better]] = zt[better]
            fi[sel[better]] = i
            if normals and up.any():
                fn[sel[better]] = pl[np.nonzero(up)[0][face[better]], :3]
        return (out, idx, nrm) if normals else (out, idx)

    def height(self, road):
        """The tire's road: `road(x, y)` (m, the same frame as the rocks' z), or a rock's top
        where one is higher."""
        def h(x, y):
            base = np.asarray(road(x, y), dtype=float)
            t, _ = self.top(x, y)
            return np.maximum(base, t)
        return h

    def shares(self, road, xy, heading, width, radius, zu, along=13, across=6):
        """Each tire's contact with the rocks, as a bed of radial springs over the footprint
        (along: -0.6 to 0.6 radius, the front and rear of a tire climbing a rock; across: the
        width). Each sample's depth into the tire, max(0, surface - the tire's undeformed
        bottom there), goes to whichever surface is highest there.

        Returns per tire a list of (rock, share, gradient along, gradient across, point):
        the rock's depth over the tire's whole depth (its share of the tire's load), and the
        depth-weighted gradients of its faces under the tire along the heading and to its
        left (exact, from the hull's planes; at most GRAD_MAX)."""
        out = []
        u = np.linspace(-0.6, 0.6, along)
        v = (np.arange(across) + 0.5) / across - 0.5
        for k in range(len(xy)):
            r = float(radius[k])
            hx, hy = heading[k]
            uu, vv = np.meshgrid(u * r, v * float(width[k]))  # (across, along)
            px = xy[k, 0] + hx * uu - hy * vv
            py = xy[k, 1] + hy * uu + hx * vv
            near = self.near(px, py)
            if not near.size:
                out.append([])
                continue
            bottom = float(zu[k]) - np.sqrt(np.maximum(r * r - uu * uu, 0.0))
            g = np.asarray(road(px, py), dtype=float)
            t, idx, n = self.top(px, py, near, normals=True)
            rock = t > g
            depth = np.maximum(np.where(rock, t, g) - bottom, 0.0)
            total = depth.sum()
            res = []
            if total > 0:
                nz = np.maximum(n[..., 2], 1e-6)
                gx, gy = -n[..., 0] / nz, -n[..., 1] / nz
                gu = np.clip(gx * hx + gy * hy, -GRAD_MAX, GRAD_MAX)
                gv = np.clip(-gx * hy + gy * hx, -GRAD_MAX, GRAD_MAX)
                for i in near:
                    m = rock & (idx == i)
                    d = depth[m].sum()
                    if d > 0:
                        w = depth[m] / d
                        res.append((int(i), float(d / total), float((gu[m] * w).sum()),
                                    float((gv[m] * w).sum()),
                                    np.array([(px[m] * w).sum(), (py[m] * w).sum(),
                                              (t[m] * w).sum()])))
            out.append(res)
        return out


def reactions(shares, load, heading):
    """The forces on the rocks and the car's lateral force from them, N.
    Returns ({rock: [(force (3,), point (3,)), ...]}, per tire the lateral force (2,) on the
    car). A rock takes its share of the tire's load, down, and the push its faces give it: a
    contact force is normal to the face, so horizontal over vertical is the face's gradient,
    forward on a face rising ahead, aside on one rising across. The car takes the lateral
    reversed (the cam has none); the cam's slope force already holds the push along."""
    out = {}
    lat = np.zeros((len(shares), 2))
    for k, res in enumerate(shares):
        hx, hy = heading[k][0], heading[k][1]
        for i, s, gu, gv, p in res:
            fv = s * load[k]
            fu, fl = fv * gu, fv * gv  # on the rock, along the heading and to its left
            f = np.array([fu * hx - fl * hy, fu * hy + fl * hx, -fv])
            lat[k] -= (-fl * hy, fl * hx)
            out.setdefault(i, []).append((f, p))
    return out, lat


def limit(forces, mass, v_rock, v_push, dt):
    """The forces on a rock with their horizontal part scaled down so that one step's impulse
    takes the rock no faster, horizontally, than the surface pushing it (`v_push`, m/s: a
    rolling tire's surface at height z over its contact moves at v z / r). Applied as a force
    for the whole step, an unlimited push on a light rock throws it far faster than any
    tire could. Returns the scaled list and the scale."""
    fh = sum(f[:2] for f, _ in forces)
    j = np.linalg.norm(fh) * dt
    if j <= 0:
        return forces, 1.0
    u = fh / np.linalg.norm(fh)
    room = max(0.0, v_push - float(np.dot(v_rock[:2], u))) * mass
    sc = min(1.0, room / j)
    return [(np.array([f[0] * sc, f[1] * sc, f[2]]), p) for f, p in forces], sc
