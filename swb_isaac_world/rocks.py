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
GRID = 4.0  # m, RockSet's index cells
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
    proto: int = None  # its prototype (`prototypes`), None: a shape of its own
    scale: float = 1.0  # its size over the prototype's

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


PROTOTYPES = 32  # placed rocks' shapes: PhysX cooks each once, every prim scales one
_PROTOS = []


def prototypes():
    """PROTOTYPES rock shapes of long axis 1 m, short axis up, seeded: what placement scales
    and turns (`scaled`). A prim of a prototype carries its points unscaled, its size as the
    prim's scale and its heading as its orientation, so PhysX's cooking cache meets the same
    mesh again (a hull cooked per rock cost the frame's step in rock_world.py)."""
    if not _PROTOS:
        rng = np.random.default_rng(2026)
        n = 48
        a = np.array(AXES) / AXES[0] / 2
        for _ in range(PROTOTYPES):
            q = _fibonacci(n) * a[None, :] * (1.0 + rng.uniform(-0.15, 0.15, size=(n, 1)))
            _PROTOS.append(from_points(q))
    return _PROTOS


def scaled(k, s):
    """Prototype k at long axis s m (its own frame: turned by its pose's orientation)."""
    r = prototypes()[k]
    pl = r.planes.copy()
    pl[:, 3] *= s
    return Rock(r.verts * s, r.faces, pl, r.volume * s ** 3, r.mass * s ** 3, r.radius * s,
                proto=k, scale=s)


def _yaw_quat(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


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
        # whether the tire may stand on the rock: one off the ground (thrown, lifted by the
        # chassis) holds nothing up, its only support its own inertia
        self.seen = np.ones(len(self.rocks), dtype=bool)
        # a grid of GRID m cells -> the rocks centred in them: `near` looks at the cells about
        # its points, not at every rock loaded (thousands, ten times a substep)
        self.cells, self.cell_of = {}, []
        for i in range(len(self.rocks)):
            self._file(i)

    def _cell(self, i):
        return (int(math.floor(self.pos[i, 0] / GRID)), int(math.floor(self.pos[i, 1] / GRID)))

    def _file(self, i):
        c = self._cell(i)
        self.cells.setdefault(c, []).append(i)
        if i < len(self.cell_of):
            self.cell_of[i] = c
        else:
            self.cell_of.append(c)

    def add(self, rocks, pos, quat):
        """More rocks; their indices follow the ones there."""
        self.rocks += list(rocks)
        self.pos = np.concatenate([self.pos, np.asarray(pos, dtype=float).reshape(-1, 3)])
        self.quat = np.concatenate([self.quat, np.asarray(quat, dtype=float).reshape(-1, 4)])
        self.r = np.concatenate([self.r, [r.radius for r in rocks]])
        self._planes += [None] * len(rocks)
        self.seen = np.concatenate([self.seen, np.ones(len(rocks), dtype=bool)])
        for i in range(len(self.rocks) - len(rocks), len(self.rocks)):
            self._file(i)

    def set_pose(self, i, pos, quat):
        self.pos[i], self.quat[i] = pos, quat
        self._planes[i] = None
        c = self._cell(i)
        if c != self.cell_of[i]:  # moved into another cell
            self.cells[self.cell_of[i]].remove(i)
            self.cells.setdefault(c, []).append(i)
            self.cell_of[i] = c

    def planes(self, i):
        """Rock i's face planes in the world: n . X <= h."""
        p = self._planes[i]
        if p is None:
            pl = self.rocks[i].planes
            n = pl[:, :3] @ _rot(self.quat[i]).T
            p = self._planes[i] = np.concatenate(
                [n, (pl[:, 3] + n @ self.pos[i])[:, None]], axis=1)
        return p

    def bottom(self, i):
        """Rock i's lowest point, world z."""
        return float(self.pos[i, 2] + (self.rocks[i].verts @ _rot(self.quat[i]).T)[:, 2].min())

    def rest(self, i, road, gap=0.03):
        """Whether rock i rests on the road (its lowest point within `gap` of the road
        under its centre), as `seen` keeps it."""
        z = float(np.asarray(road(np.array([self.pos[i, 0]]), np.array([self.pos[i, 1]])))[0])
        self.seen[i] = self.bottom(i) - z <= gap
        return self.seen[i]

    def near(self, x, y, margin=0.0):
        """The rocks the tire may stand on whose bounding circle comes within `margin` of
        any point."""
        if not self.rocks:
            return np.zeros(0, dtype=int)
        x, y = np.ravel(x), np.ravel(y)
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        pad = float(self.r.max()) + margin
        cand = [i for cx in range(int(math.floor((x0 - pad) / GRID)),
                                  int(math.floor((x1 + pad) / GRID)) + 1)
                for cy in range(int(math.floor((y0 - pad) / GRID)),
                                int(math.floor((y1 + pad) / GRID)) + 1)
                for i in self.cells.get((cx, cy), ())]
        if not cand:
            return np.zeros(0, dtype=int)
        c = np.array(cand)
        px, py = self.pos[c, 0], self.pos[c, 1]
        dx = np.maximum(np.maximum(x0 - px, px - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - py, py - y1), 0.0)
        return np.sort(c[(np.hypot(dx, dy) <= self.r[c] + margin) & self.seen[c]])

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
        width); `road` one function for all, or one per tire. Each sample's depth into the tire, max(0, surface - the tire's undeformed
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
            g = np.asarray((road[k] if isinstance(road, (list, tuple)) else road)(px, py),
                           dtype=float)
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


# --- placement ------------------------------------------------------------------------------

D_MIN = 0.09  # m, plan diameter of the smallest rock placed (about 1 kg); smaller is gravel,
# a surface class (047's soil), not a body
D_MAX = 1.5  # m
DYNAMIC_MAX_KG = 50.0  # heavier rocks are static, sunk: no car moves them
# lighter ones are loose whatever their embedding: gravel does not hold a 1 kg stone against
# a 1.9 t car's underbody (a held one stopped the car by 10 kN s in rock_world.py)


@dataclass
class Zones:
    """Golombek-Rapp rock cover k (the fraction of the ground's area under rocks) by where on
    the cross-section: assumed values for a forest road, the carriageway swept by traffic, the
    shoulder and ditch where rocks from the cut slope come to rest."""

    carriageway: float = 0.002
    shoulder: float = 0.01
    ditch: float = 0.03
    embed: tuple = (0.0, 0.4)  # the share of a rock's height below the surface, uniform
    # the carriageway is graded (assumed): a rock held in it stands at most this proud, m,
    # sunk deeper where it would stand more; and a rock lying on it is at most
    # carriageway_d_max across (a bigger one fallen on the road is a scenario's hazard)
    carriageway_protrude: float = 0.05
    carriageway_d_max: float = 0.30
    # m: with no ground filtered out for the rocks (`road_detail.yaml`'s `ground`), a loose
    # rock only where the detail is at most this under the base; None: anywhere
    loose_below: float = None

    def zone(self, section, v):
        """The zone at distance v from the centreline: carriageway, shoulder, ditch, or None
        beyond the section."""
        a = abs(float(v))
        if a <= section.half_width:
            return "carriageway"
        if a <= section.half_width + section.shoulder:
            return "shoulder"
        return "ditch" if a <= section.extent() else None

    def k(self, section, v):
        z = self.zone(section, v)
        return getattr(self, z) if z else 0.0


def golombek_rapp(k, d):
    """The number of rocks per m^2 of plan diameter between d and d + dd, per dd: from the
    cumulative fractional area F(D) = k exp(-q D), q = 1.79 + 0.152 / k (Golombek and Rapp
    1997, D in m), as -F'(D) over one rock's area."""
    q = 1.79 + 0.152 / k
    return k * q * np.exp(-q * d) / (math.pi * d * d / 4.0)


def _sizes(rng, k, area, n_grid=400):
    """Plan diameters of the rocks on `area` m^2 at cover k."""
    d = np.geomspace(D_MIN, D_MAX, n_grid)
    pdf = golombek_rapp(k, d)
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(d))])
    n = rng.poisson(area * cdf[-1])
    return np.interp(rng.uniform(0, cdf[-1], n), cdf, d)


LOOSE_BELOW = 0.02  # m: Zones.loose_below when the world's ground is not filtered out


def place(detail, i, j, zones=None):
    """The rocks of tile (i, j) of a road_detail.Detail: a list of (Rock, position (3,), yaw,
    dynamic), each a seeded draw from its corridor (its `rocks` Zones, or `zones` for all) and
    tile, so a tile placed again is the same. A rock rests on the true surface (the base, the
    corridor's base_z, plus the detail), sunk by its embed share.

    It is dynamic if light (DYNAMIC_MAX_KG), resting on the surface; a heavy one is static,
    sunk by its embed share. A loose rock rests on the collision tiles, which are the
    true surface, once the world's own ground is filtered out for the rocks; where it is not
    (Zones.loose_below set), it would rest on whichever is higher, so in a rut, on a crowned
    road's sides or in a ditch it would float over the true surface: there a rock is loose
    only where the detail is no more than loose_below under the base, and static elsewhere."""
    t = detail.tile
    out = []
    for c in detail.corridors:
        cz = zones or c.rocks
        if cz is None:
            continue
        rng = c._rng(4, i, j)
        for name in ("carriageway", "shoulder", "ditch"):
            k = getattr(cz, name)
            if k <= 0:
                continue
            for d in _sizes(rng, k, t * t):
                x, y = (i + rng.uniform()) * t, (j + rng.uniform()) * t
                u, v, near = c.uv(np.array([x]), np.array([y]))
                if not 0.0 < u[0] < c.length:
                    continue  # beyond the corridor's ends
                if not near[0] or cz.zone(c.section, v[0]) != name:
                    continue  # another zone's (each zone draws over the whole tile)
                rock = scaled(int(rng.integers(PROTOTYPES)), 2 * d / (1 + AXES[1]))
                yaw = float(rng.uniform(0.0, 2 * math.pi))
                embed = rng.uniform(*cz.embed)
                h = float(detail.height(np.array([x]), np.array([y]))[0])
                loose = rock.mass <= DYNAMIC_MAX_KG \
                    and (cz.loose_below is None or h >= -cz.loose_below)
                sink = embed * rock.height
                if name == "carriageway":
                    if loose and d > cz.carriageway_d_max:
                        continue  # cleared, or a scenario's hazard
                    if not loose:
                        sink = max(sink, rock.height - cz.carriageway_protrude)
                # a loose rock rests on the surface (sunk into the collision tile it would be
                # thrown out of it when it wakes); a held one is sunk
                z = float(c.base_z(u)[0]) + h - rock.verts[:, 2].min() \
                    - (0.0 if loose else sink)
                out.append((rock, np.array([x, y, z]), yaw, loose))
    return out


def merged(entries):
    """The rocks of `entries` ((Rock, position, yaw, dynamic)) as one triangle mesh in the
    world: (verts (k, 3), faces (f, 3)). A tile's static rocks are one collider and one render
    prim so: the stage's cost grows with its prims, not with their triangles (R4: 8531 rock
    prims took 16 ms a frame, every USD change re-walking them)."""
    vs, fs, n = [], [], 0
    for rock, p, yaw, _ in entries:
        c, s = math.cos(yaw), math.sin(yaw)
        vs.append(rock.verts @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T + p)
        fs.append(rock.faces + n)
        n += len(rock.verts)
    if not vs:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.concatenate(vs), np.concatenate(fs)
