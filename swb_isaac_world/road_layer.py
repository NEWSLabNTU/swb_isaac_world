"""The road's small-scale detail in the world server (roadmap 066 R3, with R5's and R2's tire
contact). Import after `SimulationApp` only for `MeshStream`; the rest is pure.

A world whose map directory has a `road_detail.yaml` next to its `surfaces.yaml` gets the
detail of its corridors (`road_detail.Corridor`) on top of its own ground, which stays the
base road PhysX's wheel query sees. Before every substep each ego's four wheels run
`road_contact.TireContact` on base + detail: its force goes to the body
(`apply_force(body path, force, position)`, for the next step) and each wheel's friction is the
surface layer's times the tire's load over PhysX's (`SurfaceLayer.road`). The base under a
wheel is the plane through the four wheels' last ground hits, moved to the wheel's own hit: the
base is smooth, so over a tire's patch it is a plane.

The ego's PhysX travel is the spec's plus PX_EXTRA (`vehicle_params`), its frame raised to
match, so PhysX's rigid limit lies beyond the corner's bump stop (R2).

    corridors:
      - centreline: [[x, y, z], ...]  # the map frame, m; z the base road's (optional)
        seed: 1
        roughness: {cls: C}           # road_detail.Roughness's fields
        section: {half_width: 1.5, crown: 0.0}   # Section's: a level forest road
        ruts: {}                      # Ruts' defaults; absent: none
        potholes: {per_100m: 2}
        washboard: [{u0: 120, u1: 180, pitch: 0.5}]
        water_bars: {at: [40, 95]}
        taper: 5                      # m: fades in and out at the ends (sections that overlap)
        rocks: {shoulder: 0.01, ditch: 0.03}   # rocks.Zones: Golombek-Rapp cover k (R4)
    ground: [/World/Town/Road]        # the base's prims, filtered out for the rocks (R4)

With `rocks`, the corridor's rocks are placed per tile in the tile worker and the tire stands on
them (`rocks.RockSet`, `layer.rocks`); their prims, collision tiles and sleep are
`rock_layer.RockStream`'s, which the world attaches as `layer.rock_stream`. A loose rock under
a tire takes its share of the tire's force, reversed, and the car the lateral reversed (R4).

Every key and field is listed, with its default, by `keys()` (`ros2 run swb_isaac_world
road_detail keys`); an unknown one is refused by name. Examples to start from are in the
package's `examples/road/` (installed to `share/swb_isaac_world/examples/road/`): Town01's
rough gravel road with rocks, its flat baseline, and the proving ground's. Without Isaac:

    ros2 run swb_isaac_world road_detail check <map dir | road_detail.yaml>
    ros2 run swb_isaac_world road_detail preview <map dir | road_detail.yaml> [-o road.png]
    ros2 run swb_isaac_world road_detail keys

(`python3 -m swb_isaac_world.road_layer ...` from the package's source tree is the same.)
"""
import argparse
import dataclasses
import logging
import math
import os
import sys

import numpy as np

from . import road_contact as rc
from . import road_detail as rd
from . import rocks as rk

log = logging.getLogger("swb_isaac_world")

PX_EXTRA = 0.10  # m of PhysX travel beyond the spec's (R2)
SCALE_STEP = 0.01  # the friction scale's step (as roll_bars')
FILE = "road_detail.yaml"
REACH = 48.0  # m ahead of a vehicle whose tiles are made in the worker
LEAD_S = 4.0  # s: at speed, the tiles of the next 4 s (a busy host delays the worker)
# aib.isaac.vehicle.WHEELS' order (front?, side): front left, front right, rear left, rear right
ORDER = ((True, 1), (True, -1), (False, 1), (False, -1))
WHEELS = len(ORDER)


class RoadDetailError(ValueError):
    pass


# A corridor's keys that are one of road_detail's (or rocks') dataclasses: its fields by name.
def _kinds():
    return {"roughness": rd.Roughness, "section": rd.Section, "ruts": rd.Ruts,
            "potholes": rd.Potholes, "water_bars": rd.WaterBars, "rocks": rk.Zones}


TOP_KEYS = {
    "corridors": "a list of corridors, each its centreline and its features",
    "ground": "the prims of the world's own road ground, which the rocks do not collide with "
              "(a list of prim paths; optional)",
}
CORRIDOR_KEYS = {
    "centreline": "two or more [x, y] or [x, y, z] in the map frame, m (z: the base road's)",
    "seed": "an integer: the same seed, the same road (default the corridor's index)",
    "roughness": "ISO 8608 roughness (a mapping; absent: class D)",
    "section": "the cross-section: crown, shoulder, ditch (a mapping; absent: defaults)",
    "ruts": "two wheel ruts and a windrow (a mapping, {} for defaults; absent: none)",
    "potholes": "Poisson potholes, more in sags (a mapping; absent: none)",
    "washboard": "a list of corrugation zones, each u0 and u1 m along the centreline",
    "water_bars": "oblique ridges across the road (a mapping; absent: none)",
    "rocks": "rock cover per part of the section (a mapping; absent: no rocks)",
    "taper": "m over which the detail fades in and out at the ends (default 0)",
}


REQUIRED = "(required)"  # a field with no default, in `keys()`


def _fields(cls):
    out = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # pragma: no cover
            out[f.name] = f.default_factory()
        else:
            out[f.name] = REQUIRED
    return out


def keys():
    """road_detail.yaml's schema: `{"<key>": what}` for the file's and a corridor's keys, and
    `{"corridors[].<key>.*": {field: default}}` for each mapping key (`REQUIRED`: no
    default)."""
    out = {k: v for k, v in TOP_KEYS.items()}
    for k, v in CORRIDOR_KEYS.items():
        out[f"corridors[].{k}"] = v
    for name, cls in _kinds().items():
        out[f"corridors[].{name}.*"] = _fields(cls)
    out["corridors[].washboard[].*"] = _fields(rd.Washboard)
    return out


def parse(data, where=FILE):
    """The corridors of a road_detail.yaml's contents; RoadDetailError lists every problem."""
    problems = []
    kinds = _kinds()
    out = []
    if data is not None and not isinstance(data, dict):
        raise RoadDetailError(f"{where}: not a mapping (corridors, ground)")
    unknown = set(data or {}) - set(TOP_KEYS)
    if unknown:
        problems.append(f"{where}: unknown key(s) {sorted(unknown)} (corridors, ground)")
    corridors = (data or {}).get("corridors") or []
    if not isinstance(corridors, list):
        problems.append(f"{where}: corridors must be a list")
        corridors = []
    ground = (data or {}).get("ground") or []
    if not isinstance(ground, list) or not all(isinstance(g, str) for g in ground):
        problems.append(f"{where}: ground must be a list of prim paths")
        ground = []
    for k, c in enumerate(corridors):
        at = f"{where}: corridors[{k}]"
        args = {}
        if not isinstance(c, dict):
            problems.append(f"{at}: not a mapping ({', '.join(CORRIDOR_KEYS)})")
            continue
        try:
            line = np.asarray(c.get("centreline") or [], dtype=float)
        except (TypeError, ValueError):
            line = np.zeros((0,))
        if line.ndim != 2 or line.shape[1] not in (2, 3) or len(line) < 2:
            problems.append(f"{at}: centreline must be two or more [x, y] or [x, y, z]")
            continue
        bad = set(c) - set(CORRIDOR_KEYS)
        if bad:
            problems.append(f"{at}: unknown key(s) {sorted(bad)} ({', '.join(CORRIDOR_KEYS)})")
        for name, cls in kinds.items():
            if name not in c:
                continue
            fields = {f.name for f in dataclasses.fields(cls)}
            given = c[name] or {}
            if not isinstance(given, dict):
                problems.append(f"{at}.{name}: a mapping of {', '.join(sorted(fields))}")
                continue
            if set(given) - fields:
                problems.append(f"{at}.{name}: unknown {sorted(set(given) - fields)} "
                                f"({', '.join(sorted(fields))})")
                continue
            if "at" in given:
                given = dict(given, at=tuple(given["at"]))
            if "embed" in given:
                given = dict(given, embed=tuple(given["embed"]))
            args[name] = cls(**given)
        zones = []
        wb_fields = _fields(rd.Washboard)
        for n, z in enumerate(c.get("washboard") or []):
            if not isinstance(z, dict):
                problems.append(f"{at}.washboard[{n}]: a mapping of {', '.join(wb_fields)}")
                continue
            if set(z) - set(wb_fields):
                problems.append(f"{at}.washboard[{n}]: unknown {sorted(set(z) - set(wb_fields))}"
                                f" ({', '.join(wb_fields)})")
                continue
            need = [f for f, d in wb_fields.items() if d is REQUIRED and f not in z]
            if need:
                problems.append(f"{at}.washboard[{n}]: needs {', '.join(need)} (m along the "
                                "centreline)")
                continue
            zones.append(rd.Washboard(**z))
        if "roughness" in args and args["roughness"].cls not in rd.ISO_CLASS \
                and not args["roughness"].gd0:
            problems.append(f"{at}.roughness: class {args['roughness'].cls} is not ISO 8608's "
                            f"A-H")
        if "rocks" in args and not ground:
            # the world's ground not filtered out for the rocks: loose ones only where it
            # does not hold them over the true surface (rocks.place)
            args["rocks"] = dataclasses.replace(args["rocks"], loose_below=rk.LOOSE_BELOW)
        out.append(rd.Corridor(line, seed=int(c.get("seed", k)), washboard=tuple(zones),
                               taper=float(c.get("taper", 0.0)), **args))
    if problems:
        raise RoadDetailError("\n".join(problems))
    return out


def load(path):
    import yaml

    with open(path) as f:
        return parse(yaml.safe_load(f) or {}, str(path))


def find(map_or_file, state=None):
    """road_detail.yaml for a map directory (the state directory's over the map's, `names`),
    or the file itself."""
    from . import names

    if os.path.isdir(map_or_file):
        return names.find_file(FILE, map_or_file, state)
    return str(map_or_file)


def describe(corridors):
    """One line per corridor: its length and features."""
    out = []
    for k, c in enumerate(corridors):
        parts = [f"{c.length:.0f} m", f"seed {c.seed}",
                 f"roughness {c.roughness.cls if not c.roughness.gd0 else c.roughness.gd0}",
                 f"half width {c.section.half_width:g} m, crown {c.section.crown:g}"]
        if c.ruts is not None:
            parts.append(f"ruts {c.ruts.depth:g} m")
        if c.potholes is not None:
            parts.append(f"potholes {c.potholes.per_100m:g}/100 m")
        if c.washboard:
            parts.append(f"washboard {len(c.washboard)} zone(s)")
        if c.water_bars is not None:
            parts.append(f"water bars {list(c.water_bars.at) or 'by grade'}")
        if c.rocks is not None:
            parts.append(f"rocks {c.rocks.carriageway:g}/{c.rocks.shoulder:g}/"
                         f"{c.rocks.ditch:g}")
        if c.taper:
            parts.append(f"taper {c.taper:g} m")
        out.append(f"corridor {k}: " + ", ".join(parts))
    return out


def profile(corridor, length=60.0, spacing=0.1, start=0.0):
    """The corridor's detail along its first `length` m from `start`, on a (u, v) grid:
    (u (n,), v (m,), heights (m, n) m above the base). Pure numpy (no rocks)."""
    c = corridor
    u = np.arange(start, min(start + length, c.length), spacing)
    half = c.section.extent()
    v = np.arange(-half, half + spacing / 2, spacing)
    seg = np.clip(np.searchsorted(c.seg_u, u, side="right") - 1, 0, len(c.seg_len) - 1)
    t = u - c.seg_u[seg]
    foot = c.centreline[seg] + t[:, None] * c.seg_dir[seg]
    left = np.stack([-c.seg_dir[seg, 1], c.seg_dir[seg, 0]], axis=1)
    pts = foot[None, :, :] + v[:, None, None] * left[None, :, :]
    detail = rd.Detail([c])
    try:
        h = detail.height(pts[..., 0], pts[..., 1])
    finally:
        detail.close()
    return u, v, h


def preview(path, out, corridor=0, length=60.0):
    """A picture of corridor `corridor`'s first `length` m (heights, and a cross-section and
    a wheel track's profile): writes `out` (matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = load(path)[corridor]
    u, v, h = profile(c, length)
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1, 1]})
    im = ax[0].imshow(h * 100, extent=(u[0], u[-1], v[0], v[-1]), origin="lower",
                      aspect="auto", cmap="terrain")
    fig.colorbar(im, ax=ax[0], label="cm above the base")
    ax[0].set_ylabel("v (m, left of the centreline)")
    ax[0].set_title(f"{path}: {describe([c])[0]}", fontsize=8)
    mid = len(u) // 2
    ax[1].plot(v, h[:, mid] * 100)
    ax[1].set_xlabel(f"v (m) at u {u[mid]:.1f} m")
    ax[1].set_ylabel("cm")
    track = int(np.argmin(np.abs(v - 0.8)))
    ax[2].plot(u, h[track] * 100)
    ax[2].set_xlabel("u (m along the centreline), the left wheel track (v 0.8 m)")
    ax[2].set_ylabel("cm")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="road_detail",
                                 description="Check, picture or list the keys of a "
                                             "road_detail.yaml (roadmap 066 R3)")
    ap.add_argument("command", choices=("check", "preview", "keys"))
    ap.add_argument("map", nargs="?", default="",
                    help="a map directory (its road_detail.yaml) or a road_detail.yaml")
    ap.add_argument("-o", "--output", default="",
                    help="preview: the image (default road_detail.png here)")
    ap.add_argument("--corridor", type=int, default=0, help="preview: which corridor")
    ap.add_argument("--length", type=float, default=60.0, help="preview: m along it")
    args = ap.parse_args(argv)
    if args.command == "keys":
        for k, v in keys().items():
            if isinstance(v, dict):
                fields = ", ".join(f"{f} {d}" for f, d in v.items())
                print(f"{k}: {fields}")
            else:
                print(f"{k}: {v}")
        return 0
    if not args.map:
        ap.error(f"{args.command} needs a map directory or a road_detail.yaml")
    path = find(args.map)
    if not os.path.exists(path):
        print(f"{path}: no road detail", file=sys.stderr)
        return 1
    try:
        corridors = load(path)
    except RoadDetailError as e:
        print(f"{path}: refused\n{e}", file=sys.stderr)
        return 1
    print(f"{path}: {len(corridors)} corridor(s)")
    for line in describe(corridors):
        print(f"  {line}")
    if args.command == "preview":
        if not 0 <= args.corridor < len(corridors):
            print(f"no corridor {args.corridor}", file=sys.stderr)
            return 1
        out = args.output or "road_detail.png"
        preview(path, out, args.corridor, args.length)
        print(f"wrote {out}")
    return 0


def load_ground(path):
    """road_detail.yaml's `ground`: the prims of the world's own road ground (the base), which
    the rocks do not collide with, so a loose rock rests on the detail's collision tiles."""
    import yaml

    with open(path) as f:
        return list((yaml.safe_load(f) or {}).get("ground") or [])


def vehicle_params(p):
    """aib VehicleParams `p` with PhysX's travel PX_EXTRA longer (its frame rises with it)."""
    return dataclasses.replace(p, suspension_travel=p.suspension_travel + PX_EXTRA)


class RoadLayer:
    def __init__(self, corridors, origin=(0.0, 0.0, 0.0), apply_force=None, workers=1):
        """`origin`: the world frame's origin in the map's (`SurfaceLayer.origin`)."""
        corridors = list(corridors)
        with_rocks = any(c.rocks is not None for c in corridors)
        self.detail = rd.Detail(corridors, workers=workers, max_tiles=96,
                                processes=bool(workers),
                                extras="swb_isaac_world.rocks:place" if with_rocks else None)
        # the rocks the tires stand on, world frame; filled by the world's rock_stream
        self.rocks = rk.RockSet([], np.zeros((0, 3)), np.zeros((0, 4))) if with_rocks else None
        self.rock_stream = None
        self.ox, self.oy = float(origin[0]), float(origin[1])
        self.apply_force = apply_force
        self.cars = {}  # vehicle path -> _Car
        self.cost_ms = []

    def height(self, x, y):
        """The detail at world (x, y), m above the base."""
        return self.detail.height(np.asarray(x) + self.ox, np.asarray(y) + self.oy)

    def _car(self, vehicle):
        car = self.cars.get(vehicle.path)
        if car is None:
            car = self.cars[vehicle.path] = _Car(vehicle)
        return car

    def before_substep(self, vehicles, dt):
        """Each aib vehicle's contact for the coming substep: its forces, and its friction
        scale (`scale`, per wheel) for the surface layer."""
        import time

        for v in vehicles:
            if v.last is None:
                continue
            t0 = time.perf_counter()
            car = self._car(v)
            self.detail.prefetch(self._ahead(v))
            out = car.step(self, dt)
            if out is not None and self.apply_force is not None:
                for force, position in out:
                    self.apply_force(v.path, force, position)
            self.cost_ms.append((time.perf_counter() - t0) * 1e3)

    def _ahead(self, v):
        (px, py, _), _, (vx, vy, _) = v.last
        speed = math.hypot(vx, vy) or 1.0
        reach = max(REACH, LEAD_S * speed)
        s = np.linspace(0.0, 1.0, max(1, int(reach / 8.0)) + 1)
        gx = px + self.ox + vx / speed * reach * s
        gy = py + self.oy + vy / speed * reach * s
        return self.detail.tiles_near(gx, gy, 6.0)

    def scale(self, path, i):
        car = self.cars.get(path)
        if car is None or car.scale is None:
            return 1.0
        # in steps of SCALE_STEP: aib writes a wheel's friction to USD only when it moves, and
        # on a large stage each write costs PhysX's next step a Fabric update
        return round(float(car.scale[i]) / SCALE_STEP) * SCALE_STEP

    def reset(self, path):
        car = self.cars.get(path)
        if car is not None:
            car.contact.reset()
            car.z_att_prev = None
            car.scale = None

    def forget(self, path):
        self.cars.pop(path, None)

    def close(self):
        self.detail.close()


class _Car:
    def __init__(self, vehicle):
        p = vehicle.p
        self.v = vehicle
        (front, rear), self.sag = p.suspension()
        real = p.suspension_travel - PX_EXTRA
        self.frame_z = p.wheel_radius + p.suspension_travel - self.sag
        self.attach = [((p.wheel_base if f else 0.0), side * p.wheel_tread / 2)
                       for f, side in ORDER]
        sides = {(f, side): i for i, (f, side) in enumerate(ORDER)}
        self.contact = rc.TireContact(
            [rc.Corner(*(front if f else rear), real, p.wheel_radius, p.wheel_width,
                       px_travel=p.suspension_travel) for f, _ in ORDER],
            bars=rc.anti_roll_bars((sides[(True, 1)], sides[(True, -1)]),
                                   (sides[(False, 1)], sides[(False, -1)]), front[1], rear[1]),
            kernel="auto")
        self.z_att_prev = None
        self.scale = None
        self.plane = None  # (a, b, c): z = a x + b y + c through the hits, then per wheel c_i
        self.last_out = None

    def _planes(self):
        """Per wheel (a, b, c_i) of the base under it: the plane through the ground hits, moved
        to each wheel's own hit; the last one where a wheel is off the ground."""
        hits = [c for c in (self.v.contacts or []) if c.on_ground]
        if len(hits) >= 3:
            a = np.array([[c.position[0], c.position[1], 1.0] for c in hits])
            z = np.array([c.position[2] for c in hits])
            ga, gb = _fit_slope(a[:, 0], a[:, 1], z)
            cs = []
            for k, c in enumerate(self.v.contacts):
                if c.on_ground:
                    cs.append(c.position[2] - ga * c.position[0] - gb * c.position[1])
                else:
                    cs.append(self.plane[2][k] if self.plane else
                              float(np.mean(z - ga * a[:, 0] - gb * a[:, 1])))
            self.plane = (float(ga), float(gb), np.array(cs))
        return self.plane

    def step(self, layer, dt):
        from aib.frames import qrot

        planes = self._planes()
        if planes is None:
            return None
        ga, gb, cs = planes
        (px, py, pz), q, (vx, vy, _) = self.v.last
        att = [qrot(q, (x, y, self.frame_z)) for x, y in self.attach]
        xy = np.array([[px + a[0], py + a[1]] for a in att])
        z_att = np.array([pz + a[2] for a in att])
        v_att = np.zeros(WHEELS) if self.z_att_prev is None else (z_att - self.z_att_prev) / dt
        self.z_att_prev = z_att
        fwd = qrot(q, (1.0, 0.0, 0.0))
        yaw = math.atan2(fwd[1], fwd[0])
        steer = getattr(self.v, "tire_angle", 0.0)
        heading = np.array([[math.cos(yaw + a), math.sin(yaw + a)]
                            for a in (steer, steer, 0.0, 0.0)])
        speed = np.full(WHEELS, vx * fwd[0] + vy * fwd[1])

        def base(x, y):
            x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            c = cs.reshape((WHEELS,) + (1,) * (x.ndim - 1))
            return ga * x + gb * y + c

        def road(x, y):
            return base(x, y) + layer.height(x, y)

        def road_at(k):  # wheel k's road at any points
            return lambda x, y: ga * np.asarray(x) + gb * np.asarray(y) + cs[k] \
                + layer.height(x, y)

        def road_any(x, y):  # the wheels' mean plane, within mm of each wheel's
            return ga * np.asarray(x) + gb * np.asarray(y) + float(np.mean(cs)) \
                + layer.height(x, y)

        rocks, stream = layer.rocks, layer.rock_stream
        zu0 = self.contact.zu.copy()
        # one grid lookup for the whole car: with no rock within reach of any wheel's samples
        # (the common case), no rock reads, rock tops or footprints this step
        if rocks is not None and not rocks.near(xy[:, 0], xy[:, 1],
                                                margin=self.contact.reach(dt, speed)).size:
            rocks = None
        if rocks is not None and stream is not None:
            stream.read_near(xy, road_any, dt)  # the awake rocks where PhysX has them
            height = rocks.height(road)
        else:
            rocks, height = None, road
        o = self.contact.step(dt, height, base, xy, heading, speed, z_att, v_att)
        self.scale = self.contact.friction_scale(o["load"], o["px_load"])
        self.last_out = o
        self.last_xy = xy
        p = self.v.p
        out = []
        for i in range(WHEELS):
            cz = pz + att[i][2] - self.frame_z + p.wheel_radius
            out.append(((float(o["dfx"][i] * heading[i, 0]), float(o["dfx"][i] * heading[i, 1]),
                         float(o["dfz"][i])), (float(xy[i, 0]), float(xy[i, 1]), cz)))
        if rocks is not None and len(rocks.rocks):
            sh = rocks.shares([road_at(k) for k in range(WHEELS)], xy, heading,
                              self.contact.width, self.contact.r, zu0)
            forces, lat = rk.reactions(sh, o["load"], heading)
            if forces and layer.apply_force is not None:
                sc = stream.push(forces, speed[0], p.wheel_radius, road_any, dt,
                                 layer.apply_force)
                lat = lat * sc  # the car feels only what the rocks took
            for i in range(WHEELS):  # the rocks' lateral push on the tire (the cam has none)
                if lat[i].any():
                    cz = pz + att[i][2] - self.frame_z
                    out.append(((float(lat[i][0]), float(lat[i][1]), 0.0),
                                (float(xy[i, 0]), float(xy[i, 1]), cz)))
        return out


def _fit_slope(x, y, z):
    """The slopes (a, b) of the least-squares plane z = a x + b y + c through the points, in
    closed form on centred coordinates (`np.linalg.lstsq` costs more than the rest of the
    fit for four points); lstsq's minimum-norm answer where the points are collinear."""
    dx, dy, dz = x - x.mean(), y - y.mean(), z - z.mean()
    sxx, syy, sxy = dx @ dx, dy @ dy, dx @ dy
    det = sxx * syy - sxy * sxy
    if det <= 1e-12 * max(sxx * syy, 1e-30):
        a = np.stack([x, y, np.ones_like(x)], axis=1)
        (ga, gb, _), *_ = np.linalg.lstsq(a, z, rcond=None)
        return float(ga), float(gb)
    sxz, syz = dx @ dz, dy @ dz
    return float((syy * sxz - sxy * syz) / det), float((sxx * syz - sxy * sxz) / det)


class MeshStream:
    """The detail's render mesh, no collision: a USD mesh per tile within `reach` m of a
    vehicle, from the tile's own samples (question 3: one road for wheels, cameras and lidar),
    removed again when no vehicle is near. Once a frame (`update`).

    A tile's mesh covers only the corridors' reach (`Detail.zones_tile`), so beyond it the
    world's own ground shows, not a level sheet of the detail's laid over it; its faces are
    two subsets, "road" (carriageway and shoulder) and "verge" (foreslope, ditch, backslope),
    each bound to `looks[name]` (a material's path) when given."""

    ROOT = "/World/RoadDetail"
    SUBSETS = ("road", "verge")

    def __init__(self, stage, layer, reach=40.0, looks=None):
        self.stage, self.layer, self.reach = stage, layer, reach
        self.looks = dict(looks or {})
        self.meshes = {}
        self.ms = []
        self._faces = None  # the tiles' quads, the same for every tile

    def faces(self):
        """Every quad of a tile's grid, (n * n, 4) vertex indices, row by row."""
        if self._faces is None:
            n = self.layer.detail.n
            a = (np.arange(n)[None, :] + (n + 1) * np.arange(n)[:, None]).ravel()
            self._faces = np.stack([a, a + 1, a + n + 2, a + n + 1], axis=1).astype(np.int32)
        return self._faces

    def tile_faces(self, zones):
        """The quads the tile's mesh keeps (every corner within a corridor's reach) and, of
        those, which are road (a corner on the carriageway or shoulder): (quads, road mask)."""
        q = self.faces()
        z = zones.ravel()[q]
        keep = (z > 0).all(axis=1)
        return q[keep], (z[keep] == rd.Detail.ROAD).any(axis=1)

    def update(self, points):
        import time

        from pxr import UsdGeom, Vt

        d = self.layer.detail
        want = {k for k in d.tiles_near([p[0] + self.layer.ox for p in points],
                                        [p[1] + self.layer.oy for p in points], self.reach)
                if d.touches(*k) and d.ready(k)} if points else set()
        for key in [k for k in self.meshes if k not in want]:
            self.stage.RemovePrim(self.meshes.pop(key))
        n, s, t = d.n, d.spacing, d.tile
        for key in want - set(self.meshes):
            t0 = time.perf_counter()
            h = d.surface_tile(*key)
            i, j = key
            quads, road = self.tile_faces(d.zones_tile(i, j))
            if not len(quads):
                continue
            gx, gy = np.meshgrid(i * t + np.arange(n + 1) * s - self.layer.ox,
                                 j * t + np.arange(n + 1) * s - self.layer.oy)
            pts = np.stack([gx, gy, h], axis=-1).reshape(-1, 3).astype(np.float32)
            path = f"{self.ROOT}/t_{i}_{j}".replace("-", "m")
            m = UsdGeom.Mesh.Define(self.stage, path)
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
            m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(quads.ravel()))
            m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(quads), 4, np.int32)))
            for name, mask in zip(self.SUBSETS, (road, ~road)):
                faces = np.flatnonzero(mask).astype(np.int32)
                if not len(faces):
                    continue
                sub = UsdGeom.Subset.CreateGeomSubset(
                    m, name, UsdGeom.Tokens.face, Vt.IntArray.FromNumpy(faces),
                    "materialBind", UsdGeom.Tokens.nonOverlapping)
                if name in self.looks:
                    from pxr import UsdShade

                    UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(
                        UsdShade.Material(self.stage.GetPrimAtPath(self.looks[name])))
            self.meshes[key] = path
            self.ms.append((time.perf_counter() - t0) * 1e3)

    def remove(self):
        for path in self.meshes.values():
            self.stage.RemovePrim(path)
        self.meshes.clear()


if __name__ == "__main__":
    sys.exit(main())
