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
"""
import dataclasses
import logging
import math

import numpy as np

from . import road_contact as rc
from . import road_detail as rd

log = logging.getLogger("swb_isaac_world")

PX_EXTRA = 0.10  # m of PhysX travel beyond the spec's (R2)
FILE = "road_detail.yaml"
REACH = 48.0  # m ahead of a vehicle whose tiles are made in the worker
LEAD_S = 4.0  # s: at speed, the tiles of the next 4 s (a busy host delays the worker)
# aib.isaac.vehicle.WHEELS' order (front?, side): front left, front right, rear left, rear right
ORDER = ((True, 1), (True, -1), (False, 1), (False, -1))
WHEELS = len(ORDER)


class RoadDetailError(ValueError):
    pass


def parse(data, where=FILE):
    """The corridors of a road_detail.yaml's contents; RoadDetailError lists every problem."""
    problems = []
    kinds = {"roughness": rd.Roughness, "section": rd.Section, "ruts": rd.Ruts,
             "potholes": rd.Potholes, "water_bars": rd.WaterBars}
    out = []
    unknown = set(data or {}) - {"corridors"}
    if unknown:
        problems.append(f"{where}: unknown key(s) {sorted(unknown)} (corridors)")
    for k, c in enumerate((data or {}).get("corridors") or []):
        at = f"{where}: corridors[{k}]"
        args = {}
        line = np.asarray(c.get("centreline") or [], dtype=float)
        if line.ndim != 2 or line.shape[1] not in (2, 3) or len(line) < 2:
            problems.append(f"{at}: centreline must be two or more [x, y] or [x, y, z]")
            continue
        bad = set(c) - {"centreline", "seed", "washboard", *kinds}
        if bad:
            problems.append(f"{at}: unknown key(s) {sorted(bad)}")
        for name, cls in kinds.items():
            if name not in c:
                continue
            fields = {f.name for f in dataclasses.fields(cls)}
            given = c[name] or {}
            if set(given) - fields:
                problems.append(f"{at}.{name}: unknown {sorted(set(given) - fields)} "
                                f"({', '.join(sorted(fields))})")
                continue
            if "at" in given:
                given = dict(given, at=tuple(given["at"]))
            args[name] = cls(**given)
        zones = []
        for z in c.get("washboard") or []:
            try:
                zones.append(rd.Washboard(**z))
            except TypeError as e:
                problems.append(f"{at}.washboard: {e}")
        if "roughness" in args and args["roughness"].cls not in rd.ISO_CLASS \
                and not args["roughness"].gd0:
            problems.append(f"{at}.roughness: class {args['roughness'].cls} is not ISO 8608's "
                            f"A-H")
        out.append(rd.Corridor(line, seed=int(c.get("seed", k)), washboard=tuple(zones), **args))
    if problems:
        raise RoadDetailError("\n".join(problems))
    return out


def load(path):
    import yaml

    with open(path) as f:
        return parse(yaml.safe_load(f) or {}, str(path))


def vehicle_params(p):
    """aib VehicleParams `p` with PhysX's travel PX_EXTRA longer (its frame rises with it)."""
    return dataclasses.replace(p, suspension_travel=p.suspension_travel + PX_EXTRA)


class RoadLayer:
    def __init__(self, corridors, origin=(0.0, 0.0, 0.0), apply_force=None, workers=1):
        """`origin`: the world frame's origin in the map's (`SurfaceLayer.origin`)."""
        self.detail = rd.Detail(corridors, workers=workers, max_tiles=96,
                                processes=bool(workers))
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
        return 1.0 if car is None or car.scale is None else float(car.scale[i])

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
                                   (sides[(False, 1)], sides[(False, -1)]), front[1], rear[1]))
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
            (ga, gb, _), *_ = np.linalg.lstsq(a, z, rcond=None)
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

        def height(x, y):
            return base(x, y) + layer.height(x, y)

        o = self.contact.step(dt, height, base, xy, heading, speed, z_att, v_att)
        self.scale = self.contact.friction_scale(o["load"], o["px_load"])
        self.last_out = o
        p = self.v.p
        out = []
        for i in range(WHEELS):
            cz = pz + att[i][2] - self.frame_z + p.wheel_radius
            out.append(((float(o["dfx"][i] * heading[i, 0]), float(o["dfx"][i] * heading[i, 1]),
                         float(o["dfz"][i])), (float(xy[i, 0]), float(xy[i, 1]), cz)))
        return out


class MeshStream:
    """The detail's render mesh, no collision: a USD mesh per tile within `reach` m of a
    vehicle, from the tile's own samples (question 3: one road for wheels, cameras and lidar),
    removed again when no vehicle is near. Once a frame (`update`)."""

    ROOT = "/World/RoadDetail"

    def __init__(self, stage, layer, reach=40.0):
        self.stage, self.layer, self.reach = stage, layer, reach
        self.meshes = {}
        self.ms = []
        self._faces = None  # the tiles' quads, the same for every tile

    def faces(self):
        if self._faces is None:
            from pxr import Vt

            n = self.layer.detail.n
            a = (np.arange(n)[None, :] + (n + 1) * np.arange(n)[:, None]).ravel()
            idx = np.stack([a, a + 1, a + n + 2, a + n + 1], axis=1).ravel().astype(np.int32)
            self._faces = (Vt.IntArray.FromNumpy(idx),
                           Vt.IntArray.FromNumpy(np.full(n * n, 4, np.int32)))
        return self._faces

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
            gx, gy = np.meshgrid(i * t + np.arange(n + 1) * s - self.layer.ox,
                                 j * t + np.arange(n + 1) * s - self.layer.oy)
            pts = np.stack([gx, gy, h], axis=-1).reshape(-1, 3).astype(np.float32)
            path = f"{self.ROOT}/t_{i}_{j}".replace("-", "m")
            m = UsdGeom.Mesh.Define(self.stage, path)
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
            idx, counts = self.faces()
            m.CreateFaceVertexIndicesAttr(idx)
            m.CreateFaceVertexCountsAttr(counts)
            self.meshes[key] = path
            self.ms.append((time.perf_counter() - t0) * 1e3)

    def remove(self):
        for path in self.meshes.values():
            self.stage.RemovePrim(path)
        self.meshes.clear()
