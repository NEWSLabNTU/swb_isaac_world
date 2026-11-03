"""Rocks on the road in the world server (roadmap 066 R4): the USD and PhysX side of
`rocks`. Import after `SimulationApp`.

`RoadLayer.rocks` (a `rocks.RockSet`, world frame) is what the tire reads; this streams the
rest with the vehicles, once a frame (`update`):

- collision tiles of the corridors' detail at 10 cm (base + detail, a triangle mesh each)
  within COLLISION_REACH of a vehicle, in a group that rocks and the chassis collide with and
  the wheels' ground query filters out;
- prims for the rocks of the tiles within ROCK_REACH that have a collision tile, placed in the
  tile worker (`rocks.place`, the tile's extras): a tile's static rocks one merged collider,
  each loose rock its own rigid body, made at most PRIMS_PER_FRAME a frame;
- a loose rock asleep (rigidBodyEnabled off: a static collider, no pose sync) until a vehicle
  can touch it (within ACTIVE_SIDE of its line, ACTIVE_BEHIND behind to ACTIVE_S of travel
  ahead), and asleep again where it lies; a tile leaving the window writes its loose rocks' poses back.

The stage's cost grows with its prims, not their triangles (8531 rock prims took 16 ms a frame
in rock_field.py; the window holds about 200). Each streamed collider is put in its collision
group by its own path: a prim made after the simulation started is not in a group through
its parent's (issue 030). Rocks collide with the world's ground as they do with everything:
a loose one is placed only where the detail is no more than rocks.LOOSE_BELOW under the base
(`rocks.place`), so it rests where the surface is.
"""
import math
import time

import numpy as np

from . import rocks as rk

COLLISION_REACH = 32.0  # m around a vehicle with collision tiles
COLLISION_STEP = 2  # every second 5 cm sample: 10 cm (p99 2.5 mm from the tire's height)
ROCK_REACH = 32.0  # m around a vehicle with rock prims (1.4 s at 80 km/h)
# awake only in a vehicle's path: within ACTIVE_SIDE of its line (half its width and a
# margin), from ACTIVE_BEHIND behind to ACTIVE_S of travel (at least ACTIVE_MIN) ahead of its
# front; each awake rock costs PhysX and the contact (about 30 awake took 6.6 + 2.3 ms a
# frame at 80 km/h)
ACTIVE_S, ACTIVE_MIN, ACTIVE_BEHIND, ACTIVE_SIDE = 0.6, 8.0, 3.0, 1.5
FRONT = 3.0  # m from a vehicle's origin (its rear axle) to its front, about
PRIMS_PER_FRAME = 4  # rock prims made a frame at most: a tile's at once took 11 ms
MAX_DEPENETRATION = 1.0  # m/s
SPECULATIVE_CCD = True
CONTACT_OFFSET = None  # m, PhysX's own when None
SOLVER_POSITION_ITERATIONS = None  # the scene's when None
PARK_Z = -1000.0  # m: where a parked rock prim waits for its next rock
# m a frame: a rock leaving the car's path sleeps only once it moved less than this since the
# last frame (0.2 m/s at 20 Hz). A rock kicked into the air is still flying when the car has
# passed, and a static collider made of it there hangs in the air, a step the next tire meets.
SETTLED = 0.01
# a loose rock has its own rigid body only within this of the carriageway's and shoulder's
# edge, where a car on the road reaches it; beyond (ditch, backslope) it is part of the
# tile's static collider: a car out there has left the road (about 70 % of the rocks lie
# there, and each body is a prim: 543 of them took 7 ms a frame)
BAND_MARGIN = 0.5  # m
ROOT = "/World/RoadRocks"
COLL_ROOT = "/World/RoadCollision"


class RockStream:
    def __init__(self, stage, layer, physx, ground=(), groups="/World/vehicleCollisionGroups"):
        """`ground`: the world's own road ground (the base's prims, `road_detail.yaml`'s), which
        the rocks then do not collide with: a loose rock rests on the collision tiles, the
        true surface, not on the base above a rut or a crowned road's sides."""
        from pxr import Sdf, UsdGeom, UsdPhysics, Vt

        self.stage, self.layer, self.physx = stage, layer, physx
        self.rock_g, self.detail_g = f"{groups}/rocks", f"{groups}/roadDetail"
        base_g = f"{groups}/roadBase"
        for g in (self.rock_g, self.detail_g):
            UsdPhysics.CollisionGroup.Define(stage, g).CreateFilteredGroupsRel().AddTarget(
                f"{groups}/groundQuery")
            UsdPhysics.CollisionGroup.Get(stage, g).GetCollidersCollectionAPI() \
                .CreateIncludesRel()
        if ground:
            UsdPhysics.CollisionGroup.Define(stage, base_g).GetCollidersCollectionAPI() \
                .CreateIncludesRel().SetTargets([Sdf.Path(p) for p in ground])
            UsdPhysics.CollisionGroup.Get(stage, self.rock_g).GetFilteredGroupsRel() \
                .AddTarget(base_g)
        mat = UsdPhysics.MaterialAPI.Apply(UsdGeom.Scope.Define(stage, f"{ROOT}/material")
                                           .GetPrim())
        mat.CreateStaticFrictionAttr(0.7)  # rock on gravel and on steel, about 0.6-0.8
        mat.CreateDynamicFrictionAttr(0.6)
        mat.CreateRestitutionAttr(0.1)
        d = layer.detail
        self.n_c = d.n // COLLISION_STEP
        a = (np.arange(self.n_c)[None, :] + (self.n_c + 1) * np.arange(self.n_c)[:, None]).ravel()
        self.coll_idx = Vt.IntArray.FromNumpy(
            np.stack([a, a + 1, a + self.n_c + 2, a + self.n_c + 1], axis=1).ravel()
            .astype(np.int32))
        self.coll_cnt = Vt.IntArray.FromNumpy(np.full(self.n_c * self.n_c, 4, np.int32))
        self.coll = {}  # tile -> prim path
        self.loaded = {}  # tile -> its rocks' indices in layer.rocks
        self.win = {}  # tile -> its static rocks' prim path (or None)
        self.paths, self.dyn, self.mass = [], [], []  # per rock
        self.ints = {}
        self.active = {}  # rock -> awake
        self.settling = {}  # rock leaving the path -> its position at the last frame
        self.vel = {}  # rock -> its velocity over the last read, world
        self.read_at = {}  # rock -> the read count of its last read
        self.reads = 0
        self.queue = []
        self.pool = {}  # prototype -> parked prims of it, for the next rock of that shape
        self.serial = 0
        self.made, self.reused = 0, 0
        self.coll_ms, self.prim_ms = [], []

    # --- the stage ---------------------------------------------------------------------------

    def _group(self, group, path, on):
        from pxr import UsdPhysics

        rel = UsdPhysics.CollisionGroup.Get(self.stage, group).GetCollidersCollectionAPI() \
            .GetIncludesRel()
        (rel.AddTarget if on else rel.RemoveTarget)(path)

    def _mesh(self, path, pts, idx, cnt, collision):
        from pxr import UsdGeom, UsdPhysics, Vt

        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateFaceVertexCountsAttr(cnt)
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(collision)
        return m

    def update(self, points):
        """Once a frame: `points` the vehicles' (x, y, yaw[, speed m/s]) in the world."""
        d, ox, oy = self.layer.detail, self.layer.ox, self.layer.oy
        xs = [p[0] + ox for p in points]
        ys = [p[1] + oy for p in points]
        # collision tiles
        want = {k for k in d.tiles_near(xs, ys, COLLISION_REACH)
                if d.touches(*k) and d.ready(k)} if points else set()
        for k in [k for k in self.coll if k not in want]:
            path = self.coll.pop(k)
            self.stage.RemovePrim(path)
            self._group(self.detail_g, path, False)
        for k in want - set(self.coll):
            t0 = time.perf_counter()
            from pxr import UsdGeom

            i, j = k
            h = d.surface_tile(i, j)[::COLLISION_STEP, ::COLLISION_STEP]
            s = d.spacing * COLLISION_STEP
            gx, gy = np.meshgrid(i * d.tile + np.arange(self.n_c + 1) * s - ox,
                                 j * d.tile + np.arange(self.n_c + 1) * s - oy)
            path = f"{COLL_ROOT}/t_{i}_{j}".replace("-", "m")
            m = self._mesh(path, np.stack([gx, gy, h], axis=-1).reshape(-1, 3), self.coll_idx,
                           self.coll_cnt, "none")
            m.CreateVisibilityAttr(UsdGeom.Tokens.invisible)  # MeshStream's is the render
            self._group(self.detail_g, path, True)
            self.coll[k] = path
            self.coll_ms.append((time.perf_counter() - t0) * 1e3)
        self._stream_rocks(xs, ys, points)
        self._activate(points)

    def _stream_rocks(self, xs, ys, points):
        from pxr import Vt

        d = self.layer.detail
        want = {k for k in d.tiles_near(xs, ys, ROCK_REACH)
                if k in self.coll and d.extra(*k) is not None} if points else set()
        rs = self.layer.rocks
        for k in [k for k in self.win if k not in want]:
            for i in self.loaded[k]:
                if self.paths[i] is not None:
                    if self.active.get(i):
                        self.read(i)  # write-back: the data keeps where it lies
                        self.active[i] = False
                        self.settling.pop(i, None)
                    self._release(i)
            if self.win[k]:
                self.stage.RemovePrim(self.win[k])
                self._group(self.rock_g, self.win[k], False)
            del self.win[k]
        for k in want - set(self.win):
            t0 = time.perf_counter()
            if k not in self.loaded:
                self._load(k)
            ids = self.loaded[k]
            statics = [(rs.rocks[i], rs.pos[i], _yaw(rs.quat[i]), False)
                       for i in ids if not self.dyn[i]]
            path = None
            if statics:
                v, f = rk.merged(statics)
                path = f"{ROOT}/s_{k[0]}_{k[1]}".replace("-", "m")
                self._mesh(path, v, Vt.IntArray.FromNumpy(f.ravel().astype(np.int32)),
                           Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)), "none")
                self._group(self.rock_g, path, True)
            self.win[k] = path
            self.queue.extend(i for i in ids if self.dyn[i])
            self.prim_ms.append((time.perf_counter() - t0) * 1e3)
        made = 0
        while self.queue and made < PRIMS_PER_FRAME:
            i = self.queue.pop(0)
            if self.paths[i] is None and any(i in self.loaded[k] for k in self.win):
                self._rock_prim(i)
                made += 1

    def _in_band(self, p):
        """Whether map point p lies on a corridor's carriageway or shoulder (+ BAND_MARGIN)."""
        for c in self.layer.detail.corridors:
            _, v, near = c.uv(np.array([p[0]]), np.array([p[1]]))
            if near[0] and abs(v[0]) <= c.section.half_width + c.section.shoulder + BAND_MARGIN:
                return True
        return False

    def _load(self, k):
        """The tile's rocks into the data, once, in the world frame; a loose one beyond the
        band a car on the road reaches is held, in the tile's static collider."""
        entries = [(r, p, yaw, dyn and self._in_band(p))
                   for r, p, yaw, dyn in (self.layer.detail.extra(*k) or [])]
        off = np.array([self.layer.ox, self.layer.oy, 0.0])
        n0 = len(self.paths)
        self.layer.rocks.add([r for r, *_ in entries], [p - off for _, p, _, _ in entries],
                             [rk._yaw_quat(yaw) for _, _, yaw, _ in entries])
        self.paths.extend([None] * len(entries))
        self.dyn.extend([bool(dy) for *_, dy in entries])
        self.mass.extend([r.mass for r, *_ in entries])
        self.loaded[k] = list(range(n0, n0 + len(entries)))

    def _release(self, i):
        """Rock i's prim off the road: a prototype's parked for the next rock of its shape
        (collision off, out of the way), others removed. A prim made is a USD change the
        stage takes in with all its population; moving a parked one is three attributes."""
        from pxr import Gf, UsdGeom, UsdPhysics

        path, r = self.paths[i], self.layer.rocks.rocks[i]
        self.paths[i] = None
        if r.proto is None:
            self.stage.RemovePrim(path)
            self._group(self.rock_g, path, False)
            return
        prim = self.stage.GetPrimAtPath(path)
        UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)  # a falling one
        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        UsdGeom.Imageable(prim).MakeInvisible()
        UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(Gf.Vec3d(0.0, 0.0, PARK_Z))
        self.pool.setdefault(r.proto, []).append(path)

    def _rock_prim(self, i):
        """Loose rock i's own prim, a rigid body asleep (static) at its present pose: a parked
        one of its prototype if there is one, else a new one."""
        from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics, Vt

        rs = self.layer.rocks
        r = rs.rocks[i]
        free = self.pool.get(r.proto) if r.proto is not None else None
        if free:
            path = free.pop()
            prim = self.stage.GetPrimAtPath(path)
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
            ops[0].Set(Gf.Vec3d(*map(float, rs.pos[i])))
            ops[1].Set(Gf.Quatf(*map(float, rs.quat[i])))
            ops[2].Set(Gf.Vec3f(float(r.scale), float(r.scale), float(r.scale)))
            UsdGeom.Imageable(prim).MakeVisible()
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(True)
            self.paths[i] = path
            self.ints[i] = PhysicsSchemaTools.sdfPathToInt(path)
            self.reused += 1
            return
        self.serial += 1
        self.made += 1
        path = f"{ROOT}/r{self.serial}"
        # a prototype's rock: its points unscaled and the size as the prim's scale, so PhysX
        # cooks the shape once for every rock of it
        pts = rk.prototypes()[r.proto].verts if r.proto is not None else r.verts
        m = self._mesh(path, pts, Vt.IntArray.FromNumpy(r.faces.ravel().astype(np.int32)),
                       Vt.IntArray.FromNumpy(np.full(len(r.faces), 3, np.int32)),
                       "convexHull")
        m.AddTranslateOp().Set(Gf.Vec3d(*map(float, rs.pos[i])))
        m.AddOrientOp().Set(Gf.Quatf(*map(float, rs.quat[i])))
        if r.proto is not None:
            m.AddScaleOp().Set(Gf.Vec3f(float(r.scale), float(r.scale), float(r.scale)))
        UsdPhysics.CollisionAPI(m.GetPrim()).CreateCollisionEnabledAttr(True)
        rb = UsdPhysics.RigidBodyAPI.Apply(m.GetPrim())
        rb.CreateRigidBodyEnabledAttr(False)  # asleep: a static collider, no pose sync
        UsdPhysics.MassAPI.Apply(m.GetPrim()).CreateDensityAttr(rk.DENSITY)
        prb = PhysxSchema.PhysxRigidBodyAPI.Apply(m.GetPrim())
        prb.CreateEnableSpeculativeCCDAttr(SPECULATIVE_CCD)
        if SOLVER_POSITION_ITERATIONS is not None:
            prb.CreateSolverPositionIterationCountAttr(SOLVER_POSITION_ITERATIONS)
        if CONTACT_OFFSET is not None:
            col = PhysxSchema.PhysxCollisionAPI.Apply(m.GetPrim())
            col.CreateContactOffsetAttr(CONTACT_OFFSET)
            col.CreateRestOffsetAttr(0.0)
        # out of an overlap (the surface's mm, a neighbour) at a walk, not PhysX's default
        # (a stone thrown at tens of m/s as it woke)
        prb.CreateMaxDepenetrationVelocityAttr(MAX_DEPENETRATION)
        self._group(self.rock_g, path, True)
        self.paths[i] = path
        self.ints[i] = PhysicsSchemaTools.sdfPathToInt(path)

    # --- awake and asleep --------------------------------------------------------------------

    def _sleep(self, i, on):
        from pxr import Gf, UsdGeom, UsdPhysics

        prim = self.stage.GetPrimAtPath(self.paths[i])
        if on:  # where it lies, then static
            rs = self.layer.rocks
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
            ops[0].Set(Gf.Vec3d(*map(float, rs.pos[i])))
            ops[1].Set(Gf.Quatf(*map(float, rs.quat[i])))
        UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(not on)

    def _activate(self, points):
        """Each rock with a prim awake while a vehicle can touch it: one array pass over the
        window's rocks, a USD write only for those that change."""
        rs = self.layer.rocks
        ids = np.array([i for k in self.win for i in self.loaded[k] if self.paths[i] is not None],
                       dtype=int)
        if not ids.size:
            return
        near = np.zeros(ids.size, dtype=bool)
        px, py = rs.pos[ids, 0], rs.pos[ids, 1]
        for pt in points:
            x, y, yaw = pt[:3]
            ahead = max(ACTIVE_MIN, ACTIVE_S * abs(pt[3])) if len(pt) > 3 else ACTIVE_MIN
            c, s = math.cos(yaw), math.sin(yaw)
            along = (px - x) * c + (py - y) * s
            across = -(px - x) * s + (py - y) * c
            near |= (along > -ACTIVE_BEHIND) & (along < FRONT + ahead) \
                & (np.abs(across) < ACTIVE_SIDE)
        for i, n in zip(ids.tolist(), near.tolist()):
            a = self.active.get(i, False)
            if n and not a:
                self._sleep(i, False)
                self.active[i] = True
            elif n:
                self.settling.pop(i, None)
            elif a and self._settled(i):
                self._sleep(i, True)
                self.active[i] = False

    def _settled(self, i):
        """Whether awake rock i has come to rest (PhysX's pose, read now, within SETTLED of the
        last frame's), so it may sleep where it lies."""
        if not self.read(i):
            return False
        pos = self.layer.rocks.pos[i].copy()
        last = self.settling.get(i)
        self.settling[i] = pos
        if last is None or np.linalg.norm(pos - last) > SETTLED:
            return False
        del self.settling[i]
        return True

    def awake(self):
        return sum(1 for v in self.active.values() if v)

    # --- PhysX ---------------------------------------------------------------------------------

    def read(self, i, dt=None):
        """Rock i's pose from PhysX into the data (and its velocity over dt, when its last read
        was the step before; otherwise none yet)."""
        t = self.physx.get_rigidbody_transformation(self.paths[i])
        if not t or "rotation" not in t:  # not in PhysX yet (just made or woken)
            self.vel[i] = np.zeros(3)
            return False
        x, y, z, w = t["rotation"]  # PhysX's (x, y, z, w)
        pos = np.array(tuple(t["position"]))
        if dt:
            if self.read_at.get(i) == self.reads - 1:
                self.vel[i] = (pos - self.layer.rocks.pos[i]) / dt
            else:
                self.vel[i] = np.zeros(3)
            self.read_at[i] = self.reads
        self.layer.rocks.set_pose(i, tuple(pos), (w, x, y, z))
        return True

    def read_near(self, xy, road, dt):
        """Before a vehicle's contact: the awake rocks near its wheels from PhysX, and whether
        each rests on the road (one off the ground holds no tire up)."""
        rs = self.layer.rocks
        rs.seen[:] = True
        self.reads += 1
        ids = rs.near(xy[:, 0], xy[:, 1], margin=1.0)
        for i in ids:
            if self.active.get(i):
                self.read(i, dt)
        rs.rest_many(ids, road)  # asleep too: one frozen off the road must hold no tire up

    def push(self, forces, speed, radius, road, dt, apply_force):
        """The tires' reactions on the awake rocks (`rocks.reactions`), each no faster than the
        tire surface pushing it; returns the scale the car's lateral is to take."""
        sc_all = 1.0
        for i, fs in forces.items():
            if not self.active.get(i):
                continue
            zc = max(p[2] - float(np.asarray(road(np.array([p[0]]), np.array([p[1]])))[0])
                     for _, p in fs)
            fs, sc = rk.limit(fs, self.mass[i], self.vel.get(i, np.zeros(3)),
                              abs(speed) * max(zc, 0.0) / radius, dt)
            sc_all = min(sc_all, sc)
            for f, p in fs:
                apply_force(self.paths[i], tuple(float(v) for v in f),
                            tuple(float(v) for v in p))
        return sc_all

    def remove(self):
        for path in list(self.coll.values()) + [p for p in self.paths if p] + \
                [p for p in self.win.values() if p] + [p for v in self.pool.values() for p in v]:
            self.stage.RemovePrim(path)
        self.pool.clear()
        self.coll.clear()
        self.win.clear()
        self.paths = [None] * len(self.paths)
        self.active.clear()
        self.settling.clear()


def _yaw(q):
    """The heading of a (w, x, y, z) quaternion that turns about z only, rad."""
    return 2.0 * math.atan2(float(q[3]), float(q[0]))
