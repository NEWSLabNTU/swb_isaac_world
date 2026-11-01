"""Roadmap 066 R4, second slice: aib's Model 3 over 2 km of R3's class C corridor with its
rocks placed (`rocks.place`, Golombek-Rapp by zone), Isaac at 60 Hz, the tire contact ours.
The cost of rocks at scale.

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/rock_field.py \\
        [<out dir>]
    # prints one SPIKE-RESULT json

- Rocks are placed per tile in the tile worker (`rocks.place` as the tile's extras), and have
  prims only for the tiles near the car (ROCK_REACH) that have a collision tile under them:
  a tile's static rocks one merged triangle-mesh collider, each loose rock its own rigid
  body, asleep (rigidBodyEnabled off: a static collider, no pose sync) until a car can touch
  it (ACTIVE_*), asleep again where it lies once out of reach. A tile leaving the window
  writes its loose rocks' poses back into the data and drops its prims. The stage's cost
  grows with its prims: 8531 rock prims took 16 ms a frame.
- Collision tiles: R3's detail around the car at 10 cm, a triangle mesh each, in a group
  that rocks and the chassis collide with and the wheels' query does not. Rocks do not
  collide with the base road, so a loose rock rests in a rut, not on the base above it.
- The left wheel track has a loose rock every ROCK_TRACK_EVERY m (fallen from the cut slope;
  the carriageway's own Golombek-Rapp cover places almost none of driveable size), so the
  run exercises the contact at scale.
- A skid plate at Tesla's 138 mm ground clearance, as in rock_fixtures.py.
Per speed: frame wall time against the same road without rocks (ROCKS=0), awake rocks, rock
prims, streaming costs, rocks under the tires, rocks lost (fallen through), and where the car
left the road or stuck, as outcomes. ROCK_FABRIC, ROCK_USD (0/1) set PhysX's Fabric and USD
write-back; ROCK_FRICTION=0 drops the per-step friction writes; ROCK_CONTACTS=1 reports the
car's contacts; ROCK_FROM starts the run elsewhere than X_START.
"""
import json
import math
import os
import sys
import time
from types import SimpleNamespace

os.environ.pop("DISPLAY", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdPhysics, Vt  # noqa: E402

from aib.control import GEAR_DRIVE, Command, State  # noqa: E402
from aib.frames import qrot  # noqa: E402
from aib.isaac.vehicle import GROUPS, WHEELS, Vehicle, ensure_vehicle_scene  # noqa: E402
from aib.rig import EgoConfig  # noqa: E402
from swb_isaac_world import road_contact as rc  # noqa: E402
from swb_isaac_world import road_detail as rd  # noqa: E402
from swb_isaac_world import road_layer  # noqa: E402
from swb_isaac_world import rocks as rk  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/rock_field", time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(OUT, exist_ok=True)
SPEEDS = [float(v) for v in os.environ.get("ROCK_SPEEDS", "40").split(",")]
ROCKS_ON = os.environ.get("ROCKS", "1") == "1"
LENGTH = float(os.environ.get("ROCK_LENGTH", "2000"))
X_START = 20.0
RUN_FROM = float(os.environ.get("ROCK_FROM", X_START))
FRAME, SUB = 0.05, 3
DT = FRAME / SUB
W_LAT, Z_LAT = 0.8, 0.9
# awake: the loose rocks a car can touch, within ACTIVE_SIDE of its track (half its width and
# a wheel's), ACTIVE_AHEAD (1.4 s at 80 km/h) to ACTIVE_BEHIND; asleep elsewhere
ACTIVE_AHEAD, ACTIVE_BEHIND, ACTIVE_SIDE = 30.0, 5.0, 2.5
PRIMS_PER_FRAME = 4  # rock prims made a frame at most: a tile's at once took 11 ms
COLLISION_REACH = 48.0  # m around the car with collision tiles
COLLISION_STEP = 2  # every second 5 cm sample: 10 cm
ROCK_TRACK_EVERY = 100.0
ROCK_REACH = float(os.environ.get("ROCK_REACH", "48"))  # m around the car with rock prims
CLEARANCE = 0.138
CONTACTS_ON = os.environ.get("ROCK_CONTACTS") == "1"
FRICTION_WRITES = os.environ.get("ROCK_FRICTION", "1") == "1"

import carb  # noqa: E402

for key, env in (("/physics/fabricEnabled", "ROCK_FABRIC"), ("/physics/updateToUsd", "ROCK_USD")):
    if env in os.environ:
        carb.settings.get_settings().set_bool(key, os.environ[env] == "1")
world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
x0, x1 = X_START - 100.0, X_START + LENGTH + 200.0
base = UsdGeom.Mesh.Define(stage, "/World/env/Base/mesh")
base.CreatePointsAttr([Gf.Vec3f(x0, -20, 0), Gf.Vec3f(x1, -20, 0), Gf.Vec3f(x1, 20, 0),
                       Gf.Vec3f(x0, 20, 0)])
base.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
base.CreateFaceVertexCountsAttr([4])
UsdPhysics.CollisionAPI.Apply(base.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(base.GetPrim()).CreateApproximationAttr("none")
if not any(pr.IsA(UsdPhysics.Scene) for pr in stage.Traverse()):
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
ensure_vehicle_scene(stage)
ROCK_G, DETAIL_G, BASE_G = f"{GROUPS}/rocks", f"{GROUPS}/roadDetail", f"{GROUPS}/base"
rock_g = UsdPhysics.CollisionGroup.Define(stage, ROCK_G)
rock_g.CreateFilteredGroupsRel().SetTargets([Sdf.Path(f"{GROUPS}/groundQuery"),
                                             Sdf.Path(BASE_G)])
UsdPhysics.CollisionGroup.Define(stage, DETAIL_G).CreateFilteredGroupsRel().AddTarget(
    f"{GROUPS}/groundQuery")
UsdPhysics.CollisionGroup.Define(stage, BASE_G).GetCollidersCollectionAPI() \
    .CreateIncludesRel().AddTarget(base.GetPath())
mat = UsdPhysics.MaterialAPI.Apply(UsdGeom.Scope.Define(stage, "/World/rockMaterial").GetPrim())
mat.CreateStaticFrictionAttr(0.7)
mat.CreateDynamicFrictionAttr(0.6)
mat.CreateRestitutionAttr(0.1)

corridor = rd.Corridor(
    np.array([[0.0, 0.0], [x1, 0.0]]), seed=66, roughness=rd.Roughness(cls="C"),
    ruts=rd.Ruts(offset=0.83), potholes=rd.Potholes(),
    washboard=(rd.Washboard(X_START + 600.0, X_START + 700.0, pitch=0.5),),
    water_bars=rd.WaterBars(at=(X_START + 1500.0, X_START + 1550.0, X_START + 1600.0)))
detail = rd.Detail([corridor], max_tiles=96, workers=1, processes=True,
                   extras="swb_isaac_world.rocks:place" if ROCKS_ON else None)

# --- the rocks: placed per tile in the worker, prims only near the car ---------------------
TRACK = {}  # tile -> loose rocks in the left wheel track (fallen from the cut slope)
if ROCKS_ON:
    rng = np.random.default_rng(9)
    for xr in np.arange(X_START + 50.0, X_START + LENGTH, ROCK_TRACK_EVERY):
        r = rk.make_rock(rng, rng.uniform(0.15, 0.30), rough=0.10)
        y = 0.83 + rng.normal(0.0, 0.08)
        TRACK.setdefault((int(xr // detail.tile), int(y // detail.tile)), []).append(
            (r, xr, y))
rockset = rk.RockSet([], np.zeros((0, 3)), np.zeros((0, 4)))
PATHS, DYN, MASS = [], [], []  # per rock: its prim (None: none now), loose, kg
LOADED = {}  # tile -> its rocks' indices (data kept once loaded; prims come and go)
WIN = {}  # tile -> its static prim's path (or None), with prims now
ROCK_INT = {}
place_ms, prim_ms, queue_ms = [], [], []
QUEUE = []  # loose rocks whose prim is to be made
placed = ROCKS_ON


def load_tile(k):
    """The tile's rocks into the data (once): its placement and its track rocks."""
    entries = list(detail.extra(*k) or [])
    for r, xr, y in TRACK.get(k, []):
        z = float(detail.height(np.array([xr]), np.array([y]))[0]) - r.verts[:, 2].min()
        entries.append((r, np.array([xr, y, z]), 0.0, True))
    n0 = len(PATHS)
    rockset.add([r for r, *_ in entries], [p for _, p, _, _ in entries],
                [(1.0, 0.0, 0.0, 0.0)] * len(entries))
    PATHS.extend([None] * len(entries))
    DYN.extend([bool(d) for *_, d in entries])
    MASS.extend([r.mass for r, *_ in entries])
    LOADED[k] = list(range(n0, n0 + len(entries)))


def rock_prim(i):
    """Rock i's own prim, a rigid body asleep (static) at its present pose."""
    r = rockset.rocks[i]
    path = f"/World/rocks/r{i}"
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(r.verts.astype(np.float32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(r.faces.ravel().astype(np.int32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(r.faces), 3, np.int32)))
    q = rockset.quat[i]
    m.AddTranslateOp().Set(Gf.Vec3d(*map(float, rockset.pos[i])))
    m.AddOrientOp().Set(Gf.Quatf(*map(float, q)))
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    rb = UsdPhysics.RigidBodyAPI.Apply(m.GetPrim())
    rb.CreateRigidBodyEnabledAttr(False)  # asleep: a static collider, no pose sync
    UsdPhysics.MassAPI.Apply(m.GetPrim()).CreateDensityAttr(rk.DENSITY)
    PhysxSchema.PhysxRigidBodyAPI.Apply(m.GetPrim()).CreateEnableSpeculativeCCDAttr(True)
    group(path, True)
    PATHS[i] = path
    ROCK_INT[i] = PhysicsSchemaTools.sdfPathToInt(path)


def group(path, on):
    """The prim in the rocks' collision group, by its own path: a prim made after the
    simulation started is not in a group through its parent's (the wheels' query hit such
    rocks and threw the car, issue 029's catapult)."""
    rel = UsdPhysics.CollisionGroup.Get(stage, ROCK_G).GetCollidersCollectionAPI() \
        .GetIncludesRel()
    if on:
        rel.AddTarget(path)
    else:
        rel.RemoveTarget(path)


def stream_rocks(x):
    """Prims for the rocks of the tiles near the car that have a collision tile under them;
    the loose ones queued, PRIMS_PER_FRAME a frame."""
    if not ROCKS_ON:
        return
    want = {k for k in detail.tiles_near([x], [0.0], ROCK_REACH)
            if k in COLL and detail.extra(*k) is not None}
    for k in [k for k in WIN if k not in want]:
        for i in LOADED[k]:
            if PATHS[i] is not None:
                if ACTIVE.get(i):
                    read_rock(i)  # write-back: the data keeps where it lies
                    ACTIVE[i] = False
                stage.RemovePrim(PATHS[i])
                group(PATHS[i], False)
                PATHS[i] = None
        if WIN[k]:
            stage.RemovePrim(WIN[k])
            group(WIN[k], False)
        del WIN[k]
    for k in want - set(WIN):
        t0 = time.perf_counter()
        if k not in LOADED:
            load_tile(k)
        t1 = time.perf_counter()
        ids = LOADED[k]
        statics = [(rockset.rocks[i], rockset.pos[i], 0.0, False) for i in ids if not DYN[i]]
        path = None
        if statics:
            v, f = rk.merged(statics)
            path = f"/World/rocks/s_{k[0]}_{k[1]}".replace("-", "m")
            m = UsdGeom.Mesh.Define(stage, path)
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
            m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel().astype(np.int32)))
            m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("none")
            group(path, True)
        WIN[k] = path
        QUEUE.extend(i for i in ids if DYN[i])
        place_ms.append((t1 - t0) * 1e3)
        prim_ms.append((time.perf_counter() - t1) * 1e3)
    t0 = time.perf_counter()
    made = 0
    while QUEUE and made < PRIMS_PER_FRAME:
        i = QUEUE.pop(0)
        if PATHS[i] is None and any(i in LOADED[k] for k in WIN):
            rock_prim(i)
            made += 1
    if made:
        queue_ms.append((time.perf_counter() - t0) * 1e3)


UsdGeom.Scope.Define(stage, "/World/rocks")
UsdPhysics.CollisionGroup.Get(stage, ROCK_G).GetCollidersCollectionAPI().CreateIncludesRel()

cfg = EgoConfig.load(os.environ.get("AIB_EGO") or os.path.join(
    HERE, "../deps/autoware_isaac_bridge/src/aib/config/ego.yaml"))
rig = cfg.load_rig()
vp = rig.vehicle
REAL_TRAVEL = vp.suspension_travel
import dataclasses  # noqa: E402

vp = dataclasses.replace(vp, suspension_travel=REAL_TRAVEL + 0.10)
veh = Vehicle(stage, "/World/ego", vp, rig.ecu, rig.box_center, rig.box_dims,
              ((X_START, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0)))
PhysxSchema.PhysxVehicleAPI(stage.GetPrimAtPath("/World/ego")).GetSuspensionLineQueryTypeAttr() \
    .Set(PhysxSchema.Tokens.raycast)
plate = UsdGeom.Cube.Define(stage, "/World/ego/skid_plate")
plate.CreateSizeAttr(1.0)
UsdGeom.XformCommonAPI(plate.GetPrim()).SetTranslate(
    Gf.Vec3d(vp.wheel_base / 2, 0.0, CLEARANCE + 0.02))
UsdGeom.XformCommonAPI(plate.GetPrim()).SetScale(Gf.Vec3f(vp.wheel_base - 0.7, 1.30, 0.04))
UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
if CONTACTS_ON:  # every contact of the car's body, reported
    PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath("/World/ego")) \
        .CreateThresholdAttr(0.0)
UsdPhysics.CollisionGroup.Get(stage, f"{GROUPS}/chassis").GetCollidersCollectionAPI() \
    .CreateIncludesRel().AddTarget(plate.GetPath())
(front_axle, rear_axle), sag = vp.suspension()
FRAME_Z = vp.wheel_radius + vp.suspension_travel - sag
ATTACH = [((vp.wheel_base if front else 0.0), side * vp.wheel_tread / 2)
          for _, front, side in WHEELS]
SIDES = {(front, side): i for i, (_, front, side) in enumerate(WHEELS)}
contact = rc.TireContact(
    [rc.Corner(*(front_axle if front else rear_axle), REAL_TRAVEL, vp.wheel_radius,
               vp.wheel_width, px_travel=vp.suspension_travel) for _, front, _ in WHEELS],
    bars=rc.anti_roll_bars((SIDES[(True, 1)], SIDES[(True, -1)]),
                           (SIDES[(False, 1)], SIDES[(False, -1)]),
                           front_axle[1], rear_axle[1]))
meshes = road_layer.MeshStream(stage, SimpleNamespace(detail=detail, ox=0.0, oy=0.0),
                               reach=40.0)

sim = omni.physx.get_physx_simulation_interface()
STAGE_ID = omni.usd.get_context().get_stage_id()
BODY = PhysicsSchemaTools.sdfPathToInt("/World/ego")
t_reset = time.perf_counter()
world.reset()
world.play()
reset_s = time.perf_counter() - t_reset
physx = omni.physx.get_physx_interface()
state = State()
cmd = Command(received=True, gear=GEAR_DRIVE)


def flat(x, y):
    return np.zeros(np.broadcast(x, y).shape)


road = detail.height
height = rockset.height(road)

# --- collision tiles ---------------------------------------------------------------------
COLL = {}  # tile -> prim path
coll_ms = []
n_c = detail.n // COLLISION_STEP
_a = (np.arange(n_c)[None, :] + (n_c + 1) * np.arange(n_c)[:, None]).ravel()
COLL_IDX = Vt.IntArray.FromNumpy(np.stack([_a, _a + 1, _a + n_c + 2, _a + n_c + 1],
                                          axis=1).ravel().astype(np.int32))
COLL_CNT = Vt.IntArray.FromNumpy(np.full(n_c * n_c, 4, np.int32))


def stream_collision(x):
    want = {k for k in detail.tiles_near([x], [0.0], COLLISION_REACH)
            if detail.touches(*k) and detail.ready(k)}
    for k in [k for k in COLL if k not in want]:
        stage.RemovePrim(COLL.pop(k))
    for k in want - set(COLL):
        t0 = time.perf_counter()
        i, j = k
        h = detail.surface_tile(i, j)[::COLLISION_STEP, ::COLLISION_STEP]
        s = detail.spacing * COLLISION_STEP
        gx, gy = np.meshgrid(i * detail.tile + np.arange(n_c + 1) * s,
                             j * detail.tile + np.arange(n_c + 1) * s)
        pts = np.stack([gx, gy, h], axis=-1).reshape(-1, 3).astype(np.float32)
        path = f"/World/roadCollision/t_{i}_{j}".replace("-", "m")
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        m.CreateFaceVertexIndicesAttr(COLL_IDX)
        m.CreateFaceVertexCountsAttr(COLL_CNT)
        m.CreateVisibilityAttr(UsdGeom.Tokens.invisible)  # the render mesh is MeshStream's
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("none")
        UsdPhysics.CollisionGroup.Get(stage, DETAIL_G).GetCollidersCollectionAPI() \
            .CreateIncludesRel().AddTarget(m.GetPath())
        COLL[k] = path
        coll_ms.append((time.perf_counter() - t0) * 1e3)


# --- activation ----------------------------------------------------------------------------
ACTIVE = {}  # rock -> awake
toggles = [0]


def kinematic(i, on):
    """Rock i asleep (on: a static collider where it lies, its pose written back to USD
    first) or awake (a rigid body)."""
    prim = stage.GetPrimAtPath(PATHS[i])
    rb = UsdPhysics.RigidBodyAPI(prim)
    if on:
        p, q = rockset.pos[i], rockset.quat[i]
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        ops[0].Set(Gf.Vec3d(*map(float, p)))
        ops[1].Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    rb.GetRigidBodyEnabledAttr().Set(not on)
    toggles[0] += 1


def activate(x):
    y = state.position[1]
    for k in WIN:
        for i in LOADED[k]:
            if PATHS[i] is None:
                continue
            px, py = rockset.pos[i, 0], rockset.pos[i, 1]
            near = x - ACTIVE_BEHIND < px < x + vp.wheel_base + ACTIVE_AHEAD \
                and abs(py - y) < ACTIVE_SIDE
            if near and not ACTIVE.get(i):
                kinematic(i, False)
                ACTIVE[i] = True
            elif ACTIVE.get(i) and not near:
                read_rock(i)  # write-back: it stays where it lies
                kinematic(i, True)
                ACTIVE[i] = False


ROCK_V = {}


def read_rock(i):
    t = physx.get_rigidbody_transformation(PATHS[i])
    x, y, z, w = t["rotation"]
    pos = np.array(tuple(t["position"]))
    ROCK_V[i] = (pos - rockset.pos[i]) / DT
    rockset.set_pose(i, tuple(pos), (w, x, y, z))


def awake_n():
    return sum(1 for v in ACTIVE.values() if v)


z_att_prev = [None]
stats = {"under_tire": set(), "push_max": 0.0, "rock_v_max": 0.0, "read_ms": []}
contact_ms = []
TRACE = []  # per substep, the last 20000-40000
CONTACTS = []  # (x of the car, the other actor, the collider of ours, impulse N s, point, z)


def road_contact():
    t0 = time.perf_counter()
    pos, q = state.position, state.quat
    att = [qrot(q, (x, y, FRAME_Z)) for x, y in ATTACH]
    xy = np.array([[pos[0] + a[0], pos[1] + a[1]] for a in att])
    # the awake rocks near the wheels, read from PhysX (the rest are where they were)
    tr = time.perf_counter()
    rockset.seen[:] = True  # asleep and static rocks rest where they lie
    for i in rockset.near(xy[:, 0], xy[:, 1], margin=1.0):
        if ACTIVE.get(i):
            read_rock(i)
            rockset.rest(i, road)  # one off the ground does not hold the tire up
    stats["read_ms"].append((time.perf_counter() - tr) * 1e3)
    z_att = np.array([pos[2] + a[2] for a in att])
    v_att = np.zeros(4) if z_att_prev[0] is None else (z_att - z_att_prev[0]) / DT
    z_att_prev[0] = z_att
    fwd = qrot(q, (1.0, 0.0, 0.0))
    yaw = math.atan2(fwd[1], fwd[0])
    steer = (veh.tire_angle, veh.tire_angle, 0.0, 0.0)
    heading = np.array([[math.cos(yaw + a), math.sin(yaw + a)] for a in steer])
    zu0 = contact.zu.copy()
    o = contact.step(DT, height, flat, xy, heading, np.full(4, state.velocity[0]), z_att, v_att)
    scale = contact.friction_scale(o["load"], o["px_load"])
    for i in range(4):
        cz = pos[2] + att[i][2] - FRAME_Z + vp.wheel_radius
        f = (float(o["dfx"][i] * heading[i, 0]), float(o["dfx"][i] * heading[i, 1]),
             float(o["dfz"][i]))
        sim.apply_force_at_pos(STAGE_ID, BODY, f, (float(xy[i, 0]), float(xy[i, 1]), cz))
        if FRICTION_WRITES:
            veh.set_wheel_friction(i, vp.tire_friction * float(scale[i]))
    forces, lat = {}, np.zeros((4, 2))
    if placed:
        sh = rockset.shares(road, xy, heading, contact.width, contact.r, zu0)
        forces, lat = rk.reactions(sh, o["load"], heading)
        v_car = abs(state.velocity[0])
        for i, fs in forces.items():
            stats["under_tire"].add(i)
            if not ACTIVE.get(i):
                continue
            zc = max(p[2] - float(road(np.array([p[0]]), np.array([p[1]]))[0]) for _, p in fs)
            fs, sc = rk.limit(fs, MASS[i], ROCK_V.get(i, np.zeros(3)),
                              v_car * max(zc, 0.0) / vp.wheel_radius, DT)
            if sc < 1.0:  # the car feels only what the rock took
                lat = lat * sc
            for f, p in fs:
                sim.apply_force_at_pos(STAGE_ID, ROCK_INT[i], tuple(float(v) for v in f),
                                       tuple(float(v) for v in p))
                stats["push_max"] = max(stats["push_max"], float(np.linalg.norm(f)))
            stats["rock_v_max"] = max(stats["rock_v_max"],
                                      float(np.linalg.norm(ROCK_V.get(i, np.zeros(3)))))
        for i in range(4):  # the rocks' lateral push on the tire (the cam has none)
            if lat[i].any():
                cz = pos[2] + att[i][2] - FRAME_Z
                sim.apply_force_at_pos(STAGE_ID, BODY, (float(lat[i][0]), float(lat[i][1]),
                                                        0.0), (float(xy[i, 0]),
                                                               float(xy[i, 1]), cz))
    contact_ms.append((time.perf_counter() - t0) * 1e3)
    # the trace: x, y, yaw, vx, steer, loads (4), lateral on the car (4), slope push (4),
    # rocks under the tires, z
    lat_n = [float(np.hypot(*lat[i])) * np.sign(-lat[i][0] * heading[i, 1]
                                                 + lat[i][1] * heading[i, 0])
             for i in range(4)]
    if len(TRACE) > 40000:
        del TRACE[:20000]
    TRACE.append([pos[0], pos[1], yaw, state.velocity[0], veh.tire_angle, *o["load"],
                  *lat_n, *o["dfx"], len(forces), pos[2]])


def read_contacts():
    headers, data = sim.get_contact_report()
    for h in headers:
        a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
        a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        c0 = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
        c1 = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
        ours, other = (c0, a1) if a0.startswith("/World/ego") else (c1, a0)
        imp, pt = 0.0, None
        for k in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
            d = data[k]
            m = math.sqrt(sum(float(c) ** 2 for c in d.impulse))
            if m >= imp:
                imp, pt = m, tuple(round(float(c), 3) for c in d.position)
        if len(CONTACTS) > 40000:
            del CONTACTS[:20000]
        CONTACTS.append((round(state.position[0], 2), other, ours, round(imp, 1), pt,
                         round(state.position[2], 3)))


def substep():
    if veh.last is not None:
        road_contact()
    veh.apply(cmd, DT, state)
    world.step(render=False)
    veh.sample(physx, DT, state)
    if CONTACTS_ON:
        read_contacts()


def prefetch(x, v):
    ahead = max(road_layer.REACH, road_layer.LEAD_S * v)
    xs = np.arange(x - COLLISION_REACH, x + ahead, detail.tile / 2)
    detail.prefetch(detail.tiles_near(xs, np.zeros_like(xs), 6.0))


def run(kmh):
    v = kmh / 3.6
    veh.teleport(((RUN_FROM - 60.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0)), physx)
    contact.reset()
    z_att_prev[0] = None
    for _ in range(20 * SUB):
        cmd.velocity, cmd.acceleration, cmd.tire_angle = 0.0, 0.0, 0.0
        x = state.position[0]
        prefetch(x, v)
        stream_collision(x)
        stream_rocks(x)
        activate(x)
        substep()
    frames, awake, prims, n0 = [], [], [], len(contact_ms)
    left = stuck = None
    slow = 0.0
    CONTACTS.clear()
    TRACE.clear()
    stats["under_tire"].clear()
    while state.position[0] < X_START + LENGTH:
        t0 = time.perf_counter()
        x = state.position[0]
        if state.position[2] < -2.0 or abs(state.position[1]) > 6.0:
            left = [round(float(c), 2) for c in state.position]  # an outcome, not an error
            break
        slow = slow + FRAME if abs(state.velocity[0]) < 0.3 else 0.0
        if slow > 5.0:
            stuck = [round(float(c), 2) for c in state.position]
            break
        prefetch(x, v)
        meshes.update([(x, 0.0)])
        stream_collision(x)
        stream_rocks(x)
        activate(x)
        cmd.velocity = v
        cmd.acceleration = max(-3.0, min(3.0, 1.5 * (v - state.velocity[0])))
        fwd = qrot(state.quat, (1.0, 0.0, 0.0))
        yaw = math.atan2(fwd[1], fwd[0])
        vs = max(v, 1.0)
        cmd.tire_angle = max(-0.05, min(0.05, -vp.wheel_base * (
            W_LAT ** 2 * state.position[1] / vs ** 2 + 2 * Z_LAT * W_LAT * yaw / vs)))
        for _ in range(SUB):
            substep()
        if x > RUN_FROM + 40.0:
            frames.append((time.perf_counter() - t0) * 1e3)
            awake.append(awake_n())
            prims.append(sum(p is not None for p in PATHS) + sum(1 for v in WIN.values() if v))
        if len(frames) % 400 == 1:
            print(f"  {kmh:g} km/h at {x:.0f} m: {awake_n()} awake, {len(COLL)} "
                  f"collision tiles", flush=True)
    np.save(os.path.join(OUT, f"trace_{kmh:g}.npy"), np.array(TRACE))
    if CONTACTS:
        with open(os.path.join(OUT, f"contacts_{kmh:g}.json"), "w") as fh:
            json.dump(CONTACTS, fh)
    # rocks lost: awake ones below the surface
    lost = 0
    for i in [i for i, a in ACTIVE.items() if a]:
        read_rock(i)
        p = rockset.pos[i]
        if p[2] < float(road(np.array([p[0]]), np.array([p[1]]))[0]) - 0.3:
            lost += 1
    f = np.array(frames) if frames else np.zeros(1)
    cm = np.array(contact_ms[n0:]) if len(contact_ms) > n0 else np.zeros(1)
    rm = np.array(stats["read_ms"]) if stats["read_ms"] else np.zeros(1)
    np.save(os.path.join(OUT, f"frame_ms_{kmh:g}.npy"), f)
    return {"kmh": kmh, "frame_ms": {"p50": round(float(np.median(f)), 2),
                                     "p95": round(float(np.percentile(f, 95)), 2),
                                     "max": round(float(f.max()), 1),
                                     "over_50": int((f > 50).sum()), "n": len(frames)},
            "contact_ms_p50": round(float(np.median(cm)), 3),
            "rock_read_ms_p95": round(float(np.percentile(rm, 95)), 3),
            "awake": {"p50": int(np.median(awake)) if awake else 0,
                      "max": int(max(awake)) if awake else 0},
            "rock_prims": {"p50": int(np.median(prims)) if prims else 0,
                           "max": int(max(prims)) if prims else 0},
            "rocks_known": len(PATHS), "rocks_loose": int(sum(DYN)),
            "toggles": toggles[0], "rocks_under_tires": len(stats["under_tire"]),
            "push_max_N": round(stats["push_max"], 1),
            "rock_v_max": round(stats["rock_v_max"], 2), "lost": lost,
            "left_road_at": left, "stuck_at": stuck,
            "plate_hits": [(c[0], c[3]) for c in CONTACTS
                           if c[2].endswith("skid_plate") and c[3] > 1000.0][:12]}


out = {"settings": {k: carb.settings.get_settings().get(k) for k in
                    ("/physics/fabricEnabled", "/physics/updateToUsd")},
       "friction_writes": FRICTION_WRITES, "rock_reach_m": ROCK_REACH,
       "reset_s": round(reset_s, 1), "runs": {}}
print("SETUP", json.dumps(out), flush=True)
for kmh in SPEEDS:
    r = run(kmh)
    out["runs"][f"{kmh:g}"] = r
    print(f"{kmh:g} km/h", json.dumps(r), flush=True)
if place_ms:
    out["rock_tile_ms"] = {"load_p50": round(float(np.median(place_ms)), 2),
                           "prims_p50": round(float(np.median(prim_ms)), 2),
                           "prims_max": round(float(max(prim_ms)), 1), "n": len(prim_ms),
                           "queue_p50": round(float(np.median(queue_ms)), 2) if queue_ms else 0,
                           "queue_max": round(float(max(queue_ms)), 1) if queue_ms else 0}
if coll_ms:
    c = sorted(coll_ms)
    out["collision_tile_ms"] = {"p50": round(c[len(c) // 2], 1), "max": round(c[-1], 1),
                                "n": len(c)}
print("SPIKE-RESULT " + json.dumps(out))
detail.close()
app.close()
