"""Roadmap 066 R4, first slice: aib's Model 3 over single rocks, Isaac at 60 Hz with the tire
contact ours (`road_contact`, R5 and R2) reading the rocks' hulls (`rocks`).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/rock_fixtures.py \\
        [<out dir>]
    # prints one SPIKE-RESULT json

The road is a flat base. Each rock is a convex hull in a collision group the wheels' ground
query filters out, so PhysX's raycast sees the base only; the tire contact sees the hull's top
(question 4: one rock). A dynamic rock under a tire takes its share of the tire's force,
reversed (question 2). A skid plate at Tesla's 138 mm ground clearance (the spike's own; aib's
chassis box sits at about 0.42 m) is in the chassis group, so it hits rocks and is not seen by
the wheels.

Fixtures (ROCK_FIXTURES), each from rest 40 m before its rock, at a held speed:
- embedded_15: a static rock 15 cm tall, 30 cm long, under the left wheels, 40 km/h
- loose_15: the same rock loose (about 10 kg), 40 km/h
- loose_8: a loose stone 8 cm tall (about 1.4 kg) under the left tires' outer edge, 40 km/h
- boulder_35: a static boulder 35 cm tall between the wheel tracks, 10 km/h (high-centring)
"""
import json
import math
import os
import sys
import time

os.environ.pop("DISPLAY", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

from aib.control import GEAR_DRIVE, Command, State  # noqa: E402
from aib.frames import qrot  # noqa: E402
from aib.isaac.vehicle import GROUPS, WHEELS, Vehicle, ensure_vehicle_scene  # noqa: E402
from aib.rig import EgoConfig  # noqa: E402
from swb_isaac_world import road_contact as rc  # noqa: E402
from swb_isaac_world import rocks as rk  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/rock_fixtures", time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(OUT, exist_ok=True)
FRAME, SUB = 0.05, 3
DT = FRAME / SUB
CLEARANCE = 0.138  # m, Tesla's Model 3 ground clearance (Long Range AWD, 5.4 in)
W_LAT, Z_LAT = 0.8, 0.9

world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
base = UsdGeom.Mesh.Define(stage, "/World/env/Base/mesh")
base.CreatePointsAttr([Gf.Vec3f(-100, -30, 0), Gf.Vec3f(1100, -30, 0), Gf.Vec3f(1100, 30, 0),
                       Gf.Vec3f(-100, 30, 0)])
base.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
base.CreateFaceVertexCountsAttr([4])
UsdPhysics.CollisionAPI.Apply(base.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(base.GetPrim()).CreateApproximationAttr("none")
if not any(pr.IsA(UsdPhysics.Scene) for pr in stage.Traverse()):
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
ensure_vehicle_scene(stage)
ROCK_GROUP = f"{GROUPS}/rocks"
UsdPhysics.CollisionGroup.Define(stage, ROCK_GROUP).CreateFilteredGroupsRel().AddTarget(
    f"{GROUPS}/groundQuery")
mat = UsdPhysics.MaterialAPI.Apply(UsdGeom.Scope.Define(stage, "/World/rockMaterial").GetPrim())
mat.CreateStaticFrictionAttr(0.7)  # rock on gravel and on steel, about 0.6-0.8
mat.CreateDynamicFrictionAttr(0.6)
mat.CreateRestitutionAttr(0.1)

cfg = EgoConfig.load(os.environ.get("AIB_EGO") or os.path.join(
    HERE, "../deps/autoware_isaac_bridge/src/aib/config/ego.yaml"))
rig = cfg.load_rig()
vp = rig.vehicle
REAL_TRAVEL = vp.suspension_travel
import dataclasses  # noqa: E402

vp = dataclasses.replace(vp, suspension_travel=REAL_TRAVEL + 0.10)  # R2
veh = Vehicle(stage, "/World/ego", vp, rig.ecu, rig.box_center, rig.box_dims,
              ((0.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0)))
PhysxSchema.PhysxVehicleAPI(stage.GetPrimAtPath("/World/ego")).GetSuspensionLineQueryTypeAttr() \
    .Set(PhysxSchema.Tokens.raycast)
(front_axle, rear_axle), sag = vp.suspension()
FRAME_Z = vp.wheel_radius + vp.suspension_travel - sag
# the skid plate: between the axles, the body's frame at rest on the road (z 0 = the road)
plate = UsdGeom.Cube.Define(stage, "/World/ego/skid_plate")
plate.CreateSizeAttr(1.0)
PLATE = (0.35, vp.wheel_base - 0.35, 1.30, 0.04)  # x0, x1, width, thickness (body frame)
UsdGeom.XformCommonAPI(plate.GetPrim()).SetTranslate(
    Gf.Vec3d((PLATE[0] + PLATE[1]) / 2, 0.0, CLEARANCE + PLATE[3] / 2))
UsdGeom.XformCommonAPI(plate.GetPrim()).SetScale(Gf.Vec3f(PLATE[1] - PLATE[0], PLATE[2],
                                                          PLATE[3]))
UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
UsdPhysics.CollisionGroup.Get(stage, f"{GROUPS}/chassis").GetCollidersCollectionAPI() \
    .CreateIncludesRel().AddTarget(plate.GetPath())
ATTACH = [((vp.wheel_base if front else 0.0), side * vp.wheel_tread / 2)
          for _, front, side in WHEELS]
SIDES = {(front, side): i for i, (_, front, side) in enumerate(WHEELS)}
FL = SIDES[(True, 1)]
contact = rc.TireContact(
    [rc.Corner(*(front_axle if front else rear_axle), REAL_TRAVEL, vp.wheel_radius,
               vp.wheel_width, px_travel=vp.suspension_travel) for _, front, _ in WHEELS],
    bars=rc.anti_roll_bars((SIDES[(True, 1)], SIDES[(True, -1)]),
                           (SIDES[(False, 1)], SIDES[(False, -1)]),
                           front_axle[1], rear_axle[1]))

# --- the fixtures' rocks, each on its own stretch -----------------------------------------
HALF_TREAD = vp.wheel_tread / 2
FIXTURES = {  # name: (rock x, y, long axis d, dynamic, km/h)
    "embedded_15": (100.0, HALF_TREAD, 0.30, False, 40.0),
    "loose_15": (300.0, HALF_TREAD, 0.30, True, 40.0),
    "loose_8": (500.0, HALF_TREAD + vp.wheel_width * 0.4, 0.16, True, 40.0),
    "boulder_35": (700.0, 0.0, 0.70, False, 10.0),
}
CHOSEN = [f for f in os.environ.get("ROCK_FIXTURES", ",".join(FIXTURES)).split(",") if f]
rng = np.random.default_rng(4)
ROCKS, PATHS, DYNAMIC = [], [], []
for k, (name, (x, y, d, dyn, _)) in enumerate(FIXTURES.items()):
    r = rk.make_rock(rng, d, rough=0.10)
    ROCKS.append(r)
    path = f"/World/rocks/{name}"
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(r.verts.astype(np.float32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(r.faces.ravel().astype(np.int32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(r.faces), 3, np.int32)))
    m.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.42, 0.38)])
    z = -float(r.verts[:, 2].min())  # resting on the road
    UsdGeom.XformCommonAPI(m.GetPrim()).SetTranslate(Gf.Vec3d(x, y, z))
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    PhysxSchema.PhysxCollisionAPI.Apply(m.GetPrim()).CreateContactOffsetAttr(0.01)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath("/World/rockMaterial")),
        UsdShade.Tokens.weakerThanDescendants, "physics")
    UsdPhysics.CollisionGroup.Get(stage, ROCK_GROUP).GetCollidersCollectionAPI() \
        .CreateIncludesRel().AddTarget(m.GetPath())
    if dyn:
        UsdPhysics.RigidBodyAPI.Apply(m.GetPrim())
        UsdPhysics.MassAPI.Apply(m.GetPrim()).CreateDensityAttr(rk.DENSITY)
        PhysxSchema.PhysxRigidBodyAPI.Apply(m.GetPrim()).CreateEnableSpeculativeCCDAttr(True)
    PATHS.append(path)
    DYNAMIC.append(dyn)
rockset = rk.RockSet(ROCKS, [(FIXTURES[n][0], FIXTURES[n][1], -float(r.verts[:, 2].min()))
                             for n, r in zip(FIXTURES, ROCKS)], [(1, 0, 0, 0)] * len(ROCKS))
ROCK_INT = [PhysicsSchemaTools.sdfPathToInt(p) for p in PATHS]

sim = omni.physx.get_physx_simulation_interface()
STAGE_ID = omni.usd.get_context().get_stage_id()
BODY = PhysicsSchemaTools.sdfPathToInt("/World/ego")
world.reset()
world.play()
physx = omni.physx.get_physx_interface()
state = State()
cmd = Command(received=True, gear=GEAR_DRIVE)


def flat(x, y):
    return np.zeros(np.broadcast(x, y).shape)


height = rockset.height(flat)
log = {}


ROCK_V = np.zeros((len(PATHS), 3))  # each dynamic rock's velocity over the last step
ROCK_PEAK = [[] for _ in PATHS]


def read_rocks():
    for i, p in enumerate(PATHS):
        if not DYNAMIC[i]:
            continue
        t = physx.get_rigidbody_transformation(p)
        x, y, z, w = t["rotation"]  # PhysX's (x, y, z, w)
        pos = np.array(tuple(t["position"]))
        ROCK_V[i] = (pos - rockset.pos[i]) / DT
        ROCK_PEAK[i].append(ROCK_V[i].copy())
        rockset.set_pose(i, tuple(pos), (w, x, y, z))


z_att_prev = [None]


def road_contact():
    read_rocks()
    pos, q = state.position, state.quat
    att = [qrot(q, (x, y, FRAME_Z)) for x, y in ATTACH]
    xy = np.array([[pos[0] + a[0], pos[1] + a[1]] for a in att])
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
        veh.set_wheel_friction(i, vp.tire_friction * float(scale[i]))
    # the rocks' share of each tire's force, reversed, on the dynamic ones
    sh = rockset.shares(flat, xy, heading, contact.width, contact.r, zu0)
    pushed = 0.0
    forces, lat = rk.reactions(sh, o["load"], heading)
    v_car = abs(state.velocity[0])
    for i, fs in forces.items():
        if not DYNAMIC[i]:
            continue
        zc = max(p[2] for _, p in fs)  # the contact's height over the road (flat, 0)
        fs, sc = rk.limit(fs, ROCKS[i].mass, ROCK_V[i], v_car * zc / vp.wheel_radius, DT)
        if sc < 1.0:  # the car feels only what the rock took
            lat = lat * sc
        for f, p in fs:
            sim.apply_force_at_pos(STAGE_ID, ROCK_INT[i], tuple(float(v) for v in f),
                                   tuple(float(v) for v in p))
            pushed = max(pushed, float(np.linalg.norm(f)))
    for i in range(4):  # the rocks' lateral push on the tire (the cam has none)
        if lat[i].any():
            cz = pos[2] + att[i][2] - FRAME_Z
            sim.apply_force_at_pos(STAGE_ID, BODY, (float(lat[i][0]), float(lat[i][1]), 0.0),
                                   (float(xy[i, 0]), float(xy[i, 1]), cz))
    return o, sh, pushed


def substep(rec):
    o = sh = None
    pushed = 0.0
    if veh.last is not None:
        o, sh, pushed = road_contact()
    veh.apply(cmd, DT, state)
    world.step(render=False)
    veh.sample(physx, DT, state)
    if o is not None and rec is not None:
        rec.append((o, sh, pushed))


def pitch_roll(q):
    f = qrot(q, (1.0, 0.0, 0.0))
    s = qrot(q, (0.0, 1.0, 0.0))
    return math.degrees(math.asin(-f[2])), math.degrees(math.asin(s[2]))


def run(name):
    rx, ry, d, dyn, kmh = FIXTURES[name]
    i_rock = list(FIXTURES).index(name)
    v = kmh / 3.6
    veh.teleport(((rx - 40.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0)), physx)
    contact.reset()
    z_att_prev[0] = None
    for _ in range(20 * SUB):
        cmd.velocity, cmd.acceleration, cmd.tire_angle = 0.0, 0.0, 0.0
        substep(None)
    p0 = np.array(physx.get_rigidbody_transformation(PATHS[i_rock])["position"]) if dyn \
        else rockset.pos[i_rock].copy()
    pr0 = pitch_roll(state.quat)
    rec, body = [], []
    t_end = None
    t = 0.0
    slow = 0.0  # s at under 0.2 m/s near the rock: stuck (high-centred)
    while True:
        x = state.position[0]
        if t_end is None and x > rx + 8.0:
            t_end = t + 2.0
        slow = slow + FRAME if (x > rx - 6.0 and abs(state.velocity[0]) < 0.2) else 0.0
        if (t_end is not None and t > t_end) or t > 60.0 or slow > 3.0:
            break
        if state.position[2] < -2.0 or abs(state.position[1]) > 10.0:
            raise RuntimeError(f"{name}: the car left the road at {state.position}")
        cmd.velocity = v
        cmd.acceleration = max(-3.0, min(3.0, 1.5 * (v - state.velocity[0])))
        fwd = qrot(state.quat, (1.0, 0.0, 0.0))
        yaw = math.atan2(fwd[1], fwd[0])
        vs = max(v, 1.0)
        cmd.tire_angle = max(-0.05, min(0.05, -vp.wheel_base * (
            W_LAT ** 2 * state.position[1] / vs ** 2 + 2 * Z_LAT * W_LAT * yaw / vs)))
        for _ in range(SUB):
            substep(rec if x > rx - 10.0 else None)
            if x > rx - 10.0:
                pr = pitch_roll(state.quat)
                body.append((t, state.position[0], state.position[2], pr[0] - pr0[0],
                             pr[1] - pr0[1], state.velocity[0]))
            t += DT
    b = np.array(body)
    az = np.diff(b[:, 2], 2) / DT ** 2
    fs = np.concatenate([o["susp_inner"] for o, _, _ in rec], axis=1)  # (4, steps * inner)
    ft = np.concatenate([o["load_inner"] for o, _, _ in rec], axis=1)
    jounce = np.array([o["jounce"] for o, _, _ in rec])
    share = np.array([max([r[1] for r in sh[FL] if r[0] == i_rock] or [0.0])
                      for _, sh, _ in rec])
    out = {"kmh": kmh, "rock_mass_kg": round(ROCKS[i_rock].mass, 2),
           "rock_height_m": round(ROCKS[i_rock].height, 3), "dynamic": dyn,
           "fl_jounce_rise_m": round(float(jounce[:, FL].max() - jounce[:5, FL].mean()), 3),
           "fl_share_max": round(float(share.max()), 3),
           "fl_susp_peak_kN": round(float(fs[FL].max()) / 1e3, 2),
           "fl_tire_peak_kN": round(float(ft[FL].max()) / 1e3, 2),
           "susp_peak_kN": round(float(fs.max()) / 1e3, 2),
           "px_limit_steps": int(sum(o["px_limit"].any() for o, _, _ in rec)),
           "rim_strike_mm": round(float(max(o["rim_strike"].max() for o, _, _ in rec)) * 1e3, 1),
           "pitch_deg": [round(float(b[:, 3].min()), 2), round(float(b[:, 3].max()), 2)],
           "roll_deg": [round(float(b[:, 4].min()), 2), round(float(b[:, 4].max()), 2)],
           "body_az_peak": round(float(np.abs(az).max()), 1),
           "v_min": round(float(b[:, 5].min()), 2), "x_end": round(float(b[-1, 1]), 2),
           "push_peak_N": round(max(p for _, _, p in rec), 1),
           "rock_v_peak": round(float(max(np.linalg.norm(v) for v in ROCK_PEAK[i_rock])), 2)
           if ROCK_PEAK[i_rock] else 0.0}
    if dyn:
        p1 = np.array(physx.get_rigidbody_transformation(PATHS[i_rock])["position"])
        out["rock_moved_m"] = [round(float(c), 3) for c in (p1 - p0)]
    if name == "boulder_35":
        out["stuck"] = slow > 3.0
        out["front_axle_to_rock_m"] = round(float(rx - (b[-1, 1] + vp.wheel_base)), 2)
    np.save(os.path.join(OUT, f"{name}_body.npy"), b)
    np.save(os.path.join(OUT, f"{name}_fs.npy"), fs)
    return out


t = physx.get_rigidbody_transformation(PATHS[1])
print("ROCK-POSE", PATHS[1], t, "set", rockset.pos[1], flush=True)  # (x, y, z, w) at rest
res = {}
for name in CHOSEN:
    r = run(name)
    res[name] = r
    print(name, json.dumps(r), flush=True)
print("SPIKE-RESULT " + json.dumps(res))
app.close()
