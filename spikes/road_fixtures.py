"""Roadmap 066 R0: aib's Model 3 over the fixture course, each wheel logged every PhysX step.

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/road_fixtures.py \\
        [<out dir>]
    # prints one SPIKE-RESULT json; each case's per-step trace goes to <out dir>/<case>.jsonl

The lanes are `fixture_course.LANES` (cleats, a pothole under the left wheels, washboard, a
kicker crest, a boulder, a loose 10 kg rock), triangle meshes with vertical edges where the
profile has them. Each case drives a lane at a held speed (aib's onboard computers, no ROS,
no surface layer: every wheel at the rig's dry friction) and records, per PhysX step, the body
(height, pitch, roll, vertical velocity and acceleration) and each wheel (on the ground, the
suspension force, the hit point, the jounce where PhysX reports it).

What it answers (roadmap 066, "The five open questions"):
- question 1, how often PhysX samples the ground: `cleat_10` at 20 m/s from five start offsets
  0.07 m apart. A query once per PhysX step (0.33 m of travel at 60 Hz) lands on the 0.2 m
  cleat in some offsets only; a query per vehicle substep (8 per step) lands in every one.
- question 2, whether a rock under a wheel is pushed: the `rock` lane's rock, its velocity
  after the left wheels pass, and whether the wheel query reported it as the ground.
- the baseline against `fixture_course.reference()` per fixture.

Variants (environment): ROAD_QUERY=raycast|sweep (default raycast, today's), ROAD_LIMIT=1
(`limitSuspensionExpansionVelocity`), ROAD_RATE=<PhysX Hz> (default 60; the frame stays 50 ms),
ROAD_VSUB=<vehicle substeps per PhysX step> (default the rig's), ROAD_TRAVEL=<m> (suspension
travel, default the rig's), ROAD_ONLY=<case,...>.

ROAD_TIRE=ours (066 R5): PhysX's query sees only the base road (`fixture_course.BASE`: every
lane flat but the kicker), and `swb_isaac_world.road_contact` gives each wheel its contact with
the true profile every PhysX step: its force on the body (`apply_force_at_pos`, for the next
step) and the tire's load as each wheel's friction. Each row then also has the model's
per-wheel load (`tl`), its inner-step peak (`tlmax`) and PhysX's own (`pl`). The rock lane is
R4's and is skipped.

R2 (with ROAD_TIRE=ours): the corner's damper is digressive with more rebound than bump
(ROAD_DAMPER=linear: the rig's), the anti-roll bars are on (ROAD_BARS=0: off), and
ROAD_TRAVEL is PhysX's travel only, its frame raised to match: the corner keeps the rig's
travel, so PhysX's rigid limit lies beyond our bump stop. Each row has the suspension forces at
the inner steps (`fs_in`, all four wheels summed) and whether PhysX's jounce reached its limit.
The `circle` cases drive a flat pad at a held speed and steer angle: the roll gradient.
"""
import dataclasses
import json
import math
import os
import sys
import time

os.environ.pop("DISPLAY", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.physx.bindings._physx as pxb  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Gf, PhysxSchema, PhysicsSchemaTools, UsdGeom, UsdPhysics  # noqa: E402

import fixture_course as fc  # noqa: E402
from aib.control import GEAR_DRIVE, Command, State  # noqa: E402
from aib.isaac.vehicle import WHEELS, Vehicle, ensure_vehicle_scene  # noqa: E402
from aib.rig import EgoConfig  # noqa: E402
from aib.frames import qrot  # noqa: E402
from swb_isaac_world import road_contact as rc  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/road_fixtures", time.strftime("%Y-%m-%d_%H-%M-%S"))
QUERY = os.environ.get("ROAD_QUERY", "raycast")
LIMIT = os.environ.get("ROAD_LIMIT", "0") == "1"
RATE = float(os.environ.get("ROAD_RATE", "60"))
ONLY = [c for c in os.environ.get("ROAD_ONLY", "").split(",") if c]
OURS = os.environ.get("ROAD_TIRE", "physx") == "ours"
FRAME = 0.05
SUB = max(1, round(FRAME * RATE))
DT = FRAME / SUB
os.makedirs(OUT, exist_ok=True)

STATE_KEYS = sorted(n for n in dir(pxb) if n.startswith("VEHICLE_WHEEL_STATE_"))
K = {n[len("VEHICLE_WHEEL_STATE_"):]: getattr(pxb, n) for n in STATE_KEYS}

world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)


def lane_mesh(name):
    y0 = fc.LANES[name][0]
    xs, ys, hz = fc.lane_grid(name, base=OURS)
    nx, ny = len(xs), len(ys)
    pts = [Gf.Vec3f(float(x), float(y + y0), float(hz[j, i]))
           for j, y in enumerate(ys) for i, x in enumerate(xs)]
    idx = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            idx += [a, a + 1, a + nx + 1, a + nx]
    mesh = UsdGeom.Mesh.Define(stage, f"/World/env/Road_{name}/mesh")
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateFaceVertexCountsAttr([4] * ((nx - 1) * (ny - 1)))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    return len(pts)


verts = {name: lane_mesh(name) for name in fc.LANES}
# A flat pad for the steady circles (R2's roll gradient): x 40-160, y -190 to -70
PAD = (40.0, 160.0, -190.0, -70.0)
pad = UsdGeom.Mesh.Define(stage, "/World/env/Pad/mesh")
pad.CreatePointsAttr([Gf.Vec3f(PAD[0], PAD[2], 0), Gf.Vec3f(PAD[1], PAD[2], 0),
                      Gf.Vec3f(PAD[1], PAD[3], 0), Gf.Vec3f(PAD[0], PAD[3], 0)])
pad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
pad.CreateFaceVertexCountsAttr([4])
UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(pad.GetPrim()).CreateApproximationAttr("none")
ROCK = "/World/env/rock"
ROCK_SIZE = (0.25, 0.25, 0.15)
ROCK_AT = (fc.X0 + 0.15, fc.LANES["rock"][0] + fc.TREAD_HALF, ROCK_SIZE[2] / 2 + 0.002)
rock = UsdGeom.Cube.Define(stage, ROCK)
rock.CreateSizeAttr(1.0)
UsdGeom.XformCommonAPI(rock.GetPrim()).SetTranslate(Gf.Vec3d(*ROCK_AT))
UsdGeom.XformCommonAPI(rock.GetPrim()).SetScale(Gf.Vec3f(*ROCK_SIZE))
UsdPhysics.CollisionAPI.Apply(rock.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(rock.GetPrim())
UsdPhysics.MassAPI.Apply(rock.GetPrim()).CreateMassAttr(10.0)
if not any(pr.IsA(UsdPhysics.Scene) for pr in stage.Traverse()):
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
ensure_vehicle_scene(stage)

cfg = EgoConfig.load(os.environ.get("AIB_EGO") or os.path.join(
    HERE, "../deps/autoware_isaac_bridge/src/aib/config/ego.yaml"))
rig = cfg.load_rig()
vp = rig.vehicle
REAL_TRAVEL = vp.suspension_travel  # the corner's, whatever PhysX's is
if os.environ.get("ROAD_TRAVEL"):  # more compression room, the same sag
    vp = dataclasses.replace(vp, suspension_travel=float(os.environ["ROAD_TRAVEL"]))
if os.environ.get("ROAD_VSUB"):
    vp = dataclasses.replace(vp, physx_substeps=int(os.environ["ROAD_VSUB"]))
veh = Vehicle(stage, "/World/ego", vp, rig.ecu, rig.box_center, rig.box_dims,
              ((0.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0)))
vapi = PhysxSchema.PhysxVehicleAPI(stage.GetPrimAtPath("/World/ego"))
vapi.GetSuspensionLineQueryTypeAttr().Set(
    PhysxSchema.Tokens.sweep if QUERY == "sweep" else PhysxSchema.Tokens.raycast)
vapi.CreateLimitSuspensionExpansionVelocityAttr(LIMIT)
(front_axle, rear_axle), sag = vp.suspension()
FRAME_Z = vp.wheel_radius + vp.suspension_travel - sag
ATTACH = [((vp.wheel_base if front else 0.0), side * vp.wheel_tread / 2)
          for _, front, side in WHEELS]
LINEAR = os.environ.get("ROAD_DAMPER", "") == "linear"
BARS = os.environ.get("ROAD_BARS", "1") != "0"
CORNERS = [rc.Corner(*(front_axle if front else rear_axle), REAL_TRAVEL, vp.wheel_radius,
                     vp.wheel_width, px_travel=vp.suspension_travel,
                     **({"rebound_ratio": 1.0, "damper_high": 1.0} if LINEAR else {}))
           for _, front, _ in WHEELS]
SIDES = {(front, side): i for i, (_, front, side) in enumerate(WHEELS)}
BAR_RATES = rc.anti_roll_bars((SIDES[(True, 1)], SIDES[(True, -1)]),
                              (SIDES[(False, 1)], SIDES[(False, -1)]),
                              front_axle[1], rear_axle[1]) if BARS else []
contact = rc.TireContact(CORNERS, bars=BAR_RATES) if OURS else None
sim = omni.physx.get_physx_simulation_interface()
STAGE_ID = omni.usd.get_context().get_stage_id()
BODY = PhysicsSchemaTools.sdfPathToInt("/World/ego")
road_ms = []
last_contact = {}
z_att_prev = None

world.reset()
world.play()
physx = omni.physx.get_physx_interface()
state = State()
cmd = Command(received=True, gear=GEAR_DRIVE)
step_s = []


def road_contact():
    """066 R5: the model's step from the body as the last step left it; its forces for the
    next step, its loads as each wheel's friction."""
    global z_att_prev
    t0 = time.perf_counter()
    pos, q = state.position, state.quat
    att = [qrot(q, (x, y, FRAME_Z)) for x, y in ATTACH]
    xy = np.array([[pos[0] + a[0], pos[1] + a[1]] for a in att])
    z_att = np.array([pos[2] + a[2] for a in att])
    v_att = np.zeros(4) if z_att_prev is None else (z_att - z_att_prev) / DT
    z_att_prev = z_att
    fwd = qrot(q, (1.0, 0.0, 0.0))
    yaw = math.atan2(fwd[1], fwd[0])
    steer = (veh.tire_angle, veh.tire_angle, 0.0, 0.0)  # WHEELS' order
    heading = np.array([[math.cos(yaw + a), math.sin(yaw + a)] for a in steer])
    o = contact.step(DT, fc.world_height, lambda x, y: fc.world_height(x, y, base=True), xy,
                     heading, np.full(4, state.velocity[0]), z_att, v_att)
    scale = contact.friction_scale(o["load"], o["px_load"])
    for i in range(4):
        cz = pos[2] + att[i][2] - FRAME_Z + vp.wheel_radius  # about the wheel's centre
        f = (float(o["dfx"][i] * heading[i, 0]), float(o["dfx"][i] * heading[i, 1]),
             float(o["dfz"][i]))
        sim.apply_force_at_pos(STAGE_ID, BODY, f, (float(xy[i, 0]), float(xy[i, 1]), cz))
        veh.set_wheel_friction(i, vp.tire_friction * float(scale[i]))
    last_contact.update(fs_in=o["susp_inner"].sum(axis=0), px_limit=o["px_limit"] * 1.0,
                        rim=o["rim_strike"] * 1000.0, tl=o["load"], tlmax=o["load_inner"].max(axis=1), pl=o["px_load"],
                        za=z_att * 1000.0)  # mm
    road_ms.append(time.perf_counter() - t0)


def substep():
    if OURS and veh.last is not None:
        road_contact()
    veh.apply(cmd, DT, state)
    t0 = time.perf_counter()
    world.step(render=False)
    step_s.append(time.perf_counter() - t0)
    veh.sample(physx, DT, state)


def euler(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return math.degrees(roll), math.degrees(pitch)


def wheel_row(wp):
    w = physx.get_wheel_state(wp) or {}
    on = bool(w.get(K["IS_ON_GROUND"], False))
    f = w.get(K["SUSPENSION_FORCE"], (0, 0, 0))
    hit = w.get(K["GROUND_HIT_POSITION"], (0, 0, 0))
    r = {"on": on, "load": round(math.sqrt(sum(float(c) ** 2 for c in f)), 1) if on else 0.0,
         "hit": [round(float(c), 3) for c in hit] if on else None,
         "shape": str(w.get(K["GROUND_SHAPE"], "")) if on else ""}
    if on and K.get("GROUND_PLANE") in w:
        r["plane"] = [round(float(c), 3) for c in w[K["GROUND_PLANE"]]]
    for key in ("SUSPENSION_JOUNCE", "GROUND_ACTOR"):
        if key in K and K[key] in w:
            v = w[K[key]]
            r[key.lower()] = round(float(v), 4) if isinstance(v, (int, float)) else str(v)
    return r


BRAKE_AT = 3.0  # m before the fixture, where a braking case starts its stop
BRAKE = -6.0  # m/s^2 asked of the onboard computers


def drive(name, speed, offset=0.0, brake=False):
    """Hold `speed` along the lane from x = 40 + offset through the fixture; per-step rows
    while the front axle is between X0 - 5 and the fixture's end + 12."""
    y = fc.LANES[name][0]
    veh.teleport(((40.0 + offset, y, 0.05), (1.0, 0.0, 0.0, 0.0)), physx)
    if name == "rock":
        rp = stage.GetPrimAtPath(ROCK)
        UsdGeom.XformCommonAPI(rp).SetTranslate(Gf.Vec3d(*ROCK_AT))
        UsdPhysics.RigidBodyAPI(rp).GetVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
        UsdPhysics.RigidBodyAPI(rp).GetAngularVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
    cmd.tire_angle = 0.0
    if OURS:
        global z_att_prev
        contact.reset()
        z_att_prev = None
        for i in range(4):
            veh.set_wheel_friction(i, vp.tire_friction)
    for _ in range(20 * SUB):
        cmd.velocity, cmd.acceleration = 0.0, 0.0
        substep()
    rows, t = [], 0.0
    lo, hi = fc.X0 - 5.0, fc.X0 + fc.EXTENT[name] + 12.0
    while True:
        front = state.position[0] + vp.wheel_base
        cmd.velocity = speed
        cmd.acceleration = max(-3.0, min(3.0, 1.5 * (speed - state.velocity[0])))
        if brake and front >= fc.X0 - BRAKE_AT:
            cmd.velocity, cmd.acceleration = 0.0, BRAKE
            if state.velocity[0] < 0.05:
                break
        if front > hi + (60.0 if brake else 0.0) or t > 60.0:
            break
        z_prev = state.position[2]
        substep()
        t += DT
        if front >= lo:
            roll, pitch = euler(state.quat)
            row = {"t": round(t, 4), "x": round(state.position[0], 4),
                         "z": round(state.position[2], 4), "vz_w": round(
                             (state.position[2] - z_prev) / DT, 4),
                         "roll": round(roll, 3), "pitch": round(pitch, 3),
                         "vx": round(state.velocity[0], 3),
                   "ax": round(state.acceleration[0], 3),
                   "slip": [round(ws.slip, 3) for ws in veh.wheel_states or ()],
                   "wheels": [wheel_row(wp) for wp in veh.wheel_paths]}
            if OURS:
                for key, v in last_contact.items():
                    row[key] = [round(float(c), 1) for c in v]
            rows.append(row)
    tag = "brake" if brake else ""
    with open(os.path.join(OUT, f"{name}@{speed:g}+{offset:g}{tag}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


CIRCLE_START = (100.0, -175.0)


def circle(speed, steer, settle=10.0, hold=6.0):
    """A steady circle on the pad: roll against lateral acceleration (v x yaw rate)."""
    veh.teleport(((CIRCLE_START[0], CIRCLE_START[1], 0.05), (1.0, 0.0, 0.0, 0.0)), physx)
    if OURS:
        global z_att_prev
        contact.reset()
        z_att_prev = None
    for _ in range(20 * SUB):
        cmd.velocity, cmd.acceleration, cmd.tire_angle = 0.0, 0.0, 0.0
        substep()
    t, rows, yaw_prev = 0.0, [], None
    while t < settle + hold:
        cmd.velocity = speed
        cmd.acceleration = max(-3.0, min(3.0, 1.5 * (speed - state.velocity[0])))
        cmd.tire_angle = steer if state.velocity[0] > 0.5 * speed else 0.0
        substep()
        t += DT
        fwd = qrot(state.quat, (1.0, 0.0, 0.0))
        yaw = math.atan2(fwd[1], fwd[0])
        if yaw_prev is not None and t > settle:
            r = math.remainder(yaw - yaw_prev, 2 * math.pi) / DT
            roll, _ = euler(state.quat)
            rows.append((state.velocity[0] * r, roll, state.position[0], state.position[1]))
        yaw_prev = yaw
    a = np.array(rows)
    out = {"ay": round(float(a[:, 0].mean()), 3), "roll_deg": round(float(a[:, 1].mean()), 3),
           "on_pad": bool(PAD[0] < a[:, 2].min() and a[:, 2].max() < PAD[1]
                          and PAD[2] < a[:, 3].min() and a[:, 3].max() < PAD[3])}
    out["deg_per_g"] = round(abs(out["roll_deg"]) / (abs(out["ay"]) / 9.81), 3)
    return out


def landing(rows):
    """The kicker's landing: the four suspension forces' sum at the inner steps (`fs_in`),
    its peak as g of the car's weight and its width at half the peak above static (ms)."""
    fs = np.concatenate([r["fs_in"] for r in rows if "fs_in" in r])
    if not len(fs):
        return None
    weight = vp.mass * 9.81
    i = int(fs.argmax())
    half = weight + (fs[i] - weight) / 2
    lo = i
    while lo > 0 and fs[lo - 1] > half:
        lo -= 1
    hi = i
    while hi < len(fs) - 1 and fs[hi + 1] > half:
        hi += 1
    return {"peak_g": round(float(fs[i] / weight), 2),
            "half_width_ms": round((hi - lo + 1) * DT / contact.inner * 1e3, 1)}


def summary(name, speed, rows):
    """The front-left wheel and the body over the fixture, in the reference's terms."""
    pre = [r for r in rows if r["x"] + vp.wheel_base < fc.X0 - 2.0]
    static = float(np.mean([r["wheels"][0]["load"] for r in pre])) if pre else float("nan")
    z0 = float(np.mean([r["z"] for r in pre])) if pre else rows[0]["z"]
    load = np.array([r["wheels"][0]["load"] for r in rows])
    peak = load
    if OURS:  # the model's tire, not PhysX's spring on the base road
        static = float(np.mean([r["tl"][0] for r in pre])) if pre else float("nan")
        load = np.array([r["tl"][0] for r in rows])
        peak = np.array([r["tlmax"][0] for r in rows])
    fx = np.array([r["x"] + vp.wheel_base for r in rows]) - fc.X0
    rise = np.nonzero(load > static * 1.05)[0]
    vz = np.array([r["vz_w"] for r in rows])
    az = np.diff(vz) / DT if len(vz) > 1 else np.zeros(1)
    t_off = sum(DT for r in rows if not r["wheels"][0]["on"]) if not OURS else \
        float(sum(DT for r in rows if r["tl"][0] <= 0.0))
    hits = [r["wheels"][0]["hit"][0] - fc.X0 for r in rows
            if r["wheels"][0]["on"] and -0.5 < r["wheels"][0]["hit"][0] - fc.X0 < 1.0]
    tops = [r["wheels"][0]["hit"][2] for r in rows if r["wheels"][0]["on"]]
    return {"static_N": round(static, 1), "peak_N": round(float(peak.max()), 1),
            "min_N": round(float(load.min()), 1),
            "first_rise_x": None if not len(rise) else round(float(fx[rise[0]]), 3),
            "body_acc_peak": round(float(np.abs(az).max()), 2),
            "airborne_s": round(t_off, 3), "body_rise_m": round(float(max(r["z"] for r in rows)
                                                                      - z0), 3),
            "pitch_max": round(float(max(abs(r["pitch"]) for r in rows)), 2),
            "fl_corner_rise_m": round((max(r["za"][0] for r in rows) - float(np.mean(
                [r["za"][0] for r in pre]))) / 1000.0, 3) if OURS and pre else None,
            "fl_hit_max_z": round(float(max(tops)), 3) if tops else None,
            # PhysX's jounce at its rigid limit, steps (R2: none)
            "rigid_limit_steps": sum(1 for r in rows if any(
                w.get("suspension_jounce", 0.0) >= vp.suspension_travel - 1e-3
                for w in r["wheels"] if w["on"])),
            "landing": landing(rows) if OURS and name == "kicker" else None,
            "fl_hits_near": [round(h, 3) for h in hits][:12]}


CASES = [("flat", 20.0, 0.0), ("cleat_05", 10.0, 0.0), ("cleat_10", 10.0, 0.0),
         ("cleat_15", 10.0, 0.0), ("pothole", 10.0, 0.0), ("washboard", 10.0, 0.0),
         ("washboard", 20.0, 0.0), ("kicker", 15.0, 0.0), ("boulder", 10.0, 0.0),
         ("rock", 5.0, 0.0)] + [("cleat_10", 20.0, 0.07 * k) for k in range(5)]
# Braking from BRAKE_AT before the fixture (R5: no kick at a step, the stop on washboard)
BRAKES = [("flat", 15.0), ("cleat_10", 10.0), ("washboard", 15.0), ("pothole", 10.0)]
out = {"tire": "ours" if OURS else "physx", "query": QUERY, "limit_expansion": LIMIT, "physx_hz": RATE, "substeps_per_frame": SUB,
       "vehicle_substeps": vp.physx_substeps, "wheel_state_keys": sorted(K),
       "rig": {"front": front_axle, "rear": rear_axle, "sag": sag,
               "travel": REAL_TRAVEL, "physx_travel": vp.suspension_travel,
               "damper": "linear" if LINEAR else "digressive", "bars": BAR_RATES, "wheel_radius": vp.wheel_radius,
               "wheel_mass": vp.wheel_mass, "wheel_base": vp.wheel_base},
       "lane_vertices": verts, "cases": {}}
for name, speed, offset in CASES:
    key = f"{name}@{speed:g}+{offset:g}"
    if ONLY and name not in ONLY and key not in ONLY:
        continue
    if OURS and name == "rock":
        continue
    t0 = time.perf_counter()
    rows = drive(name, speed, offset)
    s = summary(name, speed, rows)
    if name == "rock":
        tr = physx.get_rigidbody_transformation(ROCK)
        s["rock_moved_m"] = round(math.dist(tuple(float(c) for c in tr["position"]),
                                            ROCK_AT), 3)
        s["rock_end"] = [round(float(c), 3) for c in tr["position"]]
        s["rock_as_ground"] = sorted({w["shape"] for r in rows for w in r["wheels"][::2]
                                      if "rock" in w["shape"]})
    s["wall_s"] = round(time.perf_counter() - t0, 1)
    if name != "flat" and name != "rock":
        sprung, k, c = front_axle
        s["reference"] = fc.reference(name, speed, sprung, k, c, travel=REAL_TRAVEL,
                                      sag=sag, radius=vp.wheel_radius)[1]
        if OURS:  # the same quarter car on road_contact's effective road: the coupling alone
            one = rc.TireContact([CORNERS[0]])
            x0 = fc.X0 - 5.0
            sx, w, _ = one.envelope(fc.LANES[name][1], np.array([[x0, fc.TREAD_HALF]]),
                                    np.array([[1.0, 0.0]]), fc.EXTENT[name] + 20.0)
            s["reference_cam"] = fc.reference(
                name, speed, sprung, k, c, travel=REAL_TRAVEL, sag=sag,
                radius=vp.wheel_radius, road=(x0 + sx, w[0]), rim=rc.Corner.rim_clearance,
                damper=CORNERS[0].damper)[1]
    out["cases"][key] = s
    print(key, json.dumps(s), flush=True)
CIRCLES = [(8.0, 0.12), (11.0, 0.12), (13.0, 0.12)]
for speed, steer in CIRCLES:
    key = f"circle@{speed:g}"
    if ONLY and "circle" not in ONLY and key not in ONLY:
        continue
    out["cases"][key] = circle(speed, steer)
    print(key, json.dumps(out["cases"][key]), flush=True)
ci = [out["cases"][f"circle@{v:g}"] for v, _ in CIRCLES if f"circle@{v:g}" in out["cases"]]
if len(ci) > 1:  # the roll gradient: the slope of roll over lateral g
    ay = np.array([c["ay"] for c in ci]) / 9.81
    out["roll_gradient_deg_per_g"] = round(float(abs(np.polyfit(ay, [c["roll_deg"] for c in ci],
                                                              1)[0])), 3)
for name, speed in BRAKES:
    key = f"{name}@{speed:g}brake"
    if ONLY and key not in ONLY:
        continue
    rows = drive(name, speed, brake=True)
    br = [r for r in rows if r["x"] + vp.wheel_base >= fc.X0 - BRAKE_AT]
    x0 = br[0]["x"] if br else float("nan")
    ax = np.array([r["ax"] for r in br])
    out["cases"][key] = {
        "stop_m": round(br[-1]["x"] - x0, 2) if br else None,
        "ax_min": round(float(ax.min()), 2), "ax_max": round(float(ax.max()), 2),
        # the largest step-to-step change of the deceleration: a kick
        "ax_jump": round(float(np.abs(np.diff(ax)).max()), 2) if len(ax) > 1 else None,
        "fl_slip_min": round(float(min(r["slip"][0] for r in br if r["slip"])), 3),
        "fl_airborne_s": round(float(sum(DT for r in br if OURS and r["tl"][0] <= 0)), 3),
        "rim_strike_mm": round(float(max(max(r.get("rim", [0])) for r in br)), 1)}
    print(key, json.dumps(out["cases"][key]), flush=True)
st = sorted(step_s)
out["physx_step_ms"] = {"p50": round(st[len(st) // 2] * 1e3, 3),
                        "p95": round(st[int(len(st) * 0.95)] * 1e3, 3), "n": len(st)}
if road_ms:
    rm = sorted(road_ms)
    out["road_contact_ms"] = {"p50": round(rm[len(rm) // 2] * 1e3, 3),
                              "p95": round(rm[int(len(rm) * 0.95)] * 1e3, 3), "n": len(rm)}
print("SPIKE-RESULT " + json.dumps(out))
app.close()
