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
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402

import fixture_course as fc  # noqa: E402
from aib.control import GEAR_DRIVE, Command, State  # noqa: E402
from aib.isaac.vehicle import Vehicle, ensure_vehicle_scene  # noqa: E402
from aib.rig import EgoConfig  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/road_fixtures", time.strftime("%Y-%m-%d_%H-%M-%S"))
QUERY = os.environ.get("ROAD_QUERY", "raycast")
LIMIT = os.environ.get("ROAD_LIMIT", "0") == "1"
RATE = float(os.environ.get("ROAD_RATE", "60"))
ONLY = [c for c in os.environ.get("ROAD_ONLY", "").split(",") if c]
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
    xs, ys, hz = fc.lane_grid(name)
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

world.reset()
world.play()
physx = omni.physx.get_physx_interface()
state = State()
cmd = Command(received=True, gear=GEAR_DRIVE)
step_s = []


def substep():
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


def drive(name, speed, offset=0.0):
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
    for _ in range(20 * SUB):
        cmd.velocity, cmd.acceleration = 0.0, 0.0
        substep()
    rows, t = [], 0.0
    lo, hi = fc.X0 - 5.0, fc.X0 + fc.EXTENT[name] + 12.0
    while True:
        front = state.position[0] + vp.wheel_base
        cmd.velocity = speed
        cmd.acceleration = max(-3.0, min(3.0, 1.5 * (speed - state.velocity[0])))
        if front > hi or t > 60.0:
            break
        z_prev = state.position[2]
        substep()
        t += DT
        if front >= lo:
            roll, pitch = euler(state.quat)
            rows.append({"t": round(t, 4), "x": round(state.position[0], 4),
                         "z": round(state.position[2], 4), "vz_w": round(
                             (state.position[2] - z_prev) / DT, 4),
                         "roll": round(roll, 3), "pitch": round(pitch, 3),
                         "vx": round(state.velocity[0], 3),
                         "wheels": [wheel_row(wp) for wp in veh.wheel_paths]})
    with open(os.path.join(OUT, f"{name}@{speed:g}+{offset:g}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def summary(name, speed, rows):
    """The front-left wheel and the body over the fixture, in the reference's terms."""
    pre = [r for r in rows if r["x"] + vp.wheel_base < fc.X0 - 2.0]
    static = float(np.mean([r["wheels"][0]["load"] for r in pre])) if pre else float("nan")
    z0 = float(np.mean([r["z"] for r in pre])) if pre else rows[0]["z"]
    load = np.array([r["wheels"][0]["load"] for r in rows])
    fx = np.array([r["x"] + vp.wheel_base for r in rows]) - fc.X0
    rise = np.nonzero(load > static * 1.05)[0]
    vz = np.array([r["vz_w"] for r in rows])
    az = np.diff(vz) / DT if len(vz) > 1 else np.zeros(1)
    t_off = sum(DT for r in rows if not r["wheels"][0]["on"])
    hits = [r["wheels"][0]["hit"][0] - fc.X0 for r in rows
            if r["wheels"][0]["on"] and -0.5 < r["wheels"][0]["hit"][0] - fc.X0 < 1.0]
    tops = [r["wheels"][0]["hit"][2] for r in rows if r["wheels"][0]["on"]]
    return {"static_N": round(static, 1), "peak_N": round(float(load.max()), 1),
            "min_N": round(float(load.min()), 1),
            "first_rise_x": None if not len(rise) else round(float(fx[rise[0]]), 3),
            "body_acc_peak": round(float(np.abs(az).max()), 2),
            "airborne_s": round(t_off, 3), "body_rise_m": round(float(max(r["z"] for r in rows)
                                                                      - z0), 3),
            "pitch_max": round(float(max(abs(r["pitch"]) for r in rows)), 2),
            "fl_hit_max_z": round(float(max(tops)), 3) if tops else None,
            "fl_hits_near": [round(h, 3) for h in hits][:12]}


CASES = [("flat", 20.0, 0.0), ("cleat_05", 10.0, 0.0), ("cleat_10", 10.0, 0.0),
         ("cleat_15", 10.0, 0.0), ("pothole", 10.0, 0.0), ("washboard", 10.0, 0.0),
         ("washboard", 20.0, 0.0), ("kicker", 15.0, 0.0), ("boulder", 10.0, 0.0),
         ("rock", 5.0, 0.0)] + [("cleat_10", 20.0, 0.07 * k) for k in range(5)]
out = {"query": QUERY, "limit_expansion": LIMIT, "physx_hz": RATE, "substeps_per_frame": SUB,
       "vehicle_substeps": vp.physx_substeps, "wheel_state_keys": sorted(K),
       "rig": {"front": front_axle, "rear": rear_axle, "sag": sag,
               "travel": vp.suspension_travel, "wheel_radius": vp.wheel_radius,
               "wheel_mass": vp.wheel_mass, "wheel_base": vp.wheel_base},
       "lane_vertices": verts, "cases": {}}
for name, speed, offset in CASES:
    key = f"{name}@{speed:g}+{offset:g}"
    if ONLY and name not in ONLY and key not in ONLY:
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
        s["reference"] = fc.reference(name, speed, sprung, k, c, travel=vp.suspension_travel,
                                      sag=sag, radius=vp.wheel_radius)[1]
    out["cases"][key] = s
    print(key, json.dumps(s), flush=True)
st = sorted(step_s)
out["physx_step_ms"] = {"p50": round(st[len(st) // 2] * 1e3, 3),
                        "p95": round(st[int(len(st) * 0.95)] * 1e3, 3), "n": len(st)}
print("SPIKE-RESULT " + json.dumps(out))
app.close()
