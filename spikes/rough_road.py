"""Roadmap 066 R3: aib's Model 3 over 2 km of generated gravel road (`road_detail`), Isaac at
60 Hz with the tire contact ours (`road_contact`, R5 and R2).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/rough_road.py \\
        [<out dir>]
    # prints one SPIKE-RESULT json

PhysX's wheel query sees a flat base; the corridor's detail (an ISO 8608 class, ROUGH_CLASS,
default C, with gravel's break; a 4 % crown, ruts and a windrow, potholes, a washboard zone at
600-700 m and three water bars at 1500-1600 m) is the tire contact's road. Its tiles are made
ahead of the car in a worker thread, and each tile near the car also becomes a render mesh
(no collision) from the same samples, removed again behind it: the streaming's cost.

Per speed (30, 45, 60, 80 km/h; ROUGH_SPEEDS): the body's vertical acceleration at 60 Hz,
ISO 2631-1 Wk weighted RMS over the run; the frame's wall time (three PhysX steps, the contact
and the streaming); the process's peak memory. The corridor's IRI is road_detail's, offline
(`corridor` here is the same road). ROUGH_TIRE=physx drives the same flat
base with PhysX alone, for the frame time without the detail.
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
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402

from aib.control import GEAR_DRIVE, Command, State  # noqa: E402
from aib.frames import qrot  # noqa: E402
from aib.isaac.vehicle import WHEELS, Vehicle, ensure_vehicle_scene  # noqa: E402
from aib.rig import EgoConfig  # noqa: E402
from swb_isaac_world import road_contact as rc  # noqa: E402
from swb_isaac_world import road_detail as rd  # noqa: E402
from swb_isaac_world import road_layer  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/rough_road", time.strftime("%Y-%m-%d_%H-%M-%S"))
OURS = os.environ.get("ROUGH_TIRE", "ours") == "ours"
CLS = os.environ.get("ROUGH_CLASS", "C")
SPEEDS = [float(v) for v in os.environ.get("ROUGH_SPEEDS", "30,45,60,80").split(",")]
LENGTH = 2000.0
X_START = 20.0
FRAME, SUB = 0.05, 3
DT = FRAME / SUB
W_LAT, Z_LAT = 0.8, 0.9  # the lateral hold's natural frequency (rad/s) and damping
RENDER_REACH = 40.0  # m ahead and behind with a render mesh
os.makedirs(OUT, exist_ok=True)

world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
base = UsdGeom.Mesh.Define(stage, "/World/env/Base/mesh")
x0, x1 = X_START - 100.0, X_START + LENGTH + 200.0  # each run starts 60 m before X_START
base.CreatePointsAttr([Gf.Vec3f(x0, -20, 0), Gf.Vec3f(x1, -20, 0), Gf.Vec3f(x1, 20, 0),
                       Gf.Vec3f(x0, 20, 0)])
base.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
base.CreateFaceVertexCountsAttr([4])
UsdPhysics.CollisionAPI.Apply(base.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(base.GetPrim()).CreateApproximationAttr("none")
if not any(pr.IsA(UsdPhysics.Scene) for pr in stage.Traverse()):
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
ensure_vehicle_scene(stage)

corridor = rd.Corridor(
    np.array([[0.0, 0.0], [x1, 0.0]]), seed=66, roughness=rd.Roughness(cls=CLS),
    ruts=rd.Ruts(offset=0.83), potholes=rd.Potholes(),
    washboard=(rd.Washboard(X_START + 600.0, X_START + 700.0, pitch=0.5),),
    water_bars=rd.WaterBars(at=(X_START + 1500.0, X_START + 1550.0, X_START + 1600.0)))
detail = rd.Detail([corridor], max_tiles=48, workers=1,
                   processes=os.environ.get("ROUGH_WORKER", "process") == "process")


def prefetch(x, v):
    """Tiles the car reaches in the next road_layer's lead, made in the worker."""
    ahead = max(road_layer.REACH, road_layer.LEAD_S * v)
    xs = np.arange(x, x + ahead, detail.tile / 2)
    detail.prefetch(detail.tiles_near(xs, np.zeros_like(xs), 6.0))


height = detail.height


def flat(x, y):
    return np.zeros(np.broadcast(x, y).shape)


meshes = road_layer.MeshStream(stage, SimpleNamespace(detail=detail, ox=0.0, oy=0.0),
                               reach=RENDER_REACH)


def stream_meshes(x):
    meshes.update([(x, 0.0)])


cfg = EgoConfig.load(os.environ.get("AIB_EGO") or os.path.join(
    HERE, "../deps/autoware_isaac_bridge/src/aib/config/ego.yaml"))
rig = cfg.load_rig()
vp = rig.vehicle
REAL_TRAVEL = vp.suspension_travel
if OURS:  # R2: PhysX's travel beyond the real one, its frame raised to match
    import dataclasses
    vp = dataclasses.replace(vp, suspension_travel=REAL_TRAVEL + 0.10)
veh = Vehicle(stage, "/World/ego", vp, rig.ecu, rig.box_center, rig.box_dims,
              ((X_START, -0.0, 0.05), (1.0, 0.0, 0.0, 0.0)))
PhysxSchema.PhysxVehicleAPI(stage.GetPrimAtPath("/World/ego")).GetSuspensionLineQueryTypeAttr() \
    .Set(PhysxSchema.Tokens.raycast)
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
sim = omni.physx.get_physx_simulation_interface()
STAGE_ID = omni.usd.get_context().get_stage_id()
BODY = PhysicsSchemaTools.sdfPathToInt("/World/ego")

world.reset()
world.play()
physx = omni.physx.get_physx_interface()
state = State()
cmd = Command(received=True, gear=GEAR_DRIVE)
z_att_prev = None
road_ms = []
rim = [0.0]
airborne = [0.0]


def road_contact():
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
    steer = (veh.tire_angle, veh.tire_angle, 0.0, 0.0)
    heading = np.array([[math.cos(yaw + a), math.sin(yaw + a)] for a in steer])
    o = contact.step(DT, height, flat, xy, heading, np.full(4, state.velocity[0]), z_att, v_att)
    scale = contact.friction_scale(o["load"], o["px_load"])
    for i in range(4):
        cz = pos[2] + att[i][2] - FRAME_Z + vp.wheel_radius
        f = (float(o["dfx"][i] * heading[i, 0]), float(o["dfx"][i] * heading[i, 1]),
             float(o["dfz"][i]))
        sim.apply_force_at_pos(STAGE_ID, BODY, f, (float(xy[i, 0]), float(xy[i, 1]), cz))
        veh.set_wheel_friction(i, vp.tire_friction * float(scale[i]))
    rim[0] = max(rim[0], float(o["rim_strike"].max()))
    airborne[0] += float(o["airborne"].mean()) * DT
    road_ms.append((time.perf_counter() - t0) * 1e3)


def substep():
    if OURS and veh.last is not None:
        road_contact()
    veh.apply(cmd, DT, state)
    world.step(render=False)
    veh.sample(physx, DT, state)


def run(kmh):
    """The road from X_START at a held `kmh`; per-frame wall times and the body's z."""
    global z_att_prev
    v = kmh / 3.6
    veh.teleport(((X_START - 60.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0)), physx)
    contact.reset()
    z_att_prev = None
    rim[0], airborne[0] = 0.0, 0.0
    for _ in range(20 * SUB):
        cmd.velocity, cmd.acceleration, cmd.tire_angle = 0.0, 0.0, 0.0
        substep()
    zs, frames, gen_wait = [], [], []
    while state.position[0] < X_START + LENGTH:
        t0 = time.perf_counter()
        x = state.position[0]
        if state.position[2] < -2.0 or abs(state.position[1]) > 15.0:
            raise RuntimeError(f"{kmh:g} km/h: the car left the road at {state.position}")
        if OURS:
            prefetch(x, v)
            stream_meshes(x)
        cmd.velocity = v
        cmd.acceleration = max(-3.0, min(3.0, 1.5 * (v - state.velocity[0])))
        # on the centreline: y'' = v^2 delta / L, held to a 0.8 rad/s, 0.9-damped response
        # whatever the speed (fixed gains ring at 20 Hz control above 30 km/h)
        fwd = qrot(state.quat, (1.0, 0.0, 0.0))
        yaw = math.atan2(fwd[1], fwd[0])
        vs = max(v, 1.0)
        cmd.tire_angle = max(-0.05, min(0.05, -vp.wheel_base * (
            W_LAT ** 2 * state.position[1] / vs ** 2 + 2 * Z_LAT * W_LAT * yaw / vs)))
        for _ in range(SUB):
            substep()
            if x > X_START + 40.0:
                zs.append(state.position[2])
        if x > X_START + 40.0:
            frames.append((time.perf_counter() - t0) * 1e3)
        if len(frames) % 200 == 1:
            print(f"  {kmh:g} km/h at {x:.0f} m (y {state.position[1]:+.2f}, yaw "
                  f"{math.degrees(yaw):+.1f} deg): {len(detail.tiles)} tiles, "
                  f"{len(meshes.meshes)} meshes", flush=True)
    z = np.array(zs)
    az = np.diff(z, 2) / DT ** 2
    vx = state.velocity[0]
    import resource
    out = {"kmh": kmh, "v_end": round(vx, 2),
           "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2),
           "tiles_made": len(detail.made_ms),
           "tile_stalls": len(detail.stall_ms),
           "tile_stall_ms": round(float(np.sum(detail.stall_ms)), 1),
           "az_rms": round(float(az.std()), 3),
           "wk_rms": round(float(rd.iso2631_wk(az - az.mean(), 1 / DT).std()), 3),
           "rim_strike_mm": round(rim[0] * 1e3, 1), "wheel_airborne_s": round(airborne[0], 3),
           "frame_ms": {"p50": round(float(np.percentile(frames, 50)), 2),
                        "p95": round(float(np.percentile(frames, 95)), 2),
                        "max": round(float(np.max(frames)), 2)}}
    if gen_wait:
        out["gen_wait"] = gen_wait
    np.save(os.path.join(OUT, f"z_{kmh:g}.npy"), z)
    np.save(os.path.join(OUT, f"frame_ms_{kmh:g}.npy"), np.array(frames))
    return out


out = {"tire": "ours" if OURS else "physx", "class": CLS, "length_m": LENGTH, "runs": {}}
for kmh in SPEEDS:
    r = run(kmh)
    out["runs"][f"{kmh:g}"] = r
    print(f"{kmh:g} km/h", json.dumps(r), flush=True)
if road_ms:
    rm = sorted(road_ms)
    out["road_contact_ms"] = {"p50": round(rm[len(rm) // 2], 3),
                              "p95": round(rm[int(len(rm) * 0.95)], 3)}
if meshes.ms:
    mm = sorted(meshes.ms)
    out["mesh_ms"] = {"p50": round(mm[len(mm) // 2], 1), "max": round(mm[-1], 1),
                      "n": len(mm)}
print("SPIKE-RESULT " + json.dumps(out))
detail.close()
app.close()
