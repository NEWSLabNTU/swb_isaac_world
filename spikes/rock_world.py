"""Roadmap 066 R4: the world server's road with rocks, as `world.py` runs it (`road_layer`
RoadLayer, MeshStream, `rock_layer` RockStream), under aib's Model 3 over 2 km of R3's class C
corridor at a held speed, Isaac at 60 Hz.

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/rock_world.py \\
        [<out dir>]
    # prints one SPIKE-RESULT json

The corridor's rocks are `rocks.Zones` with the carriageway's cover raised to ROCK_CARRIAGEWAY
(default 0.01; the forest road's own 0.002 puts almost none in the car's path), so the tires
meet rocks along the whole road. The skid plate is aib's, at the spec's ground clearance
(Tesla's 138 mm). ROCKS=0 runs
the same road with no rocks: the frame time to compare.

Per speed (ROCK_SPEEDS, km/h): frame wall time, the parts of the slowest frames, awake rocks,
rock prims, rocks under the tires, the fastest rock, rocks lost, where the car left the road or
stuck.
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
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402

from aib.control import GEAR_DRIVE, Command, State  # noqa: E402
from aib.frames import qrot  # noqa: E402
from aib.isaac.vehicle import Vehicle, ensure_vehicle_scene  # noqa: E402
from aib.rig import EgoConfig  # noqa: E402
from swb_isaac_world import road_detail as rd  # noqa: E402
from swb_isaac_world import road_layer  # noqa: E402
from swb_isaac_world import rock_layer  # noqa: E402
from swb_isaac_world import rocks as rk  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/rock_world", time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(OUT, exist_ok=True)
SPEEDS = [float(v) for v in os.environ.get("ROCK_SPEEDS", "40,60,80").split(",")]
ROCKS_ON = os.environ.get("ROCKS", "1") == "1"
CARRIAGEWAY = float(os.environ.get("ROCK_CARRIAGEWAY", "0.01"))
NO_COVER = os.environ.get("ROCK_COVER") == "0"  # the tiles and the stream, no rocks
if "ROCK_COLL_STEP" in os.environ:  # the collision tiles' spacing, in 5 cm samples
    rock_layer.COLLISION_STEP = int(os.environ["ROCK_COLL_STEP"])
ALL_STATIC = os.environ.get("ROCK_ALL_STATIC") == "1"  # no rock bodies: the prims' cost
if "ROCK_COLL_REACH" in os.environ:
    rock_layer.COLLISION_REACH = float(os.environ["ROCK_COLL_REACH"])
LENGTH = float(os.environ.get("ROCK_LENGTH", "2000"))
X_START = 20.0
FRAME, SUB = 0.05, 3
DT = FRAME / SUB
W_LAT, Z_LAT = 0.8, 0.9

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

corridor = rd.Corridor(
    np.array([[0.0, 0.0], [x1, 0.0]]), seed=66, roughness=rd.Roughness(cls="C"),
    ruts=rd.Ruts(offset=0.83), potholes=rd.Potholes(),
    washboard=(rd.Washboard(X_START + 600.0, X_START + 700.0, pitch=0.5),),
    water_bars=rd.WaterBars(at=(X_START + 1500.0, X_START + 1550.0, X_START + 1600.0)),
    rocks=(rk.Zones(carriageway=0.0, shoulder=0.0, ditch=0.0) if NO_COVER
           else rk.Zones(carriageway=CARRIAGEWAY)) if ROCKS_ON else None)

sim = omni.physx.get_physx_simulation_interface()
STAGE_ID = omni.usd.get_context().get_stage_id()


def apply_force(path, force, position):  # world.py's _apply_force
    sim.apply_force_at_pos(STAGE_ID, PhysicsSchemaTools.sdfPathToInt(path), force, position)


layer = road_layer.RoadLayer([corridor], apply_force=apply_force)
meshes = road_layer.MeshStream(stage, layer)

cfg = EgoConfig.load(os.environ.get("AIB_EGO") or os.path.join(
    HERE, "../deps/autoware_isaac_bridge/src/aib/config/ego.yaml"))
rig = cfg.load_rig()
vp = road_layer.vehicle_params(rig.vehicle)
veh = Vehicle(stage, "/World/ego", vp, rig.ecu, rig.box_center, rig.box_dims,
              ((X_START, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0)),
              ground_clearance=rig.ground_clearance)
CLEARANCE = rig.ground_clearance
assert veh.plate_path is not None, "the rig's spec gives no ground clearance"
PhysxSchema.PhysxVehicleAPI(stage.GetPrimAtPath("/World/ego")).GetSuspensionLineQueryTypeAttr() \
    .Set(PhysxSchema.Tokens.raycast)
plate = UsdGeom.Cube(stage.GetPrimAtPath(veh.plate_path))
PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath("/World/ego")).CreateThresholdAttr(
    0.0)

physx = omni.physx.get_physx_interface()
LATE = os.environ.get("ROCK_LATE") == "1"  # as the world server: the stream after play


def stream():
    layer.rock_stream = rock_layer.RockStream(stage, layer, physx,
                                              ground=["/World/env/Base/mesh"])
    if ALL_STATIC:
        layer.rock_stream._in_band = lambda p: False  # every rock in its tile's collider


if ROCKS_ON and not LATE:
    stream()
world.reset()
world.play()
if ROCKS_ON and LATE:
    stream()
state = State()
cmd = Command(received=True, gear=GEAR_DRIVE)
MU = vp.tire_friction


def substep():
    """world.py's substep for one ego: the road's contact (and its share of the friction),
    the ego, PhysX."""
    if veh.last is not None:
        layer.before_substep([veh], DT)
        for i in range(4):
            veh.set_wheel_friction(i, MU * layer.scale(veh.path, i))
    veh.apply(cmd, DT, state)
    world.step(render=False)
    veh.sample(physx, DT, state)


def pose():
    p, q = state.position, state.quat
    f = qrot(q, (1.0, 0.0, 0.0))
    return [(p[0], p[1], math.atan2(f[1], f[0]), state.velocity[0])]


def frame_streams():
    xs = [(state.position[0], state.position[1], state.position[2])]
    t0 = time.perf_counter()
    meshes.update(xs)
    t1 = time.perf_counter()
    if layer.rock_stream is not None:
        layer.rock_stream.update(pose())
    return (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3


TRACE = []  # per substep: x, y, yaw, vx, z, steer, the tires' loads (4)
CONTACTS = []  # the car's body's contacts: x, the other actor, our collider, impulse N s


def read_contacts(kmh):
    headers, data = sim.get_contact_report()
    for h in headers:
        a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
        a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        c0 = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
        c1 = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
        ours, other = (c0, a1) if a0.startswith("/World/ego") else (c1, a0)
        imp = max((math.sqrt(sum(float(c) ** 2 for c in data[k].impulse))
                   for k in range(h.contact_data_offset,
                                  h.contact_data_offset + h.num_contact_data)), default=0.0)
        if imp > 50.0:
            CONTACTS.append((round(state.position[0], 2), other, ours, round(imp, 1)))


def rest_gaps(rs):
    """Loose rocks with prims: their lowest point over the true surface (base + detail) under
    their centre, mm (p50, p95, n). Resting on the collision tiles: about 0; on the base above
    a rut or a crowned road's side: the detail's depth there."""
    g = []
    for i, p in enumerate(rs.paths):
        if p is None or not rs.dyn[i]:
            continue
        rs.read(i)
        x, y = layer.rocks.pos[i, 0], layer.rocks.pos[i, 1]
        h = float(layer.detail.height(np.array([x]), np.array([y]))[0])
        g.append((layer.rocks.bottom(i) - h) * 1e3)
    if not g:
        return None
    return [round(float(np.median(g)), 1), round(float(np.percentile(g, 95)), 1), len(g)]


PLATE_SET = []
DIAG = os.environ.get("ROCK_DIAG") == "1"


def diag(o):
    """A tire load over 30 kN: the rocks under that tire, as the contact saw them."""
    car = layer.cars[veh.path]
    k = int(np.argmax(o["load"]))
    rs, st = layer.rocks, layer.rock_stream
    xy = getattr(car, "last_xy", None)
    wx, wy = (xy[k] if xy is not None else (state.position[0], state.position[1]))
    near = rs.near(np.array([wx]), np.array([wy]), 0.6)
    road = float(layer.detail.height(np.array([wx]), np.array([wy]))[0])
    print(f"DIAG x={state.position[0]:.2f} tire {k} load {o['load'][k]:.0f} at "
          f"({wx:.2f}, {wy:.2f}) road {road:+.3f}", flush=True)
    for i in near:
        r = rs.rocks[i]
        print(f"DIAG   rock {i} {r.mass:.1f} kg r {r.radius:.2f} pos "
              f"{np.round(rs.pos[i], 3).tolist()} bottom {rs.bottom(i) - road:+.3f} "
              f"top {rs.pos[i][2] + (r.verts @ rk._rot(rs.quat[i]).T)[:, 2].max() - road:+.3f} "
              f"seen {bool(rs.seen[i])} awake {bool(st.active.get(i))} v {np.round(st.vel.get(i, np.zeros(3)), 2).tolist()}",
              flush=True)


def set_plate():
    """Where the body rests: aib puts the plate's bottom at the clearance over base_link, so
    the plate stands that much off Tesla's figure, which is over the road at rest."""
    z0 = float(state.position[2]) - float(layer.detail.height(
        np.array([state.position[0]]), np.array([state.position[1]]))[0])
    PLATE_SET.append(z0)
    print(f"PLATE origin at rest {z0 * 1e3:+.1f} mm from the road; plate bottom "
          f"{(CLEARANCE + z0) * 1e3:.0f} mm over it", flush=True)


def run(kmh):
    v = kmh / 3.6
    veh.teleport(((X_START - 60.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0)), physx)
    layer.reset(veh.path)
    for _ in range(20 * SUB):
        cmd.velocity, cmd.acceleration, cmd.tire_angle = 0.0, 0.0, 0.0
        layer.detail.prefetch(layer.detail.tiles_near([state.position[0]], [0.0], 48.0))
        frame_streams()
        substep()
    if not PLATE_SET:
        set_plate()
    frames, parts, awake = [], [], []
    left = stuck = None
    slow, vmax = 0.0, 0.0
    n_cost = len(layer.cost_ms)
    while state.position[0] < X_START + LENGTH:
        t0 = time.perf_counter()
        x = state.position[0]
        if state.position[2] < -2.0 or abs(state.position[1]) > 6.0:
            left = [round(float(c), 2) for c in state.position]
            break
        slow = slow + FRAME if abs(state.velocity[0]) < 0.3 else 0.0
        if slow > 5.0:
            stuck = [round(float(c), 2) for c in state.position]
            break
        mesh_ms, rock_ms = frame_streams()
        cmd.velocity = v
        cmd.acceleration = max(-3.0, min(3.0, 1.5 * (v - state.velocity[0])))
        _, _, yaw, _ = pose()[0]
        vs = max(v, 1.0)
        cmd.tire_angle = max(-0.05, min(0.05, -vp.wheel_base * (
            W_LAT ** 2 * state.position[1] / vs ** 2 + 2 * Z_LAT * W_LAT * yaw / vs)))
        c0 = len(layer.cost_ms)
        t1 = time.perf_counter()
        for _ in range(SUB):
            substep()
            o = layer.cars[veh.path].last_out if veh.path in layer.cars else None
            TRACE.append([state.position[0], state.position[1], pose()[0][2],
                          state.velocity[0], state.position[2], veh.tire_angle,
                          *(o["load"] if o is not None else [0.0] * 4)])
            read_contacts(kmh)
            if DIAG and o is not None and max(o["load"]) > 30e3:
                diag(o)
        steps_ms = (time.perf_counter() - t1) * 1e3
        if x > X_START + 40.0:
            frames.append((time.perf_counter() - t0) * 1e3)
            parts.append([mesh_ms, rock_ms, steps_ms, sum(layer.cost_ms[c0:])])
            rs = layer.rock_stream
            if rs is not None:
                awake.append(rs.awake())
                vmax = max([vmax] + [float(np.linalg.norm(u)) for u in rs.vel.values()])
        if len(frames) % 400 == 1:
            print(f"  {kmh:g} km/h at {x:.0f} m: {awake[-1] if awake else 0} awake", flush=True)
    np.save(os.path.join(OUT, f"trace_{kmh:g}.npy"), np.array(TRACE[-30000:]))
    TRACE.clear()
    with open(os.path.join(OUT, f"contacts_{kmh:g}.json"), "w") as fh:
        json.dump(CONTACTS[-20000:], fh)
    CONTACTS.clear()
    f = np.array(frames) if frames else np.zeros(1)
    pa = np.array(parts) if parts else np.zeros((1, 4))
    np.save(os.path.join(OUT, f"frame_ms_{kmh:g}.npy"), f)
    np.save(os.path.join(OUT, f"parts_{kmh:g}.npy"), pa)
    slowest = f >= np.percentile(f, 95)
    out = {"kmh": kmh, "frame_ms": {"p50": round(float(np.median(f)), 2),
                                    "p95": round(float(np.percentile(f, 95)), 2),
                                    "max": round(float(f.max()), 1),
                                    "over_50": int((f > 50).sum()), "n": len(frames)},
           "parts_p50_ms": dict(zip(("meshes", "rocks", "steps", "contact"),
                                    np.round(np.median(pa, axis=0), 2).tolist())),
           "parts_slowest5_ms": dict(zip(("meshes", "rocks", "steps", "contact"),
                                         np.round(pa[slowest].mean(axis=0), 1).tolist())),
           "contact_ms_p50": round(float(np.median(layer.cost_ms[n_cost:] or [0])), 3),
           "left_road_at": left, "stuck_at": stuck}
    rs = layer.rock_stream
    if rs is not None:
        lost = 0
        for i, a in rs.active.items():
            if a:
                rs.read(i)
                p = layer.rocks.pos[i]
                if p[2] < -0.5:
                    lost += 1
        out.update({"awake": {"p50": int(np.median(awake)) if awake else 0,
                              "max": int(max(awake)) if awake else 0},
                    "rocks_known": len(rs.paths), "rocks_loose": int(sum(rs.dyn)),
                    "rock_prims": sum(p is not None for p in rs.paths)
                    + sum(1 for p in rs.win.values() if p),
                    "rock_v_max": round(vmax, 2), "lost": lost,
                    "prims_made": rs.made, "prims_reused": rs.reused,
                    "rest_gap_mm": rest_gaps(rs)})
    return out


res = {"rocks": ROCKS_ON, "carriageway_k": CARRIAGEWAY, "no_cover": NO_COVER,
       "coll_step": rock_layer.COLLISION_STEP, "coll_reach": rock_layer.COLLISION_REACH,
       "runs": {}}
for kmh in SPEEDS:
    r = run(kmh)
    res["runs"][f"{kmh:g}"] = r
    print(f"{kmh:g} km/h", json.dumps(r), flush=True)
res["origin_at_rest_mm"] = round(PLATE_SET[0] * 1e3, 1) if PLATE_SET else None
print("SPIKE-RESULT " + json.dumps(res))
layer.close()
app.close()
