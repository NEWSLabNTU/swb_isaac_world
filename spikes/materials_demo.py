"""Every ground the surface layer knows, one car, three tests each, filmed (roadmap 057's demo).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/materials_demo.py \\
        [<out dir>]
    # MATERIALS=ice,mud_wet: only these pads (their keys below); TESTS=start,brake,turn
    # EGO_RIG=drift_mustang.rig.yaml (the default): the car

A flat proving ground with one 200 x 60 m pad per surface class side by side along y, each a
surface-map zone (the soil grid where the class has soil parameters) and, above the physics
ground, a coloured plate so the cameras tell them apart. The ego (aib, no Autoware, its
command set here as the vehicle interface would) runs three tests on each pad:

- start: from standstill 10 m into the pad, full drive for START_TIME, steering straight:
  wheelspin, digging in, or nothing.
- brake: up to BRAKE_SPEED on the asphalt before the pad, then full braking from the pad's
  edge, steering straight, until it stops or BRAKE_TIME: the stopping distance, locked
  wheels (drift_mustang has no ABS).
- turn: the same run-up, then a TURN_ANGLE tire-angle step at the pad's edge, the speed asked
  held, for TURN_TIME: the grip's limit, understeer, a drift or a spin.

Nothing steps in when the car slides, spins or sticks: the test runs its time. The recorder
films every test (`<out>/record/<time>-ego/`: dash, chase, course, state.jsonl), and
`<out>/tests.jsonl` has a line per test with its pad, its window in the recording's time and
its numbers; `scripts/materials_video.py` cuts the demo from them.
"""
import copy
import json
import math
import os
import sys
import time

os.environ.pop("DISPLAY", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]
                     + os.environ.get("ISAAC_KIT_ARGS", "").split()})
import logging  # noqa: E402

import yaml  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402

from swb_isaac_world.world import IsaacWorld  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("materials_demo")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/materials_demo", time.strftime("%Y-%m-%d_%H-%M-%S"))
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
FRAME, SUBSTEPS = 0.05, 3
DT = FRAME / SUBSTEPS

# (key, title, class, wetness, colour of its plate): the order the demo shows them in
PADS = [
    ("asphalt", "dry asphalt", "asphalt", None, (0.20, 0.20, 0.22)),
    ("asphalt_wet", "wet asphalt", "asphalt", "wetWithPuddles", (0.10, 0.11, 0.14)),
    ("concrete", "concrete", "concrete", None, (0.62, 0.62, 0.60)),
    ("gravel", "gravel", "gravel", None, (0.48, 0.46, 0.42)),
    ("grass", "grass", "grass", None, (0.25, 0.45, 0.15)),
    ("dirt", "dirt", "dirt", None, (0.45, 0.33, 0.20)),
    ("sand", "sand", "sand", None, (0.80, 0.70, 0.48)),
    ("sand_loose", "loose sand", "sand_loose", None, (0.88, 0.78, 0.55)),
    ("mud_wet", "wet mud", "mud_wet", None, (0.30, 0.22, 0.14)),
    ("mud_soft", "soft mud", "mud_soft", None, (0.24, 0.17, 0.10)),
    ("snow_packed", "packed snow", "snow_packed", None, (0.86, 0.88, 0.92)),
    ("snow_loose", "loose snow", "snow_loose", None, (0.97, 0.97, 1.00)),
    ("ice", "ice", "ice", None, (0.70, 0.84, 0.95)),
]
SOIL = {"dirt", "sand", "sand_loose", "mud_wet", "mud_soft", "snow_loose"}
PAD_X0, PAD_X1, PAD_W, LANE0 = 60.0, 260.0, 60.0, 30.0
EXTENT = (-100.0, 300.0, 0.0, LANE0 + PAD_W * len(PADS))
X_RUNUP, X_START = -80.0, PAD_X0 + 10.0
START_TIME = 6.0
BRAKE_SPEED, BRAKE_TIME = 50 / 3.6, 7.0
TURN_SPEED, TURN_ANGLE, TURN_TIME = 50 / 3.6, math.radians(25.0), 5.0
RUNUP_LIMIT = 40.0  # s: a car that cannot reach the speed on asphalt is a fault


def lane(i):
    return LANE0 + PAD_W * i


def make_ground(path, pads):
    """The physics ground (one flat mesh labelled Road, a triangle-mesh collider) and a plate
    per pad, 2 mm above it, rendered only."""
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())

    def quad(prim_path, x0, x1, y0, y1, z, colour, grid):
        nx, ny = max(1, int((x1 - x0) / grid)), max(1, int((y1 - y0) / grid))
        pts = [Gf.Vec3f(x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny, z)
               for j in range(ny + 1) for i in range(nx + 1)]
        idx = []
        for j in range(ny):
            for i in range(nx):
                a = j * (nx + 1) + i
                idx += [a, a + 1, a + nx + 2, a + nx + 1]
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * (nx * ny)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colour)]))
        return mesh.GetPrim()

    x0, x1, y0, y1 = EXTENT
    ground = quad("/Root/Ground", x0, x1, y0, y1, 0.0, (0.25, 0.25, 0.27), 10.0)
    UsdPhysics.CollisionAPI.Apply(ground)
    UsdPhysics.MeshCollisionAPI.Apply(ground).CreateApproximationAttr("none")
    ground.CreateAttribute("semantics:labels:wikidata_qcode",
                           Sdf.ValueTypeNames.TokenArray).Set(["Q34442"])
    UsdGeom.Xform.Define(stage, "/Root/Plates")
    for i, (key, _, _, _, colour) in enumerate(pads):
        quad(f"/Root/Plates/{key}", PAD_X0, PAD_X1, lane(i) - 25.0, lane(i) + 35.0, 0.002,
             colour, 20.0)
    stage.GetRootLayer().Save()


def surface_map(path, pads):
    zones = []
    for i, (key, _, cls, wet, _) in enumerate(pads):
        z = {"id": f"{key}_pad", "class": cls,
             "polygon": [[PAD_X0, lane(i) - 25.0], [PAD_X1, lane(i) - 25.0],
                         [PAD_X1, lane(i) + 35.0], [PAD_X0, lane(i) + 35.0]]}
        if wet:
            z["wetness"] = wet
        if cls in SOIL:
            z["soil"] = "grid"
        zones.append(z)
    with open(path, "w") as f:
        yaml.safe_dump({"allow_unvalidated": ["mud_wet", "snow_loose"],
                        "defaults": {"Road": "asphalt", "RoadLine": "asphalt",
                                     "SideWalk": "concrete", "Terrain": "grass", "Ground": "dirt"},
                        "zones": zones}, f, sort_keys=False)


os.makedirs(OUT, exist_ok=True)
keys = os.environ.get("MATERIALS")
pads = [p for p in PADS if not keys or p[0] in keys.split(",")]
tests = os.environ.get("TESTS", "start,brake,turn").split(",")
cfg = yaml.safe_load(open(EGO_YAML))
cfg["rig"] = os.path.join(os.path.dirname(EGO_YAML), os.environ.get("EGO_RIG", "drift_mustang.rig.yaml"))
EGO_YAML = os.path.join(OUT, "ego.yaml")
with open(EGO_YAML, "w") as f:
    yaml.safe_dump(cfg, f)
GROUND_USD = os.path.join(OUT, "materials_ground.usda")
SURFACES = os.path.join(OUT, "materials_surfaces.yaml")
make_ground(GROUND_USD, PADS)  # every pad's lane stays where it is, whichever run
surface_map(SURFACES, PADS)
world = IsaacWorld(physics_dt=DT, ego_config=EGO_YAML,
                   ego_domain=int(os.environ.get("STEP_DOMAIN", "77")),
                   surface_maps={os.path.realpath(GROUND_USD): (SURFACES, (0.0, 0.0, 0.0))},
                   record=os.path.join(OUT, "record"))
for s in world.ego_rig.sensors.values():
    s.enabled = False
usd = os.path.realpath(GROUND_USD)
world.preload(usd)
world.load(usd, ["/World/env/Ground"])
from aib.control import GEAR_DRIVE  # noqa: E402

surface = world.surface
L = world.ego_rig.vehicle.wheel_base
handle = [100]


def yaw_of(q):
    w, qx, qy, qz = q
    return math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def spawn(x, y):
    handle[0] += 1
    world.spawn(handle[0], "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5),
                ((x, y, 0.3), (1.0, 0.0, 0.0, 0.0)))  # heading +x
    ego = world.egos[handle[0]]
    cmd = copy.copy(ego.ros.cmd)

    def before_substep(dt, e=ego, c=cmd):
        e.vehicle.apply(c, dt, e.state)
    ego.before_substep = before_substep
    cmd.gear, cmd.received = GEAR_DRIVE, True
    return ego, cmd


def hold(cmd, v, target):
    cmd.acceleration, cmd.velocity = max(-3.0, min(3.0, 1.5 * (target - v))), target


def straight(cmd, x, y, yaw, y0):
    """Pure pursuit onto the lane y0, 15 m ahead."""
    a = math.atan2(y0 - y, 15.0) - yaw
    cmd.tire_angle = math.atan2(2 * L * math.sin(a), 15.0)


def run(i, test):
    key, title = pads[i][0], pads[i][1]
    y0 = lane(PADS.index(pads[i]))
    if surface is not None:
        surface.fresh_soil()
    ego, cmd = spawn(X_START if test == "start" else X_RUNUP, y0)
    t, t_go, x_go, rows = 0.0, None, None, []
    peak_beta, peak_sink = 0.0, 0.0
    while True:
        s = ego.state
        v, x, y = s.velocity[0], s.position[0], s.position[1]
        yaw = yaw_of(s.quat)
        if test == "start":
            if t_go is None:
                t_go, x_go = t, x
            cmd.acceleration, cmd.velocity, cmd.tire_angle = 10.0, 60.0, 0.0
            done = t - t_go >= START_TIME
        elif t_go is None:  # the run-up, on asphalt
            target = BRAKE_SPEED if test == "brake" else TURN_SPEED
            hold(cmd, v, target)
            straight(cmd, x, y, yaw, y0)
            if x >= PAD_X0 + 5.0:
                t_go, x_go = t, x
            done = t > RUNUP_LIMIT
        elif test == "brake":
            cmd.acceleration, cmd.velocity, cmd.tire_angle = -10.0, 0.0, 0.0
            done = (abs(v) < 0.05 and t - t_go > 0.5) or t - t_go >= BRAKE_TIME
        else:  # turn
            hold(cmd, v, TURN_SPEED)
            cmd.tire_angle = TURN_ANGLE * min(1.0, (t - t_go) / 0.2)
            done = t - t_go >= TURN_TIME
        if t_go is not None:
            beta = math.degrees(math.atan2(s.velocity[1], max(abs(v), 0.5)))
            peak_beta = max(peak_beta, abs(beta))
            ws = surface.wheels.get(ego.vehicle.path) if surface is not None else None
            sink = max((g or 0.0) for g in getattr(ws, "sinkage", ()) or (0.0,)) if ws else 0.0
            peak_sink = max(peak_sink, sink)
            rows.append((round(t, 3), round(x, 2), round(y, 2), round(v, 3)))
        if done:
            break
        world.step({}, SUBSTEPS, DT, True, sim_time=t)
        t += FRAME
    rec = world.recorder.dir if world.recorder is not None else None
    world.despawn(handle[0])
    for _ in range(5):
        world.step({}, SUBSTEPS, DT, False)
    out = {"pad": key, "title": title, "test": test, "record": rec,
           "t_from": t_go, "t_to": t, "peak_sideslip_deg": round(peak_beta, 1),
           "peak_sinkage_m": round(peak_sink, 3)}
    if rows:
        out["speed_end_kmh"] = round(3.6 * rows[-1][3], 1)
        if test == "start":
            out["distance_m"] = round(rows[-1][1] - x_go, 1)
        if test == "brake":
            out["stopping_distance_m"] = round(rows[-1][1] - x_go, 1)
            out["stopped"] = abs(rows[-1][3]) < 0.1
    log.info("test %s", json.dumps(out))
    with open(os.path.join(OUT, "tests.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")
    return out


results = []
for i in range(len(pads)):
    for test in tests:
        try:
            results.append(run(i, test))
        except Exception as e:  # noqa: BLE001
            logging.exception("test %s %s", pads[i][0], test)
            results.append({"pad": pads[i][0], "test": test, "error": repr(e)})
print("SPIKE-RESULT " + json.dumps({"out": OUT, "tests": results}), flush=True)
app.close()
