"""Crossing old ruts on a town (roadmap 054's acceptance): Town01's SimReady world with a wet mud
zone that has its own ground (`ground_z`: render tiles, the town cut out under it, each wheel
on a plate fitted to the ruts under it), aib's Model 3 without Autoware.

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/rut_town.py \\
        [<out dir>]                      # default play_log/rut_town/<time>
    # ANGLES=6,3: the crossing angles, deg; prints one SPIKE-RESULT json, frames to <out>/*.jsonl

What `manoeuvres.py`'s `rut_cross` cannot show: there the wheels stay on the flat pad (no
`ground_z`), so a rut is felt only through T2's side force. Here a wheel drops into a rut and
climbs its wall, as PhysX sees the plates.

The road is town01_ego_drive's, the lane at y -130 heading -x; the zone (`rut_town.yaml`) is
x 136.0..177.6, y -134.4..-128.0. Per angle: the soil made fresh, a pass at CROSS_SPEED along
the lane lays ruts (y -130 +- 0.8, the track), then the car crosses them on a straight line at
that angle, from the far lane into the zone at x 176 and reaching y -129.6 at x 138, the
pure-pursuit driver on the line; then the same crossing on fresh soil. While the car is in the
zone, per crossing: the most it is pushed off the line, its yaw rate, the steering the driver
needed, the soil's largest side force on a wheel, and how far the wheels dropped below the
zone's ground.
"""
import json
import math
import os
import sys
import time

os.environ.pop("DISPLAY", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]
                     + os.environ.get("ISAAC_KIT_ARGS", "").split()})  # the world server's host setting
import copy  # noqa: E402
import logging  # noqa: E402

import yaml  # noqa: E402

from swb_isaac_world.world import IsaacWorld  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rut_town")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/rut_town", time.strftime("%Y-%m-%d_%H-%M-%S"))
WORLD_FILE = os.path.join(REPO, "data/carla-autoware-bridge/Town01/isaac/world.yaml")
SURFACES = os.path.join(HERE, "rut_town.yaml")
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
SUBSTEPS, DT = 3, 1 / 60
FRAME = SUBSTEPS * DT
LANE_Y, TRACK = -130.0, 0.8  # the lane, and its ruts at LANE_Y +- TRACK
X_SPAWN, X_IN, X_END, Y_END = 215.0, 176.0, 138.0, -129.6
X_ZONE = (136.0, 177.6)
CROSS_SPEED = 20 / 3.6
ANGLES = [float(a) for a in os.environ.get("ANGLES", "6,3").split(",")]
HEADING = (0.0, 0.0, 0.0, 1.0)  # w, x, y, z: -x

os.makedirs(OUT, exist_ok=True)
with open(WORLD_FILE) as f:
    wf = yaml.safe_load(f)
usd = os.path.realpath(wf["usd"])
world = IsaacWorld(ego_config=EGO_YAML, ego_domain=int(os.environ.get("STEP_DOMAIN", "77")),
                   surface_maps={usd: (SURFACES, tuple(wf.get("map_origin", (0, 0, 0))))})
for s in world.ego_rig.sensors.values():
    s.enabled = False
L = world.ego_rig.vehicle.wheel_base
t0 = time.monotonic()
world.load(usd, wf["ground_paths"])
log.info("Town01 loaded in %.0f s", time.monotonic() - t0)
from aib.control import GEAR_DRIVE  # noqa: E402

surface = world.surface
handle = [100]


def spawn(y):
    handle[0] += 1
    world.spawn(handle[0], "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5),
                ((X_SPAWN, y, 0.3), HEADING))
    ego = world.egos[handle[0]]
    cmd = copy.copy(ego.ros.cmd)

    def before_substep(dt, e=ego, c=cmd):
        e.vehicle.apply(c, dt, e.state)
    ego.before_substep = before_substep
    cmd.gear, cmd.received = GEAR_DRIVE, True
    for _ in range(20):
        world.step({}, SUBSTEPS, DT, False)
    return ego, cmd


def yaw_of(q):
    w, qx, qy, qz = q
    return math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def pursue(x, y, yaw, tx, ty):
    """Pure pursuit from the rear axle to the point (tx, ty): the bicycle tire angle."""
    a = math.atan2(ty - y, tx - x) - yaw
    ld = max(math.hypot(tx - x, ty - y), 1.0)
    return math.atan2(2 * L * math.sin(a), ld)


def drive(line, label):
    """One pass at CROSS_SPEED heading -x along y = line(x), the soil left as it is."""
    ego, cmd = spawn(line(X_SPAWN))
    rows, t = [], 0.0
    yaw_prev = yaw_of(ego.state.quat)
    yaw_u = yaw_prev
    while t < 120.0:
        s = ego.state
        v, x, y = s.velocity[0], s.position[0], s.position[1]
        yaw = yaw_of(s.quat)
        yaw_u += (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_prev = yaw
        cmd.acceleration = max(-3.0, min(3.0, 1.5 * (CROSS_SPEED - v)))
        cmd.velocity = CROSS_SPEED
        look = max(6.0, v)
        cmd.tire_angle = pursue(x, y, yaw_u, x - look, line(x - look))
        if x < X_ZONE[0] - 8.0:
            break
        world.step({}, SUBSTEPS, DT, False, sim_time=t)
        t += FRAME
        veh = ego.vehicle
        cp = veh.contacts or ()
        rows.append({
            "t": round(t, 3), "phase": label, "x": round(x, 3), "y": round(y, 3),
            "yaw": round(math.degrees(yaw_u), 3),
            "r": round(math.degrees(s.angular_velocity[2]), 3), "vx": round(v, 3),
            "tire": round(math.degrees(s.tire_angle), 3),
            "line_err": round(y - line(x), 3),
            "wheel_z": [round(c.position[2], 4) if c.on_ground else None for c in cp],
            "wheel_y": [round(c.position[1], 3) if c.on_ground else None for c in cp],
            "load": [round(c.load) for c in cp],
            "side_n": [round(surface.side_force.get((veh.path, i), 0.0)) for i in range(4)]})
    world.despawn(handle[0])
    for _ in range(5):
        world.step({}, SUBSTEPS, DT, False)
    return rows


def summary(rows):
    w = [r for r in rows if X_END - 2.0 <= r["x"] <= X_IN]
    if not w:
        return {"error": "never reached the zone"}
    tire0 = sum(r["tire"] for r in w) / len(w)
    zs = [z for r in w for z in r["wheel_z"] if z is not None]
    return {"peak_line_error_m": round(max(abs(r["line_err"]) for r in w), 3),
            "peak_yaw_rate": max((r["r"] for r in w), key=abs),
            "tire_angle_swing_deg": round(max(abs(r["tire"] - tire0) for r in w), 2),
            "peak_side_force_n": max(abs(f) for r in w for f in r["side_n"]),
            "lowest_wheel_cm": round(min(zs) * 100, 2) if zs else None,
            "speed_kmh": round(3.6 * sum(r["vx"] for r in w) / len(w), 1)}


def lay():
    return drive(lambda x: LANE_Y, "lay")


result = {"out": OUT, "cases": []}
for angle in ANGLES:
    tan = math.tan(math.radians(angle))
    y_in = Y_END - tan * (X_IN - X_END)

    def crossing(x, tan=tan, y_in=y_in):
        return y_in + tan * max(0.0, X_IN - x)

    surface.fresh_soil()
    rows = {"lay": lay()}
    lay_rows = [r for r in rows["lay"] if X_END <= r["x"] <= X_IN]
    rut = surface.grids.get("mud_road")
    rows["rutted"] = drive(crossing, "rutted")
    surface.fresh_soil()
    rows["fresh"] = drive(crossing, "fresh")
    with open(os.path.join(OUT, f"cross_{angle:g}.jsonl"), "w") as f:
        for k in ("lay", "rutted", "fresh"):
            for r in rows[k]:
                f.write(json.dumps(r) + "\n")
    zs = [z for r in lay_rows for z in r["wheel_z"] if z is not None]
    out = {"case": f"cross_{angle:g}", "angle_deg": angle, "line_y_in": round(y_in, 2),
           "lay": {"lowest_wheel_cm": round(min(zs) * 100, 2) if zs else None,
                   "rut_depth_cm": None if rut is None
                   else round(rut.depth(157.0, LANE_Y - TRACK) * 100, 2)},
           "rutted": summary(rows["rutted"]), "fresh": summary(rows["fresh"])}
    print(out["case"], json.dumps(out), flush=True)
    result["cases"].append(out)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(result, f, indent=1)
print("SPIKE-RESULT " + json.dumps(result), flush=True)
os._exit(0)
