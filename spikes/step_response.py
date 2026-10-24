"""The Isaac ego's step responses on a surface map's soil (roadmap 052 step 2).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/step_response.py \\
        [<Town01 world.yaml> [<surfaces.yaml> <x> <y>]]
    # prints one SPIKE-RESULT json

On flat ground `scripts/step_response/isaac_step.py` (051 K3) and `vehicle_check` measure
these; this one runs in a world with a surface map, so the same kind of steps can start in a
soil zone. Autoware's contract (an acceleration and a tire angle, `aib.control`) set on the
ego's command as the vehicle interface would, with no Autoware and no ROS traffic:

- acceleration 1 / 2 / 3 / 10 m/s^2 from rest, 6 s each: speed at 1, 2, 4, 6 s, the mean
  acceleration over 1-3 s against the request, the peak over a 0.25 s window, the time to
  63 % of the request (or of the peak, for a request beyond it);
- deceleration 1 / 2.5 / 5 / 10 m/s^2 from 10.4 m/s (reached at 3 m/s^2): stop time and
  distance, mean deceleration;
- tire angle at 5 m/s, held by a speed loop: the angles that give CARLA's three curvatures
  in K0 with the rig's wheel base, kinematically; the steady yaw rate and its curvature. On
  the server's ground plane, not Town01: a 19 m circle does not fit between its kerbs.

The world server's frame (3 substeps of 1/60 s), sensors off. With a surface map and a start
in one of its soil zones, the same steps on that soil; the ego heads -x from <x> <y>
(default: Town01's road at y -130 from x 330, flat asphalt for 300 m). `spikes/
soil_steps_sand.yaml` and `soil_steps_dirt.yaml` put 88 m of either there (start at x 322).
"""
import json
import math
import os
import sys

os.environ.pop("DISPLAY", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
import logging  # noqa: E402

import yaml  # noqa: E402

from swb_isaac_world.surface_layer import register  # noqa: E402
from swb_isaac_world.world import IsaacWorld  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
WORLD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "data/carla-autoware-bridge/Town01/isaac/world.yaml")
SURFACES = sys.argv[2] if len(sys.argv) > 2 else None
X = float(sys.argv[3]) if len(sys.argv) > 3 else 330.0
Y = float(sys.argv[4]) if len(sys.argv) > 4 else -130.0
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
SUBSTEPS, DT = 3, 1 / 60
FRAME = SUBSTEPS * DT
START = ((X, Y, 0.3), (0.0, 0.0, 0.0, 1.0))  # heading -x
CARLA_CURVATURE = (0.0182, 0.0353, 0.0527)  # K0, steer 0.05 / 0.10 / 0.15 at 5 m/s

with open(WORLD) as f:
    w = yaml.safe_load(f)
usd, ground = os.path.realpath(w["usd"]), list(w.get("ground_paths") or [])
assert register([WORLD])
maps = {}
if SURFACES:
    maps = {usd: (SURFACES, tuple(w.get("map_origin") or (0.0, 0.0, 0.0)))}
world = IsaacWorld(ego_config=EGO_YAML, ego_domain=int(os.environ.get("STEP_DOMAIN", "77")),
                   surface_maps=maps)
for s in world.ego_rig.sensors.values():
    s.enabled = False
world.preload(usd)
world.load(usd, ground)
from aib.control import GEAR_DRIVE  # noqa: E402

wheel_base = world.ego_rig.vehicle.wheel_base
handle = [100]


def spawn(start=START):
    if world.surface is not None:
        world.surface.fresh_soil()  # each test on untouched ground, not the last one's ruts
    handle[0] += 1
    world.spawn(handle[0], "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5), start)
    ego = world.egos[handle[0]]
    for _ in range(20):  # settle on the hold brake
        world.step({}, SUBSTEPS, DT, False)
    return ego


def command(ego, accel, velocity, tire_angle=0.0):
    c = ego.ros.cmd
    c.acceleration, c.velocity, c.tire_angle = accel, velocity, tire_angle
    c.gear, c.received = GEAR_DRIVE, True


def frame(ego, trace, t):
    world.step({}, SUBSTEPS, DT, False)
    s = ego.state
    trace.append((t, s.velocity[0], s.angular_velocity[2], s.position[0], s.position[1]))
    return s.velocity[0]


def v_at(trace, t):
    return round(min(trace, key=lambda r: abs(r[0] - t))[1], 2)


def accel_step(a):
    ego = spawn()
    trace, t = [], 0.0
    command(ego, a, 40.0)
    while t < 6.0 and (not trace or trace[-1][1] < 25.0):
        t += FRAME
        frame(ego, trace, t)
    world.despawn(handle[0])
    win = max(1, round(0.25 / FRAME))
    acc = [(trace[i][1] - trace[i - win][1]) / (trace[i][0] - trace[i - win][0])
           for i in range(win, len(trace))]
    peak = max(acc)
    one_three = [r for r in trace if 1.0 <= r[0] <= 3.0]
    mean = (one_three[-1][1] - one_three[0][1]) / (one_three[-1][0] - one_three[0][0])
    target = 0.63 * min(a, peak)
    rise = next((trace[i + win][0] for i, x in enumerate(acc) if x >= target), None)
    return {"request": a, "v_1_2_4_6": [v_at(trace, s) for s in (1, 2, 4, 6)],
            "mean_1_3": round(mean, 3), "error_1_3": round(mean - a, 3),
            "peak": round(peak, 3), "rise_63_s": rise and round(rise, 2)}


def brake_step(d):
    ego = spawn()
    trace, t = [], 0.0
    command(ego, 3.0, 40.0)
    while (not trace or trace[-1][1] < 10.4) and t < 15.0:  # soft soil may not get there
        t += FRAME
        frame(ego, trace, t)
    v0, t0, x0, y0 = trace[-1][1], t, trace[-1][3], trace[-1][4]
    command(ego, -d, 0.0)
    while trace[-1][1] > 0.05 and t - t0 < 20.0:
        t += FRAME
        frame(ego, trace, t)
    world.despawn(handle[0])
    stop = t - t0
    dist = math.hypot(trace[-1][3] - x0, trace[-1][4] - y0)
    return {"request": d, "from": round(v0, 2), "stop_s": round(stop, 2),
            "distance_m": round(dist, 2),
            "mean_decel": round(v0 / stop, 3) if stop > 0 else None}


def steer_step(kappa):
    angle = math.atan(kappa * wheel_base)
    ego = spawn(((0.0, 0.0, 0.3), START[1]))
    trace, t = [], 0.0
    v = 0.0
    while t < 12.0:
        accel = max(-3.0, min(3.0, 1.5 * (5.0 - v)))
        command(ego, accel, 5.0, angle if t >= 6.0 else 0.0)
        t += FRAME
        v = frame(ego, trace, t)
    world.despawn(handle[0])
    steady = [r for r in trace if r[0] >= 10.0]
    speed = sum(r[1] for r in steady) / len(steady)
    yaw = sum(r[2] for r in steady) / len(steady)
    return {"carla_curvature": kappa, "tire_angle": round(angle, 4),
            "speed": round(speed, 3), "yaw_rate_deg_s": round(math.degrees(yaw), 2),
            "curvature": round(yaw / speed, 4)}


def rows(kind, step, values):
    out[kind] = []
    for v in values:
        out[kind].append(step(v))
        print(kind, out[kind][-1], flush=True)


out = {"wheel_base": wheel_base, "start": START[0], "surfaces": SURFACES}
rows("accel", accel_step, (1.0, 2.0, 3.0, 10.0))
rows("brake", brake_step, (1.0, 2.5, 5.0, 10.0))
if not SURFACES:
    world.load("", [])  # the ground plane
    rows("steer", steer_step, CARLA_CURVATURE)
print("SPIKE-RESULT " + json.dumps(out), flush=True)
app.close()
