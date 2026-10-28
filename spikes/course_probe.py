"""A generated course (roadmap 058) in Isaac, without Autoware: load its world, put the ego on
the run's lane and drive it down by pure pursuit, filmed (C9a).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/course_probe.py \\
        <map dir> [<out dir>]
    # S_END=600 (m along the centreline to stop at), SPEED=8 (m/s), HAIRPIN_SPEED=5 (m/s
    # within HAIRPIN_NEAR m of a hairpin's apex), EGO_RIG=drift_mustang.rig.yaml

It answers C9a's questions: the world loads and renders (the recorder's cameras, the course
camera among them), the car stands on the road ribbon, not in or under the terrain, and it
can drive the lane through the hairpins. Per second it logs the car's position, height over
the road, distance off the lane and speed, and `<out>/probe.jsonl` has one line per second;
`<out>/record/` has the pictures.
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

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from swb_isaac_world.world import IsaacWorld  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("course_probe")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
MAP = os.path.realpath(sys.argv[1])
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    REPO, "play_log/course_probe", time.strftime("%Y-%m-%d_%H-%M-%S"))
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
FRAME, SUBSTEPS = 0.05, 3
DT = FRAME / SUBSTEPS
S_START = 20.0  # m, as the scenario (course.scenario.START)
S_END = float(os.environ.get("S_END", "600"))
SPEED = float(os.environ.get("SPEED", "8"))
HAIRPIN_SPEED = float(os.environ.get("HAIRPIN_SPEED", "5"))
HAIRPIN_NEAR = 30.0  # m of the centreline either side of a hairpin's apex
TIME_LIMIT = 400.0  # s of simulated time

os.makedirs(OUT, exist_ok=True)
world_cfg = yaml.safe_load(open(os.path.join(MAP, "isaac/world.yaml")))
course_yaml = yaml.safe_load(open(os.path.join(REPO, "src/swb_isaac_world/swb_isaac_world/courses",
                                               os.path.basename(MAP) + ".yaml")))
LANE_OFF = course_yaml["road"]["lane_width"] / 2  # the left lane's middle, as the scenario
c = np.load(os.path.join(MAP, "course/centreline.npz"))
hairpins = json.load(open(os.path.join(MAP, "course/centreline.json")))["hairpins"]
STEP = float(c["s"][1] - c["s"][0])
lane = np.stack([c["x"] - LANE_OFF * np.sin(c["heading"]),
                 c["y"] + LANE_OFF * np.cos(c["heading"])], axis=1)

cfg = yaml.safe_load(open(EGO_YAML))
cfg["rig"] = os.path.join(os.path.dirname(EGO_YAML), os.environ.get("EGO_RIG", "drift_mustang.rig.yaml"))
EGO_YAML = os.path.join(OUT, "ego.yaml")
with open(EGO_YAML, "w") as f:
    yaml.safe_dump(cfg, f)
usd = os.path.realpath(world_cfg["usd"])
world = IsaacWorld(physics_dt=DT, ego_config=EGO_YAML,
                   ego_domain=int(os.environ.get("STEP_DOMAIN", "77")),
                   surface_maps={usd: (os.path.join(MAP, "surfaces.yaml"), (0.0, 0.0, 0.0))},
                   record=os.path.join(OUT, "record"))
for sensor in world.ego_rig.sensors.values():
    sensor.enabled = False
t0 = time.time()
world.preload(usd)
world.load(usd, world_cfg["ground_paths"])
log.info("world loaded in %.1f s", time.time() - t0)
from aib.control import GEAR_DRIVE  # noqa: E402

WB = world.ego_rig.vehicle.wheel_base


def yaw_of(q):
    w, qx, qy, qz = q
    return math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


i0 = int(round(S_START / STEP))
h0 = float(c["heading"][i0])
world.spawn(1, "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5),
            ((float(lane[i0, 0]), float(lane[i0, 1]), float(c["z"][i0]) + 0.5),
             (math.cos(h0 / 2), 0.0, 0.0, math.sin(h0 / 2))))
ego = world.egos[1]
cmd = copy.copy(ego.ros.cmd)


def before_substep(dt, e=ego, cm=cmd):
    e.vehicle.apply(cm, dt, e.state)


ego.before_substep = before_substep
cmd.gear, cmd.received = GEAR_DRIVE, True
apex = np.array([h["s"] for h in hairpins])

t, i_near, rows, worst_off, worst_dz = 0.0, i0, [], 0.0, 0.0
result = "timeout"
while t < TIME_LIMIT:
    s = ego.state
    v = s.velocity[0]
    x, y, z = s.position
    yaw = yaw_of(s.quat)
    # the nearest lane point, searched forward from the last one
    lo, hi = max(0, i_near - 20), min(len(lane), i_near + 200)
    d = np.hypot(lane[lo:hi, 0] - x, lane[lo:hi, 1] - y)
    i_near = lo + int(np.argmin(d))
    off, dz = float(d.min()), float(z - c["z"][i_near])
    worst_off, worst_dz = max(worst_off, off), max(worst_dz, abs(dz))
    s_now = float(c["s"][i_near])
    if s_now >= S_END:
        result = "reached"
        break
    if t > 10.0 and off > 6.0:
        result = "off the lane"
        break
    near_hp = len(apex) and np.min(np.abs(apex - s_now)) < HAIRPIN_NEAR
    target = HAIRPIN_SPEED if near_hp else SPEED
    cmd.acceleration = max(-3.0, min(2.0, 1.5 * (target - v)))
    cmd.velocity = target
    look = max(5.0, 0.9 * abs(v))
    j = min(len(lane) - 1, i_near + int(look / STEP))
    a = math.atan2(lane[j, 1] - y, lane[j, 0] - x) - yaw
    a = math.remainder(a, math.tau)
    cmd.tire_angle = max(-0.6, min(0.6, math.atan2(2 * WB * math.sin(a), look)))
    if int(t / FRAME) % int(1.0 / FRAME) == 0:
        row = {"t": round(t, 1), "s": round(s_now, 1), "x": round(x, 2), "y": round(y, 2),
               "z": round(z, 2), "dz_road": round(dz, 2), "off_lane": round(off, 2),
               "speed": round(v, 2), "hairpin": bool(near_hp)}
        rows.append(row)
        log.info("probe %s", json.dumps(row))
        with open(os.path.join(OUT, "probe.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    world.step({}, SUBSTEPS, DT, True, sim_time=t)
    t += FRAME

summary = {"result": result, "t": round(t, 1), "s_reached": rows[-1]["s"] if rows else None,
           "worst_off_lane_m": round(worst_off, 2), "worst_dz_road_m": round(worst_dz, 2),
           "record": world.recorder.dir if world.recorder is not None else None}
print("SPIKE-RESULT " + json.dumps(summary), flush=True)
app.close()
