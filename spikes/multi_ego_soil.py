"""Several egos on one soil zone (roadmap 052 step 5).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/multi_ego_soil.py \\
        [<Town01 world.yaml> [<surfaces.yaml>]]
    # MULTI_SENSORS=0: the rig's sensors off; MULTI_CYCLES: the memory cycles (default 8)
    # prints one SPIKE-RESULT json

Several aib egos in one world server, on one zone's soil grid, without Autoware: one world
server hosts one ego ROS domain today (`EgoRos`, 039 makes one per ego), so the egos here
share it, and each gets its own command (a speed loop to 4 m/s), set on its vehicle as the
vehicle interface would. Default ground: `soil_steps_dirt.yaml`, 88 m of dirt along Town01's
road at y -130, two lanes of it (y -129.6 and -132.6), the egos heading -x from x 322.

- **Frame time**: one, two and four egos through the zone: the world server's frame (3
  substeps of 1/60 s, a render on the last when a sensor renders), p50 and p95, and the share
  of it in the soil layer (`SurfaceLayer` update, place, begin_step and flush).
- **Ruts felt by the others**: two egos in one lane 12 m apart (the second in the first's
  ruts) against two side by side (both on fresh soil): each ego's wheel sinkage, soil
  resistance and drive over the stretch x 255-300.
- **Memory over a batch**: four egos spawned, driven 10 s and despawned, the soil made fresh,
  `MULTI_CYCLES` times: the process's RSS and its GPU memory after each cycle.
"""
import json
import os
import statistics
import subprocess
import sys
import time

os.environ.pop("DISPLAY", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
import copy  # noqa: E402
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
SURFACES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "soil_steps_dirt.yaml")
SENSORS = os.environ.get("MULTI_SENSORS", "1") != "0"
CYCLES = int(os.environ.get("MULTI_CYCLES", "8"))
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
SUBSTEPS, DT = 3, 1 / 60
FRAME = SUBSTEPS * DT
NORTH, SOUTH = -129.6, -132.6
X0, GAP = 322.0, 12.0
SPEED = 4.0
STRETCH = (255.0, 300.0)
END_X = 245.0

with open(WORLD) as f:
    w = yaml.safe_load(f)
usd, ground = os.path.realpath(w["usd"]), list(w.get("ground_paths") or [])
assert register([WORLD])
world = IsaacWorld(ego_config=EGO_YAML, ego_domain=int(os.environ.get("STEP_DOMAIN", "77")),
                   surface_maps={usd: (SURFACES, tuple(w.get("map_origin") or (0.0, 0.0, 0.0)))})
if not SENSORS:
    for s in world.ego_rig.sensors.values():
        s.enabled = False
world.preload(usd)
world.load(usd, ground)
from aib.control import GEAR_DRIVE  # noqa: E402

surface = world.surface
soil_time = [0.0]


def timed(name):
    inner = getattr(surface, name)

    def call(*a, **k):
        t = time.perf_counter()
        try:
            return inner(*a, **k)
        finally:
            soil_time[0] += time.perf_counter() - t
    setattr(surface, name, call)


for name in ("update", "place", "begin_step", "flush"):
    timed(name)

next_handle = [100]


def own_command(ego):
    """The egos share one EgoRos and so one `cmd`: give this one its own."""
    cmd = copy.copy(ego.ros.cmd)

    def before_substep(dt, e=ego, c=cmd):
        e.vehicle.apply(c, dt, e.state)
        for s in e.sensors:
            s.before_substep(e.vehicle, e.state, dt)
    ego.before_substep = before_substep
    return cmd


def spawn(starts):
    egos = []
    for x, y in starts:
        next_handle[0] += 1
        world.spawn(next_handle[0], f"ego{len(egos)}", "EGO", "", (1.5, 0.0, 0.9),
                    (4.8, 2.1, 1.5), ((x, y, 0.3), (0.0, 0.0, 0.0, 1.0)))  # heading -x
        ego = world.egos[next_handle[0]]
        egos.append((next_handle[0], ego, own_command(ego)))
    for _ in range(20):  # settle on the hold brake
        world.step({}, SUBSTEPS, DT, False)
    return egos


def despawn(egos):
    for handle, _, _ in egos:
        world.despawn(handle)


def wheel_soil(ego):
    """This frame's (mean sinkage m, total resistance N) under the ego's wheels."""
    outs = [v[0] for k, v in surface.soil_out.items() if k[0] == ego.path]
    if not outs:
        return None
    return sum(o.sinkage for o in outs) / len(outs), sum(o.resistance for o in outs)


def drive(egos, max_t=40.0):
    """Every ego to SPEED, until all have passed END_X or `max_t`."""
    frames, soil_frames, t = [], [], 0.0
    rows = {h: {"sink": [], "resist": [], "accel": [], "speed": []} for h, _, _ in egos}
    while t < max_t and any(e.state.position[0] > END_X for _, e, _ in egos):
        for _, e, c in egos:
            v = e.state.velocity[0]
            c.acceleration, c.velocity = max(-3.0, min(3.0, 1.5 * (SPEED - v))), SPEED
            c.tire_angle, c.gear, c.received = 0.0, GEAR_DRIVE, True
        soil_time[0] = 0.0
        t0 = time.perf_counter()
        world.step({}, SUBSTEPS, DT, False, sim_time=t)
        frames.append(time.perf_counter() - t0)
        soil_frames.append(soil_time[0])
        t += FRAME
        for h, e, c in egos:
            if STRETCH[0] <= e.state.position[0] <= STRETCH[1]:
                ws = wheel_soil(e)
                if ws is not None:
                    r = rows[h]
                    r["sink"].append(ws[0])
                    r["resist"].append(ws[1])
                    r["accel"].append(c.acceleration)
                    r["speed"].append(e.state.velocity[0])
    return frames[10:], soil_frames[10:], rows, t


def ms(xs, q):
    xs = sorted(xs)
    return round(1000 * xs[min(len(xs) - 1, int(q * len(xs)))], 2)


def case(name, starts):
    surface.fresh_soil()
    egos = spawn(starts)
    frames, soil, rows, t = drive(egos)
    out = {"egos": len(egos), "frames": len(frames), "driven_s": round(t, 1),
           "frame_p50_ms": ms(frames, 0.5), "frame_p95_ms": ms(frames, 0.95),
           "soil_p50_ms": ms(soil, 0.5), "soil_p95_ms": ms(soil, 0.95), "egos_detail": []}
    for (h, e, _), (x, y) in zip(egos, starts):
        r = rows[h]
        mean = (lambda k: round(statistics.fmean(r[k]), 4) if r[k] else None)
        out["egos_detail"].append({"start": [x, y], "sinkage_m": mean("sink"),
                                   "resistance_n": mean("resist") and round(mean("resist")),
                                   "accel_request": mean("accel"), "speed": mean("speed"),
                                   "end_x": round(e.state.position[0], 1)})
    despawn(egos)
    print(name, json.dumps(out), flush=True)
    return out


def memory():
    pid = os.getpid()
    rss = next(int(line.split()[1]) for line in open(f"/proc/{pid}/status")
               if line.startswith("VmRSS")) // 1024
    gpu = 0
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        gpu = sum(int(m) for p, m in (l.split(", ") for l in q.stdout.splitlines() if l)
                  if int(p) == pid)
    except (OSError, ValueError):
        pass
    return rss, gpu


out = {"surfaces": SURFACES, "sensors": SENSORS, "speed": SPEED, "stretch": STRETCH}
out["one"] = case("one", [(X0, NORTH)])
out["column2"] = case("column2", [(X0, NORTH), (X0 - GAP, NORTH)])
out["side2"] = case("side2", [(X0, NORTH), (X0, SOUTH)])
out["four"] = case("four", [(X0, NORTH), (X0, SOUTH), (X0 - GAP, NORTH), (X0 - GAP, SOUTH)])
out["memory"] = [memory()]
for i in range(CYCLES):
    surface.fresh_soil()
    egos = spawn([(X0, NORTH), (X0, SOUTH), (X0 - GAP, NORTH), (X0 - GAP, SOUTH)])
    drive(egos, max_t=10.0)
    despawn(egos)
    for _ in range(5):
        world.step({}, SUBSTEPS, DT, True)
    out["memory"].append(memory())
    print("memory", i, out["memory"][-1], flush=True)
print("SPIKE-RESULT " + json.dumps(out), flush=True)
app.close()
