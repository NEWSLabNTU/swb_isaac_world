"""Today's model at the grip limit on mud, without Autoware.

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/limit_mud.py \\
        [<out dir> [<Town01 world.yaml> [<surfaces.yaml>]]]
    # prints one SPIKE-RESULT json; each case's frames go to <out dir>/<case>.jsonl

One aib ego, its command set here as the vehicle interface would (no ROS), on
`limit_mud.yaml`: 50 m of wet mud (x 150-200) along Town01's road at y -130, the road's
asphalt before it as the baseline. Each case speeds up to 40 km/h on asphalt, then at the
trigger x (285 on asphalt, 195 in the mud, both wheel axles on the ground under test) runs one
open-loop manoeuvre, steering and pedals fixed, as the standard tests do:

- brake: full braking (8 m/s^2 asked, ABS on), steering straight (ISO 21994's straight stop).
- split: the same with the left wheels in mud and the right on asphalt (ISO 14512).
- step: a 5 deg tire-angle step over 0.15 s, speed held (ISO 7401): 3.6 m/s^2 of lateral
  demand on the kinematic path, under asphalt's limit and over mud's.
- brake_turn: a steady 1.5 deg turn for 1 s, then full braking with the steering held
  (ISO 7975).

Every case ends when the car stops, has moved 3 m sideways (the mud's and the road's edge), or
after 8 s. Measured each frame: position, heading, yaw rate, sideslip (atan vy/vx), lateral
acceleration, and each wheel's ground, friction, slip and ABS factor.
"""
import json
import math
import os
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
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/limit_mud", time.strftime("%Y-%m-%d_%H-%M-%S"))
WORLD = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    REPO, "data/carla-autoware-bridge/Town01/isaac/world.yaml")
SURFACES = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "limit_mud.yaml")
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
SUBSTEPS, DT = 3, 1 / 60
FRAME = SUBSTEPS * DT
SPEED = 40 / 3.6
SOUTH, SPLIT = -132.6, -129.85  # all four wheels in the mud; the left two only
ASPHALT = (322.0, 285.0)  # start x, trigger x
MUD = (232.0, 195.0)
BRAKE = -8.0
STEP, STEP_RAMP, STEP_HOLD = math.radians(-5.0), 0.15, 1.2  # negative: to the right, +y
TURN, TURN_HOLD = math.radians(-1.5), 1.0
SIDEWAYS, MAX_T = 3.0, 8.0

os.makedirs(OUT, exist_ok=True)
with open(WORLD) as f:
    w = yaml.safe_load(f)
usd, ground = os.path.realpath(w["usd"]), list(w.get("ground_paths") or [])
assert register([WORLD])
world = IsaacWorld(ego_config=EGO_YAML, ego_domain=int(os.environ.get("STEP_DOMAIN", "77")),
                   surface_maps={usd: (SURFACES, tuple(w.get("map_origin") or (0.0, 0.0, 0.0)))})
for s in world.ego_rig.sensors.values():
    s.enabled = False
world.preload(usd)
world.load(usd, ground)
from aib.control import GEAR_DRIVE  # noqa: E402

surface = world.surface
handle = [100]


def spawn(x, y):
    handle[0] += 1
    world.spawn(handle[0], "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5),
                ((x, y, 0.3), (0.0, 0.0, 0.0, 1.0)))  # heading -x
    ego = world.egos[handle[0]]
    cmd = copy.copy(ego.ros.cmd)

    def before_substep(dt, e=ego, c=cmd):
        e.vehicle.apply(c, dt, e.state)
    ego.before_substep = before_substep
    cmd.gear, cmd.received = GEAR_DRIVE, True
    for _ in range(20):  # settle on the hold brake
        world.step({}, SUBSTEPS, DT, False)
    return ego, cmd


def yaw_of(q):
    w_, qx, qy, qz = q
    return math.atan2(2 * (w_ * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def row(t, phase, ego, cmd):
    s, v = ego.state, ego.vehicle
    vx, vy = s.velocity[0], s.velocity[1]
    ws = surface.wheels.get(v.path)
    dbg = v.ecu.debug
    wheels = []
    for i in range(len(v.wheel_paths)):
        g = ws.surfaces[i] if ws is not None else None
        wheels.append({"class": g.cls if g is not None else None,
                       "mu": None if ws is None or ws.mu[i] is None else round(ws.mu[i], 3),
                       "slip": None if v.wheel_states is None
                       else round(v.wheel_states[i].slip, 3)})
    return {"t": round(t, 3), "phase": phase, "x": round(s.position[0], 3),
            "y": round(s.position[1], 3), "yaw": round(math.degrees(yaw_of(s.quat)), 3),
            "r": round(math.degrees(s.angular_velocity[2]), 3), "vx": round(vx, 3),
            "vy": round(vy, 3),
            "beta": round(math.degrees(math.atan2(vy, max(abs(vx), 0.5))), 3),
            "ax": round(s.acceleration[0], 3), "ay": round(s.acceleration[1], 3),
            "tire": round(math.degrees(s.tire_angle), 3),
            "cmd_a": cmd.acceleration, "cmd_tire": round(math.degrees(cmd.tire_angle), 3),
            "abs": [round(a, 2) for a in dbg.get("abs", ())], "wheels": wheels}


def run(name, kind, start, y):
    x0, trigger = start
    surface.fresh_soil()
    ego, cmd = spawn(x0, y)
    rows, t, t_go = [], 0.0, None
    yaw0 = math.degrees(yaw_of(ego.state.quat))
    while t < 40.0:
        s = ego.state
        v = s.velocity[0]
        if t_go is None and s.position[0] <= trigger:
            t_go = t
            y_go, x_go = s.position[1], s.position[0]
        if t_go is None:
            phase = "runup"
            cmd.acceleration, cmd.velocity = max(-3.0, min(3.0, 1.5 * (SPEED - v))), SPEED
            cmd.tire_angle = 0.0
        else:
            tm = t - t_go
            if kind in ("brake", "split"):
                phase = "brake"
                cmd.acceleration, cmd.velocity, cmd.tire_angle = BRAKE, 0.0, 0.0
            elif kind == "step":
                phase = "step"
                cmd.acceleration, cmd.velocity = max(-3.0, min(3.0, 1.5 * (SPEED - v))), SPEED
                cmd.tire_angle = STEP * min(1.0, tm / STEP_RAMP)
            else:  # brake_turn
                cmd.tire_angle = TURN
                if tm < TURN_HOLD:
                    phase = "turn"
                    cmd.acceleration, cmd.velocity = \
                        max(-3.0, min(3.0, 1.5 * (SPEED - v))), SPEED
                else:
                    phase = "brake"
                    cmd.acceleration, cmd.velocity = BRAKE, 0.0
            done = (abs(s.position[1] - y_go) > SIDEWAYS or tm > MAX_T
                    or (kind == "step" and tm > STEP_HOLD)
                    or (phase == "brake" and abs(v) < 0.05))
            if done:
                break
        world.step({}, SUBSTEPS, DT, False, sim_time=t)
        t += FRAME
        rows.append(row(t, phase, ego, cmd))
    world.despawn(handle[0])
    for _ in range(5):
        world.step({}, SUBSTEPS, DT, False)
    with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return summarise(name, kind, rows, t_go, x_go, y_go, yaw0)


def summarise(name, kind, rows, t_go, x_go, y_go, yaw0):
    m = [r for r in rows if r["phase"] != "runup"]
    if not m:
        return {"case": name, "error": "never reached the trigger"}
    last = m[-1]
    go = m[0]
    out = {"case": name, "speed_at_trigger_kmh": round(3.6 * go["vx"], 1),
           "ground_at_trigger": [w["class"] for w in go["wheels"]],
           "mu_at_trigger": [w["mu"] for w in go["wheels"]],
           "duration_s": round(last["t"] - go["t"], 2),
           "distance_m": round(x_go - last["x"], 2),
           "sideways_m": round(last["y"] - y_go, 3),
           "yaw_change_deg": round(last["yaw"] - go["yaw"], 2),
           "peak_yaw_rate_deg_s": max((r["r"] for r in m), key=abs),
           "peak_sideslip_deg": max((r["beta"] for r in m), key=abs),
           "peak_ay": max((r["ay"] for r in m), key=abs),
           "end_speed_kmh": round(3.6 * last["vx"], 1)}
    brake = [r for r in m if r["phase"] == "brake"]
    if brake:
        out["mean_decel"] = round(-sum(r["ax"] for r in brake) / len(brake), 2)
        out["abs_active_share"] = round(sum(1 for r in brake if r["abs"] and min(r["abs"]) < 0.99)
                                        / len(brake), 2)
        out["max_slip"] = max((w["slip"] or 0.0 for r in brake for w in r["wheels"]), key=abs)
    if kind == "step":
        steady = [r for r in m if 0.6 <= r["t"] - go["t"] <= STEP_HOLD]
        r_meas = sum(r["r"] for r in steady) / max(1, len(steady))
        v = sum(r["vx"] for r in steady) / max(1, len(steady))
        r_kin = math.degrees(v * math.tan(STEP) / 3.005)
        out.update(yaw_rate_steady=round(r_meas, 2), yaw_rate_kinematic=round(r_kin, 2),
                   yaw_rate_ratio=round(r_meas / r_kin, 3) if r_kin else None,
                   ay_steady=round(sum(r["ay"] for r in steady) / max(1, len(steady)), 2),
                   ay_kinematic=round(v * v * math.tan(STEP) / 3.005, 2))
        rise = next((r["t"] - go["t"] for r in m if abs(r["r"]) >= 0.9 * abs(r_meas)), None)
        out["yaw_rate_90pct_s"] = rise and round(rise, 2)
    if kind == "brake_turn":
        turn = [r for r in m if r["phase"] == "turn" and r["t"] - go["t"] >= 0.5]
        out["yaw_rate_before_brake"] = round(sum(r["r"] for r in turn) / max(1, len(turn)), 2)
        if brake:
            out["yaw_rate_peak_braking"] = max((r["r"] for r in brake), key=abs)
            out["yaw_rate_end"] = brake[-1]["r"]
    print(name, json.dumps(out), flush=True)
    return out


cases = [("brake_asphalt", "brake", ASPHALT, SOUTH), ("brake_mud", "brake", MUD, SOUTH),
         ("split_mud", "split", MUD, SPLIT), ("step_asphalt", "step", ASPHALT, SOUTH),
         ("step_mud", "step", MUD, SOUTH), ("brake_turn_asphalt", "brake_turn", ASPHALT, SOUTH),
         ("brake_turn_mud", "brake_turn", MUD, SOUTH)]
only = os.environ.get("LIMIT_CASES")
result = {"surfaces": SURFACES, "out": OUT, "cases": []}
for name, kind, start, y in cases:
    if only and name not in only.split(","):
        continue
    try:
        result["cases"].append(run(name, kind, start, y))
    except Exception as e:  # noqa: BLE001
        logging.exception("case %s", name)
        result["cases"].append({"case": name, "error": repr(e)})
print("SPIKE-RESULT " + json.dumps(result), flush=True)
app.close()
