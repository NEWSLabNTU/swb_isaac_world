"""Low speed and edges on soil (roadmap 054 step 6): no oscillation, no jump.

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/low_speed.py \\
        [<out dir>]
    # CASES=edge,side,...: only these; DRIVER=p: the speed held by a proportional driver alone
    # (the default is PI, as Autoware's longitudinal controller); SLIP=real: a diagnosis, the
    # soil digging at the wheel's real slip as before docs/issues/010; SOIL=dirt: the mud and split pads of this class instead
    # of mud_wet; prints one SPIKE-RESULT json; each case's substeps go to <out dir>/<case>.jsonl

One aib ego without Autoware on manoeuvres' proving ground (`play_log/manoeuvres/
proving_ground.usda`, made by a manoeuvres run) and its pads (`proving_ground.yaml`): mud
(`mud_wet`, the soil grid, lane y 255, from x 80) and the split pad (left wheels in mud,
lane y 375). The speed is held by a PI driver (gains near Autoware's longitudinal
controller's): the drive unit is feedforward only (`m a` plus the asphalt road load it
expects), so the soil's extra resistance is the driver's to overcome, as on a real car. Every case on mud has the same on dry asphalt (lane y 15) as its baseline, since
the driver and the powertrain have low-speed behaviour of their own:

- edge_<v>: from x 60 on asphalt into the mud at v = 0.5, 1, 2, 3 m/s, speed held, straight.
- side_<v>: the split pad's lane at v, speed held: the left wheels in mud, the right not.
- turn_<v>: a 15 deg tire angle held at v = 0.5, 1, 2 m/s for 12 s, in the mud.
- ramp: the target speed from rest to 4 m/s at 0.25 m/s^2 and back to rest, 5 deg of tire angle: the speed
  passes every low-speed threshold (the lag's LAG_MIN_SPEED, the resistance's RESIST_V0,
  the brake module's slip offset) both ways.
- stop: from x 60 on asphalt into the soil at 3 m/s, braking at 2 m/s^2 to rest, held 10 s (0.0 m/s, -1.5 m/s^2 asked), then from
  rest to 3 m/s out of the wheels' own sinkage.
- rut: the stop's car gone, its ruts kept, a new car at rest 0.15 m to the left of them (each
  tire on a rut's wall), held 3 s, then from rest to 3 m/s.

Every substep is kept: speed, lateral velocity, yaw rate, acceleration, each wheel's soil
side force, resistance and sinkage. Oscillation is what is left of yaw rate, lateral velocity
and acceleration after a centred 0.5 s moving average (rms, and its zero crossings per second);
a jump is the largest change between two substeps. Both per phase, beside the baseline's.
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

from swb_isaac_world import soil  # noqa: E402
from swb_isaac_world.world import IsaacWorld  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("low_speed")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/low_speed", time.strftime("%Y-%m-%d_%H-%M-%S"))
GROUND_USD = os.path.join(REPO, "play_log/manoeuvres/proving_ground.usda")
SURFACES = os.path.join(HERE, "proving_ground.yaml")
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
SUBSTEPS, DT = 3, 1 / 60
LANES = {"dry": 15.0, "mud": 255.0, "split": 375.0}
X_EDGE = 80.0
WINDOW = 31  # substeps, the moving average's (0.5 s)
RUT_OFFSET = 0.15
DRIVER = os.environ.get("DRIVER", "pi")
if os.environ.get("SLIP") == "real":  # issue 010's defect, for the comparison
    soil.tire_slip = lambda spin, speed, floor: soil.slip_ratio(spin, speed)
KP, KI, I_MAX = (1.5, 0.0, 0.0) if DRIVER == "p" else (1.5, 0.5, 2.0)  # 1/s, 1/s^2, m/s^2

if not os.path.exists(GROUND_USD):
    sys.exit(f"{GROUND_USD} missing: run spikes/manoeuvres.py once to make it")
os.makedirs(OUT, exist_ok=True)
SOIL = os.environ.get("SOIL", "")
if SOIL:  # the pads' class swapped, into a copy of the ground's map
    import yaml
    with open(SURFACES) as f:
        m = yaml.safe_load(f)
    for z in m["zones"]:
        if z.get("soil"):
            z["class"] = SOIL
    SURFACES = os.path.join(OUT, "proving_ground.yaml")
    with open(SURFACES, "w") as f:
        yaml.safe_dump(m, f)
usd = os.path.realpath(GROUND_USD)
world = IsaacWorld(ego_config=EGO_YAML, ego_domain=int(os.environ.get("STEP_DOMAIN", "77")),
                   surface_maps={usd: (SURFACES, (0.0, 0.0, 0.0))})
for s in world.ego_rig.sensors.values():
    s.enabled = False
P = world.ego_rig.vehicle
L = P.wheel_base
world.preload(usd)
world.load(usd, ["/World/env/Ground"])
from aib.control import GEAR_DRIVE  # noqa: E402

surface = world.surface
handle = [100]


def yaw_of(q):
    w, qx, qy, qz = q
    return math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


class Run:
    """One car, its command set by `drive(run)` before every substep, every substep kept."""

    def __init__(self, x, y, drive):
        handle[0] += 1
        self.h = handle[0]
        world.spawn(self.h, "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5),
                    ((x, y, 0.3), (1.0, 0.0, 0.0, 0.0)))  # heading +x
        self.ego = world.egos[self.h]
        self.cmd = copy.copy(self.ego.ros.cmd)
        self.cmd.gear, self.cmd.received = GEAR_DRIVE, True
        self.cmd.acceleration, self.cmd.velocity = -1.5, 0.0
        self.drive, self.t, self.phase, self.rows, self.done = None, 0.0, "settle", [], False
        self.integral = 0.0
        self.ego.before_substep = self.substep
        for _ in range(20):  # settle on the hold brake
            world.step({}, SUBSTEPS, DT, False)
        self.drive, self.t0 = drive, None

    def substep(self, dt):
        e = self.ego
        if self.drive is not None:
            self.drive(self)
            self.rows.append(self.row())
            self.t += dt
        e.vehicle.apply(self.cmd, dt, e.state)

    def hold(self, target, limit=3.0):
        v = self.ego.state.velocity[0]
        self.integral = max(-I_MAX, min(I_MAX, self.integral + KI * (target - v) * DT))
        self.cmd.acceleration = max(-limit, min(limit, KP * (target - v) + self.integral))
        self.cmd.velocity = target

    def row(self):
        s, v = self.ego.state, self.ego.vehicle
        side, res, sink = [], [], []
        for i in range(len(v.wheel_paths)):
            key = (v.path, i)
            side.append(round(surface.side_force.get(key, 0.0), 1))
            so = surface.soil_out.get(key)
            res.append(round(so[0].resistance * so[1], 1) if so else 0.0)
            sink.append(round(so[0].sinkage, 4) if so else 0.0)
        dbg = v.ecu.debug
        return {"t": round(self.t, 4), "phase": self.phase, "x": round(s.position[0], 3),
                "y": round(s.position[1], 3), "yaw": round(math.degrees(yaw_of(s.quat)), 3),
                "vx": round(s.velocity[0], 4), "vy": round(s.velocity[1], 4),
                "r": round(math.degrees(s.angular_velocity[2]), 4),
                "ax": round(s.acceleration[0], 3), "ay": round(s.acceleration[1], 3),
                "tire": round(math.degrees(s.tire_angle), 3),
                "cmd_a": round(self.cmd.acceleration, 3),
                "abs": [round(a, 2) for a in dbg.get("abs", ())], "tc": dbg.get("tc"),
                "side": side, "res": res, "sink": sink,
                "slip": [round(w.slip, 4) for w in v.wheel_states] if v.wheel_states else None,
                "alpha": [round(math.degrees(w.slip_angle), 3) for w in v.wheel_states]
                if v.wheel_states else None}

    def go(self, limit_s):
        while not self.done and self.t < limit_s:
            world.step({}, SUBSTEPS, DT, False, sim_time=self.t)
        world.despawn(self.h)
        for _ in range(5):
            world.step({}, SUBSTEPS, DT, False)
        return self.rows


def edge(v, pad):
    def drive(r):
        x = r.ego.state.position[0]
        r.phase = "asphalt" if x < X_EDGE - L - 1.0 else "edge" if x < X_EDGE + 1.0 else "soil"
        r.hold(v)
        r.cmd.tire_angle = 0.0
        r.done = x > X_EDGE + 15.0
    return Run(X_EDGE - 20.0, LANES[pad], drive).go(200.0)


def side(v, pad):
    def drive(r):
        x = r.ego.state.position[0]
        r.phase = "accel" if r.t < 3.0 + 2.0 * v else "steady"
        r.hold(v)
        r.cmd.tire_angle = 0.0
        r.done = x > 120.0 + 15.0 * v
    return Run(100.0, LANES[pad], drive).go(200.0)


def turn(v, pad):
    def drive(r):
        r.phase = "accel" if r.t < 3.0 + 2.0 * v else "steady"
        r.hold(v)
        r.cmd.tire_angle = math.radians(15.0)
        r.done = r.t > 12.0 + 3.0 + 2.0 * v
    return Run(100.0, LANES[pad], drive).go(200.0)


def ramp(pad):
    def drive(r):
        r.cmd.tire_angle = math.radians(5.0)
        target = 4.0 - abs(4.0 - 0.25 * r.t)
        r.phase = "up" if r.t < 16.0 else "down" if target > 0.0 else "rest"
        if target > 0.0:
            r.hold(target)
        else:
            r.cmd.acceleration, r.cmd.velocity = -1.5, 0.0
        r.done = r.t > 37.0
    return Run(100.0, LANES[pad], drive).go(80.0)


def stop(pad):
    state = {"stopped": None}

    def drive(r):
        s = r.ego.state
        r.cmd.tire_angle = 0.0
        if r.phase in ("settle", "accel"):
            r.phase = "accel"
            r.hold(3.0)
            if s.position[0] > X_EDGE + 15.0:
                r.phase = "brake"
        if r.phase == "brake":
            r.cmd.acceleration, r.cmd.velocity = -2.0, 0.0
            if abs(s.velocity[0]) < 0.01:
                r.phase, state["stopped"] = "hold", r.t
        if r.phase == "hold":
            r.cmd.acceleration, r.cmd.velocity = -1.5, 0.0
            if r.t > state["stopped"] + 10.0:
                r.phase, r.integral = "start", 0.0
        if r.phase == "start":
            r.hold(3.0)
            if s.velocity[0] >= 2.9:
                r.phase = "after"
        if r.phase == "after":
            r.hold(3.0)
            r.done = r.t > state["stopped"] + 10.0 + 8.0
    return Run(X_EDGE - 20.0, LANES[pad], drive).go(90.0)


def rut(pad):
    def drive(r):
        r.cmd.tire_angle = 0.0
        if r.t < 3.0:
            r.phase = "hold"
            r.cmd.acceleration, r.cmd.velocity = -1.5, 0.0
        else:
            r.phase = "start" if r.ego.state.velocity[0] < 2.9 and r.phase != "after" else "after"
            r.hold(3.0)
            r.done = r.t > 12.0
    return Run(X_EDGE + 5.0, LANES[pad] + RUT_OFFSET, drive).go(40.0)


def smooth(xs, n=WINDOW):
    h, out = n // 2, []
    for i in range(len(xs)):
        w = xs[max(0, i - h):i + h + 1]
        out.append(sum(w) / len(w))
    return out


def wobble(rows, key):
    """(rms, zero crossings per s, largest step) of `key` over `rows`, its 0.5 s mean off."""
    xs = [r[key] for r in rows]
    if len(xs) < WINDOW:
        return None
    hp = [a - b for a, b in zip(xs, smooth(xs))]
    hp = hp[WINDOW // 2:-(WINDOW // 2)] or hp
    rms = math.sqrt(sum(x * x for x in hp) / len(hp))
    big = 0.25 * rms
    sig = [x for x in hp if abs(x) > big]
    crossings = sum(1 for a, b in zip(sig, sig[1:]) if a * b < 0)
    jump = max(abs(b - a) for a, b in zip(xs, xs[1:]))
    return [round(rms, 4), round(crossings / (len(hp) * DT), 2), round(jump, 4)]


def summarise(name, rows):
    out = {"case": name, "substeps": len(rows)}
    if not rows:
        return out
    phases = []
    for r in rows:
        if r["phase"] not in phases:
            phases.append(r["phase"])
    for ph in phases:
        m = [r for r in rows if r["phase"] == ph]
        o = {"s": round(len(m) * DT, 2), "v": [round(min(r["vx"] for r in m), 3),
                                              round(max(r["vx"] for r in m), 3)]}
        for key in ("r", "vy", "ax"):
            o[key] = wobble(m, key)
        o["side_max"] = round(max(abs(f) for r in m for f in r["side"]), 1)
        flips = sum(1 for a, b in zip(m, m[1:]) for i in range(4)
                    if a["side"][i] * b["side"][i] < 0 and abs(a["side"][i] - b["side"][i]) > 20)
        o["side_flips_per_s"] = round(flips / max(len(m) * DT, DT), 2)
        o["drift_m"] = round(math.hypot(m[-1]["x"] - m[0]["x"], m[-1]["y"] - m[0]["y"]), 4)
        o["yaw_change_deg"] = round(m[-1]["yaw"] - m[0]["yaw"], 3)
        o["tc_or_abs"] = sum(1 for r in m if (r["tc"] is not None and r["tc"] < 0.99)
                             or (r["abs"] and min(r["abs"]) < 0.99))
        out[ph] = o
    return out


CASES = {}
for v in (0.5, 1.0, 2.0, 3.0):
    CASES[f"edge_{v:g}"] = (lambda pad, v=v: edge(v, pad), ("mud", "dry"))
    CASES[f"side_{v:g}"] = (lambda pad, v=v: side(v, pad), ("split", "dry"))
for v in (0.5, 1.0, 2.0):
    CASES[f"turn_{v:g}"] = (lambda pad, v=v: turn(v, pad), ("mud", "dry"))
CASES["ramp"] = (ramp, ("mud", "dry"))
CASES["stop"] = (stop, ("dry", "mud"))  # mud last: rut takes its ruts
CASES["rut"] = (rut, ("mud",))  # after stop on mud: its ruts

wanted = os.environ.get("CASES", "")
groups = wanted.split(",") if wanted else None
result = {"out": OUT, "soil": SOIL or "mud_wet", "driver": DRIVER,
          "slip": os.environ.get("SLIP", "tire"), "cases": []}
for name, (fn, pads) in CASES.items():
    if groups and not any(name == g or name.startswith(g + "_") for g in groups):
        continue
    for pad in pads:
        case = f"{pad}_{name}"
        if surface is not None and name != "rut":
            surface.fresh_soil()  # rut runs on what stop left
        try:
            rows = fn(pad)
            with open(os.path.join(OUT, f"{case}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            out = summarise(case, rows)
        except Exception as e:  # noqa: BLE001
            logging.exception("case %s", case)
            out = {"case": case, "error": repr(e)}
        print(case, json.dumps(out), flush=True)
        result["cases"].append(out)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(result, f, indent=1)
print("SPIKE-RESULT " + json.dumps(result), flush=True)
app.close()
