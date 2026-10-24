"""Which PhysX tire attributes take effect when changed while the simulation plays (roadmap
054 step 1).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/tire_runtime.py \\
        [<out dir> [<Town01 world.yaml> [<surfaces.yaml>]]]
    # prints one SPIKE-RESULT json; grep the log for PhysX's complaints about the edits

One aib ego without Autoware on Town01's asphalt (`limit_mud.yaml`'s ground, before the mud),
speeding up to 40 km/h from x 322 and running one open-loop manoeuvre at x 285. Each case first
edits one attribute of every wheel's `PhysxVehicleTireAPI` on the playing stage, then compares
what the manoeuvre measures with the unedited car's:

- lateral stiffness graph x 0.25 (set at spawn, and again set mid run-up): a 5 deg step steer
  should show about four times the slip angles if it took.
- longitudinal stiffness x 0.25: braking at 4 m/s^2 asked should show about four times the
  wheel slip.
- friction-vs-slip graph halved: full braking should lose about half its deceleration (047
  found PhysX refusing this one; checked again here).
- camber stiffness: no effect expected on a car with no camber; only whether PhysX complains.

Each case spawns a fresh vehicle, so no edit carries into the next.
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
from pxr import Gf, PhysxSchema  # noqa: E402

from swb_isaac_world.surface_layer import register  # noqa: E402
from swb_isaac_world.world import IsaacWorld  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tire_runtime")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/tire_runtime", time.strftime("%Y-%m-%d_%H-%M-%S"))
WORLD = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    REPO, "data/carla-autoware-bridge/Town01/isaac/world.yaml")
SURFACES = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "limit_mud.yaml")
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
SUBSTEPS, DT = 3, 1 / 60
FRAME = SUBSTEPS * DT
SPEED = 40 / 3.6
X0, Y0, TRIGGER, MID = 322.0, -132.6, 285.0, 300.0
L = 3.005
STEP, STEP_RAMP, STEP_HOLD = math.radians(-5.0), 0.15, 1.2

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

stage = world.stage
handle = [100]


def tires(ego):
    return [PhysxSchema.PhysxVehicleTireAPI(stage.GetPrimAtPath(p))
            for p in ego.vehicle.wheel_paths]


def edit(ego, what):
    """Change `what` on every wheel's tire; returns (before, after) of the first wheel."""
    out = None
    for t in tires(ego):
        if what == "lateral":
            a = t.GetLateralStiffnessGraphAttr()
            old = a.Get()
            new = Gf.Vec2f(old[0], old[1] * 0.25)
        elif what == "longitudinal":
            a = t.GetLongitudinalStiffnessAttr()
            old = a.Get()
            new = old * 0.25
        elif what == "friction_vs_slip":
            a = t.GetFrictionVsSlipGraphAttr()
            old = a.Get()
            new = [Gf.Vec2f(p[0], p[1] * 0.5) for p in old]
        elif what == "camber":
            a = t.GetCamberStiffnessAttr()
            old = a.Get()
            new = 1000.0
        a.Set(new)
        if out is None:
            out = (str(old), str(a.Get()))
    log.info("edited %s on %s: %s -> %s", what, ego.vehicle.path, *out)
    return out


def spawn():
    handle[0] += 1
    world.spawn(handle[0], "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5),
                ((X0, Y0, 0.3), (0.0, 0.0, 0.0, 1.0)))  # heading -x
    ego = world.egos[handle[0]]
    cmd = copy.copy(ego.ros.cmd)

    def before_substep(dt, e=ego, c=cmd):
        e.vehicle.apply(c, dt, e.state)
    ego.before_substep = before_substep
    cmd.gear, cmd.received = GEAR_DRIVE, True
    for _ in range(20):
        world.step({}, SUBSTEPS, DT, False)
    return ego, cmd


def hold(cmd, v):
    cmd.acceleration, cmd.velocity = max(-3.0, min(3.0, 1.5 * (SPEED - v))), SPEED


def run(name, manoeuvre, what=None, when="spawn"):
    ego, cmd = spawn()
    edited = edit(ego, what) if what and when == "spawn" else None
    rows, t, t_go = [], 0.0, None
    while t < 40.0:
        s = ego.state
        v = s.velocity[0]
        x = s.position[0]
        if what and when == "mid" and edited is None and x <= MID:
            edited = edit(ego, what)
        if t_go is None and x <= TRIGGER:
            t_go = t
        if t_go is None:
            hold(cmd, v)
            cmd.tire_angle = 0.0
        else:
            tm = t - t_go
            if manoeuvre == "step":
                hold(cmd, v)
                cmd.tire_angle = STEP * min(1.0, tm / STEP_RAMP)
                if tm > STEP_HOLD:
                    break
            else:
                cmd.acceleration, cmd.velocity, cmd.tire_angle = manoeuvre, 0.0, 0.0
                if abs(v) < 0.5 or tm > 8.0:
                    break
        world.step({}, SUBSTEPS, DT, False, sim_time=t)
        t += FRAME
        if t_go is not None:
            s = ego.state
            ws = ego.vehicle.wheel_states
            rows.append({"tm": round(t - t_go, 3), "vx": s.velocity[0], "vy": s.velocity[1],
                         "r": s.angular_velocity[2], "ax": s.acceleration[0],
                         "ay": s.acceleration[1], "tire": s.tire_angle,
                         "slip": [ws[i].slip for i in range(4)] if ws else None})
    world.despawn(handle[0])
    for _ in range(5):
        world.step({}, SUBSTEPS, DT, False)
    with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = {"case": name, "edit": what, "when": when, "edited": edited}
    if manoeuvre == "step":
        st = [r for r in rows if 0.6 <= r["tm"] <= STEP_HOLD]
        mean = lambda k: sum(r[k] for r in st) / len(st)  # noqa: E731
        vx, vy, rr = mean("vx"), mean("vy"), mean("r")
        out.update(yaw_rate_deg_s=round(math.degrees(rr), 2),
                   yaw_rate_kinematic=round(math.degrees(vx * math.tan(STEP) / L), 2),
                   ay=round(mean("ay"), 2),
                   rear_slip_angle_deg=round(math.degrees(math.atan2(vy, vx)), 2),
                   front_slip_angle_deg=round(math.degrees(
                       math.atan2(vy + L * rr, vx) - mean("tire")), 2),
                   yaw_rate_at_0_3s=round(math.degrees(next(
                       r["r"] for r in rows if r["tm"] >= 0.3)), 2))
    else:
        br = [r for r in rows if r["tm"] >= 0.4 and r["vx"] > 2.0]
        out.update(decel=round(-sum(r["ax"] for r in br) / max(1, len(br)), 2),
                   mean_abs_slip=round(sum(abs(x) for r in br for x in r["slip"])
                                       / max(1, 4 * len(br)), 4),
                   frames=len(br))
    print(name, json.dumps(out), flush=True)
    return out


cases = [("step_base", "step", None, "spawn"),
         ("step_lateral", "step", "lateral", "spawn"),
         ("step_lateral_mid", "step", "lateral", "mid"),
         ("step_camber", "step", "camber", "spawn"),
         ("brake4_base", -4.0, None, "spawn"),
         ("brake4_longitudinal", -4.0, "longitudinal", "spawn"),
         ("brake8_base", -8.0, None, "spawn"),
         ("brake8_friction_vs_slip", -8.0, "friction_vs_slip", "spawn")]
result = {"out": OUT, "cases": []}
for name, m, what, when in cases:
    try:
        result["cases"].append(run(name, m, what, when))
    except Exception as e:  # noqa: BLE001
        logging.exception("case %s", name)
        result["cases"].append({"case": name, "error": repr(e)})
print("SPIKE-RESULT " + json.dumps(result), flush=True)
app.close()
