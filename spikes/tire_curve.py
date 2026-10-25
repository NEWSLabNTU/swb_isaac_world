"""PhysX's tire, measured: force over load against slip, sideways and lengthways (roadmap 054
step 4a).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/tire_curve.py \\
        [<out dir>]
    # CASES=dry,dry035,mud: only these; NO_STIFFNESS=1: T1's stiffness not written; prints one SPIKE-RESULT json; each case's frames go to
    # <out dir>/<case>_<sweep>.jsonl, one row per frame with every wheel's PhysX state

T1's design (054, "T1 in detail") rests on PhysX's default tire building force linearly with
slip at the tire's stiffness C and saturating at friction x load near a slip of
2 mu F_z / C, flat beyond. This measures it, on manoeuvres' proving ground (made by a
manoeuvres run, `play_log/manoeuvres/proving_ground.usda`), one aib ego without Autoware:

- lateral: from 8 m/s (29 km/h), speed held, the tire angle ramped from 0 to 25 deg to the
  left at 4 deg/s. The front axle saturates first and its slip angle keeps growing with the
  steering, so the fronts trace the curve past the peak.
- longitudinal: from 40 km/h (30 on mud), ABS off, the deceleration asked ramped from 0 to
  10 m/s^2 over 2 s, steering straight: the wheels' slip grows to lock.

Cases: `dry` (the rig's tire on dry asphalt), `dry035` (the same ground with the rig's
tire_friction at 0.35: the same tire on a lower friction, nothing else changed), `mud` (the
mud pad, the soil layer's friction and resistance). Every frame keeps each wheel's
TIRE_FORCE, its lateral and longitudinal directions, the suspension force, both slips and
the friction PhysX used (TIRE_FRICTION), so the curve and its peak come out of the rows.
"""
import dataclasses
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
import copy  # noqa: E402
import logging  # noqa: E402

import omni.physx  # noqa: E402
from omni.physx.bindings._physx import (  # noqa: E402
    VEHICLE_WHEEL_STATE_IS_ON_GROUND, VEHICLE_WHEEL_STATE_SUSPENSION_FORCE,
    VEHICLE_WHEEL_STATE_TIRE_FORCE, VEHICLE_WHEEL_STATE_TIRE_FRICTION,
    VEHICLE_WHEEL_STATE_TIRE_LATERAL_DIRECTION, VEHICLE_WHEEL_STATE_TIRE_LATERAL_SLIP,
    VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_DIRECTION, VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_SLIP)

from swb_isaac_world.world import IsaacWorld  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tire_curve")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/tire_curve", time.strftime("%Y-%m-%d_%H-%M-%S"))
GROUND_USD = os.path.join(REPO, "play_log/manoeuvres/proving_ground.usda")
SURFACES = os.path.join(HERE, "proving_ground.yaml")
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
SUBSTEPS, DT = 3, 1 / 60
FRAME = SUBSTEPS * DT
G = 9.81
LANES = {"dry": 15.0, "dry035": 15.0, "mud": 255.0}
FRICTION = {"dry035": 0.35}
X_SPAWN, X_TRIGGER = -80.0, 100.0
STEER_RATE, STEER_MAX = math.radians(4.0), math.radians(25.0)
BRAKE_RAMP, BRAKE_MAX = 2.0, 10.0

if not os.path.exists(GROUND_USD):
    sys.exit(f"{GROUND_USD} missing: run spikes/manoeuvres.py once to make it")
os.makedirs(OUT, exist_ok=True)
usd = os.path.realpath(GROUND_USD)
world = IsaacWorld(ego_config=EGO_YAML, ego_domain=int(os.environ.get("STEP_DOMAIN", "77")),
                   surface_maps={usd: (SURFACES, (0.0, 0.0, 0.0))})
for s in world.ego_rig.sensors.values():
    s.enabled = False
BASE_RIG = world.ego_rig
P = BASE_RIG.vehicle
world.preload(usd)
world.load(usd, ["/World/env/Ground"])
from aib.control import GEAR_DRIVE  # noqa: E402

physx = omni.physx.get_physx_interface()
handle = [100]


def spawn(case):
    rig = BASE_RIG
    vehicle = dataclasses.replace(P, tire_friction=FRICTION.get(case, P.tire_friction))
    world.ego_rig = dataclasses.replace(rig, vehicle=vehicle,
                                        ecu=dataclasses.replace(rig.ecu, abs_enabled=False))
    handle[0] += 1
    world.spawn(handle[0], "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5),
                ((X_SPAWN, LANES[case], 0.3), (1.0, 0.0, 0.0, 0.0)))  # heading +x
    ego = world.egos[handle[0]]
    if os.environ.get("NO_STIFFNESS"):  # a diagnosis: T1's friction without its stiffness
        ego.vehicle.set_wheel_stiffness = lambda i, lateral, longitudinal: None
    cmd = copy.copy(ego.ros.cmd)

    def before_substep(dt, e=ego, c=cmd):
        e.vehicle.apply(c, dt, e.state)
    ego.before_substep = before_substep
    cmd.gear, cmd.received = GEAR_DRIVE, True
    for _ in range(20):
        world.step({}, SUBSTEPS, DT, False)
    return ego, cmd


def dot(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b))


def wheels(ego):
    out = []
    for wp in ego.vehicle.wheel_paths:
        w = physx.get_wheel_state(wp)
        if not w or not w[VEHICLE_WHEEL_STATE_IS_ON_GROUND]:
            out.append(None)
            continue
        f = w[VEHICLE_WHEEL_STATE_TIRE_FORCE]
        fz = math.sqrt(sum(float(c) ** 2 for c in w[VEHICLE_WHEEL_STATE_SUSPENSION_FORCE]))
        out.append({"fx": round(dot(f, w[VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_DIRECTION]), 1),
                    "fy": round(dot(f, w[VEHICLE_WHEEL_STATE_TIRE_LATERAL_DIRECTION]), 1),
                    "fz": round(fz, 1),
                    "kappa": round(float(w[VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_SLIP]), 4),
                    "alpha": round(float(w[VEHICLE_WHEEL_STATE_TIRE_LATERAL_SLIP]), 4),
                    "mu": round(float(w[VEHICLE_WHEEL_STATE_TIRE_FRICTION]), 4)})
    return out


def hold(cmd, v, target):
    cmd.acceleration, cmd.velocity = max(-3.0, min(3.0, 1.5 * (target - v))), target


def run(case, sweep):
    if world.surface is not None:
        world.surface.fresh_soil()
    ego, cmd = spawn(case)
    v_go = 8.0 if sweep == "lateral" else (30 if case == "mud" else 40) / 3.6
    rows, t, t_go = [], 0.0, None
    while t < 120.0:
        s = ego.state
        v, x = s.velocity[0], s.position[0]
        if t_go is None and x >= X_TRIGGER:
            t_go = t
        if t_go is None:
            hold(cmd, v, v_go)
            cmd.tire_angle = 0.02 * (LANES[case] - s.position[1])
        else:
            tm = t - t_go
            if sweep == "lateral":
                hold(cmd, v, v_go)
                cmd.tire_angle = min(STEER_MAX, STEER_RATE * tm)
                if tm > STEER_MAX / STEER_RATE + 1.0:
                    break
            else:
                cmd.acceleration = -BRAKE_MAX * min(1.0, tm / BRAKE_RAMP)
                cmd.velocity, cmd.tire_angle = 0.0, 0.0
                if abs(v) < 1.0:
                    break
        world.step({}, SUBSTEPS, DT, False, sim_time=t)
        t += FRAME
        if t_go is not None:
            s = ego.state
            so = world.surface.soil_under(ego.vehicle) if world.surface else []
            soil = [None if x is None else {"sink": round(x[0].sinkage, 4),
                                            "limit": round(x[0].traction_limit, 1),
                                            "load": round(x[0].load, 1), "share": round(x[1], 3)}
                    for x in so]
            rows.append({"soil": soil, "tm": round(t - t_go, 3), "v": round(s.velocity[0], 3),
                         "vy": round(s.velocity[1], 3), "r": round(s.angular_velocity[2], 4),
                         "tire": round(s.tire_angle, 4), "w": wheels(ego)})
    world.despawn(handle[0])
    for _ in range(5):
        world.step({}, SUBSTEPS, DT, False)
    with open(os.path.join(OUT, f"{case}_{sweep}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = {"case": case, "sweep": sweep, "frames": len(rows), "wheels": []}
    key, force = ("alpha", "fy") if sweep == "lateral" else ("kappa", "fx")
    for i in range(4):
        pts = [(abs(r["w"][i][key]), abs(r["w"][i][force]) / r["w"][i]["fz"], r["w"][i]["mu"],
                r["w"][i]["fz"]) for r in rows if r["w"][i] and r["w"][i]["fz"] > 500]
        if not pts:
            out["wheels"].append(None)
            continue
        top = max(pts, key=lambda p: p[1])
        last = max(pts, key=lambda p: p[0])
        out["wheels"].append({"peak_at": round(top[0], 4), "peak": round(top[1], 3),
                              "mu_at_peak": top[2], "fz_at_peak": top[3],
                              "largest_slip": round(last[0], 4), "ratio_there": round(last[1], 3),
                              "mu_there": last[2]})
    print(case, sweep, json.dumps(out), flush=True)
    return out


cases = os.environ.get("CASES", "dry,dry035,mud").split(",")
result = {"out": OUT, "rig": {"lat_stiff_max_load": P.lat_stiff_max_load,
                              "lat_stiff_value": P.lat_stiff_value,
                              "long_stiff_value": P.long_stiff_value,
                              "friction_vs_slip": P.friction_vs_slip}, "runs": []}
for case in cases:
    for sweep in ("lateral", "longitudinal"):
        try:
            result["runs"].append(run(case, sweep))
        except Exception as e:  # noqa: BLE001
            logging.exception("case %s %s", case, sweep)
            result["runs"].append({"case": case, "sweep": sweep, "error": repr(e)})
print("SPIKE-RESULT " + json.dumps(result), flush=True)
app.close()
