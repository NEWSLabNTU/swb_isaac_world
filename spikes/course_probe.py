"""A generated course (roadmap 058) in Isaac, without Autoware: load its world, put the ego on
the run's lane and drive it down by pure pursuit, filmed (C9a).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/course_probe.py \\
        <map dir> [<out dir>]
    # S_END=600 (m along the centreline to stop at), SPEED=8 (m/s), HAIRPIN_SPEED=5 (m/s
    # within HAIRPIN_NEAR m of a hairpin's apex), EGO_RIG=drift_mustang.rig.yaml

With DRIFT=<point name> (course/drift_points.yaml, e.g. H1) it drifts that hairpin with
autoware-drift's drift_controller law (no ROS, no Autoware: the lane stands in for Autoware's
trajectory): the car comes up to the point at ENTRY m/s (default 12), the law takes it from
`trigger_distance`, and the run ends DRIFT_AFTER m (default 60) past the point or when the law
hands back. LAW_PARAMS="k=v,..." overrides law.Params (angles in deg where the field is an
angle); AUTOWARE_DRIFT is the checkout (default ~/repos/autoware-drift). `<out>/drift.jsonl`
has every frame of the drift (phase, sideslip, yaw rate, distance off the lane, contacts).

With PLAN=<npz> (autoware-drift race_planner.plan's, on this course's map frame) it tracks that
minimum-time trajectory instead: a pure-pursuit run-up on the centreline to the plan's first
speed, then from the plan's first s its steering as feedforward with feedback on the CoM's
offset from the planned path, the body's yaw from the planned yaw and its yaw rate from the
planned one, and its wheel torques as the acceleration request (the ECU's own feedforward turns
m a + road load back into those torques) with feedback on the speed, never past the plan's
drive (the rear's grip: more only spins it). The drive and the brakes are eased, as a driver
lifts or comes off the pedal, while the sideslip is past the plan's: from SLIP_CUT's first angle
to none at its second (deg, default "3,10"); DRIVE_SHARE (default 1) of the plan's drive.
TRACK_GAINS="k_lat,k_yaw,k_speed,k_r" (default 0.15,0.5,1.0,0.3; k_lat at 10 m/s and below,
falling as 10 / v above); FRAME_MS, the control's frame (default 50; physics at 60 Hz or the
frame's multiple above it). `<out>/plan_track.jsonl` has every frame (with each wheel's torques
and tyre: `wheels`, see wheel_rows); the result compares the
section's time with the plan's.

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
FRAME = float(os.environ.get("FRAME_MS", "50")) / 1000   # the control's frame; physics 60 Hz or faster
SUBSTEPS = max(1, math.ceil(FRAME * 60 - 1e-9))
DT = FRAME / SUBSTEPS
S_START = 20.0  # m, as the scenario (course.scenario.START)
S_END = float(os.environ.get("S_END", "600"))
SPEED = float(os.environ.get("SPEED", "8"))
HAIRPIN_SPEED = float(os.environ.get("HAIRPIN_SPEED", "5"))
HAIRPIN_NEAR = 30.0  # m of the centreline either side of a hairpin's apex
TIME_LIMIT = 400.0  # s of simulated time
DRIFT = os.environ.get("DRIFT")
PLAN = os.environ.get("PLAN")
K_LAT, K_YAW, K_SPEED, K_R = (float(g) for g in
                              os.environ.get("TRACK_GAINS", "0.15,0.5,1.0,0.3").split(","))
SLIP_CUT = tuple(math.radians(float(a)) for a in os.environ.get("SLIP_CUT", "3,10").split(","))
DRIVE_SHARE = float(os.environ.get("DRIVE_SHARE", "1.0"))   # of the plan's drive, a grip margin
ENTRY = float(os.environ.get("ENTRY", "12"))
DRIFT_AFTER = float(os.environ.get("DRIFT_AFTER", "60"))
ANGLES = {"max_steer", "enter_sideslip", "target_sideslip", "hold_angle", "exit_lead", "beta_min",
          "beta_max", "exit_sideslip", "max_sideslip", "recover_sideslip"}

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

plan = None
if PLAN:
    plan = dict(np.load(PLAN))
    S_END = float(plan["s"][-1]) - 2.0
    spec = yaml.safe_load(open(os.path.join(REPO, "src/sim_vehicle_spec/vehicles/drift_mustang.yaml")))
    MASS = spec["inertia"]["mass"]
    RW = spec["geometry"]["wheel_radius"]
    LR = WB * spec["inertia"]["com_front_fraction"]       # base_link (rear axle) to the CoM
    res = spec["resistance"]
    DRAG = 0.5 * res["air_density"] * res["drag_coefficient"] * res["frontal_area"]
    ROLL = res["rolling_coefficient"]
    STEER_MAX = spec["geometry"]["max_steer_angle"]
    lane = np.stack([c["x"], c["y"]], axis=1)   # the run-up on the centreline, as the plan
    log.info("plan %s: s %.0f-%.0f, %.2f s", PLAN, plan["s"][0], plan["s"][-1], float(plan["time"]))
plan_rows, plan_j, plan_t0, plan_t1 = [], 0, None, None

law, drift_s, drift_rows = None, None, []
if DRIFT:
    sys.path.insert(0, os.path.join(os.environ.get("AUTOWARE_DRIFT", os.path.expanduser(
        "~/repos/autoware-drift")), "src/control/drift_controller"))
    from drift_controller import law as dl  # noqa: E402

    pts = yaml.safe_load(open(os.path.join(MAP, "course/drift_points.yaml")))["points"]
    pt = next(q for q in pts if q["name"] == DRIFT)
    params = dl.Params()
    for kv in filter(None, os.environ.get("LAW_PARAMS", "").split(",")):
        k, v = kv.split("=")
        setattr(params, k, math.radians(float(v)) if k in ANGLES else type(getattr(params, k))(float(v)))
    params.min_entry_speed = min(params.min_entry_speed, ENTRY - 1.0)
    law = dl.DriftController([dl.DriftPoint(pt["x"], pt["y"], pt["radius"], pt["name"],
                                            math.radians(pt["angle_deg"]))], params)
    d_pt = np.hypot(lane[:, 0] - pt["x"], lane[:, 1] - pt["y"])
    drift_s = float(c["s"][int(np.argmin(d_pt))])
    S_END = drift_s + DRIFT_AFTER
    log.info("drift %s at s %.1f, R %.1f, %d deg, entry %.1f m/s, params %s", DRIFT, drift_s,
             pt["radius"], pt["angle_deg"], ENTRY, os.environ.get("LAW_PARAMS", ""))

t, i_near, rows, worst_off, worst_dz = 0.0, i0, [], 0.0, 0.0
contacts_now, all_contacts = [], []


def wheel_rows(v):
    """Each wheel (fl, fr, rl, rr) after the last step: the ECU's drive and brake torques (N m),
    PhysX's tyre forces along and across (N), load (N), slip and slip angle."""
    if v.wheel_states is None or v.torques is None:
        return None
    cp = v.contacts or (None,) * len(v.wheel_states)
    return [[round(d, 1), round(b, 1), round(w.force[0], 1), round(w.force[1], 1),
             round(c.load, 1) if c is not None and c.on_ground else 0.0,
             round(w.slip, 4), round(w.slip_angle, 4)]
            for d, b, w, c in zip(v.torques.drive, v.torques.brake, v.wheel_states, cp)]


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
    if t > 10.0 and off > (12.0 if law is not None and law.phase is not dl.Phase.CRUISE else 6.0):
        result = "off the lane"
        break
    near_hp = len(apex) and np.min(np.abs(apex - s_now)) < HAIRPIN_NEAR
    target = HAIRPIN_SPEED if near_hp else SPEED
    if law is not None and drift_s - 120.0 < s_now < drift_s + 5.0:
        target = ENTRY  # the run-up: the drift's entry speed
    cmd.acceleration = max(-3.0, min(2.0, 1.5 * (target - v)))
    cmd.velocity = target
    look = max(5.0, 0.9 * abs(v))
    j = min(len(lane) - 1, i_near + int(look / STEP))
    a = math.atan2(lane[j, 1] - y, lane[j, 0] - x) - yaw
    a = math.remainder(a, math.tau)
    pursuit = max(-0.6, min(0.6, math.atan2(2 * WB * math.sin(a), look)))
    cmd.tire_angle = pursuit
    if plan is not None:
        cmd.acceleration = max(-3.0, min(2.0, 1.5 * (float(plan["speed"][0]) - v)))
        cmd.velocity = float(plan["speed"][0])
        yaw_r = float(s.angular_velocity[2])
        xc, yc = x + LR * math.cos(yaw), y + LR * math.sin(yaw)    # the CoM, as the plan
        vy_c = s.velocity[1] + yaw_r * LR
        speed_c = math.hypot(s.velocity[0], vy_c)
        lo2, hi2 = max(0, plan_j - 5), min(len(plan["s"]), plan_j + 40)
        dj = np.hypot(plan["x"][lo2:hi2] - xc, plan["y"][lo2:hi2] - yc)
        j = lo2 + int(np.argmin(dj))
        if plan_t0 is None and j > 0 and s_now >= plan["s"][0]:
            plan_t0 = t
        if plan_t0 is not None:
            plan_j = j
            ps = float(plan["psi_path"][j])
            e_lat = -(xc - plan["x"][j]) * math.sin(ps) + (yc - plan["y"][j]) * math.cos(ps)
            e_yaw = math.remainder(yaw - float(plan["yaw"][j]), math.tau)
            ahead = min(len(plan["s"]) - 1, j + 1)
            k_lat = K_LAT * 10.0 / max(10.0, speed_c)
            steer = (float(plan["delta"][ahead]) - k_lat * e_lat - K_YAW * e_yaw
                     + K_R * (float(plan["r"][j]) - yaw_r))
            cmd.tire_angle = max(-STEER_MAX, min(STEER_MAX, steer))
            push = float(np.sum(plan["torque"][j])) / RW              # the plan's wheels, N
            road = DRAG * speed_c * speed_c + ROLL * MASS * 9.81
            a_ff = (push * (DRIVE_SHARE if push > 0 else 1.0) - road) / MASS
            a = a_ff + K_SPEED * (float(plan["speed"][j]) - speed_c)
            if a_ff > 0:   # the plan drives at the rear's grip: more only spins it
                a = min(a, a_ff)
            slip_err = abs(math.atan2(vy_c, s.velocity[0])) - abs(float(plan["beta"][j]))
            # sliding more than the plan the tyres have no grip to spare: lift, or ease the
            # brakes (sliding less they have: brake as planned, or the car arrives too fast)
            a *= min(1.0, max(0.0, (SLIP_CUT[1] - slip_err) / (SLIP_CUT[1] - SLIP_CUT[0])))
            cmd.acceleration = a
            cmd.velocity = 60.0 if a > 0 else 0.0
            plan_rows.append({"t": round(t - plan_t0, 3), "s": round(float(plan["s"][j]), 1),
                              "t_plan": round(float(plan["t"][j]), 3), "e_lat": round(e_lat, 3),
                              "e_yaw_deg": round(math.degrees(e_yaw), 2), "speed": round(speed_c, 2),
                              "speed_plan": round(float(plan["speed"][j]), 2),
                              "beta_deg": round(math.degrees(math.atan2(vy_c, s.velocity[0])), 2),
                              "beta_plan_deg": round(math.degrees(float(plan["beta"][j])), 2),
                              "r": round(yaw_r, 3), "r_plan": round(float(plan["r"][j]), 3),
                              "steer_deg": round(math.degrees(cmd.tire_angle), 2),
                              "steer_plan_deg": round(math.degrees(float(plan["delta"][j])), 2),
                              "accel": round(a, 2), "x": round(xc, 2), "y": round(yc, 2),
                              "contacts": contacts_now, "wheels": wheel_rows(ego.vehicle)})
            if j >= len(plan["s"]) - 2:
                plan_t1 = t
                result = "plan done"
                break
    if law is not None:
        # the lane as Autoware's trajectory: the car's signed distance (+ left) and heading error
        hdg = float(c["heading"][i_near])
        lat = -(x - lane[i_near, 0]) * math.sin(hdg) + (y - lane[i_near, 1]) * math.cos(hdg)
        vx_b, vy_b = s.velocity[0], s.velocity[1]
        st = dl.State(x, y, yaw, vx_b, vy_b, float(s.angular_velocity[2]),
                      path_lat=lat, path_yaw=math.remainder(yaw - hdg, math.tau),
                      path_steer=pursuit, path_speed=target)
        dc = law.step(st, FRAME)
        if dc.override:
            cmd.tire_angle = dc.steer
            cmd.acceleration = dc.accel
            cmd.velocity = 60.0 if dc.accel > 0 else 0.0
        if law.phase is not dl.Phase.CRUISE or drift_rows:
            drift_rows.append({"t": round(t, 2), "s": round(s_now, 1), "phase": law.phase.value,
                               "speed": round(st.speed, 2), "beta_deg": round(math.degrees(st.beta), 1),
                               "r": round(st.r, 3), "steer_deg": round(math.degrees(cmd.tire_angle), 1),
                               "accel": round(cmd.acceleration, 2), "off_lane": round(lat, 2),
                               "x": round(x, 2), "y": round(y, 2), "contacts": contacts_now})
        if drift_rows and law.phase is dl.Phase.CRUISE and any(
                r["phase"] != "cruise" for r in drift_rows) and law.log and \
                law.log[-1][0] == "cruise":
            result = "handed back"
            break
    if int(t / FRAME) % int(1.0 / FRAME) == 0:
        row = {"t": round(t, 1), "s": round(s_now, 1), "x": round(x, 2), "y": round(y, 2),
               "z": round(z, 2), "dz_road": round(dz, 2), "off_lane": round(off, 2),
               "speed": round(v, 2), "hairpin": bool(near_hp)}
        rows.append(row)
        log.info("probe %s", json.dumps(row))
        with open(os.path.join(OUT, "probe.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    _, contacts = world.step({}, SUBSTEPS, DT, True, sim_time=t)
    contacts_now = [str(cp[2]) for cp in contacts]
    if contacts_now:
        all_contacts.extend(contacts_now)
    t += FRAME

if plan_rows:
    with open(os.path.join(OUT, "plan_track.jsonl"), "w") as f:
        for r in plan_rows:
            f.write(json.dumps(r) + "\n")
if drift_rows:
    with open(os.path.join(OUT, "drift.jsonl"), "w") as f:
        for r in drift_rows:
            f.write(json.dumps(r) + "\n")
summary = {"result": result, "t": round(t, 1), "s_reached": rows[-1]["s"] if rows else None,
           "worst_off_lane_m": round(worst_off, 2), "worst_dz_road_m": round(worst_dz, 2),
           "record": world.recorder.dir if world.recorder is not None else None,
           "contacts": sorted(set(all_contacts))[:20]}
if law is not None:
    hold = [r for r in drift_rows if r["phase"] == "hold"]
    summary.update({"drift": DRIFT, "law_log": law.log,
                    "peak_beta_deg": max((abs(r["beta_deg"]) for r in drift_rows), default=0.0),
                    "hold_s": round(len(hold) * FRAME, 2),
                    "hold_beta_deg_mean": round(float(np.mean([abs(r["beta_deg"]) for r in hold])), 1)
                    if hold else None,
                    "worst_off_lane_in_drift_m": max((abs(r["off_lane"]) for r in drift_rows), default=0.0)})
if plan is not None and plan_rows:
    took = plan_rows[-1]["t"]
    planned = plan_rows[-1]["t_plan"] - plan_rows[0]["t_plan"]
    summary.update({"plan": PLAN, "plan_time_s": round(planned, 2), "isaac_time_s": round(took, 2),
                    "time_ratio": round(took / planned, 3) if planned > 0 else None,
                    "s_tracked": [plan_rows[0]["s"], plan_rows[-1]["s"]],
                    "worst_e_lat_m": max(abs(r["e_lat"]) for r in plan_rows),
                    "rms_e_lat_m": round(float(np.sqrt(np.mean([r["e_lat"] ** 2 for r in plan_rows]))), 3),
                    "rms_beta_err_deg": round(float(np.sqrt(np.mean(
                        [(r["beta_deg"] - r["beta_plan_deg"]) ** 2 for r in plan_rows]))), 2),
                    "gains": [K_LAT, K_YAW, K_SPEED, K_R], "frame_s": FRAME, "drive_share": DRIVE_SHARE,
                    "slip_cut_deg": [round(math.degrees(a), 1) for a in SLIP_CUT]})
print("SPIKE-RESULT " + json.dumps(summary), flush=True)
app.close()
