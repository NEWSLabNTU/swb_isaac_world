"""The standard handling manoeuvres on dry asphalt, wet asphalt and mud (roadmap 054 step 2).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/manoeuvres.py \\
        [<out dir>]
    # MANOEUVRES=brake,step,...: only these manoeuvres; PADS=dry,mud: only these pads
    # ECU='{"esc_enabled": true}': the onboard computers' calibration overrides, every case
    # RIG='{"tire_friction": 0.9}': the PhysX vehicle's (VehicleParams) overrides
    # ESTIMATOR=truth: the onboard computers read PhysX's speed, slips, sideslip and friction
    #   instead of estimating them (roadmap 055's check that the move alone changed nothing)
    # prints one SPIKE-RESULT json; each case's frames go to <out dir>/<pad>_<manoeuvre>.jsonl

One aib ego without Autoware, its command set here as the vehicle interface would, on a flat
proving ground made at the first run (`<out dir>/../proving_ground.usda`: one 660 x 560 m
mesh labelled Road) with `proving_ground.yaml`'s pads: dry, wet (`wetWithPuddles`) and mud
(`mud_wet`, the closed-form wheel), and a split pad with the left wheels in mud. The car speeds
up on dry asphalt from x -80, heading +x, and runs the manoeuvre from x 100, inside the pad:

- brake: full braking (8 m/s^2 asked) from 40 km/h, steering straight (ISO 21994's stop);
  on the split pad, split-mu braking (ISO 14512).
- step: a 5 deg tire-angle step to the left over 0.15 s at 40 km/h, speed held (ISO 7401).
- brake_turn: 1.5 deg to the left for 1 s at 40 km/h, then full braking, steering held
  (ISO 7975).
- sine: FMVSS 126's sine with dwell from 80 km/h (50 on mud): 0.7 Hz, a 0.5 s
  dwell at the second peak, no acceleration asked (FMVSS releases the
  throttle; here the powertrain still covers the road load), amplitude 6.5 times the tire angle of 0.3 g on the kinematic path.
- dlc: a double lane change after ISO 3888-1's layout (3.5 m over 30 m, 25 m held, back over
  25 m), driven by a pure-pursuit driver at 80 km/h (40 on mud), speed held.
- circle: ISO 4138's constant radius: a 40 m circle to the left, the pure-pursuit driver on
  it, the speed raised 0.15 m/s^2 from 5 m/s until the car leaves the circle by 4 m.
- sweep_lat: the tire's curve sideways (054 step 4b): from 10 m/s, speed held, the tire angle
  ramped from 0 to 35 deg to the left at 5 deg/s; the front axle saturates and its slip
  angle keeps growing with the steering.
- sweep_long: the tire's curve lengthways: from 40 km/h (30 on mud), ABS off, the
  deceleration asked ramped from 0 to 15 m/s^2 (past what any ground gives) over 1 s,
  steering straight: the wheels lock.

The sweeps keep each wheel's tire force, slips, load and the surface layer's stiffness and
share of the peak (`tires` in a row), and summarise each axle's curve: the most force over load
and the slip where it was, and the force over load at the largest slip (past the peak).

The driver and the inputs are fixed, so the numbers compare one model with another (054
steps 3-5), not with a real Model 3; the kinematic references use the rig's wheel base.
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
                     + os.environ.get("ISAAC_KIT_ARGS", "").split()})  # the world server's host setting
import copy  # noqa: E402
import logging  # noqa: E402

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402

from swb_isaac_world.world import IsaacWorld  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("manoeuvres")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/manoeuvres", time.strftime("%Y-%m-%d_%H-%M-%S"))
GROUND_USD = os.path.join(os.path.dirname(OUT), "proving_ground.usda")
SURFACES = os.path.join(HERE, "proving_ground.yaml")
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
SUBSTEPS, DT = 3, 1 / 60
FRAME = SUBSTEPS * DT
G = 9.81
EXTENT = (-100.0, 560.0, -40.0, 520.0)  # the ground's x0, x1, y0, y1
GRID = 10.0
LANES = {"dry": 15.0, "wet": 135.0, "mud": 255.0, "split": 375.0}
X_SPAWN, X_TRIGGER = -80.0, 100.0
MANOEUVRES = ("brake", "step", "brake_turn", "sine", "dlc", "circle", "sweep_lat", "sweep_long")
STEER_RATE, STEER_MAX = math.radians(5.0), math.radians(35.0)
BRAKE_RAMP, BRAKE_MAX = 1.0, 15.0
RADIUS = 40.0
X_CIRCLE = 160.0


def make_ground(path):
    """A flat mesh labelled Road (CARLA's Q-code) with a triangle-mesh collider."""
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    x0, x1, y0, y1 = EXTENT
    nx, ny = int((x1 - x0) / GRID), int((y1 - y0) / GRID)
    pts = [Gf.Vec3f(x0 + i * GRID, y0 + j * GRID, 0.0) for j in range(ny + 1)
           for i in range(nx + 1)]
    idx = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            idx += [a, a + 1, a + nx + 2, a + nx + 1]
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Ground")
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * (nx * ny)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.25, 0.25, 0.27)]))
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    prim.CreateAttribute("semantics:labels:wikidata_qcode",
                         Sdf.ValueTypeNames.TokenArray).Set(["Q34442"])
    stage.GetRootLayer().Save()
    log.info("proving ground %s: %d x %d m", path, x1 - x0, y1 - y0)


os.makedirs(OUT, exist_ok=True)
if not os.path.exists(GROUND_USD):
    make_ground(GROUND_USD)
usd = os.path.realpath(GROUND_USD)
world = IsaacWorld(ego_config=EGO_YAML, ego_domain=int(os.environ.get("STEP_DOMAIN", "77")),
                   surface_maps={usd: (SURFACES, (0.0, 0.0, 0.0))})
for s in world.ego_rig.sensors.values():
    s.enabled = False
RIG = json.loads(os.environ.get("RIG", "{}"))
ECU = json.loads(os.environ.get("ECU", "{}"))
TRUTH = os.environ.get("ESTIMATOR", "") == "truth"
if RIG or ECU:
    world.ego_rig = dataclasses.replace(
        world.ego_rig, vehicle=dataclasses.replace(world.ego_rig.vehicle, **RIG),
        ecu=dataclasses.replace(world.ego_rig.ecu, **ECU))
    log.info("overrides: vehicle %s, ecu %s", RIG, ECU)


class TruthEstimator:
    """PhysX's own speed, slips, sideslip and friction, as aib's powertrain and ESC read them
    before roadmap 055: a spike's check, never shipped."""

    def __init__(self, ego):
        self.ego = ego

    def reset(self):
        pass

    def step(self, sensors, dt, tire_angle, activity):
        from sim_vehicle_ecu.estimator import Estimate

        s, v = self.ego.state, self.ego.vehicle
        vx, vy = s.velocity[0], s.velocity[1]
        return Estimate(vx, v.wheel_states, math.atan2(vy, max(abs(vx), 0.1)),
                        sum(v.friction_values) / len(v.friction_values))


P = world.ego_rig.vehicle
L = P.wheel_base
world.preload(usd)
world.load(usd, ["/World/env/Ground"])
from aib.control import GEAR_DRIVE  # noqa: E402

surface = world.surface
handle = [100]


def spawn(y, abs_off=False):
    if abs_off:
        rig = world.ego_rig
        world.ego_rig = dataclasses.replace(rig, ecu=dataclasses.replace(rig.ecu,
                                                                         abs_enabled=False))
    handle[0] += 1
    world.spawn(handle[0], "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5),
                ((X_SPAWN, y, 0.3), (1.0, 0.0, 0.0, 0.0)))  # heading +x
    ego = world.egos[handle[0]]
    if abs_off:
        world.ego_rig = rig
    if TRUTH:
        ego.vehicle.ecu.estimator = TruthEstimator(ego)
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


def hold(cmd, v, target):
    cmd.acceleration, cmd.velocity = max(-3.0, min(3.0, 1.5 * (target - v))), target


def pursue(x, y, yaw, tx, ty):
    """Pure pursuit from the rear axle to the point (tx, ty): the bicycle tire angle."""
    a = math.atan2(ty - y, tx - x) - yaw
    ld = max(math.hypot(tx - x, ty - y), 1.0)
    return math.atan2(2 * L * math.sin(a), ld)


def dlc_offset(s):
    """ISO 3888-1's lateral path, approximately: 0 for 15 m, 3.5 m reached over 30 m, held
    25 m, back over 25 m, 0 after; `s` m from the trigger."""
    def blend(u):
        return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, u)))
    if s < 15:
        return 0.0
    if s < 45:
        return 3.5 * blend((s - 15) / 30)
    if s < 70:
        return 3.5
    return 3.5 * (1 - blend((s - 70) / 25))


SINE_F, SINE_DWELL = 0.7, 0.5
SINE_END = 1 / SINE_F + SINE_DWELL  # 1.93 s


def sine_angle(tm, amp):
    """FMVSS 126's sine with dwell, tire angle: up for half a cycle, down to -amp, held
    SINE_DWELL, back to 0 over the last quarter."""
    w = 2 * math.pi * SINE_F
    t_dwell = 0.75 / SINE_F
    if tm < t_dwell:
        return amp * math.sin(w * tm)
    if tm < t_dwell + SINE_DWELL:
        return -amp
    if tm < SINE_END:
        return amp * math.sin(w * (tm - SINE_DWELL))
    return 0.0


def speeds(pad, kind):
    mud = pad in ("mud", "split")
    return {"brake": 40, "step": 40, "brake_turn": 40, "sine": 50 if mud else 80,
            "dlc": 40 if mud else 80, "circle": 5 * 3.6, "sweep_lat": 10 * 3.6,
            "sweep_long": 30 if mud else 40}[kind] / 3.6


def tires(v, ws, cp):
    """Each wheel's tire: PhysX's force (along, across), slips, load, the surface layer's
    stiffness shares and share of the peak (054 T1)."""
    out = []
    for i, w in enumerate(v.wheel_states or ()):
        c = cp[i] if cp else None
        out.append({"fx": round(w.force[0], 1), "fy": round(w.force[1], 1),
                    "fz": round(c.load, 1) if c and c.on_ground else 0.0,
                    "kappa": round(w.slip, 4), "alpha": round(w.slip_angle, 4),
                    "friction": round(v.friction_values[i], 4),
                    "stiffness": [round(x, 3) for x in ws.stiffness[i]] if ws else None,
                    "g": round(ws.g[i], 3) if ws else None})
    return out


def row(t, phase, ego, cmd, yaw_u, sweep=False):
    s, v = ego.state, ego.vehicle
    vx, vy = s.velocity[0], s.velocity[1]
    ws = surface.wheels.get(v.path) if surface is not None else None
    dbg = v.ecu.debug
    est = dbg.get("estimate") or {}
    return {"t": round(t, 3), "phase": phase, "x": round(s.position[0], 3),
            "y": round(s.position[1], 3), "yaw": round(math.degrees(yaw_u), 3),
            "r": round(math.degrees(s.angular_velocity[2]), 3), "vx": round(vx, 3),
            "vy": round(vy, 3), "beta": round(math.degrees(math.atan2(vy, max(abs(vx), 0.5))), 3),
            "ax": round(s.acceleration[0], 3), "ay": round(s.acceleration[1], 3),
            "tire": round(math.degrees(s.tire_angle), 3),
            "cmd_a": round(cmd.acceleration, 3), "cmd_tire": round(math.degrees(cmd.tire_angle), 3),
            "abs": [round(a, 2) for a in dbg.get("abs", ())], "tc": dbg.get("tc"),
            "esc": dbg.get("esc"),
            "est": {"v": round(est["speed"], 3), "beta": round(math.degrees(est["sideslip"]), 3),
                    "mu": round(est["friction"], 3),
                    "slip": est["slip"] and [round(x, 3) for x in est["slip"]]} if est else None,
            "mu": [None if ws is None or m is None else round(m, 3) for m in ws.mu]
            if ws is not None else None,
            "slip": [round(w.slip, 3) for w in v.wheel_states] if v.wheel_states else None,
            "cls": [None if ws is None or g is None else g.cls for g in ws.surfaces]
            if ws is not None else None,
            "tires": tires(v, ws, v.contacts) if sweep else None}


def run(pad, kind):
    name = f"{pad}_{kind}"
    y0 = LANES[pad]
    if surface is not None:
        surface.fresh_soil()
    ego, cmd = spawn(y0, abs_off=kind == "sweep_long")
    v_go = speeds(pad, kind)
    amp = 6.5 * math.atan(L * 0.3 * G / (v_go * v_go))
    rows, t, t_go = [], 0.0, None
    yaw_prev = yaw_of(ego.state.quat)
    yaw_u = yaw_prev
    trigger = X_CIRCLE if kind == "circle" else X_TRIGGER
    while t < 200.0:
        s = ego.state
        v, x, y = s.velocity[0], s.position[0], s.position[1]
        yaw = yaw_of(s.quat)
        yaw_u += (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_prev = yaw
        if t_go is None and x >= trigger:
            t_go = t
        if t_go is None:
            phase = "runup"
            hold(cmd, v, v_go)
            cmd.tire_angle = pursue(x, y, yaw_u, x + max(6.0, v), y0)
        else:
            tm = t - t_go
            phase = kind
            done = False
            if kind == "brake":
                cmd.acceleration, cmd.velocity, cmd.tire_angle = -8.0, 0.0, 0.0
                done = abs(v) < 0.05
            elif kind == "step":
                hold(cmd, v, v_go)
                cmd.tire_angle = math.radians(5.0) * min(1.0, tm / 0.15)
                done = tm > 2.0
            elif kind == "brake_turn":
                cmd.tire_angle = math.radians(1.5)
                if tm < 1.0:
                    phase = "turn"
                    hold(cmd, v, v_go)
                else:
                    phase = "brake"
                    cmd.acceleration, cmd.velocity = -8.0, 0.0
                    done = abs(v) < 0.05
            elif kind == "sine":
                cmd.acceleration, cmd.velocity = 0.0, max(v, 0.0)
                cmd.tire_angle = sine_angle(tm, amp)
                done = tm > SINE_END + 2.0
            elif kind == "dlc":
                hold(cmd, v, v_go)
                look = max(6.0, 0.6 * v)
                s_ahead = x + look - trigger
                cmd.tire_angle = pursue(x, y, yaw_u, x + look, y0 + dlc_offset(s_ahead))
                done = x - trigger > 130.0
            elif kind == "sweep_lat":
                hold(cmd, v, v_go)
                cmd.tire_angle = min(STEER_MAX, STEER_RATE * tm)
                done = tm > STEER_MAX / STEER_RATE + 1.0
            elif kind == "sweep_long":
                cmd.acceleration = -BRAKE_MAX * min(1.0, tm / BRAKE_RAMP)
                cmd.velocity, cmd.tire_angle = 0.0, 0.0
                done = abs(v) < 1.0
            else:  # circle
                hold(cmd, v, 5.0 + 0.15 * tm)
                cx, cy = X_CIRCLE, y0 + RADIUS
                th = math.atan2(y - cy, x - cx) + max(6.0, 0.6 * v) / RADIUS
                cmd.tire_angle = pursue(x, y, yaw_u, cx + RADIUS * math.cos(th),
                                        cy + RADIUS * math.sin(th))
                err = math.hypot(x - cx, y - cy) - RADIUS
                done = abs(err) > 4.0 or v > 25.0 or tm > 150.0
            if done or abs(y - y0) > 100.0 or x > EXTENT[1] - 20:
                break
        world.step({}, SUBSTEPS, DT, False, sim_time=t)
        t += FRAME
        rows.append(row(t, phase, ego, cmd, yaw_u, kind.startswith("sweep")))
    world.despawn(handle[0])
    for _ in range(5):
        world.step({}, SUBSTEPS, DT, False)
    with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = summarise(pad, kind, [r for r in rows if r["phase"] != "runup"], v_go, amp, y0)
    print(name, json.dumps(out), flush=True)
    return out


def peak(rows, key):
    return max((r[key] for r in rows), key=abs) if rows else None


def summarise(pad, kind, m, v_go, amp, y0):
    out = {"case": f"{pad}_{kind}"}
    if not m:
        out["error"] = "never reached the trigger"
        return out
    go, last = m[0], m[-1]
    out.update(speed_kmh=round(3.6 * go["vx"], 1), ground=go["cls"], mu=go["mu"],
               duration_s=round(last["t"] - go["t"], 2),
               heading_change_deg=round(last["yaw"] - go["yaw"], 2),
               sideways_m=round(last["y"] - go["y"], 3), peak_yaw_rate=peak(m, "r"),
               peak_sideslip=peak(m, "beta"), peak_ay=peak(m, "ay"))
    braking = [r for r in m if r["cmd_a"] <= -7.9]
    if braking:
        out.update(distance_m=round(last["x"] - braking[0]["x"], 2),
                   mean_decel=round(-sum(r["ax"] for r in braking) / len(braking), 2),
                   abs_share=round(sum(1 for r in braking if r["abs"] and min(r["abs"]) < 0.99)
                                   / len(braking), 2))
    if kind == "step":
        st = [r for r in m if 1.0 <= r["t"] - go["t"] <= 2.0]
        r_ss = sum(r["r"] for r in st) / len(st)
        v = sum(r["vx"] for r in st) / len(st)
        r_kin = math.degrees(v * math.tan(math.radians(5.0)) / L)
        rise = next((r["t"] - go["t"] for r in m if abs(r["r"]) >= 0.9 * abs(r_ss)), None)
        out.update(yaw_rate_steady=round(r_ss, 2), yaw_rate_ratio=round(r_ss / r_kin, 3),
                   ay_steady=round(sum(r["ay"] for r in st) / len(st), 2),
                   sideslip_steady=round(sum(r["beta"] for r in st) / len(st), 2),
                   yaw_rate_rise_90_s=rise and round(rise, 2),
                   yaw_rate_overshoot=round(peak(m, "r") / r_ss, 3) if r_ss else None)
    if kind == "brake_turn":
        turn = [r for r in m if r["phase"] == "turn" and r["t"] - go["t"] >= 0.5]
        out["yaw_rate_before_brake"] = round(sum(r["r"] for r in turn) / max(1, len(turn)), 2)
    if kind == "sine":
        t0 = go["t"]
        at = lambda dt: min(m, key=lambda r: abs(r["t"] - t0 - dt))  # noqa: E731
        second = [r for r in m if 1 / (2 * SINE_F) <= r["t"] - t0 <= SINE_END + 1.0]
        r_peak = min((r["r"] for r in second), default=0.0)  # the lobe after the reversal
        yaw0 = math.radians(go["yaw"])
        r107 = at(1.07)
        lat = -(r107["x"] - go["x"]) * math.sin(yaw0) + (r107["y"] - go["y"]) * math.cos(yaw0)
        out.update(amplitude_deg=round(math.degrees(amp), 2), second_peak_yaw_rate=r_peak,
                   ratio_1_00=round(at(SINE_END + 1.0)["r"] / r_peak, 3) if r_peak else None,
                   ratio_1_75=round(at(SINE_END + 1.75)["r"] / r_peak, 3) if r_peak else None,
                   lateral_at_1_07_m=round(lat, 2))
    if kind == "dlc":
        x0 = go["x"]
        errs = [abs(r["y"] - (y0 + dlc_offset(r["x"] - x0))) for r in m]
        out.update(max_path_error_m=round(max(errs), 2), end_speed_kmh=round(3.6 * last["vx"], 1))
    if kind == "circle":
        cx, cy = X_CIRCLE, y0 + RADIUS
        pts = []
        for r in m:
            e = math.hypot(r["x"] - cx, r["y"] - cy) - RADIUS
            if abs(e) < 1.0 and r["vx"] > 4.0:
                pts.append((abs(r["ay"]), r["tire"] - math.degrees(math.atan(L / RADIUS)), e))
        lin = [(a, d) for a, d, _ in pts if 0.5 <= a <= 0.6 * max((p[0] for p in pts),
                                                                 default=0)]
        k = None
        if len(lin) > 10:
            ma = sum(a for a, _ in lin) / len(lin)
            md = sum(d for _, d in lin) / len(lin)
            num = sum((a - ma) * (d - md) for a, d in lin)
            den = sum((a - ma) ** 2 for a, _ in lin)
            k = num / den * G if den else None
        # the most lateral acceleration held for 2 s on the circle (within 1 m)
        best, win = 0.0, int(2.0 / FRAME)
        held = [abs(r["ay"]) if abs(math.hypot(r["x"] - cx, r["y"] - cy) - RADIUS) < 1.0
                else 0.0 for r in m]
        for i in range(len(held) - win):
            best = max(best, min(held[i:i + win]))
        out.update(understeer_deg_per_g=k and round(k, 2), ay_max_held=round(best, 2),
                   end_speed_kmh=round(3.6 * last["vx"], 1))
    if kind.startswith("sweep"):
        key, force = ("alpha", "fy") if kind == "sweep_lat" else ("kappa", "fx")
        for axle, wheels in (("front", (0, 1)), ("rear", (2, 3))):
            pts = [(abs(w[key]), abs(w[force]) / w["fz"], w["stiffness"]) for r in m
                   if r["tires"] for i in wheels for w in (r["tires"][i],) if w["fz"] > 500]
            if not pts:
                continue
            top, far = max(pts, key=lambda p: p[1]), max(pts, key=lambda p: p[0])
            slip = (lambda x: round(math.degrees(x), 2)) if kind == "sweep_lat" \
                else (lambda x: round(x, 3))
            out[axle] = {"peak_force_over_load": round(top[1], 3), "at_slip": slip(top[0]),
                         "largest_slip": slip(far[0]),
                         "force_over_load_there": round(far[1], 3),
                         "share_of_peak_there": round(far[1] / top[1], 3) if top[1] else None,
                         "stiffness": top[2]}
    return out


pads = os.environ.get("PADS", "dry,wet,mud,split").split(",")
kinds = os.environ.get("MANOEUVRES", ",".join(MANOEUVRES)).split(",")
result = {"out": OUT, "rig": RIG, "ecu": ECU, "estimator": "truth" if TRUTH else "ecu",
          "cases": []}
for pad in pads:
    for kind in kinds:
        if pad == "split" and kind != "brake":
            continue
        try:
            result["cases"].append(run(pad, kind))
        except Exception as e:  # noqa: BLE001
            logging.exception("case %s %s", pad, kind)
            result["cases"].append({"case": f"{pad}_{kind}", "error": repr(e)})
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(result, f, indent=1)
print("SPIKE-RESULT " + json.dumps(result), flush=True)
app.close()
