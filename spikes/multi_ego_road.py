"""One and two egos on a rough road with rocks, every sensor on (roadmap 066 R4/R5 acceptance).

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/multi_ego_road.py \\
        [<out dir>]
    # ROAD_TOWN=../swb_isaac_world/examples/road/town01_flat/surfaces.yaml: the same Town01
    # without the road detail (baseline)
    # ROAD_SPEED=11.1: m/s; ROAD_CASES=one,side,lane: the cases, in this order
    # ROAD_MESH=0: no render mesh; ROAD_ROCKS=0: no rock stream (what each costs)
    # ROAD_PROFILE=1: the frame's parts (Isaac's step, the surface layer's calls)
    # ROAD_PROFILE=2: and the road layer's calls under cProfile (OUT/road_contact.prof)
    # ROAD_KERNEL=numpy: the tire contact in numpy, not the compiled kernel (what it saves)
    # prints one SPIKE-RESULT json

The budget's own measure (066 "The budget"; 052 step 5's 29.2 ms for one ego): aib egos in one
world server on Town01, every rig sensor on, without Autoware. Each ego gets its own command
(the egos share one EgoRos, as in `multi_ego_soil.py`): a speed loop and a pure-pursuit driver
on its lane, set on its vehicle as the vehicle interface would. Default ground:
`swb_isaac_world/examples/road/town01/surfaces.yaml`, whose `road_detail.yaml` lays R3's rough gravel corridor with R4's
rocks over the straight at y -131 (x 95-345). The egos head -x from x 330 to x 110.

- one: one ego on the north lane (y -129.6);
- side: two side by side (y -129.6 and -132.6);
- lane: two in the north lane, 12 m apart (the second meets what the first kicked).

Per case: the world server's frame (3 substeps of 1/60 s, a render on the last) p50 / p95 /
max and frames over 50 ms; the road layer's share (its contact per substep and the rock and
mesh streams once a frame); the rocks: awake, prims, lost (fallen under the road). Once: the
rock pool's time at the road's load and at the second ego's spawn (`RockStream.reserve`).
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

from swb_isaac_world.surface_layer import register  # noqa: E402
from swb_isaac_world.world import IsaacWorld  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/multi_ego_road", time.strftime("%Y-%m-%d_%H-%M-%S"))
WORLD = os.path.join(REPO, "data/carla-autoware-bridge/Town01/isaac/world.yaml")
SURFACES = os.path.join(HERE, os.environ.get(
    "ROAD_TOWN", "../swb_isaac_world/examples/road/town01/surfaces.yaml"))
SPEED = float(os.environ.get("ROAD_SPEED", "11.1"))
CASES = [c for c in os.environ.get("ROAD_CASES", "one,side,lane").split(",") if c]
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
SUBSTEPS, DT = 3, 1 / 60
FRAME = SUBSTEPS * DT
NORTH, SOUTH = -129.6, -132.6
X0, GAP, END_X = 330.0, 12.0, 110.0
SKIP_X = 40.0  # m driven before frames count (speeding up)
LOOK = 10.0  # m: the pure-pursuit point ahead on the lane

os.makedirs(OUT, exist_ok=True)
with open(WORLD) as f:
    w = yaml.safe_load(f)
usd, ground = os.path.realpath(w["usd"]), list(w.get("ground_paths") or [])
assert register([WORLD])
if os.environ.get("ROAD_KERNEL"):  # the tire contact's kernel for every car
    from swb_isaac_world import road_contact as _rc
    _tc_init = _rc.TireContact.__init__
    _rc.TireContact.__init__ = lambda self, *a, kernel=None, **k: _tc_init(
        self, *a, kernel=os.environ["ROAD_KERNEL"], **k)
world = IsaacWorld(ego_config=EGO_YAML, ego_domain=int(os.environ.get("STEP_DOMAIN", "77")),
                   surface_maps={usd: (SURFACES, tuple(w.get("map_origin") or (0.0, 0.0, 0.0)))})
# issue 039: on a failed spawn, what the stage holds for the ego and the collision groups
_spawn0 = world._spawn


def _spawn_diag(*a, **k):
    try:
        return _spawn0(*a, **k)
    except Exception:
        from pxr import Usd, UsdPhysics
        st = world.stage
        print("DIAG039 spawn failed", flush=True)
        g = st.GetPrimAtPath("/World/vehicleCollisionGroups")
        for c in (g.GetChildren() if g else []):
            inc = UsdPhysics.CollisionGroup(c).GetCollidersCollectionAPI().GetIncludesRel()
            filt = UsdPhysics.CollisionGroup(c).GetFilteredGroupsRel().GetTargets()
            print("DIAG039 group", c.GetPath(), "includes", len(inc.GetTargets()) if inc else None,
                  "filtered", [str(t) for t in filt], flush=True)
        for prim in Usd.PrimRange(st.GetPrimAtPath(world._ego_path())):
            apis = [x for x in prim.GetAppliedSchemas() if "Physics" in x or "Vehicle" in x]
            if apis:
                print("DIAG039 prim", prim.GetPath(), prim.IsActive(), apis[:6], flush=True)
        raise


world._spawn = _spawn_diag
if os.environ.get("ROAD_STEP_FIRST") == "1":  # issue 039: PhysX parses the load before an ego
    _load0 = world.load

    def _load_step(*a, **k):
        out = _load0(*a, **k)
        world.world.step(render=False)
        print("ROAD stepped once after the load", flush=True)
        return out
    world.load = _load_step
world.preload(usd)
t_load = time.perf_counter()
world.load(usd, ground)
load_s = time.perf_counter() - t_load
from aib.control import GEAR_DRIVE  # noqa: E402

surface = world.surface
road = surface.road
rs = surface.road_rocks
print(f"ROAD {SURFACES}: detail {road is not None}, rocks {rs is not None}, load {load_s:.1f} s",
      flush=True)

if os.environ.get("ROAD_MESH") == "0" and getattr(surface, "road_meshes", None) is not None:
    surface.road_meshes.update = lambda *a, **k: None  # no render mesh: what sensors pay for it
    print("ROAD render mesh off", flush=True)
if os.environ.get("ROAD_ROCKS") == "0" and rs is not None:
    rs.update = lambda *a, **k: None  # no rock or collision-tile streaming
    print("ROAD rock stream off", flush=True)
# the road's share of a frame: its contact (per substep, RoadLayer.cost_ms) and the streams
stream_ms = [0.0]
for obj, name in ((rs, "update"), (getattr(surface, "road_meshes", None), "update")):
    if obj is None:
        continue
    inner = getattr(obj, name)

    def timed(*a, _inner=inner, **k):
        t = time.perf_counter()
        try:
            return _inner(*a, **k)
        finally:
            stream_ms[0] += (time.perf_counter() - t) * 1e3
    setattr(obj, name, timed)

# ROAD_PROFILE=1: each part of the world server's frame, ms summed per frame
PARTS = {}
PROF = None
if os.environ.get("ROAD_PROFILE") == "2" and road is not None:
    import cProfile
    PROF = cProfile.Profile()
    _bs = road.before_substep

    def _profiled(*a, **k):
        PROF.enable()
        try:
            return _bs(*a, **k)
        finally:
            PROF.disable()
    road.before_substep = _profiled
if os.environ.get("ROAD_PROFILE") in ("1", "2"):
    def _timed(obj, name, key):
        inner = getattr(obj, name)

        def call(*a, **k):
            t = time.perf_counter()
            try:
                return inner(*a, **k)
            finally:
                PARTS[key] = PARTS.get(key, 0.0) + (time.perf_counter() - t) * 1e3
        setattr(obj, name, call)
    _timed(world.world, "step", "isaac_step")  # PhysX, and the render on the last substep
    for _n in ("place", "update", "begin_step", "flush"):
        _timed(surface, _n, "surface_" + _n)
    if road is not None:
        _timed(road, "before_substep", "road_contact")

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
        egos.append((next_handle[0], ego, own_command(ego), y))
    for _ in range(20):  # settle on the hold brake
        world.step({}, SUBSTEPS, DT, False)
    return egos


def yaw_of(q):
    w_, qx, qy, qz = q
    return math.atan2(2 * (w_ * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def drive(egos, max_t=60.0):
    frames, road_ms, t = [], [], 0.0
    awake, prims, parts = [], [], []
    ends = {}
    wb = world.ego_rig.vehicle.wheel_base
    while t < max_t and any(e.state.position[0] > END_X for _, e, _, _ in egos):
        for h, e, c, lane in egos:
            x, y = e.state.position[0], e.state.position[1]
            v = e.state.velocity[0]
            c.acceleration, c.velocity = max(-3.0, min(3.0, 1.5 * (SPEED - v))), SPEED
            yaw = yaw_of(e.state.quat)
            a = math.atan2(lane - y, (x - LOOK) - x) - yaw
            a = math.atan2(math.sin(a), math.cos(a))
            c.tire_angle = max(-0.3, min(0.3, math.atan2(2 * wb * math.sin(a), LOOK)))
            c.gear, c.received = GEAR_DRIVE, True
            if e.state.position[2] < -2.0 or abs(y - lane) > 6.0:
                ends.setdefault(h, ("left", [round(float(p), 1) for p in e.state.position]))
        c0 = len(road.cost_ms) if road is not None else 0
        stream_ms[0] = 0.0
        PARTS.clear()
        t0 = time.perf_counter()
        world.step({}, SUBSTEPS, DT, False, sim_time=t)
        dt_ms = (time.perf_counter() - t0) * 1e3
        t += FRAME
        if min(e.state.position[0] for _, e, _, _ in egos) < X0 - SKIP_X:
            frames.append(dt_ms)
            road_ms.append(stream_ms[0] + (sum(road.cost_ms[c0:]) if road is not None else 0.0))
            parts.append(dict(PARTS, frame=dt_ms))
            if rs is not None:
                awake.append(rs.awake())
                prims.append(sum(p is not None for p in rs.paths))
    for h, e, _, _ in egos:
        ends.setdefault(h, ("end", [round(float(p), 1) for p in e.state.position]))
    drive.parts = parts
    return frames, road_ms, awake, prims, ends, t


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 2) if len(xs) else None


def lost_rocks():
    if rs is None:
        return None
    n = 0
    for i, p in enumerate(rs.paths):
        if p is not None and rs.dyn[i]:
            rs.read(i)
            if road.rocks.pos[i][2] < -0.5:
                n += 1
    return n


STARTS = {"one": [(X0, NORTH)], "side": [(X0, NORTH), (X0, SOUTH)],
          "lane": [(X0, NORTH), (X0 + GAP, NORTH)]}
res = {"surfaces": SURFACES, "detail": road is not None, "rocks": rs is not None,
       "speed": SPEED, "load_s": round(load_s, 1), "cases": {}}
for name in CASES:
    egos = spawn(STARTS[name])
    frames, road_ms, awake, prims, ends, t = drive(egos)
    np.save(os.path.join(OUT, f"frame_ms_{name}.npy"), np.array(frames))
    out = {"egos": len(egos), "frames": len(frames), "driven_s": round(t, 1),
           "frame_ms": {"p50": pct(frames, 50), "p95": pct(frames, 95),
                        "max": round(max(frames), 1) if frames else None,
                        "over_50": int(sum(f > 50 for f in frames))},
           "road_ms": {"p50": pct(road_ms, 50), "p95": pct(road_ms, 95)},
           "ends": {str(h): v for h, v in ends.items()}}
    if drive.parts and len(drive.parts[0]) > 1:
        keys = sorted({k for p in drive.parts for k in p})
        out["parts_p50_ms"] = {k: pct([p.get(k, 0.0) for p in drive.parts], 50) for k in keys}
    if rs is not None:
        out.update({"awake": {"p50": int(np.median(awake)), "max": int(max(awake))},
                    "rock_prims_p50": int(np.median(prims)), "lost": lost_rocks(),
                    "pool": {"vehicles": rs.reserved_for, "prims": rs.prewarmed,
                             "ms": [round(m) for m in rs.prewarm_ms]},
                    "prims_made_live": rs.made, "prims_reused": rs.reused})
    for h, _, _, _ in egos:
        world.despawn(h)
    for _ in range(10):
        world.step({}, SUBSTEPS, DT, False)
    res["cases"][name] = out
    print(name, json.dumps(out), flush=True)

if PROF is not None:
    import pstats
    PROF.dump_stats(os.path.join(OUT, "road_contact.prof"))
    pstats.Stats(PROF).sort_stats("tottime").print_stats(30)
print("SPIKE-RESULT " + json.dumps(res), flush=True)
with open(os.path.join(OUT, "result.json"), "w") as f:
    json.dump(res, f, indent=1)
app.close()
