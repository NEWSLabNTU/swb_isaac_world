"""Roadmap 066 R4: the world server's road with rocks, as `world.py` runs it (`road_layer`
RoadLayer, MeshStream, `rock_layer` RockStream), under aib's Model 3 over 2 km of R3's class C
corridor at a held speed, Isaac at 60 Hz.

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/rock_world.py \\
        [<out dir>]
    # prints one SPIKE-RESULT json

The corridor's rocks are `rocks.Zones` with the carriageway's cover raised to ROCK_CARRIAGEWAY
(default 0.01; the forest road's own 0.002 puts almost none in the car's path), so the tires
meet rocks along the whole road. The skid plate is aib's, at the spec's ground clearance
(Tesla's 138 mm). ROCKS=0 runs
the same road with no rocks: the frame time to compare.

ROCK_RECORD=1 films each drive with the world server's recorder (`recorder.py`: dash, wheel,
chase and course cameras, under the sun and sky world.py lights, the ego in its catalogue
body from SWB_VEHICLE_USD) to <out dir>/record/; scripts/record_video.py makes the video.
Rendering costs frame time: not for the budget.

ROCK_COURSE=tour drives a road of six 100 m sections from the car's start instead, one condition
each, blended over TOUR_TAPER m (`Corridor.taper`): smooth graded gravel (class B), rough and
rutted (D, 4 cm ruts), potholes (15 per 100 m), washboard (3 cm, 0.5 m: the wave 50 km/h
traffic makes), water bars (four, 25 m apart) and a rocky stretch (class E, 3 % rock cover on
the carriageway); ROCK_LENGTH 600 covers it. With ROCK_RECORD the sun stands 20 deg high, so
relief casts shadows, and the record has sections.json (x range and label per section) for
record_video.py's caption.

Per speed (ROCK_SPEEDS, km/h): frame wall time, the parts of the slowest frames, awake rocks,
rock prims, rocks under the tires, the fastest rock, rocks lost, where the car left the road or
stuck.
"""
import json
import math
import os
import sys
import time

os.environ.pop("DISPLAY", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]
                     + os.environ.get("ISAAC_KIT_ARGS", "").split()})
import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402

from aib.control import GEAR_DRIVE, Command, State  # noqa: E402
from aib.frames import qrot  # noqa: E402
from aib.isaac.vehicle import Vehicle, ensure_vehicle_scene  # noqa: E402
from aib.rig import EgoConfig  # noqa: E402
from swb_isaac_world import road_detail as rd  # noqa: E402
from swb_isaac_world import road_layer  # noqa: E402
from swb_isaac_world import rock_layer  # noqa: E402
from swb_isaac_world import rocks as rk  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "play_log/rock_world", time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(OUT, exist_ok=True)
SPEEDS = [float(v) for v in os.environ.get("ROCK_SPEEDS", "40,60,80").split(",")]
ROCKS_ON = os.environ.get("ROCKS", "1") == "1"
CARRIAGEWAY = float(os.environ.get("ROCK_CARRIAGEWAY", "0.01"))
NO_COVER = os.environ.get("ROCK_COVER") == "0"  # the tiles and the stream, no rocks
if "ROCK_COLL_STEP" in os.environ:  # the collision tiles' spacing, in 5 cm samples
    rock_layer.COLLISION_STEP = int(os.environ["ROCK_COLL_STEP"])
ALL_STATIC = os.environ.get("ROCK_ALL_STATIC") == "1"  # no rock bodies: the prims' cost
if os.environ.get("ROCK_WAKE") == "0":  # no rock wakes, every loose one a static body
    rock_layer.ACTIVE_MIN, rock_layer.ACTIVE_S, rock_layer.ACTIVE_BEHIND = -99.0, 0.0, -99.0
if "ROCK_CCD" in os.environ:
    rock_layer.SPECULATIVE_CCD = os.environ["ROCK_CCD"] == "1"
if "ROCK_CONTACT_OFFSET" in os.environ:
    rock_layer.CONTACT_OFFSET = float(os.environ["ROCK_CONTACT_OFFSET"])
if "ROCK_SOLVER" in os.environ:
    rock_layer.SOLVER_POSITION_ITERATIONS = int(os.environ["ROCK_SOLVER"])
CONTACTS_ON = os.environ.get("ROCK_CONTACTS", "1") == "1"  # the ego's contact report
if "ROCK_COLL_REACH" in os.environ:
    rock_layer.COLLISION_REACH = float(os.environ["ROCK_COLL_REACH"])
LENGTH = float(os.environ.get("ROCK_LENGTH", "2000"))
RECORD = os.environ.get("ROCK_RECORD") == "1"
TOUR = os.environ.get("ROCK_COURSE") == "tour"
TOUR_TAPER = 5.0  # m over which one section blends into the next
NO_MESHES = os.environ.get("ROCK_MESHES") == "0"  # no render mesh: the bisection of issue 032
X_START = 20.0
FRAME, SUB = 0.05, 3
DT = FRAME / SUB
W_LAT, Z_LAT = 0.8, 0.9

world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
x0, x1 = X_START - 100.0, X_START + LENGTH + 200.0
base = UsdGeom.Mesh.Define(stage, "/World/env/Base/mesh")
base.CreatePointsAttr([Gf.Vec3f(x0, -20, 0), Gf.Vec3f(x1, -20, 0), Gf.Vec3f(x1, 20, 0),
                       Gf.Vec3f(x0, 20, 0)])
base.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
base.CreateFaceVertexCountsAttr([4])
UsdPhysics.CollisionAPI.Apply(base.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(base.GetPrim()).CreateApproximationAttr("none")
if not any(pr.IsA(UsdPhysics.Scene) for pr in stage.Traverse()):
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
ensure_vehicle_scene(stage)

corridor = rd.Corridor(
    np.array([[0.0, 0.0], [x1, 0.0]]), seed=66, roughness=rd.Roughness(cls="C"),
    ruts=rd.Ruts(offset=0.83), potholes=rd.Potholes(),
    washboard=(rd.Washboard(X_START + 600.0, X_START + 700.0, pitch=0.5),),
    water_bars=rd.WaterBars(at=(X_START + 1500.0, X_START + 1550.0, X_START + 1600.0)),
    rocks=(rk.Zones(carriageway=0.0, shoulder=0.0, ditch=0.0) if NO_COVER
           else rk.Zones(carriageway=CARRIAGEWAY)) if ROCKS_ON else None)
corridors = [corridor]
SECTIONS = []  # (x0, x1, label): the tour's
if TOUR:
    T, L = TOUR_TAPER, 100.0
    h = T / 2
    tour = [  # label, Corridor's keywords; u runs from the section's start - T/2
        ("1 smooth graded gravel - ISO 8608 class B", dict(roughness=rd.Roughness(cls="B"))),
        ("2 rough and rutted - class D, ruts 4 cm", dict(
            roughness=rd.Roughness(cls="D"),
            ruts=rd.Ruts(depth=0.04, offset=0.83, windrow=0.03))),
        ("3 potholes - 15 per 100 m, 3-8 cm deep", dict(
            roughness=rd.Roughness(cls="C"),
            potholes=rd.Potholes(per_100m=15.0, depth=(0.03, 0.08), diameter=(0.4, 1.0)))),
        ("4 washboard - 3 cm, 0.5 m wave", dict(
            roughness=rd.Roughness(cls="C"),
            washboard=(rd.Washboard(h, L + h, height=0.03, pitch=0.5),))),
        ("5 water bars - 6 cm, 35 deg, every 25 m", dict(
            roughness=rd.Roughness(cls="C"),
            water_bars=rd.WaterBars(at=(h + 12.5, h + 37.5, h + 62.5, h + 87.5)))),
        ("6 rocky - class E, 3 % rock cover", dict(
            roughness=rd.Roughness(cls="E"),
            rocks=rk.Zones(carriageway=0.03, shoulder=0.05, ditch=0.08) if ROCKS_ON else None)),
    ]
    corridors = []
    for k, (label, kw) in enumerate(tour):
        a, b = X_START + k * L, X_START + (k + 1) * L
        lo = x0 if k == 0 else a - h  # the first from the world's start, the last to its end
        hi = x1 if k == len(tour) - 1 else b + h
        corridors.append(rd.Corridor(np.array([[lo, 0.0], [hi, 0.0]]), seed=660 + k,
                                     taper=T, **kw))
        SECTIONS.append((a, b, label))

sim = omni.physx.get_physx_simulation_interface()
STAGE_ID = omni.usd.get_context().get_stage_id()


def apply_force(path, force, position):  # world.py's _apply_force
    sim.apply_force_at_pos(STAGE_ID, PhysicsSchemaTools.sdfPathToInt(path), force, position)


layer = road_layer.RoadLayer(corridors, apply_force=apply_force)
meshes = road_layer.MeshStream(stage, layer)

cfg = EgoConfig.load(os.environ.get("AIB_EGO") or os.path.join(
    HERE, "../deps/autoware_isaac_bridge/src/aib/config/ego.yaml"))
rig = cfg.load_rig()
vp = road_layer.vehicle_params(rig.vehicle)
veh = Vehicle(stage, "/World/ego", vp, rig.ecu, rig.box_center, rig.box_dims,
              ((X_START, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0)),
              ground_clearance=rig.ground_clearance)
CLEARANCE = rig.ground_clearance
assert veh.plate_path is not None, "the rig's spec gives no ground clearance"
PhysxSchema.PhysxVehicleAPI(stage.GetPrimAtPath("/World/ego")).GetSuspensionLineQueryTypeAttr() \
    .Set(PhysxSchema.Tokens.raycast)
plate = UsdGeom.Cube(stage.GetPrimAtPath(veh.plate_path))
if RECORD:  # world.py's light and the ego's catalogue body, for the cameras
    from pxr import UsdLux

    from swb_isaac_world import vehicle_assets
    from swb_isaac_world.catalogue import ego_model

    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(1.0)
    if TOUR:  # a low sun from behind the car's right: relief casts shadows ahead of it
        el, az = math.radians(20.0), math.radians(-135.0)  # the sun's elevation, azimuth
        to_sun = Gf.Vec3d(math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
        q = Gf.Rotation(Gf.Vec3d(0, 0, 1), to_sun).GetQuat()  # a light shines down its -Z
        UsdGeom.Xformable(sun.GetPrim()).AddOrientOp().Set(Gf.Quatf(q))
    else:
        UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(-50.0, 20.0, 30.0))  # world.py SUN_ROTATE
    UsdLux.DomeLight.Define(stage, "/World/sky").CreateIntensityAttr(600.0)
    from pxr import Sdf, UsdShade

    def look(path, colour, rough):  # a plain surface for the cameras, bound to a subtree
        mat = UsdShade.Material.Define(stage, f"/World/Looks/{path.strip('/').replace('/', '_')}")
        sh = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        prim = stage.DefinePrim(path) if not stage.GetPrimAtPath(path) else stage.GetPrimAtPath(path)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    # grass beyond the road: a picture-only plane a little under the base, which is hidden
    # (the base and the road's render mesh coincide on the flat and would fight)
    UsdGeom.Imageable(base.GetPrim()).MakeInvisible()
    grass = UsdGeom.Mesh.Define(stage, "/World/env/grass")
    grass.CreatePointsAttr([Gf.Vec3f(x0, -400, -0.04), Gf.Vec3f(x1, -400, -0.04),
                            Gf.Vec3f(x1, 400, -0.04), Gf.Vec3f(x0, 400, -0.04)])
    grass.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    grass.CreateFaceVertexCountsAttr([4])
    look("/World/env/grass", (0.20, 0.24, 0.13), 0.95)
    look(road_layer.MeshStream.ROOT, (0.42, 0.36, 0.28), 0.9)  # the gravel road
    look(rock_layer.ROOT, (0.30, 0.29, 0.28), 0.8)  # rocks
    asset = vehicle_assets.VehicleAssets().get(ego_model(rig))
    if asset is not None:
        vehicle_assets.add_ego_body(stage, veh.path, asset,
                                    collider=stage.GetPrimAtPath(f"{veh.path}/body"))
if CONTACTS_ON:  # as world.py: the ego reports every contact
    PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath("/World/ego")) \
        .CreateThresholdAttr(0.0)

physx = omni.physx.get_physx_interface()
LATE = os.environ.get("ROCK_LATE") == "1"  # as the world server: the stream after play


def stream():
    layer.rock_stream = rock_layer.RockStream(stage, layer, physx,
                                              ground=["/World/env/Base/mesh"])
    if ALL_STATIC:
        layer.rock_stream._in_band = lambda p: False  # every rock in its tile's collider


if os.environ.get("ROCK_TRACE_USD") == "1":  # each collider and group edit, in the log's order
    _mesh0, _group0 = rock_layer.RockStream._mesh, rock_layer.RockStream._group

    def _mesh_t(self, path, *a):
        carb.log_warn(f"USD mesh {path}")
        return _mesh0(self, path, *a)

    def _group_t(self, group, path, on):
        carb.log_warn(f"USD group {group.rsplit('/', 1)[-1]} {path} {on}")
        return _group0(self, group, path, on)

    rock_layer.RockStream._mesh, rock_layer.RockStream._group = _mesh_t, _group_t
    _sleep0, _prim0, _rel0 = (rock_layer.RockStream._sleep, rock_layer.RockStream._rock_prim,
                              rock_layer.RockStream._release)

    def _sleep_t(self, i, on):
        carb.log_warn(f"USD sleep {self.paths[i]} {on}")
        return _sleep0(self, i, on)

    def _prim_t(self, i):
        r = _prim0(self, i)
        carb.log_warn(f"USD prim {self.paths[i]}")
        return r

    def _rel_t(self, i):
        carb.log_warn(f"USD release {self.paths[i]}")
        return _rel0(self, i)

    rock_layer.RockStream._sleep = _sleep_t
    rock_layer.RockStream._rock_prim = _prim_t
    rock_layer.RockStream._release = _rel_t
    _step0 = world.step

    def _step_t(*a, **k):
        carb.log_warn("STEP")
        return _step0(*a, **k)

    world.step = _step_t
if ROCKS_ON and not LATE:
    stream()
world.reset()
world.play()
for pr in stage.Traverse():
    if pr.IsA(UsdPhysics.Scene):
        print("SCENE", pr.GetPath(), {a.GetName(): a.Get() for a in pr.GetAttributes()
                                      if a.HasAuthoredValue() or "gpu" in a.GetName().lower()
                                      or "broadphase" in a.GetName().lower()
                                      or "solver" in a.GetName().lower()}, flush=True)
if ROCKS_ON and LATE:
    stream()
state = State()
cmd = Command(received=True, gear=GEAR_DRIVE)
MU = vp.tire_friction


PHYS_MS = []  # world.step alone, per substep


def substep(render=False):
    """world.py's substep for one ego: the road's contact (and its share of the friction),
    the ego, PhysX."""
    if veh.last is not None:
        layer.before_substep([veh], DT)
        for i in range(4):
            veh.set_wheel_friction(i, MU * layer.scale(veh.path, i))
    veh.apply(cmd, DT, state)
    t0 = time.perf_counter()
    world.step(render=render)
    PHYS_MS.append((time.perf_counter() - t0) * 1e3)
    veh.sample(physx, DT, state)


def pose():
    p, q = state.position, state.quat
    f = qrot(q, (1.0, 0.0, 0.0))
    return [(p[0], p[1], math.atan2(f[1], f[0]), state.velocity[0], vp.wheel_tread / 2,
             vp.wheel_width / 2, CLEARANCE)]


def frame_streams():
    xs = [(state.position[0], state.position[1], state.position[2])]
    t0 = time.perf_counter()
    if not NO_MESHES:
        meshes.update(xs)
    t1 = time.perf_counter()
    if layer.rock_stream is not None:
        layer.rock_stream.update(pose())
    return (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3


TRACE = []  # per substep: x, y, yaw, vx, z, steer, the tires' loads (4)
CONTACTS = []  # the car's body's contacts: x, the other actor, our collider, impulse N s


def read_contacts(kmh):
    headers, data = sim.get_contact_report()
    for h in headers:
        a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
        a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        c0 = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
        c1 = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
        ours, other = (c0, a1) if a0.startswith("/World/ego") else (c1, a0)
        imp = max((math.sqrt(sum(float(c) ** 2 for c in data[k].impulse))
                   for k in range(h.contact_data_offset,
                                  h.contact_data_offset + h.num_contact_data)), default=0.0)
        if imp > 50.0:
            CONTACTS.append((round(state.position[0], 2), other, ours, round(imp, 1)))


def rest_gaps(rs):
    """Loose rocks with prims: their lowest point over the true surface (base + detail) under
    their centre, mm (p50, p95, n). Resting on the collision tiles: about 0; on the base above
    a rut or a crowned road's side: the detail's depth there."""
    g = []
    for i, p in enumerate(rs.paths):
        if p is None or not rs.dyn[i]:
            continue
        rs.read(i)
        x, y = layer.rocks.pos[i, 0], layer.rocks.pos[i, 1]
        h = float(layer.detail.height(np.array([x]), np.array([y]))[0])
        g.append((layer.rocks.bottom(i) - h) * 1e3)
    if not g:
        return None
    return [round(float(np.median(g)), 1), round(float(np.percentile(g, 95)), 1), len(g)]


PLATE_SET = []
DIAG = os.environ.get("ROCK_DIAG") == "1"


def diag(o):
    """A tire load over 30 kN: the rocks under that tire, as the contact saw them."""
    car = layer.cars[veh.path]
    k = int(np.argmax(o["load"]))
    rs, st = layer.rocks, layer.rock_stream
    xy = getattr(car, "last_xy", None)
    wx, wy = (xy[k] if xy is not None else (state.position[0], state.position[1]))
    near = rs.near(np.array([wx]), np.array([wy]), 0.6)
    road = float(layer.detail.height(np.array([wx]), np.array([wy]))[0])
    print(f"DIAG x={state.position[0]:.2f} tire {k} load {o['load'][k]:.0f} at "
          f"({wx:.2f}, {wy:.2f}) road {road:+.3f}", flush=True)
    for i in near:
        r = rs.rocks[i]
        print(f"DIAG   rock {i} {r.mass:.1f} kg r {r.radius:.2f} pos "
              f"{np.round(rs.pos[i], 3).tolist()} bottom {rs.bottom(i) - road:+.3f} "
              f"top {rs.pos[i][2] + (r.verts @ rk._rot(rs.quat[i]).T)[:, 2].max() - road:+.3f} "
              f"seen {bool(rs.seen[i])} awake {bool(st.active.get(i))} v {np.round(st.vel.get(i, np.zeros(3)), 2).tolist()}",
              flush=True)


def set_plate():
    """Where the body rests: aib puts the plate's bottom at the clearance over base_link, so
    the plate stands that much off Tesla's figure, which is over the road at rest."""
    z0 = float(state.position[2]) - float(layer.detail.height(
        np.array([state.position[0]]), np.array([state.position[1]]))[0])
    PLATE_SET.append(z0)
    print(f"PLATE origin at rest {z0 * 1e3:+.1f} mm from the road; plate bottom "
          f"{(CLEARANCE + z0) * 1e3:.0f} mm over it", flush=True)


def run(kmh):
    v = kmh / 3.6
    veh.teleport(((X_START - 60.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0)), physx)
    layer.reset(veh.path)
    for _ in range(20 * SUB):
        cmd.velocity, cmd.acceleration, cmd.tire_angle = 0.0, 0.0, 0.0
        layer.detail.prefetch(layer.detail.tiles_near([state.position[0]], [0.0], 48.0))
        frame_streams()
        substep()
    if not PLATE_SET:
        set_plate()
    rec = None
    if RECORD:
        from types import SimpleNamespace

        from swb_isaac_world import recorder
        from swb_isaac_world.recorder import Recorder

        if "ROCK_RECORD_SIZE" in os.environ:  # smaller pictures where the GPU is short
            recorder.SIZE = tuple(int(v) for v in os.environ["ROCK_RECORD_SIZE"].split("x"))

        rec = Recorder(os.path.join(OUT, "record"), stage,
                       SimpleNamespace(path=veh.path, state=state, vehicle=veh), rig,
                       f"{'tour' if TOUR else 'rocks'}-{kmh:g}kmh")
        rec.block = True  # every frame, at the cost of pace (issue 036)
        if SECTIONS:
            with open(os.path.join(rec.dir, "sections.json"), "w") as f:
                json.dump([{"x0": a, "x1": b, "label": lab} for a, b, lab in SECTIONS], f)
        # until every camera delivers, by the clock: the first renders compile the looks'
        # shaders, which take minutes on a loaded host whatever the frame count
        t_w, n_w = time.perf_counter(), 0
        while time.perf_counter() - t_w < 600.0:
            substep(render=True)
            n_w += 1
            if all(rec._ready(ann.get_data()) for _, _, ann in rec.views):
                break
        print(f"RECORD warm {n_w} frames {time.perf_counter() - t_w:.0f} s ready "
              f"{all(rec._ready(ann.get_data()) for _, _, ann in rec.views)}", flush=True)
    t_sim = 0.0
    frames, parts, awake = [], [], []
    left = stuck = None
    slow, vmax = 0.0, 0.0
    n_cost = len(layer.cost_ms)
    while state.position[0] < X_START + LENGTH:
        t0 = time.perf_counter()
        x = state.position[0]
        if state.position[2] < -2.0 or abs(state.position[1]) > 6.0:
            left = [round(float(c), 2) for c in state.position]
            break
        slow = slow + FRAME if abs(state.velocity[0]) < 0.3 else 0.0
        if slow > 5.0:
            stuck = [round(float(c), 2) for c in state.position]
            break
        mesh_ms, rock_ms = frame_streams()
        cmd.velocity = v
        cmd.acceleration = max(-3.0, min(3.0, 1.5 * (v - state.velocity[0])))
        yaw = pose()[0][2]
        vs = max(v, 1.0)
        cmd.tire_angle = max(-0.05, min(0.05, -vp.wheel_base * (
            W_LAT ** 2 * state.position[1] / vs ** 2 + 2 * Z_LAT * W_LAT * yaw / vs)))
        c0 = len(layer.cost_ms)
        p0 = len(PHYS_MS)
        t1 = time.perf_counter()
        if rec is not None:
            rec.place_course(FRAME)
        for k in range(SUB):
            substep(render=rec is not None and k == SUB - 1)
            o = layer.cars[veh.path].last_out if veh.path in layer.cars else None
            TRACE.append([state.position[0], state.position[1], pose()[0][2],
                          state.velocity[0], state.position[2], veh.tire_angle,
                          *(o["load"] if o is not None else [0.0] * 4)])
            if CONTACTS_ON:
                read_contacts(kmh)
            if DIAG and o is not None and max(o["load"]) > 30e3:
                diag(o)
        steps_ms = (time.perf_counter() - t1) * 1e3
        t_sim += FRAME
        if rec is not None:
            rec.capture(t_sim, None, None)
        if x > X_START + 40.0:
            frames.append((time.perf_counter() - t0) * 1e3)
            parts.append([mesh_ms, rock_ms, steps_ms, sum(layer.cost_ms[c0:]),
                          sum(PHYS_MS[p0:])])
            rs = layer.rock_stream
            if rs is not None:
                awake.append(rs.awake())
                vmax = max([vmax] + [float(np.linalg.norm(u)) for u in rs.vel.values()])
        if len(frames) % 400 == 1:
            print(f"  {kmh:g} km/h at {x:.0f} m: {awake[-1] if awake else 0} awake", flush=True)
    if rec is not None:
        rec.close()
        print(f"RECORDED {rec.dir} frames {rec.frame} dropped {rec.dropped}", flush=True)
    np.save(os.path.join(OUT, f"trace_{kmh:g}.npy"), np.array(TRACE[-30000:]))
    TRACE.clear()
    with open(os.path.join(OUT, f"contacts_{kmh:g}.json"), "w") as fh:
        json.dump(CONTACTS[-20000:], fh)
    CONTACTS.clear()
    f = np.array(frames) if frames else np.zeros(1)
    pa = np.array(parts) if parts else np.zeros((1, 5))
    np.save(os.path.join(OUT, f"frame_ms_{kmh:g}.npy"), f)
    np.save(os.path.join(OUT, f"parts_{kmh:g}.npy"), pa)
    slowest = f >= np.percentile(f, 95)
    out = {"kmh": kmh, "frame_ms": {"p50": round(float(np.median(f)), 2),
                                    "p95": round(float(np.percentile(f, 95)), 2),
                                    "max": round(float(f.max()), 1),
                                    "over_50": int((f > 50).sum()), "n": len(frames)},
           "parts_p50_ms": dict(zip(("meshes", "rocks", "steps", "contact", "physx"),
                                    np.round(np.median(pa, axis=0), 2).tolist())),
           "parts_slowest5_ms": dict(zip(("meshes", "rocks", "steps", "contact", "physx"),
                                         np.round(pa[slowest].mean(axis=0), 1).tolist())),
           "contact_ms_p50": round(float(np.median(layer.cost_ms[n_cost:] or [0])), 3),
           "left_road_at": left, "stuck_at": stuck}
    rs = layer.rock_stream
    if rs is not None:
        lost = 0
        for i, a in rs.active.items():
            if a:
                rs.read(i)
                p = layer.rocks.pos[i]
                if p[2] < -0.5:
                    lost += 1
        out.update({"awake": {"p50": int(np.median(awake)) if awake else 0,
                              "max": int(max(awake)) if awake else 0},
                    "rocks_known": len(rs.paths), "rocks_loose": int(sum(rs.dyn)),
                    "rock_prims": sum(p is not None for p in rs.paths)
                    + sum(1 for p in rs.win.values() if p),
                    "rock_v_max": round(vmax, 2), "lost": lost,
                    "prims_made": rs.made, "prims_reused": rs.reused,
                    "prims_prewarmed": rs.prewarmed,
                    "rest_gap_mm": rest_gaps(rs)})
    return out


res = {"rocks": ROCKS_ON, "carriageway_k": CARRIAGEWAY, "no_cover": NO_COVER,
       "coll_step": rock_layer.COLLISION_STEP, "coll_reach": rock_layer.COLLISION_REACH,
       "runs": {}}
PROFILE = os.environ.get("ROCK_PROFILE") == "1"  # cProfile each speed's drive
for kmh in SPEEDS:
    if PROFILE:
        import cProfile
        import io
        import pstats

        prof = cProfile.Profile()
        prof.enable()
    r = run(kmh)
    if PROFILE:
        prof.disable()
        prof.dump_stats(os.path.join(OUT, f"profile_{kmh:g}.pstats"))
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(30)
        print(buf.getvalue(), flush=True)
    res["runs"][f"{kmh:g}"] = r
    print(f"{kmh:g} km/h", json.dumps(r), flush=True)
if os.environ.get("ROCK_GROUPS") == "1":  # every collider in more than one collision group
    gs = [(str(pr.GetPath()), UsdPhysics.CollisionGroup(pr).GetCollidersCollectionAPI()
           .ComputeMembershipQuery()) for pr in stage.Traverse()
          if pr.IsA(UsdPhysics.CollisionGroup)]
    n_col = 0
    for pr in stage.Traverse():
        if not pr.HasAPI(UsdPhysics.CollisionAPI):
            continue
        n_col += 1
        mine = [g for g, q in gs if q.IsPathIncluded(pr.GetPath())]
        if len(mine) != 1:
            print("GROUPS", pr.GetPath(), mine, flush=True)
    print("GROUPS colliders", n_col, "groups", [g for g, _ in gs], flush=True)
res["origin_at_rest_mm"] = round(PLATE_SET[0] * 1e3, 1) if PLATE_SET else None
print("SPIKE-RESULT " + json.dumps(res))
layer.close()
app.close()
