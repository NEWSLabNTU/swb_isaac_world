"""What makes omni.physx log "Collisions are supported currently only in one collision group"
for colliders streamed into a group after play (roadmap 066 R4, issue 032)?

Phases, each followed by 30 steps and a marker line: colliders made one a step and added to
the group by path; one removed (prim and target); one more added; the group's targets set
whole. Count the warnings between markers in the log. The cubes' group filters the ground's,
so a cube in its group falls through the ground: the last line says which ones did.

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/group_late.py
"""
import json
import os

os.environ.pop("DISPLAY", None)
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
if not any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
G = "/World/groups"
ground = UsdGeom.Mesh.Define(stage, "/World/ground")
ground.CreatePointsAttr([Gf.Vec3f(-50, -10, 0), Gf.Vec3f(50, -10, 0), Gf.Vec3f(50, 10, 0),
                         Gf.Vec3f(-50, 10, 0)])
ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
ground.CreateFaceVertexCountsAttr([4])
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr("none")
base_g = UsdPhysics.CollisionGroup.Define(stage, f"{G}/base")
base_g.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([ground.GetPath()])
if os.environ.get("GROUP_AIB", "1") == "1":  # aib's: chassis and the wheels' ground query
    chassis = UsdPhysics.CollisionGroup.Define(stage, f"{G}/chassis")
    query = UsdPhysics.CollisionGroup.Define(stage, f"{G}/groundQuery")
    chassis.CreateFilteredGroupsRel().AddTarget(query.GetPath())
    query.CreateFilteredGroupsRel().AddTarget(chassis.GetPath())
    box = UsdGeom.Cube.Define(stage, "/World/car/box")
    box.AddTranslateOp().Set(Gf.Vec3d(0.0, 5.0, 0.5))
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    chassis.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(box.GetPath())
rock_g = UsdPhysics.CollisionGroup.Define(stage, f"{G}/rocks")
rock_g.CreateFilteredGroupsRel().AddTarget(base_g.GetPath())
if os.environ.get("GROUP_AIB", "1") == "1":
    rock_g.GetFilteredGroupsRel().AddTarget(f"{G}/groundQuery")
rel = rock_g.GetCollidersCollectionAPI().CreateIncludesRel()
physx = omni.physx.get_physx_interface()
n = [0]


def cube():
    n[0] += 1
    path = f"/World/rocks/c{n[0]}"
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(0.2)
    c.AddTranslateOp().Set(Gf.Vec3d(n[0] - 20.0, 0.0, 0.5))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(c.GetPrim()).CreateRigidBodyEnabledAttr(False)  # as a rock
    rel.AddTarget(path)
    return path


def steps(k, mark):
    for _ in range(k):
        world.step(render=False)
    print(f"MARK {mark}", flush=True)


det_g = UsdPhysics.CollisionGroup.Define(stage, f"{G}/detail")
det_rel = det_g.GetCollidersCollectionAPI().CreateIncludesRel()
t = [0]


def tile():
    """A collision tile as rock_layer makes one: a mesh, no body, its own group."""
    t[0] += 1
    path = f"/World/tiles/t{t[0]}"
    m = UsdGeom.Mesh.Define(stage, path)
    x = t[0] * 3.0 - 30.0
    m.CreatePointsAttr([Gf.Vec3f(x, -1, -0.5), Gf.Vec3f(x + 2, -1, -0.5),
                        Gf.Vec3f(x + 2, 1, -0.5), Gf.Vec3f(x, 1, -0.5)])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateFaceVertexCountsAttr([4])
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("none")
    det_rel.AddTarget(path)
    return path


from pxr import PhysicsSchemaTools, PhysxSchema  # noqa: E402

EXTRA = os.environ.get("GROUP_EXTRA", "")  # what the rock spike does besides: report, force, read
car = UsdGeom.Cube.Define(stage, "/World/car2")
car.AddTranslateOp().Set(Gf.Vec3d(0.0, -5.0, 0.5))
UsdPhysics.CollisionAPI.Apply(car.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(car.GetPrim())
if "report" in EXTRA:
    PhysxSchema.PhysxContactReportAPI.Apply(car.GetPrim()).CreateThresholdAttr(0.0)
sim = omni.physx.get_physx_simulation_interface()
STAGE_ID = omni.usd.get_context().get_stage_id()
_step = world.step


def _extra_step(*a, **k):
    if "force" in EXTRA:
        sim.apply_force_at_pos(STAGE_ID, PhysicsSchemaTools.sdfPathToInt("/World/car2"),
                               (0.0, 0.0, 1.0), (0.0, -5.0, 0.5))
    r = _step(*a, **k)
    if "read" in EXTRA:
        omni.physx.get_physx_interface().get_rigidbody_transformation("/World/car2")
    if "report" in EXTRA:
        sim.get_contact_report()
    return r


world.step = _extra_step
if "vehicle" in EXTRA:  # aib's Model 3, as the rock spike makes it
    import sys

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, ".."))
    from aib.isaac.vehicle import Vehicle
    from aib.rig import EgoConfig
    from swb_isaac_world import road_layer

    rig = EgoConfig.load(os.path.join(
        HERE, "../deps/autoware_isaac_bridge/src/aib/config/ego.yaml")).load_rig()
    Vehicle(stage, "/World/ego", road_layer.vehicle_params(rig.vehicle), rig.ecu,
            rig.box_center, rig.box_dims, ((0.0, -6.0, 0.05), (1.0, 0.0, 0.0, 0.0)),
            ground_clearance=rig.ground_clearance)
world.reset()
world.play()
steps(30, "start")
four = [cube() for _ in range(4)]
steps(30, "four added in one step")
paths = []
for _ in range(10):
    paths.append(cube())
    steps(1, "added")
steps(30, "ten added")
for p in paths[:5]:  # woken, as a rock in a car's path
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(p)).GetRigidBodyEnabledAttr().Set(True)
steps(30, "five woken")
for p in paths[:5]:
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(p)).GetRigidBodyEnabledAttr().Set(False)
steps(30, "five asleep again")
tiles = [tile() for _ in range(3)]
steps(30, "three tiles added")
stage.RemovePrim(tiles[0])
det_rel.RemoveTarget(tiles[0])
steps(30, "a tile removed")
gone = paths.pop(0)
stage.RemovePrim(gone)
rel.RemoveTarget(gone)
steps(30, "one removed")
paths.append(cube())
steps(30, "one more added")
rel.SetTargets([Sdf.Path(p) for p in paths])
steps(30, "targets set whole")
for p in paths:
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(p)).GetRigidBodyEnabledAttr().Set(True)
steps(120, "all woken")
for p in paths[:5]:  # parked, as rock_layer's pool does with a released rock's prim
    prim = stage.GetPrimAtPath(p)
    UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
steps(30, "five parked")
for p in paths[:5]:  # and taken again
    prim = stage.GetPrimAtPath(p)
    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(True)
steps(30, "five reused")
out = {p: round(float((physx.get_rigidbody_transformation(p).get("position") or [0, 0, 9])[2]), 2)
       for p in paths}
print("GROUP-RESULT " + json.dumps({"fell_through": sum(z < -1.0 for z in out.values()),
                                    "of": len(out), "z": out}), flush=True)
app.close()
