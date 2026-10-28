"""C8: a course's run, its ground and its drift points: `<map dir>/scenarios/<name>_down.xosc`,
`<map dir>/surfaces.yaml` and `<map dir>/course/drift_points.yaml`.

The run goes down the section in its lane (Japan: the left half), from `START` m after the
section's first point to `END` m before its last, Autoware driving drift_mustang. The ground:
the road asphalt, its lines asphalt, the terrain grass (the world's labels; no zones yet).
A drift point per hairpin at the start of its arc, where the curvature first passes half its
apex's, with the hairpin's radius (signed: > 0 left) and turn: drift_controller's `points:`
entries.
"""
import math
import os

import numpy as np

START, END = 20.0, 30.0   # m from the section's ends
TIMEOUT = 600             # s, simulated

XOSC = """<?xml version="1.0"?>
<OpenSCENARIO>
\t<FileHeader author="swb_isaac_world.course" date="2026-10-29T00:00:00+00:00" description="{title}: down the section in drift_mustang, Autoware driving (scenario-world-bridge roadmap 058)" revMajor="1" revMinor="2" />
\t<ParameterDeclarations />
\t<CatalogLocations><VehicleCatalog><Directory path="$(find-pkg-share sim_vehicle_spec)/catalogs" /></VehicleCatalog></CatalogLocations>
\t<RoadNetwork>
\t\t<LogicFile filepath="{map_dir}" />
\t</RoadNetwork>
\t<Entities>
\t\t<ScenarioObject name="ego">
\t\t\t<CatalogReference catalogName="vehicles" entryName="drift_mustang">
\t\t\t\t<ParameterAssignments>
\t\t\t\t\t<ParameterAssignment parameterRef="maxSpeed" value="50" />
\t\t\t\t\t<ParameterAssignment parameterRef="maxAcceleration" value="15.0" />
\t\t\t\t\t<ParameterAssignment parameterRef="maxDeceleration" value="15.0" />
\t\t\t\t\t<ParameterAssignment parameterRef="maxSteering" value="0.5" />
\t\t\t\t</ParameterAssignments>
\t\t\t</CatalogReference>
\t\t\t<ObjectController>
\t\t\t\t<Controller name="Autoware">
\t\t\t\t\t<Properties>
\t\t\t\t\t\t<Property name="maxJerk" value="1.5" />
\t\t\t\t\t\t<Property name="minJerk" value="-1.5" />
\t\t\t\t\t</Properties>
\t\t\t\t</Controller>
\t\t\t</ObjectController>
\t\t</ScenarioObject>
\t</Entities>
\t<Storyboard>
\t\t<Init>
\t\t\t<Actions>
\t\t\t\t<Private entityRef="ego">
\t\t\t\t\t<PrivateAction>
\t\t\t\t\t\t<TeleportAction>
\t\t\t\t\t\t\t<Position>
\t\t\t\t\t\t\t\t<WorldPosition x="{x0:.2f}" y="{y0:.2f}" z="{z0:.2f}" h="{h0:.5f}" p="0" r="0" />
\t\t\t\t\t\t\t</Position>
\t\t\t\t\t\t</TeleportAction>
\t\t\t\t\t</PrivateAction>
\t\t\t\t\t<PrivateAction>
\t\t\t\t\t\t<RoutingAction>
\t\t\t\t\t\t\t<AcquirePositionAction>
\t\t\t\t\t\t\t\t<Position>
\t\t\t\t\t\t\t\t\t<WorldPosition x="{x1:.2f}" y="{y1:.2f}" z="{z1:.2f}" h="{h1:.5f}" p="0" r="0" />
\t\t\t\t\t\t\t\t</Position>
\t\t\t\t\t\t\t</AcquirePositionAction>
\t\t\t\t\t\t</RoutingAction>
\t\t\t\t\t</PrivateAction>
\t\t\t\t</Private>
\t\t\t</Actions>
\t\t</Init>
\t\t<Story name="{name}_down">
\t\t\t<Act name="drive_act">
\t\t\t\t<ManeuverGroup maximumExecutionCount="1" name="ego_maneuvers">
\t\t\t\t\t<Actors selectTriggeringEntities="false">
\t\t\t\t\t\t<EntityRef entityRef="ego" />
\t\t\t\t\t</Actors>
\t\t\t\t\t<Maneuver name="end_conditions">
\t\t\t\t\t\t<Event name="success" priority="parallel">
\t\t\t\t\t\t\t<Action name="exit_success">
\t\t\t\t\t\t\t\t<UserDefinedAction>
\t\t\t\t\t\t\t\t\t<CustomCommandAction type="exitSuccess" />
\t\t\t\t\t\t\t\t</UserDefinedAction>
\t\t\t\t\t\t\t</Action>
\t\t\t\t\t\t\t<StartTrigger>
\t\t\t\t\t\t\t\t<ConditionGroup>
\t\t\t\t\t\t\t\t\t<Condition name="reach_goal" delay="0" conditionEdge="none">
\t\t\t\t\t\t\t\t\t\t<ByEntityCondition>
\t\t\t\t\t\t\t\t\t\t\t<TriggeringEntities triggeringEntitiesRule="any">
\t\t\t\t\t\t\t\t\t\t\t\t<EntityRef entityRef="ego" />
\t\t\t\t\t\t\t\t\t\t\t</TriggeringEntities>
\t\t\t\t\t\t\t\t\t\t\t<EntityCondition>
\t\t\t\t\t\t\t\t\t\t\t\t<ReachPositionCondition tolerance="5.0">
\t\t\t\t\t\t\t\t\t\t\t\t\t<Position>
\t\t\t\t\t\t\t\t\t\t\t\t\t\t<WorldPosition x="{x1:.2f}" y="{y1:.2f}" z="{z1:.2f}" h="{h1:.5f}" p="0" r="0" />
\t\t\t\t\t\t\t\t\t\t\t\t\t</Position>
\t\t\t\t\t\t\t\t\t\t\t\t</ReachPositionCondition>
\t\t\t\t\t\t\t\t\t\t\t</EntityCondition>
\t\t\t\t\t\t\t\t\t\t</ByEntityCondition>
\t\t\t\t\t\t\t\t\t</Condition>
\t\t\t\t\t\t\t\t</ConditionGroup>
\t\t\t\t\t\t\t</StartTrigger>
\t\t\t\t\t\t</Event>
\t\t\t\t\t\t<Event name="timeout" priority="parallel">
\t\t\t\t\t\t\t<Action name="exit_failure">
\t\t\t\t\t\t\t\t<UserDefinedAction>
\t\t\t\t\t\t\t\t\t<CustomCommandAction type="exitFailure" />
\t\t\t\t\t\t\t\t</UserDefinedAction>
\t\t\t\t\t\t\t</Action>
\t\t\t\t\t\t\t<StartTrigger>
\t\t\t\t\t\t\t\t<ConditionGroup>
\t\t\t\t\t\t\t\t\t<Condition name="timeout_condition" delay="0" conditionEdge="none">
\t\t\t\t\t\t\t\t\t\t<ByValueCondition>
\t\t\t\t\t\t\t\t\t\t\t<SimulationTimeCondition value="{timeout}" rule="greaterThan" />
\t\t\t\t\t\t\t\t\t\t</ByValueCondition>
\t\t\t\t\t\t\t\t\t</Condition>
\t\t\t\t\t\t\t\t</ConditionGroup>
\t\t\t\t\t\t\t</StartTrigger>
\t\t\t\t\t\t</Event>
\t\t\t\t\t</Maneuver>
\t\t\t\t</ManeuverGroup>
\t\t\t\t<StartTrigger>
\t\t\t\t\t<ConditionGroup>
\t\t\t\t\t\t<Condition name="start" delay="0" conditionEdge="none">
\t\t\t\t\t\t\t<ByValueCondition>
\t\t\t\t\t\t\t\t<SimulationTimeCondition value="0" rule="greaterThan" />
\t\t\t\t\t\t\t</ByValueCondition>
\t\t\t\t\t\t</Condition>
\t\t\t\t\t</ConditionGroup>
\t\t\t\t</StartTrigger>
\t\t\t</Act>
\t\t</Story>
\t\t<StopTrigger />
\t</Storyboard>
</OpenSCENARIO>
"""


def _lane_point(c, s, offset):
    """(x, y, z, heading) at s along the centreline, `offset` m to its left."""
    i = int(np.clip(round(s / (c["s"][1] - c["s"][0])), 0, len(c["s"]) - 1))
    h = float(c["heading"][i])
    return (float(c["x"][i] - offset * math.sin(h)), float(c["y"][i] + offset * math.cos(h)),
            float(c["z"][i]), math.remainder(h, math.tau))


def drift_points(c, hairpins, lane_off):
    """On the lane (`lane_off` m left of the centreline), its radius the lane's."""
    k = np.abs(c["curvature"])
    pts = []
    for n, hp in enumerate(hairpins):
        i = int(round(hp["s"] / (c["s"][1] - c["s"][0])))
        j = i
        while j > 0 and k[j] > 0.5 * k[i]:
            j -= 1
        # drift_controller's radius is signed: > 0 a left turn (counterclockwise, the map's z up)
        left = hp.get("direction", "left" if c["curvature"][i] > 0 else "right") == "left"
        x, y, _, _ = _lane_point(c, float(c["s"][j]), lane_off)
        r = hp["radius"] - lane_off if left else -(hp["radius"] + lane_off)  # left: the inside
        pts.append({"name": f"H{n + 1}", "x": round(x, 2), "y": round(y, 2), "radius": round(r, 1),
                    "angle_deg": round(abs(hp["turn_deg"]))})
    return pts


def build(c, hairpins, course, map_dir):
    import yaml

    r = course["road"]
    lane_off = r["lane_width"] / 2  # the left lane's middle
    s_end = float(c["s"][-1])
    x0, y0, z0, h0 = _lane_point(c, START, lane_off)
    x1, y1, z1, h1 = _lane_point(c, s_end - END, lane_off)
    os.makedirs(os.path.join(map_dir, "scenarios"), exist_ok=True)
    name = course["name"]
    path = os.path.join(map_dir, "scenarios", f"{name}_down.xosc")
    with open(path, "w") as f:
        f.write(XOSC.format(title=name, map_dir=os.path.realpath(map_dir), name=name,
                            x0=x0, y0=y0, z0=z0 + 0.3, h0=h0, x1=x1, y1=y1, z1=z1, h1=h1,
                            timeout=TIMEOUT))
    with open(os.path.join(map_dir, "surfaces.yaml"), "w") as f:
        f.write("# Generated by swb_isaac_world.course (roadmap 058): the world's own labels, no zones yet.\n")
        yaml.safe_dump({"defaults": {"Road": "asphalt", "RoadLine": "asphalt", "Terrain": "grass",
                                     "SideWalk": "concrete", "Ground": "dirt"}, "zones": []}, f,
                       sort_keys=False)
    pts = drift_points(c, hairpins, lane_off)
    with open(os.path.join(map_dir, "course", "drift_points.yaml"), "w") as f:
        f.write(f"# Generated by swb_isaac_world.course (roadmap 058): {name}'s hairpins, each at the start\n"
                "# of its arc, for autoware-drift's drift_controller (points: name, x, y, radius, angle_deg).\n")
        yaml.safe_dump({"frame": "map", "points": pts}, f, sort_keys=False, default_flow_style=None)
    return {"scenario": path, "start": [round(x0, 1), round(y0, 1)], "goal": [round(x1, 1), round(y1, 1)],
            "drift_points": len(pts)}
