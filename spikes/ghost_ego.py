"""Does a despawned ego stay in the RTX lidar's scene (roadmap 052 step 1)?

    ISAAC_ROOT=... run_spike.sh fastrtps <log> $PWD/src/swb_isaac_world/spikes/ghost_ego.py \\
        [<Town01 world.yaml>]
    # prints one SPIKE-RESULT json

044 step 7 found that a prim deleted where it stands stays in the RTX lidar's scene at its
last pose, and `IsaacWorld.despawn` buries NPCs and props first for it, but not egos. Here
the world server's own `IsaacWorld` loads Town01, a probe lidar (not the ego's) looks along
+x at an empty stretch of road, and the points it returns inside the box an ego takes at
POSE are counted:

  empty      the world loaded, nothing spawned
  present    an ego spawned at POSE
  despawned  the ego despawned as a scenario's end does
  reloaded   LoadWorld again, as the next scenario does
  next_ego   the next scenario's ego, spawned elsewhere (GHOST_NEXT_X), at the same slot

A ghost is `despawned` or later counting like `present` rather than like `empty` in its last
frames (`last`, `frames_over_5`; `in_box` is the most over the phase, which catches the
lidar's latency). GHOST_INPLACE=1 deletes the ego where it stands without the rest of
`despawn` (it does the same today).

Result (2026-10-24, 60 frames a phase, both modes the same): empty 0, present 137-142 in the
box in every frame, despawned 3 frames over 5 points and then 1, reloaded and next_ego 0-1.
A despawned ego leaves no ghost. 044's were NPCs and props; why they differ is not known
(theirs are instanceable bodies, the ego's is not).
"""
import json
import os
import sys

os.environ.pop("DISPLAY", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})
import logging  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from swb_isaac_world.surface_layer import register  # noqa: E402
from swb_isaac_world.world import IsaacWorld  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
WORLD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "data/carla-autoware-bridge/Town01/isaac/world.yaml")
EGO_YAML = os.path.join(REPO, "src/autoware_isaac_bridge/src/aib/config/ego.yaml")
FRAMES = int(os.environ.get("GHOST_FRAMES", "60"))
X, Y = 200.0, -130.0  # town01_soil's road
POSE = ((X, Y, 0.3), (0.0, 0.0, 0.0, 1.0))
NEXT = ((float(os.environ.get("GHOST_NEXT_X", "240.0")), Y, 0.3), (0.0, 0.0, 0.0, 1.0))
PROBE = (X - 15.0, Y, 2.0)  # 15 m behind POSE, looking along +x

with open(WORLD) as f:
    w = yaml.safe_load(f)
usd, ground = os.path.realpath(w["usd"]), list(w.get("ground_paths") or [])
assert register([WORLD])
world = IsaacWorld(ego_config=EGO_YAML, ego_domain=int(os.environ.get("GHOST_DOMAIN", "77")),
                   surface_maps={})
world.preload(usd)
world.load(usd, ground)

import omni.replicator.core as rep  # noqa: E402
from isaacsim.sensors.experimental.rtx import Lidar as RtxLidar  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

probe = RtxLidar.create("/World/ghost_probe", config="OS1", variant="OS1_REV7_128ch20hz1024res")
_xf = UsdGeom.Xformable(world.stage.GetPrimAtPath("/World/ghost_probe"))
_xf.ClearXformOpOrder()
_xf.AddTranslateOp().Set(Gf.Vec3d(*PROBE))
print("probe at", _xf.ComputeLocalToWorldTransform(0).ExtractTranslation(), flush=True)
rp = rep.create.render_product(probe.paths[0], resolution=(128, 128))
ann = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloud")
ann.attach([rp.path])


def count():
    """Points in the ego's box at POSE (sensor frame: the probe is axis-aligned), the most over
    FRAMES frames, and how many returned at all."""
    best, total, seen, last = 0, 0, 0, 0
    for _ in range(FRAMES):
        world.step({}, 1, 1 / 60, True)
        d = ann.get_data()
        p = np.asarray(d.get("data", []) if isinstance(d, dict) else [], dtype=np.float64)
        if p.size == 0:
            continue
        q = p.reshape(-1, 3)
        LAST["q"] = q
        n = 0
        for o in (np.array(PROBE), np.zeros(3)):  # sensor frame, or already the world's
            w = q + o
            n = max(n, int(((np.abs(w[:, 0] - X) < 2.6) & (np.abs(w[:, 1] - Y) < 1.2)
                            & (w[:, 2] > 0.4) & (w[:, 2] < 2.0)).sum()))
        best, total, seen, last = max(best, n), max(total, len(q)), seen + (n > 5), n
    return {"in_box": best, "last": last, "frames_over_5": seen, "points": total}


LAST = {}


def dump(name):
    q = LAST.get("q")
    if q is not None:
        np.save(os.path.join(os.environ.get("GHOST_OUT", "/tmp"), f"ghost_{name}.npy"), q)


out = {"inplace": os.environ.get("GHOST_INPLACE") == "1"}
if out["inplace"]:
    # The despawn before 052 step 1's fix, for the comparison: an ego deleted where it stands.
    def despawn_in_place(handle, _despawn=world.despawn):
        if handle not in world.egos:
            return _despawn(handle)
        path, _, _ = world.entities.pop(handle)
        world.by_path.pop(path, None)
        world.egos.pop(handle).close()
        world.reporters.discard(handle)
        world.stage.RemovePrim(path)

    world.despawn = despawn_in_place
out["empty"] = count()
print("empty", out["empty"], flush=True)
dump("empty")
world.spawn(101, "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5), POSE)
out["present"] = count()
print("present", out["present"], flush=True)
dump("present")
world.despawn(101)
out["despawned"] = count()
print("despawned", out["despawned"], flush=True)
dump("despawned")
world.load(usd, ground)
probe_prim = world.stage.GetPrimAtPath("/World/ghost_probe")
out["probe_survived_load"] = probe_prim.IsValid()
out["reloaded"] = count()
print("reloaded", out["reloaded"], flush=True)
world.spawn(102, "ego", "EGO", "", (1.5, 0.0, 0.9), (4.8, 2.1, 1.5), NEXT)
out["next_ego"] = count()
print("next_ego", out["next_ego"], flush=True)
world.despawn(102)
print("SPIKE-RESULT " + json.dumps(out), flush=True)
app.close()
