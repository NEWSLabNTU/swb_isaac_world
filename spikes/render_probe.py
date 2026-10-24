"""Render a stage headless, frame by frame: does this host's GPU driver survive it?

scenario-world-bridge issue 005.

    spikes/run_spike.sh cyclonedds <log> spikes/render_probe.py <usd | simple> [frames]

Opens the stage (or `simple`: 50 cubes and a light), waits for it to load and renders
`frames` frames (default 2000) with `app.update()`, no swb code, no ego, no physics.
Prints `PROBE frame <i> <ms>` for the first ten and every 500th, then `PROBE PASS <n>
frames`. On a host whose driver loses the GPU (that issue: Xid 109 then Xid 31, Vulkan
ERROR_DEVICE_LOST) Kit exits within the first frames, with no PASS line; the kernel's
`NVRM: Xid` lines (`journalctl -k`) say which. A SimReady CARLA town is the test that
matters: `$CARLA_USD/Town01/Town01_physics.usda`.
"""
import sys
import time

usd = sys.argv[1] if len(sys.argv) > 1 else "simple"
frames = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

from isaacsim import SimulationApp  # noqa: E402

# Kit uploads a crash's minidump to NVIDIA by default: kept on disk only.
app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})

import omni.usd  # noqa: E402
from isaacsim.core.utils.stage import is_stage_loading  # noqa: E402

t0 = time.perf_counter()
if usd == "simple":
    omni.usd.get_context().new_stage()
    from pxr import UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()
    for i in range(50):
        UsdGeom.Xformable(UsdGeom.Cube.Define(stage, f"/World/c{i}")).AddTranslateOp().Set(
            (i * 3.0, 0.0, 0.0))
    UsdLux.DistantLight.Define(stage, "/World/sun")
else:
    omni.usd.get_context().open_stage(usd)
while is_stage_loading():
    app.update()
print(f"PROBE loaded {usd} in {time.perf_counter() - t0:.1f} s", flush=True)
for i in range(frames):
    t = time.perf_counter()
    app.update()
    if i < 10 or i % 500 == 0:
        print(f"PROBE frame {i} {1000 * (time.perf_counter() - t):.0f} ms", flush=True)
print(f"PROBE PASS {frames} frames", flush=True)
app.close()
