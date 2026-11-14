"""A map's Isaac world (roadmaps 043, 076 A9): `ros2 run swb_isaac_world make_world`.

    ros2 run swb_isaac_world make_world <map dir> [--usd <town usd>] [--town <Town>]
        [--export] [--physics] [--state-dir <dir>] [-o <world.yaml>]

Writes the map's world file, `<map dir>/isaac/world.yaml` (or `-o`), for a CARLA town in USD:

1. the stage: `--usd` (a town's SimReady export `<Town>.usd`, or its physics layer
   `<Town>_physics.usda`), else the town in the CARLA USD set (`SWB_CARLA_USD`, else
   `CARLA_USD`): `<set>/<Town>/<Town>_physics.usda`, else `<set>/<Town>/<Town>.usd`. The town
   is `--town`, else the map directory's name (`Town10` finds `Town10HD` too);
2. `--export`: the town exported again from CARLA's editor first (`CARLA_SRC`: CARLA 0.9.16's
   editor build with SimReady and an `env.sh` exporting `UE4_ROOT` and `CARLA_ROOT`), into
   `<set>/<Town>/`; also done when the set has no `<Town>.usd` and `--export` is given. 15-60
   min a town, one at a time, logged to `<set>/<Town>/export.log`;
3. the physics layer (`physics_layer`): written when the stage is an export with no layer
   beside it, or with `--physics` (again). It goes beside the export, or with `--state-dir`
   into `<state dir>/<map>/isaac/` (a set in a read-only cache);
4. the world file (`usd_world`): ground prims from the stage's semantic labels, and its
   `usd:` a name where it can be: `${SWB_CARLA_USD}/<Town>/...` for a stage in the set.

Runs in Isaac Sim's Python (`pxr`); the `ros2 run` entry sets that up (`scripts/isaac_env.sh`).
`plan` decides what to do without pxr. Replaces `scripts/isaac_town.sh`, which now calls this.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

from . import names

PHYSICS_SUFFIX = "_physics.usda"
EXPORT_DONE = "exported to USD!"  # "<Town> exported to USD!" in the commandlet's log
COMMANDLET_END = "Execution of commandlet took"
EXPORT_EXIT_WAIT = 120.0  # s the editor may take to exit after its export
EXPORT_KILL_WAIT = 60.0  # s after SIGTERM before SIGKILL


class MakeWorldError(RuntimeError):
    """A world that cannot be made; the message says what is missing and how to fix it."""


@dataclass
class Plan:
    town: str
    export_to: Optional[str]  # directory to export the town into, or None
    export: str  # the town's SimReady export, <Town>.usd (may not exist yet)
    layer: str  # the stage the world file names (the physics layer, or --usd's layer)
    write_layer: bool  # write the physics layer `layer` over `export`
    out: str  # the world file
    map_dir: str = ""


def town_candidates(map_dir, town=None):
    """The CARLA level names a map directory may be: `town`, else its name and its name with
    `HD` (CARLA's `Town10HD` is Autoware's `Town10`)."""
    if town:
        return [town]
    name = names.map_name(map_dir)
    return [name, name + "HD"]


def plan(map_dir, usd=None, town=None, export=False, physics=False, state_dir=None,
         out=None, env=None):
    """What `make_world` does (`Plan`), checked before anything runs."""
    map_dir = os.path.abspath(names.expand(map_dir, env))
    if not os.path.isdir(map_dir):
        raise MakeWorldError(f"{map_dir}: no such map directory")
    out = os.path.abspath(names.expand(out, env)) if out else \
        os.path.join(map_dir, "isaac", "world.yaml")
    state = names.state_for(state_dir, map_dir)
    export_to = None
    if usd:
        usd = os.path.abspath(names.expand(usd, env))
        stem = os.path.splitext(os.path.basename(usd))[0]
        if usd.endswith(PHYSICS_SUFFIX):
            town = town or stem[:-len("_physics")]
            src = os.path.join(os.path.dirname(usd), town + ".usd")
            if not os.path.exists(usd) and not os.path.exists(src):
                raise MakeWorldError(f"{usd}: no such layer, and no {src} to write it over")
            return Plan(town, None, src, usd, physics or not os.path.exists(usd), out, map_dir)
        if not os.path.exists(usd):
            raise MakeWorldError(f"{usd}: no such stage")
        town = town or stem
        src = usd
    else:
        root = names.carla_usd_root(env)
        if not root:
            raise MakeWorldError(f"no --usd and {names.CARLA_USD_ENV} not set (the CARLA USD "
                                 "set's root): name the town's USD or the set")
        root = os.path.abspath(os.path.expanduser(root))
        towns = town_candidates(map_dir, town)
        found = [t for t in towns if os.path.exists(os.path.join(root, t, t + ".usd"))
                 or os.path.exists(os.path.join(root, t, t + PHYSICS_SUFFIX))]
        town = found[0] if found else towns[0]
        src = os.path.join(root, town, town + ".usd")
        if export:
            export_to = os.path.dirname(src)
        elif not found:
            raise MakeWorldError(f"no {' or '.join(towns)} in {root} (--town names CARLA's "
                                 "level; --export exports it from CARLA's editor)")
        beside = os.path.join(root, town, town + PHYSICS_SUFFIX)
        if not export and not physics and os.path.exists(beside):
            # the set ships the layer, and its manifest lists it: left as it came
            return Plan(town, None, src, beside, False, out, map_dir)
    if state:
        layer = os.path.join(state, "isaac", town + PHYSICS_SUFFIX)
    else:
        layer = os.path.join(os.path.dirname(src), town + PHYSICS_SUFFIX)
    write = export_to is not None or physics or not os.path.exists(layer)
    return Plan(town, export_to, src, layer, write, out, map_dir)


def export_town(town, out_dir, carla_src, log=print):
    """Export CARLA level `town` with SimReady into `out_dir` (the editor as a commandlet, in
    its own process group); raises MakeWorldError unless its log says it is done."""
    env_sh = os.path.join(carla_src, "env.sh")
    if not os.path.isfile(env_sh):
        raise MakeWorldError(f"no {env_sh} (CARLA_SRC: CARLA 0.9.16's editor build with "
                             "SimReady)")
    busy = subprocess.run(["pgrep", "-x", "UE4Editor-Cmd"], capture_output=True, text=True)
    if busy.stdout.strip():
        raise MakeWorldError(f"a UE editor commandlet already runs (pid "
                             f"{busy.stdout.split()[0]}): one export at a time")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "export.log")
    log(f"make_world: exporting {town} to {out_dir} (log {log_path})")
    # An AMD iGPU trips an ensure that starts CrashReportClient: the NVIDIA ICD, and the
    # -ini: override turns off the editor's unattended crash reports and usage data.
    script = '''
        source "$1/env.sh"
        export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}
        exec nice -n 15 "$UE4_ROOT/Engine/Binaries/Linux/UE4Editor-Cmd" \\
            "$CARLA_ROOT/Unreal/CarlaUE4/CarlaUE4.uproject" -run=SimReadyExport \\
            -AllowCommandletRendering "-map=$2.umap" "-path=$3" -MDL=true -COLLISION=true \\
            -log -Unattended -NoSound -RenderOffScreen -vulkan \\
            "-ini:EditorSettings:[/Script/UnrealEd.CrashReportsPrivacySettings]:bSendUnattendedBugReports=False,[/Script/UnrealEd.AnalyticsPrivacySettings]:bSendUsageData=False"
    '''
    started = time.monotonic()
    with open(log_path, "w") as f:
        proc = subprocess.Popen(["bash", "-c", script, "_", carla_src, town, out_dir],
                                stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)

    def text():
        try:
            with open(log_path, errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def kill_group(sig):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    try:
        # The commandlet's own end, not "LogExit: Exiting" (CrashReportClient logs that too).
        while proc.poll() is None and COMMANDLET_END not in text():
            time.sleep(5)
        try:
            proc.wait(timeout=EXPORT_EXIT_WAIT)
        except subprocess.TimeoutExpired:
            log("make_world: the editor hangs at exit: killing its group")
            kill_group(signal.SIGTERM)
            try:
                proc.wait(timeout=EXPORT_KILL_WAIT)
            except subprocess.TimeoutExpired:
                kill_group(signal.SIGKILL)
                proc.wait()
    except BaseException:
        kill_group(signal.SIGTERM)
        raise
    usd = os.path.join(out_dir, town + ".usd")
    # The commandlet exits 1 on SimReady's material warnings even when the export is done.
    if f"{town} {EXPORT_DONE}" not in text() or not os.path.exists(usd):
        raise MakeWorldError(f"export of {town} failed; see {log_path}")
    log(f"make_world: exported {town} in {time.monotonic() - started:.0f} s")


def run(p, carla_src=None, log=print):
    """Carry out plan `p`; the world file's path."""
    if p.export_to:
        export_town(p.town, p.export_to, carla_src or os.environ.get("CARLA_SRC", ""), log)
    from . import physics_layer, usd_world

    # The export's props are crate 0.7.0; pxr warns once per file without this.
    os.environ.setdefault("PXR_USDC_EMIT_DEPRECATION_WARNINGS", "0")
    if p.write_layer:
        if not os.path.exists(p.export):
            raise MakeWorldError(f"{p.export}: no export to write {p.layer} over")
        os.makedirs(os.path.dirname(p.layer), exist_ok=True)
        meshes, instancers = physics_layer.write_layer(os.path.realpath(p.export), p.layer)
        log(f"{p.layer}: colliders on {meshes} mesh(es); {instancers} PointInstancer(s) left "
            "without")
    else:
        log(f"make_world: {p.layer} kept (--physics writes it again)")
    out, n = usd_world.write(p.map_dir, p.layer, p.out)
    log(f"{n} ground prim(s); wrote {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="make_world", description=__doc__.splitlines()[0])
    ap.add_argument("map_dir", help="the Autoware map directory the world goes with")
    ap.add_argument("--usd", default="",
                    help="the town's USD: its SimReady export <Town>.usd or its physics layer "
                         "<Town>_physics.usda (default: the town in $SWB_CARLA_USD)")
    ap.add_argument("--town", default="", help="CARLA's level name (default: the map's name)")
    ap.add_argument("--export", action="store_true",
                    help="export the town from CARLA's editor first (CARLA_SRC)")
    ap.add_argument("--physics", action="store_true",
                    help="write the physics layer again even if it exists")
    ap.add_argument("--state-dir", default="",
                    help="write a new physics layer into <dir>/<map>/isaac/ instead of beside "
                         "the export")
    ap.add_argument("-o", "--output", default="",
                    help="the world file (default <map dir>/isaac/world.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = ap.parse_args(argv)
    try:
        p = plan(args.map_dir, args.usd or None, args.town or None, args.export, args.physics,
                 args.state_dir or None, args.output or None)
        if args.dry_run:
            for k, v in vars(p).items():
                print(f"{k}: {v}")
            return 0
        run(p)
    except (MakeWorldError, names.UnresolvedName, ValueError) as e:
        print(f"make_world: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
