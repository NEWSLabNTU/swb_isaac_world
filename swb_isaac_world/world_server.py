"""The Isaac Sim world server (roadmaps 037, 046): `python -m swb_isaac_world.world_server`.

Starts Isaac Sim (headless unless `--display`), builds the world, and serves the world
protocol on a ZMQ REP socket bound to localhost (the shell `swb_world_server`, which holds the
owner lease) until a client sends `Shutdown` or the process gets SIGINT/SIGTERM; it also
ends when its parent does (a supervisor killed hard). Run it through `just isaac-world`, which
sets Isaac's environment.

    python -m swb_isaac_world.world_server --port 5590 [--display :8] [--ego <aib ego config>]
        [--world <world>[,<world>...]] [--follow]

A world is a world file (`<map>/isaac/world.yaml`, as swb's `LoadWorld` reads it), a map
directory (its `isaac/world.yaml`) or a USD file. Several, comma separated, are all loaded and
rendered once before the server answers (roadmap 043: a town's first render compiles its
materials, minutes, which a scenario's `LoadWorld` should not have to); the first one listed
stays loaded. A world file's map directory with a `surfaces.yaml` (roadmap 047) gives that
world its surface map, checked here before Isaac starts.
"""
import argparse
import logging
import os
import sys
import threading
import time

# A world server that cannot go on exits with this (EX_TEMPFAIL, as swb_carla_world's): its
# launch respawns it fresh. docs/issues/011: one stopped answering after RTX's "Out of
# resource descriptors!", another sat in Kit's start for 16 minutes; neither exited.
EXIT_FRESH = 75
# Kit's start, the worlds' preload and their first renders, s: a cold Town01 takes about 7
# minutes; isaac_up.sh waits 30 (ISAAC_READY_TIMEOUT) for an answer.
STARTUP_LIMIT = 1200.0


def world_file(entry):
    """The world file a `--world` entry names (a map directory's `isaac/world.yaml`), or
    None for a USD file."""
    path = os.path.abspath(os.path.expanduser(entry))
    if os.path.isdir(path):
        path = os.path.join(path, "isaac", "world.yaml")
    return path if path.endswith((".yaml", ".yml")) else None


def world_usd(entry):
    """The USD stage a `--world` entry names: a world file's `usd` (relative to its map
    directory, as swb resolves it), a map directory's `isaac/world.yaml`, or a USD file."""
    path = world_file(entry)
    if path is None:
        return os.path.abspath(os.path.expanduser(entry))
    import yaml

    with open(path) as f:
        usd = (yaml.safe_load(f) or {}).get("usd", "")
    if not usd:
        raise SystemExit(f"{entry}: the world file names no usd (a flat ground plane needs no "
                         "preload)")
    map_dir = os.path.dirname(os.path.dirname(path))
    return usd if os.path.isabs(usd) else os.path.join(map_dir, usd)


def preload_list(worlds):
    """`--world`'s comma-separated entries as USD paths, checked before Isaac starts."""
    usds = list(dict.fromkeys(world_usd(e.strip()) for e in worlds.split(",") if e.strip()))
    missing = [u for u in usds if not os.path.exists(u)]
    if missing:
        raise SystemExit(f"world(s) not found: {', '.join(missing)}")
    return usds


def exit_fresh(log, why):
    """Leave now, for a fresh process (`EXIT_FRESH`): logs flushed, no clean-up, which a
    wedged renderer would hang in."""
    log.error("%s: exiting for a fresh world server", why)
    for h in logging.getLogger().handlers:
        h.flush()
    sys.stdout.flush()
    os._exit(EXIT_FRESH)


def startup_watchdog(seconds, on_timeout):
    """A started timer calling `on_timeout` unless cancelled within `seconds`."""
    timer = threading.Timer(seconds, on_timeout)
    timer.daemon = True
    timer.start()
    return timer


def fatal_consumer(on_fatal):
    """An omni.log message consumer calling `on_fatal(channel, message)` for a FATAL one."""
    def consume(channel, level, module, filename, func, line, msg, *rest):
        if getattr(level, "name", level) == "FATAL":
            on_fatal(channel, msg)
    return consume


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=5590, help="REP port on 127.0.0.1")
    ap.add_argument("--display", default="",
                    help="X display for Isaac's window (e.g. :8, the VNC one); empty: headless")
    ap.add_argument("--follow", action="store_true",
                    help="with --display: the viewport follows the ego from behind, no UI "
                         "(for watching or recording a run)")
    ap.add_argument("--world", default="",
                    help="worlds to load and render before serving, comma separated: world "
                         "files, map directories or USD files (a first render compiles the "
                         "world's materials); the first stays loaded, and swb's LoadWorld of "
                         "it keeps it")
    ap.add_argument("--ego", default="",
                    help="aib ego config (autoware_isaac_bridge's config/ego.yaml): the ego "
                         "is aib's PhysX vehicle driven by Autoware (038, 045); empty: a "
                         "kinematic box (037). aib must be on PYTHONPATH")
    ap.add_argument("--ego-domain", type=int, default=0,
                    help="the ego's ROS domain, over the ego config's (the ego stack's "
                         "ROS_DOMAIN_ID); 0: the config's")
    ap.add_argument("--record", default="",
                    help="a directory: each scenario's first ego's dash and wheel pictures "
                         "and soil state, every rendered frame (`recorder`), for a video; "
                         "empty: none")
    ap.add_argument("--physics-hz", type=float, default=60.0,
                    help="the physics rate, Hz: PhysX, the egos' onboard computers and the soil "
                         "step at it, and the bridge splits SSv2's step into steps no longer "
                         "than 1 / this (Hello's max_physics_dt). The ego's commands are taken "
                         "once per SSv2 step: a faster controller needs a shorter SSv2 step")
    args = ap.parse_args(argv)
    if not args.physics_hz > 0:
        raise SystemExit(f"--physics-hz {args.physics_hz}: a rate > 0")
    from swb_world_server import lifetime
    from swb_world_server.server import bind, run

    lifetime.die_with_parent()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("swb_isaac_world")

    preload = preload_list(args.world)
    socket = bind(args.port)
    from .surface_layer import register
    from .surfaces import SurfaceError

    entries = [e.strip() for e in args.world.split(",") if e.strip()]
    try:
        # A world's surface map (<map>/surfaces.yaml, roadmap 047), checked before Isaac starts.
        surface_maps = register([w for w in map(world_file, entries) if w])
    except SurfaceError as e:
        raise SystemExit(f"surface map refused:\n{e}") from e
    for usd, (path, _) in surface_maps.items():
        log.info("surface map %s for %s%s", path, usd,
                 "" if os.path.exists(path) else " (none yet: looked for at each LoadWorld)")
    if args.display:
        os.environ["DISPLAY"] = args.display
    else:
        os.environ.pop("DISPLAY", None)
    started = time.monotonic()
    watchdog = startup_watchdog(STARTUP_LIMIT, lambda: exit_fresh(
        log, f"not up after {STARTUP_LIMIT:.0f} s (Kit's start and the preload)"))
    from isaacsim import SimulationApp

    # Kit uploads a crash's minidump to NVIDIA by default: kept on disk only (its dumpDir).
    config = {"headless": not args.display, "extra_args": ["--/crashreporter/url="]}
    # Host-specific Kit settings (a host setting, ISAAC_KIT_ARGS): e.g. one GPU on a host
    # whose other GPU is busy, "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=1".
    config["extra_args"] += os.environ.get("ISAAC_KIT_ARGS", "").split()
    if args.display and args.follow:
        # A small render keeps the frame inside its 50 ms beside the lidar (1080p: 49.5 ms
        # p50 on a loaded host, 720p: 52); the window scales it up.
        config.update(hide_ui=True, width=854, height=480, window_width=1920,
                      window_height=1080)
    app = SimulationApp(config)
    import omni.log

    # Kept referenced: the consumer lives as long as its handle.
    fatal_watch = omni.log.get_log().add_message_consumer(fatal_consumer(  # noqa: F841
        lambda channel, msg: exit_fresh(log, f"Kit fatal from {channel}: {msg}")))

    from .world import IsaacWorld

    world = IsaacWorld(physics_dt=1 / args.physics_hz,
                       ego_config=args.ego or None, ego_domain=args.ego_domain or None,
                       follow=bool(args.display and args.follow), surface_maps=surface_maps,
                       record=args.record or None)
    # The first listed is preloaded last, so it is the one loaded when the bridge first asks.
    for usd in reversed(preload):
        world.preload(usd)
    watchdog.cancel()
    log.info("Isaac Sim %s up in %.1f s, physics at %g Hz", world.isaac_version,
             time.monotonic() - started, 1 / world.max_physics_dt)
    try:
        run(world, socket, should_stop=lambda: not app.is_running())
    finally:
        app.close()


if __name__ == "__main__":
    main()
