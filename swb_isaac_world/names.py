"""Names, not paths (scenario-world-bridge roadmap 076 A9): where a world's files are, and
where the world server writes. Pure Python, no Isaac.

**Names in a world file.** A world file's `usd:` (`<map>/isaac/world.yaml`) may name its stage
through the environment: `${SWB_CARLA_USD}/Town01/Town01_physics.usda`, `$HOME/...`, `~/...`.
`expand` resolves them, and refuses a variable that is not set by name. A relative `usd:` is
relative to the map directory, as before.

**The CARLA USD set.** `SWB_CARLA_USD` is the set's root (a project's `assets/carla-usd`; the
stack sets it). `CARLA_USD`, the older host setting, is read when it is not set.

**Course worlds' props.** A course's world (`swb_isaac_world.course.usd`) references CARLA's
props as `./carla-usd/<asset>`, through a link named `carla-usd` beside the world. A world
downloaded from a dataset has no link (a dataset holds no links) and may sit in a read-only
cache. `staged_world` then gives the world server a stand-in directory outside the map
directory: a link to every entry beside the world, plus `carla-usd` -> the set's root, and the
stage is loaded from there. USD anchors `./carla-usd/...` at the directory the layer was opened
from, not at the file's real place, so the props resolve (checked with Isaac Sim 6.1's USD
26.08). A world whose link is present, or that names no `carla-usd/` asset, is loaded as it is.

**The state directory.** What the world server writes for a map (the soil's ruts) and what a
project adds to a map (its own `surfaces.yaml`, `road_detail.yaml`, rut sets) live in
`<state dir>/<map>/` (the world server's `--state-dir`; the stack passes `<project>/state`),
never in the map directory. `<map>` is the map directory's name. Without a state directory the
world server writes into the map directory, as it always has.
"""
import hashlib
import os
import re

CARLA_USD_ENV = "SWB_CARLA_USD"
CARLA_USD_OLD_ENV = "CARLA_USD"  # the host setting before roadmap 076
LINK = "carla-usd"  # beside a course's world (course.usd.LINK)
MARKER = LINK.encode() + b"/"  # how a world's layer names an asset through the link
# Files larger than this are not scanned for the link (a CARLA town's export is gigabytes and
# never uses it); a course's stage is tens of megabytes.
SCAN_LIMIT = 512 * 2 ** 20
_VAR = re.compile(r"\$(\w+)|\$\{([^}]*)\}")


class UnresolvedName(ValueError):
    """A name that cannot be resolved; the message says which and how to fix it."""


def expand(text, env=None):
    """`text` with `~` and every `$VAR` / `${VAR}` replaced from `env` (default
    `os.environ`); a variable that is not set is refused by name."""
    env = os.environ if env is None else env
    missing = []

    def sub(m):
        name = m.group(1) or m.group(2)
        if name == CARLA_USD_ENV and name not in env and env.get(CARLA_USD_OLD_ENV):
            return env[CARLA_USD_OLD_ENV]  # the older host setting names the same set
        if name not in env:
            missing.append(name)
            return m.group(0)
        return env[name]

    out = _VAR.sub(sub, str(text))
    if missing:
        hint = (" (the CARLA USD set's root; the stack sets it, or export it)"
                if CARLA_USD_ENV in missing else "")
        raise UnresolvedName(f"{text}: {', '.join(sorted(set(missing)))} not set{hint}")
    return os.path.expanduser(out)


def carla_usd_root(env=None):
    """The CARLA USD set's root: `SWB_CARLA_USD`, else `CARLA_USD`, else None."""
    env = os.environ if env is None else env
    return env.get(CARLA_USD_ENV) or env.get(CARLA_USD_OLD_ENV) or None


def world_file(entry):
    """The world file a `--world` entry names (a map directory's `isaac/world.yaml`), or
    None for a USD file."""
    path = os.path.abspath(expand(entry))
    if os.path.isdir(path):
        path = os.path.join(path, "isaac", "world.yaml")
    return path if path.endswith((".yaml", ".yml")) else None


def map_dir_of(world_file_path):
    """The map directory of a world file `<map>/isaac/world.yaml`."""
    return os.path.dirname(os.path.dirname(os.path.abspath(world_file_path)))


def map_name(map_dir):
    """A map's name: its directory's (as linked, `assets/maps/<map>`, not the cache's)."""
    return os.path.basename(os.path.normpath(os.path.abspath(map_dir)))


def world_usd_of(world_file_path, data=None, env=None):
    """The stage a world file names: its `usd:` (or `world:`) with names expanded, relative to
    the map directory; None when it names none (a flat ground plane)."""
    if data is None:
        import yaml

        with open(world_file_path) as f:
            data = yaml.safe_load(f) or {}
    usd = data.get("usd") or data.get("world") or ""
    if not usd:
        return None
    usd = expand(usd, env)
    if not os.path.isabs(usd):
        usd = os.path.join(map_dir_of(world_file_path), usd)
    return os.path.normpath(usd)


def requested_usd(usd, map_dir=None, env=None):
    """The stage a `LoadWorld` names, as a path: `usd` with names expanded; when that does not
    exist and the request names a map directory with a world file, that file's stage (a
    bridge that joined an unexpanded `${SWB_CARLA_USD}/...` to the map directory)."""
    if not usd:
        return usd
    try:
        path = expand(usd, env)
    except UnresolvedName:
        path = usd
    if os.path.exists(path) or not map_dir:
        return path
    wf = os.path.join(map_dir, "isaac", "world.yaml")
    if os.path.isfile(wf):
        named = world_usd_of(wf, env=env)
        if named and os.path.exists(named):
            return named
    return path


def state_for(state_dir, map_dir):
    """`<state dir>/<map>` for a map directory, or None without a state directory."""
    if not state_dir:
        return None
    return os.path.join(os.path.abspath(expand(state_dir)), map_name(map_dir))


def find_file(name, map_dir, state=None):
    """A map's file `name` (`surfaces.yaml`, `road_detail.yaml`): the state directory's
    (`<state dir>/<map>/<name>`, a project's own) over the map directory's. The path it would
    be at in the map directory when neither has it."""
    if state:
        p = os.path.join(state, name)
        if os.path.exists(p):
            return p
    return os.path.join(map_dir, name)


def uses_link(usd, limit=SCAN_LIMIT):
    """Whether stage file `usd` names an asset through a `carla-usd/` link (a course's props):
    its bytes hold `carla-usd/`, which a .usda or .usdc stores as it is. A file over `limit`
    bytes is taken not to."""
    try:
        if os.path.getsize(usd) > limit:
            return False
        tail = b""
        with open(usd, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    return False
                if MARKER in tail + chunk:
                    return True
                tail = chunk[-len(MARKER):]
    except OSError:
        return False


def stand_in_dir(usd, state=None, env=None):
    """Where a stand-in for `usd`'s directory goes: `<state>/isaac` for a map with a state
    directory, else under the user's cache (`$XDG_CACHE_HOME/swb_isaac_world/worlds/`)."""
    if state:
        return os.path.join(state, "isaac")
    env = os.environ if env is None else env
    cache = env.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    real = os.path.realpath(os.path.dirname(usd))
    key = hashlib.sha1(real.encode()).hexdigest()[:12]
    return os.path.join(cache, "swb_isaac_world", "worlds",
                        f"{os.path.basename(real)}-{key}")


def staged_world(usd, state=None, env=None):
    """The path to load stage `usd` from (module docstring, "Course worlds' props"): `usd`
    itself, or its stand-in when it names assets through a `carla-usd/` link that is missing
    beside it. Refuses a stand-in it cannot make (no CARLA USD set named)."""
    here = os.path.dirname(os.path.abspath(usd))
    if os.path.exists(os.path.join(here, LINK)) or not uses_link(usd):
        return usd
    root = carla_usd_root(env)
    if not root:
        raise UnresolvedName(f"{usd}: its props are under ./{LINK}/, which is not beside it, and "
                         f"{CARLA_USD_ENV} is not set (the CARLA USD set's root)")
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise UnresolvedName(f"{usd}: {CARLA_USD_ENV}={root} is not a directory")
    out = stand_in_dir(usd, state, env)
    os.makedirs(out, exist_ok=True)
    want = {e: os.path.join(here, e) for e in os.listdir(here) if e != LINK}
    want[LINK] = root
    for e in os.listdir(out):
        p = os.path.join(out, e)
        if os.path.islink(p) and (e not in want or os.readlink(p) != want[e]):
            os.remove(p)
    for e, target in want.items():
        p = os.path.join(out, e)
        if not os.path.lexists(p):
            os.symlink(target, p)
    return os.path.join(out, os.path.basename(usd))
