# swb_isaac_world

[Isaac Sim](https://developer.nvidia.com/isaac/sim) 6.1 behind the world protocol of
[scenario-world-bridge](https://github.com/NEWSLabNTU/scenario-world-bridge) (swb): the Isaac
**world server**. It loads a world (CARLA's towns exported to USD, or a map's point cloud as
voxels), spawns and steps the scenario's vehicles, people and props in PhysX, sets traffic
lights, the scenario's weather and time of day as sky, fog and rain, and road surfaces
(off-road soil, per-wheel friction), reports contacts, and carries the ego from
[autoware_isaac_bridge](https://github.com/NEWSLabNTU/autoware_isaac_bridge) (aib) in the same
process.

It is Python, built on the protocol's Python shell (`swb_world_server`, from
[swb_world_protocol](https://github.com/NEWSLabNTU/swb_world_protocol)), and runs in Isaac's
own Python (3.12) and ROS 2 libraries (`scripts/isaac_env.sh`). Until scenario-world-bridge's
roadmap 051 it was `src/isaac_world` in that repository.

## Build and run

Prerequisites: Isaac Sim 6.1 installed with pip into a venv (`scripts/install_isaac.sh`;
`ISAAC_ROOT`, default `/mnt/disk1/isaac-sim`) and ROS 2 Humble.

```bash
git submodule update --init --recursive   # deps/swb_world_protocol, deps/autoware_isaac_bridge
just msgs                                 # once per Isaac install: Autoware's messages for Isaac's Python
just build                                # the ROS package: the launch node wrapper, launch file, config
just test                                 # the pure modules (pxr and warp tests skip outside Isaac's Python)
source install/setup.bash
play_launch launch swb_isaac_world swb_isaac_world.launch.xml ego:=none render:=headless
```

In swb's workspace, `just isaac-world` runs this launch (web UI `SWB_WEB_PORT`+5, logs in
`$SWB_LOG_ROOT/isaac/`) and `just world-health sim:=isaac` probes it.

`launch/swb_isaac_world.launch.xml` starts the server supervised (respawned 10 s after it
dies) as the node `swb_isaac_world`. Its arguments:

| argument | default | |
|---|---|---|
| `ego` | `default` | `default`: aib's `deps/autoware_isaac_bridge/src/aib/config/ego.yaml` (the Tesla Model 3, a PhysX vehicle with Autoware's sensors and vehicle interface); `none`: a kinematic box; or a path to an aib ego config |
| `world` | `none` | worlds to preload, comma separated: world files (`<map>/isaac/world.yaml`), map directories or USD files; the first stays loaded |
| `follow` | `false` | `true`: with a window, no UI and a viewport chasing the ego |
| `render` | `auto` | `auto` (a window on `DISPLAY` when set, unless `ISAAC_HEADLESS=1`), `window` or `headless` |
| `state_dir` | empty | where the server writes for a map, and a project's own map files: `<state_dir>/<map>/`; empty: the map directory |
| `wheels_topic` | `true` | publish `/swb/wheels` on the ego's domain |

Host settings come from the environment: `ISAAC_PORT` (the REP port on 127.0.0.1, default
5590), `EGO_ROS_DOMAIN_ID` (the ego's ROS domain, default 1, over the ego config's),
`DISPLAY`, `ISAAC_HEADLESS`, `ISAAC_ROOT`, `CYCLONEDDS_URI` (the ego stack's DDS),
`SWB_CARLA_USD` (the CARLA USD set's root; `CARLA_USD` is read when it is not set),
`SWB_ISAAC_DIR` (the source tree the wrapper runs, default the one it was built from).

### Names, not paths

A world file's `usd:` may name its stage through the environment, so a map directory moves
between hosts unchanged:

```yaml
# <map>/isaac/world.yaml
usd: ${SWB_CARLA_USD}/Town01/Town01_physics.usda
```

`$VAR`, `${VAR}` and `~` are expanded; a variable that is not set is refused by name. A course
world (Haruna, Tsukuba) finds its props through a `carla-usd` link beside it; when the link is
absent (a downloaded dataset holds none), the server loads the world from a stand-in directory
that adds the link to `SWB_CARLA_USD`, under the state directory (or
`~/.cache/swb_isaac_world/worlds/` without one). Nothing is written into the map directory.

### The state directory

With `state_dir:=<dir>`, everything the server writes for a map goes to `<dir>/<map>/`, `<map>`
being the map directory's name:

| Path | What |
|---|---|
| `<dir>/<map>/ruts/runs/<time>.npz` | the soil's ruts at the end of each scenario |
| `<dir>/<map>/ruts/<name>.npz` | a rut set a scenario names (`swb.ruts`); looked for here, then in `<map dir>/ruts/` |
| `<dir>/<map>/surfaces.yaml`, `road_detail.yaml` | a project's own surface map and road detail, used over the map directory's |
| `<dir>/<map>/isaac/` | a course world's stand-in directory (above) |

Without it the server writes ruts into `<map dir>/ruts/` and reads only the map directory.

### `/swb/wheels`

Each ego wheel's ground and tire state, as `diagnostic_msgs/DiagnosticArray` on the ego's
domain, sampled after a physics step at most every 0.05 s of simulation time: one status per
wheel, named `<ego>/<front_left|front_right|rear_left|rear_right>`, its message the surface
class, and its values `class`, `zone`, `friction`, `slip_ratio`, `slip_angle` (rad),
`sinkage` (m) and `normal_load` (N). The layout is in `swb_isaac_world/wheels.py`.

```bash
ros2 topic echo /swb/wheels --field status
```

## Commands

| Command | What |
|---|---|
| `ros2 run swb_isaac_world make_world <map dir> [--usd <town usd>]` | a map's Isaac world: the town's physics layer and `<map dir>/isaac/world.yaml`, the town found in `SWB_CARLA_USD` by the map's name (`--town`, `--export` from CARLA's editor, `--physics`, `--dry-run`); Isaac's Python |
| `ros2 run swb_isaac_world surfaces check <map dir>` | check a map's `surfaces.yaml` (`plot` draws it) |
| `ros2 run swb_isaac_world road_detail check <map dir>` | check a map's `road_detail.yaml` (`preview` draws a corridor, `keys` lists every key) |

Examples of the rough road (`road_detail.yaml` beside a `surfaces.yaml`) are in
`swb_isaac_world/examples/road/`, installed to `share/swb_isaac_world/examples/road/`:
`town01` (a rough gravel road with rocks on Town01's straight at y -131), `town01_flat` (its
baseline) and `proving_ground`.

The server answers nothing until every preloaded world is loaded and rendered once (a town's
first render compiles its materials: minutes the first time on a host). Ready means it
answers `Hello`.

## On the protocol

- **Capabilities**: `SIGNALS`, `TRACKED_MOTION`, `CONTACTS`, `ROAD_CONDITION`,
  `ENVIRONMENT`. Not `WEATHER` (`SetWeather`, `GetWeather`, which the bridge's
  `set_weather` and `get_weather` services use): on Isaac a scenario's weather and time of
  day reach the sky through `ENVIRONMENT`.
- **Time and episodes**: `LoadWorldReply` and `StepReply` carry `episode_id` (a new one when
  the world changes) and `sim_time_ns = nanos(sim_time) + BASE_NS`. This server's simulation
  time runs on over its whole lifetime, across world loads, so a world load does not end an
  Isaac ego stack; a restarted server starts time again, and the stack's `clock_guard`
  (autoware_sim_profile) ends it. The ego stamps `/clock` the same way (`aib.sim_time`).
- **The ego's body**: the spawn reply's body (box, wheels, `spec_hash`) comes from aib's rig
  (`catalogue.ego_body`), the file aib generates from the vehicle spec. The bridge compares
  it with SSv2's vehicle catalog and with the vehicle agent's `spec_hash` before a scenario
  starts.

## Layout

| Path | What |
|---|---|
| `swb_isaac_world/` | The Python package: the world server (`world_server`, `world`), motion, catalogue and vehicle assets, walkers and riders, signals, sky, surfaces and soil, the road's small-scale detail and the tire's contact with it (`road_detail`, `road_contact`, `road_layer`: a `road_detail.yaml` beside a map's `surfaces.yaml`, roadmap 066) and its rocks (`rocks`, `rock_layer`), USD and point-cloud worlds |
| `launch/`, `scripts/swb_isaac_world_server.in` | The launch file and the launch node wrapper (drops the system ROS environment, sources Isaac's) |
| `scripts/` | Isaac install and environment, Autoware's messages for Isaac (`build_msgs.sh`), the `ros2 run` entries' wrapper (`swb_isaac_world_entry.in`), CARLA vehicle, walker and prop export to USD (`isaac_vehicles.sh`; `isaac_town.sh` calls `make_world`), step responses against CARLA (`step_response/`) |
| `swb_isaac_world/examples/` | Files to copy into a map or project: the rough road's |
| `config/surface_classes.yaml` | Road and off-road surface classes |
| `test/` | Unit tests of the pure modules |
| `spikes/` | Experiments behind the design decisions |
| `deps/` | `swb_world_protocol` and `autoware_isaac_bridge`, submodules (with a `COLCON_IGNORE`, so a colcon run over a superproject never sees them twice) |

## License

Apache License 2.0 ([LICENSE](LICENSE)).
