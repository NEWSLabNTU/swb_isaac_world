# swb_isaac_world

[Isaac Sim](https://developer.nvidia.com/isaac/sim) 6.1 behind the world protocol of
[scenario-world-bridge](https://github.com/NEWSLabNTU/scenario-world-bridge): the Isaac
**world server**. It loads a world (CARLA's towns exported to USD, or a map's point cloud as
voxels), spawns and steps the scenario's vehicles, people and props in PhysX, sets traffic
lights, sky, weather and road surfaces (off-road soil, per-wheel friction), reports contacts,
and carries the ego from
[autoware_isaac_bridge](https://github.com/NEWSLabNTU/autoware_isaac_bridge) (aib) in the same
process.

It is Python, built on the protocol's Python shell (`swb_world_server`), and runs in Isaac's
own Python and ROS 2 libraries (`scripts/isaac_env.sh`).

```bash
git submodule update --init --recursive   # deps/swb_world_protocol, deps/autoware_isaac_bridge
just msgs                                 # once per Isaac install
just build
play_launch launch swb_isaac_world swb_isaac_world.launch.xml ego:=none render:=headless
```

| Path | What |
|---|---|
| `swb_isaac_world/` | The Python package: the world server, motion, catalogue, signals, sky, surfaces and soil |
| `launch/`, `scripts/swb_isaac_world_server.in` | The launch file and the launch node wrapper |
| `scripts/` | Isaac environment, CARLA town and vehicle export to USD, step responses |
| `config/surface_classes.yaml` | Road and off-road surface classes |
| `spikes/` | Experiments behind the design decisions |
| `deps/` | swb_world_protocol and autoware_isaac_bridge, submodules |
