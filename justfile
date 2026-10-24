# swb_isaac_world: Isaac Sim behind the world protocol. Its dependencies, swb_world_protocol
# (the Python shell) and autoware_isaac_bridge (the ego), are submodules under deps/
# (git submodule update --init --recursive). ISAAC_ROOT: the Isaac Sim 6.1 pip install.

isaac_root := env_var_or_default('ISAAC_ROOT', '/mnt/disk1/isaac-sim')

default:
    @just --list

# Build the ROS package (the launch node wrapper, launch file, config)
build:
    #!/usr/bin/env bash
    set -e
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install --base-paths .

# Tests that need no Isaac (the pure modules); pxr and warp ones skip outside Isaac's Python
test:
    python3 -m unittest discover -s test

# The tests, then pyflakes
check: test
    python3 -m pyflakes swb_isaac_world test scripts

# Autoware's message bindings for Isaac's Python (once per Isaac install)
msgs:
    ISAAC_ROOT={{isaac_root}} scripts/build_msgs.sh
