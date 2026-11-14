#!/usr/bin/env bash
# A CARLA town as an Isaac world (roadmap 043), now `make_world`'s (roadmap 076 A9):
#
#   ros2 run swb_isaac_world make_world <map dir> [--usd <town usd>] [--town <Town>] [--export]
#       [--physics]
#
# This script stays for the old arguments and calls it (swb_isaac_world.make_world):
#
#   isaac_town.sh <Town> [map_path:=<dir>] [export:=true] [physics:=true]
#
#   <Town>          CARLA's level name: Town01, Town02, Town03, Town05, Town10HD, ...
#   map_path:=<dir> the Autoware map the world file goes with (default $CARLA_MAPS/<Town>, or
#                   <Town> without "HD": Town10)
#   export:=true    export again even if <Town>.usd exists (it is exported when missing)
#   physics:=true   write <Town>_physics.usda again even if it exists (an export does too)
#
# Host paths from the environment: CARLA_SRC (the editor build with SimReady, for an export),
# SWB_CARLA_USD or CARLA_USD (the CARLA USD set, default /mnt/disk1/carla-usd), ISAAC_ROOT.
set -euo pipefail

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[ $# -ge 1 ] || usage
town=$1; shift
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)    # this repository
map_path=""; force=false; physics=false
for a in "$@"; do
    case "$a" in
        map_path:=*) map_path=${a#map_path:=} ;;
        export:=*) force=${a#export:=} ;;
        physics:=*) physics=${a#physics:=} ;;
        *) echo "isaac_town: unknown argument '$a' (map_path:=, export:=, physics:=)" >&2; exit 2 ;;
    esac
done
export CARLA_SRC=${CARLA_SRC:-/mnt/disk1/carla-src}
export SWB_CARLA_USD=${SWB_CARLA_USD:-${CARLA_USD:-/mnt/disk1/carla-usd}}
if [ -z "$map_path" ]; then
    for m in "$town" "${town%HD}"; do
        if [ -n "${CARLA_MAPS:-}" ] && [ -d "$CARLA_MAPS/$m" ]; then
            map_path="$CARLA_MAPS/$m"; break
        fi
    done
    [ -n "$map_path" ] || { echo "isaac_town: no Autoware map for $town under \$CARLA_MAPS (map_path:=)" >&2; exit 1; }
fi
args=("$map_path" --town "$town")
if [ "$force" = true ] || [ ! -f "$SWB_CARLA_USD/$town/$town.usd" ]; then
    args+=(--export)
fi
[ "$physics" = true ] && args+=(--physics)

export ISAAC_ROOT=${ISAAC_ROOT:-/mnt/disk1/isaac-sim}
set +u
source "$here/scripts/isaac_env.sh"
set -u
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
exec python -m swb_isaac_world.make_world "${args[@]}"
