#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ ! -f "$DEMO_ROOT/ros2_ws/install/local_setup.bash" ]]; then
  echo 'Build first: ./scripts/build.sh' >&2; exit 1
fi
exec ros2 launch storm_pick_place_demo pick_place_sim.launch.py \
  with_rviz:=false gz_headless:=true "$@"
