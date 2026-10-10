#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
python3 -m unittest discover -s "$DEMO_ROOT/tests" -v
python3 -m unittest discover -s "$DEMO_ROOT/ros2_ws/src/storm_pick_place_demo/test" -v
