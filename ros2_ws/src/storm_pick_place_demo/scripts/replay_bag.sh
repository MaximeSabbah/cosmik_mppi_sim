#!/usr/bin/env bash
# Replays a recorded pick_place/tower run in RViz exactly as it looked live.
# Works because these bags record /tf, /tf_static, /robot_description(+variants) and all the
# marker topics directly -- no robot_state_publisher or STORM needs to run, ros2 bag play is
# enough. --clock + rviz use_sim_time keeps RViz's notion of "now" in sync with the bag instead
# of wall-clock time.
#
# Usage: replay_bag.sh <bag_name_or_path> [extra ros2 bag play args...]
#   bag_name_or_path  Either a bag directory (absolute/relative path), or just its folder name
#                      (e.g. rosbag2_2026_08_14-13_32_26) to look up under BAG_ROOT.
#   extra args         Forwarded verbatim to `ros2 bag play`, e.g. -r 2.0 (rate), -l (loop),
#                      -p (start paused), --start-offset 30.
set -euo pipefail

BAG_ROOT="${BAG_ROOT:-/workspace/exp}"
RVIZ_CFG="$(dirname "${BASH_SOURCE[0]}")/../rviz/pick_place.rviz"

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <bag_name_or_path> [extra ros2 bag play args...]" >&2
    exit 1
fi
bag_arg="$1"
shift

if [[ -d "$bag_arg" ]]; then
    bag_path="$bag_arg"
else
    bag_path="$(find "$BAG_ROOT" -maxdepth 3 -type d -name "$bag_arg" -print -quit)"
    if [[ -z "$bag_path" ]]; then
        echo "No bag named '$bag_arg' found under $BAG_ROOT (and it's not a path itself)." >&2
        exit 1
    fi
fi
echo "Replaying: $bag_path"

rviz2 -d "$RVIZ_CFG" --ros-args -p use_sim_time:=true &
rviz_pid=$!
trap 'kill "$rviz_pid" 2>/dev/null || true' EXIT

sleep 2  # let RViz finish coming up before the bag starts pushing messages
ros2 bag play "$bag_path" --clock "$@"
