#!/usr/bin/env bash
# Source this from a clean shell. Deliberately discard unrelated ROS overlays.
export DEMO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/humble/setup.bash
if [[ -f "$DEMO_ROOT/ros2_ws/install/local_setup.bash" ]]; then
  source "$DEMO_ROOT/ros2_ws/install/local_setup.bash"
fi
export STORM_ROOT="$DEMO_ROOT/storm"
export PYTHONPATH="$STORM_ROOT:$DEMO_ROOT/research:${PYTHONPATH:-}"
export ROS_DOMAIN_ID="${DEMO_DOMAIN_ID:-73}"
export ROS_LOCALHOST_ONLY="${DEMO_LOCALHOST_ONLY:-1}"
export IGN_PARTITION="cosmik_mppi_${ROS_DOMAIN_ID}"
export ROS_LOG_DIR="$DEMO_ROOT/.runtime/ros_logs"
export PYTHONUNBUFFERED=1
mkdir -p "$ROS_LOG_DIR"
