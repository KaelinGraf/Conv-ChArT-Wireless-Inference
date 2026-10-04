# shellcheck shell=bash
# Sourced by the entrypoint and by every interactive bash (/etc/bash.bashrc).
# Keep it idempotent and free of `set -u`: ROS setup scripts touch unset variables.

# ROS 2 underlay
_p4p_ros="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
[ -f "$_p4p_ros" ] && source "$_p4p_ros"

# colcon overlay (src/ros/), once it has been built
[ -f /workspace/src/ros/install/setup.bash ] && source /workspace/src/ros/install/setup.bash

# CycloneDDS WiFi profile (docker/ros/cyclonedds.xml) unless the caller chose their own
if [ -z "${CYCLONEDDS_URI:-}" ] && [ -f /workspace/docker/ros/cyclonedds.xml ]; then
    export CYCLONEDDS_URI="file:///workspace/docker/ros/cyclonedds.xml"
fi

unset _p4p_ros
