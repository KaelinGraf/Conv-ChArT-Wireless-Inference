#!/usr/bin/env bash
# Container entrypoint (both images): source ROS 2 + the overlay workspace, then exec.
set -e
source /etc/p4p-env.sh
exec "$@"
