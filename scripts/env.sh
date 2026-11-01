#!/usr/bin/env bash
# Activate the project environment with ROS 2 isolation.
# Usage:  source scripts/env.sh
# ROS 2 Humble (sourced in ~/.bashrc) puts its packages on PYTHONPATH, which takes precedence
# over the venv and auto-loads ROS pytest plugins. Clear it for this shell only.
_DA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'ros2_humble\|/opt/ros' | paste -sd: -)"
    export LD_LIBRARY_PATH
fi
# shellcheck disable=SC1091
source "$_DA_ROOT/.venv/bin/activate"
export DRIVEAWARE_ROOT="$_DA_ROOT"
unset _DA_ROOT
