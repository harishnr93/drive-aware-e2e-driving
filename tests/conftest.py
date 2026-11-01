import os
import warnings


def pytest_configure(config):
    if "ros2_humble" in os.environ.get("PYTHONPATH", "") or "/opt/ros" in os.environ.get(
        "PYTHONPATH", ""
    ):
        warnings.warn(
            "ROS 2 paths are on PYTHONPATH; run `source scripts/env.sh` first.", stacklevel=1
        )
