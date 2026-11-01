"""Trajectory tracking: pure pursuit (lateral) + PID (longitudinal). Simulator-independent."""

from driveaware.control.controller import ControllerConfig, WaypointController

__all__ = ["ControllerConfig", "WaypointController"]
