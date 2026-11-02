"""Closed-loop agent wrapping a learned waypoint policy.

The agent receives only :class:`Observation` objects (no privileged state). The predicted
waypoints are tracked with the same pure-pursuit + PID controller used by the expert, so
driving differences come from the trajectory source, not the controller.
Target speed is derived from the predicted waypoints: |w[j] - w[i]| / ((j - i) * waypoint_dt).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from driveaware.control import ControllerConfig, WaypointController
from driveaware.core.types import ControlCommand, Observation, TrajectoryPrediction


@dataclass(frozen=True)
class AgentConfig:
    speed_pair: tuple[int, int] = (0, 2)  # waypoint indices used for the target speed
    waypoint_dt: float = 0.2  # s
    max_target_speed: float = 12.0  # m/s
    history: int = 1  # frames fed to the model (1 = single-frame)
    history_stride: int = 1


class WaypointPolicyAgent:
    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        config: AgentConfig = AgentConfig(),
        controller: ControllerConfig = ControllerConfig(),
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device)
        self.cfg = config
        self.controller = WaypointController(controller)
        self._buffer: deque[Observation] = deque(
            maxlen=(config.history - 1) * config.history_stride + 1
        )
        self._features: deque[torch.Tensor] = deque(maxlen=self._buffer.maxlen)
        # Temporal models with a frozen encoder: encode only the newest frame each step.
        self._incremental = hasattr(model, "forward_features") and config.history > 1
        self.last_prediction: TrajectoryPrediction | None = None
        self.last_latency_ms = 0.0
        self.last_hidden: np.ndarray | None = None
        self.last_feature: np.ndarray | None = None

    def reset(self) -> None:
        self.controller.reset()
        self._buffer.clear()
        self._features.clear()
        self.last_prediction = None

    def target_speed(self, waypoints: np.ndarray) -> float:
        i, j = self.cfg.speed_pair
        v = np.linalg.norm(waypoints[j] - waypoints[i]) / ((j - i) * self.cfg.waypoint_dt)
        return float(np.clip(v, 0.0, self.cfg.max_target_speed))

    @torch.no_grad()
    def predict(self, obs: Observation) -> TrajectoryPrediction:
        self._buffer.append(obs)
        t0 = time.perf_counter()
        command = torch.tensor([int(obs.command)], dtype=torch.long, device=self.device)
        if self.cfg.history == 1:
            image = torch.from_numpy(obs.image).unsqueeze(0)
            speed = torch.tensor([obs.speed], dtype=torch.float32)
            if getattr(self.model, "uses_history", False):
                image, speed = image.unsqueeze(1), speed.unsqueeze(1)
            wp = self.model(image.to(self.device), speed.to(self.device), command)
        else:  # oldest first; pad with the earliest frame available (matches the dataset)
            frames = list(self._buffer)
            idx = [
                max(0, len(frames) - 1 - k * self.cfg.history_stride)
                for k in range(self.cfg.history - 1, -1, -1)
            ]
            speed = torch.tensor([[frames[i].speed for i in idx]], dtype=torch.float32)
            speed = speed.to(self.device)
            if self._incremental:
                newest = torch.from_numpy(obs.image).unsqueeze(0).to(self.device)
                self._features.append(self.model.encode_frames(newest)[0])
                feats = list(self._features)
                stacked = torch.stack([feats[i] for i in idx]).unsqueeze(0)
                wp = self.model.forward_features(stacked, speed, command)
            else:
                image = torch.from_numpy(np.stack([frames[i].image for i in idx])).unsqueeze(0)
                wp = self.model(image.to(self.device), speed, command)
        waypoints = wp[0].float().cpu().numpy()
        self.last_latency_ms = (time.perf_counter() - t0) * 1e3
        pred = TrajectoryPrediction(obs.timestamp, waypoints, self.target_speed(waypoints))
        self.last_prediction = pred
        # Introspection inputs at time t (all derived from observations up to t).
        hidden = getattr(self.model, "last_hidden", None)
        self.last_hidden = None if hidden is None else hidden[0].float().cpu().numpy()
        self.last_feature = self._features[-1].float().cpu().numpy() if self._features else None
        return pred

    def __call__(self, obs: Observation) -> ControlCommand:
        pred = self.predict(obs)
        return self.controller(pred.waypoints, pred.target_speed, obs.speed)
