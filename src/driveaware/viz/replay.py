"""Episode logging, exact replay, and video export.

An episode is fully determined by its ScenarioSpec and action sequence, so a replay log is just
(spec, actions); :func:`replay` re-simulates it bit-identically. Videos show the policy camera
(upscaled) next to the privileged top-down debug view.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from driveaware.core.types import ControlCommand, EventType
from driveaware.sim.interface import StepResult
from driveaware.sim.lightweight.env import LightweightDrivingSim
from driveaware.sim.lightweight.renderer import TopDownRenderer
from driveaware.sim.lightweight.scenario import ScenarioSpec


@dataclass
class EpisodeLog:
    spec: ScenarioSpec
    actions: list[tuple[float, float, float]] = field(default_factory=list)
    events: list[tuple[int, str, float]] = field(default_factory=list)  # (step, type, time)

    def add(self, action: ControlCommand, result: StepResult) -> None:
        self.actions.append((action.steer, action.throttle, action.brake))
        step = len(self.actions)
        self.events += [(step, e.type.name, e.timestamp) for e in result.events]

    def save(self, path: str | Path) -> None:
        payload = {"spec": asdict(self.spec), "actions": self.actions, "events": self.events}
        Path(path).write_text(json.dumps(payload))

    @property
    def outcome(self) -> str:
        terminal = [
            name
            for _, name, _ in self.events
            if name not in (EventType.RED_LIGHT.name, EventType.LANE_DEPARTURE.name)
        ]
        return terminal[-1] if terminal else "RUNNING"


def replay(sim: LightweightDrivingSim, log: EpisodeLog) -> Iterator[StepResult]:
    """Re-simulate a logged episode, yielding the reset result and then every step."""
    res = sim.reset(log.spec)
    yield res
    for steer, throttle, brake in log.actions:
        res = sim.step(ControlCommand(steer, throttle, brake))
        yield res


def compose_frame(res: StepResult, topdown: TopDownRenderer, scale: int = 3) -> np.ndarray:
    """Camera (upscaled) | top-down view, with a status line. RGB uint8."""
    obs, p = res.observation, res.privileged
    cam = cv2.resize(
        obs.image,
        (obs.image.shape[1] * scale, obs.image.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    route = p.extra.get("route")
    top = topdown.render(
        p.ego_pose,
        p.ego_footprint,
        list(p.agents),
        route.path.points if route is not None else None,
    )
    top = cv2.resize(top, (cam.shape[0], cam.shape[0]), interpolation=cv2.INTER_AREA)
    frame = np.hstack([cam, top])
    status = (
        f"t={obs.timestamp:5.1f}s v={obs.speed:4.1f}m/s cmd={obs.command.name:<11}"
        f" s={p.route_progress:5.0f}/{p.route_length:.0f}m"
    )
    cv2.putText(
        frame, status, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )
    if res.events:
        names = " ".join(e.type.name for e in res.events)
        cv2.putText(
            frame, names, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 60), 1, cv2.LINE_AA
        )
    return frame


class VideoWriter:
    def __init__(self, path: str | Path, fps: float = 10.0) -> None:
        self.path = str(path)
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame_rgb: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame_rgb.shape[:2]
            self._writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"cannot open video writer for {self.path}")
        self._writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
