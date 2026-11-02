"""Expert episode collection with optional DART-style noise injection.

In noisy episodes, steering perturbation bursts are added to the expert's action. The expert's
own (unperturbed) action and the applied action are both recorded, together with a per-frame
``perturbed`` flag, so that labels can exclude any window containing injected noise
(see ``labels.label_validity``). Frames that start in a perturbed state but whose future is
clean are kept: they show recovery behaviour.

Episodes are written as compressed ``.npz`` (arrays) plus ``.json`` (metadata).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from driveaware.core.seeding import derive_seed, make_rng
from driveaware.core.types import ControlCommand, EventType
from driveaware.expert.expert import RuleBasedExpert
from driveaware.sim.lightweight.env import LightweightDrivingSim, SimConfig
from driveaware.sim.lightweight.scenario import ScenarioSpec

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NoiseConfig:
    episode_prob: float = 0.5  # fraction of episodes with noise injection
    mean_interval: float = 8.0  # s between burst onsets (exponential)
    duration: tuple[float, float] = (0.5, 1.2)  # s
    steer_offset: tuple[float, float] = (0.15, 0.35)  # added to normalised steer, random sign


@dataclass(frozen=True)
class EpisodeRequest:
    split: str
    episode_id: int
    spec: ScenarioSpec
    noise: bool
    noise_seed: int


@dataclass
class EpisodeData:
    arrays: dict[str, np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)


class _NoiseSchedule:
    def __init__(self, cfg: NoiseConfig, seed: int, control_period: float) -> None:
        self.cfg = cfg
        self.rng = make_rng(seed)
        self.dt = control_period
        self.next_onset = self.rng.exponential(cfg.mean_interval)
        self.burst_end = -1.0
        self.offset = 0.0

    def offset_at(self, t: float) -> float:
        if t < self.burst_end:
            return self.offset
        if t >= self.next_onset:
            dur = self.rng.uniform(*self.cfg.duration)
            self.burst_end = t + dur
            self.offset = float(self.rng.uniform(*self.cfg.steer_offset) * self.rng.choice([-1, 1]))
            self.next_onset = self.burst_end + self.rng.exponential(self.cfg.mean_interval)
            return self.offset
        return 0.0


def collect_episode(
    req: EpisodeRequest,
    noise_cfg: NoiseConfig,
    sim: LightweightDrivingSim | None = None,
) -> EpisodeData:
    sim = sim or LightweightDrivingSim(SimConfig())
    res = sim.reset(req.spec)
    expert = RuleBasedExpert()
    schedule = (
        _NoiseSchedule(noise_cfg, req.noise_seed, sim.cfg.timing.control_period)
        if req.noise
        else None
    )
    images, timestamps, speed, command, poses, steer_angle = [], [], [], [], [], []
    expert_actions, applied_actions, perturbed = [], [], []
    route_progress, route_lateral, target_speed = [], [], []
    events: list[tuple[int, int, float]] = []

    def record(r) -> None:
        images.append(r.observation.image)
        timestamps.append(r.observation.timestamp)
        speed.append(r.observation.speed)
        command.append(int(r.observation.command))
        p = r.privileged
        poses.append((p.ego_pose.x, p.ego_pose.y, p.ego_pose.yaw))
        steer_angle.append(p.ego_steer)
        route_progress.append(p.route_progress)
        route_lateral.append(p.route_lateral)

    record(res)
    while not res.done:
        action = expert(res)
        target_speed.append(expert.last_decision.target_speed)
        offset = schedule.offset_at(res.observation.timestamp) if schedule else 0.0
        applied = ControlCommand.clipped(action.steer + offset, action.throttle, action.brake)
        expert_actions.append((action.steer, action.throttle, action.brake))
        applied_actions.append((applied.steer, applied.throttle, applied.brake))
        perturbed.append(offset != 0.0)
        res = sim.step(applied)
        record(res)
        frame = len(images) - 1
        events += [(frame, int(e.type), e.timestamp) for e in res.events]

    n = len(images)
    nan3 = (np.nan, np.nan, np.nan)
    arrays = {
        "images": np.stack(images).astype(np.uint8),
        "timestamp": np.asarray(timestamps, dtype=np.float64),
        "speed": np.asarray(speed, dtype=np.float32),
        "command": np.asarray(command, dtype=np.int8),
        "ego_pose": np.asarray(poses, dtype=np.float64),
        "ego_steer": np.asarray(steer_angle, dtype=np.float32),
        "route_progress": np.asarray(route_progress, dtype=np.float32),
        "route_lateral": np.asarray(route_lateral, dtype=np.float32),
        # Per-frame actions; the final frame has none (NaN / False).
        "expert_action": np.asarray(expert_actions + [nan3], dtype=np.float32),
        "applied_action": np.asarray(applied_actions + [nan3], dtype=np.float32),
        "expert_target_speed": np.asarray(target_speed + [np.nan], dtype=np.float32),
        "perturbed": np.asarray(perturbed + [False], dtype=bool),
        "event_frame": np.asarray([e[0] for e in events], dtype=np.int32),
        "event_type": np.asarray([e[1] for e in events], dtype=np.int8),
        "event_time": np.asarray([e[2] for e in events], dtype=np.float64),
    }
    types = {EventType(t).name for t in arrays["event_type"]}
    failure_types = types & {"COLLISION", "OFF_ROAD", "ROUTE_DEVIATION", "STALL", "RED_LIGHT"}
    meta = {
        "schema_version": SCHEMA_VERSION,
        "split": req.split,
        "episode_id": req.episode_id,
        "spec": asdict(req.spec),
        "noise": req.noise,
        "noise_seed": req.noise_seed,
        "num_frames": n,
        "outcome": EventType(arrays["event_type"][-1]).name if len(events) else "NONE",
        "events": sorted(types),
        "expert_failure": bool(failure_types),
        "perturbed_fraction": float(np.mean(arrays["perturbed"][:-1])) if n > 1 else 0.0,
        "route_length": float(res.privileged.route_length),
        "final_progress": float(res.privileged.route_progress),
    }
    return EpisodeData(arrays, meta)


def save_episode(data: EpisodeData, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{data.meta['split']}_{data.meta['episode_id']:05d}"
    np.savez_compressed(directory / f"{stem}.npz", **data.arrays)
    (directory / f"{stem}.json").write_text(json.dumps(data.meta, indent=1))
    return directory / f"{stem}.npz"


def load_episode(path: str | Path) -> EpisodeData:
    path = Path(path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(path.with_suffix(".json").read_text())
    return EpisodeData(arrays, meta)


def plan_requests(
    split: str,
    map_seeds: list[int],
    routes_per_map: int,
    base_spec: ScenarioSpec,
    noise_cfg: NoiseConfig,
    dataset_seed: int,
    first_episode_id: int = 0,
) -> list[EpisodeRequest]:
    """Deterministic episode plan. Route/traffic/noise seeds are derived per (split, map, k)."""
    from dataclasses import replace

    out = []
    eid = first_episode_id
    for map_seed in map_seeds:
        for k in range(routes_per_map):
            route_seed = derive_seed(dataset_seed, split, map_seed, k, "route") % 2**31
            traffic_seed = derive_seed(dataset_seed, split, map_seed, k, "traffic") % 2**31
            noise_seed = derive_seed(dataset_seed, split, map_seed, k, "noise") % 2**31
            noisy = make_rng(noise_seed).random() < noise_cfg.episode_prob
            spec = replace(base_spec, map_seed=map_seed, route_seed=route_seed,
                           traffic_seed=traffic_seed)
            out.append(EpisodeRequest(split, eid, spec, noisy, noise_seed))
            eid += 1
    return out
