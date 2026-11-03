"""Closed-loop rollouts of a fixed policy, recorded for failure prediction.

Per decision step t (the policy has just observed frame t and acts), we store:
- non-privileged introspection inputs: GRU hidden state, newest encoder feature, predicted
  waypoints, speed, command (all functions of observations <= t);
- privileged quantities for labels, the oracle baseline and analysis only: events, route
  lateral offset, junction flag, gap / closing speed to the constraining obstacle on the route.

Rollouts reproduce ``evaluation.closed_loop.run_episode`` exactly (same agent, same simulator),
so the introspection test set is literally the driving test set of model B.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from driveaware.core.run import atomic_savez
from driveaware.core.types import EventType
from driveaware.evaluation.closed_loop import AgentSpec, _make_agent
from driveaware.sim.lightweight.agents import IDMParams, find_leader
from driveaware.sim.lightweight.env import TERMINAL_EVENTS, LightweightDrivingSim, SimConfig
from driveaware.sim.lightweight.scenario import ScenarioSpec

_W: dict[str, Any] = {}
_ORACLE_IDM = IDMParams(lookahead=40.0)


def _privileged_context(res, vehicle) -> tuple[float, float, float, bool]:
    """(gap m, closing speed m/s, |lateral| m, in_junction) from privileged state."""
    p = res.privileged
    route, road_map = p.extra["route"], p.extra["map"]
    s_centre = p.route_progress + vehicle.center_offset
    gap, dv = find_leader(
        route,
        s_centre,
        p.ego_speed,
        vehicle.length,
        vehicle.width,
        list(p.agents),
        None,
        road_map,
        p.timestamp,
        _ORACLE_IDM,
    )
    lane = road_map.lanes[route.lane_ids[int(route.lane_index_at(p.route_progress))]]
    return (
        np.inf if gap is None else float(gap),
        float(dv),
        abs(p.route_lateral),
        lane.kind == "connector",
    )


def record_episode(scenario: ScenarioSpec, agent_spec: AgentSpec) -> dict[str, np.ndarray]:
    key = repr(agent_spec)
    if _W.get("key") != key:
        _W.update(key=key, agent=_make_agent(agent_spec), sim=LightweightDrivingSim(SimConfig()))
    agent, sim = _W["agent"], _W["sim"]
    vehicle = sim.cfg.vehicle
    res = sim.reset(scenario)
    agent.reset()
    rec: dict[str, list] = {
        k: []
        for k in (
            "timestamp",
            "speed",
            "command",
            "waypoints",
            "hidden",
            "feature",
            "action",
            "gap",
            "closing_speed",
            "lateral",
            "in_junction",
            "route_progress",
        )
    }
    events: list[tuple[int, int, float]] = []
    step = 0
    while not res.done:
        obs = res.observation
        action = agent(obs)
        rec["timestamp"].append(obs.timestamp)
        rec["speed"].append(obs.speed)
        rec["command"].append(int(obs.command))
        rec["waypoints"].append(agent.last_prediction.waypoints)
        rec["hidden"].append(agent.last_hidden)
        rec["feature"].append(agent.last_feature)
        rec["action"].append((action.steer, action.throttle, action.brake))
        gap, dv, lat, junction = _privileged_context(res, vehicle)
        rec["gap"].append(gap)
        rec["closing_speed"].append(dv)
        rec["lateral"].append(lat)
        rec["in_junction"].append(junction)
        rec["route_progress"].append(res.privileged.route_progress)
        res = sim.step(action)
        events += [(step, int(e.type), e.timestamp) for e in res.events]
        step += 1
    p = res.privileged
    out = {
        "timestamp": np.asarray(rec["timestamp"], np.float64),
        "speed": np.asarray(rec["speed"], np.float32),
        "command": np.asarray(rec["command"], np.int8),
        "waypoints": np.asarray(rec["waypoints"], np.float32),
        "hidden": np.asarray(rec["hidden"], np.float16),
        "feature": np.asarray(rec["feature"], np.float16),
        "action": np.asarray(rec["action"], np.float32),
        "gap": np.asarray(rec["gap"], np.float32),
        "closing_speed": np.asarray(rec["closing_speed"], np.float32),
        "lateral": np.asarray(rec["lateral"], np.float32),
        "in_junction": np.asarray(rec["in_junction"], bool),
        "route_progress": np.asarray(rec["route_progress"], np.float32),
        "event_step": np.asarray([e[0] for e in events], np.int32),
        "event_type": np.asarray([e[1] for e in events], np.int8),
        "event_time": np.asarray([e[2] for e in events], np.float64),
        "scenario": np.asarray(scenario.key, np.int64),
        "outcome": np.asarray(
            next((e.type for e in res.events if e.type in TERMINAL_EVENTS), EventType.TIMEOUT),
            np.int8,
        ),
        "completion": np.asarray(min(1.0, p.route_progress / max(p.route_length - 3.0, 1e-6))),
    }
    return out


def _job(args: tuple[ScenarioSpec, AgentSpec, str]) -> str:
    scenario, spec, path = args
    if not Path(path).exists():
        atomic_savez(path, **record_episode(scenario, spec))
    return path


def record_rollouts(
    scenarios: list[ScenarioSpec], agent: AgentSpec, out_dir: str | Path, workers: int = 6
) -> list[Path]:
    """Record (or reuse) one .npz per scenario, in scenario order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [
        (s, agent, str(out_dir / f"ep_{k:04d}_{s.map_seed}_{s.route_seed}.npz"))
        for k, s in enumerate(scenarios)
    ]
    if workers <= 1:
        return [Path(_job(j)) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [Path(p) for p in pool.map(_job, jobs, chunksize=1)]


def load_rollouts(paths: list[Path]) -> list[dict[str, np.ndarray]]:
    out = []
    for p in paths:
        with np.load(p) as z:
            out.append({k: z[k] for k in z.files})
    return out
