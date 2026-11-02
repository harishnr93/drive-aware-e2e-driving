"""Closed-loop evaluation on fixed scenario sets (paired across agents).

Agents are specified by an :class:`AgentSpec` (picklable) so that episodes can run in parallel
worker processes, each with its own simulator and a CPU copy of the policy.
"""

from __future__ import annotations

import collections
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from driveaware.core.seeding import derive_seed
from driveaware.core.types import EventType
from driveaware.sim.lightweight.agents import AgentKind
from driveaware.sim.lightweight.env import TERMINAL_EVENTS, LightweightDrivingSim, SimConfig
from driveaware.sim.lightweight.scenario import ScenarioSpec

INFRACTIONS = ("collision", "red_light", "off_road", "lane_departure", "route_deviation", "stall")
# Secondary driving-score penalties (CARLA-leaderboard style multiplicative factors).
PENALTY = {
    "collision_pedestrian": 0.5,
    "collision_vehicle": 0.6,
    "collision_static": 0.65,
    "red_light": 0.7,
}


@dataclass(frozen=True)
class AgentSpec:
    kind: str  # "expert" | "policy"
    checkpoint: str | None = None
    history: int = 1
    history_stride: int = 1


def scenario_set(
    name: str, map_seeds: list[int], routes_per_map: int, base: ScenarioSpec, seed: int = 0
) -> list[ScenarioSpec]:
    """Deterministic closed-loop scenarios; seeds differ from the dataset's by construction."""
    out = []
    for m in map_seeds:
        for k in range(routes_per_map):
            r = derive_seed(seed, "closed_loop", name, m, k, "route") % 2**31
            t = derive_seed(seed, "closed_loop", name, m, k, "traffic") % 2**31
            out.append(replace(base, map_seed=m, route_seed=r, traffic_seed=t))
    return out


def _make_agent(spec: AgentSpec):
    if spec.kind == "expert":
        from driveaware.expert.expert import RuleBasedExpert

        return RuleBasedExpert()
    import torch

    from driveaware.agents.policy_agent import AgentConfig, WaypointPolicyAgent
    from driveaware.training.bc import load_policy

    torch.set_num_threads(1)
    model, _ = load_policy(spec.checkpoint, "cpu")
    return WaypointPolicyAgent(
        model, "cpu", AgentConfig(history=spec.history, history_stride=spec.history_stride)
    )


_WORKER: dict[str, Any] = {}


def run_episode(scenario: ScenarioSpec, agent_spec: AgentSpec, record: bool = False) -> dict:
    """Run one episode; returns a metrics row (and frames if ``record``)."""
    key = repr(agent_spec)
    if _WORKER.get("key") != key:
        _WORKER.update(
            key=key,
            agent=_make_agent(agent_spec),
            sim=LightweightDrivingSim(SimConfig(render=agent_spec.kind != "expert" or record)),
        )
    agent, sim = _WORKER["agent"], _WORKER["sim"]
    res = sim.reset(scenario)
    agent.reset()
    counts: collections.Counter[str] = collections.Counter()
    dist, latencies, speeds, frames = 0.0, [], [], []
    while not res.done:
        prev = res.privileged.ego_pose
        action = agent(res) if agent_spec.kind == "expert" else agent(res.observation)
        if agent_spec.kind != "expert":
            latencies.append(agent.last_latency_ms)
        if record:
            pred = getattr(agent, "last_prediction", None)
            frames.append((res, None if pred is None else pred.waypoints))
        res = sim.step(action)
        p = res.privileged
        dist += float(np.hypot(p.ego_pose.x - prev.x, p.ego_pose.y - prev.y))
        speeds.append(p.ego_speed)
        for e in res.events:
            name = e.type.name.lower()
            counts[name] += 1
            if e.type == EventType.COLLISION:
                kind = int(e.info["agent_kind"])
                sub = {
                    AgentKind.PEDESTRIAN: "pedestrian",
                    AgentKind.VEHICLE: "vehicle",
                    AgentKind.PARKED: "static",
                }[kind]
                counts[f"collision_{sub}"] += 1
    if record:
        frames.append((res, None))
    p = res.privileged
    route, road_map = p.extra["route"], p.extra["map"]
    end_lane = road_map.lanes[route.lane_ids[int(route.lane_index_at(p.route_progress))]]
    completion = float(min(1.0, p.route_progress / max(p.route_length - 3.0, 1e-6)))
    penalty = float(np.prod([PENALTY[k] ** counts.get(k, 0) for k in PENALTY]))
    row = {
        "map_seed": scenario.map_seed,
        "route_seed": scenario.route_seed,
        "traffic_seed": scenario.traffic_seed,
        "outcome": next((e.type.name for e in res.events if e.type in TERMINAL_EVENTS), "NONE"),
        "completion": completion,
        "driving_score": completion * penalty,
        "distance_km": dist / 1000.0,
        "duration_s": res.observation.timestamp,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "end_lane_kind": "junction" if end_lane.kind == "connector" else "road",
        "end_lane_turn": end_lane.turn.name,
        **{f"n_{k}": int(v) for k, v in counts.items()},
    }
    if latencies:
        row["latency_ms_p50"] = float(np.percentile(latencies, 50))
        row["latency_ms_p95"] = float(np.percentile(latencies, 95))
    if record:
        row["frames"] = frames
    return row


def _run(args: tuple[ScenarioSpec, AgentSpec]) -> dict:
    return run_episode(*args)


def evaluate(scenarios: list[ScenarioSpec], agent: AgentSpec, workers: int = 4) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [(s, agent) for s in scenarios]
    if workers <= 1:
        return [_run(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run, jobs, chunksize=1))


def bootstrap_ci(values: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict]) -> dict:
    km = sum(r["distance_km"] for r in rows)
    comp = np.array([r["completion"] for r in rows])
    coll = np.array([r.get("n_collision", 0) > 0 for r in rows], dtype=float)
    out = {
        "episodes": len(rows),
        "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
        "route_completion": float(comp.mean()),
        "route_completion_ci95": bootstrap_ci(comp),
        "success_rate": float(np.mean([r["outcome"] == "ROUTE_COMPLETE" for r in rows])),
        "collision_episode_rate": float(coll.mean()),
        "collision_episode_rate_ci95": bootstrap_ci(coll),
        "driving_score": float(np.mean([r["driving_score"] for r in rows])),
        "distance_km": km,
        # Rates are undefined (None) if the agent never moved, e.g. a policy that never starts.
        "per_km": {
            k: (sum(r.get(f"n_{k}", 0) for r in rows) / km if km > 0 else None) for k in INFRACTIONS
        },
        "collisions_by_kind": {
            k: sum(r.get(f"n_collision_{k}", 0) for r in rows)
            for k in ("vehicle", "pedestrian", "static")
        },
        "mean_speed": float(np.mean([r["mean_speed"] for r in rows])),
        "failure_location": dict(
            collections.Counter(
                f"{r['end_lane_kind']}/{r['end_lane_turn']}"
                for r in rows
                if r["outcome"] not in ("ROUTE_COMPLETE", "TIMEOUT")
            )
        ),
    }
    if "latency_ms_p50" in rows[0]:
        out["latency_ms_p50"] = float(np.median([r["latency_ms_p50"] for r in rows]))
        out["latency_ms_p95"] = float(np.median([r["latency_ms_p95"] for r in rows]))
    return out


def paired_difference(a: list[dict], b: list[dict], key: str) -> dict:
    """Mean and bootstrap CI of per-scenario differences (a - b); rows must align."""
    assert [(r["map_seed"], r["route_seed"]) for r in a] == [
        (r["map_seed"], r["route_seed"]) for r in b
    ], "paired comparison needs identical scenario order"
    if key == "collision":
        d = np.array(
            [
                (ra.get("n_collision", 0) > 0) - (rb.get("n_collision", 0) > 0)
                for ra, rb in zip(a, b, strict=True)
            ],
            dtype=float,
        )
    else:
        d = np.array([ra[key] - rb[key] for ra, rb in zip(a, b, strict=True)])
    return {"mean": float(d.mean()), "ci95": bootstrap_ci(d)}


def spec_dict(s: ScenarioSpec) -> dict:
    return asdict(s)
