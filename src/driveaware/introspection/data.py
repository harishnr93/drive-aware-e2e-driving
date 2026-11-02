"""Failure labels and feature sets for introspection.

Label: y_t = 1 iff a failure event occurs in (t, t + horizon]. Events at or before t never
count, and features only use quantities known at t, so there is no future information in the
inputs. Feature sets are kept strictly separate:

- ``learned``: inputs of the introspection head (GRU state, speed, command, predicted waypoints)
- ``heuristic``: simple non-privileged scores (speed, command context, novelty, inconsistency)
- ``oracle``: privileged quantities (time-to-collision, lateral offset, junction flag). Reported
  only as an upper reference; never a deployable predictor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from driveaware.core.types import EventType

FAILURE_TYPES = (
    EventType.COLLISION,
    EventType.OFF_ROAD,
    EventType.ROUTE_DEVIATION,
    EventType.RED_LIGHT,
    EventType.STALL,
)
FAILURE_TYPES_WITH_LANE = FAILURE_TYPES + (EventType.LANE_DEPARTURE,)


def failure_times(ep: dict, types=FAILURE_TYPES) -> np.ndarray:
    mask = np.isin(ep["event_type"], [int(t) for t in types])
    return ep["event_time"][mask]


def failure_labels(ep: dict, horizon: float = 2.0, types=FAILURE_TYPES) -> np.ndarray:
    t = ep["timestamp"][:, None]
    ft = failure_times(ep, types)[None, :]
    if ft.size == 0:
        return np.zeros(len(ep["timestamp"]), dtype=bool)
    return ((ft > t + 1e-9) & (ft <= t + horizon + 1e-9)).any(axis=1)


def time_to_next_failure(ep: dict, types=FAILURE_TYPES) -> np.ndarray:
    t = ep["timestamp"]
    ft = np.sort(failure_times(ep, types))
    idx = np.searchsorted(ft, t + 1e-9, side="left")
    out = np.full(len(t), np.inf)
    ok = idx < len(ft)
    out[ok] = ft[idx[ok]] - t[ok]
    return out


@dataclass
class NoveltyModel:
    """Mahalanobis distance of encoder features to the expert training distribution."""

    mean: np.ndarray
    precision: np.ndarray

    @classmethod
    def fit(cls, feats: np.ndarray, shrinkage: float = 1e-2) -> NoveltyModel:
        x = feats.astype(np.float64)
        mean = x.mean(axis=0)
        cov = np.cov(x - mean, rowvar=False)
        cov = (1 - shrinkage) * cov + shrinkage * np.trace(cov) / len(cov) * np.eye(len(cov))
        return cls(mean, np.linalg.inv(cov))

    def score(self, feats: np.ndarray) -> np.ndarray:
        d = feats.astype(np.float64) - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.precision, d))


def inconsistency(ep: dict) -> np.ndarray:
    """Change of the policy's own prediction between consecutive steps (non-privileged):
    mean absolute change of predicted lateral offsets + change of implied speed."""
    wp = ep["waypoints"].astype(np.float64)
    lat = np.abs(np.diff(wp[:, :, 1], axis=0)).mean(axis=1)
    v = np.linalg.norm(wp[:, 2] - wp[:, 0], axis=1) / 0.4
    dv = np.abs(np.diff(v))
    out = np.concatenate([[0.0], lat + 0.1 * dv])
    return out


def learned_features(ep: dict) -> np.ndarray:
    cmd = np.eye(4, dtype=np.float32)[ep["command"].astype(int)]
    return np.concatenate(
        [
            ep["hidden"].astype(np.float32),
            (ep["speed"] / 10.0)[:, None].astype(np.float32),
            cmd,
            (ep["waypoints"].reshape(len(ep["speed"]), -1) / 10.0).astype(np.float32),
        ],
        axis=1,
    )


def heuristic_features(ep: dict, novelty: NoveltyModel) -> dict[str, np.ndarray]:
    cmd = np.eye(4)[ep["command"].astype(int)]
    wp = ep["waypoints"].astype(np.float64)
    return {
        "speed": ep["speed"][:, None].astype(np.float64),
        "context": np.column_stack([ep["speed"], cmd[:, 1:]]),  # speed + junction command
        "novelty": novelty.score(ep["feature"])[:, None],
        "inconsistency": inconsistency(ep)[:, None],
        "predicted_turn": np.abs(wp[:, -1, 1])[:, None],
    }


def oracle_features(ep: dict) -> np.ndarray:
    gap = np.minimum(ep["gap"].astype(np.float64), 60.0)
    closing = ep["closing_speed"].astype(np.float64)
    ttc = np.where(closing > 0.1, gap / np.maximum(closing, 0.1), 60.0)
    return np.column_stack(
        [np.minimum(ttc, 60.0), gap, ep["lateral"], ep["in_junction"], ep["speed"]]
    )
