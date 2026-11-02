"""Failure-prediction metrics: discrimination, calibration, and event-level warning quality."""

from __future__ import annotations

import numpy as np


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum statistic (ties get average ranks)."""
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(np.asarray(s), return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - 0.5 * (counts - 1)  # 1-based average rank per unique value
    ranks = avg_rank[inverse]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """PR-AUC as average precision (step-wise, no interpolation)."""
    y = np.asarray(y, dtype=bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(s), kind="mergesort")
    yt = y[order]
    tp = np.cumsum(yt)
    precision = tp / np.arange(1, len(yt) + 1)
    return float(precision[yt].sum() / yt.sum())


def roc_curve(y: np.ndarray, s: np.ndarray, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    thr = np.quantile(s, np.linspace(0, 1, points))[::-1]
    y = np.asarray(y, dtype=bool)
    tpr = np.array([(s[y] >= t).mean() for t in thr])
    fpr = np.array([(s[~y] >= t).mean() for t in thr])
    return np.concatenate([[0], fpr, [1]]), np.concatenate([[0], tpr, [1]])


def pr_curve(y: np.ndarray, s: np.ndarray, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    thr = np.quantile(s, np.linspace(0, 0.999, points))[::-1]
    y = np.asarray(y, dtype=bool)
    rec, prec = [], []
    for t in thr:
        a = s >= t
        if a.sum() == 0:
            continue
        rec.append((a & y).sum() / max(y.sum(), 1))
        prec.append((a & y).sum() / a.sum())
    return np.array(rec), np.array(prec)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y, dtype=float)) ** 2))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 15) -> dict:
    """Equal-mass bins: mean predicted probability vs observed frequency, and ECE."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    chunks = np.array_split(order, bins)
    conf = np.array([p[c].mean() for c in chunks if len(c)])
    freq = np.array([y[c].mean() for c in chunks if len(c)])
    weight = np.array([len(c) for c in chunks if len(c)]) / len(p)
    return {
        "confidence": conf,
        "frequency": freq,
        "weight": weight,
        "ece": float(np.sum(weight * np.abs(conf - freq))),
    }


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, fpr: float) -> float:
    """Smallest threshold whose frame-level false-positive rate is <= ``fpr``."""
    neg = np.sort(np.asarray(p)[~np.asarray(y, dtype=bool)])
    return float(np.quantile(neg, 1.0 - fpr)) if len(neg) else 1.0


def alarm_onsets(p: np.ndarray, threshold: float, hold: int = 2) -> np.ndarray:
    """Indices where the probability has exceeded the threshold for ``hold`` consecutive steps
    (the alarm is raised at the end of the hold window)."""
    above = np.asarray(p) > threshold
    run = np.zeros(len(above), dtype=int)
    for i, a in enumerate(above):
        run[i] = run[i - 1] + 1 if a and i > 0 else int(a)
    return np.nonzero(run == hold)[0]


def event_level(
    episodes: list[dict],
    probs: list[np.ndarray],
    threshold: float,
    horizon: float = 2.0,
    hold: int = 2,
    dt: float = 0.1,
) -> dict:
    """Detection rate and lead time of failure events; false alarms per km.

    An alarm is active once the probability has exceeded the threshold for ``hold`` consecutive
    steps. An event at t_e is detected if an alarm is active at some step in
    [t_e - horizon, t_e) (the same horizon the predictor is trained for). Its lead time is
    t_e minus the time at which that alarm segment was raised, so persistent alarms earn long
    lead times. An alarm onset is false if no failure follows within ``horizon``.
    """
    from driveaware.introspection.data import failure_times

    leads, detected, n_events, false_alarms, km = [], 0, 0, 0, 0.0
    for ep, p in zip(episodes, probs, strict=True):
        t = ep["timestamp"]
        km += float(np.sum(ep["speed"]) * dt / 1000.0)
        above = np.asarray(p) > threshold
        run = np.zeros(len(above), dtype=int)
        for i, a in enumerate(above):
            run[i] = (run[i - 1] + 1 if i > 0 else 1) if a else 0
        active = run >= hold
        ft = np.sort(failure_times(ep))
        for te in ft:
            n_events += 1
            in_win = np.nonzero((t >= te - horizon - 1e-9) & (t < te) & active)[0]
            if len(in_win):
                detected += 1
                last = in_win[-1]
                raised = last - (run[last] - hold)  # step at which this segment's alarm fired
                leads.append(float(te - t[raised]))
        for i in alarm_onsets(p, threshold, hold):
            if not np.any((ft > t[i]) & (ft <= t[i] + horizon)):
                false_alarms += 1
    leads_a = np.array(leads)

    def pct(q: float) -> float:
        return float(np.percentile(leads_a, q)) if len(leads_a) else float("nan")

    return {
        "threshold": threshold,
        "events": n_events,
        "detection_rate": detected / max(n_events, 1),
        "lead_time_median_s": pct(50),
        "lead_time_p25_s": pct(25),
        "lead_time_p75_s": pct(75),
        "lead_times_s": leads_a.tolist(),
        "false_alarms": false_alarms,
        "false_alarms_per_km": false_alarms / max(km, 1e-9),
        "km": km,
    }


def episode_bootstrap(
    per_episode: list[tuple[np.ndarray, np.ndarray]], fn, n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% CI of fn(y, s) resampling whole episodes (frames within an episode are correlated)."""
    rng = np.random.default_rng(seed)
    vals = []
    k = len(per_episode)
    for _ in range(n):
        pick = rng.integers(0, k, k)
        y = np.concatenate([per_episode[i][0] for i in pick])
        s = np.concatenate([per_episode[i][1] for i in pick])
        if y.any() and (~y).any():
            vals.append(fn(y, s))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
