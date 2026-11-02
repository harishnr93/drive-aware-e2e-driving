import numpy as np
import pytest

from driveaware.core.types import EventType
from driveaware.introspection.data import (
    NoveltyModel,
    failure_labels,
    inconsistency,
    time_to_next_failure,
)
from driveaware.introspection.metrics import (
    alarm_onsets,
    auroc,
    average_precision,
    brier,
    event_level,
    reliability,
    threshold_at_fpr,
)
from driveaware.introspection.models import LogisticModel, PlattCalibrator


def _episode(n=60, events=()):
    t = 0.1 * np.arange(n)
    return {
        "timestamp": t,
        "speed": np.full(n, 5.0, np.float32),
        "event_type": np.array([int(e[0]) for e in events], np.int8),
        "event_time": np.array([e[1] for e in events], np.float64),
        "waypoints": np.zeros((n, 10, 2), np.float32),
    }


def test_labels_strictly_future_window():
    ep = _episode(events=[(EventType.COLLISION, 3.0)])
    y = failure_labels(ep, horizon=2.0)
    t = ep["timestamp"]
    expected = (t + 1e-9 < 3.0) & (t + 2.0 + 1e-9 >= 3.0)
    np.testing.assert_array_equal(y, expected)
    assert not y[t >= 3.0 - 1e-9].any()  # the event step itself and later are not positive
    assert y[np.isclose(t, 1.0)].all() and not y[np.isclose(t, 0.9)].any()


def test_non_failure_events_ignored():
    ep = _episode(events=[(EventType.LANE_DEPARTURE, 2.0), (EventType.ROUTE_COMPLETE, 5.0)])
    assert not failure_labels(ep).any()
    assert np.isinf(time_to_next_failure(ep)).all()


def test_time_to_next_failure():
    ep = _episode(events=[(EventType.RED_LIGHT, 1.5), (EventType.COLLISION, 4.0)])
    ttf = time_to_next_failure(ep)
    assert ttf[0] == pytest.approx(1.5) and ttf[20] == pytest.approx(2.0)


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.2
    s = np.round(rng.normal(size=300) + y, 1)  # with ties
    pos, neg = s[y], s[~y]
    pairwise = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (
        len(pos) * len(neg)
    )
    assert auroc(y, s) == pytest.approx(pairwise)
    assert auroc(y, y.astype(float)) == 1.0


def test_average_precision_perfect_and_random():
    y = np.array([1, 1, 0, 0, 0], bool)
    assert average_precision(y, np.array([5, 4, 3, 2, 1.0])) == 1.0
    assert average_precision(y, np.array([1, 2, 3, 4, 5.0])) < 0.5


def test_calibration_metrics():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = rng.random(20000) < p  # perfectly calibrated
    assert reliability(y, p)["ece"] < 0.02
    assert reliability(y, np.clip(p * 0.5, 0, 1))["ece"] > 0.2
    assert brier(y, p) == pytest.approx(np.mean((p - y) ** 2))


def test_platt_recovers_shifted_logits():
    rng = np.random.default_rng(2)
    z = rng.normal(size=20000)
    y = rng.random(20000) < 1 / (1 + np.exp(-z))
    cal = PlattCalibrator.fit(3.0 * z + 2.0, y)  # distorted scores
    assert cal.a == pytest.approx(1 / 3, abs=0.03) and cal.c == pytest.approx(-2 / 3, abs=0.1)


def test_logistic_model_learns_signal():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5000, 2))
    y = rng.random(5000) < 1 / (1 + np.exp(-2 * x[:, 0]))
    m = LogisticModel.fit(x, y)
    assert auroc(y, m.logits(x)) > 0.8 and abs(m.w[1]) < 0.2 * abs(m.w[0])


def test_threshold_and_alarms():
    y = np.array([0] * 90 + [1] * 10, bool)
    p = np.linspace(0, 1, 100)
    thr = threshold_at_fpr(y, p, 0.1)
    assert ((p[~y] > thr).mean()) <= 0.1 + 1e-9
    assert alarm_onsets(np.array([0, 1, 1, 1, 0, 1, 1]), 0.5, hold=2).tolist() == [2, 6]


def test_event_level_detection_and_lead_time():
    ep = _episode(n=60, events=[(EventType.COLLISION, 5.0)])
    p = np.zeros(60)
    p[30:50] = 0.9  # alarm active from t=3.0 (hold 2 -> 3.1)
    r = event_level([ep], [p], threshold=0.5)
    assert r["detection_rate"] == 1.0 and r["lead_time_median_s"] == pytest.approx(1.9)
    assert r["false_alarms"] == 0
    p2 = np.zeros(60)
    p2[5:8] = 0.9  # alarm at t=0.6 that ends long before the event: false, not a detection
    r2 = event_level([ep], [p2], threshold=0.5)
    assert r2["detection_rate"] == 0.0 and r2["false_alarms"] == 1
    p3 = np.zeros(60)
    p3[5:50] = 0.9  # persistent alarm raised at t=0.6 and active up to the event
    r3 = event_level([ep], [p3], threshold=0.5)
    assert r3["detection_rate"] == 1.0 and r3["lead_time_median_s"] == pytest.approx(4.4)
    assert r3["false_alarms"] == 1  # its onset had no failure within 2 s


def test_novelty_and_inconsistency():
    rng = np.random.default_rng(4)
    nm = NoveltyModel.fit(rng.normal(size=(2000, 8)))
    assert nm.score(np.full((1, 8), 5.0))[0] > nm.score(np.zeros((1, 8)))[0]
    ep = _episode(n=3)
    ep["waypoints"][2, :, 1] = 1.0
    inc = inconsistency(ep)
    assert inc[0] == 0 and inc[1] == 0 and inc[2] > 0.9
