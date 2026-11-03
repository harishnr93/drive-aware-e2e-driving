import json
import time

import numpy as np

from driveaware.core.hardware import LatencyStats, environment_info, snapshot
from driveaware.core.logging import MetricsWriter, read_metrics, setup_logging


def test_metrics_writer(tmp_path):
    with MetricsWriter(tmp_path / "m.jsonl") as w:
        w.write({"step": 1, "loss": np.float32(0.5)})
        w.write({"step": 2, "arr": np.arange(3)})
    rows = read_metrics(tmp_path / "m.jsonl")
    assert rows[0]["loss"] == 0.5 and rows[1]["arr"] == [0, 1, 2]


def test_setup_logging_writes_file(tmp_path):
    log = setup_logging(tmp_path)
    log.info("hello")
    for h in log.handlers:
        h.flush()
    assert "hello" in (tmp_path / "run.log").read_text()


def test_latency_stats():
    lat = LatencyStats()
    for _ in range(5):
        with lat.measure():
            time.sleep(0.001)
    s = lat.summary()
    assert s["count"] == 5 and s["p50_ms"] >= 0.9


def test_snapshot_and_env_info_are_serialisable():
    snap = snapshot()
    assert snap.process_rss_mib > 0
    json.dumps(snap.as_dict())
    info = environment_info()
    json.dumps(info)
    assert "python" in info and "torch" in info


def test_atomic_savez_roundtrip_and_no_tmp_left(tmp_path):
    from driveaware.core.run import atomic_savez

    out = atomic_savez(tmp_path / "a.npz", x=np.arange(5))
    with np.load(out) as z:
        assert z["x"].tolist() == [0, 1, 2, 3, 4]
    assert not list(tmp_path.glob("*.tmp"))
