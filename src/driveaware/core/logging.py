"""Structured logging: human-readable console/file logs plus JSON-lines metric records."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(run_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the ``driveaware`` logger for console and (optionally) ``run_dir/run.log``."""
    logger = logging.getLogger("driveaware")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    logger.addHandler(console)
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(run_dir) / "run.log")
        fh.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(fh)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"driveaware.{name}")


class MetricsWriter:
    """Append-only JSON-lines writer: one record per call, with wall-clock time."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a")  # noqa: SIM115 (closed in close())

    def write(self, record: dict[str, Any], **extra: Any) -> None:
        row = {"wall_time": time.time(), **record, **extra}
        self._fh.write(json.dumps(row, default=_jsonable) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "tolist"):  # numpy / torch scalars and arrays
        return obj.tolist()
    return str(obj)
