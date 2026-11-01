"""Hardware/environment introspection and resource measurement.

Measures host RAM (process RSS and system availability), GPU memory (PyTorch allocator peak
and NVML device usage, which includes other processes such as the Windows desktop under WSL),
and latency percentiles.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import psutil

MiB = 1024**2


@dataclass
class ResourceSnapshot:
    process_rss_mib: float
    system_available_mib: float
    torch_allocated_mib: float | None = None
    torch_peak_allocated_mib: float | None = None
    torch_peak_reserved_mib: float | None = None
    gpu_used_mib: float | None = None  # NVML, whole device
    gpu_total_mib: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _torch_cuda() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def _nvml_memory(device_index: int = 0) -> tuple[float, float] | None:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device_index))
            return info.used / MiB, info.total / MiB
        finally:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        return None


def snapshot(device_index: int = 0) -> ResourceSnapshot:
    snap = ResourceSnapshot(
        process_rss_mib=psutil.Process().memory_info().rss / MiB,
        system_available_mib=psutil.virtual_memory().available / MiB,
    )
    torch = _torch_cuda()
    if torch is not None:
        snap.torch_allocated_mib = torch.cuda.memory_allocated(device_index) / MiB
        snap.torch_peak_allocated_mib = torch.cuda.max_memory_allocated(device_index) / MiB
        snap.torch_peak_reserved_mib = torch.cuda.max_memory_reserved(device_index) / MiB
    nvml = _nvml_memory(device_index)
    if nvml is not None:
        snap.gpu_used_mib, snap.gpu_total_mib = nvml
    return snap


def reset_peak_gpu_memory(device_index: int = 0) -> None:
    torch = _torch_cuda()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats(device_index)


@dataclass
class LatencyStats:
    """Collects durations in seconds; reports milliseconds."""

    samples: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.samples.append(seconds)

    @contextmanager
    def measure(self, cuda_sync: bool = False) -> Iterator[None]:
        torch = _torch_cuda() if cuda_sync else None
        if torch is not None:
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch is not None:
                torch.cuda.synchronize()
            self.add(time.perf_counter() - start)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"count": 0}
        ms = np.asarray(self.samples) * 1e3
        return {
            "count": int(ms.size),
            "mean_ms": float(ms.mean()),
            "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)),
            "max_ms": float(ms.max()),
        }


def environment_info(repo_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Versions and hardware identifiers recorded with every run (``env.json``)."""
    vm = psutil.virtual_memory()
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_mib": round(vm.total / MiB),
        "numpy": np.__version__,
        "pythonpath_set": bool(os.environ.get("PYTHONPATH")),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_capability"] = f"{props.major}.{props.minor}"
            info["gpu_total_mib"] = round(props.total_memory / MiB)
            info["cudnn"] = torch.backends.cudnn.version()
            info["torch_arch_list"] = torch.cuda.get_arch_list()
    except ImportError:
        info["torch"] = None
    info.update(git_info(repo_dir))
    return info


def git_info(repo_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = run("status", "--porcelain")
    dirty = bool(status) if status is not None else None
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": dirty}
