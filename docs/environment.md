# Development Environment

Status: **Phase 0 — measured 2026-11-01. Nothing has been installed yet.**

## 1. Measured host facts

| Item | Value | How measured |
|---|---|---|
| Windows | 10.0.26200.9711 (Windows 11 build line) | `cmd.exe /c ver` |
| WSL kernel | 6.6.87.2-microsoft-standard-WSL2 | `uname -r` |
| Linux distro | Ubuntu 22.04.5 LTS | `/etc/os-release` |
| CPU | AMD Ryzen 7 4800H, 16 logical CPUs visible | `/proc/cpuinfo`, `nproc` |
| RAM (WSL) | 7.4 GiB total, ~5.1 GiB available at idle | `free -h` |
| Swap (WSL) | 2.0 GiB | `free -h` |
| Disk | 868 GB free on `/` | `df -h` |
| GPU | GeForce GTX 1650 Ti, 4096 MiB, compute capability 7.5 (Turing) | `nvidia-smi --query-gpu` |
| GPU memory at idle | ~419 MiB already used (Windows desktop compositor) | `nvidia-smi` |
| Driver | Windows 591.74, WSL userspace 590.52.01, CUDA driver API 13.1 | `nvidia-smi` |
| CUDA toolkit (nvcc) | not installed (not needed: PyTorch wheels ship their own runtime) | `which nvcc` |
| Python | system 3.10.12 (`/usr/bin/python3`), 3.12.13 also present | `python3 --version` |
| PyTorch | not installed | import test |
| Conda | miniconda3 present, base not auto-activated | `conda env list` |
| ROS 2 | Humble, source build at `~/ros2_humble`, sourced in `~/.bashrc` | `~/.bashrc` |
| Tools | gcc, cmake, ffmpeg present | `which` |

**Effective GPU budget: ~3.5 GiB**, because the Windows desktop holds ~0.4 GiB and the
CUDA context costs ~0.3–0.5 GiB more. Plan for **≤ 3 GiB of tensors** per process.

## 2. Compatibility findings

1. **PyTorch / CUDA.** The driver supports CUDA ≤ 13.1, so any cu12x wheel works.
   Index query (pip, Python 3.10):
   - `cu126` index: latest `2.14.1+cu126` has cp310 wheels.
   - `cu128` index: stops at `2.11.0+cu128`.
   - The Turing GPU (sm_75) must be present in `torch.cuda.get_arch_list()`. **This must be
     checked after install.** If sm_75 is missing, fall back to an earlier `+cu126` release.
2. **Mixed precision.** Turing has FP16 tensor cores but **no hardware BF16**. If AMP is
   adopted later, it will be `float16` with a `GradScaler`. Default is FP32.
3. **ROS 2 environment leakage (real risk).** `~/.bashrc` sources Humble, so `PYTHONPATH`
   contains dozens of ROS paths. These paths take precedence over a venv's site-packages,
   and they include pytest plugins (`launch_testing`, `launch_pytest`) that load
   automatically. **Mitigation:** the project venv is activated through `scripts/env.sh`,
   which unsets `PYTHONPATH` and the `AMENT_*`/`COLCON_*` variables. ROS bridging runs in a
   separate shell that has ROS sourced.
4. **CARLA wheels.** PyPI has `carla` 0.9.15 and 0.9.16 for the system Python. Phase 10
   checks the server side, not now (see §5).
5. **System numpy is 1.21.5** (apt). It must not be used: the venv is isolated.

## 3. Recommended environment

| Decision | Choice | Rationale |
|---|---|---|
| Interpreter | **Python 3.10** (system) in a project `.venv` | Same ABI as ROS 2 Humble `rclpy` and the CARLA client wheels; supported by current PyTorch |
| Env manager | `python3.10 -m venv .venv` + `pip` | Lighter than conda and avoids mixing conda with ROS libraries |
| PyTorch | `torch==2.14.1+cu126` (+ matching `torchvision`) from `download.pytorch.org/whl/cu126` | Latest line with a cp310 CUDA build. Verify sm_75 after install |
| CPU-only CI | `torch` CPU wheel | Core algorithms are tested without a GPU |
| Core deps | numpy, opencv-python-headless, pyyaml, matplotlib, tqdm, psutil, pynvml (nvidia-ml-py) | Small. No Hydra, no heavy RL frameworks |
| Dev deps | pytest, ruff, mypy (interfaces only) | |
| Optional groups | `[ros]` (no pip deps, uses sourced Humble), `[carla]` (`carla==0.9.15/16`, chosen in Phase 10) | Kept out of the core install |

Estimated install size: about 3 GB for torch + CUDA libraries, plus about 0.3 GB for everything else.

## 4. Resource budget (planning numbers, to be replaced by measurements)

| Workload | Budget |
|---|---|
| Simulator + expert data collection | CPU only, 1 process per core, ≤ 300 MB RAM each |
| Training | 1 GPU job at a time, ≤ 3 GiB VRAM, ≤ 4 GiB host RAM, 2–4 DataLoader workers over a memory-mapped dataset |
| Closed-loop evaluation | CPU sim + GPU policy, batch 1. Latency is measured per step |
| Dataset | 160×96×3 uint8 = 46 KB/frame. 100k frames ≈ 4.6 GB raw (disk is ample, RAM is not, so use memmap) |

## 5. CARLA feasibility (assessment only, nothing installed)

- CARLA 0.9.x (UE4) recommends a GPU with ≥ 6–8 GB. A 4 GB card is **below spec**.
- A **Linux CARLA server inside WSL2 is not a realistic path**: it needs Vulkan, and WSL2
  does not expose a native NVIDIA Vulkan driver.
- Most viable local option: run the **Windows** CARLA package natively (`-quality-level=Low`,
  small towns such as Town01/02, synchronous mode, one 160×96 camera), with the Python
  client in WSL connecting over TCP 2000–2002. WSL already reserves 7.4 GiB of host RAM,
  so the host RAM left for UE4 must be checked.
- Fallback: a remote CARLA server (cloud GPU) over the same client interface.
- Decision deferred to Phase 10. The rest of the project does not depend on CARLA.

## 6. Reproducibility record

Every training/evaluation run will write `env.json` with: git commit and dirty flag,
Python/torch/CUDA/cuDNN versions, GPU name, driver, and `pip freeze` hash.
