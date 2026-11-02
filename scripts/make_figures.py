"""Generate the README / docs figures into docs/assets/ (reproducible from seeds and runs).

Usage:
    python scripts/make_figures.py sim          # map, camera/top-down, label montage
    python scripts/make_figures.py rollout --checkpoint <best.pt>   # policy rollout GIF
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from driveaware.core.run import REPO_ROOT
from driveaware.expert.expert import RuleBasedExpert
from driveaware.sim.lightweight.env import LightweightDrivingSim
from driveaware.sim.lightweight.renderer import TopDownRenderer
from driveaware.sim.lightweight.scenario import ScenarioSpec
from driveaware.viz.replay import compose_frame

ASSETS = REPO_ROOT / "docs" / "assets"


def save_png(img_rgb: np.ndarray, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(ASSETS / name),
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    print("wrote", ASSETS / name)


def map_overview(spec: ScenarioSpec) -> np.ndarray:
    sim = LightweightDrivingSim()
    res = sim.reset(spec)
    m = sim.episode.road_map
    scale = 0.25
    img = cv2.resize(m.color, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).copy()
    route = res.privileged.extra["route"].path.points
    px = ((route - m.origin) / m.config.resolution * scale * 16).astype(np.int32)
    cv2.polylines(img, [px], False, (40, 140, 255), 3, cv2.LINE_AA, 4)
    for box in res.privileged.agents:
        c = ((box.corners() - m.origin) / m.config.resolution * scale * 16).astype(np.int32)
        cv2.fillConvexPoly(img, c, (220, 40, 40), cv2.LINE_AA, 4)
    e = (res.privileged.ego_footprint - m.origin) / m.config.resolution * scale * 16
    cv2.fillConvexPoly(img, e.astype(np.int32), (255, 160, 0), cv2.LINE_AA, 4)
    img = img[::-1].copy()  # north up
    return cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)


def expert_frames(spec: ScenarioSpec) -> np.ndarray:
    """One frame each: cruising, turning, waiting at a red light (first occurrence)."""
    from driveaware.core.types import Command

    def cruising(r):
        return r.observation.speed > 6.0 and r.observation.command == Command.LANE_FOLLOW

    def turning(r):
        return (
            r.observation.command in (Command.LEFT, Command.RIGHT)
            and abs(r.privileged.ego_steer) > 0.15
        )

    def red_wait(r):
        nxt = r.privileged.extra["next_stop"]
        return (
            r.observation.speed < 0.05
            and nxt is not None
            and nxt["state"].name == "RED"
            and nxt["s"] - r.privileged.route_progress < 9.0
        )

    sim = LightweightDrivingSim()
    res = sim.reset(spec)
    td = TopDownRenderer(sim.episode.road_map)
    expert = RuleBasedExpert()
    found: dict[str, np.ndarray] = {}
    while not res.done and len(found) < 3:
        for name, pred in (("cruise", cruising), ("turn", turning), ("red", red_wait)):
            if name not in found and pred(res):
                found[name] = compose_frame(res, td)
        res = sim.step(expert(res))
    if len(found) < 3:
        return None
    return np.vstack([found[k] for k in ("cruise", "turn", "red")])


def label_montage(split_dir: Path) -> np.ndarray:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "inspect_dataset", REPO_ROOT / "scripts/inspect_dataset.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from driveaware.data.dataset import DrivingDataset

    img = mod.montage(DrivingDataset(split_dir), num=8, seed=3, scale=2)
    return img


def rollout_gif(
    checkpoint: str, spec: ScenarioSpec, name: str, max_frames: int, stride: int, history: int = 1
) -> None:
    from driveaware.evaluation.closed_loop import AgentSpec, run_episode

    row = run_episode(spec, AgentSpec("policy", checkpoint, history), record=True)
    sim = LightweightDrivingSim()
    sim.reset(spec)
    td = TopDownRenderer(sim.episode.road_map)
    cam = sim.episode.renderer
    frames = []
    # The last ``max_frames`` sampled frames, so the clip ends with the episode outcome.
    for res, wp in row["frames"][::-1][::stride][:max_frames][::-1]:
        frame = compose_frame(res, td)
        if wp is not None:
            uv, front = cam.project_ego(wp)
            for (u, v), ok in zip(uv * 3, front, strict=True):
                if ok:
                    cv2.circle(frame, (int(u), int(v)), 4, (0, 255, 255), -1, cv2.LINE_AA)
        small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(small).quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    ASSETS.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        ASSETS / name,
        save_all=True,
        append_images=frames[1:],
        duration=int(100 * stride),
        loop=0,
        optimize=True,
    )
    print("wrote", ASSETS / name, "outcome", row["outcome"], "completion", row["completion"])


def charts(report_path: Path, run_glob: str) -> None:
    import json

    from driveaware.core.logging import read_metrics
    from driveaware.viz.charts import closed_loop_bars, training_curves

    runs = {}
    for k, run in enumerate(sorted(REPO_ROOT.glob(run_glob))):
        rows = [r for r in read_metrics(run / "metrics.jsonl") if "epoch" in r]
        runs[f"seed {k}"] = [r["val_ade"] for r in rows]
    from driveaware.evaluation.open_loop import reference_baselines

    report = json.loads(Path(report_path).read_text())
    ref = reference_baselines(REPO_ROOT / "data/expert_v1/val")  # same split as the curves
    reference = {"constant-velocity baseline (val)": ref["constant_velocity_straight"]["ade"]}
    training_curves(runs, reference, ASSETS / "training_curves_model_a.png")
    agents = report["agents"]
    policy = [a for n, a in agents.items() if n != "expert"]
    data = {}
    if "expert" in agents:
        e = agents["expert"]["closed_loop"]
        data["Expert"] = {
            "completion": [e["route_completion"]],
            "completion_ci": e["route_completion_ci95"],
            "collision": [e["collision_episode_rate"]],
            "collision_ci": e["collision_episode_rate_ci95"],
        }
    if policy:
        data["Model A"] = {
            "completion": [a["closed_loop"]["route_completion"] for a in policy],
            "collision": [a["closed_loop"]["collision_episode_rate"] for a in policy],
        }
    n = report["num_scenarios"]
    closed_loop_bars(
        data,
        ASSETS / "closed_loop_model_a.png",
        title=f"Closed-loop driving on test_id ({n} routes, same scenarios for all agents)",
        caption="bars: mean over 3 training seeds · dots: individual seeds · "
        "expert is a single deterministic run",
    )
    print("wrote charts to", ASSETS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["sim", "rollout", "charts"])
    parser.add_argument("--report")
    parser.add_argument("--runs", default="experiments/*_model_a_seed*")
    parser.add_argument("--checkpoint")
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--name", default="rollout_model_a.gif")
    parser.add_argument("--map-seed", type=int, default=3000)
    parser.add_argument("--route-seed", type=int, default=0, help="index within the map's routes")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    if args.what == "sim":
        spec = ScenarioSpec(205, 3, 3)  # map outside all dataset/evaluation splits
        save_png(map_overview(spec), "map_overview.png")
        for route_seed in range(3, 30):  # first route on map 205 showing all three situations
            frames = expert_frames(ScenarioSpec(205, route_seed, route_seed))
            if frames is not None:
                save_png(frames, "camera_topdown.png")
                break
        save_png(label_montage(REPO_ROOT / "data/expert_v1/train"), "label_montage.png")
    elif args.what == "charts":
        charts(Path(args.report), args.runs)
    else:
        from driveaware.evaluation.closed_loop import scenario_set

        base = ScenarioSpec(0, 0, 0, route_length=300.0, max_duration=150.0)
        spec = scenario_set("test_id", [args.map_seed], 5, base)[args.route_seed]
        rollout_gif(args.checkpoint, spec, args.name, args.frames, args.stride, args.history)


if __name__ == "__main__":
    main()
