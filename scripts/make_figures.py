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


def phase5(summary_path: Path) -> None:
    import json

    from driveaware.data.labels import SITUATIONS
    from driveaware.viz.charts import ablation_line, closed_loop_bars, grouped_bars

    sm = json.loads(Path(summary_path).read_text())
    cl = sm["closed_loop"]
    e = cl["Expert"]
    agents = {
        "Expert": {
            "completion": [e["route_completion"]],
            "collision": [e["collision_episode_rate"]],
            "red_light": [e["per_km"]["red_light"]],
        }
    }
    for g in ("Model A", "Model B (H=1)", "Model B"):
        agents[g] = {
            "completion": cl[g]["route_completion"],
            "collision": cl[g]["collision_episode_rate"],
            "red_light": cl[g]["per_km"]["red_light"],
        }
    closed_loop_bars(
        agents,
        ASSETS / "closed_loop_phase5.png",
        title=f"Closed-loop driving on test_id ({sm['num_scenarios']} routes, paired)",
        caption="bars: mean over 3 training seeds · dots: individual seeds · "
        "Model B = frozen model-A encoder + GRU over 4 frames; "
        "B (H=1) = same head, no history",
        panels=(
            ("completion", "Route completion (%)", 100.0),
            ("collision", "Episodes with a collision (%)", 100.0),
            ("red_light", "Red-light violations per km", 1.0),
        ),
    )
    ol = sm["open_loop_test"]
    names = {
        "start_from_rest": "start from rest",
        "stopping": "stopping",
        "holding": "holding",
        "cruising": "cruising",
    }
    order = [s for s in ("start_from_rest", "stopping", "cruising", "holding") if s in SITUATIONS]
    series, errors = {}, {}
    for g in ("Model A", "Model B (H=1)", "Model B"):
        vals = [np.asarray(ol[g][f"ade_{s}"], dtype=float) for s in order]
        series[g] = [float(v.mean()) for v in vals]
        errors[g] = [float(v.std(ddof=1)) for v in vals]
    counts = ", ".join(f"{names[s]} n={ol['Model A'][f'n_{s}'][0]}" for s in order)
    grouped_bars(
        [names[s] for s in order],
        series,
        ASSETS / "situations_phase5.png",
        "Open-loop ADE by motion situation (test_id)",
        "ADE (m)",
        caption=f"mean ± sd over 3 seeds · {counts}",
        errors=errors,
    )
    ab = sm["val_ablation"]
    hs = [
        ("model_b_h1_seed0", 1),
        ("ablate_b_h2_seed0", 2),
        ("model_b_h4_seed0", 4),
        ("ablate_b_h8_seed0", 8),
    ]
    xs = [h for k, h in hs if k in ab]
    ys = [ab[k]["best_val"]["ade"] for k, _ in hs if k in ab]
    notes = []
    for key, label in (
        ("model_a_seed0", "model A"),
        ("ablate_b_h4_speedhist_seed0", "H=4 + speed history"),
        ("ablate_b_e2e_seed0", "H=4 end-to-end"),
    ):
        if key in ab:
            notes.append((label, ab[key]["best_val"]["ade"]))
    ablation_line(
        xs,
        ys,
        ASSETS / "history_ablation.png",
        "History length, frozen encoder (val ADE, seed 0)",
        "frames of history (H)",
        "ADE (m)",
        notes=notes,
    )
    print("wrote phase-5 charts to", ASSETS)


def _intro_scenarios(name: str):
    from driveaware.evaluation.closed_loop import scenario_set

    cfg = __import__("yaml").safe_load(
        (REPO_ROOT / "configs/evaluation/closed_loop.yaml").read_text()
    )
    sc = cfg["sets"][name]
    base = ScenarioSpec(0, 0, 0, route_length=cfg["route_length"], max_duration=cfg["max_duration"])
    maps = list(range(sc["map_seed_start"], sc["map_seed_start"] + sc["num_maps"]))
    return scenario_set(name, maps, sc["routes_per_map"], base, cfg["seed"])


def phase6(run_dir: Path) -> None:
    import json

    from driveaware.introspection import metrics as M
    from driveaware.viz.charts import closed_loop_bars, histogram, reliability_diagram, roc_pr

    res = json.loads((run_dir / "results.json").read_text())
    seeds = sorted(res, key=int)
    names = {
        "introspection (C)": "Introspection (C)",
        "all heuristics": "All heuristics",
        "context (speed + command)": "Context heuristic",
        "oracle (privileged)": "Oracle (privileged)",
    }
    # ROC / PR curves for seed 0 (test_id, all frames)
    z = np.load(run_dir / f"test_predictions_seed{seeds[0]}.npz")
    n_ep = len([k for k in z.files if k.startswith("y_")])
    y = np.concatenate([z[f"y_{i}"] for i in range(n_ep)])
    keys = {
        "Introspection (C)": "head",
        "All heuristics": "all",
        "Context heuristic": "ctx",
        "Oracle (privileged)": "oracle",
    }
    curves = {}
    for label, key in keys.items():
        s_ = np.concatenate([z[f"p_{key}_{i}"] for i in range(n_ep)])
        fpr, tpr = M.roc_curve(y, s_)
        rec, prec = M.pr_curve(y, s_)
        curves[label] = {
            "fpr": fpr,
            "tpr": tpr,
            "recall": rec,
            "precision": prec,
            "auroc": M.auroc(y, s_),
            "pr_auc": M.average_precision(y, s_),
            "prevalence": float(y.mean()),
            "dashed": "Oracle" in label,
        }
    roc_pr(
        curves,
        ASSETS / "introspection_roc_pr.png",
        "Predicting a failure within 2 s (test_id, model B seed 0)",
        caption=f"{len(y):,} decision steps from {n_ep} test episodes · oracle uses privileged "
        "state (dashed, reference only)",
    )
    # Metric bars across seeds
    agents = {}
    for key, label in names.items():
        agents[label] = {
            "auroc": [res[s]["predictors"][key]["auroc"] for s in seeds],
            "pr_auc": [res[s]["predictors"][key]["pr_auc"] for s in seeds],
            "detection": [
                res[s]["predictors"][key]["event_level"]["detection_rate"] for s in seeds
            ],
        }
    closed_loop_bars(
        agents,
        ASSETS / "introspection_metrics.png",
        title="Failure prediction on test_id (150 routes per policy seed)",
        caption="bars: mean over 3 policy seeds · dots: seeds · detection: share of failure "
        "events with an active alarm in the 2 s before them (threshold: 10% FPR on "
        "intro_cal)",
        panels=(
            ("auroc", "AUROC", 1.0),
            ("pr_auc", "PR-AUC", 1.0),
            ("detection", "Events detected (%)", 100.0),
        ),
    )
    r0 = res[seeds[0]]["predictors"]["introspection (C)"]
    reliability_diagram(
        {
            "calibrated (Platt on intro_cal)": {
                **r0["reliability"],
                "ece": r0["ece"],
                "entity": "Introspection (C)",
            },
            "uncalibrated": {
                **r0["uncalibrated"]["reliability"],
                "ece": r0["uncalibrated"]["ece"],
                "entity": "Introspection (C)",
                "dashed": True,
            },
        },
        ASSETS / "introspection_reliability.png",
        "Introspection calibration (test_id, seed 0)",
    )
    leads = [
        v
        for s in seeds
        for v in res[s]["predictors"]["introspection (C)"]["event_level"]["lead_times_s"]
    ]
    histogram(
        leads,
        ASSETS / "introspection_lead_time.png",
        "Warning lead time of detected failures",
        "seconds before the failure event",
        "Introspection (C)",
        np.arange(0, max(8.0, max(leads) + 0.5), 0.5),
        caption="pooled over 3 seeds; alarms persisting from earlier count from when they "
        "were raised",
    )
    print("wrote phase-6 charts to", ASSETS)


def warning_gif(
    run_dir: Path, seed: int, outcome: str, name: str, seconds: float, stride: int
) -> None:
    """Replay a test episode with the introspection probability as a live timeline."""
    import json

    from driveaware.core.types import EventType
    from driveaware.evaluation.closed_loop import AgentSpec, run_episode

    res_json = json.loads((run_dir / f"results_seed{seed}.json").read_text())
    thr = res_json["predictors"]["introspection (C)"]["event_level"]["threshold"]
    policy = res_json["policy_run"]
    ep_dir = REPO_ROOT / "data/introspection" / policy / "test_id"
    files = sorted(ep_dir.glob("ep_*.npz"))
    pick = next(
        i for i, f in enumerate(files) if EventType(int(np.load(f)["outcome"])).name == outcome
    )
    z = np.load(run_dir / f"test_predictions_seed{seed}.npz")
    p, ts = z[f"p_head_{pick}"], z[f"t_{pick}"]
    ep = np.load(files[pick])
    fail_t = [
        t
        for t, ty in zip(ep["event_time"], ep["event_type"], strict=True)
        if EventType(int(ty)).name
        in ("COLLISION", "OFF_ROAD", "ROUTE_DEVIATION", "RED_LIGHT", "STALL")
    ]
    spec = _intro_scenarios("test_id")[pick]
    row = run_episode(
        spec, AgentSpec("policy", str(REPO_ROOT / "experiments" / policy / "best.pt")), record=True
    )
    sim = LightweightDrivingSim()
    sim.reset(spec)
    td = TopDownRenderer(sim.episode.road_map)
    cam = sim.episode.renderer
    frames_all = row["frames"]
    n_steps = len(p)
    start = max(0, n_steps - int(seconds * 10))
    images = []
    above = p > thr
    run = np.zeros(len(p), int)
    for i, a in enumerate(above):
        run[i] = (run[i - 1] + 1 if i > 0 else 1) if a else 0
    for k in list(range(start, n_steps, stride)) + [n_steps]:
        res, wp = frames_all[min(k, len(frames_all) - 1)]
        frame = compose_frame(res, td)
        if wp is not None:
            uv, front = cam.project_ego(wp)
            for (u, v), ok in zip(uv * 3, front, strict=True):
                if ok:
                    cv2.circle(frame, (int(u), int(v)), 4, (0, 255, 255), -1, cv2.LINE_AA)
        frames_all_h, w = frame.shape[:2]
        strip_h = 110
        strip = np.full((strip_h, w, 3), 26, np.uint8)
        kk = min(k, n_steps - 1)
        t_now = ts[kk]
        if k == n_steps and fail_t:  # final frame: advance the timeline to the terminal event
            t_now = max(t_now, max(fail_t))
        win = 10.0
        x0, x1, y0, y1 = 40, w - 150, 12, strip_h - 22

        def xy(t, pv, t_now=t_now, x0=x0, x1=x1, y0=y0, y1=y1, win=win):
            return (int(x0 + (t - (t_now - win)) / win * (x1 - x0)), int(y1 - pv * (y1 - y0)))

        cv2.rectangle(strip, (x0, y0), (x1, y1), (60, 60, 58), 1)
        ty = xy(t_now, thr)[1]
        cv2.line(strip, (x0, ty), (x1, ty), (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(
            strip,
            f"alarm threshold {thr:.2f}",
            (x0 + 4, ty - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
        sel = (ts > t_now - win) & (ts <= t_now)
        pts = np.array([xy(t, pv) for t, pv in zip(ts[sel], p[sel], strict=True)], np.int32)
        if len(pts) > 1:
            cv2.polylines(strip, [pts], False, (232, 123, 164), 2, cv2.LINE_AA)
        for tf in fail_t:
            if t_now - win < tf <= t_now + 0.15:
                xf = xy(tf, 0)[0]
                cv2.line(strip, (xf, y0), (xf, y1), (227, 73, 72), 2)
        cv2.putText(
            strip,
            "P(failure within 2 s)",
            (x0, strip_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            strip,
            f"{p[kk]:.2f}",
            (x1 + 12, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        if run[kk] >= 2:
            cv2.rectangle(strip, (x1 + 10, y0 + 45), (w - 10, y0 + 75), (227, 73, 72), -1)
            cv2.putText(
                strip,
                "WARNING",
                (x1 + 18, y0 + 67),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        full = np.vstack([frame, strip])
        small = cv2.resize(full, None, fx=0.62, fy=0.62, interpolation=cv2.INTER_AREA)
        images.append(Image.fromarray(small).quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    images[0].save(
        ASSETS / name,
        save_all=True,
        append_images=images[1:],
        duration=int(100 * stride),
        loop=0,
        optimize=True,
    )
    print("wrote", ASSETS / name, "episode", pick, "outcome", outcome, "frames", len(images))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["sim", "rollout", "charts", "phase5", "phase6", "warning"])
    parser.add_argument("--run-dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outcome", default="COLLISION")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--summary")
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
    elif args.what == "phase5":
        phase5(Path(args.summary))
    elif args.what == "phase6":
        phase6(Path(args.run_dir))
    elif args.what == "warning":
        warning_gif(
            Path(args.run_dir), args.seed, args.outcome, args.name, args.seconds, args.stride
        )
    else:
        from driveaware.evaluation.closed_loop import scenario_set

        base = ScenarioSpec(0, 0, 0, route_length=300.0, max_duration=150.0)
        spec = scenario_set("test_id", [args.map_seed], 5, base)[args.route_seed]
        rollout_gif(args.checkpoint, spec, args.name, args.frames, args.stride, args.history)


if __name__ == "__main__":
    main()
