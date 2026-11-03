"""Static result charts (PNG, light and dark variants) for the README and docs.

Palette and chrome follow the reference data-viz palette: categorical slots in fixed order
(blue, orange, aqua), recessive grid and axes, text in ink colours (never the series colour),
direct labels, and a legend for >= 2 series. One measure per axis (no dual axes).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": [
            "#2a78d6",
            "#eb6834",
            "#1baf7a",
            "#eda100",
            "#e87ba4",
            "#008300",
            "#6250d6",
            "#e34948",
        ],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": [
            "#3987e5",
            "#d95926",
            "#199e70",
            "#c98500",
            "#d55181",
            "#008300",
            "#9085e9",
            "#e66767",
        ],
    },
}


# Colour follows the entity, never its position in a particular chart.
ENTITY_SLOT = {
    "Expert": 0,
    "Model A": 1,
    "Model B": 2,
    "Model B (H=1)": 3,
    "Introspection (C)": 4,
    "All heuristics": 5,
    "Context heuristic": 6,
    "Oracle (privileged)": 7,
    # Phase 7 uncertainty family (separate chart family; legends always shown)
    "Ensemble": 2,
    "Ensemble disagreement": 2,
    "MC dropout": 3,
    "Laplace head": 6,
    "Laplace scale": 6,
    "Ensemble + Laplace": 0,
    "Introspection + uncertainty": 5,
    "Prediction error (ADE)": 7,
}


def _wrap(label: str, width: int = 13) -> str:
    """Wrap tick labels on spaces; parenthesised qualifiers go on their own line."""
    import textwrap

    head, sep, tail = label.partition(" (")
    lines = textwrap.wrap(head, width) or [head]
    if sep:
        lines.append("(" + tail)
    return "\n".join(lines)


def entity_color(t: dict, label: str, fallback: int) -> str:
    return t["series"][ENTITY_SLOT.get(label, fallback)]


def _style(ax, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(t: dict, ncols: int, width: float, height: float):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=150, squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    for ax in axes[0]:
        _style(ax, t)
    return fig, axes[0]


def training_curves(
    runs: dict[str, list[float]],
    reference: dict[str, float],
    out: Path,
    title: str = "Validation ADE during training (model A)",
) -> None:
    """runs: label -> per-epoch val ADE (m). reference: label -> constant line (m)."""
    for mode, t in THEMES.items():
        fig, (ax,) = _figure(t, 1, 6.4, 3.4)
        ymax = 0.0
        for k, (label, ys) in enumerate(runs.items()):
            x = np.arange(1, len(ys) + 1)
            color = t["series"][k]  # seeds of one model: slots in fixed order
            ax.plot(x, ys, color=color, linewidth=2, label=label, solid_capstyle="round")
            ax.plot(
                x[-1],
                ys[-1],
                "o",
                color=color,
                markersize=6,
                markeredgecolor=t["surface"],
                markeredgewidth=1.5,
            )
            ax.annotate(
                f"{label}  {ys[-1]:.3f} m",
                (x[-1], ys[-1]),
                xytext=(8, (k - 1) * 9),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=t["ink2"],
            )
            ymax = max(ymax, max(ys))
        for label, y in reference.items():
            ax.axhline(y, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate(
                label,
                (1, y),
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=8,
                color=t["muted"],
            )
            ymax = max(ymax, y)
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xlim(0.7, len(next(iter(runs.values()))) + 3.2)
        ax.set_xlabel("epoch", color=t["ink2"], fontsize=9)
        ax.set_ylabel("ADE (m)", color=t["ink2"], fontsize=9)
        ax.set_title(title, color=t["ink"], fontsize=11, loc="left")
        leg = ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=t["ink2"])
        leg.set_visible(len(runs) >= 2)
        fig.tight_layout()
        fig.savefig(out.with_name(f"{out.stem}_{mode}.png"), facecolor=t["surface"])
        plt.close(fig)


def closed_loop_bars(
    agents: dict[str, dict],
    out: Path,
    title: str = "Closed-loop driving on test_id (50 routes)",
    caption: str = "bars: mean over training seeds · dots: individual training seeds",
    panels: tuple[tuple[str, str, float], ...] | None = None,
) -> None:
    """agents: label -> {"completion": [per-seed], "completion_ci": (lo, hi) | None,
    "collision": [per-seed], "collision_ci": ...}. Values are fractions (0-1)."""
    panels = panels or (
        ("completion", "Route completion (%)", 100.0),
        ("collision", "Episodes with a collision (%)", 100.0),
    )
    labels = list(agents)
    for mode, t in THEMES.items():
        fig, axes = _figure(t, len(panels), 3.6 * len(panels) + 0.6 * len(labels), 3.4)
        for ax, (key, ylabel, scale) in zip(axes, panels, strict=True):
            for i, label in enumerate(labels):
                vals = scale * np.asarray(agents[label][key], dtype=float)
                color = entity_color(t, label, i)
                mean = float(vals.mean())
                ax.bar(i, mean, width=0.55, color=color, edgecolor=t["surface"], linewidth=2)
                if len(vals) > 1:
                    jitter = np.linspace(-0.12, 0.12, len(vals))
                    ax.plot(
                        i + jitter,
                        vals,
                        "o",
                        color=t["ink"],
                        markersize=4,
                        alpha=0.8,
                        markeredgecolor=t["surface"],
                        markeredgewidth=1,
                    )
                ci = agents[label].get(f"{key}_ci")
                top = float(vals.max())
                if ci is not None and ci[1] - ci[0] > 1e-9:  # skip degenerate intervals
                    ax.errorbar(
                        i,
                        mean,
                        yerr=[[mean - scale * ci[0]], [scale * ci[1] - mean]],
                        color=t["ink2"],
                        capsize=4,
                        linewidth=1.2,
                    )
                    top = max(top, scale * ci[1])
                ax.annotate(
                    f"{mean:.1f}" if scale == 100.0 else f"{mean:.2f}",
                    (i, top),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=t["ink"],
                )
            ax.set_xticks(range(len(labels)), [_wrap(lb) for lb in labels], fontsize=8)
            peak = max(scale * max(agents[lb][key]) for lb in labels)
            if key == "completion" or (scale == 100.0 and peak > 60):
                top_lim = 112.0
            elif scale == 1.0 and peak <= 1.0 and peak > 0.5:
                top_lim = 1.1  # probabilities / AUROC-like scores
            else:
                top_lim = max(1.0, 1.3 * peak)
            ax.set_ylim(0, top_lim)
            ax.set_title(ylabel, color=t["ink"], fontsize=10, loc="left")
        fig.suptitle(title, color=t["ink"], fontsize=11, x=0.02, ha="left")
        fig.text(0.02, 0.01, caption, color=t["muted"], fontsize=7)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        fig.savefig(out.with_name(f"{out.stem}_{mode}.png"), facecolor=t["surface"])
        plt.close(fig)


def grouped_bars(
    groups: list[str],
    series: dict[str, list[float]],
    out: Path,
    title: str,
    ylabel: str,
    caption: str = "",
    errors: dict[str, list[float]] | None = None,
) -> None:
    """Grouped bars: x = groups (e.g. motion situations), one bar per series (entity colours)."""
    names = list(series)
    width = 0.8 / len(names)
    for mode, t in THEMES.items():
        fig, (ax,) = _figure(t, 1, 7.2, 3.4)
        for k, name in enumerate(names):
            x = np.arange(len(groups)) + (k - (len(names) - 1) / 2) * width
            vals = np.asarray(series[name], dtype=float)
            ax.bar(
                x,
                vals,
                width=width * 0.92,
                color=entity_color(t, name, k),
                label=name,
                edgecolor=t["surface"],
                linewidth=1.5,
            )
            if errors and name in errors:
                ax.errorbar(
                    x, vals, yerr=errors[name], fmt="none", ecolor=t["ink2"], capsize=2, linewidth=1
                )
            errs = (errors or {}).get(name, [0.0] * len(vals))
            for xi, v, e in zip(x, vals, errs, strict=True):
                ax.annotate(
                    f"{v:.2f}",
                    (xi, v + e),  # above the whisker
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=t["ink2"],
                )
        ax.set_xticks(range(len(groups)), groups, fontsize=9)
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9)
        ax.set_title(title, color=t["ink"], fontsize=11, loc="left", pad=22)
        ax.legend(
            frameon=False,
            fontsize=8,
            labelcolor=t["ink2"],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=len(names),
            borderaxespad=0.3,
        )
        top = max(max(v) + max((errors or {}).get(n, [0.0] * len(v))) for n, v in series.items())
        ax.set_ylim(0, 1.15 * top)
        if caption:
            fig.text(0.02, 0.01, caption, color=t["muted"], fontsize=7)
        fig.tight_layout(rect=(0, 0.04 if caption else 0, 1, 1))
        fig.savefig(out.with_name(f"{out.stem}_{mode}.png"), facecolor=t["surface"])
        plt.close(fig)


def ablation_line(
    xs: list[float],
    ys: list[float],
    out: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    entity: str = "Model B",
    notes: list[tuple[str, float]] | None = None,
) -> None:
    """Single-series line with point labels; optional reference points drawn as hollow markers."""
    for mode, t in THEMES.items():
        fig, (ax,) = _figure(t, 1, 5.6, 3.2)
        color = entity_color(t, entity, 2)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=t["surface"],
            markeredgewidth=2,
        )
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(
                f"{y:.3f}",
                (x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=t["ink2"],
            )
        for label, y in notes or []:
            ax.axhline(y, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate(
                f"{label} {y:.3f}",
                (xs[-1], y),
                xytext=(-4, 3),
                textcoords="offset points",
                ha="right",
                fontsize=7,
                color=t["muted"],
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(int(x)) for x in xs])
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9)
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9)
        ax.set_title(title, color=t["ink"], fontsize=11, loc="left")
        lo, hi = min(ys + [y for _, y in notes or []]), max(ys + [y for _, y in notes or []])
        ax.set_ylim(lo - 0.25 * (hi - lo) - 1e-3, hi + 0.35 * (hi - lo) + 1e-3)
        fig.tight_layout()
        fig.savefig(out.with_name(f"{out.stem}_{mode}.png"), facecolor=t["surface"])
        plt.close(fig)


def roc_pr(curves: dict[str, dict], out: Path, title: str, caption: str = "") -> None:
    """curves: name -> {fpr, tpr, recall, precision, auroc, pr_auc, prevalence, dashed?}."""
    for mode, t in THEMES.items():
        fig, (ax1, ax2) = _figure(t, 2, 8.4, 3.7)
        ax1.plot([0, 1], [0, 1], color=t["axis"], linewidth=1)
        prev = next(iter(curves.values()))["prevalence"]
        ax2.axhline(prev, color=t["axis"], linewidth=1)
        ax2.annotate(
            f"prevalence {prev:.3f}",
            (1.0, prev),
            xytext=(-2, 3),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color=t["muted"],
        )
        for k, (name, c) in enumerate(curves.items()):
            color = entity_color(t, name, k)
            ls = (0, (4, 2)) if c.get("dashed") else "-"
            ax1.plot(
                c["fpr"],
                c["tpr"],
                color=color,
                linewidth=2,
                linestyle=ls,
                label=f"{name}  {c['auroc']:.3f}",
            )
            ax2.plot(
                c["recall"],
                c["precision"],
                color=color,
                linewidth=2,
                linestyle=ls,
                label=f"{name}  {c['pr_auc']:.3f}",
            )
        ax1.set(xlim=(0, 1), ylim=(0, 1.01))
        ax2.set(xlim=(0, 1), ylim=(0, 1.01))
        ax1.set_xlabel("false-positive rate", color=t["ink2"], fontsize=9)
        ax1.set_ylabel("true-positive rate", color=t["ink2"], fontsize=9)
        ax2.set_xlabel("recall", color=t["ink2"], fontsize=9)
        ax2.set_ylabel("precision", color=t["ink2"], fontsize=9)
        ax1.set_title("ROC (legend: AUROC)", color=t["ink"], fontsize=10, loc="left")
        ax2.set_title("Precision–recall (legend: PR-AUC)", color=t["ink"], fontsize=10, loc="left")
        ax1.legend(frameon=False, fontsize=7, labelcolor=t["ink2"], loc="lower right")
        ax2.legend(frameon=False, fontsize=7, labelcolor=t["ink2"], loc="upper right")
        fig.suptitle(title, color=t["ink"], fontsize=11, x=0.02, ha="left")
        if caption:
            fig.text(0.02, 0.01, caption, color=t["muted"], fontsize=7)
        fig.tight_layout(rect=(0, 0.04 if caption else 0, 1, 0.94))
        fig.savefig(out.with_name(f"{out.stem}_{mode}.png"), facecolor=t["surface"])
        plt.close(fig)


def reliability_diagram(series: dict[str, dict], out: Path, title: str) -> None:
    """series: name -> {confidence, frequency, ece}; plotted against the diagonal (log axes)."""
    for mode, t in THEMES.items():
        fig, (ax,) = _figure(t, 1, 4.6, 4.0)
        lo = 1e-3
        ax.plot([lo, 1], [lo, 1], color=t["axis"], linewidth=1)
        ax.annotate(
            "perfect calibration",
            (0.25, 0.25),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=7,
            color=t["muted"],
            rotation=40,
        )
        for k, (name, r) in enumerate(series.items()):
            # Same entity drawn twice (e.g. before/after calibration): same colour, line style
            # tells them apart.
            color = entity_color(t, r.get("entity", name), k)
            conf = np.asarray(r["confidence"])
            freq = np.asarray(r["frequency"])
            keep = (conf >= lo) & (freq >= lo)  # log axes: drop bins below the plotted range
            conf, freq = conf[keep], freq[keep]
            ax.plot(
                conf,
                freq,
                color=color,
                linewidth=2,
                linestyle=(0, (4, 2)) if r.get("dashed") else "-",
                marker="o" if not r.get("dashed") else "s",
                markersize=5,
                markeredgecolor=t["surface"],
                markeredgewidth=1.2,
                label=f"{name} (ECE {r['ece']:.3f})",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlim=(lo, 1), ylim=(lo, 1))
        ax.grid(axis="x", color=t["grid"], linewidth=0.8)
        ax.set_xlabel("predicted failure probability", color=t["ink2"], fontsize=9)
        ax.set_ylabel("observed failure frequency", color=t["ink2"], fontsize=9)
        ax.set_title(title, color=t["ink"], fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=7, labelcolor=t["ink2"], loc="upper left")
        fig.tight_layout()
        fig.savefig(out.with_name(f"{out.stem}_{mode}.png"), facecolor=t["surface"])
        plt.close(fig)


def histogram(
    values: list[float],
    out: Path,
    title: str,
    xlabel: str,
    entity: str,
    bins: np.ndarray,
    caption: str = "",
) -> None:
    for mode, t in THEMES.items():
        fig, (ax,) = _figure(t, 1, 5.6, 3.2)
        ax.hist(
            values,
            bins=bins,
            color=entity_color(t, entity, 4),
            edgecolor=t["surface"],
            linewidth=1.5,
        )
        med = float(np.median(values)) if len(values) else float("nan")
        ax.axvline(med, color=t["ink2"], linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(
            f"median {med:.1f} s",
            (med, ax.get_ylim()[1]),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
            color=t["ink2"],
        )
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9)
        ax.set_ylabel("detected failure events", color=t["ink2"], fontsize=9)
        ax.set_title(title, color=t["ink"], fontsize=10, loc="left")
        if caption:
            fig.text(0.02, 0.01, caption, color=t["muted"], fontsize=7)
        fig.tight_layout(rect=(0, 0.05 if caption else 0, 1, 1))
        fig.savefig(out.with_name(f"{out.stem}_{mode}.png"), facecolor=t["surface"])
        plt.close(fig)


def line_family(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    out: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    reference: tuple[str, np.ndarray, np.ndarray] | None = None,
    caption: str = "",
    markers: bool = True,
) -> None:
    """Several x/y series (entity colours) with an optional dashed reference series."""
    for mode, t in THEMES.items():
        fig, (ax,) = _figure(t, 1, 5.6, 3.6)
        if reference is not None:
            name, rx, ry = reference
            ax.plot(rx, ry, color=t["muted"], linewidth=1.5, linestyle=(0, (4, 3)), label=name)
        for k, (name, (x, y)) in enumerate(series.items()):
            ax.plot(
                x,
                y,
                color=entity_color(t, name, k),
                linewidth=2,
                label=name,
                marker="o" if markers else None,
                markersize=5,
                markeredgecolor=t["surface"],
                markeredgewidth=1.2,
            )
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9)
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9)
        ax.set_title(title, color=t["ink"], fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=7, labelcolor=t["ink2"])
        if caption:
            fig.text(0.02, 0.01, caption, color=t["muted"], fontsize=7)
        fig.tight_layout(rect=(0, 0.05 if caption else 0, 1, 1))
        fig.savefig(out.with_name(f"{out.stem}_{mode}.png"), facecolor=t["surface"])
        plt.close(fig)


def version_bars(
    panels: list[tuple[str, str]],
    models: list[str],
    values: dict[str, dict[str, dict[str, list[float]]]],
    out: Path,
    title: str,
    caption: str = "",
) -> None:
    """Small multiples: one panel per metric, groups = models, two bars per group (v1, v2).

    values[version][model][metric] -> per-seed list. v1 is drawn in muted grey (the reference),
    v2 in the first categorical slot; dots are individual seeds.
    """
    versions = ("v1", "v2")
    for mode, t in THEMES.items():
        fig, axes = _figure(t, len(panels), 3.3 * len(panels), 3.5)
        colours = {"v1": t["muted"], "v2": t["series"][0]}
        for ax, (metric, label) in zip(axes, panels, strict=True):
            peak = 0.0
            for i, model in enumerate(models):
                for j, v in enumerate(versions):
                    vals = np.asarray(values[v][model][metric], dtype=float)
                    x = i + (j - 0.5) * 0.38
                    mean = float(vals.mean())
                    ax.bar(
                        x,
                        mean,
                        width=0.36,
                        color=colours[v],
                        edgecolor=t["surface"],
                        linewidth=1.5,
                        label=v if i == 0 else None,
                    )
                    if len(vals) > 1:
                        ax.plot(
                            x + np.linspace(-0.07, 0.07, len(vals)),
                            vals,
                            "o",
                            color=t["ink"],
                            markersize=3,
                            alpha=0.8,
                        )
                    ax.annotate(
                        f"{mean:.1f}" if mean >= 10 else f"{mean:.2f}",
                        (x, max(mean, float(vals.max()))),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                        color=t["ink2"],
                    )
                    peak = max(peak, float(vals.max()))
            ax.set_xticks(range(len(models)), [_wrap(m, 9) for m in models], fontsize=8)
            ax.set_ylim(0, 1.2 * peak if peak > 0 else 1.0)
            ax.set_title(label, color=t["ink"], fontsize=10, loc="left")
        axes[0].legend(frameon=False, fontsize=8, labelcolor=t["ink2"], loc="upper left")
        fig.suptitle(title, color=t["ink"], fontsize=11, x=0.02, ha="left")
        if caption:
            fig.text(0.02, 0.01, caption, color=t["muted"], fontsize=7)
        fig.tight_layout(rect=(0, 0.04 if caption else 0, 1, 0.94))
        fig.savefig(out.with_name(f"{out.stem}_{mode}.png"), facecolor=t["surface"])
        plt.close(fig)
