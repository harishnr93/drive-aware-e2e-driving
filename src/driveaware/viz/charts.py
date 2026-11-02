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
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}


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
            color = t["series"][k]
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
) -> None:
    """agents: label -> {"completion": [per-seed], "completion_ci": (lo, hi) | None,
    "collision": [per-seed], "collision_ci": ...}. Values are fractions (0-1)."""
    panels = (
        ("completion", "Route completion (%)"),
        ("collision", "Episodes with a collision (%)"),
    )
    labels = list(agents)
    for mode, t in THEMES.items():
        fig, axes = _figure(t, 2, 7.2, 3.3)
        for ax, (key, ylabel) in zip(axes, panels, strict=True):
            for i, label in enumerate(labels):
                vals = 100 * np.asarray(agents[label][key], dtype=float)
                color = t["series"][i]
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
                        yerr=[[mean - 100 * ci[0]], [100 * ci[1] - mean]],
                        color=t["ink2"],
                        capsize=4,
                        linewidth=1.2,
                    )
                    top = max(top, 100 * ci[1])
                ax.annotate(
                    f"{mean:.1f}",
                    (i, top),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=t["ink"],
                )
            ax.set_xticks(range(len(labels)), labels, fontsize=9)
            ax.set_ylim(
                0,
                112
                if key == "completion"
                else max(5.0, 1.3 * max(100 * max(agents[lb][key]) for lb in labels)),
            )
            ax.set_title(ylabel, color=t["ink"], fontsize=10, loc="left")
        fig.suptitle(title, color=t["ink"], fontsize=11, x=0.02, ha="left")
        fig.text(0.02, 0.01, caption, color=t["muted"], fontsize=7)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        fig.savefig(out.with_name(f"{out.stem}_{mode}.png"), facecolor=t["surface"])
        plt.close(fig)
