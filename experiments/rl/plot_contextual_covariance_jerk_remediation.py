#!/usr/bin/env python3
"""Publication-oriented forest plots for the L96/L97 jerk study."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "raw_contextual_vs_fixed": "#0072B2",
    "smoothed_contextual_vs_raw": "#009E73",
    "smoothed_contextual_vs_fixed": "#D55E00",
}
L97_LABELS = {
    "raw_contextual_vs_fixed": "Raw contextual K50 - fixed K100",
    "smoothed_contextual_vs_raw": "Smoothed K50 - raw K50",
    "smoothed_contextual_vs_fixed": "Smoothed K50 - fixed K100",
}


def _forest(ax, rows, title, xlabel, scale=1.0, zero=True):
    y = np.arange(len(rows), dtype=float)
    for index, row in enumerate(rows):
        mean = float(row["mean"]) * scale
        low, high = [float(value) * scale for value in row["ci"]]
        ax.errorbar(
            mean, y[index], xerr=[[mean - low], [high - mean]],
            fmt="o", color=row["color"], ecolor=row["color"],
            capsize=3.0, markersize=5.5, linewidth=1.5,
        )
    if zero:
        ax.axvline(0.0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_yticks(y, [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.22, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def build_figure(l96, l97):
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.8), constrained_layout=True)
    candidate_colors = {
        "hard_yaw_slew": "#56B4E9",
        "stronger_rate_cost": "#CC79A7",
        "combined": "#009E73",
    }
    l96_rows = []
    for name in ("hard_yaw_slew", "stronger_rate_cost", "combined"):
        result = l96["contrasts_vs_current"][name]
        for metric, suffix, marker_color in (
            ("control_jerk", "issued", candidate_colors[name]),
            ("applied_control_jerk", "applied", "#666666"),
        ):
            l96_rows.append({
                "label": name.replace("_", " ") + " / " + suffix,
                "mean": result[metric + "_delta_mean"],
                "ci": result[metric + "_delta_ci95"],
                "color": marker_color,
            })
    _forest(
        axes[0, 0], l96_rows, "A  L96 selection: jerk vs current",
        "Paired jerk delta (lower is better)",
    )

    contrast_order = (
        "raw_contextual_vs_fixed",
        "smoothed_contextual_vs_raw",
        "smoothed_contextual_vs_fixed",
    )
    panels = (
        (axes[0, 1], "control_jerk", 1.0, "B  L97 issued-command jerk", "Paired jerk delta"),
        (axes[0, 2], "applied_control_jerk", 1.0, "C  L97 physically applied jerk", "Paired jerk delta"),
        (axes[1, 0], "cross_track_rmse", 1000.0, "D  L97 tracking precision", "Paired RMSE delta (mm)"),
        (axes[1, 1], "elapsed_s", 1.0, "E  L97 completion time", "Paired elapsed-time delta (s)"),
        (axes[1, 2], "planner_compute_ms_mean", 1.0, "F  L97 planner compute", "Paired compute delta (ms/step)"),
    )
    for ax, metric, scale, title, xlabel in panels:
        rows = []
        for name in contrast_order:
            result = l97["contrasts"][name]
            rows.append({
                "label": L97_LABELS[name],
                "mean": result[metric + "_delta_mean"],
                "ci": result[metric + "_delta_ci95"],
                "color": COLORS[name],
            })
        _forest(ax, rows, title, xlabel, scale=scale)
    axes[0, 2].annotate(
        "Final-package CI crosses zero\n(pre-registered jerk gate failed)",
        xy=(0.0, 2.0), xytext=(0.48, 0.14), textcoords="axes fraction",
        fontsize=8.5, color="#A33A16",
        arrowprops={"arrowstyle": "->", "color": "#A33A16", "lw": 1.0},
    )
    fig.suptitle(
        "Contextual ICODE-MPPI: half-budget efficiency and jerk remediation",
        fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, -0.018,
        "Points are paired means; bars are hierarchical bootstrap 95% CIs. "
        "L96: 24 blocks / 96 episodes. L97: 48 blocks / 144 episodes, "
        "all successful and collision-free.",
        ha="center", fontsize=8.5,
    )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--l96-summary", required=True)
    parser.add_argument("--l97-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    with Path(args.l96_summary).open("r", encoding="utf-8") as handle:
        l96 = json.load(handle)
    with Path(args.l97_summary).open("r", encoding="utf-8") as handle:
        l97 = json.load(handle)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure = build_figure(l96, l97)
    stem = output / "fig_l96_l97_jerk_remediation"
    figure.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(str(stem) + ".pdf", bbox_inches="tight", facecolor="white")
    figure.savefig(str(stem) + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
