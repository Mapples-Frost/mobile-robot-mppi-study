#!/usr/bin/env python3
"""Generate the publication forest plot for the L108 confirmation."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("cross_track_rmse", "Cross-track RMSE", 1000.0, "Delta RMSE (mm)"),
    ("elapsed_s", "Time to goal", 1.0, "Delta time (s)"),
    ("control_jerk", "Issued jerk", 1.0, "Delta jerk"),
    ("applied_control_jerk", "Applied jerk", 1.0, "Delta jerk"),
    (
        "planner_compute_ms_mean", "Planner compute", 1.0,
        "Delta compute (ms/step)",
    ),
)

COLORS = {
    "Proposed vs nominal K100": "#0072B2",
    "Contextual RL vs ICODE K100": "#D55E00",
}


def _effect(contrast, metric, scale):
    mean = float(contrast[metric + "_delta_mean"]) * scale
    low, high = contrast[metric + "_delta_ci95"]
    return mean, float(low) * scale, float(high) * scale


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    with Path(args.summary).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    contrasts = summary["contrasts"]
    series = {
        "Proposed vs nominal K100": contrasts["proposed_vs_nominal"],
        "Contextual RL vs ICODE K100": contrasts["proposed_vs_icode_k100"],
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
    })
    fig, axes = plt.subplots(1, len(METRICS), figsize=(7.25, 2.35))
    labels = list(series)
    y = np.asarray((1.0, 0.0))
    for axis_index, (axis, (metric, title, scale, xlabel)) in enumerate(
        zip(axes, METRICS)
    ):
        for position, label in zip(y, labels):
            mean, low, high = _effect(series[label], metric, scale)
            axis.errorbar(
                mean,
                position,
                xerr=np.asarray([[mean - low], [high - mean]]),
                fmt="o",
                color=COLORS[label],
                ecolor=COLORS[label],
                capsize=2.5,
                markersize=4.3,
                linewidth=1.25,
                zorder=3,
            )
        axis.axvline(0.0, color="#2E3440", linewidth=0.85, linestyle="--")
        axis.set_title(title, pad=7)
        axis.set_xlabel(xlabel, fontsize=7.3)
        axis.set_yticks(y)
        axis.set_yticklabels(
            ("vs nominal", "vs ICODE") if axis_index == 0 else ()
        )
        axis.set_ylim(-0.65, 1.65)
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
    handles = [
        plt.Line2D(
            (0,), (0,), marker="o", linestyle="none", color=COLORS[label],
            markersize=5, label=label,
        )
        for label in labels
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
    )
    fig.text(
        0.5,
        -0.01,
        "Points are paired mean differences; bars are hierarchical-bootstrap 95% CIs. "
        "Lower is better for all metrics.",
        ha="center",
        fontsize=7.5,
    )
    fig.subplots_adjust(
        left=0.085, right=0.995, top=0.76, bottom=0.28, wspace=0.52
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "fig_l108_final_package_effects"
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".png", dpi=300)
    plt.close(fig)
    print(str(stem) + ".pdf")
    print(str(stem) + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
