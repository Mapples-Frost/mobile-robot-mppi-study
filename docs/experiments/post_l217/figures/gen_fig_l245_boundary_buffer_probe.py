#!/usr/bin/env python3
"""Plot the L245 buffer probe against the frozen L242/L244 controls."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
L244_TABLE = HERE.parent / "tables" / "l244_tracking_gate_summary.csv"
L245_TABLE = HERE.parent / "tables" / "l245_boundary_buffer_probe_summary.csv"
SCENES = ("hairpin", "s_chicane", "infinity")
SCENE_LABELS = ("Hairpin", "S-Chicane", "Infinity")
METHODS = (
    "L242 buffer=0.05",
    "L244 buffer=0.05",
    "L245 buffer=0.075",
)
COLORS = ("#8C8C8C", "#D55E00", "#009E73")


def _read(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _series(rows244, rows245, method, field):
    if method == METHODS[2]:
        indexed = {row["scene"]: row for row in rows245}
    else:
        source = (
            "L242 selected-initial Full"
            if method == METHODS[0]
            else "L244 BC-anchored Full"
        )
        indexed = {
            row["scene"]: row for row in rows244 if row["method"] == source
        }
    return np.asarray([float(indexed[scene][field]) for scene in SCENES])


def main():
    rows244 = _read(L244_TABLE)
    rows245 = _read(L245_TABLE)
    if len(rows245) != 3 or {row["scene"] for row in rows245} != set(SCENES):
        raise ValueError("L245 summary must contain one row per scene")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.65))
    x = np.arange(len(SCENES))
    width = 0.24
    for index, (method, color) in enumerate(zip(METHODS, COLORS)):
        completion = _series(
            rows244, rows245, method, "path_completion_ratio"
        )
        bars = axes[0].bar(
            x + (index - 1) * width,
            completion,
            width * 0.9,
            label=method,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, completion):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.7,
                rotation=90,
            )
        boundary = _series(
            rows244, rows245, method, "boundary_violation_steps"
        )
        bars_b = axes[1].bar(
            x + (index - 1) * width,
            boundary,
            width * 0.9,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars_b, boundary):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.035,
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SCENE_LABELS)
    axes[0].set_ylabel("Path completion ratio")
    axes[0].set_ylim(0.0, 0.39)
    axes[0].set_title("Progress is not preserved")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(SCENE_LABELS, rotation=20)
    axes[1].set_ylabel("Boundary-violation steps")
    axes[1].set_ylim(0.0, 1.28)
    axes[1].set_yticks([0, 1])
    axes[1].set_title("Violations remain")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        columnspacing=1.25,
    )
    fig.suptitle(
        "L245 single-variable development probe | seed 923301001 | K=100",
        fontsize=10.5,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout(w_pad=1.8)
    fig.savefig(HERE / "fig_l245_boundary_buffer_probe.pdf")
    fig.savefig(HERE / "fig_l245_boundary_buffer_probe.png", dpi=300)


if __name__ == "__main__":
    main()
