#!/usr/bin/env python3
"""Generate the reproducible L244 single-seed development Gate figure."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
TABLE = HERE.parent / "tables" / "l244_tracking_gate_summary.csv"

SCENES = ("hairpin", "s_chicane", "infinity")
METHODS = (
    "L242 selected-initial Full",
    "L244 BC-anchored Full",
    "L244 ICODE-MPPI",
)
LABELS = {
    "hairpin": "Hairpin",
    "s_chicane": "S-Chicane",
    "infinity": "Infinity",
}
COLORS = {
    "L242 selected-initial Full": "#8C8C8C",
    "L244 BC-anchored Full": "#D55E00",
    "L244 ICODE-MPPI": "#0072B2",
}


def _values(rows, method, field):
    indexed = {
        row["scene"]: row for row in rows if row["method"] == method
    }
    return np.asarray([float(indexed[scene][field]) for scene in SCENES])


def main():
    with TABLE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(scene, method) for scene in SCENES for method in METHODS}
    observed = {(row["scene"], row["method"]) for row in rows}
    if observed != expected:
        raise ValueError("L244 figure table is incomplete or duplicated")

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

    fig, axes = plt.subplots(
        1, 2, figsize=(6.75, 2.65), gridspec_kw={"width_ratios": [1.75, 1.0]}
    )
    x = np.arange(len(SCENES))
    width = 0.24
    for index, method in enumerate(METHODS):
        values = _values(rows, method, "path_completion_ratio")
        offset = (index - 1) * width
        bars = axes[0].bar(
            x + offset,
            values,
            width * 0.9,
            label=method,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.7,
                rotation=90,
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LABELS[item] for item in SCENES])
    axes[0].set_ylabel("Path completion ratio")
    axes[0].set_ylim(0.0, 0.39)
    axes[0].set_title("Closed-loop progress")

    boundary_methods = METHODS[:2]
    width_b = 0.34
    for index, method in enumerate(boundary_methods):
        values = _values(rows, method, "boundary_violation_steps")
        offset = (index - 0.5) * width_b
        bars = axes[1].bar(
            x + offset,
            values,
            width_b * 0.9,
            label=method,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.035,
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABELS[item] for item in SCENES], rotation=20)
    axes[1].set_ylabel("Boundary-violation steps")
    axes[1].set_ylim(0.0, 1.28)
    axes[1].set_yticks([0, 1])
    axes[1].set_title("Strict Gate failure")

    fig.suptitle(
        "L244 development seed 923301001 | MuJoCo 3.2.3 | K=100",
        fontsize=10.5,
        fontweight="bold",
        y=1.04,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        columnspacing=1.2,
    )
    fig.tight_layout(w_pad=1.6)
    fig.savefig(HERE / "fig_l244_tracking_gate.pdf")
    fig.savefig(HERE / "fig_l244_tracking_gate.png", dpi=300)


if __name__ == "__main__":
    main()
