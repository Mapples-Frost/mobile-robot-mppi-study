#!/usr/bin/env python3
"""Create the publication-style L29/L30 development interaction figure."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


METHODS = (
    ("traditional_nominal", "Traditional\nnominal"),
    ("traditional_icode", "Traditional\n+ ICODE"),
    ("lcb_nominal", "Always-RL\nnominal"),
    ("lcb_icode", "Always-RL\n+ ICODE"),
    ("gated_lcb_nominal", "Spatial gate\nnominal"),
    ("gated_lcb_icode", "Spatial gate\n+ ICODE"),
    ("temporal_gated_lcb_nominal", "Temporal gate\nnominal"),
    ("temporal_gated_lcb_icode", "Temporal gate\n+ ICODE (proposed)"),
)
STRATA = (
    ("clean", "seen", "C-S"),
    ("clean", "unseen", "C-U"),
    ("static_blocking", "seen", "S-S"),
    ("static_blocking", "unseen", "S-U"),
    ("dynamic_crossing", "seen", "D-S"),
    ("dynamic_crossing", "unseen", "D-U"),
)


def _read(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _matrix(rows, field):
    lookup = {
        (row["condition"], row["scene_role"], row["physics_role"]): row
        for row in rows
    }
    output = np.zeros((len(METHODS), len(STRATA)), dtype=np.float64)
    for row_index, (condition, _) in enumerate(METHODS):
        for column_index, (scene, physics, _) in enumerate(STRATA):
            output[row_index, column_index] = float(
                lookup[(condition, scene, physics)][field]
            )
    return output


def _annotate(ax, matrix, formatter, threshold=None):
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if threshold is not None and value >= threshold else "#202124"
            ax.text(
                column,
                row,
                formatter(value),
                ha="center",
                va="center",
                fontsize=7.1,
                color=color,
                fontweight="bold" if row == len(METHODS) - 1 else "normal",
            )


def _style_panel(ax, title, show_ylabels):
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.set_xticks(np.arange(len(STRATA)))
    ax.set_xticklabels([label for _, _, label in STRATA], fontsize=7.2)
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels(
        [label if show_ylabels else "" for _, label in METHODS],
        fontsize=7.4,
    )
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(STRATA), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.add_patch(Rectangle(
        (-0.49, len(METHODS) - 1.49),
        len(STRATA) - 0.02,
        0.98,
        fill=False,
        edgecolor="#D55E00",
        linewidth=2.0,
        clip_on=False,
    ))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args(argv)

    rows = _read(args.summary_csv)
    success = 100.0 * _matrix(rows, "success_rate")
    collisions = _matrix(rows, "collisions")
    distance = _matrix(rows, "final_goal_distance_mean")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(7.15, 3.85), constrained_layout=False
    )
    fig.subplots_adjust(left=0.185, right=0.985, bottom=0.17, top=0.90, wspace=0.17)

    image = axes[0].imshow(success, cmap="Blues", vmin=0.0, vmax=100.0, aspect="auto")
    _annotate(axes[0], success, lambda value: "%d/15" % round(0.15 * value), threshold=60.0)
    _style_panel(axes[0], "(a) Success", True)

    axes[1].imshow(collisions, cmap="OrRd", vmin=0.0, vmax=15.0, aspect="auto")
    _annotate(axes[1], collisions, lambda value: "%d" % round(value), threshold=8.0)
    _style_panel(axes[1], "(b) Collisions", False)

    axes[2].imshow(distance, cmap="viridis_r", vmin=0.0, vmax=3.0, aspect="auto")
    _annotate(axes[2], distance, lambda value: "%.2f" % value, threshold=1.65)
    _style_panel(axes[2], "(c) Final distance (m)", False)

    for axis in axes:
        axis.axvline(1.5, color="#202124", linewidth=0.8)
        axis.axvline(3.5, color="#202124", linewidth=0.8)
    fig.text(
        0.585,
        0.035,
        "C/S/D: clean/static/dynamic; S/U: seen/unseen physics. Development set: 3 model blocks x 5 matched seeds per cell; orange: proposed.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#333333",
    )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(prefix) + ".pdf")
    fig.savefig(str(prefix) + ".png", dpi=300)
    plt.close(fig)
    print(str(prefix) + ".pdf")
    print(str(prefix) + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
