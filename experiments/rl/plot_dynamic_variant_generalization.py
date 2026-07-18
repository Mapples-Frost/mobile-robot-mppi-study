#!/usr/bin/env python3
"""Create the publication figure for the L31 development factorial."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


METHODS = (
    "traditional_icode",
    "lcb_icode",
    "gated_lcb_icode",
    "temporal_gated_lcb_nominal",
    "temporal_gated_lcb_icode",
)
METHOD_LABELS = (
    "MPPI + ICODE",
    "RL prior + ICODE",
    "Spatial gate + ICODE",
    "Temporal gate + nominal",
    "Temporal gate + ICODE (ours)",
)
SCENES = (
    "anchor_standard",
    "reverse_standard",
    "fast_crossing",
    "large_early",
)
SCENE_LABELS = ("Anchor", "Reverse", "Fast", "Large")
PHYSICS = ("seen", "unseen")


def _read(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _matrix(rows, field):
    lookup = {
        (row["condition"], row["scene_role"], row["physics_role"]): row
        for row in rows
    }
    values = []
    for method in METHODS:
        values.append([
            float(lookup[(method, scene, physics)][field])
            for scene in SCENES for physics in PHYSICS
        ])
    return np.asarray(values, dtype=np.float64)


def _annotate(ax, values, fmt, threshold):
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text_color = "white" if value >= threshold else "#222222"
            ax.text(
                column, row, fmt.format(value), ha="center", va="center",
                fontsize=7.2, color=text_color, fontweight="medium",
            )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", required=True)
    args = parser.parse_args(argv)
    directory = Path(args.summary_dir).resolve()
    rows = _read(directory / "condition_summary.csv")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    success = 100.0 * _matrix(rows, "success_rate")
    collisions = _matrix(rows, "collisions")
    distance = _matrix(rows, "final_goal_distance_mean")
    panels = (
        (success, "(a) Success rate (%)", "YlGn", 0.0, 100.0, "{:.0f}", 62.0),
        (collisions, "(b) Collision count", "OrRd", 0.0, max(1.0, collisions.max()), "{:.0f}", max(1.0, collisions.max()) * 0.55),
        (distance, "(c) Final goal distance (m)", "Blues", 0.0, max(0.5, distance.max()), "{:.2f}", max(0.5, distance.max()) * 0.58),
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.7), constrained_layout=True)
    xlabels = [
        "%s\n%s" % (scene, "Seen" if physics == "seen" else "Unseen")
        for scene in SCENE_LABELS for physics in PHYSICS
    ]
    for index, (ax, panel) in enumerate(zip(axes, panels)):
        values, title, cmap, vmin, vmax, fmt, threshold = panel
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        _annotate(ax, values, fmt, threshold)
        ax.set_title(title, loc="left", pad=5)
        ax.set_xticks(np.arange(len(xlabels)), xlabels)
        ax.set_yticks(np.arange(len(METHOD_LABELS)), METHOD_LABELS)
        ax.tick_params(length=0)
        for column in (1.5, 3.5, 5.5):
            ax.axvline(column, color="white", linewidth=2.0)
        ax.add_patch(Rectangle(
            (-0.49, len(METHODS) - 1.49), len(xlabels) - 0.02, 0.98,
            fill=False, edgecolor="#D55E00", linewidth=1.6,
        ))
        colorbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.012)
        colorbar.ax.tick_params(labelsize=7, width=0.6, length=2)
        if index < 2:
            ax.set_xticklabels([])
    fig.suptitle(
        "L31 dynamic-obstacle development set: 3 model blocks × 5 episode seeds",
        fontsize=9.5, fontweight="bold",
    )
    pdf = directory / "fig_l31_dynamic_variant_generalization.pdf"
    png = directory / "fig_l31_dynamic_variant_generalization.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(pdf)
    print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
