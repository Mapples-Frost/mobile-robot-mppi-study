#!/usr/bin/env python3
"""Publication-style L32 development figure from audited CSV artifacts."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = (
    "traditional_icode_no_temporal_safety",
    "traditional_icode_temporal_safety",
    "legacy_temporal_gated_lcb_icode",
    "robust_temporal_gated_lcb_icode",
    "robust_temporal_gated_lcb_icode_safety",
    "competence_gated_lcb_icode_safety",
)
LABELS = (
    "Trad. ICODE, no temporal safety",
    "Trad. ICODE + temporal safety",
    "Legacy temporal RL + ICODE",
    "Robust temporal RL + ICODE",
    "Robust RL + ICODE + temporal safety",
    "Competence-gated RL + ICODE + safety",
)
COMPARISON_LABELS = {
    "shared_safety_traditional": "Temporal safety\n(traditional)",
    "robust_estimator_without_safety": "Robust vs legacy\nestimator",
    "shared_safety_with_robust_gate": "Temporal safety\n(robust RL)",
    "competence_separation": "Competence\nseparation",
}


def _read(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    root = Path(args.run_dir).resolve()
    episodes = _read(root / "episodes.csv")
    paired = _read(root / "paired_summary.csv")

    grouped = defaultdict(list)
    for row in episodes:
        grouped[row["condition"]].append(row)
    success = np.asarray([
        100.0 * np.mean([row["success"] == "True" for row in grouped[name]])
        for name in ORDER
    ])
    collision = np.asarray([
        100.0 * np.mean([row["collision"] == "True" for row in grouped[name]])
        for name in ORDER
    ])

    strata = (
        ("anchor_standard", "seen"),
        ("anchor_standard", "unseen"),
        ("reverse_standard", "seen"),
        ("reverse_standard", "unseen"),
        ("fast_crossing", "seen"),
        ("fast_crossing", "unseen"),
        ("large_early", "seen"),
        ("large_early", "unseen"),
    )
    stratum_labels = (
        "Anchor / seen", "Anchor / unseen",
        "Reverse / seen", "Reverse / unseen",
        "Fast / seen", "Fast / unseen",
        "Large-early / seen", "Large-early / unseen",
    )
    difference = []
    for scene, physics in strata:
        selected = [
            row for row in episodes
            if row["scene_role"] == scene and row["physics_role"] == physics
        ]
        competence = np.mean([
            row["success"] == "True" for row in selected
            if row["condition"] == "competence_gated_lcb_icode_safety"
        ])
        traditional = np.mean([
            row["success"] == "True" for row in selected
            if row["condition"] == "traditional_icode_temporal_safety"
        ])
        difference.append(100.0 * (competence - traditional))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
    })
    colors = {
        "blue": "#0072B2",
        "orange": "#D55E00",
        "teal": "#009E73",
        "gray": "#9AA0A6",
        "light": "#D9DEE2",
    }
    fig = plt.figure(figsize=(7.05, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.4, 1.0))
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    y_methods = np.arange(len(ORDER))
    height = 0.34
    bars_s = ax_a.barh(
        y_methods - height / 2, success, height,
        color=colors["blue"], label="Success", edgecolor="white",
    )
    bars_c = ax_a.barh(
        y_methods + height / 2, collision, height,
        color=colors["orange"], label="Collision", edgecolor="white",
    )
    for bars in (bars_s, bars_c):
        for bar in bars:
            ax_a.text(
                bar.get_width() + 1.0,
                bar.get_y() + bar.get_height() / 2,
                "%.0f" % bar.get_width(),
                ha="left", va="center", fontsize=7,
            )
    ax_a.set_yticks(y_methods)
    ax_a.set_yticklabels(LABELS)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Episode rate (%)")
    ax_a.set_xlim(0.0, 100.0)
    ax_a.set_title("A  Overall closed-loop outcomes (n = 120 per method)", loc="left")
    ax_a.legend(ncol=2, loc="upper right")

    matrix = np.asarray(difference, dtype=np.float64)[:, None]
    image = ax_b.imshow(matrix, cmap="RdBu", vmin=-100, vmax=100, aspect="auto")
    for row, value in enumerate(difference):
        ax_b.text(
            0, row, "%+.0f pp" % value,
            ha="center", va="center",
            color="white" if abs(value) >= 45 else "#222222",
            fontsize=7.5, fontweight="bold",
        )
    ax_b.set_xticks([0])
    ax_b.set_xticklabels(["Competence gate − traditional safety"])
    ax_b.set_yticks(np.arange(len(strata)))
    ax_b.set_yticklabels(stratum_labels)
    ax_b.set_title("B  Success difference by stratum", loc="left")
    ax_b.grid(False)
    colorbar = fig.colorbar(image, ax=ax_b, fraction=0.045, pad=0.04)
    colorbar.set_label("Percentage points")

    y = np.arange(len(paired))
    means = 100.0 * np.asarray([
        float(row["success_difference_mean"]) for row in paired
    ])
    lows = 100.0 * np.asarray([
        float(row["success_difference_ci95_low"]) for row in paired
    ])
    highs = 100.0 * np.asarray([
        float(row["success_difference_ci95_high"]) for row in paired
    ])
    ax_c.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax_c.errorbar(
        means,
        y,
        xerr=np.vstack((means - lows, highs - means)),
        fmt="o",
        color=colors["teal"],
        ecolor=colors["teal"],
        capsize=3,
        markersize=5,
    )
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([
        COMPARISON_LABELS[row["comparison"]] for row in paired
    ])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Paired success difference (percentage points)")
    ax_c.set_title("C  Nested bootstrap mean and 95% interval", loc="left")

    fig.suptitle(
        "L32 development set: robust temporal perception improves validity, "
        "but reactive safety is not sufficient",
        fontsize=10.5,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5, 0.02,
        "720 MuJoCo episodes; 3 RL/ICODE model blocks × 4 motion variants × "
        "2 physics domains × 5 seeds. Development-only; sealed seeds unused.",
        ha="center", va="top", fontsize=7.3,
    )
    fig.subplots_adjust(
        left=0.27,
        right=0.97,
        top=0.91,
        bottom=0.12,
        hspace=0.48,
        wspace=0.78,
    )
    fig.savefig(root / "fig_l32_temporal_safety_remediation.pdf")
    fig.savefig(root / "fig_l32_temporal_safety_remediation.png", dpi=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
