#!/usr/bin/env python3
"""Publication-ready L94 sample-efficiency figure from saved artifacts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _paired_primary(rows):
    indexed = {
        (
            row["scene"],
            row["physics_domain"],
            int(row["seed"]),
            int(row["num_samples"]),
            row["condition"],
        ): row
        for row in rows
    }
    pairs = []
    for scene, domain, seed in sorted({key[:3] for key in indexed}):
        learned = indexed[(scene, domain, seed, 50, "learned_contextual_bandit")]
        fixed = indexed[(scene, domain, seed, 100, "strongest_global_fixed")]
        pairs.append((learned, fixed))
    return pairs


def _difference_panel(ax, values, ylabel, color, letter, mean_ci=None):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(20260718)
    x = rng.normal(0.0, 0.035, size=values.size)
    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", zorder=1)
    ax.scatter(x, values, s=20, alpha=0.58, color=color, edgecolors="none", zorder=2)
    mean = float(np.mean(values))
    if mean_ci is None:
        error = float(np.std(values, ddof=1) / np.sqrt(values.size))
        yerr = error
    else:
        yerr = np.asarray((
            (mean - float(mean_ci[0]),),
            (float(mean_ci[1]) - mean,),
        ))
    ax.errorbar(
        0.0,
        mean,
        yerr=yerr,
        fmt="D",
        color="#111111",
        markerfacecolor="white",
        markersize=5.5,
        linewidth=1.4,
        capsize=3,
        zorder=3,
    )
    ax.set_xlim(-0.18, 0.18)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_title(letter, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.6, alpha=0.7)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--confirmation-dir")
    parser.add_argument("--output-prefix", default="fig_l94_sample_efficiency")
    args = parser.parse_args(argv)
    root = Path(args.input_dir).resolve()
    with (root / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with (root / "episodes.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    confirmation_root = (
        root
        if args.confirmation_dir is None
        else Path(args.confirmation_dir).resolve()
    )
    if args.confirmation_dir is None:
        confirmation_rows = rows
        confirmation_evaluation = summary["primary_sample_efficiency"]
    else:
        with (confirmation_root / "episodes.csv").open(
            "r", encoding="utf-8"
        ) as handle:
            confirmation_rows = list(csv.DictReader(handle))
        with (confirmation_root / "summary.json").open(
            "r", encoding="utf-8"
        ) as handle:
            confirmation_evaluation = json.load(handle)["evaluation"]
    pairs = _paired_primary(confirmation_rows)
    sample_counts = np.asarray((50, 100, 200, 400), dtype=np.int64)
    same = summary["same_budget_comparisons"]
    elapsed_mean = np.asarray([
        same[str(int(k))]["elapsed_s_delta_mean"] for k in sample_counts
    ])
    elapsed_ci = np.asarray([
        same[str(int(k))]["elapsed_s_delta_ci95"] for k in sample_counts
    ])
    rmse_mean = 1000.0 * np.asarray([
        same[str(int(k))]["cross_track_rmse_delta_mean"] for k in sample_counts
    ])
    rmse_ci = 1000.0 * np.asarray([
        same[str(int(k))]["cross_track_rmse_delta_ci95"] for k in sample_counts
    ])
    compute_diff = [
        float(learned["planner_compute_ms_mean"])
        - float(fixed["planner_compute_ms_mean"])
        for learned, fixed in pairs
    ]
    jerk_diff = [
        float(learned["control_jerk"]) - float(fixed["control_jerk"])
        for learned, fixed in pairs
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)
    blue = "#0072B2"
    orange = "#D55E00"

    for ax, mean, ci, ylabel, letter in (
        (axes[0, 0], elapsed_mean, elapsed_ci, "Elapsed-time delta (s)", "A"),
        (axes[0, 1], rmse_mean, rmse_ci, "Cross-track RMSE delta (mm)", "B"),
    ):
        ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
        lower = mean - ci[:, 0]
        upper = ci[:, 1] - mean
        ax.errorbar(
            sample_counts,
            mean,
            yerr=np.vstack((lower, upper)),
            color=blue,
            marker="o",
            linewidth=1.7,
            markersize=4.5,
            capsize=3,
            label="Contextual bandit - fixed",
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sample_counts, labels=[str(value) for value in sample_counts])
        ax.set_xlabel("MPPI samples K (same budget)")
        ax.set_ylabel(ylabel)
        ax.set_title(letter, loc="left", fontweight="bold")
        ax.grid(color="#d8d8d8", linewidth=0.6, alpha=0.7)
    axes[0, 0].legend(frameon=False, loc="best")

    _difference_panel(
        axes[1, 0],
        compute_diff,
        "Planner compute delta (ms/step)",
        blue,
        "C  Half budget: learned K=50 - fixed K=100",
        confirmation_evaluation["planner_compute_ms_mean_delta_ci95"],
    )
    _difference_panel(
        axes[1, 1],
        jerk_diff,
        "Control-jerk delta",
        orange,
        "D  Half-budget trade-off",
        confirmation_evaluation["control_jerk_delta_ci95"],
    )
    fig.suptitle(
        "Route-context exploration preserves performance with half the MPPI samples",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "Frozen ICODE + L89 bandit; held-out hairpin/reverse-S; 4 physics domains; "
        "5 seeds (40 paired contexts). A-B: L94 hierarchical bootstrap 95% CI; "
        "C-D: L95 single-process dots and hierarchical 95% CI.",
        ha="center",
        va="top",
        fontsize=7.3,
    )
    prefix = root / args.output_prefix
    fig.savefig(str(prefix) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(prefix) + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(str(prefix) + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({
        "pairs": len(pairs),
        "png": str(prefix) + ".png",
        "pdf": str(prefix) + ".pdf",
        "svg": str(prefix) + ".svg",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
