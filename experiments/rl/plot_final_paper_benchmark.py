#!/usr/bin/env python3
"""Generate reproducible publication figures for the L214 benchmark."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("traditional_mppi", "Traditional"),
    ("icode_mppi", "ICODE"),
    ("rl_driven_mppi", "RL-driven"),
    ("simple_combination", "Simple I+RL"),
    ("value_fixed", "Value-aligned"),
    ("ordinary_adaptive", "Adaptive HSS"),
    ("full_proposed", "Full proposed"),
)
DOMAINS = (
    ("nominal_seen", "Nominal"),
    ("long_delay_seen", "Long delay"),
    ("combined_unseen", "Combined unseen"),
)
COLORS = {
    "traditional_mppi": "#7F7F7F",
    "icode_mppi": "#0072B2",
    "rl_driven_mppi": "#E69F00",
    "simple_combination": "#56B4E9",
    "value_fixed": "#CC79A7",
    "ordinary_adaptive": "#009E73",
    "full_proposed": "#D55E00",
}


def _rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value):
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return float(value.lower() == "true")
    return float(value)


def _mean(rows, arm, domain, metric):
    selected = [
        _float(row[metric])
        for row in rows
        if row["benchmark_arm"] == arm
        and (domain == "all" or row["physics_domain"] == domain)
    ]
    if not selected:
        raise ValueError("empty method/domain cell")
    return float(np.mean(selected))


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linestyle": "-",
    })


def _annotated_heatmap(ax, values, title, fmt, cmap, vmin=None, vmax=None):
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels([label for _, label in DOMAINS], rotation=20, ha="right")
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([label for _, label in METHODS])
    ax.set_title(title)
    midpoint = 0.5 * (float(np.nanmin(values)) + float(np.nanmax(values)))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > midpoint else "#222222"
            ax.text(
                j,
                i,
                format(values[i, j], fmt),
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    return image


def plot_method_overview(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    method_ids = [method for method, _ in METHODS]
    labels = [label for _, label in METHODS]
    colors = [COLORS[method] for method in method_ids]
    y = np.arange(len(METHODS))
    success = np.asarray([
        _mean(rows, method, "all", "success") for method in method_ids
    ])
    distance = np.asarray([
        _mean(rows, method, "all", "final_goal_distance")
        for method in method_ids
    ])
    axes[0, 0].barh(y, 100.0 * success, color=colors, height=0.64)
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels(labels)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Success rate (%)")
    axes[0, 0].set_title("(a) Overall success")
    axes[0, 0].set_xlim(0, 75)
    for index, value in enumerate(success):
        axes[0, 0].text(100.0 * value + 1, index, "%.1f" % (100.0 * value), va="center", fontsize=7)

    axes[0, 1].barh(y, distance, color=colors, height=0.64)
    axes[0, 1].set_yticks(y)
    axes[0, 1].set_yticklabels(labels)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Final goal distance (m, lower is better)")
    axes[0, 1].set_title("(b) Overall final distance")
    for index, value in enumerate(distance):
        axes[0, 1].text(value + 0.035, index, "%.2f" % value, va="center", fontsize=7)

    success_grid = np.asarray([
        [100.0 * _mean(rows, method, domain, "success") for domain, _ in DOMAINS]
        for method, _ in METHODS
    ])
    distance_grid = np.asarray([
        [_mean(rows, method, domain, "final_goal_distance") for domain, _ in DOMAINS]
        for method, _ in METHODS
    ])
    image = _annotated_heatmap(
        axes[1, 0], success_grid, "(c) Success by physics domain", ".0f", "YlGn", 0, 100
    )
    fig.colorbar(image, ax=axes[1, 0], fraction=0.04, pad=0.03, label="Success (%)")
    image = _annotated_heatmap(
        axes[1, 1], distance_grid, "(d) Final distance by physics domain", ".2f", "YlOrRd"
    )
    fig.colorbar(image, ax=axes[1, 1], fraction=0.04, pad=0.03, label="Distance (m)")
    fig.suptitle("L214 Seven-arm Point-goal Benchmark (10 seeds, 3 physics domains)", y=1.01, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "fig_l214_method_overview.pdf")
    fig.savefig(output / "fig_l214_method_overview.png", dpi=300)
    plt.close(fig)


def _effect(paired, comparison, metric):
    item = paired[comparison]["metrics"][metric]
    return float(item["favorable_effect"]), [float(value) for value in item["ci95"]]


def plot_core_ablation(paired, output):
    comparisons = (
        ("value_at_fixed", "Value alignment"),
        ("hss_at_ordinary", "Adaptive HSS"),
        ("full_vs_simple", "Full vs. simple"),
    )
    panels = (
        ("success", "Success-rate improvement", "percentage points", 100.0),
        ("final_goal_distance", "Final-distance improvement", "m", 1.0),
        ("control_jerk", "Control-jerk improvement", "lower-is-better units", 1.0),
        ("planner_compute_ms_mean", "Planner-time improvement", "ms", 1.0),
    )
    colors = ["#CC79A7", "#009E73", "#D55E00"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    y = np.arange(len(comparisons))
    for ax, (metric, title, unit, scale) in zip(axes.flat, panels):
        effects = []
        low = []
        high = []
        for comparison, _ in comparisons:
            effect, ci = _effect(paired, comparison, metric)
            effects.append(scale * effect)
            low.append(scale * (effect - ci[0]))
            high.append(scale * (ci[1] - effect))
        for index, color in enumerate(colors):
            ax.errorbar(
                effects[index],
                y[index],
                xerr=np.asarray([[low[index]], [high[index]]]),
                fmt="o",
                color=color,
                ecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=6,
                elinewidth=2,
                capsize=3,
                zorder=3,
            )
        ax.axvline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels([label for _, label in comparisons])
        ax.invert_yaxis()
        ax.set_xlabel("Favorable change (%s)" % unit)
        ax.set_title(title)
    fig.suptitle("Frozen Coupling Ablation: Seed-cluster 95% Bootstrap CIs", y=1.01, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "fig_l214_core_ablation.pdf")
    fig.savefig(output / "fig_l214_core_ablation.png", dpi=300)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args(argv)
    result_dir = Path(args.result_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else result_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    rows = _rows(result_dir / "progress.csv")
    paired = json.loads((result_dir / "paired_comparisons.json").read_text(encoding="utf-8"))
    _style()
    plot_method_overview(rows, output)
    plot_core_ablation(paired, output)
    print(json.dumps({"output_dir": str(output), "figures": 4}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
