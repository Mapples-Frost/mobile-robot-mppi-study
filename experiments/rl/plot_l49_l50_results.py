#!/usr/bin/env python3
"""Create the reproducible L49/L50 prediction and control evidence figure."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.summarize_icode_path_tracking import _hierarchical_bootstrap


OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "gray": "#8C8C8C",
}


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numeric_effects(path):
    rows = _read_csv(path)
    numeric = (
        "path_length_reduction_m", "control_jerk_reduction",
        "applied_jerk_reduction", "cross_track_improvement_m",
        "relative_cross_track_reduction",
    )
    for row in rows:
        row["model_block"] = int(row["model_block"])
        row["episode_seed"] = int(row["episode_seed"])
        for field in numeric:
            row[field] = float(row[field])
    return rows


def _prediction_rows(summary_path):
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    return payload["rows"]


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot(prediction_summary, effects_csv, output_prefix, bootstrap_seed=2026071622):
    _style()
    prediction = _prediction_rows(prediction_summary)
    effects = _numeric_effects(effects_csv)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), constrained_layout=True)

    # A: offline H=36 improvements, with independent initialisation seeds visible.
    ax = axes[0]
    metrics = (
        ("rollout_relative_improvement", "State"),
        ("position_relative_improvement", "Position"),
        ("heading_relative_improvement", "Heading"),
    )
    splits = (("test", OKABE_ITO["blue"]), ("unseen", OKABE_ITO["orange"]))
    x = np.arange(len(metrics), dtype=float)
    width = 0.32
    for split_index, (split, color) in enumerate(splits):
        offset = (split_index - 0.5) * width
        means = []
        for field, _ in metrics:
            values = [float(row[field]) for row in prediction if row["split"] == split]
            means.append(float(np.mean(values)))
            ax.scatter(
                np.full(len(values), x[len(means)-1] + offset),
                100.0 * np.asarray(values), s=12, facecolor="white",
                edgecolor=color, linewidth=0.8, zorder=4,
            )
        ax.bar(x + offset, 100.0 * np.asarray(means), width * 0.86,
               color=color, alpha=0.78, label=split.capitalize(), zorder=3)
    ax.axhline(0.0, color="#333333", linewidth=0.7)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("H=36 RMSE reduction (%)")
    ax.set_title("A  Held-out prediction")
    ax.legend(loc="upper left")

    # B: preregistered control endpoints, stratified by frozen physics domain.
    ax = axes[1]
    domains = (
        ("combined_matched_delay", "Matched delay", OKABE_ITO["gray"]),
        ("combined_long_delay", "Long delay", OKABE_ITO["green"]),
    )
    jerk_fields = (
        ("control_jerk_reduction", "Issued"),
        ("applied_jerk_reduction", "Applied"),
    )
    x = np.arange(len(jerk_fields), dtype=float)
    width = 0.34
    for domain_index, (domain, label, color) in enumerate(domains):
        subset = [row for row in effects if row["physics_domain"] == domain]
        offset = (domain_index - 0.5) * width
        estimates, lower, upper = [], [], []
        for field_index, (field, _) in enumerate(jerk_fields):
            interval = _hierarchical_bootstrap(
                subset, field, int(bootstrap_seed) + 10 * domain_index + field_index,
                20000,
            )
            estimates.append(interval["estimate"])
            lower.append(interval["estimate"] - interval["ci95_lower"])
            upper.append(interval["ci95_upper"] - interval["estimate"])
        estimates = np.asarray(estimates)
        ax.bar(x + offset, estimates, width * 0.86, color=color, alpha=0.82,
               label=label, zorder=3)
        ax.errorbar(x + offset, estimates, yerr=np.asarray((lower, upper)),
                    fmt="none", ecolor="#2F2F2F", elinewidth=0.8,
                    capsize=2.2, capthick=0.8, zorder=4)
    ax.axhline(0.0, color="#333333", linewidth=0.7)
    ax.set_xticks(x, [label for _, label in jerk_fields])
    ax.set_ylabel("Jerk reduction (positive is better)")
    ax.set_title("B  Closed-loop smoothness")
    ax.legend(loc="upper left")

    # C: expose heterogeneity rather than hiding it in an aggregate mean.
    ax = axes[2]
    scenes = (
        ("path_gentle_s_l43", "Gentle S"),
        ("path_double_turn_l43", "Double turn"),
        ("path_slalom_l43", "Slalom"),
    )
    positions = []
    values = []
    colors = []
    for scene_index, (scene, _) in enumerate(scenes):
        for domain_index, (domain, _, color) in enumerate(domains):
            positions.append(scene_index + (domain_index - 0.5) * 0.26)
            values.append([
                row["path_length_reduction_m"] for row in effects
                if row["scene"] == scene and row["physics_domain"] == domain
            ])
            colors.append(color)
    boxes = ax.boxplot(
        values, positions=positions, widths=0.21, patch_artist=True,
        showfliers=False, medianprops={"color": "#222222", "linewidth": 0.8},
        whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7},
        boxprops={"linewidth": 0.7},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.axhline(0.0, color="#333333", linewidth=0.7)
    ax.set_xticks(np.arange(len(scenes)), [label for _, label in scenes], rotation=12)
    ax.set_ylabel("Path-length reduction (m)")
    ax.set_title("C  Task/domain heterogeneity")

    fig.suptitle(
        "Task-specific ICODE: prediction accuracy and paired MuJoCo control effects",
        fontsize=10.2, fontweight="bold",
    )
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_prefix) + ".pdf")
    fig.savefig(str(output_prefix) + ".png", dpi=300)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-summary", required=True)
    parser.add_argument("--effects-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args(argv)
    plot(args.prediction_summary, args.effects_csv, args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
