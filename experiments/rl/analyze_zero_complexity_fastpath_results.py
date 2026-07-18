#!/usr/bin/env python3
"""Paired nested timing analysis and publication figure for L27."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.run_scene_complexity_gate_ablation import _write_csv


BOOTSTRAP_SEED = 20260772
BOOTSTRAP_REPLICATES = 20000


def _read(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pairs(episodes):
    lookup = {
        (int(row["training_seed"]), row["scene"], int(row["episode_seed"]), row["condition"]): row
        for row in episodes
    }
    groups = sorted({key[:-1] for key in lookup})
    result = []
    for training_seed, scene, episode_seed in groups:
        standard = lookup[(training_seed, scene, episode_seed, "standard_complexity")]
        fast = lookup[(training_seed, scene, episode_seed, "zero_complexity_fastpath")]
        result.append({
            "training_seed": training_seed,
            "scene": scene,
            "scene_role": standard["scene_role"],
            "episode_seed": episode_seed,
            "standard_planner_ms": float(standard["planner_compute_ms_mean"]),
            "fastpath_planner_ms": float(fast["planner_compute_ms_mean"]),
            "paired_reduction_fraction": float(
                1.0 - float(fast["planner_compute_ms_mean"]) / float(standard["planner_compute_ms_mean"])
            ),
        })
    return result


def _nested_timing_bootstrap(rows, rng, replicates):
    """Vectorized checkpoint/episode bootstrap for paired timing outcomes."""
    training_ids = sorted({int(row["training_seed"]) for row in rows})
    episode_ids = sorted({int(row["episode_seed"]) for row in rows})
    scenes = sorted({row["scene"] for row in rows})
    if len(training_ids) < 2:
        raise ValueError("nested bootstrap requires multiple training checkpoints")
    lookup = {
        (int(row["training_seed"]), int(row["episode_seed"]), row["scene"]): row
        for row in rows
    }
    expected = len(training_ids) * len(episode_ids) * len(scenes)
    if len(lookup) != expected:
        raise ValueError("timing bootstrap requires a balanced checkpoint/episode/scene design")

    shape = (len(training_ids), len(episode_ids), len(scenes))
    standard = np.empty(shape, dtype=np.float64)
    fast = np.empty(shape, dtype=np.float64)
    for training_index, training_seed in enumerate(training_ids):
        for episode_index, episode_seed in enumerate(episode_ids):
            for scene_index, scene in enumerate(scenes):
                row = lookup[(training_seed, episode_seed, scene)]
                standard[training_index, episode_index, scene_index] = row["standard_planner_ms"]
                fast[training_index, episode_index, scene_index] = row["fastpath_planner_ms"]

    replicate_count = int(replicates)
    selected_training = rng.randint(
        0, len(training_ids), size=(replicate_count, len(training_ids))
    )
    selected_episodes = rng.randint(
        0,
        len(episode_ids),
        size=(replicate_count, len(training_ids), len(episode_ids)),
    )
    expanded_training = np.broadcast_to(
        selected_training[:, :, np.newaxis], selected_episodes.shape
    )
    standard_means = standard[expanded_training, selected_episodes].mean(axis=(1, 2, 3))
    fast_means = fast[expanded_training, selected_episodes].mean(axis=(1, 2, 3))
    reductions = 1.0 - fast_means / standard_means

    def interval(values):
        return (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )

    return interval(standard_means), interval(fast_means), interval(reductions)


def _analysis(pairs, rng, replicates=BOOTSTRAP_REPLICATES):
    result = {}
    for role in ("control", "blocking"):
        rows = [row for row in pairs if row["scene_role"] == role]
        standard = float(np.mean([row["standard_planner_ms"] for row in rows]))
        fast = float(np.mean([row["fastpath_planner_ms"] for row in rows]))
        reduction = float(1.0 - fast / standard)
        standard_ci, fast_ci, reduction_ci = _nested_timing_bootstrap(
            rows, rng, replicates
        )
        result[role] = {
            "pairs": len(rows),
            "standard_planner_ms": standard,
            "standard_ci95": standard_ci,
            "fastpath_planner_ms": fast,
            "fastpath_ci95": fast_ci,
            "time_reduction_fraction": reduction,
            "reduction_ci95": reduction_ci,
        }
    return result


def _plot(result, output):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })
    roles = ("control", "blocking")
    labels = ("Clean control", "Blocking scenes")
    x = np.arange(len(roles))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.65))
    methods = (
        ("standard_planner_ms", "standard_ci95", "Standard", "#8C8C8C", "//"),
        ("fastpath_planner_ms", "fastpath_ci95", "Fast-path", "#0072B2", ""),
    )
    for index, (value_key, ci_key, label, color, hatch) in enumerate(methods):
        values = np.asarray([result[role][value_key] for role in roles])
        low = np.asarray([result[role][ci_key][0] for role in roles])
        high = np.asarray([result[role][ci_key][1] for role in roles])
        positions = x + (index - 0.5) * width
        axes[0].bar(positions, values, width * 0.9, label=label, color=color, hatch=hatch, edgecolor="white", linewidth=0.6)
        axes[0].errorbar(positions, values, yerr=np.vstack((values - low, high - values)), fmt="none", ecolor="#333333", capsize=3, linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Planner time per step (ms)")
    axes[0].set_title("(a) Behavior-preserving runtime", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    reductions = np.asarray([result[role]["time_reduction_fraction"] * 100.0 for role in roles])
    low = np.asarray([result[role]["reduction_ci95"][0] * 100.0 for role in roles])
    high = np.asarray([result[role]["reduction_ci95"][1] * 100.0 for role in roles])
    bars = axes[1].bar(x, reductions, 0.52, color="#009E73", edgecolor="white", linewidth=0.6)
    axes[1].errorbar(x, reductions, yerr=np.vstack((reductions - low, high - reductions)), fmt="none", ecolor="#333333", capsize=3, linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Planner-time reduction (%)")
    axes[1].set_title("(b) Paired reduction", loc="left", fontweight="bold")
    for bar, value in zip(bars, reductions):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, "%.1f%%" % value, ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(wspace=0.30)
    fig.savefig(output / "fig_l27_fastpath_timing.pdf")
    fig.savefig(output / "fig_l27_fastpath_timing.png", dpi=300)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.input_dir).resolve()
    episodes = _read(output / "episodes.csv")
    pairs = _pairs(episodes)
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    result = _analysis(pairs, rng)
    _write_csv(output / "timing_pairs.csv", pairs)
    report = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "method": "paired two-stage bootstrap over checkpoint then episode seed",
        "timing": result,
    }
    (output / "timing_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot(result, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
