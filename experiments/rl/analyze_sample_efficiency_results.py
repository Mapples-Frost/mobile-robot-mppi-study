#!/usr/bin/env python3
"""Reproducible paired/nested analysis and figures for L26."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.run_scene_complexity_gate_ablation import _write_csv


BOOTSTRAP_SEED = 20260762
BOOTSTRAP_REPLICATES = 20000
METHODS = ("traditional_mppi", "complexity_lcb")
METHOD_LABELS = {
    "traditional_mppi": "Traditional MPPI",
    "complexity_lcb": "Complexity-Gated RL",
}
COLORS = {"traditional_mppi": "#8C8C8C", "complexity_lcb": "#0072B2"}
MARKERS = {"traditional_mppi": "s", "complexity_lcb": "o"}
LINESTYLES = {"traditional_mppi": "--", "complexity_lcb": "-"}


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _boolean(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _float(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError("non-finite %s" % name)
    return value


def _mcnemar_exact(gains, losses):
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(int(gains), int(losses)) + 1)
    ) / float(2 ** discordant)
    return min(1.0, 2.0 * tail)


def _holm_adjust(p_values):
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, (index, value) in enumerate(indexed):
        candidate = min(1.0, (total - rank) * float(value))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _nested_bootstrap(rows, value_function, rng, replicates=BOOTSTRAP_REPLICATES):
    by_training = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_training[int(row["training_seed"])][int(row["episode_seed"])].append(row)
    training_ids = sorted(by_training)
    if len(training_ids) < 2:
        raise ValueError("nested bootstrap requires multiple training checkpoints")
    estimates = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sample = []
        selected_training = rng.choice(training_ids, size=len(training_ids), replace=True)
        for training_id in selected_training:
            episode_map = by_training[int(training_id)]
            episode_ids = sorted(episode_map)
            selected_episodes = rng.choice(
                episode_ids, size=len(episode_ids), replace=True
            )
            for episode_id in selected_episodes:
                sample.extend(episode_map[int(episode_id)])
        estimates[replicate] = float(value_function(sample))
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _mean_success(rows):
    return float(np.mean([_boolean(row["success"]) for row in rows]))


def _mean_compute(rows):
    return float(np.mean([_float(row, "planner_compute_ms_mean") for row in rows]))


def _paired_difference(rows):
    return float(np.mean([
        float(_boolean(row["success"])) - float(_boolean(row["traditional_success"]))
        for row in rows
    ]))


def _descriptive_curves(episodes, rng):
    records = []
    grouped = defaultdict(list)
    for row in episodes:
        grouped[(row["scene_role"], row["scene"], int(row["num_samples"]), row["condition"])].append(row)
    for (role, scene, num_samples, condition), rows in sorted(grouped.items()):
        success = _mean_success(rows)
        success_ci = _nested_bootstrap(rows, _mean_success, rng)
        compute = _mean_compute(rows)
        compute_ci = _nested_bootstrap(rows, _mean_compute, rng)
        records.append({
            "scene_role": role,
            "scene": scene,
            "num_samples": num_samples,
            "condition": condition,
            "episodes": len(rows),
            "success_rate": success,
            "success_ci95_low": success_ci[0],
            "success_ci95_high": success_ci[1],
            "planner_compute_ms_mean": compute,
            "planner_compute_ci95_low": compute_ci[0],
            "planner_compute_ci95_high": compute_ci[1],
        })
    return records


def _paired_inference(paired, rng):
    records = []
    grouped = defaultdict(list)
    for row in paired:
        grouped[(row["scene_role"], row["scene"], int(row["num_samples"]))].append(row)
    for (role, scene, num_samples), rows in sorted(grouped.items()):
        gains = int(sum(_boolean(row["traditional_success_gained"]) for row in rows))
        losses = int(sum(_boolean(row["traditional_success_lost"]) for row in rows))
        difference = _paired_difference(rows)
        ci = _nested_bootstrap(rows, _paired_difference, rng)
        records.append({
            "scene_role": role,
            "scene": scene,
            "num_samples": num_samples,
            "pairs": len(rows),
            "success_gains": gains,
            "success_losses": losses,
            "paired_success_rate_difference": difference,
            "difference_ci95_low": ci[0],
            "difference_ci95_high": ci[1],
            "mcnemar_exact_p": _mcnemar_exact(gains, losses),
        })
    adjusted = _holm_adjust([row["mcnemar_exact_p"] for row in records])
    for row, value in zip(records, adjusted):
        row["holm_adjusted_p"] = value
    return records


def _cross_budget(episodes, rng, low_k=100, high_k=400):
    records = []
    blocking_scenes = sorted({
        row["scene"] for row in episodes if row["scene_role"] == "blocking"
    })
    for scene in blocking_scenes:
        lookup = {
            (
                int(row["training_seed"]),
                int(row["episode_seed"]),
                int(row["num_samples"]),
                row["condition"],
            ): row
            for row in episodes if row["scene"] == scene
        }
        rows = []
        groups = sorted({key[:2] for key in lookup})
        for training_seed, episode_seed in groups:
            gated = lookup[(training_seed, episode_seed, low_k, "complexity_lcb")]
            traditional = lookup[(training_seed, episode_seed, high_k, "traditional_mppi")]
            rows.append({
                "training_seed": training_seed,
                "episode_seed": episode_seed,
                "success": gated["success"],
                "traditional_success": traditional["success"],
                "planner_compute_ms_mean": gated["planner_compute_ms_mean"],
                "traditional_planner_compute_ms_mean": traditional["planner_compute_ms_mean"],
            })
        diff = _paired_difference(rows)
        diff_ci = _nested_bootstrap(rows, _paired_difference, rng)
        ratio_fn = lambda sample: float(np.mean([
            _float(row, "planner_compute_ms_mean") for row in sample
        ]) / np.mean([
            _float(row, "traditional_planner_compute_ms_mean") for row in sample
        ]))
        ratio = ratio_fn(rows)
        ratio_ci = _nested_bootstrap(rows, ratio_fn, rng)
        records.append({
            "scene": scene,
            "gated_low_k": low_k,
            "traditional_high_k": high_k,
            "pairs": len(rows),
            "paired_success_rate_difference": diff,
            "difference_ci95_low": diff_ci[0],
            "difference_ci95_high": diff_ci[1],
            "planner_time_ratio": ratio,
            "planner_ratio_ci95_low": ratio_ci[0],
            "planner_ratio_ci95_high": ratio_ci[1],
        })
    return records


def _lookup_curve(curves, scene, method):
    return sorted(
        [row for row in curves if row["scene"] == scene and row["condition"] == method],
        key=lambda row: int(row["num_samples"]),
    )


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })


def _plot_scene_curves(curves, output):
    _style()
    names = [
        ("clean_dynamics", "Clean dynamics"),
        ("clean_single_obstacle", "Single obstacle"),
        ("narrow_corridor", "Narrow corridor"),
        ("u_trap_long_board", "U-trap"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.1), sharex=True, sharey=True)
    for panel, (ax, (scene, title)) in enumerate(zip(axes.flat, names)):
        for method in METHODS:
            rows = _lookup_curve(curves, scene, method)
            x = np.asarray([int(row["num_samples"]) for row in rows])
            y = np.asarray([float(row["success_rate"]) for row in rows])
            low = np.asarray([float(row["success_ci95_low"]) for row in rows])
            high = np.asarray([float(row["success_ci95_high"]) for row in rows])
            ax.plot(
                x, y, label=METHOD_LABELS[method], color=COLORS[method],
                marker=MARKERS[method], linestyle=LINESTYLES[method], linewidth=1.8,
            )
            ax.fill_between(x, low, high, color=COLORS[method], alpha=0.13)
        ax.set_title("(%s) %s" % (chr(97 + panel), title), loc="left", fontweight="bold")
        ax.set_xticks((50, 100, 200, 400))
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
    axes[1, 0].set_xlabel("MPPI samples, K")
    axes[1, 1].set_xlabel("MPPI samples, K")
    axes[0, 0].set_ylabel("Success rate")
    axes[1, 0].set_ylabel("Success rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.13, hspace=0.25, wspace=0.12)
    fig.savefig(output / "fig_l26_scene_success_vs_k.pdf")
    fig.savefig(output / "fig_l26_scene_success_vs_k.png", dpi=300)
    plt.close(fig)


def _pooled_curves(episodes, rng):
    records = []
    for num_samples in sorted({int(row["num_samples"]) for row in episodes}):
        for method in METHODS:
            rows = [
                row for row in episodes
                if row["scene_role"] == "blocking"
                and int(row["num_samples"]) == num_samples
                and row["condition"] == method
            ]
            success = _mean_success(rows)
            success_ci = _nested_bootstrap(rows, _mean_success, rng)
            compute = _mean_compute(rows)
            compute_ci = _nested_bootstrap(rows, _mean_compute, rng)
            records.append({
                "num_samples": num_samples,
                "condition": method,
                "episodes": len(rows),
                "success_rate": success,
                "success_ci95_low": success_ci[0],
                "success_ci95_high": success_ci[1],
                "planner_compute_ms_mean": compute,
                "planner_compute_ci95_low": compute_ci[0],
                "planner_compute_ci95_high": compute_ci[1],
            })
    return records


def _plot_tradeoff(pooled, output):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.7))
    for method in METHODS:
        rows = sorted(
            [row for row in pooled if row["condition"] == method],
            key=lambda row: int(row["num_samples"]),
        )
        x = np.asarray([int(row["num_samples"]) for row in rows])
        for ax, value, low_name, high_name, ylabel in (
            (axes[0], "success_rate", "success_ci95_low", "success_ci95_high", "Pooled blocking success rate"),
            (axes[1], "planner_compute_ms_mean", "planner_compute_ci95_low", "planner_compute_ci95_high", "Planner time per step (ms)"),
        ):
            y = np.asarray([float(row[value]) for row in rows])
            low = np.asarray([float(row[low_name]) for row in rows])
            high = np.asarray([float(row[high_name]) for row in rows])
            ax.plot(x, y, label=METHOD_LABELS[method], color=COLORS[method], marker=MARKERS[method], linestyle=LINESTYLES[method], linewidth=1.8)
            ax.fill_between(x, low, high, color=COLORS[method], alpha=0.13)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("MPPI samples, K")
            ax.set_xticks((50, 100, 200, 400))
    axes[0].set_title("(a) Control performance", loc="left", fontweight="bold")
    axes[1].set_title("(b) Wall-clock cost", loc="left", fontweight="bold")
    axes[0].set_ylim(-0.03, 1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.25, wspace=0.32)
    fig.savefig(output / "fig_l26_success_compute_tradeoff.pdf")
    fig.savefig(output / "fig_l26_success_compute_tradeoff.png", dpi=300)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.input_dir).resolve()
    episodes = _read_csv(output / "episodes.csv")
    paired = _read_csv(output / "paired_episodes.csv")
    with (output / "development_gate.json").open("r", encoding="utf-8") as handle:
        gate = json.load(handle)

    rng = np.random.RandomState(BOOTSTRAP_SEED)
    curves = _descriptive_curves(episodes, rng)
    inference = _paired_inference(paired, rng)
    cross_budget = _cross_budget(episodes, rng)
    pooled = _pooled_curves(episodes, rng)
    _write_csv(output / "bootstrap_curves.csv", curves)
    _write_csv(output / "paired_inference.csv", inference)
    _write_csv(output / "cross_budget_comparisons.csv", cross_budget)
    _write_csv(output / "pooled_curves.csv", pooled)
    _plot_scene_curves(curves, output)
    _plot_tradeoff(pooled, output)
    report = {
        "study": "L26_sample_efficiency",
        "analysis_type": "development_confirmatory_plus_declared_descriptive",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "independence_guard": (
            "two-stage bootstrap resamples training checkpoint, then episode seed; "
            "control steps are never treated as independent samples"
        ),
        "multiple_testing": "Holm correction across all scene-by-K paired McNemar tests",
        "development_gate": gate,
        "paired_inference": inference,
        "cross_budget_comparisons": cross_budget,
        "pooled_curves": pooled,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with (output / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({
        "development_gate_passed": gate["passed"],
        "cross_budget_comparisons": cross_budget,
        "pooled_curves": pooled,
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
