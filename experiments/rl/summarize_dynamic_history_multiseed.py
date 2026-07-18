#!/usr/bin/env python3
"""Audit the preregistered three-seed L34 dynamic-training replication."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.summarize_dynamic_history_training import (
    _audit,
    _best_trained,
    _bool,
    _checkpoint_rows,
    _paired_rows,
    _read_csv,
    _write_csv,
)


def _training_seed(run_dir):
    with (run_dir / "run_metadata.json").open("r", encoding="utf-8") as handle:
        return int(json.load(handle)["training"]["seed"])


def _summarize_run(run_dir, expected_steps):
    validation = _read_csv(run_dir / "validation_episodes.csv")
    checkpoints = _checkpoint_rows("l34_bounded_correction", validation)
    initial = next(row for row in checkpoints if row["global_step"] == 0)
    best = _best_trained(checkpoints)
    paired = _paired_rows(validation, validation, best["global_step"])
    episodes = _read_csv(run_dir / "episodes.csv")
    gains = int(sum(row["success_difference"] > 0 for row in paired))
    losses = int(sum(row["success_difference"] < 0 for row in paired))
    collision_regressions = int(sum(
        row["collision_difference"] > 0 for row in paired
    ))
    collision_improvements = int(sum(
        row["collision_difference"] < 0 for row in paired
    ))
    seed = _training_seed(run_dir)
    row = {
        "training_seed": seed,
        "best_step": int(best["global_step"]),
        "initial_successes": int(initial["successes"]),
        "best_successes": int(best["successes"]),
        "success_gains": gains,
        "success_losses": losses,
        "net_success_gain": gains - losses,
        "initial_collisions": int(initial["collisions"]),
        "best_collisions": int(best["collisions"]),
        "collision_regressions": collision_regressions,
        "collision_improvements": collision_improvements,
        "initial_mean_goal_distance_m": float(initial["mean_goal_distance"]),
        "best_mean_goal_distance_m": float(best["mean_goal_distance"]),
        "mean_goal_distance_improvement_m": float(
            initial["mean_goal_distance"] - best["mean_goal_distance"]
        ),
        "training_episodes": len(episodes),
        "training_successes": int(sum(_bool(item["success"]) for item in episodes)),
        "training_collisions": int(sum(_bool(item["collision"]) for item in episodes)),
    }
    audit = _audit(run_dir, validation, checkpoints, expected_steps)
    return row, paired, audit, checkpoints


def _plot(output_dir, rows):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    seeds = [str(row["training_seed"])[-2:] for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.8))
    metrics = (
        ("successes", "Successful episodes (of 6)", "A  Held-out success"),
        ("collisions", "Colliding episodes (of 6)", "B  Held-out collision"),
        ("mean_goal_distance_m", "Mean final distance (m)", "C  Goal-distance error"),
    )
    for axis, (metric, ylabel, title) in zip(axes, metrics):
        initial = [row["initial_" + metric] for row in rows]
        best = [row["best_" + metric] for row in rows]
        axis.bar(x - width / 2, initial, width, label="Step 0", color="#9AA0A6")
        axis.bar(x + width / 2, best, width, label="Best trained", color="#0072B2")
        axis.set_xticks(x)
        axis.set_xticklabels(seeds)
        axis.set_xlabel("Training seed suffix")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(
        "L34 bounded RL correction: independent training-seed replication",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Each bar: two held-out motion paths × three fixed episode seeds in the combined-unseen MuJoCo domain.",
        ha="center",
        fontsize=7.2,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.92), w_pad=1.0)
    fig.savefig(output_dir / "fig_l34_dynamic_multiseed.png")
    fig.savefig(output_dir / "fig_l34_dynamic_multiseed.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--evaluation-interval", type=int, default=5000)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_steps = tuple(range(0, args.total_steps + 1, args.evaluation_interval))

    rows = []
    all_pairs = []
    audits = {}
    checkpoint_rows = []
    for value in args.run_dir:
        run_dir = Path(value).resolve()
        row, paired, audit, checkpoints = _summarize_run(run_dir, expected_steps)
        rows.append(row)
        for item in paired:
            all_pairs.append(dict(item, training_seed=row["training_seed"]))
        audits[str(row["training_seed"])] = audit
        for item in checkpoints:
            checkpoint_rows.append(dict(item, training_seed=row["training_seed"]))
    rows.sort(key=lambda row: row["training_seed"])

    complete_audits = all(
        audit["observed_steps"] == list(expected_steps)
        and audit["observed_validation_rows"] == audit["expected_validation_rows"]
        and audit["duplicate_validation_keys"] == 0
        and audit["nonfinite_validation_rows"] == 0
        and audit["config_snapshot_exists"]
        and audit["initial_checkpoint_exists"]
        and audit["best_checkpoint_exists"]
        and audit["latest_checkpoint_exists"]
        for audit in audits.values()
    )
    positive_seed_count = int(sum(row["net_success_gain"] > 0 for row in rows))
    collision_noninferior_count = int(sum(
        row["best_collisions"] <= row["initial_collisions"] for row in rows
    ))
    pooled_gains = int(sum(row["success_gains"] for row in rows))
    pooled_losses = int(sum(row["success_losses"] for row in rows))
    pooled_collision_regressions = int(sum(
        row["collision_regressions"] for row in rows
    ))
    pooled_collision_improvements = int(sum(
        row["collision_improvements"] for row in rows
    ))
    mean_distance_improvement = float(np.mean([
        row["mean_goal_distance_improvement_m"] for row in rows
    ]))
    gate = {
        "training_seed_count": len(rows),
        "positive_net_success_seed_count": positive_seed_count,
        "collision_noninferior_seed_count": collision_noninferior_count,
        "pooled_success_gains": pooled_gains,
        "pooled_success_losses": pooled_losses,
        "pooled_collision_regressions": pooled_collision_regressions,
        "pooled_collision_improvements": pooled_collision_improvements,
        "mean_goal_distance_improvement_m": mean_distance_improvement,
        "artifact_integrity": bool(complete_audits),
    }
    gate["passed"] = bool(
        len(rows) == 3
        and positive_seed_count >= 2
        and collision_noninferior_count >= 2
        and pooled_gains > pooled_losses
        and pooled_collision_regressions <= pooled_collision_improvements
        and mean_distance_improvement > 0.0
        and complete_audits
    )
    summary = {"training_seed_rows": rows, "audits": audits, "development_gate": gate}
    _write_csv(output_dir / "training_seed_summary.csv", rows)
    _write_csv(output_dir / "paired_episode_effects.csv", all_pairs)
    _write_csv(output_dir / "checkpoint_summary.csv", checkpoint_rows)
    with (output_dir / "multiseed_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _plot(output_dir, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
