#!/usr/bin/env python3
"""Audit L33/L34 training runs and plot fixed-seed validation trajectories."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _checkpoint_rows(run_name, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["global_step"])].append(row)
    output = []
    for step, values in sorted(grouped.items()):
        success = int(sum(_bool(row["success"]) for row in values))
        collision = int(sum(_bool(row["collision"]) for row in values))
        mean_return = float(np.mean([float(row["return"]) for row in values]))
        output.append({
            "run": run_name,
            "global_step": step,
            "episodes": len(values),
            "successes": success,
            "success_rate": success / float(len(values)),
            "collisions": collision,
            "collision_rate": collision / float(len(values)),
            "mean_goal_distance": float(np.mean([
                float(row["goal_distance"]) for row in values
            ])),
            "mean_return": mean_return,
            "scalar_score": (
                1000.0 * success / float(len(values))
                - 1000.0 * collision / float(len(values))
                + mean_return
            ),
            "selected": any(_bool(row.get("selection_selected", False)) for row in values),
        })
    return output


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _audit(run_dir, validation_rows, checkpoints, expected_steps):
    keys = [
        (int(row["global_step"]), str(row["scene"]), int(row["seed"]))
        for row in validation_rows
    ]
    nonfinite = 0
    for row in validation_rows:
        values = (
            float(row["return"]),
            float(row["goal_distance"]),
        )
        nonfinite += int(not all(math.isfinite(value) for value in values))
    episodes_per_step = {
        int(row["global_step"]): int(row["episodes"])
        for row in checkpoints
    }
    expected_per_step = max(episodes_per_step.values()) if episodes_per_step else 0
    expected_rows = len(expected_steps) * expected_per_step
    return {
        "expected_steps": list(expected_steps),
        "observed_steps": sorted(episodes_per_step),
        "expected_validation_rows": expected_rows,
        "observed_validation_rows": len(validation_rows),
        "unique_validation_keys": len(set(keys)),
        "duplicate_validation_keys": len(keys) - len(set(keys)),
        "nonfinite_validation_rows": nonfinite,
        "config_snapshot_exists": (run_dir / "config_snapshot.json").is_file(),
        "initial_checkpoint_exists": (run_dir / "checkpoints" / "initial.pt").is_file(),
        "best_checkpoint_exists": (run_dir / "checkpoints" / "best.pt").is_file(),
        "latest_checkpoint_exists": (run_dir / "checkpoints" / "latest.pt").is_file(),
    }


def _best_trained(rows):
    candidates = [row for row in rows if int(row["global_step"]) > 0]
    return max(candidates, key=lambda row: float(row["scalar_score"]))


def _paired_rows(initial_rows, candidate_rows, candidate_step):
    reference = {
        (str(row["scene"]), int(row["seed"])): row
        for row in initial_rows if int(row["global_step"]) == 0
    }
    candidate = {
        (str(row["scene"]), int(row["seed"])): row
        for row in candidate_rows if int(row["global_step"]) == int(candidate_step)
    }
    output = []
    for key in sorted(set(reference).intersection(candidate)):
        before = reference[key]
        after = candidate[key]
        output.append({
            "scene": key[0],
            "seed": key[1],
            "success_difference": int(_bool(after["success"])) - int(_bool(before["success"])),
            "collision_difference": int(_bool(after["collision"])) - int(_bool(before["collision"])),
            "goal_distance_improvement_m": (
                float(before["goal_distance"]) - float(after["goal_distance"])
            ),
            "return_difference": float(after["return"]) - float(before["return"]),
        })
    return output


def _plot(output_dir, l33, l34, best_l34):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    colors = {"l33": "#D55E00", "l34": "#0072B2", "collision": "#CC3311"}
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.75))
    for data, label, color in (
        (l33, "L33 full learned prior", colors["l33"]),
        (l34, "L34 bounded correction", colors["l34"]),
    ):
        steps = np.asarray([row["global_step"] for row in data]) / 1000.0
        axes[0].plot(steps, [row["successes"] for row in data], "o-", label=label, color=color)
        axes[1].plot(steps, [row["collisions"] for row in data], "o-", label=label, color=color)
        axes[2].plot(steps, [row["mean_goal_distance"] for row in data], "o-", label=label, color=color)
    axes[0].set_title("A  Held-out success", loc="left")
    axes[0].set_ylabel("Successful episodes (of 6)")
    axes[1].set_title("B  Held-out collision", loc="left")
    axes[1].set_ylabel("Colliding episodes (of 6)")
    axes[2].set_title("C  Goal-distance error", loc="left")
    axes[2].set_ylabel("Mean final distance (m)")
    for axis in axes:
        axis.set_xlabel("SAC environment steps (thousands)")
        axis.set_xticks(np.arange(0, 31, 10))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=2,
        fontsize=7.2,
        frameon=False,
    )
    axes[0].annotate(
        "best trained L34",
        xy=(best_l34["global_step"] / 1000.0, best_l34["successes"]),
        xytext=(2, 16),
        textcoords="offset points",
        fontsize=7,
        arrowprops={"arrowstyle": "->", "linewidth": 0.7},
    )
    fig.suptitle(
        "Dynamic-obstacle development: bounded RL correction preserves learnability",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Two held-out motion paths × three fixed seeds in a combined-unseen MuJoCo physics domain; development-only.",
        ha="center",
        fontsize=7.2,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.80), w_pad=1.0)
    fig.savefig(output_dir / "fig_l33_l34_dynamic_training.png")
    fig.savefig(output_dir / "fig_l33_l34_dynamic_training.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--l33-run", required=True)
    parser.add_argument("--l34-run", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--evaluation-interval", type=int, default=5000)
    args = parser.parse_args(argv)
    l33_dir = Path(args.l33_run).resolve()
    l34_dir = Path(args.l34_run).resolve()
    output_dir = Path(args.output_dir or l34_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    l33_validation = _read_csv(l33_dir / "validation_episodes.csv")
    l34_validation = _read_csv(l34_dir / "validation_episodes.csv")
    l33_checkpoints = _checkpoint_rows("l33_full_prior", l33_validation)
    l34_checkpoints = _checkpoint_rows("l34_bounded_correction", l34_validation)
    expected_steps = tuple(range(0, args.total_steps + 1, args.evaluation_interval))
    l33_audit = _audit(l33_dir, l33_validation, l33_checkpoints, expected_steps)
    l34_audit = _audit(l34_dir, l34_validation, l34_checkpoints, expected_steps)

    best_l33 = _best_trained(l33_checkpoints)
    best_l34 = _best_trained(l34_checkpoints)
    l34_initial = next(row for row in l34_checkpoints if row["global_step"] == 0)
    training_episodes = _read_csv(l34_dir / "episodes.csv")
    training_successes = int(sum(_bool(row["success"]) for row in training_episodes))
    integrity = all((
        l34_audit["observed_steps"] == list(expected_steps),
        l34_audit["observed_validation_rows"] == l34_audit["expected_validation_rows"],
        l34_audit["duplicate_validation_keys"] == 0,
        l34_audit["nonfinite_validation_rows"] == 0,
        l34_audit["config_snapshot_exists"],
        l34_audit["initial_checkpoint_exists"],
        l34_audit["best_checkpoint_exists"],
        l34_audit["latest_checkpoint_exists"],
    ))
    improvement_rule = bool(
        best_l34["successes"] > l34_initial["successes"]
        or (
            best_l34["successes"] == l34_initial["successes"]
            and l34_initial["mean_goal_distance"] - best_l34["mean_goal_distance"] >= 0.05
        )
    )
    gate = {
        "best_trained_step": int(best_l34["global_step"]),
        "collision_noninferior_to_l34_initial": bool(
            best_l34["collisions"] <= l34_initial["collisions"]
        ),
        "success_or_distance_improvement_over_l34_initial": improvement_rule,
        "success_superior_to_l33_best_trained": bool(
            best_l34["successes"] > best_l33["successes"]
        ),
        "training_contains_successful_episode": bool(training_successes > 0),
        "artifact_integrity": bool(integrity),
    }
    gate["passed"] = bool(all(gate[name] for name in gate if name not in ("best_trained_step", "passed")))

    paired = _paired_rows(l34_validation, l34_validation, best_l34["global_step"])
    summary = {
        "l33_audit": l33_audit,
        "l34_audit": l34_audit,
        "l33_best_trained": best_l33,
        "l34_initial": l34_initial,
        "l34_best_trained": best_l34,
        "l34_training_episodes": len(training_episodes),
        "l34_training_successes": training_successes,
        "l34_training_collisions": int(sum(_bool(row["collision"]) for row in training_episodes)),
        "development_gate": gate,
    }
    _write_csv(output_dir / "checkpoint_summary.csv", l33_checkpoints + l34_checkpoints)
    _write_csv(output_dir / "l34_initial_vs_best_pairs.csv", paired)
    with (output_dir / "l33_l34_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _plot(output_dir, l33_checkpoints, l34_checkpoints, best_l34)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
