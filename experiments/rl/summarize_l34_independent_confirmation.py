#!/usr/bin/env python3
"""Audit and summarize the preregistered L35 paired confirmation."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml


def _resolved_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty confirmation table")
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _exact_two_sided_sign_p(positive, negative):
    n = int(positive) + int(negative)
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(0, min(positive, negative) + 1))
    return min(1.0, 2.0 * tail / float(2 ** n))


def _paired_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (int(row["training_seed"]), str(row["scene"]), int(row["seed"]))
        role = str(row["checkpoint_role"])
        if role in grouped[key]:
            raise ValueError("duplicate L35 episode key %s/%s" % (key, role))
        grouped[key][role] = row
    pairs = []
    for key, values in sorted(grouped.items()):
        if set(values) != {"initial", "best"}:
            raise ValueError("missing initial/best member for L35 key %s" % (key,))
        initial = values["initial"]
        best = values["best"]
        pairs.append({
            "training_seed": key[0],
            "scene": key[1],
            "episode_seed": key[2],
            "initial_success": int(_bool(initial["success"])),
            "best_success": int(_bool(best["success"])),
            "success_difference": int(_bool(best["success"])) - int(_bool(initial["success"])),
            "initial_collision": int(_bool(initial["collision"])),
            "best_collision": int(_bool(best["collision"])),
            "collision_difference": int(_bool(best["collision"])) - int(_bool(initial["collision"])),
            "initial_final_goal_distance_m": float(initial["final_goal_distance"]),
            "best_final_goal_distance_m": float(best["final_goal_distance"]),
            "goal_distance_improvement_m": float(initial["final_goal_distance"]) - float(best["final_goal_distance"]),
        })
    return pairs


def _hierarchical_bootstrap(pairs, field, seed, replicates):
    grouped = defaultdict(list)
    for row in pairs:
        grouped[int(row["training_seed"])].append(row)
    training_seeds = np.asarray(sorted(grouped), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        selected_models = rng.choice(training_seeds, size=len(training_seeds), replace=True)
        model_means = []
        for training_seed in selected_models:
            values = np.asarray(
                [float(row[field]) for row in grouped[int(training_seed)]],
                dtype=np.float64,
            )
            model_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        samples[index] = float(np.mean(model_means))
    return {
        "estimate": float(np.mean([
            np.mean([float(row[field]) for row in grouped[training_seed]])
            for training_seed in sorted(grouped)
        ])),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "replicates": int(replicates),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    design = config["rl"]["confirmation"]
    gate_config = design["gate"]
    input_dir = _resolved_path(args.input_dir)
    rows = _read_csv(input_dir / "confirmation_episodes.csv")
    pairs = _paired_rows(rows)

    expected_episode_count = (
        len(design["training_runs"]) * 2 * len(design["scenes"])
        * len(design["episode_seeds"])
    )
    expected_pair_count = expected_episode_count // 2
    finite_fields = (
        "final_goal_distance", "minimum_clearance", "control_jerk",
        "planner_compute_ms_mean",
    )
    nonfinite_rows = sum(
        any(
            row.get(name) not in (None, "") and not np.isfinite(float(row[name]))
            for name in finite_fields
        )
        for row in rows
    )
    metadata_path = input_dir / "run_metadata.json"
    manifest_path = input_dir / "frozen_manifest.json"
    integrity = bool(
        len(rows) == expected_episode_count
        and len(pairs) == expected_pair_count
        and nonfinite_rows == 0
        and metadata_path.exists()
        and manifest_path.exists()
    )

    seed_rows = []
    by_seed = defaultdict(list)
    for row in pairs:
        by_seed[int(row["training_seed"])].append(row)
    for training_seed in sorted(by_seed):
        selected = by_seed[training_seed]
        initial_success = sum(row["initial_success"] for row in selected)
        best_success = sum(row["best_success"] for row in selected)
        initial_collision = sum(row["initial_collision"] for row in selected)
        best_collision = sum(row["best_collision"] for row in selected)
        seed_rows.append({
            "training_seed": training_seed,
            "pairs": len(selected),
            "initial_successes": initial_success,
            "best_successes": best_success,
            "net_success_gain": best_success - initial_success,
            "initial_collisions": initial_collision,
            "best_collisions": best_collision,
            "net_collision_increase": best_collision - initial_collision,
            "mean_goal_distance_improvement_m": float(np.mean([
                row["goal_distance_improvement_m"] for row in selected
            ])),
        })

    success_gains = sum(row["success_difference"] > 0 for row in pairs)
    success_losses = sum(row["success_difference"] < 0 for row in pairs)
    collision_improvements = sum(row["collision_difference"] < 0 for row in pairs)
    collision_regressions = sum(row["collision_difference"] > 0 for row in pairs)
    positive_success_seeds = sum(row["net_success_gain"] > 0 for row in seed_rows)
    collision_noninferior_seeds = sum(
        row["net_collision_increase"] <= 0 for row in seed_rows
    )
    mean_distance_improvement = float(np.mean([
        row["mean_goal_distance_improvement_m"] for row in seed_rows
    ]))
    pooled_net_success_gain = int(success_gains - success_losses)
    pooled_net_collision_increase = int(collision_regressions - collision_improvements)

    gate = {
        "artifact_integrity": integrity,
        "positive_success_training_seeds": int(positive_success_seeds),
        "collision_noninferior_training_seeds": int(collision_noninferior_seeds),
        "pooled_success_gains": int(success_gains),
        "pooled_success_losses": int(success_losses),
        "pooled_net_success_gain": pooled_net_success_gain,
        "pooled_collision_improvements": int(collision_improvements),
        "pooled_collision_regressions": int(collision_regressions),
        "pooled_net_collision_increase": pooled_net_collision_increase,
        "mean_goal_distance_improvement_m": mean_distance_improvement,
    }
    gate["passed"] = bool(
        integrity
        and positive_success_seeds >= int(gate_config["minimum_positive_success_training_seeds"])
        and collision_noninferior_seeds >= int(gate_config["minimum_collision_noninferior_training_seeds"])
        and pooled_net_success_gain >= int(gate_config["minimum_pooled_net_success_gain"])
        and pooled_net_collision_increase <= int(gate_config["maximum_pooled_net_collision_increase"])
        and mean_distance_improvement >= float(gate_config["minimum_mean_goal_distance_improvement_m"])
    )

    bootstrap_seed = int(design["bootstrap_seed"])
    bootstrap_replicates = int(design["bootstrap_replicates"])
    summary = {
        "design_id": str(design["design_id"]),
        "episodes": len(rows),
        "paired_cells": len(pairs),
        "training_seed_summary": seed_rows,
        "descriptives": {
            "initial_success_rate": float(np.mean([row["initial_success"] for row in pairs])),
            "best_success_rate": float(np.mean([row["best_success"] for row in pairs])),
            "initial_collision_rate": float(np.mean([row["initial_collision"] for row in pairs])),
            "best_collision_rate": float(np.mean([row["best_collision"] for row in pairs])),
            "mean_goal_distance_improvement_m": mean_distance_improvement,
        },
        "paired_exact_tests": {
            "success_gain_vs_loss_two_sided_p": _exact_two_sided_sign_p(success_gains, success_losses),
            "collision_improvement_vs_regression_two_sided_p": _exact_two_sided_sign_p(
                collision_improvements, collision_regressions
            ),
        },
        "hierarchical_bootstrap": {
            "success_rate_difference": _hierarchical_bootstrap(
                pairs, "success_difference", bootstrap_seed, bootstrap_replicates
            ),
            "collision_rate_difference": _hierarchical_bootstrap(
                pairs, "collision_difference", bootstrap_seed + 1, bootstrap_replicates
            ),
            "goal_distance_improvement_m": _hierarchical_bootstrap(
                pairs, "goal_distance_improvement_m", bootstrap_seed + 2, bootstrap_replicates
            ),
        },
        "integrity": {
            "expected_episodes": expected_episode_count,
            "observed_episodes": len(rows),
            "expected_pairs": expected_pair_count,
            "observed_pairs": len(pairs),
            "nonfinite_rows": int(nonfinite_rows),
        },
        "confirmation_gate": gate,
        "interpretation_guard": (
            "This confirms bounded-RL training only under nominal rollout dynamics; "
            "it does not establish an RL-by-ICODE interaction."
        ),
    }
    _write_csv(input_dir / "paired_confirmation_effects.csv", pairs)
    _write_csv(input_dir / "training_seed_confirmation_summary.csv", seed_rows)
    (input_dir / "confirmation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

