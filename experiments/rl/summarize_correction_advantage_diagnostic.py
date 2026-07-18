#!/usr/bin/env python3
"""Audit and combine independent L18 correction-advantage runs."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.run_correction_advantage_diagnostic import (
    _condition_summary,
    _write_csv,
)


BOOL_FIELDS = {
    "success",
    "collision",
    "bc_success",
    "bc_collision",
    "bc_success_lost",
    "success_gained",
    "collision_regression",
}
INT_FIELDS = {"seed", "training_seed", "steps", "success_delta", "collision_delta"}
TEXT_FIELDS = {
    "condition",
    "scene",
    "gate_mode",
    "critic_source",
    "checkpoint",
    "policy_mode",
    "correction_advantage_gate_mode",
    "correction_advantage_critic_source",
}


def _bool(value):
    lowered = str(value).strip().lower()
    if lowered in ("true", "1"):
        return True
    if lowered in ("false", "0"):
        return False
    raise ValueError("invalid boolean value: %s" % value)


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for raw in rows:
        row = {}
        for name, value in raw.items():
            if name in TEXT_FIELDS:
                row[name] = value
            elif name in BOOL_FIELDS:
                row[name] = _bool(value)
            elif name in INT_FIELDS:
                row[name] = int(value)
            else:
                row[name] = float(value)
        result.append(row)
    return result


def _finite_numeric(rows):
    for row in rows:
        for name, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                return False, "%s=%s" % (name, value)
    return True, None


def _run_quality(
    run_dir,
    expected_conditions,
    expected_seeds,
    extra_critical_step_fields=(),
):
    episodes = _read_csv(run_dir / "episodes.csv")
    paired = _read_csv(run_dir / "paired_episodes.csv")
    steps = _read_csv(run_dir / "steps.csv")
    episode_keys = [
        (row["condition"], row["seed"]) for row in episodes
    ]
    paired_keys = [
        (row["condition"], row["seed"]) for row in paired
    ]
    step_keys = [
        (row["condition"], row["seed"], int(row["step"])) for row in steps
    ]
    finite, finite_problem = _finite_numeric(episodes + paired + steps)
    actual_conditions = sorted(set(row["condition"] for row in episodes))
    actual_seeds = sorted(set(int(row["seed"]) for row in episodes))
    critical_step_fields = (
        "online_conservative_advantage",
        "target_conservative_advantage",
        "correction_advantage_gate_alpha",
        "raw_applied_correction_abs_mean",
        "applied_correction_abs_mean",
    ) + tuple(extra_critical_step_fields)
    missing_critical = sum(
        any(name not in row for name in critical_step_fields) for row in steps
    )
    return {
        "episodes_rows": len(episodes),
        "paired_rows": len(paired),
        "steps_rows": len(steps),
        "episode_duplicate_keys": len(episode_keys) - len(set(episode_keys)),
        "paired_duplicate_keys": len(paired_keys) - len(set(paired_keys)),
        "step_duplicate_keys": len(step_keys) - len(set(step_keys)),
        "finite_numeric": finite,
        "finite_problem": finite_problem,
        "missing_critical_step_rows": missing_critical,
        "conditions_match": actual_conditions == sorted(expected_conditions),
        "seeds_match": actual_seeds == sorted(expected_seeds),
        "bc_raw_correction_max": float(max(
            (
                row["raw_applied_correction_abs_mean"]
                for row in steps if row["condition"] == "bc"
            ),
            default=0.0,
        )),
    }, paired


def _eligibility(condition, per_seed, pooled):
    if condition == "correction_none":
        return {
            "eligible": False,
            "reason": "ungated_reference_condition",
        }
    correlations = [
        values["online_return_spearman"]
        if condition == "correction_online_hard"
        else values["target_return_spearman"]
        for values in per_seed
    ]
    positive_correlation_seeds = sum(
        value is not None and value > 0.0 for value in correlations
    )
    no_success_losses = pooled["bc_success_losses"] == 0
    no_collision_regressions = pooled["collision_regressions"] == 0
    nondegenerate_gate = all(
        0.0 < values["mean_gate_accept_fraction"] < 1.0
        for values in per_seed
    )
    has_benefit = bool(
        pooled["success_gains"] > 0
        or pooled["mean_goal_distance_improvement"] > 0.0
    )
    checks = {
        "positive_correlation_training_seeds": positive_correlation_seeds,
        "requires_positive_correlation_training_seeds": 2,
        "no_bc_success_losses": no_success_losses,
        "no_collision_regressions": no_collision_regressions,
        "nondegenerate_gate_each_training_seed": nondegenerate_gate,
        "has_success_gain_or_distance_benefit": has_benefit,
    }
    checks["eligible"] = bool(
        positive_correlation_seeds >= 2
        and no_success_losses
        and no_collision_regressions
        and nondegenerate_gate
        and has_benefit
    )
    checks["reason"] = (
        "all_preregistered_checks_passed"
        if checks["eligible"]
        else "one_or_more_preregistered_checks_failed"
    )
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    run_dirs = [Path(value).resolve() for value in args.run_dir]
    if len(run_dirs) < 2:
        raise ValueError("multi-seed summary requires at least two run dirs")

    expected_conditions = (
        "bc",
        "correction_none",
        "correction_online_hard",
        "correction_target_hard",
    )
    quality = {}
    all_paired = []
    run_summaries = []
    training_seeds = []
    expected_diagnostic_seeds = None
    for run_dir in run_dirs:
        with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        training_seed = int(summary["training_seed"])
        if training_seed in training_seeds:
            raise ValueError("training seeds must be unique across run dirs")
        training_seeds.append(training_seed)
        diagnostic_seeds = [int(value) for value in summary["diagnostic_seeds"]]
        if expected_diagnostic_seeds is None:
            expected_diagnostic_seeds = diagnostic_seeds
        elif diagnostic_seeds != expected_diagnostic_seeds:
            raise ValueError("all runs must use the same ordered diagnostic seeds")
        run_quality, paired = _run_quality(
            run_dir, expected_conditions, diagnostic_seeds
        )
        quality[str(training_seed)] = run_quality
        all_paired.extend(paired)
        run_summaries.append(summary)

    pooled = _condition_summary(all_paired)
    per_training_seed = {}
    eligibility = {}
    for condition in sorted(pooled):
        values = [
            summary["conditions"][condition] for summary in run_summaries
        ]
        per_training_seed[condition] = {
            str(seed): value for seed, value in zip(training_seeds, values)
        }
        eligibility[condition] = _eligibility(condition, values, pooled[condition])

    quality_passed = all(
        values["episode_duplicate_keys"] == 0
        and values["paired_duplicate_keys"] == 0
        and values["step_duplicate_keys"] == 0
        and values["finite_numeric"]
        and values["missing_critical_step_rows"] == 0
        and values["conditions_match"]
        and values["seeds_match"]
        and values["bc_raw_correction_max"] == 0.0
        for values in quality.values()
    )
    aggregate = {
        "training_seeds": training_seeds,
        "diagnostic_seeds": expected_diagnostic_seeds,
        "independent_training_repeats": len(training_seeds),
        "paired_episode_rows": len(all_paired),
        "quality_passed": quality_passed,
        "quality": quality,
        "per_training_seed": per_training_seed,
        "pooled_descriptive": pooled,
        "preregistered_eligibility": eligibility,
        "interpretation_guard": (
            "episode rows are paired repeated tasks; the independent training "
            "replicate count is the number of training seeds"
        ),
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "paired_episodes_all_training_seeds.csv", all_paired)
    with (output / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
