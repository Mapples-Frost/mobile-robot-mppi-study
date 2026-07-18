#!/usr/bin/env python3
"""Audit Gate 3 development and sealed confirmation as one paired study."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.evaluation.paired_checkpoint import (
    paired_checkpoint_effects,
)


METRICS = {
    "success": True,
    "collision": False,
    "final_goal_distance": False,
    "control_jerk": False,
    "stuck_steps": False,
    "planner_compute_ms_mean": False,
    "paper_total_rollouts_mean": False,
}


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_run(directory):
    directory = Path(directory).resolve()
    provenance = json.loads(
        (directory / "provenance.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (directory / "paired_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    fixed = _read_csv(directory / "fixed_episodes.csv")
    adaptive = _read_csv(directory / "adaptive_episodes.csv")
    return {
        "directory": str(directory),
        "provenance": provenance,
        "comparison": comparison,
        "fixed": fixed,
        "adaptive": adaptive,
    }


def _validate_runs(development, confirmation):
    immutable = (
        "actor_checkpoint_sha256",
        "icode_checkpoints",
        "calibration_summary_sha256",
        "calibration_config_sha256",
        "scenes",
        "physics_domains",
        "total_rollouts",
        "iterations",
    )
    changed = [
        key for key in immutable
        if development["provenance"].get(key)
        != confirmation["provenance"].get(key)
    ]
    if changed:
        raise ValueError(
            "development/confirmation differ in %s" % ", ".join(changed)
        )
    development_seeds = {
        int(value) for value in development["provenance"]["seeds"]
    }
    confirmation_seeds = {
        int(value) for value in confirmation["provenance"]["seeds"]
    }
    overlap = development_seeds & confirmation_seeds
    if overlap:
        raise ValueError(
            "development and confirmation seeds overlap: %s"
            % sorted(overlap)
        )
    for run in (development, confirmation):
        if len(run["fixed"]) != len(run["adaptive"]):
            raise ValueError("fixed/adaptive episode counts differ")
        if not run["comparison"]["gate"][
            "equal_model_rollout_budget"
        ]:
            raise ValueError("a source run violated equal rollout budget")
    return {
        "development_seeds": sorted(development_seeds),
        "confirmation_seeds": sorted(confirmation_seeds),
        "combined_seeds": sorted(
            development_seeds | confirmation_seeds
        ),
        "immutable_fields": list(immutable),
    }


def _confirmation_gate(comparison, tolerance=1e-12):
    metrics = comparison["metrics"]
    favorable_primary_intervals = {
        name: (
            metrics[name]["ci95"] is not None
            and metrics[name]["ci95"][0] > tolerance
        )
        for name in ("success", "final_goal_distance", "stuck_steps")
    }
    goal = metrics["final_goal_distance"]
    jerk = metrics["control_jerk"]
    success = metrics["success"]
    collision = metrics["collision"]
    return {
        "primary_direction_favorable": bool(
            goal["favorable_effect"] > tolerance
            or success["favorable_effect"] > tolerance
            or metrics["stuck_steps"]["favorable_effect"] > tolerance
        ),
        "success_not_worse": bool(
            success["favorable_effect"] >= -tolerance
        ),
        "zero_collision_regression": bool(
            collision["control_mean"] == 0.0
            and collision["aligned_mean"] == 0.0
        ),
        "jerk_within_one_percent_margin": bool(
            jerk["favorable_effect"]
            >= -0.01 * max(abs(jerk["control_mean"]), tolerance)
        ),
        "favorable_primary_bootstrap_interval": bool(
            any(favorable_primary_intervals.values())
        ),
        "favorable_primary_intervals": favorable_primary_intervals,
    }


def _stratified_effects(fixed_rows, adaptive_rows, bootstrap, seed):
    fixed_by = defaultdict(list)
    adaptive_by = defaultdict(list)
    for row in fixed_rows:
        fixed_by[(row["scene"], row["physics_domain"])].append(row)
    for row in adaptive_rows:
        adaptive_by[(row["scene"], row["physics_domain"])].append(row)
    if set(fixed_by) != set(adaptive_by):
        raise ValueError("stratified fixed/adaptive cells differ")
    rows = []
    for index, key in enumerate(sorted(fixed_by)):
        result = paired_checkpoint_effects(
            fixed_by[key],
            adaptive_by[key],
            method="gate3_hss",
            metrics=METRICS,
            bootstrap_samples=bootstrap,
            seed=seed + index,
        )
        rows.append({
            "scene": key[0],
            "physics_domain": key[1],
            "paired_seeds": result["independent_clusters"],
            "goal_distance_fixed": result["metrics"][
                "final_goal_distance"
            ]["control_mean"],
            "goal_distance_adaptive": result["metrics"][
                "final_goal_distance"
            ]["aligned_mean"],
            "goal_distance_favorable_effect": result["metrics"][
                "final_goal_distance"
            ]["favorable_effect"],
            "jerk_favorable_effect": result["metrics"][
                "control_jerk"
            ]["favorable_effect"],
            "stuck_favorable_effect": result["metrics"][
                "stuck_steps"
            ]["favorable_effect"],
        })
    return rows


def _write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(development_dir, confirmation_dir, output_dir, bootstrap, seed):
    development = _load_run(development_dir)
    confirmation = _load_run(confirmation_dir)
    audit = _validate_runs(development, confirmation)
    combined_fixed = development["fixed"] + confirmation["fixed"]
    combined_adaptive = development["adaptive"] + confirmation["adaptive"]
    combined = paired_checkpoint_effects(
        combined_fixed,
        combined_adaptive,
        method="gate3_hss",
        metrics=METRICS,
        bootstrap_samples=bootstrap,
        seed=seed,
    )
    confirmation_gate = _confirmation_gate(
        confirmation["comparison"]
    )
    confirmation_gate["confirmation_passed"] = bool(
        all(
            value
            for key, value in confirmation_gate.items()
            if key != "favorable_primary_intervals"
        )
    )
    strata = _stratified_effects(
        combined_fixed,
        combined_adaptive,
        bootstrap,
        seed + 100,
    )
    adaptive_low = float(np.mean([
        float(row["reliability_low_fraction"])
        for row in combined_adaptive
    ]))
    adaptive_medium = float(np.mean([
        float(row["reliability_medium_fraction"])
        for row in combined_adaptive
    ]))
    adaptive_high = float(np.mean([
        float(row["reliability_high_fraction"])
        for row in combined_adaptive
    ]))
    summary = {
        "schema_version": 1,
        "audit": audit,
        "development": development["comparison"],
        "confirmation": confirmation["comparison"],
        "confirmation_gate": confirmation_gate,
        "combined": combined,
        "stratified_effects": strata,
        "combined_authority_step_fractions": {
            "low": adaptive_low,
            "medium": adaptive_medium,
            "high": adaptive_high,
        },
        "interpretation_boundary": (
            "Fixed versus reliability-adaptive HSS with identical 100-rollout "
            "budget, fixed value-aligned ICODE ensemble and frozen SAC "
            "Actor/critic. Both arms had zero success in the 180-step horizon; "
            "results support progress, stuck, smoothness and compute claims, "
            "not a success-rate claim."
        ),
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate3_hss_analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "stratified_effects.csv", strata)
    print(json.dumps({
        "confirmation_gate": confirmation_gate,
        "combined_metrics": combined["metrics"],
        "authority_step_fractions": summary[
            "combined_authority_step_fractions"
        ],
    }, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-dir", required=True)
    parser.add_argument("--confirmation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    analyze(
        args.development_dir,
        args.confirmation_dir,
        args.output_dir,
        args.bootstrap_samples,
        args.seed,
    )


if __name__ == "__main__":
    main()

