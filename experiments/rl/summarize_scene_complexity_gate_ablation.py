#!/usr/bin/env python3
"""Audit and summarize multiple L25 training-seed runs."""

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

from mobile_robot_mppi.core.config import git_sha, load_yaml
from experiments.rl.run_scene_complexity_gate_ablation import (
    CONDITIONS,
    _resolved_path,
    _write_csv,
)


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _boolean(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _float(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError("non-finite %s in L25 row" % name)
    return value


def _episode_key(row):
    return (
        int(row["training_seed"]),
        str(row["scene"]),
        int(row["episode_seed"]),
        str(row["condition"]),
    )


def _pair_rows(episodes):
    lookup = {_episode_key(row): row for row in episodes}
    groups = sorted({key[:3] for key in lookup})
    paired = []
    for group in groups:
        references = {
            condition: lookup[group + (condition,)] for condition in CONDITIONS
        }
        traditional = references["traditional_mppi"]
        bc = references["frozen_bc_prior"]
        for condition in ("frozen_bc_prior", "lcb_always", "complexity_lcb"):
            candidate = references[condition]
            row = dict(candidate)
            for prefix, reference in (("traditional", traditional), ("bc", bc)):
                row[prefix + "_success"] = _boolean(reference["success"])
                row[prefix + "_collision"] = _boolean(reference["collision"])
                row[prefix + "_final_goal_distance"] = _float(
                    reference, "final_goal_distance"
                )
                row[prefix + "_success_lost"] = bool(
                    _boolean(reference["success"])
                    and not _boolean(candidate["success"])
                )
                row[prefix + "_success_gained"] = bool(
                    not _boolean(reference["success"])
                    and _boolean(candidate["success"])
                )
                row[prefix + "_collision_regression"] = bool(
                    not _boolean(reference["collision"])
                    and _boolean(candidate["collision"])
                )
                row[prefix + "_distance_improvement"] = float(
                    _float(reference, "final_goal_distance")
                    - _float(candidate, "final_goal_distance")
                )
            paired.append(row)
    return paired


def _condition_summary(rows):
    output = {}
    for (role, scene, condition), group in sorted(rows.items()):
        clearance = [
            _float(row, "minimum_clearance") for row in group
            if str(row.get("minimum_clearance", "")) not in ("", "None")
        ]
        output.setdefault(role, {}).setdefault(scene, {})[condition] = {
            "episodes": len(group),
            "training_seeds": len({int(row["training_seed"]) for row in group}),
            "successes": int(sum(_boolean(row["success"]) for row in group)),
            "collisions": int(sum(_boolean(row["collision"]) for row in group)),
            "final_goal_distance_mean_m": float(np.mean([
                _float(row, "final_goal_distance") for row in group
            ])),
            "minimum_clearance_mean_m": (
                float(np.mean(clearance)) if clearance else None
            ),
            "control_jerk_mean": float(np.mean([
                _float(row, "control_jerk") for row in group
            ])),
            "planner_compute_ms_mean": float(np.mean([
                _float(row, "planner_compute_ms_mean") for row in group
            ])),
            "gate_alpha_episode_mean": float(np.mean([
                _float(row, "rl_gate_alpha_mean") for row in group
            ])),
            "gate_active_episode_mean": float(np.mean([
                _float(row, "rl_gate_active_fraction") for row in group
            ])),
        }
    return output


def _pooled_gate_metrics(steps, role, condition, epsilon):
    selected = [
        row for row in steps
        if row["scene_role"] == role and row["condition"] == condition
    ]
    if not selected:
        raise ValueError("missing L25 gate steps for %s/%s" % (role, condition))
    alpha = np.asarray([
        _float(row, "rl_gate_alpha") for row in selected
    ], dtype=np.float64)
    score = np.asarray([
        _float(row, "rl_scene_complexity_score") for row in selected
    ], dtype=np.float64)
    return {
        "steps": int(alpha.size),
        "mean_alpha": float(alpha.mean()),
        "active_fraction": float(np.mean(alpha > float(epsilon))),
        "score_mean": float(score.mean()),
        "score_max": float(score.max()),
    }


def _development_gate(episodes, steps, design):
    epsilon = float(design["activation_epsilon"])
    simple = _pooled_gate_metrics(steps, "simple", "complexity_lcb", epsilon)
    complex_values = _pooled_gate_metrics(
        steps, "complex", "complexity_lcb", epsilon
    )
    always_steps = [
        row for row in steps if row["condition"] == "lcb_always"
    ]
    always_alpha = float(np.mean([
        _float(row, "rl_gate_alpha") for row in always_steps
    ]))
    complexity_steps = [
        row for row in steps if row["condition"] == "complexity_lcb"
    ]
    complexity_alpha = float(np.mean([
        _float(row, "rl_gate_alpha") for row in complexity_steps
    ]))
    alpha_reduction = (
        1.0 - complexity_alpha / always_alpha if always_alpha > 0.0 else 0.0
    )
    candidate = [
        row for row in _pair_rows(episodes)
        if row["condition"] == "complexity_lcb"
    ]
    simple_pairs = [row for row in candidate if row["scene_role"] == "simple"]
    complex_pairs = [row for row in candidate if row["scene_role"] == "complex"]
    simple_losses = int(sum(_boolean(row["traditional_success_lost"]) for row in simple_pairs))
    collision_regressions = int(sum(
        _boolean(row["traditional_collision_regression"]) for row in candidate
    ))
    complex_gains = int(sum(_boolean(row["traditional_success_gained"]) for row in complex_pairs))
    complex_losses = int(sum(_boolean(row["traditional_success_lost"]) for row in complex_pairs))
    complex_mean_distance = float(np.mean([
        _float(row, "traditional_distance_improvement") for row in complex_pairs
    ]))
    checks = {
        "simple_mean_alpha": simple["mean_alpha"]
        <= float(design["maximum_simple_mean_gate_alpha"]),
        "simple_active_fraction": simple["active_fraction"]
        <= float(design["maximum_simple_active_fraction"]),
        "complex_mean_alpha": complex_values["mean_alpha"]
        >= float(design["minimum_complex_mean_gate_alpha"]),
        "complex_active_fraction_lower": complex_values["active_fraction"]
        >= float(design["minimum_complex_active_fraction"]),
        "complex_active_fraction_upper": complex_values["active_fraction"]
        <= float(design["maximum_complex_active_fraction"]),
        "alpha_reduction_vs_always": alpha_reduction
        >= float(design["minimum_alpha_reduction_vs_always"]),
        "simple_success_noninferiority": simple_losses
        <= int(design["maximum_simple_success_losses_vs_traditional"]),
        "no_collision_regression": collision_regressions
        <= int(design["maximum_collision_regressions_vs_traditional"]),
        "complex_net_success_gain": (complex_gains - complex_losses)
        >= int(design["minimum_complex_net_success_gain_vs_traditional"]),
        "complex_distance_improvement": complex_mean_distance
        >= float(design["minimum_complex_mean_distance_improvement_m"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "simple_gate": simple,
        "complex_gate": complex_values,
        "always_mean_alpha": always_alpha,
        "complexity_mean_alpha": complexity_alpha,
        "alpha_reduction_vs_always": alpha_reduction,
        "simple_success_losses_vs_traditional": simple_losses,
        "collision_regressions_vs_traditional": collision_regressions,
        "complex_success_gains_vs_traditional": complex_gains,
        "complex_success_losses_vs_traditional": complex_losses,
        "complex_net_success_gain_vs_traditional": complex_gains - complex_losses,
        "complex_mean_distance_improvement_m": complex_mean_distance,
        "decision": (
            "eligible_to_open_sealed_test"
            if all(checks.values())
            else "retain_traditional_mppi_fallback_and_keep_test_sealed"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dirs", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    base = load_yaml(_resolved_path(args.config))
    design = base["rl"]["scene_complexity_ablation"]
    inputs = [_resolved_path(value) for value in args.input_dirs.split(",") if value.strip()]
    if not inputs:
        raise ValueError("L25 summary requires at least one input directory")
    episodes = []
    steps = []
    metadata = []
    for directory in inputs:
        episodes.extend(_read_csv(directory / "episodes.csv"))
        steps.extend(_read_csv(directory / "gate_steps.csv"))
        with (directory / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata.append(json.load(handle))

    episode_keys = [_episode_key(row) for row in episodes]
    duplicates = len(episode_keys) - len(set(episode_keys))
    training_seeds = sorted({int(row["training_seed"]) for row in episodes})
    expected_scenes = {}
    for role, key in (("simple", "simple_scenes"), ("complex", "complex_scenes")):
        for path in design[key]:
            scene = load_yaml(_resolved_path(path))
            expected_scenes[str(scene["scene"]["name"])] = role
    expected_episode_seeds = {
        int(value) for value in design["development_episode_seeds"]
    }
    expected_keys = {
        (training_seed, scene, episode_seed, condition)
        for training_seed in training_seeds
        for scene in expected_scenes
        for episode_seed in expected_episode_seeds
        for condition in CONDITIONS
    }
    actual_keys = set(episode_keys)
    sealed = {int(value) for value in design["sealed_test_episode_seeds"]}
    sealed_used = sorted({int(row["episode_seed"]) for row in episodes} & sealed)
    audit_checks = {
        "no_duplicate_episode_keys": duplicates == 0,
        "exact_preregistered_coverage": actual_keys == expected_keys,
        "all_conditions_present": set(row["condition"] for row in episodes) == set(CONDITIONS),
        "all_scenes_present": set(row["scene"] for row in episodes) == set(expected_scenes),
        "sealed_test_unopened": not sealed_used,
        "all_episode_metrics_finite": all(
            math.isfinite(float(row[name]))
            for row in episodes
            for name in (
                "final_goal_distance",
                "trajectory_length",
                "control_jerk",
                "planner_compute_ms_mean",
                "rl_gate_alpha_mean",
            )
        ),
        "all_gate_steps_finite": all(
            math.isfinite(float(row[name]))
            for row in steps
            for name in (
                "rl_gate_alpha",
                "rl_scene_complexity_score",
                "goal_distance",
            )
        ),
    }
    if not all(audit_checks.values()):
        raise RuntimeError("L25 audit failed: %s" % audit_checks)

    grouped = defaultdict(list)
    for row in episodes:
        grouped[(row["scene_role"], row["scene"], row["condition"])].append(row)
    paired = _pair_rows(episodes)
    gate = _development_gate(episodes, steps, design)
    summary = {
        "experiment": "L25 scene complexity gate development",
        "training_seeds": training_seeds,
        "episode_seeds": sorted(expected_episode_seeds),
        "scenes": expected_scenes,
        "conditions": list(CONDITIONS),
        "episodes": len(episodes),
        "steps": len(steps),
        "condition_summary": _condition_summary(grouped),
        "development_gate": gate,
        "sealed_test_opened": False,
        "run_git_sha": git_sha(ROOT),
        "interpretation_guard": (
            "development eligibility is a deterministic preregistered gate, "
            "not a confirmatory p-value; checkpoint nesting remains explicit"
        ),
    }
    audit = {
        "checks": audit_checks,
        "duplicates": duplicates,
        "expected_episode_rows": len(expected_keys),
        "actual_episode_rows": len(episodes),
        "sealed_test_seeds_used": sealed_used,
        "input_metadata": metadata,
    }
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "episodes.csv", episodes)
    _write_csv(output / "paired_episodes.csv", paired)
    _write_csv(output / "gate_steps.csv", steps)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    with (output / "development_gate.json").open("w", encoding="utf-8") as handle:
        json.dump(gate, handle, indent=2, sort_keys=True, allow_nan=False)
    with (output / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
