#!/usr/bin/env python3
"""Merge, audit and gate the preregistered L26 K-factorial."""

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
from experiments.rl.run_scene_complexity_gate_ablation import _resolved_path, _write_csv
from experiments.rl.run_scene_complexity_sample_efficiency import CONDITIONS, _scene_entries


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _boolean(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _float(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError("non-finite %s in L26 row" % name)
    return value


def _episode_key(row):
    return (
        int(row["training_seed"]),
        str(row["scene"]),
        int(row["episode_seed"]),
        int(row["num_samples"]),
        str(row["condition"]),
    )


def _pair_rows(episodes):
    lookup = {_episode_key(row): row for row in episodes}
    groups = sorted({key[:-1] for key in lookup})
    paired = []
    for group in groups:
        traditional = lookup[group + ("traditional_mppi",)]
        candidate = lookup[group + ("complexity_lcb",)]
        row = dict(candidate)
        row.update({
            "traditional_success": _boolean(traditional["success"]),
            "traditional_collision": _boolean(traditional["collision"]),
            "traditional_final_goal_distance": _float(
                traditional, "final_goal_distance"
            ),
            "traditional_success_lost": bool(
                _boolean(traditional["success"])
                and not _boolean(candidate["success"])
            ),
            "traditional_success_gained": bool(
                not _boolean(traditional["success"])
                and _boolean(candidate["success"])
            ),
            "traditional_collision_regression": bool(
                not _boolean(traditional["collision"])
                and _boolean(candidate["collision"])
            ),
            "traditional_distance_improvement": float(
                _float(traditional, "final_goal_distance")
                - _float(candidate, "final_goal_distance")
            ),
            "traditional_planner_compute_ms_mean": _float(
                traditional, "planner_compute_ms_mean"
            ),
        })
        paired.append(row)
    return paired


def _step_key(row):
    return (
        int(row["training_seed"]),
        str(row["scene"]),
        int(row["episode_seed"]),
        int(row["num_samples"]),
        str(row["condition"]),
        int(row["step"]),
    )


def _clean_exact_fallback(steps):
    selected = [row for row in steps if row["scene_role"] == "control"]
    lookup = {_step_key(row): row for row in selected}
    groups = sorted({key[:-2] for key in lookup})
    fields = ("executed_v", "executed_omega", "goal_distance")
    max_abs = {field: 0.0 for field in fields}
    boolean_mismatches = 0
    alpha_max = 0.0
    missing = 0
    compared = 0
    for group in groups:
        traditional_steps = sorted(
            key[-1] for key in lookup
            if key[:-2] == group and key[-2] == "traditional_mppi"
        )
        gated_steps = sorted(
            key[-1] for key in lookup
            if key[:-2] == group and key[-2] == "complexity_lcb"
        )
        if traditional_steps != gated_steps:
            missing += 1
            continue
        for step in traditional_steps:
            traditional = lookup[group + ("traditional_mppi", step)]
            gated = lookup[group + ("complexity_lcb", step)]
            compared += 1
            for field in fields:
                max_abs[field] = max(
                    max_abs[field], abs(_float(traditional, field) - _float(gated, field))
                )
            for field in ("collision", "safety_override"):
                boolean_mismatches += int(
                    _boolean(traditional[field]) != _boolean(gated[field])
                )
            alpha_max = max(alpha_max, abs(_float(gated, "rl_gate_alpha")))
    exact = (
        compared > 0
        and missing == 0
        and boolean_mismatches == 0
        and alpha_max == 0.0
        and all(value == 0.0 for value in max_abs.values())
    )
    return {
        "exact": exact,
        "steps_compared": compared,
        "missing_or_length_mismatch_groups": missing,
        "boolean_mismatches": boolean_mismatches,
        "gate_alpha_max_abs": alpha_max,
        "max_abs_differences": max_abs,
    }


def _condition_summary(episodes):
    grouped = defaultdict(list)
    for row in episodes:
        grouped[(row["scene_role"], row["scene"], int(row["num_samples"]), row["condition"])].append(row)
    records = []
    for (role, scene, num_samples, condition), rows in sorted(grouped.items()):
        records.append({
            "scene_role": role,
            "scene": scene,
            "num_samples": num_samples,
            "condition": condition,
            "episodes": len(rows),
            "training_seeds": len({int(row["training_seed"]) for row in rows}),
            "successes": int(sum(_boolean(row["success"]) for row in rows)),
            "success_rate": float(np.mean([_boolean(row["success"]) for row in rows])),
            "collisions": int(sum(_boolean(row["collision"]) for row in rows)),
            "final_goal_distance_mean_m": float(np.mean([
                _float(row, "final_goal_distance") for row in rows
            ])),
            "planner_compute_ms_mean": float(np.mean([
                _float(row, "planner_compute_ms_mean") for row in rows
            ])),
            "planner_compute_ms_max": float(np.max([
                _float(row, "planner_compute_ms_max") for row in rows
            ])),
            "gate_alpha_mean": float(np.mean([
                _float(row, "rl_gate_alpha_mean") for row in rows
            ])),
        })
    return records


def _development_gate(episodes, paired, steps, design, audit):
    low_counts = {int(value) for value in design["low_budget_sample_counts"]}
    blocking_low = [
        row for row in paired
        if row["scene_role"] == "blocking"
        and int(row["num_samples"]) in low_counts
    ]
    low_gain = int(sum(_boolean(row["traditional_success_gained"]) for row in blocking_low))
    low_loss = int(sum(_boolean(row["traditional_success_lost"]) for row in blocking_low))
    low_net = low_gain - low_loss
    collision_regressions = int(sum(
        _boolean(row["traditional_collision_regression"]) for row in paired
    ))

    low_k = int(design["comparison_low_sample_count"])
    high_k = int(design["comparison_high_sample_count"])
    scene_names = sorted({row["scene"] for row in episodes if row["scene_role"] == "blocking"})
    comparisons = []
    matching = 0
    for scene in scene_names:
        gated = [
            row for row in episodes
            if row["scene"] == scene
            and row["condition"] == "complexity_lcb"
            and int(row["num_samples"]) == low_k
        ]
        traditional = [
            row for row in episodes
            if row["scene"] == scene
            and row["condition"] == "traditional_mppi"
            and int(row["num_samples"]) == high_k
        ]
        gated_rate = float(np.mean([_boolean(row["success"]) for row in gated]))
        traditional_rate = float(np.mean([_boolean(row["success"]) for row in traditional]))
        shortfall = traditional_rate - gated_rate
        match = shortfall <= float(design["maximum_low_vs_high_success_rate_shortfall"])
        matching += int(match)
        comparisons.append({
            "scene": scene,
            "gated_low_k": low_k,
            "gated_low_success_rate": gated_rate,
            "traditional_high_k": high_k,
            "traditional_high_success_rate": traditional_rate,
            "success_rate_shortfall": shortfall,
            "matches_within_margin": match,
        })

    gated_low_all = [
        row for row in episodes
        if row["scene_role"] == "blocking"
        and row["condition"] == "complexity_lcb"
        and int(row["num_samples"]) == low_k
    ]
    traditional_high_all = [
        row for row in episodes
        if row["scene_role"] == "blocking"
        and row["condition"] == "traditional_mppi"
        and int(row["num_samples"]) == high_k
    ]
    planner_ratio = float(np.mean([
        _float(row, "planner_compute_ms_mean") for row in gated_low_all
    ]) / np.mean([
        _float(row, "planner_compute_ms_mean") for row in traditional_high_all
    ]))
    fallback = _clean_exact_fallback(steps)
    checks = {
        "complete_factorial": bool(audit["complete_factorial"]),
        "no_duplicate_episode_keys": audit["duplicate_episode_keys"] == 0,
        "sealed_seeds_untouched": (
            audit["sealed_l25_seeds_used"] == []
            and audit["sealed_l26_seeds_used"] == []
        ),
        "clean_exact_fallback": bool(fallback["exact"]),
        "no_collision_regression": collision_regressions <= int(
            design["maximum_collision_regressions"]
        ),
        "low_budget_net_success_gain": low_net >= int(
            design["minimum_low_budget_blocking_net_success_gain"]
        ),
        "low_k_matches_high_k_in_enough_scenes": matching >= int(
            design["minimum_blocking_scenes_matching_high_budget_traditional"]
        ),
        "low_k_compute_ratio": planner_ratio <= float(
            design["maximum_low_vs_high_planner_time_ratio"]
        ),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "low_budget_blocking_success_gains": low_gain,
        "low_budget_blocking_success_losses": low_loss,
        "low_budget_blocking_net_success_gain": low_net,
        "collision_regressions": collision_regressions,
        "low_vs_high_scene_comparisons": comparisons,
        "blocking_scenes_matching_high_budget_traditional": matching,
        "low_vs_high_planner_time_ratio": planner_ratio,
        "clean_exact_fallback": fallback,
        "decision": (
            "development_pass_keep_L26_test_sealed_until_explicit_confirmatory_run"
            if passed
            else "development_fail_keep_all_test_seeds_sealed"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dirs", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    config_path = _resolved_path(args.config)
    base = load_yaml(config_path)
    design = base["rl"]["sample_efficiency_ablation"]
    input_dirs = [_resolved_path(value.strip()) for value in args.input_dirs.split(",") if value.strip()]
    if not input_dirs:
        raise ValueError("L26 requires at least one input directory")

    episodes = []
    steps = []
    metadata = []
    for directory in input_dirs:
        episodes.extend(_read_csv(directory / "episodes.csv"))
        steps.extend(_read_csv(directory / "gate_steps.csv"))
        with (directory / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata.append(json.load(handle))

    episode_keys = [_episode_key(row) for row in episodes]
    duplicate_keys = len(episode_keys) - len(set(episode_keys))
    scenes = _scene_entries(base)
    training_seeds = sorted({int(row["training_seed"]) for row in episodes})
    expected = (
        len(training_seeds)
        * len(design["development_episode_seeds"])
        * len(scenes)
        * len(design["sample_counts"])
        * len(CONDITIONS)
    )
    observed_keys = set(episode_keys)
    expected_keys = {
        (training_seed, scene["name"], int(seed), int(num_samples), condition)
        for training_seed in training_seeds
        for scene in scenes
        for seed in design["development_episode_seeds"]
        for num_samples in design["sample_counts"]
        for condition in CONDITIONS
    }
    audit = {
        "training_seeds": training_seeds,
        "expected_episodes": expected,
        "observed_episodes": len(episodes),
        "unique_episode_keys": len(observed_keys),
        "duplicate_episode_keys": duplicate_keys,
        "missing_episode_keys": len(expected_keys - observed_keys),
        "unexpected_episode_keys": len(observed_keys - expected_keys),
        "complete_factorial": (
            len(episodes) == expected
            and duplicate_keys == 0
            and observed_keys == expected_keys
        ),
        "sealed_l25_seeds_used": sorted({
            value for item in metadata for value in item["sealed_l25_seeds_used"]
        }),
        "sealed_l26_seeds_used": sorted({
            value for item in metadata for value in item["sealed_l26_seeds_used"]
        }),
        "finite_episode_metrics": all(
            math.isfinite(float(row[name]))
            for row in episodes
            for name in (
                "final_goal_distance",
                "planner_compute_ms_mean",
                "planner_compute_ms_max",
                "rl_gate_alpha_mean",
            )
        ),
        "run_git_shas": sorted({str(item["run_git_sha"]) for item in metadata}),
        "checkpoint_sha256": sorted({
            str(item["candidate_checkpoint_sha256"]) for item in metadata
        }),
    }
    if not audit["finite_episode_metrics"]:
        raise ValueError("L26 contains non-finite episode metrics")
    if not audit["complete_factorial"]:
        raise ValueError("L26 factorial is incomplete: %s" % audit)

    paired = _pair_rows(episodes)
    summary = _condition_summary(episodes)
    gate = _development_gate(episodes, paired, steps, design, audit)
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "episodes.csv", episodes)
    _write_csv(output / "paired_episodes.csv", paired)
    _write_csv(output / "gate_steps.csv", steps)
    _write_csv(output / "summary.csv", summary)
    for name, data in (
        ("audit.json", audit),
        ("development_gate.json", gate),
        ("summary.json", {
            "study": "L26_sample_efficiency",
            "config": str(config_path),
            "input_dirs": [str(value) for value in input_dirs],
            "run_git_sha": git_sha(ROOT),
            "summary": summary,
            "development_gate": gate,
            "interpretation_guard": (
                "development-only result; episode/K rows are paired within each of "
                "three independent policy checkpoints"
            ),
        }),
    ):
        with (output / name).open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"audit": audit, "development_gate": gate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
