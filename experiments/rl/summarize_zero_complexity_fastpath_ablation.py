#!/usr/bin/env python3
"""Merge and gate the preregistered L27 fast-path ablation."""

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

from mobile_robot_mppi.core.config import load_yaml
from experiments.rl.run_scene_complexity_gate_ablation import _resolved_path, _write_csv
from experiments.rl.run_scene_complexity_sample_efficiency import _scene_entries
from experiments.rl.run_zero_complexity_fastpath_ablation import CONDITIONS


def _read(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _float(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError("non-finite %s" % name)
    return value


def _episode_key(row):
    return (int(row["training_seed"]), row["scene"], int(row["episode_seed"]), row["condition"])


def _step_key(row):
    return (int(row["training_seed"]), row["scene"], int(row["episode_seed"]), row["condition"], int(row["step"]))


def _behavior_equivalence(episodes, steps):
    episode_lookup = {_episode_key(row): row for row in episodes}
    episode_groups = sorted({key[:-1] for key in episode_lookup})
    success_mismatches = 0
    collision_mismatches = 0
    for group in episode_groups:
        standard = episode_lookup[group + ("standard_complexity",)]
        fast = episode_lookup[group + ("zero_complexity_fastpath",)]
        success_mismatches += int(_bool(standard["success"]) != _bool(fast["success"]))
        collision_mismatches += int(_bool(standard["collision"]) != _bool(fast["collision"]))

    lookup = {_step_key(row): row for row in steps}
    step_groups = sorted({key[:-2] for key in lookup})
    fields = ("executed_v", "executed_omega", "goal_distance", "rl_gate_alpha")
    max_abs = {name: 0.0 for name in fields}
    behavior_mismatches = 0
    length_mismatches = 0
    compared = 0
    for group in step_groups:
        standard_steps = sorted(key[-1] for key in lookup if key[:-2] == group and key[-2] == "standard_complexity")
        fast_steps = sorted(key[-1] for key in lookup if key[:-2] == group and key[-2] == "zero_complexity_fastpath")
        if standard_steps != fast_steps:
            length_mismatches += 1
            continue
        for step in standard_steps:
            standard = lookup[group + ("standard_complexity", step)]
            fast = lookup[group + ("zero_complexity_fastpath", step)]
            compared += 1
            for name in fields:
                delta = abs(_float(standard, name) - _float(fast, name))
                max_abs[name] = max(max_abs[name], delta)
                behavior_mismatches += int(delta != 0.0)
            for name in ("collision", "safety_override"):
                behavior_mismatches += int(_bool(standard[name]) != _bool(fast[name]))
    return {
        "episode_success_mismatches": success_mismatches,
        "episode_collision_mismatches": collision_mismatches,
        "step_behavior_mismatches": behavior_mismatches,
        "step_length_mismatch_groups": length_mismatches,
        "steps_compared": compared,
        "max_abs_differences": max_abs,
    }


def _timing(episodes, steps):
    output = {}
    for role in ("control", "blocking"):
        role_result = {}
        for condition in CONDITIONS:
            rows = [row for row in episodes if row["scene_role"] == role and row["condition"] == condition]
            step_rows = [row for row in steps if row["scene_role"] == role and row["condition"] == condition]
            role_result[condition] = {
                "episodes": len(rows),
                "planner_compute_ms_mean": float(np.mean([_float(row, "planner_compute_ms_mean") for row in rows])),
                "inference_skip_fraction": float(np.mean([_float(row, "rl_learned_inference_skipped") for row in step_rows])),
            }
        standard = role_result["standard_complexity"]["planner_compute_ms_mean"]
        fast = role_result["zero_complexity_fastpath"]["planner_compute_ms_mean"]
        role_result["time_reduction_fraction"] = float(1.0 - fast / standard)
        output[role] = role_result
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dirs", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    base = load_yaml(_resolved_path(args.config))
    design = base["rl"]["zero_complexity_fastpath_ablation"]
    directories = [_resolved_path(value.strip()) for value in args.input_dirs.split(",") if value.strip()]
    episodes, steps, metadata = [], [], []
    for directory in directories:
        episodes.extend(_read(directory / "episodes.csv"))
        steps.extend(_read(directory / "gate_steps.csv"))
        metadata.append(json.loads((directory / "metadata.json").read_text(encoding="utf-8")))

    keys = [_episode_key(row) for row in episodes]
    training_seeds = sorted({int(row["training_seed"]) for row in episodes})
    expected = len(training_seeds) * len(design["development_episode_seeds"]) * len(_scene_entries(base)) * len(CONDITIONS)
    audit = {
        "expected_episodes": expected,
        "observed_episodes": len(episodes),
        "duplicate_episode_keys": len(keys) - len(set(keys)),
        "protected_seeds_used": sorted({value for item in metadata for value in item["protected_seeds_used"]}),
        "complete": len(episodes) == expected and len(keys) == len(set(keys)),
    }
    behavior = _behavior_equivalence(episodes, steps)
    timing = _timing(episodes, steps)
    checks = {
        "complete": audit["complete"],
        "protected_seeds_untouched": audit["protected_seeds_used"] == [],
        "step_behavior_exact": behavior["step_behavior_mismatches"] <= int(design["maximum_behavior_mismatch_steps"]) and behavior["step_length_mismatch_groups"] == 0,
        "episode_success_exact": behavior["episode_success_mismatches"] <= int(design["maximum_success_mismatches"]),
        "episode_collision_exact": behavior["episode_collision_mismatches"] <= int(design["maximum_collision_mismatches"]),
        "control_skip_fraction": timing["control"]["zero_complexity_fastpath"]["inference_skip_fraction"] >= float(design["minimum_control_skip_fraction"]),
        "blocking_skip_fraction": timing["blocking"]["zero_complexity_fastpath"]["inference_skip_fraction"] >= float(design["minimum_blocking_skip_fraction"]),
        "control_time_reduction": timing["control"]["time_reduction_fraction"] >= float(design["minimum_control_time_reduction_fraction"]),
        "blocking_time_reduction": timing["blocking"]["time_reduction_fraction"] >= float(design["minimum_blocking_time_reduction_fraction"]),
    }
    gate = {"passed": all(checks.values()), "checks": checks, "behavior": behavior, "timing": timing, "decision": "development_pass" if all(checks.values()) else "development_fail"}
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "episodes.csv", episodes)
    _write_csv(output / "gate_steps.csv", steps)
    for name, data in (("audit.json", audit), ("development_gate.json", gate)):
        (output / name).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"audit": audit, "development_gate": gate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
