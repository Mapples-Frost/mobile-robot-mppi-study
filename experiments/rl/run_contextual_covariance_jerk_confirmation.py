#!/usr/bin/env python3
"""Single-process three-arm confirmation of half-budget jerk remediation."""

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rl.evaluate_rl_sampling_prior import _apply_scene_config
from experiments.rl.run_contextual_covariance_jerk_screening import (
    _audit_rows,
    _blocked_schedule,
    _descriptives,
    _paired_contrast,
    _write_csv,
)
from mobile_robot_mppi.core.config import deep_merge, load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _half_budget_gate(contrast, rmse_margin):
    result = dict(contrast)
    result["safety_gate_passed"] = bool(
        result["success_delta_mean"] >= 0.0
        and result["collision_delta_mean"] <= 0.0
    )
    result["precision_gate_passed"] = bool(
        result["cross_track_rmse_delta_ci95"][1] <= float(rmse_margin)
    )
    result["elapsed_gate_passed"] = bool(
        result["elapsed_s_delta_ci95"][1] <= 0.0
    )
    result["compute_gate_passed"] = bool(
        result["planner_compute_ms_mean_delta_ci95"][1] < 0.0
    )
    result["gate_passed"] = bool(
        result["safety_gate_passed"]
        and result["precision_gate_passed"]
        and result["elapsed_gate_passed"]
        and result["compute_gate_passed"]
    )
    return result


def _smoothing_gate(contrast, rmse_margin, elapsed_margin):
    result = dict(contrast)
    result["safety_gate_passed"] = bool(
        result["success_delta_mean"] >= 0.0
        and result["collision_delta_mean"] <= 0.0
    )
    result["precision_gate_passed"] = bool(
        result["cross_track_rmse_delta_ci95"][1] <= float(rmse_margin)
    )
    result["elapsed_gate_passed"] = bool(
        result["elapsed_s_delta_ci95"][1] <= float(elapsed_margin)
    )
    result["issued_jerk_gate_passed"] = bool(
        result["control_jerk_delta_ci95"][1] < 0.0
    )
    result["applied_jerk_gate_passed"] = bool(
        result["applied_control_jerk_delta_ci95"][1] < 0.0
    )
    result["gate_passed"] = bool(
        result["safety_gate_passed"]
        and result["precision_gate_passed"]
        and result["elapsed_gate_passed"]
        and result["issued_jerk_gate_passed"]
        and result["applied_jerk_gate_passed"]
    )
    return result


def _complete_package_gate(contrast, rmse_margin):
    result = _half_budget_gate(contrast, rmse_margin)
    result["issued_jerk_gate_passed"] = bool(
        result["control_jerk_delta_ci95"][1] <= 0.0
    )
    result["applied_jerk_gate_passed"] = bool(
        result["applied_control_jerk_delta_ci95"][1] <= 0.0
    )
    result["gate_passed"] = bool(
        result["gate_passed"]
        and result["issued_jerk_gate_passed"]
        and result["applied_jerk_gate_passed"]
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-blocks", type=int)
    args = parser.parse_args(argv)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    base = load_yaml(ROOT / spec["base_config"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    schedule = _blocked_schedule(spec)
    with (output / "schedule.json").open("w", encoding="utf-8") as handle:
        json.dump(schedule, handle, indent=2, sort_keys=True)
    selected_blocks = sorted({int(item["block_index"]) for item in schedule})
    if args.max_blocks is not None:
        selected_blocks = selected_blocks[:int(args.max_blocks)]
    selected_blocks = set(selected_blocks)
    rows = []
    for item in schedule:
        if int(item["block_index"]) not in selected_blocks:
            continue
        arm = item["arm"]
        domain = item["physics_domain"]
        config = _apply_scene_config(base, ROOT / item["scene_path"])
        config = deep_merge(config, {"plant": domain.get("plant_override", {})})
        config = deep_merge(config, arm.get("config_override", {}))
        config["experiment"]["seed"] = int(item["seed"])
        config["experiment"]["name"] = "l97_%s_%s_%s_%d" % (
            config["scene"]["name"], domain["name"], arm["name"], item["seed"]
        )
        config["planner"]["num_samples"] = int(arm["num_samples"])
        if arm["kind"] == "contextual_bandit":
            config["planner"]["sampling_prior"] = "contextual_bandit_covariance"
            config.setdefault("rl", {})["enabled"] = True
            config["rl"]["policy_id"] = "linucb_contextual_covariance_l89"
            config["rl"]["checkpoint"] = spec["checkpoint"]
        elif arm["kind"] == "fixed":
            config["planner"]["sampling_prior"] = "fixed_covariance"
            config["planner"]["fixed_covariance_scale"] = list(
                arm["fixed_covariance_scale"]
            )
            config.setdefault("rl", {})["enabled"] = False
            config["rl"].pop("checkpoint", None)
        else:
            raise ValueError("unknown L97 arm kind")
        run_dir = (
            output / "runs" / domain["name"] / config["scene"]["name"]
            / arm["name"] / ("seed_%d" % item["seed"])
        )
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                episode = json.load(handle)
        else:
            episode = ExperimentRunner(
                copy.deepcopy(config), ROOT, run_dir, headless=True
            ).run().summary
        row = dict(episode)
        row.update({
            "block_index": int(item["block_index"]),
            "run_position": int(item["run_position"]),
            "scene": str(config["scene"]["name"]),
            "physics_domain": str(domain["name"]),
            "condition": str(arm["name"]),
            "num_samples": int(arm["num_samples"]),
            "seed": int(item["seed"]),
            "elapsed_s": float(episode["steps"])
            * float(config["experiment"]["control_dt"]),
        })
        rows.append(row)
        _write_csv(output / "episodes.csv", rows)
    if len(rows) != len(schedule):
        print(json.dumps({
            "complete": False,
            "completed_runs": len(rows),
            "expected_runs": len(schedule),
        }, indent=2))
        return 0
    seed = int(spec["bootstrap_seed"])
    margin = float(spec["cross_track_noninferiority_margin_m"])
    raw_fixed = _half_budget_gate(_paired_contrast(
        rows, "contextual_k50_current", "fixed_k100", seed
    ), margin)
    smooth_raw = _smoothing_gate(_paired_contrast(
        rows, "contextual_k50_smoothed", "contextual_k50_current", seed + 20
    ), margin, float(spec["smoothing_elapsed_noninferiority_margin_s"]))
    smooth_fixed = _complete_package_gate(_paired_contrast(
        rows, "contextual_k50_smoothed", "fixed_k100", seed + 40
    ), margin)
    audit = _audit_rows(rows, spec)
    primary = bool(
        audit["passed"]
        and raw_fixed["gate_passed"]
        and smooth_raw["gate_passed"]
        and smooth_fixed["gate_passed"]
    )
    summary = {
        "schema_version": 1,
        "design_id": spec["design_id"],
        "completed_runs": len(rows),
        "audit": audit,
        "descriptives": _descriptives(rows),
        "contrasts": {
            "raw_contextual_vs_fixed": raw_fixed,
            "smoothed_contextual_vs_raw": smooth_raw,
            "smoothed_contextual_vs_fixed": smooth_fixed,
        },
        "primary_gate_passed": primary,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
