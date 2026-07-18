#!/usr/bin/env python3
"""Blocked ICODE x contextual-sampling composition experiment.

The runner is shared by the L98 development screen and the sealed L99
confirmation.  It deliberately keeps dynamics, sampling and sample budget as
separate factors so the final package cannot hide a regression in one layer.
"""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rl.evaluate_rl_sampling_prior import _apply_scene_config
from experiments.rl.run_contextual_covariance_jerk_screening import _blocked_schedule
from experiments.rl.run_covariance_context_oracle import _hierarchical_ci
from mobile_robot_mppi.core.config import deep_merge, load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


METRICS = (
    "cross_track_rmse",
    "elapsed_s",
    "control_jerk",
    "applied_control_jerk",
    "planner_compute_ms_mean",
    "planner_compute_ms_p95",
    "profile_mppi_batch_rollout_ms_mean",
    "final_goal_distance",
)

# Clean path-tracking scenes contain no obstacle, so clearance is undefined
# rather than infinite or zero.  Keep it in the report without using it in
# contrasts or integrity gates that require a finite numeric value.
OPTIONAL_METRICS = ("minimum_clearance",)


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _configure_arm(config, arm, spec):
    """Apply one factorial arm without changing the plant or safety chain."""

    result = copy.deepcopy(config)
    result = deep_merge(result, spec.get("shared_override", {}))
    mode = str(arm["prediction_mode"])
    if mode == "nominal":
        result["planner"]["prediction_mode"] = "nominal"
        result["planner"].pop("checkpoint", None)
    elif mode == "icode_residual":
        result["planner"]["prediction_mode"] = "icode_residual"
        result["planner"]["checkpoint"] = str(spec["icode_checkpoint"])
        result["planner"]["residual_torch_num_threads"] = 1
        result["planner"]["residual_torchscript"] = True
    else:
        raise ValueError("unknown prediction mode: %s" % mode)

    result["planner"]["num_samples"] = int(arm["num_samples"])
    sampler = str(arm["sampler_kind"])
    result.setdefault("rl", {})
    if sampler == "contextual_bandit":
        result["planner"]["sampling_prior"] = "contextual_bandit_covariance"
        result["rl"]["enabled"] = True
        result["rl"]["policy_id"] = "linucb_contextual_covariance_l89"
        result["rl"]["checkpoint"] = str(spec["bandit_checkpoint"])
    elif sampler == "fixed":
        scale = np.asarray(arm["fixed_covariance_scale"], dtype=np.float64)
        if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("fixed covariance scale must contain two positive values")
        result["planner"]["sampling_prior"] = "fixed_covariance"
        result["planner"]["fixed_covariance_scale"] = scale.tolist()
        result["rl"]["enabled"] = False
        result["rl"].pop("checkpoint", None)
    else:
        raise ValueError("unknown sampler kind: %s" % sampler)
    return result


def _paired_contrast(rows, treatment, reference, bootstrap_seed):
    indexed = {
        (
            str(row["scene"]),
            str(row["physics_domain"]),
            int(row["seed"]),
            str(row["condition"]),
        ): row
        for row in rows
    }
    contexts = sorted({key[:3] for key in indexed})
    differences = {metric: {} for metric in METRICS}
    safety = []
    for scene, domain, seed in contexts:
        treated = indexed[(scene, domain, seed, treatment)]
        control = indexed[(scene, domain, seed, reference)]
        group = scene + "__" + domain
        for metric in METRICS:
            differences[metric].setdefault(group, []).append(
                float(treated[metric]) - float(control[metric])
            )
        safety.append((
            float(treated["success"]) - float(control["success"]),
            float(treated["collision"]) - float(control["collision"]),
        ))
    result = {
        "treatment": str(treatment),
        "reference": str(reference),
        "paired_episodes": len(contexts),
        "success_delta_mean": float(np.mean([value[0] for value in safety])),
        "collision_delta_mean": float(np.mean([value[1] for value in safety])),
    }
    for offset, (metric, grouped) in enumerate(differences.items()):
        flat = [value for values in grouped.values() for value in values]
        result[metric + "_delta_mean"] = float(np.mean(flat))
        result[metric + "_delta_ci95"] = _hierarchical_ci(
            grouped, int(bootstrap_seed) + offset
        )
    return result


def _difference_in_differences(
    rows, treatment_a, reference_a, treatment_b, reference_b, bootstrap_seed
):
    """Estimate whether one paired treatment effect depends on dynamics.

    The calculation stays at the episode-block level.  It does not treat the
    controller steps inside an episode as additional independent observations.
    """

    indexed = {
        (
            str(row["scene"]), str(row["physics_domain"]), int(row["seed"]),
            str(row["condition"]),
        ): row
        for row in rows
    }
    contexts = sorted({key[:3] for key in indexed})
    output = {
        "effect_a": "%s - %s" % (treatment_a, reference_a),
        "effect_b": "%s - %s" % (treatment_b, reference_b),
        "paired_episodes": len(contexts),
    }
    for offset, metric in enumerate(METRICS):
        grouped = {}
        for scene, domain, seed in contexts:
            group = scene + "__" + domain
            first = (
                float(indexed[(scene, domain, seed, treatment_a)][metric])
                - float(indexed[(scene, domain, seed, reference_a)][metric])
            )
            second = (
                float(indexed[(scene, domain, seed, treatment_b)][metric])
                - float(indexed[(scene, domain, seed, reference_b)][metric])
            )
            grouped.setdefault(group, []).append(first - second)
        flat = [value for values in grouped.values() for value in values]
        output[metric + "_mean"] = float(np.mean(flat))
        output[metric + "_ci95"] = _hierarchical_ci(
            grouped, int(bootstrap_seed) + offset
        )
    return output


def _safety_gate(contrast):
    return bool(
        contrast["success_delta_mean"] >= 0.0
        and contrast["collision_delta_mean"] <= 0.0
    )


def _dynamics_gate(contrast, contextual_compute_mean, deadline_ms):
    result = dict(contrast)
    result["safety_gate_passed"] = _safety_gate(result)
    result["tracking_superiority_gate_passed"] = bool(
        result["cross_track_rmse_delta_ci95"][1] < 0.0
    )
    result["absolute_compute_gate_passed"] = bool(
        float(contextual_compute_mean) <= float(deadline_ms)
    )
    result["gate_passed"] = bool(
        result["safety_gate_passed"]
        and result["tracking_superiority_gate_passed"]
        and result["absolute_compute_gate_passed"]
    )
    return result


def _half_budget_gate(contrast, rmse_margin):
    result = dict(contrast)
    result["safety_gate_passed"] = _safety_gate(result)
    result["tracking_noninferiority_gate_passed"] = bool(
        result["cross_track_rmse_delta_ci95"][1] <= float(rmse_margin)
    )
    result["elapsed_superiority_gate_passed"] = bool(
        result["elapsed_s_delta_ci95"][1] <= 0.0
    )
    result["compute_superiority_gate_passed"] = bool(
        result["planner_compute_ms_mean_delta_ci95"][1] < 0.0
    )
    result["gate_passed"] = bool(
        result["safety_gate_passed"]
        and result["tracking_noninferiority_gate_passed"]
        and result["elapsed_superiority_gate_passed"]
        and result["compute_superiority_gate_passed"]
    )
    return result


def _final_package_gate(contrast, package_compute_mean, deadline_ms):
    result = dict(contrast)
    result["safety_gate_passed"] = _safety_gate(result)
    result["tracking_superiority_gate_passed"] = bool(
        result["cross_track_rmse_delta_ci95"][1] < 0.0
    )
    result["absolute_compute_gate_passed"] = bool(
        float(package_compute_mean) <= float(deadline_ms)
    )
    result["gate_passed"] = bool(
        result["safety_gate_passed"]
        and result["tracking_superiority_gate_passed"]
        and result["absolute_compute_gate_passed"]
    )
    return result


def _audit_rows(rows, spec):
    arm_names = {str(arm["name"]) for arm in spec["arms"]}
    expected_blocks = (
        len(spec["scenes"]) * len(spec["physics_domains"]) * len(spec["seeds"])
    )
    expected_rows = expected_blocks * len(arm_names)
    keys = [
        (
            str(row["scene"]), str(row["physics_domain"]), int(row["seed"]),
            str(row["condition"]),
        )
        for row in rows
    ]
    blocks = {}
    for row in rows:
        block = (str(row["scene"]), str(row["physics_domain"]), int(row["seed"]))
        blocks.setdefault(block, set()).add(str(row["condition"]))
    finite = all(
        np.isfinite(float(row[metric])) for row in rows for metric in METRICS
    )
    optional_missing = {
        metric: int(sum(row.get(metric) is None for row in rows))
        for metric in OPTIONAL_METRICS
    }
    optional_nonfinite = {
        metric: int(sum(
            row.get(metric) is not None
            and not np.isfinite(float(row[metric]))
            for row in rows
        ))
        for metric in OPTIONAL_METRICS
    }
    run_positions = {
        name: {
            str(position): sum(
                str(row["condition"]) == name
                and int(row["run_position"]) == position
                for row in rows
            )
            for position in range(len(arm_names))
        }
        for name in sorted(arm_names)
    }
    position_values = [
        count for values in run_positions.values() for count in values.values()
    ]
    balanced_positions = bool(
        position_values and max(position_values) == min(position_values)
    )
    passed = bool(
        len(rows) == expected_rows
        and len(set(keys)) == expected_rows
        and len(blocks) == expected_blocks
        and all(values == arm_names for values in blocks.values())
        and finite
        and balanced_positions
    )
    return {
        "passed": passed,
        "expected_rows": int(expected_rows),
        "observed_rows": int(len(rows)),
        "unique_keys": int(len(set(keys))),
        "expected_blocks": int(expected_blocks),
        "complete_blocks": int(sum(values == arm_names for values in blocks.values())),
        "finite_metric_values": bool(finite),
        "optional_metric_missing_counts": optional_missing,
        "optional_metric_nonfinite_counts": optional_nonfinite,
        "balanced_run_positions": balanced_positions,
        "run_position_counts": run_positions,
        "successes": int(sum(bool(row["success"]) for row in rows)),
        "collisions": int(sum(bool(row["collision"]) for row in rows)),
    }


def _descriptives(rows):
    output = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        selected = [row for row in rows if str(row["condition"]) == condition]
        output[condition] = {"episodes": len(selected)}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            output[condition][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        for metric in OPTIONAL_METRICS:
            available = [
                float(row[metric]) for row in selected
                if row.get(metric) is not None
                and np.isfinite(float(row[metric]))
            ]
            output[condition][metric] = {
                "available": len(available),
                "missing": len(selected) - len(available),
                "mean": None if not available else float(np.mean(available)),
                "minimum": None if not available else float(np.min(available)),
                "maximum": None if not available else float(np.max(available)),
            }
    return output


def _all_contrasts(rows, bootstrap_seed):
    pairs = {
        "icode_effect_contextual_k50": (
            "icode_contextual_k50", "nominal_contextual_k50"
        ),
        "icode_effect_fixed_k50": ("icode_fixed_k50", "nominal_fixed_k50"),
        "icode_effect_fixed_k100": ("icode_fixed_k100", "nominal_fixed_k100"),
        "icode_same_budget_contextual": (
            "icode_contextual_k50", "icode_fixed_k50"
        ),
        "nominal_same_budget_contextual": (
            "nominal_contextual_k50", "nominal_fixed_k50"
        ),
        "icode_half_budget": ("icode_contextual_k50", "icode_fixed_k100"),
        "nominal_half_budget": ("nominal_contextual_k50", "nominal_fixed_k100"),
        "final_package": ("icode_contextual_k50", "nominal_fixed_k100"),
    }
    return {
        name: _paired_contrast(rows, treatment, reference, bootstrap_seed + 20 * index)
        for index, (name, (treatment, reference)) in enumerate(pairs.items())
    }


def _factorial_interactions(rows, bootstrap_seed):
    return {
        "same_budget_contextual_by_dynamics": _difference_in_differences(
            rows,
            "icode_contextual_k50", "icode_fixed_k50",
            "nominal_contextual_k50", "nominal_fixed_k50",
            bootstrap_seed,
        ),
        "half_budget_contextual_by_dynamics": _difference_in_differences(
            rows,
            "icode_contextual_k50", "icode_fixed_k100",
            "nominal_contextual_k50", "nominal_fixed_k100",
            bootstrap_seed + 20,
        ),
        "icode_effect_contextual_vs_fixed_k100": _difference_in_differences(
            rows,
            "icode_contextual_k50", "nominal_contextual_k50",
            "icode_fixed_k100", "nominal_fixed_k100",
            bootstrap_seed + 40,
        ),
    }


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
        handle.write("\n")

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
        config = _configure_arm(config, arm, spec)
        config["experiment"]["seed"] = int(item["seed"])
        config["experiment"]["name"] = "%s_%s_%s_%s_%d" % (
            str(spec["phase"]), config["scene"]["name"], domain["name"],
            arm["name"], item["seed"],
        )
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
            "prediction_mode": str(arm["prediction_mode"]),
            "sampler_kind": str(arm["sampler_kind"]),
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

    descriptives = _descriptives(rows)
    contrasts = _all_contrasts(rows, int(spec["bootstrap_seed"]))
    interactions = _factorial_interactions(
        rows, int(spec["bootstrap_seed"]) + 1000
    )
    package_compute = descriptives["icode_contextual_k50"][
        "planner_compute_ms_mean"
    ]["mean"]
    deadline = float(spec["planner_mean_deadline_ms"])
    primary = {
        "icode_contribution": _dynamics_gate(
            contrasts["icode_effect_contextual_k50"], package_compute, deadline
        ),
        "rl_half_budget_inside_icode": _half_budget_gate(
            contrasts["icode_half_budget"],
            float(spec["cross_track_noninferiority_margin_m"]),
        ),
        "complete_package": _final_package_gate(
            contrasts["final_package"], package_compute, deadline
        ),
    }
    audit = _audit_rows(rows, spec)
    summary = {
        "schema_version": 1,
        "design_id": str(spec["design_id"]),
        "phase": str(spec["phase"]),
        "completed_runs": len(rows),
        "audit": audit,
        "descriptives": descriptives,
        "contrasts": contrasts,
        "factorial_interactions": interactions,
        "primary_gates": primary,
        "primary_gate_passed": bool(
            audit["passed"] and all(value["gate_passed"] for value in primary.values())
        ),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
