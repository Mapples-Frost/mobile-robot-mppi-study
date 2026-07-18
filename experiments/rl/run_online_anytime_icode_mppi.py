#!/usr/bin/env python3
"""Blocked whole-episode evaluation of online STOP/ADD ICODE-MPPI.

This runner evaluates the frozen L103 contextual budget policy as an actual
closed-loop controller.  Episode is the experimental unit; MPPI steps are
mechanism observations and are never counted as independent replicates.
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
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.rl.evaluate_rl_sampling_prior import _apply_scene_config
from experiments.rl.run_contextual_covariance_jerk_screening import (
    _blocked_schedule,
)
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
    "final_goal_distance",
    "mean_samples",
)


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _configure_arm(base, arm):
    result = copy.deepcopy(base)
    planner = result.setdefault("planner", {})
    result.setdefault("memory", {})["enable"] = False
    result.setdefault("rl", {})["enabled"] = True
    planner["prediction_mode"] = "icode_residual"
    planner["sampling_prior"] = "contextual_bandit_covariance"
    planner["num_samples"] = int(arm["num_samples"])
    optimizer = str(arm["optimizer"])
    planner["optimizer"] = optimizer
    if optimizer == "standard":
        planner.pop("anytime_bandit", None)
    elif optimizer != "anytime_bandit":
        raise ValueError("unknown L104 optimizer: %s" % optimizer)
    return result


def _paired_contrast(rows, treatment, reference, bootstrap_seed):
    indexed = {
        (
            str(row["scene"]), str(row["physics_domain"]), int(row["seed"]),
            str(row["condition"]),
        ): row
        for row in rows
    }
    contexts = sorted({key[:3] for key in indexed})
    result = {
        "treatment": str(treatment),
        "reference": str(reference),
        "paired_episodes": len(contexts),
        "success_delta_mean": float(np.mean([
            float(indexed[context + (treatment,)]["success"])
            - float(indexed[context + (reference,)]["success"])
            for context in contexts
        ])),
        "collision_delta_mean": float(np.mean([
            float(indexed[context + (treatment,)]["collision"])
            - float(indexed[context + (reference,)]["collision"])
            for context in contexts
        ])),
    }
    for offset, metric in enumerate(METRICS):
        grouped = {}
        for context in contexts:
            scene, domain, _seed = context
            difference = (
                float(indexed[context + (treatment,)][metric])
                - float(indexed[context + (reference,)][metric])
            )
            grouped.setdefault(scene + "__" + domain, []).append(difference)
        flat = [value for values in grouped.values() for value in values]
        result[metric + "_delta_mean"] = float(np.mean(flat))
        result[metric + "_delta_ci95"] = _hierarchical_ci(
            grouped, int(bootstrap_seed) + offset
        )
    return result


def _audit(rows, spec):
    arm_names = {str(arm["name"]) for arm in spec["arms"]}
    expected_blocks = (
        len(spec["scenes"]) * len(spec["physics_domains"])
        * len(spec["seeds"])
    )
    blocks = {}
    keys = []
    for row in rows:
        block = (str(row["scene"]), str(row["physics_domain"]), int(row["seed"]))
        blocks.setdefault(block, set()).add(str(row["condition"]))
        keys.append(block + (str(row["condition"]),))
    finite = all(
        np.isfinite(float(row[metric]))
        for row in rows for metric in METRICS
    )
    return {
        "passed": bool(
            len(rows) == expected_blocks * len(arm_names)
            and len(set(keys)) == len(rows)
            and len(blocks) == expected_blocks
            and all(value == arm_names for value in blocks.values())
            and finite
        ),
        "expected_rows": expected_blocks * len(arm_names),
        "observed_rows": len(rows),
        "complete_blocks": sum(value == arm_names for value in blocks.values()),
        "expected_blocks": expected_blocks,
        "finite_metrics": bool(finite),
    }


def _descriptives(rows):
    result = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        selected = [row for row in rows if str(row["condition"]) == condition]
        values = {
            "episodes": len(selected),
            "success_rate": float(np.mean([float(row["success"]) for row in selected])),
            "collision_rate": float(np.mean([float(row["collision"]) for row in selected])),
            "anytime_add_fraction_mean": float(np.mean([
                float(row["anytime_add_fraction"]) for row in selected
            ])),
            "anytime_decision_refresh_fraction_mean": float(np.mean([
                float(row.get("anytime_decision_refresh_fraction", 0.0))
                for row in selected
            ])),
        }
        for metric in METRICS:
            array = np.asarray([float(row[metric]) for row in selected])
            values[metric + "_mean"] = float(np.mean(array))
            values[metric + "_std"] = float(np.std(array, ddof=1))
        result[condition] = values
    return result


def _mechanism(rows, condition):
    selected = [row for row in rows if str(row["condition"]) == condition]
    by_scene = {}
    by_domain = {}
    for key, destination in (("scene", by_scene), ("physics_domain", by_domain)):
        for value in sorted({str(row[key]) for row in selected}):
            fractions = [
                float(row["anytime_add_fraction"])
                for row in selected if str(row[key]) == value
            ]
            destination[value] = {
                "mean_add_fraction": float(np.mean(fractions)),
                "minimum": float(np.min(fractions)),
                "maximum": float(np.max(fractions)),
                "both_actions_observed": bool(0.0 < np.mean(fractions) < 1.0),
            }
    return {
        "by_scene": by_scene,
        "by_domain": by_domain,
        "scenes_with_stop_and_add": sum(
            value["both_actions_observed"] for value in by_scene.values()
        ),
        "domains_with_stop_and_add": sum(
            value["both_actions_observed"] for value in by_domain.values()
        ),
    }


def _gate(rows, spec, contrast, mechanism):
    anytime = [
        row for row in rows
        if str(row["condition"]) == "icode_anytime_k50_100"
    ]
    mean_budget = float(np.mean([float(row["mean_samples"]) for row in anytime]))
    checks = {
        "safety": bool(
            contrast["success_delta_mean"] >= 0.0
            and contrast["collision_delta_mean"] <= 0.0
        ),
        "mean_budget": bool(mean_budget <= float(spec["maximum_mean_budget"])),
        "tracking_noninferiority": bool(
            contrast["cross_track_rmse_delta_ci95"][1]
            <= float(spec["cross_track_noninferiority_margin_m"])
        ),
        "compute_superiority": bool(
            contrast["planner_compute_ms_mean_delta_ci95"][1] < 0.0
        ),
        "elapsed_noninferiority": bool(
            contrast["elapsed_s_delta_ci95"][1]
            <= float(spec["elapsed_noninferiority_margin_s"])
        ),
        "contextual_actions": bool(
            mechanism["scenes_with_stop_and_add"] >= 2
            and mechanism["domains_with_stop_and_add"] >= 2
        ),
    }
    expected_refresh = spec.get("expected_decision_refresh_fraction")
    if expected_refresh is not None:
        actual_refresh = float(np.mean([
            float(row["anytime_decision_refresh_fraction"])
            for row in anytime
        ]))
        checks["decision_refresh_contract"] = bool(
            abs(actual_refresh - float(expected_refresh))
            <= float(spec.get("decision_refresh_fraction_tolerance", 0.03))
        )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "mean_budget": mean_budget,
        "decision_refresh_fraction": float(np.mean([
            float(row["anytime_decision_refresh_fraction"])
            for row in anytime
        ])),
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
    with (output / "design_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)

    selected_blocks = sorted({int(item["block_index"]) for item in schedule})
    if args.max_blocks is not None:
        selected_blocks = selected_blocks[:int(args.max_blocks)]
    selected_blocks = set(selected_blocks)
    rows = []
    episodes_path = output / "episodes.csv"
    for item in schedule:
        if int(item["block_index"]) not in selected_blocks:
            continue
        arm = item["arm"]
        domain = item["physics_domain"]
        config = _apply_scene_config(base, ROOT / item["scene_path"])
        config = deep_merge(config, {"plant": domain.get("plant_override", {})})
        config = _configure_arm(config, arm)
        config["experiment"]["seed"] = int(item["seed"])
        config["experiment"]["name"] = "l104_%s_%s_%s_%d" % (
            config["scene"]["name"], domain["name"], arm["name"], item["seed"]
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
        optimizer = str(arm["optimizer"])
        mean_samples = (
            float(episode["anytime_mean_samples"])
            if optimizer == "anytime_bandit" else float(arm["num_samples"])
        )
        row = dict(episode)
        row.update({
            "block_index": int(item["block_index"]),
            "run_position": int(item["run_position"]),
            "scene": str(config["scene"]["name"]),
            "physics_domain": str(domain["name"]),
            "condition": str(arm["name"]),
            "seed": int(item["seed"]),
            "elapsed_s": float(episode["steps"])
            * float(config["experiment"]["control_dt"]),
            "mean_samples": mean_samples,
        })
        rows.append(row)
        _write_csv(episodes_path, rows)

    expected = len(selected_blocks) * len(spec["arms"])
    if len(rows) != expected or len(selected_blocks) != (
        len(spec["scenes"]) * len(spec["physics_domains"]) * len(spec["seeds"])
    ):
        print(json.dumps({
            "complete": False,
            "completed_runs": len(rows),
            "selected_blocks": len(selected_blocks),
        }, indent=2))
        return 0

    anytime = "icode_anytime_k50_100"
    contrast_k100 = _paired_contrast(
        rows, anytime, "icode_contextual_k100", int(spec["bootstrap_seed"])
    )
    contrast_k50 = _paired_contrast(
        rows, anytime, "icode_contextual_k50", int(spec["bootstrap_seed"]) + 100
    )
    mechanism = _mechanism(rows, anytime)
    summary = {
        "schema_version": 1,
        "design_id": str(spec["design_id"]),
        "phase": str(spec["phase"]),
        "completed_runs": len(rows),
        "audit": _audit(rows, spec),
        "descriptives": _descriptives(rows),
        "anytime_vs_k100": contrast_k100,
        "anytime_vs_k50": contrast_k50,
        "mechanism": mechanism,
        "primary_gate": _gate(rows, spec, contrast_k100, mechanism),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
