#!/usr/bin/env python3
"""Paired MuJoCo sample-efficiency evaluation for the L89 bandit."""

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
from experiments.rl.run_covariance_context_oracle import _hierarchical_ci
from mobile_robot_mppi.core.config import deep_merge, load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _paired_comparison(rows, learned_samples, fixed_samples, bootstrap_seed):
    indexed = {
        (
            row["scene"],
            row["physics_domain"],
            int(row["seed"]),
            int(row["num_samples"]),
            row["condition"],
        ): row
        for row in rows
    }
    contexts = sorted({key[:3] for key in indexed})
    metrics = (
        "cross_track_rmse",
        "elapsed_s",
        "control_jerk",
        "planner_compute_ms_mean",
    )
    differences = {metric: {} for metric in metrics}
    safety = []
    for scene, domain, seed in contexts:
        learned = indexed[(
            scene,
            domain,
            seed,
            int(learned_samples),
            "learned_contextual_bandit",
        )]
        fixed = indexed[(
            scene,
            domain,
            seed,
            int(fixed_samples),
            "strongest_global_fixed",
        )]
        context = scene + "__" + domain
        for metric in metrics:
            differences[metric].setdefault(context, []).append(
                float(learned[metric]) - float(fixed[metric])
            )
        safety.append((
            float(learned["success"]) - float(fixed["success"]),
            float(learned["collision"]) - float(fixed["collision"]),
        ))
    result = {
        "learned_samples": int(learned_samples),
        "fixed_samples": int(fixed_samples),
        "paired_episodes": len(contexts),
    }
    for offset, (metric, values) in enumerate(differences.items()):
        flat = [item for group in values.values() for item in group]
        result[metric + "_delta_mean"] = float(np.mean(flat))
        result[metric + "_delta_ci95"] = _hierarchical_ci(
            values, int(bootstrap_seed) + offset
        )
    result["success_delta_mean"] = float(np.mean([item[0] for item in safety]))
    result["collision_delta_mean"] = float(np.mean([item[1] for item in safety]))
    return result


def _add_gate(result, margin):
    result = dict(result)
    result["safety_gate_passed"] = bool(
        result["success_delta_mean"] >= 0.0
        and result["collision_delta_mean"] <= 0.0
    )
    result["precision_gate_passed"] = bool(
        result["cross_track_rmse_delta_ci95"][1] <= float(margin)
    )
    result["elapsed_gate_passed"] = bool(
        result["elapsed_s_delta_ci95"][1] <= 0.0
    )
    result["compute_gate_passed"] = bool(
        result["planner_compute_ms_mean_delta_ci95"][1] < 0.0
    )
    result["primary_gate_passed"] = bool(
        result["safety_gate_passed"]
        and result["precision_gate_passed"]
        and result["elapsed_gate_passed"]
        and result["compute_gate_passed"]
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args(argv)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    base = load_yaml(ROOT / spec["base_config"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    schedule = []
    for scene_path in spec["scenes"]:
        for domain in spec["physics_domains"]:
            for samples in spec["sample_counts"]:
                for condition in spec["conditions"]:
                    for seed in spec["seeds"]:
                        schedule.append({
                            "scene_path": scene_path,
                            "physics_domain": domain,
                            "num_samples": int(samples),
                            "condition": condition,
                            "seed": int(seed),
                        })
    np.random.RandomState(int(spec["schedule_seed"])).shuffle(schedule)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must lie in [0, shard_count)")
    with (output / "schedule.json").open("w", encoding="utf-8") as handle:
        json.dump(schedule, handle, indent=2, sort_keys=True)
    rows = []
    attempted = 0
    for schedule_index, item in enumerate(schedule):
        if schedule_index % args.shard_count != args.shard_index:
            continue
        if args.max_runs is not None and attempted >= int(args.max_runs):
            break
        attempted += 1
        condition = item["condition"]
        domain = item["physics_domain"]
        samples = int(item["num_samples"])
        config = _apply_scene_config(base, ROOT / item["scene_path"])
        config = deep_merge(config, {"plant": domain.get("plant_override", {})})
        config["experiment"]["seed"] = item["seed"]
        config["experiment"]["name"] = "l94_%s_%s_k%d_%s_%d" % (
            config["scene"]["name"],
            domain["name"],
            samples,
            condition["name"],
            item["seed"],
        )
        config["planner"]["num_samples"] = samples
        if condition["kind"] == "contextual_bandit":
            config["planner"]["sampling_prior"] = "contextual_bandit_covariance"
            config.setdefault("rl", {})["enabled"] = True
            config["rl"]["policy_id"] = "linucb_contextual_covariance_l89"
            config["rl"]["checkpoint"] = spec["checkpoint"]
        elif condition["kind"] == "fixed":
            config["planner"]["sampling_prior"] = "fixed_covariance"
            config["planner"]["fixed_covariance_scale"] = list(condition["scale"])
            config.setdefault("rl", {})["enabled"] = False
            config["rl"].pop("checkpoint", None)
        else:
            raise ValueError("unknown evaluation condition kind")
        run_dir = (
            output / "runs" / domain["name"] / config["scene"]["name"]
            / ("k_%03d" % samples) / condition["name"]
            / ("seed_%d" % item["seed"])
        )
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        else:
            summary = ExperimentRunner(
                copy.deepcopy(config), ROOT, run_dir, headless=True
            ).run().summary
        row = dict(summary)
        row.update({
            "scene": str(config["scene"]["name"]),
            "physics_domain": str(domain["name"]),
            "num_samples": samples,
            "condition": str(condition["name"]),
            "seed": int(item["seed"]),
            "elapsed_s": float(summary["steps"])
            * float(config["experiment"]["control_dt"]),
        })
        rows.append(row)
    episode_name = (
        "episodes.csv" if args.shard_count == 1
        else "episodes_shard_%03d_of_%03d.csv" % (
            args.shard_index, args.shard_count
        )
    )
    _write_csv(output / episode_name, rows)
    expected = (
        len(spec["scenes"]) * len(spec["physics_domains"])
        * len(spec["sample_counts"]) * len(spec["conditions"])
        * len(spec["seeds"])
    )
    if len(rows) != expected:
        print(json.dumps({"complete": False, "completed_runs": len(rows)}, indent=2))
        return 0
    same_budget = {}
    for offset, samples in enumerate(spec["sample_counts"]):
        same_budget[str(int(samples))] = _paired_comparison(
            rows, samples, samples, int(spec["bootstrap_seed"]) + 20 * offset
        )
    efficiency = _paired_comparison(
        rows,
        spec["efficiency_learned_samples"],
        spec["efficiency_fixed_samples"],
        spec["bootstrap_seed"],
    )
    efficiency = _add_gate(
        efficiency, spec["cross_track_noninferiority_margin_m"]
    )
    summary = {
        "schema_version": 1,
        "design_id": spec["design_id"],
        "completed_runs": len(rows),
        "checkpoint": spec["checkpoint"],
        "primary_sample_efficiency": efficiency,
        "same_budget_comparisons": same_budget,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
