#!/usr/bin/env python3
"""Blocked 2x2 screen for contextual half-budget jerk remediation."""

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


METRICS = (
    "cross_track_rmse",
    "elapsed_s",
    "control_jerk",
    "applied_control_jerk",
    "planner_compute_ms_mean",
)


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _blocked_schedule(spec):
    """Return a randomized complete-block schedule with balanced run positions."""

    blocks = []
    for scene_path in spec["scenes"]:
        for domain in spec["physics_domains"]:
            for seed in spec["seeds"]:
                blocks.append((scene_path, domain, int(seed)))
    rng = np.random.RandomState(int(spec["schedule_seed"]))
    rng.shuffle(blocks)
    arms = list(spec["arms"])
    base_order = list(rng.permutation(len(arms)))
    schedule = []
    for block_index, (scene_path, domain, seed) in enumerate(blocks):
        shift = block_index % len(arms)
        order = base_order[shift:] + base_order[:shift]
        for run_position, arm_index in enumerate(order):
            schedule.append({
                "block_index": int(block_index),
                "run_position": int(run_position),
                "scene_path": scene_path,
                "physics_domain": domain,
                "seed": int(seed),
                "arm": arms[int(arm_index)],
            })
    return schedule


def _paired_contrast(rows, treatment, reference, bootstrap_seed):
    indexed = {
        (str(row["scene"]), str(row["physics_domain"]), int(row["seed"]), str(row["condition"])): row
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


def _add_screen_gate(contrast, rmse_margin, elapsed_margin):
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
    result["candidate_gate_passed"] = bool(
        result["safety_gate_passed"]
        and result["precision_gate_passed"]
        and result["elapsed_gate_passed"]
        and result["issued_jerk_gate_passed"]
        and result["applied_jerk_gate_passed"]
    )
    return result


def _select_candidate(contrasts):
    eligible = [
        value for value in contrasts.values()
        if value["candidate_gate_passed"]
    ]
    if not eligible:
        return None
    selected = min(eligible, key=lambda value: (
        value["applied_control_jerk_delta_mean"],
        value["control_jerk_delta_mean"],
        value["elapsed_s_delta_mean"],
        value["treatment"],
    ))
    return str(selected["treatment"])


def _audit_rows(rows, spec):
    keys = [
        (str(row["scene"]), str(row["physics_domain"]), int(row["seed"]), str(row["condition"]))
        for row in rows
    ]
    expected = (
        len(spec["scenes"]) * len(spec["physics_domains"])
        * len(spec["seeds"]) * len(spec["arms"])
    )
    arm_names = {str(arm["name"]) for arm in spec["arms"]}
    blocks = {}
    for row in rows:
        block = (str(row["scene"]), str(row["physics_domain"]), int(row["seed"]))
        blocks.setdefault(block, set()).add(str(row["condition"]))
    finite = all(
        np.isfinite(float(row[metric]))
        for row in rows for metric in METRICS
    )
    run_positions = {
        name: {
            str(position): sum(
                str(row["condition"]) == name
                and int(row["run_position"]) == position
                for row in rows
            )
            for position in range(len(spec["arms"]))
        }
        for name in sorted(arm_names)
    }
    passed = bool(
        len(rows) == expected
        and len(set(keys)) == expected
        and finite
        and all(values == arm_names for values in blocks.values())
    )
    return {
        "passed": passed,
        "expected_rows": int(expected),
        "observed_rows": int(len(rows)),
        "unique_keys": int(len(set(keys))),
        "complete_blocks": int(sum(values == arm_names for values in blocks.values())),
        "expected_blocks": int(expected // len(spec["arms"])),
        "finite_metric_values": bool(finite),
        "successes": int(sum(bool(row["success"]) for row in rows)),
        "collisions": int(sum(bool(row["collision"]) for row in rows)),
        "run_position_counts": run_positions,
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
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--block-shard-count", type=int, default=1)
    parser.add_argument("--block-shard-index", type=int, default=0)
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
    if (
        args.block_shard_count <= 0
        or not 0 <= args.block_shard_index < args.block_shard_count
    ):
        raise ValueError("block shard index must lie in [0, block_shard_count)")
    selected_blocks = [
        block for block in selected_blocks
        if block % int(args.block_shard_count) == int(args.block_shard_index)
    ]
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
        config["experiment"]["name"] = "l96_%s_%s_%s_%d" % (
            config["scene"]["name"], domain["name"], arm["name"], item["seed"]
        )
        config["planner"]["num_samples"] = int(spec["num_samples"])
        config["planner"]["sampling_prior"] = "contextual_bandit_covariance"
        config.setdefault("rl", {})["enabled"] = True
        config["rl"]["policy_id"] = "linucb_contextual_covariance_l89"
        config["rl"]["checkpoint"] = spec["checkpoint"]
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
            "seed": int(item["seed"]),
            "elapsed_s": float(episode["steps"])
            * float(config["experiment"]["control_dt"]),
        })
        rows.append(row)
        episode_path = (
            output / "episodes.csv"
            if int(args.block_shard_count) == 1
            else output / (
                "episodes_shard_%03d_of_%03d.csv"
                % (args.block_shard_index, args.block_shard_count)
            )
        )
        _write_csv(episode_path, rows)
    expected = len(schedule)
    if int(args.block_shard_count) > 1:
        print(json.dumps({
            "complete": len(rows) == len(selected_blocks) * len(spec["arms"]),
            "completed_runs": len(rows),
            "expected_shard_runs": len(selected_blocks) * len(spec["arms"]),
            "block_shard_index": int(args.block_shard_index),
            "block_shard_count": int(args.block_shard_count),
        }, indent=2))
        return 0
    if len(rows) != expected:
        print(json.dumps({
            "complete": False,
            "completed_runs": len(rows),
            "expected_runs": expected,
        }, indent=2))
        return 0
    baseline = str(spec["baseline_arm"])
    contrasts = {}
    for offset, arm in enumerate(spec["arms"]):
        name = str(arm["name"])
        if name == baseline:
            continue
        contrast = _paired_contrast(
            rows, name, baseline, int(spec["bootstrap_seed"]) + 20 * offset
        )
        contrasts[name] = _add_screen_gate(
            contrast,
            float(spec["cross_track_noninferiority_margin_m"]),
            float(spec["elapsed_noninferiority_margin_s"]),
        )
    summary = {
        "schema_version": 1,
        "design_id": spec["design_id"],
        "completed_runs": len(rows),
        "baseline_arm": baseline,
        "arm_factors": {
            str(arm["name"]): dict(arm.get("factors", {}))
            for arm in spec["arms"]
        },
        "contrasts_vs_current": contrasts,
        "audit": _audit_rows(rows, spec),
        "descriptives": _descriptives(rows),
        "selected_candidate": _select_candidate(contrasts),
        "selection_gate_passed": any(
            value["candidate_gate_passed"] for value in contrasts.values()
        ),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
