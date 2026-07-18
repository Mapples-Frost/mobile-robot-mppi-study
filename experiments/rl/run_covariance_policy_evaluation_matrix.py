#!/usr/bin/env python3
"""Paired learned-vs-fixed covariance evaluation for L84.

Every condition sees the same block, scene and plant seed.  Completed runs are
loaded from ``metrics.json`` so a long matrix can be safely resumed.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.rl.evaluate_rl_sampling_prior import _apply_scene_config
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


NUMERIC_METRICS = (
    "cross_track_rmse",
    "cross_track_mean",
    "cross_track_max",
    "final_goal_distance",
    "trajectory_length",
    "minimum_clearance",
    "time_to_goal_s",
    "mean_abs_omega",
    "control_jerk",
    "planner_compute_ms_mean",
    "planner_compute_ms_max",
)


def _selection(text):
    if text is None:
        return None
    return {value.strip() for value in text.split(",") if value.strip()}


def _selected(items, names):
    if names is None:
        return list(items)
    result = [item for item in items if str(item["name"]) in names]
    missing = names - {str(item["name"]) for item in result}
    if missing:
        raise ValueError("unknown selection: %s" % sorted(missing))
    return result


def _load_completed(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _bootstrap_mean_ci(values, seed=84017, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    if values.size == 1:
        value = float(values[0])
        return [value, value]
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, values.size, size=(samples, values.size))
    means = np.mean(values[indices], axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _aggregate(rows):
    groups = {}
    for row in rows:
        key = (row["block"], row["condition"], row["scene"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        item = {
            "block": key[0],
            "condition": key[1],
            "scene": key[2],
            "episodes": len(values),
            "success_rate": float(np.mean([float(v["success"]) for v in values])),
            "collision_rate": float(np.mean([float(v["collision"]) for v in values])),
        }
        for metric in NUMERIC_METRICS:
            finite = [
                float(v[metric]) for v in values
                if v.get(metric) is not None and np.isfinite(float(v[metric]))
            ]
            item[metric + "_mean"] = float(np.mean(finite)) if finite else None
        output.append(item)
    return output


def _paired_primary(
    rows,
    learned="learned_covariance",
    comparator="fixed_broad",
    cross_track_noninferiority_margin=None,
    require_time_superiority=False,
):
    indexed = {
        (r["block"], r["scene"], int(r["seed"]), r["condition"]): r
        for r in rows
    }
    differences = []
    block_differences = {}
    time_differences = []
    safety = []
    keys = sorted({key[:3] for key in indexed})
    for block, scene, seed in keys:
        left = indexed.get((block, scene, seed, learned))
        right = indexed.get((block, scene, seed, comparator))
        if left is None or right is None:
            continue
        difference = float(left["cross_track_rmse"]) - float(
            right["cross_track_rmse"]
        )
        differences.append(difference)
        block_differences.setdefault(block, []).append(difference)
        left_time = left.get("time_to_goal_s")
        right_time = right.get("time_to_goal_s")
        if (
            left_time is not None
            and right_time is not None
            and np.isfinite(float(left_time))
            and np.isfinite(float(right_time))
        ):
            time_differences.append(float(left_time) - float(right_time))
        safety.append({
            "success_delta": float(left["success"]) - float(right["success"]),
            "collision_delta": float(left["collision"]) - float(right["collision"]),
        })
    if not differences:
        return None
    per_block = {
        block: float(np.mean(values))
        for block, values in sorted(block_differences.items())
    }
    cross_track_ci = _bootstrap_mean_ci(differences)
    time_ci = _bootstrap_mean_ci(time_differences, seed=84018)
    legacy_gate = bool(
        per_block
        and all(value < 0.0 for value in per_block.values())
        and np.mean([v["success_delta"] for v in safety]) >= 0.0
        and np.mean([v["collision_delta"] for v in safety]) <= 0.0
    )
    if cross_track_noninferiority_margin is None:
        precision_gate = bool(all(value < 0.0 for value in per_block.values()))
    else:
        margin = float(cross_track_noninferiority_margin)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("cross-track noninferiority margin must be non-negative")
        precision_gate = bool(cross_track_ci[1] <= margin)
    time_gate = bool(
        not require_time_superiority
        or (time_ci is not None and time_ci[1] < 0.0)
    )
    safety_gate = bool(
        np.mean([v["success_delta"] for v in safety]) >= 0.0
        and np.mean([v["collision_delta"] for v in safety]) <= 0.0
    )
    return {
        "learned": learned,
        "comparator": comparator,
        "paired_episodes": len(differences),
        "cross_track_rmse_delta_mean": float(np.mean(differences)),
        "cross_track_rmse_delta_ci95": cross_track_ci,
        "cross_track_rmse_delta_by_block": per_block,
        "blocks_improved": int(sum(value < 0.0 for value in per_block.values())),
        "blocks_total": len(per_block),
        "success_delta_mean": float(np.mean([v["success_delta"] for v in safety])),
        "collision_delta_mean": float(np.mean([v["collision_delta"] for v in safety])),
        "time_to_goal_s_delta_mean": (
            float(np.mean(time_differences)) if time_differences else None
        ),
        "time_to_goal_s_delta_ci95": time_ci,
        "cross_track_noninferiority_margin": (
            None
            if cross_track_noninferiority_margin is None
            else float(cross_track_noninferiority_margin)
        ),
        "precision_gate_passed": precision_gate,
        "time_superiority_required": bool(require_time_superiority),
        "time_gate_passed": time_gate,
        "safety_gate_passed": safety_gate,
        "legacy_primary_gate_passed": legacy_gate,
        "primary_gate_passed": bool(
            precision_gate and time_gate and safety_gate
        ),
    }


def _paired_condition_metrics(
    rows, learned="learned_covariance", comparator="fixed_broad"
):
    indexed = {
        (r["block"], r["scene"], int(r["seed"]), r["condition"]): r
        for r in rows
    }
    pairs = []
    for key in sorted({value[:3] for value in indexed}):
        left = indexed.get(key + (learned,))
        right = indexed.get(key + (comparator,))
        if left is not None and right is not None:
            pairs.append((left, right))
    if not pairs:
        return None
    metrics = (
        "cross_track_rmse",
        "time_to_goal_s",
        "trajectory_length",
        "control_jerk",
        "planner_compute_ms_mean",
    )
    output = {
        "learned": learned,
        "comparator": comparator,
        "paired_episodes": len(pairs),
        "success_delta_mean": float(np.mean([
            float(left["success"]) - float(right["success"])
            for left, right in pairs
        ])),
        "collision_delta_mean": float(np.mean([
            float(left["collision"]) - float(right["collision"])
            for left, right in pairs
        ])),
    }
    for metric_index, metric in enumerate(metrics):
        differences = [
            float(left[metric]) - float(right[metric])
            for left, right in pairs
            if left.get(metric) is not None
            and right.get(metric) is not None
            and np.isfinite(float(left[metric]))
            and np.isfinite(float(right[metric]))
        ]
        output[metric + "_delta_mean"] = (
            float(np.mean(differences)) if differences else None
        )
        output[metric + "_delta_ci95"] = (
            _bootstrap_mean_ci(differences, seed=84017 + metric_index)
            if differences else None
        )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blocks")
    parser.add_argument("--conditions")
    parser.add_argument("--scenes")
    parser.add_argument("--seeds", help="comma-separated override")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args(argv)

    with Path(args.config).open("r", encoding="utf-8") as handle:
        specification = yaml.safe_load(handle)
    if not isinstance(specification, dict):
        raise ValueError("matrix config must contain a mapping")
    blocks = _selected(specification["blocks"], _selection(args.blocks))
    conditions = _selected(
        specification["conditions"], _selection(args.conditions)
    )
    scene_paths = [str(value) for value in specification["scenes"]]
    scene_filter = _selection(args.scenes)
    if scene_filter is not None:
        scene_paths = [
            value for value in scene_paths
            if Path(value).stem in scene_filter or value in scene_filter
        ]
        missing = scene_filter - {
            name for value in scene_paths for name in (value, Path(value).stem)
        }
        if missing:
            raise ValueError("unknown scenes: %s" % sorted(missing))
    seeds = (
        [int(value) for value in args.seeds.split(",")]
        if args.seeds
        else [int(value) for value in specification["episode_seeds"]]
    )
    if not seeds:
        raise ValueError("at least one episode seed is required")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    attempted = 0
    for block in blocks:
        base = load_yaml(ROOT / block["config"])
        checkpoint = (ROOT / block["checkpoint"]).resolve()
        for scene_path in scene_paths:
            scene = _apply_scene_config(base, ROOT / scene_path)
            scene_name = str(scene["scene"]["name"])
            for condition in conditions:
                for seed in seeds:
                    if args.max_runs is not None and attempted >= args.max_runs:
                        break
                    attempted += 1
                    config = copy.deepcopy(scene)
                    config["experiment"]["seed"] = seed
                    if args.max_steps is not None:
                        if args.max_steps <= 0:
                            raise ValueError("--max-steps must be positive")
                        config["experiment"]["max_steps"] = args.max_steps
                    if args.num_samples is not None:
                        if args.num_samples <= 0:
                            raise ValueError("--num-samples must be positive")
                        config["planner"]["num_samples"] = args.num_samples
                    config["experiment"]["name"] = "%s_%s_%s_%d" % (
                        block["name"], scene_name, condition["name"], seed
                    )
                    if condition["kind"] == "learned":
                        if not checkpoint.exists():
                            raise FileNotFoundError(str(checkpoint))
                        config["planner"]["sampling_prior"] = "rl"
                        config.setdefault("rl", {}).update({
                            "enabled": True,
                            "checkpoint": str(checkpoint),
                        })
                    elif condition["kind"] == "fixed":
                        config["planner"]["sampling_prior"] = "fixed_covariance"
                        config["planner"]["fixed_covariance_scale"] = list(
                            condition["scale"]
                        )
                        config.setdefault("rl", {})["enabled"] = False
                        config["rl"].pop("checkpoint", None)
                    else:
                        raise ValueError("condition kind must be learned or fixed")
                    run_dir = output / "runs" / str(block["name"]) / scene_name / str(condition["name"]) / ("seed_%d" % seed)
                    metrics_path = run_dir / "metrics.json"
                    summary = _load_completed(metrics_path)
                    if summary is None:
                        summary = ExperimentRunner(
                            config, ROOT, run_dir, headless=True
                        ).run().summary
                    row = dict(summary)
                    row.update({
                        "block": str(block["name"]),
                        "scene": scene_name,
                        "condition": str(condition["name"]),
                        "seed": seed,
                        "checkpoint": (
                            str(checkpoint)
                            if condition["kind"] == "learned" else ""
                        ),
                        "covariance_scale": json.dumps(
                            condition.get("scale")
                        ),
                    })
                    rows.append(row)
                if args.max_runs is not None and attempted >= args.max_runs:
                    break
            if args.max_runs is not None and attempted >= args.max_runs:
                break
        if args.max_runs is not None and attempted >= args.max_runs:
            break

    if not rows:
        raise RuntimeError("evaluation matrix produced no rows")
    fieldnames = sorted({key for row in rows for key in row})
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = _aggregate(rows)
    with (output / "condition_scene_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    summary = {
        "schema_version": 1,
        "design_id": specification.get("design_id"),
        "completed_runs": len(rows),
        "primary_comparison": _paired_primary(
            rows,
            comparator=str(specification.get("primary_comparator", "fixed_broad")),
            cross_track_noninferiority_margin=specification.get(
                "cross_track_noninferiority_margin"
            ),
            require_time_superiority=bool(
                specification.get("require_time_superiority", False)
            ),
        ),
        "fixed_comparisons": {
            str(condition["name"]): _paired_condition_metrics(
                rows, comparator=str(condition["name"])
            )
            for condition in conditions
            if str(condition["name"]) != "learned_covariance"
        },
        "conditions": sorted({row["condition"] for row in rows}),
        "blocks": sorted({row["block"] for row in rows}),
        "scenes": sorted({row["scene"] for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
