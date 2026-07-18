#!/usr/bin/env python3
"""Closed-loop dynamic-obstacle covariance context oracle for L91."""

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
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["scene"], row["candidate"]), []).append(row)
    output = {}
    for (scene, candidate), values in groups.items():
        output.setdefault(scene, {})[candidate] = {
            "episodes": len(values),
            "success_rate": float(np.mean([float(v["success"]) for v in values])),
            "collision_rate": float(np.mean([float(v["collision"]) for v in values])),
            "elapsed_s": float(np.mean([float(v["elapsed_s"]) for v in values])),
            "final_goal_distance": float(np.mean([
                float(v["final_goal_distance"]) for v in values
            ])),
        }
    return output


def _choose(candidates):
    best_success = max(value["success_rate"] for value in candidates.values())
    success_tied = {
        name: value for name, value in candidates.items()
        if abs(value["success_rate"] - best_success) <= 1e-12
    }
    best_collision = min(value["collision_rate"] for value in success_tied.values())
    safe_tied = {
        name: value for name, value in success_tied.items()
        if abs(value["collision_rate"] - best_collision) <= 1e-12
    }
    return min(
        safe_tied,
        key=lambda name: (
            safe_tied[name]["elapsed_s"],
            safe_tied[name]["final_goal_distance"],
            name,
        ),
    )


def select_mapping(rows):
    summary = _candidate_summary(rows)
    mapping = {
        scene: _choose(candidates)
        for scene, candidates in sorted(summary.items())
    }
    pooled = {}
    for candidate in sorted({row["candidate"] for row in rows}):
        values = [row for row in rows if row["candidate"] == candidate]
        pooled[candidate] = {
            "episodes": len(values),
            "success_rate": float(np.mean([float(v["success"]) for v in values])),
            "collision_rate": float(np.mean([float(v["collision"]) for v in values])),
            "elapsed_s": float(np.mean([float(v["elapsed_s"]) for v in values])),
            "final_goal_distance": float(np.mean([
                float(v["final_goal_distance"]) for v in values
            ])),
        }
    return mapping, _choose(pooled), summary, pooled


def evaluate_mapping(
    rows,
    mapping,
    global_candidate,
    bootstrap_seed,
    gate_mode="elapsed",
):
    indexed = {
        (row["scene"], int(row["seed"]), row["candidate"]): row
        for row in rows
    }
    metrics = (
        "elapsed_s",
        "final_goal_distance",
        "control_jerk",
        "safety_interventions",
    )
    differences = {metric: {} for metric in metrics}
    clearance = {}
    safety = {"success": {}, "collision": {}}
    pairs = 0
    for scene, selected in sorted(mapping.items()):
        seeds = sorted({seed for (name, seed, _candidate) in indexed if name == scene})
        for seed in seeds:
            context = indexed[(scene, seed, selected)]
            fixed = indexed[(scene, seed, global_candidate)]
            pairs += 1
            for metric in metrics:
                differences[metric].setdefault(scene, []).append(
                    float(context[metric]) - float(fixed[metric])
                )
            if context.get("minimum_clearance") is not None and fixed.get("minimum_clearance") is not None:
                clearance.setdefault(scene, []).append(
                    float(context["minimum_clearance"])
                    - float(fixed["minimum_clearance"])
                )
            safety["success"].setdefault(scene, []).append(
                float(context["success"]) - float(fixed["success"])
            )
            safety["collision"].setdefault(scene, []).append(
                float(context["collision"]) - float(fixed["collision"])
            )
    result = {"paired_episodes": pairs}
    for offset, (metric, values) in enumerate(differences.items()):
        flat = [item for group in values.values() for item in group]
        result[metric + "_delta_mean"] = float(np.mean(flat))
        result[metric + "_delta_ci95"] = _hierarchical_ci(
            values, int(bootstrap_seed) + offset
        )
    if clearance:
        flat = [item for group in clearance.values() for item in group]
        result["minimum_clearance_delta_mean"] = float(np.mean(flat))
        result["minimum_clearance_delta_ci95"] = _hierarchical_ci(
            clearance, int(bootstrap_seed) + len(differences)
        )
    success_flat = [
        item for values in safety["success"].values() for item in values
    ]
    collision_flat = [
        item for values in safety["collision"].values() for item in values
    ]
    result["success_delta_ci95"] = _hierarchical_ci(
        safety["success"], int(bootstrap_seed) + len(differences) + 1
    )
    result["collision_delta_ci95"] = _hierarchical_ci(
        safety["collision"], int(bootstrap_seed) + len(differences) + 2
    )
    different = sum(value != global_candidate for value in mapping.values())
    result.update({
        "gate_mode": str(gate_mode),
        "global_candidate": global_candidate,
        "selected_candidate_count": len(set(mapping.values())),
        "contexts_total": len(mapping),
        "contexts_different_from_global": different,
        "success_delta_mean": float(np.mean(success_flat)),
        "collision_delta_mean": float(np.mean(collision_flat)),
    })
    result["heterogeneity_gate_passed"] = bool(
        result["selected_candidate_count"] >= 2 and different >= 1
    )
    result["safety_gate_passed"] = bool(
        result["success_delta_mean"] >= 0.0
        and result["collision_delta_mean"] <= 0.0
    )
    result["time_gate_passed"] = bool(result["elapsed_s_delta_ci95"][1] < 0.0)
    result["success_superiority_gate_passed"] = bool(
        result["success_delta_ci95"][0] > 0.0
    )
    result["collision_noninferiority_gate_passed"] = bool(
        result["collision_delta_ci95"][1] <= 0.0
    )
    if gate_mode == "elapsed":
        result["primary_gate_passed"] = bool(
            result["heterogeneity_gate_passed"]
            and result["safety_gate_passed"]
            and result["time_gate_passed"]
        )
    elif gate_mode == "safety_confirmation":
        result["primary_gate_passed"] = bool(
            result["heterogeneity_gate_passed"]
            and result["success_superiority_gate_passed"]
            and result["collision_noninferiority_gate_passed"]
        )
    else:
        raise ValueError("unknown dynamic oracle gate mode: %s" % gate_mode)
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
    for split, seeds in (
        ("selection", spec["selection_seeds"]),
        ("evaluation", spec["evaluation_seeds"]),
    ):
        for scene_path in spec["scenes"]:
            for candidate in spec["candidates"]:
                for seed in seeds:
                    schedule.append({
                        "split": split,
                        "scene_path": scene_path,
                        "candidate": candidate,
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
        candidate = item["candidate"]
        config = _apply_scene_config(base, ROOT / item["scene_path"])
        config["experiment"]["seed"] = item["seed"]
        config["experiment"]["name"] = "l91_%s_%s_%s_%d" % (
            item["split"], config["scene"]["name"], candidate["name"], item["seed"]
        )
        config["planner"]["sampling_prior"] = "fixed_covariance"
        config["planner"]["fixed_covariance_scale"] = list(candidate["scale"])
        config.setdefault("rl", {})["enabled"] = False
        config["rl"].pop("checkpoint", None)
        run_dir = (
            output / "runs" / item["split"] / config["scene"]["name"]
            / candidate["name"] / ("seed_%d" % item["seed"])
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
            "split": item["split"],
            "scene": str(config["scene"]["name"]),
            "candidate": str(candidate["name"]),
            "scale": json.dumps(candidate["scale"]),
            "seed": int(item["seed"]),
            "elapsed_s": float(summary["steps"]) * float(config["experiment"]["control_dt"]),
        })
        rows.append(row)
    episode_name = (
        "episodes.csv" if args.shard_count == 1
        else "episodes_shard_%03d_of_%03d.csv" % (args.shard_index, args.shard_count)
    )
    _write_csv(output / episode_name, rows)
    selection = [row for row in rows if row["split"] == "selection"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    factor = len(spec["scenes"]) * len(spec["candidates"])
    if (
        len(selection) != factor * len(spec["selection_seeds"])
        or len(evaluation) != factor * len(spec["evaluation_seeds"])
    ):
        print(json.dumps({"complete": False, "completed_runs": len(rows)}, indent=2))
        return 0
    mapping, global_candidate, context_summary, pooled = select_mapping(selection)
    result = evaluate_mapping(
        evaluation,
        mapping,
        global_candidate,
        spec["bootstrap_seed"],
        spec.get("gate", {}).get("mode", "elapsed"),
    )
    summary = {
        "schema_version": 1,
        "design_id": spec["design_id"],
        "completed_runs": len(rows),
        "selection_runs": len(selection),
        "evaluation_runs": len(evaluation),
        "context_mapping": mapping,
        "global_candidate": global_candidate,
        "selection_context_summary": context_summary,
        "selection_pooled_summary": pooled,
        "evaluation": result,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
