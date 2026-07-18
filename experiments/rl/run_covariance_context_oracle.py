#!/usr/bin/env python3
"""Select and evaluate a non-deployable context covariance oracle.

Selection and evaluation seeds are strictly separated.  Completed episode
artifacts are reused, making the long MuJoCo factorial safely resumable.
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
from mobile_robot_mppi.core.config import deep_merge, load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _context_key(row):
    return str(row["scene"]), str(row["physics_domain"])


def _candidate_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((_context_key(row), row["candidate"]), []).append(row)
    output = {}
    for (context, candidate), values in grouped.items():
        output.setdefault(context, {})[candidate] = {
            "episodes": len(values),
            "success_rate": float(np.mean([float(v["success"]) for v in values])),
            "collision_rate": float(np.mean([float(v["collision"]) for v in values])),
            "cross_track_rmse": float(np.mean([
                float(v["cross_track_rmse"]) for v in values
            ])),
            "time_to_goal_s": float(np.mean([
                float(v["time_to_goal_s"]) for v in values
            ])),
        }
    return output


def _choose(candidates, rmse_slack):
    eligible = {
        name: value for name, value in candidates.items()
        if value["success_rate"] >= 1.0 - 1e-12
        and value["collision_rate"] <= 1e-12
    }
    if not eligible:
        raise RuntimeError("no safe successful covariance candidate")
    best_rmse = min(value["cross_track_rmse"] for value in eligible.values())
    precise = {
        name: value for name, value in eligible.items()
        if value["cross_track_rmse"] <= best_rmse + float(rmse_slack) + 1e-12
    }
    return min(
        precise,
        key=lambda name: (precise[name]["time_to_goal_s"], name),
    )


def select_mapping(rows, rmse_slack):
    """Select per-context and global candidates using selection rows only."""

    summary = _candidate_summary(rows)
    mapping = {
        context: _choose(candidates, rmse_slack)
        for context, candidates in sorted(summary.items())
    }
    pooled = {}
    candidate_names = sorted({row["candidate"] for row in rows})
    for candidate in candidate_names:
        values = [row for row in rows if row["candidate"] == candidate]
        pooled[candidate] = {
            "episodes": len(values),
            "success_rate": float(np.mean([float(v["success"]) for v in values])),
            "collision_rate": float(np.mean([float(v["collision"]) for v in values])),
            "cross_track_rmse": float(np.mean([
                float(v["cross_track_rmse"]) for v in values
            ])),
            "time_to_goal_s": float(np.mean([
                float(v["time_to_goal_s"]) for v in values
            ])),
        }
    return mapping, _choose(pooled, rmse_slack), summary, pooled


def _hierarchical_ci(context_values, seed, samples=10000):
    contexts = sorted(context_values)
    rng = np.random.RandomState(int(seed))
    draws = np.empty(int(samples), dtype=np.float64)
    for draw in range(int(samples)):
        selected_contexts = [
            contexts[index]
            for index in rng.randint(0, len(contexts), size=len(contexts))
        ]
        values = []
        for context in selected_contexts:
            within = np.asarray(context_values[context], dtype=np.float64)
            values.extend(rng.choice(within, size=within.size, replace=True))
        draws[draw] = float(np.mean(values))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def evaluate_mapping(rows, mapping, global_candidate, rmse_margin, bootstrap_seed):
    indexed = {
        (_context_key(row), int(row["seed"]), row["candidate"]): row
        for row in rows
    }
    differences = {"cross_track_rmse": {}, "time_to_goal_s": {}, "control_jerk": {}}
    safety = []
    pairs = 0
    for context, selected in sorted(mapping.items()):
        context_seeds = sorted({
            seed for (key, seed, _candidate) in indexed if key == context
        })
        for seed in context_seeds:
            oracle = indexed[(context, seed, selected)]
            fixed = indexed[(context, seed, global_candidate)]
            pairs += 1
            for metric in differences:
                differences[metric].setdefault(context, []).append(
                    float(oracle[metric]) - float(fixed[metric])
                )
            safety.append((
                float(oracle["success"]) - float(fixed["success"]),
                float(oracle["collision"]) - float(fixed["collision"]),
            ))
    metrics = {}
    for index, (metric, context_values) in enumerate(differences.items()):
        flat = [value for values in context_values.values() for value in values]
        metrics[metric + "_delta_mean"] = float(np.mean(flat))
        metrics[metric + "_delta_ci95"] = _hierarchical_ci(
            context_values, int(bootstrap_seed) + index
        )
    selected_values = set(mapping.values())
    different_contexts = sum(
        candidate != global_candidate for candidate in mapping.values()
    )
    precision_gate = bool(
        metrics["cross_track_rmse_delta_ci95"][1] <= float(rmse_margin)
    )
    time_gate = bool(metrics["time_to_goal_s_delta_ci95"][1] < 0.0)
    safety_gate = bool(
        np.mean([value[0] for value in safety]) >= 0.0
        and np.mean([value[1] for value in safety]) <= 0.0
    )
    heterogeneity_gate = bool(
        len(selected_values) >= 2
        and different_contexts >= int(np.ceil(0.25 * len(mapping)))
    )
    return {
        "paired_episodes": pairs,
        "global_candidate": global_candidate,
        "selected_candidate_count": len(selected_values),
        "contexts_different_from_global": different_contexts,
        "contexts_total": len(mapping),
        "success_delta_mean": float(np.mean([value[0] for value in safety])),
        "collision_delta_mean": float(np.mean([value[1] for value in safety])),
        "precision_gate_passed": precision_gate,
        "time_gate_passed": time_gate,
        "safety_gate_passed": safety_gate,
        "heterogeneity_gate_passed": heterogeneity_gate,
        "primary_gate_passed": bool(
            precision_gate and time_gate and safety_gate and heterogeneity_gate
        ),
        **metrics,
    }


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
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
            for domain in spec["physics_domains"]:
                for candidate in spec["candidates"]:
                    for seed in seeds:
                        schedule.append({
                            "split": split,
                            "scene_path": scene_path,
                            "physics_domain": domain,
                            "candidate": candidate,
                            "seed": int(seed),
                        })
    rng = np.random.RandomState(int(spec["schedule_seed"]))
    rng.shuffle(schedule)
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
        domain = item["physics_domain"]
        candidate = item["candidate"]
        config = _apply_scene_config(base, ROOT / item["scene_path"])
        config = deep_merge(config, {"plant": domain.get("plant_override", {})})
        config["experiment"]["seed"] = item["seed"]
        config["experiment"]["name"] = "l87_%s_%s_%s_%s_%d" % (
            item["split"], config["scene"]["name"], domain["name"],
            candidate["name"], item["seed"],
        )
        config["planner"]["sampling_prior"] = "fixed_covariance"
        config["planner"]["fixed_covariance_scale"] = list(candidate["scale"])
        config.setdefault("rl", {})["enabled"] = False
        config["rl"].pop("checkpoint", None)
        run_dir = (
            output / "runs" / item["split"] / domain["name"]
            / config["scene"]["name"] / candidate["name"]
            / ("seed_%d" % item["seed"])
        )
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            summary = json.load(metrics_path.open("r", encoding="utf-8"))
        else:
            summary = ExperimentRunner(
                copy.deepcopy(config), ROOT, run_dir, headless=True
            ).run().summary
        row = dict(summary)
        row.update({
            "split": item["split"],
            "scene": str(config["scene"]["name"]),
            "physics_domain": str(domain["name"]),
            "candidate": str(candidate["name"]),
            "scale": json.dumps(candidate["scale"]),
            "seed": int(item["seed"]),
        })
        rows.append(row)
    episode_name = (
        "episodes.csv"
        if args.shard_count == 1
        else "episodes_shard_%03d_of_%03d.csv" % (
            args.shard_index, args.shard_count
        )
    )
    _write_csv(output / episode_name, rows)
    selection_rows = [row for row in rows if row["split"] == "selection"]
    evaluation_rows = [row for row in rows if row["split"] == "evaluation"]
    expected_selection = (
        len(spec["scenes"]) * len(spec["physics_domains"])
        * len(spec["candidates"]) * len(spec["selection_seeds"])
    )
    expected_evaluation = (
        len(spec["scenes"]) * len(spec["physics_domains"])
        * len(spec["candidates"]) * len(spec["evaluation_seeds"])
    )
    if len(selection_rows) != expected_selection or len(evaluation_rows) != expected_evaluation:
        print(json.dumps({
            "completed_runs": len(rows),
            "complete": False,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
        }, indent=2))
        return 0
    mapping, global_candidate, context_summary, pooled = select_mapping(
        selection_rows, spec["selection_rmse_slack_m"]
    )
    result = evaluate_mapping(
        evaluation_rows,
        mapping,
        global_candidate,
        spec["evaluation_rmse_noninferiority_margin_m"],
        spec["bootstrap_seed"],
    )
    serial_mapping = {
        "%s__%s" % context: candidate
        for context, candidate in sorted(mapping.items())
    }
    summary = {
        "schema_version": 1,
        "design_id": spec["design_id"],
        "completed_runs": len(rows),
        "selection_runs": len(selection_rows),
        "evaluation_runs": len(evaluation_rows),
        "context_mapping": serial_mapping,
        "global_candidate": global_candidate,
        "selection_context_summary": {
            "%s__%s" % context: values
            for context, values in sorted(context_summary.items())
        },
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
