#!/usr/bin/env python3
"""Preregistered local-branch covariance heterogeneity oracle for L90."""

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
from mobile_robot_mppi.rl.contextual_bandit import (
    polyline_geometry_features,
    polyline_pose_at_progress,
    polyline_window,
    project_polyline_progress,
)
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _context(row):
    return str(row["scene"]), str(row["anchor_id"])


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((_context(row), row["candidate"]), []).append(row)
    output = {}
    for (context, candidate), values in grouped.items():
        output.setdefault(context, {})[candidate] = {
            "branches": len(values),
            "collision_rate": float(np.mean([float(v["collision"]) for v in values])),
            "cross_track_rmse": float(np.mean([
                float(v["cross_track_rmse"]) for v in values
            ])),
            "local_progress_m": float(np.mean([
                float(v["local_progress_m"]) for v in values
            ])),
        }
    return output


def _choose(candidates, rmse_slack):
    safe = {
        name: value for name, value in candidates.items()
        if value["collision_rate"] <= 1e-12
    }
    if not safe:
        raise RuntimeError("no collision-free local covariance candidate")
    minimum_rmse = min(value["cross_track_rmse"] for value in safe.values())
    precise = {
        name: value for name, value in safe.items()
        if value["cross_track_rmse"] <= minimum_rmse + float(rmse_slack) + 1e-12
    }
    return max(
        precise,
        key=lambda name: (precise[name]["local_progress_m"], name),
    )


def select_mapping(rows, rmse_slack):
    context_summary = _summary(rows)
    mapping = {
        context: _choose(candidates, rmse_slack)
        for context, candidates in sorted(context_summary.items())
    }
    pooled = {}
    for candidate in sorted({row["candidate"] for row in rows}):
        values = [row for row in rows if row["candidate"] == candidate]
        pooled[candidate] = {
            "branches": len(values),
            "collision_rate": float(np.mean([float(v["collision"]) for v in values])),
            "cross_track_rmse": float(np.mean([
                float(v["cross_track_rmse"]) for v in values
            ])),
            "local_progress_m": float(np.mean([
                float(v["local_progress_m"]) for v in values
            ])),
        }
    return mapping, _choose(pooled, rmse_slack), context_summary, pooled


def evaluate_mapping(rows, mapping, global_candidate, rmse_margin, bootstrap_seed):
    indexed = {
        (_context(row), row["physics_domain"], int(row["seed"]), row["candidate"]): row
        for row in rows
    }
    differences = {
        "local_progress_m": {},
        "cross_track_rmse": {},
        "control_jerk": {},
    }
    safety = []
    pairs = 0
    for context, candidate in sorted(mapping.items()):
        blocks = sorted({
            (domain, seed)
            for (key, domain, seed, _candidate) in indexed
            if key == context
        })
        for domain, seed in blocks:
            local = indexed[(context, domain, seed, candidate)]
            fixed = indexed[(context, domain, seed, global_candidate)]
            pairs += 1
            for metric in differences:
                differences[metric].setdefault(context, []).append(
                    float(local[metric]) - float(fixed[metric])
                )
            safety.append(float(local["collision"]) - float(fixed["collision"]))
    result = {"paired_branches": pairs}
    for offset, (metric, values) in enumerate(differences.items()):
        flat = [item for group in values.values() for item in group]
        result[metric + "_delta_mean"] = float(np.mean(flat))
        result[metric + "_delta_ci95"] = _hierarchical_ci(
            values, int(bootstrap_seed) + offset
        )
    different = sum(value != global_candidate for value in mapping.values())
    result.update({
        "global_candidate": global_candidate,
        "selected_candidate_count": len(set(mapping.values())),
        "contexts_total": len(mapping),
        "contexts_different_from_global": different,
        "collision_delta_mean": float(np.mean(safety)),
    })
    result["heterogeneity_gate_passed"] = bool(
        result["selected_candidate_count"] >= 2
        and different >= int(np.ceil(0.25 * len(mapping)))
    )
    result["progress_gate_passed"] = bool(
        result["local_progress_m_delta_ci95"][0] > 0.0
    )
    result["precision_gate_passed"] = bool(
        result["cross_track_rmse_delta_ci95"][1] <= float(rmse_margin)
    )
    result["safety_gate_passed"] = bool(result["collision_delta_mean"] <= 0.0)
    result["primary_gate_passed"] = bool(
        result["heterogeneity_gate_passed"]
        and result["progress_gate_passed"]
        and result["precision_gate_passed"]
        and result["safety_gate_passed"]
    )
    return result


def _last_xy(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("local branch trajectory is empty")
    return float(rows[-1]["x"]), float(rows[-1]["y"])


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
            scene_config = load_yaml(ROOT / scene_path)
            points = np.asarray(scene_config["task"]["points"], dtype=np.float64)
            total = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            for anchor_fraction in spec["anchor_fractions"]:
                anchor_progress = float(anchor_fraction) * total
                for domain in spec["physics_domains"]:
                    for candidate in spec["candidates"]:
                        for seed in seeds:
                            schedule.append({
                                "split": split,
                                "scene_path": scene_path,
                                "anchor_fraction": float(anchor_fraction),
                                "anchor_progress": anchor_progress,
                                "physics_domain": domain,
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
        domain = item["physics_domain"]
        candidate = item["candidate"]
        config = _apply_scene_config(base, ROOT / item["scene_path"])
        config = deep_merge(config, {"plant": domain.get("plant_override", {})})
        points = np.asarray(config["task"]["points"], dtype=np.float64)
        point, theta = polyline_pose_at_progress(points, item["anchor_progress"])
        config["experiment"]["seed"] = item["seed"]
        config["experiment"]["max_steps"] = int(spec["branch_steps"])
        config["experiment"]["initial_state"] = [
            float(point[0]), float(point[1]), theta,
            float(spec["initial_speed_mps"]), 0.0,
        ]
        anchor_id = "p%03d" % int(round(100.0 * item["anchor_fraction"]))
        config["experiment"]["name"] = "l90_%s_%s_%s_%s_%s_%d" % (
            item["split"], config["scene"]["name"], anchor_id,
            domain["name"], candidate["name"], item["seed"],
        )
        config["planner"]["sampling_prior"] = "fixed_covariance"
        config["planner"]["fixed_covariance_scale"] = list(candidate["scale"])
        config.setdefault("rl", {})["enabled"] = False
        config["rl"].pop("checkpoint", None)
        run_dir = (
            output / "runs" / item["split"] / domain["name"]
            / config["scene"]["name"] / anchor_id / candidate["name"]
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
        final_xy = _last_xy(run_dir / "trajectory.csv")
        final_progress = project_polyline_progress(
            points,
            final_xy,
            minimum=max(0.0, item["anchor_progress"] - 0.1),
            maximum=item["anchor_progress"]
            + float(spec["maximum_projection_advance_m"]),
        )
        local_points = polyline_window(
            points,
            item["anchor_progress"],
            float(spec["local_window_distance_m"]),
        )
        row = dict(summary)
        row.update({
            "split": item["split"],
            "scene": str(config["scene"]["name"]),
            "anchor_id": anchor_id,
            "anchor_fraction": float(item["anchor_fraction"]),
            "anchor_progress_m": float(item["anchor_progress"]),
            "local_progress_m": max(0.0, final_progress - item["anchor_progress"]),
            "local_features": json.dumps(polyline_geometry_features(local_points).tolist()),
            "physics_domain": str(domain["name"]),
            "candidate": str(candidate["name"]),
            "scale": json.dumps(candidate["scale"]),
            "seed": int(item["seed"]),
        })
        rows.append(row)
    episode_name = (
        "branches.csv" if args.shard_count == 1
        else "branches_shard_%03d_of_%03d.csv" % (args.shard_index, args.shard_count)
    )
    _write_csv(output / episode_name, rows)
    selection = [row for row in rows if row["split"] == "selection"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    factor = (
        len(spec["scenes"]) * len(spec["anchor_fractions"])
        * len(spec["physics_domains"]) * len(spec["candidates"])
    )
    if (
        len(selection) != factor * len(spec["selection_seeds"])
        or len(evaluation) != factor * len(spec["evaluation_seeds"])
    ):
        print(json.dumps({"complete": False, "completed_runs": len(rows)}, indent=2))
        return 0
    mapping, global_candidate, context_summary, pooled = select_mapping(
        selection, spec["selection_rmse_slack_m"]
    )
    result = evaluate_mapping(
        evaluation,
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
        "selection_runs": len(selection),
        "evaluation_runs": len(evaluation),
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
