#!/usr/bin/env python3
"""Matched-seed nominal/MLP/ICODE benchmark across static MuJoCo scenes."""

import argparse
import copy
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import canonical_json, config_hash, git_sha, load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_CONFIGS = (
    "configs/research/mujoco_strong_mppi_baseline.yaml",
    "configs/research/mujoco_lab_complex.yaml",
    "configs/research/mujoco_narrow_corridor.yaml",
    "configs/research/mujoco_u_trap_long_board.yaml",
)


def parse_seeds(value):
    seeds = [int(item) for item in str(value).split(",") if item.strip()]
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be a non-empty list of nonnegative integers")
    return seeds


def validate_shared_protocol(configs):
    sections = ("task", "state_space", "action_space", "plant", "sensors", "perception")
    planner_exclusions = {"prediction_mode", "checkpoint", "seed"}
    reference = configs[0]
    for index, config in enumerate(configs[1:], start=1):
        for section in sections:
            if canonical_json(reference[section]) != canonical_json(config[section]):
                raise ValueError("scene config %d changes frozen section %s" % (index, section))
        left = {k: v for k, v in reference["planner"].items() if k not in planner_exclusions}
        right = {k: v for k, v in config["planner"].items() if k not in planner_exclusions}
        if canonical_json(left) != canonical_json(right):
            raise ValueError("scene config %d changes frozen planner parameters" % index)


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["scene"], row["method"]), []).append(row)
    summary = []
    numeric = (
        "final_goal_distance", "trajectory_length", "minimum_clearance",
        "control_jerk", "applied_control_jerk", "mean_abs_omega",
        "safety_interventions", "stuck_steps", "spin_steps",
        "planner_compute_ms_mean", "planner_compute_ms_p95",
        "planner_compute_ms_max", "planner_deadline_misses",
        "planner_deadline_miss_rate",
    )
    for (scene, method), values in sorted(groups.items()):
        record = {
            "scene": scene,
            "method": method,
            "num_runs": len(values),
            "success_rate": float(np.mean([row["success"] for row in values])),
            "collision_rate": float(np.mean([row["collision"] for row in values])),
        }
        for name in numeric:
            available = [float(row[name]) for row in values if row.get(name) is not None]
            record[name + "_mean"] = float(np.mean(available)) if available else None
            record[name + "_std"] = float(np.std(available)) if available else None
        reached = [float(row["time_to_goal_s"]) for row in values if row.get("time_to_goal_s") is not None]
        record["time_to_goal_s_mean_successes"] = float(np.mean(reached)) if reached else None
        summary.append(record)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--mlp-checkpoint")
    parser.add_argument("--icode-checkpoint")
    args = parser.parse_args(argv)
    if args.num_samples <= 0:
        raise ValueError("num-samples must be positive")
    configs = [load_yaml(path) for path in args.configs]
    validate_shared_protocol(configs)
    seeds = parse_seeds(args.seeds)
    methods = [("nominal", "nominal", None)]
    if args.mlp_checkpoint:
        methods.append(("mlp", "mlp_residual", args.mlp_checkpoint))
    if args.icode_checkpoint:
        methods.append(("icode", "icode_residual", args.icode_checkpoint))
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory is not empty: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for base in configs:
        scene = str(base.get("scene", {}).get("name", "unknown"))
        for method, prediction_mode, checkpoint in methods:
            for seed in seeds:
                config = copy.deepcopy(base)
                config["experiment"]["seed"] = seed
                if args.max_steps is not None:
                    config["experiment"]["max_steps"] = int(args.max_steps)
                config["planner"]["seed"] = seed
                config["planner"]["num_samples"] = int(args.num_samples)
                config["planner"]["prediction_mode"] = prediction_mode
                if checkpoint:
                    config["planner"]["checkpoint"] = checkpoint
                run_dir = output / "runs" / scene / method / ("seed_%04d" % seed)
                result = ExperimentRunner(config, ROOT, run_dir, headless=True).run()
                row = dict(result.summary)
                row.update({
                    "scene": scene,
                    "method": method,
                    "seed": seed,
                    "num_samples": int(args.num_samples),
                })
                rows.append(row)
                print(
                    "scene=%s method=%s seed=%d success=%s collision=%s distance=%.3f"
                    % (scene, method, seed, row["success"], row["collision"], row["final_goal_distance"])
                )
    summary = aggregate(rows)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_fields = sorted(set().union(*(row.keys() for row in summary)))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)
    payload = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(ROOT),
            "seeds": seeds,
            "num_samples": int(args.num_samples),
            "configs": [
                {"path": str(Path(path).resolve()), "hash": config_hash(config)}
                for path, config in zip(args.configs, configs)
            ],
            "checkpoints": {"mlp": args.mlp_checkpoint, "icode": args.icode_checkpoint},
        },
        "summary": summary,
        "episodes": rows,
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
