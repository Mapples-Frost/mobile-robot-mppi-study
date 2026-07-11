#!/usr/bin/env python3
"""Multi-seed physical-platform benchmark with sample-efficiency support."""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def parse_ints(value):
    return [int(item) for item in value.split(",") if item.strip()]


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["num_samples"]), []).append(row)
    summary = []
    for (method, samples), values in sorted(groups.items()):
        numeric = (
            "final_goal_distance", "trajectory_length", "mean_abs_omega", "control_jerk",
            "mean_slip_ratio", "safety_interventions", "planner_compute_ms_mean", "planner_compute_ms_max",
        )
        record = {
            "method": method,
            "num_samples": samples,
            "seeds": len(values),
            "success_rate": float(np.mean([float(value["success"]) for value in values])),
            "collision_rate": float(np.mean([float(value["collision"]) for value in values])),
        }
        for name in numeric:
            data = np.asarray([float(value[name]) for value in values], dtype=np.float64)
            record[name + "_mean"] = float(data.mean())
            record[name + "_std"] = float(data.std())
        summary.append(record)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/research/mujoco_diff_drive.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="11,12,13,14,15")
    parser.add_argument("--samples", default="50,100,200,400")
    parser.add_argument("--mlp-checkpoint")
    parser.add_argument("--icode-checkpoint")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    base = load_yaml(args.config)
    seeds = parse_ints(args.seeds)
    samples = parse_ints(args.samples)
    if args.smoke:
        seeds = seeds[:2]
        samples = samples[:1]
    methods = [("nominal", "nominal", None)]
    if args.mlp_checkpoint:
        methods.append(("mlp_residual", "mlp_residual", args.mlp_checkpoint))
    if args.icode_checkpoint:
        methods.append(("icode_residual", "icode_residual", args.icode_checkpoint))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method_name, mode, checkpoint in methods:
        for num_samples in samples:
            for seed in seeds:
                config = copy.deepcopy(base)
                config["experiment"]["seed"] = seed
                config["experiment"]["name"] = "%s_s%d_n%d" % (method_name, seed, num_samples)
                if args.max_steps is not None:
                    config["experiment"]["max_steps"] = args.max_steps
                if args.smoke:
                    config["experiment"]["max_steps"] = min(int(config["experiment"]["max_steps"]), 20)
                config["planner"]["num_samples"] = num_samples
                config["planner"]["prediction_mode"] = mode
                if checkpoint:
                    config["planner"]["checkpoint"] = checkpoint
                run_dir = output / "runs" / config["experiment"]["name"]
                result = ExperimentRunner(config, ROOT, run_dir, headless=True).run()
                row = dict(result.summary)
                row.update({"method": method_name, "seed": seed, "num_samples": num_samples})
                rows.append(row)
    summary = aggregate(rows)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"episodes": rows, "summary": summary}, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
