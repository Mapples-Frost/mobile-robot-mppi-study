#!/usr/bin/env python3
"""Run baseline/RL/gated-RL and optional ICODE+RL research ablations."""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _list(text, cast=str):
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def _variants(rl_checkpoint, icode_checkpoint=None):
    variants = [
        {"name": "mppi", "prior": "goal_warm_start", "prediction": "nominal"},
        {"name": "rl_mppi", "prior": "rl", "gate": "none", "prediction": "nominal"},
        {"name": "gated_rl_mppi", "prior": "rl", "gate": "ood", "prediction": "nominal"},
    ]
    if icode_checkpoint:
        variants.extend((
            {"name": "icode_mppi", "prior": "goal_warm_start", "prediction": "icode_residual", "residual": icode_checkpoint},
            {"name": "icode_rl_mppi", "prior": "rl", "gate": "none", "prediction": "icode_residual", "residual": icode_checkpoint},
            {"name": "icode_gated_rl_mppi", "prior": "rl", "gate": "ood", "prediction": "icode_residual", "residual": icode_checkpoint},
        ))
    return variants


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="comma-separated scene YAML files")
    parser.add_argument("--rl-checkpoint", required=True)
    parser.add_argument("--icode-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="11,12,13,14,15")
    parser.add_argument("--samples", default="100,200,400")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--ood-soft-threshold", type=float)
    parser.add_argument("--ood-hard-threshold", type=float)
    parser.add_argument("--use-critic-disagreement", action="store_true")
    parser.add_argument("--critic-soft-threshold", type=float)
    parser.add_argument("--critic-hard-threshold", type=float)
    parser.add_argument(
        "--pose-source", choices=("wheel_odometry", "ground_truth")
    )
    parser.add_argument(
        "--twist-source", choices=("wheel_odometry", "ground_truth")
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    scene_files = _list(args.configs)
    seeds = _list(args.seeds, int)
    samples = _list(args.samples, int)
    if args.smoke:
        scene_files = scene_files[:1]
        seeds = seeds[:1]
        samples = samples[:1]
    output = Path(args.output_dir).resolve()
    rows = []
    for scene_file in scene_files:
        base = load_yaml(scene_file)
        scene_name = str(base.get("scene", {}).get("name", Path(scene_file).stem))
        for variant in _variants(args.rl_checkpoint, args.icode_checkpoint):
            for sample_count in samples:
                for seed in seeds:
                    config = copy.deepcopy(base)
                    config["experiment"]["seed"] = seed
                    if args.max_steps is not None:
                        config["experiment"]["max_steps"] = args.max_steps
                    if args.smoke:
                        config["experiment"]["max_steps"] = min(
                            int(config["experiment"]["max_steps"]), 20
                        )
                    config["planner"]["num_samples"] = sample_count
                    if args.pose_source is not None:
                        config.setdefault("sensors", {})[
                            "pose_source"
                        ] = args.pose_source
                    if args.twist_source is not None:
                        config.setdefault("sensors", {})[
                            "twist_source"
                        ] = args.twist_source
                    config["planner"]["sampling_prior"] = variant["prior"]
                    config["planner"]["prediction_mode"] = variant["prediction"]
                    config.setdefault("rl", {})
                    config["rl"].update({
                        "enabled": variant["prior"] == "rl",
                        "checkpoint": str(Path(args.rl_checkpoint).resolve()),
                    })
                    config["rl"].setdefault("gate", {})["mode"] = variant.get("gate", "none")
                    if args.ood_soft_threshold is not None:
                        config["rl"]["gate"][
                            "ood_soft_threshold"
                        ] = args.ood_soft_threshold
                    if args.ood_hard_threshold is not None:
                        config["rl"]["gate"][
                            "ood_hard_threshold"
                        ] = args.ood_hard_threshold
                    if args.use_critic_disagreement:
                        config["rl"]["gate"][
                            "use_critic_disagreement"
                        ] = True
                    if args.critic_soft_threshold is not None:
                        config["rl"]["gate"][
                            "critic_soft_threshold"
                        ] = args.critic_soft_threshold
                    if args.critic_hard_threshold is not None:
                        config["rl"]["gate"][
                            "critic_hard_threshold"
                        ] = args.critic_hard_threshold
                    if variant.get("residual"):
                        config["planner"]["checkpoint"] = str(Path(variant["residual"]).resolve())
                    run_id = "%s_%s_k%d_s%d" % (
                        scene_name, variant["name"], sample_count, seed
                    )
                    result = ExperimentRunner(
                        config, ROOT, output / "runs" / run_id, headless=True
                    ).run()
                    row = dict(result.summary)
                    row.update({
                        "scene": scene_name,
                        "method": variant["name"],
                        "num_samples": sample_count,
                        "seed": seed,
                        "pose_source": str(config.get("sensors", {}).get(
                            "pose_source", "wheel_odometry"
                        )),
                        "twist_source": str(config.get("sensors", {}).get(
                            "twist_source", "wheel_odometry"
                        )),
                    })
                    rows.append(row)
    groups = {}
    for row in rows:
        key = (row["scene"], row["method"], row["num_samples"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (scene, method, samples_count), values in sorted(groups.items()):
        summaries.append({
            "scene": scene,
            "method": method,
            "num_samples": samples_count,
            "pose_source": values[0]["pose_source"],
            "twist_source": values[0]["twist_source"],
            "seeds": len(values),
            "success_rate": float(np.mean([float(value["success"]) for value in values])),
            "collision_rate": float(np.mean([float(value["collision"]) for value in values])),
            "final_goal_distance_mean": float(np.mean([value["final_goal_distance"] for value in values])),
            "planner_compute_ms_mean": float(np.mean([value["planner_compute_ms_mean"] for value in values])),
            "rl_gate_alpha_mean": float(np.mean([value["rl_gate_alpha_mean"] for value in values])),
        })
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("episodes.csv", rows), ("summary.csv", summaries)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"episodes": rows, "summary": summaries}, handle, indent=2, sort_keys=True)
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
