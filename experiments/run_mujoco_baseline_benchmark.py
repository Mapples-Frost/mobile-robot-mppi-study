#!/usr/bin/env python3
"""Run the frozen nominal MPPI + MuJoCo baseline over multiple seeds."""

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mobile_robot_mppi.core.config import config_hash, git_sha, load_yaml
from mobile_robot_mppi.evaluation.benchmark import write_benchmark_artifacts
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def validate_nominal_mujoco_baseline(config):
    if config.get("plant", {}).get("backend") != "mujoco_diff_drive":
        raise ValueError("baseline benchmark requires plant.backend=mujoco_diff_drive")
    if config.get("planner", {}).get("prediction_mode", "nominal") != "nominal":
        raise ValueError("baseline benchmark requires planner.prediction_mode=nominal")
    if bool(config.get("memory", {}).get("enable", False)):
        raise ValueError("baseline benchmark requires memory.enable=false")
    if bool(config.get("rl", {}).get("enabled", False)):
        raise ValueError("baseline benchmark requires rl.enabled=false")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Multi-seed nominal MPPI benchmark on the MuJoCo true plant"
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs/research/mujoco_strong_mppi_baseline.yaml")
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-samples", type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_yaml(args.config)
    validate_nominal_mujoco_baseline(config)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("benchmark output directory is not empty: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        run_config = copy.deepcopy(config)
        run_config["experiment"]["seed"] = int(seed)
        run_config["planner"]["seed"] = int(seed)
        if args.max_steps is not None:
            run_config["experiment"]["max_steps"] = int(args.max_steps)
        if args.num_samples is not None:
            run_config["planner"]["num_samples"] = int(args.num_samples)
        run_dir = output / ("seed_%04d" % seed)
        result = ExperimentRunner(run_config, ROOT, run_dir, headless=True).run()
        row = {"seed": int(seed)}
        row.update(result.summary)
        rows.append(row)
        print("seed=%d success=%s final_goal_distance=%.4f" % (
            seed, result.summary["success"], result.summary["final_goal_distance"]
        ))
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "config_hash": config_hash(config),
        "config_path": str(Path(args.config).resolve()),
        "seeds": [int(seed) for seed in args.seeds],
        "baseline_contract": {
            "plant_backend": "mujoco_diff_drive",
            "prediction_mode": "nominal",
            "memory_enabled": False,
            "rl_enabled": False,
        },
    }
    payload = write_benchmark_artifacts(output, rows, metadata)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    print("artifacts:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
