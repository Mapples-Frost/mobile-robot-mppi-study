#!/usr/bin/env python3
"""Run the frozen L243 direct-control teacher collection from YAML."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rl.collect_scripted_subgoal_demonstrations import main as collect_main
from mobile_robot_mppi.core.config import load_yaml


DEFAULT_CONFIG = "configs/rl/direct_control_bc_teacher_l243.yaml"


def collector_argv(config_path, output_dir=None):
    config_path = str(config_path)
    config = load_yaml(ROOT / config_path)
    collection = dict(config.get("collection", {}))
    required = (
        "output_dir",
        "scene_configs",
        "seeds",
        "split_seed",
        "lookahead",
        "cruise_speed",
        "yaw_gain",
        "num_samples",
        "max_steps",
    )
    missing = [key for key in required if key not in collection]
    if missing:
        raise ValueError("missing frozen L243 collection fields: %s" % missing)
    scenes = [str(value) for value in collection["scene_configs"]]
    seeds = [int(value) for value in collection["seeds"]]
    if len(scenes) != 6 or len(set(scenes)) != 6:
        raise ValueError("L243 collection requires exactly six unique scenes")
    if len(seeds) != 15 or len(set(seeds)) != 15:
        raise ValueError("L243 collection requires exactly fifteen unique seeds")

    argv = [
        "--rl-config", config_path,
        "--configs", *scenes,
        "--output-dir", str(output_dir or collection["output_dir"]),
        "--seeds", ",".join(str(value) for value in seeds),
        "--split-seed", str(int(collection["split_seed"])),
        "--validation-fraction", str(float(collection.get("validation_fraction", 0.2))),
        "--test-fraction", str(float(collection.get("test_fraction", 0.2))),
        "--route-source", "task_points",
        "--teacher-action-mode", "direct_control",
        "--teacher-pose-source", "ground_truth",
        "--teacher-cruise-speed", str(float(collection["cruise_speed"])),
        "--teacher-yaw-gain", str(float(collection["yaw_gain"])),
        "--lookahead", str(float(collection["lookahead"])),
        "--num-samples", str(int(collection["num_samples"])),
        "--max-steps", str(int(collection["max_steps"])),
    ]
    if bool(collection.get("allow_non_ground_truth_localization", False)):
        argv.append("--allow-non-ground-truth-localization")
    return argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    return collect_main(collector_argv(args.config, args.output_dir))


if __name__ == "__main__":
    main()
