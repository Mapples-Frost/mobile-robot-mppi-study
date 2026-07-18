#!/usr/bin/env python3
"""Pretrain the local-subgoal SAC actor from privileged teacher episodes."""

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import deep_merge, load_yaml
from mobile_robot_mppi.rl.behavior_cloning import BehaviorCloningTrainer


def _resolve(path):
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def _environment_config(config):
    training = config.get("rl", {}).get("training", {})
    scenes = list(training.get("scene_configs", ()))
    if not scenes:
        return copy.deepcopy(config)
    scene = load_yaml(_resolve(scenes[0]))
    return deep_merge(scene, {
        "planner": copy.deepcopy(config["planner"]),
        "rl": copy.deepcopy(config["rl"]),
    })


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-bc")
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help=(
            "explicitly permit an older BC checkpoint without the full "
            "resume contract/best-actor snapshot"
        ),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.allow_legacy_resume and not args.resume_bc:
        parser.error("--allow-legacy-resume requires --resume-bc")
    config = load_yaml(_resolve(args.config))
    bc = config.setdefault("rl", {}).setdefault("behavior_cloning", {})
    if args.epochs is not None:
        bc["epochs"] = int(args.epochs)
    if args.seed is not None:
        if not 0 <= int(args.seed) <= 2 ** 32 - 1:
            raise ValueError("--seed must be in [0, 2**32 - 1]")
        bc["seed"] = int(args.seed)
    if args.device is not None:
        bc["device"] = args.device
    if args.smoke:
        bc["epochs"] = min(int(bc.get("epochs", 3)), 3)
        bc["batch_size"] = min(int(bc.get("batch_size", 32)), 32)
        bc["early_stopping_patience"] = 0
    trainer = BehaviorCloningTrainer(
        _environment_config(config),
        config,
        _resolve(args.dataset_dir),
        ROOT,
        _resolve(args.output_dir),
    )
    if args.resume_bc:
        trainer.resume(
            _resolve(args.resume_bc),
            allow_legacy=bool(args.allow_legacy_resume),
        )
    result = trainer.run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
