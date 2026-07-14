#!/usr/bin/env python3
"""Evaluate BC action imitation on an episode-held-out dataset split."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.rl.behavior_cloning import evaluate_behavior_cloning
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint
from mobile_robot_mppi.rl.demonstrations import load_demonstration_split
from mobile_robot_mppi.rl.observation import RunningNormalizer
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    payload = load_sac_checkpoint(args.checkpoint, map_location=args.device)
    state = payload["agent"]
    agent = SACAgent(
        int(state["observation_dim"]),
        int(state["action_dim"]),
        SACConfig.from_mapping(state["config"]),
        device=args.device,
        seed=0,
    )
    agent.load_state_dict(state, load_optimizers=False)
    normalizer = RunningNormalizer.from_state_dict(payload["normalizer"])
    arrays = load_demonstration_split(args.dataset_dir, args.split)
    metrics = evaluate_behavior_cloning(agent, normalizer, arrays)
    result = {
        "policy_stage": str(
            payload.get("training_state", {}).get("phase", "unknown")
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "split": args.split,
        "metrics": metrics,
    }
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
