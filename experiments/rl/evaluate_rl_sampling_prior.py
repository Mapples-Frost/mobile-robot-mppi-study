#!/usr/bin/env python3
"""Evaluate a trained SAC prior through the standard MuJoCo runner."""

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

from mobile_robot_mppi.core.config import deep_merge, load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _seeds(text):
    return [int(value) for value in text.split(",") if value.strip()]


def _apply_scene_config(base, scene_path):
    """Put one explicit evaluation scene under the selected RL setup."""

    if scene_path is None:
        return base
    scene = load_yaml(scene_path)
    overlay = {
        "planner": copy.deepcopy(base["planner"]),
        "rl": copy.deepcopy(base.get("rl", {})),
    }
    if "memory" in base:
        overlay["memory"] = copy.deepcopy(base["memory"])
    return deep_merge(scene, overlay)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--scene-config",
        help=(
            "explicit plant/perception/geometry config; planner and RL settings "
            "continue to come from --config"
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="11,12,13,14,15")
    parser.add_argument(
        "--gate-mode", choices=("none", "fixed", "ood", "exploration")
    )
    parser.add_argument("--fixed-alpha", type=float)
    parser.add_argument("--ood-soft-threshold", type=float)
    parser.add_argument("--ood-hard-threshold", type=float)
    parser.add_argument("--use-critic-disagreement", action="store_true")
    parser.add_argument("--critic-soft-threshold", type=float)
    parser.add_argument("--critic-hard-threshold", type=float)
    parser.add_argument(
        "--exploration-signal", choices=("ood", "critic_disagreement")
    )
    parser.add_argument("--exploration-soft-threshold", type=float)
    parser.add_argument("--exploration-hard-threshold", type=float)
    parser.add_argument("--exploration-trigger-window-s", type=float)
    parser.add_argument("--disable-exploration-latch", action="store_true")
    parser.add_argument("--prediction-mode", choices=("nominal", "mlp_residual", "icode_residual"))
    parser.add_argument("--residual-checkpoint")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--goal-running-weight", type=float)
    parser.add_argument("--goal-terminal-weight", type=float)
    parser.add_argument("--near-goal-fallback", action="store_true")
    parser.add_argument("--near-goal-full-fallback-distance", type=float, default=0.55)
    parser.add_argument("--near-goal-full-rl-distance", type=float, default=1.75)
    parser.add_argument(
        "--pose-source", choices=("wheel_odometry", "ground_truth")
    )
    parser.add_argument(
        "--twist-source", choices=("wheel_odometry", "ground_truth")
    )
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args(argv)
    base = _apply_scene_config(load_yaml(args.config), args.scene_config)
    base.setdefault("rl", {})
    base["rl"].update({"enabled": True, "checkpoint": str(Path(args.checkpoint).resolve())})
    if args.gate_mode:
        base["rl"].setdefault("gate", {})["mode"] = args.gate_mode
    if args.fixed_alpha is not None:
        base["rl"].setdefault("gate", {})["fixed_alpha"] = args.fixed_alpha
    if args.ood_soft_threshold is not None:
        base["rl"].setdefault("gate", {})[
            "ood_soft_threshold"
        ] = args.ood_soft_threshold
    if args.ood_hard_threshold is not None:
        base["rl"].setdefault("gate", {})[
            "ood_hard_threshold"
        ] = args.ood_hard_threshold
    if args.use_critic_disagreement:
        base["rl"].setdefault("gate", {})[
            "use_critic_disagreement"
        ] = True
    if args.critic_soft_threshold is not None:
        base["rl"].setdefault("gate", {})[
            "critic_soft_threshold"
        ] = args.critic_soft_threshold
    if args.critic_hard_threshold is not None:
        base["rl"].setdefault("gate", {})[
            "critic_hard_threshold"
        ] = args.critic_hard_threshold
    if args.exploration_signal is not None:
        base["rl"].setdefault("gate", {})[
            "exploration_signal"
        ] = args.exploration_signal
    if args.exploration_soft_threshold is not None:
        base["rl"].setdefault("gate", {})[
            "exploration_soft_threshold"
        ] = args.exploration_soft_threshold
    if args.exploration_hard_threshold is not None:
        base["rl"].setdefault("gate", {})[
            "exploration_hard_threshold"
        ] = args.exploration_hard_threshold
    if args.exploration_trigger_window_s is not None:
        base["rl"].setdefault("gate", {})[
            "exploration_trigger_window_s"
        ] = args.exploration_trigger_window_s
    if args.disable_exploration_latch:
        base["rl"].setdefault("gate", {})["exploration_latch"] = False
    if args.near_goal_fallback:
        base["rl"].setdefault("gate", {}).update({
            "near_goal_fallback_enabled": True,
            "near_goal_full_fallback_distance": args.near_goal_full_fallback_distance,
            "near_goal_full_rl_distance": args.near_goal_full_rl_distance,
        })
    base["planner"]["sampling_prior"] = "rl"
    if args.pose_source is not None:
        base.setdefault("sensors", {})["pose_source"] = args.pose_source
    if args.twist_source is not None:
        base.setdefault("sensors", {})["twist_source"] = args.twist_source
    if args.prediction_mode:
        base["planner"]["prediction_mode"] = args.prediction_mode
    if args.residual_checkpoint:
        base["planner"]["checkpoint"] = str(Path(args.residual_checkpoint).resolve())
    if args.goal_running_weight is not None:
        base["planner"]["goal_running_weight"] = args.goal_running_weight
    if args.goal_terminal_weight is not None:
        base["planner"]["goal_terminal_weight"] = args.goal_terminal_weight
    if base["planner"].get("prediction_mode") in ("mlp_residual", "icode_residual") and not base["planner"].get("checkpoint"):
        raise ValueError("learned residual prediction requires --residual-checkpoint")
    output = Path(args.output_dir).resolve()
    rows = []
    selected_seeds = _seeds(args.seeds)
    if args.view and len(selected_seeds) != 1:
        raise ValueError("--view requires exactly one seed")
    for seed in selected_seeds:
        config = copy.deepcopy(base)
        config["experiment"]["seed"] = seed
        config["experiment"]["name"] = "rl_prior_seed_%d" % seed
        if args.max_steps is not None:
            config["experiment"]["max_steps"] = args.max_steps
        if args.num_samples is not None:
            config["planner"]["num_samples"] = args.num_samples
        result = ExperimentRunner(
            config,
            ROOT,
            output / "runs" / ("seed_%d" % seed),
            headless=not args.view,
        ).run()
        row = dict(result.summary)
        row["seed"] = seed
        rows.append(row)
    numeric = (
        "final_goal_distance",
        "trajectory_length",
        "minimum_clearance",
        "control_jerk",
        "planner_compute_ms_mean",
        "rl_gate_alpha_mean",
        "rl_ood_score_mean",
        "rl_critic_disagreement_mean",
        "rl_exploration_activation_mean",
        "rl_exploration_latch_alpha_mean",
    )
    summary = {
        "seeds": len(rows),
        "scene": str(base.get("scene", {}).get("name", "unknown")),
        "success_rate": float(np.mean([float(row["success"]) for row in rows])),
        "collision_rate": float(np.mean([float(row["collision"]) for row in rows])),
        "pose_source": str(base.get("sensors", {}).get(
            "pose_source", "wheel_odometry"
        )),
        "twist_source": str(base.get("sensors", {}).get(
            "twist_source", "wheel_odometry"
        )),
        "gate_mode": str(base.get("rl", {}).get("gate", {}).get(
            "mode", "none"
        )),
        "ood_soft_threshold": float(base.get("rl", {}).get("gate", {}).get(
            "ood_soft_threshold", 3.0
        )),
        "ood_hard_threshold": float(base.get("rl", {}).get("gate", {}).get(
            "ood_hard_threshold", 7.0
        )),
        "use_critic_disagreement": bool(base.get("rl", {}).get(
            "gate", {}
        ).get("use_critic_disagreement", False)),
        "critic_soft_threshold": float(base.get("rl", {}).get("gate", {}).get(
            "critic_soft_threshold", 2.0
        )),
        "critic_hard_threshold": float(base.get("rl", {}).get("gate", {}).get(
            "critic_hard_threshold", 10.0
        )),
        "exploration_signal": str(base.get("rl", {}).get("gate", {}).get(
            "exploration_signal", "critic_disagreement"
        )),
        "exploration_soft_threshold": float(base.get("rl", {}).get(
            "gate", {}
        ).get("exploration_soft_threshold", 0.05)),
        "exploration_hard_threshold": float(base.get("rl", {}).get(
            "gate", {}
        ).get("exploration_hard_threshold", 0.20)),
        "exploration_latch": bool(base.get("rl", {}).get("gate", {}).get(
            "exploration_latch", True
        )),
        "exploration_trigger_window_s": float(base.get("rl", {}).get(
            "gate", {}
        ).get("exploration_trigger_window_s", 0.50)),
    }
    for name in numeric:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        summary[name + "_mean"] = float(np.mean(values)) if values else None
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
