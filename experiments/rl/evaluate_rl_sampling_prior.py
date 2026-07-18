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
    perception_overrides = dict(
        base.get("rl", {}).get("training", {}).get(
            "perception_overrides", {}
        )
    )
    if perception_overrides:
        overlay["perception"] = deep_merge(
            scene.get("perception", {}), perception_overrides
        )
    if "memory" in base:
        overlay["memory"] = copy.deepcopy(base["memory"])
    return deep_merge(scene, overlay)


def _apply_physics_domain(base, specification_path, domain_name):
    if specification_path is None and domain_name is None:
        return base
    if specification_path is None or domain_name is None:
        raise ValueError(
            "--physics-domain-config and --physics-domain must be provided together"
        )
    specification = load_yaml(specification_path)
    matches = [
        item for item in specification.get("physics_domains", {}).get("domains", ())
        if str(item.get("name")) == str(domain_name)
    ]
    if len(matches) != 1:
        raise ValueError("physics domain must match exactly one configured domain")
    domain = matches[0]
    result = deep_merge(base, {"plant": domain.get("plant_override", {})})
    scene_name = str(result.get("scene", {}).get("name", "scene"))
    result.setdefault("scene", {})["name"] = "%s__%s" % (
        scene_name, domain_name
    )
    result.setdefault("experiment", {})["physics_domain"] = str(domain_name)
    result["experiment"]["physics_domain_role"] = str(
        domain.get("role", "unknown")
    )
    return result


def _require_explicit_scene_for_training_config(
    base, scene_path, allow_embedded_scene=False
):
    rl_config = base.get("rl", {})
    training = rl_config.get("training", {})
    training_markers = any(
        training.get(name)
        for name in (
            "scene_configs",
            "validation_scene_configs",
            "physics_domain_config",
        )
    ) or bool(rl_config.get("behavior_cloning"))
    if scene_path is None and training_markers and not allow_embedded_scene:
        raise ValueError(
            "RL training configs require explicit --scene-config for evaluation; "
            "use --allow-embedded-scene only for a documented legacy condition"
        )


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
    parser.add_argument(
        "--allow-embedded-scene",
        action="store_true",
        help="explicitly evaluate the scene embedded in --config",
    )
    parser.add_argument("--physics-domain-config")
    parser.add_argument("--physics-domain")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--fixed-covariance-scale",
        nargs=2,
        type=float,
        metavar=("V_SCALE", "OMEGA_SCALE"),
        help=(
            "evaluate the non-learning covariance comparator instead of a "
            "checkpoint; values multiply MPPI sampling standard deviations"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="11,12,13,14,15")
    parser.add_argument(
        "--gate-mode",
        choices=(
            "none",
            "fixed",
            "ood",
            "exploration",
            "complexity",
            "complexity_confidence",
        ),
    )
    parser.add_argument("--fixed-alpha", type=float)
    parser.add_argument("--ood-soft-threshold", type=float)
    parser.add_argument("--ood-hard-threshold", type=float)
    parser.add_argument("--use-critic-disagreement", action="store_true")
    parser.add_argument("--critic-soft-threshold", type=float)
    parser.add_argument("--critic-hard-threshold", type=float)
    parser.add_argument(
        "--correction-advantage-gate-mode",
        choices=("none", "hard", "lcb"),
    )
    parser.add_argument(
        "--correction-advantage-critic-source", choices=("online", "target")
    )
    parser.add_argument("--correction-advantage-threshold", type=float)
    parser.add_argument(
        "--correction-advantage-uncertainty-multiplier", type=float
    )
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
    base = load_yaml(args.config)
    _require_explicit_scene_for_training_config(
        base, args.scene_config, args.allow_embedded_scene
    )
    base = _apply_scene_config(base, args.scene_config)
    base = _apply_physics_domain(
        base, args.physics_domain_config, args.physics_domain
    )
    if (args.checkpoint is None) == (args.fixed_covariance_scale is None):
        raise ValueError(
            "provide exactly one of --checkpoint or --fixed-covariance-scale"
        )
    base.setdefault("rl", {})
    # Training-time localization overrides are part of the resolved
    # experimental condition.  Apply them explicitly here; command-line
    # overrides below remain the final authority and summary.json records the
    # resulting sources.
    sensor_overrides = dict(base["rl"].get("sensor_overrides", {}))
    if sensor_overrides:
        base.setdefault("sensors", {}).update(sensor_overrides)
    if args.checkpoint is not None:
        base["rl"].update({
            "enabled": True,
            "checkpoint": str(Path(args.checkpoint).resolve()),
        })
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
    if args.correction_advantage_gate_mode is not None:
        base["rl"].setdefault("gate", {})[
            "correction_advantage_gate_mode"
        ] = args.correction_advantage_gate_mode
    if args.correction_advantage_critic_source is not None:
        base["rl"].setdefault("gate", {})[
            "correction_advantage_critic_source"
        ] = args.correction_advantage_critic_source
    if args.correction_advantage_threshold is not None:
        base["rl"].setdefault("gate", {})[
            "correction_advantage_threshold"
        ] = args.correction_advantage_threshold
    if args.correction_advantage_uncertainty_multiplier is not None:
        base["rl"].setdefault("gate", {})[
            "correction_advantage_uncertainty_multiplier"
        ] = args.correction_advantage_uncertainty_multiplier
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
    if args.fixed_covariance_scale is None:
        base["planner"]["sampling_prior"] = "rl"
    else:
        if any(value <= 0.0 for value in args.fixed_covariance_scale):
            raise ValueError("fixed covariance scales must be positive")
        base["planner"]["sampling_prior"] = "fixed_covariance"
        base["planner"]["fixed_covariance_scale"] = list(
            args.fixed_covariance_scale
        )
        base["rl"]["enabled"] = False
        base["rl"].pop("checkpoint", None)
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
    output.mkdir(parents=True, exist_ok=True)
    selected_seeds = _seeds(args.seeds)
    if not selected_seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    if args.view and len(selected_seeds) != 1:
        raise ValueError("--view requires exactly one seed")
    snapshot = {
        "schema_version": 1,
        "source_config": str(Path(args.config).resolve()),
        "scene_config": (
            None if args.scene_config is None
            else str(Path(args.scene_config).resolve())
        ),
        "physics_domain_config": (
            None if args.physics_domain_config is None
            else str(Path(args.physics_domain_config).resolve())
        ),
        "physics_domain": args.physics_domain,
        "checkpoint": (
            None if args.checkpoint is None
            else str(Path(args.checkpoint).resolve())
        ),
        "fixed_covariance_scale": args.fixed_covariance_scale,
        "episode_seeds": selected_seeds,
        "max_steps_override": args.max_steps,
        "num_samples_override": args.num_samples,
        "headless": not bool(args.view),
        "resolved_config": base,
    }
    with (output / "evaluation_config_snapshot.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.write("\n")
    rows = []
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
        "cross_track_rmse",
        "cross_track_mean",
        "cross_track_max",
        "minimum_clearance",
        "control_jerk",
        "planner_compute_ms_mean",
        "rl_gate_alpha_mean",
        "rl_ood_score_mean",
        "rl_scene_complexity_score_mean",
        "rl_scene_complexity_score_max",
        "rl_gate_active_fraction",
        "rl_critic_disagreement_mean",
        "rl_exploration_activation_mean",
        "rl_exploration_latch_alpha_mean",
        "rl_correction_gate_alpha_mean",
        "rl_base_action_abs_mean",
        "rl_unit_correction_abs_mean",
        "rl_applied_correction_abs_mean",
        "rl_applied_correction_abs_max",
        "rl_raw_applied_correction_abs_mean",
        "rl_correction_advantage_gate_alpha_mean",
        "rl_selected_consensus_lcb_mean",
        "rl_online_conservative_advantage_mean",
        "rl_online_conservative_advantage_min",
        "rl_online_conservative_advantage_max",
        "rl_online_positive_advantage_fraction",
        "rl_target_conservative_advantage_mean",
        "rl_target_conservative_advantage_min",
        "rl_target_conservative_advantage_max",
        "rl_target_positive_advantage_fraction",
    )
    correction_scale = base.get("rl", {}).get("sac", {}).get(
        "correction_scale", (0.20,)
    )
    if np.isscalar(correction_scale):
        correction_scale = [float(correction_scale)]
    else:
        correction_scale = [float(value) for value in correction_scale]
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
        "policy_mode": str(base.get("rl", {}).get("sac", {}).get(
            "policy_mode", "direct"
        )),
        "correction_scale": correction_scale,
        "correction_gate_alpha": float(base.get("rl", {}).get(
            "sac", {}
        ).get("correction_gate_alpha", 1.0)),
        "correction_advantage_gate_mode": str(base.get("rl", {}).get(
            "gate", {}
        ).get("correction_advantage_gate_mode", "none")),
        "correction_advantage_critic_source": str(base.get("rl", {}).get(
            "gate", {}
        ).get("correction_advantage_critic_source", "online")),
        "correction_advantage_threshold": float(base.get("rl", {}).get(
            "gate", {}
        ).get("correction_advantage_threshold", 0.0)),
        "correction_advantage_uncertainty_multiplier": float(
            base.get("rl", {}).get("gate", {}).get(
                "correction_advantage_uncertainty_multiplier", 1.0
            )
        ),
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
