#!/usr/bin/env python3
"""Collect paired one-step correction interventions from deterministic replay."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.rl.run_correction_advantage_diagnostic import (
    _evaluation_config,
    _load_agent,
    _resolved_path,
    _seeds,
    _write_csv,
)
from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.rl.environment import MppiPriorEnv
from mobile_robot_mppi.rl.risk_dataset import (
    CounterfactualLabelConfig,
    burst_accounting_valid,
    classify_counterfactual,
    feature_vector,
    validate_sealed_episode_split,
)
from mobile_robot_mppi.rl.trajectory_features import (
    paired_trajectory_feature_vector,
)


def _branch_steps(values):
    if isinstance(values, str):
        result = [int(value) for value in values.split(",") if value.strip()]
    else:
        result = [int(value) for value in values]
    if (
        not result
        or any(value < 0 for value in result)
        or len(set(result)) != len(result)
        or result != sorted(result)
    ):
        raise ValueError(
            "branch steps must be unique, non-negative and increasing"
        )
    return result


def _episode_seeds(values):
    """Parse CLI comma strings or YAML integer sequences consistently."""

    if isinstance(values, str):
        return _seeds(values)
    try:
        return _seeds(",".join(str(int(value)) for value in values))
    except (TypeError, ValueError):
        raise ValueError("episode seeds must be an integer sequence")


def _base_action(agent, normalizer, observation):
    normalized = normalizer.normalize(observation)
    return agent.frozen_base_action(normalized)


def _reference_rollout(resolved, agent, normalizer, seed):
    environment = MppiPriorEnv(resolved, ROOT, seed=seed)
    observation, reset_info = environment.reset(seed=seed)
    observations = [observation.copy()]
    actions = []
    rewards = []
    last_info = {
        "goal_distance": float(reset_info["goal_distance"]),
        "success": False,
        "collision": False,
    }
    terminated = truncated = False
    try:
        while not (terminated or truncated):
            action = _base_action(agent, normalizer, observation)
            observation, reward, terminated, truncated, last_info = (
                environment.step(action)
            )
            actions.append(action.copy())
            rewards.append(float(reward))
            observations.append(observation.copy())
    finally:
        environment.close()
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "summary": {
            "seed": int(seed),
            "steps": len(actions),
            "return": float(sum(rewards)),
            "success": bool(last_info.get("success", False)),
            "collision": bool(last_info.get("collision", False)),
            "goal_distance": float(last_info["goal_distance"]),
        },
    }


def _run_branch(
    resolved,
    agent,
    normalizer,
    seed,
    reference,
    branch_step,
    branch_policy,
    intervention_steps,
    beta,
    horizon,
    replay_atol,
    preview_state_features=None,
    preview_actions=None,
):
    if branch_policy not in ("baseline", "candidate"):
        raise ValueError("unknown counterfactual branch policy")
    if intervention_steps <= 0 or horizon < intervention_steps:
        raise ValueError(
            "intervention steps must be positive and no larger than horizon"
        )
    environment = MppiPriorEnv(resolved, ROOT, seed=seed)
    observation, reset_info = environment.reset(seed=seed)
    max_replay_error = float(np.max(np.abs(
        observation - reference["observations"][0]
    )))
    if max_replay_error > replay_atol:
        environment.close()
        raise RuntimeError("counterfactual reset replay drifted")
    terminated = truncated = False
    last_info = {
        "goal_distance": float(reset_info["goal_distance"]),
        "success": False,
        "collision": False,
        "minimum_clearance": float("inf"),
        "safety_override": False,
    }
    preview_features = preview_schema = None
    try:
        for step in range(branch_step):
            observation, _, terminated, truncated, last_info = environment.step(
                reference["actions"][step]
            )
            error = float(np.max(np.abs(
                observation - reference["observations"][step + 1]
            )))
            max_replay_error = max(max_replay_error, error)
            if error > replay_atol:
                raise RuntimeError(
                    "counterfactual replay drifted before branch step"
                )
            if terminated or truncated:
                raise RuntimeError(
                    "reference replay terminated before branch step"
                )

        if (preview_state_features is None) != (preview_actions is None):
            raise ValueError(
                "trajectory preview features and actions must be supplied together"
            )
        if preview_actions is not None:
            if len(preview_actions) != 2:
                raise ValueError(
                    "trajectory preview requires base and candidate actions"
                )
            controller = environment.components["controller"]
            previous_action = controller.previous_action.copy()
            base_plan = environment.preview_policy_action(preview_actions[0])
            candidate_plan = environment.preview_policy_action(
                preview_actions[1]
            )
            final_target = environment.final_target
            preview_features, preview_schema = (
                paired_trajectory_feature_vector(
                    preview_state_features,
                    base_plan,
                    candidate_plan,
                    (final_target.x, final_target.y),
                    environment.perceived.observation.local_obstacles,
                    controller.state_spec.position_indices,
                    controller.action_spec.names,
                    previous_action,
                    controller.config.robot_radius,
                    controller.config.obstacle_influence,
                )
            )

        total_reward = 0.0
        minimum_clearance = float("inf")
        safety_overrides = 0
        branch_steps_run = 0
        intervention_steps_run = 0
        intervention_accepted_steps = 0
        intervention_rejected_steps = 0
        first_gate_accepted = False
        intervention_corrections = []
        intervention_lcb_scores = []
        for horizon_step in range(horizon):
            base_action = _base_action(agent, normalizer, observation)
            action = base_action
            if branch_policy == "candidate" and horizon_step < intervention_steps:
                normalized = normalizer.normalize(observation)
                candidate_action, _ = agent.select_action(
                    normalized, deterministic=True
                )
                action, diagnostics = agent.filter_correction_by_advantage(
                    normalized,
                    candidate_action,
                    gate_mode="lcb",
                    critic_source="target",
                    uncertainty_multiplier=beta,
                )
                accepted = bool(
                    diagnostics["correction_advantage_gate_alpha"] >= 0.5
                )
                if horizon_step == 0:
                    first_gate_accepted = accepted
                intervention_steps_run += 1
                intervention_accepted_steps += int(accepted)
                intervention_rejected_steps += int(not accepted)
                intervention_corrections.append(
                    np.abs(
                        np.asarray(action, dtype=np.float64)
                        - np.asarray(base_action, dtype=np.float64)
                    )
                )
                intervention_lcb_scores.append(
                    float(diagnostics["target_consensus_lcb"])
                )
            observation, reward, terminated, truncated, last_info = (
                environment.step(action)
            )
            total_reward += float(reward)
            branch_steps_run += 1
            minimum_clearance = min(
                minimum_clearance,
                float(last_info["minimum_clearance"]),
            )
            safety_overrides += int(bool(last_info["safety_override"]))
            if terminated or truncated:
                break
    finally:
        environment.close()
    if not np.isfinite(minimum_clearance):
        raise FloatingPointError(
            "counterfactual branch requires finite obstacle clearance"
        )
    if intervention_corrections:
        correction_values = np.concatenate(intervention_corrections)
        correction_mean = float(np.mean(correction_values))
        correction_max = float(np.max(correction_values))
        lcb_mean = float(np.mean(intervention_lcb_scores))
        lcb_min = float(np.min(intervention_lcb_scores))
        lcb_max = float(np.max(intervention_lcb_scores))
    else:
        correction_mean = correction_max = 0.0
        lcb_mean = lcb_min = lcb_max = 0.0
    result = {
        "return": float(total_reward),
        "goal_distance": float(last_info["goal_distance"]),
        "success": bool(last_info.get("success", False)),
        "collision": bool(last_info.get("collision", False)),
        "minimum_clearance": float(minimum_clearance),
        "safety_overrides": int(safety_overrides),
        "steps": int(branch_steps_run),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "max_replay_observation_error": float(max_replay_error),
        "intervention_steps_requested": int(
            intervention_steps if branch_policy == "candidate" else 0
        ),
        "intervention_steps_run": int(intervention_steps_run),
        "intervention_accepted_steps": int(intervention_accepted_steps),
        "intervention_rejected_steps": int(intervention_rejected_steps),
        "intervention_first_gate_accepted": bool(first_gate_accepted),
        "intervention_accept_fraction": float(
            intervention_accepted_steps / intervention_steps_run
            if intervention_steps_run else 0.0
        ),
        "intervention_correction_abs_mean": correction_mean,
        "intervention_correction_abs_max": correction_max,
        "intervention_lcb_mean": lcb_mean,
        "intervention_lcb_min": lcb_min,
        "intervention_lcb_max": lcb_max,
    }
    if preview_features is not None:
        result["_preview_features"] = preview_features
        result["_preview_schema"] = preview_schema
    return result


def _prefixed(row, prefix, values):
    for name, value in values.items():
        row[prefix + name] = value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--train-seeds", required=True)
    parser.add_argument("--validation-seeds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--branch-steps")
    parser.add_argument("--branch-horizon", type=int)
    parser.add_argument("--intervention-steps", type=int)
    args = parser.parse_args(argv)

    train_seeds = _seeds(args.train_seeds)
    validation_seeds = _seeds(args.validation_seeds)
    resolved = _evaluation_config(args.config, args.scene_config)
    risk_config = dict(resolved.get("rl", {}).get("risk_dataset", {}))
    trajectory_feature_config = dict(
        risk_config.get("trajectory_features", {})
    )
    trajectory_features_enabled = bool(
        trajectory_feature_config.get("enabled", False)
    )
    sealed_test_seeds = _episode_seeds(
        risk_config.get(
            "sealed_test_episode_seeds", range(20281311, 20281316)
        )
    )
    validate_sealed_episode_split(
        train_seeds, validation_seeds, sealed_test_seeds
    )
    split_by_seed = {
        **{seed: "train" for seed in train_seeds},
        **{seed: "validation" for seed in validation_seeds},
    }
    steps = _branch_steps(
        args.branch_steps
        if args.branch_steps is not None
        else risk_config.get("branch_steps", (10, 30, 50, 70, 90, 110))
    )
    horizon = int(
        args.branch_horizon
        if args.branch_horizon is not None
        else risk_config.get("branch_horizon", 20)
    )
    if horizon <= 0:
        raise ValueError("branch horizon must be positive")
    intervention_steps = int(
        args.intervention_steps
        if args.intervention_steps is not None
        else risk_config.get("intervention_steps", 1)
    )
    if intervention_steps <= 0 or intervention_steps > horizon:
        raise ValueError(
            "intervention steps must be positive and no larger than horizon"
        )
    intervention_type = str(
        risk_config.get(
            "intervention_type",
            "single_step_correction"
            if intervention_steps == 1
            else "gated_correction_burst",
        )
    )
    beta = float(risk_config.get("candidate_beta", 2.0))
    replay_atol = float(risk_config.get("replay_atol", 1e-10))
    if not np.isfinite(beta) or beta < 1.0:
        raise ValueError("candidate beta must be finite and at least one")
    if not np.isfinite(replay_atol) or replay_atol < 0.0:
        raise ValueError("replay tolerance must be finite and non-negative")
    label_config = CounterfactualLabelConfig.from_mapping(risk_config)
    label_config.validate()

    checkpoint = _resolved_path(args.checkpoint)
    agent, normalizer, payload = _load_agent(checkpoint, args.device)
    sample_rows = []
    feature_rows = []
    label_rows = []
    reference_rows = []
    feature_schema = None
    for seed in train_seeds + validation_seeds:
        reference = _reference_rollout(
            resolved, agent, normalizer, seed
        )
        considered = accepted = skipped_terminal = 0
        episode_max_replay_error = 0.0
        for branch_step in steps:
            if branch_step >= len(reference["actions"]):
                skipped_terminal += 1
                continue
            considered += 1
            observation = reference["observations"][branch_step]
            normalized = normalizer.normalize(observation)
            base_action = agent.frozen_base_action(normalized)
            candidate_action, policy = agent.select_action(
                normalized, deterministic=True
            )
            gated_action, diagnostics = agent.filter_correction_by_advantage(
                normalized,
                candidate_action,
                gate_mode="lcb",
                critic_source="target",
                uncertainty_multiplier=beta,
            )
            if diagnostics["correction_advantage_gate_alpha"] < 0.5:
                continue
            if not np.array_equal(gated_action, candidate_action):
                raise RuntimeError("accepted LCB action differs from candidate")
            accepted += 1
            state_features, state_schema = feature_vector(
                normalized,
                base_action,
                candidate_action,
                diagnostics,
                branch_step,
            )
            baseline = _run_branch(
                resolved,
                agent,
                normalizer,
                seed,
                reference,
                branch_step,
                "baseline",
                intervention_steps,
                beta,
                horizon,
                replay_atol,
                preview_state_features=(
                    state_features if trajectory_features_enabled else None
                ),
                preview_actions=(
                    (base_action, candidate_action)
                    if trajectory_features_enabled
                    else None
                ),
            )
            if trajectory_features_enabled:
                features = baseline.pop("_preview_features")
                schema = baseline.pop("_preview_schema")
                schema["state_feature_schema"] = state_schema
                schema["diagnostic_names"] = state_schema[
                    "diagnostic_names"
                ]
                expected_dimensions = {
                    "state_feature_dim": trajectory_feature_config.get(
                        "expected_state_feature_dim"
                    ),
                    "trajectory_metric_dim": trajectory_feature_config.get(
                        "expected_metric_dim"
                    ),
                    "feature_dim": trajectory_feature_config.get(
                        "expected_total_dim"
                    ),
                }
                for name, expected_value in expected_dimensions.items():
                    if (
                        expected_value is not None
                        and int(schema[name]) != int(expected_value)
                    ):
                        raise RuntimeError(
                            "trajectory feature %s differs from frozen contract"
                            % name
                        )
            else:
                features, schema = state_features, state_schema
            candidate = _run_branch(
                resolved,
                agent,
                normalizer,
                seed,
                reference,
                branch_step,
                "candidate",
                intervention_steps,
                beta,
                horizon,
                replay_atol,
            )
            prefixed_candidate = {
                "candidate_" + name: value
                for name, value in candidate.items()
            }
            if not burst_accounting_valid(prefixed_candidate):
                raise RuntimeError(
                    "candidate burst violated its execution accounting contract"
                )
            episode_max_replay_error = max(
                episode_max_replay_error,
                baseline["max_replay_observation_error"],
                candidate["max_replay_observation_error"],
            )
            targets = classify_counterfactual(
                baseline, candidate, label_config
            )
            if feature_schema is None:
                feature_schema = schema
            elif feature_schema != schema:
                raise RuntimeError("counterfactual feature schema drifted")
            row = {
                "training_seed": int(args.training_seed),
                "episode_seed": int(seed),
                "split": split_by_seed[seed],
                "branch_step": int(branch_step),
                "branch_horizon": int(horizon),
                "intervention_type": intervention_type,
                "intervention_steps": int(intervention_steps),
                "candidate_beta": beta,
                **targets,
            }
            _prefixed(row, "base_", baseline)
            _prefixed(row, "candidate_", candidate)
            for name in feature_schema["diagnostic_names"]:
                row[name] = float(diagnostics[name])
            for index, value in enumerate(features):
                row["feature_%03d" % index] = float(value)
            sample_rows.append(row)
            feature_rows.append(features)
            label_rows.append(int(targets["label_index"]))
        reference_row = dict(reference["summary"])
        reference_row.update({
            "training_seed": int(args.training_seed),
            "split": split_by_seed[seed],
            "scheduled_branch_points": len(steps),
            "considered_branch_points": considered,
            "accepted_branch_points": accepted,
            "skipped_terminal_branch_points": skipped_terminal,
            "max_replay_observation_error": episode_max_replay_error,
        })
        reference_rows.append(reference_row)

    if not sample_rows or feature_schema is None:
        raise RuntimeError("counterfactual collector produced no accepted branches")
    features = np.stack(feature_rows).astype(np.float32)
    labels = np.asarray(label_rows, dtype=np.int64)
    training_seed_array = np.asarray([
        row["training_seed"] for row in sample_rows
    ], dtype=np.int64)
    episode_seed_array = np.asarray([
        row["episode_seed"] for row in sample_rows
    ], dtype=np.int64)
    branch_step_array = np.asarray([
        row["branch_step"] for row in sample_rows
    ], dtype=np.int64)
    return_delta = np.asarray([
        row["return_delta"] for row in sample_rows
    ], dtype=np.float32)
    distance_improvement = np.asarray([
        row["goal_distance_improvement"] for row in sample_rows
    ], dtype=np.float32)
    split_array = np.asarray([
        0 if row["split"] == "train" else 1 for row in sample_rows
    ], dtype=np.int8)
    if not (
        np.isfinite(features).all()
        and np.isfinite(return_delta).all()
        and np.isfinite(distance_improvement).all()
    ):
        raise FloatingPointError("counterfactual dataset contains NaN or Inf")

    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "samples.npz",
        features=features,
        labels=labels,
        training_seed=training_seed_array,
        episode_seed=episode_seed_array,
        branch_step=branch_step_array,
        split=split_array,
        return_delta=return_delta,
        goal_distance_improvement=distance_improvement,
    )
    _write_csv(output / "samples.csv", sample_rows)
    _write_csv(output / "reference_episodes.csv", reference_rows)
    label_counts = {
        split: dict(Counter(
            row["label"] for row in sample_rows if row["split"] == split
        ))
        for split in ("train", "validation")
    }
    metadata = {
        "training_seed": int(args.training_seed),
        "train_episode_seeds": train_seeds,
        "validation_episode_seeds": validation_seeds,
        "sealed_test_episode_seeds": sealed_test_seeds,
        "branch_steps": steps,
        "branch_horizon": horizon,
        "intervention_type": intervention_type,
        "intervention_steps": intervention_steps,
        "candidate_beta": beta,
        "replay_atol": replay_atol,
        "label_config": label_config.to_dict(),
        "feature_schema": feature_schema,
        "trajectory_features": {
            "enabled": trajectory_features_enabled,
            "common_random_numbers": trajectory_features_enabled,
        },
        "samples": len(sample_rows),
        "label_counts": label_counts,
        "checkpoint": str(checkpoint),
        "checkpoint_git_sha": payload.get("git_sha"),
        "run_git_sha": git_sha(ROOT),
        "frozen_base_actor_sha256": agent.frozen_base_actor_sha256(),
        "scene": str(resolved.get("scene", {}).get("name", "unknown")),
        "interpretation_guard": (
            "branch rows are nested within episode seeds and training "
            "checkpoints; they are not independent training replicates"
        ),
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with (output / "config_snapshot.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
