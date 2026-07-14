#!/usr/bin/env python3
"""Collect residual transitions from the actuated MuJoCo true plant."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mobile_robot_mppi.core.config import config_hash, git_sha, load_yaml
from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.learning.dataset_quality import (
    assert_episode_disjoint_splits,
    assert_residual_dataset_quality,
)
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction, LegacyUnicyclePrediction
from mobile_robot_mppi.runtime.factories import make_components
from src.learning.residual_dataset import ResidualDataset


def state_vector(truth, state_spec):
    values = {
        "x": truth.pose.x, "y": truth.pose.y, "theta": truth.pose.theta,
        "v": truth.twist.v, "omega": truth.twist.omega,
        "wheel_left": truth.wheel_speeds[0], "wheel_right": truth.wheel_speeds[1],
    }
    return np.asarray([values[name] for name in state_spec.names], dtype=np.float64)


def observed_derivative(state, following, dt, periodic_indices):
    difference = np.asarray(following) - np.asarray(state)
    for index in periodic_indices:
        difference[index] = np.arctan2(np.sin(difference[index]), np.cos(difference[index]))
    return difference / float(dt)


def collect(
    config,
    episodes,
    steps,
    source,
    episode_prefix="mujoco",
    disturbance_type="mujoco_physics",
):
    records = []
    base_seed = int(config["experiment"].get("seed", 0))
    for episode in range(int(episodes)):
        components = make_components(config, ROOT)
        plant = components["plant"]
        state_spec = components["state_spec"]
        action_spec = components["action_spec"]
        if state_spec.dimension == 3:
            nominal = LegacyUnicyclePrediction()
        elif state_spec.dimension == 5:
            nominal = DynamicUnicyclePrediction(
                config["plant"].get("nominal_velocity_time_constant", 0.18),
                config["plant"].get("nominal_yaw_time_constant", 0.12),
            )
        else:
            raise ValueError("collector currently supports state dimensions 3 and 5")
        seed = base_seed + episode
        rng = np.random.RandomState(seed)
        initial = np.asarray(config["experiment"].get("initial_state", (0, 0, 0)), dtype=np.float64)
        truth = plant.reset(seed, initial)
        observation = components["sensors"].reset(truth, seed)
        components["controller"].reset()
        dt = float(config["experiment"]["control_dt"])
        collection_cfg = dict(config.get("data_collection", {}))
        hold_steps = max(1, int(collection_cfg.get("random_hold_steps", 5)))
        smoothing = float(collection_cfg.get("random_action_smoothing", 0.65))
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("data_collection.random_action_smoothing must be in [0, 1)")
        random_action = np.zeros(action_spec.dimension, dtype=np.float64)
        episode_source = source
        if source == "mixed":
            episode_source = "random_exploration" if episode % 2 == 0 else "task_specific"
        for step_index in range(int(steps)):
            current = state_vector(truth, state_spec)
            if episode_source == "task_specific":
                perceived = components["perception"].process(observation)
                plan = components["controller"].plan(perceived.observation, components["reference"])
                decision = components["safety"].arbitrate(plan.proposed_control, perceived.guard)
                components["controller"].observe_safety_decision(decision)
                commanded = decision.executed_control
            else:
                if step_index % hold_steps == 0:
                    target_action = rng.uniform(action_spec.lower, action_spec.upper)
                random_action = smoothing * random_action + (1.0 - smoothing) * target_action
                proposed = ControlCommand(
                    action_spec.clip(random_action), truth.timestamp, "random_exploration"
                )
                # Exploration is still downstream of the protected LaserScan
                # safety chain; data collection is not allowed to bypass it.
                perceived = components["perception"].process(observation)
                decision = components["safety"].arbitrate(proposed, perceived.guard)
                commanded = decision.executed_control
            transition = plant.step(commanded, dt)
            following_truth = transition.ground_truth
            following = state_vector(following_truth, state_spec)
            observed = observed_derivative(current, following, dt, state_spec.periodic_indices)
            nominal_value = np.asarray(nominal.derivative(current, commanded.values), dtype=np.float64)
            model_version = following_truth.metadata.get("model_hash", "legacy_kinematic")
            average_applied = np.asarray(
                transition.metadata.get(
                    "average_applied_control", transition.executed_control.values
                ),
                dtype=np.float64,
            )
            records.append({
                "episode_id": "%s_%05d" % (episode_prefix, episode),
                "seed": seed,
                "step": step_index,
                "time": truth.timestamp,
                "dt": dt,
                "state_t": current,
                "control_t": commanded.values,
                "applied_control_t": average_applied,
                "state_t_plus_1": following,
                "nominal_derivative": nominal_value,
                "observed_derivative": observed,
                "residual_target": observed - nominal_value,
                "disturbance_type": str(disturbance_type),
                "disturbance_parameters": {
                    "slip_ratio": following_truth.slip_ratio,
                    "wheel_speeds": list(following_truth.wheel_speeds),
                    "actuator_effort": list(following_truth.actuator_effort),
                    "plant": config["plant"],
                    "domain": str(disturbance_type),
                },
                "scene": str(config.get("scene", {}).get("name", "unknown")),
                "data_source": episode_source,
                "model_version": str(model_version),
            })
            truth = following_truth
            observation = components["sensors"].observe(truth)
            if truth.collision:
                break
        plant.close()
    metadata = {
        "generator": "collect_mujoco_residual_data.py",
        "git_sha": git_sha(ROOT),
        "config_hash": config_hash(config),
        "config": config,
        "episodes_requested": int(episodes),
        "steps_requested": int(steps),
        "source": source,
        "episode_prefix": str(episode_prefix),
        "disturbance_type": str(disturbance_type),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "state_semantics": [str(name) for name in state_spec.names],
        "control_semantics": [str(name) for name in action_spec.names],
        "control_t_semantics": "safety-issued command presented to prediction dynamics",
        "applied_control_t_semantics": "physics-substep average command over transition",
    }
    return ResidualDataset.from_records(records, metadata=metadata)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/research/mujoco_diff_drive.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--source", choices=("random_exploration", "task_specific", "mixed"), default="mixed")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    dataset = collect(config, args.episodes, args.steps, args.source)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset.save(destination / "all.npz")
    quality = assert_residual_dataset_quality(dataset, angle_indices=(2,))
    splits = dataset.split(validation_fraction=0.15, test_fraction=0.15, seed=int(config["experiment"].get("seed", 0)))
    split_audit = assert_episode_disjoint_splits(splits)
    for name in ("train", "validation", "test", "unseen"):
        splits[name].save(destination / (name + ".npz"))
    manifest = {
        "dataset_summary": dataset.summary(),
        "quality_gate": quality,
        "split_audit": split_audit,
        "git_sha": git_sha(ROOT),
        "config_hash": config_hash(config),
    }
    with (destination / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
