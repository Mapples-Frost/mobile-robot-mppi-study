#!/usr/bin/env python3
"""Collect residual transitions from the actuated MuJoCo true plant."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mobile_robot_mppi.core.config import config_hash, git_sha, load_yaml
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


def collect(config, episodes, steps, source):
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
        episode_source = source
        if source == "mixed":
            episode_source = "random_exploration" if episode % 2 == 0 else "task_specific"
        for step_index in range(int(steps)):
            current = state_vector(truth, state_spec)
            if episode_source == "task_specific":
                perceived = components["perception"].process(observation)
                plan = components["controller"].plan(perceived.observation, components["reference"])
                decision = components["safety"].arbitrate(plan.proposed_control, perceived.guard)
                commanded = decision.executed_control
            else:
                from mobile_robot_mppi.core.types import ControlCommand
                action = rng.uniform(action_spec.lower, action_spec.upper)
                commanded = ControlCommand(action, truth.timestamp, "random_exploration")
            transition = plant.step(commanded, dt)
            following_truth = transition.ground_truth
            following = state_vector(following_truth, state_spec)
            observed = observed_derivative(current, following, dt, state_spec.periodic_indices)
            nominal_value = np.asarray(nominal.derivative(current, commanded.values), dtype=np.float64)
            model_version = following_truth.metadata.get("model_hash", "legacy_kinematic")
            records.append({
                "episode_id": "mujoco_%05d" % episode,
                "seed": seed,
                "step": step_index,
                "time": truth.timestamp,
                "dt": dt,
                "state_t": current,
                "control_t": commanded.values,
                "applied_control_t": transition.executed_control.values,
                "state_t_plus_1": following,
                "nominal_derivative": nominal_value,
                "observed_derivative": observed,
                "residual_target": observed - nominal_value,
                "disturbance_type": "mujoco_physics",
                "disturbance_parameters": {
                    "slip_ratio": following_truth.slip_ratio,
                    "wheel_speeds": list(following_truth.wheel_speeds),
                    "actuator_effort": list(following_truth.actuator_effort),
                    "plant": config["plant"],
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
    splits = dataset.split(validation_fraction=0.15, test_fraction=0.15, seed=int(config["experiment"].get("seed", 0)))
    for name in ("train", "validation", "test", "unseen"):
        splits[name].save(destination / (name + ".npz"))
    print(json.dumps(dataset.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
