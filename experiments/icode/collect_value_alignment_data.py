#!/usr/bin/env python3
"""Collect task-specific residual trajectories with frozen SAC critic context."""

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.learning.models import load_platform_checkpoint
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction
from mobile_robot_mppi.rl.environment import DirectControlEnv
from mobile_robot_mppi.rl.paper_policy import PaperDirectControlPolicy


def _resolve(path):
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state(truth):
    return np.asarray(
        (
            truth.pose.x,
            truth.pose.y,
            truth.pose.theta,
            truth.twist.v,
            truth.twist.omega,
        ),
        dtype=np.float64,
    )


def _wrapped_difference(following, current):
    difference = np.asarray(following, dtype=np.float64) - np.asarray(
        current, dtype=np.float64
    )
    difference[2] = np.arctan2(np.sin(difference[2]), np.cos(difference[2]))
    return difference


def _selected_environments(
    snapshot,
    source,
    roles,
    scene_prefixes=None,
):
    key = (
        "training_environments"
        if str(source) == "training"
        else "validation_environments"
    )
    allowed = set(str(role) for role in roles)
    prefixes = tuple(str(value) for value in (scene_prefixes or ()))
    result = []
    for value in snapshot.get(key, ()):
        config = copy.deepcopy(dict(value))
        role = str(config.get("experiment", {}).get("physics_domain_role", "seen"))
        scene = str(config.get("scene", {}).get("name", ""))
        scene_allowed = (
            not prefixes
            or any(scene.startswith(prefix) for prefix in prefixes)
        )
        if role in allowed and scene_allowed:
            result.append(config)
    if not result:
        raise ValueError(
            "no %s environments matched roles %s and scene prefixes %s"
            % (source, sorted(allowed), list(prefixes))
        )
    return result


def _environment_signature(config):
    return (
        str(config.get("scene", {}).get("name", "unknown")),
        str(
            config.get("experiment", {}).get(
                "physics_domain", "nominal"
            )
        ),
    )


def _records():
    return {
        "episode_id": [],
        "seed": [],
        "step": [],
        "time": [],
        "dt": [],
        "state_t": [],
        "control_t": [],
        "state_t_plus_1": [],
        "nominal_derivative": [],
        "observed_derivative": [],
        "residual_target": [],
        "raw_observation_t_plus_1": [],
        "target_position_t_plus_1": [],
        "path_context_t_plus_1": [],
        "scene": [],
        "physics_domain": [],
        "physics_domain_role": [],
        "safety_override": [],
    }


def _append(records, **values):
    for name in records:
        records[name].append(values[name])


def _archive(records):
    integer = {"seed", "step", "safety_override"}
    string = {
        "episode_id",
        "scene",
        "physics_domain",
        "physics_domain_role",
    }
    arrays = {}
    for name, values in records.items():
        if name in integer:
            arrays[name] = np.asarray(values, dtype=np.int64)
        elif name in string:
            arrays[name] = np.asarray(values, dtype=np.str_)
        else:
            arrays[name] = np.asarray(values, dtype=np.float64)
    return arrays


def collect_split(
    split_name,
    split_config,
    snapshot,
    actor_checkpoint,
    output_dir,
    action_noise_std,
    random_action_probability,
    global_seed,
):
    environments = _selected_environments(
        snapshot,
        split_config.get("source", "training"),
        split_config.get("roles", ("seen",)),
        split_config.get("scene_prefixes", ()),
    )
    episodes_per_environment = int(
        split_config.get("episodes_per_environment", 1)
    )
    maximum_steps = int(split_config.get("max_steps", 180))
    seed_base = int(split_config.get("seed_base", global_seed))
    if episodes_per_environment <= 0 or maximum_steps <= 0:
        raise ValueError("collection episode count and max_steps must be positive")
    records = _records()
    episode_summaries = []
    policy = None
    rng = np.random.RandomState(global_seed + seed_base)
    for environment_index, config in enumerate(environments):
        config = copy.deepcopy(config)
        config.setdefault("memory", {})["enable"] = False
        config.setdefault("experiment", {})["max_steps"] = maximum_steps
        environment = DirectControlEnv(config, ROOT, seed=seed_base)
        try:
            if policy is None:
                policy = PaperDirectControlPolicy.from_checkpoint(
                    actor_checkpoint,
                    environment.action_spec,
                    device="cpu",
                )
            elif tuple(policy.action_spec.names) != tuple(environment.action_spec.names):
                raise ValueError("all collection environments must share actions")
            nominal = DynamicUnicyclePrediction(
                config["plant"].get("nominal_velocity_time_constant", 0.18),
                config["plant"].get("nominal_yaw_time_constant", 0.12),
            )
            for episode_offset in range(episodes_per_environment):
                episode_seed = (
                    seed_base
                    + environment_index * 1009
                    + episode_offset * 37
                )
                raw_observation, reset_info = environment.reset(seed=episode_seed)
                episode_id = "%s_%03d_%02d" % (
                    split_name,
                    environment_index,
                    episode_offset,
                )
                success = False
                collision = False
                override_count = 0
                steps = 0
                for step in range(maximum_steps):
                    state_t = _state(environment.truth)
                    normalized = policy.normalizer.normalize(raw_observation)
                    action, _ = policy.agent.select_action(
                        normalized, deterministic=True
                    )
                    action = np.asarray(action, dtype=np.float64)
                    if rng.uniform() < float(random_action_probability):
                        action = rng.uniform(-1.0, 1.0, size=action.shape)
                    elif action_noise_std > 0.0:
                        action = action + rng.normal(
                            0.0, action_noise_std, size=action.shape
                        )
                    action = np.clip(action, -1.0, 1.0)
                    (
                        next_raw_observation,
                        _,
                        terminated,
                        truncated,
                        info,
                    ) = environment.step(action)
                    state_next = _state(environment.truth)
                    dt = float(config["experiment"]["control_dt"])
                    executed = np.asarray(
                        info["executed_control"], dtype=np.float64
                    )
                    observed = _wrapped_difference(state_next, state_t) / dt
                    nominal_value = nominal.derivative(state_t, executed)
                    residual = observed - nominal_value
                    target = environment.components["reference"].target_at(
                        environment.truth.timestamp,
                        environment.truth.pose.as_array(),
                    )
                    path_context = environment.encoder.path_context(
                        environment.components["reference"],
                        environment.truth.pose.as_array(),
                        target=target,
                    )
                    if path_context is None:
                        path_context = np.zeros(6, dtype=np.float32)
                    _append(
                        records,
                        episode_id=episode_id,
                        seed=episode_seed,
                        step=step,
                        time=float(environment.truth.timestamp) - dt,
                        dt=dt,
                        state_t=state_t,
                        control_t=executed,
                        state_t_plus_1=state_next,
                        nominal_derivative=nominal_value,
                        observed_derivative=observed,
                        residual_target=residual,
                        raw_observation_t_plus_1=next_raw_observation,
                        target_position_t_plus_1=(
                            target.pose.x,
                            target.pose.y,
                        ),
                        path_context_t_plus_1=path_context,
                        scene=str(info["scene"]),
                        physics_domain=str(
                            config.get("experiment", {}).get(
                                "physics_domain", "nominal"
                            )
                        ),
                        physics_domain_role=str(
                            config.get("experiment", {}).get(
                                "physics_domain_role", "seen"
                            )
                        ),
                        safety_override=int(bool(info["safety_override"])),
                    )
                    raw_observation = next_raw_observation
                    override_count += int(bool(info["safety_override"]))
                    success = bool(info["success"])
                    collision = bool(info["collision"])
                    steps = step + 1
                    if terminated or truncated:
                        break
                episode_summaries.append({
                    "split": split_name,
                    "episode_id": episode_id,
                    "seed": episode_seed,
                    "scene": str(reset_info["scene"]),
                    "physics_domain": str(
                        config.get("experiment", {}).get(
                            "physics_domain", "nominal"
                        )
                    ),
                    "physics_domain_role": str(
                        config.get("experiment", {}).get(
                            "physics_domain_role", "seen"
                        )
                    ),
                    "steps": steps,
                    "success": int(success),
                    "collision": int(collision),
                    "safety_overrides": override_count,
                })
        finally:
            environment.close()
    arrays = _archive(records)
    if arrays["state_t"].shape[0] == 0:
        raise RuntimeError("value-alignment collection produced no transitions")
    target = output_dir / (split_name + ".npz")
    np.savez_compressed(target, **arrays)
    return target, episode_summaries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/icode/gate2_value_alignment_data_l177.yaml"),
    )
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("value-alignment collection config must be a mapping")
    actor_checkpoint = _resolve(config["actor_checkpoint"])
    actor_run_dir = _resolve(config["actor_run_dir"])
    snapshot_path = actor_run_dir / "config_snapshot.json"
    with snapshot_path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 20260718))
    split_environments = {}
    environment_owners = {}
    require_disjoint = bool(
        config.get("require_disjoint_environments", False)
    )
    for split_name, split_config in dict(config["splits"]).items():
        selected = _selected_environments(
            snapshot,
            split_config.get("source", "training"),
            split_config.get("roles", ("seen",)),
            split_config.get("scene_prefixes", ()),
        )
        signatures = [
            _environment_signature(environment) for environment in selected
        ]
        if len(signatures) != len(set(signatures)):
            raise ValueError(
                "%s contains duplicate scene/domain environments"
                % split_name
            )
        if require_disjoint:
            overlap = sorted(
                signature
                for signature in signatures
                if signature in environment_owners
            )
            if overlap:
                raise ValueError(
                    "environment leakage between %s and %s: %s"
                    % (
                        environment_owners[overlap[0]],
                        split_name,
                        overlap,
                    )
                )
        for signature in signatures:
            environment_owners[signature] = split_name
        split_environments[split_name] = [
            {
                "scene": signature[0],
                "physics_domain": signature[1],
            }
            for signature in signatures
        ]
    summaries = []
    artifacts = {}
    for split_name, split_config in dict(config["splits"]).items():
        path, rows = collect_split(
            split_name,
            split_config,
            snapshot,
            actor_checkpoint,
            output_dir,
            float(config.get("action_noise_std", 0.03)),
            float(config.get("random_action_probability", 0.05)),
            seed,
        )
        artifacts[split_name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "transitions": int(
                np.load(path, allow_pickle=False)["state_t"].shape[0]
            ),
            "episodes": len(rows),
        }
        summaries.extend(rows)
    with (output_dir / "episodes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    metadata = {
        "schema_version": 1,
        "created_from_config": str(config_path),
        "git_sha": git_sha(ROOT),
        "actor_checkpoint": str(actor_checkpoint),
        "actor_checkpoint_sha256": _sha256(actor_checkpoint),
        "actor_config_snapshot": str(snapshot_path),
        "actor_config_snapshot_sha256": _sha256(snapshot_path),
        "seed": seed,
        "action_noise_std": float(config.get("action_noise_std", 0.03)),
        "random_action_probability": float(
            config.get("random_action_probability", 0.05)
        ),
        "require_disjoint_environments": require_disjoint,
        "split_environments": split_environments,
        "artifacts": artifacts,
        "config": config,
    }
    with (output_dir / "dataset_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
