#!/usr/bin/env python3
"""Run the frozen L276 recovery-sequence imitation initialization Gate."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rl.generate_l267_recovery_dataset import _inside_field
from experiments.rl.run_l263_counterfactual_actor_diagnosis import (
    _load_agent,
    _prepare_reset,
)
from experiments.rl.run_l269_horizon_diagnostic import _resolved_environment
from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig


DEFAULT_CONFIG = ROOT / "configs/rl/l276_recovery_sequence_imitation_initialization.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l276_recovery_sequence_imitation_initialization"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(str(name).encode("utf-8"))
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
        else:
            array = np.asarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _normalizer_sha256(normalizer) -> str:
    return _state_sha256(normalizer.state_dict())


def _nonactor_hashes(agent, normalizer):
    return {
        "critic1": _state_sha256(agent.critic1.state_dict()),
        "critic2": _state_sha256(agent.critic2.state_dict()),
        "target_critic1": _state_sha256(agent.target_critic1.state_dict()),
        "target_critic2": _state_sha256(agent.target_critic2.state_dict()),
        "log_alpha": hashlib.sha256(
            agent.log_alpha.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "normalizer": _normalizer_sha256(normalizer),
    }


def _write_csv(path: Path, rows):
    rows = list(rows)
    fields = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_config(path: Path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["l276"]
    checks = {
        ROOT / config["source_checkpoint"]: config["source_checkpoint_sha256"],
        ROOT / config["recovery_dataset"] / "manifest.json": (
            config["recovery_dataset_manifest_sha256"]
        ),
        ROOT / config["l275_summary"]: config["l275_summary_sha256"],
    }
    for artifact, expected in checks.items():
        if _sha256(artifact) != expected:
            raise ValueError("L276 input SHA256 mismatch: %s" % artifact)
    references = "\n".join(str(item) for item in checks).lower()
    entered = [
        token for token in config["forbidden_tokens"]
        if token.lower() in references
    ]
    if entered:
        raise ValueError("forbidden artifact entered L276: %s" % entered)
    l275 = json.loads((ROOT / config["l275_summary"]).read_text(encoding="utf-8"))
    if l275["decision"] != "continuation_policy_mismatch" or not l275["gate_pass"]:
        raise ValueError("L276 requires the frozen positive L275 decision")
    if not config["initialization_gate_only"] or config["sac_training_authorized"]:
        raise ValueError("L276 must remain an initialization-only Gate")
    return config, checks


def _load_chains(dataset: Path, manifest):
    chains = []
    for metadata in manifest["chains"]:
        with np.load(dataset / metadata["student_npz"], allow_pickle=False) as shard:
            chain = {
                **metadata,
                "observations": shard["observations"].copy(),
                "actions": shard["actions"].copy(),
                "rewards": shard["rewards"].reshape(-1).copy(),
            }
        if chain["observations"].shape[1:] != (69,) or chain["actions"].shape[1:] != (2,):
            raise ValueError("L276 recovery shard shape mismatch")
        chains.append(chain)
    return chains


def _sample_recovery(chains_by_scene, rng, count):
    scenes = sorted(chains_by_scene)
    observations = []
    actions = []
    for _ in range(int(count)):
        scene = scenes[int(rng.randint(len(scenes)))]
        chain_list = chains_by_scene[scene]
        chain = chain_list[int(rng.randint(len(chain_list)))]
        index = int(rng.randint(len(chain["actions"])))
        observations.append(chain["observations"][index])
        actions.append(chain["actions"][index])
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def _predict_actions(agent, normalizer, observations, batch_size=1024):
    raw = np.asarray(observations, dtype=np.float32)
    predictions = []
    agent.eval()
    with torch.no_grad():
        for start in range(0, len(raw), int(batch_size)):
            normalized = normalizer.normalize(raw[start:start + int(batch_size)])
            tensor = torch.as_tensor(
                normalized, dtype=torch.float32, device=agent.device
            )
            predictions.append(agent.actor.mean_action(tensor).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def _teacher_metrics(agent, normalizer, chains):
    observations = np.concatenate([row["observations"] for row in chains], axis=0)
    targets = np.concatenate([row["actions"] for row in chains], axis=0)
    predictions = _predict_actions(agent, normalizer, observations)
    error = predictions.astype(np.float64) - targets.astype(np.float64)
    return {
        "samples": int(len(targets)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "v_rmse": float(np.sqrt(np.mean(np.square(error[:, 0])))),
        "omega_rmse": float(np.sqrt(np.mean(np.square(error[:, 1])))),
    }


def _rollout_actor(
    environment, chain, agent, normalizer, reset_tolerance,
):
    expected = np.asarray(chain["observations"][0], dtype=np.float32)
    observation, _, path_state = _prepare_reset(
        environment,
        np.asarray(chain["initial_state"], dtype=np.float64),
        int(chain["seed"]),
    )
    reset_error = float(np.max(np.abs(observation - expected)))
    if reset_error > float(reset_tolerance):
        raise RuntimeError("L276 fixed-chain reset observation drifted")
    initial_cte = float(path_state["cross_track_error"])
    initial_progress = float(path_state["progress"])
    gamma = float(environment.gamma)
    cumulative = 0.0
    corridor_reentry = initial_cte <= 0.75
    collision = False
    boundary = False
    terminated = truncated = False
    last_info = None
    steps = 0
    agent.eval()
    for step in range(len(chain["actions"])):
        if terminated or truncated:
            break
        action, _ = agent.select_action(
            normalizer.normalize(observation), deterministic=True
        )
        observation, reward, terminated, truncated, last_info = environment.step(action)
        cumulative += gamma ** step * float(reward)
        steps += 1
        collision = collision or bool(last_info["collision"])
        boundary = boundary or not _inside_field(
            environment.config, environment.truth.pose.as_array()
        )
        corridor_reentry = corridor_reentry or (
            float(last_info["cross_track_error"]) <= 0.75
        )
    if last_info is None:
        raise RuntimeError("L276 actor rollout produced no transition")
    numeric = np.asarray((
        cumulative,
        reset_error,
        last_info["cross_track_error"],
        last_info["path_progress"],
    ))
    if not np.isfinite(numeric).all():
        raise FloatingPointError("L276 rollout produced non-finite values")
    return {
        "horizon": int(len(chain["actions"])),
        "steps_executed": int(steps),
        "discounted_return": float(cumulative),
        "cross_track_delta": float(last_info["cross_track_error"] - initial_cte),
        "path_progress_delta": float(last_info["path_progress"] - initial_progress),
        "corridor_reentry": bool(corridor_reentry),
        "collision": bool(collision),
        "boundary_violation": bool(boundary),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "max_reset_observation_error": reset_error,
    }


def _save_checkpoint(path, source_payload, agent, seed, updates, config, hashes):
    payload = copy.deepcopy(source_payload)
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    payload["git_sha"] = git_sha(ROOT)
    payload["agent"] = agent.state_dict()
    payload["training_state"] = {
        "phase": "l276_recovery_sequence_imitation_initialization",
        "protocol": "L276",
        "seed": int(seed),
        "actor_only_updates": int(updates),
        "source_checkpoint": config["source_checkpoint"],
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "recovery_dataset_manifest_sha256": config[
            "recovery_dataset_manifest_sha256"
        ],
        "l275_summary_sha256": config["l275_summary_sha256"],
        "nonactor_hashes": hashes,
        "sac_training_authorized": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    return _sha256(path)


def _gate(seed_metrics, rollout_rows, nonactor_unchanged, all_finite, gate):
    relative = [row["relative_test_rmse_improvement"] for row in seed_metrics]
    validation_gains = [row["median_validation_return_gain"] for row in seed_metrics]
    test_gains = [row["median_test_return_gain"] for row in seed_metrics]
    test_reentry_gains = [row["test_reentry_fraction_gain"] for row in seed_metrics]
    drifts = [row["source_replay_mean_absolute_action_drift"] for row in seed_metrics]
    by_scene = defaultdict(list)
    for row in rollout_rows:
        if row["split"] == "test" and row["policy"] != "source_actor":
            by_scene[row["scene"]].append(float(row["return_gain_vs_source"]))
    scene_gains = {scene: float(np.mean(values)) for scene, values in sorted(by_scene.items())}
    scenes_improved = sum(value > 0.0 for value in scene_gains.values())
    source_failures = sum(
        int(row["collision"] or row["boundary_violation"])
        for row in rollout_rows
        if row["policy"] == "source_actor"
    )
    initialized_failures = sum(
        int(row["collision"] or row["boundary_violation"])
        for row in rollout_rows
        if row["policy"] != "source_actor"
    )
    # The source baseline is repeated once per seed in the paired table.
    failure_increase = initialized_failures - source_failures
    metrics = {
        "seed_relative_test_rmse_improvements": relative,
        "median_relative_test_rmse_improvement": float(np.median(relative)),
        "median_validation_return_gain": float(np.median(validation_gains)),
        "median_test_return_gain": float(np.median(test_gains)),
        "median_test_reentry_fraction_gain": float(np.median(test_reentry_gains)),
        "test_scene_return_gains": scene_gains,
        "test_scenes_with_return_improvement": int(scenes_improved),
        "maximum_source_replay_mean_absolute_action_drift": float(max(drifts)),
        "source_collision_boundary_failures": int(source_failures),
        "initialized_collision_boundary_failures": int(initialized_failures),
        "collision_boundary_failure_increase": int(failure_increase),
    }
    checks = {
        "all_three_test_rmse_directions_positive": all(value > 0.0 for value in relative),
        "test_rmse_improvement": (
            metrics["median_relative_test_rmse_improvement"]
            >= float(gate["minimum_median_relative_test_rmse_improvement"])
        ),
        "validation_return_gain": (
            metrics["median_validation_return_gain"]
            > float(gate["minimum_median_validation_return_gain"])
        ),
        "test_return_gain": (
            metrics["median_test_return_gain"]
            > float(gate["minimum_median_test_return_gain"])
        ),
        "test_reentry_gain": (
            metrics["median_test_reentry_fraction_gain"]
            >= float(gate["minimum_test_reentry_fraction_gain"])
        ),
        "test_scene_coverage": (
            scenes_improved >= int(gate["minimum_test_scenes_with_return_improvement"])
        ),
        "source_replay_drift": (
            metrics["maximum_source_replay_mean_absolute_action_drift"]
            <= float(gate["maximum_source_replay_mean_absolute_action_drift"])
        ),
        "collision_boundary_nonincrease": (
            failure_increase
            <= float(gate["maximum_collision_boundary_failure_increase"])
        ),
        "frozen_nonactor_hashes": bool(nonactor_unchanged),
        "all_finite": bool(all_finite),
    }
    return checks, metrics


def run(config_path: Path, output: Path, device: str, smoke_updates=None):
    config, inputs = _load_config(config_path)
    output.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / config["recovery_dataset"]
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    chains = _load_chains(dataset, manifest)
    counts = {split: sum(row["split"] == split for row in chains) for split in ("train", "validation", "test")}
    if counts != config["split_counts"]:
        raise ValueError("L276 frozen split count mismatch")
    splits = {
        split: [row for row in chains if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    train_by_scene = defaultdict(list)
    for chain in splits["train"]:
        train_by_scene[chain["scene"]].append(chain)
    source_payload, source_agent, normalizer = _load_agent(
        ROOT / config["source_checkpoint"], device
    )
    source_hashes = _nonactor_hashes(source_agent, normalizer)
    replay = source_payload["replay_buffer"]
    replay_size = int(replay["size"])
    if replay_size != 6000:
        raise ValueError("L276 requires the frozen 6000-transition source replay")
    source_observations = np.asarray(replay["observations"][:replay_size], dtype=np.float32)
    source_targets = _predict_actions(source_agent, normalizer, source_observations)
    training = config["training"]
    updates = int(training["updates"] if smoke_updates is None else smoke_updates)
    checkpoint_interval = int(training["checkpoint_interval"])
    progress_rows = []
    final_paths = []
    for seed in training["seeds"]:
        state = source_payload["agent"]
        agent = SACAgent(
            int(state["observation_dim"]),
            int(state["action_dim"]),
            SACConfig.from_mapping(state["config"]),
            device=device,
            seed=int(seed),
        )
        agent.load_state_dict(state, load_optimizers=False)
        agent.train()
        before = _nonactor_hashes(agent, normalizer)
        rng = np.random.RandomState(int(seed))
        for update in range(1, updates + 1):
            recovery_obs, recovery_actions = _sample_recovery(
                train_by_scene, rng, int(training["recovery_batch_size"])
            )
            anchor_indices = rng.randint(
                replay_size, size=int(training["source_anchor_batch_size"])
            )
            batch_observations = np.concatenate((
                normalizer.normalize(recovery_obs),
                normalizer.normalize(source_observations[anchor_indices]),
            ), axis=0)
            batch_targets = np.concatenate((
                recovery_actions,
                source_targets[anchor_indices],
            ), axis=0)
            metrics = agent.behavior_cloning_update(
                batch_observations,
                batch_targets,
                log_std_weight=float(training["log_std_weight"]),
                target_log_std=float(training["target_log_std"]),
            )
            if update % checkpoint_interval == 0 or update == updates:
                current = _nonactor_hashes(agent, normalizer)
                unchanged = current == before == source_hashes
                checkpoint = output / ("seed_%d" % seed) / "checkpoints" / (
                    "update_%07d.pt" % update
                )
                checkpoint_sha = _save_checkpoint(
                    checkpoint, source_payload, agent, seed, update, config, current
                )
                progress_rows.append({
                    "seed": int(seed),
                    "update": int(update),
                    "checkpoint": str(checkpoint.relative_to(output)),
                    "checkpoint_sha256": checkpoint_sha,
                    "bc_loss": metrics["bc_loss"],
                    "bc_mean_rmse": metrics["bc_mean_rmse"],
                    "bc_gradient_norm": metrics["bc_gradient_norm"],
                    "nonactor_hashes_unchanged": bool(unchanged),
                    "device": device,
                })
                _write_csv(output / "progress.csv", progress_rows)
                print(json.dumps({
                    "stage": "training",
                    "seed": int(seed),
                    "update": int(update),
                    "target_updates": int(updates),
                    "checkpoint_count": len(progress_rows),
                    "device": device,
                }, sort_keys=True), flush=True)
        final_paths.append(
            output / ("seed_%d" % seed) / "checkpoints" / ("update_%07d.pt" % updates)
        )
    if smoke_updates is not None:
        # Exercise the real Windows CUDA + MuJoCo evaluation path on one frozen
        # validation recovery start before accepting the implementation smoke.
        base = load_yaml(ROOT / "configs/rl/l262_coverage_gated_value_6k.yaml")
        smoke_chain = splits["validation"][0]
        _, smoke_agent, smoke_normalizer = _load_agent(final_paths[0], device)
        environment = _resolved_environment(
            base,
            smoke_chain["scene_config"],
            len(smoke_chain["actions"]),
            smoke_chain["seed"],
            "l268",
        )
        try:
            source_rollout = _rollout_actor(
                environment, smoke_chain, source_agent, normalizer, 1e-6
            )
            initialized_rollout = _rollout_actor(
                environment, smoke_chain, smoke_agent, smoke_normalizer, 1e-6
            )
        finally:
            environment.close()
        summary = {
            "protocol": "L276",
            "status": "smoke_complete",
            "updates": updates,
            "checkpoint_count": len(progress_rows),
            "all_nonactor_hashes_unchanged": all(
                bool(row["nonactor_hashes_unchanged"]) for row in progress_rows
            ),
            "mujoco_rollout_checked": True,
            "smoke_chain_id": int(smoke_chain["chain_id"]),
            "source_rollout": source_rollout,
            "initialized_rollout": initialized_rollout,
        }
        _json_dump(output / "smoke_summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return summary

    # Frozen evaluation begins only after every seed has finished training.
    base = load_yaml(ROOT / "configs/rl/l262_coverage_gated_value_6k.yaml")
    source_teacher = {
        split: _teacher_metrics(source_agent, normalizer, splits[split])
        for split in ("validation", "test")
    }
    trained = []
    for seed, path in zip(training["seeds"], final_paths):
        payload, agent, loaded_normalizer = _load_agent(path, device)
        trained.append((int(seed), payload, agent, loaded_normalizer))
    rollout_rows = []
    seed_metrics = []
    for split in ("validation", "test"):
        by_scene = defaultdict(list)
        for chain in splits[split]:
            by_scene[chain["scene_config"]].append(chain)
        source_rows = {}
        for scene_config, scene_chains in sorted(by_scene.items()):
            maximum = max(len(row["actions"]) for row in scene_chains)
            environment = _resolved_environment(
                base, scene_config, maximum, scene_chains[0]["seed"], "l268"
            )
            try:
                for chain in scene_chains:
                    result = _rollout_actor(
                        environment, chain, source_agent, normalizer, 1e-6
                    )
                    source_rows[int(chain["chain_id"])] = result
            finally:
                environment.close()
        for seed, _, agent, loaded_normalizer in trained:
            for scene_config, scene_chains in sorted(by_scene.items()):
                maximum = max(len(row["actions"]) for row in scene_chains)
                environment = _resolved_environment(
                    base, scene_config, maximum, scene_chains[0]["seed"], "l268"
                )
                try:
                    for chain in scene_chains:
                        source = source_rows[int(chain["chain_id"])]
                        treatment = _rollout_actor(
                            environment, chain, agent, loaded_normalizer, 1e-6
                        )
                        common = {
                            "seed": int(seed),
                            "split": split,
                            "chain_id": int(chain["chain_id"]),
                            "scene": chain["scene"],
                            "severity": chain["severity"],
                            "side": int(chain["side"]),
                        }
                        rollout_rows.append({
                            **common,
                            "policy": "source_actor",
                            **source,
                            "return_gain_vs_source": 0.0,
                        })
                        rollout_rows.append({
                            **common,
                            "policy": "l276_initialized",
                            **treatment,
                            "return_gain_vs_source": (
                                treatment["discounted_return"]
                                - source["discounted_return"]
                            ),
                        })
                finally:
                    environment.close()
    for seed, _, agent, loaded_normalizer in trained:
        validation_teacher = _teacher_metrics(
            agent, loaded_normalizer, splits["validation"]
        )
        test_teacher = _teacher_metrics(agent, loaded_normalizer, splits["test"])
        treatment_source = _predict_actions(
            agent, loaded_normalizer, source_observations
        )
        rows = [row for row in rollout_rows if int(row["seed"]) == seed]
        validation_gains = [
            float(row["return_gain_vs_source"]) for row in rows
            if row["split"] == "validation" and row["policy"] == "l276_initialized"
        ]
        test_gains = [
            float(row["return_gain_vs_source"]) for row in rows
            if row["split"] == "test" and row["policy"] == "l276_initialized"
        ]
        test_source = [
            row for row in rows if row["split"] == "test" and row["policy"] == "source_actor"
        ]
        test_treatment = [
            row for row in rows if row["split"] == "test" and row["policy"] == "l276_initialized"
        ]
        seed_metrics.append({
            "seed": seed,
            "source_validation_rmse": source_teacher["validation"]["rmse"],
            "initialized_validation_rmse": validation_teacher["rmse"],
            "source_test_rmse": source_teacher["test"]["rmse"],
            "initialized_test_rmse": test_teacher["rmse"],
            "relative_test_rmse_improvement": (
                (source_teacher["test"]["rmse"] - test_teacher["rmse"])
                / source_teacher["test"]["rmse"]
            ),
            "median_validation_return_gain": float(np.median(validation_gains)),
            "median_test_return_gain": float(np.median(test_gains)),
            "test_reentry_fraction_gain": float(
                np.mean([row["corridor_reentry"] for row in test_treatment])
                - np.mean([row["corridor_reentry"] for row in test_source])
            ),
            "source_replay_mean_absolute_action_drift": float(
                np.mean(np.abs(treatment_source - source_targets))
            ),
        })
    final_nonactor = all(
        _nonactor_hashes(agent, loaded_normalizer) == source_hashes
        for _, _, agent, loaded_normalizer in trained
    )
    numeric_values = []
    for row in seed_metrics:
        numeric_values.extend(value for value in row.values() if isinstance(value, (int, float)))
    numeric_values.extend(float(row["discounted_return"]) for row in rollout_rows)
    all_finite = bool(np.isfinite(np.asarray(numeric_values, dtype=np.float64)).all())
    checks, metrics = _gate(
        seed_metrics, rollout_rows, final_nonactor, all_finite, config["gate"]
    )
    passed = all(checks.values())
    summary = {
        "protocol": "L276",
        "status": "complete",
        "decision": (
            "recovery_initialization_gate_pass"
            if passed else "recovery_initialization_gate_fail"
        ),
        "small_budget_sac_preregistration_authorized": bool(passed),
        "git_sha": git_sha(ROOT),
        "device": device,
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "config_sha256": _sha256(config_path),
        "input_sha256": {str(path.relative_to(ROOT)): value for path, value in inputs.items()},
        "seed_count": len(trained),
        "updates_per_seed": updates,
        "checkpoint_count": len(progress_rows),
        "split_counts": counts,
        "all_finite": all_finite,
        "nonactor_hashes_unchanged": final_nonactor,
        "checks": checks,
        "metrics": metrics,
        "seed_metrics": seed_metrics,
        "gate_pass": passed,
    }
    _write_csv(output / "rollouts.csv", rollout_rows)
    _write_csv(output / "seed_metrics.csv", seed_metrics)
    _json_dump(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke-updates", type=int)
    args = parser.parse_args()
    run(
        args.config.resolve(), args.output_dir.resolve(), args.device,
        smoke_updates=args.smoke_updates,
    )


if __name__ == "__main__":
    main()
