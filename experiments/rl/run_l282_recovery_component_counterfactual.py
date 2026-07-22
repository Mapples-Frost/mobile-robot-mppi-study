#!/usr/bin/env python3
"""Run the frozen L282 recovery action-component counterfactual diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from experiments.rl.run_l276_recovery_sequence_imitation_initialization import (
    _load_chains,
)
from mobile_robot_mppi.core.config import git_sha, load_yaml


DEFAULT_CONFIG = ROOT / "configs/rl/l282_recovery_component_counterfactual.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l282_recovery_component_counterfactual"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["protocol"] != "L282":
        raise ValueError("L282 protocol identifier mismatch")
    if not config["diagnostic_only"] or config["actor_training_authorized"]:
        raise ValueError("L282 must remain evaluation-only")
    if config["arms"] != [
        "teacher_both",
        "actor_both",
        "teacher_v_actor_omega",
        "actor_v_teacher_omega",
    ]:
        raise ValueError("L282 arm order drifted")
    checks = {
        ROOT / config["l281_summary"]["path"]: config["l281_summary"]["sha256"],
        ROOT / config["recovery_dataset"] / "manifest.json": config[
            "recovery_manifest_sha256"
        ],
    }
    for item in config["checkpoints"]:
        checks[ROOT / item["path"]] = item["sha256"]
    for artifact, expected in checks.items():
        if _sha256(artifact) != expected:
            raise ValueError("L282 input SHA256 mismatch: %s" % artifact)
    summary = json.loads(
        (ROOT / config["l281_summary"]["path"]).read_text(encoding="utf-8")
    )
    if summary["decision"] != "component_separated_anchor_gate_fail":
        raise ValueError("L282 requires the frozen L281 Gate decision")
    path_references = "\n".join(str(item) for item in checks).lower()
    forbidden = [
        token for token in config["forbidden_tokens"]
        if token.lower() in path_references
    ]
    if forbidden:
        raise ValueError("forbidden input path entered L282: %s" % forbidden)
    return config, checks


def _compose_action(arm, teacher_action, actor_action):
    teacher = np.asarray(teacher_action, dtype=np.float32)
    actor = np.asarray(actor_action, dtype=np.float32)
    if teacher.shape != (2,) or actor.shape != (2,):
        raise ValueError("L282 requires two-dimensional actions")
    if arm == "teacher_both":
        return teacher.copy()
    if arm == "actor_both":
        return actor.copy()
    if arm == "teacher_v_actor_omega":
        return np.asarray((teacher[0], actor[1]), dtype=np.float32)
    if arm == "actor_v_teacher_omega":
        return np.asarray((actor[0], teacher[1]), dtype=np.float32)
    raise ValueError("unknown L282 arm: %s" % arm)


def _goal_distance(environment):
    goal = np.asarray(environment.components["reference"].points[-1], dtype=np.float64)
    pose = np.asarray(environment.truth.pose.as_array(), dtype=np.float64)
    return float(np.linalg.norm(goal[:2] - pose[:2]))


def _rollout_hybrid(environment, chain, agent, normalizer, arm, reset_tolerance):
    expected = np.asarray(chain["observations"][0], dtype=np.float32)
    observation, _, path_state = _prepare_reset(
        environment,
        np.asarray(chain["initial_state"], dtype=np.float64),
        int(chain["seed"]),
    )
    reset_error = float(np.max(np.abs(observation - expected)))
    if reset_error > float(reset_tolerance):
        raise RuntimeError("L282 fixed-chain reset observation drifted")
    initial_cte = float(path_state["cross_track_error"])
    initial_progress = float(path_state["progress"])
    initial_goal_distance = _goal_distance(environment)
    gamma = float(environment.gamma)
    cumulative = 0.0
    reward_error = 0.0
    corridor_reentry = initial_cte <= 0.75
    collision = False
    boundary = False
    terminated = truncated = False
    last_info = None
    executed = 0
    agent.eval()
    for step, teacher_action in enumerate(chain["actions"]):
        if terminated or truncated:
            break
        actor_action, _ = agent.select_action(
            normalizer.normalize(observation), deterministic=True
        )
        action = _compose_action(arm, teacher_action, actor_action)
        observation, reward, terminated, truncated, last_info = environment.step(action)
        cumulative += gamma ** step * float(reward)
        executed += 1
        if arm == "teacher_both":
            reward_error = max(
                reward_error, abs(float(reward) - float(chain["rewards"][step]))
            )
        collision = collision or bool(last_info["collision"])
        boundary = boundary or not _inside_field(
            environment.config, environment.truth.pose.as_array()
        )
        corridor_reentry = corridor_reentry or (
            float(last_info["cross_track_error"]) <= 0.75
        )
    if last_info is None:
        raise RuntimeError("L282 rollout produced no transition")
    final_goal_distance = _goal_distance(environment)
    numeric = np.asarray((
        cumulative,
        reset_error,
        reward_error,
        last_info["cross_track_error"],
        last_info["path_progress"],
        initial_goal_distance,
        final_goal_distance,
    ))
    if not np.isfinite(numeric).all():
        raise FloatingPointError("L282 rollout produced non-finite values")
    return {
        "horizon": int(len(chain["actions"])),
        "steps_executed": int(executed),
        "discounted_return": float(cumulative),
        "initial_cross_track_error": initial_cte,
        "final_cross_track_error": float(last_info["cross_track_error"]),
        "cross_track_delta": float(last_info["cross_track_error"] - initial_cte),
        "path_progress_delta": float(last_info["path_progress"] - initial_progress),
        "goal_distance_delta": float(final_goal_distance - initial_goal_distance),
        "corridor_reentry": bool(corridor_reentry),
        "collision": bool(collision),
        "boundary_violation": bool(boundary),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "max_reset_observation_error": reset_error,
        "max_teacher_reward_error": reward_error,
    }


def _build_advantages(rollout_rows):
    grouped = defaultdict(dict)
    for row in rollout_rows:
        key = (int(row["seed"]), int(row["chain_id"]))
        grouped[key][row["arm"]] = row
    advantages = []
    for (seed, chain_id), arms in sorted(grouped.items()):
        if set(arms) != {
            "teacher_both", "actor_both", "teacher_v_actor_omega",
            "actor_v_teacher_omega",
        }:
            raise ValueError("L282 incomplete arm set")
        teacher = float(arms["teacher_both"]["discounted_return"])
        actor = float(arms["actor_both"]["discounted_return"])
        gap = teacher - actor
        velocity_gain = (
            float(arms["teacher_v_actor_omega"]["discounted_return"]) - actor
        )
        steering_gain = (
            float(arms["actor_v_teacher_omega"]["discounted_return"]) - actor
        )
        advantages.append({
            "seed": seed,
            "chain_id": chain_id,
            "scene": arms["actor_both"]["scene"],
            "teacher_gap": gap,
            "teacher_velocity_gain": velocity_gain,
            "teacher_steering_gain": steering_gain,
            "teacher_velocity_recovered_fraction": (
                velocity_gain / gap if gap > 0.0 else float("nan")
            ),
            "teacher_steering_recovered_fraction": (
                steering_gain / gap if gap > 0.0 else float("nan")
            ),
        })
    return advantages


def _decision(advantages, gate):
    by_seed_scene = defaultdict(list)
    for row in advantages:
        by_seed_scene[(int(row["seed"]), row["scene"])].append(row)
    threshold = float(gate["minimum_recovered_loss_fraction"])
    minimum_scenes = int(gate["minimum_attributed_scenes"])
    minimum_seeds = int(gate["minimum_attributed_seeds"])
    seed_rows = []
    for (seed, scene), rows in sorted(by_seed_scene.items()):
        gaps = np.asarray([row["teacher_gap"] for row in rows], dtype=np.float64)
        gap = float(np.mean(gaps))
        velocity_gain = float(np.mean([
            row["teacher_velocity_gain"] for row in rows
        ]))
        steering_gain = float(np.mean([
            row["teacher_steering_gain"] for row in rows
        ]))
        seed_rows.append({
            "seed": seed,
            "scene": scene,
            "teacher_gap": gap,
            "teacher_velocity_recovered_fraction": (
                velocity_gain / gap if gap > 0.0 else float("nan")
            ),
            "teacher_steering_recovered_fraction": (
                steering_gain / gap if gap > 0.0 else float("nan")
            ),
        })
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    coverage = {}
    for seed in seeds:
        rows = [row for row in seed_rows if int(row["seed"]) == seed]
        positive = [row for row in rows if row["teacher_gap"] > 0.0]
        velocity_scenes = sum(
            row["teacher_velocity_recovered_fraction"] >= threshold
            for row in positive
        )
        steering_scenes = sum(
            row["teacher_steering_recovered_fraction"] >= threshold
            for row in positive
        )
        coverage[seed] = {
            "positive_teacher_gap_scenes": len(positive),
            "velocity_attributed_scenes": int(velocity_scenes),
            "steering_attributed_scenes": int(steering_scenes),
        }
    velocity_seed_count = sum(
        row["velocity_attributed_scenes"] >= minimum_scenes
        for row in coverage.values()
    )
    steering_seed_count = sum(
        row["steering_attributed_scenes"] >= minimum_scenes
        for row in coverage.values()
    )
    coupled_seed_count = sum(
        row["positive_teacher_gap_scenes"] >= minimum_scenes
        for row in coverage.values()
    )
    velocity = velocity_seed_count >= minimum_seeds
    steering = steering_seed_count >= minimum_seeds
    if velocity and steering:
        decision = "both_components_contribute"
    elif velocity:
        decision = "velocity_component_bottleneck"
    elif steering:
        decision = "steering_component_bottleneck"
    elif coupled_seed_count >= minimum_seeds:
        decision = "coupled_sequence_bottleneck"
    else:
        decision = "component_attribution_unresolved"
    return decision, {
        "seed_scene_rows": seed_rows,
        "seed_coverage": {str(key): value for key, value in coverage.items()},
        "velocity_attributed_seed_count": int(velocity_seed_count),
        "steering_attributed_seed_count": int(steering_seed_count),
        "coupled_positive_gap_seed_count": int(coupled_seed_count),
    }


def run(config_path: Path, output: Path, device: str, maximum_chains=None):
    config, inputs = _load_config(config_path)
    output.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / config["recovery_dataset"]
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    chains = [
        row for row in _load_chains(dataset, manifest)
        if row["split"] == config["split"]
    ]
    if len(chains) != int(config["expected_chains"]):
        raise ValueError("L282 frozen chain count mismatch")
    if maximum_chains is not None:
        chains = chains[:int(maximum_chains)]
    base = load_yaml(ROOT / config["base_config"])
    rollout_rows = []
    progress_rows = []
    for checkpoint_index, checkpoint in enumerate(config["checkpoints"]):
        seed = int(checkpoint["seed"])
        _, agent, normalizer = _load_agent(ROOT / checkpoint["path"], device)
        by_scene = defaultdict(list)
        for chain in chains:
            by_scene[chain["scene_config"]].append(chain)
        for scene_config, scene_chains in sorted(by_scene.items()):
            maximum = max(len(row["actions"]) for row in scene_chains)
            environment = _resolved_environment(
                base, scene_config, maximum, scene_chains[0]["seed"],
                config["reset_contract"],
            )
            try:
                for chain in scene_chains:
                    for arm in config["arms"]:
                        result = _rollout_hybrid(
                            environment, chain, agent, normalizer, arm,
                            config["reset_tolerance"],
                        )
                        rollout_rows.append({
                            "seed": seed,
                            "chain_id": int(chain["chain_id"]),
                            "scene": chain["scene"],
                            "scene_config": chain["scene_config"],
                            "arm": arm,
                            **result,
                        })
                    progress_rows.append({
                        "checkpoint_index": int(checkpoint_index),
                        "seed": seed,
                        "completed_chains": int(sum(
                            int(row["seed"] == seed) for row in progress_rows
                        ) + 1),
                        "rollout_rows": len(rollout_rows),
                        "device": device,
                    })
                    _write_csv(output / "progress.csv", progress_rows)
                    print(json.dumps({
                        "stage": "evaluation",
                        "seed": seed,
                        "completed_chain_seed_pairs": len(progress_rows),
                        "rollout_rows": len(rollout_rows),
                        "device": device,
                    }, sort_keys=True), flush=True)
            finally:
                environment.close()
    advantages = _build_advantages(rollout_rows)
    expected_rollouts = len(config["checkpoints"]) * len(chains) * len(config["arms"])
    integrity = {
        "rollout_rows": len(rollout_rows),
        "expected_rollout_rows": int(expected_rollouts),
        "advantage_rows": len(advantages),
        "expected_advantage_rows": int(len(config["checkpoints"]) * len(chains)),
        "maximum_reset_observation_error": float(max(
            row["max_reset_observation_error"] for row in rollout_rows
        )),
        "maximum_teacher_reward_error": float(max(
            row["max_teacher_reward_error"] for row in rollout_rows
            if row["arm"] == "teacher_both"
        )),
        "all_rollouts_finite": bool(all(
            math.isfinite(float(row[key]))
            for row in rollout_rows
            for key in (
                "discounted_return", "cross_track_delta", "path_progress_delta",
                "goal_distance_delta", "max_reset_observation_error",
                "max_teacher_reward_error",
            )
        )),
    }
    integrity["pass"] = bool(
        integrity["rollout_rows"] == integrity["expected_rollout_rows"]
        and integrity["advantage_rows"] == integrity["expected_advantage_rows"]
        and integrity["maximum_reset_observation_error"] <= config["reset_tolerance"]
        and integrity["maximum_teacher_reward_error"] <= config["teacher_reward_tolerance"]
        and integrity["all_rollouts_finite"]
    )
    if not integrity["pass"]:
        raise RuntimeError("L282 integrity Gate failed")
    decision, attribution = _decision(advantages, config["gate"])
    _write_csv(output / "rollouts.csv", rollout_rows)
    _write_csv(output / "advantages.csv", advantages)
    _write_csv(output / "seed_scene_attribution.csv", attribution.pop("seed_scene_rows"))
    summary = {
        "protocol": "L282",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "device": device,
        "decision": decision,
        "integrity": integrity,
        "attribution": attribution,
        "actor_training_authorized": False,
        "final_map_evaluation_authorized": False,
        "inputs": {str(path.relative_to(ROOT)): expected for path, expected in inputs.items()},
    }
    _json_dump(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-chains", type=int)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    run(config_path, output, args.device, args.maximum_chains)


if __name__ == "__main__":
    main()
