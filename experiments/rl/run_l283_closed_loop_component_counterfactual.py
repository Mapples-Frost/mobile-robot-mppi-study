#!/usr/bin/env python3
"""Run the frozen L283 closed-loop recovery-component diagnosis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
from experiments.rl.run_l282_recovery_component_counterfactual import (
    _build_advantages,
    _compose_action,
    _decision,
    _goal_distance,
    _json_dump,
    _sha256,
    _write_csv,
)
from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.rl.scripted_direct_control import (
    ScriptedDirectControlConfig,
    ScriptedPolylineDirectControl,
)


DEFAULT_CONFIG = ROOT / "configs/rl/l283_closed_loop_component_counterfactual.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l283_closed_loop_component_counterfactual"


def _load_config(path: Path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["protocol"] != "L283":
        raise ValueError("L283 protocol identifier mismatch")
    if not config["diagnostic_only"] or config["actor_training_authorized"]:
        raise ValueError("L283 must remain evaluation-only")
    checks = {
        ROOT / config["l282_summary"]["path"]: config["l282_summary"]["sha256"],
        ROOT / config["recovery_dataset"] / "manifest.json": config[
            "recovery_manifest_sha256"
        ],
        ROOT / config["teacher_config"]: config["teacher_config_sha256"],
    }
    for checkpoint in config["checkpoints"]:
        checks[ROOT / checkpoint["path"]] = checkpoint["sha256"]
    for artifact, expected in checks.items():
        if _sha256(artifact) != expected:
            raise ValueError("L283 input SHA256 mismatch: %s" % artifact)
    l282 = json.loads(
        (ROOT / config["l282_summary"]["path"]).read_text(encoding="utf-8")
    )
    if l282["decision"] != "coupled_sequence_bottleneck":
        raise ValueError("L283 requires the frozen coupled L282 decision")
    references = "\n".join(str(path) for path in checks).lower()
    forbidden = [
        token for token in config["forbidden_tokens"] if token.lower() in references
    ]
    if forbidden:
        raise ValueError("forbidden input path entered L283: %s" % forbidden)
    teacher_protocol = yaml.safe_load(
        (ROOT / config["teacher_config"]).read_text(encoding="utf-8")
    )["l267"]
    return config, checks, teacher_protocol["collection"]["teacher"]


def _new_teacher(environment, chain, teacher_config):
    route = np.asarray(environment.config["task"]["points"], dtype=np.float64)
    teacher = ScriptedPolylineDirectControl(
        route,
        environment.action_spec,
        ScriptedDirectControlConfig(
            lookahead_distance=float(teacher_config["lookahead_m"]),
            cruise_speed=float(teacher_config["cruise_speed_mps"]),
            yaw_gain=float(teacher_config["yaw_gain"]),
        ),
    )
    teacher.reset()
    teacher.tracker.progress = float(chain["progress_m"])
    return teacher


def _closed_loop_teacher_action(teacher, environment, chain):
    action, _ = teacher.action(environment.truth.pose)
    return np.clip(
        np.asarray(action, dtype=np.float64)
        + np.asarray((chain["delta_v"], chain["delta_omega"]), dtype=np.float64),
        -1.0,
        1.0,
    ).astype(np.float32)


def _rollout_closed_loop(
    environment, chain, agent, normalizer, arm, teacher_config, reset_tolerance,
):
    expected = np.asarray(chain["observations"][0], dtype=np.float32)
    observation, _, path_state = _prepare_reset(
        environment,
        np.asarray(chain["initial_state"], dtype=np.float64),
        int(chain["seed"]),
    )
    reset_error = float(np.max(np.abs(observation - expected)))
    if reset_error > float(reset_tolerance):
        raise RuntimeError("L283 fixed-chain reset observation drifted")
    teacher = _new_teacher(environment, chain, teacher_config)
    initial_cte = float(path_state["cross_track_error"])
    initial_progress = float(path_state["progress"])
    initial_goal_distance = _goal_distance(environment)
    gamma = float(environment.gamma)
    cumulative = 0.0
    reward_error = 0.0
    action_error = 0.0
    corridor_reentry = initial_cte <= 0.75
    collision = False
    boundary = False
    terminated = truncated = False
    last_info = None
    executed = 0
    agent.eval()
    for step in range(len(chain["actions"])):
        if terminated or truncated:
            break
        teacher_action = _closed_loop_teacher_action(teacher, environment, chain)
        actor_action, _ = agent.select_action(
            normalizer.normalize(observation), deterministic=True
        )
        action = _compose_action(arm, teacher_action, actor_action)
        observation, reward, terminated, truncated, last_info = environment.step(action)
        cumulative += gamma ** step * float(reward)
        executed += 1
        if arm == "teacher_both":
            action_error = max(
                action_error,
                float(np.max(np.abs(teacher_action - chain["actions"][step]))),
            )
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
        raise RuntimeError("L283 rollout produced no transition")
    final_goal_distance = _goal_distance(environment)
    numeric = np.asarray((
        cumulative, reset_error, reward_error, action_error,
        last_info["cross_track_error"], last_info["path_progress"],
        initial_goal_distance, final_goal_distance,
    ))
    if not np.isfinite(numeric).all():
        raise FloatingPointError("L283 rollout produced non-finite values")
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
        "max_teacher_action_error": action_error,
        "max_teacher_reward_error": reward_error,
    }


def run(config_path: Path, output: Path, device: str, maximum_chains=None):
    config, inputs, teacher_config = _load_config(config_path)
    output.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / config["recovery_dataset"]
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    chains = [
        chain for chain in _load_chains(dataset, manifest)
        if chain["split"] == config["split"]
    ]
    if len(chains) != int(config["expected_chains"]):
        raise ValueError("L283 frozen chain count mismatch")
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
        completed_seed_chains = 0
        for scene_config, scene_chains in sorted(by_scene.items()):
            maximum = max(len(chain["actions"]) for chain in scene_chains)
            environment = _resolved_environment(
                base, scene_config, maximum, scene_chains[0]["seed"],
                config["reset_contract"],
            )
            try:
                for chain in scene_chains:
                    for arm in config["arms"]:
                        result = _rollout_closed_loop(
                            environment, chain, agent, normalizer, arm,
                            teacher_config, config["reset_tolerance"],
                        )
                        rollout_rows.append({
                            "seed": seed,
                            "chain_id": int(chain["chain_id"]),
                            "scene": chain["scene"],
                            "scene_config": chain["scene_config"],
                            "arm": arm,
                            **result,
                        })
                    completed_seed_chains += 1
                    progress_rows.append({
                        "checkpoint_index": int(checkpoint_index),
                        "seed": seed,
                        "completed_chains": int(completed_seed_chains),
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
        "maximum_teacher_action_error": float(max(
            row["max_teacher_action_error"] for row in rollout_rows
            if row["arm"] == "teacher_both"
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
                "max_teacher_action_error", "max_teacher_reward_error",
            )
        )),
    }
    integrity["pass"] = bool(
        integrity["rollout_rows"] == integrity["expected_rollout_rows"]
        and integrity["advantage_rows"] == integrity["expected_advantage_rows"]
        and integrity["maximum_reset_observation_error"] <= config["reset_tolerance"]
        and integrity["maximum_teacher_action_error"] <= config["teacher_action_tolerance"]
        and integrity["maximum_teacher_reward_error"] <= config["teacher_reward_tolerance"]
        and integrity["all_rollouts_finite"]
    )
    if not integrity["pass"]:
        raise RuntimeError("L283 integrity Gate failed")
    decision, attribution = _decision(advantages, config["gate"])
    _write_csv(output / "rollouts.csv", rollout_rows)
    _write_csv(output / "advantages.csv", advantages)
    _write_csv(output / "seed_scene_attribution.csv", attribution.pop("seed_scene_rows"))
    summary = {
        "protocol": "L283",
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
