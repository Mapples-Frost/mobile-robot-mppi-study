#!/usr/bin/env python3
"""Run the frozen L275 recovery-commitment continuation diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rl.run_l263_counterfactual_actor_diagnosis import (
    _load_agent,
    _prepare_reset,
)
from experiments.rl.run_l269_horizon_diagnostic import _resolved_environment
from mobile_robot_mppi.core.config import git_sha, load_yaml


DEFAULT_CONFIG = ROOT / "configs/rl/l275_recovery_commitment_continuation_diagnosis.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l275_recovery_commitment_continuation_diagnosis"


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
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["l275"]
    checks = {
        ROOT / config["source_checkpoint"]: config["source_checkpoint_sha256"],
        ROOT / config["l268_dataset"] / "manifest.json": (
            config["l268_dataset_manifest_sha256"]
        ),
        ROOT / config["l268_heldout_states"]: config["l268_heldout_states_sha256"],
        ROOT / config["l269_summary"]: config["l269_summary_sha256"],
        ROOT / config["l274_summary"]: config["l274_summary_sha256"],
    }
    for artifact, expected in checks.items():
        actual = _sha256(artifact)
        if actual != expected:
            raise ValueError("L275 input SHA256 mismatch: %s" % artifact)
    references = "\n".join(str(item) for item in checks).lower()
    entered = [
        token for token in config["forbidden_tokens"]
        if token.lower() in references
    ]
    if entered:
        raise ValueError("forbidden artifact entered L275: %s" % entered)
    l269 = json.loads((ROOT / config["l269_summary"]).read_text(encoding="utf-8"))
    l274 = json.loads((ROOT / config["l274_summary"]).read_text(encoding="utf-8"))
    if l269["status"] != "horizon_gate_fail":
        raise ValueError("L275 requires the frozen negative L269 status")
    if l274["decision"] != "temporal_or_target_dynamics_unresolved":
        raise ValueError("L275 requires the frozen unresolved L274 decision")
    return config, checks


def _rollout_commitment(
    environment,
    initial_state,
    seed,
    expected_observation,
    recorded_actions,
    recorded_rewards,
    commitment,
    agent,
    normalizer,
    reset_tolerance,
):
    observation, _, initial_path = _prepare_reset(environment, initial_state, seed)
    reset_error = float(np.max(np.abs(observation - expected_observation)))
    if reset_error > float(reset_tolerance):
        raise RuntimeError("L275 fixed-state reset observation drifted")
    horizon = len(recorded_actions)
    commitment_steps = horizon if commitment == "full_chain" else int(commitment)
    commitment_steps = min(commitment_steps, horizon)
    initial_distance = float(environment._distance_to_final(environment.truth))
    initial_cte = float(initial_path["cross_track_error"])
    initial_signed_cte = float(initial_path["signed_cross_track_error"])
    initial_progress = float(initial_path["progress"])
    gamma = float(environment.gamma)
    cumulative = 0.0
    reward_terms = defaultdict(float)
    prefix_error = 0.0
    corridor_reentry = initial_cte <= 0.75
    terminated = truncated = False
    last_info = None
    for step in range(horizon):
        if terminated or truncated:
            break
        if step < commitment_steps:
            action = np.asarray(recorded_actions[step], dtype=np.float32)
        else:
            action, _ = agent.select_action(
                normalizer.normalize(observation), deterministic=True
            )
        observation, reward, terminated, truncated, last_info = environment.step(action)
        cumulative += gamma ** step * float(reward)
        for name, value in last_info["reward_terms"].items():
            reward_terms[name] += gamma ** step * float(value)
        if step < commitment_steps:
            prefix_error = max(
                prefix_error, abs(float(reward) - float(recorded_rewards[step]))
            )
        corridor_reentry = corridor_reentry or (
            float(last_info["cross_track_error"]) <= 0.75
        )
    if last_info is None:
        raise RuntimeError("L275 rollout produced no transition")
    numeric = np.asarray((
        cumulative,
        reset_error,
        prefix_error,
        last_info["cross_track_error"],
        last_info["path_progress"],
        last_info["goal_distance"],
    ), dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("L275 rollout produced non-finite output")
    return {
        "commitment": str(commitment),
        "commitment_steps": int(commitment_steps),
        "horizon": int(horizon),
        "steps_executed": int(environment.steps),
        "discounted_return": float(cumulative),
        "cross_track_delta": float(last_info["cross_track_error"] - initial_cte),
        "signed_cross_track_delta": float(
            last_info["signed_cross_track_error"] - initial_signed_cte
        ),
        "path_progress_delta": float(last_info["path_progress"] - initial_progress),
        "goal_distance_delta": float(last_info["goal_distance"] - initial_distance),
        "corridor_reentry": bool(corridor_reentry),
        "collision": bool(last_info["collision"]),
        "safety_override": bool(last_info["safety_override"]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "max_reset_observation_error": reset_error,
        "max_recorded_prefix_reward_error": prefix_error,
        "reward_terms": json.dumps(dict(reward_terms), sort_keys=True),
    }


def _gate(advantage_rows, integrity, gate):
    order = ("1", "5", "10", "20", "40", "full_chain")
    by_state = defaultdict(dict)
    scene_full = defaultdict(list)
    for row in advantage_rows:
        by_state[row["state_id"]][str(row["commitment"])] = float(
            row["recovery_advantage"]
        )
        if str(row["commitment"]) == "full_chain":
            scene_full[row["scene"]].append(float(row["recovery_advantage"]) > 0.0)
    if not by_state or any(set(values) != set(order) for values in by_state.values()):
        raise ValueError("L275 commitment coverage is incomplete")
    positive_fractions = {
        key: float(np.mean([values[key] > 0.0 for values in by_state.values()]))
        for key in order
    }
    median_advantages = {
        key: float(np.median([values[key] for values in by_state.values()]))
        for key in order
    }
    first_positive = []
    for values in by_state.values():
        candidates = [key for key in order if values[key] > 0.0]
        first_positive.append(None if not candidates else candidates[0])
    late_minimum = int(gate["late_commitment_minimum_steps"])
    late_fraction = float(np.mean([
        value is not None
        and (value == "full_chain" or int(value) >= late_minimum)
        for value in first_positive
    ]))
    scene_fractions = {
        scene: float(np.mean(values)) for scene, values in sorted(scene_full.items())
    }
    scenes_majority = sum(value >= 0.50 for value in scene_fractions.values())
    monotonic_states = []
    for values in by_state.values():
        sequence = np.asarray([values[key] for key in order], dtype=np.float64)
        monotonic_states.append(bool(np.all(np.diff(sequence) >= -1e-12)))
    metrics = {
        "positive_fractions": positive_fractions,
        "median_advantages": median_advantages,
        "positive_fraction_gain_full_vs_one": (
            positive_fractions["full_chain"] - positive_fractions["1"]
        ),
        "late_first_positive_fraction": late_fraction,
        "first_positive_commitment_counts": dict(Counter(
            "never" if value is None else value for value in first_positive
        )),
        "scene_full_chain_positive_fractions": scene_fractions,
        "scenes_with_full_chain_majority_positive": int(scenes_majority),
        "monotonic_state_fraction": float(np.mean(monotonic_states)),
    }
    checks = {
        "full_chain_positive_fraction": (
            positive_fractions["full_chain"]
            >= float(gate["minimum_full_chain_positive_fraction"])
        ),
        "positive_fraction_gain_full_vs_one": (
            metrics["positive_fraction_gain_full_vs_one"]
            >= float(gate["minimum_positive_fraction_gain_full_vs_one"])
        ),
        "median_full_greater_than_one": (
            median_advantages["full_chain"] > median_advantages["1"]
        ),
        "late_first_positive_fraction": (
            late_fraction >= float(gate["minimum_late_first_positive_fraction"])
        ),
        "scene_coverage": (
            scenes_majority
            >= int(gate["minimum_scenes_with_full_chain_majority_positive"])
        ),
        "integrity": bool(integrity),
    }
    return checks, metrics


def run(config_path: Path, output: Path, device: str):
    config, inputs = _load_config(config_path)
    if not config["diagnostic_only"] or config["actor_training_authorized"]:
        raise ValueError("L275 must remain diagnostic-only")
    payload, agent, normalizer = _load_agent(ROOT / config["source_checkpoint"], device)
    if int(payload["agent"]["observation_dim"]) != 69:
        raise ValueError("L275 requires the frozen 69D Actor")
    output.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / config["l268_dataset"]
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    chains = {int(row["chain_id"]): row for row in manifest["chains"]}
    states = json.loads((ROOT / config["l268_heldout_states"]).read_text(encoding="utf-8"))
    if len(states) != int(config["expected_states"]):
        raise ValueError("L275 held-out state count mismatch")
    if len({row["scene"] for row in states}) != int(config["expected_scenes"]):
        raise ValueError("L275 scene count mismatch")
    base = load_yaml(ROOT / "configs/rl/l262_coverage_gated_value_6k.yaml")
    rollout_rows = []
    advantage_rows = []
    grouped = defaultdict(list)
    for state in states:
        grouped[state["scene_config"]].append(state)
    completed = 0
    for scene_config, scene_states in sorted(grouped.items()):
        maximum = max(int(chains[int(row["chain_id"])]["steps"]) for row in scene_states)
        environment = _resolved_environment(
            base, scene_config, maximum, scene_states[0]["seed"], "l268"
        )
        try:
            for state in scene_states:
                chain = chains[int(state["chain_id"])]
                with np.load(dataset / chain["student_npz"], allow_pickle=False) as shard:
                    observations = shard["observations"].copy()
                    actions = shard["actions"].copy()
                    rewards = shard["rewards"].reshape(-1).copy()
                if len(actions) < 40:
                    raise ValueError("L275 held-out chain shorter than 40 steps")
                common = {
                    "state_id": state["state_id"],
                    "chain_id": int(state["chain_id"]),
                    "scene": state["scene"],
                    "scene_index": int(state["scene_index"]),
                    "split": state["split"],
                    "severity": state["severity"],
                    "side": int(state["side"]),
                    "initial_cte_m": float(state["initial_cte_m"]),
                }
                baseline = _rollout_commitment(
                    environment,
                    np.asarray(state["initial_state"], dtype=np.float64),
                    int(state["seed"]),
                    observations[0],
                    actions,
                    rewards,
                    0,
                    agent,
                    normalizer,
                    config["reset_tolerance"],
                )
                baseline["commitment"] = "source_actor"
                baseline["commitment_steps"] = 0
                rollout_rows.append({**common, **baseline})
                for commitment in config["commitments"]:
                    result = _rollout_commitment(
                        environment,
                        np.asarray(state["initial_state"], dtype=np.float64),
                        int(state["seed"]),
                        observations[0],
                        actions,
                        rewards,
                        commitment,
                        agent,
                        normalizer,
                        config["reset_tolerance"],
                    )
                    rollout_rows.append({**common, **result})
                    advantage_rows.append({
                        **common,
                        "commitment": str(commitment),
                        "commitment_steps": result["commitment_steps"],
                        "horizon": result["horizon"],
                        "recovery_return": result["discounted_return"],
                        "actor_return": baseline["discounted_return"],
                        "recovery_advantage": (
                            result["discounted_return"] - baseline["discounted_return"]
                        ),
                    })
                completed += 1
                print(json.dumps({
                    "completed_states": completed,
                    "expected_states": int(config["expected_states"]),
                    "rollout_rows": len(rollout_rows),
                    "device": device,
                }, sort_keys=True), flush=True)
        finally:
            environment.close()
    max_reset = max(float(row["max_reset_observation_error"]) for row in rollout_rows)
    max_prefix = max(
        float(row["max_recorded_prefix_reward_error"])
        for row in rollout_rows if row["commitment"] != "source_actor"
    )
    finite = all(np.isfinite(float(row["discounted_return"])) for row in rollout_rows)
    integrity = (
        finite
        and max_reset <= float(config["reset_tolerance"])
        and max_prefix <= float(config["recorded_prefix_reward_tolerance"])
    )
    checks, metrics = _gate(advantage_rows, integrity, config["gate"])
    passed = all(checks.values())
    summary = {
        "protocol": "L275",
        "status": "complete",
        "decision": (
            "continuation_policy_mismatch" if passed
            else "temporal_history_direct_return_crossfit_required"
        ),
        "actor_training_authorized": False,
        "git_sha": git_sha(ROOT),
        "device": device,
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "config_sha256": _sha256(config_path),
        "input_sha256": {str(path.relative_to(ROOT)): value for path, value in inputs.items()},
        "state_count": len(states),
        "scene_count": len({row["scene"] for row in states}),
        "rollout_rows": len(rollout_rows),
        "advantage_rows": len(advantage_rows),
        "all_finite": finite,
        "maximum_reset_observation_error": max_reset,
        "maximum_recorded_prefix_reward_error": max_prefix,
        "checks": checks,
        "metrics": metrics,
        "gate_pass": passed,
    }
    _write_csv(output / "commitment_rollouts.csv", rollout_rows)
    _write_csv(output / "commitment_advantages.csv", advantage_rows)
    _json_dump(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(args.config.resolve(), args.output_dir.resolve(), args.device)


if __name__ == "__main__":
    main()

