#!/usr/bin/env python3
"""Run the preregistered L269 fixed-state horizon credit diagnosis."""

from __future__ import annotations

import argparse
import copy
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
    _rollout,
)
from experiments.rl.train_rl_sampling_prior import _scene_configs
from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.rl.environment import DirectControlEnv


DEFAULT_CONFIG = ROOT / "configs/rl/l269_long_horizon_credit_probe.yaml"
DEFAULT_OUTPUT = ROOT / (
    "results/research_platform/rl/l269_long_horizon_credit_probe/"
    "horizon_diagnostic"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows):
    rows = list(rows)
    fields = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                fields.append(name)
                seen.add(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["l269"]
    checkpoint = ROOT / config["source_checkpoint"]
    checks = {
        checkpoint: config["source_checkpoint_sha256"],
        ROOT / config["l263"]["directory"] / "state_manifest.json": (
            config["l263"]["state_manifest_sha256"]
        ),
        ROOT / config["l263"]["directory"] / "critic_actions.csv": (
            config["l263"]["critic_actions_sha256"]
        ),
        ROOT / config["l263"]["directory"] / "counterfactual_rollouts.csv": (
            config["l263"]["counterfactual_rollouts_sha256"]
        ),
        ROOT / config["l268"]["dataset"] / "manifest.json": (
            config["l268"]["dataset_manifest_sha256"]
        ),
        ROOT / config["l268"]["heldout_diagnostic"] / "summary.json": (
            config["l268"]["heldout_summary_sha256"]
        ),
        ROOT / config["l268"]["heldout_diagnostic"] / "state_manifest.json": (
            config["l268"]["heldout_states_sha256"]
        ),
        ROOT / config["l268"]["heldout_diagnostic"] / "action_returns_h40.csv": (
            config["l268"]["heldout_h40_sha256"]
        ),
    }
    for artifact, expected in checks.items():
        actual = _sha256(artifact)
        if actual != expected:
            raise ValueError("L269 input SHA256 mismatch: %s" % artifact)
    references = "\n".join(str(path) for path in checks).lower()
    forbidden = [
        token for token in config["forbidden_tokens"]
        if token.lower() in references
    ]
    if forbidden:
        raise ValueError("forbidden L269 input references: %s" % forbidden)
    return config, checkpoint, checks


def _resolved_environment(base, scene_config, maximum_steps, seed, reset_contract):
    resolved = _scene_configs(base, [scene_config])[0]
    resolved = copy.deepcopy(resolved)
    resolved["experiment"]["max_steps"] = int(maximum_steps) + 1
    zero_noise = [0.0] * len(resolved["experiment"]["initial_state"])
    if reset_contract == "l263":
        # Match run_l263_counterfactual_actor_diagnosis exactly.  L263 froze
        # the training-level reset noise and left its curriculum mapping
        # otherwise untouched.
        resolved["rl"]["training"]["initial_state_noise"] = zero_noise
    elif reset_contract == "l268":
        # Match generate_l267_recovery_dataset as used by formal L268.
        resolved["experiment"]["initial_state_noise"] = zero_noise
        resolved["rl"]["training"]["initial_state_curriculum"] = {"enabled": False}
    else:
        raise ValueError("unknown frozen reset contract: %s" % reset_contract)
    return DirectControlEnv(resolved, ROOT, seed=int(seed))


def _candidate_rows(
    state_set, state_id, state, environment, agent, normalizer, horizons,
    recovery_action, actor_action, expected_observation,
):
    rows = []
    initial = np.asarray(state["initial_state"], dtype=np.float64)
    seed = int(state["seed"])
    for action_id, action in (
        ("recovery", np.asarray(recovery_action, dtype=np.float32)),
        ("actor", np.asarray(actor_action, dtype=np.float32)),
    ):
        records = _rollout(
            environment, initial, seed, expected_observation, action,
            "actor_follow", agent, normalizer, tuple(horizons),
        )
        for horizon in horizons:
            rows.append({
                "state_set": state_set,
                "state_id": state_id,
                "scene": state["scene"],
                "scene_index": state.get("scene_index", ""),
                "split": state.get("split", state.get("scene_role", "")),
                "severity": state.get("severity", state.get("kind", "")),
                "initial_cte_m": float(
                    state.get("initial_cte_m", state["path_state"]["cross_track_error"])
                ),
                "action_id": action_id,
                "normalized_v": float(action[0]),
                "normalized_omega": float(action[1]),
                "continuation": "actor_follow",
                **records[int(horizon)],
            })
    return rows


def _discounted_prefix(rewards, gamma, horizon):
    count = min(int(horizon), len(rewards))
    powers = gamma ** np.arange(count, dtype=np.float64)
    return float(np.sum(powers * np.asarray(rewards[:count], dtype=np.float64)))


def _advantage_rows(candidate_rows):
    grouped = defaultdict(dict)
    metadata = {}
    for row in candidate_rows:
        key = (row["state_set"], row["state_id"], int(row["horizon"]))
        grouped[key][row["action_id"]] = float(row["discounted_return"])
        metadata[key] = row
    result = []
    for key, values in sorted(grouped.items()):
        if set(values) != {"recovery", "actor"}:
            raise ValueError("candidate action coverage is incomplete")
        row = metadata[key]
        result.append({
            "state_set": key[0],
            "state_id": key[1],
            "scene": row["scene"],
            "scene_index": row["scene_index"],
            "split": row["split"],
            "severity": row["severity"],
            "initial_cte_m": row["initial_cte_m"],
            "horizon": key[2],
            "recovery_return": values["recovery"],
            "actor_return": values["actor"],
            "recovery_advantage": values["recovery"] - values["actor"],
        })
    return result


def _horizon_gate(sequence_advantages, full_chain_rows, horizons, gate):
    by_state = defaultdict(dict)
    scene_h40 = defaultdict(list)
    for row in sequence_advantages:
        by_state[row["state_id"]][int(row["horizon"])] = float(
            row["recovery_advantage"]
        )
        if int(row["horizon"]) == 40:
            scene_h40[row["scene"]].append(float(row["recovery_advantage"]) > 0.0)
    if not by_state or any(set(values) != set(horizons) for values in by_state.values()):
        raise ValueError("L269 sequence horizon coverage is incomplete")
    h1 = np.asarray([values[1] for values in by_state.values()])
    h40 = np.asarray([values[40] for values in by_state.values()])
    first_positive = []
    for values in by_state.values():
        positive = [horizon for horizon in horizons if values[horizon] > 0.0]
        first_positive.append(None if not positive else min(positive))
    late_fraction = float(np.mean([
        value is not None
        and value >= int(gate["late_first_positive_minimum_horizon"])
        for value in first_positive
    ]))
    full_fraction = float(np.mean([
        float(row["recovery_advantage"]) > 0.0 for row in full_chain_rows
    ]))
    scene_fractions = {
        scene: float(np.mean(values)) for scene, values in sorted(scene_h40.items())
    }
    scenes_majority = sum(value >= 0.50 for value in scene_fractions.values())
    metrics = {
        "h1_positive_fraction": float(np.mean(h1 > 0.0)),
        "h40_positive_fraction": float(np.mean(h40 > 0.0)),
        "positive_fraction_gain_h1_to_h40": float(
            np.mean(h40 > 0.0) - np.mean(h1 > 0.0)
        ),
        "median_h1_recovery_advantage": float(np.median(h1)),
        "median_h40_recovery_advantage": float(np.median(h40)),
        "late_first_positive_fraction": late_fraction,
        "full_chain_positive_fraction": full_fraction,
        "scene_h40_positive_fractions": scene_fractions,
        "scenes_with_h40_majority_positive": scenes_majority,
        "first_positive_horizon_counts": dict(Counter(
            "never" if value is None else str(value) for value in first_positive
        )),
    }
    checks = {
        "h40_recovery_advantage_fraction": (
            metrics["h40_positive_fraction"]
            >= float(gate["minimum_h40_recovery_advantage_fraction"])
        ),
        "positive_fraction_gain_h1_to_h40": (
            metrics["positive_fraction_gain_h1_to_h40"]
            >= float(gate["minimum_positive_fraction_gain_h1_to_h40"])
        ),
        "median_advantage_grows": (
            metrics["median_h40_recovery_advantage"]
            > metrics["median_h1_recovery_advantage"]
        ),
        "late_first_positive_fraction": (
            metrics["late_first_positive_fraction"]
            >= float(gate["minimum_late_first_positive_fraction"])
        ),
        "full_chain_recovery_advantage_fraction": (
            metrics["full_chain_positive_fraction"]
            >= float(gate["minimum_full_chain_recovery_advantage_fraction"])
        ),
        "scene_coverage": (
            scenes_majority
            >= int(gate["minimum_scenes_with_h40_majority_positive"])
        ),
    }
    return checks, metrics


def run(config_path: Path, output: Path, device: str):
    config, checkpoint, inputs = _load_config(config_path)
    horizons = tuple(int(value) for value in config["horizon_diagnostic"]["horizons"])
    payload, agent, normalizer = _load_agent(checkpoint, device)
    if int(payload["agent"]["observation_dim"]) != 69:
        raise ValueError("L269 requires the frozen 69D source Actor")
    output.mkdir(parents=True, exist_ok=False)
    base = load_yaml(ROOT / "configs/rl/l262_coverage_gated_value_6k.yaml")

    candidate_rows = []
    l263_states = json.loads(
        (ROOT / config["l263"]["directory"] / "state_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    l263_by_scene = defaultdict(list)
    for state in l263_states:
        l263_by_scene[state["scene_config"]].append(state)
    for scene_config, states in sorted(l263_by_scene.items()):
        environment = _resolved_environment(
            base, scene_config, 50, states[0]["seed"], "l263"
        )
        try:
            for state in states:
                candidate_rows.extend(_candidate_rows(
                    "l263", state["state_id"], state, environment, agent, normalizer,
                    horizons, state["recovery_action"], state["actor_action"],
                    np.asarray(state["raw_observation"], dtype=np.float32),
                ))
                print(json.dumps({
                    "stage": "l263", "completed_states": len({
                        row["state_id"] for row in candidate_rows
                        if row["state_set"] == "l263"
                    }),
                }, sort_keys=True), flush=True)
        finally:
            environment.close()

    dataset = ROOT / config["l268"]["dataset"]
    dataset_manifest = json.loads(
        (dataset / "manifest.json").read_text(encoding="utf-8")
    )
    chain_by_id = {
        int(row["chain_id"]): row for row in dataset_manifest["chains"]
    }
    heldout_dir = ROOT / config["l268"]["heldout_diagnostic"]
    heldout_states = json.loads(
        (heldout_dir / "state_manifest.json").read_text(encoding="utf-8")
    )
    l268_by_scene = defaultdict(list)
    for state in heldout_states:
        l268_by_scene[state["scene_config"]].append(state)
    sequence_rows = []
    full_chain_rows = []
    for scene_config, states in sorted(l268_by_scene.items()):
        maximum = max(int(chain_by_id[int(state["chain_id"])]["steps"]) for state in states)
        environment = _resolved_environment(
            base, scene_config, maximum, states[0]["seed"], "l268"
        )
        try:
            for state in states:
                chain = chain_by_id[int(state["chain_id"])]
                shard = dataset / chain["student_npz"]
                with np.load(shard, allow_pickle=False) as student:
                    expected = student["observations"][0].copy()
                    recorded_actions = student["actions"].copy()
                    rewards = student["rewards"].reshape(-1).astype(np.float64)
                if len(rewards) < max(horizons):
                    raise ValueError("held-out recovery chain is shorter than H40")
                normalized = normalizer.normalize(expected)
                actor_action, _ = agent.select_action(normalized, deterministic=True)
                stored_actor = np.asarray(state.get("actor_action", actor_action))
                if stored_actor.shape == actor_action.shape and np.max(
                    np.abs(stored_actor - actor_action)
                ) > 1e-6:
                    raise RuntimeError("frozen source Actor action drifted")
                candidate_rows.extend(_candidate_rows(
                    "l268", state["state_id"], state, environment, agent, normalizer,
                    horizons, recorded_actions[0], actor_action, expected,
                ))
                full_horizon = int(len(rewards))
                actor_records = _rollout(
                    environment, np.asarray(state["initial_state"], dtype=np.float64),
                    int(state["seed"]), expected, actor_action, "actor_follow",
                    agent, normalizer, tuple(sorted(set(horizons + (full_horizon,)))),
                )
                gamma = float(environment.gamma)
                for horizon in horizons:
                    sequence_rows.append({
                        "state_id": state["state_id"],
                        "chain_id": int(state["chain_id"]),
                        "scene": state["scene"],
                        "scene_index": int(state["scene_index"]),
                        "split": state["split"],
                        "severity": state["severity"],
                        "initial_cte_m": float(state["initial_cte_m"]),
                        "horizon": int(horizon),
                        "recovery_return": _discounted_prefix(rewards, gamma, horizon),
                        "actor_return": float(actor_records[horizon]["discounted_return"]),
                        "recovery_advantage": (
                            _discounted_prefix(rewards, gamma, horizon)
                            - float(actor_records[horizon]["discounted_return"])
                        ),
                    })
                full_chain_rows.append({
                    "state_id": state["state_id"],
                    "chain_id": int(state["chain_id"]),
                    "scene": state["scene"],
                    "split": state["split"],
                    "severity": state["severity"],
                    "horizon": full_horizon,
                    "recovery_return": _discounted_prefix(rewards, gamma, full_horizon),
                    "actor_return": float(actor_records[full_horizon]["discounted_return"]),
                    "recovery_advantage": (
                        _discounted_prefix(rewards, gamma, full_horizon)
                        - float(actor_records[full_horizon]["discounted_return"])
                    ),
                })
                print(json.dumps({
                    "stage": "l268", "completed_states": len(full_chain_rows)
                }, sort_keys=True), flush=True)
        finally:
            environment.close()

    candidate_advantages = _advantage_rows(candidate_rows)
    checks, metrics = _horizon_gate(
        sequence_rows, full_chain_rows, horizons,
        config["horizon_diagnostic"]["gate"],
    )
    passed = all(checks.values())
    summary = {
        "protocol": "L269",
        "status": "horizon_gate_pass" if passed else "horizon_gate_fail",
        "git_sha": git_sha(ROOT),
        "device": device,
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "config_sha256": _sha256(config_path),
        "input_sha256": {str(path.relative_to(ROOT)): value for path, value in inputs.items()},
        "horizons": list(horizons),
        "l263_state_count": len(l263_states),
        "l268_state_count": len(heldout_states),
        "candidate_rollout_rows": len(candidate_rows),
        "sequence_advantage_rows": len(sequence_rows),
        "full_chain_rows": len(full_chain_rows),
        "checks": checks,
        "metrics": metrics,
        "horizon_gate_pass": passed,
        "critic_probe_authorized": passed,
        "decision": "start_critic_probe" if passed else "stop_without_critic_probe",
        "actor_updated": False,
    }
    _write_csv(output / "candidate_rollouts.csv", candidate_rows)
    _write_csv(output / "candidate_advantages.csv", candidate_advantages)
    _write_csv(output / "recovery_sequence_advantages.csv", sequence_rows)
    _write_csv(output / "full_chain_advantages.csv", full_chain_rows)
    _json_dump(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
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
