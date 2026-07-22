#!/usr/bin/env python3
"""Freeze held-out L268 recovery states, actions, and H40 returns."""

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
    _prepare_reset,
    _rollout,
)
from experiments.rl.train_rl_sampling_prior import _scene_configs
from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.rl.environment import DirectControlEnv


DEFAULT_CONFIG = ROOT / "configs/rl/l268_recovery_balanced_intervention.yaml"
DEFAULT_DATASET = ROOT / (
    "results/research_platform/rl/l268_recovery_balanced_intervention/"
    "recovery_dataset"
)
DEFAULT_OUTPUT = ROOT / (
    "results/research_platform/rl/l268_recovery_balanced_intervention/"
    "heldout_recovery_diagnostic"
)
FORMAL_CRITIC_OUTPUT = ROOT / (
    "results/research_platform/rl/l268_recovery_balanced_intervention/"
    "critic_only"
)
STUDENT_FIELDS = {
    "observations", "actions", "rewards", "constraint_costs",
    "next_observations", "dones", "groups", "chain_ids", "steps",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = dict(payload["l268"])
    checkpoint = ROOT / config["source_checkpoint"]
    digest = _sha256(checkpoint)
    if digest != config["source_checkpoint_sha256"]:
        raise ValueError("L268 source checkpoint SHA256 mismatch")
    if config["diagnostic"]["continuation"] != "actor_follow":
        raise ValueError("L268 held-out continuation must remain actor_follow")
    if int(config["diagnostic"]["horizon"]) != 40:
        raise ValueError("L268 held-out horizon must remain H40")
    return config, checkpoint


def _validate_dataset(dataset: Path, config):
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "L268" or manifest.get("status") != "complete":
        raise ValueError("held-out diagnostic requires the complete formal L268 dataset")
    if manifest.get("source_checkpoint_sha256") != config["source_checkpoint_sha256"]:
        raise ValueError("dataset source checkpoint provenance mismatch")
    chains = list(manifest.get("chains", ()))
    if len(chains) != 108 or manifest.get("accepted_chain_count") != 108:
        raise ValueError("formal L268 dataset must contain exactly 108 chains")
    if manifest.get("privileged_fields_in_student_shards") is not False:
        raise ValueError("privileged fields entered L268 student shards")
    if set(manifest.get("student_fields", ())) != STUDENT_FIELDS:
        raise ValueError("L268 student field manifest drifted")

    scene_counts = Counter(str(row["scene"]) for row in chains)
    split_counts = Counter(str(row["split"]) for row in chains)
    if set(scene_counts.values()) != {18} or len(scene_counts) != 6:
        raise ValueError("L268 scene quotas are not 18 x 6")
    if split_counts != {"train": 72, "validation": 18, "test": 18}:
        raise ValueError("L268 whole-chain split quotas drifted")
    identifiers = [(int(row["chain_id"]), str(row["attempt_id"])) for row in chains]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate L268 chain or attempt identifier")

    shard_rows = []
    all_finite = True
    maximum_cte_error = 0.0
    for row in chains:
        if not bool(row["accepted"]) or row["reason"] != "accepted":
            raise ValueError("non-accepted chain entered the formal manifest")
        if float(row["initial_cte_m"]) <= float(config["collection"]["start_cte_m"]):
            raise ValueError("recovery chain does not start off path")
        cte_error = abs(float(row["initial_cte_m"]) - float(row["target_cte_m"]))
        maximum_cte_error = max(maximum_cte_error, cte_error)
        if cte_error > float(config["collection"]["realized_cte_tolerance_m"]):
            raise ValueError("recovery chain violates realized CTE tolerance")
        if float(row["minimum_cte_m"]) >= float(config["collection"]["stable_cte_m"]):
            raise ValueError("recovery chain never re-entered the stable corridor")
        if int(row["stable_steps"]) < int(config["collection"]["stable_steps"]):
            raise ValueError("recovery chain did not hold the stable corridor")
        if float(row["final_progress_m"]) - float(row["reentry_progress_m"]) < (
            float(config["collection"]["minimum_post_reentry_progress_m"]) - 1e-9
        ):
            raise ValueError("recovery chain lacks post-reentry progress")

        shard = dataset / row["student_npz"]
        audit = dataset / "raw" / "audit" / (str(row["attempt_id"]) + ".csv")
        with np.load(shard, allow_pickle=False) as payload:
            if set(payload.files) != STUDENT_FIELDS:
                raise ValueError("student shard field whitelist violation")
            count = int(payload["actions"].shape[0])
            if payload["observations"].shape != (count, 69):
                raise ValueError("student shard observation shape is not Nx69")
            if payload["actions"].shape != (count, 2):
                raise ValueError("student shard action shape is not Nx2")
            finite = all(np.isfinite(payload[name]).all() for name in payload.files)
            all_finite = all_finite and finite
            if not finite:
                raise FloatingPointError("student shard contains NaN or Inf")
            if not np.all(payload["groups"] == int(row["scene_index"])):
                raise ValueError("student shard scene group mismatch")
            if not np.all(payload["chain_ids"] == int(row["chain_id"])):
                raise ValueError("student shard chain ID mismatch")
            if not np.array_equal(payload["steps"], np.arange(count)):
                raise ValueError("student shard step sequence drifted")
            if count != int(row["steps"]):
                raise ValueError("student shard transition count mismatch")
        with audit.open("r", newline="", encoding="utf-8") as handle:
            audit_rows = list(csv.DictReader(handle))
        if len(audit_rows) != int(row["steps"]):
            raise ValueError("privileged audit length mismatch")
        if any(
            item["finite"].lower() != "true"
            or item["collision"].lower() == "true"
            or item["boundary_violation"].lower() == "true"
            for item in audit_rows
        ):
            raise ValueError("accepted chain contains invalid privileged audit rows")
        shard_rows.append({
            "chain_id": int(row["chain_id"]),
            "attempt_id": row["attempt_id"],
            "scene": row["scene"],
            "split": row["split"],
            "transition_count": int(row["steps"]),
            "student_npz_sha256": _sha256(shard),
            "audit_csv_sha256": _sha256(audit),
        })

    fingerprint = hashlib.sha256()
    for row in sorted(shard_rows, key=lambda item: item["chain_id"]):
        fingerprint.update(row["student_npz_sha256"].encode("ascii"))
        fingerprint.update(row["audit_csv_sha256"].encode("ascii"))
    integrity = {
        "status": "pass",
        "protocol": "L268",
        "accepted_chain_count": len(chains),
        "accepted_by_scene": dict(sorted(scene_counts.items())),
        "accepted_by_split": dict(sorted(split_counts.items())),
        "failed_attempt_count": int(manifest["failed_attempt_count"]),
        "all_student_values_finite": all_finite,
        "maximum_realized_cte_error_m": maximum_cte_error,
        "student_fields": sorted(STUDENT_FIELDS),
        "student_shard_count": len(shard_rows),
        "dataset_content_fingerprint": fingerprint.hexdigest(),
        "manifest_sha256": _sha256(manifest_path),
        "shards": shard_rows,
    }
    return manifest, integrity


def _action_rows(actor_action, recorded_recovery_action):
    return (
        ("recorded_recovery", np.asarray(recorded_recovery_action, dtype=np.float32)),
        ("source_actor", np.asarray(actor_action, dtype=np.float32)),
        ("fast_forward", np.asarray((1.0, 0.0), dtype=np.float32)),
    )


def generate(config_path: Path, dataset: Path, output: Path, device: str):
    config, checkpoint = _load_config(config_path)
    if FORMAL_CRITIC_OUTPUT.exists():
        raise RuntimeError(
            "formal Critic output already exists; held-out returns must be frozen first"
        )
    manifest, dataset_integrity = _validate_dataset(dataset, config)
    payload, agent, normalizer = _load_agent(checkpoint, device)
    if int(payload["agent"]["observation_dim"]) != 69:
        raise ValueError("L268 held-out diagnostic requires the frozen 69D Actor")

    heldout = [
        row for row in manifest["chains"]
        if row["split"] in ("validation", "test")
    ]
    if len(heldout) != 36:
        raise ValueError("L268 held-out diagnostic requires 36 whole chains")
    output.mkdir(parents=True, exist_ok=False)
    _json_dump(output / "dataset_integrity.json", dataset_integrity)

    by_scene = defaultdict(list)
    for row in heldout:
        by_scene[int(row["scene_index"])].append(row)
    base = load_yaml(ROOT / "configs/rl/l262_coverage_gated_value_6k.yaml")
    state_rows = []
    return_rows = []
    maximum_reset_error = 0.0
    for scene_index in range(6):
        records = sorted(by_scene[scene_index], key=lambda row: int(row["chain_id"]))
        scene_relative = records[0]["scene_config"]
        resolved = _scene_configs(base, [scene_relative])[0]
        resolved = copy.deepcopy(resolved)
        resolved["experiment"]["max_steps"] = 50
        resolved["experiment"]["initial_state_noise"] = [0.0] * len(
            resolved["experiment"]["initial_state"]
        )
        resolved["rl"]["training"]["initial_state_curriculum"] = {"enabled": False}
        environment = DirectControlEnv(resolved, ROOT, seed=int(records[0]["seed"]))
        try:
            for record in records:
                shard = dataset / record["student_npz"]
                with np.load(shard, allow_pickle=False) as student:
                    expected_observation = student["observations"][0].copy()
                    recorded_action = student["actions"][0].copy()
                initial_state = np.asarray(record["initial_state"], dtype=np.float64)
                observation, _, path_state = _prepare_reset(
                    environment, initial_state, int(record["seed"])
                )
                reset_error = float(np.max(np.abs(observation - expected_observation)))
                maximum_reset_error = max(maximum_reset_error, reset_error)
                if reset_error > 1e-6:
                    raise RuntimeError("held-out recovery reset observation drifted")
                normalized = normalizer.normalize(observation)
                actor_action, _ = agent.select_action(normalized, deterministic=True)
                state_id = "chain_%03d" % int(record["chain_id"])
                state_rows.append({
                    "state_id": state_id,
                    "chain_id": int(record["chain_id"]),
                    "attempt_id": record["attempt_id"],
                    "scene": record["scene"],
                    "scene_index": scene_index,
                    "scene_config": scene_relative,
                    "split": record["split"],
                    "severity": record["severity"],
                    "side": int(record["side"]),
                    "heading_class": record["heading_class"],
                    "seed": int(record["seed"]),
                    "initial_cte_m": float(record["initial_cte_m"]),
                    "initial_state": initial_state.tolist(),
                    "raw_observation": observation.tolist(),
                    "normalized_observation": normalized.tolist(),
                    "path_state": path_state,
                    "reset_observation_error": reset_error,
                })
                for action_id, action in _action_rows(actor_action, recorded_action):
                    values = _rollout(
                        environment, initial_state, int(record["seed"]),
                        expected_observation, action,
                        config["diagnostic"]["continuation"], agent, normalizer,
                        (int(config["diagnostic"]["horizon"]),),
                    )[int(config["diagnostic"]["horizon"])]
                    return_rows.append({
                        "state_id": state_id,
                        "chain_id": int(record["chain_id"]),
                        "scene": record["scene"],
                        "scene_index": scene_index,
                        "split": record["split"],
                        "severity": record["severity"],
                        "side": int(record["side"]),
                        "action_id": action_id,
                        "normalized_v": float(action[0]),
                        "normalized_omega": float(action[1]),
                        "continuation": config["diagnostic"]["continuation"],
                        **values,
                    })
                print(json.dumps({
                    "protocol": "L268",
                    "completed_states": len(state_rows),
                    "return_rows": len(return_rows),
                }, sort_keys=True), flush=True)
        finally:
            environment.close()

    recovery = {
        row["state_id"]: float(row["discounted_return"])
        for row in return_rows if row["action_id"] == "recorded_recovery"
    }
    forward = {
        row["state_id"]: float(row["discounted_return"])
        for row in return_rows if row["action_id"] == "fast_forward"
    }
    summary = {
        "protocol": "L268",
        "status": "heldout_recovery_diagnostic_complete",
        "git_sha": git_sha(ROOT),
        "config_sha256": _sha256(config_path),
        "dataset_manifest_sha256": dataset_integrity["manifest_sha256"],
        "dataset_content_fingerprint": dataset_integrity["dataset_content_fingerprint"],
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "device": device,
        "state_count": len(state_rows),
        "action_return_rows": len(return_rows),
        "split_counts": dict(Counter(row["split"] for row in state_rows)),
        "severity_counts": dict(Counter(row["severity"] for row in state_rows)),
        "scene_counts": dict(Counter(row["scene"] for row in state_rows)),
        "continuation": config["diagnostic"]["continuation"],
        "horizon": int(config["diagnostic"]["horizon"]),
        "maximum_reset_observation_error": maximum_reset_error,
        "true_recovery_better_than_forward_count": sum(
            recovery[state_id] > forward[state_id] for state_id in recovery
        ),
        "true_recovery_forward_tie_count": sum(
            abs(recovery[state_id] - forward[state_id]) <= 1e-12
            for state_id in recovery
        ),
        "formal_critic_results_read_before_freeze": False,
    }
    _json_dump(output / "state_manifest.json", state_rows)
    _write_csv(output / "action_returns_h40.csv", return_rows)
    _json_dump(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    summary = generate(
        args.config.resolve(), args.dataset.resolve(),
        args.output_dir.resolve(), args.device,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
