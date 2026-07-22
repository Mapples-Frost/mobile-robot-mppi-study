#!/usr/bin/env python3
"""Generate the frozen L284 closed-loop roll-in joint anchor dataset."""

from __future__ import annotations

import argparse
import json
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

from experiments.rl.run_l263_counterfactual_actor_diagnosis import (
    _load_agent,
    _prepare_reset,
)
from experiments.rl.run_l269_horizon_diagnostic import _resolved_environment
from experiments.rl.run_l276_recovery_sequence_imitation_initialization import (
    _load_chains,
)
from experiments.rl.run_l283_closed_loop_component_counterfactual import (
    _closed_loop_teacher_action,
    _new_teacher,
)
from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.rl.recovery_retention import (
    FIELDS,
    load_manifest,
    load_split,
    sha256_file,
    write_manifest,
    write_shard,
)


DEFAULT_CONFIG = ROOT / "configs/rl/l284_closed_loop_rollin_anchor_dataset.yaml"


def _episode(observation, action, kind, identifier):
    count = int(len(action))
    return {
        "observation": np.asarray(observation, dtype=np.float32),
        "teacher_action": np.asarray(action, dtype=np.float32),
        "source_kind": np.full(count, int(kind), dtype=np.int8),
        "episode_id": np.full(count, int(identifier), dtype=np.int64),
        "step": np.arange(count, dtype=np.int64),
    }


def _arrays(rows):
    return {
        name: np.concatenate([row[name] for row in rows], axis=0)
        for name in FIELDS
    }


def _split_to_arrays(split):
    return {
        "observation": split["observations"].copy(),
        "teacher_action": split["teacher_actions"].copy(),
        "source_kind": split["source_kinds"].copy(),
        "episode_id": split["episode_ids"].copy(),
        "step": split["steps"].copy(),
    }


def _load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["protocol"] != "L284" or config["schema_version"] != 1:
        raise ValueError("L284 roll-in dataset contract drifted")
    checks = {
        ROOT / config["l283_summary"]["path"]: config["l283_summary"]["sha256"],
        ROOT / config["source_anchor_dataset"] / "manifest.json": config[
            "source_anchor_manifest_sha256"
        ],
        ROOT / config["recovery_dataset"] / "manifest.json": config[
            "recovery_dataset_manifest_sha256"
        ],
        ROOT / config["teacher_config"]: config["teacher_config_sha256"],
    }
    for checkpoint in config["rollin_checkpoints"]:
        checks[ROOT / checkpoint["path"]] = checkpoint["sha256"]
    for artifact, expected in checks.items():
        if sha256_file(artifact) != expected:
            raise ValueError("L284 input SHA256 mismatch: %s" % artifact)
    l283 = json.loads(
        (ROOT / config["l283_summary"]["path"]).read_text(encoding="utf-8")
    )
    if l283["decision"] != "coupled_sequence_bottleneck":
        raise ValueError("L284 requires the frozen coupled L283 decision")
    references = "\n".join(str(path) for path in checks).lower()
    forbidden = [
        token for token in config["forbidden_tokens"] if token.lower() in references
    ]
    if forbidden:
        raise ValueError("forbidden input path entered L284")
    teacher = yaml.safe_load(
        (ROOT / config["teacher_config"]).read_text(encoding="utf-8")
    )["l267"]["collection"]["teacher"]
    return config, checks, teacher


def _rollin(environment, chain, agent, normalizer, teacher_config, tolerance):
    expected = np.asarray(chain["observations"][0], dtype=np.float32)
    observation, _, _ = _prepare_reset(
        environment,
        np.asarray(chain["initial_state"], dtype=np.float64),
        int(chain["seed"]),
    )
    reset_error = float(np.max(np.abs(observation - expected)))
    if reset_error > float(tolerance):
        raise RuntimeError("L284 roll-in reset drifted")
    teacher = _new_teacher(environment, chain, teacher_config)
    observations = []
    teacher_actions = []
    terminated = truncated = False
    agent.eval()
    for _ in range(len(chain["actions"])):
        if terminated or truncated:
            break
        teacher_action = _closed_loop_teacher_action(teacher, environment, chain)
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        teacher_actions.append(teacher_action.copy())
        actor_action, _ = agent.select_action(
            normalizer.normalize(observation), deterministic=True
        )
        observation, _, terminated, truncated, _ = environment.step(actor_action)
    if not observations:
        raise RuntimeError("L284 roll-in produced no state")
    observations = np.asarray(observations, dtype=np.float32)
    teacher_actions = np.asarray(teacher_actions, dtype=np.float32)
    if observations.shape[1:] != (69,) or teacher_actions.shape[1:] != (2,):
        raise RuntimeError("L284 roll-in shape mismatch")
    if not np.isfinite(observations).all() or not np.isfinite(teacher_actions).all():
        raise FloatingPointError("L284 roll-in contains non-finite values")
    return observations, teacher_actions, reset_error


def generate(config_path, output, device="cuda", smoke=False):
    config, checks, teacher_config = _load_config(config_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("L284 roll-in dataset already exists")
    recovery_root = ROOT / config["recovery_dataset"]
    recovery_manifest = json.loads(
        (recovery_root / "manifest.json").read_text(encoding="utf-8")
    )
    chains = _load_chains(recovery_root, recovery_manifest)
    split_chains = {
        name: [chain for chain in chains if chain["split"] == name]
        for name in ("train", "validation", "test")
    }
    if {name: len(rows) for name, rows in split_chains.items()} != config["split_counts"]:
        raise ValueError("L284 recovery split count mismatch")
    base = load_yaml(ROOT / config["base_config"])
    selected_chains = split_chains["train"][:1] if smoke else split_chains["train"]
    rollins = defaultdict(dict)
    max_reset_error = 0.0
    for checkpoint_index, checkpoint in enumerate(config["rollin_checkpoints"]):
        if smoke and checkpoint_index > 0:
            break
        _, agent, normalizer = _load_agent(ROOT / checkpoint["path"], device)
        by_scene = defaultdict(list)
        for chain in selected_chains:
            by_scene[chain["scene_config"]].append(chain)
        for scene_config, scene_chains in sorted(by_scene.items()):
            maximum = max(len(chain["actions"]) for chain in scene_chains)
            environment = _resolved_environment(
                base, scene_config, maximum, scene_chains[0]["seed"],
                config["reset_contract"],
            )
            try:
                for chain in scene_chains:
                    observations, actions, reset_error = _rollin(
                        environment, chain, agent, normalizer, teacher_config,
                        config["reset_tolerance"],
                    )
                    rollins[(int(checkpoint["seed"]), int(chain["chain_id"]))] = {
                        "observations": observations,
                        "actions": actions,
                    }
                    max_reset_error = max(max_reset_error, reset_error)
                    print(json.dumps({
                        "stage": "rollin",
                        "actor_seed": int(checkpoint["seed"]),
                        "chain_id": int(chain["chain_id"]),
                        "rollin_steps": int(len(actions)),
                        "device": device,
                    }, sort_keys=True), flush=True)
            finally:
                environment.close()
    if smoke:
        output.mkdir(parents=True, exist_ok=False)
        summary = {
            "protocol": "L284",
            "status": "smoke_complete",
            "rollins": len(rollins),
            "maximum_reset_observation_error": max_reset_error,
            "device": device,
        }
        (output / "smoke_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
        return summary

    source_root = ROOT / config["source_anchor_dataset"]
    source_manifest = load_manifest(source_root)
    source_train = load_split(source_root, "train")
    source_mask = source_train["source_kinds"] == 1
    source_rows = _episode(
        source_train["observations"][source_mask],
        source_train["teacher_actions"][source_mask],
        1,
        900000,
    )
    if len(source_rows["teacher_action"]) != int(config["train"]["source_samples"]):
        raise ValueError("L284 source-distillation count drifted")
    rng = np.random.RandomState(int(config["dataset_seed"]))
    recovery_rows = []
    checkpoints = sorted(config["rollin_checkpoints"], key=lambda row: int(row["seed"]))
    for chain in sorted(split_chains["train"], key=lambda row: int(row["chain_id"])):
        chain_id = int(chain["chain_id"])
        for actor_index, checkpoint in enumerate(checkpoints):
            trajectory = rollins[(int(checkpoint["seed"]), chain_id)]
            count = int(checkpoint["samples_per_chain"])
            indices = rng.randint(len(trajectory["actions"]), size=count)
            recovery_rows.append(_episode(
                trajectory["observations"][indices],
                trajectory["actions"][indices],
                0,
                500000 + actor_index * 1000 + chain_id,
            ))
    split_arrays = {"train": _arrays(recovery_rows + [source_rows])}
    split_arrays["validation"] = _split_to_arrays(load_split(source_root, "validation"))
    split_arrays["test"] = _split_to_arrays(load_split(source_root, "test"))
    if int(np.sum(split_arrays["train"]["source_kind"] == 0)) != int(
        config["train"]["recovery_samples"]
    ):
        raise ValueError("L284 recovery sample count mismatch")
    if int(np.sum(split_arrays["train"]["source_kind"] == 1)) != int(
        config["train"]["source_samples"]
    ):
        raise ValueError("L284 source sample count mismatch")
    output.mkdir(parents=True, exist_ok=False)
    descriptors = {}
    for split, arrays in split_arrays.items():
        filename = "%s.npz" % split
        descriptors[split] = {"file": filename, **write_shard(output / filename, arrays)}
    manifest = {
        "schema": config["schema"],
        "schema_version": int(config["schema_version"]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "action_mode": "direct_control",
        "observation_dim": 69,
        "action_dim": 2,
        "dataset_seed": int(config["dataset_seed"]),
        "source_checkpoint": source_manifest["source_checkpoint"],
        "source_checkpoint_sha256": source_manifest["source_checkpoint_sha256"],
        "recovery_manifest_sha256": config["recovery_dataset_manifest_sha256"],
        "student_fields": sorted(FIELDS),
        "split_chain_ids": source_manifest["split_chain_ids"],
        "splits": descriptors,
        "provenance": {
            "l283_summary_sha256": config["l283_summary"]["sha256"],
            "source_anchor_manifest_sha256": config["source_anchor_manifest_sha256"],
            "teacher_config_sha256": config["teacher_config_sha256"],
            "rollin_checkpoints": config["rollin_checkpoints"],
            "rollin_trajectory_count": len(rollins),
            "maximum_reset_observation_error": max_reset_error,
            "sampling": config["train"],
            "input_sha256": {str(path.relative_to(ROOT)): value for path, value in checks.items()},
        },
        "forbidden_artifacts_absent": True,
    }
    write_manifest(output, manifest)
    summary = {
        "stage": "complete",
        "device": device,
        "output": str(output),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "splits": descriptors,
        "rollin_trajectory_count": len(rollins),
        "maximum_reset_observation_error": max_reset_error,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output_dir or ROOT / config["output_dir"])
    generate(Path(args.config).resolve(), output, args.device, args.smoke)


if __name__ == "__main__":
    main()
