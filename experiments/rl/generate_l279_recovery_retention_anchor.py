#!/usr/bin/env python3
"""Materialize the preregistered leakage-safe L279 retention anchor."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rl.run_l263_counterfactual_actor_diagnosis import _load_agent
from experiments.rl.run_l276_recovery_sequence_imitation_initialization import (
    _load_chains, _predict_actions,
)
from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.rl.recovery_retention import (
    FIELDS, sha256_file, write_manifest, write_shard,
)


DEFAULT_CONFIG = ROOT / "configs/rl/l279_recovery_retention_anchor_dataset.yaml"


def _arrays(rows):
    return {
        name: np.concatenate([row[name] for row in rows], axis=0)
        for name in FIELDS
    }


def _episode(observation, action, kind, identifier):
    count = int(len(action))
    return {
        "observation": np.asarray(observation, dtype=np.float32),
        "teacher_action": np.asarray(action, dtype=np.float32),
        "source_kind": np.full(count, int(kind), dtype=np.int8),
        "episode_id": np.full(count, int(identifier), dtype=np.int64),
        "step": np.arange(count, dtype=np.int64),
    }


def generate(config_path, output, device="cuda"):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config["schema_version"] != 1 or config["schema"] != "mobile_robot_mppi.recovery_retention_anchor":
        raise ValueError("L279 dataset schema is not frozen")
    source = ROOT / config["source_checkpoint"]
    recovery_root = ROOT / config["recovery_dataset"]
    recovery_manifest_path = recovery_root / "manifest.json"
    if sha256_file(source) != config["source_checkpoint_sha256"]:
        raise ValueError("L279 source checkpoint SHA256 mismatch")
    if sha256_file(recovery_manifest_path) != config["recovery_dataset_manifest_sha256"]:
        raise ValueError("L279 recovery manifest SHA256 mismatch")
    references = "\n".join((
        str(config["source_checkpoint"]), str(config["recovery_dataset"]),
    )).lower()
    entered = [token for token in config["forbidden_tokens"] if token.lower() in references]
    if entered:
        raise ValueError("forbidden artifact entered L279 dataset config")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("L279 anchor dataset already exists")
    recovery_manifest = json.loads(recovery_manifest_path.read_text(encoding="utf-8"))
    chains = _load_chains(recovery_root, recovery_manifest)
    split_chains = {
        name: [row for row in chains if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    if {name: len(rows) for name, rows in split_chains.items()} != config["split_counts"]:
        raise ValueError("L279 recovery split count mismatch")
    payload, source_agent, normalizer = _load_agent(source, device)
    replay = payload["replay_buffer"]
    replay_size = int(replay["size"])
    if replay_size != 6000:
        raise ValueError("L279 requires frozen 6000-transition source replay")
    source_observations = np.asarray(
        replay["observations"][:replay_size], dtype=np.float32
    )
    source_actions = _predict_actions(
        source_agent, normalizer, source_observations
    )
    rng = np.random.RandomState(int(config["dataset_seed"]))
    per_chain = int(config["train"]["samples_per_recovery_chain"])
    train_rows = []
    source_hashes = {}
    for chain in sorted(
        split_chains["train"], key=lambda row: int(row["chain_id"])
    ):
        chain_id = int(chain["chain_id"])
        indices = rng.randint(len(chain["actions"]), size=per_chain)
        train_rows.append(_episode(
            chain["observations"][indices], chain["actions"][indices],
            0, 100000 + chain_id,
        ))
        source_indices = rng.randint(replay_size, size=per_chain)
        train_rows.append(_episode(
            source_observations[source_indices], source_actions[source_indices],
            1, 200000 + chain_id,
        ))
        source_hashes[str(chain_id)] = sha256_file(
            recovery_root / chain["student_npz"]
        )
    split_arrays = {"train": _arrays(train_rows)}
    for split, prefix in (("validation", 300000), ("test", 400000)):
        rows = []
        for chain in sorted(
            split_chains[split], key=lambda row: int(row["chain_id"])
        ):
            chain_id = int(chain["chain_id"])
            rows.append(_episode(
                chain["observations"], chain["actions"], 0,
                prefix + chain_id,
            ))
            source_hashes[str(chain_id)] = sha256_file(
                recovery_root / chain["student_npz"]
            )
        split_arrays[split] = _arrays(rows)
    if int(np.sum(split_arrays["train"]["source_kind"] == 0)) != int(
        config["train"]["recovery_samples"]
    ):
        raise ValueError("L279 frozen recovery sample count mismatch")
    if int(np.sum(split_arrays["train"]["source_kind"] == 1)) != int(
        config["train"]["source_samples"]
    ):
        raise ValueError("L279 frozen source sample count mismatch")
    output.mkdir(parents=True, exist_ok=False)
    descriptors = {}
    for split, arrays in split_arrays.items():
        filename = "%s.npz" % split
        descriptors[split] = {
            "file": filename, **write_shard(output / filename, arrays)
        }
    manifest = {
        "schema": config["schema"],
        "schema_version": int(config["schema_version"]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "action_mode": "direct_control",
        "observation_dim": 69,
        "action_dim": 2,
        "dataset_seed": int(config["dataset_seed"]),
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "recovery_manifest_sha256": config["recovery_dataset_manifest_sha256"],
        "student_fields": sorted(FIELDS),
        "split_chain_ids": {
            name: [int(row["chain_id"]) for row in split_chains[name]]
            for name in split_chains
        },
        "splits": descriptors,
        "provenance": {
            "source_replay_size": replay_size,
            "recovery_chain_shard_sha256": source_hashes,
            "sampling": config["train"],
        },
        "forbidden_artifacts_absent": True,
    }
    write_manifest(output, manifest)
    print(json.dumps({
        "stage": "complete", "device": device, "output": str(output),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "splits": descriptors,
    }, sort_keys=True), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output_dir or ROOT / config["output_dir"])
    generate(Path(args.config).resolve(), output, args.device)


if __name__ == "__main__":
    main()
