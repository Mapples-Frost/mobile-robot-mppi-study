import json
from pathlib import Path

import numpy as np
import pytest

from mobile_robot_mppi.rl.recovery_retention import (
    FIELDS, load_split, manifest_fingerprint, write_manifest, write_shard,
)
from mobile_robot_mppi.rl.trainer import BehaviorCloningAnchor


def _arrays(kind=0, episode_id=1, count=4):
    return {
        "observation": np.zeros((count, 69), dtype=np.float32),
        "teacher_action": np.zeros((count, 2), dtype=np.float32),
        "source_kind": np.full(count, kind, dtype=np.int8),
        "episode_id": np.full(count, episode_id, dtype=np.int64),
        "step": np.arange(count, dtype=np.int64),
    }


def _dataset(tmp_path):
    splits = {}
    for split, kind, episode in (
        ("train", 0, 1), ("validation", 0, 2), ("test", 0, 3)
    ):
        file = "%s.npz" % split
        splits[split] = {"file": file, **write_shard(tmp_path / file, _arrays(kind, episode))}
    manifest = {
        "schema": "mobile_robot_mppi.recovery_retention_anchor",
        "schema_version": 1,
        "created_utc": "2026-07-23T00:00:00+00:00",
        "git_sha": "a" * 40,
        "action_mode": "direct_control",
        "observation_dim": 69,
        "action_dim": 2,
        "dataset_seed": 1,
        "source_checkpoint": "source.pt",
        "source_checkpoint_sha256": "a" * 64,
        "recovery_manifest_sha256": "b" * 64,
        "student_fields": sorted(FIELDS),
        "split_chain_ids": {"train": [1], "validation": [2], "test": [3]},
        "splits": splits,
        "provenance": {"source_replay_size": 4},
        "forbidden_artifacts_absent": True,
    }
    write_manifest(tmp_path, manifest)
    return manifest


def test_l279_custom_dataset_loads_only_approved_fields(tmp_path):
    manifest = _dataset(tmp_path)
    arrays = load_split(tmp_path, "train")
    assert arrays["observations"].shape == (4, 69)
    assert arrays["teacher_actions"].shape == (4, 2)
    assert len(manifest_fingerprint(manifest)) == 64


def test_l279_anchor_accepts_custom_format(tmp_path):
    manifest = _dataset(tmp_path)
    anchor = BehaviorCloningAnchor({
        "enabled": True,
        "dataset_dir": str(tmp_path),
        "dataset_format": "recovery_retention_v1",
        "batch_size": 2,
        "mean_weight": 2.0,
    }, tmp_path, 69, 2, seed=7, action_mode="direct_control")
    assert anchor.metadata()["dataset_format"] == "recovery_retention_v1"
    assert anchor.manifest_fingerprint == manifest_fingerprint(manifest)


def test_l279_manifest_rejects_split_leakage(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["split_chain_ids"]["test"] = [1]
    with pytest.raises(ValueError, match="leakage"):
        write_manifest(tmp_path / "bad", manifest)


def test_l279_loader_rejects_shard_tampering(tmp_path):
    _dataset(tmp_path)
    path = tmp_path / "train.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="provenance"):
        load_split(tmp_path, "train")
