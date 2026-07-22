"""Fail-closed dataset contract for the L279 recovery-retention Actor anchor."""

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "mobile_robot_mppi.recovery_retention_anchor"
SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")
FIELDS = frozenset((
    "observation", "teacher_action", "source_kind", "episode_id", "step",
))


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_fingerprint(manifest):
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_arrays(arrays, observation_dim=69, action_dim=2, allow_empty=False):
    if set(arrays) != FIELDS:
        raise ValueError("recovery-retention shard fields do not match allowlist")
    observation = np.asarray(arrays["observation"])
    action = np.asarray(arrays["teacher_action"])
    kind = np.asarray(arrays["source_kind"])
    episode = np.asarray(arrays["episode_id"])
    step = np.asarray(arrays["step"])
    count = int(observation.shape[0]) if observation.ndim == 2 else -1
    if observation.ndim != 2 or observation.shape[1] != int(observation_dim):
        raise ValueError("recovery-retention observation shape is invalid")
    if action.shape != (count, int(action_dim)):
        raise ValueError("recovery-retention action shape is invalid")
    if any(value.shape != (count,) for value in (kind, episode, step)):
        raise ValueError("recovery-retention identifier shape is invalid")
    if not allow_empty and count <= 0:
        raise ValueError("recovery-retention shard cannot be empty")
    if not np.issubdtype(observation.dtype, np.floating) or not np.issubdtype(
        action.dtype, np.floating
    ):
        raise ValueError("recovery-retention observations/actions must be floating")
    if any(not np.issubdtype(value.dtype, np.integer) for value in (kind, episode, step)):
        raise ValueError("recovery-retention identifiers must be integers")
    if not np.isfinite(observation).all() or not np.isfinite(action).all():
        raise ValueError("recovery-retention shard contains non-finite values")
    if np.any(action < -1.000001) or np.any(action > 1.000001):
        raise ValueError("recovery-retention actions must lie in [-1, 1]")
    if not set(np.unique(kind)).issubset({0, 1}):
        raise ValueError("recovery-retention source_kind is invalid")
    if np.any(episode < 0) or np.any(step < 0):
        raise ValueError("recovery-retention identifiers must be non-negative")
    for identifier in np.unique(episode):
        actual = np.sort(step[episode == identifier].astype(np.int64))
        if not np.array_equal(actual, np.arange(actual.size, dtype=np.int64)):
            raise ValueError("recovery-retention episode steps must be contiguous")
    return {
        "samples": count,
        "episodes": int(np.unique(episode).size),
        "recovery_samples": int(np.sum(kind == 0)),
        "source_samples": int(np.sum(kind == 1)),
    }


def write_shard(path, arrays):
    summary = validate_arrays(arrays)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        observation=np.asarray(arrays["observation"], dtype=np.float32),
        teacher_action=np.asarray(arrays["teacher_action"], dtype=np.float32),
        source_kind=np.asarray(arrays["source_kind"], dtype=np.int8),
        episode_id=np.asarray(arrays["episode_id"], dtype=np.int64),
        step=np.asarray(arrays["step"], dtype=np.int64),
    )
    return {**summary, "bytes": int(target.stat().st_size), "sha256": sha256_file(target)}


def _safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / str(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("recovery-retention shard path escapes dataset")
    return path


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("recovery-retention manifest must be an object")
    required = {
        "schema", "schema_version", "created_utc", "git_sha", "action_mode",
        "observation_dim", "action_dim", "dataset_seed", "source_checkpoint",
        "source_checkpoint_sha256", "recovery_manifest_sha256", "student_fields",
        "split_chain_ids", "splits", "provenance", "forbidden_artifacts_absent",
    }
    if set(manifest) != required:
        raise ValueError("recovery-retention manifest fields are invalid")
    if manifest["schema"] != SCHEMA or int(manifest["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("unsupported recovery-retention schema")
    if manifest["action_mode"] != "direct_control":
        raise ValueError("recovery-retention action mode must be direct_control")
    if int(manifest["observation_dim"]) != 69 or int(manifest["action_dim"]) != 2:
        raise ValueError("recovery-retention dimensions are invalid")
    if set(manifest["student_fields"]) != FIELDS:
        raise ValueError("recovery-retention public fields are invalid")
    if set(manifest["splits"]) != set(SPLITS) or set(manifest["split_chain_ids"]) != set(SPLITS):
        raise ValueError("recovery-retention splits are incomplete")
    chain_sets = [set(int(x) for x in manifest["split_chain_ids"][name]) for name in SPLITS]
    if any(chain_sets[i].intersection(chain_sets[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("recovery-retention chain leakage across splits")
    for name in SPLITS:
        descriptor = manifest["splits"][name]
        if set(descriptor) != {
            "file", "sha256", "bytes", "samples", "episodes",
            "recovery_samples", "source_samples",
        }:
            raise ValueError("recovery-retention split descriptor is invalid")
        if len(str(descriptor["sha256"])) != 64:
            raise ValueError("recovery-retention checksum is invalid")
    if manifest["forbidden_artifacts_absent"] is not True:
        raise ValueError("forbidden artifacts entered recovery-retention dataset")
    return manifest


def write_manifest(dataset_dir, manifest):
    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    validated = validate_manifest(dict(manifest))
    path = root / "manifest.json"
    path.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifest(dataset_dir):
    path = Path(dataset_dir) / "manifest.json"
    return validate_manifest(json.loads(path.read_text(encoding="utf-8")))


def load_split(dataset_dir, split):
    if split not in SPLITS:
        raise ValueError("recovery-retention split is invalid")
    manifest = load_manifest(dataset_dir)
    descriptor = manifest["splits"][split]
    path = _safe_path(dataset_dir, descriptor["file"])
    if sha256_file(path) != descriptor["sha256"] or int(path.stat().st_size) != int(descriptor["bytes"]):
        raise ValueError("recovery-retention shard provenance mismatch")
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != FIELDS:
            raise ValueError("recovery-retention shard fields do not match allowlist")
        arrays = {name: np.asarray(loaded[name]).copy() for name in FIELDS}
    summary = validate_arrays(arrays, allow_empty=True)
    for key in ("samples", "episodes", "recovery_samples", "source_samples"):
        if int(summary[key]) != int(descriptor[key]):
            raise ValueError("recovery-retention shard count mismatch")
    return {
        "observations": arrays["observation"].astype(np.float32, copy=False),
        "teacher_actions": arrays["teacher_action"].astype(np.float32, copy=False),
        "source_kinds": arrays["source_kind"].astype(np.int8, copy=False),
        "episode_ids": arrays["episode_id"].astype(np.int64, copy=False),
        "steps": arrays["step"].astype(np.int64, copy=False),
    }
