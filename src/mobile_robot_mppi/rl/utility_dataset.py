"""Loading and split contracts for counterfactual utility regression."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .utility_model import effect_group_counts, group_keys


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_counterfactual_utility_dataset(
    dataset_dir,
    model_selection_episode_seeds,
    calibration_episode_seeds,
):
    root = Path(dataset_dir).resolve()
    npz_path = root / "samples_all_training_seeds.npz"
    audit_path = root / "audit.json"
    if not npz_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("utility dataset requires merged NPZ and audit JSON")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool(audit.get("quality_passed", False)):
        raise ValueError("counterfactual dataset quality audit did not pass")
    data = np.load(npz_path)
    required = (
        "features",
        "training_seed",
        "episode_seed",
        "branch_step",
        "split",
        "goal_distance_improvement",
    )
    if any(name not in data.files for name in required):
        raise ValueError("utility dataset NPZ schema is incomplete")
    count = int(data["features"].shape[0])
    if count == 0 or any(data[name].shape[0] != count for name in required):
        raise ValueError("utility dataset arrays are empty or misaligned")
    features = np.asarray(data["features"], dtype=np.float32)
    targets = np.asarray(
        data["goal_distance_improvement"], dtype=np.float64
    ).reshape(-1)
    training_seed = np.asarray(data["training_seed"], dtype=np.int64)
    episode_seed = np.asarray(data["episode_seed"], dtype=np.int64)
    original_split = np.asarray(data["split"], dtype=np.int8)
    branch_step = np.asarray(data["branch_step"], dtype=np.int64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("utility features are invalid")
    if not np.isfinite(targets).all():
        raise ValueError("utility targets contain NaN or Inf")
    selection_seeds = set(int(value) for value in model_selection_episode_seeds)
    calibration_seeds = set(int(value) for value in calibration_episode_seeds)
    if (
        not selection_seeds
        or not calibration_seeds
        or selection_seeds.intersection(calibration_seeds)
    ):
        raise ValueError("selection/calibration episode seeds must be disjoint")
    train_mask = original_split == 0
    validation_mask = original_split == 1
    selection_mask = np.asarray(
        [int(value) in selection_seeds for value in episode_seed]
    )
    calibration_mask = np.asarray(
        [int(value) in calibration_seeds for value in episode_seed]
    )
    if np.any((selection_mask | calibration_mask) & ~validation_mask):
        raise ValueError("selection/calibration seeds leaked into train split")
    if np.any(validation_mask & ~(selection_mask | calibration_mask)):
        raise ValueError("validation rows are not assigned to selection/calibration")
    masks = {
        "train": train_mask,
        "selection": selection_mask,
        "calibration": calibration_mask,
    }
    groups = group_keys(training_seed, episode_seed)
    splits = {}
    for name, mask in masks.items():
        if not np.any(mask):
            raise ValueError("utility %s split is empty" % name)
        splits[name] = {
            "features": features[mask].copy(),
            "targets": targets[mask].copy(),
            "groups": groups[mask].copy(),
            "training_seed": training_seed[mask].copy(),
            "episode_seed": episode_seed[mask].copy(),
            "branch_step": branch_step[mask].copy(),
        }
    train_episodes = set(int(value) for value in splits["train"]["episode_seed"])
    if train_episodes.intersection(selection_seeds | calibration_seeds):
        raise ValueError("train episode seeds overlap development holdouts")
    return {
        "splits": splits,
        "feature_dim": int(features.shape[1]),
        "feature_schema": audit["contract"]["feature_schema"],
        "dataset_contract": audit["contract"],
        "dataset_sha256": _sha256(npz_path),
        "audit_sha256": _sha256(audit_path),
        "source": str(root),
    }


def utility_dataset_summary(dataset, meaningful_effect=0.03):
    result = {
        "feature_dim": int(dataset["feature_dim"]),
        "source": dataset["source"],
        "dataset_sha256": dataset["dataset_sha256"],
        "audit_sha256": dataset["audit_sha256"],
        "splits": {},
    }
    for name, split in dataset["splits"].items():
        targets = split["targets"]
        result["splits"][name] = {
            "rows": int(targets.size),
            "target_min": float(np.min(targets)),
            "target_median": float(np.median(targets)),
            "target_max": float(np.max(targets)),
            **effect_group_counts(
                targets, split["groups"], threshold=meaningful_effect
            ),
            "training_seeds": sorted(set(
                int(value) for value in split["training_seed"]
            )),
            "episode_seeds": sorted(set(
                int(value) for value in split["episode_seed"]
            )),
        }
    return result


__all__ = [
    "load_counterfactual_utility_dataset",
    "utility_dataset_summary",
]
