#!/usr/bin/env python3
"""Finalize an already-collected L243 dataset without rerunning MuJoCo."""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.rl.demonstrations import (
    DEMONSTRATION_SCHEMA,
    DEMONSTRATION_SCHEMA_VERSION,
    DEMONSTRATION_SPLITS,
    sha256_file,
    split_episode_seeds,
    write_demonstration_manifest,
)
from mobile_robot_mppi.rl.environment import DirectControlEnv


DEFAULT_CONFIG = "configs/rl/direct_control_bc_teacher_l243.yaml"


def _true(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def finalize_dataset(config_path=DEFAULT_CONFIG, dataset_dir=None):
    config = load_yaml(ROOT / config_path)
    collection = dict(config["collection"])
    output = ROOT / str(dataset_dir or collection["output_dir"])
    if (output / "manifest.json").exists():
        raise FileExistsError("refusing to overwrite existing manifest")
    seeds = [int(value) for value in collection["seeds"]]
    scene_configs = [str(value) for value in collection["scene_configs"]]
    split_plan = split_episode_seeds(
        seeds,
        validation_fraction=float(collection["validation_fraction"]),
        test_fraction=float(collection["test_fraction"]),
        seed=int(collection["split_seed"]),
    )

    with (output / "audit" / "episodes.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    expected_pairs = {
        (scene_index, seed)
        for scene_index in range(len(scene_configs))
        for seed in seeds
    }
    actual_pairs = {
        (int(row["episode_id"]) // len(seeds), int(row["seed"]))
        for row in episodes
    }
    if len(episodes) != 90 or actual_pairs != expected_pairs:
        raise ValueError("existing audit is not the frozen six-scene x fifteen-seed matrix")
    if len(list((output / "audit" / "trajectories").glob("episode_*.csv"))) != 90:
        raise ValueError("existing audit does not contain exactly 90 trajectories")
    if len(list((output / "audit" / "resolved_configs").glob("episode_*.json"))) != 90:
        raise ValueError("existing audit does not contain exactly 90 resolved configs")

    first_config_path = output / "audit" / "resolved_configs" / "episode_000000.json"
    with first_config_path.open(encoding="utf-8") as handle:
        first_config = json.load(handle)
    environment = DirectControlEnv(first_config, ROOT, seed=seeds[0])
    try:
        observation_dim = int(environment.observation_dim)
        observation_config = environment.encoder.config.to_dict()
        prior_config = environment.parameterization.config.to_dict()
    finally:
        environment.close()
    if observation_dim != 69 or observation_config.get("include_absolute_pose"):
        raise ValueError("student observation contract is not the frozen 69D non-absolute form")

    successful = [row for row in episodes if _true(row["included_in_training_shard"])]
    split_descriptors = {}
    for split in DEMONSTRATION_SPLITS:
        shard_path = output / "splits" / (split + ".npz")
        with np.load(shard_path, allow_pickle=False) as shard:
            sample_count = int(shard["observation"].shape[0])
            episode_ids = set(int(value) for value in shard["episode_id"].tolist())
            if shard["observation"].shape[1:] != (69,):
                raise ValueError("student shard observation dimension changed")
            if shard["teacher_action"].shape != (sample_count, 2):
                raise ValueError("student shard action dimension changed")
        expected_ids = {
            int(row["episode_id"]) for row in successful if row["split"] == split
        }
        if episode_ids != expected_ids:
            raise ValueError("student shard episode ids disagree with audit")
        split_descriptors[split] = {
            "file": str(shard_path.relative_to(output).as_posix()),
            "sha256": sha256_file(shard_path),
            "bytes": int(shard_path.stat().st_size),
            "samples": sample_count,
            "episodes": len(episode_ids),
            "seeds": sorted({
                int(row["seed"]) for row in successful if row["split"] == split
            }),
        }

    sensor_conditions = {}
    for scene_index in range(len(scene_configs)):
        row = episodes[scene_index * len(seeds)]
        sensor_conditions[row["scene"]] = {
            "pose_source": row["pose_source"],
            "twist_source": row["twist_source"],
        }
    collection_sha = str(collection["collection_git_sha"])
    manifest = {
        "schema": DEMONSTRATION_SCHEMA,
        "schema_version": DEMONSTRATION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": collection_sha,
        "config": {
            "rl_config": str(config_path),
            "scene_configs": scene_configs,
            "lookahead": float(collection["lookahead"]),
            "teacher_action_mode": "direct_control",
            "teacher_cruise_speed": float(collection["cruise_speed"]),
            "teacher_yaw_gain": float(collection["yaw_gain"]),
            "teacher_pose_source": "ground_truth",
            "route_source": "task_points",
            "num_samples": int(collection["num_samples"]),
            "max_steps": int(collection["max_steps"]),
            "split_seed": int(collection["split_seed"]),
            "validation_fraction": float(collection["validation_fraction"]),
            "test_fraction": float(collection["test_fraction"]),
            "sensor_conditions": sensor_conditions,
            "manifest_finalization_git_sha": git_sha(ROOT),
        },
        "observation_dim": observation_dim,
        "action_dim": 2,
        "action_mode": "direct_control",
        "observation_encoder": observation_config,
        "prior_parameterization": prior_config,
        "teacher": {
            "class": "ScriptedPolylineDirectControl",
            "action_space": "normalized_direct_control_v_omega",
            "student_observation_source": "DirectControlEnv.reset_and_step",
            "privileged_route_training_only": True,
            "teacher_pose_source": "ground_truth",
            "privileged_pose_training_only": True,
        },
        "split_plan": {
            name: [int(value) for value in split_plan[name]]
            for name in DEMONSTRATION_SPLITS
        },
        "splits": split_descriptors,
        "counts": {
            "requested_episodes": len(episodes),
            "successful_episodes": len(successful),
            "failed_episodes": len(episodes) - len(successful),
            "training_samples": sum(
                value["samples"] for value in split_descriptors.values()
            ),
            "failed_episodes_in_training_shards": 0,
        },
        "audit": {
            "directory": "audit",
            "included_in_training_shards": False,
            "contains_privileged_route": True,
            "contains_simulator_truth": True,
            "contains_teacher_waypoints": True,
            "episodes_csv": "audit/episodes.csv",
        },
    }
    write_demonstration_manifest(output, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir")
    args = parser.parse_args(argv)
    manifest = finalize_dataset(args.config, args.dataset_dir)
    print(json.dumps({"counts": manifest["counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
