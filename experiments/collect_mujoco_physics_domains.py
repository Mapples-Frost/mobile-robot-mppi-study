#!/usr/bin/env python3
"""Collect seen and held-out MuJoCo physics domains without episode leakage."""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.collect_mujoco_residual_data import collect
from mobile_robot_mppi.core.config import config_hash, deep_merge, git_sha, load_yaml
from mobile_robot_mppi.learning.dataset_quality import (
    assert_episode_disjoint_splits,
    assert_residual_dataset_quality,
)
from src.learning.residual_dataset import REQUIRED_FIELDS, ResidualDataset


def combine_datasets(datasets, metadata):
    if not datasets:
        raise ValueError("at least one physics domain is required")
    arrays = {
        field: np.concatenate([getattr(dataset, field) for dataset in datasets], axis=0)
        for field in REQUIRED_FIELDS
    }
    return ResidualDataset.from_mapping(arrays, metadata=metadata)


@dataclass(frozen=True)
class DomainSplits:
    train: ResidualDataset
    validation: ResidualDataset
    test: ResidualDataset
    unseen: ResidualDataset


def stratified_domain_split(dataset, domains, validation_fraction, test_fraction, seed):
    groups = {"train": [], "validation": [], "test": [], "unseen": []}
    for domain_index, entry in enumerate(domains):
        name = str(entry["name"])
        indices = np.flatnonzero(dataset.disturbance_type == name)
        domain_dataset = dataset.subset(indices, metadata_updates={"domain": name})
        if str(entry.get("role", "seen")) == "unseen":
            groups["unseen"].append(domain_dataset)
            continue
        local = domain_dataset.split(
            validation_fraction=float(validation_fraction),
            test_fraction=float(test_fraction),
            seed=int(seed) + domain_index,
        )
        for split_name in ("train", "validation", "test"):
            subset = getattr(local, split_name)
            if len(subset) == 0:
                raise ValueError(
                    "domain %s has too few episodes for non-empty %s split"
                    % (name, split_name)
                )
            groups[split_name].append(subset)
    for split_name, values in groups.items():
        if not values:
            raise ValueError("physics suite produced an empty %s split" % split_name)
    return DomainSplits(**{
        name: combine_datasets(values, {"split": name, "stratified_by": "physics_domain"})
        for name, values in groups.items()
    })


def validate_domains(entries):
    names = [str(entry.get("name", "")) for entry in entries]
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("physics domain names must be non-empty and unique")
    roles = [str(entry.get("role", "seen")) for entry in entries]
    if any(role not in ("seen", "unseen") for role in roles):
        raise ValueError("physics domain role must be seen or unseen")
    if "seen" not in roles or "unseen" not in roles:
        raise ValueError("physics suite requires both seen and unseen domains")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs/research/mujoco_physics_domains.yaml")
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes-per-domain", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--source", choices=("random_exploration", "task_specific", "mixed")
    )
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    suite = dict(config["physics_domains"])
    domains = list(suite.get("domains", ()))
    validate_domains(domains)
    episodes = int(args.episodes_per_domain or suite.get("episodes_per_domain", 12))
    steps = int(args.steps or suite.get("steps_per_episode", 100))
    source = str(args.source or suite.get("source", "mixed"))
    if episodes <= 0 or steps <= 0:
        raise ValueError("episodes and steps must be positive")
    datasets = []
    domain_summaries = {}
    unseen_names = []
    for domain_index, entry in enumerate(domains):
        name = str(entry["name"])
        role = str(entry.get("role", "seen"))
        domain_config = deep_merge(config, {
            "plant": dict(entry.get("plant_override", {})),
            "experiment": {"seed": int(config["experiment"].get("seed", 0)) + 1000 * domain_index},
        })
        dataset = collect(
            domain_config,
            episodes,
            steps,
            source,
            episode_prefix=name,
            disturbance_type=name,
        )
        datasets.append(dataset)
        domain_summaries[name] = {
            "role": role,
            "summary": dataset.summary(),
            "plant_override": entry.get("plant_override", {}),
        }
        if role == "unseen":
            unseen_names.append(name)
    metadata = {
        "generator": "collect_mujoco_physics_domains.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "config_hash": config_hash(config),
        "config": config,
        "domains": domain_summaries,
    }
    dataset = combine_datasets(datasets, metadata)
    quality = assert_residual_dataset_quality(dataset, angle_indices=(2,))
    splits = stratified_domain_split(
        dataset,
        domains,
        validation_fraction=float(suite.get("validation_fraction", 0.15)),
        test_fraction=float(suite.get("test_fraction", 0.15)),
        seed=int(config["experiment"].get("seed", 0)),
    )
    split_audit = assert_episode_disjoint_splits(splits)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset.save(destination / "all.npz")
    for name in ("train", "validation", "test", "unseen"):
        getattr(splits, name).save(destination / (name + ".npz"))
    manifest = {
        "dataset_summary": dataset.summary(),
        "domain_summaries": domain_summaries,
        "unseen_domains": unseen_names,
        "quality_gate": quality,
        "split_audit": split_audit,
        "git_sha": git_sha(ROOT),
        "config_hash": config_hash(config),
    }
    with (destination / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
