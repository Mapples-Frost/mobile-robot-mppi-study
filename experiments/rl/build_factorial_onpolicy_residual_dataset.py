#!/usr/bin/env python3
"""Build episode-disjoint applied-control residual data from a factorial run."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.rl.build_l38_onpolicy_residual_dataset import _records_for_episode
from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.learning.dataset_quality import assert_residual_dataset_quality
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction
from src.learning.residual_dataset import ResidualDataset


def _resolved_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _split_episode_units(specification, design, scenes, domains):
    required = {"train", "validation", "test", "unseen"}
    expected_seeds = {
        int(value) for value in design["development_episode_seeds"]
    }
    scene_set, domain_set = set(scenes), set(domains)
    selectors = specification.get("split_selectors")
    if selectors is None:
        splits = {
            str(name): {int(value) for value in values}
            for name, values in specification["splits"].items()
        }
        if set(splits) != required:
            raise ValueError(
                "residual dataset requires train/validation/test/unseen splits"
            )
        all_seeds = [value for values in splits.values() for value in values]
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError("residual dataset split seeds must be disjoint")
        if set(all_seeds) != expected_seeds:
            raise ValueError(
                "residual dataset splits must cover the source development seeds"
            )
        units = {
            split: {
                (scene, domain, seed)
                for scene in scenes for domain in domains for seed in seeds
            }
            for split, seeds in splits.items()
        }
        contract = {
            split: {
                "scenes": list(scenes), "domains": list(domains),
                "seeds": sorted(seeds),
            }
            for split, seeds in splits.items()
        }
        return units, contract

    if set(selectors) != required:
        raise ValueError(
            "split_selectors requires train/validation/test/unseen mappings"
        )
    units, contract, owner = {}, {}, {}
    for split, raw in selectors.items():
        selected_scenes = tuple(str(value) for value in raw.get("scenes", scenes))
        selected_domains = tuple(str(value) for value in raw.get("domains", domains))
        selected_seeds = tuple(int(value) for value in raw.get("seeds", expected_seeds))
        if not selected_scenes or not selected_domains or not selected_seeds:
            raise ValueError("each split selector must choose at least one episode")
        if not set(selected_scenes) <= scene_set:
            raise ValueError("split selector names an unknown scene")
        if not set(selected_domains) <= domain_set:
            raise ValueError("split selector names an unknown physics domain")
        if not set(selected_seeds) <= expected_seeds:
            raise ValueError("split selector uses a seed outside the source run")
        selected_units = {
            (scene, domain, seed)
            for scene in selected_scenes for domain in selected_domains
            for seed in selected_seeds
        }
        for unit in selected_units:
            if unit in owner:
                raise ValueError(
                    "episode unit %s appears in both %s and %s"
                    % (unit, owner[unit], split)
                )
            owner[unit] = str(split)
        units[str(split)] = selected_units
        contract[str(split)] = {
            "scenes": list(selected_scenes),
            "domains": list(selected_domains),
            "seeds": sorted(selected_seeds),
        }
    return units, contract


def build(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    specification = config["residual_dataset"]
    source_block = int(specification.get("source_block", 0))
    condition = str(specification.get("source_condition", "traditional_nominal"))
    control_source = str(specification.get("control_source", "applied"))
    dt = float(config["experiment"]["control_dt"])
    initial_state = config["experiment"]["initial_state"]
    nominal = DynamicUnicyclePrediction(
        config["plant"].get("nominal_velocity_time_constant", 0.18),
        config["plant"].get("nominal_yaw_time_constant", 0.12),
    )
    scenes = [
        str(load_yaml(_resolved_path(item["path"]))["scene"]["name"])
        for item in design["scenes"]
    ]
    domains = [str(item["name"]) for item in design["physics_domains"]]
    units, split_contract = _split_episode_units(
        specification, design, scenes, domains
    )
    records = {name: [] for name in units}
    for split, split_units in units.items():
        for scene, domain, seed in sorted(split_units):
            path = (
                Path(input_dir) / ("block_%d" % source_block) / "runs"
                / scene / domain / condition / ("seed_%d" % seed)
                / "trajectory.csv"
            )
            records[split].extend(_records_for_episode(
                path, initial_state, scene, domain, seed, nominal, dt,
                control_source=control_source,
            ))
    metadata = {
        "generator": "build_factorial_onpolicy_residual_dataset.py",
        "source_experiment": str(Path(input_dir).resolve()),
        "source_block": source_block,
        "source_condition": condition,
        "control_source": control_source,
        "split_episode_contract": split_contract,
        "excluded_source_episode_units": int(
            len(scenes) * len(domains)
            * len(design["development_episode_seeds"])
            - sum(len(values) for values in units.values())
        ),
        "git_sha": git_sha(ROOT),
    }
    datasets = {}
    for split, split_records in records.items():
        dataset = ResidualDataset.from_records(
            split_records, metadata={**metadata, "split": split}
        )
        assert_residual_dataset_quality(dataset, angle_indices=(2,))
        datasets[split] = dataset
    return datasets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    datasets = build(config, _resolved_path(args.input_dir))
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"splits": {}}
    for name, dataset in datasets.items():
        dataset.save(output / (name + ".npz"))
        manifest["splits"][name] = dataset.summary()
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
