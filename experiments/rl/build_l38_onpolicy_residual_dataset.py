#!/usr/bin/env python3
"""Build episode-disjoint residual data from frozen nominal-MPPI L38 runs."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.learning.dataset_quality import assert_residual_dataset_quality
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction
from src.learning.residual_dataset import ResidualDataset, wrapped_finite_difference


STATE_FIELDS = ("x", "y", "theta", "v", "omega")
COMMAND_FIELDS = ("executed_v", "executed_omega")
APPLIED_FIELDS = ("applied_v", "applied_omega")
SPLIT_SEEDS = {
    "train": {20760731, 20760732},
    "validation": {20760733},
    "test": {20760734},
    "unseen": {20760735},
}


def _resolved_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _records_for_episode(
    path, initial_state, scene, domain, seed, nominal, dt,
    control_source="command",
):
    rows = _read_csv(path)
    state = np.asarray(initial_state, dtype=np.float64)
    records = []
    episode_id = "%s__%s__%d" % (scene, domain, int(seed))
    for step, row in enumerate(rows):
        following = np.asarray([float(row[name]) for name in STATE_FIELDS], dtype=np.float64)
        command = np.asarray([float(row[name]) for name in COMMAND_FIELDS], dtype=np.float64)
        applied = np.asarray([float(row[name]) for name in APPLIED_FIELDS], dtype=np.float64)
        collision = str(row["collision"]).strip().lower() in ("1", "true", "yes")
        if collision:
            break
        if control_source == "command":
            model_control = command
        elif control_source == "applied":
            model_control = applied
        else:
            raise ValueError("control_source must be 'command' or 'applied'")
        observed = wrapped_finite_difference(state, following, dt, angle_indices=(2,))
        nominal_value = np.asarray(nominal.derivative(state, model_control), dtype=np.float64)
        records.append({
            "episode_id": episode_id,
            "seed": int(seed),
            "step": int(step),
            "time": float(step) * dt,
            "dt": dt,
            "state_t": state,
            "control_t": model_control,
            "applied_control_t": applied,
            "state_t_plus_1": following,
            "nominal_derivative": nominal_value,
            "observed_derivative": observed,
            "residual_target": observed - nominal_value,
            "disturbance_type": domain,
            "disturbance_parameters": json.dumps({"physics_domain": domain}, sort_keys=True),
            "scene": scene,
            "data_source": "nominal_mppi_onpolicy",
            "model_version": "dynamic_unicycle_5_%s_input_v1" % control_source,
        })
        state = following
    return records


def build(config, input_dir, control_source="command"):
    design = config["rl"]["cross_layer_factorial"]
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
    split_records = {name: [] for name in SPLIT_SEEDS}
    for scene in scenes:
        for domain in domains:
            for seed in design["development_episode_seeds"]:
                split = next(name for name, values in SPLIT_SEEDS.items() if int(seed) in values)
                path = Path(input_dir) / "block_0" / "runs" / scene / domain / "traditional_nominal" / ("seed_%d" % int(seed)) / "trajectory.csv"
                split_records[split].extend(_records_for_episode(
                    path, initial_state, scene, domain, int(seed), nominal, dt,
                    control_source=control_source,
                ))
    metadata = {
        "generator": "build_l38_onpolicy_residual_dataset.py",
        "source_experiment": str(Path(input_dir).resolve()),
        "source_condition": "traditional_nominal_block_0",
        "control_semantics": (
            "MuJoCo interval-average delayed command"
            if control_source == "applied"
            else "safety-executed command presented to MPPI prediction"
        ),
        "applied_control_semantics": "MuJoCo interval-average delayed command",
        "split_seed_contract": {name: sorted(values) for name, values in SPLIT_SEEDS.items()},
        "git_sha": git_sha(ROOT),
    }
    datasets = {}
    for name, records in split_records.items():
        dataset = ResidualDataset.from_records(records, metadata={**metadata, "split": name})
        assert_residual_dataset_quality(dataset, angle_indices=(2,))
        datasets[name] = dataset
    return datasets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--control-source", choices=("command", "applied"), default="command"
    )
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    datasets = build(
        config, _resolved_path(args.input_dir), control_source=args.control_source
    )
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
