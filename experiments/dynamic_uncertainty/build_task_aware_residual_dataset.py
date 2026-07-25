#!/usr/bin/env python3
"""Build an episode-disjoint residual dataset with dynamic-obstacle coverage.

The frozen L56 splits remain intact.  One complete collision-free nominal
episode is appended to each of train, validation, and test so that no obstacle
trajectory leaks across model fitting, checkpoint selection, and evaluation.
Near-obstacle transitions receive larger positive sample weights; the original
L56 transitions retain unit weight.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.learning.dataset_quality import (
    assert_residual_dataset_quality,
)


STATE_FIELDS = ("x", "y", "theta", "v", "omega")
APPLIED_FIELDS = ("applied_v", "applied_omega")
TASK_SEEDS = {
    "train": 730100001,
    "validation": 730100003,
    "test": 730100005,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _nominal_derivative(state, control, tau_v, tau_w):
    x, y, theta, velocity, yaw_rate = state
    del x, y
    return np.asarray(
        [
            velocity * math.cos(theta),
            velocity * math.sin(theta),
            yaw_rate,
            (control[0] - velocity) / tau_v,
            (control[1] - yaw_rate) / tau_w,
        ],
        dtype=np.float64,
    )


def _wrapped_derivative(state, following, dt):
    delta = np.asarray(following, dtype=np.float64) - np.asarray(
        state, dtype=np.float64
    )
    delta[2] = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
    return delta / float(dt)


def _task_records(
    episode_dir,
    seed,
    base_weight,
    proximity_gain,
    proximity_scale,
    probability_gain,
):
    episode_dir = Path(episode_dir)
    config = load_yaml(episode_dir / "config_resolved.yaml")
    metrics = json.loads(
        (episode_dir / "metrics.json").read_text(encoding="utf-8")
    )
    if bool(metrics.get("collision", True)):
        raise ValueError("task-aware residual data must be collision-free")
    rows = _read_rows(episode_dir / "trajectory.csv")
    if not rows:
        raise ValueError("task-aware trajectory is empty")
    dt = float(config["experiment"]["control_dt"])
    state = np.asarray(config["experiment"]["initial_state"], dtype=np.float64)
    if state.shape != (5,):
        raise ValueError("task-aware residual data requires five-state dynamics")
    plant = dict(config.get("plant", {}))
    tau_v = float(plant.get("nominal_velocity_time_constant", 0.18))
    tau_w = float(plant.get("nominal_yaw_time_constant", 0.12))
    scene = str(config.get("scene", {}).get("name", "dynamic_uncertainty_v3"))
    episode_id = "dynamic_uncertainty_v3_nominal__%d" % int(seed)
    records = []
    for step, row in enumerate(rows):
        following = np.asarray(
            [float(row[name]) for name in STATE_FIELDS], dtype=np.float64
        )
        applied = np.asarray(
            [float(row[name]) for name in APPLIED_FIELDS], dtype=np.float64
        )
        observed = _wrapped_derivative(state, following, dt)
        nominal = _nominal_derivative(state, applied, tau_v, tau_w)
        clearance = float(row["clearance"])
        proximity = math.exp(
            -max(clearance, 0.0) / float(proximity_scale)
        )
        probability = float(
            row.get("probabilistic_obstacle_maximum_step_probability", 0.0)
        )
        probability = min(max(probability, 0.0), 1.0)
        sample_weight = (
            float(base_weight)
            + float(proximity_gain) * proximity
            + float(probability_gain) * probability
        )
        records.append(
            {
                "episode_id": episode_id,
                "seed": int(seed),
                "step": int(step),
                "time": float(step) * dt,
                "dt": dt,
                "state_t": state.copy(),
                "control_t": applied.copy(),
                "applied_control_t": applied.copy(),
                "state_t_plus_1": following.copy(),
                "nominal_derivative": nominal,
                "observed_derivative": observed,
                "residual_target": observed - nominal,
                "disturbance_type": "dynamic_uncertainty_v3_fixed_plant",
                "disturbance_parameters": json.dumps(
                    {
                        "obstacle_seed": int(seed),
                        "control_source": "applied",
                        "clearance_m": clearance,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "scene": scene,
                "data_source": "dynamic_obstacle_nominal",
                "model_version": "dynamic_unicycle_5_applied_input_task_aware_v1",
                "sample_weight": sample_weight,
            }
        )
        state = following
    return records


def _records_to_arrays(records):
    vector_fields = {
        "state_t",
        "control_t",
        "applied_control_t",
        "state_t_plus_1",
        "nominal_derivative",
        "observed_derivative",
        "residual_target",
    }
    integer_fields = {"seed", "step"}
    result = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if key in vector_fields:
            result[key] = np.asarray(values, dtype=np.float64)
        elif key in integer_fields:
            result[key] = np.asarray(values, dtype=np.int64)
        elif key in {"time", "dt", "sample_weight"}:
            result[key] = np.asarray(values, dtype=np.float64)
        else:
            result[key] = np.asarray(values, dtype=str)
    return result


def _merge(base, task):
    result = {}
    keys = set(base).union(task)
    if keys.difference(set(base).union({"sample_weight"})):
        raise ValueError("task records contain unsupported fields")
    for key in sorted(keys):
        if key == "sample_weight" and key not in base:
            base_value = np.ones(len(base["state_t"]), dtype=np.float64)
        else:
            base_value = base[key]
        result[key] = np.concatenate((base_value, task[key]), axis=0)
    assert_residual_dataset_quality(result, angle_indices=(2,))
    return result


def build(args):
    base_dir = Path(args.base_dataset_dir).resolve()
    trajectory_root = Path(args.trajectory_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {}
    for split in ("train", "validation", "test", "unseen"):
        base_path = base_dir / ("%s.npz" % split)
        with np.load(base_path, allow_pickle=False) as archive:
            base = {key: archive[key] for key in archive.files}
        if split in TASK_SEEDS:
            seed = TASK_SEEDS[split]
            task = _records_to_arrays(
                _task_records(
                    trajectory_root / ("seed_%d" % seed),
                    seed,
                    args.task_base_weight,
                    args.proximity_gain,
                    args.proximity_scale,
                    args.probability_gain,
                )
            )
            merged = _merge(base, task)
            task_count = int(len(task["state_t"]))
            task_weight = {
                "minimum": float(np.min(task["sample_weight"])),
                "median": float(np.median(task["sample_weight"])),
                "maximum": float(np.max(task["sample_weight"])),
            }
        else:
            merged = dict(base)
            merged["sample_weight"] = np.ones(
                len(base["state_t"]), dtype=np.float64
            )
            assert_residual_dataset_quality(merged, angle_indices=(2,))
            task_count = 0
            task_weight = None
        destination = output_dir / ("%s.npz" % split)
        np.savez_compressed(destination, **merged)
        split_manifest[split] = {
            "path": str(destination),
            "sha256": _sha256(destination),
            "base_sha256": _sha256(base_path),
            "transition_count": int(len(merged["state_t"])),
            "episode_count": int(np.unique(merged["episode_id"]).size),
            "task_transition_count": task_count,
            "task_sample_weight": task_weight,
        }
    manifest = {
        "schema_version": 1,
        "design": "task_aware_residual_retraining_v1",
        "base_dataset_dir": str(base_dir),
        "trajectory_root": str(trajectory_root),
        "episode_disjoint_task_seed_assignment": TASK_SEEDS,
        "weighting": {
            "original_transition_weight": 1.0,
            "task_base_weight": float(args.task_base_weight),
            "proximity_gain": float(args.proximity_gain),
            "proximity_scale_m": float(args.proximity_scale),
            "probability_gain": float(args.probability_gain),
        },
        "splits": split_manifest,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset-dir", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-base-weight", type=float, default=4.0)
    parser.add_argument("--proximity-gain", type=float, default=4.0)
    parser.add_argument("--proximity-scale", type=float, default=0.5)
    parser.add_argument("--probability-gain", type=float, default=2.0)
    args = parser.parse_args(argv)
    for name in (
        "task_base_weight",
        "proximity_gain",
        "proximity_scale",
        "probability_gain",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error("--%s must be finite and positive" % name.replace("_", "-"))
    manifest = build(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
