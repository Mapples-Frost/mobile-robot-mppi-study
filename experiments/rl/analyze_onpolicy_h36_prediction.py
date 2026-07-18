#!/usr/bin/env python3
"""Evaluate current ICODE checkpoints on frozen nominal-MPPI L38 trajectories."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.core.spaces import dynamic_unicycle_state
from mobile_robot_mppi.learning.models import PlatformResidualDynamics, load_platform_checkpoint
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction, ResidualPrediction
from mobile_robot_mppi.planning.mppi import integrate_batch


HORIZONS = (1, 5, 10, 20, 36)
STATE_FIELDS = ("x", "y", "theta", "v", "omega")
CONTROL_FIELDS = ("executed_v", "executed_omega")


def _resolved_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _trajectory_arrays(path, initial_state):
    rows = _read_csv(path)
    states = [np.asarray(initial_state, dtype=np.float64)]
    controls = []
    collisions = []
    for row in rows:
        states.append(np.asarray([float(row[name]) for name in STATE_FIELDS], dtype=np.float64))
        controls.append(np.asarray([float(row[name]) for name in CONTROL_FIELDS], dtype=np.float64))
        collisions.append(str(row["collision"]).strip().lower() in ("1", "true", "yes"))
    return np.asarray(states), np.asarray(controls), np.asarray(collisions, dtype=bool)


def _hash_arrays(states, controls, collisions):
    digest = hashlib.sha256()
    for value in (states, controls, collisions):
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()


def _rollout(dynamics, state, controls, dt):
    spec = dynamic_unicycle_state()
    current = np.asarray(state, dtype=np.float64)[None, :]
    values = []
    for control in np.asarray(controls, dtype=np.float64):
        current = integrate_batch(
            dynamics, current, control[None, :], float(dt), spec, "rk4"
        )
        values.append(current[0].copy())
    return np.asarray(values, dtype=np.float64)


def _wrapped_error(predicted, target):
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    error[..., 2] = np.arctan2(np.sin(error[..., 2]), np.cos(error[..., 2]))
    return error


def _feature_zscore(model, states, controls):
    state = np.asarray(states, dtype=np.float32)
    features = np.concatenate(
        (state[..., :2], np.sin(state[..., 2:3]), np.cos(state[..., 2:3]), state[..., 3:]),
        axis=-1,
    )
    feature_mean = model.feature_mean.detach().cpu().numpy()
    feature_scale = model.feature_scale.detach().cpu().numpy()
    control_mean = model.control_mean.detach().cpu().numpy()
    control_scale = model.control_scale.detach().cpu().numpy()
    feature_z = np.abs((features - feature_mean) / feature_scale)
    control_z = np.abs((np.asarray(controls, dtype=np.float32) - control_mean) / control_scale)
    return float(np.max(feature_z)), float(np.max(control_z))


def _episode_windows(states, controls, collisions, horizon):
    output = []
    start = 0
    while start + horizon <= len(controls):
        stop = start + horizon
        if not np.any(collisions[start:stop]):
            output.append((start, stop))
        start += horizon
    return output


def _metrics(errors):
    stacked = np.asarray(errors, dtype=np.float64)
    position = np.linalg.norm(stacked[..., :2], axis=-1)
    endpoint = stacked[:, -1, :]
    endpoint_position = np.linalg.norm(endpoint[:, :2], axis=-1)
    return {
        "window_count": int(stacked.shape[0]),
        "position_rmse_m": float(np.sqrt(np.mean(position ** 2))),
        "endpoint_position_rmse_m": float(np.sqrt(np.mean(endpoint_position ** 2))),
        "endpoint_heading_rmse_rad": float(np.sqrt(np.mean(endpoint[:, 2] ** 2))),
        "endpoint_v_rmse_mps": float(np.sqrt(np.mean(endpoint[:, 3] ** 2))),
        "endpoint_omega_rmse_radps": float(np.sqrt(np.mean(endpoint[:, 4] ** 2))),
    }


def analyze(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    initial_state = config["experiment"]["initial_state"]
    dt = float(config["experiment"]["control_dt"])
    scene_names = [
        str(load_yaml(_resolved_path(item["path"]))["scene"]["name"])
        for item in design["scenes"]
    ]
    domains = [str(item["name"]) for item in design["physics_domains"]]
    seeds = [int(value) for value in design["development_episode_seeds"]]
    canonical = {}
    identity_failures = []
    for scene in scene_names:
        for domain in domains:
            for seed in seeds:
                arrays = []
                for block in range(len(design["model_blocks"])):
                    path = Path(input_dir) / ("block_%d" % block) / "runs" / scene / domain / "traditional_nominal" / ("seed_%d" % seed) / "trajectory.csv"
                    arrays.append(_trajectory_arrays(path, initial_state))
                hashes = [_hash_arrays(*item) for item in arrays]
                if len(set(hashes)) != 1:
                    identity_failures.append({"scene": scene, "domain": domain, "episode_seed": seed, "hashes": hashes})
                canonical[(scene, domain, seed)] = arrays[0]

    nominal = DynamicUnicyclePrediction(
        config["plant"].get("nominal_velocity_time_constant", 0.18),
        config["plant"].get("nominal_yaw_time_constant", 0.12),
    )
    models = {"nominal": (nominal, None)}
    checkpoint_meta = {}
    for block, item in enumerate(design["model_blocks"]):
        checkpoint = _resolved_path(item["icode_checkpoint"])
        residual = PlatformResidualDynamics.from_checkpoint(checkpoint, device="cpu", use_torchscript=True)
        model, payload = load_platform_checkpoint(checkpoint, "cpu")
        name = "icode_block_%d" % block
        models[name] = (ResidualPrediction(nominal, residual), model)
        checkpoint_meta[name] = {
            "path": str(checkpoint),
            "icode_seed": int(item["icode_seed"]),
            "epoch": int(payload["epoch"]),
            "training_rollout_horizon": int(payload["training_config"]["training"]["rollout_horizon"]),
        }

    rows = []
    for (scene, domain, seed), (states, controls, collisions) in sorted(canonical.items()):
        for horizon in HORIZONS:
            windows = _episode_windows(states, controls, collisions, horizon)
            for model_name, (dynamics, residual_model) in models.items():
                errors = []
                max_feature_z = 0.0
                max_control_z = 0.0
                for start, stop in windows:
                    prediction = _rollout(dynamics, states[start], controls[start:stop], dt)
                    target = states[start + 1:stop + 1]
                    errors.append(_wrapped_error(prediction, target))
                    if residual_model is not None:
                        feature_z, control_z = _feature_zscore(
                            residual_model, prediction, controls[start:stop]
                        )
                        max_feature_z = max(max_feature_z, feature_z)
                        max_control_z = max(max_control_z, control_z)
                if not errors:
                    continue
                metrics = _metrics(errors)
                rows.append({
                    "scene": scene, "physics_domain": domain, "episode_seed": seed,
                    "model": model_name, "horizon": horizon,
                    "max_feature_abs_z": max_feature_z,
                    "max_control_abs_z": max_control_z,
                    **metrics,
                })

    h36 = [row for row in rows if int(row["horizon"]) == 36]
    lookup = {
        (row["scene"], row["physics_domain"], int(row["episode_seed"]), row["model"]): row
        for row in h36
    }
    block_summaries = []
    for block in range(len(design["model_blocks"])):
        improvements = []
        heading_ratios = []
        for key in canonical:
            nominal_row = lookup.get(key + ("nominal",))
            icode_row = lookup.get(key + (("icode_block_%d" % block),))
            if nominal_row is None or icode_row is None:
                continue
            improvements.append(
                nominal_row["endpoint_position_rmse_m"] - icode_row["endpoint_position_rmse_m"]
            )
            heading_ratios.append(
                icode_row["endpoint_heading_rmse_rad"] / max(nominal_row["endpoint_heading_rmse_rad"], 1e-9)
            )
        block_summaries.append({
            "model_block": block,
            "episodes_with_h36": len(improvements),
            "mean_endpoint_position_improvement_m": float(np.mean(improvements)) if improvements else None,
            "mean_heading_rmse_ratio": float(np.mean(heading_ratios)) if heading_ratios else None,
        })

    nominal_h36 = [row for row in h36 if row["model"] == "nominal"]
    icode_h36 = [row for row in h36 if row["model"].startswith("icode_block_")]
    nominal_mean = float(np.mean([row["endpoint_position_rmse_m"] for row in nominal_h36]))
    icode_mean = float(np.mean([row["endpoint_position_rmse_m"] for row in icode_h36]))
    relative_improvement = (nominal_mean - icode_mean) / max(nominal_mean, 1e-12)
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for name, value in row.items()
        if name not in ("scene", "physics_domain", "model")
    )
    gate = {
        "trajectory_identity": not identity_failures,
        "finite_predictions": finite,
        "positive_h36_checkpoints": int(sum(
            row["mean_endpoint_position_improvement_m"] is not None
            and row["mean_endpoint_position_improvement_m"] > 0.0
            for row in block_summaries
        )),
        "pooled_h36_relative_position_improvement": relative_improvement,
        "heading_nonregression_checkpoints": int(sum(
            row["mean_heading_rmse_ratio"] is not None
            and row["mean_heading_rmse_ratio"] <= 1.20
            for row in block_summaries
        )),
    }
    gate["passed"] = bool(
        gate["trajectory_identity"] and gate["finite_predictions"]
        and gate["positive_h36_checkpoints"] >= 2
        and relative_improvement >= 0.05
        and gate["heading_nonregression_checkpoints"] == len(block_summaries)
    )
    return rows, {
        "design_id": "l39_onpolicy_h36_prediction_v1",
        "canonical_episodes": len(canonical),
        "trajectory_identity_failures": identity_failures,
        "checkpoint_metadata": checkpoint_meta,
        "h36": {
            "nominal_mean_endpoint_position_rmse_m": nominal_mean,
            "icode_mean_endpoint_position_rmse_m": icode_mean,
            "relative_position_improvement": relative_improvement,
            "block_summaries": block_summaries,
        },
        "gate": gate,
        "interpretation_guard": "Windows are nested within frozen nominal-MPPI episodes; they are not independent experimental replicates.",
    }


def _write_csv(path, rows):
    fields = sorted({name for row in rows for name in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    input_dir = _resolved_path(args.input_dir)
    rows, summary = analyze(config, input_dir)
    _write_csv(input_dir / "onpolicy_horizon_metrics.csv", rows)
    path = input_dir / "onpolicy_h36_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
