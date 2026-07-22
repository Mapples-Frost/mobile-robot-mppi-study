#!/usr/bin/env python3
"""Run the preregistered L270 read-only observation-aliasing diagnosis."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.rl.train_rl_sampling_prior import _scene_configs
from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.rl.environment import DirectControlEnv


PROTOCOL = "L270"
DEFAULT_CONFIG = ROOT / "configs/rl/l270_observation_aliasing_diagnosis.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows):
    rows = list(rows)
    fields, seen = [], set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fields.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _verify_file(path: Path, expected: str):
    actual = _sha256(path)
    if actual != expected:
        raise ValueError("SHA256 mismatch for %s: %s" % (path, actual))
    return actual


def _load_config(path: Path):
    config = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    section = config["l270"]
    if section["protocol"] != PROTOCOL or section["conditional_training"] is not False:
        raise ValueError("L270 must remain diagnostic and training-disabled")
    references = json.dumps({
        key: value for key, value in section.items()
        if key not in ("forbidden_tokens",)
    }, sort_keys=True).lower()
    forbidden = [
        token for token in section["forbidden_tokens"]
        if token.lower() in references
    ]
    if forbidden:
        raise ValueError("forbidden artifact entered L270 config: %s" % forbidden)
    return config, section


def _grid_oracles(section):
    directory = ROOT / section["l263"]["directory"]
    states_path = directory / "state_manifest.json"
    rollout_path = directory / "counterfactual_rollouts.csv"
    _verify_file(states_path, section["l263"]["state_manifest_sha256"])
    _verify_file(rollout_path, section["l263"]["counterfactual_rollouts_sha256"])
    states = json.loads(states_path.read_text(encoding="utf-8"))
    action_rows = _read_csv(directory / "critic_actions.csv")
    action_by_key = {
        (row["state_id"], row["action_id"]): row for row in action_rows
    }
    grouped = defaultdict(list)
    audit = section["state_probe"]
    for row in _read_csv(rollout_path):
        if (
            row["continuation"] == audit["continuation"]
            and int(row["horizon"]) == int(audit["horizon"])
            and "grid_" in row["semantic_labels"]
        ):
            grouped[row["state_id"]].append(row)
    output = []
    for state in states:
        rows = grouped[state["state_id"]]
        if len(rows) != 25:
            raise ValueError("L263 oracle grid must contain exactly 25 actions")
        def order(row):
            action = action_by_key[(row["state_id"], row["action_id"])]
            vector = np.asarray(
                (float(action["normalized_v"]), float(action["normalized_omega"])),
                dtype=np.float64,
            )
            return (-float(row["discounted_return"]), float(np.linalg.norm(vector)), row["action_id"])
        best = sorted(rows, key=order)[0]
        action = action_by_key[(best["state_id"], best["action_id"])]
        output.append({
            "state_id": state["state_id"],
            "scene": state["scene"],
            "scene_role": state["scene_role"],
            "kind": state["kind"],
            "observation": np.asarray(state["normalized_observation"], dtype=np.float64),
            "oracle_action_id": best["action_id"],
            "oracle_v": float(action["normalized_v"]),
            "oracle_omega": float(action["normalized_omega"]),
            "oracle_return": float(best["discounted_return"]),
        })
    return output


def _neighbor_conflicts(oracles, probe):
    matrix = np.stack([row["observation"] for row in oracles])
    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    k = int(probe["nearest_neighbors"])
    rows = []
    for left_index, left in enumerate(oracles):
        neighbors = np.argsort(distances[left_index], kind="mergesort")[:k]
        for rank, right_index in enumerate(neighbors, start=1):
            right = oracles[int(right_index)]
            left_omega, right_omega = left["oracle_omega"], right["oracle_omega"]
            opposite = (
                abs(left_omega) >= float(probe["omega_sign_minimum"])
                and abs(right_omega) >= float(probe["omega_sign_minimum"])
                and np.sign(left_omega) != np.sign(right_omega)
            )
            delta = abs(left_omega - right_omega)
            rows.append({
                "state_id": left["state_id"],
                "neighbor_state_id": right["state_id"],
                "neighbor_rank": rank,
                "scene": left["scene"],
                "neighbor_scene": right["scene"],
                "same_scene": left["scene"] == right["scene"],
                "distance_69d": float(distances[left_index, right_index]),
                "oracle_omega": left_omega,
                "neighbor_oracle_omega": right_omega,
                "omega_delta": delta,
                "opposite_sign": bool(opposite),
                "steering_conflict": bool(
                    opposite or delta >= float(probe["omega_conflict_delta"])
                ),
            })
    threshold = float(np.quantile(
        [row["distance_69d"] for row in rows], float(probe["close_pair_quantile"])
    ))
    for row in rows:
        row["close_pair"] = bool(row["distance_69d"] <= threshold)
    close = [row for row in rows if row["close_pair"]]
    summary = {
        "state_count": len(oracles),
        "directed_neighbor_pair_count": len(rows),
        "close_distance_threshold": threshold,
        "close_pair_count": len(close),
        "close_pair_conflict_fraction": float(np.mean([
            row["steering_conflict"] for row in close
        ])),
        "within_scene_close_pair_conflict_fraction": _optional_mean([
            row["steering_conflict"] for row in close if row["same_scene"]
        ]),
        "cross_scene_close_pair_conflict_fraction": _optional_mean([
            row["steering_conflict"] for row in close if not row["same_scene"]
        ]),
    }
    return rows, summary


def _optional_mean(values):
    return None if not values else float(np.mean(values))


def _even_indices(length: int, minimum: int, count: int):
    if length <= minimum:
        return np.empty(0, dtype=np.int64)
    values = np.linspace(minimum, length - 1, min(count, length - minimum))
    return np.unique(np.rint(values).astype(np.int64))


def _far_preview(reference, pose, progress, distances, scale):
    poses = np.asarray(reference.preview_poses(
        np.asarray(distances, dtype=np.float64), progress_floor=float(progress)
    ), dtype=np.float64)
    delta = poses[:, :2] - pose[None, :2]
    cosine, sine = math.cos(float(pose[2])), math.sin(float(pose[2]))
    body_x = cosine * delta[:, 0] + sine * delta[:, 1]
    body_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    return np.clip(
        np.column_stack((body_x, body_y)).reshape(-1) / float(scale), -1.0, 1.0
    ).astype(np.float32)


def _reference_cache(base, chains):
    cache, environments = {}, []
    for scene_config in sorted({row["scene_config"] for row in chains}):
        resolved = copy.deepcopy(_scene_configs(base, [scene_config])[0])
        environment = DirectControlEnv(resolved, ROOT, seed=0)
        environments.append(environment)
        cache[scene_config] = environment.components["reference"]
    return cache, environments


def _recovery_rows(config, section):
    dataset = ROOT / section["l268"]["dataset"]
    manifest_path = dataset / "manifest.json"
    integrity_path = ROOT / section["l268"]["dataset_integrity"]
    _verify_file(manifest_path, section["l268"]["dataset_manifest_sha256"])
    _verify_file(integrity_path, section["l268"]["dataset_integrity_sha256"])
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity["dataset_content_fingerprint"] != section["l268"]["dataset_content_fingerprint"]:
        raise ValueError("L268 dataset content fingerprint mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chains = [row for row in manifest["chains"] if row["accepted"]]
    base = load_yaml(ROOT / "configs/rl/l262_coverage_gated_value_6k.yaml")
    references, environments = _reference_cache(base, chains)
    probe = section["recovery_probe"]
    rows = []
    try:
        for chain in sorted(chains, key=lambda row: int(row["chain_id"])):
            shard = dataset / Path(chain["student_npz"])
            audit = dataset / "raw" / "audit" / (chain["attempt_id"] + ".csv")
            audit_rows = _read_csv(audit)
            with np.load(shard, allow_pickle=False) as arrays:
                observations = np.asarray(arrays["observations"], dtype=np.float32)
                actions = np.asarray(arrays["actions"], dtype=np.float32)
            if len(audit_rows) != len(observations) or len(actions) != len(observations):
                raise ValueError("L268 recovery chain length mismatch")
            reference = references[chain["scene_config"]]
            indices = _even_indices(
                len(observations), int(probe["minimum_history_step"]),
                int(probe["samples_per_chain"]),
            )
            for index in indices:
                # Dataset observations/actions are recorded before env.step(), while
                # each audit row stores the truth pose after that same step.  Align
                # candidate observable features with observation[t] by using the
                # post-step truth from t - 1.  The frozen minimum history step is 2,
                # so every retained sample has this predecessor available.
                audit_index = int(index) - 1
                audit_row = audit_rows[audit_index]
                if int(audit_row["step"]) != audit_index:
                    raise ValueError("L268 audit step order drifted")
                pose = np.asarray((
                    float(audit_row["truth_x"]), float(audit_row["truth_y"]),
                    float(audit_row["truth_theta"]),
                ), dtype=np.float64)
                progress = float(audit_row["path_progress_m"])
                projection = reference.project(
                    pose[:2], minimum_progress=max(0.0, progress - 1e-6)
                )
                total = float(reference.total_length)
                progress_features = np.asarray((
                    np.clip(progress / total, 0.0, 1.0),
                    np.clip((total - progress) / total, 0.0, 1.0),
                    float(projection.segment_index) /
                    max(1.0, float(len(reference.segment_lengths) - 1)),
                ), dtype=np.float32)
                current = observations[int(index)]
                history = np.concatenate((
                    observations[int(index) - 2], observations[int(index) - 1], current
                )).astype(np.float32)
                far = _far_preview(
                    reference, pose, progress,
                    probe["far_preview_distances_m"], probe["far_preview_scale_m"],
                )
                rows.append({
                    "chain_id": int(chain["chain_id"]),
                    "attempt_id": chain["attempt_id"],
                    "scene": chain["scene"],
                    "split": chain["split"],
                    "step": int(index),
                    "aligned_audit_step": audit_index,
                    "current_69d": current,
                    "far_preview": np.concatenate((current, far)),
                    "three_frame_history": history,
                    "progress_segment": np.concatenate((current, progress_features)),
                    "all_privileged": np.concatenate((history, far, progress_features)),
                    "target": actions[int(index)],
                })
    finally:
        for environment in environments:
            environment.close()
    return rows, integrity


def _metrics(target, prediction, sign_minimum):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    mask = np.abs(target[:, 1]) >= float(sign_minimum)
    return {
        "rows": int(len(target)),
        "v_mae": float(np.mean(np.abs(error[:, 0]))),
        "omega_mae": float(np.mean(np.abs(error[:, 1]))),
        "action_rmse": float(np.sqrt(np.mean(error ** 2))),
        "omega_sign_rows": int(np.sum(mask)),
        "omega_sign_accuracy": None if not np.any(mask) else float(np.mean(
            np.sign(prediction[mask, 1]) == np.sign(target[mask, 1])
        )),
    }


def _fit_probes(rows, section):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    arms = section["recovery_probe"]["feature_arms"]
    splits = np.asarray([row["split"] for row in rows])
    target = np.stack([row["target"] for row in rows])
    sign_minimum = section["recovery_probe"]["omega_sign_evaluation_minimum"]
    predictions, metrics = [], []
    model_specs = [("ridge", None)] + [
        ("random_forest", int(seed))
        for seed in section["recovery_probe"]["model_seeds"]
    ]
    for arm in arms:
        features = np.stack([row[arm] for row in rows])
        for family, seed in model_specs:
            if family == "ridge":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(alpha=float(
                        section["recovery_probe"]["ridge_alpha"]
                    ))),
                ])
            else:
                spec = section["recovery_probe"]["random_forest"]
                model = Pipeline([("regressor", RandomForestRegressor(
                    n_estimators=int(spec["n_estimators"]),
                    max_depth=int(spec["max_depth"]),
                    min_samples_leaf=int(spec["min_samples_leaf"]),
                    max_features=float(spec["max_features"]),
                    n_jobs=int(spec["n_jobs"]), random_state=seed,
                ))])
            train = splits == "train"
            model.fit(features[train], target[train])
            for split in ("validation", "test"):
                selected = np.flatnonzero(splits == split)
                predicted = np.asarray(model.predict(features[selected]), dtype=np.float64)
                overall = _metrics(target[selected], predicted, sign_minimum)
                metrics.append({
                    "feature_arm": arm, "model_family": family,
                    "model_seed": "" if seed is None else seed,
                    "split": split, "scene": "__aggregate__", **overall,
                })
                for scene in sorted({rows[index]["scene"] for index in selected}):
                    scene_indices = np.asarray([
                        local for local, index in enumerate(selected)
                        if rows[index]["scene"] == scene
                    ], dtype=np.int64)
                    scene_metrics = _metrics(
                        target[selected][scene_indices], predicted[scene_indices], sign_minimum
                    )
                    metrics.append({
                        "feature_arm": arm, "model_family": family,
                        "model_seed": "" if seed is None else seed,
                        "split": split, "scene": scene, **scene_metrics,
                    })
                for local, index in enumerate(selected):
                    predictions.append({
                        "feature_arm": arm, "model_family": family,
                        "model_seed": "" if seed is None else seed,
                        "split": split, "scene": rows[index]["scene"],
                        "chain_id": rows[index]["chain_id"], "step": rows[index]["step"],
                        "target_v": float(target[index, 0]),
                        "target_omega": float(target[index, 1]),
                        "predicted_v": float(predicted[local, 0]),
                        "predicted_omega": float(predicted[local, 1]),
                    })
    return predictions, metrics


def _median_metric(rows, arm, split, scene, name):
    values = [
        float(row[name]) for row in rows
        if row["feature_arm"] == arm
        and row["model_family"] == "random_forest"
        and row["split"] == split and row["scene"] == scene
        and row[name] not in (None, "")
    ]
    if not values:
        raise ValueError("missing primary probe metric")
    return float(np.median(values))


def _gate(metrics, neighbor_summary, section):
    gate = section["gate"]
    baseline = "current_69d"
    scenes = sorted({
        row["scene"] for row in metrics
        if row["split"] == "test" and row["scene"] != "__aggregate__"
    })
    rows = []
    for arm in section["selection_order"]:
        base_test = _median_metric(metrics, baseline, "test", "__aggregate__", "omega_mae")
        arm_test = _median_metric(metrics, arm, "test", "__aggregate__", "omega_mae")
        base_val = _median_metric(metrics, baseline, "validation", "__aggregate__", "omega_mae")
        arm_val = _median_metric(metrics, arm, "validation", "__aggregate__", "omega_mae")
        base_sign = _median_metric(
            metrics, baseline, "test", "__aggregate__", "omega_sign_accuracy"
        )
        arm_sign = _median_metric(
            metrics, arm, "test", "__aggregate__", "omega_sign_accuracy"
        )
        scene_changes = {}
        for scene in scenes:
            base_scene = _median_metric(metrics, baseline, "test", scene, "omega_mae")
            arm_scene = _median_metric(metrics, arm, "test", scene, "omega_mae")
            scene_changes[scene] = (base_scene - arm_scene) / max(base_scene, 1e-12)
        seed_directions = []
        for seed in section["recovery_probe"]["model_seeds"]:
            def value(feature):
                matches = [row for row in metrics if (
                    row["feature_arm"] == feature
                    and row["model_family"] == "random_forest"
                    and int(row["model_seed"]) == int(seed)
                    and row["split"] == "test" and row["scene"] == "__aggregate__"
                )]
                return float(matches[0]["omega_mae"])
            seed_directions.append(value(arm) < value(baseline))
        test_improvement = (base_test - arm_test) / max(base_test, 1e-12)
        validation_improvement = (base_val - arm_val) / max(base_val, 1e-12)
        checks = {
            "close_pair_conflict_fraction": neighbor_summary[
                "close_pair_conflict_fraction"
            ] >= float(gate["minimum_close_pair_conflict_fraction"]),
            "test_omega_mae": test_improvement >= float(
                gate["minimum_test_omega_mae_relative_improvement"]
            ),
            "test_omega_sign_accuracy": arm_sign - base_sign >= float(
                gate["minimum_test_omega_sign_accuracy_gain"]
            ),
            "validation_omega_mae": validation_improvement >= float(
                gate["minimum_validation_omega_mae_relative_improvement"]
            ),
            "all_model_seed_directions_positive": (
                all(seed_directions)
                if gate["require_all_model_seed_directions_positive"] else True
            ),
            "scene_coverage": sum(value > 0.0 for value in scene_changes.values())
            >= int(gate["minimum_scenes_improved"]),
            "no_scene_regression": min(scene_changes.values()) >= -float(
                gate["maximum_scene_omega_mae_relative_regression"]
            ),
        }
        rows.append({
            "feature_arm": arm,
            "test_omega_mae_relative_improvement": test_improvement,
            "test_omega_sign_accuracy_gain": arm_sign - base_sign,
            "validation_omega_mae_relative_improvement": validation_improvement,
            "model_seed_directions": seed_directions,
            "scenes_improved": sum(value > 0.0 for value in scene_changes.values()),
            "minimum_scene_relative_improvement": min(scene_changes.values()),
            "scene_relative_improvements": scene_changes,
            "checks": checks,
            "gate_pass": all(checks.values()),
        })
    passing = [row["feature_arm"] for row in rows if row["gate_pass"]]
    selected = next((arm for arm in section["selection_order"] if arm in passing), None)
    if selected is not None:
        decision = "observation_aliasing_supported"
    elif neighbor_summary["close_pair_conflict_fraction"] >= float(
        gate["minimum_close_pair_conflict_fraction"]
    ):
        decision = "close_conflicts_but_privileged_probe_unresolved"
    else:
        decision = "observation_aliasing_not_supported"
    return rows, selected, decision


def run(config_path: Path, output: Path):
    config, section = _load_config(config_path)
    checkpoint = ROOT / section["source_checkpoint"]
    _verify_file(checkpoint, section["source_checkpoint_sha256"])
    l269_dir = ROOT / section["l269"]["directory"]
    _verify_file(l269_dir / "summary.json", section["l269"]["summary_sha256"])
    _verify_file(
        l269_dir / "candidate_advantages.csv",
        section["l269"]["candidate_advantages_sha256"],
    )
    l269_summary = json.loads((l269_dir / "summary.json").read_text(encoding="utf-8"))
    if l269_summary["status"] != section["l269"]["required_status"]:
        raise ValueError("L270 requires the frozen negative L269 Gate")

    oracles = _grid_oracles(section)
    neighbor_rows, neighbor_summary = _neighbor_conflicts(
        oracles, section["state_probe"]
    )
    recovery_rows, integrity = _recovery_rows(config, section)
    split_chains = {
        split: len({row["chain_id"] for row in recovery_rows if row["split"] == split})
        for split in ("train", "validation", "test")
    }
    if split_chains != {"train": 72, "validation": 18, "test": 18}:
        raise ValueError("L270 recovery-chain split drifted")
    feature_dimensions = {
        arm: int(np.asarray(recovery_rows[0][arm]).size)
        for arm in section["recovery_probe"]["feature_arms"]
    }
    if feature_dimensions["current_69d"] != 69:
        raise ValueError("L270 current observation is not frozen 69D")
    predictions, metrics = _fit_probes(recovery_rows, section)
    gate_rows, selected, decision = _gate(metrics, neighbor_summary, section)

    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "nearest_neighbor_conflicts.csv", neighbor_rows)
    _write_csv(output / "probe_metrics.csv", metrics)
    _write_csv(output / "probe_predictions.csv", predictions)
    _write_csv(output / "gate_by_arm.csv", [{
        **{key: value for key, value in row.items()
           if key not in ("model_seed_directions", "scene_relative_improvements", "checks")},
        "model_seed_directions": json.dumps(row["model_seed_directions"]),
        "scene_relative_improvements": json.dumps(
            row["scene_relative_improvements"], sort_keys=True
        ),
        "checks": json.dumps(row["checks"], sort_keys=True),
    } for row in gate_rows])
    summary = {
        "protocol": PROTOCOL,
        "status": "complete",
        "decision": decision,
        "selected_feature_arm": selected,
        "conditional_training_authorized": False,
        "git_sha": git_sha(ROOT),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path.resolve()),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "l269_status": l269_summary["status"],
        "neighbor_summary": neighbor_summary,
        "recovery_chain_splits": split_chains,
        "sample_rows": len(recovery_rows),
        "feature_dimensions": feature_dimensions,
        "dataset_content_fingerprint": integrity["dataset_content_fingerprint"],
        "gate_by_arm": gate_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/research_platform/rl/l270_observation_aliasing_diagnosis",
    )
    args = parser.parse_args(argv)
    run(args.config.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
