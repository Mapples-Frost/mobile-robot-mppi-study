#!/usr/bin/env python3
"""Calibrate and seal Gate 3A reliability thresholds.

Threshold selection uses only episode-level validation data. Test and unseen
splits are loaded only after a single candidate has been selected. Overlapping
windows are retained as within-episode measurements and are never counted as
independent replications.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.learning.models import (
    PlatformResidualDynamics,
    PlatformResidualEnsemble,
)
from mobile_robot_mppi.planning.dynamics import (
    DynamicUnicyclePrediction,
    ResidualPrediction,
)
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint
from mobile_robot_mppi.rl.observation import RunningNormalizer
from mobile_robot_mppi.rl.reliability import (
    decreasing_linear_confidence,
    fuse_hybrid_confidence,
)


LEVELS = ("low", "medium", "high")


def _resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_split(directory, name):
    path = Path(directory) / ("%s.npz" % name)
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    required = (
        "episode_id",
        "step",
        "dt",
        "state_t",
        "control_t",
        "state_t_plus_1",
        "raw_observation_t_plus_1",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError("%s is missing %s" % (path, ", ".join(missing)))
    return result, path


def validate_actor_representation(dataset, normalizer, config, split_name):
    """Fail closed when calibration data do not match the frozen Actor."""

    raw = np.asarray(dataset["raw_observation_t_plus_1"])
    if (
        raw.ndim != 2
        or raw.shape[1] != int(normalizer.dimension)
        or not np.isfinite(raw).all()
    ):
        raise ValueError(
            "%s Actor observations must be finite [N,%d], got %s"
            % (split_name, normalizer.dimension, raw.shape)
        )
    result = {
        "samples": int(raw.shape[0]),
        "actor_observation_dimension": int(raw.shape[1]),
        "path_context_required": bool(
            config.get("require_path_context", False)
        ),
    }
    if not result["path_context_required"]:
        result.update({
            "path_context_present": "path_context_t_plus_1" in dataset,
            "path_valid_fraction": None,
        })
        return result
    if "path_context_t_plus_1" not in dataset:
        raise ValueError(
            "%s lacks path_context_t_plus_1 required by calibration"
            % split_name
        )
    context = np.asarray(
        dataset["path_context_t_plus_1"], dtype=np.float64
    )
    if (
        context.shape != (raw.shape[0], 6)
        or not np.isfinite(context).all()
    ):
        raise ValueError(
            "%s path context must be finite [N,6], got %s"
            % (split_name, context.shape)
        )
    valid = context[:, 5] >= 0.5
    if not np.all(valid):
        raise ValueError(
            "%s includes %d non-polyline samples in path calibration"
            % (split_name, int(np.sum(~valid)))
        )
    result.update({
        "path_context_present": True,
        "path_valid_fraction": float(np.mean(valid)),
        "path_signed_cross_track_range": [
            float(np.min(context[:, 0])),
            float(np.max(context[:, 0])),
        ],
        "path_curvature_range": [
            float(np.min(context[:, 3])),
            float(np.max(context[:, 3])),
        ],
        "path_remaining_range": [
            float(np.min(context[:, 4])),
            float(np.max(context[:, 4])),
        ],
    })
    return result


def _state_error(predicted, target):
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    error[..., 2] = np.arctan2(
        np.sin(error[..., 2]), np.cos(error[..., 2])
    )
    return error


def _integrate(dynamics, state, control, dt, method):
    state = np.asarray(state, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    dt = float(dt)
    if dt <= 0.0 or not np.isfinite(state).all():
        raise ValueError("integration requires finite state and positive dt")
    if method == "euler":
        result = state + dt * dynamics.derivative(state, control)
    elif method == "rk4":
        k1 = dynamics.derivative(state, control)
        k2 = dynamics.derivative(state + 0.5 * dt * k1, control)
        k3 = dynamics.derivative(state + 0.5 * dt * k2, control)
        k4 = dynamics.derivative(state + dt * k3, control)
        result = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    else:
        raise ValueError("unknown integrator %s" % method)
    result = np.asarray(result, dtype=np.float64)
    result[2] = math.atan2(math.sin(result[2]), math.cos(result[2]))
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite calibration rollout")
    return result


def _load_models(config):
    checkpoint_paths = [
        _resolve(value) for value in config["ensemble_checkpoints"]
    ]
    members = [
        PlatformResidualDynamics.from_checkpoint(
            path, device="cpu", use_torchscript=False
        )
        for path in checkpoint_paths
    ]
    ensemble_config = config["ensemble"]
    ensemble = PlatformResidualEnsemble(
        members,
        disagreement_scales=ensemble_config["disagreement_scales"],
        innovation_scales=ensemble_config["innovation_scales"],
        support_soft_z=ensemble_config["support_soft_z"],
        support_hard_z=ensemble_config["support_hard_z"],
        innovation_decay=ensemble_config["innovation_decay"],
        member_paths=checkpoint_paths,
    )
    nominal_config = config["nominal_dynamics"]
    nominal = DynamicUnicyclePrediction(
        nominal_config["velocity_time_constant"],
        nominal_config["yaw_time_constant"],
    )
    combined = ResidualPrediction(nominal, ensemble)
    actor_path = _resolve(config["actor_checkpoint"])
    actor_payload = load_sac_checkpoint(actor_path, map_location="cpu")
    normalizer = RunningNormalizer.from_state_dict(
        actor_payload["normalizer"]
    )
    return {
        "ensemble": ensemble,
        "nominal": nominal,
        "combined": combined,
        "normalizer": normalizer,
        "checkpoint_paths": checkpoint_paths,
        "actor_path": actor_path,
    }


def extract_window_signals(dataset, models, config, split_name):
    """Return window rows while updating innovation in strict causal order."""

    horizon = int(config["rollout_horizon"])
    method = str(config["integrator"])
    rollout_scales = np.asarray(
        config["rollout_error_scales"], dtype=np.float64
    )
    ensemble = models["ensemble"]
    combined = models["combined"]
    nominal = models["nominal"]
    normalizer = models["normalizer"]
    rows = []
    episode_ids = sorted(set(dataset["episode_id"].tolist()))
    for episode_id in episode_ids:
        indices = np.flatnonzero(dataset["episode_id"] == episode_id)
        order = np.argsort(dataset["step"][indices], kind="stable")
        indices = indices[order]
        observed_steps = dataset["step"][indices].astype(int)
        if not np.array_equal(
            observed_steps, np.arange(observed_steps.size)
        ):
            raise ValueError("episode %s is not contiguous" % episode_id)
        ensemble.reset()
        for local_index, data_index in enumerate(indices):
            remaining = indices.size - local_index
            if remaining >= horizon:
                window = indices[local_index : local_index + horizon]
                state = np.asarray(
                    dataset["state_t"][data_index], dtype=np.float64
                ).copy()
                predicted_states = []
                rollout_states = []
                controls = []
                for future_index in window:
                    control = np.asarray(
                        dataset["control_t"][future_index],
                        dtype=np.float64,
                    )
                    rollout_states.append(state.copy())
                    controls.append(control.copy())
                    state = _integrate(
                        combined,
                        state,
                        control,
                        float(dataset["dt"][future_index]),
                        method,
                    )
                    predicted_states.append(state.copy())
                predicted_states = np.asarray(predicted_states)
                rollout_states = np.asarray(rollout_states)
                controls = np.asarray(controls)
                targets = np.asarray(
                    dataset["state_t_plus_1"][window],
                    dtype=np.float64,
                )
                normalized_error = (
                    _state_error(predicted_states, targets)
                    / rollout_scales
                )
                step_error = np.sqrt(
                    np.mean(normalized_error ** 2, axis=-1)
                )
                disagreement = ensemble.disagreement(
                    rollout_states, controls
                )
                support = ensemble.support_confidence(
                    rollout_states, controls
                )
                raw_observations = np.asarray(
                    dataset["raw_observation_t_plus_1"][window],
                    dtype=np.float32,
                )
                actor_ood = np.max(
                    np.abs(normalizer.normalize(raw_observations)), axis=-1
                )
                path_context = (
                    np.asarray(
                        dataset["path_context_t_plus_1"][window],
                        dtype=np.float64,
                    )
                    if "path_context_t_plus_1" in dataset
                    else None
                )
                rows.append({
                    "split": split_name,
                    "episode_id": str(episode_id),
                    "seed": int(dataset["seed"][data_index]),
                    "scene": str(dataset["scene"][data_index]),
                    "physics_domain": str(
                        dataset["physics_domain"][data_index]
                    ),
                    "start_step": int(dataset["step"][data_index]),
                    "rollout_error": float(np.mean(step_error)),
                    "terminal_error": float(step_error[-1]),
                    "ensemble_disagreement_max": float(
                        np.max(disagreement)
                    ),
                    "residual_support_confidence_min": float(
                        np.min(support)
                    ),
                    "innovation_error_ema": float(
                        ensemble.innovation_error_ema
                    ),
                    "innovation_samples": int(
                        ensemble.innovation_samples
                    ),
                    "actor_ood_score_max": float(np.max(actor_ood)),
                    "path_cross_track_abs_max": (
                        float(np.max(np.abs(path_context[:, 0])))
                        if path_context is not None else 0.0
                    ),
                    "path_curvature_abs_max": (
                        float(np.max(np.abs(path_context[:, 3])))
                        if path_context is not None else 0.0
                    ),
                    "path_remaining_mean": (
                        float(np.mean(path_context[:, 4]))
                        if path_context is not None else 0.0
                    ),
                })

            # This completed transition becomes available only after the
            # current decision/window signals have been recorded.
            current_state = np.asarray(
                dataset["state_t"][data_index], dtype=np.float64
            )
            current_control = np.asarray(
                dataset["control_t"][data_index], dtype=np.float64
            )
            target = np.asarray(
                dataset["state_t_plus_1"][data_index], dtype=np.float64
            )
            dt = float(dataset["dt"][data_index])
            nominal_prediction = _integrate(
                nominal, current_state, current_control, dt, method
            )
            combined_prediction = _integrate(
                combined, current_state, current_control, dt, method
            )
            ensemble.observe_prediction_errors(
                _state_error(nominal_prediction, target),
                _state_error(combined_prediction, target),
            )
    return rows


def _rank_correlation(x_values, y_values):
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while (
            end < values.size
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _continuous_rank_gate(episodes, grid):
    authorities = np.asarray(
        [row["authority"] for row in episodes], dtype=np.float64
    )
    errors = np.asarray(
        [row["rollout_error"] for row in episodes], dtype=np.float64
    )
    count = int(authorities.size)
    minimum_episodes = int(grid["minimum_episode_count"])
    tail_fraction = float(grid.get("tail_fraction", 0.25))
    minimum_tail = int(grid.get("minimum_tail_episodes", 3))
    tail_count = max(
        minimum_tail,
        int(math.ceil(tail_fraction * count)),
    )
    if 2 * tail_count > count:
        raise ValueError(
            "continuous reliability tails exceed episode count"
        )
    order = np.argsort(authorities, kind="mergesort")
    low_error = float(np.mean(errors[order[:tail_count]]))
    high_error = float(np.mean(errors[order[-tail_count:]]))
    relative_separation = float(
        (low_error - high_error) / max(abs(low_error), 1e-12)
    )
    rank = _rank_correlation(authorities, errors)
    bootstrap_samples = int(grid.get("bootstrap_samples", 5000))
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.RandomState(int(grid.get("bootstrap_seed", 0)))
    bootstrap = []
    for _ in range(bootstrap_samples):
        indices = rng.randint(0, count, size=count)
        bootstrap.append(_rank_correlation(
            authorities[indices], errors[indices]
        ))
    rank_ci = [
        float(np.percentile(bootstrap, 2.5)),
        float(np.percentile(bootstrap, 97.5)),
    ]
    passed = bool(
        count >= minimum_episodes
        and rank <= float(grid["maximum_rank_correlation"])
        and rank_ci[1] < float(grid.get("maximum_rank_ci_upper", 0.0))
        and relative_separation
        >= float(grid["minimum_relative_tail_separation"])
    )
    return {
        "passed": passed,
        "gate_mode": "continuous_rank",
        "episode_count": count,
        "authority_error_rank_correlation": rank,
        "authority_error_rank_correlation_ci95": rank_ci,
        "tail_fraction": tail_fraction,
        "tail_episode_count": tail_count,
        "low_authority_tail_error_mean": low_error,
        "high_authority_tail_error_mean": high_error,
        "error_separation": float(low_error - high_error),
        "relative_tail_error_separation": relative_separation,
        "minimum_episode_count": minimum_episodes,
        "maximum_rank_correlation": float(
            grid["maximum_rank_correlation"]
        ),
        "maximum_rank_ci_upper": float(
            grid.get("maximum_rank_ci_upper", 0.0)
        ),
        "minimum_relative_tail_separation": float(
            grid["minimum_relative_tail_separation"]
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": int(grid.get("bootstrap_seed", 0)),
    }


def _threshold_pairs(values, grid):
    values = np.asarray(values, dtype=np.float64)
    pairs = []
    for soft_q, hard_q in itertools.product(
        grid["soft_quantiles"], grid["hard_quantiles"]
    ):
        soft = float(np.quantile(values, float(soft_q)))
        hard = float(np.quantile(values, float(hard_q)))
        if hard > soft + 1e-12:
            pairs.append((float(soft_q), float(hard_q), soft, hard))
    if not pairs:
        raise ValueError("calibration scores do not span valid thresholds")
    return pairs


def apply_candidate(rows, candidate, config):
    output = []
    minimum_samples = int(
        config["ensemble"]["innovation_minimum_samples"]
    )
    for source in rows:
        disagreement_confidence = float(decreasing_linear_confidence(
            source["ensemble_disagreement_max"],
            candidate["ensemble_disagreement_soft"],
            candidate["ensemble_disagreement_hard"],
        ))
        innovation_ready = (
            int(source["innovation_samples"]) >= minimum_samples
        )
        innovation_confidence = (
            float(decreasing_linear_confidence(
                source["innovation_error_ema"],
                candidate["innovation_error_soft"],
                candidate["innovation_error_hard"],
            ))
            if innovation_ready else 1.0
        )
        actor_confidence = float(decreasing_linear_confidence(
            source["actor_ood_score_max"],
            config["actor_ood_soft"],
            config["actor_ood_hard"],
        ))
        dynamics_confidence, actor_authority_factor = (
            fuse_hybrid_confidence(
                disagreement_confidence,
                innovation_confidence,
                float(source["residual_support_confidence_min"]),
                actor_confidence,
                innovation_ready,
                config.get("fusion_mode", "conservative_min"),
            )
        )
        authority = float(np.clip(
            dynamics_confidence ** float(config["dynamics_power"])
            * actor_authority_factor,
            0.0,
            1.0,
        ))
        if authority >= float(config["high_confidence"]):
            level = "high"
        elif authority >= float(config["medium_confidence"]):
            level = "medium"
        else:
            level = "low"
        row = dict(source)
        row.update({
            "disagreement_confidence": disagreement_confidence,
            "innovation_confidence": innovation_confidence,
            "innovation_ready": innovation_ready,
            "dynamics_confidence": dynamics_confidence,
            "actor_confidence": actor_confidence,
            "actor_authority_factor": actor_authority_factor,
            "fusion_mode": str(
                config.get("fusion_mode", "conservative_min")
            ),
            "authority": authority,
            "level": level,
        })
        output.append(row)
    return output


def episode_level_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode_id"]].append(row)
    episodes = []
    for episode_id, values in sorted(grouped.items()):
        mean_authority = float(np.mean([
            row["authority"] for row in values
        ]))
        if mean_authority >= 0.67:
            level = "high"
        elif mean_authority >= 0.33:
            level = "medium"
        else:
            level = "low"
        episodes.append({
            "episode_id": episode_id,
            "split": values[0]["split"],
            "seed": values[0]["seed"],
            "scene": values[0]["scene"],
            "physics_domain": values[0]["physics_domain"],
            "authority": mean_authority,
            "level": level,
            "rollout_error": float(np.mean([
                row["rollout_error"] for row in values
            ])),
            "terminal_error": float(np.mean([
                row["terminal_error"] for row in values
            ])),
            "windows": len(values),
        })
    return episodes


def evaluate_episode_gate(episodes, grid):
    if str(grid.get("gate_mode", "legacy_bins")) == "continuous_rank":
        return _continuous_rank_gate(episodes, grid)
    level_rows = {
        level: [row for row in episodes if row["level"] == level]
        for level in LEVELS
    }
    occupied = [level for level in LEVELS if level_rows[level]]
    counts = {level: len(level_rows[level]) for level in LEVELS}
    errors = {
        level: (
            float(np.mean([
                row["rollout_error"] for row in level_rows[level]
            ]))
            if level_rows[level] else None
        )
        for level in LEVELS
    }
    ordered_errors = [
        errors[level] for level in ("high", "medium", "low")
        if errors[level] is not None
    ]
    monotone = all(
        ordered_errors[index] <= ordered_errors[index + 1] + 1e-12
        for index in range(len(ordered_errors) - 1)
    )
    rank = _rank_correlation(
        [row["authority"] for row in episodes],
        [row["rollout_error"] for row in episodes],
    )
    minimum_count = int(grid["minimum_episodes_per_occupied_bin"])
    minimum_bins = int(grid["minimum_occupied_bins"])
    passed = bool(
        len(occupied) >= minimum_bins
        and all(counts[level] >= minimum_count for level in occupied)
        and counts["low"] >= minimum_count
        and monotone
        and rank <= 0.0
    )
    separation = (
        max(value for value in errors.values() if value is not None)
        - min(value for value in errors.values() if value is not None)
        if len(occupied) >= 2 else 0.0
    )
    return {
        "passed": passed,
        "episode_count": len(episodes),
        "occupied_levels": occupied,
        "level_episode_counts": counts,
        "level_mean_rollout_error": errors,
        "monotone_high_to_low": monotone,
        "authority_error_rank_correlation": rank,
        "error_separation": float(separation),
    }


def select_candidate(validation_rows, config):
    grid = config["calibration_grid"]
    disagreement_pairs = _threshold_pairs(
        [row["ensemble_disagreement_max"] for row in validation_rows],
        grid,
    )
    ready_innovations = [
        row["innovation_error_ema"] for row in validation_rows
        if row["innovation_samples"]
        >= int(config["ensemble"]["innovation_minimum_samples"])
    ]
    innovation_pairs = _threshold_pairs(ready_innovations, grid)
    candidates = []
    for candidate_id, (disagreement, innovation) in enumerate(
        itertools.product(disagreement_pairs, innovation_pairs)
    ):
        candidate = {
            "candidate_id": candidate_id,
            "ensemble_disagreement_soft_quantile": disagreement[0],
            "ensemble_disagreement_hard_quantile": disagreement[1],
            "ensemble_disagreement_soft": disagreement[2],
            "ensemble_disagreement_hard": disagreement[3],
            "innovation_error_soft_quantile": innovation[0],
            "innovation_error_hard_quantile": innovation[1],
            "innovation_error_soft": innovation[2],
            "innovation_error_hard": innovation[3],
        }
        evaluated = apply_candidate(validation_rows, candidate, config)
        gate = evaluate_episode_gate(
            episode_level_summary(evaluated), grid
        )
        candidate["validation_gate"] = gate
        candidates.append(candidate)
    passed = [
        candidate for candidate in candidates
        if candidate["validation_gate"]["passed"]
    ]
    pool = passed if passed else candidates
    selected = min(pool, key=lambda item: (
        item["validation_gate"]["authority_error_rank_correlation"],
        -item["validation_gate"]["error_separation"],
        item["candidate_id"],
    ))
    return selected, candidates


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to write empty CSV")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def calibrate(config_path):
    config_path = _resolve(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = _resolve(config["dataset_dir"])
    models = _load_models(config)

    validation, validation_path = _load_split(
        dataset_dir, "validation"
    )
    representation_audit = {
        "validation": validate_actor_representation(
            validation, models["normalizer"], config, "validation"
        )
    }
    validation_signals = extract_window_signals(
        validation, models, config, "validation"
    )
    selected, candidates = select_candidate(
        validation_signals, config
    )
    validation_evaluated = apply_candidate(
        validation_signals, selected, config
    )
    validation_episodes = episode_level_summary(
        validation_evaluated
    )

    # Only now are protected evaluation files opened.
    split_results = {}
    artifact_hashes = {
        "validation": _sha256(validation_path),
    }
    all_windows = list(validation_evaluated)
    all_episodes = list(validation_episodes)
    for split_name in ("test", "unseen"):
        dataset, path = _load_split(dataset_dir, split_name)
        representation_audit[split_name] = (
            validate_actor_representation(
                dataset,
                models["normalizer"],
                config,
                split_name,
            )
        )
        artifact_hashes[split_name] = _sha256(path)
        models["ensemble"].reset()
        signals = extract_window_signals(
            dataset, models, config, split_name
        )
        evaluated = apply_candidate(signals, selected, config)
        episodes = episode_level_summary(evaluated)
        gate = evaluate_episode_gate(
            episodes, config["calibration_grid"]
        )
        split_results[split_name] = gate
        all_windows.extend(evaluated)
        all_episodes.extend(episodes)

    gate_passed = bool(
        selected["validation_gate"]["passed"]
        and all(result["passed"] for result in split_results.values())
    )
    selected_runtime_config = {
        "enabled": True,
        "ensemble_disagreement_soft": selected[
            "ensemble_disagreement_soft"
        ],
        "ensemble_disagreement_hard": selected[
            "ensemble_disagreement_hard"
        ],
        "innovation_error_soft": selected["innovation_error_soft"],
        "innovation_error_hard": selected["innovation_error_hard"],
        "innovation_minimum_samples": int(
            config["ensemble"]["innovation_minimum_samples"]
        ),
        "actor_ood_soft": float(config["actor_ood_soft"]),
        "actor_ood_hard": float(config["actor_ood_hard"]),
        "medium_confidence": float(config["medium_confidence"]),
        "high_confidence": float(config["high_confidence"]),
        "low_guided_fraction": 0.0,
        "medium_guided_fraction": 0.30,
        "high_guided_fraction": 0.60,
        "dynamics_power": float(config["dynamics_power"]),
        "fusion_mode": str(
            config.get("fusion_mode", "conservative_min")
        ),
    }
    summary = {
        "schema_version": 1,
        "gate": "Gate 3A reliability calibration",
        "gate_passed": gate_passed,
        "git_sha": git_sha(ROOT),
        "config_path": str(config_path),
        "config": config,
        "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
        "dataset_manifest_sha256": _sha256(
            dataset_dir / "dataset_manifest.json"
        ),
        "split_sha256": artifact_hashes,
        "ensemble_checkpoints": [
            {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for path in models["checkpoint_paths"]
        ],
        "actor_checkpoint": {
            "path": str(models["actor_path"]),
            "sha256": _sha256(models["actor_path"]),
        },
        "representation_audit": representation_audit,
        "selection_split": "validation",
        "selected_candidate": selected,
        "runtime_reliability_config": selected_runtime_config,
        "evaluation": split_results,
        "independent_unit": "episode",
        "actor_ood_offline_proxy": (
            "recorded post-transition Actor observations; runtime uses "
            "hypothetical Actor-mean-rollout observations"
        ),
        "candidate_count": len(candidates),
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "runtime_reliability.yaml").write_text(
        yaml.safe_dump(
            {"reliability": selected_runtime_config},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "window_signals.csv", all_windows)
    _write_csv(output_dir / "episode_calibration.csv", all_episodes)
    _write_csv(
        output_dir / "candidate_summary.csv",
        [
            {
                "candidate_id": item["candidate_id"],
                "validation_passed": item["validation_gate"]["passed"],
                "validation_rank_correlation": item[
                    "validation_gate"
                ]["authority_error_rank_correlation"],
                "validation_error_separation": item[
                    "validation_gate"
                ]["error_separation"],
                "disagreement_soft": item[
                    "ensemble_disagreement_soft"
                ],
                "disagreement_hard": item[
                    "ensemble_disagreement_hard"
                ],
                "innovation_soft": item["innovation_error_soft"],
                "innovation_hard": item["innovation_error_hard"],
            }
            for item in candidates
        ],
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = calibrate(args.config)
    print(json.dumps({
        "gate_passed": summary["gate_passed"],
        "selected_candidate": summary["selected_candidate"],
        "evaluation": summary["evaluation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
