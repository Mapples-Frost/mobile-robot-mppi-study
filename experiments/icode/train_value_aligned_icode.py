#!/usr/bin/env python3
"""Fine-tune a validated ICODE checkpoint with a frozen SAC value objective."""

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.learning.models import load_platform_checkpoint
from mobile_robot_mppi.learning.trainer import contiguous_windows, rollout_step_weights
from mobile_robot_mppi.learning.value_alignment import (
    FrozenDirectSACValue,
    ValueAlignedResidualObjective,
)


def _resolve(path):
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_split(directory, name):
    path = Path(directory) / (name + ".npz")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    required = (
        "episode_id",
        "step",
        "dt",
        "state_t",
        "control_t",
        "state_t_plus_1",
        "residual_target",
        "raw_observation_t_plus_1",
        "target_position_t_plus_1",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError("%s split is missing %s" % (name, ", ".join(missing)))
    return result


def _batch(dataset, starts, horizon, device):
    offsets = np.arange(horizon, dtype=np.int64)[None, :]
    indices = np.asarray(starts, dtype=np.int64)[:, None] + offsets
    tensor = lambda values: torch.as_tensor(
        values, dtype=torch.float32, device=device
    )
    result = {
        "initial_state": tensor(dataset["state_t"][starts]),
        "states_t": tensor(dataset["state_t"][indices]),
        "controls": tensor(dataset["control_t"][indices]),
        "dt": tensor(dataset["dt"][indices]),
        "target_states": tensor(dataset["state_t_plus_1"][indices]),
        "residual_targets": tensor(dataset["residual_target"][indices]),
        "raw_observations": tensor(
            dataset["raw_observation_t_plus_1"][indices]
        ),
        "target_positions": tensor(
            dataset["target_position_t_plus_1"][indices]
        ),
    }
    if "path_context_t_plus_1" in dataset:
        result["path_contexts"] = tensor(
            dataset["path_context_t_plus_1"][indices]
        )
    return result


def _rank_correlation(predicted, target):
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if predicted.size < 2 or np.std(predicted) <= 1e-12 or np.std(target) <= 1e-12:
        return 0.0
    predicted_rank = np.argsort(np.argsort(predicted, kind="mergesort"))
    target_rank = np.argsort(np.argsort(target, kind="mergesort"))
    return float(np.corrcoef(predicted_rank, target_rank)[0, 1])


def evaluate(model, objective, value_model, dataset, starts, horizon, device, batch_size):
    model.eval()
    sums = {}
    count = 0
    terminal_predicted = []
    terminal_true = []
    terminal_state_errors = []
    with torch.no_grad():
        for begin in range(0, len(starts), batch_size):
            selected = starts[begin : begin + batch_size]
            batch = _batch(dataset, selected, horizon, device)
            values = objective(model, batch)
            size = len(selected)
            count += size
            for name in (
                "total",
                "derivative",
                "one_step",
                "multistep",
                "value",
                "value_rmse",
                "value_ranking",
                "value_ranking_accuracy",
                "value_ranking_pair_fraction",
                "anchor",
                "confidence_mean",
                "competence_mean",
            ):
                sums[name] = sums.get(name, 0.0) + float(
                    values[name].cpu()
                ) * size
            trajectory = values["predicted_trajectory"]
            raw_terminal = batch["raw_observations"][:, -1]
            target_position = batch["target_positions"][:, -1]
            predicted_value = value_model.value_from_state(
                trajectory[:, -1],
                raw_terminal,
                target_position,
                reference_state=batch["target_states"][:, -1],
                path_context_template=(
                    batch.get("path_contexts")[:, -1]
                    if "path_contexts" in batch
                    else None
                ),
            )
            true_value = value_model.value_from_raw(raw_terminal)
            terminal_predicted.extend(predicted_value.cpu().numpy().tolist())
            terminal_true.extend(true_value.cpu().numpy().tolist())
            error = trajectory[:, -1] - batch["target_states"][:, -1]
            error[:, 2] = torch.atan2(torch.sin(error[:, 2]), torch.cos(error[:, 2]))
            terminal_state_errors.append(error.cpu().numpy())
    if count <= 0:
        raise ValueError("evaluation requires at least one rollout window")
    metrics = {name: value / count for name, value in sums.items()}
    predicted = np.asarray(terminal_predicted, dtype=np.float64)
    target = np.asarray(terminal_true, dtype=np.float64)
    errors = np.concatenate(terminal_state_errors, axis=0)
    metrics.update({
        "terminal_value_rmse": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "terminal_value_rank_correlation": _rank_correlation(predicted, target),
        "terminal_position_rmse": float(
            np.sqrt(np.mean(np.sum(errors[:, :2] ** 2, axis=1)))
        ),
        "terminal_heading_rmse": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
        "rollout_rmse": float(math.sqrt(max(metrics["multistep"], 0.0))),
        "windows": int(count),
    })
    return metrics


def _value_scale(value_model, dataset, device, batch_size=1024):
    raw = dataset["raw_observation_t_plus_1"]
    values = []
    with torch.no_grad():
        for begin in range(0, len(raw), batch_size):
            tensor = torch.as_tensor(
                raw[begin : begin + batch_size],
                dtype=torch.float32,
                device=device,
            )
            values.append(value_model.value_from_raw(tensor).cpu().numpy())
    result = np.concatenate(values)
    scale = float(np.std(result))
    return max(scale, 1.0)


def _episode_outcomes(dataset_dir, split):
    path = Path(dataset_dir) / "episodes.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        if str(row["split"]) != str(split):
            continue
        episode_id = str(row["episode_id"])
        if episode_id in result:
            raise ValueError(
                "duplicate episode outcome for %s" % episode_id
            )
        result[episode_id] = bool(int(row["success"]))
    if not result:
        raise ValueError("no episode outcomes found for split %s" % split)
    return result


def _episode_bootstrap_windows(dataset, starts, seed):
    """Sample independent training episodes with replacement.

    Windows inside a selected episode remain correlated and are carried as one
    bootstrap cluster.  Sampling individual timesteps would create false
    independence and leak nearly identical trajectories across ensemble
    members.
    """

    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    if starts.size == 0:
        raise ValueError("episode bootstrap requires rollout windows")
    episode_at_start = np.asarray(dataset["episode_id"])[starts].astype(str)
    episode_ids = np.asarray(sorted(set(episode_at_start.tolist())))
    rng = np.random.RandomState(int(seed))
    selected_ids = rng.choice(
        episode_ids, size=len(episode_ids), replace=True
    )
    windows = np.concatenate([
        starts[episode_at_start == episode_id]
        for episode_id in selected_ids
    ])
    if windows.size == 0:
        raise RuntimeError("episode bootstrap produced no windows")
    counts = {
        str(episode_id): int(np.sum(selected_ids == episode_id))
        for episode_id in episode_ids
    }
    return windows.astype(np.int64, copy=False), {
        "enabled": True,
        "seed": int(seed),
        "independent_unit": "episode",
        "source_episode_count": int(len(episode_ids)),
        "draw_count": int(len(selected_ids)),
        "selected_episode_ids": selected_ids.tolist(),
        "selection_counts": counts,
        "window_count": int(len(windows)),
    }


def _calibrate_value_competence(
    value_model,
    dataset,
    outcomes,
    device,
    failure_quantile=0.5,
    success_quantile=0.5,
    batch_size=1024,
):
    """Calibrate critic authority from training-episode outcomes only."""

    failure_quantile = float(failure_quantile)
    success_quantile = float(success_quantile)
    if (
        not 0.0 <= failure_quantile <= 1.0
        or not 0.0 <= success_quantile <= 1.0
    ):
        raise ValueError("competence quantiles must lie in [0,1]")
    labels = np.asarray(
        [outcomes[str(value)] for value in dataset["episode_id"]],
        dtype=bool,
    )
    if not np.any(labels) or np.all(labels):
        raise ValueError(
            "competence calibration needs successful and failed episodes"
        )
    raw = dataset["raw_observation_t_plus_1"]
    values = []
    with torch.no_grad():
        for begin in range(0, len(raw), batch_size):
            tensor = torch.as_tensor(
                raw[begin : begin + batch_size],
                dtype=torch.float32,
                device=device,
            )
            values.append(value_model.value_from_raw(tensor).cpu().numpy())
    values = np.concatenate(values).astype(np.float64, copy=False)
    off_value = float(np.quantile(values[~labels], failure_quantile))
    on_value = float(np.quantile(values[labels], success_quantile))
    if not math.isfinite(off_value) or not math.isfinite(on_value):
        raise FloatingPointError(
            "critic competence calibration is non-finite"
        )
    if on_value <= off_value:
        raise ValueError(
            "successful critic values do not exceed failed values at the "
            "configured calibration quantiles"
        )
    return {
        "enabled": True,
        "source_split": "train",
        "independent_unit": "episode outcome",
        "failure_quantile": failure_quantile,
        "success_quantile": success_quantile,
        "off_value": off_value,
        "on_value": on_value,
        "success_transitions": int(np.sum(labels)),
        "failure_transitions": int(np.sum(~labels)),
        "success_episodes": int(sum(outcomes.values())),
        "failure_episodes": int(
            len(outcomes) - sum(outcomes.values())
        ),
        "success_value_mean": float(np.mean(values[labels])),
        "failure_value_mean": float(np.mean(values[~labels])),
    }


def _write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/icode/gate2_value_aligned_icode_l178.yaml"),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--value-weight", type=float)
    parser.add_argument("--anchor-weight", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("value-alignment training config must be a mapping")
    config = copy.deepcopy(config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.value_weight is not None:
        config.setdefault("loss_weights", {})["value"] = args.value_weight
    if args.anchor_weight is not None:
        config.setdefault("loss_weights", {})["anchor"] = args.anchor_weight
    if args.seed is not None:
        config["seed"] = args.seed

    seed = int(config.get("seed", 20260718))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(str(config.get("device", "cpu")))
    dataset_dir = _resolve(config["dataset_dir"])
    base_checkpoint = _resolve(config["base_icode_checkpoint"])
    actor_checkpoint = _resolve(config["actor_checkpoint"])
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _load_split(dataset_dir, "train")
    validation = _load_split(dataset_dir, "validation")
    test = _load_split(dataset_dir, "test")
    unseen = _load_split(dataset_dir, "unseen")
    model, base_payload = load_platform_checkpoint(base_checkpoint, device)
    anchor_model = copy.deepcopy(model)
    value_model = FrozenDirectSACValue.from_checkpoint(
        actor_checkpoint,
        device=device,
        critic_source=str(config.get("critic_source", "target")),
        support_soft_z=float(config.get("support_soft_z", 3.0)),
        support_hard_z=float(config.get("support_hard_z", 7.0)),
        disagreement_scale=float(config.get("disagreement_scale", 1.0)),
    )
    horizon = int(config.get("rollout_horizon", 10))
    if horizon <= 0:
        raise ValueError("rollout_horizon must be positive")
    training = dict(config.get("training", {}))
    horizon_weights = rollout_step_weights(
        {
            "rollout_step_weighting": training.get(
                "rollout_step_weighting", "linear"
            ),
            "rollout_terminal_weight": training.get(
                "rollout_terminal_weight", 4.0
            ),
        },
        horizon,
    )
    calibrated_value_scale = (
        _value_scale(value_model, train, device)
        if config.get("value_scale", "auto") == "auto"
        else float(config["value_scale"])
    )
    competence_config = dict(config.get("value_competence", {}))
    if bool(competence_config.get("enabled", False)):
        competence_calibration = _calibrate_value_competence(
            value_model,
            train,
            _episode_outcomes(dataset_dir, "train"),
            device,
            failure_quantile=float(
                competence_config.get("failure_quantile", 0.5)
            ),
            success_quantile=float(
                competence_config.get("success_quantile", 0.5)
            ),
        )
    else:
        competence_calibration = {"enabled": False}
    weights = dict(config.get("loss_weights", {}))
    objective = ValueAlignedResidualObjective(
        anchor_model,
        value_model,
        base_payload["training_config"]["dynamics"],
        derivative_weight=float(weights.get("derivative", 0.25)),
        one_step_weight=float(weights.get("one_step", 1.0)),
        multistep_weight=float(weights.get("multistep", 4.0)),
        value_weight=float(weights.get("value", 0.05)),
        value_ranking_weight=float(weights.get("value_ranking", 0.0)),
        value_ranking_margin=float(
            training.get("value_ranking_margin", 0.05)
        ),
        value_ranking_temperature=float(
            training.get("value_ranking_temperature", 0.10)
        ),
        anchor_weight=float(weights.get("anchor", 1.0)),
        value_scale=calibrated_value_scale,
        competence_off_value=competence_calibration.get("off_value"),
        competence_on_value=competence_calibration.get("on_value"),
        state_weights=training.get(
            "state_loss_weights", (4.0, 4.0, 2.0, 1.0, 1.0)
        ),
        horizon_weights=horizon_weights,
    ).to(device)
    windows = {
        name: contiguous_windows(values, horizon)
        for name, values in (
            ("train", train),
            ("validation", validation),
            ("test", test),
            ("unseen", unseen),
        )
    }
    if any(values.size == 0 for values in windows.values()):
        raise ValueError("every value-alignment split needs contiguous windows")
    bootstrap_config = dict(config.get("episode_bootstrap", {}))
    if bool(bootstrap_config.get("enabled", False)):
        training_windows, bootstrap_manifest = (
            _episode_bootstrap_windows(
                train,
                windows["train"],
                int(bootstrap_config.get("seed", seed)),
            )
        )
    else:
        training_windows = windows["train"]
        bootstrap_manifest = {
            "enabled": False,
            "independent_unit": "episode",
            "window_count": int(len(training_windows)),
        }
    batch_size = int(training.get("batch_size", 128))
    validation_batch_size = int(training.get("validation_batch_size", 256))
    maximum_train_windows = int(
        training.get("maximum_train_windows_per_epoch", len(windows["train"]))
    )
    validation_windows = windows["validation"][
        : int(training.get("maximum_validation_windows", len(windows["validation"])))
    ]
    baseline_validation = evaluate(
        model,
        objective,
        value_model,
        validation,
        validation_windows,
        horizon,
        device,
        validation_batch_size,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-5)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(training.get("epochs", 20))
    maximum_rollout_degradation = float(
        training.get("maximum_rollout_degradation", 0.03)
    )
    minimum_value_improvement = float(
        training.get("minimum_value_improvement", 0.0)
    )
    selection_metric = str(
        training.get("selection_metric", "terminal_value_rmse")
    )
    if selection_metric not in (
        "terminal_value_rmse", "terminal_value_rank_correlation"
    ):
        raise ValueError("unsupported value-alignment selection_metric")
    maximum_value_degradation = float(
        training.get("maximum_value_rmse_degradation", 0.05)
    )
    minimum_rank_improvement = float(
        training.get("minimum_rank_improvement", 0.0)
    )
    best_score = float(baseline_validation[selection_metric])
    best_epoch = 0
    best_eligible = True
    history = []

    def save_checkpoint(path, epoch, metrics):
        payload = copy.deepcopy(base_payload)
        payload["model_state"] = model.state_dict()
        payload["optimizer_state"] = optimizer.state_dict()
        payload["epoch"] = int(epoch)
        payload["git_sha"] = git_sha(ROOT)
        payload["value_alignment"] = {
            "format": "critic_informed_icode_v1",
            "base_icode_checkpoint": str(base_checkpoint),
            "base_icode_sha256": _sha256(base_checkpoint),
            "actor_checkpoint": str(actor_checkpoint),
            "actor_checkpoint_sha256": _sha256(actor_checkpoint),
            "dataset_dir": str(dataset_dir),
            "dataset_manifest_sha256": _sha256(
                dataset_dir / "dataset_manifest.json"
            ),
            "config": config,
            "value_scale": calibrated_value_scale,
            "value_competence": competence_calibration,
            "baseline_validation": baseline_validation,
            "selected_validation": metrics,
            "selected_epoch": int(epoch),
            "episode_bootstrap": bootstrap_manifest,
        }
        torch.save(payload, path)

    save_checkpoint(output_dir / "best.pt", 0, baseline_validation)
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.RandomState(seed + epoch).permutation(
            training_windows
        )
        order = order[:maximum_train_windows]
        accumulated = {}
        seen = 0
        for begin in range(0, len(order), batch_size):
            selected = order[begin : begin + batch_size]
            batch = _batch(train, selected, horizon, device)
            values = objective(model, batch)
            optimizer.zero_grad(set_to_none=True)
            values["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training.get("gradient_clip_norm", 1.0)),
                error_if_nonfinite=True,
            )
            optimizer.step()
            size = len(selected)
            seen += size
            for name in (
                "total",
                "derivative",
                "one_step",
                "multistep",
                "value",
                "value_rmse",
                "value_ranking",
                "value_ranking_accuracy",
                "value_ranking_pair_fraction",
                "anchor",
                "confidence_mean",
                "competence_mean",
            ):
                accumulated[name] = accumulated.get(name, 0.0) + float(
                    values[name].detach().cpu()
                ) * size
        train_metrics = {
            name: value / max(seen, 1) for name, value in accumulated.items()
        }
        validation_metrics = evaluate(
            model,
            objective,
            value_model,
            validation,
            validation_windows,
            horizon,
            device,
            validation_batch_size,
        )
        rollout_limit = (
            baseline_validation["rollout_rmse"]
            * (1.0 + maximum_rollout_degradation)
        )
        eligible = bool(validation_metrics["rollout_rmse"] <= rollout_limit)
        value_improvement = (
            baseline_validation["terminal_value_rmse"]
            - validation_metrics["terminal_value_rmse"]
        ) / max(baseline_validation["terminal_value_rmse"], 1e-12)
        rank_improvement = (
            validation_metrics["terminal_value_rank_correlation"]
            - baseline_validation["terminal_value_rank_correlation"]
        )
        value_degradation = (
            validation_metrics["terminal_value_rmse"]
            - baseline_validation["terminal_value_rmse"]
        ) / max(baseline_validation["terminal_value_rmse"], 1e-12)
        if selection_metric == "terminal_value_rmse":
            selected = bool(
                eligible
                and value_improvement >= minimum_value_improvement
                and validation_metrics[selection_metric] < best_score
            )
        else:
            selected = bool(
                eligible
                and value_degradation <= maximum_value_degradation
                and rank_improvement >= minimum_rank_improvement
                and validation_metrics[selection_metric] > best_score
            )
        if selected:
            best_score = validation_metrics[selection_metric]
            best_epoch = epoch
            best_eligible = eligible
            save_checkpoint(output_dir / "best.pt", epoch, validation_metrics)
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
            },
            output_dir / "last_training_state.pt",
        )
        record = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_value_rmse": train_metrics["value_rmse"],
            "train_value_ranking": train_metrics["value_ranking"],
            "train_value_ranking_accuracy": train_metrics[
                "value_ranking_accuracy"
            ],
            "train_competence_mean": train_metrics[
                "competence_mean"
            ],
            "train_rollout_rmse": math.sqrt(
                max(train_metrics["multistep"], 0.0)
            ),
            "validation_value_rmse": validation_metrics["terminal_value_rmse"],
            "validation_competence_mean": validation_metrics[
                "competence_mean"
            ],
            "validation_value_rank_correlation": validation_metrics[
                "terminal_value_rank_correlation"
            ],
            "validation_value_ranking": validation_metrics[
                "value_ranking"
            ],
            "validation_value_ranking_accuracy": validation_metrics[
                "value_ranking_accuracy"
            ],
            "validation_rollout_rmse": validation_metrics["rollout_rmse"],
            "validation_position_rmse": validation_metrics[
                "terminal_position_rmse"
            ],
            "validation_anchor": validation_metrics["anchor"],
            "eligible": int(eligible),
            "selected": int(selected),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    best_model, best_payload = load_platform_checkpoint(
        output_dir / "best.pt", device
    )
    evaluations = {}
    for name, dataset in (
        ("validation", validation),
        ("test", test),
        ("unseen", unseen),
    ):
        selected_windows = windows[name][
            : int(training.get("maximum_evaluation_windows", len(windows[name])))
        ]
        base_metrics = evaluate(
            anchor_model,
            objective,
            value_model,
            dataset,
            selected_windows,
            horizon,
            device,
            validation_batch_size,
        )
        aligned_metrics = evaluate(
            best_model,
            objective,
            value_model,
            dataset,
            selected_windows,
            horizon,
            device,
            validation_batch_size,
        )
        evaluations[name] = {
            "base": base_metrics,
            "value_aligned": aligned_metrics,
            "relative_value_rmse_reduction": (
                base_metrics["terminal_value_rmse"]
                - aligned_metrics["terminal_value_rmse"]
            ) / max(base_metrics["terminal_value_rmse"], 1e-12),
            "relative_rollout_rmse_change": (
                aligned_metrics["rollout_rmse"]
                - base_metrics["rollout_rmse"]
            ) / max(base_metrics["rollout_rmse"], 1e-12),
            "value_rank_correlation_change": (
                aligned_metrics["terminal_value_rank_correlation"]
                - base_metrics["terminal_value_rank_correlation"]
            ),
        }
    gate = {
        "best_epoch": int(best_epoch),
        "selected_fine_tuned_checkpoint": bool(best_epoch > 0),
        "rollout_constraint_satisfied": bool(best_eligible),
        "test_value_improved": bool(
            evaluations["test"]["relative_value_rmse_reduction"] > 0.0
        ),
        "unseen_value_improved": bool(
            evaluations["unseen"]["relative_value_rmse_reduction"] > 0.0
        ),
        "test_value_rank_improved": bool(
            evaluations["test"]["value_rank_correlation_change"] > 0.0
        ),
        "unseen_value_rank_improved": bool(
            evaluations["unseen"]["value_rank_correlation_change"] > 0.0
        ),
        "test_rollout_within_limit": bool(
            evaluations["test"]["relative_rollout_rmse_change"]
            <= maximum_rollout_degradation
        ),
        "unseen_rollout_within_limit": bool(
            evaluations["unseen"]["relative_rollout_rmse_change"]
            <= maximum_rollout_degradation
        ),
    }
    if selection_metric == "terminal_value_rank_correlation":
        required_gate_keys = (
            "selected_fine_tuned_checkpoint",
            "rollout_constraint_satisfied",
            "test_value_rank_improved",
            "unseen_value_rank_improved",
            "test_rollout_within_limit",
            "unseen_rollout_within_limit",
        )
    else:
        required_gate_keys = (
            "selected_fine_tuned_checkpoint",
            "rollout_constraint_satisfied",
            "test_value_improved",
            "unseen_value_improved",
            "test_rollout_within_limit",
            "unseen_rollout_within_limit",
        )
    gate["offline_gate_passed"] = bool(
        all(gate[name] for name in required_gate_keys)
    )
    gate["selection_metric"] = selection_metric
    summary = {
        "schema_version": 1,
        "git_sha": git_sha(ROOT),
        "config_path": str(config_path),
        "config": config,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": _sha256(base_checkpoint),
        "actor_checkpoint": str(actor_checkpoint),
        "actor_checkpoint_sha256": _sha256(actor_checkpoint),
        "dataset_manifest_sha256": _sha256(dataset_dir / "dataset_manifest.json"),
        "value_scale": calibrated_value_scale,
        "value_competence": competence_calibration,
        "episode_bootstrap": bootstrap_manifest,
        "baseline_validation": baseline_validation,
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(output_dir / "best.pt"),
        "best_checkpoint_sha256": _sha256(output_dir / "best.pt"),
        "evaluations": evaluations,
        "gate": gate,
    }
    _write_json(output_dir / "training_summary.json", summary)
    with (output_dir / "training_log.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
