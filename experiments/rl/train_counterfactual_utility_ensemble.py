#!/usr/bin/env python3
"""Train and calibrate the preregistered L23 utility ensemble."""

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from learning.checkpointing import atomic_torch_save
from mobile_robot_mppi.core.config import config_hash, git_sha, load_yaml
from mobile_robot_mppi.rl.utility_dataset import (
    load_counterfactual_utility_dataset,
    utility_dataset_summary,
)
from mobile_robot_mppi.rl.utility_model import (
    CounterfactualUtilityConfig,
    CounterfactualUtilityMLP,
    Standardizer,
    effect_group_counts,
    ensemble_statistics,
    fit_ridge,
    group_bootstrap_indices,
    group_conformal_multiplier,
    group_lcb_coverage,
    lower_confidence_bound,
    meaningful_sign_metrics,
    predict_ridge,
    regression_metrics,
    unique_group_tuples,
)


def _resolve(path):
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def _device(name):
    value = str(name)
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _predict(model, features, device, batch_size=4096):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            tensor = torch.as_tensor(
                features[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model(tensor).cpu().numpy())
    result = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("utility prediction contains NaN or Inf")
    return result


def _group_mean_metrics(target, prediction, groups):
    rows = []
    for key in unique_group_tuples(groups):
        mask = (groups[:, 0] == key[0]) & (groups[:, 1] == key[1])
        rows.append((float(np.mean(target[mask])), float(np.mean(prediction[mask]))))
    return regression_metrics(
        np.asarray([row[0] for row in rows]),
        np.asarray([row[1] for row in rows]),
    )


def _metric_bundle(target, prediction, groups, meaningful_effect):
    return {
        "row": regression_metrics(target, prediction),
        "group_mean": _group_mean_metrics(target, prediction, groups),
        "meaningful_sign": meaningful_sign_metrics(
            target, prediction, threshold=meaningful_effect
        ),
    }


def _train_member(
    member_index,
    seed,
    config,
    train_features,
    train_targets,
    train_groups,
    selection_features,
    selection_targets_raw,
    target_standardizer,
    device,
):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.RandomState(int(seed))
    model = CounterfactualUtilityMLP(
        train_features.shape[1], config.hidden_sizes, config.activation
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.scheduler_factor),
        patience=int(config.scheduler_patience),
        min_lr=float(config.minimum_learning_rate),
    )
    criterion = torch.nn.HuberLoss(delta=float(config.huber_delta))
    bootstrap = group_bootstrap_indices(train_groups, rng)
    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    records = []
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        order = bootstrap[rng.permutation(bootstrap.size)]
        batch_losses = []
        for start in range(0, order.size, int(config.batch_size)):
            indices = order[start:start + int(config.batch_size)]
            x = torch.as_tensor(
                train_features[indices], dtype=torch.float32, device=device
            )
            y = torch.as_tensor(
                train_targets[indices], dtype=torch.float32, device=device
            )
            prediction = model(x)
            loss = criterion(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("utility training loss is NaN or Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.gradient_clip_norm)
            )
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        selection_normalized = _predict(
            model, selection_features, device
        )
        selection_raw = target_standardizer.inverse(selection_normalized)
        selection_rmse = regression_metrics(
            selection_targets_raw, selection_raw
        )["rmse"]
        scheduler.step(selection_rmse)
        improved = selection_rmse < best_rmse - 1.0e-9
        if improved:
            best_rmse = float(selection_rmse)
            best_epoch = int(epoch)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        records.append({
            "member": int(member_index),
            "seed": int(seed),
            "epoch": int(epoch),
            "train_huber": float(np.mean(batch_losses)),
            "selection_rmse_m": float(selection_rmse),
            "best_selection_rmse_m": float(best_rmse),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "bootstrap_rows": int(bootstrap.size),
            "bootstrap_unique_rows": int(np.unique(bootstrap).size),
        })
        if stale >= int(config.early_stopping_patience):
            break
    if best_state is None:
        raise RuntimeError("utility member did not produce a best checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "member": int(member_index),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_selection_rmse_m": float(best_rmse),
        "parameter_count": model.parameter_count,
        "bootstrap_rows": int(bootstrap.size),
        "bootstrap_groups": len(unique_group_tuples(train_groups)),
    }, records


def _development_gate(
    config,
    summary,
    selection_metrics,
    calibration_gate,
):
    selection = summary["splits"]["selection"]
    calibration = summary["splits"]["calibration"]
    train = summary["splits"]["train"]
    checks = {
        "enough_train_rows": train["rows"] >= config.minimum_train_rows,
        "enough_selection_rows": selection["rows"]
        >= config.minimum_selection_rows,
        "enough_calibration_rows": calibration["rows"]
        >= config.minimum_calibration_rows,
        "selection_positive_effect_groups": selection[
            "positive_effect_groups"
        ] >= config.minimum_effect_groups_per_sign,
        "selection_negative_effect_groups": selection[
            "negative_effect_groups"
        ] >= config.minimum_effect_groups_per_sign,
        "calibration_positive_effect_groups": calibration[
            "positive_effect_groups"
        ] >= config.minimum_effect_groups_per_sign,
        "calibration_negative_effect_groups": calibration[
            "negative_effect_groups"
        ] >= config.minimum_effect_groups_per_sign,
        "all_checkpoints_in_all_splits": all(
            len(summary["splits"][name]["training_seeds"]) == 3
            for name in ("train", "selection", "calibration")
        ),
        "ensemble_beats_zero_rmse": selection_metrics["ensemble"]["row"][
            "rmse"
        ] <= (
            1.0 - config.minimum_zero_rmse_improvement_fraction
        ) * selection_metrics["zero"]["row"]["rmse"],
        "ensemble_not_worse_than_ridge": selection_metrics["ensemble"][
            "row"
        ]["rmse"] <= config.maximum_ridge_rmse_ratio * selection_metrics[
            "ridge"
        ]["row"]["rmse"],
        "meaningful_sign_accuracy": selection_metrics["ensemble"][
            "meaningful_sign"
        ]["balanced_accuracy"]
        >= config.minimum_meaningful_sign_balanced_accuracy,
        "calibration_acceptance_nontrivial": calibration_gate[
            "accept_fraction"
        ] >= config.minimum_calibration_accept_fraction,
        "calibration_accepted_mean_is_meaningful": (
            calibration_gate["accepted_mean_true_utility_m"] is not None
            and calibration_gate["accepted_mean_true_utility_m"]
            >= config.minimum_calibration_accepted_mean_utility_m
        ),
        "calibration_has_no_harmful_acceptance": calibration_gate[
            "harmful_accepted_rows"
        ] == 0,
    }
    if config.minimum_state_ablation_rmse_improvement_fraction > 0.0:
        checks["trajectory_features_beat_state_ablation"] = (
            "state_ensemble" in selection_metrics
            and selection_metrics["ensemble"]["row"]["rmse"]
            <= (
                1.0
                - config.minimum_state_ablation_rmse_improvement_fraction
            )
            * selection_metrics["state_ensemble"]["row"]["rmse"]
        )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "decision": (
            "eligible_to_open_sealed_test"
            if all(checks.values())
            else "retain_bc_fallback_and_keep_test_sealed"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    resolved_config = load_yaml(_resolve(args.config))
    config = CounterfactualUtilityConfig.from_mapping(
        resolved_config.get("rl", {}).get("counterfactual_utility", {})
    )
    device = _device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(int(resolved_config.get("rl", {}).get(
            "torch_num_threads", 1
        )))
    dataset = load_counterfactual_utility_dataset(
        _resolve(args.dataset_dir),
        config.model_selection_episode_seeds,
        config.calibration_episode_seeds,
    )
    summary = utility_dataset_summary(
        dataset, meaningful_effect=config.meaningful_effect_m
    )
    splits = dataset["splits"]
    feature_standardizer = Standardizer.fit(splits["train"]["features"])
    target_standardizer = Standardizer.fit(splits["train"]["targets"])
    standardized_features = {
        name: feature_standardizer.transform(split["features"])
        for name, split in splits.items()
    }
    standardized_train_targets = target_standardizer.transform(
        splits["train"]["targets"]
    )
    state_feature_dim = int(
        dataset["feature_schema"].get(
            "state_feature_dim", dataset["feature_dim"]
        )
    )
    if not 1 <= state_feature_dim <= int(dataset["feature_dim"]):
        raise ValueError("state feature ablation dimension is invalid")
    has_state_ablation = state_feature_dim < int(dataset["feature_dim"])
    if has_state_ablation:
        state_feature_standardizer = Standardizer.fit(
            splits["train"]["features"][:, :state_feature_dim]
        )
        standardized_state_features = {
            name: state_feature_standardizer.transform(
                split["features"][:, :state_feature_dim]
            )
            for name, split in splits.items()
        }
    else:
        state_feature_standardizer = None
        standardized_state_features = None
    ridge_weights = fit_ridge(
        standardized_features["train"],
        splits["train"]["targets"],
        regularization=config.ridge_lambda,
    )

    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models = []
    member_summaries = []
    training_records = []
    for member_index, seed in enumerate(config.ensemble_seeds):
        model, member_summary, records = _train_member(
            member_index,
            seed,
            config,
            standardized_features["train"],
            standardized_train_targets,
            splits["train"]["groups"],
            standardized_features["selection"],
            splits["selection"]["targets"],
            target_standardizer,
            device,
        )
        models.append(model)
        member_summary["feature_set"] = "trajectory_augmented"
        member_summaries.append(member_summary)
        for record in records:
            record["feature_set"] = "trajectory_augmented"
        training_records.extend(records)
    state_models = []
    state_member_summaries = []
    if has_state_ablation:
        for member_index, seed in enumerate(config.ensemble_seeds):
            model, member_summary, records = _train_member(
                member_index,
                seed,
                config,
                standardized_state_features["train"],
                standardized_train_targets,
                splits["train"]["groups"],
                standardized_state_features["selection"],
                splits["selection"]["targets"],
                target_standardizer,
                device,
            )
            state_models.append(model)
            member_summary["feature_set"] = "state_only"
            state_member_summaries.append(member_summary)
            for record in records:
                record["feature_set"] = "state_only"
            training_records.extend(records)
    _write_csv(output / "training.csv", training_records)

    predictions = {}
    metrics = {}
    ensemble_details = {}
    for split_name, split in splits.items():
        member_predictions = []
        for model in models:
            normalized = _predict(
                model, standardized_features[split_name], device
            )
            member_predictions.append(target_standardizer.inverse(normalized))
        member_predictions = np.stack(member_predictions)
        mean, epistemic, scale = ensemble_statistics(
            member_predictions, config.uncertainty_floor_m
        )
        zero = np.zeros_like(mean)
        ridge = predict_ridge(
            standardized_features[split_name], ridge_weights
        )
        predictions[split_name] = {
            "members": member_predictions,
            "mean": mean,
            "epistemic_std": epistemic,
            "scale": scale,
            "zero": zero,
            "ridge": ridge,
        }
        metrics[split_name] = {
            name: _metric_bundle(
                split["targets"], prediction, split["groups"],
                config.meaningful_effect_m,
            )
            for name, prediction in (
                ("zero", zero),
                ("ridge", ridge),
                ("ensemble", mean),
            )
        }
        if has_state_ablation:
            state_member_predictions = []
            for model in state_models:
                normalized = _predict(
                    model,
                    standardized_state_features[split_name],
                    device,
                )
                state_member_predictions.append(
                    target_standardizer.inverse(normalized)
                )
            state_mean = np.mean(
                np.stack(state_member_predictions), axis=0
            )
            predictions[split_name]["state_ensemble"] = state_mean
            metrics[split_name]["state_ensemble"] = _metric_bundle(
                split["targets"],
                state_mean,
                split["groups"],
                config.meaningful_effect_m,
            )
        ensemble_details[split_name] = {
            "epistemic_std_mean_m": float(np.mean(epistemic)),
            "epistemic_std_median_m": float(np.median(epistemic)),
            "epistemic_std_max_m": float(np.max(epistemic)),
        }

    calibration = group_conformal_multiplier(
        splits["calibration"]["targets"],
        predictions["calibration"]["mean"],
        predictions["calibration"]["scale"],
        splits["calibration"]["groups"],
        alpha=config.conformal_alpha,
    )
    calibration_lcb = lower_confidence_bound(
        predictions["calibration"]["mean"],
        predictions["calibration"]["scale"],
        calibration["multiplier"],
    )
    accepted = calibration_lcb > 0.0
    calibration_gate = {
        "rows": int(accepted.size),
        "accepted_rows": int(np.sum(accepted)),
        "accept_fraction": float(np.mean(accepted)),
        "accepted_mean_true_utility_m": (
            float(np.mean(splits["calibration"]["targets"][accepted]))
            if np.any(accepted) else None
        ),
        "accepted_min_true_utility_m": (
            float(np.min(splits["calibration"]["targets"][accepted]))
            if np.any(accepted) else None
        ),
        "harmful_accepted_rows": int(np.sum(
            accepted & (
                splits["calibration"]["targets"]
                <= config.harmful_accepted_utility_m
            )
        )),
        "empirical_group_coverage": group_lcb_coverage(
            splits["calibration"]["targets"],
            calibration_lcb,
            splits["calibration"]["groups"],
        ),
    }
    development_gate = _development_gate(
        config, summary, metrics["selection"], calibration_gate
    )
    result = {
        "phase": "counterfactual_utility_development",
        "git_sha": git_sha(ROOT),
        "config_hash": config_hash(resolved_config),
        "config": config.to_dict(),
        "dataset": summary,
        "member_summaries": member_summaries,
        "state_ablation": {
            "enabled": has_state_ablation,
            "state_feature_dim": state_feature_dim,
            "full_feature_dim": int(dataset["feature_dim"]),
            "member_summaries": state_member_summaries,
        },
        "metrics": metrics,
        "ensemble_details": ensemble_details,
        "calibration": {
            name: value
            for name, value in calibration.items()
            if name != "group_scores"
        },
        "calibration_gate": calibration_gate,
        "development_gate": development_gate,
        "sealed_test_opened": False,
        "interpretation_guard": (
            "branch rows are nested within checkpoint/episode groups; only a "
            "passing development gate may unlock sealed test collection"
        ),
    }
    payload = {
        "format": "counterfactual_utility_ensemble",
        "format_version": 1,
        "model_config": models[0].config_dict(),
        "member_state_dicts": [
            {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            }
            for model in models
        ],
        "feature_standardizer": feature_standardizer.state_dict(),
        "target_standardizer": target_standardizer.state_dict(),
        "ridge_weights": ridge_weights,
        "utility_config": config.to_dict(),
        "feature_schema": dataset["feature_schema"],
        "dataset_contract": dataset["dataset_contract"],
        "dataset_sha256": dataset["dataset_sha256"],
        "audit_sha256": dataset["audit_sha256"],
        "member_summaries": member_summaries,
        "state_ablation": (
            {
                "feature_dim": state_feature_dim,
                "feature_standardizer": (
                    state_feature_standardizer.state_dict()
                ),
                "member_state_dicts": [
                    {
                        name: tensor.detach().cpu()
                        for name, tensor in model.state_dict().items()
                    }
                    for model in state_models
                ],
                "member_summaries": state_member_summaries,
            }
            if has_state_ablation
            else None
        ),
        "calibration": result["calibration"],
        "calibration_gate": calibration_gate,
        "development_gate": development_gate,
        "git_sha": result["git_sha"],
        "config_hash": result["config_hash"],
    }
    checkpoint_path = atomic_torch_save(
        payload, output / "utility_ensemble.pt"
    )
    result["checkpoint"] = str(checkpoint_path)
    result["checkpoint_sha256"] = _sha256(checkpoint_path)
    with (output / "development_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    with (output / "config_snapshot.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(resolved_config, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
