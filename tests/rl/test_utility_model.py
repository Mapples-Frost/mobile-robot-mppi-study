import json

import numpy as np
import pytest
import torch

from experiments.rl.train_counterfactual_utility_ensemble import (
    _development_gate,
)
from mobile_robot_mppi.rl.utility_dataset import (
    load_counterfactual_utility_dataset,
    utility_dataset_summary,
)
from mobile_robot_mppi.rl.utility_model import (
    CounterfactualUtilityConfig,
    CounterfactualUtilityEnsemble,
    CounterfactualUtilityMLP,
    Standardizer,
    ensemble_statistics,
    fit_ridge,
    group_bootstrap_indices,
    group_conformal_multiplier,
    group_lcb_coverage,
    lower_confidence_bound,
    load_counterfactual_utility_ensemble,
    meaningful_sign_metrics,
    predict_ridge,
    regression_metrics,
)


def _config(**overrides):
    values = {
        "model_selection_episode_seeds": [3],
        "calibration_episode_seeds": [4],
    }
    values.update(overrides)
    return CounterfactualUtilityConfig.from_mapping(values)


def test_standardizer_and_mlp_have_stable_shapes():
    values = np.asarray([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32)
    standardizer = Standardizer.fit(values)
    transformed = standardizer.transform(values)
    np.testing.assert_allclose(transformed[:, 0], (-1.0, 1.0))
    assert standardizer.scale[1] == 1.0
    np.testing.assert_allclose(transformed[:, 1], (0.0, 0.0))
    np.testing.assert_allclose(standardizer.inverse(transformed), values)
    restored = Standardizer.from_state_dict(standardizer.state_dict())
    np.testing.assert_allclose(restored.transform(values), transformed)

    model = CounterfactualUtilityMLP(2, (4, 3), "silu")
    prediction = model(torch.as_tensor(transformed))
    assert prediction.shape == (2,)
    assert model.parameter_count > 0
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros((2, 3)))


def test_group_bootstrap_keeps_complete_episode_rows():
    groups = np.asarray([
        [1, 10], [1, 10], [1, 20], [1, 20], [1, 20]
    ])
    indices = group_bootstrap_indices(groups, np.random.RandomState(5))
    selected = groups[indices]
    count_a = int(np.sum((selected == (1, 10)).all(axis=1)))
    count_b = int(np.sum((selected == (1, 20)).all(axis=1)))
    assert count_a % 2 == 0
    assert count_b % 3 == 0
    assert count_a + count_b == indices.size


def test_ridge_and_metrics_recover_simple_linear_target():
    x = np.arange(12, dtype=np.float64).reshape(6, 2)
    y = 2.0 * x[:, 0] - x[:, 1] + 0.5
    weights = fit_ridge(x, y, regularization=0.0)
    prediction = predict_ridge(x, weights)
    metrics = regression_metrics(y, prediction)
    assert metrics["rmse"] < 1.0e-8
    sign = meaningful_sign_metrics(
        np.asarray((-0.1, 0.1)), np.asarray((-0.2, 0.2)), 0.03
    )
    assert sign["balanced_accuracy"] == 1.0


def test_group_conformal_uses_worst_row_per_group():
    target = np.asarray((0.0, -1.0, 0.0, -0.1))
    mean = np.zeros(4)
    scale = np.ones(4)
    groups = np.asarray(((1, 1), (1, 1), (1, 2), (1, 2)))
    calibration = group_conformal_multiplier(
        target, mean, scale, groups, alpha=0.5
    )
    assert calibration["groups"] == 2
    assert calibration["multiplier"] == pytest.approx(1.0)
    lcb = lower_confidence_bound(mean, scale, calibration["multiplier"])
    assert group_lcb_coverage(target, lcb, groups) == 1.0


def test_ensemble_statistics_apply_raw_unit_floor():
    predictions = np.asarray(((0.0, 1.0), (0.002, 1.004)))
    mean, epistemic, scale = ensemble_statistics(predictions, 0.005)
    np.testing.assert_allclose(mean, (0.001, 1.002))
    assert np.all(epistemic < 0.005)
    np.testing.assert_allclose(scale, (0.005, 0.005))


def test_inference_wrapper_and_checkpoint_fail_closed(tmp_path):
    models = [CounterfactualUtilityMLP(2, (3,), "silu") for _ in range(2)]
    feature_standardizer = Standardizer.fit(
        np.asarray(((0.0, 0.0), (1.0, 1.0)))
    )
    target_standardizer = Standardizer.fit(np.asarray((0.0, 1.0)))
    ensemble = CounterfactualUtilityEnsemble(
        models,
        feature_standardizer,
        target_standardizer,
        uncertainty_floor=0.005,
        conformal_multiplier=1.0,
    )
    prediction = ensemble.predict(np.asarray((0.5, 0.5)))
    assert set(prediction) == {
        "mean_utility_m",
        "epistemic_std_m",
        "calibration_scale_m",
        "lower_confidence_bound_m",
        "accept",
    }
    config = _config(
        ensemble_size=2,
        ensemble_seeds=[11, 12],
        hidden_sizes=[3],
    )
    payload = {
        "format": "counterfactual_utility_ensemble",
        "format_version": 1,
        "model_config": models[0].config_dict(),
        "member_state_dicts": [model.state_dict() for model in models],
        "feature_standardizer": feature_standardizer.state_dict(),
        "target_standardizer": target_standardizer.state_dict(),
        "utility_config": config.to_dict(),
        "calibration": {"multiplier": 1.0},
        "development_gate": {"passed": False},
    }
    path = tmp_path / "utility.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="failed"):
        load_counterfactual_utility_ensemble(path)
    loaded, restored = load_counterfactual_utility_ensemble(
        path, require_eligible=False
    )
    assert loaded.input_dim == 2
    assert restored["development_gate"]["passed"] is False


def test_dataset_loader_splits_validation_by_episode_seed(tmp_path):
    features = np.arange(18, dtype=np.float32).reshape(6, 3)
    np.savez_compressed(
        tmp_path / "samples_all_training_seeds.npz",
        features=features,
        labels=np.ones(6, dtype=np.int64),
        training_seed=np.asarray((1, 1, 1, 1, 1, 1)),
        episode_seed=np.asarray((1, 1, 3, 3, 4, 4)),
        branch_step=np.arange(6),
        split=np.asarray((0, 0, 1, 1, 1, 1), dtype=np.int8),
        return_delta=np.zeros(6),
        goal_distance_improvement=np.asarray(
            (-0.04, 0.05, -0.05, 0.06, -0.07, 0.08)
        ),
    )
    audit = {
        "quality_passed": True,
        "contract": {"feature_schema": {"feature_dim": 3}},
    }
    (tmp_path / "audit.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    dataset = load_counterfactual_utility_dataset(tmp_path, (3,), (4,))
    assert dataset["splits"]["train"]["features"].shape == (2, 3)
    assert dataset["splits"]["selection"]["targets"].shape == (2,)
    assert dataset["splits"]["calibration"]["targets"].shape == (2,)
    summary = utility_dataset_summary(dataset, meaningful_effect=0.03)
    assert summary["splits"]["selection"]["positive_effect_groups"] == 1
    assert summary["splits"]["selection"]["negative_effect_groups"] == 1


def test_utility_config_rejects_split_overlap_and_seed_mismatch():
    with pytest.raises(ValueError, match="invalid"):
        _config(calibration_episode_seeds=[3])
    with pytest.raises(ValueError, match="match"):
        _config(ensemble_size=3)


def test_development_gate_is_fail_closed():
    config = _config(
        minimum_train_rows=1,
        minimum_selection_rows=1,
        minimum_calibration_rows=1,
        minimum_effect_groups_per_sign=1,
    )
    summary = {
        "splits": {
            name: {
                "rows": 2,
                "positive_effect_groups": 1,
                "negative_effect_groups": 1,
                "training_seeds": [1, 2, 3],
            }
            for name in ("train", "selection", "calibration")
        }
    }
    metric = lambda rmse, balanced=0.7: {
        "row": {"rmse": rmse},
        "meaningful_sign": {"balanced_accuracy": balanced},
    }
    metrics = {
        "zero": metric(1.0),
        "ridge": metric(0.9),
        "ensemble": metric(0.8),
    }
    calibration_gate = {
        "accept_fraction": 0.1,
        "accepted_mean_true_utility_m": 0.02,
        "harmful_accepted_rows": 0,
    }
    passed = _development_gate(
        config, summary, metrics, calibration_gate
    )
    assert passed["passed"] is True
    calibration_gate["harmful_accepted_rows"] = 1
    failed = _development_gate(
        config, summary, metrics, calibration_gate
    )
    assert failed["passed"] is False


def test_development_gate_requires_trajectory_ablation_improvement():
    config = _config(
        minimum_train_rows=1,
        minimum_selection_rows=1,
        minimum_calibration_rows=1,
        minimum_effect_groups_per_sign=1,
        minimum_state_ablation_rmse_improvement_fraction=0.05,
    )
    summary = {
        "splits": {
            name: {
                "rows": 2,
                "positive_effect_groups": 1,
                "negative_effect_groups": 1,
                "training_seeds": [1, 2, 3],
            }
            for name in ("train", "selection", "calibration")
        }
    }
    metric = lambda rmse: {
        "row": {"rmse": rmse},
        "meaningful_sign": {"balanced_accuracy": 0.7},
    }
    calibration_gate = {
        "accept_fraction": 0.1,
        "accepted_mean_true_utility_m": 0.02,
        "harmful_accepted_rows": 0,
    }
    selection = {
        "zero": metric(1.0),
        "ridge": metric(0.8),
        "ensemble": metric(0.8),
        "state_ensemble": metric(0.82),
    }
    failed = _development_gate(
        config, summary, selection, calibration_gate
    )
    assert failed["checks"]["trajectory_features_beat_state_ablation"] is False
    selection["state_ensemble"] = metric(0.9)
    passed = _development_gate(
        config, summary, selection, calibration_gate
    )
    assert passed["checks"]["trajectory_features_beat_state_ablation"] is True
    assert passed["passed"] is True
