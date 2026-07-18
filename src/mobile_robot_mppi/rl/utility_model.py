"""Independent counterfactual utility models and group-aware uncertainty tools."""

import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values, minimum_scale=1.0e-6):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim not in (1, 2) or array.shape[0] == 0:
            raise ValueError("standardizer requires a non-empty 1D/2D array")
        if not np.isfinite(array).all():
            raise ValueError("standardizer values contain NaN or Inf")
        if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
            raise ValueError("minimum scale must be finite and positive")
        mean = np.mean(array, axis=0)
        scale = np.std(array, axis=0)
        # Match the standard zero-variance convention: a feature with no
        # measurable train variation is centered but not amplified.  Dividing
        # by epsilon turns a harmless holdout deviation into an arbitrarily
        # large number and makes uncertainty itself numerically meaningless.
        scale = np.where(scale < float(minimum_scale), 1.0, scale)
        return cls(
            np.asarray(mean, dtype=np.float64),
            np.asarray(scale, dtype=np.float64),
        )

    def transform(self, values):
        array = np.asarray(values, dtype=np.float64)
        result = (array - self.mean) / self.scale
        if not np.isfinite(result).all():
            raise FloatingPointError("standardized values contain NaN or Inf")
        return result.astype(np.float32)

    def inverse(self, values):
        array = np.asarray(values, dtype=np.float64)
        result = array * self.scale + self.mean
        if not np.isfinite(result).all():
            raise FloatingPointError("inverse-standardized values contain NaN or Inf")
        return result

    def state_dict(self):
        return {
            "mean": np.asarray(self.mean, dtype=np.float64),
            "scale": np.asarray(self.scale, dtype=np.float64),
        }

    @classmethod
    def from_state_dict(cls, state):
        if not isinstance(state, Mapping) or any(
            name not in state for name in ("mean", "scale")
        ):
            raise ValueError("standardizer state is incomplete")
        instance = cls(
            np.asarray(state["mean"], dtype=np.float64),
            np.asarray(state["scale"], dtype=np.float64),
        )
        if (
            instance.mean.shape != instance.scale.shape
            or not np.isfinite(instance.mean).all()
            or not np.isfinite(instance.scale).all()
            or np.any(instance.scale <= 0.0)
        ):
            raise ValueError("standardizer state is invalid")
        return instance


@dataclass(frozen=True)
class CounterfactualUtilityConfig:
    target: str = "goal_distance_improvement"
    model_selection_episode_seeds: tuple = ()
    calibration_episode_seeds: tuple = ()
    hidden_sizes: tuple = (64, 32)
    activation: str = "silu"
    ensemble_size: int = 5
    ensemble_seeds: tuple = (20260731, 20260732, 20260733, 20260734, 20260735)
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    huber_delta: float = 1.0
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 40
    scheduler_patience: int = 15
    scheduler_factor: float = 0.5
    minimum_learning_rate: float = 1.0e-5
    ridge_lambda: float = 0.01
    conformal_alpha: float = 0.10
    uncertainty_floor_m: float = 0.005
    meaningful_effect_m: float = 0.03
    minimum_train_rows: int = 180
    minimum_selection_rows: int = 60
    minimum_calibration_rows: int = 60
    minimum_effect_groups_per_sign: int = 4
    minimum_zero_rmse_improvement_fraction: float = 0.05
    minimum_state_ablation_rmse_improvement_fraction: float = 0.0
    maximum_ridge_rmse_ratio: float = 1.02
    minimum_meaningful_sign_balanced_accuracy: float = 0.60
    minimum_calibration_accept_fraction: float = 0.05
    minimum_calibration_accepted_mean_utility_m: float = 0.01
    harmful_accepted_utility_m: float = -0.03

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        sequence_names = {
            "model_selection_episode_seeds",
            "calibration_episode_seeds",
            "hidden_sizes",
            "ensemble_seeds",
        }
        kwargs = {}
        for name, field in cls.__dataclass_fields__.items():
            value = values.get(name, field.default)
            kwargs[name] = tuple(value) if name in sequence_names else value
        instance = cls(**kwargs)
        instance.validate()
        return instance

    def validate(self):
        if self.target != "goal_distance_improvement":
            raise ValueError("L23 utility target must be goal_distance_improvement")
        selection = tuple(int(v) for v in self.model_selection_episode_seeds)
        calibration = tuple(int(v) for v in self.calibration_episode_seeds)
        if (
            not selection
            or not calibration
            or len(selection) != len(set(selection))
            or len(calibration) != len(set(calibration))
            or set(selection).intersection(calibration)
        ):
            raise ValueError("utility selection/calibration seeds are invalid")
        if any(int(v) <= 0 for v in self.hidden_sizes):
            raise ValueError("utility hidden sizes must be positive")
        _activation(self.activation)
        if int(self.ensemble_size) < 2:
            raise ValueError("utility ensemble requires at least two members")
        if len(self.ensemble_seeds) != int(self.ensemble_size):
            raise ValueError("utility ensemble seeds must match ensemble size")
        integer_positive = (
            self.epochs,
            self.batch_size,
            self.early_stopping_patience,
            self.scheduler_patience,
            self.minimum_train_rows,
            self.minimum_selection_rows,
            self.minimum_calibration_rows,
            self.minimum_effect_groups_per_sign,
        )
        if any(int(value) <= 0 for value in integer_positive):
            raise ValueError("utility integer settings must be positive")
        numeric_positive = (
            self.learning_rate,
            self.huber_delta,
            self.gradient_clip_norm,
            self.minimum_learning_rate,
            self.uncertainty_floor_m,
            self.meaningful_effect_m,
            self.maximum_ridge_rmse_ratio,
            self.minimum_calibration_accepted_mean_utility_m,
        )
        if any(not np.isfinite(value) or float(value) <= 0.0 for value in numeric_positive):
            raise ValueError("utility positive settings are invalid")
        numeric_nonnegative = (self.weight_decay, self.ridge_lambda)
        if any(not np.isfinite(value) or float(value) < 0.0 for value in numeric_nonnegative):
            raise ValueError("utility non-negative settings are invalid")
        fractions = (
            self.scheduler_factor,
            self.conformal_alpha,
            self.minimum_zero_rmse_improvement_fraction,
            self.minimum_meaningful_sign_balanced_accuracy,
            self.minimum_calibration_accept_fraction,
        )
        if any(not 0.0 < float(value) < 1.0 for value in fractions):
            raise ValueError("utility fraction settings must be in (0,1)")
        if not (
            0.0
            <= float(self.minimum_state_ablation_rmse_improvement_fraction)
            < 1.0
        ):
            raise ValueError(
                "state-ablation improvement fraction must be in [0,1)"
            )
        if not np.isfinite(self.harmful_accepted_utility_m) or (
            self.harmful_accepted_utility_m >= 0.0
        ):
            raise ValueError("harmful accepted utility threshold must be negative")

    def to_dict(self):
        value = asdict(self)
        for name in (
            "model_selection_episode_seeds",
            "calibration_episode_seeds",
            "hidden_sizes",
            "ensemble_seeds",
        ):
            value[name] = list(value[name])
        return value


def _activation(name):
    factories = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "softplus": nn.Softplus,
        "tanh": nn.Tanh,
    }
    key = str(name).lower()
    if key not in factories:
        raise ValueError("unsupported utility activation: %s" % name)
    return factories[key]


class CounterfactualUtilityMLP(nn.Module):
    """Small scalar regressor for paired burst utility."""

    def __init__(self, input_dim, hidden_sizes=(64, 32), activation="silu"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.activation_name = str(activation).lower()
        if self.input_dim <= 0 or any(value <= 0 for value in self.hidden_sizes):
            raise ValueError("utility model dimensions must be positive")
        layers = []
        previous = self.input_dim
        activation_class = _activation(self.activation_name)
        for width in self.hidden_sizes:
            linear = nn.Linear(previous, width)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers.extend((linear, activation_class()))
            previous = width
        output = nn.Linear(previous, 1)
        nn.init.orthogonal_(output.weight, gain=0.01)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError("utility model input shape is invalid")
        prediction = self.network(features).squeeze(-1)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("utility model produced NaN or Inf")
        return prediction

    def config_dict(self):
        return {
            "input_dim": self.input_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation_name,
        }

    @property
    def parameter_count(self):
        return int(sum(parameter.numel() for parameter in self.parameters()))


class CounterfactualUtilityEnsemble:
    """CPU/CUDA inference wrapper with a fail-closed calibrated LCB."""

    def __init__(
        self,
        models,
        feature_standardizer,
        target_standardizer,
        uncertainty_floor,
        conformal_multiplier,
        device="cpu",
    ):
        self.models = tuple(models)
        if len(self.models) < 2:
            raise ValueError("utility inference requires at least two models")
        self.feature_standardizer = feature_standardizer
        self.target_standardizer = target_standardizer
        self.uncertainty_floor = float(uncertainty_floor)
        self.conformal_multiplier = float(conformal_multiplier)
        self.device = torch.device(device)
        if (
            not np.isfinite(self.uncertainty_floor)
            or self.uncertainty_floor <= 0.0
            or not np.isfinite(self.conformal_multiplier)
            or self.conformal_multiplier < 0.0
        ):
            raise ValueError("utility inference uncertainty contract is invalid")
        dimensions = set(model.input_dim for model in self.models)
        if len(dimensions) != 1:
            raise ValueError("utility ensemble member dimensions differ")
        self.input_dim = dimensions.pop()
        if np.asarray(self.feature_standardizer.mean).shape != (self.input_dim,):
            raise ValueError("utility feature standardizer dimension differs")
        for model in self.models:
            model.to(self.device)
            model.eval()

    def predict(self, features):
        values = np.asarray(features, dtype=np.float64)
        single = values.ndim == 1
        if single:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("utility inference feature shape is invalid")
        normalized = self.feature_standardizer.transform(values)
        tensor = torch.as_tensor(
            normalized, dtype=torch.float32, device=self.device
        )
        members = []
        with torch.no_grad():
            for model in self.models:
                prediction = model(tensor).cpu().numpy()
                members.append(self.target_standardizer.inverse(prediction))
        members = np.stack(members)
        mean, epistemic, scale = ensemble_statistics(
            members, self.uncertainty_floor
        )
        lcb = lower_confidence_bound(
            mean, scale, self.conformal_multiplier
        )
        result = {
            "mean_utility_m": mean,
            "epistemic_std_m": epistemic,
            "calibration_scale_m": scale,
            "lower_confidence_bound_m": lcb,
            "accept": lcb > 0.0,
        }
        if single:
            return {
                name: bool(value[0]) if name == "accept" else float(value[0])
                for name, value in result.items()
            }
        return result


def load_counterfactual_utility_ensemble(
    path, device="cpu", require_eligible=True
):
    """Load a trusted local utility artifact and reject ineligible models."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    kwargs = {"map_location": torch.device(device)}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    payload = torch.load(str(source), **kwargs)
    if not isinstance(payload, Mapping):
        raise TypeError("utility checkpoint must contain a mapping")
    required = (
        "format",
        "format_version",
        "model_config",
        "member_state_dicts",
        "feature_standardizer",
        "target_standardizer",
        "utility_config",
        "calibration",
        "development_gate",
    )
    if any(name not in payload for name in required):
        raise ValueError("utility checkpoint schema is incomplete")
    if (
        payload["format"] != "counterfactual_utility_ensemble"
        or int(payload["format_version"]) != 1
    ):
        raise ValueError("unsupported utility checkpoint format")
    if require_eligible and not bool(payload["development_gate"].get("passed")):
        raise ValueError("utility checkpoint failed its development gate")
    utility_config = CounterfactualUtilityConfig.from_mapping(
        payload["utility_config"]
    )
    model_config = dict(payload["model_config"])
    states = tuple(payload["member_state_dicts"])
    if len(states) != utility_config.ensemble_size:
        raise ValueError("utility checkpoint ensemble size differs")
    models = []
    for state in states:
        model = CounterfactualUtilityMLP(**model_config)
        model.load_state_dict(state, strict=True)
        models.append(model)
    calibration = dict(payload["calibration"])
    ensemble = CounterfactualUtilityEnsemble(
        models,
        Standardizer.from_state_dict(payload["feature_standardizer"]),
        Standardizer.from_state_dict(payload["target_standardizer"]),
        uncertainty_floor=utility_config.uncertainty_floor_m,
        conformal_multiplier=float(calibration["multiplier"]),
        device=device,
    )
    return ensemble, dict(payload)


def group_keys(training_seed, episode_seed):
    training = np.asarray(training_seed, dtype=np.int64).reshape(-1)
    episode = np.asarray(episode_seed, dtype=np.int64).reshape(-1)
    if training.shape != episode.shape or training.size == 0:
        raise ValueError("group key arrays must be non-empty and aligned")
    return np.stack((training, episode), axis=1)


def unique_group_tuples(groups):
    values = np.asarray(groups, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("groups must have shape [N,2]")
    return sorted(set((int(a), int(b)) for a, b in values))


def group_bootstrap_indices(groups, rng):
    """Sample complete checkpoint/episode groups with replacement."""

    values = np.asarray(groups, dtype=np.int64)
    unique = unique_group_tuples(values)
    draws = rng.randint(0, len(unique), size=len(unique))
    indices = []
    for draw in draws:
        key = unique[int(draw)]
        selected = np.flatnonzero(
            (values[:, 0] == key[0]) & (values[:, 1] == key[1])
        )
        indices.extend(selected.tolist())
    result = np.asarray(indices, dtype=np.int64)
    if result.size == 0:
        raise RuntimeError("group bootstrap produced no rows")
    return result


def fit_ridge(features, targets, regularization=0.01):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or y.size == 0:
        raise ValueError("ridge arrays are not aligned")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge arrays contain NaN or Inf")
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("ridge regularization must be finite and non-negative")
    design = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    penalty = np.eye(design.shape[1]) * float(regularization)
    penalty[-1, -1] = 0.0
    matrix = design.T @ design + penalty
    rhs = design.T @ y
    try:
        weights = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(matrix) @ rhs
    if not np.isfinite(weights).all():
        raise FloatingPointError("ridge fit produced NaN or Inf")
    return weights


def predict_ridge(features, weights):
    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or w.size != x.shape[1] + 1:
        raise ValueError("ridge prediction dimensions do not match")
    result = x @ w[:-1] + w[-1]
    if not np.isfinite(result).all():
        raise FloatingPointError("ridge prediction produced NaN or Inf")
    return result


def regression_metrics(target, prediction):
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != pred.shape or y.size == 0:
        raise ValueError("regression metric arrays are not aligned")
    error = pred - y
    denominator = float(np.sum(np.square(y - np.mean(y))))
    r2 = (
        1.0 - float(np.sum(np.square(error))) / denominator
        if denominator > 0.0 else 0.0
    )
    correlation = (
        float(np.corrcoef(y, pred)[0, 1])
        if np.std(y) > 0.0 and np.std(pred) > 0.0 else 0.0
    )
    result = {
        "samples": int(y.size),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": r2,
        "pearson": correlation,
    }
    if not np.isfinite(tuple(result.values())).all():
        raise FloatingPointError("regression metrics contain NaN or Inf")
    return result


def meaningful_sign_metrics(target, prediction, threshold=0.03):
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != pred.shape or y.size == 0:
        raise ValueError("sign metric arrays are not aligned")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("meaningful threshold must be positive")
    positive = y >= threshold
    negative = y <= -threshold
    positive_accuracy = (
        float(np.mean(pred[positive] > 0.0)) if np.any(positive) else 0.0
    )
    negative_accuracy = (
        float(np.mean(pred[negative] < 0.0)) if np.any(negative) else 0.0
    )
    balanced = (
        0.5 * (positive_accuracy + negative_accuracy)
        if np.any(positive) and np.any(negative) else 0.0
    )
    return {
        "threshold": float(threshold),
        "positive_rows": int(np.sum(positive)),
        "negative_rows": int(np.sum(negative)),
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "balanced_accuracy": float(balanced),
    }


def effect_group_counts(target, groups, threshold=0.03):
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    values = np.asarray(groups, dtype=np.int64)
    if values.shape != (y.size, 2):
        raise ValueError("effect targets and groups are not aligned")
    positive = set()
    negative = set()
    for value, key in zip(y, values):
        group = (int(key[0]), int(key[1]))
        if value >= threshold:
            positive.add(group)
        if value <= -threshold:
            negative.add(group)
    return {
        "groups": len(unique_group_tuples(values)),
        "positive_effect_groups": len(positive),
        "negative_effect_groups": len(negative),
        "threshold": float(threshold),
    }


def ensemble_statistics(predictions, uncertainty_floor=0.005):
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("ensemble predictions must have shape [M,N], M>=2")
    if not np.isfinite(values).all():
        raise ValueError("ensemble predictions contain NaN or Inf")
    if not np.isfinite(uncertainty_floor) or uncertainty_floor <= 0.0:
        raise ValueError("uncertainty floor must be finite and positive")
    mean = np.mean(values, axis=0)
    epistemic = np.std(values, axis=0, ddof=0)
    scale = np.maximum(epistemic, float(uncertainty_floor))
    return mean, epistemic, scale


def group_conformal_multiplier(
    target,
    prediction_mean,
    prediction_scale,
    groups,
    alpha=0.10,
):
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    mean = np.asarray(prediction_mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(prediction_scale, dtype=np.float64).reshape(-1)
    values = np.asarray(groups, dtype=np.int64)
    if y.shape != mean.shape or y.shape != scale.shape:
        raise ValueError("conformal arrays are not aligned")
    if values.shape != (y.size, 2) or np.any(scale <= 0.0):
        raise ValueError("conformal groups/scales are invalid")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("conformal alpha must be in (0,1)")
    row_scores = (mean - y) / scale
    group_scores = []
    for key in unique_group_tuples(values):
        mask = (values[:, 0] == key[0]) & (values[:, 1] == key[1])
        group_scores.append(float(np.max(row_scores[mask])))
    scores = np.asarray(group_scores, dtype=np.float64)
    count = scores.size
    quantile_level = min(
        1.0,
        float(np.ceil((count + 1) * (1.0 - float(alpha)))) / count,
    )
    try:
        quantile = np.quantile(scores, quantile_level, method="higher")
    except TypeError:  # NumPy < 1.22 compatibility.
        quantile = np.quantile(
            scores, quantile_level, interpolation="higher"
        )
    multiplier = max(0.0, float(quantile))
    return {
        "alpha": float(alpha),
        "groups": int(count),
        "quantile_level": float(quantile_level),
        "multiplier": multiplier,
        "group_scores": scores,
    }


def lower_confidence_bound(prediction_mean, prediction_scale, multiplier):
    mean = np.asarray(prediction_mean, dtype=np.float64)
    scale = np.asarray(prediction_scale, dtype=np.float64)
    if mean.shape != scale.shape or np.any(scale <= 0.0):
        raise ValueError("LCB mean/scale arrays are invalid")
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("LCB multiplier must be finite and non-negative")
    result = mean - float(multiplier) * scale
    if not np.isfinite(result).all():
        raise FloatingPointError("LCB contains NaN or Inf")
    return result


def group_lcb_coverage(target, lower_bound, groups):
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    lcb = np.asarray(lower_bound, dtype=np.float64).reshape(-1)
    values = np.asarray(groups, dtype=np.int64)
    if y.shape != lcb.shape or values.shape != (y.size, 2):
        raise ValueError("coverage arrays are not aligned")
    covered = []
    for key in unique_group_tuples(values):
        mask = (values[:, 0] == key[0]) & (values[:, 1] == key[1])
        covered.append(bool(np.all(y[mask] >= lcb[mask])))
    return float(np.mean(covered))


__all__ = [
    "CounterfactualUtilityConfig",
    "CounterfactualUtilityEnsemble",
    "CounterfactualUtilityMLP",
    "Standardizer",
    "effect_group_counts",
    "ensemble_statistics",
    "fit_ridge",
    "group_bootstrap_indices",
    "group_conformal_multiplier",
    "group_keys",
    "group_lcb_coverage",
    "lower_confidence_bound",
    "load_counterfactual_utility_ensemble",
    "meaningful_sign_metrics",
    "predict_ridge",
    "regression_metrics",
    "unique_group_tuples",
]
