"""Contracts for counterfactual correction-risk datasets."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CounterfactualLabelConfig:
    harmful_return_delta: float = -0.5
    harmful_distance_improvement: float = -0.03
    beneficial_return_delta: float = 0.5
    beneficial_distance_improvement: float = 0.03

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] = None):
        values = dict(values or {})
        return cls(**{
            name: float(values.get(name, field.default))
            for name, field in cls.__dataclass_fields__.items()
        })

    def validate(self):
        values = np.asarray(tuple(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("counterfactual label thresholds must be finite")
        if self.harmful_return_delta >= 0.0:
            raise ValueError("harmful return threshold must be negative")
        if self.harmful_distance_improvement >= 0.0:
            raise ValueError("harmful distance threshold must be negative")
        if self.beneficial_return_delta <= 0.0:
            raise ValueError("beneficial return threshold must be positive")
        if self.beneficial_distance_improvement <= 0.0:
            raise ValueError("beneficial distance threshold must be positive")

    def to_dict(self):
        return asdict(self)


def classify_counterfactual(base, candidate, config=None):
    """Return continuous paired targets and a preregistered 3-class label."""

    config = (
        config
        if isinstance(config, CounterfactualLabelConfig)
        else CounterfactualLabelConfig.from_mapping(config)
    )
    config.validate()
    required = (
        "return",
        "goal_distance",
        "success",
        "collision",
        "minimum_clearance",
        "safety_overrides",
    )
    if any(name not in base or name not in candidate for name in required):
        raise ValueError("counterfactual branch summary is incomplete")
    numeric = np.asarray([
        float(row[name])
        for row in (base, candidate)
        for name in (
            "return",
            "goal_distance",
            "minimum_clearance",
            "safety_overrides",
        )
    ], dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("counterfactual branch summary contains NaN or Inf")

    return_delta = float(candidate["return"] - base["return"])
    distance_improvement = float(
        base["goal_distance"] - candidate["goal_distance"]
    )
    clearance_improvement = float(
        candidate["minimum_clearance"] - base["minimum_clearance"]
    )
    safety_override_delta = int(candidate["safety_overrides"]) - int(
        base["safety_overrides"]
    )
    collision_regression = bool(
        not bool(base["collision"]) and bool(candidate["collision"])
    )
    success_loss = bool(
        bool(base["success"]) and not bool(candidate["success"])
    )
    success_gain = bool(
        not bool(base["success"]) and bool(candidate["success"])
    )
    harmful = bool(
        collision_regression
        or success_loss
        or (
            return_delta <= config.harmful_return_delta
            and distance_improvement
            <= config.harmful_distance_improvement
        )
    )
    beneficial = bool(
        not harmful
        and (
            success_gain
            or (
                return_delta >= config.beneficial_return_delta
                and distance_improvement
                >= config.beneficial_distance_improvement
            )
        )
    )
    label = "harmful" if harmful else "beneficial" if beneficial else "neutral"
    label_index = {"harmful": 0, "neutral": 1, "beneficial": 2}[label]
    return {
        "return_delta": return_delta,
        "goal_distance_improvement": distance_improvement,
        "minimum_clearance_improvement": clearance_improvement,
        "safety_override_delta": safety_override_delta,
        "collision_regression": collision_regression,
        "success_loss": success_loss,
        "success_gain": success_gain,
        "label": label,
        "label_index": label_index,
    }


def validate_episode_split(train_seeds, validation_seeds):
    """Validate group-level dataset splits before branch collection."""

    groups = []
    for name, values in (
        ("train", train_seeds),
        ("validation", validation_seeds),
    ):
        seeds = tuple(int(value) for value in values)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("%s episode seeds must be non-empty and unique" % name)
        if any(value < 0 or value > 2 ** 32 - 1 for value in seeds):
            raise ValueError("%s episode seeds are out of range" % name)
        groups.append(set(seeds))
    if groups[0].intersection(groups[1]):
        raise ValueError("train and validation episode seeds must be disjoint")


def validate_sealed_episode_split(
    train_seeds, validation_seeds, sealed_test_seeds
):
    """Ensure development collection cannot silently consume sealed seeds."""

    validate_episode_split(train_seeds, validation_seeds)
    sealed = tuple(int(value) for value in sealed_test_seeds)
    if not sealed or len(set(sealed)) != len(sealed):
        raise ValueError("sealed test episode seeds must be non-empty and unique")
    if any(value < 0 or value > 2 ** 32 - 1 for value in sealed):
        raise ValueError("sealed test episode seeds are out of range")
    development = set(int(value) for value in train_seeds).union(
        int(value) for value in validation_seeds
    )
    if development.intersection(sealed):
        raise ValueError("development and sealed test seeds must be disjoint")


def burst_accounting_valid(row):
    """Return whether a candidate burst row satisfies its execution contract."""

    names = (
        "candidate_intervention_steps_requested",
        "candidate_intervention_steps_run",
        "candidate_intervention_accepted_steps",
        "candidate_intervention_rejected_steps",
        "candidate_intervention_first_gate_accepted",
        "candidate_intervention_accept_fraction",
        "candidate_intervention_correction_abs_mean",
        "candidate_intervention_correction_abs_max",
    )
    if any(name not in row for name in names):
        return False
    try:
        requested = int(row[names[0]])
        steps_run = int(row[names[1]])
        accepted = int(row[names[2]])
        rejected = int(row[names[3]])
        first_value = row[names[4]]
        first_accepted = (
            first_value
            if isinstance(first_value, (bool, np.bool_))
            else str(first_value).strip().lower() in ("true", "1")
        )
        fraction = float(row[names[5]])
        correction_mean = float(row[names[6]])
        correction_max = float(row[names[7]])
    except (TypeError, ValueError):
        return False
    numeric = np.asarray(
        (fraction, correction_mean, correction_max), dtype=np.float64
    )
    return bool(
        requested >= 1
        and 1 <= steps_run <= requested
        and accepted >= 1
        and rejected >= 0
        and accepted + rejected == steps_run
        and first_accepted
        and np.isfinite(numeric).all()
        and np.isclose(fraction, float(accepted) / float(steps_run))
        and 0.0 <= fraction <= 1.0
        and 0.0 <= correction_mean <= correction_max
    )


def feature_vector(
    normalized_observation: Sequence[float],
    base_action: Sequence[float],
    candidate_action: Sequence[float],
    diagnostics: Mapping[str, Any],
    branch_step: int,
):
    """Build the leakage-free feature vector used by the future risk model."""

    observation = np.asarray(
        normalized_observation, dtype=np.float32
    ).reshape(-1)
    base = np.asarray(base_action, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate_action, dtype=np.float32).reshape(-1)
    if observation.size == 0 or base.size == 0 or candidate.shape != base.shape:
        raise ValueError("counterfactual feature arrays have invalid shapes")
    if not all(np.isfinite(value).all() for value in (observation, base, candidate)):
        raise ValueError("counterfactual feature arrays contain NaN or Inf")
    diagnostic_names = (
        "online_advantage_q1",
        "online_advantage_q2",
        "online_advantage_mean",
        "online_advantage_half_disagreement",
        "online_consensus_lcb",
        "target_advantage_q1",
        "target_advantage_q2",
        "target_advantage_mean",
        "target_advantage_half_disagreement",
        "target_consensus_lcb",
        "raw_applied_correction_abs_mean",
        "raw_applied_correction_abs_max",
    )
    if any(name not in diagnostics for name in diagnostic_names):
        raise ValueError("counterfactual diagnostics are incomplete")
    diagnostic_values = np.asarray(
        [float(diagnostics[name]) for name in diagnostic_names],
        dtype=np.float32,
    )
    if not np.isfinite(diagnostic_values).all() or int(branch_step) < 0:
        raise ValueError("counterfactual scalar features are invalid")
    vector = np.concatenate((
        observation,
        base,
        candidate,
        candidate - base,
        diagnostic_values,
        np.asarray((float(branch_step),), dtype=np.float32),
    )).astype(np.float32, copy=False)
    return vector, {
        "observation_dim": int(observation.size),
        "action_dim": int(base.size),
        "diagnostic_names": list(diagnostic_names),
        "feature_dim": int(vector.size),
    }


@dataclass(frozen=True)
class RiskDatasetSufficiencyConfig:
    minimum_train_samples: int = 60
    minimum_validation_samples: int = 12
    minimum_train_harmful: int = 8
    minimum_train_beneficial: int = 8
    minimum_validation_harmful: int = 2
    minimum_validation_beneficial: int = 2
    minimum_training_seeds_per_split: int = 3

    def validate(self):
        values = tuple(asdict(self).values())
        if any(int(value) != value or int(value) < 0 for value in values):
            raise ValueError("risk dataset sufficiency counts are invalid")
        if self.minimum_training_seeds_per_split <= 0:
            raise ValueError("risk dataset requires at least one training seed")

    def to_dict(self):
        return asdict(self)


def assess_risk_dataset_sufficiency(summary, config=None):
    """Apply the preregistered fail-closed gate before model training."""

    config = config or RiskDatasetSufficiencyConfig()
    if not isinstance(config, RiskDatasetSufficiencyConfig):
        config = RiskDatasetSufficiencyConfig(**dict(config))
    config.validate()
    required = (
        "quality_passed",
        "train_samples",
        "validation_samples",
        "label_counts",
        "train_training_seeds",
        "validation_training_seeds",
    )
    if any(name not in summary for name in required):
        raise ValueError("risk dataset audit summary is incomplete")
    labels = dict(summary["label_counts"])
    train = dict(labels.get("train", {}))
    validation = dict(labels.get("validation", {}))
    checks = {
        "quality_passed": bool(summary["quality_passed"]),
        "enough_train_samples": int(summary["train_samples"])
        >= config.minimum_train_samples,
        "enough_validation_samples": int(summary["validation_samples"])
        >= config.minimum_validation_samples,
        "enough_train_harmful": int(train.get("harmful", 0))
        >= config.minimum_train_harmful,
        "enough_train_beneficial": int(train.get("beneficial", 0))
        >= config.minimum_train_beneficial,
        "enough_validation_harmful": int(validation.get("harmful", 0))
        >= config.minimum_validation_harmful,
        "enough_validation_beneficial": int(validation.get("beneficial", 0))
        >= config.minimum_validation_beneficial,
        "all_training_seeds_in_train": len(set(
            int(value) for value in summary["train_training_seeds"]
        )) >= config.minimum_training_seeds_per_split,
        "all_training_seeds_in_validation": len(set(
            int(value) for value in summary["validation_training_seeds"]
        )) >= config.minimum_training_seeds_per_split,
    }
    eligible = all(checks.values())
    return {
        "train_risk_model": bool(eligible),
        "decision_reason": (
            "all_preregistered_data_gates_passed"
            if eligible
            else "one_or_more_preregistered_data_gates_failed"
        ),
        "checks": checks,
        "config": config.to_dict(),
    }
