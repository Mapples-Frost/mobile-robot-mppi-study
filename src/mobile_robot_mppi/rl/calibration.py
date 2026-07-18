"""Fail-closed selection for a globally calibrated critic-advantage margin."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AdvantageMarginSelectionConfig:
    candidate_margins: Sequence[float] = (
        0.0,
        0.005,
        0.010,
        0.020,
        0.040,
        0.080,
    )
    maximum_success_losses: int = 0
    maximum_collision_regressions: int = 0
    minimum_success_gains: int = 1
    minimum_mean_goal_distance_improvement: float = 0.005
    minimum_training_seed_mean_return_delta: float = 0.0
    minimum_training_seed_accept_fraction: float = 0.05
    maximum_training_seed_accept_fraction: float = 0.95

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] = None):
        values = dict(values or {})
        fields = cls.__dataclass_fields__
        return cls(**{
            name: values.get(name, field.default)
            for name, field in fields.items()
        })

    def validate(self):
        margins = np.asarray(self.candidate_margins, dtype=np.float64).reshape(-1)
        if (
            margins.size == 0
            or not np.isfinite(margins).all()
            or np.any(margins < 0.0)
            or len(set(float(value) for value in margins)) != margins.size
            or np.any(np.diff(margins) <= 0.0)
        ):
            raise ValueError(
                "candidate advantage margins must be unique, finite, "
                "non-negative and strictly increasing"
            )
        integer_fields = (
            self.maximum_success_losses,
            self.maximum_collision_regressions,
            self.minimum_success_gains,
        )
        if any(int(value) != value or int(value) < 0 for value in integer_fields):
            raise ValueError("advantage margin count thresholds are invalid")
        numeric = np.asarray((
            self.minimum_mean_goal_distance_improvement,
            self.minimum_training_seed_mean_return_delta,
            self.minimum_training_seed_accept_fraction,
            self.maximum_training_seed_accept_fraction,
        ), dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("advantage margin thresholds must be finite")
        if self.minimum_mean_goal_distance_improvement < 0.0:
            raise ValueError("minimum distance improvement cannot be negative")
        if not (
            0.0
            <= self.minimum_training_seed_accept_fraction
            < self.maximum_training_seed_accept_fraction
            <= 1.0
        ):
            raise ValueError("advantage margin acceptance interval is invalid")

    def to_dict(self):
        result = asdict(self)
        result["candidate_margins"] = [
            float(value) for value in self.candidate_margins
        ]
        return result


def _candidate_rank(pooled, per_training_seed, margin):
    return (
        int(pooled["success_gains"]),
        float(min(
            row["mean_return_delta"] for row in per_training_seed.values()
        )),
        float(pooled["mean_return_delta"]),
        float(pooled["mean_goal_distance_improvement"]),
        float(margin),
    )


def select_advantage_margin(candidates, config=None):
    """Select one global margin or explicitly return the BC fallback.

    ``candidates`` maps each numeric margin to a mapping with ``pooled`` and
    ``per_training_seed`` summaries.  Episode rows remain nested within the
    independently trained policies; no episode is treated as a training
    replicate by this selector.
    """

    config = (
        config
        if isinstance(config, AdvantageMarginSelectionConfig)
        else AdvantageMarginSelectionConfig.from_mapping(config)
    )
    config.validate()
    expected = tuple(float(value) for value in config.candidate_margins)
    actual = tuple(sorted(float(value) for value in candidates))
    if actual != expected:
        raise ValueError(
            "advantage margin candidate grid does not match preregistration"
        )

    decisions = []
    eligible = []
    expected_training_seeds = None
    for margin in expected:
        candidate = dict(candidates[margin])
        pooled = dict(candidate.get("pooled", {}))
        per_seed = dict(candidate.get("per_training_seed", {}))
        if not per_seed:
            raise ValueError("advantage margin candidate has no training seeds")
        training_seeds = tuple(sorted(str(value) for value in per_seed))
        if expected_training_seeds is None:
            expected_training_seeds = training_seeds
        elif training_seeds != expected_training_seeds:
            raise ValueError(
                "advantage margin candidates use different training seeds"
            )
        required_pooled = (
            "bc_success_losses",
            "collision_regressions",
            "success_gains",
            "mean_return_delta",
            "mean_goal_distance_improvement",
        )
        if any(name not in pooled for name in required_pooled):
            raise ValueError("advantage margin pooled summary is incomplete")
        for row in per_seed.values():
            if any(
                name not in row
                for name in (
                    "mean_return_delta",
                    "mean_gate_accept_fraction",
                )
            ):
                raise ValueError(
                    "advantage margin per-seed summary is incomplete"
                )
        finite_values = [
            float(pooled[name]) for name in required_pooled
        ] + [
            float(row[name])
            for row in per_seed.values()
            for name in (
                "mean_return_delta",
                "mean_gate_accept_fraction",
            )
        ]
        if not np.isfinite(finite_values).all():
            raise ValueError("advantage margin summaries contain NaN or Inf")

        no_success_losses = int(pooled["bc_success_losses"]) <= int(
            config.maximum_success_losses
        )
        no_collision_regressions = int(
            pooled["collision_regressions"]
        ) <= int(config.maximum_collision_regressions)
        seed_return_noninferiority = all(
            float(row["mean_return_delta"])
            >= config.minimum_training_seed_mean_return_delta
            for row in per_seed.values()
        )
        nondegenerate = all(
            config.minimum_training_seed_accept_fraction
            <= float(row["mean_gate_accept_fraction"])
            <= config.maximum_training_seed_accept_fraction
            for row in per_seed.values()
        )
        meaningful_benefit = bool(
            int(pooled["success_gains"]) >= config.minimum_success_gains
            or float(pooled["mean_goal_distance_improvement"])
            >= config.minimum_mean_goal_distance_improvement
        )
        is_eligible = bool(
            no_success_losses
            and no_collision_regressions
            and seed_return_noninferiority
            and nondegenerate
            and meaningful_benefit
        )
        rank = _candidate_rank(pooled, per_seed, margin)
        decision = {
            "margin": float(margin),
            "eligible": is_eligible,
            "no_success_losses": no_success_losses,
            "no_collision_regressions": no_collision_regressions,
            "training_seed_return_noninferiority": (
                seed_return_noninferiority
            ),
            "nondegenerate_each_training_seed": nondegenerate,
            "meaningful_benefit": meaningful_benefit,
            "rank": list(rank),
        }
        decisions.append(decision)
        if is_eligible:
            eligible.append((rank, margin))

    if eligible:
        selected_margin = float(max(eligible)[1])
        selected_mode = "target_advantage_margin"
        reason = "best_preregistered_eligible_margin"
    else:
        selected_margin = None
        selected_mode = "bc_fallback"
        reason = "no_preregistered_margin_eligible"
    return {
        "selected_mode": selected_mode,
        "selected_margin": selected_margin,
        "selection_reason": reason,
        "training_seeds": list(expected_training_seeds or ()),
        "config": config.to_dict(),
        "candidate_decisions": decisions,
    }


@dataclass(frozen=True)
class CriticConsensusSelectionConfig:
    """Pre-registered selection contract for dimensionless LCB beta."""

    candidate_betas: Sequence[float] = (
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        8.0,
    )
    maximum_success_losses: int = 0
    maximum_collision_regressions: int = 0
    minimum_success_gains: int = 1
    minimum_mean_goal_distance_improvement: float = 0.005
    minimum_training_seed_mean_return_delta: float = 0.0
    minimum_training_seed_accept_fraction: float = 0.05
    maximum_training_seed_accept_fraction: float = 0.95

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] = None):
        values = dict(values or {})
        fields = cls.__dataclass_fields__
        return cls(**{
            name: values.get(name, field.default)
            for name, field in fields.items()
        })

    def validate(self):
        betas = np.asarray(self.candidate_betas, dtype=np.float64).reshape(-1)
        if (
            betas.size == 0
            or not np.isfinite(betas).all()
            or np.any(betas < 1.0)
            or len(set(float(value) for value in betas)) != betas.size
            or np.any(np.diff(betas) <= 0.0)
        ):
            raise ValueError(
                "critic consensus betas must be unique, finite, at least "
                "one and strictly increasing"
            )
        integer_fields = (
            self.maximum_success_losses,
            self.maximum_collision_regressions,
            self.minimum_success_gains,
        )
        if any(int(value) != value or int(value) < 0 for value in integer_fields):
            raise ValueError("critic consensus count thresholds are invalid")
        numeric = np.asarray((
            self.minimum_mean_goal_distance_improvement,
            self.minimum_training_seed_mean_return_delta,
            self.minimum_training_seed_accept_fraction,
            self.maximum_training_seed_accept_fraction,
        ), dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("critic consensus thresholds must be finite")
        if self.minimum_mean_goal_distance_improvement < 0.0:
            raise ValueError("minimum distance improvement cannot be negative")
        if not (
            0.0
            <= self.minimum_training_seed_accept_fraction
            < self.maximum_training_seed_accept_fraction
            <= 1.0
        ):
            raise ValueError("critic consensus acceptance interval is invalid")

    def to_dict(self):
        result = asdict(self)
        result["candidate_betas"] = [
            float(value) for value in self.candidate_betas
        ]
        return result


def select_critic_consensus_beta(candidates, config=None):
    """Select one global scale-invariant consensus beta or recover BC."""

    config = (
        config
        if isinstance(config, CriticConsensusSelectionConfig)
        else CriticConsensusSelectionConfig.from_mapping(config)
    )
    config.validate()
    expected = tuple(float(value) for value in config.candidate_betas)
    actual = tuple(sorted(float(value) for value in candidates))
    if actual != expected:
        raise ValueError(
            "critic consensus beta grid does not match preregistration"
        )

    decisions = []
    eligible = []
    expected_training_seeds = None
    for beta in expected:
        candidate = dict(candidates[beta])
        pooled = dict(candidate.get("pooled", {}))
        per_seed = dict(candidate.get("per_training_seed", {}))
        if not per_seed:
            raise ValueError("critic consensus candidate has no training seeds")
        training_seeds = tuple(sorted(str(value) for value in per_seed))
        if expected_training_seeds is None:
            expected_training_seeds = training_seeds
        elif training_seeds != expected_training_seeds:
            raise ValueError(
                "critic consensus candidates use different training seeds"
            )
        required_pooled = (
            "bc_success_losses",
            "collision_regressions",
            "success_gains",
            "mean_return_delta",
            "mean_goal_distance_improvement",
        )
        if any(name not in pooled for name in required_pooled):
            raise ValueError("critic consensus pooled summary is incomplete")
        per_seed_fields = (
            "mean_return_delta",
            "mean_gate_accept_fraction",
        )
        if any(
            any(name not in row for name in per_seed_fields)
            for row in per_seed.values()
        ):
            raise ValueError("critic consensus per-seed summary is incomplete")
        finite_values = [
            float(pooled[name]) for name in required_pooled
        ] + [
            float(row[name])
            for row in per_seed.values()
            for name in per_seed_fields
        ]
        if not np.isfinite(finite_values).all():
            raise ValueError("critic consensus summaries contain NaN or Inf")

        no_success_losses = int(pooled["bc_success_losses"]) <= int(
            config.maximum_success_losses
        )
        no_collision_regressions = int(
            pooled["collision_regressions"]
        ) <= int(config.maximum_collision_regressions)
        seed_return_noninferiority = all(
            float(row["mean_return_delta"])
            >= config.minimum_training_seed_mean_return_delta
            for row in per_seed.values()
        )
        nondegenerate = all(
            config.minimum_training_seed_accept_fraction
            <= float(row["mean_gate_accept_fraction"])
            <= config.maximum_training_seed_accept_fraction
            for row in per_seed.values()
        )
        meaningful_benefit = bool(
            int(pooled["success_gains"]) >= config.minimum_success_gains
            or float(pooled["mean_goal_distance_improvement"])
            >= config.minimum_mean_goal_distance_improvement
        )
        is_eligible = bool(
            no_success_losses
            and no_collision_regressions
            and seed_return_noninferiority
            and nondegenerate
            and meaningful_benefit
        )
        rank = _candidate_rank(pooled, per_seed, beta)
        decision = {
            "beta": float(beta),
            "eligible": is_eligible,
            "no_success_losses": no_success_losses,
            "no_collision_regressions": no_collision_regressions,
            "training_seed_return_noninferiority": (
                seed_return_noninferiority
            ),
            "nondegenerate_each_training_seed": nondegenerate,
            "meaningful_benefit": meaningful_benefit,
            "rank": list(rank),
        }
        decisions.append(decision)
        if is_eligible:
            eligible.append((rank, beta))

    if eligible:
        selected_beta = float(max(eligible)[1])
        selected_mode = "target_advantage_consensus_lcb"
        reason = "best_preregistered_eligible_beta"
    else:
        selected_beta = None
        selected_mode = "bc_fallback"
        reason = "no_preregistered_beta_eligible"
    return {
        "selected_mode": selected_mode,
        "selected_beta": selected_beta,
        "selection_reason": reason,
        "training_seeds": list(expected_training_seeds or ()),
        "config": config.to_dict(),
        "candidate_decisions": decisions,
    }
