"""Calibrated cross-layer reliability for policy-guided MPPI.

The outputs are bounded engineering confidence scores, not probabilities and
not formal epistemic guarantees.  Every signal is available online: ICODE
ensemble disagreement, ICODE training-support distance, completed-transition
innovation, and Actor training-support distance.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def decreasing_linear_confidence(value, soft, hard):
    """Map a non-negative risk score to [0,1], with lower being safer."""

    value = np.asarray(value, dtype=np.float64)
    soft = float(soft)
    hard = float(hard)
    if (
        not np.isfinite(value).all()
        or not np.isfinite((soft, hard)).all()
        or soft < 0.0
        or hard <= soft
    ):
        raise ValueError(
            "confidence mapping requires finite value and 0 <= soft < hard"
        )
    confidence = np.clip((hard - value) / (hard - soft), 0.0, 1.0)
    confidence = np.where(value <= soft, 1.0, confidence)
    return confidence


@dataclass(frozen=True)
class HybridSamplingReliabilityConfig:
    enabled: bool = False
    ensemble_disagreement_soft: float = 0.02
    ensemble_disagreement_hard: float = 0.10
    innovation_error_soft: float = 0.10
    innovation_error_hard: float = 0.40
    innovation_minimum_samples: int = 3
    actor_ood_soft: float = 3.0
    actor_ood_hard: float = 7.0
    medium_confidence: float = 0.33
    high_confidence: float = 0.67
    low_guided_fraction: float = 0.0
    medium_guided_fraction: float = 0.30
    high_guided_fraction: float = 0.60
    dynamics_power: float = 1.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        values = dict(values or {})
        result = cls(**{
            name: values.get(name, field.default)
            for name, field in cls.__dataclass_fields__.items()
        })
        result.validate()
        return result

    def validate(self):
        pairs = (
            (
                self.ensemble_disagreement_soft,
                self.ensemble_disagreement_hard,
                "ensemble disagreement",
            ),
            (
                self.innovation_error_soft,
                self.innovation_error_hard,
                "innovation error",
            ),
            (self.actor_ood_soft, self.actor_ood_hard, "Actor OOD"),
        )
        for soft, hard, name in pairs:
            if (
                not np.isfinite((soft, hard)).all()
                or soft < 0.0
                or hard <= soft
            ):
                raise ValueError(
                    "%s thresholds require 0 <= soft < hard" % name
                )
        if int(self.innovation_minimum_samples) < 0:
            raise ValueError(
                "innovation_minimum_samples must be non-negative"
            )
        if not (
            0.0
            <= float(self.medium_confidence)
            < float(self.high_confidence)
            <= 1.0
        ):
            raise ValueError(
                "confidence levels require 0 <= medium < high <= 1"
            )
        fractions = np.asarray(
            (
                self.low_guided_fraction,
                self.medium_guided_fraction,
                self.high_guided_fraction,
            ),
            dtype=np.float64,
        )
        if (
            not np.isfinite(fractions).all()
            or np.any(fractions < 0.0)
            or np.any(fractions >= 1.0)
            or not np.all(np.diff(fractions) >= 0.0)
        ):
            raise ValueError(
                "guided fractions must be monotone values in [0,1)"
            )
        if not np.isfinite(self.dynamics_power) or self.dynamics_power <= 0.0:
            raise ValueError("dynamics_power must be finite and positive")


class HybridSamplingReliability:
    """Evaluate one Actor mean rollout and assign discrete sampling authority."""

    def __init__(self, config=None):
        self.config = (
            config
            if isinstance(config, HybridSamplingReliabilityConfig)
            else HybridSamplingReliabilityConfig.from_mapping(config or {})
        )

    def evaluate(
        self,
        residual,
        states,
        controls,
        actor_ood_scores,
    ):
        if not self.config.enabled:
            raise RuntimeError(
                "disabled reliability evaluator must not be queried"
            )
        disagreement = getattr(residual, "disagreement", None)
        support = getattr(residual, "support_confidence", None)
        if not callable(disagreement) or not callable(support):
            raise TypeError(
                "reliability-calibrated HSS requires a residual ensemble"
            )
        states = np.asarray(states, dtype=np.float64)
        controls = np.asarray(controls, dtype=np.float64)
        actor_ood_scores = np.asarray(
            actor_ood_scores, dtype=np.float64
        ).reshape(-1)
        if (
            states.ndim != 2
            or controls.ndim != 2
            or states.shape[0] != controls.shape[0]
            or states.shape[0] != actor_ood_scores.size
            or states.shape[0] <= 0
        ):
            raise ValueError(
                "reliability rollout arrays must share a non-empty horizon"
            )
        disagreement_values = np.asarray(
            disagreement(states, controls), dtype=np.float64
        ).reshape(-1)
        support_values = np.asarray(
            support(states, controls), dtype=np.float64
        ).reshape(-1)
        if (
            disagreement_values.shape != actor_ood_scores.shape
            or support_values.shape != actor_ood_scores.shape
            or not np.isfinite(disagreement_values).all()
            or not np.isfinite(support_values).all()
            or not np.isfinite(actor_ood_scores).all()
        ):
            raise FloatingPointError(
                "reliability rollout signals are invalid"
            )
        disagreement_max = float(np.max(disagreement_values))
        support_min = float(np.clip(np.min(support_values), 0.0, 1.0))
        actor_ood_max = float(np.max(actor_ood_scores))
        disagreement_confidence = float(
            decreasing_linear_confidence(
                disagreement_max,
                self.config.ensemble_disagreement_soft,
                self.config.ensemble_disagreement_hard,
            )
        )
        innovation_samples = int(
            getattr(residual, "innovation_samples", 0)
        )
        innovation_error = float(
            getattr(residual, "innovation_error_ema", 0.0)
        )
        innovation_ready = bool(
            innovation_samples
            >= int(self.config.innovation_minimum_samples)
        )
        innovation_confidence = (
            float(
                decreasing_linear_confidence(
                    innovation_error,
                    self.config.innovation_error_soft,
                    self.config.innovation_error_hard,
                )
            )
            if innovation_ready
            else 1.0
        )
        dynamics_confidence = min(
            disagreement_confidence,
            innovation_confidence,
            support_min,
        )
        actor_confidence = float(
            decreasing_linear_confidence(
                actor_ood_max,
                self.config.actor_ood_soft,
                self.config.actor_ood_hard,
            )
        )
        authority = float(np.clip(
            dynamics_confidence ** self.config.dynamics_power
            * actor_confidence,
            0.0,
            1.0,
        ))
        if authority >= self.config.high_confidence:
            level = "high"
            guided_fraction = self.config.high_guided_fraction
        elif authority >= self.config.medium_confidence:
            level = "medium"
            guided_fraction = self.config.medium_guided_fraction
        else:
            level = "low"
            guided_fraction = self.config.low_guided_fraction
        return {
            "reliability_level": level,
            "reliability_authority": authority,
            "dynamics_confidence": float(dynamics_confidence),
            "actor_confidence": actor_confidence,
            "ensemble_disagreement_max": disagreement_max,
            "ensemble_disagreement_mean": float(
                np.mean(disagreement_values)
            ),
            "residual_support_confidence_min": support_min,
            "residual_support_confidence_mean": float(
                np.mean(support_values)
            ),
            "innovation_error_ema": innovation_error,
            "innovation_samples": innovation_samples,
            "innovation_ready": innovation_ready,
            "actor_ood_score_max": actor_ood_max,
            "actor_ood_score_mean": float(np.mean(actor_ood_scores)),
            "guided_fraction": float(guided_fraction),
        }
