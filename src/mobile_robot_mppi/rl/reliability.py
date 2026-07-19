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


def fuse_hybrid_confidence(
    disagreement_confidence,
    innovation_confidence,
    support_confidence,
    actor_confidence,
    innovation_ready,
    mode="conservative_min",
):
    """Fuse calibrated signals without changing legacy behavior by default."""

    values = np.asarray(
        (
            disagreement_confidence,
            innovation_confidence,
            support_confidence,
            actor_confidence,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(
        values > 1.0
    ):
        raise ValueError("hybrid confidence inputs must lie in [0,1]")
    mode = str(mode)
    if mode == "conservative_min":
        dynamics = min(values[0], values[1], values[2])
        actor_factor = values[3]
    elif mode == "innovation_anchor":
        # Completed-transition innovation is causal and was the only signal
        # with consistent error ranking across the L190 path splits.  Before
        # enough transitions exist, retain the conservative ensemble/support
        # fallback.  Actor support becomes a hard out-of-support veto instead
        # of continuously penalizing legitimate held-out route geometry.
        dynamics = (
            values[1]
            if bool(innovation_ready)
            else min(values[0], values[2])
        )
        actor_factor = 1.0 if values[3] > 0.0 else 0.0
    else:
        raise ValueError("unknown hybrid confidence fusion mode: %s" % mode)
    return float(dynamics), float(actor_factor)


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
    dynamics_routing_mode: str = "trust_weighted"
    policy_rescue_floor: float = 0.50
    fusion_mode: str = "conservative_min"
    source_competence_enabled: bool = False
    source_competence_initial: float = 0.50
    source_competence_decay: float = 0.90
    source_competence_prior_success: float = 1.0
    source_competence_prior_failure: float = 1.0
    source_competence_ratio_off: float = 0.0
    source_competence_ratio_on: float = 1.0

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
        if self.dynamics_routing_mode not in (
            "trust_weighted",
            "policy_rescue",
        ):
            raise ValueError(
                "dynamics_routing_mode must be trust_weighted or "
                "policy_rescue"
            )
        if (
            not np.isfinite(self.policy_rescue_floor)
            or not 0.0 <= float(self.policy_rescue_floor) <= 1.0
        ):
            raise ValueError("policy_rescue_floor must lie in [0,1]")
        if self.fusion_mode not in (
            "conservative_min",
            "innovation_anchor",
        ):
            raise ValueError(
                "fusion_mode must be conservative_min or innovation_anchor"
            )
        if not 0.0 <= float(self.source_competence_initial) <= 1.0:
            raise ValueError(
                "source_competence_initial must lie in [0,1]"
            )
        if not 0.0 <= float(self.source_competence_decay) < 1.0:
            raise ValueError(
                "source_competence_decay must lie in [0,1)"
            )
        if (
            not np.isfinite((
                self.source_competence_prior_success,
                self.source_competence_prior_failure,
            )).all()
            or float(self.source_competence_prior_success) <= 0.0
            or float(self.source_competence_prior_failure) <= 0.0
        ):
            raise ValueError(
                "source competence Beta prior must be finite and positive"
            )
        if (
            not np.isfinite((
                self.source_competence_ratio_off,
                self.source_competence_ratio_on,
            )).all()
            or float(self.source_competence_ratio_off) < 0.0
            or float(self.source_competence_ratio_on)
            <= float(self.source_competence_ratio_off)
        ):
            raise ValueError(
                "source competence calibration requires finite "
                "0 <= ratio_off < ratio_on"
            )


class SourceRelativeCompetence:
    """Causal Actor-source competence from MPPI elite opportunity rates."""

    def __init__(self, config=None):
        self.config = (
            config
            if isinstance(config, HybridSamplingReliabilityConfig)
            else HybridSamplingReliabilityConfig.from_mapping(config or {})
        )
        self.reset()

    def reset(self):
        self.confidence = float(self.config.source_competence_initial)
        self.updates = 0

    def update(
        self,
        guided_elites,
        gaussian_elites,
        guided_opportunities,
        gaussian_opportunities,
    ):
        values = np.asarray(
            (
                guided_elites,
                gaussian_elites,
                guided_opportunities,
                gaussian_opportunities,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(
                "source competence counts must be finite and nonnegative"
            )
        guided_elites, gaussian_elites, guided_total, gaussian_total = (
            values.tolist()
        )
        if (
            guided_elites > guided_total
            or gaussian_elites > gaussian_total
        ):
            raise ValueError(
                "source competence elites cannot exceed opportunities"
            )
        if guided_total <= 0.0 or gaussian_total <= 0.0:
            return {
                "updated": False,
                "raw_confidence": self.confidence,
                "mapped_confidence": self.confidence,
                "confidence": self.confidence,
                "guided_yield": 0.0,
                "gaussian_yield": 0.0,
                "updates": self.updates,
            }
        alpha = float(self.config.source_competence_prior_success)
        beta = float(self.config.source_competence_prior_failure)
        guided_yield = (guided_elites + alpha) / (
            guided_total + alpha + beta
        )
        gaussian_yield = (gaussian_elites + alpha) / (
            gaussian_total + alpha + beta
        )
        raw_ratio = float(np.clip(
            guided_yield / max(gaussian_yield, 1e-12),
            0.0,
            1.0,
        ))
        ratio_off = float(self.config.source_competence_ratio_off)
        ratio_on = float(self.config.source_competence_ratio_on)
        mapped = float(np.clip(
            (raw_ratio - ratio_off) / (ratio_on - ratio_off),
            0.0,
            1.0,
        ))
        decay = float(self.config.source_competence_decay)
        self.confidence = float(np.clip(
            decay * self.confidence + (1.0 - decay) * mapped,
            0.0,
            1.0,
        ))
        self.updates += 1
        return {
            "updated": True,
            "raw_confidence": raw_ratio,
            "mapped_confidence": mapped,
            "confidence": self.confidence,
            "guided_yield": float(guided_yield),
            "gaussian_yield": float(gaussian_yield),
            "updates": self.updates,
        }


class HybridSamplingReliability:
    """Evaluate one Actor mean rollout and assign discrete sampling authority."""

    def __init__(self, config=None):
        self.config = (
            config
            if isinstance(config, HybridSamplingReliabilityConfig)
            else HybridSamplingReliabilityConfig.from_mapping(config or {})
        )

    def allocation_from_authority(self, authority):
        """Map one finite authority score to the frozen discrete allocation."""

        authority = float(authority)
        if not np.isfinite(authority) or not 0.0 <= authority <= 1.0:
            raise ValueError("reliability authority must lie in [0,1]")
        if authority >= self.config.high_confidence:
            return "high", float(self.config.high_guided_fraction)
        if authority >= self.config.medium_confidence:
            return "medium", float(self.config.medium_guided_fraction)
        return "low", float(self.config.low_guided_fraction)

    def authority_from_components(
        self, dynamics_confidence, actor_authority_factor
    ):
        """Fuse role-specific factors using the configured routing rule."""

        values = np.asarray(
            (dynamics_confidence, actor_authority_factor),
            dtype=np.float64,
        )
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError(
                "routing confidence factors must lie in [0,1]"
            )
        calibrated_dynamics = float(
            float(dynamics_confidence) ** self.config.dynamics_power
        )
        if self.config.dynamics_routing_mode == "trust_weighted":
            model_routing_factor = calibrated_dynamics
        else:
            # A model-free Actor and learned rollout model have different
            # failure modes.  If the Actor is supported and competitive, low
            # ICODE confidence allocates more Actor proposals; the configured
            # floor retains a bounded allocation when ICODE is confident.
            floor = float(self.config.policy_rescue_floor)
            model_routing_factor = float(
                floor + (1.0 - floor) * (1.0 - calibrated_dynamics)
            )
        authority = float(np.clip(
            model_routing_factor * float(actor_authority_factor),
            0.0,
            1.0,
        ))
        return authority, model_routing_factor

    def evaluate(
        self,
        residual,
        states,
        controls,
        actor_ood_scores,
        actor_competence_confidence=1.0,
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
        actor_confidence = float(
            decreasing_linear_confidence(
                actor_ood_max,
                self.config.actor_ood_soft,
                self.config.actor_ood_hard,
            )
        )
        dynamics_confidence, actor_authority_factor = (
            fuse_hybrid_confidence(
                disagreement_confidence,
                innovation_confidence,
                support_min,
                actor_confidence,
                innovation_ready,
                self.config.fusion_mode,
            )
        )
        actor_competence_confidence = float(
            actor_competence_confidence
        )
        if (
            not np.isfinite(actor_competence_confidence)
            or not 0.0 <= actor_competence_confidence <= 1.0
        ):
            raise ValueError(
                "Actor competence confidence must lie in [0,1]"
            )
        actor_support_authority_factor = actor_authority_factor
        actor_authority_factor *= actor_competence_confidence
        authority, model_routing_factor = self.authority_from_components(
            dynamics_confidence, actor_authority_factor
        )
        level, guided_fraction = self.allocation_from_authority(authority)
        return {
            "reliability_level": level,
            "reliability_authority": authority,
            "dynamics_confidence": float(dynamics_confidence),
            "actor_confidence": actor_confidence,
            "actor_support_authority_factor": (
                actor_support_authority_factor
            ),
            "actor_competence_confidence": (
                actor_competence_confidence
            ),
            "actor_authority_factor": actor_authority_factor,
            "dynamics_routing_mode": self.config.dynamics_routing_mode,
            "model_routing_factor": model_routing_factor,
            "fusion_mode": self.config.fusion_mode,
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


@dataclass(frozen=True)
class ConservativeTerminalReliabilityConfig:
    """Candidate-level confidence for the incremental SAC terminal cost.

    These values are bounded engineering scores rather than calibrated
    probabilities.  When disabled, the paper controller preserves the fixed
    terminal-value behavior byte-for-byte.
    """

    enabled: bool = False
    ensemble_disagreement_soft: float = 0.02
    ensemble_disagreement_hard: float = 0.10
    innovation_error_soft: float = 0.10
    innovation_error_hard: float = 0.40
    innovation_minimum_samples: int = 3
    critic_ood_soft: float = 3.0
    critic_ood_hard: float = 7.0
    critic_disagreement_soft: float = 0.5
    critic_disagreement_hard: float = 2.0
    use_critic_support: bool = True
    use_critic_disagreement: bool = True
    uncertainty_penalty_weight: float = 0.0

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
            (
                self.critic_ood_soft,
                self.critic_ood_hard,
                "critic OOD",
            ),
            (
                self.critic_disagreement_soft,
                self.critic_disagreement_hard,
                "critic disagreement",
            ),
        )
        for soft, hard, name in pairs:
            if (
                not np.isfinite((soft, hard)).all()
                or float(soft) < 0.0
                or float(hard) <= float(soft)
            ):
                raise ValueError(
                    "%s thresholds require 0 <= soft < hard" % name
                )
        if int(self.innovation_minimum_samples) < 0:
            raise ValueError(
                "innovation_minimum_samples must be non-negative"
            )
        if (
            not np.isfinite(self.uncertainty_penalty_weight)
            or float(self.uncertainty_penalty_weight) < 0.0
        ):
            raise ValueError(
                "uncertainty_penalty_weight must be finite and non-negative"
            )


class ConservativeTerminalReliability:
    """Evaluate terminal candidates without simulator truth or future data."""

    def __init__(self, config=None):
        self.config = (
            config
            if isinstance(config, ConservativeTerminalReliabilityConfig)
            else ConservativeTerminalReliabilityConfig.from_mapping(
                config or {}
            )
        )

    def evaluate(
        self,
        residual,
        states,
        controls,
        critic_ood_scores,
        critic_disagreement,
    ):
        if not self.config.enabled:
            raise RuntimeError(
                "disabled conservative terminal evaluator must not be queried"
            )
        disagreement = getattr(residual, "disagreement", None)
        support = getattr(residual, "support_confidence", None)
        if not callable(disagreement) or not callable(support):
            raise TypeError(
                "conservative terminal requires an ensemble residual"
            )
        states = np.asarray(states, dtype=np.float64)
        controls = np.asarray(controls, dtype=np.float64)
        critic_ood_scores = np.asarray(
            critic_ood_scores, dtype=np.float64
        ).reshape(-1)
        critic_disagreement = np.asarray(
            critic_disagreement, dtype=np.float64
        ).reshape(-1)
        count = states.shape[0] if states.ndim == 2 else 0
        if (
            count <= 0
            or controls.ndim != 2
            or controls.shape[0] != count
            or critic_ood_scores.shape != (count,)
            or critic_disagreement.shape != (count,)
            or not np.isfinite(states).all()
            or not np.isfinite(controls).all()
            or not np.isfinite(critic_ood_scores).all()
            or not np.isfinite(critic_disagreement).all()
        ):
            raise ValueError(
                "terminal reliability arrays must share a finite batch"
            )
        dynamics_disagreement = np.asarray(
            disagreement(states, controls), dtype=np.float64
        ).reshape(-1)
        dynamics_support = np.asarray(
            support(states, controls), dtype=np.float64
        ).reshape(-1)
        if (
            dynamics_disagreement.shape != (count,)
            or dynamics_support.shape != (count,)
            or not np.isfinite(dynamics_disagreement).all()
            or not np.isfinite(dynamics_support).all()
        ):
            raise FloatingPointError(
                "terminal ICODE reliability signals are invalid"
            )
        disagreement_confidence = decreasing_linear_confidence(
            dynamics_disagreement,
            self.config.ensemble_disagreement_soft,
            self.config.ensemble_disagreement_hard,
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
        dynamics_confidence = np.minimum(
            disagreement_confidence,
            np.clip(dynamics_support, 0.0, 1.0),
        )
        dynamics_confidence = np.minimum(
            dynamics_confidence, innovation_confidence
        )
        critic_support_confidence = (
            decreasing_linear_confidence(
                critic_ood_scores,
                self.config.critic_ood_soft,
                self.config.critic_ood_hard,
            )
            if self.config.use_critic_support
            else np.ones(count, dtype=np.float64)
        )
        critic_agreement_confidence = (
            decreasing_linear_confidence(
                critic_disagreement,
                self.config.critic_disagreement_soft,
                self.config.critic_disagreement_hard,
            )
            if self.config.use_critic_disagreement
            else np.ones(count, dtype=np.float64)
        )
        critic_confidence = np.minimum(
            critic_support_confidence, critic_agreement_confidence
        )
        authority = np.clip(
            dynamics_confidence * critic_confidence, 0.0, 1.0
        )
        uncertainty = np.clip(
            (
                dynamics_disagreement
                - self.config.ensemble_disagreement_soft
            )
            / (
                self.config.ensemble_disagreement_hard
                - self.config.ensemble_disagreement_soft
            ),
            0.0,
            1.0,
        )
        if not all(
            np.isfinite(values).all()
            for values in (
                authority,
                dynamics_confidence,
                critic_confidence,
                uncertainty,
            )
        ):
            raise FloatingPointError(
                "terminal reliability produced NaN or Inf"
            )
        return {
            "authority": authority,
            "dynamics_confidence": dynamics_confidence,
            "critic_confidence": critic_confidence,
            "critic_support_confidence": critic_support_confidence,
            "critic_agreement_confidence": critic_agreement_confidence,
            "dynamics_disagreement": dynamics_disagreement,
            "dynamics_support_confidence": dynamics_support,
            "dynamics_uncertainty": uncertainty,
            "innovation_error": innovation_error,
            "innovation_samples": innovation_samples,
            "innovation_ready": innovation_ready,
        }
