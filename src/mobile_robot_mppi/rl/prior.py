"""Training and inference sampling priors for RL-guided MPPI."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from mobile_robot_mppi.core.types import Pose2D, RobotObservation, Twist2D
from mobile_robot_mppi.policies.priors import (
    GoalWarmStartPrior,
    PriorOutput,
    ProposalDistribution,
)
from .checkpointing import load_sac_checkpoint
from .competence import (
    ProgressCompetence,
    ProgressCompetenceConfig,
    ProgressCompetenceGate,
)
from .observation import ObservationEncoder, ObservationEncoderConfig, RunningNormalizer
from .parameterization import PriorParameterization, PriorParameterizationConfig
from .sac import SACAgent, SACConfig
from .scene_complexity import SceneComplexityConfig, score_scene_complexity


@dataclass(frozen=True)
class GateConfig:
    mode: str = "none"
    fixed_alpha: float = 1.0
    ood_soft_threshold: float = 3.0
    ood_hard_threshold: float = 7.0
    critic_soft_threshold: float = 2.0
    critic_hard_threshold: float = 10.0
    use_critic_disagreement: bool = False
    exploration_signal: str = "critic_disagreement"
    exploration_soft_threshold: float = 0.05
    exploration_hard_threshold: float = 0.20
    exploration_latch: bool = True
    exploration_trigger_window_s: float = 0.50
    fallback_after_safety_override: bool = False
    near_goal_fallback_enabled: bool = False
    near_goal_full_fallback_distance: float = 0.60
    near_goal_full_rl_distance: float = 1.50
    correction_advantage_gate_mode: str = "none"
    correction_advantage_critic_source: str = "online"
    correction_advantage_threshold: float = 0.0
    correction_advantage_uncertainty_multiplier: float = 1.0
    correction_support_gate_enabled: bool = False
    correction_support_soft_threshold: float = 3.0
    correction_support_hard_threshold: float = 4.5
    skip_zero_complexity_inference: bool = False
    compute_unselected_critic_diagnostics: bool = True
    reuse_policy_base_action: bool = False
    profile_inference: bool = False
    temporal_closing_enabled: bool = False
    temporal_closing_source: str = "legacy_sector_minimum"
    temporal_closing_soft_mps: float = 0.05
    temporal_closing_hard_mps: float = 0.20
    temporal_closing_max_clearance_m: float = 1.50
    temporal_closing_hold_s: float = 0.60
    progress: ProgressCompetenceConfig = field(
        default_factory=ProgressCompetenceConfig
    )
    complexity: SceneComplexityConfig = field(
        default_factory=SceneComplexityConfig
    )

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        return cls(
            mode=str(values.get("mode", "none")),
            fixed_alpha=float(values.get("fixed_alpha", 1.0)),
            ood_soft_threshold=float(values.get("ood_soft_threshold", 3.0)),
            ood_hard_threshold=float(values.get("ood_hard_threshold", 7.0)),
            critic_soft_threshold=float(values.get("critic_soft_threshold", 2.0)),
            critic_hard_threshold=float(values.get("critic_hard_threshold", 10.0)),
            use_critic_disagreement=bool(values.get("use_critic_disagreement", False)),
            exploration_signal=str(
                values.get("exploration_signal", "critic_disagreement")
            ),
            exploration_soft_threshold=float(
                values.get("exploration_soft_threshold", 0.05)
            ),
            exploration_hard_threshold=float(
                values.get("exploration_hard_threshold", 0.20)
            ),
            exploration_latch=bool(values.get("exploration_latch", True)),
            exploration_trigger_window_s=float(
                values.get("exploration_trigger_window_s", 0.50)
            ),
            fallback_after_safety_override=bool(
                values.get("fallback_after_safety_override", False)
            ),
            near_goal_fallback_enabled=bool(
                values.get("near_goal_fallback_enabled", False)
            ),
            near_goal_full_fallback_distance=float(
                values.get("near_goal_full_fallback_distance", 0.60)
            ),
            near_goal_full_rl_distance=float(
                values.get("near_goal_full_rl_distance", 1.50)
            ),
            correction_advantage_gate_mode=str(
                values.get("correction_advantage_gate_mode", "none")
            ),
            correction_advantage_critic_source=str(
                values.get("correction_advantage_critic_source", "online")
            ),
            correction_advantage_threshold=float(
                values.get("correction_advantage_threshold", 0.0)
            ),
            correction_advantage_uncertainty_multiplier=float(
                values.get(
                    "correction_advantage_uncertainty_multiplier", 1.0
                )
            ),
            correction_support_gate_enabled=bool(
                values.get("correction_support_gate_enabled", False)
            ),
            correction_support_soft_threshold=float(
                values.get("correction_support_soft_threshold", 3.0)
            ),
            correction_support_hard_threshold=float(
                values.get("correction_support_hard_threshold", 4.5)
            ),
            skip_zero_complexity_inference=bool(
                values.get("skip_zero_complexity_inference", False)
            ),
            compute_unselected_critic_diagnostics=bool(
                values.get(
                    "compute_unselected_critic_diagnostics", True
                )
            ),
            reuse_policy_base_action=bool(
                values.get("reuse_policy_base_action", False)
            ),
            profile_inference=bool(values.get("profile_inference", False)),
            temporal_closing_enabled=bool(
                values.get("temporal_closing_enabled", False)
            ),
            temporal_closing_source=str(
                values.get(
                    "temporal_closing_source", "legacy_sector_minimum"
                )
            ),
            temporal_closing_soft_mps=float(
                values.get("temporal_closing_soft_mps", 0.05)
            ),
            temporal_closing_hard_mps=float(
                values.get("temporal_closing_hard_mps", 0.20)
            ),
            temporal_closing_max_clearance_m=float(
                values.get("temporal_closing_max_clearance_m", 1.50)
            ),
            temporal_closing_hold_s=float(
                values.get("temporal_closing_hold_s", 0.60)
            ),
            progress=ProgressCompetenceConfig.from_mapping(
                values.get("progress", {})
            ),
            complexity=SceneComplexityConfig.from_mapping(
                values.get("complexity", {})
            ),
        )

    def validate(self):
        if self.mode not in (
            "none",
            "fixed",
            "ood",
            "exploration",
            "complexity",
            "complexity_confidence",
            "progress_complexity",
        ):
            raise ValueError(
                "RL gate mode must be none, fixed, ood, exploration, "
                "complexity, complexity_confidence or progress_complexity"
            )
        if self.temporal_closing_source not in (
            "legacy_sector_minimum",
            "perception_scan_flow",
        ):
            raise ValueError(
                "temporal closing source must be legacy_sector_minimum or "
                "perception_scan_flow"
            )
        self.complexity.validate()
        self.progress.validate()
        if self.exploration_signal not in ("ood", "critic_disagreement"):
            raise ValueError(
                "RL exploration signal must be ood or critic_disagreement"
            )
        if self.correction_advantage_gate_mode not in (
            "none", "hard", "lcb", "base"
        ):
            raise ValueError(
                "correction advantage gate mode must be none, hard, lcb or base"
            )
        if self.correction_advantage_critic_source not in ("online", "target"):
            raise ValueError(
                "correction advantage critic source must be online or target"
            )
        if not np.isfinite(self.correction_advantage_threshold):
            raise ValueError("correction advantage threshold must be finite")
        if (
            not np.isfinite(
                self.correction_advantage_uncertainty_multiplier
            )
            or self.correction_advantage_uncertainty_multiplier < 1.0
        ):
            raise ValueError(
                "correction advantage uncertainty multiplier must be finite "
                "and at least one"
            )
        if (
            not np.isfinite(self.correction_support_soft_threshold)
            or not np.isfinite(self.correction_support_hard_threshold)
            or self.correction_support_soft_threshold < 0.0
            or self.correction_support_hard_threshold
            <= self.correction_support_soft_threshold
        ):
            raise ValueError(
                "correction support hard threshold must exceed its "
                "non-negative soft threshold"
            )
        if not 0.0 <= self.fixed_alpha <= 1.0:
            raise ValueError("RL fixed gate alpha must be in [0, 1]")
        if (
            not np.isfinite(self.temporal_closing_soft_mps)
            or not np.isfinite(self.temporal_closing_hard_mps)
            or self.temporal_closing_soft_mps < 0.0
            or self.temporal_closing_hard_mps
            <= self.temporal_closing_soft_mps
        ):
            raise ValueError(
                "temporal closing hard rate must exceed its non-negative soft rate"
            )
        if (
            not np.isfinite(self.temporal_closing_max_clearance_m)
            or self.temporal_closing_max_clearance_m <= 0.0
        ):
            raise ValueError("temporal closing clearance must be positive")
        if (
            not np.isfinite(self.temporal_closing_hold_s)
            or self.temporal_closing_hold_s < 0.0
        ):
            raise ValueError("temporal closing hold must be non-negative")
        pairs = (
            (self.ood_soft_threshold, self.ood_hard_threshold),
            (self.critic_soft_threshold, self.critic_hard_threshold),
        )
        if any(soft < 0.0 or hard <= soft for soft, hard in pairs):
            raise ValueError("RL gate hard thresholds must exceed non-negative soft thresholds")
        if (
            self.exploration_soft_threshold < 0.0
            or self.exploration_hard_threshold
            <= self.exploration_soft_threshold
        ):
            raise ValueError(
                "RL exploration hard threshold must exceed its non-negative soft threshold"
            )
        if (
            not np.isfinite(self.exploration_trigger_window_s)
            or self.exploration_trigger_window_s <= 0.0
        ):
            raise ValueError("RL exploration trigger window must be positive")
        if (
            self.near_goal_full_fallback_distance < 0.0
            or self.near_goal_full_rl_distance
            <= self.near_goal_full_fallback_distance
        ):
            raise ValueError(
                "near-goal full-RL distance must exceed non-negative fallback distance"
            )


def _linear_confidence(score, soft, hard):
    if score <= soft:
        return 1.0
    if score >= hard:
        return 0.0
    return float((hard - score) / (hard - soft))


def _linear_activation(score, soft, hard):
    if score <= soft:
        return 0.0
    if score >= hard:
        return 1.0
    return float((score - soft) / (hard - soft))


def _distance_gate_alpha(distance, config):
    if not config.near_goal_fallback_enabled:
        return 1.0
    low = config.near_goal_full_fallback_distance
    high = config.near_goal_full_rl_distance
    if distance <= low:
        return 0.0
    if distance >= high:
        return 1.0
    unit = float((distance - low) / (high - low))
    # Smoothstep avoids an abrupt prior jump as the robot crosses the gate.
    return unit * unit * (3.0 - 2.0 * unit)


class ExternalActionPrior:
    """Stateful prior used by the training environment.

    SAC selects a normalized latent action.  The environment installs it here,
    then ordinary MPPI calls :meth:`propose`; no final-control shortcut exists.
    """

    def __init__(
        self,
        parameterization,
        fallback_prior=None,
        gate_alpha=1.0,
        gate_config=None,
    ):
        self.parameterization = parameterization
        self.fallback_prior = fallback_prior or GoalWarmStartPrior()
        self.gate_alpha = float(gate_alpha)
        if not 0.0 <= self.gate_alpha <= 1.0:
            raise ValueError("training gate alpha must be in [0, 1]")
        self.gate_config = (
            gate_config
            if isinstance(gate_config, GateConfig)
            else GateConfig.from_mapping(gate_config)
        )
        self.gate_config.validate()
        self.parameters = np.zeros(parameterization.parameter_dimension, dtype=np.float32)
        self.previous_action = np.zeros(
            parameterization.action_spec.dimension, dtype=np.float64
        )

    def reset(self):
        self.parameters.fill(0.0)
        self.previous_action.fill(0.0)

    def observe_safety_decision(self, decision):
        value = np.asarray(
            decision.executed_control.values, dtype=np.float64
        ).reshape(-1)
        expected = (self.parameterization.action_spec.dimension,)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError("executed control does not match RL prior action space")
        self.previous_action = value.copy()

    def set_parameters(self, parameters):
        value = np.asarray(parameters, dtype=np.float32).reshape(-1)
        if value.shape != (self.parameterization.parameter_dimension,) or not np.isfinite(value).all():
            raise ValueError("external RL prior parameters are invalid")
        self.parameters = np.clip(value, -1.0, 1.0)

    def propose(self, observation, reference, horizon, action_spec):
        baseline = self.fallback_prior.propose(observation, reference, horizon, action_spec)
        learned_mean, covariance, diagnostics = self.parameterization.decode(
            self.parameters,
            horizon,
            baseline.mean,
            current_twist=observation.twist.as_array(),
            previous_control=self.previous_action,
        )
        target = reference.target_at(
            observation.timestamp, observation.pose.as_array()
        )
        goal_distance = float(np.hypot(
            target.pose.x - observation.pose.x,
            target.pose.y - observation.pose.y,
        ))
        distance_alpha = _distance_gate_alpha(goal_distance, self.gate_config)
        effective_alpha = self.gate_alpha * distance_alpha
        mean = baseline.mean + effective_alpha * (learned_mean - baseline.mean)
        covariance = self.parameterization.blend_covariance(
            covariance, effective_alpha, baseline.covariance
        )
        metadata = dict(diagnostics)
        metadata.update({
            "type": "rl_external_training",
            "gate_mode": "fixed_training",
            "gate_alpha": effective_alpha,
            "base_gate_alpha": self.gate_alpha,
            "distance_gate_alpha": distance_alpha,
            "goal_distance": goal_distance,
            "covariance_gate_alpha": effective_alpha,
        })
        return PriorOutput(
            mean,
            covariance,
            metadata,
            proposals=(
                ProposalDistribution("rl", learned_mean, covariance),
                ProposalDistribution(
                    "base", baseline.mean, baseline.covariance
                ),
            ),
        )


class TorchSACPrior:
    """Deterministic checkpoint inference with optional auditable fallback gate."""

    def __init__(
        self,
        agent,
        encoder,
        normalizer,
        parameterization,
        gate_config=None,
        fallback_prior=None,
        policy_id="sac_mppi_prior",
        checkpoint_path=None,
    ):
        self.agent = agent
        self.agent.eval()
        self.encoder = encoder
        self.normalizer = normalizer
        self.parameterization = parameterization
        self.gate_config = gate_config if isinstance(gate_config, GateConfig) else GateConfig.from_mapping(gate_config)
        self.gate_config.validate()
        self.fallback_prior = fallback_prior or GoalWarmStartPrior()
        self.policy_id = str(policy_id)
        self.checkpoint_path = None if checkpoint_path is None else str(Path(checkpoint_path).resolve())
        self.previous_action = np.zeros(encoder.action_spec.dimension, dtype=np.float64)
        self.safety_override = False
        self.exploration_latch_alpha = 0.0
        self.exploration_start_time = None
        self.previous_complexity_clearances = None
        self.previous_complexity_timestamp = None
        self.temporal_closing_hold_alpha = 0.0
        self.temporal_closing_hold_until = None
        self.progress_competence_gate = ProgressCompetenceGate(
            self.gate_config.progress
        )

    def reset(self):
        self.previous_action.fill(0.0)
        self.safety_override = False
        self.exploration_latch_alpha = 0.0
        self.exploration_start_time = None
        self.previous_complexity_clearances = None
        self.previous_complexity_timestamp = None
        self.temporal_closing_hold_alpha = 0.0
        self.temporal_closing_hold_until = None
        self.progress_competence_gate.reset()
        self.encoder.reset()

    def observe_safety_decision(self, decision):
        self.previous_action = np.asarray(decision.executed_control.values, dtype=np.float64).copy()
        self.safety_override = bool(decision.overridden)

    def _gate(
        self,
        raw_observation,
        normalized_observation,
        action,
        goal_distance,
        timestamp,
        complexity_score,
        temporal_closing_alpha=0.0,
        progress_stagnation_alpha=0.0,
    ):
        cfg = self.gate_config
        ood_score = self.normalizer.ood_score(raw_observation)
        needs_critic = bool(
            cfg.use_critic_disagreement
            or (
                cfg.mode == "exploration"
                and cfg.exploration_signal == "critic_disagreement"
            )
        )
        critic_disagreement = (
            self.agent.critic_disagreement(normalized_observation, action)
            if needs_critic
            else 0.0
        )
        exploration_activation = 0.0
        exploration_elapsed = 0.0
        exploration_trigger_open = False
        hazard_activation = 1.0
        competence_confidence = 1.0
        if cfg.mode == "none":
            alpha = 1.0
        elif cfg.mode == "fixed":
            alpha = cfg.fixed_alpha
        elif cfg.mode == "ood":
            alpha = _linear_confidence(
                ood_score, cfg.ood_soft_threshold, cfg.ood_hard_threshold
            )
            if cfg.use_critic_disagreement:
                alpha *= _linear_confidence(
                    critic_disagreement,
                    cfg.critic_soft_threshold,
                    cfg.critic_hard_threshold,
                )
        elif cfg.mode == "exploration":
            if self.exploration_start_time is None:
                self.exploration_start_time = float(timestamp)
            exploration_elapsed = max(
                0.0, float(timestamp) - self.exploration_start_time
            )
            exploration_trigger_open = bool(
                exploration_elapsed <= cfg.exploration_trigger_window_s + 1e-12
            )
            exploration_score = (
                ood_score
                if cfg.exploration_signal == "ood"
                else critic_disagreement
            )
            if exploration_trigger_open:
                exploration_activation = _linear_activation(
                    exploration_score,
                    cfg.exploration_soft_threshold,
                    cfg.exploration_hard_threshold,
                )
            if cfg.exploration_latch:
                self.exploration_latch_alpha = max(
                    self.exploration_latch_alpha, exploration_activation
                )
                alpha = self.exploration_latch_alpha
            else:
                alpha = exploration_activation
        else:
            hazard_activation = _linear_activation(
                complexity_score,
                cfg.complexity.soft_threshold,
                cfg.complexity.hard_threshold,
            )
            hazard_activation = max(
                hazard_activation, float(temporal_closing_alpha)
            )
            if cfg.mode == "complexity_confidence":
                competence_confidence = _linear_confidence(
                    ood_score,
                    cfg.ood_soft_threshold,
                    cfg.ood_hard_threshold,
                )
                if cfg.use_critic_disagreement:
                    competence_confidence *= _linear_confidence(
                        critic_disagreement,
                        cfg.critic_soft_threshold,
                        cfg.critic_hard_threshold,
                    )
                alpha = hazard_activation * competence_confidence
            elif cfg.mode == "progress_complexity":
                competence_confidence = float(progress_stagnation_alpha)
                alpha = hazard_activation * competence_confidence
            else:
                alpha = hazard_activation
        if cfg.fallback_after_safety_override and self.safety_override:
            alpha = 0.0
        distance_alpha = _distance_gate_alpha(goal_distance, cfg)
        alpha *= distance_alpha
        return (
            float(alpha),
            float(ood_score),
            float(critic_disagreement),
            float(distance_alpha),
            float(exploration_activation),
            bool(exploration_trigger_open),
            float(exploration_elapsed),
            float(hazard_activation),
            float(competence_confidence),
        )

    def _temporal_closing_risk(self, complexity, timestamp, observation=None):
        """Estimate local closing risk from consecutive LaserScan sectors.

        This deliberately uses only scan-derived sector clearances.  It does
        not identify an obstacle, access simulator geometry, or assume that
        an obstacle is static.  Positive risk means at least one nearby sector
        is closing; a short hold prevents one-frame beam reassociation from
        immediately dropping the learned prior.
        """

        current = np.asarray((
            complexity.front_clearance_m,
            complexity.left_clearance_m,
            complexity.right_clearance_m,
        ), dtype=np.float64)
        now = float(timestamp)
        rate = 0.0
        activation = 0.0
        cfg = self.gate_config
        if (
            cfg.temporal_closing_enabled
            and cfg.temporal_closing_source == "perception_scan_flow"
        ):
            auxiliary = dict(getattr(observation, "auxiliary", {}) or {})
            flow = dict(auxiliary.get("temporal_scan_flow", {}) or {})
            if flow:
                return (
                    float(flow.get("risk_alpha", 0.0)),
                    float(flow.get("closing_rate_mps", 0.0)),
                    bool(flow.get("held", False)),
                )
            return 0.0, 0.0, False
        previous = self.previous_complexity_clearances
        previous_time = self.previous_complexity_timestamp
        if (
            cfg.temporal_closing_enabled
            and complexity.scan_valid
            and previous is not None
            and previous_time is not None
        ):
            dt = now - float(previous_time)
            if np.isfinite(dt) and dt > 1e-9:
                closing = (previous - current) / dt
                nearby = current <= cfg.temporal_closing_max_clearance_m
                if np.any(nearby):
                    rate = max(0.0, float(np.max(closing[nearby])))
                activation = _linear_activation(
                    rate,
                    cfg.temporal_closing_soft_mps,
                    cfg.temporal_closing_hard_mps,
                )

        self.previous_complexity_clearances = current
        self.previous_complexity_timestamp = now
        held = bool(
            self.temporal_closing_hold_until is not None
            and now <= self.temporal_closing_hold_until + 1e-12
        )
        if activation > 0.0:
            self.temporal_closing_hold_alpha = max(
                self.temporal_closing_hold_alpha if held else 0.0,
                activation,
            )
            self.temporal_closing_hold_until = (
                now + cfg.temporal_closing_hold_s
            )
            held = cfg.temporal_closing_hold_s > 0.0
        elif not held:
            self.temporal_closing_hold_alpha = 0.0
            self.temporal_closing_hold_until = None
        output = max(activation, self.temporal_closing_hold_alpha)
        if not cfg.temporal_closing_enabled:
            output = 0.0
            held = False
        return float(output), float(rate), bool(held)

    def propose(self, observation, reference, horizon, action_spec):
        profiling = self.gate_config.profile_inference
        profile_started = time.perf_counter() if profiling else None
        stage_started = profile_started
        profile = {}

        def mark(name):
            nonlocal stage_started
            if not profiling:
                return
            now = time.perf_counter()
            profile["profile_prior_%s_ms" % name] = 1000.0 * (
                now - stage_started
            )
            stage_started = now

        baseline = self.fallback_prior.propose(observation, reference, horizon, action_spec)
        mark("fallback")
        raw = self.encoder.encode(
            observation,
            reference,
            previous_action=self.previous_action,
            safety_override=self.safety_override,
        )
        normalized = self.normalizer.normalize(raw)
        support_ood_score = self.normalizer.ood_score(raw)
        mark("encode_normalize")
        target = reference.target_at(
            observation.timestamp, observation.pose.as_array()
        )
        goal_distance = float(np.hypot(
            target.pose.x - observation.pose.x,
            target.pose.y - observation.pose.y,
        ))
        complexity = score_scene_complexity(
            observation.scan, self.gate_config.complexity
        )
        (
            temporal_closing_alpha,
            temporal_closing_rate,
            temporal_closing_held,
        ) = self._temporal_closing_risk(
            complexity, observation.timestamp, observation
        )
        progress_competence = ProgressCompetence(
            progress_m=0.0,
            window_coverage_s=0.0,
            ready=False,
            stagnation_activation=0.0,
            held=False,
        )
        if self.gate_config.mode == "progress_complexity":
            progress_competence = self.progress_competence_gate.update(
                observation.timestamp, goal_distance
            )
        mark("gate_features")
        complexity_alpha = _linear_activation(
            complexity.score,
            self.gate_config.complexity.soft_threshold,
            self.gate_config.complexity.hard_threshold,
        )
        distance_alpha = _distance_gate_alpha(goal_distance, self.gate_config)
        skip_learned = bool(
            self.gate_config.mode in (
                "complexity",
                "complexity_confidence",
                "progress_complexity",
            )
            and self.gate_config.skip_zero_complexity_inference
            and not self.parameterization.config.learn_covariance
            and (
                max(complexity_alpha, temporal_closing_alpha) <= 0.0
                or (
                    self.gate_config.mode == "progress_complexity"
                    and progress_competence.stagnation_activation <= 0.0
                )
                or distance_alpha <= 0.0
                or (
                    self.gate_config.fallback_after_safety_override
                    and self.safety_override
                )
            )
        )
        if skip_learned:
            metadata = dict(baseline.metadata)
            metadata.update({
                "type": "rl_sac",
                "policy_id": self.policy_id,
                "checkpoint": self.checkpoint_path,
                "gate_mode": self.gate_config.mode,
                "gate_alpha": 0.0,
                "ood_score": support_ood_score,
                "critic_disagreement": 0.0,
                "distance_gate_alpha": distance_alpha,
                "exploration_signal": self.gate_config.exploration_signal,
                "exploration_activation": 0.0,
                "exploration_latch_alpha": self.exploration_latch_alpha,
                "exploration_trigger_open": False,
                "exploration_elapsed_s": 0.0,
                "goal_distance": goal_distance,
                "previous_safety_override": self.safety_override,
                "correction_advantage_gate_mode": (
                    self.gate_config.correction_advantage_gate_mode
                ),
                "correction_advantage_critic_source": (
                    self.gate_config.correction_advantage_critic_source
                ),
                "correction_advantage_threshold": (
                    self.gate_config.correction_advantage_threshold
                ),
                "correction_advantage_uncertainty_multiplier": (
                    self.gate_config
                    .correction_advantage_uncertainty_multiplier
                ),
                "correction_advantage_gate_alpha": 0.0,
                "correction_support_gate_enabled": (
                    self.gate_config.correction_support_gate_enabled
                ),
                "correction_support_confidence": (
                    _linear_confidence(
                        support_ood_score,
                        self.gate_config.correction_support_soft_threshold,
                        self.gate_config.correction_support_hard_threshold,
                    )
                    if self.gate_config.correction_support_gate_enabled
                    else 1.0
                ),
                "correction_effective_gate_alpha": 0.0,
                "learned_inference_skipped": True,
                "scene_complexity_score": complexity.score,
                "scene_complexity_front_proximity": (
                    complexity.front_proximity
                ),
                "scene_complexity_constriction": complexity.constriction,
                "scene_complexity_density": complexity.density,
                "scene_complexity_front_clearance_m": (
                    complexity.front_clearance_m
                ),
                "scene_complexity_left_clearance_m": (
                    complexity.left_clearance_m
                ),
                "scene_complexity_right_clearance_m": (
                    complexity.right_clearance_m
                ),
                "scene_complexity_near_obstacle_fraction": (
                    complexity.near_obstacle_fraction
                ),
                "scene_complexity_scan_valid": complexity.scan_valid,
                "temporal_closing_gate_alpha": temporal_closing_alpha,
                "temporal_closing_rate_mps": temporal_closing_rate,
                "temporal_closing_held": temporal_closing_held,
                "hazard_activation": max(
                    complexity_alpha, temporal_closing_alpha
                ),
                "competence_confidence": 0.0,
                "baseline_progress_m": progress_competence.progress_m,
                "baseline_progress_window_s": (
                    progress_competence.window_coverage_s
                ),
                "baseline_progress_gate_ready": progress_competence.ready,
                "baseline_stagnation_activation": (
                    progress_competence.stagnation_activation
                ),
                "baseline_stagnation_held": progress_competence.held,
            })
            mark("fastpath_metadata")
            if profiling:
                profile.update({
                    "profile_prior_actor_ms": 0.0,
                    "profile_prior_advantage_ms": 0.0,
                    "profile_prior_decoder_ms": 0.0,
                    "profile_prior_outer_gate_ms": 0.0,
                    "profile_prior_total_ms": 1000.0 * (
                        time.perf_counter() - profile_started
                    ),
                })
                metadata.update(profile)
            return PriorOutput(
                baseline.mean,
                baseline.covariance,
                metadata,
                proposals=(
                    ProposalDistribution(
                        "base", baseline.mean, baseline.covariance
                    ),
                ),
            )
        if self.gate_config.reuse_policy_base_action:
            action, policy_diagnostics = self.agent.select_action(
                normalized, deterministic=True, include_internal=True
            )
        else:
            action, policy_diagnostics = self.agent.select_action(
                normalized, deterministic=True
            )
        reusable_base_action = policy_diagnostics.pop(
            "_base_action", None
        )
        mark("actor")
        correction_support_confidence = 1.0
        correction_effective_gate_alpha = 1.0
        if bool(getattr(self.agent, "is_correction_policy", False)):
            advantage_kwargs = {}
            if not self.gate_config.compute_unselected_critic_diagnostics:
                advantage_kwargs[
                    "compute_unselected_diagnostics"
                ] = False
            if reusable_base_action is not None:
                advantage_kwargs["base_action"] = reusable_base_action
            action, advantage_diagnostics = (
                self.agent.filter_correction_by_advantage(
                    normalized,
                    action,
                    gate_mode=(
                        self.gate_config.correction_advantage_gate_mode
                    ),
                    critic_source=(
                        self.gate_config.correction_advantage_critic_source
                    ),
                    threshold=(
                        self.gate_config.correction_advantage_threshold
                    ),
                    uncertainty_multiplier=(
                        self.gate_config
                        .correction_advantage_uncertainty_multiplier
                    ),
                    **advantage_kwargs
                )
            )
            policy_diagnostics.update(advantage_diagnostics)
            correction_effective_gate_alpha = float(
                advantage_diagnostics["correction_advantage_gate_alpha"]
            )
            if self.gate_config.correction_support_gate_enabled:
                base_action = (
                    reusable_base_action
                    if reusable_base_action is not None
                    else self.agent.frozen_base_action(normalized)
                )
                correction_support_confidence = _linear_confidence(
                    support_ood_score,
                    self.gate_config.correction_support_soft_threshold,
                    self.gate_config.correction_support_hard_threshold,
                )
                action = base_action + correction_support_confidence * (
                    action - base_action
                )
                effective_correction = action - base_action
                policy_diagnostics.update({
                    "applied_correction_abs_mean": float(
                        np.mean(np.abs(effective_correction))
                    ),
                    "applied_correction_abs_max": float(
                        np.max(np.abs(effective_correction))
                    ),
                })
                correction_effective_gate_alpha *= (
                    correction_support_confidence
                )
        elif self.gate_config.correction_support_gate_enabled:
            raise ValueError(
                "correction support gate requires a frozen-BC correction "
                "policy checkpoint"
            )
        policy_diagnostics.update({
            "correction_support_gate_enabled": (
                self.gate_config.correction_support_gate_enabled
            ),
            "correction_support_confidence": (
                correction_support_confidence
            ),
            "correction_effective_gate_alpha": (
                correction_effective_gate_alpha
            ),
        })
        mark("advantage")
        learned_mean, covariance, decoder_diagnostics = self.parameterization.decode(
            action,
            horizon,
            baseline.mean,
            current_twist=observation.twist.as_array(),
            previous_control=self.previous_action,
        )
        mark("decoder")
        (
            alpha,
            ood_score,
            critic_disagreement,
            distance_alpha,
            exploration_activation,
            exploration_trigger_open,
            exploration_elapsed,
            hazard_activation,
            competence_confidence,
        ) = self._gate(
            raw,
            normalized,
            action,
            goal_distance,
            observation.timestamp,
            complexity.score,
            temporal_closing_alpha,
            progress_competence.stagnation_activation,
        )
        mark("outer_gate")
        mean = baseline.mean + alpha * (learned_mean - baseline.mean)
        covariance = self.parameterization.blend_covariance(
            covariance, alpha, baseline.covariance
        )
        metadata = dict(decoder_diagnostics)
        metadata.update(policy_diagnostics)
        metadata.update({
            "type": "rl_sac",
            "policy_id": self.policy_id,
            "checkpoint": self.checkpoint_path,
            "gate_mode": self.gate_config.mode,
            "gate_alpha": alpha,
            "ood_score": ood_score,
            "critic_disagreement": critic_disagreement,
            "distance_gate_alpha": distance_alpha,
            "exploration_signal": self.gate_config.exploration_signal,
            "exploration_activation": exploration_activation,
            "exploration_latch_alpha": self.exploration_latch_alpha,
            "exploration_trigger_open": exploration_trigger_open,
            "exploration_elapsed_s": exploration_elapsed,
            "goal_distance": goal_distance,
            "covariance_gate_alpha": alpha,
            "previous_safety_override": self.safety_override,
            "learned_inference_skipped": False,
            "scene_complexity_score": complexity.score,
            "scene_complexity_front_proximity": (
                complexity.front_proximity
            ),
            "scene_complexity_constriction": complexity.constriction,
            "scene_complexity_density": complexity.density,
            "scene_complexity_front_clearance_m": (
                complexity.front_clearance_m
            ),
            "scene_complexity_left_clearance_m": (
                complexity.left_clearance_m
            ),
            "scene_complexity_right_clearance_m": (
                complexity.right_clearance_m
            ),
            "scene_complexity_near_obstacle_fraction": (
                complexity.near_obstacle_fraction
            ),
            "scene_complexity_scan_valid": complexity.scan_valid,
            "temporal_closing_gate_alpha": temporal_closing_alpha,
            "temporal_closing_rate_mps": temporal_closing_rate,
            "temporal_closing_held": temporal_closing_held,
            "hazard_activation": hazard_activation,
            "competence_confidence": competence_confidence,
            "baseline_progress_m": progress_competence.progress_m,
            "baseline_progress_window_s": (
                progress_competence.window_coverage_s
            ),
            "baseline_progress_gate_ready": progress_competence.ready,
            "baseline_stagnation_activation": (
                progress_competence.stagnation_activation
            ),
            "baseline_stagnation_held": progress_competence.held,
        })
        mark("metadata")
        if profiling:
            profile["profile_prior_total_ms"] = 1000.0 * (
                time.perf_counter() - profile_started
            )
            metadata.update(profile)
        return PriorOutput(
            mean,
            covariance,
            metadata,
            proposals=(
                ProposalDistribution("rl", learned_mean, covariance),
                ProposalDistribution(
                    "base", baseline.mean, baseline.covariance
                ),
            ),
        )

    def terminal_value(
        self,
        terminal_states,
        terminal_controls,
        observation,
        target,
        state_spec,
        horizon_dt,
        critic_source="target",
    ):
        """Estimate long-horizon return for predicted MPPI terminal states.

        The critic was trained on the same high-level prior action used by the
        SAC environment.  Each hypothetical terminal observation is therefore
        passed through the deterministic actor before twin-Q evaluation.  The
        encoder history is never mutated.  LaserScan is held at the latest
        real observation in this simple baseline; the limitation is exposed in
        diagnostics rather than hidden.
        """

        states = np.asarray(terminal_states, dtype=np.float64)
        controls = np.asarray(terminal_controls, dtype=np.float64)
        if (
            states.ndim != 2
            or states.shape[1] != state_spec.dimension
            or states.shape[0] <= 0
            or not np.isfinite(states).all()
        ):
            raise ValueError("terminal states must be finite [B, state_dim]")
        if (
            controls.ndim != 2
            or controls.shape
            != (states.shape[0], self.parameterization.action_spec.dimension)
            or not np.isfinite(controls).all()
        ):
            raise ValueError("terminal controls must be finite [B, action_dim]")

        x_index = state_spec.index("x")
        y_index = state_spec.index("y")
        theta_index = state_spec.index("theta")
        v_index = state_spec.index("v") if "v" in state_spec.names else None
        omega_index = (
            state_spec.index("omega") if "omega" in state_spec.names else None
        )
        raw_rows = []
        for state, previous_control in zip(states, controls):
            hypothetical = RobotObservation(
                timestamp=float(observation.timestamp) + float(horizon_dt),
                pose=Pose2D(
                    float(state[x_index]),
                    float(state[y_index]),
                    float(state[theta_index]),
                ),
                twist=Twist2D(
                    float(observation.twist.v if v_index is None else state[v_index]),
                    float(
                        observation.twist.omega
                        if omega_index is None
                        else state[omega_index]
                    ),
                ),
                scan=observation.scan,
                local_obstacles=observation.local_obstacles,
                auxiliary=dict(observation.auxiliary),
            )
            raw_rows.append(
                self.encoder.encode_to_target(
                    hypothetical,
                    target,
                    previous_action=previous_control,
                    safety_override=False,
                    update_history=False,
                )
            )
        raw_batch = np.stack(raw_rows).astype(np.float32, copy=False)
        normalized = np.stack([
            self.normalizer.normalize(row) for row in raw_batch
        ]).astype(np.float32, copy=False)
        latent_actions = self.agent.select_action_batch(
            normalized, deterministic=True
        )
        values = self.agent.expected_twin_q(
            normalized, latent_actions, critic_source=critic_source
        )
        result = np.asarray(values["minimum"], dtype=np.float64)
        if result.shape != (states.shape[0],) or not np.isfinite(result).all():
            raise FloatingPointError("terminal critic returned invalid values")
        diagnostics = {
            "terminal_q_mean": float(np.mean(result)),
            "terminal_q_min": float(np.min(result)),
            "terminal_q_max": float(np.max(result)),
            "terminal_q_disagreement_mean": float(
                np.mean(values["disagreement"])
            ),
            "terminal_critic_source": str(values["critic_source"]),
            "terminal_scan_assumption": "latest_observed_scan",
        }
        return result, diagnostics

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        action_spec,
        base_noise_sigma,
        device="cpu",
        gate_config=None,
        fallback_prior=None,
        policy_id="sac_mppi_prior",
    ):
        payload = load_sac_checkpoint(checkpoint_path, map_location=device)
        saved_action = payload["action_spec"]
        if tuple(saved_action["names"]) != tuple(action_spec.names):
            raise ValueError("RL checkpoint action names do not match experiment")
        if not np.allclose(saved_action["lower"], action_spec.lower) or not np.allclose(
            saved_action["upper"], action_spec.upper
        ):
            raise ValueError("RL checkpoint action bounds do not match experiment")
        encoder_config = ObservationEncoderConfig.from_mapping(payload["encoder_config"])
        encoder = ObservationEncoder(encoder_config, action_spec)
        parameter_config = PriorParameterizationConfig.from_mapping(
            payload["parameterization_config"]
        )
        parameterization = PriorParameterization(
            parameter_config, action_spec, base_noise_sigma
        )
        agent_state = payload["agent"]
        sac_config = SACConfig.from_mapping(agent_state["config"])
        agent = SACAgent(
            agent_state["observation_dim"],
            agent_state["action_dim"],
            sac_config,
            device=device,
        )
        if encoder.dimension != agent.observation_dim:
            raise ValueError("RL checkpoint encoder and actor observation dimensions differ")
        if parameterization.parameter_dimension != agent.action_dim:
            raise ValueError("RL checkpoint decoder and actor action dimensions differ")
        agent.load_state_dict(agent_state, load_optimizers=False)
        normalizer = RunningNormalizer.from_state_dict(payload["normalizer"])
        return cls(
            agent,
            encoder,
            normalizer,
            parameterization,
            gate_config=gate_config,
            fallback_prior=fallback_prior,
            policy_id=policy_id,
            checkpoint_path=checkpoint_path,
        )
