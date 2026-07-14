"""Training and inference sampling priors for RL-guided MPPI."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from mobile_robot_mppi.policies.priors import GoalWarmStartPrior, PriorOutput
from .checkpointing import load_sac_checkpoint
from .observation import ObservationEncoder, ObservationEncoderConfig, RunningNormalizer
from .parameterization import PriorParameterization, PriorParameterizationConfig
from .sac import SACAgent, SACConfig


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
        )

    def validate(self):
        if self.mode not in ("none", "fixed", "ood", "exploration"):
            raise ValueError(
                "RL gate mode must be none, fixed, ood or exploration"
            )
        if self.exploration_signal not in ("ood", "critic_disagreement"):
            raise ValueError(
                "RL exploration signal must be ood or critic_disagreement"
            )
        if not 0.0 <= self.fixed_alpha <= 1.0:
            raise ValueError("RL fixed gate alpha must be in [0, 1]")
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
        metadata = dict(diagnostics)
        metadata.update({
            "type": "rl_external_training",
            "gate_mode": "fixed_training",
            "gate_alpha": effective_alpha,
            "base_gate_alpha": self.gate_alpha,
            "distance_gate_alpha": distance_alpha,
            "goal_distance": goal_distance,
        })
        return PriorOutput(mean, covariance, metadata)


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

    def reset(self):
        self.previous_action.fill(0.0)
        self.safety_override = False
        self.exploration_latch_alpha = 0.0
        self.exploration_start_time = None
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
        else:
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
        )

    def propose(self, observation, reference, horizon, action_spec):
        baseline = self.fallback_prior.propose(observation, reference, horizon, action_spec)
        raw = self.encoder.encode(
            observation,
            reference,
            previous_action=self.previous_action,
            safety_override=self.safety_override,
        )
        normalized = self.normalizer.normalize(raw)
        action, policy_diagnostics = self.agent.select_action(normalized, deterministic=True)
        learned_mean, covariance, decoder_diagnostics = self.parameterization.decode(
            action,
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
        (
            alpha,
            ood_score,
            critic_disagreement,
            distance_alpha,
            exploration_activation,
            exploration_trigger_open,
            exploration_elapsed,
        ) = self._gate(
            raw,
            normalized,
            action,
            goal_distance,
            observation.timestamp,
        )
        mean = baseline.mean + alpha * (learned_mean - baseline.mean)
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
            "previous_safety_override": self.safety_override,
        })
        return PriorOutput(mean, covariance, metadata)

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
