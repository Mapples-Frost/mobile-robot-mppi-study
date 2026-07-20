"""Paper-faithful low-level Actor and distributional-Critic inference.

This module is intentionally separate from the legacy high-level MPPI-prior
policy.  The Actor action is normalized internally for SAC, but represents the
physical command ``[v_cmd, omega_cmd]`` after an affine bounds transform.
"""

from pathlib import Path

import numpy as np

from mobile_robot_mppi.core.types import Pose2D, RobotObservation, Twist2D
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior

from .checkpointing import load_sac_checkpoint
from .observation import (
    ObservationEncoder,
    ObservationEncoderConfig,
    RunningNormalizer,
)
from .sac import SACAgent, SACConfig
from .residual_context import ResidualCorrectionAuthority


class PaperDirectControlPolicy:
    """Frozen physical-control policy/value interface for RL-Driven MPPI."""

    def __init__(
        self,
        agent,
        encoder,
        normalizer,
        action_spec,
        fallback_prior=None,
        checkpoint_path=None,
        residual_context=None,
        residual_correction_authority=None,
    ):
        if int(agent.action_dim) != int(action_spec.dimension):
            raise ValueError(
                "direct Actor action dimension must match physical action space"
            )
        if bool(getattr(agent, "is_correction_policy", False)) and not bool(
            getattr(agent, "base_actor_initialized", False)
        ):
            raise ValueError(
                "bounded correction policy requires an initialized frozen base Actor"
            )
        self.agent = agent
        self.agent.eval()
        self.encoder = encoder
        self.normalizer = normalizer
        self.action_spec = action_spec
        self.fallback_prior = fallback_prior or GoalWarmStartPrior()
        self.checkpoint_path = (
            None
            if checkpoint_path is None
            else str(Path(checkpoint_path).resolve())
        )
        self.residual_context = residual_context
        self.residual_correction_authority = None
        context_enabled = bool(self.encoder.config.include_residual_context)
        if context_enabled != (self.residual_context is not None):
            raise ValueError(
                "checkpoint residual-context observation and runtime context "
                "provider must be enabled together"
            )
        if (
            context_enabled
            and int(self.residual_context.dimension)
            != int(self.encoder.config.residual_context_dimension)
        ):
            raise ValueError("residual context provider dimension differs from checkpoint")
        authority_mapping = dict(residual_correction_authority or {})
        if bool(authority_mapping.get("enabled", False)):
            if self.residual_context is None:
                raise ValueError(
                    "residual correction authority requires residual context"
                )
            if not bool(getattr(self.agent, "is_correction_policy", False)):
                raise ValueError(
                    "residual correction authority requires a correction Actor"
                )
            self.residual_correction_authority = ResidualCorrectionAuthority(
                self.residual_context.dimension, authority_mapping
            )
        self.previous_action = np.zeros(
            self.action_spec.dimension, dtype=np.float64
        )
        self.safety_override = False
        if int(self.encoder.config.history_frames) != 1:
            raise ValueError(
                "Gate 1 autoregressive policy currently requires "
                "observation.history_frames=1"
            )

    @property
    def action_center(self):
        return 0.5 * (self.action_spec.lower + self.action_spec.upper)

    @property
    def action_half_range(self):
        return 0.5 * (self.action_spec.upper - self.action_spec.lower)

    def reset(self):
        self.previous_action.fill(0.0)
        self.safety_override = False
        self.encoder.reset()
        if self.residual_context is not None:
            self.residual_context.reset()

    def set_previous(self, sequence):
        values = np.asarray(sequence, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != self.action_spec.dimension
            or values.shape[0] <= 0
            or not np.isfinite(values).all()
        ):
            raise ValueError("previous direct-control sequence is invalid")
        self.previous_action = values[0].copy()

    def observe_safety_decision(self, decision):
        values = np.asarray(
            decision.executed_control.values, dtype=np.float64
        ).reshape(-1)
        if (
            values.shape != (self.action_spec.dimension,)
            or not np.isfinite(values).all()
        ):
            raise ValueError("executed control does not match policy action space")
        self.previous_action = values.copy()
        self.safety_override = bool(decision.overridden)

    def propose(self, observation, reference, horizon, action_spec):
        """Supply a valid placeholder prior for the common controller API.

        The paper-faithful controller replaces this mean with an autoregressive
        Actor rollout before sampling.  Keeping ``propose`` side-effect free
        preserves the standard MPPI planner contract and provides a trusted
        fallback for diagnostics.
        """

        if tuple(action_spec.names) != tuple(self.action_spec.names):
            raise ValueError("planner and direct Actor action spaces differ")
        output = self.fallback_prior.propose(
            observation, reference, horizon, action_spec
        )
        raw = self.encoder.encode(
            observation,
            reference,
            previous_action=self.previous_action,
            safety_override=self.safety_override,
        )
        metadata = dict(output.metadata)
        metadata.update({
            "type": "paper_direct_control",
            "checkpoint": self.checkpoint_path,
            "actor_action_semantics": "physical_low_level_control",
            "actor_support_ood_score": float(
                self.normalizer.ood_score(raw)
            ),
            "hypothetical_scan_assumption": "latest_observed_scan",
        })
        return type(output)(
            output.mean, output.covariance, metadata, output.proposals
        )

    def normalized_to_physical(self, action):
        values = np.asarray(action, dtype=np.float64)
        expected_tail = (self.action_spec.dimension,)
        if values.shape[-1:] != expected_tail or not np.isfinite(values).all():
            raise ValueError("normalized direct action has an invalid shape")
        return np.clip(
            self.action_center + self.action_half_range * values,
            self.action_spec.lower,
            self.action_spec.upper,
        )

    def support_ood_scores(self, raw_observations):
        """Return per-observation standardized support distance.

        The scalar ``RunningNormalizer.ood_score`` predates batched Actor
        rollouts and reduces over every dimension.  Gate 3 needs one score per
        horizon state, so this method preserves the same maximum-z semantics
        while reducing only over the feature axis.
        """

        values = np.asarray(raw_observations, dtype=np.float32)
        if (
            values.ndim < 2
            or values.shape[-1] != self.normalizer.dimension
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Actor support observations require [...,feature_dim]"
            )
        return np.max(
            np.abs(self.normalizer.normalize(values)), axis=-1
        ).astype(np.float64, copy=False)

    def _hypothetical_observation(
        self, state, observation, state_spec, time_offset
    ):
        x_index = state_spec.index("x")
        y_index = state_spec.index("y")
        theta_index = state_spec.index("theta")
        v = (
            observation.twist.v
            if "v" not in state_spec.names
            else state[state_spec.index("v")]
        )
        omega = (
            observation.twist.omega
            if "omega" not in state_spec.names
            else state[state_spec.index("omega")]
        )
        return RobotObservation(
            timestamp=float(observation.timestamp) + float(time_offset),
            pose=Pose2D(
                float(state[x_index]),
                float(state[y_index]),
                float(state[theta_index]),
            ),
            twist=Twist2D(float(v), float(omega)),
            scan=observation.scan,
            local_obstacles=observation.local_obstacles,
            auxiliary=dict(observation.auxiliary),
        )

    def _encoded_batch(
        self,
        states,
        previous_controls,
        observation,
        reference,
        state_spec,
        time_offset,
    ):
        states = np.asarray(states, dtype=np.float64)
        previous_controls = np.asarray(previous_controls, dtype=np.float64)
        if (
            states.ndim != 2
            or states.shape[1] != state_spec.dimension
            or states.shape[0] <= 0
            or not np.isfinite(states).all()
        ):
            raise ValueError("policy rollout states must be finite [B,state_dim]")
        if (
            previous_controls.shape
            != (states.shape[0], self.action_spec.dimension)
            or not np.isfinite(previous_controls).all()
        ):
            raise ValueError(
                "previous controls must be finite [B,action_dim]"
            )
        offsets = np.asarray(time_offset, dtype=np.float64)
        if offsets.ndim == 0:
            offsets = np.full(states.shape[0], float(offsets))
        offsets = offsets.reshape(-1)
        if offsets.shape != (states.shape[0],) or not np.isfinite(offsets).all():
            raise ValueError("policy rollout time offsets are invalid")
        # Every hypothetical state in this batch is conditioned on the same
        # latest real LaserScan. Sectorization is therefore invariant across
        # candidates and must be computed once, rather than once per state.
        # This preserves the paper-level observation semantics while avoiding
        # an O(batch_size * scan_rays) Python hot path.
        scan_encoding = self.encoder._scan_features(observation.scan)
        x_index = state_spec.index("x")
        y_index = state_spec.index("y")
        theta_index = state_spec.index("theta")
        poses = states[:, (x_index, y_index, theta_index)]
        twists = np.empty((states.shape[0], 2), dtype=np.float64)
        twists[:, 0] = (
            states[:, state_spec.index("v")]
            if "v" in state_spec.names
            else float(observation.twist.v)
        )
        twists[:, 1] = (
            states[:, state_spec.index("omega")]
            if "omega" in state_spec.names
            else float(observation.twist.omega)
        )
        targets = np.empty((states.shape[0], 2), dtype=np.float64)
        path_contexts = (
            np.empty((states.shape[0], 6), dtype=np.float64)
            if self.encoder.config.include_path_context
            else None
        )
        path_previews = (
            np.empty(
                (
                    states.shape[0],
                    2 * len(self.encoder.config.path_preview_distances),
                ),
                dtype=np.float64,
            )
            if self.encoder.config.include_path_preview
            else None
        )
        preview_target = getattr(reference, "preview_target_at", None)
        progress_floor = getattr(reference, "progress", None)
        for index, (state, offset) in enumerate(zip(states, offsets)):
            pose = state[[x_index, y_index, theta_index]]
            if callable(preview_target):
                target = preview_target(
                    float(observation.timestamp) + float(offset),
                    pose,
                    progress_floor=progress_floor,
                )
            else:
                target = reference.target_at(
                    float(observation.timestamp) + float(offset), pose
                )
            targets[index] = (target.pose.x, target.pose.y)
            if path_contexts is not None:
                path_contexts[index] = self.encoder.path_context(
                    reference,
                    pose,
                    target=target,
                    progress_floor=progress_floor,
                )
            if path_previews is not None:
                path_previews[index] = self.encoder.path_preview(
                    reference,
                    pose,
                    progress_floor=progress_floor,
                )
        raw = self.encoder.encode_kinematic_batch(
            poses,
            twists,
            targets,
            previous_controls,
            scan_encoding,
            safety_override=False,
            path_context_features=path_contexts,
            residual_context_features=(
                None
                if self.residual_context is None
                else self.residual_context.features(states, previous_controls)
            ),
            path_preview_features=path_previews,
        )
        normalized = self.normalizer.normalize(raw).astype(
            np.float32, copy=False
        )
        return raw, normalized

    def action_distribution(
        self,
        states,
        previous_controls,
        observation,
        reference,
        state_spec,
        time_offset,
    ):
        """Return the exact pre-tanh Actor Gaussian and physical mean."""

        raw, normalized = self._encoded_batch(
            states,
            previous_controls,
            observation,
            reference,
            state_spec,
            time_offset,
        )
        pre_tanh_mean, log_std = self.agent.policy_gaussian_parameters_batch(
            normalized
        )
        normalized_mean = np.tanh(pre_tanh_mean)
        correction_authority = None
        context_features = None
        if self.residual_context is not None:
            dimension = int(self.residual_context.dimension)
            context_features = np.asarray(
                raw[:, -dimension:], dtype=np.float64
            )
        if self.residual_correction_authority is not None:
            base_pre_tanh, base_log_std = (
                self.agent.correction_base_gaussian_parameters_batch(
                    normalized
                )
            )
            base_mean = np.tanh(base_pre_tanh)
            correction_authority = self.residual_correction_authority.evaluate(
                context_features
            )
            normalized_mean = (
                base_mean
                + correction_authority[:, None]
                * (normalized_mean - base_mean)
            )
            bounded_mean = np.clip(normalized_mean, -1.0 + 1e-6, 1.0 - 1e-6)
            base_post_tanh_std = np.maximum(
                (1.0 - base_mean ** 2) * np.exp(base_log_std), 1e-6
            )
            pre_tanh_mean = np.arctanh(bounded_mean)
            log_std = np.log(
                base_post_tanh_std
                / np.maximum(1.0 - bounded_mean ** 2, 1e-6)
            )
        physical_mean = self.normalized_to_physical(normalized_mean)
        # Delta-method variance is used only to initialize MPPI covariance.
        normalized_std = (
            (1.0 - normalized_mean ** 2) * np.exp(log_std)
        )
        physical_std = np.maximum(
            np.abs(self.action_half_range) * normalized_std, 1e-6
        )
        result = {
            "raw_observation": raw,
            "normalized_observation": normalized,
            "pre_tanh_mean": pre_tanh_mean,
            "log_std": log_std,
            "physical_mean": physical_mean,
            "physical_std": physical_std,
        }
        if context_features is not None:
            result["residual_context_features"] = context_features
        if correction_authority is not None:
            result["residual_correction_authority"] = correction_authority
        return result

    def sample_actions(
        self,
        states,
        previous_controls,
        observation,
        reference,
        state_spec,
        time_offset,
        rng,
    ):
        distribution = self.action_distribution(
            states,
            previous_controls,
            observation,
            reference,
            state_spec,
            time_offset,
        )
        return self.sample_from_distribution(distribution, rng), distribution

    def sample_from_distribution(self, distribution, rng):
        """Draw physical commands from a previously evaluated Actor batch."""

        noise = rng.normal(size=distribution["pre_tanh_mean"].shape)
        normalized = np.tanh(
            distribution["pre_tanh_mean"]
            + np.exp(distribution["log_std"]) * noise
        )
        return self.normalized_to_physical(normalized)

    def terminal_value(
        self,
        terminal_states,
        terminal_controls,
        observation,
        reference,
        state_spec,
        horizon_dt,
        critic_source="target",
    ):
        """Evaluate physical terminal states with Actor actions and twin Q."""

        details = self.terminal_value_details(
            terminal_states,
            terminal_controls,
            observation,
            reference,
            state_spec,
            horizon_dt,
            critic_source=critic_source,
        )
        return details["returns"], details["diagnostics"]

    def terminal_value_details(
        self,
        terminal_states,
        terminal_controls,
        observation,
        reference,
        state_spec,
        horizon_dt,
        critic_source="target",
    ):
        """Return per-candidate critic/support signals for conservative use."""

        states = np.asarray(terminal_states, dtype=np.float64)
        controls = np.asarray(terminal_controls, dtype=np.float64)
        raw_observation, normalized_observation = self._encoded_batch(
            states,
            controls,
            observation,
            reference,
            state_spec,
            horizon_dt,
        )
        normalized_actions = self.agent.select_action_batch(
            normalized_observation, deterministic=True
        )
        values = self.agent.expected_twin_q(
            normalized_observation,
            normalized_actions,
            critic_source=critic_source,
        )
        conservative = np.asarray(values["minimum"], dtype=np.float64)
        disagreement = np.asarray(
            values["disagreement"], dtype=np.float64
        )
        ood_scores = self.support_ood_scores(raw_observation)
        if not all(
            np.isfinite(item).all()
            for item in (conservative, disagreement, ood_scores)
        ):
            raise FloatingPointError("terminal critic returned NaN or Inf")
        diagnostics = {
            "terminal_q_mean": float(np.mean(conservative)),
            "terminal_q_min": float(np.min(conservative)),
            "terminal_q_max": float(np.max(conservative)),
            "terminal_q_disagreement_mean": float(
                np.mean(disagreement)
            ),
            "terminal_q_disagreement_max": float(np.max(disagreement)),
            "terminal_critic_ood_mean": float(np.mean(ood_scores)),
            "terminal_critic_ood_max": float(np.max(ood_scores)),
            "terminal_critic_source": str(values["critic_source"]),
            "terminal_action_semantics": "normalized_physical_control",
            "terminal_scan_assumption": "latest_observed_scan",
        }
        return {
            "returns": conservative,
            "critic_disagreement": disagreement,
            "critic_ood_scores": ood_scores,
            "diagnostics": diagnostics,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        action_spec,
        device="cpu",
        fallback_prior=None,
        residual_context=None,
        residual_correction_authority=None,
    ):
        payload = load_sac_checkpoint(checkpoint_path, map_location=device)
        action_mode = str(
            payload.get(
                "action_mode",
                dict(payload.get("resolved_config", {}).get("rl", {}))
                .get("training", {})
                .get("action_mode", "mppi_prior"),
            )
        )
        if action_mode != "direct_control":
            raise ValueError(
                "paper-faithful policy requires a direct_control checkpoint"
            )
        saved_action = payload["action_spec"]
        if tuple(saved_action["names"]) != tuple(action_spec.names):
            raise ValueError("checkpoint physical action names do not match")
        if (
            not np.allclose(saved_action["lower"], action_spec.lower)
            or not np.allclose(saved_action["upper"], action_spec.upper)
        ):
            raise ValueError("checkpoint physical action bounds do not match")
        encoder_config = ObservationEncoderConfig.from_mapping(
            payload["encoder_config"]
        )
        encoder = ObservationEncoder(encoder_config, action_spec)
        agent_state = payload["agent"]
        agent = SACAgent(
            agent_state["observation_dim"],
            agent_state["action_dim"],
            SACConfig.from_mapping(agent_state["config"]),
            device=device,
        )
        if encoder.dimension != agent.observation_dim:
            raise ValueError("checkpoint encoder and Actor dimensions differ")
        if agent.action_dim != action_spec.dimension:
            raise ValueError(
                "checkpoint Actor action is not low-level physical control"
            )
        agent.load_state_dict(agent_state, load_optimizers=False)
        normalizer = RunningNormalizer.from_state_dict(payload["normalizer"])
        return cls(
            agent,
            encoder,
            normalizer,
            action_spec,
            fallback_prior=fallback_prior,
            checkpoint_path=checkpoint_path,
            residual_context=residual_context,
            residual_correction_authority=residual_correction_authority,
        )
