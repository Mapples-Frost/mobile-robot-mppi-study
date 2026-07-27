"""RL-Driven MPPI with auditable hybrid candidate sources.

This controller is opt-in.  The legacy :class:`MppiController` remains the
default, so existing MuJoCo, ROS, memory, perception and safety behavior is
unchanged unless ``planner.optimizer`` is explicitly set to ``rl_driven``.
"""

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from mobile_robot_mppi.planning.mppi import MppiController, integrate_batch
from mobile_robot_mppi.rl.reliability import (
    ConservativeTerminalReliability,
    HybridSamplingReliability,
    ProposalAdvantageGate,
    SourceRelativeCompetence,
)


@dataclass(frozen=True)
class RLDrivenMppiConfig:
    iterations: int = 2
    rl_fraction: float = 0.30
    shifted_fraction: float = 0.40
    base_fraction: float = 0.30
    elite_fraction: float = 0.20
    covariance_smoothing: float = 0.50
    covariance_min_scale: float = 0.25
    covariance_max_scale: float = 2.00
    terminal_value_weight: float = 0.0
    terminal_critic_source: str = "target"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        values = dict(values or {})
        return cls(
            iterations=int(values.get("iterations", 2)),
            rl_fraction=float(values.get("rl_fraction", 0.30)),
            shifted_fraction=float(values.get("shifted_fraction", 0.40)),
            base_fraction=float(values.get("base_fraction", 0.30)),
            elite_fraction=float(values.get("elite_fraction", 0.20)),
            covariance_smoothing=float(
                values.get("covariance_smoothing", 0.50)
            ),
            covariance_min_scale=float(
                values.get("covariance_min_scale", 0.25)
            ),
            covariance_max_scale=float(
                values.get("covariance_max_scale", 2.00)
            ),
            terminal_value_weight=float(
                values.get("terminal_value_weight", 0.0)
            ),
            terminal_critic_source=str(
                values.get("terminal_critic_source", "target")
            ),
        )

    def validate(self, total_samples):
        if self.iterations <= 0 or self.iterations > int(total_samples):
            raise ValueError(
                "RL-Driven MPPI iterations must lie in [1, num_samples]"
            )
        fractions = np.asarray(
            (self.rl_fraction, self.shifted_fraction, self.base_fraction),
            dtype=np.float64,
        )
        if (
            not np.isfinite(fractions).all()
            or np.any(fractions < 0.0)
            or not np.isclose(float(np.sum(fractions)), 1.0, atol=1e-9)
        ):
            raise ValueError(
                "proposal fractions must be non-negative and sum to one"
            )
        if self.shifted_fraction <= 0.0 or self.base_fraction <= 0.0:
            raise ValueError("shifted and base proposal fractions must be positive")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must lie in (0, 1]")
        if not 0.0 <= self.covariance_smoothing < 1.0:
            raise ValueError("covariance_smoothing must lie in [0, 1)")
        if (
            not np.isfinite(self.covariance_min_scale)
            or not np.isfinite(self.covariance_max_scale)
            or self.covariance_min_scale <= 0.0
            or self.covariance_max_scale < self.covariance_min_scale
        ):
            raise ValueError("invalid covariance scale limits")
        if not np.isfinite(self.terminal_value_weight) or self.terminal_value_weight < 0.0:
            raise ValueError("terminal_value_weight must be finite and non-negative")
        if self.terminal_critic_source not in ("online", "target"):
            raise ValueError("terminal_critic_source must be online or target")


class RLDrivenMppiController(MppiController):
    """Pool RL, shifted-solution, and conventional MPPI proposals.

    ``num_samples`` is a total rollout budget, not a per-iteration budget.  It
    is partitioned across refinement iterations so comparisons with standard
    MPPI retain the same number of model rollouts per control decision.
    """

    SOURCE_NAMES = ("rl", "shifted", "base")

    def __init__(self, *args, rl_driven_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rl_driven_config = (
            rl_driven_config
            if isinstance(rl_driven_config, RLDrivenMppiConfig)
            else RLDrivenMppiConfig.from_mapping(rl_driven_config or {})
        )
        self.rl_driven_config.validate(self.config.num_samples)
        if self.config.importance_sampling_correction:
            raise ValueError(
                "mixture proposals do not support the single-Gaussian "
                "importance_sampling_correction"
            )
        if not callable(getattr(self.sampling_prior, "propose", None)):
            raise ValueError("RL-Driven MPPI requires a proposal prior")

    @staticmethod
    def _allocate(total, fractions):
        raw = np.asarray(fractions, dtype=np.float64) * int(total)
        counts = np.floor(raw).astype(np.int64)
        remainder = int(total) - int(np.sum(counts))
        order = np.argsort(-(raw - counts), kind="stable")
        for index in order[:remainder]:
            counts[index] += 1
        if int(total) >= len(fractions):
            for index in np.flatnonzero(
                (counts == 0) & (np.asarray(fractions) > 0.0)
            ):
                donor = int(np.argmax(counts))
                if counts[donor] <= 1:
                    break
                counts[donor] -= 1
                counts[index] += 1
        if int(np.sum(counts)) != int(total) or np.any(counts < 0):
            raise RuntimeError("candidate allocation failed")
        return counts

    def _proposal_means(self, prior):
        proposals = {item.label: item for item in prior.proposals}
        if "rl" not in proposals or "base" not in proposals:
            raise ValueError(
                "RL-Driven MPPI requires explicit 'rl' and 'base' proposals"
            )
        expected = (self.config.horizon, self.action_spec.dimension)
        means = {}
        covariances = {}
        for name in ("rl", "base"):
            mean = np.asarray(proposals[name].mean, dtype=np.float64)
            if mean.shape != expected or not np.isfinite(mean).all():
                raise ValueError("%s proposal mean is invalid" % name)
            means[name] = np.clip(
                mean, self.action_spec.lower, self.action_spec.upper
            )
            covariances[name] = proposals[name].covariance
        if self.previous_sequence is None:
            means["shifted"] = means["base"].copy()
        else:
            shifted = np.empty_like(self.previous_sequence)
            shifted[:-1] = self.previous_sequence[1:]
            shifted[-1] = self.previous_sequence[-1]
            means["shifted"] = np.clip(
                shifted, self.action_spec.lower, self.action_spec.upper
            )
        covariances["shifted"] = covariances["base"]
        return means, covariances

    def _iteration_budgets(self):
        base = self.config.num_samples // self.rl_driven_config.iterations
        remainder = self.config.num_samples % self.rl_driven_config.iterations
        return [
            base + (1 if index < remainder else 0)
            for index in range(self.rl_driven_config.iterations)
        ]

    def _sample_mixture(
        self, means, variance, count, rng, source_variances=None
    ):
        cfg = self.rl_driven_config
        source_counts = self._allocate(
            count,
            (cfg.rl_fraction, cfg.shifted_fraction, cfg.base_fraction),
        )
        batches = []
        labels = []
        for name, source_count in zip(self.SOURCE_NAMES, source_counts):
            selected_variance = (
                variance
                if source_variances is None
                else source_variances[name]
            )
            standard_deviation = np.sqrt(selected_variance)
            noise = rng.normal(
                size=(int(source_count),) + means[name].shape
            ) * standard_deviation[None, :, :]
            values = means[name][None, :, :] + noise
            values = np.clip(
                values, self.action_spec.lower, self.action_spec.upper
            )
            if source_count > 0:
                values[0] = means[name]
            batches.append(values)
            labels.extend([name] * int(source_count))
        return np.concatenate(batches, axis=0), np.asarray(labels), source_counts

    def _terminal_constraints(self, state, target, controls):
        samples = np.asarray(controls, dtype=np.float64).copy()
        x_index, y_index = self.state_spec.position_indices
        dx = float(target.pose.x - state[x_index])
        dy = float(target.pose.y - state[y_index])
        distance = float(np.hypot(dx, dy))
        control_region_active = bool(
            self.config.terminal_control_radius is None
            or distance <= self.config.terminal_control_radius
        )
        heading_gate_active = bool(
            self.config.terminal_translation_heading_gate_rad is not None
            and target.phase in ("terminal_approach", "terminal")
            and control_region_active
            and "v_cmd" in self.action_spec.names
            and "theta" in self.state_spec.names
        )
        bearing_error = 0.0
        if "theta" in self.state_spec.names and distance > 1.0e-12:
            theta = float(state[self.state_spec.index("theta")])
            desired = float(np.arctan2(dy, dx))
            bearing_error = float(np.arctan2(
                np.sin(desired - theta), np.cos(desired - theta)
            ))
        translation_scale = 1.0
        if heading_gate_active:
            gate = float(self.config.terminal_translation_heading_gate_rad)
            gate_cosine = float(np.cos(gate))
            if abs(bearing_error) >= gate:
                translation_scale = 0.0
            else:
                translation_scale = max(
                    0.0,
                    (float(np.cos(bearing_error)) - gate_cosine)
                    / max(1.0 - gate_cosine, 1e-12),
                )
        speed_limit_active = bool(
            self.config.terminal_translation_speed_limit is not None
            and target.phase in ("terminal_approach", "terminal")
            and control_region_active
            and "v_cmd" in self.action_spec.names
        )
        v_index = None
        if speed_limit_active or heading_gate_active:
            v_index = self.action_spec.index("v_cmd")
        if speed_limit_active:
            samples[..., v_index] = np.minimum(
                samples[..., v_index],
                self.config.terminal_translation_speed_limit,
            )
        if heading_gate_active:
            samples[:, 0, v_index] *= translation_scale
        return samples, {
            "terminal_speed_limit_active": speed_limit_active,
            "terminal_heading_gate_active": heading_gate_active,
            "terminal_bearing_error": bearing_error,
            "target_bearing_error": bearing_error,
            "terminal_translation_scale": translation_scale,
            "terminal_control_distance": distance,
            "terminal_control_region_active": control_region_active,
            "v_index": v_index,
        }

    def _finalize_terminal_action(self, sequence, constraints):
        """Apply one shared terminal action law after actuator slew clipping."""

        sequence = np.asarray(sequence, dtype=np.float64).copy()
        action = self.action_spec.clip(
            sequence[0], self.previous_action, self.config.dt
        )
        v_index = constraints["v_index"]
        if constraints["terminal_speed_limit_active"]:
            action[v_index] = min(
                action[v_index],
                self.config.terminal_translation_speed_limit,
            )
        if constraints["terminal_heading_gate_active"]:
            action[v_index] *= constraints["terminal_translation_scale"]
        alignment_active = bool(
            constraints["terminal_heading_gate_active"]
            and self.config.terminal_alignment_yaw_gain is not None
            and "omega_cmd" in self.action_spec.names
        )
        alignment_omega = 0.0
        if alignment_active:
            omega_index = self.action_spec.index("omega_cmd")
            alignment_omega = (
                float(self.config.terminal_alignment_yaw_gain)
                * constraints["terminal_bearing_error"]
            )
            action[omega_index] = np.clip(
                alignment_omega,
                self.action_spec.lower[omega_index],
                self.action_spec.upper[omega_index],
            )
        sequence[0] = action
        return action, sequence, {
            "terminal_translation_speed_limit": (
                0.0
                if self.config.terminal_translation_speed_limit is None
                else float(self.config.terminal_translation_speed_limit)
            ),
            "terminal_translation_heading_gate_rad": (
                0.0
                if self.config.terminal_translation_heading_gate_rad is None
                else float(self.config.terminal_translation_heading_gate_rad)
            ),
            "terminal_speed_limit_active": bool(
                constraints["terminal_speed_limit_active"]
            ),
            "terminal_heading_gate_active": bool(
                constraints["terminal_heading_gate_active"]
            ),
            "terminal_bearing_error": float(
                constraints["terminal_bearing_error"]
            ),
            "target_bearing_error": float(
                constraints["target_bearing_error"]
            ),
            "terminal_translation_scale": float(
                constraints["terminal_translation_scale"]
            ),
            "terminal_alignment_active": alignment_active,
            "terminal_alignment_yaw_gain": (
                0.0
                if self.config.terminal_alignment_yaw_gain is None
                else float(self.config.terminal_alignment_yaw_gain)
            ),
            "terminal_control_radius": (
                0.0
                if self.config.terminal_control_radius is None
                else float(self.config.terminal_control_radius)
            ),
            "terminal_control_distance": float(
                constraints["terminal_control_distance"]
            ),
            "terminal_control_region_active": bool(
                constraints["terminal_control_region_active"]
            ),
            "terminal_alignment_omega": float(alignment_omega),
        }

    def _terminal_value_cost(
        self, trajectories, controls, observation, target
    ):
        weight = self.rl_driven_config.terminal_value_weight
        if weight <= 0.0:
            return np.zeros(controls.shape[0], dtype=np.float64), {
                "terminal_value_enabled": False,
                "terminal_value_weight": 0.0,
                "terminal_q_mean": 0.0,
                "terminal_q_disagreement_mean": 0.0,
            }
        evaluator = getattr(self.sampling_prior, "terminal_value", None)
        if not callable(evaluator):
            raise ValueError(
                "terminal value weight requires a prior with terminal_value"
            )
        values, diagnostics = evaluator(
            trajectories[:, -1, :],
            controls[:, -1, :],
            observation,
            target,
            self.state_spec,
            self.config.horizon * self.config.dt,
            critic_source=self.rl_driven_config.terminal_critic_source,
        )
        # SAC learns return (larger is better); MPPI minimizes cost.
        contribution = -float(weight) * np.asarray(values, dtype=np.float64)
        diagnostics = dict(diagnostics)
        diagnostics.update({
            "terminal_value_enabled": True,
            "terminal_value_weight": float(weight),
            "terminal_value_cost_mean": float(np.mean(contribution)),
            "terminal_value_cost_std": float(np.std(contribution)),
        })
        return contribution, diagnostics

    def _solve_plan(
        self, state, prior, target, obstacles, rng, observation=None,
        reference=None,
    ):
        if observation is None:
            raise ValueError("RL-Driven MPPI requires the current observation")
        means, proposal_covariances = self._proposal_means(prior)
        cfg = self.rl_driven_config
        sigma = np.asarray(self.config.noise_sigma, dtype=np.float64)
        base_variance = np.broadcast_to(
            sigma[None, :] ** 2,
            (self.config.horizon, self.action_spec.dimension),
        ).copy()
        variance = base_variance.copy()
        minimum_variance = base_variance * cfg.covariance_min_scale ** 2
        maximum_variance = base_variance * cfg.covariance_max_scale ** 2
        source_variances = {}
        for name in self.SOURCE_NAMES:
            covariance = proposal_covariances[name]
            if covariance is None:
                source_variances[name] = base_variance.copy()
            else:
                matrix = self._sampling_covariance(
                    type(prior)(means[name], covariance, {})
                )
                source_variances[name] = np.broadcast_to(
                    np.diag(matrix)[None, :], base_variance.shape
                ).copy()
        budgets = self._iteration_budgets()

        all_costs = []
        all_samples = []
        total_counts = {name: 0 for name in self.SOURCE_NAMES}
        total_elites = {name: 0 for name in self.SOURCE_NAMES}
        final_weights = None
        final_sequence = means["shifted"].copy()
        terminal_diagnostics = {}
        constraints = None
        for iteration, budget in enumerate(budgets):
            samples, labels, counts = self._sample_mixture(
                means,
                variance,
                budget,
                rng,
                source_variances if iteration == 0 else None,
            )
            samples, constraints = self._terminal_constraints(
                state, target, samples
            )
            for name, count in zip(self.SOURCE_NAMES, counts):
                total_counts[name] += int(count)
            trajectories = self.rollout(state, samples)
            costs = self._cost(
                trajectories,
                samples,
                target,
                obstacles,
                reference=reference,
            )
            terminal_cost, terminal_diagnostics = self._terminal_value_cost(
                trajectories, samples, observation, target
            )
            costs = np.asarray(costs, dtype=np.float64) + terminal_cost
            if not np.isfinite(costs).all():
                raise FloatingPointError("RL-Driven MPPI cost is not finite")

            elite_count = max(
                2, min(budget, int(np.ceil(cfg.elite_fraction * budget)))
            )
            elite_indices = np.argsort(costs, kind="stable")[:elite_count]
            elite_costs = costs[elite_indices]
            beta = float(np.min(elite_costs))
            exponent = np.clip(
                -(elite_costs - beta) / self.config.temperature,
                -700.0,
                0.0,
            )
            weights = np.exp(exponent)
            weights /= max(float(np.sum(weights)), 1e-12)
            elites = samples[elite_indices]
            final_sequence = np.sum(
                weights[:, None, None] * elites, axis=0
            )
            centered = elites - final_sequence[None, :, :]
            estimate = np.sum(
                weights[:, None, None] * centered ** 2, axis=0
            )
            variance = np.clip(
                cfg.covariance_smoothing * variance
                + (1.0 - cfg.covariance_smoothing) * estimate,
                minimum_variance,
                maximum_variance,
            )
            # The shifted-solution share becomes the current elite solution on
            # the next refinement iteration. RL and conventional anchors stay
            # fixed, preventing policy or optimizer collapse from deleting the
            # other candidate families.
            means["shifted"] = final_sequence.copy()
            for name in self.SOURCE_NAMES:
                total_elites[name] += int(np.sum(labels[elite_indices] == name))
            all_costs.append(costs)
            all_samples.append(samples)
            final_weights = weights

        sequence = np.clip(
            final_sequence, self.action_spec.lower, self.action_spec.upper
        )
        sequence, constraints = self._terminal_constraints(
            state, target, sequence[None, :, :]
        )
        sequence = sequence[0]
        action, sequence, terminal_action_diagnostics = (
            self._finalize_terminal_action(sequence, constraints)
        )
        updated_trajectory = self.rollout(state, sequence)[0]

        costs = np.concatenate(all_costs)
        samples = np.concatenate(all_samples)
        effective_sample_size = float(
            1.0 / np.sum(np.asarray(final_weights) ** 2)
        )
        diagnostics = {
            "optimizer": "rl_driven",
            "rl_driven_iterations": int(cfg.iterations),
            "rl_driven_total_rollouts": int(np.sum(budgets)),
            "rl_driven_iteration_budgets": list(budgets),
            "rl_source_samples": int(total_counts["rl"]),
            "shifted_source_samples": int(total_counts["shifted"]),
            "base_source_samples": int(total_counts["base"]),
            "rl_elite_count": int(total_elites["rl"]),
            "shifted_elite_count": int(total_elites["shifted"]),
            "base_elite_count": int(total_elites["base"]),
            "rl_elite_fraction": float(
                total_elites["rl"] / max(total_counts["rl"], 1)
            ),
            "covariance_scale_mean": float(
                np.mean(np.sqrt(variance / base_variance))
            ),
            "covariance_scale_min": float(
                np.min(np.sqrt(variance / base_variance))
            ),
            "covariance_scale_max": float(
                np.max(np.sqrt(variance / base_variance))
            ),
            "cost_min": float(np.min(costs)),
            "cost_mean": float(np.mean(costs)),
            "cost_std": float(np.std(costs)),
            "cost_q10": float(np.quantile(costs, 0.10)),
            "cost_q50": float(np.quantile(costs, 0.50)),
            "cost_q90": float(np.quantile(costs, 0.90)),
            "effective_sample_size": effective_sample_size,
            "effective_sample_fraction": float(
                effective_sample_size / len(final_weights)
            ),
            "importance_sampling_correction": False,
            "importance_cost_mean": 0.0,
            "weighted_perturbation_norm": float(
                np.linalg.norm(sequence - prior.mean)
            ),
            "sample_saturation_fraction": float(np.mean(
                (samples <= self.action_spec.lower[None, None, :])
                | (samples >= self.action_spec.upper[None, None, :])
            )),
            "reference_id": target.reference_id,
            "target_x": float(target.pose.x),
            "target_y": float(target.pose.y),
            "target_theta": float(target.pose.theta),
            "target_is_terminal": bool(target.is_terminal),
            "target_phase": str(target.phase),
            "prior": dict(prior.metadata),
            **terminal_action_diagnostics,
        }
        diagnostics.update(terminal_diagnostics)
        residual = getattr(self.dynamics, "residual", None)
        confidence = getattr(residual, "confidence", None)
        if callable(confidence):
            support_controls = self._prediction_controls(
                sequence[None, :, :]
            )[0]
            support = np.asarray(
                confidence(updated_trajectory[:-1], support_controls),
                dtype=np.float64,
            ).reshape(-1)
            diagnostics.update({
                "residual_support_confidence_mean": float(np.mean(support)),
                "residual_support_confidence_min": float(np.min(support)),
                "residual_support_reduced_fraction": float(
                    np.mean(support < 1.0 - 1e-12)
                ),
                "residual_support_disabled_fraction": float(
                    np.mean(support <= 1e-12)
                ),
            })
        reliability = getattr(residual, "diagnostics", None)
        if callable(reliability):
            diagnostics.update(reliability())
        return action, sequence, updated_trajectory, diagnostics


@dataclass(frozen=True)
class PaperRLDrivenMppiConfig:
    """Algorithm-level settings for the paper-faithful Gate 1 baseline."""

    iterations: int = 2
    guided_fraction: float = 0.30
    elite_fraction: float = 0.20
    covariance_smoothing: float = 0.50
    covariance_min_scale: float = 0.25
    covariance_max_scale: float = 2.00
    terminal_value_weight: float = 1.0
    terminal_critic_source: str = "target"
    reliability: Any = None
    conservative_terminal: Any = None
    terminal_guidance_radius: float = 0.0
    terminal_guided_fraction_floor: float = 0.0
    completion_handover_full_fallback_distance: float = 0.0
    completion_handover_full_rl_distance: float = 0.0
    counterfactual_proposal_gate_enabled: bool = False
    counterfactual_progress_soft_m: float = 0.0
    counterfactual_progress_hard_m: float = -0.15
    counterfactual_cross_track_weight: float = 0.50
    proposal_advantage_gate: Any = None
    standard_fallback_on_advantage_veto: bool = False
    same_cycle_guided_cost_filter: bool = False
    same_cycle_guided_relative_margin: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        values = dict(values or {})
        return cls(**{
            name: values.get(name, field.default)
            for name, field in cls.__dataclass_fields__.items()
        })

    def validate(self, samples):
        values = np.asarray((
            self.guided_fraction,
            self.elite_fraction,
            self.covariance_smoothing,
            self.covariance_min_scale,
            self.covariance_max_scale,
            self.terminal_value_weight,
            self.terminal_guidance_radius,
            self.terminal_guided_fraction_floor,
            self.completion_handover_full_fallback_distance,
            self.completion_handover_full_rl_distance,
            self.counterfactual_progress_soft_m,
            self.counterfactual_progress_hard_m,
            self.counterfactual_cross_track_weight,
            self.same_cycle_guided_relative_margin,
        ), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("paper RL-Driven MPPI settings must be finite")
        if self.iterations <= 0:
            raise ValueError("paper RL-Driven MPPI iterations must be positive")
        if not 0.0 <= self.guided_fraction < 1.0:
            raise ValueError("guided_fraction must lie in [0, 1)")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must lie in (0, 1]")
        if not 0.0 <= self.covariance_smoothing < 1.0:
            raise ValueError("covariance_smoothing must lie in [0, 1)")
        if (
            self.covariance_min_scale <= 0.0
            or self.covariance_max_scale < self.covariance_min_scale
        ):
            raise ValueError("paper covariance scale limits are invalid")
        if self.terminal_value_weight < 0.0:
            raise ValueError("terminal_value_weight must be non-negative")
        if self.terminal_guidance_radius < 0.0:
            raise ValueError(
                "terminal_guidance_radius must be non-negative"
            )
        if not 0.0 <= self.terminal_guided_fraction_floor < 1.0:
            raise ValueError(
                "terminal_guided_fraction_floor must lie in [0, 1)"
            )
        if (
            self.terminal_guided_fraction_floor > 0.0
            and self.terminal_guidance_radius <= 0.0
        ):
            raise ValueError(
                "a positive terminal guidance floor requires a positive radius"
            )
        fallback = self.completion_handover_full_fallback_distance
        full_rl = self.completion_handover_full_rl_distance
        if (fallback == 0.0) != (full_rl == 0.0):
            raise ValueError(
                "completion handover distances must both be zero or positive"
            )
        if fallback < 0.0 or full_rl < 0.0:
            raise ValueError(
                "completion handover distances must be non-negative"
            )
        if full_rl > 0.0 and full_rl <= fallback:
            raise ValueError(
                "full-RL distance must exceed full-fallback distance"
            )
        if self.terminal_critic_source not in ("online", "target"):
            raise ValueError("terminal_critic_source must be online or target")
        if (
            self.counterfactual_progress_hard_m
            >= self.counterfactual_progress_soft_m
        ):
            raise ValueError(
                "counterfactual progress hard threshold must be below soft"
            )
        if self.counterfactual_cross_track_weight < 0.0:
            raise ValueError(
                "counterfactual cross-track weight must be non-negative"
            )
        if self.same_cycle_guided_relative_margin < 0.0:
            raise ValueError(
                "same-cycle guided relative margin must be non-negative"
            )
        HybridSamplingReliability(self.reliability or {})
        ConservativeTerminalReliability(self.conservative_terminal or {})
        proposal_advantage = ProposalAdvantageGate(
            self.proposal_advantage_gate or {}
        )
        if (
            proposal_advantage.config.enabled
            and not HybridSamplingReliability(
                self.reliability or {}
            ).config.enabled
        ):
            raise ValueError(
                "proposal advantage gate requires adaptive HSS"
            )
        if self.standard_fallback_on_advantage_veto and (
            not proposal_advantage.config.enabled
            or proposal_advantage.config.mode != "episode_latched_veto"
        ):
            raise ValueError(
                "standard fallback requires an active episode-latched "
                "proposal advantage veto"
            )
        guided = int(round(float(samples) * self.guided_fraction))
        if guided >= int(samples):
            raise ValueError("at least one current-Gaussian sample is required")
        terminal_guided = int(round(
            float(samples) * self.terminal_guided_fraction_floor
        ))
        if terminal_guided >= int(samples):
            raise ValueError(
                "terminal guidance floor must retain a Gaussian candidate"
            )


class PaperRLDrivenMppiController(RLDrivenMppiController):
    """Faithful low-level Actor/critic integration used by Gate 1.

    The Actor is rolled out autoregressively through the controller's declared
    prediction dynamics. Its mean and stochastic spread initialize MPPI.
    Stochastic Actor sequences are generated exactly once per control decision
    and then reused in every refinement iteration, matching the persistent
    guided set in RL-Driven MPPI. The final command still comes from MPPI and
    remains subject to the unchanged external safety chain.
    """

    def __init__(
        self,
        *args,
        paper_rl_driven_config=None,
        reliability_residual=None,
        reliability_nominal_dynamics=None,
        maneuver_proposal_policy=None,
        **kwargs,
    ):
        MppiController.__init__(self, *args, **kwargs)
        if (reliability_residual is None) != (
            reliability_nominal_dynamics is None
        ):
            raise ValueError(
                "paper HSS sidecar requires both residual and nominal dynamics"
            )
        self.reliability_residual = reliability_residual
        self.reliability_nominal_dynamics = reliability_nominal_dynamics
        self.maneuver_proposal_policy = maneuver_proposal_policy
        if reliability_residual is not None:
            if (
                int(reliability_residual.state_dim)
                != self.state_spec.dimension
                or int(reliability_residual.control_dim)
                != self.action_spec.dimension
                or int(reliability_nominal_dynamics.state_dim)
                != self.state_spec.dimension
                or int(reliability_nominal_dynamics.control_dim)
                != self.action_spec.dimension
            ):
                raise ValueError(
                    "paper HSS sidecar dimensions do not match the controller"
                )
            missing = [
                name
                for name in ("disagreement", "support_confidence")
                if not callable(getattr(reliability_residual, name, None))
            ]
            if missing:
                raise ValueError(
                    "paper HSS sidecar is missing: %s" % ", ".join(missing)
                )
        self.paper_rl_driven_config = (
            paper_rl_driven_config
            if isinstance(
                paper_rl_driven_config, PaperRLDrivenMppiConfig
            )
            else PaperRLDrivenMppiConfig.from_mapping(
                paper_rl_driven_config or {}
            )
        )
        self.paper_rl_driven_config.validate(self.config.num_samples)
        # Reuse the strictly tested terminal-constraint helper without
        # inheriting the legacy mixture-proposal initialization contract.
        self.rl_driven_config = self.paper_rl_driven_config
        self.hybrid_sampling_reliability = HybridSamplingReliability(
            self.paper_rl_driven_config.reliability or {}
        )
        self.source_relative_competence = SourceRelativeCompetence(
            self.hybrid_sampling_reliability.config
        )
        self.proposal_advantage_gate = ProposalAdvantageGate(
            self.paper_rl_driven_config.proposal_advantage_gate or {}
        )
        self._standard_fallback_controller = None
        if self.paper_rl_driven_config.standard_fallback_on_advantage_veto:
            fallback_config = replace(
                self.config,
                num_samples=(
                    int(self.config.num_samples)
                    * int(self.paper_rl_driven_config.iterations)
                ),
                importance_sampling_correction=True,
            )
            self._standard_fallback_controller = MppiController(
                dynamics=self.dynamics,
                state_spec=self.state_spec,
                action_spec=self.action_spec,
                config=fallback_config,
                memory_cost=self.memory_cost,
            )
        self.conservative_terminal_reliability = (
            ConservativeTerminalReliability(
                self.paper_rl_driven_config.conservative_terminal or {}
            )
        )
        self._applied_guided_fraction = float(
            self.paper_rl_driven_config.guided_fraction
        )
        if self.maneuver_proposal_policy is not None and (
            int(self.maneuver_proposal_policy.horizon)
            != int(self.config.horizon)
            or int(self.maneuver_proposal_policy.action_dim)
            != int(self.action_spec.dimension)
        ):
            raise ValueError(
                "supervised maneuver Actor geometry disagrees with MPPI"
            )
        required = (
            "action_distribution",
            "sample_actions",
            "terminal_value",
        )
        missing = [
            name
            for name in required
            if not callable(getattr(self.sampling_prior, name, None))
        ]
        if missing:
            raise ValueError(
                "paper RL-Driven MPPI policy is missing: %s"
                % ", ".join(missing)
            )
        if self.config.importance_sampling_correction:
            raise ValueError(
                "paper hybrid sampling does not use the single-proposal "
                "importance correction"
            )

    def reset(self, seed=None):
        super().reset(seed)
        if self._standard_fallback_controller is not None:
            self._standard_fallback_controller.reset(seed)
        prediction_residual = getattr(self.dynamics, "residual", None)
        if (
            self.reliability_residual is not None
            and self.reliability_residual is not prediction_residual
        ):
            reset = getattr(self.reliability_residual, "reset", None)
            if callable(reset):
                reset()
        self._applied_guided_fraction = float(
            self.paper_rl_driven_config.guided_fraction
        )
        self.source_relative_competence.reset()
        self.proposal_advantage_gate.reset()

    def _solve_standard_advantage_fallback(
        self,
        state,
        prior,
        target,
        obstacles,
        rng,
        observation,
        reference,
    ):
        """Execute the exact standard-MPPI solver after an Actor veto.

        Paper MPPI normally splits the fixed rollout budget over refinement
        iterations and adds learned terminal value.  Merely setting guided
        authority to zero therefore does not reproduce RL/HSS-off.  This
        opt-in path uses the same total rollout budget in one standard MPPI
        update, restores its importance correction, and bypasses Actor/HSS
        inference after the episode latch.
        """

        fallback = self._standard_fallback_controller
        if fallback is None or self.proposal_advantage_gate.authority != 0.0:
            raise RuntimeError("standard advantage fallback is not active")
        fallback.previous_action = self.previous_action.copy()
        action, sequence, trajectory, diagnostics = MppiController._solve_plan(
            fallback,
            state,
            prior,
            target,
            obstacles,
            rng,
            observation,
            reference,
        )
        gate_diagnostics = self.proposal_advantage_gate.update(
            0.0,
            0.0,
            observed=False,
            authority_applied=0.0,
        )
        total_rollouts = int(fallback.config.num_samples)
        diagnostics.update({
            "optimizer": "paper_standard_mppi_fallback",
            "paper_faithful_gate1": True,
            "paper_standard_fallback_active": True,
            "paper_standard_fallback_contract": "standard_mppi_exact_v1",
            "paper_iterations": 1,
            "paper_candidates_per_iteration": total_rollouts,
            "paper_total_rollouts": total_rollouts,
            "paper_guided_unique_sequences": 0,
            "paper_guided_reuses": 0,
            "paper_guided_generation_calls": 0,
            "paper_gaussian_samples_per_iteration": total_rollouts,
            "paper_guided_elite_count": 0,
            "paper_gaussian_elite_count": 0,
            "paper_guided_opportunity_count": 0,
            "paper_gaussian_opportunity_count": total_rollouts,
            "paper_guided_cost_observed": False,
            "paper_gaussian_cost_observed": True,
            "paper_guided_cost_min": 0.0,
            "paper_guided_cost_mean": 0.0,
            "paper_guided_cost_p50": 0.0,
            "paper_guided_minus_gaussian_cost_min": 0.0,
            "paper_guided_minus_gaussian_cost_mean": 0.0,
            "paper_actor_mean_initialization": False,
            "paper_actor_covariance_initialization": False,
            "paper_actor_joint_batched": False,
            "paper_guided_set_persistent": False,
            "reliability_hss_enabled": True,
            "reliability_guided_fraction_applied": 0.0,
            "reliability_guided_fraction_next": 0.0,
            "reliability_guided_fraction_raw_applied": 0.0,
            "reliability_proposal_authority": 0.0,
            "reliability_proposal_fallback_fraction": 1.0,
            "terminal_value_enabled": False,
            **gate_diagnostics,
        })
        return action, sequence, trajectory, diagnostics

    def observe_completed_transition(
        self,
        previous_state,
        applied_control,
        current_state,
        residual_derivative=None,
    ):
        prediction_updated = super().observe_completed_transition(
            previous_state,
            applied_control,
            current_state,
            residual_derivative=residual_derivative,
        )
        if self.reliability_residual is None:
            return prediction_updated
        sidecar_updated = self._observe_residual_prediction_errors(
            previous_state,
            applied_control,
            current_state,
            residual=self.reliability_residual,
            nominal=self.reliability_nominal_dynamics,
        )
        return bool(prediction_updated or sidecar_updated)

    def _observe_residual_reliability(self, current_state):
        if self.reliability_residual is None:
            return super()._observe_residual_reliability(current_state)
        if (
            self._reliability_previous_state is None
            or self._reliability_pending_control is None
        ):
            return
        self.observe_completed_transition(
            self._reliability_previous_state,
            self._reliability_pending_control,
            current_state,
        )
        self._reliability_pending_control = None

    def _delayed_control(self, current, preceding):
        fraction = float(self.config.command_delay_s / self.config.dt)
        return fraction * preceding + (1.0 - fraction) * current

    def _completion_preserving_guidance(self, state, target):
        """Return the causal terminal floor and its current-state diagnostics."""

        cfg = self.paper_rl_driven_config
        position = np.asarray(state, dtype=np.float64)[
            list(self.state_spec.position_indices)
        ]
        target_xy = np.asarray(
            (target.pose.x, target.pose.y), dtype=np.float64
        )
        distance = float(np.linalg.norm(target_xy - position))
        enabled = bool(
            self.hybrid_sampling_reliability.config.enabled
            and cfg.terminal_guidance_radius > 0.0
            and cfg.terminal_guided_fraction_floor > 0.0
        )
        terminal_phase = str(
            getattr(target, "phase", "terminal")
        ) in ("terminal_approach", "terminal")
        active = bool(
            enabled
            and terminal_phase
            and distance <= cfg.terminal_guidance_radius
        )
        floor = (
            float(cfg.terminal_guided_fraction_floor) if active else 0.0
        )
        return floor, {
            "terminal_guidance_floor_enabled": enabled,
            "terminal_guidance_floor_active": active,
            "terminal_guidance_terminal_phase": terminal_phase,
            "terminal_guidance_distance": distance,
            "terminal_guidance_radius": float(
                cfg.terminal_guidance_radius
            ),
            "terminal_guided_fraction_floor": float(
                cfg.terminal_guided_fraction_floor
            ),
        }

    def _completion_handover(self, state, target):
        """Return a continuous Actor-to-MPPI authority near the goal.

        The handover is opt-in and only applies to terminal references.  It
        closes a semantic gap in reliability-adaptive HSS: reducing the
        number of Actor samples alone did not remove the Actor mean used to
        initialize the Gaussian proposal.  Authority therefore controls both
        the persistent Actor share and the proposal mean/covariance.
        """

        cfg = self.paper_rl_driven_config
        low = float(cfg.completion_handover_full_fallback_distance)
        high = float(cfg.completion_handover_full_rl_distance)
        position = np.asarray(state, dtype=np.float64)[
            list(self.state_spec.position_indices)
        ]
        target_xy = np.asarray(
            (target.pose.x, target.pose.y), dtype=np.float64
        )
        distance = float(np.linalg.norm(target_xy - position))
        terminal_phase = str(
            getattr(target, "phase", "terminal")
        ) in ("terminal_approach", "terminal")
        enabled = bool(low > 0.0 and high > low and terminal_phase)
        if not enabled or distance >= high:
            authority = 1.0
        elif distance <= low:
            authority = 0.0
        else:
            # Smoothstep avoids an abrupt proposal jump at either boundary.
            ratio = (distance - low) / (high - low)
            authority = float(ratio * ratio * (3.0 - 2.0 * ratio))
        return authority, {
            "completion_handover_enabled": enabled,
            "completion_handover_terminal_phase": terminal_phase,
            "completion_handover_distance": distance,
            "completion_handover_authority": authority,
            "completion_handover_full_fallback_distance": low,
            "completion_handover_full_rl_distance": high,
        }

    def _actor_mean_rollout(self, state, observation, reference):
        states = np.asarray(state, dtype=np.float64).reshape(1, -1)
        previous = self.previous_action.reshape(1, -1).copy()
        means = np.empty(
            (self.config.horizon, self.action_spec.dimension),
            dtype=np.float64,
        )
        variances = np.empty_like(means)
        rollout_states = np.empty(
            (self.config.horizon, self.state_spec.dimension),
            dtype=np.float64,
        )
        reliability_controls = np.empty_like(means)
        actor_ood_scores = np.zeros(
            self.config.horizon, dtype=np.float64
        )
        support_evaluator = getattr(
            self.sampling_prior, "support_ood_scores", None
        )
        residual_context_rows = []
        residual_authority_rows = []
        for step in range(self.config.horizon):
            rollout_states[step] = states[0]
            distribution = self.sampling_prior.action_distribution(
                states,
                previous,
                observation,
                reference,
                self.state_spec,
                step * self.config.dt,
            )
            if "residual_context_features" in distribution:
                residual_context_rows.append(np.asarray(
                    distribution["residual_context_features"][0],
                    dtype=np.float64,
                ))
            if "residual_correction_authority" in distribution:
                residual_authority_rows.append(float(
                    distribution["residual_correction_authority"][0]
                ))
            command = self.action_spec.clip(
                distribution["physical_mean"],
                previous=previous,
                dt=self.config.dt,
            )
            means[step] = command[0]
            variances[step] = distribution["physical_std"][0] ** 2
            if self.hybrid_sampling_reliability.config.enabled:
                if not callable(support_evaluator):
                    raise TypeError(
                        "reliability-calibrated HSS requires Actor "
                        "support_ood_scores"
                    )
                actor_ood_scores[step] = float(
                    support_evaluator(
                        distribution["raw_observation"][:1]
                    )[0]
                )
            applied = self._delayed_control(command, previous)
            reliability_controls[step] = applied[0]
            states = integrate_batch(
                self.dynamics,
                states,
                applied,
                self.config.dt,
                self.state_spec,
                self.config.integrator,
            )
            previous = command
        if not np.isfinite(means).all() or not np.isfinite(variances).all():
            raise FloatingPointError("Actor mean rollout produced NaN or Inf")
        return means, variances, {
            "states": rollout_states,
            "controls": reliability_controls,
            "actor_ood_scores": actor_ood_scores,
            "residual_context": np.asarray(
                residual_context_rows, dtype=np.float64
            ),
            "residual_authority": np.asarray(
                residual_authority_rows, dtype=np.float64
            ),
            "terminal_state": states[0].copy(),
        }

    def _guided_rollouts(
        self, state, observation, reference, count, rng
    ):
        if count <= 0:
            return np.empty(
                (0, self.config.horizon, self.action_spec.dimension),
                dtype=np.float64,
            )
        states = np.repeat(
            np.asarray(state, dtype=np.float64).reshape(1, -1),
            int(count),
            axis=0,
        )
        previous = np.repeat(
            self.previous_action.reshape(1, -1), int(count), axis=0
        )
        sequences = np.empty(
            (int(count), self.config.horizon, self.action_spec.dimension),
            dtype=np.float64,
        )
        for step in range(self.config.horizon):
            command, _ = self.sampling_prior.sample_actions(
                states,
                previous,
                observation,
                reference,
                self.state_spec,
                step * self.config.dt,
                rng,
            )
            command = self.action_spec.clip(
                command, previous=previous, dt=self.config.dt
            )
            sequences[:, step, :] = command
            applied = self._delayed_control(command, previous)
            states = integrate_batch(
                self.dynamics,
                states,
                applied,
                self.config.dt,
                self.state_spec,
                self.config.integrator,
            )
            previous = command
        if not np.isfinite(sequences).all():
            raise FloatingPointError("guided Actor rollout produced NaN or Inf")
        return sequences

    def _supervised_maneuver_rollouts(
        self, state, observation, reference
    ):
        policy = self.maneuver_proposal_policy
        if policy is None:
            return np.empty(
                (0, self.config.horizon, self.action_spec.dimension),
                dtype=np.float64,
            )
        encoder = getattr(self.sampling_prior, "_encoded_batch", None)
        if not callable(encoder):
            raise ValueError(
                "supervised maneuver Actor requires the frozen paper "
                "Actor causal observation encoder"
            )
        previous = np.asarray(
            self.previous_action, dtype=np.float64
        ).reshape(1, -1)
        raw, _ = encoder(
            np.asarray(state, dtype=np.float64).reshape(1, -1),
            previous,
            observation,
            reference,
            self.state_spec,
            0.0,
        )
        proposals = np.asarray(
            policy.propose(raw[0], self.action_spec), dtype=np.float64
        )
        expected = (
            int(policy.heads),
            int(self.config.horizon),
            int(self.action_spec.dimension),
        )
        if proposals.shape != expected or not np.isfinite(proposals).all():
            raise ValueError(
                "supervised maneuver Actor returned invalid proposals"
            )
        # Enforce the same actuator slew contract as online execution at each
        # step without changing any downstream optimizer or safety decision.
        limited = np.empty_like(proposals)
        for head in range(proposals.shape[0]):
            prior = np.asarray(self.previous_action, dtype=np.float64)
            for step in range(proposals.shape[1]):
                prior = self.action_spec.clip(
                    proposals[head, step], prior, self.config.dt
                )
                limited[head, step] = prior
        return limited

    def _joint_actor_rollouts(
        self, state, observation, reference, guided_count, rng
    ):
        """Batch deterministic and stochastic Actor trajectories together.

        The first trajectory is the deterministic Actor mean used to
        initialize MPPI. Remaining trajectories are the persistent stochastic
        guided set. They share the same autoregressive dynamics calls, but
        stochastic noise is drawn only for the guided rows so fixed-seed
        sampling semantics remain unchanged.
        """

        count = int(guided_count)
        batch = count + 1
        states = np.repeat(
            np.asarray(state, dtype=np.float64).reshape(1, -1),
            batch,
            axis=0,
        )
        previous = np.repeat(
            self.previous_action.reshape(1, -1), batch, axis=0
        )
        means = np.empty(
            (self.config.horizon, self.action_spec.dimension),
            dtype=np.float64,
        )
        variances = np.empty_like(means)
        guided = np.empty(
            (count, self.config.horizon, self.action_spec.dimension),
            dtype=np.float64,
        )
        rollout_states = np.empty(
            (self.config.horizon, self.state_spec.dimension),
            dtype=np.float64,
        )
        reliability_controls = np.empty_like(means)
        actor_ood_scores = np.zeros(
            self.config.horizon, dtype=np.float64
        )
        support_evaluator = getattr(
            self.sampling_prior, "support_ood_scores", None
        )
        residual_context_rows = []
        residual_authority_rows = []
        for step in range(self.config.horizon):
            rollout_states[step] = states[0]
            distribution = self.sampling_prior.action_distribution(
                states,
                previous,
                observation,
                reference,
                self.state_spec,
                step * self.config.dt,
            )
            if "residual_context_features" in distribution:
                residual_context_rows.append(np.asarray(
                    distribution["residual_context_features"][0],
                    dtype=np.float64,
                ))
            if "residual_correction_authority" in distribution:
                residual_authority_rows.append(float(
                    distribution["residual_correction_authority"][0]
                ))
            command = np.asarray(
                distribution["physical_mean"], dtype=np.float64
            ).copy()
            means[step] = command[0]
            variances[step] = (
                np.asarray(distribution["physical_std"])[0] ** 2
            )
            if self.hybrid_sampling_reliability.config.enabled:
                if not callable(support_evaluator):
                    raise TypeError(
                        "reliability-calibrated HSS requires Actor "
                        "support_ood_scores"
                    )
                actor_ood_scores[step] = float(
                    support_evaluator(
                        distribution["raw_observation"][:1]
                    )[0]
                )
            if count:
                guided_distribution = {
                    key: (
                        np.asarray(value)[1:]
                        if isinstance(value, np.ndarray)
                        and value.shape[:1] == (batch,)
                        else value
                    )
                    for key, value in distribution.items()
                }
                command[1:] = (
                    self.sampling_prior.sample_from_distribution(
                        guided_distribution, rng
                    )
                )
            command = self.action_spec.clip(
                command, previous=previous, dt=self.config.dt
            )
            means[step] = command[0]
            if count:
                guided[:, step, :] = command[1:]
            applied = self._delayed_control(command, previous)
            reliability_controls[step] = applied[0]
            states = integrate_batch(
                self.dynamics,
                states,
                applied,
                self.config.dt,
                self.state_spec,
                self.config.integrator,
            )
            previous = command
        if not all(
            np.isfinite(values).all()
            for values in (means, variances, guided)
        ):
            raise FloatingPointError(
                "joint Actor rollout produced NaN or Inf"
            )
        return means, variances, guided, {
            "states": rollout_states,
            "controls": reliability_controls,
            "actor_ood_scores": actor_ood_scores,
            "residual_context": np.asarray(
                residual_context_rows, dtype=np.float64
            ),
            "residual_authority": np.asarray(
                residual_authority_rows, dtype=np.float64
            ),
            "terminal_state": states[0].copy(),
        }

    def _counterfactual_proposal_gate(
        self,
        state,
        actor_terminal_state,
        baseline_mean,
        reference,
    ):
        """Compare Actor and trusted-baseline path progress under ICODE.

        This is a causal model-based authority signal.  Both alternatives are
        rolled out from the same current state with the same declared planner
        dynamics.  The public polyline projection is read-only; no simulator
        obstacle truth or future execution state is consulted.
        """

        cfg = self.paper_rl_driven_config
        project = getattr(reference, "project", None)
        enabled = bool(
            cfg.counterfactual_proposal_gate_enabled
            and self.hybrid_sampling_reliability.config.enabled
            and callable(project)
        )
        diagnostics = {
            "reliability_counterfactual_enabled": enabled,
            "reliability_counterfactual_authority": 1.0,
            "reliability_counterfactual_actor_progress": 0.0,
            "reliability_counterfactual_baseline_progress": 0.0,
            "reliability_counterfactual_actor_cross_track": 0.0,
            "reliability_counterfactual_baseline_cross_track": 0.0,
            "reliability_counterfactual_advantage": 0.0,
        }
        if not enabled:
            return 1.0, diagnostics
        baseline = np.asarray(baseline_mean, dtype=np.float64)
        if baseline.shape != (
            self.config.horizon,
            self.action_spec.dimension,
        ):
            raise ValueError("counterfactual baseline sequence is invalid")
        baseline_terminal = self.rollout(
            np.asarray(state, dtype=np.float64), baseline[None, :, :]
        )[0, -1]
        actor_terminal = np.asarray(
            actor_terminal_state, dtype=np.float64
        ).reshape(-1)
        position_indices = list(self.state_spec.position_indices)
        if actor_terminal.shape != (self.state_spec.dimension,):
            raise ValueError("counterfactual Actor terminal state is invalid")
        progress_floor = getattr(reference, "progress", None)
        actor_projection = project(
            actor_terminal[position_indices],
            minimum_progress=progress_floor,
        )
        baseline_projection = project(
            baseline_terminal[position_indices],
            minimum_progress=progress_floor,
        )
        actor_progress = float(actor_projection.progress)
        baseline_progress = float(baseline_projection.progress)
        actor_cross_track = float(actor_projection.cross_track_error)
        baseline_cross_track = float(baseline_projection.cross_track_error)
        advantage = (
            actor_progress
            - baseline_progress
            - float(cfg.counterfactual_cross_track_weight)
            * max(0.0, actor_cross_track - baseline_cross_track)
        )
        soft = float(cfg.counterfactual_progress_soft_m)
        hard = float(cfg.counterfactual_progress_hard_m)
        authority = float(np.clip(
            (advantage - hard) / (soft - hard), 0.0, 1.0
        ))
        if advantage >= soft:
            authority = 1.0
        diagnostics.update({
            "reliability_counterfactual_authority": authority,
            "reliability_counterfactual_actor_progress": actor_progress,
            "reliability_counterfactual_baseline_progress": baseline_progress,
            "reliability_counterfactual_actor_cross_track": actor_cross_track,
            "reliability_counterfactual_baseline_cross_track": baseline_cross_track,
            "reliability_counterfactual_advantage": float(advantage),
        })
        return authority, diagnostics

    @staticmethod
    def _residual_policy_context_diagnostics(context):
        values = np.asarray(context, dtype=np.float64)
        if values.size == 0:
            return {"residual_policy_context_enabled": False}
        if values.ndim != 2 or values.shape[1] < 5:
            raise ValueError("residual policy context diagnostics are invalid")
        state_count = (values.shape[1] - 3) // 2
        if 2 * state_count + 3 != values.shape[1] or state_count <= 0:
            raise ValueError("residual policy context dimension is invalid")
        residual = values[:, :state_count]
        innovation = values[:, state_count : 2 * state_count]
        disagreement = values[:, -3]
        support = values[:, -2]
        valid = values[:, -1]
        return {
            "residual_policy_context_enabled": True,
            "residual_policy_predicted_abs_mean": float(
                np.mean(np.abs(residual))
            ),
            "residual_policy_predicted_abs_max": float(
                np.max(np.abs(residual))
            ),
            "residual_policy_innovation_abs_mean": float(
                np.mean(np.abs(innovation))
            ),
            "residual_policy_innovation_abs_max": float(
                np.max(np.abs(innovation))
            ),
            "residual_policy_disagreement_mean": float(
                np.mean(disagreement)
            ),
            "residual_policy_support_mean": float(np.mean(support)),
            "residual_policy_innovation_valid_fraction": float(
                np.mean(valid > 0.5)
            ),
        }

    def _residual_for_reliability(self):
        if self.reliability_residual is not None:
            return self.reliability_residual
        residual = getattr(self.dynamics, "residual", None)
        visited = set()
        while residual is not None and id(residual) not in visited:
            visited.add(id(residual))
            if (
                callable(getattr(residual, "disagreement", None))
                and callable(
                    getattr(residual, "support_confidence", None)
                )
            ):
                return residual
            residual = getattr(residual, "residual", None)
        raise TypeError(
            "reliability-calibrated HSS requires an ensemble residual"
        )

    def _gaussian_samples(self, mean, variance, count, rng):
        samples = mean[None, :, :] + rng.normal(
            size=(int(count),) + mean.shape
        ) * np.sqrt(variance)[None, :, :]
        samples = np.clip(
            samples, self.action_spec.lower, self.action_spec.upper
        )
        if count > 0:
            samples[0] = mean
        previous = np.repeat(
            self.previous_action.reshape(1, -1), int(count), axis=0
        )
        for step in range(self.config.horizon):
            samples[:, step, :] = self.action_spec.clip(
                samples[:, step, :],
                previous=previous,
                dt=self.config.dt,
            )
            previous = samples[:, step, :]
        return samples

    def _paper_terminal_cost(
        self,
        trajectories,
        controls,
        observation,
        reference,
        causal_dynamics_confidence=1.0,
    ):
        cfg = self.paper_rl_driven_config
        if cfg.terminal_value_weight <= 0.0:
            return np.zeros(controls.shape[0], dtype=np.float64), {
                "terminal_value_enabled": False,
                "terminal_value_weight": 0.0,
            }
        terminal_states = trajectories[:, -1, :]
        terminal_controls = controls[:, -1, :]
        terminal_cfg = self.conservative_terminal_reliability.config
        if terminal_cfg.enabled:
            evaluator = getattr(
                self.sampling_prior, "terminal_value_details", None
            )
            if not callable(evaluator):
                raise TypeError(
                    "conservative terminal requires "
                    "terminal_value_details"
                )
            details = evaluator(
                terminal_states,
                terminal_controls,
                observation,
                reference,
                self.state_spec,
                self.config.horizon * self.config.dt,
                critic_source=cfg.terminal_critic_source,
            )
            returns = np.asarray(details["returns"], dtype=np.float64)
            diagnostics = dict(details["diagnostics"])
            confidence = self.conservative_terminal_reliability.evaluate(
                self._residual_for_reliability(),
                terminal_states,
                terminal_controls,
                details["critic_ood_scores"],
                details["critic_disagreement"],
            )
            causal_cap = float(np.clip(
                causal_dynamics_confidence, 0.0, 1.0
            ))
            raw_authority = np.asarray(
                confidence["authority"], dtype=np.float64
            )
            # The terminal-state estimator must not overrule causal evidence
            # already available at the current executed state.  Without this
            # cap the planner can reject the RL proposal as dynamically
            # untrustworthy while simultaneously granting its critic full
            # authority, which lets an extrapolating value function dominate
            # the geometric MPPI cost.  The cap implements the intended
            # cross-layer contract: value authority cannot exceed current
            # ICODE/innovation dynamics confidence.
            authority = np.minimum(raw_authority, causal_cap)
            uncertainty_cost = (
                terminal_cfg.uncertainty_penalty_weight
                * confidence["dynamics_uncertainty"]
            )
            cost = (
                -cfg.terminal_value_weight * authority * returns
                + uncertainty_cost
            )
            diagnostics.update({
                "terminal_value_enabled": True,
                "terminal_value_weight": float(
                    cfg.terminal_value_weight
                ),
                "terminal_value_conservative_enabled": True,
                "terminal_value_authority_mean": float(
                    np.mean(authority)
                ),
                "terminal_value_authority_min": float(
                    np.min(authority)
                ),
                "terminal_value_authority_max": float(
                    np.max(authority)
                ),
                "terminal_value_raw_authority_mean": float(
                    np.mean(raw_authority)
                ),
                "terminal_value_causal_dynamics_cap": causal_cap,
                "terminal_value_causal_cap_active_fraction": float(
                    np.mean(authority < raw_authority - 1e-12)
                ),
                "terminal_value_dynamics_confidence_mean": float(
                    np.mean(confidence["dynamics_confidence"])
                ),
                "terminal_value_critic_confidence_mean": float(
                    np.mean(confidence["critic_confidence"])
                ),
                "terminal_value_uncertainty_mean": float(
                    np.mean(confidence["dynamics_uncertainty"])
                ),
                "terminal_value_uncertainty_cost_mean": float(
                    np.mean(uncertainty_cost)
                ),
                "terminal_value_uncertainty_penalty_weight": float(
                    terminal_cfg.uncertainty_penalty_weight
                ),
                "terminal_value_cost_mean": float(np.mean(cost)),
                "terminal_value_cost_std": float(np.std(cost)),
                "terminal_value_sign": (
                    "cost=-weight*authority*return+uncertainty"
                ),
                "terminal_value_safe_fallback": (
                    "existing_mppi_geometric_terminal"
                ),
            })
            if cost.shape != (controls.shape[0],) or not np.isfinite(
                cost
            ).all():
                raise FloatingPointError(
                    "conservative terminal critic cost is invalid"
                )
            return cost, diagnostics
        returns, diagnostics = self.sampling_prior.terminal_value(
            terminal_states,
            terminal_controls,
            observation,
            reference,
            self.state_spec,
            self.config.horizon * self.config.dt,
            critic_source=cfg.terminal_critic_source,
        )
        # SAC maximizes return; MPPI minimizes cost. This sign conversion is
        # explicit and regression tested.
        cost = -cfg.terminal_value_weight * np.asarray(
            returns, dtype=np.float64
        )
        if cost.shape != (controls.shape[0],) or not np.isfinite(cost).all():
            raise FloatingPointError("paper terminal critic cost is invalid")
        result = dict(diagnostics)
        result.update({
            "terminal_value_enabled": True,
            "terminal_value_weight": float(cfg.terminal_value_weight),
            "terminal_value_conservative_enabled": False,
            "terminal_value_cost_mean": float(np.mean(cost)),
            "terminal_value_cost_std": float(np.std(cost)),
            "terminal_value_sign": "cost=-weight*return",
        })
        return cost, result

    def _solve_plan(
        self,
        state,
        prior,
        target,
        obstacles,
        rng,
        observation=None,
        reference=None,
    ):
        if observation is None or reference is None:
            raise ValueError(
                "paper RL-Driven MPPI requires observation and reference"
            )
        tracker_diagnostics = dict(
            observation.auxiliary.get("dynamic_obstacle_tracker", {})
        )
        known_static_obstacles = tuple(
            observation.auxiliary.get("known_static_obstacles", ())
        )
        if (
            self.config.known_static_map_cost_enabled
            and not known_static_obstacles
        ):
            raise ValueError(
                "known static-map cost is enabled but the observation "
                "contains no static geometry"
            )
        probabilistic_obstacles = ()
        if self.config.probabilistic_obstacle_risk_enabled:
            probabilistic_obstacles = tuple(
                observation.auxiliary.get(
                    self.config.probabilistic_obstacle_forecast_key, ()
                )
            )
            if not probabilistic_obstacles:
                # Standard MPPI owns the frozen missing-forecast fail-closed
                # contract.  Its early return happens before sampling, so this
                # delegates only the deterministic stop path and never swaps
                # the Paper optimizer when a forecast is available.
                return MppiController._solve_plan(
                    self,
                    state,
                    prior,
                    target,
                    obstacles,
                    rng,
                    observation,
                    reference,
                )
        cfg = self.paper_rl_driven_config
        if (
            cfg.standard_fallback_on_advantage_veto
            and self.proposal_advantage_gate.authority == 0.0
        ):
            return self._solve_standard_advantage_fallback(
                state,
                prior,
                target,
                obstacles,
                rng,
                observation,
                reference,
            )
        proposal_advantage_authority_applied = float(
            self.proposal_advantage_gate.authority
        )
        raw_applied_guided_fraction = (
            self._applied_guided_fraction
            if self.hybrid_sampling_reliability.config.enabled
            else cfg.guided_fraction
        )
        terminal_floor, terminal_floor_diagnostics = (
            self._completion_preserving_guidance(state, target)
        )
        handover_authority, handover_diagnostics = (
            self._completion_handover(state, target)
        )
        pre_handover_guided_fraction = max(
            float(raw_applied_guided_fraction), terminal_floor
        )
        applied_guided_fraction = (
            pre_handover_guided_fraction * handover_authority
            * proposal_advantage_authority_applied
        )
        guided_count = int(round(
            self.config.num_samples * applied_guided_fraction
        ))
        hard_boundary_filter = bool(
            self.config.path_boundary_candidate_filter_enabled
        )
        hard_static_filter = bool(
            self.config.known_static_map_candidate_filter_enabled
            and known_static_obstacles
        )
        risk_candidate_filter = bool(
            self.config.probabilistic_obstacle_risk_enabled
            and self.config.probabilistic_obstacle_candidate_filter_enabled
        )
        supervised_head_count = int(
            getattr(self.maneuver_proposal_policy, "heads", 0)
            if self.maneuver_proposal_policy is not None else 0
        )
        # Reserve the frozen proposal-only Actor's three rows inside K before
        # any candidates are sampled.  A hard candidate filter also owns slot
        # zero as the deterministic braking reserve, so that row cannot hold a
        # supervised head.  Actor-off allocation and RNG consumption are
        # untouched by this conditional floor.
        supervised_reserved_rows = supervised_head_count + int(
            hard_boundary_filter
            or hard_static_filter
            or risk_candidate_filter
        )
        if supervised_reserved_rows > self.config.num_samples:
            raise ValueError(
                "fixed candidate budget cannot hold supervised maneuver "
                "proposals and the braking reserve"
            )
        if supervised_head_count:
            guided_count = max(guided_count, supervised_reserved_rows)
        joint_actor_batch = callable(
            getattr(
                self.sampling_prior,
                "sample_from_distribution",
                None,
            )
        )
        if joint_actor_batch:
            mean, actor_variance, guided, reliability_context = (
                self._joint_actor_rollouts(
                state,
                observation,
                reference,
                guided_count,
                rng,
            )
            )
        else:
            mean, actor_variance, reliability_context = self._actor_mean_rollout(
                state, observation, reference
            )
            guided = self._guided_rollouts(
                state, observation, reference, guided_count, rng
            )
        reliability_diagnostics = {
            "reliability_hss_enabled": False,
            "reliability_guided_fraction_applied": float(
                applied_guided_fraction
            ),
            "reliability_guided_fraction_next": float(
                applied_guided_fraction
            ),
            "reliability_causal_lag_steps": 0,
            "reliability_guided_fraction_raw_applied": float(
                raw_applied_guided_fraction
            ),
            **terminal_floor_diagnostics,
            **handover_diagnostics,
        }
        if self.hybrid_sampling_reliability.config.enabled:
            reliability_diagnostics = (
                self.hybrid_sampling_reliability.evaluate(
                    self._residual_for_reliability(),
                    reliability_context["states"],
                    reliability_context["controls"],
                    reliability_context["actor_ood_scores"],
                    actor_competence_confidence=(
                        self.source_relative_competence.confidence
                        if self.source_relative_competence.config
                        .source_competence_enabled
                        else 1.0
                    ),
                )
            )
            raw_next_fraction = float(
                reliability_diagnostics["guided_fraction"]
            )
            next_fraction = max(raw_next_fraction, terminal_floor)
            self._applied_guided_fraction = next_fraction
            reliability_diagnostics.update({
                "reliability_hss_enabled": True,
                "reliability_guided_fraction_applied": float(
                    applied_guided_fraction
                ),
                "reliability_guided_fraction_next": (
                    next_fraction * handover_authority
                ),
                "reliability_guided_fraction_raw_next": raw_next_fraction,
                "reliability_guided_fraction_raw_applied": float(
                    raw_applied_guided_fraction
                ),
                # Authority estimated during the Actor mean rollout is applied
                # on the next control cycle so stochastic guided trajectories
                # remain jointly batched and no second Actor pass is added.
                "reliability_causal_lag_steps": 1,
                **terminal_floor_diagnostics,
                **handover_diagnostics,
            })
        reliability_diagnostics.update({
            "reliability_sidecar_enabled": bool(
                self.reliability_residual is not None
            ),
            "reliability_sidecar_type": (
                type(self.reliability_residual).__name__
                if self.reliability_residual is not None
                else "prediction_residual"
            ),
        })
        actor_mean = mean
        baseline_mean = np.asarray(prior.mean, dtype=np.float64)
        if baseline_mean.shape != actor_mean.shape:
            raise ValueError("baseline and Actor proposal means disagree")
        actor_baseline_abs_delta = np.abs(actor_mean - baseline_mean)
        counterfactual_authority, counterfactual_diagnostics = (
            self._counterfactual_proposal_gate(
                state,
                reliability_context["terminal_state"],
                baseline_mean,
                reference,
            )
        )
        reliability_diagnostics.update(counterfactual_diagnostics)
        if guided.size and counterfactual_authority < 1.0:
            # Preserve the fixed candidate count while turning the learned
            # proposal into a bounded residual around the trusted baseline.
            # This is the same causal authority used for the sampling centre;
            # it does not add privileged information or extra rollouts.
            guided = (
                counterfactual_authority * guided
                + (1.0 - counterfactual_authority)
                * baseline_mean[None, :, :]
            )
            guided = np.clip(
                guided,
                self.action_spec.lower[None, None, :],
                self.action_spec.upper[None, None, :],
            )
        # HSS authority must govern the proposal centre as well as the share
        # of guided samples. Otherwise an OOD Actor can retain complete
        # control of the Gaussian sampling mean after HSS assigns it zero
        # authority. The current authority is causal: it uses only checkpoint
        # support and already-completed transition innovation.
        proposal_reliability_authority = (
            float(reliability_diagnostics["reliability_authority"])
            if self.hybrid_sampling_reliability.config.enabled
            else 1.0
        )
        requested_proposal_authority = float(np.clip(
            handover_authority
            * proposal_reliability_authority
            * counterfactual_authority,
            0.0,
            1.0,
        )) * proposal_advantage_authority_applied
        requested_proposal_authority = float(np.clip(
            requested_proposal_authority,
            0.0,
            1.0,
        ))
        base_variance = np.broadcast_to(
            np.asarray(self.config.noise_sigma, dtype=np.float64)[None, :] ** 2,
            actor_mean.shape,
        ).copy()
        proposal_variance = (
            requested_proposal_authority * actor_variance
            + (1.0 - requested_proposal_authority) * base_variance
        )
        # The same-cycle comparator must be an Actor-free control arm.  Merely
        # removing labelled guided elites is insufficient when the Gaussian
        # population was itself sampled around an Actor-blended mean and
        # covariance.  In filtered mode the learned policy therefore affects
        # only the explicitly guided population; the Gaussian population
        # starts from the trusted baseline proposal and may absorb guided
        # elites only after they demonstrate a same-cycle cost advantage.
        same_cycle_gaussian_actor_isolated = bool(
            cfg.same_cycle_guided_cost_filter
        )
        proposal_authority = (
            0.0
            if same_cycle_gaussian_actor_isolated
            else requested_proposal_authority
        )
        if same_cycle_gaussian_actor_isolated:
            mean = baseline_mean.copy()
            variance = base_variance.copy()
        else:
            mean = (
                proposal_authority * actor_mean
                + (1.0 - proposal_authority) * baseline_mean
            )
            variance = np.clip(
                proposal_variance,
                base_variance * cfg.covariance_min_scale ** 2,
                base_variance * cfg.covariance_max_scale ** 2,
            )
        initial_proposal_mean = mean.copy()
        reliability_diagnostics.update({
            "reliability_proposal_authority": proposal_authority,
            "reliability_proposal_fallback_fraction": (
                1.0 - proposal_authority
            ),
            "reliability_requested_proposal_authority": (
                requested_proposal_authority
            ),
            "paper_same_cycle_gaussian_actor_isolated": (
                same_cycle_gaussian_actor_isolated
            ),
        })
        supervised = self._supervised_maneuver_rollouts(
            state, observation, reference
        )
        supervised_count = int(supervised.shape[0])
        if supervised_count != supervised_head_count:
            raise ValueError(
                "supervised maneuver proposal count changed after fixed "
                "allocation"
            )
        if supervised_count > guided_count:
            raise ValueError(
                "fixed guided allocation cannot hold all supervised "
                "maneuver proposals"
            )
        supervised_start = guided_count - supervised_count
        if supervised_count:
            # Replace existing guided rows; never add samples or rollouts.
            # Use the tail of the guided allocation because slot zero is the
            # controller's existing deterministic braking reserve.
            guided[supervised_start:guided_count] = supervised
        gaussian_count = self.config.num_samples - guided_count
        # Critical fidelity property: generated once, reused unchanged.
        minimum_variance = (
            base_variance * cfg.covariance_min_scale ** 2
        )
        maximum_variance = (
            base_variance * cfg.covariance_max_scale ** 2
        )

        total_guided_elites = 0
        total_gaussian_elites = 0
        total_supervised_elites = 0
        total_supervised_risk_feasible = 0
        total_supervised_opportunities = 0
        supervised_selected_count = 0
        supervised_selected_head = -1
        supervised_elite_weight_sum = 0.0
        supervised_head_risk_feasible = np.zeros(
            supervised_count, dtype=np.int64
        )
        supervised_head_elites = np.zeros(
            supervised_count, dtype=np.int64
        )
        supervised_head_selected = np.zeros(
            supervised_count, dtype=np.int64
        )
        supervised_head_behavior = []
        behavior_window = max(1, min(self.config.horizon, 12))
        for head in range(supervised_count):
            mean_omega = float(np.mean(
                supervised[head, :behavior_window, 1]
            ))
            if mean_omega > 0.05:
                supervised_head_behavior.append("left")
            elif mean_omega < -0.05:
                supervised_head_behavior.append("right")
            else:
                supervised_head_behavior.append("yield")
        total_guided_opportunities = 0
        total_gaussian_opportunities = 0
        guided_costs_by_iteration = []
        gaussian_costs_by_iteration = []
        guided_feasible_count = 0
        guided_boundary_feasible_count = 0
        guided_static_feasible_count = 0
        guided_risk_feasible_count = 0
        guided_boundary_first_failure_steps = []
        guided_static_first_failure_steps = []
        guided_risk_first_failure_steps = []
        gaussian_feasible_count = 0
        costs_by_iteration = []
        terminal_diagnostics = {}
        constraints = None
        final_weights = None
        boundary_feasible_fractions = []
        boundary_no_feasible_iterations = 0
        boundary_last_feasible = None
        boundary_last_first_step_feasible = None
        boundary_last_emergency_prefix_feasible = None
        boundary_last_samples = None
        boundary_last_costs = None
        static_feasible_fractions = []
        static_no_feasible_iterations = 0
        static_last_feasible = None
        static_last_min_clearance = None
        static_last_samples = None
        static_last_costs = None
        risk_feasible_fractions = []
        risk_no_feasible_iterations = 0
        risk_last_candidate = None
        risk_last_feasible = None
        risk_last_samples = None
        risk_last_costs = None
        reference_authority = 1.0
        reference_risk_raw = 0.0
        reference_authority_updated = False
        emergency_context = self._probabilistic_emergency_context(
            observation, state, probabilistic_obstacles
        )
        traversal_context = self._probabilistic_traversal_window_context(
            state,
            reference,
            probabilistic_obstacles,
            temporal_emergency_triggered=bool(
                emergency_context["triggered"]
            ),
            temporal_emergency_raw_triggered=bool(
                emergency_context.get("raw_triggered", False)
            ),
            temporal_emergency_closing_observed=bool(
                emergency_context.get("closing_observed", False)
            ),
            temporal_emergency_ttc_s=float(
                emergency_context.get("ttc_s", float("inf"))
            ),
            terminal_phase=bool(target.phase == "terminal"),
        )
        # Preserve the raw causal LaserScan warning separately from the
        # one-shot emergency intent/rearm latch.  The post-center transaction
        # must not interpret an exhausted intent hold as a cleared obstacle
        # while the same scan flow still reports closing motion.
        traversal_context["temporal_emergency_raw_triggered"] = bool(
            emergency_context.get("raw_triggered", False)
        )
        traversal_context["temporal_emergency_closing_observed"] = bool(
            emergency_context.get("closing_observed", False)
        )
        traversal_context["temporal_emergency_scan_valid"] = bool(
            emergency_context.get("scan_valid", False)
        )
        traversal_context["temporal_emergency_scan_flow_match"] = bool(
            emergency_context.get("scan_flow_match", False)
        )
        traversal_context["temporal_emergency_ttc_s"] = float(
            emergency_context.get("ttc_s", float("inf"))
        )
        traversal_context[
            "temporal_emergency_safety_hard_stop_ttc_s"
        ] = float(
            emergency_context.get("safety_hard_stop_ttc_s", 0.0)
        )
        traversal_context["temporal_emergency_rearm_ready"] = bool(
            emergency_context.get("rearm_ready", True)
        )
        post_center_forward_exit_commit_requested = bool(
            self.config
            .probabilistic_obstacle_traversal_window_post_center_forward_exit_commit_enabled
            and traversal_context.get(
                "post_center_forward_exit_commit_active", False
            )
        )
        emergency_context[
            "post_center_forward_exit_commit_requested"
        ] = post_center_forward_exit_commit_requested
        traversal_context[
            "post_center_forward_exit_commit_requested"
        ] = post_center_forward_exit_commit_requested
        post_center_low_ttc_nonforward_coverage_requested = bool(
            self.config
            .probabilistic_obstacle_traversal_window_post_center_low_ttc_nonforward_coverage_enabled
            and self.config
            .probabilistic_obstacle_traversal_window_post_center_low_ttc_continuity_guard_enabled
            and traversal_context.get("candidate_requested", False)
            and traversal_context.get("commit_started", False)
            and not traversal_context.get("retreat_requested", False)
            and not traversal_context.get("rearm_pending", False)
            and float(traversal_context.get("current_progress", 0.0))
            >= float(traversal_context.get("crossing_progress", 0.0))
            - 1.0e-9
            and float(traversal_context.get("current_progress", 0.0))
            < float(traversal_context.get("clear_progress", 0.0))
            - 1.0e-9
            and emergency_context.get("scan_valid", False)
            and float(
                emergency_context.get("safety_hard_stop_ttc_s", 0.0)
            ) > 0.0
            and np.isfinite(float(
                emergency_context.get("ttc_s", float("inf"))
            ))
            and 0.0 < float(
                emergency_context.get("ttc_s", float("inf"))
            ) <= float(
                emergency_context.get("safety_hard_stop_ttc_s", 0.0)
            )
            and not emergency_context.get("closing_observed", False)
            and not emergency_context.get("rearm_ready", True)
            and not post_center_forward_exit_commit_requested
        )
        emergency_context[
            "post_center_low_ttc_nonforward_coverage_requested"
        ] = post_center_low_ttc_nonforward_coverage_requested
        traversal_context[
            "post_center_low_ttc_nonforward_coverage_requested"
        ] = post_center_low_ttc_nonforward_coverage_requested
        emergency_context, traversal_context = (
            self._bind_probabilistic_traversal_exit_deadline_retreat_escape(
                emergency_context, traversal_context, state
            )
        )
        emergency_context, traversal_context = (
            self._bind_probabilistic_traversal_post_center_temporal_escape(
                emergency_context, traversal_context, state
            )
        )
        exit_deadline_retreat_emergency_lattice_requested = bool(
            traversal_context.get(
                "exit_deadline_retreat_escape_transaction_active", False
            )
        )
        admission_exit_deadline_emergency_lattice_requested = bool(
            traversal_context.get(
                "commit_admission_exit_deadline_hold_requested", False
            )
        )
        uncommitted_temporal_staging_emergency_lattice_requested = bool(
            traversal_context.get(
                "uncommitted_temporal_staging_hold_requested", False
            )
        )
        rearm_hard_risk_emergency_lattice_requested = bool(
            self.config
            .probabilistic_obstacle_traversal_window_rearm_hard_risk_temporal_lattice_override_enabled
            and traversal_context.get("candidate_requested", False)
            and traversal_context.get("rearm_pending", False)
            and traversal_context.get(
                "temporal_emergency_raw_triggered", False
            )
        )
        temporal_retreat_raw_emergency_lattice_requested = bool(
            self.config
            .probabilistic_obstacle_traversal_window_temporal_retreat_raw_lattice_binding_enabled
            and traversal_context.get("candidate_requested", False)
            and traversal_context.get("retreat_requested", False)
            and traversal_context.get(
                "retreat_temporal_lattice_requested", False
            )
            and traversal_context.get(
                "temporal_emergency_raw_triggered", False
            )
        )
        traversal_context["temporal_retreat_raw_lattice_requested"] = bool(
            temporal_retreat_raw_emergency_lattice_requested
        )
        post_center_emergency_lattice_requested = bool(
            self.config
            .probabilistic_obstacle_traversal_window_post_center_hard_risk_override_enabled
            and traversal_context.get("candidate_requested", False)
            and traversal_context.get("commit_started", False)
            and not traversal_context.get("retreat_requested", False)
            and not traversal_context.get("rearm_pending", False)
            and float(traversal_context.get("current_progress", 0.0))
            >= float(traversal_context.get("crossing_progress", 0.0))
            - 1.0e-9
            and float(traversal_context.get("current_progress", 0.0))
            < float(traversal_context.get("clear_progress", 0.0))
            - 1.0e-9
        )
        emergency_last_mask = np.zeros(
            self.config.num_samples, dtype=bool
        )
        traversal_last_index = -1
        same_cycle_guided_filter_iterations = 0
        same_cycle_guided_filtered_candidates = 0
        for _ in range(cfg.iterations):
            gaussian = self._gaussian_samples(
                mean, variance, gaussian_count, rng
            )
            samples = np.concatenate((guided, gaussian), axis=0)
            labels = np.concatenate((
                np.ones(guided_count, dtype=np.int8),
                np.zeros(gaussian_count, dtype=np.int8),
            ))
            if supervised_count:
                labels[supervised_start:guided_count] = (
                    2 + np.arange(supervised_count, dtype=np.int8)
                )
            if (
                hard_boundary_filter
                or hard_static_filter
                or risk_candidate_filter
            ):
                # Keep K fixed while reserving one deterministic braking
                # candidate.  Label -1 excludes it from guided/Gaussian source
                # competence accounting.
                # When no guided proposals are allocated, Gaussian candidate
                # zero is the exact proposal mean.  Preserve that deterministic
                # warm start before replacing slot zero with braking; otherwise
                # boundary filtering removes the only unperturbed proposal in
                # the high-dimensional sequence search.
                if guided_count == 0 and self.config.num_samples > 1:
                    samples[1] = samples[0]
                    labels[1] = labels[0]
                samples[0] = 0.0
                samples[0, 0] = self.action_spec.clip(
                    samples[0, 0], self.previous_action, self.config.dt
                )
                labels[0] = -1
            emergency_mask = np.zeros(
                self.config.num_samples, dtype=bool
            )
            if (
                self.config.probabilistic_obstacle_risk_enabled
                and self.config
                .probabilistic_obstacle_emergency_candidates_enabled
                and (
                    emergency_context["triggered"]
                    or post_center_emergency_lattice_requested
                    or exit_deadline_retreat_emergency_lattice_requested
                    or admission_exit_deadline_emergency_lattice_requested
                    or uncommitted_temporal_staging_emergency_lattice_requested
                    or rearm_hard_risk_emergency_lattice_requested
                    or temporal_retreat_raw_emergency_lattice_requested
                )
            ):
                emergency_mask = (
                    self._inject_probabilistic_emergency_candidates(
                        samples, mean, emergency_context
                    )
                )
                # Emergency lattice slots are neither Actor-guided nor
                # Gaussian evidence for the HSS competence estimator.
                labels[emergency_mask] = -2
            traversal_index = self._inject_probabilistic_traversal_candidate(
                samples, traversal_context
            )
            protected_candidate_mask = emergency_mask.copy()
            if traversal_index >= 0:
                # The traversal proposal occupies its own fixed-budget slot.
                # Protect it from first-action slew clipping without
                # misreporting it as a member of the six-direction emergency
                # lattice passed to the same-cycle action guard.
                protected_candidate_mask[traversal_index] = True
                labels[traversal_index] = -3
            samples, constraints = self._terminal_constraints(
                state, target, samples
            )
            if (
                hard_boundary_filter
                or hard_static_filter
                or risk_candidate_filter
            ):
                regular_candidates = ~protected_candidate_mask
                samples[regular_candidates, 0, :] = self.action_spec.clip(
                    samples[regular_candidates, 0, :],
                    self.previous_action,
                    self.config.dt,
                )
            trajectories = self.rollout(state, samples)
            boundary_margins = (
                self._path_boundary_margins(trajectories, reference)
                if hard_boundary_filter else None
            )
            candidate_risk = (
                self._probabilistic_collision_risk(
                    trajectories, probabilistic_obstacles
                )
                if (
                    risk_candidate_filter
                    or self.config
                    .probabilistic_reference_authority_enabled
                )
                else None
            )
            if not reference_authority_updated:
                reference_authority, reference_risk_raw = (
                    self._probabilistic_reference_authority(
                        candidate_risk,
                        1 if self.config.num_samples > 1 else 0,
                    )
                )
                reference_authority_updated = True
            running = self._cost(
                trajectories,
                samples,
                target,
                obstacles,
                reference=reference,
                probabilistic_obstacles=probabilistic_obstacles,
                known_static_obstacles=known_static_obstacles,
                path_boundary_margins=boundary_margins,
                probabilistic_risk=candidate_risk,
                reference_authority=reference_authority,
            )
            terminal, terminal_diagnostics = self._paper_terminal_cost(
                trajectories,
                samples,
                observation,
                reference,
                causal_dynamics_confidence=float(
                    reliability_diagnostics.get("dynamics_confidence", 1.0)
                ),
            )
            # A learned critic is useful beyond the finite MPPI horizon, but
            # its coarse value geometry should not override the exact goal
            # cost during final convergence.  The same continuous authority
            # that hands the proposal back to the trusted baseline therefore
            # attenuates only the *incremental* critic term.  The unchanged
            # MPPI geometric terminal cost remains present in ``running``.
            terminal = terminal * handover_authority
            terminal_diagnostics.update({
                "terminal_value_completion_authority": float(
                    handover_authority
                ),
                "terminal_value_completion_handover_enabled": bool(
                    handover_diagnostics["completion_handover_enabled"]
                ),
            })
            costs = np.asarray(running, dtype=np.float64) + terminal
            if not np.isfinite(costs).all():
                raise FloatingPointError(
                    "paper RL-Driven MPPI cost contains NaN or Inf"
                )
            if hard_boundary_filter:
                boundary_feasible = np.min(
                    boundary_margins[:, 1:], axis=1
                ) >= 0.0
                boundary_feasible_fractions.append(float(
                    np.mean(boundary_feasible)
                ))
                boundary_last_feasible = boundary_feasible
                boundary_last_first_step_feasible = (
                    boundary_margins[:, 1] >= 0.0
                )
                emergency_prefix_steps = max(1, min(
                    int(
                        self.config
                        .probabilistic_obstacle_emergency_candidate_prefix_steps
                    ),
                    boundary_margins.shape[1] - 1,
                ))
                boundary_last_emergency_prefix_feasible = np.min(
                    boundary_margins[:, 1:emergency_prefix_steps + 1],
                    axis=1,
                ) >= 0.0
                boundary_last_samples = samples.copy()
                boundary_last_costs = costs.copy()
            else:
                boundary_feasible = np.ones(
                    self.config.num_samples, dtype=bool
                )
            static_feasible = np.ones(
                self.config.num_samples, dtype=bool
            )
            if hard_static_filter:
                static_clearance = self._known_static_map_clearance(
                    trajectories, known_static_obstacles
                )
                static_min_clearance = np.min(
                    static_clearance[:, 1:], axis=1
                )
                static_feasible = static_min_clearance >= 0.0
                static_feasible_fractions.append(float(
                    np.mean(static_feasible)
                ))
                static_last_feasible = static_feasible
                static_last_min_clearance = static_min_clearance
                static_last_samples = samples.copy()
                static_last_costs = costs.copy()
            risk_feasible = np.ones(
                self.config.num_samples, dtype=bool
            )
            if risk_candidate_filter:
                risk_feasible = ~candidate_risk.hard_violation
                risk_feasible_fractions.append(float(
                    np.mean(risk_feasible)
                ))
            geometry_feasible = (
                boundary_feasible & static_feasible
            )
            jointly_feasible = geometry_feasible & risk_feasible
            if (
                not hard_boundary_filter
                and not hard_static_filter
                and not risk_candidate_filter
            ):
                optimizer_feasible = jointly_feasible
            elif (
                (hard_boundary_filter or hard_static_filter)
                and risk_candidate_filter
                and np.any(jointly_feasible)
            ):
                optimizer_feasible = jointly_feasible
            elif (
                (hard_boundary_filter or hard_static_filter)
                and np.any(geometry_feasible)
            ):
                optimizer_feasible = geometry_feasible
            elif risk_candidate_filter and np.any(risk_feasible):
                optimizer_feasible = risk_feasible
            else:
                # Candidate zero is the deterministic braking sequence.  It
                # remains the fail-closed update if neither hard filter has a
                # feasible stochastic candidate.
                optimizer_feasible = np.zeros(
                    self.config.num_samples, dtype=bool
                )
                optimizer_feasible[0] = True
                if hard_boundary_filter and not np.any(boundary_feasible):
                    boundary_no_feasible_iterations += 1
                if hard_static_filter and not np.any(static_feasible):
                    static_no_feasible_iterations += 1
                if risk_candidate_filter and not np.any(risk_feasible):
                    risk_no_feasible_iterations += 1
            risk_last_candidate = candidate_risk
            risk_last_feasible = risk_feasible
            risk_last_samples = samples.copy()
            risk_last_costs = costs.copy()
            emergency_last_mask = emergency_mask.copy()
            traversal_last_index = int(traversal_index)
            guided_mask = labels >= 1
            supervised_mask = labels >= 2
            gaussian_mask = labels == 0
            guided_boundary_feasible_count += int(np.sum(
                boundary_feasible & guided_mask
            ))
            guided_static_feasible_count += int(np.sum(
                static_feasible & guided_mask
            ))
            guided_risk_feasible_count += int(np.sum(
                risk_feasible & guided_mask
            ))
            total_supervised_risk_feasible += int(np.sum(
                risk_feasible & supervised_mask
            ))
            total_supervised_opportunities += int(np.sum(supervised_mask))
            for head in range(supervised_count):
                supervised_head_risk_feasible[head] += int(np.sum(
                    risk_feasible & (labels == 2 + head)
                ))
            if hard_boundary_filter and np.any(guided_mask):
                guided_boundary_violations = (
                    boundary_margins[guided_mask, 1:] < 0.0
                )
                guided_boundary_failed = np.any(
                    guided_boundary_violations, axis=1
                )
                guided_boundary_first_failure_steps.extend(
                    (
                        np.argmax(
                            guided_boundary_violations[
                                guided_boundary_failed
                            ],
                            axis=1,
                        )
                        + 1
                    ).tolist()
                )
            if hard_static_filter and np.any(guided_mask):
                guided_static_violations = (
                    static_clearance[guided_mask, 1:] < 0.0
                )
                guided_static_failed = np.any(
                    guided_static_violations, axis=1
                )
                guided_static_first_failure_steps.extend(
                    (
                        np.argmax(
                            guided_static_violations[
                                guided_static_failed
                            ],
                            axis=1,
                        )
                        + 1
                    ).tolist()
                )
            if risk_candidate_filter and np.any(guided_mask):
                guided_risk_violations = (
                    candidate_risk.step_probability_upper_bound[guided_mask]
                    >= self.config.probabilistic_obstacle_hard_threshold
                )
                guided_risk_failed = np.any(
                    guided_risk_violations, axis=1
                )
                guided_risk_first_failure_steps.extend(
                    (
                        np.argmax(
                            guided_risk_violations[guided_risk_failed],
                            axis=1,
                        )
                        + 1
                    ).tolist()
                )
            if cfg.same_cycle_guided_cost_filter:
                guided_eligible = optimizer_feasible & guided_mask
                gaussian_eligible = optimizer_feasible & gaussian_mask
                if np.any(guided_eligible) and np.any(gaussian_eligible):
                    guided_best = float(np.min(costs[guided_eligible]))
                    gaussian_best = float(np.min(costs[gaussian_eligible]))
                    relative_disadvantage = (
                        (guided_best - gaussian_best)
                        / max(abs(gaussian_best), 1.0)
                    )
                    if relative_disadvantage > float(
                        cfg.same_cycle_guided_relative_margin
                    ):
                        same_cycle_guided_filter_iterations += 1
                        same_cycle_guided_filtered_candidates += int(
                            np.sum(guided_eligible)
                        )
                        optimizer_feasible = (
                            optimizer_feasible & ~guided_mask
                        )
            feasible_indices = np.flatnonzero(optimizer_feasible)
            guided_iteration_costs = costs[guided_mask]
            gaussian_iteration_costs = costs[gaussian_mask]
            if guided_iteration_costs.size:
                guided_costs_by_iteration.append(guided_iteration_costs)
            if gaussian_iteration_costs.size:
                gaussian_costs_by_iteration.append(gaussian_iteration_costs)
            total_guided_opportunities += int(np.sum(guided_mask))
            total_gaussian_opportunities += int(np.sum(gaussian_mask))
            guided_feasible_count += int(np.sum(
                jointly_feasible & guided_mask
            ))
            gaussian_feasible_count += int(np.sum(
                jointly_feasible & gaussian_mask
            ))
            requested_elites = max(
                2,
                int(np.ceil(
                    cfg.elite_fraction * self.config.num_samples
                )),
            )
            elite_count = min(int(feasible_indices.size), requested_elites)
            ranked_feasible = np.argsort(
                costs[feasible_indices], kind="stable"
            )
            elite_indices = feasible_indices[ranked_feasible[:elite_count]]
            elite_costs = costs[elite_indices]
            beta = float(np.min(elite_costs))
            exponent = np.clip(
                -(elite_costs - beta) / self.config.temperature,
                -700.0,
                0.0,
            )
            final_weights = np.exp(exponent)
            final_weights /= max(float(np.sum(final_weights)), 1e-12)
            elites = samples[elite_indices]
            mean = np.sum(
                final_weights[:, None, None] * elites, axis=0
            )
            centered = elites - mean[None, :, :]
            estimate = np.sum(
                final_weights[:, None, None] * centered ** 2, axis=0
            )
            variance = np.clip(
                cfg.covariance_smoothing * variance
                + (1.0 - cfg.covariance_smoothing) * estimate,
                minimum_variance,
                maximum_variance,
            )
            total_guided_elites += int(
                np.sum(labels[elite_indices] >= 1)
            )
            total_gaussian_elites += int(
                np.sum(labels[elite_indices] == 0)
            )
            supervised_elite_mask = labels[elite_indices] >= 2
            total_supervised_elites += int(np.sum(
                supervised_elite_mask
            ))
            supervised_elite_weight_sum = float(np.sum(
                final_weights[supervised_elite_mask]
            ))
            selected_label = int(
                labels[elite_indices[int(np.argmax(final_weights))]]
            )
            supervised_selected_count = int(selected_label >= 2)
            supervised_selected_head = (
                selected_label - 2 if selected_label >= 2 else -1
            )
            for head in range(supervised_count):
                supervised_head_elites[head] += int(np.sum(
                    labels[elite_indices] == 2 + head
                ))
                supervised_head_selected[head] = int(
                    supervised_selected_head == head
                )
            costs_by_iteration.append(costs)

        source_competence = {
            "updated": False,
            "raw_confidence": 1.0,
            "mapped_confidence": 1.0,
            "confidence": 1.0,
            "guided_yield": 0.0,
            "gaussian_yield": 0.0,
            "updates": 0,
        }
        if (
            self.hybrid_sampling_reliability.config.enabled
            and self.source_relative_competence.config
            .source_competence_enabled
        ):
            source_competence = self.source_relative_competence.update(
                total_guided_elites,
                total_gaussian_elites,
                guided_count * cfg.iterations,
                gaussian_count * cfg.iterations,
            )
            actor_factor = float(
                reliability_diagnostics[
                    "actor_support_authority_factor"
                ]
            ) * float(source_competence["confidence"])
            next_authority, next_model_routing = (
                self.hybrid_sampling_reliability
                .authority_from_components(
                    reliability_diagnostics["dynamics_confidence"],
                    actor_factor,
                )
            )
            next_level, raw_next_fraction = (
                self.hybrid_sampling_reliability
                .allocation_from_authority(next_authority)
            )
            next_fraction = max(raw_next_fraction, terminal_floor)
            self._applied_guided_fraction = next_fraction
            reliability_diagnostics.update({
                "reliability_level": next_level,
                "reliability_authority": next_authority,
                "model_routing_factor": next_model_routing,
                "actor_authority_factor": actor_factor,
                "actor_competence_confidence": float(
                    source_competence["confidence"]
                ),
                "actor_competence_raw_confidence": float(
                    source_competence["raw_confidence"]
                ),
                "actor_competence_mapped_confidence": float(
                    source_competence["mapped_confidence"]
                ),
                "actor_competence_guided_yield": float(
                    source_competence["guided_yield"]
                ),
                "actor_competence_gaussian_yield": float(
                    source_competence["gaussian_yield"]
                ),
                "actor_competence_updates": int(
                    source_competence["updates"]
                ),
                "actor_competence_updated": bool(
                    source_competence["updated"]
                ),
                "reliability_guided_fraction_next": (
                    next_fraction * handover_authority
                ),
                "reliability_guided_fraction_raw_next": (
                    raw_next_fraction
                ),
                "reliability_causal_lag_steps": 1,
            })
        elif self.hybrid_sampling_reliability.config.enabled:
            reliability_diagnostics.update({
                "actor_competence_raw_confidence": 1.0,
                "actor_competence_mapped_confidence": 1.0,
                "actor_competence_guided_yield": 0.0,
                "actor_competence_gaussian_yield": 0.0,
                "actor_competence_updates": 0,
                "actor_competence_updated": False,
            })

        sequence = np.clip(
            mean, self.action_spec.lower, self.action_spec.upper
        )
        sequence, constraints = self._terminal_constraints(
            state, target, sequence[None, :, :]
        )
        sequence = sequence[0]
        action, sequence, terminal_action_diagnostics = (
            self._finalize_terminal_action(sequence, constraints)
        )
        trajectory = self.rollout(state, sequence)[0]
        boundary_fallback_used = False
        boundary_fallback_candidate_index = -1
        boundary_final_min_margin = 0.0
        boundary_weighted_update_feasible = True
        if hard_boundary_filter:
            boundary_final_min_margin = float(np.min(
                self._path_boundary_margins(
                    trajectory[None, ...], reference
                )[0, 1:]
            ))
            boundary_weighted_update_feasible = bool(
                boundary_final_min_margin >= 0.0
            )
            if (
                not boundary_weighted_update_feasible
                and boundary_last_feasible is not None
                and np.any(boundary_last_feasible)
            ):
                feasible_indices = np.flatnonzero(boundary_last_feasible)
                boundary_fallback_candidate_index = int(
                    feasible_indices[np.argmin(
                        boundary_last_costs[feasible_indices]
                    )]
                )
                sequence = boundary_last_samples[
                    boundary_fallback_candidate_index
                ].copy()
                sequence, constraints = self._terminal_constraints(
                    state, target, sequence[None, :, :]
                )
                sequence = sequence[0]
                action, sequence, terminal_action_diagnostics = (
                    self._finalize_terminal_action(sequence, constraints)
                )
                trajectory = self.rollout(state, sequence)[0]
                boundary_fallback_used = True
                boundary_final_min_margin = float(np.min(
                    self._path_boundary_margins(
                        trajectory[None, ...], reference
                    )[0, 1:]
                ))
                boundary_weighted_update_feasible = bool(
                    boundary_final_min_margin >= 0.0
                )
        static_weighted_update_feasible = True
        static_fallback_used = False
        static_fallback_candidate_index = -1
        static_final_min_clearance = 0.0
        if hard_static_filter:
            static_final_min_clearance = float(np.min(
                self._known_static_map_clearance(
                    trajectory[None, ...],
                    known_static_obstacles,
                )[0, 1:]
            ))
            static_weighted_update_feasible = bool(
                static_final_min_clearance >= 0.0
            )
            geometry_last_feasible = static_last_feasible.copy()
            if hard_boundary_filter:
                geometry_last_feasible &= boundary_last_feasible
            if (
                not static_weighted_update_feasible
                and np.any(geometry_last_feasible)
            ):
                feasible_mask = geometry_last_feasible.copy()
                if (
                    risk_candidate_filter
                    and risk_last_feasible is not None
                    and np.any(feasible_mask & risk_last_feasible)
                ):
                    feasible_mask &= risk_last_feasible
                feasible_indices = np.flatnonzero(feasible_mask)
                static_fallback_candidate_index = int(
                    feasible_indices[np.argmin(
                        static_last_costs[feasible_indices]
                    )]
                )
                sequence = static_last_samples[
                    static_fallback_candidate_index
                ].copy()
                sequence, constraints = self._terminal_constraints(
                    state, target, sequence[None, :, :]
                )
                sequence = sequence[0]
                action, sequence, terminal_action_diagnostics = (
                    self._finalize_terminal_action(sequence, constraints)
                )
                trajectory = self.rollout(state, sequence)[0]
                static_fallback_used = True
                static_final_min_clearance = float(np.min(
                    self._known_static_map_clearance(
                        trajectory[None, ...],
                        known_static_obstacles,
                    )[0, 1:]
                ))
                static_weighted_update_feasible = bool(
                    static_final_min_clearance >= 0.0
                )
        guard_candidate_eligible = np.ones(
            self.config.num_samples, dtype=bool
        )
        if hard_boundary_filter:
            guard_candidate_eligible &= boundary_last_feasible
        if hard_static_filter:
            guard_candidate_eligible &= static_last_feasible
        boundary_handoff_scope = bool(
            hard_boundary_filter
            and traversal_context.get(
                "post_center_low_ttc_nonforward_coverage_requested", False
            )
        )
        prefix_boundary_handoff_requested = bool(
            boundary_handoff_scope
            and self.config
            .probabilistic_obstacle_traversal_window_post_center_low_ttc_prefix_boundary_handoff_enabled
        )
        first_step_boundary_handoff_requested = bool(
            boundary_handoff_scope
            and self.config
            .probabilistic_obstacle_traversal_window_post_center_low_ttc_first_step_boundary_handoff_enabled
            and not prefix_boundary_handoff_requested
        )
        handoff_boundary_eligible = (
            boundary_last_emergency_prefix_feasible
            if prefix_boundary_handoff_requested
            and boundary_last_emergency_prefix_feasible is not None
            else boundary_last_first_step_feasible
            if boundary_last_first_step_feasible is not None
            else guard_candidate_eligible
        )
        guard_candidate_eligible, first_step_boundary_handoff_mask = (
            self._emergency_first_step_boundary_handoff(
                guard_candidate_eligible,
                handoff_boundary_eligible,
                emergency_last_mask,
                enabled=(
                    first_step_boundary_handoff_requested
                    or prefix_boundary_handoff_requested
                ),
            )
        )
        if hard_static_filter:
            # A boundary handoff may deliberately relax the path corridor, but
            # it must never relax the known-static-map collision contract.
            guard_candidate_eligible &= static_last_feasible
        traversal_context[
            "post_center_low_ttc_first_step_boundary_handoff_requested"
        ] = first_step_boundary_handoff_requested
        traversal_context[
            "_post_center_low_ttc_first_step_boundary_handoff_mask"
        ] = first_step_boundary_handoff_mask
        traversal_context[
            "post_center_low_ttc_prefix_boundary_handoff_requested"
        ] = prefix_boundary_handoff_requested
        action, sequence, trajectory, probabilistic_risk_diagnostics = (
            self._apply_probabilistic_obstacle_action_guard(
                state,
                action,
                sequence,
                trajectory,
                risk_last_samples,
                risk_last_costs,
                probabilistic_obstacles,
                candidate_eligible=guard_candidate_eligible,
                candidate_risk=risk_last_candidate,
                emergency_candidate_mask=emergency_last_mask,
                temporal_emergency_triggered=bool(
                    emergency_context["triggered"]
                ),
                traversal_context=traversal_context,
                traversal_candidate_index=traversal_last_index,
            )
        )
        if hard_static_filter:
            static_final_min_clearance = float(np.min(
                self._known_static_map_clearance(
                    trajectory[None, ...],
                    known_static_obstacles,
                )[0, 1:]
            ))
            static_weighted_update_feasible = bool(
                static_final_min_clearance >= 0.0
            )
            if (
                not static_weighted_update_feasible
                and static_last_feasible is not None
            ):
                geometry_last_feasible = static_last_feasible.copy()
                if hard_boundary_filter:
                    geometry_last_feasible &= boundary_last_feasible
                if np.any(geometry_last_feasible):
                    feasible_mask = geometry_last_feasible.copy()
                    if (
                        risk_candidate_filter
                        and risk_last_feasible is not None
                        and np.any(feasible_mask & risk_last_feasible)
                    ):
                        feasible_mask &= risk_last_feasible
                    feasible_indices = np.flatnonzero(feasible_mask)
                    static_fallback_candidate_index = int(
                        feasible_indices[np.argmin(
                            static_last_costs[feasible_indices]
                        )]
                    )
                    sequence = static_last_samples[
                        static_fallback_candidate_index
                    ].copy()
                    sequence, constraints = self._terminal_constraints(
                        state, target, sequence[None, :, :]
                    )
                    sequence = sequence[0]
                    action, sequence, terminal_action_diagnostics = (
                        self._finalize_terminal_action(
                            sequence, constraints
                        )
                    )
                    trajectory = self.rollout(state, sequence)[0]
                    static_fallback_used = True
                    static_final_min_clearance = float(np.min(
                        self._known_static_map_clearance(
                            trajectory[None, ...],
                            known_static_obstacles,
                        )[0, 1:]
                    ))
                    static_weighted_update_feasible = bool(
                        static_final_min_clearance >= 0.0
                    )
        all_costs = np.concatenate(costs_by_iteration)
        guided_costs = (
            np.concatenate(guided_costs_by_iteration)
            if guided_costs_by_iteration
            else np.asarray([], dtype=np.float64)
        )
        gaussian_costs = (
            np.concatenate(gaussian_costs_by_iteration)
            if gaussian_costs_by_iteration
            else np.asarray([], dtype=np.float64)
        )
        guided_cost_observed = bool(guided_costs.size)
        gaussian_cost_observed = bool(gaussian_costs.size)
        guided_cost_min = (
            float(np.min(guided_costs)) if guided_cost_observed else 0.0
        )
        gaussian_cost_min = (
            float(np.min(gaussian_costs))
            if gaussian_cost_observed else 0.0
        )
        guided_cost_mean = (
            float(np.mean(guided_costs)) if guided_cost_observed else 0.0
        )
        gaussian_cost_mean = (
            float(np.mean(gaussian_costs))
            if gaussian_cost_observed else 0.0
        )
        guided_cost_p50 = (
            float(np.median(guided_costs)) if guided_cost_observed else 0.0
        )
        gaussian_cost_p50 = (
            float(np.median(gaussian_costs))
            if gaussian_cost_observed else 0.0
        )
        proposal_advantage_diagnostics = self.proposal_advantage_gate.update(
            guided_cost_min,
            gaussian_cost_min,
            observed=(guided_cost_observed and gaussian_cost_observed),
            authority_applied=proposal_advantage_authority_applied,
        )
        reliability_diagnostics["reliability_guided_fraction_next"] = float(
            reliability_diagnostics.get(
                "reliability_guided_fraction_next",
                applied_guided_fraction,
            )
            * proposal_advantage_diagnostics[
                "reliability_proposal_advantage_authority_next"
            ]
        )
        optimizer_diagnostics = {
            "optimizer_diagnostics_enabled": False,
            "optimizer_best_candidate_cost": 0.0,
            "optimizer_selected_sequence_cost": 0.0,
            "optimizer_selected_cost_gap": 0.0,
            "optimizer_best_first_v": 0.0,
            "optimizer_best_first_omega": 0.0,
            "optimizer_selected_first_v": 0.0,
            "optimizer_selected_first_omega": 0.0,
            "optimizer_first_action_cancellation_ratio": 1.0,
            "optimizer_initial_proposal_first_v": 0.0,
            "optimizer_initial_proposal_first_omega": 0.0,
        }
        if self.config.optimizer_diagnostics_enabled:
            diagnostic_eligible = np.ones(
                self.config.num_samples, dtype=bool
            )
            if hard_boundary_filter:
                diagnostic_eligible &= boundary_last_feasible
            if hard_static_filter:
                diagnostic_eligible &= static_last_feasible
            diagnostic_indices = (
                np.flatnonzero(diagnostic_eligible)
                if (
                    (hard_boundary_filter or hard_static_filter)
                    and np.any(diagnostic_eligible)
                )
                else np.arange(self.config.num_samples, dtype=np.int64)
            )
            best_index = int(diagnostic_indices[np.argmin(
                static_last_costs[diagnostic_indices]
                if hard_static_filter and static_last_costs is not None
                else boundary_last_costs[diagnostic_indices]
                if hard_boundary_filter and boundary_last_costs is not None
                else costs[diagnostic_indices]
            )])
            diagnostic_samples = (
                static_last_samples
                if hard_static_filter and static_last_samples is not None
                else boundary_last_samples
                if hard_boundary_filter and boundary_last_samples is not None
                else samples
            )
            diagnostic_costs = (
                static_last_costs
                if hard_static_filter and static_last_costs is not None
                else boundary_last_costs
                if hard_boundary_filter and boundary_last_costs is not None
                else costs
            )
            best_action = self.action_spec.clip(
                diagnostic_samples[best_index, 0],
                self.previous_action,
                self.config.dt,
            )
            selected_running = float(self._cost(
                trajectory[None, ...],
                sequence[None, ...],
                target,
                obstacles,
                reference=reference,
                probabilistic_obstacles=probabilistic_obstacles,
                known_static_obstacles=known_static_obstacles,
                reference_authority=reference_authority,
            )[0])
            selected_terminal = float(self._paper_terminal_cost(
                trajectory[None, ...],
                sequence[None, ...],
                observation,
                reference,
                causal_dynamics_confidence=float(
                    reliability_diagnostics.get(
                        "dynamics_confidence", 1.0
                    )
                ),
            )[0][0]) * handover_authority
            selected_cost = selected_running + selected_terminal
            v_index_diag = (
                self.action_spec.index("v_cmd")
                if "v_cmd" in self.action_spec.names else None
            )
            omega_index_diag = (
                self.action_spec.index("omega_cmd")
                if "omega_cmd" in self.action_spec.names else None
            )
            best_norm = float(np.linalg.norm(best_action))
            selected_norm = float(np.linalg.norm(action))
            optimizer_diagnostics = {
                "optimizer_diagnostics_enabled": True,
                "optimizer_best_candidate_cost": float(
                    diagnostic_costs[best_index]
                ),
                "optimizer_selected_sequence_cost": selected_cost,
                "optimizer_selected_cost_gap": (
                    selected_cost - float(diagnostic_costs[best_index])
                ),
                "optimizer_best_first_v": (
                    float(best_action[v_index_diag])
                    if v_index_diag is not None else 0.0
                ),
                "optimizer_best_first_omega": (
                    float(best_action[omega_index_diag])
                    if omega_index_diag is not None else 0.0
                ),
                "optimizer_selected_first_v": (
                    float(action[v_index_diag])
                    if v_index_diag is not None else 0.0
                ),
                "optimizer_selected_first_omega": (
                    float(action[omega_index_diag])
                    if omega_index_diag is not None else 0.0
                ),
                "optimizer_first_action_cancellation_ratio": (
                    selected_norm / max(best_norm, 1e-12)
                ),
                "optimizer_initial_proposal_first_v": (
                    float(initial_proposal_mean[0, v_index_diag])
                    if v_index_diag is not None else 0.0
                ),
                "optimizer_initial_proposal_first_omega": (
                    float(initial_proposal_mean[0, omega_index_diag])
                    if omega_index_diag is not None else 0.0
                ),
            }
        effective_sample_size = float(
            1.0 / np.sum(np.asarray(final_weights) ** 2)
        )
        online_tracker_diagnostics = {
            "dynamic_obstacle_tracker_enabled": bool(
                tracker_diagnostics.get("enabled", False)
            ),
            "dynamic_obstacle_tracker_cluster_count": int(
                tracker_diagnostics.get("cluster_count", 0)
            ),
            "dynamic_obstacle_tracker_observation_count": int(
                tracker_diagnostics.get(
                    "associated_observation_count", 0
                )
            ),
            "dynamic_obstacle_tracker_history_length": int(
                tracker_diagnostics.get(
                    "measurement_history_length", 0
                )
            ),
            "dynamic_obstacle_tracker_imm_initialized": bool(
                tracker_diagnostics.get("imm_initialized", False)
            ),
            "dynamic_obstacle_tracker_imm_initialized_track_count": int(
                tracker_diagnostics.get(
                    "imm_initialized_track_count",
                    int(bool(tracker_diagnostics.get(
                        "imm_initialized", False
                    ))),
                )
            ),
            "dynamic_obstacle_tracker_associated": bool(
                tracker_diagnostics.get("associated", False)
            ),
            "dynamic_obstacle_tracker_association_distance_m": float(
                tracker_diagnostics.get("association_distance_m", 0.0)
                or 0.0
            ),
            "dynamic_obstacle_tracker_measurement_x": float(
                tracker_diagnostics.get("measurement_x", 0.0) or 0.0
            ),
            "dynamic_obstacle_tracker_measurement_y": float(
                tracker_diagnostics.get("measurement_y", 0.0) or 0.0
            ),
            "dynamic_obstacle_tracker_support_beams": int(
                tracker_diagnostics.get("support_beams", 0)
            ),
            "dynamic_obstacle_tracker_unobserved_duration_s": float(
                tracker_diagnostics.get("unobserved_duration_s", 0.0)
                or 0.0
            ),
            "dynamic_obstacle_tracker_forecast_valid": bool(
                tracker_diagnostics.get("forecast_valid", False)
            ),
            "dynamic_obstacle_tracker_forecast_unavailable_reason": str(
                tracker_diagnostics.get(
                    "forecast_unavailable_reason", "unknown"
                )
            ),
            "dynamic_obstacle_tracker_forecast_availability": float(
                tracker_diagnostics.get("forecast_availability", 0.0)
            ),
            "dynamic_obstacle_tracker_innovation_nis": float(
                tracker_diagnostics.get("innovation_nis", 0.0) or 0.0
            ),
            "dynamic_obstacle_tracker_change_triggered": bool(
                tracker_diagnostics.get("change_triggered", False)
            ),
            "dynamic_obstacle_tracker_dropout_guard_triggered": bool(
                tracker_diagnostics.get(
                    "dropout_guard_triggered", False
                )
            ),
            "dynamic_obstacle_tracker_recovery_active": bool(
                tracker_diagnostics.get("recovery_active", False)
            ),
        }
        diagnostics = {
            "optimizer": "paper_rl_driven",
            "paper_faithful_gate1": True,
            "paper_iterations": int(cfg.iterations),
            "paper_candidates_per_iteration": int(
                self.config.num_samples
            ),
            "paper_total_rollouts": int(
                self.config.num_samples * cfg.iterations
            ),
            "paper_guided_unique_sequences": int(guided_count),
            "paper_guided_reuses": int(guided_count * cfg.iterations),
            "paper_guided_generation_calls": 1,
            "supervised_maneuver_actor_enabled": bool(
                self.maneuver_proposal_policy is not None
            ),
            "supervised_proposal_count": int(supervised_count),
            "supervised_proposal_reuses": int(
                supervised_count * cfg.iterations
            ),
            "supervised_risk_feasible_count": int(
                total_supervised_risk_feasible
            ),
            "supervised_risk_feasible_fraction": float(
                total_supervised_risk_feasible
                / total_supervised_opportunities
                if total_supervised_opportunities else 0.0
            ),
            "supervised_elite_count": int(total_supervised_elites),
            "supervised_elite_weight_sum_final_iteration": float(
                supervised_elite_weight_sum
            ),
            "supervised_selected_count": int(
                supervised_selected_count
            ),
            "supervised_selected_head_index": int(
                supervised_selected_head
            ),
            "supervised_replaced_guided_count": int(supervised_count),
            "supervised_added_rollout_count": 0,
            "paper_gaussian_samples_per_iteration": int(gaussian_count),
            "paper_guided_elite_count": int(total_guided_elites),
            "paper_gaussian_elite_count": int(total_gaussian_elites),
            "paper_guided_opportunity_count": int(
                total_guided_opportunities
            ),
            "paper_gaussian_opportunity_count": int(
                total_gaussian_opportunities
            ),
            "paper_guided_cost_observed": guided_cost_observed,
            "paper_gaussian_cost_observed": gaussian_cost_observed,
            "paper_guided_cost_min": guided_cost_min,
            "paper_gaussian_cost_min": gaussian_cost_min,
            "paper_guided_cost_mean": guided_cost_mean,
            "paper_gaussian_cost_mean": gaussian_cost_mean,
            "paper_guided_cost_p50": guided_cost_p50,
            "paper_gaussian_cost_p50": gaussian_cost_p50,
            "paper_guided_minus_gaussian_cost_min": (
                guided_cost_min - gaussian_cost_min
                if guided_cost_observed and gaussian_cost_observed else 0.0
            ),
            "paper_guided_minus_gaussian_cost_mean": (
                guided_cost_mean - gaussian_cost_mean
                if guided_cost_observed and gaussian_cost_observed else 0.0
            ),
            "paper_guided_feasible_fraction": float(
                guided_feasible_count / total_guided_opportunities
                if total_guided_opportunities else 0.0
            ),
            "paper_guided_boundary_feasible_fraction": float(
                guided_boundary_feasible_count
                / total_guided_opportunities
                if total_guided_opportunities else 0.0
            ),
            "paper_guided_static_feasible_fraction": float(
                guided_static_feasible_count
                / total_guided_opportunities
                if total_guided_opportunities else 0.0
            ),
            "paper_guided_risk_feasible_fraction": float(
                guided_risk_feasible_count
                / total_guided_opportunities
                if total_guided_opportunities else 0.0
            ),
            "paper_guided_boundary_first_failure_step_mean": float(
                np.mean(guided_boundary_first_failure_steps)
                if guided_boundary_first_failure_steps else -1.0
            ),
            "paper_guided_static_first_failure_step_mean": float(
                np.mean(guided_static_first_failure_steps)
                if guided_static_first_failure_steps else -1.0
            ),
            "paper_guided_risk_first_failure_step_mean": float(
                np.mean(guided_risk_first_failure_steps)
                if guided_risk_first_failure_steps else -1.0
            ),
            "paper_gaussian_feasible_fraction": float(
                gaussian_feasible_count / total_gaussian_opportunities
                if total_gaussian_opportunities else 0.0
            ),
            "paper_actor_first_v": float(actor_mean[0, 0]),
            "paper_actor_first_omega": float(actor_mean[0, 1]),
            "paper_baseline_first_v": float(baseline_mean[0, 0]),
            "paper_baseline_first_omega": float(baseline_mean[0, 1]),
            "paper_actor_baseline_mean_abs_delta": float(
                np.mean(actor_baseline_abs_delta)
            ),
            "paper_actor_baseline_first_action_l2_delta": float(
                np.linalg.norm(actor_mean[0] - baseline_mean[0])
            ),
            **proposal_advantage_diagnostics,
            "paper_actor_mean_initialization": True,
            "paper_actor_covariance_initialization": True,
            "paper_same_cycle_guided_cost_filter_enabled": bool(
                cfg.same_cycle_guided_cost_filter
            ),
            "paper_same_cycle_guided_cost_filter_iterations": int(
                same_cycle_guided_filter_iterations
            ),
            "paper_same_cycle_guided_filtered_candidates": int(
                same_cycle_guided_filtered_candidates
            ),
            "paper_same_cycle_guided_relative_margin": float(
                cfg.same_cycle_guided_relative_margin
            ),
            "paper_actor_joint_batched": bool(joint_actor_batch),
            "paper_actor_rollout_dynamics": type(self.dynamics).__name__,
            "paper_candidate_rollout_dynamics": type(self.dynamics).__name__,
            "paper_guided_set_persistent": True,
            "paper_action_semantics": "physical_low_level_control",
            "cost_min": float(np.min(all_costs)),
            "cost_mean": float(np.mean(all_costs)),
            "cost_std": float(np.std(all_costs)),
            "effective_sample_size": effective_sample_size,
            "path_boundary_candidate_filter_enabled": hard_boundary_filter,
            "path_boundary_candidate_feasible_fraction": float(
                np.mean(boundary_feasible_fractions)
                if boundary_feasible_fractions else 1.0
            ),
            "path_boundary_candidate_feasible_fraction_min": float(
                np.min(boundary_feasible_fractions)
                if boundary_feasible_fractions else 1.0
            ),
            "path_boundary_no_feasible_candidates": bool(
                boundary_no_feasible_iterations > 0
            ),
            "path_boundary_no_feasible_iteration_fraction": float(
                boundary_no_feasible_iterations / cfg.iterations
                if hard_boundary_filter else 0.0
            ),
            "path_boundary_weighted_update_feasible": (
                boundary_weighted_update_feasible
            ),
            "path_boundary_final_min_margin": boundary_final_min_margin,
            "path_boundary_fallback_used": boundary_fallback_used,
            "path_boundary_fallback_candidate_index": (
                boundary_fallback_candidate_index
            ),
            "known_static_map_candidate_filter_enabled": (
                hard_static_filter
            ),
            "known_static_map_candidate_feasible_count": int(
                np.sum(static_last_feasible)
                if static_last_feasible is not None
                else self.config.num_samples
            ),
            "known_static_map_candidate_feasible_fraction": float(
                np.mean(static_feasible_fractions)
                if static_feasible_fractions else 1.0
            ),
            "known_static_map_candidate_feasible_fraction_min": float(
                np.min(static_feasible_fractions)
                if static_feasible_fractions else 1.0
            ),
            "known_static_map_candidate_min_clearance": float(
                np.nanmin(static_last_min_clearance)
                if static_last_min_clearance is not None else 0.0
            ),
            "known_static_map_no_feasible_candidates": bool(
                static_no_feasible_iterations > 0
            ),
            "known_static_map_no_feasible_iteration_fraction": float(
                static_no_feasible_iterations / cfg.iterations
                if hard_static_filter else 0.0
            ),
            "known_static_map_weighted_update_feasible": (
                static_weighted_update_feasible
            ),
            "known_static_map_fallback_used": static_fallback_used,
            "known_static_map_fallback_candidate_index": int(
                static_fallback_candidate_index
            ),
            "known_static_map_weighted_update_min_clearance": float(
                static_final_min_clearance
            ),
            "known_static_map_cost_enabled": bool(
                self.config.known_static_map_cost_enabled
            ),
            "known_static_map_obstacle_count": len(
                known_static_obstacles
            ),
            "known_static_map_minimum_clearance": float(
                static_final_min_clearance
            ),
            "probabilistic_reference_authority_enabled": bool(
                self.config.probabilistic_reference_authority_enabled
            ),
            "probabilistic_reference_risk_raw": float(
                reference_risk_raw
            ),
            "probabilistic_reference_risk_filtered": float(
                self._probabilistic_reference_risk_filtered
            ),
            "probabilistic_reference_authority": float(
                reference_authority
            ),
            "probabilistic_reference_progress_weight": float(
                self.config.probabilistic_reference_progress_weight
            ),
            "probabilistic_obstacle_candidate_filter_enabled": (
                risk_candidate_filter
            ),
            "probabilistic_obstacle_candidate_feasible_fraction_min": float(
                np.min(risk_feasible_fractions)
                if risk_feasible_fractions else 1.0
            ),
            "probabilistic_obstacle_no_feasible_iteration_fraction": float(
                risk_no_feasible_iterations / cfg.iterations
                if risk_candidate_filter else 0.0
            ),
            "probabilistic_obstacle_emergency_near_distance_triggered": bool(
                emergency_context.get("near_distance_triggered", False)
            ),
            "probabilistic_obstacle_emergency_critical_distance_triggered": bool(
                emergency_context.get(
                    "critical_distance_triggered", False
                )
            ),
            "probabilistic_obstacle_emergency_surface_range_m": float(
                emergency_context.get("surface_range_m", float("inf"))
            ),
            "probabilistic_obstacle_emergency_forecast_corroboration_enabled": bool(
                emergency_context.get(
                    "forecast_corroboration_enabled", False
                )
            ),
            "probabilistic_obstacle_emergency_forecast_corroborated": bool(
                emergency_context.get("forecast_corroborated", False)
            ),
            "probabilistic_obstacle_emergency_forecast_stop_maximum_probability": float(
                emergency_context.get(
                    "forecast_stop_maximum_probability", 0.0
                )
            ),
            "probabilistic_obstacle_counterflow_escape_applied": bool(
                emergency_context.get("counterflow_escape_applied", False)
            ),
            "probabilistic_obstacle_preferred_escape_direction_x": float(
                emergency_context.get(
                    "preferred_escape_direction_x", 0.0
                )
            ),
            "probabilistic_obstacle_preferred_escape_direction_y": float(
                emergency_context.get(
                    "preferred_escape_direction_y", 0.0
                )
            ),
            **probabilistic_risk_diagnostics,
            **online_tracker_diagnostics,
            **optimizer_diagnostics,
            "covariance_scale_mean": float(
                np.mean(np.sqrt(variance / base_variance))
            ),
            "reference_id": target.reference_id,
            "target_x": float(target.pose.x),
            "target_y": float(target.pose.y),
            "target_theta": float(target.pose.theta),
            "target_is_terminal": bool(target.is_terminal),
            "target_phase": str(target.phase),
            "prior": dict(prior.metadata),
            **terminal_action_diagnostics,
            **handover_diagnostics,
        }
        diagnostics.update(terminal_diagnostics)
        diagnostics.update(reliability_diagnostics)
        diagnostics.update(self._residual_policy_context_diagnostics(
            reliability_context.get("residual_context", np.empty((0, 0)))
        ))
        residual_authority = np.asarray(
            reliability_context.get("residual_authority", np.empty(0)),
            dtype=np.float64,
        )
        diagnostics.update({
            "residual_policy_authority_enabled": bool(
                residual_authority.size
            ),
            "residual_policy_authority_mean": float(
                np.mean(residual_authority)
                if residual_authority.size else 0.0
            ),
            "residual_policy_authority_max": float(
                np.max(residual_authority)
                if residual_authority.size else 0.0
            ),
        })
        for head in range(3):
            present = head < supervised_count
            diagnostics.update({
                f"supervised_head_{head}_source_id": (
                    f"supervised_head_{head}" if present else "disabled"
                ),
                f"supervised_head_{head}_insertion_index": (
                    int(supervised_start + head) if present else -1
                ),
                f"supervised_head_{head}_behavior_label": (
                    supervised_head_behavior[head]
                    if present else "disabled"
                ),
                f"supervised_head_{head}_proposal_count": int(present),
                f"supervised_head_{head}_risk_feasible_count": (
                    int(supervised_head_risk_feasible[head])
                    if present else 0
                ),
                f"supervised_head_{head}_elite_count": (
                    int(supervised_head_elites[head])
                    if present else 0
                ),
                f"supervised_head_{head}_selected_count": (
                    int(supervised_head_selected[head])
                    if present else 0
                ),
            })
        return action, sequence, trajectory, diagnostics
