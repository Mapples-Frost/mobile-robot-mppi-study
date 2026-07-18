"""RL-Driven MPPI with auditable hybrid candidate sources.

This controller is opt-in.  The legacy :class:`MppiController` remains the
default, so existing MuJoCo, ROS, memory, perception and safety behavior is
unchanged unless ``planner.optimizer`` is explicitly set to ``rl_driven``.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from mobile_robot_mppi.planning.mppi import MppiController, integrate_batch
from mobile_robot_mppi.rl.reliability import HybridSamplingReliability


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
        heading_gate_active = bool(
            self.config.terminal_translation_heading_gate_rad is not None
            and target.phase in ("terminal_approach", "terminal")
            and "v_cmd" in self.action_spec.names
            and "theta" in self.state_spec.names
        )
        bearing_error = 0.0
        translation_scale = 1.0
        if heading_gate_active:
            theta = float(state[self.state_spec.index("theta")])
            x_index, y_index = self.state_spec.position_indices
            dx = float(target.pose.x - state[x_index])
            dy = float(target.pose.y - state[y_index])
            if np.hypot(dx, dy) > 1e-12:
                desired = float(np.arctan2(dy, dx))
                bearing_error = float(np.arctan2(
                    np.sin(desired - theta), np.cos(desired - theta)
                ))
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
            "terminal_translation_scale": translation_scale,
            "v_index": v_index,
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
        del reference
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
            costs = self._cost(trajectories, samples, target, obstacles)
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
        action = self.action_spec.clip(
            sequence[0], self.previous_action, self.config.dt
        )
        v_index = constraints["v_index"]
        if constraints["terminal_speed_limit_active"]:
            action[v_index] = min(
                action[v_index], self.config.terminal_translation_speed_limit
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
            "terminal_translation_scale": float(
                constraints["terminal_translation_scale"]
            ),
            "terminal_alignment_active": alignment_active,
            "terminal_alignment_yaw_gain": (
                0.0
                if self.config.terminal_alignment_yaw_gain is None
                else float(self.config.terminal_alignment_yaw_gain)
            ),
            "terminal_alignment_omega": float(alignment_omega),
            "prior": dict(prior.metadata),
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
        if self.terminal_critic_source not in ("online", "target"):
            raise ValueError("terminal_critic_source must be online or target")
        HybridSamplingReliability(self.reliability or {})
        guided = int(round(float(samples) * self.guided_fraction))
        if guided >= int(samples):
            raise ValueError("at least one current-Gaussian sample is required")


class PaperRLDrivenMppiController(RLDrivenMppiController):
    """Faithful low-level Actor/critic integration used by Gate 1.

    The Actor is rolled out autoregressively through the controller's declared
    prediction dynamics. Its mean and stochastic spread initialize MPPI.
    Stochastic Actor sequences are generated exactly once per control decision
    and then reused in every refinement iteration, matching the persistent
    guided set in RL-Driven MPPI. The final command still comes from MPPI and
    remains subject to the unchanged external safety chain.
    """

    def __init__(self, *args, paper_rl_driven_config=None, **kwargs):
        MppiController.__init__(self, *args, **kwargs)
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
        self._applied_guided_fraction = float(
            self.paper_rl_driven_config.guided_fraction
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
        self._applied_guided_fraction = float(
            self.paper_rl_driven_config.guided_fraction
        )

    def _delayed_control(self, current, preceding):
        fraction = float(self.config.command_delay_s / self.config.dt)
        return fraction * preceding + (1.0 - fraction) * current

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
        actor_ood_scores = np.zeros(
            self.config.horizon, dtype=np.float64
        )
        support_evaluator = getattr(
            self.sampling_prior, "support_ood_scores", None
        )
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
            "controls": means.copy(),
            "actor_ood_scores": actor_ood_scores,
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
        actor_ood_scores = np.zeros(
            self.config.horizon, dtype=np.float64
        )
        support_evaluator = getattr(
            self.sampling_prior, "support_ood_scores", None
        )
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
            "controls": means.copy(),
            "actor_ood_scores": actor_ood_scores,
        }

    def _residual_for_reliability(self):
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
        self, trajectories, controls, observation, reference
    ):
        cfg = self.paper_rl_driven_config
        if cfg.terminal_value_weight <= 0.0:
            return np.zeros(controls.shape[0], dtype=np.float64), {
                "terminal_value_enabled": False,
                "terminal_value_weight": 0.0,
            }
        returns, diagnostics = self.sampling_prior.terminal_value(
            trajectories[:, -1, :],
            controls[:, -1, :],
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
        cfg = self.paper_rl_driven_config
        applied_guided_fraction = (
            self._applied_guided_fraction
            if self.hybrid_sampling_reliability.config.enabled
            else cfg.guided_fraction
        )
        guided_count = int(round(
            self.config.num_samples * applied_guided_fraction
        ))
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
        }
        if self.hybrid_sampling_reliability.config.enabled:
            reliability_diagnostics = (
                self.hybrid_sampling_reliability.evaluate(
                    self._residual_for_reliability(),
                    reliability_context["states"],
                    reliability_context["controls"],
                    reliability_context["actor_ood_scores"],
                )
            )
            next_fraction = float(
                reliability_diagnostics["guided_fraction"]
            )
            self._applied_guided_fraction = next_fraction
            reliability_diagnostics.update({
                "reliability_hss_enabled": True,
                "reliability_guided_fraction_applied": float(
                    applied_guided_fraction
                ),
                "reliability_guided_fraction_next": next_fraction,
                # Authority estimated during the Actor mean rollout is applied
                # on the next control cycle so stochastic guided trajectories
                # remain jointly batched and no second Actor pass is added.
                "reliability_causal_lag_steps": 1,
            })
        base_variance = np.broadcast_to(
            np.asarray(self.config.noise_sigma, dtype=np.float64)[None, :] ** 2,
            mean.shape,
        ).copy()
        variance = np.clip(
            actor_variance,
            base_variance * cfg.covariance_min_scale ** 2,
            base_variance * cfg.covariance_max_scale ** 2,
        )
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
        costs_by_iteration = []
        terminal_diagnostics = {}
        constraints = None
        final_weights = None
        for _ in range(cfg.iterations):
            gaussian = self._gaussian_samples(
                mean, variance, gaussian_count, rng
            )
            samples = np.concatenate((guided, gaussian), axis=0)
            labels = np.concatenate((
                np.ones(guided_count, dtype=np.int8),
                np.zeros(gaussian_count, dtype=np.int8),
            ))
            samples, constraints = self._terminal_constraints(
                state, target, samples
            )
            trajectories = self.rollout(state, samples)
            running = self._cost(
                trajectories, samples, target, obstacles
            )
            terminal, terminal_diagnostics = self._paper_terminal_cost(
                trajectories, samples, observation, reference
            )
            costs = np.asarray(running, dtype=np.float64) + terminal
            if not np.isfinite(costs).all():
                raise FloatingPointError(
                    "paper RL-Driven MPPI cost contains NaN or Inf"
                )
            elite_count = max(
                2,
                min(
                    self.config.num_samples,
                    int(np.ceil(
                        cfg.elite_fraction * self.config.num_samples
                    )),
                ),
            )
            elite_indices = np.argsort(costs, kind="stable")[:elite_count]
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
                np.sum(labels[elite_indices] == 1)
            )
            total_gaussian_elites += int(
                np.sum(labels[elite_indices] == 0)
            )
            costs_by_iteration.append(costs)

        sequence = np.clip(
            mean, self.action_spec.lower, self.action_spec.upper
        )
        sequence, constraints = self._terminal_constraints(
            state, target, sequence[None, :, :]
        )
        sequence = sequence[0]
        action = self.action_spec.clip(
            sequence[0], self.previous_action, self.config.dt
        )
        sequence[0] = action
        trajectory = self.rollout(state, sequence)[0]
        all_costs = np.concatenate(costs_by_iteration)
        effective_sample_size = float(
            1.0 / np.sum(np.asarray(final_weights) ** 2)
        )
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
            "paper_gaussian_samples_per_iteration": int(gaussian_count),
            "paper_guided_elite_count": int(total_guided_elites),
            "paper_gaussian_elite_count": int(total_gaussian_elites),
            "paper_actor_mean_initialization": True,
            "paper_actor_covariance_initialization": True,
            "paper_actor_joint_batched": bool(joint_actor_batch),
            "paper_actor_rollout_dynamics": type(self.dynamics).__name__,
            "paper_candidate_rollout_dynamics": type(self.dynamics).__name__,
            "paper_guided_set_persistent": True,
            "paper_action_semantics": "physical_low_level_control",
            "cost_min": float(np.min(all_costs)),
            "cost_mean": float(np.mean(all_costs)),
            "cost_std": float(np.std(all_costs)),
            "effective_sample_size": effective_sample_size,
            "covariance_scale_mean": float(
                np.mean(np.sqrt(variance / base_variance))
            ),
            "terminal_speed_limit_active": bool(
                constraints["terminal_speed_limit_active"]
            ),
            "terminal_heading_gate_active": bool(
                constraints["terminal_heading_gate_active"]
            ),
            "prior": dict(prior.metadata),
        }
        diagnostics.update(terminal_diagnostics)
        diagnostics.update(reliability_diagnostics)
        return action, sequence, trajectory, diagnostics
