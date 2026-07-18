"""RL-Driven MPPI with auditable hybrid candidate sources.

This controller is opt-in.  The legacy :class:`MppiController` remains the
default, so existing MuJoCo, ROS, memory, perception and safety behavior is
unchanged unless ``planner.optimizer`` is explicitly set to ``rl_driven``.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from mobile_robot_mppi.planning.mppi import MppiController


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
