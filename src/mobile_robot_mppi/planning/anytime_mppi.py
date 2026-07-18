"""Contextual-bandit anytime MPPI with nested sample reuse."""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from mobile_robot_mppi.planning.mppi import MppiController
from mobile_robot_mppi.rl.budget_bandit import PrimalDualBudgetBandit
from mobile_robot_mppi.rl.contextual_bandit import (
    REFERENCE_GEOMETRY_FEATURE_NAMES,
    polyline_geometry_features,
    polyline_window,
)


@dataclass(frozen=True)
class AnytimeMppiConfig:
    base_samples: int = 50
    local_route_window_m: float = 1.25
    bandit_checkpoint: str = ""
    decision_interval_steps: int = 1

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        values = dict(values or {})
        return cls(
            base_samples=int(values.get("base_samples", 50)),
            local_route_window_m=float(
                values.get("local_route_window_m", 1.25)
            ),
            bandit_checkpoint=str(values.get("checkpoint", "")),
            decision_interval_steps=int(
                values.get("decision_interval_steps", 1)
            ),
        )

    def validate(self, maximum_samples):
        if not 2 <= self.base_samples < int(maximum_samples):
            raise ValueError(
                "anytime base_samples must lie in [2, num_samples)"
            )
        if (
            not np.isfinite(self.local_route_window_m)
            or self.local_route_window_m <= 0.0
        ):
            raise ValueError("local_route_window_m must be finite and positive")
        if not self.bandit_checkpoint:
            raise ValueError("anytime MPPI requires a bandit checkpoint")
        if self.decision_interval_steps <= 0:
            raise ValueError("decision_interval_steps must be positive")


def load_budget_bandit(path):
    checkpoint = Path(path)
    with checkpoint.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    state = payload.get("bandit", payload)
    bandit = PrimalDualBudgetBandit.from_state_dict(state)
    if "deployment_dual_price" in payload:
        bandit.dual_price = float(payload["deployment_dual_price"])
    return bandit


class AnytimeMppiController(MppiController):
    """Choose STOP/ADD after evaluating a byte-identical first sample batch."""

    def __init__(self, *args, anytime_config=None, budget_bandit=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.anytime_config = (
            anytime_config
            if isinstance(anytime_config, AnytimeMppiConfig)
            else AnytimeMppiConfig.from_mapping(anytime_config or {})
        )
        self.anytime_config.validate(self.config.num_samples)
        self.budget_bandit = (
            budget_bandit
            if budget_bandit is not None
            else load_budget_bandit(self.anytime_config.bandit_checkpoint)
        )
        expected = {
            "ess_fraction",
            "weight_entropy_fraction",
            "maximum_weight",
            "split_control_disagreement",
            "weighted_action_dispersion",
            "normalized_cost_spread",
            "sample_saturation_fraction",
            "weighted_perturbation_norm",
            "residual_reliability_alpha",
            "residual_reliability_last_relative_improvement",
        } | {
            "local_" + name
            for name in REFERENCE_GEOMETRY_FEATURE_NAMES[1:]
        }
        if set(self.budget_bandit.feature_names) != expected:
            raise ValueError("budget bandit feature schema is incompatible")
        self._latched_decision = None
        self._latched_features = None
        self._decision_age_steps = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Deployment is frozen: decisions never update model or dual state.
        self._latched_decision = None
        self._latched_features = None
        self._decision_age_steps = 0

    def _weights(self, prior, candidates, base_costs, context=None):
        result = self.importance_weights(
            prior.mean, candidates, base_costs, prior.covariance
        )
        perturbations = candidates - prior.mean[None, :, :]
        sequence = prior.mean + np.sum(
            result.weights[:, None, None] * perturbations, axis=0
        )
        if context is not None:
            sequence, _action, _alignment_omega = self._finalize_sequence(
                sequence, context
            )
        return sequence, result

    def _finalize_sequence(self, sequence, context):
        """Apply the same feasibility map used by the deployed MPPI action.

        The frozen budget bandit was fitted from counterfactual K50/K100
        records whose disagreement and perturbation features were computed
        *after* action bounds, terminal constraints, and the first-command
        slew limit.  Keeping that map here prevents an otherwise subtle
        train/deployment feature shift.
        """

        values = np.clip(
            np.asarray(sequence, dtype=np.float64).copy(),
            self.action_spec.lower,
            self.action_spec.upper,
        )
        v_index = None
        if context["speed_limit"] or context["heading_gate"]:
            v_index = self.action_spec.index("v_cmd")
        if context["speed_limit"]:
            values[:, v_index] = np.minimum(
                values[:, v_index],
                self.config.terminal_translation_speed_limit,
            )
        if context["heading_gate"]:
            values[0, v_index] *= context["translation_scale"]
        alignment_omega = 0.0
        if context["alignment_active"]:
            omega_index = self.action_spec.index("omega_cmd")
            alignment_omega = (
                float(self.config.terminal_alignment_yaw_gain)
                * context["bearing_error"]
            )
            values[0, omega_index] = alignment_omega
        action = self.action_spec.clip(
            values[0], self.previous_action, self.config.dt
        )
        if context["speed_limit"]:
            action[v_index] = min(
                action[v_index],
                self.config.terminal_translation_speed_limit,
            )
        if context["heading_gate"]:
            action[v_index] *= context["translation_scale"]
        values[0] = action
        return values, action, float(alignment_omega)

    def _first_batch_features(
        self, prior, candidates, base_costs, reference, context
    ):
        sequence, weighting = self._weights(
            prior, candidates, base_costs, context
        )
        weights = weighting.weights
        count = len(candidates)
        entropy = -float(np.sum(
            weights * np.log(np.maximum(weights, 1e-300))
        ))
        action_scale = np.maximum(
            self.action_spec.upper - self.action_spec.lower, 1e-9
        )
        first_actions = candidates[:, 0, :]
        mean_action = np.sum(weights[:, None] * first_actions, axis=0)
        centered = (
            first_actions - mean_action[None, :]
        ) / action_scale[None, :]
        dispersion = float(np.sum(weights[:, None] * centered ** 2))
        split_actions = []
        for indices in (np.arange(0, count, 2), np.arange(1, count, 2)):
            split_sequence, _ = self._weights(
                prior, candidates[indices], base_costs[indices], context
            )
            split_actions.append(split_sequence[0])
        split_disagreement = float(np.linalg.norm(
            (split_actions[0] - split_actions[1]) / action_scale
        ))
        q10, q50, q90 = np.quantile(base_costs, (0.10, 0.50, 0.90))
        residual = getattr(self.dynamics, "residual", None)
        reliability = getattr(residual, "diagnostics", None)
        reliability_values = reliability() if callable(reliability) else {}
        result = {
            "ess_fraction": float(weighting.effective_sample_size / count),
            "weight_entropy_fraction": float(entropy / np.log(count)),
            "maximum_weight": float(np.max(weights)),
            "split_control_disagreement": split_disagreement,
            "weighted_action_dispersion": dispersion,
            "normalized_cost_spread": float(
                (q90 - q10) / (abs(q50) + 1e-9)
            ),
            "sample_saturation_fraction": float(np.mean(
                (candidates <= self.action_spec.lower[None, None, :])
                | (candidates >= self.action_spec.upper[None, None, :])
            )),
            "weighted_perturbation_norm": float(
                np.linalg.norm(sequence - prior.mean)
            ),
            "residual_reliability_alpha": float(
                reliability_values.get("residual_reliability_alpha", 1.0)
            ),
            "residual_reliability_last_relative_improvement": float(
                reliability_values.get(
                    "residual_reliability_last_relative_improvement", 0.0
                )
            ),
        }
        if not hasattr(reference, "points") or not hasattr(reference, "progress"):
            raise ValueError("anytime bandit currently requires a polyline reference")
        local = polyline_geometry_features(polyline_window(
            reference.points,
            reference.progress,
            self.anytime_config.local_route_window_m,
        ))
        result.update({
            "local_" + name: float(value)
            for name, value in zip(
                REFERENCE_GEOMETRY_FEATURE_NAMES[1:], local[1:]
            )
        })
        if not np.isfinite(list(result.values())).all():
            raise FloatingPointError("anytime diagnostics contain NaN or Inf")
        return result

    def _terminal_context(self, state, target):
        heading_gate = bool(
            self.config.terminal_translation_heading_gate_rad is not None
            and target.phase in ("terminal_approach", "terminal")
            and "v_cmd" in self.action_spec.names
            and "theta" in self.state_spec.names
        )
        bearing_error = 0.0
        translation_scale = 1.0
        if heading_gate:
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
                cosine = float(np.cos(gate))
                if abs(bearing_error) >= gate:
                    translation_scale = 0.0
                else:
                    translation_scale = max(
                        0.0,
                        (float(np.cos(bearing_error)) - cosine)
                        / max(1.0 - cosine, 1e-12),
                    )
        speed_limit = bool(
            self.config.terminal_translation_speed_limit is not None
            and target.phase in ("terminal_approach", "terminal")
            and "v_cmd" in self.action_spec.names
        )
        return {
            "heading_gate": heading_gate,
            "speed_limit": speed_limit,
            "bearing_error": bearing_error,
            "translation_scale": translation_scale,
            "alignment_active": bool(
                heading_gate
                and self.config.terminal_alignment_yaw_gain is not None
                and "omega_cmd" in self.action_spec.names
            ),
        }

    def _constrain_candidates(self, controls, context):
        values = np.asarray(controls, dtype=np.float64).copy()
        if context["speed_limit"] or context["heading_gate"]:
            v_index = self.action_spec.index("v_cmd")
            if context["speed_limit"]:
                values[..., v_index] = np.minimum(
                    values[..., v_index],
                    self.config.terminal_translation_speed_limit,
                )
            if context["heading_gate"]:
                values[:, 0, v_index] *= context["translation_scale"]
        return values

    def _solve_plan(
        self, state, prior, target, obstacles, rng, observation=None,
        reference=None,
    ):
        del observation
        profiling = bool(self.config.profile_components)
        solve_started = time.perf_counter()
        last_mark = solve_started
        profile = {}

        def mark(name):
            nonlocal last_mark
            if profiling:
                now = time.perf_counter()
                key = "profile_mppi_" + name + "_ms"
                profile[key] = profile.get(key, 0.0) + 1000.0 * (
                    now - last_mark
                )
                last_mark = now

        if reference is None:
            raise ValueError("anytime MPPI requires the task reference")
        maximum = int(self.config.num_samples)
        base = int(self.anytime_config.base_samples)
        context = self._terminal_context(state, target)
        samples = self._constrain_candidates(
            self._sample(prior, rng=rng), context
        )
        mark("sampling")
        refresh_decision = bool(
            self._latched_decision is None
            or self._decision_age_steps
            >= self.anytime_config.decision_interval_steps
        )
        first_trajectories = None
        first_costs = None
        if refresh_decision:
            first_trajectories = self.rollout(state, samples[:base])
            mark("batch_rollout")
            first_costs = self._cost(
                first_trajectories, samples[:base], target, obstacles
            )
            mark("cost")
            features = self._first_batch_features(
                prior, samples[:base], first_costs, reference, context
            )
            decision = self.budget_bandit.decide(features, explore=False)
            self._latched_decision = decision
            self._latched_features = dict(features)
            self._decision_age_steps = 1
            mark("weighting_update")
        else:
            decision = self._latched_decision
            features = dict(self._latched_features)
            self._decision_age_steps += 1
        selected_count = maximum if decision.add_samples else base
        if not refresh_decision and decision.add_samples:
            # A latched ADD needs no intermediate STOP counterfactual.  Evaluate
            # Kmax in one vectorized call to avoid paying two ICODE batch
            # launch overheads at every 10 Hz controller step.
            trajectories = self.rollout(state, samples)
            mark("batch_rollout")
            base_costs = self._cost(
                trajectories, samples, target, obstacles
            )
            mark("cost")
        elif decision.add_samples:
            remaining_trajectories = self.rollout(state, samples[base:])
            mark("batch_rollout")
            remaining_costs = self._cost(
                remaining_trajectories, samples[base:], target, obstacles
            )
            mark("cost")
            trajectories = np.concatenate(
                (first_trajectories, remaining_trajectories), axis=0
            )
            base_costs = np.concatenate((first_costs, remaining_costs))
        else:
            samples = samples[:base]
            if first_trajectories is None:
                trajectories = self.rollout(state, samples)
                mark("batch_rollout")
                base_costs = self._cost(
                    trajectories, samples, target, obstacles
                )
                mark("cost")
            else:
                trajectories = first_trajectories
                base_costs = first_costs
        sequence, weighting = self._weights(prior, samples, base_costs)
        sequence, action, alignment_omega = self._finalize_sequence(
            sequence, context
        )
        mark("weighting_update")
        alignment_active = bool(context["alignment_active"])
        updated_trajectory = self.rollout(state, sequence)[0]
        mark("final_rollout")
        adjusted = weighting.adjusted_costs
        diagnostics = {
            "optimizer": "anytime_bandit",
            "anytime_base_samples": base,
            "anytime_maximum_samples": maximum,
            "anytime_selected_samples": selected_count,
            "anytime_add_samples": bool(decision.add_samples),
            "anytime_predicted_advantage": float(decision.predicted_advantage),
            "anytime_confidence_width": float(decision.confidence_width),
            "anytime_dual_price": float(decision.dual_price),
            "anytime_score": float(decision.score),
            "anytime_decision_refreshed": bool(refresh_decision),
            "anytime_decision_age_steps": int(self._decision_age_steps),
            "anytime_decision_interval_steps": int(
                self.anytime_config.decision_interval_steps
            ),
            "cost_min": float(np.min(adjusted)),
            "cost_mean": float(np.mean(adjusted)),
            "cost_std": float(np.std(adjusted)),
            "cost_q10": float(np.quantile(adjusted, 0.10)),
            "cost_q50": float(np.quantile(adjusted, 0.50)),
            "cost_q90": float(np.quantile(adjusted, 0.90)),
            "effective_sample_size": weighting.effective_sample_size,
            "effective_sample_fraction": float(
                weighting.effective_sample_size / selected_count
            ),
            "importance_sampling_correction": bool(
                self.config.importance_sampling_correction
            ),
            "importance_cost_mean": float(np.mean(
                adjusted - base_costs
            )),
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
            "terminal_speed_limit_active": context["speed_limit"],
            "terminal_heading_gate_active": context["heading_gate"],
            "terminal_bearing_error": context["bearing_error"],
            "terminal_translation_scale": context["translation_scale"],
            "terminal_alignment_active": alignment_active,
            "terminal_alignment_omega": alignment_omega,
            "prior": dict(prior.metadata),
        }
        diagnostics.update({
            "anytime_feature_" + name: float(value)
            for name, value in features.items()
        })
        residual = getattr(self.dynamics, "residual", None)
        confidence = getattr(residual, "confidence", None)
        if callable(confidence):
            support = np.asarray(confidence(
                updated_trajectory[:-1],
                self._prediction_controls(sequence[None, :, :])[0],
            ), dtype=np.float64).reshape(-1)
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
        if profiling:
            diagnostics.update(profile)
            diagnostics["profile_mppi_solve_total_ms"] = 1000.0 * (
                time.perf_counter() - solve_started
            )
        return action, sequence, updated_trajectory, diagnostics


__all__ = [
    "AnytimeMppiConfig",
    "AnytimeMppiController",
    "load_budget_bandit",
]
