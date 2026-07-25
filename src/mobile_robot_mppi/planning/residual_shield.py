"""Nominal-noninterference shield for residual-dynamics MPPI."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time

import numpy as np

from mobile_robot_mppi.core.types import (
    ControlCommand,
    PlanResult,
    SafetyDecision,
)


@dataclass(frozen=True)
class ResidualSafetyShieldConfig:
    parallel_planning_enabled: bool = False
    maximum_nominal_risk_increase: float = 0.0
    maximum_position_deviation_m: float = 0.10
    maximum_nominal_progress_regression_m: float = 0.02
    deterministic_clearance_fallback_enabled: bool = False
    maximum_nominal_clearance_regression_m: float = 0.0
    require_residual_hard_safe: bool = True
    require_nominal_hard_safe: bool = True
    action_authority_names: tuple = ()

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        config = cls(
            parallel_planning_enabled=bool(
                values.get("parallel_planning_enabled", False)
            ),
            maximum_nominal_risk_increase=float(
                values.get("maximum_nominal_risk_increase", 0.0)
            ),
            maximum_position_deviation_m=float(
                values.get("maximum_position_deviation_m", 0.10)
            ),
            maximum_nominal_progress_regression_m=float(
                values.get(
                    "maximum_nominal_progress_regression_m", 0.02
                )
            ),
            deterministic_clearance_fallback_enabled=bool(
                values.get(
                    "deterministic_clearance_fallback_enabled", False
                )
            ),
            maximum_nominal_clearance_regression_m=float(
                values.get(
                    "maximum_nominal_clearance_regression_m", 0.0
                )
            ),
            require_residual_hard_safe=bool(
                values.get("require_residual_hard_safe", True)
            ),
            require_nominal_hard_safe=bool(
                values.get("require_nominal_hard_safe", True)
            ),
            action_authority_names=tuple(
                str(value)
                for value in values.get("action_authority_names", ())
            ),
        )
        numeric = (
            config.maximum_nominal_risk_increase,
            config.maximum_position_deviation_m,
            config.maximum_nominal_progress_regression_m,
            config.maximum_nominal_clearance_regression_m,
        )
        if (
            not np.isfinite(numeric).all()
            or config.maximum_nominal_risk_increase < 0.0
            or config.maximum_position_deviation_m <= 0.0
            or config.maximum_nominal_progress_regression_m < 0.0
            or config.maximum_nominal_clearance_regression_m < 0.0
        ):
            raise ValueError(
                "residual safety shield thresholds must be finite and "
                "nonnegative, with positive position deviation"
            )
        if (
            len(set(config.action_authority_names))
            != len(config.action_authority_names)
            or any(not name for name in config.action_authority_names)
        ):
            raise ValueError(
                "residual safety shield action authority names must be "
                "unique and nonempty"
            )
        return config


class ResidualSafetyShieldController:
    """Accept residual MPPI only when it cannot degrade nominal certificates.

    The nominal controller is a complete concurrent baseline. The residual
    controller's selected sequence is replayed through nominal dynamics and is
    accepted only when both model views are hard-safe, its nominal-view risk
    does not exceed the nominal controller's own risk, its nominal-view
    progress is noninferior, and residual/nominal predicted positions remain
    within a fixed tube. Otherwise the exact nominal plan is returned.
    """

    def __init__(
        self,
        residual_controller,
        nominal_controller,
        shield_config,
    ):
        self.residual_controller = residual_controller
        self.nominal_controller = nominal_controller
        self.shield_config = (
            shield_config
            if isinstance(shield_config, ResidualSafetyShieldConfig)
            else ResidualSafetyShieldConfig.from_mapping(shield_config)
        )
        self.config = residual_controller.config
        self.state_spec = residual_controller.state_spec
        self.action_spec = residual_controller.action_spec
        self.dynamics = residual_controller.dynamics
        self.sampling_prior = residual_controller.sampling_prior
        if (
            nominal_controller.state_spec.names != self.state_spec.names
            or nominal_controller.action_spec.names != self.action_spec.names
            or nominal_controller.config != self.config
        ):
            raise ValueError(
                "residual shield controllers must share state, action and "
                "MPPI configuration contracts"
            )
        if (
            not self.config.probabilistic_obstacle_risk_enabled
            and not self.shield_config.deterministic_clearance_fallback_enabled
        ):
            raise ValueError(
                "residual safety shield requires probabilistic obstacle risk "
                "or the explicit deterministic-clearance fallback"
            )
        unknown_authority = [
            name
            for name in self.shield_config.action_authority_names
            if name not in self.action_spec.names
        ]
        if unknown_authority:
            raise ValueError(
                "residual safety shield action names are unavailable: %s"
                % unknown_authority
            )
        self._authority_indices = tuple(
            self.action_spec.index(name)
            for name in self.shield_config.action_authority_names
        )
        self._last_nominal_plan = None
        self._last_residual_plan = None
        self._planner_executor = (
            ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="residual-shield-planner",
            )
            if self.shield_config.parallel_planning_enabled
            else None
        )

    @property
    def previous_action(self):
        return self.residual_controller.previous_action

    @previous_action.setter
    def previous_action(self, value):
        values = np.asarray(value, dtype=np.float64).copy()
        self.residual_controller.previous_action = values.copy()
        self.nominal_controller.previous_action = values.copy()

    def reset(self, seed=None):
        self.residual_controller.reset(seed=seed)
        self.nominal_controller.reset(seed=seed)
        self._last_nominal_plan = None
        self._last_residual_plan = None

    def close(self):
        if self._planner_executor is not None:
            self._planner_executor.shutdown(wait=True)
            self._planner_executor = None

    def _plan_pair(self, method_name, observation, reference):
        if self._planner_executor is None:
            return (
                getattr(self.nominal_controller, method_name)(
                    observation, reference
                ),
                getattr(self.residual_controller, method_name)(
                    observation, reference
                ),
            )
        nominal_future = self._planner_executor.submit(
            getattr(self.nominal_controller, method_name),
            observation,
            reference,
        )
        residual_future = self._planner_executor.submit(
            getattr(self.residual_controller, method_name),
            observation,
            reference,
        )
        try:
            return nominal_future.result(), residual_future.result()
        except BaseException:
            nominal_future.cancel()
            residual_future.cancel()
            raise

    def state_from_observation(self, observation):
        return self.residual_controller.state_from_observation(observation)

    def _risk(self, controller, trajectory, observation):
        forecasts = tuple(
            observation.auxiliary.get(
                self.config.probabilistic_obstacle_forecast_key, ()
            )
        )
        if not forecasts:
            return None
        return controller._probabilistic_collision_risk(
            np.asarray(trajectory, dtype=np.float64)[None, ...],
            forecasts,
        )

    def _minimum_local_clearance(self, trajectory, observation):
        obstacles = tuple(observation.local_obstacles or ())
        if not obstacles:
            return float("inf")
        values = np.asarray(obstacles, dtype=np.float64)
        centers = values[:, :2]
        radii = (
            values[:, 2]
            if values.shape[1] >= 3
            else np.zeros(values.shape[0], dtype=np.float64)
        )
        positions = np.asarray(trajectory, dtype=np.float64)[
            :, self.state_spec.position_indices
        ]
        distances = np.linalg.norm(
            positions[:, None, :] - centers[None, :, :], axis=2
        )
        clearances = (
            distances - float(self.config.robot_radius) - radii[None, :]
        )
        return float(np.min(clearances))

    def _authority_limited_plan(
        self, nominal_plan, residual_plan, observation
    ):
        if not self._authority_indices:
            return residual_plan
        sequence = np.asarray(
            nominal_plan.control_sequence, dtype=np.float64
        ).copy()
        residual_sequence = np.asarray(
            residual_plan.control_sequence, dtype=np.float64
        )
        sequence[:, self._authority_indices] = residual_sequence[
            :, self._authority_indices
        ]
        state = self.residual_controller.state_from_observation(
            observation
        )
        trajectory = self.residual_controller.rollout(
            state, sequence
        )[0]
        diagnostics = dict(residual_plan.diagnostics)
        diagnostics[
            "residual_safety_shield_action_authority_names"
        ] = list(self.shield_config.action_authority_names)
        return PlanResult(
            proposed_control=ControlCommand(
                sequence[0],
                residual_plan.proposed_control.timestamp,
                "residual_shield_candidate",
            ),
            control_sequence=sequence,
            predicted_trajectory=trajectory,
            diagnostics=diagnostics,
        )

    def _evaluate(self, nominal_plan, residual_plan, observation):
        state = self.nominal_controller.state_from_observation(observation)
        nominal_view = self.nominal_controller.rollout(
            state, residual_plan.control_sequence
        )[0]
        residual_view = np.asarray(
            residual_plan.predicted_trajectory, dtype=np.float64
        )
        nominal_baseline = np.asarray(
            nominal_plan.predicted_trajectory, dtype=np.float64
        )
        position_indices = self.state_spec.position_indices
        position_deviation = np.linalg.norm(
            residual_view[:, position_indices]
            - nominal_view[:, position_indices],
            axis=1,
        )
        maximum_deviation = float(np.max(position_deviation))
        target_xy = np.asarray(
            (
                nominal_plan.diagnostics["target_x"],
                nominal_plan.diagnostics["target_y"],
            ),
            dtype=np.float64,
        )
        candidate_goal_distance = float(np.linalg.norm(
            nominal_view[-1, position_indices] - target_xy
        ))
        baseline_goal_distance = float(np.linalg.norm(
            nominal_baseline[-1, position_indices] - target_xy
        ))
        tube_safe = bool(
            maximum_deviation
            <= self.shield_config.maximum_position_deviation_m + 1.0e-12
        )
        progress_noninferior = bool(
            candidate_goal_distance
            <= baseline_goal_distance
            + self.shield_config.maximum_nominal_progress_regression_m
            + 1.0e-12
        )

        nominal_candidate_risk = self._risk(
            self.nominal_controller, nominal_view, observation
        )
        residual_candidate_risk = self._risk(
            self.residual_controller, residual_view, observation
        )
        nominal_baseline_risk = self._risk(
            self.nominal_controller, nominal_baseline, observation
        )
        if (
            nominal_candidate_risk is None
            or residual_candidate_risk is None
            or nominal_baseline_risk is None
        ):
            if (
                not self.config.probabilistic_obstacle_risk_enabled
                and self.shield_config.deterministic_clearance_fallback_enabled
            ):
                nominal_candidate_clearance = self._minimum_local_clearance(
                    nominal_view, observation
                )
                residual_candidate_clearance = self._minimum_local_clearance(
                    residual_view, observation
                )
                nominal_baseline_clearance = self._minimum_local_clearance(
                    nominal_baseline, observation
                )
                nominal_hard_safe = nominal_candidate_clearance > 0.0
                residual_hard_safe = residual_candidate_clearance > 0.0
                clearance_noninferior = bool(
                    (
                        math.isinf(nominal_candidate_clearance)
                        and math.isinf(nominal_baseline_clearance)
                    )
                    or nominal_candidate_clearance
                    + self.shield_config.maximum_nominal_clearance_regression_m
                    + 1.0e-12
                    >= nominal_baseline_clearance
                )
                accepted = bool(
                    (
                        nominal_hard_safe
                        or not self.shield_config.require_nominal_hard_safe
                    )
                    and (
                        residual_hard_safe
                        or not self.shield_config.require_residual_hard_safe
                    )
                    and clearance_noninferior
                    and tube_safe
                    and progress_noninferior
                )
                values = {
                    "missing_forecast": False,
                    "certificate_source": "causal_local_scan_clearance",
                    "nominal_hard_safe": nominal_hard_safe,
                    "residual_hard_safe": residual_hard_safe,
                    "nominal_risk_nonincrease": clearance_noninferior,
                    "position_tube_safe": tube_safe,
                    "nominal_progress_noninferior": progress_noninferior,
                    "nominal_baseline_maximum_probability": 0.0,
                    "candidate_nominal_maximum_probability": 0.0,
                    "candidate_residual_maximum_probability": 0.0,
                    "candidate_conservative_probability_mass": 0.0,
                    "candidate_conservative_union_bound": 0.0,
                    "candidate_conservative_maximum_probability": 0.0,
                    "candidate_conservative_hard_violation": bool(
                        not nominal_hard_safe or not residual_hard_safe
                    ),
                    "nominal_baseline_minimum_clearance_m": (
                        nominal_baseline_clearance
                    ),
                    "candidate_nominal_minimum_clearance_m": (
                        nominal_candidate_clearance
                    ),
                    "candidate_residual_minimum_clearance_m": (
                        residual_candidate_clearance
                    ),
                    "maximum_position_deviation_m": maximum_deviation,
                    "nominal_baseline_goal_distance_m": baseline_goal_distance,
                    "candidate_nominal_goal_distance_m": candidate_goal_distance,
                }
                return accepted, values, nominal_view
            values = {
                "missing_forecast": True,
                "nominal_hard_safe": False,
                "residual_hard_safe": False,
                "nominal_risk_nonincrease": False,
                "position_tube_safe": False,
                "nominal_progress_noninferior": False,
            }
            return False, values, nominal_view

        nominal_candidate_max = float(
            nominal_candidate_risk.maximum_step_probability[0]
        )
        residual_candidate_max = float(
            residual_candidate_risk.maximum_step_probability[0]
        )
        nominal_baseline_max = float(
            nominal_baseline_risk.maximum_step_probability[0]
        )
        nominal_hard_safe = bool(
            not nominal_candidate_risk.hard_violation[0]
        )
        residual_hard_safe = bool(
            not residual_candidate_risk.hard_violation[0]
        )
        risk_nonincrease = bool(
            nominal_candidate_max
            <= nominal_baseline_max
            + self.shield_config.maximum_nominal_risk_increase
            + 1.0e-12
        )
        accepted = bool(
            (nominal_hard_safe or not self.shield_config.require_nominal_hard_safe)
            and (
                residual_hard_safe
                or not self.shield_config.require_residual_hard_safe
            )
            and risk_nonincrease
            and tube_safe
            and progress_noninferior
        )
        values = {
            "missing_forecast": False,
            "nominal_hard_safe": nominal_hard_safe,
            "residual_hard_safe": residual_hard_safe,
            "nominal_risk_nonincrease": risk_nonincrease,
            "position_tube_safe": tube_safe,
            "nominal_progress_noninferior": progress_noninferior,
            "nominal_baseline_maximum_probability": nominal_baseline_max,
            "candidate_nominal_maximum_probability": nominal_candidate_max,
            "candidate_residual_maximum_probability": residual_candidate_max,
            "candidate_conservative_probability_mass": float(max(
                nominal_candidate_risk.accumulated_probability_mass[0],
                residual_candidate_risk.accumulated_probability_mass[0],
            )),
            "candidate_conservative_union_bound": float(max(
                nominal_candidate_risk.horizon_union_bound[0],
                residual_candidate_risk.horizon_union_bound[0],
            )),
            "candidate_conservative_maximum_probability": float(max(
                nominal_candidate_max, residual_candidate_max
            )),
            "candidate_conservative_hard_violation": bool(
                not nominal_hard_safe or not residual_hard_safe
            ),
            "maximum_position_deviation_m": maximum_deviation,
            "nominal_baseline_goal_distance_m": baseline_goal_distance,
            "candidate_nominal_goal_distance_m": candidate_goal_distance,
        }
        return accepted, values, nominal_view

    def _combined_plan(
        self,
        nominal_plan,
        residual_plan,
        accepted,
        shield_values,
        elapsed_ms,
    ):
        selected = residual_plan if accepted else nominal_plan
        diagnostics = dict(selected.diagnostics)
        diagnostics.update(
            {
                "residual_safety_shield_enabled": True,
                "residual_safety_shield_parallel_planning_enabled": bool(
                    self.shield_config.parallel_planning_enabled
                ),
                "residual_safety_shield_accepted": bool(accepted),
                "residual_safety_shield_fallback_used": bool(not accepted),
                "residual_safety_shield_selected_source": (
                    "residual" if accepted else "nominal"
                ),
                "residual_safety_shield_action_authority_names": list(
                    self.shield_config.action_authority_names
                ),
                "residual_safety_shield_nominal_compute_ms": float(
                    nominal_plan.diagnostics.get("compute_ms", 0.0)
                ),
                "residual_safety_shield_residual_compute_ms": float(
                    residual_plan.diagnostics.get("compute_ms", 0.0)
                ),
                "compute_ms": float(elapsed_ms),
            }
        )
        for name, value in shield_values.items():
            diagnostics[
                "residual_safety_shield_%s" % name
            ] = value
        if accepted:
            # Downstream safety must see the risk of the actual shield-selected
            # sequence.  This matters when action authority creates a hybrid
            # sequence whose diagnostics cannot be copied from either planner.
            diagnostics["probabilistic_obstacle_probability_mass"] = float(
                shield_values[
                    "candidate_conservative_probability_mass"
                ]
            )
            diagnostics["probabilistic_obstacle_union_bound"] = float(
                shield_values["candidate_conservative_union_bound"]
            )
            diagnostics[
                "probabilistic_obstacle_maximum_step_probability"
            ] = float(
                shield_values[
                    "candidate_conservative_maximum_probability"
                ]
            )
            diagnostics["probabilistic_obstacle_hard_violation"] = bool(
                shield_values[
                    "candidate_conservative_hard_violation"
                ]
            )
        for name, value in residual_plan.diagnostics.items():
            if (
                name.startswith("residual_reliability_")
                or name.startswith("residual_stall_guard_")
                or name.startswith("residual_support_")
            ):
                diagnostics[name] = value
        return PlanResult(
            proposed_control=selected.proposed_control,
            control_sequence=np.asarray(
                selected.control_sequence, dtype=np.float64
            ).copy(),
            predicted_trajectory=np.asarray(
                selected.predicted_trajectory, dtype=np.float64
            ).copy(),
            diagnostics=diagnostics,
        )

    def plan(self, observation, reference):
        started = time.perf_counter()
        nominal_plan, residual_plan = self._plan_pair(
            "plan", observation, reference
        )
        candidate_plan = self._authority_limited_plan(
            nominal_plan, residual_plan, observation
        )
        accepted, values, _ = self._evaluate(
            nominal_plan, candidate_plan, observation
        )
        self._last_nominal_plan = nominal_plan
        self._last_residual_plan = residual_plan
        return self._combined_plan(
            nominal_plan,
            candidate_plan,
            accepted,
            values,
            1000.0 * (time.perf_counter() - started),
        )

    def preview_plan(self, observation, reference):
        started = time.perf_counter()
        nominal_plan, residual_plan = self._plan_pair(
            "preview_plan", observation, reference
        )
        candidate_plan = self._authority_limited_plan(
            nominal_plan, residual_plan, observation
        )
        accepted, values, _ = self._evaluate(
            nominal_plan, candidate_plan, observation
        )
        return self._combined_plan(
            nominal_plan,
            candidate_plan,
            accepted,
            values,
            1000.0 * (time.perf_counter() - started),
        )

    def observe_safety_decision(self, decision):
        nominal_proposed = (
            decision.proposed_control
            if self._last_nominal_plan is None
            else self._last_nominal_plan.proposed_control
        )
        residual_proposed = (
            decision.proposed_control
            if self._last_residual_plan is None
            else self._last_residual_plan.proposed_control
        )
        self.nominal_controller.observe_safety_decision(SafetyDecision(
            proposed_control=nominal_proposed,
            executed_control=decision.executed_control,
            overridden=decision.overridden,
            reason=decision.reason,
            diagnostics=dict(decision.diagnostics),
        ))
        self.residual_controller.observe_safety_decision(SafetyDecision(
            proposed_control=residual_proposed,
            executed_control=decision.executed_control,
            overridden=decision.overridden,
            reason=decision.reason,
            diagnostics=dict(decision.diagnostics),
        ))

    def observe_completed_transition(self, *args, **kwargs):
        nominal = self.nominal_controller.observe_completed_transition(
            *args, **kwargs
        )
        residual = self.residual_controller.observe_completed_transition(
            *args, **kwargs
        )
        return bool(nominal or residual)
