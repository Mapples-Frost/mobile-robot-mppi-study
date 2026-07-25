from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from mobile_robot_mppi.core.spaces import (
    body_velocity_action,
    dynamic_unicycle_state,
)
from mobile_robot_mppi.core.types import (
    ControlCommand,
    PlanResult,
    Pose2D,
    RobotObservation,
    SafetyDecision,
    Twist2D,
)
from mobile_robot_mppi.planning.residual_shield import (
    ResidualSafetyShieldConfig,
    ResidualSafetyShieldController,
)


def _trajectory(final_x):
    result = np.zeros((4, 5), dtype=np.float64)
    result[:, 0] = np.linspace(0.0, float(final_x), 4)
    return result


def _plan(action_v, final_x, compute_ms=5.0):
    return PlanResult(
        proposed_control=ControlCommand(np.asarray((action_v, 0.0))),
        control_sequence=np.tile(
            np.asarray((action_v, 0.0)), (3, 1)
        ),
        predicted_trajectory=_trajectory(final_x),
        diagnostics={
            "target_x": 2.0,
            "target_y": 0.0,
            "compute_ms": compute_ms,
            "probabilistic_obstacle_maximum_step_probability": 0.0,
            "residual_reliability_alpha": 0.75,
        },
    )


class _FakeController:
    def __init__(self, plan, candidate_nominal_final_x, risks, config):
        self._plan = plan
        self._candidate_nominal_final_x = candidate_nominal_final_x
        self._risks = dict(risks)
        self.config = config
        self.state_spec = dynamic_unicycle_state()
        self.action_spec = body_velocity_action((0.0, 0.4), 1.0)
        self.dynamics = object()
        self.sampling_prior = object()
        self.previous_action = np.zeros(2)
        self.decisions = []
        self.reset_seeds = []
        self.plan_barrier = None

    def plan(self, observation, reference):
        del observation, reference
        if self.plan_barrier is not None:
            self.plan_barrier.wait(timeout=1.0)
        return self._plan

    def preview_plan(self, observation, reference):
        del observation, reference
        return self._plan

    def state_from_observation(self, observation):
        return np.asarray(
            (
                observation.pose.x,
                observation.pose.y,
                observation.pose.theta,
                observation.twist.v,
                observation.twist.omega,
            )
        )

    def rollout(self, state, controls):
        del state, controls
        return _trajectory(self._candidate_nominal_final_x)[None, ...]

    def _probabilistic_collision_risk(self, trajectories, forecasts):
        del forecasts
        final_x = round(float(trajectories[0, -1, 0]), 6)
        probability = float(self._risks[final_x])
        return SimpleNamespace(
            accumulated_probability_mass=np.asarray((probability * 4.0,)),
            horizon_union_bound=np.asarray((min(1.0, probability * 4.0),)),
            maximum_step_probability=np.asarray((probability,)),
            hard_violation=np.asarray(
                (probability > self.config.probabilistic_obstacle_hard_threshold,)
            ),
        )

    def observe_safety_decision(self, decision):
        self.decisions.append(decision)

    def observe_completed_transition(self, *args, **kwargs):
        del args, kwargs
        return False

    def reset(self, seed=None):
        self.reset_seeds.append(seed)


def _observation():
    return RobotObservation(
        0.0,
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.0, 0.0),
        auxiliary={"probabilistic_obstacle_forecasts": (object(),)},
    )


def _shield(candidate_risk=0.08, residual_final_x=1.07):
    config = SimpleNamespace(
        probabilistic_obstacle_risk_enabled=True,
        probabilistic_obstacle_forecast_key=(
            "probabilistic_obstacle_forecasts"
        ),
        probabilistic_obstacle_hard_threshold=0.20,
    )
    nominal = _FakeController(
        _plan(0.1, 1.0, 3.0),
        candidate_nominal_final_x=1.05,
        risks={1.0: 0.10, 1.05: candidate_risk, residual_final_x: 0.09},
        config=config,
    )
    residual = _FakeController(
        _plan(0.2, residual_final_x, 7.0),
        candidate_nominal_final_x=1.05,
        risks={1.0: 0.10, 1.05: candidate_risk, residual_final_x: 0.09},
        config=config,
    )
    return ResidualSafetyShieldController(
        residual, nominal, ResidualSafetyShieldConfig()
    )


def test_accepts_residual_only_when_both_views_are_noninterfering():
    shield = _shield()
    result = shield.plan(_observation(), object())
    assert result.proposed_control.v == 0.2
    assert result.diagnostics["residual_safety_shield_accepted"] is True
    assert (
        result.diagnostics["residual_safety_shield_selected_source"]
        == "residual"
    )
    assert (
        result.diagnostics[
            "residual_safety_shield_candidate_nominal_maximum_probability"
        ]
        == 0.08
    )
    assert result.diagnostics["compute_ms"] >= 0.0


def test_parallel_shield_runs_independent_planners_concurrently():
    shield = _shield()
    barrier = threading.Barrier(2)
    shield.nominal_controller.plan_barrier = barrier
    shield.residual_controller.plan_barrier = barrier
    shield.close()
    shield.shield_config = ResidualSafetyShieldConfig(
        parallel_planning_enabled=True
    )
    shield._planner_executor = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="test-residual-shield"
    )
    try:
        result = shield.plan(_observation(), object())
    finally:
        shield.close()

    assert result.diagnostics[
        "residual_safety_shield_parallel_planning_enabled"
    ]


def test_nominal_risk_increase_returns_exact_nominal_plan():
    shield = _shield(candidate_risk=0.11)
    result = shield.plan(_observation(), object())
    assert result.proposed_control.v == 0.1
    assert result.diagnostics["residual_safety_shield_accepted"] is False
    assert result.diagnostics["residual_safety_shield_fallback_used"] is True
    assert (
        result.diagnostics["residual_safety_shield_nominal_risk_nonincrease"]
        is False
    )


def test_deterministic_clearance_fallback_supports_probability_off_cell():
    config = SimpleNamespace(
        probabilistic_obstacle_risk_enabled=False,
        probabilistic_obstacle_forecast_key="probabilistic_obstacle_forecasts",
        probabilistic_obstacle_hard_threshold=0.20,
        robot_radius=0.25,
    )
    nominal = _FakeController(
        _plan(0.1, 1.0), 1.05, {}, config
    )
    residual = _FakeController(
        _plan(0.2, 1.07), 1.05, {}, config
    )
    shield = ResidualSafetyShieldController(
        residual,
        nominal,
        ResidualSafetyShieldConfig(
            deterministic_clearance_fallback_enabled=True,
            maximum_nominal_clearance_regression_m=0.0,
        ),
    )
    observation = RobotObservation(
        0.0,
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.0, 0.0),
        local_obstacles=((1.50, 0.0, 0.10),),
    )
    result = shield.plan(observation, object())
    assert result.diagnostics[
        "residual_safety_shield_certificate_source"
    ] == "causal_local_scan_clearance"
    assert result.diagnostics["residual_safety_shield_accepted"] is False
    assert result.proposed_control.v == 0.1


def test_position_tube_violation_returns_exact_nominal_plan():
    shield = _shield(residual_final_x=1.20)
    result = shield.plan(_observation(), object())
    assert result.proposed_control.v == 0.1
    assert (
        result.diagnostics["residual_safety_shield_position_tube_safe"]
        is False
    )


def test_safety_feedback_is_mapped_to_each_internal_proposal():
    shield = _shield()
    selected = shield.plan(_observation(), object())
    decision = SafetyDecision(
        proposed_control=selected.proposed_control,
        executed_control=ControlCommand(np.asarray((0.0, 0.0))),
        overridden=True,
        reason="test",
    )
    shield.observe_safety_decision(decision)
    assert (
        shield.nominal_controller.decisions[-1].proposed_control.v == 0.1
    )
    assert (
        shield.residual_controller.decisions[-1].proposed_control.v == 0.2
    )
    assert (
        shield.nominal_controller.decisions[-1].executed_control.v == 0.0
    )
    assert (
        shield.residual_controller.decisions[-1].executed_control.v == 0.0
    )


def test_speed_only_authority_preserves_nominal_yaw_sequence():
    shield = _shield()
    shield.shield_config = ResidualSafetyShieldConfig(
        action_authority_names=("v_cmd",)
    )
    shield._authority_indices = (
        shield.action_spec.index("v_cmd"),
    )
    nominal = shield.nominal_controller._plan
    residual_sequence = shield.residual_controller._plan.control_sequence.copy()
    residual_sequence[:, 1] = 0.7
    shield.residual_controller._plan = PlanResult(
        proposed_control=ControlCommand(residual_sequence[0]),
        control_sequence=residual_sequence,
        predicted_trajectory=shield.residual_controller._plan.predicted_trajectory,
        diagnostics=shield.residual_controller._plan.diagnostics,
    )
    candidate = shield._authority_limited_plan(
        nominal, shield.residual_controller._plan, _observation()
    )
    np.testing.assert_array_equal(
        candidate.control_sequence[:, 0], residual_sequence[:, 0]
    )
    np.testing.assert_array_equal(
        candidate.control_sequence[:, 1], nominal.control_sequence[:, 1]
    )


@pytest.mark.parametrize(
    "mapping",
    (
        {"maximum_nominal_risk_increase": -0.01},
        {"maximum_position_deviation_m": 0.0},
        {"maximum_nominal_progress_regression_m": -0.01},
        {"maximum_nominal_clearance_regression_m": -0.01},
        {"action_authority_names": ["v_cmd", "v_cmd"]},
    ),
)
def test_invalid_shield_thresholds_are_rejected(mapping):
    with pytest.raises(ValueError):
        ResidualSafetyShieldConfig.from_mapping(mapping)
