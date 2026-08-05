import numpy as np

from mobile_robot_mppi.core.types import ControlCommand, PlanResult
from mobile_robot_mppi.real_robot.forward_passage import (
    ForwardPassageConfig,
    ForwardPassageController,
)


def _candidate(index, v, omega, cost=1.0, probability=0.0, mass=0.0, **kw):
    value = {
        "index": index,
        "eligible": True,
        "hard_violation": False,
        "first_v": v,
        "first_omega": omega,
        "cost": cost,
        "maximum_probability": probability,
        "probability_mass": mass,
        "sequence": [[v, omega]] * 12,
    }
    value.update(kw)
    return value


def _plan(trace=(), proposed=(-0.35, 0.0), **diagnostics):
    values = {
        "probabilistic_obstacle_active_fallback_kind": (
            "temporal_scan_vetted_emergency_candidate"
        ),
        "probabilistic_obstacle_emergency_candidate_trace": trace,
        "probabilistic_obstacle_maximum_step_probability": 0.0,
        "probabilistic_obstacle_probability_mass": 0.0,
        "probabilistic_obstacle_hard_violation": False,
    }
    values.update(diagnostics)
    return PlanResult(
        proposed_control=ControlCommand(np.asarray(proposed), timestamp=4.2),
        control_sequence=np.tile(np.asarray(proposed), (12, 1)),
        predicted_trajectory=np.zeros((13, 3)),
        diagnostics=values,
    )


def _controller(**kwargs):
    return ForwardPassageController(
        ForwardPassageConfig(enabled=True, **kwargs)
    )


def _prime_forward(controller, v=0.3, omega=0.0, steps=3):
    for _ in range(steps):
        controller.observe_executed(
            ControlCommand([v, omega]), "front_clear"
        )


def test_disabled_mode_returns_bit_exact_original_object():
    controller = ForwardPassageController(ForwardPassageConfig())
    plan = _plan([_candidate(1, 0.7, 0.0)])
    assert controller.apply(plan) is plan


def test_semantic_mode_inhibit_releases_transaction_and_never_changes_plan():
    controller = _controller()
    _prime_forward(controller)
    started = controller.apply(_plan([
        _candidate(1, 0.4, 0.1, cost=0.1),
    ]))
    assert controller.active

    original = _plan([
        _candidate(2, 0.4, -0.1, cost=0.1),
    ], proposed=(0.2, -0.2))
    inhibited = controller.apply(
        original, inhibit_reason="encounter_mode:straight_crossing"
    )

    assert not controller.active
    assert inhibited.proposed_control is original.proposed_control
    assert inhibited.diagnostics[
        "real_robot_forward_passage_externally_inhibited"
    ] is True
    assert inhibited.diagnostics["real_robot_forward_passage_inhibit_reason"] == (
        "encounter_mode:straight_crossing"
    )
    assert started.diagnostics["real_robot_forward_passage_applied"] is True


def test_reverse_or_stopped_history_cannot_start_forward_transaction():
    controller = _controller()
    plan = _plan([_candidate(1, 0.7, 0.0)])
    result = controller.apply(plan)
    assert result.proposed_control is plan.proposed_control
    assert result.diagnostics["real_robot_forward_passage_applied"] is False
    controller.observe_executed(ControlCommand([-0.2, 0.0]), "front_clear")
    result = controller.apply(plan)
    assert result.proposed_control is plan.proposed_control


def test_safe_directionally_consistent_forward_continuation_starts():
    controller = _controller()
    _prime_forward(controller, omega=0.1)
    plan = _plan([
        _candidate(1, 0.7, 0.8, cost=0.1),
        _candidate(2, 0.7, 0.0, cost=2.0),
    ])
    result = controller.apply(plan)
    assert np.array_equal(result.proposed_control.values, [0.7, 0.0])
    assert result.proposed_control.source == "real_robot_forward_passage_v3"
    assert result.diagnostics["real_robot_forward_passage_started"] is True
    assert result.diagnostics["real_robot_forward_passage_applied"] is True


def test_latched_sequence_bridges_missing_trace_without_sign_flip():
    controller = _controller(commit_steps=4)
    _prime_forward(controller)
    first = controller.apply(_plan([_candidate(1, 0.7, 0.0)]))
    controller.observe_executed(first.proposed_control, "front_clear")
    second = controller.apply(_plan((), proposed=(-0.29, 0.1),
                                    probabilistic_obstacle_active_fallback_kind="none"))
    assert second.diagnostics["real_robot_forward_passage_active"] is True
    assert second.diagnostics["real_robot_forward_passage_applied"] is True
    assert second.proposed_control.v == 0.7


def test_fresh_hard_risk_releases_latched_transaction():
    controller = _controller(commit_steps=4)
    _prime_forward(controller)
    controller.apply(_plan([_candidate(1, 0.7, 0.0)]))
    unsafe = _plan(
        (), proposed=(-0.29, 0.1),
        probabilistic_obstacle_active_fallback_kind="none",
        probabilistic_obstacle_maximum_step_probability=0.081,
    )
    result = controller.apply(unsafe)
    assert result.proposed_control is unsafe.proposed_control
    assert result.diagnostics["real_robot_forward_passage_release"] == "fresh_risk"
    assert controller.active is False
    blocked = controller.apply(_plan([_candidate(1, 0.7, 0.0)]))
    assert blocked.proposed_control.v < 0.0
    assert blocked.diagnostics[
        "real_robot_forward_passage_cooldown_remaining"
    ] > 0


def test_missing_fresh_risk_diagnostics_fail_closed():
    controller = _controller(commit_steps=4)
    _prime_forward(controller)
    controller.apply(_plan([_candidate(1, 0.7, 0.0)]))
    missing = _plan((), proposed=(-0.29, 0.1))
    del missing.diagnostics[
        "probabilistic_obstacle_maximum_step_probability"
    ]
    result = controller.apply(missing)
    assert result.proposed_control is missing.proposed_control
    assert result.diagnostics["real_robot_forward_passage_release"] == "fresh_risk"
    assert controller.active is False


def test_existing_traversal_transaction_blocks_and_releases():
    controller = _controller()
    _prime_forward(controller)
    plan = _plan(
        [_candidate(1, 0.7, 0.0)],
        probabilistic_obstacle_traversal_retreat_requested=True,
    )
    result = controller.apply(plan)
    assert result.proposed_control is plan.proposed_control
    assert result.diagnostics["real_robot_forward_passage_release"] == (
        "existing_transaction"
    )


def test_safety_stop_or_reverse_releases_transaction():
    controller = _controller()
    _prime_forward(controller)
    controller.apply(_plan([_candidate(1, 0.7, 0.0)]))
    assert controller.active is True
    controller.observe_executed(ControlCommand([0.0, 0.0]), "near_body_hard_stop")
    assert controller.active is False
    _prime_forward(controller, steps=4)
    controller.apply(_plan([_candidate(1, 0.7, 0.0)]))
    controller.observe_executed(ControlCommand([-0.1, 0.0]), "front_clear")
    assert controller.active is False


def test_candidate_limits_are_enforced_before_start():
    controller = _controller()
    _prime_forward(controller)
    plan = _plan([
        _candidate(1, 0.7, 0.0, probability=0.081),
        _candidate(2, 0.7, 0.0, mass=1.51),
        _candidate(3, 0.7, 0.0, hard_violation=True),
    ])
    result = controller.apply(plan)
    assert result.proposed_control is plan.proposed_control
    assert result.diagnostics[
        "real_robot_forward_passage_eligible_candidate_count"
    ] == 0


def test_requires_stable_forward_feedback_before_start():
    controller = _controller(minimum_forward_feedback_steps=3)
    controller.observe_executed(ControlCommand([0.3, 0.0]), "front_clear")
    plan = _plan([_candidate(1, 0.7, 0.0)])
    result = controller.apply(plan)
    assert result.proposed_control is plan.proposed_control
    assert result.diagnostics[
        "real_robot_forward_passage_forward_feedback_streak"
    ] == 1


def test_committed_prefix_must_remain_forward_and_steering_continuous():
    controller = _controller(commit_steps=4)
    _prime_forward(controller)
    reversing = _candidate(
        1, 0.7, 0.0,
        sequence=[[0.7, 0.0], [0.7, 0.0], [-0.1, 0.0], [0.7, 0.0]],
    )
    steering_flip = _candidate(
        2, 0.7, 0.0,
        sequence=[[0.7, 0.0], [0.7, 0.1], [0.7, 0.8], [0.7, 0.8]],
    )
    plan = _plan([reversing, steering_flip])
    result = controller.apply(plan)
    assert result.proposed_control is plan.proposed_control


def test_latched_prefix_advances_without_per_frame_relatch():
    controller = _controller(commit_steps=4)
    _prime_forward(controller)
    original = _candidate(
        1, 0.3, 0.0,
        sequence=[[0.3, 0.0], [0.4, 0.1], [0.5, 0.2], [0.6, 0.25]],
    )
    first = controller.apply(_plan([original]))
    controller.observe_executed(first.proposed_control, "front_clear")
    cheaper_new = _candidate(2, 0.9, 0.0, cost=0.0)
    second = controller.apply(_plan([cheaper_new]))
    assert np.allclose(second.proposed_control.values, [0.4, 0.1])
    assert np.array_equal(
        second.control_sequence[0], second.proposed_control.values
    )
    assert second.diagnostics["real_robot_forward_passage_refreshed"] is False


def test_completion_enters_cooldown_before_any_reentry():
    controller = _controller(commit_steps=2, reentry_cooldown_steps=4)
    _prime_forward(controller)
    candidate = _candidate(1, 0.7, 0.0)
    first = controller.apply(_plan([candidate]))
    controller.observe_executed(first.proposed_control, "front_clear")
    second = controller.apply(_plan([candidate]))
    assert second.diagnostics["real_robot_forward_passage_completed"] is True
    controller.observe_executed(second.proposed_control, "front_clear")
    blocked = _plan([candidate])
    result = controller.apply(blocked)
    assert result.proposed_control is blocked.proposed_control
    assert result.diagnostics[
        "real_robot_forward_passage_cooldown_remaining"
    ] > 0


def test_final_nonforward_command_releases_even_with_forward_feedback():
    controller = _controller(commit_steps=4)
    _prime_forward(controller)
    controller.apply(_plan([_candidate(1, 0.7, 0.0)]))
    assert controller.active is True
    controller.observe_commanded(ControlCommand([0.0, 0.0]), "front_clear")
    assert controller.active is False


def test_repeated_forward_command_feedback_mismatch_releases():
    controller = _controller(
        commit_steps=4, maximum_feedback_mismatch_steps=2
    )
    _prime_forward(controller)
    first = controller.apply(_plan([_candidate(1, 0.7, 0.0)]))
    controller.observe_commanded(first.proposed_control, "front_clear")
    controller.observe_feedback(ControlCommand([-0.1, 0.0]))
    assert controller.active is True
    controller.observe_feedback(ControlCommand([-0.1, 0.0]))
    assert controller.active is False
    next_plan = _plan([_candidate(1, 0.7, 0.0)])
    result = controller.apply(next_plan)
    assert result.proposed_control is next_plan.proposed_control
    assert result.diagnostics["real_robot_forward_passage_release"] == (
        "feedback_direction_mismatch"
    )


def test_post_arbitration_release_reason_is_logged_next_cycle():
    controller = _controller(commit_steps=4)
    _prime_forward(controller)
    controller.apply(_plan([_candidate(1, 0.7, 0.0)]))
    controller.observe_commanded(
        ControlCommand([0.0, 0.0]), "near_body_hard_stop"
    )
    next_plan = _plan([_candidate(1, 0.7, 0.0)])
    result = controller.apply(next_plan)
    assert result.proposed_control is next_plan.proposed_control
    assert result.diagnostics["real_robot_forward_passage_release"] == (
        "safety_near_body_hard_stop"
    )
