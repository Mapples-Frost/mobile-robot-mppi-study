import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from types import SimpleNamespace

from mobile_robot_mppi.core.references import PointGoal, PolylineReference
from mobile_robot_mppi.core.spaces import (
    body_velocity_action,
    unicycle_state,
)
from mobile_robot_mppi.obstacles.collision_risk import (
    CollisionRiskConfig,
    GaussianMixtureObstacleForecast,
    component_collision_probability_upper_bound,
    evaluate_collision_risk,
    mixture_collision_probability_upper_bound,
    standard_normal_cdf,
)
from mobile_robot_mppi.core.types import Pose2D, RobotObservation, Twist2D
from mobile_robot_mppi.planning.dynamics import LegacyUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController


def _forecast(
    means,
    covariance=0.04,
    weights=None,
    radius=0.15,
    timestamp=0.0,
    dt=0.1,
):
    means = np.asarray(means, dtype=np.float64)
    if means.ndim == 2:
        means = means[:, None, :]
    horizon, modes, _ = means.shape
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim == 0:
        covariances = np.repeat(
            (float(covariance) * np.eye(2))[None, None, :, :],
            horizon * modes,
            axis=0,
        ).reshape(horizon, modes, 2, 2)
    elif covariance.shape == (2, 2):
        covariances = np.repeat(
            covariance[None, None, :, :],
            horizon * modes,
            axis=0,
        ).reshape(horizon, modes, 2, 2)
    else:
        covariances = covariance
    if weights is None:
        weights = np.full((horizon, modes), 1.0 / modes)
    return GaussianMixtureObstacleForecast(
        timestamp=timestamp,
        dt=dt,
        component_means=means,
        component_covariances=covariances,
        component_weights=weights,
        radius_m=radius,
        source="synthetic_test",
    )


def _risk_config(**overrides):
    values = {
        "robot_radius_m": 0.25,
        "safety_margin_m": 0.10,
        "minimum_position_std_m": 0.01,
        "hard_probability_threshold": 0.20,
    }
    values.update(overrides)
    return CollisionRiskConfig(**values)


def test_forecast_contract_rejects_bad_weights_and_covariance():
    means = np.zeros((2, 2, 2))
    covariances = np.repeat(
        np.eye(2)[None, None, :, :], 4, axis=0
    ).reshape(2, 2, 2, 2)
    with pytest.raises(ValueError, match="sum to one"):
        GaussianMixtureObstacleForecast(
            0.0,
            0.1,
            means,
            covariances,
            np.full((2, 2), 0.4),
            0.15,
        )
    covariances[0, 0, 0, 1] = 0.5
    with pytest.raises(ValueError, match="symmetric"):
        GaussianMixtureObstacleForecast(
            0.0,
            0.1,
            means,
            covariances,
            np.full((2, 2), 0.5),
            0.15,
        )


def test_normal_cdf_matches_reference_values():
    values = np.asarray((-4.0, -1.96, -1.0, 0.0, 1.0, 1.96, 4.0))
    expected = np.asarray(
        (
            0.0000316712,
            0.0249978951,
            0.1586552539,
            0.5,
            0.8413447461,
            0.9750021049,
            0.9999683288,
        )
    )
    np.testing.assert_allclose(
        standard_normal_cdf(values), expected, atol=8.0e-8, rtol=0.0
    )


def test_inside_is_one_and_outside_ordering_is_monotone():
    config = _risk_config()
    robot = np.zeros((1, 1, 2))
    inside = _forecast([[[0.40, 0.0]]])
    near = _forecast([[[0.70, 0.0]]])
    far = _forecast([[[1.20, 0.0]]])
    inside_value = mixture_collision_probability_upper_bound(
        robot, inside, config
    )[0, 0]
    near_value = mixture_collision_probability_upper_bound(
        robot, near, config
    )[0, 0]
    far_value = mixture_collision_probability_upper_bound(
        robot, far, config
    )[0, 0]
    assert inside_value == 1.0
    assert 0.0 <= far_value < near_value < inside_value


def test_larger_radial_uncertainty_increases_outside_risk():
    config = _risk_config()
    robot = np.zeros((1, 1, 2))
    low = _forecast([[[0.85, 0.0]]], covariance=np.diag((0.01, 0.04)))
    high = _forecast([[[0.85, 0.0]]], covariance=np.diag((0.09, 0.04)))
    low_value = mixture_collision_probability_upper_bound(
        robot, low, config
    )[0, 0]
    high_value = mixture_collision_probability_upper_bound(
        robot, high, config
    )[0, 0]
    assert high_value > low_value


def test_mixture_is_weight_linear_and_component_permutation_invariant():
    config = _risk_config()
    robot = np.zeros((1, 1, 2))
    means = np.asarray([[[0.70, 0.0], [1.20, 0.0]]])
    weights = np.asarray([[0.25, 0.75]])
    forecast = _forecast(means, weights=weights)
    component = component_collision_probability_upper_bound(
        robot, forecast, config
    )
    mixture = mixture_collision_probability_upper_bound(
        robot, forecast, config
    )
    expected = np.sum(component * weights[None, :, :], axis=-1)
    np.testing.assert_allclose(mixture, expected, atol=1.0e-14)

    permuted = _forecast(
        means[:, ::-1],
        covariance=forecast.component_covariances[:, ::-1],
        weights=weights[:, ::-1],
    )
    np.testing.assert_allclose(
        mixture_collision_probability_upper_bound(
            robot, permuted, config
        ),
        mixture,
        atol=1.0e-14,
    )


def test_crossing_path_has_more_risk_and_union_bound_dominates_steps():
    config = _risk_config()
    forecast = _forecast(
        [
            [[0.5, 0.5]],
            [[1.0, 0.0]],
            [[1.5, -0.5]],
        ],
        covariance=np.diag((0.03, 0.03)),
    )
    crossing = np.asarray([[[0.5, 0.0], [1.0, 0.0], [1.5, 0.0]]])
    bypass = crossing.copy()
    bypass[:, :, 1] = 1.5
    evaluation = evaluate_collision_risk(
        np.concatenate((crossing, bypass), axis=0),
        (forecast,),
        config,
    )
    assert (
        evaluation.accumulated_probability_mass[0]
        > evaluation.accumulated_probability_mass[1]
    )
    assert np.all(
        evaluation.horizon_union_bound[:, None]
        >= evaluation.step_probability_upper_bound - 1.0e-14
    )
    assert evaluation.hard_violation[0]


def test_short_robot_path_uses_causal_prefix_of_longer_forecast():
    forecast = _forecast(
        [[[0.8, 0.0]], [[1.0, 0.0]], [[1.2, 0.0]]],
        covariance=0.01,
    )
    robot = np.asarray([[[0.0, 0.0], [0.2, 0.0]]])

    actual = evaluate_collision_risk(robot, (forecast,), _risk_config())
    sliced = _forecast(
        forecast.component_means[:2],
        covariance=forecast.component_covariances[:2],
        weights=forecast.component_weights[:2],
    )
    expected = evaluate_collision_risk(robot, (sliced,), _risk_config())

    np.testing.assert_allclose(
        actual.step_probability_upper_bound,
        expected.step_probability_upper_bound,
        atol=1.0e-14,
    )


def _traversal_controller(
    translation_heading_gate_rad=0.0,
    abort_probability=0.0,
    hard_violation_action="penalize",
    retreat_margin_m=0.0,
    temporal_abort_mass_floor=0.0,
    temporal_full_horizon_corroboration=False,
    commit_admission_full_horizon=False,
    commit_admission_exit_deadline=False,
    temporal_abort_nearest_exit=False,
    temporal_midpoint_guard=False,
    temporal_midpoint_lattice_override=False,
    temporal_exit_deadline_guard=False,
    temporal_exit_deadline_escape_latch=False,
    rearm_no_crossing_clear=False,
    rearm_no_crossing_certified_handoff=False,
    commit_admission_safe_hold_steps=1,
    commit_admission_prealign=False,
    uncommitted_temporal_staging_hold=False,
    uncommitted_temporal_staging_terminal_release=False,
    rearm_staging_approach=False,
    rearm_staging_frontier=False,
    post_center_forward_exit_commit=False,
):
    return MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=10,
            num_samples=16,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_risk_weight=1.0,
            probabilistic_obstacle_hard_violation_action=hard_violation_action,
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_traversal_window_enabled=True,
            probabilistic_obstacle_traversal_window_horizon_steps=80,
            probabilistic_obstacle_traversal_window_activation_distance_m=2.0,
            probabilistic_obstacle_traversal_window_cross_track_m=0.45,
            probabilistic_obstacle_traversal_window_clearance_margin_m=0.05,
            probabilistic_obstacle_traversal_window_probability_ceiling=0.02,
            probabilistic_obstacle_traversal_window_mass_ceiling=0.10,
            probabilistic_obstacle_traversal_window_translation_heading_gate_rad=(
                translation_heading_gate_rad
            ),
            probabilistic_obstacle_traversal_window_abort_probability=(
                abort_probability
            ),
            probabilistic_obstacle_traversal_window_temporal_abort_mass_floor=(
                temporal_abort_mass_floor
            ),
            probabilistic_obstacle_traversal_window_temporal_abort_full_horizon_corroboration_enabled=(
                temporal_full_horizon_corroboration
            ),
            probabilistic_obstacle_traversal_window_commit_admission_full_horizon_enabled=(
                commit_admission_full_horizon
            ),
            probabilistic_obstacle_traversal_window_commit_admission_exit_deadline_enabled=(
                commit_admission_exit_deadline
            ),
            probabilistic_obstacle_traversal_window_commit_admission_safe_hold_steps=(
                commit_admission_safe_hold_steps
            ),
            probabilistic_obstacle_traversal_window_commit_admission_prealign_enabled=(
                commit_admission_prealign
            ),
            probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_hold_enabled=(
                uncommitted_temporal_staging_hold
            ),
            probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_terminal_release_enabled=(
                uncommitted_temporal_staging_terminal_release
            ),
            probabilistic_obstacle_traversal_window_rearm_staging_approach_enabled=(
                rearm_staging_approach
            ),
            probabilistic_obstacle_traversal_window_rearm_staging_frontier_enabled=(
                rearm_staging_frontier
            ),
            probabilistic_obstacle_traversal_window_post_center_forward_exit_commit_enabled=(
                post_center_forward_exit_commit
            ),
            probabilistic_obstacle_traversal_window_temporal_abort_nearest_exit_enabled=(
                temporal_abort_nearest_exit
            ),
            probabilistic_obstacle_traversal_window_temporal_midpoint_guard_enabled=(
                temporal_midpoint_guard
            ),
            probabilistic_obstacle_traversal_window_temporal_midpoint_lattice_override_enabled=(
                temporal_midpoint_lattice_override
            ),
            probabilistic_obstacle_traversal_window_temporal_exit_deadline_guard_enabled=(
                temporal_exit_deadline_guard
            ),
            probabilistic_obstacle_traversal_window_temporal_exit_deadline_escape_latch_enabled=(
                temporal_exit_deadline_escape_latch
            ),
            probabilistic_obstacle_traversal_window_rearm_no_crossing_clear_enabled=(
                rearm_no_crossing_clear
            ),
            probabilistic_obstacle_traversal_window_rearm_no_crossing_certified_handoff_enabled=(
                rearm_no_crossing_certified_handoff
            ),
            probabilistic_obstacle_traversal_window_retreat_margin_m=(
                retreat_margin_m
            ),
        ),
    )


def _traversal_reference():
    return PolylineReference(
        ((-1.0, 0.0), (3.0, 0.0)),
        projection_forward_distance=4.0,
    )


def _traversal_forecast(*, occupied):
    means = np.empty((80, 1, 2), dtype=np.float64)
    means[:, 0, 0] = 0.8
    means[:, 0, 1] = 0.0 if occupied else 2.0
    if not occupied:
        means[0, 0, 1] = 0.0
    return _forecast(means, covariance=1.0e-6, radius=0.15)


def _noncrossing_traversal_forecast():
    means = np.empty((80, 1, 2), dtype=np.float64)
    means[:, 0, 0] = 0.8
    means[:, 0, 1] = 2.0
    return _forecast(means, covariance=1.0e-6, radius=0.15)


def test_uncommitted_temporal_closing_stages_before_crossing_is_visible():
    controller = _traversal_controller(
        uncommitted_temporal_staging_hold=True,
    )
    state = np.zeros(controller.state_spec.dimension)

    context = controller._probabilistic_traversal_window_context(
        state,
        _traversal_reference(),
        (_noncrossing_traversal_forecast(),),
        temporal_emergency_closing_observed=True,
    )

    assert context["candidate_requested"]
    assert context["uncommitted_temporal_staging_hold_requested"]
    assert not context["commit_active"]
    assert not context["retreat_requested"]
    assert not context["rearm_pending"]
    v_index = controller.action_spec.index("v_cmd")
    np.testing.assert_allclose(context["sequence"][:, v_index], 0.0)


def test_disabled_uncommitted_temporal_staging_preserves_prior_behavior():
    controller = _traversal_controller(
        uncommitted_temporal_staging_hold=False,
    )
    context = controller._probabilistic_traversal_window_context(
        np.zeros(controller.state_spec.dimension),
        _traversal_reference(),
        (_noncrossing_traversal_forecast(),),
        temporal_emergency_closing_observed=True,
    )

    assert not context["candidate_requested"]
    assert not context["uncommitted_temporal_staging_hold_requested"]
    assert context["sequence"] is None


def test_terminal_phase_releases_only_uncommitted_temporal_staging():
    controller = _traversal_controller(
        uncommitted_temporal_staging_hold=True,
        uncommitted_temporal_staging_terminal_release=True,
    )
    context = controller._probabilistic_traversal_window_context(
        np.zeros(controller.state_spec.dimension),
        _traversal_reference(),
        (_noncrossing_traversal_forecast(),),
        temporal_emergency_closing_observed=True,
        terminal_phase=True,
    )

    assert context["uncommitted_temporal_staging_terminal_release_active"]
    assert not context["uncommitted_temporal_staging_hold_requested"]
    assert not context["candidate_requested"]


def test_uncommitted_temporal_staging_does_not_replace_rearm_transaction():
    controller = _traversal_controller(
        uncommitted_temporal_staging_hold=True,
    )
    controller._probabilistic_traversal_rearm_pending = True
    context = controller._probabilistic_traversal_window_context(
        np.zeros(controller.state_spec.dimension),
        _traversal_reference(),
        (_noncrossing_traversal_forecast(),),
        temporal_emergency_closing_observed=True,
    )

    assert context["candidate_requested"]
    assert context["rearm_pending"]
    assert not context["uncommitted_temporal_staging_hold_requested"]


def test_feasible_uncommitted_staging_outvotes_forward_without_commit():
    controller = _traversal_controller(
        hard_violation_action="active_avoidance",
        uncommitted_temporal_staging_hold=True,
    )
    state = np.zeros(controller.state_spec.dimension)
    forecasts = (_noncrossing_traversal_forecast(),)
    context = controller._probabilistic_traversal_window_context(
        state,
        _traversal_reference(),
        forecasts,
        temporal_emergency_closing_observed=True,
    )
    context["temporal_emergency_raw_triggered"] = True
    samples = np.zeros((16, 10, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((10, 2), dtype=np.float64),
        {"triggered": True, "away_heading_error_rad": 0.0},
    )
    traversal_index = controller._inject_probabilistic_traversal_candidate(
        samples, context
    )
    candidate_risk = controller._probabilistic_collision_risk(
        controller.rollout(state, samples), forecasts
    )
    v_index = controller.action_spec.index("v_cmd")
    forward_sequence = np.zeros((10, 2), dtype=np.float64)
    forward_sequence[:, v_index] = controller.action_spec.upper[v_index]

    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            forward_sequence[0].copy(),
            forward_sequence,
            controller.rollout(state, forward_sequence)[0],
            samples,
            np.zeros(16, dtype=np.float64),
            forecasts,
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=True,
            traversal_context=context,
            traversal_candidate_index=traversal_index,
        )
    )

    assert action[v_index] == pytest.approx(0.0)
    assert diagnostics[
        "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_selected"
    ]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "traversal_uncommitted_temporal_staging_hold"
    )
    assert not controller._probabilistic_traversal_commit_started


def test_traversal_window_starts_only_for_certified_low_risk_window():
    controller = _traversal_controller()
    state = np.zeros(controller.state_spec.dimension)
    reference = _traversal_reference()

    low_risk = controller._probabilistic_traversal_window_context(
        state, reference, (_traversal_forecast(occupied=False),)
    )

    assert low_risk["window_safe"]
    assert low_risk["candidate_requested"]
    assert low_risk["commit_active"]
    assert not low_risk["commit_started"]
    assert low_risk["required_steps"] > controller.config.horizon

    controller.reset()
    high_risk = controller._probabilistic_traversal_window_context(
        state, reference, (_traversal_forecast(occupied=True),)
    )
    assert not high_risk["window_safe"]
    assert not high_risk["candidate_requested"]
    assert not high_risk["commit_active"]


def test_traversal_commit_cancels_before_entry_but_persists_after_entry():
    controller = _traversal_controller()
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    unsafe_forecast = (_traversal_forecast(occupied=True),)

    started = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    assert started["commit_active"] and not started["commit_started"]
    cancelled = controller._probabilistic_traversal_window_context(
        state, reference, unsafe_forecast
    )
    assert cancelled["commit_cancelled"]
    assert not cancelled["candidate_requested"]

    controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    state[controller.state_spec.index("x")] = 0.30
    persisted = controller._probabilistic_traversal_window_context(
        state, reference, unsafe_forecast
    )
    assert persisted["commit_active"]
    assert persisted["commit_started"]
    assert persisted["candidate_requested"]
    assert not persisted["window_safe"]

    controller.reset()
    assert controller._probabilistic_traversal_clear_progress is None


def test_traversal_candidate_replaces_one_fixed_budget_slot():
    controller = _traversal_controller()
    samples = np.zeros((16, 10, 2), dtype=np.float64)
    sequence = np.full((10, 2), (0.35, 0.0), dtype=np.float64)

    index = controller._inject_probabilistic_traversal_candidate(
        samples,
        {"candidate_requested": True, "sequence": sequence},
    )

    assert samples.shape == (16, 10, 2)
    assert index == 9
    np.testing.assert_allclose(samples[index], sequence)


def test_selected_traversal_candidate_latches_started_commit():
    controller = _traversal_controller()
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    forecasts = (_traversal_forecast(occupied=False),)
    context = controller._probabilistic_traversal_window_context(
        state, reference, forecasts
    )
    samples = np.zeros((16, 10, 2), dtype=np.float64)
    index = controller._inject_probabilistic_traversal_candidate(
        samples, context
    )
    candidate_trajectories = controller.rollout(state, samples)
    candidate_risk = controller._probabilistic_collision_risk(
        candidate_trajectories, forecasts
    )
    stop_sequence = np.zeros((10, 2), dtype=np.float64)
    stop_trajectory = controller.rollout(state, stop_sequence)[0]

    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            stop_trajectory,
            samples,
            np.zeros(16, dtype=np.float64),
            forecasts,
            candidate_risk=candidate_risk,
            traversal_context=context,
            traversal_candidate_index=index,
        )
    )

    assert action[0] == pytest.approx(0.35)
    assert controller._probabilistic_traversal_commit_started
    assert diagnostics[
        "probabilistic_obstacle_traversal_candidate_selected"
    ]
    assert diagnostics["probabilistic_obstacle_traversal_commit_started"]

    persisted = controller._probabilistic_traversal_window_context(
        state, reference, (_traversal_forecast(occupied=True),)
    )
    assert persisted["commit_started"]
    assert persisted["candidate_requested"]


def test_traversal_candidate_translates_inside_heading_gate():
    controller = _traversal_controller(translation_heading_gate_rad=0.15)
    reference = _traversal_reference()
    forecasts = (_traversal_forecast(occupied=False),)
    state = np.zeros(controller.state_spec.dimension)
    state[controller.state_spec.index("theta")] = 0.10

    aligned = controller._probabilistic_traversal_candidate(
        state, reference, forecasts, current_progress=1.0, clear_progress=2.35
    )
    assert aligned["sequence"][0, controller.action_spec.index("v_cmd")] == (
        pytest.approx(0.35)
    )

    state[controller.state_spec.index("theta")] = 0.50
    turning = controller._probabilistic_traversal_candidate(
        state, reference, forecasts, current_progress=1.0, clear_progress=2.35
    )
    assert turning["sequence"][0, controller.action_spec.index("v_cmd")] == (
        pytest.approx(0.0)
    )


def test_started_commit_aborts_on_hard_risk_before_crossing_only():
    controller = _traversal_controller(abort_probability=0.20)
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    unsafe_forecast = (_traversal_forecast(occupied=True),)
    context = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    samples = np.zeros((16, 10, 2), dtype=np.float64)
    index = controller._inject_probabilistic_traversal_candidate(samples, context)
    candidate_risk = controller._probabilistic_collision_risk(
        controller.rollout(state, samples), safe_forecast
    )
    stop_sequence = np.zeros((10, 2), dtype=np.float64)
    controller._apply_probabilistic_obstacle_action_guard(
        state,
        stop_sequence[0].copy(),
        stop_sequence,
        controller.rollout(state, stop_sequence)[0],
        samples,
        np.zeros(16, dtype=np.float64),
        safe_forecast,
        candidate_risk=candidate_risk,
        traversal_context=context,
        traversal_candidate_index=index,
    )

    aborted = controller._probabilistic_traversal_window_context(
        state, reference, unsafe_forecast
    )
    assert aborted["commit_cancelled"]
    assert not aborted["candidate_requested"]

    controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    controller._probabilistic_traversal_commit_started = True
    state[controller.state_spec.index("x")] = 0.85
    past_crossing = controller._probabilistic_traversal_window_context(
        state, reference, unsafe_forecast
    )
    assert not past_crossing["commit_cancelled"]
    assert past_crossing["candidate_requested"]


def test_aborted_commit_requests_reverse_until_staging_margin():
    controller = _traversal_controller(
        abort_probability=0.20, retreat_margin_m=0.15
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    unsafe_forecast = (_traversal_forecast(occupied=True),)
    controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    controller._probabilistic_traversal_commit_started = True
    state[controller.state_spec.index("x")] = 0.30

    retreat = controller._probabilistic_traversal_window_context(
        state, reference, unsafe_forecast
    )

    assert retreat["commit_cancelled"]
    assert retreat["retreat_requested"]
    assert retreat["candidate_requested"]
    assert retreat["sequence"][0, controller.action_spec.index("v_cmd")] == (
        pytest.approx(-0.35)
    )
    # Online route progress is deliberately monotonic and therefore remains
    # ahead while the physical robot retreats.
    reference.progress = 1.50
    held = controller._probabilistic_traversal_window_context(
        state, reference, unsafe_forecast
    )
    assert held["retreat_requested"]
    assert held["sequence"][0, controller.action_spec.index("v_cmd")] < 0.0

    state[controller.state_spec.index("x")] = -0.10
    completed = controller._probabilistic_traversal_window_context(
        state, reference, unsafe_forecast
    )
    assert completed["retreat_completed"]
    assert completed["rearm_pending"]
    assert completed["candidate_requested"]
    assert completed["sequence"][0, controller.action_spec.index("v_cmd")] == 0.0
    assert completed["current_progress"] == pytest.approx(0.90)
    assert reference.progress == pytest.approx(0.90)

    reopened = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    assert reopened["candidate_requested"]
    assert not reopened["rearm_pending"]
    assert reopened["sequence"][0, controller.action_spec.index("v_cmd")] > 0.0

    controller._probabilistic_traversal_retreat_progress = 0.75
    controller._probabilistic_traversal_rearm_pending = True
    controller.reset()
    assert controller._probabilistic_traversal_retreat_progress is None
    assert not controller._probabilistic_traversal_rearm_pending


def test_temporal_closing_aborts_pre_center_commit_before_probability_jump():
    controller = _traversal_controller(
        abort_probability=0.20, retreat_margin_m=0.15
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    controller._probabilistic_traversal_commit_started = True
    state[controller.state_spec.index("x")] = 0.30

    context = controller._probabilistic_traversal_window_context(
        state,
        reference,
        safe_forecast,
        temporal_emergency_triggered=True,
    )

    assert context["commit_cancelled"]
    assert context["commit_cancelled_by_temporal_closing"]
    assert context["retreat_requested"]
    assert context["sequence"][0, controller.action_spec.index("v_cmd")] < 0.0


def test_temporal_closing_requires_same_crossing_mass_corroboration():
    controller = _traversal_controller(
        abort_probability=0.0,
        retreat_margin_m=0.15,
        temporal_abort_mass_floor=2.0,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    controller._probabilistic_traversal_commit_started = True
    state[controller.state_spec.index("x")] = 0.30

    context = controller._probabilistic_traversal_window_context(
        state,
        reference,
        safe_forecast,
        temporal_emergency_triggered=True,
    )

    assert context["probability_mass"] < 2.0
    assert not context["commit_cancelled"]
    assert context["commit_active"]


def test_temporal_closing_can_use_same_candidate_full_horizon_corroboration(
    monkeypatch,
):
    controller = _traversal_controller(
        abort_probability=0.0,
        retreat_margin_m=0.15,
        temporal_abort_mass_floor=0.10,
        temporal_full_horizon_corroboration=True,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    controller._probabilistic_traversal_commit_started = True
    state[controller.state_spec.index("x")] = 0.30
    original = controller._probabilistic_traversal_candidate

    def delayed_hazard(*args, **kwargs):
        result = original(*args, **kwargs)
        result["maximum_probability"] = 0.01
        result["probability_mass"] = 0.01
        result["temporal_corroboration_maximum_probability"] = 0.30
        result["temporal_corroboration_probability_mass"] = 0.40
        return result

    monkeypatch.setattr(
        controller, "_probabilistic_traversal_candidate", delayed_hazard
    )
    context = controller._probabilistic_traversal_window_context(
        state,
        reference,
        safe_forecast,
        temporal_emergency_triggered=True,
    )

    assert context["maximum_probability"] == pytest.approx(0.01)
    assert context["probability_mass"] == pytest.approx(0.01)
    assert context["temporal_corroboration_probability_mass"] == pytest.approx(
        0.40
    )
    assert context["commit_cancelled"]
    assert context["commit_cancelled_by_temporal_closing"]
    assert context["retreat_requested"]


def test_new_commit_requires_same_candidate_full_horizon_certificate(
    monkeypatch,
):
    controller = _traversal_controller(commit_admission_full_horizon=True)
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    original = controller._probabilistic_traversal_candidate

    def delayed_hazard(*args, **kwargs):
        result = original(*args, **kwargs)
        result["safe"] = True
        result["maximum_probability"] = 0.01
        result["probability_mass"] = 0.05
        result["temporal_corroboration_maximum_probability"] = 0.03
        result["temporal_corroboration_probability_mass"] = 0.11
        result["commit_admission_full_horizon_safe"] = False
        return result

    monkeypatch.setattr(
        controller, "_probabilistic_traversal_candidate", delayed_hazard
    )
    rejected = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )

    assert rejected["window_safe"]
    assert not rejected["commit_admission_full_horizon_safe"]
    assert rejected["commit_admission_rejected"]
    assert not rejected["commit_active"]
    assert not rejected["candidate_requested"]

    def fully_safe(*args, **kwargs):
        result = delayed_hazard(*args, **kwargs)
        result["temporal_corroboration_maximum_probability"] = 0.01
        result["temporal_corroboration_probability_mass"] = 0.05
        result["commit_admission_full_horizon_safe"] = True
        return result

    monkeypatch.setattr(
        controller, "_probabilistic_traversal_candidate", fully_safe
    )
    admitted = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )

    assert admitted["commit_admission_full_horizon_safe"]
    assert not admitted["commit_admission_rejected"]
    assert admitted["commit_active"]
    assert admitted["candidate_requested"]


def test_commit_admission_requires_consecutive_safe_certificates():
    controller = _traversal_controller(
        commit_admission_full_horizon=True,
        commit_admission_safe_hold_steps=3,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)

    first = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    second = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    released = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )

    assert first["commit_admission_safe_streak"] == 1
    assert second["commit_admission_safe_streak"] == 2
    assert first["commit_admission_waiting"]
    assert second["commit_admission_waiting"]
    assert not first["commit_active"]
    assert not second["commit_active"]
    assert np.allclose(first["sequence"], 0.0)
    assert released["commit_admission_safe_streak"] == 3
    assert released["commit_admission_released"]
    assert released["commit_active"]
    assert not released["commit_admission_waiting"]


def test_commit_admission_prealigns_without_translating_before_release():
    controller = _traversal_controller(
        translation_heading_gate_rad=0.05,
        commit_admission_full_horizon=True,
        commit_admission_safe_hold_steps=3,
        commit_admission_prealign=True,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    state[controller.state_spec.index("theta")] = 0.5 * np.pi
    safe_forecast = (_traversal_forecast(occupied=False),)

    first = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )

    assert first["commit_admission_waiting"]
    assert first["commit_admission_prealign_requested"]
    assert first["candidate_requested"]
    assert not first["commit_active"]
    v_index = controller.action_spec.index("v_cmd")
    omega_index = controller.action_spec.index("omega_cmd")
    np.testing.assert_allclose(first["sequence"][:, v_index], 0.0)
    assert abs(first["sequence"][0, omega_index]) > 0.0


def test_commit_admission_streak_resets_on_unsafe_evidence_and_reset():
    controller = _traversal_controller(
        commit_admission_full_horizon=True,
        commit_admission_safe_hold_steps=3,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    unsafe_forecast = (_traversal_forecast(occupied=True),)

    first = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    unsafe = controller._probabilistic_traversal_window_context(
        state, reference, unsafe_forecast
    )
    restarted = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )

    assert first["commit_admission_safe_streak"] == 1
    assert unsafe["commit_admission_safe_streak"] == 0
    assert not unsafe["commit_active"]
    assert restarted["commit_admission_safe_streak"] == 1
    controller.reset()
    assert controller._probabilistic_traversal_admission_safe_streak == 0
    assert controller._probabilistic_traversal_admission_signature is None


def test_commit_admission_streak_resets_when_crossing_geometry_changes(
    monkeypatch,
):
    controller = _traversal_controller(
        commit_admission_full_horizon=True,
        commit_admission_safe_hold_steps=3,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    original = controller._probabilistic_traversal_crossing
    first = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )

    def shifted_crossing(*args, **kwargs):
        distance, progress, clearance, index, cross_track = original(
            *args, **kwargs
        )
        return (
            distance,
            progress + 0.10,
            clearance,
            index,
            cross_track,
        )

    monkeypatch.setattr(
        controller, "_probabilistic_traversal_crossing", shifted_crossing
    )
    changed = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )

    assert first["commit_admission_safe_streak"] == 1
    assert changed["commit_admission_safe_streak"] == 1
    assert changed["commit_admission_waiting"]


def test_admission_wait_hold_does_not_latch_started_commit():
    controller = _traversal_controller(
        commit_admission_full_horizon=True,
        commit_admission_safe_hold_steps=3,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    forecasts = (_traversal_forecast(occupied=False),)
    context = controller._probabilistic_traversal_window_context(
        state, reference, forecasts
    )
    samples = np.zeros((16, 10, 2), dtype=np.float64)
    index = controller._inject_probabilistic_traversal_candidate(
        samples, context
    )
    candidate_trajectories = controller.rollout(state, samples)
    candidate_risk = controller._probabilistic_collision_risk(
        candidate_trajectories, forecasts
    )
    stop_sequence = np.zeros((10, 2), dtype=np.float64)

    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            np.zeros(16, dtype=np.float64),
            forecasts,
            candidate_risk=candidate_risk,
            traversal_context=context,
            traversal_candidate_index=index,
        )
    )

    assert np.allclose(action, 0.0)
    assert not controller._probabilistic_traversal_commit_started
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "traversal_admission_stability_hold"
    )
    assert diagnostics[
        "probabilistic_obstacle_traversal_commit_admission_waiting"
    ]


def test_active_commit_is_not_reheld_by_admission_stability():
    controller = _traversal_controller(
        commit_admission_full_horizon=True,
        commit_admission_safe_hold_steps=3,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    for _ in range(3):
        released = controller._probabilistic_traversal_window_context(
            state, reference, safe_forecast
        )
    assert released["commit_admission_released"]

    active = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )

    assert active["commit_active"]
    assert not active["commit_admission_waiting"]
    assert active["sequence"][0, controller.action_spec.index("v_cmd")] > 0.0


def test_preentry_commit_is_cancelled_when_full_horizon_becomes_unsafe(
    monkeypatch,
):
    controller = _traversal_controller(commit_admission_full_horizon=True)
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    admitted = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    assert admitted["commit_active"] and not admitted["commit_started"]
    original = controller._probabilistic_traversal_candidate

    def delayed_hazard(*args, **kwargs):
        result = original(*args, **kwargs)
        result["safe"] = True
        result["commit_admission_full_horizon_safe"] = False
        return result

    monkeypatch.setattr(
        controller, "_probabilistic_traversal_candidate", delayed_hazard
    )
    cancelled = controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )

    assert cancelled["commit_cancelled"]
    assert cancelled["commit_admission_rejected"]
    assert not cancelled["commit_active"]


def test_temporal_abort_preserves_commit_when_safe_forward_exit_is_nearer(
    monkeypatch,
):
    controller = _traversal_controller(
        retreat_margin_m=0.15,
        temporal_abort_mass_floor=0.10,
        temporal_full_horizon_corroboration=True,
        temporal_abort_nearest_exit=True,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    controller._probabilistic_traversal_commit_started = True
    state[controller.state_spec.index("x")] = 0.75
    original = controller._probabilistic_traversal_candidate

    def delayed_hazard(*args, **kwargs):
        result = original(*args, **kwargs)
        result["safe"] = True
        result["maximum_probability"] = 0.01
        result["probability_mass"] = 0.01
        result["temporal_corroboration_maximum_probability"] = 0.30
        result["temporal_corroboration_probability_mass"] = 0.40
        return result

    monkeypatch.setattr(
        controller, "_probabilistic_traversal_candidate", delayed_hazard
    )
    context = controller._probabilistic_traversal_window_context(
        state,
        reference,
        safe_forecast,
        temporal_emergency_triggered=True,
    )

    assert context["forward_exit_distance_m"] <= context[
        "retreat_exit_distance_m"
    ]
    assert context["commit_preserved_for_nearest_safe_exit"]
    assert not context["commit_cancelled"]
    assert not context["retreat_requested"]
    assert context["commit_active"]
    assert context["commit_started"]


def test_raw_temporal_midpoint_guard_aborts_before_reverse_exit_loses(
    monkeypatch,
):
    controller = _traversal_controller(
        retreat_margin_m=0.15,
        temporal_abort_mass_floor=1.0,
        temporal_full_horizon_corroboration=True,
        temporal_midpoint_guard=True,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    controller._probabilistic_traversal_commit_started = True
    state[controller.state_spec.index("x")] = 0.72
    original = controller._probabilistic_traversal_candidate

    def delayed_hazard(*args, **kwargs):
        result = original(*args, **kwargs)
        result["safe"] = True
        result["maximum_probability"] = 0.01
        result["probability_mass"] = 0.01
        result["temporal_corroboration_maximum_probability"] = 0.03
        result["temporal_corroboration_probability_mass"] = 0.40
        return result

    monkeypatch.setattr(
        controller, "_probabilistic_traversal_candidate", delayed_hazard
    )
    context = controller._probabilistic_traversal_window_context(
        state,
        reference,
        safe_forecast,
        temporal_emergency_triggered=True,
    )

    assert context["retreat_exit_distance_m"] <= context[
        "forward_exit_distance_m"
    ]
    assert context["forward_exit_distance_m"] - context[
        "retreat_exit_distance_m"
    ] <= controller.action_spec.upper[
        controller.action_spec.index("v_cmd")
    ] * controller.config.dt
    assert context["commit_cancelled"]
    assert context["commit_cancelled_by_temporal_midpoint_guard"]
    assert context["retreat_requested"]
    assert context["retreat_temporal_lattice_requested"]

def test_raw_temporal_exit_deadline_aborts_when_neither_exit_can_clear():
    controller = _traversal_controller(
        retreat_margin_m=0.15,
        temporal_abort_mass_floor=1.0,
        temporal_exit_deadline_guard=True,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)
    controller._probabilistic_traversal_window_context(
        state, reference, safe_forecast
    )
    controller._probabilistic_traversal_commit_started = True
    state[controller.state_spec.index("x")] = 0.50

    context = controller._probabilistic_traversal_window_context(
        state,
        reference,
        safe_forecast,
        temporal_emergency_triggered=True,
        temporal_emergency_raw_triggered=True,
        temporal_emergency_ttc_s=0.80,
    )

    assert min(
        context["forward_exit_optimistic_time_s"],
        context["retreat_exit_optimistic_time_s"],
    ) >= context["temporal_exit_deadline_ttc_s"]
    assert context["temporal_exit_deadline_guard_triggered"]
    assert context["commit_cancelled"]
    assert context["commit_cancelled_by_temporal_closing"]
    assert context["commit_cancelled_by_temporal_exit_deadline_guard"]
    assert not context["commit_cancelled_by_temporal_midpoint_guard"]
    assert context["retreat_requested"]
    assert context["retreat_temporal_lattice_requested"]

    intent_only = _traversal_controller(
        retreat_margin_m=0.15,
        temporal_abort_mass_floor=1.0,
        temporal_exit_deadline_guard=True,
    )
    intent_only._probabilistic_traversal_window_context(
        np.zeros(intent_only.state_spec.dimension),
        reference,
        safe_forecast,
    )
    intent_only._probabilistic_traversal_commit_started = True
    intent_state = np.zeros(intent_only.state_spec.dimension)
    intent_state[intent_only.state_spec.index("x")] = 0.50
    intent_context = intent_only._probabilistic_traversal_window_context(
        intent_state,
        reference,
        safe_forecast,
        temporal_emergency_triggered=True,
        temporal_emergency_raw_triggered=False,
        temporal_emergency_ttc_s=0.80,
    )
    assert not intent_context["temporal_exit_deadline_guard_triggered"]
    assert not intent_context[
        "commit_cancelled_by_temporal_exit_deadline_guard"
    ]


def test_matched_closing_deadline_holds_commit_admission_until_feasible():
    controller = _traversal_controller(
        commit_admission_exit_deadline=True,
        commit_admission_safe_hold_steps=2,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    safe_forecast = (_traversal_forecast(occupied=False),)

    blocked = controller._probabilistic_traversal_window_context(
        state,
        reference,
        safe_forecast,
        temporal_emergency_closing_observed=True,
        temporal_emergency_ttc_s=0.80,
    )

    assert blocked["commit_admission_exit_deadline_rejected"]
    assert blocked["commit_admission_exit_deadline_hold_requested"]
    assert not blocked["commit_admission_exit_deadline_safe"]
    assert blocked["commit_admission_exit_deadline_margin_s"] < 0.0
    assert blocked["commit_admission_rejected"]
    assert blocked["commit_admission_waiting"]
    assert not blocked["commit_admission_released"]
    assert blocked["commit_admission_safe_streak"] == 0
    assert blocked["candidate_requested"]
    np.testing.assert_allclose(blocked["sequence"], 0.0)

    mismatch_first = controller._probabilistic_traversal_window_context(
        state,
        reference,
        safe_forecast,
        temporal_emergency_closing_observed=False,
        temporal_emergency_ttc_s=0.80,
    )
    mismatch_second = controller._probabilistic_traversal_window_context(
        state,
        reference,
        safe_forecast,
        temporal_emergency_closing_observed=False,
        temporal_emergency_ttc_s=0.80,
    )

    assert mismatch_first["commit_admission_waiting"]
    assert mismatch_first["commit_admission_safe_streak"] == 1
    assert not mismatch_first["commit_admission_exit_deadline_rejected"]
    assert mismatch_second["commit_admission_released"]
    assert mismatch_second["commit_active"]


def test_deadline_rejection_holds_unsafe_admission_after_emergency_intent_expires():
    controller = _traversal_controller(
        commit_admission_full_horizon=True,
        commit_admission_exit_deadline=True,
        commit_admission_safe_hold_steps=2,
    )
    reference = _traversal_reference()
    state = np.zeros(controller.state_spec.dimension)
    unsafe_forecast = (_traversal_forecast(occupied=True),)
    context = controller._probabilistic_traversal_window_context(
        state,
        reference,
        unsafe_forecast,
        temporal_emergency_closing_observed=True,
        temporal_emergency_ttc_s=0.80,
    )

    assert not context["commit_admission_waiting"]
    assert context["commit_admission_exit_deadline_rejected"]
    assert context["commit_admission_exit_deadline_hold_requested"]
    assert context["commit_admission_safe_streak"] == 0
    assert context["candidate_requested"]
    np.testing.assert_allclose(context["sequence"], 0.0)

    controller._probabilistic_emergency_intent_remaining = 0
    samples = np.zeros((16, 10, 2), dtype=np.float64)
    index = controller._inject_probabilistic_traversal_candidate(samples, context)
    candidate_risk = controller._probabilistic_collision_risk(
        controller.rollout(state, samples), unsafe_forecast
    )
    forward_sequence = np.zeros((10, 2), dtype=np.float64)
    forward_sequence[:, controller.action_spec.index("v_cmd")] = 0.35
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            forward_sequence[0].copy(),
            forward_sequence,
            controller.rollout(state, forward_sequence)[0],
            samples,
            np.zeros(16, dtype=np.float64),
            unsafe_forecast,
            candidate_risk=candidate_risk,
            traversal_context=context,
            traversal_candidate_index=index,
        )
    )

    np.testing.assert_allclose(action, 0.0)
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "traversal_admission_exit_deadline_hold"
    )
    assert diagnostics[
        "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_requested"
    ]
    assert not controller._probabilistic_traversal_commit_started

    controller.reset(seed=730100253)
    released_first = controller._probabilistic_traversal_window_context(
        state,
        reference,
        (_traversal_forecast(occupied=False),),
        temporal_emergency_closing_observed=False,
        temporal_emergency_ttc_s=0.80,
    )
    released_second = controller._probabilistic_traversal_window_context(
        state,
        reference,
        (_traversal_forecast(occupied=False),),
        temporal_emergency_closing_observed=False,
        temporal_emergency_ttc_s=0.80,
    )
    assert not released_first["commit_admission_exit_deadline_hold_requested"]
    assert released_first["commit_admission_waiting"]
    assert released_second["commit_admission_released"]
    assert released_second["commit_active"]


def test_hard_risk_deadline_hold_filters_forward_lattice_members(monkeypatch):
    controller = _traversal_controller(
        commit_admission_full_horizon=True,
        commit_admission_exit_deadline=True,
        hard_violation_action="active_avoidance",
    )
    state = np.zeros(controller.state_spec.dimension)
    forecasts = (_traversal_forecast(occupied=False),)
    samples = np.zeros((16, 10, 2), dtype=np.float64)
    traversal_index = 9
    emergency_indices = np.arange(10, 16, dtype=np.int64)
    v_index = controller.action_spec.index("v_cmd")
    omega_index = controller.action_spec.index("omega_cmd")
    patterns = (
        (0.35, 0.0),
        (-0.35, 0.0),
        (0.35, 0.9),
        (0.35, -0.9),
        (-0.35, 0.9),
        (-0.35, -0.9),
    )
    for index, pattern in zip(emergency_indices, patterns):
        samples[index, :3, v_index] = pattern[0]
        samples[index, :3, omega_index] = pattern[1]
    emergency_mask = np.zeros(16, dtype=bool)
    emergency_mask[emergency_indices] = True
    hard_violation = np.zeros(16, dtype=bool)
    hard_violation[traversal_index] = True
    hard_violation[10] = True
    maximum_probability = np.zeros(16, dtype=np.float64)
    maximum_probability[emergency_indices] = (0.30, 0.12, 0.05, 0.06, 0.08, 0.10)
    probability_mass = np.zeros(16, dtype=np.float64)
    probability_mass[emergency_indices] = (3.0, 1.2, 0.5, 0.6, 0.8, 1.0)
    candidate_risk = SimpleNamespace(
        hard_violation=hard_violation,
        maximum_step_probability=maximum_probability,
        accumulated_probability_mass=probability_mass,
    )

    def single_safe_risk(trajectories, _forecasts):
        count = np.asarray(trajectories).shape[0]
        return SimpleNamespace(
            hard_violation=np.zeros(count, dtype=bool),
            maximum_step_probability=np.zeros(count, dtype=np.float64),
            accumulated_probability_mass=np.zeros(count, dtype=np.float64),
            horizon_union_bound=np.zeros(count, dtype=np.float64),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", single_safe_risk
    )
    context = {
        "candidate_requested": True,
        "commit_started": False,
        "commit_admission_waiting": False,
        "commit_admission_exit_deadline_hold_requested": True,
        "retreat_requested": False,
        "rearm_pending": False,
    }
    forward_sequence = np.zeros((10, 2), dtype=np.float64)
    forward_sequence[:, v_index] = 0.35

    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            forward_sequence[0].copy(),
            forward_sequence,
            controller.rollout(state, forward_sequence)[0],
            samples,
            np.zeros(16, dtype=np.float64),
            forecasts,
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            traversal_context=context,
            traversal_candidate_index=traversal_index,
        )
    )

    np.testing.assert_allclose(action, samples[14, 0])
    assert action[v_index] < 0.0
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "admission_exit_deadline_hold_hard_risk_emergency_candidate"
    )
    assert diagnostics[
        "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_hard_risk_override"
    ]
    assert diagnostics[
        "probabilistic_obstacle_traversal_commit_admission_exit_deadline_forward_lattice_filtered"
    ]
    assert diagnostics["probabilistic_obstacle_emergency_candidate_count"] == 6
    assert not controller._probabilistic_traversal_commit_started


def test_exit_deadline_retreat_escape_outlives_intent_and_raw_trigger():
    controller = _traversal_controller(
        retreat_margin_m=0.15,
        temporal_exit_deadline_guard=True,
        temporal_exit_deadline_escape_latch=True,
    )
    state = np.zeros(controller.state_spec.dimension, dtype=np.float64)
    state[controller.state_spec.index("theta")] = 0.20
    controller._start_probabilistic_traversal_exit_deadline_retreat_escape()
    assert controller._latch_probabilistic_traversal_exit_deadline_retreat_escape(
        (-0.35, 0.90), state
    )
    assert not controller._latch_probabilistic_traversal_exit_deadline_retreat_escape(
        (0.35, 0.0), state
    )

    controller._probabilistic_emergency_intent_remaining = 0
    controller._probabilistic_emergency_latched_pattern = None
    next_state = state.copy()
    next_state[controller.state_spec.index("theta")] = 0.29
    emergency_context, traversal_context = (
        controller._bind_probabilistic_traversal_exit_deadline_retreat_escape(
            {"triggered": False, "raw_triggered": True},
            {
                "retreat_requested": True,
                "retreat_temporal_lattice_requested": True,
                "temporal_emergency_raw_triggered": True,
            },
            next_state,
        )
    )

    assert traversal_context[
        "exit_deadline_retreat_escape_transaction_active"
    ]
    assert traversal_context["exit_deadline_retreat_escape_reused"]
    assert emergency_context["latched_escape_pattern"][0] == -0.35
    assert 0.0 < emergency_context["latched_escape_pattern"][1] < 0.90
    assert controller._probabilistic_emergency_latched_pattern is None

    persisted_emergency_context, persisted_context = (
        controller._bind_probabilistic_traversal_exit_deadline_retreat_escape(
            {"triggered": False, "raw_triggered": False},
            {
                "retreat_requested": True,
                "retreat_temporal_lattice_requested": True,
                "temporal_emergency_raw_triggered": False,
            },
            next_state,
        )
    )
    assert persisted_context[
        "exit_deadline_retreat_escape_transaction_active"
    ]
    assert persisted_context["exit_deadline_retreat_escape_reused"]
    assert persisted_emergency_context["latched_escape_pattern"][0] == -0.35
    assert controller._probabilistic_traversal_exit_deadline_retreat_pattern == (
        -0.35,
        0.90,
    )

    _, cleared_context = (
        controller._bind_probabilistic_traversal_exit_deadline_retreat_escape(
            {"triggered": False, "raw_triggered": False},
            {
                "retreat_requested": False,
                "retreat_temporal_lattice_requested": False,
                "temporal_emergency_raw_triggered": False,
            },
            next_state,
        )
    )
    assert not cleared_context[
        "exit_deadline_retreat_escape_transaction_active"
    ]
    assert (
        controller._probabilistic_traversal_exit_deadline_retreat_pattern
        is None
    )

    controller._start_probabilistic_traversal_exit_deadline_retreat_escape()
    assert controller._latch_probabilistic_traversal_exit_deadline_retreat_escape(
        (-0.35, 0.90), state
    )
    controller._probabilistic_traversal_retreat_progress = 0.20
    controller._probabilistic_traversal_retreat_temporal_lattice = True
    completed_state = np.zeros(controller.state_spec.dimension)
    completed_state[controller.state_spec.index("x")] = -0.90
    completed = controller._probabilistic_traversal_window_context(
        completed_state,
        _traversal_reference(),
        (_traversal_forecast(occupied=False),),
    )
    assert completed["retreat_completed"]
    assert not controller._probabilistic_traversal_exit_deadline_retreat_active
    assert (
        controller._probabilistic_traversal_exit_deadline_retreat_pattern
        is None
    )

    controller._start_probabilistic_traversal_exit_deadline_retreat_escape()
    assert controller._latch_probabilistic_traversal_exit_deadline_retreat_escape(
        (-0.35, 0.90), state
    )
    controller.reset(seed=7)
    assert not controller._probabilistic_traversal_exit_deadline_retreat_active
    assert (
        controller._probabilistic_traversal_exit_deadline_retreat_pattern
        is None
    )
    assert (
        controller._probabilistic_traversal_exit_deadline_retreat_heading
        is None
    )


def test_post_retreat_rearm_holds_when_forecast_drops_out():
    controller = _traversal_controller(
        abort_probability=0.20, retreat_margin_m=0.15
    )
    controller._probabilistic_traversal_rearm_pending = True

    context = controller._probabilistic_traversal_window_context(
        np.zeros(controller.state_spec.dimension), _traversal_reference(), ()
    )

    assert context["rearm_pending"]
    assert context["candidate_requested"]
    np.testing.assert_allclose(context["sequence"], 0.0)


def test_post_retreat_rearm_staging_approaches_and_stops_at_entry():
    controller = _traversal_controller(
        commit_admission_full_horizon=True,
        commit_admission_safe_hold_steps=3,
        rearm_staging_approach=True,
    )
    controller._probabilistic_traversal_rearm_pending = True
    state = np.zeros(controller.state_spec.dimension)
    reference = _traversal_reference()

    context = controller._probabilistic_traversal_window_context(
        state,
        reference,
        (_traversal_forecast(occupied=False),),
    )

    assert context["rearm_pending"]
    assert context["rearm_staging_approach_requested"]
    assert context["rearm_staging_approach_safe"]
    assert controller._probabilistic_traversal_rearm_pending
    v_index = controller.action_spec.index("v_cmd")
    assert context["sequence"][0, v_index] > 0.0
    trajectory = controller.rollout(state, context["sequence"])[0]
    positions = trajectory[1:, list(controller.state_spec.position_indices)]
    progress = reference.project_batch(
        positions, minimum_progress=context["current_progress"]
    ).progress
    entry_progress = context["entry_progress"]
    reached = np.flatnonzero(progress >= entry_progress - 1.0e-9)
    assert reached.size
    assert np.max(progress) <= entry_progress + 1.0e-9
    np.testing.assert_allclose(
        context["sequence"][int(reached[0]) + 1:, v_index], 0.0
    )


def test_disabled_rearm_staging_approach_preserves_hold():
    controller = _traversal_controller(
        commit_admission_full_horizon=True,
        commit_admission_safe_hold_steps=3,
        rearm_staging_approach=False,
    )
    controller._probabilistic_traversal_rearm_pending = True

    context = controller._probabilistic_traversal_window_context(
        np.zeros(controller.state_spec.dimension),
        _traversal_reference(),
        (_traversal_forecast(occupied=False),),
    )

    assert context["rearm_pending"]
    assert not context["rearm_staging_approach_requested"]
    assert not context["rearm_staging_approach_safe"]
    np.testing.assert_allclose(context["sequence"], 0.0)


def test_post_retreat_rearm_releases_after_stable_causal_no_crossing_clearance():
    controller = _traversal_controller(
        abort_probability=0.20,
        retreat_margin_m=0.15,
        rearm_no_crossing_clear=True,
        commit_admission_safe_hold_steps=2,
    )
    controller._probabilistic_traversal_rearm_pending = True
    state = np.zeros(controller.state_spec.dimension)
    reference = _traversal_reference()
    means = np.zeros((80, 1, 2), dtype=np.float64)
    means[:, 0, 0] = 0.8
    means[:, 0, 1] = 2.0
    off_path_forecast = (_forecast(
        means, covariance=1.0e-6, radius=0.15
    ),)

    closing = controller._probabilistic_traversal_window_context(
        state,
        reference,
        off_path_forecast,
        temporal_emergency_closing_observed=True,
    )
    assert closing["rearm_pending"]
    assert closing["rearm_no_crossing_safe_streak"] == 0

    first_clear = controller._probabilistic_traversal_window_context(
        state, reference, off_path_forecast
    )
    assert first_clear["rearm_pending"]
    assert first_clear["candidate_requested"]
    assert first_clear["rearm_no_crossing_safe_streak"] == 1
    np.testing.assert_allclose(first_clear["sequence"], 0.0)

    dropout = controller._probabilistic_traversal_window_context(
        state, reference, ()
    )
    assert dropout["rearm_pending"]
    assert dropout["candidate_requested"]

    fresh_clear = controller._probabilistic_traversal_window_context(
        state, reference, off_path_forecast
    )
    assert fresh_clear["rearm_pending"]
    assert fresh_clear["rearm_no_crossing_safe_streak"] == 1

    released = controller._probabilistic_traversal_window_context(
        state, reference, off_path_forecast
    )
    assert not released["rearm_pending"]
    assert not released["candidate_requested"]
    assert released["sequence"] is None
    assert released["rearm_no_crossing_safe_streak"] == 2
    assert released["rearm_released_by_no_crossing_clearance"]
    assert not controller._probabilistic_traversal_rearm_pending


def test_post_retreat_no_crossing_certified_handoff_keeps_rearm_authority():
    controller = _traversal_controller(
        abort_probability=0.20,
        retreat_margin_m=0.15,
        rearm_no_crossing_clear=True,
        rearm_no_crossing_certified_handoff=True,
        commit_admission_full_horizon=True,
        commit_admission_safe_hold_steps=2,
    )
    controller._probabilistic_traversal_rearm_pending = True
    state = np.zeros(controller.state_spec.dimension)
    reference = _traversal_reference()
    means = np.zeros((80, 1, 2), dtype=np.float64)
    means[:, 0, 0] = 0.8
    means[:, 0, 1] = 2.0
    off_path_forecast = (_forecast(
        means, covariance=1.0e-6, radius=0.15
    ),)

    first_clear = controller._probabilistic_traversal_window_context(
        state, reference, off_path_forecast
    )
    handoff = controller._probabilistic_traversal_window_context(
        state, reference, off_path_forecast
    )

    assert first_clear["rearm_pending"]
    assert first_clear["rearm_no_crossing_safe_streak"] == 1
    assert first_clear["rearm_no_crossing_certified_handoff_safe"]
    assert not first_clear["rearm_no_crossing_certified_handoff_active"]
    np.testing.assert_allclose(first_clear["sequence"], 0.0)
    assert handoff["rearm_pending"]
    assert handoff["candidate_requested"]
    assert handoff["rearm_no_crossing_safe_streak"] == 2
    assert handoff["rearm_no_crossing_certified_handoff_active"]
    assert handoff["rearm_no_crossing_certified_handoff_safe"]
    assert not handoff["rearm_released_by_no_crossing_clearance"]
    assert controller._probabilistic_traversal_rearm_pending
    v_index = controller.action_spec.index("v_cmd")
    assert handoff["sequence"][0, v_index] > 0.0
    samples = np.zeros((16, 10, 2), dtype=np.float64)
    handoff_index = controller._inject_probabilistic_traversal_candidate(
        samples, handoff
    )
    assert handoff_index == 9
    np.testing.assert_allclose(samples[handoff_index], handoff["sequence"])

    closing = controller._probabilistic_traversal_window_context(
        state,
        reference,
        off_path_forecast,
        temporal_emergency_closing_observed=True,
    )
    assert closing["rearm_pending"]
    assert closing["rearm_no_crossing_safe_streak"] == 0
    assert not closing["rearm_no_crossing_certified_handoff_active"]
    np.testing.assert_allclose(closing["sequence"], 0.0)

    crossing_forecast = (_traversal_forecast(occupied=False),)
    admission_first = controller._probabilistic_traversal_window_context(
        state, reference, crossing_forecast
    )
    admission_second = controller._probabilistic_traversal_window_context(
        state, reference, crossing_forecast
    )
    assert admission_first["rearm_pending"]
    assert admission_first["commit_admission_safe_streak"] == 1
    assert not admission_first["commit_active"]
    np.testing.assert_allclose(admission_first["sequence"], 0.0)
    assert admission_second["commit_admission_released"]
    assert admission_second["commit_active"]
    assert not admission_second["rearm_pending"]
    assert not controller._probabilistic_traversal_rearm_pending


def test_prediction_adapter_preserves_imm_components():
    prediction = SimpleNamespace(
        component_means=np.zeros((3, 4, 4)),
        component_covariances=np.repeat(
            np.eye(4)[None, None, :, :], 12, axis=0
        ).reshape(3, 4, 4, 4),
        mode_probabilities=np.full((3, 4), 0.25),
    )
    result = GaussianMixtureObstacleForecast.from_prediction(
        prediction, timestamp=2.0, dt=0.1, radius_m=0.2, source="change"
    )
    assert result.component_means.shape == (3, 4, 2)
    assert result.component_covariances.shape == (3, 4, 2, 2)
    assert result.mode_count == 4


def test_public_risk_api_has_no_truth_or_change_label_inputs():
    forbidden = {
        "truth",
        "truth_states",
        "true_modes",
        "change_flags",
        "events",
    }
    parameters = set(inspect.signature(evaluate_collision_risk).parameters)
    fields = set(GaussianMixtureObstacleForecast.__dataclass_fields__)
    assert forbidden.isdisjoint(parameters)
    assert forbidden.isdisjoint(fields)


def test_tangent_halfspace_bound_dominates_seeded_monte_carlo_cases():
    rng = np.random.RandomState(730199990)
    config = _risk_config()
    robot = np.zeros((1, 1, 2))
    angle = 0.55
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle)),
            (np.sin(angle), np.cos(angle)),
        )
    )
    cases = (
        (np.asarray((0.70, 0.0)), np.diag((0.04, 0.01))),
        (np.asarray((0.85, 0.10)), np.diag((0.09, 0.02))),
        (
            np.asarray((0.75, -0.20)),
            rotation.dot(np.diag((0.06, 0.01))).dot(rotation.T),
        ),
        (
            np.asarray((1.00, 0.25)),
            rotation.dot(np.diag((0.12, 0.03))).dot(rotation.T),
        ),
    )
    combined_radius = (
        config.robot_radius_m + 0.15 + config.safety_margin_m
    )
    violations = 0
    sample_count = 120000
    for mean, covariance in cases:
        forecast = _forecast(
            [[mean]], covariance=covariance, radius=0.15
        )
        bound = mixture_collision_probability_upper_bound(
            robot, forecast, config
        )[0, 0]
        samples = rng.multivariate_normal(
            mean, covariance, size=sample_count
        )
        empirical = float(
            np.mean(np.linalg.norm(samples, axis=1) <= combined_radius)
        )
        standard_error = np.sqrt(
            max(empirical * (1.0 - empirical), 1.0 / sample_count)
            / sample_count
        )
        violations += int(
            empirical > bound + 5.0 * standard_error + 1.0 / sample_count
        )
    assert violations == 0


def test_mppi_adapter_is_identical_when_disabled_and_penalizes_crossing():
    horizon = 3
    action = body_velocity_action((0.0, 0.5), 1.0)
    disabled = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        action,
        MppiConfig(
            horizon=horizon,
            num_samples=4,
            dt=0.1,
            noise_sigma=(0.1, 0.2),
            probabilistic_obstacle_risk_enabled=False,
        ),
    )
    enabled = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        action,
        MppiConfig(
            horizon=horizon,
            num_samples=4,
            dt=0.1,
            noise_sigma=(0.1, 0.2),
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_risk_weight=75.0,
            probabilistic_obstacle_hard_threshold=0.20,
            probabilistic_obstacle_hard_penalty=10000.0,
        ),
    )
    trajectories = np.asarray(
        (
            (
                (0.0, 0.0, 0.0),
                (0.3, 0.0, 0.0),
                (0.6, 0.0, 0.0),
                (0.9, 0.0, 0.0),
            ),
            (
                (0.0, 1.2, 0.0),
                (0.3, 1.2, 0.0),
                (0.6, 1.2, 0.0),
                (0.9, 1.2, 0.0),
            ),
        )
    )
    controls = np.zeros((2, horizon, 2))
    forecast = _forecast(
        [[[0.3, 0.0]], [[0.6, 0.0]], [[0.9, 0.0]]],
        covariance=np.diag((0.01, 0.01)),
        radius=0.15,
    )
    target = PointGoal(2.0, 0.0).target_at(0.0, trajectories[0, 0])
    baseline = disabled.cost_trajectories(
        trajectories, controls, target
    )
    ignored = disabled.cost_trajectories(
        trajectories,
        controls,
        target,
        probabilistic_obstacles=(forecast,),
    )
    np.testing.assert_array_equal(baseline, ignored)

    probabilistic = enabled.cost_trajectories(
        trajectories,
        controls,
        target,
        probabilistic_obstacles=(forecast,),
    )
    assert probabilistic[0] > probabilistic[1] + 9000.0


def test_enabled_mppi_fails_closed_on_missing_or_misaligned_forecast():
    horizon = 3
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=horizon,
            num_samples=4,
            dt=0.1,
            noise_sigma=(0.1, 0.2),
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_risk_weight=75.0,
        ),
    )
    missing = RobotObservation(
        0.0, Pose2D(0.0, 0.0, 0.0), Twist2D(0.0, 0.0)
    )
    with pytest.raises(ValueError, match="contains no forecasts"):
        controller.plan(missing, PointGoal(2.0, 0.0))

    online_controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=horizon,
            num_samples=4,
            dt=0.1,
            noise_sigma=(0.1, 0.2),
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_risk_weight=75.0,
            probabilistic_obstacle_missing_forecast_action="stop",
        ),
    )
    stopped = online_controller.plan(
        missing, PointGoal(2.0, 0.0)
    )
    np.testing.assert_array_equal(
        stopped.proposed_control.values, np.zeros(2)
    )
    np.testing.assert_array_equal(
        stopped.control_sequence, np.zeros((horizon, 2))
    )
    assert stopped.diagnostics[
        "probabilistic_obstacle_fail_closed"
    ]
    assert stopped.diagnostics[
        "probabilistic_obstacle_forecast_count"
    ] == 0

    forecast = _forecast(
        np.zeros((horizon, 1, 2)), timestamp=0.1
    )
    misaligned = RobotObservation(
        0.0,
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.0, 0.0),
        auxiliary={"probabilistic_obstacle_forecasts": (forecast,)},
    )
    with pytest.raises(ValueError, match="timestamp must match"):
        controller.plan(misaligned, PointGoal(2.0, 0.0))

    wrong_dt = _forecast(
        np.zeros((horizon, 1, 2)), timestamp=0.0, dt=0.2
    )
    trajectories = np.zeros((1, horizon + 1, 3))
    controls = np.zeros((1, horizon, 2))
    target = PointGoal(2.0, 0.0).target_at(0.0, trajectories[0, 0])
    with pytest.raises(ValueError, match="dt must match"):
        controller.cost_trajectories(
            trajectories,
            controls,
            target,
            probabilistic_obstacles=(wrong_dt,),
        )


def test_enabled_mppi_can_fail_closed_on_selected_hard_violation():
    horizon = 3
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            noise_sigma=(0.1, 0.2),
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_risk_weight=75.0,
            probabilistic_obstacle_hard_threshold=0.20,
            probabilistic_obstacle_hard_penalty=10000.0,
            probabilistic_obstacle_hard_violation_action="stop",
            seed=3,
        ),
    )
    forecast = _forecast(
        np.zeros((horizon, 1, 2)),
        covariance=np.diag((1.0e-4, 1.0e-4)),
        radius=0.20,
    )
    observation = RobotObservation(
        0.0,
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.2, 0.0),
        auxiliary={
            "probabilistic_obstacle_forecasts": (forecast,)
        },
    )
    stopped = controller.plan(
        observation, PointGoal(2.0, 0.0)
    )
    np.testing.assert_array_equal(
        stopped.proposed_control.values, np.zeros(2)
    )
    np.testing.assert_array_equal(
        stopped.control_sequence, np.zeros((horizon, 2))
    )
    assert stopped.diagnostics[
        "probabilistic_obstacle_hard_violation"
    ]
    assert stopped.diagnostics[
        "probabilistic_obstacle_fail_closed"
    ]


def test_active_avoidance_filters_stop_and_selects_safe_motion():
    horizon = 3
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            noise_sigma=(0.1, 0.2),
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_risk_weight=75.0,
            probabilistic_obstacle_hard_threshold=0.20,
            probabilistic_obstacle_hard_penalty=10000.0,
            probabilistic_obstacle_candidate_filter_enabled=True,
            probabilistic_obstacle_hard_violation_action=(
                "active_avoidance"
            ),
            seed=4,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    samples[0, :, 0] = 0.5
    controller._sample = lambda prior, rng=None: samples.copy()
    forecast = _forecast(
        np.repeat([[[-0.49, 0.0]]], horizon, axis=0),
        covariance=np.diag((1.0e-4, 1.0e-4)),
        radius=0.15,
    )
    observation = RobotObservation(
        0.0,
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.0, 0.0),
        auxiliary={
            "probabilistic_obstacle_forecasts": (forecast,)
        },
    )
    plan = controller.plan(
        observation, PointGoal(2.0, 0.0)
    )
    assert plan.proposed_control.values[0] > 0.45
    assert not plan.diagnostics[
        "probabilistic_obstacle_hard_violation"
    ]
    assert plan.diagnostics[
        "probabilistic_obstacle_candidate_feasible_fraction"
    ] < 1.0


def test_active_avoidance_motion_is_not_zeroed_by_speed_governor():
    horizon = 3
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=horizon,
            num_samples=2,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action=(
                "active_avoidance_motion"
            ),
            probabilistic_obstacle_speed_governor_enabled=True,
            probabilistic_obstacle_stopping_feasibility_enabled=True,
        ),
    )
    samples = np.zeros((2, horizon, 2), dtype=np.float64)
    samples[1, :, 0] = 0.5
    candidate_risk = SimpleNamespace(
        hard_violation=np.asarray((True, True)),
        maximum_step_probability=np.asarray((1.0, 1.0)),
        accumulated_probability_mass=np.asarray((3.0, 2.0)),
        horizon_union_bound=np.asarray((1.0, 1.0)),
    )

    def risk_for_trajectory(trajectories, _forecasts):
        moving = bool(np.asarray(trajectories)[0, 1, 0] > 0.0)
        mass = 2.0 if moving else 3.0
        return SimpleNamespace(
            hard_violation=np.asarray((True,)),
            maximum_step_probability=np.asarray((1.0,)),
            accumulated_probability_mass=np.asarray((mass,)),
            horizon_union_bound=np.asarray((1.0,)),
        )

    controller._probabilistic_collision_risk = risk_for_trajectory
    stop_sequence = samples[0].copy()
    stop_trajectory = controller.rollout(
        np.zeros(controller.state_spec.dimension), stop_sequence
    )[0]
    action, sequence, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            np.zeros(controller.state_spec.dimension),
            stop_sequence[0].copy(),
            stop_sequence,
            stop_trajectory,
            samples,
            np.asarray((0.0, 1.0)),
            (object(),),
            candidate_risk=candidate_risk,
        )
    )

    assert action[0] > 0.0
    assert sequence[0, 0] > 0.0
    assert diagnostics[
        "probabilistic_obstacle_speed_governor_bypassed_for_active_avoidance"
    ]


def test_active_avoidance_uses_probability_mass_when_maximum_risk_saturates(
    monkeypatch,
):
    horizon = 3
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=horizon,
            num_samples=2,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_stopping_feasibility_enabled=True,
        ),
    )
    samples = np.zeros((2, horizon, 2), dtype=np.float64)
    samples[1, :, 0] = 0.5
    candidate_risk = SimpleNamespace(
        hard_violation=np.asarray((True, True)),
        maximum_step_probability=np.asarray((1.0, 1.0)),
        accumulated_probability_mass=np.asarray((3.0, 2.0)),
        horizon_union_bound=np.asarray((1.0, 1.0)),
    )

    def risk_for_trajectory(trajectories, _forecasts):
        moving = bool(np.asarray(trajectories)[0, 1, 0] > 0.0)
        mass = 2.0 if moving else 3.0
        return SimpleNamespace(
            hard_violation=np.asarray((True,)),
            maximum_step_probability=np.asarray((1.0,)),
            accumulated_probability_mass=np.asarray((mass,)),
            horizon_union_bound=np.asarray((1.0,)),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", risk_for_trajectory
    )
    stop_sequence = samples[0].copy()
    stop_trajectory = controller.rollout(
        np.zeros(controller.state_spec.dimension), stop_sequence
    )[0]
    action, sequence, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            np.zeros(controller.state_spec.dimension),
            stop_sequence[0].copy(),
            stop_sequence,
            stop_trajectory,
            samples,
            np.asarray((0.0, 1.0)),
            (object(),),
            candidate_risk=candidate_risk,
        )
    )

    assert diagnostics["probabilistic_obstacle_active_fallback_index"] == 1
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "minimum_accumulated_risk_active_candidate"
    )
    assert diagnostics["probabilistic_obstacle_stop_probability_mass"] == 3.0
    assert action[0] > 0.0
    assert sequence[0, 0] > 0.0


def test_stopping_feasibility_probability_mass_order_is_opt_in():
    assert not MppiController._risk_is_strictly_better(
        1.0,
        2.0,
        1.0,
        3.0,
    )
    assert MppiController._risk_is_strictly_better(
        1.0,
        2.0,
        1.0,
        3.0,
        allow_equal_maximum_mass=True,
    )


def test_emergency_escape_candidates_replace_budgeted_samples():
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    prior = SimpleNamespace(mean=np.zeros((horizon, 2), dtype=np.float64))

    mask = controller._inject_probabilistic_emergency_candidates(
        samples, prior
    )

    assert samples.shape == (8, horizon, 2)
    assert int(np.sum(mask)) == 6
    assert not np.any(mask[:2])
    expected = np.asarray((
        (0.35, 0.0),
        (-0.35, 0.0),
        (0.35, 0.9),
        (0.35, -0.9),
        (-0.35, 0.9),
        (-0.35, -0.9),
    ))
    np.testing.assert_allclose(
        samples[2:, :3], np.repeat(expected[:, None, :], 3, axis=1)
    )
    np.testing.assert_allclose(samples[2:, 3:], 0.0)


def test_low_ttc_continuity_reserves_reverse_coverage_within_six_slots():
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    prior = SimpleNamespace(mean=np.zeros((horizon, 2), dtype=np.float64))
    forward_preferences = tuple(
        (0.35, omega) for omega in (0.2, 0.3, 0.4, 0.5, 0.6)
    )

    mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        prior,
        {
            "latched_escape_pattern": (0.35, 0.1),
            "forecast_escape_patterns": forward_preferences,
            "post_center_low_ttc_nonforward_coverage_requested": True,
        },
    )

    emergency_indices = np.flatnonzero(mask)
    assert emergency_indices.size == 6
    assert samples.shape == (8, horizon, 2)
    np.testing.assert_allclose(
        samples[emergency_indices[-3:], 0],
        np.asarray(((-0.35, 0.0), (-0.35, 0.9), (-0.35, -0.9))),
    )
    assert np.sum(samples[emergency_indices, 0, 0] < 0.0) == 3


def test_post_center_forward_commit_reserves_forward_coverage_within_six_slots():
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    reverse_preferences = tuple(
        (-0.35, omega) for omega in (-0.1, -0.2, -0.3, -0.4, -0.5)
    )

    mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {
            "latched_escape_pattern": (-0.35, 0.1),
            "forecast_escape_patterns": reverse_preferences,
            "post_center_forward_exit_commit_requested": True,
        },
    )

    emergency_indices = np.flatnonzero(mask)
    assert emergency_indices.size == 6
    np.testing.assert_allclose(
        samples[emergency_indices[-3:], 0],
        np.asarray(((0.35, 0.0), (0.35, 0.9), (0.35, -0.9))),
    )
    assert np.sum(samples[emergency_indices, 0, 0] > 0.0) == 3
    assert controller._probabilistic_emergency_forward_exit_coverage_applied


def test_low_ttc_first_step_boundary_handoff_is_emergency_only_and_opt_in():
    full_horizon = np.asarray((True, False, False, False), dtype=bool)
    first_step = np.asarray((True, True, True, False), dtype=bool)
    emergency = np.asarray((False, True, False, True), dtype=bool)

    disabled, disabled_admitted = (
        MppiController._emergency_first_step_boundary_handoff(
            full_horizon,
            first_step,
            emergency,
            enabled=False,
        )
    )
    enabled, admitted = (
        MppiController._emergency_first_step_boundary_handoff(
            full_horizon,
            first_step,
            emergency,
            enabled=True,
        )
    )

    np.testing.assert_array_equal(disabled, full_horizon)
    assert not np.any(disabled_admitted)
    np.testing.assert_array_equal(enabled, (True, True, False, False))
    np.testing.assert_array_equal(admitted, (False, True, False, False))


def test_temporal_emergency_selects_forecast_vetted_away_candidate(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_risk_weight=75.0,
            probabilistic_obstacle_hard_violation_action=(
                "active_avoidance"
            ),
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {
            "triggered": True,
            "away_heading_error_rad": -2.30,
        },
    )
    away_index = int(np.flatnonzero(emergency_mask)[0])
    assert samples[away_index, 0, 0] == pytest.approx(-0.35)
    assert samples[away_index, 0, 1] >= 0.0

    candidate_risk = SimpleNamespace(
        hard_violation=np.zeros(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.05),
        accumulated_probability_mass=np.full(8, 0.20),
        horizon_union_bound=np.full(8, 0.20),
    )
    candidate_risk.maximum_step_probability[away_index] = 0.02
    candidate_risk.accumulated_probability_mass[away_index] = 0.08

    def selected_risk(trajectories, _forecasts):
        moving_away = bool(np.asarray(trajectories)[0, 1, 0] < -0.01)
        probability = 0.02 if moving_away else 0.05
        mass = 0.08 if moving_away else 0.20
        return SimpleNamespace(
            hard_violation=np.asarray((False,)),
            maximum_step_probability=np.asarray((probability,)),
            accumulated_probability_mass=np.asarray((mass,)),
            horizon_union_bound=np.asarray((mass,)),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    stop_trajectory = controller.rollout(state, stop_sequence)[0]
    action, sequence, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            stop_trajectory,
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=True,
        )
    )

    assert action[0] == pytest.approx(-0.35)
    assert action[1] >= 0.0
    np.testing.assert_allclose(sequence[0], action)
    assert diagnostics[
        "probabilistic_obstacle_temporal_emergency_triggered"
    ]
    assert diagnostics[
        "probabilistic_obstacle_temporal_emergency_vetted"
    ]
    assert diagnostics[
        "probabilistic_obstacle_emergency_candidate_selected"
    ]
    assert diagnostics[
        "probabilistic_obstacle_active_fallback_kind"
    ] == "temporal_scan_vetted_emergency_candidate"


def test_hard_risk_retreat_yields_to_existing_emergency_lattice(monkeypatch):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_retreat_hard_risk_override_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {"triggered": True, "away_heading_error_rad": 0.0},
    )
    traversal_index = 0
    samples[traversal_index, :, 0] = -0.35
    candidate_risk = SimpleNamespace(
        hard_violation=np.ones(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.50),
        accumulated_probability_mass=np.full(8, 2.0),
        horizon_union_bound=np.ones(8),
    )
    safe_emergency_index = int(np.flatnonzero(emergency_mask)[0])
    candidate_risk.hard_violation[safe_emergency_index] = False
    candidate_risk.maximum_step_probability[safe_emergency_index] = 0.05
    candidate_risk.accumulated_probability_mass[safe_emergency_index] = 0.20

    def selected_risk(trajectories, _forecasts):
        moving_forward = bool(np.asarray(trajectories)[0, 1, 0] > 0.01)
        probability = 0.05 if moving_forward else 0.50
        mass = 0.20 if moving_forward else 2.0
        return SimpleNamespace(
            hard_violation=np.asarray((not moving_forward,)),
            maximum_step_probability=np.asarray((probability,)),
            accumulated_probability_mass=np.asarray((mass,)),
            horizon_union_bound=np.asarray((min(1.0, mass),)),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=True,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "retreat_requested": True,
                "sequence": samples[traversal_index].copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert diagnostics[
        "probabilistic_obstacle_traversal_retreat_overridden_by_hard_risk"
    ]
    assert diagnostics["probabilistic_obstacle_temporal_emergency_vetted"]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "temporal_scan_vetted_emergency_candidate"
    )
    assert action[0] > 0.0


def test_hard_risk_rearm_hold_yields_to_persistent_emergency_lattice(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_rearm_hard_risk_temporal_lattice_override_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {"triggered": True, "away_heading_error_rad": np.pi},
    )
    traversal_index = 0
    candidate_risk = SimpleNamespace(
        hard_violation=np.ones(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.50),
        accumulated_probability_mass=np.full(8, 2.0),
        horizon_union_bound=np.ones(8),
    )
    emergency_indices = np.flatnonzero(emergency_mask)
    safe_emergency_index = int(
        emergency_indices[np.argmin(samples[emergency_indices, 0, 0])]
    )
    candidate_risk.hard_violation[safe_emergency_index] = False
    candidate_risk.maximum_step_probability[safe_emergency_index] = 0.05
    candidate_risk.accumulated_probability_mass[safe_emergency_index] = 0.20

    def selected_risk(trajectories, _forecasts):
        moving_reverse = bool(np.asarray(trajectories)[0, 1, 0] < -0.01)
        probability = 0.05 if moving_reverse else 0.50
        mass = 0.20 if moving_reverse else 2.0
        return SimpleNamespace(
            hard_violation=np.asarray((not moving_reverse,)),
            maximum_step_probability=np.asarray((probability,)),
            accumulated_probability_mass=np.asarray((mass,)),
            horizon_union_bound=np.asarray((min(1.0, mass),)),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    state = np.zeros(controller.state_spec.dimension)
    hold_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            hold_sequence[0].copy(),
            hold_sequence,
            controller.rollout(state, hold_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=False,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "commit_active": False,
                "commit_started": False,
                "retreat_requested": False,
                "rearm_pending": True,
                "temporal_emergency_raw_triggered": True,
                "sequence": hold_sequence.copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert diagnostics[
        "probabilistic_obstacle_traversal_rearm_hold_overridden_by_hard_risk"
    ]
    assert diagnostics["probabilistic_obstacle_temporal_emergency_vetted"]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "rearm_hold_hard_risk_emergency_candidate"
    )
    np.testing.assert_allclose(action, samples[safe_emergency_index, 0])
    assert action[0] < 0.0


def test_hard_risk_uncommitted_staging_hold_yields_to_emergency_lattice(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_hold_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {"triggered": False, "away_heading_error_rad": np.pi},
    )
    traversal_index = 0
    emergency_index = int(np.flatnonzero(emergency_mask)[0])
    candidate_risk = SimpleNamespace(
        hard_violation=np.ones(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.50),
        accumulated_probability_mass=np.full(8, 2.0),
        horizon_union_bound=np.ones(8),
    )
    candidate_risk.hard_violation[emergency_index] = False
    candidate_risk.maximum_step_probability[emergency_index] = 0.05
    candidate_risk.accumulated_probability_mass[emergency_index] = 0.20

    def selected_risk(trajectories, _forecasts):
        count = np.asarray(trajectories).shape[0]
        return SimpleNamespace(
            hard_violation=np.zeros(count, dtype=bool),
            maximum_step_probability=np.full(count, 0.05),
            accumulated_probability_mass=np.full(count, 0.20),
            horizon_union_bound=np.full(count, 0.20),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    state = np.zeros(controller.state_spec.dimension)
    hold_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            hold_sequence[0].copy(),
            hold_sequence,
            controller.rollout(state, hold_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=False,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "commit_active": False,
                "commit_started": False,
                "retreat_requested": False,
                "rearm_pending": False,
                "uncommitted_temporal_staging_hold_requested": True,
                "sequence": hold_sequence.copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert diagnostics[
        "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_hard_risk_override"
    ]
    assert not diagnostics[
        "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_selected"
    ]
    assert diagnostics["probabilistic_obstacle_temporal_emergency_vetted"]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "uncommitted_temporal_staging_hold_hard_risk_emergency_candidate"
    )
    np.testing.assert_allclose(action, samples[emergency_index, 0])
    assert not controller._probabilistic_traversal_commit_started


def test_temporal_closing_rearm_hold_yields_before_forecast_hard_risk(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_rearm_temporal_closing_lattice_override_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {"triggered": True, "away_heading_error_rad": np.pi},
    )
    traversal_index = 0
    candidate_risk = SimpleNamespace(
        hard_violation=np.zeros(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.01),
        accumulated_probability_mass=np.full(8, 0.05),
        horizon_union_bound=np.full(8, 0.05),
    )

    def selected_risk(_trajectories, _forecasts):
        return SimpleNamespace(
            hard_violation=np.asarray((False,)),
            maximum_step_probability=np.asarray((0.01,)),
            accumulated_probability_mass=np.asarray((0.05,)),
            horizon_union_bound=np.asarray((0.05,)),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    state = np.zeros(controller.state_spec.dimension)
    hold_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            hold_sequence[0].copy(),
            hold_sequence,
            controller.rollout(state, hold_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=False,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "commit_active": False,
                "commit_started": False,
                "retreat_requested": False,
                "rearm_pending": True,
                "temporal_emergency_raw_triggered": True,
                "sequence": hold_sequence.copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert not diagnostics[
        "probabilistic_obstacle_traversal_rearm_hold_overridden_by_hard_risk"
    ]
    assert diagnostics[
        "probabilistic_obstacle_traversal_rearm_hold_overridden_by_temporal_closing"
    ]
    assert diagnostics["probabilistic_obstacle_temporal_emergency_vetted"]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "rearm_hold_temporal_closing_emergency_candidate"
    )
    assert action[0] < 0.0


def test_post_center_hard_risk_commit_yields_to_existing_emergency_lattice(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_post_center_hard_risk_override_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {"triggered": True, "away_heading_error_rad": 0.0},
    )
    traversal_index = 0
    samples[traversal_index, :, 0] = 0.35
    candidate_risk = SimpleNamespace(
        hard_violation=np.ones(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.50),
        accumulated_probability_mass=np.full(8, 2.0),
        horizon_union_bound=np.ones(8),
    )
    emergency_indices = np.flatnonzero(emergency_mask)
    safe_emergency_index = int(
        emergency_indices[np.argmin(samples[emergency_indices, 0, 0])]
    )
    candidate_risk.hard_violation[safe_emergency_index] = False
    candidate_risk.maximum_step_probability[safe_emergency_index] = 0.05
    candidate_risk.accumulated_probability_mass[safe_emergency_index] = 0.20

    def selected_risk(trajectories, _forecasts):
        moving_forward = bool(np.asarray(trajectories)[0, 1, 0] > 0.01)
        probability = 0.50 if moving_forward else 0.05
        mass = 2.0 if moving_forward else 0.20
        return SimpleNamespace(
            hard_violation=np.asarray((moving_forward,)),
            maximum_step_probability=np.asarray((probability,)),
            accumulated_probability_mass=np.asarray((mass,)),
            horizon_union_bound=np.asarray((min(1.0, mass),)),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    controller._probabilistic_traversal_commit_started = True
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=False,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "commit_started": True,
                "current_progress": 2.3,
                "crossing_progress": 2.2,
                "clear_progress": 2.9,
                "sequence": samples[traversal_index].copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert diagnostics[
        "probabilistic_obstacle_traversal_commit_overridden_by_post_center_hard_risk"
    ]
    assert diagnostics["probabilistic_obstacle_traversal_commit_started"]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "post_center_hard_risk_emergency_candidate"
    )
    assert diagnostics[
        "probabilistic_obstacle_emergency_candidate_selected"
    ]
    assert diagnostics["probabilistic_obstacle_emergency_candidate_count"] == 6
    assert not np.allclose(action, samples[traversal_index, 0])
    assert controller._probabilistic_traversal_commit_started


@pytest.mark.parametrize(
    ("continuity_guard_enabled", "expected_translation_sign"),
    ((False, 1.0), (True, -1.0)),
)
def test_post_center_low_ttc_dropout_cannot_flip_to_forward_when_enabled(
    monkeypatch,
    continuity_guard_enabled,
    expected_translation_sign,
):
    """Reproduce the seed-253 all-hard forward-selection failure."""

    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_post_center_hard_risk_override_enabled=True,
            probabilistic_obstacle_traversal_window_post_center_low_ttc_continuity_guard_enabled=(
                continuity_guard_enabled
            ),
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {"triggered": False},
    )
    traversal_index = 0
    samples[traversal_index, :, 0] = 0.35
    candidate_risk = SimpleNamespace(
        hard_violation=np.ones(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.90),
        accumulated_probability_mass=np.full(8, 3.0),
        horizon_union_bound=np.ones(8),
    )
    emergency_indices = np.flatnonzero(emergency_mask)
    for index in emergency_indices:
        if samples[index, 0, 0] > 0.0:
            candidate_risk.maximum_step_probability[index] = 0.30
            candidate_risk.accumulated_probability_mass[index] = 0.80
        else:
            candidate_risk.maximum_step_probability[index] = 0.50
            candidate_risk.accumulated_probability_mass[index] = 1.50

    def selected_risk(trajectories, _forecasts):
        first_translation = float(np.asarray(trajectories)[0, 1, 0])
        if first_translation > 0.01:
            probability, mass = 0.30, 0.80
        elif first_translation < -0.01:
            probability, mass = 0.50, 1.50
        else:
            probability, mass = 0.60, 2.00
        return SimpleNamespace(
            hard_violation=np.asarray((True,)),
            maximum_step_probability=np.asarray((probability,)),
            accumulated_probability_mass=np.asarray((mass,)),
            horizon_union_bound=np.asarray((1.0,)),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    controller._probabilistic_traversal_commit_started = True
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=False,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "commit_started": True,
                "retreat_requested": False,
                "rearm_pending": False,
                "current_progress": 2.3,
                "crossing_progress": 2.2,
                "clear_progress": 2.9,
                "temporal_emergency_scan_valid": True,
                "temporal_emergency_ttc_s": 0.30,
                "temporal_emergency_safety_hard_stop_ttc_s": 0.80,
                "temporal_emergency_closing_observed": False,
                "temporal_emergency_rearm_ready": False,
                "sequence": samples[traversal_index].copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert np.sign(action[0]) == expected_translation_sign
    assert diagnostics[
        "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_requested"
    ] is continuity_guard_enabled
    assert diagnostics[
        "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_forward_lattice_filtered"
    ] is continuity_guard_enabled
    assert diagnostics[
        "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_hard_risk_fallback"
    ] is continuity_guard_enabled
    assert diagnostics["probabilistic_obstacle_emergency_candidate_count"] == 6
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "post_center_low_ttc_continuity_emergency_candidate"
        if continuity_guard_enabled
        else "post_center_hard_risk_emergency_candidate"
    )


def test_post_center_forward_commit_rejects_lower_risk_reverse_fallback(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_post_center_hard_risk_override_enabled=True,
            probabilistic_obstacle_traversal_window_post_center_low_ttc_continuity_guard_enabled=True,
            probabilistic_obstacle_traversal_window_post_center_forward_exit_commit_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {
            "forecast_escape_patterns": tuple(
                (-0.35, omega) for omega in (0.1, 0.2, 0.3, 0.4)
            ),
            "post_center_forward_exit_commit_requested": True,
        },
    )
    traversal_index = 0
    samples[traversal_index, :, 0] = 0.35
    candidate_risk = SimpleNamespace(
        hard_violation=np.ones(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.90),
        accumulated_probability_mass=np.full(8, 3.0),
        horizon_union_bound=np.ones(8),
    )
    emergency_indices = np.flatnonzero(emergency_mask)
    for index in emergency_indices:
        if samples[index, 0, 0] < 0.0:
            candidate_risk.maximum_step_probability[index] = 0.10
            candidate_risk.accumulated_probability_mass[index] = 0.20
        else:
            candidate_risk.maximum_step_probability[index] = 0.60
            candidate_risk.accumulated_probability_mass[index] = 1.80

    def selected_risk(trajectories, _forecasts):
        first_translation = float(np.asarray(trajectories)[0, 1, 0])
        if first_translation < -0.01:
            probability, mass = 0.10, 0.20
        elif first_translation > 0.01:
            probability, mass = 0.60, 1.80
        else:
            probability, mass = 0.70, 2.00
        return SimpleNamespace(
            hard_violation=np.asarray((True,)),
            maximum_step_probability=np.asarray((probability,)),
            accumulated_probability_mass=np.asarray((mass,)),
            horizon_union_bound=np.asarray((1.0,)),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    controller._probabilistic_traversal_commit_started = True
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=False,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "commit_started": True,
                "retreat_requested": False,
                "rearm_pending": False,
                "current_progress": 2.3,
                "crossing_progress": 2.2,
                "clear_progress": 2.9,
                "post_center_forward_exit_commit_requested": True,
                "temporal_emergency_scan_valid": True,
                "temporal_emergency_ttc_s": 0.30,
                "temporal_emergency_safety_hard_stop_ttc_s": 0.80,
                "temporal_emergency_closing_observed": False,
                "temporal_emergency_rearm_ready": False,
                "sequence": samples[traversal_index].copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert action[0] > 0.0
    assert diagnostics[
        "probabilistic_obstacle_traversal_post_center_forward_exit_commit_requested"
    ]
    assert diagnostics[
        "probabilistic_obstacle_traversal_post_center_forward_exit_coverage_applied"
    ]
    assert diagnostics[
        "probabilistic_obstacle_traversal_post_center_forward_exit_candidate_count"
    ] == 3
    assert diagnostics[
        "probabilistic_obstacle_traversal_post_center_forward_exit_lattice_filtered"
    ]
    assert not diagnostics[
        "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_requested"
    ]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "post_center_forward_exit_emergency_candidate"
    )


@pytest.mark.parametrize(
    ("x_position", "expected_active"),
    ((1.0, False), (1.3, True)),
)
def test_post_center_forward_commit_has_frozen_transaction_scope(
    x_position,
    expected_active,
):
    controller = _traversal_controller(
        commit_admission_full_horizon=True,
        post_center_forward_exit_commit=True,
    )
    controller._probabilistic_traversal_crossing_progress = 2.2
    controller._probabilistic_traversal_entry_progress = 1.25
    controller._probabilistic_traversal_clear_progress = 2.9
    controller._probabilistic_traversal_commit_started = True
    state = np.zeros(controller.state_spec.dimension)
    state[controller.state_spec.index("x")] = x_position

    context = controller._probabilistic_traversal_window_context(
        state,
        _traversal_reference(),
        (_traversal_forecast(occupied=False),),
    )

    assert context["post_center_forward_exit_commit_active"] is expected_active


@pytest.mark.parametrize(
    ("retreat_requested", "rearm_pending"),
    ((True, False), (False, True)),
)
def test_post_center_low_ttc_continuity_does_not_change_other_transactions(
    monkeypatch,
    retreat_requested,
    rearm_pending,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_traversal_window_post_center_hard_risk_override_enabled=True,
            probabilistic_obstacle_traversal_window_post_center_low_ttc_continuity_guard_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    traversal_index = 0
    samples[traversal_index, :, 0] = -0.20 if retreat_requested else 0.0
    candidate_risk = SimpleNamespace(
        hard_violation=np.ones(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.50),
        accumulated_probability_mass=np.full(8, 2.0),
        horizon_union_bound=np.ones(8),
    )

    def selected_risk(trajectories, _forecasts):
        count = np.asarray(trajectories).shape[0]
        return SimpleNamespace(
            hard_violation=np.ones(count, dtype=bool),
            maximum_step_probability=np.full(count, 0.50),
            accumulated_probability_mass=np.full(count, 2.0),
            horizon_union_bound=np.ones(count),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    controller._probabilistic_traversal_commit_started = True
    state = np.zeros(controller.state_spec.dimension)
    _, _, _, diagnostics = controller._apply_probabilistic_obstacle_action_guard(
        state,
        samples[traversal_index, 0].copy(),
        samples[traversal_index].copy(),
        controller.rollout(state, samples[traversal_index])[0],
        samples,
        np.arange(8, dtype=np.float64),
        (object(),),
        candidate_risk=candidate_risk,
        emergency_candidate_mask=np.zeros(8, dtype=bool),
        temporal_emergency_triggered=False,
        traversal_context={
            "enabled": True,
            "candidate_requested": True,
            "commit_started": True,
            "retreat_requested": retreat_requested,
            "rearm_pending": rearm_pending,
            "current_progress": 2.3,
            "crossing_progress": 2.2,
            "clear_progress": 2.9,
            "temporal_emergency_scan_valid": True,
            "temporal_emergency_ttc_s": 0.30,
            "temporal_emergency_safety_hard_stop_ttc_s": 0.80,
            "temporal_emergency_closing_observed": False,
            "temporal_emergency_rearm_ready": False,
            "sequence": samples[traversal_index].copy(),
        },
        traversal_candidate_index=traversal_index,
    )

    assert not diagnostics[
        "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_requested"
    ]
    assert not diagnostics[
        "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_forward_lattice_filtered"
    ]


def test_post_center_low_ttc_continuity_leaves_safe_forward_commit_unchanged(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_traversal_window_post_center_hard_risk_override_enabled=True,
            probabilistic_obstacle_traversal_window_post_center_low_ttc_continuity_guard_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples, np.zeros((horizon, 2), dtype=np.float64)
    )
    traversal_index = 0
    samples[traversal_index, :, 0] = 0.35
    candidate_risk = SimpleNamespace(
        hard_violation=np.ones(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.50),
        accumulated_probability_mass=np.full(8, 2.0),
        horizon_union_bound=np.ones(8),
    )
    candidate_risk.hard_violation[traversal_index] = False
    candidate_risk.maximum_step_probability[traversal_index] = 0.02
    candidate_risk.accumulated_probability_mass[traversal_index] = 0.10

    def selected_risk(trajectories, _forecasts):
        count = np.asarray(trajectories).shape[0]
        return SimpleNamespace(
            hard_violation=np.zeros(count, dtype=bool),
            maximum_step_probability=np.full(count, 0.02),
            accumulated_probability_mass=np.full(count, 0.10),
            horizon_union_bound=np.full(count, 0.10),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    controller._probabilistic_traversal_commit_started = True
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=False,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "commit_started": True,
                "retreat_requested": False,
                "rearm_pending": False,
                "current_progress": 2.3,
                "crossing_progress": 2.2,
                "clear_progress": 2.9,
                "temporal_emergency_scan_valid": True,
                "temporal_emergency_ttc_s": 0.30,
                "temporal_emergency_safety_hard_stop_ttc_s": 0.80,
                "temporal_emergency_closing_observed": False,
                "temporal_emergency_rearm_ready": False,
                "sequence": samples[traversal_index].copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert action[0] > 0.0
    assert diagnostics[
        "probabilistic_obstacle_active_fallback_kind"
    ] == "certified_traversal_window_candidate"
    assert not diagnostics[
        "probabilistic_obstacle_traversal_post_center_low_ttc_continuity_requested"
    ]
    assert diagnostics["probabilistic_obstacle_emergency_candidate_count"] == 6


def test_post_center_corroborated_temporal_warning_yields_before_hard_risk(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.60), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_emergency_candidate_pareto_forward_commit_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_abort_mass_floor=0.05,
            probabilistic_obstacle_traversal_window_post_center_temporal_override_enabled=True,
            probabilistic_obstacle_traversal_window_post_center_temporal_escape_latch_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {"triggered": True, "latched_escape_pattern": (-0.35, 0.0)},
    )
    traversal_index = 0
    samples[traversal_index, :, 0] = 0.60
    candidate_risk = SimpleNamespace(
        hard_violation=np.zeros(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.05),
        accumulated_probability_mass=np.full(8, 0.20),
        horizon_union_bound=np.full(8, 0.20),
    )
    emergency_indices = np.flatnonzero(emergency_mask)
    preferred_index = int(emergency_indices[0])
    forward_index = int(next(
        index for index in emergency_indices
        if samples[index, 0, 0] > 0.0
        and np.isclose(samples[index, 0, 1], 0.0)
    ))
    candidate_risk.hard_violation[preferred_index] = True
    candidate_risk.maximum_step_probability[preferred_index] = 0.30
    candidate_risk.accumulated_probability_mass[preferred_index] = 2.30
    candidate_risk.maximum_step_probability[forward_index] = 0.16
    candidate_risk.accumulated_probability_mass[forward_index] = 1.20
    candidate_costs = np.arange(8, dtype=np.float64)
    candidate_costs[forward_index] = -1.0

    def selected_risk(trajectories, _forecasts):
        positions = np.asarray(trajectories)[0, :, :2]
        displacement = float(positions[-1, 0] - positions[0, 0])
        if abs(displacement) <= 1.0e-9:
            probability, mass = 1.0, 5.0
        elif displacement < 0.0:
            probability, mass = 0.30, 2.30
        else:
            probability, mass = 0.16, 1.20
        return SimpleNamespace(
            hard_violation=np.asarray((probability >= 0.20,)),
            maximum_step_probability=np.asarray((probability,)),
            accumulated_probability_mass=np.asarray((mass,)),
            horizon_union_bound=np.asarray((mass,)),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    controller._probabilistic_traversal_commit_started = True
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            candidate_costs,
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=False,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "commit_started": True,
                "current_progress": 2.30,
                "crossing_progress": 2.20,
                "clear_progress": 2.90,
                "temporal_corroboration_probability_mass": 0.10,
                "temporal_emergency_raw_triggered": True,
                "sequence": samples[traversal_index].copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert diagnostics[
        "probabilistic_obstacle_traversal_commit_overridden_by_post_center_temporal_risk"
    ]
    assert diagnostics[
        "probabilistic_obstacle_traversal_post_center_temporal_raw_triggered"
    ]
    assert not diagnostics[
        "probabilistic_obstacle_traversal_commit_overridden_by_post_center_hard_risk"
    ]
    assert diagnostics["probabilistic_obstacle_traversal_commit_started"]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "post_center_temporal_corroborated_emergency_candidate"
    )
    assert diagnostics["probabilistic_obstacle_emergency_candidate_count"] == 6
    assert not diagnostics["probabilistic_obstacle_pareto_forward_commit_applied"]
    assert action[0] < 0.0
    assert not np.allclose(action, samples[traversal_index, 0])
    assert controller._probabilistic_traversal_commit_started
    assert controller._probabilistic_traversal_post_center_temporal_pattern == (
        -0.35,
        0.0,
    )
    assert diagnostics[
        "probabilistic_obstacle_traversal_post_center_temporal_escape_latched"
    ]
    assert not diagnostics[
        "probabilistic_obstacle_traversal_post_center_temporal_escape_reused"
    ]


def test_post_center_temporal_escape_latch_outlives_global_intent_and_clears():
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.60), 0.9),
        MppiConfig(
            horizon=5,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_temporal_abort_mass_floor=0.05,
            probabilistic_obstacle_traversal_window_post_center_temporal_override_enabled=True,
            probabilistic_obstacle_traversal_window_post_center_temporal_escape_latch_enabled=True,
        ),
    )
    state = np.zeros(controller.state_spec.dimension, dtype=np.float64)
    state[controller.state_spec.index("theta")] = 0.20
    controller._probabilistic_traversal_commit_started = True

    assert controller._latch_probabilistic_traversal_post_center_temporal_escape(
        (-0.35, 0.90), state
    )
    assert not controller._latch_probabilistic_traversal_post_center_temporal_escape(
        (0.60, 0.0), state
    )
    controller._probabilistic_emergency_intent_remaining = 0
    controller._probabilistic_emergency_latched_pattern = None
    controller._probabilistic_emergency_latched_heading = None
    next_state = state.copy()
    next_state[controller.state_spec.index("theta")] = 0.29
    emergency_context, traversal_context = (
        controller._bind_probabilistic_traversal_post_center_temporal_escape(
            {"triggered": False, "raw_triggered": True},
            {
                "candidate_requested": True,
                "commit_started": True,
                "retreat_requested": False,
                "rearm_pending": False,
                "temporal_emergency_raw_triggered": True,
                "temporal_corroboration_probability_mass": 0.10,
                "current_progress": 2.30,
                "crossing_progress": 2.20,
                "clear_progress": 2.90,
            },
            next_state,
        )
    )

    assert traversal_context["post_center_temporal_escape_reused"]
    assert emergency_context["latched_escape_pattern"][0] == -0.35
    assert 0.0 < emergency_context["latched_escape_pattern"][1] < 0.90
    assert controller._probabilistic_emergency_latched_pattern is None
    assert controller._probabilistic_traversal_commit_started

    _, cleared_context = (
        controller._bind_probabilistic_traversal_post_center_temporal_escape(
            {"triggered": False, "raw_triggered": False},
            {
                "candidate_requested": True,
                "commit_started": True,
                "temporal_emergency_raw_triggered": False,
                "temporal_corroboration_probability_mass": 0.10,
                "current_progress": 2.31,
                "crossing_progress": 2.20,
                "clear_progress": 2.90,
            },
            next_state,
        )
    )
    assert not cleared_context["post_center_temporal_escape_reused"]
    assert controller._probabilistic_traversal_post_center_temporal_pattern is None
    assert controller._probabilistic_traversal_commit_started

    assert controller._latch_probabilistic_traversal_post_center_temporal_escape(
        (-0.35, 0.90), state
    )
    controller.reset(seed=7)
    assert controller._probabilistic_traversal_post_center_temporal_pattern is None
    assert controller._probabilistic_traversal_post_center_temporal_heading is None


def test_midpoint_retreat_uses_existing_emergency_lattice_immediately(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_temporal_midpoint_lattice_override_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {"triggered": True, "away_heading_error_rad": 0.0},
    )
    traversal_index = 0
    samples[traversal_index, :, 0] = -0.35
    candidate_risk = SimpleNamespace(
        hard_violation=np.zeros(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.05),
        accumulated_probability_mass=np.full(8, 0.20),
        horizon_union_bound=np.full(8, 0.20),
    )

    def selected_risk(trajectories, _forecasts):
        moving_forward = bool(np.asarray(trajectories)[0, 1, 0] > 0.01)
        probability = 0.05 if moving_forward else 0.50
        mass = 0.20 if moving_forward else 2.0
        return SimpleNamespace(
            hard_violation=np.asarray((not moving_forward,)),
            maximum_step_probability=np.asarray((probability,)),
            accumulated_probability_mass=np.asarray((mass,)),
            horizon_union_bound=np.asarray((min(1.0, mass),)),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=False,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "retreat_requested": True,
                "retreat_temporal_lattice_requested": True,
                "sequence": samples[traversal_index].copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert diagnostics[
        "probabilistic_obstacle_traversal_retreat_overridden_by_temporal_midpoint_guard"
    ]
    assert diagnostics["probabilistic_obstacle_temporal_emergency_vetted"]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "temporal_scan_vetted_emergency_candidate"
    )
    assert action[0] > 0.0


def test_raw_bound_midpoint_retreat_restores_fixed_candidate_after_clear(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_midpoint_lattice_override_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_retreat_raw_lattice_binding_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    traversal_index = 0
    samples[traversal_index, :, 0] = -0.35
    candidate_risk = SimpleNamespace(
        hard_violation=np.zeros(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.01),
        accumulated_probability_mass=np.full(8, 0.02),
        horizon_union_bound=np.full(8, 0.02),
    )

    def selected_risk(trajectories, _forecasts):
        count = np.asarray(trajectories).shape[0]
        return SimpleNamespace(
            hard_violation=np.zeros(count, dtype=bool),
            maximum_step_probability=np.full(count, 0.01),
            accumulated_probability_mass=np.full(count, 0.02),
            horizon_union_bound=np.full(count, 0.02),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=np.zeros(8, dtype=bool),
            temporal_emergency_triggered=False,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "retreat_requested": True,
                "retreat_temporal_lattice_requested": True,
                "temporal_emergency_raw_triggered": False,
                "exit_deadline_retreat_escape_transaction_active": False,
                "sequence": samples[traversal_index].copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert not diagnostics[
        "probabilistic_obstacle_traversal_retreat_overridden_by_temporal_midpoint_guard"
    ]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "certified_traversal_window_candidate"
    )
    assert action[0] == -0.35


def test_post_intent_all_hard_temporal_retreat_filters_forward_lattice(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_temporal_midpoint_lattice_override_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_retreat_raw_lattice_binding_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_retreat_post_intent_all_hard_forward_filter_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {"triggered": False, "away_heading_error_rad": 0.0},
    )
    traversal_index = 0
    samples[traversal_index, :, 0] = -0.35
    emergency_indices = np.flatnonzero(emergency_mask)
    forward_indices = emergency_indices[samples[emergency_indices, 0, 0] > 0.0]
    reverse_indices = emergency_indices[samples[emergency_indices, 0, 0] < 0.0]
    best_forward = int(forward_indices[0])
    best_reverse = int(reverse_indices[0])
    candidate_risk = SimpleNamespace(
        hard_violation=np.ones(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.50),
        accumulated_probability_mass=np.full(8, 2.0),
        horizon_union_bound=np.ones(8),
    )
    candidate_risk.maximum_step_probability[best_forward] = 0.05
    candidate_risk.accumulated_probability_mass[best_forward] = 0.10
    candidate_risk.maximum_step_probability[best_reverse] = 0.10
    candidate_risk.accumulated_probability_mass[best_reverse] = 0.20

    def selected_risk(trajectories, _forecasts):
        count = np.asarray(trajectories).shape[0]
        return SimpleNamespace(
            hard_violation=np.ones(count, dtype=bool),
            maximum_step_probability=np.full(count, 0.50),
            accumulated_probability_mass=np.full(count, 2.0),
            horizon_union_bound=np.ones(count),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    common = dict(
        candidate_risk=candidate_risk,
        emergency_candidate_mask=emergency_mask,
        temporal_emergency_triggered=False,
        traversal_context={
            "enabled": True,
            "candidate_requested": True,
            "retreat_requested": True,
            "retreat_temporal_lattice_requested": True,
            "temporal_emergency_raw_triggered": True,
            "temporal_retreat_raw_lattice_requested": True,
            "sequence": samples[traversal_index].copy(),
        },
        traversal_candidate_index=traversal_index,
    )
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            **common,
        )
    )

    assert diagnostics[
        "probabilistic_obstacle_traversal_temporal_retreat_post_intent_forward_lattice_filtered"
    ]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "temporal_retreat_post_intent_emergency_candidate"
    )
    np.testing.assert_allclose(action, samples[best_reverse, 0])
    assert action[0] < 0.0
    assert not diagnostics[
        "probabilistic_obstacle_pareto_forward_commit_applied"
    ]

    candidate_risk.hard_violation[best_forward] = False
    safe_action, _, _, safe_diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            **common,
        )
    )
    assert not safe_diagnostics[
        "probabilistic_obstacle_traversal_temporal_retreat_post_intent_forward_lattice_filtered"
    ]
    assert safe_action[0] > 0.0


def test_midpoint_retreat_filters_forward_during_live_global_intent(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.60), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_traversal_window_temporal_midpoint_lattice_override_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_retreat_raw_lattice_binding_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_midpoint_retreat_reverse_filter_enabled=True,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {"triggered": True, "away_heading_error_rad": 0.0},
    )
    traversal_index = 0
    samples[traversal_index, :, 0] = -0.35
    emergency_indices = np.flatnonzero(emergency_mask)
    forward_indices = emergency_indices[
        samples[emergency_indices, 0, 0] > 0.0
    ]
    reverse_indices = emergency_indices[
        samples[emergency_indices, 0, 0] < 0.0
    ]
    best_forward = int(forward_indices[0])
    best_reverse = int(reverse_indices[-1])
    candidate_risk = SimpleNamespace(
        hard_violation=np.zeros(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.15),
        accumulated_probability_mass=np.full(8, 0.50),
        horizon_union_bound=np.ones(8),
    )
    candidate_risk.maximum_step_probability[best_forward] = 0.001
    candidate_risk.accumulated_probability_mass[best_forward] = 0.002
    candidate_risk.maximum_step_probability[best_reverse] = 0.01
    candidate_risk.accumulated_probability_mass[best_reverse] = 0.02

    def selected_risk(trajectories, _forecasts):
        count = np.asarray(trajectories).shape[0]
        return SimpleNamespace(
            hard_violation=np.zeros(count, dtype=bool),
            maximum_step_probability=np.full(count, 0.01),
            accumulated_probability_mass=np.full(count, 0.02),
            horizon_union_bound=np.ones(count),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=True,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "retreat_requested": True,
                "retreat_temporal_lattice_requested": True,
                "temporal_emergency_raw_triggered": True,
                "temporal_retreat_raw_lattice_requested": True,
                "exit_deadline_retreat_escape_transaction_active": False,
                "sequence": samples[traversal_index].copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert diagnostics[
        "probabilistic_obstacle_traversal_temporal_midpoint_retreat_forward_lattice_filtered"
    ]
    assert not diagnostics[
        "probabilistic_obstacle_traversal_temporal_retreat_post_intent_forward_lattice_filtered"
    ]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "temporal_midpoint_retreat_emergency_candidate"
    )
    np.testing.assert_allclose(action, samples[best_reverse, 0])
    assert action[0] < 0.0
    assert not diagnostics[
        "probabilistic_obstacle_pareto_forward_commit_applied"
    ]


def test_exit_deadline_retreat_latches_first_escape_and_blocks_forward_promotion(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.60), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_emergency_candidate_pareto_forward_commit_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_midpoint_lattice_override_enabled=True,
            probabilistic_obstacle_traversal_window_temporal_exit_deadline_escape_latch_enabled=True,
        ),
    )
    controller._start_probabilistic_traversal_exit_deadline_retreat_escape()
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {"triggered": True, "latched_escape_pattern": (-0.35, 0.90)},
    )
    traversal_index = 0
    samples[traversal_index, :, 0] = -0.35
    candidate_risk = SimpleNamespace(
        hard_violation=np.zeros(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.15),
        accumulated_probability_mass=np.full(8, 0.60),
        horizon_union_bound=np.full(8, 0.60),
    )
    emergency_indices = np.flatnonzero(emergency_mask)
    preferred_index = int(emergency_indices[0])
    forward_index = int(next(
        index for index in emergency_indices
        if samples[index, 0, 0] > 0.0
        and np.isclose(samples[index, 0, 1], 0.0)
    ))
    candidate_risk.maximum_step_probability[preferred_index] = 0.10
    candidate_risk.accumulated_probability_mass[preferred_index] = 0.40
    candidate_risk.maximum_step_probability[forward_index] = 0.05
    candidate_risk.accumulated_probability_mass[forward_index] = 0.20
    candidate_costs = np.arange(8, dtype=np.float64)
    candidate_costs[forward_index] = -1.0

    monkeypatch.setattr(
        controller,
        "_probabilistic_collision_risk",
        lambda trajectories, forecasts: SimpleNamespace(
            hard_violation=np.asarray((False,)),
            maximum_step_probability=np.asarray((0.10,)),
            accumulated_probability_mass=np.asarray((0.40,)),
            horizon_union_bound=np.asarray((0.40,)),
        ),
    )
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            controller.rollout(state, stop_sequence)[0],
            samples,
            candidate_costs,
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=False,
            traversal_context={
                "enabled": True,
                "candidate_requested": True,
                "retreat_requested": True,
                "retreat_temporal_lattice_requested": True,
                "exit_deadline_retreat_escape_transaction_active": True,
                "sequence": samples[traversal_index].copy(),
            },
            traversal_candidate_index=traversal_index,
        )
    )

    assert action[0] < 0.0
    assert np.isclose(action[1], 0.90)
    assert not diagnostics["probabilistic_obstacle_pareto_forward_commit_applied"]
    assert diagnostics["probabilistic_obstacle_active_fallback_kind"] == (
        "exit_deadline_retreat_emergency_candidate"
    )
    assert diagnostics[
        "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_latched"
    ]
    assert controller._probabilistic_traversal_exit_deadline_retreat_pattern == (
        -0.35,
        0.90,
    )


def test_temporal_emergency_requires_sustained_ttc_clearance_to_rearm():
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=5,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_trigger_ttc_s=1.5,
            probabilistic_obstacle_emergency_candidate_intent_hold_steps=2,
            probabilistic_obstacle_emergency_candidate_rearm_ttc_s=3.0,
            probabilistic_obstacle_emergency_candidate_rearm_clear_steps=3,
        ),
    )

    def observation(ttc_s, *, valid=True, matched=True):
        return SimpleNamespace(auxiliary={
            "dynamic_obstacle_escape_context": {
                "temporal_scan_valid": valid,
                "temporal_scan_ttc_s": ttc_s,
                "dynamic_obstacle_scan_flow_match": matched,
                "dynamic_obstacle_away_heading_error_rad": 0.0,
            }
        })

    closing_beyond_trigger = controller._probabilistic_emergency_context(
        observation(4.0)
    )
    assert closing_beyond_trigger["closing_observed"]
    assert not closing_beyond_trigger["raw_triggered"]
    assert not closing_beyond_trigger["triggered"]

    assert controller._probabilistic_emergency_context(
        observation(1.0)
    )["triggered"]
    assert controller._probabilistic_emergency_context(
        observation(1.0)
    )["intent_held"]
    assert controller._probabilistic_emergency_context(
        observation(1.0)
    )["intent_held"]

    # A one-frame association dropout must not rearm while TTC evidence still
    # indicates an unresolved encounter.
    unresolved = controller._probabilistic_emergency_context(
        observation(1.0, matched=False)
    )
    assert not unresolved["triggered"]
    assert not unresolved["rearm_ready"]
    assert unresolved["rearm_clear_count"] == 0

    for expected in (1, 2, 0):
        cleared = controller._probabilistic_emergency_context(
            observation(float("inf"), valid=False, matched=False)
        )
        assert cleared["rearm_clear_count"] == expected
    assert cleared["rearm_ready"]
    assert controller._probabilistic_emergency_context(
        observation(1.0)
    )["triggered"]


def test_temporal_emergency_uses_nearest_multi_track_forecast():
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=5,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_trigger_ttc_s=1.5,
        ),
    )
    state = np.zeros(controller.state_spec.dimension)
    state[controller.state_spec.index("x")] = 1.0

    def forecast(velocity):
        means = np.zeros((5, 1, 2), dtype=np.float64)
        for step in range(5):
            means[step, 0] = step * np.asarray(velocity)
        return SimpleNamespace(
            component_means=means,
            component_weights=np.ones((5, 1), dtype=np.float64),
        )

    observation = SimpleNamespace(auxiliary={
        "dynamic_obstacle_escape_context": {
            "temporal_scan_valid": True,
            "temporal_scan_ttc_s": 1.0,
            "dynamic_obstacle_scan_flow_match": True,
            "dynamic_obstacle_away_heading_error_rad": 0.0,
            "dynamic_obstacle_forecast_index": 1,
        }
    })
    context = controller._probabilistic_emergency_context(
        observation,
        state,
        (forecast((1.0, 0.0)), forecast((0.0, 1.0))),
    )

    assert context["escape_forecast_index"] == 1
    assert context["forecast_escape_patterns"][0] == pytest.approx(
        (0.35, 0.0)
    )


def test_hard_temporal_emergency_uses_lowest_risk_lattice_member(
    monkeypatch,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {"triggered": True, "latched_escape_pattern": (0.35, 0.9)},
    )
    indices = np.flatnonzero(emergency_mask)
    preferred_index = int(indices[0])
    safer_index = int(indices[1])
    candidate_risk = SimpleNamespace(
        hard_violation=np.ones(8, dtype=bool),
        maximum_step_probability=np.ones(8, dtype=np.float64),
        accumulated_probability_mass=np.full(8, 0.45),
        horizon_union_bound=np.ones(8, dtype=np.float64),
    )
    candidate_risk.accumulated_probability_mass[preferred_index] = 0.40
    candidate_risk.accumulated_probability_mass[safer_index] = 0.10

    def single_risk(_trajectories, _forecasts):
        return SimpleNamespace(
            hard_violation=np.asarray((True,)),
            maximum_step_probability=np.asarray((1.0,)),
            accumulated_probability_mass=np.asarray((0.50,)),
            horizon_union_bound=np.asarray((1.0,)),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", single_risk
    )
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    stop_trajectory = controller.rollout(state, stop_sequence)[0]
    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            stop_trajectory,
            samples,
            np.arange(8, dtype=np.float64),
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=True,
        )
    )

    np.testing.assert_allclose(action, samples[safer_index, 0])
    assert diagnostics[
        "probabilistic_obstacle_minimum_risk_emergency_index"
    ] == safer_index


@pytest.mark.parametrize(
    (
        "reverse_risk",
        "forward_risk",
        "forward_ceilings",
        "expect_low_risk_commit",
    ),
    (
        ((0.05, 0.20), (0.01, 0.05), (0.0, 0.0), False),
        ((0.01, 0.05), (0.012, 0.06), (0.02, 0.10), True),
    ),
)
def test_pareto_forward_escape_replaces_and_reheads_reverse_intent(
    monkeypatch,
    reverse_risk,
    forward_risk,
    forward_ceilings,
    expect_low_risk_commit,
):
    horizon = 5
    controller = MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((-0.35, 0.35), 0.9),
        MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_risk_weight=75.0,
            probabilistic_obstacle_hard_violation_action=(
                "active_avoidance"
            ),
            probabilistic_obstacle_emergency_candidates_enabled=True,
            probabilistic_obstacle_emergency_candidate_prefix_steps=3,
            probabilistic_obstacle_emergency_candidate_trigger_ttc_s=1.5,
            probabilistic_obstacle_emergency_candidate_intent_hold_steps=2,
            probabilistic_obstacle_emergency_candidate_pareto_forward_commit_enabled=True,
            probabilistic_obstacle_emergency_candidate_forward_risk_ceiling=(
                forward_ceilings[0]
            ),
            probabilistic_obstacle_emergency_candidate_forward_mass_ceiling=(
                forward_ceilings[1]
            ),
        ),
    )
    samples = np.zeros((8, horizon, 2), dtype=np.float64)
    emergency_mask = controller._inject_probabilistic_emergency_candidates(
        samples,
        np.zeros((horizon, 2), dtype=np.float64),
        {
            "triggered": True,
            "latched_escape_pattern": (-0.35, -0.9),
        },
    )
    emergency_indices = np.flatnonzero(emergency_mask)
    reverse_index = int(emergency_indices[0])
    forward_index = int(np.flatnonzero(
        emergency_mask
        & np.isclose(samples[:, 0, 0], 0.35)
        & np.isclose(samples[:, 0, 1], 0.9)
    )[0])
    candidate_risk = SimpleNamespace(
        hard_violation=np.zeros(8, dtype=bool),
        maximum_step_probability=np.full(8, 0.10),
        accumulated_probability_mass=np.full(8, 0.50),
        horizon_union_bound=np.full(8, 0.50),
    )
    candidate_risk.maximum_step_probability[reverse_index] = reverse_risk[0]
    candidate_risk.accumulated_probability_mass[reverse_index] = reverse_risk[1]
    candidate_risk.maximum_step_probability[forward_index] = forward_risk[0]
    candidate_risk.accumulated_probability_mass[forward_index] = forward_risk[1]
    costs = np.full(8, 20.0)
    costs[reverse_index] = 10.0
    costs[forward_index] = 5.0

    def selected_risk(_trajectories, _forecasts):
        return SimpleNamespace(
            hard_violation=np.asarray((False,)),
            maximum_step_probability=np.asarray((0.01,)),
            accumulated_probability_mass=np.asarray((0.05,)),
            horizon_union_bound=np.asarray((0.05,)),
        )

    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", selected_risk
    )
    state = np.zeros(controller.state_spec.dimension)
    stop_sequence = np.zeros((horizon, 2), dtype=np.float64)
    stop_trajectory = controller.rollout(state, stop_sequence)[0]

    action, _, _, diagnostics = (
        controller._apply_probabilistic_obstacle_action_guard(
            state,
            stop_sequence[0].copy(),
            stop_sequence,
            stop_trajectory,
            samples,
            costs,
            (object(),),
            candidate_risk=candidate_risk,
            emergency_candidate_mask=emergency_mask,
            temporal_emergency_triggered=True,
        )
    )

    np.testing.assert_allclose(action, (0.35, 0.9))
    assert diagnostics[
        "probabilistic_obstacle_pareto_forward_commit_applied"
    ]
    assert diagnostics[
        "probabilistic_obstacle_low_risk_forward_commit_applied"
    ] is expect_low_risk_commit
    assert controller._probabilistic_emergency_latched_heading == (
        pytest.approx(0.27)
    )

    controller._probabilistic_emergency_intent_remaining = 1
    controller._probabilistic_emergency_rearm_ready = False
    aligned_state = state.copy()
    aligned_state[controller.state_spec.index("theta")] = 0.27
    held = controller._probabilistic_emergency_context(
        SimpleNamespace(auxiliary={
            "dynamic_obstacle_escape_context": {
                "temporal_scan_valid": False,
                "temporal_scan_ttc_s": float("inf"),
                "dynamic_obstacle_scan_flow_match": False,
            }
        }),
        aligned_state,
    )
    assert held["intent_held"]
    assert held["latched_escape_pattern"][0] == pytest.approx(0.35)
    assert held["latched_escape_pattern"][1] == pytest.approx(0.0)
