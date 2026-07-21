import numpy as np
import pytest

from mobile_robot_mppi.core.references import PolylineReference
from mobile_robot_mppi.core.spaces import (
    ActionSpec,
    body_velocity_action,
    dynamic_unicycle_state,
)
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController
from mobile_robot_mppi.policies.priors import PriorOutput


def test_corridor_width_preview_interpolates_profile():
    reference = PolylineReference(
        [(0.0, 0.0), (10.0, 0.0)],
        corridor_half_width=0.8,
        footprint_radius=0.2,
        corridor_half_width_profile=((0.0, 0.8), (0.5, 0.7), (1.0, 0.8)),
    )

    widths = reference.preview_corridor_half_widths(np.asarray([0.0, 5.0, 10.0]))

    np.testing.assert_allclose(widths, [0.8, 0.7, 0.8])


def test_mppi_boundary_cost_rejects_footprint_outside_corridor():
    config = MppiConfig(
        horizon=2,
        num_samples=2,
        dt=0.1,
        noise_sigma=(0.1, 0.1),
        goal_running_weight=0.0,
        goal_terminal_weight=0.0,
        path_preview_enabled=True,
        path_boundary_enabled=True,
        path_boundary_buffer=0.05,
        path_boundary_weight=100.0,
        path_boundary_violation_penalty=10000.0,
        control_weight=0.0,
        control_rate_weight=0.0,
    )
    controller = MppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.65), 1.25),
        config,
    )
    reference = PolylineReference(
        [(0.0, 0.0), (10.0, 0.0)],
        corridor_half_width=0.8,
        footprint_radius=0.2,
    )
    target = reference.target_at(0.0, np.zeros(5))
    trajectories = np.zeros((2, 3, 5), dtype=np.float64)
    trajectories[1, :, 1] = 0.61
    controls = np.zeros((2, 2, 2), dtype=np.float64)

    costs = controller._cost(
        trajectories, controls, target, obstacles=(), reference=reference
    )

    assert costs[0] == pytest.approx(0.0)
    assert costs[1] >= config.path_boundary_violation_penalty


def test_boundary_margin_is_spatial_not_time_indexed_around_bend():
    config = MppiConfig(
        horizon=4,
        num_samples=2,
        dt=0.1,
        noise_sigma=(0.1, 0.1),
        path_preview_enabled=True,
        path_boundary_enabled=True,
        path_boundary_violation_penalty=10000.0,
    )
    controller = MppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.65), 1.25),
        config,
    )
    reference = PolylineReference(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)],
        corridor_half_width=0.8,
        footprint_radius=0.2,
    )
    reference.progress = 0.8
    trajectories = np.zeros((2, 5, 5), dtype=np.float64)
    trajectories[:, :, 0] = 0.8
    trajectories[1, :, 1] = -0.7

    margins = controller._path_boundary_margins(trajectories, reference)

    # The stopped candidate remains 0.6 m inside the spatial corridor even
    # though a time-indexed preview would already have advanced around the bend.
    np.testing.assert_allclose(margins[0], 0.6)
    assert np.all(margins[1] < 0.0)


def test_boundary_enforcement_is_opt_in_for_baseline_compatibility():
    values = MppiConfig.from_mapping({}, action_dim=2)

    assert not values.path_boundary_enabled


def test_candidate_filter_requires_boundary_enforcement():
    config = MppiConfig(
        path_boundary_candidate_filter_enabled=True,
    )

    with pytest.raises(ValueError, match="requires path boundary enforcement"):
        config.validate(action_dim=2)


def test_candidate_filter_removes_infeasible_candidate_from_mppi_update(monkeypatch):
    config = MppiConfig(
        horizon=2,
        num_samples=3,
        dt=0.1,
        temperature=1.0,
        noise_sigma=(0.1, 0.1),
        goal_running_weight=0.0,
        goal_terminal_weight=0.0,
        path_preview_enabled=True,
        path_boundary_enabled=True,
        path_boundary_buffer=0.0,
        path_boundary_weight=0.0,
        path_boundary_violation_penalty=1.0,
        path_boundary_candidate_filter_enabled=True,
        control_weight=0.0,
        control_rate_weight=0.0,
        previous_sequence_blend=0.0,
    )
    action_spec = ActionSpec(
        ("v_cmd", "omega_cmd"),
        lower=np.asarray((-1.0, -1.0)),
        upper=np.asarray((1.0, 1.0)),
    )
    controller = MppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        action_spec,
        config,
    )
    reference = PolylineReference(
        [(0.0, 0.0), (10.0, 0.0)],
        corridor_half_width=0.3,
        footprint_radius=0.2,
    )
    state = np.zeros(5, dtype=np.float64)
    target = reference.target_at(0.0, state)
    samples = np.zeros((3, 2, 2), dtype=np.float64)
    samples[1, :, 0] = 0.1  # Marked unsafe by the synthetic rollout.
    samples[2, :, 0] = 0.2

    def fake_rollout(_state, controls):
        controls = np.asarray(controls, dtype=np.float64)
        if controls.ndim == 2:
            controls = controls[None, ...]
        paths = np.zeros((controls.shape[0], 3, 5), dtype=np.float64)
        unsafe = np.isclose(controls[:, 0, 0], 0.1)
        paths[unsafe, :, 1] = 0.2
        return paths

    monkeypatch.setattr(controller, "_sample", lambda prior, rng: samples.copy())
    monkeypatch.setattr(controller, "rollout", fake_rollout)
    monkeypatch.setattr(
        controller,
        "_cost",
        lambda *args, **kwargs: np.asarray([5.0, 0.0, 10.0]),
    )
    monkeypatch.setattr(
        controller,
        "_importance_sampling_cost",
        lambda *args, **kwargs: np.zeros(3, dtype=np.float64),
    )

    _, sequence, trajectory, diagnostics = controller._solve_plan(
        state,
        PriorOutput(np.zeros((2, 2)), None, {}),
        target,
        (),
        np.random.RandomState(0),
        reference=reference,
    )

    assert diagnostics["path_boundary_candidate_feasible_count"] == 2
    assert diagnostics["path_boundary_candidate_feasible_fraction"] == pytest.approx(2 / 3)
    assert diagnostics["path_boundary_weighted_update_feasible"]
    assert np.max(np.abs(trajectory[:, 1])) == pytest.approx(0.0)
    assert not np.isclose(sequence[0, 0], 0.1)
