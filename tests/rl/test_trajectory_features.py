import numpy as np
import pytest

from mobile_robot_mppi.core.types import ControlCommand, PlanResult
from mobile_robot_mppi.rl.trajectory_features import (
    TRAJECTORY_METRIC_NAMES,
    paired_trajectory_feature_vector,
    plan_trajectory_metrics,
)


def _plan(offset=0.0):
    trajectory = np.asarray(
        ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.4 + offset, 0.0, 0.0)),
        dtype=np.float64,
    )
    controls = np.asarray(((0.2, 0.1), (0.3, -0.2)), dtype=np.float64)
    diagnostics = {
        "target_x": 1.0,
        "target_y": 0.0,
        "cost_min": 1.0 - offset,
        "cost_q10": 1.2 - offset,
        "cost_q50": 2.0 - offset,
        "cost_q90": 3.0 - offset,
        "cost_std": 0.5,
        "effective_sample_fraction": 0.25,
        "sample_saturation_fraction": 0.1,
    }
    return PlanResult(
        proposed_control=ControlCommand(controls[0]),
        control_sequence=controls,
        predicted_trajectory=trajectory,
        diagnostics=diagnostics,
    )


def test_plan_trajectory_metrics_are_finite_and_in_physical_units():
    values = plan_trajectory_metrics(
        _plan(),
        final_target_xy=(2.0, 0.0),
        obstacles=((0.5, 0.0, 0.05),),
        position_indices=(0, 1),
        action_names=("v_cmd", "omega_cmd"),
        previous_action=(0.0, 0.0),
        robot_radius=0.1,
        obstacle_influence=0.5,
    )
    assert values.shape == (len(TRAJECTORY_METRIC_NAMES),)
    assert np.isfinite(values).all()
    lookup = dict(zip(TRAJECTORY_METRIC_NAMES, values))
    assert lookup["predicted_local_progress_m"] == pytest.approx(0.4)
    assert lookup["predicted_final_progress_m"] == pytest.approx(0.4)
    assert lookup["trajectory_length_m"] == pytest.approx(0.4)
    assert lookup["minimum_clearance_m"] == pytest.approx(-0.05)
    assert lookup["collision_fraction"] > 0.0


def test_paired_trajectory_features_preserve_state_prefix_and_delta():
    state = np.asarray((1.0, 2.0, 3.0), dtype=np.float32)
    vector, schema = paired_trajectory_feature_vector(
        state,
        _plan(),
        _plan(offset=0.1),
        final_target_xy=(2.0, 0.0),
        obstacles=(),
        position_indices=(0, 1),
        action_names=("v_cmd", "omega_cmd"),
        previous_action=(0.0, 0.0),
        robot_radius=0.1,
        obstacle_influence=0.5,
    )
    metric_dim = len(TRAJECTORY_METRIC_NAMES)
    assert schema["state_feature_dim"] == 3
    assert schema["feature_dim"] == 3 + 3 * metric_dim
    np.testing.assert_array_equal(vector[:3], state)
    base = vector[3:3 + metric_dim]
    candidate = vector[3 + metric_dim:3 + 2 * metric_dim]
    delta = vector[3 + 2 * metric_dim:]
    np.testing.assert_allclose(delta, candidate - base)
    assert schema["common_random_numbers"] is True


def test_trajectory_features_reject_incomplete_diagnostics():
    plan = _plan()
    del plan.diagnostics["cost_q90"]
    with pytest.raises(ValueError, match="incomplete"):
        plan_trajectory_metrics(
            plan,
            final_target_xy=(2.0, 0.0),
            obstacles=(),
            position_indices=(0, 1),
            action_names=("v_cmd", "omega_cmd"),
            previous_action=(0.0, 0.0),
            robot_radius=0.1,
            obstacle_influence=0.5,
        )
