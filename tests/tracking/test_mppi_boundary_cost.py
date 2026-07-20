import numpy as np
import pytest

from mobile_robot_mppi.core.references import PolylineReference
from mobile_robot_mppi.core.spaces import body_velocity_action, dynamic_unicycle_state
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController


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


def test_boundary_enforcement_is_opt_in_for_baseline_compatibility():
    values = MppiConfig.from_mapping({}, action_dim=2)

    assert not values.path_boundary_enabled
