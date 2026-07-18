import numpy as np
import pytest

from mobile_robot_mppi.core.spaces import ActionSpec, StateSpec
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController


def _controller(delay):
    state_spec = StateSpec(("x", "y", "theta", "v", "omega"), periodic_indices=(2,))
    action_spec = ActionSpec(
        ("v_cmd", "omega_cmd"),
        np.asarray([0.0, -1.0]), np.asarray([1.0, 1.0]),
        np.asarray([10.0, 10.0]),
    )
    return MppiController(
        DynamicUnicyclePrediction(), state_spec, action_spec,
        MppiConfig(horizon=3, num_samples=2, dt=0.1, command_delay_s=delay),
    )


def test_fractional_delay_uses_interval_average_and_previous_action():
    controller = _controller(0.04)
    controller.previous_action = np.asarray([0.2, -0.4])
    commands = np.asarray([[[0.4, 0.2], [0.8, 0.6], [1.0, -0.2]]])
    effective = controller._prediction_controls(commands)
    expected = np.asarray([[[0.32, -0.04], [0.64, 0.44], [0.92, 0.12]]])
    np.testing.assert_allclose(effective, expected, atol=1e-12)


def test_one_interval_delay_shifts_sequence_exactly():
    controller = _controller(0.1)
    controller.previous_action = np.asarray([0.1, 0.3])
    commands = np.asarray([[[0.4, 0.2], [0.8, 0.6], [1.0, -0.2]]])
    np.testing.assert_allclose(
        controller._prediction_controls(commands),
        [[[0.1, 0.3], [0.4, 0.2], [0.8, 0.6]]],
    )


@pytest.mark.parametrize("delay", (-0.01, 0.10001, float("nan")))
def test_invalid_delay_is_rejected(delay):
    with pytest.raises(ValueError):
        _controller(delay)

