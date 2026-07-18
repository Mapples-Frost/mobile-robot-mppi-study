import numpy as np
import pytest

from mobile_robot_mppi.core.spaces import ActionSpec
from mobile_robot_mppi.rl.environment import DirectControlEnv, MppiPriorEnv
from mobile_robot_mppi.rl.trainer import environment_class_from_config


def _direct_environment_without_backend(rate_limits=None):
    environment = DirectControlEnv.__new__(DirectControlEnv)
    environment.action_spec = ActionSpec(
        names=("v_cmd", "omega_cmd"),
        lower=np.asarray((-0.2, -1.5), dtype=np.float64),
        upper=np.asarray((0.6, 1.5), dtype=np.float64),
        rate_limits=(
            None
            if rate_limits is None
            else np.asarray(rate_limits, dtype=np.float64)
        ),
    )
    environment.previous_control = np.asarray((0.0, 0.0), dtype=np.float64)
    environment.config = {"experiment": {"control_dt": 0.1}}
    return environment


def test_direct_control_action_maps_normalized_bounds_to_physical_bounds():
    environment = _direct_environment_without_backend()

    np.testing.assert_allclose(
        environment.normalized_to_physical_action((-1.0, -1.0)),
        (-0.2, -1.5),
    )
    np.testing.assert_allclose(
        environment.normalized_to_physical_action((1.0, 1.0)),
        (0.6, 1.5),
    )
    np.testing.assert_allclose(
        environment.normalized_to_physical_action((0.0, 0.0)),
        (0.2, 0.0),
    )


def test_direct_control_action_obeys_physical_rate_limits():
    environment = _direct_environment_without_backend(rate_limits=(1.0, 2.0))

    physical = environment.normalized_to_physical_action((1.0, 1.0))

    np.testing.assert_allclose(physical, (0.1, 0.2))


@pytest.mark.parametrize(
    "action, message",
    [
        ((0.0,), "shape"),
        ((0.0, np.nan), "finite"),
        ((1.01, 0.0), r"\[-1, 1\]"),
    ],
)
def test_direct_control_action_fails_closed_on_invalid_policy_output(
    action, message
):
    environment = _direct_environment_without_backend()

    with pytest.raises(ValueError, match=message):
        environment.normalized_to_physical_action(action)


def test_environment_action_mode_is_explicit_and_legacy_default_is_preserved():
    assert environment_class_from_config({}) is MppiPriorEnv
    assert environment_class_from_config(
        {"rl": {"training": {"action_mode": "mppi_prior"}}}
    ) is MppiPriorEnv
    assert environment_class_from_config(
        {"rl": {"training": {"action_mode": "direct_control"}}}
    ) is DirectControlEnv

    with pytest.raises(ValueError, match="action_mode"):
        environment_class_from_config(
            {"rl": {"training": {"action_mode": "unknown"}}}
        )
