import numpy as np
import pytest

from mobile_robot_mppi.learning.models import (
    CausalStallGatedResidualDynamics,
)


class _ConstantResidual:
    state_dim = 5
    control_dim = 2
    model = None

    def __init__(self):
        self.reset_count = 0

    def derivative(self, state, control, time=None):
        del control, time
        return np.ones_like(np.asarray(state, dtype=np.float64))

    def reset(self):
        self.reset_count += 1


def _guard(**overrides):
    values = {
        "position_indices": (0, 1),
        "speed_index": 3,
        "speed_threshold_mps": 0.02,
        "maximum_low_risk_probability": 0.05,
        "consecutive_steps": 3,
        "goal_exclusion_distance_m": 0.45,
        "latch_for_episode": True,
    }
    values.update(overrides)
    return CausalStallGatedResidualDynamics(
        _ConstantResidual(), **values
    )


def test_guard_latches_after_consecutive_causal_low_risk_stall():
    guard = _guard()
    state = np.asarray((0.0, 0.0, 0.0, 0.01, 0.0))
    target = np.asarray((1.0, 0.0))
    control = np.zeros(2)

    for _ in range(2):
        assert guard.observe_context(state, target, 0.04) == 1.0
        np.testing.assert_array_equal(
            guard.derivative(state, control), np.ones(5)
        )
    assert guard.observe_context(state, target, 0.04) == 0.0
    np.testing.assert_array_equal(
        guard.derivative(state, control), np.zeros(5)
    )
    diagnostics = guard.diagnostics()
    assert diagnostics["residual_stall_guard_latched"] is True
    assert diagnostics["residual_stall_guard_latch_step"] == 2
    assert diagnostics["residual_stall_guard_authority"] == 0.0


@pytest.mark.parametrize(
    "state,target,probability",
    (
        ((0.0, 0.0, 0.0, 0.03, 0.0), (1.0, 0.0), 0.04),
        ((0.0, 0.0, 0.0, 0.01, 0.0), (1.0, 0.0), 0.06),
        ((0.0, 0.0, 0.0, 0.01, 0.0), (0.4, 0.0), 0.04),
    ),
)
def test_nonqualifying_context_resets_consecutive_counter(
    state, target, probability
):
    guard = _guard()
    qualifying = np.asarray((0.0, 0.0, 0.0, 0.01, 0.0))
    guard.observe_context(qualifying, (1.0, 0.0), 0.04)
    guard.observe_context(state, target, probability)
    assert guard.qualifying_steps == 0
    assert guard.authority == 1.0


def test_episode_reset_restores_residual_authority_and_inner_state():
    guard = _guard(consecutive_steps=1)
    inner_reset_count = guard.residual.reset_count
    guard.observe_context(
        np.asarray((0.0, 0.0, 0.0, 0.0, 0.0)),
        np.asarray((1.0, 0.0)),
        0.0,
    )
    assert guard.authority == 0.0
    guard.reset()
    assert guard.authority == 1.0
    assert guard.latched is False
    assert guard.residual.reset_count == inner_reset_count + 1


@pytest.mark.parametrize(
    "overrides",
    (
        {"position_indices": (0,)},
        {"position_indices": (0, 1), "speed_index": 1},
        {"speed_threshold_mps": -0.01},
        {"maximum_low_risk_probability": 1.1},
        {"consecutive_steps": 0},
        {"goal_exclusion_distance_m": 0.0},
        {"latch_for_episode": False},
    ),
)
def test_invalid_stall_guard_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        _guard(**overrides)
