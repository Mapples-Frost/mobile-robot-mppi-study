import json

import numpy as np
import pytest

from src.dynamics import (
    DisturbanceConfig,
    DisturbedUnicycle,
    NominalUnicycle,
    integrate_step,
)


def test_nominal_unicycle_derivative_matches_kinematic_equations():
    model = NominalUnicycle()
    state = np.asarray([4.0, -3.0, np.pi / 2])
    control = np.asarray([2.5, -0.4])

    actual = model.derivative(state, control, time=7.0)

    np.testing.assert_allclose(actual, [0.0, 2.5, -0.4], atol=1e-14)


def test_integrator_wraps_nominal_heading_after_complete_step():
    state = np.asarray([1.0, -2.0, np.pi - 0.05])

    actual = integrate_step(
        NominalUnicycle(), state, [0.0, 1.0], 0.1, method="euler"
    )

    np.testing.assert_allclose(actual, [1.0, -2.0, -np.pi + 0.05])
    assert -np.pi <= actual[2] <= np.pi


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_default_disturbed_model_is_exactly_nominal(method):
    nominal = NominalUnicycle()
    true_model = DisturbedUnicycle(DisturbanceConfig())
    state = np.asarray([0.7, -1.2, 0.35])
    control = np.asarray([1.1, -0.25])

    assert true_model.is_zero_mismatch
    np.testing.assert_array_equal(
        true_model.derivative(state, control, time=0.8),
        nominal.derivative(state, control, time=0.8),
    )
    np.testing.assert_array_equal(
        true_model.step(state, control, 0.05, method=method, time=0.8),
        nominal.step(state, control, 0.05, method=method, time=0.8),
    )


@pytest.mark.parametrize(
    "config, effective_control",
    [
        (DisturbanceConfig(velocity_gain=1.75), [2.1, -0.4]),
        (DisturbanceConfig(yaw_gain=2.5), [1.2, -1.0]),
        (DisturbanceConfig(yaw_bias=0.3), [1.2, -0.1]),
    ],
)
def test_control_gain_and_bias_disturbances_are_applied_independently(
    config, effective_control
):
    state = np.asarray([0.5, -0.25, 0.6])
    command = np.asarray([1.2, -0.4])
    actual = DisturbedUnicycle(config).derivative(state, command, time=0.9)
    expected = NominalUnicycle().derivative(state, effective_control, time=0.9)

    np.testing.assert_allclose(actual, expected)


def test_world_disturbance_adds_three_time_varying_derivative_components():
    amplitude = np.asarray([0.7, -1.2, 0.45])
    frequency = np.asarray([0.5, 1.5, -0.75])
    phase = np.asarray([0.2, -0.4, 1.1])
    time = 0.8
    config = DisturbanceConfig(
        world_disturbance_amplitude=tuple(amplitude),
        world_disturbance_frequency=tuple(frequency),
        world_disturbance_phase=tuple(phase),
    )
    state = np.asarray([0.5, -0.25, 0.6])
    control = np.asarray([1.2, -0.4])

    actual = DisturbedUnicycle(config).derivative(state, control, time=time)
    expected = NominalUnicycle().derivative(state, control, time=time)
    expected = expected + amplitude * np.sin(frequency * time + phase)

    np.testing.assert_allclose(actual, expected)


def test_state_disturbance_adds_position_and_periodic_heading_terms():
    gains = np.asarray([0.2, -0.5, 1.25])
    state = np.asarray([3.0, -2.0, 0.6])
    control = np.asarray([1.2, -0.4])
    config = DisturbanceConfig(state_disturbance_gain=tuple(gains))

    actual = DisturbedUnicycle(config).derivative(state, control, time=1.3)
    expected = NominalUnicycle().derivative(state, control, time=1.3)
    expected = expected + np.asarray(
        [gains[0] * state[0], gains[1] * state[1], gains[2] * np.sin(state[2])]
    )

    np.testing.assert_allclose(actual, expected)


def test_two_step_command_delay_advances_once_per_plant_step():
    plant = DisturbedUnicycle(DisturbanceConfig(control_delay_steps=2))
    state = np.zeros(3)
    commands = [
        np.asarray([1.0, 0.1]),
        np.asarray([2.0, 0.2]),
        np.asarray([3.0, 0.3]),
    ]

    assert len(plant.pending_controls) == 2
    for queued in plant.pending_controls:
        np.testing.assert_array_equal(queued, np.zeros(2))

    state = plant.step(state, commands[0], 0.1, method="euler")
    np.testing.assert_array_equal(plant.last_applied_control, np.zeros(2))
    state = plant.step(state, commands[1], 0.1, method="euler")
    np.testing.assert_array_equal(plant.last_applied_control, np.zeros(2))
    state = plant.step(state, commands[2], 0.1, method="euler")

    np.testing.assert_array_equal(plant.last_applied_control, commands[0])
    np.testing.assert_allclose(state, [0.1, 0.0, 0.01])
    for actual, expected in zip(plant.pending_controls, commands[1:]):
        np.testing.assert_array_equal(actual, expected)


def test_delay_reset_refills_history_with_requested_initial_control():
    plant = DisturbedUnicycle(DisturbanceConfig(control_delay_steps=2))
    plant.step(np.zeros(3), [2.0, -1.0], 0.1, method="euler")
    initial = np.asarray([-0.5, 0.25])

    plant.reset(initial)

    np.testing.assert_array_equal(plant.last_applied_control, initial)
    for queued in plant.pending_controls:
        np.testing.assert_array_equal(queued, initial)
    actual = plant.step(np.zeros(3), [9.0, 9.0], 0.2, method="euler")
    np.testing.assert_allclose(actual, [-0.1, 0.0, 0.05])


def test_clone_copies_delay_history_without_sharing_future_updates():
    plant = DisturbedUnicycle(DisturbanceConfig(control_delay_steps=2))
    plant.step(np.zeros(3), [1.0, 0.1], 0.1, method="euler")
    clone = plant.clone()

    for cloned, original in zip(clone.pending_controls, plant.pending_controls):
        np.testing.assert_array_equal(cloned, original)

    plant.step(np.zeros(3), [2.0, 0.2], 0.1, method="euler")
    clone_snapshot = clone.pending_controls
    plant.step(np.zeros(3), [3.0, 0.3], 0.1, method="euler")

    for actual, expected in zip(clone.pending_controls, clone_snapshot):
        np.testing.assert_array_equal(actual, expected)
    assert not all(
        np.array_equal(cloned, original)
        for cloned, original in zip(clone.pending_controls, plant.pending_controls)
    )


@pytest.mark.parametrize(
    "config",
    [
        DisturbanceConfig(velocity_gain=9.0, enable_velocity_gain=False),
        DisturbanceConfig(yaw_gain=9.0, enable_yaw_gain=False),
        DisturbanceConfig(yaw_bias=9.0, enable_yaw_bias=False),
        DisturbanceConfig(
            world_disturbance_amplitude=(9.0, 9.0, 9.0),
            enable_world_disturbance=False,
        ),
        DisturbanceConfig(
            state_disturbance_gain=(9.0, 9.0, 9.0),
            enable_state_disturbance=False,
        ),
    ],
)
def test_each_instantaneous_mismatch_can_be_disabled(config):
    state = np.asarray([0.7, -0.2, 0.4])
    control = np.asarray([0.8, -0.3])

    actual = DisturbedUnicycle(config).derivative(state, control, time=1.2)

    np.testing.assert_array_equal(
        actual, NominalUnicycle().derivative(state, control, time=1.2)
    )


def test_failed_delay_step_does_not_mutate_history():
    plant = DisturbedUnicycle(DisturbanceConfig(control_delay_steps=2))
    before = plant.pending_controls

    with pytest.raises(ValueError, match="dt|greater than zero"):
        plant.step(np.zeros(3), [1.0, 0.2], 0.0)

    for actual, expected in zip(plant.pending_controls, before):
        np.testing.assert_array_equal(actual, expected)


def test_delay_preview_and_derivative_are_queue_pure():
    plant = DisturbedUnicycle(DisturbanceConfig(control_delay_steps=1))
    before = plant.pending_controls

    np.testing.assert_array_equal(plant.preview_applied_control([1.0, 0.2]), [0.0, 0.0])
    plant.derivative([0.0, 0.0, 0.0], [1.0, 0.2], time=0.0)

    for actual, expected in zip(plant.pending_controls, before):
        np.testing.assert_array_equal(actual, expected)


def test_disturbance_metadata_is_json_serializable_and_marks_delay_history():
    plant = DisturbedUnicycle(
        DisturbanceConfig(control_delay_steps=1, velocity_gain=0.8)
    )

    metadata = plant.disturbance_metadata()

    assert metadata["delay_is_non_markov_without_history"] is True
    assert metadata["config"]["velocity_gain"] == pytest.approx(0.8)
    assert json.loads(json.dumps(metadata))["config"]["control_delay_steps"] == 1
