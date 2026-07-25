import numpy as np
import pytest

from mobile_robot_mppi.obstacles.motion import (
    NOISE_PROFILES,
    PROCESS_NAMES,
    NoiseProfile,
    generate_obstacle_trajectory,
)


@pytest.mark.parametrize("process", PROCESS_NAMES)
def test_registered_motion_processes_are_finite_and_well_formed(process):
    trajectory = generate_obstacle_trajectory(
        process, 730100001, NOISE_PROFILES["medium"], steps=60
    )
    trajectory.validate()
    assert trajectory.states.shape == (61, 4)
    assert trajectory.observations.shape == (61, 2)
    assert np.isfinite(trajectory.states).all()
    if process == "noisy_cv":
        assert not trajectory.change_flags.any()
    else:
        assert trajectory.change_flags.any()
    if process == "combined_change":
        assert (~trajectory.observed_mask).any()
    else:
        assert trajectory.observed_mask.all()


def test_observation_noise_does_not_change_true_motion():
    low_observation_noise = NoiseProfile(
        "obs_low", acceleration_std=0.05, observation_std=0.01
    )
    high_observation_noise = NoiseProfile(
        "obs_high", acceleration_std=0.05, observation_std=0.30
    )
    first = generate_obstacle_trajectory(
        "combined_change", 730100002, low_observation_noise
    )
    second = generate_obstacle_trajectory(
        "combined_change", 730100002, high_observation_noise
    )
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.change_flags, second.change_flags)
    assert np.array_equal(first.observed_mask, second.observed_mask)
    assert not np.allclose(
        first.observations[first.observed_mask],
        second.observations[second.observed_mask],
    )


def test_invalid_motion_inputs_fail_closed():
    with pytest.raises(ValueError):
        generate_obstacle_trajectory(
            "unknown", 1, NOISE_PROFILES["low"]
        )
    with pytest.raises(ValueError):
        generate_obstacle_trajectory(
            "noisy_cv", 1, NOISE_PROFILES["low"], dt=0.0
        )

