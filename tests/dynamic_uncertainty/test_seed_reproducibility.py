import numpy as np

from mobile_robot_mppi.obstacles.motion import (
    NOISE_PROFILES,
    PROCESS_NAMES,
    generate_obstacle_trajectory,
)


def test_every_process_is_bitwise_reproducible_for_one_seed():
    for process in PROCESS_NAMES:
        first = generate_obstacle_trajectory(
            process, 730100003, NOISE_PROFILES["medium"]
        )
        second = generate_obstacle_trajectory(
            process, 730100003, NOISE_PROFILES["medium"]
        )
        assert np.array_equal(first.times, second.times)
        assert np.array_equal(first.states, second.states)
        assert np.array_equal(
            first.observations, second.observations, equal_nan=True
        )
        assert np.array_equal(first.observed_mask, second.observed_mask)
        assert first.true_modes == second.true_modes
        assert first.metadata == second.metadata


def test_different_seeds_change_stochastic_realization():
    first = generate_obstacle_trajectory(
        "direction_change", 730100003, NOISE_PROFILES["medium"]
    )
    second = generate_obstacle_trajectory(
        "direction_change", 730100004, NOISE_PROFILES["medium"]
    )
    assert not np.array_equal(first.states, second.states)

