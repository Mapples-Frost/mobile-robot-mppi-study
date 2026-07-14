import pytest

from mobile_robot_mppi.rl.environment import RewardConfig, _progress_signal


def test_distance_delta_progress_has_no_stationary_reward():
    assert _progress_signal(4.0, 4.0, 0.99, "distance_delta") == pytest.approx(0.0)
    assert _progress_signal(4.0, 3.9, 0.99, "distance_delta") > 0.0
    assert _progress_signal(4.0, 4.1, 0.99, "distance_delta") < 0.0


def test_discounted_potential_mode_preserves_original_expression():
    assert _progress_signal(4.0, 4.0, 0.99, "discounted_potential") == pytest.approx(0.04)


def test_reward_config_parses_new_costs_and_rejects_unknown_mode():
    config = RewardConfig.from_mapping({
        "progress_mode": "distance_delta",
        "distance_penalty_weight": 0.2,
        "path_length_penalty_weight": 0.3,
    })
    config.validate()
    assert config.distance_penalty_weight == pytest.approx(0.2)
    assert config.path_length_penalty_weight == pytest.approx(0.3)

    with pytest.raises(ValueError, match="progress_mode"):
        RewardConfig(progress_mode="unknown").validate()
