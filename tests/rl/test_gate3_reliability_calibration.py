import numpy as np
import pytest

from experiments.rl.calibrate_gate3_reliability import (
    episode_level_summary,
    evaluate_episode_gate,
    validate_actor_representation,
)
from mobile_robot_mppi.rl.observation import RunningNormalizer


def _row(episode, authority, error):
    return {
        "episode_id": episode,
        "split": "validation",
        "seed": int(episode[-1]),
        "scene": "clean",
        "physics_domain": "test",
        "authority": authority,
        "rollout_error": error,
        "terminal_error": error,
    }


def test_episode_gate_counts_episodes_not_overlapping_windows():
    rows = []
    for episode, authority, error in (
        ("e0", 0.1, 3.0),
        ("e1", 0.2, 2.8),
        ("e2", 0.8, 1.0),
        ("e3", 0.9, 1.2),
    ):
        rows.extend([_row(episode, authority, error)] * 20)
    episodes = episode_level_summary(rows)
    gate = evaluate_episode_gate(
        episodes,
        {
            "minimum_episodes_per_occupied_bin": 2,
            "minimum_occupied_bins": 2,
        },
    )

    assert gate["episode_count"] == 4
    assert gate["level_episode_counts"] == {
        "low": 2, "medium": 0, "high": 2
    }
    assert gate["passed"]


def test_episode_gate_rejects_single_low_confidence_episode():
    episodes = episode_level_summary([
        _row("e0", 0.1, 3.0),
        _row("e1", 0.8, 1.0),
        _row("e2", 0.9, 1.2),
    ])
    gate = evaluate_episode_gate(
        episodes,
        {
            "minimum_episodes_per_occupied_bin": 2,
            "minimum_occupied_bins": 2,
        },
    )

    assert not gate["passed"]
    assert gate["level_episode_counts"]["low"] == 1


def test_path_calibration_requires_actor_dimension_and_valid_context():
    normalizer = RunningNormalizer(5)
    dataset = {
        "raw_observation_t_plus_1": np.zeros((3, 5)),
        "path_context_t_plus_1": np.asarray([
            [0.1, 0.0, 1.0, 0.2, 0.9, 1.0],
            [-0.2, 0.1, 0.9, -0.3, 0.5, 1.0],
            [0.0, -0.1, 0.9, 0.0, 0.1, 1.0],
        ]),
    }

    audit = validate_actor_representation(
        dataset,
        normalizer,
        {"require_path_context": True},
        "validation",
    )

    assert audit["actor_observation_dimension"] == 5
    assert audit["path_valid_fraction"] == 1.0


def test_path_calibration_rejects_point_goal_or_old_actor_data():
    normalizer = RunningNormalizer(5)
    wrong_dimension = {
        "raw_observation_t_plus_1": np.zeros((3, 4)),
        "path_context_t_plus_1": np.ones((3, 6)),
    }
    with pytest.raises(ValueError, match="Actor observations"):
        validate_actor_representation(
            wrong_dimension,
            normalizer,
            {"require_path_context": True},
            "test",
        )

    point_goal = {
        "raw_observation_t_plus_1": np.zeros((3, 5)),
        "path_context_t_plus_1": np.zeros((3, 6)),
    }
    with pytest.raises(ValueError, match="non-polyline"):
        validate_actor_representation(
            point_goal,
            normalizer,
            {"require_path_context": True},
            "unseen",
        )


def test_continuous_rank_gate_uses_episode_bootstrap_and_tail_effect():
    episodes = episode_level_summary([
        _row(
            "e%02d" % index,
            0.05 + 0.05 * index,
            0.40 - 0.015 * index,
        )
        for index in range(16)
    ])
    gate = evaluate_episode_gate(
        episodes,
        {
            "gate_mode": "continuous_rank",
            "minimum_episode_count": 16,
            "tail_fraction": 0.25,
            "minimum_tail_episodes": 3,
            "maximum_rank_correlation": -0.5,
            "maximum_rank_ci_upper": 0.0,
            "minimum_relative_tail_separation": 0.2,
            "bootstrap_samples": 500,
            "bootstrap_seed": 19,
        },
    )

    assert gate["passed"]
    assert gate["tail_episode_count"] == 4
    assert gate["authority_error_rank_correlation"] == pytest.approx(-1.0)
    assert gate["authority_error_rank_correlation_ci95"][1] < 0.0
    assert gate["relative_tail_error_separation"] > 0.2
