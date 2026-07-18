import pytest

from experiments.rl.calibrate_gate3_reliability import (
    episode_level_summary,
    evaluate_episode_gate,
)


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
