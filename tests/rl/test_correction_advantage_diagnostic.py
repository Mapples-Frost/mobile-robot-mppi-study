import pytest

from experiments.rl.run_correction_advantage_diagnostic import (
    _condition_summary,
    _paired_rows,
    _rank,
)


def _episode(condition, seed, success, episode_return, distance, advantage=0.0):
    return {
        "condition": condition,
        "seed": seed,
        "return": episode_return,
        "success": success,
        "collision": False,
        "goal_distance": distance,
        "online_conservative_advantage_mean": advantage,
        "target_conservative_advantage_mean": 0.5 * advantage,
        "advantage_gate_accept_fraction": 0.5,
    }


def test_rank_uses_average_rank_for_ties():
    assert _rank([2.0, 1.0, 2.0]).tolist() == [1.5, 0.0, 1.5]


def test_paired_advantage_summary_preserves_success_losses():
    episodes = [
        _episode("bc", 1, True, 10.0, 0.2),
        _episode("bc", 2, False, 2.0, 0.8),
        _episode("correction_none", 1, False, 5.0, 0.7, advantage=0.4),
        _episode("correction_none", 2, True, 8.0, 0.3, advantage=0.8),
    ]
    paired = _paired_rows(episodes)
    assert paired[0]["bc_success_lost"] is True
    assert paired[0]["return_delta"] == -5.0
    assert paired[0]["goal_distance_improvement"] == pytest.approx(-0.5)
    assert paired[1]["success_gained"] is True

    summary = _condition_summary(paired)["correction_none"]
    assert summary["successes"] == 1
    assert summary["bc_success_losses"] == 1
    assert summary["success_gains"] == 1
    assert summary["mean_return_delta"] == 0.5
    assert summary["online_return_pearson"] == pytest.approx(1.0)
