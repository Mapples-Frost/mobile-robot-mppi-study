import numpy as np

from experiments.rl.run_l264_l266_critic_failure_diagnosis import (
    _continuity,
    _decision,
    _local_support,
    _select_l264_indices,
    _spearman,
)


def test_l264_sample_is_scene_balanced_and_unique():
    groups = np.repeat(np.arange(6), 30)
    replay = {
        "groups": groups,
        "dones": np.zeros((groups.size, 1), dtype=np.float32),
    }
    replay["dones"][[5, 35, 65, 95, 125, 155], 0] = 1.0
    indices = _select_l264_indices(replay)
    assert indices.shape == (120,)
    assert np.unique(indices).size == 120
    assert {group: int(np.sum(groups[indices] == group)) for group in range(6)} == {
        group: 20 for group in range(6)
    }


def test_local_support_thresholds_use_only_local_replay_actions():
    replay_observations = np.arange(64, dtype=np.float32)[:, None]
    replay_actions = np.stack((
        np.linspace(-1.0, 1.0, 64), np.zeros(64)
    ), axis=1).astype(np.float32)
    candidates = np.asarray(((0.0, 0.0), (0.01, 0.0), (0.0, 1.0)))
    result = _local_support(
        replay_observations, replay_actions, np.asarray((0.0,)), candidates
    )
    assert result["support"].shape == (3,)
    assert result["support"][2] == "out_of_support"
    assert result["local_action_median"] <= result["local_action_q90"]


def test_continuity_detects_unrecorded_reset_from_observation_jump():
    replay = {
        "transition_ids": np.arange(4),
        "groups": np.asarray((0, 0, 0, 1)),
        "dones": np.zeros((4, 1), dtype=np.float32),
        "observations": np.asarray(((0.0,), (1.0,), (9.0,), (3.0,))),
        "next_observations": np.asarray(((1.0,), (2.0,), (3.0,), (4.0,))),
    }
    _, continuous, episodes = _continuity(replay)
    assert continuous.tolist() == [True, False, False, False]
    assert episodes.tolist() == [0, 0, 1, 2]


def test_spearman_and_decision_follow_frozen_lexicographic_rule():
    assert np.isclose(_spearman([1, 2, 3], [4, 5, 6]), 1.0)
    support = [
        {
            "support": "in_support", "state_metrics_count": 6,
            "mean_spearman": 0.4, "pair_count": 4,
            "recovery_forward_pair_accuracy": 0.75,
        },
        {
            "support": "near_support", "state_metrics_count": 6,
            "mean_spearman": 0.2, "pair_count": 4,
            "recovery_forward_pair_accuracy": 0.5,
        },
        {
            "support": "out_of_support", "state_metrics_count": 6,
            "mean_spearman": 0.1, "pair_count": 4,
            "recovery_forward_pair_accuracy": 0.5,
        },
    ]
    decision = _decision(
        True, support, [{"scene": "scene_0"}] * 6, {0: "scene_0"}
    )
    assert decision["cause"] == "action_extrapolation_dominant"
    bellman = _decision(
        False, support, [{"scene": "scene_0"}] * 6, {0: "scene_0"}
    )
    assert bellman["cause"] == "bellman_implementation_mismatch"
