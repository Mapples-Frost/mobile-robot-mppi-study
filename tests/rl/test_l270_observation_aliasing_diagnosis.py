import numpy as np

from experiments.rl.run_l270_observation_aliasing_diagnosis import (
    _even_indices,
    _neighbor_conflicts,
)


def test_even_indices_are_unique_bounded_and_equal_weighted():
    values = _even_indices(length=75, minimum=2, count=12)
    assert len(values) == 12
    assert len(set(values.tolist())) == 12
    assert values[0] == 2
    assert values[-1] == 74


def test_neighbor_conflict_detects_close_opposite_steering():
    oracles = [
        {
            "state_id": "a", "scene": "s1", "scene_role": "train", "kind": "k",
            "observation": np.zeros(69), "oracle_omega": -1.0,
        },
        {
            "state_id": "b", "scene": "s1", "scene_role": "train", "kind": "k",
            "observation": np.full(69, 1e-4), "oracle_omega": 1.0,
        },
        {
            "state_id": "c", "scene": "s2", "scene_role": "validation", "kind": "k",
            "observation": np.ones(69), "oracle_omega": 0.0,
        },
    ]
    rows, summary = _neighbor_conflicts(oracles, {
        "nearest_neighbors": 1,
        "close_pair_quantile": 0.25,
        "omega_sign_minimum": 0.25,
        "omega_conflict_delta": 0.75,
    })
    close = [row for row in rows if row["close_pair"]]
    assert close
    assert all(row["steering_conflict"] for row in close)
    assert summary["close_pair_conflict_fraction"] == 1.0

