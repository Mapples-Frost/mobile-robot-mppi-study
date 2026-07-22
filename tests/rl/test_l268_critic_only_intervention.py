import numpy as np
import torch

from experiments.rl.run_l268_critic_only_intervention import (
    _buffer_from_arrays,
    _sample_chain_balanced,
    _tree_hash,
)


def test_l268_chain_balanced_sampling_gives_equal_chain_quota():
    chain_ids = np.repeat(np.arange(12), np.arange(12) + 20)
    pool = {"chain_ids": chain_ids}
    selected, audit = _sample_chain_balanced(
        pool, 300, np.random.RandomState(20263071)
    )
    assert selected.shape == (300,)
    sampled = chain_ids[selected]
    assert {int(value): int(np.sum(sampled == value)) for value in range(12)} == {
        value: 25 for value in range(12)
    }
    assert all(row["sample_count"] == 25 for row in audit.values())


def test_l268_buffer_contract_preserves_six_groups():
    count = 6000
    arrays = {
        "observations": np.zeros((count, 69), dtype=np.float32),
        "actions": np.zeros((count, 2), dtype=np.float32),
        "rewards": np.zeros((count, 1), dtype=np.float32),
        "constraint_costs": np.zeros((count, 1), dtype=np.float32),
        "next_observations": np.ones((count, 69), dtype=np.float32),
        "dones": np.zeros((count, 1), dtype=np.float32),
        "groups": np.repeat(np.arange(6), 1000).astype(np.int32),
        "outcomes": np.full((count,), -1, dtype=np.int8),
        "transition_ids": np.arange(count, dtype=np.int64),
    }
    replay = _buffer_from_arrays(arrays, seed=7)
    assert replay.size == 6000
    assert replay.group_counts() == {value: 1000 for value in range(6)}
    batch = replay.sample(256, strategy="scene_balanced")
    counts = np.bincount(batch["groups"], minlength=6)
    assert int(counts.max() - counts.min()) <= 1


def test_l268_tree_hash_detects_mutation_and_ignores_dict_order():
    first = {"b": [torch.tensor([1.0, 2.0])], "a": {"x": 3}}
    reordered = {"a": {"x": 3}, "b": [torch.tensor([1.0, 2.0])]}
    changed = {"a": {"x": 3}, "b": [torch.tensor([1.0, 2.1])]}
    assert _tree_hash(first) == _tree_hash(reordered)
    assert _tree_hash(first) != _tree_hash(changed)
