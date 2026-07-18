import pytest

from experiments.rl.build_factorial_onpolicy_residual_dataset import (
    _split_episode_units,
)


def _design():
    return {"development_episode_seeds": [1, 2, 3, 4]}


def test_scene_and_seed_selectors_are_episode_disjoint():
    specification = {
        "split_selectors": {
            "train": {"scenes": ["a", "b"], "seeds": [1, 2]},
            "validation": {"scenes": ["a", "b"], "seeds": [3]},
            "test": {"scenes": ["a", "b"], "seeds": [4]},
            "unseen": {"scenes": ["c"], "seeds": [1, 2, 3, 4]},
        }
    }
    units, contract = _split_episode_units(
        specification, _design(), ("a", "b", "c"), ("plant",)
    )
    flat = [unit for values in units.values() for unit in values]
    assert len(flat) == len(set(flat)) == 12
    assert {unit[0] for unit in units["unseen"]} == {"c"}
    assert contract["train"]["seeds"] == [1, 2]


def test_selector_rejects_episode_overlap_and_unknown_names():
    overlap = {
        "split_selectors": {
            "train": {"scenes": ["a"], "seeds": [1]},
            "validation": {"scenes": ["a"], "seeds": [1]},
            "test": {"scenes": ["b"], "seeds": [2]},
            "unseen": {"scenes": ["c"], "seeds": [3]},
        }
    }
    with pytest.raises(ValueError, match="appears in both"):
        _split_episode_units(overlap, _design(), ("a", "b", "c"), ("plant",))

    unknown = {
        "split_selectors": {
            "train": {"scenes": ["missing"], "seeds": [1]},
            "validation": {"scenes": ["a"], "seeds": [2]},
            "test": {"scenes": ["b"], "seeds": [3]},
            "unseen": {"scenes": ["c"], "seeds": [4]},
        }
    }
    with pytest.raises(ValueError, match="unknown scene"):
        _split_episode_units(unknown, _design(), ("a", "b", "c"), ("plant",))


def test_legacy_seed_splits_remain_supported():
    specification = {
        "splits": {
            "train": [1], "validation": [2], "test": [3], "unseen": [4]
        }
    }
    units, _ = _split_episode_units(
        specification, _design(), ("a", "b"), ("plant",)
    )
    assert len(units["train"]) == 2
    assert ("a", "plant", 1) in units["train"]

