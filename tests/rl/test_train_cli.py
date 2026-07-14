import pytest

from experiments.rl.train_rl_sampling_prior import _apply_training_seed


def test_training_seed_override_updates_training_and_experiment_seed():
    config = {
        "experiment": {"seed": 1},
        "rl": {"training": {"seed": 2}},
    }
    result = _apply_training_seed(config, 20260719)

    assert result is config
    assert config["experiment"]["seed"] == 20260719
    assert config["rl"]["training"]["seed"] == 20260719


def test_training_seed_override_can_be_omitted_and_validates_range():
    config = {"experiment": {"seed": 4}}
    assert _apply_training_seed(config, None) is config
    assert config["experiment"]["seed"] == 4

    with pytest.raises(ValueError, match="--seed"):
        _apply_training_seed(config, -1)
