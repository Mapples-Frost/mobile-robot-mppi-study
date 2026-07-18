import copy

import pytest

from experiments.rl.run_correction_advantage_lcb_grid import (
    _beta_condition,
    _betas,
)
from mobile_robot_mppi.rl.calibration import (
    CriticConsensusSelectionConfig,
    select_critic_consensus_beta,
)


def _candidate(
    losses=0,
    collisions=0,
    gains=1,
    distance=0.01,
    return_delta=1.0,
    accept=0.5,
):
    return {
        "pooled": {
            "bc_success_losses": losses,
            "collision_regressions": collisions,
            "success_gains": gains,
            "mean_return_delta": return_delta,
            "mean_goal_distance_improvement": distance,
        },
        "per_training_seed": {
            "1": {
                "mean_return_delta": return_delta,
                "mean_gate_accept_fraction": accept,
            },
            "2": {
                "mean_return_delta": return_delta + 0.1,
                "mean_gate_accept_fraction": accept + 0.05,
            },
            "3": {
                "mean_return_delta": return_delta + 0.2,
                "mean_gate_accept_fraction": accept - 0.05,
            },
        },
    }


def _grid(default=None):
    default = default or _candidate(losses=1)
    return {
        float(beta): copy.deepcopy(default)
        for beta in CriticConsensusSelectionConfig().candidate_betas
    }


def test_beta_grid_parser_and_condition_names_are_canonical():
    assert _betas("1,1.5,2") == [1.0, 1.5, 2.0]
    assert _beta_condition(1.0) == "target_lcb_beta_1"
    assert _beta_condition(1.5) == "target_lcb_beta_1p5"
    with pytest.raises(ValueError, match="at least one"):
        _betas("0.5,1")


def test_consensus_selector_uses_preregistered_lexicographic_rank():
    candidates = _grid()
    candidates[2.0] = _candidate(gains=1, return_delta=1.0)
    candidates[3.0] = _candidate(gains=2, return_delta=0.2)
    candidates[5.0] = _candidate(gains=2, return_delta=0.5)
    result = select_critic_consensus_beta(candidates)
    assert result["selected_mode"] == "target_advantage_consensus_lcb"
    assert result["selected_beta"] == 5.0


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(losses=1),
        _candidate(collisions=1),
        _candidate(gains=0, distance=0.0),
        _candidate(return_delta=-0.01),
        _candidate(accept=0.99),
    ],
)
def test_consensus_selector_fails_closed(candidate):
    result = select_critic_consensus_beta(_grid(candidate))
    assert result["selected_mode"] == "bc_fallback"
    assert result["selected_beta"] is None


def test_consensus_selector_rejects_grid_or_training_seed_drift():
    candidates = _grid(_candidate())
    candidates.pop(8.0)
    with pytest.raises(ValueError, match="grid"):
        select_critic_consensus_beta(candidates)

    candidates = _grid(_candidate())
    candidates[8.0]["per_training_seed"].pop("3")
    with pytest.raises(ValueError, match="different training seeds"):
        select_critic_consensus_beta(candidates)


@pytest.mark.parametrize(
    "values,match",
    [
        ({"candidate_betas": [0.5, 1.0]}, "at least one"),
        ({"candidate_betas": [1.0, 1.0]}, "strictly increasing"),
        (
            {
                "minimum_training_seed_accept_fraction": 0.8,
                "maximum_training_seed_accept_fraction": 0.2,
            },
            "interval",
        ),
    ],
)
def test_consensus_selector_config_validation(values, match):
    with pytest.raises(ValueError, match=match):
        CriticConsensusSelectionConfig.from_mapping(values).validate()
