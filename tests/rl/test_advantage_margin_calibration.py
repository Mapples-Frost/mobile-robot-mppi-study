import copy

import pytest

from experiments.rl.run_correction_advantage_margin_grid import (
    _margin_condition,
    _margins,
)
from mobile_robot_mppi.rl.calibration import (
    AdvantageMarginSelectionConfig,
    select_advantage_margin,
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
        float(margin): copy.deepcopy(default)
        for margin in AdvantageMarginSelectionConfig().candidate_margins
    }


def test_margin_grid_parser_and_condition_names_are_canonical():
    assert _margins("0,0.005,0.01") == [0.0, 0.005, 0.01]
    assert _margin_condition(0.0) == "target_margin_0"
    assert _margin_condition(0.005) == "target_margin_0p005"
    with pytest.raises(ValueError, match="strictly increasing"):
        _margins("0.01,0.005")


def test_margin_selector_uses_preregistered_lexicographic_rank():
    candidates = _grid()
    candidates[0.01] = _candidate(gains=1, return_delta=1.0)
    candidates[0.02] = _candidate(gains=2, return_delta=0.2)
    candidates[0.04] = _candidate(gains=2, return_delta=0.5)
    result = select_advantage_margin(candidates)
    assert result["selected_mode"] == "target_advantage_margin"
    assert result["selected_margin"] == 0.04
    assert result["selection_reason"] == "best_preregistered_eligible_margin"


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
def test_margin_selector_fails_closed(candidate):
    result = select_advantage_margin(_grid(candidate))
    assert result["selected_mode"] == "bc_fallback"
    assert result["selected_margin"] is None


def test_margin_selector_rejects_grid_or_training_seed_drift():
    candidates = _grid(_candidate())
    candidates.pop(0.08)
    with pytest.raises(ValueError, match="grid"):
        select_advantage_margin(candidates)

    candidates = _grid(_candidate())
    candidates[0.08]["per_training_seed"].pop("3")
    with pytest.raises(ValueError, match="different training seeds"):
        select_advantage_margin(candidates)


@pytest.mark.parametrize(
    "values,match",
    [
        ({"candidate_margins": [0.0, 0.0]}, "strictly increasing"),
        ({"minimum_training_seed_accept_fraction": 0.8,
          "maximum_training_seed_accept_fraction": 0.2}, "interval"),
    ],
)
def test_margin_selector_config_validation(values, match):
    with pytest.raises(ValueError, match=match):
        AdvantageMarginSelectionConfig.from_mapping(values).validate()
