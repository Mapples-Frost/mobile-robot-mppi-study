from copy import deepcopy

import pytest

from experiments.dynamic_uncertainty.run_complex_supervised_maneuver_actor_gate_d import (
    DEFAULT_PROTOCOL,
    _load_protocol,
    build_gate_d_config,
)
from experiments.dynamic_uncertainty.analyze_complex_supervised_maneuver_actor_gate_d import (
    gate_decision,
)


def _without_arm_specific_fields(config):
    value = deepcopy(config)
    value["planner"]["paper_rl_driven"].pop(
        "supervised_maneuver_actor"
    )
    value["experiment"].pop("name")
    value.pop("gate_d_contract")
    return value


def test_gate_d_pair_differs_only_by_proposal_actor_interface():
    control = build_gate_d_config(
        DEFAULT_PROTOCOL, "chapter1", 791101019, "frozen_full"
    )
    treatment = build_gate_d_config(
        DEFAULT_PROTOCOL,
        "chapter1",
        791101019,
        "full_plus_bootstrap_actor",
    )

    assert _without_arm_specific_fields(control) == (
        _without_arm_specific_fields(treatment)
    )
    assert not control["planner"]["paper_rl_driven"][
        "supervised_maneuver_actor"
    ]["enabled"]
    assert treatment["planner"]["paper_rl_driven"][
        "supervised_maneuver_actor"
    ]["enabled"]
    for config in (control, treatment):
        assert (
            config["planner"]["num_samples"]
            * config["planner"]["paper_rl_driven"]["iterations"]
            == 600
        )


def test_gate_d_rejects_nonreserved_seed():
    with pytest.raises(ValueError, match="not a frozen Gate D paired block"):
        build_gate_d_config(
            DEFAULT_PROTOCOL,
            "chapter1",
            791101011,
            "full_plus_bootstrap_actor",
        )


def test_gate_d_requires_safe_gain_and_mechanism_contribution():
    protocol = _load_protocol(DEFAULT_PROTOCOL)
    pairs = []
    for index in range(3):
        control = {
            "collision": False,
            "safe_success": False,
            "final_goal_distance": 2.0,
            "paper_total_rollouts_mean": 600.0,
            "supervised_proposal_count_total": 0,
            "supervised_risk_feasible_count_total": 0,
            "supervised_elite_count_total": 0,
            "supervised_selected_count_total": 0,
            "supervised_added_rollout_count_total": 0,
        }
        treatment = {
            **control,
            "safe_success": index == 0,
            "final_goal_distance": 0.1 if index == 0 else 1.5,
            "supervised_proposal_count_total": 30,
            "supervised_risk_feasible_count_total": 10 if index < 2 else 0,
            "supervised_elite_count_total": 1 if index == 0 else 0,
        }
        pairs.append({
            "frozen_full": control,
            "full_plus_bootstrap_actor": treatment,
        })

    result = gate_decision(protocol, pairs)

    assert result["status"] == "pass"
    assert result["success_gains"] == 1
    assert result["collision_regressions"] == 0
    assert result["total_supervised_elite_or_selected_count"] == 1


def test_gate_d_fails_on_actor_collision_regression():
    protocol = _load_protocol(DEFAULT_PROTOCOL)
    base = {
        "collision": False,
        "safe_success": False,
        "final_goal_distance": 2.0,
        "paper_total_rollouts_mean": 600.0,
        "supervised_proposal_count_total": 30,
        "supervised_risk_feasible_count_total": 10,
        "supervised_elite_count_total": 1,
        "supervised_selected_count_total": 0,
        "supervised_added_rollout_count_total": 0,
    }
    pairs = [{
        "frozen_full": {**base, "supervised_proposal_count_total": 0},
        "full_plus_bootstrap_actor": {
            **base,
            "collision": index == 0,
            "safe_success": index == 1,
            "final_goal_distance": 0.1 if index == 1 else 1.5,
        },
    } for index in range(3)]

    result = gate_decision(protocol, pairs)

    assert result["status"] == "fail"
    assert result["collision_regressions"] == 1
