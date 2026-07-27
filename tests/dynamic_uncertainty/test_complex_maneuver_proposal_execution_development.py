from copy import deepcopy

import pytest

from experiments.dynamic_uncertainty.analyze_complex_maneuver_proposal_execution_development import (
    _evaluate_pair,
    _rollout_contract,
)
from experiments.dynamic_uncertainty.run_complex_maneuver_proposal_execution_development import (
    DEFAULT_GATE_D_PROTOCOL,
    DEFAULT_PROTOCOL,
    build_development_config,
)


def _normalized(config):
    value = deepcopy(config)
    value["planner"]["paper_rl_driven"].pop(
        "same_cycle_guided_relative_margin"
    )
    value["experiment"].pop("name")
    value.pop("proposal_execution_contract")
    return value


def test_round1_configs_change_only_the_frozen_shared_margin():
    control = build_development_config(
        DEFAULT_PROTOCOL,
        DEFAULT_GATE_D_PROTOCOL,
        "chapter1",
        791101019,
        "margin0_control",
    )
    treatment = build_development_config(
        DEFAULT_PROTOCOL,
        DEFAULT_GATE_D_PROTOCOL,
        "chapter1",
        791101019,
        "margin002_treatment",
    )

    control_paper = control["planner"]["paper_rl_driven"]
    treatment_paper = treatment["planner"]["paper_rl_driven"]
    assert control_paper["same_cycle_guided_relative_margin"] == 0.0
    assert treatment_paper["same_cycle_guided_relative_margin"] == 0.02
    assert control_paper["supervised_maneuver_actor"]["enabled"]
    assert treatment_paper["supervised_maneuver_actor"]["enabled"]
    assert _normalized(control) == _normalized(treatment)
    assert control["planner"]["num_samples"] * control_paper["iterations"] == 600
    assert (
        treatment["planner"]["num_samples"]
        * treatment_paper["iterations"]
        == 600
    )


@pytest.mark.parametrize(
    "map_name,seed",
    [
        ("chapter1", 791101020),
        ("chapter2", 790202020),
        ("chapter3", 791103020),
        ("chapter4", 791101019),
    ],
)
def test_round1_rejects_unopened_blocks(map_name, seed):
    with pytest.raises(ValueError, match="not an opened development block"):
        build_development_config(
            DEFAULT_PROTOCOL,
            DEFAULT_GATE_D_PROTOCOL,
            map_name,
            seed,
            "margin002_treatment",
        )


def test_rollout_contract_requires_fixed_budget_and_three_heads():
    rows = [
        {
            "paper_total_rollouts": "0",
            "supervised_head_0_proposal_count": "0",
            "supervised_head_1_proposal_count": "0",
            "supervised_head_2_proposal_count": "0",
            "supervised_added_rollout_count": "0",
        },
        {
            "paper_total_rollouts": "600",
            "supervised_head_0_proposal_count": "1",
            "supervised_head_1_proposal_count": "1",
            "supervised_head_2_proposal_count": "1",
            "supervised_added_rollout_count": "0",
        },
    ]
    assert _rollout_contract(rows)["pass"]
    rows[1]["paper_total_rollouts"] = "603"
    assert not _rollout_contract(rows)["pass"]


def _summary(**overrides):
    value = {
        "collision": False,
        "success": False,
        "steps": 100,
        "final_goal_distance": 4.0,
        "safety_interventions": 2,
        "safety_intervention_fraction": 0.02,
        "selected_cycles": 3,
        "counterfactual_active_cycles": 20,
        "influential_cycles": 4,
        "counterfactual_first_action_delta_norm_mean_active": 0.1,
        "influence_survived_guard_fraction_active": 0.2,
        "rollout_contract": {"pass": True},
    }
    value.update(overrides)
    return value


def test_pair_gate_requires_task_and_mechanism_improvement_without_collision():
    control = _summary()
    treatment = _summary(
        final_goal_distance=3.5,
        counterfactual_first_action_delta_norm_mean_active=0.2,
        influence_survived_guard_fraction_active=0.3,
    )
    result = _evaluate_pair(control, treatment)
    assert result["pass"]

    collision = _evaluate_pair(
        control,
        _summary(
            collision=True,
            final_goal_distance=3.5,
            counterfactual_first_action_delta_norm_mean_active=0.2,
            influence_survived_guard_fraction_active=0.3,
        ),
    )
    assert not collision["pass"]
    assert not collision["checks"]["collision_not_worse"]
