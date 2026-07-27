from copy import deepcopy

import pytest

from experiments.dynamic_uncertainty.run_complex_supervised_maneuver_actor_gate_d import (
    DEFAULT_PROTOCOL,
    _load_protocol,
    build_gate_d_config,
)
from experiments.dynamic_uncertainty.analyze_complex_supervised_maneuver_actor_gate_d import (
    _metric,
    _normalized_pair_config,
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
    protocol = _load_protocol(DEFAULT_PROTOCOL)
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
    assert protocol["online_interface"]["supervised_head_rows"] == 3
    assert (
        protocol["online_interface"][
            "minimum_actor_enabled_guided_allocation"
        ]
        == protocol["online_interface"]["supervised_head_rows"]
        + protocol["online_interface"][
            "deterministic_braking_reserve_rows"
        ]
        == 4
    )


def test_gate_d_rejects_nonreserved_seed():
    with pytest.raises(ValueError, match="not a frozen Gate D paired block"):
        build_gate_d_config(
            DEFAULT_PROTOCOL,
            "chapter1",
            791101011,
            "full_plus_bootstrap_actor",
        )


def test_gate_d_analysis_accepts_runtime_metric_names_and_legacy_aliases():
    runtime_metrics = {
        "trajectory_length": 4.5,
        "safety_interventions": 7,
    }
    assert _metric(
        runtime_metrics, "trajectory_length", "path_length"
    ) == pytest.approx(4.5)
    assert _metric(
        runtime_metrics, "safety_interventions", "safety_overrides"
    ) == 7

    legacy_metrics = {"path_length": 3.25, "safety_overrides": 2}
    assert _metric(
        legacy_metrics, "trajectory_length", "path_length"
    ) == pytest.approx(3.25)
    assert _metric(
        legacy_metrics, "safety_interventions", "safety_overrides"
    ) == 2


def test_gate_d_analysis_normalizes_only_arm_specific_config(tmp_path):
    control = build_gate_d_config(
        DEFAULT_PROTOCOL, "chapter2", 790202019, "frozen_full"
    )
    treatment = build_gate_d_config(
        DEFAULT_PROTOCOL,
        "chapter2",
        790202019,
        "full_plus_bootstrap_actor",
    )
    control_path = tmp_path / "control.yaml"
    treatment_path = tmp_path / "treatment.yaml"
    import yaml

    control_path.write_text(yaml.safe_dump(control), encoding="utf-8")
    treatment_path.write_text(yaml.safe_dump(treatment), encoding="utf-8")
    assert _normalized_pair_config(control_path) == (
        _normalized_pair_config(treatment_path)
    )

    treatment["planner"]["num_samples"] += 1
    treatment_path.write_text(yaml.safe_dump(treatment), encoding="utf-8")
    assert _normalized_pair_config(control_path) != (
        _normalized_pair_config(treatment_path)
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
            "per_cycle_rollout_counts": [0, 600],
            "active_paper_optimizer_budget_valid": True,
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
        "per_cycle_rollout_counts": [0, 600],
        "active_paper_optimizer_budget_valid": True,
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
