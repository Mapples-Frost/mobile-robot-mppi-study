import pytest

from experiments.dynamic_uncertainty.run_dynamic_actor_mujoco_probe import (
    _extended_summary,
    _resolve_arm_order,
)
from experiments.dynamic_uncertainty.run_dynamic_actor_expanded_development import (
    _aggregate,
    _evaluate_gate,
    _nonregressive,
    _probe_arguments_and_expectations,
)


def test_dynamic_actor_probe_arm_order_is_explicit_and_balancable():
    assert _resolve_arm_order("source_first") == (
        "source_actor",
        "dynamic_actor",
    )
    assert _resolve_arm_order("candidate_first") == (
        "dynamic_actor",
        "source_actor",
    )
    with pytest.raises(ValueError, match="unknown dynamic Actor arm order"):
        _resolve_arm_order("alternating")


def test_dynamic_actor_probe_exposes_same_cycle_filter_mechanism_metrics():
    summary = _extended_summary({
        "paper_same_cycle_guided_cost_filter_enabled_fraction": 1.0,
        "paper_same_cycle_guided_cost_filter_active_fraction": 0.625,
        "paper_same_cycle_guided_cost_filter_iterations_total": 75,
        "paper_same_cycle_guided_filtered_candidates_total": 9600,
    })

    assert summary["same_cycle_filter_enabled_fraction"] == 1.0
    assert summary["same_cycle_filter_active_fraction"] == 0.625
    assert summary["same_cycle_filter_iterations_total"] == 75
    assert summary["same_cycle_filtered_candidates_total"] == 9600


def test_expanded_protocol_binds_pareto_forward_probe_settings():
    arguments, expectations = _probe_arguments_and_expectations({
        "probe_settings": {
            "completion_handover_full_fallback_distance": 0.4,
            "completion_handover_full_rl_distance": 0.8,
            "vetted_reactive_escape": True,
            "dynamic_escape_min_probability_mass_relative_improvement": 0.1,
            "probabilistic_emergency_trigger_ttc_s": 1.5,
            "probabilistic_emergency_intent_hold_steps": 16,
            "probabilistic_emergency_pareto_forward_commit": True,
            "probabilistic_emergency_forward_risk_ceiling": 0.005,
            "probabilistic_emergency_forward_mass_ceiling": 0.02,
        }
    })

    assert "--probabilistic-emergency-pareto-forward-commit" in arguments
    assert expectations[
        "probabilistic_emergency_pareto_forward_commit"
    ] is True
    assert expectations[
        "probabilistic_emergency_intent_hold_steps"
    ] == 16
    assert "--probabilistic-emergency-forward-risk-ceiling" in arguments
    assert "--probabilistic-emergency-forward-mass-ceiling" in arguments
    assert expectations[
        "probabilistic_emergency_forward_risk_ceiling"
    ] == 0.005
    assert expectations[
        "probabilistic_emergency_forward_mass_ceiling"
    ] == 0.02


def _pair(source_success=False, candidate_success=True):
    def arm(success, steps, distance):
        return {
            "success": success,
            "collision": False,
            "steps": steps,
            "final_goal_distance_m": distance,
            "same_cycle_filter_enabled_fraction": 1.0,
            "same_cycle_filter_iterations_total": 5,
            "same_cycle_filtered_candidates_total": 640,
        }

    return {
        "results": {
            "source_actor": arm(source_success, 400, 1.0),
            "dynamic_actor": arm(candidate_success, 350, 0.3),
        }
    }


def test_expanded_development_gate_requires_complete_safe_paired_gain():
    pooled = _aggregate([_pair()] * 24)
    batch = _aggregate([_pair()] * 8)
    batch["nonregressive"] = _nonregressive(batch)
    gate = _evaluate_gate(
        pooled,
        {"batch1": batch, "batch2": batch, "batch3": batch},
        protocol_integrity=True,
    )

    assert gate["passed"] is True
    assert all(gate["checks"].values())


def test_expanded_development_gate_rejects_lost_source_success():
    records = [_pair()] * 23 + [_pair(True, False)]
    pooled = _aggregate(records)
    batch = _aggregate([_pair()] * 8)
    batch["nonregressive"] = _nonregressive(batch)
    gate = _evaluate_gate(
        pooled,
        {"batch1": batch, "batch2": batch, "batch3": batch},
        protocol_integrity=True,
    )

    assert gate["passed"] is False
    assert gate["checks"]["zero_lost_source_successes"] is False


def test_outcome_aware_efficiency_does_not_reward_early_failure():
    record = _pair(source_success=False, candidate_success=True)
    record["results"]["source_actor"]["steps"] = 100
    record["results"]["dynamic_actor"]["steps"] = 350
    records = [record] * 24

    raw_pooled = _aggregate(records)
    raw_batch = _aggregate(records[:8])
    raw_batch["nonregressive"] = _nonregressive(raw_batch)
    raw_gate = _evaluate_gate(
        raw_pooled,
        {"batch1": raw_batch, "batch2": raw_batch, "batch3": raw_batch},
        protocol_integrity=True,
    )
    assert raw_gate["checks"]["total_steps_noninferior"] is False

    pooled = _aggregate(records, failure_step_penalty=400)
    batch = _aggregate(records[:8], failure_step_penalty=400)
    batch["nonregressive"] = _nonregressive(
        batch,
        use_outcome_aware_efficiency=True,
    )
    gate = _evaluate_gate(
        pooled,
        {"batch1": batch, "batch2": batch, "batch3": batch},
        protocol_integrity=True,
        use_outcome_aware_efficiency=True,
        efficiency_relative_noninferiority_margin=0.01,
    )

    assert pooled["source_efficiency_steps"] == 24 * 400
    assert pooled["candidate_efficiency_steps"] == 24 * 350
    assert gate["passed"] is True
