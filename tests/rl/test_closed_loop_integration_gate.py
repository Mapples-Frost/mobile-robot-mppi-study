import copy

import pytest

from experiments.rl.analyze_closed_loop_integration_gate import analyze


ARMS = (
    "ordinary_fixed",
    "value_fixed",
    "ordinary_adaptive",
    "full_proposed",
)


def _rows():
    rows = []
    for seed in (558, 559, 560):
        for arm in ARMS:
            adaptive = arm in ("ordinary_adaptive", "full_proposed")
            cross_track = {
                "ordinary_fixed": 0.10,
                "value_fixed": 0.095,
                "ordinary_adaptive": 0.096,
                "full_proposed": 0.094,
            }[arm]
            jerk = {
                "ordinary_fixed": 0.20,
                "value_fixed": 0.19,
                "ordinary_adaptive": 0.19,
                "full_proposed": 0.205,
            }[arm]
            rows.append({
                "factorial_arm": arm,
                "seed": seed,
                "success": "True",
                "collision": "False",
                "cross_track_rmse": cross_track,
                "control_jerk": jerk,
                "rollout_budget_per_decision": 100,
                "paper_iterations": 2,
                "reliability_authority_mean": 0.7 if adaptive else 1.0,
                "reliability_guided_fraction_applied_mean": (
                    0.25 if adaptive else 0.30
                ),
                "reliability_low_fraction": 0.2 if adaptive else 0.0,
                "reliability_medium_fraction": 0.3 if adaptive else 0.0,
                "reliability_high_fraction": 0.5 if adaptive else 0.0,
            })
    return rows


def test_closed_loop_gate_passes_complete_positive_block():
    result = analyze(_rows(), [558, 559, 560])
    assert result["gate_passed"] is True
    assert result["episode_count"] == 12
    assert result["checks"]["adaptive_mechanism_exercised"] is True


def test_closed_loop_gate_rejects_unexercised_adaptive_mechanism():
    rows = _rows()
    for row in rows:
        if row["factorial_arm"] == "full_proposed":
            row["reliability_low_fraction"] = 0.0
            row["reliability_medium_fraction"] = 0.0
            row["reliability_high_fraction"] = 1.0
    result = analyze(rows, [558, 559, 560])
    assert result["gate_passed"] is False
    assert (
        result["adaptive_diagnostics"]["full_proposed"]["passed"]
        is False
    )


def test_closed_loop_gate_rejects_budget_drift_and_tracking_regression():
    rows = _rows()
    changed = copy.deepcopy(rows)
    changed[0]["rollout_budget_per_decision"] = 120
    for row in changed:
        if row["factorial_arm"] == "full_proposed":
            row["cross_track_rmse"] = 0.20
    result = analyze(changed, [558, 559, 560])
    assert result["gate_passed"] is False
    assert result["checks"]["equal_frozen_budget"] is False
    assert (
        result["checks"]["tracking_vs_ordinary_within_tolerance"]
        is False
    )


def test_closed_loop_gate_rejects_incomplete_or_duplicate_blocks():
    with pytest.raises(ValueError, match="expected 12 episodes"):
        analyze(_rows()[:-1], [558, 559, 560])
    duplicated = _rows()
    duplicated[-1] = dict(duplicated[0])
    with pytest.raises(ValueError, match="duplicate seed-arm"):
        analyze(duplicated, [558, 559, 560])


def test_source_competence_mechanism_audit_uses_episode_variation():
    rows = _rows()
    for row in rows:
        if row["factorial_arm"] in (
            "ordinary_adaptive", "full_proposed"
        ):
            row.update({
                "reliability_actor_competence_mean": 0.46,
                "reliability_actor_competence_min": 0.40,
                "reliability_actor_competence_max": 0.50,
                "reliability_guided_fraction_raw_applied_mean": 0.30,
            })
    result = analyze(
        rows,
        [558, 559, 560],
        mechanism_mode="source_competence",
    )
    assert result["gate_passed"] is True
    assert result["mechanism_mode"] == "source_competence"


def test_source_competence_mechanism_rejects_constant_confidence():
    rows = _rows()
    for row in rows:
        if row["factorial_arm"] in (
            "ordinary_adaptive", "full_proposed"
        ):
            row.update({
                "reliability_actor_competence_mean": 0.5,
                "reliability_actor_competence_min": 0.5,
                "reliability_actor_competence_max": 0.5,
                "reliability_guided_fraction_raw_applied_mean": 0.6,
            })
    result = analyze(
        rows,
        [558, 559, 560],
        mechanism_mode="source_competence",
    )
    assert result["gate_passed"] is False
    assert result["checks"]["adaptive_mechanism_exercised"] is False
