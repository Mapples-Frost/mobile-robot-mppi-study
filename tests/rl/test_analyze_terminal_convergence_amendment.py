import pytest

from experiments.rl.analyze_terminal_convergence_amendment import (
    confirmation_decision,
    diagnostic_decision,
)


def rows(successes, cross_track=0.10, jerk=0.20):
    result = []
    for index, arm in enumerate((
        "ordinary_fixed",
        "value_fixed",
        "ordinary_adaptive",
        "full_proposed",
    )):
        result.append({
            "factorial_arm": arm,
            "physics_domain": "nominal",
            "seed": 553,
            "success": index < successes,
            "collision": False,
            "cross_track_rmse": cross_track,
            "control_jerk": jerk,
        })
    return result


def test_diagnostic_selects_improved_safe_candidate():
    result = diagnostic_decision(
        rows(0), rows(2, cross_track=0.105, jerk=0.21)
    )

    assert result["candidate_selected"]
    assert result["t1"]["successes"] == 2


def test_diagnostic_rejects_excessive_jerk_regression():
    result = diagnostic_decision(rows(0), rows(2, jerk=0.23))

    assert not result["candidate_selected"]
    assert not result["criteria"]["jerk_regression_within_10pct"]


def test_confirmation_requires_full_success_and_bounded_regressions():
    first = rows(4)
    second = [dict(item, seed=554) for item in rows(4)]
    result = confirmation_decision(first + second)
    assert result["integration_gate_passed"]

    failed = first + [
        dict(item, seed=554, success=(
            False if item["factorial_arm"] == "full_proposed"
            else item["success"]
        ))
        for item in rows(4)
    ]
    assert not confirmation_decision(failed)["integration_gate_passed"]


def test_diagnostic_rejects_mismatched_cells():
    with pytest.raises(ValueError, match="cells differ"):
        diagnostic_decision(rows(0), rows(2)[:-1])
