import pytest

from experiments.rl.run_covariance_context_oracle import (
    evaluate_mapping,
    select_mapping,
)


def _row(scene, domain, candidate, seed, rmse, time_value):
    return {
        "scene": scene,
        "physics_domain": domain,
        "candidate": candidate,
        "seed": seed,
        "success": True,
        "collision": False,
        "cross_track_rmse": rmse,
        "time_to_goal_s": time_value,
        "control_jerk": 0.1,
    }


def test_selection_uses_precision_set_then_time_and_separates_contexts():
    rows = []
    for seed in range(3):
        rows.extend((
            _row("a", "p", "speed", seed, 0.040, 10.0),
            _row("a", "p", "turn", seed, 0.041, 9.0),
            _row("b", "p", "speed", seed, 0.040, 10.0),
            _row("b", "p", "turn", seed, 0.050, 8.0),
        ))
    mapping, global_candidate, _, _ = select_mapping(rows, 0.002)
    assert mapping[("a", "p")] == "turn"
    assert mapping[("b", "p")] == "speed"
    assert global_candidate == "speed"


def test_evaluation_requires_heterogeneity_precision_time_and_safety():
    rows = []
    mapping = {}
    for index in range(4):
        context = ("scene_%d" % index, "plant")
        mapping[context] = "turn" if index else "speed"
        for seed in range(3):
            rows.append(_row(*context, "speed", seed, 0.040, 10.0))
            rows.append(_row(
                *context,
                mapping[context],
                seed,
                0.040,
                9.5 if mapping[context] != "speed" else 10.0,
            ))
    # The context mapped to speed creates duplicate dictionary keys by design;
    # remove exact duplicate rows as the real factorial contains one such run.
    unique = {}
    for row in rows:
        unique[(row["scene"], row["physics_domain"], row["seed"], row["candidate"])] = row
    result = evaluate_mapping(
        list(unique.values()), mapping, "speed", 0.002, 17
    )
    assert result["precision_gate_passed"]
    assert result["time_gate_passed"]
    assert result["safety_gate_passed"]
    assert result["heterogeneity_gate_passed"]
    assert result["primary_gate_passed"]
    assert result["time_to_goal_s_delta_mean"] == pytest.approx(-0.375)
