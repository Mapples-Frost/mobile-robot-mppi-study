import numpy as np

from experiments.rl.run_l269_horizon_diagnostic import (
    _discounted_prefix,
    _horizon_gate,
)


GATE = {
    "minimum_h40_recovery_advantage_fraction": 0.60,
    "minimum_positive_fraction_gain_h1_to_h40": 0.15,
    "require_median_h40_greater_than_h1": True,
    "minimum_late_first_positive_fraction": 0.20,
    "late_first_positive_minimum_horizon": 10,
    "minimum_full_chain_recovery_advantage_fraction": 0.60,
    "minimum_scenes_with_h40_majority_positive": 4,
}


def test_l269_discounted_prefix_is_exact():
    rewards = np.asarray((1.0, 2.0, 3.0))
    assert np.isclose(_discounted_prefix(rewards, 0.5, 2), 2.0)
    assert np.isclose(_discounted_prefix(rewards, 0.5, 10), 2.75)


def _passing_rows():
    rows = []
    for state in range(10):
        scene = "scene_%d" % (state % 5)
        for horizon in (1, 5, 10, 20, 40):
            advantage = -1.0 if horizon < 10 else 1.0 + 0.1 * state
            rows.append({
                "state_id": "state_%d" % state,
                "scene": scene,
                "horizon": horizon,
                "recovery_advantage": advantage,
            })
    return rows


def test_l269_horizon_gate_recognizes_late_credit_signature():
    full = [{"recovery_advantage": 1.0} for _ in range(10)]
    checks, metrics = _horizon_gate(
        _passing_rows(), full, (1, 5, 10, 20, 40), GATE
    )
    assert all(checks.values())
    assert metrics["late_first_positive_fraction"] == 1.0


def test_l269_horizon_gate_fails_when_recovery_never_wins():
    rows = _passing_rows()
    for row in rows:
        row["recovery_advantage"] = -1.0
    full = [{"recovery_advantage": -1.0} for _ in range(10)]
    checks, _ = _horizon_gate(rows, full, (1, 5, 10, 20, 40), GATE)
    assert not checks["h40_recovery_advantage_fraction"]
    assert not checks["full_chain_recovery_advantage_fraction"]
