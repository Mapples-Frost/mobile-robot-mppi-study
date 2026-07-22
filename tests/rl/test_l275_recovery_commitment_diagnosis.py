from experiments.rl.run_l275_recovery_commitment_continuation_diagnosis import _gate


GATE = {
    "minimum_full_chain_positive_fraction": 0.75,
    "minimum_positive_fraction_gain_full_vs_one": 0.20,
    "require_median_full_greater_than_one": True,
    "minimum_late_first_positive_fraction": 0.25,
    "late_commitment_minimum_steps": 5,
    "minimum_scenes_with_full_chain_majority_positive": 4,
}


def _rows(passing=True):
    rows = []
    commitments = ("1", "5", "10", "20", "40", "full_chain")
    for state in range(12):
        for index, commitment in enumerate(commitments):
            advantage = float(index - 1) if passing else -1.0
            rows.append({
                "state_id": "state_%02d" % state,
                "scene": "scene_%d" % (state % 6),
                "commitment": commitment,
                "recovery_advantage": advantage,
            })
    return rows


def test_l275_gate_recognizes_commitment_signature():
    checks, metrics = _gate(_rows(), True, GATE)
    assert all(checks.values())
    assert metrics["positive_fractions"]["full_chain"] == 1.0
    assert metrics["late_first_positive_fraction"] == 1.0


def test_l275_gate_fails_without_recovery_advantage():
    checks, _ = _gate(_rows(passing=False), True, GATE)
    assert not checks["full_chain_positive_fraction"]
    assert not checks["positive_fraction_gain_full_vs_one"]


def test_l275_gate_fails_closed_on_integrity():
    checks, _ = _gate(_rows(), False, GATE)
    assert not checks["integrity"]

