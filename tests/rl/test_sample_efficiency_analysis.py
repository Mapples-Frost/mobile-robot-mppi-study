from experiments.rl.analyze_sample_efficiency_results import (
    _holm_adjust,
    _mcnemar_exact,
)


def test_mcnemar_exact_handles_one_sided_discordance_and_tie():
    assert _mcnemar_exact(0, 0) == 1.0
    assert abs(_mcnemar_exact(5, 0) - 0.0625) < 1e-12
    assert _mcnemar_exact(3, 3) == 1.0


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    raw = [0.04, 0.001, 0.02]
    adjusted = _holm_adjust(raw)
    assert adjusted[1] == 0.003
    assert adjusted[2] == 0.04
    assert adjusted[0] == 0.04
