from experiments.rl.analyze_path_reliability_components import (
    _rankdata,
    _spearman,
)


def test_rankdata_uses_average_ranks_for_ties():
    ranks = _rankdata([2.0, 1.0, 2.0, 4.0])
    assert ranks.tolist() == [1.5, 0.0, 1.5, 3.0]


def test_spearman_reports_monotone_inverse_component():
    assert _spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert _spearman([1.0, 1.0, 1.0], [3.0, 2.0, 1.0]) is None
