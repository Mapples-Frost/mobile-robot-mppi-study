from experiments.rl.plot_contextual_covariance_sample_efficiency import (
    _paired_primary,
)


def test_plot_pair_extraction_matches_half_budget_arms():
    rows = []
    for samples, condition in (
        (50, "learned_contextual_bandit"),
        (100, "strongest_global_fixed"),
    ):
        rows.append({
            "scene": "route",
            "physics_domain": "domain",
            "seed": 1,
            "num_samples": samples,
            "condition": condition,
        })
    pairs = _paired_primary(rows)
    assert len(pairs) == 1
    assert int(pairs[0][0]["num_samples"]) == 50
    assert int(pairs[0][1]["num_samples"]) == 100
