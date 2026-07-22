import numpy as np

from experiments.rl.run_l280_anchor_gradient_alignment_diagnosis import (
    _comparison, _decision,
)


def test_l280_comparison_reports_cosine_and_norm_ratio():
    metrics = _comparison(np.asarray([1.0, 0.0]), np.asarray([-2.0, 0.0]))
    assert metrics["cosine"] == -1.0
    assert metrics["second_over_first_norm"] == 2.0


def test_l280_decision_requires_scene_and_seed_coverage():
    rows = []
    for seed in (1, 2, 3):
        for scene in range(6):
            row = {"seed": seed, "stage": "step6000"}
            for component in ("trunk", "v_mean", "omega_mean"):
                row["sac_vs_combined_%s_cosine" % component] = (
                    -0.5 if component == "omega_mean" and scene < 4 else 0.5
                )
                row["sac_vs_combined_%s_second_over_first_norm" % component] = 2.0
                row["recovery_vs_source_%s_cosine" % component] = 0.5
            rows.append(row)
    config = {"gate": {
        "minimum_conflicted_scenes": 4,
        "minimum_conflicted_seeds": 2,
        "maximum_conflict_cosine": 0.0,
        "minimum_conflict_norm_ratio": 1.0,
        "minimum_dominance_norm_ratio": 4.0,
    }}
    decision, metrics = _decision(rows, config)
    assert decision == "online_vs_anchor_conflict"
    assert metrics["online_conflicted_seed_counts"]["omega_mean"] == 3
