from experiments.rl.analyze_cross_layer_budget_headroom import (
    CONDITIONS,
    OBSERVABLE_DIAGNOSTICS,
    OUTCOMES,
    _spearman_rho,
    analyze,
)


def _row(seed, condition, rmse, elapsed, compute):
    row = {
        "scene": "route",
        "physics_domain": "plant",
        "seed": str(seed),
        "condition": condition,
        "success": "True",
        "collision": "False",
    }
    for metric in OUTCOMES:
        row[metric] = str({
            "cross_track_rmse": rmse,
            "elapsed_s": elapsed,
            "planner_compute_ms_mean": compute,
            "control_jerk": 0.1,
            "applied_control_jerk": 0.08,
        }[metric])
    for feature in OBSERVABLE_DIAGNOSTICS:
        row[feature] = "1.0"
    return row


def test_hindsight_oracle_uses_k100_only_when_k50_misses_margin():
    rows = []
    # Seed 1: contextual K50 is within +2 mm and faster.
    rows.extend((
        _row(1, "icode_fixed_k50", 0.040, 29.0, 24.0),
        _row(1, "icode_contextual_k50", 0.036, 28.0, 24.0),
        _row(1, "icode_fixed_k100", 0.035, 30.0, 31.0),
    ))
    # Seed 2: both K50 arms exceed the margin, so K100 is necessary.
    rows.extend((
        _row(2, "icode_fixed_k50", 0.050, 28.0, 24.0),
        _row(2, "icode_contextual_k50", 0.049, 27.0, 24.0),
        _row(2, "icode_fixed_k100", 0.035, 30.0, 31.0),
    ))
    summary, decisions = analyze(rows, rmse_margin_m=0.002, bootstrap_seed=3)
    assert summary["icode_blocks"] == 2
    assert summary["needs_k100_fraction"] == 0.5
    assert [row["selected_condition"] for row in decisions] == [
        "icode_contextual_k50", "icode_fixed_k100"
    ]
    assert summary["oracle_vs_fixed_k100"][
        "planner_compute_ms_mean_delta_mean"
    ] == -3.5


def test_incomplete_icode_block_is_rejected():
    rows = [
        _row(1, condition, 0.04, 30.0, 25.0)
        for condition in CONDITIONS[:-1]
    ]
    try:
        analyze(rows)
    except ValueError as error:
        assert "incomplete ICODE blocks" in str(error)
    else:
        raise AssertionError("incomplete block should fail")


def test_spearman_rho_handles_ties_and_constant_inputs():
    assert _spearman_rho([1.0, 2.0, 2.0, 4.0], [1.0, 2.0, 2.0, 4.0]) == 1.0
    assert _spearman_rho([4.0, 2.0, 2.0, 1.0], [1.0, 2.0, 2.0, 4.0]) == -1.0
    assert _spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
