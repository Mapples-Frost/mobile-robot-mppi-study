import pytest

from experiments.rl.run_covariance_policy_evaluation_matrix import (
    _paired_primary,
)


def test_paired_primary_requires_every_block_to_improve_without_safety_loss():
    rows = []
    for block, learned, fixed in (
        ("block_0", 0.04, 0.05),
        ("block_1", 0.06, 0.07),
    ):
        for condition, cross_track in (
            ("learned_covariance", learned),
            ("fixed_broad", fixed),
        ):
            rows.append({
                "block": block,
                "scene": "curve",
                "seed": 11,
                "condition": condition,
                "cross_track_rmse": cross_track,
                "success": True,
                "collision": False,
            })

    result = _paired_primary(rows)
    assert result["cross_track_rmse_delta_mean"] == pytest.approx(-0.01)
    assert result["blocks_improved"] == 2
    assert result["primary_gate_passed"] is True


def test_paired_primary_fails_when_one_block_regresses():
    rows = [
        {
            "block": "block_0",
            "scene": "curve",
            "seed": 11,
            "condition": condition,
            "cross_track_rmse": value,
            "success": True,
            "collision": False,
        }
        for condition, value in (
            ("learned_covariance", 0.06),
            ("fixed_broad", 0.05),
        )
    ]
    assert _paired_primary(rows)["primary_gate_passed"] is False


def test_preregistered_gate_requires_precision_and_time_separately():
    rows = []
    for seed in range(5):
        for condition, cross_track, time_value in (
            ("learned_covariance", 0.041, 10.0),
            ("fixed_speed", 0.040, 10.0),
        ):
            rows.append({
                "block": "block_0",
                "scene": "curve",
                "seed": seed,
                "condition": condition,
                "cross_track_rmse": cross_track,
                "time_to_goal_s": time_value,
                "success": True,
                "collision": False,
            })
    result = _paired_primary(
        rows,
        comparator="fixed_speed",
        cross_track_noninferiority_margin=0.002,
        require_time_superiority=True,
    )
    assert result["precision_gate_passed"] is True
    assert result["time_gate_passed"] is False
    assert result["safety_gate_passed"] is True
    assert result["primary_gate_passed"] is False

    for row in rows:
        if row["condition"] == "learned_covariance":
            row["time_to_goal_s"] = 9.5
    improved = _paired_primary(
        rows,
        comparator="fixed_speed",
        cross_track_noninferiority_margin=0.002,
        require_time_superiority=True,
    )
    assert improved["time_gate_passed"] is True
    assert improved["primary_gate_passed"] is True
