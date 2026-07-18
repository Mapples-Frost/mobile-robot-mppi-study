import pytest

from mobile_robot_mppi.evaluation.factorial import (
    blocked_factorial_contrasts,
    complete_factorial_blocks,
)


def _rows():
    rows = []
    # RL improves by 2, ICODE by 1, and their combination gains one extra
    # unit beyond additivity: 10 -> 9 / 8 -> 6.
    cells = {
        "traditional_mppi": 10.0,
        "icode_mppi": 9.0,
        "rl_driven_mppi": 8.0,
        "simple_combination": 6.0,
    }
    for block, shift in (("a", 0.0), ("b", 1.0)):
        for method, value in cells.items():
            rows.append({
                "block": block,
                "method": method,
                "loss": value + shift,
            })
    return rows


def test_blocked_factorial_contrasts_recover_main_and_interaction_effects():
    result = blocked_factorial_contrasts(
        _rows(), "loss", bootstrap_samples=100, seed=7
    )

    assert result["blocks"] == 2
    assert result["effects"]["icode_main_effect"]["estimate"] == -1.5
    assert result["effects"]["rl_main_effect"]["estimate"] == -2.5
    assert (
        result["effects"]["icode_by_rl_interaction"]["estimate"]
        == -1.0
    )
    assert (
        result["effects"]["combination_vs_traditional"]["estimate"]
        == -4.0
    )
    assert result["effects"]["icode_by_rl_interaction"]["favorable"]
    assert result["effects"]["icode_by_rl_interaction"]["ci95"] == [
        -1.0, -1.0
    ]


def test_factorial_analysis_rejects_incomplete_or_duplicated_blocks():
    incomplete = _rows()[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        complete_factorial_blocks(incomplete, "loss")

    duplicated = _rows() + [_rows()[0]]
    with pytest.raises(ValueError, match="duplicated"):
        complete_factorial_blocks(duplicated, "loss")
