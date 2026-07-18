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
    assert result["independent_clusters"] == 2
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


def test_factorial_bootstrap_clusters_repeated_strata_by_seed():
    rows = []
    for seed, values in (
        (11, (10.0, 9.0, 8.0, 6.0)),
        (12, (10.0, 9.0, 8.0, 8.0)),
    ):
        for scene in ("a", "b"):
            block = "%s_seed%d" % (scene, seed)
            for method, value in zip(
                (
                    "traditional_mppi",
                    "icode_mppi",
                    "rl_driven_mppi",
                    "simple_combination",
                ),
                values,
            ):
                rows.append({
                    "block": block,
                    "seed": seed,
                    "method": method,
                    "loss": value,
                })
    result = blocked_factorial_contrasts(
        rows,
        "loss",
        bootstrap_samples=100,
        seed=7,
        cluster_key="seed",
    )

    interaction = result["effects"]["icode_by_rl_interaction"]
    assert result["blocks"] == 4
    assert result["independent_clusters"] == 2
    assert interaction["per_cluster"] == [-1.0, 1.0]
    assert interaction["estimate"] == 0.0
    assert interaction["ci95"] == [-1.0, 1.0]


def test_factorial_analysis_rejects_incomplete_or_duplicated_blocks():
    incomplete = _rows()[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        complete_factorial_blocks(incomplete, "loss")

    duplicated = _rows() + [_rows()[0]]
    with pytest.raises(ValueError, match="duplicated"):
        complete_factorial_blocks(duplicated, "loss")


def test_factorial_analysis_accepts_csv_boolean_values():
    rows = []
    for method, value in (
        ("traditional_mppi", "False"),
        ("icode_mppi", "False"),
        ("rl_driven_mppi", "True"),
        ("simple_combination", "True"),
    ):
        rows.append({
            "block": "a",
            "method": method,
            "success": value,
        })
    _, values = complete_factorial_blocks(rows, "success")
    assert values.tolist() == [[0.0, 0.0, 1.0, 1.0]]
