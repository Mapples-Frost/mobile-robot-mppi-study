import copy

from experiments.rl.run_final_paper_benchmark import (
    ARMS,
    _core_factorial_rows,
    build_arm_config,
    final_schedule,
)


def _base():
    return {
        "experiment": {"seed": 1},
        "scene": {"name": "clean"},
        "planner": {},
        "memory": {"enable": True},
        "plant": {},
    }


def _calibration():
    return {
        "ensemble": {
            "disagreement_scales": [1.0, 1.0, 1.0],
            "innovation_scales": [1.0, 1.0, 1.0],
            "innovation_decay": 0.9,
            "support_soft_z": 2.0,
            "support_hard_z": 4.0,
        },
        "runtime": {"enabled": True, "low_guided_fraction": 0.0},
        "summary_path": "summary.json",
    }


def _build(arm):
    return build_arm_config(
        copy.deepcopy(_base()),
        {"arm": arm, "seed": 7},
        "actor.pt",
        ["o1.pt", "o2.pt"],
        ["v1.pt", "v2.pt"],
        _calibration(),
        _calibration(),
        100,
        2,
        {"name": "nominal", "role": "seen", "plant_override": {}},
        300,
        1.26,
        0.30,
    )


def test_schedule_is_complete_and_randomized_within_every_block():
    jobs = final_schedule(
        (101, 102),
        ({"name": "d1"}, {"name": "d2"}),
        ({"name": "s1"},),
        20260719,
    )
    assert len(jobs) == 2 * 2 * len(ARMS)
    blocks = {}
    for job in jobs:
        blocks.setdefault(job["block"], []).append(job)
    assert len(blocks) == 4
    for block in blocks.values():
        assert {job["arm"] for job in block} == set(ARMS)
        assert sorted(job["run_order_within_block"] for job in block) == list(
            range(len(ARMS))
        )


def test_contextual_baselines_keep_their_intended_semantics():
    traditional, flags, _, _ = _build("traditional_mppi")
    assert flags["use_icode"] is False
    assert flags["use_rl"] is False
    assert traditional["planner"]["optimizer"] == "standard"
    assert traditional["planner"]["prediction_mode"] == "nominal"

    icode, flags, _, _ = _build("icode_mppi")
    assert flags["use_icode"] is True
    assert flags["use_rl"] is False
    assert icode["planner"]["optimizer"] == "standard"
    assert icode["planner"]["prediction_mode"] == "icode_residual"
    assert icode["planner"]["checkpoints"] == ["o1.pt", "o2.pt"]

    rl, flags, _, _ = _build("rl_driven_mppi")
    assert flags["use_icode"] is False
    assert flags["use_rl"] is True
    assert rl["planner"]["prediction_mode"] == "nominal"
    assert rl["planner"]["optimizer"] == "paper_rl_driven"


def test_confirmatory_arms_change_only_frozen_factors():
    simple, simple_flags, _, _ = _build("simple_combination")
    value, value_flags, _, _ = _build("value_fixed")
    adaptive, adaptive_flags, _, _ = _build("ordinary_adaptive")
    full, full_flags, _, _ = _build("full_proposed")

    assert simple_flags["value_aligned"] is False
    assert simple_flags["adaptive_hss"] is False
    assert value_flags["value_aligned"] is True
    assert value_flags["adaptive_hss"] is False
    assert adaptive_flags["value_aligned"] is False
    assert adaptive_flags["adaptive_hss"] is True
    assert full_flags["value_aligned"] is True
    assert full_flags["adaptive_hss"] is True
    assert simple["planner"]["paper_rl_driven"]["reliability"][
        "enabled"
    ] is False
    assert adaptive["planner"]["paper_rl_driven"]["reliability"][
        "enabled"
    ] is True
    assert full["planner"]["paper_rl_driven"][
        "terminal_guidance_radius"
    ] == 1.26
    assert value["planner"]["checkpoints"] == ["v1.pt", "v2.pt"]


def test_core_factorial_rows_use_only_the_four_confirmatory_arms():
    rows = [
        {"benchmark_arm": arm, "method": arm, "block": "b"}
        for arm in ARMS
    ]
    core = _core_factorial_rows(rows)
    assert len(core) == 4
    assert {row["method"] for row in core} == {
        "traditional_mppi",
        "icode_mppi",
        "rl_driven_mppi",
        "simple_combination",
    }
