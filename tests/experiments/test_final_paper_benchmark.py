import copy
import hashlib
import json
from pathlib import Path

import yaml

from experiments.rl.run_final_paper_benchmark import (
    ARMS,
    _core_factorial_rows,
    build_arm_config,
    final_schedule,
    load_benchmark_manifest,
    _load_reliability_with_gate,
    _comparison,
    resolve_benchmark_seeds,
)
from experiments.rl.analyze_final_paper_benchmark import (
    aggregate,
    validate_completed_rows,
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


def test_shared_planner_and_sensor_overrides_apply_to_every_arm():
    for arm in ARMS:
        config, _, _, _ = build_arm_config(
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
            0.0,
            0.0,
            planner_overrides={"terminal_control_radius": 0.8},
            sensor_overrides={
                "pose_source": "ground_truth",
                "twist_source": "ground_truth",
            },
        )
        assert config["planner"]["terminal_control_radius"] == 0.8
        assert config["sensors"]["pose_source"] == "ground_truth"
        assert config["sensors"]["twist_source"] == "ground_truth"


def test_reliability_override_is_confined_to_adaptive_arms():
    override = {
        "dynamics_routing_mode": "policy_rescue",
        "policy_rescue_floor": 0.5,
    }
    configs = {}
    for arm in ARMS:
        configs[arm], _, _, _ = build_arm_config(
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
            0.0,
            0.0,
            reliability_overrides=override,
        )
    for arm in ("ordinary_adaptive", "full_proposed"):
        reliability = configs[arm]["planner"]["paper_rl_driven"][
            "reliability"
        ]
        assert reliability["dynamics_routing_mode"] == "policy_rescue"
    for arm in ("rl_driven_mppi", "simple_combination", "value_fixed"):
        reliability = configs[arm]["planner"]["paper_rl_driven"][
            "reliability"
        ]
        assert "dynamics_routing_mode" not in reliability


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


def test_frozen_manifest_is_a_manifest_not_an_experiment_config():
    root = Path(__file__).resolve().parents[2]
    path = root / (
        "configs/research/final_paper_benchmark_point_goal_l214.yaml"
    )
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    assert manifest["final_benchmark"]["status"] == "preregistered"
    assert manifest["final_benchmark"]["formal_seeds"] == list(
        range(101, 111)
    )
    for key in (
        "ordinary_calibration_config",
        "value_calibration_config",
    ):
        calibration_path = root / manifest["final_benchmark"][key]
        with calibration_path.open("r", encoding="utf-8") as handle:
            calibration = yaml.safe_load(handle)
        assert "ensemble" in calibration


def test_development_manifest_inheritance_is_recursive():
    root = Path(__file__).resolve().parents[2]
    completion = load_benchmark_manifest(
        root / "configs/research/complex_navigation_development_l216_completion_only.yaml"
    )["final_benchmark"]
    rescue = load_benchmark_manifest(
        root / "configs/research/complex_navigation_development_l216_policy_rescue.yaml"
    )["final_benchmark"]

    assert completion["scene_configs"]
    assert completion["completion_handover_full_fallback_distance"] == 0.0
    assert completion["sensor_overrides"]["pose_source"] == "ground_truth"
    assert rescue["reliability_overrides"] == {
        "dynamics_routing_mode": "policy_rescue",
        "policy_rescue_floor": 0.5,
    }


def test_formal_seed_shards_are_derived_from_preregistered_seeds():
    frozen = {"sealed_seeds": [11, 12, 13, 14, 15]}
    selected, sealed = resolve_benchmark_seeds(
        "", frozen, False, shard_index=1, shard_count=2
    )
    assert sealed == (11, 12, 13, 14, 15)
    assert selected == (12, 14)

    legacy, _ = resolve_benchmark_seeds(
        "", {"formal_seeds": [21, 22]}, False
    )
    assert legacy == (21, 22)

    selected, _ = resolve_benchmark_seeds(
        "12,14", frozen, False, shard_index=1, shard_count=2
    )
    assert selected == (12, 14)


def test_formal_seed_override_and_invalid_shards_are_rejected():
    frozen = {"sealed_seeds": [11, 12, 13, 14]}
    for cli, index, count in (
        ("11,12", 0, 1),
        ("", 2, 2),
        ("", 0, 0),
    ):
        try:
            resolve_benchmark_seeds(
                cli, frozen, False, shard_index=index, shard_count=count
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid formal seed selection accepted")


def test_qualification_requires_explicit_unsharded_seeds():
    selected, sealed = resolve_benchmark_seeds(
        "48,49", {}, True
    )
    assert selected == sealed == (48, 49)
    for cli, index, count in (("", 0, 1), ("48", 1, 2)):
        try:
            resolve_benchmark_seeds(
                cli, {}, True, shard_index=index, shard_count=count
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid qualification seeds accepted")


def test_l217_manifest_freezes_unseen_seeds_and_policy_rescue():
    root = Path(__file__).resolve().parents[2]
    frozen = load_benchmark_manifest(
        root / "configs/research/complex_navigation_sealed_l217.yaml"
    )["final_benchmark"]
    assert frozen["status"] == "preregistered"
    assert frozen["sealed_seeds"] == list(range(78006, 78016))
    assert frozen["bootstrap_samples"] == 10000
    assert frozen["completion_handover_full_fallback_distance"] == 0.0
    assert frozen["completion_handover_full_rl_distance"] == 0.0
    assert frozen["reliability_overrides"] == {
        "dynamics_routing_mode": "policy_rescue",
        "policy_rescue_floor": 0.5,
    }


def test_external_gate_evidence_must_bind_exact_calibration(
    tmp_path, monkeypatch
):
    summary = tmp_path / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()
    evidence.write_text(
        json.dumps({
            "gate_passed": True,
            "calibration_summary_sha256": digest,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "experiments.rl.run_final_paper_benchmark._load_reliability",
        lambda *args, **kwargs: {"runtime": {}, "ensemble": {}},
    )
    loaded = _load_reliability_with_gate(
        summary, tmp_path / "runtime.yaml", evidence
    )
    assert loaded["gate_evidence_sha256"]

    evidence.write_text(
        json.dumps({
            "gate_passed": True,
            "calibration_summary_sha256": "wrong",
        }),
        encoding="utf-8",
    )
    try:
        _load_reliability_with_gate(
            summary, tmp_path / "runtime.yaml", evidence
        )
    except ValueError as exc:
        assert "does not bind calibration" in str(exc)
    else:
        raise AssertionError("mismatched evidence must be rejected")


def _formal_row(seed, arm, domain="nominal_seen"):
    return {
        "scene": "clean",
        "physics_domain": domain,
        "seed": seed,
        "benchmark_arm": arm,
        "block": "clean::%s::seed%d" % (domain, seed),
        "qualification": 0,
        "final_goal_distance": 1.0,
    }


def test_completed_row_validation_rejects_qualification_and_missing_arm():
    rows = [_formal_row(101, arm) for arm in ARMS]
    audit = validate_completed_rows(rows, (101,))
    assert audit["episodes"] == 7
    assert audit["blocks"] == 1
    rows[0]["qualification"] = 1
    try:
        validate_completed_rows(rows, (101,))
    except ValueError as exc:
        assert "qualification" in str(exc)
    else:
        raise AssertionError("qualification data must be rejected")

    rows = [_formal_row(101, arm) for arm in ARMS[:-1]]
    try:
        validate_completed_rows(rows, (101,))
    except ValueError as exc:
        assert "incomplete formal blocks" in str(exc)
    else:
        raise AssertionError("an incomplete block must be rejected")


def test_final_comparison_uses_benchmark_arm_rows():
    rows = []
    for seed in (101, 102):
        before = _formal_row(seed, "simple_combination")
        after = _formal_row(seed, "full_proposed")
        before["final_goal_distance"] = 1.0
        after["final_goal_distance"] = 0.8
        rows.extend((before, after))
    result = _comparison(
        rows,
        "simple_combination",
        "full_proposed",
        "full_vs_simple",
        100,
        7,
        {"final_goal_distance": False},
    )
    assert abs(
        result["metrics"]["final_goal_distance"]["favorable_effect"]
        - 0.2
    ) < 1e-12


def test_descriptive_aggregate_accepts_csv_boolean_strings():
    rows = [_formal_row(101, "traditional_mppi")]
    rows[0].update({"success": "True", "collision": "False"})
    summary = aggregate(rows, ("success", "collision"))
    assert summary[0]["success_mean"] == 1.0
    assert summary[0]["collision_mean"] == 0.0
