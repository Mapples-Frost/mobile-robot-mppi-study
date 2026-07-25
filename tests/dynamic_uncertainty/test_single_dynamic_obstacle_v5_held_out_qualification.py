from copy import deepcopy
from pathlib import Path

from experiments.dynamic_uncertainty import (
    run_single_dynamic_obstacle_v5_held_out_qualification as qualification,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "configs"
    / "research"
    / "single_dynamic_obstacle_v5_held_out_qualification.yaml"
)


def test_registry_is_deterministic_disjoint_complete_and_balanced():
    checked = qualification.validate(PROTOCOL)
    protocol = checked["protocol"]
    registry = checked["registry"]
    assert registry["status"] == "sealed_before_held_out_qualification_execution"
    assert registry["outcome_selection_used"] is False
    assert registry["controller_outcomes_opened"] is False
    assert len(registry["splits"]["id"]) == 16
    assert len(registry["splits"]["ood"]) == 16
    assert len(registry["schedule"]) == 32
    assert registry["schedule_sha256"] == qualification.development._canonical_sha256(
        registry["schedule"]
    )
    seeds = {
        int(item["seed"])
        for split in ("id", "ood")
        for item in registry["splits"][split]
    }
    assert len(seeds) == 32
    for seed in seeds:
        assert not qualification._seed_in_ranges(
            seed, protocol["seed_selection"]["forbidden_seed_ranges"]
        )
        assert not qualification._seed_in_ranges(
            seed, protocol["seed_selection"]["reserved_future_formal_ranges"]
        )
    for split in ("id", "ood"):
        orders = [
            tuple(block["arm_order"])
            for block in registry["schedule"]
            if block["split"] == split
        ]
        assert orders.count(("V4_full_frozen", "V5_dual_horizon_full")) == 8
        assert orders.count(("V5_dual_horizon_full", "V4_full_frozen")) == 8


def test_held_out_arm_configuration_changes_only_frozen_candidate_fields():
    checked = qualification.validate(PROTOCOL)
    protocol = checked["protocol"]
    source = checked["source_protocol"]
    block = checked["registry"]["schedule"][0]
    control, control_changes = qualification._qualification_config(
        protocol, source, block, "V4_full_frozen"
    )
    candidate, candidate_changes = qualification._qualification_config(
        protocol, source, block, "V5_dual_horizon_full"
    )
    assert control_changes == {}
    expected = {
        "perception.scan_guard.%s" % key
        for key, value in protocol["candidate_overrides"].items()
        if control["perception"]["scan_guard"].get(key, "<MISSING>") != value
    }
    assert set(candidate_changes) == expected
    for config in (control, candidate):
        assert int(config["planner"]["num_samples"]) * int(
            config["planner"]["paper_rl_driven"]["iterations"]
        ) == 600
        assert config["scope_guards"]["paper_v5_held_out_qualification"] is True
        assert config["scope_guards"]["formal_effect_estimation"] is False


def _synthetic_rows_and_pairs():
    rows = []
    pairs = []
    for index in range(4):
        control_success = index >= 2
        candidate_success = True
        for arm, success in (
            ("V4_full_frozen", control_success),
            ("V5_dual_horizon_full", candidate_success),
        ):
            rows.append({
                "arm": arm,
                "success": success,
                "collision": False,
                "zero_speed_risk_steps": 10,
                "stuck_steps": 10,
                "direction_switch_count": 1,
                "three_phase_oscillation_count": 0,
                "paper_total_rollouts_mean": 600.0,
            })
        pairs.append({
            "split": "id" if index % 2 == 0 else "ood",
            "seed": 750100001 + index,
            "v4_success": control_success,
            "v5_success": candidate_success,
            "v4_collision": False,
            "v5_collision": False,
            "success_delta": int(candidate_success) - int(control_success),
            "collision_delta": 0,
            "minimum_clearance_delta": 0.0,
            "conflict_q05_clearance_delta": 0.0,
            "final_goal_distance_delta": -0.1,
            "zero_speed_risk_steps_delta": 0,
            "stuck_steps_delta": 0,
        })
    return rows, pairs


def test_qualification_gate_passes_positive_safe_generalization(monkeypatch):
    protocol = qualification.v4._load_yaml(PROTOCOL)
    rows, pairs = _synthetic_rows_and_pairs()
    monkeypatch.setattr(
        qualification, "_pairs_and_rows", lambda *_args: (rows, pairs)
    )
    result = qualification._analyze(protocol, {}, Path("unused"), [])
    assert result["status"] == "qualification_pass"
    assert result["paired_safe_success_net_gain"] == 2
    assert all(result["checks"].values())


def test_qualification_gate_rejects_new_paired_collision(monkeypatch):
    protocol = qualification.v4._load_yaml(PROTOCOL)
    rows, pairs = _synthetic_rows_and_pairs()
    failed_rows = deepcopy(rows)
    failed_pairs = deepcopy(pairs)
    failed_rows[-1]["collision"] = True
    failed_pairs[-1]["v5_collision"] = True
    failed_pairs[-1]["collision_delta"] = 1
    monkeypatch.setattr(
        qualification,
        "_pairs_and_rows",
        lambda *_args: (failed_rows, failed_pairs),
    )
    result = qualification._analyze(protocol, {}, Path("unused"), [])
    assert result["status"] == "qualification_fail"
    assert result["checks"]["no_new_paired_collisions"] is False
    assert result["candidate_authorized_for_formal_protocol_freeze"] is False
