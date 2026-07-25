from experiments.dynamic_uncertainty import (
    run_single_dynamic_obstacle_paper_v4 as v4,
)
from experiments.dynamic_uncertainty import (
    run_single_dynamic_obstacle_v5_recovery_development as development,
)


def _protocols():
    protocol = v4._load_yaml(development.DEFAULT_PROTOCOL)
    _, source, _, _, _ = v4.validate_protocol(
        development._repo_path(protocol["source_protocol"])
    )
    return protocol, source


def _deadline_protocols():
    path = (
        development.ROOT
        / "configs"
        / "research"
        / "single_dynamic_obstacle_v5_deadline_development_b1.yaml"
    )
    protocol = v4._load_yaml(path)
    _, source, _, _, _ = v4.validate_protocol(
        development._repo_path(protocol["source_protocol"])
    )
    return protocol, source


def _deadline_b2_protocols():
    path = (
        development.ROOT
        / "configs"
        / "research"
        / "single_dynamic_obstacle_v5_deadline_development_b2.yaml"
    )
    protocol = v4._load_yaml(path)
    _, source, _, _, _ = v4.validate_protocol(
        development._repo_path(protocol["source_protocol"])
    )
    return protocol, source


def _deadline_b3_protocols():
    path = (
        development.ROOT
        / "configs"
        / "research"
        / "single_dynamic_obstacle_v5_deadline_development_b3.yaml"
    )
    protocol = v4._load_yaml(path)
    _, source, _, _, _ = v4.validate_protocol(
        development._repo_path(protocol["source_protocol"])
    )
    return protocol, source


def _dual_horizon_c1_protocols():
    path = (
        development.ROOT
        / "configs"
        / "research"
        / "single_dynamic_obstacle_v5_dual_horizon_development_c1.yaml"
    )
    protocol = v4._load_yaml(path)
    _, source, _, _, _ = v4.validate_protocol(
        development._repo_path(protocol["source_protocol"])
    )
    return protocol, source


def _dual_horizon_c2_protocols():
    path = (
        development.ROOT
        / "configs"
        / "research"
        / "single_dynamic_obstacle_v5_dual_horizon_development_c2.yaml"
    )
    protocol = v4._load_yaml(path)
    _, source, _, _, _ = v4.validate_protocol(
        development._repo_path(protocol["source_protocol"])
    )
    return protocol, source


def _dual_horizon_c3_protocols():
    path = (
        development.ROOT
        / "configs"
        / "research"
        / "single_dynamic_obstacle_v5_dual_horizon_development_c3.yaml"
    )
    protocol = v4._load_yaml(path)
    _, source, _, _, _ = v4.validate_protocol(
        development._repo_path(protocol["source_protocol"])
    )
    return protocol, source


def test_v5_development_is_outcome_informed_and_not_effect_estimation():
    protocol, _ = _protocols()

    assert protocol["scope"] == (
        "targeted_mechanism_development_not_effect_estimation"
    )
    assert protocol["design"]["outcome_informed_development_selection"]
    assert not protocol["design"]["formal_effect_estimation_authorized"]
    assert protocol["design"]["future_v5_registry_seeds_must_be_disjoint"]
    assert protocol["design"][
        "future_held_out_qualification_seeds_must_be_disjoint"
    ]
    assert len(protocol["development_blocks"]) == 8
    assert len({
        int(block["seed"]) for block in protocol["development_blocks"]
    }) == 8
    assert development._sha256(
        development._repo_path(protocol["source_analysis"])
    ) == protocol["source_analysis_sha256"]
    assert development._sha256(
        development._repo_path(protocol["source_analysis_manifest"])
    ) == protocol["source_analysis_manifest_sha256"]
    development._validate_parent_attempt(protocol)


def test_v5_recovery_changes_only_declared_scan_guard_fields():
    protocol, source = _protocols()
    block = dict(protocol["development_blocks"][0])

    v4_config, v4_changes = development._configure_arm(
        protocol, source, block, "V4_full_frozen"
    )
    v5_config, v5_changes = development._configure_arm(
        protocol, source, block, "V5_recovery_full"
    )

    assert not v4_changes
    assert not v4_config["perception"]["scan_guard"].get(
        "dynamic_recovery_progressive_acceleration_enabled", False
    )
    assert v5_config["perception"]["scan_guard"][
        "dynamic_recovery_progressive_acceleration_enabled"
    ]
    assert set(v5_changes) <= {
        "perception.scan_guard.%s" % key
        for key in protocol["v5_recovery_overrides"]
    }
    assert all(
        key.startswith("perception.scan_guard.") for key in v5_changes
    )

    for section in ("planner", "rl"):
        assert v5_config[section] == v4_config[section]
    assert (
        int(v5_config["planner"]["num_samples"])
        * int(v5_config["planner"]["paper_rl_driven"]["iterations"])
        == 600
    )


def test_v5_recovery_contract_keeps_frozen_risk_thresholds_and_budget():
    protocol, source = _protocols()
    block = dict(protocol["development_blocks"][0])
    v4_config, _ = development._configure_arm(
        protocol, source, block, "V4_full_frozen"
    )
    v5_config, _ = development._configure_arm(
        protocol, source, block, "V5_recovery_full"
    )

    for key in (
        "probabilistic_obstacle_hard_threshold",
        "probabilistic_obstacle_safety_margin",
        "probabilistic_obstacle_stopping_feasibility_enabled",
        "probabilistic_obstacle_emergency_candidates_enabled",
    ):
        assert v5_config["planner"][key] == v4_config["planner"][key]
    assert v5_config["planner"]["horizon"] == 36
    assert v5_config["experiment"]["control_dt"] == 0.1
    contract = protocol["recovery_contract"]
    assert not contract["new_collision_probability_threshold_added"]
    assert contract["new_rollout_candidates_added"] == 0
    assert contract["total_rollouts_per_decision"] == 600
    assert contract["alignment_creep_requires_front_clear"]
    assert contract["alignment_creep_aborts_on_existing_abort_conditions"]
    assert contract[
        "alignment_creep_requires_non_decreasing_scan_clearance"
    ]
    assert contract["progress_watch_cancelled_by_existing_risk_guards"]
    assert contract["progress_watch_uses_online_goal_distance_only"]
    assert contract["initial_forward_commit_s"] == 0.8
    assert (
        protocol["v5_recovery_overrides"][
            "dynamic_recovery_minimum_forward_commit_steps"
        ]
        == 8
    )
    assert protocol["v5_recovery_overrides"][
        "dynamic_recovery_progress_watch_enabled"
    ]


def test_v5_analysis_uses_certificate_enriched_schedule_blocks(monkeypatch):
    protocol, source = _protocols()
    blocks = [dict(protocol["development_blocks"][0])]
    blocks[0]["certificate"] = {"conflict_windows_s": [[1.0, 2.0]]}
    seen = []

    def fake_episode_row(output, block, arm, source_protocol):
        seen.append(block)
        return {
            "arm": arm,
            "split": block["split"],
            "seed": int(block["seed"]),
            "model_block": int(block["model_block"]),
            "stratum": block["stratum"],
            "success": False,
            "collision": False,
            "steps": 400,
            "final_goal_distance": 1.0,
            "minimum_clearance": 0.5,
            "stuck_steps": 10,
            "planner_compute_ms_p95_health_only": 1.0,
            "paper_total_rollouts_mean": 600.0,
            "dynamic_recovery_active_steps": 0,
            "dynamic_recovery_advance_steps": 0,
            "dynamic_recovery_forward_commit_steps": 0,
            "dynamic_recovery_speed_floor_mean": 0.0,
            "zero_speed_risk_steps": 10,
            "release_delay_max_s": 1.0,
        }

    monkeypatch.setattr(development, "_episode_row", fake_episode_row)
    development._analyze(protocol, source, "unused", blocks)

    assert len(seen) == 2
    assert all(row["certificate"] == blocks[0]["certificate"] for row in seen)


def test_v5_deadline_b1_is_a_new_outcome_informed_mechanism_family():
    protocol, _ = _deadline_protocols()

    assert protocol["design"]["development_attempt"] == "B1"
    assert protocol["design"]["mechanism_family"] == (
        "online_deadline_feasibility_supervisor"
    )
    assert protocol["design"]["outcome_informed_development_selection"]
    assert not protocol["design"]["formal_effect_estimation_authorized"]
    assert protocol["design"]["paired_arms"] == [
        "V4_full_frozen",
        "V5_deadline_full",
    ]
    development._validate_parent_attempt(protocol)


def test_v5_deadline_b1_changes_only_declared_scan_guard_fields():
    protocol, source = _deadline_protocols()
    block = dict(protocol["development_blocks"][0])

    v4_config, v4_changes = development._configure_arm(
        protocol, source, block, "V4_full_frozen"
    )
    v5_config, v5_changes = development._configure_arm(
        protocol, source, block, "V5_deadline_full"
    )

    assert not v4_changes
    assert set(v5_changes) <= {
        "perception.scan_guard.%s" % key
        for key in protocol["candidate_overrides"]
    }
    assert all(
        key.startswith("perception.scan_guard.") for key in v5_changes
    )
    assert v5_config["perception"]["scan_guard"][
        "dynamic_deadline_supervisor_enabled"
    ]
    assert not v5_config["perception"]["scan_guard"][
        "dynamic_recovery_progress_watch_enabled"
    ]
    for section in ("planner", "rl"):
        assert v5_config[section] == v4_config[section]
    assert v5_config["experiment"]["max_steps"] == 400
    assert v5_config["experiment"]["control_dt"] == 0.1
    assert v5_config["task"]["position_tolerance"] == 0.3
    assert (
        int(v5_config["planner"]["num_samples"])
        * int(v5_config["planner"]["paper_rl_driven"]["iterations"])
        == 600
    )


def test_v5_deadline_b1_gate_is_paired_safety_noninferiority():
    protocol, _ = _deadline_protocols()
    gate = protocol["development_go_no_go"]
    contract = protocol["deadline_supervisor_contract"]

    assert "maximum_v5_collisions" not in gate
    assert gate["maximum_new_paired_collisions"] == 0
    assert gate["require_collision_count_not_worse"]
    assert gate["minimum_challenge_conversions"] == 2
    assert contract["requires_prior_dynamic_escape"]
    assert contract["renewed_hazard_has_immediate_priority"]
    assert not contract["uses_truth_conflict_window"]
    assert not contract["uses_future_obstacle_trajectory"]
    assert not contract["changes_steering_command"]
    assert contract["total_rollouts_per_decision"] == 600


def test_v5_deadline_b2_isolates_deadline_supervisor_from_a5_recovery():
    protocol, source = _deadline_b2_protocols()
    assert protocol["design"]["development_attempt"] == "B2"
    assert protocol["design"]["mechanism_family"] == (
        "online_deadline_feasibility_supervisor"
    )
    assert protocol["design"]["paired_arms"] == [
        "V4_full_frozen",
        "V5_deadline_isolated_full",
    ]
    development._validate_parent_attempt(protocol)

    overrides = protocol["candidate_overrides"]
    assert len(overrides) == 11
    assert all(key.startswith("dynamic_deadline_") for key in overrides)
    assert not any(key.startswith("dynamic_recovery_") for key in overrides)

    block = dict(protocol["development_blocks"][0])
    v4_config, v4_changes = development._configure_arm(
        protocol, source, block, "V4_full_frozen"
    )
    v5_config, v5_changes = development._configure_arm(
        protocol, source, block, "V5_deadline_isolated_full"
    )
    assert not v4_changes
    assert set(v5_changes) == {
        "perception.scan_guard.%s" % key for key in overrides
    }
    v4_guard = v4_config["perception"]["scan_guard"]
    v5_guard = v5_config["perception"]["scan_guard"]
    assert v4_guard["dynamic_recovery_enabled"]
    assert {
        key: value
        for key, value in v5_guard.items()
        if key.startswith("dynamic_recovery_")
    } == {
        key: value
        for key, value in v4_guard.items()
        if key.startswith("dynamic_recovery_")
    }
    for section in ("planner", "rl"):
        assert v5_config[section] == v4_config[section]
    assert v5_config["experiment"]["max_steps"] == 400
    assert v5_config["experiment"]["control_dt"] == 0.1
    assert v5_config["task"]["position_tolerance"] == 0.3
    assert (
        int(v5_config["planner"]["num_samples"])
        * int(v5_config["planner"]["paper_rl_driven"]["iterations"])
        == 600
    )


def test_v5_deadline_b2_gate_is_identical_to_b1():
    b1, _ = _deadline_protocols()
    b2, _ = _deadline_b2_protocols()
    assert b2["development_go_no_go"] == b1["development_go_no_go"]
    assert b2["deadline_supervisor_contract"] == (
        b1["deadline_supervisor_contract"]
    )


def test_v5_deadline_b3_changes_only_terminal_reserve_from_b2():
    b2, _ = _deadline_b2_protocols()
    b3, source = _deadline_b3_protocols()
    development._validate_parent_attempt(b3)
    assert b3["design"]["development_attempt"] == "B3"
    assert b3["design"]["final_local_attempt_in_family"]
    assert b3["development_go_no_go"] == b2["development_go_no_go"]
    assert b3["deadline_supervisor_contract"] == (
        b2["deadline_supervisor_contract"]
    )
    left = dict(b2["candidate_overrides"])
    right = dict(b3["candidate_overrides"])
    assert left.pop("dynamic_deadline_reserve_steps") == 2
    assert right.pop("dynamic_deadline_reserve_steps") == 0
    assert right == left

    block = dict(b3["development_blocks"][0])
    v4_config, v4_changes = development._configure_arm(
        b3, source, block, "V4_full_frozen"
    )
    v5_config, v5_changes = development._configure_arm(
        b3, source, block, "V5_deadline_isolated_full"
    )
    assert not v4_changes
    assert set(v5_changes) == {
        "perception.scan_guard.%s" % key
        for key in b3["candidate_overrides"]
    }
    assert v5_config["perception"]["scan_guard"][
        "dynamic_deadline_reserve_steps"
    ] == 0
    assert (
        int(v5_config["planner"]["num_samples"])
        * int(v5_config["planner"]["paper_rl_driven"]["iterations"])
        == 600
    )


def test_v5_dual_horizon_c1_is_new_structural_family_with_same_gate():
    b2, _ = _deadline_b2_protocols()
    c1, source = _dual_horizon_c1_protocols()
    development._validate_parent_attempt(c1)
    assert c1["design"]["development_attempt"] == "C1"
    assert c1["design"]["mechanism_family"] == (
        "dual_horizon_deadline_supervisor"
    )
    assert c1["development_go_no_go"] == b2["development_go_no_go"]
    assert len(c1["candidate_overrides"]) == 12
    assert c1["candidate_overrides"][
        "dynamic_deadline_reserve_steps"
    ] == 2
    assert c1["candidate_overrides"][
        "dynamic_deadline_authority_reserve_steps"
    ] == 0

    block = dict(c1["development_blocks"][0])
    v4_config, v4_changes = development._configure_arm(
        c1, source, block, "V4_full_frozen"
    )
    v5_config, v5_changes = development._configure_arm(
        c1, source, block, "V5_dual_horizon_full"
    )
    assert not v4_changes
    assert set(v5_changes) == {
        "perception.scan_guard.%s" % key
        for key in c1["candidate_overrides"]
    }
    v4_guard = v4_config["perception"]["scan_guard"]
    v5_guard = v5_config["perception"]["scan_guard"]
    assert {
        key: value
        for key, value in v5_guard.items()
        if key.startswith("dynamic_recovery_")
    } == {
        key: value
        for key, value in v4_guard.items()
        if key.startswith("dynamic_recovery_")
    }
    for section in ("planner", "rl"):
        assert v5_config[section] == v4_config[section]
    assert v5_config["experiment"]["max_steps"] == 400
    assert v5_config["experiment"]["control_dt"] == 0.1
    assert v5_config["task"]["position_tolerance"] == 0.3
    assert (
        int(v5_config["planner"]["num_samples"])
        * int(v5_config["planner"]["paper_rl_driven"]["iterations"])
        == 600
    )


def test_v5_dual_horizon_c2_changes_only_urgency_reserve():
    c1, _ = _dual_horizon_c1_protocols()
    c2, source = _dual_horizon_c2_protocols()
    development._validate_parent_attempt(c2)
    assert c2["design"]["development_attempt"] == "C2"
    assert c2["design"]["mechanism_family"] == (
        "dual_horizon_deadline_supervisor"
    )
    assert c2["development_go_no_go"] == c1["development_go_no_go"]
    left = dict(c1["candidate_overrides"])
    right = dict(c2["candidate_overrides"])
    assert left.pop("dynamic_deadline_reserve_steps") == 2
    assert right.pop("dynamic_deadline_reserve_steps") == 3
    assert right == left

    block = dict(c2["development_blocks"][0])
    v4_config, v4_changes = development._configure_arm(
        c2, source, block, "V4_full_frozen"
    )
    v5_config, v5_changes = development._configure_arm(
        c2, source, block, "V5_dual_horizon_full"
    )
    assert not v4_changes
    assert set(v5_changes) == {
        "perception.scan_guard.%s" % key
        for key in c2["candidate_overrides"]
    }
    assert v5_config["perception"]["scan_guard"][
        "dynamic_deadline_authority_reserve_steps"
    ] == 0
    assert (
        int(v5_config["planner"]["num_samples"])
        * int(v5_config["planner"]["paper_rl_driven"]["iterations"])
        == 600
    )


def test_v5_dual_horizon_c3_is_final_and_changes_only_urgency_reserve():
    c2, _ = _dual_horizon_c2_protocols()
    c3, source = _dual_horizon_c3_protocols()
    development._validate_parent_attempt(c3)
    assert c3["design"]["development_attempt"] == "C3"
    assert c3["design"]["final_local_attempt_in_family"]
    assert c3["development_go_no_go"] == c2["development_go_no_go"]
    left = dict(c2["candidate_overrides"])
    right = dict(c3["candidate_overrides"])
    assert left.pop("dynamic_deadline_reserve_steps") == 3
    assert right.pop("dynamic_deadline_reserve_steps") == 5
    assert right == left

    block = dict(c3["development_blocks"][0])
    v4_config, v4_changes = development._configure_arm(
        c3, source, block, "V4_full_frozen"
    )
    v5_config, v5_changes = development._configure_arm(
        c3, source, block, "V5_dual_horizon_full"
    )
    assert not v4_changes
    assert set(v5_changes) == {
        "perception.scan_guard.%s" % key
        for key in c3["candidate_overrides"]
    }
    assert v5_config["perception"]["scan_guard"][
        "dynamic_deadline_authority_reserve_steps"
    ] == 0
    assert (
        int(v5_config["planner"]["num_samples"])
        * int(v5_config["planner"]["paper_rl_driven"]["iterations"])
        == 600
    )
