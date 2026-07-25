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
