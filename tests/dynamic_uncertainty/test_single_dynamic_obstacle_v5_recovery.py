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
