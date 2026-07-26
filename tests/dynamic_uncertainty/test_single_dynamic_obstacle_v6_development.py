from experiments.dynamic_uncertainty import (
    run_single_dynamic_obstacle_paper_v4 as v4,
)
from experiments.dynamic_uncertainty import (
    run_single_dynamic_obstacle_v6_development as v6,
)


def _protocols():
    protocol = v4._load_yaml(v6.DEFAULT_PROTOCOL)
    _, source, _, _, _ = v4.validate_protocol(
        v6._repo_path(protocol["source_protocol"])
    )
    return protocol, source


def test_v6_a1_is_development_only_and_binds_failed_v5_evidence():
    protocol, _ = _protocols()
    assert protocol["design"]["outcome_informed_development_selection"]
    assert protocol["design"][
        "previously_opened_v5_held_out_seeds_reused_for_development"
    ]
    assert not protocol["design"]["formal_effect_estimation_authorized"]
    assert protocol["design"]["fresh_held_out_required_before_formal_freeze"]
    assert len(protocol["development_blocks"]) == 8
    assert len({block["seed"] for block in protocol["development_blocks"]}) == 8
    v6._validate_parent(protocol)


def test_v6_a1_changes_only_emergency_candidate_and_escape_vetting_paths():
    protocol, source = _protocols()
    block = dict(protocol["development_blocks"][0])
    control, control_changes = v6._configure_arm(
        protocol, source, block, "V4_full_frozen"
    )
    candidate, candidate_changes = v6._configure_arm(
        protocol, source, block, "V6_temporal_emergency_full"
    )
    assert control_changes == {}
    assert set(candidate_changes) == {
        path
        for path, value in protocol["candidate_override_paths"].items()
        if v4._flatten(control).get(path, "<MISSING>") != value
    }
    assert all(
        path.startswith(tuple(protocol["change_scope"]["allowed_prefixes"]))
        for path in candidate_changes
    )
    assert not candidate["perception"]["scan_guard"].get(
        "dynamic_deadline_supervisor_enabled", False
    )
    assert int(candidate["planner"]["num_samples"]) * int(
        candidate["planner"]["paper_rl_driven"]["iterations"]
    ) == 600
    for frozen in ("rl",):
        assert candidate[frozen] == control[frozen]


def test_v6_a1_uses_current_hazard_evidence_without_truth_or_deadline():
    protocol, _ = _protocols()
    contract = protocol["mechanism_contract"]
    assert contract["trigger_uses_current_online_temporal_scan_ttc"]
    assert contract["requires_same_cycle_probabilistic_vetting"]
    assert contract["renewed_hazard_has_immediate_priority"]
    assert contract["stop_candidate_retained"]
    assert not contract["truth_conflict_window_used_for_control"]
    assert not contract["future_obstacle_truth_used_for_control"]
    assert not contract["deadline_or_remaining_episode_time_used_for_control"]
    assert contract["additional_total_rollouts"] == 0
    assert contract["total_rollouts_per_decision"] == 600
