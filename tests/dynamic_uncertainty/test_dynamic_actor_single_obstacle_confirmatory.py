from pathlib import Path

import yaml

from experiments.dynamic_uncertainty.analyze_dynamic_actor_single_obstacle_confirmatory import (
    _one_sided_binomial_upper,
)
from experiments.dynamic_uncertainty.run_dynamic_actor_single_obstacle_confirmatory import (
    _load_yaml,
    build_schedule,
    configure_arm,
    validate_protocol,
)
from experiments.dynamic_uncertainty.run_rl_hss_stage4 import _mapping
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/research/dynamic_actor_single_obstacle_confirmatory.yaml"


def test_confirmatory_protocol_is_frozen_complete_and_disjoint():
    _, protocol, registry, blocks, jobs, verified = validate_protocol(PROTOCOL)
    assert len(blocks) == 100
    assert len(jobs) == 350
    assert len(verified) == len(protocol["bindings"])
    assert {block["seed"] for block in blocks}.isdisjoint(
        range(730100001, 730100301)
    )
    assert sum(block["ablation_block"] for block in blocks) == 50
    assert sum(job["arm"] == "dynamic_actor" for job in jobs) == 100
    assert sum(job["arm"] == "source_actor" for job in jobs) == 100
    for arm in protocol["design"]["ablation_arms"]:
        assert sum(job["arm"] == arm for job in jobs) == 50


def test_latin_arm_positions_are_balanced_within_each_split():
    protocol = _load_yaml(PROTOCOL)
    registry = _load_yaml(ROOT / protocol["seed_registry_path"])
    _, jobs = build_schedule(protocol, registry)
    arms = list(protocol["arms"])
    for split in ("held_out_id", "held_out_ood"):
        ablation = [
            job for job in jobs if job["split"] == split and job["ablation_block"]
        ]
        for arm in arms:
            counts = [
                sum(job["arm"] == arm and job["arm_position"] == position for job in ablation)
                for position in range(1, 6)
            ]
            assert counts == [5, 5, 5, 5, 5]


def test_ood_process_shift_changes_motion_but_not_physical_limits():
    base = yaml.safe_load(
        (ROOT / "configs/research/dynamic_obstacle_process_v3.yaml").read_text(encoding="utf-8")
    )
    ood = yaml.safe_load(
        (ROOT / "configs/research/dynamic_obstacle_process_v3_ood_confirmatory.yaml").read_text(encoding="utf-8")
    )
    assert ood["physical_limits"] == base["physical_limits"]
    assert ood["noise_profiles"]["process_shift"]["acceleration_std"] > base["noise_profiles"]["medium"]["acceleration_std"]
    assert ood["navigation"]["maximum_cruise_speed_mps"] > base["navigation"]["maximum_cruise_speed_mps"]
    assert ood["scope_guards"]["sensor_shift_not_claimed"] is True


def test_configured_jobs_preserve_budget_causality_and_ablation_scope():
    _, protocol, registry, _, jobs, _ = validate_protocol(PROTOCOL)
    base = load_yaml(ROOT / protocol["base_config"])
    stage3 = _mapping(ROOT / protocol["stage3_protocol"])
    stage4 = _mapping(ROOT / protocol["stage4_protocol"])
    selected = {}
    for job in jobs:
        selected.setdefault(job["arm"], job)
    configs = {
        arm: configure_arm(protocol, job, base, stage3, stage4)
        for arm, job in selected.items()
    }
    for config in configs.values():
        paper = config["planner"]["paper_rl_driven"]
        assert config["planner"]["horizon"] == 36
        assert config["planner"]["num_samples"] * paper["iterations"] == 600
        assert len(config["scene"]["obstacles"]) == 1
        assert config["scope_guards"]["simulator_truth_for_control"] is False
        assert config["scope_guards"]["exact_ground_truth_pose_for_control"] is False
    assert configs["dynamic_actor"]["planner"]["paper_rl_driven"]["same_cycle_guided_cost_filter"] is True
    assert configs["ablation_no_same_cycle_filter"]["planner"]["paper_rl_driven"]["same_cycle_guided_cost_filter"] is False
    assert configs["dynamic_actor"]["planner"]["probabilistic_obstacle_emergency_candidate_pareto_forward_commit_enabled"] is True
    assert configs["ablation_no_pareto_commit"]["planner"]["probabilistic_obstacle_emergency_candidate_pareto_forward_commit_enabled"] is False
    assert configs["ablation_no_temporal_escape_package"]["perception"]["scan_guard"]["dynamic_escape_use_vetted_planner_control"] is False


def test_registered_zero_event_precision_matches_preregistration():
    assert abs(_one_sided_binomial_upper(0, 100) - 0.029513049607039932) < 1e-12
    assert abs(_one_sided_binomial_upper(0, 50) - 0.058155079116972264) < 1e-12
