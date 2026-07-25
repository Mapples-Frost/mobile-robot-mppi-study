from pathlib import Path

import pytest

from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "mujoco_three_dynamic_crossing_development_amendment65.yaml"
)
TIME_BUDGET_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "mujoco_three_dynamic_crossing_development_amendment71.yaml"
)
A69_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "mujoco_three_dynamic_crossing_development_amendment69.yaml"
)
A70_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "mujoco_three_dynamic_crossing_development_amendment70.yaml"
)
A72_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "mujoco_three_dynamic_crossing_development_amendment72.yaml"
)
A73_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "mujoco_three_dynamic_crossing_development_amendment73.yaml"
)


def test_three_obstacle_randomization_preserves_conflict_and_feasible_gaps():
    config = load_yaml(CONFIG)
    obstacles = config["scene"]["obstacles"]
    task = config["task"]
    planner = config["planner"]
    contract = config["three_obstacle_feasibility_contract"]

    assert len(obstacles) == 3
    assert task["points"][0][1] == pytest.approx(task["points"][-1][1])
    route_y = float(task["points"][0][1])
    crossing_width = float(contract["certified_crossing_width_m"])
    robot_speed = float(contract["maximum_robot_speed_mps"])
    actuator_margin = float(contract["actuator_margin_s"])
    required_gap = crossing_width / robot_speed + actuator_margin
    assert required_gap == pytest.approx(
        contract["minimum_required_free_gap_s"]
    )
    assert planner["num_samples"] == 600

    for obstacle in obstacles:
        motion = obstacle["motion"]
        start = motion["start"]
        end = motion["end"]
        jitter = float(motion["endpoint_jitter_m"])
        scale_low, scale_high = motion["period_scale_range"]

        assert motion["type"] == "linear_ping_pong"
        assert float(motion["phase_jitter_s"]) > 0.0
        assert float(scale_low) < 1.0 < float(scale_high)
        assert jitter > 0.0
        assert min(start[1], end[1]) + jitter < route_y
        assert max(start[1], end[1]) - jitter > route_y

        nominal_span = abs(float(end[1]) - float(start[1]))
        worst_span = nominal_span - 2.0 * jitter
        worst_period = float(motion["period_s"]) * float(scale_low)
        # Ping-pong traverses the endpoint span in half a period.  The
        # obstacle occupies a crossing band of ``crossing_width``; the rest of
        # that half-cycle is the controller's usable free gap.
        worst_free_gap = worst_period * (
            0.5 - crossing_width / (2.0 * worst_span)
        )
        assert worst_free_gap >= required_gap


def test_feasible_period_revision_extends_episode_time_budget():
    config = load_yaml(TIME_BUDGET_CONFIG)
    contract = config["three_obstacle_time_budget_contract"]
    original = contract["original_obstacle_periods_s"]
    feasible = contract["feasible_obstacle_periods_s"]
    dt = float(contract["control_dt_s"])
    additional_wait = 0.5 * sum(
        float(new) - float(old)
        for old, new in zip(original, feasible)
    )
    minimum_steps = int(contract["original_max_steps"]) + int(
        round(additional_wait / dt)
    )

    assert additional_wait == pytest.approx(
        contract["additional_worst_half_cycle_wait_s"]
    )
    assert minimum_steps == contract["minimum_adjusted_max_steps"]
    assert config["experiment"]["max_steps"] == 600
    assert config["experiment"]["max_steps"] >= minimum_steps
    assert config["planner"]["num_samples"] == 600


def test_amendment71_reuses_frozen_ttc_and_rollout_contracts():
    config = load_yaml(TIME_BUDGET_CONFIG)
    a69 = load_yaml(A69_CONFIG)
    a70 = load_yaml(A70_CONFIG)
    assert config["planner"][
        "probabilistic_obstacle_traversal_window_post_center_low_ttc_continuity_guard_enabled"
    ]
    contract = config["three_obstacle_low_ttc_continuity_contract"]
    assert contract["requires_current_temporal_scan_valid"]
    assert contract["requires_prior_matched_encounter_unrearmed"]
    assert contract["total_rollouts_per_decision"] == 600
    assert config["planner"]["num_samples"] == 600
    assert not a69["planner"].get(
        "probabilistic_obstacle_traversal_window_post_center_low_ttc_nonforward_coverage_enabled",
        False,
    )
    assert not a70["planner"].get(
        "probabilistic_obstacle_traversal_window_post_center_low_ttc_first_step_boundary_handoff_enabled",
        False,
    )
    assert config["planner"][
        "probabilistic_obstacle_traversal_window_post_center_low_ttc_nonforward_coverage_enabled"
    ]
    coverage = config[
        "three_obstacle_low_ttc_nonforward_coverage_contract"
    ]
    assert coverage["reverse_templates_reserved"] == 3
    assert coverage["total_emergency_slots"] == 6
    assert coverage["total_rollouts_per_decision"] == 600
    assert config["planner"][
        "probabilistic_obstacle_traversal_window_post_center_low_ttc_first_step_boundary_handoff_enabled"
    ]
    handoff = config["three_obstacle_first_step_boundary_handoff_contract"]
    assert handoff[
        "ordinary_full_horizon_boundary_filter_preserved"
    ]
    assert handoff[
        "emergency_candidate_must_be_first_step_boundary_feasible"
    ]
    assert handoff["candidates_added"] == 0
    assert handoff["total_rollouts_per_decision"] == 600


def test_amendment72_replaces_first_step_with_frozen_prefix_certificate():
    a71 = load_yaml(TIME_BUDGET_CONFIG)
    a72 = load_yaml(A72_CONFIG)

    assert a71["planner"][
        "probabilistic_obstacle_traversal_window_post_center_low_ttc_first_step_boundary_handoff_enabled"
    ]
    assert not a71["planner"].get(
        "probabilistic_obstacle_traversal_window_post_center_low_ttc_prefix_boundary_handoff_enabled",
        False,
    )
    assert not a72["planner"][
        "probabilistic_obstacle_traversal_window_post_center_low_ttc_first_step_boundary_handoff_enabled"
    ]
    assert a72["planner"][
        "probabilistic_obstacle_traversal_window_post_center_low_ttc_prefix_boundary_handoff_enabled"
    ]
    contract = a72["three_obstacle_prefix_boundary_handoff_contract"]
    assert contract["amendment71_first_step_handoff_disabled"]
    assert contract[
        "emergency_candidate_must_be_complete_prefix_boundary_feasible"
    ]
    assert contract["prefix_steps"] == a72["planner"][
        "probabilistic_obstacle_emergency_candidate_prefix_steps"
    ]
    assert contract["prefix_boundary_margin_floor_m"] == 0.0
    assert contract["candidates_added"] == 0
    assert contract["total_rollouts_per_decision"] == 600


def test_amendment73_is_opt_in_and_reuses_frozen_geometry_and_budget():
    a72 = load_yaml(A72_CONFIG)
    a73 = load_yaml(A73_CONFIG)

    for key in (
        "probabilistic_obstacle_traversal_window_rearm_staging_approach_enabled",
        "probabilistic_obstacle_traversal_window_post_center_forward_exit_commit_enabled",
    ):
        assert not a72["planner"].get(key, False)
        assert a73["planner"][key]

    staging = a73["three_obstacle_rearm_staging_approach_contract"]
    assert staging["target"] == "existing_frozen_crossing_entry_progress"
    assert staging["open_loop_sequence_stops_at_target"]
    assert not staging["conflict_entry_may_be_crossed_while_rearm_pending"]
    assert not staging["threshold_added"]

    forward = a73["three_obstacle_post_center_forward_exit_contract"]
    assert forward["direction_transaction"] == (
        "nonreverse_until_existing_clear_progress"
    )
    assert forward["forward_templates_reserved"] == 3
    assert forward["emergency_templates_added"] == 0
    assert forward["total_emergency_slots"] == 6
    assert forward["total_rollouts_per_decision"] == 600
    assert a73["planner"]["num_samples"] == 600
