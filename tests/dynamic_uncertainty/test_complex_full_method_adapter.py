from pathlib import Path

from experiments.dynamic_uncertainty.complex_full_method import (
    MAPS,
    ROOT,
    build_complex_full_config,
)
from mobile_robot_mppi.runtime.factories import make_components


SHARED_PLANNER_KEYS = (
    "horizon",
    "num_samples",
    "known_static_map_cost_enabled",
    "known_static_map_influence_m",
    "known_static_map_weight",
    "known_static_map_collision_penalty",
    "known_static_map_candidate_filter_enabled",
    "probabilistic_reference_authority_enabled",
    "probabilistic_reference_authority_filter_alpha",
    "probabilistic_reference_authority_minimum",
    "probabilistic_reference_authority_risk_source",
    "probabilistic_reference_progress_weight",
    "probabilistic_obstacle_emergency_candidates_enabled",
    "probabilistic_obstacle_emergency_candidate_trigger_ttc_s",
    "probabilistic_obstacle_emergency_candidate_trigger_distance_m",
    "probabilistic_obstacle_emergency_candidate_critical_distance_m",
    "static_astar_replan_enabled",
    "static_astar_replan_resolution_m",
    "static_astar_replan_clearance_margin_m",
    "static_astar_replan_deviation_m",
    "static_astar_replan_stagnation_steps",
    "static_astar_replan_minimum_progress_m",
    "static_astar_replan_cooldown_steps",
    "path_boundary_enabled",
    "path_boundary_candidate_filter_enabled",
)


def test_all_maps_use_the_actual_b11_full_stack_and_600_rollouts():
    configs = {
        name: build_complex_full_config(name, 790100001)
        for name in MAPS
    }

    for name, config in configs.items():
        planner = config["planner"]
        tracker = config["perception"]["dynamic_obstacle_tracker"]
        obstacles = config["scene"]["obstacles"]
        dynamics = [
            item for item in obstacles
            if isinstance(item.get("motion"), dict)
        ]

        assert config["complex_method_contract"]["arm"] == (
            "B11_full_proposed"
        )
        assert planner["sampling_prior"] == "paper_direct_rl"
        assert config["rl"]["enabled"]
        assert planner["probabilistic_obstacle_risk_enabled"]
        assert planner[
            "probabilistic_obstacle_emergency_candidates_enabled"
        ]
        assert (
            planner[
                "probabilistic_obstacle_emergency_candidate_trigger_ttc_s"
            ]
            > 0.0
        )
        assert (
            planner[
                "probabilistic_obstacle_emergency_candidate_trigger_distance_m"
            ]
            > planner[
                "probabilistic_obstacle_emergency_candidate_critical_distance_m"
            ]
            > 0.0
        )
        assert tracker["enabled"]
        assert tracker["predictor_mode"] == "change_aware"
        assert tracker["maximum_tracks"] == 3
        assert tracker["known_static_filter_enabled"]
        assert config["perception"]["scan_guard"][
            "dynamic_escape_use_vetted_planner_control"
        ]
        assert len(dynamics) == 3
        assert (
            planner["num_samples"]
            * planner["paper_rl_driven"]["iterations"]
        ) == 600
        assert not planner["path_boundary_enabled"]
        assert not planner["path_boundary_candidate_filter_enabled"]
        assert not config["task"]["terminate_on_boundary_violation"]
        assert Path(config["rl"]["checkpoint"]).is_file()
        assert config["experiment"]["complex_map"] == name


def test_all_maps_share_one_controller_parameter_contract():
    configs = [
        build_complex_full_config(name, 790100003)
        for name in MAPS
    ]
    expected_planner = {
        key: configs[0]["planner"][key]
        for key in SHARED_PLANNER_KEYS
    }
    expected_action = configs[0]["action_space"]
    expected_tracker = configs[0]["perception"][
        "dynamic_obstacle_tracker"
    ]
    expected_guard = configs[0]["perception"]["scan_guard"]

    for config in configs[1:]:
        assert {
            key: config["planner"][key]
            for key in SHARED_PLANNER_KEYS
        } == expected_planner
        assert config["action_space"] == expected_action
        tracker = config["perception"]["dynamic_obstacle_tracker"]
        assert config["perception"]["scan_guard"][
            "dynamic_escape_use_vetted_planner_control"
        ] == expected_guard["dynamic_escape_use_vetted_planner_control"]
        for key in (
            "maximum_tracks",
            "known_static_filter_enabled",
            "known_static_filter_tolerance_m",
            "association_gate_m",
            "maximum_unobserved_duration_s",
            "track_retirement_duration_s",
            "motion_confirmation_enabled",
            "motion_confirmation_required_observations",
            "motion_confirmation_minimum_speed_mps",
        ):
            assert tracker[key] == expected_tracker[key]


def test_common_override_does_not_add_a_navigation_decision_layer():
    config = build_complex_full_config("chapter3", 790100005)

    assert "maneuver_manager" not in config
    assert "space_time_corridor" not in config
    assert "navigation_field" not in config
    assert config["complex_method_contract"] == {
        "arm": "B11_full_proposed",
        "core_method_modified": False,
        "global_reference": "soft_static_astar",
        "dynamic_obstacles_in_astar": False,
        "shared_common_override": config[
            "complex_method_contract"
        ]["shared_common_override"],
    }


def test_runtime_constructs_paper_rl_inside_the_frozen_residual_shield():
    config = build_complex_full_config("chapter3", 790100007)
    components = make_components(config, ROOT)
    controller = components["controller"]
    try:
        assert type(controller).__name__ == (
            "ResidualSafetyShieldController"
        )
        assert type(controller.residual_controller).__name__ == (
            "PaperRLDrivenMppiController"
        )
        assert type(controller.nominal_controller).__name__ == (
            "PaperRLDrivenMppiController"
        )
        assert type(
            controller.residual_controller.dynamics
        ).__name__ == "ResidualPrediction"
    finally:
        controller.close()
        components["plant"].close()
