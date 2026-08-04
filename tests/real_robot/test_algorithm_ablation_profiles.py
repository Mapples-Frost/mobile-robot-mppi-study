from pathlib import Path

import pytest

from deploy.raspberry_pi5_scout.build_pi5_full_config import (
    apply_pi5_algorithm_features,
    build_pi5_full_config,
    resolve_pi5_algorithm_features,
)
from experiments.dynamic_uncertainty.complex_full_method import (
    build_complex_full_config,
)
from mobile_robot_mppi.planning.mppi import MppiController
from mobile_robot_mppi.runtime.factories import make_components


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _weight_root(tmp_path: Path) -> Path:
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in (
        "icode_stage2_task_aware.pt",
        "actor_full_proposed.pt",
        "hss_member1.pt",
        "hss_member2.pt",
        "hss_member3.pt",
    ):
        (weights / name).write_bytes(b"profile-contract-placeholder")
    return tmp_path


def _config(tmp_path: Path):
    return build_pi5_full_config(
        _weight_root(tmp_path),
        goal_x=5.0,
        goal_y=2.0,
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )


def test_no_enhancement_switches_resolve_to_traditional_mppi():
    features = resolve_pi5_algorithm_features()
    assert features
    assert not any(features.values())


def test_real_robot_tracker_gate_does_not_change_simulation_profile():
    simulation = build_complex_full_config(
        "chapter2_admissible", 20260803, arm="B11_full_proposed"
    )
    assert simulation["perception"]["dynamic_obstacle_tracker"][
        "motion_confirmation_enabled"
    ] is True


def test_full_preset_can_disable_only_residual_learning():
    features = resolve_pi5_algorithm_features(
        full_proposed=True,
        disable_residual_learning=True,
    )
    assert features["residual_learning"] is False
    assert all(
        enabled
        for name, enabled in features.items()
        if name != "residual_learning"
    )


@pytest.mark.parametrize(
    "arguments, message",
    (
        ({"enable_hss_reliability": True}, "HSS"),
        ({"enable_probabilistic_risk": True}, "change-aware"),
        ({"enable_forward_passage": True}, "probabilistic"),
        (
            {
                "traditional_mppi": True,
                "enable_ar1_sampling": True,
            },
            "traditional",
        ),
    ),
)
def test_invalid_feature_dependencies_fail_before_arming(arguments, message):
    with pytest.raises(ValueError, match=message):
        resolve_pi5_algorithm_features(**arguments)


def test_traditional_profile_is_standard_mppi_with_common_safety(tmp_path):
    config = _config(tmp_path)
    hard_stop = config["perception"]["scan_guard"]["hard_stop_distance"]
    local_obstacles = dict(config["perception"]["local_obstacle_layer"])

    features = resolve_pi5_algorithm_features(traditional_mppi=True)
    apply_pi5_algorithm_features(config, features)

    planner = config["planner"]
    assert planner["optimizer"] == "standard"
    assert planner["sampling_prior"] == "goal_warm_start"
    assert planner["num_samples"] == 600
    assert planner["prediction_mode"] == "nominal"
    assert planner["noise_basis"] == "iid"
    assert planner["residual_safety_shield"]["enabled"] is False
    assert config["rl"]["enabled"] is False
    assert planner["paper_rl_driven"]["reliability_sidecar"] == {}
    assert config["perception"]["dynamic_obstacle_tracker"]["enabled"] is False
    assert all(
        not value
        for key, value in planner.items()
        if key.startswith("probabilistic_") and key.endswith("_enabled")
    )

    # Necessary physical and static-geometry layers are common controls, not
    # Full-Proposed enhancements, and therefore remain identical.
    assert config["perception"]["local_obstacle_layer"] == local_obstacles
    assert config["perception"]["scan_guard"][
        "hard_stop_distance"
    ] == hard_stop
    assert config["perception"]["temporal_scan_guard"][
        "safety_enabled"
    ] is True
    assert config["real_robot_deployment"][
        "algorithm_profile"
    ] == "traditional_mppi"

    components = make_components(config, PROJECT_ROOT)
    assert type(components["controller"]) is MppiController


def test_recommended_profile_keeps_everything_except_residual(tmp_path):
    config = _config(tmp_path)
    features = resolve_pi5_algorithm_features(
        full_proposed=True,
        disable_residual_learning=True,
    )
    apply_pi5_algorithm_features(config, features)

    planner = config["planner"]
    paper = planner["paper_rl_driven"]
    assert planner["optimizer"] == "paper_rl_driven"
    assert planner["sampling_prior"] == "paper_direct_rl"
    assert planner["prediction_mode"] == "nominal"
    assert planner["residual_safety_shield"]["enabled"] is False
    assert config["rl"]["enabled"] is True
    assert paper["reliability"]["enabled"] is True
    assert paper["reliability_sidecar"]["contract"] == (
        "frozen_l217_value_hss_v1"
    )
    assert config["perception"]["dynamic_obstacle_tracker"]["enabled"] is True
    assert planner["probabilistic_obstacle_risk_enabled"] is True
    assert planner["noise_basis"] == "ar1:2.0"
    assert features["forward_passage"] is True
    assert config["real_robot_deployment"][
        "algorithm_profile"
    ] == "custom_ablation"


def test_residual_only_profile_uses_standard_mppi_shield_contract(tmp_path):
    config = _config(tmp_path)
    features = resolve_pi5_algorithm_features(
        enable_residual_learning=True,
    )
    apply_pi5_algorithm_features(config, features)

    planner = config["planner"]
    assert planner["optimizer"] == "standard"
    assert planner["prediction_mode"] == "icode_residual"
    assert planner["residual_safety_shield"]["enabled"] is True
    assert planner["residual_safety_shield"]["rl_hss_integration"] == ""
    assert config["rl"]["enabled"] is False
