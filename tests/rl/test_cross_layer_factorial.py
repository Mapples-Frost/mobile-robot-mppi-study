from pathlib import Path

import pytest

from mobile_robot_mppi.core.config import load_yaml
from experiments.rl.run_cross_layer_factorial import (
    CONDITION_SPECS,
    L29_CONDITIONS,
    _condition_config,
    _domain_entries,
    _parse_seeds,
    _parse_conditions,
    _prior_instance_key,
    _protected_previous_seeds,
    _scene_entries,
)
from experiments.rl.summarize_cross_layer_factorial import (
    _clean_fallback_audit,
    _development_gate,
    _nested_bootstrap,
)
from experiments.rl.summarize_temporal_gate_remediation import _eligibility
from experiments.rl.summarize_dynamic_variant_generalization import (
    PRIMARY as L31_PRIMARY,
    _development_gate as _l31_development_gate,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/cross_layer_factorial_l29.yaml"
L31_CONFIG = ROOT / "configs/rl/dynamic_variant_generalization_l31.yaml"
L51_CONFIG = ROOT / "configs/rl/icode_oracle_domain_gate_l51.yaml"
L59_CONFIG = ROOT / "configs/rl/icode_high_dynamic_confirmation_l59.yaml"


def _design():
    base = load_yaml(CONFIG)
    return base, base["rl"]["cross_layer_factorial"]


def test_l29_is_complete_nested_factorial_with_unique_scenes_and_domains():
    _, design = _design()
    assert set(design["conditions"]) == set(L29_CONDITIONS)
    assert len(design["conditions"]) == 6
    scenes = _scene_entries(design)
    domains = _domain_entries(design)
    assert {item["role"] for item in scenes} == {
        "clean", "static_blocking", "dynamic_crossing"
    }
    assert {item["role"] for item in domains} == {"seen", "unseen"}


@pytest.mark.parametrize("condition", tuple(CONDITION_SPECS))
def test_l29_condition_changes_only_declared_prediction_and_policy_factors(condition):
    base, design = _design()
    scene = _scene_entries(design)[0]
    domain = _domain_entries(design)[1]
    config = _condition_config(
        base,
        scene["path"],
        domain,
        condition,
        ROOT / "fake_rl.pt",
        ROOT / "fake_icode.pt",
        123,
        ROOT / "fake_mlp.pt",
    )
    spec = CONDITION_SPECS[condition]
    assert config["memory"]["enable"] is False
    assert config["experiment"]["seed"] == 123
    assert config["plant"]["actuator"]["command_delay"] == pytest.approx(0.10)
    assert config["planner"]["prediction_mode"] == (
        "%s_residual" % spec["residual"]
        if spec["residual"] in ("icode", "mlp") else "nominal"
    )
    assert config["planner"]["sampling_prior"] == (
        "goal_warm_start" if spec["policy"] == "traditional" else "rl"
    )
    assert config["rl"]["enabled"] is (spec["policy"] != "traditional")
    if spec["policy"] == "gated_lcb":
        assert config["rl"]["gate"]["mode"] == "complexity"
    elif spec["policy"] == "frozen_bc":
        assert config["rl"]["gate"]["mode"] == "none"
        assert config["rl"]["gate"][
            "correction_advantage_gate_mode"
        ] == "base"
    elif spec["policy"] == "complexity_bc":
        assert config["rl"]["gate"]["mode"] == "complexity"
        assert config["rl"]["gate"][
            "correction_advantage_gate_mode"
        ] == "base"
    elif spec["policy"] == "temporal_gated_lcb":
        assert config["rl"]["gate"]["mode"] == spec.get(
            "gate_mode", "complexity"
        )
        assert config["rl"]["gate"]["temporal_closing_enabled"] is True
    elif spec["policy"] == "lcb":
        assert config["rl"]["gate"]["mode"] == "none"
    temporal = config["perception"]["temporal_scan_guard"]
    assert temporal["enabled"] is bool(
        spec.get("temporal_scan_enabled", False)
    )
    assert temporal["safety_enabled"] is bool(
        spec.get("temporal_safety_enabled", False)
    )


def test_l60_mlp_condition_uses_only_the_declared_mlp_checkpoint():
    base, design = _design()
    config = _condition_config(
        base,
        _scene_entries(design)[0]["path"],
        _domain_entries(design)[0],
        "traditional_mlp",
        ROOT / "fake_rl.pt",
        ROOT / "fake_icode.pt",
        17,
        ROOT / "fake_mlp.pt",
    )
    assert config["planner"]["prediction_mode"] == "mlp_residual"
    assert config["planner"]["checkpoint"].endswith("fake_mlp.pt")
    assert config["planner"]["sampling_prior"] == "goal_warm_start"
    assert config["rl"]["enabled"] is False


def test_l60_mlp_condition_fails_closed_without_checkpoint():
    base, design = _design()
    with pytest.raises(ValueError, match="requires a mlp checkpoint"):
        _condition_config(
            base,
            _scene_entries(design)[0]["path"],
            _domain_entries(design)[0],
            "traditional_mlp",
            ROOT / "fake_rl.pt",
            ROOT / "fake_icode.pt",
            17,
        )


def test_domain_sensor_override_is_optional_and_takes_precedence():
    base, design = _design()
    raw = dict(design["physics_domains"][0])
    raw["sensor_override"] = {
        "pose_source": "ground_truth",
        "twist_source": "ground_truth",
        "latency": 0.05,
        "odom_noise_std": [0.01, 0.01, 0.02],
    }
    domain = _domain_entries({"physics_domains": [raw]})[0]
    config = _condition_config(
        base,
        _scene_entries(design)[0]["path"],
        domain,
        "traditional_nominal",
        ROOT / "fake_rl.pt",
        ROOT / "fake_icode.pt",
        17,
    )
    assert config["sensors"]["pose_source"] == "ground_truth"
    assert config["sensors"]["twist_source"] == "ground_truth"
    assert config["sensors"]["latency"] == pytest.approx(0.05)
    assert config["sensors"]["odom_noise_std"] == [0.01, 0.01, 0.02]


def test_domain_entries_preserve_empty_sensor_override_for_legacy_configs():
    _, design = _design()
    assert all(
        domain["sensor_override"] == {} for domain in _domain_entries(design)
    )


def test_l51_oracle_domain_gate_selects_only_the_frozen_active_domain():
    base = load_yaml(L51_CONFIG)
    design = base["rl"]["cross_layer_factorial"]
    scene = _scene_entries(design)[0]
    domains = {item["name"]: item for item in _domain_entries(design)}
    common = (
        base, scene["path"], None,
        "traditional_icode_oracle_domain_gate",
        ROOT / "fake_rl.pt", ROOT / "fake_icode.pt", 123,
    )
    matched = _condition_config(
        common[0], common[1], domains["combined_matched_delay"], *common[3:]
    )
    delayed = _condition_config(
        common[0], common[1], domains["combined_long_delay"], *common[3:]
    )
    assert matched["planner"]["prediction_mode"] == "nominal"
    assert "checkpoint" not in matched["planner"]
    assert delayed["planner"]["prediction_mode"] == "icode_residual"
    assert delayed["planner"]["checkpoint"].endswith("fake_icode.pt")
    assert matched["factorial"]["residual_factor"] == "icode_oracle_domain_gate"
    assert delayed["factorial"]["residual_factor"] == "icode_oracle_domain_gate"


def test_dynamic_crossing_config_has_one_prescribed_obstacle_but_no_planner_truth():
    config = load_yaml(ROOT / "configs/research/mujoco_dynamic_crossing.yaml")
    obstacle = config["scene"]["obstacles"][0]
    assert obstacle["motion"]["type"] == "linear_ping_pong"
    assert obstacle["motion"]["period_s"] == pytest.approx(16.0)
    assert "motion" not in config["planner"]
    assert "obstacles" not in config["planner"]


def test_l29_seed_parser_rejects_empty_and_duplicate_values():
    assert _parse_seeds("1,2", ()) == [1, 2]
    with pytest.raises(ValueError, match="nonempty and unique"):
        _parse_seeds("", ())
    with pytest.raises(ValueError, match="nonempty and unique"):
        _parse_seeds("1,1", ())


def test_stateful_prior_instances_are_isolated_by_experimental_condition():
    l32 = load_yaml(
        ROOT / "configs/rl/temporal_safety_remediation_l32.yaml"
    )
    conditions = l32["rl"]["cross_layer_factorial"]["conditions"]
    keys = [_prior_instance_key(condition) for condition in conditions]
    learned = [key for key in keys if key is not None]
    assert len(learned) == len(set(learned))
    assert keys[0] is None
    assert keys[1] is None
    assert (
        _prior_instance_key("robust_temporal_gated_lcb_icode")
        != _prior_instance_key("robust_temporal_gated_lcb_icode_safety")
    )


def test_l29_condition_parser_allows_declared_remediation_subset_only():
    configured = tuple(CONDITION_SPECS)
    assert _parse_conditions(
        "gated_lcb_nominal,gated_lcb_icode", configured
    ) == ["gated_lcb_nominal", "gated_lcb_icode"]
    with pytest.raises(ValueError, match="unknown"):
        _parse_conditions("oracle", configured)
    with pytest.raises(ValueError, match="absent"):
        _parse_conditions("lcb_nominal", ("gated_lcb_nominal",))


def test_l29_development_and_confirmation_seeds_do_not_reuse_l25_l28_protected_seeds():
    _, design = _design()
    selected = {
        int(value) for key in (
            "development_episode_seeds", "sealed_confirmation_episode_seeds"
        ) for value in design[key]
    }
    assert not selected.intersection(_protected_previous_seeds())


def test_l59_can_unlock_only_predeclared_sealed_seeds():
    design = load_yaml(L59_CONFIG)["rl"]["cross_layer_factorial"]
    selected = set(int(value) for value in design["development_episode_seeds"])
    sealed = set(int(value) for value in design["sealed_confirmation_episode_seeds"])
    protected = _protected_previous_seeds(design["protected_config_paths"])
    l58 = load_yaml(
        ROOT / "configs/rl/icode_high_dynamic_closed_loop_l58.yaml"
    )["rl"]["cross_layer_factorial"]

    assert design["confirmation_mode"] is True
    assert selected == sealed
    assert not selected.intersection(protected)
    assert set(l58["development_episode_seeds"]).issubset(protected)


def test_l31_is_fresh_complete_dynamic_variant_design():
    config = load_yaml(L31_CONFIG)
    design = config["rl"]["cross_layer_factorial"]
    assert len(_scene_entries(design)) == 4
    assert len(_domain_entries(design)) == 2
    assert len(design["conditions"]) == 5
    assert len(design["model_blocks"]) == 3
    assert len(design["development_episode_seeds"]) == 5
    assert 3 * 4 * 2 * 5 * 5 == 600
    selected = {
        int(value) for key in (
            "development_episode_seeds", "sealed_confirmation_episode_seeds"
        ) for value in design[key]
    }
    assert not selected.intersection(_protected_previous_seeds(
        design["protected_config_paths"]
    ))
    assert design["minimum_efficacy_checks_required"] == 3


@pytest.mark.parametrize(
    "name,period,radius",
    (
        ("mujoco_dynamic_crossing_reverse.yaml", 16.0, 0.20),
        ("mujoco_dynamic_crossing_fast.yaml", 10.0, 0.20),
        ("mujoco_dynamic_crossing_large_early.yaml", 20.0, 0.28),
    ),
)
def test_l31_dynamic_variants_are_physical_scan_visible_obstacles(
    name, period, radius
):
    config = load_yaml(ROOT / "configs/research" / name)
    obstacle = config["scene"]["obstacles"][0]
    assert obstacle["motion"]["type"] == "linear_ping_pong"
    assert obstacle["motion"]["period_s"] == pytest.approx(period)
    assert obstacle["radius"] == pytest.approx(radius)
    assert "motion" not in config["planner"]
    assert "obstacles" not in config["planner"]


def test_l31_gate_keeps_diagnostic_nominal_separate_from_primary_safety():
    config = load_yaml(L31_CONFIG)
    audit = {
        "expected_episode_keys": 600,
        "observed_episode_rows": 600,
        "missing_episode_keys": 0,
        "unexpected_episode_keys": 0,
        "duplicate_episode_keys": 0,
        "protected_seeds_used": [],
        "sealed_confirmation_seeds_used": [],
        "invalid_metric_rows": 0,
        "wrong_checkpoint_blocks": 0,
        "model_block_metadata_files": 3,
    }
    primary = [{
        "condition": L31_PRIMARY,
        "success": True,
        "collision": False,
    } for _ in range(120)]
    effects = []
    for index in range(120):
        common = {
            "model_block": index % 3,
            "episode_seed": index % 5,
            "success_difference": 1,
            "collision_difference": 0,
        }
        effects.append(dict(common, comparator="gated_lcb_icode"))
        # The diagnostic nominal method may itself collide; only the primary
        # candidate's safety is a required deployment check.
        effects.append(dict(
            common,
            comparator="temporal_gated_lcb_nominal",
            success_difference=0,
            collision_difference=-1,
        ))
    rankings = [{
        "primary_best_or_tied": True,
        "primary_successes": 15,
    } for _ in range(8)]
    gate = _l31_development_gate(
        config, audit, primary, effects, rankings
    )
    assert gate["passed"]
    primary[0]["collision"] = True
    assert not _l31_development_gate(
        config, audit, primary, effects, rankings
    )["passed"]


def _step(condition):
    return {
        "model_block": "0",
        "scene": "clean_goal_3_3",
        "scene_role": "clean",
        "physics_domain": "nominal_seen",
        "episode_seed": "1",
        "condition": condition,
        "step": "0",
        "goal_distance": "1.0",
        "collision": "0.0",
        "executed_v": "0.2",
        "executed_omega": "0.1",
        "safety_override": "0.0",
        "safety_reason": "front_clear",
        "rl_gate_alpha": "0.0",
    }


def test_clean_fallback_audit_requires_stepwise_identity():
    rows = []
    for traditional, gated in (
        ("traditional_nominal", "gated_lcb_nominal"),
        ("traditional_icode", "gated_lcb_icode"),
    ):
        rows.extend((_step(traditional), _step(gated)))
    audit = _clean_fallback_audit(rows)
    assert audit["exact"]
    assert audit["mismatch_fields"] == 0
    rows[-1]["executed_v"] = "0.21"
    assert not _clean_fallback_audit(rows)["exact"]


def test_clean_fallback_audit_ignores_nonfallback_lcb_trajectory_length():
    rows = []
    for traditional, gated in (
        ("traditional_nominal", "gated_lcb_nominal"),
        ("traditional_icode", "gated_lcb_icode"),
    ):
        rows.extend((_step(traditional), _step(gated)))
    lcb = _step("lcb_nominal")
    lcb["step"] = "99"
    rows.append(lcb)
    assert _clean_fallback_audit(rows)["exact"]


def test_nested_bootstrap_respects_constant_block_episode_effect():
    rows = [
        {"model_block": block, "episode_seed": episode, "value": 0.2}
        for block in range(3) for episode in range(5)
    ]
    assert _nested_bootstrap(rows, replicates=100) == pytest.approx([0.2, 0.2])


def test_l29_gate_separates_required_safety_from_efficacy_count():
    config, _ = _design()
    audit = {
        "missing_episode_keys": 0,
        "unexpected_episode_keys": 0,
        "duplicate_episode_keys": 0,
        "protected_seeds_used": [],
    }
    fallback = {"mismatch_fields": 0, "missing_step_pairs": 0}
    effects = []
    for index in range(6):
        effects.append({
            "effect": "gated_minus_traditional",
            "scene_role": "static_blocking",
            "physics_role": "seen",
            "model_block": index % 3,
            "episode_seed": index,
            "success_difference": 1,
            "collision_difference": 0,
            "final_distance_improvement_m": 0.1,
        })
    effects.append({
        "effect": "gated_minus_traditional",
        "scene_role": "dynamic_crossing",
        "physics_role": "unseen",
        "model_block": 0,
        "episode_seed": 0,
        "success_difference": 0,
        "collision_difference": 0,
        "final_distance_improvement_m": 0.0,
    })
    for block in range(3):
        for episode in range(5):
            effects.append({
                "effect": "icode_minus_nominal",
                "scene_role": "clean",
                "physics_role": "unseen",
                "model_block": block,
                "episode_seed": episode,
                "success_difference": 0,
                "collision_difference": 0,
                "final_distance_improvement_m": 0.1,
            })
    rankings = [{"combined_best_or_tied": index < 3} for index in range(4)]
    gate = _development_gate(config, audit, fallback, effects, rankings)
    assert gate["passed"]
    effects[0]["collision_difference"] = 1
    assert not _development_gate(config, audit, fallback, effects, rankings)["passed"]


def test_l29_gate_checks_always_on_lcb_collision_regression_too():
    config, _ = _design()
    audit = {
        "missing_episode_keys": 0,
        "unexpected_episode_keys": 0,
        "duplicate_episode_keys": 0,
        "protected_seeds_used": [],
    }
    fallback = {"mismatch_fields": 0, "missing_step_pairs": 0}
    effects = [{
        "effect": "lcb_minus_traditional",
        "scene_role": "dynamic_crossing",
        "physics_role": "seen",
        "model_block": 0,
        "episode_seed": 1,
        "success_difference": 0,
        "collision_difference": 1,
        "final_distance_improvement_m": 0.0,
    }]
    for block in range(3):
        for episode in range(5):
            effects.append({
                "effect": "icode_minus_nominal",
                "scene_role": "clean",
                "physics_role": "unseen",
                "model_block": block,
                "episode_seed": episode,
                "success_difference": 0,
                "collision_difference": 0,
                "final_distance_improvement_m": 0.1,
            })
    effects.extend({
        "effect": "gated_minus_traditional",
        "scene_role": "static_blocking",
        "physics_role": "seen",
        "model_block": index % 3,
        "episode_seed": index,
        "success_difference": 1,
        "collision_difference": 0,
        "final_distance_improvement_m": 0.1,
    } for index in range(6))
    effects.append({
        "effect": "gated_minus_traditional",
        "scene_role": "dynamic_crossing",
        "physics_role": "unseen",
        "model_block": 0,
        "episode_seed": 0,
        "success_difference": 0,
        "collision_difference": 0,
        "final_distance_improvement_m": 0.0,
    })
    rankings = [{"combined_best_or_tied": index < 3} for index in range(4)]
    gate = _development_gate(config, audit, fallback, effects, rankings)
    assert not gate["passed"]
    assert gate["estimands"]["collision_regressions_vs_traditional"] == 1


def test_l30_eligibility_requires_both_dynamic_strata_and_static_pairs():
    audit = {
        "missing_episode_keys": 0,
        "unexpected_episode_keys": 0,
        "duplicate_episode_keys": 0,
        "protected_seeds_used": [],
    }
    fallback = {"exact": True}
    paired = []
    for physics in ("seen", "unseen"):
        paired.append({
            "comparator": "always_lcb",
            "scene_role": "dynamic_crossing",
            "physics_role": physics,
            "success_difference": 0,
            "collision_difference": 0,
        })
    paired.append({
        "comparator": "spatial_gate",
        "scene_role": "static_blocking",
        "physics_role": "seen",
        "success_difference": -4,
        "collision_difference": 0,
    })
    rankings = [
        {"combined_best_or_tied": index < 3} for index in range(4)
    ]
    assert _eligibility(audit, fallback, paired, rankings)["passed"]
    incomplete = paired[:1] + paired[2:]
    assert not _eligibility(audit, fallback, incomplete, rankings)["passed"]
