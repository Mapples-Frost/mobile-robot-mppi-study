from pathlib import Path

from mobile_robot_mppi.core.config import load_yaml
from experiments.rl.summarize_conservative_schedule_training import (
    _selected_blocks_satisfy_noninferiority,
    _state_dict_exact,
)
from experiments.rl.summarize_conservative_schedule_deployment import (
    main as deployment_summary_main,
)
from experiments.rl.run_cross_layer_factorial import (
    _protected_previous_seeds,
)


ROOT = Path(__file__).resolve().parents[2]


def test_l78_changes_only_preregistered_conservative_training_schedule():
    l72 = load_yaml(
        ROOT / "configs/rl/sac_mppi_static_risk_balanced_l72.yaml"
    )
    l78 = load_yaml(
        ROOT / "configs/rl/sac_mppi_static_conservative_schedule_l78.yaml"
    )

    assert l72["rl"]["sac"]["policy_mode"] == "frozen_bc_correction"
    assert l78["rl"]["sac"]["policy_mode"] == "frozen_bc_correction"
    assert l72["rl"]["sac"]["actor_lr"] == 0.0001
    assert l78["rl"]["sac"]["actor_lr"] == 0.00005
    assert l72["rl"]["sac"]["correction_penalty_weight"] == 0.25
    assert l78["rl"]["sac"]["correction_penalty_weight"] == 1.0
    assert l72["rl"]["training"]["actor_update_after"] == 5000
    assert l78["rl"]["training"]["actor_update_after"] == 10000

    for section in ("planner", "plant", "sensors", "memory", "scene"):
        assert l78[section] == l72[section]
    for name in (
        "total_steps",
        "warmup_steps",
        "update_after",
        "evaluation_interval",
        "evaluation_episodes",
        "checkpoint_interval",
        "replay_capacity",
        "replay_sampling",
        "replay_success_fraction",
        "scene_configs",
        "validation_scene_configs",
        "checkpoint_selection",
    ):
        assert l78["rl"]["training"][name] == l72["rl"]["training"][name]


def test_l78_binds_independent_training_and_icode_blocks():
    expected = {
        20260774: "seed20261201",
        20260775: "seed20261202",
        20260776: "seed20261203",
    }
    for seed, icode_token in expected.items():
        config = load_yaml(
            ROOT
            / (
                "configs/rl/"
                "sac_mppi_static_conservative_schedule_l78_seed%d.yaml"
                % seed
            )
        )
        assert config["rl"]["training"]["seed"] == seed
        assert config["rl"]["training"]["validation_seed_base"] == 22300801
        assert config["rl"]["training"]["replay_sampling"] == (
            "scene_outcome_balanced"
        )
        assert icode_token in config["planner"]["checkpoint"]
        assert config["memory"]["enable"] is False


def test_l78_training_gate_requires_two_noninferior_nonzero_blocks():
    def block(index, step, gains, losses, collisions, distance):
        return {
            "block": index,
            "selected_nonzero": step > 0,
            "paired_success_gains": gains,
            "paired_success_losses": losses,
            "paired_collision_regressions": collisions,
            "paired_mean_goal_distance_improvement_m": distance,
        }

    rows = _selected_blocks_satisfy_noninferiority([
        block(0, 15000, 1, 0, 0, 0.01),
        block(1, 25000, 0, 0, 0, 0.006),
        block(2, 0, 0, 0, 0, 0.0),
    ])
    assert [row["eligible"] for row in rows] == [True, True, False]

    failed = _selected_blocks_satisfy_noninferiority([
        block(0, 15000, 1, 1, 0, 0.01),
        block(1, 25000, 0, 0, 0, -0.001),
        block(2, 0, 0, 0, 0, 0.0),
    ])
    assert not any(row["eligible"] for row in failed)


def test_l79_summarizer_rejects_non_l79_design(tmp_path):
    output = tmp_path / "summary"
    try:
        deployment_summary_main([
            "--config",
            str(
                ROOT
                / "configs/rl/training_internal_checkpoint_rule_l77.yaml"
            ),
            "--input-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(output),
        ])
    except ValueError as error:
        assert "expected an L79" in str(error)
    else:
        raise AssertionError("non-L79 design was accepted")


def test_l78_actor_delay_audit_compares_parameters_exactly():
    import torch

    reference = {"weight": torch.tensor([1.0, 2.0])}
    assert _state_dict_exact(reference, {"weight": reference["weight"].clone()})
    assert not _state_dict_exact(
        reference, {"weight": torch.tensor([1.0, 2.001])}
    )


def test_l79_locks_l78_selection_and_fresh_deployment_seeds():
    config = load_yaml(
        ROOT / "configs/rl/conservative_schedule_deployment_l79.yaml"
    )
    design = config["rl"]["cross_layer_factorial"]
    assert design["study_label"] == "L79"
    assert [row["selected_step"] for row in design["model_blocks"]] == [
        25000, 0, 25000
    ]
    assert design["development_episode_seeds"] == [
        22201101, 22201102, 22201103, 22201104, 22201105
    ]
    protected = _protected_previous_seeds(design["protected_config_paths"])
    assert not protected.intersection(design["development_episode_seeds"])
    assert not set(design["development_episode_seeds"]).intersection(
        design["sealed_confirmation_episode_seeds"]
    )
    block_one = design["model_blocks"][1]["condition_checkpoints"]
    assert block_one["complexity_bc_icode"] == block_one["gated_lcb_icode"]


def test_l80_changes_only_actor_group_aggregation_from_l78():
    l78 = load_yaml(
        ROOT / "configs/rl/sac_mppi_static_conservative_schedule_l78.yaml"
    )
    l80 = load_yaml(
        ROOT / "configs/rl/sac_mppi_static_group_robust_l80.yaml"
    )
    assert not l78["rl"]["sac"].get("actor_group_robust_enabled", False)
    assert l80["rl"]["sac"]["actor_group_robust_enabled"]
    assert l80["rl"]["sac"]["actor_group_robust_temperature"] == 0.1
    for section in ("planner", "plant", "sensors", "memory", "scene"):
        assert l80[section] == l78[section]
    for name in (
        "actor_lr",
        "critic_lr",
        "correction_penalty_weight",
        "correction_scale",
        "policy_mode",
    ):
        assert l80["rl"]["sac"][name] == l78["rl"]["sac"][name]
    for name in (
        "total_steps",
        "actor_update_after",
        "replay_sampling",
        "replay_success_fraction",
        "checkpoint_selection",
    ):
        assert l80["rl"]["training"][name] == l78["rl"]["training"][name]

    expected = {
        20260777: "seed20261201",
        20260778: "seed20261202",
        20260779: "seed20261203",
    }
    for seed, icode_token in expected.items():
        config = load_yaml(
            ROOT
            / (
                "configs/rl/sac_mppi_static_group_robust_l80_seed%d.yaml"
                % seed
            )
        )
        assert config["rl"]["training"]["seed"] == seed
        assert config["rl"]["training"]["validation_seed_base"] == 22310801
        assert icode_token in config["planner"]["checkpoint"]


def test_l81_changes_only_return_distribution_from_l78():
    l78 = load_yaml(
        ROOT / "configs/rl/sac_mppi_static_conservative_schedule_l78.yaml"
    )
    l81 = load_yaml(
        ROOT / "configs/rl/sac_mppi_static_quantile_cvar_l81.yaml"
    )
    sac = l81["rl"]["sac"]
    assert sac["critic_distribution"] == "quantile"
    assert sac["critic_num_quantiles"] == 25
    assert sac["critic_quantile_huber_kappa"] == 1.0
    assert sac["actor_cvar_fraction"] == 0.2
    assert not sac["actor_group_robust_enabled"]
    for section in ("planner", "plant", "sensors", "memory", "scene"):
        assert l81[section] == l78[section]
    for name in (
        "actor_lr",
        "critic_lr",
        "correction_penalty_weight",
        "correction_scale",
        "policy_mode",
    ):
        assert l81["rl"]["sac"][name] == l78["rl"]["sac"][name]
    for name in (
        "total_steps",
        "actor_update_after",
        "replay_sampling",
        "replay_success_fraction",
        "checkpoint_selection",
    ):
        assert l81["rl"]["training"][name] == l78["rl"]["training"][name]

    expected = {
        20260784: "seed20261201",
        20260785: "seed20261202",
        20260786: "seed20261203",
    }
    for seed, icode_token in expected.items():
        config = load_yaml(
            ROOT
            / (
                "configs/rl/sac_mppi_static_quantile_cvar_l81_seed%d.yaml"
                % seed
            )
        )
        assert config["rl"]["training"]["seed"] == seed
        assert config["rl"]["training"]["validation_seed_base"] == 22320801
        assert icode_token in config["planner"]["checkpoint"]
