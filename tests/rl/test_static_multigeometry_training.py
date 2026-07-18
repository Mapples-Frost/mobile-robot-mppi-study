import json
from pathlib import Path

from experiments.rl.summarize_static_multigeometry_training import summarize
from experiments.rl.run_static_multigeometry_deployment_selection import (
    _candidate_entries,
)
from experiments.rl.summarize_static_risk_balanced_confirmation import (
    _aggregate,
    _pair_rows,
)
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_l71_uses_fresh_deployment_selection_seeds_and_fixed_candidates():
    config = load_yaml(
        ROOT / "configs/rl/static_multigeometry_deployment_selection_l71.yaml"
    )
    design = config["rl"]["cross_layer_factorial"]
    selection = design["deployment_checkpoint_selection"]
    assert selection["candidate_steps"] == [
        5000, 10000, 15000, 20000, 25000, 30000
    ]
    assert len(design["model_blocks"]) == 3
    assert len(design["development_episode_seeds"]) == 3
    assert not set(design["development_episode_seeds"]).intersection(
        design["sealed_confirmation_episode_seeds"]
    )
    assert selection["minimum_nonzero_selected_blocks"] == 2
    assert selection["maximum_success_losses"] == 0
    assert selection["maximum_collision_regressions"] == 0


def test_l71_candidate_entries_bind_every_saved_training_checkpoint():
    config = load_yaml(
        ROOT / "configs/rl/static_multigeometry_deployment_selection_l71.yaml"
    )
    design = config["rl"]["cross_layer_factorial"]
    for block in design["model_blocks"]:
        entries = _candidate_entries(design, block)
        assert entries[0]["candidate_id"] == "base"
        assert entries[0]["condition"] == "complexity_bc_icode"
        assert [row["candidate_step"] for row in entries[1:]] == [
            5000, 10000, 15000, 20000, 25000, 30000
        ]
        assert all(row["condition"] == "gated_lcb_icode" for row in entries[1:])


def test_l72_changes_only_replay_strategy_and_fresh_randomness_from_l70():
    l70 = load_yaml(ROOT / "configs/rl/sac_mppi_static_multigeometry_l70.yaml")
    l72 = load_yaml(ROOT / "configs/rl/sac_mppi_static_risk_balanced_l72.yaml")
    assert l70["rl"]["training"]["replay_sampling"] == "scene_balanced"
    assert l72["rl"]["training"]["replay_sampling"] == "scene_outcome_balanced"
    assert l72["rl"]["training"]["replay_success_fraction"] == 0.5
    for section in ("planner", "plant", "sensors", "memory", "scene"):
        assert l72[section] == l70[section]
    for seed in (20260751, 20260752, 20260753):
        config = load_yaml(
            ROOT / ("configs/rl/sac_mppi_static_risk_balanced_l72_seed%d.yaml" % seed)
        )
        assert config["rl"]["training"]["seed"] == seed
        assert config["rl"]["training"]["replay_sampling"] == "scene_outcome_balanced"


def test_l73_uses_fresh_deployment_seeds_and_l72_checkpoints():
    config = load_yaml(
        ROOT / "configs/rl/static_risk_balanced_deployment_selection_l73.yaml"
    )
    design = config["rl"]["cross_layer_factorial"]
    assert design["study_label"] == "L73"
    assert design["development_episode_seeds"] == [
        22170801, 22170802, 22170803
    ]
    assert not set(design["development_episode_seeds"]).intersection(
        design["sealed_confirmation_episode_seeds"]
    )
    assert design["deployment_checkpoint_selection"]["candidate_steps"] == [
        5000, 10000, 15000, 20000, 25000, 30000
    ]
    for block in design["model_blocks"]:
        assert "l72_static_risk_balanced" in (
            design["deployment_checkpoint_selection"][
                "run_directory_template"
            ]
        )
        assert [row["candidate_step"] for row in _candidate_entries(
            design, block
        )] == [0, 5000, 10000, 15000, 20000, 25000, 30000]


def test_l74_locks_selected_checkpoint_and_binds_sealed_seeds():
    config = load_yaml(
        ROOT / "configs/rl/static_risk_balanced_sealed_confirmation_l74.yaml"
    )
    design = config["rl"]["cross_layer_factorial"]
    assert design["confirmation_mode"]
    assert design["study_label"] == "L74"
    assert design["development_episode_seeds"] == (
        design["sealed_confirmation_episode_seeds"]
    )
    assert design["deployment_checkpoint_selection"]["candidate_steps"] == [
        30000
    ]
    for block in design["model_blocks"]:
        assert [row["candidate_step"] for row in _candidate_entries(
            design, block
        )] == [0, 30000]


def test_confirmation_pairing_preserves_discordant_successes():
    common = {
        "model_block": 0,
        "scene": "corridor",
        "episode_seed": 7,
        "collision": False,
        "final_goal_distance": 1.0,
        "minimum_clearance": 0.2,
        "planner_compute_ms_mean": 10.0,
        "control_jerk": 0.1,
        "stuck_steps": 3,
        "spin_steps": 2,
        "safety_interventions": 4,
        "rl_correction_advantage_gate_alpha_mean": 0.0,
    }
    base = dict(common, candidate_id="base", success=False)
    sac = dict(
        common,
        candidate_id="step_000030000",
        success=True,
        final_goal_distance=0.2,
        rl_correction_advantage_gate_alpha_mean=0.4,
    )
    pairs = _pair_rows([base, sac], "step_000030000")
    summary = _aggregate(pairs, "overall")
    assert summary["success_gains"] == 1
    assert summary["success_losses"] == 0
    assert summary["net_success_gain"] == 1
    assert summary["mean_final_goal_distance_improvement_m"] == 0.8


def test_l70_audit_fails_closed_when_only_one_block_selects_nonzero(tmp_path):
    configs = [
        ROOT / ("configs/rl/sac_mppi_static_multigeometry_l70_seed%d.yaml" % seed)
        for seed in (20260741, 20260742, 20260743)
    ]
    runs = []
    for index, selected in enumerate((0, 20000, 0)):
        run = tmp_path / ("run_%d" % index)
        run.mkdir()
        (run / "training_summary.json").write_text(json.dumps({
            "global_step": 30000,
            "interrupted_episodes": 0,
            "replay_sampling": "scene_balanced",
            "replay_group_counts": {"0": 10, "1": 10, "2": 10},
            "checkpoint_selection": {"best_global_step": selected},
        }), encoding="utf-8")
        rows = ["global_step,scene,success,collision,goal_distance"]
        for step in range(0, 30001, 5000):
            for scene in ("a", "b", "c"):
                for _ in range(5):
                    rows.append("%d,%s,True,False,0.2" % (step, scene))
        (run / "validation_episodes.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        runs.append(run)
    result = summarize(configs, runs, minimum_nonzero_blocks=2)
    assert result["checks"]["artifact_integrity"], result
    assert result["nonzero_selected_blocks"] == 1
    assert not result["gate_passed"]
