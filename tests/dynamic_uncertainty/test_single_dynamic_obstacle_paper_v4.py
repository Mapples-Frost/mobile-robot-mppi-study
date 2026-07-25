from pathlib import Path
import csv
import hashlib
import json

import pytest
import yaml

from experiments.dynamic_uncertainty.analyze_single_dynamic_obstacle_paper_v4 import (
    _ablation_analysis,
    analyze,
    audit_formal_completion,
    continuous_contrast,
    exact_mcnemar_p,
    factorial_effect_vectors,
    holm_adjust,
    tango_matched_score_interval,
)
from experiments.dynamic_uncertainty.run_single_dynamic_obstacle_paper_v4 import (
    _canonical_sha256,
    _execution_manifest,
    build_qualification_registry,
    build_schedule,
    factor_separability_audit,
    validate_protocol,
    configure_arm,
)
from experiments.dynamic_uncertainty.run_rl_hss_stage4 import _mapping
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/research/single_dynamic_obstacle_paper_v4.yaml"


def test_v4_preflight_contract_and_qualification_schedule():
    _, protocol, id_config, ood_config, verified = validate_protocol(PROTOCOL)
    registry = build_qualification_registry(protocol, id_config, ood_config)
    blocks = build_schedule(registry, protocol, qualification=True)
    assert len(verified) == 7
    assert len(blocks) == 16
    assert sum(len(block["arm_sequence"]) for block in blocks) == 112
    assert all(block["certificate"]["conflict_windows_s"] for block in blocks)
    assert all(len(block["arm_sequence"]) == 7 for block in blocks)


def test_v4_learning_probability_factors_are_separable():
    _, protocol, _, _, _ = validate_protocol(PROTOCOL)
    assert factor_separability_audit(protocol)["status"] == "pass"


def test_v4_ablation_switches_and_cpu_budget_are_exact():
    _, protocol, _, _, _ = validate_protocol(PROTOCOL)
    base = load_yaml(ROOT / protocol["base_config"])
    stage3 = _mapping(ROOT / protocol["stage3_protocol"])
    stage4 = _mapping(ROOT / protocol["stage4_protocol"])
    block = {"split": "id", "seed": 730100001, "model_block": 0}
    configs = {
        arm: configure_arm(protocol, block, arm, base, stage3, stage4)
        for arm in protocol["arm_contracts"]
    }
    for arm, config in configs.items():
        iterations = int(config["planner"].get("paper_rl_driven", {}).get("iterations", 1))
        assert config["planner"]["num_samples"] * iterations == 600
        assert config["planner"]["device"] == "cpu"
    assert configs["A_full_ordinary_imm"]["perception"]["dynamic_obstacle_tracker"]["predictor_mode"] == "ordinary"
    assert configs["A_no_icode"]["planner"]["prediction_mode"] == "nominal"
    fixed = configs["A_fixed_hss"]["planner"]["paper_rl_driven"]
    assert fixed["reliability"]["enabled"] is True
    assert fixed["guided_fraction"] == 0.30
    assert fixed["reliability"]["low_guided_fraction"] == 0.30
    assert fixed["reliability"]["medium_guided_fraction"] == 0.30
    assert fixed["reliability"]["high_guided_fraction"] == 0.30
    for arm in ("B00_strong_nominal_mppi", "B10_learning_only"):
        assert configs[arm]["perception"]["dynamic_obstacle_tracker"]["enabled"] is False
        assert configs[arm]["planner"]["probabilistic_obstacle_risk_enabled"] is False


def test_tango_interval_symmetry_and_direction():
    balanced = tango_matched_score_interval(12, 12, 76)
    assert balanced[0] < 0.0 < balanced[1]
    positive = tango_matched_score_interval(24, 2, 74)
    swapped = tango_matched_score_interval(2, 24, 74)
    assert positive[0] > 0.0
    assert abs(positive[0] + swapped[1]) < 1e-9
    assert abs(positive[1] + swapped[0]) < 1e-9
    assert exact_mcnemar_p(24, 2) < 0.05


def test_v4_holm_and_factorial_directions_are_frozen():
    assert holm_adjust([0.01, 0.03, 0.04]) == pytest.approx([0.03, 0.06, 0.06])
    effects = factorial_effect_vectors({
        "B00_strong_nominal_mppi": [0.0, 0.0],
        "B10_learning_only": [1.0, 2.0],
        "B01_probability_only": [2.0, 1.0],
        "B11_full_proposed": [4.0, 5.0],
    })
    assert effects["learning_main"] == pytest.approx([1.5, 3.0])
    assert effects["probability_main"] == pytest.approx([2.5, 2.0])
    assert effects["learning_by_probability_interaction"] == pytest.approx([1.0, 2.0])


def test_v4_continuous_contrast_is_deterministic_and_favorable_direction():
    pytest.importorskip("scipy")
    first = continuous_contrast(
        [8.0, 9.0, 7.0], [10.0, 10.0, 10.0], False, 200, 741990047
    )
    second = continuous_contrast(
        [8.0, 9.0, 7.0], [10.0, 10.0, 10.0], False, 200, 741990047
    )
    assert first == second
    assert first["mean_treatment_minus_control"] == pytest.approx(-2.0)
    assert first["mean_favorable_effect"] == pytest.approx(2.0)


def test_v4_ablation_analysis_uses_only_frozen_ablation_blocks():
    pytest.importorskip("scipy")
    full = "B11_full_proposed"
    ablations = ["A_full_ordinary_imm", "A_no_icode", "A_fixed_hss"]

    def row(split, seed, arm, ablation_block, offset):
        return {
            "split": split,
            "seed": seed,
            "arm": arm,
            "ablation_block": ablation_block,
            "safe_success": offset > 0,
            "collision": offset < 0,
            "failure_penalized_steps": 100.0 - offset,
            "final_goal_distance": 1.0 - 0.1 * offset,
            "minimum_clearance": 0.5 + 0.1 * offset,
            "conflict_q05_clearance": 0.4 + 0.1 * offset,
        }

    rows = []
    # These two Full rows represent the 220 core-only seeds.  They must not be
    # requested as ablation pairs.
    rows.extend([
        row("id", 11, full, False, 1),
        row("ood", 12, full, False, 1),
    ])
    for split, seed in (("id", 1), ("ood", 2)):
        rows.append(row(split, seed, full, True, 1))
        rows.extend(row(split, seed, arm, True, 0) for arm in ablations)

    protocol = {
        "design": {"ablation_arms": ablations},
        "analysis": {"bootstrap_replicates": 20, "bootstrap_seed": 741990047},
    }
    result = _ablation_analysis(rows, protocol)

    assert len(result) == 54
    safe_success = [item for item in result if item["endpoint"] == "safe_success"]
    assert {item["n"] for item in safe_success if item["split"] == "pooled"} == {2}
    assert {item["n"] for item in safe_success if item["split"] == "id"} == {1}
    assert {item["n"] for item in safe_success if item["split"] == "ood"} == {1}


def test_v4_outcome_lock_rejects_incomplete_progress_before_metrics(tmp_path):
    root = tmp_path / "repo"
    protocol_path = root / "configs" / "research" / "protocol.yaml"
    registry_path = root / "configs" / "seeds" / "sealed.yaml"
    formal = tmp_path / "formal"
    protocol_path.parent.mkdir(parents=True)
    registry_path.parent.mkdir(parents=True)
    formal.mkdir()
    blocks = [
        {"split": "id", "seed": 1, "arm_sequence": ["B00"]},
        {"split": "ood", "seed": 2, "arm_sequence": ["B00"]},
    ]
    subset = {"id": [], "ood": []}
    registry = {
        "status": "sealed_before_formal_execution",
        "splits": {"id": [{"seed": 1}], "ood": [{"seed": 2}]},
        "ablation_subset": subset,
        "ablation_subset_sha256": _canonical_sha256(subset),
        "schedule": blocks,
        "schedule_sha256": _canonical_sha256(blocks),
    }
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    digest = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    registry_path.with_suffix(".yaml.sha256").write_text(
        "%s  sealed.yaml\n" % digest, encoding="ascii"
    )
    protocol_path.write_text(yaml.safe_dump({
        "formal_registry": "configs/seeds/sealed.yaml",
        "formal_output": str(formal),
        "design": {
            "core_seed_count_per_split": 1,
            "total_episode_jobs": 2,
        },
    }), encoding="utf-8")
    (formal / "progress.json").write_text(json.dumps({
        "status": "running",
        "completed_blocks": 0,
        "total_blocks": 2,
        "completed_episode_jobs": 0,
        "total_episode_jobs": 2,
        "outcomes_opened": False,
        "failures": [],
    }), encoding="utf-8")
    (formal / "schedule.json").write_text("{}", encoding="utf-8")
    (formal / "execution_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="outcomes remain locked"):
        audit_formal_completion(protocol_path)


def test_v4_execution_manifest_hashes_transitive_runtime_code():
    _, protocol, _, _, verified = validate_protocol(PROTOCOL)
    manifest = _execution_manifest(PROTOCOL, protocol, verified)
    normalized = {name.replace("\\", "/") for name in manifest["files"]}
    assert "src/mobile_robot_mppi/evaluation/metrics.py" in normalized
    assert "src/mobile_robot_mppi/runtime/experiment_runner.py" in normalized
    assert "experiments/dynamic_uncertainty/run_rl_hss_stage4.py" in normalized
    assert protocol["analysis_amendment"].replace("\\", "/") in normalized
    assert manifest["manifest_sha256"] == _canonical_sha256(manifest["files"])
    assert manifest["runtime_sha256"] == _canonical_sha256(manifest["runtime"])


def test_v4_complete_synthetic_matrix_unlocks_once_and_writes_tables(tmp_path):
    pytest.importorskip("scipy")
    root = tmp_path / "repo"
    protocol_path = root / "configs" / "research" / "protocol.yaml"
    registry_path = root / "configs" / "seeds" / "sealed.yaml"
    formal = tmp_path / "formal"
    output = tmp_path / "analysis"
    protocol_path.parent.mkdir(parents=True)
    registry_path.parent.mkdir(parents=True)
    formal.mkdir()
    core = [
        "B00_strong_nominal_mppi", "B10_learning_only",
        "B01_probability_only", "B11_full_proposed",
    ]
    ablations = ["A_full_ordinary_imm", "A_no_icode", "A_fixed_hss"]
    arms = core + ablations
    blocks = []
    splits = {"id": [], "ood": []}
    for order, (split, seed) in enumerate((("id", 1), ("ood", 2)), start=1):
        certificate = {
            "seed": seed,
            "split": split,
            "conflict_windows_s": [[0.0, 0.2]],
        }
        splits[split].append(certificate)
        blocks.append({
            "split": split,
            "seed": seed,
            "model_block": order - 1,
            "ablation_block": True,
            "arm_sequence": arms,
            "certificate": certificate,
            "block_order": order,
        })
    subset = {"id": [1], "ood": [2]}
    registry = {
        "status": "sealed_before_formal_execution",
        "splits": splits,
        "ablation_subset": subset,
        "ablation_subset_sha256": _canonical_sha256(subset),
        "schedule": blocks,
        "schedule_sha256": _canonical_sha256(blocks),
    }
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    registry_digest = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    registry_path.with_suffix(".yaml.sha256").write_text(
        "%s  sealed.yaml\n" % registry_digest, encoding="ascii"
    )
    protocol = {
        "study_id": "synthetic_v4",
        "formal_registry": "configs/seeds/sealed.yaml",
        "formal_output": str(formal),
        "analysis_output": str(output),
        "design": {
            "core_arms": core,
            "ablation_arms": ablations,
            "core_seed_count_per_split": 1,
            "total_episode_jobs": 14,
            "maximum_episode_steps": 400,
            "control_period_s": 0.1,
            "total_rollouts_per_decision": 600,
        },
        "arm_contracts": {
            arm: {"learning": arm not in (core[0], core[2])} for arm in arms
        },
        "conflict_certificate": {
            "robot_start_xy": [-4.6, -1.8],
            "robot_goal_xy": [4.2, -1.8],
            "ghost_speed_mps": 0.35,
        },
        "analysis": {
            "alpha": 0.05,
            "collision_noninferiority_margin": 0.02,
            "split_success_harm_margin": -0.05,
            "split_collision_harm_margin": 0.05,
            "bootstrap_replicates": 20,
            "bootstrap_seed": 741990047,
        },
    }
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    schedule_payload = {
        "scope": "paper_v4_formal",
        "blocks": blocks,
        "schedule_sha256": registry["schedule_sha256"],
    }
    (formal / "schedule.json").write_text(
        json.dumps(schedule_payload), encoding="utf-8"
    )
    (formal / "progress.json").write_text(json.dumps({
        "status": "complete",
        "completed_blocks": 2,
        "total_blocks": 2,
        "completed_episode_jobs": 14,
        "total_episode_jobs": 14,
        "outcomes_opened": False,
        "failures": [],
    }), encoding="utf-8")
    manifest = {
        "formal_experiment_started": True,
        "files": {},
        "manifest_sha256": _canonical_sha256({}),
        "runtime": {},
        "runtime_sha256": _canonical_sha256({}),
    }
    (formal / "execution_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for block in blocks:
        for arm_index, arm in enumerate(arms):
            run = formal / "runs" / block["split"] / ("seed_%d" % block["seed"]) / arm
            run.mkdir(parents=True)
            config = {
                "experiment": {
                    "seed": block["seed"],
                    "paper_v4_arm": arm,
                    "paper_v4_split": block["split"],
                    "paper_v4_model_block": block["model_block"],
                },
                "planner": {"num_samples": 600},
            }
            (run / "config_resolved.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
            (run / "paper_v4_job.json").write_text(json.dumps({
                "arm": arm,
                "block": block,
                "resolved_config_sha256": _canonical_sha256(config),
            }), encoding="utf-8")
            provenance = {"config_hash": "synthetic", "git_sha": "synthetic"}
            (run / "provenance.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
            success = arm != core[0]
            metrics = {
                "provenance": provenance,
                "steps": 3,
                "success": success,
                "collision": False,
                "time_to_goal_s": 0.2 if success else None,
                "final_goal_distance": 0.2 if success else 1.0,
                "trajectory_length": 0.2,
                "minimum_clearance": 0.4,
                "applied_control_jerk": 0.0,
                "stuck_steps": 0,
                "spin_steps": 0,
                "planner_compute_ms_p95": 50.0 + arm_index,
                "planner_deadline_miss_rate": 0.0,
                "paper_total_rollouts_mean": 600.0 if protocol["arm_contracts"][arm]["learning"] else 0.0,
            }
            (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            trajectory = [
                {
                    "time": 0.0, "x": -4.6, "y": -1.8, "applied_v": 0.1,
                    "clearance": 0.5, "temporal_scan_ttc_s": 2.0,
                    "temporal_scan_closing_rate_mps": 0.0,
                    "temporal_scan_risk_alpha": 0.0,
                },
                {
                    "time": 0.1, "x": -4.5, "y": -1.8, "applied_v": 0.1,
                    "clearance": 0.4, "temporal_scan_ttc_s": 2.0,
                    "temporal_scan_closing_rate_mps": 0.0,
                    "temporal_scan_risk_alpha": 0.0,
                },
                {
                    "time": 0.2, "x": -4.4, "y": -1.8, "applied_v": 0.1,
                    "clearance": 0.5, "temporal_scan_ttc_s": 2.0,
                    "temporal_scan_closing_rate_mps": 0.0,
                    "temporal_scan_risk_alpha": 0.0,
                },
            ]
            with (run / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(trajectory[0]))
                writer.writeheader()
                writer.writerows(trajectory)
    result = analyze(protocol_path)
    assert result["formal_outcomes_opened_after_complete_matrix"] is True
    assert (output / "primary_confirmatory_analysis.json").is_file()
    assert (output / "factorial_mechanism_effects.csv").is_file()
    assert (output / "analysis_bundle_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        analyze(protocol_path)
