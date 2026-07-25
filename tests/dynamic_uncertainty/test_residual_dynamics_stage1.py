import copy
import csv
import json
from pathlib import Path

import numpy as np

from experiments.dynamic_uncertainty.run_residual_dynamics_stage1 import (
    HORIZONS,
    _finalize_statistics,
    _first_true_index,
    _probe_early_stop_reason,
    _protocol_stage,
    build_schedule,
    configure_condition,
    prediction_sufficient_statistics,
)
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction


def _protocol():
    return {
        "design": {
            "development_episode_seeds": [11, 13, 17],
            "schedule_seed": 23,
        },
        "residual_model_blocks": [
            {"seed": 101, "checkpoint": "a.pt"},
            {"seed": 102, "checkpoint": "b.pt"},
            {"seed": 103, "checkpoint": "c.pt"},
        ],
    }


def test_schedule_uses_one_shared_nominal_and_three_residual_blocks():
    schedule = build_schedule(_protocol())
    assert len(schedule) == 12
    nominal = [row for row in schedule if row["condition"] == "nominal"]
    residual = [
        row for row in schedule if row["condition"] == "icode_residual"
    ]
    assert sorted(row["episode_seed"] for row in nominal) == [11, 13, 17]
    assert {
        (row["model_block"], row["episode_seed"]) for row in residual
    } == {
        (block, seed)
        for block in range(3)
        for seed in (11, 13, 17)
    }
    assert [row["experimental_key"] for row in schedule] == [
        row["experimental_key"] for row in build_schedule(_protocol())
    ]


def test_condition_switch_changes_only_registered_prediction_fields():
    base = {
        "experiment": {"name": "base", "seed": 1},
        "planner": {
            "seed": 1,
            "prediction_mode": "nominal",
            "probabilistic_obstacle_risk_enabled": True,
            "horizon": 36,
            "num_samples": 600,
        },
    }
    original = copy.deepcopy(base)
    nominal_job = {
        "condition": "nominal",
        "episode_seed": 11,
        "model_block": -1,
    }
    residual_job = {
        "condition": "icode_residual",
        "episode_seed": 11,
        "model_block": 0,
    }
    nominal = configure_condition(base, nominal_job, _protocol())
    residual = configure_condition(base, residual_job, _protocol())
    assert base == original
    assert nominal["planner"]["prediction_mode"] == "nominal"
    assert "checkpoint" not in nominal["planner"]
    assert residual["planner"]["prediction_mode"] == "icode_residual"
    assert residual["planner"]["checkpoint"] == "a.pt"
    assert residual["planner"]["horizon"] == nominal["planner"]["horizon"]
    assert residual["planner"]["num_samples"] == nominal["planner"]["num_samples"]
    assert residual["planner"][
        "probabilistic_obstacle_risk_enabled"
    ] is True


def test_registered_canonicalization_is_applied_only_to_residual_condition():
    protocol = _protocol()
    protocol["residual_input_canonicalization"] = {
        "enabled": True,
        "state_names": ["x", "y"],
        "values": [3.25, 0.31],
    }
    base = {
        "experiment": {"name": "base", "seed": 1},
        "planner": {
            "seed": 1,
            "prediction_mode": "nominal",
            "probabilistic_obstacle_risk_enabled": True,
        },
    }
    nominal = configure_condition(
        base,
        {"condition": "nominal", "episode_seed": 11, "model_block": -1},
        protocol,
    )
    residual = configure_condition(
        base,
        {
            "condition": "icode_residual",
            "episode_seed": 11,
            "model_block": 0,
        },
        protocol,
    )
    assert "residual_state_canonicalization" not in nominal["planner"]
    assert residual["planner"]["residual_state_canonicalization"] == {
        "enabled": True,
        "state_names": ["x", "y"],
        "values": [3.25, 0.31],
    }


def test_registered_stall_guard_is_applied_only_to_residual_condition():
    protocol = _protocol()
    protocol["residual_stall_guard"] = {
        "enabled": True,
        "speed_threshold_mps": 0.02,
        "maximum_low_risk_probability": 0.05,
        "consecutive_steps": 10,
        "goal_exclusion_distance_m": 0.45,
        "latch_for_episode": True,
    }
    base = {
        "experiment": {"name": "base", "seed": 1},
        "planner": {
            "seed": 1,
            "prediction_mode": "nominal",
            "probabilistic_obstacle_risk_enabled": True,
            "residual_stall_guard": {"enabled": False},
        },
    }
    nominal = configure_condition(
        base,
        {"condition": "nominal", "episode_seed": 11, "model_block": -1},
        protocol,
    )
    residual = configure_condition(
        base,
        {
            "condition": "icode_residual",
            "episode_seed": 11,
            "model_block": 0,
        },
        protocol,
    )
    assert "residual_stall_guard" not in nominal["planner"]
    assert residual["planner"]["residual_stall_guard"] == {
        key: protocol["residual_stall_guard"][key]
        for key in (
            "enabled",
            "speed_threshold_mps",
            "maximum_low_risk_probability",
            "consecutive_steps",
            "goal_exclusion_distance_m",
            "latch_for_episode",
        )
    }


def test_registered_reliability_gate_is_applied_only_to_residual_condition():
    protocol = _protocol()
    protocol["residual_reliability_gate"] = {
        "enabled": True,
        "state_names": ["v", "omega"],
        "state_scales": [0.25, 0.6],
        "forgetting_factor": 0.95,
        "minimum_samples": 8,
        "confidence_z": 0.0,
        "off_threshold": -0.05,
        "on_threshold": 0.05,
        "rise_rate": 0.25,
        "fall_rate": 0.5,
        "actuation_delay_context": {"enabled": False},
    }
    base = {
        "experiment": {"name": "base", "seed": 1},
        "planner": {
            "seed": 1,
            "prediction_mode": "nominal",
            "probabilistic_obstacle_risk_enabled": True,
            "residual_reliability_gate": {"enabled": False},
        },
    }
    nominal = configure_condition(
        base,
        {"condition": "nominal", "episode_seed": 11, "model_block": -1},
        protocol,
    )
    residual = configure_condition(
        base,
        {
            "condition": "icode_residual",
            "episode_seed": 11,
            "model_block": 0,
        },
        protocol,
    )
    assert "residual_reliability_gate" not in nominal["planner"]
    assert residual["planner"]["residual_reliability_gate"] == (
        protocol["residual_reliability_gate"]
    )


def test_registered_component_mask_is_applied_only_to_residual_condition():
    protocol = _protocol()
    protocol["residual_component_mask"] = [0.0, 0.0, 0.0, 1.0, 1.0]
    base = {
        "experiment": {"name": "base", "seed": 1},
        "planner": {
            "seed": 1,
            "prediction_mode": "nominal",
            "probabilistic_obstacle_risk_enabled": True,
            "residual_component_mask": [1.0] * 5,
        },
    }
    nominal = configure_condition(
        base,
        {"condition": "nominal", "episode_seed": 11, "model_block": -1},
        protocol,
    )
    residual = configure_condition(
        base,
        {
            "condition": "icode_residual",
            "episode_seed": 11,
            "model_block": 0,
        },
        protocol,
    )
    assert "residual_component_mask" not in nominal["planner"]
    assert residual["planner"]["residual_component_mask"] == [
        0.0, 0.0, 0.0, 1.0, 1.0
    ]


def test_registered_safety_shield_is_applied_only_to_residual_condition():
    protocol = _protocol()
    protocol["residual_safety_shield"] = {
        "enabled": True,
        "maximum_nominal_risk_increase": 0.0,
        "maximum_position_deviation_m": 0.10,
        "maximum_nominal_progress_regression_m": 0.02,
        "require_residual_hard_safe": True,
        "require_nominal_hard_safe": True,
    }
    base = {
        "experiment": {"name": "base", "seed": 1},
        "planner": {
            "seed": 1,
            "prediction_mode": "nominal",
            "probabilistic_obstacle_risk_enabled": True,
            "residual_safety_shield": {"enabled": False},
        },
    }
    nominal = configure_condition(
        base,
        {"condition": "nominal", "episode_seed": 11, "model_block": -1},
        protocol,
    )
    residual = configure_condition(
        base,
        {
            "condition": "icode_residual",
            "episode_seed": 11,
            "model_block": 0,
        },
        protocol,
    )
    assert "residual_safety_shield" not in nominal["planner"]
    assert residual["planner"]["residual_safety_shield"] == (
        protocol["residual_safety_shield"]
    )


def test_probe_stops_on_completed_residual_collision(tmp_path):
    protocol = _protocol()
    protocol["design"]["stop_on_first_residual_gate_failure"] = True
    protocol["gate"] = {
        "maximum_enabled_planner_p95_compute_ms": 150.0
    }
    schedule = build_schedule(protocol)
    job = next(
        row for row in schedule if row["condition"] == "icode_residual"
    )
    from experiments.dynamic_uncertainty.run_residual_dynamics_stage1 import (
        _run_dir,
    )

    run_dir = _run_dir(Path(tmp_path), job)
    run_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text("ok: true\n")
    (run_dir / "trajectory.csv").write_text("x\n0\n")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "collision": True,
                "planner_compute_ms_p95": 120.0,
            }
        )
    )
    reason = _probe_early_stop_reason(protocol, schedule, Path(tmp_path))
    assert reason["reasons"] == ["residual_collision"]
    assert reason["experimental_key"] == job["experimental_key"]


def test_collision_only_probe_does_not_stop_on_compute_miss(tmp_path):
    protocol = _protocol()
    protocol["design"].update(
        {
            "stop_on_first_residual_gate_failure": True,
            "early_stop_conditions": ["residual_collision"],
        }
    )
    protocol["gate"] = {
        "maximum_enabled_planner_p95_compute_ms": 150.0
    }
    schedule = build_schedule(protocol)
    job = next(
        row for row in schedule if row["condition"] == "icode_residual"
    )
    from experiments.dynamic_uncertainty.run_residual_dynamics_stage1 import (
        _run_dir,
    )

    run_dir = _run_dir(Path(tmp_path), job)
    run_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text("ok: true\n")
    (run_dir / "trajectory.csv").write_text("x\n0\n")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "collision": False,
                "planner_compute_ms_p95": 151.0,
            }
        )
    )
    assert _probe_early_stop_reason(
        protocol, schedule, Path(tmp_path)
    ) is None


def test_first_true_index_handles_empty_and_first_match():
    assert _first_true_index([False, False]) is None
    assert _first_true_index([False, True, True]) == 1


def test_protocol_stage_distinguishes_task_aware_stage2_artifacts():
    assert (
        _protocol_stage(
            {
                "preregistration": (
                    "docs/RESIDUAL_DYNAMICS_STAGE2_TASK_AWARE_"
                    "PREREGISTRATION.md"
                )
            }
        )
        == "residual_dynamics_stage2_task_aware_development"
    )
    assert (
        _protocol_stage({"preregistration": "docs/STAGE1.md"})
        == "residual_dynamics_stage1_development"
    )
    assert (
        _protocol_stage(
            {
                "preregistration": (
                    "docs/RESIDUAL_RUNTIME_STAGE3_CLOSED_LOOP_"
                    "PREREGISTRATION.md"
                )
            }
        )
        == "residual_runtime_stage3_development"
    )


def test_prediction_audit_is_exact_for_generated_nominal_trajectory(tmp_path):
    dt = 0.1
    dynamics = DynamicUnicyclePrediction(0.18, 0.12)
    controls = np.tile(np.asarray([[0.2, 0.1]]), (50, 1))
    states = [np.zeros(5, dtype=np.float64)]
    from experiments.dynamic_uncertainty.run_residual_dynamics_stage1 import (
        _rk4_step,
    )

    for control in controls:
        states.append(_rk4_step(dynamics, states[-1], control, dt))
    path = Path(tmp_path) / "trajectory.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "x",
                "y",
                "theta",
                "v",
                "omega",
                "applied_v",
                "applied_omega",
            ],
        )
        writer.writeheader()
        for state, control in zip(states[1:], controls):
            writer.writerow(
                {
                    "x": state[0],
                    "y": state[1],
                    "theta": state[2],
                    "v": state[3],
                    "omega": state[4],
                    "applied_v": control[0],
                    "applied_omega": control[1],
                }
            )
    config = {
        "experiment": {
            "initial_state": [0.0] * 5,
            "control_dt": dt,
        }
    }
    statistics = prediction_sufficient_statistics(config, path, dynamics)
    finalized = _finalize_statistics(statistics)
    for horizon in HORIZONS:
        assert (
            finalized["horizons"][str(horizon)]["overall_state_rmse"]
            < 1.0e-12
        )
