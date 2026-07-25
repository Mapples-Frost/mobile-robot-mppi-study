"""Run residual-dynamics Stage 1 under the frozen Amendment 17 environment.

The online comparison changes only the prediction model. Simulator state is
used after each run for model-error and environment-contract audits; it is
never supplied to the controller.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.learning.models import PlatformResidualDynamics
from mobile_robot_mppi.learning.models import (
    CanonicalizedStateResidualDynamics,
    ResidualComponentMaskedDynamics,
)
from mobile_robot_mppi.obstacles.artifacts import (
    environment_manifest,
    repository_provenance,
    sha256_file,
    write_csv,
    write_json,
    write_yaml,
)
from mobile_robot_mppi.planning.dynamics import (
    DynamicUnicyclePrediction,
    ResidualPrediction,
)
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner

from experiments.dynamic_uncertainty.run_online_v3_tracking_mppi_smoke import (
    _audit_episode,
    _obstacle_trajectory,
)

DEFAULT_PROTOCOL = (
    ROOT
    / "configs/research/dynamic_uncertainty_residual_stage1_protocol.yaml"
)
HORIZONS = (1, 5, 10, 20, 36)
STATE_NAMES = ("x", "y", "theta", "v", "omega")


def _load_yaml_mapping(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping")
    return value


def _resolve(value):
    path = Path(str(value))
    return (path if path.is_absolute() else ROOT / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol(protocol, base_config):
    design = protocol["design"]
    frozen = protocol["frozen"]
    revision = int(protocol.get("protocol_revision", 0))
    runtime_stage3 = "residual_runtime_stage3" in str(
        protocol.get("preregistration", "")
    ).lower()
    if revision not in (2, 3, 4, 5, 6, 7, 8, 9) and not (
        runtime_stage3 and revision == 1
    ):
        raise ValueError("unsupported Stage 1 protocol revision")
    if tuple(design["conditions"]) != ("nominal", "icode_residual"):
        raise ValueError("Stage 1 conditions must be nominal and icode_residual")
    if bool(design.get("sealed_seeds_authorized", True)):
        raise ValueError("sealed seeds are not authorized in Stage 1")
    if not bool(design.get("common_random_numbers", False)):
        raise ValueError("Stage 1 requires common random numbers")
    if not bool(design.get("shared_nominal_reference_per_episode_seed", False)):
        raise ValueError("Stage 1 requires a shared nominal reference")
    seeds = [int(value) for value in design["development_episode_seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("development episode seeds must be non-empty and unique")

    planner = base_config["planner"]
    if int(base_config["design_amendment"]["version"]) != int(
        frozen["environment_amendment"]
    ):
        raise ValueError("base config does not match the frozen amendment")
    if str(planner.get("prediction_mode", "nominal")) != "nominal":
        raise ValueError("base config must use nominal prediction")
    if not bool(planner["probabilistic_obstacle_risk_enabled"]):
        raise ValueError("Stage 1 requires the risk-enabled controller")
    if int(planner["horizon"]) != int(frozen["mppi_horizon"]):
        raise ValueError("MPPI horizon differs from the frozen protocol")
    if int(planner["num_samples"]) != int(frozen["mppi_candidates"]):
        raise ValueError("MPPI sample budget differs from the frozen protocol")
    if not math.isclose(
        float(planner["probabilistic_obstacle_hard_threshold"]),
        float(frozen["hard_collision_probability_threshold"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("collision-probability threshold differs")
    if bool(base_config.get("rl", {}).get("enabled", False)):
        raise ValueError("RL must remain disabled in residual Stage 1")

    stall_guard = dict(protocol.get("residual_stall_guard", {}))
    if bool(stall_guard.get("enabled", False)):
        required = (
            "speed_threshold_mps",
            "maximum_low_risk_probability",
            "consecutive_steps",
            "goal_exclusion_distance_m",
            "latch_for_episode",
        )
        missing = [name for name in required if name not in stall_guard]
        if missing:
            raise ValueError(
                "residual stall guard is missing fields: %s" % missing
            )
        numeric = np.asarray(
            (
                stall_guard["speed_threshold_mps"],
                stall_guard["maximum_low_risk_probability"],
                stall_guard["goal_exclusion_distance_m"],
            ),
            dtype=np.float64,
        )
        if (
            not np.isfinite(numeric).all()
            or numeric[0] < 0.0
            or not 0.0 <= numeric[1] <= 1.0
            or numeric[2] <= 0.0
            or int(stall_guard["consecutive_steps"]) <= 0
            or not bool(stall_guard["latch_for_episode"])
        ):
            raise ValueError("residual stall guard thresholds are invalid")
    reliability_gate = dict(
        protocol.get("residual_reliability_gate", {})
    )
    if bool(reliability_gate.get("enabled", False)):
        required = (
            "state_names",
            "state_scales",
            "forgetting_factor",
            "minimum_samples",
            "confidence_z",
            "off_threshold",
            "on_threshold",
            "rise_rate",
            "fall_rate",
        )
        missing = [name for name in required if name not in reliability_gate]
        if missing:
            raise ValueError(
                "residual reliability gate is missing fields: %s" % missing
            )
        if tuple(reliability_gate["state_names"]) != ("v", "omega"):
            raise ValueError(
                "Stage 1 reliability evidence must use only v and omega"
            )
        scales = np.asarray(
            reliability_gate["state_scales"], dtype=np.float64
        )
        if (
            scales.shape != (2,)
            or not np.isfinite(scales).all()
            or np.any(scales <= 0.0)
        ):
            raise ValueError(
                "reliability state scales must be finite and positive"
            )
        if float(reliability_gate["on_threshold"]) <= float(
            reliability_gate["off_threshold"]
        ):
            raise ValueError(
                "reliability on threshold must exceed off threshold"
            )
    component_mask = protocol.get("residual_component_mask")
    if component_mask is not None:
        values = np.asarray(component_mask, dtype=np.float64)
        if not np.array_equal(
            values, np.asarray((0.0, 0.0, 0.0, 1.0, 1.0))
        ):
            raise ValueError(
                "Stage 1 structure-preserving mask must retain only v/omega"
            )
    safety_shield = dict(protocol.get("residual_safety_shield", {}))
    if bool(safety_shield.get("enabled", False)):
        required = (
            "maximum_nominal_risk_increase",
            "maximum_position_deviation_m",
            "maximum_nominal_progress_regression_m",
            "require_residual_hard_safe",
            "require_nominal_hard_safe",
        )
        missing = [name for name in required if name not in safety_shield]
        if missing:
            raise ValueError(
                "residual safety shield is missing fields: %s" % missing
            )
        if (
            float(safety_shield["maximum_nominal_risk_increase"]) != 0.0
            or not math.isclose(
                float(safety_shield["maximum_position_deviation_m"]),
                float(planner["probabilistic_obstacle_safety_margin"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not bool(safety_shield["require_residual_hard_safe"])
            or not bool(safety_shield["require_nominal_hard_safe"])
        ):
            raise ValueError(
                "Stage 1 shield requires zero risk increase, the frozen "
                "safety-margin tube and both hard-safety checks"
            )
        authority_names = tuple(
            safety_shield.get("action_authority_names", ())
        )
        if authority_names and authority_names != ("v_cmd",):
            raise ValueError(
                "Stage 1 route-preserving shield may authorize only v_cmd"
            )

    checkpoint_rows = []
    checkpoint_seeds = set()
    for block_index, block in enumerate(protocol["residual_model_blocks"]):
        checkpoint_seed = int(block["seed"])
        if checkpoint_seed in checkpoint_seeds:
            raise ValueError("residual checkpoint seeds must be unique")
        checkpoint_seeds.add(checkpoint_seed)
        path = _resolve(block["checkpoint"])
        if not path.is_file():
            raise FileNotFoundError("residual checkpoint missing: %s" % path)
        actual_hash = _sha256(path)
        expected_hash = str(block["sha256"]).lower()
        if actual_hash != expected_hash:
            raise ValueError("residual checkpoint hash mismatch: %s" % path)
        model = PlatformResidualDynamics.from_checkpoint(
            path, device="cpu", use_torchscript=True
        )
        probe = np.asarray(
            model.derivative(np.zeros(5), np.zeros(2)), dtype=np.float64
        )
        if (
            model.model.model_type != "icode_residual"
            or model.state_dim != 5
            or model.control_dim != 2
            or probe.shape != (5,)
            or not np.isfinite(probe).all()
        ):
            raise ValueError("checkpoint compatibility preflight failed")
        checkpoint_rows.append(
            {
                "model_block": int(block_index),
                "checkpoint_seed": checkpoint_seed,
                "checkpoint": str(path.relative_to(ROOT)),
                "sha256": actual_hash,
                "zero_probe": probe.tolist(),
                "feature_mean_xy": (
                    model.model.feature_mean[:2].detach().cpu().tolist()
                ),
            }
        )
    if len(checkpoint_rows) != 3:
        raise ValueError("Stage 1 requires exactly three model blocks")

    canonicalization = dict(
        protocol.get("residual_input_canonicalization", {})
    )
    if bool(canonicalization.get("enabled", False)):
        if tuple(canonicalization.get("state_names", ())) != ("x", "y"):
            raise ValueError("Stage 1 canonicalization may change only x and y")
        values = np.asarray(
            canonicalization.get("values", ()), dtype=np.float64
        )
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError("canonical x/y values must be two finite values")
        for row in checkpoint_rows:
            if not np.allclose(
                values,
                np.asarray(row["feature_mean_xy"], dtype=np.float64),
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise ValueError(
                    "canonical x/y values must equal checkpoint feature means"
                )
    return {
        "development_episode_seeds": seeds,
        "checkpoint_blocks": checkpoint_rows,
        "sealed_seeds_opened": False,
        "base_environment_amendment": int(
            base_config["design_amendment"]["version"]
        ),
        "residual_input_canonicalization": canonicalization,
    }


def build_schedule(protocol):
    design = protocol["design"]
    seeds = [int(value) for value in design["development_episode_seeds"]]
    jobs = [
        {
            "condition": "nominal",
            "episode_seed": seed,
            "model_block": -1,
            "checkpoint_seed": -1,
        }
        for seed in seeds
    ]
    for block_index, block in enumerate(protocol["residual_model_blocks"]):
        for seed in seeds:
            jobs.append(
                {
                    "condition": "icode_residual",
                    "episode_seed": seed,
                    "model_block": int(block_index),
                    "checkpoint_seed": int(block["seed"]),
                }
            )
    order = np.random.RandomState(int(design["schedule_seed"])).permutation(
        len(jobs)
    )
    scheduled = []
    for run_order, job_index in enumerate(order):
        row = dict(jobs[int(job_index)])
        row["run_order"] = int(run_order)
        row["experimental_key"] = (
            "nominal::seed%d" % row["episode_seed"]
            if row["condition"] == "nominal"
            else "block%d::seed%d::icode_residual"
            % (row["model_block"], row["episode_seed"])
        )
        scheduled.append(row)
    return scheduled


def configure_condition(base_config, job, protocol):
    config = deepcopy(base_config)
    planner = config["planner"]
    seed = int(job["episode_seed"])
    config["experiment"]["seed"] = seed
    planner["seed"] = seed
    planner["probabilistic_obstacle_risk_enabled"] = True
    planner.pop("checkpoints", None)
    planner.pop("residual_stall_guard", None)
    planner.pop("residual_reliability_gate", None)
    planner.pop("residual_component_mask", None)
    planner.pop("residual_safety_shield", None)
    if job["condition"] == "nominal":
        planner["prediction_mode"] = "nominal"
        planner.pop("checkpoint", None)
        suffix = "nominal"
    elif job["condition"] == "icode_residual":
        block = protocol["residual_model_blocks"][int(job["model_block"])]
        planner["prediction_mode"] = "icode_residual"
        planner["checkpoint"] = str(block["checkpoint"])
        planner["residual_torch_num_threads"] = 1
        planner["residual_torchscript"] = True
        canonicalization = dict(
            protocol.get("residual_input_canonicalization", {})
        )
        if bool(canonicalization.get("enabled", False)):
            planner["residual_state_canonicalization"] = {
                "enabled": True,
                "state_names": list(canonicalization["state_names"]),
                "values": [
                    float(value) for value in canonicalization["values"]
                ],
            }
        else:
            planner.pop("residual_state_canonicalization", None)
        component_mask = protocol.get("residual_component_mask")
        if component_mask is not None:
            planner["residual_component_mask"] = [
                float(value) for value in component_mask
            ]
        stall_guard = dict(protocol.get("residual_stall_guard", {}))
        if bool(stall_guard.get("enabled", False)):
            planner["residual_stall_guard"] = {
                key: stall_guard[key]
                for key in (
                    "enabled",
                    "speed_threshold_mps",
                    "maximum_low_risk_probability",
                    "consecutive_steps",
                    "goal_exclusion_distance_m",
                    "latch_for_episode",
                )
            }
        reliability_gate = dict(
            protocol.get("residual_reliability_gate", {})
        )
        if bool(reliability_gate.get("enabled", False)):
            planner["residual_reliability_gate"] = deepcopy(
                reliability_gate
            )
        safety_shield = dict(
            protocol.get("residual_safety_shield", {})
        )
        if bool(safety_shield.get("enabled", False)):
            planner["residual_safety_shield"] = deepcopy(
                safety_shield
            )
        suffix = "icode_block%d" % int(job["model_block"])
    else:
        raise ValueError("unknown Stage 1 condition")
    config["experiment"]["name"] = "residual_stage1_seed%d_%s" % (
        seed,
        suffix,
    )
    return config


def _run_dir(output_dir, job):
    seed = int(job["episode_seed"])
    if job["condition"] == "nominal":
        return output_dir / "runs" / "nominal" / ("seed_%d" % seed)
    return (
        output_dir
        / "runs"
        / "icode_residual"
        / ("block_%d" % int(job["model_block"]))
        / ("seed_%d" % seed)
    )


def _complete_run(run_dir):
    required = ("config_resolved.yaml", "metrics.json", "trajectory.csv")
    return all((run_dir / name).is_file() for name in required)


def _probe_early_stop_reason(protocol, schedule, output_dir):
    if not bool(
        protocol["design"].get(
            "stop_on_first_residual_gate_failure",
            str(protocol.get("status", "")) == "preregistered_probe",
        )
    ):
        return None
    threshold = float(
        protocol["gate"]["maximum_enabled_planner_p95_compute_ms"]
    )
    enabled_conditions = set(
        protocol["design"].get(
            "early_stop_conditions",
            ("residual_collision", "planner_p95_above_threshold"),
        )
    )
    for job in schedule:
        if job["condition"] != "icode_residual":
            continue
        run_dir = _run_dir(output_dir, job)
        if not _complete_run(run_dir):
            continue
        metrics = json.loads(
            (run_dir / "metrics.json").read_text(encoding="utf-8")
        )
        reasons = []
        if (
            "residual_collision" in enabled_conditions
            and bool(metrics["collision"])
        ):
            reasons.append("residual_collision")
        if (
            "planner_p95_above_threshold" in enabled_conditions
            and float(metrics["planner_compute_ms_p95"]) > threshold
        ):
            reasons.append("planner_p95_above_threshold")
        if reasons:
            return {
                "experimental_key": str(job["experimental_key"]),
                "episode_seed": int(job["episode_seed"]),
                "model_block": int(job["model_block"]),
                "checkpoint_seed": int(job["checkpoint_seed"]),
                "reasons": reasons,
                "collision": bool(metrics["collision"]),
                "planner_p95_compute_ms": float(
                    metrics["planner_compute_ms_p95"]
                ),
                "planner_p95_threshold_ms": threshold,
            }
    return None


def _read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _episode_record(base_config, job, run_dir, trajectory):
    record = _audit_episode(
        base_config,
        int(job["episode_seed"]),
        "risk_enabled",
        int(job["run_order"]),
        run_dir,
        trajectory,
    )
    record["experimental_key"] = str(job["experimental_key"])
    record["condition"] = str(job["condition"])
    record["model_block"] = int(job["model_block"])
    record["checkpoint_seed"] = int(job["checkpoint_seed"])
    record.pop("arm", None)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    record["dynamic_recovery_active_steps"] = int(
        metrics.get("dynamic_recovery_active_steps", 0)
    )
    record["dynamic_recovery_align_steps"] = int(
        metrics.get("dynamic_recovery_align_steps", 0)
    )
    record["dynamic_recovery_advance_steps"] = int(
        metrics.get("dynamic_recovery_advance_steps", 0)
    )
    record["dynamic_recovery_abort_steps"] = int(
        metrics.get("dynamic_recovery_abort_steps", 0)
    )
    record["safety_interventions"] = int(
        metrics.get("safety_interventions", 0)
    )
    record["residual_stall_guard_latched"] = int(
        bool(metrics.get("residual_stall_guard_latched", False))
    )
    record["residual_stall_guard_latch_step"] = int(
        metrics.get("residual_stall_guard_latch_step", -1)
    )
    record["residual_stall_guard_minimum_authority"] = float(
        metrics.get("residual_stall_guard_minimum_authority", 1.0)
    )
    record["residual_reliability_alpha_mean"] = float(
        metrics.get("residual_reliability_alpha_mean", 0.0)
    )
    record["residual_reliability_alpha_max"] = float(
        metrics.get("residual_reliability_alpha_max", 0.0)
    )
    record["residual_safety_shield_acceptance_fraction"] = float(
        metrics.get("residual_safety_shield_acceptance_fraction", 0.0)
    )
    record["residual_safety_shield_fallback_fraction"] = float(
        metrics.get("residual_safety_shield_fallback_fraction", 0.0)
    )
    return record


def _state_control_arrays(config, trajectory_path):
    rows = _read_rows(trajectory_path)
    if not rows:
        raise ValueError("trajectory is empty")
    post_states = np.asarray(
        [
            [
                float(row["x"]),
                float(row["y"]),
                float(row["theta"]),
                float(row["v"]),
                float(row["omega"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    controls = np.asarray(
        [
            [float(row["applied_v"]), float(row["applied_omega"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    initial = np.asarray(
        config["experiment"]["initial_state"], dtype=np.float64
    )
    states = np.vstack((initial, post_states))
    if states.shape != (controls.shape[0] + 1, 5):
        raise ValueError("trajectory state/control alignment is invalid")
    if not np.isfinite(states).all() or not np.isfinite(controls).all():
        raise FloatingPointError("trajectory contains NaN or Inf")
    return states, controls


def _rk4_step(dynamics, state, control, dt):
    k1 = dynamics.derivative(state, control)
    k2 = dynamics.derivative(state + 0.5 * dt * k1, control)
    k3 = dynamics.derivative(state + 0.5 * dt * k2, control)
    k4 = dynamics.derivative(state + dt * k3, control)
    result = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    result[..., 2] = np.arctan2(
        np.sin(result[..., 2]), np.cos(result[..., 2])
    )
    return result


def prediction_sufficient_statistics(config, trajectory_path, dynamics):
    states, controls = _state_control_arrays(config, trajectory_path)
    dt = float(config["experiment"]["control_dt"])
    actual_derivative = (states[1:] - states[:-1]) / dt
    actual_derivative[:, 2] = np.arctan2(
        np.sin(states[1:, 2] - states[:-1, 2]),
        np.cos(states[1:, 2] - states[:-1, 2]),
    ) / dt
    derivative_error = (
        np.asarray(dynamics.derivative(states[:-1], controls))
        - actual_derivative
    )
    result = {
        "derivative": {
            "count": int(derivative_error.shape[0]),
            "sum_sq": np.sum(np.square(derivative_error), axis=0).tolist(),
        },
        "horizons": {},
    }
    control_count = int(controls.shape[0])
    for horizon in HORIZONS:
        window_count = control_count - int(horizon) + 1
        if window_count <= 0:
            raise ValueError("trajectory is shorter than a required horizon")
        starts = np.arange(window_count, dtype=np.int64)
        predicted = states[starts].copy()
        for offset in range(int(horizon)):
            predicted = _rk4_step(
                dynamics, predicted, controls[starts + offset], dt
            )
        error = predicted - states[starts + int(horizon)]
        error[:, 2] = np.arctan2(
            np.sin(error[:, 2]), np.cos(error[:, 2])
        )
        result["horizons"][str(horizon)] = {
            "count": int(window_count),
            "sum_sq": np.sum(np.square(error), axis=0).tolist(),
        }
    return result


def _merge_statistics(rows):
    merged = {
        "derivative": {"count": 0, "sum_sq": np.zeros(5)},
        "horizons": {
            str(horizon): {"count": 0, "sum_sq": np.zeros(5)}
            for horizon in HORIZONS
        },
    }
    for row in rows:
        merged["derivative"]["count"] += int(row["derivative"]["count"])
        merged["derivative"]["sum_sq"] += np.asarray(
            row["derivative"]["sum_sq"], dtype=np.float64
        )
        for horizon in HORIZONS:
            source = row["horizons"][str(horizon)]
            target = merged["horizons"][str(horizon)]
            target["count"] += int(source["count"])
            target["sum_sq"] += np.asarray(
                source["sum_sq"], dtype=np.float64
            )
    return merged


def _finalize_statistics(statistics):
    derivative = statistics["derivative"]
    derivative_rmse = np.sqrt(
        np.asarray(derivative["sum_sq"], dtype=np.float64)
        / float(derivative["count"])
    )
    result = {
        "derivative_count": int(derivative["count"]),
        "derivative_state_rmse": {
            name: float(value)
            for name, value in zip(STATE_NAMES, derivative_rmse)
        },
        "derivative_active_v_omega_rmse": float(
            np.sqrt(np.mean(np.square(derivative_rmse[3:5])))
        ),
        "horizons": {},
    }
    for horizon in HORIZONS:
        row = statistics["horizons"][str(horizon)]
        rmse = np.sqrt(
            np.asarray(row["sum_sq"], dtype=np.float64)
            / float(row["count"])
        )
        result["horizons"][str(horizon)] = {
            "window_count": int(row["count"]),
            "state_rmse": {
                name: float(value)
                for name, value in zip(STATE_NAMES, rmse)
            },
            "position_rmse": float(
                np.sqrt(np.mean(np.square(rmse[:2])))
            ),
            "heading_rmse": float(rmse[2]),
            "active_v_omega_rmse": float(
                np.sqrt(np.mean(np.square(rmse[3:5])))
            ),
            "overall_state_rmse": float(
                np.sqrt(np.mean(np.square(rmse)))
            ),
        }
    return result


def analyze_prediction(base_config, protocol, output_dir):
    nominal_dynamics = DynamicUnicyclePrediction(
        base_config["plant"]["nominal_velocity_time_constant"],
        base_config["plant"]["nominal_yaw_time_constant"],
    )
    seeds = [
        int(value)
        for value in protocol["design"]["development_episode_seeds"]
    ]
    nominal_by_seed = {}
    for seed in seeds:
        path = (
            output_dir
            / "runs"
            / "nominal"
            / ("seed_%d" % seed)
            / "trajectory.csv"
        )
        nominal_by_seed[seed] = prediction_sufficient_statistics(
            base_config, path, nominal_dynamics
        )
    nominal_summary = _finalize_statistics(
        _merge_statistics(list(nominal_by_seed.values()))
    )

    block_rows = []
    per_seed_rows = []
    block_details = []
    for block_index, block in enumerate(protocol["residual_model_blocks"]):
        residual = PlatformResidualDynamics.from_checkpoint(
            _resolve(block["checkpoint"]),
            device="cpu",
            use_torchscript=True,
        )
        canonicalization = dict(
            protocol.get("residual_input_canonicalization", {})
        )
        if bool(canonicalization.get("enabled", False)):
            residual = CanonicalizedStateResidualDynamics(
                residual,
                state_indices=(0, 1),
                canonical_values=canonicalization["values"],
            )
        component_mask = protocol.get("residual_component_mask")
        if component_mask is not None:
            residual = ResidualComponentMaskedDynamics(
                residual, component_mask
            )
        dynamics = ResidualPrediction(nominal_dynamics, residual)
        residual_by_seed = {}
        for seed in seeds:
            path = (
                output_dir
                / "runs"
                / "nominal"
                / ("seed_%d" % seed)
                / "trajectory.csv"
            )
            residual_by_seed[seed] = prediction_sufficient_statistics(
                base_config, path, dynamics
            )
            nominal_seed = _finalize_statistics(nominal_by_seed[seed])
            residual_seed = _finalize_statistics(residual_by_seed[seed])
            for horizon in HORIZONS:
                n_value = nominal_seed["horizons"][str(horizon)][
                    "active_v_omega_rmse"
                ]
                r_value = residual_seed["horizons"][str(horizon)][
                    "active_v_omega_rmse"
                ]
                per_seed_rows.append(
                    {
                        "model_block": int(block_index),
                        "checkpoint_seed": int(block["seed"]),
                        "episode_seed": int(seed),
                        "horizon": int(horizon),
                        "nominal_active_v_omega_rmse": float(n_value),
                        "residual_active_v_omega_rmse": float(r_value),
                        "relative_improvement": float(
                            (n_value - r_value) / n_value
                        ),
                    }
                )
        residual_summary = _finalize_statistics(
            _merge_statistics(list(residual_by_seed.values()))
        )
        nominal_h36 = nominal_summary["horizons"]["36"][
            "active_v_omega_rmse"
        ]
        residual_h36 = residual_summary["horizons"]["36"][
            "active_v_omega_rmse"
        ]
        block_rows.append(
            {
                "model_block": int(block_index),
                "checkpoint_seed": int(block["seed"]),
                "nominal_h36_active_v_omega_rmse": float(nominal_h36),
                "residual_h36_active_v_omega_rmse": float(residual_h36),
                "h36_relative_improvement": float(
                    (nominal_h36 - residual_h36) / nominal_h36
                ),
                "h36_improved": int(residual_h36 < nominal_h36),
            }
        )
        block_details.append(
            {
                "model_block": int(block_index),
                "checkpoint_seed": int(block["seed"]),
                "checkpoint": str(block["checkpoint"]),
                "nominal": nominal_summary,
                "icode_residual": residual_summary,
            }
        )
    result = {
        "schema_version": 1,
        "source_trajectory_condition": "nominal",
        "truth_use": "posthoc_prediction_audit_only",
        "primary_h36_active_formula": (
            "sqrt((RMSE_v^2 + RMSE_omega^2) / 2)"
        ),
        "nominal": nominal_summary,
        "model_blocks": block_details,
    }
    write_json(output_dir / "prediction_metrics.json", result)
    write_csv(output_dir / "prediction_block_summary.csv", block_rows)
    write_csv(output_dir / "prediction_per_seed.csv", per_seed_rows)
    return block_rows, result


def analyze_closed_loop(protocol, records, output_dir):
    nominal = {
        int(row["seed"]): row
        for row in records
        if row["condition"] == "nominal"
    }
    pairs = []
    for row in records:
        if row["condition"] != "icode_residual":
            continue
        reference = nominal[int(row["seed"])]
        pairs.append(
            {
                "model_block": int(row["model_block"]),
                "checkpoint_seed": int(row["checkpoint_seed"]),
                "episode_seed": int(row["seed"]),
                "nominal_collision": int(reference["collision"]),
                "residual_collision": int(row["collision"]),
                "collision_delta": int(row["collision"])
                - int(reference["collision"]),
                "nominal_completion": float(reference["completion"]),
                "residual_completion": float(row["completion"]),
                "completion_delta": float(row["completion"])
                - float(reference["completion"]),
                "nominal_final_goal_distance_m": float(
                    reference["final_goal_distance_m"]
                ),
                "residual_final_goal_distance_m": float(
                    row["final_goal_distance_m"]
                ),
                "final_goal_distance_delta_m": float(
                    row["final_goal_distance_m"]
                )
                - float(reference["final_goal_distance_m"]),
                "nominal_minimum_clearance_m": float(
                    reference["minimum_clearance_m"]
                ),
                "residual_minimum_clearance_m": float(
                    row["minimum_clearance_m"]
                ),
                "clearance_delta_m": float(row["minimum_clearance_m"])
                - float(reference["minimum_clearance_m"]),
                "nominal_planner_p95_compute_ms": float(
                    reference["planner_p95_compute_ms"]
                ),
                "residual_planner_p95_compute_ms": float(
                    row["planner_p95_compute_ms"]
                ),
            }
        )
    write_csv(output_dir / "closed_loop_pairs.csv", pairs)
    return pairs


def _first_true_index(values):
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if indices.size == 0 else int(indices[0])


def analyze_failure_mechanism(protocol, records, output_dir):
    nominal_records = {
        int(row["seed"]): row
        for row in records
        if row["condition"] == "nominal"
    }
    divergence_rows = []
    for row in records:
        if row["condition"] != "icode_residual":
            continue
        reference = nominal_records[int(row["seed"])]
        residual_rows = _read_rows(ROOT / row["run_dir"] / "trajectory.csv")
        nominal_rows = _read_rows(
            ROOT / reference["run_dir"] / "trajectory.csv"
        )
        count = min(len(residual_rows), len(nominal_rows))
        residual_action = np.asarray(
            [
                [float(item["executed_v"]), float(item["executed_omega"])]
                for item in residual_rows[:count]
            ]
        )
        nominal_action = np.asarray(
            [
                [float(item["executed_v"]), float(item["executed_omega"])]
                for item in nominal_rows[:count]
            ]
        )
        residual_xy = np.asarray(
            [
                [float(item["x"]), float(item["y"])]
                for item in residual_rows[:count]
            ]
        )
        nominal_xy = np.asarray(
            [
                [float(item["x"]), float(item["y"])]
                for item in nominal_rows[:count]
            ]
        )
        action_index = _first_true_index(
            np.linalg.norm(residual_action - nominal_action, axis=1) > 0.05
        )
        position_index = _first_true_index(
            np.linalg.norm(residual_xy - nominal_xy, axis=1) > 0.10
        )
        collision_index = _first_true_index(
            [float(item["collision"]) > 0.5 for item in residual_rows]
        )
        divergence_rows.append(
            {
                "model_block": int(row["model_block"]),
                "checkpoint_seed": int(row["checkpoint_seed"]),
                "episode_seed": int(row["seed"]),
                "first_control_divergence_step": action_index,
                "first_control_divergence_time_s": (
                    None
                    if action_index is None
                    else float(
                        (action_index + 1)
                        * float(
                            protocol.get(
                                "control_dt",
                                0.1,
                            )
                        )
                    )
                ),
                "first_position_divergence_step": position_index,
                "first_position_divergence_time_s": (
                    None
                    if position_index is None
                    else float(
                        (position_index + 1)
                        * float(protocol.get("control_dt", 0.1))
                    )
                ),
                "collision_step": collision_index,
                "collision_time_s": (
                    None
                    if collision_index is None
                    else float(
                        (collision_index + 1)
                        * float(protocol.get("control_dt", 0.1))
                    )
                ),
                "active_escape_steps": int(
                    sum(
                        item["safety_reason"] == "dynamic_active_escape"
                        for item in residual_rows
                    )
                ),
                "hard_or_soft_stop_steps": int(
                    sum(
                        item["safety_reason"]
                        in (
                            "near_body_hard_stop",
                            "front_soft_block",
                            "probabilistic_risk_stop",
                        )
                        for item in residual_rows
                    )
                ),
            }
        )

    first_block = protocol["residual_model_blocks"][0]
    model = PlatformResidualDynamics.from_checkpoint(
        _resolve(first_block["checkpoint"]), device="cpu"
    ).model
    mean = model.feature_mean.detach().cpu().numpy().astype(np.float64)
    scale = model.feature_scale.detach().cpu().numpy().astype(np.float64)
    support_rows = []
    for seed, reference in sorted(nominal_records.items()):
        rows = _read_rows(ROOT / reference["run_dir"] / "trajectory.csv")
        state = np.asarray(
            [
                [
                    float(item["x"]),
                    float(item["y"]),
                    float(item["theta"]),
                    float(item["v"]),
                    float(item["omega"]),
                ]
                for item in rows
            ]
        )
        features = np.column_stack(
            (
                state[:, 0],
                state[:, 1],
                np.sin(state[:, 2]),
                np.cos(state[:, 2]),
                state[:, 3],
                state[:, 4],
            )
        )
        z_score = np.abs((features - mean) / scale)
        support_rows.append(
            {
                "episode_seed": int(seed),
                "maximum_feature_z": float(np.max(z_score)),
                "maximum_spatial_z": float(np.max(z_score[:, :2])),
                "maximum_nonspatial_z": float(np.max(z_score[:, 2:])),
                "fraction_steps_spatial_z_above_5": float(
                    np.mean(np.any(z_score[:, :2] > 5.0, axis=1))
                ),
            }
        )
    result = {
        "schema_version": 1,
        "interpretation": (
            "posthoc mechanism diagnosis; not an independent safety claim"
        ),
        "divergence_thresholds": {
            "executed_control_l2": 0.05,
            "position_l2_m": 0.10,
        },
        "paired_divergence": divergence_rows,
        "checkpoint_spatial_feature_mean": mean[:2].tolist(),
        "checkpoint_spatial_feature_scale": scale[:2].tolist(),
        "nominal_trajectory_support": support_rows,
    }
    write_json(output_dir / "failure_diagnostics.json", result)
    return result


def _protocol_stage(protocol):
    preregistration = str(protocol.get("preregistration", "")).lower()
    if "residual_runtime_stage3" in preregistration:
        return "residual_runtime_stage3_development"
    if "stage2_task_aware" in preregistration:
        return "residual_dynamics_stage2_task_aware_development"
    return "residual_dynamics_stage1_development"


def evaluate_gate(protocol, checkpoint_audit, prediction_rows, records, pairs):
    gate = protocol["gate"]
    positive_blocks = int(sum(row["h36_improved"] for row in prediction_rows))
    collision_increases = []
    for block_index in range(len(protocol["residual_model_blocks"])):
        collision_increases.append(
            int(
                sum(
                    row["collision_delta"]
                    for row in pairs
                    if int(row["model_block"]) == block_index
                )
            )
        )
    completion_delta = float(
        np.median([row["completion_delta"] for row in pairs])
    )
    clearance_delta = float(
        np.median([row["clearance_delta_m"] for row in pairs])
    )
    residual_records = [
        row for row in records if row["condition"] == "icode_residual"
    ]
    planner_p95 = float(
        max(row["planner_p95_compute_ms"] for row in residual_records)
    )
    forecast_contracts = all(
        int(row["forecast_contract_valid"]) == 1 for row in records
    )
    required_stall_latch = bool(
        gate.get("require_stall_guard_latch_per_residual_block", False)
    )
    latch_by_block = [
        int(
            any(
                int(row.get("residual_stall_guard_latched", 0)) == 1
                for row in residual_records
                if int(row["model_block"]) == block_index
            )
        )
        for block_index in range(len(protocol["residual_model_blocks"]))
    ]
    minimum_reliability_alpha = gate.get(
        "minimum_maximum_reliability_alpha_per_residual_block"
    )
    reliability_alpha_by_block = [
        float(
            max(
                (
                    row.get("residual_reliability_alpha_max", 0.0)
                    for row in residual_records
                    if int(row["model_block"]) == block_index
                ),
                default=0.0,
            )
        )
        for block_index in range(len(protocol["residual_model_blocks"]))
    ]
    minimum_shield_acceptance = gate.get(
        "minimum_shield_acceptance_fraction_per_residual_block"
    )
    shield_acceptance_by_block = [
        float(
            max(
                (
                    row.get(
                        "residual_safety_shield_acceptance_fraction", 0.0
                    )
                    for row in residual_records
                    if int(row["model_block"]) == block_index
                ),
                default=0.0,
            )
        )
        for block_index in range(len(protocol["residual_model_blocks"]))
    ]
    gates = {
        "checkpoint_load_and_finite": {
            "value": int(len(checkpoint_audit["checkpoint_blocks"])),
            "threshold": 3,
            "pass": len(checkpoint_audit["checkpoint_blocks"]) == 3,
        },
        "positive_h36_model_blocks": {
            "value": positive_blocks,
            "threshold": int(gate["minimum_positive_h36_model_blocks"]),
            "pass": positive_blocks
            >= int(gate["minimum_positive_h36_model_blocks"]),
        },
        "maximum_collision_count_increase_per_block": {
            "value": int(max(collision_increases)),
            "per_block": collision_increases,
            "threshold": int(
                gate["maximum_collision_count_increase_per_block"]
            ),
            "pass": max(collision_increases)
            <= int(gate["maximum_collision_count_increase_per_block"]),
        },
        "median_completion_delta": {
            "value": completion_delta,
            "threshold": float(gate["minimum_median_completion_delta"]),
            "pass": completion_delta
            >= float(gate["minimum_median_completion_delta"]),
        },
        "median_clearance_delta_m": {
            "value": clearance_delta,
            "threshold": float(gate["minimum_median_clearance_delta_m"]),
            "pass": clearance_delta
            >= float(gate["minimum_median_clearance_delta_m"]),
        },
        "residual_planner_p95_compute_ms": {
            "value": planner_p95,
            "threshold": float(
                gate["maximum_enabled_planner_p95_compute_ms"]
            ),
            "pass": planner_p95
            <= float(gate["maximum_enabled_planner_p95_compute_ms"]),
        },
        "forecast_contract_and_artifact_integrity": {
            "value": int(forecast_contracts),
            "threshold": 1,
            "pass": bool(forecast_contracts),
        },
    }
    if required_stall_latch:
        gates["stall_guard_latched_per_residual_block"] = {
            "value": int(sum(latch_by_block)),
            "per_block": latch_by_block,
            "threshold": len(protocol["residual_model_blocks"]),
            "pass": all(latch_by_block),
        }
    if minimum_reliability_alpha is not None:
        gates["reliability_alpha_active_per_residual_block"] = {
            "value": float(min(reliability_alpha_by_block)),
            "per_block": reliability_alpha_by_block,
            "threshold": float(minimum_reliability_alpha),
            "pass": all(
                value >= float(minimum_reliability_alpha)
                for value in reliability_alpha_by_block
            ),
        }
    if minimum_shield_acceptance is not None:
        gates["shield_acceptance_active_per_residual_block"] = {
            "value": float(min(shield_acceptance_by_block)),
            "per_block": shield_acceptance_by_block,
            "threshold": float(minimum_shield_acceptance),
            "pass": all(
                value >= float(minimum_shield_acceptance)
                for value in shield_acceptance_by_block
            ),
        }
    return {
        "schema_version": 1,
        "stage": _protocol_stage(protocol),
        "overall_pass": bool(all(row["pass"] for row in gates.values())),
        "sealed_seeds_opened": False,
        "independent_unit": "complete_episode",
        "model_block_unit": "residual_checkpoint_seed",
        "gates": gates,
    }


def _write_readme(output_dir, gate_result):
    status = "PASS" if gate_result["overall_pass"] else "FAIL"
    stage_title = (
        {
            "residual_runtime_stage3_development": (
                "Residual Runtime Stage 3 Development"
            ),
            "residual_dynamics_stage2_task_aware_development": (
                "Residual-Dynamics Stage 2 Task-Aware Development"
            ),
        }.get(
            gate_result["stage"], "Residual-Dynamics Stage 1 Development"
        )
    )
    if gate_result["stage"] == "residual_runtime_stage3_development":
        scope_lines = [
            "The Stage 2 checkpoints, environment, MPPI budget, residual mask",
            "and shield thresholds remained frozen. Only the runtime backend",
            "and scheduling options recorded in the resolved config changed.",
            "Simulator truth was used only for posthoc audit and model error.",
            "No sealed seed was opened and RL remained disabled.",
        ]
    else:
        scope_lines = [
            "Amendment 17 remained frozen. The online controller changed only",
            "`planner.prediction_mode` and the registered checkpoint path.",
            "Simulator truth was used only for posthoc audit and model error.",
            "No sealed seed was opened.",
        ]
    lines = [
        "# %s" % stage_title,
        "",
        "Overall Gate: **%s**" % status,
        "",
        *scope_lines,
        "",
        "Primary files:",
        "",
        "- `gate.json`",
        "- `episode_summary.csv`",
        "- `closed_loop_pairs.csv`",
        "- `prediction_metrics.json`",
        "- `prediction_block_summary.csv`",
        "- `checkpoint_audit.json`",
        "",
    ]
    (output_dir / "README.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(protocol_path=DEFAULT_PROTOCOL, output_override=None, max_jobs=None):
    protocol_path = Path(protocol_path).resolve()
    protocol = _load_yaml_mapping(protocol_path)
    base_config_path = _resolve(protocol["base_config"])
    base_config = load_yaml(base_config_path)
    checkpoint_audit = validate_protocol(protocol, base_config)
    output_dir = Path(
        output_override or _resolve(protocol["design"]["output_dir"])
    ).resolve()
    if ROOT not in output_dir.parents:
        raise ValueError("Stage 1 output must remain inside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_hash = sha256_file(protocol_path)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol_sha256"] != protocol_hash:
            raise ValueError("cannot resume with a changed Stage 1 protocol")
    else:
        write_json(
            manifest_path,
            {
                "schema_version": 1,
                "protocol": str(protocol_path.relative_to(ROOT)),
                "protocol_sha256": protocol_hash,
                "base_config": str(base_config_path.relative_to(ROOT)),
                "base_config_sha256": sha256_file(base_config_path),
                "shared_nominal_reference_per_episode_seed": True,
                "sealed_seeds_opened": False,
            },
        )
        write_yaml(output_dir / "protocol_resolved.yaml", protocol)
        write_yaml(output_dir / "base_config_resolved.yaml", base_config)
        write_json(output_dir / "checkpoint_audit.json", checkpoint_audit)
        write_json(
            output_dir / "environment.json", environment_manifest(ROOT)
        )

    schedule = build_schedule(protocol)
    write_json(
        output_dir / "schedule.json",
        {
            "schema_version": 1,
            "schedule_seed": int(protocol["design"]["schedule_seed"]),
            "common_random_numbers": True,
            "shared_nominal_reference_per_episode_seed": True,
            "jobs": schedule,
        },
    )
    trajectories = {
        int(seed): _obstacle_trajectory(base_config, int(seed))
        for seed in protocol["design"]["development_episode_seeds"]
    }
    completed_this_call = 0
    early_stop = _probe_early_stop_reason(
        protocol, schedule, output_dir
    )
    for job in schedule:
        if early_stop is not None:
            break
        run_dir = _run_dir(output_dir, job)
        if _complete_run(run_dir):
            continue
        if max_jobs is not None and completed_this_call >= int(max_jobs):
            break
        run_config = configure_condition(base_config, job, protocol)
        print(
            "[%d/%d] seed=%d condition=%s block=%d"
            % (
                int(job["run_order"]) + 1,
                len(schedule),
                int(job["episode_seed"]),
                str(job["condition"]),
                int(job["model_block"]),
            ),
            flush=True,
        )
        ExperimentRunner(
            run_config,
            ROOT,
            output_dir=run_dir,
            headless=True,
        ).run()
        completed_this_call += 1
        early_stop = _probe_early_stop_reason(
            protocol, schedule, output_dir
        )

    complete_jobs = [
        job for job in schedule if _complete_run(_run_dir(output_dir, job))
    ]
    records = [
        _episode_record(
            base_config,
            job,
            _run_dir(output_dir, job),
            trajectories[int(job["episode_seed"])],
        )
        for job in complete_jobs
    ]
    records.sort(key=lambda row: int(row["run_order"]))
    if records:
        write_csv(output_dir / "episode_summary.csv", records)
    if len(complete_jobs) != len(schedule):
        progress = {
            "status": (
                "stopped_by_preregistered_rule"
                if early_stop is not None
                else "incomplete"
            ),
            "completed_jobs": len(complete_jobs),
            "total_jobs": len(schedule),
            "remaining_jobs": len(schedule) - len(complete_jobs),
            "output_dir": str(output_dir),
        }
        if early_stop is not None:
            progress["early_stop"] = early_stop
            write_json(output_dir / "early_stop.json", progress)
        write_json(output_dir / "progress.json", progress)
        return progress

    block_rows, _ = analyze_prediction(base_config, protocol, output_dir)
    pairs = analyze_closed_loop(protocol, records, output_dir)
    analyze_failure_mechanism(protocol, records, output_dir)
    gate_result = evaluate_gate(
        protocol, checkpoint_audit, block_rows, records, pairs
    )
    write_json(output_dir / "gate.json", gate_result)
    _write_readme(output_dir, gate_result)
    artifact_names = (
        "run_manifest.json",
        "protocol_resolved.yaml",
        "base_config_resolved.yaml",
        "checkpoint_audit.json",
        "environment.json",
        "schedule.json",
        "episode_summary.csv",
        "closed_loop_pairs.csv",
        "failure_diagnostics.json",
        "prediction_metrics.json",
        "prediction_block_summary.csv",
        "prediction_per_seed.csv",
        "gate.json",
        "README.md",
    )
    provenance = {
        "schema_version": 1,
        "stage": _protocol_stage(protocol),
        **repository_provenance(ROOT),
        "sealed_seeds_opened": False,
        "artifacts": {
            name: {
                "path": name,
                "sha256": sha256_file(output_dir / name),
            }
            for name in artifact_names
        },
    }
    write_json(output_dir / "provenance.json", provenance)
    summary = {
        "status": "complete",
        "overall_pass": bool(gate_result["overall_pass"]),
        "episode_count": len(records),
        "paired_comparison_count": len(pairs),
        "model_block_count": len(block_rows),
        "sealed_seeds_opened": False,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args(argv)
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("--max-jobs must be positive")
    result = run(args.protocol, args.output, args.max_jobs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
