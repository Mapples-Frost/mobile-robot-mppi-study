#!/usr/bin/env python3
"""Run the clean nominal-versus-oracle residual MPPI ablation.

This is an obstacle-free dynamics/control benchmark.  The executed plant is a
fresh :class:`DisturbedUnicycle` for every episode; the planner receives only
the prediction adapter selected for its ablation mode.  Memory, perception,
MuJoCo, ROS, and safety arbitration are intentionally outside this script.

``--smoke`` validates wiring and deterministic artifact production only.  Its
numbers must not be reported as formal experimental results.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
import random
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


os.environ.setdefault("MPLBACKEND", "Agg")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.dynamics import (  # noqa: E402
    CombinedDynamics,
    DisturbanceConfig,
    DisturbedUnicycle,
    NominalUnicycle,
)
from src.dynamics.residual.oracle_residual import OracleResidual  # noqa: E402
from src.learning.checkpointing import get_git_sha  # noqa: E402
# This module owns the repository's established receding-horizon sampling,
# rollout, cost, and weighted-update helpers.  Importing it does not construct
# a MuJoCo environment; this benchmark uses only its pure planning functions.
from src.planners import mppi_mujoco_receding_horizon_experiment as mppi  # noqa: E402
from src.planners.mppi_dynamics_adapter import (  # noqa: E402
    MppiDynamicsAdapter,
    build_prediction_dynamics,
)
from src.planners.sampling_prior import (  # noqa: E402
    GoalWarmStartPrior,
    PreviousSequencePrior,
)


GOAL: Tuple[float, float] = (3.0, 3.0)
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "icode" / "oracle_ablation.yaml"
ARTIFACT_FILENAMES = {
    "metrics": "metrics.json",
    "summary": "summary.csv",
    "trajectory": "trajectory.csv",
    "config": "config_snapshot.yaml",
}

SUMMARY_FIELDS = (
    "run_type",
    "mode",
    "seed",
    "success",
    "bounds_violation",
    "steps",
    "duration_seconds",
    "final_goal_distance",
    "path_length",
    "executed_cost",
    "control_effort",
    "yaw_effort",
    "mean_planner_compute_ms",
    "p95_planner_compute_ms",
    "max_planner_compute_ms",
    "total_planner_compute_ms",
    "proposed_executed_max_abs_difference",
    "git_sha",
)

TRAJECTORY_FIELDS = (
    "run_type",
    "mode",
    "seed",
    "step",
    "time",
    "state_x",
    "state_y",
    "state_theta",
    "next_state_x",
    "next_state_y",
    "next_state_theta",
    "goal_distance_before",
    "goal_distance_after",
    "proposed_velocity",
    "proposed_yaw_rate",
    "executed_velocity",
    "executed_yaw_rate",
    "planner_compute_ms",
    "predicted_terminal_x",
    "predicted_terminal_y",
    "predicted_terminal_theta",
    "predicted_cost",
    "bounds_violation",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("{} must be a mapping".format(name))
    return dict(value)


def _finite_float(value: Any, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError("{} must be a finite scalar".format(name))
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        raise ValueError("{} has an invalid value".format(name))
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("{} must be a positive integer".format(name))
    result = int(value)
    if result <= 0:
        raise ValueError("{} must be positive".format(name))
    return result


def _vector(value: Any, length: int, name: str) -> List[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("{} must be a length-{} sequence".format(name, length))
    if len(value) != length:
        raise ValueError("{} must have length {}".format(name, length))
    return [_finite_float(item, "{}[{}]".format(name, index)) for index, item in enumerate(value)]


def _path_from_root(value: Any) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (Path, os.PathLike)):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError("cannot serialize {}".format(type(value).__name__))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp-{}".format(path.name, uuid.uuid4().hex))
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fields})
    _atomic_write_text(path, stream.getvalue())


def load_config(path: Path) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    with source.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ValueError("benchmark config must contain a YAML mapping")
    return copy.deepcopy(dict(value))


def resolve_config(
    value: Mapping[str, Any],
    config_path: Path,
    output_dir: Optional[Path] = None,
    smoke: bool = False,
    headless: bool = False,
) -> Dict[str, Any]:
    """Resolve CLI overrides and enforce the clean benchmark contract."""

    config = copy.deepcopy(_mapping(value, "config"))
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("unsupported config schema_version")
    run = _mapping(config.get("run"), "run")
    benchmark = _mapping(config.get("benchmark"), "benchmark")
    plant = _mapping(config.get("plant"), "plant")
    planner = _mapping(config.get("planner"), "planner")
    identity = _mapping(config.get("identity_check"), "identity_check")
    smoke_config = _mapping(config.get("smoke"), "smoke")

    run["run_type"] = "smoke" if smoke else "research"
    run["headless"] = bool(headless or run.get("headless", False))
    seeds_value = run.get("smoke_seeds" if smoke else "seeds")
    if isinstance(seeds_value, (str, bytes)) or not isinstance(seeds_value, Sequence):
        raise TypeError("run seeds must be a non-empty sequence")
    seeds = [_positive_int(seed, "run.seed") for seed in seeds_value]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("run seeds must be non-empty and unique")
    run["effective_seeds"] = seeds
    run["output_dir"] = str(
        _path_from_root(output_dir if output_dir is not None else run.get("output_dir"))
    )
    run["source_config"] = str(Path(config_path).expanduser().resolve())

    benchmark["start_state"] = _vector(benchmark.get("start_state"), 3, "benchmark.start_state")
    benchmark["goal"] = _vector(benchmark.get("goal"), 2, "benchmark.goal")
    if tuple(benchmark["goal"]) != GOAL:
        raise ValueError("this benchmark fixes GOAL at {}".format(GOAL))
    benchmark["goal_tolerance"] = _finite_float(
        benchmark.get("goal_tolerance"), "benchmark.goal_tolerance", positive=True
    )
    benchmark["dt"] = _finite_float(benchmark.get("dt"), "benchmark.dt", positive=True)
    benchmark["max_steps"] = _positive_int(benchmark.get("max_steps"), "benchmark.max_steps")
    method = str(benchmark.get("integration_method", "rk4")).strip().lower()
    if method not in ("euler", "rk4"):
        raise ValueError("benchmark.integration_method must be euler or rk4")
    benchmark["integration_method"] = method
    if benchmark.get("obstacle_free") is not True or benchmark.get("obstacles") not in ([], ()):
        raise ValueError("oracle ablation must remain explicitly obstacle-free")
    if benchmark.get("memory_enabled") is not False:
        raise ValueError("memory must be explicitly disabled")
    benchmark["robot_radius"] = _finite_float(
        benchmark.get("robot_radius"), "benchmark.robot_radius", positive=True
    )
    bounds = _mapping(benchmark.get("bounds"), "benchmark.bounds")
    for key in ("x_min", "x_max", "y_min", "y_max"):
        bounds[key] = _finite_float(bounds.get(key), "benchmark.bounds.{}".format(key))
    if bounds["x_min"] >= bounds["x_max"] or bounds["y_min"] >= bounds["y_max"]:
        raise ValueError("benchmark bounds are invalid")
    benchmark["bounds"] = bounds

    if int(plant.get("control_delay_steps", 0)) != 0:
        raise ValueError("oracle benchmark forbids non-Markov command delay")
    planner["horizon"] = _positive_int(planner.get("horizon"), "planner.horizon")
    planner["num_samples"] = _positive_int(planner.get("num_samples"), "planner.num_samples")
    planner["temperature"] = _finite_float(planner.get("temperature"), "planner.temperature", positive=True)
    planner["nominal_control"] = _vector(planner.get("nominal_control"), 2, "planner.nominal_control")
    planner["tail_control"] = _vector(planner.get("tail_control"), 2, "planner.tail_control")
    planner["warm_start_prefix_steps"] = _positive_int(
        planner.get("warm_start_prefix_steps"), "planner.warm_start_prefix_steps"
    )
    for key in ("velocity_std", "yaw_rate_std", "velocity_max", "yaw_rate_max"):
        planner[key] = _finite_float(planner.get(key), "planner.{}".format(key), positive=True)
    planner["velocity_min"] = _finite_float(planner.get("velocity_min"), "planner.velocity_min")
    if planner["velocity_min"] >= planner["velocity_max"]:
        raise ValueError("planner velocity limits are invalid")
    if not isinstance(planner.get("anisotropic_sampling"), bool):
        raise TypeError("planner.anisotropic_sampling must be boolean")
    for key in ("sdf_influence_distance", "sigma_parallel", "sigma_perpendicular"):
        planner[key] = _finite_float(planner.get(key), "planner.{}".format(key), positive=True)

    identity["absolute_tolerance"] = _finite_float(
        identity.get("absolute_tolerance"), "identity_check.absolute_tolerance", positive=True
    )
    identity["minimum_nominal_mismatch"] = _finite_float(
        identity.get("minimum_nominal_mismatch"),
        "identity_check.minimum_nominal_mismatch",
        positive=True,
    )
    identity["states"] = [
        _vector(state, 3, "identity_check.states") for state in identity.get("states", [])
    ]
    identity["controls"] = [
        _vector(control, 2, "identity_check.controls") for control in identity.get("controls", [])
    ]
    identity["times"] = [
        _finite_float(item, "identity_check.times") for item in identity.get("times", [])
    ]
    if not identity["states"] or not (
        len(identity["states"]) == len(identity["controls"]) == len(identity["times"])
    ):
        raise ValueError("identity check states, controls, and times must have equal nonzero length")

    if smoke:
        planner["horizon"] = _positive_int(smoke_config.get("horizon"), "smoke.horizon")
        planner["num_samples"] = _positive_int(smoke_config.get("num_samples"), "smoke.num_samples")
        benchmark["max_steps"] = _positive_int(smoke_config.get("max_steps"), "smoke.max_steps")

    config.update(
        {
            "run": run,
            "benchmark": benchmark,
            "plant": plant,
            "planner": planner,
            "identity_check": identity,
            "smoke": smoke_config,
        }
    )
    return config


def disturbance_config(config: Mapping[str, Any]) -> DisturbanceConfig:
    plant = _mapping(config["plant"], "plant")
    for key in (
        "world_disturbance_amplitude",
        "world_disturbance_frequency",
        "world_disturbance_phase",
        "state_disturbance_gain",
    ):
        if key in plant:
            plant[key] = tuple(plant[key])
    return DisturbanceConfig(**plant)


def build_oracle_modes(
    config: Mapping[str, Any]
) -> Tuple[Dict[str, MppiDynamicsAdapter], Dict[str, Any]]:
    """Create separate nominal and exact-oracle planner prediction models."""

    method = str(config["benchmark"]["integration_method"])
    true_for_oracle = DisturbedUnicycle(config=disturbance_config(config))
    nominal_adapter = build_prediction_dynamics(
        mode="nominal", integration_method=method, clone_per_rollout=False
    )
    oracle_adapter = build_prediction_dynamics(
        mode="oracle_residual",
        true_dynamics=true_for_oracle,
        integration_method=method,
        clone_per_rollout=False,
    )
    return {
        "nominal_mismatch": nominal_adapter,
        "oracle_residual": oracle_adapter,
    }, {"true_for_oracle": true_for_oracle, "oracle_adapter": oracle_adapter}


def verify_oracle_identity(
    config: Mapping[str, Any], oracle_adapter: MppiDynamicsAdapter
) -> Dict[str, Any]:
    """Fail fast unless nominal plus oracle is exactly the configured plant."""

    check = config["identity_check"]
    truth = DisturbedUnicycle(config=disturbance_config(config))
    nominal = NominalUnicycle()
    residual = OracleResidual(truth, nominal)
    combined = CombinedDynamics(nominal, residual)
    errors: List[float] = []
    mismatch_norms: List[float] = []
    adapter_errors: List[float] = []
    dt = float(config["benchmark"]["dt"])
    method = str(config["benchmark"]["integration_method"])
    for state, control, evaluation_time in zip(
        check["states"], check["controls"], check["times"]
    ):
        state_array = np.asarray(state, dtype=np.float64)
        control_array = np.asarray(control, dtype=np.float64)
        true_derivative = truth.derivative(state_array, control_array, time=evaluation_time)
        combined_derivative = combined.derivative(
            state_array, control_array, time=evaluation_time
        )
        nominal_derivative = nominal.derivative(
            state_array, control_array, time=evaluation_time
        )
        errors.append(float(np.max(np.abs(true_derivative - combined_derivative))))
        mismatch_norms.append(float(np.linalg.norm(true_derivative - nominal_derivative)))
        expected_next = truth.clone().step(
            state_array, control_array, dt, method=method, time=0.0
        )
        adapter_next = np.asarray(
            oracle_adapter.rollout(state_array, [control_array], dt)[-1],
            dtype=np.float64,
        )
        adapter_errors.append(float(np.max(np.abs(expected_next - adapter_next))))
    maximum_error = max(errors)
    maximum_adapter_error = max(adapter_errors)
    maximum_nominal_mismatch = max(mismatch_norms)
    tolerance = float(check["absolute_tolerance"])
    if maximum_error > tolerance or maximum_adapter_error > tolerance:
        raise RuntimeError(
            "oracle identity check failed: derivative_error={} adapter_error={}".format(
                maximum_error, maximum_adapter_error
            )
        )
    if maximum_nominal_mismatch < float(check["minimum_nominal_mismatch"]):
        raise RuntimeError("configured plant is effectively nominal; ablation wiring is invalid")
    return {
        "passed": True,
        "samples": len(errors),
        "maximum_derivative_absolute_error": maximum_error,
        "maximum_one_step_absolute_error": maximum_adapter_error,
        "maximum_nominal_mismatch_l2": maximum_nominal_mismatch,
        "absolute_tolerance": tolerance,
    }


def _goal_distance(state: Sequence[float]) -> float:
    return float(math.hypot(GOAL[0] - float(state[0]), GOAL[1] - float(state[1])))


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0


def _plan_once(
    current_state: Tuple[float, float, float],
    nominal_sequence: Sequence[Tuple[float, float]],
    prediction_model: MppiDynamicsAdapter,
    config: Mapping[str, Any],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float]], float]:
    benchmark = config["benchmark"]
    planner = config["planner"]
    sampled_sequences = mppi.sample_control_sequences(
        nominal_sequence=nominal_sequence,
        current_state=current_state,
        dt=float(benchmark["dt"]),
        obstacles=[],
        bounds=benchmark["bounds"],
        robot_radius=float(benchmark["robot_radius"]),
        num_samples=int(planner["num_samples"]),
        v_std=float(planner["velocity_std"]),
        omega_std=float(planner["yaw_rate_std"]),
        v_min=float(planner["velocity_min"]),
        v_max=float(planner["velocity_max"]),
        omega_max=float(planner["yaw_rate_max"]),
        use_anisotropic_sampling=bool(planner["anisotropic_sampling"]),
        sdf_influence_distance=float(planner["sdf_influence_distance"]),
        sigma_parallel=float(planner["sigma_parallel"]),
        sigma_perp=float(planner["sigma_perpendicular"]),
        dynamics_model=prediction_model,
    )
    costs: List[float] = []
    for sequence in sampled_sequences:
        trajectory = mppi.rollout_control_sequence(
            current_state,
            sequence,
            float(benchmark["dt"]),
            dynamics_model=prediction_model,
        )
        cost, _ = mppi.trajectory_cost(
            trajectory,
            sequence,
            GOAL,
            [],
            float(benchmark["robot_radius"]),
            benchmark["bounds"],
        )
        costs.append(float(cost))
    weights = mppi.compute_weights(costs, float(planner["temperature"]))
    updated = [tuple(item) for item in mppi.weighted_update_sequence(sampled_sequences, weights)]
    updated_trajectory = [
        tuple(item)
        for item in mppi.rollout_control_sequence(
            current_state,
            updated,
            float(benchmark["dt"]),
            dynamics_model=prediction_model,
        )
    ]
    updated_cost, _ = mppi.trajectory_cost(
        updated_trajectory,
        updated,
        GOAL,
        [],
        float(benchmark["robot_radius"]),
        benchmark["bounds"],
    )
    return updated, updated_trajectory, float(updated_cost)


def run_episode(
    mode: str,
    prediction_model: MppiDynamicsAdapter,
    seed: int,
    config: Mapping[str, Any],
    git_sha: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Execute one true-plant episode using one planner prediction model."""

    random.seed(seed)
    np.random.seed(seed)
    benchmark = config["benchmark"]
    planner = config["planner"]
    plant = DisturbedUnicycle(config=disturbance_config(config))
    plant.reset()
    state = tuple(float(item) for item in benchmark["start_state"])
    states: List[Tuple[float, float, float]] = [state]
    proposed_controls: List[Tuple[float, float]] = []
    executed_controls: List[Tuple[float, float]] = []
    compute_times: List[float] = []
    rows: List[Dict[str, Any]] = []
    warm_start_prior = GoalWarmStartPrior(
        goal=GOAL,
        dt=float(benchmark["dt"]),
        v_max=float(planner["velocity_max"]),
        omega_max=float(planner["yaw_rate_max"]),
        warm_start_prefix_steps=int(planner["warm_start_prefix_steps"]),
        nominal_tail_control=tuple(planner["nominal_control"]),
    )
    nominal_sequence = warm_start_prior.mean_control_sequence(
        state, int(planner["horizon"])
    )
    previous_prior = PreviousSequencePrior(
        nominal_sequence,
        pad_control=tuple(planner["tail_control"]),
        v_limits=(float(planner["velocity_min"]), float(planner["velocity_max"])),
        omega_limits=(
            -float(planner["yaw_rate_max"]),
            float(planner["yaw_rate_max"]),
        ),
    )
    bounds_violation = False
    for step in range(int(benchmark["max_steps"])):
        start = time.perf_counter()
        updated, predicted_trajectory, predicted_cost = _plan_once(
            state, nominal_sequence, prediction_model, config
        )
        compute_ms = (time.perf_counter() - start) * 1000.0
        proposed = tuple(float(item) for item in updated[0])
        # There is intentionally no safety-arbitration layer in this clean
        # dynamics benchmark.  Recording both values makes that fact auditable.
        executed = proposed
        next_state_array = plant.step(
            state,
            executed,
            float(benchmark["dt"]),
            method=str(benchmark["integration_method"]),
            time=step * float(benchmark["dt"]),
        )
        next_state = tuple(float(item) for item in next_state_array)
        bounds_violation = not bool(
            mppi.is_inside_effective_bounds(
                next_state, benchmark["bounds"], float(benchmark["robot_radius"])
            )
        )
        terminal = predicted_trajectory[-1]
        rows.append(
            {
                "run_type": config["run"]["run_type"],
                "mode": mode,
                "seed": seed,
                "step": step,
                "time": step * float(benchmark["dt"]),
                "state_x": state[0],
                "state_y": state[1],
                "state_theta": state[2],
                "next_state_x": next_state[0],
                "next_state_y": next_state[1],
                "next_state_theta": next_state[2],
                "goal_distance_before": _goal_distance(state),
                "goal_distance_after": _goal_distance(next_state),
                "proposed_velocity": proposed[0],
                "proposed_yaw_rate": proposed[1],
                "executed_velocity": executed[0],
                "executed_yaw_rate": executed[1],
                "planner_compute_ms": compute_ms,
                "predicted_terminal_x": terminal[0],
                "predicted_terminal_y": terminal[1],
                "predicted_terminal_theta": terminal[2],
                "predicted_cost": predicted_cost,
                "bounds_violation": int(bounds_violation),
            }
        )
        proposed_controls.append(proposed)
        executed_controls.append(executed)
        compute_times.append(compute_ms)
        states.append(next_state)
        state = next_state
        previous_prior.update(updated)
        nominal_sequence = previous_prior.mean_control_sequence(
            state, int(planner["horizon"])
        )
        if _goal_distance(state) <= float(benchmark["goal_tolerance"]) or bounds_violation:
            break

    success = _goal_distance(state) <= float(benchmark["goal_tolerance"]) and not bounds_violation
    path_length = sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(states[:-1], states[1:])
    )
    control_difference = max(
        (
            max(abs(a - b) for a, b in zip(proposed, executed))
            for proposed, executed in zip(proposed_controls, executed_controls)
        ),
        default=0.0,
    )
    executed_cost, collided = mppi.executed_trajectory_cost(
        states,
        executed_controls,
        GOAL,
        [],
        float(benchmark["robot_radius"]),
        benchmark["bounds"],
    )
    summary = {
        "run_type": config["run"]["run_type"],
        "mode": mode,
        "seed": seed,
        "success": int(success),
        "bounds_violation": int(bounds_violation or collided),
        "steps": len(executed_controls),
        "duration_seconds": len(executed_controls) * float(benchmark["dt"]),
        "final_goal_distance": _goal_distance(state),
        "path_length": float(path_length),
        "executed_cost": float(executed_cost),
        "control_effort": float(
            sum(v * v + omega * omega for v, omega in executed_controls)
            * float(benchmark["dt"])
        ),
        "yaw_effort": float(
            sum(abs(omega) for _, omega in executed_controls) * float(benchmark["dt"])
        ),
        "mean_planner_compute_ms": float(statistics.mean(compute_times)) if compute_times else 0.0,
        "p95_planner_compute_ms": _percentile(compute_times, 95.0),
        "max_planner_compute_ms": max(compute_times) if compute_times else 0.0,
        "total_planner_compute_ms": sum(compute_times),
        "proposed_executed_max_abs_difference": control_difference,
        "git_sha": git_sha,
    }
    detail = dict(summary)
    detail.update(
        {
            "initial_state": list(states[0]),
            "final_state": list(states[-1]),
            "states": [list(item) for item in states],
            "proposed_controls": [list(item) for item in proposed_controls],
            "executed_controls": [list(item) for item in executed_controls],
            "planner_compute_ms": list(compute_times),
            "memory_enabled": False,
            "obstacle_free": True,
        }
    )
    return summary, rows, detail


def aggregate_summaries(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    modes = sorted({str(row["mode"]) for row in rows})
    numeric = (
        "steps",
        "final_goal_distance",
        "path_length",
        "executed_cost",
        "control_effort",
        "mean_planner_compute_ms",
        "p95_planner_compute_ms",
    )
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        entry: Dict[str, Any] = {
            "episodes": len(selected),
            "success_rate": sum(float(row["success"]) for row in selected) / len(selected),
        }
        for name in numeric:
            values = [float(row[name]) for row in selected]
            entry["{}_mean".format(name)] = float(statistics.mean(values))
            entry["{}_population_std".format(name)] = float(statistics.pstdev(values))
        result[mode] = entry
    return result


def run_modes(
    config: Mapping[str, Any],
    planner_modes: Mapping[str, MppiDynamicsAdapter],
    identity_check: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """Run fixed seeds for each mode and atomically publish all artifacts."""

    git_sha = get_git_sha(REPOSITORY_ROOT)
    summaries: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    for mode in sorted(planner_modes):
        for seed in config["run"]["effective_seeds"]:
            summary, rows, detail = run_episode(
                mode, planner_modes[mode], int(seed), config, git_sha
            )
            summaries.append(summary)
            trajectories.extend(rows)
            episodes.append(detail)
            print(
                "mode={} seed={} success={} final_distance={:.4f} steps={} mean_plan_ms={:.3f}".format(
                    mode,
                    seed,
                    bool(summary["success"]),
                    summary["final_goal_distance"],
                    summary["steps"],
                    summary["mean_planner_compute_ms"],
                )
            )
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / filename for name, filename in ARTIFACT_FILENAMES.items()}
    metrics = {
        "schema_version": 1,
        "benchmark": "clean_obstacle_free_residual_mppi",
        "run_type": config["run"]["run_type"],
        "generated_at_utc": _utc_now(),
        "git_sha": git_sha,
        "goal": list(GOAL),
        "obstacle_free": True,
        "memory_enabled": False,
        "formal_results": False if config["run"]["run_type"] == "smoke" else None,
        "smoke_disclaimer": (
            "Smoke results validate wiring and reproducibility only; they are not formal results."
            if config["run"]["run_type"] == "smoke"
            else None
        ),
        "identity_check": dict(identity_check or {}),
        "provenance": dict(provenance or {}),
        "aggregate_by_mode": aggregate_summaries(summaries),
        "episodes": episodes,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    snapshot = copy.deepcopy(dict(config))
    snapshot["provenance"] = {
        "git_sha": git_sha,
        "generated_at_utc": metrics["generated_at_utc"],
        **dict(provenance or {}),
    }
    _atomic_write_text(
        paths["metrics"],
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_csv(paths["summary"], summaries, SUMMARY_FIELDS)
    _write_csv(paths["trajectory"], trajectories, TRAJECTORY_FIELDS)
    _atomic_write_text(
        paths["config"], yaml.safe_dump(_json_safe(snapshot), sort_keys=False)
    )
    print("metrics={}".format(paths["metrics"]))
    print("summary={}".format(paths["summary"]))
    print("trajectory={}".format(paths["trajectory"]))
    print("config_snapshot={}".format(paths["config"]))
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="run fixed reduced smoke seeds/settings")
    parser.add_argument("--headless", action="store_true", help="force a non-interactive plotting backend")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = resolve_config(
            load_config(args.config),
            args.config,
            output_dir=args.output_dir,
            smoke=bool(args.smoke),
            headless=bool(args.headless),
        )
        modes, context = build_oracle_modes(config)
        identity = verify_oracle_identity(config, context["oracle_adapter"])
        run_modes(
            config,
            modes,
            identity_check=identity,
            provenance={
                "planner_modes": ["nominal_mismatch", "oracle_residual"],
                "oracle_access_to_true_parameters": True,
                "true_parameters_passed_to_learned_planner": False,
            },
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
