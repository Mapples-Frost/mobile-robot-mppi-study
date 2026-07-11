#!/usr/bin/env python3
"""Collect reproducible synthetic residual-dynamics transition datasets.

The default run mixes piecewise-constant random exploration with an honest,
obstacle-free proportional goal tracker.  It does not claim that the latter is
learned MPPI on-policy data.  A future aggregation loop can pass a policy
callback and an explicit model version to :func:`collect_dataset`.

Each transition retains both the commanded control and the control actually
applied by the disturbed plant.  The residual label is deliberately

``wrapped_finite_difference(x_t, x_t_plus_1) - f_nom(x_t, commanded_u_t)``.

Consequently a held-out delay stress case is visible in the target even though
the current three-state learner has no command-history input.  Seen training
disturbances do not use delay, avoiding irreducible non-Markov training noise.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.dynamics import (  # noqa: E402
    DisturbanceConfig,
    DisturbedUnicycle,
    NominalUnicycle,
    wrap_angle,
)
from src.learning.residual_dataset import (  # noqa: E402
    ResidualDataset,
    wrapped_finite_difference,
)


DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "icode" / "data_collection.yaml"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "icode" / "datasets"

RANDOM_SOURCE = "random_exploration"
TASK_SOURCE = "task_specific_goal_tracking"
DATA_SOURCES = (RANDOM_SOURCE, TASK_SOURCE)
DOMAINS = ("seen", "unseen")
ANGLE_INDICES = (2,)

PolicyCallback = Callable[[np.ndarray, Mapping[str, Any]], Sequence[float]]

_SUMMARY_FIELDS = (
    "split",
    "domain",
    "episode_id",
    "seed",
    "data_source",
    "controller",
    "model_version",
    "disturbance_type",
    "disturbance_parameters",
    "scene",
    "transitions",
    "dt",
    "integrator",
    "start_x",
    "start_y",
    "start_theta",
    "final_x",
    "final_y",
    "final_theta",
    "goal_x",
    "goal_y",
    "final_goal_distance",
    "residual_rmse",
    "max_residual_l2",
    "command_applied_mismatch_steps",
)

_MANAGED_OUTPUTS = {
    "all.npz",
    "all.metadata.json",
    "train.npz",
    "train.metadata.json",
    "validation.npz",
    "validation.metadata.json",
    "test.npz",
    "test.metadata.json",
    "unseen.npz",
    "unseen.metadata.json",
    "episode_summary.csv",
    "config_snapshot.yaml",
    "run_manifest.json",
    "all.summary.csv",
    "train.summary.csv",
    "validation.summary.csv",
    "test.summary.csv",
    "unseen.summary.csv",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_value(*args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git"] + list(args),
            cwd=str(REPOSITORY_ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _provenance(config_path: Optional[Path], run_type: str) -> Dict[str, Any]:
    return {
        "created_at_utc": _utc_now(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "repository_root": str(REPOSITORY_ROOT),
        "collector": "experiments/icode/collect_synthetic_residual_data.py",
        "config_path": None if config_path is None else str(config_path.resolve()),
        "run_type": run_type,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("{} must be a mapping".format(name))
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("{} must be a non-negative integer".format(name))
    result = int(value)
    if result < 0:
        raise ValueError("{} must be non-negative".format(name))
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError("{} must be positive".format(name))
    return result


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError("{} must be a positive finite scalar".format(name))
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("{} must be a positive finite scalar".format(name))
    return result


def _range_pair(value: Any, name: str) -> Tuple[float, float]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError("{} must contain [low, high]".format(name)) from exc
    if len(values) != 2:
        raise ValueError("{} must contain exactly [low, high]".format(name))
    low, high = float(values[0]), float(values[1])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError("{} must have finite low < high".format(name))
    return low, high


def _disturbance_case(value: Any, name: str) -> Tuple[str, DisturbanceConfig]:
    case = dict(_as_mapping(value, name))
    disturbance_id = case.pop("id", None)
    if not isinstance(disturbance_id, str) or not disturbance_id.strip():
        raise ValueError("{}.id must be a non-empty string".format(name))
    try:
        disturbance = DisturbanceConfig(**case)
    except (TypeError, ValueError) as exc:
        raise type(exc)("invalid {}: {}".format(name, exc)) from exc
    return disturbance_id.strip(), disturbance


def validate_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and deep-copy a collection configuration."""

    result = copy.deepcopy(dict(_as_mapping(config, "config")))
    master_seed = _nonnegative_int(result.get("master_seed"), "master_seed")
    if master_seed > np.iinfo(np.uint32).max:
        raise ValueError("master_seed must fit in uint32")

    collection = _as_mapping(result.get("collection"), "collection")
    _positive_float(collection.get("dt"), "collection.dt")
    integrator = collection.get("integrator")
    if not isinstance(integrator, str) or integrator.strip().lower() not in ("euler", "rk4"):
        raise ValueError("collection.integrator must be 'euler' or 'rk4'")

    episodes = _as_mapping(collection.get("episodes"), "collection.episodes")
    total_episodes = 0
    for domain in DOMAINS:
        domain_counts = _as_mapping(episodes.get(domain), "collection.episodes.{}".format(domain))
        for source in DATA_SOURCES:
            total_episodes += _nonnegative_int(
                domain_counts.get(source),
                "collection.episodes.{}.{}".format(domain, source),
            )
    if total_episodes == 0:
        raise ValueError("configuration must request at least one episode")

    steps = _as_mapping(collection.get("steps_per_episode"), "collection.steps_per_episode")
    for source in DATA_SOURCES:
        _positive_int(steps.get(source), "collection.steps_per_episode.{}".format(source))

    state_ranges = _as_mapping(result.get("state_ranges"), "state_ranges")
    for source in DATA_SOURCES:
        source_ranges = _as_mapping(state_ranges.get(source), "state_ranges.{}".format(source))
        for state_name in ("x", "y", "theta"):
            _range_pair(source_ranges.get(state_name), "state_ranges.{}.{}".format(source, state_name))

    control_ranges = _as_mapping(result.get("control_ranges"), "control_ranges")
    velocity_range = _range_pair(control_ranges.get("velocity"), "control_ranges.velocity")
    _range_pair(control_ranges.get("yaw_rate"), "control_ranges.yaw_rate")
    if velocity_range[0] < 0.0:
        raise ValueError("control_ranges.velocity must be non-negative for MPPI compatibility")

    goal = np.asarray(result.get("goal"), dtype=np.float64)
    if goal.shape != (2,) or not np.all(np.isfinite(goal)):
        raise ValueError("goal must be a finite [x, y] pair")

    random_options = _as_mapping(result.get(RANDOM_SOURCE), RANDOM_SOURCE)
    hold_low, hold_high = random_options.get("command_hold_steps", (None, None))
    hold_low = _positive_int(hold_low, "random_exploration.command_hold_steps[0]")
    hold_high = _positive_int(hold_high, "random_exploration.command_hold_steps[1]")
    if hold_low > hold_high:
        raise ValueError("random_exploration.command_hold_steps must have low <= high")

    task_options = _as_mapping(result.get(TASK_SOURCE), TASK_SOURCE)
    if task_options.get("controller") != "proportional_goal_tracker":
        raise ValueError("the built-in task controller must be labeled proportional_goal_tracker")
    if not isinstance(task_options.get("controller_version"), str) or not task_options.get("controller_version"):
        raise ValueError("task_specific_goal_tracking.controller_version must be non-empty")
    _positive_float(task_options.get("distance_gain"), "task_specific_goal_tracking.distance_gain")
    _positive_float(task_options.get("heading_gain"), "task_specific_goal_tracking.heading_gain")
    _positive_float(task_options.get("goal_tolerance"), "task_specific_goal_tracking.goal_tolerance")

    split = _as_mapping(result.get("split"), "split")
    split_fractions = []
    for key in ("train_fraction", "validation_fraction", "test_fraction"):
        value = float(split.get(key))
        if not np.isfinite(value) or value <= 0.0 or value >= 1.0:
            raise ValueError("split.{} must lie strictly between zero and one".format(key))
        split_fractions.append(value)
    if not math.isclose(sum(split_fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("train/validation/test fractions must sum to one")

    train_cases = result.get("train_disturbances")
    unseen_cases = result.get("unseen_disturbances")
    if not isinstance(train_cases, list) or not train_cases:
        raise ValueError("train_disturbances must be a non-empty list")
    if not isinstance(unseen_cases, list) or not unseen_cases:
        raise ValueError("unseen_disturbances must be a non-empty list")
    parsed_train = [_disturbance_case(case, "train_disturbances[{}]".format(index)) for index, case in enumerate(train_cases)]
    parsed_unseen = [_disturbance_case(case, "unseen_disturbances[{}]".format(index)) for index, case in enumerate(unseen_cases)]
    duplicate_ids = set(item[0] for item in parsed_train) & set(item[0] for item in parsed_unseen)
    if duplicate_ids:
        raise ValueError("disturbance ids must be unique across seen and unseen: {}".format(sorted(duplicate_ids)))
    delayed_seen = [item[0] for item in parsed_train if item[1].control_delay_steps > 0 and item[1].enable_control_delay]
    if delayed_seen:
        raise ValueError(
            "seen training disturbances may not use control delay because the current learner has no command history: {}".format(delayed_seen)
        )
    return result


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    return validate_config(loaded)


def _smoke_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a tiny config that still exercises all disturbances and splits."""

    result = copy.deepcopy(dict(config))
    seen_cases = len(result["train_disturbances"])
    unseen_cases = len(result["unseen_disturbances"])
    for source in DATA_SOURCES:
        result["collection"]["episodes"]["seen"][source] = 3 * seen_cases
        result["collection"]["episodes"]["unseen"][source] = unseen_cases
        result["collection"]["steps_per_episode"][source] = min(
            int(result["collection"]["steps_per_episode"][source]), 8
        )
    return validate_config(result)


def _episode_seed(
    master_seed: int,
    domain_index: int,
    source_index: int,
    disturbance_index: int,
    episode_index: int,
) -> int:
    sequence = np.random.SeedSequence(
        [master_seed, domain_index, source_index, disturbance_index, episode_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _state_bounds(config: Mapping[str, Any], source: str) -> Tuple[np.ndarray, np.ndarray]:
    ranges = config["state_ranges"][source]
    pairs = [_range_pair(ranges[name], "state range") for name in ("x", "y", "theta")]
    return (
        np.asarray([pair[0] for pair in pairs], dtype=np.float64),
        np.asarray([pair[1] for pair in pairs], dtype=np.float64),
    )


def _control_bounds(config: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    velocity = _range_pair(config["control_ranges"]["velocity"], "velocity range")
    yaw_rate = _range_pair(config["control_ranges"]["yaw_rate"], "yaw-rate range")
    return (
        np.asarray([velocity[0], yaw_rate[0]], dtype=np.float64),
        np.asarray([velocity[1], yaw_rate[1]], dtype=np.float64),
    )


def _goal_tracking_control(
    state: np.ndarray,
    goal: np.ndarray,
    options: Mapping[str, Any],
) -> np.ndarray:
    delta = goal - state[:2]
    distance = float(np.linalg.norm(delta))
    if distance <= float(options["goal_tolerance"]):
        return np.zeros(2, dtype=np.float64)
    desired_heading = math.atan2(float(delta[1]), float(delta[0]))
    heading_error = float(wrap_angle(desired_heading - state[2]))
    # Suppressing forward motion while facing away produces a clean rotate-then-go
    # trajectory and respects the repository's non-negative velocity convention.
    alignment = max(0.0, math.cos(heading_error))
    return np.asarray(
        [
            float(options["distance_gain"]) * distance * alignment,
            float(options["heading_gain"]) * heading_error,
        ],
        dtype=np.float64,
    )


def _policy_control(
    policy: PolicyCallback,
    state: np.ndarray,
    context: Mapping[str, Any],
) -> np.ndarray:
    try:
        command = np.asarray(policy(state.copy(), context), dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(
            "policy callback failed in episode {!r} at step {}".format(
                context["episode_id"], context["step"]
            )
        ) from exc
    if command.shape != (2,) or not np.all(np.isfinite(command)):
        raise ValueError("policy callback must return a finite control with shape (2,)")
    return command


def _empty_records() -> Dict[str, List[Any]]:
    return {
        "episode_id": [],
        "seed": [],
        "step": [],
        "time": [],
        "dt": [],
        "state_t": [],
        "control_t": [],
        "applied_control_t": [],
        "state_t_plus_1": [],
        "nominal_derivative": [],
        "observed_derivative": [],
        "residual_target": [],
        "disturbance_type": [],
        "disturbance_parameters": [],
        "scene": [],
        "data_source": [],
        "model_version": [],
    }


def _safe_prefix(value: str) -> str:
    if not value:
        return ""
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-") + "_"


def _collect_episode(
    records: Dict[str, List[Any]],
    config: Mapping[str, Any],
    domain: str,
    source: str,
    episode_index: int,
    disturbance_index: int,
    disturbance_id: str,
    disturbance: DisturbanceConfig,
    seed: int,
    episode_id_prefix: str,
    policy: Optional[PolicyCallback],
    task_model_version: str,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    state_low, state_high = _state_bounds(config, source)
    control_low, control_high = _control_bounds(config)
    state = rng.uniform(state_low, state_high).astype(np.float64)
    initial_state = state.copy()
    nominal = NominalUnicycle()
    # Instantiate and explicitly reset every episode so delay history can never
    # leak across episode boundaries, including the unseen delay stress case.
    plant = DisturbedUnicycle(config=disturbance, nominal_dynamics=nominal)
    plant.reset(initial_control=np.zeros(2, dtype=np.float64))

    episode_id = "{}{}_{}_d{:02d}_e{:05d}".format(
        _safe_prefix(episode_id_prefix), domain, source, disturbance_index, episode_index
    )
    steps = int(config["collection"]["steps_per_episode"][source])
    dt = float(config["collection"]["dt"])
    integrator = str(config["collection"]["integrator"]).lower()
    goal = np.asarray(config["goal"], dtype=np.float64)
    task_options = config[TASK_SOURCE]
    random_options = config[RANDOM_SOURCE]
    disturbance_parameters = disturbance.to_dict()
    scene = (
        "synthetic_free_space_random_exploration"
        if source == RANDOM_SOURCE
        else "clean_obstacle_free_goal_tracking"
    )
    if source == RANDOM_SOURCE:
        controller = "piecewise_constant_random_exploration"
        transition_model_version = "not_applicable_random_exploration"
    elif policy is None:
        controller = "proportional_goal_tracker"
        transition_model_version = task_model_version
    else:
        controller = "external_policy_callback"
        transition_model_version = task_model_version

    command = np.zeros(2, dtype=np.float64)
    hold_remaining = 0
    residuals = []
    mismatch_steps = 0
    for step in range(steps):
        time = float(step) * dt
        if source == RANDOM_SOURCE:
            if hold_remaining <= 0:
                command = rng.uniform(control_low, control_high).astype(np.float64)
                hold_low, hold_high = [int(value) for value in random_options["command_hold_steps"]]
                hold_remaining = int(rng.integers(hold_low, hold_high + 1))
            hold_remaining -= 1
        elif policy is None:
            command = _goal_tracking_control(state, goal, task_options)
        else:
            context = {
                "goal": goal.copy(),
                "step": step,
                "time": time,
                "dt": dt,
                "rng": rng,
                "episode_id": episode_id,
                "episode_seed": seed,
                "disturbance_type": disturbance_id,
                "control_low": control_low.copy(),
                "control_high": control_high.copy(),
                "model_version": task_model_version,
            }
            command = _policy_control(policy, state, context)
        command = np.clip(command, control_low, control_high)

        applied_control = plant.preview_applied_control(command)
        next_state = plant.step(state, command, dt, method=integrator, time=time)
        if not np.array_equal(plant.last_applied_control, applied_control):
            raise RuntimeError("plant applied-control preview disagrees with executed control")
        nominal_derivative = nominal.derivative(state, command, time=time)
        observed_derivative = wrapped_finite_difference(
            state,
            next_state,
            dt,
            angle_indices=ANGLE_INDICES,
        )
        residual_target = observed_derivative - nominal_derivative

        records["episode_id"].append(episode_id)
        records["seed"].append(seed)
        records["step"].append(step)
        records["time"].append(time)
        records["dt"].append(dt)
        records["state_t"].append(state.copy())
        records["control_t"].append(command.copy())
        records["applied_control_t"].append(applied_control.copy())
        records["state_t_plus_1"].append(next_state.copy())
        records["nominal_derivative"].append(nominal_derivative.copy())
        records["observed_derivative"].append(observed_derivative.copy())
        records["residual_target"].append(residual_target.copy())
        records["disturbance_type"].append(disturbance_id)
        records["disturbance_parameters"].append(disturbance_parameters)
        records["scene"].append(scene)
        records["data_source"].append(source)
        records["model_version"].append(transition_model_version)

        residuals.append(residual_target)
        mismatch_steps += int(not np.allclose(command, applied_control, rtol=0.0, atol=1.0e-14))
        state = next_state

    residual_array = np.asarray(residuals, dtype=np.float64)
    goal_distance = float(np.linalg.norm(goal - state[:2])) if source == TASK_SOURCE else None
    return {
        "split": "unassigned",
        "domain": domain,
        "episode_id": episode_id,
        "seed": seed,
        "data_source": source,
        "controller": controller,
        "model_version": transition_model_version,
        "disturbance_type": disturbance_id,
        "disturbance_parameters": json.dumps(disturbance_parameters, sort_keys=True),
        "scene": scene,
        "transitions": steps,
        "dt": dt,
        "integrator": integrator,
        "start_x": float(initial_state[0]),
        "start_y": float(initial_state[1]),
        "start_theta": float(initial_state[2]),
        "final_x": float(state[0]),
        "final_y": float(state[1]),
        "final_theta": float(state[2]),
        "goal_x": float(goal[0]) if source == TASK_SOURCE else None,
        "goal_y": float(goal[1]) if source == TASK_SOURCE else None,
        "final_goal_distance": goal_distance,
        "residual_rmse": float(np.sqrt(np.mean(np.square(residual_array)))),
        "max_residual_l2": float(np.max(np.linalg.norm(residual_array, axis=1))),
        "command_applied_mismatch_steps": mismatch_steps,
    }


def _records_to_dataset(records: Mapping[str, Sequence[Any]], metadata: Mapping[str, Any]) -> ResidualDataset:
    arrays = {
        "episode_id": np.asarray(records["episode_id"], dtype=np.str_),
        "seed": np.asarray(records["seed"], dtype=np.int64),
        "step": np.asarray(records["step"], dtype=np.int64),
        "time": np.asarray(records["time"], dtype=np.float64),
        "dt": np.asarray(records["dt"], dtype=np.float64),
        "state_t": np.asarray(records["state_t"], dtype=np.float64),
        "control_t": np.asarray(records["control_t"], dtype=np.float64),
        "applied_control_t": np.asarray(records["applied_control_t"], dtype=np.float64),
        "state_t_plus_1": np.asarray(records["state_t_plus_1"], dtype=np.float64),
        "nominal_derivative": np.asarray(records["nominal_derivative"], dtype=np.float64),
        "observed_derivative": np.asarray(records["observed_derivative"], dtype=np.float64),
        "residual_target": np.asarray(records["residual_target"], dtype=np.float64),
        "disturbance_type": np.asarray(records["disturbance_type"], dtype=np.str_),
        "disturbance_parameters": list(records["disturbance_parameters"]),
        "scene": np.asarray(records["scene"], dtype=np.str_),
        "data_source": np.asarray(records["data_source"], dtype=np.str_),
        "model_version": np.asarray(records["model_version"], dtype=np.str_),
    }
    return ResidualDataset.from_mapping(arrays, metadata=dict(metadata))


def _fresh_collection(
    config: Mapping[str, Any],
    policy: Optional[PolicyCallback],
    model_version: Optional[str],
    episode_id_prefix: str,
    provenance: Mapping[str, Any],
) -> Tuple[ResidualDataset, List[Dict[str, Any]]]:
    validated = validate_config(config)
    if policy is not None and (not isinstance(model_version, str) or not model_version.strip()):
        raise ValueError("model_version is required when policy callback is provided")
    task_model_version = (
        str(validated[TASK_SOURCE]["controller_version"])
        if policy is None
        else str(model_version).strip()
    )
    parsed_cases = {
        "seen": [
            _disturbance_case(case, "train_disturbances")
            for case in validated["train_disturbances"]
        ],
        "unseen": [
            _disturbance_case(case, "unseen_disturbances")
            for case in validated["unseen_disturbances"]
        ],
    }
    records = _empty_records()
    summaries: List[Dict[str, Any]] = []
    master_seed = int(validated["master_seed"])
    for domain_index, domain in enumerate(DOMAINS):
        cases = parsed_cases[domain]
        for source_index, source in enumerate(DATA_SOURCES):
            episode_count = int(validated["collection"]["episodes"][domain][source])
            for episode_index in range(episode_count):
                disturbance_index = episode_index % len(cases)
                disturbance_id, disturbance = cases[disturbance_index]
                seed = _episode_seed(
                    master_seed,
                    domain_index,
                    source_index,
                    disturbance_index,
                    episode_index,
                )
                summaries.append(
                    _collect_episode(
                        records=records,
                        config=validated,
                        domain=domain,
                        source=source,
                        episode_index=episode_index,
                        disturbance_index=disturbance_index,
                        disturbance_id=disturbance_id,
                        disturbance=disturbance,
                        seed=seed,
                        episode_id_prefix=episode_id_prefix,
                        policy=policy,
                        task_model_version=task_model_version,
                    )
                )

    has_unseen_delay = any(
        disturbance.control_delay_steps > 0 and disturbance.enable_control_delay
        for _, disturbance in parsed_cases["unseen"]
    )
    metadata = {
        "schema_version": int(validated.get("schema_version", 1)),
        "dataset_kind": "synthetic_unicycle_residual_transitions",
        "master_seed": master_seed,
        "state_dim": 3,
        "control_dim": 2,
        "angle_indices": list(ANGLE_INDICES),
        "episode_count": len(summaries),
        "transition_count": len(records["episode_id"]),
        "provenance": dict(provenance),
        "config": validated,
        "sources": {
            RANDOM_SOURCE: "piecewise-constant random exploration",
            TASK_SOURCE: (
                "obstacle-free proportional goal tracker; not learned MPPI on-policy"
                if policy is None
                else "external task policy callback tagged by model_version"
            ),
        },
        "transition_semantics": {
            "control_t": "commanded control",
            "applied_control_t": "control executed after plant delay",
            "observed_derivative": "finite difference with wrapped theta delta divided by dt",
            "nominal_derivative": "NominalUnicycle derivative at state_t and commanded control_t",
            "residual_target": "observed_derivative minus nominal_derivative(commanded control_t)",
        },
        "control_delay_limitation": {
            "seen_training_has_delay": False,
            "unseen_stress_has_delay": has_unseen_delay,
            "learner_state_contains_command_history": False,
            "note": "delay is an intentionally non-Markov held-out stress case for [x,y,theta] inputs",
        },
    }
    return _records_to_dataset(records, metadata), summaries


def _append_datasets(existing: ResidualDataset, new: ResidualDataset) -> ResidualDataset:
    old_mapping = existing.as_dict()
    new_mapping = new.as_dict()
    old_ids = set(np.asarray(old_mapping["episode_id"]).astype(str).tolist())
    new_ids = set(np.asarray(new_mapping["episode_id"]).astype(str).tolist())
    duplicates = old_ids & new_ids
    if duplicates:
        example = sorted(duplicates)[0]
        raise ValueError(
            "cannot append duplicate episode ids (for example {!r}); use a distinct episode_id_prefix or master_seed".format(example)
        )
    combined = {
        field: np.concatenate((np.asarray(old_mapping[field]), np.asarray(new_mapping[field])), axis=0)
        for field in old_mapping
    }
    metadata = copy.deepcopy(getattr(existing, "metadata", {}))
    appended_metadata = copy.deepcopy(getattr(new, "metadata", {}))
    aggregation_entry = {
        "previous_transition_count": len(existing),
        "appended_transition_count": len(new),
        "appended_episode_count": len(new_ids),
        "appended_at_utc": _utc_now(),
        "appended_model_versions": sorted(set(new.model_version.astype(str).tolist())),
        "appended_provenance": appended_metadata.get("provenance", {}),
        "appended_config": appended_metadata.get("config", {}),
    }
    history = list(metadata.get("aggregation_history", []))
    history.append(aggregation_entry)
    metadata.update(
        {
            "aggregation": aggregation_entry,
            "aggregation_history": history,
            "transition_count": len(existing) + len(new),
            "episode_count": len(old_ids) + len(new_ids),
        }
    )
    return ResidualDataset.from_mapping(combined, metadata=metadata)


def collect_dataset(
    config: Mapping[str, Any],
    policy: Optional[PolicyCallback] = None,
    model_version: Optional[str] = None,
    existing_dataset: Optional[ResidualDataset] = None,
    episode_id_prefix: str = "",
) -> ResidualDataset:
    """Collect data for reuse by training or future on-policy aggregation.

    ``policy`` replaces only the task-specific proportional controller and is
    called as ``policy(state, context)``.  Its context contains the goal, time,
    step, seeded RNG, bounds, episode identifiers, and disturbance label.
    ``model_version`` is mandatory for an external policy.  Pass
    ``existing_dataset`` plus a distinct ``episode_id_prefix`` to append a new
    aggregation round without episode-id collisions.
    """

    dataset, _ = _fresh_collection(
        config=config,
        policy=policy,
        model_version=model_version,
        episode_id_prefix=episode_id_prefix,
        provenance=_provenance(config_path=None, run_type="library"),
    )
    if existing_dataset is None:
        return dataset
    if not isinstance(existing_dataset, ResidualDataset):
        raise TypeError("existing_dataset must be a ResidualDataset")
    return _append_datasets(existing_dataset, dataset)


def _split_counts(count: int, fractions: Sequence[float]) -> List[int]:
    if count < 0:
        raise ValueError("count must be non-negative")
    positive = sum(fraction > 0.0 for fraction in fractions)
    if count >= positive:
        base = [1 if fraction > 0.0 else 0 for fraction in fractions]
        remaining = count - sum(base)
    else:
        base = [0 for _ in fractions]
        remaining = count
    raw = np.asarray(fractions, dtype=np.float64) * remaining
    floors = np.floor(raw).astype(np.int64)
    result = [base[index] + int(floors[index]) for index in range(len(fractions))]
    leftover = count - sum(result)
    remainder_order = np.argsort(-(raw - floors), kind="stable")
    for index in remainder_order[:leftover]:
        result[int(index)] += 1
    return result


def _text_seed(master_seed: int, text_value: str) -> int:
    digest = hashlib.sha256(text_value.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int(np.random.SeedSequence([master_seed, value]).generate_state(1, dtype=np.uint32)[0])


def _episode_splits(
    summaries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, List[str]]:
    split_options = config["split"]
    fractions = (
        float(split_options["train_fraction"]),
        float(split_options["validation_fraction"]),
        float(split_options["test_fraction"]),
    )
    result = {"train": [], "validation": [], "test": [], "unseen": []}
    grouped: Dict[Tuple[str, str], List[str]] = {}
    for summary in summaries:
        if summary["domain"] == "unseen":
            result["unseen"].append(str(summary["episode_id"]))
            continue
        key = (str(summary["data_source"]), str(summary["disturbance_type"]))
        grouped.setdefault(key, []).append(str(summary["episode_id"]))
    for key in sorted(grouped):
        episode_ids = np.asarray(sorted(grouped[key]), dtype=np.str_)
        rng = np.random.default_rng(_text_seed(int(config["master_seed"]), "{}|{}".format(*key)))
        episode_ids = episode_ids[rng.permutation(len(episode_ids))]
        train_count, validation_count, test_count = _split_counts(len(episode_ids), fractions)
        first = train_count
        second = first + validation_count
        third = second + test_count
        result["train"].extend(episode_ids[:first].tolist())
        result["validation"].extend(episode_ids[first:second].tolist())
        result["test"].extend(episode_ids[second:third].tolist())
    for name in result:
        result[name] = sorted(result[name])
    assigned_seen = set(result["train"]) | set(result["validation"]) | set(result["test"])
    expected_seen = {str(item["episode_id"]) for item in summaries if item["domain"] == "seen"}
    if assigned_seen != expected_seen:
        raise RuntimeError("episode split did not cover every seen episode exactly once")
    if (
        set(result["train"]) & set(result["validation"])
        or set(result["train"]) & set(result["test"])
        or set(result["validation"]) & set(result["test"])
        or assigned_seen & set(result["unseen"])
    ):
        raise RuntimeError("episode leakage detected between dataset splits")
    return result


def _subset_with_metadata(
    dataset: ResidualDataset,
    episode_ids: Iterable[str],
    split_name: str,
) -> ResidualDataset:
    mapping = dataset.as_dict()
    requested = set(episode_ids)
    mask = np.asarray([str(value) in requested for value in mapping["episode_id"]], dtype=bool)
    subset = dataset.subset(np.flatnonzero(mask))
    metadata = copy.deepcopy(getattr(subset, "metadata", {}))
    metadata.update(
        {
            "split": split_name,
            "episode_count": len(requested),
            "transition_count": len(subset),
            "episode_group_split": True,
        }
    )
    return ResidualDataset.from_mapping(subset.as_dict(), metadata=metadata)


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    occupied = list(path.iterdir())
    if occupied and not overwrite:
        raise FileExistsError(
            "output directory is not empty: {} (pass --overwrite to replace collection artifacts)".format(path)
        )
    if overwrite:
        # Preserve unrelated user files even when overwrite is requested.
        for name in _MANAGED_OUTPUTS:
            target = path / name
            if target.is_file() or target.is_symlink():
                target.unlink()


def _write_episode_summary(
    path: Path,
    summaries: Sequence[Dict[str, Any]],
    split_ids: Mapping[str, Sequence[str]],
) -> None:
    split_by_episode = {
        episode_id: split_name
        for split_name, episode_ids in split_ids.items()
        for episode_id in episode_ids
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(_SUMMARY_FIELDS))
        writer.writeheader()
        for summary in sorted(summaries, key=lambda item: str(item["episode_id"])):
            row = dict(summary)
            row["split"] = split_by_episode[str(summary["episode_id"])]
            writer.writerow({field: row.get(field) for field in _SUMMARY_FIELDS})


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("cannot serialize {!r}".format(type(value).__name__))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_collection(
    config_path: Path,
    output_dir: Path,
    smoke: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Run collection, episode-group splitting, and artifact serialization."""

    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config = load_config(config_path)
    if smoke:
        config = _smoke_config(config)
    _prepare_output_dir(output_dir, overwrite=overwrite)
    provenance = _provenance(config_path=config_path, run_type="smoke" if smoke else "full")
    dataset, summaries = _fresh_collection(
        config=config,
        policy=None,
        model_version=None,
        episode_id_prefix="",
        provenance=provenance,
    )
    split_ids = _episode_splits(summaries, config)
    datasets = {
        "all": ResidualDataset.from_mapping(
            dataset.as_dict(),
            metadata=dict(
                copy.deepcopy(dataset.metadata),
                split="all_including_unseen",
                episode_split_assignments=copy.deepcopy(split_ids),
            ),
        ),
        "train": _subset_with_metadata(dataset, split_ids["train"], "train"),
        "validation": _subset_with_metadata(dataset, split_ids["validation"], "validation"),
        "test": _subset_with_metadata(dataset, split_ids["test"], "test"),
        "unseen": _subset_with_metadata(dataset, split_ids["unseen"], "unseen_disturbance"),
    }
    for name, split_dataset in datasets.items():
        split_dataset.save(output_dir / "{}.npz".format(name))
    _write_episode_summary(output_dir / "episode_summary.csv", summaries, split_ids)
    with (output_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    artifact_names = sorted(
        path.name for path in output_dir.iterdir() if path.name != "run_manifest.json" and path.is_file()
    )
    manifest = {
        "schema_version": 1,
        "provenance": provenance,
        "output_dir": str(output_dir),
        "master_seed": int(config["master_seed"]),
        "smoke": bool(smoke),
        "dataset_sizes": {name: len(value) for name, value in datasets.items()},
        "episode_counts": {name: len(ids) for name, ids in split_ids.items()},
        "artifacts": {
            name: {"sha256": _sha256_file(output_dir / name)} for name in artifact_names
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="collection YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="collect a tiny deterministic run that still exercises every split",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace collector-owned artifacts while preserving unrelated files",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = run_collection(
        config_path=args.config,
        output_dir=args.output_dir,
        smoke=args.smoke,
        overwrite=args.overwrite,
    )
    sizes = manifest["dataset_sizes"]
    print("Synthetic residual dataset written to {}".format(manifest["output_dir"]))
    print(
        "transitions: all={all}, train={train}, validation={validation}, test={test}, unseen={unseen}".format(
            **sizes
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
