"""Audit complex-map oracle feasibility before supervised Actor training.

Gate v4 is deliberately offline and privileged.  It replays saved development
episodes to a frozen set of states before the first online hard-risk event,
searches deterministic four-stage control sequences with exact MuJoCo future
truth, refines promising sequences with CEM, and only then audits the frozen
online ICODE, Collision Risk, and Safety contracts.

The oracle is a label-feasibility diagnostic.  It is not a deployment planner,
does not alter the paper method, and must never be used on formal test maps.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.audit_complex_privileged_teacher_gate import (
    _anchor_index,
    _icode_controller,
    _read_rows,
    _replay_to_anchor,
    _risk_config,
)
from mobile_robot_mppi.core.config import config_hash, git_sha, load_yaml
from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.evaluation.scene_feasibility import point_clearance
from mobile_robot_mppi.obstacles.collision_risk import evaluate_collision_risk
from mobile_robot_mppi.runtime.factories import make_components


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/research/"
    "complex_supervised_actor_oracle_feasibility_gate_v4.yaml"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("oracle feasibility protocol must be a mapping")
    if value.get("protocol") != (
        "complex_supervised_actor_oracle_feasibility_timing_gate_v1"
    ):
        raise ValueError("oracle feasibility protocol mismatch")
    if bool(value["gate"]["formal_server_launch_authorized"]):
        raise ValueError("oracle gate cannot authorize a formal server")
    if bool(value["frozen_contract"]["maps_changed"]):
        raise ValueError("Gate v4 must use the frozen Gate v3 maps")
    if bool(value["frozen_contract"]["risk_thresholds_changed"]):
        raise ValueError("Gate v4 cannot change Collision Risk thresholds")
    if bool(value["frozen_contract"]["safety_contract_changed"]):
        raise ValueError("Gate v4 cannot change Safety")
    search = value["oracle_search"]
    if int(search["segments"]) != 4:
        raise ValueError("Gate v4 requires four-stage controls")
    if not 2000 <= int(search["coarse_candidate_count"]) <= 5000:
        raise ValueError("coarse search must contain 2000-5000 candidates")
    if sorted(float(item) for item in search["horizons_s"]) != [8.0, 12.0]:
        raise ValueError("Gate v4 must audit both 8 s and 12 s horizons")
    if not 1 <= int(search.get("parallel_workers", 1)) <= 8:
        raise ValueError("parallel_workers must be between 1 and 8")
    return value


def _action_bounds(action_spec):
    lower = np.asarray(action_spec.lower, dtype=np.float64)
    upper = np.asarray(action_spec.upper, dtype=np.float64)
    if lower.shape != (2,) or upper.shape != (2,):
        raise ValueError("oracle expects differential-drive [v, omega]")
    return lower, upper


def _expand_segments(parameters, horizon_steps):
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("four-stage parameters must have shape [N, 8]")
    edges = np.rint(np.linspace(0, horizon_steps, 5)).astype(int)
    result = np.zeros((values.shape[0], horizon_steps, 2), dtype=np.float64)
    shaped = values.reshape(values.shape[0], 4, 2)
    for stage in range(4):
        result[:, edges[stage]:edges[stage + 1], :] = shaped[:, stage, None, :]
    return result


def _family_parameters(family, rng, lower, upper):
    v_reverse = abs(float(lower[0]))
    v_forward = float(upper[0])
    omega_left = float(upper[1])
    omega_right = abs(float(lower[1]))
    forward = rng.uniform(0.22, 0.95, size=4) * v_forward
    small = rng.uniform(-0.15, 0.15) * min(omega_left, omega_right)
    turn = rng.uniform(0.35, 1.0)
    recover = rng.uniform(0.20, 0.75)
    params = np.zeros((4, 2), dtype=np.float64)

    if family == "left":
        params[:, 0] = forward
        params[:, 1] = (
            turn * omega_left,
            rng.uniform(0.15, 0.65) * omega_left,
            -recover * omega_right,
            small,
        )
    elif family == "right":
        params[:, 0] = forward
        params[:, 1] = (
            -turn * omega_right,
            -rng.uniform(0.15, 0.65) * omega_right,
            recover * omega_left,
            small,
        )
    elif family == "yield_left":
        params[0, 0] = rng.uniform(-1.0, 0.10) * v_reverse
        params[1:, 0] = forward[1:]
        params[:, 1] = (
            rng.uniform(-0.20, 0.20) * omega_left,
            turn * omega_left,
            rng.uniform(0.10, 0.55) * omega_left,
            -recover * omega_right,
        )
    elif family == "yield_right":
        params[0, 0] = rng.uniform(-1.0, 0.10) * v_reverse
        params[1:, 0] = forward[1:]
        params[:, 1] = (
            rng.uniform(-0.20, 0.20) * omega_right,
            -turn * omega_right,
            -rng.uniform(0.10, 0.55) * omega_right,
            recover * omega_left,
        )
    elif family == "s_left":
        params[:, 0] = forward
        params[:, 1] = (
            turn * omega_left,
            -rng.uniform(0.25, 0.90) * omega_right,
            rng.uniform(0.10, 0.55) * omega_left,
            small,
        )
    elif family == "s_right":
        params[:, 0] = forward
        params[:, 1] = (
            -turn * omega_right,
            rng.uniform(0.25, 0.90) * omega_left,
            -rng.uniform(0.10, 0.55) * omega_right,
            small,
        )
    else:
        raise ValueError("unknown oracle behavior family: %s" % family)
    return np.clip(params.reshape(-1), np.tile(lower, 4), np.tile(upper, 4))


def _structured_parameters(action_spec, count, seed):
    lower, upper = _action_bounds(action_spec)
    families = (
        "left",
        "right",
        "yield_left",
        "yield_right",
        "s_left",
        "s_right",
    )
    rng = np.random.default_rng(int(seed))
    values = []
    labels = []
    for index in range(int(count)):
        family = families[index % len(families)]
        values.append(_family_parameters(family, rng, lower, upper))
        labels.append(family)

    # Deterministic canonical maneuvers occupy the first slots.  The remaining
    # candidates retain stratified family balance under the frozen RNG seed.
    v_max = float(upper[0])
    w_left = float(upper[1])
    w_right = abs(float(lower[1]))
    canonical = (
        ("left", (0.35, 0.90, 0.55, 0.20, 0.60, -0.45, 0.65, 0.00)),
        ("right", (0.35, -0.85, 0.55, -0.20, 0.60, 0.45, 0.65, 0.00)),
        ("yield_left", (-0.25, 0.00, 0.30, 0.80, 0.55, 0.25, 0.65, -0.30)),
        ("yield_right", (-0.25, 0.00, 0.30, -0.80, 0.55, -0.25, 0.65, 0.30)),
    )
    scale = np.asarray(
        (v_max, w_left, v_max, w_left, v_max, w_left, v_max, w_left),
        dtype=np.float64,
    )
    for index, (label, normalized) in enumerate(canonical):
        raw = np.asarray(normalized, dtype=np.float64) * scale
        values[index] = np.clip(raw, np.tile(lower, 4), np.tile(upper, 4))
        labels[index] = label
    return tuple(labels), np.asarray(values, dtype=np.float64)


def _minimum_dynamic_clearance(truth, plant, dynamic_radii, robot_radius):
    xy = np.asarray((truth.pose.x, truth.pose.y), dtype=np.float64)
    values = []
    for center, radius in zip(plant.dynamic_obstacle_states(), dynamic_radii):
        center_xy = np.asarray((center["x"], center["y"]), dtype=np.float64)
        values.append(
            float(np.linalg.norm(xy - center_xy))
            - float(robot_radius)
            - float(radius)
        )
    return min(values, default=float("inf"))


def _true_branch_metrics(
    plant,
    snapshot,
    candidates,
    dt,
    static_obstacles,
    dynamic_radii,
    robot_radius,
    target_xy,
    stopping_tail_steps,
):
    count, horizon, _ = candidates.shape
    collision = np.zeros(count, dtype=bool)
    tail_collision = np.zeros(count, dtype=bool)
    progress = np.zeros(count, dtype=np.float64)
    static_clearance = np.full(count, np.inf, dtype=np.float64)
    dynamic_clearance = np.full(count, np.inf, dtype=np.float64)
    tail_dynamic_clearance = np.full(count, np.inf, dtype=np.float64)
    terminal_dynamic_clearance = np.full(count, np.inf, dtype=np.float64)
    start_truth = plant.restore(snapshot)
    start_xy = np.asarray(
        (start_truth.pose.x, start_truth.pose.y), dtype=np.float64
    )
    start_distance = float(np.linalg.norm(start_xy - target_xy))

    for candidate_index, sequence in enumerate(candidates):
        truth = plant.restore(snapshot)
        for controls in sequence:
            transition = plant.step(
                ControlCommand(
                    controls,
                    plant.time,
                    "oracle_feasibility_branch",
                ),
                dt,
            )
            truth = transition.ground_truth
            xy = np.asarray((truth.pose.x, truth.pose.y), dtype=np.float64)
            static_clearance[candidate_index] = min(
                static_clearance[candidate_index],
                point_clearance(
                    float(xy[0]),
                    float(xy[1]),
                    static_obstacles,
                    float(robot_radius),
                ),
            )
            dynamic_clearance[candidate_index] = min(
                dynamic_clearance[candidate_index],
                _minimum_dynamic_clearance(
                    truth, plant, dynamic_radii, robot_radius
                ),
            )
            collision[candidate_index] |= bool(truth.collision)
            if collision[candidate_index]:
                break

        final_xy = np.asarray((truth.pose.x, truth.pose.y), dtype=np.float64)
        progress[candidate_index] = (
            start_distance - float(np.linalg.norm(final_xy - target_xy))
        )
        if not collision[candidate_index]:
            stop = ControlCommand(
                np.zeros(2, dtype=np.float64),
                plant.time,
                "oracle_stopping_tail",
            )
            for _ in range(int(stopping_tail_steps)):
                truth = plant.step(stop, dt).ground_truth
                value = _minimum_dynamic_clearance(
                    truth, plant, dynamic_radii, robot_radius
                )
                tail_dynamic_clearance[candidate_index] = min(
                    tail_dynamic_clearance[candidate_index], value
                )
                tail_collision[candidate_index] |= bool(truth.collision)
                if tail_collision[candidate_index]:
                    break
        terminal_dynamic_clearance[candidate_index] = (
            _minimum_dynamic_clearance(
                truth, plant, dynamic_radii, robot_radius
            )
        )
    plant.restore(snapshot)
    return {
        "collision": collision,
        "stopping_tail_collision": tail_collision,
        "local_target_progress_m": progress,
        "minimum_static_clearance_m": static_clearance,
        "minimum_dynamic_clearance_m": dynamic_clearance,
        "stopping_tail_minimum_dynamic_clearance_m": tail_dynamic_clearance,
        "terminal_dynamic_clearance_m": terminal_dynamic_clearance,
    }


def _physical_mask(metrics, criteria):
    return (
        ~metrics["collision"]
        & ~metrics["stopping_tail_collision"]
        & (
            metrics["local_target_progress_m"]
            >= float(criteria["minimum_local_target_progress_m"])
        )
        & (
            metrics["minimum_static_clearance_m"]
            >= float(criteria["minimum_static_clearance_m"])
        )
        & (
            metrics["minimum_dynamic_clearance_m"]
            >= float(criteria["minimum_dynamic_clearance_m"])
        )
        & (
            metrics["stopping_tail_minimum_dynamic_clearance_m"]
            >= float(criteria["minimum_stopping_tail_dynamic_clearance_m"])
        )
        & (
            metrics["terminal_dynamic_clearance_m"]
            >= float(criteria["minimum_terminal_dynamic_clearance_m"])
        )
    )


def _oracle_score(metrics, criteria):
    progress = metrics["local_target_progress_m"]
    static = np.clip(metrics["minimum_static_clearance_m"], -1.0, 1.0)
    dynamic = np.clip(metrics["minimum_dynamic_clearance_m"], -1.0, 2.0)
    tail = np.clip(
        metrics["stopping_tail_minimum_dynamic_clearance_m"], -1.0, 2.0
    )
    score = progress + 0.10 * static + 0.20 * dynamic + 0.10 * tail
    score -= 50.0 * metrics["collision"].astype(np.float64)
    score -= 30.0 * metrics["stopping_tail_collision"].astype(np.float64)
    score -= 10.0 * np.maximum(
        0.0,
        float(criteria["minimum_dynamic_clearance_m"])
        - metrics["minimum_dynamic_clearance_m"],
    )
    score -= 5.0 * np.maximum(
        0.0,
        float(criteria["minimum_static_clearance_m"])
        - metrics["minimum_static_clearance_m"],
    )
    return score


def _merge_metrics(parts):
    return {
        key: np.concatenate([value[key] for value in parts])
        for key in parts[0]
    }


def _cem_refine(
    plant,
    snapshot,
    initial_parameters,
    initial_labels,
    initial_metrics,
    horizon_steps,
    action_spec,
    search,
    criteria,
    evaluator,
    seed,
):
    lower, upper = _action_bounds(action_spec)
    low = np.tile(lower, 4)
    high = np.tile(upper, 4)
    scores = _oracle_score(initial_metrics, criteria)
    restart_count = int(search["cem_restarts"])
    iterations = int(search["cem_iterations"])
    sample_count = int(search["cem_samples_per_restart"])
    elite_count = int(search["cem_elite_count"])
    seed_indices = np.argsort(scores)[-restart_count:][::-1]
    all_parameters = [initial_parameters]
    all_labels = [np.asarray(initial_labels, dtype=object)]
    all_metrics = [initial_metrics]
    rng = np.random.default_rng(int(seed))

    for restart, seed_index in enumerate(seed_indices):
        mean = initial_parameters[int(seed_index)].copy()
        std = np.asarray(
            [
                0.20 * (high[index] - low[index])
                for index in range(high.size)
            ],
            dtype=np.float64,
        )
        family = str(initial_labels[int(seed_index)])
        for iteration in range(iterations):
            samples = rng.normal(
                mean[None, :], std[None, :], size=(sample_count, 8)
            )
            samples = np.clip(samples, low[None, :], high[None, :])
            sequences = _expand_segments(samples, horizon_steps)
            metrics = evaluator(sequences)
            values = _oracle_score(metrics, criteria)
            elite = np.argsort(values)[-elite_count:]
            mean = 0.35 * mean + 0.65 * np.mean(samples[elite], axis=0)
            std = np.maximum(
                0.04 * (high - low),
                0.50 * std + 0.50 * np.std(samples[elite], axis=0),
            )
            all_parameters.append(samples)
            all_labels.append(np.asarray(
                [
                    "cem%d_%s_i%d" % (restart, family, iteration)
                    for _ in range(sample_count)
                ],
                dtype=object,
            ))
            all_metrics.append(metrics)
    return (
        np.concatenate(all_parameters),
        np.concatenate(all_labels),
        _merge_metrics(all_metrics),
    )


def _offset_indices(rows, conflict_index, offsets_s):
    times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    conflict_time = float(times[int(conflict_index)])
    result = []
    for offset in offsets_s:
        target = max(float(times[0]), conflict_time + float(offset))
        index = int(np.searchsorted(times, target, side="right") - 1)
        index = max(0, min(index, int(conflict_index)))
        item = (float(offset), index, float(times[index]))
        if not result or result[-1][1] != index:
            result.append(item)
    return result


def _behavior_family(label):
    value = str(label)
    if value.startswith("cem"):
        value = value.split("_", 1)[1]
        value = value.rsplit("_i", 1)[0]
    if value.startswith("yield"):
        return "yield"
    if value.endswith("left"):
        return "left"
    if value.endswith("right"):
        return "right"
    return value


def _audit_risk_and_safety(
    candidates,
    labels,
    physical,
    state,
    perceived,
    components,
    static,
    criteria,
):
    indices = np.flatnonzero(physical)
    if not indices.size:
        return {
            "risk_accepted_count": 0,
            "safety_nonstop_count": 0,
            "accepted_behavior_families": [],
            "safety_nonstop_indices": [],
            "selected": None,
            "forecast_count": 0,
        }
    icode = _icode_controller(components["controller"])
    forecast_key = icode.config.probabilistic_obstacle_forecast_key
    forecasts = tuple(
        perceived.observation.auxiliary.get(forecast_key, ())
    )
    if not forecasts:
        return {
            "risk_accepted_count": 0,
            "safety_nonstop_count": 0,
            "accepted_behavior_families": [],
            "safety_nonstop_indices": [],
            "selected": None,
            "forecast_count": 0,
        }
    prefix = int(criteria["deployment_prefix_steps"])
    subset = candidates[indices, :prefix]
    predicted = icode.rollout(state, subset)
    risk = evaluate_collision_risk(
        predicted[:, 1:, :2],
        forecasts,
        _risk_config(icode),
    )
    predicted_static = np.asarray([
        min(
            point_clearance(
                float(position[0]),
                float(position[1]),
                static,
                float(components["controller"].config.robot_radius),
            )
            for position in path[1:, :2]
        )
        for path in predicted
    ])
    accepted_local = (
        ~risk.hard_violation
        & (
            predicted_static
            >= float(criteria["minimum_predicted_static_clearance_m"])
        )
    )
    accepted = indices[np.flatnonzero(accepted_local)]
    safety_indices = []
    for index in accepted:
        reset_safety = getattr(components["safety"], "reset", None)
        if callable(reset_safety):
            reset_safety()
        decision = components["safety"].arbitrate(
            ControlCommand(
                candidates[index, 0],
                float(perceived.observation.timestamp),
                "oracle_feasibility_gate",
            ),
            perceived.guard,
        )
        executed = np.asarray(
            decision.executed_control.values, dtype=np.float64
        )
        if (
            abs(float(executed[0])) >= 0.02
            or abs(float(executed[1])) >= 0.05
        ):
            safety_indices.append(int(index))
    families = sorted({
        _behavior_family(labels[index])
        for index in safety_indices
    })
    selected = int(safety_indices[0]) if safety_indices else None
    return {
        "risk_accepted_count": int(accepted.size),
        "safety_nonstop_count": int(len(safety_indices)),
        "accepted_behavior_families": families,
        # Expose the already-computed accepted rows for downstream supervised
        # label export.  This does not alter candidate generation, ranking,
        # Risk, Safety, or the frozen Gate v4 JSON contract.
        "safety_nonstop_indices": list(safety_indices),
        "selected": selected,
        "forecast_count": int(len(forecasts)),
    }


def _audit_state(
    map_name,
    rows,
    anchor_index,
    offset_s,
    anchor_time_s,
    config,
    protocol,
):
    search = protocol["oracle_search"]
    criteria = protocol["teacher_criteria"]
    results = []
    components = make_components(config, ROOT)
    plant = components["plant"]
    try:
        truth, perceived, state, target = _replay_to_anchor(
            config, components, rows, anchor_index
        )
        snapshot = plant.snapshot()
        obstacles = tuple(config["scene"]["obstacles"])
        static = tuple(
            item for item in obstacles
            if not isinstance(item.get("motion"), dict)
        )
        dynamic = tuple(
            item for item in obstacles
            if isinstance(item.get("motion"), dict)
        )
        dynamic_radii = tuple(float(item["radius"]) for item in dynamic)
        target_xy = np.asarray(
            (target.pose.x, target.pose.y), dtype=np.float64
        )
        dt = float(config["experiment"]["control_dt"])
        tail_steps = int(round(
            float(criteria["stopping_tail_s"]) / dt
        ))

        for horizon_s in search["horizons_s"]:
            horizon_steps = int(round(float(horizon_s) / dt))
            coarse_seed = (
                int(search["search_seed"])
                + 100000 * int(config["experiment"]["seed"])
                + 1000 * int(round(float(offset_s) * 10.0))
                + horizon_steps
            )
            labels, parameters = _structured_parameters(
                components["action_spec"],
                int(search["coarse_candidate_count"]),
                coarse_seed,
            )
            candidates = _expand_segments(parameters, horizon_steps)

            def evaluate(values):
                return _true_branch_metrics(
                    plant,
                    snapshot,
                    values,
                    dt,
                    static,
                    dynamic_radii,
                    float(components["controller"].config.robot_radius),
                    target_xy,
                    tail_steps,
                )

            coarse_metrics = evaluate(candidates)
            parameters, labels, metrics = _cem_refine(
                plant,
                snapshot,
                parameters,
                labels,
                coarse_metrics,
                horizon_steps,
                components["action_spec"],
                search,
                criteria,
                evaluate,
                coarse_seed + 17,
            )
            candidates = _expand_segments(parameters, horizon_steps)
            physical = _physical_mask(metrics, criteria)
            online = _audit_risk_and_safety(
                candidates,
                labels,
                physical,
                state,
                perceived,
                components,
                static,
                criteria,
            )
            physical_indices = np.flatnonzero(physical)
            best = (
                None
                if not physical_indices.size
                else int(physical_indices[np.argmax(
                    metrics["local_target_progress_m"][physical_indices]
                    + 0.10 * metrics[
                        "minimum_dynamic_clearance_m"
                    ][physical_indices]
                )])
            )
            results.append({
                "map": str(map_name),
                "seed": int(config["experiment"]["seed"]),
                "offset_s": float(offset_s),
                "anchor_index": int(anchor_index),
                "anchor_time_s": float(anchor_time_s),
                "horizon_s": float(horizon_s),
                "candidate_count": int(candidates.shape[0]),
                "coarse_candidate_count": int(
                    search["coarse_candidate_count"]
                ),
                "cem_candidate_count": int(
                    candidates.shape[0]
                    - int(search["coarse_candidate_count"])
                ),
                "physical_teacher_count": int(np.sum(physical)),
                "physical_teacher_exists": bool(np.any(physical)),
                "forecast_count": int(online["forecast_count"]),
                "risk_accepted_teacher_count": int(
                    online["risk_accepted_count"]
                ),
                "safety_nonstop_teacher_count": int(
                    online["safety_nonstop_count"]
                ),
                "accepted_behavior_families": online[
                    "accepted_behavior_families"
                ],
                "best_physical_behavior": (
                    None if best is None else str(labels[best])
                ),
                "best_physical_parameters": (
                    None
                    if best is None
                    else parameters[best].reshape(4, 2).tolist()
                ),
                "best_physical_progress_m": (
                    None
                    if best is None
                    else float(
                        metrics["local_target_progress_m"][best]
                    )
                ),
                "best_physical_minimum_static_clearance_m": (
                    None
                    if best is None
                    else float(
                        metrics["minimum_static_clearance_m"][best]
                    )
                ),
                "best_physical_minimum_dynamic_clearance_m": (
                    None
                    if best is None
                    else float(
                        metrics["minimum_dynamic_clearance_m"][best]
                    )
                ),
                "best_physical_stopping_tail_dynamic_clearance_m": (
                    None
                    if best is None
                    else float(
                        metrics[
                            "stopping_tail_minimum_dynamic_clearance_m"
                        ][best]
                    )
                ),
            })
        return results
    finally:
        plant.close()


def _root_cause(rows):
    physical = [row for row in rows if row["physical_teacher_exists"]]
    risk = [row for row in rows if row["risk_accepted_teacher_count"] > 0]
    safety = [row for row in rows if row["safety_nonstop_teacher_count"] > 0]
    current = [
        row for row in rows
        if abs(float(row["offset_s"])) < 1.0e-9
    ]
    earlier = [
        row for row in rows if float(row["offset_s"]) < -1.0e-9
    ]
    if not physical:
        return "no_oracle_teacher_even_with_earlier_state_and_strong_search"
    if any(row["physical_teacher_exists"] for row in earlier) and not any(
        row["physical_teacher_exists"] for row in current
    ):
        return "online_conflict_trigger_or_saved_state_is_too_late"
    if any(row["physical_teacher_exists"] for row in current):
        if not risk:
            return "physical_teacher_exists_but_online_risk_rejects"
        if not safety:
            return "risk_accepts_teacher_but_safety_stops_first_action"
        return "v3_candidate_space_was_insufficient"
    return "physical_teacher_exists_only_in_partial_timing_window"


def _positive_control(protocol):
    path = (ROOT / protocol["positive_control"]["gate_result"]).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    chapter2 = next(
        row for row in value["maps"] if row["map"] == "chapter2"
    )
    return {
        "source": str(path),
        "source_sha256": _sha256(path),
        "physical_teacher_exists": bool(
            chapter2["physical_teacher_exists"]
        ),
        "risk_accepted_teacher_exists": bool(
            chapter2["risk_accepted_teacher_exists"]
        ),
        "first_step_safety_nonstop": bool(
            chapter2["selected_first_step_safety_nonstop"]
        ),
        "selected_mode": chapter2["selected_mode"],
    }


def _dry_run(protocol):
    search = protocol["oracle_search"]
    return {
        "status": "dry_run_pass",
        "maps": sorted(protocol["saved_episodes"]),
        "offsets_s": [float(value) for value in search["offsets_s"]],
        "horizons_s": [float(value) for value in search["horizons_s"]],
        "coarse_candidate_count": int(search["coarse_candidate_count"]),
        "cem_candidate_count_per_state_horizon": int(
            search["cem_restarts"]
            * search["cem_iterations"]
            * search["cem_samples_per_restart"]
        ),
        "parallel_workers": int(search.get("parallel_workers", 1)),
        "formal_server_launch_authorized": False,
        "actor_training_started": False,
    }


def _audit_state_job(
    protocol_path,
    map_name,
    artifact,
    anchor_index,
    offset_s,
    anchor_time_s,
):
    protocol = _load_protocol(protocol_path)
    artifact = Path(artifact)
    config = load_yaml(artifact / "config_resolved.yaml")
    rows = _read_rows(artifact / "trajectory.csv")
    return _audit_state(
        map_name,
        rows,
        int(anchor_index),
        float(offset_s),
        float(anchor_time_s),
        config,
        protocol,
    )


def analyze(protocol_path, output):
    protocol_path = Path(protocol_path).resolve()
    protocol = _load_protocol(protocol_path)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "oracle gate output already contains evidence: %s" % output
        )
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "protocol_resolved.yaml"
    resolved.write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    progress_path = output / "progress.json"
    offsets = protocol["oracle_search"]["offsets_s"]
    horizons = protocol["oracle_search"]["horizons_s"]
    completed = 0
    all_rows = {}
    jobs = []

    for map_name, item in protocol["saved_episodes"].items():
        artifact = (ROOT / item["artifact"]).resolve()
        config_path = artifact / "config_resolved.yaml"
        trajectory_path = artifact / "trajectory.csv"
        config = load_yaml(config_path)
        rows = _read_rows(trajectory_path)
        conflict_index = _anchor_index(rows, item["conflict_rule"])
        for offset_s, anchor_index, anchor_time_s in _offset_indices(
            rows, conflict_index, offsets
        ):
            jobs.append((
                str(map_name),
                str(artifact),
                int(anchor_index),
                float(offset_s),
                float(anchor_time_s),
            ))
        all_rows[str(map_name)] = {
            "artifact": str(artifact),
            "config_sha256": _sha256(config_path),
            "trajectory_sha256": _sha256(trajectory_path),
            "conflict_index": int(conflict_index),
            "conflict_time_s": float(rows[conflict_index]["time"]),
            "states": [],
        }

    total = len(jobs) * len(horizons)
    workers = int(protocol["oracle_search"].get("parallel_workers", 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _audit_state_job,
                str(protocol_path),
                *job,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            state_rows = future.result()
            all_rows[job[0]]["states"].extend(state_rows)
            completed += len(state_rows)
            _write_json(progress_path, {
                "status": "running",
                "completed_state_horizons": int(completed),
                "total_state_horizons": int(total),
                "last_completed_map": str(job[0]),
                "last_completed_offset_s": float(job[3]),
                "parallel_workers": int(workers),
            })

    for map_name, summary in all_rows.items():
        state_rows = sorted(
            summary["states"],
            key=lambda row: (
                float(row["offset_s"]),
                float(row["horizon_s"]),
            ),
        )
        summary["states"] = state_rows
        summary["physical_teacher_exists"] = any(
            row["physical_teacher_exists"] for row in state_rows
        )
        summary["risk_accepted_teacher_exists"] = any(
            row["risk_accepted_teacher_count"] > 0 for row in state_rows
        )
        summary["safety_nonstop_teacher_exists"] = any(
            row["safety_nonstop_teacher_count"] > 0
            for row in state_rows
        )
        summary["accepted_behavior_families"] = sorted({
            family
            for row in state_rows
            for family in row["accepted_behavior_families"]
        })
        summary["root_cause"] = _root_cause(state_rows)

    positive = _positive_control(protocol)
    gate = protocol["gate"]
    target_maps_pass = all(
        row["physical_teacher_exists"]
        and row["risk_accepted_teacher_exists"]
        and row["safety_nonstop_teacher_exists"]
        for row in all_rows.values()
    )
    behavior_families = sorted({
        family
        for row in all_rows.values()
        for family in row["accepted_behavior_families"]
    })
    positive_pass = (
        positive["physical_teacher_exists"]
        and positive["risk_accepted_teacher_exists"]
        and positive["first_step_safety_nonstop"]
    )
    behavior_pass = len(behavior_families) >= int(
        gate["minimum_accepted_behavior_family_count"]
    )
    status = "pass" if (
        target_maps_pass and positive_pass and behavior_pass
    ) else "fail"
    result = {
        "schema_version": 1,
        "protocol": protocol["protocol"],
        "scope": protocol["scope"],
        "status": status,
        "actor_training_authorized": bool(
            status == "pass"
            and gate["actor_training_authorized_only_on_pass"]
        ),
        "formal_server_launch_authorized": False,
        "maps": all_rows,
        "positive_control": positive,
        "accepted_behavior_families": behavior_families,
        "behavior_diversity_gate_pass": bool(behavior_pass),
        "target_maps_gate_pass": bool(target_maps_pass),
        "positive_control_gate_pass": bool(positive_pass),
    }
    result_path = output / "oracle_gate_result.json"
    _write_json(result_path, result)
    _write_json(progress_path, {
        "status": "complete",
        "completed_state_horizons": int(total),
        "total_state_horizons": int(total),
        "gate_status": status,
    })
    manifest = {
        "git_sha": git_sha(ROOT),
        "protocol_sha256": _sha256(resolved),
        "result_sha256": _sha256(result_path),
        "progress_sha256": _sha256(progress_path),
        "protocol_config_hash": config_hash(protocol),
        "future_truth_training_only": True,
        "deployment_privileged_input": False,
        "actor_training_started": False,
        "parallel_workers": int(workers),
    }
    _write_json(output / "artifact_manifest.json", manifest)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if status == "pass" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    protocol = _load_protocol(args.protocol.resolve())
    if args.dry_run:
        print(json.dumps(_dry_run(protocol), indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required unless --dry-run is used")
    return analyze(args.protocol, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
