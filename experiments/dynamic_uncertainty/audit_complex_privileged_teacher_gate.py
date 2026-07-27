"""Audit whether privileged complex-map maneuvers survive the frozen stack.

This is a development-only pretraining gate.  Future obstacle truth is used
only to rank a deterministic maneuver lattice in exact MuJoCo branches.
Candidate prefixes are then rolled out by the frozen online ICODE model and
scored by the unchanged probabilistic Collision Risk contract.  No privileged
quantity is exposed to a deployed policy.
"""

from __future__ import annotations

import argparse
import csv
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

from mobile_robot_mppi.core.config import config_hash, git_sha, load_yaml
from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.evaluation.scene_feasibility import point_clearance
from mobile_robot_mppi.obstacles.collision_risk import (
    CollisionRiskConfig,
    evaluate_collision_risk,
)
from mobile_robot_mppi.runtime.factories import make_components


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/research/complex_supervised_actor_teacher_gate_v1.yaml"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_vector(truth, state_spec):
    values = {
        "x": truth.pose.x,
        "y": truth.pose.y,
        "theta": truth.pose.theta,
        "v": truth.twist.v,
        "omega": truth.twist.omega,
        "wheel_left": truth.wheel_speeds[0],
        "wheel_right": truth.wheel_speeds[1],
    }
    return np.asarray(
        [values[name] for name in state_spec.names], dtype=np.float64
    )


def _truth(value):
    return str(value).strip().lower() in {"1", "1.0", "true"}


def _load_protocol(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("privileged teacher protocol must be a mapping")
    if value.get("protocol") != (
        "complex_supervised_actor_privileged_teacher_gate_v1"
    ):
        raise ValueError("privileged teacher protocol mismatch")
    teacher = value["teacher"]
    if not bool(teacher["future_truth_used_for_selection"]):
        raise ValueError("teacher gate must explicitly use future truth")
    if int(teacher["deployment_prefix_steps"]) != 36:
        raise ValueError("frozen deployment prefix must remain 36 steps")
    if bool(value["gate"]["formal_server_launch_authorized"]):
        raise ValueError("teacher gate cannot authorize a formal server")
    return value


def _candidate_lattice(action_spec, teacher):
    horizon = int(teacher["truth_horizon_steps"])
    lower = np.asarray(action_spec.lower, dtype=np.float64)
    upper = np.asarray(action_spec.upper, dtype=np.float64)
    forward_max = float(upper[0])
    reverse_max = abs(float(lower[0]))
    yaw_max = min(abs(float(lower[1])), abs(float(upper[1])))
    recovery_fraction = float(teacher["recovery_fraction"])
    sequences = []
    labels = []

    def append(label, controls):
        bounded = np.clip(controls, lower[None, :], upper[None, :])
        key = np.round(bounded, decimals=9).tobytes()
        if key not in seen:
            seen.add(key)
            labels.append(label)
            sequences.append(bounded)

    seen = set()
    for speed_fraction in teacher["speed_fractions"]:
        speed = forward_max * float(speed_fraction)
        straight = np.zeros((horizon, 2), dtype=np.float64)
        straight[:, 0] = speed
        append("forward", straight)
        for direction, mode in ((1.0, "left"), (-1.0, "right")):
            for yaw_fraction in teacher["turn_rate_fractions"]:
                yaw = direction * yaw_max * float(yaw_fraction)
                for turn_steps in teacher["turn_steps"]:
                    turn_steps = int(turn_steps)
                    recovery_steps = min(turn_steps, horizon - turn_steps)
                    for wait_steps in teacher["wait_steps"]:
                        wait_steps = int(wait_steps)
                        for reverse_steps in teacher[
                            "reverse_prefix_steps"
                        ]:
                            reverse_steps = int(reverse_steps)
                            if wait_steps and reverse_steps:
                                continue
                            controls = np.zeros(
                                (horizon, 2), dtype=np.float64
                            )
                            prefix = wait_steps + reverse_steps
                            if reverse_steps:
                                controls[:reverse_steps, 0] = -reverse_max
                                controls[:reverse_steps, 1] = (
                                    -0.35 * yaw
                                )
                            start = prefix
                            middle = min(horizon, start + turn_steps)
                            end = min(
                                horizon, middle + recovery_steps
                            )
                            controls[start:, 0] = speed
                            controls[start:middle, 1] = yaw
                            controls[middle:end, 1] = (
                                -recovery_fraction * yaw
                            )
                            prefix_name = (
                                "wait_"
                                if wait_steps
                                else "reverse_"
                                if reverse_steps
                                else ""
                            )
                            append(prefix_name + mode, controls)
    stop = np.zeros((horizon, 2), dtype=np.float64)
    append("stop", stop)
    reverse = np.zeros((horizon, 2), dtype=np.float64)
    reverse[:, 0] = -reverse_max
    append("reverse", reverse)
    return tuple(labels), np.stack(sequences)


def _read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _anchor_index(rows, rule):
    if rule != "first_forecast_valid_hard_risk":
        raise ValueError("unknown saved-state anchor rule: %s" % rule)
    for index, row in enumerate(rows):
        if (
            _truth(row.get("dynamic_obstacle_tracker_forecast_valid", 0))
            and _truth(
                row.get(
                    "probabilistic_obstacle_hard_violation", 0
                )
            )
        ):
            return index
    raise ValueError("saved episode contains no forecast-valid hard-risk row")


def _replay_to_anchor(config, components, rows, anchor_index):
    seed = int(config["experiment"]["seed"])
    initial = np.asarray(
        config["experiment"]["initial_state"], dtype=np.float64
    )
    plant = components["plant"]
    truth = plant.reset(seed, initial)
    observation = components["sensors"].reset(truth, seed)
    reset_perception = getattr(components["perception"], "reset", None)
    if callable(reset_perception):
        reset_perception()
    reset_reference = getattr(components["reference"], "reset", None)
    if callable(reset_reference):
        reset_reference()
    components["controller"].reset(seed=seed)
    dt = float(config["experiment"]["control_dt"])
    for row in rows[:anchor_index]:
        perceived = components["perception"].process(observation)
        state = _state_vector(truth, components["state_spec"])
        components["reference"].target_at(observation.timestamp, state)
        command = ControlCommand(
            np.asarray(
                [float(row["applied_v"]), float(row["applied_omega"])],
                dtype=np.float64,
            ),
            plant.time,
            "saved_teacher_replay",
        )
        transition = plant.step(command, dt)
        truth = transition.ground_truth
        if truth.collision:
            raise RuntimeError("saved replay collided before selected anchor")
        observation = components["sensors"].observe(truth)
    perceived = components["perception"].process(observation)
    state = _state_vector(truth, components["state_spec"])
    target = components["reference"].target_at(
        observation.timestamp, state
    )
    return truth, perceived, state, target


def _true_branch_metrics(
    plant,
    snapshot,
    candidates,
    state_spec,
    dt,
    static_obstacles,
    dynamic_radii,
    robot_radius,
    target_xy,
):
    count, horizon, _ = candidates.shape
    collisions = np.zeros(count, dtype=bool)
    progress = np.zeros(count, dtype=np.float64)
    static_clearance = np.full(count, np.inf, dtype=np.float64)
    dynamic_clearance = np.full(count, np.inf, dtype=np.float64)
    start_truth = plant.restore(snapshot)
    start_xy = np.asarray(
        [start_truth.pose.x, start_truth.pose.y], dtype=np.float64
    )
    start_distance = float(np.linalg.norm(start_xy - target_xy))
    for candidate_index, sequence in enumerate(candidates):
        truth = plant.restore(snapshot)
        for step_index in range(horizon):
            transition = plant.step(
                ControlCommand(
                    sequence[step_index],
                    plant.time,
                    "privileged_teacher_branch",
                ),
                dt,
            )
            truth = transition.ground_truth
            collisions[candidate_index] |= bool(truth.collision)
            xy = np.asarray(
                [truth.pose.x, truth.pose.y], dtype=np.float64
            )
            static_clearance[candidate_index] = min(
                static_clearance[candidate_index],
                point_clearance(
                    float(xy[0]),
                    float(xy[1]),
                    static_obstacles,
                    robot_radius,
                ),
            )
            centers = plant.dynamic_obstacle_states()
            for center, radius in zip(centers, dynamic_radii):
                distance = float(np.linalg.norm(
                    xy - np.asarray([center["x"], center["y"]])
                ))
                dynamic_clearance[candidate_index] = min(
                    dynamic_clearance[candidate_index],
                    distance - robot_radius - radius,
                )
            if collisions[candidate_index]:
                break
        final_xy = np.asarray(
            [truth.pose.x, truth.pose.y], dtype=np.float64
        )
        progress[candidate_index] = (
            start_distance - float(np.linalg.norm(final_xy - target_xy))
        )
    plant.restore(snapshot)
    return {
        "collision": collisions,
        "local_target_progress_m": progress,
        "minimum_static_clearance_m": static_clearance,
        "minimum_dynamic_center_clearance_m": dynamic_clearance,
    }


def _risk_config(controller):
    config = controller.config
    return CollisionRiskConfig(
        robot_radius_m=float(config.robot_radius),
        safety_margin_m=float(
            config.probabilistic_obstacle_safety_margin
        ),
        minimum_position_std_m=float(
            config.probabilistic_obstacle_minimum_std
        ),
        hard_probability_threshold=float(
            config.probabilistic_obstacle_hard_threshold
        ),
    )


def _audit_map(map_name, item, teacher):
    artifact = (ROOT / item["artifact"]).resolve()
    config_path = artifact / "config_resolved.yaml"
    trajectory_path = artifact / "trajectory.csv"
    if not config_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError("saved conflict artifact is incomplete")
    config = load_yaml(config_path)
    components = make_components(config, ROOT)
    plant = components["plant"]
    try:
        rows = _read_rows(trajectory_path)
        anchor_index = _anchor_index(rows, item["anchor_rule"])
        truth, perceived, state, target = _replay_to_anchor(
            config, components, rows, anchor_index
        )
        labels, candidates = _candidate_lattice(
            components["action_spec"], teacher
        )
        snapshot = plant.snapshot()
        obstacles = tuple(config["scene"]["obstacles"])
        static = tuple(
            value for value in obstacles
            if not isinstance(value.get("motion"), dict)
        )
        dynamic = tuple(
            value for value in obstacles
            if isinstance(value.get("motion"), dict)
        )
        dynamic_radii = tuple(float(value["radius"]) for value in dynamic)
        target_xy = np.asarray(
            [target.pose.x, target.pose.y], dtype=np.float64
        )
        truth_metrics = _true_branch_metrics(
            plant,
            snapshot,
            candidates,
            components["state_spec"],
            float(config["experiment"]["control_dt"]),
            static,
            dynamic_radii,
            float(components["controller"].config.robot_radius),
            target_xy,
        )
        prefix = int(teacher["deployment_prefix_steps"])
        predicted = components["controller"].rollout(
            state, candidates[:, :prefix]
        )
        forecast_key = (
            components["controller"].config
            .probabilistic_obstacle_forecast_key
        )
        forecasts = tuple(
            perceived.observation.metadata.get(forecast_key, ())
        )
        if not forecasts:
            raise RuntimeError("selected conflict anchor has no forecasts")
        risk = evaluate_collision_risk(
            predicted[:, 1:, :2],
            forecasts,
            _risk_config(components["controller"]),
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
        physical = (
            ~truth_metrics["collision"]
            & (
                truth_metrics["local_target_progress_m"]
                >= float(teacher["minimum_local_target_progress_m"])
            )
            & (
                truth_metrics["minimum_static_clearance_m"]
                >= float(teacher["minimum_true_static_clearance_m"])
            )
        )
        accepted = (
            physical
            & ~risk.hard_violation
            & (
                predicted_static
                >= float(
                    teacher["minimum_predicted_static_clearance_m"]
                )
            )
        )
        eligible = np.flatnonzero(accepted)
        physical_indices = np.flatnonzero(physical)
        pool = eligible if eligible.size else physical_indices
        if pool.size:
            selected = int(pool[np.argmax(
                truth_metrics["local_target_progress_m"][pool]
                + 0.05 * np.minimum(
                    truth_metrics["minimum_dynamic_center_clearance_m"][pool],
                    2.0,
                )
            )])
            decision = components["safety"].arbitrate(
                ControlCommand(
                    candidates[selected, 0],
                    float(perceived.observation.timestamp),
                    "privileged_teacher_gate",
                ),
                perceived.guard,
            )
            executed = np.asarray(
                decision.executed_control.values, dtype=np.float64
            )
            first_step_nonstop = bool(
                abs(float(executed[0])) >= 0.02
                or abs(float(executed[1])) >= 0.05
            )
        else:
            selected = -1
            executed = np.asarray([np.nan, np.nan])
            first_step_nonstop = False
        summary = {
            "map": map_name,
            "saved_artifact": str(artifact),
            "saved_config_sha256": _sha256(config_path),
            "saved_trajectory_sha256": _sha256(trajectory_path),
            "seed": int(config["experiment"]["seed"]),
            "anchor_index": int(anchor_index),
            "anchor_time_s": float(plant.time),
            "candidate_count": int(candidates.shape[0]),
            "forecast_count": int(len(forecasts)),
            "physically_safe_teacher_count": int(np.sum(physical)),
            "risk_accepted_teacher_count": int(np.sum(accepted)),
            "selected_index": selected,
            "selected_mode": (
                None if selected < 0 else str(labels[selected])
            ),
            "selected_true_progress_m": (
                None
                if selected < 0
                else float(
                    truth_metrics["local_target_progress_m"][selected]
                )
            ),
            "selected_true_static_clearance_m": (
                None
                if selected < 0
                else float(
                    truth_metrics["minimum_static_clearance_m"][selected]
                )
            ),
            "selected_true_dynamic_clearance_m": (
                None
                if selected < 0
                else float(
                    truth_metrics[
                        "minimum_dynamic_center_clearance_m"
                    ][selected]
                )
            ),
            "selected_online_maximum_probability": (
                None
                if selected < 0
                else float(risk.maximum_step_probability[selected])
            ),
            "selected_online_probability_mass": (
                None
                if selected < 0
                else float(risk.accumulated_probability_mass[selected])
            ),
            "selected_online_hard_violation": (
                None
                if selected < 0
                else bool(risk.hard_violation[selected])
            ),
            "selected_predicted_static_clearance_m": (
                None
                if selected < 0
                else float(predicted_static[selected])
            ),
            "selected_first_control": (
                None
                if selected < 0
                else candidates[selected, 0].tolist()
            ),
            "selected_safety_executed_control": (
                None if selected < 0 else executed.tolist()
            ),
            "selected_first_step_safety_nonstop": first_step_nonstop,
            "physical_teacher_exists": bool(physical_indices.size),
            "risk_accepted_teacher_exists": bool(eligible.size),
        }
        return summary
    finally:
        plant.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "teacher-gate output already contains evidence: %s" % output
        )
    output.mkdir(parents=True, exist_ok=True)
    protocol = _load_protocol(args.protocol.resolve())
    teacher = protocol["teacher"]
    results = [
        _audit_map(map_name, item, teacher)
        for map_name, item in protocol["saved_conflict_states"].items()
    ]
    gate = protocol["gate"]
    physical_pass = all(
        row["physical_teacher_exists"] for row in results
    )
    risk_pass = all(
        row["risk_accepted_teacher_exists"] for row in results
    )
    safety_pass = all(
        row["selected_first_step_safety_nonstop"] for row in results
    )
    status = "pass" if (
        physical_pass and risk_pass and safety_pass
    ) else "fail"
    summary = {
        "schema_version": 1,
        "protocol": protocol["protocol"],
        "scope": protocol["scope"],
        "status": status,
        "physical_teacher_gate_pass": physical_pass,
        "risk_acceptance_gate_pass": risk_pass,
        "first_step_safety_gate_pass": safety_pass,
        "actor_training_authorized": bool(
            status == "pass"
            and gate["actor_training_authorized_only_on_pass"]
        ),
        "formal_server_launch_authorized": False,
        "maps": results,
    }
    result_path = output / "teacher_gate_result.json"
    result_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolved_path = output / "protocol_resolved.yaml"
    resolved_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    manifest = {
        "git_sha": git_sha(ROOT),
        "protocol_sha256": _sha256(resolved_path),
        "result_sha256": _sha256(result_path),
        "protocol_config_hash": config_hash(protocol),
        "future_truth_training_only": True,
        "deployment_privileged_input": False,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
