"""Leakage-free paired MPPI trajectory features for utility learning."""

from typing import Sequence

import numpy as np


TRAJECTORY_METRIC_NAMES = (
    "predicted_local_progress_m",
    "predicted_final_progress_m",
    "terminal_local_distance_m",
    "terminal_final_distance_m",
    "trajectory_length_m",
    "minimum_clearance_m",
    "collision_fraction",
    "obstacle_proximity_mean_sq",
    "control_effort_mean_sq",
    "control_rate_rms",
    "control_jerk_rms",
    "mean_abs_omega_cmd",
    "max_abs_omega_cmd",
    "cost_min",
    "cost_q10",
    "cost_q50",
    "cost_q90",
    "cost_std",
    "effective_sample_fraction",
    "sample_saturation_fraction",
)


def _finite_vector(values, name):
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("%s must be a finite non-empty vector" % name)
    return vector


def plan_trajectory_metrics(
    plan,
    final_target_xy: Sequence[float],
    obstacles,
    position_indices,
    action_names,
    previous_action: Sequence[float],
    robot_radius: float,
    obstacle_influence: float,
):
    """Summarize one pre-execution MPPI plan in physical units.

    Only the predicted trajectory, proposed sequence, perceived local
    obstacles and planner diagnostics available before control execution are
    used.  No counterfactual branch outcome enters these features.
    """

    trajectory = np.asarray(plan.predicted_trajectory, dtype=np.float64)
    controls = np.asarray(plan.control_sequence, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] < 2:
        raise ValueError("predicted trajectory must have shape [H+1,nx]")
    if controls.ndim != 2 or controls.shape[0] + 1 != trajectory.shape[0]:
        raise ValueError("control sequence must align with predicted trajectory")
    if not np.isfinite(trajectory).all() or not np.isfinite(controls).all():
        raise ValueError("trajectory preview contains NaN or Inf")
    position_indices = tuple(int(value) for value in position_indices)
    if len(position_indices) != 2 or any(
        value < 0 or value >= trajectory.shape[1]
        for value in position_indices
    ):
        raise ValueError("trajectory features require two valid position indices")
    action_names = tuple(str(value) for value in action_names)
    if len(action_names) != controls.shape[1]:
        raise ValueError("action names do not match control sequence")
    previous = _finite_vector(previous_action, "previous action")
    if previous.shape != (controls.shape[1],):
        raise ValueError("previous action dimension differs from control sequence")
    final_target = _finite_vector(final_target_xy, "final target")
    if final_target.shape != (2,):
        raise ValueError("final target must contain x/y")
    if (
        not np.isfinite((robot_radius, obstacle_influence)).all()
        or robot_radius < 0.0
        or obstacle_influence <= 0.0
    ):
        raise ValueError("trajectory geometry parameters are invalid")

    xy = trajectory[:, list(position_indices)]
    local_target = np.asarray(
        (
            float(plan.diagnostics["target_x"]),
            float(plan.diagnostics["target_y"]),
        ),
        dtype=np.float64,
    )
    initial_local_distance = float(np.linalg.norm(xy[0] - local_target))
    terminal_local_distance = float(np.linalg.norm(xy[-1] - local_target))
    initial_final_distance = float(np.linalg.norm(xy[0] - final_target))
    terminal_final_distance = float(np.linalg.norm(xy[-1] - final_target))
    trajectory_length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())

    all_clearances = []
    for obstacle in obstacles:
        ox, oy = float(obstacle[0]), float(obstacle[1])
        radius = float(obstacle[2]) if len(obstacle) >= 3 else 0.08
        if not np.isfinite((ox, oy, radius)).all() or radius < 0.0:
            raise ValueError("trajectory obstacle is invalid")
        clearance = np.linalg.norm(xy - (ox, oy), axis=1)
        clearance -= radius + float(robot_radius)
        all_clearances.append(clearance)
    if all_clearances:
        clearance_matrix = np.stack(all_clearances, axis=1)
        point_clearance = clearance_matrix.min(axis=1)
        minimum_clearance = float(point_clearance.min())
        collision_fraction = float(np.mean(point_clearance <= 0.0))
        proximity = np.maximum(
            0.0, float(obstacle_influence) - point_clearance
        )
        obstacle_proximity = float(np.mean(proximity ** 2))
    else:
        minimum_clearance = float(2.0 * obstacle_influence)
        collision_fraction = 0.0
        obstacle_proximity = 0.0

    control_effort = float(np.mean(controls ** 2))
    rates = np.diff(controls, axis=0, prepend=previous[None, :])
    control_rate_rms = float(np.sqrt(np.mean(rates ** 2)))
    jerks = np.diff(rates, axis=0, prepend=np.zeros((1, rates.shape[1])))
    control_jerk_rms = float(np.sqrt(np.mean(jerks ** 2)))
    if "omega_cmd" in action_names:
        omega = controls[:, action_names.index("omega_cmd")]
        mean_abs_omega = float(np.mean(np.abs(omega)))
        max_abs_omega = float(np.max(np.abs(omega)))
    else:
        mean_abs_omega = max_abs_omega = 0.0

    diagnostic_names = (
        "cost_min",
        "cost_q10",
        "cost_q50",
        "cost_q90",
        "cost_std",
        "effective_sample_fraction",
        "sample_saturation_fraction",
    )
    try:
        planner_values = [
            float(plan.diagnostics[name]) for name in diagnostic_names
        ]
    except KeyError as exc:
        raise ValueError(
            "trajectory preview diagnostics are incomplete"
        ) from exc

    values = np.asarray(
        (
            initial_local_distance - terminal_local_distance,
            initial_final_distance - terminal_final_distance,
            terminal_local_distance,
            terminal_final_distance,
            trajectory_length,
            minimum_clearance,
            collision_fraction,
            obstacle_proximity,
            control_effort,
            control_rate_rms,
            control_jerk_rms,
            mean_abs_omega,
            max_abs_omega,
            *planner_values,
        ),
        dtype=np.float32,
    )
    if values.shape != (len(TRAJECTORY_METRIC_NAMES),):
        raise RuntimeError("trajectory metric schema and values differ")
    if not np.isfinite(values).all():
        raise FloatingPointError("trajectory metrics contain NaN or Inf")
    return values


def paired_trajectory_feature_vector(
    state_features,
    base_plan,
    candidate_plan,
    final_target_xy,
    obstacles,
    position_indices,
    action_names,
    previous_action,
    robot_radius,
    obstacle_influence,
):
    """Append BC, candidate and candidate-minus-BC trajectory summaries."""

    state = _finite_vector(state_features, "state features").astype(np.float32)
    base = plan_trajectory_metrics(
        base_plan,
        final_target_xy,
        obstacles,
        position_indices,
        action_names,
        previous_action,
        robot_radius,
        obstacle_influence,
    )
    candidate = plan_trajectory_metrics(
        candidate_plan,
        final_target_xy,
        obstacles,
        position_indices,
        action_names,
        previous_action,
        robot_radius,
        obstacle_influence,
    )
    vector = np.concatenate((state, base, candidate, candidate - base)).astype(
        np.float32, copy=False
    )
    metric_dim = len(TRAJECTORY_METRIC_NAMES)
    schema = {
        "feature_set": "state_plus_paired_mppi_trajectory",
        "state_feature_dim": int(state.size),
        "trajectory_metric_dim": int(metric_dim),
        "trajectory_metric_names": list(TRAJECTORY_METRIC_NAMES),
        "trajectory_layout": [
            "state_features",
            "base_trajectory_metrics",
            "candidate_trajectory_metrics",
            "candidate_minus_base_metrics",
        ],
        "common_random_numbers": True,
        "feature_dim": int(vector.size),
    }
    return vector, schema


__all__ = [
    "TRAJECTORY_METRIC_NAMES",
    "paired_trajectory_feature_vector",
    "plan_trajectory_metrics",
]
