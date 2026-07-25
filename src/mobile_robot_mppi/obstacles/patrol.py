"""Recurrent semi-Markov waypoint obstacle process (V3)."""

from typing import Dict, Mapping, Sequence

import numpy as np

from .motion import NoiseProfile, ObstacleTrajectory


GENERATOR_VERSION_V3 = "recurrent_semimarkov_v3"
V3_PROCESS_NAMES = (
    "stochastic_cruise",
    "stochastic_shuttle",
    "branching_patrol",
    "hybrid_patrol",
)
_PROCESS_CODES = {
    name: index + 1 for index, name in enumerate(V3_PROCESS_NAMES)
}
_INTERVENTION_KINDS = (
    "early_retarget",
    "hesitation_stop",
    "speed_replan",
)


def _rng(seed: int, process: str, stream: int) -> np.random.RandomState:
    sequence = np.random.SeedSequence(
        [int(seed), int(_PROCESS_CODES[process]), 300, int(stream)]
    )
    value = int(sequence.generate_state(1, dtype=np.uint32)[0])
    return np.random.RandomState(value)


def _wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _clip_norm(vector, maximum):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= float(maximum) or norm <= 1.0e-15:
        return vector
    return vector * (float(maximum) / norm)


def _sample_range(rng, values):
    low, high = (float(value) for value in values)
    return float(rng.uniform(low, high))


def validate_v3_config(config: Mapping[str, object]) -> None:
    if str(config.get("generator_version")) != GENERATOR_VERSION_V3:
        raise ValueError("unexpected V3 generator version")
    if tuple(config.get("processes", ())) != V3_PROCESS_NAMES:
        raise ValueError("V3 processes must use the frozen registered order")
    duration = float(config["trajectory"]["duration_s"])
    dt = float(config["trajectory"]["dt"])
    if duration <= 0.0 or dt <= 0.0 or not np.isfinite((duration, dt)).all():
        raise ValueError("duration and dt must be finite and positive")
    if abs(round(duration / dt) * dt - duration) > 1.0e-10:
        raise ValueError("duration must contain an integer number of steps")
    limits = config["physical_limits"]
    for name in (
        "maximum_speed_mps",
        "maximum_acceleration_mps2",
        "maximum_braking_mps2",
        "maximum_moving_yaw_rate_radps",
        "yaw_rate_audit_minimum_speed_mps",
        "stopped_speed_threshold_mps",
    ):
        if float(limits[name]) <= 0.0:
            raise ValueError("physical limits must be positive")
    navigation = config["navigation"]
    minimum_speed = float(navigation["minimum_cruise_speed_mps"])
    maximum_speed = float(navigation["maximum_cruise_speed_mps"])
    if not 0.0 < minimum_speed <= maximum_speed <= float(
        limits["maximum_speed_mps"]
    ):
        raise ValueError("cruise-speed range violates physical limits")
    for process in V3_PROCESS_NAMES[1:]:
        process_config = config["processes_config"][process]
        waypoints = np.asarray(process_config["waypoints_m"], dtype=np.float64)
        if (
            waypoints.ndim != 2
            or waypoints.shape[1] != 2
            or waypoints.shape[0] < 2
            or not np.isfinite(waypoints).all()
        ):
            raise ValueError("waypoints must be a finite N x 2 array")
        initial = int(process_config["initial_waypoint"])
        if not 0 <= initial < waypoints.shape[0]:
            raise ValueError("initial waypoint is invalid")
        if process != "stochastic_shuttle":
            adjacency = process_config["adjacency"]
            if len(adjacency) != waypoints.shape[0]:
                raise ValueError("adjacency must have one row per waypoint")
            for source, neighbors in enumerate(adjacency):
                if not neighbors or source in neighbors:
                    raise ValueError("waypoint adjacency is empty or self-linked")
                if any(
                    int(target) < 0 or int(target) >= waypoints.shape[0]
                    for target in neighbors
                ):
                    raise ValueError("waypoint adjacency target is invalid")
    probabilities = config["processes_config"]["hybrid_patrol"][
        "intervention_probabilities"
    ]
    if tuple(probabilities.keys()) != _INTERVENTION_KINDS:
        raise ValueError("hybrid intervention kinds do not match the registry")
    if abs(sum(float(value) for value in probabilities.values()) - 1.0) > 1.0e-10:
        raise ValueError("hybrid intervention probabilities must sum to one")


def _choose_neighbor(rng, adjacency, source, excluded=None):
    choices = [int(value) for value in adjacency[int(source)]]
    filtered = [value for value in choices if value != excluded]
    if filtered:
        choices = filtered
    return int(rng.choice(choices))


def _sample_cruise_speed(rng, config):
    navigation = config["navigation"]
    return float(
        rng.uniform(
            float(navigation["minimum_cruise_speed_mps"]),
            float(navigation["maximum_cruise_speed_mps"]),
        )
    )


def _dropout_intervals(seed, process, config, steps):
    if process != "hybrid_patrol":
        return ()
    rng = _rng(seed, process, stream=4)
    observation = config["observation"]
    low_count, high_count = (
        int(value) for value in observation["hybrid_dropout_count_range"]
    )
    count = int(rng.randint(low_count, high_count + 1))
    dt = float(config["trajectory"]["dt"])
    duration = float(config["trajectory"]["duration_s"])
    time_s = float(rng.uniform(2.0, 6.0))
    intervals = []
    for _ in range(count):
        dropout_duration = _sample_range(
            rng, observation["dropout_duration_s"]
        )
        start = int(round(time_s / dt))
        end = min(
            steps,
            max(start + 1, int(round((time_s + dropout_duration) / dt))),
        )
        intervals.append((start, end))
        time_s = (
            end * dt
            + _sample_range(
                rng,
                (
                    float(observation["minimum_dropout_gap_s"]),
                    float(observation["minimum_dropout_gap_s"]) + 3.0,
                ),
            )
        )
        if time_s >= duration - 1.0 and len(intervals) < count:
            raise RuntimeError("V3 dropout renewal schedule exceeded episode")
    return tuple(intervals)


def _constrained_velocity_step(
    previous_velocity,
    desired_velocity,
    process_noise,
    dt,
    limits,
):
    previous_velocity = np.asarray(previous_velocity, dtype=np.float64)
    desired_velocity = np.asarray(desired_velocity, dtype=np.float64)
    previous_speed = float(np.linalg.norm(previous_velocity))
    desired_speed = float(np.linalg.norm(desired_velocity))
    acceleration_limit = (
        float(limits["maximum_braking_mps2"])
        if desired_speed + 1.0e-12 < previous_speed
        else float(limits["maximum_acceleration_mps2"])
    )
    requested = (desired_velocity - previous_velocity) / dt
    acceleration = _clip_norm(requested + process_noise, acceleration_limit)
    velocity = previous_velocity + acceleration * dt
    velocity = _clip_norm(velocity, float(limits["maximum_speed_mps"]))
    candidate_speed = float(np.linalg.norm(velocity))
    if previous_speed > 1.0e-12 and candidate_speed > 1.0e-12:
        previous_heading = float(
            np.arctan2(previous_velocity[1], previous_velocity[0])
        )
        candidate_heading = float(np.arctan2(velocity[1], velocity[0]))
        increment = _wrap_angle(candidate_heading - previous_heading)
        maximum_increment = (
            float(limits["maximum_moving_yaw_rate_radps"]) * dt
        )
        constrained_heading = previous_heading + float(
            np.clip(increment, -maximum_increment, maximum_increment)
        )
        velocity = candidate_speed * np.asarray(
            (np.cos(constrained_heading), np.sin(constrained_heading)),
            dtype=np.float64,
        )
    return velocity


def _event(step, dt, kind, **parameters):
    return {
        "step": int(step),
        "time_s": float(step * dt),
        "kind": str(kind),
        "parameters": parameters,
    }


def generate_patrol_trajectory(
    process: str,
    seed: int,
    noise_profile: NoiseProfile,
    config: Mapping[str, object],
) -> ObstacleTrajectory:
    validate_v3_config(config)
    noise_profile.validate()
    if process not in V3_PROCESS_NAMES:
        raise ValueError("unknown V3 obstacle process")
    if not 0 <= int(seed) <= 2 ** 32 - 1:
        raise ValueError("seed must be in [0, 2**32 - 1]")

    dt = float(config["trajectory"]["dt"])
    duration = float(config["trajectory"]["duration_s"])
    steps = int(round(duration / dt))
    times = np.arange(steps + 1, dtype=np.float64) * dt
    states = np.zeros((steps + 1, 4), dtype=np.float64)
    true_modes = ["cruise"] * (steps + 1)
    changes = np.zeros(steps + 1, dtype=bool)
    events = []
    limits = config["physical_limits"]
    navigation = config["navigation"]
    route_rng = _rng(seed, process, stream=1)
    process_rng = _rng(seed, process, stream=2)
    intervention_rng = _rng(seed, process, stream=5)

    waypoints = None
    adjacency = None
    current_waypoint = None
    target_waypoint = None
    previous_waypoint = None
    waypoint_visits = []
    phase = "travel"
    dwell_until = -1.0
    cruise_speed = 0.0
    hesitation_until = -1.0
    next_intervention_time = float("inf")
    retarget_used_this_leg = False

    if process == "stochastic_cruise":
        process_config = config["processes_config"][process]
        position = np.asarray(
            process_config["initial_position_m"], dtype=np.float64
        )
        speed = float(process_config["initial_speed_mps"])
        heading = np.deg2rad(float(process_config["initial_heading_deg"]))
        states[0, :2] = position
        states[0, 2:] = speed * np.asarray(
            (np.cos(heading), np.sin(heading))
        )
        cruise_speed = speed
        fixed_heading = float(heading)
    else:
        process_config = config["processes_config"][process]
        waypoints = np.asarray(
            process_config["waypoints_m"], dtype=np.float64
        )
        current_waypoint = int(process_config["initial_waypoint"])
        waypoint_visits = [current_waypoint]
        states[0, :2] = waypoints[current_waypoint]
        if process == "stochastic_shuttle":
            target_waypoint = int(process_config["initial_target"])
        else:
            adjacency = process_config["adjacency"]
            target_waypoint = _choose_neighbor(
                route_rng, adjacency, current_waypoint
            )
        direction = waypoints[target_waypoint] - waypoints[current_waypoint]
        heading = float(np.arctan2(direction[1], direction[0]))
        states[0, 2:] = 0.0
        cruise_speed = _sample_cruise_speed(route_rng, config)
        events.append(
            _event(
                0,
                dt,
                "depart",
                source_waypoint=current_waypoint,
                target_waypoint=target_waypoint,
                cruise_speed_mps=cruise_speed,
            )
        )
        if process == "hybrid_patrol":
            next_intervention_time = _sample_range(
                intervention_rng,
                process_config["intervention_gap_s"],
            )

    for index in range(1, steps + 1):
        timestamp = float(times[index])
        previous_state = states[index - 1]
        previous_velocity = previous_state[2:]
        previous_speed = float(np.linalg.norm(previous_velocity))
        mode = "cruise"

        if process == "stochastic_cruise":
            desired_velocity = cruise_speed * np.asarray(
                (np.cos(fixed_heading), np.sin(fixed_heading))
            )
        else:
            target_position = waypoints[target_waypoint]
            distance = float(np.linalg.norm(target_position - previous_state[:2]))
            stopped_threshold = float(
                limits["stopped_speed_threshold_mps"]
            )
            if (
                phase == "travel"
                and distance <= float(navigation["arrival_radius_m"])
                and previous_speed <= stopped_threshold
            ):
                phase = "dwell"
                current_waypoint = int(target_waypoint)
                retarget_used_this_leg = False
                waypoint_visits.append(current_waypoint)
                dwell = _sample_range(route_rng, navigation["dwell_time_s"])
                dwell_until = timestamp + dwell
                changes[index] = True
                events.append(
                    _event(
                        index,
                        dt,
                        "waypoint_arrival",
                        waypoint=current_waypoint,
                        completed_leg=len(waypoint_visits) - 1,
                        dwell_s=dwell,
                    )
                )
            if phase == "dwell" and timestamp >= dwell_until:
                previous_waypoint, source = previous_waypoint, current_waypoint
                if process == "stochastic_shuttle":
                    target_waypoint = 1 - int(current_waypoint)
                else:
                    target_waypoint = _choose_neighbor(
                        route_rng,
                        adjacency,
                        current_waypoint,
                        excluded=previous_waypoint,
                    )
                previous_waypoint = source
                cruise_speed = _sample_cruise_speed(route_rng, config)
                phase = "travel"
                changes[index] = True
                events.append(
                    _event(
                        index,
                        dt,
                        "depart",
                        source_waypoint=current_waypoint,
                        target_waypoint=target_waypoint,
                        cruise_speed_mps=cruise_speed,
                    )
                )
                if (
                    process == "hybrid_patrol"
                    and next_intervention_time <= timestamp
                ):
                    next_intervention_time = timestamp + _sample_range(
                        intervention_rng,
                        process_config["intervention_gap_s"],
                    )

            if (
                process == "hybrid_patrol"
                and phase == "travel"
                and timestamp >= next_intervention_time
            ):
                probabilities = process_config["intervention_probabilities"]
                kind = str(
                    intervention_rng.choice(
                        list(probabilities.keys()),
                        p=[float(value) for value in probabilities.values()],
                    )
                )
                parameters: Dict[str, object] = {}
                if kind == "early_retarget":
                    if retarget_used_this_leg:
                        kind = "speed_replan"
                        old_speed = float(cruise_speed)
                        cruise_speed = _sample_cruise_speed(
                            intervention_rng, config
                        )
                        parameters = {
                            "old_cruise_speed_mps": old_speed,
                            "new_cruise_speed_mps": cruise_speed,
                            "reason": "retarget_limit_for_current_leg",
                        }
                    else:
                        old_target = int(target_waypoint)
                        target_waypoint = _choose_neighbor(
                            intervention_rng,
                            adjacency,
                            current_waypoint,
                            excluded=old_target,
                        )
                        retarget_used_this_leg = True
                        parameters = {
                            "old_target_waypoint": old_target,
                            "new_target_waypoint": int(target_waypoint),
                        }
                elif kind == "hesitation_stop":
                    sampled = _sample_range(
                        intervention_rng,
                        process_config["hesitation_duration_s"],
                    )
                    minimum = previous_speed / float(
                        limits["maximum_braking_mps2"]
                    ) + 0.35
                    duration_s = max(sampled, minimum)
                    hesitation_until = timestamp + duration_s
                    parameters = {"duration_s": duration_s}
                elif kind == "speed_replan":
                    old_speed = float(cruise_speed)
                    cruise_speed = _sample_cruise_speed(
                        intervention_rng, config
                    )
                    parameters = {
                        "old_cruise_speed_mps": old_speed,
                        "new_cruise_speed_mps": cruise_speed,
                    }
                else:
                    raise RuntimeError("unknown V3 intervention")
                changes[index] = True
                events.append(_event(index, dt, kind, **parameters))
                next_intervention_time = timestamp + _sample_range(
                    intervention_rng,
                    process_config["intervention_gap_s"],
                )

            target_position = waypoints[target_waypoint]
            delta = target_position - previous_state[:2]
            distance = float(np.linalg.norm(delta))
            if phase == "dwell":
                desired_speed = 0.0
                desired_heading = (
                    float(np.arctan2(previous_velocity[1], previous_velocity[0]))
                    if previous_speed > 1.0e-12
                    else 0.0
                )
                mode = "dwell_wp_%d" % int(current_waypoint)
            elif timestamp < hesitation_until:
                desired_speed = 0.0
                desired_heading = (
                    float(np.arctan2(previous_velocity[1], previous_velocity[0]))
                    if previous_speed > 1.0e-12
                    else float(np.arctan2(delta[1], delta[0]))
                )
                mode = "hesitation_stop"
            else:
                desired_heading = float(np.arctan2(delta[1], delta[0]))
                braking_speed = np.sqrt(
                    max(
                        0.0,
                        2.0
                        * float(limits["maximum_braking_mps2"])
                        * max(
                            0.0,
                            distance
                            - 0.35
                            * float(navigation["arrival_radius_m"]),
                        ),
                    )
                )
                desired_speed = min(float(cruise_speed), float(braking_speed))
                mode = "travel_to_wp_%d" % int(target_waypoint)
            desired_velocity = desired_speed * np.asarray(
                (np.cos(desired_heading), np.sin(desired_heading))
            )

        process_scale = float(noise_profile.acceleration_std)
        if process != "stochastic_cruise" and (
            phase == "dwell" or timestamp < hesitation_until
        ):
            process_scale *= 0.06
        process_noise = process_rng.normal(size=2) * process_scale
        velocity = _constrained_velocity_step(
            previous_velocity,
            desired_velocity,
            process_noise,
            dt,
            limits,
        )
        realized_acceleration = (velocity - previous_velocity) / dt
        position = (
            previous_state[:2]
            + previous_velocity * dt
            + 0.5 * realized_acceleration * dt * dt
        )
        states[index, :2] = position
        states[index, 2:] = velocity
        true_modes[index] = mode

    dropout_intervals = _dropout_intervals(seed, process, config, steps)
    observation_rng = _rng(seed, process, stream=3)
    observations = states[:, :2] + float(
        noise_profile.observation_std
    ) * observation_rng.normal(size=(steps + 1, 2))
    observed_mask = np.ones(steps + 1, dtype=bool)
    for start, end in dropout_intervals:
        observed_mask[start:end] = False
    observations[~observed_mask] = np.nan

    metadata = {
        "schema_version": 3,
        "generator_version": GENERATOR_VERSION_V3,
        "process": process,
        "seed": int(seed),
        "dt": dt,
        "duration_s": duration,
        "steps": steps,
        "noise_profile": noise_profile.name,
        "process_acceleration_std": float(noise_profile.acceleration_std),
        "observation_position_std": float(noise_profile.observation_std),
        "events": events,
        "waypoint_visits": [int(value) for value in waypoint_visits],
        "waypoints_m": (
            [] if waypoints is None else waypoints.tolist()
        ),
        "dropout_intervals": [
            {
                "start_step": int(start),
                "end_step_exclusive": int(end),
                "start_time_s": float(times[start]),
                "end_time_s": float(times[end]),
            }
            for start, end in dropout_intervals
        ],
        "physical_limits": dict(limits),
        "random_streams": {
            "route": "SeedSequence(seed, process_code, 300, 1)",
            "process": "SeedSequence(seed, process_code, 300, 2)",
            "observation": "SeedSequence(seed, process_code, 300, 3)",
            "dropout": "SeedSequence(seed, process_code, 300, 4)",
            "intervention": "SeedSequence(seed, process_code, 300, 5)",
        },
    }
    trajectory = ObstacleTrajectory(
        process=process,
        seed=int(seed),
        noise_profile=noise_profile.name,
        times=times,
        states=states,
        observations=observations,
        observed_mask=observed_mask,
        true_modes=tuple(true_modes),
        change_flags=changes,
        metadata=metadata,
    )
    trajectory.validate()
    return trajectory


def audit_patrol_trajectory(
    trajectory: ObstacleTrajectory,
    config: Mapping[str, object],
) -> Dict[str, object]:
    dt = float(config["trajectory"]["dt"])
    limits = config["physical_limits"]
    velocities = trajectory.states[:, 2:]
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = np.diff(velocities, axis=0) / dt
    acceleration_norms = np.linalg.norm(accelerations, axis=1)
    position_steps = np.linalg.norm(
        np.diff(trajectory.states[:, :2], axis=0), axis=1
    )
    headings = np.unwrap(np.arctan2(velocities[:, 1], velocities[:, 0]))
    moving = (
        speeds[:-1] >= float(limits["yaw_rate_audit_minimum_speed_mps"])
    ) & (
        speeds[1:] >= float(limits["yaw_rate_audit_minimum_speed_mps"])
    )
    yaw_rates = np.abs(np.diff(headings) / dt)
    moving_yaw_rates = yaw_rates[moving]
    events = list(trajectory.metadata["events"])
    intervention_kinds = [
        str(event["kind"])
        for event in events
        if str(event["kind"]) in _INTERVENTION_KINDS
    ]
    visits = [int(value) for value in trajectory.metadata["waypoint_visits"]]
    round_trip = any(
        visits[index] == visits[index + 2]
        and visits[index] != visits[index + 1]
        for index in range(max(0, len(visits) - 2))
    )
    return {
        "finite_truth": bool(np.isfinite(trajectory.states).all()),
        "finite_available_observations": bool(
            np.isfinite(
                trajectory.observations[trajectory.observed_mask]
            ).all()
        ),
        "maximum_speed_mps": float(speeds.max()),
        "maximum_acceleration_mps2": float(acceleration_norms.max()),
        "maximum_moving_yaw_rate_radps": (
            float(moving_yaw_rates.max()) if moving_yaw_rates.size else 0.0
        ),
        "maximum_position_step_m": float(position_steps.max()),
        "speed_within_limit": bool(
            speeds.max() <= float(limits["maximum_speed_mps"]) + 1.0e-8
        ),
        "acceleration_within_limit": bool(
            acceleration_norms.max()
            <= float(limits["maximum_braking_mps2"]) + 1.0e-8
        ),
        "yaw_rate_within_limit": bool(
            not moving_yaw_rates.size
            or moving_yaw_rates.max()
            <= float(limits["maximum_moving_yaw_rate_radps"]) + 1.0e-8
        ),
        "teleport_free": bool(
            position_steps.max()
            <= float(limits["maximum_speed_mps"]) * dt + 1.0e-8
        ),
        "completed_legs": max(0, len(visits) - 1),
        "waypoint_visits": visits,
        "distinct_waypoints_visited": len(set(visits)),
        "round_trip_observed": bool(round_trip),
        "intervention_count": len(intervention_kinds),
        "intervention_kinds": intervention_kinds,
        "dropout_interval_count": len(
            trajectory.metadata["dropout_intervals"]
        ),
        "missing_observation_count": int(
            (~trajectory.observed_mask).sum()
        ),
    }


__all__: Sequence[str] = (
    "GENERATOR_VERSION_V3",
    "V3_PROCESS_NAMES",
    "audit_patrol_trajectory",
    "generate_patrol_trajectory",
    "validate_v3_config",
)
