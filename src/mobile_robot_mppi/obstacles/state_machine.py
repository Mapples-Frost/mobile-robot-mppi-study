"""Constrained multi-event stochastic obstacle process (V2).

The discrete event program is seeded independently from continuous process
noise and observation noise. Hidden commands may switch suddenly, while the
physical velocity follows through bounded acceleration and turn rate.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .motion import (
    PROCESS_NAMES,
    NoiseProfile,
    ObstacleTrajectory,
)


GENERATOR_VERSION = "constrained_multievent_v2"
_PROCESS_CODES = {name: index + 1 for index, name in enumerate(PROCESS_NAMES)}


def _wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _rng(seed: int, process: str, stream: int) -> np.random.RandomState:
    sequence = np.random.SeedSequence(
        [int(seed), int(_PROCESS_CODES[process]), 200, int(stream)]
    )
    value = int(sequence.generate_state(1, dtype=np.uint32)[0])
    return np.random.RandomState(value)


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm <= 1.0e-15:
        return vector
    return vector * (float(maximum) / norm)


@dataclass(frozen=True)
class MotionEvent:
    step: int
    time_s: float
    kind: str
    parameters: Mapping[str, object]

    def to_dict(self):
        return {
            "step": int(self.step),
            "time_s": float(self.time_s),
            "kind": self.kind,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class MotionProgram:
    events: Tuple[MotionEvent, ...]
    dropout_intervals: Tuple[Tuple[int, int], ...]

    def validate(self, steps: int) -> None:
        event_steps = [event.step for event in self.events]
        if event_steps != sorted(event_steps) or len(event_steps) != len(set(event_steps)):
            raise ValueError("motion events must have unique increasing steps")
        if any(step <= 0 or step >= steps for step in event_steps):
            raise ValueError("motion event step lies outside the episode")
        previous_end = -1
        for start, end in self.dropout_intervals:
            if not 0 < start < end <= steps:
                raise ValueError("dropout interval lies outside the episode")
            if start < previous_end:
                raise ValueError("dropout intervals overlap")
            previous_end = end


def validate_v2_config(config: Mapping[str, object]) -> None:
    if str(config.get("generator_version")) != GENERATOR_VERSION:
        raise ValueError("unexpected V2 generator version")
    trajectory = config["trajectory"]
    limits = config["physical_limits"]
    event = config["event_program"]
    duration = float(trajectory["duration_s"])
    dt = float(trajectory["dt"])
    if not np.isfinite((duration, dt)).all() or duration <= 0.0 or dt <= 0.0:
        raise ValueError("trajectory duration and dt must be finite and positive")
    if abs(round(duration / dt) * dt - duration) > 1.0e-10:
        raise ValueError("trajectory duration must be an integer number of steps")
    positive_limits = (
        "maximum_speed_mps",
        "maximum_acceleration_mps2",
        "maximum_braking_mps2",
        "maximum_moving_yaw_rate_radps",
        "yaw_rate_audit_minimum_speed_mps",
        "stopped_speed_threshold_mps",
    )
    if any(float(limits[name]) <= 0.0 for name in positive_limits):
        raise ValueError("physical limits must be positive")
    turn_low, turn_high = (float(value) for value in event["turn_duration_s"])
    if not 0.0 < turn_low <= turn_high:
        raise ValueError("turn-duration range is invalid")
    stop_low, stop_high = (float(value) for value in event["stop_dwell_s"])
    if not 0.0 < stop_low <= stop_high:
        raise ValueError("stop-dwell range is invalid")
    if not 0.0 <= float(event["reverse_probability"]) <= 1.0:
        raise ValueError("reverse probability must be in [0, 1]")
    for process in PROCESS_NAMES:
        initial = config["initial_conditions"][process]
        position = np.asarray(initial["position_m"], dtype=np.float64)
        values = (float(initial["speed_mps"]), float(initial["heading_deg"]))
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("initial position must be a finite 2-vector")
        if not np.isfinite(values).all() or values[0] <= 0.0:
            raise ValueError("initial speed and heading must be finite")


def _sample_range(rng, values):
    low, high = (float(value) for value in values)
    return float(rng.uniform(low, high))


def _snap_step(time_s: float, dt: float) -> int:
    return int(round(float(time_s) / float(dt)))


def _jittered_times(
    bases: Sequence[float],
    jitter: float,
    rng: np.random.RandomState,
) -> List[float]:
    return [float(base + rng.uniform(-jitter, jitter)) for base in bases]


def _turn_event(
    time_s: float,
    dt: float,
    rng: np.random.RandomState,
    config: Mapping[str, object],
) -> MotionEvent:
    angle_deg = float(rng.choice(config["event_program"]["turn_angles_deg"]))
    limits = config["physical_limits"]
    minimum_duration = abs(np.deg2rad(angle_deg)) / float(
        limits["maximum_moving_yaw_rate_radps"]
    )
    sampled_duration = _sample_range(
        rng, config["event_program"]["turn_duration_s"]
    )
    duration = max(minimum_duration, sampled_duration)
    turn_rate = float(np.deg2rad(angle_deg) / duration)
    return MotionEvent(
        step=_snap_step(time_s, dt),
        time_s=_snap_step(time_s, dt) * dt,
        kind="turn_left" if angle_deg > 0.0 else "turn_right",
        parameters={
            "angle_deg": angle_deg,
            "duration_s": duration,
            "turn_rate_radps": turn_rate,
        },
    )


def _speed_event(
    time_s: float,
    dt: float,
    rng: np.random.RandomState,
    config: Mapping[str, object],
) -> MotionEvent:
    scale = float(rng.choice(config["event_program"]["speed_scales"]))
    return MotionEvent(
        step=_snap_step(time_s, dt),
        time_s=_snap_step(time_s, dt) * dt,
        kind="accelerate" if scale > 1.0 else "decelerate",
        parameters={"speed_scale": scale},
    )


def _stop_restart_events(
    time_s: float,
    dt: float,
    rng: np.random.RandomState,
    config: Mapping[str, object],
) -> Tuple[MotionEvent, MotionEvent]:
    event_config = config["event_program"]
    stop_step = _snap_step(time_s, dt)
    dwell = _sample_range(rng, event_config["stop_dwell_s"])
    resume_step = max(stop_step + 1, _snap_step(time_s + dwell, dt))
    reverse = bool(rng.uniform() < float(event_config["reverse_probability"]))
    speed_scale = _sample_range(rng, event_config["restart_speed_scale"])
    stop = MotionEvent(
        step=stop_step,
        time_s=stop_step * dt,
        kind="stop",
        parameters={"requested_dwell_s": dwell},
    )
    resume = MotionEvent(
        step=resume_step,
        time_s=resume_step * dt,
        kind="reverse" if reverse else "restart_forward",
        parameters={
            "restart_speed_scale": speed_scale,
            "requested_dwell_s": dwell,
        },
    )
    return stop, resume


def build_motion_program(
    process: str,
    seed: int,
    config: Mapping[str, object],
) -> MotionProgram:
    validate_v2_config(config)
    if process not in PROCESS_NAMES:
        raise ValueError("unknown obstacle process")
    dt = float(config["trajectory"]["dt"])
    steps = int(round(float(config["trajectory"]["duration_s"]) / dt))
    rng = _rng(seed, process, stream=1)
    jitter = float(config["event_program"]["time_jitter_s"])
    events: List[MotionEvent] = []

    if process == "speed_change":
        times = _jittered_times((4.0, 10.0), jitter, rng)
        events.append(_speed_event(times[0], dt, rng, config))
        events.extend(_stop_restart_events(times[1], dt, rng, config))
    elif process == "direction_change":
        times = _jittered_times((4.0, 10.0), jitter, rng)
        events.extend(_turn_event(time_s, dt, rng, config) for time_s in times)
    elif process == "combined_change":
        times = _jittered_times((2.7, 5.9, 9.1, 12.3), jitter, rng)
        group_kinds = ["turn", "speed", "stop_restart"]
        group_kinds.append(str(rng.choice(("turn", "speed"))))
        rng.shuffle(group_kinds)
        for time_s, kind in zip(times, group_kinds):
            if kind == "turn":
                events.append(_turn_event(time_s, dt, rng, config))
            elif kind == "speed":
                events.append(_speed_event(time_s, dt, rng, config))
            else:
                events.extend(_stop_restart_events(time_s, dt, rng, config))

    events.sort(key=lambda event: event.step)
    event_steps = {event.step for event in events}
    if len(event_steps) != len(events):
        raise RuntimeError("seeded V2 event schedule contains duplicate steps")

    dropout_intervals: List[Tuple[int, int]] = []
    if process == "combined_change":
        observation = config["observation"]
        count = int(observation["combined_dropout_count"])
        candidate_events = [event for event in events if event.kind != "restart_forward"]
        selected = np.linspace(
            0, max(0, len(candidate_events) - 1), count, dtype=int
        )
        for slot, event_index in enumerate(selected):
            anchor = candidate_events[int(event_index)].time_s
            start_time = anchor + (0.20 if slot % 2 == 0 else -0.15)
            duration = _sample_range(
                rng, observation["dropout_duration_s"]
            )
            start = max(1, _snap_step(start_time, dt))
            end = min(steps, max(start + 1, _snap_step(start_time + duration, dt)))
            dropout_intervals.append((start, end))
        dropout_intervals.sort()
        # The event layout keeps them separated, but fail closed if a future
        # configuration violates that assumption.
        for index in range(1, len(dropout_intervals)):
            if dropout_intervals[index][0] < dropout_intervals[index - 1][1]:
                raise RuntimeError("seeded dropout intervals overlap")

    program = MotionProgram(
        events=tuple(events),
        dropout_intervals=tuple(dropout_intervals),
    )
    program.validate(steps)
    return program


def _initial_state(process: str, config: Mapping[str, object]):
    initial = config["initial_conditions"][process]
    position = np.asarray(initial["position_m"], dtype=np.float64)
    speed = float(initial["speed_mps"])
    heading = float(np.deg2rad(float(initial["heading_deg"])))
    velocity = speed * np.asarray((np.cos(heading), np.sin(heading)))
    return np.concatenate((position, velocity)), speed, heading


def generate_state_machine_trajectory(
    process: str,
    seed: int,
    noise_profile: NoiseProfile,
    config: Mapping[str, object],
) -> ObstacleTrajectory:
    validate_v2_config(config)
    noise_profile.validate()
    program = build_motion_program(process, seed, config)
    dt = float(config["trajectory"]["dt"])
    duration = float(config["trajectory"]["duration_s"])
    steps = int(round(duration / dt))
    times = np.arange(steps + 1, dtype=np.float64) * dt
    states = np.empty((steps + 1, 4), dtype=np.float64)
    states[0], base_speed, heading = _initial_state(process, config)
    modes = ["cv"] * (steps + 1)
    changes = np.zeros(steps + 1, dtype=bool)
    limits = config["physical_limits"]
    maximum_speed = float(limits["maximum_speed_mps"])
    maximum_acceleration = float(limits["maximum_acceleration_mps2"])
    maximum_braking = float(limits["maximum_braking_mps2"])
    target_speed = float(base_speed)
    stop_active = False
    turn_rate = 0.0
    turn_end_step = -1
    turn_mode = "cv"
    reverse_label_until = -1
    events_by_step = {event.step: event for event in program.events}
    process_rng = _rng(seed, process, stream=2)

    for index in range(1, steps + 1):
        if index == turn_end_step:
            turn_rate = 0.0
            turn_mode = "cv"
            changes[index] = True
        event = events_by_step.get(index)
        if event is not None:
            changes[index] = True
            if event.kind in ("turn_left", "turn_right"):
                turn_rate = float(event.parameters["turn_rate_radps"])
                turn_end_step = min(
                    steps,
                    index
                    + max(
                        1,
                        int(round(float(event.parameters["duration_s"]) / dt)),
                    ),
                )
                turn_mode = (
                    "turn_l" if event.kind == "turn_left" else "turn_r"
                )
            elif event.kind in ("accelerate", "decelerate"):
                target_speed = float(
                    np.clip(
                        target_speed * float(event.parameters["speed_scale"]),
                        0.10,
                        maximum_speed,
                    )
                )
            elif event.kind == "stop":
                target_speed = 0.0
                stop_active = True
            elif event.kind in ("restart_forward", "reverse"):
                if event.kind == "reverse":
                    heading = _wrap_angle(heading + np.pi)
                    reverse_label_until = index + max(1, int(round(0.6 / dt)))
                target_speed = float(
                    np.clip(
                        base_speed
                        * float(event.parameters["restart_speed_scale"]),
                        0.10,
                        maximum_speed,
                    )
                )
                stop_active = False
            else:
                raise RuntimeError("unknown scheduled event")

        if turn_rate != 0.0 and index < turn_end_step:
            heading = _wrap_angle(heading + turn_rate * dt)

        previous_state = states[index - 1]
        previous_velocity = previous_state[2:]
        previous_speed = float(np.linalg.norm(previous_velocity))
        desired_velocity = target_speed * np.asarray(
            (np.cos(heading), np.sin(heading)), dtype=np.float64
        )
        requested_acceleration = (desired_velocity - previous_velocity) / dt
        acceleration_limit = (
            maximum_braking
            if target_speed + 1.0e-12 < previous_speed
            else maximum_acceleration
        )
        commanded_acceleration = _clip_norm(
            requested_acceleration, acceleration_limit
        )
        process_scale = (
            float(noise_profile.acceleration_std)
            * (0.08 if stop_active else 1.0)
        )
        process_acceleration = process_rng.normal(size=2) * process_scale
        total_acceleration = _clip_norm(
            commanded_acceleration + process_acceleration,
            acceleration_limit,
        )
        velocity = previous_velocity + total_acceleration * dt
        velocity = _clip_norm(velocity, maximum_speed)
        candidate_speed = float(np.linalg.norm(velocity))
        if previous_speed > 1.0e-12 and candidate_speed > 1.0e-12:
            previous_heading = float(
                np.arctan2(previous_velocity[1], previous_velocity[0])
            )
            candidate_heading = float(np.arctan2(velocity[1], velocity[0]))
            heading_increment = _wrap_angle(candidate_heading - previous_heading)
            maximum_heading_increment = (
                float(limits["maximum_moving_yaw_rate_radps"]) * dt
            )
            constrained_increment = float(
                np.clip(
                    heading_increment,
                    -maximum_heading_increment,
                    maximum_heading_increment,
                )
            )
            constrained_heading = previous_heading + constrained_increment
            velocity = candidate_speed * np.asarray(
                (np.cos(constrained_heading), np.sin(constrained_heading)),
                dtype=np.float64,
            )
            # Reducing the angular increment cannot increase the distance from
            # the previous velocity when both vectors retain their norms, so
            # the acceleration projection above remains valid.
        realized_acceleration = (velocity - previous_velocity) / dt
        position = (
            previous_state[:2]
            + previous_velocity * dt
            + 0.5 * realized_acceleration * dt * dt
        )
        states[index, :2] = position
        states[index, 2:] = velocity

        if stop_active:
            modes[index] = "brake_stop"
        elif index <= reverse_label_until:
            modes[index] = "reverse"
        elif turn_rate != 0.0 and index < turn_end_step:
            modes[index] = turn_mode
        else:
            modes[index] = "cv"

    observation_rng = _rng(seed, process, stream=3)
    observations = states[:, :2] + (
        float(noise_profile.observation_std)
        * observation_rng.normal(size=(steps + 1, 2))
    )
    observed_mask = np.ones(steps + 1, dtype=bool)
    for start, end in program.dropout_intervals:
        observed_mask[start:end] = False
    observations[~observed_mask] = np.nan

    metadata: Dict[str, object] = {
        "schema_version": 2,
        "generator_version": GENERATOR_VERSION,
        "process": process,
        "seed": int(seed),
        "dt": dt,
        "duration_s": duration,
        "steps": steps,
        "noise_profile": noise_profile.name,
        "process_acceleration_std": float(noise_profile.acceleration_std),
        "observation_position_std": float(noise_profile.observation_std),
        "events": [event.to_dict() for event in program.events],
        "dropout_intervals": [
            {
                "start_step": int(start),
                "end_step_exclusive": int(end),
                "start_time_s": float(times[start]),
                "end_time_s": float(times[end]),
            }
            for start, end in program.dropout_intervals
        ],
        "physical_limits": dict(limits),
        "event_stream": "SeedSequence(seed, process_code, 200, 1)",
        "process_stream": "SeedSequence(seed, process_code, 200, 2)",
        "observation_stream": "SeedSequence(seed, process_code, 200, 3)",
    }
    result = ObstacleTrajectory(
        process=process,
        seed=int(seed),
        noise_profile=noise_profile.name,
        times=times,
        states=states,
        observations=observations,
        observed_mask=observed_mask,
        true_modes=tuple(modes),
        change_flags=changes,
        metadata=metadata,
    )
    result.validate()
    return result


def audit_state_machine_trajectory(
    trajectory: ObstacleTrajectory,
    config: Mapping[str, object],
) -> Dict[str, object]:
    dt = float(config["trajectory"]["dt"])
    limits = config["physical_limits"]
    velocities = trajectory.states[:, 2:]
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = np.diff(velocities, axis=0) / dt
    acceleration_norms = np.linalg.norm(accelerations, axis=1)
    headings = np.unwrap(np.arctan2(velocities[:, 1], velocities[:, 0]))
    moving = (
        speeds[:-1]
        >= float(limits["yaw_rate_audit_minimum_speed_mps"])
    ) & (
        speeds[1:]
        >= float(limits["yaw_rate_audit_minimum_speed_mps"])
    )
    yaw_rates = np.abs(np.diff(headings) / dt)
    moving_yaw_rates = yaw_rates[moving]
    events = list(trajectory.metadata["events"])
    event_kinds = [str(event["kind"]) for event in events]
    stop_speeds = speeds[
        np.asarray(trajectory.true_modes, dtype=object) == "brake_stop"
    ]
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
        "minimum_stop_mode_speed_mps": (
            float(stop_speeds.min()) if stop_speeds.size else None
        ),
        "event_count": len(events),
        "event_kinds": event_kinds,
        "change_flag_count": int(trajectory.change_flags.sum()),
        "dropout_interval_count": len(trajectory.metadata["dropout_intervals"]),
        "missing_observation_count": int((~trajectory.observed_mask).sum()),
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
            <= float(limits["maximum_moving_yaw_rate_radps"]) + 0.08
        ),
    }


__all__: Sequence[str] = (
    "GENERATOR_VERSION",
    "MotionEvent",
    "MotionProgram",
    "audit_state_machine_trajectory",
    "build_motion_program",
    "generate_state_machine_trajectory",
    "validate_v2_config",
)
