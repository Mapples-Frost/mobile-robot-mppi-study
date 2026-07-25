"""Seeded stochastic motion processes for one planar dynamic obstacle.

The state is ``[px, py, vx, vy]``.  Motion/process noise changes the true
state, while observation noise changes only the measured position.  Their
random streams are deliberately separate so experiments can vary either source
without silently changing the other one.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


PROCESS_NAMES = (
    "noisy_cv",
    "speed_change",
    "direction_change",
    "combined_change",
)

_PROCESS_CODES = {name: index + 1 for index, name in enumerate(PROCESS_NAMES)}


@dataclass(frozen=True)
class NoiseProfile:
    name: str
    acceleration_std: float
    observation_std: float

    def validate(self) -> None:
        values = (self.acceleration_std, self.observation_std)
        if not np.isfinite(values).all() or any(value < 0.0 for value in values):
            raise ValueError("noise standard deviations must be finite and non-negative")
        if not self.name:
            raise ValueError("noise profile name must be non-empty")


NOISE_PROFILES: Dict[str, NoiseProfile] = {
    "low": NoiseProfile("low", acceleration_std=0.025, observation_std=0.025),
    "medium": NoiseProfile(
        "medium", acceleration_std=0.075, observation_std=0.075
    ),
}


@dataclass(frozen=True)
class ObstacleTrajectory:
    process: str
    seed: int
    noise_profile: str
    times: np.ndarray
    states: np.ndarray
    observations: np.ndarray
    observed_mask: np.ndarray
    true_modes: Tuple[str, ...]
    change_flags: np.ndarray
    metadata: Mapping[str, object]

    def validate(self) -> None:
        count = self.times.shape[0]
        registered_extension = (
            self.metadata.get("generator_version") == "recurrent_semimarkov_v3"
            and self.metadata.get("process") == self.process
            and self.process
            in (
                "stochastic_cruise",
                "stochastic_shuttle",
                "branching_patrol",
                "hybrid_patrol",
            )
        )
        if self.process not in PROCESS_NAMES and not registered_extension:
            raise ValueError("unknown obstacle process")
        if self.states.shape != (count, 4):
            raise ValueError("obstacle states must have shape (T, 4)")
        if self.observations.shape != (count, 2):
            raise ValueError("obstacle observations must have shape (T, 2)")
        if self.observed_mask.shape != (count,):
            raise ValueError("observation mask must have shape (T,)")
        if self.change_flags.shape != (count,):
            raise ValueError("change flags must have shape (T,)")
        if len(self.true_modes) != count:
            raise ValueError("true mode count must match trajectory length")
        if count < 2 or not np.isfinite(self.times).all():
            raise ValueError("trajectory times must be finite and non-empty")
        if not np.all(np.diff(self.times) > 0.0):
            raise ValueError("trajectory times must increase strictly")
        if not np.isfinite(self.states).all():
            raise FloatingPointError("true obstacle states contain NaN or Inf")
        if not np.isfinite(self.observations[self.observed_mask]).all():
            raise FloatingPointError("available observations contain NaN or Inf")
        if not np.isnan(self.observations[~self.observed_mask]).all():
            raise ValueError("missing observations must be represented by NaN")


def _rng(seed: int, process: str, stream: int) -> np.random.RandomState:
    if not 0 <= int(seed) <= 2 ** 32 - 1:
        raise ValueError("seed must be in [0, 2**32 - 1]")
    sequence = np.random.SeedSequence(
        [int(seed), int(_PROCESS_CODES[process]), int(stream)]
    )
    child_seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
    return np.random.RandomState(child_seed)


def _initial_state(process: str) -> np.ndarray:
    states = {
        "noisy_cv": (-3.0, -1.25, 0.72, 0.24),
        "speed_change": (-3.0, 0.0, 0.78, 0.0),
        "direction_change": (-2.3, -2.1, 0.56, 0.52),
        "combined_change": (-3.0, -1.65, 0.74, 0.31),
    }
    return np.asarray(states[process], dtype=np.float64)


def _rotate(vector: np.ndarray, angle_rad: float) -> np.ndarray:
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    matrix = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    return matrix.dot(np.asarray(vector, dtype=np.float64))


def _event_plan(
    process: str, steps: int, event_rng: np.random.RandomState
) -> Dict[str, object]:
    if process == "noisy_cv":
        return {"change_steps": []}
    low = max(4, int(round(0.28 * steps)))
    high = min(steps - 4, int(round(0.62 * steps)))
    first = int(event_rng.randint(low, high + 1))
    plan: Dict[str, object] = {"change_steps": [first]}
    if process == "speed_change":
        plan["speed_event"] = str(
            event_rng.choice(("accelerate", "decelerate", "stop"))
        )
    elif process == "direction_change":
        angle_deg = float(event_rng.choice((-90.0, -45.0, 45.0, 90.0)))
        plan["turn_angle_deg"] = angle_deg
    elif process == "combined_change":
        second_low = min(steps - 3, first + max(5, steps // 6))
        second_high = min(steps - 2, first + max(8, steps // 3))
        second = int(event_rng.randint(second_low, second_high + 1))
        plan["change_steps"] = [first, second]
        plan["turn_angle_deg"] = float(
            event_rng.choice((-90.0, -45.0, 45.0, 90.0))
        )
        plan["speed_event"] = str(
            event_rng.choice(("decelerate", "stop", "reverse"))
        )
        dropout_start = min(steps - 2, first + 2)
        dropout_length = int(event_rng.randint(3, 7))
        plan["dropout_start"] = dropout_start
        plan["dropout_end"] = min(steps, dropout_start + dropout_length)
    return plan


def _apply_speed_event(velocity: np.ndarray, event: str) -> np.ndarray:
    if event == "accelerate":
        return 1.55 * velocity
    if event == "decelerate":
        return 0.42 * velocity
    if event == "stop":
        return np.zeros(2, dtype=np.float64)
    if event == "reverse":
        return -0.85 * velocity
    raise ValueError("unknown speed event: %s" % event)


def generate_obstacle_trajectory(
    process: str,
    seed: int,
    noise_profile: NoiseProfile,
    dt: float = 0.1,
    steps: int = 80,
) -> ObstacleTrajectory:
    """Generate one truth/observation pair with reproducible random streams."""

    if process not in PROCESS_NAMES:
        raise ValueError("process must be one of %s" % (PROCESS_NAMES,))
    noise_profile.validate()
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if int(steps) < 12:
        raise ValueError("steps must be at least 12")
    steps = int(steps)

    event_rng = _rng(seed, process, stream=1)
    motion_rng = _rng(seed, process, stream=2)
    observation_rng = _rng(seed, process, stream=3)
    plan = _event_plan(process, steps, event_rng)

    times = np.arange(steps + 1, dtype=np.float64) * float(dt)
    states = np.empty((steps + 1, 4), dtype=np.float64)
    states[0] = _initial_state(process)
    modes = ["cv"] * (steps + 1)
    changes = np.zeros(steps + 1, dtype=bool)

    change_steps = tuple(int(value) for value in plan.get("change_steps", ()))
    turn_mode_until = -1
    stopped = False
    for index in range(1, steps + 1):
        velocity = states[index - 1, 2:].copy()
        if index in change_steps:
            changes[index] = True
            if (
                process in ("direction_change", "combined_change")
                and index == change_steps[0]
            ):
                angle_deg = float(plan["turn_angle_deg"])
                velocity = _rotate(velocity, np.deg2rad(angle_deg))
                modes[index] = "turn_l" if angle_deg > 0.0 else "turn_r"
                turn_mode_until = index + 4
            if process == "speed_change" or (
                process == "combined_change" and index == change_steps[-1]
            ):
                speed_event = str(plan["speed_event"])
                velocity = _apply_speed_event(velocity, speed_event)
                stopped = speed_event == "stop"
                modes[index] = "brake_stop" if stopped else "cv"
        elif index <= turn_mode_until:
            modes[index] = modes[index - 1]
        elif stopped:
            modes[index] = "brake_stop"
            velocity *= 0.72
        else:
            modes[index] = "cv"

        standardized_acceleration = motion_rng.normal(size=2)
        acceleration_scale = float(noise_profile.acceleration_std)
        if stopped:
            acceleration_scale *= 0.12
        acceleration = standardized_acceleration * acceleration_scale
        states[index, :2] = (
            states[index - 1, :2]
            + velocity * dt
            + 0.5 * acceleration * dt * dt
        )
        states[index, 2:] = velocity + acceleration * dt

    observation_noise = observation_rng.normal(size=(steps + 1, 2))
    observations = states[:, :2] + (
        float(noise_profile.observation_std) * observation_noise
    )
    observed_mask = np.ones(steps + 1, dtype=bool)
    if process == "combined_change":
        start = int(plan["dropout_start"])
        end = int(plan["dropout_end"])
        observed_mask[start:end] = False
        observations[~observed_mask] = np.nan

    metadata: Dict[str, object] = {
        "schema_version": 1,
        "process": process,
        "seed": int(seed),
        "dt": float(dt),
        "steps": int(steps),
        "noise_profile": noise_profile.name,
        "process_acceleration_std": float(noise_profile.acceleration_std),
        "observation_position_std": float(noise_profile.observation_std),
        "change_steps": list(change_steps),
        "change_times_s": [float(times[index]) for index in change_steps],
        "process_stream": "SeedSequence(seed, process_code, 2)",
        "observation_stream": "SeedSequence(seed, process_code, 3)",
    }
    for name in (
        "speed_event",
        "turn_angle_deg",
        "dropout_start",
        "dropout_end",
    ):
        if name in plan:
            metadata[name] = plan[name]

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


def noise_profiles_from_mapping(
    values: Mapping[str, Mapping[str, float]]
) -> Dict[str, NoiseProfile]:
    profiles: Dict[str, NoiseProfile] = {}
    for name, config in values.items():
        profile = NoiseProfile(
            name=str(name),
            acceleration_std=float(config["acceleration_std"]),
            observation_std=float(config["observation_std"]),
        )
        profile.validate()
        profiles[profile.name] = profile
    if not profiles:
        raise ValueError("at least one noise profile is required")
    return profiles


__all__: Sequence[str] = (
    "NOISE_PROFILES",
    "PROCESS_NAMES",
    "NoiseProfile",
    "ObstacleTrajectory",
    "generate_obstacle_trajectory",
    "noise_profiles_from_mapping",
)
