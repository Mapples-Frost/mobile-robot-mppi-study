"""Named state/action spaces used instead of hard-coded dimensions."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class StateSpec:
    names: Tuple[str, ...]
    periodic_indices: Tuple[int, ...] = ()
    position_indices: Tuple[int, int] = (0, 1)

    def __post_init__(self) -> None:
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("state names must be non-empty and unique")
        if any(index < 0 or index >= self.dimension for index in self.periodic_indices):
            raise ValueError("periodic state index is out of range")

    @property
    def dimension(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def validate(self, state: Sequence[float]) -> np.ndarray:
        value = np.asarray(state, dtype=np.float64).reshape(-1)
        if value.shape != (self.dimension,) or not np.isfinite(value).all():
            raise ValueError("state must be finite with shape (%d,)" % self.dimension)
        return value

    def error(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
        for index in self.periodic_indices:
            error[..., index] = np.arctan2(np.sin(error[..., index]), np.cos(error[..., index]))
        return error


@dataclass(frozen=True)
class ActionSpec:
    names: Tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    rate_limits: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("action names must be non-empty and unique")
        lower = np.asarray(self.lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(self.upper, dtype=np.float64).reshape(-1)
        if lower.shape != (len(self.names),) or upper.shape != lower.shape:
            raise ValueError("action bounds do not match action dimension")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
            raise ValueError("action bounds must be finite and lower < upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        if self.rate_limits is not None:
            rate = np.asarray(self.rate_limits, dtype=np.float64).reshape(-1)
            if rate.shape != lower.shape or np.any(rate <= 0.0) or not np.isfinite(rate).all():
                raise ValueError("rate limits must be positive and match action dimension")
            object.__setattr__(self, "rate_limits", rate)

    @property
    def dimension(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def clip(self, action: np.ndarray, previous: Optional[np.ndarray] = None, dt: Optional[float] = None) -> np.ndarray:
        value = np.clip(np.asarray(action, dtype=np.float64), self.lower, self.upper)
        if self.rate_limits is not None and previous is not None and dt is not None:
            previous_value = np.asarray(previous, dtype=np.float64)
            delta = self.rate_limits * float(dt)
            value = np.clip(value, previous_value - delta, previous_value + delta)
        return value


def unicycle_state() -> StateSpec:
    return StateSpec(("x", "y", "theta"), periodic_indices=(2,))


def dynamic_unicycle_state() -> StateSpec:
    return StateSpec(("x", "y", "theta", "v", "omega"), periodic_indices=(2,))


def wheel_augmented_state() -> StateSpec:
    return StateSpec(
        ("x", "y", "theta", "v", "omega", "wheel_left", "wheel_right"),
        periodic_indices=(2,),
    )


def body_velocity_action(v_limits=(-0.15, 0.45), omega_limit=1.2) -> ActionSpec:
    return ActionSpec(
        ("v_cmd", "omega_cmd"),
        lower=np.asarray((v_limits[0], -abs(omega_limit)), dtype=np.float64),
        upper=np.asarray((v_limits[1], abs(omega_limit)), dtype=np.float64),
    )


def state_spec_from_config(config: Mapping[str, object]) -> StateSpec:
    preset = str(config.get("preset", "dynamic_unicycle_5"))
    if preset == "unicycle_3":
        return unicycle_state()
    if preset == "dynamic_unicycle_5":
        return dynamic_unicycle_state()
    if preset == "wheel_augmented_7":
        return wheel_augmented_state()
    names = tuple(str(value) for value in config["names"])
    periodic = tuple(int(value) for value in config.get("periodic_indices", ()))
    return StateSpec(names, periodic_indices=periodic)


def action_spec_from_config(config: Mapping[str, object]) -> ActionSpec:
    names = tuple(str(value) for value in config.get("names", ("v_cmd", "omega_cmd")))
    return ActionSpec(
        names=names,
        lower=np.asarray(config.get("lower", (-0.15, -1.2)), dtype=np.float64),
        upper=np.asarray(config.get("upper", (0.45, 1.2)), dtype=np.float64),
        rate_limits=(
            None if config.get("rate_limits") is None
            else np.asarray(config["rate_limits"], dtype=np.float64)
        ),
    )
