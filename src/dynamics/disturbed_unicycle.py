"""Configurable true-plant model for controlled model-mismatch studies."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import numpy as np

from .interfaces import (
    validate_derivative,
    validate_state_control,
    validate_time,
    validate_vector,
)
from .nominal_unicycle import NominalUnicycle


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError("{} must be a finite scalar".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("{} must be a finite scalar".format(name)) from exc
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _finite_tuple(value: Sequence[float], length: int, name: str) -> Tuple[float, ...]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError("{} must be a sequence of length {}".format(name, length)) from exc
    if len(values) != length:
        raise ValueError("{} must have length {}, got {}".format(name, length, len(values)))
    return tuple(_finite_scalar(item, "{}[{}]".format(name, index)) for index, item in enumerate(values))


@dataclass(frozen=True)
class DisturbanceConfig:
    """Serializable switches and parameters for a disturbed unicycle plant.

    Neutral defaults make the plant exactly equal to :class:`NominalUnicycle`.
    Each disturbance can also be disabled explicitly while retaining its
    configured magnitude, which is useful for controlled ablations.
    """

    velocity_gain: float = 1.0
    yaw_gain: float = 1.0
    yaw_bias: float = 0.0
    control_delay_steps: int = 0
    world_disturbance_amplitude: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_disturbance_frequency: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_disturbance_phase: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    state_disturbance_gain: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    enable_velocity_gain: bool = True
    enable_yaw_gain: bool = True
    enable_yaw_bias: bool = True
    enable_control_delay: bool = True
    enable_world_disturbance: bool = True
    enable_state_disturbance: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "velocity_gain", _finite_scalar(self.velocity_gain, "velocity_gain")
        )
        object.__setattr__(self, "yaw_gain", _finite_scalar(self.yaw_gain, "yaw_gain"))
        object.__setattr__(self, "yaw_bias", _finite_scalar(self.yaw_bias, "yaw_bias"))
        if isinstance(self.control_delay_steps, (bool, np.bool_)) or not isinstance(
            self.control_delay_steps, (int, np.integer)
        ):
            raise TypeError("control_delay_steps must be a non-negative integer")
        if int(self.control_delay_steps) < 0:
            raise ValueError("control_delay_steps must be non-negative")
        object.__setattr__(self, "control_delay_steps", int(self.control_delay_steps))
        object.__setattr__(
            self,
            "world_disturbance_amplitude",
            _finite_tuple(
                self.world_disturbance_amplitude, 3, "world_disturbance_amplitude"
            ),
        )
        object.__setattr__(
            self,
            "world_disturbance_frequency",
            _finite_tuple(
                self.world_disturbance_frequency, 3, "world_disturbance_frequency"
            ),
        )
        object.__setattr__(
            self,
            "world_disturbance_phase",
            _finite_tuple(self.world_disturbance_phase, 3, "world_disturbance_phase"),
        )
        object.__setattr__(
            self,
            "state_disturbance_gain",
            _finite_tuple(self.state_disturbance_gain, 3, "state_disturbance_gain"),
        )
        for name in (
            "enable_velocity_gain",
            "enable_yaw_gain",
            "enable_yaw_bias",
            "enable_control_delay",
            "enable_world_disturbance",
            "enable_state_disturbance",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError("{} must be boolean".format(name))
            object.__setattr__(self, name, bool(getattr(self, name)))

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-friendly disturbance metadata."""

        result = asdict(self)
        for key in (
            "world_disturbance_amplitude",
            "world_disturbance_frequency",
            "world_disturbance_phase",
            "state_disturbance_gain",
        ):
            result[key] = list(result[key])
        return result


class DisturbedUnicycle:
    """Unicycle true plant with independently configurable mismatch terms.

    The instantaneous derivative applies gain/bias, time-varying world-frame,
    and state-dependent disturbances.  State-dependent terms are
    ``[k_x*x, k_y*y, k_theta*sin(theta)]`` so the angular contribution remains
    periodic.

    Discrete command delay is intentionally *not* implemented inside
    :meth:`derivative`: a delay queue is history-dependent and therefore is not
    Markov in ``[x, y, theta]``.  :meth:`step` selects one delayed command,
    integrates all Euler/RK4 stages with that fixed applied command, and only
    then advances the queue once.  Use an augmented state if a planner must
    model delay as part of its rollout dynamics.
    """

    state_dim: int = 3
    control_dim: int = 2
    angle_indices: Tuple[int, ...] = (2,)

    def __init__(
        self,
        config: Optional[DisturbanceConfig] = None,
        nominal_dynamics: Optional[NominalUnicycle] = None,
    ) -> None:
        self.config = DisturbanceConfig() if config is None else config
        if not isinstance(self.config, DisturbanceConfig):
            raise TypeError("config must be a DisturbanceConfig")
        self.nominal_dynamics = (
            NominalUnicycle() if nominal_dynamics is None else nominal_dynamics
        )
        if (
            self.nominal_dynamics.state_dim != self.state_dim
            or self.nominal_dynamics.control_dim != self.control_dim
            or tuple(self.nominal_dynamics.angle_indices) != self.angle_indices
        ):
            raise ValueError("nominal_dynamics is incompatible with unicycle dimensions")
        self._control_queue: Deque[np.ndarray] = deque()
        self._initial_control = np.zeros(self.control_dim, dtype=np.float64)
        self._last_applied_control = self._initial_control.copy()
        self.reset()

    @property
    def delay_steps(self) -> int:
        return (
            self.config.control_delay_steps
            if self.config.enable_control_delay
            else 0
        )

    @property
    def has_control_delay(self) -> bool:
        return self.delay_steps > 0

    @property
    def is_zero_mismatch(self) -> bool:
        """Whether both instantaneous and history-dependent mismatch are neutral."""

        return bool(not self.has_instantaneous_mismatch and not self.has_control_delay)

    @property
    def has_instantaneous_mismatch(self) -> bool:
        """Whether :meth:`derivative` differs from the nominal derivative."""

        velocity_neutral = (
            not self.config.enable_velocity_gain
            or self.config.velocity_gain == 1.0
        )
        yaw_gain_neutral = (
            not self.config.enable_yaw_gain or self.config.yaw_gain == 1.0
        )
        yaw_bias_neutral = (
            not self.config.enable_yaw_bias or self.config.yaw_bias == 0.0
        )
        world_neutral = (
            not self.config.enable_world_disturbance
            or self.config.world_disturbance_amplitude == (0.0, 0.0, 0.0)
        )
        state_neutral = (
            not self.config.enable_state_disturbance
            or self.config.state_disturbance_gain == (0.0, 0.0, 0.0)
        )
        return not bool(
            velocity_neutral
            and yaw_gain_neutral
            and yaw_bias_neutral
            and world_neutral
            and state_neutral
        )

    @property
    def last_applied_control(self) -> np.ndarray:
        return self._last_applied_control.copy()

    @property
    def pending_controls(self) -> Tuple[np.ndarray, ...]:
        """Copies of controls currently in the delay queue, oldest first."""

        return tuple(control.copy() for control in self._control_queue)

    def reset(self, initial_control: Optional[Any] = None) -> None:
        """Reset delay history, filling it with a safe initial command.

        For delay ``N``, the first ``N`` calls to :meth:`step` apply this
        initial command (zero by default).  Thereafter step ``k`` applies the
        command submitted at step ``k-N``.
        """

        if initial_control is None:
            control = np.zeros(self.control_dim, dtype=np.float64)
        else:
            control = validate_vector(initial_control, self.control_dim, "initial_control")
        self._initial_control = control.copy()
        self._control_queue = deque(
            control.copy() for _ in range(self.delay_steps)
        )
        self._last_applied_control = control.copy()

    def clone(self) -> "DisturbedUnicycle":
        """Return an independent plant including an exact copy of delay history."""

        return copy.deepcopy(self)

    def _world_disturbance(self, time: float) -> np.ndarray:
        if not self.config.enable_world_disturbance:
            return np.zeros(self.state_dim, dtype=np.float64)
        amplitude = np.asarray(
            self.config.world_disturbance_amplitude, dtype=np.float64
        )
        frequency = np.asarray(
            self.config.world_disturbance_frequency, dtype=np.float64
        )
        phase = np.asarray(self.config.world_disturbance_phase, dtype=np.float64)
        return amplitude * np.sin(frequency * time + phase)

    def _state_disturbance(self, state: np.ndarray) -> np.ndarray:
        if not self.config.enable_state_disturbance:
            return np.zeros(self.state_dim, dtype=np.float64)
        gain_x, gain_y, gain_theta = self.config.state_disturbance_gain
        return np.asarray(
            [gain_x * state[0], gain_y * state[1], gain_theta * np.sin(state[2])],
            dtype=np.float64,
        )

    def derivative_components(
        self,
        state: Any,
        control: Any,
        time: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Expose additive terms for dataset checks and ablation debugging."""

        state_array, control_array = validate_state_control(self, state, control)
        numeric_time = validate_time(time)
        evaluation_time = 0.0 if numeric_time is None else numeric_time
        nominal = self.nominal_dynamics.derivative(
            state_array, control_array, time=numeric_time
        )

        effective_control = control_array.copy()
        if self.config.enable_velocity_gain:
            effective_control[0] *= self.config.velocity_gain
        if self.config.enable_yaw_gain:
            effective_control[1] *= self.config.yaw_gain
        if self.config.enable_yaw_bias:
            effective_control[1] += self.config.yaw_bias
        gain_and_bias_model = self.nominal_dynamics.derivative(
            state_array, effective_control, time=numeric_time
        )
        gain_and_bias = gain_and_bias_model - nominal
        world = self._world_disturbance(evaluation_time)
        state_dependent = self._state_disturbance(state_array)
        total = validate_derivative(
            nominal + gain_and_bias + world + state_dependent, self.state_dim
        )
        return {
            "nominal": nominal,
            "gain_and_bias": gain_and_bias,
            "world": world,
            "state_dependent": state_dependent,
            "total": total,
        }

    def derivative(
        self,
        state: Any,
        control: Any,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Return true derivative for the supplied *already applied* control.

        This method never reads or mutates the discrete delay queue.  Therefore
        an oracle residual computed from it is exact for the instantaneous
        plant given the applied control, while command delay remains a
        plant-step/history effect.
        """

        state_array, control_array = validate_state_control(self, state, control)
        numeric_time = validate_time(time)
        if not self.has_instantaneous_mismatch:
            # Preserve bit-for-bit nominal behavior for the zero-mismatch case.
            return self.nominal_dynamics.derivative(
                state_array, control_array, time=numeric_time
            )
        return self.derivative_components(
            state_array, control_array, time=numeric_time
        )["total"]

    def preview_applied_control(self, command: Any) -> np.ndarray:
        """Return the next applied command without changing delay history."""

        command_array = validate_vector(command, self.control_dim, "control")
        if not self.has_control_delay:
            return command_array.copy()
        return self._control_queue[0].copy()

    def _commit_command(self, command: np.ndarray, applied: np.ndarray) -> None:
        if self.has_control_delay:
            self._control_queue.popleft()
            self._control_queue.append(command.copy())
        self._last_applied_control = applied.copy()

    def step(
        self,
        state: Any,
        control: Any,
        dt: float,
        method: str = "rk4",
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Execute one plant interval and advance delay history exactly once."""

        from .integrators import integrate_step

        state_array, command = validate_state_control(self, state, control)
        applied = self.preview_applied_control(command)
        # Commit only after successful integration, so validation/numeric errors
        # do not partially mutate the plant's history.
        next_state = integrate_step(
            self,
            state_array,
            applied,
            dt,
            method=method,
            time=time,
        )
        self._commit_command(command, applied)
        return next_state

    def disturbance_metadata(self) -> Dict[str, Any]:
        """JSON-friendly description suitable for every collected dataset."""

        return {
            "model": self.__class__.__name__,
            "state_dim": self.state_dim,
            "control_dim": self.control_dim,
            "angle_indices": list(self.angle_indices),
            "config": self.config.to_dict(),
            "delay_is_non_markov_without_history": self.has_control_delay,
        }


__all__ = ["DisturbanceConfig", "DisturbedUnicycle"]
