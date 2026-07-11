"""Torch-free mean-control priors for MPPI sampling.

The classes in this module only construct the deterministic control sequence
around which an MPPI implementation may sample.  They deliberately do not
know about costs, obstacles, learned models, or PyTorch.
"""

from __future__ import annotations

import math
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable


Control = Tuple[float, float]
Bounds = Tuple[float, float]


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError("{} must be a finite scalar".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("{} must be a finite scalar".format(name)) from exc
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _pair(value: Sequence[float], name: str) -> Tuple[float, float]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError("{} must be a sequence of length 2".format(name)) from exc
    if len(items) != 2:
        raise ValueError("{} must have length 2, got {}".format(name, len(items)))
    return (
        _finite_scalar(items[0], "{}[0]".format(name)),
        _finite_scalar(items[1], "{}[1]".format(name)),
    )


def _bounds(value: Optional[Sequence[float]], name: str) -> Optional[Bounds]:
    if value is None:
        return None
    lower, upper = _pair(value, name)
    if lower > upper:
        raise ValueError("{} lower bound must not exceed its upper bound".format(name))
    return lower, upper


def _horizon(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("horizon must be a non-negative integer")
    if value < 0:
        raise ValueError("horizon must be non-negative")
    return value


def _state(value: Sequence[float]) -> Tuple[float, float, float]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError("state must be a sequence of length 3") from exc
    if len(items) != 3:
        raise ValueError("state must have length 3, got {}".format(len(items)))
    return (
        _finite_scalar(items[0], "state[0]"),
        _finite_scalar(items[1], "state[1]"),
        _finite_scalar(items[2], "state[2]"),
    )


def _wrap_angle(theta: float) -> float:
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


@runtime_checkable
class SamplingPrior(Protocol):
    """Structural interface for an MPPI mean-control sequence provider."""

    def mean_control_sequence(
        self,
        state: Sequence[float],
        horizon: int,
    ) -> List[Control]:
        """Return exactly ``horizon`` finite ``(v, omega)`` controls."""


class PreviousSequencePrior:
    """Shift a previous solution by one interval for receding-horizon reuse.

    ``pad_control=None`` repeats the previous sequence's last control, the
    customary receding-horizon behavior.  An empty sequence falls back to
    ``(0.0, 0.0)``.  Optional per-channel limits are applied after shifting,
    truncating, and padding, so even an explicitly configured pad is clipped.
    Calling :meth:`mean_control_sequence` does not mutate the stored solution;
    the controller should call :meth:`update` after it obtains a new solution.
    """

    def __init__(
        self,
        previous_sequence: Sequence[Sequence[float]] = (),
        pad_control: Optional[Sequence[float]] = None,
        v_limits: Optional[Sequence[float]] = None,
        omega_limits: Optional[Sequence[float]] = None,
    ) -> None:
        self.pad_control = (
            None if pad_control is None else _pair(pad_control, "pad_control")
        )
        self.v_limits = _bounds(v_limits, "v_limits")
        self.omega_limits = _bounds(omega_limits, "omega_limits")
        self._previous_sequence: Tuple[Control, ...] = ()
        self.update(previous_sequence)

    @property
    def previous_sequence(self) -> Tuple[Control, ...]:
        """The validated stored solution, exposed as an immutable tuple."""

        return self._previous_sequence

    def update(self, previous_sequence: Sequence[Sequence[float]]) -> None:
        """Replace the stored solution after validating every control."""

        if isinstance(previous_sequence, (str, bytes)):
            raise TypeError("previous_sequence must be a sequence of controls")
        try:
            controls = tuple(previous_sequence)
        except TypeError as exc:
            raise TypeError(
                "previous_sequence must be a sequence of controls"
            ) from exc
        self._previous_sequence = tuple(
            _pair(control, "previous_sequence[{}]".format(index))
            for index, control in enumerate(controls)
        )

    def _clip(self, control: Control) -> Control:
        v, omega = control
        if self.v_limits is not None:
            v = max(self.v_limits[0], min(self.v_limits[1], v))
        if self.omega_limits is not None:
            omega = max(
                self.omega_limits[0], min(self.omega_limits[1], omega)
            )
        return float(v), float(omega)

    def mean_control_sequence(
        self,
        state: Sequence[float],
        horizon: int,
    ) -> List[Control]:
        # ``state`` is intentionally unused: this prior represents temporal
        # reuse, not state feedback.  Keeping it in the signature makes priors
        # interchangeable without imposing unnecessary state validation here.
        del state
        length = _horizon(horizon)
        shifted = list(self._previous_sequence[1:length + 1])

        if self.pad_control is not None:
            padding = self.pad_control
        elif self._previous_sequence:
            padding = self._previous_sequence[-1]
        else:
            padding = (0.0, 0.0)

        if len(shifted) < length:
            shifted.extend([padding] * (length - len(shifted)))
        return [self._clip(control) for control in shifted]

    def build(self, state: Sequence[float], horizon: int) -> List[Control]:
        """Compatibility alias for callers that use a generic builder name."""

        return self.mean_control_sequence(state, horizon)


class GoalWarmStartPrior:
    """Generate a short unicycle rollout steered toward one explicit goal.

    The steering and speed schedule matches the repository's established MPPI
    goal warm-start helper.  ``goal`` has no default on purpose: an experiment
    must pass its fixed goal explicitly instead of silently importing mutable
    runtime configuration.
    """

    def __init__(
        self,
        goal: Sequence[float],
        dt: float,
        v_max: float,
        omega_max: float,
        warm_start_prefix_steps: int = 8,
        nominal_tail_control: Sequence[float] = (1.0, 0.0),
    ) -> None:
        self.goal = _pair(goal, "goal")
        self.dt = _finite_scalar(dt, "dt")
        if self.dt <= 0.0:
            raise ValueError("dt must be greater than zero")
        self.v_max = _finite_scalar(v_max, "v_max")
        if self.v_max < 0.0:
            raise ValueError("v_max must be non-negative")
        self.omega_max = _finite_scalar(omega_max, "omega_max")
        if self.omega_max < 0.0:
            raise ValueError("omega_max must be non-negative")
        if (
            isinstance(warm_start_prefix_steps, bool)
            or not isinstance(warm_start_prefix_steps, int)
        ):
            raise TypeError("warm_start_prefix_steps must be a non-negative integer")
        if warm_start_prefix_steps < 0:
            raise ValueError("warm_start_prefix_steps must be non-negative")
        self.warm_start_prefix_steps = warm_start_prefix_steps
        self.nominal_tail_control = _pair(
            nominal_tail_control, "nominal_tail_control"
        )

    def mean_control_sequence(
        self,
        state: Sequence[float],
        horizon: int,
    ) -> List[Control]:
        x, y, theta = _state(state)
        length = _horizon(horizon)
        prefix_steps = min(length, self.warm_start_prefix_steps)
        controls: List[Control] = []

        goal_x, goal_y = self.goal
        for _ in range(prefix_steps):
            desired_heading = math.atan2(goal_y - y, goal_x - x)
            heading_error = _wrap_angle(desired_heading - theta)
            omega = max(
                -self.omega_max,
                min(self.omega_max, heading_error / self.dt),
            )

            if abs(heading_error) > 0.6:
                v = 0.4 * self.v_max
            elif abs(heading_error) > 0.25:
                v = 0.7 * self.v_max
            else:
                v = self.v_max

            control = (float(v), float(omega))
            controls.append(control)
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            theta = _wrap_angle(theta + omega * self.dt)

        controls.extend(
            [self.nominal_tail_control] * (length - prefix_steps)
        )
        return controls

    def build(self, state: Sequence[float], horizon: int) -> List[Control]:
        """Compatibility alias for callers that use a generic builder name."""

        return self.mean_control_sequence(state, horizon)


__all__ = [
    "Control",
    "GoalWarmStartPrior",
    "PreviousSequencePrior",
    "SamplingPrior",
]
