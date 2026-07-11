"""Stage 1 tutorial: make nominal, true, and residual dynamics visible.

This script intentionally stays independent from MuJoCo, ROS, MPPI, memory,
PyTorch, and RL.  The "true plant" below is a controlled teaching model, not
a complete representation of a real vehicle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


State = Tuple[float, float, float]
Control = Tuple[float, float]
StateDerivative = Tuple[float, float, float]
DynamicsFn = Callable[[State, Control], StateDerivative]

DEFAULT_DT = 0.10
DEFAULT_DURATION = 10.0
RANDOM_SEED = 7
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "figures"
    / "icode_tutorial"
    / "icode_step1_model_mismatch.png"
)


def wrap_angle(theta: float) -> float:
    """Normalize an angle to [-pi, pi] using a numerically stable identity."""

    theta = float(theta)
    if not math.isfinite(theta):
        raise ValueError("theta must be finite")
    return math.atan2(math.sin(theta), math.cos(theta))


@dataclass(frozen=True)
class TruePlantParameters:
    """Parameters used to create a simple, controllable model mismatch.

    ``velocity_gain`` and ``yaw_gain`` model actuator effectiveness, while
    ``yaw_bias`` models a constant angular-rate bias in radians per second.
    These values define a teaching plant only; they are not a full vehicle
    dynamics model or parameters identified from a physical robot.
    """

    velocity_gain: float = 0.85
    yaw_gain: float = 1.10
    yaw_bias: float = 0.04

    def __post_init__(self) -> None:
        """Reject non-finite parameters before they enter a rollout."""

        values = (self.velocity_gain, self.yaw_gain, self.yaw_bias)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("true plant parameters must be finite")


def _validate_vector(
    values: Sequence[float], expected_size: int, name: str
) -> Tuple[float, ...]:
    """Convert a state-like vector to finite floats and check its dimension."""

    try:
        converted = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a sequence of real numbers") from exc

    if len(converted) != expected_size:
        raise ValueError(f"{name} must contain exactly {expected_size} values")
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _validate_dt(dt: float) -> float:
    """Return a valid integration time step or raise a clear error."""

    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and greater than zero")
    return dt


def f_nom(state: State, control: Control) -> StateDerivative:
    """Return the nominal unicycle state derivative.

    Args:
        state: Robot pose ``(x, y, theta)`` in meters, meters, and radians.
        control: Command ``(v, omega)`` in meters/second and radians/second.

    Returns:
        ``(x_dot, y_dot, theta_dot)`` according to the ideal unicycle model.
    """

    _, _, theta = _validate_vector(state, 3, "state")
    v, omega = _validate_vector(control, 2, "control")
    return v * math.cos(theta), v * math.sin(theta), omega


def f_true(
    state: State,
    control: Control,
    parameters: TruePlantParameters,
) -> StateDerivative:
    """Return the derivative of the deliberately mismatched teaching plant.

    The commanded speed and yaw rate are transformed as
    ``v_real = velocity_gain * v`` and
    ``omega_real = yaw_gain * omega + yaw_bias``.
    """

    _, _, theta = _validate_vector(state, 3, "state")
    v, omega = _validate_vector(control, 2, "control")

    v_real = parameters.velocity_gain * v
    omega_real = parameters.yaw_gain * omega + parameters.yaw_bias
    return (
        v_real * math.cos(theta),
        v_real * math.sin(theta),
        omega_real,
    )


def compute_residual_derivative(
    state: State,
    control: Control,
    parameters: TruePlantParameters,
) -> StateDerivative:
    """Compute the oracle residual derivative ``f_true - f_nom``.

    This residual is known because the teaching plant is explicitly defined.
    A later learned residual model will instead approximate it from data.
    """

    nominal_derivative = f_nom(state, control)
    true_derivative = f_true(state, control, parameters)
    return (
        true_derivative[0] - nominal_derivative[0],
        true_derivative[1] - nominal_derivative[1],
        true_derivative[2] - nominal_derivative[2],
    )


# Paper-friendly notation: f_res(x, u) = f_true(x, u) - f_nom(x, u).
f_res = compute_residual_derivative


def euler_step(
    state: State,
    control: Control,
    dt: float,
    dynamics_fn: DynamicsFn,
) -> State:
    """Advance one step with ``x_next = x + x_dot * dt``.

    Passing the dynamics function explicitly keeps the integrator reusable for
    nominal, true, oracle-residual-augmented, and future learned dynamics.
    """

    current_state = _validate_vector(state, 3, "state")
    current_control = _validate_vector(control, 2, "control")
    dt = _validate_dt(dt)
    if not callable(dynamics_fn):
        raise TypeError("dynamics_fn must be callable")

    derivative = _validate_vector(
        dynamics_fn(current_state, current_control),
        3,
        "state derivative",
    )
    x, y, theta = current_state
    x_dot, y_dot, theta_dot = derivative
    return (
        x + x_dot * dt,
        y + y_dot * dt,
        wrap_angle(theta + theta_dot * dt),
    )


def generate_control_sequence(
    steps: int,
    dt: float,
    seed: int = RANDOM_SEED,
) -> List[Control]:
    """Create a reproducible, gently time-varying excitation sequence."""

    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    dt = _validate_dt(dt)

    rng = np.random.default_rng(seed)
    controls: List[Control] = []
    for step_index in range(steps):
        time_s = step_index * dt
        v = (
            0.38
            + 0.05 * math.sin(0.55 * time_s)
            + float(rng.normal(0.0, 0.004))
        )
        omega = (
            0.16 * math.sin(0.65 * time_s)
            + 0.12 * math.cos(0.23 * time_s)
            + float(rng.normal(0.0, 0.004))
        )
        controls.append((max(0.05, v), omega))
    return controls


def rollout(
    initial_state: State,
    control_sequence: Sequence[Control],
    dt: float,
    dynamics_fn: DynamicsFn,
) -> np.ndarray:
    """Roll a dynamics model forward and retain every state, including t=0."""

    controls = list(control_sequence)
    if not controls:
        raise ValueError("rollout requires at least one control step")
    dt = _validate_dt(dt)

    current_state = _validate_vector(initial_state, 3, "initial_state")
    trajectory: List[State] = [current_state]
    for control in controls:
        current_state = euler_step(
            current_state,
            control,
            dt,
            dynamics_fn,
        )
        trajectory.append(current_state)
    return np.asarray(trajectory, dtype=float)


def calculate_trajectory_errors(
    nominal_trajectory: np.ndarray,
    true_trajectory: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return position error in meters and absolute heading error in radians."""

    nominal = np.asarray(nominal_trajectory, dtype=float)
    actual = np.asarray(true_trajectory, dtype=float)
    if nominal.shape != actual.shape:
        raise ValueError("nominal and true trajectories must have equal shapes")
    if nominal.ndim != 2 or nominal.shape[0] == 0 or nominal.shape[1] != 3:
        raise ValueError("each trajectory must have shape (number_of_states, 3)")
    if not np.isfinite(nominal).all() or not np.isfinite(actual).all():
        raise ValueError("trajectories must contain only finite values")

    xy_difference = actual[:, :2] - nominal[:, :2]
    position_error = np.linalg.norm(xy_difference, axis=1)
    heading_error = np.asarray(
        [
            abs(wrap_angle(float(true_theta - nominal_theta)))
            for true_theta, nominal_theta in zip(actual[:, 2], nominal[:, 2])
        ],
        dtype=float,
    )
    return position_error, heading_error


def plot_results(
    nominal_trajectory: np.ndarray,
    true_trajectory: np.ndarray,
    times: np.ndarray,
    position_error: np.ndarray,
    heading_error: np.ndarray,
    output_path: Path,
) -> None:
    """Plot trajectory divergence and error growth, then save one PNG file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, (trajectory_axis, error_axis) = plt.subplots(
        1, 2, figsize=(12.0, 5.2)
    )

    trajectory_axis.plot(
        nominal_trajectory[:, 0],
        nominal_trajectory[:, 1],
        color="tab:blue",
        linewidth=2.0,
        label="Nominal trajectory",
    )
    trajectory_axis.plot(
        true_trajectory[:, 0],
        true_trajectory[:, 1],
        color="tab:orange",
        linewidth=2.0,
        label="True trajectory",
    )
    trajectory_axis.scatter(
        nominal_trajectory[0, 0],
        nominal_trajectory[0, 1],
        color="black",
        marker="o",
        s=55,
        label="Start",
        zorder=4,
    )
    trajectory_axis.scatter(
        nominal_trajectory[-1, 0],
        nominal_trajectory[-1, 1],
        color="tab:blue",
        marker="X",
        s=75,
        label="Nominal end",
        zorder=4,
    )
    trajectory_axis.scatter(
        true_trajectory[-1, 0],
        true_trajectory[-1, 1],
        color="tab:orange",
        marker="X",
        s=75,
        label="True end",
        zorder=4,
    )
    trajectory_axis.set_title("Nominal trajectory vs true trajectory")
    trajectory_axis.set_xlabel("X position [m]")
    trajectory_axis.set_ylabel("Y position [m]")
    trajectory_axis.axis("equal")
    trajectory_axis.grid(True, alpha=0.3)
    trajectory_axis.legend(loc="best")

    position_line = error_axis.plot(
        times,
        position_error,
        color="tab:blue",
        linewidth=2.0,
        label="Position error [m]",
    )[0]
    error_axis.set_title("Model mismatch error over time")
    error_axis.set_xlabel("Time [s]")
    error_axis.set_ylabel("Position error [m]", color="tab:blue")
    error_axis.tick_params(axis="y", labelcolor="tab:blue")
    error_axis.grid(True, alpha=0.3)

    heading_axis = error_axis.twinx()
    heading_line = heading_axis.plot(
        times,
        heading_error,
        color="tab:red",
        linewidth=2.0,
        label="Heading error [rad]",
    )[0]
    heading_axis.set_ylabel("Absolute heading error [rad]", color="tab:red")
    heading_axis.tick_params(axis="y", labelcolor="tab:red")
    error_axis.legend(
        [position_line, heading_line],
        [position_line.get_label(), heading_line.get_label()],
        loc="upper left",
    )

    figure.suptitle("Stage 1: controlled model mismatch in unicycle dynamics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_sanity_checks(
    initial_state: State,
    control_sequence: Sequence[Control],
    dt: float,
    mismatch_parameters: TruePlantParameters,
) -> None:
    """Verify the zero-mismatch identity and a finite nonzero mismatch case."""

    zero_mismatch_parameters = TruePlantParameters(
        velocity_gain=1.0,
        yaw_gain=1.0,
        yaw_bias=0.0,
    )
    sample_state = (0.0, 0.0, 0.30)
    sample_control = (0.40, 0.20)

    zero_residual = np.asarray(
        f_res(sample_state, sample_control, zero_mismatch_parameters)
    )
    assert np.allclose(zero_residual, 0.0, atol=1e-12)

    nominal_trajectory = rollout(
        initial_state,
        control_sequence,
        dt,
        f_nom,
    )
    zero_mismatch_trajectory = rollout(
        initial_state,
        control_sequence,
        dt,
        partial(f_true, parameters=zero_mismatch_parameters),
    )
    assert np.allclose(
        nominal_trajectory,
        zero_mismatch_trajectory,
        atol=1e-12,
    )

    mismatch_trajectory = rollout(
        initial_state,
        control_sequence,
        dt,
        partial(f_true, parameters=mismatch_parameters),
    )
    position_error, heading_error = calculate_trajectory_errors(
        nominal_trajectory,
        mismatch_trajectory,
    )
    arrays_to_check = (
        nominal_trajectory,
        zero_mismatch_trajectory,
        mismatch_trajectory,
        position_error,
        heading_error,
    )
    assert all(np.isfinite(values).all() for values in arrays_to_check)
    assert position_error[-1] > 0.0

    nominal_derivative = np.asarray(f_nom(sample_state, sample_control))
    residual_derivative = np.asarray(
        f_res(sample_state, sample_control, mismatch_parameters)
    )
    true_derivative = np.asarray(
        f_true(sample_state, sample_control, mismatch_parameters)
    )
    assert np.allclose(
        nominal_derivative + residual_derivative,
        true_derivative,
        atol=1e-12,
    )

    print("[PASS] Zero mismatch gives a zero residual derivative.")
    print("[PASS] Zero mismatch gives identical nominal and true trajectories.")
    print("[PASS] Default mismatch gives a finite, nonzero final error.")
    print("[PASS] f_nom + f_res equals f_true for the teaching plant.")


def _format_vector(values: Sequence[float]) -> str:
    """Format a short numeric vector for readable console output."""

    return "(" + ", ".join(f"{float(value): .6f}" for value in values) + ")"


def main() -> None:
    """Run the single-step demo, multi-step rollout, checks, and plotting."""

    parameters = TruePlantParameters()

    single_step_state: State = (0.0, 0.0, 0.30)
    single_step_control: Control = (0.40, 0.20)
    single_step_dt = 0.10
    nominal_derivative = f_nom(single_step_state, single_step_control)
    true_derivative = f_true(
        single_step_state,
        single_step_control,
        parameters,
    )
    residual_derivative = f_res(
        single_step_state,
        single_step_control,
        parameters,
    )
    nominal_next_state = euler_step(
        single_step_state,
        single_step_control,
        single_step_dt,
        f_nom,
    )
    true_next_state = euler_step(
        single_step_state,
        single_step_control,
        single_step_dt,
        partial(f_true, parameters=parameters),
    )
    one_step_position_error = math.hypot(
        true_next_state[0] - nominal_next_state[0],
        true_next_state[1] - nominal_next_state[1],
    )
    one_step_heading_error = abs(
        wrap_angle(true_next_state[2] - nominal_next_state[2])
    )

    print("=== Single-step comparison ===")
    print(f"Current state             : {_format_vector(single_step_state)}")
    print(f"Commanded control         : {_format_vector(single_step_control)}")
    print(f"Nominal derivative f_nom  : {_format_vector(nominal_derivative)}")
    print(f"True derivative f_true    : {_format_vector(true_derivative)}")
    print(f"Residual derivative f_res : {_format_vector(residual_derivative)}")
    print(f"Nominal next state        : {_format_vector(nominal_next_state)}")
    print(f"True next state           : {_format_vector(true_next_state)}")
    print(f"One-step position error   : {one_step_position_error:.6f} m")
    print(
        "One-step heading error    : "
        f"{one_step_heading_error:.6f} rad "
        f"({math.degrees(one_step_heading_error):.3f} deg)"
    )

    steps = int(round(DEFAULT_DURATION / DEFAULT_DT))
    controls = generate_control_sequence(steps, DEFAULT_DT, RANDOM_SEED)
    initial_state: State = single_step_state
    nominal_trajectory = rollout(
        initial_state,
        controls,
        DEFAULT_DT,
        f_nom,
    )
    true_trajectory = rollout(
        initial_state,
        controls,
        DEFAULT_DT,
        partial(f_true, parameters=parameters),
    )
    position_error, heading_error = calculate_trajectory_errors(
        nominal_trajectory,
        true_trajectory,
    )
    times = np.arange(nominal_trajectory.shape[0], dtype=float) * DEFAULT_DT

    print("\n=== Multi-step rollout ===")
    print(
        f"Configuration             : {steps} steps, "
        f"dt={DEFAULT_DT:.2f} s, duration={times[-1]:.1f} s, "
        f"seed={RANDOM_SEED}"
    )
    print(f"Nominal final state       : {_format_vector(nominal_trajectory[-1])}")
    print(f"True final state          : {_format_vector(true_trajectory[-1])}")
    print(f"Final position error      : {position_error[-1]:.6f} m")
    print(
        f"Final heading error       : {heading_error[-1]:.6f} rad "
        f"({math.degrees(float(heading_error[-1])):.3f} deg)"
    )
    print(f"Maximum position error    : {position_error.max():.6f} m")

    print("\n=== Sanity checks ===")
    run_sanity_checks(
        initial_state,
        controls,
        DEFAULT_DT,
        parameters,
    )

    plot_results(
        nominal_trajectory,
        true_trajectory,
        times,
        position_error,
        heading_error,
        OUTPUT_PATH,
    )
    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
        raise RuntimeError(f"plot was not created correctly: {OUTPUT_PATH}")
    print(f"\nFigure saved to           : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
