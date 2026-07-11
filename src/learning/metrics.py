"""Metrics and lightweight performance helpers for residual dynamics models.

The numerical helpers deliberately support both NumPy arrays and Torch tensors.
Torch inputs stay on their original device and remain differentiable; reporting
helpers convert only their final scalar results to ordinary Python values.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch


Array = Union[np.ndarray, torch.Tensor]


def _angle_indices(values: Sequence[int], state_dim: int) -> Tuple[int, ...]:
    if values is None:
        return ()
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError("angle_indices must contain integers")
        index = int(value)
        if index < 0 or index >= state_dim:
            raise ValueError(
                "angle index {} is outside state dimension {}".format(
                    index, state_dim
                )
            )
        if index in result:
            raise ValueError("duplicate angle index {}".format(index))
        result.append(index)
    return tuple(result)


def wrap_angle(values: Array) -> Array:
    """Wrap radians to the shortest-arc interval ``[-pi, pi]``.

    The conventional ambiguity at odd multiples of pi is resolved by the
    underlying ``atan2`` implementation.  No special-case replacement of
    ``-pi`` with ``pi`` is made, which keeps NumPy and Torch behavior aligned.
    """

    if torch.is_tensor(values):
        if not values.is_floating_point():
            values = values.to(dtype=torch.get_default_dtype())
        return torch.atan2(torch.sin(values), torch.cos(values))
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("angles must be numeric")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    return np.arctan2(np.sin(array), np.cos(array))


wrap_to_pi = wrap_angle


def _matching_values(prediction: Array, target: Array) -> Tuple[Array, Array]:
    if torch.is_tensor(prediction) or torch.is_tensor(target):
        if torch.is_tensor(prediction):
            reference = prediction
        else:
            assert torch.is_tensor(target)
            reference = target
        if not reference.is_floating_point():
            reference = reference.to(dtype=torch.get_default_dtype())
        predicted = torch.as_tensor(
            prediction, dtype=reference.dtype, device=reference.device
        )
        expected = torch.as_tensor(target, dtype=reference.dtype, device=reference.device)
        return predicted, expected
    predicted_np = np.asarray(prediction)
    expected_np = np.asarray(target)
    if not np.issubdtype(predicted_np.dtype, np.number) or not np.issubdtype(
        expected_np.dtype, np.number
    ):
        raise TypeError("prediction and target must be numeric")
    dtype = np.result_type(predicted_np.dtype, expected_np.dtype, np.float32)
    return predicted_np.astype(dtype, copy=False), expected_np.astype(dtype, copy=False)


def wrapped_state_error(
    prediction: Array,
    target: Array,
    angle_indices: Sequence[int] = (2,),
) -> Array:
    """Return ``prediction - target`` with periodic components shortest-wrapped."""

    predicted, expected = _matching_values(prediction, target)
    if predicted.shape != expected.shape:
        raise ValueError(
            "prediction shape {} does not match target {}".format(
                tuple(predicted.shape), tuple(expected.shape)
            )
        )
    if predicted.ndim < 1 or predicted.shape[-1] <= 0:
        raise ValueError("states must have a non-empty final feature dimension")
    indices = _angle_indices(angle_indices, int(predicted.shape[-1]))
    error = predicted - expected
    if not indices:
        return error
    if torch.is_tensor(error):
        # Avoid an in-place update on a view that could be needed by autograd.
        components = []
        angle_set = set(indices)
        for index in range(int(error.shape[-1])):
            component = error[..., index : index + 1]
            components.append(wrap_angle(component) if index in angle_set else component)
        return torch.cat(components, dim=-1)
    result = np.array(error, copy=True)
    result[..., list(indices)] = wrap_angle(result[..., list(indices)])
    return result


state_error = wrapped_state_error


def root_mean_square(
    values: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """Compute ``sqrt(mean(values**2))`` while preserving the array backend."""

    if torch.is_tensor(values):
        if not values.is_floating_point():
            values = values.to(dtype=torch.get_default_dtype())
        if axis is None:
            return torch.sqrt(torch.mean(torch.square(values)))
        return torch.sqrt(torch.mean(torch.square(values), dim=axis, keepdim=keepdims))
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("values must be numeric")
    return np.sqrt(np.mean(np.square(array), axis=axis, keepdims=keepdims))


rmse = root_mean_square


def state_rmse(
    prediction: Array,
    target: Array,
    angle_indices: Sequence[int] = (2,),
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """RMSE of periodic-aware state errors."""

    return root_mean_square(
        wrapped_state_error(prediction, target, angle_indices),
        axis=axis,
        keepdims=keepdims,
    )


def rmse_by_horizon(
    prediction: Array,
    target: Array,
    angle_indices: Sequence[int] = (2,),
    horizon_axis: int = -2,
) -> Array:
    """Return one all-state RMSE value for every rollout horizon.

    Inputs conventionally have shape ``(batch, horizon, state_dim)``.  Any
    dimensions other than the horizon dimension are reduced, including the
    state dimension.  ``(horizon, state_dim)`` inputs are also accepted.
    """

    error = wrapped_state_error(prediction, target, angle_indices)
    ndim = int(error.ndim)
    if ndim < 2:
        raise ValueError("rollout states must include horizon and state dimensions")
    normalized_axis = horizon_axis if horizon_axis >= 0 else ndim + horizon_axis
    if normalized_axis < 0 or normalized_axis >= ndim - 1:
        raise ValueError("horizon_axis must identify a non-state dimension")
    reduction_axes = tuple(index for index in range(ndim) if index != normalized_axis)
    return root_mean_square(error, axis=reduction_axes)


per_horizon_rmse = rmse_by_horizon


def rmse_at_horizons(
    prediction: Array,
    target: Array,
    horizons: Optional[Iterable[int]] = None,
    angle_indices: Sequence[int] = (2,),
    horizon_axis: int = -2,
    includes_initial_state: bool = False,
) -> Dict[int, float]:
    """Report selected 1-based horizon RMSE values as a plain mapping.

    If a trajectory includes its initial state, set ``includes_initial_state``
    so horizon 1 reads index 1 instead of index 0.
    """

    values = rmse_by_horizon(
        prediction,
        target,
        angle_indices=angle_indices,
        horizon_axis=horizon_axis,
    )
    count = int(values.shape[0])
    available = count - 1 if includes_initial_state else count
    if available <= 0:
        raise ValueError("rollout does not contain any predicted horizons")
    selected = range(1, available + 1) if horizons is None else horizons
    result: Dict[int, float] = {}
    for value in selected:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("horizons must contain positive integers")
        horizon = int(value)
        if horizon <= 0 or horizon > available:
            raise ValueError(
                "horizon {} is outside available range 1..{}".format(
                    horizon, available
                )
            )
        index = horizon if includes_initial_state else horizon - 1
        item = values[index]
        result[horizon] = float(item.detach().cpu().item() if torch.is_tensor(item) else item)
    return result


horizon_rmse = rmse_at_horizons
rollout_rmse_at_horizons = rmse_at_horizons


def parameter_count(model: Any, trainable_only: bool = False) -> int:
    """Count model parameters, optionally excluding frozen parameters."""

    if not hasattr(model, "parameters"):
        raise TypeError("model must provide a parameters() iterator")
    total = 0
    for parameter in model.parameters():
        if not trainable_only or bool(parameter.requires_grad):
            total += int(parameter.numel())
    return total


count_parameters = parameter_count


def _synchronize(device: Optional[torch.device]) -> None:
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _infer_device(callable_or_model: Any, args: Tuple[Any, ...]) -> Optional[torch.device]:
    for value in args:
        if torch.is_tensor(value):
            return value.device
    if hasattr(callable_or_model, "parameters"):
        try:
            return next(callable_or_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    return None


def benchmark_inference(
    callable_or_model: Any,
    *args: Any,
    warmup: int = 10,
    iterations: int = 100,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any
) -> Dict[str, float]:
    """Benchmark a callable and return latency statistics in milliseconds.

    A sole tuple positional argument is unpacked for convenience.  CUDA work is
    synchronized around every measurement; CPU and NumPy callables incur no
    synchronization.  A Torch module's train/eval mode is restored afterward.
    """

    for value, name, allow_zero in (
        (warmup, "warmup", True),
        (iterations, "iterations", False),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("{} must be an integer".format(name))
        if int(value) < (0 if allow_zero else 1):
            raise ValueError("{} is out of range".format(name))
    warmup = int(warmup)
    iterations = int(iterations)
    call_args = tuple(args[0]) if len(args) == 1 and isinstance(args[0], tuple) else args
    measured_device = (
        torch.device(device)
        if device is not None
        else _infer_device(callable_or_model, call_args)
    )
    was_training = getattr(callable_or_model, "training", None)
    if hasattr(callable_or_model, "eval"):
        callable_or_model.eval()

    samples = []
    try:
        context = torch.inference_mode()
        with context:
            for _ in range(warmup):
                callable_or_model(*call_args, **kwargs)
            _synchronize(measured_device)
            for _ in range(iterations):
                _synchronize(measured_device)
                start = time.perf_counter()
                callable_or_model(*call_args, **kwargs)
                _synchronize(measured_device)
                samples.append((time.perf_counter() - start) * 1000.0)
    finally:
        if was_training is not None and hasattr(callable_or_model, "train"):
            callable_or_model.train(bool(was_training))

    array = np.asarray(samples, dtype=np.float64)
    mean_ms = float(np.mean(array))
    return {
        "mean_ms": mean_ms,
        "median_ms": float(np.median(array)),
        "std_ms": float(np.std(array)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "iterations": float(iterations),
        "calls_per_second": math.inf if mean_ms == 0.0 else 1000.0 / mean_ms,
    }


measure_inference_time = benchmark_inference


def inference_time_ms(
    callable_or_model: Any,
    *args: Any,
    warmup: int = 10,
    iterations: int = 100,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any
) -> float:
    """Return mean inference latency in milliseconds."""

    return benchmark_inference(
        callable_or_model,
        *args,
        warmup=warmup,
        iterations=iterations,
        device=device,
        **kwargs
    )["mean_ms"]


__all__ = [
    "benchmark_inference",
    "count_parameters",
    "horizon_rmse",
    "inference_time_ms",
    "measure_inference_time",
    "parameter_count",
    "per_horizon_rmse",
    "rmse",
    "rmse_at_horizons",
    "rmse_by_horizon",
    "rollout_rmse_at_horizons",
    "root_mean_square",
    "state_error",
    "state_rmse",
    "wrap_angle",
    "wrap_to_pi",
    "wrapped_state_error",
]
