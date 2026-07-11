"""Self-describing, atomic checkpoints for residual-dynamics training."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import math
import os
import random
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np
import torch

from .normalization import NormalizerBundle


CHECKPOINT_FORMAT = "mobile_robot_residual_dynamics"
CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1

# Canonical keys.  The writer also includes ``*_state_dict`` aliases because
# they make artifacts convenient to consume from small standalone scripts.
REQUIRED_CHECKPOINT_KEYS: Tuple[str, ...] = (
    "format_version",
    "schema_version",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "normalizer_bundle_state",
    "config",
    "epoch",
    "best_validation_metric",
    "git_sha",
    "model_class",
    "model_config",
    "state_dim",
    "control_dim",
)


def _plain_value(value: Any, path: str = "config") -> Any:
    """Convert configuration values to a stable, pickle-free value domain."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and math.isnan(value):
            raise ValueError("{} contains NaN".format(path))
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        converted = float(value)
        if math.isnan(converted):
            raise ValueError("{} contains NaN".format(path))
        return converted
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, np.ndarray):
        return _plain_value(value.tolist(), path)
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("{} keys must be strings".format(path))
            result[key] = _plain_value(item, "{}.{}".format(path, key))
        return result
    if isinstance(value, (list, tuple)):
        return [
            _plain_value(item, "{}[{}]".format(path, index))
            for index, item in enumerate(value)
        ]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain_value(value.to_dict(), path)
    raise TypeError(
        "{} contains unsupported value of type {}".format(path, type(value).__name__)
    )


def get_git_sha(cwd: Optional[Union[str, os.PathLike]] = None) -> str:
    """Return the current commit identifier, or ``"unknown"`` outside Git."""

    working_directory = (
        str(cwd)
        if cwd is not None
        else str(Path(__file__).resolve().parents[2])
    )
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_directory,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = result.stdout.strip()
    return sha if sha else "unknown"


def capture_rng_state() -> Dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and available CUDA RNG state."""

    numpy_state = np.random.get_state()
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": numpy_state[1].astype(np.uint32, copy=False).tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_nested_tuple(item) for item in value)
    return value


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    if not isinstance(state, Mapping):
        raise TypeError("rng_state must be a mapping")
    if "python" in state:
        random.setstate(_nested_tuple(state["python"]))
    if "numpy" in state:
        numpy_state = state["numpy"]
        if not isinstance(numpy_state, Mapping):
            raise TypeError("rng_state.numpy must be a mapping")
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    if "torch_cpu" in state:
        torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).cpu())
    cuda_state = state.get("torch_cuda", [])
    if cuda_state and torch.cuda.is_available():
        available = torch.cuda.device_count()
        if len(cuda_state) != available:
            raise ValueError(
                "checkpoint has RNG state for {} CUDA devices; runtime has {}".format(
                    len(cuda_state), available
                )
            )
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8).cpu() for item in cuda_state]
        )


def model_class_name(model: Any) -> str:
    cls = type(model)
    return "{}.{}".format(cls.__module__, cls.__qualname__)


def model_config(model: Any) -> Dict[str, Any]:
    """Extract constructor metadata from the repository's model conventions."""

    for name in ("config_dict", "get_config", "to_config"):
        method = getattr(model, name, None)
        if callable(method):
            value = method()
            if not isinstance(value, Mapping):
                raise TypeError("model.{}() must return a mapping".format(name))
            return _plain_value(value, "model_config")
    value = getattr(model, "config", None)
    if value is not None:
        converted = _plain_value(value, "model_config")
        if not isinstance(converted, dict):
            raise TypeError("model.config must be a mapping or dataclass")
        return converted
    return {}


def _state_dict(value: Any, name: str) -> Optional[Mapping[str, Any]]:
    if value is None:
        return None
    method = getattr(value, "state_dict", None)
    if not callable(method):
        raise TypeError("{} must provide state_dict()".format(name))
    state = method()
    if not isinstance(state, Mapping):
        raise TypeError("{}.state_dict() must return a mapping".format(name))
    return state


def build_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    normalizers: NormalizerBundle,
    config: Any,
    epoch: int,
    best_validation_metric: float,
    scheduler: Optional[Any] = None,
    git_sha: Optional[str] = None,
    include_rng_state: bool = True,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a validated, self-describing training checkpoint mapping."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(normalizers, NormalizerBundle):
        raise TypeError("normalizers must be a NormalizerBundle")
    if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)):
        raise TypeError("epoch must be a nonnegative integer")
    epoch = int(epoch)
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    metric = float(best_validation_metric)
    if math.isnan(metric):
        raise ValueError("best_validation_metric cannot be NaN")
    state_dim = getattr(model, "state_dim", None)
    control_dim = getattr(model, "control_dim", None)
    for value, name in ((state_dim, "state_dim"), (control_dim, "control_dim")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("model must expose a positive integer {}".format(name))
        if int(value) <= 0:
            raise ValueError("model.{} must be positive".format(name))

    model_state = _state_dict(model, "model")
    optimizer_state = _state_dict(optimizer, "optimizer")
    scheduler_state = _state_dict(scheduler, "scheduler")
    normalizer_state = normalizers.state_dict()
    payload: Dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "normalizer_bundle_state": normalizer_state,
        "config": _plain_value(config),
        "epoch": epoch,
        "best_validation_metric": metric,
        "git_sha": get_git_sha() if git_sha is None else str(git_sha),
        "model_class": model_class_name(model),
        "model_config": model_config(model),
        "state_dim": int(state_dim),
        "control_dim": int(control_dim),
        "rng_state": capture_rng_state() if include_rng_state else None,
        # Familiar aliases for ad-hoc consumers.  Torch serialization memoizes
        # these references, so they do not duplicate tensor storage.
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "normalizer_state_dict": normalizer_state,
    }
    if extra is not None:
        if not isinstance(extra, Mapping):
            raise TypeError("extra must be a mapping")
        payload["extra"] = _plain_value(extra, "extra")
    validate_checkpoint(payload)
    return payload


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    model: Optional[torch.nn.Module] = None,
    strict_version: bool = True,
) -> Mapping[str, Any]:
    """Validate checkpoint structure, versions, metadata, and optional model."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        raise ValueError("checkpoint is missing keys: {}".format(", ".join(missing)))
    if strict_version:
        if checkpoint.get("format", CHECKPOINT_FORMAT) != CHECKPOINT_FORMAT:
            raise ValueError("unsupported checkpoint format {!r}".format(checkpoint["format"]))
        if int(checkpoint["format_version"]) != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "unsupported checkpoint format version {}".format(
                    checkpoint["format_version"]
                )
            )
        if int(checkpoint["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported checkpoint schema version {}".format(
                    checkpoint["schema_version"]
                )
            )
    if not isinstance(checkpoint["model_state"], Mapping):
        raise TypeError("model_state must be a mapping")
    for key in ("optimizer_state", "scheduler_state"):
        if checkpoint[key] is not None and not isinstance(checkpoint[key], Mapping):
            raise TypeError("{} must be a mapping or None".format(key))
    if not isinstance(checkpoint["normalizer_bundle_state"], Mapping):
        raise TypeError("normalizer_bundle_state must be a mapping")
    # This also validates normalization dimensions and fitted statistics.
    normalizers = NormalizerBundle.from_state_dict(checkpoint["normalizer_bundle_state"])
    for key in ("state_dim", "control_dim"):
        value = checkpoint[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("{} must be a positive integer".format(key))
        if int(value) <= 0:
            raise ValueError("{} must be positive".format(key))
    if normalizers.control.feature_dim != int(checkpoint["control_dim"]):
        raise ValueError("normalizer control dimension does not match checkpoint")
    if normalizers.residual.feature_dim != int(checkpoint["state_dim"]):
        raise ValueError("normalizer residual dimension does not match checkpoint")
    if isinstance(checkpoint["epoch"], bool) or int(checkpoint["epoch"]) < 0:
        raise ValueError("epoch must be a nonnegative integer")
    if math.isnan(float(checkpoint["best_validation_metric"])):
        raise ValueError("best_validation_metric cannot be NaN")
    if not isinstance(checkpoint["config"], Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(checkpoint["model_config"], Mapping):
        raise TypeError("model_config must be a mapping")
    if not isinstance(checkpoint["model_class"], str) or not checkpoint["model_class"]:
        raise ValueError("model_class must be a non-empty string")
    if model is not None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        for key in ("state_dim", "control_dim"):
            actual = getattr(model, key, None)
            if actual is None or int(actual) != int(checkpoint[key]):
                raise ValueError("model {} does not match checkpoint".format(key))
    return checkpoint


def atomic_torch_save(checkpoint: Mapping[str, Any], path: Union[str, os.PathLike]) -> Path:
    """Durably write a Torch artifact and publish it with one atomic replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        ".{}.tmp-{}".format(destination.name, uuid.uuid4().hex)
    )
    try:
        with temporary.open("wb") as stream:
            torch.save(dict(checkpoint), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(destination))
        # Best effort directory sync on POSIX.  Some Windows/UNC filesystems do
        # not permit opening directories as descriptors.
        try:
            descriptor = os.open(str(destination.parent), os.O_RDONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def save_checkpoint(
    path: Union[str, os.PathLike],
    checkpoint: Optional[Mapping[str, Any]] = None,
    **build_kwargs: Any
) -> Path:
    """Validate and atomically save a payload, or build one from keyword args."""

    # Also accept the common ``save_checkpoint(payload, path)`` ordering.
    if isinstance(path, Mapping) and isinstance(checkpoint, (str, os.PathLike)):
        path, checkpoint = checkpoint, path
    if checkpoint is not None and build_kwargs:
        raise ValueError("provide checkpoint or checkpoint-building arguments, not both")
    payload = build_checkpoint(**build_kwargs) if checkpoint is None else dict(checkpoint)
    validate_checkpoint(payload)
    return atomic_torch_save(payload, path)


atomic_save_checkpoint = save_checkpoint


def _torch_load(
    path: Union[str, os.PathLike], map_location: Any, weights_only: bool
) -> Any:
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = bool(weights_only)
    return torch.load(str(path), **kwargs)


def load_checkpoint(
    path: Union[str, os.PathLike],
    map_location: Any = "cpu",
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    restore_rng: bool = False,
    weights_only: bool = True,
) -> Dict[str, Any]:
    """Safely load and optionally restore a full training state.

    The default ``map_location='cpu'`` prevents checkpoints created on CUDA
    hosts from allocating GPU memory during inspection. ``weights_only=True``
    uses Torch's restricted unpickler on supported Torch versions.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    payload = _torch_load(source, map_location=map_location, weights_only=weights_only)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint file must contain a dictionary")
    validate_checkpoint(payload, model=model)
    if model is not None:
        model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None:
        state = payload["optimizer_state"]
        if state is None:
            raise ValueError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(state)
    if scheduler is not None:
        state = payload["scheduler_state"]
        if state is None:
            raise ValueError("checkpoint does not contain scheduler state")
        scheduler.load_state_dict(state)
    if restore_rng:
        state = payload.get("rng_state")
        if state is None:
            raise ValueError("checkpoint does not contain RNG state")
        restore_rng_state(state)
    return payload


def normalizers_from_checkpoint(
    checkpoint: Mapping[str, Any], state_encoder: Optional[Any] = None
) -> NormalizerBundle:
    validate_checkpoint(checkpoint)
    return NormalizerBundle.from_state_dict(
        checkpoint["normalizer_bundle_state"], state_encoder=state_encoder
    )


def _factory_call(factory: Callable[..., torch.nn.Module], config: Mapping[str, Any]) -> torch.nn.Module:
    signature = inspect.signature(factory)
    try:
        signature.bind(**dict(config))
    except TypeError:
        accepted = {
            name: value
            for name, value in config.items()
            if name in signature.parameters
            and name not in ("self", "cls")
            and signature.parameters[name].kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        try:
            signature.bind(**accepted)
        except TypeError:
            pass
        else:
            return factory(**accepted)
        try:
            signature.bind(dict(config))
        except TypeError as exc:
            raise TypeError(
                "model factory accepts neither model_config keywords nor one config mapping"
            ) from exc
        return factory(dict(config))
    return factory(**dict(config))


def _import_model_class(name: str) -> Callable[..., torch.nn.Module]:
    module_name, separator, attribute = name.rpartition(".")
    if not separator or not module_name:
        raise ValueError("model_class is not a fully-qualified name")
    module = importlib.import_module(module_name)
    value: Any = module
    for component in attribute.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError("checkpoint model_class is not callable")
    return value


def create_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    factory: Optional[Callable[..., torch.nn.Module]] = None,
    registry: Optional[Mapping[str, Callable[..., torch.nn.Module]]] = None,
    strict: bool = True,
    allow_import: bool = False,
) -> torch.nn.Module:
    """Construct and restore a model through an explicit factory or registry.

    Dynamic importing is disabled by default.  Set ``allow_import=True`` only
    for checkpoints whose provenance is trusted.
    """

    validate_checkpoint(checkpoint)
    class_name = str(checkpoint["model_class"])
    selected = factory
    if selected is None and registry is not None:
        selected = registry.get(class_name)
        if selected is None:
            selected = registry.get(class_name.rsplit(".", 1)[-1])
    if selected is None and allow_import:
        selected = _import_model_class(class_name)
    if selected is None:
        raise ValueError(
            "no factory registered for {}; pass factory/registry or allow_import=True".format(
                class_name
            )
        )
    model = _factory_call(selected, checkpoint["model_config"])
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model factory must return torch.nn.Module")
    validate_checkpoint(checkpoint, model=model)
    model.load_state_dict(checkpoint["model_state"], strict=strict)
    return model


model_from_checkpoint = create_model_from_checkpoint


def load_model_checkpoint(
    path: Union[str, os.PathLike],
    factory: Optional[Callable[..., torch.nn.Module]] = None,
    registry: Optional[Mapping[str, Callable[..., torch.nn.Module]]] = None,
    map_location: Any = "cpu",
    strict: bool = True,
    allow_import: bool = False,
    state_encoder: Optional[Any] = None,
) -> Tuple[torch.nn.Module, NormalizerBundle, Dict[str, Any]]:
    """Load an artifact and return ``(model, normalizers, checkpoint)``."""

    checkpoint = load_checkpoint(path, map_location=map_location)
    model = create_model_from_checkpoint(
        checkpoint,
        factory=factory,
        registry=registry,
        strict=strict,
        allow_import=allow_import,
    )
    if isinstance(map_location, (str, torch.device)):
        model.to(torch.device(map_location))
    normalizers = normalizers_from_checkpoint(checkpoint, state_encoder=state_encoder)
    return model, normalizers, checkpoint


load_model_from_checkpoint = load_model_checkpoint


def restore_training_state(
    checkpoint: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    restore_rng: bool = True,
) -> int:
    """Restore in-memory objects and return the next epoch number."""

    validate_checkpoint(checkpoint, model=model)
    model.load_state_dict(checkpoint["model_state"], strict=strict)
    if optimizer is not None:
        if checkpoint["optimizer_state"] is None:
            raise ValueError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        if checkpoint["scheduler_state"] is None:
            raise ValueError("checkpoint does not contain scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if restore_rng:
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            restore_rng_state(rng_state)
    return int(checkpoint["epoch"]) + 1


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "REQUIRED_CHECKPOINT_KEYS",
    "atomic_save_checkpoint",
    "atomic_torch_save",
    "build_checkpoint",
    "capture_rng_state",
    "create_model_from_checkpoint",
    "get_git_sha",
    "load_checkpoint",
    "load_model_checkpoint",
    "load_model_from_checkpoint",
    "model_class_name",
    "model_config",
    "model_from_checkpoint",
    "normalizers_from_checkpoint",
    "restore_rng_state",
    "restore_training_state",
    "save_checkpoint",
    "validate_checkpoint",
]
