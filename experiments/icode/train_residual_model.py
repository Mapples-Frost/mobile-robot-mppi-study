#!/usr/bin/env python3
"""Train an MLP or control-affine ICODE residual-dynamics model.

The configuration selects the architecture; the command-line overrides only
run-local choices such as data, output, device, epoch count, and smoke mode.
Normalization statistics are always fitted from ``train.npz`` and never from
the validation artifact.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.dynamics.residual.icode_residual import ICODEResidual  # noqa: E402
from src.dynamics.residual.mlp_residual import MLPResidual  # noqa: E402
from src.dynamics.state_encoding import StateEncoder  # noqa: E402
from src.learning.checkpointing import get_git_sha  # noqa: E402
from src.learning.normalization import NormalizerBundle  # noqa: E402
from src.learning.residual_dataset import ResidualDataset  # noqa: E402


DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "icode" / "icode_residual.yaml"
SUPPORTED_MODEL_TYPES = ("mlp_residual", "icode_residual")


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("{} must be a mapping".format(name))
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("{} must be a positive integer".format(name))
    result = int(value)
    if result <= 0:
        raise ValueError("{} must be positive".format(name))
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("{} must be a non-negative integer".format(name))
    result = int(value)
    if result < 0:
        raise ValueError("{} must be non-negative".format(name))
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError("{} must be a finite non-negative scalar".format(name))
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("{} must be finite and non-negative".format(name))
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result == 0.0:
        raise ValueError("{} must be positive".format(name))
    return result


def _path_from_root(value: Any, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError("{} must be a filesystem path".format(name))
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp-{}".format(path.name, uuid.uuid4().hex))
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError("cannot serialize {} in run metadata".format(type(value).__name__))


def load_config(path: Path) -> Dict[str, Any]:
    """Load one YAML mapping without applying command-line overrides."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    with source.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, Mapping):
        raise ValueError("training config must contain a YAML mapping")
    return copy.deepcopy(dict(loaded))


def _apply_smoke_overrides(config: Dict[str, Any]) -> None:
    model = _mapping(config.get("model"), "model")
    training = _mapping(config.get("training"), "training")
    model["hidden_sizes"] = [16, 16]
    training["batch_size"] = min(int(training.get("batch_size", 16)), 16)
    training["rollout_batch_size"] = min(
        int(training.get("rollout_batch_size", 8)), 8
    )
    training["train_rollout_horizon"] = 2
    training["validation_rollout_horizon"] = 3
    training["epochs"] = min(max(int(training.get("epochs", 3)), 2), 3)
    config["model"] = model
    config["training"] = training


def resolve_config(
    config: Mapping[str, Any],
    config_path: Path,
    dataset_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
    epochs: Optional[int] = None,
    smoke: bool = False,
    resume: Optional[Path] = None,
) -> Dict[str, Any]:
    """Apply overrides, resolve paths, and validate the public config schema."""

    resolved = copy.deepcopy(_mapping(config, "config"))
    if int(resolved.get("schema_version", -1)) != 1:
        raise ValueError("unsupported training config schema_version")
    run = _mapping(resolved.get("run"), "run")
    paths = _mapping(resolved.get("paths"), "paths")
    dynamics = _mapping(resolved.get("dynamics"), "dynamics")
    model = _mapping(resolved.get("model"), "model")
    training = _mapping(resolved.get("training"), "training")

    if dataset_dir is not None:
        paths["dataset_dir"] = str(dataset_dir)
    if output_dir is not None:
        paths["output_dir"] = str(output_dir)
    if device is not None:
        run["device"] = device
    if epochs is not None:
        training["epochs"] = epochs

    resolved.update(
        {"run": run, "paths": paths, "dynamics": dynamics, "model": model, "training": training}
    )
    if smoke:
        _apply_smoke_overrides(resolved)
        model = _mapping(resolved["model"], "model")
        training = _mapping(resolved["training"], "training")

    seed = _nonnegative_int(run.get("seed"), "run.seed")
    run["seed"] = seed
    run["run_type"] = "smoke" if smoke else "research"
    requested_device = run.get("device", "auto")
    if not isinstance(requested_device, str) or not requested_device.strip():
        raise TypeError("run.device must be a non-empty string")
    run["device"] = requested_device.strip().lower()
    dtype_name = str(run.get("dtype", "float32")).strip().lower()
    if dtype_name not in ("float32", "float64"):
        raise ValueError("run.dtype must be float32 or float64")
    run["dtype"] = dtype_name
    deterministic = run.get("deterministic", True)
    if not isinstance(deterministic, bool):
        raise TypeError("run.deterministic must be a bool")
    run["deterministic"] = deterministic

    state_dim = _positive_int(dynamics.get("state_dim"), "dynamics.state_dim")
    control_dim = _positive_int(dynamics.get("control_dim"), "dynamics.control_dim")
    dynamics["state_dim"] = state_dim
    dynamics["control_dim"] = control_dim
    method = str(dynamics.get("integration_method", "rk4")).strip().lower()
    if method not in ("euler", "rk4"):
        raise ValueError("dynamics.integration_method must be euler or rk4")
    dynamics["integration_method"] = method
    angle_indices = tuple(int(index) for index in dynamics.get("angle_indices", (2,)))
    StateEncoder("raw", state_dim=state_dim, angle_indices=angle_indices)
    dynamics["angle_indices"] = list(angle_indices)

    model_type = str(model.get("type", "")).strip().lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "model.type must be one of {}".format(", ".join(SUPPORTED_MODEL_TYPES))
        )
    model["type"] = model_type
    encoding = _mapping(model.get("state_encoding"), "model.state_encoding")
    encoding_mode = str(encoding.get("mode", "")).strip().lower()
    encoding_angles = tuple(int(index) for index in encoding.get("angle_indices", ()))
    encoder = StateEncoder(
        mode=encoding_mode,
        state_dim=state_dim,
        angle_indices=encoding_angles,
    )
    encoding["mode"] = encoder.mode
    encoding["angle_indices"] = list(encoder.angle_indices)
    encoding["output_dim"] = encoder.output_dim
    model["state_encoding"] = encoding
    hidden_sizes = model.get("hidden_sizes")
    if isinstance(hidden_sizes, (str, bytes)) or not isinstance(hidden_sizes, Sequence):
        raise TypeError("model.hidden_sizes must be a sequence")
    model["hidden_sizes"] = [
        _positive_int(width, "model.hidden_sizes[{}]".format(index))
        for index, width in enumerate(hidden_sizes)
    ]
    if not model["hidden_sizes"]:
        raise ValueError("model.hidden_sizes must not be empty")
    model["activation"] = str(model.get("activation", "softplus")).strip().lower()
    model["final_layer_scale"] = _positive_float(
        model.get("final_layer_scale", 1.0e-3), "model.final_layer_scale"
    )

    training["epochs"] = _positive_int(training.get("epochs"), "training.epochs")
    training["batch_size"] = _positive_int(
        training.get("batch_size"), "training.batch_size"
    )
    training["rollout_batch_size"] = _positive_int(
        training.get("rollout_batch_size", training["batch_size"]),
        "training.rollout_batch_size",
    )
    training["train_rollout_horizon"] = _positive_int(
        training.get("train_rollout_horizon"), "training.train_rollout_horizon"
    )
    training["validation_rollout_horizon"] = _positive_int(
        training.get("validation_rollout_horizon"),
        "training.validation_rollout_horizon",
    )
    training["rollout_stride"] = _positive_int(
        training.get("rollout_stride", 1), "training.rollout_stride"
    )
    for name, default in (
        ("require_time_continuity", True),
        ("require_constant_dt", False),
    ):
        value = training.get(name, default)
        if not isinstance(value, bool):
            raise TypeError("training.{} must be a bool".format(name))
        training[name] = value
    if not isinstance(training.get("shuffle", True), bool):
        raise TypeError("training.shuffle must be a bool")
    training["num_workers"] = int(training.get("num_workers", 0))
    if training["num_workers"] < 0:
        raise ValueError("training.num_workers must be non-negative")

    optimizer = _mapping(training.get("optimizer"), "training.optimizer")
    optimizer_type = str(optimizer.get("type", "")).strip().lower()
    if optimizer_type != "adam":
        raise ValueError("training.optimizer.type must be adam")
    optimizer["type"] = optimizer_type
    optimizer["learning_rate"] = _positive_float(
        optimizer.get("learning_rate"), "training.optimizer.learning_rate"
    )
    optimizer["weight_decay"] = _nonnegative_float(
        optimizer.get("weight_decay", 0.0), "training.optimizer.weight_decay"
    )
    training["optimizer"] = optimizer

    loss = _mapping(training.get("loss"), "training.loss")
    for name in (
        "derivative_weight",
        "one_step_weight",
        "multistep_weight",
        "regularization_weight",
    ):
        loss[name] = _nonnegative_float(loss.get(name), "training.loss.{}".format(name))
    if not any(
        loss[name] > 0.0
        for name in (
            "derivative_weight",
            "one_step_weight",
            "multistep_weight",
            "regularization_weight",
        )
    ):
        raise ValueError("at least one training loss weight must be positive")
    training["loss"] = loss
    training["gradient_clip_norm"] = _positive_float(
        training.get("gradient_clip_norm"), "training.gradient_clip_norm"
    )

    scheduler = _mapping(training.get("scheduler"), "training.scheduler")
    if str(scheduler.get("type", "")).strip().lower() != "reduce_on_plateau":
        raise ValueError("training.scheduler.type must be reduce_on_plateau")
    scheduler["type"] = "reduce_on_plateau"
    scheduler["factor"] = _positive_float(
        scheduler.get("factor"), "training.scheduler.factor"
    )
    if scheduler["factor"] >= 1.0:
        raise ValueError("training.scheduler.factor must be less than one")
    scheduler["patience"] = _nonnegative_int(
        scheduler.get("patience"), "training.scheduler.patience"
    )
    scheduler["threshold"] = _nonnegative_float(
        scheduler.get("threshold", 1.0e-4), "training.scheduler.threshold"
    )
    scheduler["cooldown"] = _nonnegative_int(
        scheduler.get("cooldown", 0), "training.scheduler.cooldown"
    )
    scheduler["min_learning_rate"] = _nonnegative_float(
        scheduler.get("min_learning_rate", 0.0),
        "training.scheduler.min_learning_rate",
    )
    training["scheduler"] = scheduler

    early = _mapping(training.get("early_stopping"), "training.early_stopping")
    early["patience"] = _positive_int(
        early.get("patience"), "training.early_stopping.patience"
    )
    early["min_delta"] = _nonnegative_float(
        early.get("min_delta", 0.0), "training.early_stopping.min_delta"
    )
    training["early_stopping"] = early

    data_root = _path_from_root(paths.get("dataset_dir"), "paths.dataset_dir")
    result_root = _path_from_root(paths.get("output_dir"), "paths.output_dir")
    paths["dataset_dir"] = str(data_root)
    paths["output_dir"] = str(result_root)
    for key in ("train_file", "validation_file"):
        filename = paths.get(key)
        if not isinstance(filename, str) or not filename.strip():
            raise TypeError("paths.{} must be a non-empty filename".format(key))
        paths[key] = filename
    paths["config_source"] = str(Path(config_path).expanduser().resolve())
    paths["resume"] = None if resume is None else str(Path(resume).expanduser().resolve())
    resolved["run"] = run
    resolved["paths"] = paths
    resolved["dynamics"] = dynamics
    resolved["model"] = model
    resolved["training"] = training
    return resolved


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("invalid device {!r}".format(requested)) from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("training device must be CPU or CUDA")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_datasets(config: Mapping[str, Any]) -> Tuple[ResidualDataset, ResidualDataset]:
    paths = _mapping(config["paths"], "paths")
    dataset_dir = Path(paths["dataset_dir"])
    train_path = Path(paths["train_file"])
    validation_path = Path(paths["validation_file"])
    if not train_path.is_absolute():
        train_path = dataset_dir / train_path
    if not validation_path.is_absolute():
        validation_path = dataset_dir / validation_path
    train_dataset = ResidualDataset.load(train_path)
    validation_dataset = ResidualDataset.load(validation_path)
    expected_state = int(config["dynamics"]["state_dim"])
    expected_control = int(config["dynamics"]["control_dim"])
    for dataset, name in (
        (train_dataset, "training"),
        (validation_dataset, "validation"),
    ):
        if dataset.state_dim != expected_state or dataset.control_dim != expected_control:
            raise ValueError(
                "{} dataset dimensions ({}, {}) do not match config ({}, {})".format(
                    name,
                    dataset.state_dim,
                    dataset.control_dim,
                    expected_state,
                    expected_control,
                )
            )
    return train_dataset, validation_dataset


def _build_model(config: Mapping[str, Any], encoder: StateEncoder) -> torch.nn.Module:
    dynamics = config["dynamics"]
    model_config = config["model"]
    constructor = {
        "mlp_residual": MLPResidual,
        "icode_residual": ICODEResidual,
    }[model_config["type"]]
    return constructor(
        state_feature_dim=encoder.output_dim,
        state_dim=int(dynamics["state_dim"]),
        control_dim=int(dynamics["control_dim"]),
        hidden_sizes=tuple(model_config["hidden_sizes"]),
        activation=model_config["activation"],
        final_layer_scale=float(model_config["final_layer_scale"]),
    )


def _build_trainer(
    model: torch.nn.Module,
    normalizers: NormalizerBundle,
    encoder: StateEncoder,
    config: Mapping[str, Any],
    device: torch.device,
) -> Any:
    # Trainer imports stay here so ``--help`` and config inspection retain a
    # useful error if the optional training layer is absent.
    try:
        from src.learning.trainer import LossWeights, ResidualTrainer, TrainerConfig
    except ImportError:
        try:
            from src.learning.residual_trainer import (
                LossWeights,
                ResidualTrainer,
                TrainerConfig,
            )
        except ImportError as exc:
            raise RuntimeError("ResidualTrainer is unavailable in src.learning") from exc

    training = config["training"]
    optimizer = training["optimizer"]
    scheduler = training["scheduler"]
    early = training["early_stopping"]
    trainer_config = TrainerConfig(
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        rollout_batch_size=int(training["rollout_batch_size"]),
        learning_rate=float(optimizer["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        rollout_horizon=int(training["train_rollout_horizon"]),
        validation_rollout_horizon=int(training["validation_rollout_horizon"]),
        rollout_stride=int(training["rollout_stride"]),
        require_time_continuity=bool(training["require_time_continuity"]),
        require_constant_dt=bool(training["require_constant_dt"]),
        loss_weights=LossWeights(
            derivative=float(training["loss"]["derivative_weight"]),
            one_step=float(training["loss"]["one_step_weight"]),
            rollout=float(training["loss"]["multistep_weight"]),
            regularization=float(training["loss"]["regularization_weight"]),
        ),
        grad_clip_norm=float(training["gradient_clip_norm"]),
        scheduler=str(scheduler["type"]),
        scheduler_factor=float(scheduler["factor"]),
        scheduler_patience=int(scheduler["patience"]),
        scheduler_threshold=float(scheduler["threshold"]),
        scheduler_cooldown=int(scheduler["cooldown"]),
        scheduler_min_lr=float(scheduler["min_learning_rate"]),
        early_stopping_patience=int(early["patience"]),
        early_stopping_min_delta=float(early["min_delta"]),
        device=str(device),
        dtype=str(config["run"]["dtype"]),
        seed=int(config["run"]["seed"]),
        deterministic=bool(config["run"]["deterministic"]),
        shuffle=bool(training["shuffle"]),
        angle_indices=tuple(config["dynamics"]["angle_indices"]),
        integration_method=str(config["dynamics"]["integration_method"]),
        output_dir=str(config["paths"]["output_dir"]),
        extra={"resolved_config": _json_safe(config)},
    )
    return ResidualTrainer(
        model=model,
        normalizers=normalizers,
        config=trainer_config,
        state_encoder=encoder,
    )


def run_training(resolved_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute one resolved training run and return its JSON-safe manifest."""

    config = copy.deepcopy(_mapping(resolved_config, "resolved_config"))
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config_snapshot.yaml"
    _atomic_write_text(
        snapshot_path,
        yaml.safe_dump(_json_safe(config), sort_keys=False, allow_unicode=True),
    )

    seed = int(config["run"]["seed"])
    _set_seed(seed)
    device = _select_device(config["run"]["device"])
    dtype = torch.float32 if config["run"]["dtype"] == "float32" else torch.float64
    train_dataset, validation_dataset = _load_datasets(config)
    encoding = config["model"]["state_encoding"]
    encoder = StateEncoder(
        mode=encoding["mode"],
        state_dim=int(config["dynamics"]["state_dim"]),
        angle_indices=tuple(encoding["angle_indices"]),
    )
    # This is deliberately the only fit call: validation rows never influence
    # normalization statistics.
    normalizers = NormalizerBundle.fit(
        train_data=train_dataset,
        state_encoder=encoder,
    )
    model = _build_model(config, encoder).to(device=device, dtype=dtype)
    trainer = _build_trainer(model, normalizers, encoder, config, device)
    resume_value = config["paths"].get("resume")
    result = trainer.fit(
        train_dataset,
        validation_dataset,
        resume_from=None if resume_value is None else Path(resume_value),
    )

    best_checkpoint = Path(result.best_checkpoint).resolve()
    log_path = Path(result.log_path).resolve()
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "git_sha": get_git_sha(REPOSITORY_ROOT),
        "seed": seed,
        "run_type": config["run"]["run_type"],
        "model_type": config["model"]["type"],
        "model_class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
        "state_dim": int(model.state_dim),
        "state_feature_dim": int(model.state_feature_dim),
        "control_dim": int(model.control_dim),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": str(device),
        "dataset_dir": config["paths"]["dataset_dir"],
        "train_transitions": int(train_dataset.size),
        "validation_transitions": int(validation_dataset.size),
        "normalizer_fit_samples": int(normalizers.fit_sample_count),
        "output_dir": str(output_dir.resolve()),
        "resolved_config": str(snapshot_path.resolve()),
        "resume_from": resume_value,
        "best_validation_metric": float(result.best_metric),
        "best_epoch": int(result.best_epoch),
        "last_epoch": int(result.last_epoch),
        "stopped_early": bool(result.stopped_early),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(Path(result.last_checkpoint).resolve()),
        "log": str(log_path),
    }
    manifest_path = output_dir / "run_manifest.json"
    _atomic_write_text(
        manifest_path,
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="training YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="override the directory containing train.npz and validation.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override the checkpoint and log directory",
    )
    parser.add_argument(
        "--device",
        help="override device (auto, cpu, cuda, or cuda:N)",
    )
    parser.add_argument("--epochs", type=int, help="override the configured epoch count")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use a tiny network, batches, horizons, and 2-3 epochs",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume model, optimizer, scheduler, and epoch state from a checkpoint",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        raw_config = load_config(args.config)
        resolved = resolve_config(
            raw_config,
            config_path=args.config,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            device=args.device,
            epochs=args.epochs,
            smoke=args.smoke,
            resume=args.resume,
        )
        manifest = run_training(resolved)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    print("Training complete: {}".format(manifest["best_checkpoint"]))
    print("Log: {}".format(manifest["log"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
