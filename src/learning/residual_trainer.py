"""Reproducible PyTorch training loop for residual dynamics models.

The trainer deliberately builds batches from :class:`ResidualDataset` itself.
Rollout batches are based only on ``contiguous_rollout_windows``, so neither a
random sampler nor a minibatch boundary can join two episodes.
"""

from __future__ import annotations

import csv
import dataclasses
import inspect
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .checkpointing import (
    build_checkpoint,
    load_checkpoint,
    normalizers_from_checkpoint,
    save_checkpoint,
)
from .normalization import NormalizerBundle
from .residual_dataset import ResidualDataset, contiguous_rollout_windows


@dataclass
class LossWeights:
    """Weights for the supervised, integration, rollout, and regularizer terms."""

    derivative: float = 1.0
    one_step: float = 1.0
    rollout: float = 1.0
    regularization: float = 1.0e-6

    def __post_init__(self) -> None:
        for name in ("derivative", "one_step", "rollout", "regularization"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("loss weight {} must be finite and nonnegative".format(name))
            setattr(self, name, value)
        if self.derivative + self.one_step + self.rollout + self.regularization == 0.0:
            raise ValueError("at least one loss weight must be positive")

    @classmethod
    def from_value(cls, value: Any) -> "LossWeights":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return cls(**value.to_dict())
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = dataclasses.asdict(value)
        if not isinstance(value, Mapping):
            raise TypeError("loss_weights must be a LossWeights or mapping")
        aliases = {
            "residual": "derivative",
            "residual_derivative": "derivative",
            "lambda_residual": "derivative",
            "lambda_derivative": "derivative",
            "lambda_one_step": "one_step",
            "multistep": "rollout",
            "lambda_multistep": "rollout",
            "lambda_rollout": "rollout",
            "lambda_reg": "regularization",
            "lambda_regularization": "regularization",
        }
        kwargs: Dict[str, Any] = {}
        for key, item in value.items():
            normalized = aliases.get(str(key), str(key))
            if normalized in ("derivative", "one_step", "rollout", "regularization"):
                kwargs[normalized] = item
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, float]:
        return {
            "derivative": self.derivative,
            "one_step": self.one_step,
            "rollout": self.rollout,
            "regularization": self.regularization,
        }


@dataclass
class TrainerConfig:
    """Complete configuration for :class:`ResidualTrainer`."""

    epochs: int = 100
    max_epochs: Optional[int] = None
    batch_size: int = 256
    rollout_batch_size: Optional[int] = None
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1.0e-8
    grad_clip_norm: Optional[float] = 1.0
    rollout_horizon: int = 5
    validation_rollout_horizon: Optional[int] = None
    rollout_stride: int = 1
    require_time_continuity: bool = True
    require_constant_dt: bool = False
    loss_weights: LossWeights = field(default_factory=LossWeights)
    scheduler: Optional[str] = "reduce_on_plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8
    scheduler_threshold: float = 1.0e-4
    scheduler_cooldown: int = 0
    scheduler_min_lr: float = 1.0e-6
    early_stopping_patience: Optional[int] = 20
    early_stopping_min_delta: float = 0.0
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 0
    deterministic: bool = True
    deterministic_warn_only: bool = False
    shuffle: bool = True
    angle_indices: Tuple[int, ...] = (2,)
    integration_method: str = "rk4"
    output_dir: str = "results/residual_training"
    best_checkpoint_name: str = "best.pt"
    last_checkpoint_name: str = "last.pt"
    csv_log_name: str = "epochs.csv"
    restore_best_at_end: bool = True
    restore_rng_on_resume: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_epochs is not None:
            self.epochs = self.max_epochs
        for name in ("epochs", "batch_size", "rollout_horizon", "rollout_stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("{} must be a positive integer".format(name))
            if int(value) <= 0:
                raise ValueError("{} must be positive".format(name))
            setattr(self, name, int(value))
        self.max_epochs = self.epochs
        if self.validation_rollout_horizon is None:
            self.validation_rollout_horizon = self.rollout_horizon
        if isinstance(self.validation_rollout_horizon, bool) or not isinstance(
            self.validation_rollout_horizon, (int, np.integer)
        ):
            raise TypeError("validation_rollout_horizon must be a positive integer")
        self.validation_rollout_horizon = int(self.validation_rollout_horizon)
        if self.validation_rollout_horizon <= 0:
            raise ValueError("validation_rollout_horizon must be positive")
        if self.rollout_batch_size is None:
            self.rollout_batch_size = self.batch_size
        if isinstance(self.rollout_batch_size, bool) or not isinstance(
            self.rollout_batch_size, (int, np.integer)
        ):
            raise TypeError("rollout_batch_size must be a positive integer")
        self.rollout_batch_size = int(self.rollout_batch_size)
        if self.rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")
        for name in ("scheduler_patience", "scheduler_cooldown", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("{} must be a nonnegative integer".format(name))
            if int(value) < 0:
                raise ValueError("{} must be nonnegative".format(name))
            setattr(self, name, int(value))
        if self.early_stopping_patience is not None:
            value = self.early_stopping_patience
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("early_stopping_patience must be a positive integer or None")
            self.early_stopping_patience = int(value)
            if self.early_stopping_patience <= 0:
                raise ValueError("early_stopping_patience must be positive or None")
        for name, positive, allow_zero in (
            ("learning_rate", True, False),
            ("weight_decay", False, True),
            ("adam_eps", True, False),
            ("scheduler_factor", True, False),
            ("scheduler_threshold", False, True),
            ("scheduler_min_lr", False, True),
            ("early_stopping_min_delta", False, True),
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (positive and value <= 0.0) or (
                not positive and not allow_zero and value <= 0.0
            ) or (allow_zero and value < 0.0):
                raise ValueError("{} has an invalid value".format(name))
            setattr(self, name, value)
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must lie strictly between zero and one")
        if len(self.adam_betas) != 2:
            raise ValueError("adam_betas must contain two values")
        self.adam_betas = (float(self.adam_betas[0]), float(self.adam_betas[1]))
        if any(not 0.0 <= value < 1.0 for value in self.adam_betas):
            raise ValueError("adam_betas values must lie in [0, 1)")
        self.loss_weights = LossWeights.from_value(self.loss_weights)
        if self.scheduler is not None:
            name = str(self.scheduler).strip().lower()
            if name in ("", "none", "off", "false"):
                self.scheduler = None
            elif name in ("plateau", "reduce_on_plateau", "reducelronplateau"):
                self.scheduler = "reduce_on_plateau"
            else:
                raise ValueError("unsupported scheduler {!r}".format(self.scheduler))
        if not isinstance(self.device, str):
            self.device = str(self.device)
        self.dtype = str(self.dtype).lower().replace("torch.", "")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(self.integration_method, str):
            raise TypeError("integration_method must be a string")
        self.integration_method = self.integration_method.strip().lower()
        if self.integration_method not in ("euler", "rk4"):
            raise ValueError("integration_method must be 'euler' or 'rk4'")
        angles = []
        for value in self.angle_indices:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("angle_indices must contain integers")
            index = int(value)
            if index < 0 or index in angles:
                raise ValueError("angle_indices must be unique and nonnegative")
            angles.append(index)
        self.angle_indices = tuple(angles)
        for name in ("best_checkpoint_name", "last_checkpoint_name", "csv_log_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or Path(value).name != value:
                raise ValueError("{} must be a simple non-empty filename".format(name))

    @classmethod
    def from_value(cls, value: Any) -> "TrainerConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return cls(**value.to_dict())
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = dataclasses.asdict(value)
        if not isinstance(value, Mapping):
            raise TypeError("config must be a TrainerConfig, dataclass, or mapping")
        source: Mapping[str, Any] = value
        for section in ("trainer", "training"):
            candidate = value.get(section)
            if isinstance(candidate, Mapping):
                source = candidate
                break
        merged = dict(source)
        aliases = {
            "lr": "learning_rate",
            "num_epochs": "epochs",
            "gradient_clip_norm": "grad_clip_norm",
            "checkpoint_dir": "output_dir",
            "log_filename": "csv_log_name",
            "horizon": "rollout_horizon",
        }
        for old, new in aliases.items():
            if old in merged and new not in merged:
                merged[new] = merged[old]
        optimizer = merged.get("optimizer")
        if isinstance(optimizer, Mapping):
            for old, new in (
                ("lr", "learning_rate"),
                ("learning_rate", "learning_rate"),
                ("weight_decay", "weight_decay"),
                ("betas", "adam_betas"),
                ("eps", "adam_eps"),
            ):
                if old in optimizer and new not in merged:
                    merged[new] = optimizer[old]
        scheduler = merged.get("scheduler")
        if isinstance(scheduler, Mapping):
            merged["scheduler"] = scheduler.get("name", scheduler.get("type", "reduce_on_plateau"))
            for key in ("factor", "patience", "threshold", "cooldown", "min_lr"):
                target = "scheduler_{}".format(key)
                if key in scheduler and target not in merged:
                    merged[target] = scheduler[key]
        stopping = merged.get("early_stopping")
        if isinstance(stopping, Mapping):
            if "patience" in stopping and "early_stopping_patience" not in merged:
                merged["early_stopping_patience"] = stopping["patience"]
            if "min_delta" in stopping and "early_stopping_min_delta" not in merged:
                merged["early_stopping_min_delta"] = stopping["min_delta"]
        fields = {item.name for item in dataclasses.fields(cls)}
        kwargs = {key: item for key, item in merged.items() if key in fields and key != "extra"}
        kwargs["extra"] = {key: item for key, item in merged.items() if key not in fields}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        result["loss_weights"] = self.loss_weights.to_dict()
        result["angle_indices"] = list(self.angle_indices)
        result["adam_betas"] = list(self.adam_betas)
        return result


@dataclass
class TrainingResult:
    history: List[Dict[str, Any]]
    best_metric: float
    best_epoch: Optional[int]
    last_epoch: int
    stopped_early: bool
    best_checkpoint: Path
    last_checkpoint: Path
    log_path: Path

    @property
    def best_validation_metric(self) -> float:
        return self.best_metric

    def as_dict(self) -> Dict[str, Any]:
        return {
            "history": self.history,
            "best_metric": self.best_metric,
            "best_validation_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "last_epoch": self.last_epoch,
            "stopped_early": self.stopped_early,
            "best_checkpoint": str(self.best_checkpoint),
            "last_checkpoint": str(self.last_checkpoint),
            "log_path": str(self.log_path),
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def set_deterministic_seed(seed: int, deterministic: bool = True, warn_only: bool = False) -> None:
    """Seed Python, NumPy, and Torch and configure deterministic kernels."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except TypeError:  # Torch versions before the warn_only argument.
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


seed_everything = set_deterministic_seed


def resolve_device(value: Union[str, torch.device]) -> torch.device:
    if isinstance(value, torch.device):
        device = value
    else:
        name = str(value).strip().lower()
        device = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("training device must be CPU or CUDA")
    return device


def deterministic_minibatches(
    size: int,
    batch_size: int,
    seed: int,
    epoch: int = 0,
    stream: int = 0,
    shuffle: bool = True,
) -> List[np.ndarray]:
    """Return an epoch-specific deterministic partition of row indices."""

    for value, name, positive in (
        (size, "size", False),
        (batch_size, "batch_size", True),
        (seed, "seed", False),
        (epoch, "epoch", False),
        (stream, "stream", False),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("{} must be an integer".format(name))
        if int(value) < (1 if positive else 0):
            raise ValueError("{} is out of range".format(name))
    order = np.arange(int(size), dtype=np.int64)
    if shuffle and order.size > 1:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(stream)]))
        order = rng.permutation(order)
    return [
        np.ascontiguousarray(order[start : start + int(batch_size)])
        for start in range(0, int(size), int(batch_size))
    ]


def _tensor(values: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)


def build_transition_batch(
    dataset: ResidualDataset,
    indices: Sequence[int],
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Gather physical transition fields for the composite residual loss."""

    if not isinstance(dataset, ResidualDataset):
        raise TypeError("dataset must be a ResidualDataset")
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    if np.any(selected < 0) or np.any(selected >= len(dataset)):
        raise IndexError("transition batch index is out of range")
    target_device = resolve_device(device)
    batch = {
        "state_t": _tensor(dataset.state_t[selected], target_device, dtype),
        "control_t": _tensor(dataset.control_t[selected], target_device, dtype),
        "applied_control_t": _tensor(dataset.applied_control_t[selected], target_device, dtype),
        "state_t_plus_1": _tensor(dataset.state_t_plus_1[selected], target_device, dtype),
        "nominal_derivative": _tensor(dataset.nominal_derivative[selected], target_device, dtype),
        "observed_derivative": _tensor(dataset.observed_derivative[selected], target_device, dtype),
        "residual_target": _tensor(dataset.residual_target[selected], target_device, dtype),
        "dt": _tensor(dataset.dt[selected], target_device, dtype),
        "time": _tensor(dataset.time[selected], target_device, dtype),
        "indices": torch.as_tensor(selected, dtype=torch.long, device=target_device),
    }
    # Compact aliases make the helper useful with simple custom losses too.
    batch["state"] = batch["state_t"]
    batch["control"] = batch["control_t"]
    batch["next_state"] = batch["state_t_plus_1"]
    batch["target_residual"] = batch["residual_target"]
    return batch


def build_rollout_batch(
    dataset: ResidualDataset,
    windows: np.ndarray,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Gather episode-safe rollout sequences from explicit row-index windows."""

    if not isinstance(dataset, ResidualDataset):
        raise TypeError("dataset must be a ResidualDataset")
    selected = np.asarray(windows, dtype=np.int64)
    if selected.ndim != 2 or selected.shape[0] == 0 or selected.shape[1] == 0:
        raise ValueError("windows must have shape (batch, horizon) with nonzero dimensions")
    if np.any(selected < 0) or np.any(selected >= len(dataset)):
        raise IndexError("rollout window index is out of range")
    # Defensively prove continuity even if callers did not obtain these rows
    # from contiguous_rollout_windows.
    for window in selected:
        episodes = dataset.episode_id[window]
        if np.any(episodes != episodes[0]) or np.any(np.diff(dataset.step[window]) != 1):
            raise ValueError("rollout windows may not cross episodes or step gaps")
    target_device = resolve_device(device)
    return {
        "rollout_initial_state": _tensor(dataset.state_t[selected[:, 0]], target_device, dtype),
        "rollout_controls": _tensor(dataset.control_t[selected], target_device, dtype),
        "rollout_dt": _tensor(dataset.dt[selected], target_device, dtype),
        "rollout_target_states": _tensor(dataset.state_t_plus_1[selected], target_device, dtype),
        "rollout_indices": torch.as_tensor(selected, dtype=torch.long, device=target_device),
    }


class ResidualTrainer:
    """Adam trainer with rollout-based selection, resume, and atomic artifacts."""

    _CSV_FIELDS = (
        "epoch",
        "train_total_loss",
        "train_derivative_loss",
        "train_one_step_loss",
        "train_rollout_loss",
        "train_regularization_loss",
        "train_rollout_rmse",
        "validation_total_loss",
        "validation_derivative_loss",
        "validation_one_step_loss",
        "validation_rollout_loss",
        "validation_regularization_loss",
        "validation_rollout_rmse",
        "selection_metric",
        "learning_rate",
        "gradient_norm",
        "improved",
    )

    def __init__(
        self,
        model: torch.nn.Module,
        normalizers: Optional[NormalizerBundle] = None,
        config: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
        state_encoder: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if normalizers is not None and not isinstance(normalizers, NormalizerBundle):
            raise TypeError("normalizers must be a NormalizerBundle or None")
        self.config = TrainerConfig.from_value(config)
        self.requested_config = config
        set_deterministic_seed(
            self.config.seed,
            deterministic=self.config.deterministic,
            warn_only=self.config.deterministic_warn_only,
        )
        self.device = resolve_device(self.config.device)
        self.dtype = torch.float32 if self.config.dtype == "float32" else torch.float64
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.normalizers = normalizers
        self.state_encoder = state_encoder
        self.loss_fn = loss_fn
        self._owns_loss = loss_fn is None
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("model has no trainable parameters")
        self.optimizer = optimizer or torch.optim.Adam(
            parameters,
            lr=self.config.learning_rate,
            betas=self.config.adam_betas,
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
        )
        if scheduler is not None:
            self.scheduler = scheduler
        elif self.config.scheduler == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                threshold=self.config.scheduler_threshold,
                cooldown=self.config.scheduler_cooldown,
                min_lr=self.config.scheduler_min_lr,
            )
        else:
            self.scheduler = None
        self.best_validation_metric = math.inf
        self.best_epoch: Optional[int] = None
        self.epochs_without_improvement = 0
        self.start_epoch = 0
        self.history: List[Dict[str, Any]] = []

    @property
    def output_dir(self) -> Path:
        return Path(self.config.output_dir)

    @property
    def best_checkpoint_path(self) -> Path:
        return self.output_dir / self.config.best_checkpoint_name

    @property
    def last_checkpoint_path(self) -> Path:
        return self.output_dir / self.config.last_checkpoint_name

    @property
    def log_path(self) -> Path:
        return self.output_dir / self.config.csv_log_name

    def _effective_checkpoint_config(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"trainer": self.config.to_dict()}
        if self.requested_config is not None:
            if isinstance(self.requested_config, TrainerConfig):
                result["requested_config"] = self.requested_config.to_dict()
            elif dataclasses.is_dataclass(self.requested_config) and not isinstance(self.requested_config, type):
                result["requested_config"] = dataclasses.asdict(self.requested_config)
            elif isinstance(self.requested_config, Mapping):
                result["requested_config"] = dict(self.requested_config)
        return result

    def _validate_data(self, dataset: ResidualDataset, name: str) -> None:
        if not isinstance(dataset, ResidualDataset):
            raise TypeError("{} must be a ResidualDataset".format(name))
        if len(dataset) == 0:
            raise ValueError("{} must contain at least one transition".format(name))
        for attribute, actual in (("state_dim", dataset.state_dim), ("control_dim", dataset.control_dim)):
            expected = getattr(self.model, attribute, None)
            if expected is None or int(expected) != int(actual):
                raise ValueError("model {} does not match {}".format(attribute, name))
        if any(index >= dataset.state_dim for index in self.config.angle_indices):
            raise ValueError("angle_indices exceed dataset state dimension")

    def _prepare_normalizers(self, train_data: ResidualDataset) -> None:
        if self.normalizers is None:
            self.normalizers = NormalizerBundle.fit(
                train_data=train_data, state_encoder=self.state_encoder
            )
        assert self.normalizers is not None
        if self.normalizers.control.feature_dim != train_data.control_dim:
            raise ValueError("control normalizer dimension does not match dataset")
        if self.normalizers.residual.feature_dim != train_data.state_dim:
            raise ValueError("residual normalizer dimension does not match dataset")
        feature_dim = getattr(self.model, "state_feature_dim", self.normalizers.state.feature_dim)
        if int(feature_dim) != self.normalizers.state.feature_dim:
            raise ValueError("state normalizer feature dimension does not match model")

    def _prepare_loss(self) -> None:
        if self.loss_fn is None:
            if self.normalizers is None:
                raise RuntimeError("normalizers must be prepared before the loss")
            from .residual_losses import CompositeResidualLoss

            weights = self.config.loss_weights
            self.loss_fn = CompositeResidualLoss(
                model=self.model,
                normalizers=self.normalizers,
                lambda_residual=weights.derivative,
                lambda_one_step=weights.one_step,
                lambda_multistep=weights.rollout,
                lambda_reg=weights.regularization,
                method=self.config.integration_method,
                angle_indices=self.config.angle_indices,
            )
        if isinstance(self.loss_fn, torch.nn.Module):
            self.loss_fn.to(device=self.device)

    def _invoke_loss(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.loss_fn is None:
            raise RuntimeError("loss function has not been prepared")
        target = self.loss_fn.forward if isinstance(self.loss_fn, torch.nn.Module) else self.loss_fn
        signature = inspect.signature(target)
        transition = {key: value for key, value in batch.items() if not key.startswith("rollout_")}
        rollout = {key: value for key, value in batch.items() if key.startswith("rollout_")}
        candidates = (
            ((batch,), {}),
            ((self.model, batch), {}),
            ((self.model, transition, rollout if rollout else None), {}),
            ((), {"batch": batch}),
            ((), {"model": self.model, "batch": batch}),
        )
        result: Any = None
        bound = False
        for args, kwargs in candidates:
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                continue
            result = self.loss_fn(*args, **kwargs)
            bound = True
            break
        if not bound:
            raise TypeError("loss_fn does not accept a supported batch signature")
        return self._canonical_loss_result(result)

    def _canonical_loss_result(self, result: Any) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(result):
            raw: Mapping[str, Any] = {"total": result}
        elif isinstance(result, Mapping):
            flattened = dict(result)
            components = flattened.get("components")
            if isinstance(components, Mapping):
                flattened.update(components)
            raw = flattened
        elif isinstance(result, (tuple, list)) and result and torch.is_tensor(result[0]):
            raw = {"total": result[0]}
        else:
            raise TypeError("loss_fn must return a tensor or mapping of tensors")
        aliases = {
            "total": ("total", "total_loss", "loss"),
            "derivative": ("derivative", "derivative_loss", "residual", "residual_loss", "residual_derivative_loss"),
            "one_step": ("one_step", "one_step_loss"),
            "rollout": ("rollout", "rollout_loss", "multistep", "multistep_loss"),
            "regularization": ("regularization", "regularization_loss", "reg_loss"),
            "rollout_rmse": ("rollout_rmse", "multistep_rmse"),
        }
        values: Dict[str, torch.Tensor] = {}
        for canonical, names in aliases.items():
            for name in names:
                value = raw.get(name)
                if torch.is_tensor(value) and value.numel() == 1:
                    values[canonical] = value.reshape(())
                    break
                if isinstance(value, (int, float, np.number)):
                    values[canonical] = torch.as_tensor(value, dtype=self.dtype, device=self.device)
                    break
        if "total" not in values:
            terms = []
            weights = self.config.loss_weights
            for name, weight in (
                ("derivative", weights.derivative),
                ("one_step", weights.one_step),
                ("rollout", weights.rollout),
                ("regularization", weights.regularization),
            ):
                if name in values and weight:
                    terms.append(values[name] * weight)
            if not terms:
                raise ValueError("loss result does not contain a total or recognized component")
            values["total"] = sum(terms[1:], terms[0])
        if not bool(torch.isfinite(values["total"]).item()):
            raise FloatingPointError("non-finite training loss")
        return values

    def _windows(self, dataset: ResidualDataset, horizon: Optional[int] = None) -> np.ndarray:
        selected_horizon = self.config.rollout_horizon if horizon is None else int(horizon)
        return contiguous_rollout_windows(
            dataset,
            horizon=selected_horizon,
            stride=self.config.rollout_stride,
            require_time_continuity=self.config.require_time_continuity,
            require_constant_dt=self.config.require_constant_dt,
        )

    def _run_epoch(self, dataset: ResidualDataset, windows: np.ndarray, epoch: int, training: bool) -> Dict[str, float]:
        transition_batches = deterministic_minibatches(
            len(dataset), self.config.batch_size, self.config.seed, epoch,
            stream=0 if training else 10, shuffle=self.config.shuffle if training else False,
        )
        rollout_batches = deterministic_minibatches(
            int(windows.shape[0]), int(self.config.rollout_batch_size), self.config.seed,
            epoch, stream=1 if training else 11,
            shuffle=self.config.shuffle if training else False,
        )
        steps = max(len(transition_batches), len(rollout_batches), 1)
        sums: Dict[str, float] = {}
        gradient_norm_sum = 0.0
        gradient_steps = 0
        self.model.train(training)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for step in range(steps):
                indices = transition_batches[step % len(transition_batches)]
                batch = build_transition_batch(dataset, indices, self.device, self.dtype)
                if rollout_batches:
                    selected_windows = windows[rollout_batches[step % len(rollout_batches)]]
                    batch.update(build_rollout_batch(dataset, selected_windows, self.device, self.dtype))
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                losses = self._invoke_loss(batch)
                if training:
                    losses["total"].backward()
                    if self.config.grad_clip_norm is not None:
                        norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=float(self.config.grad_clip_norm),
                            error_if_nonfinite=True,
                        )
                        gradient_norm_sum += float(norm.detach().cpu().item())
                    else:
                        squared = 0.0
                        for parameter in self.model.parameters():
                            if parameter.grad is not None:
                                squared += float(torch.sum(parameter.grad.detach() ** 2).cpu().item())
                        gradient_norm_sum += math.sqrt(squared)
                    gradient_steps += 1
                    self.optimizer.step()
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu().item())
        metrics = {key: value / float(steps) for key, value in sums.items()}
        if "rollout_rmse" not in metrics and "rollout" in metrics:
            metrics["rollout_rmse"] = math.sqrt(max(metrics["rollout"], 0.0))
        metrics["gradient_norm"] = gradient_norm_sum / float(max(gradient_steps, 1))
        return metrics

    def resume(self, path: Union[str, os.PathLike], strict: bool = True) -> int:
        """Restore full model/optimizer/scheduler/normalizer/RNG training state."""

        checkpoint = load_checkpoint(
            path,
            map_location=self.device,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            strict=strict,
            restore_rng=self.config.restore_rng_on_resume,
        )
        if self.scheduler is None and checkpoint["scheduler_state"] is not None:
            raise ValueError("checkpoint contains a scheduler but trainer configuration disables it")
        encoder = self.state_encoder
        if encoder is None and self.normalizers is not None:
            encoder = self.normalizers.state_encoder
        self.normalizers = normalizers_from_checkpoint(checkpoint, state_encoder=encoder)
        self.best_validation_metric = float(checkpoint["best_validation_metric"])
        trainer_state = checkpoint.get("extra", {}).get("trainer_state", {})
        if isinstance(trainer_state, Mapping):
            best_epoch = trainer_state.get("best_epoch")
            self.best_epoch = None if best_epoch is None else int(best_epoch)
            self.epochs_without_improvement = int(trainer_state.get("epochs_without_improvement", 0))
        self.start_epoch = int(checkpoint["epoch"]) + 1
        if self._owns_loss:
            self.loss_fn = None
        return self.start_epoch

    def _checkpoint(self, epoch: int, stopped_early: bool = False) -> Dict[str, Any]:
        assert self.normalizers is not None
        return build_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            normalizers=self.normalizers,
            config=self._effective_checkpoint_config(),
            epoch=epoch,
            best_validation_metric=self.best_validation_metric,
            extra={
                "trainer_state": {
                    "best_epoch": self.best_epoch,
                    "epochs_without_improvement": self.epochs_without_improvement,
                    "stopped_early": stopped_early,
                }
            },
        )

    def _write_log_row(self, row: Mapping[str, Any]) -> None:
        exists = self.log_path.is_file() and self.log_path.stat().st_size > 0
        with self.log_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self._CSV_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in self._CSV_FIELDS})
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _row_metrics(prefix: str, metrics: Mapping[str, float]) -> Dict[str, float]:
        return {
            "{}_total_loss".format(prefix): metrics.get("total", math.nan),
            "{}_derivative_loss".format(prefix): metrics.get("derivative", math.nan),
            "{}_one_step_loss".format(prefix): metrics.get("one_step", math.nan),
            "{}_rollout_loss".format(prefix): metrics.get("rollout", math.nan),
            "{}_regularization_loss".format(prefix): metrics.get("regularization", math.nan),
            "{}_rollout_rmse".format(prefix): metrics.get("rollout_rmse", math.nan),
        }

    def fit(
        self,
        train_dataset: ResidualDataset,
        validation_dataset: Optional[ResidualDataset] = None,
        resume_from: Optional[Union[str, os.PathLike]] = None,
    ) -> TrainingResult:
        """Train through ``config.epochs`` and return artifact/selection metadata."""

        self._validate_data(train_dataset, "train_dataset")
        if validation_dataset is not None:
            self._validate_data(validation_dataset, "validation_dataset")
        self._prepare_normalizers(train_dataset)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if resume_from is not None:
            self.resume(resume_from)
        else:
            self.start_epoch = 0
            self.history = []
            # A fresh run owns its configured log path.
            try:
                self.log_path.unlink()
            except FileNotFoundError:
                pass
        self._prepare_loss()
        train_windows = self._windows(train_dataset, self.config.rollout_horizon)
        validation_windows = (
            self._windows(validation_dataset, self.config.validation_rollout_horizon)
            if validation_dataset is not None
            else np.empty(
                (0, int(self.config.validation_rollout_horizon)), dtype=np.int64
            )
        )
        stopped_early = False
        last_epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, self.config.epochs):
            train_metrics = self._run_epoch(train_dataset, train_windows, epoch, training=True)
            if validation_dataset is not None:
                validation_metrics = self._run_epoch(validation_dataset, validation_windows, epoch, training=False)
            else:
                validation_metrics = dict(train_metrics)
            selection = validation_metrics.get("rollout_rmse")
            if selection is None or not math.isfinite(selection):
                selection = validation_metrics.get("total", math.inf)
            if not math.isfinite(selection):
                raise FloatingPointError("validation selection metric is not finite")
            if self.scheduler is not None:
                self.scheduler.step(selection)
            improved = selection < self.best_validation_metric - self.config.early_stopping_min_delta
            if improved:
                self.best_validation_metric = float(selection)
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            row: Dict[str, Any] = {"epoch": epoch}
            row.update(self._row_metrics("train", train_metrics))
            row.update(self._row_metrics("validation", validation_metrics))
            row.update(
                {
                    "selection_metric": float(selection),
                    "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                    "gradient_norm": train_metrics.get("gradient_norm", math.nan),
                    "improved": int(improved),
                }
            )
            self.history.append(row)
            self._write_log_row(row)
            if improved:
                save_checkpoint(self.best_checkpoint_path, self._checkpoint(epoch))
            should_stop = (
                self.config.early_stopping_patience is not None
                and self.epochs_without_improvement >= self.config.early_stopping_patience
            )
            save_checkpoint(self.last_checkpoint_path, self._checkpoint(epoch, stopped_early=should_stop))
            last_epoch = epoch
            if should_stop:
                stopped_early = True
                break
        if self.config.restore_best_at_end and self.best_checkpoint_path.is_file():
            load_checkpoint(
                self.best_checkpoint_path,
                map_location=self.device,
                model=self.model,
                strict=True,
                restore_rng=False,
            )
        return TrainingResult(
            history=list(self.history),
            best_metric=float(self.best_validation_metric),
            best_epoch=self.best_epoch,
            last_epoch=last_epoch,
            stopped_early=stopped_early,
            best_checkpoint=self.best_checkpoint_path,
            last_checkpoint=self.last_checkpoint_path,
            log_path=self.log_path,
        )


__all__ = [
    "LossWeights",
    "ResidualTrainer",
    "TrainerConfig",
    "TrainingResult",
    "build_rollout_batch",
    "build_transition_batch",
    "deterministic_minibatches",
    "resolve_device",
    "seed_everything",
    "set_deterministic_seed",
]
