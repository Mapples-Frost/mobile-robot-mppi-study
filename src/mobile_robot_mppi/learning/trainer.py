import copy
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from mobile_robot_mppi.core.config import git_sha
from .dataset_quality import assert_residual_dataset_quality
from .models import ResidualNetwork, encode_state, load_platform_checkpoint


def nominal_derivative(state, control, config):
    state_dim = state.shape[-1]
    theta = state[..., 2]
    if state_dim == 3:
        v_value = control[..., 0]
        omega = control[..., 1]
        return torch.stack((v_value * torch.cos(theta), v_value * torch.sin(theta), omega), dim=-1)
    if state_dim == 5:
        v_value = state[..., 3]
        omega = state[..., 4]
        tau_v = float(config.get("velocity_time_constant", 0.18))
        tau_w = float(config.get("yaw_time_constant", 0.12))
        return torch.stack((
            v_value * torch.cos(theta), v_value * torch.sin(theta), omega,
            (control[..., 0] - v_value) / tau_v,
            (control[..., 1] - omega) / tau_w,
        ), dim=-1)
    raise ValueError("training nominal dynamics supports state_dim 3 or 5")


def integrate(model, state, control, dt, dynamics_config, method="rk4"):
    def derivative(value):
        return nominal_derivative(value, control, dynamics_config) + model(value, control)
    if method == "euler":
        result = state + dt * derivative(state)
    elif method == "rk4":
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt * k1)
        k3 = derivative(state + 0.5 * dt * k2)
        k4 = derivative(state + dt * k3)
        result = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    else:
        raise ValueError("unknown integrator")
    wrapped = torch.atan2(torch.sin(result[..., 2:3]), torch.cos(result[..., 2:3]))
    return torch.cat((result[..., :2], wrapped, result[..., 3:]), dim=-1)


def _weighted_mean(values, sample_weights=None):
    if sample_weights is None:
        return torch.mean(values)
    weights = torch.as_tensor(
        sample_weights, dtype=values.dtype, device=values.device
    )
    if weights.shape != values.shape:
        try:
            weights = torch.broadcast_to(weights, values.shape)
        except RuntimeError as exc:
            raise ValueError("sample_weights must match the batch shape") from exc
    if not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
        raise ValueError("sample_weights must be finite and positive")
    return torch.sum(values * weights) / torch.sum(weights)


def state_mse(predicted, target, component_weights=None, sample_weights=None):
    error = predicted - target
    wrapped = torch.atan2(torch.sin(error[..., 2:3]), torch.cos(error[..., 2:3]))
    periodic_error = torch.cat((error[..., :2], wrapped, error[..., 3:]), dim=-1)
    squared = periodic_error ** 2
    if component_weights is not None:
        weights = torch.as_tensor(
            component_weights, dtype=squared.dtype, device=squared.device
        ).reshape(-1)
        if weights.shape != (squared.shape[-1],):
            raise ValueError("state_loss_weights must match state dimension")
        if not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
            raise ValueError("state_loss_weights must be finite and positive")
        squared = squared * weights / torch.mean(weights)
    return _weighted_mean(torch.mean(squared, dim=-1), sample_weights)


def transition_weights(dataset, indices):
    if "sample_weight" not in dataset:
        return None
    values = np.asarray(dataset["sample_weight"][indices], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("sample_weight values must be finite and positive")
    return values


def _split_path(directory, name):
    candidates = (Path(directory) / (name + ".npz"), Path(directory) / ("dataset_" + name + ".npz"))
    path = next((value for value in candidates if value.exists()), None)
    if path is None:
        raise FileNotFoundError("missing %s dataset split" % name)
    return path


def load_split(directory, name):
    path = _split_path(directory, name)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_provenance(directory):
    """Return immutable identifiers for every split visible to a run."""

    root = Path(directory).resolve()
    splits = {}
    for name in ("train", "validation", "test", "unseen"):
        try:
            path = _split_path(root, name).resolve()
        except FileNotFoundError:
            continue
        with np.load(path, allow_pickle=False) as archive:
            transitions = int(archive["state_t"].shape[0])
            episodes = int(np.unique(archive["episode_id"]).size)
        splits[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "transition_count": transitions,
            "episode_count": episodes,
        }
    manifest = root / "dataset_manifest.json"
    return {
        "directory": str(root),
        "splits": splits,
        "manifest": (
            {"path": str(manifest), "sha256": _sha256_file(manifest)}
            if manifest.exists() else None
        ),
    }


def statistics(train, angle_indices):
    state = torch.as_tensor(train["state_t"], dtype=torch.float32)
    feature = encode_state(state, angle_indices).numpy()
    values = {}
    for name, array in (
        ("feature", feature),
        ("control", train["control_t"]),
        ("residual", train["residual_target"]),
    ):
        mean = np.asarray(array, dtype=np.float64).mean(axis=0)
        scale = np.asarray(array, dtype=np.float64).std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        values[name + "_mean"] = mean.astype(np.float32)
        values[name + "_scale"] = scale.astype(np.float32)
    return values


def contiguous_windows(dataset, horizon):
    episode = dataset["episode_id"]
    step = dataset["step"]
    starts = []
    for index in range(0, len(step) - horizon + 1):
        selection = slice(index, index + horizon)
        if np.all(episode[selection] == episode[index]) and np.all(np.diff(step[selection]) == 1):
            starts.append(index)
    return np.asarray(starts, dtype=np.int64)


def rollout_loss(
    model, dataset, starts, horizon, dynamics_config, device,
    max_windows=128, component_weights=None, step_weights=None,
):
    if starts.size == 0:
        return torch.zeros((), device=device)
    selected = starts if max_windows is None else starts[:max_windows]
    state = torch.as_tensor(dataset["state_t"][selected], dtype=torch.float32, device=device)
    loss = torch.zeros((), device=device)
    if step_weights is None:
        resolved_step_weights = np.ones(horizon, dtype=np.float64)
    else:
        resolved_step_weights = np.asarray(step_weights, dtype=np.float64).reshape(-1)
        if resolved_step_weights.shape != (horizon,):
            raise ValueError("rollout step weights must match rollout_horizon")
        if not np.isfinite(resolved_step_weights).all() or np.any(resolved_step_weights <= 0.0):
            raise ValueError("rollout step weights must be finite and positive")
    resolved_step_weights /= float(np.sum(resolved_step_weights))
    for offset in range(horizon):
        control = torch.as_tensor(dataset["control_t"][selected + offset], dtype=torch.float32, device=device)
        dt = torch.as_tensor(dataset["dt"][selected + offset], dtype=torch.float32, device=device).unsqueeze(-1)
        state = integrate(model, state, control, dt, dynamics_config, dynamics_config.get("integrator", "rk4"))
        target = torch.as_tensor(dataset["state_t_plus_1"][selected + offset], dtype=torch.float32, device=device)
        loss = loss + float(resolved_step_weights[offset]) * state_mse(
            state, target, component_weights,
            sample_weights=transition_weights(dataset, selected + offset),
        )
    return loss


def rollout_step_weights(training, horizon):
    explicit = training.get("rollout_step_weights")
    if explicit is not None:
        values = np.asarray(explicit, dtype=np.float64).reshape(-1)
    else:
        mode = str(training.get("rollout_step_weighting", "uniform"))
        terminal = float(training.get("rollout_terminal_weight", 1.0))
        if not math.isfinite(terminal) or terminal <= 0.0:
            raise ValueError("rollout_terminal_weight must be finite and positive")
        if mode == "uniform":
            values = np.ones(horizon, dtype=np.float64)
        elif mode == "linear":
            values = np.linspace(1.0, terminal, horizon, dtype=np.float64)
        elif mode == "exponential":
            values = np.geomspace(1.0, terminal, horizon, dtype=np.float64)
        else:
            raise ValueError("unknown rollout_step_weighting: %s" % mode)
    if values.shape != (horizon,) or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("rollout step weights must be positive and match horizon")
    return values


def rollout_endpoint_error(model, dataset, starts, horizon, dynamics_config, device, max_windows=512):
    if starts.size == 0:
        return torch.empty((0, dataset["state_t"].shape[1]), device=device)
    selected = starts if max_windows is None else starts[:max_windows]
    state = torch.as_tensor(dataset["state_t"][selected], dtype=torch.float32, device=device)
    target = None
    for offset in range(horizon):
        control = torch.as_tensor(
            dataset["control_t"][selected + offset], dtype=torch.float32, device=device
        )
        dt = torch.as_tensor(
            dataset["dt"][selected + offset], dtype=torch.float32, device=device
        ).unsqueeze(-1)
        state = integrate(
            model, state, control, dt, dynamics_config,
            dynamics_config.get("integrator", "rk4"),
        )
        target = torch.as_tensor(
            dataset["state_t_plus_1"][selected + offset],
            dtype=torch.float32,
            device=device,
        )
    error = state - target
    wrapped = torch.atan2(torch.sin(error[..., 2:3]), torch.cos(error[..., 2:3]))
    return torch.cat((error[..., :2], wrapped, error[..., 3:]), dim=-1)


def train_residual(config: Mapping[str, object], dataset_dir, output_dir, device="cpu", epochs=None):
    config = copy.deepcopy(dict(config))
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    train = load_split(dataset_dir, "train")
    validation = load_split(dataset_dir, "validation")
    data_provenance = dataset_provenance(dataset_dir)
    if len(train["state_t"]) == 0 or len(validation["state_t"]) == 0:
        raise ValueError("training and validation splits must both be non-empty")
    assert_residual_dataset_quality(train, angle_indices=(2,))
    assert_residual_dataset_quality(validation, angle_indices=(2,))
    state_dim = int(train["state_t"].shape[1])
    control_dim = int(train["control_t"].shape[1])
    model_cfg = dict(config["model"])
    angle_indices = tuple(model_cfg.get("angle_indices", (2,)))
    training = config.get("training", {})
    initial_checkpoint = training.get("initial_checkpoint")
    if initial_checkpoint:
        model, initial_payload = load_platform_checkpoint(
            Path(initial_checkpoint), device=device
        )
        if model.state_dim != state_dim or model.control_dim != control_dim:
            raise ValueError(
                "initial checkpoint dimensions do not match the training dataset"
            )
        checkpoint_model_cfg = dict(
            initial_payload["model_config"].get("model", {})
        )
        for key in (
            "type", "angle_indices", "hidden_sizes", "activation",
            "residual_output_mask",
        ):
            if key in model_cfg and key in checkpoint_model_cfg:
                if model_cfg[key] != checkpoint_model_cfg[key]:
                    raise ValueError(
                        "initial checkpoint model config mismatch for %s" % key
                    )
        model.train()
    else:
        model = ResidualNetwork(
            state_dim, control_dim, model_cfg, statistics(train, angle_indices)
        ).to(device)
    anchor_weight = float(training.get("anchor_output_weight", 0.0))
    if not math.isfinite(anchor_weight) or anchor_weight < 0.0:
        raise ValueError("anchor_output_weight must be finite and nonnegative")
    if anchor_weight > 0.0 and not initial_checkpoint:
        raise ValueError(
            "anchor_output_weight requires training.initial_checkpoint"
        )
    anchor_model = None
    if anchor_weight > 0.0:
        anchor_model = copy.deepcopy(model).to(device).eval()
        for parameter in anchor_model.parameters():
            parameter.requires_grad_(False)
    anchor_sources = tuple(
        str(value) for value in training.get("anchor_data_sources", ())
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning_rate", 5e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    batch_size = int(training.get("batch_size", 256))
    total_epochs = int(epochs or training.get("epochs", 100))
    if total_epochs <= 0:
        raise ValueError("epochs must be positive")
    horizon = int(training.get("rollout_horizon", 10))
    component_weights = training.get("state_loss_weights")
    if component_weights is not None:
        component_weights = np.asarray(component_weights, dtype=np.float64).reshape(-1)
        if component_weights.shape != (state_dim,):
            raise ValueError("state_loss_weights must match state dimension")
        if not np.isfinite(component_weights).all() or np.any(component_weights <= 0.0):
            raise ValueError("state_loss_weights must be finite and positive")
    for name, dataset in (("train", train), ("validation", validation)):
        if "sample_weight" in dataset:
            values = np.asarray(dataset["sample_weight"], dtype=np.float64)
            if values.shape != (len(dataset["state_t"]),):
                raise ValueError("%s sample_weight must match transitions" % name)
            if not np.isfinite(values).all() or np.any(values <= 0.0):
                raise ValueError(
                    "%s sample_weight values must be finite and positive" % name
                )
    step_weights = rollout_step_weights(training, horizon)
    train_windows = contiguous_windows(train, horizon)
    validation_windows = contiguous_windows(validation, horizon)
    if train_windows.size == 0 or validation_windows.size == 0:
        raise ValueError(
            "rollout_horizon=%d requires at least one contiguous window in train and validation"
            % horizon
        )
    weights = training.get("loss_weights", {})
    derivative_weight = float(weights.get("derivative", 1.0))
    one_step_weight = float(weights.get("one_step", 1.0))
    multistep_weight = float(weights.get("multistep", 1.0))
    best = float("inf")
    patience = int(training.get("early_stopping_patience", 20))
    remaining = patience
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(total_epochs):
        model.train()
        order = np.random.RandomState(seed + epoch).permutation(len(train["state_t"]))
        epoch_losses = []
        epoch_anchor_losses = []
        for begin in range(0, len(order), batch_size):
            indices = order[begin:begin + batch_size]
            state = torch.as_tensor(train["state_t"][indices], dtype=torch.float32, device=device)
            control = torch.as_tensor(train["control_t"][indices], dtype=torch.float32, device=device)
            target_residual = torch.as_tensor(train["residual_target"][indices], dtype=torch.float32, device=device)
            next_state = torch.as_tensor(train["state_t_plus_1"][indices], dtype=torch.float32, device=device)
            dt = torch.as_tensor(train["dt"][indices], dtype=torch.float32, device=device).unsqueeze(-1)
            derivative_squared = (model(state, control) - target_residual) ** 2
            derivative_mask = model.residual_output_mask.reshape(1, -1)
            derivative_per_sample = torch.sum(
                derivative_squared * derivative_mask, dim=-1
            ) / torch.sum(derivative_mask)
            batch_sample_weights = transition_weights(train, indices)
            derivative_loss = _weighted_mean(
                derivative_per_sample, batch_sample_weights
            )
            predicted_next = integrate(model, state, control, dt, config["dynamics"], config["dynamics"].get("integrator", "rk4"))
            one_step_loss = state_mse(
                predicted_next, next_state, component_weights,
                sample_weights=batch_sample_weights,
            )
            loss = derivative_weight * derivative_loss + one_step_weight * one_step_loss
            anchor_loss = torch.zeros((), dtype=loss.dtype, device=device)
            if anchor_model is not None:
                if anchor_sources:
                    source_mask = np.isin(
                        np.asarray(train["data_source"])[indices],
                        np.asarray(anchor_sources),
                    )
                else:
                    source_mask = np.ones(len(indices), dtype=bool)
                if np.any(source_mask):
                    selected = np.flatnonzero(source_mask)
                    selected_tensor = torch.as_tensor(
                        selected, dtype=torch.long, device=device
                    )
                    current_output = model(
                        torch.index_select(state, 0, selected_tensor),
                        torch.index_select(control, 0, selected_tensor),
                    )
                    with torch.no_grad():
                        anchor_output = anchor_model(
                            torch.index_select(state, 0, selected_tensor),
                            torch.index_select(control, 0, selected_tensor),
                        )
                    active = model.residual_output_mask.reshape(1, -1)
                    anchor_per_sample = torch.sum(
                        (current_output - anchor_output) ** 2 * active, dim=-1
                    ) / torch.sum(active)
                    selected_weights = (
                        None if batch_sample_weights is None
                        else np.asarray(batch_sample_weights)[selected]
                    )
                    anchor_loss = _weighted_mean(
                        anchor_per_sample, selected_weights
                    )
                    loss = loss + anchor_weight * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip_norm", 1.0)))
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_anchor_losses.append(float(anchor_loss.detach().cpu()))
        model.train()
        optimizer.zero_grad()
        window_order = np.random.RandomState(seed + 100000 + epoch).permutation(train_windows)
        multi = rollout_loss(
            model, train, window_order, horizon, config["dynamics"], device,
            component_weights=component_weights, step_weights=step_weights,
        )
        if multi.requires_grad and multistep_weight > 0.0:
            (multistep_weight * multi).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip_norm", 1.0)))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_multi = float(rollout_loss(
                model, validation, validation_windows, horizon, config["dynamics"], device,
                component_weights=component_weights, step_weights=step_weights,
            ).cpu())
        scheduler.step(validation_multi)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "train_anchor_mse": float(np.mean(epoch_anchor_losses)),
            "train_multistep_mse": float(multi.detach().cpu()),
            "validation_multistep_rmse": math.sqrt(max(validation_multi, 0.0)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        payload = {
            "format": "mobile_robot_mppi_residual_v1",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "model_config": model.checkpoint_config(),
            "training_config": config,
            "epoch": epoch + 1,
            "best_validation_multistep_rmse": min(best, record["validation_multistep_rmse"]),
            "git_sha": git_sha(Path(__file__).resolve().parents[3]),
            "parameter_count": model.parameter_count(),
            "state_dim": state_dim,
            "control_dim": control_dim,
            "dataset_provenance": data_provenance,
        }
        torch.save(payload, output / "last.pt")
        if record["validation_multistep_rmse"] < best:
            best = record["validation_multistep_rmse"]
            remaining = patience
            torch.save(payload, output / "best.pt")
        else:
            remaining -= 1
            if remaining <= 0:
                break
    with (output / "training_log.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (output / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    with (output / "training_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "dataset": data_provenance,
            "git_sha": git_sha(Path(__file__).resolve().parents[3]),
            "seed": seed,
            "state_dim": state_dim,
            "control_dim": control_dim,
            "model_type": model.model_type,
        }, handle, indent=2, sort_keys=True)
    return {
        "model": model,
        "history": history,
        "best_validation_multistep_rmse": best,
        "parameter_count": model.parameter_count(),
    }
