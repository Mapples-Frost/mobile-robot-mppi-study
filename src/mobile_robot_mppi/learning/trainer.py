import copy
import json
import math
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from mobile_robot_mppi.core.config import git_sha
from .models import ResidualNetwork, encode_state


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


def state_mse(predicted, target):
    error = predicted - target
    wrapped = torch.atan2(torch.sin(error[..., 2:3]), torch.cos(error[..., 2:3]))
    periodic_error = torch.cat((error[..., :2], wrapped, error[..., 3:]), dim=-1)
    return torch.mean(periodic_error ** 2)


def load_split(directory, name):
    candidates = (Path(directory) / (name + ".npz"), Path(directory) / ("dataset_" + name + ".npz"))
    path = next((value for value in candidates if value.exists()), None)
    if path is None:
        raise FileNotFoundError("missing %s dataset split" % name)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


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


def rollout_loss(model, dataset, starts, horizon, dynamics_config, device, max_windows=128):
    if starts.size == 0:
        return torch.zeros((), device=device)
    selected = starts[:max_windows]
    state = torch.as_tensor(dataset["state_t"][selected], dtype=torch.float32, device=device)
    loss = torch.zeros((), device=device)
    for offset in range(horizon):
        control = torch.as_tensor(dataset["control_t"][selected + offset], dtype=torch.float32, device=device)
        dt = torch.as_tensor(dataset["dt"][selected + offset], dtype=torch.float32, device=device).unsqueeze(-1)
        state = integrate(model, state, control, dt, dynamics_config, dynamics_config.get("integrator", "rk4"))
        target = torch.as_tensor(dataset["state_t_plus_1"][selected + offset], dtype=torch.float32, device=device)
        loss = loss + state_mse(state, target)
    return loss / float(horizon)


def train_residual(config: Mapping[str, object], dataset_dir, output_dir, device="cpu", epochs=None):
    config = copy.deepcopy(dict(config))
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    train = load_split(dataset_dir, "train")
    validation = load_split(dataset_dir, "validation")
    state_dim = int(train["state_t"].shape[1])
    control_dim = int(train["control_t"].shape[1])
    model_cfg = dict(config["model"])
    angle_indices = tuple(model_cfg.get("angle_indices", (2,)))
    model = ResidualNetwork(state_dim, control_dim, model_cfg, statistics(train, angle_indices)).to(device)
    training = config.get("training", {})
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning_rate", 5e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    batch_size = int(training.get("batch_size", 256))
    total_epochs = int(epochs or training.get("epochs", 100))
    horizon = int(training.get("rollout_horizon", 10))
    train_windows = contiguous_windows(train, horizon)
    validation_windows = contiguous_windows(validation, horizon)
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
        for begin in range(0, len(order), batch_size):
            indices = order[begin:begin + batch_size]
            state = torch.as_tensor(train["state_t"][indices], dtype=torch.float32, device=device)
            control = torch.as_tensor(train["control_t"][indices], dtype=torch.float32, device=device)
            target_residual = torch.as_tensor(train["residual_target"][indices], dtype=torch.float32, device=device)
            next_state = torch.as_tensor(train["state_t_plus_1"][indices], dtype=torch.float32, device=device)
            dt = torch.as_tensor(train["dt"][indices], dtype=torch.float32, device=device).unsqueeze(-1)
            derivative_loss = torch.mean((model(state, control) - target_residual) ** 2)
            predicted_next = integrate(model, state, control, dt, config["dynamics"], config["dynamics"].get("integrator", "rk4"))
            one_step_loss = state_mse(predicted_next, next_state)
            loss = derivative_weight * derivative_loss + one_step_weight * one_step_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip_norm", 1.0)))
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.train()
        optimizer.zero_grad()
        multi = rollout_loss(model, train, train_windows, horizon, config["dynamics"], device)
        if multi.requires_grad and multistep_weight > 0.0:
            (multistep_weight * multi).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip_norm", 1.0)))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_multi = float(rollout_loss(
                model, validation, validation_windows, horizon, config["dynamics"], device
            ).cpu())
        scheduler.step(validation_multi)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
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
    return {"model": model, "history": history, "best_validation_multistep_rmse": best}
