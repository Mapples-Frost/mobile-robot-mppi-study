"""Train and gate a frozen-base dynamic-obstacle Actor correction on CUDA."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint
from mobile_robot_mppi.rl.observation import (
    ObservationEncoderConfig,
    RunningNormalizer,
)
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig


DEFAULT_CONFIG = ROOT / "configs/research/dynamic_actor_correction_development.yaml"


def _target_log_std_values(value, action_dim):
    values = np.asarray(value, dtype=np.float32)
    if values.ndim == 0:
        return np.full(int(action_dim), float(values), dtype=np.float32)
    if values.shape != (int(action_dim),):
        raise ValueError("target_log_std must be scalar or action-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("target_log_std must be finite")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_dataset(root: Path):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = {}
    keys = None
    for descriptor in manifest["shards"]:
        shard_path = root / descriptor["file"]
        if _sha256(shard_path) != descriptor["sha256"]:
            raise ValueError("dynamic Actor teacher shard SHA256 mismatch")
        with np.load(shard_path, allow_pickle=False) as shard:
            current = set(shard.files)
            if keys is None:
                keys = current
                arrays = {name: [] for name in keys}
            elif current != keys:
                raise ValueError("dynamic Actor teacher shard schema drifted")
            for name in keys:
                arrays[name].append(shard[name].copy())
    result = {
        name: np.concatenate(values, axis=0) for name, values in arrays.items()
    }
    if result["observations"].shape[1:] != (manifest["observation_dim"],):
        raise ValueError("dynamic Actor teacher observation shape mismatch")
    if result["target_actions"].shape[1:] != (manifest["action_dim"],):
        raise ValueError("dynamic Actor teacher action shape mismatch")
    return manifest, result, _sha256(manifest_path)


def _physical_to_normalized(actions, lower, upper):
    center = 0.5 * (upper + lower)
    half = 0.5 * (upper - lower)
    return np.clip((np.asarray(actions) - center) / half, -1.0, 1.0)


def _normalized_to_physical(actions, lower, upper):
    center = 0.5 * (upper + lower)
    half = 0.5 * (upper - lower)
    return np.clip(center + half * np.asarray(actions), lower, upper)


def _temporal_scan_observations(observations, episode_seeds, steps, config):
    """Append causal sector-range deltas reconstructed within each episode."""

    values = np.asarray(observations, dtype=np.float32)
    encoder = ObservationEncoderConfig.from_mapping(config)
    if encoder.history_frames != 1:
        raise ValueError("temporal scan expansion requires one-frame source data")
    sectors = int(encoder.lidar_sectors)
    scan_stop = values.shape[1] - 1
    scan_start = scan_stop - sectors
    if scan_start < 0:
        raise ValueError("source observation does not contain lidar sectors")
    delta = np.zeros((len(values), sectors), dtype=np.float32)
    seeds = np.asarray(episode_seeds)
    step_values = np.asarray(steps)
    consecutive = (
        (seeds[1:] == seeds[:-1])
        & (step_values[1:] == step_values[:-1] + 1)
    )
    raw_delta = values[1:, scan_start:scan_stop] - values[:-1, scan_start:scan_stop]
    current_indices = np.flatnonzero(consecutive) + 1
    delta[current_indices] = np.clip(
        raw_delta[consecutive] / float(encoder.temporal_scan_delta_scale),
        -1.0,
        1.0,
    )
    return np.concatenate((values, delta), axis=1)


def _expanded_normalizer(source, raw_observations):
    """Preserve legacy normalization exactly and fit only appended features."""

    values = np.asarray(raw_observations, dtype=np.float64)
    if values.shape[1] < source.dimension:
        raise ValueError("expanded normalizer cannot shrink observations")
    if values.shape[1] == source.dimension:
        return source
    extra = values[:, source.dimension:]
    result = RunningNormalizer(
        values.shape[1], min_std=source.min_std, clip=source.clip
    )
    result.count = int(source.count)
    result.mean[:source.dimension] = source.mean
    result.m2[:source.dimension] = source.m2
    result.mean[source.dimension:] = np.mean(extra, axis=0)
    if result.count >= 2:
        variance = np.var(extra, axis=0, ddof=1)
        result.m2[source.dimension:] = variance * float(result.count - 1)
    return result


def _copy_source_nonactor(source_state, agent):
    for name in ("critic1", "critic2", "target_critic1", "target_critic2"):
        getattr(agent, name).load_state_dict(source_state[name])
    agent.log_alpha.data.copy_(
        torch.as_tensor(source_state["log_alpha"], device=agent.device)
    )
    agent.target_entropy = float(source_state.get(
        "target_entropy", agent.target_entropy
    ))


def _action_migration(payload, training):
    saved = payload["action_spec"]
    source_lower = np.asarray(saved["lower"], dtype=np.float32)
    source_upper = np.asarray(saved["upper"], dtype=np.float32)
    target_lower = np.asarray(
        training.get("action_lower", source_lower), dtype=np.float32
    )
    target_upper = np.asarray(
        training.get("action_upper", source_upper), dtype=np.float32
    )
    if (
        target_lower.shape != source_lower.shape
        or target_upper.shape != source_upper.shape
        or not np.isfinite(target_lower).all()
        or not np.isfinite(target_upper).all()
        or np.any(target_lower >= target_upper)
        or np.any(target_lower > source_lower)
        or np.any(target_upper < source_upper)
    ):
        raise ValueError(
            "dynamic Actor migrated action bounds must contain the source space"
        )
    source_center = 0.5 * (source_upper + source_lower)
    source_half = 0.5 * (source_upper - source_lower)
    target_center = 0.5 * (target_upper + target_lower)
    target_half = 0.5 * (target_upper - target_lower)
    base_scale = source_half / target_half
    base_offset = (source_center - target_center) / target_half
    if np.any(np.abs(base_offset) + base_scale > 1.0 + 1.0e-6):
        raise ValueError("dynamic Actor base-action migration leaves bounds")
    return {
        "source_lower": source_lower,
        "source_upper": source_upper,
        "target_lower": target_lower,
        "target_upper": target_upper,
        "base_scale": base_scale.astype(np.float32),
        "base_offset": base_offset.astype(np.float32),
        "performed": bool(
            not np.allclose(source_lower, target_lower)
            or not np.allclose(source_upper, target_upper)
        ),
    }


def _make_agents(payload, training, migration, observation_dim):
    state = payload["agent"]
    source_agent = SACAgent(
        state["observation_dim"],
        state["action_dim"],
        SACConfig.from_mapping(state["config"]),
        device=training["device"],
        seed=int(training["seed"]),
    )
    source_agent.load_state_dict(state, load_optimizers=False)
    source_agent.eval()

    correction_mapping = dict(state["config"])
    correction_mapping.update({
        "policy_mode": "frozen_bc_correction",
        "actor_lr": float(training["learning_rate"]),
        "correction_scale": tuple(float(value) for value in training[
            "correction_scale"
        ]),
        "correction_base_action_scale": tuple(
            float(value) for value in migration["base_scale"]
        ),
        "correction_base_action_offset": tuple(
            float(value) for value in migration["base_offset"]
        ),
        "correction_base_observation_dim": int(state["observation_dim"]),
        "correction_initial_log_std": float(np.mean(
            _target_log_std_values(
                training["target_log_std"], state["action_dim"]
            )
        )),
    })
    agent = SACAgent(
        int(observation_dim),
        state["action_dim"],
        SACConfig.from_mapping(correction_mapping),
        device=training["device"],
        seed=int(training["seed"]),
    )
    if int(observation_dim) == int(state["observation_dim"]):
        _copy_source_nonactor(state, agent)
    agent.initialize_frozen_base_actor(state["actor"])
    agent.train()
    return source_agent, agent


def _predict(agent, normalized_observations, batch_size=4096):
    predictions = []
    agent.eval()
    for start in range(0, len(normalized_observations), int(batch_size)):
        predictions.append(agent.select_action_batch(
            normalized_observations[start:start + int(batch_size)],
            deterministic=True,
        ))
    return np.concatenate(predictions, axis=0)


def _metrics(
    source_normalized,
    candidate_normalized,
    target_normalized,
    arrays,
    mask,
    lower,
    upper,
    retention_mask,
    critical_mask=None,
):
    selected = np.asarray(mask, dtype=bool)
    opportunity = selected & arrays["low_risk_opportunity"].astype(bool)
    retention = selected & np.asarray(retention_mask, dtype=bool)
    critical = selected & np.asarray(
        critical_mask if critical_mask is not None else np.zeros_like(selected),
        dtype=bool,
    )
    source_physical = _normalized_to_physical(source_normalized, lower, upper)
    candidate_physical = _normalized_to_physical(
        candidate_normalized, lower, upper
    )
    target_physical = _normalized_to_physical(target_normalized, lower, upper)
    reverse = selected & (target_physical[:, 0] < -0.02)

    def _rmse(values):
        return float(np.sqrt(np.mean(np.square(values))))

    source_error = source_normalized[selected] - target_normalized[selected]
    candidate_error = (
        candidate_normalized[selected] - target_normalized[selected]
    )
    minimum_active = 0.10
    stopped = 0.05
    result = {
        "samples": int(np.sum(selected)),
        "opportunity_samples": int(np.sum(opportunity)),
        "retention_samples": int(np.sum(retention)),
        "reverse_samples": int(np.sum(reverse)),
        "critical_samples": int(np.sum(critical)),
        "source_teacher_rmse": _rmse(source_error),
        "candidate_teacher_rmse": _rmse(candidate_error),
        "source_physical_teacher_rmse": _rmse(
            source_physical[selected] - target_physical[selected]
        ),
        "candidate_physical_teacher_rmse": _rmse(
            candidate_physical[selected] - target_physical[selected]
        ),
        "source_opportunity_active_passage_fraction": float(np.mean(
            source_physical[opportunity, 0] >= minimum_active
        )) if np.any(opportunity) else 0.0,
        "candidate_opportunity_active_passage_fraction": float(np.mean(
            candidate_physical[opportunity, 0] >= minimum_active
        )) if np.any(opportunity) else 0.0,
        "source_opportunity_unnecessary_stop_fraction": float(np.mean(
            source_physical[opportunity, 0] < stopped
        )) if np.any(opportunity) else 0.0,
        "candidate_opportunity_unnecessary_stop_fraction": float(np.mean(
            candidate_physical[opportunity, 0] < stopped
        )) if np.any(opportunity) else 0.0,
        "source_reverse_recall": float(np.mean(
            source_physical[reverse, 0] < -0.02
        )) if np.any(reverse) else 0.0,
        "candidate_reverse_recall": float(np.mean(
            candidate_physical[reverse, 0] < -0.02
        )) if np.any(reverse) else 0.0,
        "candidate_reverse_velocity_mae": float(np.mean(np.abs(
            candidate_physical[reverse, 0] - target_physical[reverse, 0]
        ))) if np.any(reverse) else 0.0,
        "source_critical_physical_teacher_rmse": _rmse(
            source_physical[critical] - target_physical[critical]
        ) if np.any(critical) else 0.0,
        "candidate_critical_physical_teacher_rmse": _rmse(
            candidate_physical[critical] - target_physical[critical]
        ) if np.any(critical) else 0.0,
        "retention_mean_absolute_drift": float(np.mean(np.abs(
            candidate_physical[retention] - source_physical[retention]
        ))) if np.any(retention) else 0.0,
        "mean_absolute_correction": float(np.mean(np.abs(
            candidate_physical[selected] - source_physical[selected]
        ))),
    }
    result["relative_teacher_rmse_improvement"] = float(
        (result["source_teacher_rmse"] - result["candidate_teacher_rmse"])
        / max(result["source_teacher_rmse"], 1.0e-12)
    )
    result["opportunity_active_passage_gain"] = float(
        result["candidate_opportunity_active_passage_fraction"]
        - result["source_opportunity_active_passage_fraction"]
    )
    return result


def _sample(rng, indices, count):
    values = np.asarray(indices, dtype=np.int64)
    if values.size == 0:
        raise ValueError("dynamic Actor training stratum is empty")
    return values[rng.randint(values.size, size=int(count))]


def _save_checkpoint(
    path,
    source_payload,
    agent,
    training_state,
    migration,
    normalizer,
    encoder_config,
):
    payload = copy.deepcopy(source_payload)
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    payload["git_sha"] = git_sha(ROOT)
    payload["agent"] = agent.state_dict()
    payload["normalizer"] = normalizer.state_dict()
    payload["encoder_config"] = dict(encoder_config)
    payload["training_state"] = dict(training_state)
    source_action = copy.deepcopy(source_payload["action_spec"])
    payload["encoder_action_spec"] = source_action
    payload["action_spec"] = {
        "names": tuple(source_action["names"]),
        "lower": migration["target_lower"].copy(),
        "upper": migration["target_upper"].copy(),
    }
    resolved = copy.deepcopy(payload.get("resolved_config", {}))
    resolved.setdefault("rl", {}).setdefault("training", {}).update({
        "action_mode": "direct_control",
        "policy_mode": "frozen_bc_correction",
        "dynamic_actor_correction": True,
    })
    payload["resolved_config"] = resolved
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    return _sha256(path)


def train(
    config_path=DEFAULT_CONFIG,
    dataset_dir=None,
    output_dir=None,
    additional_dataset_dirs=None,
):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = dict(config["training"])
    device_name = str(training["device"]).lower()
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
    elif not (
        device_name == "cpu"
        and bool(training.get("allow_cpu_development", False))
    ):
        raise ValueError(
            "dynamic Actor correction training requires CUDA unless "
            "allow_cpu_development is explicitly enabled"
        )
    dataset_root = Path(
        dataset_dir or ROOT / config["collection"]["output_dir"]
    ).resolve()
    output = Path(output_dir or ROOT / training["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError("dynamic Actor correction output already exists")
    output.mkdir(parents=True, exist_ok=False)
    dataset_roots = [dataset_root] + [
        Path(value).resolve() for value in (additional_dataset_dirs or ())
    ]
    loaded = [_load_dataset(value) for value in dataset_roots]
    manifests = [value[0] for value in loaded]
    manifest_hashes = [value[2] for value in loaded]
    schema_keys = set(loaded[0][1])
    if any(set(value[1]) != schema_keys for value in loaded[1:]):
        raise ValueError("dynamic Actor datasets have incompatible schemas")
    observation_dims = [value[1]["observations"].shape[1] for value in loaded]
    base_observation_dim = min(observation_dims)
    for _, dataset, _ in loaded:
        if dataset["observations"].shape[1] != base_observation_dim:
            extra = dataset["observations"].shape[1] - base_observation_dim
            if extra != 36:
                raise ValueError(
                    "dynamic Actor dataset observation expansion is incompatible"
                )
            dataset["observations"] = dataset["observations"][
                :, :base_observation_dim
            ]
    arrays = {
        name: np.concatenate([value[1][name] for value in loaded], axis=0)
        for name in schema_keys
    }
    manifest = manifests[0]
    manifest_sha = hashlib.sha256(
        "\n".join(manifest_hashes).encode("ascii")
    ).hexdigest()
    source_checkpoint = ROOT / config["source_checkpoint"]
    if _sha256(source_checkpoint) != config["source_checkpoint_sha256"]:
        raise ValueError("dynamic Actor source checkpoint SHA256 mismatch")
    source_payload = load_sac_checkpoint(
        source_checkpoint, map_location=training["device"]
    )
    retention_payload = source_payload
    retention_checkpoint_value = config.get("retention_checkpoint")
    if retention_checkpoint_value is not None:
        retention_checkpoint = ROOT / retention_checkpoint_value
        expected_retention_sha = str(config["retention_checkpoint_sha256"])
        if _sha256(retention_checkpoint) != expected_retention_sha:
            raise ValueError("dynamic Actor retention checkpoint SHA256 mismatch")
        retention_payload = load_sac_checkpoint(
            retention_checkpoint, map_location=training["device"]
        )
    migration = _action_migration(source_payload, training)
    source_normalizer = RunningNormalizer.from_state_dict(
        retention_payload["normalizer"]
    )
    encoder_config = dict(source_payload["encoder_config"])
    temporal_enabled = bool(training.get("include_temporal_scan_delta", False))
    if temporal_enabled:
        encoder_config.update({
            "include_temporal_scan_delta": True,
            "temporal_scan_delta_scale": float(
                training.get("temporal_scan_delta_scale", 0.05)
            ),
        })
        raw_observations = _temporal_scan_observations(
            arrays["observations"],
            arrays["episode_seeds"],
            arrays["steps"],
            encoder_config,
        )
    else:
        raw_observations = arrays["observations"]
    normalizer = _expanded_normalizer(source_normalizer, raw_observations)
    source_agent, agent = _make_agents(
        source_payload, training, migration, raw_observations.shape[1]
    )
    if retention_payload is not source_payload:
        retention_state = retention_payload["agent"]
        source_agent = SACAgent(
            retention_state["observation_dim"],
            retention_state["action_dim"],
            SACConfig.from_mapping(retention_state["config"]),
            device=training["device"],
            seed=int(training["seed"]),
        )
        source_agent.load_state_dict(retention_state, load_optimizers=False)
        source_agent.eval()
    observations = normalizer.normalize(raw_observations).astype(np.float32)
    source_observations = source_normalizer.normalize(
        arrays["observations"]
    ).astype(np.float32)
    lower = migration["target_lower"]
    upper = migration["target_upper"]
    target_field = (
        "executed_actions" if migration["performed"] else "target_actions"
    )
    targets = _physical_to_normalized(
        np.clip(arrays[target_field], lower, upper), lower, upper
    ).astype(np.float32)
    source_latent = _predict(source_agent, source_observations)
    source_physical = _normalized_to_physical(
        source_latent,
        migration["source_lower"],
        migration["source_upper"],
    )
    source_actions = _physical_to_normalized(
        source_physical, lower, upper
    ).astype(np.float32)
    validation = arrays["validation"].astype(bool)
    train_mask = ~validation
    validation_mask = validation
    if not np.any(validation_mask):
        episode_seeds = np.unique(arrays["episode_seeds"])
        validation_mask = arrays["episode_seeds"] == episode_seeds[-1]
        train_mask = ~validation_mask
    opportunity_mask = arrays["low_risk_opportunity"].astype(bool)
    intervention_mask = (
        arrays["safety_override"].astype(bool)
        | (arrays["maximum_probability"] > float(
            config["collection"]["low_risk_maximum_probability"]
        ))
    )
    target_physical = _normalized_to_physical(targets, lower, upper)
    reverse_mask = target_physical[:, 0] < -0.02
    collision_exposure = arrays["episode_collision"].astype(bool)
    critical_probability_floor = float(
        training.get("critical_minimum_probability", 0.02)
    )
    critical_mask = (
        (
            collision_exposure
            & (
                arrays["nearest_dynamic_obstacle_distance"]
                <= float(training.get(
                    "critical_maximum_obstacle_distance_m", 1.50
                ))
            )
        )
        | (arrays["maximum_probability"] >= critical_probability_floor)
        | arrays["safety_override"].astype(bool)
    )
    agreement = np.max(np.abs(source_actions - targets), axis=1) <= float(
        training.get("retention_agreement_tolerance", 0.20)
    )
    retention_mask = (
        (arrays["maximum_probability"] <= 0.05)
        & ~arrays["safety_override"].astype(bool)
        & agreement
    )
    train_indices = np.flatnonzero(train_mask)
    opportunity_indices = np.flatnonzero(train_mask & opportunity_mask)
    intervention_indices = np.flatnonzero(train_mask & intervention_mask)
    retention_indices = np.flatnonzero(train_mask & retention_mask)
    reverse_indices = np.flatnonzero(train_mask & reverse_mask)
    critical_indices = np.flatnonzero(train_mask & critical_mask)
    if retention_indices.size == 0:
        retention_indices = np.flatnonzero(train_mask & agreement)
    if (
        opportunity_indices.size == 0
        or intervention_indices.size == 0
        or (
            float(training.get("reverse_fraction", 0.0)) > 0.0
            and reverse_indices.size == 0
        )
        or (
            float(training.get("critical_fraction", 0.0)) > 0.0
            and critical_indices.size == 0
        )
    ):
        raise ValueError("dynamic Actor dataset lacks required strata")

    batch_size = int(training["batch_size"])
    counts = {
        "opportunity": int(round(
            batch_size * float(training["opportunity_fraction"])
        )),
        "intervention": int(round(
            batch_size * float(training["intervention_fraction"])
        )),
        "reverse": int(round(
            batch_size * float(training.get("reverse_fraction", 0.0))
        )),
        "critical": int(round(
            batch_size * float(training.get("critical_fraction", 0.0))
        )),
    }
    counts["retention"] = (
        batch_size
        - counts["opportunity"]
        - counts["intervention"]
        - counts["reverse"]
        - counts["critical"]
    )
    if counts["retention"] <= 0:
        raise ValueError("dynamic Actor retention batch must be nonempty")
    rng = np.random.RandomState(int(training["seed"]))
    torch.manual_seed(int(training["seed"]))
    torch.cuda.manual_seed_all(int(training["seed"]))
    action_weights = torch.as_tensor(
        [training["velocity_loss_weight"], training["yaw_loss_weight"]],
        dtype=torch.float32,
        device=agent.device,
    ).view(1, -1)
    target_log_std = torch.as_tensor(
        _target_log_std_values(training["target_log_std"], len(lower)),
        dtype=torch.float32,
        device=agent.device,
    ).view(1, -1)
    velocity_center = float(0.5 * (upper[0] + lower[0]))
    velocity_half = float(0.5 * (upper[0] - lower[0]))
    minimum_active_normalized = float(
        (float(training["anti_freeze_minimum_speed_mps"]) - velocity_center)
        / velocity_half
    )
    reverse_boundary_normalized = float(
        (-0.02 - velocity_center) / velocity_half
    )
    progress = []
    best_score = float("inf")
    best_path = output / "checkpoints" / "best.pt"
    best_update = 0
    improved_checkpoints = []
    updates = int(training["updates"])
    for update in range(1, updates + 1):
        opportunity_batch = _sample(
            rng, opportunity_indices, counts["opportunity"]
        )
        intervention_batch = _sample(
            rng, intervention_indices, counts["intervention"]
        )
        reverse_batch = (
            _sample(rng, reverse_indices, counts["reverse"])
            if counts["reverse"] > 0
            else np.empty(0, dtype=np.int64)
        )
        critical_batch = (
            _sample(rng, critical_indices, counts["critical"])
            if counts["critical"] > 0
            else np.empty(0, dtype=np.int64)
        )
        retention_batch = _sample(
            rng, retention_indices, counts["retention"]
        )
        indices = np.concatenate((
            opportunity_batch,
            intervention_batch,
            reverse_batch,
            critical_batch,
            retention_batch,
        ))
        kinds = np.concatenate((
            np.zeros(counts["opportunity"], dtype=np.int64),
            np.ones(counts["intervention"], dtype=np.int64),
            np.full(counts["reverse"], 3, dtype=np.int64),
            np.full(counts["critical"], 4, dtype=np.int64),
            np.full(counts["retention"], 2, dtype=np.int64),
        ))
        order = rng.permutation(batch_size)
        indices = indices[order]
        kinds = kinds[order]
        observation_tensor = torch.as_tensor(
            observations[indices], dtype=torch.float32, device=agent.device
        )
        target_tensor = torch.as_tensor(
            targets[indices], dtype=torch.float32, device=agent.device
        )
        source_tensor = torch.as_tensor(
            source_actions[indices], dtype=torch.float32, device=agent.device
        )
        kind_tensor = torch.as_tensor(kinds, device=agent.device)
        with torch.no_grad():
            base_tensor = agent._map_frozen_base_action(
                agent.base_actor.mean_action(
                    agent._base_observation(observation_tensor)
                )
            )
        correction_mean, correction_log_std = agent.actor.distribution(
            observation_tensor
        )
        unit_correction = torch.tanh(correction_mean)
        predicted, applied, _ = agent._compose_correction(
            base_tensor, unit_correction
        )
        teacher_mask = (kind_tensor != 2).to(torch.float32).view(-1, 1)
        teacher_error = (predicted - target_tensor).square() * action_weights
        reverse_tensor = kind_tensor == 3
        if bool(torch.any(reverse_tensor)):
            teacher_error[reverse_tensor, 0] *= float(
                training.get("reverse_velocity_loss_multiplier", 1.0)
            )
        critical_tensor = kind_tensor == 4
        if bool(torch.any(critical_tensor)):
            teacher_error[critical_tensor] *= float(
                training.get("critical_teacher_loss_multiplier", 1.0)
            )
        teacher_loss = torch.sum(teacher_error * teacher_mask) / torch.clamp(
            teacher_mask.sum() * action_weights.sum(), min=1.0
        )
        opportunity_tensor = kind_tensor == 0
        anti_freeze_loss = torch.relu(
            minimum_active_normalized - predicted[opportunity_tensor, 0]
        ).square().mean()
        retention_tensor = kind_tensor == 2
        retention_loss = (
            predicted[retention_tensor] - source_tensor[retention_tensor]
        ).square().mean()
        nonreverse_tensor = target_tensor[:, 0] >= reverse_boundary_normalized
        false_reverse_loss = torch.relu(
            reverse_boundary_normalized - predicted[nonreverse_tensor, 0]
        ).square().mean()
        correction_loss = applied.square().mean()
        log_std_loss = (
            correction_log_std - target_log_std
        ).square().mean()
        loss = (
            teacher_loss
            + float(training["anti_freeze_weight"]) * anti_freeze_loss
            + float(training["retention_weight"]) * retention_loss
            + float(training.get("false_reverse_weight", 0.0))
            * false_reverse_loss
            + float(training["correction_weight"]) * correction_loss
            + float(training["log_std_weight"]) * log_std_loss
        )
        agent.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = agent._clip(agent.actor)
        agent.actor_optimizer.step()
        agent.bc_update_steps += 1

        validate_now = (
            update % int(training["validation_interval"]) == 0
            or update == updates
        )
        if validate_now:
            candidate_actions = _predict(agent, observations)
            validation_metrics = _metrics(
                source_actions,
                candidate_actions,
                targets,
                arrays,
                validation_mask,
                lower,
                upper,
                retention_mask,
                critical_mask,
            )
            training_metrics_for_selection = _metrics(
                source_actions,
                candidate_actions,
                targets,
                arrays,
                train_mask,
                lower,
                upper,
                retention_mask,
                critical_mask,
            )
            score = (
                validation_metrics["candidate_physical_teacher_rmse"]
                + float(training.get("selection_reverse_weight", 0.0))
                * validation_metrics["candidate_reverse_velocity_mae"]
                + float(training.get("selection_retention_weight", 0.0))
                * validation_metrics["retention_mean_absolute_drift"]
                + float(training.get("selection_critical_weight", 0.0))
                * training_metrics_for_selection[
                    "candidate_critical_physical_teacher_rmse"
                ]
            )
            row = {
                "update": update,
                "loss": float(loss.detach().cpu()),
                "teacher_loss": float(teacher_loss.detach().cpu()),
                "anti_freeze_loss": float(anti_freeze_loss.detach().cpu()),
                "retention_loss": float(retention_loss.detach().cpu()),
                "false_reverse_loss": float(false_reverse_loss.detach().cpu()),
                "correction_loss": float(correction_loss.detach().cpu()),
                "gradient_norm": float(gradient),
                "selection_critical_physical_teacher_rmse": (
                    training_metrics_for_selection[
                        "candidate_critical_physical_teacher_rmse"
                    ]
                ),
                **validation_metrics,
            }
            progress.append(row)
            _write_json(output / "progress.json", progress)
            if score < best_score:
                best_score = score
                best_update = update
                _save_checkpoint(best_path, source_payload, agent, {
                    "phase": "dynamic_actor_correction_imitation",
                    "protocol": config["protocol"],
                    "device": str(agent.device),
                    "seed": int(training["seed"]),
                    "updates": update,
                    "source_checkpoint": config["source_checkpoint"],
                    "source_checkpoint_sha256": config[
                        "source_checkpoint_sha256"
                    ],
                    "retention_checkpoint": retention_checkpoint_value,
                    "retention_checkpoint_sha256": config.get(
                        "retention_checkpoint_sha256"
                    ),
                    "teacher_manifest_sha256": manifest_sha,
                    "teacher_manifest_sha256_list": manifest_hashes,
                    "frozen_base_actor_sha256": agent.frozen_base_actor_sha256(),
                    "action_space_migration_performed": migration["performed"],
                    "source_action_lower": migration["source_lower"].tolist(),
                    "source_action_upper": migration["source_upper"].tolist(),
                    "target_action_lower": migration["target_lower"].tolist(),
                    "target_action_upper": migration["target_upper"].tolist(),
                    "base_action_scale": migration["base_scale"].tolist(),
                    "base_action_offset": migration["base_offset"].tolist(),
                    "future_truth_available_to_student": False,
                }, migration, normalizer, encoder_config)
                if bool(training.get("save_improved_checkpoints", False)):
                    improved_path = output / "checkpoints" / (
                        "improved_update_%06d.pt" % update
                    )
                    _save_checkpoint(
                        improved_path,
                        source_payload,
                        agent,
                        {
                            "phase": "dynamic_actor_correction_imitation",
                            "protocol": config["protocol"],
                            "device": str(agent.device),
                            "seed": int(training["seed"]),
                            "updates": update,
                            "offline_selection_score": float(score),
                            "teacher_manifest_sha256": manifest_sha,
                            "teacher_manifest_sha256_list": manifest_hashes,
                            "future_truth_available_to_student": False,
                        },
                        migration,
                        normalizer,
                        encoder_config,
                    )
                    improved_checkpoints.append(str(
                        improved_path.relative_to(ROOT)
                    ))
            print(json.dumps({
                "stage": "actor_training",
                "device": str(agent.device),
                "update": update,
                "updates": updates,
                "validation_rmse": score,
                "best_update": best_update,
            }, sort_keys=True), flush=True)
        agent.train()

    best_payload = load_sac_checkpoint(best_path, map_location=training["device"])
    best_agent = SACAgent(
        best_payload["agent"]["observation_dim"],
        best_payload["agent"]["action_dim"],
        SACConfig.from_mapping(best_payload["agent"]["config"]),
        device=training["device"],
    )
    best_agent.load_state_dict(best_payload["agent"], load_optimizers=False)
    best_actions = _predict(best_agent, observations)
    train_metrics = _metrics(
        source_actions, best_actions, targets, arrays, train_mask,
        lower, upper, retention_mask, critical_mask,
    )
    validation_metrics = _metrics(
        source_actions, best_actions, targets, arrays, validation_mask,
        lower, upper, retention_mask, critical_mask,
    )
    gate = config["offline_gate"]
    checks = {
        "teacher_rmse_improvement": (
            validation_metrics["relative_teacher_rmse_improvement"]
            >= float(gate["minimum_relative_teacher_rmse_improvement"])
        ),
        "active_passage_gain": (
            validation_metrics["opportunity_active_passage_gain"]
            >= float(gate["minimum_opportunity_active_passage_gain"])
        ),
        "unnecessary_stop": (
            validation_metrics[
                "candidate_opportunity_unnecessary_stop_fraction"
            ] <= float(gate[
                "maximum_opportunity_unnecessary_stop_fraction"
            ])
        ),
        "retention_drift": (
            validation_metrics["retention_mean_absolute_drift"]
            <= float(gate["maximum_retention_mean_absolute_drift"])
        ),
        "reverse_recall": (
            validation_metrics["candidate_reverse_recall"]
            >= float(gate.get("minimum_reverse_recall", 0.0))
        ),
        "cuda_training": str(best_agent.device).startswith("cuda"),
        "frozen_base_unchanged": (
            best_agent.frozen_base_actor_sha256()
            == agent.frozen_base_actor_sha256()
        ),
        "all_finite": bool(np.isfinite(np.asarray([
            value
            for metrics in (train_metrics, validation_metrics)
            for value in metrics.values()
            if isinstance(value, (int, float))
        ], dtype=np.float64)).all()),
    }
    summary = {
        "schema_version": 1,
        "protocol": config["protocol"],
        "status": "complete",
        "decision": "offline_gate_pass" if all(checks.values()) else "offline_gate_fail",
        "gate_pass": bool(all(checks.values())),
        "device": str(best_agent.device),
        "gpu_name": (
            torch.cuda.get_device_name(best_agent.device)
            if str(best_agent.device).startswith("cuda")
            else None
        ),
        "best_update": best_update,
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "best_checkpoint_sha256": _sha256(best_path),
        "improved_checkpoints": improved_checkpoints,
        "critical_training_samples": int(np.sum(train_mask & critical_mask)),
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "teacher_manifest_sha256": manifest_sha,
        "teacher_manifest_sha256_list": manifest_hashes,
        "teacher_dataset_samples": int(len(train_indices) + np.sum(validation_mask)),
        "reverse_teacher_actions_available": int(np.sum(reverse_mask)),
        "reverse_teacher_actions_clipped": (
            0 if migration["performed"]
            else int(np.sum(arrays["reverse_requested"]))
        ),
        "action_space_migration_performed": migration["performed"],
        "source_action_lower": migration["source_lower"].tolist(),
        "source_action_upper": migration["source_upper"].tolist(),
        "target_action_lower": migration["target_lower"].tolist(),
        "target_action_upper": migration["target_upper"].tolist(),
        "base_action_scale": migration["base_scale"].tolist(),
        "base_action_offset": migration["base_offset"].tolist(),
        "checks": checks,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--additional-dataset-dir", type=Path, action="append", default=[]
    )
    args = parser.parse_args(argv)
    train(
        args.config,
        args.dataset_dir,
        args.output_dir,
        args.additional_dataset_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
