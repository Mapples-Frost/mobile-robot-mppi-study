"""Train the bootstrap three-head maneuver proposal Actor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mobile_robot_mppi.core.config import config_hash, git_sha
from mobile_robot_mppi.rl.maneuver_actor import (
    ManeuverProposalActor,
    best_of_m_loss,
)


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/research/complex_supervised_maneuver_actor_training_v1.yaml"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value.get("protocol") != "complex_supervised_maneuver_actor_training_v1":
        raise ValueError("maneuver Actor training protocol mismatch")
    frozen = value["frozen_contract"]
    if not bool(frozen["actor_input_causal_only"]):
        raise ValueError("Actor input must remain causal")
    if bool(frozen["future_truth_in_student_input"]):
        raise ValueError("future truth cannot enter Actor inputs")
    for name in ("risk_changed", "icode_changed", "mppi_changed",
                 "safety_changed", "maps_changed",
                 "formal_server_launch_authorized"):
        if bool(frozen[name]):
            raise ValueError("training changed frozen contract: %s" % name)
    return value


def _load_dataset(protocol):
    directory = (ROOT / protocol["dataset"]["directory"]).resolve()
    path = directory / protocol["dataset"]["file"]
    if _sha256(path) != protocol["dataset"]["expected_sha256"]:
        raise ValueError("training dataset hash mismatch")
    gate = json.loads(
        (directory / protocol["dataset"]["gate"]).read_text(encoding="utf-8")
    )
    if gate["status"] != protocol["dataset"]["gate_expected_status"]:
        raise ValueError("dataset Gate did not pass")
    if (
        bool(protocol["dataset"]["require_engineering_training_authorized"])
        and not bool(gate["engineering_training_authorized"])
    ):
        raise ValueError("dataset did not authorize engineering training")
    data = np.load(path)
    train_scenarios = set(data["scenario_id"][data["split"] == "train"])
    validation_scenarios = set(
        data["scenario_id"][data["split"] == "validation"]
    )
    if train_scenarios & validation_scenarios:
        raise ValueError("scenario leakage between training and validation")
    return path, data


def _normalizer(data, train_mask):
    keys = np.asarray([
        "%s:%d" % (scenario, anchor)
        for scenario, anchor in zip(
            data["scenario_id"][train_mask],
            data["anchor_index"][train_mask],
        )
    ])
    _, first = np.unique(keys, return_index=True)
    values = data["observations"][train_mask][np.sort(first)].astype(np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.maximum(std, 1.0e-3)
    return mean.astype(np.float32), std.astype(np.float32), int(values.shape[0])


def _metrics(model, observations, targets, families):
    model.eval()
    with torch.no_grad():
        predictions = model(observations)
        _, details = best_of_m_loss(predictions, targets)
        assignments = details["assignments"]
        batch = torch.arange(targets.shape[0], device=targets.device)
        selected = predictions[batch, assignments]
        best_rmse = torch.sqrt(torch.mean((selected - targets).square()))
        first = min(12, targets.shape[1])
        first_rmse = torch.sqrt(torch.mean(
            (selected[:, :first] - targets[:, :first]).square()
        ))
    selected_np = selected.cpu().numpy()
    direction_correct = []
    for index, family in enumerate(families):
        if family not in {"left", "right"}:
            continue
        yaw = float(np.mean(selected_np[index, :first, 1]))
        direction_correct.append(
            yaw > 0.0 if family == "left" else yaw < 0.0
        )
    return {
        "best_head_rmse": float(best_rmse),
        "first_segment_rmse": float(first_rmse),
        "left_right_direction_consistency": (
            1.0 if not direction_correct
            else float(np.mean(direction_correct))
        ),
        "assigned_head_counts": np.bincount(
            assignments.cpu().numpy(), minlength=model.heads
        ).tolist(),
    }


def train(protocol_path, output):
    protocol_path = Path(protocol_path).resolve()
    protocol = _load_protocol(protocol_path)
    dataset_path, data = _load_dataset(protocol)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("training output already contains evidence")
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "protocol_resolved.yaml"
    resolved.write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    cfg = protocol["training"]
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg["torch_num_threads"]))
    device = torch.device(str(cfg["device"]))

    train_mask = data["split"] == "train"
    validation_mask = data["split"] == "validation"
    mean, std, normalizer_count = _normalizer(data, train_mask)
    x_train = torch.as_tensor(
        (data["observations"][train_mask] - mean) / std,
        dtype=torch.float32, device=device,
    )
    y_train = torch.as_tensor(
        data["normalized_teacher_controls"][train_mask],
        dtype=torch.float32, device=device,
    )
    x_validation = torch.as_tensor(
        (data["observations"][validation_mask] - mean) / std,
        dtype=torch.float32, device=device,
    )
    y_validation = torch.as_tensor(
        data["normalized_teacher_controls"][validation_mask],
        dtype=torch.float32, device=device,
    )
    model_cfg = protocol["model"]
    model = ManeuverProposalActor(
        x_train.shape[1],
        heads=model_cfg["heads"],
        horizon=model_cfg["horizon"],
        action_dim=model_cfg["action_dim"],
        hidden_sizes=model_cfg["hidden_sizes"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    batch_size = min(int(cfg["batch_size"]), x_train.shape[0])
    checkpoint = output / "best.pt"

    for epoch in range(int(cfg["epochs"])):
        model.train()
        order = rng.permutation(x_train.shape[0])
        train_losses = []
        for start in range(0, len(order), batch_size):
            indices = torch.as_tensor(
                order[start:start + batch_size],
                dtype=torch.long, device=device,
            )
            predictions = model(x_train[indices])
            loss, _ = best_of_m_loss(
                predictions,
                y_train[indices],
                diversity_margin=float(cfg["diversity_margin"]),
                diversity_weight=float(cfg["diversity_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["gradient_clip_norm"])
            )
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_predictions = model(x_validation)
            validation_loss, _ = best_of_m_loss(
                validation_predictions,
                y_validation,
                diversity_margin=float(cfg["diversity_margin"]),
                diversity_weight=float(cfg["diversity_weight"]),
            )
        value = float(validation_loss)
        if value < best_loss - 1.0e-7:
            best_loss = value
            best_epoch = epoch
            stale = 0
            torch.save({
                "schema_version": 1,
                "model": model.state_dict(),
                "model_config": dict(model_cfg),
                "observation_dim": int(x_train.shape[1]),
                "observation_mean": mean,
                "observation_std": std,
                "dataset_sha256": _sha256(dataset_path),
                "student_input_causal_only": True,
                "execution_authority": "proposal_only",
                "epoch": int(epoch),
            }, checkpoint)
        else:
            stale += 1
        if epoch % 25 == 0 or stale == 0:
            history.append({
                "epoch": int(epoch),
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": value,
            })
        if stale >= int(cfg["early_stopping_patience"]):
            break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    train_metrics = _metrics(
        model, x_train, y_train, data["behavior_family"][train_mask]
    )
    validation_metrics = _metrics(
        model,
        x_validation,
        y_validation,
        data["behavior_family"][validation_mask],
    )
    gate = protocol["gate_a"]
    checks = {
        "validation_best_head_rmse": (
            validation_metrics["best_head_rmse"]
            <= float(gate["maximum_validation_best_head_rmse"])
        ),
        "validation_first_segment_rmse": (
            validation_metrics["first_segment_rmse"]
            <= float(gate["maximum_validation_first_segment_rmse"])
        ),
        "left_right_direction_consistency": (
            validation_metrics["left_right_direction_consistency"]
            >= float(gate["minimum_left_right_direction_consistency"])
        ),
        "all_heads_used_on_training": (
            all(value > 0 for value in train_metrics["assigned_head_counts"])
            if gate["require_all_heads_used_on_training"] else True
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    result = {
        "status": status,
        "gate": "A_imitation_only",
        "gate_b_c_still_required": True,
        "closed_loop_gate_d_authorized": False,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "normalizer_unique_train_state_count": int(normalizer_count),
        "train": train_metrics,
        "validation": validation_metrics,
        "checks": checks,
    }
    _write_json(output / "training_result.json", result)
    _write_json(output / "training_history.json", history)
    _write_json(output / "artifact_manifest.json", {
        "git_sha": git_sha(ROOT),
        "protocol_config_hash": config_hash(protocol),
        "protocol_sha256": _sha256(resolved),
        "dataset_sha256": _sha256(dataset_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "result_sha256": _sha256(output / "training_result.json"),
        "student_input_contains_future_truth": False,
        "execution_authority": "proposal_only",
        "risk_icode_mppi_safety_changed": False,
        "formal_server_launch_authorized": False,
    })
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if status == "pass" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    protocol = _load_protocol(args.protocol.resolve())
    _load_dataset(protocol)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_pass",
            "heads": int(protocol["model"]["heads"]),
            "horizon": int(protocol["model"]["horizon"]),
            "student_input_causal_only": True,
            "execution_authority": "proposal_only",
            "formal_server_launch_authorized": False,
        }, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required unless --dry-run is used")
    return train(args.protocol, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
