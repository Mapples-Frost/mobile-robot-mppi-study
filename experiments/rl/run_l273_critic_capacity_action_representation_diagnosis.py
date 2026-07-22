#!/usr/bin/env python3
"""Run the frozen L273 Critic function-class oracle-fit diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.rl.run_l263_counterfactual_actor_diagnosis import _spearman
from mobile_robot_mppi.core.config import git_sha
from mobile_robot_mppi.rl.sac import QNetwork, SACConfig, quantile_huber_loss


PROTOCOL = "L273"
DEFAULT_CONFIG = ROOT / "configs/rl/l273_critic_capacity_action_representation_diagnosis.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l273_critic_capacity_action_representation_diagnosis"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> str:
    actual = _sha256(path)
    if actual != str(expected).lower():
        raise ValueError("SHA256 mismatch for %s: %s" % (path, actual))
    return actual


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows):
    rows = list(rows)
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_config(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = payload["l273"]
    if section["protocol"] != PROTOCOL or section["actor_training_authorized"] is not False:
        raise ValueError("L273 must remain diagnostic-only")
    references = json.dumps(
        {key: value for key, value in section.items() if key != "forbidden_tokens"},
        sort_keys=True,
    ).lower()
    entered = [
        token for token in section["forbidden_tokens"] if token.lower() in references
    ]
    if entered:
        raise ValueError("forbidden artifact entered L273 config: %s" % entered)
    return payload, section


def _action_features(actions, encoding):
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("actions must be finite [batch, 2]")
    if encoding == "raw":
        return values.copy()
    if encoding != "polynomial_degree3":
        raise ValueError("unknown action encoding: %s" % encoding)
    v, omega = values[:, 0], values[:, 1]
    return np.stack(
        (v, omega, v * v, omega * omega, v * omega, v ** 3, omega ** 3,
         v * v * omega, v * omega * omega),
        axis=1,
    ).astype(np.float32)


def _standardize_by_state(rows, floor):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state_id"]].append(row)
    output = []
    for state_id, state_rows in sorted(grouped.items()):
        returns = np.asarray([row["return"] for row in state_rows], dtype=np.float64)
        scale = max(float(returns.std()), float(floor))
        center = float(returns.mean())
        for row in state_rows:
            output.append({
                **row,
                "target": float((row["return"] - center) / scale),
            })
    return output


def _load_datasets(section):
    oracle = section["oracle_fit"]
    state_path = ROOT / oracle["l263_states"]
    return_path = ROOT / oracle["l263_returns"]
    action_path = ROOT / oracle["l263_actions"]
    _verify(state_path, oracle["l263_states_sha256"])
    _verify(return_path, oracle["l263_returns_sha256"])
    _verify(action_path, oracle["l263_actions_sha256"])
    states = json.loads(state_path.read_text(encoding="utf-8"))
    state_by_id = {row["state_id"]: row for row in states}
    action_by_key = {
        (row["state_id"], row["action_id"]): row for row in _read_csv(action_path)
    }
    returns = [
        row for row in _read_csv(return_path)
        if row["continuation"] == oracle["continuation"]
        and int(row["horizon"]) == int(oracle["horizon"])
    ]
    rows = []
    for row in returns:
        state = state_by_id[row["state_id"]]
        action = action_by_key[(row["state_id"], row["action_id"])]
        rows.append({
            "state_id": row["state_id"],
            "action_id": row["action_id"],
            "scene": state["scene"],
            "role": state["scene_role"],
            "observation": np.asarray(state["normalized_observation"], dtype=np.float32),
            "action": np.asarray(
                [float(action["normalized_v"]), float(action["normalized_omega"])],
                dtype=np.float32,
            ),
            "return": float(row["discounted_return"]),
        })
    counts = defaultdict(int)
    for row in rows:
        counts[row["state_id"]] += 1
    if set(counts.values()) != {int(oracle["expected_actions_per_state"])}:
        raise ValueError("L263 action count per state is invalid")
    train_states = {row["state_id"] for row in rows if row["role"] == oracle["training_role"]}
    validation_states = {
        row["state_id"] for row in rows if row["role"] == oracle["validation_role"]
    }
    if len(train_states) != int(oracle["expected_train_states"]):
        raise ValueError("L273 L263 train state count mismatch")
    if len(validation_states) != int(oracle["expected_validation_states"]):
        raise ValueError("L273 L263 validation state count mismatch")
    rows = _standardize_by_state(rows, oracle["target_std_floor"])
    train = [row for row in rows if row["role"] == oracle["training_role"]]
    validation = [row for row in rows if row["role"] == oracle["validation_role"]]

    external = section["external_evaluation"]
    external_states_path = ROOT / external["states"]
    external_returns_path = ROOT / external["returns"]
    _verify(external_states_path, external["states_sha256"])
    _verify(external_returns_path, external["returns_sha256"])
    external_states = json.loads(external_states_path.read_text(encoding="utf-8"))
    external_state_by_id = {row["state_id"]: row for row in external_states}
    external_rows = []
    for row in _read_csv(external_returns_path):
        state = external_state_by_id[row["state_id"]]
        external_rows.append({
            "state_id": row["state_id"],
            "action_id": row["action_id"],
            "scene": row["scene"],
            "role": "external",
            "observation": np.asarray(state["normalized_observation"], dtype=np.float32),
            "action": np.asarray(
                [float(row["normalized_v"]), float(row["normalized_omega"])],
                dtype=np.float32,
            ),
            "return": float(row["discounted_return"]),
        })
    external_counts = defaultdict(int)
    for row in external_rows:
        external_counts[row["state_id"]] += 1
    if len(external_counts) != int(external["expected_states"]):
        raise ValueError("L273 external state count mismatch")
    if set(external_counts.values()) != {int(external["expected_actions_per_state"])}:
        raise ValueError("L273 external action count mismatch")
    external_rows = _standardize_by_state(
        external_rows, oracle["target_std_floor"]
    )
    for collection in (train, validation, external_rows):
        if not all(
            row["observation"].shape == (69,)
            and row["action"].shape == (2,)
            and np.isfinite(row["observation"]).all()
            and np.isfinite(row["action"]).all()
            and np.isfinite(row["return"])
            and np.isfinite(row["target"])
            for row in collection
        ):
            raise FloatingPointError("L273 dataset contains invalid rows")
    return train, validation, external_rows


class StateActionSampler:
    def __init__(self, rows, seed):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["state_id"]].append(row)
        self.groups = [grouped[key] for key in sorted(grouped)]
        self.rng = np.random.RandomState(int(seed))

    def sample(self, count):
        selected = []
        for state_index in self.rng.randint(0, len(self.groups), size=int(count)):
            group = self.groups[int(state_index)]
            selected.append(group[int(self.rng.randint(0, len(group)))])
        return selected


def _sac_config(source, hidden_sizes):
    return SACConfig(
        hidden_sizes=tuple(int(value) for value in hidden_sizes),
        activation=source["activation"],
        critic_distribution=source["critic_distribution"],
        critic_num_quantiles=int(source["critic_num_quantiles"]),
        critic_quantile_huber_kappa=float(source["critic_quantile_huber_kappa"]),
    )


def _batch(rows, encoding, device):
    observations = torch.as_tensor(
        np.stack([row["observation"] for row in rows]),
        dtype=torch.float32, device=device,
    )
    actions = torch.as_tensor(
        _action_features(np.stack([row["action"] for row in rows]), encoding),
        dtype=torch.float32, device=device,
    )
    targets = torch.as_tensor(
        np.asarray([row["target"] for row in rows], dtype=np.float32)[:, None],
        dtype=torch.float32, device=device,
    )
    return observations, actions, targets


def _predict(model, rows, encoding, device):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), 1024):
            chunk = rows[start:start + 1024]
            observations, actions, _ = _batch(chunk, encoding, device)
            values = model(observations, actions).mean(dim=-1).cpu().numpy()
            for row, value in zip(chunk, values):
                output.append({**row, "prediction": float(value)})
    return output


def _metrics(predictions, external_pair=None):
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["state_id"]].append(row)
    state_rows, scene_pairs = [], defaultdict(list)
    for state_id, rows in sorted(grouped.items()):
        truth = np.asarray([row["return"] for row in rows], dtype=np.float64)
        predicted = np.asarray([row["prediction"] for row in rows], dtype=np.float64)
        top1 = int(np.argmax(truth) == np.argmax(predicted))
        record = {
            "state_id": state_id,
            "scene": rows[0]["scene"],
            "spearman": float(_spearman(predicted, truth)),
            "top1": top1,
            "target_mse": float(np.mean(
                (predicted - np.asarray([row["target"] for row in rows])) ** 2
            )),
        }
        if external_pair is not None:
            by_action = {row["action_id"]: row for row in rows}
            recovery, forward = external_pair
            true_better = by_action[recovery]["return"] > by_action[forward]["return"]
            predicted_better = (
                by_action[recovery]["prediction"] > by_action[forward]["prediction"]
            )
            record["pair_correct"] = int(true_better == predicted_better)
            scene_pairs[record["scene"]].append(record["pair_correct"])
        state_rows.append(record)
    summary = {
        "mean_state_spearman": float(np.mean([row["spearman"] for row in state_rows])),
        "top1_agreement": float(np.mean([row["top1"] for row in state_rows])),
        "target_mse": float(np.mean([row["target_mse"] for row in state_rows])),
    }
    if external_pair is not None:
        summary["recovery_forward_accuracy"] = float(np.mean([
            row["pair_correct"] for row in state_rows
        ]))
        summary["scene_pair_accuracy"] = {
            scene: float(np.mean(values)) for scene, values in sorted(scene_pairs.items())
        }
    return state_rows, summary


def _paired_arm_check(results, treatment, decision):
    by_seed_arm = {(row["seed"], row["arm"]): row for row in results}
    seeds = sorted({row["seed"] for row in results})
    gains = {"train": [], "validation": [], "external": []}
    consistent = 0
    for seed in seeds:
        baseline = by_seed_arm[(seed, "baseline_raw")]
        candidate = by_seed_arm[(seed, treatment)]
        values = {
            "train": candidate["train_mean_state_spearman"] - baseline["train_mean_state_spearman"],
            "validation": candidate["validation_mean_state_spearman"] - baseline["validation_mean_state_spearman"],
            "external": candidate["external_recovery_forward_accuracy"] - baseline["external_recovery_forward_accuracy"],
        }
        for key, value in values.items():
            gains[key].append(float(value))
        consistent += int(all(value > 0.0 for value in values.values()))
    scenes = sorted(by_seed_arm[(seeds[0], "baseline_raw")]["external_scene_pair_accuracy"])
    scene_changes = {}
    for scene in scenes:
        candidate = np.mean([
            by_seed_arm[(seed, treatment)]["external_scene_pair_accuracy"][scene]
            for seed in seeds
        ])
        baseline = np.mean([
            by_seed_arm[(seed, "baseline_raw")]["external_scene_pair_accuracy"][scene]
            for seed in seeds
        ])
        scene_changes[scene] = float(candidate - baseline)
    medians = {key: float(np.median(values)) for key, values in gains.items()}
    scenes_improved = sum(value > 0.0 for value in scene_changes.values())
    passed = bool(
        medians["train"] >= float(decision["minimum_paired_median_train_spearman_gain"])
        and medians["validation"] >= float(decision["minimum_paired_median_validation_spearman_gain"])
        and medians["external"] >= float(decision["minimum_paired_median_external_pair_gain"])
        and consistent >= int(decision["minimum_consistent_seed_blocks"])
        and scenes_improved >= int(decision["minimum_external_scenes_improved"])
        and min(scene_changes.values()) >= -float(decision["maximum_external_scene_pair_decrease"])
    )
    return {
        "arm": treatment,
        "gate_pass": passed,
        "paired_median_train_spearman_gain": medians["train"],
        "paired_median_validation_spearman_gain": medians["validation"],
        "paired_median_external_pair_gain": medians["external"],
        "consistent_seed_blocks": consistent,
        "external_scenes_improved": scenes_improved,
        "minimum_external_scene_change": min(scene_changes.values()),
        "external_scene_changes": scene_changes,
    }


def _decision(results, config):
    thresholds = config["decision"]
    baseline = [row for row in results if row["arm"] == "baseline_raw"]
    sufficient_blocks = sum(
        row["train_mean_state_spearman"] >= float(thresholds["minimum_train_spearman_for_sufficiency"])
        and row["validation_mean_state_spearman"] >= float(thresholds["minimum_validation_spearman_for_sufficiency"])
        and row["external_recovery_forward_accuracy"] >= float(thresholds["minimum_external_pair_accuracy_for_sufficiency"])
        for row in baseline
    )
    comparisons = {
        arm: _paired_arm_check(results, arm, thresholds)
        for arm in (
            "baseline_polynomial", "capacity_wide_raw", "capacity_wide_polynomial"
        )
    }
    if sufficient_blocks >= int(thresholds["minimum_consistent_seed_blocks"]):
        status = "bellman_learning_not_function_class"
        selected = "baseline_raw"
    elif comparisons["baseline_polynomial"]["gate_pass"]:
        status = "action_representation_limited"
        selected = "baseline_polynomial"
    elif comparisons["capacity_wide_raw"]["gate_pass"]:
        status = "capacity_limited"
        selected = "capacity_wide_raw"
    elif comparisons["capacity_wide_polynomial"]["gate_pass"]:
        status = "combined_capacity_action_limited"
        selected = "capacity_wide_polynomial"
    else:
        status = "observation_or_target_generalization_limited"
        selected = None
    return status, selected, sufficient_blocks, list(comparisons.values())


def _save_checkpoint(path, model, optimizer, sampler, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": PROTOCOL,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler_rng_state": sampler.rng.get_state(),
        "metadata": metadata,
    }, path)
    return _sha256(path)


def run(config_path: Path, output: Path):
    payload, section = _load_config(config_path)
    l272_summary = ROOT / section["l272"]["summary"]
    _verify(l272_summary, section["l272"]["summary_sha256"])
    l272 = json.loads(l272_summary.read_text(encoding="utf-8"))
    if l272["decision"] != section["l272"]["required_decision"]:
        raise ValueError("L273 requires the frozen negative L272 decision")
    train, validation, external = _load_datasets(section)
    design = section["design"]
    device = torch.device(design["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("L273 requires CUDA")
    output.mkdir(parents=True, exist_ok=False)
    _json_dump(output / "arm_order.json", {
        str(seed): [
            design["arms"][index]["name"] for index in
            np.random.RandomState(int(design["arm_order_seed"]) + int(seed)).permutation(
                len(design["arms"])
            )
        ] for seed in design["paired_seeds"]
    })
    arm_by_name = {row["name"]: row for row in design["arms"]}
    arm_orders = json.loads((output / "arm_order.json").read_text(encoding="utf-8"))
    results, progress, prediction_rows, state_metric_rows = [], [], [], []
    for seed in design["paired_seeds"]:
        for arm_name in arm_orders[str(seed)]:
            arm = arm_by_name[arm_name]
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            encoding = arm["action_encoding"]
            action_features = 2 if encoding == "raw" else 9
            sac_config = _sac_config(section["source_architecture"], arm["hidden_sizes"])
            model = QNetwork(69, action_features, sac_config).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=float(design["learning_rate"])
            )
            sampler = StateActionSampler(train, int(seed))
            last_loss = None
            model.train()
            for update in range(1, int(design["updates"]) + 1):
                batch_rows = sampler.sample(int(design["batch_size"]))
                observations, actions, targets = _batch(batch_rows, encoding, device)
                predictions = model(observations, actions)
                loss = quantile_huber_loss(
                    predictions, targets,
                    float(section["source_architecture"]["critic_quantile_huber_kappa"]),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(design["gradient_clip_norm"])
                )
                optimizer.step()
                last_loss = float(loss.detach().cpu())
                if not np.isfinite(last_loss):
                    raise FloatingPointError("L273 training loss is non-finite")
                if update in design["checkpoint_updates"]:
                    checkpoint = output / ("seed_%d" % seed) / arm_name / "checkpoints" / (
                        "update_%06d.pt" % update
                    )
                    digest = _save_checkpoint(
                        checkpoint, model, optimizer, sampler,
                        {"seed": seed, "arm": arm_name, "updates": update,
                         "config_sha256": _sha256(config_path)},
                    )
                    progress.append({
                        "seed": seed, "arm": arm_name, "completed_updates": update,
                        "checkpoint_sha256": digest, "finite": True,
                    })
                    _write_csv(output / "progress.csv", progress)
            split_summaries = {}
            for split, rows in (
                ("train", train), ("validation", validation), ("external", external)
            ):
                predictions = _predict(model, rows, encoding, device)
                state_rows, summary = _metrics(
                    predictions,
                    tuple(section["external_evaluation"]["primary_pair"])
                    if split == "external" else None,
                )
                split_summaries[split] = summary
                for row in predictions:
                    prediction_rows.append({
                        "seed": seed, "arm": arm_name, "split": split,
                        "state_id": row["state_id"], "action_id": row["action_id"],
                        "scene": row["scene"], "return": row["return"],
                        "standardized_target": row["target"],
                        "prediction": row["prediction"],
                    })
                state_metric_rows.extend({
                    "seed": seed, "arm": arm_name, "split": split, **row
                } for row in state_rows)
            results.append({
                "seed": seed, "arm": arm_name, "finite": True,
                "parameter_count": sum(value.numel() for value in model.parameters()),
                "final_training_loss": last_loss,
                "train_mean_state_spearman": split_summaries["train"]["mean_state_spearman"],
                "train_top1_agreement": split_summaries["train"]["top1_agreement"],
                "train_target_mse": split_summaries["train"]["target_mse"],
                "validation_mean_state_spearman": split_summaries["validation"]["mean_state_spearman"],
                "validation_top1_agreement": split_summaries["validation"]["top1_agreement"],
                "validation_target_mse": split_summaries["validation"]["target_mse"],
                "external_mean_state_spearman": split_summaries["external"]["mean_state_spearman"],
                "external_top1_agreement": split_summaries["external"]["top1_agreement"],
                "external_recovery_forward_accuracy": split_summaries["external"]["recovery_forward_accuracy"],
                "external_scene_pair_accuracy": split_summaries["external"]["scene_pair_accuracy"],
            })
            print(json.dumps({"completed_seed": seed, "completed_arm": arm_name}))
            del model, optimizer
            torch.cuda.empty_cache()
    expected = len(design["paired_seeds"]) * len(design["arms"])
    if len(results) != expected or not all(row["finite"] for row in results):
        raise RuntimeError("L273 incomplete result matrix")
    status, selected, sufficient_blocks, comparisons = _decision(results, section)
    flat_results = []
    for row in results:
        flat_results.append({
            **{key: value for key, value in row.items() if key != "external_scene_pair_accuracy"},
            "external_scene_pair_accuracy": json.dumps(
                row["external_scene_pair_accuracy"], sort_keys=True
            ),
        })
    _write_csv(output / "results.csv", flat_results)
    _write_csv(output / "predictions.csv", prediction_rows)
    _write_csv(output / "metrics_by_state.csv", state_metric_rows)
    _write_csv(output / "comparisons.csv", [
        {**{key: value for key, value in row.items() if key not in (
            "external_scene_changes",
        )}, "external_scene_changes": json.dumps(row["external_scene_changes"], sort_keys=True)}
        for row in comparisons
    ])
    summary = {
        "protocol": PROTOCOL,
        "status": "complete",
        "decision": status,
        "selected_diagnostic_arm": selected,
        "actor_training_authorized": False,
        "git_sha": git_sha(ROOT),
        "config_sha256": _sha256(config_path),
        "paired_seed_count": len(design["paired_seeds"]),
        "arm_count": len(design["arms"]),
        "l263_train_states": len({row["state_id"] for row in train}),
        "l263_validation_states": len({row["state_id"] for row in validation}),
        "external_states": len({row["state_id"] for row in external}),
        "baseline_sufficient_seed_blocks": sufficient_blocks,
        "comparisons": comparisons,
        "all_finite": True,
    }
    _json_dump(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
