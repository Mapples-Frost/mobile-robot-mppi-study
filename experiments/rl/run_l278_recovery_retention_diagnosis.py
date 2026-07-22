#!/usr/bin/env python3
"""Evaluate recovery-behavior retention across frozen L277 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rl.run_l263_counterfactual_actor_diagnosis import _load_agent
from experiments.rl.run_l269_horizon_diagnostic import _resolved_environment
from experiments.rl.run_l276_recovery_sequence_imitation_initialization import (
    _load_chains,
    _rollout_actor,
    _teacher_metrics,
)
from mobile_robot_mppi.core.config import load_yaml


DEFAULT_CONFIG = ROOT / "configs/rl/l278_recovery_retention_diagnosis.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l278_recovery_retention_diagnosis"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gate(seed_metrics, rollout_rows, gate):
    relative_rmse = [row["relative_test_rmse_increase_6k"] for row in seed_metrics]
    return_losses = [row["median_test_return_loss_6k"] for row in seed_metrics]
    reentry_losses = [row["test_reentry_loss_6k"] for row in seed_metrics]
    by_scene = defaultdict(list)
    for row in rollout_rows:
        if row["split"] == "test" and row["stage"] == "step6000":
            by_scene[row["scene"]].append(float(row["return_change_vs_initial"]))
    scene_changes = {
        scene: float(np.mean(values)) for scene, values in sorted(by_scene.items())
    }
    initial_failures = sum(
        int(row["collision"] or row["boundary_violation"])
        for row in rollout_rows if row["stage"] == "initial"
    )
    final_failures = sum(
        int(row["collision"] or row["boundary_violation"])
        for row in rollout_rows if row["stage"] == "step6000"
    )
    finite_values = []
    for row in seed_metrics:
        finite_values.extend(float(value) for key, value in row.items() if key != "seed")
    finite_values.extend(float(value) for value in scene_changes.values())
    checks = {
        "test_rmse_retained": float(np.median(relative_rmse))
        <= float(gate["maximum_median_relative_test_rmse_increase"]),
        "test_return_retained": float(np.median(return_losses))
        <= float(gate["maximum_median_test_return_loss"]),
        "test_reentry_retained": float(np.median(reentry_losses))
        <= float(gate["maximum_median_test_reentry_loss"]),
        "scene_return_retained": sum(value >= 0.0 for value in scene_changes.values())
        >= int(gate["minimum_scenes_with_nonnegative_test_return_change"]),
        "collision_boundary_nonincrease": (
            final_failures - initial_failures
            <= int(gate["maximum_collision_boundary_failure_increase"])
        ),
        "all_finite": bool(np.isfinite(np.asarray(finite_values)).all()),
    }
    metrics = {
        "median_relative_test_rmse_increase_6k": float(np.median(relative_rmse)),
        "median_test_return_loss_6k": float(np.median(return_losses)),
        "median_test_reentry_loss_6k": float(np.median(reentry_losses)),
        "test_scene_return_changes_6k": scene_changes,
        "test_scenes_nonnegative_6k": int(sum(value >= 0.0 for value in scene_changes.values())),
        "initial_collision_boundary_failures": int(initial_failures),
        "step6000_collision_boundary_failures": int(final_failures),
    }
    return checks, metrics


def run(config_path: Path, output: Path, device: str):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary_path = ROOT / config["l277_summary"]["path"]
    if _sha256(summary_path) != config["l277_summary"]["sha256"]:
        raise ValueError("L278 L277 summary SHA256 mismatch")
    if json.loads(summary_path.read_text(encoding="utf-8"))["gate_pass"]:
        raise ValueError("L278 is frozen for the negative L277 Gate")
    dataset = ROOT / config["recovery_dataset"]
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    chains = _load_chains(dataset, manifest)
    splits = {
        split: [row for row in chains if row["split"] == split]
        for split in ("validation", "test")
    }
    if {name: len(rows) for name, rows in splits.items()} != config["split_counts"]:
        raise ValueError("L278 recovery split count mismatch")
    output.mkdir(parents=True, exist_ok=False)
    base = load_yaml(ROOT / "configs/rl/l262_coverage_gated_value_6k.yaml")
    rollout_rows = []
    teacher_rows = []
    progress = []
    for run_spec in config["checkpoints"]:
        seed = int(run_spec["seed"])
        stage_rollouts = {}
        for stage in ("initial", "step3000", "step6000"):
            checkpoint = ROOT / run_spec[stage]["path"]
            if _sha256(checkpoint) != run_spec[stage]["sha256"]:
                raise ValueError("L278 checkpoint SHA256 mismatch")
            _, agent, normalizer = _load_agent(checkpoint, device)
            for split, split_chains in splits.items():
                teacher = _teacher_metrics(agent, normalizer, split_chains)
                teacher_rows.append({
                    "seed": seed,
                    "stage": stage,
                    "split": split,
                    **teacher,
                })
                by_scene = defaultdict(list)
                for chain in split_chains:
                    by_scene[chain["scene_config"]].append(chain)
                for scene_config, scene_chains in sorted(by_scene.items()):
                    maximum = max(len(row["actions"]) for row in scene_chains)
                    environment = _resolved_environment(
                        base, scene_config, maximum, scene_chains[0]["seed"], "l268"
                    )
                    try:
                        for chain in scene_chains:
                            result = _rollout_actor(
                                environment, chain, agent, normalizer, 1e-6
                            )
                            key = (split, int(chain["chain_id"]))
                            stage_rollouts[(stage, *key)] = result
                    finally:
                        environment.close()
            progress.append({"seed": seed, "stage": stage, "device": device})
            _write_csv(output / "progress.csv", progress)
            print(json.dumps({
                "seed": seed,
                "stage": stage,
                "completed_stage_count": len(progress),
                "expected_stage_count": 9,
                "device": device,
            }, sort_keys=True), flush=True)
        for split, split_chains in splits.items():
            for chain in split_chains:
                key = (split, int(chain["chain_id"]))
                initial = stage_rollouts[("initial", *key)]
                for stage in ("initial", "step3000", "step6000"):
                    result = stage_rollouts[(stage, *key)]
                    rollout_rows.append({
                        "seed": seed,
                        "stage": stage,
                        "split": split,
                        "chain_id": int(chain["chain_id"]),
                        "scene": chain["scene"],
                        "severity": chain["severity"],
                        "side": int(chain["side"]),
                        **result,
                        "return_change_vs_initial": float(
                            result["discounted_return"] - initial["discounted_return"]
                        ),
                        "reentry_change_vs_initial": float(
                            int(result["corridor_reentry"])
                            - int(initial["corridor_reentry"])
                        ),
                    })
    seed_metrics = []
    for run_spec in config["checkpoints"]:
        seed = int(run_spec["seed"])
        teacher = {
            (row["stage"], row["split"]): row
            for row in teacher_rows if row["seed"] == seed
        }
        seed_rollouts = [row for row in rollout_rows if row["seed"] == seed]
        def stage_split(stage, split):
            return [row for row in seed_rollouts if row["stage"] == stage and row["split"] == split]
        initial_test = stage_split("initial", "test")
        row = {"seed": seed}
        for stage, label in (("step3000", "3k"), ("step6000", "6k")):
            current = stage_split(stage, "test")
            row["relative_test_rmse_increase_%s" % label] = float(
                teacher[(stage, "test")]["rmse"] / teacher[("initial", "test")]["rmse"] - 1.0
            )
            row["median_test_return_loss_%s" % label] = float(np.median([
                base_row["discounted_return"] - current_row["discounted_return"]
                for base_row, current_row in zip(initial_test, current)
            ]))
            row["test_reentry_loss_%s" % label] = float(
                np.mean([float(item["corridor_reentry"]) for item in initial_test])
                - np.mean([float(item["corridor_reentry"]) for item in current])
            )
        seed_metrics.append(row)
    checks, metrics = _gate(seed_metrics, rollout_rows, config["gate"])
    summary = {
        "protocol": "L278",
        "status": "complete",
        "retention_gate_pass": bool(all(checks.values())),
        "decision": (
            "recovery_retained" if all(checks.values())
            else "recovery_retention_failed_anchor_probe_authorized"
        ),
        "checks": checks,
        "metrics": metrics,
        "seed_metrics": seed_metrics,
        "actor_training_authorized": False,
        "final_map_evaluation_authorized": False,
    }
    _write_csv(output / "teacher_metrics.csv", teacher_rows)
    _write_csv(output / "rollouts.csv", rollout_rows)
    _write_csv(output / "seed_metrics.csv", seed_metrics)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(Path(args.config).resolve(), Path(args.output_dir).resolve(), args.device)


if __name__ == "__main__":
    main()
