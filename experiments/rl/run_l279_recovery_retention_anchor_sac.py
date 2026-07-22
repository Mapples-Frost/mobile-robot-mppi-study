#!/usr/bin/env python3
"""Run and evaluate the frozen paired L279 retention-anchor SAC probe."""

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rl.run_l277_recovery_initialized_sac_probe import (
    _gate as _validation_gate,
    _paired_metrics,
    _read_csv,
    _validate_run,
)
from experiments.rl.run_l278_recovery_retention_diagnosis import run as run_retention
from mobile_robot_mppi.rl.recovery_retention import (
    load_manifest, manifest_fingerprint, sha256_file,
)


DEFAULT_CONFIG = ROOT / "configs/rl/l279_recovery_retention_anchor_gate.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l279_recovery_retention_anchor_sac_probe"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _retention_config(config, output):
    checkpoints = []
    for run in config["runs"]:
        run_output = ROOT / run["output_dir"]
        stages = {
            "initial": run_output / "checkpoints/initial.pt",
            "step3000": run_output / "checkpoints/step_000003000.pt",
            "step6000": run_output / "checkpoints/step_000006000.pt",
        }
        checkpoints.append({
            "seed": int(run["seed"]),
            **{
                name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for name, path in stages.items()
            },
        })
    gate = config["gate"]
    return {
        "l277_summary": config["l277_summary"],
        "recovery_dataset": "results/research_platform/rl/l268_recovery_balanced_intervention/recovery_dataset",
        "split_counts": {"validation": 18, "test": 18},
        "checkpoints": checkpoints,
        "gate": {
            "maximum_median_relative_test_rmse_increase": gate["maximum_relative_test_rmse_increase"],
            "maximum_median_test_return_loss": gate["maximum_test_return_loss"],
            "maximum_median_test_reentry_loss": gate["maximum_test_reentry_loss"],
            "minimum_scenes_with_nonnegative_test_return_change": gate["minimum_test_scenes_nonnegative_return"],
            "maximum_collision_boundary_failure_increase": gate["maximum_collision_boundary_failure_increase"],
        },
    }


def _validation_results(config):
    run_results = []
    scene_values = defaultdict(list)
    rows_out = []
    for run in config["runs"]:
        rows, metrics, engineering = _validate_run(run, config["training"])
        run_results.append({
            "seed": int(run["seed"]), "metrics": metrics,
            "engineering": engineering,
        })
        for row in rows:
            rows_out.append({"seed": int(run["seed"]), **row})
        lookup = {
            (row["scene"], int(row["global_step"])): float(row["cross_track_rmse"])
            for row in rows
        }
        for scene in sorted({row["scene"] for row in rows}):
            scene_values[scene].append(
                lookup[(scene, 0)] - lookup[(scene, 6000)]
            )
    scene_improvements = {
        name: float(np.mean(values)) for name, values in sorted(scene_values.items())
    }
    checks, metrics = _validation_gate(run_results, scene_improvements, config["gate"])
    return run_results, rows_out, checks, metrics


def _anchor_engineering(config, fingerprint):
    checks = []
    rows = []
    for run in config["runs"]:
        checkpoint = ROOT / run["output_dir"] / "checkpoints/step_000006000.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["training_state"]
        agent = payload["agent"]
        resolved_anchor = payload["resolved_config"]["rl"]["training"]["bc_anchor"]
        row = {
            "seed": int(run["seed"]),
            "anchor_updates": int(agent.get("bc_anchor_update_steps", -1)),
            "manifest_fingerprint": state.get("bc_anchor_dataset_manifest_sha256"),
            "dataset_format": resolved_anchor.get("dataset_format"),
            "global_step": int(state.get("global_step", -1)),
        }
        row["complete"] = bool(
            row["anchor_updates"] == int(config["training"]["expected_anchor_update_steps"])
            and row["manifest_fingerprint"] == fingerprint
            and row["dataset_format"] == config["training"]["dataset_format"]
            and row["global_step"] == int(config["training"]["total_steps"])
        )
        checks.append(row["complete"])
        rows.append(row)
    return bool(all(checks)), rows


def _combined_gate(config, validation_checks, retention, anchor_complete):
    control = json.loads((ROOT / config["l278_summary"]["path"]).read_text(encoding="utf-8"))
    control_by_seed = {int(row["seed"]): row for row in control["seed_metrics"]}
    intervention_by_seed = {int(row["seed"]): row for row in retention["seed_metrics"]}
    improvements = []
    for seed in sorted(control_by_seed):
        control_ratio = 1.0 + float(control_by_seed[seed]["relative_test_rmse_increase_6k"])
        treatment_ratio = 1.0 + float(intervention_by_seed[seed]["relative_test_rmse_increase_6k"])
        improvements.append((control_ratio - treatment_ratio) / control_ratio)
    checks = {
        "anchor_engineering_complete": bool(anchor_complete),
        "recovery_retention_gate": bool(retention["retention_gate_pass"]),
        "validation_gate": bool(all(validation_checks.values())),
        "rmse_improvement_vs_l277": float(np.median(improvements)) >= float(
            config["gate"]["minimum_relative_test_rmse_improvement_vs_l277"]
        ),
    }
    return checks, {
        "paired_relative_test_rmse_improvement_vs_l277": improvements,
        "median_relative_test_rmse_improvement_vs_l277": float(np.median(improvements)),
    }


def evaluate(config, output, device):
    dataset = ROOT / config["anchor_dataset"]
    manifest = load_manifest(dataset)
    fingerprint = manifest_fingerprint(manifest)
    run_results, validation_rows, validation_checks, validation_metrics = _validation_results(config)
    retention_config = _retention_config(config, output)
    retention_config_path = output / "retention_config_resolved.yaml"
    retention_config_path.parent.mkdir(parents=True, exist_ok=True)
    retention_config_path.write_text(yaml.safe_dump(retention_config, sort_keys=False), encoding="utf-8")
    retention_output = output / "retention"
    retention = run_retention(retention_config_path, retention_output, device)
    anchor_complete, anchor_rows = _anchor_engineering(config, fingerprint)
    checks, comparative = _combined_gate(
        config, validation_checks, retention, anchor_complete
    )
    summary = {
        "protocol": "L279",
        "status": "complete",
        "gate_pass": bool(all(checks.values())),
        "decision": (
            "retention_anchor_gate_pass" if all(checks.values())
            else "retention_anchor_gate_fail"
        ),
        "checks": checks,
        "validation_checks": validation_checks,
        "validation_metrics": validation_metrics,
        "comparative_metrics": comparative,
        "retention_summary": retention,
        "anchor_engineering": anchor_rows,
        "runs": run_results,
        "larger_validation_preregistration_authorized": bool(all(checks.values())),
        "final_map_evaluation_authorized": False,
    }
    _write_csv(output / "validation_rows.csv", validation_rows)
    _write_csv(output / "anchor_engineering.csv", anchor_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def run(config_path, output, device="cuda", evaluate_only=False):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    for key in ("l277_summary", "l278_summary"):
        path = ROOT / config[key]["path"]
        if sha256_file(path) != config[key]["sha256"]:
            raise ValueError("L279 frozen summary SHA256 mismatch")
    if json.loads((ROOT / config["l278_summary"]["path"]).read_text(encoding="utf-8"))["decision"] != "recovery_retention_failed_anchor_probe_authorized":
        raise ValueError("L279 requires frozen negative L278 decision")
    dataset_manifest = ROOT / config["anchor_dataset"] / "manifest.json"
    if sha256_file(dataset_manifest) != config["anchor_dataset_manifest_sha256"]:
        raise ValueError("L279 anchor manifest SHA256 mismatch")
    for run_spec in config["runs"]:
        initialization = ROOT / run_spec["initialization"]
        if sha256_file(initialization) != run_spec["initialization_sha256"]:
            raise ValueError("L279 initialization SHA256 mismatch")
        if evaluate_only:
            continue
        run_output = ROOT / run_spec["output_dir"]
        if run_output.exists():
            raise FileExistsError("L279 formal output already exists")
        command = [
            sys.executable, "-u",
            str(ROOT / "experiments/rl/train_rl_sampling_prior.py"),
            "--config", str(ROOT / run_spec["config"]),
            "--initialize-agent-from", str(initialization),
            "--device", device,
            "--output-dir", str(run_output),
        ]
        print(json.dumps({
            "stage": "training_start", "seed": int(run_spec["seed"]),
            "device": device,
        }, sort_keys=True), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        print(json.dumps({
            "stage": "training_complete", "seed": int(run_spec["seed"]),
            "device": device,
        }, sort_keys=True), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    return evaluate(config, output, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    run(Path(args.config).resolve(), Path(args.output_dir).resolve(), args.device, args.evaluate_only)


if __name__ == "__main__":
    main()
