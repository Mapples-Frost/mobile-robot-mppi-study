#!/usr/bin/env python3
"""Run the frozen paired L281 component-separated anchor SAC Gate."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rl.run_l279_recovery_retention_anchor_sac import (
    evaluate as _l279_evaluate,
)
from mobile_robot_mppi.rl.recovery_retention import sha256_file


DEFAULT_CONFIG = ROOT / "configs/rl/l281_component_separated_anchor_gate.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l281_component_separated_anchor_sac_probe"


def _component_engineering(config):
    expected = config["training"]
    rows = []
    for run in config["runs"]:
        run_output = ROOT / run["output_dir"]
        checkpoint = run_output / "checkpoints/step_000006000.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        anchor = payload["resolved_config"]["rl"]["training"]["bc_anchor"]
        updates_path = run_output / "updates.csv"
        with updates_path.open("r", encoding="utf-8", newline="") as handle:
            import csv
            updates = list(csv.DictReader(handle))
        active = [
            row for row in updates
            if float(row.get("actor_update_applied", 0.0)) == 1.0
        ]
        fractions = [float(row["bc_anchor_recovery_fraction"]) for row in active]
        velocity = [float(row["bc_anchor_velocity_weight_mean"]) for row in active]
        angular = [float(row["bc_anchor_angular_weight_mean"]) for row in active]
        row = {
            "seed": int(run["seed"]),
            "active_update_records": len(active),
            "source_kind_balanced": bool(anchor.get("source_kind_balanced", False)),
            "recovery_action_mean_weights": anchor.get("recovery_action_mean_weights"),
            "source_action_mean_weights": anchor.get("source_action_mean_weights"),
            "recovery_fraction_exact": bool(
                fractions and np.allclose(fractions, 0.5, rtol=0.0, atol=1e-12)
            ),
            "velocity_weight_exact": bool(
                velocity and np.allclose(velocity, 0.5, rtol=0.0, atol=1e-12)
            ),
            "angular_weight_exact": bool(
                angular and np.allclose(angular, 1.0, rtol=0.0, atol=1e-12)
            ),
        }
        row["complete"] = bool(
            row["source_kind_balanced"]
            and row["active_update_records"]
            == int(expected["expected_anchor_update_steps"])
            and row["recovery_action_mean_weights"]
            == expected["recovery_action_mean_weights"]
            and row["source_action_mean_weights"]
            == expected["source_action_mean_weights"]
            and row["recovery_fraction_exact"]
            and row["velocity_weight_exact"]
            and row["angular_weight_exact"]
        )
        rows.append(row)
    return bool(all(row["complete"] for row in rows)), rows


def evaluate(config, output, device):
    summary = _l279_evaluate(config, output, device)
    component_complete, component_rows = _component_engineering(config)
    summary["protocol"] = "L281"
    summary["checks"]["component_separation_complete"] = component_complete
    summary["component_engineering"] = component_rows
    summary["gate_pass"] = bool(all(summary["checks"].values()))
    summary["decision"] = (
        "component_separated_anchor_gate_pass"
        if summary["gate_pass"]
        else "component_separated_anchor_gate_fail"
    )
    summary["larger_validation_preregistration_authorized"] = summary["gate_pass"]
    summary["final_map_evaluation_authorized"] = False
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def run(config_path, output, device="cuda", evaluate_only=False):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    for key in ("l277_summary", "l278_summary", "l279_summary", "l280_summary"):
        path = ROOT / config[key]["path"]
        if sha256_file(path) != config[key]["sha256"]:
            raise ValueError("L281 frozen summary SHA256 mismatch")
    l280 = json.loads(
        (ROOT / config["l280_summary"]["path"]).read_text(encoding="utf-8")
    )
    if l280["decision"] != "internal_anchor_conflict":
        raise ValueError("L281 requires the frozen L280 internal-conflict decision")
    manifest = ROOT / config["anchor_dataset"] / "manifest.json"
    if sha256_file(manifest) != config["anchor_dataset_manifest_sha256"]:
        raise ValueError("L281 anchor manifest SHA256 mismatch")
    for run_spec in config["runs"]:
        initialization = ROOT / run_spec["initialization"]
        if sha256_file(initialization) != run_spec["initialization_sha256"]:
            raise ValueError("L281 initialization SHA256 mismatch")
        if evaluate_only:
            continue
        run_output = ROOT / run_spec["output_dir"]
        if run_output.exists():
            raise FileExistsError("L281 formal output already exists")
        command = [
            sys.executable,
            "-u",
            str(ROOT / "experiments/rl/train_rl_sampling_prior.py"),
            "--config",
            str(ROOT / run_spec["config"]),
            "--initialize-agent-from",
            str(initialization),
            "--device",
            device,
            "--output-dir",
            str(run_output),
        ]
        print(json.dumps({
            "stage": "training_start",
            "seed": int(run_spec["seed"]),
            "device": device,
        }, sort_keys=True), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        print(json.dumps({
            "stage": "training_complete",
            "seed": int(run_spec["seed"]),
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
