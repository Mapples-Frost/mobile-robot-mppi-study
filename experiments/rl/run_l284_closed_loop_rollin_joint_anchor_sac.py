#!/usr/bin/env python3
"""Run the frozen paired L284 closed-loop roll-in joint-anchor SAC Gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rl.run_l279_recovery_retention_anchor_sac import (
    evaluate as _l279_evaluate,
)
from experiments.rl.run_l281_component_separated_anchor_sac import (
    _component_engineering,
)
from mobile_robot_mppi.rl.recovery_retention import sha256_file


DEFAULT_CONFIG = ROOT / "configs/rl/l284_closed_loop_rollin_joint_anchor_gate.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_platform/rl/l284_closed_loop_rollin_joint_anchor_sac_probe"


def _paired_recovery_return_gain(l281, treatment):
    baseline = {
        int(row["seed"]): row
        for row in l281["retention_summary"]["seed_metrics"]
    }
    current = {
        int(row["seed"]): row
        for row in treatment["retention_summary"]["seed_metrics"]
    }
    if set(baseline) != set(current):
        raise ValueError("L284 paired recovery seed set drifted")
    gains = [
        float(baseline[seed]["median_test_return_loss_6k"])
        - float(current[seed]["median_test_return_loss_6k"])
        for seed in sorted(baseline)
    ]
    return gains, float(np.median(gains))


def evaluate(config, output, device):
    summary = _l279_evaluate(config, output, device)
    component_complete, component_rows = _component_engineering(config)
    l281 = json.loads(
        (ROOT / config["l281_summary"]["path"]).read_text(encoding="utf-8")
    )
    gains, median_gain = _paired_recovery_return_gain(l281, summary)
    summary["protocol"] = "L284"
    summary["checks"]["component_separation_complete"] = component_complete
    summary["checks"]["median_recovery_return_gain_vs_l281"] = bool(
        median_gain > float(config["gate"]["minimum_median_recovery_return_gain_vs_l281"])
    )
    summary["component_engineering"] = component_rows
    summary["comparative_metrics"]["paired_recovery_return_gain_vs_l281"] = gains
    summary["comparative_metrics"]["median_recovery_return_gain_vs_l281"] = median_gain
    summary["gate_pass"] = bool(all(summary["checks"].values()))
    summary["decision"] = (
        "closed_loop_rollin_joint_anchor_gate_pass"
        if summary["gate_pass"]
        else "closed_loop_rollin_joint_anchor_gate_fail"
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
    for key in ("l283_summary", "l281_summary", "l277_summary", "l278_summary"):
        path = ROOT / config[key]["path"]
        if sha256_file(path) != config[key]["sha256"]:
            raise ValueError("L284 frozen summary SHA256 mismatch")
    l283 = json.loads(
        (ROOT / config["l283_summary"]["path"]).read_text(encoding="utf-8")
    )
    if l283["decision"] != "coupled_sequence_bottleneck":
        raise ValueError("L284 requires the frozen coupled L283 decision")
    manifest = ROOT / config["anchor_dataset"] / "manifest.json"
    if sha256_file(manifest) != config["anchor_dataset_manifest_sha256"]:
        raise ValueError("L284 anchor manifest SHA256 mismatch")
    for run_spec in config["runs"]:
        initialization = ROOT / run_spec["initialization"]
        if sha256_file(initialization) != run_spec["initialization_sha256"]:
            raise ValueError("L284 initialization SHA256 mismatch")
        if evaluate_only:
            continue
        run_output = ROOT / run_spec["output_dir"]
        if run_output.exists():
            raise FileExistsError("L284 formal output already exists")
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
