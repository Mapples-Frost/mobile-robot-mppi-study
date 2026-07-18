#!/usr/bin/env python3
"""Calibrate L52 from causal one-step innovations on frozen L51 trajectories."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.core.spaces import state_spec_from_config
from mobile_robot_mppi.learning.models import (
    InnovationGatedResidualDynamics,
    PlatformResidualDynamics,
)
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction
from mobile_robot_mppi.planning.mppi import integrate_batch


def _resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to write an empty calibration table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class _CombinedDynamics:
    def __init__(self, nominal, residual):
        self.nominal = nominal
        self.residual = residual
        self.state_dim = int(nominal.state_dim)
        self.control_dim = int(nominal.control_dim)

    def derivative(self, state, control, time=None):
        return np.asarray(
            self.nominal.derivative(state, control, time), dtype=np.float64
        ) + np.asarray(
            self.residual.derivative(state, control, time), dtype=np.float64
        )


class _ErrorReplayResidual:
    state_dim = 5
    control_dim = 2

    def derivative(self, state, control, time=None):  # pragma: no cover - not used
        del state, control, time
        return np.zeros(self.state_dim, dtype=np.float64)


def _candidate_grid(calibration):
    grid = calibration["parameter_grid"]
    names = (
        "forgetting_factor", "minimum_samples", "confidence_z",
        "off_threshold", "on_threshold",
    )
    values = [list(grid[name]) for name in names]
    return [dict(zip(names, items)) for items in itertools.product(*values)]


def _scene_config_map(design):
    result = {}
    for item in design["scenes"]:
        path = item["path"] if isinstance(item, dict) else item
        config = load_yaml(_resolved(path))
        name = str(config.get("scene", {}).get("name", Path(path).stem))
        result[name] = config
    return result


def _innovation_sequences(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    calibration = design["reliability_calibration"]
    source = str(calibration["source_condition"])
    state_names = tuple(str(value) for value in calibration["state_names"])
    state_scales = np.asarray(calibration["state_scales"], dtype=np.float64)
    scenes = _scene_config_map(design)
    domains = {str(item["name"]): item for item in design["physics_domains"]}
    expected_seeds = {int(value) for value in design["development_episode_seeds"]}
    sealed = {int(value) for value in design["sealed_confirmation_episode_seeds"]}
    sequences = []
    audit = {
        "sealed_seeds_used": [], "metadata_sealed_seeds_used": [],
        "protected_seeds_used": [], "missing_episode_keys": [],
        "noncontiguous_step_keys": [],
    }

    for block_index, block in enumerate(design["model_blocks"]):
        block_dir = Path(input_dir) / ("block_%d" % block_index)
        metadata = json.loads(
            (block_dir / "metadata.json").read_text(encoding="utf-8")
        )
        audit["metadata_sealed_seeds_used"].extend(
            int(value) for value in metadata["sealed_confirmation_seeds_used"]
        )
        audit["protected_seeds_used"].extend(
            int(value) for value in metadata["previous_protected_seeds_used"]
        )
        checkpoint = _resolved(block["icode_checkpoint"])
        residual = PlatformResidualDynamics.from_checkpoint(checkpoint, device="cpu")
        rows = [
            row for row in _read_csv(block_dir / "factorial_steps.csv")
            if str(row["condition"]) == source
        ]
        grouped = defaultdict(list)
        for row in rows:
            seed = int(row["episode_seed"])
            if seed in sealed:
                audit["sealed_seeds_used"].append(seed)
            grouped[(str(row["scene"]), str(row["physics_domain"]), seed)].append(row)

        expected = {
            (scene, domain, seed)
            for scene in scenes for domain in domains for seed in expected_seeds
        }
        audit["missing_episode_keys"].extend(
            [(block_index,) + key for key in sorted(expected - set(grouped))]
        )
        for key in sorted(expected & set(grouped)):
            scene_name, domain_name, episode_seed = key
            scene_config = scenes[scene_name]
            domain = domains[domain_name]
            state_spec = state_spec_from_config(scene_config["state_space"])
            indices = [state_spec.index(name) for name in state_names]
            if residual.state_dim != state_spec.dimension:
                raise ValueError("checkpoint and scene state dimensions differ")
            dt = float(scene_config["experiment"]["control_dt"])
            delay = float(
                domain.get("plant_override", {}).get("actuator", {}).get(
                    "command_delay",
                    scene_config["plant"].get("actuator", {}).get("command_delay", 0.0),
                )
            )
            delay_fraction = delay / dt
            if not 0.0 <= delay_fraction <= 1.0:
                raise ValueError("L52 calibration supports command_delay in [0, dt]")
            nominal = DynamicUnicyclePrediction(
                scene_config["plant"].get("nominal_velocity_time_constant", 0.18),
                scene_config["plant"].get("nominal_yaw_time_constant", 0.12),
            )
            combined = _CombinedDynamics(nominal, residual)
            previous_state = np.asarray(
                scene_config["experiment"]["initial_state"], dtype=np.float64
            )
            previous_executed = np.zeros(2, dtype=np.float64)
            episode_rows = sorted(grouped[key], key=lambda row: int(row["step"]))
            observed_steps = [int(row["step"]) for row in episode_rows]
            if observed_steps != list(range(len(episode_rows))):
                audit["noncontiguous_step_keys"].append((block_index,) + key)
                continue
            errors = []
            for row in episode_rows:
                current = np.asarray(
                    [row["x"], row["y"], row["theta"], row["v"], row["omega"]],
                    dtype=np.float64,
                )
                executed = np.asarray(
                    [row["executed_v"], row["executed_omega"]], dtype=np.float64
                )
                effective = (
                    delay_fraction * previous_executed
                    + (1.0 - delay_fraction) * executed
                )
                nominal_prediction = integrate_batch(
                    nominal, previous_state, effective, dt, state_spec, "rk4"
                )
                residual_prediction = integrate_batch(
                    combined, previous_state, effective, dt, state_spec, "rk4"
                )
                errors.append((
                    state_spec.error(nominal_prediction, current),
                    state_spec.error(residual_prediction, current),
                ))
                previous_state = current
                previous_executed = executed
            sequences.append({
                "model_block": block_index,
                "scene": scene_name,
                "physics_domain": domain_name,
                "episode_seed": episode_seed,
                "state_indices": indices,
                "state_scales": state_scales.copy(),
                "context_value": delay_fraction,
                "errors": errors,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            })
    return sequences, audit


def _evaluate_candidate(candidate, sequences, calibration, candidate_id):
    domain_values = defaultdict(list)
    active_values = defaultdict(list)
    cold_start_violations = 0
    active_threshold = float(calibration["active_alpha_threshold"])
    delay_context = dict(calibration.get("actuation_delay_context", {}))
    for sequence in sequences:
        gate = InnovationGatedResidualDynamics(
            _ErrorReplayResidual(),
            state_indices=sequence["state_indices"],
            state_scales=sequence["state_scales"],
            forgetting_factor=float(candidate["forgetting_factor"]),
            minimum_samples=int(candidate["minimum_samples"]),
            confidence_z=float(candidate["confidence_z"]),
            off_threshold=float(candidate["off_threshold"]),
            on_threshold=float(candidate["on_threshold"]),
            rise_rate=float(calibration["rise_rate"]),
            fall_rate=float(calibration["fall_rate"]),
            context_value=(
                sequence.get("context_value")
                if bool(delay_context.get("enabled", False)) else None
            ),
            context_off_threshold=float(
                delay_context.get("off_threshold", 0.5)
            ),
            context_on_threshold=float(
                delay_context.get("on_threshold", 0.8)
            ),
        )
        usable_alphas = []
        for index, (nominal_error, residual_error) in enumerate(sequence["errors"]):
            alpha = gate.observe_prediction_errors(nominal_error, residual_error)
            if index + 1 < int(candidate["minimum_samples"]) and alpha != 0.0:
                cold_start_violations += 1
            # The final transition cannot affect another action in this episode.
            if index + 1 < len(sequence["errors"]):
                usable_alphas.append(alpha)
        key = (sequence["model_block"], sequence["physics_domain"])
        domain_values[key].extend(usable_alphas)
        active_values[key].extend(alpha >= active_threshold for alpha in usable_alphas)

    matched = str(calibration["matched_domain"])
    beneficial = str(calibration["beneficial_domain"])
    blocks = sorted({int(sequence["model_block"]) for sequence in sequences})
    per_block = []
    for block in blocks:
        matched_alpha = float(np.mean(domain_values[(block, matched)]))
        beneficial_alpha = float(np.mean(domain_values[(block, beneficial)]))
        matched_active = float(np.mean(active_values[(block, matched)]))
        beneficial_active = float(np.mean(active_values[(block, beneficial)]))
        per_block.append({
            "model_block": block,
            "matched_mean_alpha": matched_alpha,
            "beneficial_mean_alpha": beneficial_alpha,
            "alpha_separation": beneficial_alpha - matched_alpha,
            "matched_active_fraction": matched_active,
            "beneficial_active_fraction": beneficial_active,
        })
    result = dict(candidate)
    result.update({
        "candidate_id": int(candidate_id),
        "worst_beneficial_mean_alpha": min(row["beneficial_mean_alpha"] for row in per_block),
        "worst_matched_mean_alpha": max(row["matched_mean_alpha"] for row in per_block),
        "worst_block_alpha_separation": min(row["alpha_separation"] for row in per_block),
        "worst_beneficial_active_fraction": min(row["beneficial_active_fraction"] for row in per_block),
        "worst_matched_active_fraction": max(row["matched_active_fraction"] for row in per_block),
        "cold_start_violations": int(cold_start_violations),
        "per_model_block": per_block,
    })
    gate = calibration["selection_gate"]
    result["passed"] = bool(
        result["worst_beneficial_mean_alpha"]
        >= float(gate["minimum_beneficial_domain_mean_alpha"])
        and result["worst_matched_mean_alpha"]
        <= float(gate["maximum_matched_domain_mean_alpha"])
        and result["worst_block_alpha_separation"]
        >= float(gate["minimum_worst_model_block_alpha_separation"])
        and result["worst_beneficial_active_fraction"]
        >= float(gate["minimum_beneficial_domain_active_fraction"])
        and result["worst_matched_active_fraction"]
        <= float(gate["maximum_matched_domain_active_fraction"])
        and result["cold_start_violations"] == 0
    )
    return result


def calibrate(config, input_dir):
    design = config["rl"]["cross_layer_factorial"]
    calibration = design["reliability_calibration"]
    sequences, audit = _innovation_sequences(config, input_dir)
    expected_sequences = (
        len(design["model_blocks"]) * len(design["scenes"])
        * len(design["physics_domains"])
        * len(design["development_episode_seeds"])
    )
    candidates = [
        _evaluate_candidate(candidate, sequences, calibration, index)
        for index, candidate in enumerate(_candidate_grid(calibration))
    ]
    passing = [row for row in candidates if row["passed"]]
    passing.sort(key=lambda row: (
        -row["worst_block_alpha_separation"],
        row["worst_matched_mean_alpha"],
        -row["worst_beneficial_mean_alpha"],
        row["candidate_id"],
    ))
    integrity = bool(
        len(sequences) == expected_sequences
        and not audit["sealed_seeds_used"]
        and not audit["metadata_sealed_seeds_used"]
        and not audit["protected_seeds_used"]
        and not audit["missing_episode_keys"]
        and not audit["noncontiguous_step_keys"]
        and all(len(sequence["errors"]) >= 2 for sequence in sequences)
    )
    selected = passing[0] if integrity and passing else None
    frozen = {
        "enabled": bool(selected is not None),
        "state_names": list(calibration["state_names"]),
        "state_scales": list(calibration["state_scales"]),
        "rise_rate": float(calibration["rise_rate"]),
        "fall_rate": float(calibration["fall_rate"]),
    }
    if bool(calibration.get("actuation_delay_context", {}).get("enabled", False)):
        frozen["actuation_delay_context"] = dict(
            calibration["actuation_delay_context"]
        )
    if selected is not None:
        for name in (
            "forgetting_factor", "minimum_samples", "confidence_z",
            "off_threshold", "on_threshold",
        ):
            frozen[name] = selected[name]
    return candidates, {
        "design_id": str(design["design_id"]),
        "artifact_integrity": integrity,
        "expected_sequences": expected_sequences,
        "observed_sequences": len(sequences),
        "candidate_count": len(candidates),
        "passing_candidate_count": len(passing),
        "selection_passed": bool(selected is not None),
        "selected_candidate": selected,
        "best_diagnostic_candidate": sorted(
            candidates,
            key=lambda row: (-row["worst_block_alpha_separation"], row["candidate_id"]),
        )[0],
        "frozen_reliability_gate": frozen,
        "audit": audit,
        "source_git_sha": git_sha(ROOT),
        "interpretation_guard": (
            "Physics-domain labels were used only for offline hyperparameter "
            "selection; the runtime gate consumes completed transitions only."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved(args.config))
    output = _resolved(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates, summary = calibrate(config, _resolved(args.input_dir))
    table_rows = []
    for candidate in candidates:
        row = {key: value for key, value in candidate.items() if key != "per_model_block"}
        table_rows.append(row)
    _write_csv(output / "reliability_calibration_candidates.csv", table_rows)
    (output / "reliability_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "frozen_reliability_gate.yaml").write_text(
        yaml.safe_dump(
            {"planner": {"residual_reliability_gate": summary["frozen_reliability_gate"]}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["selection_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
