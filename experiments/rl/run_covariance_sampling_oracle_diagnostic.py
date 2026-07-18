#!/usr/bin/env python3
"""L83 oracle diagnostic for state-dependent MPPI covariance scales."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.rl.run_control_sequence_ranking_diagnostic import (
    _array_sha256,
    _state_vector,
    _true_trajectories,
)
from mobile_robot_mppi.core.config import config_hash, deep_merge, git_sha, load_yaml
from mobile_robot_mppi.runtime.factories import make_components


SCALES = (
    ("narrow", (0.50, 0.50)),
    ("baseline", (1.00, 1.00)),
    ("turn_explore", (0.75, 1.75)),
    ("speed_explore", (1.75, 0.75)),
    ("broad", (1.75, 1.75)),
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_anchor_steps(value):
    result = tuple(
        sorted(set(int(item.strip()) for item in value.split(",") if item.strip()))
    )
    if not result or result[0] < 0:
        raise ValueError("anchor steps must contain non-negative integers")
    return result


def _terminal_context(controller, state, target):
    config = controller.config
    action_spec = controller.action_spec
    state_spec = controller.state_spec
    heading_gate_active = bool(
        config.terminal_translation_heading_gate_rad is not None
        and target.phase in ("terminal_approach", "terminal")
        and "v_cmd" in action_spec.names
        and "theta" in state_spec.names
    )
    bearing_error = 0.0
    translation_scale = 1.0
    if heading_gate_active:
        theta = float(state[state_spec.index("theta")])
        dx = float(target.pose.x - state[state_spec.position_indices[0]])
        dy = float(target.pose.y - state[state_spec.position_indices[1]])
        if np.hypot(dx, dy) > 1e-12:
            desired = float(np.arctan2(dy, dx))
            bearing_error = float(np.arctan2(
                np.sin(desired - theta), np.cos(desired - theta)
            ))
            gate = float(config.terminal_translation_heading_gate_rad)
            gate_cosine = float(np.cos(gate))
            if abs(bearing_error) >= gate:
                translation_scale = 0.0
            else:
                translation_scale = max(
                    0.0,
                    (float(np.cos(bearing_error)) - gate_cosine)
                    / max(1.0 - gate_cosine, 1e-12),
                )
    return {
        "speed_limit_active": bool(
            config.terminal_translation_speed_limit is not None
            and target.phase in ("terminal_approach", "terminal")
            and "v_cmd" in action_spec.names
        ),
        "heading_gate_active": heading_gate_active,
        "bearing_error": bearing_error,
        "translation_scale": translation_scale,
        "alignment_active": bool(
            heading_gate_active
            and config.terminal_alignment_yaw_gain is not None
            and "omega_cmd" in action_spec.names
        ),
    }


def _apply_terminal_candidate_constraints(controller, candidates, context):
    values = np.asarray(candidates, dtype=np.float64).copy()
    if context["speed_limit_active"]:
        v_index = controller.action_spec.index("v_cmd")
        values[..., v_index] = np.minimum(
            values[..., v_index],
            controller.config.terminal_translation_speed_limit,
        )
    if context["heading_gate_active"]:
        v_index = controller.action_spec.index("v_cmd")
        values[:, 0, v_index] *= context["translation_scale"]
    return values


def _finalize_weighted_sequence(controller, sequence, preceding_action, context):
    values = np.clip(
        np.asarray(sequence, dtype=np.float64).copy(),
        controller.action_spec.lower,
        controller.action_spec.upper,
    )
    if context["speed_limit_active"]:
        v_index = controller.action_spec.index("v_cmd")
        values[:, v_index] = np.minimum(
            values[:, v_index],
            controller.config.terminal_translation_speed_limit,
        )
    if context["heading_gate_active"]:
        v_index = controller.action_spec.index("v_cmd")
        values[0, v_index] *= context["translation_scale"]
    if context["alignment_active"]:
        omega_index = controller.action_spec.index("omega_cmd")
        values[0, omega_index] = (
            controller.config.terminal_alignment_yaw_gain
            * context["bearing_error"]
        )
    values[0] = controller.action_spec.clip(
        values[0], preceding_action, controller.config.dt
    )
    if context["speed_limit_active"]:
        values[0, v_index] = min(
            values[0, v_index],
            controller.config.terminal_translation_speed_limit,
        )
    if context["heading_gate_active"]:
        values[0, v_index] *= context["translation_scale"]
    return values


def _summary(records):
    anchor_keys = sorted(set((row["scene"], row["anchor_step"]) for row in records))
    lookup = {
        (row["scene"], row["anchor_step"], row["scale_name"]): row
        for row in records
    }
    mean_cost_by_scale = {
        name: float(np.mean([
            lookup[(scene, anchor, name)]["true_weighted_cost"]
            for scene, anchor in anchor_keys
        ]))
        for name, _ in SCALES
    }
    best_fixed = min(mean_cost_by_scale, key=mean_cost_by_scale.get)
    best_counts = Counter()
    oracle_costs = []
    fixed_costs = []
    baseline_costs = []
    per_anchor = []
    for scene, anchor in anchor_keys:
        rows = [lookup[(scene, anchor, name)] for name, _ in SCALES]
        best = min(rows, key=lambda row: row["true_weighted_cost"])
        best_counts[best["scale_name"]] += 1
        oracle_cost = float(best["true_weighted_cost"])
        fixed_cost = float(
            lookup[(scene, anchor, best_fixed)]["true_weighted_cost"]
        )
        baseline_cost = float(
            lookup[(scene, anchor, "baseline")]["true_weighted_cost"]
        )
        oracle_costs.append(oracle_cost)
        fixed_costs.append(fixed_cost)
        baseline_costs.append(baseline_cost)
        per_anchor.append({
            "scene": scene,
            "anchor_step": anchor,
            "oracle_scale": best["scale_name"],
            "oracle_cost": oracle_cost,
            "best_fixed_cost": fixed_cost,
            "baseline_cost": baseline_cost,
            "context_improvement": fixed_cost - oracle_cost,
            "baseline_regret": baseline_cost - oracle_cost,
        })
    probabilities = np.asarray(
        [best_counts[name] / len(anchor_keys) for name, _ in SCALES],
        dtype=np.float64,
    )
    active = probabilities > 0.0
    entropy = float(-np.sum(probabilities[active] * np.log(probabilities[active])))
    return {
        "anchor_count": len(anchor_keys),
        "mean_cost_by_scale": mean_cost_by_scale,
        "best_fixed_scale": best_fixed,
        "best_scale_counts": {name: int(best_counts[name]) for name, _ in SCALES},
        "best_scale_entropy_nats": entropy,
        "context_oracle_improvement": float(
            np.mean(np.asarray(fixed_costs) - np.asarray(oracle_costs))
        ),
        "baseline_oracle_regret": float(
            np.mean(np.asarray(baseline_costs) - np.asarray(oracle_costs))
        ),
        "baseline_optimal_fraction": float(
            best_counts["baseline"] / len(anchor_keys)
        ),
        "per_anchor": per_anchor,
    }


def run(args):
    scene_path = Path(args.scene_config).resolve()
    checkpoint_path = Path(args.icode_checkpoint).resolve()
    if not scene_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("scene config or ICODE checkpoint is missing")
    config = load_yaml(scene_path)
    if config.get("scene", {}).get("obstacles"):
        raise ValueError("L83 dynamics diagnostic requires an obstacle-free scene")
    config = deep_merge(
        config,
        {
            "experiment": {"seed": int(args.seed)},
            "planner": {
                "prediction_mode": "icode_residual",
                "checkpoint": str(checkpoint_path),
                "sampling_prior": "goal_warm_start",
            },
            "memory": {"enable": False},
            "rl": {"enabled": False, "policy_backend": "none"},
        },
    )
    anchors = _parse_anchor_steps(args.anchor_steps)
    components = make_components(config, ROOT)
    plant = components["plant"]
    controller = components["controller"]
    reference = components["reference"]
    state_spec = components["state_spec"]
    action_spec = components["action_spec"]
    dt = float(config["experiment"]["control_dt"])
    base_sigma = np.asarray(controller.config.noise_sigma, dtype=np.float64)
    if action_spec.dimension != 2:
        raise ValueError("L83 frozen scale grid requires two control dimensions")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    arrays = {}
    rng = np.random.RandomState(int(args.seed) ^ 0xC0A4)
    initial = np.asarray(config["experiment"]["initial_state"], dtype=np.float64)
    truth = plant.reset(int(args.seed), initial)
    observation = components["sensors"].reset(truth, int(args.seed))
    controller.reset(seed=int(args.seed))
    reset_reference = getattr(reference, "reset", None)
    if callable(reset_reference):
        reset_reference()

    try:
        for step_index in range(anchors[-1] + 1):
            perceived = components["perception"].process(observation)
            if step_index in anchors:
                state = _state_vector(truth, state_spec)
                preceding_action = controller.previous_action.copy()
                target = reference.target_at(observation.timestamp, state)
                prior = controller._prior(perceived.observation, reference)
                center = np.asarray(prior.mean, dtype=np.float64)
                terminal_context = _terminal_context(controller, state, target)
                snapshot = plant.snapshot()
                standard_noise = rng.normal(
                    size=(int(args.candidate_count),) + center.shape
                )
                anchor_key = "anchor_%04d" % step_index
                arrays[anchor_key + "_center"] = center
                arrays[anchor_key + "_standard_noise"] = standard_noise

                for scale_name, scale_values in SCALES:
                    scale = np.asarray(scale_values, dtype=np.float64)
                    sigma = base_sigma * scale
                    covariance = np.diag(sigma ** 2)
                    candidates = np.clip(
                        center[None, :, :]
                        + standard_noise * sigma[None, None, :],
                        action_spec.lower[None, None, :],
                        action_spec.upper[None, None, :],
                    )
                    candidates[0] = center
                    candidates = _apply_terminal_candidate_constraints(
                        controller, candidates, terminal_context
                    )
                    evaluation = controller.evaluate_control_sequences(
                        state, candidates, target, ()
                    )
                    weighting = controller.importance_weights(
                        center, candidates, evaluation.costs, covariance
                    )
                    perturbations = candidates - center[None, :, :]
                    weighted_sequence = center + np.sum(
                        weighting.weights[:, None, None] * perturbations,
                        axis=0,
                    )
                    weighted_sequence = _finalize_weighted_sequence(
                        controller,
                        weighted_sequence,
                        preceding_action,
                        terminal_context,
                    )
                    true_paths, collisions = _true_trajectories(
                        plant,
                        snapshot,
                        weighted_sequence[None, :, :],
                        state_spec,
                        dt,
                    )
                    if np.any(collisions):
                        raise RuntimeError("L83 obstacle-free branch collided")
                    controller.previous_action = preceding_action.copy()
                    true_cost = float(controller.cost_trajectories(
                        true_paths,
                        weighted_sequence[None, :, :],
                        target,
                        (),
                    )[0])
                    arrays[anchor_key + "_" + scale_name + "_candidates"] = candidates
                    arrays[anchor_key + "_" + scale_name + "_predicted_costs"] = (
                        evaluation.costs
                    )
                    arrays[anchor_key + "_" + scale_name + "_weights"] = (
                        weighting.weights
                    )
                    arrays[anchor_key + "_" + scale_name + "_weighted_sequence"] = (
                        weighted_sequence
                    )
                    arrays[anchor_key + "_" + scale_name + "_true_trajectory"] = (
                        true_paths[0]
                    )
                    records.append({
                        "scene": str(config["scene"]["name"]),
                        "seed": int(args.seed),
                        "anchor_step": int(step_index),
                        "snapshot_time": float(snapshot.time),
                        "target_phase": str(target.phase),
                        "scale_name": scale_name,
                        "velocity_scale": float(scale[0]),
                        "yaw_scale": float(scale[1]),
                        "candidate_count": int(args.candidate_count),
                        "horizon": int(controller.config.horizon),
                        "candidate_sha256": _array_sha256(candidates),
                        "weighted_sequence_sha256": _array_sha256(weighted_sequence),
                        "predicted_cost_min": float(np.min(evaluation.costs)),
                        "predicted_cost_mean": float(np.mean(evaluation.costs)),
                        "effective_sample_size": weighting.effective_sample_size,
                        "true_weighted_cost": true_cost,
                    })
                plant.restore(snapshot)
                controller.previous_action = preceding_action.copy()

            plan = controller.plan(perceived.observation, reference)
            decision = components["safety"].arbitrate(
                plan.proposed_control, perceived.guard
            )
            controller.observe_safety_decision(decision)
            transition = plant.step(decision.executed_control, dt)
            truth = transition.ground_truth
            if truth.collision:
                raise RuntimeError("L83 live reference episode collided")
            observation = components["sensors"].observe(truth)
    finally:
        plant.close()

    with (output_dir / "scale_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(output_dir / "scale_arrays.npz", **arrays)
    summary = _summary(records)
    manifest = {
        "design_id": "l83_covariance_only_oracle_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "scene_config": str(scene_path),
        "scene_config_hash": config_hash(config),
        "icode_checkpoint": str(checkpoint_path),
        "icode_checkpoint_sha256": _sha256_file(checkpoint_path),
        "seed": int(args.seed),
        "anchor_steps": list(anchors),
        "candidate_count": int(args.candidate_count),
        "scale_grid": {name: list(values) for name, values in SCALES},
        "base_noise_sigma": base_sigma.tolist(),
        "memory_enabled": False,
        "rl_enabled": False,
        "counterfactual_scene_obstacle_count": 0,
        "record_count": len(records),
        "summary": summary,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--icode-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20268501)
    parser.add_argument("--anchor-steps", default="20,50")
    parser.add_argument("--candidate-count", type=int, default=64)
    args = parser.parse_args(argv)
    if args.candidate_count < 10:
        parser.error("candidate-count must be at least 10")
    run(args)


if __name__ == "__main__":
    main()
