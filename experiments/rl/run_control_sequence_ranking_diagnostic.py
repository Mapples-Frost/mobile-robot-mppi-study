#!/usr/bin/env python3
"""Paired control-sequence ranking diagnostic for nominal, MLP and ICODE.

The script branches the true MuJoCo plant from exact snapshots.  It is an
offline mechanism diagnostic: branch commands are bounded but intentionally
bypass the online safety arbiter in obstacle-free scenes so model error is not
confounded with perception or arbitration.
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mobile_robot_mppi.core.config import config_hash, deep_merge, git_sha, load_yaml
from mobile_robot_mppi.core.references import ReferenceTarget
from mobile_robot_mppi.core.types import ControlCommand, Pose2D
from mobile_robot_mppi.runtime.factories import make_components


def _state_vector(truth, state_spec):
    values = {
        "x": truth.pose.x,
        "y": truth.pose.y,
        "theta": truth.pose.theta,
        "v": truth.twist.v,
        "omega": truth.twist.omega,
        "wheel_left": truth.wheel_speeds[0],
        "wheel_right": truth.wheel_speeds[1],
    }
    return np.asarray([values[name] for name in state_spec.names], dtype=np.float64)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(predicted, observed):
    first = _rankdata(predicted)
    second = _rankdata(observed)
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return 0.0 if denominator <= 1e-15 else float(np.dot(first, second) / denominator)


def _mppi_weights(costs, temperature):
    costs = np.asarray(costs, dtype=np.float64)
    exponent = np.clip(
        -(costs - float(np.min(costs))) / float(temperature), -700.0, 0.0
    )
    weights = np.exp(exponent)
    return weights / float(np.sum(weights))


def _jensen_shannon(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    mixture = 0.5 * (first + second)

    def divergence(p, q):
        active = p > 0.0
        return float(np.sum(p[active] * np.log(p[active] / q[active])))

    return 0.5 * divergence(first, mixture) + 0.5 * divergence(second, mixture)


def _ranking_metrics(predicted, observed, temperature, elite_fraction=0.10):
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    count = predicted.size
    elite_count = max(1, int(np.ceil(float(elite_fraction) * count)))
    predicted_order = np.argsort(predicted, kind="mergesort")
    observed_order = np.argsort(observed, kind="mergesort")
    predicted_elite = set(int(value) for value in predicted_order[:elite_count])
    observed_elite = set(int(value) for value in observed_order[:elite_count])
    iqr = float(np.quantile(observed, 0.75) - np.quantile(observed, 0.25))
    predicted_best = int(predicted_order[0])
    observed_best = int(observed_order[0])
    return {
        "spearman": _spearman(predicted, observed),
        "elite_recall": float(len(predicted_elite & observed_elite) / elite_count),
        "normalized_regret": float(
            (observed[predicted_best] - observed[observed_best]) / (iqr + 1e-9)
        ),
        "top1_hit": int(predicted_best == observed_best),
        "top5_hit": int(predicted_best in set(int(v) for v in observed_order[:5])),
        "weight_js_divergence": _jensen_shannon(
            _mppi_weights(predicted, temperature),
            _mppi_weights(observed, temperature),
        ),
        "predicted_best_index": predicted_best,
        "true_best_index": observed_best,
        "elite_count": elite_count,
    }


def _controller_for(base_config, mode, checkpoint):
    planner_override = {
        "prediction_mode": mode,
        "sampling_prior": "goal_warm_start",
    }
    if checkpoint is not None:
        planner_override["checkpoint"] = str(Path(checkpoint).resolve())
    config = deep_merge(
        base_config,
        {
            "planner": planner_override,
            "memory": {"enable": False},
            "rl": {"enabled": False, "policy_backend": "none"},
        },
    )
    components = make_components(config, ROOT)
    components["controller"].reset(seed=int(config["experiment"].get("seed", 0)))
    components["plant"].close()
    return components["controller"]


def _target_from_plan(plan):
    diagnostics = plan.diagnostics
    return ReferenceTarget(
        pose=Pose2D(
            float(diagnostics["target_x"]),
            float(diagnostics["target_y"]),
            float(diagnostics["target_theta"]),
        ),
        reference_id=str(diagnostics["reference_id"]),
        is_terminal=bool(diagnostics["target_is_terminal"]),
        phase=str(diagnostics["target_phase"]),
    )


def _true_trajectories(plant, snapshot, candidates, state_spec, dt):
    count, horizon, _ = candidates.shape
    trajectories = np.empty(
        (count, horizon + 1, state_spec.dimension), dtype=np.float64
    )
    collisions = np.zeros(count, dtype=np.int64)
    for candidate_index in range(count):
        truth = plant.restore(snapshot)
        trajectories[candidate_index, 0] = _state_vector(truth, state_spec)
        for step_index in range(horizon):
            command = ControlCommand(
                candidates[candidate_index, step_index],
                plant.time,
                "l82_counterfactual",
            )
            transition = plant.step(command, dt)
            truth = transition.ground_truth
            trajectories[candidate_index, step_index + 1] = _state_vector(
                truth, state_spec
            )
            collisions[candidate_index] |= int(truth.collision)
    plant.restore(snapshot)
    return trajectories, collisions


def _parse_anchor_steps(value):
    steps = tuple(sorted(set(int(item.strip()) for item in value.split(",") if item.strip())))
    if not steps or steps[0] < 0:
        raise ValueError("anchor steps must contain non-negative integers")
    return steps


def run(args):
    scene_path = Path(args.scene_config).resolve()
    icode_path = Path(args.icode_checkpoint).resolve()
    mlp_path = Path(args.mlp_checkpoint).resolve()
    for path in (scene_path, icode_path, mlp_path):
        if not path.exists():
            raise FileNotFoundError(path)
    base_config = load_yaml(scene_path)
    if base_config.get("scene", {}).get("obstacles"):
        raise ValueError("L82 dynamics diagnostic requires an obstacle-free scene")
    if int(base_config["planner"]["horizon"]) <= 0:
        raise ValueError("planner horizon must be positive")
    base_config = deep_merge(
        base_config,
        {"experiment": {"seed": int(args.seed)}, "memory": {"enable": False}},
    )
    anchor_steps = _parse_anchor_steps(args.anchor_steps)
    if anchor_steps[-1] >= int(base_config["experiment"]["max_steps"]):
        raise ValueError("anchor step exceeds experiment max_steps")

    live = make_components(base_config, ROOT)
    plant = live["plant"]
    controller = live["controller"]
    state_spec = live["state_spec"]
    action_spec = live["action_spec"]
    reference = live["reference"]
    nominal = _controller_for(base_config, "nominal", None)
    mlp = _controller_for(base_config, "mlp_residual", mlp_path)
    icode = _controller_for(base_config, "icode_residual", icode_path)
    model_controllers = {"nominal": nominal, "mlp": mlp, "icode": icode}

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    arrays = {}
    rng = np.random.RandomState(int(args.seed) ^ 0x1C0DE)
    dt = float(base_config["experiment"]["control_dt"])
    initial = np.asarray(base_config["experiment"]["initial_state"], dtype=np.float64)
    truth = plant.reset(int(args.seed), initial)
    observation = live["sensors"].reset(truth, int(args.seed))
    controller.reset(seed=int(args.seed))
    reset_reference = getattr(reference, "reset", None)
    if callable(reset_reference):
        reset_reference()

    try:
        for step_index in range(anchor_steps[-1] + 1):
            perceived = live["perception"].process(observation)
            preceding_action = controller.previous_action.copy()
            plan = controller.plan(perceived.observation, reference)

            if step_index in anchor_steps:
                snapshot = plant.snapshot()
                center = np.asarray(plan.control_sequence, dtype=np.float64)
                noise = rng.normal(
                    size=(int(args.candidate_count),) + center.shape
                )
                noise *= (
                    float(args.candidate_noise_scale)
                    * np.asarray(controller.config.noise_sigma)[None, None, :]
                )
                candidates = np.clip(
                    center[None, :, :] + noise,
                    action_spec.lower[None, None, :],
                    action_spec.upper[None, None, :],
                )
                candidates[0] = center
                target = _target_from_plan(plan)
                state = _state_vector(truth, state_spec)
                candidate_hash = _array_sha256(candidates)

                true_paths, collisions = _true_trajectories(
                    plant, snapshot, candidates, state_spec, dt
                )
                if np.any(collisions):
                    raise RuntimeError(
                        "counterfactual collision in obstacle-free L82 branch"
                    )
                if not np.isfinite(true_paths).all():
                    raise FloatingPointError("true branch produced NaN or Inf")

                anchor_key = "anchor_%04d" % step_index
                arrays[anchor_key + "_candidates"] = candidates
                arrays[anchor_key + "_true_trajectories"] = true_paths
                true_costs = None
                predicted_costs = {}
                for model_name, model_controller in model_controllers.items():
                    model_controller.previous_action = preceding_action.copy()
                    evaluation = model_controller.evaluate_control_sequences(
                        state, candidates, target, ()
                    )
                    predicted_costs[model_name] = evaluation.costs
                    arrays[anchor_key + "_" + model_name + "_trajectories"] = (
                        evaluation.trajectories
                    )
                    arrays[anchor_key + "_" + model_name + "_costs"] = (
                        evaluation.costs
                    )
                    if model_name == "nominal":
                        true_costs = model_controller.cost_trajectories(
                            true_paths, candidates, target, ()
                        )
                arrays[anchor_key + "_true_costs"] = true_costs

                for model_name in ("nominal", "mlp", "icode"):
                    metrics = _ranking_metrics(
                        predicted_costs[model_name],
                        true_costs,
                        controller.config.temperature,
                    )
                    records.append({
                        "scene": str(base_config["scene"]["name"]),
                        "seed": int(args.seed),
                        "anchor_step": int(step_index),
                        "snapshot_time": float(snapshot.time),
                        "model": model_name,
                        "candidate_count": int(args.candidate_count),
                        "horizon": int(controller.config.horizon),
                        "candidate_sha256": candidate_hash,
                        "true_cost_min": float(np.min(true_costs)),
                        "true_cost_iqr": float(
                            np.quantile(true_costs, 0.75)
                            - np.quantile(true_costs, 0.25)
                        ),
                        **metrics,
                    })

            decision = live["safety"].arbitrate(
                plan.proposed_control, perceived.guard
            )
            controller.observe_safety_decision(decision)
            transition = plant.step(decision.executed_control, dt)
            truth = transition.ground_truth
            if truth.collision:
                raise RuntimeError("live reference episode collided")
            observation = live["sensors"].observe(truth)
    finally:
        plant.close()

    with (output_dir / "ranking_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(output_dir / "ranking_arrays.npz", **arrays)

    by_model = {}
    for model_name in ("nominal", "mlp", "icode"):
        subset = [row for row in records if row["model"] == model_name]
        by_model[model_name] = {
            metric: float(np.mean([row[metric] for row in subset]))
            for metric in (
                "spearman",
                "elite_recall",
                "normalized_regret",
                "top1_hit",
                "top5_hit",
                "weight_js_divergence",
            )
        }
    summary = {
        "nominal": by_model["nominal"],
        "mlp": by_model["mlp"],
        "icode": by_model["icode"],
        "icode_minus_mlp": {
            "spearman": by_model["icode"]["spearman"] - by_model["mlp"]["spearman"],
            "elite_recall": by_model["icode"]["elite_recall"] - by_model["mlp"]["elite_recall"],
            "normalized_regret": by_model["icode"]["normalized_regret"] - by_model["mlp"]["normalized_regret"],
            "weight_js_divergence": by_model["icode"]["weight_js_divergence"] - by_model["mlp"]["weight_js_divergence"],
        },
    }
    manifest = {
        "design_id": "l82_control_sequence_ranking_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "scene_config": str(scene_path),
        "scene_config_hash": config_hash(base_config),
        "icode_checkpoint": str(icode_path),
        "icode_checkpoint_sha256": _sha256_file(icode_path),
        "mlp_checkpoint": str(mlp_path),
        "mlp_checkpoint_sha256": _sha256_file(mlp_path),
        "seed": int(args.seed),
        "anchor_steps": list(anchor_steps),
        "candidate_count": int(args.candidate_count),
        "candidate_noise_scale": float(args.candidate_noise_scale),
        "horizon": int(controller.config.horizon),
        "control_dt": dt,
        "memory_enabled": False,
        "rl_enabled": False,
        "online_safety_chain_preserved": True,
        "counterfactual_branch_safety_bypass": True,
        "counterfactual_scene_obstacle_count": 0,
        "records": len(records),
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
    parser.add_argument("--mlp-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20268201)
    parser.add_argument("--anchor-steps", default="5,15")
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--candidate-noise-scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.candidate_count < 10:
        parser.error("candidate-count must be at least 10")
    if not np.isfinite(args.candidate_noise_scale) or args.candidate_noise_scale <= 0.0:
        parser.error("candidate-noise-scale must be finite and positive")
    run(args)


if __name__ == "__main__":
    main()
