"""Apply the frozen Chapter 1 noise-basis screening gate to paired episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

DEFAULT_PROTOCOL = ROOT / "configs/research/noise_basis_ab_development_v1.yaml"


def _flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    try:
        return bool(float(value))
    except (TypeError, ValueError):
        return str(value).strip().lower() in {"true", "yes", "on"}


def _number(row, key, default=float("nan")) -> float:
    value = row.get(key)
    if value in (None, "", "nan", "None"):
        return float(default)
    return float(value)


def _finite(values):
    return [float(value) for value in values if math.isfinite(float(value))]


def _run_statistics(mask):
    runs = []
    current = 0
    for value in mask:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return len(runs), max(runs, default=0)


def _point_segment_distance(point, start, end):
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1.0e-16:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * segment)))


def _static_clearance(rows, config):
    robot_radius = float(
        config.get("plant", {}).get("robot", {}).get("collision_radius", 0.25)
    )
    static_obstacles = [
        obstacle
        for obstacle in config.get("scene", {}).get("obstacles", ())
        if not obstacle.get("motion")
    ]
    minimum = float("inf")
    for row in rows:
        point = np.asarray((_number(row, "x"), _number(row, "y")))
        for obstacle in static_obstacles:
            kind = str(obstacle.get("type", "cylinder"))
            if kind == "segment":
                clearance = (
                    _point_segment_distance(
                        point, obstacle["start"], obstacle["end"]
                    )
                    - 0.5 * float(obstacle.get("thickness", 0.20))
                    - robot_radius
                )
            elif kind == "box":
                position = np.asarray(obstacle.get("position", (0.0, 0.0)))
                yaw = float(obstacle.get("yaw", 0.0))
                size = np.asarray(obstacle.get("size", (0.25, 0.25)))
                delta = point - position
                local = np.asarray((
                    math.cos(yaw) * delta[0] + math.sin(yaw) * delta[1],
                    -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1],
                ))
                outside = np.maximum(np.abs(local) - size[:2], 0.0)
                inside = min(float(np.max(np.abs(local) - size[:2])), 0.0)
                clearance = float(np.linalg.norm(outside) + inside - robot_radius)
            else:
                position = np.asarray(obstacle.get("position", (0.0, 0.0)))
                clearance = (
                    float(np.linalg.norm(point - position))
                    - float(obstacle.get("radius", 0.25))
                    - robot_radius
                )
            minimum = min(minimum, clearance)
    return minimum


def _rate_audit(rows, config, prefix):
    rate_limits = config.get("action_space", {}).get("rate_limits")
    if rate_limits is None:
        return {"count": 0, "fraction": 0.0}
    dt = float(config["experiment"]["control_dt"])
    limits = np.asarray(rate_limits, dtype=np.float64) * dt
    controls = np.asarray([
        (_number(row, f"{prefix}_v"), _number(row, f"{prefix}_omega"))
        for row in rows
    ])
    previous = np.vstack((np.zeros((1, 2)), controls[:-1]))
    violations = np.any(np.abs(controls - previous) > limits[None, :] + 1.0e-12, axis=1)
    return {
        "count": int(np.sum(violations)),
        "fraction": float(np.mean(violations)) if len(violations) else 0.0,
    }


def _episode(path: Path, expected_protocol_hash: str):
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        (path / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    recorded_hash = (path / "protocol_sha256.txt").read_text(
        encoding="ascii"
    ).strip()
    if recorded_hash != expected_protocol_hash:
        raise ValueError(f"protocol hash mismatch at {path}")

    with (path / "trajectory.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(metrics["steps"]):
        raise ValueError(f"trajectory/metrics step mismatch at {path}")

    feasible = _finite(
        _number(row, "probabilistic_obstacle_candidate_feasible_fraction")
        for row in rows
    )
    zero_mask = [value <= 0.0 for value in feasible]
    zero_run_count, longest_zero_run = _run_statistics(zero_mask)
    hard_violations = [
        _flag(row.get("probabilistic_obstacle_hard_violation")) for row in rows
    ]
    fallback = [
        _flag(row.get("probabilistic_obstacle_active_fallback_used"))
        for row in rows
    ]
    applied = np.asarray([
        (_number(row, "applied_v"), _number(row, "applied_omega"))
        for row in rows
    ])
    applied_delta = (
        np.diff(applied, axis=0) if len(applied) > 1 else np.zeros((0, 2))
    )
    executed_rate = _rate_audit(rows, config, "executed")
    applied_rate = _rate_audit(rows, config, "applied")

    collision = bool(metrics["collision"])
    success = bool(metrics["success"]) and not collision
    termination_reason = str(metrics["termination_reason"])
    max_steps = int(config["experiment"]["max_steps"])
    timeout = bool(
        not success
        and not collision
        and (
            int(metrics["steps"]) >= max_steps
            or termination_reason.lower() in {"timeout", "max_steps"}
        )
    )
    outcome = (
        "success"
        if success
        else "collision"
        if collision
        else "timeout"
        if timeout
        else termination_reason
    )

    guided_opportunities = int(
        metrics.get("paper_guided_opportunity_count_total", 0)
    )
    gaussian_opportunities = int(
        metrics.get("paper_gaussian_opportunity_count_total", 0)
    )
    total_opportunities = guided_opportunities + gaussian_opportunities
    supervised_opportunities = int(metrics.get("supervised_proposal_count_total", 0))
    supervised_selected_steps = sum(
        _number(row, "supervised_selected_count", 0.0) > 0.0 for row in rows
    )
    safety_override_steps = sum(_flag(row.get("safety_override")) for row in rows)

    return {
        "artifact": str(path.relative_to(ROOT)).replace("\\", "/"),
        "noise_basis": str(config["planner"]["noise_basis"]),
        "outcome": outcome,
        "safe_success": success,
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "termination_reason": termination_reason,
        "steps": int(metrics["steps"]),
        "final_goal_distance": float(metrics["final_goal_distance"]),
        "path_progress_m": float(metrics.get("path_progress_m_max", float("nan"))),
        "path_progress_ratio": float(
            metrics.get("path_progress_ratio_max", float("nan"))
        ),
        "minimum_static_clearance": float(_static_clearance(rows, config)),
        "minimum_dynamic_obstacle_center_distance": float(
            metrics.get(
                "minimum_dynamic_obstacle_center_distance", float("nan")
            )
        ),
        "minimum_overall_clearance": float(
            metrics.get("minimum_clearance", float("nan"))
        ),
        "stuck_steps": int(metrics.get("stuck_steps", 0)),
        "zero_feasible_step_count": int(sum(zero_mask)),
        "zero_feasible_run_count": int(zero_run_count),
        "longest_zero_feasible_run": int(longest_zero_run),
        "hard_violation_count": int(sum(hard_violations)),
        "hard_violation_frequency": float(np.mean(hard_violations)),
        "fallback_activation_count": int(sum(fallback)),
        "fallback_activation_frequency": float(np.mean(fallback)),
        "candidate_feasible_fraction_mean": (
            float(np.mean(feasible)) if feasible else float("nan")
        ),
        "candidate_feasible_fraction_min": (
            float(np.min(feasible)) if feasible else float("nan")
        ),
        "planner_compute_ms_mean": float(metrics["planner_compute_ms_mean"]),
        "planner_compute_ms_p50": float(metrics["planner_compute_ms_p50"]),
        "planner_compute_ms_p95": float(metrics["planner_compute_ms_p95"]),
        "planner_compute_ms_max": float(metrics["planner_compute_ms_max"]),
        "planner_deadline_miss_rate": float(
            metrics["planner_deadline_miss_rate"]
        ),
        "applied_v_mean": float(np.mean(applied[:, 0])),
        "applied_v_abs_mean": float(np.mean(np.abs(applied[:, 0]))),
        "applied_v_std": float(np.std(applied[:, 0])),
        "applied_omega_mean": float(np.mean(applied[:, 1])),
        "applied_omega_abs_mean": float(np.mean(np.abs(applied[:, 1]))),
        "applied_omega_std": float(np.std(applied[:, 1])),
        "applied_action_delta_l2_mean": (
            float(np.mean(np.linalg.norm(applied_delta, axis=1)))
            if len(applied_delta) else 0.0
        ),
        "applied_action_delta_l2_p95": (
            float(np.percentile(np.linalg.norm(applied_delta, axis=1), 95))
            if len(applied_delta) else 0.0
        ),
        "executed_rate_limit_violation_count": executed_rate["count"],
        "executed_rate_limit_violation_fraction": executed_rate["fraction"],
        "applied_rate_limit_violation_count": applied_rate["count"],
        "applied_rate_limit_violation_fraction": applied_rate["fraction"],
        "gaussian_raw_rate_limit_violation_fraction": float(
            metrics.get(
                "paper_gaussian_raw_rate_limit_violation_fraction_mean", 0.0
            )
        ),
        "gaussian_raw_rate_limit_violating_candidate_fraction": float(
            metrics.get(
                "paper_gaussian_raw_rate_limit_violating_candidate_fraction_mean",
                0.0,
            )
        ),
        "gaussian_post_rate_limit_violation_fraction": float(
            metrics.get(
                "paper_gaussian_post_rate_limit_violation_fraction_mean", 0.0
            )
        ),
        "gaussian_candidate_fraction": (
            float(gaussian_opportunities / total_opportunities)
            if total_opportunities else float("nan")
        ),
        "guided_candidate_fraction": (
            float(guided_opportunities / total_opportunities)
            if total_opportunities else float("nan")
        ),
        "guided_elite_survival_fraction": (
            float(metrics.get("paper_guided_elite_count_total", 0))
            / guided_opportunities
            if guided_opportunities else 0.0
        ),
        "gaussian_elite_survival_fraction": (
            float(metrics.get("paper_gaussian_elite_count_total", 0))
            / gaussian_opportunities
            if gaussian_opportunities else 0.0
        ),
        "supervised_actor_candidate_fraction": (
            float(supervised_opportunities / total_opportunities)
            if total_opportunities else 0.0
        ),
        "supervised_actor_elite_survival_fraction": (
            float(metrics.get("supervised_elite_count_total", 0))
            / supervised_opportunities
            if supervised_opportunities else 0.0
        ),
        "supervised_actor_selected_steps": int(supervised_selected_steps),
        "safety_override_steps": int(safety_override_steps),
        "execution_source_note": (
            "per-step action is a weighted MPPI update; exact Gaussian-vs-guided "
            "source is not categorical. supervised selection and safety override "
            "counts are reported separately"
        ),
        "reliability_authority_mean": float(
            metrics.get("reliability_authority_mean", 1.0)
        ),
        "reliability_proposal_authority_mean": float(
            metrics.get("reliability_proposal_authority_mean", 1.0)
        ),
        "reliability_guided_fraction_applied_mean": float(
            metrics.get("reliability_guided_fraction_applied_mean", 0.0)
        ),
    }


def _median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)

    protocol_path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    thresholds = protocol["decision_rule"]["thresholds"]
    seeds = [int(value) for value in protocol["design"]["seeds"]]
    root = args.root.resolve()

    pairs, missing = [], []
    for seed in seeds:
        control_path = root / f"seed{seed}" / "control"
        treatment_path = root / f"seed{seed}" / "treatment"
        if not (control_path / "metrics.json").exists() or not (
            treatment_path / "metrics.json"
        ).exists():
            missing.append(seed)
            continue
        control = _episode(control_path, protocol_hash)
        treatment = _episode(treatment_path, protocol_hash)
        if control["noise_basis"] != protocol["intervention"]["control"]:
            raise ValueError(f"seed {seed} control arm has the wrong noise_basis")
        if treatment["noise_basis"] != protocol["intervention"]["treatment"]:
            raise ValueError(f"seed {seed} treatment arm has the wrong noise_basis")
        pairs.append({
            "seed": seed,
            "control": control,
            "treatment": treatment,
        })

    if missing:
        print(f"INCOMPLETE: {len(missing)} pairs missing: {missing}")
        print("The decision rule evaluates only after all pairs are complete.")
        return 1

    treatment_only_collisions = [
        pair["seed"] for pair in pairs
        if pair["treatment"]["collision"] and not pair["control"]["collision"]
    ]
    control_only_collisions = [
        pair["seed"] for pair in pairs
        if pair["control"]["collision"] and not pair["treatment"]["collision"]
    ]
    gained = [
        pair["seed"] for pair in pairs
        if pair["treatment"]["safe_success"]
        and not pair["control"]["safe_success"]
    ]
    lost = [
        pair["seed"] for pair in pairs
        if pair["control"]["safe_success"]
        and not pair["treatment"]["safe_success"]
    ]
    net_success = len(gained) - len(lost)
    goal_reductions = [
        pair["control"]["final_goal_distance"]
        - pair["treatment"]["final_goal_distance"]
        for pair in pairs
    ]
    median_goal_reduction = _median(goal_reductions)

    if len(treatment_only_collisions) >= int(
        thresholds["treatment_only_collision_pairs"]
    ):
        verdict = "FAIL_SAFETY"
        action = (
            "Seal the evidence, stop expansion, and pause this mechanism family."
        )
    elif (
        net_success >= int(thresholds["net_success_pairs"])
        or median_goal_reduction
        >= float(thresholds["median_goal_distance_reduction_m"])
    ):
        verdict = "PASS_TO_CONFIRMATORY"
        action = (
            "Pre-register a fresh-seed confirmatory with n >= 20 paired seeds."
        )
    else:
        verdict = "FAIL_NO_EFFECT"
        action = (
            "Do not proceed; the offline coverage gain did not clear the "
            "closed-loop screening threshold."
        )

    outcomes = {}
    for arm in ("control", "treatment"):
        episodes = [pair[arm] for pair in pairs]
        outcomes[arm] = {
            "success": sum(value["success"] for value in episodes),
            "collision": sum(value["collision"] for value in episodes),
            "timeout": sum(value["timeout"] for value in episodes),
        }

    def paired_delta(field):
        return _median([
            pair["treatment"][field] - pair["control"][field] for pair in pairs
        ])

    aggregate = {
        "outcomes": outcomes,
        "success_gained_seeds": gained,
        "success_lost_seeds": lost,
        "net_success_pairs": net_success,
        "treatment_only_collision_seeds": treatment_only_collisions,
        "control_only_collision_seeds": control_only_collisions,
        "median_goal_distance_reduction_m": median_goal_reduction,
        "median_treatment_minus_control_steps": paired_delta("steps"),
        "median_treatment_minus_control_minimum_static_clearance_m": paired_delta(
            "minimum_static_clearance"
        ),
        "median_treatment_minus_control_minimum_overall_clearance_m": paired_delta(
            "minimum_overall_clearance"
        ),
        "median_treatment_minus_control_zero_feasible_steps": paired_delta(
            "zero_feasible_step_count"
        ),
        "median_treatment_minus_control_stuck_steps": paired_delta("stuck_steps"),
        "median_treatment_minus_control_planner_p95_ms": paired_delta(
            "planner_compute_ms_p95"
        ),
        "median_treatment_minus_control_raw_rate_violation_fraction": paired_delta(
            "gaussian_raw_rate_limit_violation_fraction"
        ),
        "executed_rate_limit_violations": {
            arm: int(sum(
                pair[arm]["executed_rate_limit_violation_count"]
                for pair in pairs
            ))
            for arm in ("control", "treatment")
        },
        "applied_rate_limit_violations": {
            arm: int(sum(
                pair[arm]["applied_rate_limit_violation_count"]
                for pair in pairs
            ))
            for arm in ("control", "treatment")
        },
    }

    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_hash,
        "map": protocol["map"]["name"],
        "control_basis": protocol["intervention"]["control"],
        "treatment_basis": protocol["intervention"]["treatment"],
        "n_pairs": len(pairs),
        "verdict": verdict,
        "action": action,
        "thresholds": thresholds,
        "aggregate": aggregate,
        "pairs": pairs,
    }
    (root / "ab_screen_result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    csv_columns = [
        "seed",
        "control_outcome",
        "treatment_outcome",
        "control_final_goal_distance",
        "treatment_final_goal_distance",
        "control_minimum_overall_clearance",
        "treatment_minimum_overall_clearance",
        "control_steps",
        "treatment_steps",
        "control_planner_compute_ms_p95",
        "treatment_planner_compute_ms_p95",
    ]
    with (root / "paired_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({
                "seed": pair["seed"],
                **{
                    f"{arm}_{field}": pair[arm][field]
                    for arm in ("control", "treatment")
                    for field in (
                        "outcome",
                        "final_goal_distance",
                        "minimum_overall_clearance",
                        "steps",
                        "planner_compute_ms_p95",
                    )
                },
            })

    print("=" * 88)
    print("NOISE-BASIS A/B CLOSED-LOOP SCREEN")
    print("=" * 88)
    for pair in pairs:
        control, treatment = pair["control"], pair["treatment"]
        print(
            f"{pair['seed']}  {control['outcome']:>9s} -> "
            f"{treatment['outcome']:<9s}  distance "
            f"{control['final_goal_distance']:.3f} -> "
            f"{treatment['final_goal_distance']:.3f} m"
        )
    print(json.dumps(aggregate, indent=2))
    print(f"VERDICT: {verdict}")
    print(action)
    print("Screening result only; no statistical efficacy claim is authorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
