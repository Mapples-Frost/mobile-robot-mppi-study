"""Qualify the frozen conservative Gaussian-mixture collision-risk interface."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from mobile_robot_mppi.core.references import PointGoal
from mobile_robot_mppi.core.spaces import (
    body_velocity_action,
    unicycle_state,
)
from mobile_robot_mppi.obstacles.artifacts import (
    environment_manifest,
    repository_provenance,
    sha256_file,
    write_csv,
    write_json,
    write_yaml,
)
from mobile_robot_mppi.obstacles.collision_risk import (
    CollisionRiskConfig,
    GaussianMixtureObstacleForecast,
    component_collision_probability_upper_bound,
    evaluate_collision_risk,
    mixture_collision_probability_upper_bound,
)
from mobile_robot_mppi.planning.dynamics import LegacyUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "configs/research/probabilistic_collision_risk_v1.yaml"
)


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _forecast(
    means,
    covariance=0.04,
    weights=None,
    radius=0.15,
    timestamp=0.0,
    dt=0.1,
):
    means = np.asarray(means, dtype=np.float64)
    if means.ndim == 2:
        means = means[:, None, :]
    horizon, modes, _ = means.shape
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim == 0:
        covariances = np.broadcast_to(
            float(covariance) * np.eye(2),
            (horizon, modes, 2, 2),
        ).copy()
    elif covariance.shape == (2, 2):
        covariances = np.broadcast_to(
            covariance, (horizon, modes, 2, 2)
        ).copy()
    else:
        covariances = covariance
    if weights is None:
        weights = np.full((horizon, modes), 1.0 / modes)
    return GaussianMixtureObstacleForecast(
        timestamp=timestamp,
        dt=dt,
        component_means=means,
        component_covariances=covariances,
        component_weights=weights,
        radius_m=radius,
        source="frozen_synthetic_qualification",
    )


def _input_contract_checks():
    means = np.zeros((2, 2, 2))
    covariance = np.broadcast_to(
        np.eye(2), (2, 2, 2, 2)
    ).copy()
    rejected = {}
    cases = {
        "bad_weight_sum": dict(
            timestamp=0.0,
            dt=0.1,
            component_means=means,
            component_covariances=covariance,
            component_weights=np.full((2, 2), 0.4),
            radius_m=0.15,
        ),
        "negative_radius": dict(
            timestamp=0.0,
            dt=0.1,
            component_means=means,
            component_covariances=covariance,
            component_weights=np.full((2, 2), 0.5),
            radius_m=-0.15,
        ),
        "non_psd_covariance": dict(
            timestamp=0.0,
            dt=0.1,
            component_means=means,
            component_covariances=np.broadcast_to(
                np.asarray(((1.0, 0.0), (0.0, -1.0))),
                (2, 2, 2, 2),
            ).copy(),
            component_weights=np.full((2, 2), 0.5),
            radius_m=0.15,
        ),
    }
    for name, values in cases.items():
        try:
            GaussianMixtureObstacleForecast(**values)
            rejected[name] = False
        except (TypeError, ValueError):
            rejected[name] = True
    return rejected


def _analytic_qualification(risk_config):
    robot = np.zeros((1, 1, 2))
    inside = _forecast([[[0.40, 0.0]]])
    near = _forecast([[[0.70, 0.0]]])
    far = _forecast([[[1.20, 0.0]]])
    low_uncertainty = _forecast(
        [[[0.85, 0.0]]], covariance=np.diag((0.01, 0.04))
    )
    high_uncertainty = _forecast(
        [[[0.85, 0.0]]], covariance=np.diag((0.09, 0.04))
    )
    values = {
        name: float(
            mixture_collision_probability_upper_bound(
                robot, forecast, risk_config
            )[0, 0]
        )
        for name, forecast in (
            ("inside", inside),
            ("near", near),
            ("far", far),
            ("low_uncertainty", low_uncertainty),
            ("high_uncertainty", high_uncertainty),
        )
    }

    means = np.asarray([[[0.70, 0.0], [1.20, 0.0]]])
    weights = np.asarray([[0.25, 0.75]])
    mixture = _forecast(means, weights=weights)
    components = component_collision_probability_upper_bound(
        robot, mixture, risk_config
    )
    actual = mixture_collision_probability_upper_bound(
        robot, mixture, risk_config
    )
    expected = np.sum(
        components * weights[None, :, :], axis=-1
    )
    permuted = _forecast(
        means[:, ::-1],
        covariance=mixture.component_covariances[:, ::-1],
        weights=weights[:, ::-1],
    )
    permuted_value = mixture_collision_probability_upper_bound(
        robot, permuted, risk_config
    )

    crossing_forecast = _forecast(
        (
            ((0.5, 0.5),),
            ((1.0, 0.0),),
            ((1.5, -0.5),),
        ),
        covariance=np.diag((0.03, 0.03)),
    )
    crossing = np.asarray(
        (((0.5, 0.0), (1.0, 0.0), (1.5, 0.0)),)
    )
    bypass = crossing.copy()
    bypass[:, :, 1] = 1.5
    trajectory = evaluate_collision_risk(
        np.concatenate((crossing, bypass), axis=0),
        (crossing_forecast,),
        risk_config,
    )
    return {
        "scalar_values": values,
        "mixture_linearity_error": float(
            np.max(np.abs(actual - expected))
        ),
        "permutation_error": float(
            np.max(np.abs(actual - permuted_value))
        ),
        "crossing_probability_mass": float(
            trajectory.accumulated_probability_mass[0]
        ),
        "bypass_probability_mass": float(
            trajectory.accumulated_probability_mass[1]
        ),
        "union_bound_minimum_step_gap": float(
            np.min(
                trajectory.horizon_union_bound[:, None]
                - trajectory.step_probability_upper_bound
            )
        ),
        "all_outputs_finite": bool(
            all(np.isfinite(value) for value in values.values())
            and np.isfinite(
                trajectory.step_probability_upper_bound
            ).all()
        ),
        "all_probabilities_in_unit_interval": bool(
            np.all(trajectory.step_probability_upper_bound >= 0.0)
            and np.all(trajectory.step_probability_upper_bound <= 1.0)
        ),
    }


def _monte_carlo_qualification(risk_config):
    rng = np.random.RandomState(730199990)
    robot = np.zeros((1, 1, 2))
    angle = 0.55
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle)),
            (np.sin(angle), np.cos(angle)),
        )
    )
    cases = (
        (np.asarray((0.70, 0.0)), np.diag((0.04, 0.01))),
        (np.asarray((0.85, 0.10)), np.diag((0.09, 0.02))),
        (
            np.asarray((0.75, -0.20)),
            rotation.dot(np.diag((0.06, 0.01))).dot(rotation.T),
        ),
        (
            np.asarray((1.00, 0.25)),
            rotation.dot(np.diag((0.12, 0.03))).dot(rotation.T),
        ),
    )
    combined_radius = (
        risk_config.robot_radius_m
        + 0.15
        + risk_config.safety_margin_m
    )
    sample_count = 120000
    records = []
    for index, (mean, covariance) in enumerate(cases):
        forecast = _forecast(
            [[mean]], covariance=covariance, radius=0.15
        )
        bound = float(
            mixture_collision_probability_upper_bound(
                robot, forecast, risk_config
            )[0, 0]
        )
        samples = rng.multivariate_normal(
            mean, covariance, size=sample_count
        )
        empirical = float(
            np.mean(np.linalg.norm(samples, axis=1) <= combined_radius)
        )
        standard_error = float(
            np.sqrt(
                max(
                    empirical * (1.0 - empirical),
                    1.0 / sample_count,
                )
                / sample_count
            )
        )
        tolerance = 5.0 * standard_error + 1.0 / sample_count
        records.append(
            {
                "case": index + 1,
                "mean_x": float(mean[0]),
                "mean_y": float(mean[1]),
                "bound": bound,
                "empirical_probability": empirical,
                "standard_error": standard_error,
                "tolerance": tolerance,
                "upper_bound_violation": bool(
                    empirical > bound + tolerance
                ),
            }
        )
    return records


def _controller(enabled, horizon=3):
    return MppiController(
        LegacyUnicyclePrediction(),
        unicycle_state(),
        body_velocity_action((0.0, 0.5), 1.0),
        MppiConfig(
            horizon=horizon,
            num_samples=4,
            dt=0.1,
            noise_sigma=(0.1, 0.2),
            probabilistic_obstacle_risk_enabled=enabled,
            probabilistic_obstacle_risk_weight=75.0,
            probabilistic_obstacle_hard_threshold=0.20,
            probabilistic_obstacle_hard_penalty=10000.0,
        ),
    )


def _mppi_adapter_qualification():
    horizon = 3
    trajectories = np.asarray(
        (
            (
                (0.0, 0.0, 0.0),
                (0.3, 0.0, 0.0),
                (0.6, 0.0, 0.0),
                (0.9, 0.0, 0.0),
            ),
            (
                (0.0, 1.2, 0.0),
                (0.3, 1.2, 0.0),
                (0.6, 1.2, 0.0),
                (0.9, 1.2, 0.0),
            ),
        )
    )
    controls = np.zeros((2, horizon, 2))
    forecast = _forecast(
        (((0.3, 0.0),), ((0.6, 0.0),), ((0.9, 0.0),)),
        covariance=np.diag((0.01, 0.01)),
    )
    target = PointGoal(2.0, 0.0).target_at(
        0.0, trajectories[0, 0]
    )
    disabled = _controller(False, horizon)
    baseline = disabled.cost_trajectories(
        trajectories, controls, target
    )
    ignored = disabled.cost_trajectories(
        trajectories,
        controls,
        target,
        probabilistic_obstacles=(forecast,),
    )
    enabled = _controller(True, horizon).cost_trajectories(
        trajectories,
        controls,
        target,
        probabilistic_obstacles=(forecast,),
    )
    return {
        "disabled_maximum_absolute_difference": float(
            np.max(np.abs(baseline - ignored))
        ),
        "enabled_crossing_cost": float(enabled[0]),
        "enabled_bypass_cost": float(enabled[1]),
    }


def _performance_qualification(risk_config):
    rng = np.random.RandomState(730199989)
    batch, horizon, modes = 600, 36, 4
    robot = rng.normal(size=(batch, horizon, 2))
    means = rng.normal(size=(horizon, modes, 2))
    factors = rng.normal(size=(horizon, modes, 2, 2))
    covariance = (
        np.einsum("hmik,hmjk->hmij", factors, factors) * 0.02
    )
    weights = rng.uniform(size=(horizon, modes))
    weights /= weights.sum(axis=1, keepdims=True)
    forecast = _forecast(
        means, covariance=covariance, weights=weights, radius=0.20
    )
    for _ in range(5):
        evaluate_collision_risk(robot, (forecast,), risk_config)
    records = []
    for repeat in range(30):
        started = time.perf_counter()
        evaluate_collision_risk(robot, (forecast,), risk_config)
        records.append(
            {
                "repeat": repeat + 1,
                "runtime_ms": 1000.0
                * (time.perf_counter() - started),
            }
        )
    values = np.asarray(
        [record["runtime_ms"] for record in records]
    )
    return records, {
        "batch": batch,
        "horizon": horizon,
        "modes": modes,
        "repeat_count": len(records),
        "minimum_ms": float(values.min()),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.quantile(values, 0.90)),
        "maximum_ms": float(values.max()),
    }


def run_qualification(config_path=DEFAULT_CONFIG, output_override=None):
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    if config["platform"] != "windows_native_only":
        raise ValueError("risk qualification requires native Windows")
    guards = config["scope_guards"]
    if not all(
        bool(guards[name])
        for name in (
            "obstacle_v3_frozen",
            "change_aware_imm_frozen",
            "synthetic_geometry_only",
            "held_out_predictor_results_not_reused_for_tuning",
        )
    ):
        raise ValueError("risk qualification prerequisite is disabled")
    if any(
        bool(guards[name])
        for name in (
            "sealed_registry_imported",
            "closed_loop_mppi_enabled",
            "rl_icode_hss_enabled",
        )
    ):
        raise ValueError("risk qualification scope was violated")

    risk_values = config["risk_interface"]
    risk_config = CollisionRiskConfig(
        robot_radius_m=float(risk_values["robot_radius_m"]),
        safety_margin_m=float(risk_values["safety_margin_m"]),
        minimum_position_std_m=float(
            risk_values["minimum_position_std_m"]
        ),
        hard_probability_threshold=float(
            config["planner_adapter"][
                "development_smoke_hard_threshold"
            ]
        ),
    )
    contracts = _input_contract_checks()
    analytic = _analytic_qualification(risk_config)
    monte_carlo = _monte_carlo_qualification(risk_config)
    adapter = _mppi_adapter_qualification()
    runtimes, performance = _performance_qualification(risk_config)
    gate = config["qualification_gate"]
    scalar = analytic["scalar_values"]
    checks = {
        "all_input_contract_tests_pass": all(contracts.values()),
        "all_outputs_finite_and_in_unit_interval": (
            analytic["all_outputs_finite"]
            and analytic["all_probabilities_in_unit_interval"]
        ),
        "mean_inside_combined_radius_returns_one": (
            scalar["inside"] == 1.0
        ),
        "closer_outside_mean_has_higher_risk": (
            scalar["far"] < scalar["near"] < scalar["inside"]
        ),
        "larger_radial_covariance_has_higher_risk": (
            scalar["high_uncertainty"] > scalar["low_uncertainty"]
        ),
        "mixture_is_weight_linear": (
            analytic["mixture_linearity_error"] <= 1.0e-14
        ),
        "component_permutation_invariant": (
            analytic["permutation_error"] <= 1.0e-14
        ),
        "crossing_trajectory_risk_exceeds_bypass": (
            analytic["crossing_probability_mass"]
            > analytic["bypass_probability_mass"]
        ),
        "temporal_union_bound_dominates_every_step": (
            analytic["union_bound_minimum_step_gap"] >= -1.0e-14
        ),
        "disabled_mppi_adapter_is_cost_identical": (
            adapter["disabled_maximum_absolute_difference"] == 0.0
        ),
        "enabled_mppi_adapter_penalizes_crossing": (
            adapter["enabled_crossing_cost"]
            > adapter["enabled_bypass_cost"]
        ),
        "synthetic_monte_carlo_upper_bound_violations": (
            sum(row["upper_bound_violation"] for row in monte_carlo)
            == int(gate["synthetic_monte_carlo_upper_bound_violations"])
        ),
        "maximum_reference_batch_runtime_ms": (
            performance["maximum_ms"]
            <= float(gate["maximum_reference_batch_runtime_ms"])
        ),
    }
    passed = bool(all(checks.values()))
    summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "qualification_pass": passed,
        "input_contract_rejections": contracts,
        "analytic": analytic,
        "monte_carlo": monte_carlo,
        "mppi_adapter": adapter,
        "performance": performance,
        "gate_checks": checks,
    }
    output_dir = (
        Path(output_override).resolve()
        if output_override is not None
        else (
            ROOT
            / "research_artifacts"
            / "probabilistic_collision_risk_v1_qualification"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": write_yaml(
            output_dir / "manifest.yaml",
            {
                "schema_version": 1,
                "study_id": config["study_id"],
                "stage": config["stage"],
                "platform": config["platform"],
                "config_path": str(config_path.relative_to(ROOT)),
                "scope_guards": guards,
            },
        ),
        "config": write_yaml(
            output_dir / "config_resolved.yaml", config
        ),
        "environment": write_json(
            output_dir / "environment_manifest.json",
            environment_manifest(ROOT),
        ),
        "runtime": write_csv(
            output_dir / "analysis/runtime_benchmark.csv", runtimes
        ),
        "summary": write_json(
            output_dir / "analysis/summary.json", summary
        ),
    }
    report_path = output_dir / "analysis/summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Probabilistic collision risk V1 qualification\n\n"
        f"- Qualification: {passed}\n"
        f"- Crossing probability mass: "
        f"{analytic['crossing_probability_mass']:.6f}\n"
        f"- Bypass probability mass: "
        f"{analytic['bypass_probability_mass']:.6f}\n"
        f"- Monte Carlo upper-bound violations: "
        f"{sum(row['upper_bound_violation'] for row in monte_carlo)}\n"
        f"- Reference batch median runtime: "
        f"{performance['median_ms']:.6f} ms\n"
        f"- Reference batch maximum runtime: "
        f"{performance['maximum_ms']:.6f} ms\n",
        encoding="utf-8",
    )
    paths["report"] = report_path
    provenance = {
        "schema_version": 1,
        "stage": config["stage"],
        **repository_provenance(ROOT),
        "files": {
            name: {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    write_json(output_dir / "provenance.json", provenance)
    write_json(
        output_dir / "integrity_audit.json",
        {
            "schema_version": 1,
            "all_expected_files_exist": all(
                path.is_file() for path in paths.values()
            ),
            "qualification_pass": passed,
            "files": {
                name: sha256_file(path)
                for name, path in paths.items()
            },
        },
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_qualification(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

