"""Run preregistered offline CV/Gaussian-CV/ordinary-IMM development study."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from mobile_robot_mppi.obstacles.artifacts import (
    environment_manifest,
    repository_provenance,
    sha256_file,
    trajectory_records,
    validate_split_registry,
    write_csv,
    write_json,
    write_yaml,
)
from mobile_robot_mppi.obstacles.imm import (
    IMM_MODEL_NAMES,
    OrdinaryIMMPredictor,
    run_online_imm_forecasts,
)
from mobile_robot_mppi.obstacles.motion import noise_profiles_from_mapping
from mobile_robot_mppi.obstacles.patrol import (
    V3_PROCESS_NAMES,
    generate_patrol_trajectory,
)
from mobile_robot_mppi.obstacles.prediction import (
    DeterministicCVPredictor,
    GaussianCVKalmanPredictor,
    run_online_forecasts,
)
from mobile_robot_mppi.obstacles.predictor_evaluation import (
    evaluate_forecasts,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/research/ordinary_imm_v3_development.yaml"
PREDICTOR_NAMES = (
    "deterministic_cv",
    "gaussian_cv_kalman",
    "ordinary_imm",
)


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping")
    return value


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_schedule(processes, noise_profiles, seeds, schedule_seed):
    if tuple(processes) != V3_PROCESS_NAMES:
        raise ValueError("ordinary IMM study requires all V3 processes")
    if tuple(noise_profiles) != ("low", "medium"):
        raise ValueError("ordinary IMM study requires low/medium noise")
    if len(seeds) != 10 or len(set(int(seed) for seed in seeds)) != 10:
        raise ValueError("ordinary IMM study requires ten unique seeds")
    jobs = [
        {
            "process": process,
            "noise_profile": noise,
            "seed": int(seed),
        }
        for process in processes
        for noise in noise_profiles
        for seed in seeds
    ]
    permutation = np.random.RandomState(int(schedule_seed)).permutation(
        len(jobs)
    )
    result = []
    for run_order, job_index in enumerate(permutation):
        job = dict(jobs[int(job_index)])
        job["run_order"] = int(run_order)
        job["experimental_key"] = (
            f"{job['process']}::{job['noise_profile']}::seed{job['seed']}"
        )
        result.append(job)
    return result


def _imm_predictor(config, observation_std):
    imm = config["ordinary_imm"]
    return OrdinaryIMMPredictor(
        observation_std=float(observation_std),
        initial_velocity_std=float(
            config["prediction"]["initial_velocity_std"]
        ),
        initial_mode_probabilities=imm["initial_mode_probabilities"],
        transition_matrix=imm["transition_matrix"],
        turn_rate_radps=imm["turn_rate_radps"],
        brake_decay_rate_per_s=imm["brake_decay_rate_per_s"],
        process_acceleration_std=imm["process_acceleration_std"],
    )


def _filtered_digest(records, imm=False):
    digest = hashlib.sha256()
    for record in records:
        digest.update(np.asarray(record.filtered_mean).tobytes())
        if record.filtered_covariance is not None:
            digest.update(np.asarray(record.filtered_covariance).tobytes())
        if imm:
            digest.update(
                np.asarray(record.filtered_mode_probabilities).tobytes()
            )
    return digest.hexdigest()


def _imm_numerical_audit(records):
    minimum_eigenvalue = float("inf")
    maximum_asymmetry = 0.0
    maximum_weight_sum_error = 0.0
    minimum_weight = 1.0
    covariance_count = 0
    for record in records:
        probabilities = np.vstack(
            (
                record.filtered_mode_probabilities[None, :],
                record.future.mode_probabilities,
            )
        )
        maximum_weight_sum_error = max(
            maximum_weight_sum_error,
            float(np.abs(probabilities.sum(axis=1) - 1.0).max()),
        )
        minimum_weight = min(minimum_weight, float(probabilities.min()))
        covariance_batches = (
            record.future.covariances[:, None, :, :],
            record.future.component_covariances,
        )
        covariance = np.concatenate(covariance_batches, axis=1).reshape(
            -1, 4, 4
        )
        covariance_count += covariance.shape[0]
        maximum_asymmetry = max(
            maximum_asymmetry,
            float(np.abs(covariance - covariance.transpose(0, 2, 1)).max()),
        )
        minimum_eigenvalue = min(
            minimum_eigenvalue,
            float(np.linalg.eigvalsh(covariance).min()),
        )
    return {
        "covariance_count": covariance_count,
        "finite": bool(
            np.isfinite(
                (
                    minimum_eigenvalue,
                    maximum_asymmetry,
                    maximum_weight_sum_error,
                    minimum_weight,
                )
            ).all()
        ),
        "minimum_covariance_eigenvalue": minimum_eigenvalue,
        "maximum_covariance_asymmetry": maximum_asymmetry,
        "minimum_mode_probability": minimum_weight,
        "maximum_mode_probability_sum_error": maximum_weight_sum_error,
        "positive_semidefinite": minimum_eigenvalue >= -1.0e-9,
        "symmetric": maximum_asymmetry <= 1.0e-10,
        "probabilities_valid": (
            minimum_weight >= -1.0e-12
            and maximum_weight_sum_error <= 1.0e-10
        ),
    }


def _trace_rows(predictor_name, records, truth_states, dt):
    rows = []
    horizon_steps = (int(round(1.0 / dt)), int(round(2.0 / dt)), int(round(3.0 / dt)))
    for record in records:
        for horizon_step in horizon_steps:
            if (
                horizon_step > record.future.means.shape[0]
                or record.time_index + horizon_step >= truth_states.shape[0]
            ):
                continue
            index = horizon_step - 1
            covariance = (
                None
                if record.future.covariances is None
                else record.future.covariances[index]
            )
            probabilities = (
                None
                if not hasattr(record.future, "mode_probabilities")
                else record.future.mode_probabilities[index]
            )
            truth = truth_states[record.time_index + horizon_step]
            row = {
                "predictor": predictor_name,
                "origin_index": int(record.time_index),
                "origin_time_s": float(record.timestamp),
                "origin_observation_available": int(
                    record.observation_available
                ),
                "horizon_s": float(horizon_step * dt),
                "pred_px": float(record.future.means[index, 0]),
                "pred_py": float(record.future.means[index, 1]),
                "true_px": float(truth[0]),
                "true_py": float(truth[1]),
                "cov_xx": None if covariance is None else float(covariance[0, 0]),
                "cov_xy": None if covariance is None else float(covariance[0, 1]),
                "cov_yy": None if covariance is None else float(covariance[1, 1]),
            }
            for mode_index, mode_name in enumerate(IMM_MODEL_NAMES):
                row["prob_%s" % mode_name] = (
                    None
                    if probabilities is None
                    else float(probabilities[mode_index])
                )
            rows.append(row)
    return rows


def _write_run_artifacts(
    run_dir,
    trajectory,
    config,
    job,
    metrics,
    trace_rows,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": write_yaml(
            run_dir / "config_resolved.yaml",
            {
                "study_id": config["study_id"],
                "job": dict(job),
                "prediction": dict(config["prediction"]),
                "ordinary_imm": dict(config["ordinary_imm"]),
                "scope_guards": dict(config["scope_guards"]),
            },
        ),
        "trajectory": write_csv(
            run_dir / "trajectory.csv", trajectory_records(trajectory)
        ),
        "forecast_trace": write_csv(
            run_dir / "forecast_checkpoints.csv", trace_rows
        ),
        "metrics": write_json(run_dir / "metrics.json", metrics),
    }
    provenance = {
        "schema_version": 1,
        "stage": "offline_predictor_baseline",
        **repository_provenance(ROOT),
        "files": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    write_json(run_dir / "provenance.json", provenance)
    return provenance


def _run_one_worker(job, config_path, output_dir):
    config = _load_yaml(config_path)
    obstacle_config = _load_yaml(_resolve(config["obstacle_config_path"]))
    profiles = noise_profiles_from_mapping(obstacle_config["noise_profiles"])
    profile = profiles[str(job["noise_profile"])]
    process = str(job["process"])
    seed = int(job["seed"])
    trajectory = generate_patrol_trajectory(
        process, seed, profile, obstacle_config
    )
    dt = float(obstacle_config["trajectory"]["dt"])
    horizon = int(round(float(config["prediction"]["horizon_s"]) / dt))
    stride = int(config["prediction"]["forecast_stride_steps"])
    warmup = int(round(float(config["prediction"]["warmup_s"]) / dt))
    common = {
        "horizon": horizon,
        "forecast_dt": dt,
        "forecast_stride_steps": stride,
        "warmup_steps": warmup,
    }
    deterministic = run_online_forecasts(
        trajectory.times,
        trajectory.observations,
        trajectory.observed_mask,
        DeterministicCVPredictor(),
        **common,
    )
    gaussian = run_online_forecasts(
        trajectory.times,
        trajectory.observations,
        trajectory.observed_mask,
        GaussianCVKalmanPredictor(
            process_acceleration_std=float(
                config["prediction"][
                    "gaussian_cv_process_acceleration_std"
                ]
            ),
            observation_std=profile.observation_std,
            initial_velocity_std=float(
                config["prediction"]["initial_velocity_std"]
            ),
        ),
        **common,
    )
    imm = run_online_imm_forecasts(
        trajectory.times,
        trajectory.observations,
        trajectory.observed_mask,
        _imm_predictor(config, profile.observation_std),
        **common,
    )
    replay = run_online_imm_forecasts(
        trajectory.times,
        trajectory.observations,
        trajectory.observed_mask,
        _imm_predictor(config, profile.observation_std),
        horizon=1,
        forecast_dt=dt,
        forecast_stride_steps=stride,
        warmup_steps=warmup,
    )
    replay_exact = _filtered_digest(imm, imm=True) == _filtered_digest(
        replay, imm=True
    )
    numerical = _imm_numerical_audit(imm)
    metrics = {
        "schema_version": 1,
        "process": process,
        "noise_profile": profile.name,
        "seed": seed,
        "predictors": {
            "deterministic_cv": evaluate_forecasts(
                deterministic,
                trajectory.states,
                dt,
                trajectory.metadata["events"],
            ),
            "gaussian_cv_kalman": evaluate_forecasts(
                gaussian,
                trajectory.states,
                dt,
                trajectory.metadata["events"],
            ),
            "ordinary_imm": evaluate_forecasts(
                imm,
                trajectory.states,
                dt,
                trajectory.metadata["events"],
            ),
        },
        "imm_numerical_audit": numerical,
        "imm_filter_replay_exact": replay_exact,
        "invariants_pass": bool(
            replay_exact
            and numerical["finite"]
            and numerical["positive_semidefinite"]
            and numerical["symmetric"]
            and numerical["probabilities_valid"]
        ),
    }
    traces = []
    for name, records in (
        ("deterministic_cv", deterministic),
        ("gaussian_cv_kalman", gaussian),
        ("ordinary_imm", imm),
    ):
        traces.extend(
            _trace_rows(name, records, trajectory.states, dt)
        )
    run_dir = (
        Path(output_dir)
        / "runs"
        / process
        / profile.name
        / f"seed_{seed}"
    )
    _write_run_artifacts(
        run_dir, trajectory, config, job, metrics, traces
    )
    progress = {
        "run_order": int(job["run_order"]),
        "experimental_key": str(job["experimental_key"]),
        "process": process,
        "noise_profile": profile.name,
        "seed": seed,
        "deterministic_ade_m": metrics["predictors"][
            "deterministic_cv"
        ]["overall"]["ade_m"],
        "gaussian_ade_m": metrics["predictors"][
            "gaussian_cv_kalman"
        ]["overall"]["ade_m"],
        "gaussian_coverage_90": metrics["predictors"][
            "gaussian_cv_kalman"
        ]["overall"]["coverage_90"],
        "imm_ade_m": metrics["predictors"]["ordinary_imm"]["overall"][
            "ade_m"
        ],
        "imm_nll": metrics["predictors"]["ordinary_imm"]["overall"][
            "nll"
        ],
        "imm_coverage_90": metrics["predictors"][
            "ordinary_imm"
        ]["overall"]["coverage_90"],
        "imm_mixture_spread_m": metrics["predictors"][
            "ordinary_imm"
        ]["overall"]["mean_mixture_spread_m"],
        "imm_minimum_covariance_eigenvalue": numerical[
            "minimum_covariance_eigenvalue"
        ],
        "imm_maximum_probability_sum_error": numerical[
            "maximum_mode_probability_sum_error"
        ],
        "imm_filter_replay_exact": int(replay_exact),
        "invariants_pass": int(metrics["invariants_pass"]),
        "run_dir": str(run_dir.relative_to(output_dir)),
    }
    return {"progress": progress, "metrics": metrics}


def _mean_present(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _aggregate_metric(rows, predictor, section, metric, process=None):
    values = []
    for row in rows:
        if process is not None and row["progress"]["process"] != process:
            continue
        value = row["metrics"]["predictors"][predictor]
        for key in section:
            value = value[key]
        values.append(value[metric])
    return _mean_present(values)


def _aggregate_results(rows):
    by_process = {}
    for process in V3_PROCESS_NAMES:
        by_process[process] = {}
        for predictor in PREDICTOR_NAMES:
            by_process[process][predictor] = {
                metric: _aggregate_metric(
                    rows, predictor, ("overall",), metric, process
                )
                for metric in (
                    "ade_m",
                    "fde_3s_m",
                    "nll",
                    "coverage_50",
                    "coverage_90",
                    "coverage_95",
                    "mean_95_area_m2",
                    "mean_mixture_spread_m",
                )
            }
    macro = {}
    for predictor in PREDICTOR_NAMES:
        macro[predictor] = {}
        for metric in (
            "ade_m",
            "fde_3s_m",
            "nll",
            "coverage_50",
            "coverage_90",
            "coverage_95",
            "mean_95_area_m2",
            "mean_mixture_spread_m",
        ):
            macro[predictor][metric] = _mean_present(
                by_process[process][predictor][metric]
                for process in V3_PROCESS_NAMES
            )
    event_windows = {}
    for window in (
        "pre_change",
        "post_change_0_1",
        "post_change_1_3",
        "steady",
    ):
        event_windows[window] = {}
        for predictor in PREDICTOR_NAMES:
            event_windows[window][predictor] = {
                metric: _aggregate_metric(
                    rows,
                    predictor,
                    ("event_windows", window),
                    metric,
                )
                for metric in ("ade_m", "nll", "coverage_90")
            }
    observation_status = {}
    for status in ("observation_available", "observation_missing"):
        observation_status[status] = {}
        for predictor in PREDICTOR_NAMES:
            observation_status[status][predictor] = {
                metric: _aggregate_metric(
                    rows,
                    predictor,
                    ("observation_status", status),
                    metric,
                )
                for metric in ("ade_m", "nll", "coverage_90")
            }
    return {
        "by_process": by_process,
        "macro": macro,
        "event_windows": event_windows,
        "observation_status": observation_status,
    }


def _plot_summary(summary, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    process_labels = ("P1", "P2", "P3", "P4")
    colors = {
        "deterministic_cv": "#7A7A7A",
        "gaussian_cv_kalman": "#0072B2",
        "ordinary_imm": "#D55E00",
    }
    figure, axes = plt.subplots(
        2, 2, figsize=(12.0, 8.3), constrained_layout=True
    )
    x = np.arange(4, dtype=np.float64)
    width = 0.24
    for offset, predictor in enumerate(PREDICTOR_NAMES):
        values = [
            summary["by_process"][process][predictor]["ade_m"]
            for process in V3_PROCESS_NAMES
        ]
        axes[0, 0].bar(
            x + (offset - 1) * width,
            values,
            width,
            color=colors[predictor],
            label=predictor,
        )
    axes[0, 0].set_xticks(x, process_labels)
    axes[0, 0].set_ylabel("ADE [m]")
    axes[0, 0].set_title("Position error by V3 process")
    axes[0, 0].legend(fontsize=8)

    for offset, predictor in enumerate(
        ("gaussian_cv_kalman", "ordinary_imm")
    ):
        values = [
            summary["by_process"][process][predictor]["coverage_90"]
            for process in V3_PROCESS_NAMES
        ]
        axes[0, 1].bar(
            x + (offset - 0.5) * 0.34,
            values,
            0.34,
            color=colors[predictor],
            label=predictor,
        )
    axes[0, 1].axhline(
        0.90, color="#222222", linestyle="--", linewidth=1.2, label="nominal 90%"
    )
    axes[0, 1].set_ylim(0.0, 1.02)
    axes[0, 1].set_xticks(x, process_labels)
    axes[0, 1].set_ylabel("Moment-matched 90% coverage")
    axes[0, 1].set_title("Calibration by V3 process")
    axes[0, 1].legend(fontsize=8)

    windows = (
        "pre_change",
        "post_change_0_1",
        "post_change_1_3",
        "steady",
    )
    window_labels = ("pre -2–0", "post 0–1", "post 1–3", "steady")
    for predictor in ("gaussian_cv_kalman", "ordinary_imm"):
        axes[1, 0].plot(
            window_labels,
            [
                summary["event_windows"][window][predictor]["ade_m"]
                for window in windows
            ],
            marker="o",
            linewidth=2.0,
            color=colors[predictor],
            label=predictor,
        )
        axes[1, 1].plot(
            window_labels,
            [
                summary["event_windows"][window][predictor]["coverage_90"]
                for window in windows
            ],
            marker="o",
            linewidth=2.0,
            color=colors[predictor],
            label=predictor,
        )
    axes[1, 0].set_ylabel("ADE [m]")
    axes[1, 0].set_title("Error around registered events")
    axes[1, 1].axhline(
        0.90, color="#222222", linestyle="--", linewidth=1.2
    )
    axes[1, 1].set_ylim(0.0, 1.02)
    axes[1, 1].set_ylabel("90% coverage")
    axes[1, 1].set_title("Calibration around registered events")
    for axis in axes.flat:
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=12)
    figure.suptitle(
        "Ordinary IMM development baseline on frozen V3 obstacles",
        fontsize=15,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return output_path


def run_development(config_path=DEFAULT_CONFIG, output_override=None, workers=None):
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    if str(config.get("platform")) != "windows_native_only":
        raise ValueError("ordinary IMM study is restricted to native Windows")
    guards = config["scope_guards"]
    required_true = ("obstacle_v3_frozen", "development_seeds_only")
    required_false = (
        "sealed_registry_imported",
        "change_aware_enabled",
        "collision_risk_enabled",
        "mppi_enabled",
        "rl_icode_hss_enabled",
    )
    if not all(bool(guards[name]) for name in required_true):
        raise ValueError("ordinary IMM prerequisite guards are not enabled")
    if any(bool(guards[name]) for name in required_false):
        raise ValueError("ordinary IMM scope guard was violated")
    obstacle_config_path = _resolve(config["obstacle_config_path"]).resolve()
    obstacle_config = _load_yaml(obstacle_config_path)
    if obstacle_config["generator_version"] != "recurrent_semimarkov_v3":
        raise ValueError("ordinary IMM study requires frozen V3")
    registry_path = _resolve(config["seed_registry_path"]).resolve()
    registry = _load_yaml(registry_path)
    validate_split_registry(registry)
    development = registry["splits"]["development"]
    development_set = set(
        range(
            int(development["seed_start"]),
            int(development["seed_start"]) + int(development["count"]),
        )
    )
    start = int(config["development_seed_start"])
    count = int(config["development_seed_count"])
    seeds = list(range(start, start + count))
    if not set(seeds).issubset(development_set):
        raise ValueError("ordinary IMM seeds must belong to development")
    if set(seeds).intersection(range(730100001, 730100041)):
        raise ValueError("ordinary IMM study may not reuse qualification seeds")
    schedule = build_schedule(
        config["processes"],
        config["noise_profiles"],
        seeds,
        config["schedule_seed"],
    )
    output_dir = (
        Path(output_override).resolve()
        if output_override is not None
        else _resolve(config["output_dir"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(
        output_dir / "manifest.yaml",
        {
            "schema_version": 1,
            "study_id": config["study_id"],
            "stage": config["stage"],
            "platform": config["platform"],
            "config_path": str(config_path.relative_to(ROOT)),
            "obstacle_config_path": str(
                obstacle_config_path.relative_to(ROOT)
            ),
            "seed_registry_path": str(registry_path.relative_to(ROOT)),
            "expected_runs": 80,
            "development_seeds": seeds,
            "predictors": list(PREDICTOR_NAMES),
            "scope_guards": dict(guards),
        },
    )
    write_json(
        output_dir / "schedule.json",
        {"schedule_seed": config["schedule_seed"], "jobs": schedule},
    )
    write_json(
        output_dir / "environment_manifest.json",
        environment_manifest(ROOT),
    )
    worker_count = int(
        config["worker_count"] if workers is None else workers
    )
    results = []
    if worker_count == 1:
        for job in schedule:
            results.append(
                _run_one_worker(job, config_path, output_dir)
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_one_worker, job, config_path, output_dir
                ): job
                for job in schedule
            }
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda row: row["progress"]["run_order"])
    write_csv(
        output_dir / "progress.csv",
        [row["progress"] for row in results],
    )
    aggregate = _aggregate_results(results)
    imm_macro = aggregate["macro"]["ordinary_imm"]
    gaussian_macro = aggregate["macro"]["gaussian_cv_kalman"]
    ratio = imm_macro["ade_m"] / gaussian_macro["ade_m"]
    gate = config["gate"]
    gate_checks = {
        "run_count_complete": len(results) == 80,
        "unique_keys": len(
            {row["progress"]["experimental_key"] for row in results}
        )
        == 80,
        "all_invariants_pass": all(
            bool(row["progress"]["invariants_pass"]) for row in results
        ),
        "macro_coverage_above_minimum": (
            imm_macro["coverage_90"]
            >= float(gate["minimum_macro_coverage_90"])
        ),
        "macro_coverage_below_maximum": (
            imm_macro["coverage_90"]
            <= float(gate["maximum_macro_coverage_90"])
        ),
        "macro_ade_ratio_within_limit": (
            ratio
            <= float(gate["maximum_imm_to_gaussian_macro_ade_ratio"])
        ),
    }
    predictor_gate_pass = all(gate_checks.values())
    summary = {
        "schema_version": 1,
        "run_count": len(results),
        "passed_invariant_runs": sum(
            int(row["progress"]["invariants_pass"]) for row in results
        ),
        "aggregate": aggregate,
        "imm_to_gaussian_macro_ade_ratio": float(ratio),
        "gate_checks": gate_checks,
        "predictor_gate_pass": predictor_gate_pass,
    }
    write_json(output_dir / "analysis/summary.json", summary)
    figure_path = _plot_summary(
        aggregate, output_dir / "figures/predictor_comparison.png"
    )
    summary_text = (
        "# Ordinary IMM development summary\n\n"
        f"- Runs: {summary['passed_invariant_runs']}/{summary['run_count']} "
        "passed numerical invariants\n"
        f"- Gaussian CV macro ADE: {gaussian_macro['ade_m']:.6f} m\n"
        f"- Ordinary IMM macro ADE: {imm_macro['ade_m']:.6f} m\n"
        f"- Ordinary IMM macro NLL: {imm_macro['nll']:.6f}\n"
        f"- Ordinary IMM macro 90% coverage: "
        f"{imm_macro['coverage_90']:.6f}\n"
        f"- IMM/Gaussian ADE ratio: {ratio:.6f}\n"
        f"- Predictor Gate: {predictor_gate_pass}\n"
    )
    analysis_md = output_dir / "analysis/summary.md"
    analysis_md.write_text(summary_text, encoding="utf-8")
    integrity_paths = (
        output_dir / "manifest.yaml",
        output_dir / "schedule.json",
        output_dir / "environment_manifest.json",
        output_dir / "progress.csv",
        output_dir / "analysis/summary.json",
        analysis_md,
        figure_path,
        figure_path.with_suffix(".pdf"),
    )
    write_json(
        output_dir / "integrity_audit.json",
        {
            "schema_version": 1,
            "all_expected_files_exist": all(
                path.is_file() for path in integrity_paths
            ),
            "files": {
                str(path.relative_to(output_dir)): sha256_file(path)
                for path in integrity_paths
            },
            "predictor_gate_pass": predictor_gate_pass,
        },
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_development(
        args.config, args.output_dir, workers=args.workers
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
