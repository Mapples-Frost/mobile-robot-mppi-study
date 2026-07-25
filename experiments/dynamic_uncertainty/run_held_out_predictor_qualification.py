"""Run frozen held-out ID/OOD predictor qualification."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from mobile_robot_mppi.obstacles.artifacts import (
    covariance_audit,
    environment_manifest,
    repository_provenance,
    sha256_file,
    trajectory_records,
    validate_split_registry,
    write_csv,
    write_json,
    write_yaml,
)
from mobile_robot_mppi.obstacles.change_detection_evaluation import (
    evaluate_change_detections,
)
from mobile_robot_mppi.obstacles.motion import noise_profiles_from_mapping
from mobile_robot_mppi.obstacles.patrol import (
    V3_PROCESS_NAMES,
    audit_patrol_trajectory,
    generate_patrol_trajectory,
    validate_v3_config,
)
from mobile_robot_mppi.obstacles.prediction import (
    DeterministicCVPredictor,
    GaussianCVKalmanPredictor,
    run_online_forecasts,
)
from mobile_robot_mppi.obstacles.predictor_evaluation import (
    evaluate_forecasts,
)
from mobile_robot_mppi.obstacles.qualification import (
    deep_merge_ood_config,
    holm_adjust,
    paired_seed_summary,
)

try:
    from .run_change_aware_imm_v3_development import (
        CHANGE_PROCESSES,
        _aggregate_detection,
        _aggregate_predictor,
        _change_predictor,
        _run_records,
    )
    from .run_ordinary_imm_v3_development import (
        _filtered_digest,
        _imm_numerical_audit,
        _imm_predictor,
        _load_yaml,
        _resolve,
        _trace_rows,
    )
except ImportError:
    from run_change_aware_imm_v3_development import (
        CHANGE_PROCESSES,
        _aggregate_detection,
        _aggregate_predictor,
        _change_predictor,
        _run_records,
    )
    from run_ordinary_imm_v3_development import (
        _filtered_digest,
        _imm_numerical_audit,
        _imm_predictor,
        _load_yaml,
        _resolve,
        _trace_rows,
    )


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "configs/research/held_out_predictor_qualification.yaml"
)
PREDICTOR_NAMES = (
    "deterministic_cv",
    "gaussian_cv_kalman",
    "ordinary_imm",
    "change_aware_imm",
)
FROZEN_CHANGE_CANDIDATE = "dual_975_999"


def build_schedule(config):
    jobs = []
    for split_name, split in config["splits"].items():
        start = int(split["seed_start"])
        seeds = range(start, start + int(split["seed_count"]))
        for process in config["processes"]:
            for noise in split["noise_profiles"]:
                for seed in seeds:
                    jobs.append(
                        {
                            "split": str(split_name),
                            "process": str(process),
                            "noise_profile": str(noise),
                            "seed": int(seed),
                        }
                    )
    permutation = np.random.RandomState(
        int(config["schedule_seed"])
    ).permutation(len(jobs))
    result = []
    for run_order, source_index in enumerate(permutation):
        job = dict(jobs[int(source_index)])
        job["run_order"] = int(run_order)
        job["experimental_key"] = (
            f"{job['split']}::{job['process']}::{job['noise_profile']}"
            f"::seed{job['seed']}"
        )
        result.append(job)
    return result


def _obstacle_config_for_split(config, split_name):
    base = _load_yaml(_resolve(config["id_obstacle_config_path"]))
    if split_name == "held_out_id":
        result = base
    elif split_name == "held_out_ood":
        override = _load_yaml(
            _resolve(config["ood_override_config_path"])
        )
        result = deep_merge_ood_config(base, override)
    else:
        raise ValueError("unknown held-out split")
    validate_v3_config(result)
    return result


def _trajectory_invariants(audit):
    return all(
        bool(audit[name])
        for name in (
            "finite_truth",
            "finite_available_observations",
            "speed_within_limit",
            "acceleration_within_limit",
            "yaw_rate_within_limit",
            "teleport_free",
        )
    )


def _write_run(
    run_dir,
    trajectory,
    config,
    ordinary_config,
    change_config,
    obstacle_config,
    job,
    metrics,
    traces,
    detection_events,
    provenance_snapshot,
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
                "ordinary_imm": dict(ordinary_config["ordinary_imm"]),
                "change_candidate": FROZEN_CHANGE_CANDIDATE,
                "change_detector": dict(
                    change_config["pilot_candidates"][
                        FROZEN_CHANGE_CANDIDATE
                    ]
                ),
                "change_response": dict(
                    change_config["change_response"]
                ),
                "obstacle_config": obstacle_config,
                "scope_guards": dict(config["scope_guards"]),
            },
        ),
        "trajectory": write_csv(
            run_dir / "trajectory.csv", trajectory_records(trajectory)
        ),
        "forecast_trace": write_csv(
            run_dir / "forecast_checkpoints.csv", traces
        ),
        "detections": write_json(
            run_dir / "detection_events.json",
            {"events": detection_events},
        ),
        "metrics": write_json(run_dir / "metrics.json", metrics),
    }
    write_json(
        run_dir / "provenance.json",
        {
            "schema_version": 1,
            "stage": "offline_predictor_qualification",
            **provenance_snapshot,
            "files": {
                name: {"path": path.name, "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        },
    )


def _run_one_worker(job, config_path, output_dir, provenance_snapshot):
    config = _load_yaml(config_path)
    ordinary_config = _load_yaml(
        _resolve(config["ordinary_config_path"])
    )
    change_config = _load_yaml(
        _resolve(config["change_aware_config_path"])
    )
    obstacle_config = _obstacle_config_for_split(
        config, str(job["split"])
    )
    profiles = noise_profiles_from_mapping(
        obstacle_config["noise_profiles"]
    )
    profile = profiles[str(job["noise_profile"])]
    trajectory = generate_patrol_trajectory(
        str(job["process"]),
        int(job["seed"]),
        profile,
        obstacle_config,
    )
    trajectory_audit = audit_patrol_trajectory(
        trajectory, obstacle_config
    )
    dt = float(obstacle_config["trajectory"]["dt"])
    horizon = int(
        round(float(config["prediction"]["horizon_s"]) / dt)
    )
    stride = int(config["prediction"]["forecast_stride_steps"])
    warmup_s = float(config["prediction"]["warmup_s"])
    warmup = int(round(warmup_s / dt))
    online = {
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
        **online,
    )
    gaussian_predictor = GaussianCVKalmanPredictor(
        process_acceleration_std=float(
            ordinary_config["prediction"][
                "gaussian_cv_process_acceleration_std"
            ]
        ),
        observation_std=profile.observation_std,
        initial_velocity_std=float(
            ordinary_config["prediction"]["initial_velocity_std"]
        ),
    )
    gaussian = run_online_forecasts(
        trajectory.times,
        trajectory.observations,
        trajectory.observed_mask,
        gaussian_predictor,
        **online,
    )
    ordinary_predictor = _imm_predictor(
        ordinary_config, profile.observation_std
    )
    ordinary = _run_records(
        ordinary_predictor,
        trajectory,
        horizon,
        dt,
        stride,
        warmup,
    )
    change_predictor = _change_predictor(
        change_config,
        ordinary_config,
        profile.observation_std,
        FROZEN_CHANGE_CANDIDATE,
    )
    change = _run_records(
        change_predictor,
        trajectory,
        horizon,
        dt,
        stride,
        warmup,
    )
    events = trajectory.metadata["events"]
    predictor_metrics = {
        "deterministic_cv": evaluate_forecasts(
            deterministic, trajectory.states, dt, events
        ),
        "gaussian_cv_kalman": evaluate_forecasts(
            gaussian, trajectory.states, dt, events
        ),
        "ordinary_imm": evaluate_forecasts(
            ordinary, trajectory.states, dt, events
        ),
        "change_aware_imm": evaluate_forecasts(
            change, trajectory.states, dt, events
        ),
    }
    detection = evaluate_change_detections(
        change_predictor.detection_events,
        trajectory.times,
        trajectory.change_flags,
        warmup_s=warmup_s,
        maximum_match_delay_s=float(
            change_config["detection_evaluation"][
                "maximum_match_delay_s"
            ]
        ),
    )
    ordinary_replay = _run_records(
        _imm_predictor(ordinary_config, profile.observation_std),
        trajectory,
        1,
        dt,
        stride,
        warmup,
    )
    replay_change_predictor = _change_predictor(
        change_config,
        ordinary_config,
        profile.observation_std,
        FROZEN_CHANGE_CANDIDATE,
    )
    change_replay = _run_records(
        replay_change_predictor,
        trajectory,
        1,
        dt,
        stride,
        warmup,
    )
    replays = {
        "ordinary_imm": (
            _filtered_digest(ordinary, imm=True)
            == _filtered_digest(ordinary_replay, imm=True)
        ),
        "change_aware_imm": (
            _filtered_digest(change, imm=True)
            == _filtered_digest(change_replay, imm=True)
            and change_predictor.detection_events
            == replay_change_predictor.detection_events
        ),
    }
    audits = {
        "gaussian_cv_kalman": covariance_audit(gaussian),
        "ordinary_imm": _imm_numerical_audit(ordinary),
        "change_aware_imm": _imm_numerical_audit(change),
    }
    gaussian_valid = all(
        bool(audits["gaussian_cv_kalman"][name])
        for name in ("finite", "symmetric", "positive_semidefinite")
    )
    imm_valid = all(
        bool(audits[predictor][name])
        for predictor in ("ordinary_imm", "change_aware_imm")
        for name in (
            "finite",
            "positive_semidefinite",
            "symmetric",
            "probabilities_valid",
        )
    )
    invariants_pass = bool(
        _trajectory_invariants(trajectory_audit)
        and gaussian_valid
        and imm_valid
        and all(replays.values())
    )
    metrics = {
        "schema_version": 1,
        "split": str(job["split"]),
        "process": str(job["process"]),
        "noise_profile": profile.name,
        "seed": int(job["seed"]),
        "predictors": predictor_metrics,
        "detection": {
            "change_aware_imm": detection,
            "ordinary_imm": None,
        },
        "trajectory_audit": trajectory_audit,
        "numerical_audit": audits,
        "filter_replay_exact": replays,
        "invariants_pass": invariants_pass,
    }
    records = {
        "deterministic_cv": deterministic,
        "gaussian_cv_kalman": gaussian,
        "ordinary_imm": ordinary,
        "change_aware_imm": change,
    }
    traces = []
    for predictor, forecast_records in records.items():
        traces.extend(
            _trace_rows(
                predictor, forecast_records, trajectory.states, dt
            )
        )
    run_dir = (
        Path(output_dir)
        / str(job["split"])
        / "runs"
        / str(job["process"])
        / profile.name
        / f"seed_{int(job['seed'])}"
    )
    _write_run(
        run_dir,
        trajectory,
        config,
        ordinary_config,
        change_config,
        obstacle_config,
        job,
        metrics,
        traces,
        change_predictor.detection_events,
        provenance_snapshot,
    )
    progress = {
        "run_order": int(job["run_order"]),
        "experimental_key": str(job["experimental_key"]),
        "split": str(job["split"]),
        "process": str(job["process"]),
        "noise_profile": profile.name,
        "seed": int(job["seed"]),
        "gaussian_ade_m": predictor_metrics["gaussian_cv_kalman"][
            "overall"
        ]["ade_m"],
        "gaussian_nll": predictor_metrics["gaussian_cv_kalman"][
            "overall"
        ]["nll"],
        "ordinary_ade_m": predictor_metrics["ordinary_imm"]["overall"][
            "ade_m"
        ],
        "ordinary_nll": predictor_metrics["ordinary_imm"]["overall"][
            "nll"
        ],
        "change_ade_m": predictor_metrics["change_aware_imm"][
            "overall"
        ]["ade_m"],
        "change_nll": predictor_metrics["change_aware_imm"][
            "overall"
        ]["nll"],
        "change_coverage_90": predictor_metrics["change_aware_imm"][
            "overall"
        ]["coverage_90"],
        "nis_trigger_count": detection["nis_trigger_count"],
        "invariants_pass": int(invariants_pass),
        "run_dir": str(run_dir.relative_to(output_dir)),
    }
    return {"progress": progress, "metrics": metrics}


def _run_jobs(jobs, config_path, output_dir, provenance_snapshot, workers):
    rows = []
    if int(workers) == 1:
        for job in jobs:
            rows.append(
                _run_one_worker(
                    job,
                    config_path,
                    output_dir,
                    provenance_snapshot,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {
                executor.submit(
                    _run_one_worker,
                    job,
                    config_path,
                    output_dir,
                    provenance_snapshot,
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: row["progress"]["run_order"])
    return rows


def _seed_level_nll_differences(rows):
    by_seed = {}
    for row in rows:
        if row["progress"]["process"] not in CHANGE_PROCESSES:
            continue
        seed = int(row["progress"]["seed"])
        ordinary = row["metrics"]["predictors"]["ordinary_imm"][
            "overall"
        ]["nll"]
        change = row["metrics"]["predictors"]["change_aware_imm"][
            "overall"
        ]["nll"]
        by_seed.setdefault(seed, []).append(float(change - ordinary))
    if any(len(values) != 6 for values in by_seed.values()):
        raise ValueError("each seed must contain P2--P4 x two profiles")
    return {
        seed: float(np.mean(values))
        for seed, values in sorted(by_seed.items())
    }


def _split_summary(rows):
    predictors = {
        name: _aggregate_predictor(rows, name)
        for name in PREDICTOR_NAMES
    }
    detection_p1 = _aggregate_detection(
        rows, "change_aware_imm", ("stochastic_cruise",)
    )
    detection_change = _aggregate_detection(
        rows, "change_aware_imm", CHANGE_PROCESSES
    )
    return {
        "run_count": len(rows),
        "predictors": predictors,
        "detection_p1": detection_p1,
        "detection_change_processes": detection_change,
    }


def _plot_summary(summary, seed_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "gaussian_cv_kalman": "#009E73",
        "ordinary_imm": "#0072B2",
        "change_aware_imm": "#D55E00",
    }
    labels = {
        "gaussian_cv_kalman": "Gaussian CV",
        "ordinary_imm": "ordinary IMM",
        "change_aware_imm": "change-aware IMM",
    }
    domains = ("held_out_id", "held_out_ood")
    domain_labels = ("ID", "OOD")
    predictors = tuple(colors)
    figure, axes = plt.subplots(
        2, 2, figsize=(12.0, 8.3), constrained_layout=True
    )
    x = np.arange(2)
    width = 0.24
    for offset, predictor in enumerate(predictors):
        nll = [
            summary[domain]["predictors"][predictor][
                "change_process_macro"
            ]["nll"]
            for domain in domains
        ]
        coverage = [
            summary[domain]["predictors"][predictor][
                "change_process_macro"
            ]["coverage_90"]
            for domain in domains
        ]
        axes[0, 0].bar(
            x + (offset - 1) * width,
            nll,
            width,
            color=colors[predictor],
            label=labels[predictor],
        )
        axes[0, 1].bar(
            x + (offset - 1) * width,
            coverage,
            width,
            color=colors[predictor],
            label=labels[predictor],
        )
    axes[0, 0].set_xticks(x, domain_labels)
    axes[0, 0].set_ylabel("P2--P4 exact mixture NLL")
    axes[0, 0].set_title("Probability quality")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_xticks(x, domain_labels)
    axes[0, 1].set_ylim(0.0, 1.02)
    axes[0, 1].set_ylabel("P2--P4 90% coverage")
    axes[0, 1].set_title("Calibration")
    axes[0, 1].axhline(
        0.90, color="#222222", linestyle="--", linewidth=1.2
    )
    axes[0, 1].legend(fontsize=8)

    positions = np.arange(10)
    for domain, color, marker in (
        ("held_out_id", "#0072B2", "o"),
        ("held_out_ood", "#D55E00", "s"),
    ):
        values = [
            row["nll_difference"]
            for row in seed_rows
            if row["split"] == domain
        ]
        axes[1, 0].plot(
            positions,
            values,
            marker=marker,
            linewidth=1.5,
            color=color,
            label=domain.replace("held_out_", "").upper(),
        )
    axes[1, 0].axhline(0.0, color="#222222", linewidth=1.0)
    axes[1, 0].set_xticks(positions, [str(i + 1) for i in positions])
    axes[1, 0].set_xlabel("Held-out seed index")
    axes[1, 0].set_ylabel("Change-aware minus ordinary NLL")
    axes[1, 0].set_title("Paired seed-level differences")
    axes[1, 0].legend(fontsize=8)

    windows = ("post_change_0_1", "post_change_1_3")
    window_labels = ("post 0--1 s", "post 1--3 s")
    for domain, color, marker in (
        ("held_out_id", "#0072B2", "o"),
        ("held_out_ood", "#D55E00", "s"),
    ):
        ratios = [
            summary[domain]["predictors"]["change_aware_imm"][
                "event_windows_change_processes"
            ][window]["nll"]
            / summary[domain]["predictors"]["ordinary_imm"][
                "event_windows_change_processes"
            ][window]["nll"]
            for window in windows
        ]
        axes[1, 1].plot(
            window_labels,
            ratios,
            marker=marker,
            linewidth=2.0,
            color=color,
            label=domain.replace("held_out_", "").upper(),
        )
    axes[1, 1].axhline(
        1.0, color="#222222", linestyle="--", linewidth=1.2
    )
    axes[1, 1].set_ylabel("NLL ratio to ordinary IMM")
    axes[1, 1].set_title("Post-change recovery")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle(
        "Frozen predictor qualification on held-out ID and OOD",
        fontsize=15,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path, dpi=220, bbox_inches="tight", facecolor="white"
    )
    figure.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return output_path


def _validate_config(config, registry):
    if config["platform"] != "windows_native_only":
        raise ValueError("held-out study requires native Windows")
    guards = config["scope_guards"]
    if not all(
        bool(guards[name])
        for name in (
            "id_v3_frozen",
            "ood_shift_frozen",
            "predictor_parameters_frozen",
            "held_out_only",
        )
    ):
        raise ValueError("held-out prerequisite guard is disabled")
    if any(
        bool(guards[name])
        for name in (
            "sealed_registry_imported",
            "collision_risk_enabled",
            "mppi_enabled",
            "rl_icode_hss_enabled",
        )
    ):
        raise ValueError("held-out scope guard was violated")
    validate_split_registry(registry)
    if tuple(config["processes"]) != V3_PROCESS_NAMES:
        raise ValueError("held-out processes changed")
    for name in ("held_out_id", "held_out_ood"):
        declared = registry["splits"][name]
        split = config["splits"][name]
        if int(split["seed_count"]) != 10:
            raise ValueError("held-out seed count changed")
        if int(split["seed_start"]) != int(declared["seed_start"]):
            raise ValueError("held-out seed namespace mismatch")


def run_qualification(config_path=DEFAULT_CONFIG, output_override=None, workers=None):
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    registry = _load_yaml(_resolve(config["seed_registry_path"]))
    _validate_config(config, registry)
    output_dir = (
        Path(output_override).resolve()
        if output_override is not None
        else _resolve(config["output_dir"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_schedule(config)
    if len(jobs) != 160:
        raise ValueError("held-out schedule must contain 160 trajectories")
    provenance_snapshot = repository_provenance(ROOT)
    write_yaml(
        output_dir / "manifest.yaml",
        {
            "schema_version": 1,
            "study_id": config["study_id"],
            "stage": config["stage"],
            "platform": config["platform"],
            "config_path": str(config_path.relative_to(ROOT)),
            "expected_runs": 160,
            "expected_runs_per_split": 80,
            "predictors": list(PREDICTOR_NAMES),
            "frozen_change_candidate": FROZEN_CHANGE_CANDIDATE,
            "scope_guards": dict(config["scope_guards"]),
        },
    )
    write_json(
        output_dir / "schedule.json",
        {"schedule_seed": config["schedule_seed"], "jobs": jobs},
    )
    write_json(
        output_dir / "environment_manifest.json",
        environment_manifest(ROOT),
    )
    worker_count = int(
        config["worker_count"] if workers is None else workers
    )
    rows = _run_jobs(
        jobs,
        config_path,
        output_dir,
        provenance_snapshot,
        worker_count,
    )
    write_csv(
        output_dir / "progress.csv",
        [row["progress"] for row in rows],
    )
    split_rows = {
        split: [
            row for row in rows if row["progress"]["split"] == split
        ]
        for split in ("held_out_id", "held_out_ood")
    }
    split_summary = {
        split: _split_summary(values)
        for split, values in split_rows.items()
    }
    seed_rows = []
    statistics = {}
    stats_config = config["statistics"]
    for split_index, split in enumerate(
        ("held_out_id", "held_out_ood")
    ):
        differences = _seed_level_nll_differences(split_rows[split])
        for seed, value in differences.items():
            seed_rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "nll_difference": value,
                }
            )
        statistics[split] = paired_seed_summary(
            list(differences.values()),
            bootstrap_seed=int(stats_config["bootstrap_seed"])
            + split_index,
            bootstrap_resamples=int(
                stats_config["bootstrap_resamples"]
            ),
        )
    adjusted = holm_adjust(
        {
            split: result["exact_sign_flip_p"]
            for split, result in statistics.items()
        }
    )
    for split in statistics:
        statistics[split]["holm_adjusted_p"] = adjusted[split]
    write_csv(output_dir / "analysis/seed_level_nll.csv", seed_rows)

    gate = config["gate"]
    ratios = {}
    for split in ("held_out_id", "held_out_ood"):
        predictors = split_summary[split]["predictors"]
        change = predictors["change_aware_imm"]
        ordinary = predictors["ordinary_imm"]
        gaussian = predictors["gaussian_cv_kalman"]
        ratios[split] = {
            "change_nll_to_ordinary": (
                change["change_process_macro"]["nll"]
                / ordinary["change_process_macro"]["nll"]
            ),
            "change_nll_to_gaussian": (
                change["change_process_macro"]["nll"]
                / gaussian["change_process_macro"]["nll"]
            ),
            "change_coverage_to_ordinary": (
                change["change_process_macro"]["coverage_90"]
                / ordinary["change_process_macro"]["coverage_90"]
            ),
            "overall_ade_to_ordinary": (
                change["macro"]["ade_m"]
                / ordinary["macro"]["ade_m"]
            ),
            "post_change_1_3_nll_to_ordinary": (
                change["event_windows_change_processes"][
                    "post_change_1_3"
                ]["nll"]
                / ordinary["event_windows_change_processes"][
                    "post_change_1_3"
                ]["nll"]
            ),
        }
    identifier = split_summary["held_out_id"]
    id_change = identifier["predictors"]["change_aware_imm"]
    id_ordinary = identifier["predictors"]["ordinary_imm"]
    checks = {
        "run_count_complete": (
            len(split_rows["held_out_id"]) == 80
            and len(split_rows["held_out_ood"]) == 80
        ),
        "unique_keys": len(
            {row["progress"]["experimental_key"] for row in rows}
        )
        == 160,
        "all_invariants_pass": all(
            row["metrics"]["invariants_pass"] for row in rows
        ),
        "id_statistical_superiority": (
            statistics["held_out_id"]["mean_difference"] < 0.0
            and statistics["held_out_id"]["bootstrap_95_ci"][1] < 0.0
            and statistics["held_out_id"]["holm_adjusted_p"]
            <= float(gate["maximum_holm_adjusted_p"])
        ),
        "ood_statistical_superiority": (
            statistics["held_out_ood"]["mean_difference"] < 0.0
            and statistics["held_out_ood"]["bootstrap_95_ci"][1] < 0.0
            and statistics["held_out_ood"]["holm_adjusted_p"]
            <= float(gate["maximum_holm_adjusted_p"])
        ),
        "id_nll_ratio_to_ordinary": (
            ratios["held_out_id"]["change_nll_to_ordinary"]
            <= float(
                gate["maximum_id_change_nll_ratio_to_ordinary"]
            )
        ),
        "ood_nll_ratio_to_ordinary": (
            ratios["held_out_ood"]["change_nll_to_ordinary"]
            <= float(
                gate["maximum_ood_change_nll_ratio_to_ordinary"]
            )
        ),
        "id_nll_ratio_to_gaussian": (
            ratios["held_out_id"]["change_nll_to_gaussian"]
            <= float(
                gate["maximum_id_change_nll_ratio_to_gaussian"]
            )
        ),
        "ood_nll_ratio_to_gaussian": (
            ratios["held_out_ood"]["change_nll_to_gaussian"]
            <= float(
                gate["maximum_ood_change_nll_ratio_to_gaussian"]
            )
        ),
        "id_coverage_90": (
            float(gate["minimum_id_macro_coverage_90"])
            <= id_change["macro"]["coverage_90"]
            <= float(gate["maximum_id_macro_coverage_90"])
        ),
        "id_coverage_95": (
            float(gate["minimum_id_macro_coverage_95"])
            <= id_change["macro"]["coverage_95"]
            <= float(gate["maximum_id_macro_coverage_95"])
        ),
        "coverage_not_lower": all(
            ratios[split]["change_coverage_to_ordinary"]
            >= float(
                gate["minimum_change_coverage_ratio_to_ordinary"]
            )
            for split in ratios
        ),
        "overall_ade_noninferior": all(
            ratios[split]["overall_ade_to_ordinary"]
            <= float(gate["maximum_overall_ade_ratio_to_ordinary"])
            for split in ratios
        ),
        "id_p1_ade_noninferior": (
            id_change["by_process"]["stochastic_cruise"]["ade_m"]
            / id_ordinary["by_process"]["stochastic_cruise"]["ade_m"]
            <= float(gate["maximum_id_p1_ade_ratio_to_ordinary"])
        ),
        "post_change_1_3_nll_lower": all(
            ratios[split]["post_change_1_3_nll_to_ordinary"] < 1.0
            for split in ratios
        ),
        "id_p1_false_trigger_rate": (
            identifier["detection_p1"]["false_triggers_per_minute"]
            <= float(
                gate["maximum_id_p1_false_triggers_per_minute"]
            )
        ),
    }
    qualification_pass = bool(all(checks.values()))
    summary = {
        "schema_version": 1,
        "split_summary": split_summary,
        "paired_statistics": statistics,
        "ratios": ratios,
        "gate_checks": checks,
        "predictor_qualification_pass": qualification_pass,
    }
    summary_path = write_json(
        output_dir / "analysis/summary.json", summary
    )
    figure_path = _plot_summary(
        split_summary,
        seed_rows,
        output_dir / "figures/held_out_predictor_qualification.png",
    )
    analysis_md = output_dir / "analysis/summary.md"
    analysis_md.write_text(
        "# Held-out predictor qualification summary\n\n"
        f"- Runs: {len(rows)}/160\n"
        f"- ID seed-level NLL difference: "
        f"{statistics['held_out_id']['mean_difference']:.6f}, "
        f"95% bootstrap CI "
        f"[{statistics['held_out_id']['bootstrap_95_ci'][0]:.6f}, "
        f"{statistics['held_out_id']['bootstrap_95_ci'][1]:.6f}]\n"
        f"- OOD seed-level NLL difference: "
        f"{statistics['held_out_ood']['mean_difference']:.6f}, "
        f"95% bootstrap CI "
        f"[{statistics['held_out_ood']['bootstrap_95_ci'][0]:.6f}, "
        f"{statistics['held_out_ood']['bootstrap_95_ci'][1]:.6f}]\n"
        f"- ID macro 90% coverage: "
        f"{id_change['macro']['coverage_90']:.6f}\n"
        f"- ID macro 95% coverage: "
        f"{id_change['macro']['coverage_95']:.6f}\n"
        f"- Predictor Qualification: {qualification_pass}\n",
        encoding="utf-8",
    )
    integrity_paths = (
        output_dir / "manifest.yaml",
        output_dir / "schedule.json",
        output_dir / "environment_manifest.json",
        output_dir / "progress.csv",
        output_dir / "analysis/seed_level_nll.csv",
        summary_path,
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
            "predictor_qualification_pass": qualification_pass,
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
    summary = run_qualification(
        args.config, args.output_dir, workers=args.workers
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
