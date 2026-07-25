"""Run the preregistered Change-Aware IMM pilot and confirmation study."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

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
from mobile_robot_mppi.obstacles.change_detection_evaluation import (
    evaluate_change_detections,
)
from mobile_robot_mppi.obstacles.imm import (
    ChangeAwareIMMPredictor,
    run_online_imm_forecasts,
)
from mobile_robot_mppi.obstacles.motion import noise_profiles_from_mapping
from mobile_robot_mppi.obstacles.patrol import (
    V3_PROCESS_NAMES,
    generate_patrol_trajectory,
)
from mobile_robot_mppi.obstacles.predictor_evaluation import (
    evaluate_forecasts,
)

try:
    from .run_ordinary_imm_v3_development import (
        _filtered_digest,
        _imm_numerical_audit,
        _imm_predictor,
        _load_yaml,
        _resolve,
        _trace_rows,
    )
except ImportError:
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
    ROOT / "configs/research/change_aware_imm_v3_development.yaml"
)
CHANGE_PROCESSES = V3_PROCESS_NAMES[1:]


def build_schedule(processes, noise_profiles, seeds, schedule_seed, phase):
    if tuple(processes) != V3_PROCESS_NAMES:
        raise ValueError("change-aware study requires every frozen V3 process")
    if tuple(noise_profiles) != ("low", "medium"):
        raise ValueError("change-aware study requires low/medium noise")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("change-aware schedule seeds must be unique")
    jobs = [
        {
            "phase": str(phase),
            "process": process,
            "noise_profile": noise,
            "seed": seed,
        }
        for process in processes
        for noise in noise_profiles
        for seed in seeds
    ]
    permutation = np.random.RandomState(int(schedule_seed)).permutation(
        len(jobs)
    )
    scheduled = []
    for run_order, source_index in enumerate(permutation):
        job = dict(jobs[int(source_index)])
        job["run_order"] = int(run_order)
        job["experimental_key"] = (
            f"{phase}::{job['process']}::{job['noise_profile']}"
            f"::seed{job['seed']}"
        )
        scheduled.append(job)
    return scheduled


def _change_predictor(change_config, ordinary_config, observation_std, candidate):
    detector = change_config["pilot_candidates"][str(candidate)]
    response = change_config["change_response"]
    ordinary = ordinary_config["ordinary_imm"]
    return ChangeAwareIMMPredictor(
        observation_std=float(observation_std),
        initial_velocity_std=float(
            ordinary_config["prediction"]["initial_velocity_std"]
        ),
        initial_mode_probabilities=ordinary[
            "initial_mode_probabilities"
        ],
        transition_matrix=ordinary["transition_matrix"],
        turn_rate_radps=ordinary["turn_rate_radps"],
        brake_decay_rate_per_s=ordinary["brake_decay_rate_per_s"],
        process_acceleration_std=ordinary[
            "process_acceleration_std"
        ],
        nis_threshold=detector["nis_threshold"],
        required_exceedances=detector["required_exceedances"],
        window_observations=detector["window_observations"],
        single_exceedance_threshold=detector.get(
            "single_exceedance_threshold"
        ),
        reset_mode_probabilities=response[
            "reset_mode_probabilities"
        ],
        state_covariance_inflation=response[
            "state_covariance_inflation"
        ],
        recovery_process_noise_scale=response[
            "recovery_process_noise_scale"
        ],
        recovery_duration_s=response["recovery_duration_s"],
        refractory_period_s=response["refractory_period_s"],
        dropout_guard_after_s=response["dropout_guard_after_s"],
        dropout_covariance_inflation=response[
            "dropout_covariance_inflation"
        ],
    )


def _run_records(
    predictor,
    trajectory,
    horizon,
    dt,
    stride,
    warmup,
):
    return run_online_imm_forecasts(
        trajectory.times,
        trajectory.observations,
        trajectory.observed_mask,
        predictor,
        horizon=horizon,
        forecast_dt=dt,
        forecast_stride_steps=stride,
        warmup_steps=warmup,
    )


def _ordinary_frozen_metrics_match(job, ordinary_metrics, ordinary_config):
    prior_root = _resolve(ordinary_config["output_dir"])
    path = (
        prior_root
        / "runs"
        / str(job["process"])
        / str(job["noise_profile"])
        / f"seed_{int(job['seed'])}"
        / "metrics.json"
    )
    if not path.is_file():
        return None
    prior = json.loads(path.read_text(encoding="utf-8"))
    return prior["predictors"]["ordinary_imm"] == ordinary_metrics


def _predictor_result(
    records,
    predictor,
    trajectory,
    dt,
    warmup_s,
    maximum_match_delay_s,
):
    metrics = evaluate_forecasts(
        records,
        trajectory.states,
        dt,
        trajectory.metadata["events"],
    )
    numerical = _imm_numerical_audit(records)
    detection = (
        None
        if not isinstance(predictor, ChangeAwareIMMPredictor)
        else evaluate_change_detections(
            predictor.detection_events,
            trajectory.times,
            trajectory.change_flags,
            warmup_s=warmup_s,
            maximum_match_delay_s=maximum_match_delay_s,
        )
    )
    return metrics, numerical, detection


def _write_run_artifacts(
    run_dir,
    trajectory,
    change_config,
    ordinary_config,
    job,
    candidate_names,
    metrics,
    traces,
    detections,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": write_yaml(
            run_dir / "config_resolved.yaml",
            {
                "study_id": change_config["study_id"],
                "job": dict(job),
                "candidate_names": list(candidate_names),
                "prediction": dict(change_config["prediction"]),
                "change_response": dict(
                    change_config["change_response"]
                ),
                "candidate_parameters": {
                    name: dict(
                        change_config["pilot_candidates"][name]
                    )
                    for name in candidate_names
                },
                "ordinary_imm": dict(ordinary_config["ordinary_imm"]),
                "scope_guards": dict(change_config["scope_guards"]),
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
            {"predictors": detections},
        ),
        "metrics": write_json(run_dir / "metrics.json", metrics),
    }
    provenance = {
        "schema_version": 1,
        "stage": "offline_change_aware_predictor",
        **repository_provenance(ROOT),
        "files": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    write_json(run_dir / "provenance.json", provenance)


def _run_one_worker(
    job,
    config_path,
    output_dir,
    candidate_names,
):
    change_config = _load_yaml(config_path)
    ordinary_config = _load_yaml(
        _resolve(change_config["ordinary_config_path"])
    )
    obstacle_config = _load_yaml(
        _resolve(change_config["obstacle_config_path"])
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
    dt = float(obstacle_config["trajectory"]["dt"])
    horizon = int(
        round(float(change_config["prediction"]["horizon_s"]) / dt)
    )
    stride = int(change_config["prediction"]["forecast_stride_steps"])
    warmup_s = float(change_config["prediction"]["warmup_s"])
    warmup = int(round(warmup_s / dt))
    max_delay = float(
        change_config["detection_evaluation"][
            "maximum_match_delay_s"
        ]
    )
    ordinary_predictor = _imm_predictor(
        ordinary_config, profile.observation_std
    )
    ordinary_records = _run_records(
        ordinary_predictor,
        trajectory,
        horizon,
        dt,
        stride,
        warmup,
    )
    ordinary_metrics, ordinary_numerical, _ = _predictor_result(
        ordinary_records,
        ordinary_predictor,
        trajectory,
        dt,
        warmup_s,
        max_delay,
    )
    ordinary_replay = _run_records(
        _imm_predictor(ordinary_config, profile.observation_std),
        trajectory,
        1,
        dt,
        stride,
        warmup,
    )
    ordinary_replay_exact = _filtered_digest(
        ordinary_records, imm=True
    ) == _filtered_digest(ordinary_replay, imm=True)
    frozen_match = _ordinary_frozen_metrics_match(
        job, ordinary_metrics, ordinary_config
    )
    predictors = {"ordinary_imm": ordinary_metrics}
    numerical = {"ordinary_imm": ordinary_numerical}
    replay_exact = {"ordinary_imm": ordinary_replay_exact}
    detections = {"ordinary_imm": []}
    detection_metrics = {"ordinary_imm": None}
    records_by_name = {"ordinary_imm": ordinary_records}
    for candidate in candidate_names:
        predictor = _change_predictor(
            change_config,
            ordinary_config,
            profile.observation_std,
            candidate,
        )
        records = _run_records(
            predictor, trajectory, horizon, dt, stride, warmup
        )
        result, audit, detection = _predictor_result(
            records,
            predictor,
            trajectory,
            dt,
            warmup_s,
            max_delay,
        )
        replay_predictor = _change_predictor(
            change_config,
            ordinary_config,
            profile.observation_std,
            candidate,
        )
        replay = _run_records(
            replay_predictor, trajectory, 1, dt, stride, warmup
        )
        exact = (
            _filtered_digest(records, imm=True)
            == _filtered_digest(replay, imm=True)
            and predictor.detection_events
            == replay_predictor.detection_events
        )
        predictors[candidate] = result
        numerical[candidate] = audit
        replay_exact[candidate] = exact
        detections[candidate] = list(predictor.detection_events)
        detection_metrics[candidate] = detection
        records_by_name[candidate] = records
    invariant_flags = {}
    for name in ("ordinary_imm", *candidate_names):
        audit = numerical[name]
        invariant_flags[name] = bool(
            replay_exact[name]
            and audit["finite"]
            and audit["positive_semidefinite"]
            and audit["symmetric"]
            and audit["probabilities_valid"]
        )
    invariant_flags["ordinary_frozen_metrics_match"] = (
        frozen_match is not False
    )
    metrics = {
        "schema_version": 1,
        "phase": str(job["phase"]),
        "process": str(job["process"]),
        "noise_profile": profile.name,
        "seed": int(job["seed"]),
        "predictors": predictors,
        "detection": detection_metrics,
        "numerical_audit": numerical,
        "filter_replay_exact": replay_exact,
        "ordinary_frozen_metrics_match": frozen_match,
        "invariants_pass": bool(all(invariant_flags.values())),
        "invariant_flags": invariant_flags,
    }
    traces = []
    for name, records in records_by_name.items():
        traces.extend(_trace_rows(name, records, trajectory.states, dt))
    run_dir = (
        Path(output_dir)
        / str(job["phase"])
        / "runs"
        / str(job["process"])
        / profile.name
        / f"seed_{int(job['seed'])}"
    )
    _write_run_artifacts(
        run_dir,
        trajectory,
        change_config,
        ordinary_config,
        job,
        candidate_names,
        metrics,
        traces,
        detections,
    )
    progress = {
        "run_order": int(job["run_order"]),
        "experimental_key": str(job["experimental_key"]),
        "phase": str(job["phase"]),
        "process": str(job["process"]),
        "noise_profile": profile.name,
        "seed": int(job["seed"]),
        "ordinary_ade_m": ordinary_metrics["overall"]["ade_m"],
        "ordinary_nll": ordinary_metrics["overall"]["nll"],
        "ordinary_coverage_90": ordinary_metrics["overall"][
            "coverage_90"
        ],
        "ordinary_frozen_metrics_match": (
            -1 if frozen_match is None else int(frozen_match)
        ),
        "invariants_pass": int(metrics["invariants_pass"]),
        "run_dir": str(run_dir.relative_to(output_dir)),
    }
    for candidate in candidate_names:
        progress[f"{candidate}_ade_m"] = predictors[candidate][
            "overall"
        ]["ade_m"]
        progress[f"{candidate}_nll"] = predictors[candidate]["overall"][
            "nll"
        ]
        progress[f"{candidate}_coverage_90"] = predictors[candidate][
            "overall"
        ]["coverage_90"]
        progress[f"{candidate}_nis_triggers"] = detection_metrics[
            candidate
        ]["nis_trigger_count"]
    return {"progress": progress, "metrics": metrics}


def _run_jobs(
    jobs,
    config_path,
    output_dir,
    candidate_names,
    workers,
):
    rows = []
    if int(workers) == 1:
        for job in jobs:
            rows.append(
                _run_one_worker(
                    job, config_path, output_dir, candidate_names
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
                    candidate_names,
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: row["progress"]["run_order"])
    return rows


def _mean_present(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _metric(
    rows,
    predictor,
    section,
    metric,
    processes=V3_PROCESS_NAMES,
):
    values = []
    for row in rows:
        if row["progress"]["process"] not in processes:
            continue
        value = row["metrics"]["predictors"][predictor]
        for key in section:
            value = value[key]
        values.append(value[metric])
    return _mean_present(values)


def _aggregate_predictor(rows, predictor):
    by_process = {}
    metric_names = (
        "ade_m",
        "fde_3s_m",
        "nll",
        "coverage_50",
        "coverage_90",
        "coverage_95",
        "mean_95_area_m2",
        "mean_mixture_spread_m",
    )
    for process in V3_PROCESS_NAMES:
        by_process[process] = {
            metric: _metric(
                rows, predictor, ("overall",), metric, (process,)
            )
            for metric in metric_names
        }
    macro = {
        metric: _mean_present(
            by_process[process][metric] for process in V3_PROCESS_NAMES
        )
        for metric in metric_names
    }
    change_macro = {
        metric: _mean_present(
            by_process[process][metric] for process in CHANGE_PROCESSES
        )
        for metric in metric_names
    }
    event_windows = {}
    for window in (
        "pre_change",
        "post_change_0_1",
        "post_change_1_3",
        "steady",
    ):
        event_windows[window] = {
            metric: _metric(
                rows,
                predictor,
                ("event_windows", window),
                metric,
                CHANGE_PROCESSES,
            )
            for metric in ("ade_m", "nll", "coverage_90")
        }
    observation_status = {}
    for status in ("observation_available", "observation_missing"):
        observation_status[status] = {
            metric: _metric(
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
        "change_process_macro": change_macro,
        "event_windows_change_processes": event_windows,
        "observation_status": observation_status,
    }


def _aggregate_detection(rows, predictor, processes):
    selected = [
        row
        for row in rows
        if row["progress"]["process"] in processes
    ]
    metrics = [
        row["metrics"]["detection"][predictor] for row in selected
    ]
    true_changes = sum(item["true_change_count"] for item in metrics)
    matched = sum(item["matched_trigger_count"] for item in metrics)
    false_triggers = sum(item["false_trigger_count"] for item in metrics)
    delays = [
        float(match["delay_s"])
        for item in metrics
        for match in item["matches"]
    ]
    duration_minutes = len(metrics) * (45.0 - 1.0) / 60.0
    return {
        "trajectory_count": len(metrics),
        "true_change_count": true_changes,
        "nis_trigger_count": sum(
            item["nis_trigger_count"] for item in metrics
        ),
        "dropout_guard_count": sum(
            item["dropout_guard_count"] for item in metrics
        ),
        "matched_trigger_count": matched,
        "false_trigger_count": false_triggers,
        "false_triggers_per_minute": float(
            false_triggers / duration_minutes
        ),
        "true_change_recall": (
            None if true_changes == 0 else float(matched / true_changes)
        ),
        "median_detection_delay_s": (
            None if not delays else float(np.median(delays))
        ),
        "p90_detection_delay_s": (
            None if not delays else float(np.quantile(delays, 0.90))
        ),
    }


def _pilot_summary(rows, config):
    ordinary = _aggregate_predictor(rows, "ordinary_imm")
    candidates = {}
    limits = config["pilot_feasibility"]
    for candidate in config["pilot_candidates"]:
        aggregate = _aggregate_predictor(rows, candidate)
        detection_p1 = _aggregate_detection(
            rows, candidate, ("stochastic_cruise",)
        )
        detection_change = _aggregate_detection(
            rows, candidate, CHANGE_PROCESSES
        )
        p1_ratio = (
            aggregate["by_process"]["stochastic_cruise"]["ade_m"]
            / ordinary["by_process"]["stochastic_cruise"]["ade_m"]
        )
        area_ratio = (
            aggregate["macro"]["mean_95_area_m2"]
            / ordinary["macro"]["mean_95_area_m2"]
        )
        post_nll_score = float(
            np.mean(
                (
                    aggregate["event_windows_change_processes"][
                        "post_change_0_1"
                    ]["nll"],
                    aggregate["event_windows_change_processes"][
                        "post_change_1_3"
                    ]["nll"],
                )
            )
        )
        checks = {
            "all_invariants_pass": all(
                row["metrics"]["invariants_pass"] for row in rows
            ),
            "p1_false_trigger_rate": (
                detection_p1["false_triggers_per_minute"]
                <= float(
                    limits["maximum_p1_false_triggers_per_minute"]
                )
            ),
            "p1_ade_ratio": (
                p1_ratio
                <= float(limits["maximum_p1_ade_ratio_to_ordinary"])
            ),
            "area_ratio": (
                area_ratio
                <= float(limits["maximum_area_ratio_to_ordinary"])
            ),
        }
        candidates[candidate] = {
            "aggregate": aggregate,
            "detection_p1": detection_p1,
            "detection_change_processes": detection_change,
            "p1_ade_ratio_to_ordinary": float(p1_ratio),
            "area_ratio_to_ordinary": float(area_ratio),
            "post_change_nll_selection_score": post_nll_score,
            "feasibility_checks": checks,
            "feasible": bool(all(checks.values())),
        }
    feasible = [
        name for name, result in candidates.items() if result["feasible"]
    ]
    selected = None
    if feasible:
        selected = min(
            feasible,
            key=lambda name: (
                round(
                    candidates[name][
                        "post_change_nll_selection_score"
                    ],
                    9,
                ),
                candidates[name]["detection_p1"][
                    "false_triggers_per_minute"
                ],
                name,
            ),
        )
    return {
        "ordinary_imm": ordinary,
        "candidates": candidates,
        "selected_candidate": selected,
        "pilot_pass": selected is not None,
    }


def _confirmation_summary(rows, selected, config):
    ordinary = _aggregate_predictor(rows, "ordinary_imm")
    change = _aggregate_predictor(rows, selected)
    detection_p1 = _aggregate_detection(
        rows, selected, ("stochastic_cruise",)
    )
    detection_change = _aggregate_detection(
        rows, selected, CHANGE_PROCESSES
    )
    ratios = {
        "p1_ade": (
            change["by_process"]["stochastic_cruise"]["ade_m"]
            / ordinary["by_process"]["stochastic_cruise"]["ade_m"]
        ),
        "change_process_nll": (
            change["change_process_macro"]["nll"]
            / ordinary["change_process_macro"]["nll"]
        ),
        "post_change_0_1_nll": (
            change["event_windows_change_processes"][
                "post_change_0_1"
            ]["nll"]
            / ordinary["event_windows_change_processes"][
                "post_change_0_1"
            ]["nll"]
        ),
        "post_change_1_3_nll": (
            change["event_windows_change_processes"][
                "post_change_1_3"
            ]["nll"]
            / ordinary["event_windows_change_processes"][
                "post_change_1_3"
            ]["nll"]
        ),
        "change_process_coverage_90": (
            change["change_process_macro"]["coverage_90"]
            / ordinary["change_process_macro"]["coverage_90"]
        ),
        "area": (
            change["macro"]["mean_95_area_m2"]
            / ordinary["macro"]["mean_95_area_m2"]
        ),
    }
    gate = config["confirmation_gate"]
    checks = {
        "run_count_complete": len(rows) == 56,
        "unique_keys": (
            len(
                {
                    row["progress"]["experimental_key"]
                    for row in rows
                }
            )
            == 56
        ),
        "all_invariants_pass": all(
            row["metrics"]["invariants_pass"] for row in rows
        ),
        "ordinary_frozen_metrics_match": all(
            row["metrics"]["ordinary_frozen_metrics_match"] is not False
            for row in rows
        ),
        "p1_ade_ratio": (
            ratios["p1_ade"]
            <= float(gate["maximum_p1_ade_ratio_to_ordinary"])
        ),
        "change_process_nll_ratio": (
            ratios["change_process_nll"]
            <= float(
                gate[
                    "maximum_change_process_nll_ratio_to_ordinary"
                ]
            )
        ),
        "post_change_0_1_nll_ratio": (
            ratios["post_change_0_1_nll"]
            <= float(
                gate["maximum_post_change_nll_ratio_to_ordinary"]
            )
        ),
        "post_change_1_3_nll_ratio": (
            ratios["post_change_1_3_nll"]
            <= float(
                gate["maximum_post_change_nll_ratio_to_ordinary"]
            )
        ),
        "change_process_coverage_ratio": (
            ratios["change_process_coverage_90"]
            >= float(
                gate[
                    "minimum_change_process_coverage_ratio_to_ordinary"
                ]
            )
        ),
        "area_ratio": (
            ratios["area"]
            <= float(gate["maximum_area_ratio_to_ordinary"])
        ),
        "p1_false_trigger_rate": (
            detection_p1["false_triggers_per_minute"]
            <= float(
                gate["maximum_p1_false_triggers_per_minute"]
            )
        ),
        "true_change_recall": (
            detection_change["true_change_recall"]
            >= float(gate["minimum_true_change_recall"])
        ),
        "median_detection_delay": (
            detection_change["median_detection_delay_s"] is not None
            and detection_change["median_detection_delay_s"]
            <= float(gate["maximum_median_detection_delay_s"])
        ),
    }
    return {
        "selected_candidate": selected,
        "ordinary_imm": ordinary,
        "change_aware_imm": change,
        "detection_p1": detection_p1,
        "detection_change_processes": detection_change,
        "ratios_to_ordinary": ratios,
        "gate_checks": checks,
        "predictor_gate_pass": bool(all(checks.values())),
    }


def _plot_confirmation(summary, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "ordinary_imm": "#0072B2",
        "change_aware_imm": "#D55E00",
    }
    labels = {"ordinary_imm": "ordinary IMM", "change_aware_imm": "change-aware IMM"}
    figure, axes = plt.subplots(
        2, 2, figsize=(12.0, 8.3), constrained_layout=True
    )
    x = np.arange(4)
    process_labels = ("P1", "P2", "P3", "P4")
    for offset, name in enumerate(("ordinary_imm", "change_aware_imm")):
        values = [
            summary[name]["by_process"][process]["ade_m"]
            for process in V3_PROCESS_NAMES
        ]
        axes[0, 0].bar(
            x + (offset - 0.5) * 0.34,
            values,
            0.34,
            color=colors[name],
            label=labels[name],
        )
        coverage = [
            summary[name]["by_process"][process]["coverage_90"]
            for process in V3_PROCESS_NAMES
        ]
        axes[0, 1].bar(
            x + (offset - 0.5) * 0.34,
            coverage,
            0.34,
            color=colors[name],
            label=labels[name],
        )
    axes[0, 0].set_xticks(x, process_labels)
    axes[0, 0].set_ylabel("ADE [m]")
    axes[0, 0].set_title("Position error by V3 process")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_xticks(x, process_labels)
    axes[0, 1].set_ylim(0.0, 1.02)
    axes[0, 1].set_ylabel("Moment-matched 90% coverage")
    axes[0, 1].set_title("Calibration by V3 process")
    axes[0, 1].axhline(
        0.90, color="#222222", linestyle="--", linewidth=1.2
    )
    axes[0, 1].legend(fontsize=8)

    windows = (
        "pre_change",
        "post_change_0_1",
        "post_change_1_3",
        "steady",
    )
    window_labels = ("pre -2--0", "post 0--1", "post 1--3", "steady")
    for name in ("ordinary_imm", "change_aware_imm"):
        axes[1, 0].plot(
            window_labels,
            [
                summary[name]["event_windows_change_processes"][
                    window
                ]["nll"]
                for window in windows
            ],
            marker="o",
            linewidth=2.0,
            color=colors[name],
            label=labels[name],
        )
    axes[1, 0].set_ylabel("Exact mixture NLL")
    axes[1, 0].set_title("Probability quality around true changes")
    axes[1, 0].legend(fontsize=8)

    detection = summary["detection_change_processes"]
    recall_by_process = []
    for process in CHANGE_PROCESSES:
        # The aggregate summary does not retain per-process detection; this
        # panel reports the preregistered aggregate and its delay.
        recall_by_process.append(detection["true_change_recall"])
    axes[1, 1].bar(
        ("P2--P4 recall", "median delay / 1.5 s"),
        (
            detection["true_change_recall"],
            detection["median_detection_delay_s"] / 1.5,
        ),
        color=("#009E73", "#CC79A7"),
    )
    axes[1, 1].set_ylim(0.0, 1.02)
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].set_title("Causal NIS detection summary")
    for axis in axes.flat:
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=10)
    figure.suptitle(
        "Change-Aware IMM confirmation on frozen V3 obstacles",
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


def _validate_config(config, ordinary_config, registry):
    if config["platform"] != "windows_native_only":
        raise ValueError("change-aware study is restricted to native Windows")
    guards = config["scope_guards"]
    if not all(
        bool(guards[name])
        for name in (
            "obstacle_v3_frozen",
            "ordinary_imm_frozen",
            "development_seeds_only",
            "change_aware_enabled",
        )
    ):
        raise ValueError("change-aware prerequisite guard is disabled")
    if any(
        bool(guards[name])
        for name in (
            "sealed_registry_imported",
            "collision_risk_enabled",
            "mppi_enabled",
            "rl_icode_hss_enabled",
        )
    ):
        raise ValueError("change-aware scope guard was violated")
    validate_split_registry(registry)
    development = registry["splits"]["development"]
    allowed = set(
        range(
            int(development["seed_start"]),
            int(development["seed_start"]) + int(development["count"]),
        )
    )
    pilot = tuple(int(seed) for seed in config["pilot_seeds"])
    confirmation = tuple(
        int(seed) for seed in config["confirmation_seeds"]
    )
    if len(pilot) != 3 or len(confirmation) != 7:
        raise ValueError("change-aware pilot/confirmation sizes changed")
    if set(pilot).intersection(confirmation):
        raise ValueError("pilot and confirmation seeds overlap")
    if not set(pilot + confirmation).issubset(allowed):
        raise ValueError("change-aware seed is outside development")
    if ordinary_config["scope_guards"]["change_aware_enabled"]:
        raise ValueError("ordinary baseline configuration is not frozen")


def run_development(config_path=DEFAULT_CONFIG, output_override=None, workers=None):
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    ordinary_config = _load_yaml(
        _resolve(config["ordinary_config_path"])
    )
    obstacle_config = _load_yaml(_resolve(config["obstacle_config_path"]))
    registry = _load_yaml(_resolve(config["seed_registry_path"]))
    _validate_config(config, ordinary_config, registry)
    if obstacle_config["generator_version"] != "recurrent_semimarkov_v3":
        raise ValueError("change-aware study requires frozen V3")
    output_dir = (
        Path(output_override).resolve()
        if output_override is not None
        else _resolve(config["output_dir"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_count = int(
        config["worker_count"] if workers is None else workers
    )
    candidates = tuple(config["pilot_candidates"].keys())
    pilot_jobs = build_schedule(
        config["processes"],
        config["noise_profiles"],
        config["pilot_seeds"],
        config["schedule_seed"],
        "pilot",
    )
    confirmation_jobs = build_schedule(
        config["processes"],
        config["noise_profiles"],
        config["confirmation_seeds"],
        int(config["schedule_seed"]) + 1,
        "confirmation",
    )
    write_yaml(
        output_dir / "manifest.yaml",
        {
            "schema_version": 1,
            "study_id": config["study_id"],
            "stage": config["stage"],
            "platform": config["platform"],
            "config_path": str(config_path.relative_to(ROOT)),
            "pilot_expected_runs": len(pilot_jobs),
            "confirmation_expected_runs": len(confirmation_jobs),
            "pilot_candidates": list(candidates),
            "scope_guards": dict(config["scope_guards"]),
        },
    )
    write_json(
        output_dir / "schedule.json",
        {
            "schedule_seed": config["schedule_seed"],
            "pilot_jobs": pilot_jobs,
            "confirmation_jobs": confirmation_jobs,
        },
    )
    write_json(
        output_dir / "environment_manifest.json",
        environment_manifest(ROOT),
    )
    pilot_rows = _run_jobs(
        pilot_jobs,
        config_path,
        output_dir,
        candidates,
        worker_count,
    )
    write_csv(
        output_dir / "pilot/progress.csv",
        [row["progress"] for row in pilot_rows],
    )
    pilot = _pilot_summary(pilot_rows, config)
    write_json(output_dir / "analysis/pilot_summary.json", pilot)
    selected = pilot["selected_candidate"]
    if selected is None:
        summary = {
            "schema_version": 1,
            "pilot": pilot,
            "confirmation": None,
            "predictor_gate_pass": False,
            "stop_reason": "no preregistered pilot candidate was feasible",
        }
        write_json(output_dir / "analysis/summary.json", summary)
        return summary
    confirmation_rows = _run_jobs(
        confirmation_jobs,
        config_path,
        output_dir,
        (selected,),
        worker_count,
    )
    write_csv(
        output_dir / "confirmation/progress.csv",
        [row["progress"] for row in confirmation_rows],
    )
    confirmation = _confirmation_summary(
        confirmation_rows, selected, config
    )
    summary = {
        "schema_version": 1,
        "pilot": pilot,
        "confirmation": confirmation,
        "predictor_gate_pass": confirmation["predictor_gate_pass"],
        "stop_reason": (
            None
            if confirmation["predictor_gate_pass"]
            else "confirmation gate failed; preserve artifacts"
        ),
    }
    summary_path = write_json(
        output_dir / "analysis/summary.json", summary
    )
    figure_path = _plot_confirmation(
        confirmation,
        output_dir / "figures/change_aware_confirmation.png",
    )
    analysis_md = output_dir / "analysis/summary.md"
    analysis_md.write_text(
        "# Change-Aware IMM development summary\n\n"
        f"- Selected pilot candidate: {selected}\n"
        f"- Confirmation runs: {len(confirmation_rows)}/56\n"
        f"- Change-process NLL ratio: "
        f"{confirmation['ratios_to_ordinary']['change_process_nll']:.6f}\n"
        f"- Post-change 0--1 s NLL ratio: "
        f"{confirmation['ratios_to_ordinary']['post_change_0_1_nll']:.6f}\n"
        f"- Post-change 1--3 s NLL ratio: "
        f"{confirmation['ratios_to_ordinary']['post_change_1_3_nll']:.6f}\n"
        f"- True-change recall: "
        f"{confirmation['detection_change_processes']['true_change_recall']:.6f}\n"
        f"- Median detection delay: "
        f"{confirmation['detection_change_processes']['median_detection_delay_s']:.6f} s\n"
        f"- Predictor Gate: {confirmation['predictor_gate_pass']}\n",
        encoding="utf-8",
    )
    integrity_paths = (
        output_dir / "manifest.yaml",
        output_dir / "schedule.json",
        output_dir / "environment_manifest.json",
        output_dir / "pilot/progress.csv",
        output_dir / "confirmation/progress.csv",
        output_dir / "analysis/pilot_summary.json",
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
            "predictor_gate_pass": confirmation[
                "predictor_gate_pass"
            ],
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
    result = run_development(
        args.config, args.output_dir, workers=args.workers
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
