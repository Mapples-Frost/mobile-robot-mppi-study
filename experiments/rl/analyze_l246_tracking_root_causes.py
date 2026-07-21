#!/usr/bin/env python3
"""Read-only L246 root-cause audit for the three frozen L244 Tracking runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


SCENES = {
    "hairpin": "tracking_l244_bc_anchor_hairpin_seed923301001",
    "s_chicane": "tracking_l244_bc_anchor_s_chicane_seed923301001",
    "infinity": "tracking_l244_bc_anchor_infinity_seed923301001",
}
EXPECTED_SEED = 923301001
EXPECTED_GIT_PREFIX = "edf9e1a"
EXPECTED_ACTOR_SHA256 = (
    "e8e446cbe9a9520a4053012a28995e63238a85dae4f0462e5574e2a0c0746c3c"
)
EPISODE_LIMIT = 700
STALL_WINDOW_STEPS = 50
STALL_MAX_ARC_GAIN_M = 0.05
SAFETY_DENSE_THRESHOLD = 0.5
PROGRESS_REGRESSION_THRESHOLD = -0.002
PROGRESS_JUMP_THRESHOLD = 0.03

NUMERIC_COLUMNS = (
    "time",
    "path_arc_length",
    "path_progress_ratio",
    "remaining_path_length",
    "signed_cross_track_error",
    "minimum_footprint_boundary_margin",
    "proposed_v",
    "executed_v",
    "applied_v",
    "proposed_omega",
    "executed_omega",
    "applied_omega",
    "safety_override",
    "reliability_proposal_authority",
    "reliability_proposal_fallback_fraction",
    "paper_guided_elite_count",
    "planner_compute_ms",
    "branch_id",
    "center_crossing_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_numeric_trajectory(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    missing = sorted(set(NUMERIC_COLUMNS) - set(rows[0]))
    if missing:
        raise ValueError(f"trajectory missing required columns {missing}: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in NUMERIC_COLUMNS
    }


def rolling_window_gain(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window < 2:
        raise ValueError("window must be at least 2")
    result = np.full(values.shape, np.nan, dtype=float)
    if values.size >= window:
        result[window - 1 :] = values[window - 1 :] - values[: -(window - 1)]
    return result


def rolling_window_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    if values.size >= window:
        kernel = np.ones(window, dtype=float) / float(window)
        result[window - 1 :] = np.convolve(values, kernel, mode="valid")
    return result


def first_true_index(mask: np.ndarray) -> int:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if indices.size else -1


def estimate_step_budget(
    path_length_m: float,
    control_dt_s: float,
    maximum_speed_mps: float,
    mean_forward_applied_speed_mps: float,
    observed_steps: int,
    final_completion_ratio: float,
) -> Dict[str, int]:
    if path_length_m <= 0.0 or control_dt_s <= 0.0 or maximum_speed_mps <= 0.0:
        raise ValueError("path length, dt and maximum speed must be positive")
    theoretical = int(math.ceil(path_length_m / (maximum_speed_mps * control_dt_s)))
    applied = (
        int(math.ceil(path_length_m / (mean_forward_applied_speed_mps * control_dt_s)))
        if mean_forward_applied_speed_mps > 0.0
        else -1
    )
    empirical = (
        int(math.ceil(observed_steps / final_completion_ratio))
        if final_completion_ratio > 0.0
        else -1
    )
    return {
        "theoretical_min_steps": theoretical,
        "applied_velocity_estimated_steps": applied,
        "empirical_progress_estimated_steps": empirical,
    }


def detect_projection_events(progress: np.ndarray) -> Dict[str, np.ndarray]:
    progress = np.asarray(progress, dtype=float)
    delta = np.diff(progress, prepend=progress[0])
    return {
        "delta": delta,
        "regression_indices": np.flatnonzero(delta < PROGRESS_REGRESSION_THRESHOLD),
        "jump_indices": np.flatnonzero(delta > PROGRESS_JUMP_THRESHOLD),
    }


def _find_single(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern} below {path}, found {matches}")
    return matches[0]


def locate_scene_inputs(repo_root: Path, scene: str) -> Dict[str, Path]:
    run_root = repo_root / "results" / "research_platform" / "rl" / SCENES[scene]
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    episode_root = _find_single(run_root / "runs" / "full_proposed", "*")
    return {
        "run_root": run_root,
        "trajectory": episode_root / "trajectory.csv",
        "metrics": episode_root / "metrics.json",
        "resolved_config": episode_root / "config_resolved.yaml",
        "episode_provenance": episode_root / "provenance.json",
        "run_provenance": run_root / "provenance.json",
    }


def verify_provenance(paths: Mapping[str, Path]) -> Dict[str, object]:
    for key, path in paths.items():
        if key != "run_root" and not path.is_file():
            raise FileNotFoundError(path)
    run_provenance = json.loads(paths["run_provenance"].read_text(encoding="utf-8"))
    episode_provenance = json.loads(
        paths["episode_provenance"].read_text(encoding="utf-8")
    )
    if run_provenance.get("status") != "pipeline_qualification":
        raise ValueError("L244 input is not a pipeline qualification run")
    if run_provenance.get("seeds") != [EXPECTED_SEED]:
        raise ValueError("unexpected development seed")
    actor_sha = run_provenance.get("coupled_actor_checkpoint", {}).get("sha256")
    if actor_sha != EXPECTED_ACTOR_SHA256:
        raise ValueError("unexpected coupled Actor checkpoint")
    git_sha = str(episode_provenance.get("git_sha", ""))
    if not git_sha.startswith(EXPECTED_GIT_PREFIX):
        raise ValueError(f"unexpected L244 Git SHA: {git_sha}")
    return {
        "status": run_provenance["status"],
        "seed": EXPECTED_SEED,
        "actor_sha256": actor_sha,
        "git_sha": git_sha,
        "manifest_sha256": run_provenance.get("manifest_sha256"),
    }


def analyze_scene(scene: str, paths: Mapping[str, Path]) -> Tuple[dict, dict, dict]:
    trajectory = load_numeric_trajectory(paths["trajectory"])
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    config = yaml.safe_load(paths["resolved_config"].read_text(encoding="utf-8"))
    time = trajectory["time"]
    steps = int(time.size)
    dt = float(np.median(np.diff(time))) if steps > 1 else float(
        config["experiment"]["control_dt"]
    )
    path_total = trajectory["path_arc_length"] + trajectory["remaining_path_length"]
    path_length = float(np.median(path_total))
    maximum_speed = float(config["action_space"]["upper"][0])
    mean_forward_speed = float(np.mean(np.maximum(trajectory["applied_v"], 0.0)))
    final_completion = float(trajectory["path_progress_ratio"][-1])
    budget = estimate_step_budget(
        path_length,
        dt,
        maximum_speed,
        mean_forward_speed,
        steps,
        final_completion,
    )

    progress_gain = rolling_window_gain(
        trajectory["path_arc_length"], STALL_WINDOW_STEPS
    )
    stall = np.isfinite(progress_gain) & (progress_gain <= STALL_MAX_ARC_GAIN_M)
    first_stall = first_true_index(stall)
    safety_density = rolling_window_mean(
        trajectory["safety_override"], STALL_WINDOW_STEPS
    )
    dense_safety = np.isfinite(safety_density) & (
        safety_density >= SAFETY_DENSE_THRESHOLD
    )
    first_dense_safety = first_true_index(dense_safety)
    densest_safety = int(np.nanargmax(safety_density)) if np.isfinite(safety_density).any() else -1

    if first_stall >= 0:
        start = first_stall - STALL_WINDOW_STEPS + 1
        stall_safety_fraction = float(
            np.mean(trajectory["safety_override"][start : first_stall + 1])
        )
        stall_abs_omega = float(
            np.mean(np.abs(trajectory["applied_omega"][start : first_stall + 1]))
        )
        if stall_safety_fraction >= SAFETY_DENSE_THRESHOLD:
            stall_class = "safety_intervention"
        elif stall_abs_omega >= 0.3:
            stall_class = "repeated_turning"
        else:
            stall_class = "local_minimum"
    else:
        stall_safety_fraction = math.nan
        stall_abs_omega = math.nan
        stall_class = "no_sustained_stall"

    max_cross_track_step = int(np.argmax(np.abs(trajectory["signed_cross_track_error"])))
    min_boundary_step = int(np.argmin(trajectory["minimum_footprint_boundary_margin"]))
    projection = detect_projection_events(trajectory["path_progress_ratio"])
    branches = trajectory["branch_id"].astype(int)
    centers = trajectory["center_crossing_count"].astype(int)
    branch_change_indices = np.flatnonzero(np.diff(branches, prepend=branches[0]) != 0)
    center_change_indices = np.flatnonzero(np.diff(centers, prepend=centers[0]) != 0)

    deadline = float(metrics.get("planner_deadline_ms", math.nan))
    planner = trajectory["planner_compute_ms"]
    deadline_miss_rate = (
        float(np.mean(planner > deadline)) if math.isfinite(deadline) else math.nan
    )
    boundary_steps = int(np.sum(trajectory["minimum_footprint_boundary_margin"] < 0.0))
    budget_row = {
        "scene": scene,
        "observed_steps": steps,
        "control_dt_s": dt,
        "path_length_m": path_length,
        "maximum_speed_mps": maximum_speed,
        "mean_forward_applied_speed_mps": mean_forward_speed,
        "final_completion_ratio": final_completion,
        **budget,
        "theoretical_margin_vs_700_steps": EPISODE_LIMIT
        - budget["theoretical_min_steps"],
        "applied_velocity_margin_vs_700_steps": EPISODE_LIMIT
        - budget["applied_velocity_estimated_steps"],
        "empirical_progress_margin_vs_700_steps": EPISODE_LIMIT
        - budget["empirical_progress_estimated_steps"],
    }
    event_row = {
        "scene": scene,
        "termination_reason": str(metrics.get("termination_reason", "unknown")),
        "success": bool(metrics.get("success", False)),
        "metric_stuck_steps": int(metrics.get("stuck_steps", 0)),
        "metric_spin_steps": int(metrics.get("spin_steps", 0)),
        "metric_safety_interventions": int(metrics.get("safety_interventions", 0)),
        "first_stall_step": first_stall + 1 if first_stall >= 0 else -1,
        "first_stall_time_s": float(time[first_stall]) if first_stall >= 0 else math.nan,
        "stall_class": stall_class,
        "stall_safety_fraction": stall_safety_fraction,
        "stall_mean_abs_applied_omega": stall_abs_omega,
        "first_dense_safety_step": first_dense_safety + 1
        if first_dense_safety >= 0
        else -1,
        "densest_safety_window_end_step": densest_safety + 1
        if densest_safety >= 0
        else -1,
        "densest_safety_fraction": float(safety_density[densest_safety])
        if densest_safety >= 0
        else math.nan,
        "max_abs_cross_track_step": max_cross_track_step + 1,
        "max_abs_cross_track_m": float(
            abs(trajectory["signed_cross_track_error"][max_cross_track_step])
        ),
        "min_boundary_step": min_boundary_step + 1,
        "min_boundary_margin_m": float(
            trajectory["minimum_footprint_boundary_margin"][min_boundary_step]
        ),
        "boundary_violation_steps": boundary_steps,
        "proposal_authority_mean": float(
            np.mean(trajectory["reliability_proposal_authority"])
        ),
        "proposal_authority_max": float(
            np.max(trajectory["reliability_proposal_authority"])
        ),
        "proposal_fallback_fraction_mean": float(
            np.mean(trajectory["reliability_proposal_fallback_fraction"])
        ),
        "guided_elites_total": int(np.sum(trajectory["paper_guided_elite_count"])),
        "planner_ms_p50": float(np.percentile(planner, 50)),
        "planner_ms_p95": float(np.percentile(planner, 95)),
        "planner_ms_p99": float(np.percentile(planner, 99)),
        "planner_deadline_ms": deadline,
        "planner_deadline_miss_rate": deadline_miss_rate,
        "branch_unique": "|".join(str(value) for value in sorted(set(branches))),
        "branch_changes": int(branch_change_indices.size),
        "progress_regressions": int(projection["regression_indices"].size),
        "progress_jumps": int(projection["jump_indices"].size),
        "center_crossing_changes": int(center_change_indices.size),
    }
    derived = {
        "progress_gain": progress_gain,
        "stall": stall.astype(float),
        "safety_density": safety_density,
        "projection_delta": projection["delta"],
        "branch_change_indices": branch_change_indices,
        "center_change_indices": center_change_indices,
        "regression_indices": projection["regression_indices"],
        "jump_indices": projection["jump_indices"],
    }
    return budget_row, event_row, {"trajectory": trajectory, "derived": derived}


def classify_root_causes(budget: Mapping[str, object], event: Mapping[str, object]) -> dict:
    step_budget_impossible = int(budget["theoretical_min_steps"]) > EPISODE_LIMIT
    boundary_termination = event["termination_reason"] == "boundary_violation"
    repeated_turning = event["stall_class"] == "repeated_turning"
    safety_dense = int(event["first_dense_safety_step"]) >= 0
    planner_latency = (
        float(event["planner_ms_p99"]) > float(event["planner_deadline_ms"])
        and float(event["planner_deadline_miss_rate"]) > 0.0
    )
    terminal_logic = (
        float(budget["final_completion_ratio"]) >= 0.95 and not bool(event["success"])
    )
    projection_status = "not_applicable"
    if budget["scene"] == "infinity":
        if int(event["center_crossing_changes"]) == 0:
            projection_status = "center_not_reached"
        elif int(event["progress_regressions"]) or int(event["progress_jumps"]):
            projection_status = "discontinuity_detected"
        else:
            projection_status = "continuous_at_observed_crossing"
    primary = []
    secondary = []
    if step_budget_impossible:
        primary.append("step_budget")
    if boundary_termination:
        primary.append("candidate_boundary_cost_or_termination")
    if repeated_turning:
        secondary.append("repeated_turning_local_minimum")
    if safety_dense:
        secondary.append("safety_intervention_dense_window")
    if float(event["proposal_fallback_fraction_mean"]) >= 0.8:
        secondary.append("actor_proposal_mostly_fallback")
    if planner_latency:
        secondary.append("planner_real_time_deadline")
    if terminal_logic:
        secondary.append("terminal_logic")
    if not primary:
        primary.append("control_efficiency_or_local_minimum")
    return {
        "scene": budget["scene"],
        "primary_root_causes": "|".join(primary),
        "secondary_root_causes": "|".join(secondary) if secondary else "none",
        "step_budget_impossible": step_budget_impossible,
        "boundary_termination": boundary_termination,
        "repeated_turning_stall": repeated_turning,
        "safety_dense_window": safety_dense,
        "actor_fallback_fraction_ge_0p8": float(
            event["proposal_fallback_fraction_mean"]
        )
        >= 0.8,
        "planner_deadline_failed": planner_latency,
        "terminal_logic_flag": terminal_logic,
        "infinity_projection_status": projection_status,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_scene(scene: str, data: Mapping[str, object], output_dir: Path) -> None:
    tr = data["trajectory"]
    derived = data["derived"]
    time = tr["time"]
    fig, axes = plt.subplots(7, 1, figsize=(11.0, 13.5), sharex=True)
    axes[0].plot(time, tr["path_progress_ratio"], color="#0072B2")
    axes[0].set_ylabel("Completion")
    axes[0].set_ylim(bottom=0.0)
    for column, label, color in (
        ("proposed_v", "proposed", "#999999"),
        ("executed_v", "executed", "#E69F00"),
        ("applied_v", "applied", "#009E73"),
    ):
        axes[1].plot(time, tr[column], label=label, color=color, linewidth=1.0)
    axes[1].set_ylabel("v [m/s]")
    axes[1].legend(ncol=3, loc="upper right")
    for column, label, color in (
        ("proposed_omega", "proposed", "#999999"),
        ("executed_omega", "executed", "#D55E00"),
        ("applied_omega", "applied", "#CC79A7"),
    ):
        axes[2].plot(time, tr[column], label=label, color=color, linewidth=1.0)
    axes[2].set_ylabel("omega [rad/s]")
    axes[2].legend(ncol=3, loc="upper right")
    axes[3].plot(time, tr["signed_cross_track_error"], label="cross-track")
    axes[3].plot(
        time,
        tr["minimum_footprint_boundary_margin"],
        label="footprint margin",
    )
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_ylabel("Geometry [m]")
    axes[3].legend(ncol=2, loc="upper right")
    axes[4].fill_between(
        time, 0.0, tr["safety_override"], color="#D55E00", alpha=0.55, label="safety"
    )
    axes[4].step(
        time, derived["stall"], where="post", color="#000000", label="derived stall"
    )
    axes[4].set_ylabel("Safety / stall")
    axes[4].legend(ncol=2, loc="upper right")
    axes[5].plot(
        time,
        tr["reliability_proposal_authority"],
        label="proposal authority",
        color="#0072B2",
    )
    elite_scaled = tr["paper_guided_elite_count"] / max(
        1.0, float(np.max(tr["paper_guided_elite_count"]))
    )
    axes[5].plot(time, elite_scaled, label="guided elites (scaled)", color="#009E73")
    axes[5].set_ylabel("RL use")
    axes[5].legend(ncol=2, loc="upper right")
    axes[6].plot(time, tr["planner_compute_ms"], color="#56B4E9")
    axes[6].set_ylabel("Planner [ms]")
    axes[6].set_xlabel("Simulation time [s]")
    axes[6].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"L246 read-only diagnosis | {scene} | seed {EXPECTED_SEED} | L244 Full",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"fig_l246_{scene}_timeseries.png", dpi=220)
    fig.savefig(output_dir / f"fig_l246_{scene}_timeseries.pdf")
    plt.close(fig)


def plot_infinity_projection(data: Mapping[str, object], output_dir: Path) -> None:
    tr = data["trajectory"]
    derived = data["derived"]
    time = tr["time"]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)
    axes[0].plot(time, tr["path_progress_ratio"], color="#0072B2")
    axes[0].set_ylabel("Progress")
    axes[1].plot(time, derived["projection_delta"], color="#D55E00")
    axes[1].axhline(PROGRESS_REGRESSION_THRESHOLD, color="black", linestyle="--")
    axes[1].axhline(PROGRESS_JUMP_THRESHOLD, color="black", linestyle="--")
    axes[1].set_ylabel("Delta progress")
    axes[2].step(time, tr["branch_id"], where="post", label="branch id")
    axes[2].step(
        time,
        tr["center_crossing_count"],
        where="post",
        label="center crossings",
    )
    axes[2].set_ylabel("Identity")
    axes[2].set_xlabel("Simulation time [s]")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("L246 Infinity path-projection continuity audit", fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig_l246_infinity_projection.png", dpi=220)
    fig.savefig(output_dir / "fig_l246_infinity_projection.pdf")
    plt.close(fig)


def plot_budget_summary(budgets: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    labels = [str(row["scene"]).replace("_", " ").title() for row in budgets]
    x = np.arange(len(labels))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].bar(x, [float(row["path_length_m"]) for row in budgets], color="#0072B2")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Path length [m]")
    axes[0].set_title("Frozen Tracking path lengths")
    series = (
        ("theoretical_min_steps", "Theoretical minimum", "#009E73"),
        ("applied_velocity_estimated_steps", "Applied-velocity estimate", "#E69F00"),
        ("empirical_progress_estimated_steps", "Empirical-progress estimate", "#D55E00"),
    )
    for index, (field, label, color) in enumerate(series):
        axes[1].bar(
            x + (index - 1) * width,
            [float(row[field]) for row in budgets],
            width,
            label=label,
            color=color,
        )
    axes[1].axhline(EPISODE_LIMIT, color="black", linestyle="--", label="700-step limit")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Estimated total steps (log scale)")
    axes[1].set_title("Episode budget is structurally insufficient")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"L246 read-only step-budget audit | seed {EXPECTED_SEED}",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig_l246_step_budget_summary.png", dpi=240)
    fig.savefig(output_dir / "fig_l246_step_budget_summary.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("docs/experiments/post_l217/tables"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("docs/experiments/post_l217/figures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    table_dir = (repo_root / args.table_dir).resolve()
    figure_dir = (repo_root / args.figure_dir).resolve()
    budgets: List[dict] = []
    events: List[dict] = []
    manifest = {
        "analysis": "L246_tracking_root_cause_audit",
        "protocol": "docs/experiments/post_l217/l246_tracking_root_cause_audit_protocol.md",
        "read_only": True,
        "episode_limit": EPISODE_LIMIT,
        "inputs": {},
        "thresholds": {
            "stall_window_steps": STALL_WINDOW_STEPS,
            "stall_max_arc_gain_m": STALL_MAX_ARC_GAIN_M,
            "safety_dense_threshold": SAFETY_DENSE_THRESHOLD,
            "progress_regression_threshold": PROGRESS_REGRESSION_THRESHOLD,
            "progress_jump_threshold": PROGRESS_JUMP_THRESHOLD,
        },
    }
    analyzed = {}
    for scene in SCENES:
        paths = locate_scene_inputs(repo_root, scene)
        provenance = verify_provenance(paths)
        budget, event, data = analyze_scene(scene, paths)
        budgets.append(budget)
        events.append(event)
        analyzed[scene] = data
        plot_scene(scene, data, figure_dir)
        manifest["inputs"][scene] = {
            "provenance": provenance,
            **{
                key: {
                    "path": str(path.relative_to(repo_root)),
                    "sha256": sha256_file(path),
                }
                for key, path in paths.items()
                if key != "run_root"
            },
        }
    plot_infinity_projection(analyzed["infinity"], figure_dir)
    plot_budget_summary(budgets, figure_dir)
    write_csv(table_dir / "l246_tracking_step_budget.csv", budgets)
    write_csv(table_dir / "l246_tracking_key_events.csv", events)
    root_causes = [
        classify_root_causes(budget, event)
        for budget, event in zip(budgets, events)
    ]
    write_csv(table_dir / "l246_root_cause_classification.csv", root_causes)
    infinity = analyzed["infinity"]
    tr = infinity["trajectory"]
    derived = infinity["derived"]
    indices = sorted(
        set(derived["branch_change_indices"].tolist())
        | set(derived["center_change_indices"].tolist())
        | set(derived["regression_indices"].tolist())
        | set(derived["jump_indices"].tolist())
    )
    infinity_rows = [
        {
            "step": index + 1,
            "time_s": tr["time"][index],
            "branch_id": int(tr["branch_id"][index]),
            "center_crossing_count": int(tr["center_crossing_count"][index]),
            "path_progress_ratio": tr["path_progress_ratio"][index],
            "progress_delta": derived["projection_delta"][index],
            "branch_change": index in set(derived["branch_change_indices"].tolist()),
            "center_change": index in set(derived["center_change_indices"].tolist()),
            "regression": index in set(derived["regression_indices"].tolist()),
            "jump": index in set(derived["jump_indices"].tolist()),
        }
        for index in indices
    ]
    if infinity_rows:
        write_csv(table_dir / "l246_infinity_projection_events.csv", infinity_rows)
    else:
        (table_dir / "l246_infinity_projection_events.csv").write_text(
            "step,time_s,branch_id,center_crossing_count,path_progress_ratio,"
            "progress_delta,branch_change,center_change,regression,jump\n",
            encoding="utf-8",
        )
    manifest_path = table_dir / "l246_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"budgets": budgets, "events": events, "root_causes": root_causes},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
