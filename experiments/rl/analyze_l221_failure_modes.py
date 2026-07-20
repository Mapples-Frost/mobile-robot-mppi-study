#!/usr/bin/env python3
"""Reproducible root-cause audit for the three failed L221 MuJoCo maps.

This is a development-only diagnostic.  It preserves every seed and compares
the configured reference against both the physical robot footprint and the
unchanged scan_guard hard-stop envelope.  Planner obstacles are never rebuilt
from simulator truth; static geometry is used only for this offline audit.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.evaluation.scene_feasibility import audit_reference_path


SCENES = (
    ("Serpentine", "l218_serpentine", "mujoco_l218_serpentine_polyline.yaml"),
    ("Nested U", "l218_nested_u", "mujoco_l218_nested_u_polyline.yaml"),
    (
        "Cylinder spiral",
        "l218_cylinder_spiral",
        "mujoco_l218_cylinder_spiral_polyline.yaml",
    ),
)
ARMS = ("icode_mppi", "full_proposed")
ARM_STYLE = {
    "icode_mppi": ("ICODE-MPPI", "#0072B2"),
    "full_proposed": ("Full proposed", "#D55E00"),
}


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to write an empty L221 audit table")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _as_float(row, name):
    value = row[name]
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        value = value.strip().lower() == "true"
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite %s" % name)
    return result


def _box(ax, obstacle, inflation, **kwargs):
    cx, cy = (float(value) for value in obstacle["position"])
    hx, hy = (float(value) + float(inflation) for value in obstacle["size"])
    patch = Rectangle((cx - hx, cy - hy), 2.0 * hx, 2.0 * hy, **kwargs)
    patch.set_transform(
        Affine2D().rotate_around(cx, cy, float(obstacle.get("yaw", 0.0)))
        + ax.transData
    )
    ax.add_patch(patch)


def _obstacle(ax, obstacle, inflation, **kwargs):
    if str(obstacle.get("type", "cylinder")) == "box":
        _box(ax, obstacle, inflation, **kwargs)
        return
    cx, cy = (float(value) for value in obstacle["position"])
    ax.add_patch(Circle(
        (cx, cy),
        float(obstacle.get("radius", 0.25)) + float(inflation),
        **kwargs,
    ))


def _episode_dir(run_dir, arm, progress_row):
    expected = (
        "%s__%s__%s__seed%s"
        % (
            arm,
            progress_row["scene"],
            progress_row["physics_domain"],
            progress_row["seed"],
        )
    )
    path = run_dir / "runs" / arm / expected
    if not path.is_dir():
        raise ValueError("missing L221 episode directory: %s" % path)
    return path


def _load_runs(run_glob, expected_git_sha):
    hard_scene_tokens = {
        scene[len("l218_"):] if scene.startswith("l218_") else scene
        for _, scene, _ in SCENES
    }
    run_dirs = sorted(
        path for path in ROOT.glob(run_glob)
        if path.is_dir() and any(
            ("_%s_seed" % token) in path.name for token in hard_scene_tokens
        )
    )
    if len(run_dirs) != 9:
        raise ValueError("expected nine hard-map run directories, got %d" % len(run_dirs))
    trajectories = defaultdict(list)
    episode_rows = []
    observed = set()
    config_contracts = set()
    for run_dir in run_dirs:
        provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        if provenance["git_sha"] != expected_git_sha:
            raise ValueError("unexpected L221 Git SHA: %s" % run_dir)
        progress = _read_csv(run_dir / "progress.csv")
        if len(progress) != 3:
            raise ValueError("L221 block must contain exactly three arms: %s" % run_dir)
        by_arm = {row["benchmark_arm"]: row for row in progress}
        for arm in ARMS:
            row = by_arm[arm]
            key = (row["scene"], row["seed"], arm)
            if key in observed:
                raise ValueError("duplicate L221 hard-map cell: %s" % (key,))
            observed.add(key)
            episode = _episode_dir(run_dir, arm, row)
            metrics = json.loads((episode / "metrics.json").read_text(encoding="utf-8"))
            metadata = metrics["metadata"]
            if (
                metadata.get("plant_backend") != "mujoco_diff_drive"
                or str(metadata.get("mujoco_version")) != "3.2.3"
                or metadata.get("prediction_mode") != "icode_residual"
            ):
                raise ValueError("invalid MuJoCo/ICODE provenance: %s" % episode)
            resolved = load_yaml(episode / "config_resolved.yaml")
            config_contracts.add((
                int(resolved["planner"]["horizon"]),
                float(resolved["experiment"]["control_dt"]),
                float(resolved["perception"]["scan_guard"]["near_body_stop_radius"]),
                float(resolved["planner"]["obstacle_influence"]),
                int(float(row["rollout_budget_per_decision"])),
                int(float(row["paper_iterations"])),
                int(float(row["steps"])),
            ))
            values = _read_csv(episode / "trajectory.csv")
            if not values:
                raise ValueError("empty L221 trajectory: %s" % episode)
            tail = values[-min(200, len(values)):]
            anchor = values[max(0, len(values) - 201)]
            last = values[-1]
            reason_counts = Counter(item["safety_reason"] for item in tail)
            displacement = math.hypot(
                _as_float(last, "x") - _as_float(anchor, "x"),
                _as_float(last, "y") - _as_float(anchor, "y"),
            )
            tail_count = len(tail)
            episode_rows.append({
                "scene": row["scene"],
                "seed": int(row["seed"]),
                "arm": arm,
                "success": int(_as_float(row, "success")),
                "collision": int(_as_float(row, "collision")),
                "steps": int(_as_float(row, "steps")),
                "path_completion_ratio": _as_float(row, "path_completion_ratio"),
                "safety_interventions": int(_as_float(row, "safety_interventions")),
                "last_200_displacement_m": displacement,
                "last_200_abs_proposed_v_mean": float(np.mean([
                    abs(_as_float(item, "proposed_v")) for item in tail
                ])),
                "last_200_abs_applied_v_mean": float(np.mean([
                    abs(_as_float(item, "applied_v")) for item in tail
                ])),
                "last_200_abs_applied_omega_mean": float(np.mean([
                    abs(_as_float(item, "applied_omega")) for item in tail
                ])),
                "last_200_front_slow_fraction": (
                    reason_counts["front_obstacle_slow"] / float(tail_count)
                ),
                "last_200_front_soft_block_fraction": (
                    reason_counts["front_soft_block"] / float(tail_count)
                ),
                "last_200_near_body_hard_stop_fraction": (
                    reason_counts["near_body_hard_stop"] / float(tail_count)
                ),
                "final_x": _as_float(last, "x"),
                "final_y": _as_float(last, "y"),
                "final_clearance": _as_float(last, "clearance"),
                "final_target_x": _as_float(last, "target_x"),
                "final_target_y": _as_float(last, "target_y"),
                "counterfactual_authority_mean": _as_float(
                    row, "reliability_counterfactual_authority_mean"
                ),
                "proposal_authority_mean": _as_float(
                    row, "reliability_proposal_authority_mean"
                ),
                "run_dir": str(episode),
            })
            trajectories[(row["scene"], arm)].append((int(row["seed"]), values))
    expected = {
        (scene, str(seed), arm)
        for _, scene, _ in SCENES
        for seed in (91001, 91002, 91003)
        for arm in ARMS
    }
    if observed != expected:
        raise ValueError("L221 hard-map cells differ from the frozen design")
    if len(config_contracts) != 1:
        raise ValueError("L221 hard-map configuration contract changed across cells")
    contract = next(iter(config_contracts))
    return episode_rows, trajectories, {
        "horizon_steps": contract[0],
        "control_dt_s": contract[1],
        "prediction_horizon_s": contract[0] * contract[1],
        "near_body_stop_radius_m": contract[2],
        "obstacle_influence_m": contract[3],
        "rollouts_per_decision": contract[4],
        "iterations": contract[5],
        "max_steps_observed": contract[6],
    }


def _scene_audits(guard_radius):
    rows = []
    configs = {}
    for _, scene_name, filename in SCENES:
        config_path = ROOT / "configs" / "research" / filename
        config = load_yaml(config_path)
        configs[scene_name] = config
        radius = float(config["plant"]["robot"]["collision_radius"])
        physical = audit_reference_path(
            config["scene"], config["task"]["points"], radius,
            margin=0.0, sample_spacing=0.005,
        )
        guard = audit_reference_path(
            config["scene"], config["task"]["points"], guard_radius,
            margin=0.0, sample_spacing=0.005,
        )
        rows.append({
            "scene": scene_name,
            "config": str(config_path),
            "robot_collision_radius_m": radius,
            "scan_guard_near_body_radius_m": guard_radius,
            "reference_length_m": physical["reference_length"],
            "physical_minimum_clearance_m": physical["minimum_clearance"],
            "physical_minimum_x": physical["minimum_clearance_location"][0],
            "physical_minimum_y": physical["minimum_clearance_location"][1],
            "physical_reference_clear": int(physical["path_clear"]),
            "guard_minimum_clearance_m": guard["minimum_clearance"],
            "guard_minimum_x": guard["minimum_clearance_location"][0],
            "guard_minimum_y": guard["minimum_clearance_location"][1],
            "guard_reference_clear": int(guard["path_clear"]),
        })
    return rows, configs


def _aggregate_episode_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scene"], row["arm"])].append(row)
    output = []
    metrics = (
        "success", "collision", "path_completion_ratio",
        "safety_interventions", "last_200_displacement_m",
        "last_200_abs_proposed_v_mean", "last_200_abs_applied_v_mean",
        "last_200_abs_applied_omega_mean", "last_200_front_slow_fraction",
        "last_200_front_soft_block_fraction",
        "last_200_near_body_hard_stop_fraction", "final_clearance",
        "counterfactual_authority_mean", "proposal_authority_mean",
    )
    for (scene, arm), group in sorted(groups.items()):
        item = {"scene": scene, "arm": arm, "episodes": len(group)}
        for metric in metrics:
            item[metric + "_mean"] = float(np.mean([row[metric] for row in group]))
        output.append(item)
    return output


def _plot(configs, scene_audits, trajectories, output, dpi):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": int(dpi),
        "savefig.dpi": int(dpi),
        "savefig.bbox": "tight",
    })
    audit_lookup = {row["scene"]: row for row in scene_audits}
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.75), constrained_layout=True)
    for panel, ((title, scene_name, _), ax) in enumerate(zip(SCENES, axes)):
        config = configs[scene_name]
        scene = config["scene"]
        guard_radius = audit_lookup[scene_name]["scan_guard_near_body_radius_m"]
        for obstacle in scene.get("obstacles", ()):
            _obstacle(
                ax, obstacle, guard_radius,
                facecolor="#D55E00", edgecolor="none", alpha=0.075, zorder=0,
            )
        for obstacle in scene.get("obstacles", ()):
            _obstacle(
                ax, obstacle, 0.0,
                facecolor="#6B7280", edgecolor="#374151", linewidth=0.55,
                alpha=0.90, zorder=1,
            )
        points = np.asarray(config["task"]["points"], dtype=np.float64)
        ax.plot(
            points[:, 0], points[:, 1], linestyle="--", color="#009E73",
            linewidth=1.4, marker="o", markersize=2.0, zorder=2,
        )
        for arm in ARMS:
            _, color = ARM_STYLE[arm]
            for seed, values in trajectories[(scene_name, arm)]:
                xy = np.asarray([
                    (_as_float(row, "x"), _as_float(row, "y")) for row in values
                ])
                ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.0, alpha=0.60)
                ax.plot(
                    xy[-1, 0], xy[-1, 1], marker="x", color=color,
                    markersize=5.0, markeredgewidth=1.0, zorder=4,
                )
        audit = audit_lookup[scene_name]
        ax.plot(
            audit["guard_minimum_x"], audit["guard_minimum_y"],
            marker="X", color="#CC0000", markersize=6.0, zorder=5,
        )
        ax.set_title("(%s) %s" % (chr(ord("a") + panel), title))
        ax.text(
            0.02, 0.02,
            "guard-clearance min = %.3f m" % audit["guard_minimum_clearance_m"],
            transform=ax.transAxes, fontsize=7.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        ax.set_xlim(0.0, 6.5)
        ax.set_ylim(0.0, 6.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(color="#D1D5DB", linewidth=0.45, alpha=0.55)
    handles = [
        Line2D([0], [0], color="#009E73", linestyle="--", label="Reference"),
        Line2D([0], [0], color=ARM_STYLE["icode_mppi"][1], label="ICODE-MPPI"),
        Line2D([0], [0], color=ARM_STYLE["full_proposed"][1], label="Full proposed"),
        Line2D([0], [0], marker="X", color="#CC0000", linestyle="none",
               label="Worst guard-clearance point"),
        Rectangle((0, 0), 1, 1, facecolor="#D55E00", alpha=0.15,
                  label="scan_guard inflated geometry"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(
        "L221 hard-map failure audit: trajectories and unchanged safety envelope",
        fontsize=11.5, y=1.12,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("l221_hard_map_failure_modes.%s" % suffix), dpi=int(dpi))
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-glob",
        default=(
            "results/research_platform/rl/"
            "l218_dev_screen_l221_gate_s9100*_*_seed9100*"
        ),
    )
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    episode_rows, trajectories, contract = _load_runs(
        args.run_glob, args.expected_git_sha
    )
    scene_rows, configs = _scene_audits(contract["near_body_stop_radius_m"])
    aggregate_rows = _aggregate_episode_rows(episode_rows)
    _write_csv(output / "root_cause_scene_audit.csv", scene_rows)
    _write_csv(output / "trajectory_stagnation_audit.csv", episode_rows)
    _write_csv(output / "trajectory_stagnation_summary.csv", aggregate_rows)
    _plot(configs, scene_rows, trajectories, output, args.dpi)
    report = {
        "status": "complete_development_diagnostic",
        "formal_claim_allowed": False,
        "independent_unit": "seed",
        "independent_seeds": [91001, 91002, 91003],
        "repeated_strata": ["scene"],
        "run_git_sha": args.expected_git_sha,
        "configuration_contract": contract,
        "all_reference_paths_physically_collision_free": all(
            bool(row["physical_reference_clear"]) for row in scene_rows
        ),
        "all_reference_paths_scan_guard_feasible": all(
            bool(row["guard_reference_clear"]) for row in scene_rows
        ),
        "diagnosis_scope": (
            "Offline geometry and executed-trajectory audit only; static obstacle "
            "truth was never supplied to an online planner."
        ),
    }
    (output / "root_cause_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
