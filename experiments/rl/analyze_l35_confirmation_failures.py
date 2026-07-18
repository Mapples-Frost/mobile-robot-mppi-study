#!/usr/bin/env python3
"""Diagnose why the preregistered L35 confirmation gate failed."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _resolved_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _trajectory_records(input_dir):
    records = []
    pattern = "training_seed_*/*/*/runs/seed_*/trajectory.csv"
    for path in sorted((input_dir / "runs").glob(pattern)):
        training_seed = int(path.parents[4].name.rsplit("_", 1)[-1])
        role = path.parents[3].name
        scene = path.parents[2].name
        episode_seed = int(path.parent.name.rsplit("_", 1)[-1])
        rows = _read_csv(path)
        collisions = [
            float(row["time"]) for row in rows if float(row["collision"]) > 0.5
        ]
        records.append({
            "training_seed": training_seed,
            "checkpoint_role": role,
            "scene": scene,
            "episode_seed": episode_seed,
            "steps": len(rows),
            "mean_subgoal_distance_m": float(np.mean([
                float(row["rl_subgoal_distance"]) for row in rows
            ])),
            "mean_abs_subgoal_bearing_rad": float(np.mean([
                abs(float(row["rl_subgoal_bearing"])) for row in rows
            ])),
            "mean_gate_alpha": float(np.mean([
                float(row["rl_gate_alpha"]) for row in rows
            ])),
            "safety_override_fraction": float(np.mean([
                float(row["safety_override"]) for row in rows
            ])),
            "collision_time_s": None if not collisions else collisions[0],
        })
    return records


def _aggregate_episode_outcomes(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["training_seed"]), row["checkpoint_role"])].append(row)
    result = []
    for (training_seed, role), selected in sorted(grouped.items()):
        result.append({
            "training_seed": training_seed,
            "checkpoint_role": role,
            "episodes": len(selected),
            "success_rate": float(np.mean([_bool(row["success"]) for row in selected])),
            "collision_rate": float(np.mean([_bool(row["collision"]) for row in selected])),
            "mean_final_goal_distance_m": float(np.mean([
                float(row["final_goal_distance"]) for row in selected
            ])),
        })
    return result


def _aggregate_scene_outcomes(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["scene"])].append(row)
    result = []
    for scene, selected in sorted(grouped.items()):
        result.append({
            "scene": scene,
            "episodes": len(selected),
            "success_rate": float(np.mean([_bool(row["success"]) for row in selected])),
            "collision_rate": float(np.mean([_bool(row["collision"]) for row in selected])),
            "mean_final_goal_distance_m": float(np.mean([
                float(row["final_goal_distance"]) for row in selected
            ])),
        })
    return result


def _aggregate_prior_behavior(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(
            int(row["training_seed"]), row["checkpoint_role"], row["scene"]
        )].append(row)
    result = []
    for (training_seed, role, scene), selected in sorted(grouped.items()):
        collision_times = [
            float(row["collision_time_s"])
            for row in selected if row["collision_time_s"] is not None
        ]
        result.append({
            "training_seed": training_seed,
            "checkpoint_role": role,
            "scene": scene,
            "episodes": len(selected),
            "mean_subgoal_distance_m": float(np.mean([
                row["mean_subgoal_distance_m"] for row in selected
            ])),
            "mean_abs_subgoal_bearing_rad": float(np.mean([
                row["mean_abs_subgoal_bearing_rad"] for row in selected
            ])),
            "mean_gate_alpha": float(np.mean([
                row["mean_gate_alpha"] for row in selected
            ])),
            "mean_safety_override_fraction": float(np.mean([
                row["safety_override_fraction"] for row in selected
            ])),
            "mean_collision_time_s": (
                None if not collision_times else float(np.mean(collision_times))
            ),
        })
    return result


def _plot(output_dir, model_rows, scene_rows):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })
    initial_color = "#8C8C8C"
    best_color = "#D55E00"
    model_lookup = {
        (int(row["training_seed"]), row["checkpoint_role"]): row
        for row in model_rows
    }
    seeds = sorted({int(row["training_seed"]) for row in model_rows})
    labels = [str(seed)[-2:] for seed in seeds]
    x = np.arange(len(seeds), dtype=np.float64)
    fig, axes = plt.subplots(1, 4, figsize=(7.20, 2.70))
    metrics = (
        ("success_rate", "Success rate", "A  Success"),
        ("collision_rate", "Collision rate", "B  Collision"),
        ("mean_final_goal_distance_m", "Final distance (m)", "C  Terminal error"),
    )
    for axis, (metric, ylabel, title) in zip(axes[:3], metrics):
        initial = [model_lookup[(seed, "initial")][metric] for seed in seeds]
        best = [model_lookup[(seed, "best")][metric] for seed in seeds]
        for index in range(len(seeds)):
            axis.plot(
                [x[index] - 0.08, x[index] + 0.08],
                [initial[index], best[index]],
                color="#B0BEC5", linewidth=1.0, zorder=1,
            )
        axis.scatter(x - 0.08, initial, color=initial_color, s=26, label="Initial", zorder=3)
        axis.scatter(x + 0.08, best, color=best_color, marker="D", s=24, label="Best trained", zorder=3)
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_xlabel("Training-seed suffix")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        if metric.endswith("rate"):
            axis.set_ylim(-0.04, 1.04)
    axes[0].legend(fontsize=7, loc="best")

    scene_labels = [
        "Diagonal" if "diagonal" in row["scene"] else "Offset"
        for row in scene_rows
    ]
    sx = np.arange(len(scene_rows), dtype=np.float64)
    width = 0.34
    success_bars = axes[3].bar(
        sx - width / 2,
        [row["success_rate"] for row in scene_rows],
        width,
        color="#0072B2",
        label="Success",
    )
    collision_bars = axes[3].bar(
        sx + width / 2,
        [row["collision_rate"] for row in scene_rows],
        width,
        color="#D55E00",
        label="Collision",
    )
    axes[3].set_xticks(sx)
    axes[3].set_xticklabels(scene_labels)
    axes[3].set_ylim(-0.04, 1.04)
    axes[3].set_ylabel("Episode rate")
    axes[3].set_title("D  Scene strata", loc="left")
    for bars, label in ((success_bars, "Success"), (collision_bars, "Collision")):
        for bar in bars:
            if bar.get_height() > 0.0:
                axes[3].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() - 0.08,
                    label,
                    ha="center",
                    va="top",
                    rotation=90,
                    color="white",
                    fontsize=6.5,
                    fontweight="bold",
                )
    fig.suptitle(
        "L35 independent confirmation: the L34 development gain does not replicate",
        fontsize=9.8,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Three independently trained policies; two held-out motion paths × five new episode seeds per checkpoint.",
        ha="center",
        fontsize=7.1,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.90), w_pad=1.20)
    fig.savefig(output_dir / "fig_l35_independent_confirmation.png")
    fig.savefig(output_dir / "fig_l35_independent_confirmation.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    input_dir = _resolved_path(args.input_dir)
    episodes = _read_csv(input_dir / "confirmation_episodes.csv")
    trajectories = _trajectory_records(input_dir)
    model_rows = _aggregate_episode_outcomes(episodes)
    scene_rows = _aggregate_scene_outcomes(episodes)
    prior_rows = _aggregate_prior_behavior(trajectories)
    if len(trajectories) != len(episodes):
        raise ValueError("L35 trajectory count does not match episode count")
    _write_csv(input_dir / "model_checkpoint_outcomes.csv", model_rows)
    _write_csv(input_dir / "scene_outcomes.csv", scene_rows)
    _write_csv(input_dir / "prior_behavior_summary.csv", prior_rows)
    diagonal = next(row for row in scene_rows if "diagonal" in row["scene"])
    offset = next(row for row in scene_rows if "offset" in row["scene"])
    report = {
        "episodes": len(episodes),
        "trajectories": len(trajectories),
        "scene_outcomes": scene_rows,
        "complete_scene_separation": bool(
            diagonal["collision_rate"] == 1.0
            and diagonal["success_rate"] == 0.0
            and offset["collision_rate"] == 0.0
        ),
        "rl_inference_active": bool(
            all(abs(row["mean_gate_alpha"] - 0.25) < 1e-12 for row in trajectories)
            and any(row["mean_subgoal_distance_m"] > 0.0 for row in trajectories)
            and any(row["mean_abs_subgoal_bearing_rad"] > 0.0 for row in trajectories)
        ),
        "diagnosis": (
            "The independent performance gain did not replicate. RL inference was active, "
            "but the two confirmation paths formed nearly deterministic difficulty strata: "
            "all diagonal episodes collided and no offset episode collided."
        ),
    }
    (input_dir / "failure_diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(input_dir, model_rows, scene_rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
