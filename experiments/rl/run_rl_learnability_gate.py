#!/usr/bin/env python3
"""Fixed-seed baseline/checkpoint benchmark for the RL learnability gate."""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


def _csv_list(text, cast=str):
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def _checkpoint_pairs(text):
    pairs = []
    for item in _csv_list(text):
        if "=" not in item:
            raise ValueError("checkpoint entries must be label=path")
        label, path = item.split("=", 1)
        source = Path(path)
        if not source.is_absolute():
            source = ROOT / source
        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError("checkpoint does not exist: %s" % source)
        pairs.append((label.strip(), source))
    if not pairs:
        raise ValueError("at least one RL checkpoint is required")
    return pairs


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["scene"], row["method"]), []).append(row)
    summary = []
    for (scene, method), values in sorted(groups.items()):
        success = np.asarray([float(value["success"]) for value in values])
        collision = np.asarray([float(value["collision"]) for value in values])
        distance = np.asarray([float(value["final_goal_distance"]) for value in values])
        compute = np.asarray([float(value["planner_compute_ms_mean"]) for value in values])
        interventions = np.asarray([float(value["safety_interventions"]) for value in values])
        summary.append({
            "scene": scene,
            "method": method,
            "seeds": len(values),
            "success_rate": float(success.mean()),
            "collision_rate": float(collision.mean()),
            "final_goal_distance_mean": float(distance.mean()),
            "final_goal_distance_std": float(distance.std()),
            "planner_compute_ms_mean": float(compute.mean()),
            "safety_interventions_mean": float(interventions.mean()),
        })
    return summary


def _plot_summary(summary, output):
    scenes = sorted(set(row["scene"] for row in summary))
    methods = []
    for row in summary:
        if row["method"] not in methods:
            methods.append(row["method"])
    figure, axes = plt.subplots(1, 4, figsize=(17.0, 4.2), constrained_layout=True)
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(scenes), dtype=np.float64)
    for method_index, method in enumerate(methods):
        rows = {(row["scene"], row["method"]): row for row in summary}
        offset = (method_index - 0.5 * (len(methods) - 1)) * width
        success = [rows[(scene, method)]["success_rate"] for scene in scenes]
        distance = [rows[(scene, method)]["final_goal_distance_mean"] for scene in scenes]
        interventions = [rows[(scene, method)]["safety_interventions_mean"] for scene in scenes]
        collisions = [rows[(scene, method)]["collision_rate"] for scene in scenes]
        axes[0].bar(x + offset, success, width=width, label=method)
        axes[1].bar(x + offset, distance, width=width, label=method)
        axes[2].bar(x + offset, interventions, width=width, label=method)
        axes[3].bar(x + offset, collisions, width=width, label=method)
    labels = (
        "Success rate",
        "Final goal distance (m)",
        "Safety interventions",
        "Collision rate",
    )
    for axis, label in zip(axes, labels):
        axis.set_xticks(x)
        axis.set_xticklabels(scenes, rotation=15, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.0, 1.05)
    axes[3].set_ylim(0.0, 1.05)
    if all(float(row["collision_rate"]) == 0.0 for row in summary):
        axes[3].text(
            0.5,
            0.5,
            "0 collisions\nin all 40 runs",
            ha="center",
            va="center",
            transform=axes[3].transAxes,
        )
    axes[0].legend(fontsize=8)
    figure.suptitle("RL sampling-prior learnability gate (fixed seeds)")
    figure.savefig(output / "learnability_gate_summary.png", dpi=180)
    plt.close(figure)


def _draw_obstacles(axis, config):
    for obstacle in config.get("scene", {}).get("obstacles", ()):
        position = obstacle.get("position", (0.0, 0.0))
        if str(obstacle.get("type", "cylinder")) == "box":
            size = obstacle.get("size", (0.25, 0.25))
            rectangle = Rectangle(
                (-float(size[0]), -float(size[1])),
                2.0 * float(size[0]),
                2.0 * float(size[1]),
                facecolor="0.70",
                edgecolor="0.25",
                alpha=0.65,
            )
            rectangle.set_transform(
                Affine2D()
                .rotate(float(obstacle.get("yaw", 0.0)))
                .translate(float(position[0]), float(position[1]))
                + axis.transData
            )
            axis.add_patch(rectangle)
        else:
            axis.add_patch(Circle(
                (float(position[0]), float(position[1])),
                float(obstacle.get("radius", 0.25)),
                facecolor="0.70",
                edgecolor="0.25",
                alpha=0.65,
            ))


def _plot_trajectories(output, scene_configs, methods, representative_seed):
    scene_names = list(scene_configs)
    figure, axes = plt.subplots(
        len(scene_names), 1, figsize=(7.2, 5.0 * len(scene_names)), squeeze=False,
        constrained_layout=True,
    )
    for scene_index, scene in enumerate(scene_names):
        axis = axes[scene_index, 0]
        _draw_obstacles(axis, scene_configs[scene])
        for method in methods:
            trajectory = output / "runs" / scene / method / ("seed_%d" % representative_seed) / "trajectory.csv"
            if not trajectory.exists():
                continue
            data = np.genfromtxt(str(trajectory), delimiter=",", names=True, dtype=None, encoding="utf-8")
            axis.plot(data["x"], data["y"], linewidth=1.8, label=method)
            axis.scatter(data["x"][0], data["y"][0], marker="o", s=30)
            axis.scatter(data["x"][-1], data["y"][-1], marker="x", s=40)
        axis.scatter((3.0,), (3.0,), marker="*", s=110, label="goal")
        axis.set_title("%s, seed=%d" % (scene, representative_seed))
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output / "learnability_gate_trajectories.png", dpi=180)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    parser.add_argument("--checkpoints", required=True, help="label=path,label=path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="41,42,43,44,45")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--goal-running-weight", type=float)
    parser.add_argument("--goal-terminal-weight", type=float)
    parser.add_argument("--near-goal-fallback", action="store_true")
    parser.add_argument("--near-goal-full-fallback-distance", type=float, default=0.55)
    parser.add_argument("--near-goal-full-rl-distance", type=float, default=1.75)
    args = parser.parse_args(argv)
    scene_files = _csv_list(args.configs)
    checkpoint_pairs = _checkpoint_pairs(args.checkpoints)
    seeds = _csv_list(args.seeds, int)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    methods = ["mppi"] + [label for label, _ in checkpoint_pairs]
    scene_configs = {}
    for scene_file in scene_files:
        base = load_yaml(scene_file)
        scene = str(base.get("scene", {}).get("name", Path(scene_file).stem))
        scene_configs[scene] = base
        variants = [("mppi", None)] + checkpoint_pairs
        for method, checkpoint in variants:
            for seed in seeds:
                config = copy.deepcopy(base)
                config["experiment"]["seed"] = seed
                config["planner"]["num_samples"] = args.num_samples
                if args.goal_running_weight is not None:
                    config["planner"]["goal_running_weight"] = args.goal_running_weight
                if args.goal_terminal_weight is not None:
                    config["planner"]["goal_terminal_weight"] = args.goal_terminal_weight
                config["planner"]["prediction_mode"] = "nominal"
                config["memory"]["enable"] = False
                if args.max_steps is not None:
                    config["experiment"]["max_steps"] = args.max_steps
                if checkpoint is None:
                    config["planner"]["sampling_prior"] = "goal_warm_start"
                    config.setdefault("rl", {})["enabled"] = False
                else:
                    config["planner"]["sampling_prior"] = "rl"
                    config.setdefault("rl", {}).update({
                        "enabled": True,
                        "checkpoint": str(checkpoint),
                        "device": "cpu",
                        "torch_num_threads": 1,
                        "gate": {
                            "mode": "none",
                            "near_goal_fallback_enabled": bool(
                                args.near_goal_fallback
                            ),
                            "near_goal_full_fallback_distance": float(
                                args.near_goal_full_fallback_distance
                            ),
                            "near_goal_full_rl_distance": float(
                                args.near_goal_full_rl_distance
                            ),
                        },
                    })
                run_dir = output / "runs" / scene / method / ("seed_%d" % seed)
                result = ExperimentRunner(config, ROOT, run_dir, headless=True).run()
                row = dict(result.summary)
                row.update({
                    "scene": scene,
                    "method": method,
                    "seed": seed,
                    "num_samples": args.num_samples,
                    "checkpoint": None if checkpoint is None else str(checkpoint),
                })
                rows.append(row)
    summary = _aggregate(rows)
    _write_csv(output / "episodes.csv", rows)
    _write_csv(output / "summary.csv", summary)
    manifest = {
        "git_sha": git_sha(ROOT),
        "configs": scene_files,
        "checkpoints": {label: str(path) for label, path in checkpoint_pairs},
        "seeds": seeds,
        "num_samples": args.num_samples,
        "goal_running_weight": args.goal_running_weight,
        "goal_terminal_weight": args.goal_terminal_weight,
        "near_goal_gate": {
            "enabled": bool(args.near_goal_fallback),
            "full_fallback_distance": args.near_goal_full_fallback_distance,
            "full_rl_distance": args.near_goal_full_rl_distance,
        },
        "summary": summary,
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    _plot_summary(summary, output)
    _plot_trajectories(output, scene_configs, methods, seeds[0])
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
