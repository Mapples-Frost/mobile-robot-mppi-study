#!/usr/bin/env python3
"""Create publication-style L223 six-scene development figures."""

import argparse
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
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mobile_robot_mppi.core.config import load_yaml


ORDER = (
    ("l218_giant_u", "Giant U"),
    ("l218_opposed_u", "Opposed U"),
    ("l222_nested_u_safe", "Nested U"),
    ("l218_cylinder_forest", "Cylinder forest"),
    ("l222_cylinder_spiral_safe", "Cylinder spiral"),
    ("l222_serpentine_safe", "Serpentine"),
)


def _rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _draw_obstacle(ax, obstacle):
    color = "#6B7280"
    if str(obstacle.get("type", "cylinder")) == "box":
        cx, cy = (float(value) for value in obstacle["position"])
        hx, hy = (float(value) for value in obstacle["size"])
        patch = Rectangle(
            (cx - hx, cy - hy), 2.0 * hx, 2.0 * hy,
            facecolor=color, edgecolor="#374151", linewidth=0.45, alpha=0.88,
        )
        patch.set_transform(
            Affine2D().rotate_around(cx, cy, float(obstacle.get("yaw", 0.0)))
            + ax.transData
        )
        ax.add_patch(patch)
    else:
        cx, cy = (float(value) for value in obstacle["position"])
        ax.add_patch(Circle(
            (cx, cy), float(obstacle.get("radius", 0.25)),
            facecolor=color, edgecolor="#374151", linewidth=0.45, alpha=0.88,
        ))


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _trajectory_figure(audit, output):
    by_scene = {row["scene"]: row for row in audit["episodes"]}
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.75), constrained_layout=False)
    fig.subplots_adjust(
        left=0.07, right=0.985, bottom=0.075, top=0.82, wspace=0.28, hspace=0.42
    )
    for index, ((scene, title), ax) in enumerate(zip(ORDER, axes.flat)):
        row = by_scene[scene]
        episode = Path(row["episode_dir"])
        config = load_yaml(episode / "config_resolved.yaml")
        trajectory = _rows(episode / "trajectory.csv")
        for obstacle in config["scene"].get("obstacles", ()):
            _draw_obstacle(ax, obstacle)
        points = np.asarray(config["task"]["points"], dtype=np.float64)
        xy = np.asarray([
            (float(item["x"]), float(item["y"])) for item in trajectory
        ])
        ax.plot(
            points[:, 0], points[:, 1], "--", color="#009E73", linewidth=1.25,
            label="Reference", zorder=2,
        )
        ax.plot(
            xy[:, 0], xy[:, 1], color="#0072B2", linewidth=1.35,
            label="ICODE-MPPI", zorder=3,
        )
        ax.scatter(points[0, 0], points[0, 1], marker="o", s=22,
                   color="#E69F00", edgecolor="white", linewidth=0.4, zorder=4)
        ax.scatter(points[-1, 0], points[-1, 1], marker="*", s=42,
                   color="#D55E00", edgecolor="white", linewidth=0.4, zorder=4)
        ax.set_title(
            "(%s) %s\n%d steps | min clearance %.2f m"
            % (chr(ord("a") + index), title, row["steps"], row["minimum_clearance_m"]),
            fontsize=8.8,
        )
        ax.set_xlim(0.0, 6.5)
        ax.set_ylim(0.0, 6.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(color="#D1D5DB", linewidth=0.4, alpha=0.55)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.915))
    fig.suptitle(
        "L223 MuJoCo six-scene confirmation (development seed 91001; 6/6 success)",
        fontsize=11, y=0.985,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("l223_six_scene_trajectories.%s" % suffix))
    plt.close(fig)


def _calibration_figure(l222_path, l223_path, output):
    before = {row["scene"]: row for row in _rows(l222_path)}
    after = {row["scene"]: row for row in _rows(l223_path)}
    scenes = (
        ("l222_serpentine_safe", "Serpentine"),
        ("l222_nested_u_safe", "Nested U"),
        ("l222_cylinder_spiral_safe", "Cylinder spiral"),
    )
    completion_before = [100.0 * float(before[key]["final_route_progress_ratio"])
                         for key, _ in scenes]
    completion_after = [100.0 * float(after[key]["final_route_progress_ratio"])
                        for key, _ in scenes]
    steps_before = [float(before[key]["steps"]) for key, _ in scenes]
    steps_after = [float(after[key]["steps"]) for key, _ in scenes]
    labels = [label for _, label in scenes]
    x = np.arange(len(scenes))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), constrained_layout=True)
    colors = ("#B0BEC5", "#0072B2")
    for ax, left, right, ylabel, title in (
        (axes[0], completion_before, completion_after, "Route completion (%)",
         "(a) Completion"),
        (axes[1], steps_before, steps_after, "Executed steps",
         "(b) Time-to-goal / budget"),
    ):
        bars0 = ax.bar(x - width / 2, left, width, color=colors[0], label="L222 strict guard")
        bars1 = ax.bar(x + width / 2, right, width, color=colors[1], label="L223 calibrated guard")
        for bars, values in ((bars0, left), (bars1, right)):
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        "%.0f" % value, ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylim(0.0, 108.0)
    axes[1].set_ylim(0.0, 1520.0)
    axes[1].axhline(1400.0, color="#D55E00", linestyle="--", linewidth=0.9,
                    label="1400-step budget")
    handles0, labels0 = axes[0].get_legend_handles_labels()
    budget_handle = axes[1].lines[-1]
    fig.legend(handles0 + [budget_handle], labels0 + ["1400-step budget"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Bounded simulator guard calibration: same ICODE-MPPI and seed", y=1.18)
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("l223_guard_calibration_effect.%s" % suffix))
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--l222-csv", required=True)
    parser.add_argument("--l223-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))
    if audit["status"] != "gate_a_passed" or audit["successes"] != 6:
        raise ValueError("six-scene audit has not passed")
    _style()
    _trajectory_figure(audit, output)
    _calibration_figure(args.l222_csv, args.l223_csv, output)
    print(json.dumps({"status": "complete", "output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
