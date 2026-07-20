#!/usr/bin/env python3
"""Render the exact L218 MuJoCo scene YAMLs and offline feasibility audit."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.evaluation.scene_feasibility import (
    audit_reference_path,
    audit_static_scene,
)


SCENES = (
    ("Serpentine", "mujoco_l218_serpentine_polyline.yaml"),
    ("Giant U", "mujoco_l218_giant_u_polyline.yaml"),
    ("Opposed U", "mujoco_l218_opposed_u_polyline.yaml"),
    ("Nested U", "mujoco_l218_nested_u_polyline.yaml"),
    ("Cylinder forest", "mujoco_l218_cylinder_forest_polyline.yaml"),
    ("Cylinder rings", "mujoco_l218_cylinder_spiral_polyline.yaml"),
)


def _box(ax, obstacle, inflation, **kwargs):
    cx, cy = (float(value) for value in obstacle["position"])
    hx, hy = (float(value) + inflation for value in obstacle["size"])
    patch = Rectangle((cx - hx, cy - hy), 2.0 * hx, 2.0 * hy, **kwargs)
    yaw = float(obstacle.get("yaw", 0.0))
    patch.set_transform(
        Affine2D().rotate_around(cx, cy, yaw) + ax.transData
    )
    ax.add_patch(patch)


def _obstacle(ax, obstacle, inflation, **kwargs):
    if str(obstacle.get("type", "cylinder")) == "box":
        _box(ax, obstacle, inflation, **kwargs)
    else:
        cx, cy = (float(value) for value in obstacle["position"])
        ax.add_patch(Circle(
            (cx, cy),
            float(obstacle.get("radius", 0.25)) + inflation,
            **kwargs,
        ))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 3, figsize=(13.2, 8.8), constrained_layout=True
    )
    reports = []
    for index, ((title, filename), ax) in enumerate(zip(SCENES, axes.flat)):
        path = ROOT / "configs" / "research" / filename
        config = load_yaml(path)
        if config["plant"].get("backend") != "mujoco_diff_drive":
            raise ValueError("L218 scene is not backed by MuJoCo: %s" % path)
        if config["task"].get("type") != "polyline":
            raise ValueError("L218 scene must define a polyline task: %s" % path)
        scene = config["scene"]
        start = config["experiment"]["initial_state"][:2]
        points = config["task"]["points"]
        goal = points[-1]
        radius = float(config["plant"]["robot"]["collision_radius"])
        connectivity = audit_static_scene(
            scene, start, goal, radius, margin=0.03
        )
        reference = audit_reference_path(
            scene, points, radius, margin=0.03, sample_spacing=0.01
        )
        if not (
            connectivity["start_free"]
            and connectivity["goal_free"]
            and connectivity["path_exists"]
            and reference["path_clear"]
        ):
            raise ValueError("scene feasibility Gate failed: %s" % filename)
        reports.append({
            "scene": scene["name"],
            "config": str(path),
            "plant_backend": config["plant"]["backend"],
            "obstacle_count": len(scene.get("obstacles", ())),
            **connectivity,
            **reference,
        })

        for obstacle in scene.get("obstacles", ()):
            _obstacle(
                ax,
                obstacle,
                radius + 0.10,
                facecolor="none",
                edgecolor="#d95f02",
                linewidth=0.8,
                linestyle="--",
                alpha=0.65,
            )
        for obstacle in scene.get("obstacles", ()):
            _obstacle(
                ax,
                obstacle,
                0.0,
                facecolor="#6b7280",
                edgecolor="#374151",
                linewidth=0.8,
                alpha=0.88,
            )
        values = list(zip(*points))
        ax.plot(
            values[0], values[1], color="#0072B2", linewidth=2.0,
            marker="o", markersize=2.5, zorder=4,
        )
        ax.add_patch(Circle(
            tuple(start), radius, facecolor="#009E73", edgecolor="white",
            linewidth=1.0, zorder=5,
        ))
        ax.add_patch(Circle(
            tuple(goal), float(config["task"]["position_tolerance"]),
            facecolor="#F0E442", edgecolor="#9A7D0A", linewidth=1.0,
            alpha=0.65, zorder=3,
        ))
        ax.set_title("(%s) %s" % (chr(ord("a") + index), title))
        ax.text(
            0.02,
            0.02,
            "route %.2f m | min route clearance %.2f m"
            % (reference["reference_length"], reference["minimum_clearance"]),
            transform=ax.transAxes,
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        ax.set_xlim(0.0, 6.5)
        ax.set_ylim(0.0, 6.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(color="#d1d5db", linewidth=0.5, alpha=0.65)

    fig.suptitle(
        "L218 expanded MuJoCo navigation suite (exact YAML geometry)",
        fontsize=14,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output / ("l218_expanded_scene_suite.%s" % suffix),
            dpi=int(args.dpi),
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    (output / "scene_feasibility_audit.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "scene_feasibility_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "scene",
            "plant_backend",
            "obstacle_count",
            "reference_length",
            "minimum_clearance",
            "grid_shortest_path_length",
            "start_clearance",
            "goal_clearance",
            "path_exists",
            "path_clear",
            "config",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            writer.writerow({key: report[key] for key in fields})
    print(json.dumps({
        "output_dir": str(output),
        "scene_count": len(reports),
        "all_feasible": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
