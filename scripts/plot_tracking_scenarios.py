#!/usr/bin/env python3
"""Render and geometrically audit the three constrained Tracking scenes."""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.transforms import Affine2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.core.references import PolylineReference


DEFAULT_SCENES = (
    ROOT / "configs/research/mujoco_tracking_grand_infinity_l234.yaml",
    ROOT / "configs/research/mujoco_tracking_grand_s_chicane_l234.yaml",
    ROOT / "configs/research/mujoco_tracking_grand_hairpin_l234.yaml",
)


def _corridor_patches(ax, points, half_widths):
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        vector = end - start
        length = float(np.linalg.norm(vector))
        unit_normal = np.asarray((-vector[1], vector[0])) / length
        start_normal = half_widths[index] * unit_normal
        end_normal = half_widths[index + 1] * unit_normal
        ax.add_patch(Polygon(
            (start + start_normal, end + end_normal,
             end - end_normal, start - start_normal),
            closed=True,
            facecolor="#d9edf7",
            edgecolor="none",
            alpha=0.70,
            zorder=0,
        ))
    for point, half_width in zip(points, half_widths):
        ax.add_patch(Circle(
            point, half_width, facecolor="#d9edf7", edgecolor="none",
            alpha=0.70, zorder=0,
        ))


def _audit(config):
    task = config["task"]
    points = np.asarray(task["points"], dtype=np.float64)
    reference = PolylineReference(points)
    half_width = float(task["corridor_half_width"])
    profile = np.asarray(
        task.get("corridor_half_width_profile", ((0.0, half_width), (1.0, half_width))),
        dtype=np.float64,
    )
    footprint = float(task["footprint_radius"])
    obstacles = config.get("scene", {}).get("obstacles", ())
    obstacle_rows = []
    for obstacle in obstacles:
        position = np.asarray(obstacle["position"], dtype=np.float64)
        projection = reference.project(position)
        local_half_width = float(np.interp(
            projection.normalized_progress, profile[:, 0], profile[:, 1]
        ))
        if str(obstacle.get("type", "cylinder")) == "box":
            size = np.asarray(obstacle["size"], dtype=np.float64)
            relative_yaw = float(obstacle.get("yaw", 0.0)) - projection.tangent_heading
            lateral_extent = float(
                abs(np.sin(relative_yaw)) * size[0]
                + abs(np.cos(relative_yaw)) * size[1]
            )
        else:
            lateral_extent = float(obstacle.get("radius", 0.0))
        signed = float(projection.signed_cross_track_error)
        left_gap = local_half_width - (signed + lateral_extent)
        right_gap = local_half_width - (-signed + lateral_extent)
        best_gap = max(left_gap, right_gap)
        diameter = 2.0 * footprint
        recommended_safety = 0.15 * diameter
        required_gap = diameter + 2.0 * recommended_safety
        bypass_margin = best_gap - required_gap
        obstacle_rows.append({
            "label": str(obstacle.get("label", "obstacle")),
            "type": str(obstacle.get("type", "cylinder")),
            "position": position.tolist(),
            "lateral_half_extent": lateral_extent,
            "cross_track_to_reference": projection.cross_track_error,
            "signed_cross_track_to_reference": signed,
            "local_corridor_half_width": local_half_width,
            "best_free_gap": best_gap,
            "required_gap_with_0.15D_safety": required_gap,
            "recommended_safety_bypass_margin": bypass_margin,
            "safe_bypass_exists": bool(bypass_margin > 0.0),
        })
    return {
        "scene": config["scene"]["name"],
        "path_length": reference.total_length,
        "corridor_half_width": half_width,
        "corridor_full_width_nominal": 2.0 * half_width,
        "corridor_full_width_min": 2.0 * float(np.min(profile[:, 1])),
        "footprint_radius": footprint,
        "centre_clearance_at_boundary_nominal": half_width - footprint,
        "obstacles": obstacle_rows,
        "geometry_passed": bool(all(
            row["safe_bypass_exists"] for row in obstacle_rows
        )),
    }


def render(scene_paths, output_dir):
    configs = [load_yaml(path) for path in scene_paths]
    audits = [_audit(config) for config in configs]
    fig, axes = plt.subplots(
        1, len(configs), figsize=(16.2, 5.5), constrained_layout=True
    )
    if len(configs) == 1:
        axes = (axes,)
    for label, ax, config, audit in zip("ABC", axes, configs, audits):
        task = config["task"]
        points = np.asarray(task["points"], dtype=np.float64)
        half_width = float(task["corridor_half_width"])
        footprint = float(task["footprint_radius"])
        reference = PolylineReference(points)
        progress = reference.cumulative / reference.total_length
        profile = np.asarray(
            task.get("corridor_half_width_profile", ((0.0, half_width), (1.0, half_width))),
            dtype=np.float64,
        )
        half_widths = np.interp(progress, profile[:, 0], profile[:, 1])
        _corridor_patches(ax, points, half_widths)
        ax.plot(points[:, 0], points[:, 1], color="#1f4e79", linewidth=2.2,
                marker="o", markersize=2.8, label="Reference centreline", zorder=3)
        ax.scatter(*points[0], marker="^", s=90, color="#2ca02c", zorder=6,
                   label="Start")
        ax.scatter(*points[-1], marker="*", s=150, color="#ffbf00", zorder=6,
                   edgecolor="black", linewidth=0.5, label="Goal")
        ax.add_patch(Circle(points[0], footprint, fill=False, edgecolor="#2ca02c",
                            linewidth=1.5, linestyle="--", zorder=5,
                            label="Robot footprint"))
        for obstacle in config.get("scene", {}).get("obstacles", ()):
            position = np.asarray(obstacle["position"], dtype=np.float64)
            if str(obstacle.get("type", "cylinder")) == "box":
                size = np.asarray(obstacle["size"], dtype=np.float64)
                yaw = float(obstacle.get("yaw", 0.0))
                patch = Rectangle(
                    (-size[0], -size[1]), 2.0 * size[0], 2.0 * size[1],
                    facecolor="#d62728", edgecolor="#7f0000", linewidth=1.0,
                    zorder=5, label="Static obstacle",
                )
                patch.set_transform(
                    Affine2D().rotate(yaw).translate(*position) + ax.transData
                )
                ax.add_patch(patch)
            else:
                radius = float(obstacle["radius"])
                ax.add_patch(Circle(position, radius, facecolor="#d62728",
                                    edgecolor="#7f0000", linewidth=1.0, zorder=5,
                                    label="Static obstacle"))
                ax.add_patch(Circle(position, radius + footprint, fill=False,
                                    edgecolor="#d62728", linewidth=1.0,
                                    linestyle=":", zorder=4,
                                    label="Collision envelope"))
            ax.annotate(
                str(obstacle.get("label", "O")), position,
                xytext=(4, 5), textcoords="offset points", fontsize=8,
                color="#7f0000", zorder=7,
            )
        field = config.get("scene", {}).get("field_size", (6.5, 6.5))
        ax.set_xlim(0.0, float(field[0]))
        ax.set_ylim(0.0, float(field[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.20, linewidth=0.6)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        title = config["scene"]["name"].replace("tracking_", "").replace("_l234", "")
        ax.set_title(
            f"({label}) {title}\nlength={audit['path_length']:.2f} m, "
            f"corridor={audit['corridor_full_width_min']:.2f}–"
            f"{audit['corridor_full_width_nominal']:.2f} m"
        )
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="outside lower center", ncol=6,
               frameon=False)
    fig.suptitle(
        "L234 Grand Tracking benchmark design (top view; axes in metres)",
        fontsize=14,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "tracking_scene_design_l234.png"
    pdf = output_dir / "tracking_scene_design_l234.pdf"
    audit_path = output_dir / "tracking_scene_geometry_audit_l234.json"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    audit_path.write_text(
        json.dumps(audits, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return png, pdf, audit_path, audits


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research_artifacts/tracking_l234_design/figures")
    parser.add_argument("--scenes", nargs="*", default=[str(path) for path in DEFAULT_SCENES])
    args = parser.parse_args(argv)
    png, pdf, audit, rows = render(
        [Path(value).resolve() for value in args.scenes],
        Path(args.output_dir).resolve(),
    )
    if not all(row["geometry_passed"] for row in rows):
        raise SystemExit("tracking geometry audit failed")
    print(json.dumps({
        "png": str(png), "pdf": str(pdf), "audit": str(audit),
        "geometry_passed": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
