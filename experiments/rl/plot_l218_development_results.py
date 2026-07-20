#!/usr/bin/env python3
"""Plot the audited L218 MuJoCo development qualification results."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobile_robot_mppi.core.config import load_yaml

from plot_l218_expanded_scenes import SCENES, _obstacle


METHODS = ("simple_combination", "full_proposed")
METHOD_LABELS = {
    "simple_combination": "Simple combination",
    "full_proposed": "Full proposed",
}
COLORS = {
    "simple_combination": "#D55E00",
    "full_proposed": "#0072B2",
}
MARKERS = {
    "simple_combination": "x",
    "full_proposed": "o",
}


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scene_key(filename):
    return filename.replace("mujoco_l218_", "").replace("_polyline.yaml", "")


def _draw_geometry(ax, config):
    radius = float(config["plant"]["robot"]["collision_radius"])
    for obstacle in config["scene"].get("obstacles", ()):
        _obstacle(
            ax,
            obstacle,
            0.0,
            facecolor="#9CA3AF",
            edgecolor="#4B5563",
            linewidth=0.55,
            alpha=0.78,
        )
    points = np.asarray(config["task"]["points"], dtype=np.float64)
    ax.plot(
        points[:, 0], points[:, 1], "--", color="#111827",
        linewidth=1.0, alpha=0.7, label="Reference",
    )
    start = np.asarray(config["experiment"]["initial_state"][:2])
    goal = points[-1]
    ax.scatter(*start, s=34, color="#009E73", edgecolor="white", zorder=6)
    ax.scatter(*goal, s=72, marker="*", color="#F0E442", edgecolor="#8A6D00", zorder=6)
    ax.set_xlim(0.0, 6.5)
    ax.set_ylim(0.0, 6.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(color="#E5E7EB", linewidth=0.45, alpha=0.7)
    return radius


def _trajectory_path(results_root, seed, scene, method):
    root = results_root / (
        "l218_dev_screen_coupled_gate_v2_s%d_%s_seed%d" %
        (seed, scene, seed)
    )
    matches = list((root / "runs" / method).glob("*/trajectory.csv"))
    if len(matches) != 1:
        raise ValueError("expected one trajectory for %s/%s/%d" % (scene, method, seed))
    return matches[0]


def plot_trajectories(output, results_root, seed, dpi):
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.9), constrained_layout=True)
    for index, ((title, filename), ax) in enumerate(zip(SCENES, axes.flat)):
        config = load_yaml(ROOT / "configs" / "research" / filename)
        _draw_geometry(ax, config)
        scene = _scene_key(filename)
        for method in METHODS:
            rows = _read_csv(_trajectory_path(results_root, seed, scene, method))
            xy = np.asarray([[float(row["x"]), float(row["y"])] for row in rows])
            ax.plot(
                xy[:, 0], xy[:, 1], color=COLORS[method], linewidth=1.5,
                alpha=0.9, label=METHOD_LABELS[method],
            )
            ax.scatter(
                xy[-1, 0], xy[-1, 1], s=24, marker=MARKERS[method],
                color=COLORS[method], zorder=7,
            )
        ax.set_title("(%s) %s" % (chr(ord("a") + index), title))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(), unique.keys(), loc="outside lower center", ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "L218 MuJoCo trajectories | development seed %d | K=30, H=36" % seed,
        fontsize=14,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output / ("l218_seed%d_trajectory_comparison.%s" % (seed, suffix)),
            dpi=dpi, bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)


def _float(row, name):
    value = str(row[name]).strip()
    if value.lower() in {"true", "false"}:
        return float(value.lower() == "true")
    return float(value)


def plot_summary(output, analysis_dir, dpi):
    rows = _read_csv(analysis_dir / "combined_progress.csv")
    scene_order = ["l218_" + _scene_key(filename) for _, filename in SCENES]
    short_labels = ["Serpentine", "Giant U", "Opposed U", "Nested U", "Forest", "Rings"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark_arm"], row["scene"])].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2), constrained_layout=True)
    metrics = (
        ("success", "Success rate", True),
        ("path_completion_ratio", "Path completion ratio", True),
        ("path_cross_track_rmse_recomputed", "Cross-track RMSE (m)", False),
        ("safety_interventions", "Safety interventions (count)", False),
    )
    x = np.arange(len(scene_order), dtype=np.float64)
    offsets = {"simple_combination": -0.18, "full_proposed": 0.18}
    for panel, (metric, ylabel, bounded) in zip(axes.flat, metrics):
        for method in METHODS:
            means = []
            values_by_scene = []
            for scene in scene_order:
                values = np.asarray(
                    [_float(row, metric) for row in grouped[(method, scene)]],
                    dtype=np.float64,
                )
                means.append(float(np.mean(values)))
                values_by_scene.append(values)
            panel.bar(
                x + offsets[method], means, width=0.34,
                color=COLORS[method], alpha=0.78, label=METHOD_LABELS[method],
            )
            for index, values in enumerate(values_by_scene):
                jitter = np.linspace(-0.045, 0.045, len(values))
                panel.scatter(
                    np.full(len(values), x[index] + offsets[method]) + jitter,
                    values, s=18, facecolor="white", edgecolor=COLORS[method],
                    linewidth=0.8, zorder=4,
                )
        panel.set_ylabel(ylabel)
        panel.set_xticks(x, short_labels, rotation=18, ha="right")
        panel.grid(axis="y", color="#E5E7EB", linewidth=0.55, alpha=0.8)
        if bounded:
            panel.set_ylim(0.0, 1.05)
    axes.flat[0].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "L218 MuJoCo development qualification | 3 seeds × 6 scenes\n"
        "dots are seed-level observations; scenes are repeated strata",
        fontsize=13.5,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output / ("l218_development_summary.%s" % suffix),
            dpi=dpi, bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)

    gate = json.loads((analysis_dir / "development_gate_decision.json").read_text(encoding="utf-8"))
    (output / "figure_metadata.json").write_text(
        json.dumps({
            "status": "development_qualification_not_formal_claim",
            "simulator": "MuJoCo 3.2.3 headless",
            "seeds": [91001, 91002, 91003],
            "rollouts_per_decision": 30,
            "mppi_horizon": 36,
            "physics_domain": "nominal_seen",
            "gate_interpretation": gate["interpretation"],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trajectory-seed", type=int, default=91001)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_trajectories(
        output, Path(args.results_root).resolve(), int(args.trajectory_seed), int(args.dpi)
    )
    plot_summary(output, Path(args.analysis_dir).resolve(), int(args.dpi))
    print(json.dumps({"output_dir": str(output), "figure_sets": 2}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
