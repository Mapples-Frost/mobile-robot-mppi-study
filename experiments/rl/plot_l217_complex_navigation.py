#!/usr/bin/env python3
"""Publication figures for the L217 sealed complex-navigation benchmark."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
import numpy as np
import yaml


METHODS = (
    ("traditional_mppi", "Traditional MPPI"),
    ("icode_mppi", "ICODE-MPPI"),
    ("rl_driven_mppi", "RL-driven MPPI"),
    ("simple_combination", "Simple combination"),
    ("value_fixed", "Value-aligned"),
    ("ordinary_adaptive", "Role-aware HSS"),
    ("full_proposed", "Full proposed"),
)
CORE_METHODS = (
    ("simple_combination", "Simple"),
    ("value_fixed", "+ Value"),
    ("ordinary_adaptive", "+ HSS"),
    ("full_proposed", "Full"),
)
SCENES = (
    ("lab_complex", "Lab complex"),
    ("narrow_corridor", "Narrow corridor"),
    ("u_trap_long_board", "Long-board U-trap"),
)
DOMAINS = (
    ("nominal_seen", "Seen"),
    ("combined_unseen", "Unseen"),
)
COLORS = {
    "traditional_mppi": "#7F7F7F",
    "icode_mppi": "#0072B2",
    "rl_driven_mppi": "#E69F00",
    "simple_combination": "#56B4E9",
    "value_fixed": "#CC79A7",
    "ordinary_adaptive": "#009E73",
    "full_proposed": "#D55E00",
}


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(row, field):
    raw = row[field]
    if str(raw).lower() in ("true", "false"):
        return float(str(raw).lower() == "true")
    return float(raw)


def selected(rows, arm, scene=None, domain=None):
    return [
        row for row in rows
        if row["benchmark_arm"] == arm
        and (scene is None or row["scene"] == scene)
        and (domain is None or row["physics_domain"] == domain)
    ]


def mean(rows, arm, field, scene=None, domain=None):
    items = selected(rows, arm, scene, domain)
    if not items:
        raise ValueError("empty plot cell")
    return float(np.mean([value(row, field) for row in items]))


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
    })


def save(fig, output, stem):
    fig.savefig(output / (stem + ".pdf"))
    fig.savefig(output / (stem + ".png"), dpi=300)
    plt.close(fig)


def method_overview(rows, output):
    ids = [item[0] for item in METHODS]
    labels = [item[1] for item in METHODS]
    colors = [COLORS[item] for item in ids]
    success = np.asarray([100.0 * mean(rows, item, "success") for item in ids])
    steps = np.asarray([mean(rows, item, "steps") for item in ids])
    y = np.arange(len(ids))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    axes[0].barh(y, success, color=colors, height=0.65)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 106)
    axes[0].set_xlabel("Success rate (%)")
    axes[0].set_title("(a) Safety-aware goal reaching")
    for index, number in enumerate(success):
        axes[0].text(number + 1.2, index, "%.1f" % number, va="center", fontsize=7)
    axes[1].barh(y, steps, color=colors, height=0.65)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Executed control steps (lower is better)")
    axes[1].set_title("(b) Bounded completion time")
    axes[1].axvline(600, color="#333333", linestyle="--", linewidth=0.8)
    for index, number in enumerate(steps):
        axes[1].text(number + 5, index, "%.1f" % number, va="center", fontsize=7)
    fig.suptitle("L217 sealed benchmark: 10 seeds × 3 scenes × 2 physics domains", y=1.02)
    fig.tight_layout()
    save(fig, output, "fig1_l217_method_overview")


def full_simple_tradeoffs(paired, output):
    metrics = (
        ("steps", "Executed steps"),
        ("trajectory_length", "Path length"),
        ("stuck_steps", "Stuck steps"),
        ("minimum_clearance", "Minimum clearance"),
        ("control_jerk", "Control jerk"),
        ("planner_compute_ms_mean", "Planner mean time"),
    )
    comparison = paired["full_vs_simple"]["metrics"]
    centers, lows, highs = [], [], []
    for metric, _ in metrics:
        item = comparison[metric]
        denominator = abs(float(item["control_mean"]))
        scale = 100.0 / denominator
        centers.append(scale * float(item["favorable_effect"]))
        lows.append(scale * float(item["ci95"][0]))
        highs.append(scale * float(item["ci95"][1]))
    centers = np.asarray(centers)
    y = np.arange(len(metrics))
    colors = ["#009E73" if item >= 0 else "#D55E00" for item in centers]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for index in range(len(metrics)):
        ax.errorbar(
            centers[index], y[index],
            xerr=[[centers[index] - lows[index]], [highs[index] - centers[index]]],
            fmt="o", color=colors[index], ecolor=colors[index], capsize=3,
            markersize=7, markeredgecolor="white", markeredgewidth=0.7,
        )
    ax.axvline(0.0, color="#333333", linewidth=0.9, linestyle="--")
    ax.set_yticks(y, [item[1] for item in metrics])
    ax.invert_yaxis()
    ax.set_xlabel("Relative favorable change of Full vs. Simple (%)")
    ax.set_title("Full method improves efficiency, with clearance/jerk trade-offs")
    ax.text(0.01, -0.18, "Points: seed-cluster mean; bars: frozen 10,000-bootstrap 95% CI",
            transform=ax.transAxes, fontsize=7.5)
    fig.tight_layout()
    save(fig, output, "fig2_l217_full_vs_simple_tradeoffs")


def core_ablation(rows, output):
    labels = [item[1] for item in CORE_METHODS]
    ids = [item[0] for item in CORE_METHODS]
    colors = [COLORS[item] for item in ids]
    seed_values = defaultdict(list)
    for arm in ids:
        by_seed = defaultdict(list)
        for row in selected(rows, arm):
            by_seed[int(row["seed"])].append(value(row, "steps"))
        seed_values[arm] = [float(np.mean(by_seed[seed])) for seed in sorted(by_seed)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    rng = np.random.RandomState(20260724)
    for index, arm in enumerate(ids):
        values = np.asarray(seed_values[arm])
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        axes[0].scatter(index + jitter, values, s=17, color=colors[index], alpha=0.62)
        axes[0].plot(index, np.mean(values), marker="D", markersize=7, color="#111111")
    axes[0].set_xticks(range(len(ids)), labels)
    axes[0].set_ylabel("Seed-cluster mean executed steps")
    axes[0].set_title("(a) Frozen 2×2 coupling ablation")

    grid = np.zeros((len(SCENES), len(DOMAINS)))
    for i, (scene, _) in enumerate(SCENES):
        for j, (domain, _) in enumerate(DOMAINS):
            grid[i, j] = (
                mean(rows, "simple_combination", "steps", scene, domain)
                - mean(rows, "full_proposed", "steps", scene, domain)
            )
    limit = max(abs(float(np.min(grid))), abs(float(np.max(grid))))
    image = axes[1].imshow(grid, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    axes[1].set_xticks(range(len(DOMAINS)), [item[1] for item in DOMAINS])
    axes[1].set_yticks(range(len(SCENES)), [item[1] for item in SCENES])
    axes[1].set_title("(b) Full step reduction by stratum")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            axes[1].text(j, i, "%+.1f" % grid[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Simple − Full steps")
    fig.tight_layout()
    save(fig, output, "fig3_l217_core_ablation")


def locate_run(provenance, arm, scene, domain, seed):
    name = "%s__%s__%s__seed%d" % (arm, scene, domain, seed)
    for shard in provenance["source_shards"]:
        candidate = Path(shard["path"]) / "runs" / arm / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(name)


def draw_obstacles(ax, config):
    for obstacle in config.get("scene", {}).get("obstacles", []):
        x, y = obstacle["position"]
        if obstacle["type"] == "cylinder":
            patch = Circle((x, y), float(obstacle["radius"]), facecolor="#777777",
                           edgecolor="#333333", linewidth=0.5, alpha=0.70)
        else:
            hx, hy = obstacle["size"]
            patch = Rectangle((x - hx, y - hy), 2 * hx, 2 * hy,
                              facecolor="#777777", edgecolor="#333333",
                              linewidth=0.5, alpha=0.70)
            yaw = float(obstacle.get("yaw", 0.0))
            patch.set_transform(Affine2D().rotate_around(x, y, yaw) + ax.transData)
        ax.add_patch(patch)


def fixed_seed_trajectories(result_dir, rows, output):
    provenance = json.loads((result_dir / "provenance.json").read_text(encoding="utf-8"))
    seed = min(int(item) for item in provenance["sealed_seeds"])
    domain = "combined_unseen"
    fig, axes = plt.subplots(1, len(SCENES), figsize=(7.8, 2.9), sharex=True, sharey=True)
    for ax, (scene, label) in zip(axes, SCENES):
        for arm, linestyle, linewidth in (
            ("simple_combination", "--", 1.4),
            ("full_proposed", "-", 2.0),
        ):
            run = locate_run(provenance, arm, scene, domain, seed)
            trajectory = read_csv(run / "trajectory.csv")
            xy = np.asarray([[float(row["x"]), float(row["y"])] for row in trajectory])
            ax.plot(xy[:, 0], xy[:, 1], linestyle=linestyle, linewidth=linewidth,
                    color=COLORS[arm], label=dict(CORE_METHODS)[arm])
        config = yaml.safe_load((run / "config_resolved.yaml").read_text(encoding="utf-8"))
        draw_obstacles(ax, config)
        goal = config["task"]["position"]
        ax.scatter(xy[0, 0], xy[0, 1], marker="o", s=24, color="#111111", zorder=5)
        ax.scatter(goal[0], goal[1], marker="*", s=70, color="#F0E442",
                   edgecolor="#222222", linewidth=0.5, zorder=5)
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        # The controller is intentionally free to route outside the obstacle
        # bounding box.  These limits include every trajectory of the fixed
        # seed instead of cropping the lower/right detours.
        ax.set_xlim(-0.35, 4.2)
        ax.set_ylim(-0.8, 3.35)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Fixed first sealed seed 78006, combined-unseen physics", y=1.03)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, output, "fig4_l217_fixed_seed_trajectories")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args(argv)
    result_dir = Path(args.result_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else result_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(result_dir / "progress.csv")
    paired = json.loads((result_dir / "paired_comparisons.json").read_text(encoding="utf-8"))
    style()
    method_overview(rows, output)
    full_simple_tradeoffs(paired, output)
    core_ablation(rows, output)
    fixed_seed_trajectories(result_dir, rows, output)
    generated = sorted(path.name for path in output.glob("fig*_l217_*.*"))
    print(json.dumps({"output_dir": str(output), "files": generated}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
