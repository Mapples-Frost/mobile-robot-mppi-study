#!/usr/bin/env python3
"""Create the publication figure/table for L67/L68 observation robustness."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "nominal": "#8C8C8C",
    "mlp": "#56B4E9",
    "icode": "#D55E00",
    "development": "#0072B2",
    "confirmation": "#009E73",
    "point": "#2E3440",
}
METHODS = (
    ("traditional_nominal", "Nominal", COLORS["nominal"]),
    ("traditional_mlp", "MLP", COLORS["mlp"]),
    ("traditional_icode", "ICODE", COLORS["icode"]),
)
DOMAIN_LABELS = {
    "clean_ground_truth": "Clean state",
    "latency_100ms": "100 ms latency",
    "combined_medium_100ms": "Noise + latency",
}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _episodes(root):
    rows = []
    for block in range(3):
        rows.extend(_csv(Path(root) / ("block_%d" % block) / "episodes.csv"))
    return rows


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(development, confirmation, development_root, confirmation_root, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.7,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), layout="constrained")

    # (a) Direct sealed ICODE-vs-MLP effect across primary observation domains.
    rows = {
        row["physics_domain"]: row
        for row in confirmation["contrasts"]["icode_vs_mlp"]["per_domain"]
    }
    order = ("clean_ground_truth", "latency_100ms", "combined_medium_100ms")
    x = np.arange(len(order), dtype=float)
    means = np.asarray([
        100.0 * float(rows[name]["mean_relative_cross_track_reduction"])
        for name in order
    ])
    bars = axes[0].bar(
        x, means, width=0.58, color=COLORS["icode"],
        edgecolor="white", linewidth=0.5, zorder=2,
    )
    axes[0].axhline(0.0, color="#555555", linewidth=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DOMAIN_LABELS[name] for name in order], rotation=28, ha="right")
    axes[0].set_ylabel("ICODE - MLP RMSE reduction (%)")
    axes[0].set_title("(a) Sealed observation domains")
    axes[0].set_ylim(0.0, max(means) * 1.22)
    for bar, value in zip(bars, means):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, value + 0.2,
            "%.1f" % value, ha="center", fontsize=7.2,
        )

    # (b) Development-to-confirmation hierarchical replication.
    stages = (("Development", development, COLORS["development"]),
              ("Sealed", confirmation, COLORS["confirmation"]))
    estimates = np.asarray([
        1000.0 * stage["contrasts"]["icode_vs_mlp"]["primary"]["bootstrap"]["estimate"]
        for _, stage, _ in stages
    ])
    lowers = np.asarray([
        1000.0 * stage["contrasts"]["icode_vs_mlp"]["primary"]["bootstrap"]["ci95_lower"]
        for _, stage, _ in stages
    ])
    uppers = np.asarray([
        1000.0 * stage["contrasts"]["icode_vs_mlp"]["primary"]["bootstrap"]["ci95_upper"]
        for _, stage, _ in stages
    ])
    y = np.arange(len(stages))
    axes[1].barh(
        y, estimates, xerr=np.vstack((estimates - lowers, uppers - estimates)),
        color=[color for _, _, color in stages], height=0.52,
        edgecolor="white", linewidth=0.5, capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": COLORS["point"]},
        zorder=2,
    )
    axes[1].axvline(0.0, color="#555555", linewidth=0.7)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([label for label, _, _ in stages])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("ICODE - MLP improvement\n(mm; hierarchical 95% CI)")
    axes[1].set_title("(b) Primary-effect replication")
    axes[1].set_xlim(0.0, max(uppers) * 1.27)
    for index, (estimate, upper) in enumerate(zip(estimates, uppers)):
        axes[1].text(upper + 0.08, index, "%.2f" % estimate, va="center", fontsize=7.2)

    # (c) Explicitly expose the non-replicating wheel-odometry stress result.
    stage_episodes = (("Development", _episodes(development_root)),
                      ("Sealed", _episodes(confirmation_root)))
    width = 0.23
    stage_x = np.arange(2, dtype=float)
    table_rows = []
    for method_index, (condition, label, color) in enumerate(METHODS):
        rates = []
        counts = []
        for stage, episode_rows in stage_episodes:
            selected = [
                row for row in episode_rows
                if row["physics_domain"] == "raw_wheel_odometry"
                and row["condition"] == condition
            ]
            successes = sum(str(row["success"]).lower() in ("true", "1") for row in selected)
            rates.append(100.0 * successes / len(selected))
            counts.append((successes, len(selected)))
            table_rows.append({
                "stage": stage,
                "condition": condition,
                "successes": successes,
                "episodes": len(selected),
                "success_rate": "%.9f" % (successes / len(selected)),
            })
        offset = (method_index - 1) * width
        bars = axes[2].bar(
            stage_x + offset, rates, width=width * 0.9,
            color=color, edgecolor="white", linewidth=0.5,
            label=label, zorder=2,
        )
        for bar, (successes, episodes) in zip(bars, counts):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                "%d/%d" % (successes, episodes), ha="center", fontsize=6.7,
            )
    axes[2].set_xticks(stage_x)
    axes[2].set_xticklabels(("Development", "Sealed"))
    axes[2].set_ylabel("Raw wheel-odometry success (%)")
    axes[2].set_title("(c) Non-gating stress result")
    axes[2].set_ylim(0.0, 100.0)
    axes[2].legend(loc="upper center", ncol=3)

    stem = output_dir / "fig_l67_l68_observation_robustness"
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".png", dpi=300)
    plt.close(fig)
    _write_csv(output_dir / "table_l68_wheel_odometry_stress.csv", table_rows)
    return stem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-summary", required=True)
    parser.add_argument("--confirmation-summary", required=True)
    parser.add_argument("--development-root", required=True)
    parser.add_argument("--confirmation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    stem = plot(
        _json(args.development_summary),
        _json(args.confirmation_summary),
        args.development_root,
        args.confirmation_root,
        args.output_dir,
    )
    print(stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

