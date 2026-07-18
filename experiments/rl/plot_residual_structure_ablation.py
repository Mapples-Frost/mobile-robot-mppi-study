#!/usr/bin/env python3
"""Create the publication figure and compact table for L60/L62."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "mlp": "#56B4E9",
    "icode": "#D55E00",
    "direct": "#009E73",
    "point": "#2E3440",
}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_table(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(offline, closed_loop, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
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

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.8), layout="constrained"
    )

    # Panel A: block-level offline reductions. Individual training seeds remain
    # visible so the three model replicates are not hidden behind a mean bar.
    splits = ("test", "unseen")
    x = np.arange(len(splits), dtype=float)
    width = 0.31
    rng = np.random.default_rng(2026071641)
    for offset, model, label in (
        (-width / 2, "mlp", "Direct MLP"),
        (width / 2, "icode", "ICODE (control-affine)"),
    ):
        values = np.asarray([
            100.0 * offline["aggregates"][
                "mean_%s_%s_h36_rollout_reduction" % (model, split)
            ]
            for split in splits
        ])
        bars = axes[0].bar(
            x + offset, values, width=width * 0.92,
            color=COLORS[model], edgecolor="white", linewidth=0.6,
            label=label, zorder=2,
        )
        for split_index, split in enumerate(splits):
            points = [
                100.0 * block[model]["%s_h36_rollout_reduction" % split]
                for block in offline["model_blocks"]
            ]
            jitter = rng.uniform(-0.025, 0.025, size=len(points))
            axes[0].scatter(
                np.full(len(points), x[split_index] + offset) + jitter,
                points, s=13, facecolors="white", edgecolors=COLORS["point"],
                linewidths=0.7, zorder=3,
            )
        for bar, value in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, value + 0.8,
                "%.1f" % value, ha="center", va="bottom", fontsize=7.3,
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(("Held-out paths", "Unseen reverse-S"))
    axes[0].set_ylabel("H=36 RMSE reduction (%)")
    axes[0].set_ylim(0.0, 52.0)
    axes[0].set_title("(a) Offline prediction")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2)

    # Panel B: sealed closed-loop paired effects. The direct ICODE-vs-MLP
    # contrast is shown explicitly rather than inferred by subtracting bars.
    contrasts = (
        ("mlp_vs_nominal", "MLP - Nominal", COLORS["mlp"]),
        ("icode_vs_nominal", "ICODE - Nominal", COLORS["icode"]),
        ("icode_vs_mlp", "ICODE - MLP", COLORS["direct"]),
    )
    means = np.asarray([
        100.0 * closed_loop["hierarchical_bootstrap"][key]["estimate"]
        for key, _, _ in contrasts
    ])
    lowers = np.asarray([
        100.0 * closed_loop["hierarchical_bootstrap"][key]["ci95_lower"]
        for key, _, _ in contrasts
    ])
    uppers = np.asarray([
        100.0 * closed_loop["hierarchical_bootstrap"][key]["ci95_upper"]
        for key, _, _ in contrasts
    ])
    y = np.arange(len(contrasts))
    bars = axes[1].barh(
        y, means, xerr=np.vstack((means - lowers, uppers - means)),
        color=[color for _, _, color in contrasts], height=0.58,
        edgecolor="white", linewidth=0.6, capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#2E3440"},
        zorder=2,
    )
    axes[1].axvline(0.0, color="#555555", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([label for _, label, _ in contrasts])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Cross-track RMSE improvement (cm; 95% CI)")
    axes[1].set_title("(b) Sealed closed-loop confirmation")
    axes[1].set_xlim(0.0, max(uppers) * 1.28)
    for bar, mean, upper in zip(bars, means, uppers):
        axes[1].text(
            upper + 0.04, bar.get_y() + bar.get_height() / 2,
            "%.2f" % mean, va="center", fontsize=7.5,
        )
    stem = output_dir / "fig_l60_l62_residual_structure_ablation"
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".png", dpi=300)
    plt.close(fig)

    rows = []
    for key, label, _ in contrasts:
        bootstrap = closed_loop["hierarchical_bootstrap"][key]
        pooled = closed_loop["pooled_contrasts"][key]
        rows.append({
            "contrast": label,
            "pairs": pooled["pairs"],
            "mean_cross_track_improvement_m": "%.9f" % bootstrap["estimate"],
            "ci95_lower_m": "%.9f" % bootstrap["ci95_lower"],
            "ci95_upper_m": "%.9f" % bootstrap["ci95_upper"],
            "mean_relative_cross_track_reduction": "%.9f" % pooled[
                "mean_relative_cross_track_reduction"
            ],
            "net_success_gain": pooled["net_success_gain"],
            "net_collision_increase": pooled["net_collision_increase"],
        })
    _write_table(output_dir / "table_l60_l62_residual_structure.csv", rows)
    return stem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-summary", required=True)
    parser.add_argument("--closed-loop-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    stem = plot(
        _load(args.offline_summary), _load(args.closed_loop_summary),
        args.output_dir,
    )
    print(str(stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
