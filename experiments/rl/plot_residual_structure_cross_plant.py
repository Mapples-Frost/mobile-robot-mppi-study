#!/usr/bin/env python3
"""Create the publication figure/table for L64/L65 cross-plant evidence."""

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
DOMAIN_LABELS = {
    "train_anchor": "Anchor",
    "mass_light": "Light mass",
    "friction_high": "High friction",
    "torque_weak_v2": "Weak actuator",
    "delay_long": "100 ms delay",
    "combined_moderate_b": "Combined shift",
}


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _domain_map(summary, contrast):
    return {
        row["physics_domain"]: row
        for row in summary["per_domain"][contrast]
    }


def _write_table(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(development, confirmation, effects, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8.4,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.6,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linestyle": "-",
    })

    order = [
        "train_anchor", "mass_light", "friction_high", "torque_weak_v2",
        "delay_long", "combined_moderate_b",
    ]
    labels = [DOMAIN_LABELS[name] for name in order]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), layout="constrained")

    # (a) Confirmation relative reductions against nominal.
    x = np.arange(len(order), dtype=float)
    width = 0.36
    for offset, contrast, label, color in (
        (-width / 2, "mlp_vs_nominal", "MLP vs Nominal", COLORS["mlp"]),
        (width / 2, "icode_vs_nominal", "ICODE vs Nominal", COLORS["icode"]),
    ):
        lookup = _domain_map(confirmation, contrast)
        values = np.asarray([
            100.0 * float(lookup[name]["mean_relative_cross_track_reduction"])
            for name in order
        ])
        axes[0].bar(
            x + offset, values, width * 0.92, label=label, color=color,
            edgecolor="white", linewidth=0.5, zorder=2,
        )
    axes[0].axhline(0.0, color="#555555", linewidth=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("Cross-track RMSE reduction (%)")
    axes[0].set_title("(a) Learned vs nominal")
    axes[0].legend(loc="lower left")

    # (b) Direct ICODE-vs-MLP effect, retaining the three training blocks.
    direct = _domain_map(confirmation, "icode_vs_mlp")
    means_mm = np.asarray([
        1000.0 * float(direct[name]["mean_cross_track_improvement_m"])
        for name in order
    ])
    bars = axes[1].bar(
        x, means_mm, width=0.62, color=COLORS["confirmation"],
        edgecolor="white", linewidth=0.5, zorder=2,
    )
    grouped = defaultdict(lambda: defaultdict(list))
    for row in effects:
        if row["contrast"] == "icode_vs_mlp":
            grouped[row["physics_domain"]][int(row["model_block"])].append(
                1000.0 * float(row["cross_track_improvement_m"])
            )
    rng = np.random.default_rng(2026071652)
    for index, name in enumerate(order):
        block_means = [
            float(np.mean(grouped[name][block])) for block in sorted(grouped[name])
        ]
        axes[1].scatter(
            np.full(len(block_means), x[index])
            + rng.uniform(-0.055, 0.055, size=len(block_means)),
            block_means, s=13, facecolors="white", edgecolors=COLORS["point"],
            linewidths=0.7, zorder=3,
        )
    axes[1].axhline(0.0, color="#555555", linewidth=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("ICODE - MLP improvement (mm)")
    axes[1].set_title("(b) Direct structure effect")
    label_heights = []
    for index, (bar, value, name) in enumerate(zip(bars, means_mm, order)):
        block_means = [
            float(np.mean(grouped[name][block])) for block in sorted(grouped[name])
        ]
        label_height = max([value] + block_means) + 0.16
        label_heights.append(label_height)
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, label_height,
            "%.1f" % value, ha="center", va="bottom", fontsize=6.7,
        )
    axes[1].set_ylim(0.0, max(label_heights) + 0.42)

    # (c) Development-to-confirmation replication on shifted plants.
    stages = (("Development", development, COLORS["development"]),
              ("Sealed", confirmation, COLORS["confirmation"]))
    estimates = np.asarray([
        1000.0 * stage["shifted_hierarchical_bootstrap"]["icode_vs_mlp"]["estimate"]
        for _, stage, _ in stages
    ])
    lowers = np.asarray([
        1000.0 * stage["shifted_hierarchical_bootstrap"]["icode_vs_mlp"]["ci95_lower"]
        for _, stage, _ in stages
    ])
    uppers = np.asarray([
        1000.0 * stage["shifted_hierarchical_bootstrap"]["icode_vs_mlp"]["ci95_upper"]
        for _, stage, _ in stages
    ])
    y = np.arange(len(stages))
    axes[2].barh(
        y, estimates, xerr=np.vstack((estimates - lowers, uppers - estimates)),
        color=[color for _, _, color in stages], height=0.52,
        edgecolor="white", linewidth=0.5, capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": COLORS["point"]},
        zorder=2,
    )
    axes[2].axvline(0.0, color="#555555", linewidth=0.7)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([label for label, _, _ in stages])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("ICODE - MLP improvement\n(mm; hierarchical 95% CI)")
    axes[2].set_title("(c) Shifted-plant replication")
    axes[2].set_xlim(0.0, max(uppers) * 1.24)
    for index, (estimate, upper) in enumerate(zip(estimates, uppers)):
        axes[2].text(upper + 0.08, index, "%.2f" % estimate, va="center", fontsize=7)

    stem = output_dir / "fig_l64_l65_cross_plant_residual_structure"
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".png", dpi=300)
    plt.close(fig)

    rows = []
    for name in order:
        row = direct[name]
        rows.append({
            "physics_domain": name,
            "pairs": row["pairs"],
            "icode_vs_mlp_improvement_m": "%.9f" % float(
                row["mean_cross_track_improvement_m"]
            ),
            "icode_vs_mlp_relative_reduction": "%.9f" % float(
                row["mean_relative_cross_track_reduction"]
            ),
        })
    _write_table(output_dir / "table_l65_cross_plant_by_domain.csv", rows)
    return stem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-summary", required=True)
    parser.add_argument("--confirmation-summary", required=True)
    parser.add_argument("--confirmation-effects", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    stem = plot(
        _load_json(args.development_summary),
        _load_json(args.confirmation_summary),
        _read_csv(args.confirmation_effects),
        args.output_dir,
    )
    print(str(stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
