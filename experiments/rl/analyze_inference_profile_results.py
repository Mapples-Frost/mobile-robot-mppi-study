#!/usr/bin/env python3
"""Generate L28 checkpoint summaries and publication figures."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.run_scene_complexity_gate_ablation import _write_csv


CONDITIONS = (
    "reference_full_diagnostics",
    "selected_critic_only",
    "selected_critic_base_reuse",
)
LABELS = ("Reference", "Selected", "+ Base reuse")


def _read(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _checkpoint_summary(pairs):
    output = []
    for candidate in CONDITIONS[1:]:
        for training_seed in sorted({int(row["training_seed"]) for row in pairs}):
            rows = [
                row for row in pairs
                if row["candidate"] == candidate
                and row["scene_role"] == "blocking"
                and int(row["training_seed"]) == training_seed
            ]
            reference_planner = float(np.mean([
                float(row["reference_planner_ms"]) for row in rows
            ]))
            candidate_planner = float(np.mean([
                float(row["candidate_planner_ms"]) for row in rows
            ]))
            reference_prior = float(np.mean([
                float(row["reference_active_prior_ms"]) for row in rows
            ]))
            candidate_prior = float(np.mean([
                float(row["candidate_active_prior_ms"]) for row in rows
            ]))
            output.append({
                "candidate": candidate,
                "training_seed": training_seed,
                "pairs": len(rows),
                "active_prior_reduction_fraction": (
                    1.0 - candidate_prior / reference_prior
                ),
                "blocking_planner_reduction_fraction": (
                    1.0 - candidate_planner / reference_planner
                ),
            })
    return output


def _blocking_components(rows):
    lookup = {
        row["condition"]: row for row in rows
        if row["scene_role"] == "blocking"
    }
    categories = (
        ("Encode + gate features", (
            "profile_prior_fallback_ms_active",
            "profile_prior_encode_normalize_ms_active",
            "profile_prior_gate_features_ms_active",
        )),
        ("Actor", ("profile_prior_actor_ms_active",)),
        ("Advantage critics", ("profile_prior_advantage_ms_active",)),
        ("Decoder", ("profile_prior_decoder_ms_active",)),
    )
    values = []
    for condition in CONDITIONS:
        row = lookup[condition]
        parts = [
            sum(float(row[name]) for name in fields)
            for _, fields in categories
        ]
        total = float(row["profile_prior_total_ms_active"])
        parts.append(max(0.0, total - sum(parts)))
        values.append(parts)
    return [name for name, _ in categories] + ["Other"], np.asarray(values)


def _plot(components, timing, output):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
    })
    component_names, component_values = components
    colors = ("#56B4E9", "#E69F00", "#D55E00", "#009E73", "#8C8C8C")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(len(CONDITIONS))
    bottom = np.zeros(len(CONDITIONS))
    for index, name in enumerate(component_names):
        axes[0].bar(
            x, component_values[:, index], 0.62,
            bottom=bottom, label=name, color=colors[index],
            edgecolor="white", linewidth=0.5,
        )
        bottom += component_values[:, index]
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LABELS)
    axes[0].set_ylabel("Active prior time (ms/step)")
    axes[0].set_title("(a) Active-prior component profile", loc="left", fontweight="bold")
    axes[0].set_ylim(0.0, 3.35)
    axes[0].legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.0,
        handletextpad=0.5,
    )

    candidates = CONDITIONS[1:]
    reductions = np.asarray([
        100.0 * timing[name]["blocking_planner_reduction_fraction"]
        for name in candidates
    ])
    low = np.asarray([
        100.0 * timing[name]["blocking_planner_reduction_ci95"][0]
        for name in candidates
    ])
    high = np.asarray([
        100.0 * timing[name]["blocking_planner_reduction_ci95"][1]
        for name in candidates
    ])
    bars = axes[1].bar(
        np.arange(2), reductions, 0.56,
        color=("#0072B2", "#009E73"), edgecolor="white", linewidth=0.5,
    )
    axes[1].errorbar(
        np.arange(2), reductions,
        yerr=np.vstack((reductions - low, high - reductions)),
        fmt="none", ecolor="#333333", capsize=3, linewidth=0.8,
    )
    axes[1].axhline(5.0, color="#D55E00", linestyle="--", linewidth=1.0, label="Preregistered 5% gate")
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(("Selected critic", "+ Base reuse"))
    axes[1].set_ylabel("Blocking planner reduction (%)")
    axes[1].set_title("(b) End-to-end effect", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].set_ylim(0.0, max(7.0, float(high.max()) + 0.8))
    for bar, value in zip(bars, reductions):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.18,
            "%.2f%%" % value,
            ha="center", va="bottom", fontsize=8,
        )
    fig.subplots_adjust(wspace=0.38)
    fig.savefig(output / "fig_l28_inference_profile.pdf")
    fig.savefig(output / "fig_l28_inference_profile.png", dpi=300)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.input_dir).resolve()
    components = _read(output / "component_summary.csv")
    pairs = _read(output / "paired_timing.csv")
    gate = json.loads(
        (output / "development_gate.json").read_text(encoding="utf-8")
    )
    checkpoint = _checkpoint_summary(pairs)
    _write_csv(output / "checkpoint_timing.csv", checkpoint)
    _plot(_blocking_components(components), gate["timing"], output)
    report = {
        "checkpoint_timing": checkpoint,
        "figure": "fig_l28_inference_profile.pdf",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
