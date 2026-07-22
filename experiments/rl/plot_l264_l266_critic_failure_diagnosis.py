#!/usr/bin/env python3
"""Render the frozen L264-L266 diagnostic evidence as PNG and PDF."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # bundled document runtime provides Pillow only
    plt = None
    from PIL import Image, ImageDraw, ImageFont


DEFAULT_INPUT = Path(
    "results/research_platform/rl/l264_l266_critic_failure_diagnosis"
)


def _read(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label(text: str) -> str:
    replacements = {
        "in_support": "in support",
        "near_support": "near support",
        "out_of_support": "out of support",
        "normal_tracking": "normal",
        "near_boundary": "near boundary",
        "off_path": "off path",
        "obstacle_avoidance": "obstacle",
        "terminal_approach": "terminal",
        "cte_lt_0.25": "<0.25",
        "cte_0.25_0.75": "0.25–0.75",
        "cte_0.75_1.5": "0.75–1.5",
        "cte_1.5_3.0": "1.5–3.0",
        "cte_gt_3.0": ">3.0",
    }
    return replacements.get(text, text.replace("l260_train_", "").replace("_", " "))


def _pillow_font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size=size)


def _pillow_bar_panel(draw, box, title, labels, values, color, *, log=False, ymin=None, ymax=None):
    left, top, right, bottom = box
    title_font = _pillow_font(21, bold=True)
    label_font = _pillow_font(15)
    tick_font = _pillow_font(14)
    draw.text((left, top), title, fill="#111111", font=title_font)
    plot_left, plot_top = left + 72, top + 45
    plot_right, plot_bottom = right - 18, bottom - 70
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#333333", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#333333", width=2)
    numeric = np.asarray(values, dtype=float)
    transformed = np.log10(np.maximum(numeric, 1.0)) if log else numeric
    low = float(np.min(transformed)) if ymin is None else float(ymin)
    high = float(np.max(transformed)) if ymax is None else float(ymax)
    if high <= low:
        high = low + 1.0
    if not log and low > 0:
        low = 0.0
    zero_y = plot_bottom - int((0.0 - low) / (high - low) * (plot_bottom - plot_top))
    if low < 0 < high:
        draw.line((plot_left, zero_y, plot_right, zero_y), fill="#888888", width=1)
    count = max(len(labels), 1)
    slot = (plot_right - plot_left) / count
    for index, (label, value, mapped) in enumerate(zip(labels, numeric, transformed)):
        x0 = int(plot_left + index * slot + slot * 0.16)
        x1 = int(plot_left + (index + 1) * slot - slot * 0.16)
        y = plot_bottom - int((mapped - low) / (high - low) * (plot_bottom - plot_top))
        base = zero_y if low < 0 else plot_bottom
        draw.rectangle((x0, min(y, base), x1, max(y, base)), fill=color[index] if isinstance(color, list) else color)
        value_text = "%.3g" % value
        bbox = draw.textbbox((0, 0), value_text, font=tick_font)
        draw.text(((x0 + x1 - (bbox[2] - bbox[0])) / 2, min(y, base) - 20), value_text, fill="#222222", font=tick_font)
        short = label if len(label) <= 15 else label[:14] + "…"
        bbox = draw.textbbox((0, 0), short, font=label_font)
        draw.text(((x0 + x1 - (bbox[2] - bbox[0])) / 2, plot_bottom + 10), short, fill="#222222", font=label_font)


def _render_pillow(input_dir, support, stage, cte, scenes, chain_count, png, pdf):
    image = Image.new("RGB", (2400, 1700), "white")
    draw = ImageDraw.Draw(image)
    title = "L264–L266 frozen diagnosis: sparse recovery coverage drives in-support Critic failure"
    draw.text((90, 35), title, fill="#111111", font=_pillow_font(32, bold=True))
    boxes = [(90, 110, 1170, 820), (1230, 110, 2310, 820), (90, 880, 1170, 1590), (1230, 880, 2310, 1590)]
    _pillow_bar_panel(
        draw, boxes[0], "A  In-support ranking diagnostics",
        [_label(row["support"]) for row in support],
        [float(row["mean_spearman"]) for row in support],
        "#0072B2", ymin=-0.2, ymax=0.2,
    )
    stage_colors = ["#D55E00" if row["category"] == "recovery" else "#56B4E9" for row in stage]
    _pillow_bar_panel(
        draw, boxes[1], "B  Replay stages (log count)",
        [_label(row["category"]) for row in stage],
        [int(row["count"]) for row in stage], stage_colors, log=True, ymin=0, ymax=3.5,
    )
    _pillow_bar_panel(
        draw, boxes[2], "C  Cross-track-error coverage",
        [_label(row["category"]) for row in cte],
        [int(row["count"]) for row in cte], "#009E73", ymin=0, ymax=1800,
    )
    scene_names = [row["category"] for row in scenes]
    chain_values = [chain_count[name] for name in scene_names]
    _pillow_bar_panel(
        draw, boxes[3], "D  Complete recovery chains by scene",
        [_label(name) for name in scene_names], chain_values,
        ["#D55E00" if value else "#BDBDBD" for value in chain_values], ymin=0, ymax=1.25,
    )
    image.save(png, dpi=(300, 300))
    image.save(pdf, "PDF", resolution=300.0)


def render(input_dir: Path) -> tuple[Path, Path]:
    support = _read(input_dir / "l265_support_summary.csv")
    coverage = _read(input_dir / "l266_coverage_bins.csv")
    chains = _read(input_dir / "l266_recovery_chains.csv")

    stage = [row for row in coverage if row["dimension"] == "stage" and int(row["count"]) > 0]
    cte = [row for row in coverage if row["dimension"] == "cte"]
    scenes = [row for row in coverage if row["dimension"] == "scene"]
    chain_count = {row["category"]: 0 for row in scenes}
    for row in chains:
        chain_count[row["scene"]] += 1

    figures = input_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "l264_l266_critic_failure_diagnosis.png"
    pdf = figures / "l264_l266_critic_failure_diagnosis.pdf"
    if plt is None:
        _render_pillow(input_dir, support, stage, cte, scenes, chain_count, png, pdf)
        return png, pdf

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = ["#0072B2", "#E69F00", "#009E73"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    x = np.arange(len(support))
    width = 0.24
    metrics = [
        ("mean_spearman", "Spearman"),
        ("mean_top1_agreement", "Top-1"),
        ("mean_top3_agreement", "Top-3"),
    ]
    for offset, ((key, name), color) in enumerate(zip(metrics, colors)):
        values = [float(row[key]) for row in support]
        ax.bar(x + (offset - 1) * width, values, width, label=name, color=color)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, [_label(row["support"]) for row in support])
    ax.set_ylim(-0.2, 0.9)
    ax.set_ylabel("ranking / agreement")
    ax.set_title("A  Critic ranking remains poor inside replay support", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper right")

    ax = axes[0, 1]
    values = [int(row["count"]) for row in stage]
    bars = ax.bar(range(len(stage)), values, color="#56B4E9")
    recovery_index = [row["category"] for row in stage].index("recovery")
    bars[recovery_index].set_color("#D55E00")
    ax.set_yscale("log")
    ax.set_xticks(range(len(stage)), [_label(row["category"]) for row in stage], rotation=30, ha="right")
    ax.set_ylabel("transitions (log scale)")
    ax.set_title("B  Explicit recovery transitions are nearly absent", loc="left")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.15, str(value), ha="center", va="bottom", fontsize=7)

    ax = axes[1, 0]
    values = [int(row["count"]) for row in cte]
    ax.bar(range(len(cte)), values, color="#009E73")
    ax.set_xticks(range(len(cte)), [_label(row["category"]) for row in cte])
    ax.set_xlabel("cross-track error (m)")
    ax.set_ylabel("transitions")
    ax.set_title("C  Off-path states exist, but recovery events do not", loc="left")
    for index, value in enumerate(values):
        ax.text(index, value + max(values) * 0.018, str(value), ha="center", va="bottom", fontsize=7)

    ax = axes[1, 1]
    scene_names = [row["category"] for row in scenes]
    values = [chain_count[name] for name in scene_names]
    bars = ax.bar(range(len(scenes)), values, color=["#D55E00" if value else "#BDBDBD" for value in values])
    ax.set_xticks(range(len(scenes)), [_label(name) for name in scene_names], rotation=30, ha="right")
    ax.set_ylim(0, 1.35)
    ax.set_yticks([0, 1])
    ax.set_ylabel("complete recovery chains")
    ax.set_title("D  Complete recovery covers only 2 of 6 scenes", loc="left")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, str(value), ha="center", va="bottom", fontsize=8)

    fig.suptitle("L264–L266 frozen diagnosis: sparse recovery coverage drives in-support Critic failure", fontsize=12)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    png, pdf = render(args.input_dir.resolve())
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
