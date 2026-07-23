#!/usr/bin/env python3
"""Generate reproducible advisor-facing figures from completed L275--L284 summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(r"C:\Users\lenovo\Desktop\导师指导材料_Tracking_2026-07-23")


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linestyle": "-",
    })


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300)
    fig.savefig(OUTPUT / f"{stem}.pdf")
    plt.close(fig)


def figure_recovery_commitment() -> None:
    summary = load(
        "results/research_platform/rl/"
        "l275_recovery_commitment_continuation_diagnosis/summary.json"
    )
    metrics = summary["metrics"]
    order = ["1", "5", "10", "20", "40", "full_chain"]
    labels = ["1", "5", "10", "20", "40", "完整链"]
    fractions = [100.0 * metrics["positive_fractions"][key] for key in order]
    advantages = [metrics["median_advantages"][key] for key in order]

    fig, axes = plt.subplots(2, 1, figsize=(6.75, 5.0), sharex=True)
    x = np.arange(len(labels))
    color = "#0072B2"
    axes[0].plot(x, fractions, color=color, marker="o", linewidth=2.2)
    axes[0].fill_between(x, fractions, color=color, alpha=0.10)
    axes[0].axhline(50, color="#8C8C8C", linestyle="--", linewidth=1)
    axes[0].set_ylabel("恢复动作取得正优势的状态比例 (%)")
    axes[0].set_ylim(45, 100)
    axes[0].set_title("L275：恢复收益依赖连续动作承诺，而非单步动作")
    for xi, value in zip(x, fractions):
        axes[0].text(xi, value + 1.8, f"{value:.1f}%", ha="center", fontsize=8)

    axes[1].bar(x, advantages, color="#009E73", width=0.62)
    axes[1].axhline(0, color="#444444", linewidth=0.9)
    axes[1].set_ylabel("真实累计回报中位优势")
    axes[1].set_xlabel("教师恢复动作连续承诺步数")
    axes[1].set_xticks(x, labels)
    for xi, value in zip(x, advantages):
        axes[1].text(xi, value + 0.28, f"{value:.2f}", ha="center", fontsize=8)

    fig.text(
        0.5,
        0.01,
        "固定36个 held-out recovery states；完整链正优势比例 91.7%，单步仅 58.3%。",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "图1_L275恢复优势随动作承诺时长变化")


def figure_validation_interventions() -> None:
    specs = [
        ("L277\nSAC微调", "l277_recovery_initialized_sac_probe"),
        ("L279\n混合锚点", "l279_recovery_retention_anchor_sac_probe"),
        ("L281\n分量隔离", "l281_component_separated_anchor_sac_probe"),
        ("L284\n闭环roll-in", "l284_closed_loop_rollin_joint_anchor_sac_probe"),
    ]
    values = []
    for label, directory in specs:
        data = load(f"results/research_platform/rl/{directory}/summary.json")
        metrics = data.get("metrics") or data["validation_metrics"]
        values.append((
            label,
            metrics["median_completion_change"],
            metrics["median_cte_improvement"],
            metrics["median_goal_distance_improvement"],
        ))

    labels = [row[0] for row in values]
    arrays = [np.asarray([row[idx] for row in values]) for idx in (1, 2, 3)]
    titles = ["路径完成率变化", "CTE 改善 (m)", "终点距离改善 (m)"]
    colors = ["#56B4E9", "#E69F00", "#009E73", "#D55E00"]
    ylabels = ["Δ completion", "baseline CTE − method CTE", "baseline goal − method goal"]

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.35))
    x = np.arange(len(labels))
    for ax, arr, title, ylabel in zip(axes, arrays, titles, ylabels):
        bars = ax.bar(x, arr, color=colors, width=0.66, edgecolor="white", linewidth=0.7)
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_title(title)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_xticks(x, labels, fontsize=8)
        margin = max(0.02, 0.07 * max(1e-9, np.ptp(arr)))
        for bar, value in zip(bars, arr):
            va = "bottom" if value >= 0 else "top"
            offset = margin if value >= 0 else -margin
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.3f}",
                ha="center",
                va=va,
                fontsize=7.5,
            )
    fig.suptitle("L277–L284：干预方向有效，但跨场景与跨 seed 稳定性仍未通过", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "正值表示优于各自冻结基线；所有四轮最终 Gate 均未通过，不能只看中位数。",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout()
    save(fig, "图2_L277至L284验证集干预结果")


def figure_learning_and_forgetting() -> None:
    l276 = load(
        "results/research_platform/rl/"
        "l276_recovery_sequence_imitation_initialization/summary.json"
    )
    l278 = load(
        "results/research_platform/rl/"
        "l278_recovery_retention_diagnosis/summary.json"
    )
    l279 = load(
        "results/research_platform/rl/"
        "l279_recovery_retention_anchor_sac_probe/summary.json"
    )
    l281 = load(
        "results/research_platform/rl/"
        "l281_component_separated_anchor_sac_probe/summary.json"
    )
    l284 = load(
        "results/research_platform/rl/"
        "l284_closed_loop_rollin_joint_anchor_sac_probe/summary.json"
    )
    labels = [
        "L276 模仿初始化\n相对 source",
        "L278 SAC 6k\n相对 L276",
        "L279 混合锚点\n相对 L277",
        "L281 分量隔离\n相对 L277",
        "L284 闭环roll-in\n相对 L277",
    ]
    improvements = [
        100.0 * l276["metrics"]["median_relative_test_rmse_improvement"],
        -100.0 * l278["metrics"]["median_relative_test_rmse_increase_6k"],
        100.0 * l279["comparative_metrics"]["median_relative_test_rmse_improvement_vs_l277"],
        100.0 * l281["comparative_metrics"]["median_relative_test_rmse_improvement_vs_l277"],
        100.0 * l284["comparative_metrics"]["median_relative_test_rmse_improvement_vs_l277"],
    ]
    colors = ["#009E73" if value >= 0 else "#D55E00" for value in improvements]
    fig, ax = plt.subplots(figsize=(7.4, 3.25))
    x = np.arange(len(labels))
    bars = ax.bar(x, improvements, color=colors, width=0.64)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x, labels, fontsize=8.2)
    ax.set_ylabel("恢复动作 RMSE 相对改善 (%)")
    ax.set_title("恢复技能可以被学会，但持续 SAC 更新会显著遗忘")
    for bar, value in zip(bars, improvements):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (2.0 if value >= 0 else -2.0),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8.5,
            fontweight="bold",
        )
    fig.text(
        0.5,
        -0.03,
        "注意：各柱参考基线写在横轴中，不能把柱高解释为同一条连续训练曲线。",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout()
    save(fig, "图3_恢复技能学习遗忘与修复幅度")


def main() -> None:
    setup_style()
    figure_recovery_commitment()
    figure_validation_interventions()
    figure_learning_and_forgetting()


if __name__ == "__main__":
    main()
