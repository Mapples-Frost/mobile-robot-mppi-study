#!/usr/bin/env python3
"""Audit and summarize the preregistered L247 development qualification."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCENES = ("hairpin", "s_chicane", "infinity")
SCENE_LABELS = {
    "hairpin": "Hairpin",
    "s_chicane": "S-Chicane",
    "infinity": "Infinity",
}
SCENE_NAMES = {
    "hairpin": "tracking_grand_hairpin_l234",
    "s_chicane": "tracking_grand_s_chicane_l234",
    "infinity": "tracking_grand_infinity_l234",
}
BUDGETS = {"hairpin": 2210, "s_chicane": 1405, "infinity": 2030}
ARMS = ("icode_mppi", "full_proposed")
ARM_LABELS = {"icode_mppi": "ICODE-MPPI", "full_proposed": "Full Proposed"}
SEED = 923301001


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _float(value, default=0.0):
    if value in (None, ""):
        return float(default)
    return float(value)


def _int(value, default=0):
    if value in (None, ""):
        return int(default)
    return int(float(value))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("refusing to write an empty L247 table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_l247_rows(repo_root):
    rows = []
    for scene in SCENES:
        path = Path(repo_root) / (
            "results/research_platform/rl/"
            f"tracking_l247_fair_budget_{scene}_seed{SEED}/progress.csv"
        )
        for row in _read_csv(path):
            item = dict(row)
            item["scene_key"] = scene
            item["source_progress"] = str(path)
            rows.append(item)
    return rows


def load_l244_rows(repo_root):
    rows = []
    for scene in SCENES:
        path = Path(repo_root) / (
            "results/research_platform/rl/"
            f"tracking_l244_bc_anchor_{scene}_seed{SEED}/progress.csv"
        )
        for row in _read_csv(path):
            item = dict(row)
            item["scene_key"] = scene
            item["source_progress"] = str(path)
            rows.append(item)
    return rows


def validate_l247_rows(rows):
    problems = []
    keys = []
    for row in rows:
        scene = str(row.get("scene_key", ""))
        arm = str(row.get("benchmark_arm", ""))
        keys.append((scene, arm))
        if scene not in SCENES or arm not in ARMS:
            problems.append("unexpected experiment key: %r" % ((scene, arm),))
            continue
        if str(row.get("scene")) != SCENE_NAMES[scene]:
            problems.append("scene name mismatch: %s" % scene)
        if _int(row.get("seed")) != SEED:
            problems.append("seed mismatch: %s/%s" % (scene, arm))
        if str(row.get("physics_domain")) != "nominal_seen":
            problems.append("physics mismatch: %s/%s" % (scene, arm))
        if _int(row.get("qualification")) != 1:
            problems.append("qualification mismatch: %s/%s" % (scene, arm))
        if _int(row.get("max_steps_budget")) != BUDGETS[scene]:
            problems.append("budget mismatch: %s/%s" % (scene, arm))
        if _int(row.get("rollout_budget_per_decision")) != 100:
            problems.append("rollout budget mismatch: %s/%s" % (scene, arm))
        expected_iterations = 2 if arm == "full_proposed" else 1
        if _int(row.get("paper_iterations")) != expected_iterations:
            problems.append("iteration mismatch: %s/%s" % (scene, arm))
    expected = {(scene, arm) for scene in SCENES for arm in ARMS}
    if len(rows) != 6 or set(keys) != expected or len(set(keys)) != len(keys):
        problems.append("incomplete or duplicate L247 experiment matrix")
    return problems


def evaluate_design_gate(rows):
    problems = validate_l247_rows(rows)
    indexed = {
        (row["scene_key"], row["benchmark_arm"]): row for row in rows
    }
    old_limit_failures = []
    for scene in ("hairpin", "infinity"):
        for arm in ARMS:
            row = indexed.get((scene, arm))
            if row is None:
                continue
            if (
                str(row.get("termination_reason")) == "max_steps"
                and _int(row.get("steps")) == 700
            ):
                old_limit_failures.append("%s/%s" % (scene, arm))
    return {
        "gate_passed": not problems and not old_limit_failures,
        "structural_problems": problems,
        "old_700_step_limit_failures": old_limit_failures,
        "interpretation": (
            "success denominator repaired; this is not a performance pass"
        ),
    }


def episode_table(rows):
    result = []
    for row in sorted(
        rows, key=lambda item: (SCENES.index(item["scene_key"]), ARMS.index(item["benchmark_arm"]))
    ):
        result.append({
            "scene": SCENE_LABELS[row["scene_key"]],
            "method": ARM_LABELS[row["benchmark_arm"]],
            "seed": _int(row["seed"]),
            "max_steps_budget": _int(row["max_steps_budget"]),
            "steps": _int(row["steps"]),
            "success": int(_bool(row["success"])),
            "termination_reason": row["termination_reason"],
            "path_completion_ratio": _float(row["path_completion_ratio"]),
            "collision": int(_bool(row["collision"])),
            "boundary_violation_steps": _int(row["boundary_violation_steps"]),
            "minimum_footprint_boundary_margin": _float(
                row["minimum_footprint_boundary_margin"]
            ),
            "cross_track_rmse": _float(row["cross_track_rmse"]),
            "safety_interventions": _int(row["safety_interventions"]),
            "stuck_steps": _int(row["stuck_steps"]),
            "spin_steps": _int(row["spin_steps"]),
            "proposal_authority_mean": (
                _float(row["reliability_proposal_authority_mean"])
                if row["benchmark_arm"] == "full_proposed" else ""
            ),
            "proposal_fallback_fraction_mean": (
                _float(row["reliability_proposal_fallback_fraction_mean"])
                if row["benchmark_arm"] == "full_proposed" else ""
            ),
            "guided_elites": (
                _int(row["paper_guided_elite_count_total"])
                if row["benchmark_arm"] == "full_proposed" else ""
            ),
            "planner_compute_ms_p99": _float(row["planner_compute_ms_p99"]),
        })
    return result


def paired_table(l247_rows, l244_rows):
    new = {(r["scene_key"], r["benchmark_arm"]): r for r in l247_rows}
    old = {(r["scene_key"], r["benchmark_arm"]): r for r in l244_rows}
    result = []
    for scene in SCENES:
        for arm in ARMS:
            before, after = old[(scene, arm)], new[(scene, arm)]
            result.append({
                "scene": SCENE_LABELS[scene],
                "method": ARM_LABELS[arm],
                "l244_steps": _int(before["steps"]),
                "l247_steps": _int(after["steps"]),
                "l244_termination": before["termination_reason"],
                "l247_termination": after["termination_reason"],
                "l244_completion": _float(before["path_completion_ratio"]),
                "l247_completion": _float(after["path_completion_ratio"]),
                "completion_delta": (
                    _float(after["path_completion_ratio"])
                    - _float(before["path_completion_ratio"])
                ),
                "l244_boundary_steps": _int(before["boundary_violation_steps"]),
                "l247_boundary_steps": _int(after["boundary_violation_steps"]),
            })
    return result


def input_manifest(repo_root, l247_rows, l244_rows):
    files = set()
    for row in l247_rows + l244_rows:
        files.add(Path(row["source_progress"]))
    for scene in SCENES:
        out = Path(repo_root) / (
            "results/research_platform/rl/"
            f"tracking_l247_fair_budget_{scene}_seed{SEED}"
        )
        files.add(out / "provenance.json")
        for arm in ARMS:
            run = out / "runs" / arm / (
                f"{arm}__{SCENE_NAMES[scene]}__nominal_seen__seed{SEED}"
            )
            for name in (
                "config_resolved.yaml", "trajectory.csv", "metrics.json", "provenance.json"
            ):
                files.add(run / name)
    return [
        {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(files, key=lambda item: str(item))
    ]


def plot_budget_effect(l247_rows, l244_rows, output_stem):
    new = {(r["scene_key"], r["benchmark_arm"]): r for r in l247_rows}
    old = {(r["scene_key"], r["benchmark_arm"]): r for r in l244_rows}
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    colors = {"icode_mppi": "#0072B2", "full_proposed": "#D55E00"}
    x = np.arange(len(SCENES))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    width = 0.18
    offsets = {
        ("icode_mppi", "L244"): -1.5 * width,
        ("icode_mppi", "L247"): -0.5 * width,
        ("full_proposed", "L244"): 0.5 * width,
        ("full_proposed", "L247"): 1.5 * width,
    }
    for arm in ARMS:
        for phase, source, alpha, hatch in (
            ("L244", old, 0.38, "//"),
            ("L247", new, 1.0, None),
        ):
            values = [_float(source[(scene, arm)]["path_completion_ratio"]) for scene in SCENES]
            axes[0].bar(
                x + offsets[(arm, phase)], values, width,
                color=colors[arm], alpha=alpha, hatch=hatch,
                label=f"{ARM_LABELS[arm]} ({phase})",
            )
    axes[0].set_xticks(x, [SCENE_LABELS[s] for s in SCENES])
    axes[0].set_ylim(0.0, 0.40)
    axes[0].set_ylabel("Path completion ratio")
    axes[0].set_title("(a) Completion: L244 vs. L247", fontsize=9.5)
    axes[0].legend(
        fontsize=6.8, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, -0.18), frameon=False,
    )

    for index, arm in enumerate(ARMS):
        steps = [_int(new[(scene, arm)]["steps"]) for scene in SCENES]
        offset = (-0.18 if arm == "icode_mppi" else 0.18)
        axes[1].bar(
            x + offset, steps, 0.34, color=colors[arm], label=ARM_LABELS[arm]
        )
    axes[1].scatter(x, [BUDGETS[s] for s in SCENES], marker="_", s=260,
                    linewidths=2.2, color="black", label="Frozen max steps", zorder=5)
    for i, scene in enumerate(SCENES):
        for offset, arm in ((-0.18, "icode_mppi"), (0.18, "full_proposed")):
            row = new[(scene, arm)]
            label = "B" if row["termination_reason"] == "boundary_violation" else "M"
            axes[1].text(i + offset, _int(row["steps"]) + 45, label,
                         ha="center", va="bottom", fontsize=7)
    axes[1].set_xticks(x, [SCENE_LABELS[s] for s in SCENES])
    axes[1].set_ylabel("Executed control steps")
    axes[1].set_ylim(0, max(BUDGETS.values()) * 1.15)
    axes[1].set_title("(b) Executed steps and termination", fontsize=9.5)
    axes[1].legend(
        fontsize=6.8, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.18), ncol=3,
    )
    axes[1].text(
        0.98, 0.93, "M: max steps\nB: boundary violation",
        transform=axes[1].transAxes, ha="right", va="top", fontsize=6.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )
    fig.subplots_adjust(bottom=0.27, wspace=0.30)
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def _mean(rows, arm, field):
    selected = [r for r in rows if r["benchmark_arm"] == arm]
    return sum(_float(r[field]) for r in selected) / len(selected)


def report_text(l247_rows, l244_rows, gate):
    full_completion = _mean(l247_rows, "full_proposed", "path_completion_ratio")
    icode_completion = _mean(l247_rows, "icode_mppi", "path_completion_ratio")
    old_full = _mean(l244_rows, "full_proposed", "path_completion_ratio")
    old_icode = _mean(l244_rows, "icode_mppi", "path_completion_ratio")
    boundary_full = sum(
        _int(r["boundary_violation_steps"])
        for r in l247_rows if r["benchmark_arm"] == "full_proposed"
    )
    successes = sum(_bool(r["success"]) for r in l247_rows)
    collisions = sum(_bool(r["collision"]) for r in l247_rows)
    rows = episode_table(l247_rows)
    table = [
        "| 场景 | 方法 | 步数/预算 | 终止 | 完成度 | 碰撞 | 边界步 |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            "| {scene} | {method} | {steps}/{max_steps_budget} | "
            "{termination_reason} | {path_completion_ratio:.4f} | "
            "{collision} | {boundary_violation_steps} |".format(**row)
        )
    return """# L247 Tracking 公平回合预算 Development 报告

## Material Passport

- Verification Status: ANALYZED
- Frozen Protocol Commit: `2da62df`
- Development Seed: `923301001`
- Physics: `nominal_seen`
- MuJoCo: `3.2.3`
- Independent experimental unit: seed（本轮仅 1 个 development seed）

## 1. 结论

L247 的**实验设计 Gate 通过**：三个场景都使用了预注册的场景特定预算，Hairpin 和
Infinity 不再被旧的 700-step 上限截断。但这是设计有效性通过，不是性能通过。

最终 6 个回合成功数为 **{successes}/6**，碰撞数为 **{collisions}/6**。Full Proposed
在三个场景均因单步 footprint boundary violation 提前终止；ICODE-MPPI 在三个新预算内
均运行至 max steps，但长期停滞，完成度基本没有随预算增加。

这说明 L246 的判断成立：延长回合只消除了不公平的时间上限，不能解决候选轨迹越界和
局部最小值。下一步应按冻结路线进入 L248，将 footprint boundary constraint 前移到所有
MPPI arms 的候选评价阶段，而不是继续增加步数或立即扩大网络。

## 2. 最终结果

{table}

## 3. 与 L244 的只读配对比较

- Full 平均完成度：`{old_full:.4f}` → `{full_completion:.4f}`，变化 `{full_delta:+.4f}`；
- ICODE 平均完成度：`{old_icode:.4f}` → `{icode_completion:.4f}`，变化 `{icode_delta:+.4f}`；
- Full 的 boundary violation 总步数：`{boundary_full}`，三个场景各 1 步；
- Hairpin Full 从 L244 的 700-step max termination 延长到 step 885，随后越界；
- S-Chicane 与 Infinity Full 的终止步和完成度与 L244 基本一致，说明预算不是其主根因；
- ICODE 虽运行到 1405–2210 步，完成度只变化约 0.0003–0.0005，表明长期安全抑制/停滞。

![L247 budget effect](figures/fig_l247_budget_effect.png)

## 4. Gate 判定

```json
{gate_json}
```

Gate 通过仅允许进入 L248 development，不允许注册 sealed Tracking，也不能作为论文正式结论。

## 5. 推断边界

本轮只有一个 development seed，没有进行显著性检验或置信区间估计。L244 与 L247 使用相同
seed、地图和方法，因此可做确定性的工程配对诊断；不能据此宣称总体成功率或跨 seed 优势。
全部失败均已保留，没有筛 seed、删除负向回合或修改核心 RL+ICODE 机制。

## 6. 输出索引

```text
docs/experiments/post_l217/tables/l247_episode_results.csv
docs/experiments/post_l217/tables/l247_vs_l244_paired.csv
docs/experiments/post_l217/tables/l247_gate_audit.json
docs/experiments/post_l217/tables/l247_input_manifest.json
docs/experiments/post_l217/figures/fig_l247_budget_effect.png
docs/experiments/post_l217/figures/fig_l247_budget_effect.pdf
```
""".format(
        successes=successes,
        collisions=collisions,
        table="\n".join(table),
        old_full=old_full,
        full_completion=full_completion,
        full_delta=full_completion - old_full,
        old_icode=old_icode,
        icode_completion=icode_completion,
        icode_delta=icode_completion - old_icode,
        boundary_full=boundary_full,
        gate_json=json.dumps(gate, indent=2, ensure_ascii=False),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    table_dir = root / "docs/experiments/post_l217/tables"
    figure_dir = root / "docs/experiments/post_l217/figures"
    l247_rows = load_l247_rows(root)
    l244_rows = load_l244_rows(root)
    problems = validate_l247_rows(l247_rows)
    if problems:
        raise ValueError("L247 integrity failure: %s" % "; ".join(problems))
    gate = evaluate_design_gate(l247_rows)
    _write_csv(table_dir / "l247_episode_results.csv", episode_table(l247_rows))
    _write_csv(
        table_dir / "l247_vs_l244_paired.csv",
        paired_table(l247_rows, l244_rows),
    )
    (table_dir / "l247_gate_audit.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (table_dir / "l247_input_manifest.json").write_text(
        json.dumps(input_manifest(root, l247_rows, l244_rows), indent=2) + "\n",
        encoding="utf-8",
    )
    plot_budget_effect(
        l247_rows, l244_rows, figure_dir / "fig_l247_budget_effect"
    )
    report = root / "docs/experiments/post_l217/l247_fair_episode_budget_report.md"
    report.write_text(
        report_text(l247_rows, l244_rows, gate), encoding="utf-8"
    )
    print(json.dumps({
        "gate_passed": gate["gate_passed"],
        "episodes": len(l247_rows),
        "report": str(report),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
