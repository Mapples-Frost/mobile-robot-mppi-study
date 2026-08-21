#!/usr/bin/env python3
"""Reproducible analysis for the standalone Redesign E four-arm test."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import yaml


ARMS = ("B00", "B01", "B10", "B11")
EXPECTED_SEEDS = tuple(range(795100001, 795100051))
CORE_PATH = Path(__file__).with_name("redesignE_analysis_core.py")


def load_core():
    spec = importlib.util.spec_from_file_location("redesignE_analysis_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load analysis core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.GROUPS = (10,)
    return module


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_report(
    raw_parent: Path,
    output_root: Path,
    copy_gate: Mapping[str, Any],
    structure_audit: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    binary_rows: Sequence[Mapping[str, Any]],
    continuous_rows: Sequence[Mapping[str, Any]],
    hard_rows: Sequence[Mapping[str, Any]],
    anomaly_rows: Sequence[Mapping[str, Any]],
    env_summary: Mapping[str, Any],
    independent_gate: Mapping[str, Any],
    core: Any,
) -> str:
    summary = {row["arm"]: row for row in summaries}
    binary = {row["baseline"]: row for row in binary_rows}

    arm_contract_rows = []
    for arm in ARMS:
        contracts = env_summary["arm_contracts"].get(arm, [])
        contract = contracts[0] if len(contracts) == 1 else {}
        arm_contract_rows.append(
            [
                arm,
                contract.get("learning", "数据未提供"),
                contract.get("probability", "数据未提供"),
                contract.get("predictor", "数据未提供"),
                contract.get("icode", "数据未提供"),
                contract.get("hss", "数据未提供"),
            ]
        )

    outcome_rows = []
    performance_rows = []
    telemetry_rows = []
    for arm in ARMS:
        row = summary[arm]
        outcome_rows.append(
            [
                arm,
                f"{row['safe_success_n']}/50 ({core.fmt_percent(row['safe_success_rate'])})",
                f"[{core.fmt_percent(row['safe_success_ci95_low_wilson'])}, "
                f"{core.fmt_percent(row['safe_success_ci95_high_wilson'])}]",
                f"{row['boundary_safe_success_n']}/"
                f"{row['boundary_safe_success_available_n']}",
                f"{row['collision_n']}/50 ({core.fmt_percent(row['collision_rate'])})",
                row["timeout_n"],
                row["livelock_n"],
                row["other_termination_n"],
            ]
        )
        performance_rows.append(
            [
                arm,
                f"{core.fmt_number(row['steps_mean'], 1)} / "
                f"{core.fmt_number(row['steps_q25'], 1)} / "
                f"{core.fmt_number(row['steps_median'], 1)} / "
                f"{core.fmt_number(row['steps_q75'], 1)}",
                f"{core.fmt_number(row['final_goal_distance_mean'])} / "
                f"{core.fmt_number(row['final_goal_distance_median'])}",
                f"{core.fmt_number(row['minimum_clearance_mean'])} / "
                f"{core.fmt_number(row['minimum_clearance_median'])}",
                f"{core.fmt_number(row['planner_compute_ms_mean_mean'], 1)} / "
                f"{core.fmt_number(row['planner_compute_ms_p95_mean'], 1)}",
            ]
        )
        telemetry_rows.append(
            [
                arm,
                core.fmt_percent(row["path_progress_ratio_max_mean"]),
                core.fmt_number(row["stuck_steps_mean"], 1),
                core.fmt_number(row["safety_interventions_mean"], 1),
                core.fmt_percent(
                    row["known_static_map_candidate_feasible_fraction_mean_mean"]
                ),
                core.fmt_percent(row["planner_deadline_miss_rate_mean"]),
                core.fmt_number(row["control_jerk_mean"]),
                core.fmt_number(row["cross_track_p95_mean"]),
                core.fmt_number(
                    row["minimum_dynamic_obstacle_center_distance_mean"]
                ),
                core.fmt_number(row["obstacle_pass_events_mean"], 1),
            ]
        )

    binary_table = []
    for baseline in core.BASELINES:
        row = binary[baseline]
        relative = (
            "不可定义"
            if row["relative_improvement"] is None
            else core.fmt_percent(row["relative_improvement"])
        )
        binary_table.append(
            [
                f"B11 vs {baseline}",
                row["both_fail_n00"],
                row["baseline_fail_b11_success_n01"],
                row["baseline_success_b11_fail_n10"],
                row["both_success_n11"],
                f"{row['absolute_percentage_points']:.1f} pp",
                relative,
                f"[{core.fmt_percent(row['paired_risk_difference_ci95_low_bootstrap'])}, "
                f"{core.fmt_percent(row['paired_risk_difference_ci95_high_bootstrap'])}]",
                core.fmt_p_value(row["exact_mcnemar_p"]),
            ]
        )

    continuous_b10 = [
        row for row in continuous_rows if row["comparison"] == "B11_vs_B10"
    ]
    continuous_table = [
        [
            row["metric_cn"],
            row["n_pairs_available"],
            core.fmt_number(row["baseline_mean"]),
            core.fmt_number(row["b11_mean"]),
            core.fmt_number(row["mean_difference_b11_minus_baseline"]),
            core.fmt_number(row["median_difference_b11_minus_baseline"]),
            row["test"],
            f"[{core.fmt_number(row['ci95_low'])}, "
            f"{core.fmt_number(row['ci95_high'])}]",
            f"{row['effect_size_name']}={core.fmt_number(row['effect_size'])}",
            core.fmt_p_value(row["p_value"]),
            core.fmt_p_value(row["p_value_holm_within_scope"]),
        ]
        for row in continuous_b10
    ]

    all_four_failed = [
        str(row["seed"]) for row in hard_rows if bool(row["all_four_failed"])
    ]
    failure_distribution: dict[int, int] = {}
    for row in hard_rows:
        count = int(row["failure_arm_count"])
        failure_distribution[count] = failure_distribution.get(count, 0) + 1

    lines = [
        "# Redesign E 四臂独立测试分析",
        "",
        "> 本报告只使用数据包内的 50 个 matched cases（4 arms × 50），"
        "测试 ID：`redesignE_independent_test_v1`；显示编号 "
        "795100001–795100050。",
        "",
        "## 1. 独立数据包与完整性",
        "",
        f"- 独立原始数据：`{raw_parent / 'redesignE_four_arm_exact'}`",
        f"- episode：200（4 arms × 50 matched cases）",
        f"- 复制文件：{copy_gate['actual_files']:,}/{copy_gate['expected_files']:,}",
        f"- 复制字节：{copy_gate['verified_bytes']:,}",
        f"- SHA-256：{copy_gate['verified_files']:,}/"
        f"{copy_gate['expected_files']:,} 通过；缺失、错误、额外文件均为 0。",
        f"- 结构 Gate：{structure_audit['episodes_loaded']}/"
        f"{structure_audit['expected_episodes']} episode；"
        f"场景不一致 {structure_audit['physical_scenario_mismatch_count']}。",
        f"- 独立分析 Gate："
        f"{'通过' if independent_gate.get('passed') else '失败'}；"
        f"{independent_gate.get('arm_summary_rows', 0)} 个 arm 汇总、"
        f"{independent_gate.get('binary_comparison_rows', 0)} 个二元配对、"
        f"{independent_gate.get('continuous_comparison_rows', 0)} 个连续配对。",
        "- 本分析为自包含运行，不依赖其他实验的分析目录或统计结果。",
        "",
        "## 2. 四臂冻结契约与环境",
        "",
        core.markdown_table(
            ["Arm", "Learning", "Probability", "Predictor", "ICODE", "HSS"],
            arm_contract_rows,
        ),
        "",
        f"- METHOD_REVISION disabled："
        f"{env_summary['revision_disabled_episode_count']}/200。",
        f"- MuJoCo：{core.fmt_recorded_values(env_summary['mujoco_version_values'])}。",
        f"- Torch：{core.fmt_recorded_values(env_summary['torch_version_values_recorded'])}。",
        f"- Git SHA：{core.fmt_recorded_values(env_summary['git_sha_values'])}。",
        f"- planner device："
        f"{core.fmt_recorded_values(env_summary['planner_device_values'])}。",
        f"- 展开物理场景 hash 唯一值："
        f"{len(env_summary['physical_scenario_hash_values'])}；"
        "同 seed 四臂场景、任务、初始状态及 max_steps 完全一致。",
        "- B11 与 baseline 的记录性路径字符串不同；"
        "展开物理场景一致，因此未观察到地图几何混杂，但 Git SHA 为 unknown，"
        "路径 provenance 仍属于审计限制。",
        "",
        "## 3. 成功、碰撞与终止模式",
        "",
        core.markdown_table(
            [
                "Arm",
                "Safe success",
                "Wilson 95% CI",
                "Boundary-safe",
                "Collision",
                "Timeout",
                "Livelock",
                "其他",
            ],
            outcome_rows,
        ),
        "",
        "`safe_success` 定义为 `metrics.success AND NOT metrics.collision`。"
        "`boundary_safe_success` 是更严格的次级 endpoint，不能与正式成功数混用。",
        "",
        "## 4. 主要连续指标",
        "",
        core.markdown_table(
            [
                "Arm",
                "Steps 均值/Q25/中位/Q75",
                "最终距离均值/中位 (m)",
                "最小净空均值/中位 (m)",
                "Planner mean/P95 (ms)",
            ],
            performance_rows,
        ),
        "",
        "## 5. 其他实际遥测",
        "",
        core.markdown_table(
            [
                "Arm",
                "路线进度",
                "Stuck",
                "Safety intervention",
                "候选可行率",
                "Deadline miss",
                "Control jerk",
                "Cross-track P95",
                "动态障碍中心距",
                "Pass events",
            ],
            telemetry_rows,
        ),
        "",
        "## 6. B11 与各基线的配对成功比较",
        "",
        "二元成功采用双侧 exact McNemar；配对风险差区间为固定随机种子、"
        "20,000 次 paired bootstrap 95% CI。",
        "",
        core.markdown_table(
            [
                "比较",
                "两者失败",
                "改善",
                "退化",
                "两者成功",
                "绝对差",
                "相对改善",
                "RD 95% CI",
                "McNemar p",
            ],
            binary_table,
        ),
        "",
        "## 7. B11 与 B10 的配对连续指标",
        "",
        "差值均为 B11−B10。逐对差值 Shapiro–Wilk 未拒绝正态时用配对 t，"
        "否则用 Wilcoxon signed-rank；同时报告效应量、95% CI、原始 p 和"
        "组内 Holm 校正 p。",
        "",
        core.markdown_table(
            [
                "指标",
                "N",
                "B10 mean",
                "B11 mean",
                "均值差",
                "中位差",
                "检验",
                "95% CI",
                "效应量",
                "p",
                "Holm p",
            ],
            continuous_table,
        ),
        "",
        "完成步数不能脱离终止方式解释：B10 的多数 episode 较早碰撞，"
        "因此更少的步数不代表更高效率。",
        "",
        "## 8. 一致困难 Seed",
        "",
        f"- 失败 arm 数分布：{dict(sorted(failure_distribution.items()))}。",
        f"- 四臂全部失败：{len(all_four_failed)} 个 seed："
        f"`{', '.join(all_four_failed)}`。",
        f"- 自动异常 episode：{len(anomaly_rows)}。",
        "",
        "## 9. 主要发现",
        "",
        f"1. safe success：B00 {summary['B00']['safe_success_n']}/50，"
        f"B01 {summary['B01']['safe_success_n']}/50，"
        f"B10 {summary['B10']['safe_success_n']}/50，"
        f"B11 {summary['B11']['safe_success_n']}/50。",
        f"2. B11 相对 B10 有 "
        f"{binary['B10']['baseline_fail_b11_success_n01']} 对改善、"
        f"{binary['B10']['baseline_success_b11_fail_n10']} 对退化，"
        f"配对风险差 {binary['B10']['absolute_percentage_points']:.1f} 个百分点。",
        f"3. 碰撞数：B00 {summary['B00']['collision_n']}/50，"
        f"B01 {summary['B01']['collision_n']}/50，"
        f"B10 {summary['B10']['collision_n']}/50，"
        f"B11 {summary['B11']['collision_n']}/50。",
        f"4. B11 最终目标距离中位数为 "
        f"{core.fmt_number(summary['B11']['final_goal_distance_median'])} m；"
        f"最小净空中位数为 "
        f"{core.fmt_number(summary['B11']['minimum_clearance_median'])} m。",
        "5. 所有比较均限于本数据包内的 matched cases；不合并或调用其他实验结果。",
        "",
        "## 10. 局限与复现位置",
        "",
        "- 本报告覆盖一个自包含的 50-case matched test；"
        "结论仅适用于该测试集和冻结配置。",
        "- 统计检验仅描述本测试集内部的配对差异。",
        "- Git SHA 未记录、Torch 版本未提供，限制独立环境重建。",
        f"- 分析入口：`{Path(__file__).resolve()}`",
        f"- 统计核心：`{CORE_PATH.resolve()}`",
        f"- 完整性审计：`{output_root.parent / 'audit' / 'COPY_VERIFICATION.json'}`",
        f"- episode 索引：`{output_root / 'REDESIGNE_EPISODE_INDEX.csv'}`",
        f"- arm 汇总：`{output_root / 'REDESIGNE_ARM_SUMMARY.csv'}`",
        f"- 二元配对：`{output_root / 'REDESIGNE_PAIRED_BINARY.csv'}`",
        f"- 连续配对：`{output_root / 'REDESIGNE_PAIRED_CONTINUOUS.csv'}`",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    core = load_core()
    raw_parent = Path(args.raw_parent).resolve()
    output_root = Path(args.output).resolve()
    copy_gate_path = Path(args.copy_verification).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    copy_gate = read_json(copy_gate_path)
    if not bool(copy_gate.get("passed")):
        raise RuntimeError(f"Copy verification did not pass: {copy_gate_path}")

    episodes, structure_audit = core.load_and_audit_episodes(
        raw_parent, output_root
    )
    if not bool(structure_audit.get("passed")):
        raise RuntimeError(
            f"Standalone structure Gate failed: "
            f"{output_root / 'dataset_structure_audit.json'}"
        )
    seeds_by_arm = {
        arm: tuple(sorted(e.seed for e in episodes if e.arm == arm))
        for arm in ARMS
    }
    if any(seeds != EXPECTED_SEEDS for seeds in seeds_by_arm.values()):
        raise RuntimeError(f"Unexpected seed membership: {seeds_by_arm}")

    summaries_all = core.group_arm_summaries(episodes)
    summaries = [
        row
        for row in summaries_all
        if row["scope"] == "group" and str(row["group"]) == "10"
    ]
    binary_all = core.binary_paired_comparisons(episodes)
    binary_rows = [
        row
        for row in binary_all
        if row["scope"] == "group" and str(row["group"]) == "10"
    ]
    continuous_all = core.continuous_paired_comparisons(episodes)
    continuous_rows = [
        row
        for row in continuous_all
        if row["scope"] == "group" and str(row["group"]) == "10"
    ]
    hard_rows = core.hard_seed_rows(episodes)
    anomaly_rows = core.anomalous_episode_rows(episodes)
    env_summary = core.environment_summary(episodes)

    for rows in (
        summaries,
        binary_rows,
        continuous_rows,
        hard_rows,
        anomaly_rows,
    ):
        for row in rows:
            row.pop("group", None)
            if row.get("scope") == "group":
                row["scope"] = "test_set"
            row["test_set"] = "redesignE_four_arm_exact"

    core.write_csv(
        output_root / "REDESIGNE_EPISODE_INDEX.csv",
        core.episode_index_rows(episodes),
    )
    core.write_csv(output_root / "REDESIGNE_ARM_SUMMARY.csv", summaries)
    core.write_csv(output_root / "REDESIGNE_PAIRED_BINARY.csv", binary_rows)
    core.write_csv(
        output_root / "REDESIGNE_PAIRED_CONTINUOUS.csv", continuous_rows
    )
    core.write_csv(
        output_root / "REDESIGNE_HARD_SEED_CONSISTENCY.csv", hard_rows
    )
    core.write_csv(
        output_root / "REDESIGNE_ANOMALOUS_EPISODES.csv", anomaly_rows
    )
    core.write_json(
        output_root / "REDESIGNE_ENVIRONMENT_SUMMARY.json", env_summary
    )

    independent_gate = {
        "experiment_id": "redesignE_independent_test_v1",
        "generated_utc": utc_now(),
        "copy_verification_passed": bool(copy_gate.get("passed")),
        "structure_gate_passed": bool(structure_audit.get("passed")),
        "expected_episode_count": 200,
        "episodes_loaded": len(episodes),
        "expected_seeds": list(EXPECTED_SEEDS),
        "arm_seed_counts": {
            arm: len(seeds_by_arm[arm]) for arm in ARMS
        },
        "physical_scenario_matched_sets": structure_audit.get(
            "physical_scenario_matched_sets_checked"
        ),
        "physical_scenario_mismatch_count": structure_audit.get(
            "physical_scenario_mismatch_count"
        ),
        "arm_summary_rows": len(summaries),
        "binary_comparison_rows": len(binary_rows),
        "continuous_comparison_rows": len(continuous_rows),
        "external_analysis_dependency": False,
        "passed": (
            bool(copy_gate.get("passed"))
            and bool(structure_audit.get("passed"))
            and len(episodes) == 200
            and all(seeds_by_arm[arm] == EXPECTED_SEEDS for arm in ARMS)
            and len(summaries) == 4
            and len(binary_rows) == 3
            and len(continuous_rows) == 15
        ),
    }
    core.write_json(
        output_root / "INDEPENDENT_ANALYSIS_GATE.json", independent_gate
    )
    if not independent_gate["passed"]:
        raise RuntimeError(
            f"Independent analysis Gate failed: "
            f"{output_root / 'INDEPENDENT_ANALYSIS_GATE.json'}"
        )

    report = build_report(
        raw_parent,
        output_root,
        copy_gate,
        structure_audit,
        summaries,
        binary_rows,
        continuous_rows,
        hard_rows,
        anomaly_rows,
        env_summary,
        independent_gate,
        core,
    )
    report_path = output_root / "REDESIGNE_ANALYSIS_CN.md"
    report_path.write_text(report, encoding="utf-8")
    core.write_json(
        output_root / "REDESIGNE_ANALYSIS_METADATA.json",
        {
            "experiment_id": "redesignE_independent_test_v1",
            "analysis_completed_utc": utc_now(),
            "raw_parent": str(raw_parent),
            "output_root": str(output_root),
            "copy_verification": str(copy_gate_path),
            "external_analysis_dependency": False,
            "independent_analysis_gate": str(
                output_root / "INDEPENDENT_ANALYSIS_GATE.json"
            ),
            "safe_success_definition": "metrics.success AND NOT metrics.collision",
            "secondary_endpoint": "metrics.boundary_safe_success",
            "binary_test": "two-sided exact McNemar",
            "bootstrap_samples": core.BOOTSTRAP_SAMPLES,
            "bootstrap_seed": core.BOOTSTRAP_SEED,
            "continuous_test_selection": (
                "paired t if Shapiro-Wilk p>=0.05, otherwise Wilcoxon"
            ),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyyaml": yaml.__version__,
        },
    )
    print(f"[OK] Standalone Redesign E test report: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--copy-verification", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
