#!/usr/bin/env python3
"""Rolling metadata digest for vehicle safe-shortening v1 devval64.

Reads only completed fresh development-validation shard artifacts for the
IMPROVED vehicle safe-shortening v1 campaign. It performs no rollout, no
training and no sealed-test access. Output directory is keyed by completed
shards so repeated use after later shards creates new immutable digest artifacts.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1"
DIAG_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
H_GRID = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
SEEDS = (0, 1, 2)
SERVICE_START = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fsum(values: Iterable[float]) -> float:
    return float(math.fsum(float(v) for v in values))


def mean(values: Sequence[float]) -> Any:
    return float(statistics.mean(values)) if values else None


def median(values: Sequence[float]) -> Any:
    return float(statistics.median(values)) if values else None


def pct(values: Sequence[float], q: float) -> Any:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def add_counts(target: Dict[str, int], source: Mapping[str, Any]) -> None:
    for key, value in (source or {}).items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def decision_sum(episode: Mapping[str, Any]) -> float:
    return float((episode.get("decision_timing_s") or {}).get("sum", 0.0))


def verify_completed_marker(done_path: Path) -> Dict[str, Any]:
    done = read_json(done_path)
    if done.get("passed") is not True:
        raise RuntimeError("Completed marker did not pass: %s" % rel(done_path))
    mismatches: List[Dict[str, str]] = []
    for name, expected in sorted((done.get("hashes") or {}).items()):
        path = ROOT / name
        if not path.exists():
            mismatches.append({"path": name, "expected": str(expected), "actual": "MISSING"})
            continue
        actual = sha256(path)
        if actual != expected:
            mismatches.append({"path": name, "expected": str(expected), "actual": actual})
    if mismatches:
        raise RuntimeError("Shard hash audit failed: %r" % mismatches[:5])
    return done


def aggregate_episodes(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    hcounts: Dict[str, int] = {}
    raw_hcounts: Dict[str, int] = {}
    decision_values = [decision_sum(e) for e in items]
    steps = int(sum(int(e.get("steps", 0)) for e in items))
    for e in items:
        add_counts(hcounts, e.get("horizon_counts") or {})
        add_counts(raw_hcounts, e.get("raw_horizon_counts_before_clamp") or {})
    decision_total = fsum(decision_values)
    return {
        "episodes": len(items),
        "cases": sorted({int(e.get("case")) for e in items}),
        "steps": steps,
        "success_count": int(sum(1 for e in items if e.get("success"))),
        "episode_failure_count": int(sum(1 for e in items if e.get("episode_failure"))),
        "constraint_count": int(sum(1 for e in items if e.get("constraint"))),
        "initial_failed_steps": int(sum(int(e.get("initial_failed_steps", 0)) for e in items)),
        "solver_failure_steps": int(sum(int(e.get("solver_failure_steps", 0)) for e in items)),
        "retries": int(sum(int(e.get("retries", 0)) for e in items)),
        "switches": int(sum(int(e.get("switches", 0)) for e in items)),
        "clamped_steps": int(sum(int(e.get("clamped_steps", 0)) for e in items)),
        "solver_failure_fallback_steps": int(sum(int(e.get("solver_failure_fallback_steps", 0)) for e in items)),
        "physical_constraint_cost_sum": fsum(float(e.get("physical_constraint_cost", 0.0)) for e in items),
        "total_cost_sum": fsum(float(e.get("total_cost", 0.0)) for e in items),
        "h_penalty_sum": fsum(float(e.get("h_penalty", 0.0)) for e in items),
        "decision_total_s": decision_total,
        "decision_mean_s_per_step": (decision_total / steps if steps else None),
        "decision_episode_mean_s": mean(decision_values),
        "decision_episode_median_s": median(decision_values),
        "decision_episode_p95_s": pct(decision_values, 0.95),
        "horizon_counts": hcounts,
        "raw_horizon_counts_before_clamp": raw_hcounts,
        "unique_horizons": sorted(int(h) for h in hcounts),
        "used_h_above_25": any(int(h) > 25 for h in hcounts),
        "used_h_below_25": any(int(h) < 25 for h in hcounts),
    }


def paired_vs_fixed(by_arm_case: Mapping[Tuple[str, int], Mapping[str, Any]], adaptive_id: str, fixed_id: str) -> Dict[str, Any]:
    cases = sorted(c for (arm, c) in by_arm_case if arm == adaptive_id and (fixed_id, c) in by_arm_case)
    rows: List[Dict[str, Any]] = []
    ratios: List[float] = []
    adaptive_decision = 0.0
    fixed_decision = 0.0
    adaptive_h: Dict[str, int] = {}
    fixed_h: Dict[str, int] = {}
    for case in cases:
        a = by_arm_case[(adaptive_id, case)]
        f = by_arm_case[(fixed_id, case)]
        a_dec = decision_sum(a)
        f_dec = decision_sum(f)
        adaptive_decision += a_dec
        fixed_decision += f_dec
        if f_dec > 0.0:
            ratios.append(a_dec / f_dec)
        add_counts(adaptive_h, a.get("horizon_counts") or {})
        add_counts(fixed_h, f.get("horizon_counts") or {})
        rows.append({
            "case": case,
            "adaptive_success": bool(a.get("success")),
            "fixed_success": bool(f.get("success")),
            "adaptive_steps": int(a.get("steps", 0)),
            "fixed_steps": int(f.get("steps", 0)),
            "delta_physical_constraint_cost": float(a.get("physical_constraint_cost", 0.0)) - float(f.get("physical_constraint_cost", 0.0)),
            "delta_total_cost": float(a.get("total_cost", 0.0)) - float(f.get("total_cost", 0.0)),
            "decision_ratio": (a_dec / f_dec if f_dec > 0.0 else None),
            "adaptive_horizons": a.get("horizon_counts"),
            "fixed_horizons": f.get("horizon_counts"),
        })
    return {
        "paired_cases": cases,
        "paired_case_count": len(cases),
        "adaptive_success_count": int(sum(1 for r in rows if r["adaptive_success"])),
        "fixed_success_count": int(sum(1 for r in rows if r["fixed_success"])),
        "adaptive_failure_count": int(sum(1 for r in rows if not r["adaptive_success"])),
        "fixed_failure_count": int(sum(1 for r in rows if not r["fixed_success"])),
        "physical_delta_sum": fsum(r["delta_physical_constraint_cost"] for r in rows),
        "physical_delta_mean_case": mean([r["delta_physical_constraint_cost"] for r in rows]),
        "total_delta_sum": fsum(r["delta_total_cost"] for r in rows),
        "total_delta_mean_case": mean([r["delta_total_cost"] for r in rows]),
        "decision_ratio_mean_unweighted_cases": mean(ratios),
        "decision_ratio_median_unweighted_cases": median(ratios),
        "decision_ratio_p95_unweighted_cases": pct(ratios, 0.95),
        "decision_ratio_overall_total_decision_s": (adaptive_decision / fixed_decision if fixed_decision > 0.0 else None),
        "adaptive_decision_total_s": adaptive_decision,
        "fixed_decision_total_s": fixed_decision,
        "adaptive_horizon_counts": adaptive_h,
        "fixed_horizon_counts": fixed_h,
        "used_adaptive_h_above_25": any(int(h) > 25 for h in adaptive_h),
        "used_adaptive_h_below_25": any(int(h) < 25 for h in adaptive_h),
        "case_deltas": rows,
    }


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return "%.6g" % x
    return str(x)


def write_summary(raw: Mapping[str, Any], path: Path) -> None:
    p = raw["campaign_progress"]
    lines: List[str] = [
        "# Vehicle safe-shortening v1 devval64 progress digest v2",
        "",
        "Created UTC: `%s`." % raw["created_utc"],
        "",
        "Metadata-only digest over already completed fresh development-validation shards. It performed no new rollout, no training, no historical validation64 bank reopen, and no sealed-test access.",
        "",
        "## Campaign progress",
        "",
        "- Completed shards: `%s/%s` -> `%s`." % (p["completed_shard_count"], p["planned_shards"], p["completed_shards"]),
        "- Fresh cases scored: `%s/%s` -> `%s`." % (p["fresh_cases_scored"], p["planned_cases"], p["cases"]),
        "- Input episodes/control steps: `%s` / `%s`." % (p["episodes"], p["control_steps"]),
        "- Access flags: sealed_test_accessed=`%s`, historical_validation64_bank_opened=`%s`." % (raw["access_flags"]["sealed_test_accessed"], raw["access_flags"]["historical_validation64_bank_opened"]),
        "",
        "## Primary adaptive vs same-seed matched-terminal fixed H25",
        "",
    ]
    for seed in sorted(raw["primary_H25_by_seed"], key=int):
        item = raw["primary_H25_by_seed"][seed]
        lines.append(
            "- seed %s: cases=%d, success adaptive/fixed=%d/%d, phys_delta_sum=%s, total_delta_sum=%s, decision_ratio_overall=%s, decision_ratio_mean_cases=%s, adaptive_horizons=%s, H>25=%s"
            % (
                seed,
                item["paired_case_count"],
                item["adaptive_success_count"],
                item["fixed_success_count"],
                fmt(item["physical_delta_sum"]),
                fmt(item["total_delta_sum"]),
                fmt(item["decision_ratio_overall_total_decision_s"]),
                fmt(item["decision_ratio_mean_unweighted_cases"]),
                item["adaptive_horizon_counts"],
                item["used_adaptive_h_above_25"],
            )
        )
    lines += ["", "## Preliminary same-seed matched-terminal fixed-H grid deltas", ""]
    lines.append("Adaptive minus fixed, paired over completed cases only; this is not final model selection.")
    for seed in sorted(raw["adaptive_vs_matched_grid_by_seed"], key=int):
        lines.append("")
        lines.append("### seed %s" % seed)
        for h in map(str, H_GRID):
            item = raw["adaptive_vs_matched_grid_by_seed"][seed][h]
            lines.append(
                "- H%s: fixed_success=%d/%d, phys_delta_sum=%s, total_delta_sum=%s, decision_ratio_overall=%s"
                % (
                    h,
                    item["fixed_success_count"],
                    item["paired_case_count"],
                    fmt(item["physical_delta_sum"]),
                    fmt(item["total_delta_sum"]),
                    fmt(item["decision_ratio_overall_total_decision_s"]),
                )
            )
    lines += ["", "## Adaptive arm aggregates", ""]
    for arm_id in sorted(raw["adaptive_arm_aggregates"]):
        a = raw["adaptive_arm_aggregates"][arm_id]
        lines.append(
            "- `%s`: episodes=%d, success=%d, failures=%d, phys_sum=%s, total_sum=%s, decision_mean_s_per_step=%s, horizons=%s, H<25=%s, H>25=%s"
            % (
                arm_id,
                a["episodes"],
                a["success_count"],
                a["episode_failure_count"],
                fmt(a["physical_constraint_cost_sum"]),
                fmt(a["total_cost_sum"]),
                fmt(a["decision_mean_s_per_step"]),
                a["horizon_counts"],
                a["used_h_below_25"],
                a["used_h_above_25"],
            )
        )
    lines += ["", "## Interpretation limits", ""]
    for item in raw["interpretation_limits"]:
        lines.append("- " + item)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    token = "<!-- vehicle-safe-shortening-v1-devval-progress-digest-v2-%s -->" % "-".join("%02d" % s for s in raw["campaign_progress"]["completed_shards"])
    primary_brief = {
        seed: {
            k: raw["primary_H25_by_seed"][seed][k]
            for k in (
                "paired_case_count",
                "adaptive_success_count",
                "fixed_success_count",
                "physical_delta_sum",
                "total_delta_sum",
                "decision_ratio_overall_total_decision_s",
                "adaptive_horizon_counts",
                "used_adaptive_h_above_25",
            )
        }
        for seed in sorted(raw["primary_H25_by_seed"], key=int)
    }
    text = (
        token + "\n"
        "## 2026-09-27 vehicle safe-shortening v1 devval64 progress digest v2\n\n"
        "UTC: %s. Metadata-only digest over completed fresh devval shards %s: %d episodes, %d control steps, %d/%d cases scored. No rollout/training, no historical validation64 bank reopen, and no sealed-test access. Primary same-seed H25 preliminary deltas: %s. Continue frozen shards before model selection/final-test gate. Artifacts: `%s`, `%s`, `%s`.\n"
        % (
            raw["created_utc"],
            raw["campaign_progress"]["completed_shards"],
            raw["campaign_progress"]["episodes"],
            raw["campaign_progress"]["control_steps"],
            raw["campaign_progress"]["fresh_cases_scored"],
            raw["campaign_progress"]["planned_cases"],
            primary_brief,
            raw["artifacts"]["summary_md"],
            raw["artifacts"]["raw_json"],
            raw["artifacts"]["completed_json"],
        )
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if token not in old:
                path.write_text(old.rstrip() + "\n\n" + text, encoding="utf-8")


def main() -> int:
    if not CAMPAIGN.exists():
        raise RuntimeError("Campaign directory missing: %s" % rel(CAMPAIGN))

    shard_dirs = sorted(p for p in CAMPAIGN.glob("shard[0-9][0-9]") if (p / "completed.json").exists())
    if not shard_dirs:
        raise RuntimeError("No completed shards found")

    raws: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    hash_inputs: Dict[str, Any] = {}
    access = {"sealed_test_accessed": False, "historical_validation64_bank_opened": False}
    for shard_dir in shard_dirs:
        done_path = shard_dir / "completed.json"
        raw_path = shard_dir / "raw.json"
        done = verify_completed_marker(done_path)
        raw = read_json(raw_path)
        shard_index = int(raw["shard_index"])
        flags = raw.get("access_flags") or {}
        access["sealed_test_accessed"] = bool(access["sealed_test_accessed"] or flags.get("sealed_test_bank_opened") or raw.get("sealed_test_bank_opened"))
        access["historical_validation64_bank_opened"] = bool(access["historical_validation64_bank_opened"] or flags.get("historical_validation64_bank_opened") or raw.get("historical_validation64_bank_opened"))
        hash_inputs["shard%02d" % shard_index] = {
            "completed": rel(done_path),
            "completed_sha256": sha256(done_path),
            "raw": rel(raw_path),
            "raw_sha256": sha256(raw_path),
            "episodes": int(done.get("episodes", 0)),
            "control_steps": int(done.get("control_steps", 0)),
            "cases": done.get("cases"),
        }
        raws.append(raw)
        episodes.extend(raw.get("episodes") or [])

    if access["sealed_test_accessed"] or access["historical_validation64_bank_opened"]:
        raise RuntimeError("Unexpected opened split flag: %r" % access)

    by_arm: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_arm_case: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for e in episodes:
        arm_id = str(e["arm_id"])
        case = int(e["case"])
        by_arm[arm_id].append(e)
        if (arm_id, case) in by_arm_case:
            raise RuntimeError("Duplicate arm/case episode: %s case %s" % (arm_id, case))
        by_arm_case[(arm_id, case)] = e

    primary: Dict[str, Any] = {}
    matched_grid: Dict[str, Any] = {}
    independent_seed0_grid: Dict[str, Any] = {}
    for seed in SEEDS:
        adaptive_id = "safe_shortening_v1_vehicle_s%d" % seed
        primary[str(seed)] = paired_vs_fixed(by_arm_case, adaptive_id, "matched_terminal_fixed_H25_vehicle_s%d" % seed)
        matched_grid[str(seed)] = {}
        independent_seed0_grid[str(seed)] = {}
        for h in H_GRID:
            matched_grid[str(seed)][str(h)] = paired_vs_fixed(by_arm_case, adaptive_id, "matched_terminal_fixed_H%d_vehicle_s%d" % (h, seed))
            independent_seed0_grid[str(seed)][str(h)] = paired_vs_fixed(by_arm_case, adaptive_id, "independent_terminal_seed0_fixed_H%d" % h)

    cases = sorted({int(e["case"]) for e in episodes})
    completed_shards = sorted(int(raw["shard_index"]) for raw in raws)
    shard_key = "shards" + "-".join("%02d" % s for s in completed_shards)
    diag = DIAG_ROOT / ("vehicle_safe_shortening_v1_devval_progress_digest_20260927_v2_" + shard_key)
    if diag.exists() and not (diag / "completed.json").exists():
        raise RuntimeError("Partial digest directory exists: %s" % rel(diag))
    diag.mkdir(parents=True, exist_ok=True)

    now = dt.datetime.now(dt.timezone.utc)
    elapsed_s = (now - SERVICE_START).total_seconds()
    raw_out: Dict[str, Any] = {
        "created_utc": now.isoformat(),
        "method": "IMPROVED_vehicle_safe_shortening_v1_devval_progress_digest_v2_metadata_only_not_original_SAC",
        "source": {"path": rel(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "campaign_root": rel(CAMPAIGN),
        "access_flags": access,
        "budget_actual": {
            "new_rollout_episodes": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "sealed_test_episodes": 0,
            "metadata_input_shards": len(shard_dirs),
            "metadata_input_episodes": len(episodes),
            "metadata_input_control_steps": int(sum(int(e.get("steps", 0)) for e in episodes)),
        },
        "campaign_progress": {
            "completed_shards": completed_shards,
            "completed_shard_count": len(completed_shards),
            "planned_shards": 16,
            "cases": cases,
            "fresh_cases_scored": len(cases),
            "planned_cases": 64,
            "episodes": len(episodes),
            "control_steps": int(sum(int(e.get("steps", 0)) for e in episodes)),
        },
        "hash_audit": {"completed_shard_hashes_verified": True, "inputs": hash_inputs},
        "primary_H25_by_seed": primary,
        "adaptive_vs_matched_grid_by_seed": matched_grid,
        "adaptive_vs_independent_seed0_grid_by_seed": independent_seed0_grid,
        "adaptive_arm_aggregates": {arm: aggregate_episodes(items) for arm, items in sorted(by_arm.items()) if arm.startswith("safe_shortening_v1_vehicle_s")},
        "fixed_arm_aggregates": {arm: aggregate_episodes(items) for arm, items in sorted(by_arm.items()) if arm.startswith("matched_terminal_fixed_") or arm.startswith("independent_terminal_seed0_fixed_")},
        "service_elapsed_at_digest": {
            "since_2026-09-26T10:55:29.419331Z_seconds": elapsed_s,
            "since_2026-09-26T10:55:29.419331Z_hours": elapsed_s / 3600.0,
            "research_sqlite_total_tokens": "unknown; research.sqlite path unavailable to repository tools",
        },
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
        "interpretation_limits": [
            "rolling digest over completed development-validation shards only; not final-test evidence",
            "IMPROVED safe-shortening wrapper using reused historical gated policies; not ORIGINAL Bøhn SAC",
            "no model selection or final-test request from %d/%d shards" % (len(completed_shards), 16),
            "timing evidence is measured wall-clock timing but incomplete until all paired AWS case blocks are aggregated",
            "H distribution is recorded as policy behavior only and is not by itself an acceleration claim",
            "previously inspected/diagnostic validation cases remain contaminated for development; this digest uses the fresh devval64 campaign only",
        ],
    }

    raw_path = diag / "raw.json"
    summary_path = diag / "summary.md"
    completed_path = diag / "completed.json"
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    backup_request = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL_PROGRESS_DIGEST_V2_%s_%s.json" % (shard_key, stamp))
    raw_out["artifacts"] = {"raw_json": rel(raw_path), "summary_md": rel(summary_path), "completed_json": rel(completed_path), "backup_request": rel(backup_request)}

    write_json(raw_path, raw_out)
    write_summary(raw_out, summary_path)
    write_json(backup_request, {
        "requested_utc": now.isoformat(),
        "reason": "backup shard02 plus metadata-only v2 progress digest before continuing additional fresh devval shards",
        "artifacts": [rel(raw_path), rel(summary_path), rel(completed_path), rel(Path(__file__).resolve()), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "EXPERIMENT_REGISTRY.csv"],
        "input_completed_shards": completed_shards,
        "input_episodes": len(episodes),
        "input_control_steps": raw_out["campaign_progress"]["control_steps"],
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
    })
    append_docs(raw_out)

    files = [raw_path, summary_path, backup_request, Path(__file__).resolve()]
    for info in hash_inputs.values():
        files.append(ROOT / info["completed"])
        files.append(ROOT / info["raw"])
    write_json(completed_path, {
        "passed": True,
        "method": raw_out["method"],
        "development_validation_digest": True,
        "formal_final_test_evidence": False,
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "input_completed_shards": completed_shards,
        "input_episodes": len(episodes),
        "input_control_steps": raw_out["campaign_progress"]["control_steps"],
        "backup_request": rel(backup_request),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files))},
    })

    print(json.dumps({
        "completed": rel(completed_path),
        "summary": rel(summary_path),
        "backup_request": rel(backup_request),
        "completed_shards": completed_shards,
        "cases_scored": len(cases),
        "episodes": len(episodes),
        "control_steps": raw_out["campaign_progress"]["control_steps"],
        "primary_H25_by_seed": {
            seed: {
                k: primary[seed][k]
                for k in ("paired_case_count", "adaptive_success_count", "fixed_success_count", "physical_delta_sum", "total_delta_sum", "decision_ratio_overall_total_decision_s", "adaptive_horizon_counts", "used_adaptive_h_above_25")
            }
            for seed in sorted(primary, key=int)
        },
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
