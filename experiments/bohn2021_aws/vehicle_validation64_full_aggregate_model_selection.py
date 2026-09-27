#!/usr/bin/env python3
"""Metadata-only full vehicle validation64 aggregate/model-selection diagnostic.

Reads only already-created vehicle validation64 shard outputs (completed/raw/
summary metadata and per-episode summary.json files).  It does not open the
validation bank, does not read traces, performs no rollout/control/training, and
never opens or hashes the sealed final test bank.

Purpose: aggregate the completed 2688 validation episodes into arm/seed/case
summaries, paired learned-vs-fixed deltas, validation fixed-H nominations, timing
summaries, budget accounting, and case43 sensitivity.  This is validation/model-
selection evidence for the frozen IMPROVED latency-tree vehicle campaign only,
not final-test evidence and not ORIGINAL SAC reproduction.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ROOT = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_validation64_full_aggregate_model_selection.py"
GATE_JSON = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"
SHARDS = list(range(12))
H_GRID = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
LEARNED_KEYS = ["learned_s0", "learned_s1", "learned_s2"]
DOC_MARKER = "vehicle-validation64-full-aggregate-model-selection-20260927"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def pct(sorted_xs: Sequence[float], p: float) -> Optional[float]:
    if not sorted_xs:
        return None
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    pos = (len(sorted_xs) - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_xs[lo])
    frac = pos - lo
    return float(sorted_xs[lo] * (1.0 - frac) + sorted_xs[hi] * frac)


def stats(values: Iterable[Any]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in values if as_float(v) is not None)
    if not xs:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    s = math.fsum(xs)
    return {
        "count": len(xs), "sum": float(s), "mean": float(s / len(xs)),
        "median": pct(xs, 50), "p05": pct(xs, 5), "p95": pct(xs, 95),
        "min": float(xs[0]), "max": float(xs[-1]),
    }


def merge_counts(dst: Dict[str, int], src: Mapping[Any, Any]) -> None:
    for k, v in (src or {}).items():
        dst[str(k)] = dst.get(str(k), 0) + as_int(v)


def timing(summary: Mapping[str, Any], name: str) -> Dict[str, Any]:
    t = summary.get(name) or {}
    return {
        "count": as_int(t.get("count")),
        "sum": as_float(t.get("sum"), 0.0) or 0.0,
        "mean": as_float(t.get("mean")),
        "median": as_float(t.get("median")),
        "p95": as_float(t.get("p95")),
        "max": as_float(t.get("max")),
    }


def classify_key(key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rollout_key": key, "is_learned": key in LEARNED_KEYS}
    if key in LEARNED_KEYS:
        out.update({"kind": "learned", "learned_seed": int(key[-1]), "seed": int(key[-1]), "terminal_h": 25, "controller_h_int": None})
        return out
    m = re.match(r"^fixed_seed(\d+)_terminal(\d+)_controllerH(\d+)$", key)
    if m:
        seed = int(m.group(1)); terminal_h = int(m.group(2)); controller_h = int(m.group(3))
        out.update({
            "kind": "fixed", "seed": seed, "terminal_h": terminal_h, "controller_h_int": controller_h,
            "is_matched_terminal25_grid": terminal_h == 25 and seed in [0, 1, 2] and controller_h in H_GRID,
            "is_independent_terminal_seed0_grid": seed == 0 and terminal_h == controller_h and controller_h in H_GRID,
        })
    else:
        out.update({"kind": "unknown"})
    return out


def episode_record(summary_path: Path, shard: int) -> Dict[str, Any]:
    s = read_json(summary_path)
    key = str(s.get("rollout_key"))
    c = classify_key(key)
    reset = s.get("reset") or {}
    dec = timing(s, "decision_timing_s")
    dec_gross = timing(s, "decision_gross_timing_s")
    solver = timing(s, "solver_attempt_timing_s")
    selection = timing(s, "selection_timing_s")
    logging_t = timing(s, "logging_timing_s")
    rec: Dict[str, Any] = {
        "summary_path": rel(summary_path), "summary_sha256": sha256(summary_path),
        "shard": shard, "execution_index": as_int(s.get("execution_index")),
        "rollout_key": key, "family": s.get("family"), "seed": as_int(s.get("seed")),
        "case": as_int(s.get("case", s.get("validation_case_index"))),
        "controller_h": s.get("controller_h"), "terminal_source": s.get("terminal_source"),
        "success": bool(s.get("success")), "episode_failure": bool(s.get("episode_failure")),
        "termination": s.get("termination"), "constraint": bool(s.get("constraint")),
        "steps": as_int(s.get("steps")), "total_cost": as_float(s.get("total_cost"), 0.0) or 0.0,
        "performance_cost": as_float(s.get("performance_cost"), 0.0) or 0.0,
        "physical_constraint_cost": as_float(s.get("physical_constraint_cost"), 0.0) or 0.0,
        "constraint_cost": as_float(s.get("constraint_cost"), 0.0) or 0.0,
        "h_penalty": as_float(s.get("h_penalty"), 0.0) or 0.0,
        "initial_failed_steps": as_int(s.get("initial_failed_steps")),
        "solver_failure_steps": as_int(s.get("solver_failure_steps")),
        "recovered_steps": as_int(s.get("recovered_steps")), "retries": as_int(s.get("retries")),
        "deadline_exceed_steps": as_int(s.get("deadline_exceed_steps")), "switches": as_int(s.get("switches")),
        "horizon_counts": dict(s.get("horizon_counts") or {}),
        "unique_horizons": list(s.get("unique_horizons") or []),
        "construction_s": as_float(s.get("construction_s"), 0.0) or 0.0,
        "terminal_load_s_reference": as_float(s.get("terminal_load_s_reference"), 0.0) or 0.0,
        "logging_total_s": as_float(s.get("logging_total_s"), 0.0) or 0.0,
        "episode_wall_s": as_float(s.get("episode_wall_s_including_construction_reset_tracewrites"), 0.0) or 0.0,
        "reset_gross_s": as_float(reset.get("reset_gross_s"), 0.0) or 0.0,
        "reset_controller_s": as_float(reset.get("controller_s"), 0.0) or 0.0,
        "reset_controller_gross_s": as_float(reset.get("controller_gross_s"), 0.0) or 0.0,
        "reset_logging_s": as_float(reset.get("logging_s"), 0.0) or 0.0,
        "decision_count": dec["count"], "decision_sum_s": dec["sum"], "decision_mean_s": dec["mean"],
        "decision_median_s": dec["median"], "decision_p95_s": dec["p95"], "decision_max_s": dec["max"],
        "decision_gross_count": dec_gross["count"], "decision_gross_sum_s": dec_gross["sum"], "decision_gross_mean_s": dec_gross["mean"],
        "decision_gross_p95_s": dec_gross["p95"],
        "solver_count": solver["count"], "solver_sum_s": solver["sum"], "solver_mean_s": solver["mean"],
        "solver_median_s": solver["median"], "solver_p95_s": solver["p95"], "solver_max_s": solver["max"],
        "selection_sum_s": selection["sum"], "selection_mean_s": selection["mean"],
        "logging_timing_sum_s": logging_t["sum"], "logging_timing_mean_s": logging_t["mean"],
    }
    rec.update({"parsed_" + k: v for k, v in c.items() if k != "rollout_key"})
    return rec


def aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    hcounts: Dict[str, int] = {}
    for r in records:
        merge_counts(hcounts, r.get("horizon_counts") or {})
    decision_count = sum(as_int(r.get("decision_count")) for r in records)
    solver_count = sum(as_int(r.get("solver_count")) for r in records)
    decision_sum = math.fsum(as_float(r.get("decision_sum_s"), 0.0) or 0.0 for r in records)
    solver_sum = math.fsum(as_float(r.get("solver_sum_s"), 0.0) or 0.0 for r in records)
    decision_gross_sum = math.fsum(as_float(r.get("decision_gross_sum_s"), 0.0) or 0.0 for r in records)
    return {
        "episodes": len(records), "cases": len({as_int(r.get("case")) for r in records}),
        "steps": sum(as_int(r.get("steps")) for r in records),
        "success_count": sum(1 for r in records if r.get("success")),
        "episode_failure_count": sum(1 for r in records if r.get("episode_failure")),
        "termination_counts": {k: sum(1 for r in records if str(r.get("termination")) == k) for k in sorted({str(r.get("termination")) for r in records})},
        "constraint_episode_count": sum(1 for r in records if r.get("constraint")),
        "constraint_cost_sum": math.fsum(as_float(r.get("constraint_cost"), 0.0) or 0.0 for r in records),
        "initial_failed_steps": sum(as_int(r.get("initial_failed_steps")) for r in records),
        "solver_failure_steps": sum(as_int(r.get("solver_failure_steps")) for r in records),
        "recovered_steps": sum(as_int(r.get("recovered_steps")) for r in records),
        "retries": sum(as_int(r.get("retries")) for r in records),
        "deadline_exceed_steps": sum(as_int(r.get("deadline_exceed_steps")) for r in records),
        "switches": sum(as_int(r.get("switches")) for r in records),
        "horizon_counts": hcounts, "unique_horizons": sorted(int(h) for h in hcounts),
        "total_cost_sum": math.fsum(as_float(r.get("total_cost"), 0.0) or 0.0 for r in records),
        "physical_constraint_cost_sum": math.fsum(as_float(r.get("physical_constraint_cost"), 0.0) or 0.0 for r in records),
        "performance_cost_sum": math.fsum(as_float(r.get("performance_cost"), 0.0) or 0.0 for r in records),
        "h_penalty_sum": math.fsum(as_float(r.get("h_penalty"), 0.0) or 0.0 for r in records),
        "total_cost_episode_distribution": stats(r.get("total_cost") for r in records),
        "physical_constraint_cost_episode_distribution": stats(r.get("physical_constraint_cost") for r in records),
        "steps_episode_distribution": stats(r.get("steps") for r in records),
        "decision_total_s": decision_sum, "decision_count": decision_count,
        "decision_mean_s_per_step_exact_from_episode_sums": (decision_sum / decision_count if decision_count else None),
        "decision_gross_total_s": decision_gross_sum,
        "decision_gross_mean_s_per_step_exact_from_episode_sums": (decision_gross_sum / decision_count if decision_count else None),
        "decision_episode_mean_s_distribution": stats(r.get("decision_mean_s") for r in records),
        "decision_episode_p95_s_distribution": stats(r.get("decision_p95_s") for r in records),
        "decision_global_step_p95_s": None,
        "decision_global_step_p95_note": "not exact from metadata-only episode summaries; traces intentionally not read in this diagnostic",
        "solver_total_s": solver_sum, "solver_count": solver_count,
        "solver_mean_s_per_attempt_exact_from_episode_sums": (solver_sum / solver_count if solver_count else None),
        "solver_episode_mean_s_distribution": stats(r.get("solver_mean_s") for r in records),
        "solver_episode_p95_s_distribution": stats(r.get("solver_p95_s") for r in records),
        "solver_global_attempt_p95_s": None,
        "solver_global_attempt_p95_note": "not exact from metadata-only episode summaries; solver call traces intentionally not read",
        "selection_total_s": math.fsum(as_float(r.get("selection_sum_s"), 0.0) or 0.0 for r in records),
        "construction_total_s": math.fsum(as_float(r.get("construction_s"), 0.0) or 0.0 for r in records),
        "reset_gross_total_s": math.fsum(as_float(r.get("reset_gross_s"), 0.0) or 0.0 for r in records),
        "logging_total_s": math.fsum(as_float(r.get("logging_total_s"), 0.0) or 0.0 for r in records),
        "episode_wall_total_s": math.fsum(as_float(r.get("episode_wall_s"), 0.0) or 0.0 for r in records),
    }


def by_key(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    d: Dict[str, List[Mapping[str, Any]]] = {}
    for r in records:
        d.setdefault(str(r["rollout_key"]), []).append(r)
    return d


def aggregate_by_key(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, rs in sorted(by_key(records).items()):
        out[key] = aggregate(rs)
        out[key].update(classify_key(key))
    return out


def arm_rank_tuple(s: Mapping[str, Any]) -> Tuple[Any, ...]:
    # Higher success is better; lower failures/cost/time is better.  This is a
    # transparent validation nomination, not a post-hoc success-threshold change.
    return (-as_int(s.get("success_count")), as_int(s.get("episode_failure_count")),
            as_float(s.get("physical_constraint_cost_sum"), float("inf")) or float("inf"),
            as_float(s.get("total_cost_sum"), float("inf")) or float("inf"),
            as_float(s.get("decision_mean_s_per_step_exact_from_episode_sums"), float("inf")) or float("inf"))


def choose_nomination(group: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    if not group:
        return {"available": False}
    ranked = sorted(group.items(), key=lambda kv: arm_rank_tuple(kv[1]))
    perf_key, perf = ranked[0]
    best_success = perf["success_count"]
    best_phys = perf["physical_constraint_cost_sum"]
    eligible = [(k, v) for k, v in group.items()
                if v.get("success_count") == best_success and v.get("physical_constraint_cost_sum") <= best_phys * 1.03]
    speed_key, speed = sorted(eligible, key=lambda kv: (kv[1].get("decision_mean_s_per_step_exact_from_episode_sums") or float("inf"), kv[1].get("physical_constraint_cost_sum") or float("inf")))[0]
    return {
        "available": True,
        "performance_optimal_key": perf_key,
        "performance_optimal_summary": perf,
        "speed_within_3pct_cost_key": speed_key,
        "speed_within_3pct_cost_summary": speed,
        "ranking_rule": "max success_count, min episode_failure_count, min physical_constraint_cost_sum, min total_cost_sum, min decision mean; speed-within-3pct uses same best success and physical cost <= 1.03*performance optimum",
    }


def paired_delta(learned_key: str, comp_key: str, recs_by_key_case: Mapping[Tuple[str, int], Mapping[str, Any]]) -> Dict[str, Any]:
    pairs = []
    for case in range(64):
        lr = recs_by_key_case.get((learned_key, case)); cr = recs_by_key_case.get((comp_key, case))
        if lr is None or cr is None:
            continue
        pairs.append((lr, cr))
    deltas = [{
        "case": as_int(l.get("case")),
        "delta_total_cost_learned_minus_fixed": (l["total_cost"] - c["total_cost"]),
        "delta_physical_constraint_cost_learned_minus_fixed": (l["physical_constraint_cost"] - c["physical_constraint_cost"]),
        "delta_steps_learned_minus_fixed": (l["steps"] - c["steps"]),
        "delta_decision_mean_s_learned_minus_fixed": ((l.get("decision_mean_s") or 0.0) - (c.get("decision_mean_s") or 0.0)),
        "delta_decision_sum_s_learned_minus_fixed": (l["decision_sum_s"] - c["decision_sum_s"]),
        "learned_success": bool(l["success"]), "fixed_success": bool(c["success"]),
        "learned_failure": bool(l["episode_failure"]), "fixed_failure": bool(c["episode_failure"]),
        "learned_horizon_counts": l.get("horizon_counts"), "fixed_horizon_counts": c.get("horizon_counts"),
    } for l, c in pairs]
    return {
        "learned_key": learned_key, "comparator_key": comp_key, "paired_cases": len(deltas),
        "learned_success_count": sum(1 for d in deltas if d["learned_success"]),
        "fixed_success_count": sum(1 for d in deltas if d["fixed_success"]),
        "learned_only_success_cases": [d["case"] for d in deltas if d["learned_success"] and not d["fixed_success"]],
        "fixed_only_success_cases": [d["case"] for d in deltas if d["fixed_success"] and not d["learned_success"]],
        "both_failure_cases": [d["case"] for d in deltas if (not d["learned_success"] and not d["fixed_success"])],
        "delta_total_cost_stats": stats(d["delta_total_cost_learned_minus_fixed"] for d in deltas),
        "delta_physical_constraint_cost_stats": stats(d["delta_physical_constraint_cost_learned_minus_fixed"] for d in deltas),
        "delta_steps_stats": stats(d["delta_steps_learned_minus_fixed"] for d in deltas),
        "delta_decision_mean_s_stats": stats(d["delta_decision_mean_s_learned_minus_fixed"] for d in deltas),
        "delta_decision_sum_s_stats": stats(d["delta_decision_sum_s_learned_minus_fixed"] for d in deltas),
        "case_deltas": deltas,
    }


def sensitivity_without_case(records: Sequence[Mapping[str, Any]], key: str, case: int) -> Dict[str, Any]:
    all_rs = [r for r in records if r["rollout_key"] == key]
    wo = [r for r in all_rs if as_int(r.get("case")) != case]
    return {"all_cases": aggregate(all_rs), "without_case%d" % case: aggregate(wo)}


def write_csvs(records: Sequence[Mapping[str, Any]], arm_summaries: Mapping[str, Mapping[str, Any]], paired: Sequence[Mapping[str, Any]], case43: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    episode_csv = OUT_DIR / "episode_table.csv"
    cols = ["shard", "execution_index", "rollout_key", "family", "seed", "case", "controller_h", "success", "episode_failure", "termination", "steps", "total_cost", "physical_constraint_cost", "performance_cost", "h_penalty", "initial_failed_steps", "solver_failure_steps", "retries", "deadline_exceed_steps", "switches", "horizon_counts", "decision_mean_s", "decision_sum_s", "decision_p95_s", "solver_mean_s", "solver_sum_s", "solver_p95_s", "construction_s", "reset_gross_s", "episode_wall_s", "summary_path", "summary_sha256"]
    with episode_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in sorted(records, key=lambda x: (str(x["rollout_key"]), as_int(x["case"]))):
            row = {c: r.get(c) for c in cols}; row["horizon_counts"] = json.dumps(row.get("horizon_counts"), sort_keys=True)
            w.writerow(row)

    arm_csv = OUT_DIR / "arm_summary.csv"
    arm_cols = ["rollout_key", "kind", "seed", "terminal_h", "controller_h_int", "episodes", "cases", "steps", "success_count", "episode_failure_count", "total_cost_sum", "physical_constraint_cost_sum", "h_penalty_sum", "decision_mean_s_per_step_exact_from_episode_sums", "decision_total_s", "solver_mean_s_per_attempt_exact_from_episode_sums", "solver_total_s", "construction_total_s", "reset_gross_total_s", "horizon_counts", "unique_horizons"]
    with arm_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=arm_cols); w.writeheader()
        for key, s in sorted(arm_summaries.items()):
            row = {c: s.get(c) for c in arm_cols}; row["rollout_key"] = key
            row["horizon_counts"] = json.dumps(row.get("horizon_counts"), sort_keys=True)
            row["unique_horizons"] = json.dumps(row.get("unique_horizons"))
            w.writerow(row)

    paired_csv = OUT_DIR / "paired_deltas_summary.csv"
    pcols = ["learned_key", "comparator_key", "paired_cases", "learned_success_count", "fixed_success_count", "delta_physical_mean", "delta_physical_median", "delta_physical_p95", "delta_total_mean", "delta_decision_mean_s_mean", "learned_only_success_cases", "fixed_only_success_cases"]
    with paired_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pcols); w.writeheader()
        for p in paired:
            row = {
                "learned_key": p["learned_key"], "comparator_key": p["comparator_key"], "paired_cases": p["paired_cases"],
                "learned_success_count": p["learned_success_count"], "fixed_success_count": p["fixed_success_count"],
                "delta_physical_mean": p["delta_physical_constraint_cost_stats"].get("mean"),
                "delta_physical_median": p["delta_physical_constraint_cost_stats"].get("median"),
                "delta_physical_p95": p["delta_physical_constraint_cost_stats"].get("p95"),
                "delta_total_mean": p["delta_total_cost_stats"].get("mean"),
                "delta_decision_mean_s_mean": p["delta_decision_mean_s_stats"].get("mean"),
                "learned_only_success_cases": json.dumps(p["learned_only_success_cases"]),
                "fixed_only_success_cases": json.dumps(p["fixed_only_success_cases"]),
            }
            w.writerow(row)

    case_csv = OUT_DIR / "case43_table.csv"
    with case_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in sorted(case43, key=lambda x: str(x["rollout_key"])):
            row = {c: r.get(c) for c in cols}; row["horizon_counts"] = json.dumps(row.get("horizon_counts"), sort_keys=True)
            w.writerow(row)
    return {"episode_table_csv": rel(episode_csv), "arm_summary_csv": rel(arm_csv), "paired_deltas_summary_csv": rel(paired_csv), "case43_table_csv": rel(case_csv)}


def append_once(path: Path, marker: str, body: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    token = "<!-- %s -->" % marker
    if token in old:
        return
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + body.strip() + "\n", encoding="utf-8")


def short_arm_line(key: str, s: Mapping[str, Any]) -> str:
    return (f"`{key}`: episodes={s.get('episodes')}, success={s.get('success_count')}, failures={s.get('episode_failure_count')}, "
            f"phys={s.get('physical_constraint_cost_sum'):.6g}, total={s.get('total_cost_sum'):.6g}, "
            f"decision_mean_s={s.get('decision_mean_s_per_step_exact_from_episode_sums')}, horizons={s.get('horizon_counts')}")


def write_summary_md(path: Path, raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle validation64 full aggregate/model-selection diagnostic",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Metadata-only aggregate over the already completed 12 validation64 shards. No new simulations/control steps/training; no validation bank reopen; sealed final test remains closed, unopened, and unhashed.",
        "",
        "## Budget/access accounting",
        "",
        f"- Existing validation episodes read from per-episode summaries: `{raw['budget_accounting']['existing_validation_episodes_read']}`; control steps represented: `{raw['budget_accounting']['existing_validation_control_steps_read']}`.",
        f"- New simulations/control/gradient steps: `0/0/0`; failed formal attempts preserved: `{raw['budget_accounting']['preserved_failed_formal_attempts']}`.",
        "",
        "## Learned candidates (all 64 cases)",
        "",
    ]
    for key in LEARNED_KEYS:
        lines.append("- " + short_arm_line(key, raw["arm_summaries"][key]))
    lines += ["", "## Fixed-H validation nominations", ""]
    nom = raw["model_selection"]
    lines.append(f"- Matched-terminal grid aggregated across seeds: performance-optimal `{nom['matched_terminal_all_seeds']['performance_optimal_key']}`, speed-within-3%-cost `{nom['matched_terminal_all_seeds']['speed_within_3pct_cost_key']}`.")
    lines.append(f"- Independent-terminal seed0 grid: performance-optimal `{nom['independent_terminal_seed0']['performance_optimal_key']}`, speed-within-3%-cost `{nom['independent_terminal_seed0']['speed_within_3pct_cost_key']}`.")
    for seed, seed_nom in nom["matched_terminal_by_seed"].items():
        lines.append(f"- Matched-terminal seed {seed}: performance-optimal `{seed_nom['performance_optimal_key']}`, speed-within-3%-cost `{seed_nom['speed_within_3pct_cost_key']}`.")
    lines += ["", "## Adaptive gate diagnostic", ""]
    for key, g in raw["adaptive_gate_diagnostic"].items():
        lines.append(f"- `{key}`: adaptive_used={g['adaptive_horizons_used']}, horizons={g['unique_horizons']}, pass_vs_same_seed_speed_nomination={g['passes_against_same_seed_speed_nomination']}, reasons={g['reasons']}")
    lines += ["", "## Case43 sensitivity", ""]
    c43 = raw["case43_sensitivity"]
    lines.append(f"- learned_s2 case43 primary record: `{c43['learned_s2_case43']}`")
    lines.append(f"- learned_s2 all-vs-without-case43 physical cost sums: `{c43['learned_s2_all_vs_without_case43']['all_cases']['physical_constraint_cost_sum']}` vs `{c43['learned_s2_all_vs_without_case43']['without_case43']['physical_constraint_cost_sum']}`. Case43 is retained in all primary summaries.")
    lines += ["", "## Artifacts", ""]
    for k, v in raw["tables"].items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Interpretation", "", raw["interpretation"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if OUT_DIR.exists() and (OUT_DIR / "completed.json").exists():
        done = read_json(OUT_DIR / "completed.json")
        print(json.dumps({"already_completed": True, "completed": rel(OUT_DIR / "completed.json"), "passed": done.get("passed")}, indent=2, sort_keys=True))
        return 0 if done.get("passed") else 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    stamp = created.replace("-", "").replace(":", "").replace("+00:00", "")
    failures: List[str] = []

    shard_files: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []
    for shard in SHARDS:
        sdir = VALIDATION_ROOT / ("shard%02d" % shard)
        completed = sdir / "completed.json"; raw = sdir / "raw.json"; summary_md = sdir / "summary.md"
        if not completed.exists() or not summary_md.exists() or not raw.exists():
            failures.append("missing shard top-level artifact shard%02d" % shard); continue
        c = read_json(completed)
        if c.get("passed") is not True:
            failures.append("shard%02d completed marker not passed" % shard)
        shard_files["shard%02d" % shard] = {"completed": rel(completed), "completed_sha256": sha256(completed), "raw": rel(raw), "raw_sha256": sha256(raw), "summary": rel(summary_md), "summary_sha256": sha256(summary_md)}
        for sp in sorted((sdir / "episodes").glob("*/summary.json")):
            records.append(episode_record(sp, shard))

    if len(records) != 2688:
        failures.append("expected 2688 episode summaries, found %d" % len(records))
    if sum(r["steps"] for r in records) != 236348:
        failures.append("expected 236348 control steps from cumulative audit, found %d" % sum(r["steps"] for r in records))
    recs_by_key_case: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    duplicate_pairs: List[str] = []
    for r in records:
        pair = (str(r["rollout_key"]), as_int(r["case"]))
        if pair in recs_by_key_case:
            duplicate_pairs.append("%s/%s" % pair)
        recs_by_key_case[pair] = r
    if duplicate_pairs:
        failures.append("duplicate rollout_key/case records: %s" % duplicate_pairs[:10])

    arm_summaries = aggregate_by_key(records)
    if len(arm_summaries) != 42:
        failures.append("expected 42 unique rollout keys, found %d" % len(arm_summaries))
    for key, s in arm_summaries.items():
        if s.get("episodes") != 64:
            failures.append("rollout key %s expected 64 episodes, found %s" % (key, s.get("episodes")))

    matched_by_seed: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    matched_by_h_records: Dict[int, List[Mapping[str, Any]]] = {h: [] for h in H_GRID}
    independent_seed0: Dict[str, Mapping[str, Any]] = {}
    for h in H_GRID:
        for seed in [0, 1, 2]:
            key = f"fixed_seed{seed}_terminal25_controllerH{h}"
            if key in arm_summaries:
                matched_by_seed.setdefault(str(seed), {})[key] = arm_summaries[key]
                matched_by_h_records[h].extend([r for r in records if r["rollout_key"] == key])
        ikey = f"fixed_seed0_terminal{h}_controllerH{h}"
        if ikey in arm_summaries:
            independent_seed0[ikey] = arm_summaries[ikey]
    matched_all_h = {f"matched_terminal_allseeds_H{h}": aggregate(matched_by_h_records[h]) for h in H_GRID if matched_by_h_records[h]}
    model_selection = {
        "matched_terminal_by_seed": {seed: choose_nomination(group) for seed, group in matched_by_seed.items()},
        "matched_terminal_all_seeds": choose_nomination(matched_all_h),
        "independent_terminal_seed0": choose_nomination(independent_seed0),
        "matched_terminal_all_seeds_by_H": matched_all_h,
        "rule_is_validation_only": True,
    }

    paired: List[Dict[str, Any]] = []
    for learned in LEARNED_KEYS:
        for seed in [0, 1, 2]:
            for h in H_GRID:
                key = f"fixed_seed{seed}_terminal25_controllerH{h}"
                if key in arm_summaries:
                    paired.append(paired_delta(learned, key, recs_by_key_case))
        for h in H_GRID:
            key = f"fixed_seed0_terminal{h}_controllerH{h}"
            if key in arm_summaries:
                paired.append(paired_delta(learned, key, recs_by_key_case))

    adaptive_gate: Dict[str, Any] = {}
    for learned in LEARNED_KEYS:
        ls = arm_summaries[learned]
        seed = int(learned[-1])
        speed_key = model_selection["matched_terminal_by_seed"][str(seed)]["speed_within_3pct_cost_key"]
        fs = arm_summaries[speed_key]
        adaptive_used = len(ls.get("unique_horizons") or []) > 1
        success_noninferior = ls["success_count"] >= math.ceil(0.98 * fs["success_count"])
        cost_ok = ls["physical_constraint_cost_sum"] <= 1.03 * fs["physical_constraint_cost_sum"]
        decision_ok = (ls["decision_mean_s_per_step_exact_from_episode_sums"] is not None and fs["decision_mean_s_per_step_exact_from_episode_sums"] is not None and ls["decision_mean_s_per_step_exact_from_episode_sums"] <= 0.90 * fs["decision_mean_s_per_step_exact_from_episode_sums"])
        reasons = []
        if not adaptive_used: reasons.append("not adaptive on validation: only one horizon used")
        if not success_noninferior: reasons.append("success count below 2% noninferiority proxy vs same-seed speed nomination")
        if not cost_ok: reasons.append("physical/control cost exceeds +3% proxy vs same-seed speed nomination")
        if not decision_ok: reasons.append("decision mean does not show >=10% reduction vs same-seed speed nomination")
        adaptive_gate[learned] = {
            "same_seed_speed_nomination": speed_key,
            "adaptive_horizons_used": adaptive_used,
            "unique_horizons": ls.get("unique_horizons"),
            "success_noninferior_proxy": success_noninferior,
            "cost_within_3pct_proxy": cost_ok,
            "decision_at_least_10pct_faster_proxy": decision_ok,
            "passes_against_same_seed_speed_nomination": bool(adaptive_used and success_noninferior and cost_ok and decision_ok),
            "reasons": reasons,
        }

    case43_records = [r for r in records if as_int(r.get("case")) == 43]
    case43_by_key = {str(r["rollout_key"]): r for r in case43_records}
    case43_sensitivity = {
        "case43_table_records": case43_records,
        "learned_s2_case43": case43_by_key.get("learned_s2"),
        "fixed_seed2_terminal25_controllerH25_case43": case43_by_key.get("fixed_seed2_terminal25_controllerH25"),
        "learned_s2_all_vs_without_case43": sensitivity_without_case(records, "learned_s2", 43),
        "learned_s0_all_vs_without_case43": sensitivity_without_case(records, "learned_s0", 43),
        "learned_s1_all_vs_without_case43": sensitivity_without_case(records, "learned_s1", 43),
    }

    tables = write_csvs(records, arm_summaries, paired, case43_records)
    interpretation = (
        "The frozen validation campaign is complete and shows that the current IMPROVED latency-tree candidate is not a robust 3-seed adaptive-horizon success: learned_s0 and learned_s1 are H25-only, while learned_s2 is adaptive but has one catastrophic validation failure. "
        "These findings are validation/development model-selection evidence only; the sealed final test remains unauthorized. Paired fixed-H comparisons and timing here should guide diagnosis/revision, not a reproduction-success claim."
    )
    raw = {
        "created_utc": created, "script": rel(SCRIPT), "script_sha256": sha256(SCRIPT),
        "formal_scientific_evidence_created": True,
        "evidence_type": "derived_validation64_aggregate_model_selection_from_existing_outputs",
        "new_simulations": 0, "new_control_steps": 0, "new_gradient_steps": 0,
        "validation_accessed": True, "validation_bank_reopened": False,
        "validation64_existing_outputs_read_only": True,
        "test_accessed": False, "sealed_test_bank_content_opened": False, "sealed_test_bank_hashed": False,
        "gate_json_hash": sha256(GATE_JSON) if GATE_JSON.exists() else None,
        "shard_files": shard_files,
        "budget_accounting": {
            "existing_validation_episodes_read": len(records),
            "existing_validation_control_steps_read": sum(r["steps"] for r in records),
            "planned_validation_episodes": 2688,
            "new_validation_episodes": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "preserved_failed_formal_attempts": ["20260926T161356_fb71c8d7 shard02 modern-interpreter TensorFlow-missing attempt: validation bank opened, 0 episodes/control steps/gradient steps"],
        },
        "arm_summaries": arm_summaries,
        "model_selection": model_selection,
        "paired_deltas": paired,
        "adaptive_gate_diagnostic": adaptive_gate,
        "case43_sensitivity": case43_sensitivity,
        "tables": tables,
        "interpretation": interpretation,
        "failures": failures,
    }
    raw_path = OUT_DIR / "raw.json"; summary_path = OUT_DIR / "summary.md"; completed_path = OUT_DIR / "completed.json"
    write_json(raw_path, raw)
    write_summary_md(summary_path, raw)

    backup_request = {
        "created_utc": created,
        "request": "external_backup_after_vehicle_validation64_full_aggregate_model_selection",
        "reason": "Preserve full validation64 aggregate/model-selection diagnostic, tables, docs, and run registry before subsequent case43 replay/selection-log diagnostic or method revision.",
        "validation_access": "read existing validation64 shard completed/raw/summary metadata and per-episode summary.json only; validation bank not reopened; no traces read",
        "sealed_test_access": "none; sealed final test remains closed, unauthorized, unopened, and unhashed",
        "formal_scientific_evidence_created": True,
        "new_simulations": 0, "new_control_steps": 0, "new_gradient_steps": 0,
        "artifacts_to_backup": [rel(SCRIPT), rel(raw_path), rel(summary_path), rel(completed_path)] + list(tables.values()),
    }
    backup_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_VALIDATION64_FULL_AGGREGATE_MODEL_SELECTION_%s.json" % stamp)
    write_json(backup_path, backup_request)

    passed = not failures
    completed = {
        "created_utc": created, "passed": passed,
        "raw": rel(raw_path), "raw_sha256": sha256(raw_path),
        "summary": rel(summary_path), "summary_sha256": sha256(summary_path),
        "tables": {k: {"path": v, "sha256": sha256(ROOT / v)} for k, v in tables.items()},
        "backup_request": rel(backup_path), "backup_request_sha256": sha256(backup_path),
        "formal_scientific_evidence_created": True,
        "new_simulations": 0, "new_control_steps": 0, "new_gradient_steps": 0,
        "validation_accessed": True, "validation_bank_reopened": False,
        "test_accessed": False, "sealed_test_bank_content_opened": False, "sealed_test_bank_hashed": False,
        "headline": {
            "matched_terminal_all_seeds_performance_optimal": model_selection["matched_terminal_all_seeds"].get("performance_optimal_key"),
            "matched_terminal_all_seeds_speed_within_3pct_cost": model_selection["matched_terminal_all_seeds"].get("speed_within_3pct_cost_key"),
            "independent_terminal_seed0_performance_optimal": model_selection["independent_terminal_seed0"].get("performance_optimal_key"),
            "adaptive_gate_diagnostic": adaptive_gate,
        },
        "failures": failures,
    }
    write_json(completed_path, completed)

    doc_body = f"""
### Vehicle validation64 full aggregate/model-selection diagnostic ({created})

- Metadata-only validation aggregation over existing shard outputs: 2688 episode summaries and {sum(r['steps'] for r in records)} represented control steps; no new simulations/control/training, no validation-bank reopen, no sealed-test access/hash.
- Artifacts: `{rel(raw_path)}`, `{rel(summary_path)}`, completed marker `{rel(completed_path)}`; tables `{tables}`.
- Matched-terminal all-seed fixed-H nominations: performance `{model_selection['matched_terminal_all_seeds'].get('performance_optimal_key')}`, speed-within-3%-cost `{model_selection['matched_terminal_all_seeds'].get('speed_within_3pct_cost_key')}`.
- Independent-terminal seed0 nominations: performance `{model_selection['independent_terminal_seed0'].get('performance_optimal_key')}`, speed-within-3%-cost `{model_selection['independent_terminal_seed0'].get('speed_within_3pct_cost_key')}`.
- Learned candidates: {short_arm_line('learned_s0', arm_summaries['learned_s0'])}; {short_arm_line('learned_s1', arm_summaries['learned_s1'])}; {short_arm_line('learned_s2', arm_summaries['learned_s2'])}.
- Adaptive diagnostic: `{adaptive_gate}`.
- Interpretation: {interpretation}
- Next action after backup: prioritize targeted diagnosis of selection/training collapse (s0/s1 constant H25) and learned_s2 case43 deterministic replay/one-variable ablations before any final-test gate; likely prepare a versioned IMPROVED method revision.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md"]:
        append_once(ROOT / doc, DOC_MARKER + "-" + doc, doc_body)
    append_once(ROOT / "DECISIONS.md", DOC_MARKER + "-decision", f"""
### Decision after full vehicle validation64 aggregate ({created})

Before evidence: all 12 validation shards had passed audits, v3 policy-collapse diagnostic showed learned_s0/s1 structurally constant H25 and learned_s2 adaptive with one catastrophic validation failure.

Aggregate change: no controller/model/data change; this diagnostic derived paired validation/model-selection summaries from existing per-episode summary files only.

Decision: do not open final test for the current frozen candidate. Treat current IMPROVED latency-tree as insufficient for a robust 3-seed adaptive superiority claim unless a later independent validation revision changes the evidence. Next bounded diagnostics should target (1) training/selection collapse for s0/s1 and (2) learned_s2 case43 replay/ablation causality, then freeze a versioned IMPROVED method revision.

Evidence: `{rel(completed_path)}` headline `{completed['headline']}`.
""")

    print(json.dumps(completed, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
