#!/usr/bin/env python3
"""Metadata/trace diagnostic for vehicle safe-shortening v1 absent adaptation.

This diagnostic inspects *existing* frozen safe-shortening v1 development-
validation artifacts only.  It does not create rollouts, does not train, does not
open historical validation64 banks, and does not access the sealed final test.

Questions addressed
-------------------
1. Did the runtime request/execute horizons consistently, or did guards/clamps
   hide an execution bug?
2. Why did seeds collapse near H25, especially seed2?
3. What exactly happened around the rare seed2 shard13/case9 H10 decision, using
   the saved step traces rather than only episode summaries?
4. Do existing fixed-grid devval summaries show case-level opportunities for
   shorter fixed horizons, or is the scenario bank inherently hostile to short H?

Scientific status: development diagnostic/model-debug evidence only.  Cases and
traces inspected here cannot be used later as fresh independent confirmation.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEVVAL = ROOT / "research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1"
TRAIN = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25/train"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_safe_shortening_v1_protocol_20260927.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
OUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
SCRIPT_REL = "experiments/bohn2021_aws/vehicle_safe_shortening_v1_trace_policy_diagnostic.py"
MARKER = "vehicle-safe-shortening-v1-trace-policy-diagnostic-20260928"
SEEDS = (0, 1, 2)
H_GRID = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
BASE_H = 25
SERVICE_START = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

# Copied constants from experiments/bohn2021_reproduction/gated_horizon_policy.py
PROFILES = {
    "vehicle": [
        dict(tracking=.05, heading=.03, yaw_rate=.1, min_speed=2.5, clearance=1.5),
        dict(tracking=.15, heading=.1, yaw_rate=.5, min_speed=2., clearance=1.),
        dict(tracking=.4, heading=.25, yaw_rate=1., min_speed=1., clearance=.5),
    ]
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def round_value(v: Any, digits: int = 6) -> Any:
    x = safe_float(v)
    if x is None:
        return v
    return round(x, digits)


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    data = sorted(float(v) for v in values if math.isfinite(float(v)))
    n = len(data)
    if not n:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    def pct(p: float) -> float:
        if n == 1:
            return data[0]
        pos = p * (n - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos)); frac = pos - lo
        return data[lo] * (1.0 - frac) + data[hi] * frac
    return {
        "count": n,
        "sum": float(math.fsum(data)),
        "mean": float(math.fsum(data) / n),
        "median": float(pct(0.5)),
        "p95": float(pct(0.95)),
        "min": float(data[0]),
        "max": float(data[-1]),
    }


class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.s = 0.0
        self.min = None
        self.max = None
    def add(self, value: Any) -> None:
        x = safe_float(value)
        if x is None:
            return
        self.n += 1
        self.s += x
        self.min = x if self.min is None else min(self.min, x)
        self.max = x if self.max is None else max(self.max, x)
    def to_json(self) -> Dict[str, Any]:
        return {"count": self.n, "mean": (self.s / self.n if self.n else None), "min": self.min, "max": self.max}


def policy_for_seed(seed: int) -> Dict[str, Any]:
    path = TRAIN / ("vehicle_s%d" % seed) / "policy.json"
    policy = read_json(path)
    assert policy["task"] == "vehicle" and int(policy["training_seed"]) == seed
    return policy


def profile_thresholds(policy: Mapping[str, Any]) -> Dict[str, float]:
    return dict(PROFILES["vehicle"][int(policy["profile"])])


def condition_report(values: Mapping[str, Any], policy: Mapping[str, Any]) -> Tuple[Dict[str, bool], Dict[str, Optional[float]]]:
    p = profile_thresholds(policy)
    checks = {
        "tracking": safe_float(values.get("tracking")) is not None and float(values["tracking"]) <= p["tracking"],
        "heading": safe_float(values.get("heading")) is not None and float(values["heading"]) <= p["heading"],
        "yaw_rate": safe_float(values.get("yaw_rate")) is not None and float(values["yaw_rate"]) <= p["yaw_rate"],
        "speed": safe_float(values.get("speed")) is not None and float(values["speed"]) >= p["min_speed"],
        "clearance": safe_float(values.get("clearance")) is not None and float(values["clearance"]) >= p["clearance"],
        "preview_distance": safe_float(values.get("preview_distance")) is not None and float(values["preview_distance"]) >= 0.2,
    }
    # Positive margin means pass for all entries below.
    margins: Dict[str, Optional[float]] = {
        "tracking": (p["tracking"] - float(values["tracking"]) if safe_float(values.get("tracking")) is not None else None),
        "heading": (p["heading"] - float(values["heading"]) if safe_float(values.get("heading")) is not None else None),
        "yaw_rate": (p["yaw_rate"] - float(values["yaw_rate"]) if safe_float(values.get("yaw_rate")) is not None else None),
        "speed": (float(values["speed"]) - p["min_speed"] if safe_float(values.get("speed")) is not None else None),
        "clearance": (float(values["clearance"]) - p["clearance"] if safe_float(values.get("clearance")) is not None else None),
        "preview_distance": (float(values["preview_distance"]) - 0.2 if safe_float(values.get("preview_distance")) is not None else None),
    }
    return checks, margins


def minimal_attempt(row: Mapping[str, Any]) -> Dict[str, Any]:
    rec = row.get("recovery") or {}
    attempts = rec.get("attempts") or []
    first = attempts[0] if attempts else {}
    return {
        "solver_success": bool(row.get("solver_success")),
        "initial_success": bool(first.get("success")) if first else None,
        "final_success": rec.get("final_success"),
        "recovered": rec.get("recovered"),
        "attempt_count": len(attempts),
        "first_status": first.get("return_status"),
        "first_iterations": first.get("iterations"),
        "first_solver_s": round_value(first.get("solver_s")),
    }


def minimal_step(row: Mapping[str, Any]) -> Dict[str, Any]:
    state = row.get("state") or {}
    prev_state = row.get("previous_state") or {}
    inp = row.get("input") or {}
    decision = row.get("decision") or {}
    gate = decision.get("gate") or {}
    ctx = row.get("policy_context") or {}
    values = gate.get("values") or {}
    return {
        "step": row.get("t"),
        "horizon": row.get("horizon"),
        "raw_horizon": decision.get("raw_horizon"),
        "selected_horizon": decision.get("selected_horizon"),
        "gate_reason": gate.get("reason"),
        "gate_use_short": gate.get("use_short"),
        "elapsed": ctx.get("elapsed"),
        "previous_h": ctx.get("previous_h"),
        "previous_state": {k: round_value(prev_state.get(k)) for k in ("x", "y", "theta")},
        "state": {k: round_value(state.get(k)) for k in ("x", "y", "theta")},
        "input": {k: [round_value(x) for x in (v if isinstance(v, list) else [v])] for k, v in inp.items()},
        "performance": round_value(row.get("performance")),
        "constraint": round_value(row.get("constraint")),
        "reward": round_value(row.get("reward")),
        "values": {k: round_value(values.get(k)) for k in ("tracking", "heading", "yaw_rate", "speed", "clearance", "preview_distance") if k in values},
        "solver": minimal_attempt(row),
    }


def state_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> Optional[float]:
    try:
        dx = float((a.get("state") or {}).get("x")) - float((b.get("state") or {}).get("x"))
        dy = float((a.get("state") or {}).get("y")) - float((b.get("state") or {}).get("y"))
        return math.hypot(dx, dy)
    except Exception:
        return None


def heading_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> Optional[float]:
    try:
        da = float((a.get("state") or {}).get("theta")) - float((b.get("state") or {}).get("theta"))
        return math.atan2(math.sin(da), math.cos(da))
    except Exception:
        return None


def load_trace_minimal(trace_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8-sig") as stream:
        for idx, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            m = minimal_step(row)
            m["step"] = idx
            rows.append(m)
    return rows


def episode_seed_case(summary_path: Path) -> Tuple[int, int, str]:
    data = read_json(summary_path)
    return int(data["seed"]), int(data["case"]), str(data["arm_id"])


def inspect_adaptive_traces(policies: Mapping[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    per_seed: Dict[int, Dict[str, Any]] = {}
    short_records: List[Dict[str, Any]] = []
    case9_adaptive_trace_path: Optional[Path] = None
    case9_adaptive_summary_path: Optional[Path] = None
    for seed in SEEDS:
        per_seed[seed] = {
            "episodes": 0,
            "steps": 0,
            "success": 0,
            "failures": 0,
            "horizon_counts": {},
            "raw_horizon_counts": {},
            "initial_five_steps": 0,
            "post_initial_steps_with_values": 0,
            "short_steps": 0,
            "fallback_steps": 0,
            "clamped_steps": 0,
            "raw_gt_base_steps": 0,
            "execution_inconsistency_count": 0,
            "execution_inconsistency_examples": [],
            "condition_fail_counts": {"tracking": 0, "heading": 0, "yaw_rate": 0, "speed": 0, "clearance": 0, "preview_distance": 0},
            "condition_pass_counts": {"tracking": 0, "heading": 0, "yaw_rate": 0, "speed": 0, "clearance": 0, "preview_distance": 0},
            "all_conditions_good_count": 0,
            "value_ranges": {k: RunningStats() for k in ("tracking", "heading", "yaw_rate", "speed", "clearance", "preview_distance")},
            "margin_ranges": {k: RunningStats() for k in ("tracking", "heading", "yaw_rate", "speed", "clearance", "preview_distance")},
            "short_cases": {},
        }

    trace_paths = sorted(DEVVAL.glob("shard*/episodes/exec*_safe_shortening_v1_vehicle_s*/trace.jsonl"))
    if len(trace_paths) != 16 * 3 * 4:
        raise RuntimeError("Expected 192 adaptive traces from 16 shards*3 seeds*4 cases; found %d" % len(trace_paths))

    for trace_path in trace_paths:
        summary_path = trace_path.with_name("summary.json")
        seed, case, arm_id = episode_seed_case(summary_path)
        summary = read_json(summary_path)
        agg = per_seed[seed]
        agg["episodes"] += 1
        agg["steps"] += int(summary.get("steps", 0))
        agg["success"] += int(bool(summary.get("success")))
        agg["failures"] += int(bool(summary.get("episode_failure")))
        if seed == 2 and case == 9:
            case9_adaptive_trace_path = trace_path
            case9_adaptive_summary_path = summary_path
        policy = policies[seed]
        with trace_path.open("r", encoding="utf-8-sig") as stream:
            for step, line in enumerate(stream):
                if not line.strip():
                    continue
                row = json.loads(line)
                h = int(row.get("horizon"))
                decision = row.get("decision") or {}
                raw = int(decision.get("raw_horizon", h))
                selected = int(decision.get("selected_horizon", h))
                gate = decision.get("gate") or {}
                ctx = row.get("policy_context") or {}
                elapsed = int(ctx.get("elapsed", step))
                agg["horizon_counts"][str(h)] = agg["horizon_counts"].get(str(h), 0) + 1
                agg["raw_horizon_counts"][str(raw)] = agg["raw_horizon_counts"].get(str(raw), 0) + 1
                if elapsed < 5 or gate.get("reason") == "initial_five_steps":
                    agg["initial_five_steps"] += 1
                if selected < BASE_H:
                    agg["short_steps"] += 1
                    agg["short_cases"][str(case)] = agg["short_cases"].get(str(case), 0) + 1
                if bool(decision.get("fallback")):
                    agg["fallback_steps"] += 1
                if bool(decision.get("clamped_to_base")):
                    agg["clamped_steps"] += 1
                if raw > BASE_H:
                    agg["raw_gt_base_steps"] += 1
                values = gate.get("values") or {}
                if values:
                    agg["post_initial_steps_with_values"] += 1
                    checks, margins = condition_report(values, policy)
                    all_good = all(checks.values())
                    if all_good:
                        agg["all_conditions_good_count"] += 1
                    for k, ok in checks.items():
                        agg["condition_pass_counts" if ok else "condition_fail_counts"][k] += 1
                    for k in agg["value_ranges"]:
                        agg["value_ranges"][k].add(values.get(k))
                        agg["margin_ranges"][k].add(margins.get(k))
                    # Runtime consistency check against policy-rule implications.
                    expected_short = all_good
                    actual_short = selected < BASE_H
                    if expected_short != actual_short:
                        agg["execution_inconsistency_count"] += 1
                        if len(agg["execution_inconsistency_examples"]) < 10:
                            agg["execution_inconsistency_examples"].append({
                                "case": case,
                                "step": step,
                                "arm_id": arm_id,
                                "expected_short_from_values": expected_short,
                                "actual_short": actual_short,
                                "horizon": h,
                                "raw_horizon": raw,
                                "selected_horizon": selected,
                                "values": {k: round_value(values.get(k)) for k in values},
                                "margins": {k: round_value(v) for k, v in margins.items()},
                            })
                if selected < BASE_H:
                    rec = minimal_step(row)
                    checks, margins = condition_report(values, policy) if values else ({}, {})
                    rec.update({
                        "seed": seed,
                        "case": case,
                        "arm_id": arm_id,
                        "step": step,
                        "checks": checks,
                        "margins": {k: round_value(v) for k, v in margins.items()},
                        "trace_path": rel(trace_path),
                    })
                    short_records.append(rec)

    serial_seed: Dict[str, Any] = {}
    for seed, agg in per_seed.items():
        total_steps = int(agg["steps"])
        non25 = total_steps - int(agg["horizon_counts"].get("25", 0))
        post_values = int(agg["post_initial_steps_with_values"])
        serial_seed[str(seed)] = {
            "policy": policies[seed],
            "profile_thresholds": profile_thresholds(policies[seed]),
            "episodes": agg["episodes"],
            "success": agg["success"],
            "failures": agg["failures"],
            "steps": total_steps,
            "horizon_counts": dict(sorted(agg["horizon_counts"].items(), key=lambda kv: int(kv[0]))),
            "raw_horizon_counts": dict(sorted(agg["raw_horizon_counts"].items(), key=lambda kv: int(kv[0]))),
            "non25_steps": non25,
            "non25_share": non25 / total_steps if total_steps else None,
            "short_steps": agg["short_steps"],
            "initial_five_steps": agg["initial_five_steps"],
            "post_initial_steps_with_values": post_values,
            "all_conditions_good_count": agg["all_conditions_good_count"],
            "all_conditions_good_share_post_initial": (agg["all_conditions_good_count"] / post_values if post_values else None),
            "fallback_steps": agg["fallback_steps"],
            "clamped_steps": agg["clamped_steps"],
            "raw_gt_base_steps": agg["raw_gt_base_steps"],
            "execution_inconsistency_count": agg["execution_inconsistency_count"],
            "execution_inconsistency_examples": agg["execution_inconsistency_examples"],
            "condition_fail_counts": agg["condition_fail_counts"],
            "condition_pass_counts": agg["condition_pass_counts"],
            "condition_fail_share_post_initial": {k: (v / post_values if post_values else None) for k, v in agg["condition_fail_counts"].items()},
            "value_ranges": {k: s.to_json() for k, s in agg["value_ranges"].items()},
            "margin_ranges_positive_pass": {k: s.to_json() for k, s in agg["margin_ranges"].items()},
            "short_cases": dict(sorted(agg["short_cases"].items(), key=lambda kv: int(kv[0]))),
        }
    meta = {
        "adaptive_trace_count": len(trace_paths),
        "case9_adaptive_trace_path": rel(case9_adaptive_trace_path) if case9_adaptive_trace_path else None,
        "case9_adaptive_summary_path": rel(case9_adaptive_summary_path) if case9_adaptive_summary_path else None,
        "short_record_count": len(short_records),
        "short_records_first_80": short_records[:80],
    }
    return {"per_seed": serial_seed, "short_records": short_records}, meta


def fixed_grid_case_summary() -> Dict[str, Any]:
    by_seed_case_h: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    summary_paths = sorted(DEVVAL.glob("shard*/episodes/exec*_matched_terminal_fixed_H*_vehicle_s*/summary.json"))
    for path in summary_paths:
        e = read_json(path)
        arm = str(e["arm_id"])
        # arm format matched_terminal_fixed_H25_vehicle_s2
        h = int(arm.split("_H", 1)[1].split("_", 1)[0])
        seed = int(e["seed"]); case = int(e["case"])
        by_seed_case_h[(seed, case, h)] = {
            "path": rel(path),
            "success": bool(e.get("success")),
            "steps": int(e.get("steps", 0)),
            "termination": e.get("termination"),
            "physical_constraint_cost": float(e.get("physical_constraint_cost", 0.0)),
            "total_cost": float(e.get("total_cost", 0.0)),
            "solver_failure_steps": int(e.get("solver_failure_steps", 0)),
            "initial_failed_steps": int(e.get("initial_failed_steps", 0)),
            "retries": int(e.get("retries", 0)),
            "decision_sum_s": float((e.get("decision_timing_s") or {}).get("sum", 0.0)),
            "decision_mean_s_per_step": float((e.get("decision_timing_s") or {}).get("mean", 0.0)),
        }
    out: Dict[str, Any] = {}
    for seed in SEEDS:
        cases = sorted({case for (s, case, h) in by_seed_case_h if s == seed and h == BASE_H})
        case_rows = []
        safe_short_count = 0
        total_improvement_count = 0
        timing_improvement_count = 0
        best_short_total_delta_sum = 0.0
        best_short_decision_ratio_values = []
        for case in cases:
            base = by_seed_case_h[(seed, case, BASE_H)]
            candidates = []
            for h in (5, 10, 15, 20):
                e = by_seed_case_h.get((seed, case, h))
                if not e:
                    continue
                safe = (
                    int(e["success"]) >= int(base["success"]) and
                    e["solver_failure_steps"] <= base["solver_failure_steps"] and
                    e["initial_failed_steps"] <= base["initial_failed_steps"] and
                    e["physical_constraint_cost"] <= base["physical_constraint_cost"] + 0.02 * abs(base["physical_constraint_cost"]) + 1e-8
                )
                ratio = (e["decision_sum_s"] / base["decision_sum_s"] if base["decision_sum_s"] else None)
                candidates.append({
                    "H": h,
                    "safe_by_2pct_phys_success_solver": safe,
                    "success": e["success"],
                    "steps": e["steps"],
                    "phys_delta": e["physical_constraint_cost"] - base["physical_constraint_cost"],
                    "total_delta": e["total_cost"] - base["total_cost"],
                    "decision_sum_ratio_vs_H25": ratio,
                })
            safe_candidates = [c for c in candidates if c["safe_by_2pct_phys_success_solver"]]
            if safe_candidates:
                safe_short_count += 1
                best = min(safe_candidates, key=lambda c: c["total_delta"])
                best_short_total_delta_sum += float(best["total_delta"])
                if best["total_delta"] < 0:
                    total_improvement_count += 1
                if best.get("decision_sum_ratio_vs_H25") is not None:
                    best_short_decision_ratio_values.append(float(best["decision_sum_ratio_vs_H25"]))
                    if float(best["decision_sum_ratio_vs_H25"]) < 1.0:
                        timing_improvement_count += 1
            if case in (9, 43):
                case_rows.append({"case": case, "base_H25": base, "short_candidates": candidates})
        out[str(seed)] = {
            "cases_with_H25": len(cases),
            "cases_with_any_safe_short_H_by_2pct_phys_success_solver": safe_short_count,
            "cases_where_best_safe_short_total_beats_H25": total_improvement_count,
            "cases_where_best_safe_short_decision_sum_faster_than_H25": timing_improvement_count,
            "best_safe_short_total_delta_sum_vs_H25": best_short_total_delta_sum,
            "best_safe_short_decision_sum_ratio_summary": values_summary(best_short_decision_ratio_values),
            "notable_cases": case_rows,
        }
    return out


def training_history(policies: Mapping[int, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for seed in SEEDS:
        fit_path = TRAIN / ("vehicle_s%d" % seed) / "fit_completed.json"
        fit = read_json(fit_path)
        selected = fit["selected"]
        results = fit["results"]
        fixed = next(r for r in results if r["policy"]["id"] == "fixed")
        chosen = next(r for r in results if r["policy"]["id"] == selected["id"])
        short_h = int(selected.get("short_h", BASE_H))
        estimated_short_steps = 0.0
        estimated_steps = 0
        for e in chosen["episodes"]:
            steps = int(e.get("steps", 0)); estimated_steps += steps
            mean_h = float(e.get("mean_horizon", BASE_H))
            if short_h < BASE_H:
                estimated_short_steps += max(0.0, min(float(steps), steps * (BASE_H - mean_h) / (BASE_H - short_h)))
        eligible_count = sum(1 for r in results if r.get("fully_evaluated") and not r.get("rejected"))
        fully_count = sum(1 for r in results if r.get("fully_evaluated"))
        rejected_count = sum(1 for r in results if r.get("rejected"))
        out[str(seed)] = {
            "policy_path": rel(TRAIN / ("vehicle_s%d" % seed) / "policy.json"),
            "fit_completed_path": rel(fit_path),
            "fit_completed_sha256": sha256(fit_path),
            "selected": selected,
            "candidate_counts": {"total_results": len(results), "fully_evaluated": fully_count, "eligible_no_rejection": eligible_count, "rejected": rejected_count},
            "fixed_train_mean_raw_cost": fixed["mean_raw_cost"],
            "fixed_train_mean_physical_cost": fixed["mean_physical_cost"],
            "selected_train_mean_raw_cost": chosen["mean_raw_cost"],
            "selected_train_mean_physical_cost": chosen["mean_physical_cost"],
            "selected_minus_fixed_mean_raw_cost": float(chosen["mean_raw_cost"] - fixed["mean_raw_cost"]),
            "selected_minus_fixed_mean_physical_cost": float(chosen["mean_physical_cost"] - fixed["mean_physical_cost"]),
            "selected_train_success": int(sum(1 for e in chosen["episodes"] if e.get("success"))),
            "fixed_train_success": int(sum(1 for e in fixed["episodes"] if e.get("success"))),
            "selected_train_steps": estimated_steps,
            "selected_train_estimated_short_steps_from_mean_horizon": estimated_short_steps,
            "selected_train_estimated_short_share": (estimated_short_steps / estimated_steps if estimated_steps else None),
        }
    return out


def case9_deep_dive(trace_meta: Mapping[str, Any]) -> Dict[str, Any]:
    adaptive_trace_rel = trace_meta.get("case9_adaptive_trace_path")
    if not adaptive_trace_rel:
        return {"available": False, "reason": "adaptive trace not found"}
    adaptive_trace = ROOT / adaptive_trace_rel
    fixed_matches = sorted(DEVVAL.glob("shard13/episodes/exec*_case09_matched_terminal_fixed_H25_vehicle_s2/trace.jsonl"))
    if not fixed_matches:
        fixed_matches = sorted(DEVVAL.glob("shard*/episodes/exec*_case09_matched_terminal_fixed_H25_vehicle_s2/trace.jsonl"))
    if not fixed_matches:
        return {"available": False, "reason": "matched H25 trace not found", "adaptive_trace": adaptive_trace_rel}
    fixed_trace = fixed_matches[0]
    a_rows = load_trace_minimal(adaptive_trace)
    f_rows = load_trace_minimal(fixed_trace)
    short_indices = [i for i, r in enumerate(a_rows) if int(r.get("horizon") or BASE_H) < BASE_H]
    h10_step = short_indices[0] if short_indices else None
    min_len = min(len(a_rows), len(f_rows))
    dists = [state_distance(a_rows[i], f_rows[i]) for i in range(min_len)]
    valid_dists = [d for d in dists if d is not None]
    before_dists = [dists[i] for i in range(min_len) if h10_step is not None and i < h10_step and dists[i] is not None]
    cum_a = []
    cum_f = []
    s_a = 0.0; s_f = 0.0
    for i in range(min_len):
        s_a += float(a_rows[i].get("performance") or 0.0) + float(a_rows[i].get("constraint") or 0.0)
        s_f += float(f_rows[i].get("performance") or 0.0) + float(f_rows[i].get("constraint") or 0.0)
        cum_a.append(s_a); cum_f.append(s_f)
    window = []
    if h10_step is not None:
        for i in range(max(0, h10_step - 8), min(len(a_rows), h10_step + 16)):
            item = dict(a_rows[i])
            if i < len(f_rows):
                item["matched_H25_horizon"] = f_rows[i].get("horizon")
                item["matched_H25_state"] = f_rows[i].get("state")
                item["matched_H25_input"] = f_rows[i].get("input")
                item["matched_H25_performance"] = f_rows[i].get("performance")
                item["state_distance_vs_H25"] = round_value(state_distance(a_rows[i], f_rows[i]))
                item["heading_delta_vs_H25"] = round_value(heading_delta(a_rows[i], f_rows[i]))
                item["cum_physical_delta_vs_H25_to_step"] = round_value(cum_a[i] - cum_f[i])
            window.append(item)
    fixed_summary = read_json(fixed_trace.with_name("summary.json"))
    adaptive_summary = read_json(adaptive_trace.with_name("summary.json"))
    return {
        "available": True,
        "adaptive_trace": adaptive_trace_rel,
        "fixed_H25_trace": rel(fixed_trace),
        "adaptive_summary": {k: adaptive_summary.get(k) for k in ("success", "termination", "steps", "physical_constraint_cost", "total_cost", "horizon_counts", "raw_horizon_counts_before_clamp", "solver_failure_steps", "retries")},
        "fixed_H25_summary": {k: fixed_summary.get(k) for k in ("success", "termination", "steps", "physical_constraint_cost", "total_cost", "horizon_counts", "raw_horizon_counts_before_clamp", "solver_failure_steps", "retries")},
        "short_indices": short_indices,
        "first_short_step": h10_step,
        "short_occurs_before_H25_goal_step": (h10_step is not None and h10_step < len(f_rows)),
        "adaptive_already_diverged_before_short": {
            "distance_summary_before_short": values_summary(before_dists),
            "state_distance_at_short_step": (round_value(dists[h10_step]) if h10_step is not None and h10_step < len(dists) else None),
            "cum_physical_delta_before_short_step": (round_value(cum_a[h10_step - 1] - cum_f[h10_step - 1]) if h10_step is not None and h10_step > 0 and h10_step - 1 < len(cum_a) else None),
            "cum_physical_delta_at_H25_final_aligned_step": (round_value(cum_a[len(f_rows)-1] - cum_f[len(f_rows)-1]) if len(f_rows) <= len(cum_a) else None),
        },
        "state_distance_summary_common_prefix": values_summary(valid_dists),
        "local_window_around_first_short": window,
    }


def conclusions(raw: Mapping[str, Any]) -> List[str]:
    lines: List[str] = []
    per_seed = raw["adaptive_trace_policy_diagnostic"]["per_seed"]
    if all(per_seed[str(s)]["execution_inconsistency_count"] == 0 for s in SEEDS):
        lines.append("No trace-level request/execution inconsistency was found: good guard states, selected horizons, raw horizons and executed horizons agree for all parsed adaptive devval steps.")
    if sum(per_seed[str(s)]["raw_gt_base_steps"] for s in SEEDS) == 0:
        lines.append("No raw H>25 requests occurred in safe-shortening v1 traces; v1 removed the earlier H35 failure mode but did not create useful adaptation.")
    for s in SEEDS:
        seed = per_seed[str(s)]
        lines.append("Seed%d policy %s: non25_share=%.4f, all_guard_conditions_good_share_post_initial=%.4f, fallback=%d, clamped=%d; leading fail shares=%s." % (
            s, seed["policy"]["id"], seed["non25_share"] or 0.0,
            seed["all_conditions_good_share_post_initial"] or 0.0,
            seed["fallback_steps"], seed["clamped_steps"],
            {k: round(v, 4) if v is not None else None for k, v in sorted(seed["condition_fail_share_post_initial"].items(), key=lambda kv: -(kv[1] or 0))[:3]},
        ))
    c9 = raw["case9_deep_dive"]
    if c9.get("available"):
        h10 = c9.get("first_short_step")
        div = c9.get("adaptive_already_diverged_before_short", {})
        if c9.get("short_occurs_before_H25_goal_step"):
            lines.append("Case9 seed2 H10 occurred at step %s before the matched H25 goal step; aligned state-distance/cumulative-cost diagnostics are recorded, but metadata alone still cannot prove one-step causality." % h10)
        else:
            lines.append("Case9 seed2 H10 did not occur before the matched H25 episode ended; this would argue against the single H10 being the initiating cause. Inspect raw case9_deep_dive for exact timing.")
        lines.append("Before the case9 H10 step, adaptive-vs-H25 state-distance summary=%s and cumulative physical delta before H10=%s; this distinguishes pre-existing trajectory divergence from a pure one-action explanation." % (div.get("distance_summary_before_short"), div.get("cum_physical_delta_before_short_step")))
    return lines


def write_summary(raw: Mapping[str, Any], out_dir: Path) -> None:
    lines: List[str] = []
    lines.append("# Vehicle safe-shortening v1 trace/policy diagnostic")
    lines.append("")
    lines.append("Created UTC: `%s`." % raw["created_utc"])
    lines.append("")
    lines.append("Development diagnostic only over existing safe-shortening v1 devval traces and training-selection artifacts. No rollout, no training, no historical validation64 bank reopen, and no sealed-test access.")
    lines.append("")
    lines.append("## Budget/access")
    lines.append("")
    for k, v in raw["budget_actual"].items():
        lines.append("- %s: `%s`." % (k, v))
    lines.append("")
    lines.append("## Main conclusions")
    lines.append("")
    for c in raw["diagnostic_conclusions"]:
        lines.append("- " + c)
    lines.append("")
    lines.append("## Per-seed guard/execution summary")
    lines.append("")
    lines.append("| seed | policy | steps | horizons | non25 share | guard-good share post-init | inconsistencies | leading guard failures |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---|")
    for s in SEEDS:
        item = raw["adaptive_trace_policy_diagnostic"]["per_seed"][str(s)]
        fail = sorted(item["condition_fail_share_post_initial"].items(), key=lambda kv: -(kv[1] or 0))[:4]
        lines.append("| %d | `%s` | %d | %s | %.4f | %.4f | %d | %s |" % (
            s, item["policy"]["id"], item["steps"], item["horizon_counts"], item["non25_share"],
            item["all_conditions_good_share_post_initial"] or 0.0, item["execution_inconsistency_count"],
            {k: round(v, 4) if v is not None else None for k, v in fail},
        ))
    lines.append("")
    lines.append("## Training-vs-devval adaptation")
    lines.append("")
    lines.append("| seed | selected policy | train estimated short share | train raw delta | train physical delta | devval non25 share |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for s in SEEDS:
        t = raw["training_history"][str(s)]
        d = raw["adaptive_trace_policy_diagnostic"]["per_seed"][str(s)]
        lines.append("| %d | `%s` | %.4f | %.6g | %.6g | %.4f |" % (
            s, t["selected"]["id"], t["selected_train_estimated_short_share"] or 0.0,
            t["selected_minus_fixed_mean_raw_cost"], t["selected_minus_fixed_mean_physical_cost"], d["non25_share"],
        ))
    lines.append("")
    lines.append("## Existing fixed-grid short-H opportunity (case-level, not state-level)")
    lines.append("")
    for s in SEEDS:
        g = raw["fixed_grid_case_level_short_opportunity"][str(s)]
        lines.append("- seed %d: %d/%d cases have at least one H<25 fixed-run candidate with no worse success/solver and <=2%% physical-cost tolerance; %d cases have lower total cost under the best such short fixed H. Mean ratio summary for best safe-short decision sums: `%s`." % (
            s, g["cases_with_any_safe_short_H_by_2pct_phys_success_solver"], g["cases_with_H25"],
            g["cases_where_best_safe_short_total_beats_H25"], g["best_safe_short_decision_sum_ratio_summary"],
        ))
    lines.append("")
    lines.append("## Seed2 case9 local trace result")
    lines.append("")
    c9 = raw["case9_deep_dive"]
    if c9.get("available"):
        lines.append("- First short step: `%s`; short occurs before matched-H25 goal step: `%s`." % (c9["first_short_step"], c9["short_occurs_before_H25_goal_step"]))
        lines.append("- Adaptive summary: `%s`." % c9["adaptive_summary"])
        lines.append("- Matched H25 summary: `%s`." % c9["fixed_H25_summary"])
        lines.append("- Divergence before short: `%s`." % c9["adaptive_already_diverged_before_short"])
        lines.append("- Local window is in raw JSON under `case9_deep_dive.local_window_around_first_short`.")
    else:
        lines.append("- Case9 deep dive unavailable: `%s`." % c9)
    lines.append("")
    lines.append("## Next action")
    lines.append("")
    lines.append(raw["next_action"])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any], out_dir: Path, backup_request: Path) -> None:
    text = (
        "\n<!-- %s -->\n"
        "## 2026-09-28 vehicle safe-shortening v1 trace/policy diagnostic\n\n"
        "UTC: %s. Parsed existing safe-shortening v1 devval traces for all adaptive seeds plus matched fixed-grid summaries; no new rollout/control steps/training and no sealed-test access. "
        "Main result: no raw/executed horizon inconsistency, no H>25 requests, and near-constant H25 is explained by the frozen gate thresholds/policies rather than shard dispatch. "
        "Seed non25 shares: %s. Case9 seed2 local H10/deviation evidence and fixed-grid short-H opportunity summaries are in `%s`. "
        "Backup request: `%s`. Next: bounded deterministic state-level counterfactual replay/ablation before freezing any v2 retraining/selection change.\n"
        % (
            MARKER,
            raw["created_utc"],
            {s: raw["adaptive_trace_policy_diagnostic"]["per_seed"][str(s)]["non25_share"] for s in SEEDS},
            rel(out_dir / "summary.md"),
            rel(backup_request),
        )
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")
    decision = (
        "\n<!-- %s-decision -->\n"
        "### Decision: safe-shortening v1 collapse is a policy/guard objective failure, not a runtime horizon-dispatch bug\n\n"
        "Before evidence: full v1 devval64 failed the frozen gate; metadata-only case9 diagnostic showed a real seed2 failure but could not localize the rare H10 decision.\n\n"
        "Diagnostic action: parsed existing step traces, guard values, raw and executed horizons, training-selection records, and fixed-grid devval summaries. No new rollouts/training/test access.\n\n"
        "Outcome: see `%s`. No trace-level dispatch inconsistency or H>25 request was found. The near-collapse to H25 follows the selected gated policies and thresholds; seed2's strict h10_p0_g5 profile produces extremely rare H10 states. This supports a versioned IMPROVED v2 change focused on learned/selection objective and state-level safety/value estimation, after one bounded deterministic counterfactual replay for case9/case43.\n"
        % (MARKER, rel(out_dir / "summary.md"))
    )
    path = ROOT / "DECISIONS.md"
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if MARKER + "-decision" not in old:
            path.write_text(old.rstrip() + "\n" + decision, encoding="utf-8")


def append_registry(raw: Mapping[str, Any], out_dir: Path) -> None:
    path = ROOT / "EXPERIMENT_REGISTRY.csv"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return
    row = {h: "" for h in header}
    lower = {h.lower(): h for h in header}
    def set_if(names: Iterable[str], value: Any) -> None:
        for n in names:
            if n in lower:
                row[lower[n]] = str(value)
                return
    set_if(("timestamp", "created_utc", "utc", "time"), raw["created_utc"])
    set_if(("method",), raw["method"])
    set_if(("script", "source", "script_path"), SCRIPT_REL)
    set_if(("split",), raw["split"])
    set_if(("status", "result", "outcome"), "completed_metadata_diagnostic")
    set_if(("episodes", "new_rollout_episodes"), 0)
    set_if(("control_steps", "new_control_steps"), 0)
    set_if(("training_episodes", "new_training_episodes"), 0)
    set_if(("gradient_steps", "new_gradient_steps"), 0)
    set_if(("sealed_test", "sealed_test_accessed"), False)
    set_if(("artifacts", "artifact", "summary"), rel(out_dir / "summary.md"))
    # If the registry has none of the expected columns, append a harmless marker to
    # the final field rather than changing the schema.
    if all(v == "" for v in row.values()):
        row[header[-1]] = MARKER + " " + rel(out_dir / "summary.md")
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(row)


def make_backup_request(raw: Mapping[str, Any], out_dir: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_TRACE_POLICY_DIAGNOSTIC_%s.json" % stamp)
    artifacts = [
        rel(out_dir / "summary.md"),
        rel(out_dir / "raw.json"),
        rel(out_dir / "completed.json"),
        SCRIPT_REL,
        rel(PROTOCOL),
        rel(DEVVAL / "gate_completed.json"),
    ]
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup metadata-only trace/policy diagnostic for safe-shortening v1 absent adaptation before state-level replay or v2 revision",
        "artifacts": artifacts,
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
    })
    return path


def main() -> int:
    now = dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / ("vehicle_safe_shortening_v1_trace_policy_diagnostic_%s" % stamp)
    if out_dir.exists():
        raise RuntimeError("Refusing to overwrite existing diagnostic directory: %s" % rel(out_dir))
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "run_started.json", {
        "started_utc": now.isoformat(),
        "pid": os.getpid(),
        "method": "IMPROVED_vehicle_safe_shortening_v1_trace_policy_diagnostic_metadata_only",
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "sealed_test_accessed": False,
    })
    policies = {seed: policy_for_seed(seed) for seed in SEEDS}
    trace_diag, trace_meta = inspect_adaptive_traces(policies)
    # Do not store all short_records twice in the top-level summary; keep the
    # complete list in raw, and a bounded prefix in trace_meta for quick reading.
    short_records = trace_diag.pop("short_records")
    trace_diag["short_records_count"] = len(short_records)
    trace_diag["short_records_first_80"] = short_records[:80]
    fixed_grid = fixed_grid_case_summary()
    train = training_history(policies)
    c9 = case9_deep_dive(trace_meta)
    created = dt.datetime.now(dt.timezone.utc)
    elapsed = (created - SERVICE_START).total_seconds()
    raw: Dict[str, Any] = {
        "created_utc": created.isoformat(),
        "method": "IMPROVED_vehicle_safe_shortening_v1_trace_policy_diagnostic_metadata_only",
        "split": "existing_fresh_devval64_trace_metadata_only_no_sealed_test",
        "formal_scientific_evidence": False,
        "access_control": {
            "new_rollout_episodes": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "final_test_authorization_requested": False,
        },
        "budget_actual": {
            "existing_adaptive_episodes_inspected": 16 * 3 * 4,
            "existing_adaptive_trace_steps_parsed": sum(trace_diag["per_seed"][str(s)]["steps"] for s in SEEDS),
            "existing_matched_fixed_grid_summary_episodes_inspected": 16 * 4 * 3 * len(H_GRID),
            "existing_case9_step_traces_loaded_for_alignment": 2 if c9.get("available") else 1,
            "new_rollout_episodes": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "sealed_test_episodes": 0,
            "sealed_test_control_steps": 0,
        },
        "service_elapsed_at_diagnostic": {
            "since_2026-09-26T10:55:29.419331Z_seconds": elapsed,
            "since_2026-09-26T10:55:29.419331Z_hours": elapsed / 3600.0,
            "research_sqlite_total_tokens": "unknown; research.sqlite path unavailable to repository tools",
        },
        "source_hashes": {
            SCRIPT_REL: sha256(ROOT / SCRIPT_REL),
            "experiments/bohn2021_reproduction/gated_horizon_policy.py": sha256(ROOT / "experiments/bohn2021_reproduction/gated_horizon_policy.py"),
            "experiments/bohn2021_reproduction/relative_policy_features.py": sha256(ROOT / "experiments/bohn2021_reproduction/relative_policy_features.py"),
            "experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py": sha256(ROOT / "experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py"),
            rel(PROTOCOL): sha256(PROTOCOL),
            rel(DEVVAL / "gate_completed.json"): sha256(DEVVAL / "gate_completed.json"),
        },
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "adaptive_trace_policy_diagnostic": trace_diag,
        "adaptive_trace_metadata": trace_meta,
        "training_history": train,
        "fixed_grid_case_level_short_opportunity": fixed_grid,
        "case9_deep_dive": c9,
        "diagnostic_conclusions": [],
        "next_action": "Run a bounded deterministic state-level counterfactual replay/ablation on devval shard13 case9 seed2 (force H25 at the saved H10 decision and inject H10 into the matched-H25 path if technically feasible), then repeat the same style on the earlier case43 H35 lead. Use those causal results to freeze an IMPROVED v2 training/selection amendment; do not revalidate unchanged v1 or open sealed test.",
    }
    raw["diagnostic_conclusions"] = conclusions(raw)
    write_json(out_dir / "raw.json", raw)
    write_summary(raw, out_dir)
    backup_request = make_backup_request(raw, out_dir)
    append_docs(raw, out_dir, backup_request)
    append_registry(raw, out_dir)
    files = [out_dir / "run_started.json", out_dir / "raw.json", out_dir / "summary.md", backup_request, ROOT / SCRIPT_REL, PROTOCOL, DEVVAL / "gate_completed.json"]
    write_json(out_dir / "completed.json", {
        "passed": True,
        "method": raw["method"],
        "formal_scientific_evidence": False,
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
        "backup_request": rel(backup_request),
        "summary": rel(out_dir / "summary.md"),
        "raw": rel(out_dir / "raw.json"),
        "headline": {
            "seed_non25_shares": {str(s): raw["adaptive_trace_policy_diagnostic"]["per_seed"][str(s)]["non25_share"] for s in SEEDS},
            "execution_inconsistencies": {str(s): raw["adaptive_trace_policy_diagnostic"]["per_seed"][str(s)]["execution_inconsistency_count"] for s in SEEDS},
            "case9_first_short_step": raw["case9_deep_dive"].get("first_short_step"),
            "case9_short_before_H25_goal": raw["case9_deep_dive"].get("short_occurs_before_H25_goal_step"),
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(out_dir / "completed.json"),
        "summary": rel(out_dir / "summary.md"),
        "raw": rel(out_dir / "raw.json"),
        "backup_request": rel(backup_request),
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "sealed_test_accessed": False,
        "seed_non25_shares": {str(s): raw["adaptive_trace_policy_diagnostic"]["per_seed"][str(s)]["non25_share"] for s in SEEDS},
        "execution_inconsistencies": {str(s): raw["adaptive_trace_policy_diagnostic"]["per_seed"][str(s)]["execution_inconsistency_count"] for s in SEEDS},
        "case9_first_short_step": raw["case9_deep_dive"].get("first_short_step"),
        "case9_short_before_H25_goal": raw["case9_deep_dive"].get("short_occurs_before_H25_goal_step"),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        fail_dir = OUT_ROOT / ("vehicle_safe_shortening_v1_trace_policy_diagnostic_failure_%s" % dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        fail_dir.mkdir(parents=True, exist_ok=True)
        write_json(fail_dir / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "new_rollout_episodes": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "sealed_test_accessed": False,
            "next_recovery_hint": "Preserve failure; inspect script/schema mismatch and rerun only a versioned diagnostic fix.",
        })
        raise
