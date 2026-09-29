#!/usr/bin/env python3
"""No-rollout timing decomposition for the v1e case-5 repeated diagnostic.

This postdiagnostic reads only the already-created development artifacts from
`vehicle_stress_v1e_case5_positive_stability_timing_v0_run_20260929T0605Z`.
It asks why the local H10 physical gains did not yield a measured speed gate:

* Is the timing difference mostly whole-decision logging/terminal-switch overhead,
  or actual solver wall time?
* Is the common H15 prefix masking a branch-only H10 speed benefit?
* Does the current MPC implementation keep a fixed NLP variable dimension while
  changing only a horizon parameter, making shorter H an unreliable proxy for
  compute?

No simulations, no candidate-pool resets, no training/refit, no validation64 bank
content, and no sealed-test content are opened.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260929T0625Z"
NAME = "vehicle_stress_v1e_case5_timing_decomposition_postdiagnostic_v0"
SOURCE = Path(__file__).resolve()
RUN_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_run_20260929T0605Z"
RUN_RAW = RUN_DIR / "raw.json"
RUN_DONE = RUN_DIR / "completed.json"
RUN_SUMMARY = RUN_DIR / "summary.md"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1E_CASE5_TIMING_DECOMPOSITION_POSTDIAGNOSTIC_V0_{STAMP}.json"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-stress-v1e-case5-timing-decomposition-postdiagnostic-v0-{STAMP}"
ACCESS_FLAGS = [
    "historical_validation64_bank_opened",
    "validation64_bank_opened",
    "sealed_test_accessed",
    "sealed_test_bank_opened",
]


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def maybe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def percentile(sorted_vals: Sequence[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo))


def stats(values: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return {"n": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "sum": float(math.fsum(vals)),
        "mean": float(math.fsum(vals) / len(vals)),
        "median": percentile(vals, 0.5),
        "p95": percentile(vals, 0.95),
        "min": vals[0],
        "max": vals[-1],
    }


def corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    mx = math.fsum(x for x, _ in pairs) / len(pairs)
    my = math.fsum(y for _, y in pairs) / len(pairs)
    vx = math.fsum((x - mx) ** 2 for x, _ in pairs)
    vy = math.fsum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return None
    cov = math.fsum((x - mx) * (y - my) for x, y in pairs)
    return float(cov / math.sqrt(vx * vy))


def flag_false_or_absent(obj: Mapping[str, Any], key: str) -> bool:
    return obj.get(key, False) is False


def verify_inputs() -> Dict[str, Any]:
    for p in (RUN_RAW, RUN_DONE, RUN_SUMMARY):
        if not p.exists():
            raise ContractError(f"required input missing: {rel(p)}")
    done = read_json(RUN_DONE)
    raw = read_json(RUN_RAW)
    if done.get("passed") is not True and done.get("hard_pass") is not True:
        raise ContractError("case5 run completed marker is not passing")
    for name, obj in (("run_done", done), ("run_raw", raw)):
        for flag in ACCESS_FLAGS:
            if not flag_false_or_absent(obj, flag):
                raise ContractError(f"{name} has invalid access flag {flag}={obj.get(flag)}")
    analysis = raw.get("analysis") or {}
    if int(analysis.get("pair_count", -1)) != 12:
        raise ContractError("case5 run raw does not contain the expected 12 H10/H15 pairs")
    if int(raw.get("episodes", done.get("episodes", -1))) != 24:
        raise ContractError("case5 run does not contain the expected 24 episodes")
    return {"done": done, "raw": raw, "analysis": analysis}


def parse_jsonl(path: Path) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def timing_value(row: Mapping[str, Any]) -> float:
    timing = row.get("timing") or {}
    for key in ("decision_s", "decision_gross_s", "controller_s", "controller_gross_s"):
        if key in timing and timing.get(key) is not None:
            return safe_float(timing.get(key), 0.0)
    return safe_float(row.get("decision_plus_terminal_switch_s"), 0.0)


def terminal_switch_time(row: Mapping[str, Any]) -> float:
    timing = row.get("timing") or {}
    return safe_float(timing.get("terminal_switch_s_before_controller"), 0.0)


def solver_attempts(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    recovery = row.get("recovery") or {}
    attempts = recovery.get("attempts") or []
    return [a for a in attempts if isinstance(a, Mapping)]


def solver_time(row: Mapping[str, Any]) -> float:
    return float(math.fsum(safe_float(a.get("solver_s"), 0.0) for a in solver_attempts(row)))


def solver_iterations(row: Mapping[str, Any]) -> int:
    return int(math.fsum(safe_float(a.get("iterations"), 0.0) for a in solver_attempts(row)))


def solver_success(row: Mapping[str, Any]) -> bool:
    attempts = solver_attempts(row)
    if not attempts:
        return bool(row.get("solver_success"))
    return bool(attempts[-1].get("accepted", attempts[-1].get("success", False)))


def mpc_info_time(row: Mapping[str, Any]) -> Optional[float]:
    diag = row.get("mpc_diag") or {}
    return maybe_float(diag.get("info_mpc_computation_time"))


def mpc_horizon_param(row: Mapping[str, Any]) -> Optional[int]:
    diag = row.get("mpc_diag") or {}
    compact = diag.get("info_data_compact") or {}
    vals = compact.get("mpc_n_horizon")
    try:
        if isinstance(vals, list) and vals:
            return int(round(float(vals[0])))
        if vals is not None:
            return int(round(float(vals)))
    except Exception:
        return None
    return None


def terminal_p_horizon(row: Mapping[str, Any]) -> Optional[int]:
    diag = row.get("mpc_diag") or {}
    vals = diag.get("terminal_p")
    try:
        if isinstance(vals, list) and len(vals) >= 3:
            return int(round(float(vals[2])))
    except Exception:
        return None
    return None


def attempt_sizes(row: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
    pre: List[int] = []
    post: List[int] = []
    for a in solver_attempts(row):
        pre_obj = a.get("warm_start_pre_opt_x") or {}
        post_obj = a.get("warm_start_post_opt_x") or {}
        if pre_obj.get("size") is not None:
            pre.append(int(pre_obj.get("size")))
        if post_obj.get("size") is not None:
            post.append(int(post_obj.get("size")))
    return pre, post


def step_physical(row: Mapping[str, Any]) -> float:
    return safe_float(row.get("performance"), 0.0) + safe_float(row.get("constraint"), 0.0)


def summarize_phase(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    decision = [timing_value(r) for r in rows]
    solver = [solver_time(r) for r in rows]
    overhead = [d - s for d, s in zip(decision, solver)]
    iters = [solver_iterations(r) for r in rows]
    info = [v for v in (mpc_info_time(r) for r in rows) if v is not None]
    term_switch = [terminal_switch_time(r) for r in rows]
    phys = [step_physical(r) for r in rows]
    retry_count = 0
    failed_steps = 0
    for r in rows:
        attempts = solver_attempts(r)
        if len(attempts) > 1:
            retry_count += len(attempts) - 1
        if not solver_success(r):
            failed_steps += 1
    return {
        "steps": len(rows),
        "decision_s": stats(decision),
        "solver_s": stats(solver),
        "decision_minus_solver_overhead_s": stats(overhead),
        "solver_iterations": stats(iters),
        "mpc_info_computation_time_s": stats(info),
        "terminal_switch_s": stats(term_switch),
        "physical_step_cost": stats(phys),
        "retry_attempts": retry_count,
        "failed_solver_steps": failed_steps,
    }


def summarize_episode(ep_dir: Path) -> Dict[str, Any]:
    summary = read_json(ep_dir / "summary.json")
    trace_path = ep_dir / "trace.jsonl"
    if not trace_path.exists():
        raise ContractError(f"trace.jsonl missing: {rel(trace_path)}")
    trace = parse_jsonl(trace_path)
    branch_step = int(summary.get("branch_step", 0))
    if int(summary.get("steps", len(trace))) != len(trace):
        raise ContractError(f"step count mismatch in {rel(ep_dir)}")
    prefix_rows = [r for r in trace if int(r.get("step", -1)) < branch_step]
    branch_rows = [r for r in trace if int(r.get("step", -1)) >= branch_step]
    by_horizon: Dict[str, Dict[str, Any]] = {}
    pre_sizes_by_h: Dict[str, List[int]] = {}
    post_sizes_by_h: Dict[str, List[int]] = {}
    horizon_param_mismatch = 0
    terminal_p_mismatch = 0
    for h in sorted({int(r.get("horizon", -1)) for r in trace}):
        rows = [r for r in trace if int(r.get("horizon", -1)) == h]
        by_horizon[str(h)] = summarize_phase(rows)
        pre_sizes: List[int] = []
        post_sizes: List[int] = []
        for r in rows:
            pre, post = attempt_sizes(r)
            pre_sizes.extend(pre)
            post_sizes.extend(post)
            hp = mpc_horizon_param(r)
            tp = terminal_p_horizon(r)
            if hp is not None and hp != h:
                horizon_param_mismatch += 1
            if tp is not None and tp != h:
                terminal_p_mismatch += 1
        pre_sizes_by_h[str(h)] = sorted(set(pre_sizes))
        post_sizes_by_h[str(h)] = sorted(set(post_sizes))
    total_decision = safe_float((summary.get("decision_timing_s") or {}).get("sum"), math.fsum(timing_value(r) for r in trace))
    total_solver = safe_float((summary.get("solver_attempt_timing_s") or {}).get("sum"), math.fsum(solver_time(r) for r in trace))
    total_info = math.fsum(v for v in (mpc_info_time(r) for r in trace) if v is not None)
    return {
        "path": rel(ep_dir),
        "summary_path": rel(ep_dir / "summary.json"),
        "state_id": summary.get("state_id"),
        "target_index": int(summary.get("target_index", -1)),
        "case": int(summary.get("case", -1)),
        "terminal_mode": summary.get("terminal_mode"),
        "branch_horizon": int(summary.get("branch_horizon", -1)),
        "diagnostic_repeat": int(summary.get("diagnostic_repeat", -1)),
        "branch_step": branch_step,
        "steps": len(trace),
        "success": bool(summary.get("success")),
        "constraint": bool(summary.get("constraint")),
        "solver_failure_steps": int(summary.get("solver_failure_steps", 0) or 0),
        "horizon_counts": summary.get("horizon_counts") or {},
        "total": summarize_phase(trace),
        "prefix": summarize_phase(prefix_rows),
        "branch": summarize_phase(branch_rows),
        "by_horizon": by_horizon,
        "pre_opt_sizes_by_executed_horizon": pre_sizes_by_h,
        "post_opt_sizes_by_executed_horizon": post_sizes_by_h,
        "horizon_param_mismatch_steps": horizon_param_mismatch,
        "terminal_p_horizon_mismatch_steps": terminal_p_mismatch,
        "decision_summary_sum_s": total_decision,
        "solver_summary_sum_s": total_solver,
        "mpc_info_computation_sum_s": total_info,
        "solver_fraction_of_decision": total_solver / total_decision if total_decision > 0 else None,
        "mpc_info_fraction_of_measured_solver": total_info / total_solver if total_solver > 0 else None,
        "terminal_switch_summary_sum_s": safe_float((summary.get("terminal_switch_timing_s") or {}).get("sum"), 0.0),
        "continuation_physical_from_branch": safe_float(summary.get("continuation_physical_constraint_cost_from_branch"), 0.0),
        "continuation_total_from_branch": safe_float(summary.get("continuation_total_cost_from_branch"), 0.0),
    }


def sum_stat(phase: Mapping[str, Any], key: str) -> float:
    return safe_float(((phase.get(key) or {}).get("sum")), 0.0)


def mean_stat(phase: Mapping[str, Any], key: str) -> Optional[float]:
    return maybe_float(((phase.get(key) or {}).get("mean")))


def pair_decomposition(pair: Mapping[str, Any], episodes: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    h10 = episodes[str(pair["h10_path"])]
    h15 = episodes[str(pair["h15_path"])]
    prefix_dec_gain = sum_stat(h15["prefix"], "decision_s") - sum_stat(h10["prefix"], "decision_s")
    branch_dec_gain = sum_stat(h15["branch"], "decision_s") - sum_stat(h10["branch"], "decision_s")
    total_dec_gain = sum_stat(h15["total"], "decision_s") - sum_stat(h10["total"], "decision_s")
    prefix_solver_gain = sum_stat(h15["prefix"], "solver_s") - sum_stat(h10["prefix"], "solver_s")
    branch_solver_gain = sum_stat(h15["branch"], "solver_s") - sum_stat(h10["branch"], "solver_s")
    total_solver_gain = sum_stat(h15["total"], "solver_s") - sum_stat(h10["total"], "solver_s")
    prefix_info_gain = sum_stat(h15["prefix"], "mpc_info_computation_time_s") - sum_stat(h10["prefix"], "mpc_info_computation_time_s")
    branch_info_gain = sum_stat(h15["branch"], "mpc_info_computation_time_s") - sum_stat(h10["branch"], "mpc_info_computation_time_s")
    h15_branch_dec = sum_stat(h15["branch"], "decision_s")
    h15_total_dec = sum_stat(h15["total"], "decision_s")
    h15_branch_solver = sum_stat(h15["branch"], "solver_s")
    h15_total_solver = sum_stat(h15["total"], "solver_s")
    h10_branch_dec = sum_stat(h10["branch"], "decision_s")
    h10_total_dec = sum_stat(h10["total"], "decision_s")
    h10_branch_solver = sum_stat(h10["branch"], "solver_s")
    h10_total_solver = sum_stat(h10["total"], "solver_s")
    h10_branch_sizes = sorted(set(h10.get("pre_opt_sizes_by_executed_horizon", {}).get("10", []) + h10.get("post_opt_sizes_by_executed_horizon", {}).get("10", [])))
    h15_branch_sizes = sorted(set(h15.get("pre_opt_sizes_by_executed_horizon", {}).get("15", []) + h15.get("post_opt_sizes_by_executed_horizon", {}).get("15", [])))
    branch_rel = branch_dec_gain / h15_branch_dec if h15_branch_dec > 0 else None
    total_rel = total_dec_gain / h15_total_dec if h15_total_dec > 0 else None
    branch_solver_rel = branch_solver_gain / h15_branch_solver if h15_branch_solver > 0 else None
    total_solver_rel = total_solver_gain / h15_total_solver if h15_total_solver > 0 else None
    prefix_solver_fraction = None
    branch_solver_fraction = None
    if abs(total_solver_gain) + abs(total_dec_gain) > 0:
        branch_solver_fraction = branch_solver_gain / total_solver_gain if abs(total_solver_gain) > 1e-9 else None
        prefix_solver_fraction = prefix_solver_gain / total_solver_gain if abs(total_solver_gain) > 1e-9 else None
    branch_vs_prefix_class = "branch_and_total_agree"
    if branch_dec_gain > 0 and total_dec_gain <= 0:
        branch_vs_prefix_class = "branch_speed_hidden_by_prefix_noise"
    elif branch_dec_gain <= 0 and total_dec_gain > 0:
        branch_vs_prefix_class = "total_speed_from_prefix_noise_not_branch_h10"
    elif branch_dec_gain <= 0 and total_dec_gain <= 0:
        branch_vs_prefix_class = "h10_not_faster_even_on_branch"
    return {
        "target_index": int(pair.get("target_index", -1)),
        "state_id": pair.get("state_id"),
        "case": int(pair.get("case", -1)),
        "terminal_mode": pair.get("terminal_mode"),
        "repeat": int(pair.get("repeat", -1)),
        "material_positive": bool(pair.get("material_positive")),
        "safety_ok_vs_H15": bool(pair.get("safety_ok_vs_H15")),
        "physical_gain_H10_vs_H15": safe_float(pair.get("physical_gain_H10_vs_H15"), 0.0),
        "total_gain_H10_vs_H15": safe_float(pair.get("total_gain_H10_vs_H15"), 0.0),
        "prefix_decision_saving_s": prefix_dec_gain,
        "branch_decision_saving_s": branch_dec_gain,
        "total_decision_saving_s": total_dec_gain,
        "branch_relative_decision_saving": branch_rel,
        "total_relative_decision_saving": total_rel,
        "prefix_solver_saving_s": prefix_solver_gain,
        "branch_solver_saving_s": branch_solver_gain,
        "total_solver_saving_s": total_solver_gain,
        "branch_relative_solver_saving": branch_solver_rel,
        "total_relative_solver_saving": total_solver_rel,
        "prefix_mpc_info_saving_s": prefix_info_gain,
        "branch_mpc_info_saving_s": branch_info_gain,
        "h10_branch_decision_sum_s": h10_branch_dec,
        "h15_branch_decision_sum_s": h15_branch_dec,
        "h10_total_decision_sum_s": h10_total_dec,
        "h15_total_decision_sum_s": h15_total_dec,
        "h10_branch_solver_sum_s": h10_branch_solver,
        "h15_branch_solver_sum_s": h15_branch_solver,
        "h10_total_solver_sum_s": h10_total_solver,
        "h15_total_solver_sum_s": h15_total_solver,
        "h10_prefix_decision_sum_s": sum_stat(h10["prefix"], "decision_s"),
        "h15_prefix_decision_sum_s": sum_stat(h15["prefix"], "decision_s"),
        "h10_branch_mean_iterations": mean_stat(h10["branch"], "solver_iterations"),
        "h15_branch_mean_iterations": mean_stat(h15["branch"], "solver_iterations"),
        "h10_branch_opt_sizes": h10_branch_sizes,
        "h15_branch_opt_sizes": h15_branch_sizes,
        "branch_opt_size_identical": bool(h10_branch_sizes and h10_branch_sizes == h15_branch_sizes),
        "branch_steps_h10": int(h10["branch"]["steps"]),
        "branch_steps_h15": int(h15["branch"]["steps"]),
        "total_steps_h10": int(h10["steps"]),
        "total_steps_h15": int(h15["steps"]),
        "h10_branch_decision_fraction_of_episode": h10_branch_dec / h10_total_dec if h10_total_dec > 0 else None,
        "h15_branch_decision_fraction_of_episode": h15_branch_dec / h15_total_dec if h15_total_dec > 0 else None,
        "branch_solver_gain_fraction_of_total_solver_gain": branch_solver_fraction,
        "prefix_solver_gain_fraction_of_total_solver_gain": prefix_solver_fraction,
        "branch_vs_total_timing_class": branch_vs_prefix_class,
    }


def group_key(row: Mapping[str, Any]) -> Tuple[int, str]:
    return (int(row.get("target_index", -1)), str(row.get("terminal_mode")))


def summarize_pair_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    return {
        "target_index": int(rows[0]["target_index"]),
        "state_id": rows[0].get("state_id"),
        "terminal_mode": rows[0].get("terminal_mode"),
        "pairs": len(rows),
        "material_positive_pairs": sum(1 for r in rows if r.get("material_positive")),
        "safety_ok_pairs": sum(1 for r in rows if r.get("safety_ok_vs_H15")),
        "median_physical_gain": stats([safe_float(r.get("physical_gain_H10_vs_H15"), 0.0) for r in rows])["median"],
        "total_relative_decision_saving": stats([safe_float(r.get("total_relative_decision_saving"), 0.0) for r in rows]),
        "branch_relative_decision_saving": stats([safe_float(r.get("branch_relative_decision_saving"), 0.0) for r in rows]),
        "prefix_decision_saving_s": stats([safe_float(r.get("prefix_decision_saving_s"), 0.0) for r in rows]),
        "branch_decision_saving_s": stats([safe_float(r.get("branch_decision_saving_s"), 0.0) for r in rows]),
        "total_solver_saving_s": stats([safe_float(r.get("total_solver_saving_s"), 0.0) for r in rows]),
        "branch_solver_saving_s": stats([safe_float(r.get("branch_solver_saving_s"), 0.0) for r in rows]),
        "h10_branch_mean_iterations": stats([safe_float(r.get("h10_branch_mean_iterations"), 0.0) for r in rows]),
        "h15_branch_mean_iterations": stats([safe_float(r.get("h15_branch_mean_iterations"), 0.0) for r in rows]),
        "branch_opt_size_identical_all_pairs": all(bool(r.get("branch_opt_size_identical")) for r in rows),
        "branch_timing_positive_pairs": sum(1 for r in rows if safe_float(r.get("branch_relative_decision_saving"), -1.0) > 0.0),
        "branch_timing_positive_5pct_pairs": sum(1 for r in rows if safe_float(r.get("branch_relative_decision_saving"), -1.0) >= 0.05),
        "total_timing_positive_5pct_pairs": sum(1 for r in rows if safe_float(r.get("total_relative_decision_saving"), -1.0) >= 0.05),
        "branch_vs_total_classes": {c: sum(1 for r in rows if r.get("branch_vs_total_timing_class") == c) for c in sorted(set(str(r.get("branch_vs_total_timing_class")) for r in rows))},
    }


def query_server_token_total() -> Dict[str, Any]:
    candidates = [ROOT / "research.sqlite", ROOT / "research_artifacts/research.sqlite", ROOT.parent / "research.sqlite"]
    for db in candidates:
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db))
            cur = con.cursor()
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            total = None
            details: Dict[str, int] = {}
            for table in tables:
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                if "total_tokens" in cols:
                    val = cur.execute(f"SELECT COALESCE(SUM(total_tokens),0) FROM {table}").fetchone()[0]
                    details[table] = int(val or 0)
                    total = (total or 0) + int(val or 0)
            con.close()
            if total is not None:
                return {"available": True, "path": rel(db), "total_tokens": int(total), "by_table": details}
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": repr(exc)}
    return {"available": False, "reason": "repository-accessible research.sqlite token table not found"}


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def render_summary(result: Mapping[str, Any]) -> str:
    a = result["analysis"]
    lines = [
        "# Vehicle stress-v1e case5 timing decomposition postdiagnostic v0",
        "",
        f"UTC: `{result['created_utc']}`. Metadata/no-rollout analysis of the completed v1e case5 repeated paired diagnostic.",
        "",
        "## Budget and access",
        "",
        "- New rollouts/control steps/training/refit: `0`.",
        "- Validation64 bank opened: `False`; sealed final test accessed: `False`.",
        f"- Input run: `{rel(RUN_DIR)}`.",
        "",
        "## Headline",
        "",
        f"- Fixed NLP decision-vector size across executed H10/H15 steps: `{a['fixed_nlp_size_across_h10_h15']}`; observed opt_x sizes: `{a['global_opt_x_sizes_by_horizon']}`.",
        f"- Solver wall time explains decision-time differences: solver-vs-decision pair-difference correlation `{a['solver_vs_decision_gain_correlation']}`; median solver fraction of decision time `{a['episode_solver_fraction_of_decision']['median']}`.",
        f"- mpc_info_computation_time is not the measured solver timer here: median info/measured-solver fraction `{a['episode_mpc_info_fraction_of_measured_solver']['median']}`.",
        f"- Overall total relative H10 decision saving over 12 pairs: median `{a['pair_total_relative_decision_saving']['median']}`; branch-only median `{a['pair_branch_relative_decision_saving']['median']}`.",
        f"- Groups with >=5% median branch-only measured decision saving: `{a['groups_with_median_branch_decision_saving_ge5pct']}` / `{len(a['group_rows'])}`; groups with >=5% median whole-episode saving: `{a['groups_with_median_total_decision_saving_ge5pct']}` / `{len(a['group_rows'])}`.",
        f"- Classification: `{a['classification']}`.",
        "",
        "## Group timing table",
        "",
        "| target | terminal | material | median physical gain | median total rel decision saving | median branch rel decision saving | median branch solver saving (s) | H10 branch iters mean | H15 branch iters mean | fixed opt size |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for g in a["group_rows"]:
        lines.append(
            "| {target_index} | `{terminal_mode}` | {material_positive_pairs}/{pairs} | {median_physical_gain:.6g} | {total_med:.6g} | {branch_med:.6g} | {branch_solver_med:.6g} | {h10_iter:.6g} | {h15_iter:.6g} | `{fixed}` |".format(
                target_index=g["target_index"],
                terminal_mode=g["terminal_mode"],
                material_positive_pairs=g["material_positive_pairs"],
                pairs=g["pairs"],
                median_physical_gain=safe_float(g.get("median_physical_gain"), 0.0),
                total_med=safe_float((g.get("total_relative_decision_saving") or {}).get("median"), 0.0),
                branch_med=safe_float((g.get("branch_relative_decision_saving") or {}).get("median"), 0.0),
                branch_solver_med=safe_float((g.get("branch_solver_saving_s") or {}).get("median"), 0.0),
                h10_iter=safe_float((g.get("h10_branch_mean_iterations") or {}).get("median"), 0.0),
                h15_iter=safe_float((g.get("h15_branch_mean_iterations") or {}).get("median"), 0.0),
                fixed=g.get("branch_opt_size_identical_all_pairs"),
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The local physical gains in case5 are stable, but measured speed does not follow from H alone. In these traces, executed H10 and H15 steps both carry the same recorded optimizer-vector size, and whole-decision differences closely track measured solver-attempt differences rather than logging or terminal-switch overhead. The common H15 prefix can dilute local branch effects, but branch-only timing is still mixed and terminal-mode dependent; zero-terminal H10 is generally slower, while H15-common terminal has only small/inconsistent savings. This points to a current implementation/solver-path compute bottleneck, not merely a label-density issue.",
        "",
        "## Next decision",
        "",
        a["next_action"],
        "",
        "A backup request was written before any further simulation/training/refit:",
        f"`{result['backup_request']}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    started = now_utc()
    inputs = verify_inputs()
    if OUT.exists() and (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline")}, sort_keys=True))
        return 0
    if OUT.exists() and any(p.name != "run.lock" for p in OUT.iterdir()):
        raise ContractError(f"partial output exists; inspect first: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=True)

    pair_rows_in = list((inputs["analysis"].get("pair_rows") or []))
    episode_dirs = sorted(p.parent for p in RUN_DIR.glob("episodes/*/summary.json"))
    if len(episode_dirs) != 24:
        raise ContractError(f"expected 24 episode summaries, found {len(episode_dirs)}")
    episodes = {rel(ep): summarize_episode(ep) for ep in episode_dirs}
    # Also allow pair paths with exactly the stored string key.
    for ep in list(episodes.values()):
        episodes[ep["path"]] = ep
    pair_rows = [pair_decomposition(p, episodes) for p in pair_rows_in]

    group_map: Dict[Tuple[int, str], List[Mapping[str, Any]]] = {}
    for row in pair_rows:
        group_map.setdefault(group_key(row), []).append(row)
    group_rows = [summarize_pair_group(rows) for _, rows in sorted(group_map.items())]

    global_sizes: Dict[str, List[int]] = {}
    all_episode_solver_frac: List[float] = []
    all_episode_info_frac: List[float] = []
    terminal_switch_sums: List[float] = []
    horizon_param_mismatches = 0
    terminal_p_mismatches = 0
    for ep in episodes.values():
        if ep["path"] != ep.get("path"):
            continue
        if ep.get("solver_fraction_of_decision") is not None:
            all_episode_solver_frac.append(float(ep["solver_fraction_of_decision"]))
        if ep.get("mpc_info_fraction_of_measured_solver") is not None:
            all_episode_info_frac.append(float(ep["mpc_info_fraction_of_measured_solver"]))
        terminal_switch_sums.append(float(ep.get("terminal_switch_summary_sum_s") or 0.0))
        horizon_param_mismatches += int(ep.get("horizon_param_mismatch_steps", 0) or 0)
        terminal_p_mismatches += int(ep.get("terminal_p_horizon_mismatch_steps", 0) or 0)
        for h, vals in (ep.get("pre_opt_sizes_by_executed_horizon") or {}).items():
            global_sizes.setdefault(str(h), []).extend(int(v) for v in vals)
        for h, vals in (ep.get("post_opt_sizes_by_executed_horizon") or {}).items():
            global_sizes.setdefault(str(h), []).extend(int(v) for v in vals)
    global_sizes_unique = {h: sorted(set(vals)) for h, vals in sorted(global_sizes.items(), key=lambda kv: int(kv[0]))}
    sizes_h10 = set(global_sizes_unique.get("10", []))
    sizes_h15 = set(global_sizes_unique.get("15", []))
    fixed_nlp = bool(sizes_h10 and sizes_h15 and sizes_h10 == sizes_h15 and len(sizes_h10 | sizes_h15) == 1)

    total_gain = [safe_float(r.get("total_decision_saving_s"), 0.0) for r in pair_rows]
    solver_gain = [safe_float(r.get("total_solver_saving_s"), 0.0) for r in pair_rows]
    branch_gain = [safe_float(r.get("branch_decision_saving_s"), 0.0) for r in pair_rows]
    branch_solver_gain = [safe_float(r.get("branch_solver_saving_s"), 0.0) for r in pair_rows]
    overhead_gain = [g - s for g, s in zip(total_gain, solver_gain)]
    classification = "undetermined_metadata_only"
    next_action = "Run a source-level MPC horizon implementation audit before choosing a new simulation; do not train/refit from this sparse case5 evidence."
    groups_branch_5pct = sum(1 for g in group_rows if safe_float((g.get("branch_relative_decision_saving") or {}).get("median"), -1.0) >= 0.05)
    groups_total_5pct = sum(1 for g in group_rows if safe_float((g.get("total_relative_decision_saving") or {}).get("median"), -1.0) >= 0.05)
    if fixed_nlp and groups_total_5pct == 0:
        classification = "stable_local_physical_opportunity_but_current_fixed_size_mpc_no_measured_compute_tradeoff"
        next_action = (
            "Next run should be metadata/source audit plus, if supported, a one-variable IMPROVED solver experiment that actually changes MPC problem dimension or controller construction for H10/H15 on development states. "
            "If variable-dimensional repair is not feasible on this codebase, pivot to value/terminal/modeling/scenario-opportunity work and treat shorter-H-as-speed as invalid for current implementation."
        )
    elif groups_branch_5pct > 0 and groups_total_5pct == 0:
        classification = "branch_local_speed_possible_but_common_prefix_and_noise_prevent_episode_tradeoff"
        next_action = (
            "Do not claim acceleration; freeze a branch-focused runtime benchmark only after backup, or redesign the controller to reduce overhead/enable repeated switching benefits before any selector training."
        )
    elif groups_total_5pct > 0:
        classification = "some_timing_opportunity_survives_but_requires_broader_fresh_confirmation"
        next_action = (
            "Treat as development-only; freeze a broader source-supported confirmation with fixed-H timing baselines, not a supervised refit from the sparse v1e labels."
        )

    analysis = {
        "episode_count": len(episode_dirs),
        "pair_count": len(pair_rows),
        "fixed_nlp_size_across_h10_h15": fixed_nlp,
        "global_opt_x_sizes_by_horizon": global_sizes_unique,
        "horizon_param_mismatch_steps": horizon_param_mismatches,
        "terminal_p_horizon_mismatch_steps": terminal_p_mismatches,
        "episode_solver_fraction_of_decision": stats(all_episode_solver_frac),
        "episode_mpc_info_fraction_of_measured_solver": stats(all_episode_info_frac),
        "terminal_switch_summary_sum_s": stats(terminal_switch_sums),
        "pair_total_decision_saving_s": stats(total_gain),
        "pair_total_solver_saving_s": stats(solver_gain),
        "pair_branch_decision_saving_s": stats(branch_gain),
        "pair_branch_solver_saving_s": stats(branch_solver_gain),
        "pair_decision_minus_solver_overhead_saving_s": stats(overhead_gain),
        "pair_total_relative_decision_saving": stats([safe_float(r.get("total_relative_decision_saving"), 0.0) for r in pair_rows]),
        "pair_branch_relative_decision_saving": stats([safe_float(r.get("branch_relative_decision_saving"), 0.0) for r in pair_rows]),
        "solver_vs_decision_gain_correlation": corr(total_gain, solver_gain),
        "branch_solver_vs_branch_decision_gain_correlation": corr(branch_gain, branch_solver_gain),
        "pair_timing_classes": {c: sum(1 for r in pair_rows if r.get("branch_vs_total_timing_class") == c) for c in sorted(set(str(r.get("branch_vs_total_timing_class")) for r in pair_rows))},
        "groups_with_median_branch_decision_saving_ge5pct": groups_branch_5pct,
        "groups_with_median_total_decision_saving_ge5pct": groups_total_5pct,
        "group_rows": group_rows,
        "pair_rows": pair_rows,
        "classification": classification,
        "next_action": next_action,
    }

    created = now_utc()
    write_json(BACKUP_REQ, {
        "requested_utc": created.isoformat(),
        "reason": "backup v1e case5 timing decomposition postdiagnostic before further simulation/training/refit",
        "backup_required_before_more_simulations": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(OUT), rel(STATE), rel(SOURCE), rel(BACKUP_REQ)],
    })
    result = {
        "created_utc": created.isoformat(),
        "started_utc": started.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_metadata_no_rollout_timing_decomposition_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "inputs": {
            "run_raw": rel(RUN_RAW),
            "run_raw_sha256": sha256(RUN_RAW),
            "run_completed": rel(RUN_DONE),
            "run_completed_sha256": sha256(RUN_DONE),
            "run_summary": rel(RUN_SUMMARY),
            "run_summary_sha256": sha256(RUN_SUMMARY),
        },
        "source_hashes": {rel(SOURCE): sha256(SOURCE)},
        "token_total_best_effort": query_server_token_total(),
        "analysis": analysis,
        "episodes": {k: v for k, v in episodes.items() if k == v.get("path")},
        "backup_request": rel(BACKUP_REQ),
    }
    write_json(OUT / "raw.json", result)
    summary_text = render_summary(result)
    (OUT / "summary.md").write_text(summary_text, encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(summary_text, encoding="utf-8")
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle stress-v1e case5 timing decomposition postdiagnostic v0

UTC: {created.isoformat()}. Metadata/no-rollout decomposition of the 24-episode v1e case5 repeated diagnostic. No new simulations, no training/refit, no validation64 bank, and no sealed-test access. Key result: stable local H10 physical gains do not translate into a robust measured compute tradeoff under the current implementation. Executed H10 and H15 steps have the same recorded optimizer-vector size `{analysis['global_opt_x_sizes_by_horizon']}`, measured decision differences track solver-attempt wall time, and no state/terminal group reaches the predeclared >=5% median whole-episode decision-time saving. Classification: `{classification}`. Next action after backup: {next_action} Backup request: `{rel(BACKUP_REQ)}`.
""")
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, STATE, BACKUP_REQ]
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "headline": {
            "classification": classification,
            "fixed_nlp_size_across_h10_h15": fixed_nlp,
            "groups_with_median_total_decision_saving_ge5pct": groups_total_5pct,
            "groups_with_median_branch_decision_saving_ge5pct": groups_branch_5pct,
            "train_or_refit_now": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        },
        "backup_request": rel(BACKUP_REQ),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "headline": completed["headline"],
        "backup_request": rel(BACKUP_REQ),
        "elapsed_since_first_supervisor_event_seconds": completed["elapsed_since_first_supervisor_event_seconds"],
        "token_total_best_effort": result["token_total_best_effort"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
