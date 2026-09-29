#!/usr/bin/env python3
"""No-simulation solver/decision timing schema postdiagnostic for vehicle stress-v1d.

The v1d Pareto/timing diagnostic found some compute-safe states using measured
whole-decision time, but its solver-time fields were null because v1d smoke rows
store solver timing as `solver_attempt_sum_s`, not `solver_sum_s`.  This script
repairs that schema at analysis time and asks whether the apparent compute-safe
opportunity survives on both whole-decision and solver-attempt timing, or whether
it is dominated by one constant short horizon / timing noise.

No rollouts, no candidate-pool resets, no training/refit, no validation64 bank,
and no sealed test are opened.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260929T0340Z"
SMOKE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/raw.json"
SMOKE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/completed.json"
PARETO_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_pareto_timing_objective_postdiagnostic_v0_20260929T0325Z/raw.json"
PARETO_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_pareto_timing_objective_postdiagnostic_v0_20260929T0325Z/completed.json"
OUT = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1d_solver_timing_schema_postdiagnostic_v0_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1d_solver_timing_schema_postdiagnostic_v0_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1D_SOLVER_TIMING_SCHEMA_POSTDIAGNOSTIC_V0_{STAMP}.json"
TERMINAL_MODES = ["zero_terminal", "h15_common_terminal"]
HORIZONS = [10, 15, 25, 30, 35, 45, 50]
STRICT_EPS = 0.01
STRICT_ABS_GAIN_S = 0.05
STRICT_REL_GAIN = 0.02
RELAXED_EPS = 0.1
RELAXED_ABS_GAIN_S = 0.02
STATE_DISTANCE_TOL = 1e-5
MARKER = f"vehicle-stress-v1d-solver-timing-schema-postdiagnostic-v0-{STAMP}"
FIRST_SUPERVISOR_EVENT = dt.datetime(2026, 9, 26, 10, 55, 29, 419331, tzinfo=dt.timezone.utc)


class ContractError(RuntimeError):
    pass


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
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def maybe_float(x: Any) -> Optional[float]:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def stats(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    def pct(q: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * q / 100.0
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {"count": len(xs), "min": xs[0], "median": pct(50), "mean": sum(xs) / len(xs), "p95": pct(95), "max": xs[-1], "sum": sum(xs)}


def verify_completed(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"completed marker not passing: {rel(path)}")
    for flag in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if obj.get(flag) is not False:
            raise ContractError(f"{rel(path)} has nonfalse access flag {flag}={obj.get(flag)}")
    return obj


def verify_inputs() -> Dict[str, Any]:
    for p in (SMOKE_RAW, SMOKE_DONE, PARETO_RAW, PARETO_DONE):
        if not p.exists():
            raise ContractError(f"required input missing: {rel(p)}")
    smoke_done = verify_completed(SMOKE_DONE)
    pareto_done = verify_completed(PARETO_DONE)
    raw = read_json(SMOKE_RAW)
    pareto = read_json(PARETO_RAW)
    for name, obj in (("smoke_raw", raw), ("pareto_raw", pareto)):
        for flag in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
            if obj.get(flag) is not False:
                raise ContractError(f"{name} has nonfalse access flag {flag}={obj.get(flag)}")
    if int((raw.get("analysis") or {}).get("state_count", -1)) != 12:
        raise ContractError("unexpected v1d smoke state_count")
    if int(pareto.get("state_count", -1)) != 12:
        raise ContractError("unexpected Pareto state_count")
    return {"smoke_raw": raw, "smoke_done": smoke_done, "pareto_raw": pareto, "pareto_done": pareto_done}


def decision_time(row: Mapping[str, Any]) -> Optional[float]:
    return maybe_float(row.get("decision_sum_s"))


def solver_time(row: Mapping[str, Any]) -> Optional[float]:
    # v1d smoke state_rows use solver_attempt_sum_s.  The previous Pareto script
    # looked for solver_sum_s and therefore reported null solver gains.
    for key in ("solver_attempt_sum_s", "solver_sum_s", "solver_total_s", "solve_sum_s", "mpc_solver_sum_s"):
        if key in row and row.get(key) is not None:
            return maybe_float(row.get(key))
    return None


def comp_safety_ok(row: Mapping[str, Any]) -> bool:
    return (
        bool(row.get("branch_reached"))
        and bool(row.get("physical_prefix_matches_H15"))
        and finite(row.get("state_distance_vs_H15"), 1e9) <= STATE_DISTANCE_TOL
        and bool(row.get("no_success_constraint_solver_regression_vs_H15"))
    )


def collect_by_mode(state_row: Mapping[str, Any]) -> Dict[str, Dict[int, Mapping[str, Any]]]:
    out: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for mode in TERMINAL_MODES:
        comps = (((state_row.get("mode_results") or {}).get(mode) or {}).get("comparisons") or [])
        out[mode] = {int(c.get("horizon")): c for c in comps if c.get("horizon") is not None}
        if 15 not in out[mode]:
            raise ContractError(f"state {state_row.get('state_id')} missing H15 comparison for mode {mode}")
    return out


def evaluate_state(state_row: Mapping[str, Any]) -> Dict[str, Any]:
    by_mode = collect_by_mode(state_row)
    horizon_rows: List[Dict[str, Any]] = []
    for h in HORIZONS:
        if h == 15:
            continue
        mode_metrics: Dict[str, Any] = {}
        safe_all = True
        phys_gains: List[float] = []
        decision_gains: List[float] = []
        decision_rel_gains: List[float] = []
        solver_gains: List[float] = []
        solver_rel_gains: List[float] = []
        overhead_delta_gains: List[float] = []
        missing_solver = False
        missing_decision = False
        for mode in TERMINAL_MODES:
            c = by_mode[mode].get(h)
            ref = by_mode[mode][15]
            if c is None:
                raise ContractError(f"state {state_row.get('state_id')} missing H{h} comparison for mode {mode}")
            safe = comp_safety_ok(c)
            safe_all = safe_all and safe
            pg = finite(c.get("gain_vs_H15_physical"))
            phys_gains.append(pg)
            dec = decision_time(c); dec_ref = decision_time(ref)
            d_gain = None; d_rel = None
            if dec is None or dec_ref is None:
                missing_decision = True
            else:
                d_gain = dec_ref - dec
                d_rel = d_gain / max(dec_ref, 1e-9)
                decision_gains.append(d_gain)
                decision_rel_gains.append(d_rel)
            sol = solver_time(c); sol_ref = solver_time(ref)
            s_gain = None; s_rel = None
            if sol is None or sol_ref is None:
                missing_solver = True
            else:
                s_gain = sol_ref - sol
                s_rel = s_gain / max(sol_ref, 1e-9)
                solver_gains.append(s_gain)
                solver_rel_gains.append(s_rel)
            overhead_c = None if dec is None or sol is None else dec - sol
            overhead_ref = None if dec_ref is None or sol_ref is None else dec_ref - sol_ref
            overhead_delta = None if overhead_c is None or overhead_ref is None else overhead_ref - overhead_c
            if overhead_delta is not None:
                overhead_delta_gains.append(overhead_delta)
            mode_metrics[mode] = {
                "safe": safe,
                "physical_gain": pg,
                "physical_loss": max(0.0, -pg),
                "decision_sum_s": dec,
                "decision_ref_H15_sum_s": dec_ref,
                "decision_gain_s": d_gain,
                "decision_rel_gain": d_rel,
                "solver_attempt_sum_s": sol,
                "solver_ref_H15_attempt_sum_s": sol_ref,
                "solver_attempt_gain_s": s_gain,
                "solver_attempt_rel_gain": s_rel,
                "overhead_sum_s": overhead_c,
                "overhead_ref_H15_sum_s": overhead_ref,
                "overhead_gain_s": overhead_delta,
                "success": bool(c.get("success")),
                "constraint": bool(c.get("constraint")),
                "solver_failure_steps": int(c.get("solver_failure_steps", 0) or 0),
                "steps": int(c.get("steps", 0) or 0),
                "ref_steps": int(ref.get("steps", 0) or 0),
            }
        max_phys_loss = max(0.0, -min(phys_gains))
        min_decision_gain = min(decision_gains) if len(decision_gains) == len(TERMINAL_MODES) else None
        min_decision_rel = min(decision_rel_gains) if len(decision_rel_gains) == len(TERMINAL_MODES) else None
        min_solver_gain = min(solver_gains) if len(solver_gains) == len(TERMINAL_MODES) else None
        min_solver_rel = min(solver_rel_gains) if len(solver_rel_gains) == len(TERMINAL_MODES) else None
        min_overhead_gain = min(overhead_delta_gains) if len(overhead_delta_gains) == len(TERMINAL_MODES) else None
        decision_strict = bool(safe_all and max_phys_loss <= STRICT_EPS and min_decision_gain is not None and min_decision_rel is not None and min_decision_gain >= STRICT_ABS_GAIN_S and min_decision_rel >= STRICT_REL_GAIN)
        decision_relaxed = bool(safe_all and max_phys_loss <= RELAXED_EPS and min_decision_gain is not None and min_decision_gain >= RELAXED_ABS_GAIN_S)
        solver_strict = bool(safe_all and max_phys_loss <= STRICT_EPS and min_solver_gain is not None and min_solver_rel is not None and min_solver_gain >= STRICT_ABS_GAIN_S and min_solver_rel >= STRICT_REL_GAIN)
        solver_relaxed = bool(safe_all and max_phys_loss <= RELAXED_EPS and min_solver_gain is not None and min_solver_gain >= RELAXED_ABS_GAIN_S)
        both_strict = decision_strict and solver_strict
        both_relaxed = decision_relaxed and solver_relaxed
        # Timing is suspect when whole-decision and solver-attempt signs differ,
        # or when overhead accounts for more than half the smaller positive gain.
        sign_disagreement = bool(
            min_decision_gain is not None
            and min_solver_gain is not None
            and ((min_decision_gain > 0.0) != (min_solver_gain > 0.0))
        )
        overhead_large = bool(
            min_decision_gain is not None
            and min_solver_gain is not None
            and min_overhead_gain is not None
            and min(abs(min_decision_gain), abs(min_solver_gain)) > 1e-9
            and abs(min_overhead_gain) > 0.5 * min(abs(min_decision_gain), abs(min_solver_gain))
        )
        horizon_rows.append({
            "horizon": h,
            "safe_all_modes": safe_all,
            "max_physical_loss": max_phys_loss,
            "min_physical_gain": min(phys_gains),
            "min_decision_gain_s": min_decision_gain,
            "min_decision_rel_gain": min_decision_rel,
            "min_solver_attempt_gain_s": min_solver_gain,
            "min_solver_attempt_rel_gain": min_solver_rel,
            "min_overhead_gain_s": min_overhead_gain,
            "decision_strict_compute": decision_strict,
            "decision_relaxed_compute": decision_relaxed,
            "solver_strict_compute": solver_strict,
            "solver_relaxed_compute": solver_relaxed,
            "both_strict_compute": both_strict,
            "both_relaxed_compute": both_relaxed,
            "decision_only_strict_compute": decision_strict and not solver_strict,
            "decision_only_relaxed_compute": decision_relaxed and not solver_relaxed,
            "timing_sign_disagreement": sign_disagreement,
            "overhead_large_vs_gain": overhead_large,
            "missing_decision_timing": missing_decision,
            "missing_solver_attempt_timing": missing_solver,
            "mode_metrics": mode_metrics,
        })
    best_both_strict = None
    strict_rows = [r for r in horizon_rows if r["both_strict_compute"]]
    if strict_rows:
        best_both_strict = max(strict_rows, key=lambda r: (finite(r.get("min_solver_attempt_gain_s"), -1e99), finite(r.get("min_decision_gain_s"), -1e99), -finite(r.get("max_physical_loss"), 1e99)))
    relaxed_rows = [r for r in horizon_rows if r["both_relaxed_compute"]]
    best_both_relaxed = None
    if relaxed_rows:
        best_both_relaxed = max(relaxed_rows, key=lambda r: (finite(r.get("min_solver_attempt_gain_s"), -1e99), finite(r.get("min_decision_gain_s"), -1e99), -finite(r.get("max_physical_loss"), 1e99)))
    best_decision_strict = None
    ds = [r for r in horizon_rows if r["decision_strict_compute"]]
    if ds:
        best_decision_strict = max(ds, key=lambda r: (finite(r.get("min_decision_gain_s"), -1e99), -finite(r.get("max_physical_loss"), 1e99)))
    return {
        "state_id": state_row.get("state_id"),
        "case": int(state_row.get("case", -1)),
        "selection_group": state_row.get("selection_group"),
        "target_role": state_row.get("target_role"),
        "branch_step": int(state_row.get("branch_step", -1)),
        "horizon_rows": horizon_rows,
        "decision_strict_horizons": [int(r["horizon"]) for r in horizon_rows if r["decision_strict_compute"]],
        "decision_relaxed_horizons": [int(r["horizon"]) for r in horizon_rows if r["decision_relaxed_compute"]],
        "solver_strict_horizons": [int(r["horizon"]) for r in horizon_rows if r["solver_strict_compute"]],
        "solver_relaxed_horizons": [int(r["horizon"]) for r in horizon_rows if r["solver_relaxed_compute"]],
        "both_strict_horizons": [int(r["horizon"]) for r in horizon_rows if r["both_strict_compute"]],
        "both_relaxed_horizons": [int(r["horizon"]) for r in horizon_rows if r["both_relaxed_compute"]],
        "decision_only_strict_horizons": [int(r["horizon"]) for r in horizon_rows if r["decision_only_strict_compute"]],
        "decision_only_relaxed_horizons": [int(r["horizon"]) for r in horizon_rows if r["decision_only_relaxed_compute"]],
        "best_both_strict": None if best_both_strict is None else {k: v for k, v in best_both_strict.items() if k != "mode_metrics"},
        "best_both_relaxed": None if best_both_relaxed is None else {k: v for k, v in best_both_relaxed.items() if k != "mode_metrics"},
        "best_decision_strict": None if best_decision_strict is None else {k: v for k, v in best_decision_strict.items() if k != "mode_metrics"},
    }


def horizon_count(states: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    out = {str(h): 0 for h in HORIZONS if h != 15}
    for st in states:
        for h in st.get(key, []):
            out[str(int(h))] += 1
    return out


def best_horizon_counts(states: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    out = {str(h): 0 for h in HORIZONS if h != 15}
    for st in states:
        row = st.get(key)
        if row:
            out[str(int(row["horizon"]))] += 1
    return out


def query_server_token_total() -> Dict[str, Any]:
    # Best-effort only; if the supervisor database is outside the repository the
    # final status will report unavailable rather than fabricate a number.
    candidates = [ROOT / "research.sqlite", ROOT / "research_artifacts/research.sqlite", ROOT.parent / "research.sqlite"]
    for db in candidates:
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db))
            cur = con.cursor()
            # Try likely schemas defensively.
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            total = None
            details = {}
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
            return {"available": False, "error": repr(exc), "path": rel(db)}
    return {"available": False, "reason": "research.sqlite not found in repository-visible candidate paths"}


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def run() -> int:
    inputs = verify_inputs()
    state_rows = (inputs["smoke_raw"].get("analysis") or {}).get("state_rows") or []
    states = [evaluate_state(r) for r in state_rows]
    decision_strict_states = [s for s in states if s["decision_strict_horizons"]]
    decision_relaxed_states = [s for s in states if s["decision_relaxed_horizons"]]
    solver_strict_states = [s for s in states if s["solver_strict_horizons"]]
    solver_relaxed_states = [s for s in states if s["solver_relaxed_horizons"]]
    both_strict_states = [s for s in states if s["both_strict_horizons"]]
    both_relaxed_states = [s for s in states if s["both_relaxed_horizons"]]
    decision_only_strict_states = [s for s in states if s["decision_only_strict_horizons"]]
    decision_only_relaxed_states = [s for s in states if s["decision_only_relaxed_horizons"]]
    suspect_rows = []
    decision_best_solver_gains = []
    both_strict_solver_gains = []
    both_relaxed_solver_gains = []
    overhead_deltas = []
    for st in states:
        if st.get("best_decision_strict"):
            decision_best_solver_gains.append(finite(st["best_decision_strict"].get("min_solver_attempt_gain_s"), float("nan")))
        if st.get("best_both_strict"):
            both_strict_solver_gains.append(finite(st["best_both_strict"].get("min_solver_attempt_gain_s")))
        if st.get("best_both_relaxed"):
            both_relaxed_solver_gains.append(finite(st["best_both_relaxed"].get("min_solver_attempt_gain_s")))
        for hr in st["horizon_rows"]:
            if hr["timing_sign_disagreement"] or hr["overhead_large_vs_gain"] or hr["missing_solver_attempt_timing"]:
                suspect_rows.append({
                    "state_id": st["state_id"],
                    "case": st["case"],
                    "horizon": int(hr["horizon"]),
                    "max_physical_loss": hr["max_physical_loss"],
                    "min_decision_gain_s": hr["min_decision_gain_s"],
                    "min_solver_attempt_gain_s": hr["min_solver_attempt_gain_s"],
                    "min_overhead_gain_s": hr["min_overhead_gain_s"],
                    "decision_strict": hr["decision_strict_compute"],
                    "solver_strict": hr["solver_strict_compute"],
                    "decision_relaxed": hr["decision_relaxed_compute"],
                    "solver_relaxed": hr["solver_relaxed_compute"],
                    "timing_sign_disagreement": hr["timing_sign_disagreement"],
                    "overhead_large_vs_gain": hr["overhead_large_vs_gain"],
                })
            if hr.get("min_overhead_gain_s") is not None:
                overhead_deltas.append(float(hr["min_overhead_gain_s"]))

    both_strict_h_counts = horizon_count(states, "both_strict_horizons")
    both_relaxed_h_counts = horizon_count(states, "both_relaxed_horizons")
    best_both_strict_counts = best_horizon_counts(states, "best_both_strict")
    best_both_relaxed_counts = best_horizon_counts(states, "best_both_relaxed")
    dominant_both_strict = max(both_strict_h_counts.items(), key=lambda kv: kv[1])
    dominant_both_relaxed = max(both_relaxed_h_counts.items(), key=lambda kv: kv[1])
    dominant_best_both_relaxed = max(best_both_relaxed_counts.items(), key=lambda kv: kv[1])

    if len(both_strict_states) >= 6 and dominant_both_strict[1] < len(both_strict_states):
        classification = "solver_and_decision_compute_target_nonconstant_but_needs_repeated_timing_confirmation"
        next_action = "freeze a small blocked repeated-timing continuation confirmation for the nonconstant states before any selector/refit"
    elif len(both_relaxed_states) >= 6 and dominant_both_relaxed[1] < len(both_relaxed_states):
        classification = "only_relaxed_solver_confirmed_compute_target_nonconstant_needs_noise_confirmation"
        next_action = "run a small blocked repeated-timing continuation confirmation; do not train/refit until timing survives repeats and fixed-H dominance check"
    elif len(both_strict_states) or len(both_relaxed_states):
        classification = "solver_confirmed_compute_opportunity_sparse_or_constantH_absorbable"
        next_action = "do not train/refit yet; compare dominant short-H fixed baseline with repeated timing or freeze sparse-opportunity diagnosis"
    elif len(decision_strict_states) or len(decision_relaxed_states):
        classification = "decision_time_opportunity_not_solver_confirmed_schema_repair_blocks_training"
        next_action = "treat speed evidence as timing-overhead/noise-sensitive; repair future timing schema and run blocked repeats before any selector/refit"
    else:
        classification = "no_decision_or_solver_compute_opportunity_after_schema_repair"
        next_action = "freeze negative current-scenario opportunity diagnosis or versioned stress-scenario amendment; no selector/refit"
    train_or_refit_now = False

    now = dt.datetime.now(dt.timezone.utc)
    token_info = query_server_token_total()
    elapsed_seconds = (now - FIRST_SUPERVISOR_EVENT).total_seconds()
    result = {
        "created_utc": now.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": elapsed_seconds,
        "server_api_total_tokens_best_effort": token_info,
        "method": "vehicle_stress_v1d_solver_timing_schema_postdiagnostic_v0_no_simulation",
        "classification": "development_no_simulation_solver_timing_schema_repair_no_validation64_no_sealed_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "thresholds": {
            "strict_eps_physical_loss": STRICT_EPS,
            "strict_abs_gain_s": STRICT_ABS_GAIN_S,
            "strict_rel_gain": STRICT_REL_GAIN,
            "relaxed_eps_physical_loss": RELAXED_EPS,
            "relaxed_abs_gain_s": RELAXED_ABS_GAIN_S,
            "timing_source_note": "solver_attempt_sum_s is present in v1d smoke raw and was missing from the previous postdiagnostic's solver extractor",
        },
        "inputs": {
            "smoke_raw": rel(SMOKE_RAW),
            "smoke_raw_sha256": sha256(SMOKE_RAW),
            "smoke_completed": rel(SMOKE_DONE),
            "smoke_completed_sha256": sha256(SMOKE_DONE),
            "pareto_raw": rel(PARETO_RAW),
            "pareto_raw_sha256": sha256(PARETO_RAW),
            "pareto_completed": rel(PARETO_DONE),
            "pareto_completed_sha256": sha256(PARETO_DONE),
        },
        "previous_pareto_headline": (inputs["pareto_done"].get("headline") or {}),
        "state_count": len(states),
        "decision_strict_state_count": len(decision_strict_states),
        "decision_relaxed_state_count": len(decision_relaxed_states),
        "solver_strict_state_count": len(solver_strict_states),
        "solver_relaxed_state_count": len(solver_relaxed_states),
        "both_decision_and_solver_strict_state_count": len(both_strict_states),
        "both_decision_and_solver_relaxed_state_count": len(both_relaxed_states),
        "decision_only_strict_state_count": len(decision_only_strict_states),
        "decision_only_relaxed_state_count": len(decision_only_relaxed_states),
        "both_strict_cases": sorted(set(int(s["case"]) for s in both_strict_states)),
        "both_relaxed_cases": sorted(set(int(s["case"]) for s in both_relaxed_states)),
        "horizon_counts": {
            "decision_strict": horizon_count(states, "decision_strict_horizons"),
            "decision_relaxed": horizon_count(states, "decision_relaxed_horizons"),
            "solver_strict": horizon_count(states, "solver_strict_horizons"),
            "solver_relaxed": horizon_count(states, "solver_relaxed_horizons"),
            "both_strict": both_strict_h_counts,
            "both_relaxed": both_relaxed_h_counts,
            "best_both_strict": best_both_strict_counts,
            "best_both_relaxed": best_both_relaxed_counts,
        },
        "dominance": {
            "dominant_both_strict_horizon": {"horizon": dominant_both_strict[0], "state_count": dominant_both_strict[1]},
            "dominant_both_relaxed_horizon": {"horizon": dominant_both_relaxed[0], "state_count": dominant_both_relaxed[1]},
            "dominant_best_both_relaxed_horizon": {"horizon": dominant_best_both_relaxed[0], "state_count": dominant_best_both_relaxed[1]},
        },
        "timing_summaries": {
            "decision_best_strict_solver_attempt_gains_s": stats(decision_best_solver_gains),
            "best_both_strict_solver_attempt_gains_s": stats(both_strict_solver_gains),
            "best_both_relaxed_solver_attempt_gains_s": stats(both_relaxed_solver_gains),
            "min_overhead_gain_s_across_horizon_rows": stats(overhead_deltas),
        },
        "suspect_timing_rows": suspect_rows,
        "per_state": states,
        "decision": {
            "classification": classification,
            "train_or_refit_now": train_or_refit_now,
            "next_action": next_action,
            "rationale": "Measured speed must be supported by both whole-decision timing and the solver-attempt timing actually recorded in v1d raw data; otherwise shorter H or overhead noise is insufficient for selector training.",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "raw.json", result)
    lines = [
        "# Vehicle stress-v1d solver/decision timing schema postdiagnostic v0",
        "",
        f"UTC: `{now.isoformat()}`. No simulations, no training/refit, no validation64/sealed-test access.",
        "",
        "## Why this was run",
        "",
        "The previous Pareto/timing diagnostic found decision-time opportunities but left solver-time gains null because the v1d smoke schema uses `solver_attempt_sum_s`. This diagnostic repairs that extraction without new rollouts.",
        "",
        "## Headline",
        "",
        f"- Previous decision-time strict/relaxed counts: `{result['previous_pareto_headline'].get('strict_compute_pareto_state_count')}` / `{result['previous_pareto_headline'].get('relaxed_compute_pareto_state_count')}`.",
        f"- Recomputed decision-time strict/relaxed counts: `{len(decision_strict_states)}/12` / `{len(decision_relaxed_states)}/12`.",
        f"- Solver-attempt strict/relaxed counts: `{len(solver_strict_states)}/12` / `{len(solver_relaxed_states)}/12`.",
        f"- Both decision+solver strict/relaxed counts: `{len(both_strict_states)}/12` / `{len(both_relaxed_states)}/12`.",
        f"- Decision-only strict/relaxed states: `{len(decision_only_strict_states)}` / `{len(decision_only_relaxed_states)}`.",
        f"- Dominant both-relaxed horizon: `{result['dominance']['dominant_both_relaxed_horizon']}`; dominant best-both-relaxed horizon: `{result['dominance']['dominant_best_both_relaxed_horizon']}`.",
        f"- Suspect timing rows (sign disagreement, large overhead term, or missing solver timing): `{len(suspect_rows)}`.",
        f"- Best both-relaxed solver-attempt gain summary (s): `{result['timing_summaries']['best_both_relaxed_solver_attempt_gains_s']}`.",
        "",
        "## Per-state timing labels",
        "",
        "| state | case | branch | decision strict H | solver strict H | both strict H | both relaxed H | best both relaxed H/solver gain |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for st in states:
        br = st.get("best_both_relaxed") or {}
        label = "" if not br else f"H{br.get('horizon')} / {br.get('min_solver_attempt_gain_s')}"
        lines.append("| `%s` | %d | %d | `%s` | `%s` | `%s` | `%s` | `%s` |" % (
            st["state_id"], int(st["case"]), int(st["branch_step"]), st["decision_strict_horizons"], st["solver_strict_horizons"], st["both_strict_horizons"], st["both_relaxed_horizons"], label))
    lines += [
        "",
        "## Interpretation and decision",
        "",
        f"Decision classification: `{classification}`.",
        f"Train/refit now: `{train_or_refit_now}`.",
        f"Next action: {next_action}.",
        "",
        "This remains development-only. It repairs a timing-schema diagnostic issue but does not itself prove acceleration; any speed claim still needs blocked/repeated timing and strong fixed-H baselines.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        f"# v1d solver/decision timing schema postdiagnostic state\n\nUTC: {now.isoformat()}. both_strict={len(both_strict_states)}/12; both_relaxed={len(both_relaxed_states)}/12; decision_only_strict={len(decision_only_strict_states)}; classification={classification}; train_or_refit_now={train_or_refit_now}. Next: {next_action}. No simulations/training/validation/test.\n",
        encoding="utf-8",
    )
    write_json(BACKUP_REQ, {
        "requested_utc": now.isoformat(),
        "reason": "backup v1d solver/decision timing schema postdiagnostic before further simulation/refit/training",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(OUT), rel(STATE), rel(Path(__file__).resolve()), rel(BACKUP_REQ)],
    })
    docs_block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle stress-v1d solver/decision timing schema postdiagnostic

UTC: {now.isoformat()}. No-simulation timing-schema repair over existing v1d smoke outputs. The earlier Pareto script reported solver-time gains as null because v1d rows store `solver_attempt_sum_s`; this diagnostic re-extracted solver-attempt timing while keeping whole-decision timing separate. Both decision+solver strict compute-safe states={len(both_strict_states)}/12, both relaxed={len(both_relaxed_states)}/12; decision-only strict/relaxed states={len(decision_only_strict_states)}/{len(decision_only_relaxed_states)}. Classification `{classification}`; train/refit now remains false. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`. Backup request: `{rel(BACKUP_REQ)}`. Validation64 and sealed test stayed closed.
"""
    append_docs(docs_block)
    files = [OUT / "raw.json", OUT / "summary.md", STATE, BACKUP_REQ, Path(__file__).resolve(), SMOKE_RAW, SMOKE_DONE, PARETO_RAW, PARETO_DONE]
    write_json(OUT / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": now.isoformat(),
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "headline": {
            "both_decision_and_solver_strict_state_count": len(both_strict_states),
            "both_decision_and_solver_relaxed_state_count": len(both_relaxed_states),
            "decision_only_strict_state_count": len(decision_only_strict_states),
            "decision_only_relaxed_state_count": len(decision_only_relaxed_states),
            "classification": classification,
            "train_or_refit_now": train_or_refit_now,
            "next_action": next_action,
        },
        "backup_request": rel(BACKUP_REQ),
        "hashes": {rel(p): sha256(p) for p in sorted(files) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "both_decision_and_solver_strict_state_count": len(both_strict_states),
        "both_decision_and_solver_relaxed_state_count": len(both_relaxed_states),
        "decision_only_strict_state_count": len(decision_only_strict_states),
        "decision_only_relaxed_state_count": len(decision_only_relaxed_states),
        "classification": classification,
        "train_or_refit_now": train_or_refit_now,
        "next_action": next_action,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(BACKUP_REQ),
        "elapsed_since_first_supervisor_event_seconds": elapsed_seconds,
        "server_api_total_tokens_best_effort": token_info,
    }, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-accept-development-postdiagnostic", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_postdiagnostic:
        raise ContractError("explicit --i-accept-development-postdiagnostic required")
    if OUT.exists() and (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        if done.get("passed") is not True and done.get("hard_pass") is not True:
            raise ContractError("existing completed marker does not pass")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if OUT.exists() and any(p.name != "run.lock" for p in OUT.iterdir()):
        raise ContractError(f"partial output exists; inspect first: {rel(OUT)}")
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
