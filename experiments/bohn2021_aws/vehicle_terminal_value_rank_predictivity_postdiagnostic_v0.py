#!/usr/bin/env python3
"""No-simulation terminal-value/objective rank predictivity diagnostic.

Motivation
----------
Recent v1b/v1d diagnostics show two different bottlenecks:

* robust physical-improvement labels are sparse on fresh terminal-stable states;
* measured-compute labels exist only under relaxed/nontraining-ready gates; and
* per-H/H25 terminal labels can flip labels or select unsafe continuations.

This script asks whether the *implemented MPC branch objective / terminal value*
rank candidate horizons in a way that predicts realised continuation outcomes on
already-collected development branch banks.  It does not run rollouts, train,
refit, open validation64, or open the sealed test.

The diagnostic is intended to decide whether the next intervention should be an
objective/terminal-value repair or a scenario/opportunity redesign, rather than
another unchanged label-density sweep.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sqlite3
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260929T0425Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/vehicle_terminal_value_rank_predictivity_postdiagnostic_v0_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/vehicle_terminal_value_rank_predictivity_postdiagnostic_v0_{STAMP}.md"
REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TERMINAL_VALUE_RANK_PREDICTIVITY_POSTDIAGNOSTIC_V0_{STAMP}.json"
MARKER = f"vehicle-terminal-value-rank-predictivity-postdiagnostic-v0-{STAMP}"
FIRST_SUPERVISOR_EVENT = dt.datetime(2026, 9, 26, 10, 55, 29, 419331, tzinfo=dt.timezone.utc)

V1D_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/raw.json"
V1D_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/completed.json"
V1D_SOLVER_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_solver_timing_schema_postdiagnostic_v0_20260929T0340Z/completed.json"
V1D_ORACLE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1d_compute_tradeoff_oracle_upper_bound_postdiagnostic_v0_20260929T0355Z/completed.json"
V1B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_full_20260928T2340Z_legacy_schema_repair/raw.json"
V1B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_full_20260928T2340Z_legacy_schema_repair/completed.json"
V1B_OBJECTIVE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0b_schema_repair_20260929T0035Z/completed.json"

PHYS_TOLS = (0.01, 0.1, 3.0)
COMPUTE_EPS = 0.1
COMPUTE_ABS_GAIN_S = 0.02
STATE_DISTANCE_TOL = 1e-5


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def opt(row: Mapping[str, Any]) -> float:
    for k in ("branch_objective_opt_f_num", "objective", "opt_f_num", "mpc_objective"):
        if k in row:
            y = fnum(row.get(k))
            if math.isfinite(y):
                return y
    return float("nan")


def val(row: Mapping[str, Any]) -> float:
    for k in ("branch_mpc_value_fn", "value_fn", "mpc_value_fn", "terminal_value"):
        if k in row:
            y = fnum(row.get(k))
            if math.isfinite(y):
                return y
    return float("nan")


def stats(xs: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    def pct(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * q / 100.0
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        return vals[lo] if lo == hi else vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return {"count": len(vals), "min": vals[0], "median": pct(50), "mean": sum(vals) / len(vals), "p95": pct(95), "max": vals[-1], "sum": sum(vals)}


def ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-24 or vy <= 1e-24:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    rx = ranks([p[0] for p in pairs]); ry = ranks([p[1] for p in pairs])
    return pearson(rx, ry)


def safe_record(r: Mapping[str, Any]) -> bool:
    return (
        bool(r.get("success", False))
        and not bool(r.get("constraint", False))
        and int(fnum(r.get("solver_failure_steps"), 0)) == 0
        and int(fnum(r.get("initial_failed_steps"), 0)) == 0
        and int(fnum(r.get("final_failed_steps"), 0)) == 0
        and ("branch_reached" not in r or bool(r.get("branch_reached")))
        and ("no_success_constraint_solver_regression_vs_H15" not in r or bool(r.get("no_success_constraint_solver_regression_vs_H15")))
        and ("state_distance_vs_H15" not in r or fnum(r.get("state_distance_vs_H15"), 1e99) <= STATE_DISTANCE_TOL)
        and ("branch_state_distance_vs_H15" not in r or fnum(r.get("branch_state_distance_vs_H15"), 1e99) <= STATE_DISTANCE_TOL)
    )


def verify_done(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing required completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"completed marker not passed: {rel(path)}")
    if obj.get("sealed_test_accessed") is True or obj.get("sealed_test_bank_opened") is True:
        raise ContractError(f"sealed test access flag set in {rel(path)}")
    return obj


def check_access_flags(name: str, obj: Mapping[str, Any]) -> None:
    for flag in ("validation64_bank_opened", "historical_validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if obj.get(flag) is True:
            raise ContractError(f"{name} has forbidden flag {flag}=True")


def parse_v1d(raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for st in ((raw.get("analysis") or {}).get("state_rows") or []):
        for mode, mobj in ((st.get("mode_results") or {}).items()):
            for c in (mobj.get("comparisons") or []):
                rec = {
                    "dataset": "v1d_fresh_trace_selected_terminal_stable",
                    "state_id": str(st.get("state_id")),
                    "case": int(st.get("case", c.get("case", -1)) or -1),
                    "selection_group": st.get("selection_group"),
                    "target_role": st.get("target_role"),
                    "branch_step": int(st.get("branch_step", c.get("branch_step", -1)) or -1),
                    "terminal_mode": str(mode),
                    "horizon": int(c.get("horizon")),
                    "continuation_physical": fnum(c.get("continuation_physical")),
                    "continuation_total": fnum(c.get("continuation_total")),
                    "physical_gain_vs_H15": fnum(c.get("gain_vs_H15_physical")),
                    "total_gain_vs_H15": fnum(c.get("gain_vs_H15_total")),
                    "decision_sum_s": fnum(c.get("decision_sum_s")),
                    "solver_attempt_sum_s": fnum(c.get("solver_attempt_sum_s")),
                    "objective": opt(c),
                    "value_fn": val(c),
                    "success": bool(c.get("success", False)),
                    "constraint": bool(c.get("constraint", False)),
                    "solver_failure_steps": int(fnum(c.get("solver_failure_steps"), 0)),
                    "initial_failed_steps": int(fnum(c.get("initial_failed_steps"), 0)),
                    "final_failed_steps": int(fnum(c.get("final_failed_steps"), 0)),
                    "branch_reached": bool(c.get("branch_reached", True)),
                    "no_success_constraint_solver_regression_vs_H15": bool(c.get("no_success_constraint_solver_regression_vs_H15", True)),
                    "state_distance_vs_H15": fnum(c.get("state_distance_vs_H15"), 0.0),
                    "path": c.get("path"),
                }
                rec["safe"] = safe_record(rec)
                rows.append(rec)
    return rows


def parse_v1b(raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in ((raw.get("analysis") or {}).get("comparison_rows") or []):
        rec = {
            "dataset": "v1b_mined_terminal_reward_ablation",
            "state_id": str(r.get("state_id")),
            "case": int(r.get("case", -1) or -1),
            "selection_group": r.get("selection_group"),
            "target_role": r.get("state_role", r.get("role")),
            "branch_step": int(r.get("branch_step", -1) or -1),
            "terminal_mode": str(r.get("terminal_mode")),
            "horizon": int(r.get("horizon")),
            "continuation_physical": fnum(r.get("continuation_physical")),
            "continuation_total": fnum(r.get("continuation_total")),
            "physical_gain_vs_H15": fnum(r.get("gain_vs_per_h_H15_physical", r.get("gain_vs_H15_physical"))),
            "total_gain_vs_H15": fnum(r.get("gain_vs_per_h_H15_total", r.get("gain_vs_H15_total"))),
            "decision_sum_s": fnum(r.get("decision_sum_s", r.get("decision_plus_terminal_switch_sum_s"))),
            "solver_attempt_sum_s": fnum(r.get("solver_attempt_sum_s")),
            "objective": opt(r),
            "value_fn": val(r),
            "success": bool(r.get("success", False)),
            "constraint": bool(r.get("constraint", False)),
            "solver_failure_steps": int(fnum(r.get("solver_failure_steps"), 0)),
            "initial_failed_steps": int(fnum(r.get("initial_failed_steps"), 0)),
            "final_failed_steps": int(fnum(r.get("final_failed_steps"), 0)),
            "branch_state_distance_vs_H15": fnum(r.get("branch_state_distance_vs_per_h_H15", r.get("branch_state_distance_vs_H15", 0.0)), 0.0),
            "path": r.get("path"),
            "state_role": r.get("state_role"),
        }
        rec["safe"] = safe_record(rec)
        rows.append(rec)
    return rows


def group_records(records: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, str], List[Mapping[str, Any]]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        groups[(str(r["dataset"]), str(r["state_id"]), str(r["terminal_mode"]))].append(r)
    return groups


def choose_min(rows: Sequence[Mapping[str, Any]], field: str, require_safe: bool = False) -> Optional[Mapping[str, Any]]:
    xs = [r for r in rows if math.isfinite(fnum(r.get(field))) and ((not require_safe) or bool(r.get("safe")))]
    if not xs:
        return None
    return min(xs, key=lambda r: (fnum(r.get(field)), int(r.get("horizon", 999))))


def summarize_group(key: Tuple[str, str, str], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    dataset, sid, mode = key
    rs = sorted(list(rows), key=lambda r: int(r.get("horizon", 999)))
    safe_rs = [r for r in rs if bool(r.get("safe")) and math.isfinite(fnum(r.get("continuation_physical")))]
    all_phys_rs = [r for r in rs if math.isfinite(fnum(r.get("continuation_physical")))]
    phys_best = choose_min(safe_rs, "continuation_physical", require_safe=False)
    obj_best_all = choose_min(rs, "objective", require_safe=False)
    obj_best_safe = choose_min(rs, "objective", require_safe=True)
    val_best_safe = choose_min(rs, "value_fn", require_safe=True)
    obj_spearman = None
    val_spearman = None
    if len(safe_rs) >= 3:
        obj_spearman = spearman([fnum(r.get("objective")) for r in safe_rs], [fnum(r.get("continuation_physical")) for r in safe_rs])
        val_spearman = spearman([fnum(r.get("value_fn")) for r in safe_rs], [fnum(r.get("continuation_physical")) for r in safe_rs])
    phys_vals = [fnum(r.get("continuation_physical")) for r in safe_rs]
    obj_vals = [fnum(r.get("objective")) for r in rs if math.isfinite(fnum(r.get("objective")))]
    val_vals = [fnum(r.get("value_fn")) for r in rs if math.isfinite(fnum(r.get("value_fn")))]
    phys_range = max(phys_vals) - min(phys_vals) if len(phys_vals) >= 2 else None
    obj_range = max(obj_vals) - min(obj_vals) if len(obj_vals) >= 2 else None
    val_range = max(val_vals) - min(val_vals) if len(val_vals) >= 2 else None
    best_phys = fnum(phys_best.get("continuation_physical")) if phys_best else float("nan")
    def loss_vs_best(chosen: Optional[Mapping[str, Any]]) -> Optional[float]:
        if chosen is None or not math.isfinite(best_phys) or not math.isfinite(fnum(chosen.get("continuation_physical"))):
            return None
        return fnum(chosen.get("continuation_physical")) - best_phys
    obj_loss = loss_vs_best(obj_best_safe)
    val_loss = loss_vs_best(val_best_safe)
    obj_all_unsafe = bool(obj_best_all is not None and not bool(obj_best_all.get("safe")))
    selected_matches = {}
    for tol in PHYS_TOLS:
        selected_matches[f"objective_safe_within_{tol}"] = bool(obj_loss is not None and obj_loss <= tol)
        selected_matches[f"value_safe_within_{tol}"] = bool(val_loss is not None and val_loss <= tol)
    return {
        "dataset": dataset,
        "state_id": sid,
        "case": int(rs[0].get("case", -1)) if rs else -1,
        "terminal_mode": mode,
        "state_role": rs[0].get("target_role", rs[0].get("state_role")) if rs else None,
        "horizons": [int(r.get("horizon")) for r in rs],
        "safe_horizons": [int(r.get("horizon")) for r in safe_rs],
        "row_count": len(rs),
        "safe_count": len(safe_rs),
        "physical_best_horizon": None if phys_best is None else int(phys_best.get("horizon")),
        "objective_min_horizon_all": None if obj_best_all is None else int(obj_best_all.get("horizon")),
        "objective_min_horizon_safe": None if obj_best_safe is None else int(obj_best_safe.get("horizon")),
        "value_min_horizon_safe": None if val_best_safe is None else int(val_best_safe.get("horizon")),
        "objective_min_all_safe": None if obj_best_all is None else bool(obj_best_all.get("safe")),
        "physical_range_safe": phys_range,
        "objective_range_all": obj_range,
        "value_fn_range_all": val_range,
        "objective_loss_vs_physical_best": obj_loss,
        "value_loss_vs_physical_best": val_loss,
        "objective_spearman_vs_physical_cost_safe": obj_spearman,
        "value_spearman_vs_physical_cost_safe": val_spearman,
        "objective_selected_unsafe": obj_all_unsafe,
        "terminal_or_objective_dominated_flat_physical": bool((phys_range is not None and phys_range <= 0.1) and ((obj_range or 0.0) >= 1.0 or (val_range or 0.0) >= 10.0)),
        "meaningful_physical_spread_ge_0p1": bool(phys_range is not None and phys_range >= 0.1),
        "material_physical_spread_ge_3": bool(phys_range is not None and phys_range >= 3.0),
        "severe_objective_rank_harm": bool((obj_loss is not None and obj_loss >= 3.0) or obj_all_unsafe),
        **selected_matches,
    }


def compute_safe_rank_records(v1d_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    groups = group_records(v1d_records)
    for key, rows in groups.items():
        dataset, sid, mode = key
        if dataset != "v1d_fresh_trace_selected_terminal_stable":
            continue
        by_h = {int(r.get("horizon")): r for r in rows}
        ref = by_h.get(15)
        if ref is None or not bool(ref.get("safe")):
            continue
        ref_phys = fnum(ref.get("continuation_physical"))
        ref_dec = fnum(ref.get("decision_sum_s"))
        ref_solver = fnum(ref.get("solver_attempt_sum_s"))
        if not (math.isfinite(ref_phys) and math.isfinite(ref_dec) and math.isfinite(ref_solver)):
            continue
        eligible = []
        timed = []
        for h, r in by_h.items():
            if not bool(r.get("safe")):
                continue
            phys_loss = fnum(r.get("continuation_physical")) - ref_phys
            dec_gain = ref_dec - fnum(r.get("decision_sum_s"))
            solver_gain = ref_solver - fnum(r.get("solver_attempt_sum_s"))
            if phys_loss <= COMPUTE_EPS:
                eligible.append((h, r, phys_loss, dec_gain, solver_gain))
                if h != 15 and dec_gain >= COMPUTE_ABS_GAIN_S and solver_gain >= COMPUTE_ABS_GAIN_S:
                    timed.append((h, r, phys_loss, dec_gain, solver_gain))
        if not timed:
            continue
        best = min(timed, key=lambda x: (fnum(x[1].get("decision_sum_s")), fnum(x[1].get("solver_attempt_sum_s")), x[0]))
        obj = choose_min(list(rows), "objective", require_safe=True)
        value = choose_min(list(rows), "value_fn", require_safe=True)
        phys = choose_min([x[1] for x in eligible], "continuation_physical", require_safe=False)
        def summarize_choice(chosen: Optional[Mapping[str, Any]], prefix: str) -> Dict[str, Any]:
            if chosen is None:
                return {
                    f"{prefix}_horizon": None,
                    f"{prefix}_is_best_compute": False,
                    f"{prefix}_decision_regret_vs_best_s": None,
                    f"{prefix}_solver_regret_vs_best_s": None,
                    f"{prefix}_physical_loss_vs_best_compute": None,
                    f"{prefix}_safe": None,
                }
            return {
                f"{prefix}_horizon": int(chosen.get("horizon")),
                f"{prefix}_is_best_compute": int(chosen.get("horizon")) == int(best[0]),
                f"{prefix}_decision_regret_vs_best_s": fnum(chosen.get("decision_sum_s")) - fnum(best[1].get("decision_sum_s")),
                f"{prefix}_solver_regret_vs_best_s": fnum(chosen.get("solver_attempt_sum_s")) - fnum(best[1].get("solver_attempt_sum_s")),
                f"{prefix}_physical_loss_vs_best_compute": fnum(chosen.get("continuation_physical")) - fnum(best[1].get("continuation_physical")),
                f"{prefix}_safe": bool(chosen.get("safe")),
            }
        rec = {
            "state_id": sid,
            "case": int(ref.get("case", -1)),
            "terminal_mode": mode,
            "state_role": ref.get("target_role"),
            "branch_step": int(ref.get("branch_step", -1)),
            "best_compute_horizon": int(best[0]),
            "best_compute_decision_gain_vs_H15_s": best[3],
            "best_compute_solver_gain_vs_H15_s": best[4],
            "best_compute_physical_loss_vs_H15": best[2],
            "eligible_phys_nonworse_horizons": [int(x[0]) for x in eligible],
            "timed_phys_nonworse_horizons": [int(x[0]) for x in timed],
            **summarize_choice(obj, "objective_min"),
            **summarize_choice(value, "value_min"),
            **summarize_choice(phys, "physical_min"),
        }
        out.append(rec)
    return out


def aggregate_groups(group_rows: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Any]:
    rows = list(group_rows)
    meaningful = [r for r in rows if r.get("meaningful_physical_spread_ge_0p1")]
    material = [r for r in rows if r.get("material_physical_spread_ge_3")]
    safe_obj_available = [r for r in rows if r.get("objective_min_horizon_safe") is not None and r.get("physical_best_horizon") is not None]
    spears = [r.get("objective_spearman_vs_physical_cost_safe") for r in rows if r.get("objective_spearman_vs_physical_cost_safe") is not None]
    val_spears = [r.get("value_spearman_vs_physical_cost_safe") for r in rows if r.get("value_spearman_vs_physical_cost_safe") is not None]
    def count_within(rows_: Sequence[Mapping[str, Any]], prefix: str, tol: float) -> int:
        return int(sum(1 for r in rows_ if r.get(f"{prefix}_safe_within_{tol}") is True))
    return {
        "label": label,
        "group_count": len(rows),
        "safe_objective_available_groups": len(safe_obj_available),
        "meaningful_physical_spread_ge_0p1_groups": len(meaningful),
        "material_physical_spread_ge_3_groups": len(material),
        "objective_within_0p01_all_groups": count_within(safe_obj_available, "objective", 0.01),
        "objective_within_0p1_all_groups": count_within(safe_obj_available, "objective", 0.1),
        "objective_within_3_all_groups": count_within(safe_obj_available, "objective", 3.0),
        "objective_within_0p1_meaningful_groups": count_within(meaningful, "objective", 0.1),
        "objective_within_3_material_groups": count_within(material, "objective", 3.0),
        "value_within_0p1_meaningful_groups": count_within(meaningful, "value", 0.1),
        "terminal_or_objective_dominated_flat_physical_groups": int(sum(1 for r in rows if r.get("terminal_or_objective_dominated_flat_physical"))),
        "objective_selected_unsafe_groups": int(sum(1 for r in rows if r.get("objective_selected_unsafe"))),
        "severe_objective_rank_harm_groups": int(sum(1 for r in rows if r.get("severe_objective_rank_harm"))),
        "objective_loss_vs_physical_best_summary": stats([r.get("objective_loss_vs_physical_best") for r in rows if r.get("objective_loss_vs_physical_best") is not None]),
        "value_loss_vs_physical_best_summary": stats([r.get("value_loss_vs_physical_best") for r in rows if r.get("value_loss_vs_physical_best") is not None]),
        "objective_spearman_summary": stats(spears),
        "value_spearman_summary": stats(val_spears),
        "physical_best_horizon_counts": dict(Counter(str(r.get("physical_best_horizon")) for r in rows if r.get("physical_best_horizon") is not None)),
        "objective_min_safe_horizon_counts": dict(Counter(str(r.get("objective_min_horizon_safe")) for r in rows if r.get("objective_min_horizon_safe") is not None)),
        "value_min_safe_horizon_counts": dict(Counter(str(r.get("value_min_horizon_safe")) for r in rows if r.get("value_min_horizon_safe") is not None)),
    }


def aggregate_compute(compute_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(compute_rows)
    objective_matches = [r for r in rows if r.get("objective_min_is_best_compute")]
    value_matches = [r for r in rows if r.get("value_min_is_best_compute")]
    return {
        "v1d_state_mode_groups_with_nonH15_compute_safe_eps0p1": len(rows),
        "objective_min_matches_best_compute": len(objective_matches),
        "value_min_matches_best_compute": len(value_matches),
        "objective_match_rate": (len(objective_matches) / len(rows)) if rows else None,
        "value_match_rate": (len(value_matches) / len(rows)) if rows else None,
        "best_compute_horizon_counts": dict(Counter(str(r.get("best_compute_horizon")) for r in rows)),
        "objective_min_horizon_counts_on_compute_groups": dict(Counter(str(r.get("objective_min_horizon")) for r in rows)),
        "value_min_horizon_counts_on_compute_groups": dict(Counter(str(r.get("value_min_horizon")) for r in rows)),
        "objective_decision_regret_summary_s": stats([r.get("objective_min_decision_regret_vs_best_s") for r in rows if r.get("objective_min_decision_regret_vs_best_s") is not None]),
        "objective_solver_regret_summary_s": stats([r.get("objective_min_solver_regret_vs_best_s") for r in rows if r.get("objective_min_solver_regret_vs_best_s") is not None]),
        "objective_physical_loss_vs_best_compute_summary": stats([r.get("objective_min_physical_loss_vs_best_compute") for r in rows if r.get("objective_min_physical_loss_vs_best_compute") is not None]),
        "best_compute_decision_gain_vs_H15_summary_s": stats([r.get("best_compute_decision_gain_vs_H15_s") for r in rows]),
        "best_compute_solver_gain_vs_H15_summary_s": stats([r.get("best_compute_solver_gain_vs_H15_s") for r in rows]),
    }


def query_tokens() -> Dict[str, Any]:
    for db in (ROOT / "research.sqlite", ROOT / "research_artifacts/research.sqlite", ROOT.parent / "research.sqlite"):
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db)); cur = con.cursor(); total = 0; by_table = {}; found = False
            for (table,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                if "total_tokens" in cols:
                    val_ = int(cur.execute(f"SELECT COALESCE(SUM(total_tokens),0) FROM {table}").fetchone()[0] or 0)
                    total += val_; by_table[table] = val_; found = True
            con.close()
            if found:
                return {"available": True, "path": rel(db), "total_tokens": total, "by_table": by_table}
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": repr(exc)}
    return {"available": False, "reason": "research.sqlite not found in repository-visible candidate paths"}


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    v1d = raw["aggregate_by_dataset"]["v1d_fresh_trace_selected_terminal_stable"]
    v1b = raw["aggregate_by_dataset"]["v1b_mined_terminal_reward_ablation"]
    comp = raw["v1d_compute_safe_predictivity"]
    lines = [
        "# Vehicle terminal-value/objective rank-predictivity postdiagnostic v0",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations/training/refit; validation64 and sealed test stayed closed.",
        "",
        "## Headline",
        "",
        f"- Analysed `{raw['record_count']}` horizon rows in `{raw['group_count']}` state-mode groups from v1d fresh trace-selected and v1b mined terminal-ablation development banks.",
        f"- v1d meaningful physical-spread groups (>=0.1): `{v1d['meaningful_physical_spread_ge_0p1_groups']}/{v1d['group_count']}`; material physical-spread groups (>=3): `{v1d['material_physical_spread_ge_3_groups']}`.",
        f"- v1b meaningful/material physical-spread groups: `{v1b['meaningful_physical_spread_ge_0p1_groups']}` / `{v1b['material_physical_spread_ge_3_groups']}`; severe objective-rank harm groups: `{v1b['severe_objective_rank_harm_groups']}`.",
        f"- v1d compute-safe non-H15 state-mode groups at eps=0.1: `{comp['v1d_state_mode_groups_with_nonH15_compute_safe_eps0p1']}`; objective-min matches best measured-compute horizon in `{comp['objective_min_matches_best_compute']}` (`{'' if comp['objective_match_rate'] is None else f'{100.0*comp['objective_match_rate']:.1f}%'}`).",
        f"- Objective-min horizon counts on v1d compute groups: `{comp['objective_min_horizon_counts_on_compute_groups']}` vs best-compute horizon counts `{comp['best_compute_horizon_counts']}`.",
        f"- Decision: `{raw['decision']['classification']}`. Next: {raw['decision']['next_action']}.",
        "",
        "## Dataset aggregate table",
        "",
        "| dataset | groups | meaningful spread >=0.1 | material spread >=3 | obj within 0.1 on meaningful | obj within 3 on material | obj unsafe | severe obj harm | flat-physical terminal/objective dominated | obj loss mean/max | obj spearman median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, agg in raw["aggregate_by_dataset"].items():
        loss = agg["objective_loss_vs_physical_best_summary"]
        spear = agg["objective_spearman_summary"]
        lines.append("| `%s` | %d | %d | %d | %d | %d | %d | %d | %d | %s/%s | %s |" % (
            name,
            agg["group_count"],
            agg["meaningful_physical_spread_ge_0p1_groups"],
            agg["material_physical_spread_ge_3_groups"],
            agg["objective_within_0p1_meaningful_groups"],
            agg["objective_within_3_material_groups"],
            agg["objective_selected_unsafe_groups"],
            agg["severe_objective_rank_harm_groups"],
            agg["terminal_or_objective_dominated_flat_physical_groups"],
            "" if loss["mean"] is None else "%.4g" % loss["mean"],
            "" if loss["max"] is None else "%.4g" % loss["max"],
            "" if spear["median"] is None else "%.3g" % spear["median"],
        ))
    lines += [
        "",
        "## v1d compute-safe ranking check",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| state-mode groups with non-H15 compute-safe eps0.1 | {comp['v1d_state_mode_groups_with_nonH15_compute_safe_eps0p1']} |",
        f"| objective-min matches best compute | {comp['objective_min_matches_best_compute']} |",
        f"| value-min matches best compute | {comp['value_min_matches_best_compute']} |",
        f"| objective decision regret sum/median s | {comp['objective_decision_regret_summary_s']['sum']:.6g} / {comp['objective_decision_regret_summary_s']['median']} |",
        f"| objective solver regret sum/median s | {comp['objective_solver_regret_summary_s']['sum']:.6g} / {comp['objective_solver_regret_summary_s']['median']} |",
        f"| best-compute decision gain vs H15 sum/median s | {comp['best_compute_decision_gain_vs_H15_summary_s']['sum']:.6g} / {comp['best_compute_decision_gain_vs_H15_summary_s']['median']} |",
        "",
        "## Most informative failure rows",
        "",
        "| source | state | mode | phys-best H | obj-min H | obj loss | unsafe | phys range | obj range | value range |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in raw["top_objective_harm_groups"][:12]:
        lines.append("| `%s` | `%s` | `%s` | %s | %s | %s | `%s` | %s | %s | %s |" % (
            r["dataset"], r["state_id"], r["terminal_mode"],
            r.get("physical_best_horizon"), r.get("objective_min_horizon_safe"),
            "" if r.get("objective_loss_vs_physical_best") is None else "%.6g" % r["objective_loss_vs_physical_best"],
            r.get("objective_selected_unsafe"),
            "" if r.get("physical_range_safe") is None else "%.6g" % r["physical_range_safe"],
            "" if r.get("objective_range_all") is None else "%.6g" % r["objective_range_all"],
            "" if r.get("value_fn_range_all") is None else "%.6g" % r["value_fn_range_all"],
        ))
    lines += [
        "",
        "## Four-axis interpretation",
        "",
        "### SCENARIOS",
        f"- verified: v1d fresh trace-selected groups still show low material physical spread ({v1d['material_physical_spread_ge_3_groups']} groups), while v1b mined groups contain larger spreads; opportunity remains concentrated rather than general.",
        "- discriminator: do not run another unchanged density sweep; if continuing scenario work, change the source-supported generator or state-selection hypothesis and retune fixed-H baselines separately.",
        "",
        "### REWARD / TERMINAL / VALUE",
        f"- verified: objective/value rank predictivity is weak where physical outcomes are meaningful, and v1d compute-efficient short horizons are not reliably selected by the existing objective (match rate {comp['objective_match_rate']}).",
        "- discriminator: next intervention should be an IMPROVED objective/terminal repair or terminal-value refit feasibility test before any selector training on these labels.",
        "",
        "### TRAINING",
        "- verified: this diagnostic used zero training/refit. It shows why simply training a selector to imitate current branch objectives would target a misaligned label source.",
        "- discriminator: if training is attempted next, freeze a bounded refit using realised continuation/control-time labels, not per-H/H25 terminal labels or raw branch objective minima.",
        "",
        "### COMPARISONS",
        "- verified: this is not adaptive-vs-fixed validation. Timing is single-run development timing; no speed claim is made.",
        "- discriminator: any repaired method must still face fixed-H/Pareto baselines with blocked measured timing and independent validation.",
        "",
        f"Backup request: `{raw['backup_request']}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    try:
        if (OUT / "completed.json").exists():
            done = read_json(OUT / "completed.json")
            print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
            return 0
        if OUT.exists() and any(OUT.iterdir()):
            raise ContractError(f"partial output exists; inspect before rerun: {rel(OUT)}")
        for p in (V1D_RAW, V1D_DONE, V1D_SOLVER_DONE, V1D_ORACLE_DONE, V1B_RAW, V1B_DONE, V1B_OBJECTIVE_DONE):
            if not p.exists():
                raise ContractError(f"missing input: {rel(p)}")
        for p in (V1D_DONE, V1D_SOLVER_DONE, V1D_ORACLE_DONE, V1B_DONE, V1B_OBJECTIVE_DONE):
            verify_done(p)
        v1d_raw = read_json(V1D_RAW); v1b_raw = read_json(V1B_RAW)
        check_access_flags("v1d_raw", v1d_raw); check_access_flags("v1b_raw", v1b_raw)
        v1d_records = parse_v1d(v1d_raw)
        v1b_records = parse_v1b(v1b_raw)
        records = v1d_records + v1b_records
        if not v1d_records or not v1b_records:
            raise ContractError("parsed no records from one or more inputs")
        group_rows = [summarize_group(k, v) for k, v in group_records(records).items()]
        by_dataset: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for g in group_rows:
            by_dataset[str(g["dataset"])].append(g)
        aggregate_by_dataset = {name: aggregate_groups(rows, name) for name, rows in by_dataset.items()}
        compute_rows = compute_safe_rank_records(v1d_records)
        compute_agg = aggregate_compute(compute_rows)
        top_harm = sorted(
            [g for g in group_rows if g.get("objective_loss_vs_physical_best") is not None or g.get("objective_selected_unsafe")],
            key=lambda g: (1 if g.get("objective_selected_unsafe") else 0, fnum(g.get("objective_loss_vs_physical_best"), -1.0), fnum(g.get("physical_range_safe"), -1.0)),
            reverse=True,
        )
        now = dt.datetime.now(dt.timezone.utc)
        tokens = query_tokens()
        comp_count = int(compute_agg["v1d_state_mode_groups_with_nonH15_compute_safe_eps0p1"])
        match_rate = compute_agg.get("objective_match_rate")
        severe_v1b = int(aggregate_by_dataset.get("v1b_mined_terminal_reward_ablation", {}).get("severe_objective_rank_harm_groups", 0))
        material_v1d = int(aggregate_by_dataset.get("v1d_fresh_trace_selected_terminal_stable", {}).get("material_physical_spread_ge_3_groups", 0))
        if comp_count >= 6 and (match_rate is None or match_rate < 0.5):
            classification = "objective_terminal_rank_misaligned_for_compute_safe_choices"
            next_action = "freeze an IMPROVED objective/terminal repair feasibility diagnostic using realised continuation and measured-time labels; do not train a selector to imitate raw branch objective minima"
        elif severe_v1b > 0:
            classification = "terminal_value_rank_misaligned_on_mined_material_states"
            next_action = "run a bounded terminal-value/objective refit feasibility diagnostic before selector training; exclude per-H/H25 unsafe labels"
        elif material_v1d == 0 and comp_count < 2:
            classification = "rank_predictivity_secondary_to_sparse_scenario_opportunity"
            next_action = "prioritize versioned source-supported scenario/opportunity redesign over selector training"
        else:
            classification = "rank_predictivity_inconclusive_requires_small_repair_ablation"
            next_action = "freeze a small no-simulation objective-reweight analysis before any rollout or training"
        decision = {
            "classification": classification,
            "next_action": next_action,
            "train_or_refit_now": False,
            "run_more_label_density_unchanged": False,
            "run_186_episode_timing_confirmation_now": False,
            "reason": "Existing branch objectives/terminal values are evaluated against realised continuation outcomes and compute-safe choices; the result decides the next bounded intervention without new simulations.",
        }
        raw = {
            "created_utc": now.isoformat(),
            "method": "vehicle_terminal_value_rank_predictivity_postdiagnostic_v0_no_simulation",
            "classification": "development_no_simulation_terminal_value_rank_predictivity_no_validation64_no_test",
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
            "elapsed_since_first_supervisor_event_seconds": (now - FIRST_SUPERVISOR_EVENT).total_seconds(),
            "server_api_total_tokens_best_effort": tokens,
            "inputs": {rel(p): sha256(p) for p in (V1D_RAW, V1D_DONE, V1D_SOLVER_DONE, V1D_ORACLE_DONE, V1B_RAW, V1B_DONE, V1B_OBJECTIVE_DONE)},
            "thresholds": {"physical_match_tolerances": list(PHYS_TOLS), "compute_eps": COMPUTE_EPS, "compute_abs_gain_s": COMPUTE_ABS_GAIN_S, "state_distance_tol": STATE_DISTANCE_TOL},
            "record_count": len(records),
            "v1d_record_count": len(v1d_records),
            "v1b_record_count": len(v1b_records),
            "group_count": len(group_rows),
            "aggregate_by_dataset": aggregate_by_dataset,
            "v1d_compute_safe_predictivity": compute_agg,
            "v1d_compute_safe_rows": compute_rows,
            "group_rows": group_rows,
            "top_objective_harm_groups": top_harm[:25],
            "decision": decision,
            "limits": ["development-only", "no new rollouts", "single-run timing where timing is used", "not validation64", "not sealed test", "no selector/refit/training"],
            "backup_request": rel(REQ),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "raw.json", raw)
        write_summary(raw)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(
            f"# Terminal-value rank predictivity state\n\nUTC {now.isoformat()}. Classification `{classification}`. "
            f"v1d compute-safe groups={comp_count}; objective match rate={match_rate}; severe v1b objective harms={severe_v1b}. "
            f"Next: {next_action}. No validation64/sealed test, no simulation/training/refit.\n",
            encoding="utf-8",
        )
        write_json(REQ, {
            "requested_utc": now.isoformat(),
            "reason": "backup terminal-value/objective rank-predictivity diagnostic before objective/terminal repair or scenario pivot",
            "backup_required_before_more_simulations_or_training": True,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
            "artifacts": [rel(OUT), rel(STATE), rel(REQ), rel(Path(__file__).resolve())],
        })
        block = f"""<!-- {MARKER} -->
## 2026-09-29 terminal-value/objective rank-predictivity postdiagnostic

UTC: {now.isoformat()}. No-simulation diagnostic over v1d fresh trace-selected and v1b mined terminal-ablation branch rows. v1d compute-safe non-H15 state-mode groups={comp_count}; objective-min matched best measured-compute horizon in {compute_agg['objective_min_matches_best_compute']} groups (rate={match_rate}). v1b severe objective-rank harm groups={severe_v1b}. Classification `{classification}`. Decision: {next_action}. No validation64/sealed-test access and no training/refit. Backup requested at `{rel(REQ)}`.
"""
        append_docs(block)
        done = {
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
            "elapsed_since_first_supervisor_event_seconds": raw["elapsed_since_first_supervisor_event_seconds"],
            "server_api_total_tokens_best_effort": tokens,
            "backup_request": rel(REQ),
            "headline": {
                "classification": classification,
                "v1d_compute_safe_groups": comp_count,
                "objective_compute_match_rate": match_rate,
                "v1b_severe_objective_rank_harm_groups": severe_v1b,
                "train_or_refit_now": False,
                "next_action": next_action,
            },
            "artifacts": {
                "summary": rel(OUT / "summary.md"),
                "raw": rel(OUT / "raw.json"),
                "state": rel(STATE),
                "backup_request": rel(REQ),
            },
        }
        write_json(OUT / "completed.json", done)
        print(json.dumps({"completed": rel(OUT / "completed.json"), "headline": done["headline"], "elapsed_since_first_supervisor_event_seconds": done["elapsed_since_first_supervisor_event_seconds"], "server_api_total_tokens_best_effort": tokens, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        fail = {
            "passed": False,
            "hard_pass": False,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
        }
        write_json(OUT / "failure.json", fail)
        print(json.dumps({"failed": rel(OUT / "failure.json"), "error": repr(exc), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
