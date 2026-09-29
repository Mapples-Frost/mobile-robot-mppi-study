#!/usr/bin/env python3
"""Vehicle objective/terminal repair feasibility diagnostic v0.

Development-only, no-simulation diagnostic frozen in
research_artifacts/aws_protocols/vehicle_objective_terminal_repair_feasibility_v0_frozen_20260929T0410Z.md.

Question: can a simple held-state closed-form score using existing branch
objective/value/horizon features repair the rank mismatch seen in existing
branch banks, or does it collapse to horizon-only/constant behaviour and thus
argue against selector/refit training on current labels?
"""
from __future__ import annotations

import csv
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
STAMP = "20260929T0415Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/vehicle_objective_terminal_repair_feasibility_v0_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/vehicle_objective_terminal_repair_feasibility_v0_{STAMP}.md"
REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_OBJECTIVE_TERMINAL_REPAIR_FEASIBILITY_V0_{STAMP}.json"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_objective_terminal_repair_feasibility_v0_frozen_20260929T0410Z.md"
V1D_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/raw.json"
V1D_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/completed.json"
V1B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_full_20260928T2340Z_legacy_schema_repair/raw.json"
V1B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_full_20260928T2340Z_legacy_schema_repair/completed.json"
RANK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_terminal_value_rank_predictivity_postdiagnostic_v0_20260929T0425Z/completed.json"
FIRST_SUPERVISOR_EVENT = dt.datetime(2026, 9, 26, 10, 55, 29, 419331, tzinfo=dt.timezone.utc)
MARKER = f"vehicle-objective-terminal-repair-feasibility-v0-{STAMP}"
STATE_TOL = 1e-5
RIDGE_ALPHA = 1.0
LOSS_CAP = 1000.0
UNSAFE_PENALTY = 1000.0
COMPUTE_EPS = 0.1
COMPUTE_ABS_GAIN_S = 0.02
SELECTORS = [
    "raw_objective",
    "raw_value",
    "shortest_h",
    "h15_if_available",
    "horizon_preference_cv",
    "ridge_objective_value_horizon",
    "ridge_value_horizon",
    "ridge_horizon_only",
]
RIDGE_SELECTORS = {"ridge_objective_value_horizon", "ridge_value_horizon", "ridge_horizon_only"}


class ContractError(RuntimeError):
    pass


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def clean(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    return x


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        if isinstance(x, bool) or x is None:
            return default
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def inum(x: Any, default: int = 0) -> int:
    y = fnum(x)
    return int(y) if math.isfinite(y) else default


def first_finite(row: Mapping[str, Any], keys: Sequence[str]) -> float:
    for k in keys:
        if k in row:
            y = fnum(row.get(k))
            if math.isfinite(y):
                return y
    return float("nan")


def objective(row: Mapping[str, Any]) -> float:
    return first_finite(row, ["branch_objective_opt_f_num", "objective", "opt_f_num", "mpc_objective"])


def value_fn(row: Mapping[str, Any]) -> float:
    return first_finite(row, ["branch_mpc_value_fn", "value_fn", "mpc_value_fn", "terminal_value"])


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
        raise ContractError(f"missing completed marker {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True and obj.get("status") not in ("complete", "completed"):
        raise ContractError(f"non-passing completed marker {rel(path)}")
    for flag in ("validation64_bank_opened", "historical_validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=True in {rel(path)}")
    return obj


def check_raw_flags(name: str, obj: Mapping[str, Any]) -> None:
    for flag in ("validation64_bank_opened", "historical_validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=True in {name}")


def safe_record(r: Mapping[str, Any]) -> bool:
    checks = [
        bool(r.get("success", False)),
        not bool(r.get("constraint", False)),
        inum(r.get("solver_failure_steps"), 0) == 0,
        inum(r.get("initial_failed_steps"), 0) == 0,
        inum(r.get("final_failed_steps"), 0) == 0,
    ]
    if "branch_reached" in r:
        checks.append(bool(r.get("branch_reached")))
    if "physical_prefix_matches_H15" in r:
        checks.append(bool(r.get("physical_prefix_matches_H15")))
    if "no_success_constraint_solver_regression_vs_H15" in r:
        checks.append(bool(r.get("no_success_constraint_solver_regression_vs_H15")))
    if "no_safety_solver_regression_vs_H15" in r:
        checks.append(bool(r.get("no_safety_solver_regression_vs_H15")))
    for key in ("state_distance_vs_H15", "branch_state_distance_vs_H15", "branch_state_distance_vs_per_h_H15"):
        if key in r:
            checks.append(fnum(r.get(key), 1e99) <= STATE_TOL)
    return all(checks)


def parse_v1d(raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for st in ((raw.get("analysis") or {}).get("state_rows") or []):
        for mode, mobj in ((st.get("mode_results") or {}).items()):
            for c in (mobj.get("comparisons") or []):
                case_val = st.get("case") if st.get("case") is not None else c.get("case")
                rec = {
                    "dataset": "v1d_fresh_trace_selected_terminal_stable",
                    "state_id": str(st.get("state_id", c.get("state_id"))),
                    "terminal_mode": str(mode),
                    "case": inum(case_val, -1),
                    "branch_step": inum(st.get("branch_step", c.get("branch_step")), -1),
                    "selection_group": st.get("selection_group"),
                    "state_role": st.get("target_role"),
                    "horizon": inum(c.get("horizon"), -1),
                    "continuation_physical": fnum(c.get("continuation_physical")),
                    "continuation_total": fnum(c.get("continuation_total")),
                    "decision_sum_s": fnum(c.get("decision_sum_s")),
                    "solver_attempt_sum_s": fnum(c.get("solver_attempt_sum_s")),
                    "objective": objective(c),
                    "value_fn": value_fn(c),
                    "success": bool(c.get("success", False)),
                    "constraint": bool(c.get("constraint", False)),
                    "solver_failure_steps": inum(c.get("solver_failure_steps"), 0),
                    "initial_failed_steps": inum(c.get("initial_failed_steps"), 0),
                    "final_failed_steps": inum(c.get("final_failed_steps"), 0),
                    "branch_reached": bool(c.get("branch_reached", True)),
                    "physical_prefix_matches_H15": bool(c.get("physical_prefix_matches_H15", True)),
                    "no_success_constraint_solver_regression_vs_H15": bool(c.get("no_success_constraint_solver_regression_vs_H15", True)),
                    "state_distance_vs_H15": fnum(c.get("state_distance_vs_H15"), 0.0),
                    "path": c.get("path"),
                }
                rec["safe"] = safe_record(rec)
                rows.append(rec)
    return [r for r in rows if r["horizon"] >= 0]


def parse_v1b(raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in ((raw.get("analysis") or {}).get("comparison_rows") or []):
        rec = {
            "dataset": "v1b_mined_terminal_reward_ablation",
            "state_id": str(r.get("state_id")),
            "terminal_mode": str(r.get("terminal_mode")),
            "case": inum(r.get("case"), -1),
            "branch_step": inum(r.get("branch_step"), -1),
            "selection_group": r.get("selection_group"),
            "state_role": r.get("state_role", r.get("role")),
            "horizon": inum(r.get("horizon"), -1),
            "continuation_physical": fnum(r.get("continuation_physical")),
            "continuation_total": fnum(r.get("continuation_total")),
            "decision_sum_s": fnum(r.get("decision_sum_s", r.get("decision_plus_terminal_switch_sum_s"))),
            "solver_attempt_sum_s": fnum(r.get("solver_attempt_sum_s")),
            "objective": objective(r),
            "value_fn": value_fn(r),
            "success": bool(r.get("success", False)),
            "constraint": bool(r.get("constraint", False)),
            "solver_failure_steps": inum(r.get("solver_failure_steps"), 0),
            "initial_failed_steps": inum(r.get("initial_failed_steps"), 0),
            "final_failed_steps": inum(r.get("final_failed_steps"), 0),
            "no_safety_solver_regression_vs_H15": bool(r.get("no_safety_solver_regression_vs_H15", True)),
            "branch_state_distance_vs_per_h_H15": fnum(r.get("branch_state_distance_vs_per_h_H15", r.get("branch_state_distance_vs_H15")), 0.0),
            "path": r.get("path"),
        }
        rec["safe"] = safe_record(rec)
        rows.append(rec)
    return [r for r in rows if r["horizon"] >= 0]


def group_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        buckets[(str(r["dataset"]), str(r["state_id"]), str(r["terminal_mode"]))].append(r)
    groups: List[Dict[str, Any]] = []
    for key, rows in buckets.items():
        rs = sorted(rows, key=lambda r: int(r["horizon"]))
        safe_rs = [r for r in rs if bool(r.get("safe")) and math.isfinite(fnum(r.get("continuation_physical")))]
        best = min((fnum(r.get("continuation_physical")) for r in safe_rs), default=float("nan"))
        groups.append({
            "key": "|".join(key),
            "dataset": key[0],
            "state_id": key[1],
            "terminal_mode": key[2],
            "case": inum(rs[0].get("case"), -1) if rs else -1,
            "branch_step": inum(rs[0].get("branch_step"), -1) if rs else -1,
            "state_role": rs[0].get("state_role") if rs else None,
            "rows": rs,
            "safe_count": len(safe_rs),
            "best_safe_physical": best,
            "evaluable": bool(len(rs) >= 2 and safe_rs and math.isfinite(best)),
        })
    return groups


def row_loss(row: Mapping[str, Any], best: float) -> float:
    if not bool(row.get("safe")) or not math.isfinite(fnum(row.get("continuation_physical"))) or not math.isfinite(best):
        return UNSAFE_PENALTY
    return min(LOSS_CAP, max(0.0, fnum(row.get("continuation_physical")) - best))


def feature_values(row: Mapping[str, Any], selector: str) -> List[float]:
    h = fnum(row.get("horizon"), 0.0)
    obj = fnum(row.get("objective"), 0.0)
    val = fnum(row.get("value_fn"), 0.0)
    if not math.isfinite(obj):
        obj = 0.0
    if not math.isfinite(val):
        val = 0.0
    if selector == "ridge_objective_value_horizon":
        return [obj, val, h, (h * h) / 100.0]
    if selector == "ridge_value_horizon":
        return [val, h, (h * h) / 100.0]
    if selector == "ridge_horizon_only":
        return [h, (h * h) / 100.0]
    raise ValueError(selector)


def solve_linear(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        div = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            fac = M[r][col]
            if fac == 0.0:
                continue
            for j in range(col, n + 1):
                M[r][j] -= fac * M[col][j]
    return [M[i][n] for i in range(n)]


def fit_ridge(training_groups: Sequence[Mapping[str, Any]], selector: str) -> Optional[Dict[str, Any]]:
    xs: List[List[float]] = []
    ys: List[float] = []
    for g in training_groups:
        if not g.get("evaluable"):
            continue
        best = fnum(g.get("best_safe_physical"))
        for r in g["rows"]:
            xs.append(feature_values(r, selector))
            ys.append(row_loss(r, best))
    if len(xs) < 4:
        return None
    p = len(xs[0])
    means = [sum(x[j] for x in xs) / len(xs) for j in range(p)]
    stds = []
    for j in range(p):
        var = sum((x[j] - means[j]) ** 2 for x in xs) / max(1, len(xs) - 1)
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    X = [[1.0] + [(x[j] - means[j]) / stds[j] for j in range(p)] for x in xs]
    q = p + 1
    A = [[0.0 for _ in range(q)] for _ in range(q)]
    b = [0.0 for _ in range(q)]
    for x, y in zip(X, ys):
        for i in range(q):
            b[i] += x[i] * y
            for j in range(q):
                A[i][j] += x[i] * x[j]
    for i in range(q):
        A[i][i] += RIDGE_ALPHA if i > 0 else 1e-9
    beta = solve_linear(A, b)
    if beta is None:
        return None
    return {"selector": selector, "means": means, "stds": stds, "beta": beta, "training_rows": len(xs), "training_groups": len([g for g in training_groups if g.get("evaluable")])}


def predict(model: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    selector = str(model["selector"])
    vals = feature_values(row, selector)
    means = list(model["means"]); stds = list(model["stds"]); beta = list(model["beta"])
    z = [1.0] + [(vals[j] - means[j]) / (stds[j] if stds[j] else 1.0) for j in range(len(vals))]
    return sum(beta[i] * z[i] for i in range(len(beta)))


def fit_horizon_preference(training_groups: Sequence[Mapping[str, Any]]) -> Dict[int, float]:
    vals: Dict[int, List[float]] = defaultdict(list)
    for g in training_groups:
        if not g.get("evaluable"):
            continue
        best = fnum(g.get("best_safe_physical"))
        for r in g["rows"]:
            vals[int(r["horizon"])].append(row_loss(r, best))
    return {h: (sum(v) / len(v)) for h, v in vals.items() if v}


def choose(group: Mapping[str, Any], selector: str, training_groups: Sequence[Mapping[str, Any]], fit_counter: Dict[str, int]) -> Optional[Mapping[str, Any]]:
    rows = list(group["rows"])
    if not rows:
        return None
    if selector == "raw_objective":
        cand = [r for r in rows if math.isfinite(fnum(r.get("objective")))]
        return min(cand or rows, key=lambda r: (fnum(r.get("objective"), 1e99), abs(int(r["horizon"]) - 15), int(r["horizon"])))
    if selector == "raw_value":
        cand = [r for r in rows if math.isfinite(fnum(r.get("value_fn")))]
        return min(cand or rows, key=lambda r: (fnum(r.get("value_fn"), 1e99), abs(int(r["horizon"]) - 15), int(r["horizon"])))
    if selector == "shortest_h":
        return min(rows, key=lambda r: int(r["horizon"]))
    if selector == "h15_if_available":
        return min(rows, key=lambda r: (abs(int(r["horizon"]) - 15), int(r["horizon"])))
    if selector == "horizon_preference_cv":
        pref = fit_horizon_preference(training_groups)
        fit_counter[selector] += 1
        return min(rows, key=lambda r: (pref.get(int(r["horizon"]), 999.0), abs(int(r["horizon"]) - 15), int(r["horizon"])))
    if selector in RIDGE_SELECTORS:
        model = fit_ridge(training_groups, selector)
        fit_counter[selector] += 1
        if model is None:
            return min(rows, key=lambda r: (abs(int(r["horizon"]) - 15), int(r["horizon"])))
        return min(rows, key=lambda r: (predict(model, r), abs(int(r["horizon"]) - 15), int(r["horizon"])))
    raise ValueError(selector)


def evaluate(groups: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    eval_groups = [g for g in groups if g.get("evaluable")]
    fit_counter: Dict[str, int] = defaultdict(int)
    out: List[Dict[str, Any]] = []
    for g in eval_groups:
        training = [x for x in groups if x.get("dataset") == g.get("dataset") and x.get("state_id") != g.get("state_id") and x.get("evaluable")]
        if not training:
            training = [x for x in groups if x.get("state_id") != g.get("state_id") and x.get("evaluable")]
        for sel in SELECTORS:
            row = choose(g, sel, training, fit_counter)
            best = fnum(g.get("best_safe_physical"))
            safe = bool(row and row.get("safe") and math.isfinite(fnum(row.get("continuation_physical"))))
            loss = (fnum(row.get("continuation_physical")) - best) if safe else None
            out.append({
                "group_key": g["key"],
                "dataset": g["dataset"],
                "state_id": g["state_id"],
                "terminal_mode": g["terminal_mode"],
                "selector": sel,
                "selected_horizon": None if row is None else int(row["horizon"]),
                "selected_safe": safe,
                "selected_success": None if row is None else bool(row.get("success")),
                "selected_physical": None if row is None or not math.isfinite(fnum(row.get("continuation_physical"))) else fnum(row.get("continuation_physical")),
                "best_safe_physical": best,
                "loss_vs_best_safe": loss,
                "unsafe_or_missing": not safe,
                "within_0p1": bool(safe and loss is not None and loss <= 0.1),
                "within_3": bool(safe and loss is not None and loss <= 3.0),
                "severe_harm": bool((not safe) or (loss is not None and loss >= 3.0)),
                "objective": None if row is None or not math.isfinite(fnum(row.get("objective"))) else fnum(row.get("objective")),
                "value_fn": None if row is None or not math.isfinite(fnum(row.get("value_fn"))) else fnum(row.get("value_fn")),
            })
    return out, dict(fit_counter)


def aggregate_eval(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for scope in ["all"] + sorted(set(str(r["dataset"]) for r in rows)):
        scope_rows = list(rows) if scope == "all" else [r for r in rows if r["dataset"] == scope]
        out[scope] = {}
        for sel in SELECTORS:
            rs = [r for r in scope_rows if r["selector"] == sel]
            losses = [fnum(r.get("loss_vs_best_safe")) for r in rs if r.get("loss_vs_best_safe") is not None]
            out[scope][sel] = {
                "group_count": len(rs),
                "safe_selected": int(sum(1 for r in rs if r.get("selected_safe"))),
                "unsafe_or_missing": int(sum(1 for r in rs if r.get("unsafe_or_missing"))),
                "within_0p1": int(sum(1 for r in rs if r.get("within_0p1"))),
                "within_3": int(sum(1 for r in rs if r.get("within_3"))),
                "severe_harm": int(sum(1 for r in rs if r.get("severe_harm"))),
                "loss_summary": stats(losses),
                "horizon_counts": dict(Counter(str(r.get("selected_horizon")) for r in rs)),
            }
    return out


def compute_safe_groups(groups: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        if g.get("dataset") != "v1d_fresh_trace_selected_terminal_stable" or not g.get("evaluable"):
            continue
        by_h = {int(r["horizon"]): r for r in g["rows"]}
        ref = by_h.get(15)
        if ref is None or not ref.get("safe"):
            continue
        ref_phys = fnum(ref.get("continuation_physical")); ref_dec = fnum(ref.get("decision_sum_s")); ref_sol = fnum(ref.get("solver_attempt_sum_s"))
        if not (math.isfinite(ref_phys) and math.isfinite(ref_dec) and math.isfinite(ref_sol)):
            continue
        candidates = []
        for h, r in by_h.items():
            if h == 15 or not r.get("safe"):
                continue
            phys_loss = fnum(r.get("continuation_physical")) - ref_phys
            dec_gain = ref_dec - fnum(r.get("decision_sum_s"))
            sol_gain = ref_sol - fnum(r.get("solver_attempt_sum_s"))
            if phys_loss <= COMPUTE_EPS and dec_gain >= COMPUTE_ABS_GAIN_S and sol_gain >= COMPUTE_ABS_GAIN_S:
                candidates.append({"horizon": h, "row": r, "phys_loss_vs_H15": phys_loss, "decision_gain_vs_H15_s": dec_gain, "solver_gain_vs_H15_s": sol_gain})
        if candidates:
            best = min(candidates, key=lambda c: (fnum(c["row"].get("decision_sum_s")), fnum(c["row"].get("solver_attempt_sum_s")), int(c["horizon"])))
            out[g["key"]] = {"group": g, "ref": ref, "candidates": candidates, "best": best}
    return out


def aggregate_compute(eval_rows: Sequence[Mapping[str, Any]], comp_groups: Mapping[str, Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    group_map = {g["key"]: g for g in groups}
    out: Dict[str, Any] = {}
    for sel in SELECTORS:
        rs = [r for r in eval_rows if r["selector"] == sel and r["group_key"] in comp_groups]
        matches = 0; selected_safe_compute = 0; regrets_dec: List[float] = []; regrets_sol: List[float] = []; phys_losses: List[float] = []
        selected_h = []
        for er in rs:
            cg = comp_groups[er["group_key"]]
            best_row = cg["best"]["row"]
            best_h = int(cg["best"]["horizon"])
            h = er.get("selected_horizon")
            selected_h.append(str(h))
            if h == best_h:
                matches += 1
            g = group_map[er["group_key"]]
            row = next((x for x in g["rows"] if int(x["horizon"]) == int(h)), None) if h is not None else None
            ref = cg["ref"]
            if row is None or not row.get("safe"):
                continue
            phys_loss = fnum(row.get("continuation_physical")) - fnum(ref.get("continuation_physical"))
            dec_gain = fnum(ref.get("decision_sum_s")) - fnum(row.get("decision_sum_s"))
            sol_gain = fnum(ref.get("solver_attempt_sum_s")) - fnum(row.get("solver_attempt_sum_s"))
            if phys_loss <= COMPUTE_EPS and dec_gain >= COMPUTE_ABS_GAIN_S and sol_gain >= COMPUTE_ABS_GAIN_S:
                selected_safe_compute += 1
            regrets_dec.append(fnum(row.get("decision_sum_s")) - fnum(best_row.get("decision_sum_s")))
            regrets_sol.append(fnum(row.get("solver_attempt_sum_s")) - fnum(best_row.get("solver_attempt_sum_s")))
            phys_losses.append(phys_loss - fnum(cg["best"].get("phys_loss_vs_H15")))
        out[sel] = {
            "compute_group_count": len(rs),
            "matches_best_compute": matches,
            "match_rate": (matches / len(rs)) if rs else None,
            "selected_compute_safe_count": selected_safe_compute,
            "selected_horizon_counts": dict(Counter(selected_h)),
            "decision_regret_vs_best_s": stats(regrets_dec),
            "solver_regret_vs_best_s": stats(regrets_sol),
            "physical_extra_loss_vs_best_compute": stats(phys_losses),
        }
    return out


def query_tokens() -> Dict[str, Any]:
    for db in (ROOT / "research.sqlite", ROOT / "research_artifacts/research.sqlite", ROOT.parent / "research.sqlite"):
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db)); cur = con.cursor(); total = 0; by_table = {}; found = False
            for (table,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                if "total_tokens" in cols:
                    val = int(cur.execute(f"SELECT COALESCE(SUM(total_tokens),0) FROM {table}").fetchone()[0] or 0)
                    total += val; by_table[table] = val; found = True
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


def append_registry(created: str, completed: Path) -> None:
    p = ROOT / "EXPERIMENT_REGISTRY.csv"
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    if rel(completed) in existing:
        return
    row = {
        "experiment_id": "",
        "timestamp": created,
        "method": "vehicle_objective_terminal_repair_feasibility_v0_no_simulation_closed_form",
        "seed": "deterministic_leave_state_out_existing_branch_rows",
        "split": "development_existing_v1d_v1b_no_validation64_no_test",
        "commit_sha": "",
        "status": "completed_no_simulation_diagnostic",
        "exit_status": "0",
        "runtime_seconds": "",
        "peak_process_rss_kb": "",
        "record": rel(completed),
    }
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not existing.strip():
            writer.writeheader()
        writer.writerow(row)


def write_summary(raw: Mapping[str, Any]) -> None:
    agg = raw["aggregate_eval"]["all"]
    comp = raw["compute_safe_eval"]
    lines = [
        "# Vehicle objective/terminal repair feasibility v0",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only no-simulation closed-form diagnostic; validation64 and sealed test stayed closed.",
        "",
        "## Headline",
        "",
        f"- Evaluable state-mode groups: `{raw['evaluable_group_count']}/{raw['group_count']}` from `{raw['record_count']}` horizon rows.",
        f"- v1d compute-safe groups reused from existing single-run branch timing: `{raw['compute_safe_group_count']}` (not a speed claim).",
        f"- Closed-form diagnostic fits: `{raw['diagnostic_closed_form_fit_counts']}`; no controller/value checkpoint was trained or saved.",
        f"- Decision: `{raw['decision']['classification']}`. Next: {raw['decision']['next_action']}.",
        "",
        "## Held-state physical ranking",
        "",
        "| selector | groups | unsafe/missing | within 0.1 | within 3 | severe harm | loss mean | loss median | loss max | horizons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sel in SELECTORS:
        a = agg[sel]; s = a["loss_summary"]
        lines.append("| `%s` | %d | %d | %d | %d | %d | %s | %s | %s | `%s` |" % (
            sel, a["group_count"], a["unsafe_or_missing"], a["within_0p1"], a["within_3"], a["severe_harm"],
            "" if s["mean"] is None else "%.6g" % s["mean"],
            "" if s["median"] is None else "%.6g" % s["median"],
            "" if s["max"] is None else "%.6g" % s["max"],
            a["horizon_counts"],
        ))
    lines += [
        "",
        "## v1d compute-safe ranking (single-run timing; no acceleration claim)",
        "",
        "| selector | groups | match best compute | match rate | selected compute-safe | decision regret median/sum s | selected horizons |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for sel in SELECTORS:
        c = comp[sel]; d = c["decision_regret_vs_best_s"]
        lines.append("| `%s` | %d | %d | %s | %d | %s/%s | `%s` |" % (
            sel, c["compute_group_count"], c["matches_best_compute"],
            "" if c["match_rate"] is None else "%.1f%%" % (100.0 * c["match_rate"]),
            c["selected_compute_safe_count"],
            "" if d["median"] is None else "%.6g" % d["median"],
            "" if d["sum"] is None else "%.6g" % d["sum"],
            c["selected_horizon_counts"],
        ))
    lines += [
        "",
        "## Interpretation",
        "",
        "- SCENARIOS: this does not create new scenario evidence; it tests whether current development branch labels are learnably rankable. Sparse/concentrated opportunity remains preserved.",
        "- REWARD/TERMINAL/VALUE: raw branch objective/value are compared against realised continuation labels; simple repair success or failure decides whether terminal/objective refit is worth a later fresh confirmation.",
        "- TRAINING: only closed-form leave-state-out diagnostic fits were run; there were zero new gradient steps, training episodes, rollouts, or persistent checkpoints.",
        "- COMPARISONS: horizon-only and H15/shortest baselines are included to avoid mistaking a constant-H preference for adaptive learning. Fair fixed-H/Pareto validation remains required for any future method.",
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
        for p in (PROTOCOL, V1D_RAW, V1D_DONE, V1B_RAW, V1B_DONE, RANK_DONE):
            if not p.exists():
                raise ContractError(f"missing input {rel(p)}")
        for p in (V1D_DONE, V1B_DONE, RANK_DONE):
            verify_completed(p)
        v1d_raw = read_json(V1D_RAW); v1b_raw = read_json(V1B_RAW)
        check_raw_flags("v1d_raw", v1d_raw); check_raw_flags("v1b_raw", v1b_raw)
        records = parse_v1d(v1d_raw) + parse_v1b(v1b_raw)
        groups = group_records(records)
        if not records or not groups:
            raise ContractError("no records/groups parsed")
        eval_rows, fit_counts = evaluate(groups)
        agg = aggregate_eval(eval_rows)
        comp_groups = compute_safe_groups(groups)
        comp_eval = aggregate_compute(eval_rows, comp_groups, groups)
        # Compact pairwise comparisons for the primary proposed repair.
        ridge = agg["all"]["ridge_objective_value_horizon"]
        raw_obj = agg["all"]["raw_objective"]
        horizon_only = agg["all"]["ridge_horizon_only"]
        shortest = agg["all"]["shortest_h"]
        ridge_comp = comp_eval["ridge_objective_value_horizon"]
        raw_comp = comp_eval["raw_objective"]
        hcomp = comp_eval["ridge_horizon_only"]
        ridge_mean = fnum(ridge["loss_summary"].get("mean"), 1e99)
        raw_mean = fnum(raw_obj["loss_summary"].get("mean"), 1e99)
        hmean = fnum(horizon_only["loss_summary"].get("mean"), 1e99)
        ridge_match = fnum(ridge_comp.get("match_rate"), -1.0)
        raw_match = fnum(raw_comp.get("match_rate"), -1.0)
        hmatch = fnum(hcomp.get("match_rate"), -1.0)
        repair_beats_raw = (ridge["unsafe_or_missing"] <= raw_obj["unsafe_or_missing"] and ridge["severe_harm"] < raw_obj["severe_harm"] and ridge_mean < raw_mean)
        repair_beats_horizon = (ridge["severe_harm"] <= horizon_only["severe_harm"] and ridge_mean <= hmean and ridge_match >= hmatch)
        compute_improved = (ridge_match >= raw_match + 0.15 and ridge_match >= hmatch + 0.05 and ridge_comp["selected_compute_safe_count"] >= raw_comp["selected_compute_safe_count"])
        if repair_beats_raw and repair_beats_horizon and compute_improved:
            classification = "simple_objective_terminal_repair_promising_development_only"
            next_action = "after backup, freeze a compact IMPROVED terminal/objective refit smoke on fresh confirmation states with fair fixed-H/Pareto baselines"
            train_next = True
        elif repair_beats_raw and not compute_improved:
            classification = "physical_rank_repair_partial_but_compute_tradeoff_not_supported"
            next_action = "do not train compute-safe selector from current labels; inspect richer terminal representation or scenario/opportunity amendment before any refit"
            train_next = False
        else:
            classification = "simple_objective_terminal_repair_not_supported_from_current_labels"
            next_action = "do not train/refit a selector from current v1d/v1b labels; pivot to versioned scenario/opportunity redesign or richer terminal-value representation diagnostic"
            train_next = False
        now = dt.datetime.now(dt.timezone.utc)
        raw = {
            "created_utc": now.isoformat(),
            "method": "vehicle_objective_terminal_repair_feasibility_v0_no_simulation_closed_form_leave_state_out",
            "protocol": rel(PROTOCOL),
            "classification": "development_no_simulation_closed_form_objective_terminal_repair_no_validation64_no_test",
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
            "new_persistent_refit_steps": 0,
            "diagnostic_closed_form_fit_counts": fit_counts,
            "elapsed_since_first_supervisor_event_seconds": (now - FIRST_SUPERVISOR_EVENT).total_seconds(),
            "server_api_total_tokens_best_effort": query_tokens(),
            "inputs": {rel(p): sha256(p) for p in (PROTOCOL, V1D_RAW, V1D_DONE, V1B_RAW, V1B_DONE, RANK_DONE)},
            "thresholds": {"ridge_alpha": RIDGE_ALPHA, "loss_cap": LOSS_CAP, "unsafe_penalty": UNSAFE_PENALTY, "compute_eps": COMPUTE_EPS, "compute_abs_gain_s": COMPUTE_ABS_GAIN_S, "state_tol": STATE_TOL},
            "record_count": len(records),
            "group_count": len(groups),
            "evaluable_group_count": int(sum(1 for g in groups if g.get("evaluable"))),
            "skipped_no_safe_best_group_count": int(sum(1 for g in groups if not g.get("evaluable"))),
            "compute_safe_group_count": len(comp_groups),
            "aggregate_eval": agg,
            "compute_safe_eval": comp_eval,
            "compute_safe_group_keys": sorted(comp_groups.keys()),
            "selector_evaluations": eval_rows,
            "decision": {"classification": classification, "next_action": next_action, "train_or_refit_next": train_next, "reason": "Held-state closed-form repair was compared with raw objective/value and horizon-only/constant baselines using realised continuation outcomes; no validation/test or new simulation was used."},
            "backup_request": rel(REQ),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "raw.json", raw)
        write_summary(raw)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(
            f"# Objective/terminal repair feasibility v0 state\n\nUTC {now.isoformat()}. Classification `{classification}`. "
            f"Ridge severe={ridge['severe_harm']} mean_loss={ridge_mean}; raw severe={raw_obj['severe_harm']} mean_loss={raw_mean}; "
            f"horizon-only severe={horizon_only['severe_harm']} mean_loss={hmean}; compute match ridge/raw/horizon={ridge_match}/{raw_match}/{hmatch}. "
            f"Next: {next_action}. No validation64/sealed test; no rollout/training/persistent refit.\n",
            encoding="utf-8",
        )
        write_json(REQ, {
            "requested_utc": now.isoformat(),
            "reason": "backup objective/terminal repair feasibility diagnostic before scenario/value next intervention",
            "backup_required_before_more_simulations_or_training": True,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_persistent_refit_steps": 0,
            "artifacts": [rel(OUT), rel(STATE), rel(REQ), rel(Path(__file__).resolve()), rel(PROTOCOL)],
        })
        block = f"""<!-- {MARKER} -->
## 2026-09-29 objective/terminal repair feasibility v0

UTC: {now.isoformat()}. Ran frozen no-simulation leave-state-out closed-form diagnostic over existing v1d/v1b development branch rows. Classification `{classification}`. Ridge objective/value/horizon severe harms={ridge['severe_harm']} mean loss={ridge_mean:.6g}; raw objective severe harms={raw_obj['severe_harm']} mean loss={raw_mean:.6g}; horizon-only severe harms={horizon_only['severe_harm']} mean loss={hmean:.6g}. v1d compute-safe match rates ridge/raw/horizon={ridge_match}/{raw_match}/{hmatch}. Decision: {next_action}. No validation64/sealed-test access, no rollout/training/persistent refit. Backup requested at `{rel(REQ)}`.
"""
        append_docs(block)
        completed = OUT / "completed.json"
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
            "new_persistent_refit_steps": 0,
            "diagnostic_closed_form_fit_counts": fit_counts,
            "elapsed_since_first_supervisor_event_seconds": raw["elapsed_since_first_supervisor_event_seconds"],
            "server_api_total_tokens_best_effort": raw["server_api_total_tokens_best_effort"],
            "backup_request": rel(REQ),
            "headline": {
                "classification": classification,
                "evaluable_groups": raw["evaluable_group_count"],
                "compute_safe_groups": len(comp_groups),
                "ridge_severe_harm": ridge["severe_harm"],
                "raw_objective_severe_harm": raw_obj["severe_harm"],
                "horizon_only_severe_harm": horizon_only["severe_harm"],
                "ridge_mean_loss": ridge_mean,
                "raw_objective_mean_loss": raw_mean,
                "ridge_compute_match_rate": ridge_match,
                "raw_objective_compute_match_rate": raw_match,
                "train_or_refit_next": train_next,
                "next_action": next_action,
            },
            "artifacts": {"summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "state": rel(STATE), "backup_request": rel(REQ)},
        }
        write_json(completed, done)
        append_registry(now.isoformat(), completed)
        print(json.dumps({"completed": rel(completed), "headline": done["headline"], "elapsed_since_first_supervisor_event_seconds": done["elapsed_since_first_supervisor_event_seconds"], "server_api_total_tokens_best_effort": done["server_api_total_tokens_best_effort"], "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
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
            "new_persistent_refit_steps": 0,
        }
        write_json(OUT / "failure.json", fail)
        print(json.dumps({"failed": rel(OUT / "failure.json"), "error": repr(exc), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
