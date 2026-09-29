#!/usr/bin/env python3
"""Development-only cost-sensitive delta-value LOBO diagnostic for true-variable-H H10/H15.

This is a bounded, no-simulation closed-form refit diagnostic motivated by the
fresh-source v2 failure and v3b negative selector-only evidence.  It asks whether
online/pre-decision features can support a calibrated physical-risk and measured
decision-time model, rather than another nearest-neighbour label-density sweep.

Inputs are ONLY already-opened development banks fresh_v0/fresh_v1/fresh_v2.
No validation64 or sealed-test files are read.  No MPC rollout, gradient training,
or checkpoint saving is performed.  The fitted linear/ridge models are diagnostic
only and development-contaminated; any positive result would require a separately
frozen unused fresh-source confirmation before validation64.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_delta_value_model_v4_lobo"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_delta_value_model_v4_lobo.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
PRIMARY_PROFILE = "shared_h15_terminal"
MATCHED_PROFILE = "matched_terminal"
BANK_DIRS = {
    "fresh_v0": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z",
    "fresh_v1": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z",
    "fresh_v2": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z",
}
MARKER = f"vehicle-true-variable-H-delta-value-model-v4-lobo-{STAMP}"

FEATURE_SETS = [
    "obs14_abs_l2",                 # online observation only, normalized
    "obs14_pose_step_std",          # observation + robot pose + branch step
    "obs14_abs_pose_step_std",      # richer deployable representation
    "obs14_obstacle_pose_step_std", # obstacle slice emphasis
]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
CAT_WEIGHTS = [1.0, 4.0, 10.0]
RISK_THRESHOLDS = [-0.25, 0.0, 0.5, 1.0, 2.0]
CAT_THRESHOLDS = [0.20, 0.35, 0.50, 0.75]
GAIN_THRESHOLDS = [0.0, 0.10, 0.25]
MAX_PRED_DELTA_TARGET = 50.0
MIN_ROW_TOL = 2.0


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
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    return x


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def si(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def median_summary(medians: Mapping[str, Any], h: int) -> Mapping[str, Any]:
    return medians.get(str(h)) or medians.get(h) or {}


def obs14(st: Mapping[str, Any]) -> List[float]:
    raw = st.get("initial_observation_from_h15_trace") or st.get("initial_observation_at_branch")
    vals = [sf(v, 0.0) for v in raw[:14]] if isinstance(raw, list) else []
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def norm_l2(vals: Sequence[float]) -> List[float]:
    xs = [sf(v, 0.0) for v in vals]
    n = math.sqrt(sum(v * v for v in xs))
    return [v / n for v in xs] if n > 1e-12 else xs


def assert_dev_only(raw: Mapping[str, Any], done: Mapping[str, Any], bank: str) -> None:
    for name, obj in [("raw", raw), ("completed", done)]:
        for flag in ("sealed_test_accessed", "sealed_test_bank_opened", "validation64_bank_opened", "historical_validation64_bank_opened"):
            if obj.get(flag) is True:
                raise ContractError(f"forbidden {flag}=True in {bank} {name}")
    if done.get("passed") is not True and done.get("hard_pass") is not True and done.get("status") not in ("complete", "completed"):
        raise ContractError(f"bank {bank} did not complete cleanly")


def load_manifest(raw: Mapping[str, Any], path: Path) -> List[Mapping[str, Any]]:
    man = raw.get("selected_state_manifest")
    if isinstance(man, Mapping) and isinstance(man.get("selected_states"), list):
        return list(man["selected_states"])
    if isinstance(man, list):
        return list(man)
    if path.exists():
        obj = read_json(path)
        if isinstance(obj, Mapping):
            return list(obj.get("selected_states") or [])
        if isinstance(obj, list):
            return list(obj)
    raise ContractError("missing selected_state_manifest at " + rel(path))


def load_rows() -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    rows: List[Dict[str, Any]] = []
    hashes: Dict[str, str] = {rel(Path(__file__)): sha256(Path(__file__))}
    for bank, d in BANK_DIRS.items():
        raw_path = d / "raw.json"
        done_path = d / "completed.json"
        man_path = d / "selected_state_manifest.json"
        raw = read_json(raw_path); done = read_json(done_path)
        assert_dev_only(raw, done, bank)
        hashes[rel(raw_path)] = sha256(raw_path); hashes[rel(done_path)] = sha256(done_path)
        if man_path.exists():
            hashes[rel(man_path)] = sha256(man_path)
        by_base = {str(s.get("base_state_id")): s for s in load_manifest(raw, man_path)}
        for r in ((raw.get("analysis") or {}).get("state_profile_rows") or []):
            if str(r.get("terminal_profile")) != PRIMARY_PROFILE:
                continue
            base = str(r.get("base_state_id") or r.get("state_id"))
            st = by_base.get(base, {})
            med = r.get("median_by_h") or {}
            h10 = median_summary(med, 10); h15 = median_summary(med, 15)
            h10_phys = sf(h10.get("physical")); h15_phys = sf(h15.get("physical"))
            h10_dec = sf(h10.get("decision_sum_s")); h15_dec = sf(h15.get("decision_sum_s"))
            h10_sol = sf(h10.get("solver_sum_s")); h15_sol = sf(h15.get("solver_sum_s"))
            tol = max(MIN_ROW_TOL, sf(r.get("row_physical_tolerance_vs_H15", r.get("row_physical_tolerance")), MIN_ROW_TOL))
            prev = st.get("branch_previous_state") if isinstance(st.get("branch_previous_state"), Mapping) else {}
            h10_safe = bool(h10.get("safe_all")); h15_safe = bool(h15.get("safe_all"))
            beneficial = bool(r.get("h10_beneficial_vs_h15")) if "h10_beneficial_vs_h15" in r else bool(h10_safe and h15_safe and (h10_phys - h15_phys <= tol) and h10_dec < h15_dec)
            catastrophic = bool((not h10_safe) or (h15_safe and (h10_phys - h15_phys > tol)))
            rows.append({
                "bank_id": bank,
                "base_state_id": base,
                "fresh_case_index": r.get("fresh_case_index"),
                "source_candidate_index": st.get("source_candidate_index", r.get("source_candidate_index")),
                "group": str(r.get("fresh_confirmation_group") or st.get("fresh_confirmation_group") or "unknown"),
                "window": str(st.get("window") or ("early_mid" if si(r.get("branch_state_slot", st.get("branch_state_slot")), 0) == 0 else "mid_late")),
                "branch_state_slot": si(r.get("branch_state_slot", st.get("branch_state_slot")), 0),
                "branch_step": si(st.get("branch_step"), -1),
                "obs14": obs14(st),
                "prev_x": sf(prev.get("x")), "prev_y": sf(prev.get("y")), "prev_theta": sf(prev.get("theta")),
                "stage_a_trace_risk_score": sf(st.get("stage_a_trace_risk_score"), 0.0),
                "h10_physical": h10_phys, "h15_physical": h15_phys,
                "h10_decision_sum_s": h10_dec, "h15_decision_sum_s": h15_dec,
                "h10_solver_sum_s": h10_sol, "h15_solver_sum_s": h15_sol,
                "h10_safe_all": h10_safe, "h15_safe_all": h15_safe,
                "row_physical_tolerance": tol,
                "phys_delta_h10_minus_h15": h10_phys - h15_phys,
                "decision_gain_h10_vs_h15_s": h15_dec - h10_dec,
                "solver_gain_h10_vs_h15_s": h15_sol - h10_sol,
                "h10_beneficial_vs_h15": beneficial,
                "h10_catastrophic_vs_h15": catastrophic,
            })
    return rows, hashes


def feature(row: Mapping[str, Any], spec: str) -> List[float]:
    o = [sf(v) for v in (row.get("obs14") or [])[:14]]
    while len(o) < 14:
        o.append(0.0)
    pose = [sf(row.get("prev_x")) / 30.0, sf(row.get("prev_y")) / 30.0, sf(row.get("prev_theta")) / math.pi]
    step = max(0.0, sf(row.get("branch_step"))) / 150.0
    slot = sf(row.get("branch_state_slot"))
    if spec == "obs14_abs_l2":
        return norm_l2(o + [abs(v) for v in o])
    if spec == "obs14_pose_step_std":
        return o + pose + [step, slot]
    if spec == "obs14_abs_pose_step_std":
        return o + [abs(v) for v in o] + pose + [step, slot]
    if spec == "obs14_obstacle_pose_step_std":
        return o[5:14] + [abs(v) for v in o[5:14]] + pose + [step, slot]
    raise ContractError("unknown feature set " + spec)


def standardize(train_x: Sequence[Sequence[float]], xs: Sequence[Sequence[float]], do_std: bool) -> Tuple[List[List[float]], List[float], List[float]]:
    n = max((len(v) for v in train_x), default=0)
    if not do_std:
        return [[sf(v[i]) if i < len(v) else 0.0 for i in range(n)] for v in xs], [0.0] * n, [1.0] * n
    mat = [[sf(v[i]) if i < len(v) else 0.0 for i in range(n)] for v in train_x]
    means: List[float] = []
    stds: List[float] = []
    for j in range(n):
        col = [r[j] for r in mat]
        mu = sum(col) / len(col) if col else 0.0
        var = sum((z - mu) ** 2 for z in col) / max(1, len(col) - 1)
        sd = math.sqrt(var) if var > 1e-12 else 1.0
        means.append(mu); stds.append(sd)
    out = []
    for v in xs:
        vals = [sf(v[i]) if i < len(v) else 0.0 for i in range(n)]
        out.append([(vals[i] - means[i]) / stds[i] for i in range(n)])
    return out, means, stds


def solve(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
        div = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= div
        for r in range(n):
            if r == c:
                continue
            fac = M[r][c]
            if abs(fac) < 1e-18:
                continue
            for j in range(c, n + 1):
                M[r][j] -= fac * M[c][j]
    return [M[i][n] for i in range(n)]


def fit_ridge(X: Sequence[Sequence[float]], y: Sequence[float], weights: Sequence[float], alpha: float) -> List[float]:
    if not X:
        return [0.0]
    p = len(X[0]) + 1
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for xi, yi, wi in zip(X, y, weights):
        row = [1.0] + list(xi)
        w = max(1e-9, sf(wi, 1.0))
        for i in range(p):
            b[i] += w * row[i] * yi
            for j in range(p):
                A[i][j] += w * row[i] * row[j]
    for i in range(1, p):
        A[i][i] += alpha
    A[0][0] += 1e-9
    beta = solve(A, b)
    return beta if beta is not None else [sum(y) / len(y)] + [0.0] * (p - 1)


def pred(beta: Sequence[float], x: Sequence[float]) -> float:
    return sf(beta[0]) + sum(sf(beta[i + 1]) * sf(x[i]) for i in range(min(len(x), len(beta) - 1)))


def eval_choices(rows: Sequence[Mapping[str, Any]], choices: Mapping[str, int]) -> Dict[str, Any]:
    fixed_phys = sum(sf(r["h15_physical"]) for r in rows)
    fixed_dec = sum(sf(r["h15_decision_sum_s"]) for r in rows)
    fixed_sol = sum(sf(r["h15_solver_sum_s"]) for r in rows)
    tol_sum = sum(sf(r["row_physical_tolerance"]) for r in rows)
    pol_phys = pol_dec = pol_sol = 0.0
    counts = Counter(); confusion = Counter(); bad_rows = []; unsafe = []; chosen_deltas = []
    details = []
    for r in rows:
        h = int(choices.get(str(r["base_state_id"]), 15))
        counts[str(h)] += 1
        pos = bool(r["h10_beneficial_vs_h15"])
        if h == 10:
            pol_phys += sf(r["h10_physical"]); pol_dec += sf(r["h10_decision_sum_s"]); pol_sol += sf(r["h10_solver_sum_s"])
            chosen_deltas.append(sf(r["phys_delta_h10_minus_h15"]))
            if pos: confusion["TP"] += 1
            else: confusion["FP"] += 1
            if not bool(r["h10_safe_all"]):
                unsafe.append({"base_state_id": r["base_state_id"], "group": r["group"], "window": r["window"]})
            if bool(r["h10_catastrophic_vs_h15"]):
                bad_rows.append({"base_state_id": r["base_state_id"], "group": r["group"], "window": r["window"], "phys_delta": sf(r["phys_delta_h10_minus_h15"]), "decision_gain_s": sf(r["decision_gain_h10_vs_h15_s"]), "row_tolerance": sf(r["row_physical_tolerance"])})
        else:
            pol_phys += sf(r["h15_physical"]); pol_dec += sf(r["h15_decision_sum_s"]); pol_sol += sf(r["h15_solver_sum_s"])
            if pos: confusion["FN"] += 1
            else: confusion["TN"] += 1
        details.append({"base_state_id": r["base_state_id"], "selected_h": h, "label_h10_beneficial": pos, "phys_delta_h10_minus_h15": sf(r["phys_delta_h10_minus_h15"]), "decision_gain_h10_vs_h15_s": sf(r["decision_gain_h10_vs_h15_s"]), "group": r["group"], "window": r["window"]})
    dec_save = (fixed_dec - pol_dec) / fixed_dec if fixed_dec > 0 else 0.0
    sol_save = (fixed_sol - pol_sol) / fixed_sol if fixed_sol > 0 else 0.0
    phys_delta = pol_phys - fixed_phys
    return {
        "groups": len(rows), "chosen_counts": dict(counts), "confusion": dict(confusion),
        "fixed_H15": {"physical_sum": fixed_phys, "decision_sum_s": fixed_dec, "solver_sum_s": fixed_sol},
        "policy": {"physical_sum": pol_phys, "decision_sum_s": pol_dec, "solver_sum_s": pol_sol},
        "physical_delta_vs_fixed_H15": phys_delta, "physical_tolerance_sum": tol_sum,
        "decision_relative_saving_vs_fixed_H15": dec_save, "solver_relative_saving_vs_fixed_H15": sol_save,
        "physical_gate": phys_delta <= tol_sum, "no_catastrophic_fp": len(bad_rows) == 0, "unsafe_rows": unsafe,
        "catastrophic_false_positive_rows": bad_rows, "max_chosen_phys_delta": max(chosen_deltas) if chosen_deltas else None,
        "pass_10pct_no_cat_fp": bool((dec_save >= 0.10) and (phys_delta <= tol_sum) and not bad_rows and not unsafe),
        "pass_5pct_no_cat_fp": bool((dec_save >= 0.05) and (phys_delta <= tol_sum) and not bad_rows and not unsafe),
        "details": details,
    }


def stats(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {"n": len(xs), "min": xs[0], "median": xs[len(xs)//2] if len(xs)%2 else 0.5*(xs[len(xs)//2-1]+xs[len(xs)//2]), "mean": sum(xs)/len(xs), "max": xs[-1]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, hashes = load_rows()
    banks = sorted(BANK_DIRS)
    rows_by_bank = {b: [r for r in rows if r["bank_id"] == b] for b in banks}
    for b, rs in rows_by_bank.items():
        if len(rs) != 16:
            raise ContractError(f"expected 16 primary rows for {b}, got {len(rs)}")

    bank_opportunity = {}
    for b, rs in rows_by_bank.items():
        all_h10 = {str(r["base_state_id"]): 10 for r in rs}
        oracle = {str(r["base_state_id"]): (10 if r["h10_beneficial_vs_h15"] else 15) for r in rs}
        bank_opportunity[b] = {
            "h10_beneficial": int(sum(1 for r in rs if r["h10_beneficial_vs_h15"])),
            "h10_catastrophic": int(sum(1 for r in rs if r["h10_catastrophic_vs_h15"])),
            "all_h10": eval_choices(rs, all_h10),
            "row_oracle": eval_choices(rs, oracle),
        }

    # Fit risk/gain models for each held-out bank and threshold policies.
    fitted_cache: Dict[Tuple[str, str, float, float], Dict[str, Any]] = {}
    candidate_evals: List[Dict[str, Any]] = []
    closed_form_refits = 0
    for feat in FEATURE_SETS:
        for alpha in ALPHAS:
            for cat_w in CAT_WEIGHTS:
                fold_predictions: Dict[str, Dict[str, Dict[str, float]]] = {}
                for hold in banks:
                    train = [r for r in rows if r["bank_id"] != hold]
                    test = rows_by_bank[hold]
                    raw_train_x = [feature(r, feat) for r in train]
                    raw_test_x = [feature(r, feat) for r in test]
                    do_std = feat.endswith("_std")
                    train_x, means, stds = standardize(raw_train_x, raw_train_x, do_std)
                    test_x, _, _ = standardize(raw_train_x, raw_test_x, do_std)
                    y_delta = [max(-5.0, min(MAX_PRED_DELTA_TARGET, sf(r["phys_delta_h10_minus_h15"]))) for r in train]
                    w_delta = [(cat_w if r["h10_catastrophic_vs_h15"] else 1.0) for r in train]
                    y_cat = [1.0 if r["h10_catastrophic_vs_h15"] else 0.0 for r in train]
                    w_cat = [(cat_w if r["h10_catastrophic_vs_h15"] else 1.0) for r in train]
                    y_gain = [sf(r["decision_gain_h10_vs_h15_s"]) for r in train]
                    w_gain = [1.0 for _ in train]
                    beta_delta = fit_ridge(train_x, y_delta, w_delta, alpha)
                    beta_cat = fit_ridge(train_x, y_cat, w_cat, alpha)
                    beta_gain = fit_ridge(train_x, y_gain, w_gain, alpha)
                    closed_form_refits += 3
                    preds: Dict[str, Dict[str, float]] = {}
                    for r, x in zip(test, test_x):
                        preds[str(r["base_state_id"])] = {
                            "pred_delta": pred(beta_delta, x),
                            "pred_cat_score": max(0.0, min(1.0, pred(beta_cat, x))),
                            "pred_gain_s": pred(beta_gain, x),
                        }
                    fold_predictions[hold] = preds
                    fitted_cache[(feat, hold, alpha, cat_w)] = {"feature_set": feat, "alpha": alpha, "cat_weight": cat_w, "train_rows": len(train), "test_rows": len(test), "feature_dim": len(train_x[0]) if train_x else 0, "means_head": means[:5], "stds_head": stds[:5]}
                for risk_thr in RISK_THRESHOLDS:
                    for cat_thr in CAT_THRESHOLDS:
                        for gain_thr in GAIN_THRESHOLDS:
                            spec_id = f"v4_delta_value_{feat}_a{alpha:g}_cw{cat_w:g}_risk{risk_thr:g}_cat{cat_thr:g}_gain{gain_thr:g}"
                            holdout = {}
                            total_bad = 0; total_h10 = 0; min_save = 1e9; avg_save = 0.0; phys_fail = 0; weak_all = True; strong_all = True
                            for hold in banks:
                                preds = fold_predictions[hold]
                                choices = {}
                                for r in rows_by_bank[hold]:
                                    pr = preds[str(r["base_state_id"])]
                                    choose_h10 = (pr["pred_delta"] <= risk_thr) and (pr["pred_cat_score"] <= cat_thr) and (pr["pred_gain_s"] >= gain_thr)
                                    choices[str(r["base_state_id"])] = 10 if choose_h10 else 15
                                ev = eval_choices(rows_by_bank[hold], choices)
                                for d in ev["details"]:
                                    d.update(preds[str(d["base_state_id"])])
                                holdout[hold] = ev
                                total_bad += len(ev["catastrophic_false_positive_rows"])
                                total_h10 += int(ev["chosen_counts"].get("10", 0))
                                min_save = min(min_save, sf(ev["decision_relative_saving_vs_fixed_H15"]))
                                avg_save += sf(ev["decision_relative_saving_vs_fixed_H15"])
                                phys_fail += 0 if ev["physical_gate"] else 1
                                weak_all = weak_all and bool(ev["pass_5pct_no_cat_fp"])
                                strong_all = strong_all and bool(ev["pass_10pct_no_cat_fp"])
                            avg_save /= len(banks)
                            candidate_evals.append({
                                "variant_id": spec_id,
                                "spec": {"feature_set": feat, "alpha": alpha, "cat_weight": cat_w, "risk_threshold": risk_thr, "cat_threshold": cat_thr, "gain_threshold_s": gain_thr},
                                "holdout_evaluations": holdout,
                                "all_holdouts_strong": strong_all,
                                "all_holdouts_weak": weak_all,
                                "total_catastrophic_fp": total_bad,
                                "physical_gate_fail_count": phys_fail,
                                "min_decision_saving": min_save,
                                "avg_decision_saving": avg_save,
                                "total_h10": total_h10,
                            })

    def rank_key(c: Mapping[str, Any]) -> Tuple[Any, ...]:
        return (
            not bool(c["all_holdouts_strong"]),
            not bool(c["all_holdouts_weak"]),
            int(c["total_catastrophic_fp"]),
            int(c["physical_gate_fail_count"]),
            -sf(c["min_decision_saving"]),
            -sf(c["avg_decision_saving"]),
            -int(c["total_h10"]),
        )
    ranked = sorted(candidate_evals, key=rank_key)
    top = ranked[:20]
    strong = [c for c in candidate_evals if c["all_holdouts_strong"]]
    weak = [c for c in candidate_evals if c["all_holdouts_weak"]]
    no_bad = [c for c in candidate_evals if c["total_catastrophic_fp"] == 0 and c["physical_gate_fail_count"] == 0]
    best = top[0]

    # Diagnostic error analysis for the best variant.
    best_bad_preds = []
    best_missed_positive_preds = []
    for hold, ev in best["holdout_evaluations"].items():
        for d in ev["details"]:
            if d["selected_h"] == 10 and not d["label_h10_beneficial"]:
                best_bad_preds.append(dict(d, bank_id=hold))
            if d["selected_h"] == 15 and d["label_h10_beneficial"]:
                best_missed_positive_preds.append(dict(d, bank_id=hold))

    raw_out = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": "development_IMPROVED_opened_bank_delta_value_model_lobo_no_simulation_no_validation_no_test",
        "hypothesis": "A cost-sensitive closed-form value/risk model over deployable pre-decision features may separate catastrophic H10 physical-risk modes from H10 compute-beneficial states better than nearest-neighbour selector-only gates.",
        "budgets": {"new_simulations": 0, "new_training_episodes": 0, "gradient_steps": 0, "closed_form_refits": closed_form_refits, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "bank_opportunity": bank_opportunity,
        "candidate_count": len(candidate_evals),
        "closed_form_refits": closed_form_refits,
        "headline": {
            "strong_count": len(strong), "weak_count": len(weak), "zero_bad_physical_gate_count": len(no_bad),
            "best_variant": best["variant_id"], "best_strong": bool(best["all_holdouts_strong"]), "best_weak": bool(best["all_holdouts_weak"]),
            "best_total_catastrophic_fp": int(best["total_catastrophic_fp"]), "best_min_decision_saving": sf(best["min_decision_saving"]),
            "best_avg_decision_saving": sf(best["avg_decision_saving"]), "best_total_h10": int(best["total_h10"]),
        },
        "ranked_candidates": top,
        "best_error_analysis": {"bad_predictions": best_bad_preds, "missed_positive_count": len(best_missed_positive_preds), "missed_positive_examples_head": best_missed_positive_preds[:20]},
        "hashes": hashes,
        "access_flags": {"sealed_test_accessed": False, "validation64_bank_opened": False, "mobile_robot_mppi_resumed": False},
        "platform": {"python": sys.version, "platform": platform.platform()},
        "next_decision": "If no strong all-bank pass, selector-only/value-linear refit remains insufficient; next intervention should add representation/terminal-value information or a bounded value-function refit/training smoke, not validation64.",
    }
    raw_path = OUT / "raw.json"; summary_path = OUT / "summary.md"; done_path = OUT / "completed.json"
    write_json(raw_path, raw_out)

    def fmt_pct(x: Any) -> str:
        return "None" if x is None else f"{100.0 * sf(x):.2f}%"
    lines = []
    lines.append("# Vehicle true-variable-H delta-value model v4 LOBO")
    lines.append("")
    lines.append(f"UTC: `{raw_out['created_utc']}`. Development-only opened-bank closed-form risk/value diagnostic; no simulations, no validation64, no sealed test, no gradient training.")
    lines.append("")
    lines.append("## Why this was run")
    lines.append("")
    lines.append("Fresh v2 confirmed real measured H10 timing savings but failed physical gates through catastrophic H10 false positives. v3b showed deployable nearest-neighbour/veto selectors could not achieve a strong all-bank tradeoff. This diagnostic tests a different, bounded intervention: cost-sensitive closed-form prediction of H10-H15 physical delta, catastrophic risk, and measured decision-time gain from online/pre-decision features.")
    lines.append("")
    lines.append("## Headline")
    h = raw_out["headline"]
    lines.append(f"- Candidate threshold policies evaluated after cached fold fits: `{len(candidate_evals)}`.")
    lines.append(f"- Closed-form model fits/refits (not gradient training/checkpoints): `{closed_form_refits}`.")
    lines.append(f"- Strong all-held-bank candidates: `{len(strong)}`; weak all-held-bank candidates: `{len(weak)}`; zero-bad/physical-gate candidates: `{len(no_bad)}`.")
    lines.append(f"- Best variant: `{h['best_variant']}`.")
    lines.append(f"- Best strong/weak: `{h['best_strong']}` / `{h['best_weak']}`; bad FP total `{h['best_total_catastrophic_fp']}`; min/avg decision saving `{fmt_pct(h['best_min_decision_saving'])}` / `{fmt_pct(h['best_avg_decision_saving'])}`; H10 total `{h['best_total_h10']}/48`.")
    lines.append("")
    lines.append("## Opened-bank opportunity reference")
    lines.append("")
    lines.append("| bank | H10 beneficial | H10 catastrophic | oracle dec save | oracle phys Δ | all-H10 dec save | all-H10 phys Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for b in banks:
        bo = bank_opportunity[b]
        lines.append(f"| `{b}` | {bo['h10_beneficial']} | {bo['h10_catastrophic']} | {fmt_pct(bo['row_oracle']['decision_relative_saving_vs_fixed_H15'])} | {bo['row_oracle']['physical_delta_vs_fixed_H15']:.4g} | {fmt_pct(bo['all_h10']['decision_relative_saving_vs_fixed_H15'])} | {bo['all_h10']['physical_delta_vs_fixed_H15']:.4g} |")
    lines.append("")
    lines.append("## Top candidate summary")
    lines.append("")
    lines.append("| rank | strong | weak | bad | phys gate fails | min save | avg save | H10 total | variant |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, c in enumerate(top[:12], 1):
        lines.append(f"| {i} | `{c['all_holdouts_strong']}` | `{c['all_holdouts_weak']}` | {c['total_catastrophic_fp']} | {c['physical_gate_fail_count']} | {fmt_pct(c['min_decision_saving'])} | {fmt_pct(c['avg_decision_saving'])} | {c['total_h10']} | `{c['variant_id']}` |")
    lines.append("")
    lines.append("## Best candidate by held-out bank")
    lines.append("")
    lines.append("| held-out bank | H counts | confusion | bad rows | phys Δ/tol | decision save | solver save | pass10 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for b, ev in best["holdout_evaluations"].items():
        lines.append(f"| `{b}` | `{ev['chosen_counts']}` | `{ev['confusion']}` | {len(ev['catastrophic_false_positive_rows'])} | {ev['physical_delta_vs_fixed_H15']:.4g}/{ev['physical_tolerance_sum']:.4g} | {fmt_pct(ev['decision_relative_saving_vs_fixed_H15'])} | {fmt_pct(ev['solver_relative_saving_vs_fixed_H15'])} | `{ev['pass_10pct_no_cat_fp']}` |")
    lines.append("")
    if best_bad_preds:
        lines.append("## Best-candidate bad predictions")
        lines.append("")
        lines.append("| bank | state | group/window | phys Δ | decision gain s | pred Δ | pred cat | pred gain |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for d in best_bad_preds[:20]:
            lines.append(f"| `{d['bank_id']}` | `{d['base_state_id']}` | `{d['group']}::{d['window']}` | {d['phys_delta_h10_minus_h15']:.4g} | {d['decision_gain_h10_vs_h15_s']:.4g} | {d.get('pred_delta', 0):.4g} | {d.get('pred_cat_score', 0):.4g} | {d.get('pred_gain_s', 0):.4g} |")
        lines.append("")
    lines.append("## Decision")
    lines.append("")
    if strong:
        lines.append("At least one closed-form deployable-feature delta/risk model passed opened-bank LOBO strong gates. Because all banks are development-opened, this is not validation; freeze an unused fresh-source confirmation before any validation64 action.")
    elif weak:
        lines.append("Only weak all-bank candidates were found. This may be useful as a conservative development controller but does not meet the strong measured decision-time gate; do not advance to validation64.")
    else:
        lines.append("No closed-form deployable-feature delta/risk model met all-bank weak/strong gates. Together with v3b, this argues that selector-only and simple value-linear refits are insufficient; next bounded intervention should add more informative representation/terminal-value information or a value-function refit/training smoke rather than another unchanged label-density sweep.")
    lines.append("")
    lines.append("Raw: `" + rel(raw_path) + "`; completed: `" + rel(done_path) + "`.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Durable state and audit appends.
    backup_req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_DELTA_VALUE_MODEL_V4_LOBO_{STAMP}.json"
    req_obj = {"created_utc": raw_out["created_utc"], "reason": "backup after delta-value model v4 LOBO diagnostic", "paths": [rel(raw_path), rel(summary_path), rel(done_path)], "sealed_test_accessed": False, "validation64_bank_opened": False}
    write_json(backup_req, req_obj)
    completed = {"passed": True, "status": "complete", "marker": MARKER, "summary": rel(summary_path), "raw": rel(raw_path), "backup_request": rel(backup_req), "sealed_test_accessed": False, "validation64_bank_opened": False, "new_simulations": 0, "gradient_steps": 0, "closed_form_refits": closed_form_refits}
    write_json(done_path, completed)
    # update hashes now that files exist
    raw_out["hashes"].update({rel(raw_path): sha256(raw_path), rel(summary_path): sha256(summary_path), rel(done_path): sha256(done_path)})
    write_json(raw_path, raw_out)
    summary_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")

    state_text = f"""# Continue state after {NAME}

UTC: {raw_out['created_utc']}

Result: strong_count={len(strong)}, weak_count={len(weak)}, best={h['best_variant']}, best_min_decision_saving={h['best_min_decision_saving']:.6g}, best_bad={h['best_total_catastrophic_fp']}.

Access: validation64 closed, sealed test closed, no simulations, no gradient training. Closed-form diagnostic refits: {closed_form_refits}.

Next: {'freeze unused fresh-source confirmation for the passing delta-value model before validation64' if strong else 'do not advance to validation64; prioritize representation/terminal-value refit or bounded value/training smoke, because simple deployable-feature delta/value refit did not meet strong gates'}.

Artifacts: {rel(summary_path)}, {rel(raw_path)}, {rel(done_path)}.
"""
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(state_text, encoding="utf-8")
    audit_line = f"\n- {raw_out['created_utc']} `{NAME}`: closed-form delta/risk value LOBO over opened fresh_v0/v1/v2; strong={len(strong)}, weak={len(weak)}, best_min_save={h['best_min_decision_saving']:.4g}, bad={h['best_total_catastrophic_fp']}; no validation64/test/simulation; artifacts `{rel(summary_path)}`.\n"
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        p = ROOT / doc
        with p.open("a", encoding="utf-8") as f:
            f.write(audit_line)
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8") as f:
        f.write(f"{STAMP},{NAME},development_delta_value_model_lobo,no_simulation_no_rng,opened_fresh_v0_v1_v2_no_validation_no_test,0,0,0,{closed_form_refits},0,False,{rel(done_path)},{MARKER}\n")
    print(f"SUMMARY {rel(summary_path)}")
    print(json.dumps(raw_out["headline"], sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        err = {"passed": False, "status": "failed", "error": type(exc).__name__, "message": str(exc), "sealed_test_accessed": False, "validation64_bank_opened": False, "marker": MARKER}
        write_json(OUT / "completed.json", err)
        raise
