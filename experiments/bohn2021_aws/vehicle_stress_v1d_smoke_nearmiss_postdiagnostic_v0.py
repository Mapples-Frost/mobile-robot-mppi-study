#!/usr/bin/env python3
"""No-simulation postdiagnostic for vehicle stress-v1d smoke.

Purpose: after the v1d trace-selected terminal-stable opportunity smoke returned
0/12 robust-positive states, quantify whether the failure is due to (a) no
safe physical gains at all, (b) gains present but below the predeclared material
threshold, (c) terminal-mode disagreement, or (d) safety/solver regressions.

This script reads only already-produced development artifacts.  It runs no
rollouts, opens no validation64/sealed-test bank, and performs no training or
refit.  It is intended as the next bounded-cycle concrete diagnostic because
this cycle already used its one run_experiment slot for the v1d smoke.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260929T0315Z"
SMOKE_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z"
SMOKE_RAW = SMOKE_DIR / "raw.json"
SMOKE_DONE = SMOKE_DIR / "completed.json"
OUT = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1d_smoke_nearmiss_postdiagnostic_v0_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1d_smoke_nearmiss_postdiagnostic_v0_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1D_SMOKE_NEARMISS_POSTDIAGNOSTIC_V0_{STAMP}.json"
TERMINAL_MODES = ["zero_terminal", "h15_common_terminal"]
BRANCH_HORIZONS = [10, 15, 25, 30, 35, 45, 50]
THRESHOLDS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0]
MATERIAL_THRESHOLD = 3.0


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


def summary(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    def pct(q: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = (len(xs) - 1) * q / 100.0
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        return xs[lo] if lo == hi else xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"count": len(xs), "min": xs[0], "median": pct(50), "mean": sum(xs) / len(xs), "p95": pct(95), "max": xs[-1], "sum": sum(xs)}


def verify_inputs() -> Dict[str, Any]:
    if not SMOKE_DONE.exists() or not SMOKE_RAW.exists():
        raise ContractError("v1d smoke raw/completed artifacts are missing")
    done = read_json(SMOKE_DONE)
    if done.get("passed") is not True and done.get("hard_pass") is not True:
        raise ContractError("v1d smoke did not pass artifact completion")
    for key in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if done.get(key) is not False:
            raise ContractError(f"unexpected access flag in completed marker: {key}={done.get(key)}")
    if int(done.get("episodes", -1)) != 188 or int(done.get("control_steps", -1)) != 14913:
        raise ContractError("v1d smoke budget metadata changed; inspect before postdiagnostic")
    raw = read_json(SMOKE_RAW)
    for key in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if raw.get(key) is not False:
            raise ContractError(f"unexpected access flag in raw: {key}={raw.get(key)}")
    analysis = raw.get("analysis") or {}
    if int(analysis.get("state_count", -1)) != 12:
        raise ContractError("unexpected v1d state_count")
    return {"done": done, "raw": raw}


def comp_safety_ok(comp: Mapping[str, Any]) -> bool:
    return bool(comp.get("branch_reached")) and bool(comp.get("physical_prefix_matches_H15")) and finite(comp.get("state_distance_vs_H15"), 1e9) <= 1e-5 and bool(comp.get("no_success_constraint_solver_regression_vs_H15"))


def analyze_state(row: Mapping[str, Any]) -> Dict[str, Any]:
    by_mode: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    per_mode: Dict[str, Any] = {}
    single_mode_material_horizons: Dict[str, List[int]] = {}
    all_nonref_gains: List[float] = []
    all_nonref_total_gains: List[float] = []
    safety_regressions: List[Dict[str, Any]] = []
    best_cost_nonref_modes: List[Dict[str, Any]] = []
    for mode in TERMINAL_MODES:
        comps = list((((row.get("mode_results") or {}).get(mode) or {}).get("comparisons") or []))
        by_h = {int(c.get("horizon")): c for c in comps}
        by_mode[mode] = by_h
        ref = by_h.get(15)
        if ref is None:
            per_mode[mode] = {"missing_reference": True}
            continue
        nonref = [c for h, c in by_h.items() if h != 15]
        for c in nonref:
            all_nonref_gains.append(finite(c.get("gain_vs_H15_physical")))
            all_nonref_total_gains.append(finite(c.get("gain_vs_H15_total")))
            if not comp_safety_ok(c):
                safety_regressions.append({
                    "mode": mode,
                    "horizon": int(c.get("horizon")),
                    "gain_physical": finite(c.get("gain_vs_H15_physical")),
                    "success": bool(c.get("success")),
                    "constraint": bool(c.get("constraint")),
                    "solver_failure_steps": int(c.get("solver_failure_steps", 0)),
                    "initial_failed_steps": int(c.get("initial_failed_steps", 0)),
                    "final_failed_steps": int(c.get("final_failed_steps", 0)),
                    "prefix_match": bool(c.get("physical_prefix_matches_H15")),
                    "state_distance": finite(c.get("state_distance_vs_H15"), 1e9),
                })
        best = min(comps, key=lambda c: (finite(c.get("continuation_physical"), 1e99), finite(c.get("decision_sum_s"), 1e99), int(c.get("horizon", 999)))) if comps else None
        if best is not None and int(best.get("horizon")) != 15:
            best_cost_nonref_modes.append({
                "mode": mode,
                "horizon": int(best.get("horizon")),
                "best_continuation_physical": finite(best.get("continuation_physical")),
                "gain_vs_H15_physical": finite(best.get("gain_vs_H15_physical")),
                "safe": comp_safety_ok(best),
            })
        material_by_threshold = {}
        for thr in THRESHOLDS:
            hs = sorted(int(c.get("horizon")) for c in nonref if comp_safety_ok(c) and finite(c.get("gain_vs_H15_physical")) >= thr)
            material_by_threshold[str(thr)] = hs
        single_mode_material_horizons[mode] = material_by_threshold[str(MATERIAL_THRESHOLD)]
        per_mode[mode] = {
            "reference_physical": finite(ref.get("continuation_physical")),
            "best_physical_any_h": None if best is None else {"horizon": int(best.get("horizon")), "continuation_physical": finite(best.get("continuation_physical")), "gain_vs_H15_physical": finite(best.get("gain_vs_H15_physical")), "safe": comp_safety_ok(best)},
            "max_safe_gain_nonH15": None if not nonref else max((finite(c.get("gain_vs_H15_physical")) for c in nonref if comp_safety_ok(c)), default=None),
            "max_any_gain_nonH15": None if not nonref else max(finite(c.get("gain_vs_H15_physical")) for c in nonref),
            "material_by_threshold": material_by_threshold,
        }
    robust_by_threshold: Dict[str, List[int]] = {}
    robust_min_gain_by_h: Dict[str, float] = {}
    for thr in THRESHOLDS:
        hs: List[int] = []
        for h in BRANCH_HORIZONS:
            if h == 15:
                continue
            comps = [by_mode.get(mode, {}).get(h) for mode in TERMINAL_MODES]
            if any(c is None for c in comps):
                continue
            gains = [finite(c.get("gain_vs_H15_physical")) for c in comps if c is not None]
            if all(comp_safety_ok(c) for c in comps if c is not None) and min(gains) >= thr:
                hs.append(h)
                robust_min_gain_by_h[str(h)] = min(gains)
        robust_by_threshold[str(thr)] = sorted(hs)
    max_robust_min_gain = None
    max_robust_h = None
    for h in BRANCH_HORIZONS:
        if h == 15:
            continue
        comps = [by_mode.get(mode, {}).get(h) for mode in TERMINAL_MODES]
        if any(c is None for c in comps) or not all(comp_safety_ok(c) for c in comps if c is not None):
            continue
        gains = [finite(c.get("gain_vs_H15_physical")) for c in comps if c is not None]
        min_gain = min(gains)
        if max_robust_min_gain is None or min_gain > max_robust_min_gain:
            max_robust_min_gain = min_gain
            max_robust_h = h
    terminal_disagreement_horizons: List[int] = []
    for h in BRANCH_HORIZONS:
        if h == 15:
            continue
        mode_hits = []
        for mode in TERMINAL_MODES:
            c = by_mode.get(mode, {}).get(h)
            mode_hits.append(bool(c is not None and comp_safety_ok(c) and finite(c.get("gain_vs_H15_physical")) >= MATERIAL_THRESHOLD))
        if any(mode_hits) and not all(mode_hits):
            terminal_disagreement_horizons.append(h)
    return {
        "state_id": row.get("state_id"),
        "case": int(row.get("case", -1)),
        "selection_group": row.get("selection_group"),
        "target_role": row.get("target_role"),
        "branch_step": int(row.get("branch_step", -1)),
        "predeclared_label": row.get("label"),
        "predeclared_robust_positive": bool(row.get("robust_positive_state")),
        "predeclared_robust_horizons": row.get("robust_positive_horizons") or [],
        "per_mode": per_mode,
        "robust_by_threshold": robust_by_threshold,
        "max_robust_min_gain_nonH15": max_robust_min_gain,
        "max_robust_min_gain_horizon": max_robust_h,
        "terminal_disagreement_horizons_at_material_threshold": terminal_disagreement_horizons,
        "single_mode_material_horizons": single_mode_material_horizons,
        "best_cost_nonH15_modes": best_cost_nonref_modes,
        "safety_regressions": safety_regressions,
        "nonref_physical_gain_summary": summary(all_nonref_gains),
        "nonref_total_gain_summary": summary(all_nonref_total_gains),
    }


def run() -> int:
    inputs = verify_inputs()
    raw = inputs["raw"]
    state_rows = raw["analysis"]["state_rows"]
    per_state = [analyze_state(r) for r in state_rows]
    threshold_counts = {}
    threshold_cases = {}
    for thr in THRESHOLDS:
        key = str(thr)
        rows = [r for r in per_state if r["robust_by_threshold"].get(key)]
        threshold_counts[key] = len(rows)
        threshold_cases[key] = sorted(set(int(r["case"]) for r in rows))
    single_mode_material_states = [r for r in per_state if any(r["single_mode_material_horizons"].get(m) for m in TERMINAL_MODES)]
    best_nonh_states = [r for r in per_state if r["best_cost_nonH15_modes"]]
    safety_regression_count = sum(len(r["safety_regressions"]) for r in per_state)
    max_robust_candidates = [r for r in per_state if r["max_robust_min_gain_nonH15"] is not None]
    top_near = sorted(max_robust_candidates, key=lambda r: (finite(r["max_robust_min_gain_nonH15"], -1e99), -int(r["case"])), reverse=True)[:12]
    max_robust_min_gain_overall = None if not max_robust_candidates else max(finite(r["max_robust_min_gain_nonH15"], -1e99) for r in max_robust_candidates)
    if threshold_counts.get("0.0", 0) == 0:
        interpretation = "Under safety/common-prefix requirements, no non-H15 horizon weakly improves physical continuation in both terminal modes for any trace-selected state; this supports sparse/absent robust adaptive opportunity in the current source-supported vehicle scenario family rather than an immediate selector-training bottleneck."
    elif threshold_counts.get("3.0", 0) == 0:
        interpretation = "Some non-H15 horizons are robustly non-worse at relaxed thresholds, but all material physical gains are below the predeclared 3.0 threshold; next work should inspect objective scale/terminal modeling before selector training."
    elif single_mode_material_states:
        interpretation = "Material gains occur in only one terminal mode for some states; terminal-source mismatch remains the leading bottleneck."
    else:
        interpretation = "Unexpected threshold pattern; inspect per-state rows before choosing training or scenario redesign."
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "created_utc": created,
        "method": "vehicle_stress_v1d_smoke_nearmiss_postdiagnostic_v0_no_simulation",
        "classification": "development_no_simulation_postdiagnostic_no_validation64_no_sealed_test",
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
        "inputs": {"smoke_raw": rel(SMOKE_RAW), "smoke_raw_sha256": sha256(SMOKE_RAW), "smoke_completed": rel(SMOKE_DONE), "smoke_completed_sha256": sha256(SMOKE_DONE)},
        "threshold_counts": threshold_counts,
        "threshold_cases": threshold_cases,
        "predeclared_robust_positive_state_count": int(raw["analysis"].get("robust_positive_state_count", -1)),
        "predeclared_smoke_gate": bool(raw["analysis"].get("smoke_pass_to_next_label_or_refit_design")),
        "single_mode_material_state_count": len(single_mode_material_states),
        "best_cost_nonH15_state_count": len(best_nonh_states),
        "safety_regression_count": safety_regression_count,
        "max_robust_min_gain_overall": max_robust_min_gain_overall,
        "top_near_misses": top_near,
        "per_state": per_state,
        "interpretation": interpretation,
        "decision": {
            "train_or_refit_now": False,
            "next_action_if_confirmed": "freeze terminal/modeling or scenario-opportunity amendment rather than selector/refit; do not rerun unchanged v1d smoke",
            "reason": "v1d smoke failed the stable-label gate; this postdiagnostic only localizes the failure mode without new simulations",
        },
    }
    write_json(OUT / "raw.json", result)
    lines = [
        "# Vehicle stress-v1d smoke near-miss postdiagnostic v0",
        "",
        f"UTC: `{created}`. No simulations, no training/refit, no validation64/sealed-test access.",
        "",
        "## Headline",
        "",
        f"- Predeclared v1d robust-positive states: `{result['predeclared_robust_positive_state_count']}`; smoke gate: `{result['predeclared_smoke_gate']}`.",
        f"- Robust state counts by relaxed physical-gain threshold: `{threshold_counts}`.",
        f"- Cases by threshold: `{threshold_cases}`.",
        f"- Single-terminal material states at threshold 3.0: `{len(single_mode_material_states)}`.",
        f"- States whose best physical cost was non-H15 in at least one terminal mode: `{len(best_nonh_states)}`.",
        f"- Safety/solver regression comparisons: `{safety_regression_count}`.",
        f"- Max robust min-gain over all states/horizons: `{max_robust_min_gain_overall}`.",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Top near-miss states",
        "",
        "| state | case | group | branch | best robust H | max min-gain | robust thresholds | single-mode material H |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    for r in top_near:
        lines.append("| `%s` | %d | `%s` | %d | `%s` | %.6g | `%s` | `%s` |" % (r["state_id"], int(r["case"]), r.get("selection_group"), int(r["branch_step"]), str(r.get("max_robust_min_gain_horizon")), finite(r.get("max_robust_min_gain_nonH15"), float("nan")), r.get("robust_by_threshold"), r.get("single_mode_material_horizons")))
    lines += [
        "",
        "## Decision",
        "",
        "No selector/refit/training is justified by this diagnostic alone. If the relaxed-threshold counts are also zero or near-zero, prioritize terminal/modeling or scenario-opportunity redesign diagnostics over another unchanged label-density rollout.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# v1d near-miss postdiagnostic state\n\nUTC: {created}. No-simulation postdiagnostic completed. threshold_counts={threshold_counts}; single_mode_material_states={len(single_mode_material_states)}; best_cost_nonH15_states={len(best_nonh_states)}; max_robust_min_gain={max_robust_min_gain_overall}. Next action: preserve and back up, then choose terminal/modeling or scenario-opportunity amendment; train_or_refit_now=false.\n", encoding="utf-8")
    write_json(BACKUP_REQ, {
        "requested_utc": created,
        "reason": "backup v1d no-simulation near-miss postdiagnostic before further diagnostics/simulation/training",
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
    write_json(OUT / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
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
            "threshold_counts": threshold_counts,
            "single_mode_material_state_count": len(single_mode_material_states),
            "best_cost_nonH15_state_count": len(best_nonh_states),
            "max_robust_min_gain_overall": max_robust_min_gain_overall,
            "train_or_refit_now": False,
            "next_action": result["decision"]["next_action_if_confirmed"],
        },
        "backup_request": rel(BACKUP_REQ),
        "hashes": {rel(p): sha256(p) for p in sorted([OUT / "raw.json", OUT / "summary.md", STATE, BACKUP_REQ, Path(__file__).resolve(), SMOKE_DONE, SMOKE_RAW]) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "threshold_counts": threshold_counts,
        "single_mode_material_state_count": len(single_mode_material_states),
        "best_cost_nonH15_state_count": len(best_nonh_states),
        "max_robust_min_gain_overall": max_robust_min_gain_overall,
        "train_or_refit_now": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(BACKUP_REQ),
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
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "sealed_test_accessed": False, "validation64_bank_opened": False}, sort_keys=True))
        return 0
    if OUT.exists() and any(p.name != "run.lock" for p in OUT.iterdir()):
        raise ContractError(f"partial output exists; inspect first: {rel(OUT)}")
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
