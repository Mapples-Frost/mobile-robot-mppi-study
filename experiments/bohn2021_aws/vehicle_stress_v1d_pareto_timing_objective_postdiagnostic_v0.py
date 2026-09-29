#!/usr/bin/env python3
"""No-simulation Pareto/timing objective postdiagnostic for vehicle stress-v1d.

The v1d terminal-stable label-density smoke found 0/12 material physical-gain
states.  That is insufficient for a supervised "improve physical cost over H15"
selector, but it does not by itself answer the adaptive-MPC tradeoff question:
a non-H15 horizon could be physically non-worse while reducing *measured* compute
or avoiding synthetic horizon-penalty artifacts.  This diagnostic therefore
revisits the learning target using existing v1d branch-continuation artifacts
only.

It opens no validation64/sealed-test bank, runs no rollout, and performs no
training/refit.  It separates physical cost, synthetic total cost, measured
whole-decision time, solver/safety regressions and constant-H dominance.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260929T0325Z"
SMOKE_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z"
SMOKE_RAW = SMOKE_DIR / "raw.json"
SMOKE_DONE = SMOKE_DIR / "completed.json"
NEARMISS_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_smoke_nearmiss_postdiagnostic_v0_20260929T0315Z/completed.json"
OUT = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1d_pareto_timing_objective_postdiagnostic_v0_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1d_pareto_timing_objective_postdiagnostic_v0_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1D_PARETO_TIMING_OBJECTIVE_POSTDIAGNOSTIC_V0_{STAMP}.json"
TERMINAL_MODES = ["zero_terminal", "h15_common_terminal"]
HORIZONS = [10, 15, 25, 30, 35, 45, 50]
EPSILONS = [0.0, 0.01, 0.1, 0.5]
STRICT_EPS = 0.01
STRICT_ABS_TIME_GAIN_S = 0.05
STRICT_REL_TIME_GAIN = 0.02
RELAXED_EPS = 0.1
RELAXED_ABS_TIME_GAIN_S = 0.02
RELAXED_REL_TIME_GAIN = 0.0


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


def maybe_float(x: Any) -> Optional[float]:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in row and row.get(k) is not None:
            return row.get(k)
    return None


def stat(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    def pct(q: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = (len(xs) - 1) * q / 100.0
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        return xs[lo] if lo == hi else xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"count": len(xs), "min": xs[0], "median": pct(50), "mean": sum(xs) / len(xs), "p95": pct(95), "max": xs[-1], "sum": sum(xs)}


def verify_inputs() -> Dict[str, Any]:
    for p in (SMOKE_RAW, SMOKE_DONE, NEARMISS_DONE):
        if not p.exists():
            raise ContractError(f"required input missing: {rel(p)}")
    done = read_json(SMOKE_DONE)
    if done.get("passed") is not True and done.get("hard_pass") is not True:
        raise ContractError("v1d smoke completed marker is not passing")
    near = read_json(NEARMISS_DONE)
    if near.get("passed") is not True and near.get("hard_pass") is not True:
        raise ContractError("near-miss postdiagnostic completed marker is not passing")
    for obj_name, obj in (("smoke_completed", done), ("nearmiss_completed", near)):
        for flag in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
            if obj.get(flag) is not False:
                raise ContractError(f"{obj_name} has invalid access flag {flag}={obj.get(flag)}")
    raw = read_json(SMOKE_RAW)
    for flag in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if raw.get(flag) is not False:
            raise ContractError(f"raw has invalid access flag {flag}={raw.get(flag)}")
    analysis = raw.get("analysis") or {}
    if int(analysis.get("state_count", -1)) != 12:
        raise ContractError("unexpected state_count in v1d smoke raw")
    return {"raw": raw, "done": done, "near": near}


def comp_safety_ok(c: Mapping[str, Any]) -> bool:
    return (
        bool(c.get("branch_reached"))
        and bool(c.get("physical_prefix_matches_H15"))
        and finite(c.get("state_distance_vs_H15"), 1e9) <= 1e-5
        and bool(c.get("no_success_constraint_solver_regression_vs_H15"))
    )


def comp_steps(c: Mapping[str, Any]) -> Optional[int]:
    val = first_present(c, ["continuation_steps", "control_steps", "steps", "episode_steps", "n_steps"])
    if val is None:
        return None
    try:
        return int(val)
    except Exception:
        return None


def comp_decision_sum(c: Mapping[str, Any]) -> Optional[float]:
    val = first_present(c, ["decision_sum_s", "whole_decision_sum_s", "decision_total_s", "elapsed_decision_sum_s"])
    return maybe_float(val)


def comp_solver_sum(c: Mapping[str, Any]) -> Optional[float]:
    val = first_present(c, ["solver_sum_s", "solver_total_s", "solve_sum_s", "mpc_solver_sum_s"])
    return maybe_float(val)


def analyze_state(row: Mapping[str, Any]) -> Dict[str, Any]:
    by_mode: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    refs: Dict[str, Mapping[str, Any]] = {}
    for mode in TERMINAL_MODES:
        comps = list((((row.get("mode_results") or {}).get(mode) or {}).get("comparisons") or []))
        by_h = {int(c.get("horizon")): c for c in comps if c.get("horizon") is not None}
        by_mode[mode] = by_h
        if 15 in by_h:
            refs[mode] = by_h[15]
    if set(refs) != set(TERMINAL_MODES):
        raise ContractError(f"state {row.get('state_id')} lacks H15 reference in both terminal modes")

    horizon_rows: List[Dict[str, Any]] = []
    best_by_eps: Dict[str, Optional[Dict[str, Any]]] = {str(e): None for e in EPSILONS}
    for h in HORIZONS:
        if h == 15:
            continue
        comps = [by_mode[m].get(h) for m in TERMINAL_MODES]
        if any(c is None for c in comps):
            continue
        mode_metrics: Dict[str, Any] = {}
        safe_all = True
        phys_gains: List[float] = []
        total_gains: List[float] = []
        time_gains: List[float] = []
        rel_time_gains: List[float] = []
        solver_time_gains: List[float] = []
        step_deltas: List[int] = []
        for mode, c in zip(TERMINAL_MODES, comps):
            ref = refs[mode]
            safe = comp_safety_ok(c)
            safe_all = safe_all and safe
            pg = finite(c.get("gain_vs_H15_physical"))
            tg = finite(c.get("gain_vs_H15_total"))
            phys_gains.append(pg)
            total_gains.append(tg)
            ref_dec = comp_decision_sum(ref)
            dec = comp_decision_sum(c)
            time_gain = None
            rel_time_gain = None
            if ref_dec is not None and dec is not None:
                time_gain = ref_dec - dec
                rel_time_gain = time_gain / max(ref_dec, 1e-9)
                time_gains.append(time_gain)
                rel_time_gains.append(rel_time_gain)
            ref_sol = comp_solver_sum(ref)
            sol = comp_solver_sum(c)
            solver_gain = None
            if ref_sol is not None and sol is not None:
                solver_gain = ref_sol - sol
                solver_time_gains.append(solver_gain)
            ref_steps = comp_steps(ref)
            steps = comp_steps(c)
            step_delta = None
            if ref_steps is not None and steps is not None:
                step_delta = ref_steps - steps
                step_deltas.append(step_delta)
            mode_metrics[mode] = {
                "safe": safe,
                "gain_physical": pg,
                "loss_physical": max(0.0, -pg),
                "gain_total_synthetic": tg,
                "decision_sum_s": dec,
                "ref_decision_sum_s": ref_dec,
                "decision_time_gain_s": time_gain,
                "decision_rel_time_gain": rel_time_gain,
                "solver_sum_s": sol,
                "ref_solver_sum_s": ref_sol,
                "solver_time_gain_s": solver_gain,
                "steps": steps,
                "ref_steps": ref_steps,
                "step_delta_vs_H15": step_delta,
                "success": bool(c.get("success")),
                "constraint": bool(c.get("constraint")),
                "solver_failure_steps": int(c.get("solver_failure_steps", 0) or 0),
                "initial_failed_steps": int(c.get("initial_failed_steps", 0) or 0),
                "final_failed_steps": int(c.get("final_failed_steps", 0) or 0),
            }
        min_phys_gain = min(phys_gains)
        max_phys_loss = max(0.0, -min_phys_gain)
        min_total_gain = min(total_gains)
        min_time_gain = min(time_gains) if len(time_gains) == len(TERMINAL_MODES) else None
        min_rel_time_gain = min(rel_time_gains) if len(rel_time_gains) == len(TERMINAL_MODES) else None
        min_solver_time_gain = min(solver_time_gains) if len(solver_time_gains) == len(TERMINAL_MODES) else None
        min_step_delta = min(step_deltas) if len(step_deltas) == len(TERMINAL_MODES) else None
        row_h = {
            "horizon": h,
            "safe_all_modes": safe_all,
            "min_physical_gain": min_phys_gain,
            "max_physical_loss": max_phys_loss,
            "min_total_gain_synthetic": min_total_gain,
            "min_decision_time_gain_s": min_time_gain,
            "min_decision_rel_time_gain": min_rel_time_gain,
            "min_solver_time_gain_s": min_solver_time_gain,
            "min_step_delta_vs_H15": min_step_delta,
            "mode_metrics": mode_metrics,
        }
        for eps in EPSILONS:
            key = str(eps)
            physically_nonworse = safe_all and max_phys_loss <= eps
            timed = (min_time_gain is not None and min_rel_time_gain is not None)
            row_h[f"pareto_compute_eps_{key}"] = bool(physically_nonworse and timed and min_time_gain > 0.0)
            row_h[f"synthetic_total_positive_eps_{key}"] = bool(physically_nonworse and min_total_gain > 0.0)
            # Keep one best horizon per epsilon by robust decision-time gain with
            # physical loss as a tiebreaker.  This tests whether one constant H
            # dominates the measured-compute tradeoff.
            if physically_nonworse and timed:
                cur = best_by_eps[key]
                better = cur is None or (min_time_gain, -max_phys_loss, -abs(h - 15), -h) > (
                    finite(cur.get("min_decision_time_gain_s"), -1e99),
                    -finite(cur.get("max_physical_loss"), 1e99),
                    -abs(int(cur.get("horizon")) - 15),
                    -int(cur.get("horizon")),
                )
                if better:
                    best_by_eps[key] = row_h
        row_h["strict_compute_pareto"] = bool(
            safe_all
            and max_phys_loss <= STRICT_EPS
            and min_time_gain is not None
            and min_rel_time_gain is not None
            and min_time_gain >= STRICT_ABS_TIME_GAIN_S
            and min_rel_time_gain >= STRICT_REL_TIME_GAIN
        )
        row_h["relaxed_compute_pareto"] = bool(
            safe_all
            and max_phys_loss <= RELAXED_EPS
            and min_time_gain is not None
            and min_rel_time_gain is not None
            and min_time_gain >= RELAXED_ABS_TIME_GAIN_S
            and min_rel_time_gain >= RELAXED_REL_TIME_GAIN
        )
        horizon_rows.append(row_h)

    strict = [r for r in horizon_rows if r["strict_compute_pareto"]]
    relaxed = [r for r in horizon_rows if r["relaxed_compute_pareto"]]
    synthetic = [r for r in horizon_rows if r.get(f"synthetic_total_positive_eps_{STRICT_EPS}")]
    best_strict = None
    if strict:
        best_strict = max(strict, key=lambda r: (finite(r.get("min_decision_time_gain_s"), -1e99), -finite(r.get("max_physical_loss"), 1e99), -abs(int(r["horizon"]) - 15)))
    best_relaxed = None
    if relaxed:
        best_relaxed = max(relaxed, key=lambda r: (finite(r.get("min_decision_time_gain_s"), -1e99), -finite(r.get("max_physical_loss"), 1e99), -abs(int(r["horizon"]) - 15)))
    return {
        "state_id": row.get("state_id"),
        "case": int(row.get("case", -1)),
        "selection_group": row.get("selection_group"),
        "target_role": row.get("target_role"),
        "branch_step": int(row.get("branch_step", -1)),
        "predeclared_label": row.get("label"),
        "horizon_rows": horizon_rows,
        "best_by_eps": {k: (None if v is None else {kk: vv for kk, vv in v.items() if kk != "mode_metrics"}) for k, v in best_by_eps.items()},
        "strict_compute_pareto_horizons": [int(r["horizon"]) for r in strict],
        "relaxed_compute_pareto_horizons": [int(r["horizon"]) for r in relaxed],
        "synthetic_total_positive_horizons_strict_eps": [int(r["horizon"]) for r in synthetic],
        "best_strict_compute_pareto": None if best_strict is None else {k: v for k, v in best_strict.items() if k != "mode_metrics"},
        "best_relaxed_compute_pareto": None if best_relaxed is None else {k: v for k, v in best_relaxed.items() if k != "mode_metrics"},
    }


def run() -> int:
    inputs = verify_inputs()
    raw = inputs["raw"]
    state_rows = raw["analysis"]["state_rows"]
    per_state = [analyze_state(r) for r in state_rows]

    strict_states = [r for r in per_state if r["strict_compute_pareto_horizons"]]
    relaxed_states = [r for r in per_state if r["relaxed_compute_pareto_horizons"]]
    synthetic_states = [r for r in per_state if r["synthetic_total_positive_horizons_strict_eps"]]
    by_horizon = {str(h): {"strict_compute_pareto_states": 0, "relaxed_compute_pareto_states": 0, "synthetic_total_positive_states": 0, "best_eps_0.01_states": 0, "best_eps_0.1_states": 0} for h in HORIZONS if h != 15}
    strict_time_gains: List[float] = []
    relaxed_time_gains: List[float] = []
    strict_losses: List[float] = []
    relaxed_losses: List[float] = []
    step_delta_records: List[Dict[str, Any]] = []
    for st in per_state:
        for h in st["strict_compute_pareto_horizons"]:
            by_horizon[str(h)]["strict_compute_pareto_states"] += 1
        for h in st["relaxed_compute_pareto_horizons"]:
            by_horizon[str(h)]["relaxed_compute_pareto_states"] += 1
        for h in st["synthetic_total_positive_horizons_strict_eps"]:
            by_horizon[str(h)]["synthetic_total_positive_states"] += 1
        b001 = st["best_by_eps"].get(str(STRICT_EPS))
        if b001:
            by_horizon[str(int(b001["horizon"]))]["best_eps_0.01_states"] += 1
        b01 = st["best_by_eps"].get(str(RELAXED_EPS))
        if b01:
            by_horizon[str(int(b01["horizon"]))]["best_eps_0.1_states"] += 1
        if st["best_strict_compute_pareto"]:
            strict_time_gains.append(finite(st["best_strict_compute_pareto"].get("min_decision_time_gain_s")))
            strict_losses.append(finite(st["best_strict_compute_pareto"].get("max_physical_loss")))
        if st["best_relaxed_compute_pareto"]:
            relaxed_time_gains.append(finite(st["best_relaxed_compute_pareto"].get("min_decision_time_gain_s")))
            relaxed_losses.append(finite(st["best_relaxed_compute_pareto"].get("max_physical_loss")))
        for hr in st["horizon_rows"]:
            if hr.get("min_step_delta_vs_H15") not in (None, 0):
                step_delta_records.append({
                    "state_id": st["state_id"],
                    "case": st["case"],
                    "horizon": int(hr["horizon"]),
                    "min_step_delta_vs_H15": hr.get("min_step_delta_vs_H15"),
                    "min_decision_time_gain_s": hr.get("min_decision_time_gain_s"),
                    "max_physical_loss": hr.get("max_physical_loss"),
                    "strict_compute_pareto": hr.get("strict_compute_pareto"),
                    "relaxed_compute_pareto": hr.get("relaxed_compute_pareto"),
                })

    # If a single constant H is physically non-worse and faster in nearly every
    # state, that argues for a fixed-H comparator rather than adaptive training.
    dominant_h_strict = max(by_horizon.items(), key=lambda kv: kv[1]["strict_compute_pareto_states"])
    dominant_h_relaxed = max(by_horizon.items(), key=lambda kv: kv[1]["relaxed_compute_pareto_states"])
    best_eps001_h = max(by_horizon.items(), key=lambda kv: kv[1]["best_eps_0.01_states"])
    best_eps01_h = max(by_horizon.items(), key=lambda kv: kv[1]["best_eps_0.1_states"])

    if len(strict_states) >= 6 and dominant_h_strict[1]["strict_compute_pareto_states"] < len(strict_states):
        decision = "possible_compute_safe_selector_target_but_requires_fresh_confirmation_and_fixedH_timing_baseline"
        train_or_refit_now = False
        next_action = "freeze a compact IMPROVED compute-safe selector/refit smoke only after backup, with strong constant-H timing baselines and fresh confirmation; do not use validation64/sealed test"
    elif len(relaxed_states) >= 6 and dominant_h_relaxed[1]["relaxed_compute_pareto_states"] < len(relaxed_states):
        decision = "relaxed_compute_opportunity_only_requires_objective_scale_and_noise_audit_before_training"
        train_or_refit_now = False
        next_action = "audit timing noise/objective scale or run a small repeated-timing continuation confirmation before selector/refit"
    elif len(strict_states) > 0 or len(relaxed_states) > 0:
        decision = "compute_opportunity_sparse_or_constantH_absorbable"
        train_or_refit_now = False
        next_action = "compare against the dominant fixed-H/short-H baseline and avoid selector/refit unless fresh states show nonconstant opportunity"
    else:
        decision = "no_material_physical_or_measured_compute_opportunity_in_v1d_states"
        train_or_refit_now = False
        next_action = "write negative current-scenario opportunity diagnosis or freeze a versioned source-supported stress-scenario amendment before new rollouts"

    created = dt.datetime.now(dt.timezone.utc).isoformat()
    result = {
        "created_utc": created,
        "method": "vehicle_stress_v1d_pareto_timing_objective_postdiagnostic_v0_no_simulation",
        "classification": "development_no_simulation_tradeoff_postdiagnostic_no_validation64_no_sealed_test",
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
            "strict_abs_time_gain_s": STRICT_ABS_TIME_GAIN_S,
            "strict_rel_time_gain": STRICT_REL_TIME_GAIN,
            "relaxed_eps_physical_loss": RELAXED_EPS,
            "relaxed_abs_time_gain_s": RELAXED_ABS_TIME_GAIN_S,
            "relaxed_rel_time_gain": RELAXED_REL_TIME_GAIN,
            "note": "Measured decision-time thresholds are development diagnostics, not final acceptance criteria; timing noise still needs independent blocked confirmation before acceleration claims.",
        },
        "inputs": {
            "smoke_raw": rel(SMOKE_RAW),
            "smoke_raw_sha256": sha256(SMOKE_RAW),
            "smoke_completed": rel(SMOKE_DONE),
            "smoke_completed_sha256": sha256(SMOKE_DONE),
            "nearmiss_completed": rel(NEARMISS_DONE),
            "nearmiss_completed_sha256": sha256(NEARMISS_DONE),
        },
        "state_count": len(per_state),
        "strict_compute_pareto_state_count": len(strict_states),
        "relaxed_compute_pareto_state_count": len(relaxed_states),
        "synthetic_total_positive_state_count_strict_eps": len(synthetic_states),
        "strict_compute_pareto_cases": sorted(set(int(r["case"]) for r in strict_states)),
        "relaxed_compute_pareto_cases": sorted(set(int(r["case"]) for r in relaxed_states)),
        "by_horizon": by_horizon,
        "dominant_horizon_diagnostics": {
            "dominant_h_strict_compute": {"horizon": dominant_h_strict[0], **dominant_h_strict[1]},
            "dominant_h_relaxed_compute": {"horizon": dominant_h_relaxed[0], **dominant_h_relaxed[1]},
            "best_eps_0.01_horizon_by_state_count": {"horizon": best_eps001_h[0], **best_eps001_h[1]},
            "best_eps_0.1_horizon_by_state_count": {"horizon": best_eps01_h[0], **best_eps01_h[1]},
        },
        "time_gain_summary_strict_best_per_state_s": stat(strict_time_gains),
        "time_gain_summary_relaxed_best_per_state_s": stat(relaxed_time_gains),
        "physical_loss_summary_strict_best_per_state": stat(strict_losses),
        "physical_loss_summary_relaxed_best_per_state": stat(relaxed_losses),
        "step_delta_records_nonzero": step_delta_records,
        "per_state": per_state,
        "decision": {
            "classification": decision,
            "train_or_refit_now": train_or_refit_now,
            "next_action": next_action,
            "rationale": "This diagnostic asks whether a different learning target (compute-safe non-worsening) is supported after the physical-gain label gate failed; selector/refit remains deferred unless nonconstant, fresh and materially timed opportunity is confirmed.",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "raw.json", result)
    lines = [
        "# Vehicle stress-v1d Pareto/timing objective postdiagnostic v0",
        "",
        f"UTC: `{created}`. No simulations, no training/refit, no validation64/sealed-test access.",
        "",
        "## Why this was run",
        "",
        "The v1d physical-gain label gate failed, but sparse physical-improvement labels are not a universal blocker for adaptive-horizon learning. This diagnostic tests whether existing identical-prefix continuations contain a different actionable target: physically non-worse horizons with measured decision-time savings, while keeping synthetic horizon penalty separate from real timing.",
        "",
        "## Headline",
        "",
        f"- Strict compute-safe Pareto states (physical loss <= {STRICT_EPS}, time gain >= {STRICT_ABS_TIME_GAIN_S}s and >= {STRICT_REL_TIME_GAIN:.0%} in both terminal modes): `{len(strict_states)}/12`, cases `{sorted(set(int(r['case']) for r in strict_states))}`.",
        f"- Relaxed compute-safe Pareto states (physical loss <= {RELAXED_EPS}, time gain >= {RELAXED_ABS_TIME_GAIN_S}s in both terminal modes): `{len(relaxed_states)}/12`, cases `{sorted(set(int(r['case']) for r in relaxed_states))}`.",
        f"- Synthetic-total-positive states at strict physical epsilon: `{len(synthetic_states)}/12` (reported separately; not measured runtime).",
        f"- Dominant strict horizon diagnostic: `{result['dominant_horizon_diagnostics']['dominant_h_strict_compute']}`.",
        f"- Dominant relaxed horizon diagnostic: `{result['dominant_horizon_diagnostics']['dominant_h_relaxed_compute']}`.",
        f"- Strict best-per-state measured time gain summary (s): `{result['time_gain_summary_strict_best_per_state_s']}`.",
        f"- Relaxed best-per-state measured time gain summary (s): `{result['time_gain_summary_relaxed_best_per_state_s']}`.",
        f"- Nonzero continuation-length deltas observed in comparisons: `{len(step_delta_records)}`.",
        "",
        "## Per-state best tradeoff rows",
        "",
        "| state | case | group | branch | strict H | strict min time gain | strict max phys loss | relaxed H | relaxed min time gain | relaxed max phys loss |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for st in per_state:
        bs = st.get("best_strict_compute_pareto") or {}
        br = st.get("best_relaxed_compute_pareto") or {}
        lines.append(
            "| `%s` | %d | `%s` | %d | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` |" % (
                st["state_id"], int(st["case"]), st.get("selection_group"), int(st["branch_step"]),
                str(bs.get("horizon", "")), str(bs.get("min_decision_time_gain_s", "")), str(bs.get("max_physical_loss", "")),
                str(br.get("horizon", "")), str(br.get("min_decision_time_gain_s", "")), str(br.get("max_physical_loss", "")),
            )
        )
    lines += [
        "",
        "## Interpretation and decision",
        "",
        f"Decision classification: `{decision}`.",
        f"Train/refit now: `{train_or_refit_now}`.",
        f"Next action: {next_action}.",
        "",
        "This remains development evidence only. It does not support an acceleration claim without blocked/repeated timing and strong fixed-H baselines; it does not open validation64 or sealed test.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        f"# v1d Pareto/timing objective postdiagnostic state\n\nUTC: {created}. strict_compute_pareto={len(strict_states)}/12; relaxed_compute_pareto={len(relaxed_states)}/12; synthetic_total_positive={len(synthetic_states)}/12; decision={decision}; train_or_refit_now={train_or_refit_now}. Next: {next_action}. No simulations/training/validation/test.\n",
        encoding="utf-8",
    )
    write_json(BACKUP_REQ, {
        "requested_utc": created,
        "reason": "backup v1d Pareto/timing objective postdiagnostic before any further scenario simulation, selector/refit or training",
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
            "strict_compute_pareto_state_count": len(strict_states),
            "relaxed_compute_pareto_state_count": len(relaxed_states),
            "synthetic_total_positive_state_count_strict_eps": len(synthetic_states),
            "decision": decision,
            "train_or_refit_now": train_or_refit_now,
            "next_action": next_action,
        },
        "backup_request": rel(BACKUP_REQ),
        "hashes": {rel(p): sha256(p) for p in sorted([OUT / "raw.json", OUT / "summary.md", STATE, BACKUP_REQ, Path(__file__).resolve(), SMOKE_DONE, SMOKE_RAW, NEARMISS_DONE]) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "strict_compute_pareto_state_count": len(strict_states),
        "relaxed_compute_pareto_state_count": len(relaxed_states),
        "synthetic_total_positive_state_count_strict_eps": len(synthetic_states),
        "decision": decision,
        "train_or_refit_now": train_or_refit_now,
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
