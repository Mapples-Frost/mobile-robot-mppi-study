#!/usr/bin/env python3
"""v19 true-variable intermediate-horizon boundary acquisition (H12 vs H15).

Development-only IMPROVED diagnostic after v16b/v17/v18d showed that safe/useful
H10 selection does not generalize under strict opened-bank/source splits.

Hypothesis: the current failure may be partly horizon-grid/design, not only
selector representation.  H10 is a very aggressive short horizon: it is fast but
produces many high-physical-cost continuations near the mined boundary states.
A slightly longer true variable-dimension horizon (H12) may retain meaningful
measured solver/decision-time savings relative to H15 while avoiding some or all
H10 catastrophes.  If fixed H12 is already safe/useful, it becomes a strong fixed
baseline and weakens the need for H10 adaptation.  If H12 still has localized
catastrophes but an oracle H12/H15 switch has value, a follow-up selector/refit
should target H12/H15 rather than repeatedly thresholding H10.  If H12 is neither
safe nor useful, pivot to richer terminal/risk-value learning or scenario/
objective diagnostics.

This script reruns the same 24 v15 opened-development boundary states with a
blocked, interleaved true-H12 vs true-H15 paired schedule (2 repeats each).  It
uses the same case-snapshot recovery and true variable-dimension controller path
as v15/v0b.  It does not read validation64 or sealed-test banks, and it performs
no selector refit/search and no gradient/RL training.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_case5_smoke_v0_runner as case_runner  # noqa:E402
import vehicle_true_variable_horizon_v15_boundary_acquisition_v0b as v15b  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402

NAME = "vehicle_true_variable_horizon_intermediate_h12_boundary_v19"
STAMP = "20260930T0015Z"
SOURCE = Path(__file__).resolve()
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0015_after_intermediate_h12_boundary_v19.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_INTERMEDIATE_H12_BOUNDARY_V19_{STAMP}.json"
MARKER = f"vehicle-true-variable-H-intermediate-h12-boundary-v19-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

SHORT_H = 12
REF_H = 15
TRUE_HORIZONS = [SHORT_H, REF_H]
REPEATS_PER_HORIZON = 2
EXPECTED_CANDIDATES = 24
EXPECTED_EPISODES = EXPECTED_CANDIDATES * len(TRUE_HORIZONS) * REPEATS_PER_HORIZON
MAX_STEPS_PER_EPISODE = 150
CONTROL_STEP_CAP = EXPECTED_EPISODES * MAX_STEPS_PER_EPISODE
TERMINAL_PROFILE = "shared_h15_terminal"
MIN_DECISION_SAVING = 0.05

V15_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_20260929T2325Z/completed.json"
V15_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_20260929T2325Z/raw.json"
V18_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_probe_telemetry_v18d_20260930T0035Z/completed.json"


class ContractError(RuntimeError):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


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
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if hasattr(x, "item"):
        return clean(x.item())
    return x


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


def finite_summary(xs: Sequence[float]) -> Dict[str, Any]:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    def pct(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * q
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        return vals[lo] if lo == hi else vals[lo] * (hi - idx) + vals[hi] * (idx - lo)
    return {"n": len(vals), "min": vals[0], "median": pct(0.5), "mean": float(math.fsum(vals) / len(vals)), "p95": pct(0.95), "max": vals[-1], "sum": float(math.fsum(vals))}


def assert_dev_only_completed(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing prerequisite {label}: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True and obj.get("status") not in ("complete", "completed"):
        raise ContractError(f"prerequisite not complete: {label}")
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=true in {label}")
    return obj


def metric_sum(e: Mapping[str, Any], name: str) -> float:
    obj = e.get(name)
    if isinstance(obj, Mapping):
        return sf(obj.get("sum"), 0.0)
    return 0.0


def is_safe(e: Mapping[str, Any]) -> bool:
    return bool(e.get("success")) and not bool(e.get("constraint")) and si(e.get("solver_failure_steps"), 0) == 0 and si(e.get("initial_failed_steps"), 0) == 0 and si(e.get("final_failed_steps"), 0) == 0


def prepare_v15_candidates() -> List[Dict[str, Any]]:
    """Reuse v15/v0b's audited candidate recovery, without running v15."""
    assert_dev_only_completed(V15_DONE, "v15 boundary acquisition v0b")
    if V18_DONE.exists():
        assert_dev_only_completed(V18_DONE, "v18d probe telemetry")
    v15b.patch_base_runner()
    v15b.check_recovery_prerequisites()
    inputs = v15b.v0.verify_inputs()
    prepared = v15b.v0.prepare_candidates(inputs["candidates"], inputs["raw_by_bank"])
    if len(prepared) != EXPECTED_CANDIDATES:
        raise ContractError(f"expected {EXPECTED_CANDIDATES} prepared candidates, got {len(prepared)}")
    return [dict(c) for c in prepared]


def build_protocol(created: dt.datetime, prepared: Sequence[Mapping[str, Any]], backup_commit: str) -> Mapping[str, Any]:
    schedule: List[Dict[str, Any]] = []
    execution = 0
    for cand in prepared:
        ci = int(cand["candidate_index"])
        for rep in range(REPEATS_PER_HORIZON):
            # Interleave H12 and H15 within each candidate/repeat, alternating
            # order across candidates/repeats to reduce monotonic runtime drift.
            order = [SHORT_H, REF_H] if (ci + rep) % 2 == 0 else [REF_H, SHORT_H]
            for h in order:
                schedule.append({
                    "execution_index": execution,
                    "candidate_index": ci,
                    "repeat": rep,
                    "candidate_id": cand["candidate_id"],
                    "state_id": cand["candidate_id"],
                    "center_key": cand["source_key"],
                    "bank_id": cand["bank_id"],
                    "role": cand["role"],
                    "case": ci,
                    "source_candidate_index": int(cand["source_candidate_index"]),
                    "branch_step": int(cand["candidate_branch_step"]),
                    "true_mpc_n_horizon": int(h),
                    "commanded_horizon": int(h),
                    "terminal_mode": TERMINAL_PROFILE,
                    "initialization": "v19_intermediate_horizon_direct_branch_state_from_saved_H15_prefix_trace",
                })
                execution += 1
    if len(schedule) != EXPECTED_EPISODES:
        raise ContractError("unexpected v19 schedule length")
    input_paths = [SOURCE, V15_DONE, V15_RAW, V18_DONE, Path(v15b.__file__).resolve(), v15b.BASE_V0_SOURCE]
    proto = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_true_variable_intermediate_H12_boundary_acquisition_not_validation_not_test",
        "hypothesis": "H10 may be too aggressive on boundary states; true H12 may preserve measured compute savings versus H15 while materially reducing high-physical-cost continuations. Results discriminate horizon-grid/design from selector-representation failure.",
        "before_evidence": [
            "v15 H10/H15 boundary acquisition: 24 catastrophic H10 rows and 24 beneficial H10 rows over 48 pairs; fixed H10 saved 24.8% decision time but increased physical cost massively.",
            "v16b/v17/v18d strict nested selectors over H10 features could not obtain zero-catastrophe >=5% measured decision saving; v17 became safe only by selecting no H10.",
            "The user explicitly warned not to equate physical-improvement label gates with the whole control/compute tradeoff, and not to infer speed from H alone; this experiment measures actual H12/H15 decision and solver timing in blocked pairs.",
        ],
        "candidate_source": {"v15_completed": rel(V15_DONE), "v15_raw": rel(V15_RAW), "v18d_completed": rel(V18_DONE) if V18_DONE.exists() else None},
        "candidates": [{k: v for k, v in c.items() if k != "case_snapshot_from_candidate_pool"} for c in prepared],
        "candidate_role_counts": dict(Counter(str(c["role"]) for c in prepared)),
        "schedule": schedule,
        "schedule_ordering": "blocked by v15 candidate and repeat; H12/H15 order alternates by candidate index plus repeat",
        "terminal_profile": TERMINAL_PROFILE,
        "terminal_source_horizon": REF_H,
        "budget_declared": {"development_mpc_simulation_episodes": EXPECTED_EPISODES, "development_control_step_upper_bound": CONTROL_STEP_CAP, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "offline_model_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "decision_rules": {
            "fixed_H12_tradeoff_pass5": f"all H15 references safe, no catastrophic H12 rows, aggregate physical delta <= tolerance sum, and measured decision saving >= {MIN_DECISION_SAVING}",
            "oracle_H12_H15_value": "choose H12 only on safe-beneficial H12 rows and report actual measured decision/solver savings; this is an opportunity upper bound, not a deployable selector",
            "if_fixed_H12_passes": "treat H12 as a strong fixed-H baseline/tradeoff candidate and compare any future adaptive selector against tuned fixed H12/H15; do not claim adaptive advantage from H12 alone",
            "if_oracle_value_without_fixed_pass": "freeze a separate H12/H15 risk selector/refit with strict source/bank splits; avoid more H10 threshold sweeps",
            "if_no_value": "pivot to terminal/risk-value training or scenario/reward diagnostics because simply moving H10 to H12 does not solve the tradeoff",
        },
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "latest_verified_backup_before_run_from_supervisor_context": backup_commit,
        "input_hashes": {rel(p): sha256(p) for p in input_paths if p and p.exists()},
    }
    write_json(PROTOCOL, proto)
    return proto


def pair_summary(short_ep: Mapping[str, Any], ref_ep: Mapping[str, Any], cand: Mapping[str, Any], repeat: int) -> Dict[str, Any]:
    short_phys = sf(short_ep.get("physical_constraint_cost"), 0.0)
    ref_phys = sf(ref_ep.get("physical_constraint_cost"), 0.0)
    short_dec = metric_sum(short_ep, "decision_timing_s")
    ref_dec = metric_sum(ref_ep, "decision_timing_s")
    short_sol = metric_sum(short_ep, "solver_attempt_timing_s")
    ref_sol = metric_sum(ref_ep, "solver_attempt_timing_s")
    phys_delta = short_phys - ref_phys
    dec_delta = short_dec - ref_dec
    sol_delta = short_sol - ref_sol
    row_tol = max(2.0, 0.05 * abs(ref_phys))
    cat_threshold = max(2.0, 0.25 * abs(ref_phys))
    ref_safe = is_safe(ref_ep)
    short_safe = is_safe(short_ep)
    catastrophic = bool(ref_safe and ((not short_safe) or phys_delta > cat_threshold))
    beneficial = bool(ref_safe and short_safe and phys_delta <= row_tol and dec_delta < 0.0)
    return {
        "candidate_index": int(cand["candidate_index"]),
        "candidate_id": cand["candidate_id"],
        "repeat": int(repeat),
        "bank_id": cand["bank_id"],
        "source_key": cand["source_key"],
        "base_state_id": cand["base_state_id"],
        "role": cand["role"],
        "offset_from_center": int(cand["offset_from_center"]),
        "branch_step": int(cand["candidate_branch_step"]),
        "h_short": SHORT_H,
        "h_ref": REF_H,
        "short_safe": short_safe,
        "ref_safe": ref_safe,
        "short_success": bool(short_ep.get("success")),
        "ref_success": bool(ref_ep.get("success")),
        "short_constraint": bool(short_ep.get("constraint")),
        "ref_constraint": bool(ref_ep.get("constraint")),
        "short_solver_failure_steps": si(short_ep.get("solver_failure_steps"), 0),
        "ref_solver_failure_steps": si(ref_ep.get("solver_failure_steps"), 0),
        "short_steps": si(short_ep.get("steps"), 0),
        "ref_steps": si(ref_ep.get("steps"), 0),
        "short_opt_x_sizes": short_ep.get("opt_x_sizes_observed"),
        "ref_opt_x_sizes": ref_ep.get("opt_x_sizes_observed"),
        "short_physical": short_phys,
        "ref_physical": ref_phys,
        "physical_delta_short_minus_ref": phys_delta,
        "row_tolerance_vs_ref": row_tol,
        "catastrophic_threshold_vs_ref": cat_threshold,
        "short_catastrophic_vs_ref": catastrophic,
        "short_beneficial_vs_ref": beneficial,
        "short_decision_sum_s": short_dec,
        "ref_decision_sum_s": ref_dec,
        "decision_delta_short_minus_ref": dec_delta,
        "decision_gain_short_vs_ref_s": -dec_delta,
        "decision_relative_saving_short_vs_ref": None if ref_dec <= 0 else (ref_dec - short_dec) / ref_dec,
        "short_solver_sum_s": short_sol,
        "ref_solver_sum_s": ref_sol,
        "solver_delta_short_minus_ref": sol_delta,
        "solver_gain_short_vs_ref_s": -sol_delta,
        "solver_relative_saving_short_vs_ref": None if ref_sol <= 0 else (ref_sol - short_sol) / ref_sol,
        "short_path": short_ep.get("path"),
        "ref_path": ref_ep.get("path"),
    }


def analyze(episodes: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_key: Dict[Tuple[int, int, int], Mapping[str, Any]] = {}
    for e in episodes:
        by_key[(si(e.get("candidate_index"), -1), si(e.get("repeat"), -1), si(e.get("true_mpc_n_horizon"), -1))] = e
    pair_rows: List[Dict[str, Any]] = []
    cand_by_idx = {int(c["candidate_index"]): c for c in prepared}
    for ci, cand in sorted(cand_by_idx.items()):
        for rep in range(REPEATS_PER_HORIZON):
            sh = by_key.get((ci, rep, SHORT_H))
            rf = by_key.get((ci, rep, REF_H))
            if sh is None or rf is None:
                pair_rows.append({"candidate_index": ci, "candidate_id": cand["candidate_id"], "repeat": rep, "missing_pair": True, "role": cand["role"], "source_key": cand["source_key"]})
            else:
                pair_rows.append(pair_summary(sh, rf, cand, rep))
    valid = [r for r in pair_rows if not r.get("missing_pair")]
    cats = [r for r in valid if r.get("short_catastrophic_vs_ref") is True]
    bens = [r for r in valid if r.get("short_beneficial_vs_ref") is True]
    missing = [r for r in pair_rows if r.get("missing_pair")]
    all_pairs = not missing and len(pair_rows) == EXPECTED_CANDIDATES * REPEATS_PER_HORIZON
    all_ref_safe = all(r.get("ref_safe") is True for r in valid) and all_pairs
    fixed_ref_phys = math.fsum(sf(r.get("ref_physical"), 0.0) for r in valid)
    fixed_short_phys = math.fsum(sf(r.get("short_physical"), 0.0) for r in valid)
    fixed_ref_dec = math.fsum(sf(r.get("ref_decision_sum_s"), 0.0) for r in valid)
    fixed_short_dec = math.fsum(sf(r.get("short_decision_sum_s"), 0.0) for r in valid)
    fixed_ref_sol = math.fsum(sf(r.get("ref_solver_sum_s"), 0.0) for r in valid)
    fixed_short_sol = math.fsum(sf(r.get("short_solver_sum_s"), 0.0) for r in valid)
    tol_sum = math.fsum(sf(r.get("row_tolerance_vs_ref"), 0.0) for r in valid)
    phys_delta = fixed_short_phys - fixed_ref_phys
    dec_save = (fixed_ref_dec - fixed_short_dec) / fixed_ref_dec if fixed_ref_dec > 0 else 0.0
    sol_save = (fixed_ref_sol - fixed_short_sol) / fixed_ref_sol if fixed_ref_sol > 0 else 0.0
    # Oracle opportunity: choose H12 only where H12 is safe/beneficial by the predeclared label.
    oracle_phys = oracle_dec = oracle_sol = 0.0
    oracle_counts: Counter[str] = Counter()
    for r in valid:
        if r.get("short_beneficial_vs_ref") is True:
            oracle_counts[str(SHORT_H)] += 1
            oracle_phys += sf(r.get("short_physical"), 0.0)
            oracle_dec += sf(r.get("short_decision_sum_s"), 0.0)
            oracle_sol += sf(r.get("short_solver_sum_s"), 0.0)
        else:
            oracle_counts[str(REF_H)] += 1
            oracle_phys += sf(r.get("ref_physical"), 0.0)
            oracle_dec += sf(r.get("ref_decision_sum_s"), 0.0)
            oracle_sol += sf(r.get("ref_solver_sum_s"), 0.0)
    oracle_dec_save = (fixed_ref_dec - oracle_dec) / fixed_ref_dec if fixed_ref_dec > 0 else 0.0
    oracle_sol_save = (fixed_ref_sol - oracle_sol) / fixed_ref_sol if fixed_ref_sol > 0 else 0.0
    oracle_phys_delta = oracle_phys - fixed_ref_phys
    role_counts: Dict[str, Dict[str, int]] = {}
    for role in sorted({str(r.get("role")) for r in pair_rows}):
        rr = [r for r in valid if str(r.get("role")) == role]
        role_counts[role] = {
            "pairs": len(rr),
            "catastrophic_H12": sum(1 for r in rr if r.get("short_catastrophic_vs_ref") is True),
            "beneficial_H12": sum(1 for r in rr if r.get("short_beneficial_vs_ref") is True),
            "ref_unsafe": sum(1 for r in rr if r.get("ref_safe") is not True),
            "short_unsafe": sum(1 for r in rr if r.get("short_safe") is not True),
        }
    candidate_rows: List[Dict[str, Any]] = []
    for ci, cand in sorted(cand_by_idx.items()):
        rr = [r for r in valid if si(r.get("candidate_index"), -999) == ci]
        if not rr:
            candidate_rows.append({"candidate_index": ci, "candidate_id": cand["candidate_id"], "missing": True})
            continue
        candidate_rows.append({
            "candidate_index": ci,
            "candidate_id": cand["candidate_id"],
            "source_key": cand["source_key"],
            "bank_id": cand["bank_id"],
            "base_state_id": cand["base_state_id"],
            "role": cand["role"],
            "offset_from_center": int(cand["offset_from_center"]),
            "branch_step": int(cand["candidate_branch_step"]),
            "pair_repeats": len(rr),
            "catastrophic_H12_repeats": sum(1 for r in rr if r.get("short_catastrophic_vs_ref") is True),
            "beneficial_H12_repeats": sum(1 for r in rr if r.get("short_beneficial_vs_ref") is True),
            "all_ref_safe": all(r.get("ref_safe") is True for r in rr),
            "all_short_safe": all(r.get("short_safe") is True for r in rr),
            "mean_phys_delta_H12_minus_H15": math.fsum(sf(r.get("physical_delta_short_minus_ref"), 0.0) for r in rr) / len(rr),
            "mean_decision_gain_H12_vs_H15_s": math.fsum(sf(r.get("decision_gain_short_vs_ref_s"), 0.0) for r in rr) / len(rr),
            "mean_solver_gain_H12_vs_H15_s": math.fsum(sf(r.get("solver_gain_short_vs_ref_s"), 0.0) for r in rr) / len(rr),
            "short_opt_x_sizes": sorted({int(x) for r in rr for x in (r.get("short_opt_x_sizes") or [])}),
            "ref_opt_x_sizes": sorted({int(x) for r in rr for x in (r.get("ref_opt_x_sizes") or [])}),
        })
    fixed_pass5 = bool(all_pairs and all_ref_safe and not cats and phys_delta <= tol_sum and dec_save >= MIN_DECISION_SAVING)
    oracle_value5 = bool(all_pairs and all_ref_safe and oracle_dec_save >= MIN_DECISION_SAVING and oracle_phys_delta <= tol_sum)
    if fixed_pass5:
        decision = "Fixed true-H12 passes the opened-boundary H15-referenced tradeoff gate; next treat H12 as a strong fixed-H baseline/confirmation target and require any adaptive method to beat it fairly."
    elif oracle_value5 and cats:
        decision = "True-H12 still has localized catastrophic/high-cost rows, but an oracle H12/H15 switch has >=5% measured decision saving with physical gate; next freeze an H12/H15 selector/refit using strict bank/source splits rather than more H10 sweeps."
    elif oracle_value5:
        decision = "True-H12 oracle has value but fixed H12 does not pass; inspect timing/physical role distribution before a small H12/H15 selector refit."
    else:
        decision = "Intermediate H12 does not produce a sufficient safe/useful opened-boundary tradeoff; prioritize richer terminal/risk-value training/instrumentation or scenario/objective diagnostics rather than another horizon-threshold sweep."
    return {
        "pair_rows": pair_rows,
        "candidate_rows": candidate_rows,
        "role_counts": role_counts,
        "aggregate": {
            "pair_count": len(pair_rows),
            "valid_pair_count": len(valid),
            "expected_pair_count": EXPECTED_CANDIDATES * REPEATS_PER_HORIZON,
            "all_pairs_present": all_pairs,
            "all_H15_safe": all_ref_safe,
            "missing_pair_count": len(missing),
            "catastrophic_H12_rows": len(cats),
            "beneficial_H12_rows": len(bens),
            "fixed_H15_physical_sum": fixed_ref_phys,
            "fixed_H12_physical_sum": fixed_short_phys,
            "physical_delta_H12_minus_H15": phys_delta,
            "physical_tolerance_sum_vs_H15": tol_sum,
            "fixed_H15_decision_sum_s": fixed_ref_dec,
            "fixed_H12_decision_sum_s": fixed_short_dec,
            "fixed_H12_decision_relative_saving_vs_H15": dec_save,
            "fixed_H15_solver_sum_s": fixed_ref_sol,
            "fixed_H12_solver_sum_s": fixed_short_sol,
            "fixed_H12_solver_relative_saving_vs_H15": sol_save,
            "fixed_H12_tradeoff_pass5": fixed_pass5,
            "oracle_H12_H15_chosen_counts": dict(oracle_counts),
            "oracle_H12_H15_physical_delta_vs_H15": oracle_phys_delta,
            "oracle_H12_H15_decision_relative_saving_vs_H15": oracle_dec_save,
            "oracle_H12_H15_solver_relative_saving_vs_H15": oracle_sol_save,
            "oracle_H12_H15_value5": oracle_value5,
            "role_counts": role_counts,
            "decision_relative_saving_rows_summary": finite_summary([sf(r.get("decision_relative_saving_short_vs_ref"), 0.0) for r in valid]),
            "solver_relative_saving_rows_summary": finite_summary([sf(r.get("solver_relative_saving_short_vs_ref"), 0.0) for r in valid]),
        },
        "decision": decision,
        "catastrophic_rows": [{k: r.get(k) for k in ("candidate_id", "bank_id", "source_key", "role", "offset_from_center", "repeat", "physical_delta_short_minus_ref", "decision_gain_short_vs_ref_s", "short_steps", "ref_steps")} for r in cats],
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    ag = raw["analysis"]["aggregate"]
    lines = [
        "# Vehicle true-variable-H v19 intermediate H12 boundary acquisition",
        "",
        f"UTC `{raw['created_utc']}`. Development-only true-H12/H15 paired boundary run; no validation64, no sealed test, no selector refit/search, no gradient training.",
        "",
        "## Budget/access",
        "",
        f"- Episodes/control steps: `{raw['budget_actual']['development_mpc_simulation_episodes']}` / `{raw['budget_declared']['development_mpc_simulation_episodes']}` episodes; `{raw['budget_actual']['development_control_steps']}` / `{raw['budget_declared']['development_control_step_upper_bound']}` control steps.",
        f"- validation64_bank_opened: `{raw['validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Headline",
        "",
        f"- Pair rows `{ag['valid_pair_count']}` / `{ag['expected_pair_count']}`; all H15 safe `{ag['all_H15_safe']}`.",
        f"- Fixed H12 vs H15: decision saving `{100.0*sf(ag['fixed_H12_decision_relative_saving_vs_H15']):.2f}%`, solver saving `{100.0*sf(ag['fixed_H12_solver_relative_saving_vs_H15']):.2f}%`, physical delta `{ag['physical_delta_H12_minus_H15']:.6g}` (tolerance sum `{ag['physical_tolerance_sum_vs_H15']:.6g}`), catastrophic rows `{ag['catastrophic_H12_rows']}`, beneficial rows `{ag['beneficial_H12_rows']}`, pass5 `{ag['fixed_H12_tradeoff_pass5']}`.",
        f"- Oracle H12/H15: chosen counts `{ag['oracle_H12_H15_chosen_counts']}`, decision saving `{100.0*sf(ag['oracle_H12_H15_decision_relative_saving_vs_H15']):.2f}%`, solver saving `{100.0*sf(ag['oracle_H12_H15_solver_relative_saving_vs_H15']):.2f}%`, physical delta `{ag['oracle_H12_H15_physical_delta_vs_H15']:.6g}`, value5 `{ag['oracle_H12_H15_value5']}`.",
        f"- Role counts: `{ag['role_counts']}`.",
        f"- Decision: {raw['analysis']['decision']}",
        "",
        "## Candidate-level H12 label summary",
        "",
        "| idx | role | source | off | step | cat reps | ben reps | mean physΔ H12-H15 | mean decision gain s | opt_x H12/H15 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in raw["analysis"]["candidate_rows"]:
        if r.get("missing"):
            lines.append(f"| {r.get('candidate_index')} | MISSING | `{r.get('candidate_id')}` | | | | | | | |")
            continue
        lines.append("| %d | `%s` | `%s` | %d | %d | %d | %d | %.6g | %.6g | `%s/%s` |" % (
            si(r.get("candidate_index")), r.get("role"), r.get("source_key"), si(r.get("offset_from_center")), si(r.get("branch_step")),
            si(r.get("catastrophic_H12_repeats")), si(r.get("beneficial_H12_repeats")), sf(r.get("mean_phys_delta_H12_minus_H15")), sf(r.get("mean_decision_gain_H12_vs_H15_s")),
            r.get("short_opt_x_sizes"), r.get("ref_opt_x_sizes")))
    if raw["analysis"].get("catastrophic_rows"):
        lines += ["", "## Catastrophic/high-cost H12 rows", "", "| candidate | source | role | repeat | physΔ | decision gain s | steps H12/H15 |", "|---|---|---|---:|---:|---:|---:|"]
        for r in raw["analysis"]["catastrophic_rows"]:
            lines.append("| `%s` | `%s` | `%s` | %s | %.6g | %.6g | %s/%s |" % (r.get("candidate_id"), r.get("source_key"), r.get("role"), r.get("repeat"), sf(r.get("physical_delta_short_minus_ref")), sf(r.get("decision_gain_short_vs_ref_s")), r.get("short_steps"), r.get("ref_steps")))
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is opened-development boundary evidence on mined v15 candidate states. It tests whether an intermediate true horizon changes the control/compute tradeoff and provides timing evidence from blocked H12/H15 runs, but it is not validation64/final-test evidence and is not a deployable adaptive selector.",
        "",
        f"Protocol: `{rel(PROTOCOL)}`. Raw: `{rel(RUN_DIR/'raw.json')}`. Completed: `{rel(RUN_DIR/'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    ag = raw["analysis"]["aggregate"]
    block = f"""<!-- {MARKER} -->
## 2026-09-30 vehicle true-variable-H v19 intermediate H12 boundary acquisition

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only paired true-H12/H15 run on the 24 v15 boundary candidates; no validation64/sealed-test access, no selector refit/search, no gradient training. Budget actual: {raw['budget_actual']['development_mpc_simulation_episodes']} episodes, {raw['budget_actual']['development_control_steps']} control steps. Fixed H12 vs H15: decision saving={ag['fixed_H12_decision_relative_saving_vs_H15']}, solver saving={ag['fixed_H12_solver_relative_saving_vs_H15']}, physical delta={ag['physical_delta_H12_minus_H15']} (tolerance {ag['physical_tolerance_sum_vs_H15']}), catastrophic_H12={ag['catastrophic_H12_rows']}, beneficial_H12={ag['beneficial_H12_rows']}, fixed_pass5={ag['fixed_H12_tradeoff_pass5']}. Oracle H12/H15 decision saving={ag['oracle_H12_H15_decision_relative_saving_vs_H15']}, physical delta={ag['oracle_H12_H15_physical_delta_vs_H15']}, value5={ag['oracle_H12_H15_value5']}. Decision: {raw['analysis']['decision']}. Artifacts: `{rel(RUN_DIR/'summary.md')}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'completed.json')}`.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-intermediate-h12-boundary-v19", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_intermediate_h12_boundary_v19:
        raise ContractError("requires --run and explicit v19 development acknowledgement")
    if (RUN_DIR / "completed.json").exists():
        done = read_json(RUN_DIR / "completed.json")
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name != "run.lock" for p in RUN_DIR.iterdir()):
        raise ContractError("partial run output exists; inspect before rerun: " + rel(RUN_DIR))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    prepared = prepare_v15_candidates()
    protocol = build_protocol(now(), prepared, args.backup_verified_commit)
    case_runner.SMOKE_DIR = RUN_DIR
    _, stage1_runner, _ = v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
    stage1_runner.base.v1.latency_verify()
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    if REF_H not in terminals:
        raise ContractError("terminal grid missing H15")
    write_json(RUN_DIR / "run_started.json", {"started_utc": now().isoformat(), "pid": os.getpid(), "method": NAME, "backup_verified_commit_from_supervisor_context": args.backup_verified_commit, "validation64_bank_opened": False, "sealed_test_accessed": False, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0})
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})
    cand_by_idx = {int(c["candidate_index"]): c for c in prepared}
    episodes: List[Dict[str, Any]] = []
    for item in protocol["schedule"]:
        cand = cand_by_idx[int(item["candidate_index"])]
        run_item = dict(item)
        run_item["branch_previous_state"] = copy.deepcopy(cand["branch_previous_state"])
        case = cand["case_snapshot_from_candidate_pool"]
        summary = case_runner.run_true_h_episode(run_item, case, terminals[REF_H])
        summary["stage"] = "v19_intermediate_horizon_H12_H15_boundary"
        summary["candidate_index"] = int(item["candidate_index"])
        summary["candidate_id"] = cand["candidate_id"]
        summary["repeat"] = int(item["repeat"])
        summary["center_key"] = cand["source_key"]
        summary["bank_id"] = cand["bank_id"]
        summary["base_state_id"] = cand["base_state_id"]
        summary["role"] = cand["role"]
        summary["offset_from_center"] = int(cand["offset_from_center"])
        summary["source_group"] = cand.get("source_group")
        summary["source_window"] = cand.get("source_window")
        summary["terminal_source_horizon"] = REF_H
        summary["terminal_receipt_effective"] = terminal_receipts.get(str(REF_H)) or terminal_receipts.get(REF_H)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "episodes_done": len(episodes), "episodes_expected": EXPECTED_EPISODES, "control_steps_done": int(sum(si(e.get("steps"), 0) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("candidate_index", "role", "center_key", "repeat", "true_mpc_n_horizon", "steps", "success", "termination", "physical_constraint_cost")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(si(e.get("steps"), 0) for e in episodes))
    if len(episodes) != EXPECTED_EPISODES:
        raise ContractError("episode count mismatch")
    if control_steps > CONTROL_STEP_CAP:
        raise ContractError("control-step budget exceeded")
    analysis = analyze(episodes, prepared)
    created = now()
    write_json(BACKUP_REQUEST, {"requested_utc": created.isoformat(), "reason": "backup after v19 intermediate H12/H15 boundary acquisition before any H12 selector/refit or further simulations", "required_before_more_unique_science": True, "development_mpc_simulation_episodes": len(episodes), "development_control_steps": control_steps, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(SOURCE), rel(PROTOCOL), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"]})
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_true_variable_intermediate_H12_boundary_acquisition_not_validation_not_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "runtime_preflight": preflight,
        "terminal_sources": {str(k): v for k, v in terminal_receipts.items()},
        "candidates_preoutcome": [{k: v for k, v in c.items() if k != "case_snapshot_from_candidate_pool"} for c in prepared],
        "episodes": episodes,
        "analysis": analysis,
        "budget_declared": {"development_mpc_simulation_episodes": EXPECTED_EPISODES, "development_control_step_upper_bound": CONTROL_STEP_CAP, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "offline_model_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"development_mpc_simulation_episodes": len(episodes), "development_control_steps": control_steps, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "offline_model_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "platform": {"python": sys.version, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "backup_request": rel(BACKUP_REQUEST),
        "interpretation_limits": ["opened v15 development boundary candidates", "blocked timing but same host only", "not validation/test", "not a deployable selector", "not ORIGINAL SAC"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    append_docs(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"""# Continue state after v19 intermediate H12 boundary acquisition

UTC: {created.isoformat()}
Elapsed since first supervisor event: {(created - FIRST_EVENT).total_seconds()/3600.0:.2f} h.

Completed `{NAME}` development-only paired true-H12/H15 acquisition: {len(episodes)} episodes, {control_steps} control steps. No validation64, no sealed test, no selector refit/search, no training.

Aggregate: {analysis['aggregate']}
Decision: {analysis['decision']}
Summary: `{rel(RUN_DIR / 'summary.md')}`
Raw: `{rel(RUN_DIR / 'raw.json')}`
Completed: `{rel(RUN_DIR / 'completed.json')}`
Backup request: `{rel(BACKUP_REQUEST)}`

Next action after verified backup: follow the decision branch above. If oracle_H12_H15_value5 is true without fixed_H12_tradeoff_pass5, freeze an H12/H15 selector/refit with strict leave-bank/source splits and strong fixed-H12/H15 comparison; otherwise pivot to richer terminal/risk-value training/instrumentation or scenario/objective diagnostics.
""", encoding="utf-8")
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, BACKUP_REQUEST, V15_DONE, V15_RAW, V18_DONE]
    completed = {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "budget_actual": raw["budget_actual"], "headline": analysis["aggregate"], "decision": analysis["decision"], "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQUEST), "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}}
    write_json(RUN_DIR / "completed.json", completed)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": analysis["aggregate"], "decision": analysis["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(BACKUP_REQUEST)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RUN_DIR / "failure.json", {"failed_utc": now().isoformat(), "error": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "budget_declared": {"development_mpc_simulation_episodes": EXPECTED_EPISODES, "development_control_step_upper_bound": CONTROL_STEP_CAP}, "next_recovery_hint": "Preserve partial outputs. If source repair is needed, version it and request backup before rerun; do not overwrite episode directories."})
        raise
