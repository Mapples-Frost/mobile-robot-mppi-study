#!/usr/bin/env python3
"""v15 local boundary acquisition for true-variable-H H10/H15.

Development-only IMPROVED diagnostic following v14 calibrated-risk failure and
v15 boundary-centre smoke.  It executes the predeclared 24 saved H15-prefix
candidate branch states from the v14 false-positive-neighbor audit: the two
nested false-positive centres at offsets -4/0/+4, their nearest safe positive
lookalikes, and their nearest catastrophic lookalikes.  Each candidate is run as
a paired true-H continuation under H10 and H15, with two deterministic repeats
and the shared-H15 terminal profile.

No validation64 bank access, no sealed test access, no selector refit/search and
no gradient/RL training.  Budget: 96 development MPC continuation episodes,
<=14400 control steps.  The purpose is to acquire local risk-boundary labels for
a subsequent conservative selector/value refit if, and only if, the labels are
interpretable and include both safe-beneficial and catastrophic H10 outcomes.
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
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_case5_smoke_v0_runner as case_runner  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402
import vehicle_true_variable_horizon_v15_boundary_smoke_v0 as smoke  # noqa:E402

NAME = "vehicle_true_variable_horizon_v15_boundary_acquisition_v0"
STAMP = "20260929T2255Z"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

AUDIT = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/audit.json"
AUDIT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/completed.json"
PREFLIGHT = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_candidate_trace_preflight_v0b_20260929T2235Z/preflight.json"
PREFLIGHT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_candidate_trace_preflight_v0b_20260929T2235Z/completed.json"
SMOKE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_smoke_v0_20260929T2245Z/completed.json"
SMOKE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_smoke_v0_20260929T2245Z/raw.json"

RAW_BY_BANK_PATHS = {
    "fresh_v0": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/raw.json",
    "fresh_v1": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/raw.json",
    "fresh_v2": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/raw.json",
    "fresh_v8c": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/raw.json",
    "fresh_v11": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/raw.json",
}
DONE_BY_BANK_PATHS = {
    "fresh_v0": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/completed.json",
    "fresh_v1": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/completed.json",
    "fresh_v2": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/completed.json",
    "fresh_v8c": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/completed.json",
    "fresh_v11": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/completed.json",
}

RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T2255_after_v15_boundary_acquisition_v0.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
BACKUP_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V15_BOUNDARY_ACQUISITION_V0_{STAMP}.json"
MARKER = f"vehicle-true-variable-H-v15-boundary-acquisition-v0-{STAMP}"

TRUE_HORIZONS = [10, 15]
REPEATS_PER_HORIZON = 2
EXPECTED_CANDIDATES = 24
EXPECTED_EPISODES = EXPECTED_CANDIDATES * len(TRUE_HORIZONS) * REPEATS_PER_HORIZON
MAX_STEPS_PER_EPISODE = 150
CONTROL_STEP_CAP = EXPECTED_EPISODES * MAX_STEPS_PER_EPISODE
PRIMARY_TERMINAL_PROFILE = "shared_h15_terminal"


class ContractError(RuntimeError):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    return smoke.rel(path)


def clean(x: Any) -> Any:
    return smoke.clean(x)


def read_json(path: Path) -> Any:
    return smoke.read_json(path)


def write_json(path: Path, value: Any) -> None:
    smoke.write_json(path, value)


def sha256(path: Path) -> str:
    return smoke.sha256(path)


def sf(x: Any, default: float = 0.0) -> float:
    return smoke.sf(x, default)


def si(x: Any, default: int = 0) -> int:
    return smoke.si(x, default)


def safe_id(s: str) -> str:
    return smoke.safe_id(s)


def metric_sum(e: Mapping[str, Any], name: str) -> float:
    return smoke.metric_sum(e, name)


def is_safe(e: Mapping[str, Any]) -> bool:
    return smoke.is_safe(e)


def canonical_sha(value: Any) -> str:
    return smoke.canonical_sha(value)


def source_index_from_trace(trace_path: str) -> int:
    m = re.search(r"source(\d+)", trace_path)
    if not m:
        raise ContractError("cannot infer source_candidate_index from trace path: " + trace_path)
    return int(m.group(1))


def verify_inputs() -> Dict[str, Any]:
    smoke.completed_dev_only(AUDIT_DONE, "v14 false-positive audit")
    smoke.completed_dev_only(PREFLIGHT_DONE, "v15 candidate preflight")
    smoke.completed_dev_only(SMOKE_DONE, "v15 centre smoke")
    for bank, done_path in DONE_BY_BANK_PATHS.items():
        smoke.completed_dev_only(done_path, f"{bank} opened development bank")
    audit = read_json(AUDIT)
    preflight = read_json(PREFLIGHT)
    smoke_raw = read_json(SMOKE_RAW)
    for label, obj in (("audit", audit), ("preflight", preflight), ("smoke_raw", smoke_raw)):
        for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
            if obj.get(flag) is True:
                raise ContractError(f"forbidden {flag}=True in {label}")
    h = preflight.get("headline") or {}
    if si(h.get("candidate_branch_states"), -1) != EXPECTED_CANDIDATES:
        raise ContractError("unexpected preflight candidate count")
    if si(h.get("blocked_candidate_count"), 999) != 0:
        raise ContractError("preflight had blocked candidates")
    candidates = list(((audit.get("v15_candidate_plan") or {}).get("candidate_branch_states") or []))
    if len(candidates) != EXPECTED_CANDIDATES:
        raise ContractError(f"expected {EXPECTED_CANDIDATES} v15 candidates, got {len(candidates)}")
    roles = Counter(str(c.get("role")) for c in candidates)
    expected_roles = {"false_positive_center": 6, "nearest_safe_positive_lookalike": 12, "nearest_catastrophic_lookalike": 6}
    if dict(roles) != expected_roles:
        raise ContractError(f"unexpected candidate role counts {dict(roles)}")
    sm_h = (read_json(SMOKE_DONE).get("headline") or {})
    if si(sm_h.get("catastrophic_h10_rows"), -1) != 2 or sm_h.get("all_h15_safe") is not True:
        raise ContractError("v15 centre smoke did not reproduce both false-positive centres")
    raw_by_bank = {bank: read_json(path) for bank, path in RAW_BY_BANK_PATHS.items()}
    return {"audit": audit, "preflight": preflight, "smoke_raw": smoke_raw, "candidates": candidates, "raw_by_bank": raw_by_bank}


def prepare_candidates(candidates: Sequence[Mapping[str, Any]], raw_by_bank: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for idx, c in enumerate(candidates):
        bank_id = str(c.get("bank_id"))
        if bank_id not in raw_by_bank:
            raise ContractError("unsupported bank in v15 candidate plan: " + bank_id)
        trace_path = str(c.get("h15_trace_episode_path"))
        source_idx = source_index_from_trace(trace_path)
        step = si(c.get("candidate_branch_step"), -1)
        if step < 0:
            raise ContractError("candidate lacks branch step")
        row = smoke.trace_row_at(trace_path, step)
        branch_state = row.get("previous_state") or row.get("state")
        obs = row.get("observation") or row.get("next_observation") or []
        case = smoke.find_case_snapshot(bank_id, source_idx, raw_by_bank)
        candidate_id = safe_id(
            f"v15c{idx:02d}_{c.get('role')}_{bank_id}_{c.get('base_state_id')}_step{step:03d}_off{si(c.get('offset_from_center'), 0):+d}"
        )
        prepared.append({
            "candidate_index": idx,
            "candidate_id": candidate_id,
            "bank_id": bank_id,
            "base_state_id": str(c.get("base_state_id")),
            "source_key": str(c.get("source_key")),
            "source_candidate_index": source_idx,
            "role": str(c.get("role")),
            "offset_from_center": si(c.get("offset_from_center"), 0),
            "candidate_branch_step": step,
            "center_branch_step": si(c.get("center_branch_step"), -1),
            "h15_trace_episode_path": trace_path,
            "trace_row_step": si(row.get("step"), -1),
            "branch_previous_state": smoke.state_core(branch_state),
            "initial_observation_from_h15_trace": copy.deepcopy(obs),
            "case_snapshot_sha256": canonical_sha(case),
            "case_snapshot_from_candidate_pool": case,
            "source_group": c.get("source_group"),
            "source_window": c.get("source_window"),
            "source_label_positive": bool(c.get("source_label_positive")),
            "source_catastrophic": bool(c.get("source_catastrophic")),
            "source_phys_delta": sf(c.get("source_phys_delta"), 0.0),
            "source_decision_gain_s": sf(c.get("source_decision_gain_s"), 0.0),
        })
    return prepared


def build_protocol(created: dt.datetime, prepared: Sequence[Mapping[str, Any]], backup_commit: str, input_hashes: Mapping[str, str]) -> Dict[str, Any]:
    schedule: List[Dict[str, Any]] = []
    execution = 0
    for cand in prepared:
        ci = int(cand["candidate_index"])
        for rep in range(REPEATS_PER_HORIZON):
            order = [10, 15] if (ci + rep) % 2 == 0 else [15, 10]
            for h in order:
                schedule.append({
                    "execution_index": execution,
                    "candidate_index": ci,
                    "repeat": rep,
                    "candidate_id": cand["candidate_id"],
                    "center_key": cand["source_key"],
                    "state_id": cand["candidate_id"],
                    "case": ci,
                    "source_candidate_index": int(cand["source_candidate_index"]),
                    "branch_step": int(cand["candidate_branch_step"]),
                    "true_mpc_n_horizon": int(h),
                    "commanded_horizon": int(h),
                    "terminal_mode": PRIMARY_TERMINAL_PROFILE,
                    "initialization": "v15_boundary_acquisition_direct_branch_state_from_saved_H15_prefix_trace",
                })
                execution += 1
    if len(schedule) != EXPECTED_EPISODES:
        raise ContractError("unexpected full v15 schedule length")
    proto = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_true_variable_H_local_boundary_acquisition_preoutcome_not_validation_not_test",
        "hypothesis": "Local continuation labels at the two v14 false-positive boundaries and their nearest safe/catastrophic lookalikes will reveal whether current engineered features fail because sparse local risk-boundary coverage is missing. If the acquisition yields both reproducible safe-beneficial and catastrophic H10 outcomes with safe H15 references, a subsequent conservative refit can test separability; otherwise richer terminal/risk-value learning or additional observability is needed.",
        "before_evidence": [
            "v14 nested calibrated risk/value refit saved only 2.92% and had two catastrophic H10 false positives; best global zero-catastrophe config saved only 2.08%.",
            "v14 false-positive audit selected 24 local boundary candidates: false-positive centres, nearest safe positive lookalikes and nearest catastrophic lookalikes.",
            "v15 preflight v0b verified all 24 candidates have existing H15-prefix trace step coverage.",
            "v15 centre smoke reproduced both false-positive centre H10 catastrophes with safe H15 pairs in 4 development episodes.",
        ],
        "candidates": [{k: v for k, v in c.items() if k != "case_snapshot_from_candidate_pool"} for c in prepared],
        "candidate_role_counts": dict(Counter(str(c["role"]) for c in prepared)),
        "schedule": schedule,
        "schedule_ordering": "deterministic blocked-by-candidate two repeats; H10/H15 order alternates by candidate index and repeat to reduce monotonic timing drift",
        "terminal_profile": PRIMARY_TERMINAL_PROFILE,
        "terminal_source_horizon": 15,
        "budget_declared": {"development_mpc_simulation_episodes": EXPECTED_EPISODES, "development_control_step_upper_bound": CONTROL_STEP_CAP, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "decision_rules": {
            "acquisition_complete": "96 scheduled episodes finish within 14400 control steps with all H10/H15 pairs present",
            "label_data_sufficient_for_refit": "all H15 references safe and the pair labels contain at least 8 catastrophic H10 rows plus at least 8 beneficial H10 rows, with catastrophic rows among false-positive centres and beneficial rows among nearest safe lookalikes",
            "if_sufficient": "freeze a separate development-only conservative boundary-augmented refit; require zero catastrophic H10 and >=5% opened-development saving before any unused-source confirmation",
            "if_not_sufficient": "do not do another unchanged static-feature sweep; inspect whether the failure is replay mismatch, scenario boundary scarcity, terminal-value/observability limits or true lack of local adaptive opportunity",
        },
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "latest_verified_backup_before_run_from_supervisor_context": backup_commit,
        "input_hashes": input_hashes,
    }
    write_json(PROTOCOL, proto)
    return proto


def summarize_pair(h10: Mapping[str, Any], h15: Mapping[str, Any], cand: Mapping[str, Any], repeat: int) -> Dict[str, Any]:
    h10_phys = sf(h10.get("physical_constraint_cost"), 0.0)
    h15_phys = sf(h15.get("physical_constraint_cost"), 0.0)
    h10_dec = metric_sum(h10, "decision_timing_s")
    h15_dec = metric_sum(h15, "decision_timing_s")
    h10_solver = metric_sum(h10, "solver_attempt_timing_s")
    h15_solver = metric_sum(h15, "solver_attempt_timing_s")
    phys_delta = h10_phys - h15_phys
    decision_delta = h10_dec - h15_dec
    solver_delta = h10_solver - h15_solver
    row_tol = max(2.0, 0.05 * abs(h15_phys))
    catastrophic_threshold = max(2.0, 0.25 * abs(h15_phys))
    h15_safe = is_safe(h15)
    h10_safe = is_safe(h10)
    catastrophic = bool(h15_safe and ((not h10_safe) or phys_delta > catastrophic_threshold))
    beneficial = bool(h15_safe and h10_safe and phys_delta <= row_tol and decision_delta < 0.0)
    return {
        "candidate_index": int(cand["candidate_index"]),
        "candidate_id": cand["candidate_id"],
        "repeat": int(repeat),
        "source_key": cand["source_key"],
        "bank_id": cand["bank_id"],
        "base_state_id": cand["base_state_id"],
        "role": cand["role"],
        "offset_from_center": int(cand["offset_from_center"]),
        "source_group": cand.get("source_group"),
        "source_window": cand.get("source_window"),
        "branch_step": int(cand["candidate_branch_step"]),
        "h10_safe": h10_safe,
        "h15_safe": h15_safe,
        "h10_success": bool(h10.get("success")),
        "h15_success": bool(h15.get("success")),
        "h10_steps": si(h10.get("steps"), 0),
        "h15_steps": si(h15.get("steps"), 0),
        "h10_opt_x_sizes": h10.get("opt_x_sizes_observed"),
        "h15_opt_x_sizes": h15.get("opt_x_sizes_observed"),
        "h10_physical": h10_phys,
        "h15_physical": h15_phys,
        "physical_delta_h10_minus_h15": phys_delta,
        "row_tolerance_vs_H15": row_tol,
        "catastrophic_threshold": catastrophic_threshold,
        "h10_catastrophic_vs_h15": catastrophic,
        "h10_beneficial_vs_h15": beneficial,
        "h10_decision_sum_s": h10_dec,
        "h15_decision_sum_s": h15_dec,
        "decision_delta_h10_minus_h15": decision_delta,
        "decision_gain_h10_vs_h15_s": -decision_delta,
        "h10_solver_sum_s": h10_solver,
        "h15_solver_sum_s": h15_solver,
        "solver_delta_h10_minus_h15": solver_delta,
        "solver_gain_h10_vs_h15_s": -solver_delta,
        "source_expected_catastrophic": bool(cand.get("source_catastrophic")),
        "source_expected_positive": bool(cand.get("source_label_positive")),
        "source_phys_delta": cand.get("source_phys_delta"),
        "source_decision_gain_s": cand.get("source_decision_gain_s"),
        "h10_path": h10.get("path"),
        "h15_path": h15.get("path"),
        "max_branch_reset_distance": max(sf((h10.get("branch_reset") or {}).get("branch_state_distance_after_reset"), 999.0), sf((h15.get("branch_reset") or {}).get("branch_state_distance_after_reset"), 999.0)),
    }


def analyze(episodes: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_key: Dict[Tuple[int, int, int], Mapping[str, Any]] = {}
    for e in episodes:
        by_key[(si(e.get("candidate_index"), -1), si(e.get("repeat"), -1), si(e.get("true_mpc_n_horizon"), -1))] = e
    cand_by_idx = {int(c["candidate_index"]): c for c in prepared}
    pair_rows: List[Dict[str, Any]] = []
    for cand in prepared:
        ci = int(cand["candidate_index"])
        for rep in range(REPEATS_PER_HORIZON):
            h10 = by_key.get((ci, rep, 10))
            h15 = by_key.get((ci, rep, 15))
            if h10 is None or h15 is None:
                pair_rows.append({"candidate_index": ci, "candidate_id": cand["candidate_id"], "repeat": rep, "missing_pair": True, "role": cand["role"], "source_key": cand["source_key"]})
            else:
                pair_rows.append(summarize_pair(h10, h15, cand, rep))
    cats = [r for r in pair_rows if r.get("h10_catastrophic_vs_h15") is True]
    bens = [r for r in pair_rows if r.get("h10_beneficial_vs_h15") is True]
    missing = [r for r in pair_rows if r.get("missing_pair")]
    all_pairs = not missing and len(pair_rows) == EXPECTED_CANDIDATES * REPEATS_PER_HORIZON
    all_h15_safe = all(r.get("h15_safe") is True for r in pair_rows if not r.get("missing_pair"))
    fixed15_dec = math.fsum(sf(r.get("h15_decision_sum_s"), 0.0) for r in pair_rows)
    fixed10_dec = math.fsum(sf(r.get("h10_decision_sum_s"), 0.0) for r in pair_rows)
    fixed15_solver = math.fsum(sf(r.get("h15_solver_sum_s"), 0.0) for r in pair_rows)
    fixed10_solver = math.fsum(sf(r.get("h10_solver_sum_s"), 0.0) for r in pair_rows)
    fixed15_phys = math.fsum(sf(r.get("h15_physical"), 0.0) for r in pair_rows)
    fixed10_phys = math.fsum(sf(r.get("h10_physical"), 0.0) for r in pair_rows)
    role_counts: Dict[str, Dict[str, int]] = {}
    for role in sorted(set(str(r.get("role")) for r in pair_rows)):
        rr = [r for r in pair_rows if str(r.get("role")) == role]
        role_counts[role] = {
            "pairs": len(rr),
            "catastrophic_h10": sum(1 for r in rr if r.get("h10_catastrophic_vs_h15") is True),
            "beneficial_h10": sum(1 for r in rr if r.get("h10_beneficial_vs_h15") is True),
            "h15_unsafe": sum(1 for r in rr if r.get("h15_safe") is not True),
        }
    by_candidate: List[Dict[str, Any]] = []
    for ci, cand in sorted(cand_by_idx.items()):
        rr = [r for r in pair_rows if si(r.get("candidate_index"), -999) == ci and not r.get("missing_pair")]
        if not rr:
            by_candidate.append({"candidate_index": ci, "candidate_id": cand["candidate_id"], "missing": True})
            continue
        by_candidate.append({
            "candidate_index": ci,
            "candidate_id": cand["candidate_id"],
            "source_key": cand["source_key"],
            "bank_id": cand["bank_id"],
            "base_state_id": cand["base_state_id"],
            "role": cand["role"],
            "offset_from_center": int(cand["offset_from_center"]),
            "branch_step": int(cand["candidate_branch_step"]),
            "pair_repeats": len(rr),
            "catastrophic_h10_repeats": sum(1 for r in rr if r.get("h10_catastrophic_vs_h15") is True),
            "beneficial_h10_repeats": sum(1 for r in rr if r.get("h10_beneficial_vs_h15") is True),
            "all_h15_safe": all(r.get("h15_safe") is True for r in rr),
            "mean_phys_delta_h10_minus_h15": math.fsum(sf(r.get("physical_delta_h10_minus_h15"), 0.0) for r in rr) / len(rr),
            "mean_decision_gain_h10_vs_h15_s": math.fsum(sf(r.get("decision_gain_h10_vs_h15_s"), 0.0) for r in rr) / len(rr),
            "mean_h10_physical": math.fsum(sf(r.get("h10_physical"), 0.0) for r in rr) / len(rr),
            "mean_h15_physical": math.fsum(sf(r.get("h15_physical"), 0.0) for r in rr) / len(rr),
        })
    source_counts: Dict[str, Dict[str, int]] = {}
    for sk in sorted(set(str(r.get("source_key")) for r in pair_rows)):
        rr = [r for r in pair_rows if str(r.get("source_key")) == sk]
        source_counts[sk] = {
            "pairs": len(rr),
            "catastrophic_h10": sum(1 for r in rr if r.get("h10_catastrophic_vs_h15") is True),
            "beneficial_h10": sum(1 for r in rr if r.get("h10_beneficial_vs_h15") is True),
            "h15_unsafe": sum(1 for r in rr if r.get("h15_safe") is not True),
        }
    label_data_sufficient = bool(
        all_pairs
        and all_h15_safe
        and len(cats) >= 8
        and len(bens) >= 8
        and role_counts.get("false_positive_center", {}).get("catastrophic_h10", 0) >= 4
        and role_counts.get("nearest_safe_positive_lookalike", {}).get("beneficial_h10", 0) >= 4
    )
    if label_data_sufficient:
        decision = "v15 boundary acquisition produced interpretable local labels with both catastrophic and safe-beneficial H10 outcomes; next freeze a boundary-augmented conservative refit, still development-only and requiring zero catastrophic H10 plus >=5% opened-development saving before any unused-source confirmation."
    elif all_pairs and all_h15_safe:
        decision = "v15 boundary acquisition completed but label mixture is insufficient for the planned boundary-augmented selector refit; inspect role/source label distribution and consider richer risk/terminal-value learning rather than another static-feature sweep."
    else:
        decision = "v15 boundary acquisition did not meet replay/safety completeness; inspect missing/unsafe H15 pairs before any refit or validation rollout."
    return {
        "pair_rows": pair_rows,
        "candidate_rows": by_candidate,
        "source_counts": source_counts,
        "aggregate": {
            "pair_count": len(pair_rows),
            "expected_pair_count": EXPECTED_CANDIDATES * REPEATS_PER_HORIZON,
            "all_pairs_present": all_pairs,
            "all_h15_safe": all_h15_safe,
            "missing_pair_count": len(missing),
            "catastrophic_h10_rows": len(cats),
            "beneficial_h10_rows": len(bens),
            "role_counts": role_counts,
            "fixed_H15_physical_sum": fixed15_phys,
            "fixed_H10_physical_sum": fixed10_phys,
            "fixed_H15_decision_sum_s": fixed15_dec,
            "fixed_H10_decision_sum_s": fixed10_dec,
            "fixed_H10_decision_relative_saving_vs_H15": None if fixed15_dec <= 0 else (fixed15_dec - fixed10_dec) / fixed15_dec,
            "fixed_H15_solver_sum_s": fixed15_solver,
            "fixed_H10_solver_sum_s": fixed10_solver,
            "fixed_H10_solver_relative_saving_vs_H15": None if fixed15_solver <= 0 else (fixed15_solver - fixed10_solver) / fixed15_solver,
            "label_data_sufficient_for_refit": label_data_sufficient,
        },
        "decision": decision,
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    ag = a["aggregate"]
    lines = [
        "# Vehicle true-variable-H v15 local boundary acquisition v0",
        "",
        f"UTC `{raw['created_utc']}`. Development-only 96-episode local boundary acquisition from opened H15-prefix states; no validation64, no sealed test, no selector refit/search, no gradient training.",
        "",
        "## Budget/access",
        "",
        f"- Episodes/control steps: `{raw['budget_actual']['development_mpc_simulation_episodes']}` / `{raw['budget_declared']['development_mpc_simulation_episodes']}` episodes; `{raw['budget_actual']['development_control_steps']}` / `{raw['budget_declared']['development_control_step_upper_bound']}` control steps.",
        f"- validation64_bank_opened: `{raw['validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Aggregate",
        "",
        f"- Pair rows `{ag['pair_count']}` / `{ag['expected_pair_count']}`; all pairs present `{ag['all_pairs_present']}`; all H15 safe `{ag['all_h15_safe']}`.",
        f"- Catastrophic H10 rows `{ag['catastrophic_h10_rows']}`; beneficial H10 rows `{ag['beneficial_h10_rows']}`; label-data-sufficient-for-refit `{ag['label_data_sufficient_for_refit']}`.",
        f"- Fixed H10-vs-H15 decision saving over acquired pairs `{100.0 * sf(ag['fixed_H10_decision_relative_saving_vs_H15']):.2f}%`; solver saving `{100.0 * sf(ag['fixed_H10_solver_relative_saving_vs_H15']):.2f}%`; physical sums H10/H15 `{ag['fixed_H10_physical_sum']:.6g}` / `{ag['fixed_H15_physical_sum']:.6g}`.",
        f"- Role counts: `{ag['role_counts']}`.",
        f"- Decision: {a['decision']}",
        "",
        "## Candidate-level label summary",
        "",
        "| idx | role | source | off | step | cat reps | ben reps | H15 safe | mean physΔ | mean decision gain s |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in a["candidate_rows"]:
        if r.get("missing"):
            lines.append(f"| {r.get('candidate_index')} | MISSING | `{r.get('candidate_id')}` | | | | | | | |")
            continue
        lines.append(
            "| %d | `%s` | `%s` | %d | %d | %d | %d | `%s` | %.6g | %.6g |" % (
                si(r.get("candidate_index")), r.get("role"), r.get("source_key"), si(r.get("offset_from_center")), si(r.get("branch_step")),
                si(r.get("catastrophic_h10_repeats")), si(r.get("beneficial_h10_repeats")), r.get("all_h15_safe"),
                sf(r.get("mean_phys_delta_h10_minus_h15")), sf(r.get("mean_decision_gain_h10_vs_h15_s"))
            )
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is opened-development boundary-label acquisition, not validation or final-test evidence. It deliberately enriches local ambiguous cases already implicated by v14; it cannot by itself establish generalization. If used for refit, the method must be versioned as IMPROVED and confirmed on unused development sources before any validation64 rollout.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`; completed: `{rel(RUN_DIR / 'completed.json')}`; backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    ag = raw["analysis"]["aggregate"]
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v15 local boundary acquisition v0

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only local boundary acquisition on 24 saved H15-prefix candidates x H10/H15 x2 repeats; no validation64/sealed-test access, no selector refit/search, no gradient training. Budget actual: {raw['budget_actual']['development_mpc_simulation_episodes']} episodes, {raw['budget_actual']['development_control_steps']} control steps. Aggregate: pairs={ag['pair_count']}/{ag['expected_pair_count']}, all_h15_safe={ag['all_h15_safe']}, catastrophic_h10={ag['catastrophic_h10_rows']}, beneficial_h10={ag['beneficial_h10_rows']}, decision_saving_H10_vs_H15={ag['fixed_H10_decision_relative_saving_vs_H15']}, label_data_sufficient_for_refit={ag['label_data_sufficient_for_refit']}. Decision: {raw['analysis']['decision']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    tail = reg.read_text(encoding="utf-8", errors="replace")[-120000:] if reg.exists() else ""
    if MARKER not in tail:
        with reg.open("a", encoding="utf-8") as f:
            f.write(f"{STAMP},{NAME},development_v15_local_boundary_acquisition,opened_dev_false_positive_boundary_candidates,{raw['budget_actual']['development_mpc_simulation_episodes']},0,0,0,0,False,{rel(RUN_DIR / 'completed.json')},{MARKER}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-v15-boundary-acquisition", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_v15_boundary_acquisition:
        raise ContractError("requires --run and explicit development v15 boundary-acquisition acknowledgement")
    if (RUN_DIR / "completed.json").exists():
        done = read_json(RUN_DIR / "completed.json")
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name != "run.lock" for p in RUN_DIR.iterdir()):
        raise ContractError("partial run output exists; inspect before rerun: " + rel(RUN_DIR))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    inputs = verify_inputs()
    prepared = prepare_candidates(inputs["candidates"], inputs["raw_by_bank"])
    input_paths = [SOURCE, AUDIT, AUDIT_DONE, PREFLIGHT, PREFLIGHT_DONE, SMOKE_DONE, SMOKE_RAW] + list(RAW_BY_BANK_PATHS.values()) + list(DONE_BY_BANK_PATHS.values())
    protocol = build_protocol(now(), prepared, args.backup_verified_commit, {rel(p): sha256(p) for p in input_paths if p.exists()})
    case_runner.SMOKE_DIR = RUN_DIR
    _, stage1_runner, _ = v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
    stage1_runner.base.v1.latency_verify()
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    if 15 not in terminals:
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
        summary = case_runner.run_true_h_episode(run_item, case, terminals[15])
        summary["stage"] = "v15_local_boundary_acquisition_trueH10_H15"
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
        summary["source_catastrophic"] = cand.get("source_catastrophic")
        summary["source_label_positive"] = cand.get("source_label_positive")
        summary["source_phys_delta"] = cand.get("source_phys_delta")
        summary["source_decision_gain_s"] = cand.get("source_decision_gain_s")
        summary["terminal_source_horizon"] = 15
        summary["terminal_receipt_effective"] = terminal_receipts.get("15") or terminal_receipts.get(15)
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
    write_json(BACKUP_REQUEST, {"requested_utc": created.isoformat(), "reason": "backup v15 local boundary acquisition source/protocol/raw/docs before any boundary-augmented selector refit", "backup_required_before_more_science": True, "development_mpc_simulation_episodes": len(episodes), "development_control_steps": control_steps, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(SOURCE), rel(PROTOCOL), rel(RUN_DIR), rel(STATE), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"]})
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_true_variable_H_local_boundary_acquisition_not_validation_not_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "runtime_preflight": preflight,
        "terminal_sources": {str(k): v for k, v in terminal_receipts.items()},
        "candidates_preoutcome": [{k: v for k, v in c.items() if k != "case_snapshot_from_candidate_pool"} for c in prepared],
        "episodes": episodes,
        "analysis": analysis,
        "budget_declared": {"development_mpc_simulation_episodes": EXPECTED_EPISODES, "development_control_step_upper_bound": CONTROL_STEP_CAP, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"development_mpc_simulation_episodes": len(episodes), "development_control_steps": control_steps, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "platform": {"python": sys.version, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "backup_request": rel(BACKUP_REQUEST),
        "interpretation_limits": ["opened development local boundary candidates", "two repeats per horizon", "not validation/test", "no selector refit", "timing is logged but not sufficient for final speed claim"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    append_docs(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"""# Continue state after v15 local boundary acquisition v0

UTC: {created.isoformat()}
Elapsed since first supervisor event: {(created - FIRST_EVENT).total_seconds()/3600.0:.2f} h.

Completed `{NAME}` development-only acquisition: {len(episodes)} episodes, {control_steps} control steps. No validation64, no sealed test, no selector refit/search, no training.

Aggregate: {analysis['aggregate']}
Decision: {analysis['decision']}
Summary: `{rel(RUN_DIR / 'summary.md')}`
Raw: `{rel(RUN_DIR / 'raw.json')}`
Completed: `{rel(RUN_DIR / 'completed.json')}`
Backup request: `{rel(BACKUP_REQUEST)}`

Next action: externally back up source/protocol/raw/docs. If label_data_sufficient_for_refit is true, freeze a boundary-augmented conservative refit; otherwise inspect role/source failures and pivot to richer risk/terminal-value learning or observability diagnostics rather than another unchanged static-feature sweep.
""", encoding="utf-8")
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, BACKUP_REQUEST, AUDIT, PREFLIGHT, SMOKE_RAW] + list(RAW_BY_BANK_PATHS.values())
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
