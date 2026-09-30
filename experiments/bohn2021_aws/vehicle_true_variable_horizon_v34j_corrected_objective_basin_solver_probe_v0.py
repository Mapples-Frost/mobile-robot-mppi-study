#!/usr/bin/env python3
"""v34j corrected objective-vs-basin solver probe (remaining 23 solves).

Development-only continuation after:
  * v34g spent one solver call on the fixed Task-C cell
    source242_slot0_branch_start | H15 | V15_shared | canonical;
  * v34i showed partial do-mpc struct indexing is unusable for objective
    reconstruction;
  * v34k hard-passed the zero-solve full-label accessor gate and identified an
    objective reconstruction formula with relative error <= 1e-6, while the
    repaired residual aliases lb_opt_x/ub_opt_x and cons_lb/cons_ub passed.

This wrapper keeps the frozen Task-C design set (2 contexts x 2 horizons x 3
terminal contracts x 2 deterministic initializations = 24 arms) but imports the
already-spent v34g arm and executes only the remaining 23 lower-level solver
calls.  It performs zero plant steps, no env.reset/env.step after construction,
no selector search/refit, no training, no validation64 access, and no
sealed/final-test access.

The only operational repairs relative to v34f are:
  * objective reconstruction uses the v34k full-label accessor and the frozen
    passing formula: stage=x0_then_last_node, terminal=author_last_node,
    eps=False, r=False, discount=n_horizon_parameter_or_H, z=same_k_last;
  * residual/bounds capture aliases do-mpc's real lb_opt_x/ub_opt_x and
    cons_lb/cons_ub names, records an explicit name-existence table, and writes
    distinct ok/exceeded/missing_or_size_mismatch status strings;
  * backup/lead gates require the active Opus 20260930T103334Z plan and a
    verified backup after the v34k hard-pass/source preparation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import vehicle_true_variable_horizon_v34f_objective_basin_solver_probe_v0 as f  # noqa:E402
import vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0 as kacc  # noqa:E402

base = f.base
ROOT = f.ROOT
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
NAME = "vehicle_true_variable_horizon_v34j_corrected_objective_basin_solver_probe_v0"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34j_corrected_objective_basin_solver_probe.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34J_CORRECTED_OBJECTIVE_BASIN_SOLVER_PROBE_{STAMP}.json"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
OPUS_PLAN_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
OPUS_LATEST = ROOT / "docs/bohn2021_takeover/opus_lead/LATEST.md"
CURRENT_OPUS_REQUEST = "execution-result:20260930T103304_a31a51bd"
CURRENT_OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T103334Z_44f4af.md"
CURRENT_OPUS_SHA = "7fb49cf299921224dd64e6b045f971141f6a7b9db740a6f73e71343d9a9e07fc"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
REQUEST_ID = f"v34j-corrected-objective-basin-solver-probe-{STAMP}"
MARKER = f"vehicle-v34j-corrected-objective-basin-solver-probe-{STAMP}"

V34G_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0_20260930T100824Z/completed.json"
V34G_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0_20260930T100824Z/raw.json"
V34K_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_20260930T104148Z/completed.json"
V34K_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_20260930T104148Z/raw.json"

TARGET_IMPORTED_KEY = {
    "context_id": "source242_slot0_branch_start",
    "horizon": 15,
    "terminal_mode": "V15_shared",
    "initialization": "canonical",
}
IMPORTED_V34G_SOLVER_CALLS = 1
NEW_SOLVE_CAP = 23
TASK_C_TOTAL_SOLVE_BUDGET = 24
FORMULA_SPEC = {
    "stage_rule": "x0_then_last_node",
    "terminal_rule": "author_last_node",
    "include_eps": False,
    "include_rterm": False,
    "discount_rule": "n_horizon_parameter_or_H",
    "z_rule": "same_k_last",
    "candidate_id": "stage=x0_then_last_node|term=author_last_node|eps=False|r=False|discount=n_horizon_parameter_or_H|z=same_k_last",
}

_ORIGINAL_CREATE_ENV = None
_ORIGINAL_CAPTURE_MPC_STATE = None
_ORIGINAL_SOLVE_ARM = None


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    return base.rel(path)


def write_json(path: Path, value: Any) -> None:
    base.write_json(path, value)


def read_json(path: Path) -> Any:
    return base.read_json(path)


def sha256(path: Path) -> str:
    return base.sha256(path)


def clean(value: Any) -> Any:
    return base.clean(value)


def arr(value: Any) -> np.ndarray:
    return base.arr(value)


def arr_hash(value: Any) -> str:
    return base.arr_hash(value)


def finite_float(value: Any) -> Optional[float]:
    return base.finite_float(value, None)


def append_if_missing(path: Path, marker: str, block: str) -> None:
    base.append_if_missing(path, marker, block)


def hash_existing(paths: Iterable[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        try:
            if p.exists() and p.is_file():
                out[rel(p)] = sha256(p)
        except Exception:
            pass
    return out


def row_key(row: Mapping[str, Any]) -> Tuple[str, int, str, str]:
    return (str(row.get("context_id")), int(row.get("horizon")), str(row.get("terminal_mode")), str(row.get("initialization")))


def target_key_tuple() -> Tuple[str, int, str, str]:
    return (
        TARGET_IMPORTED_KEY["context_id"],
        int(TARGET_IMPORTED_KEY["horizon"]),
        TARGET_IMPORTED_KEY["terminal_mode"],
        TARGET_IMPORTED_KEY["initialization"],
    )


def parse_time(value: Any) -> Optional[dt.datetime]:
    return f._parse_time(value)


def status_for_residual(value: Any, tol: float = base.ACCEPT_RESIDUAL_TOL) -> str:
    val = finite_float(value)
    if val is None:
        return "missing_or_size_mismatch"
    if float(val) <= float(tol):
        return "ok"
    return "exceeded"


def bound_name_table(mpc: Any) -> Dict[str, Any]:
    table: Dict[str, Any] = {}
    for name in ("lb_opt_x", "ub_opt_x", "cons_lb", "cons_ub", "opt_x_lb", "opt_x_ub", "opt_g_lb", "opt_g_ub"):
        exists = hasattr(mpc, name)
        row: Dict[str, Any] = {"exists": bool(exists)}
        if exists:
            try:
                value = getattr(mpc, name)
                a = arr(value)
                row.update({"flat_size": int(a.size), "flat_hash": arr_hash(value), "finite": bool(np.all(np.isfinite(a))) if a.size else True})
            except Exception as exc:
                row.update({"error": repr(exc)})
        table[name] = row
    return table


def install_bound_aliases(mpc: Any) -> Dict[str, Any]:
    """Expose do-mpc real bound names under the legacy names used by v34 code."""
    aliases = {
        "opt_x_lb": "lb_opt_x",
        "opt_x_ub": "ub_opt_x",
        "opt_g_lb": "cons_lb",
        "opt_g_ub": "cons_ub",
    }
    actions: Dict[str, Any] = {}
    for legacy, repaired in aliases.items():
        repaired_exists = hasattr(mpc, repaired)
        legacy_preexists = hasattr(mpc, legacy)
        if repaired_exists:
            try:
                setattr(mpc, legacy, getattr(mpc, repaired))
                actions[legacy] = {"aliased_from": repaired, "repaired_exists": True, "legacy_preexisted": bool(legacy_preexists)}
            except Exception as exc:
                actions[legacy] = {"aliased_from": repaired, "repaired_exists": True, "legacy_preexisted": bool(legacy_preexists), "error": repr(exc)}
        else:
            actions[legacy] = {"aliased_from": repaired, "repaired_exists": False, "legacy_preexisted": bool(legacy_preexists)}
    return {"alias_actions": actions, "bounds_name_existence_table": bound_name_table(mpc)}


def capture_mpc_state_v34j(mpc: Any) -> Dict[str, Any]:
    install_meta = install_bound_aliases(mpc)
    if _ORIGINAL_CAPTURE_MPC_STATE is None:
        out: Dict[str, Any] = {}
    else:
        out = _ORIGINAL_CAPTURE_MPC_STATE(mpc)
    out["v34j_repaired_bound_alias_capture"] = install_meta
    return out


def create_env_v34j(h: int, terminal: Tuple[Any, Any]) -> Any:
    if _ORIGINAL_CREATE_ENV is None:
        raise ContractError("original create_env not installed")
    env = _ORIGINAL_CREATE_ENV(int(h), terminal)
    try:
        install_bound_aliases(env.control_system.controller.mpc)
    except Exception:
        pass
    return env


def label_reconstruct_stage_and_terminal_v34j(mpc: Any, h: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": False,
        "formula_source": "v34k_full_label_accessor",
        "frozen_formula_spec": dict(FORMULA_SPEC),
        "stage_lterm_total": None,
        "terminal_value_current_vf": None,
        "gamma": None,
        "gamma_source": None,
        "gamma_pow_H_terminal": None,
        "solver_objective": finite_float(getattr(mpc, "opt_f_num", None)),
        "reconstructed_total_current_objective": None,
        "relative_error_vs_solver": None,
        "terminal_x_vector": None,
        "p_excluding_n_horizon_values": None,
        "p_names_excluding_n_horizon": None,
        "failure_reasons": [],
        "component_limits": {
            "slack_separate_component_available": True,
            "input_regularization_separate_component_available": True,
            "stage_lterm_may_include_slack_or_other_terms": False,
            "v34k_evidence_eps_total_zero_in_gate_cell": True,
            "v34k_evidence_rterm_total_zero_in_gate_cell": True,
        },
    }
    try:
        install_bound_aliases(mpc)
        acc = kacc.LabelAccessor(mpc)
        comp = kacc.formula_components(
            mpc,
            acc,
            int(h),
            FORMULA_SPEC["stage_rule"],
            FORMULA_SPEC["terminal_rule"],
            bool(FORMULA_SPEC["include_eps"]),
            bool(FORMULA_SPEC["include_rterm"]),
            FORMULA_SPEC["discount_rule"],
            FORMULA_SPEC["z_rule"],
        )
        p_keep, p_names = acc.p(0, True)
        xh = acc.state(int(h), 0, -1)
        out.update({
            "available": True,
            "candidate_id": FORMULA_SPEC["candidate_id"],
            "stage_lterm_total": comp.get("stage_lterm_total"),
            "terminal_value_current_vf": comp.get("terminal_vf_raw"),
            "gamma": comp.get("gamma"),
            "gamma_source": comp.get("gamma_source"),
            "gamma_pow_H_terminal": comp.get("terminal_discounted"),
            "solver_objective": comp.get("solver_objective"),
            "reconstructed_total_current_objective": comp.get("total"),
            "relative_error_vs_solver": comp.get("relative_error"),
            "absolute_error_vs_solver": comp.get("absolute_error"),
            "terminal_x_vector": arr(xh).tolist(),
            "p_excluding_n_horizon_values": arr(p_keep).tolist(),
            "p_names_excluding_n_horizon": list(p_names),
            "label_accessor_summary": {
                "state_vars": acc.state_vars,
                "input_vars": acc.input_vars,
                "p_vars": acc.p_vars,
                "tvp_vars_count": len(acc.tvp_vars),
                "eps_vars": acc.eps_vars,
                "label_counts": {
                    "scaled_opt_x": len(acc.scaled_records),
                    "unscaled_opt_x": len(acc.unscaled_records),
                    "opt_p": len(acc.p_records),
                },
            },
            "formula_component_echo": {kk: comp.get(kk) for kk in [
                "stage_x_rule", "terminal_x_rule", "z_rule", "include_eps", "include_rterm", "discount_rule",
                "stage_lterm_total", "epsterm_total", "input_regularization_total", "terminal_discounted",
                "eps_executed", "rterm_executed", "eps_nonzero_count", "rterm_nonzero_count",
            ]},
        })
    except Exception as exc:
        out["failure_reasons"].append("v34j_label_accessor_reconstruction_failed: " + repr(exc))
        out["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
    return out


def solve_arm_v34j(row: Mapping[str, Any], context: Mapping[str, Any], terminals: Mapping[int, Tuple[Any, Any]], coeffs: Mapping[str, Any]) -> Dict[str, Any]:
    if _ORIGINAL_SOLVE_ARM is None:
        raise ContractError("original solve_arm not installed")
    arm = _ORIGINAL_SOLVE_ARM(row, context, terminals, coeffs)
    ev = arm.setdefault("solver_event", {})
    ev["bound_residual_status"] = status_for_residual(ev.get("bound_residual"))
    ev["constraint_residual_status"] = status_for_residual(ev.get("constraint_residual"))
    ev["residual_status_contract"] = {
        "ok": "numeric residual <= tolerance",
        "exceeded": "numeric residual > tolerance",
        "missing_or_size_mismatch": "residual unavailable because names/arrays were missing or incompatible",
        "tolerance": base.ACCEPT_RESIDUAL_TOL,
        "repaired_bound_names": ["lb_opt_x", "ub_opt_x"],
        "repaired_constraint_names": ["cons_lb", "cons_ub"],
    }
    arm["accepted"] = bool(ev.get("success")) and ev["bound_residual_status"] == "ok" and ev["constraint_residual_status"] == "ok" and ev.get("objective_opt_f_num") is not None
    arm["v34j_corrected_accessors"] = {
        "objective_formula": dict(FORMULA_SPEC),
        "residual_aliases": "lb_opt_x/ub_opt_x and cons_lb/cons_ub exposed to legacy opt_x/opt_g names",
    }
    return arm


def patch_runtime() -> None:
    global _ORIGINAL_CREATE_ENV, _ORIGINAL_CAPTURE_MPC_STATE, _ORIGINAL_SOLVE_ARM
    f.patch_runtime()
    _ORIGINAL_CREATE_ENV = base.create_env
    _ORIGINAL_CAPTURE_MPC_STATE = base.capture_mpc_state
    _ORIGINAL_SOLVE_ARM = base.solve_arm
    base.NAME = NAME
    base.STAMP = STAMP
    base.RUN_DIR = RUN_DIR
    base.STATE = STATE
    base.BACKUP_REQUEST = BACKUP_REQUEST
    base.REQUEST_ID = REQUEST_ID
    base.MARKER = MARKER
    base.ANALYSIS_READY = OPUS_PLAN_READY
    base.ASTRA_REPORT = CURRENT_OPUS_REPORT
    base.CURRENT_ASTRA_REQUEST = CURRENT_OPUS_REQUEST
    base.CURRENT_ASTRA_SHA = CURRENT_OPUS_SHA
    base.create_env = create_env_v34j
    base.capture_mpc_state = capture_mpc_state_v34j
    base.reconstruct_stage_and_terminal = label_reconstruct_stage_and_terminal_v34j
    base.solve_arm = solve_arm_v34j
    base.__file__ = str(Path(__file__).resolve())


def verify_opus_plan_gate() -> Dict[str, Any]:
    if not OPUS_PLAN_READY.exists():
        raise ContractError("missing active Opus PLAN_READY.json")
    ready = read_json(OPUS_PLAN_READY)
    if ready.get("request_id") != CURRENT_OPUS_REQUEST:
        raise ContractError("active Opus request mismatch: %r != %r" % (ready.get("request_id"), CURRENT_OPUS_REQUEST))
    report_path = ROOT / str(ready.get("report", ""))
    if report_path.resolve() != CURRENT_OPUS_REPORT.resolve():
        raise ContractError("active Opus report path mismatch: %s" % ready.get("report"))
    if ready.get("report_sha256") != CURRENT_OPUS_SHA:
        raise ContractError("active Opus PLAN_READY sha mismatch")
    if not CURRENT_OPUS_REPORT.exists() or sha256(CURRENT_OPUS_REPORT) != CURRENT_OPUS_SHA:
        raise ContractError("active Opus report missing or sha mismatch")
    latest_text = OPUS_LATEST.read_text(encoding="utf-8", errors="replace") if OPUS_LATEST.exists() else ""
    if rel(CURRENT_OPUS_REPORT) not in latest_text:
        raise ContractError("Opus LATEST.md does not point to active report")
    return {
        "plan_ready": rel(OPUS_PLAN_READY),
        "latest": rel(OPUS_LATEST),
        "request_id": ready.get("request_id"),
        "experiment_id": ready.get("experiment_id"),
        "primary_analyst": ready.get("primary_analyst"),
        "report": rel(CURRENT_OPUS_REPORT),
        "report_sha256": CURRENT_OPUS_SHA,
        "completed": ready.get("completed"),
    }


def verify_v34k_gate() -> Dict[str, Any]:
    for path, label in [(V34G_DONE, "v34g completed"), (V34G_RAW, "v34g raw"), (V34K_DONE, "v34k completed"), (V34K_RAW, "v34k raw")]:
        if not path.exists():
            raise ContractError(f"missing prerequisite {label}: {rel(path)}")
    g_done = read_json(V34G_DONE)
    k_done = read_json(V34K_DONE)
    k_raw = read_json(V34K_RAW)
    if int((g_done.get("budget_actual") or {}).get("solver_calls", 0)) != 1:
        raise ContractError("v34g predecessor did not record exactly one solver call")
    if k_done.get("hard_pass") is not True or (k_raw.get("gates") or {}).get("both_pass") is not True:
        raise ContractError("v34k label-accessor/localization gate did not hard-pass")
    b = k_done.get("budget_actual") or {}
    forbidden = ["solver_calls", "plant_steps", "env_reset_calls_after_construction", "env_step_calls_after_construction", "new_training_or_gradient_steps", "selector_refits", "validation64_episodes", "sealed_test_episodes"]
    if any(int(b.get(name, 0)) != 0 for name in forbidden):
        raise ContractError("v34k was not a zero-solve/zero-plant/no-validation/no-test gate")
    best = (k_raw.get("objective_localization") or {}).get("best_candidate") or {}
    if best.get("candidate_id") != FORMULA_SPEC["candidate_id"]:
        raise ContractError("v34k best formula no longer matches frozen v34j formula")
    if best.get("relative_error") is None or float(best["relative_error"]) > base.RECON_REL_TOL:
        raise ContractError("v34k best formula does not satisfy reconstruction tolerance")
    return {
        "v34g_completed": rel(V34G_DONE),
        "v34g_raw": rel(V34G_RAW),
        "v34g_completed_sha256": sha256(V34G_DONE),
        "v34g_raw_sha256": sha256(V34G_RAW),
        "v34k_completed": rel(V34K_DONE),
        "v34k_raw": rel(V34K_RAW),
        "v34k_completed_sha256": sha256(V34K_DONE),
        "v34k_raw_sha256": sha256(V34K_RAW),
        "v34k_created_utc": k_done.get("created_utc"),
        "v34k_headline": k_done.get("headline"),
        "frozen_formula_spec": dict(FORMULA_SPEC),
        "v34k_budget_actual": b,
    }


def scan_backup_gate(args: argparse.Namespace, min_time: dt.datetime) -> Dict[str, Any]:
    if not (args.backup_time and args.backup_commit and args.backup_package_sha256):
        raise ContractError("verified backup context args are required before v34j remaining solver evidence")
    current_backup_time = parse_time(args.backup_time)
    if current_backup_time is None:
        raise ContractError("backup_time not parseable")
    if current_backup_time < min_time:
        raise ContractError("backup_time predates required v34j/v34k source/evidence min_time")
    return f.scan_verified_backup_proofs(args, min_time)


def verify_gates(args: argparse.Namespace) -> Dict[str, Any]:
    lead = verify_opus_plan_gate()
    prereq = verify_v34k_gate()
    task = f.verify_task_prerequisites()
    k_time = parse_time(prereq.get("v34k_created_utc")) or now_utc()
    # The v34j source is prepared after v34k.  A remaining_changed_files==0 proof
    # matching current CLI args and at/after this min_time is required; the proof
    # itself is created by the supervisor backup cycle, not by this script.
    proof = scan_backup_gate(args, k_time)
    return {
        "active_scientific_lead_gate": lead,
        "task_prerequisites": task,
        "v34g_v34k_gate": prereq,
        "current_backup_arg": {
            "time": args.backup_time,
            "commit": args.backup_commit,
            "package_sha256": args.backup_package_sha256,
            "package_bytes": int(getattr(args, "backup_package_bytes", 0) or 0),
        },
        "matched_verified_backup_proof": proof,
        "dependency_authorization": {
            "source": rel(CURRENT_OPUS_REPORT),
            "stable_id": "A13_objective_reconstruction_contract_localization -> Task C objective-vs-basin probe",
            "gate": "v34k label_accessor_pass AND G_E_objective_reconstruction_pass AND G_F_residual_alias_offline_pass",
            "gate_passed": True,
            "new_solver_call_cap": NEW_SOLVE_CAP,
            "task_c_total_solver_budget_including_v34g": TASK_C_TOTAL_SOLVE_BUDGET,
        },
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }


def corrected_reconstruction_from_v34k(k_raw: Mapping[str, Any]) -> Dict[str, Any]:
    best = (k_raw.get("objective_localization") or {}).get("best_candidate") or {}
    p0 = (((k_raw.get("label_accessor_diagnostics") or {}).get("indexability_without_partial_struct") or {}).get("p_0") or {}).get("first_values") or []
    p_names_all = (k_raw.get("label_accessor_diagnostics") or {}).get("p_vars") or ["goal_x", "goal_y", "n_horizon"]
    p_keep: List[float] = []
    p_keep_names: List[str] = []
    for name, value in zip(p_names_all, p0):
        if "n_horizon" in str(name):
            continue
        p_keep_names.append(str(name))
        p_keep.append(float(value))
    return {
        "available": True,
        "formula_source": "v34k_full_label_accessor_imported_for_v34g_arm",
        "candidate_id": FORMULA_SPEC["candidate_id"],
        "frozen_formula_spec": dict(FORMULA_SPEC),
        "stage_lterm_total": best.get("stage_lterm_total"),
        "terminal_value_current_vf": best.get("terminal_vf_raw"),
        "gamma": best.get("gamma"),
        "gamma_source": best.get("gamma_source"),
        "gamma_pow_H_terminal": best.get("terminal_discounted"),
        "solver_objective": best.get("solver_objective"),
        "reconstructed_total_current_objective": best.get("total"),
        "relative_error_vs_solver": best.get("relative_error"),
        "absolute_error_vs_solver": best.get("absolute_error"),
        "terminal_x_vector": ((best.get("terminal_meta") or {}).get("x_vector")),
        "p_excluding_n_horizon_values": p_keep,
        "p_names_excluding_n_horizon": p_keep_names,
        "failure_reasons": [],
        "imported_v34k_raw": rel(V34K_RAW),
        "component_limits": {
            "slack_separate_component_available": True,
            "input_regularization_separate_component_available": True,
            "stage_lterm_may_include_slack_or_other_terms": False,
        },
    }


def imported_v34g_arm_for_row(row: Mapping[str, Any], k_raw: Mapping[str, Any]) -> Dict[str, Any]:
    g_raw = read_json(V34G_RAW)
    arm = copy.deepcopy(g_raw.get("arm") or {})
    if not arm:
        raise ContractError("v34g raw did not contain an arm object")
    original_key = row_key(arm)
    if original_key != target_key_tuple():
        raise ContractError(f"v34g imported arm key mismatch: {original_key} != {target_key_tuple()}")
    arm["execution_index"] = int(row.get("execution_index", arm.get("execution_index", 0)))
    arm["scheduled_row_v34j"] = clean(dict(row))
    arm["imported_from_v34g_one_cell_smoke"] = {
        "completed": rel(V34G_DONE),
        "raw": rel(V34G_RAW),
        "solver_calls_already_spent_in_prior_run": IMPORTED_V34G_SOLVER_CALLS,
        "solver_calls_charged_to_this_v34j_run": 0,
        "v34k_zero_solve_reconstruction": rel(V34K_RAW),
    }
    arm["solve_calls"] = 0
    arm["objective_reconstruction_current"] = corrected_reconstruction_from_v34k(k_raw)
    repaired = ((k_raw.get("residual_localization") or {}).get("repaired_names") or {})
    ev = arm.setdefault("solver_event", {})
    if repaired.get("bound_residual_using_lb_opt_x_names") is not None:
        ev["bound_residual"] = repaired.get("bound_residual_using_lb_opt_x_names")
    if repaired.get("constraint_residual_using_cons_lb_names") is not None:
        ev["constraint_residual"] = repaired.get("constraint_residual_using_cons_lb_names")
    ev["bound_residual_status"] = status_for_residual(ev.get("bound_residual"))
    ev["constraint_residual_status"] = status_for_residual(ev.get("constraint_residual"))
    ev["residual_status_contract"] = {
        "ok": "numeric residual <= tolerance",
        "exceeded": "numeric residual > tolerance",
        "missing_or_size_mismatch": "residual unavailable because names/arrays were missing or incompatible",
        "tolerance": base.ACCEPT_RESIDUAL_TOL,
        "repaired_bound_names": ["lb_opt_x", "ub_opt_x"],
        "repaired_constraint_names": ["cons_lb", "cons_ub"],
        "source": rel(V34K_RAW),
    }
    arm["accepted"] = bool(ev.get("success")) and ev["bound_residual_status"] == "ok" and ev["constraint_residual_status"] == "ok" and ev.get("objective_opt_f_num") is not None
    arm["v34j_corrected_accessors"] = {
        "objective_formula": dict(FORMULA_SPEC),
        "residual_aliases": "v34k repaired residuals for imported v34g arm; new arms use live aliases",
    }
    return arm


def write_progress(arms: Sequence[Mapping[str, Any]], new_solves: int, last_arm: Mapping[str, Any]) -> None:
    write_json(RUN_DIR / "progress.json", {
        "arms_done": len(arms),
        "arms_expected": TASK_C_TOTAL_SOLVE_BUDGET,
        "new_solver_calls_this_run": int(new_solves),
        "imported_solver_calls_from_v34g": IMPORTED_V34G_SOLVER_CALLS,
        "task_c_solver_calls_including_imported_v34g": int(new_solves + IMPORTED_V34G_SOLVER_CALLS),
        "plant_steps": 0,
        "env_reset_calls": 0,
        "last_arm": last_arm,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "training_or_refit": 0,
    })


def augment_analysis(analysis: Mapping[str, Any], arms: Sequence[Mapping[str, Any]], new_solves: int) -> Dict[str, Any]:
    out = copy.deepcopy(dict(analysis))
    h = dict(out.get("headline") or {})
    h["new_solver_calls_this_run"] = int(new_solves)
    h["imported_solver_calls_from_v34g"] = IMPORTED_V34G_SOLVER_CALLS
    h["task_c_solver_calls_including_imported_v34g"] = int(new_solves + IMPORTED_V34G_SOLVER_CALLS)
    h["task_c_arms_including_imported_v34g"] = len(arms)
    h["residual_bound_status_counts"] = {}
    h["residual_constraint_status_counts"] = {}
    for arm in arms:
        ev = arm.get("solver_event") or {}
        bs = str(ev.get("bound_residual_status", "missing_or_size_mismatch"))
        cs = str(ev.get("constraint_residual_status", "missing_or_size_mismatch"))
        h["residual_bound_status_counts"][bs] = h["residual_bound_status_counts"].get(bs, 0) + 1
        h["residual_constraint_status_counts"][cs] = h["residual_constraint_status_counts"].get(cs, 0) + 1
    out["headline"] = h
    out["v34j_interpretation_limits"] = [
        "development-only opened states",
        "one arm imported from v34g; 23 new solver calls in this run",
        "zero plant rollouts and no validation64/sealed-test access",
        "objective reconstruction uses v34k full-label accessor formula frozen before these 23 solves",
    ]
    return out


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["analysis"]["headline"]
    lines = [
        "# v34j corrected objective-vs-basin solver probe",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only Task-C continuation using the v34k full-label objective formula and repaired residual aliases.",
        "",
        "## Budget/accounting",
        "",
        f"- New solver calls in this run: `{h['new_solver_calls_this_run']}` / `{NEW_SOLVE_CAP}`.",
        f"- Imported solver calls from v34g: `{h['imported_solver_calls_from_v34g']}`.",
        f"- Task-C total solver calls including imported v34g: `{h['task_c_solver_calls_including_imported_v34g']}` / `{TASK_C_TOTAL_SOLVE_BUDGET}`.",
        f"- Arms analyzed: `{h['task_c_arms_including_imported_v34g']}`; accepted arms: `{h['accepted_arms']}`; solver statuses: `{h['solver_status_counts']}`.",
        f"- Plant steps: `{h['plant_steps']}`; env.reset calls: `{h['env_reset_calls']}`; training/refit: `0`.",
        f"- validation64 opened: `{h['validation64_bank_opened']}`; sealed test accessed: `{h['sealed_test_accessed']}`.",
        "",
        "## Corrected reconstruction / residual gates",
        "",
        f"- Formula: `{FORMULA_SPEC['candidate_id']}`.",
        f"- Objective reconstruction available arms: `{h['objective_reconstruction_available_arms']}`; max relative error: `{h['objective_reconstruction_max_rel_error']}`; unavailable arms: `{h['objective_reconstruction_unavailable_arms']}`.",
        f"- Bound residual statuses: `{h.get('residual_bound_status_counts')}`.",
        f"- Constraint residual statuses: `{h.get('residual_constraint_status_counts')}`.",
        "",
        "## Mechanism numeric readout (development-only)",
        "",
        f"- Same-objective basin-support pairs (goal-facing J lower than canonical by > {base.BASIN_REL_TOL} relative): `{h['basin_support_pairs']}`.",
        f"- Context/H groups whose best candidate terminal differs across common objectives: `{h['objective_ranking_reversal_groups']}`.",
        f"- Missing cross-scoring cells: `{h['cross_scoring_missing_cells_count']}`.",
        "",
        "| context | H | accepted | basin pairs | best under zero/V15/V35 objectives |",
        "|---|---:|---:|---|---|",
    ]
    for g in raw["analysis"]["groups"]:
        best = {m: None if not isinstance(g["objective_best_candidates"].get(m), Mapping) else {"terminal": g["objective_best_candidates"][m]["candidate_terminal_mode"], "init": g["objective_best_candidates"][m]["candidate_initialization"], "J": round(float(g["objective_best_candidates"][m]["J"]), 6)} for m in base.TERMINAL_MODES}
        pairs = [{"term": p["terminal_mode"], "delta": p["delta_canonical_minus_goal_facing_same_J"], "sig": p["basin_significant_same_objective"]} for p in g["basin_pairs"]]
        lines.append(f"| `{g['context_id']}` | {g['horizon']} | {g['accepted_count']} | `{pairs}` | `{best}` |")
    lines += [
        "",
        "## Limits",
        "",
        "This is a targeted opened-development-state optimization diagnostic. It is not population validation, not validation64, not final test, and not an adaptive closed-loop selector/timing claim.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_docs(raw: Mapping[str, Any]) -> None:
    h = raw["analysis"]["headline"]
    block = f"""
<!-- {MARKER} -->
## v34j corrected objective-vs-basin solver probe

UTC: {raw['created_utc']}. Executed the Opus-authorized Task-C continuation after v34k hard-pass. New solver calls this run={h['new_solver_calls_this_run']}; imported v34g solver calls={h['imported_solver_calls_from_v34g']}; Task-C total solver calls={h['task_c_solver_calls_including_imported_v34g']}/{TASK_C_TOTAL_SOLVE_BUDGET}; arms analyzed={h['task_c_arms_including_imported_v34g']}. Plant steps=0, env_reset calls=0, training/refit=0, validation64=false, sealed_test=false. Objective reconstruction available arms={h['objective_reconstruction_available_arms']} max_rel_error={h['objective_reconstruction_max_rel_error']}; residual status bound={h.get('residual_bound_status_counts')}, constraint={h.get('residual_constraint_status_counts')}; accepted arms={h['accepted_arms']}; basin-support pairs={h['basin_support_pairs']}; objective-ranking-reversal groups={h['objective_ranking_reversal_groups']}. Evidence: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`. Next lead-review request `{REQUEST_ID}`.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as ff:
        csv.writer(ff).writerow([
            raw["created_utc"], NAME, raw["classification"], "RNG_SEED=%s; imported_v34g=%s" % (base.RNG_SEED, rel(V34G_RAW)),
            "opened_development_fixed_context_corrected_objective_basin_solver_probe_no_validation_no_test",
            h["task_c_arms_including_imported_v34g"], 0, h["new_solver_calls_this_run"], 0, 0, False,
            rel(RUN_DIR / "completed.json"), MARKER,
        ])
    write_json(NEXT_REVIEW_REQUEST, {
        "request_id": REQUEST_ID,
        "created": raw["created_utc"],
        "status": "analysis_requested",
        "trigger": "v34j corrected objective-vs-basin fixed-context solver probe completed",
        "experiment_id": NAME,
        "active_lead_report": rel(CURRENT_OPUS_REPORT),
        "active_lead_report_sha256": CURRENT_OPUS_SHA,
        "question": "Analyze the corrected Task-C objective-vs-basin evidence. Decide whether terminal objective preference, warm-start/basin effects, or another mechanism should drive the next bounded research action. Preserve development-only/no validation64/no sealed-test limits.",
        "evidence_paths": [rel(RUN_DIR / "summary.md"), rel(RUN_DIR / "raw.json"), rel(RUN_DIR / "completed.json"), rel(CURRENT_OPUS_REPORT), rel(RESPONSE_LOG), rel(V34K_RAW), rel(V34G_RAW)],
        "budget_actual": raw["budget_actual"],
        "headline": h,
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "plant_steps": 0, "training_or_refit": 0},
        "backup_required_before_more_unique_science": rel(BACKUP_REQUEST),
    })
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        f"# Continue state after v34j corrected objective-vs-basin solver probe\n\nUTC: {raw['created_utc']}\n\nHeadline: {json.dumps(clean(h), sort_keys=True)[:6000]}\n\nArtifacts: {rel(RUN_DIR / 'summary.md')}, {rel(RUN_DIR / 'raw.json')}, {rel(RUN_DIR / 'completed.json')}\n\nNext: verify external backup for {rel(BACKUP_REQUEST)} and read active lead analysis for request {REQUEST_ID} before selector/refit/training/validation changes.\n",
        encoding="utf-8",
    )


def write_backup_request(raw: Mapping[str, Any]) -> None:
    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v34j_corrected_objective_basin_solver_probe",
        "created_utc": raw["created_utc"],
        "backup_required_before_more_unique_science": True,
        "reason": "new corrected 23-solve Task-C objective-vs-basin evidence, v34j source, docs and registry must be externally recoverable before further science",
        "must_cover": [
            rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(NEXT_REVIEW_REQUEST), rel(RESPONSE_LOG),
            "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv",
        ],
        "new_solver_calls": raw["budget_actual"]["new_solver_calls_this_run"],
        "imported_solver_calls_from_v34g": IMPORTED_V34G_SOLVER_CALLS,
        "task_c_total_solver_calls_including_imported_v34g": raw["budget_actual"]["task_c_solver_calls_including_imported_v34g"],
        "new_plant_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_gate": "lead analysis of corrected v34j evidence before selector/refit/training/validation64/final-test changes",
    })


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-time", required=True)
    ap.add_argument("--backup-commit", required=True)
    ap.add_argument("--backup-package-sha256", required=True)
    ap.add_argument("--backup-package-bytes", type=int, default=0)
    ap.add_argument("--i-accept-v34j-remaining-23-solve-budget", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    patch_runtime()
    try:
        started = now_utc()
        gates = verify_gates(args)
        write_json(RUN_DIR / "run_started.json", {
            "started_utc": started.isoformat(),
            "pid": os.getpid(),
            "method": NAME,
            "gates": gates,
            "new_solver_call_cap": NEW_SOLVE_CAP,
            "imported_v34g_solver_calls": IMPORTED_V34G_SOLVER_CALLS,
            "task_c_total_solver_budget": TASK_C_TOTAL_SOLVE_BUDGET,
            "plant_steps": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        })
        _, stage1_runner, _ = base.v1d.import_legacy_modules()
        preflight = stage1_runner.runtime_preflight()
        if not preflight.get("passed"):
            raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
        stage1_runner.base.v1.latency_verify()
        term_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
        terminals, terminal_receipts = stage1_runner.load_terminal_grid(term_protocol["terminal_grid_readiness_reused_from_v1"])
        for horizon in (15, 35):
            if horizon not in terminals:
                raise ContractError(f"terminal grid missing H{horizon}")
        contexts = base.load_contexts()
        schedule = base.make_schedule(contexts)
        matches = [r for r in schedule if row_key(r) == target_key_tuple()]
        if len(matches) != 1:
            raise ContractError(f"expected exactly one v34g imported cell in schedule, found {len(matches)}")
        write_json(RUN_DIR / "frozen_schedule.json", {
            "created_utc": now_utc().isoformat(),
            "contexts": [{kk: vv for kk, vv in c.items() if kk != "case_snapshot"} for c in contexts],
            "schedule": schedule,
            "imported_v34g_key": TARGET_IMPORTED_KEY,
            "new_solver_call_cap": NEW_SOLVE_CAP,
            "task_c_total_solver_budget": TASK_C_TOTAL_SOLVE_BUDGET,
            "plant_steps": 0,
            "env_reset_calls": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        })
        coeffs = base.parse_terminal_coeff_csv()
        write_json(RUN_DIR / "parsed_terminal_coefficients.json", coeffs)
        context_by_id = {str(c["context_id"]): c for c in contexts}
        k_raw = read_json(V34K_RAW)
        imported_arm = imported_v34g_arm_for_row(matches[0], k_raw)
        arms: List[Dict[str, Any]] = []
        new_solves = 0
        for row in schedule:
            if row_key(row) == target_key_tuple():
                arm = copy.deepcopy(imported_arm)
            else:
                arm = base.solve_arm(row, context_by_id[str(row["context_id"])], terminals, coeffs)
                new_solves += int(arm.get("solve_calls", 0))
                if new_solves > NEW_SOLVE_CAP:
                    raise ContractError("new solver call cap exceeded")
            arms.append(arm)
            write_progress(arms, new_solves, arm)
            print(json.dumps({
                "arms_done": len(arms),
                "new_solver_calls_this_run": new_solves,
                "task_c_total_solver_calls_including_imported": new_solves + IMPORTED_V34G_SOLVER_CALLS,
                "arm": arm.get("arm_id"),
                "imported": bool(arm.get("imported_from_v34g_one_cell_smoke")),
                "accepted": arm.get("accepted"),
                "status": (arm.get("solver_event") or {}).get("return_status"),
                "obj": (arm.get("solver_event") or {}).get("objective_opt_f_num"),
                "recon_rel_error": (arm.get("objective_reconstruction_current") or {}).get("relative_error_vs_solver"),
            }, sort_keys=True), flush=True)
        if len(arms) != TASK_C_TOTAL_SOLVE_BUDGET:
            raise ContractError("did not assemble all 24 Task-C arms")
        if new_solves != NEW_SOLVE_CAP:
            raise ContractError(f"expected {NEW_SOLVE_CAP} new solver calls, got {new_solves}")
        vf_cache = base.build_vf_evaluator_cache(terminals)
        cross = base.cross_evaluate(arms, vf_cache)
        analysis = augment_analysis(base.analyze(arms, cross), arms, new_solves)
        created = now_utc()
        raw = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_corrected_fixed_context_objective_basin_solver_probe_not_validation_not_test",
            "active_lead": "claude-opus-5-5",
            "lead_report": rel(CURRENT_OPUS_REPORT),
            "lead_report_sha256": CURRENT_OPUS_SHA,
            "hypothesis_frozen": "Using the v34k-passing full-label objective reconstruction, the 24-arm fixed-context Task-C design (1 imported v34g + 23 new solves) can distinguish terminal-objective ranking from deterministic warm-start/basin effects without plant rollouts or validation/test access.",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "plant_steps": 0,
            "env_reset_calls": 0,
            "budget_declared": {
                "new_solver_call_cap": NEW_SOLVE_CAP,
                "imported_v34g_solver_calls": IMPORTED_V34G_SOLVER_CALLS,
                "task_c_total_solver_budget": TASK_C_TOTAL_SOLVE_BUDGET,
                "plant_steps": 0,
                "env_reset_calls": 0,
                "new_training_or_gradient_steps": 0,
                "selector_refits": 0,
                "validation64_episodes": 0,
                "sealed_test_episodes": 0,
            },
            "budget_actual": {
                "arms": len(arms),
                "new_solver_calls_this_run": int(new_solves),
                "imported_solver_calls_from_v34g": IMPORTED_V34G_SOLVER_CALLS,
                "task_c_solver_calls_including_imported_v34g": int(new_solves + IMPORTED_V34G_SOLVER_CALLS),
                "plant_steps": 0,
                "env_reset_calls": 0,
                "new_training_or_gradient_steps": 0,
                "selector_refits": 0,
                "validation64_episodes": 0,
                "sealed_test_episodes": 0,
            },
            "gates": gates,
            "runtime_preflight": preflight,
            "terminal_receipts": {str(kk): vv for kk, vv in terminal_receipts.items()},
            "contexts": [{kk: vv for kk, vv in c.items() if kk != "case_snapshot"} for c in contexts],
            "schedule": schedule,
            "imported_v34g_arm_key": TARGET_IMPORTED_KEY,
            "formula_spec": FORMULA_SPEC,
            "arms": arms,
            "cross_evaluation": cross,
            "analysis": analysis,
            "input_hashes": hash_existing([
                Path(__file__).resolve(), OPUS_PLAN_READY, OPUS_LATEST, CURRENT_OPUS_REPORT, V34G_DONE, V34G_RAW, V34K_DONE, V34K_RAW,
                ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34f_objective_basin_solver_probe_v0.py",
                ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0.py",
            ]),
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
            "interpretation_limits": ["opened development contexts only", "not validation64", "not sealed/final test", "no closed-loop adaptive selector", "no plant continuation", "one arm imported from v34g one-cell smoke"],
            "next_review_request_id": REQUEST_ID,
            "backup_request_after_run": rel(BACKUP_REQUEST),
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        write_backup_request(raw)
        update_docs(raw)
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [
            Path(__file__).resolve(), STATE, BACKUP_REQUEST, NEXT_REVIEW_REQUEST, RESPONSE_LOG,
            ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv",
        ]
        completed = {
            "status": "complete",
            "passed": True,
            "hard_pass": True,
            "created_utc": created.isoformat(),
            "classification": raw["classification"],
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_actual": raw["budget_actual"],
            "headline": analysis["headline"],
            "summary": rel(RUN_DIR / "summary.md"),
            "raw": rel(RUN_DIR / "raw.json"),
            "backup_request": rel(BACKUP_REQUEST),
            "next_review_request_id": REQUEST_ID,
            "hashes": hash_existing(sorted(set(files))),
        }
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({
            "completed": rel(RUN_DIR / "completed.json"),
            "summary": rel(RUN_DIR / "summary.md"),
            "raw": rel(RUN_DIR / "raw.json"),
            "headline": completed["headline"],
            "backup_request": rel(BACKUP_REQUEST),
            "next_review_request_id": REQUEST_ID,
        }, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        fail = {
            "status": "failed",
            "created_utc": now_utc().isoformat(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "classification": "development_IMPROVED_corrected_fixed_context_objective_basin_solver_probe_not_validation_not_test",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_caps": {"new_solver_call_cap": NEW_SOLVE_CAP, "plant_steps": 0, "env_reset_calls": 0},
        }
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(
            f"# v34j corrected objective-basin solver probe failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR / 'failed.json')}\n\nNo validation64 or sealed test access was requested by this script. Preserve failure and repair exact operational defect before retry; do not rerun v34k or spend extra Task-C calls without accounting.\n",
            encoding="utf-8",
        )
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
