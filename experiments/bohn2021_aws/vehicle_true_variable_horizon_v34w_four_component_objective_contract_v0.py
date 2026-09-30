#!/usr/bin/env python3
"""v34w / T-B6+T-B7 four-component objective-contract confirmation.

Development-only bounded diagnostic authorized by the active Opus plan
`20260930T125337Z_10774a` after v34u localized the forced-nonconverged
objective residual to the do-mpc epsilon/slack component.

This script performs at most six low-level MPC solver calls on already-opened
source242 state only: four forced max_iter=1 cells inherited from v34u plus one
H12 default-iteration convergence control. It performs no plant rollout, no
env.step/reset after construction, no validation64 access, no sealed/final-test
access, no training, and no selector refit.

The acceptance gate is the active lead's T-B6/T-B7 specification:
  * J_recon4 = lterm + terminal + rterm + epsterm must match J_solver at every
    captured solve point (H35 allowance 1e-5, all other rows 1e-6);
  * the converged H12 control must have rel_err3 <= 1e-6 under the old
    no-epsilon formula;
  * a zero-solve perturbation check must show that the direct NLP evaluator is a
    genuine f/g evaluator, not a constant echo.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
REPRO_DIR = ROOT / "experiments/bohn2021_reproduction"
for _p in (AWS_DIR, REPRO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vehicle_true_variable_horizon_v34n_nonconverged_objective_contract_probe_v0 as v34n  # noqa:E402
import vehicle_true_variable_horizon_v34o_loader_gate_v0 as v34o  # noqa:E402
import vehicle_true_variable_horizon_v34s_loader_gate_v0 as v34s  # noqa:E402
import vehicle_true_variable_horizon_v34t_nonconverged_objective_contract_probe_v0 as v34t  # noqa:E402
import vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0 as v34u  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34w_four_component_objective_contract_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34w_four_component_objective_contract.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34W_FOUR_COMPONENT_OBJECTIVE_CONTRACT_{STAMP}.json"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
PLAN_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
OPUS_LATEST = ROOT / "docs/bohn2021_takeover/opus_lead/LATEST.md"
CURRENT_OPUS_REQUEST = "execution-result:20260930T125233_d07565d9"
CURRENT_OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T125337Z_10774a.md"
CURRENT_OPUS_SHA = "8c1b4d4fbb8d5c9482f90f9b14f38744ea5ee8df66beac0cb9aaedb82546b5ee"
V34U_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z/completed.json"
V34U_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z/raw.json"
V34V_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34v_eps_objective_localization_postdiagnostic_v0_20260930T130526Z/completed.json"
V34V_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34v_eps_objective_localization_postdiagnostic_v0_20260930T130526Z/raw.json"
ASTRA_REPORT = ROOT / "docs/bohn2021_takeover/astra_reviews/20260930T123247Z.md"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
REQUEST_ID = f"v34w-four-component-objective-contract-{STAMP}"
MARKER = f"vehicle-v34w-four-component-objective-contract-{STAMP}"
SOLVE_CAP = 6
FORCED_OPTIONS = {"ipopt.max_iter": 1, "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False}
QUIET_OPTIONS = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False}
TB6_CELLS = [
    {"context_id": "source242_slot0_branch_start", "horizon": 12, "terminal_mode": "V15_shared", "initialization": "canonical", "cell_role": "forced_maxiter1_H12", "solve_mode": "forced_maxiter1"},
    {"context_id": "source242_slot0_branch_start", "horizon": 15, "terminal_mode": "V15_shared", "initialization": "canonical", "cell_role": "forced_maxiter1_H15", "solve_mode": "forced_maxiter1"},
    {"context_id": "source242_slot0_branch_start", "horizon": 35, "terminal_mode": "V15_shared", "initialization": "canonical", "cell_role": "forced_maxiter1_H35", "solve_mode": "forced_maxiter1"},
    {"context_id": "source242_slot0_branch_start", "horizon": 15, "terminal_mode": "V15_shared", "initialization": "goal_facing", "cell_role": "forced_maxiter1_alias_H15", "solve_mode": "forced_maxiter1"},
    {"context_id": "source242_slot0_branch_start", "horizon": 12, "terminal_mode": "V15_shared", "initialization": "canonical", "cell_role": "converged_control_H12_default", "solve_mode": "default_converged"},
]
_SOURCE_CHECK: Dict[str, Any] = {}


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    return v34n.rel(path)


def read_json(path: Path) -> Any:
    return v34n.read_json(path)


def write_json(path: Path, value: Any) -> None:
    return v34n.write_json(path, value)


def sha256(path: Path) -> str:
    return v34n.sha256(path)


def clean(value: Any) -> Any:
    return v34n.clean(value)


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        t = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def hash_existing(paths: Iterable[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        try:
            if p.exists() and p.is_file():
                out[rel(p)] = sha256(p)
        except Exception:
            pass
    return out


def source_objective_check() -> Dict[str, Any]:
    controller_py = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/do-mpc-horizon/do_mpc/controller.py"
    optimizer_py = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/do-mpc-horizon/do_mpc/optimizer.py"
    gym_ctrl_py = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/gym-horizon/gym_let_mpc/controllers.py"
    texts = {"controller": controller_py.read_text(encoding="utf-8", errors="replace"), "optimizer": optimizer_py.read_text(encoding="utf-8", errors="replace"), "gym_controller": gym_ctrl_py.read_text(encoding="utf-8", errors="replace")}
    checks = {
        "do_mpc_objective_adds_epsterm": "obj += self.discount_factor ** k * self.epsterm_fun" in texts["controller"],
        "do_mpc_soft_constraint_creates_slack_cost": "self.slack_cost += sum1(penalty_term_cons*epsilon)" in texts["optimizer"],
        "vehicle_tta_hmpc_defines_soft_obstacle_constraints": '"soft": soft_constraint' in texts["gym_controller"] and '"cost": 1000' in texts["gym_controller"] and 'obj_{}_distance' in texts["gym_controller"],
        "ahmpc_sets_default_ipopt_max_iter_250": 'mpc_config["nlpsol_opts"]["ipopt.max_iter"] = 250' in texts["gym_controller"],
    }
    return {
        "paths": {"controller": rel(controller_py), "optimizer": rel(optimizer_py), "gym_controller": rel(gym_ctrl_py)},
        "hashes": hash_existing([controller_py, optimizer_py, gym_ctrl_py]),
        "checks": checks,
        "epsterm_source_documented_objective_present": bool(checks["do_mpc_objective_adds_epsterm"] and checks["do_mpc_soft_constraint_creates_slack_cost"] and checks["vehicle_tta_hmpc_defines_soft_obstacle_constraints"]),
        "interpretation": "Source-level check only: do-mpc adds discounted epsterm_fun to the MPC objective, Optimizer.set_nl_cons creates slack_cost for soft constraints, and vehicle TTAHMPC creates soft obstacle constraints with cost 1000.",
    }


def verify_gates(args: argparse.Namespace) -> Dict[str, Any]:
    global _SOURCE_CHECK
    ready = read_json(PLAN_READY)
    if ready.get("request_id") != CURRENT_OPUS_REQUEST:
        raise ContractError(f"active PLAN_READY request mismatch: {ready.get('request_id')} != {CURRENT_OPUS_REQUEST}")
    if ready.get("report_sha256") != CURRENT_OPUS_SHA or not CURRENT_OPUS_REPORT.exists() or sha256(CURRENT_OPUS_REPORT) != CURRENT_OPUS_SHA:
        raise ContractError("active Opus T-B6 report missing or sha mismatch")
    latest_text = OPUS_LATEST.read_text(encoding="utf-8", errors="replace") if OPUS_LATEST.exists() else ""
    report_text = CURRENT_OPUS_REPORT.read_text(encoding="utf-8", errors="replace")
    for token in ["T", "B6", "four", "component objective contract", "authorized now", "T", "B7", "direct NLP evaluator"]:
        if token not in report_text:
            raise ContractError(f"active Opus report does not contain expected T-B6/T-B7 authorization token fragment: {token!r}")
    if rel(CURRENT_OPUS_REPORT) not in latest_text:
        raise ContractError("Opus LATEST.md does not point to current T-B6 report")
    for p in [V34U_DONE, V34U_RAW, V34V_DONE, V34V_RAW]:
        if not p.exists():
            raise ContractError("missing prerequisite artifact " + rel(p))
    u_done = read_json(V34U_DONE)
    v_done = read_json(V34V_DONE)
    if int((u_done.get("budget_actual") or {}).get("new_solver_calls", 0)) != 4:
        raise ContractError("v34u did not record four solver calls")
    if (u_done.get("headline") or {}).get("T_B3_A_constructor_binding_gate_all_cells") is not True:
        raise ContractError("v34u constructor binding gate was not true")
    if (v_done.get("headline") or {}).get("diagnostic_pass") is not True:
        raise ContractError("v34v epsilon localization did not pass")
    bt = parse_time(args.backup_time)
    if bt is None:
        raise ContractError("backup_time not parseable")
    u_time = parse_time(u_done.get("created_utc"))
    if u_time is not None and bt <= u_time:
        raise ContractError("backup context does not postdate v34u artifacts")
    _SOURCE_CHECK = source_objective_check()
    return {
        "active_lead_plan_ready": rel(PLAN_READY),
        "active_lead_report": rel(CURRENT_OPUS_REPORT),
        "active_lead_report_sha256": CURRENT_OPUS_SHA,
        "active_lead_request": CURRENT_OPUS_REQUEST,
        "t_b6_t_b7_authorization_fragments_present": True,
        "v34u_completed": rel(V34U_DONE),
        "v34u_raw": rel(V34U_RAW),
        "v34u_headline": u_done.get("headline"),
        "v34v_completed": rel(V34V_DONE),
        "v34v_raw": rel(V34V_RAW),
        "v34v_headline": v_done.get("headline"),
        "source_objective_check": _SOURCE_CHECK,
        "backup_proof_from_supervisor_context": {"time": args.backup_time, "commit": args.backup_commit, "package_sha256": args.backup_package_sha256, "package_bytes": int(args.backup_package_bytes), "status": "verified_from_supervisor_context_not_revalidated_by_script", "covers_v34u_inputs": True},
        "authorized_budget": {"low_level_solver_attempt_cap": SOLVE_CAP, "scheduled_cells": len(TB6_CELLS), "plant_steps": 0, "env_step_calls_after_construction": 0, "validation64_episodes": 0, "sealed_test_episodes": 0, "training_or_refit": 0},
    }


def install_per_cell_nlpsol_patch() -> None:
    import casadi as ca  # type: ignore
    import do_mpc.controller as controller_mod  # type: ignore
    try:
        import do_mpc.optimizer as optimizer_mod  # type: ignore
    except Exception:
        optimizer_mod = None  # type: ignore
    if getattr(controller_mod, "_v34w_nlpsol_patched", False):
        return
    original = getattr(controller_mod, "nlpsol", None) or getattr(ca, "nlpsol", None)
    if original is None:
        raise ContractError("no nlpsol callable found for v34w patch")

    def wrapped_nlpsol(name: Any, solver: Any, nlp: Mapping[str, Any], opts: Optional[Mapping[str, Any]] = None) -> Any:
        opts_in = dict(opts or {})
        opts_out = dict(opts_in)
        arm_id = str(v34n._CURRENT_ARM_ID or "")
        mode = "default_converged" if "converged_control" in arm_id else "forced_maxiter1"
        if mode == "forced_maxiter1":
            opts_out.update(FORCED_OPTIONS)
        else:
            for k, val in QUIET_OPTIONS.items():
                opts_out.setdefault(k, val)
        cap = {
            "arm_id": v34n._CURRENT_ARM_ID,
            "solver_name": str(name),
            "solver_kind": str(solver),
            "solve_mode": mode,
            "opts_in_subset": {k: opts_in.get(k) for k in sorted(set(list(FORCED_OPTIONS.keys()) + ["ipopt.max_iter"]))},
            "opts_out_subset": {k: opts_out.get(k) for k in sorted(set(list(FORCED_OPTIONS.keys()) + ["ipopt.max_iter"]))},
            "effective_ipopt_max_iter": opts_out.get("ipopt.max_iter"),
            "effective_ipopt_max_iter_equals_1": opts_out.get("ipopt.max_iter") == 1,
            "constructor_lookup_site": "v34w patched do_mpc.controller.nlpsol plus casadi/do_mpc.optimizer aliases",
            "nlp_meta": {k: str(getattr(val, "shape", None)) for k, val in (nlp or {}).items() if k in ("x", "f", "g", "p")},
            "nlp_expr": nlp,
        }
        v34n._CAPTURES.append(cap)
        return original(name, solver, nlp, opts_out)

    setattr(controller_mod, "_v34w_original_nlpsol", original)
    setattr(controller_mod, "nlpsol", wrapped_nlpsol)
    setattr(ca, "nlpsol", wrapped_nlpsol)
    if optimizer_mod is not None:
        setattr(optimizer_mod, "nlpsol", wrapped_nlpsol)
    setattr(controller_mod, "_v34w_nlpsol_patched", True)
    for label in ["casadi.nlpsol", "do_mpc.controller.nlpsol"] + (["do_mpc.optimizer.nlpsol"] if optimizer_mod is not None else []):
        if label not in v34n._PATCHED_MODULES:
            v34n._PATCHED_MODULES.append(label)


def install_constructor_gate() -> None:
    original_create_env = v34n.base.create_env
    original_solve_cell = v34n.solve_cell

    def create_env_gate(h: int, terminal: Any) -> Any:
        before = len(v34n._CAPTURES)
        env = original_create_env(h, terminal)
        new_caps = v34n._CAPTURES[before:]
        arm_id = str(v34n._CURRENT_ARM_ID or "")
        mode = "default_converged" if "converged_control" in arm_id else "forced_maxiter1"
        maxiters = [c.get("effective_ipopt_max_iter") for c in new_caps]
        gate = {
            "arm_id": arm_id,
            "solve_mode": mode,
            "captures_added_during_create_env": len(new_caps),
            "effective_ipopt_max_iter_values": maxiters,
            "forced_maxiter1_gate": bool(mode == "forced_maxiter1" and any(v == 1 for v in maxiters)),
            "default_converged_gate": bool(mode == "default_converged" and new_caps and all(v != 1 for v in maxiters)),
        }
        setattr(env.control_system.controller.mpc, "_v34w_constructor_binding_gate", gate)
        if not new_caps:
            raise ContractError("T-B6 constructor gate failed: no nlpsol capture")
        if mode == "forced_maxiter1" and not gate["forced_maxiter1_gate"]:
            raise ContractError("T-B6 forced constructor gate failed: ipopt.max_iter=1 not evidenced")
        if mode == "default_converged" and not gate["default_converged_gate"]:
            raise ContractError("T-B6 converged constructor gate failed: default solve still has forced max_iter=1")
        return env

    def solve_cell_gate(row: Mapping[str, Any], context: Mapping[str, Any], terminals: Mapping[int, Any], idx: int) -> Dict[str, Any]:
        arm = original_solve_cell(row, context, terminals, idx)
        caps = [c for c in v34n._CAPTURES if c.get("arm_id") == arm.get("arm_id")]
        mode = str(row.get("solve_mode", "forced_maxiter1"))
        arm["solve_mode"] = mode
        arm["T_B6_constructor_binding"] = {
            "captures_for_arm": len(caps),
            "max_iter_values": [c.get("effective_ipopt_max_iter") for c in caps],
            "forced_maxiter1_gate": bool(mode == "forced_maxiter1" and any(c.get("effective_ipopt_max_iter") == 1 for c in caps)),
            "default_converged_gate": bool(mode == "default_converged" and caps and all(c.get("effective_ipopt_max_iter") != 1 for c in caps)),
        }
        rec3 = arm.get("objective_reconstruction") or {}
        rec4 = (arm.get("alias_branch_variants") or {}).get("eps_and_rterm") or (arm.get("alias_branch_variants") or {}).get("eps_only") or {}
        arm["objective_reconstruction_three_component_noeps"] = rec3
        arm["objective_reconstruction_four_component_with_eps"] = rec4
        return arm

    v34n.base.create_env = create_env_gate  # type: ignore[assignment]
    v34n.solve_cell = solve_cell_gate  # type: ignore[assignment]


def direct_nlp_eval_independent(mpc: Any, capture: Optional[Mapping[str, Any]], idx: int) -> Dict[str, Any]:
    base_eval = v34n.direct_nlp_eval_original(mpc, capture, idx) if hasattr(v34n, "direct_nlp_eval_original") else {}
    if v34n.ca is None:
        base_eval.update({"T_B7_available": False, "T_B7_reason": "casadi_import_failed"})
        return base_eval
    if not capture or not isinstance(capture.get("nlp_expr"), Mapping):
        base_eval.update({"T_B7_available": False, "T_B7_reason": "no_captured_nlp_expr"})
        return base_eval
    try:
        nlp = capture["nlp_expr"]
        f_fn = v34n.ca.Function(f"v34w_f_{idx}", [nlp.get("x"), nlp.get("p")], [nlp.get("f")])
        g_fn = v34n.ca.Function(f"v34w_g_{idx}", [nlp.get("x"), nlp.get("p")], [nlp.get("g")])
        x0 = np.asarray(v34n.arr(v34n.cat_arg(mpc.opt_x_num)), dtype=float).reshape((-1, 1))
        p0 = v34n.cat_arg(mpc.opt_p_num)
        e1 = np.zeros_like(x0)
        if e1.size:
            e1.reshape(-1)[0] = 1e-6
        xs = {"reported_x": x0, "x_plus_1e_minus_6_e1": x0 + e1, "x_scaled_1p5": 1.5 * x0}
        vals: Dict[str, Any] = {}
        gvals: Dict[str, np.ndarray] = {}
        for label, xv in xs.items():
            vals[label] = v34n.finite_float(f_fn(xv, p0), None)
            gvals[label] = v34n.arr(g_fn(xv, p0))
        f_base = vals.get("reported_x")
        solver = v34n.finite_float(getattr(mpc, "opt_f_num", None), None)
        deltas = {k: None if f_base is None or v is None else float(abs(float(v) - float(f_base))) for k, v in vals.items() if k != "reported_x"}
        g_base = gvals.get("reported_x", np.asarray([]))
        gdeltas: Dict[str, Optional[float]] = {}
        for k, gv in gvals.items():
            if k == "reported_x":
                continue
            gdeltas[k] = None if g_base.size == 0 or gv.size != g_base.size else float(np.max(np.abs(gv - g_base)))
        f_rel = None if f_base is None or solver is None else float(abs(float(f_base) - float(solver)) / max(1.0, abs(float(solver))))
        max_df = max([d for d in deltas.values() if d is not None] or [0.0])
        max_dg = max([d for d in gdeltas.values() if d is not None] or [0.0])
        base_eval.update({
            "T_B7_available": True,
            "T_B7_f_values": vals,
            "T_B7_f_deltas_vs_reported_x": deltas,
            "T_B7_g_linf_deltas_vs_reported_x": gdeltas,
            "T_B7_reported_x_rel_error_vs_solver": f_rel,
            "T_B7_independence_perturbation_pass": bool(f_rel is not None and f_rel <= 1e-10 and max_df > 1e-9 and max_dg > 1e-9),
            "T_B7_falsification_identical_f_all_inputs": bool(max_df <= 1e-15),
            "T_B7_note": "Zero-solve perturbation check on the captured NLP expression at post-solve x, x+1e-6*e1, and 1.5*x. No pre-solve x vector was persisted by v34u.",
        })
    except Exception as exc:
        base_eval.update({"T_B7_available": False, "T_B7_reason": "independent_eval_failed", "T_B7_error": repr(exc), "T_B7_traceback_tail": traceback.format_exc().splitlines()[-8:]})
    return base_eval


def analyze_tb6(arms: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    pass4_flags: List[bool] = []
    hard4: List[str] = []
    converged_control_pass = False
    tb7_pass_rows = 0
    forced_status = 0
    constructor_rows_pass = 0
    for arm in arms:
        rec3 = arm.get("objective_reconstruction_three_component_noeps") or arm.get("objective_reconstruction") or {}
        rec4 = arm.get("objective_reconstruction_four_component_with_eps") or {}
        hval = int(arm.get("horizon", -1))
        mode = str(arm.get("solve_mode", ""))
        status = (arm.get("solver_event_summary") or {}).get("return_status")
        success = bool((arm.get("solver_event_summary") or {}).get("success"))
        rel3 = rec3.get("relative_error")
        rel4 = rec4.get("relative_error")
        allowed = 1e-5 if hval == 35 and mode == "forced_maxiter1" else 1e-6
        pass4 = bool(rel4 is not None and float(rel4) <= allowed)
        pass4_flags.append(pass4)
        if rel4 is not None and float(rel4) > 1e-4:
            hard4.append(str(arm.get("arm_id")))
        if mode == "forced_maxiter1" and arm.get("forced_nonconverged_or_near_offoptimal"):
            forced_status += 1
        if (arm.get("T_B6_constructor_binding") or {}).get("forced_maxiter1_gate") or (arm.get("T_B6_constructor_binding") or {}).get("default_converged_gate"):
            constructor_rows_pass += 1
        dnlp = arm.get("direct_nlp_eval") or {}
        if dnlp.get("T_B7_independence_perturbation_pass") is True:
            tb7_pass_rows += 1
        epsterm = rec4.get("epsterm_total")
        rterm = rec4.get("input_regularization_total")
        if mode == "default_converged" and success and str(status) == "Solve_Succeeded" and rel3 is not None and float(rel3) <= 1e-6:
            converged_control_pass = True
        rows.append({
            "arm_id": arm.get("arm_id"), "role": arm.get("cell_role"), "horizon": hval, "initialization": arm.get("initialization"), "solve_mode": mode,
            "solve_calls": arm.get("solve_calls"), "return_status": status, "success": success, "iterations": (arm.get("solver_event_summary") or {}).get("iterations"),
            "lterm_total": rec4.get("stage_lterm_total"), "terminal_discounted": rec4.get("terminal_discounted"), "rterm_total": rterm, "epsterm_total": epsterm,
            "J_solver": rec4.get("solver_objective"), "J_recon4": rec4.get("total"), "rel_err4": rel4, "J_recon3": rec3.get("total"), "rel_err3": rel3,
            "allowed_rel_err4": allowed, "G2_cell_pass4": pass4,
            "residual3_over_terminal": (arm.get("predeclared_discriminants") or {}).get("residual_over_terminal_discounted"),
            "T_B7_independence_pass": dnlp.get("T_B7_independence_perturbation_pass"),
            "T_B7_max_f_delta": max([v for v in (dnlp.get("T_B7_f_deltas_vs_reported_x") or {}).values() if v is not None] or [None]),
            "T_B7_max_g_linf_delta": max([v for v in (dnlp.get("T_B7_g_linf_deltas_vs_reported_x") or {}).values() if v is not None] or [None]),
        })
    h1_pass = bool(len(rows) == len(TB6_CELLS) and all(pass4_flags) and not hard4)
    source_pass = bool((_SOURCE_CHECK.get("epsterm_source_documented_objective_present") is True))
    constructor_pass = bool(constructor_rows_pass == len(TB6_CELLS))
    forced_gate = bool(forced_status >= 4)
    tb7_pass = bool(tb7_pass_rows >= 1)
    return {"per_cell": rows, "headline": {
        "G2_pass": bool(h1_pass and converged_control_pass and tb7_pass and constructor_pass and forced_gate and source_pass),
        "T_B6_H1_four_component_pass_all_cells": h1_pass,
        "T_B6_H2_converged_three_component_control_pass": converged_control_pass,
        "T_B7_direct_nlp_independence_pass": tb7_pass,
        "T_B7_pass_rows": tb7_pass_rows,
        "T_B6_constructor_binding_gate_all_cells": constructor_pass,
        "T_B6_forced_status_gate_four_forced_cells": forced_gate,
        "T_B6_source_epsterm_present": source_pass,
        "hard_defect_arm_ids_rel_err4_gt_1e-4": hard4,
        "cells_completed": len(arms), "new_solver_calls": int(sum(int(a.get("solve_calls", 0) or 0) for a in arms)), "low_level_solver_call_cap": SOLVE_CAP,
        "forced_nonconverged_or_near_offoptimal_cells": forced_status,
        "converged_control_rows": int(sum(1 for a in arms if str(a.get("solve_mode")) == "default_converged")),
        "plant_steps": 0, "env_step_calls_after_construction": 0, "training_or_refit": 0, "validation64_episodes": 0, "sealed_test_episodes": 0,
        "max_rel_err4": max([float(r["rel_err4"]) for r in rows if r.get("rel_err4") is not None] or [None]),
        "max_rel_err3": max([float(r["rel_err3"]) for r in rows if r.get("rel_err3") is not None] or [None]),
        "source_objective_check": _SOURCE_CHECK,
        "interpretation": "T-B6/T-B7 development gate: pass means the corrected four-component objective accessor is numerically consistent on forced and converged cells; it does not constitute validation or final-test evidence.",
    }}


def write_outputs_tb6(raw: Mapping[str, Any]) -> None:
    h = raw["analysis"]["headline"]
    rows = raw["analysis"]["per_cell"]
    csv_path = v34n.RUN_DIR / "cell_metrics.csv"
    fields = ["arm_id", "role", "horizon", "initialization", "solve_mode", "solve_calls", "return_status", "success", "iterations", "lterm_total", "terminal_discounted", "rterm_total", "epsterm_total", "J_solver", "J_recon4", "rel_err4", "J_recon3", "rel_err3", "allowed_rel_err4", "G2_cell_pass4", "residual3_over_terminal", "T_B7_independence_pass", "T_B7_max_f_delta", "T_B7_max_g_linf_delta"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: clean(r.get(k)) for k in fields})
    lines = [
        "# v34w T-B6/T-B7 four-component objective-contract confirmation",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only objective-contract diagnostic under active Opus plan `{rel(CURRENT_OPUS_REPORT)}`.",
        "",
        "## Budget",
        f"- New low-level solver calls: `{h['new_solver_calls']}` / `{SOLVE_CAP}`.",
        "- Plant steps: `0`; env.step/reset after construction: `0`; training/refit: `0`; validation64: `0`; sealed test: `0`.",
        "",
        "## Headline",
        f"- Overall T-B6/T-B7 gate (`G2_pass`): `{h['G2_pass']}`.",
        f"- Four-component H1 pass all cells: `{h['T_B6_H1_four_component_pass_all_cells']}`; max rel_err4 `{h['max_rel_err4']}`.",
        f"- Converged H12 three-component control pass: `{h['T_B6_H2_converged_three_component_control_pass']}`.",
        f"- Direct NLP perturbation independence pass: `{h['T_B7_direct_nlp_independence_pass']}` ({h['T_B7_pass_rows']} rows).",
        f"- Source epsterm present: `{h['T_B6_source_epsterm_present']}`.",
        f"- Constructor gate all cells: `{h['T_B6_constructor_binding_gate_all_cells']}`; forced status gate: `{h['T_B6_forced_status_gate_four_forced_cells']}`.",
        "",
        "## Per-cell objective metrics",
        "",
        "| role | H | mode | status | iter | rel_err4 | rel_err3 | epsterm | rterm | T-B7 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(f"| `{r.get('role')}` | {r.get('horizon')} | `{r.get('solve_mode')}` | `{r.get('return_status')}` | {r.get('iterations')} | {r.get('rel_err4')} | {r.get('rel_err3')} | {r.get('epsterm_total')} | {r.get('rterm_total')} | `{r.get('T_B7_independence_pass')}` |")
    lines += ["", "## Source objective check", "", f"`{json.dumps(clean(_SOURCE_CHECK.get('checks', {})), sort_keys=True)}`", "", "This is not population validation, not a speed claim, not selector training, and not final-test evidence. If the gate passes, it only closes the local objective-accessor contract with the corrected four-component formula for the active-lead Task-C dependency.", "", f"Raw: `{rel(v34n.RUN_DIR/'raw.json')}`; CSV: `{rel(csv_path)}`; completed: `{rel(v34n.RUN_DIR/'completed.json')}`; backup request: `{rel(BACKUP_REQUEST)}`."]
    (v34n.RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(BACKUP_REQUEST, {"request": "backup_after_v34w_four_component_objective_contract", "created_utc": raw["created_utc"], "backup_required_before_more_unique_science": True, "reason": "new T-B6/T-B7 solver-contract evidence and source gate", "must_cover": [rel(Path(__file__).resolve()), rel(v34n.RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(NEXT_REVIEW_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "new_solver_calls": h["new_solver_calls"], "new_plant_steps": 0, "env_step_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
    write_json(NEXT_REVIEW_REQUEST, {"request_id": REQUEST_ID, "created": raw["created_utc"], "status": "gate_evidence_ready" if h["G2_pass"] else "analysis_requested", "trigger": "v34w T-B6/T-B7 four-component objective-contract diagnostic completed", "experiment_id": NAME, "active_lead_report": rel(CURRENT_OPUS_REPORT), "active_lead_report_sha256": CURRENT_OPUS_SHA, "question": "Interpret v34w T-B6/T-B7. If accepted, A13c-3 is resolved with corrected four-component formula and the active plan conditionally authorizes T-C0. If failed, classify the exact objective/source/evaluator defect. Do not open sealed test.", "evidence_paths": [rel(v34n.RUN_DIR/"summary.md"), rel(v34n.RUN_DIR/"raw.json"), rel(v34n.RUN_DIR/"completed.json"), rel(csv_path), rel(V34U_RAW), rel(V34V_RAW), rel(CURRENT_OPUS_REPORT), rel(ASTRA_REPORT), rel(RESPONSE_LOG)], "budget_actual": raw["budget_actual"], "headline": h, "backup_required_before_more_unique_science": rel(BACKUP_REQUEST)})
    block = f"""
<!-- {MARKER} -->
## v34w T-B6/T-B7 four-component objective-contract diagnostic

UTC: {raw['created_utc']}. Executed active-lead T-B6/T-B7 with new low-level solver calls={h['new_solver_calls']}/{SOLVE_CAP}, plant/env.step=0, training/refit=0, validation64=0, sealed test=0. Overall G2_pass={h['G2_pass']}; four-component all-cell pass={h['T_B6_H1_four_component_pass_all_cells']}; converged H12 three-component control pass={h['T_B6_H2_converged_three_component_control_pass']}; direct NLP perturbation pass={h['T_B7_direct_nlp_independence_pass']}; source epsterm present={h['T_B6_source_epsterm_present']}; max rel_err4={h['max_rel_err4']}. Evidence: `{rel(v34n.RUN_DIR/'summary.md')}`, `{rel(v34n.RUN_DIR/'raw.json')}`, `{rel(csv_path)}`. Backup request: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v34w T-B6/T-B7\n\nUTC: {raw['created_utc']}\n\nHeadline: {json.dumps(clean(h), sort_keys=True)}\n\nArtifacts: `{rel(v34n.RUN_DIR/'summary.md')}`, `{rel(v34n.RUN_DIR/'raw.json')}`, `{rel(csv_path)}`, `{rel(v34n.RUN_DIR/'completed.json')}`.\n\nNext: verify external backup for `{rel(BACKUP_REQUEST)}`. If G2_pass is true and active lead accepts, proceed to T-C0; otherwise return raw evidence to active lead. Validation64/sealed test remain closed.\n", encoding="utf-8")
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([raw["created_utc"], NAME, raw["classification"], "T-B6 four-component objective plus T-B7 evaluator perturbation", "opened_development_solver_contract_no_validation_no_test", h["cells_completed"], 0, h["new_solver_calls"], 0, 0, False, rel(v34n.RUN_DIR/"completed.json"), MARKER])


def patch_runtime() -> None:
    v34n.NAME = NAME
    v34n.STAMP = STAMP
    v34n.RUN_DIR = RUN_DIR
    v34n.STATE = STATE
    v34n.BACKUP_REQUEST = BACKUP_REQUEST
    v34n.REQUEST_ID = REQUEST_ID
    v34n.MARKER = MARKER
    v34n.OPUS_REPORT = CURRENT_OPUS_REPORT
    v34n.OPUS_REPORT_SHA = CURRENT_OPUS_SHA
    v34n.OPUS_REQUEST = CURRENT_OPUS_REQUEST
    v34n.SOLVE_CAP = SOLVE_CAP
    v34n.CELLS = TB6_CELLS
    v34n._CAPTURES.clear()
    v34n._PATCHED_MODULES.clear()
    v34n._CURRENT_ARM_ID = None
    v34n._TOTAL_SOLVER_CALLS = 0
    v34n.base.load_contexts = v34o.load_contexts_strict
    v34n.base.configure_context_no_reset = v34t.configure_context_no_reset_strict_numeric_tvp
    v34n.base.extract_goal_xy = v34s.strict_authoritative_goal_xy
    v34u.install_tta_hmpc_presolve_obj_guard()
    v34n.verify_gates = verify_gates  # type: ignore[assignment]
    v34n.patch_nlpsol_constructor = install_per_cell_nlpsol_patch  # type: ignore[assignment]
    v34n.direct_nlp_eval_original = v34n.direct_nlp_eval  # type: ignore[attr-defined]
    v34n.direct_nlp_eval = direct_nlp_eval_independent  # type: ignore[assignment]
    install_constructor_gate()
    v34n.analyze = analyze_tb6  # type: ignore[assignment]
    v34n.write_outputs = write_outputs_tb6  # type: ignore[assignment]


def run(argv: Optional[Sequence[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--expected-plan-request", required=True)
    ns, remaining = pre.parse_known_args(raw_args)
    if str(ns.expected_plan_request) != CURRENT_OPUS_REQUEST:
        raise ContractError(f"expected-plan-request mismatch for v34w: {ns.expected_plan_request} != {CURRENT_OPUS_REQUEST}")
    patch_runtime()
    return v34n.run(remaining)


if __name__ == "__main__":
    raise SystemExit(run())
