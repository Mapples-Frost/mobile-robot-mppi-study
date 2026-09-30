#!/usr/bin/env python3
"""v34n / A13c-3 bounded non-converged objective-contract probe.

Active Opus plan 20260930T111341Z_dd0c10 authorizes <=6 low-level
solver attempts on the already-opened source242/slot0 development cell, with
plant=0, env_step=0, validation64=0, sealed_test=0, training/refit=0.  The
scientific purpose is not controller validation: it stress-tests the v34k/v34m
objective reconstruction formula away from the single optimal H15 point and
records the predeclared residual/terminal ratio needed by the lead.

Cells executed here:
  * source242_slot0_branch_start, V15_shared, canonical, H in {12,15,35};
  * one H15 goal_facing off-optimum/alias-separation cell.

The script patches do-mpc's nlpsol construction before each fresh controller is
created, forcing ipopt.max_iter=1 where the solver is built.  This should produce
non-converged or near-off-optimal returned iterates while still spending exactly
one low-level solver attempt per cell unless a controller tries a second retry;
second retries are blocked before entering the solver and recorded.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
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

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
for _p in (AWS_DIR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0 as base  # noqa:E402
import vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0 as kacc  # noqa:E402

try:
    import casadi as ca  # type: ignore
except Exception:  # pragma: no cover - recorded in raw output.
    ca = None

NAME = "vehicle_true_variable_horizon_v34n_nonconverged_objective_contract_probe_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34n_nonconverged_objective_contract_probe.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34N_NONCONVERGED_OBJECTIVE_CONTRACT_PROBE_{STAMP}.json"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
PLAN_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
OPUS_LATEST = ROOT / "docs/bohn2021_takeover/opus_lead/LATEST.md"
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T111341Z_dd0c10.md"
OPUS_REPORT_SHA = "c2c7fa3a1c5b4e3e01d28dfeb4c92f36e54ca6b68926a0375a36f7490347c56a"
OPUS_REQUEST = "execution-result:20260930T111305_5a8be004"
V34M_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34m_residual_attribution_v0_20260930T111306Z/completed.json"
V34M_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34m_residual_attribution_v0_20260930T111306Z/raw.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
REQUEST_ID = f"v34n-a13c3-nonconverged-objective-contract-{STAMP}"
MARKER = f"vehicle-v34n-a13c3-nonconverged-objective-contract-{STAMP}"

SOLVE_CAP = 6
FORMULA_SPEC = {
    "candidate_id": "stage=x0_then_last_node|term=author_last_node|eps=False|r=False|discount=n_horizon_parameter_or_H|z=same_k_last",
    "stage_rule": "x0_then_last_node",
    "terminal_rule": "author_last_node",
    "include_eps": False,
    "include_rterm": False,
    "discount_rule": "n_horizon_parameter_or_H",
    "z_rule": "same_k_last",
}
FORCED_NLPSOL_OPTIONS = {
    "ipopt.max_iter": 1,
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "print_time": False,
}
CELLS = [
    {"context_id": "source242_slot0_branch_start", "horizon": 12, "terminal_mode": "V15_shared", "initialization": "canonical", "cell_role": "canonical_H12"},
    {"context_id": "source242_slot0_branch_start", "horizon": 15, "terminal_mode": "V15_shared", "initialization": "canonical", "cell_role": "canonical_H15"},
    {"context_id": "source242_slot0_branch_start", "horizon": 35, "terminal_mode": "V15_shared", "initialization": "canonical", "cell_role": "canonical_H35"},
    {"context_id": "source242_slot0_branch_start", "horizon": 15, "terminal_mode": "V15_shared", "initialization": "goal_facing", "cell_role": "alias_separation_off_initialization_H15"},
]

_CAPTURES: List[Dict[str, Any]] = []
_PATCHED_MODULES: List[str] = []
_CURRENT_ARM_ID: Optional[str] = None
_TOTAL_SOLVER_CALLS = 0


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(v: Any) -> Any:
    if isinstance(v, Path):
        return rel(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Mapping):
        return {str(k): clean(val) for k, val in v.items() if k != "nlp_expr"}
    if isinstance(v, (list, tuple, set)):
        return [clean(x) for x in v]
    if hasattr(v, "tolist"):
        return clean(v.tolist())
    if hasattr(v, "item"):
        return clean(v.item())
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def arr(value: Any) -> np.ndarray:
    try:
        if hasattr(value, "full"):
            return np.asarray(value.full(), dtype=float).reshape(-1)
        if hasattr(value, "cat"):
            return np.asarray(value.cat, dtype=float).reshape(-1)
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def arr_hash(value: Any) -> Optional[str]:
    a = arr(value)
    if a.size == 0:
        return None
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        a = arr(value)
        if a.size != 1:
            return default
        out = float(a[0])
        return out if math.isfinite(out) else default
    except Exception:
        return default


def hash_existing(paths: Iterable[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        try:
            if p.exists() and p.is_file():
                out[rel(p)] = sha256(p)
        except Exception:
            pass
    return out


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def verify_gates(args: argparse.Namespace) -> Dict[str, Any]:
    for p in [PLAN_READY, OPUS_REPORT, V34M_DONE, V34M_RAW]:
        if not p.exists():
            raise ContractError("missing prerequisite " + rel(p))
    ready = read_json(PLAN_READY)
    if ready.get("request_id") != OPUS_REQUEST or ready.get("report_sha256") != OPUS_REPORT_SHA:
        raise ContractError("active Opus PLAN_READY does not match A13c-3 plan")
    if sha256(OPUS_REPORT) != OPUS_REPORT_SHA:
        raise ContractError("active Opus report sha mismatch")
    latest_text = OPUS_LATEST.read_text(encoding="utf-8", errors="replace") if OPUS_LATEST.exists() else ""
    if rel(OPUS_REPORT) not in latest_text:
        raise ContractError("Opus LATEST.md does not point to active report")
    done = read_json(V34M_DONE)
    if done.get("hard_pass") is not True:
        raise ContractError("v34m predecessor did not hard-pass")
    b = done.get("budget_actual") or {}
    for key in ["solver_calls", "plant_steps", "env_reset_calls_after_construction", "env_step_calls_after_construction", "new_training_or_gradient_steps", "selector_refits", "validation64_episodes", "sealed_test_episodes"]:
        if int(b.get(key, 0)) != 0:
            raise ContractError("v34m predecessor budget was not zero for " + key)
    return {
        "active_lead_plan_ready": rel(PLAN_READY),
        "active_lead_report": rel(OPUS_REPORT),
        "active_lead_report_sha256": OPUS_REPORT_SHA,
        "active_lead_request": OPUS_REQUEST,
        "v34m_completed": rel(V34M_DONE),
        "v34m_raw": rel(V34M_RAW),
        "v34m_headline": done.get("headline"),
        "supervisor_backup_receipt_for_input_gate": {
            "time": args.backup_time,
            "commit": args.backup_commit,
            "package_sha256": args.backup_package_sha256,
            "package_bytes": int(args.backup_package_bytes),
            "status": "verified_from_supervisor_context_not_revalidated_by_script",
        },
        "authorized_budget": {"low_level_solver_attempt_cap": SOLVE_CAP, "plant_steps": 0, "env_step_calls_after_construction": 0, "validation64_episodes": 0, "sealed_test_episodes": 0, "training_or_refit": 0},
    }


def patch_nlpsol_constructor() -> None:
    """Patch do-mpc optimizer.nlpsol before fresh MPC construction.

    This both forces the predeclared low-iteration IPOPT setting and captures the
    exact symbolic NLP f/g/x/p used to build the solver, closing the earlier T1c
    instrumentation gap when available.
    """
    try:
        import do_mpc.optimizer as optimizer_mod  # type: ignore
    except Exception as exc:
        raise ContractError("could not import do_mpc.optimizer for nlpsol patch: " + repr(exc))
    if getattr(optimizer_mod, "_v34n_nlpsol_patched", False):
        return
    orig = getattr(optimizer_mod, "nlpsol", None)
    if orig is None:
        raise ContractError("do_mpc.optimizer lacks nlpsol global")

    def wrapped_nlpsol(name: Any, solver: Any, nlp: Mapping[str, Any], opts: Optional[Mapping[str, Any]] = None) -> Any:
        opts_in = dict(opts or {})
        opts_out = dict(opts_in)
        opts_out.update(FORCED_NLPSOL_OPTIONS)
        cap = {
            "arm_id": _CURRENT_ARM_ID,
            "solver_name": str(name),
            "solver_kind": str(solver),
            "opts_in_keys": sorted(str(k) for k in opts_in.keys()),
            "forced_options": dict(FORCED_NLPSOL_OPTIONS),
            "opts_out_subset": {k: opts_out.get(k) for k in sorted(FORCED_NLPSOL_OPTIONS.keys())},
            "nlp_meta": {k: str(getattr(v, "shape", None)) for k, v in (nlp or {}).items() if k in ("x", "f", "g", "p")},
            "nlp_expr": nlp,
        }
        _CAPTURES.append(cap)
        return orig(name, solver, nlp, opts_out)

    setattr(optimizer_mod, "_v34n_original_nlpsol", orig)
    setattr(optimizer_mod, "nlpsol", wrapped_nlpsol)
    setattr(optimizer_mod, "_v34n_nlpsol_patched", True)
    _PATCHED_MODULES.append(getattr(optimizer_mod, "__name__", "do_mpc.optimizer"))


def cat_arg(obj: Any) -> Any:
    if hasattr(obj, "cat"):
        return obj.cat
    if hasattr(obj, "master"):
        return obj.master
    return np.asarray(arr(obj), dtype=float).reshape((-1, 1))


def direct_nlp_eval(mpc: Any, capture: Optional[Mapping[str, Any]], idx: int) -> Dict[str, Any]:
    if ca is None:
        return {"available": False, "reason": "casadi_import_failed"}
    if not capture or not isinstance(capture.get("nlp_expr"), Mapping):
        return {"available": False, "reason": "no_captured_nlp_expr"}
    nlp = capture["nlp_expr"]
    try:
        x_sym = nlp.get("x")
        p_sym = nlp.get("p")
        f_expr = nlp.get("f")
        g_expr = nlp.get("g")
        f_fn = ca.Function(f"v34n_f_{idx}", [x_sym, p_sym], [f_expr])
        g_fn = ca.Function(f"v34n_g_{idx}", [x_sym, p_sym], [g_expr])
        x_val = cat_arg(mpc.opt_x_num)
        p_val = cat_arg(mpc.opt_p_num)
        f_val = finite_float(f_fn(x_val, p_val), None)
        g_val = arr(g_fn(x_val, p_val))
        saved_g = arr(getattr(mpc, "opt_g_num", []))
        solver = finite_float(getattr(mpc, "opt_f_num", None), None)
        f_abs = None if f_val is None or solver is None else float(abs(f_val - solver))
        return {
            "available": True,
            "direct_f_value": f_val,
            "solver_objective": solver,
            "direct_f_abs_error_vs_solver": f_abs,
            "direct_f_rel_error_vs_solver": None if f_abs is None or solver is None else float(f_abs / max(1.0, abs(solver))),
            "g_eval_size": int(g_val.size),
            "saved_opt_g_size": int(saved_g.size),
            "g_eval_minus_saved_opt_g_linf": None if g_val.size == 0 or g_val.size != saved_g.size else float(np.max(np.abs(g_val - saved_g))),
            "captured_nlp_meta": clean(capture.get("nlp_meta")),
        }
    except Exception as exc:
        return {"available": False, "reason": "direct_eval_failed", "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-6:], "captured_nlp_meta": clean(capture.get("nlp_meta"))}


def formula_try(mpc: Any, h: int, include_eps: bool = False, include_r: bool = False) -> Dict[str, Any]:
    try:
        acc = kacc.LabelAccessor(mpc)
        comp = kacc.formula_components(
            mpc, acc, int(h), FORMULA_SPEC["stage_rule"], FORMULA_SPEC["terminal_rule"],
            bool(include_eps), bool(include_r), FORMULA_SPEC["discount_rule"], FORMULA_SPEC["z_rule"],
        )
        comp["available"] = True
        comp["candidate_id"] = (
            f"stage={FORMULA_SPEC['stage_rule']}|term={FORMULA_SPEC['terminal_rule']}|eps={bool(include_eps)}|"
            f"r={bool(include_r)}|discount={FORMULA_SPEC['discount_rule']}|z={FORMULA_SPEC['z_rule']}"
        )
        return comp
    except Exception as exc:
        return {"available": False, "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-6:], "include_eps": bool(include_eps), "include_rterm": bool(include_r)}


def residual_capture(mpc: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    names = {
        "bounds_repaired_lb_opt_x_ub_opt_x": ("opt_x_num", "lb_opt_x", "ub_opt_x"),
        "bounds_legacy_opt_x_lb_ub": ("opt_x_num", "opt_x_lb", "opt_x_ub"),
        "constraints_repaired_cons_lb_cons_ub": ("opt_g_num", "cons_lb", "cons_ub"),
        "constraints_legacy_opt_g_lb_ub": ("opt_g_num", "opt_g_lb", "opt_g_ub"),
    }
    for key, (val_name, lo_name, hi_name) in names.items():
        exists = {n: hasattr(mpc, n) for n in [val_name, lo_name, hi_name]}
        res = None
        if all(exists.values()):
            res = base.residual_vs_bounds(getattr(mpc, val_name), getattr(mpc, lo_name), getattr(mpc, hi_name))
        out[key] = {"exists": exists, "residual": res, "status": "missing_or_size_mismatch" if res is None else ("ok" if float(res) <= base.ACCEPT_RESIDUAL_TOL else "exceeded")}
    return out


def solve_cell(row: Mapping[str, Any], context: Mapping[str, Any], terminals: Mapping[int, Tuple[Any, Any]], idx: int) -> Dict[str, Any]:
    global _CURRENT_ARM_ID, _TOTAL_SOLVER_CALLS
    h = int(row["horizon"])
    arm_id = f"ctx={row['context_id']}|H{h}|term={row['terminal_mode']}|init={row['initialization']}|role={row['cell_role']}"
    _CURRENT_ARM_ID = arm_id
    cap_before = len(_CAPTURES)
    terminal, term_h, term_note = base.terminal_for_mode(h, str(row["terminal_mode"]), terminals)
    result: Dict[str, Any] = {
        "arm_id": arm_id,
        "execution_index": idx,
        "context_id": row["context_id"],
        "horizon": h,
        "terminal_mode": row["terminal_mode"],
        "terminal_source_horizon": int(term_h),
        "terminal_note": term_note,
        "initialization": row["initialization"],
        "cell_role": row["cell_role"],
        "plant_steps": 0,
        "env_reset_calls": 0,
        "forced_nlpsol_options": dict(FORCED_NLPSOL_OPTIONS),
        "solve_events": [],
        "prevented_second_solver_call": False,
    }
    try:
        env = base.create_env(h, terminal)
        ctrl = env.control_system.controller
        mpc = ctrl.mpc
        result["captured_nlpsol_count_during_construction"] = len(_CAPTURES) - cap_before
        result["captured_nlpsol_meta"] = [clean({k: v for k, v in c.items() if k != "nlp_expr"}) for c in _CAPTURES[cap_before:]]
        context_meta = base.configure_context_no_reset(env, context, h)
        init_meta = base.set_initial_guess(mpc, env, context_meta, str(row["initialization"]), h)
        result["context_meta"] = {k: v for k, v in context_meta.items() if k != "shifted_tvp"}
        result["initialization_meta"] = init_meta
        original_solve = mpc.solve
        solve_count = 0
        def counted_solve(*args: Any, **kwargs: Any) -> Any:
            nonlocal solve_count
            global _TOTAL_SOLVER_CALLS
            if solve_count >= 1:
                result["prevented_second_solver_call"] = True
                raise ContractError("v34n prevented a controller retry before a second low-level solver call")
            if _TOTAL_SOLVER_CALLS >= SOLVE_CAP:
                raise ContractError("v34n global low-level solver call cap exhausted")
            solve_count += 1
            _TOTAL_SOLVER_CALLS += 1
            pre = base.capture_mpc_state(mpc)
            t0 = time.perf_counter()
            try:
                ret = original_solve(*args, **kwargs)
                exc = None
            except Exception as e:
                ret = None
                exc = repr(e)
            elapsed = time.perf_counter() - t0
            post = base.capture_mpc_state(mpc)
            stats = copy.deepcopy(getattr(mpc, "solver_stats", {}))
            event = {
                "pre": pre,
                "post": post,
                "solver_wall_s": float(elapsed),
                "solver_exception": exc,
                "solver_stats": stats,
                "return_status": stats.get("return_status"),
                "success": bool(stats.get("success", False)),
                "iterations": stats.get("iter_count") or stats.get("iterations"),
                "objective_opt_f_num": finite_float(getattr(mpc, "opt_f_num", None), None),
            }
            result["solve_events"].append(event)
            if exc is not None:
                raise ContractError("mpc.solve failed: " + exc)
            return ret
        mpc.solve = counted_solve
        t_action = time.perf_counter()
        try:
            action = ctrl.get_action(base.state_clean(context["state"]), h, tvp_values=copy.deepcopy(context_meta["shifted_tvp"]))
            result["controller_get_action_exception"] = None
            result["first_control"] = clean(action)
        except Exception as exc:
            result["controller_get_action_exception"] = repr(exc)
            result["controller_get_action_traceback_tail"] = traceback.format_exc().splitlines()[-8:]
            result["first_control"] = None
        result["controller_get_action_wall_s"] = float(time.perf_counter() - t_action)
        result["solve_calls"] = int(solve_count)
        last_event = result["solve_events"][-1] if result["solve_events"] else {}
        result["solver_event_summary"] = {k: last_event.get(k) for k in ["return_status", "success", "iterations", "objective_opt_f_num", "solver_exception", "solver_wall_s"]}
        result["forced_nonconverged_or_near_offoptimal"] = bool(last_event and (last_event.get("success") is not True or str(last_event.get("return_status")) != "Solve_Succeeded"))
        main = formula_try(mpc, h, False, False)
        eps = formula_try(mpc, h, True, False)
        rterm = formula_try(mpc, h, False, True)
        both = formula_try(mpc, h, True, True)
        result["objective_reconstruction"] = main
        result["alias_branch_variants"] = {"eps_only": eps, "rterm_only": rterm, "eps_and_rterm": both}
        mt = main.get("total") if main.get("available") else None
        result["alias_separation"] = {
            "eps_delta_vs_main": None if mt is None or not eps.get("available") else float(eps["total"] - mt),
            "rterm_delta_vs_main": None if mt is None or not rterm.get("available") else float(rterm["total"] - mt),
            "both_delta_vs_main": None if mt is None or not both.get("available") else float(both["total"] - mt),
            "eps_nonzero_count": eps.get("eps_nonzero_count"),
            "rterm_nonzero_count": rterm.get("rterm_nonzero_count"),
        }
        captures = [c for c in _CAPTURES[cap_before:] if c.get("arm_id") == arm_id]
        capture = captures[-1] if captures else (_CAPTURES[-1] if len(_CAPTURES) > cap_before else None)
        result["direct_nlp_eval"] = direct_nlp_eval(mpc, capture, idx)
        result["residual_capture"] = residual_capture(mpc)
        if main.get("available") and main.get("solver_objective") is not None:
            resid = float(main["total"] - main["solver_objective"])
            term = main.get("terminal_discounted")
            result["predeclared_discriminants"] = {
                "residual_total_minus_solver": resid,
                "abs_residual": abs(resid),
                "relative_error": main.get("relative_error"),
                "terminal_discounted": term,
                "residual_over_terminal_discounted": None if term is None or abs(float(term)) < 1e-300 else float(resid / float(term)),
                "abs_residual_over_abs_terminal_discounted": None if term is None or abs(float(term)) < 1e-300 else float(abs(resid) / abs(float(term))),
                "stage_lterm_total": main.get("stage_lterm_total"),
                "terminal_fraction_of_reconstructed_total": None if mt is None or abs(float(mt)) < 1e-300 or term is None else float(float(term) / float(mt)),
            }
        else:
            result["predeclared_discriminants"] = {"available": False}
    except Exception as exc:
        result["cell_exception"] = repr(exc)
        result["cell_traceback_tail"] = traceback.format_exc().splitlines()[-10:]
        result["solve_calls"] = int(len(result.get("solve_events") or []))
    finally:
        _CURRENT_ARM_ID = None
    return result


def analyze(arms: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    hard_defects: List[str] = []
    pass_cells: List[bool] = []
    for arm in arms:
        rec = arm.get("objective_reconstruction") or {}
        relerr = rec.get("relative_error")
        h = int(arm.get("horizon", -1))
        role = str(arm.get("cell_role"))
        allowed = 1e-6
        if h == 35 and arm.get("forced_nonconverged_or_near_offoptimal"):
            allowed = 1e-5
        ok = bool(relerr is not None and float(relerr) <= allowed)
        if relerr is not None and float(relerr) > 1e-4:
            hard_defects.append(str(arm.get("arm_id")))
        pass_cells.append(ok)
        disc = arm.get("predeclared_discriminants") or {}
        rows.append({
            "arm_id": arm.get("arm_id"),
            "role": role,
            "horizon": h,
            "initialization": arm.get("initialization"),
            "solve_calls": arm.get("solve_calls"),
            "return_status": (arm.get("solver_event_summary") or {}).get("return_status"),
            "success": (arm.get("solver_event_summary") or {}).get("success"),
            "forced_nonconverged_or_near_offoptimal": arm.get("forced_nonconverged_or_near_offoptimal"),
            "relative_error": relerr,
            "allowed_relative_error": allowed,
            "G2_cell_pass": ok,
            "residual_over_terminal_discounted": disc.get("residual_over_terminal_discounted"),
            "abs_residual_over_abs_terminal_discounted": disc.get("abs_residual_over_abs_terminal_discounted"),
            "abs_residual": disc.get("abs_residual"),
            "terminal_discounted": disc.get("terminal_discounted"),
            "direct_f_rel_error_vs_solver": (arm.get("direct_nlp_eval") or {}).get("direct_f_rel_error_vs_solver"),
            "direct_g_linf_vs_saved": (arm.get("direct_nlp_eval") or {}).get("g_eval_minus_saved_opt_g_linf"),
            "eps_delta_vs_main": (arm.get("alias_separation") or {}).get("eps_delta_vs_main"),
            "rterm_delta_vs_main": (arm.get("alias_separation") or {}).get("rterm_delta_vs_main"),
            "both_delta_vs_main": (arm.get("alias_separation") or {}).get("both_delta_vs_main"),
        })
    ratios = [r["abs_residual_over_abs_terminal_discounted"] for r in rows if r.get("abs_residual_over_abs_terminal_discounted") is not None and str(r.get("initialization")) == "canonical"]
    direct_available = sum(1 for r in rows if r.get("direct_f_rel_error_vs_solver") is not None)
    alias_active = any((r.get("eps_delta_vs_main") is not None and abs(float(r["eps_delta_vs_main"])) > 1e-12) or (r.get("rterm_delta_vs_main") is not None and abs(float(r["rterm_delta_vs_main"])) > 1e-12) for r in rows)
    return {
        "per_cell": rows,
        "headline": {
            "G2_pass": bool(len(rows) == len(CELLS) and all(pass_cells) and not hard_defects),
            "G2_hard_objective_contract_defect": bool(hard_defects),
            "hard_defect_arm_ids_rel_error_gt_1e-4": hard_defects,
            "cells_completed": len(arms),
            "new_solver_calls": int(sum(int(a.get("solve_calls", 0)) for a in arms)),
            "low_level_solver_call_cap": SOLVE_CAP,
            "forced_nonconverged_or_near_offoptimal_cells": int(sum(1 for a in arms if a.get("forced_nonconverged_or_near_offoptimal"))),
            "direct_nlp_eval_available_cells": direct_available,
            "alias_branch_separation_observed": alias_active,
            "canonical_abs_residual_over_terminal_min": None if not ratios else float(min(ratios)),
            "canonical_abs_residual_over_terminal_max": None if not ratios else float(max(ratios)),
            "canonical_abs_residual_over_terminal_span": None if len(ratios) < 2 else float(max(ratios) - min(ratios)),
            "plant_steps": 0,
            "env_step_calls_after_construction": 0,
            "training_or_refit": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
    }


def write_csv_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["arm_id", "role", "horizon", "initialization", "solve_calls", "return_status", "success", "forced_nonconverged_or_near_offoptimal", "relative_error", "allowed_relative_error", "G2_cell_pass", "abs_residual", "terminal_discounted", "residual_over_terminal_discounted", "abs_residual_over_abs_terminal_discounted", "direct_f_rel_error_vs_solver", "direct_g_linf_vs_saved", "eps_delta_vs_main", "rterm_delta_vs_main", "both_delta_vs_main"]
    with (RUN_DIR / "cell_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: clean(row.get(k)) for k in fields})


def write_outputs(raw: Mapping[str, Any]) -> None:
    h = raw["analysis"]["headline"]
    write_csv_summary(raw["analysis"]["per_cell"])
    lines = [
        "# v34n / A13c-3 non-converged objective-contract probe",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only solver-contract diagnostic under active Opus plan `{rel(OPUS_REPORT)}`.",
        "",
        "## Budget",
        f"- New low-level solver calls: `{h['new_solver_calls']}` / `{SOLVE_CAP}`.",
        "- Plant steps: `0`; env.step after construction: `0`; training/refit: `0`; validation64: `0`; sealed test: `0`.",
        "",
        "## Headline",
        f"- G2 pass: `{h['G2_pass']}`; hard defect (>1e-4 rel error): `{h['G2_hard_objective_contract_defect']}`.",
        f"- Forced non-converged/near-off-optimal cells: `{h['forced_nonconverged_or_near_offoptimal_cells']}` / `{h['cells_completed']}`.",
        f"- Direct NLP f/g available cells: `{h['direct_nlp_eval_available_cells']}` / `{h['cells_completed']}`.",
        f"- Alias branch separation observed: `{h['alias_branch_separation_observed']}`.",
        f"- Canonical |residual|/|terminal| range: `{h['canonical_abs_residual_over_terminal_min']}` to `{h['canonical_abs_residual_over_terminal_max']}` (span `{h['canonical_abs_residual_over_terminal_span']}`).",
        "",
        "## Per-cell metrics",
        "",
        "| role | H | init | status | rel err | allowed | G2 cell | |resid|/|terminal| | direct f rel | eps/r deltas |",
        "|---|---:|---|---|---:|---:|---|---:|---:|---|",
    ]
    for r in raw["analysis"]["per_cell"]:
        lines.append("| `%s` | %s | `%s` | `%s` | %s | %s | `%s` | %s | %s | `%s/%s` |" % (
            r.get("role"), r.get("horizon"), r.get("initialization"), r.get("return_status"),
            r.get("relative_error"), r.get("allowed_relative_error"), r.get("G2_cell_pass"),
            r.get("abs_residual_over_abs_terminal_discounted"), r.get("direct_f_rel_error_vs_solver"),
            r.get("eps_delta_vs_main"), r.get("rterm_delta_vs_main"),
        ))
    lines += [
        "",
        "This is not population validation, not selector training, and not final-test evidence. It only gates whether the repaired objective accessor/formula can be used for the remaining Task-C objective-vs-basin probe.",
        f"Raw: `{rel(RUN_DIR/'raw.json')}`; completed: `{rel(RUN_DIR/'completed.json')}`; CSV: `{rel(RUN_DIR/'cell_metrics.csv')}`; backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    block = f"""
<!-- {MARKER} -->
## v34n/A13c-3 non-converged objective-contract probe

UTC: {raw['created_utc']}. Ran active-Opus A13c-3 as a bounded development-only solver-contract diagnostic: new low-level solver calls={h['new_solver_calls']}/{SOLVE_CAP}, plant_steps=0, env_step_after_construction=0, training/refit=0, validation64=0, sealed_test=0. G2_pass={h['G2_pass']}; hard_objective_contract_defect={h['G2_hard_objective_contract_defect']}; forced_nonconverged_or_near_offoptimal_cells={h['forced_nonconverged_or_near_offoptimal_cells']}/{h['cells_completed']}; direct_nlp_eval_available_cells={h['direct_nlp_eval_available_cells']}; alias_branch_separation_observed={h['alias_branch_separation_observed']}; canonical_abs_residual_over_terminal_range=[{h['canonical_abs_residual_over_terminal_min']}, {h['canonical_abs_residual_over_terminal_max']}]. Evidence: `{rel(RUN_DIR/'summary.md')}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'completed.json')}`, `{rel(RUN_DIR/'cell_metrics.csv')}`. Backup request: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)
    write_json(NEXT_REVIEW_REQUEST, {
        "request_id": REQUEST_ID,
        "created": raw["created_utc"],
        "status": "gate_evidence_ready" if h["G2_pass"] else "analysis_requested",
        "trigger": "v34n/A13c-3 non-converged objective-contract probe completed",
        "experiment_id": NAME,
        "active_lead_report": rel(OPUS_REPORT),
        "active_lead_report_sha256": OPUS_REPORT_SHA,
        "question": "Review A13c-3 G2 evidence. If G2_pass is accepted, the active plan authorizes the remaining Task-C objective-vs-basin probe (23 remaining solver calls, v34g one cell already spent/imported) after external backup. If not, classify the exact objective-contract or instrumentation defect; do not use validation64 or sealed test.",
        "evidence_paths": [rel(RUN_DIR/"summary.md"), rel(RUN_DIR/"raw.json"), rel(RUN_DIR/"completed.json"), rel(RUN_DIR/"cell_metrics.csv"), rel(V34M_RAW), rel(OPUS_REPORT), rel(RESPONSE_LOG)],
        "budget_actual": raw["budget_actual"],
        "headline": h,
        "backup_required_before_more_unique_science": rel(BACKUP_REQUEST),
    })
    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v34n_nonconverged_objective_contract_probe",
        "created_utc": raw["created_utc"],
        "backup_required_before_more_unique_science": True,
        "reason": "new A13c-3 solver-contract source/output/docs must be externally recoverable before remaining Task-C 23-call probe or further unique science",
        "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(NEXT_REVIEW_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
        "new_solver_calls": h["new_solver_calls"],
        "new_plant_steps": 0,
        "env_step_calls_after_construction": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_gate": "If active lead accepts G2 and backup is verified, run corrected v34j/v34o remaining Task-C objective-vs-basin probe; otherwise preserve defect evidence for lead analysis.",
    })
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        "# Continue state after v34n/A13c-3 non-converged objective-contract probe\n\n"
        + f"UTC: {raw['created_utc']}\n\nHeadline: {json.dumps(clean(h), sort_keys=True)}\n\n"
        + f"Artifacts: {rel(RUN_DIR/'summary.md')}, {rel(RUN_DIR/'raw.json')}, {rel(RUN_DIR/'completed.json')}, {rel(RUN_DIR/'cell_metrics.csv')}\n\n"
        + f"Next: verify external backup for {rel(BACKUP_REQUEST)}. If Opus accepts G2_pass, run remaining Task-C objective-vs-basin probe with 23 remaining solver calls; if G2 failed, return raw defect evidence to Opus and do not spend Task-C calls.\n",
        encoding="utf-8",
    )
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([raw["created_utc"], NAME, raw["classification"], "FORCED_IPOPT_MAX_ITER=1; cells=4; cap=6", "opened_development_solver_contract_nonconverged_no_validation_no_test", h["cells_completed"], 0, h["new_solver_calls"], 0, 0, False, rel(RUN_DIR/"completed.json"), MARKER])


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-time", required=True)
    ap.add_argument("--backup-commit", required=True)
    ap.add_argument("--backup-package-sha256", required=True)
    ap.add_argument("--backup-package-bytes", type=int, required=True)
    ap.add_argument("--i-accept-v34n-a13c3-low-solve-budget", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        started = now_utc()
        gates = verify_gates(args)
        write_json(RUN_DIR / "run_started.json", {"started_utc": started.isoformat(), "pid": os.getpid(), "method": NAME, "gates": gates, "cells": CELLS, "solve_cap": SOLVE_CAP, "plant_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
        _, stage1_runner, _ = base.v1d.import_legacy_modules()
        patch_nlpsol_constructor()
        preflight = stage1_runner.runtime_preflight()
        if not preflight.get("passed"):
            raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
        stage1_runner.base.v1.latency_verify()
        term_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
        terminals, terminal_receipts = stage1_runner.load_terminal_grid(term_protocol["terminal_grid_readiness_reused_from_v1"])
        if 15 not in terminals:
            raise ContractError("terminal grid missing H15")
        contexts = base.load_contexts()
        matches = [c for c in contexts if str(c.get("context_id")) == "source242_slot0_branch_start"]
        if len(matches) != 1:
            raise ContractError("could not uniquely identify source242 context")
        context = matches[0]
        arms: List[Dict[str, Any]] = []
        for idx, row in enumerate(CELLS):
            if _TOTAL_SOLVER_CALLS >= SOLVE_CAP:
                raise ContractError("solver cap exhausted before scheduled cells completed")
            arm = solve_cell(row, context, terminals, idx)
            arms.append(arm)
            progress = {"arms_done": len(arms), "arms_expected": len(CELLS), "new_solver_calls": _TOTAL_SOLVER_CALLS, "last_arm": {k: arm.get(k) for k in ["arm_id", "horizon", "initialization", "solve_calls", "forced_nonconverged_or_near_offoptimal"]}, "validation64_bank_opened": False, "sealed_test_accessed": False}
            write_json(RUN_DIR / "progress.json", progress)
            print(json.dumps(clean(progress), sort_keys=True), flush=True)
        if _TOTAL_SOLVER_CALLS > SOLVE_CAP:
            raise ContractError("new solver call cap exceeded")
        analysis = analyze(arms)
        created = now_utc()
        raw = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_a13c3_nonconverged_objective_contract_probe_not_validation_not_test",
            "active_lead": "claude-opus-5-5",
            "lead_report": rel(OPUS_REPORT),
            "lead_report_sha256": OPUS_REPORT_SHA,
            "hypothesis_frozen": "Under forced max_iter=1 non-converged/near-off-optimal low-level solves on source242/V15_shared, the v34k frozen objective reconstruction should maintain predeclared relative error <=1e-6 (H35 allowance <=1e-5 if nonconverged), and residual/terminal ratios plus alias-branch variants distinguish H5/H2/H3/H4 before remaining Task-C calls.",
            "gates": gates,
            "forced_nlpsol_options": dict(FORCED_NLPSOL_OPTIONS),
            "patched_modules": list(_PATCHED_MODULES),
            "runtime_preflight": preflight,
            "terminal_receipts": {str(k): v for k, v in terminal_receipts.items()},
            "selected_context_no_case_snapshot": {k: v for k, v in context.items() if k != "case_snapshot"},
            "cells": CELLS,
            "arms": arms,
            "analysis": analysis,
            "budget_declared": {"low_level_solver_call_cap": SOLVE_CAP, "scheduled_cells": len(CELLS), "plant_steps": 0, "env_step_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "budget_actual": {"scheduled_cells_completed": len(arms), "new_solver_calls": int(_TOTAL_SOLVER_CALLS), "plant_steps": 0, "env_step_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "input_hashes": hash_existing([Path(__file__).resolve(), PLAN_READY, OPUS_LATEST, OPUS_REPORT, V34M_DONE, V34M_RAW, ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py", ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0.py"]),
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid(), "casadi_imported": ca is not None},
            "interpretation_limits": ["opened development source242 cell only", "solver-contract diagnostic only", "not validation64", "not sealed/final test", "no plant rollout", "no adaptive selector closed-loop claim", "forced low-iteration solver options differ from controller deployment"],
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_outputs(raw)
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE, BACKUP_REQUEST, NEXT_REVIEW_REQUEST, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed = {
            "status": "complete",
            "passed": True,
            "hard_pass": bool(analysis["headline"].get("G2_pass")),
            "created_utc": created.isoformat(),
            "classification": raw["classification"],
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_actual": raw["budget_actual"],
            "headline": analysis["headline"],
            "summary": rel(RUN_DIR / "summary.md"),
            "raw": rel(RUN_DIR / "raw.json"),
            "cell_metrics_csv": rel(RUN_DIR / "cell_metrics.csv"),
            "backup_request": rel(BACKUP_REQUEST),
            "next_review_request_id": REQUEST_ID,
            "hashes": hash_existing(files),
        }
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({"completed": rel(RUN_DIR/"completed.json"), "summary": rel(RUN_DIR/"summary.md"), "raw": rel(RUN_DIR/"raw.json"), "cell_metrics_csv": rel(RUN_DIR/"cell_metrics.csv"), "headline": completed["headline"], "backup_request": rel(BACKUP_REQUEST)}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        fail = {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_a13c3_nonconverged_objective_contract_probe_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_caps": {"low_level_solver_call_cap": SOLVE_CAP, "plant_steps": 0, "env_step_calls_after_construction": 0}, "new_solver_calls_recorded": _TOTAL_SOLVER_CALLS}
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(f"# v34n/A13c-3 failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR/'failed.json')}\n\nRecorded solver calls before failure: {_TOTAL_SOLVER_CALLS}. No plant, validation64, sealed-test, training or refit access was requested. Preserve failure and repair exact operational defect before retry; do not spend Task-C calls if G2 is absent.\n", encoding="utf-8")
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR/"failed.json"), "new_solver_calls_recorded": _TOTAL_SOLVER_CALLS, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
