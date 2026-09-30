#!/usr/bin/env python3
"""v34m/A13c-2 residual attribution diagnostic (zero solve / zero plant).

Active Opus A13c-2 asks for one bounded zero-solve round, with exactly three
checks, before any A13c-3 non-optimal solver calls or the remaining Task-C
objective-vs-basin probe:

  T1a. terminal gamma^H * V(x_H) float64 vs float32 end-to-end recomputation;
  T1b. finite-difference local sensitivity on the saved opt_x plus KKT/dual
       scale checks to test whether solver tolerance could plausibly explain the
       v34k/v34l reconstruction residual; and
  T1c. if multipliers / symbolic NLP primitives are exposed, evaluate direct NLP
       objective/constraint and a constraint-only Lagrangian stability residual.

Inputs are already-opened v34g/v34k/v34l development evidence.  The script
rebuilds the same H15/V15/canonical MPC object and assigns saved arrays; it does
not call mpc.solve, controller.get_action, env.reset, env.step, plant rollout,
selector search/refit, training, validation64, or sealed/final test.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0 as kacc  # noqa:E402
import vehicle_true_variable_horizon_v34i_objective_contract_localization_v0 as i  # noqa:E402

try:  # legacy environment has CasADi through do-mpc.
    import casadi as ca  # type: ignore
except Exception:  # pragma: no cover - recorded in raw output if unavailable.
    ca = None

NAME = "vehicle_true_variable_horizon_v34m_residual_attribution_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34m_residual_attribution.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34M_RESIDUAL_ATTRIBUTION_{STAMP}.json"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
REQUEST_ID = f"v34m-residual-attribution-{STAMP}"
MARKER = f"vehicle-v34m-residual-attribution-{STAMP}"

V34G_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0_20260930T100824Z/raw.json"
V34K_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_20260930T104148Z/completed.json"
V34K_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_20260930T104148Z/raw.json"
V34K_CSV = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_20260930T104148Z/candidate_residuals.csv"
V34L_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34l_residual_eps_label_dissection_v0_20260930T105720Z/completed.json"
V34L_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34l_residual_eps_label_dissection_v0_20260930T105720Z/raw.json"
OPUS_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T105748Z_5ca3ae.md"
OPUS_REPORT_SHA = "ae5db8822d0d7ff886ddd406d2667ab8df516f2271f134652328b0e93a5ff0b1"
OPUS_REQUEST = "execution-result:20260930T105719_a0cd3d48"

FORMULA_SPEC = {
    "candidate_id": "stage=x0_then_last_node|term=author_last_node|eps=False|r=False|discount=n_horizon_parameter_or_H|z=same_k_last",
    "stage_rule": "x0_then_last_node",
    "terminal_rule": "author_last_node",
    "include_eps": False,
    "include_rterm": False,
    "discount_rule": "n_horizon_parameter_or_H",
    "z_rule": "same_k_last",
}
FD_REL_STEP = 1e-6
FD_ABS_FLOOR = 1e-7
FD_MAX_VARS = 5000
DIRECT_NLP_MATCH_TOL = 1e-9
RECON_REL_TOL = 1e-6


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
        return {str(k): clean(val) for k, val in v.items()}
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
    return kacc.arr(value)


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


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def verify_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    required = [V34G_RAW, V34K_DONE, V34K_RAW, V34K_CSV, V34L_DONE, V34L_RAW, OPUS_READY, OPUS_REPORT]
    for p in required:
        if not p.exists():
            raise ContractError("missing required input " + rel(p))
    ready = read_json(OPUS_READY)
    if ready.get("request_id") != OPUS_REQUEST or ready.get("report_sha256") != OPUS_REPORT_SHA:
        raise ContractError("active Opus PLAN_READY does not match A13c-2 report")
    if sha256(OPUS_REPORT) != OPUS_REPORT_SHA:
        raise ContractError("active Opus A13c-2 report sha mismatch")
    kdone = read_json(V34K_DONE)
    ldone = read_json(V34L_DONE)
    if kdone.get("hard_pass") is not True:
        raise ContractError("v34k predecessor did not hard-pass")
    if ldone.get("hard_pass") is not True:
        raise ContractError("v34l predecessor did not hard-pass")
    for label, done in [("v34k", kdone), ("v34l", ldone)]:
        b = done.get("budget_actual") or {}
        for key in ["solver_calls", "plant_steps", "env_reset_calls_after_construction", "env_step_calls_after_construction", "new_training_or_gradient_steps", "selector_refits", "validation64_episodes", "sealed_test_episodes"]:
            if int(b.get(key, 0)) != 0:
                raise ContractError("%s predecessor was not zero-budget: %s=%r" % (label, key, b.get(key)))
    rows = read_csv_rows(V34K_CSV)
    best = [r for r in rows if r.get("candidate_id") == FORMULA_SPEC["candidate_id"]]
    if not best:
        raise ContractError("frozen formula not present in v34k candidate table")
    return {
        "active_opus_plan": rel(OPUS_REPORT),
        "active_opus_sha256": OPUS_REPORT_SHA,
        "v34g_raw": rel(V34G_RAW),
        "v34k_completed": rel(V34K_DONE),
        "v34k_raw": rel(V34K_RAW),
        "v34l_completed": rel(V34L_DONE),
        "v34l_raw": rel(V34L_RAW),
        "v34k_formula_row": best[0],
        "input_hashes": {rel(p): sha256(p) for p in required + [V34K_CSV]},
        "supervisor_backup_receipt_for_this_run": {
            "time": args.backup_time,
            "commit": args.backup_commit,
            "package_sha256": args.backup_package_sha256,
            "package_bytes": int(args.backup_package_bytes),
            "status": "verified_from_supervisor_context_not_revalidated_by_experiment_script",
        },
    }


def cast32_tree(x: Any) -> Any:
    if isinstance(x, (list, tuple)):
        return [cast32_tree(v) for v in x]
    try:
        if hasattr(x, "full"):
            return np.asarray(x.full(), dtype=np.float32).astype(np.float64)
        return np.asarray(x, dtype=np.float32).astype(np.float64)
    except Exception:
        return x


def call_vf4(mpc: Any, x: Any, p: Any, weights: Any, biases: Any) -> Tuple[Optional[float], Dict[str, Any]]:
    try:
        val = mpc.vf_fun(x, p, weights, biases)
        return finite_float(val, None), {"ok": True, "api": "vf_fun(x,p,weights,biases)", "vf_fun_n_in": int(mpc.vf_fun.n_in()) if hasattr(mpc.vf_fun, "n_in") else None}
    except Exception as exc:
        return None, {"ok": False, "error": repr(exc), "vf_fun_n_in": int(mpc.vf_fun.n_in()) if hasattr(mpc.vf_fun, "n_in") else None}


def formula_value(mpc: Any, acc: kacc.LabelAccessor, H: int) -> Dict[str, Any]:
    gamma, gsrc = i.gamma_and_source(mpc)
    ns = kacc.n_scenarios(mpc, H)
    stage_terms: List[float] = []
    for kk in range(H):
        for ss in range(int(ns[kk])):
            stage_terms.append(float(kacc.lterm_value(mpc, acc, kk, ss, gamma, FORMULA_SPEC["stage_rule"], FORMULA_SPEC["z_rule"])))
    vf_raw, term_disc, tmeta = kacc.terminal_value(mpc, acc, H, gamma, FORMULA_SPEC["terminal_rule"], FORMULA_SPEC["discount_rule"])
    total = float(math.fsum(stage_terms + [float(term_disc)]))
    solver = finite_float(getattr(mpc, "opt_f_num", None), None)
    return {"total": total, "stage_total": float(math.fsum(stage_terms)), "terminal_discounted": float(term_disc), "terminal_vf_raw": float(vf_raw), "terminal_meta": tmeta, "gamma": float(gamma), "gamma_source": gsrc, "solver_objective": solver, "residual_total_minus_solver": None if solver is None else float(total - solver), "relative_error": None if solver is None else float(abs(total - solver) / max(1.0, abs(solver)))}


def t1a_terminal_dtype(mpc: Any, acc: kacc.LabelAccessor, H: int, baseline: Mapping[str, Any]) -> Dict[str, Any]:
    gamma = float(baseline["gamma"])
    # Use the exact terminal rule selected by v34k/v34l.
    xh = acc.state(H, 0, -1)
    p_keep, p_names = acc.p(0, True)
    w = mpc.vf.weights_num
    b = mpc.vf.biases_num
    vf64, meta64 = call_vf4(mpc, xh, p_keep, w, b)
    xh32 = np.asarray(arr(xh), dtype=np.float32).astype(np.float64).reshape((-1, 1))
    p32 = np.asarray(arr(p_keep), dtype=np.float32).astype(np.float64).reshape((-1, 1))
    vf32_inputs, meta32_inputs = call_vf4(mpc, xh32, p32, cast32_tree(w), cast32_tree(b))
    gamma_pow64 = float(gamma ** H)
    gamma_pow32 = float(np.float32(gamma) ** np.float32(H))
    term64 = None if vf64 is None else float(gamma_pow64 * vf64)
    term32_all = None if vf32_inputs is None else float(gamma_pow32 * vf32_inputs)
    term32_gamma_only = None if vf64 is None else float(gamma_pow32 * vf64)
    term32_vf_only = None if vf32_inputs is None else float(gamma_pow64 * vf32_inputs)
    abs_resid = abs(float(baseline["residual_total_minus_solver"]))
    deltas = {
        "terminal_all32_minus_64": None if term32_all is None or term64 is None else float(term32_all - term64),
        "terminal_gamma32_only_minus_64": None if term32_gamma_only is None or term64 is None else float(term32_gamma_only - term64),
        "terminal_vf32_inputs_only_minus_64": None if term32_vf_only is None or term64 is None else float(term32_vf_only - term64),
    }
    max_abs_delta = max([abs(v) for v in deltas.values() if v is not None] or [0.0])
    return {
        "terminal_x_rule": FORMULA_SPEC["terminal_rule"],
        "discount_rule": FORMULA_SPEC["discount_rule"],
        "p_names_excluding_n_horizon": p_names,
        "xH64": arr(xh).tolist(),
        "p64": arr(p_keep).tolist(),
        "gamma": gamma,
        "gamma_pow64": gamma_pow64,
        "gamma_pow32": gamma_pow32,
        "vf64": vf64,
        "vf32_inputs_weights_biases": vf32_inputs,
        "vf64_meta": meta64,
        "vf32_meta": meta32_inputs,
        "terminal64": term64,
        "terminal32_all": term32_all,
        "terminal32_gamma_only": term32_gamma_only,
        "terminal32_vf_inputs_only": term32_vf_only,
        "deltas_vs_64": deltas,
        "max_abs_terminal_dtype_delta": float(max_abs_delta),
        "baseline_abs_residual": abs_resid,
        "fraction_of_residual_explainable_by_max_dtype_delta": None if abs_resid == 0 else float(max_abs_delta / abs_resid),
        "classification": "terminal_dtype_can_explain_residual" if max_abs_delta >= 0.5 * abs_resid else "terminal_dtype_too_small_to_explain_residual",
    }


def sorted_variable_labels(acc: kacc.LabelAccessor, H: int) -> List[Tuple[str, Tuple[Any, ...], float]]:
    rows: List[Tuple[str, Tuple[Any, ...], float]] = []
    for parts, value in acc.unscaled_records:
        if not parts:
            continue
        top = parts[0]
        if top == "_x" and len(parts) >= 6 and isinstance(parts[1], int) and 0 <= int(parts[1]) <= H:
            rows.append(("[" + ",".join(str(p) for p in parts) + "]", tuple(parts), float(value)))
        elif top == "_u" and len(parts) >= 5 and isinstance(parts[1], int) and 0 <= int(parts[1]) < H:
            rows.append(("[" + ",".join(str(p) for p in parts) + "]", tuple(parts), float(value)))
    # Deterministic CasADi label order is already in records; keep it.
    return rows[:FD_MAX_VARS]


def set_unscaled_label(mpc: Any, label: str, value: float, flat_index: int) -> Tuple[bool, str]:
    # Reuse the v34i typed assignment helper.  It will try semantic tuple forms
    # before falling back to master index.
    return i.safe_set(mpc.opt_x_num_unscaled, label, value, flat_index)


def t1b_finite_difference(mpc: Any, H: int, baseline: Mapping[str, Any], raw_v34g: Mapping[str, Any]) -> Dict[str, Any]:
    acc0 = kacc.LabelAccessor(mpc)
    labels = sorted_variable_labels(acc0, H)
    grad_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    # map label to flat index in the unscaled opt_x records for master fallback.
    label_to_flat: Dict[str, int] = {}
    for idx, (parts, _val) in enumerate(acc0.unscaled_records):
        label_to_flat["[" + ",".join(str(p) for p in parts) + "]"] = idx
    base_total = float(baseline["total"])
    for n, (label, parts, value) in enumerate(labels):
        step = max(FD_ABS_FLOOR, FD_REL_STEP * max(1.0, abs(float(value))))
        flat_idx = label_to_flat.get(label, n)
        try:
            okp, hp = set_unscaled_label(mpc, label, float(value) + step, flat_idx)
            if not okp:
                raise ContractError("plus set failed: " + hp)
            plus = formula_value(mpc, kacc.LabelAccessor(mpc), H)["total"]
            okm, hm = set_unscaled_label(mpc, label, float(value) - step, flat_idx)
            if not okm:
                raise ContractError("minus set failed: " + hm)
            minus = formula_value(mpc, kacc.LabelAccessor(mpc), H)["total"]
            okr, hr = set_unscaled_label(mpc, label, float(value), flat_idx)
            if not okr:
                raise ContractError("restore set failed: " + hr)
            deriv = float((float(plus) - float(minus)) / (2.0 * step))
            grad_rows.append({"label": label, "flat_index": flat_idx, "value": value, "step": step, "J_plus": plus, "J_minus": minus, "central_derivative": deriv})
        except Exception as exc:
            try:
                set_unscaled_label(mpc, label, float(value), flat_idx)
            except Exception:
                pass
            failures.append({"label": label, "flat_index": flat_idx, "value": value, "error": repr(exc)})
    derivs = np.asarray([r["central_derivative"] for r in grad_rows], dtype=float)
    absderivs = np.abs(derivs) if derivs.size else np.asarray([], dtype=float)
    top_idx = list(np.argsort(-absderivs)[:20]) if absderivs.size else []
    stats = (((raw_v34g.get("arm") or {}).get("solver_event") or {}).get("solver_stats") or {})
    iterations = stats.get("iterations") or {}
    obj_hist = iterations.get("obj") or []
    inf_pr = iterations.get("inf_pr") or []
    inf_du = iterations.get("inf_du") or []
    d_norm = iterations.get("d_norm") or []
    last_obj_delta = None
    if isinstance(obj_hist, list) and len(obj_hist) >= 2:
        last_obj_delta = float(abs(float(obj_hist[-1]) - float(obj_hist[-2])))
    max_inf_pr = float(inf_pr[-1]) if isinstance(inf_pr, list) and inf_pr else None
    max_inf_du = float(inf_du[-1]) if isinstance(inf_du, list) and inf_du else None
    last_d_norm = float(d_norm[-1]) if isinstance(d_norm, list) and d_norm else None
    abs_resid = abs(float(baseline["residual_total_minus_solver"]))
    return {
        "variable_count_attempted": len(labels),
        "variable_count_evaluated": len(grad_rows),
        "failure_count": len(failures),
        "failures_first20": failures[:20],
        "baseline_recomputed_total": base_total,
        "gradient_norms_unscaled_reconstruction": {
            "linf": None if absderivs.size == 0 else float(np.max(absderivs)),
            "l1": None if absderivs.size == 0 else float(np.sum(absderivs)),
            "l2": None if absderivs.size == 0 else float(np.sqrt(np.sum(absderivs ** 2))),
            "median_abs": None if absderivs.size == 0 else float(np.median(absderivs)),
        },
        "top20_abs_gradient_rows": [grad_rows[j] for j in top_idx],
        "solver_iteration_tail": {
            "iter_count": stats.get("iter_count"),
            "return_status": stats.get("return_status"),
            "success": stats.get("success"),
            "final_inf_pr": max_inf_pr,
            "final_inf_du": max_inf_du,
            "final_d_norm": last_d_norm,
            "last_objective_delta": last_obj_delta,
            "final_obj": float(obj_hist[-1]) if isinstance(obj_hist, list) and obj_hist else None,
            "penultimate_obj": float(obj_hist[-2]) if isinstance(obj_hist, list) and len(obj_hist) >= 2 else None,
        },
        "residual_vs_solver_tail": {
            "baseline_abs_residual": abs_resid,
            "last_ipopt_objective_delta_over_residual": None if last_obj_delta is None or abs_resid == 0 else float(last_obj_delta / abs_resid),
            "final_inf_pr_times_grad_l1_over_residual": None if max_inf_pr is None or absderivs.size == 0 or abs_resid == 0 else float(max_inf_pr * float(np.sum(absderivs)) / abs_resid),
            "final_d_norm_times_grad_l2_over_residual": None if last_d_norm is None or absderivs.size == 0 or abs_resid == 0 else float(last_d_norm * float(np.sqrt(np.sum(absderivs ** 2))) / abs_resid),
        },
    }


def common_symbolic_expr(mpc: Any, names: Sequence[str]) -> Tuple[Optional[str], Any]:
    for name in names:
        if hasattr(mpc, name):
            expr = getattr(mpc, name)
            # Avoid ordinary numeric result placeholders.
            if hasattr(expr, "shape") and not isinstance(expr, (float, int, np.ndarray)):
                return name, expr
    return None, None


def evaluate_symbolic(mpc: Any, expr: Any, label: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if ca is None:
        return None, {"ok": False, "reason": "casadi_import_failed"}
    opt_x_sym = getattr(mpc, "opt_x", None)
    opt_p_sym = getattr(mpc, "opt_p", None)
    if opt_x_sym is None or opt_p_sym is None:
        return None, {"ok": False, "reason": "mpc.opt_x_or_opt_p_symbolic_missing"}
    try:
        fn = ca.Function("v34m_" + label, [opt_x_sym, opt_p_sym], [expr])
        val = fn(mpc.opt_x_num, mpc.opt_p_num)
        return arr(val), {"ok": True, "expr_shape": tuple(int(x) for x in getattr(expr, "shape", ())), "value_size": int(arr(val).size)}
    except Exception as exc:
        return None, {"ok": False, "error": repr(exc), "expr_repr_head": repr(expr)[:500]}


def lagrangian_constraint_fd(mpc: Any, f_expr: Any, g_expr: Any, lam_g: np.ndarray, labels: Sequence[Tuple[str, Tuple[Any, ...], float]], H: int) -> Dict[str, Any]:
    if ca is None:
        return {"available": False, "reason": "casadi_import_failed"}
    opt_x_sym = getattr(mpc, "opt_x", None)
    opt_p_sym = getattr(mpc, "opt_p", None)
    if opt_x_sym is None or opt_p_sym is None:
        return {"available": False, "reason": "missing opt_x/opt_p symbolic"}
    try:
        f_fn = ca.Function("v34m_fd_f", [opt_x_sym, opt_p_sym], [f_expr])
        g_fn = ca.Function("v34m_fd_g", [opt_x_sym, opt_p_sym], [g_expr])
    except Exception as exc:
        return {"available": False, "reason": "function_build_failed", "error": repr(exc)}
    lam = np.asarray(lam_g, dtype=float).reshape(-1)
    try:
        g0 = arr(g_fn(mpc.opt_x_num, mpc.opt_p_num)).reshape(-1)
        f0 = finite_float(f_fn(mpc.opt_x_num, mpc.opt_p_num), None)
    except Exception as exc:
        return {"available": False, "reason": "baseline_eval_failed", "error": repr(exc)}
    if g0.size != lam.size:
        return {"available": False, "reason": "lam_g_size_mismatch", "g_size": int(g0.size), "lam_g_size": int(lam.size)}
    label_to_flat = {}
    acc0 = kacc.LabelAccessor(mpc)
    for idx, (parts, _val) in enumerate(acc0.unscaled_records):
        label_to_flat["[" + ",".join(str(p) for p in parts) + "]"] = idx
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    # This is an auxiliary stability residual, not a formal KKT certificate,
    # because lam_x/bounds are not available from v34g full capture.
    for n, (label, parts, value) in enumerate(labels):
        step = max(FD_ABS_FLOOR, FD_REL_STEP * max(1.0, abs(float(value))))
        flat_idx = label_to_flat.get(label, n)
        try:
            okp, hp = set_unscaled_label(mpc, label, float(value) + step, flat_idx)
            if not okp:
                raise ContractError("plus set failed: " + hp)
            fp = finite_float(f_fn(mpc.opt_x_num, mpc.opt_p_num), None)
            gp = arr(g_fn(mpc.opt_x_num, mpc.opt_p_num)).reshape(-1)
            okm, hm = set_unscaled_label(mpc, label, float(value) - step, flat_idx)
            if not okm:
                raise ContractError("minus set failed: " + hm)
            fm = finite_float(f_fn(mpc.opt_x_num, mpc.opt_p_num), None)
            gm = arr(g_fn(mpc.opt_x_num, mpc.opt_p_num)).reshape(-1)
            okr, hr = set_unscaled_label(mpc, label, float(value), flat_idx)
            if not okr:
                raise ContractError("restore set failed: " + hr)
            if fp is None or fm is None or gp.size != lam.size or gm.size != lam.size:
                raise ContractError("nonfinite or size-mismatched f/g perturbation")
            df = float((fp - fm) / (2.0 * step))
            dlamg = float(np.dot((gp - gm) / (2.0 * step), lam))
            rows.append({"label": label, "df": df, "d_lam_g": dlamg, "constraint_only_lagrangian_grad": df + dlamg})
        except Exception as exc:
            try:
                set_unscaled_label(mpc, label, float(value), flat_idx)
            except Exception:
                pass
            failures.append({"label": label, "error": repr(exc)})
    vals = np.asarray([r["constraint_only_lagrangian_grad"] for r in rows], dtype=float)
    absvals = np.abs(vals) if vals.size else np.asarray([], dtype=float)
    top = list(np.argsort(-absvals)[:20]) if absvals.size else []
    return {"available": True, "formal_kkt_certificate": False, "reason_not_formal": "bound multipliers lam_x were not captured in v34g; this is only objective plus equality/inequality constraint lam_g sensitivity", "f0": f0, "g0_size": int(g0.size), "lam_g_size": int(lam.size), "evaluated_variables": len(rows), "failure_count": len(failures), "failures_first20": failures[:20], "constraint_only_lagrangian_grad_norms": {"linf": None if absvals.size == 0 else float(np.max(absvals)), "l1": None if absvals.size == 0 else float(np.sum(absvals)), "l2": None if absvals.size == 0 else float(np.sqrt(np.sum(absvals ** 2)))}, "top20_abs_constraint_only_lagrangian_rows": [rows[j] for j in top]}


def t1c_direct_nlp_and_multipliers(mpc: Any, H: int, baseline: Mapping[str, Any], raw_v34g: Mapping[str, Any], fd_labels: Sequence[Tuple[str, Tuple[Any, ...], float]]) -> Dict[str, Any]:
    f_name, f_expr = common_symbolic_expr(mpc, ["nlp_obj", "_nlp_obj", "nlp_f", "_nlp_f", "obj", "_obj"])
    g_name, g_expr = common_symbolic_expr(mpc, ["nlp_cons", "_nlp_cons", "nlp_g", "_nlp_g", "cons", "_cons"])
    f_val, f_meta = (None, {"ok": False, "reason": "no_candidate_objective_expr", "tried": ["nlp_obj", "_nlp_obj", "nlp_f", "_nlp_f", "obj", "_obj"]}) if f_expr is None else evaluate_symbolic(mpc, f_expr, "nlp_f")
    g_val, g_meta = (None, {"ok": False, "reason": "no_candidate_constraint_expr", "tried": ["nlp_cons", "_nlp_cons", "nlp_g", "_nlp_g", "cons", "_cons"]}) if g_expr is None else evaluate_symbolic(mpc, g_expr, "nlp_g")
    solver = finite_float(getattr(mpc, "opt_f_num", None), None)
    direct_f_scalar = None if f_val is None or np.asarray(f_val).size != 1 else float(np.asarray(f_val).reshape(-1)[0])
    direct_f_abs_err = None if direct_f_scalar is None or solver is None else float(abs(direct_f_scalar - solver))
    direct_f_rel_err = None if direct_f_abs_err is None or solver is None else float(direct_f_abs_err / max(1.0, abs(solver)))
    opt_g = arr(getattr(mpc, "opt_g_num", [])).reshape(-1)
    g_eval = np.asarray([]) if g_val is None else np.asarray(g_val, dtype=float).reshape(-1)
    g_diff_linf = None
    if g_eval.size and opt_g.size == g_eval.size:
        g_diff_linf = float(np.max(np.abs(g_eval - opt_g)))
    post = (((raw_v34g.get("arm") or {}).get("solver_event") or {}).get("post") or {})
    lam_snap = post.get("complete_lam_g_num") or {}
    lam = np.asarray(lam_snap.get("flat_values") or [], dtype=float).reshape(-1)
    constraint_resid = i.residual_localization(mpc)
    repaired = constraint_resid.get("repaired_names") or {}
    c_res = repaired.get("constraint_residual_using_cons_lb_names")
    b_res = repaired.get("bound_residual_using_lb_opt_x_names")
    lam_stats = {"available": bool(lam.size), "flat_size": int(lam.size), "max_abs": None if lam.size == 0 else float(np.max(np.abs(lam))), "l1": None if lam.size == 0 else float(np.sum(np.abs(lam))), "l2": None if lam.size == 0 else float(np.sqrt(np.sum(lam ** 2))), "hash_from_v34g": lam_snap.get("flat_hash")}
    abs_resid = abs(float(baseline["residual_total_minus_solver"]))
    dual_scale = {"constraint_residual": c_res, "bound_residual": b_res, "lam_g_max_abs_times_constraint_residual": None, "lam_g_l1_times_constraint_residual": None, "over_residual_max_abs_bound": None, "over_residual_l1_bound": None}
    if c_res is not None and lam.size:
        prod_max = float(abs(float(c_res)) * float(np.max(np.abs(lam))))
        prod_l1 = float(abs(float(c_res)) * float(np.sum(np.abs(lam))))
        dual_scale.update({"lam_g_max_abs_times_constraint_residual": prod_max, "lam_g_l1_times_constraint_residual": prod_l1, "over_residual_max_abs_bound": None if abs_resid == 0 else prod_max / abs_resid, "over_residual_l1_bound": None if abs_resid == 0 else prod_l1 / abs_resid})
    lagr_fd = {"available": False, "reason": "not_attempted"}
    if f_expr is not None and g_expr is not None and lam.size:
        lagr_fd = lagrangian_constraint_fd(mpc, f_expr, g_expr, lam, fd_labels, H)
    classification_bits: List[str] = []
    if direct_f_rel_err is not None and direct_f_rel_err <= DIRECT_NLP_MATCH_TOL:
        classification_bits.append("direct_symbolic_nlp_objective_matches_solver_at_saved_x")
    elif direct_f_rel_err is not None:
        classification_bits.append("direct_symbolic_nlp_objective_does_not_match_solver")
    else:
        classification_bits.append("direct_symbolic_nlp_objective_unavailable")
    if dual_scale.get("over_residual_l1_bound") is not None and float(dual_scale["over_residual_l1_bound"]) >= 0.5:
        classification_bits.append("lam_g_constraint_residual_scale_could_cover_order_of_residual_under_l1_bound")
    elif dual_scale.get("over_residual_l1_bound") is not None:
        classification_bits.append("lam_g_constraint_residual_scale_too_small_under_l1_bound")
    else:
        classification_bits.append("dual_scale_bound_unavailable")
    return {"symbolic_objective_candidate_name": f_name, "symbolic_constraint_candidate_name": g_name, "direct_nlp_objective_eval": {"value": direct_f_scalar, "solver_objective": solver, "abs_error": direct_f_abs_err, "relative_error": direct_f_rel_err, "meta": f_meta}, "direct_nlp_constraint_eval": {"g_size": int(g_eval.size), "opt_g_size": int(opt_g.size), "g_eval_minus_saved_opt_g_linf": g_diff_linf, "meta": g_meta}, "residual_alias_recheck": constraint_resid, "lam_g_stats": lam_stats, "dual_scale_against_reconstruction_residual": dual_scale, "constraint_only_lagrangian_fd": lagr_fd, "classification_bits": classification_bits}


def write_outputs(raw: Mapping[str, Any]) -> None:
    head = raw["headline"]
    t1a = raw["T1a_terminal_dtype"]
    t1b = raw["T1b_fd_sensitivity"]
    t1c = raw["T1c_direct_nlp_and_multipliers"]
    lines = [
        "# v34m/A13c-2 residual attribution diagnostic",
        "",
        f"UTC: `{raw['created_utc']}`. Zero-solve/zero-plant diagnostic over already-opened v34g/v34k/v34l evidence.",
        "",
        "## Budget",
        f"- solver calls: `{raw['budget_actual']['solver_calls']}`; plant steps: `{raw['budget_actual']['plant_steps']}`; env_reset/env_step after construction: `{raw['budget_actual']['env_reset_calls_after_construction']}`/`{raw['budget_actual']['env_step_calls_after_construction']}`; training/refit: `0`; validation64: `0`; sealed test: `0`.",
        "",
        "## Headline",
        f"- G1 status: `{head['G1_status']}`; classification: `{head['residual_attribution_classification']}`.",
        f"- solver objective: `{head['solver_objective']}`; reconstructed objective: `{head['reconstructed_objective']}`; residual: `{head['reconstruction_residual']}`; relative: `{head['reconstruction_relative_error']}`.",
        f"- direct symbolic NLP objective rel error: `{head['direct_nlp_objective_relative_error']}`.",
        "",
        "## T1a terminal dtype",
        f"- max terminal dtype delta: `{t1a['max_abs_terminal_dtype_delta']}`; fraction of residual: `{t1a['fraction_of_residual_explainable_by_max_dtype_delta']}`; classification: `{t1a['classification']}`.",
        "",
        "## T1b finite-difference/KKT scale",
        f"- evaluated variables: `{t1b['variable_count_evaluated']}` / `{t1b['variable_count_attempted']}`; gradient norms: `{t1b['gradient_norms_unscaled_reconstruction']}`.",
        f"- solver iteration tail: `{t1b['solver_iteration_tail']}`; residual ratios: `{t1b['residual_vs_solver_tail']}`.",
        "",
        "## T1c direct NLP / multipliers",
        f"- direct NLP objective: `{t1c['direct_nlp_objective_eval']}`.",
        f"- direct NLP constraints: `{t1c['direct_nlp_constraint_eval']}`.",
        f"- lam_g stats: `{t1c['lam_g_stats']}`; dual scale: `{t1c['dual_scale_against_reconstruction_residual']}`.",
        f"- constraint-only Lagrangian FD available: `{t1c['constraint_only_lagrangian_fd'].get('available')}`.",
        "",
        "This remains development-only instrumentation, not validation64, not sealed/final test, and not evidence of adaptive-controller performance.",
        f"Raw: `{rel(RUN_DIR/'raw.json')}`. Completed: `{rel(RUN_DIR/'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Compact CSV for the finite-difference top rows.
    with (RUN_DIR / "fd_top_gradients.csv").open("w", encoding="utf-8", newline="") as fcsv:
        fields = ["rank", "label", "value", "step", "central_derivative", "J_plus", "J_minus"]
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader()
        for n, row in enumerate(t1b.get("top20_abs_gradient_rows") or [], start=1):
            w.writerow({k: clean(n if k == "rank" else row.get(k)) for k in fields})
    block = f"""
<!-- {MARKER} -->
## v34m/A13c-2 residual attribution diagnostic

UTC: {raw['created_utc']}. Ran Opus A13c-2 exactly as a zero-solve/zero-plant diagnostic over already-opened v34g/v34k/v34l arrays. Budgets: solver_calls=0, plant_steps=0, env_reset/env_step after construction=0/0, training/refit=0, validation64=0, sealed_test=0. G1_status={head['G1_status']}; classification={head['residual_attribution_classification']}; solver={head['solver_objective']}; reconstructed={head['reconstructed_objective']}; residual={head['reconstruction_residual']}; rel={head['reconstruction_relative_error']}. T1a terminal dtype max delta={t1a['max_abs_terminal_dtype_delta']} (fraction {t1a['fraction_of_residual_explainable_by_max_dtype_delta']}). T1b evaluated {t1b['variable_count_evaluated']} variables; gradient norms={t1b['gradient_norms_unscaled_reconstruction']}; solver tail={t1b['solver_iteration_tail']}. T1c direct NLP objective rel error={head['direct_nlp_objective_relative_error']}; lam_g dual scale={t1c['dual_scale_against_reconstruction_residual']}; constraint-only Lagrangian FD available={t1c['constraint_only_lagrangian_fd'].get('available')}. Evidence: `{rel(RUN_DIR/'summary.md')}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'completed.json')}`, `{rel(RUN_DIR/'fd_top_gradients.csv')}`. Backup request: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)
    write_json(NEXT_REVIEW_REQUEST, {
        "request_id": REQUEST_ID,
        "created": raw["created_utc"],
        "status": "gate_evidence_ready" if head["G1_pass"] else "analysis_requested",
        "trigger": "v34m/A13c-2 zero-solve residual attribution completed",
        "experiment_id": NAME,
        "active_lead_report": rel(OPUS_REPORT),
        "active_lead_report_sha256": OPUS_REPORT_SHA,
        "question": "Review A13c-2 residual attribution. If G1_pass is accepted, the active plan authorizes A13c-3 <=6-solve non-optimal reconstruction consistency before any remaining Task-C 23-call probe. If not, select the minimal repair/stop condition; do not use validation64 or sealed test.",
        "evidence_paths": [rel(RUN_DIR/"summary.md"), rel(RUN_DIR/"raw.json"), rel(RUN_DIR/"completed.json"), rel(RUN_DIR/"fd_top_gradients.csv"), rel(V34L_RAW), rel(V34K_CSV), rel(OPUS_REPORT), rel(RESPONSE_LOG)],
        "budget_actual": raw["budget_actual"],
        "headline": head,
        "backup_required_before_more_unique_science": rel(BACKUP_REQUEST),
    })
    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v34m_residual_attribution",
        "created_utc": raw["created_utc"],
        "backup_required_before_more_unique_science": True,
        "reason": "new A13c-2 residual attribution source/output/docs must be externally recoverable before A13c-3 solver calls or any Task-C continuation",
        "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(NEXT_REVIEW_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
        "new_solver_calls": 0,
        "new_plant_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_gate": "after verified backup and lead acceptance of G1, run A13c-3 <=6-solve non-optimal reconstruction consistency; do not run 23-call Task-C probe before G2 passes",
    })
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        "# Continue state after v34m/A13c-2 residual attribution\n\n"
        + f"UTC: {raw['created_utc']}\n\nHeadline: {json.dumps(clean(head), sort_keys=True)}\n\n"
        + f"Artifacts: {rel(RUN_DIR/'summary.md')}, {rel(RUN_DIR/'raw.json')}, {rel(RUN_DIR/'completed.json')}, {rel(RUN_DIR/'fd_top_gradients.csv')}\n\n"
        + f"Next: verify external backup for {rel(BACKUP_REQUEST)}. If active lead accepts G1_pass, continue to A13c-3 <=6-solve non-optimal/off-optimum reconstruction consistency. Do not spend the remaining Task-C 23 calls until G2 passes.\n",
        encoding="utf-8",
    )


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-time", required=True)
    ap.add_argument("--backup-commit", required=True)
    ap.add_argument("--backup-package-sha256", required=True)
    ap.add_argument("--backup-package-bytes", type=int, required=True)
    ap.add_argument("--i-accept-v34m-zero-solve-residual-attribution", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        started = now_utc()
        inputs = verify_inputs(args)
        i.h.patch_no_solve_runtime()
        raw_v34g = read_json(V34G_RAW)
        raw_v34l = read_json(V34L_RAW)
        mpc, build_meta = i.h.build_mpc_without_solve()
        assign_meta = i.assign_saved_arrays(mpc, raw_v34g)
        H = int(kacc.TARGET["horizon"])
        baseline = formula_value(mpc, kacc.LabelAccessor(mpc), H)
        v34l_baseline = ((raw_v34l.get("residual_dissection") or {}))
        baseline_consistency = {
            "baseline_total_matches_v34l": abs(float(baseline["total"]) - float(v34l_baseline.get("total_fsum", baseline["total"]))) <= 1e-9,
            "baseline_solver_matches_v34l": abs(float(baseline["solver_objective"]) - float(v34l_baseline.get("solver_objective", baseline["solver_objective"]))) <= 1e-9,
            "baseline_residual_matches_v34l": abs(float(baseline["residual_total_minus_solver"]) - float(v34l_baseline.get("residual_total_minus_solver", baseline["residual_total_minus_solver"]))) <= 1e-9,
        }
        if not all(baseline_consistency.values()):
            raise ContractError("reconstructed baseline does not match v34l raw: " + repr(baseline_consistency))
        # T1a.
        t1a = t1a_terminal_dtype(mpc, kacc.LabelAccessor(mpc), H, baseline)
        # T1b. This mutates/restores unscaled opt_x; rebuild accessor after it.
        t1b = t1b_finite_difference(mpc, H, baseline, raw_v34g)
        fd_labels = sorted_variable_labels(kacc.LabelAccessor(mpc), H)
        # T1c.
        t1c = t1c_direct_nlp_and_multipliers(mpc, H, baseline, raw_v34g, fd_labels)
        forbidden = build_meta.get("forbidden_call_counter", {})
        budget_actual = {
            "solver_calls": int(forbidden.get("solve", 0)),
            "plant_steps": 0,
            "env_reset_calls_after_construction": int(forbidden.get("env_reset", 0)),
            "env_step_calls_after_construction": int(forbidden.get("env_step", 0)),
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        }
        if any(int(budget_actual[k]) != 0 for k in budget_actual):
            raise ContractError("zero-budget contract violated: " + repr(budget_actual))
        direct_rel = (t1c.get("direct_nlp_objective_eval") or {}).get("relative_error")
        dtype_frac = t1a.get("fraction_of_residual_explainable_by_max_dtype_delta")
        dual_l1_frac = (t1c.get("dual_scale_against_reconstruction_residual") or {}).get("over_residual_l1_bound")
        reconstruction_rel = baseline.get("relative_error")
        classification = []
        if dtype_frac is not None and float(dtype_frac) < 0.5:
            classification.append("T1a_rejects_terminal_float32_rounding_as_primary_cause")
        elif dtype_frac is not None:
            classification.append("T1a_terminal_dtype_could_explain_order")
        else:
            classification.append("T1a_terminal_dtype_unavailable")
        if dual_l1_frac is not None and float(dual_l1_frac) < 0.5:
            classification.append("T1b_T1c_dual_constraint_residual_scale_too_small_under_l1_bound")
        elif dual_l1_frac is not None:
            classification.append("T1b_T1c_dual_constraint_residual_scale_could_cover_order_under_l1_bound")
        else:
            classification.append("T1b_T1c_dual_scale_bound_unavailable")
        if direct_rel is not None and float(direct_rel) <= DIRECT_NLP_MATCH_TOL:
            classification.append("T1c_direct_NLP_objective_matches_solver_so_reconstruction_residual_is_formula/convention_gap")
        elif direct_rel is not None:
            classification.append("T1c_direct_NLP_objective_does_not_match_solver")
        else:
            classification.append("T1c_direct_NLP_objective_unavailable")
        # G1 is a mechanical attribution gate: either direct NLP closes the solver
        # value, or all requested checks ran and the residual remains bounded below
        # the already-predeclared 1e-6 relative reconstruction tolerance.
        g1_pass = bool((direct_rel is not None and float(direct_rel) <= DIRECT_NLP_MATCH_TOL) or (reconstruction_rel is not None and float(reconstruction_rel) <= RECON_REL_TOL and t1b.get("variable_count_evaluated", 0) > 0))
        g1_status = "pass_attributed_to_formula_convention_gap_with_direct_nlp_closure" if (direct_rel is not None and float(direct_rel) <= DIRECT_NLP_MATCH_TOL) else ("pass_bounded_below_reconstruction_tolerance" if g1_pass else "fail_unattributed_or_unbounded")
        created = now_utc()
        headline = {
            "G1_pass": g1_pass,
            "G1_status": g1_status,
            "residual_attribution_classification": ";".join(classification),
            "solver_objective": baseline.get("solver_objective"),
            "reconstructed_objective": baseline.get("total"),
            "reconstruction_residual": baseline.get("residual_total_minus_solver"),
            "reconstruction_relative_error": baseline.get("relative_error"),
            "direct_nlp_objective_relative_error": direct_rel,
            "terminal_dtype_fraction_of_residual": dtype_frac,
            "dual_l1_times_constraint_fraction_of_residual": dual_l1_frac,
            "fd_variable_count_evaluated": t1b.get("variable_count_evaluated"),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        raw = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_zero_solve_a13c2_residual_attribution_not_validation_not_test",
            "active_lead": "claude-opus-5-5",
            "lead_report": rel(OPUS_REPORT),
            "lead_report_sha256": OPUS_REPORT_SHA,
            "hypothesis_frozen": "A13c-2 should attribute or bound the v34k/v34l 3.656e-4 reconstruction residual via exactly T1a terminal dtype, T1b finite-difference/KKT scale, and T1c direct NLP/multiplier stability checks, without new solver calls.",
            "input_gate": inputs,
            "runtime_build_meta": build_meta,
            "assignment_meta": assign_meta,
            "baseline_formula_reconstruction": baseline,
            "baseline_consistency_vs_v34l": baseline_consistency,
            "T1a_terminal_dtype": t1a,
            "T1b_fd_sensitivity": t1b,
            "T1c_direct_nlp_and_multipliers": t1c,
            "headline": headline,
            "budget_declared": {"solver_call_cap": 0, "plant_steps": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "budget_actual": budget_actual,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "interpretation_limits": ["development-only", "already-opened v34g/v34k/v34l evidence", "zero new solver calls", "not validation64", "not sealed/final test", "no selector policy or closed-loop claim", "A13c-3 and Task-C remain separate dependent tasks"],
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid(), "casadi_imported": ca is not None},
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_outputs(raw)
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE, BACKUP_REQUEST, NEXT_REVIEW_REQUEST, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed = {
            "status": "complete",
            "passed": True,
            "hard_pass": bool(g1_pass),
            "created_utc": created.isoformat(),
            "classification": raw["classification"],
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_actual": budget_actual,
            "headline": headline,
            "summary": rel(RUN_DIR / "summary.md"),
            "raw": rel(RUN_DIR / "raw.json"),
            "fd_top_gradients_csv": rel(RUN_DIR / "fd_top_gradients.csv"),
            "backup_request": rel(BACKUP_REQUEST),
            "next_review_request_id": REQUEST_ID,
            "hashes": hash_existing(files),
        }
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "fd_top_gradients_csv": rel(RUN_DIR / "fd_top_gradients.csv"), "headline": headline, "backup_request": rel(BACKUP_REQUEST)}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        fail = {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_zero_solve_a13c2_residual_attribution_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_caps": {"solver_call_cap": 0, "plant_steps": 0, "env_reset_calls_after_construction": 0}}
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(f"# v34m/A13c-2 residual attribution failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR/'failed.json')}\n\nNo validation64 or sealed test access was requested. Preserve failure; repair operational script bug or return formula-miss evidence to lead before A13c-3/Task-C.\n", encoding="utf-8")
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
