#!/usr/bin/env python3
"""v34c zero-solve contract preflight for the objective-vs-basin probe.

Astra 20260930T084456Z Task1: before spending the fixed 24 low-level solver
calls, verify that the v34 diagnostic execution chain can reconstruct the two
opened development contexts, scalarize historical singleton inputs, prepare the
same non-terminal context across terminal arms, apply two distinct typed initial
primal guesses, call the real four-argument terminal evaluator, and persist a
failure/audit ledger.  This script deliberately intercepts before mpc.solve and
therefore performs 0 NLP solves, 0 plant steps, no training/refit, no validation64
and no sealed-test access.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
REPRO_DIR = ROOT / "experiments/bohn2021_reproduction"
for _p in (AWS_DIR, REPRO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0 as m  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34c_contract_preflight_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
ARRAY_DIR = RUN_DIR / "arrays"
BACKUP_REQ = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34C_CONTRACT_PREFLIGHT_{STAMP}.json"
STATE = ROOT / "research_artifacts/aws_state" / f"continue_state_{STAMP}_after_v34c_contract_preflight.md"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-v34c-contract-preflight-{STAMP}"
TERMINAL_MODES = ["zero", "V15_shared", "V35_shared"]
INIT_MODES = ["canonical", "goal_facing"]
EXPECTED_INTERCEPTS = 24


class ContractError(RuntimeError):
    pass


class NoSolveIntercept(RuntimeError):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(x: Any) -> Any:
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if hasattr(x, "item"):
        try:
            return clean(x.item())
        except Exception:
            pass
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def arr(value: Any) -> np.ndarray:
    try:
        if hasattr(value, "full"):
            return np.asarray(value.full(), dtype=float).reshape(-1)
        if hasattr(value, "cat"):
            return np.asarray(value.cat, dtype=float).reshape(-1)
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def arr_hash(value: Any) -> str:
    a = arr(value)
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def finite_scalar(value: Any, label: str) -> float:
    try:
        a = np.asarray(value, dtype=float).reshape(-1)
    except Exception as exc:
        raise ContractError(f"could not scalarize {label}: {value!r}: {exc!r}")
    if a.size != 1:
        raise ContractError(f"expected singleton scalar for {label}, got size {a.size} value={value!r}")
    out = float(a[0])
    if not math.isfinite(out):
        raise ContractError(f"nonfinite scalar for {label}: {value!r}")
    return out


def parse_label(label: str) -> List[str]:
    return m.parse_label(label)


def get_struct_labels(obj: Any) -> List[str]:
    return m.get_struct_labels(obj)


def safe_get(obj: Any, parts: Sequence[Any]) -> Any:
    try:
        return obj[tuple(parts)]
    except Exception:
        try:
            return obj.__getitem__(tuple(parts))
        except Exception:
            return None


def safe_set(obj: Any, parts: Sequence[Any], value: Any) -> bool:
    try:
        obj[tuple(parts)] = value
        return True
    except Exception:
        try:
            obj.__setitem__(tuple(parts), value)
            return True
        except Exception:
            return False


def load_contexts_strict() -> List[Dict[str, Any]]:
    specs = m.v29.build_state_specs()
    source242 = m.find_spec("v27_case09_slot0_early_risk", specs)
    c13 = m.find_spec("v19_c13", specs)
    c13_trace_path, c13_rows = m.find_c13_trace()
    c13_step0 = c13_rows[0]
    c13_step1 = c13_rows[1]
    c13_input = c13_step0.get("input") or {}
    contexts = [
        {
            "context_id": "source242_slot0_branch_start",
            "state_label": "v27_case09_slot0_early_risk",
            "source": "v29/v33 selected branch state, original branch start",
            "case_snapshot": copy.deepcopy(source242["case_snapshot"]),
            "branch_step": int(source242["branch_step"]),
            "tvp_start_index": int(source242["branch_step"]),
            "state": m.state_clean(source242["branch_previous_state"]),
            "previous_input": {"u_omega": 0.0, "u_s": 0.0},
            "previous_input_source": "Astra-specified v33 branch-reset zero-input semantics",
            "horizons": [15, 35],
        },
        {
            "context_id": "v19_c13_step1_after_V15_H35_common_state",
            "state_label": "v19_c13",
            "source": rel(c13_trace_path) + ": second trace row previous_state",
            "case_snapshot": copy.deepcopy(c13["case_snapshot"]),
            "branch_step": int(c13["branch_step"]),
            "tvp_start_index": int(c13["branch_step"]) + 1,
            "state": m.state_clean(c13_step1.get("previous_state") or {}),
            "previous_input": {str(k): finite_scalar(v, f"c13_step0.input.{k}") for k, v in c13_input.items()},
            "previous_input_source": rel(c13_trace_path) + ": first trace row input (strict singleton scalarized)",
            "horizons": [12, 35],
        },
    ]
    expected = {"theta": 0.08060330210484318, "x": 13.74497830467862, "y": 2.8552860589337556}
    if m.state_distance(contexts[1]["state"], expected) > 1e-6:
        raise ContractError("c13 reconstructed state does not match Astra-specified fixed state")
    return contexts


def vector_state(env: Any, state: Mapping[str, float]) -> Any:
    return m.state_vector_from_dict(env, state)


def vector_input(env: Any, inputs: Mapping[str, float]) -> Any:
    return m.input_vector_from_dict(env, inputs)


def setup_context_direct(env: Any, context: Mapping[str, Any], h: int) -> Dict[str, Any]:
    shifted = m.shift_case_tvp(context["case_snapshot"], int(context["tvp_start_index"]), h)
    gx, gy, goal_source = m.extract_goal_xy(context["case_snapshot"], shifted)
    ctrl = env.control_system.controller
    state = m.state_clean(context["state"])
    prev = {str(k): finite_scalar(v, f"previous_input.{k}") for k, v in (context.get("previous_input") or {}).items()}
    for name, values in shifted.items():
        if hasattr(env.control_system, "tvps") and name in env.control_system.tvps:
            env.control_system.tvps[name].values = copy.deepcopy(values)
    env.control_system._step_count = 0
    env.control_system.current_state = copy.deepcopy(state)
    ctrl._tvp_data = copy.deepcopy(shifted)
    ctrl.goal_x = float(gx); ctrl.goal_y = float(gy)
    try:
        env.trajectory_goal_x = float(gx); env.trajectory_goal_y = float(gy)
    except Exception:
        pass
    if not hasattr(ctrl, "current_input") or not isinstance(ctrl.current_input, Mapping):
        ctrl.current_input = {}
    for name in list(getattr(ctrl, "input_names", ["u_omega", "u_s"])):
        ctrl.current_input[name] = float(prev.get(name, 0.0))
    if not hasattr(ctrl, "current_reference") or not isinstance(ctrl.current_reference, Mapping):
        ctrl.current_reference = {}
    for ref_name in list(ctrl.current_reference.keys()):
        if ref_name in shifted and shifted[ref_name]:
            ctrl.current_reference[ref_name] = shifted[ref_name][0]
    if not hasattr(ctrl, "history") or not isinstance(ctrl.history, Mapping):
        ctrl.history = {}
    for key in ("mpc_horizon", "inputs", "references", "errors", "tvp"):
        ctrl.history.setdefault(key, [])
    try:
        ctrl.mpc._x0.master = vector_state(env, state)
    except Exception:
        pass
    try:
        ctrl.mpc._u0.master = vector_input(env, ctrl.current_input)
    except Exception:
        pass
    dist = m.state_distance(state, m.state_clean(env.control_system.current_state))
    if dist > m.STATE_DISTANCE_TOL:
        raise ContractError(f"direct context state drifted before intercept: {dist}")
    return {
        "shifted_tvp": shifted,
        "goal_x": float(gx),
        "goal_y": float(gy),
        "goal_source": goal_source,
        "state_after_direct_setup": m.state_clean(env.control_system.current_state),
        "state_distance_after_direct_setup": dist,
        "previous_input_applied": copy.deepcopy(ctrl.current_input),
        "tvp_lengths": {k: len(v) for k, v in shifted.items()},
        "tvp_hash": canonical_hash(shifted),
        "direct_setup_note": "No env.reset, env.step, controller.reset, mpc.make_step, or mpc.solve was called during this setup function.",
    }


def label_var(parts: Sequence[str], names: Sequence[str]) -> Optional[str]:
    for p in parts:
        if p in names:
            return p
    return None


def first_int_after_top(parts: Sequence[str]) -> int:
    for p in parts[1:]:
        try:
            return int(p)
        except Exception:
            continue
    return 0


def typed_set_initial_guess(mpc: Any, env: Any, context_meta: Mapping[str, Any], init: str, h: int) -> Dict[str, Any]:
    ctrl = env.control_system.controller
    state_names = list(getattr(ctrl, "state_names", ["theta", "x", "y"]))
    input_names = list(getattr(ctrl, "input_names", ["u_omega", "u_s"]))
    state = m.state_clean(context_meta["state_after_direct_setup"])
    if init == "canonical":
        pred_states, pred_controls = m.zero_guess(state, h)
        rule = "repeat_initial_state_zero_control_zero_algebraic"
    elif init == "goal_facing":
        dt_s = m.as_float(getattr(mpc, "t_step", None), 0.1)
        pred_states, pred_controls = m.predicted_unicycle(state, float(context_meta["goal_x"]), float(context_meta["goal_y"]), h, dt_s)
        rule = "deterministic_unicycle_goal_facing_clip_omega_pm4_speed_0_5"
    else:
        raise ContractError("unknown init " + init)
    obj = getattr(mpc, "opt_x_num", None)
    if obj is None:
        raise ContractError("mpc.opt_x_num missing for typed initialization")
    attempted = succeeded = failed = 0
    examples: List[Dict[str, Any]] = []
    for label in get_struct_labels(obj):
        parts = parse_label(label)
        if not parts:
            continue
        top = parts[0]
        val: Optional[float] = None
        if top == "_x":
            var = label_var(parts, state_names)
            if var is None:
                continue
            k = first_int_after_top(parts)
            if 0 <= k < len(pred_states):
                val = float(pred_states[k].get(var, state.get(var, 0.0)))
        elif top == "_u":
            var = label_var(parts, input_names)
            if var is None:
                continue
            k = first_int_after_top(parts)
            if 0 <= k < len(pred_controls):
                val = float(pred_controls[k].get(var, 0.0))
        elif top in ("_z", "_eps"):
            val = 0.0
        if val is None:
            continue
        attempted += 1
        ok = safe_set(obj, parts, val)
        rb = safe_get(obj, parts) if ok else None
        rb_scalar = None
        try:
            rb_scalar = finite_scalar(rb, "readback." + label)
        except Exception:
            pass
        if ok and rb_scalar is not None and abs(rb_scalar - val) <= 1e-9 * max(1.0, abs(val)):
            succeeded += 1
        else:
            failed += 1
        if len(examples) < 8:
            examples.append({"label": label, "target": val, "ok": ok, "readback": rb_scalar})
    if succeeded <= 0 or failed > 0:
        raise ContractError(f"typed initialization failed: attempted={attempted} succeeded={succeeded} failed={failed}")
    try:
        mpc.lam_g_num = 0 * mpc.lam_g_num
    except Exception:
        pass
    return {
        "initialization": init,
        "rule": rule,
        "state_names": state_names,
        "input_names": input_names,
        "assignments_attempted": attempted,
        "assignments_succeeded": succeeded,
        "assignment_examples": examples,
        "initial_primal_hash": arr_hash(getattr(mpc, "opt_x_num", [])),
        "initial_dual_hash_after_zero": arr_hash(getattr(mpc, "lam_g_num", [])) if hasattr(mpc, "lam_g_num") else None,
        "first_three_pred_states": pred_states[:3],
        "first_three_pred_controls": pred_controls[:3],
    }


def labeled_values(obj: Any, max_items: int = 100000) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for label in get_struct_labels(obj)[:max_items]:
        parts = parse_label(label)
        try:
            out[label] = finite_scalar(safe_get(obj, parts), "struct." + label)
        except Exception:
            out[label] = None
    return out


def nonterminal_pmap(pmap: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in pmap.items() if "_vf_weights" not in k and "_vf_biases" not in k}


def bounds_hashes(mpc: Any) -> Dict[str, Any]:
    names = ["lb_opt_x", "ub_opt_x", "cons_lb", "cons_ub", "rterm_factor", "_x_scaling", "_u_scaling"]
    return {name: {"size": int(arr(getattr(mpc, name, [])).size), "sha256": arr_hash(getattr(mpc, name, []))} if hasattr(mpc, name) else None for name in names}


def save_arrays(prefix: str, mpc: Any) -> Dict[str, Any]:
    ARRAY_DIR.mkdir(parents=True, exist_ok=True)
    p = ARRAY_DIR / f"{prefix}.npz"
    arrays = {
        "opt_x_num": arr(getattr(mpc, "opt_x_num", [])),
        "opt_p_num": arr(getattr(mpc, "opt_p_num", [])),
        "lb_opt_x": arr(getattr(mpc, "lb_opt_x", [])),
        "ub_opt_x": arr(getattr(mpc, "ub_opt_x", [])),
        "cons_lb": arr(getattr(mpc, "cons_lb", [])),
        "cons_ub": arr(getattr(mpc, "cons_ub", [])),
        "x_scaling": arr(getattr(mpc, "_x_scaling", [])),
        "u_scaling": arr(getattr(mpc, "_u_scaling", [])),
    }
    np.savez_compressed(str(p), **arrays)
    return {"path": rel(p), "sha256": sha256(p), "sizes": {k: int(v.size) for k, v in arrays.items()}}


def p_excluding_n_horizon(mpc: Any) -> Tuple[np.ndarray, List[str]]:
    vals = arr(mpc.opt_p_num["_p", 0])
    labels_raw = [str(x) for x in mpc.model._p.labels()]
    names = [str(x).strip().strip("[]").split(",")[0].strip().strip("'").strip('"') for x in labels_raw]
    keep = [i for i, n in enumerate(names) if "n_horizon" not in n]
    return vals[keep], [names[i] for i in keep]


def vf4_check(mpc: Any) -> Dict[str, Any]:
    try:
        x0 = mpc.opt_p_num["_x0"]
        p_keep, p_names = p_excluding_n_horizon(mpc)
        val = mpc.vf_fun(x0, np.asarray(p_keep, dtype=float).reshape((-1, 1)), mpc.vf.weights_num, mpc.vf.biases_num)
        out = finite_scalar(val, "vf_fun_4arg")
        return {"ok": True, "value": out, "p_names_excluding_n_horizon": p_names, "vf_fun_n_in": int(mpc.vf_fun.n_in()) if hasattr(mpc.vf_fun, "n_in") else None}
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-5:]}


def lterm_preflight(mpc: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False}
    try:
        x0 = mpc.opt_p_num["_x0"]
        u0 = mpc.opt_x_num_unscaled["_u", 0, 0]
        try:
            z0 = mpc.opt_x_num_unscaled["_z", 0, 0, -1]
        except Exception:
            z0 = mpc.opt_x_num_unscaled["_z", 0, 0, 0]
        tvp0 = mpc.opt_p_num["_tvp", 0]
        p0 = mpc.opt_p_num["_p", 0]
        val = mpc.lterm_fun(x0, u0, z0, tvp0, p0)
        out.update({"ok": True, "value": finite_scalar(val, "lterm_fun"), "lterm_n_in": int(mpc.lterm_fun.n_in()) if hasattr(mpc.lterm_fun, "n_in") else None})
    except Exception as exc:
        out.update({"error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-5:]})
    try:
        out["epsterm_available"] = hasattr(mpc, "epsterm_fun")
        out["rterm_factor_size"] = int(arr(getattr(mpc, "rterm_factor", [])).size)
        out["discount_factor"] = finite_scalar(getattr(mpc, "discount_factor", 1.0), "discount_factor")
    except Exception:
        pass
    return out


def semantic_coefficients_status() -> Dict[str, Any]:
    paths = sorted(Path(ROOT / "research_artifacts/aws_diagnostics").glob("vehicle_true_variable_horizon_v33_terminal_identity_evidence_audit_v0b_*/terminal_coefficients.csv"))
    status: Dict[str, Any] = {"semantic_csv_found": bool(paths), "semantic_csv": rel(paths[-1]) if paths else None, "semantic_rows": 0, "bias_csv": rel(m.TASK1_TERMINAL_CSV), "bias_by_terminal": {}}
    if paths:
        with paths[-1].open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        status["semantic_rows"] = len(rows)
        status["semantic_inputs"] = sorted({str(r.get("input")) for r in rows})
        status["semantic_sha256"] = sha256(paths[-1])
    if m.TASK1_TERMINAL_CSV.exists():
        with m.TASK1_TERMINAL_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                term = str(row.get("terminal"))
                b = row.get("bias")
                try:
                    if b not in (None, ""):
                        status["bias_by_terminal"][term] = float(b)
                except Exception:
                    pass
        status["bias_csv_sha256"] = sha256(m.TASK1_TERMINAL_CSV)
    return status


def preflight_arm(row: Mapping[str, Any], context: Mapping[str, Any], terminals: Mapping[int, Tuple[Any, Any]]) -> Dict[str, Any]:
    h = int(row["horizon"]); mode = str(row["terminal_mode"]); init = str(row["initialization"])
    terminal, term_h, term_note = m.terminal_for_mode(h, mode, terminals)
    env = m.create_env(h, terminal)
    ctrl = env.control_system.controller
    mpc = ctrl.mpc
    context_meta = setup_context_direct(env, context, h)
    init_meta = typed_set_initial_guess(mpc, env, context_meta, init, h)
    intercepts: List[Dict[str, Any]] = []
    original_solve = mpc.solve

    def intercept_solve(*args: Any, **kwargs: Any) -> Any:
        intercepts.append({"t_perf": time.perf_counter(), "pre_opt_p_hash": arr_hash(getattr(mpc, "opt_p_num", [])), "pre_opt_x_hash": arr_hash(getattr(mpc, "opt_x_num", []))})
        raise NoSolveIntercept("v34c controlled no-solve intercept before NLP solve")

    mpc.solve = intercept_solve
    try:
        ctrl.get_action(m.state_clean(context["state"]), h, tvp_values=copy.deepcopy(context_meta["shifted_tvp"]))
        raise ContractError("controller.get_action returned without hitting the no-solve intercept")
    except NoSolveIntercept:
        pass
    finally:
        mpc.solve = original_solve
    if len(intercepts) != 1:
        raise ContractError(f"expected one controlled solve intercept, got {len(intercepts)}")
    pmap = labeled_values(getattr(mpc, "opt_p_num", None), 200000)
    nt = nonterminal_pmap(pmap)
    prefix = f"arm{int(row['execution_index']):02d}_{row['context_id']}_H{h}_{mode}_{init}".replace("/", "_")
    arrays = save_arrays(prefix, mpc)
    return {
        "execution_index": int(row["execution_index"]),
        "context_id": row["context_id"],
        "state_label": row["state_label"],
        "horizon": h,
        "terminal_mode": mode,
        "terminal_source_horizon": int(term_h),
        "terminal_note": term_note,
        "initialization": init,
        "controlled_no_solve_intercepts": len(intercepts),
        "lower_level_solver_calls": 0,
        "plant_steps": 0,
        "env_reset_calls_after_construction": 0,
        "controller_reset_calls": 0,
        "context_meta": {k: v for k, v in context_meta.items() if k != "shifted_tvp"},
        "initialization_meta": init_meta,
        "prepared_opt_p_hash": arr_hash(getattr(mpc, "opt_p_num", [])),
        "prepared_nonterminal_opt_p_hash": canonical_hash(nt),
        "prepared_nonterminal_opt_p_label_count": len(nt),
        "prepared_opt_p_labeled_hash": canonical_hash(pmap),
        "bounds_hashes": bounds_hashes(mpc),
        "solver_options_hash": canonical_hash(getattr(mpc, "nlpsol_opts", {})),
        "vf4_check": vf4_check(mpc),
        "lterm_preflight": lterm_preflight(mpc),
        "array_artifact": arrays,
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
    }


def analyze(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r["context_id"]), int(r["horizon"]))].append(r)
    group_rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    for (ctx, h), rs in sorted(groups.items()):
        g: Dict[str, Any] = {"context_id": ctx, "horizon": h, "row_count": len(rs), "checks": {}}
        for init in INIT_MODES:
            subset = [r for r in rs if r["initialization"] == init]
            g["checks"][f"initial_primal_identical_across_terminals_{init}"] = len({(r.get("initialization_meta") or {}).get("initial_primal_hash") for r in subset}) == 1
            g["checks"][f"nonterminal_opt_p_identical_across_terminals_{init}"] = len({r.get("prepared_nonterminal_opt_p_hash") for r in subset}) == 1
        for term in TERMINAL_MODES:
            subset = [r for r in rs if r["terminal_mode"] == term]
            g["checks"][f"nonterminal_opt_p_identical_across_initializations_{term}"] = len({r.get("prepared_nonterminal_opt_p_hash") for r in subset}) == 1
        can = [r for r in rs if r["initialization"] == "canonical"]
        gf = [r for r in rs if r["initialization"] == "goal_facing"]
        can_hash = (can[0].get("initialization_meta") or {}).get("initial_primal_hash") if can else None
        gf_hash = (gf[0].get("initialization_meta") or {}).get("initial_primal_hash") if gf else None
        g["checks"]["canonical_and_goal_facing_primal_hashes_distinct"] = bool(can_hash and gf_hash and can_hash != gf_hash)
        g["checks"]["all_vf4_ok"] = all((r.get("vf4_check") or {}).get("ok") is True for r in rs)
        g["checks"]["all_lterm_ok"] = all((r.get("lterm_preflight") or {}).get("ok") is True for r in rs)
        g["checks"]["all_bounds_have_real_names"] = all((r.get("bounds_hashes") or {}).get("lb_opt_x") and (r.get("bounds_hashes") or {}).get("cons_lb") for r in rs)
        for key, ok in g["checks"].items():
            if not ok:
                failures.append(f"{ctx}|H{h}|{key}")
        group_rows.append(g)
    intercepts = int(sum(int(r.get("controlled_no_solve_intercepts", 0)) for r in rows))
    lower = int(sum(int(r.get("lower_level_solver_calls", 0)) for r in rows))
    vf_ok = sum(1 for r in rows if (r.get("vf4_check") or {}).get("ok") is True)
    lterm_ok = sum(1 for r in rows if (r.get("lterm_preflight") or {}).get("ok") is True)
    passed = (len(rows) == EXPECTED_INTERCEPTS and intercepts == EXPECTED_INTERCEPTS and lower == 0 and not failures and vf_ok == len(rows) and lterm_ok == len(rows))
    return {
        "passed": passed,
        "hard_pass": passed,
        "headline": {
            "preflight_rows": len(rows),
            "controlled_no_solve_intercepts": intercepts,
            "lower_level_solver_calls": lower,
            "plant_steps": 0,
            "env_reset_calls_after_construction": 0,
            "vf4_ok_rows": vf_ok,
            "lterm_ok_rows": lterm_ok,
            "group_failure_count": len(failures),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        },
        "groups": group_rows,
        "failures": failures,
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]; h = a["headline"]
    lines = [
        "# v34c objective-basin contract preflight",
        "",
        f"UTC: `{raw['created_utc']}`. Astra Task1 zero-solve preflight for v34 objective-vs-basin diagnostic.",
        "",
        "## Budget/accounting",
        f"- Prepared/intercepted cells: `{h['preflight_rows']}`; controlled no-solve intercepts: `{h['controlled_no_solve_intercepts']}`.",
        f"- Lower-level solver calls: `{h['lower_level_solver_calls']}`; plant steps: `0`; env.reset after construction: `0`; training/refit: `0`.",
        f"- validation64 opened: `{h['validation64_bank_opened']}`; sealed test accessed: `{h['sealed_test_accessed']}`.",
        "",
        "## Contract gates",
        f"- hard_pass: `{a['hard_pass']}`.",
        f"- vf_fun four-argument rows OK: `{h['vf4_ok_rows']}` / `{h['preflight_rows']}`.",
        f"- lterm preflight rows OK: `{h['lterm_ok_rows']}` / `{h['preflight_rows']}`.",
        f"- group failure count: `{h['group_failure_count']}`.",
        "",
        "| context | H | checks |",
        "|---|---:|---|",
    ]
    for g in a["groups"]:
        lines.append(f"| `{g['context_id']}` | {g['horizon']} | `{g['checks']}` |")
    if a["failures"]:
        lines += ["", "## Failures"] + [f"- `{x}`" for x in a["failures"]]
    lines += ["", "## Artifacts", f"- raw: `{rel(RUN_DIR / 'raw.json')}`", f"- arrays: `{rel(ARRAY_DIR)}`", f"- completed: `{rel(RUN_DIR / 'completed.json')}`", f"- backup request: `{rel(BACKUP_REQ)}`"]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def update_docs(raw: Mapping[str, Any]) -> None:
    h = raw["analysis"]["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-30 v34c zero-solve objective-basin contract preflight

UTC: {raw['created_utc']}. Executed Astra Task1 repair/preflight with {h['preflight_rows']} scheduled cells and {h['controlled_no_solve_intercepts']} controlled intercepts before `mpc.solve`; lower-level solver calls=0, plant steps=0, training/refit=0, validation64=false, sealed test=false. hard_pass={raw['analysis']['hard_pass']}; vf4_ok={h['vf4_ok_rows']}/{h['preflight_rows']}; lterm_ok={h['lterm_ok_rows']}/{h['preflight_rows']}; group_failures={h['group_failure_count']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQ)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / doc, MARKER, block)
    response = f"""
<!-- {MARKER} -->
## v34c response to Astra Task1 contract-preflight request

Updated by GPT-5.5 executor at `{raw['created_utc']}` for Astra report `docs/bohn2021_takeover/astra_reviews/20260930T084456Z.md`. Implemented the requested source-specific 0-solve preflight rather than spending the 24-call solver budget immediately.

| linked recommendation(s) | disposition | verified evidence | follow-up |
|---|---|---|---|
| `A11_training_failure_modes_need_separation`, `A12_registry_backup_schema_contract` | accepted/executed | `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'summary.md')}`; headline `{h}` | If hard_pass is true and backup is verified, proceed to the fixed 24-call objective-basin probe using this contract; if false, repair the listed missing primitive only. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | preserved | same artifacts; terminal modes zero/V15/V35 retained for all frozen cells | No selector/refit/validation64/test access from this preflight. |
"""
    append_if_missing(RESPONSE_LOG, MARKER, response)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v34c contract preflight\n\nUTC: {raw['created_utc']}\n\nHeadline: {h}\n\nHard pass: {raw['analysis']['hard_pass']}\n\nArtifacts: {rel(RUN_DIR / 'summary.md')}, {rel(RUN_DIR / 'raw.json')}, {rel(RUN_DIR / 'completed.json')}\n\nNext: verify backup request {rel(BACKUP_REQ)}. If hard_pass true, run the fixed 24-call objective-basin probe with the v34c contract; if false, inspect raw failures and repair only the failed primitive.\n", encoding="utf-8")


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-time", required=True)
    ap.add_argument("--backup-commit", required=True)
    ap.add_argument("--backup-package-sha256", required=True)
    ap.add_argument("--i-accept-v34c-zero-solve-contract-preflight", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        started = now()
        write_json(RUN_DIR / "run_started.json", {"started_utc": started.isoformat(), "pid": os.getpid(), "method": NAME, "backup_context": {"time": args.backup_time, "commit": args.backup_commit, "package_sha256": args.backup_package_sha256}, "solver_calls": 0, "plant_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
        # Reuse v34's already-authorized gates, but do not run v34 solves.
        class GateArgs:
            backup_time = args.backup_time
            backup_commit = args.backup_commit
            backup_package_sha256 = args.backup_package_sha256
            backup_package_bytes = 0
        gates = m.verify_astra_and_gates(GateArgs())
        _, stage1_runner, _ = m.v1d.import_legacy_modules()
        preflight = stage1_runner.runtime_preflight()
        if not preflight.get("passed"):
            raise ContractError(f"legacy runtime preflight failed: {preflight}")
        stage1_runner.base.v1.latency_verify()
        term_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
        terminals, terminal_receipts = stage1_runner.load_terminal_grid(term_protocol["terminal_grid_readiness_reused_from_v1"])
        contexts = load_contexts_strict()
        schedule = m.make_schedule(contexts)
        if len(schedule) != EXPECTED_INTERCEPTS:
            raise ContractError(f"schedule length {len(schedule)} != {EXPECTED_INTERCEPTS}")
        write_json(RUN_DIR / "frozen_schedule.json", {"contexts": [{k: v for k, v in c.items() if k != "case_snapshot"} for c in contexts], "schedule": schedule, "controlled_no_solve_intercepts_expected": EXPECTED_INTERCEPTS, "solver_calls": 0, "plant_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
        cby = {c["context_id"]: c for c in contexts}
        rows: List[Dict[str, Any]] = []
        for row in schedule:
            r = preflight_arm(row, cby[str(row["context_id"])], terminals)
            rows.append(r)
            progress = {"rows_done": len(rows), "rows_expected": EXPECTED_INTERCEPTS, "controlled_no_solve_intercepts": sum(int(x.get("controlled_no_solve_intercepts", 0)) for x in rows), "lower_level_solver_calls": 0, "last": {"context_id": r["context_id"], "horizon": r["horizon"], "terminal_mode": r["terminal_mode"], "initialization": r["initialization"], "vf4_ok": r["vf4_check"].get("ok"), "lterm_ok": r["lterm_preflight"].get("ok")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
            write_json(RUN_DIR / "progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
        analysis = analyze(rows)
        created = now()
        raw = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_zero_solve_contract_preflight_not_validation_not_test",
            "hypothesis_frozen": "The v34 diagnostic chain can be made auditable without spending the 24-call solver budget: contexts, terminal contracts, typed initial guesses, nonterminal opt_p invariants, and four-argument vf_fun calls are verifiable under a controlled no-solve intercept.",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_declared": {"lower_level_solver_calls": 0, "plant_steps": 0, "training_or_refit": 0, "controlled_no_solve_intercepts": EXPECTED_INTERCEPTS, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "budget_actual": {"lower_level_solver_calls": 0, "plant_steps": 0, "training_or_refit": 0, "controlled_no_solve_intercepts": int(sum(int(r.get("controlled_no_solve_intercepts", 0)) for r in rows)), "preflight_rows": len(rows), "validation64_episodes": 0, "sealed_test_episodes": 0},
            "gates": gates,
            "runtime_preflight": preflight,
            "terminal_receipts": {str(k): v for k, v in terminal_receipts.items()},
            "semantic_coefficients_status": semantic_coefficients_status(),
            "contexts": [{k: v for k, v in c.items() if k != "case_snapshot"} for c in contexts],
            "schedule": schedule,
            "preflight_rows": rows,
            "analysis": analysis,
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
            "hashes_input": {rel(p): sha256(p) for p in [Path(__file__).resolve(), Path(m.__file__).resolve(), m.ANALYSIS_READY, m.ASTRA_REPORT, m.TASK1_DONE, m.TASK1_TERMINAL_CSV, m.D33_DONE] if p.exists()},
            "interpretation_limits": ["0-solve contract preflight only", "not a solver-result mechanism attribution", "not population validation", "not validation64", "not sealed test"],
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        write_json(BACKUP_REQ, {"request": "backup_after_v34c_contract_preflight", "created_utc": created.isoformat(), "backup_required_before_v34c_24call_probe_or_more_unique_science": True, "reason": "new v34c code, zero-solve preflight evidence, arrays and docs", "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQ), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "lower_level_solver_calls": 0, "plant_steps": 0, "training_or_refit": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
        update_docs(raw)
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE, BACKUP_REQ, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md"]
        completed = {"status": "complete", "passed": bool(analysis["passed"]), "hard_pass": bool(analysis["hard_pass"]), "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": raw["budget_actual"], "headline": analysis["headline"], "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "backup_request": rel(BACKUP_REQ), "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}}
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "headline": analysis["headline"], "hard_pass": analysis["hard_pass"], "backup_request": rel(BACKUP_REQ)}, sort_keys=True), flush=True)
        return 0 if analysis["hard_pass"] else 2
    except Exception as exc:
        fail = {"status": "failed", "created_utc": now().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_zero_solve_contract_preflight_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_caps": {"lower_level_solver_calls": 0, "plant_steps": 0, "training_or_refit": 0}}
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(f"# v34c contract preflight failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR / 'failed.json')}\n\nNo validation64/sealed-test access; no lower-level solver calls were intended. Repair the listed primitive before any 24-call solver probe.\n", encoding="utf-8")
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
