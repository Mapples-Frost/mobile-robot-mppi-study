#!/usr/bin/env python3
"""v34i zero-solve objective contract localization (Opus E-prime/F-prime).

Repairs the v34h offline reconstruction instrument without spending the frozen
remaining objective-vs-basin solver calls.  Inputs are the already-opened v34g
one-cell arrays.  This script:

* verifies the active Opus plan and the supervisor-provided post-v34h backup
  proof;
* rebuilds the same H15/V15/canonical MPC object without solve/reset/plant;
* assigns saved opt_x/opt_p values label-by-label (typed CasADi indices) and
  records explicit structure/indexability booleans;
* compares _x[14], _x[15], _x[-1] conventions;
* re-runs the same 65 candidate objective formulas, now using direct struct
  indexing after typed assembly; and
* rechecks residual aliases with lb_opt_x/ub_opt_x and cons_lb/cons_ub.

No plant rollout, selector search/refit, training, validation64, or sealed/final
test access is performed.
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
for _p in (AWS_DIR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vehicle_true_variable_horizon_v34h_objective_localization_v0 as h  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34i_objective_contract_localization_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34i_objective_contract_localization.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34I_OBJECTIVE_CONTRACT_LOCALIZATION_{STAMP}.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
PLAN_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
OPUS_LATEST = ROOT / "docs/bohn2021_takeover/opus_lead/LATEST.md"
ACTIVE_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T101936Z_980a3f.md"
ACTIVE_REPORT_SHA = "b540dc06fdbd279b5ccd8379f62cbcfbf46f4e65c1fa04176cfc096f9f97552f"
ACTIVE_REQUEST = "execution-result:20260930T101852_6b9c3451"
V34G_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0_20260930T100824Z/raw.json"
V34H_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34h_objective_localization_v0_20260930T101852Z/completed.json"
POST_V34H_BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260930T102045_from_user_context_after_v34h_objective_localization.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
TARGET = h.TARGET_CELL
GATE_REL_TOL = 1e-6
RESIDUAL_TOL = 1e-5
MARKER = f"vehicle-v34i-objective-contract-localization-{STAMP}"
REQUEST_ID = f"v34i-objective-contract-localization-{STAMP}"


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
    hh = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hh.update(chunk)
    return hh.hexdigest()


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


def arr_hash(value: Any) -> Optional[str]:
    a = arr(value)
    if a.size == 0:
        return None
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    a = arr(value)
    if a.size != 1:
        return default
    out = float(a[0])
    return out if math.isfinite(out) else default


def parse_label(label: str) -> List[Any]:
    s = str(label).strip().strip("[]")
    out: List[Any] = []
    for p in s.split(","):
        p = p.strip().strip("'\"")
        if not p:
            continue
        if p.lstrip("-").isdigit():
            out.append(int(p))
        else:
            out.append(p)
    return out


def candidate_indices(parts: Sequence[Any]) -> List[Tuple[Any, ...]]:
    out: List[Tuple[Any, ...]] = []
    def add(seq: Sequence[Any]) -> None:
        tup = tuple(seq)
        if tup not in out:
            out.append(tup)
    add(parts)
    if parts and isinstance(parts[-1], int):
        add(parts[:-1])
    # If a semantic variable name is followed by a scalar suffix, the no-suffix
    # form is usually accepted by do-mpc DMStruct.
    for i, p in enumerate(parts):
        if isinstance(p, str) and i > 0 and p not in ("_x", "_u", "_z", "_eps", "_p", "_tvp", "_x0", "_u_prev", "_vf_weights", "_vf_biases"):
            add(parts[: i + 1])
            break
    return out


def safe_set(obj: Any, label: str, value: float, flat_index: int) -> Tuple[bool, str]:
    parts = parse_label(label)
    errors: List[str] = []
    for idx in candidate_indices(parts):
        try:
            obj[idx] = float(value)
            return True, repr(idx)
        except Exception as exc:
            if len(errors) < 3:
                errors.append(repr(exc))
        try:
            obj.__setitem__(idx, float(value))
            return True, repr(idx)
        except Exception as exc:
            if len(errors) < 3:
                errors.append(repr(exc))
    try:
        obj.master[int(flat_index)] = float(value)
        return True, f"master[{flat_index}]"
    except Exception as exc:
        errors.append("master:" + repr(exc))
    return False, "; ".join(errors[-4:])


def safe_get(obj: Any, *idx: Any) -> Tuple[bool, Any, Optional[str]]:
    try:
        return True, obj[idx], None
    except Exception as exc1:
        try:
            return True, obj.__getitem__(idx), None
        except Exception as exc2:
            return False, None, repr(exc2 if str(exc2) else exc1)


def get_labels(obj: Any) -> List[str]:
    try:
        return [str(x) for x in obj.labels()]
    except Exception:
        return []


def snapshot_labels(snapshot: Mapping[str, Any]) -> List[str]:
    lv = snapshot.get("labeled_values")
    if isinstance(lv, Mapping):
        return list(lv.keys())
    return []


def vector_from_snapshot(snapshot: Mapping[str, Any], name: str) -> np.ndarray:
    vals = snapshot.get("flat_values")
    if not isinstance(vals, list):
        raise ContractError(f"snapshot {name} lacks flat_values")
    out = np.asarray(vals, dtype=float).reshape(-1)
    if int(snapshot.get("flat_size", out.size)) != int(out.size):
        raise ContractError(f"snapshot {name} flat size mismatch")
    return out


def assign_snapshot_to_struct(obj: Any, snapshot: Mapping[str, Any], name: str) -> Dict[str, Any]:
    values = vector_from_snapshot(snapshot, name)
    raw_labels = snapshot_labels(snapshot)
    live_labels = get_labels(obj)
    labels = raw_labels if len(raw_labels) == values.size else live_labels
    if len(labels) != values.size:
        raise ContractError(f"{name}: no label list with length {values.size}; raw={len(raw_labels)} live={len(live_labels)}")
    failed: List[Dict[str, Any]] = []
    used_master_fallback = 0
    for i, (label, value) in enumerate(zip(labels, values)):
        ok, how = safe_set(obj, label, float(value), i)
        if not ok:
            failed.append({"i": i, "label": label, "error": how})
        elif how.startswith("master["):
            used_master_fallback += 1
    if failed:
        raise ContractError(f"{name}: label assignment failed for {len(failed)} entries; first={failed[:3]}")
    # Ensure the contiguous master is also synchronized to the exact recorded vector.
    try:
        obj.master = values.reshape((-1, 1))
    except Exception:
        try:
            obj.master = values.reshape(-1)
        except Exception:
            pass
    return {
        "name": name,
        "flat_size": int(values.size),
        "raw_label_count": len(raw_labels),
        "live_label_count": len(live_labels),
        "raw_live_label_match": bool(raw_labels and raw_labels == live_labels),
        "assigned_count": int(values.size),
        "failed_count": 0,
        "used_master_fallback_count": used_master_fallback,
        "raw_flat_hash": snapshot.get("flat_hash"),
        "post_flat_hash": arr_hash(obj),
        "hash_match": arr_hash(obj) == snapshot.get("flat_hash"),
        "first5_labels": labels[:5],
    }


def assign_saved_arrays(mpc: Any, raw: Mapping[str, Any]) -> Dict[str, Any]:
    post = (((raw.get("arm") or {}).get("solver_event") or {}).get("post") or {})
    meta = {
        "opt_x_num": assign_snapshot_to_struct(mpc.opt_x_num, post.get("complete_opt_x_num") or {}, "opt_x_num"),
        "opt_x_num_unscaled": assign_snapshot_to_struct(mpc.opt_x_num_unscaled, post.get("complete_opt_x_num_unscaled") or {}, "opt_x_num_unscaled"),
        "opt_p_num": assign_snapshot_to_struct(mpc.opt_p_num, post.get("complete_opt_p_num") or {}, "opt_p_num"),
    }
    g = vector_from_snapshot(post.get("complete_opt_g_num") or {}, "opt_g_num").reshape((-1, 1))
    mpc.opt_g_num = g
    ev = (raw.get("arm") or {}).get("solver_event") or {}
    mpc.opt_f_num = float(ev.get("objective_opt_f_num"))
    meta["opt_g_num"] = {"flat_size": int(g.size), "raw_flat_hash": (post.get("complete_opt_g_num") or {}).get("flat_hash"), "post_flat_hash": hashlib.sha256(np.ascontiguousarray(g.reshape(-1), dtype=np.float64).tobytes()).hexdigest(), "hash_match": hashlib.sha256(np.ascontiguousarray(g.reshape(-1), dtype=np.float64).tobytes()).hexdigest() == (post.get("complete_opt_g_num") or {}).get("flat_hash")}
    meta["solver_objective_assigned"] = float(mpc.opt_f_num)
    return meta


def part_size(obj: Any, top: str) -> Dict[str, Any]:
    ok, val, err = safe_get(obj, top)
    return {"top": top, "index_ok": ok, "flat_size": int(arr(val).size) if ok else None, "finite": bool(ok and arr(val).size >= 0 and np.all(np.isfinite(arr(val)))), "error": err}


def index_probe(obj: Any, name: str, idx: Tuple[Any, ...]) -> Dict[str, Any]:
    ok, val, err = safe_get(obj, *idx)
    a = arr(val) if ok else np.asarray([])
    return {"name": name, "idx": repr(idx), "ok": bool(ok), "flat_size": int(a.size) if ok else None, "finite": bool(ok and np.all(np.isfinite(a))), "first_values": a[:6].tolist() if ok else [], "error": err}


def assembly_validation(mpc: Any) -> Dict[str, Any]:
    part_rows = []
    for top in ("_x", "_u", "_eps"):
        a = part_size(mpc.opt_x_num, top)
        b = part_size(mpc.opt_x_num_unscaled, top)
        part_rows.append({"top": top, "opt_x_num": a, "opt_x_num_unscaled": b, "sizes_match": a.get("flat_size") == b.get("flat_size"), "both_index_ok": bool(a.get("index_ok") and b.get("index_ok"))})
    probes = [
        index_probe(mpc.opt_x_num_unscaled, "x_0_0_last", ("_x", 0, 0, -1)),
        index_probe(mpc.opt_x_num_unscaled, "x_0_0_node0", ("_x", 0, 0, 0)),
        index_probe(mpc.opt_x_num_unscaled, "x_14_0_last", ("_x", 14, 0, -1)),
        index_probe(mpc.opt_x_num_unscaled, "x_15_0_last", ("_x", 15, 0, -1)),
        index_probe(mpc.opt_x_num_unscaled, "u_0_0", ("_u", 0, 0)),
        index_probe(mpc.opt_x_num_unscaled, "eps_0_0", ("_eps", 0, 0)),
        index_probe(mpc.opt_p_num, "p_0", ("_p", 0)),
        index_probe(mpc.opt_p_num, "tvp_0", ("_tvp", 0)),
        index_probe(mpc.opt_p_num, "x0", ("_x0",)),
        index_probe(mpc.opt_p_num, "u_prev", ("_u_prev",)),
    ]
    required = [p for p in probes if p["name"] in ("x_0_0_last", "x_14_0_last", "x_15_0_last", "u_0_0", "eps_0_0", "p_0", "tvp_0", "x0", "u_prev")]
    return {"part_size_rows": part_rows, "index_probes": probes, "required_indexability_pass": all(bool(p.get("ok") and p.get("finite")) for p in required)}


def state_at(mpc: Any, k: int, colloc: int = -1) -> Any:
    ok, val, err = safe_get(mpc.opt_x_num_unscaled, "_x", int(k), 0, int(colloc))
    if not ok:
        raise ContractError(f"state_at k={k} colloc={colloc} failed: {err}")
    return val


def control_at(mpc: Any, k: int, scaled: bool = False) -> Any:
    src = mpc.opt_x_num if scaled else mpc.opt_x_num_unscaled
    ok, val, err = safe_get(src, "_u", int(k), 0)
    if not ok:
        raise ContractError(f"control_at k={k} scaled={scaled} failed: {err}")
    return val


def eps_at(mpc: Any, k: int) -> Any:
    ok, val, err = safe_get(mpc.opt_x_num_unscaled, "_eps", int(k), 0)
    if not ok:
        raise ContractError(f"eps_at k={k} failed: {err}")
    return val


def z_at(mpc: Any, k: int, colloc: int = -1) -> Any:
    ok, val, _ = safe_get(mpc.opt_x_num_unscaled, "_z", int(k), 0, int(colloc))
    if ok:
        return val
    ok, val, _ = safe_get(mpc.opt_x_num_unscaled, "_z", int(k), 0, 0)
    if ok:
        return val
    return np.zeros((0, 1), dtype=float)


def tvp_at(mpc: Any, k: int) -> Any:
    ok, val, err = safe_get(mpc.opt_p_num, "_tvp", int(k))
    if not ok:
        raise ContractError(f"tvp_at k={k} failed: {err}")
    return val


def p_scenario(mpc: Any, s: int = 0) -> Any:
    ok, val, err = safe_get(mpc.opt_p_num, "_p", int(s))
    if not ok:
        raise ContractError(f"p_scenario {s} failed: {err}")
    return val


def x0_param(mpc: Any) -> Any:
    ok, val, err = safe_get(mpc.opt_p_num, "_x0")
    if not ok:
        raise ContractError(f"x0_param failed: {err}")
    return val


def u_prev_param(mpc: Any) -> Any:
    ok, val, err = safe_get(mpc.opt_p_num, "_u_prev")
    if not ok:
        raise ContractError(f"u_prev_param failed: {err}")
    return val


def call_scalar(fn: Any, args: Tuple[Any, ...], label: str) -> float:
    try:
        val = fn(*args)
    except Exception as exc:
        raise ContractError(f"{label} call failed: {exc!r}")
    out = finite_float(val, None)
    if out is None:
        raise ContractError(f"{label} returned non-scalar/non-finite")
    return float(out)


def gamma_and_source(mpc: Any) -> Tuple[float, str]:
    return h.get_discount(mpc)


def n_scenarios(mpc: Any, H: int) -> List[int]:
    n_robust = int(getattr(mpc, "n_robust", 0) or 0)
    n_comb = int(getattr(mpc, "n_combinations", 1) or 1)
    return [int(n_comb ** min(k, n_robust)) for k in range(int(H) + 1)]


def parent_scenario(mpc: Any, k: int, s: int) -> int:
    try:
        return int(getattr(mpc, "scenario_tree", {}).get("parent_scenario")[int(k)][int(s)])
    except Exception:
        return 0


def branch_offset(mpc: Any, k: int, s: int) -> int:
    try:
        return int(getattr(mpc, "scenario_tree", {}).get("branch_offset")[int(k)][int(s)])
    except Exception:
        return 0


def lterm_value(mpc: Any, k: int, s: int, gamma: float, stage_rule: str, z_rule: str) -> float:
    if stage_rule == "current_v34_x0_then_node0":
        xk = x0_param(mpc) if k == 0 else state_at(mpc, k, 0)
    elif stage_rule == "x0_then_last_node":
        xk = x0_param(mpc) if k == 0 else state_at(mpc, k, -1)
    elif stage_rule == "author_last_node":
        xk = state_at(mpc, k, -1)
    elif stage_rule == "node0_all_k":
        xk = state_at(mpc, k, 0)
    else:
        raise ContractError("unknown stage_rule " + stage_rule)
    zk = z_at(mpc, k + 1, -1) if z_rule == "current_v34_k_plus_1_fallback" else z_at(mpc, k, -1)
    current = branch_offset(mpc, k, s) if int(getattr(mpc, "n_robust", 0) or 0) and k < int(getattr(mpc, "n_robust", 0) or 0) else s
    val = call_scalar(mpc.lterm_fun, (xk, control_at(mpc, k, scaled=False), zk, tvp_at(mpc, k), p_scenario(mpc, current)), f"lterm k={k} s={s}")
    ns = n_scenarios(mpc, int(getattr(mpc, "n_horizon", TARGET["horizon"])))
    omega = 1.0 / float(ns[k + 1] if k + 1 < len(ns) else 1)
    return float((gamma ** k) * omega * val)


def epsterm_value(mpc: Any, k: int, s: int, gamma: float) -> float:
    if not hasattr(mpc, "epsterm_fun"):
        return 0.0
    return float((gamma ** k) * call_scalar(mpc.epsterm_fun, (eps_at(mpc, k),), f"epsterm k={k} s={s}"))


def rterm_value(mpc: Any, k: int, s: int, gamma: float) -> float:
    if not hasattr(mpc, "rterm_factor"):
        return 0.0
    u = arr(control_at(mpc, k, scaled=True)).reshape(-1)
    if k == 0:
        prev = arr(u_prev_param(mpc)).reshape(-1) / np.maximum(1e-300, arr(getattr(mpc, "_u_scaling", np.ones_like(u))).reshape(-1))
    else:
        prev = arr(safe_get(mpc.opt_x_num, "_u", k - 1, parent_scenario(mpc, k, s))[1]).reshape(-1)
    factors = arr(mpc.rterm_factor.cat).reshape(-1)
    n = min(len(u), len(prev), len(factors))
    if n == 0:
        return 0.0
    return float((gamma ** k) * np.sum(factors[:n] * (u[:n] - prev[:n]) ** 2))


def p_excluding_n_horizon(mpc: Any, scenario: int = 0) -> Tuple[np.ndarray, List[str]]:
    pfull = arr(p_scenario(mpc, scenario)).reshape(-1)
    labels = [str(x) for x in mpc.model._p.labels()]
    keep = [i for i, lab in enumerate(labels) if "n_horizon" not in lab]
    names = [labels[i].strip("[]").split(",")[0] for i in keep]
    return pfull[keep].reshape((-1, 1)), names


def terminal_value(mpc: Any, H: int, gamma: float, terminal_rule: str, discount_rule: str) -> Tuple[float, float, Dict[str, Any]]:
    if terminal_rule == "current_v34_node0":
        xh = state_at(mpc, H, 0)
    elif terminal_rule == "author_last_node":
        xh = state_at(mpc, H, -1)
    elif terminal_rule == "node3_explicit":
        xh = state_at(mpc, H, 3)
    elif terminal_rule == "h_minus_1_last_node":
        xh = state_at(mpc, H - 1, -1)
    else:
        raise ContractError("unknown terminal_rule " + terminal_rule)
    p_keep, p_names = p_excluding_n_horizon(mpc, 0)
    vf = call_scalar(mpc.vf_fun, (arr(xh).reshape((-1, 1)), p_keep, mpc.vf.weights_num, mpc.vf.biases_num), "vf_fun4")
    if discount_rule == "gamma_pow_H":
        factor = gamma ** H
    elif discount_rule == "gamma_pow_H_minus_1":
        factor = gamma ** (H - 1)
    elif discount_rule == "n_horizon_parameter_or_H":
        factor = gamma ** H
        labels = [str(x) for x in mpc.model._p.labels()]
        idxs = [i for i, lab in enumerate(labels) if "n_horizon" in lab]
        if idxs:
            pf = arr(p_scenario(mpc, 0)).reshape(-1)
            if idxs[0] < pf.size:
                factor = gamma ** float(pf[idxs[0]])
    else:
        raise ContractError("unknown discount_rule " + discount_rule)
    return float(vf), float(factor * vf), {"terminal_x_rule": terminal_rule, "discount_rule": discount_rule, "discount_factor_applied": float(factor), "vf_fun_n_in": int(mpc.vf_fun.n_in()) if hasattr(mpc.vf_fun, "n_in") else None, "p_names_excluding_n_horizon": p_names, "x_vector": arr(xh).tolist()}


def formula_components(mpc: Any, H: int, stage_rule: str, terminal_rule: str, inc_eps: bool, inc_r: bool, discount_rule: str, z_rule: str) -> Dict[str, Any]:
    gamma, gsrc = gamma_and_source(mpc)
    stage_terms: List[float] = []
    eps_terms: List[float] = []
    r_terms: List[float] = []
    ns = n_scenarios(mpc, H)
    for k in range(H):
        for s in range(int(ns[k])):
            stage_terms.append(lterm_value(mpc, k, s, gamma, stage_rule, z_rule))
            if inc_eps:
                eps_terms.append(epsterm_value(mpc, k, s, gamma))
            if inc_r:
                r_terms.append(rterm_value(mpc, k, s, gamma))
    vf_raw, term_disc, tmeta = terminal_value(mpc, H, gamma, terminal_rule, discount_rule)
    stage_total = float(math.fsum(stage_terms))
    eps_total = float(math.fsum(eps_terms))
    r_total = float(math.fsum(r_terms))
    total = float(stage_total + eps_total + r_total + term_disc)
    solver = finite_float(getattr(mpc, "opt_f_num", None), None)
    abs_err = None if solver is None else float(abs(total - solver))
    rel_err = None if solver is None else float(abs_err / max(1.0, abs(solver)))
    return {"stage_x_rule": stage_rule, "terminal_x_rule": terminal_rule, "z_rule": z_rule, "include_eps": bool(inc_eps), "include_rterm": bool(inc_r), "discount_rule": discount_rule, "gamma": gamma, "gamma_source": gsrc, "stage_lterm_total": stage_total, "epsterm_total": eps_total, "input_regularization_total": r_total, "terminal_vf_raw": vf_raw, "terminal_discounted": term_disc, "total": total, "solver_objective": solver, "absolute_error": abs_err, "relative_error": rel_err, "terminal_meta": tmeta, "stage_first3": stage_terms[:3], "eps_executed": bool(inc_eps), "rterm_executed": bool(inc_r), "eps_nonzero_count": int(sum(abs(x) > 1e-12 for x in eps_terms)), "rterm_nonzero_count": int(sum(abs(x) > 1e-12 for x in r_terms))}


def enumerate_candidates(mpc: Any, H: int) -> Dict[str, Any]:
    specs: List[Tuple[str, str, bool, bool, str, str]] = [("current_v34_x0_then_node0", "current_v34_node0", False, False, "gamma_pow_H", "current_v34_k_plus_1_fallback")]
    for sr in ["current_v34_x0_then_node0", "x0_then_last_node", "author_last_node", "node0_all_k"]:
        for tr in ["current_v34_node0", "author_last_node", "node3_explicit", "h_minus_1_last_node"]:
            for eps in [False, True]:
                for rt in [False, True]:
                    specs.append((sr, tr, eps, rt, "n_horizon_parameter_or_H", "same_k_last"))
    out: List[Dict[str, Any]] = []
    seen = set()
    for spec in specs:
        cid = f"stage={spec[0]}|term={spec[1]}|eps={spec[2]}|r={spec[3]}|discount={spec[4]}|z={spec[5]}"
        if cid in seen:
            continue
        seen.add(cid)
        try:
            row = formula_components(mpc, H, *spec)
            row["candidate_id"] = cid
            row["status"] = "evaluated"
            row["passed_1e_minus_6"] = row.get("relative_error") is not None and float(row["relative_error"]) <= GATE_REL_TOL
        except Exception as exc:
            row = {"candidate_id": cid, "status": "unavailable_due_to_index_or_formula_error", "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-4:], "include_eps": bool(spec[2]), "include_rterm": bool(spec[3]), "eps_executed": False, "rterm_executed": False, "passed_1e_minus_6": False}
        out.append(row)
    ok = [r for r in out if r.get("relative_error") is not None]
    ok.sort(key=lambda r: float(r["relative_error"]))
    passing = [r for r in ok if r.get("passed_1e_minus_6")]
    return {"candidate_count": len(out), "evaluated_candidate_count": len(ok), "passing_candidate_count": len(passing), "best_candidate": ok[0] if ok else None, "passing_candidates": passing, "top10_by_relative_error": ok[:10], "all_candidates": out}


def x_convention_slices(mpc: Any, H: int) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for k in [0, H - 1, H]:
        rows[f"x_{k}"] = {}
        for colloc in [-1, 0, 1, 2, 3]:
            try:
                rows[f"x_{k}"][str(colloc)] = arr(state_at(mpc, k, colloc)).tolist()
            except Exception as exc:
                rows[f"x_{k}"][str(colloc)] = {"error": repr(exc)}
    rows["comparisons"] = {
        "x_H_last_minus_x_H_node0_l1": None,
        "x_H_last_minus_x_H_node3_l1": None,
        "x_H_minus_1_last_minus_x_H_last_l1": None,
    }
    try:
        rows["comparisons"]["x_H_last_minus_x_H_node0_l1"] = float(np.sum(np.abs(arr(state_at(mpc, H, -1)) - arr(state_at(mpc, H, 0)))))
        rows["comparisons"]["x_H_last_minus_x_H_node3_l1"] = float(np.sum(np.abs(arr(state_at(mpc, H, -1)) - arr(state_at(mpc, H, 3)))))
        rows["comparisons"]["x_H_minus_1_last_minus_x_H_last_l1"] = float(np.sum(np.abs(arr(state_at(mpc, H - 1, -1)) - arr(state_at(mpc, H, -1)))))
    except Exception:
        pass
    return rows


def residual(values: Any, lb: Any, ub: Any) -> Optional[float]:
    x = arr(values); lo = arr(lb); hi = arr(ub)
    if x.size == 0 or lo.size != x.size or hi.size != x.size:
        return None
    return float(max(0.0, float(np.max(lo - x)), float(np.max(x - hi))))


def residual_localization(mpc: Any) -> Dict[str, Any]:
    names = ["opt_x_lb", "opt_x_ub", "opt_g_lb", "opt_g_ub", "lb_opt_x", "ub_opt_x", "cons_lb", "cons_ub", "opt_x_num", "opt_g_num"]
    summary = {n: {"exists": hasattr(mpc, n), "flat_size": int(arr(getattr(mpc, n)).size) if hasattr(mpc, n) else None, "flat_hash": arr_hash(getattr(mpc, n)) if hasattr(mpc, n) else None} for n in names}
    legacy = {"bound_residual_using_opt_x_lb_names": residual(getattr(mpc, "opt_x_num", []), getattr(mpc, "opt_x_lb", []), getattr(mpc, "opt_x_ub", [])), "constraint_residual_using_opt_g_lb_names": residual(getattr(mpc, "opt_g_num", []), getattr(mpc, "opt_g_lb", []), getattr(mpc, "opt_g_ub", []))}
    repaired = {"bound_residual_using_lb_opt_x_names": residual(getattr(mpc, "opt_x_num", []), getattr(mpc, "lb_opt_x", []), getattr(mpc, "ub_opt_x", [])), "constraint_residual_using_cons_lb_names": residual(getattr(mpc, "opt_g_num", []), getattr(mpc, "cons_lb", []), getattr(mpc, "cons_ub", []))}
    size_gate = summary["lb_opt_x"]["flat_size"] == summary["opt_x_num"]["flat_size"] and summary["ub_opt_x"]["flat_size"] == summary["opt_x_num"]["flat_size"] and summary["cons_lb"]["flat_size"] == summary["opt_g_num"]["flat_size"] and summary["cons_ub"]["flat_size"] == summary["opt_g_num"]["flat_size"]
    resid_gate = repaired["bound_residual_using_lb_opt_x_names"] is not None and repaired["constraint_residual_using_cons_lb_names"] is not None and float(repaired["bound_residual_using_lb_opt_x_names"]) <= RESIDUAL_TOL and float(repaired["constraint_residual_using_cons_lb_names"]) <= RESIDUAL_TOL
    return {"attribute_summary": summary, "legacy_names": legacy, "repaired_names": repaired, "repaired_size_gate": bool(size_gate), "repaired_residual_gate_non_none_and_le_1e-5": bool(resid_gate), "diagnosis": "capture residuals with lb_opt_x/ub_opt_x and cons_lb/cons_ub; preserve separate missing-vs-exceeded status strings"}


def verify_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    for p in [PLAN_READY, OPUS_LATEST, ACTIVE_REPORT, V34G_RAW, V34H_COMPLETED, POST_V34H_BACKUP_PROOF]:
        if not p.exists():
            raise ContractError(f"missing required input {rel(p)}")
    ready = read_json(PLAN_READY)
    if ready.get("request_id") != ACTIVE_REQUEST or ready.get("report_sha256") != ACTIVE_REPORT_SHA:
        raise ContractError("active Opus PLAN_READY does not match expected v34h follow-up plan")
    if sha256(ACTIVE_REPORT) != ACTIVE_REPORT_SHA:
        raise ContractError("active Opus report sha mismatch")
    latest = OPUS_LATEST.read_text(encoding="utf-8", errors="replace")
    if rel(ACTIVE_REPORT) not in latest:
        raise ContractError("Opus LATEST does not reference active report")
    proof = read_json(POST_V34H_BACKUP_PROOF)
    if proof.get("status") != "verified" or proof.get("backup_verified") is not True or int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("post-v34h backup proof is not verified/clean")
    pkgs = [p for p in (proof.get("packages_this_run") or []) if isinstance(p, Mapping)]
    if args.backup_commit != proof.get("commit") or not any(p.get("sha256") == args.backup_package_sha256 and int(p.get("bytes", -1)) == int(args.backup_package_bytes) for p in pkgs):
        raise ContractError("backup CLI args do not match post-v34h verified backup proof")
    v34h = read_json(V34H_COMPLETED)
    if v34h.get("status") != "complete" or (v34h.get("budget_actual") or {}).get("solver_calls") != 0:
        raise ContractError("v34h predecessor is not the expected zero-solve completed artifact")
    return {"plan_ready": rel(PLAN_READY), "active_report": rel(ACTIVE_REPORT), "active_report_sha256": ACTIVE_REPORT_SHA, "v34g_raw": rel(V34G_RAW), "v34g_raw_sha256": sha256(V34G_RAW), "v34h_completed": rel(V34H_COMPLETED), "v34h_completed_sha256": sha256(V34H_COMPLETED), "matched_backup_proof": rel(POST_V34H_BACKUP_PROOF), "matched_backup_proof_sha256": sha256(POST_V34H_BACKUP_PROOF), "backup_args": {"time": args.backup_time, "commit": args.backup_commit, "package_sha256": args.backup_package_sha256, "package_bytes": args.backup_package_bytes}}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["rank", "candidate_id", "status", "relative_error", "absolute_error", "total", "solver_objective", "stage_lterm_total", "epsterm_total", "input_regularization_total", "terminal_discounted", "terminal_vf_raw", "include_eps", "include_rterm", "eps_executed", "rterm_executed", "passed_1e_minus_6", "error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            w.writerow({k: clean(i if k == "rank" else r.get(k)) for k in fields})


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


def write_outputs(raw: Mapping[str, Any]) -> None:
    obj = raw["objective_localization"]
    best = obj.get("best_candidate") or {}
    res = raw["residual_localization"]
    lines = [
        "# v34i objective contract localization",
        "",
        f"UTC: `{raw['created_utc']}`. Zero-solve/zero-plant Opus E′/F′ diagnostic from already-opened v34g arrays.",
        "",
        "## Gates",
        f"- G-E objective reconstruction pass: `{raw['gates']['G_E_objective_reconstruction_pass']}`",
        f"- assembly/indexability pass: `{raw['gates']['assembly_indexability_pass']}`",
        f"- G-F residual alias pass: `{raw['gates']['G_F_residual_alias_offline_pass']}`",
        f"- both/hard pass: `{raw['gates']['both_pass']}`",
        "",
        "## Best candidate",
        f"- candidate: `{best.get('candidate_id')}`",
        f"- total: `{best.get('total')}`; solver: `{best.get('solver_objective')}`; abs_error: `{best.get('absolute_error')}`; rel_error: `{best.get('relative_error')}`",
        f"- components: stage `{best.get('stage_lterm_total')}`, eps `{best.get('epsterm_total')}`, rterm `{best.get('input_regularization_total')}`, terminal `{best.get('terminal_discounted')}`",
        f"- evaluated candidates: `{obj.get('evaluated_candidate_count')}` / `{obj.get('candidate_count')}`; passing: `{obj.get('passing_candidate_count')}`",
        "",
        "## Residual aliases",
        f"- legacy residuals: `{res.get('legacy_names')}`",
        f"- repaired residuals: `{res.get('repaired_names')}`",
        "",
        "No remaining Task-C solver calls, plant steps, validation64 episodes, sealed-test episodes, selector refit/search, or training were run.",
        f"Artifacts: `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'candidate_residuals.csv')}`, `{rel(RUN_DIR/'assembly_bool_table.csv')}`, `{rel(RUN_DIR/'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (RUN_DIR / "assembly_bool_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "ok", "finite", "flat_size", "sizes_match", "error", "first_values"])
        w.writeheader()
        for p in raw["assembly_validation"]["index_probes"]:
            w.writerow({"section": "index_probe", "name": p.get("name"), "ok": p.get("ok"), "finite": p.get("finite"), "flat_size": p.get("flat_size"), "sizes_match": "", "error": p.get("error"), "first_values": json.dumps(p.get("first_values"))})
        for p in raw["assembly_validation"]["part_size_rows"]:
            w.writerow({"section": "part_size", "name": p.get("top"), "ok": p.get("both_index_ok"), "finite": "", "flat_size": json.dumps({"opt_x_num": p.get("opt_x_num", {}).get("flat_size"), "opt_x_num_unscaled": p.get("opt_x_num_unscaled", {}).get("flat_size")}), "sizes_match": p.get("sizes_match"), "error": json.dumps({"opt_x_num": p.get("opt_x_num", {}).get("error"), "opt_x_num_unscaled": p.get("opt_x_num_unscaled", {}).get("error")}), "first_values": ""})
    block = f"""
<!-- {MARKER} -->
## v34i objective contract localization

UTC: {raw['created_utc']}. Executed active Opus E′/F′ zero-solve diagnostic after verified post-v34h backup. Budgets: solver_calls=0, plant_steps=0, env_reset/env_step after construction=0, training/refit=0, validation64=false, sealed_test=false. Assembly/indexability pass={raw['gates']['assembly_indexability_pass']}; G-E pass={raw['gates']['G_E_objective_reconstruction_pass']}; G-F pass={raw['gates']['G_F_residual_alias_offline_pass']}; both_pass={raw['gates']['both_pass']}. Best candidate `{best.get('candidate_id')}` rel_error={best.get('relative_error')} total={best.get('total')} solver={best.get('solver_objective')}; components stage={best.get('stage_lterm_total')}, eps={best.get('epsterm_total')}, rterm={best.get('input_regularization_total')}, terminal={best.get('terminal_discounted')}. Evidence: `{rel(RUN_DIR/'summary.md')}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'candidate_residuals.csv')}`, `{rel(RUN_DIR/'assembly_bool_table.csv')}`. Backup request: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)
    write_json(NEXT_REVIEW_REQUEST, {"request_id": REQUEST_ID, "created": raw["created_utc"], "status": "gate_evidence_ready" if raw["gates"]["both_pass"] else "analysis_requested", "trigger": "v34i E-prime/F-prime objective contract localization completed", "experiment_id": NAME, "active_lead_report": rel(ACTIVE_REPORT), "active_lead_report_sha256": ACTIVE_REPORT_SHA, "question": "Review v34i zero-solve localization. If both_pass is true, confirm the corrected v34j main 23/24-cell probe implementation should use the passing author objective formula and lb_opt_x/cons_lb residual aliases; if false, specify minimal further no-plant introspection.", "evidence_paths": [rel(RUN_DIR/"summary.md"), rel(RUN_DIR/"raw.json"), rel(RUN_DIR/"candidate_residuals.csv"), rel(RUN_DIR/"assembly_bool_table.csv"), rel(RUN_DIR/"completed.json"), rel(ACTIVE_REPORT), rel(RESPONSE_LOG)], "budget_actual": raw["budget_actual"], "gates": raw["gates"], "best_candidate": {k: best.get(k) for k in ["candidate_id", "relative_error", "absolute_error", "total", "solver_objective", "stage_lterm_total", "epsterm_total", "input_regularization_total", "terminal_discounted"]}, "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "plant_steps": 0, "training_or_refit": 0}, "backup_required_before_more_unique_science": rel(BACKUP_REQUEST)})
    write_json(BACKUP_REQUEST, {"request": "backup_after_v34i_objective_contract_localization", "created_utc": raw["created_utc"], "backup_required_before_more_unique_science": True, "reason": "new v34i zero-solve objective contract localization source/output/docs must be externally recoverable before remaining objective-vs-basin solver calls", "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(NEXT_REVIEW_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "new_solver_calls": 0, "new_plant_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "next_gate": "if both_pass true and backup verified, implement/run corrected v34j remaining fixed objective-vs-basin solver probe; otherwise return evidence to lead"})
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v34i objective contract localization\n\nUTC: {raw['created_utc']}\n\nGates: {json.dumps(raw['gates'], sort_keys=True)}\n\nBest candidate: {json.dumps({k: best.get(k) for k in ['candidate_id','relative_error','absolute_error','total','solver_objective']}, sort_keys=True)}\n\nArtifacts: {rel(RUN_DIR/'summary.md')}, {rel(RUN_DIR/'raw.json')}, {rel(RUN_DIR/'candidate_residuals.csv')}, {rel(RUN_DIR/'assembly_bool_table.csv')}, {rel(RUN_DIR/'completed.json')}\n\nNext: verify external backup for {rel(BACKUP_REQUEST)}. If both_pass true, implement corrected v34j main probe with the passing author objective formula and residual aliases before spending remaining solver calls; otherwise await/return lead analysis.\n", encoding="utf-8")


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-time", required=True)
    ap.add_argument("--backup-commit", required=True)
    ap.add_argument("--backup-package-sha256", required=True)
    ap.add_argument("--backup-package-bytes", type=int, required=True)
    ap.add_argument("--i-accept-v34i-zero-solve-localization", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        started = now_utc()
        input_gates = verify_inputs(args)
        h.patch_no_solve_runtime()
        raw_v34g = read_json(V34G_RAW)
        mpc, build_meta = h.build_mpc_without_solve()
        assign_meta = assign_saved_arrays(mpc, raw_v34g)
        assembly = assembly_validation(mpc)
        slices = x_convention_slices(mpc, int(TARGET["horizon"]))
        obj = enumerate_candidates(mpc, int(TARGET["horizon"]))
        res = residual_localization(mpc)
        best = obj.get("best_candidate") or {}
        g_e = bool(best.get("relative_error") is not None and float(best.get("relative_error")) <= GATE_REL_TOL)
        g_f = bool(res.get("repaired_size_gate") and res.get("repaired_residual_gate_non_none_and_le_1e-5"))
        assembly_pass = bool(assembly.get("required_indexability_pass"))
        forbidden = build_meta.get("forbidden_call_counter", {})
        budget_actual = {"solver_calls": int(forbidden.get("solve", 0)), "plant_steps": 0, "env_reset_calls_after_construction": int(forbidden.get("env_reset", 0)), "env_step_calls_after_construction": int(forbidden.get("env_step", 0)), "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}
        if any(int(budget_actual[k]) != 0 for k in ["solver_calls", "plant_steps", "env_reset_calls_after_construction", "env_step_calls_after_construction", "new_training_or_gradient_steps", "selector_refits", "validation64_episodes", "sealed_test_episodes"]):
            raise ContractError("zero-solve/zero-plant budget violated: " + repr(budget_actual))
        created = now_utc()
        raw = {"created_utc": created.isoformat(), "started_utc": started.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "method": NAME, "classification": "development_IMPROVED_zero_solve_objective_contract_localization_not_validation_not_test", "active_lead": "claude-opus-5-5", "lead_report": rel(ACTIVE_REPORT), "lead_report_sha256": ACTIVE_REPORT_SHA, "stable_ids": ["A13_objective_reconstruction_contract_localization", "A12_registry_backup_schema_contract/residual-bounds-capture"], "hypothesis_frozen": "v34h failed because saved arrays were not assembled into the indexable DMStruct path; label-by-label typed assembly should let the original 65 objective candidates execute, and the exact author formula should match opt_f if the instrumentation is correct.", "selected_cell": TARGET, "input_gates": input_gates, "runtime_build_meta": build_meta, "assignment_meta": assign_meta, "assembly_validation": assembly, "x_convention_slices": slices, "objective_localization": obj, "residual_localization": res, "gates": {"assembly_indexability_pass": assembly_pass, "G_E_objective_reconstruction_pass": g_e, "G_F_residual_alias_offline_pass": g_f, "both_pass": bool(assembly_pass and g_e and g_f)}, "budget_declared": {"solver_call_cap": 0, "plant_steps": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "budget_actual": budget_actual, "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()}, "interpretation_limits": ["development-only", "already-opened v34g one-cell arrays", "zero new lower-level solver calls", "not validation64", "not sealed/final test", "no selector policy or closed-loop claim"]}
        write_json(RUN_DIR / "raw.json", raw)
        all_rows = list(obj.get("all_candidates") or [])
        all_rows.sort(key=lambda r: float(r.get("relative_error")) if r.get("relative_error") is not None else float("inf"))
        write_csv(RUN_DIR / "candidate_residuals.csv", all_rows)
        write_outputs(raw)
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE, BACKUP_REQUEST, NEXT_REVIEW_REQUEST, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv", POST_V34H_BACKUP_PROOF]
        completed = {"status": "complete", "passed": bool(raw["gates"]["both_pass"]), "hard_pass": bool(raw["gates"]["both_pass"]), "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": budget_actual, "headline": {"assembly_indexability_pass": assembly_pass, "G_E_objective_reconstruction_pass": g_e, "G_F_residual_alias_offline_pass": g_f, "both_pass": bool(assembly_pass and g_e and g_f), "best_candidate_id": best.get("candidate_id"), "best_relative_error": best.get("relative_error"), "best_absolute_error": best.get("absolute_error"), "best_total": best.get("total"), "solver_objective": best.get("solver_objective"), "evaluated_candidate_count": obj.get("evaluated_candidate_count"), "passing_candidate_count": obj.get("passing_candidate_count"), "residual_repaired": res.get("repaired_names"), "residual_legacy": res.get("legacy_names")}, "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "candidate_table": rel(RUN_DIR / "candidate_residuals.csv"), "assembly_table": rel(RUN_DIR / "assembly_bool_table.csv"), "backup_request": rel(BACKUP_REQUEST), "next_review_request_id": REQUEST_ID, "hashes": hash_existing(files)}
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "candidate_table": rel(RUN_DIR / "candidate_residuals.csv"), "assembly_table": rel(RUN_DIR / "assembly_bool_table.csv"), "headline": completed["headline"], "backup_request": rel(BACKUP_REQUEST)}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        fail = {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_zero_solve_objective_contract_localization_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_caps": {"solver_call_cap": 0, "plant_steps": 0, "env_reset_calls_after_construction": 0}}
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(f"# v34i objective contract localization failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR/'failed.json')}\n\nNo validation64 or sealed test access was requested. Do not spend remaining Task-C solver calls until repaired.\n", encoding="utf-8")
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
