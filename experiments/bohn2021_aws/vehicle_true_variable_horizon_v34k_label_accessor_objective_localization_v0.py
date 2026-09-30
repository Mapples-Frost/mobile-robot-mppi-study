#!/usr/bin/env python3
"""v34k label-accessor objective localization (zero solve / zero plant).

Follow-up operational diagnostic after v34i showed that the saved vectors hash
matched but do-mpc partial indexing such as opt_x['_x', k, 0, -1] is not
indexable in the rebuilt object.  This script does not change the 65 objective
candidate formulas; it changes only the accessor: values are recovered from the
full DMStruct labels (e.g. ['_x', k, 0, c, 'theta', 0]) and assembled into the
vectors expected by lterm_fun / vf_fun.

No solver calls, plant rollouts, validation64, sealed/final test, selector
search/refit, or training are performed.
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

import vehicle_true_variable_horizon_v34i_objective_contract_localization_v0 as i  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34k_label_accessor_objective_localization.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34K_LABEL_ACCESSOR_OBJECTIVE_LOCALIZATION_{STAMP}.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
V34I_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34i_objective_contract_localization_v0_20260930T103304Z/completed.json"
V34I_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34i_objective_contract_localization_v0_20260930T103304Z/raw.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
TARGET = i.TARGET
GATE_REL_TOL = i.GATE_REL_TOL
RESIDUAL_TOL = i.RESIDUAL_TOL
MARKER = f"vehicle-v34k-label-accessor-objective-localization-{STAMP}"
REQUEST_ID = f"v34k-label-accessor-objective-localization-{STAMP}"


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


def finite_float(value: Any) -> Optional[float]:
    a = arr(value)
    if a.size != 1:
        return None
    out = float(a[0])
    return out if math.isfinite(out) else None


def parse_label(label: str) -> List[Any]:
    s = str(label).strip().strip("[]")
    out: List[Any] = []
    for p in s.split(","):
        p = p.strip().strip("'\"")
        if p == "":
            continue
        if p.lstrip("-").isdigit():
            out.append(int(p))
        else:
            out.append(p)
    return out


def get_labels(obj: Any) -> List[str]:
    try:
        return [str(x) for x in obj.labels()]
    except Exception:
        return []


def base_var_order_from_model(model_struct: Any, fallback_records: Sequence[Tuple[Tuple[Any, ...], float]], prefix: str, fixed: Tuple[Any, ...]) -> List[str]:
    out: List[str] = []
    for lab in get_labels(model_struct):
        parts = parse_label(lab)
        for p in parts:
            if isinstance(p, str) and not p.startswith("_") and p not in out:
                out.append(p)
                break
    if out:
        return out
    for parts, _ in fallback_records:
        if not parts or parts[0] != prefix:
            continue
        if len(parts) < 2 + len(fixed):
            continue
        if tuple(parts[1:1 + len(fixed)]) != tuple(fixed):
            continue
        for p in parts[1 + len(fixed):]:
            if isinstance(p, str) and not p.startswith("_") and p not in out:
                out.append(p)
    return out


class LabelAccessor:
    def __init__(self, mpc: Any) -> None:
        self.mpc = mpc
        self.scaled_records = self._records(mpc.opt_x_num)
        self.unscaled_records = self._records(mpc.opt_x_num_unscaled)
        self.p_records = self._records(mpc.opt_p_num)
        self.state_vars = base_var_order_from_model(mpc.model._x, self.unscaled_records, "_x", (0, 0, 0))
        self.input_vars = base_var_order_from_model(mpc.model._u, self.unscaled_records, "_u", (0, 0))
        self.p_vars = base_var_order_from_model(mpc.model._p, self.p_records, "_p", (0,))
        tvp_fallback_fixed = (0,)
        self.tvp_vars = base_var_order_from_model(mpc.model._tvp, self.p_records, "_tvp", tvp_fallback_fixed)
        try:
            eps_struct = mpc.model._eps
        except Exception:
            eps_struct = None
        self.eps_vars = base_var_order_from_model(eps_struct, self.unscaled_records, "_eps", (0, 0)) if eps_struct is not None else []
        if not self.eps_vars:
            self.eps_vars = self._unique_vars(self.unscaled_records, "_eps", (0, 0))
        self.max_colloc: Dict[Tuple[int, int], int] = {}
        for parts, _ in self.unscaled_records:
            if len(parts) >= 6 and parts[0] == "_x" and isinstance(parts[1], int) and isinstance(parts[2], int) and isinstance(parts[3], int):
                key = (int(parts[1]), int(parts[2]))
                self.max_colloc[key] = max(self.max_colloc.get(key, int(parts[3])), int(parts[3]))

    def _records(self, obj: Any) -> List[Tuple[Tuple[Any, ...], float]]:
        labels = get_labels(obj)
        values = arr(obj)
        if len(labels) != int(values.size):
            raise ContractError(f"label/value length mismatch: labels={len(labels)} values={values.size}")
        return [(tuple(parse_label(lab)), float(values[n])) for n, lab in enumerate(labels)]

    def _unique_vars(self, records: Sequence[Tuple[Tuple[Any, ...], float]], prefix: str, fixed: Tuple[Any, ...]) -> List[str]:
        out: List[str] = []
        for parts, _ in records:
            if not parts or parts[0] != prefix:
                continue
            if tuple(parts[1:1 + len(fixed)]) != tuple(fixed):
                continue
            for p in parts[1 + len(fixed):]:
                if isinstance(p, str) and not p.startswith("_") and p not in out:
                    out.append(p)
        return out

    def _scalar(self, records: Sequence[Tuple[Tuple[Any, ...], float]], exact: Tuple[Any, ...], fallback_prefix: str, numeric_prefix: Tuple[Any, ...], var: str) -> float:
        for parts, value in records:
            if parts == exact:
                return value
        hits: List[float] = []
        for parts, value in records:
            if not parts or parts[0] != fallback_prefix:
                continue
            if tuple(parts[1:1 + len(numeric_prefix)]) != tuple(numeric_prefix):
                continue
            if var in parts[1 + len(numeric_prefix):]:
                hits.append(value)
        if len(hits) == 1:
            return float(hits[0])
        raise ContractError(f"label scalar unavailable/ambiguous prefix={fallback_prefix} numeric={numeric_prefix} var={var} exact={exact} hits={len(hits)}")

    @staticmethod
    def col(vals: Sequence[float]) -> np.ndarray:
        return np.asarray([float(x) for x in vals], dtype=float).reshape((-1, 1))

    def state(self, k: int, s: int = 0, colloc: int = -1) -> np.ndarray:
        c = int(colloc)
        if c < 0:
            c = self.max_colloc.get((int(k), int(s)), c)
        vals = [self._scalar(self.unscaled_records, ("_x", int(k), int(s), c, var, 0), "_x", (int(k), int(s), c), var) for var in self.state_vars]
        return self.col(vals)

    def control(self, k: int, s: int = 0, scaled: bool = False) -> np.ndarray:
        records = self.scaled_records if scaled else self.unscaled_records
        vals = [self._scalar(records, ("_u", int(k), int(s), var, 0), "_u", (int(k), int(s)), var) for var in self.input_vars]
        return self.col(vals)

    def eps(self, k: int, s: int = 0) -> np.ndarray:
        vals = [self._scalar(self.unscaled_records, ("_eps", int(k), int(s), var, 0), "_eps", (int(k), int(s)), var) for var in self.eps_vars]
        return self.col(vals)

    def p(self, s: int = 0, exclude_n_horizon: bool = False) -> Tuple[np.ndarray, List[str]]:
        names = [v for v in self.p_vars if not (exclude_n_horizon and "n_horizon" in v)]
        vals = [self._scalar(self.p_records, ("_p", int(s), var, 0), "_p", (int(s),), var) for var in names]
        return self.col(vals), names

    def tvp(self, k: int) -> np.ndarray:
        vals = [self._scalar(self.p_records, ("_tvp", int(k), var, 0), "_tvp", (int(k),), var) for var in self.tvp_vars]
        return self.col(vals)

    def x0(self) -> np.ndarray:
        vals = [self._scalar(self.p_records, ("_x0", var, 0), "_x0", tuple(), var) for var in self.state_vars]
        return self.col(vals)

    def u_prev(self) -> np.ndarray:
        vals = [self._scalar(self.p_records, ("_u_prev", var, 0), "_u_prev", tuple(), var) for var in self.input_vars]
        return self.col(vals)

    def z(self, k: int, s: int = 0, colloc: int = -1) -> np.ndarray:
        # The vehicle model used here has no algebraic variables. If labels are
        # present, assemble them; otherwise pass a 0-length column to CasADi.
        z_vars: List[str] = []
        try:
            z_vars = base_var_order_from_model(self.mpc.model._z, self.unscaled_records, "_z", (int(k), int(s), 0))
        except Exception:
            z_vars = []
        if not z_vars:
            return np.zeros((0, 1), dtype=float)
        c = int(colloc)
        vals = [self._scalar(self.unscaled_records, ("_z", int(k), int(s), c, var, 0), "_z", (int(k), int(s), c), var) for var in z_vars]
        return self.col(vals)

    def diagnostics(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "state_vars": self.state_vars,
            "input_vars": self.input_vars,
            "eps_vars": self.eps_vars,
            "p_vars": self.p_vars,
            "tvp_vars_count": len(self.tvp_vars),
            "tvp_vars_first10": self.tvp_vars[:10],
            "max_colloc_first": {f"{k}:{s}": c for (k, s), c in sorted(self.max_colloc.items())[:8]},
            "label_counts": {"scaled_opt_x": len(self.scaled_records), "unscaled_opt_x": len(self.unscaled_records), "opt_p": len(self.p_records)},
            "indexability_without_partial_struct": {},
        }
        checks: List[Tuple[str, Any]] = [
            ("x0", lambda: self.x0()),
            ("u_prev", lambda: self.u_prev()),
            ("x_0_last", lambda: self.state(0, 0, -1)),
            ("x_0_node0", lambda: self.state(0, 0, 0)),
            ("x_14_last", lambda: self.state(14, 0, -1)),
            ("x_15_last", lambda: self.state(15, 0, -1)),
            ("u_0", lambda: self.control(0, 0, False)),
            ("eps_0", lambda: self.eps(0, 0)),
            ("p_0", lambda: self.p(0, False)[0]),
            ("tvp_0", lambda: self.tvp(0)),
        ]
        for name, fn in checks:
            try:
                val = fn()
                a = arr(val)
                out["indexability_without_partial_struct"][name] = {"ok": True, "flat_size": int(a.size), "finite": bool(np.all(np.isfinite(a))), "first_values": a[:8].tolist()}
            except Exception as exc:
                out["indexability_without_partial_struct"][name] = {"ok": False, "error": repr(exc)}
        out["label_accessor_pass"] = all(v.get("ok") and v.get("finite") for v in out["indexability_without_partial_struct"].values())
        return out


def call_scalar(fn: Any, args: Tuple[Any, ...], label: str) -> float:
    try:
        val = fn(*args)
    except Exception as exc:
        raise ContractError(f"{label} call failed: {exc!r}")
    out = finite_float(val)
    if out is None:
        raise ContractError(f"{label} returned non-scalar/non-finite")
    return float(out)


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


def lterm_value(mpc: Any, acc: LabelAccessor, k: int, s: int, gamma: float, stage_rule: str, z_rule: str) -> float:
    if stage_rule == "current_v34_x0_then_node0":
        xk = acc.x0() if k == 0 else acc.state(k, 0, 0)
    elif stage_rule == "x0_then_last_node":
        xk = acc.x0() if k == 0 else acc.state(k, 0, -1)
    elif stage_rule == "author_last_node":
        xk = acc.state(k, 0, -1)
    elif stage_rule == "node0_all_k":
        xk = acc.state(k, 0, 0)
    else:
        raise ContractError("unknown stage_rule " + stage_rule)
    zk = acc.z(k + 1, s, -1) if z_rule == "current_v34_k_plus_1_fallback" else acc.z(k, s, -1)
    current = branch_offset(mpc, k, s) if int(getattr(mpc, "n_robust", 0) or 0) and k < int(getattr(mpc, "n_robust", 0) or 0) else s
    pvec, _ = acc.p(current, False)
    val = call_scalar(mpc.lterm_fun, (xk, acc.control(k, s, False), zk, acc.tvp(k), pvec), f"lterm k={k} s={s}")
    ns = n_scenarios(mpc, int(getattr(mpc, "n_horizon", TARGET["horizon"])))
    omega = 1.0 / float(ns[k + 1] if k + 1 < len(ns) else 1)
    return float((gamma ** k) * omega * val)


def epsterm_value(mpc: Any, acc: LabelAccessor, k: int, s: int, gamma: float) -> float:
    if not hasattr(mpc, "epsterm_fun"):
        return 0.0
    return float((gamma ** k) * call_scalar(mpc.epsterm_fun, (acc.eps(k, s),), f"epsterm k={k} s={s}"))


def rterm_value(mpc: Any, acc: LabelAccessor, k: int, s: int, gamma: float) -> float:
    if not hasattr(mpc, "rterm_factor"):
        return 0.0
    factors = arr(getattr(mpc.rterm_factor, "cat", mpc.rterm_factor)).reshape(-1)
    if factors.size == 0:
        return 0.0
    u = arr(acc.control(k, s, True)).reshape(-1)
    prev = arr(acc.u_prev() if k == 0 else acc.control(k - 1, parent_scenario(mpc, k, s), True)).reshape(-1)
    n = min(len(u), len(prev), len(factors))
    if n == 0:
        return 0.0
    return float((gamma ** k) * np.sum(factors[:n] * (u[:n] - prev[:n]) ** 2))


def terminal_value(mpc: Any, acc: LabelAccessor, H: int, gamma: float, terminal_rule: str, discount_rule: str) -> Tuple[float, float, Dict[str, Any]]:
    if terminal_rule == "current_v34_node0":
        xh = acc.state(H, 0, 0)
    elif terminal_rule == "author_last_node":
        xh = acc.state(H, 0, -1)
    elif terminal_rule == "node3_explicit":
        xh = acc.state(H, 0, 3)
    elif terminal_rule == "h_minus_1_last_node":
        xh = acc.state(H - 1, 0, -1)
    else:
        raise ContractError("unknown terminal_rule " + terminal_rule)
    p_keep, p_names = acc.p(0, True)
    vf = call_scalar(mpc.vf_fun, (xh, p_keep, mpc.vf.weights_num, mpc.vf.biases_num), "vf_fun4")
    if discount_rule == "gamma_pow_H":
        factor = gamma ** H
    elif discount_rule == "gamma_pow_H_minus_1":
        factor = gamma ** (H - 1)
    elif discount_rule == "n_horizon_parameter_or_H":
        factor = gamma ** H
        try:
            p_full, p_all = acc.p(0, False)
            pf = arr(p_full)
            for idx, name in enumerate(p_all):
                if "n_horizon" in name and idx < pf.size:
                    factor = gamma ** float(pf[idx])
                    break
        except Exception:
            factor = gamma ** H
    else:
        raise ContractError("unknown discount_rule " + discount_rule)
    return float(vf), float(factor * vf), {"terminal_x_rule": terminal_rule, "discount_rule": discount_rule, "discount_factor_applied": float(factor), "p_names_excluding_n_horizon": p_names, "x_vector": arr(xh).tolist(), "vf_fun_n_in": int(mpc.vf_fun.n_in()) if hasattr(mpc.vf_fun, "n_in") else None}


def formula_components(mpc: Any, acc: LabelAccessor, H: int, stage_rule: str, terminal_rule: str, inc_eps: bool, inc_r: bool, discount_rule: str, z_rule: str) -> Dict[str, Any]:
    gamma, gsrc = i.gamma_and_source(mpc)
    stage_terms: List[float] = []
    eps_terms: List[float] = []
    r_terms: List[float] = []
    ns = n_scenarios(mpc, H)
    for k in range(H):
        for s in range(int(ns[k])):
            stage_terms.append(lterm_value(mpc, acc, k, s, gamma, stage_rule, z_rule))
            if inc_eps:
                eps_terms.append(epsterm_value(mpc, acc, k, s, gamma))
            if inc_r:
                r_terms.append(rterm_value(mpc, acc, k, s, gamma))
    vf_raw, term_disc, tmeta = terminal_value(mpc, acc, H, gamma, terminal_rule, discount_rule)
    stage_total = float(math.fsum(stage_terms))
    eps_total = float(math.fsum(eps_terms))
    r_total = float(math.fsum(r_terms))
    total = float(stage_total + eps_total + r_total + term_disc)
    solver = finite_float(getattr(mpc, "opt_f_num", None))
    abs_err = None if solver is None else float(abs(total - solver))
    rel_err = None if solver is None else float(abs_err / max(1.0, abs(solver)))
    return {"stage_x_rule": stage_rule, "terminal_x_rule": terminal_rule, "z_rule": z_rule, "include_eps": bool(inc_eps), "include_rterm": bool(inc_r), "discount_rule": discount_rule, "gamma": gamma, "gamma_source": gsrc, "stage_lterm_total": stage_total, "epsterm_total": eps_total, "input_regularization_total": r_total, "terminal_vf_raw": vf_raw, "terminal_discounted": term_disc, "total": total, "solver_objective": solver, "absolute_error": abs_err, "relative_error": rel_err, "terminal_meta": tmeta, "stage_first3": stage_terms[:3], "eps_executed": bool(inc_eps), "rterm_executed": bool(inc_r), "eps_nonzero_count": int(sum(abs(x) > 1e-12 for x in eps_terms)), "rterm_nonzero_count": int(sum(abs(x) > 1e-12 for x in r_terms))}


def enumerate_candidates(mpc: Any, acc: LabelAccessor, H: int) -> Dict[str, Any]:
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
            row = formula_components(mpc, acc, H, *spec)
            row["candidate_id"] = cid
            row["status"] = "evaluated"
            row["passed_1e_minus_6"] = row.get("relative_error") is not None and float(row["relative_error"]) <= GATE_REL_TOL
        except Exception as exc:
            row = {"candidate_id": cid, "status": "unavailable_due_to_label_accessor_or_formula_error", "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-4:], "include_eps": bool(spec[2]), "include_rterm": bool(spec[3]), "eps_executed": False, "rterm_executed": False, "passed_1e_minus_6": False}
        out.append(row)
    ok = [r for r in out if r.get("relative_error") is not None]
    ok.sort(key=lambda r: float(r["relative_error"]))
    passing = [r for r in ok if r.get("passed_1e_minus_6")]
    return {"candidate_count": len(out), "evaluated_candidate_count": len(ok), "passing_candidate_count": len(passing), "best_candidate": ok[0] if ok else None, "passing_candidates": passing, "top10_by_relative_error": ok[:10], "all_candidates": out}


def x_conventions(acc: LabelAccessor, H: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in [0, H - 1, H]:
        out[f"x_{k}"] = {}
        for c in [0, 1, 2, 3, -1]:
            try:
                out[f"x_{k}"][str(c)] = arr(acc.state(k, 0, c)).tolist()
            except Exception as exc:
                out[f"x_{k}"][str(c)] = {"error": repr(exc)}
    try:
        out["comparisons"] = {
            "x_H_last_minus_x_H_node0_l1": float(np.sum(np.abs(arr(acc.state(H, 0, -1)) - arr(acc.state(H, 0, 0))))),
            "x_H_last_minus_x_H_node3_l1": float(np.sum(np.abs(arr(acc.state(H, 0, -1)) - arr(acc.state(H, 0, 3))))),
            "x_H_minus_1_last_minus_x_H_last_l1": float(np.sum(np.abs(arr(acc.state(H - 1, 0, -1)) - arr(acc.state(H, 0, -1))))),
        }
    except Exception as exc:
        out["comparisons_error"] = repr(exc)
    return out


def residual_localization(mpc: Any) -> Dict[str, Any]:
    return i.residual_localization(mpc)


def hash_existing(paths: Iterable[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        try:
            if p.exists() and p.is_file():
                out[rel(p)] = sha256(p)
        except Exception:
            pass
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["rank", "candidate_id", "status", "relative_error", "absolute_error", "total", "solver_objective", "stage_lterm_total", "epsterm_total", "input_regularization_total", "terminal_discounted", "terminal_vf_raw", "include_eps", "include_rterm", "eps_executed", "rterm_executed", "passed_1e_minus_6", "error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n, r in enumerate(rows, start=1):
            w.writerow({k: clean(n if k == "rank" else r.get(k)) for k in fields})


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_outputs(raw: Mapping[str, Any]) -> None:
    obj = raw["objective_localization"]
    best = obj.get("best_candidate") or {}
    res = raw["residual_localization"]
    lines = [
        "# v34k label-accessor objective localization",
        "",
        f"UTC: `{raw['created_utc']}`. Zero-solve/zero-plant follow-up to v34i partial-struct indexing failure.",
        "",
        "## Gates",
        f"- label accessor pass: `{raw['gates']['label_accessor_pass']}`",
        f"- G-E objective reconstruction pass: `{raw['gates']['G_E_objective_reconstruction_pass']}`",
        f"- G-F residual alias pass: `{raw['gates']['G_F_residual_alias_offline_pass']}`",
        f"- hard/both pass: `{raw['gates']['both_pass']}`",
        "",
        "## Best candidate",
        f"- candidate: `{best.get('candidate_id')}`",
        f"- total: `{best.get('total')}`; solver: `{best.get('solver_objective')}`; abs_error: `{best.get('absolute_error')}`; rel_error: `{best.get('relative_error')}`",
        f"- components: stage `{best.get('stage_lterm_total')}`, eps `{best.get('epsterm_total')}`, rterm `{best.get('input_regularization_total')}`, terminal `{best.get('terminal_discounted')}`",
        f"- evaluated candidates: `{obj.get('evaluated_candidate_count')}` / `{obj.get('candidate_count')}`; passing: `{obj.get('passing_candidate_count')}`",
        "",
        "## Label accessor diagnosis",
        f"- state vars: `{raw['label_accessor_diagnostics'].get('state_vars')}`; input vars: `{raw['label_accessor_diagnostics'].get('input_vars')}`; p vars: `{raw['label_accessor_diagnostics'].get('p_vars')}`; tvp count: `{raw['label_accessor_diagnostics'].get('tvp_vars_count')}`",
        f"- partial do-mpc struct indexability from v34i remained false; this run uses full-label scalar assembly only.",
        "",
        "## Residual aliases",
        f"- legacy residuals: `{res.get('legacy_names')}`",
        f"- repaired residuals: `{res.get('repaired_names')}`",
        "",
        "No remaining Task-C solver calls, plant steps, validation64 episodes, sealed-test episodes, selector refit/search, or training were run.",
        f"Artifacts: `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'candidate_residuals.csv')}`, `{rel(RUN_DIR/'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    block = f"""
<!-- {MARKER} -->
## v34k label-accessor objective localization

UTC: {raw['created_utc']}. Ran a zero-solve/zero-plant operational diagnostic after v34i showed hash-correct saved arrays but failed partial do-mpc struct indexing. Budgets: solver_calls=0, plant_steps=0, env_reset/env_step after construction=0, training/refit=0, validation64=false, sealed_test=false. Label-accessor pass={raw['gates']['label_accessor_pass']}; G-E={raw['gates']['G_E_objective_reconstruction_pass']}; G-F={raw['gates']['G_F_residual_alias_offline_pass']}; both_pass={raw['gates']['both_pass']}. Best candidate `{best.get('candidate_id')}` rel_error={best.get('relative_error')} total={best.get('total')} solver={best.get('solver_objective')}; components stage={best.get('stage_lterm_total')}, eps={best.get('epsterm_total')}, rterm={best.get('input_regularization_total')}, terminal={best.get('terminal_discounted')}. Evidence: `{rel(RUN_DIR/'summary.md')}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'candidate_residuals.csv')}`, `{rel(RUN_DIR/'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)
    write_json(NEXT_REVIEW_REQUEST, {
        "request_id": REQUEST_ID,
        "created": raw["created_utc"],
        "status": "gate_evidence_ready" if raw["gates"]["both_pass"] else "analysis_requested",
        "trigger": "v34k label-accessor objective localization completed",
        "experiment_id": NAME,
        "active_lead_report": rel(i.ACTIVE_REPORT),
        "active_lead_report_sha256": i.ACTIVE_REPORT_SHA,
        "question": "Review v34k zero-solve label-accessor localization. If both_pass is true, confirm corrected v34j should use full-label accessors plus lb_opt_x/cons_lb aliases before spending remaining Task-C solver calls; if false, decide whether the remaining gap is formula-level or another accessor issue.",
        "evidence_paths": [rel(RUN_DIR/"summary.md"), rel(RUN_DIR/"raw.json"), rel(RUN_DIR/"candidate_residuals.csv"), rel(RUN_DIR/"completed.json"), rel(V34I_COMPLETED), rel(RESPONSE_LOG)],
        "budget_actual": raw["budget_actual"],
        "gates": raw["gates"],
        "best_candidate": {k: best.get(k) for k in ["candidate_id", "relative_error", "absolute_error", "total", "solver_objective", "stage_lterm_total", "epsterm_total", "input_regularization_total", "terminal_discounted"]},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "plant_steps": 0, "training_or_refit": 0},
        "backup_required_before_more_unique_science": rel(BACKUP_REQUEST),
    })
    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v34k_label_accessor_objective_localization",
        "created_utc": raw["created_utc"],
        "backup_required_before_more_unique_science": True,
        "reason": "new v34k source/output/docs after operational label-accessor objective localization must be externally recoverable before remaining objective-vs-basin solver calls",
        "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(NEXT_REVIEW_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
        "new_solver_calls": 0,
        "new_plant_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_gate": "if G-E and G-F pass and backup verified, implement corrected v34j main probe; otherwise return raw evidence to active lead for minimal next repair/introspection",
    })
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v34k label-accessor objective localization\n\nUTC: {raw['created_utc']}\n\nGates: {json.dumps(raw['gates'], sort_keys=True)}\n\nBest candidate: {json.dumps({k: best.get(k) for k in ['candidate_id','relative_error','absolute_error','total','solver_objective']}, sort_keys=True)}\n\nArtifacts: {rel(RUN_DIR/'summary.md')}, {rel(RUN_DIR/'raw.json')}, {rel(RUN_DIR/'candidate_residuals.csv')}, {rel(RUN_DIR/'completed.json')}\n\nNext: request/verify external backup for {rel(BACKUP_REQUEST)}. Do not spend Task-C solver calls unless G-E and G-F are both accepted by the active lead/gate.\n", encoding="utf-8")


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--i-accept-v34k-zero-solve-label-accessor", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        started = now_utc()
        for p in [i.V34G_RAW, V34I_COMPLETED, V34I_RAW, i.ACTIVE_REPORT, i.PLAN_READY]:
            if not p.exists():
                raise ContractError(f"missing required input {rel(p)}")
        v34i_completed = read_json(V34I_COMPLETED)
        if v34i_completed.get("status") != "complete" or (v34i_completed.get("budget_actual") or {}).get("solver_calls") != 0:
            raise ContractError("v34i predecessor is not the expected zero-solve completed artifact")
        i.h.patch_no_solve_runtime()
        raw_v34g = read_json(i.V34G_RAW)
        mpc, build_meta = i.h.build_mpc_without_solve()
        assign_meta = i.assign_saved_arrays(mpc, raw_v34g)
        accessor = LabelAccessor(mpc)
        acc_diag = accessor.diagnostics()
        slices = x_conventions(accessor, int(TARGET["horizon"]))
        obj = enumerate_candidates(mpc, accessor, int(TARGET["horizon"]))
        res = residual_localization(mpc)
        best = obj.get("best_candidate") or {}
        g_e = bool(best.get("relative_error") is not None and float(best.get("relative_error")) <= GATE_REL_TOL)
        g_f = bool(res.get("repaired_size_gate") and res.get("repaired_residual_gate_non_none_and_le_1e-5"))
        label_pass = bool(acc_diag.get("label_accessor_pass"))
        forbidden = build_meta.get("forbidden_call_counter", {})
        budget_actual = {"solver_calls": int(forbidden.get("solve", 0)), "plant_steps": 0, "env_reset_calls_after_construction": int(forbidden.get("env_reset", 0)), "env_step_calls_after_construction": int(forbidden.get("env_step", 0)), "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}
        if any(int(budget_actual[k]) != 0 for k in ["solver_calls", "plant_steps", "env_reset_calls_after_construction", "env_step_calls_after_construction", "new_training_or_gradient_steps", "selector_refits", "validation64_episodes", "sealed_test_episodes"]):
            raise ContractError("zero-solve/zero-plant budget violated: " + repr(budget_actual))
        created = now_utc()
        raw = {"created_utc": created.isoformat(), "started_utc": started.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "method": NAME, "classification": "development_IMPROVED_zero_solve_label_accessor_objective_localization_not_validation_not_test", "active_lead": "claude-opus-5-5", "lead_report": rel(i.ACTIVE_REPORT), "lead_report_sha256": i.ACTIVE_REPORT_SHA, "stable_ids": ["A13_objective_reconstruction_contract_localization", "A12_registry_backup_schema_contract/residual-bounds-capture"], "hypothesis_frozen": "v34i failed because rebuilt do-mpc DMStruct does not support partial hierarchical indexing; full-label scalar assembly should distinguish accessor failure from objective-formula mismatch without new solves.", "selected_cell": TARGET, "predecessor_v34i": {"completed": rel(V34I_COMPLETED), "raw": rel(V34I_RAW), "completed_sha256": sha256(V34I_COMPLETED), "raw_sha256": sha256(V34I_RAW), "hard_pass": v34i_completed.get("hard_pass")}, "input_v34g_raw_sha256": sha256(i.V34G_RAW), "runtime_build_meta": build_meta, "assignment_meta": assign_meta, "label_accessor_diagnostics": acc_diag, "x_convention_slices": slices, "objective_localization": obj, "residual_localization": res, "gates": {"label_accessor_pass": label_pass, "G_E_objective_reconstruction_pass": g_e, "G_F_residual_alias_offline_pass": g_f, "both_pass": bool(label_pass and g_e and g_f)}, "budget_declared": {"solver_call_cap": 0, "plant_steps": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "budget_actual": budget_actual, "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "interpretation_limits": ["development-only", "already-opened v34g one-cell arrays", "zero new lower-level solver calls", "not validation64", "not sealed/final test", "no selector policy or closed-loop claim"], "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()}}
        write_json(RUN_DIR / "raw.json", raw)
        rows = list(obj.get("all_candidates") or [])
        rows.sort(key=lambda r: float(r.get("relative_error")) if r.get("relative_error") is not None else float("inf"))
        write_csv(RUN_DIR / "candidate_residuals.csv", rows)
        write_outputs(raw)
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE, BACKUP_REQUEST, NEXT_REVIEW_REQUEST, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed = {"status": "complete", "passed": bool(raw["gates"]["both_pass"]), "hard_pass": bool(raw["gates"]["both_pass"]), "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": budget_actual, "headline": {"label_accessor_pass": label_pass, "G_E_objective_reconstruction_pass": g_e, "G_F_residual_alias_offline_pass": g_f, "both_pass": bool(label_pass and g_e and g_f), "best_candidate_id": best.get("candidate_id"), "best_relative_error": best.get("relative_error"), "best_absolute_error": best.get("absolute_error"), "best_total": best.get("total"), "solver_objective": best.get("solver_objective"), "evaluated_candidate_count": obj.get("evaluated_candidate_count"), "passing_candidate_count": obj.get("passing_candidate_count"), "residual_repaired": res.get("repaired_names"), "residual_legacy": res.get("legacy_names")}, "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "candidate_table": rel(RUN_DIR / "candidate_residuals.csv"), "backup_request": rel(BACKUP_REQUEST), "next_review_request_id": REQUEST_ID, "hashes": hash_existing(files)}
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "candidate_table": rel(RUN_DIR / "candidate_residuals.csv"), "headline": completed["headline"], "backup_request": rel(BACKUP_REQUEST)}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        fail = {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_zero_solve_label_accessor_objective_localization_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_caps": {"solver_call_cap": 0, "plant_steps": 0, "env_reset_calls_after_construction": 0}}
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(f"# v34k label-accessor objective localization failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR/'failed.json')}\n\nNo validation64 or sealed test access was requested. Do not spend remaining Task-C solver calls until repaired.\n", encoding="utf-8")
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
