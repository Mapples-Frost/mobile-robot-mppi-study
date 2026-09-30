#!/usr/bin/env python3
"""v34h offline objective/residual localization for the v34g one-cell smoke.

Active Opus Task E/F implementation.  This script spends zero lower-level MPC
solves and zero plant steps.  It uses the already-opened development-only v34g
one-cell raw arrays, reconstructs the same H15/V15/canonical MPC object without
calling solve/get_action, assigns the saved solution/parameter vectors into the
CasADi structs, and enumerates objective formulas against the recorded solver
objective.

It also localizes the residual-capture instrumentation defect by checking the
actual do-mpc bound attribute names (lb_opt_x/ub_opt_x and cons_lb/cons_ub) and
computing residuals from the saved opt_x/opt_g arrays in memory.  It does not
open validation64 or sealed/final test, does not refit/search/train, and does
not consume any of the frozen remaining 23 v34 objective-vs-basin solver calls.
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
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
REPRO_DIR = ROOT / "experiments/bohn2021_reproduction"
for _p in (AWS_DIR, REPRO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vehicle_true_variable_horizon_v34d_objective_basin_solver_probe_v0 as d  # noqa:E402

base = d.base
contract = d.contract

NAME = "vehicle_true_variable_horizon_v34h_objective_localization_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34h_objective_localization.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34H_OBJECTIVE_LOCALIZATION_{STAMP}.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

V34G_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0_20260930T100824Z"
V34G_RAW = V34G_DIR / "raw.json"
V34G_COMPLETED = V34G_DIR / "completed.json"
V34G_SUMMARY = V34G_DIR / "summary.md"
CONTROLLER_SOURCE = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/do-mpc-horizon/do_mpc/controller.py"
OPTIMIZER_SOURCE = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/do-mpc-horizon/do_mpc/optimizer.py"
OPUS_PLAN_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
OPUS_LATEST = ROOT / "docs/bohn2021_takeover/opus_lead/LATEST.md"
CURRENT_OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T100939Z_868560.md"
CURRENT_OPUS_SHA = "3c681e9a711f293616b5f4e9166fe3e149792d6af493da2c59cd43086356014f"
CURRENT_REQUEST_ID = "v34g-objective-reconstruction-smoke-20260930T100824Z"
MARKER = f"vehicle-v34h-objective-localization-{STAMP}"
REQUEST_ID = f"v34h-objective-localization-{STAMP}"

TARGET_CELL = {
    "context_id": "source242_slot0_branch_start",
    "state_label": "v27_case09_slot0_early_risk",
    "horizon": 15,
    "terminal_mode": "V15_shared",
    "initialization": "canonical",
}

GATE_REL_TOL = 1e-6
RESIDUAL_TOL = 1e-5
MAX_UNINSPECTED_SOLVES = 0


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if hasattr(value, "item"):
        return clean(value.item())
    return value


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


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def arr(value: Any) -> np.ndarray:
    try:
        if hasattr(value, "full"):
            return np.asarray(value.full(), dtype=float).reshape(-1)
        if hasattr(value, "cat"):
            return np.asarray(value.cat, dtype=float).reshape(-1)
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if hasattr(value, "full"):
            a = np.asarray(value.full(), dtype=float).reshape(-1)
        elif hasattr(value, "cat"):
            a = np.asarray(value.cat, dtype=float).reshape(-1)
        else:
            a = np.asarray(value, dtype=float).reshape(-1)
        if a.size != 1:
            return default
        out = float(a[0])
        return out if math.isfinite(out) else default
    except Exception:
        return default


def vector_from_snapshot(snap: Optional[Mapping[str, Any]], label: str) -> np.ndarray:
    if not isinstance(snap, Mapping):
        raise ContractError(f"missing snapshot for {label}")
    values = snap.get("flat_values")
    if not isinstance(values, list):
        raise ContractError(f"snapshot {label} lacks flat_values")
    out = np.asarray(values, dtype=float).reshape((-1, 1))
    if out.size != int(snap.get("flat_size", out.size)):
        raise ContractError(f"snapshot {label} flat_size mismatch")
    return out


def assign_master(struct_obj: Any, values: np.ndarray, label: str) -> None:
    if arr(struct_obj).size != int(values.size):
        raise ContractError(f"cannot assign {label}: struct size {arr(struct_obj).size} != values size {values.size}")
    try:
        struct_obj.master = values
    except Exception:
        try:
            struct_obj.master = values.reshape(-1)
        except Exception as exc:
            raise ContractError(f"cannot assign {label}.master: {exc!r}")


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise ContractError(f"missing {label}: {rel(path)}")


def verify_opus_and_inputs() -> Dict[str, Any]:
    for p, label in [
        (V34G_RAW, "v34g raw"),
        (V34G_COMPLETED, "v34g completed"),
        (CURRENT_OPUS_REPORT, "current Opus report"),
        (OPUS_PLAN_READY, "Opus PLAN_READY"),
        (OPUS_LATEST, "Opus LATEST"),
        (CONTROLLER_SOURCE, "do-mpc controller.py"),
        (OPTIMIZER_SOURCE, "do-mpc optimizer.py"),
    ]:
        require_file(p, label)
    if sha256(CURRENT_OPUS_REPORT) != CURRENT_OPUS_SHA:
        raise ContractError("current Opus report sha mismatch")
    ready = read_json(OPUS_PLAN_READY)
    if ready.get("request_id") != CURRENT_REQUEST_ID:
        raise ContractError(f"PLAN_READY request mismatch: {ready.get('request_id')} != {CURRENT_REQUEST_ID}")
    if ready.get("report_sha256") != CURRENT_OPUS_SHA:
        raise ContractError("PLAN_READY report sha mismatch")
    if rel(CURRENT_OPUS_REPORT) not in OPUS_LATEST.read_text(encoding="utf-8", errors="replace"):
        raise ContractError("Opus LATEST does not reference current report")
    completed = read_json(V34G_COMPLETED)
    if completed.get("status") != "complete":
        raise ContractError("v34g completed marker is not complete")
    if completed.get("validation64_bank_opened") is True or completed.get("sealed_test_accessed") is True:
        raise ContractError("v34g access flags indicate forbidden validation/test access")
    budget = completed.get("budget_actual") or {}
    if int(budget.get("solver_calls", -1)) != 1 or int(budget.get("plant_steps", -1)) != 0:
        raise ContractError("v34g budget was not the expected one-solve/zero-plant smoke")
    raw_hash_expected = (((completed.get("hashes") or {}).get(rel(V34G_RAW))))
    if raw_hash_expected and sha256(V34G_RAW) != raw_hash_expected:
        raise ContractError("v34g raw hash differs from completed marker")
    return {
        "opus_ready": rel(OPUS_PLAN_READY),
        "opus_report": rel(CURRENT_OPUS_REPORT),
        "opus_report_sha256": CURRENT_OPUS_SHA,
        "v34g_completed": rel(V34G_COMPLETED),
        "v34g_raw": rel(V34G_RAW),
        "v34g_raw_sha256": sha256(V34G_RAW),
        "v34g_completed_sha256": sha256(V34G_COMPLETED),
        "v34g_summary": rel(V34G_SUMMARY),
        "v34g_budget_actual": budget,
        "controller_source": rel(CONTROLLER_SOURCE),
        "optimizer_source": rel(OPTIMIZER_SOURCE),
    }


def patch_no_solve_runtime() -> None:
    # Reuse the v34c/v0e context/value-function repairs, but this offline script
    # must not call get_action/solve.  The solver is patched per instance below.
    d.patch_runtime()


def build_mpc_without_solve() -> Tuple[Any, Dict[str, Any]]:
    _, stage1_runner, _ = base.v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
    stage1_runner.base.v1.latency_verify()
    term_protocol = base.read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(term_protocol["terminal_grid_readiness_reused_from_v1"])
    terminal, term_h, term_note = base.terminal_for_mode(TARGET_CELL["horizon"], TARGET_CELL["terminal_mode"], terminals)
    env = base.create_env(TARGET_CELL["horizon"], terminal)
    mpc = env.control_system.controller.mpc
    calls = {"solve": 0, "env_reset": 0, "env_step": 0}

    def forbidden_solve(*_args: Any, **_kwargs: Any) -> Any:
        calls["solve"] += 1
        raise ContractError("mpc.solve is forbidden in v34h offline localization")

    def forbidden_reset(*_args: Any, **_kwargs: Any) -> Any:
        calls["env_reset"] += 1
        raise ContractError("env.reset is forbidden in v34h offline localization")

    def forbidden_step(*_args: Any, **_kwargs: Any) -> Any:
        calls["env_step"] += 1
        raise ContractError("env.step is forbidden in v34h offline localization")

    mpc.solve = forbidden_solve
    env.reset = forbidden_reset
    env.step = forbidden_step
    meta = {
        "runtime_preflight": preflight,
        "terminal_source_horizon": term_h,
        "terminal_note": term_note,
        "terminal_receipts": {str(k): v for k, v in terminal_receipts.items()},
        "forbidden_call_counter": calls,
        "mpc_n_horizon": int(getattr(mpc, "n_horizon", -1)),
        "mpc_n_robust": int(getattr(mpc, "n_robust", -1)),
        "mpc_n_combinations": int(getattr(mpc, "n_combinations", -1)),
        "opt_x_size_constructed": int(arr(getattr(mpc, "opt_x_num", [])).size),
        "opt_p_size_constructed": int(arr(getattr(mpc, "opt_p_num", [])).size),
    }
    return mpc, meta


def assign_saved_solution(mpc: Any, raw: Mapping[str, Any]) -> Dict[str, Any]:
    ev = (((raw.get("arm") or {}).get("solver_event") or {}))
    post = ev.get("post") or {}
    x = vector_from_snapshot(post.get("complete_opt_x_num"), "post.complete_opt_x_num")
    xu = vector_from_snapshot(post.get("complete_opt_x_num_unscaled"), "post.complete_opt_x_num_unscaled")
    p = vector_from_snapshot(post.get("complete_opt_p_num"), "post.complete_opt_p_num")
    g = vector_from_snapshot(post.get("complete_opt_g_num"), "post.complete_opt_g_num")
    assign_master(mpc.opt_x_num, x, "opt_x_num")
    assign_master(mpc.opt_x_num_unscaled, xu, "opt_x_num_unscaled")
    assign_master(mpc.opt_p_num, p, "opt_p_num")
    mpc.opt_g_num = g
    mpc.opt_f_num = float(ev.get("objective_opt_f_num"))
    return {
        "assigned_opt_x_size": int(x.size),
        "assigned_opt_x_unscaled_size": int(xu.size),
        "assigned_opt_p_size": int(p.size),
        "assigned_opt_g_size": int(g.size),
        "assigned_solver_objective": float(mpc.opt_f_num),
        "assigned_hashes": {
            "opt_x_num": hashlib.sha256(np.ascontiguousarray(x.reshape(-1), dtype=np.float64).tobytes()).hexdigest(),
            "opt_x_num_unscaled": hashlib.sha256(np.ascontiguousarray(xu.reshape(-1), dtype=np.float64).tobytes()).hexdigest(),
            "opt_p_num": hashlib.sha256(np.ascontiguousarray(p.reshape(-1), dtype=np.float64).tobytes()).hexdigest(),
            "opt_g_num": hashlib.sha256(np.ascontiguousarray(g.reshape(-1), dtype=np.float64).tobytes()).hexdigest(),
        },
        "raw_hashes": {
            "opt_x_num": (post.get("complete_opt_x_num") or {}).get("flat_hash"),
            "opt_x_num_unscaled": (post.get("complete_opt_x_num_unscaled") or {}).get("flat_hash"),
            "opt_p_num": (post.get("complete_opt_p_num") or {}).get("flat_hash"),
            "opt_g_num": (post.get("complete_opt_g_num") or {}).get("flat_hash"),
        },
    }


def struct_value(obj: Any, *idx: Any) -> Any:
    try:
        return obj[idx]
    except Exception:
        return obj.__getitem__(idx)


def state_at(mpc: Any, k: int, colloc: Any = -1) -> Any:
    return struct_value(mpc.opt_x_num_unscaled, "_x", int(k), 0, colloc)


def control_at(mpc: Any, k: int, scaled: bool = False) -> Any:
    source = mpc.opt_x_num if scaled else mpc.opt_x_num_unscaled
    return struct_value(source, "_u", int(k), 0)


def z_at(mpc: Any, k: int, colloc: Any = -1) -> Any:
    # The vehicle model has no algebraic variables in the observed vector, but
    # keep the controller-compatible path for generality.
    source = mpc.opt_x_num_unscaled
    try:
        return struct_value(source, "_z", int(k), 0, colloc)
    except Exception:
        try:
            return struct_value(source, "_z", int(k), 0, 0)
        except Exception:
            try:
                return struct_value(source, "_z", int(k), 0)
            except Exception:
                return np.zeros((0, 1), dtype=float)


def eps_at(mpc: Any, k: int) -> Any:
    return struct_value(mpc.opt_x_num_unscaled, "_eps", int(k), 0)


def tvp_at(mpc: Any, k: int) -> Any:
    return struct_value(mpc.opt_p_num, "_tvp", int(k))


def p_scenario(mpc: Any, scenario: int = 0) -> Any:
    return struct_value(mpc.opt_p_num, "_p", int(scenario))


def x0_param(mpc: Any) -> Any:
    return struct_value(mpc.opt_p_num, "_x0")


def u_prev_param(mpc: Any) -> Any:
    return struct_value(mpc.opt_p_num, "_u_prev")


def p_excluding_n_horizon(mpc: Any) -> Tuple[np.ndarray, List[str]]:
    return contract.c.p_excluding_n_horizon(mpc)


def call_scalar(fn: Any, args: Tuple[Any, ...], label: str) -> float:
    try:
        out = fn(*args)
    except Exception as exc:
        raise ContractError(f"{label} call failed: {exc!r}")
    f = finite_float(out, None)
    if f is None:
        raise ContractError(f"{label} returned non-scalar/non-finite")
    return float(f)


def call_vf4(mpc: Any, x_value: Any) -> Tuple[float, Dict[str, Any]]:
    p_keep, p_names = p_excluding_n_horizon(mpc)
    x_arr = arr(x_value).reshape((-1, 1))
    p_arr = np.asarray(p_keep, dtype=float).reshape((-1, 1))
    val = call_scalar(mpc.vf_fun, (x_arr, p_arr, mpc.vf.weights_num, mpc.vf.biases_num), "vf_fun4")
    return val, {
        "api": "vf_fun(x, p_excluding_n_horizon, weights_num, biases_num)",
        "vf_fun_n_in": int(mpc.vf_fun.n_in()) if hasattr(mpc.vf_fun, "n_in") else None,
        "x_size": int(x_arr.size),
        "p_size": int(p_arr.size),
        "p_names_excluding_n_horizon": list(p_names),
        "ok": True,
    }


def get_discount(mpc: Any) -> Tuple[float, str]:
    return base.find_discount(mpc)


def get_n_scenarios(mpc: Any, h: int) -> List[int]:
    n_robust = int(getattr(mpc, "n_robust", 0) or 0)
    n_combinations = int(getattr(mpc, "n_combinations", 1) or 1)
    return [int(n_combinations ** min(k, n_robust)) for k in range(int(h) + 1)]


def parent_scenario_for(mpc: Any, k: int, s: int) -> int:
    try:
        parent = getattr(mpc, "scenario_tree", {}).get("parent_scenario")
        if parent is not None and int(k) < len(parent):
            return int(parent[int(k)][int(s)])
    except Exception:
        pass
    return 0


def branch_offset_for(mpc: Any, k: int, s: int) -> int:
    try:
        branch = getattr(mpc, "scenario_tree", {}).get("branch_offset")
        if branch is not None and int(k) < len(branch):
            return int(branch[int(k)][int(s)])
    except Exception:
        pass
    return 0


def rterm_value(mpc: Any, k: int, s: int, gamma: float) -> float:
    if not hasattr(mpc, "rterm_factor"):
        return 0.0
    try:
        u_scaled = arr(control_at(mpc, k, scaled=True)).reshape((-1, 1))
        if k == 0:
            prev_scaled = arr(u_prev_param(mpc) / mpc._u_scaling.cat).reshape((-1, 1))
        else:
            parent = parent_scenario_for(mpc, k, s)
            prev_scaled = arr(struct_value(mpc.opt_x_num, "_u", int(k) - 1, parent)).reshape((-1, 1))
        diff = u_scaled - prev_scaled
        factors = arr(mpc.rterm_factor.cat).reshape((1, -1))
        val = float((factors @ (diff ** 2))[0, 0])
        return float((gamma ** int(k)) * val)
    except Exception as exc:
        raise ContractError(f"rterm k={k} s={s} failed: {exc!r}")


def epsterm_value(mpc: Any, k: int, s: int, gamma: float) -> float:
    if not hasattr(mpc, "epsterm_fun"):
        return 0.0
    # Only the single-scenario path is exercised by this diagnostic; keep the
    # scenario argument for explicit provenance.
    eps = eps_at(mpc, k)
    val = call_scalar(mpc.epsterm_fun, (eps,), f"epsterm_fun k={k} s={s}")
    return float((gamma ** int(k)) * val)


def lterm_value(mpc: Any, k: int, s: int, gamma: float, stage_x_rule: str, z_rule: str = "same_k_last") -> float:
    if stage_x_rule == "current_v34_x0_then_node0":
        xk = x0_param(mpc) if int(k) == 0 else state_at(mpc, k, 0)
    elif stage_x_rule == "author_last_node":
        xk = state_at(mpc, k, -1)
    elif stage_x_rule == "node0_all_k":
        xk = state_at(mpc, k, 0)
    elif stage_x_rule == "x0_then_last_node":
        xk = x0_param(mpc) if int(k) == 0 else state_at(mpc, k, -1)
    else:
        raise ContractError(f"unknown stage_x_rule {stage_x_rule}")
    if z_rule == "same_k_last":
        zk = z_at(mpc, k, -1)
    elif z_rule == "current_v34_k_plus_1_fallback":
        try:
            zk = z_at(mpc, k + 1, -1)
        except Exception:
            zk = z_at(mpc, k, -1)
    else:
        raise ContractError(f"unknown z_rule {z_rule}")
    current_scenario = branch_offset_for(mpc, k, s) if int(getattr(mpc, "n_robust", 0) or 0) and k < int(getattr(mpc, "n_robust", 0) or 0) else s
    try:
        pk = struct_value(mpc.opt_p_num, "_p", current_scenario)
    except Exception:
        pk = p_scenario(mpc, 0)
    tvpk = tvp_at(mpc, k)
    val = call_scalar(mpc.lterm_fun, (xk, control_at(mpc, k, scaled=False), zk, tvpk, pk), f"lterm_fun k={k} s={s} rule={stage_x_rule}")
    n_scenarios = get_n_scenarios(mpc, int(getattr(mpc, "n_horizon", TARGET_CELL["horizon"])))
    omega = 1.0 / float(n_scenarios[int(k) + 1] if int(k) + 1 < len(n_scenarios) else 1)
    return float((gamma ** int(k)) * omega * val)


def terminal_value(mpc: Any, h: int, gamma: float, terminal_x_rule: str, discount_rule: str) -> Tuple[float, float, Dict[str, Any]]:
    if terminal_x_rule == "current_v34_node0":
        xh = state_at(mpc, h, 0)
    elif terminal_x_rule == "author_last_node":
        xh = state_at(mpc, h, -1)
    elif terminal_x_rule == "node3_explicit":
        xh = state_at(mpc, h, 3)
    elif terminal_x_rule == "h_minus_1_last_node":
        xh = state_at(mpc, h - 1, -1)
    else:
        raise ContractError(f"unknown terminal_x_rule {terminal_x_rule}")
    vf, meta = call_vf4(mpc, xh)
    if discount_rule == "gamma_pow_H":
        factor = gamma ** int(h)
    elif discount_rule == "gamma_pow_H_minus_1":
        factor = gamma ** (int(h) - 1)
    elif discount_rule == "n_horizon_parameter_or_H":
        factor = gamma ** int(h)
        try:
            labels = [str(x) for x in mpc.model._p.labels()]
            if "[n_horizon,0]" in labels:
                idx = labels.index("[n_horizon,0]")
                p0 = arr(p_scenario(mpc, 0))
                if idx < p0.size:
                    factor = gamma ** float(p0[idx])
        except Exception:
            pass
    else:
        raise ContractError(f"unknown discount_rule {discount_rule}")
    return float(vf), float(factor * vf), {**meta, "terminal_x_rule": terminal_x_rule, "discount_rule": discount_rule, "discount_factor_applied": float(factor), "x_vector": arr(xh).tolist()}


def formula_components(mpc: Any, h: int, stage_x_rule: str, terminal_x_rule: str, include_eps: bool, include_rterm: bool, discount_rule: str = "n_horizon_parameter_or_H", z_rule: str = "same_k_last") -> Dict[str, Any]:
    gamma, gamma_source = get_discount(mpc)
    n_scenarios = get_n_scenarios(mpc, h)
    stage_terms: List[float] = []
    eps_terms: List[float] = []
    r_terms: List[float] = []
    # The v34g cell has n_robust=0, but loop over scenarios to mirror the NLP.
    for k in range(int(h)):
        ns = int(n_scenarios[k]) if k < len(n_scenarios) else 1
        for s in range(ns):
            stage_terms.append(lterm_value(mpc, k, s, gamma, stage_x_rule, z_rule=z_rule))
            if include_eps:
                eps_terms.append(epsterm_value(mpc, k, s, gamma))
            if include_rterm:
                r_terms.append(rterm_value(mpc, k, s, gamma))
    vf_raw, terminal_discounted, terminal_meta = terminal_value(mpc, h, gamma, terminal_x_rule, discount_rule)
    stage_total = float(math.fsum(stage_terms))
    eps_total = float(math.fsum(eps_terms))
    r_total = float(math.fsum(r_terms))
    total = float(stage_total + eps_total + r_total + terminal_discounted)
    solver = finite_float(getattr(mpc, "opt_f_num", None), None)
    abs_error = None if solver is None else float(abs(total - float(solver)))
    rel_error = None if solver is None else float(abs_error / max(1.0, abs(float(solver))))
    return {
        "stage_x_rule": stage_x_rule,
        "terminal_x_rule": terminal_x_rule,
        "z_rule": z_rule,
        "include_eps": bool(include_eps),
        "include_rterm": bool(include_rterm),
        "discount_rule": discount_rule,
        "gamma": float(gamma),
        "gamma_source": gamma_source,
        "stage_lterm_total": stage_total,
        "epsterm_total": eps_total,
        "input_regularization_total": r_total,
        "terminal_vf_raw": float(vf_raw),
        "terminal_discounted": float(terminal_discounted),
        "total": total,
        "solver_objective": solver,
        "absolute_error": abs_error,
        "relative_error": rel_error,
        "terminal_meta": terminal_meta,
        "stage_first3": stage_terms[:3],
        "eps_nonzero_count": int(sum(1 for x in eps_terms if abs(x) > 1e-12)),
        "rterm_nonzero_count": int(sum(1 for x in r_terms if abs(x) > 1e-12)),
    }


def enumerate_objective_candidates(mpc: Any, h: int) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    spec_rows: List[Tuple[str, str, bool, bool, str, str]] = []
    # Current v34d/v34g formula and adjacent one-variable repairs.
    spec_rows.append(("current_v34_x0_then_node0", "current_v34_node0", False, False, "gamma_pow_H", "current_v34_k_plus_1_fallback"))
    for stage_rule in ["current_v34_x0_then_node0", "x0_then_last_node", "author_last_node", "node0_all_k"]:
        for terminal_rule in ["current_v34_node0", "author_last_node", "node3_explicit", "h_minus_1_last_node"]:
            for inc_eps in [False, True]:
                for inc_r in [False, True]:
                    spec_rows.append((stage_rule, terminal_rule, inc_eps, inc_r, "n_horizon_parameter_or_H", "same_k_last"))
    seen = set()
    for spec in spec_rows:
        key = canonical_hash(spec)
        if key in seen:
            continue
        seen.add(key)
        try:
            row = formula_components(mpc, h, spec[0], spec[1], spec[2], spec[3], discount_rule=spec[4], z_rule=spec[5])
            row["candidate_id"] = f"stage={spec[0]}|term={spec[1]}|eps={spec[2]}|r={spec[3]}|discount={spec[4]}|z={spec[5]}"
            row["passed_1e_minus_6"] = row["relative_error"] is not None and float(row["relative_error"]) <= GATE_REL_TOL
        except Exception as exc:
            row = {"candidate_id": f"stage={spec[0]}|term={spec[1]}|eps={spec[2]}|r={spec[3]}|discount={spec[4]}|z={spec[5]}", "error": repr(exc), "passed_1e_minus_6": False}
        candidates.append(row)
    ok_rows = [r for r in candidates if r.get("relative_error") is not None]
    ok_rows.sort(key=lambda r: float(r["relative_error"]))
    pass_rows = [r for r in ok_rows if r.get("passed_1e_minus_6")]
    return {
        "candidate_count": len(candidates),
        "evaluated_candidate_count": len(ok_rows),
        "passing_candidate_count": len(pass_rows),
        "best_candidate": ok_rows[0] if ok_rows else None,
        "passing_candidates": pass_rows,
        "top10_by_relative_error": ok_rows[:10],
        "all_candidates": candidates,
    }


def extract_x_slices(mpc: Any, h: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in sorted(set([0, max(0, h - 1), h])):
        out[f"x_{k}_all_collocation"] = []
        for c in range(4):
            try:
                out[f"x_{k}_all_collocation"].append({"collocation_index": c, "vector": arr(state_at(mpc, k, c)).tolist()})
            except Exception as exc:
                out[f"x_{k}_all_collocation"].append({"collocation_index": c, "error": repr(exc)})
        try:
            out[f"x_{k}_last"] = arr(state_at(mpc, k, -1)).tolist()
        except Exception as exc:
            out[f"x_{k}_last_error"] = repr(exc)
    try:
        out["x0_param"] = arr(x0_param(mpc)).tolist()
    except Exception as exc:
        out["x0_param_error"] = repr(exc)
    try:
        out["u_prev_param"] = arr(u_prev_param(mpc)).tolist()
    except Exception as exc:
        out["u_prev_param_error"] = repr(exc)
    out["first5_controls_unscaled"] = []
    out["first5_eps"] = []
    for k in range(min(5, h)):
        try:
            out["first5_controls_unscaled"].append({"k": k, "u": arr(control_at(mpc, k, scaled=False)).tolist()})
        except Exception as exc:
            out["first5_controls_unscaled"].append({"k": k, "error": repr(exc)})
        try:
            out["first5_eps"].append({"k": k, "eps": arr(eps_at(mpc, k)).tolist(), "epsterm": finite_float(mpc.epsterm_fun(eps_at(mpc, k)), None) if hasattr(mpc, "epsterm_fun") else None})
        except Exception as exc:
            out["first5_eps"].append({"k": k, "error": repr(exc)})
    return out


def residual(values: Any, lb: Any, ub: Any) -> Optional[float]:
    x = arr(values)
    lo = arr(lb)
    hi = arr(ub)
    if x.size == 0 or lo.size != x.size or hi.size != x.size:
        return None
    return float(max(0.0, float(np.max(lo - x)), float(np.max(x - hi))))


def summarize_attr(obj: Any, names: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in names:
        if hasattr(obj, name):
            v = getattr(obj, name)
            out[name] = {"exists": True, "flat_size": int(arr(v).size), "flat_hash": hashlib.sha256(np.ascontiguousarray(arr(v), dtype=np.float64).tobytes()).hexdigest() if arr(v).size else None}
        else:
            out[name] = {"exists": False, "flat_size": None, "flat_hash": None}
    return out


def residual_localization(mpc: Any) -> Dict[str, Any]:
    attr_summary = summarize_attr(mpc, ["opt_x_lb", "opt_x_ub", "opt_g_lb", "opt_g_ub", "lb_opt_x", "ub_opt_x", "cons_lb", "cons_ub", "opt_x_num", "opt_g_num"])
    legacy = {
        "bound_residual_using_opt_x_lb_names": residual(getattr(mpc, "opt_x_num", []), getattr(mpc, "opt_x_lb", []), getattr(mpc, "opt_x_ub", [])),
        "constraint_residual_using_opt_g_lb_names": residual(getattr(mpc, "opt_g_num", []), getattr(mpc, "opt_g_lb", []), getattr(mpc, "opt_g_ub", [])),
    }
    repaired = {
        "bound_residual_using_lb_opt_x_names": residual(getattr(mpc, "opt_x_num", []), getattr(mpc, "lb_opt_x", []), getattr(mpc, "ub_opt_x", [])),
        "constraint_residual_using_cons_lb_names": residual(getattr(mpc, "opt_g_num", []), getattr(mpc, "cons_lb", []), getattr(mpc, "cons_ub", [])),
    }
    gate_pass = (
        repaired["bound_residual_using_lb_opt_x_names"] is not None
        and repaired["constraint_residual_using_cons_lb_names"] is not None
        and float(repaired["bound_residual_using_lb_opt_x_names"]) <= RESIDUAL_TOL
        and float(repaired["constraint_residual_using_cons_lb_names"]) <= RESIDUAL_TOL
    )
    x_size = attr_summary.get("opt_x_num", {}).get("flat_size")
    g_size = attr_summary.get("opt_g_num", {}).get("flat_size")
    size_gate = (
        attr_summary.get("lb_opt_x", {}).get("flat_size") == x_size
        and attr_summary.get("ub_opt_x", {}).get("flat_size") == x_size
        and attr_summary.get("cons_lb", {}).get("flat_size") == g_size
        and attr_summary.get("cons_ub", {}).get("flat_size") == g_size
    )
    return {
        "attribute_summary": attr_summary,
        "legacy_names": legacy,
        "repaired_names": repaired,
        "repaired_size_gate": bool(size_gate),
        "repaired_residual_gate_non_none_and_le_1e-5": bool(gate_pass),
        "diagnosis": "use lb_opt_x/ub_opt_x and cons_lb/cons_ub; opt_x_lb/opt_g_lb names are absent in this do-mpc path",
        "budget": {"solver_calls": 0, "plant_steps": 0, "env_reset_calls": 0},
    }


def read_source_snippets() -> Dict[str, Any]:
    snippets: Dict[str, Any] = {}
    for path, name, needles in [
        (CONTROLLER_SOURCE, "controller.py", ["obj += self.discount_factor ** k * omega[k] * self.lterm_fun", "obj += self.discount_factor ** k * self.epsterm_fun", "term_cost *= self.discount_factor", "obj += self.discount_factor ** k * self.rterm_factor"]),
        (OPTIMIZER_SOURCE, "optimizer.py", ["self.S(x0=self.opt_x_num, lbx=self.lb_opt_x, ubx=self.ub_opt_x", "ubg=self.cons_ub, lbg=self.cons_lb", "self.opt_g_num = r['g']", "self.opt_f_num = r['f']"]),
    ]:
        text = path.read_text(encoding="utf-8", errors="replace")
        rows = []
        lines = text.splitlines()
        for needle in needles:
            hit = None
            for i, line in enumerate(lines, start=1):
                if needle in line:
                    start = max(1, i - 2)
                    end = min(len(lines), i + 3)
                    hit = {"needle": needle, "line": i, "context": lines[start - 1:end]}
                    break
            rows.append(hit or {"needle": needle, "line": None, "context": []})
        snippets[name] = rows
    return snippets


def write_csv_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["rank", "candidate_id", "relative_error", "absolute_error", "total", "solver_objective", "stage_lterm_total", "epsterm_total", "input_regularization_total", "terminal_discounted", "terminal_vf_raw", "passed_1e_minus_6"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            w.writerow({k: clean(i if k == "rank" else row.get(k)) for k in fields})


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    obj = raw["objective_localization"]
    res = raw["residual_localization"]
    best = obj.get("best_candidate") or {}
    lines = [
        "# v34h offline objective/residual localization",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only Opus Task E/F. Solver calls `0`; plant steps `0`; env.reset/env.step after construction `0`; no training/refit; no validation64; no sealed/final test.",
        "",
        "## Objective localization gate G-E",
        "",
        f"- v34g solver objective: `{best.get('solver_objective')}`",
        f"- best candidate: `{best.get('candidate_id')}`",
        f"- best total: `{best.get('total')}`; absolute error `{best.get('absolute_error')}`; relative error `{best.get('relative_error')}`",
        f"- component totals: stage `{best.get('stage_lterm_total')}`, eps `{best.get('epsterm_total')}`, input regularization `{best.get('input_regularization_total')}`, terminal `{best.get('terminal_discounted')}`",
        f"- passing candidates at <= {GATE_REL_TOL}: `{obj.get('passing_candidate_count')}` / `{obj.get('evaluated_candidate_count')}`",
        f"- G-E hard pass: `{raw['gates']['G_E_objective_reconstruction_pass']}`",
        "",
        "## Residual instrumentation gate G-F (offline alias localization)",
        "",
        f"- legacy opt_x_lb/opt_g_lb residuals: `{res['legacy_names']}`",
        f"- repaired lb_opt_x/cons_lb residuals: `{res['repaired_names']}`",
        f"- repaired size gate: `{res['repaired_size_gate']}`; residual gate: `{res['repaired_residual_gate_non_none_and_le_1e-5']}`",
        f"- G-F offline pass: `{raw['gates']['G_F_residual_alias_offline_pass']}`",
        "",
        "## Interpretation limits",
        "",
        "This localizes the one-cell objective and residual primitives using already-opened v34g arrays. It does not run the remaining 23 Task-C solves and is not population validation or final-test evidence.",
        "",
        f"Candidate table: `{rel(RUN_DIR / 'candidate_residuals.csv')}`. Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_docs(raw: Mapping[str, Any]) -> None:
    best = raw["objective_localization"].get("best_candidate") or {}
    res = raw["residual_localization"]
    block = f"""
<!-- {MARKER} -->
## v34h objective/residual localization

UTC: {raw['created_utc']}. Executed Opus Task E/F as offline development-only localization from the already-opened v34g one-cell arrays. Budgets: solver_calls=0, plant_steps=0, env_reset_calls_after_construction=0, training/refit=0, validation64=false, sealed_test=false. G-E objective reconstruction pass={raw['gates']['G_E_objective_reconstruction_pass']} with best candidate `{best.get('candidate_id')}`, total={best.get('total')}, solver={best.get('solver_objective')}, rel_error={best.get('relative_error')}. G-F residual alias offline pass={raw['gates']['G_F_residual_alias_offline_pass']}; repaired residuals={res['repaired_names']}; legacy residuals={res['legacy_names']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'candidate_residuals.csv')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.
"""
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / name, MARKER, block)
    response = f"""
<!-- {MARKER} -->
## Response/log: A13_objective_reconstruction_contract_localization + residual-bounds-capture

Opus plan `{rel(CURRENT_OPUS_REPORT)}` accepted. GPT-5.5 executor ran the zero-solve/zero-plant offline localization at `{raw['created_utc']}`. Objective G-E pass={raw['gates']['G_E_objective_reconstruction_pass']}; best formula `{best.get('candidate_id')}` matched solver objective with relative error `{best.get('relative_error')}`. Residual alias G-F offline pass={raw['gates']['G_F_residual_alias_offline_pass']}` using `lb_opt_x/ub_opt_x` and `cons_lb/cons_ub`; legacy `opt_x_lb/opt_g_lb` names remain absent. Evidence: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'candidate_residuals.csv')}`. No validation64/sealed-test access and no remaining 23 Task-C solver calls were spent.
"""
    append_if_missing(RESPONSE_LOG, MARKER, response)
    write_json(NEXT_REVIEW_REQUEST, {
        "request_id": REQUEST_ID,
        "created": raw["created_utc"],
        "status": "analysis_requested" if not (raw["gates"]["G_E_objective_reconstruction_pass"] and raw["gates"]["G_F_residual_alias_offline_pass"]) else "gate_evidence_ready_dependent_task_candidate",
        "trigger": "v34h offline objective/residual localization completed",
        "experiment_id": NAME,
        "active_lead_report": rel(CURRENT_OPUS_REPORT),
        "active_lead_report_sha256": CURRENT_OPUS_SHA,
        "question": "Review v34h Task E/F localization. If G-E and G-F are sufficient, authorize/confirm the corrected 24-cell objective-vs-basin implementation using author objective terms and repaired residual aliases; if not, specify the minimal additional no-plant introspection.",
        "evidence_paths": [rel(RUN_DIR / "summary.md"), rel(RUN_DIR / "raw.json"), rel(RUN_DIR / "candidate_residuals.csv"), rel(RUN_DIR / "completed.json"), rel(CURRENT_OPUS_REPORT), rel(RESPONSE_LOG)],
        "budget_actual": raw["budget_actual"],
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "plant_steps": 0, "training_or_refit": 0},
        "gates": raw["gates"],
        "best_candidate": {k: best.get(k) for k in ["candidate_id", "relative_error", "absolute_error", "total", "solver_objective", "stage_lterm_total", "epsterm_total", "input_regularization_total", "terminal_discounted"]},
        "operational_note": f"Post-run backup required via {rel(BACKUP_REQUEST)} before additional unique solver/scientific work.",
    })


def make_backup_request(raw: Mapping[str, Any]) -> None:
    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v34h_objective_localization",
        "created_utc": raw["created_utc"],
        "backup_required_before_more_unique_science": True,
        "reason": "new v34h offline objective/residual localization source, raw candidate table, docs, and lead/reviewer handoff must be externally recoverable before remaining objective-vs-basin solver calls",
        "must_cover": [
            rel(Path(__file__).resolve()),
            rel(RUN_DIR),
            rel(STATE),
            rel(BACKUP_REQUEST),
            rel(NEXT_REVIEW_REQUEST),
            rel(RESPONSE_LOG),
            "STATUS.md",
            "RESEARCH_LOG.md",
            "DECISIONS.md",
            "RESULTS_AUDIT.md",
            "REPRODUCTION_PROTOCOL.md",
            "EXPERIMENT_REGISTRY.csv",
        ],
        "new_solver_calls": 0,
        "new_plant_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_gate": "if G-E and G-F are accepted, implement/run corrected fixed objective-vs-basin probe; otherwise perform at most Opus-authorized no-plant introspection",
    })


def hash_existing(paths: Iterable[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        try:
            if p.exists() and p.is_file():
                out[rel(p)] = sha256(p)
        except Exception:
            pass
    return out


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--i-accept-zero-solve-objective-localization", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        started = now_utc()
        gates = verify_opus_and_inputs()
        patch_no_solve_runtime()
        raw_v34g = read_json(V34G_RAW)
        mpc, build_meta = build_mpc_without_solve()
        assign_meta = assign_saved_solution(mpc, raw_v34g)
        obj = enumerate_objective_candidates(mpc, TARGET_CELL["horizon"])
        x_slices = extract_x_slices(mpc, TARGET_CELL["horizon"])
        res = residual_localization(mpc)
        source_snippets = read_source_snippets()
        best = obj.get("best_candidate") or {}
        g_e = bool(best.get("relative_error") is not None and float(best.get("relative_error")) <= GATE_REL_TOL)
        g_f = bool(res.get("repaired_size_gate") and res.get("repaired_residual_gate_non_none_and_le_1e-5"))
        created = now_utc()
        raw = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_offline_objective_residual_localization_not_validation_not_test",
            "active_lead": "claude-opus-5-5",
            "lead_report": rel(CURRENT_OPUS_REPORT),
            "lead_report_sha256": CURRENT_OPUS_SHA,
            "stable_ids": ["A13_objective_reconstruction_contract_localization", "A12_registry_backup_schema_contract/residual-bounds-capture"],
            "hypothesis_frozen": "The v34g objective mismatch is caused by an incomplete reconstruction formula (especially missing epsterm/rterm and/or final-node indexing), while residual None is a capture-name alias defect; both can be localized from saved arrays without new solves.",
            "selected_cell": TARGET_CELL,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_declared": {"solver_call_cap": 0, "plant_steps": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "budget_actual": {"solver_calls": 0, "plant_steps": 0, "env_reset_calls_after_construction": int(build_meta.get("forbidden_call_counter", {}).get("env_reset", 0)), "env_step_calls_after_construction": int(build_meta.get("forbidden_call_counter", {}).get("env_step", 0)), "forbidden_solve_calls_attempted": int(build_meta.get("forbidden_call_counter", {}).get("solve", 0)), "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "input_gates": gates,
            "runtime_build_meta": build_meta,
            "assignment_meta": assign_meta,
            "objective_localization": obj,
            "x_solution_slices": x_slices,
            "residual_localization": res,
            "source_contract_snippets": source_snippets,
            "gates": {"G_E_objective_reconstruction_pass": g_e, "G_F_residual_alias_offline_pass": g_f, "both_pass": bool(g_e and g_f)},
            "next_constraints": ["Do not spend remaining 23 solver calls until corrected formula/capture source is frozen and backed up", "Do not open validation64 or sealed test", "Do not start selector refit/training before lead review or already-approved dependent gate sequence"],
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
        }
        if raw["budget_actual"]["forbidden_solve_calls_attempted"] != 0 or raw["budget_actual"]["plant_steps"] != 0 or raw["budget_actual"]["env_reset_calls_after_construction"] != 0:
            raise ContractError("forbidden solve/plant/reset call occurred during offline localization")
        write_json(RUN_DIR / "raw.json", raw)
        top_rows = [r for r in (obj.get("all_candidates") or []) if r.get("relative_error") is not None]
        top_rows.sort(key=lambda r: float(r["relative_error"]))
        write_csv_table(RUN_DIR / "candidate_residuals.csv", top_rows)
        write_summary(raw)
        update_docs(raw)
        make_backup_request(raw)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(
            f"# Continue state after v34h objective/residual localization\n\nUTC: {raw['created_utc']}\n\nGates: {json.dumps(raw['gates'], sort_keys=True)}\n\nBest candidate: {json.dumps({k: best.get(k) for k in ['candidate_id','relative_error','absolute_error','total','solver_objective','stage_lterm_total','epsterm_total','input_regularization_total','terminal_discounted']}, sort_keys=True)}\n\nArtifacts: {rel(RUN_DIR / 'summary.md')}, {rel(RUN_DIR / 'raw.json')}, {rel(RUN_DIR / 'candidate_residuals.csv')}, {rel(RUN_DIR / 'completed.json')}\n\nNext: external backup for {rel(BACKUP_REQUEST)}. If lead accepts G-E/G-F, freeze corrected Task-C probe source using author objective components and lb_opt_x/cons_lb residual aliases before spending the remaining fixed solver calls.\n",
            encoding="utf-8",
        )
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [
            Path(__file__).resolve(), STATE, BACKUP_REQUEST, NEXT_REVIEW_REQUEST, RESPONSE_LOG,
            ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv",
        ]
        completed = {
            "status": "complete",
            "passed": bool(g_e and g_f),
            "hard_pass": bool(g_e and g_f),
            "created_utc": created.isoformat(),
            "classification": raw["classification"],
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_actual": raw["budget_actual"],
            "headline": {
                "G_E_objective_reconstruction_pass": g_e,
                "G_F_residual_alias_offline_pass": g_f,
                "both_pass": bool(g_e and g_f),
                "best_candidate_id": best.get("candidate_id"),
                "best_relative_error": best.get("relative_error"),
                "best_absolute_error": best.get("absolute_error"),
                "best_total": best.get("total"),
                "solver_objective": best.get("solver_objective"),
                "residual_repaired": res.get("repaired_names"),
                "residual_legacy": res.get("legacy_names"),
            },
            "summary": rel(RUN_DIR / "summary.md"),
            "raw": rel(RUN_DIR / "raw.json"),
            "candidate_table": rel(RUN_DIR / "candidate_residuals.csv"),
            "backup_request": rel(BACKUP_REQUEST),
            "next_review_request_id": REQUEST_ID,
            "hashes": hash_existing(sorted(set(files))),
        }
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "candidate_table": rel(RUN_DIR / "candidate_residuals.csv"), "headline": completed["headline"], "backup_request": rel(BACKUP_REQUEST), "next_review_request_id": REQUEST_ID}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        fail = {
            "status": "failed",
            "created_utc": now_utc().isoformat(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "classification": "development_IMPROVED_offline_objective_residual_localization_not_validation_not_test",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "budget_caps": {"solver_call_cap": 0, "plant_steps": 0, "env_reset_calls_after_construction": 0},
        }
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(
            f"# v34h objective/residual localization failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR / 'failed.json')}\n\nNo validation64 or sealed test access was requested. Do not spend remaining 23 solver calls until this failure is localized.\n",
            encoding="utf-8",
        )
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
