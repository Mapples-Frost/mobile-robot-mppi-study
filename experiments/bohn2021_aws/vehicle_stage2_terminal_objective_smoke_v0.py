#!/usr/bin/env python3
"""Vehicle Stage2 terminal/objective instrumentation smoke v0.

Development-only bounded diagnostic. It replays deterministic H15 prefixes for a
small subset of already-developed stress Stage2 states, then runs selected
branch horizons while recording terminal-value/objective observables that were
not stored in the original Stage2 traces.

No historical validation64 bank access, no sealed final-test access, no training
or refit. The input backup proof is required to cover the already-created Stage2
and source-audit evidence; this new diagnostic writes a post-run backup request.
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
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
REPRO = ROOT / "experiments/bohn2021_reproduction"
for p in (AWS_DIR, REPRO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import vehicle_stress_scenario_opportunity_probe_v0_runner as stage1_runner  # noqa:E402
import vehicle_v1_fresh_continuation_label_probe_v0_runner as fresh  # noqa:E402

TASK = "vehicle"
MAX_STEPS = 150
PREFIX_H = 15
MATERIAL_GAIN = 3.0
OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_stage2_terminal_objective_smoke_v0_20260928T1900Z"
STATE = ROOT / "research_artifacts/aws_state/vehicle_stage2_terminal_objective_smoke_v0_20260928T1900Z.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
STAGE1_BANK = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_20260928/bank/vehicle_stress_scenario_opportunity_probe_v0_bank.json"
STAGE1_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.json"
STAGE2_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/raw.json"
STAGE2_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/completed.json"
PREFIX_DIAG_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stage2_prefix_dynamics_hash_diagnostic_v0_20260928T1845Z/completed.json"
SOURCE_AUDIT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_terminal_objective_source_audit_v0_20260928T1830Z/completed.json"
REWARD_AUDIT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_reward_timing_terminal_audit_v0_20260928T1820Z/completed.json"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stage2_terminal_objective_smoke_v0_frozen_20260928T1900Z.json"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_stage2_terminal_objective_smoke_v0_frozen_20260928T1900Z.md"
MARKER = "vehicle-stage2-terminal-objective-smoke-v0-20260928T1900Z"

# Pre-frozen small design: one corrected-positive state plus a harm and a neutral
# control. Terminal-mode ablations are intentionally concentrated on the positive
# state where terminal mismatch is most discriminating.
STATE_SPECS = [
    {"state_id": "positive_case5_step18", "case": 5, "branch_step": 18, "role": "corrected_positive_local_state"},
    {"state_id": "harm_case0_step33", "case": 0, "branch_step": 33, "role": "large_harm_control_state"},
    {"state_id": "neutral_case8_step56", "case": 8, "branch_step": 56, "role": "neutral_low_stress_control_state"},
]
RUN_PLAN = []
for h in [10, 15, 25, 30, 35, 45]:
    RUN_PLAN.append({"state_id": "positive_case5_step18", "horizon": h, "terminal_mode": "per_h"})
for h in [10, 30, 35]:
    for mode in ["h15_terminal", "h25_terminal", "zero_terminal"]:
        RUN_PLAN.append({"state_id": "positive_case5_step18", "horizon": h, "terminal_mode": mode})
for h in [10, 15, 25, 30]:
    RUN_PLAN.append({"state_id": "harm_case0_step33", "horizon": h, "terminal_mode": "per_h"})
for h in [10, 25]:
    RUN_PLAN.append({"state_id": "harm_case0_step33", "horizon": h, "terminal_mode": "zero_terminal"})
for h in [10, 15, 30]:
    RUN_PLAN.append({"state_id": "neutral_case8_step56", "horizon": h, "terminal_mode": "per_h"})
RUN_PLAN.append({"state_id": "neutral_case8_step56", "horizon": 10, "terminal_mode": "zero_terminal"})
assert len(RUN_PLAN) == 25


class ContractError(RuntimeError):
    pass


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def serial(x: Any) -> Any:
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "item"):
        return x.item()
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    return str(x)


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=serial) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=serial).encode("utf-8")).hexdigest()


def parse_time(v: Any) -> Optional[dt.datetime]:
    if not isinstance(v, str) or not v:
        return None
    try:
        out = dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def verify_done(path: Path, check_hashes: bool = False) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing required completed marker: %s" % rel(path))
    done = read_json(path)
    if done.get("passed") is not True:
        raise ContractError("required completed marker did not pass: %s" % rel(path))
    if done.get("sealed_test_accessed") is not False or done.get("historical_validation64_bank_opened") is not False:
        raise ContractError("input marker access flags invalid: %s" % rel(path))
    if check_hashes:
        for name, expected in (done.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists() or sha256(p) != expected:
                raise ContractError("input hash mismatch from completed marker: %s" % name)
    return done


def verify_backup(path: Path, min_time: dt.datetime) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("backup proof missing: %s" % rel(path))
    proof = read_json(path)
    if not (proof.get("backup_verified") is True or proof.get("status") == "verified"):
        raise ContractError("backup proof not verified")
    if int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("backup proof remaining_changed_files != 0")
    if not proof.get("commit"):
        raise ContractError("backup proof lacks commit")
    if not (proof.get("packages_this_run") or proof.get("asset_sha256") or proof.get("package_sha256") or proof.get("release_asset_sha256")):
        raise ContractError("backup proof lacks release/package SHA")
    t = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        t = parse_time(proof.get(key))
        if t is not None:
            break
    if t is None or t < min_time:
        raise ContractError("backup proof does not postdate required inputs")
    return {"path": rel(path), "sha256": sha256(path), "time": t.isoformat(), "commit": proof.get("commit"), "remaining_changed_files": proof.get("remaining_changed_files"), "packages_this_run": proof.get("packages_this_run")}


def arr(x: Any) -> np.ndarray:
    try:
        if hasattr(x, "full"):
            return np.asarray(x.full(), dtype=float).reshape(-1)
        if hasattr(x, "cat"):
            return np.asarray(x.cat, dtype=float).reshape(-1)
        return np.asarray(x, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def arr_hash(a: np.ndarray) -> Optional[str]:
    if a.size == 0 or not np.isfinite(a).all():
        return None
    return hashlib.sha256(np.ascontiguousarray(a.astype(np.float64)).tobytes()).hexdigest()


def finite_summary(a: np.ndarray) -> Dict[str, Any]:
    if a.size == 0:
        return {"size": 0, "finite": None, "sha256": None, "mean": None, "max_abs": None, "values": []}
    finite = bool(np.isfinite(a).all())
    return {"size": int(a.size), "finite": finite, "sha256": arr_hash(a), "mean": float(np.mean(a)) if finite else None, "max_abs": float(np.max(np.abs(a))) if finite else None, "values": a[:20].tolist() if finite else []}


def install_instrumented_recovery(mpc: Any, folder: Path) -> Dict[str, Any]:
    from casadi import DM
    original = mpc.solve
    state: Dict[str, Any] = {"enabled": False, "events": [], "counts": {"solve_attempts": 0, "solve_completed": 0, "warmup_attempts": 0, "retry_attempts": 0}, "step": None, "case": None, "horizon": None}
    attempts = folder / "solver_attempts.json"
    if attempts.exists():
        raise ContractError("solver_attempts already exists")
    write_json(attempts, state["counts"])

    def residual(v: np.ndarray, lo: Any, hi: Any) -> Optional[float]:
        if v.size == 0 or not np.isfinite(v).all():
            return None
        lo_a, hi_a = arr(lo), arr(hi)
        if lo_a.size != v.size or hi_a.size != v.size:
            return None
        return float(np.maximum.reduce([np.zeros_like(v), lo_a - v, v - hi_a]).max())

    def scalar_attr(name: str) -> Optional[float]:
        a = arr(getattr(mpc, name, []))
        return float(a[0]) if a.size and np.isfinite(a).all() else None

    def call(kind: str) -> Dict[str, Any]:
        c = state["counts"]
        c["solve_attempts"] += 1
        if kind == "warmup":
            c["warmup_attempts"] += 1
        if kind.startswith("retry"):
            c["retry_attempts"] += 1
        write_json(attempts, c)
        pre_x = arr(getattr(mpc, "opt_x_num", []))
        start = time.perf_counter()
        original()
        elapsed = time.perf_counter() - start
        c["solve_completed"] += 1
        write_json(attempts, c)
        post_x = arr(getattr(mpc, "opt_x_num", []))
        g = arr(getattr(mpc, "opt_g_num", []))
        gr = residual(g, mpc.cons_lb, mpc.cons_ub) if g.size else None
        xr = residual(post_x, mpc.lb_opt_x, mpc.ub_opt_x) if post_x.size else None
        stats = dict(getattr(mpc, "solver_stats", {}) or {})
        accepted = bool((post_x.size == 0 or np.isfinite(post_x).all()) and (g.size == 0 or np.isfinite(g).all()) and bool(stats.get("success", False)) and (gr is None or gr <= 1e-5) and (xr is None or xr <= 1e-5))
        row = {
            "case": state["case"], "step": state["step"], "horizon": state["horizon"], "kind": kind,
            "success": bool(stats.get("success", False)), "return_status": stats.get("return_status"),
            "iterations": int(stats.get("iter_count", -1)) if stats.get("iter_count") is not None else None,
            "finite": bool((post_x.size == 0 or np.isfinite(post_x).all()) and (g.size == 0 or np.isfinite(g).all())),
            "constraint_residual": gr, "bound_residual": xr, "solver_s": float(elapsed), "accepted": accepted,
            "objective_opt_f_num": scalar_attr("opt_f_num"),
            "warm_start_pre_opt_x": finite_summary(pre_x),
            "warm_start_post_opt_x": finite_summary(post_x),
        }
        with (folder / "solver_calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=serial, allow_nan=False) + "\n")
        return row

    def solve() -> None:
        if not state["enabled"]:
            call("warmup")
            return
        records = [call("initial")]
        if not records[0]["accepted"]:
            saved = {name: copy.deepcopy(getattr(mpc, name)) for name in ("opt_g_num", "opt_f_num", "lam_g_num", "lam_x_num", "solver_stats")}
            initial_solution = arr(mpc.opt_x_num).copy()
            for name, previous in [("retry_zero", False), ("retry_previous", True)]:
                guess = mpc.opt_x(0)
                guess["_x"] = mpc.opt_p_num["_x0"] / mpc._x_scaling
                guess["_u"] = mpc.opt_p_num["_u_prev"] / mpc._u_scaling if previous else 0
                mpc.opt_x_num.master = guess.cat
                records.append(call(name))
                if records[-1]["accepted"]:
                    break
            if not records[-1]["accepted"]:
                mpc.opt_x_num.master = DM(initial_solution)
                try:
                    mpc.opt_x_num_unscaled.master = DM(initial_solution * arr(mpc.opt_x_scaling))
                except Exception:
                    pass
                for name, value in saved.items():
                    setattr(mpc, name, value)
                mpc.calculate_aux_num()
        state["events"].append({"attempts": records, "recovered": (not records[0]["accepted"] and records[-1]["accepted"]), "final_success": bool(mpc.solver_stats.get("success", False)), "restored_original": not records[-1]["accepted"]})

    mpc.solve = solve
    return state


def step_physical(row: Mapping[str, Any]) -> float:
    return float(row.get("performance", 0.0)) + float(row.get("constraint", 0.0))


def step_total(row: Mapping[str, Any]) -> float:
    return step_physical(row) + float(row.get("compute", 0.0))


def tail_sum(trace: Sequence[Mapping[str, Any]], start: int, which: str) -> float:
    if start >= len(trace):
        return 0.0
    if which == "physical":
        return float(math.fsum(step_physical(r) for r in trace[start:]))
    if which == "total":
        return float(math.fsum(step_total(r) for r in trace[start:]))
    raise ValueError(which)


def values_summary(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not xs:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = (len(xs) - 1) * p / 100.0
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        return xs[lo] if lo == hi else xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"count": len(xs), "sum": float(math.fsum(xs)), "mean": float(math.fsum(xs) / len(xs)), "median": float(pct(50)), "p95": float(pct(95)), "min": xs[0], "max": xs[-1]}


def state_tuple(row: Mapping[str, Any], key: str = "previous_state") -> Tuple[float, float, float]:
    s = row.get(key) or {}
    return (float(s.get("x", float("nan"))), float(s.get("y", float("nan"))), float(s.get("theta", float("nan"))))


def clean_prefix(trace: Sequence[Mapping[str, Any]], branch_step: int) -> List[Dict[str, Any]]:
    cleaned = copy.deepcopy(list(trace[:min(branch_step, len(trace))]))
    for row in cleaned:
        for k in ("timing", "decision_plus_terminal_switch_s", "mpc_diag"):
            row.pop(k, None)
        for attempt in (row.get("recovery") or {}).get("attempts", []):
            attempt.pop("solver_s", None)
            attempt.pop("objective_opt_f_num", None)
            attempt.pop("warm_start_pre_opt_x", None)
            attempt.pop("warm_start_post_opt_x", None)
    return cleaned


def terminal_weights_for(mode: str, h: int, terminals: Mapping[int, Tuple[Any, Any]]) -> Tuple[Any, Any, str]:
    if mode == "per_h":
        return terminals[h][0], terminals[h][1], "H%d" % h
    if mode == "h15_terminal":
        return terminals[15][0], terminals[15][1], "H15"
    if mode == "h25_terminal":
        return terminals[25][0], terminals[25][1], "H25"
    if mode == "zero_terminal":
        w, b = terminals[h]
        return [np.zeros_like(np.asarray(x)) for x in w], [np.zeros_like(np.asarray(x)) for x in b], "zero_like_H%d" % h
    raise ContractError("unknown terminal_mode %s" % mode)


def compact_info_data(info: Mapping[str, Any]) -> Dict[str, Any]:
    data = info.get("data") or {}
    out = {}
    for k in ("mpc_state", "mpc_next_state", "mpc_n_horizon", "mpc_rewards"):
        if k in data:
            a = arr(data[k])
            out[k] = a.tolist() if a.size else data[k]
    return out


def capture_mpc_diag(env: Any, info: Mapping[str, Any]) -> Dict[str, Any]:
    mpc = env.control_system.controller.mpc
    diag: Dict[str, Any] = {
        "info_mpc_value_fn": None if info.get("mpc_value_fn") is None else float(info.get("mpc_value_fn")),
        "info_mpc_avg_stage_cost": None if info.get("mpc_avg_stage_cost") is None else float(info.get("mpc_avg_stage_cost")),
        "info_mpc_computation_time": None if info.get("mpc_computation_time") is None else float(info.get("mpc_computation_time")),
        "info_data_compact": compact_info_data(info),
        "use_nn_vf": bool(getattr(mpc, "use_nn_vf", False)),
        "objective_opt_f_num": float(arr(getattr(mpc, "opt_f_num", []))[0]) if arr(getattr(mpc, "opt_f_num", [])).size else None,
    }
    for key, expr in [
        ("pred_terminal_state", lambda: mpc.opt_x_num_unscaled["_x", -1, 0, -1]),
        ("current_x0_param", lambda: mpc.opt_p_num["_x0"]),
        ("first_u_unscaled", lambda: mpc.opt_x_num_unscaled["_u", 0, 0]),
        ("terminal_p", lambda: mpc.opt_p_num["_p", 0]),
    ]:
        try:
            diag[key] = arr(expr()).tolist()
        except Exception as exc:
            diag[key + "_error"] = repr(exc)
    return diag


def run_one(item: Mapping[str, Any], case: Mapping[str, Any], terminals: Mapping[int, Tuple[Any, Any]], terminal_receipts: Mapping[str, Any]) -> Dict[str, Any]:
    sid = "%02d_%s_H%02d_%s" % (int(item["execution_index"]), item["state_id"], int(item["horizon"]), item["terminal_mode"])
    ep_dir = OUT / "episodes" / sid
    ep_dir.mkdir(parents=True, exist_ok=False)
    branch_step = int(item["branch_step"])
    branch_h = int(item["horizon"])
    mode = str(item["terminal_mode"])
    start_wall = time.perf_counter()
    env = fresh.v1.make_env(TASK, 0, aligned=True, scaled_obs=True)
    counts = fresh.v1.meter(env, ep_dir)
    # Prefix always uses H15 terminal to preserve the Stage2 H15-prefix construction.
    env.set_value_function_weights_and_biases(*terminals[PREFIX_H])
    active_terminal_label = "H15"
    controller = env.control_system.controller
    original_get_action = controller.get_action
    measured: List[Dict[str, float]] = []

    def timed(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        value = original_get_action(*args, **kwargs)
        gross = time.perf_counter() - t0
        measured.append({"controller_gross_s": float(gross), "controller_s": float(gross), "logging_s": 0.0})
        return value

    controller.get_action = timed
    recovery = install_instrumented_recovery(controller.mpc, ep_dir)
    recovery.update(enabled=False, events=[], case=int(item["case"]), step=-1, horizon=PREFIX_H)
    reset_start = time.perf_counter()
    obs = env.reset(**copy.deepcopy(dict(case)))
    reset_gross = time.perf_counter() - reset_start
    if obs is None or len(measured) != 1:
        raise ContractError("unexpected reset measurement count")
    reset = {"reset_gross_s": float(reset_gross), **measured.pop()}
    write_json(ep_dir / "reset.json", reset)
    recovery["enabled"] = True
    trace: List[Dict[str, Any]] = []
    terminal_switches: List[Dict[str, Any]] = []
    with (ep_dir / "trace.jsonl").open("x", encoding="utf-8") as stream:
        for t in range(MAX_STEPS):
            h = PREFIX_H if t < branch_step else branch_h
            desired_terminal_label = active_terminal_label
            switch_s = 0.0
            if t >= branch_step:
                w, b, desired_terminal_label = terminal_weights_for(mode, h, terminals)
                if desired_terminal_label != active_terminal_label:
                    st = time.perf_counter()
                    env.set_value_function_weights_and_biases(w, b)
                    switch_s = float(time.perf_counter() - st)
                    terminal_switches.append({"step": t, "from": active_terminal_label, "to": desired_terminal_label, "switch_s": switch_s})
                    active_terminal_label = desired_terminal_label
            recovery["step"] = int(t)
            recovery["horizon"] = int(h)
            prev_obs = env.get_observation().tolist()
            prev_state = copy.deepcopy(env.control_system.current_state)
            obs, reward, done, info = env.step(np.array([float(h)]))
            if len(measured) != 1:
                raise ContractError("expected exactly one measured controller call")
            timing = measured.pop()
            timing.update({"selection_s": 0.0, "decision_s": float(timing["controller_s"]), "decision_gross_s": float(timing["controller_gross_s"]), "terminal_switch_s_before_controller": switch_s})
            row = {
                "step": int(t),
                "horizon": int(info.get("executed_horizon", h)),
                "terminal_mode": mode,
                "terminal_source_label": active_terminal_label,
                "previous_state": prev_state,
                "state": copy.deepcopy(env.control_system.current_state),
                "input": copy.deepcopy(env.control_system.controller.current_input),
                "observation": prev_obs,
                "next_observation": obs.tolist(),
                "reward": float(reward),
                "performance": float(info.get("reward/performance", float("nan"))),
                "compute": float(info.get("reward/computation", float("nan"))),
                "constraint": float(info.get("reward/constraint", 0.0)),
                "solver_success": bool(info.get("solver_success")),
                "termination": info.get("termination"),
                "decision": {"kind": "stage2_terminal_objective_smoke", "phase": "prefix" if t < branch_step else "branch", "prefix_horizon": PREFIX_H, "branch_step": branch_step, "branch_horizon": branch_h, "terminal_mode": mode, "terminal_source_label": active_terminal_label},
                "timing": timing,
                "decision_plus_terminal_switch_s": float(timing["decision_s"] + switch_s),
                "recovery": recovery["events"][-1],
                "mpc_diag": capture_mpc_diag(env, info),
            }
            assert row["horizon"] == h
            stream.write(json.dumps(row, default=serial, allow_nan=False) + "\n")
            stream.flush()
            trace.append(row)
            if done:
                break
    if not trace or not trace[-1].get("termination"):
        raise ContractError("episode did not terminate within MAX_STEPS")
    fresh.v1.audit_trace(TASK, dict(case), trace)
    metric = fresh.v1.case_metrics(TASK, trace)
    branch_reached = len(trace) > branch_step
    branch_row = trace[branch_step] if branch_reached else None
    decision_times = [float(r["timing"]["decision_s"]) for r in trace]
    solver_times = []
    for r in trace:
        for a in (r.get("recovery") or {}).get("attempts", []):
            if a.get("solver_s") is not None:
                solver_times.append(float(a["solver_s"]))
    horizon_counts: Dict[str, int] = {}
    for r in trace:
        horizon_counts[str(r["horizon"])] = horizon_counts.get(str(r["horizon"]), 0) + 1
    summary = dict(metric)
    summary.update({
        "execution_index": int(item["execution_index"]),
        "state_id": item["state_id"],
        "state_role": item["role"],
        "case": int(item["case"]),
        "branch_step": branch_step,
        "branch_horizon": branch_h,
        "terminal_mode": mode,
        "effective_branch_terminal_label": active_terminal_label,
        "branch_reached": bool(branch_reached),
        "branch_previous_state": branch_row.get("previous_state") if branch_row else None,
        "prefix_clean_dynamics_sha256": canonical_sha(clean_prefix(trace, branch_step)),
        "continuation_physical_constraint_cost_from_branch": tail_sum(trace, branch_step, "physical"),
        "continuation_total_cost_from_branch": tail_sum(trace, branch_step, "total"),
        "branch_step_mpc_diag": branch_row.get("mpc_diag") if branch_row else None,
        "branch_step_recovery_final_attempt": ((branch_row.get("recovery") or {}).get("attempts") or [{}])[-1] if branch_row else None,
        "decision_timing_s": values_summary(decision_times),
        "solver_attempt_timing_s": values_summary(solver_times),
        "terminal_switches": terminal_switches,
        "terminal_switch_timing_s": values_summary(s.get("switch_s", 0.0) for s in terminal_switches),
        "horizon_counts": horizon_counts,
        "reset": reset,
        "steps_metered": int(counts.get("step_calls", 0)),
        "resets_metered": int(counts.get("reset_calls", 0)),
        "episode_wall_s": float(time.perf_counter() - start_wall),
        "terminal_receipt_effective": terminal_receipts.get(str(branch_h)) if mode == "per_h" else terminal_receipts.get("15") if mode == "h15_terminal" else terminal_receipts.get("25") if mode == "h25_terminal" else {"zero_like_horizon": branch_h},
        "path": rel(ep_dir),
    })
    write_json(ep_dir / "trace.json", trace)
    write_json(ep_dir / "summary.json", summary)
    files = [p for p in ep_dir.iterdir() if p.is_file() and p.name != "completed.json"]
    write_json(ep_dir / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})
    return summary


def no_regression(c: Mapping[str, Any], ref: Mapping[str, Any]) -> bool:
    if bool(ref.get("success")) and not bool(c.get("success")):
        return False
    if bool(c.get("constraint")) and not bool(ref.get("constraint")):
        return False
    if int(c.get("initial_failed_steps", 0)) > int(ref.get("initial_failed_steps", 0)):
        return False
    if int(c.get("solver_failure_steps", 0)) > int(ref.get("solver_failure_steps", 0)):
        return False
    return True


def analyze(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_state_mode: Dict[Tuple[str, str], Dict[int, Mapping[str, Any]]] = {}
    for e in episodes:
        by_state_mode.setdefault((str(e["state_id"]), str(e["terminal_mode"])), {})[int(e["branch_horizon"])] = e
    rows = []
    positive_counts: Dict[str, int] = {}
    terminal_sensitivity = []
    for (state_id, mode), by_h in sorted(by_state_mode.items()):
        ref = by_state_mode.get((state_id, "per_h"), {}).get(PREFIX_H)
        if not ref:
            continue
        for h, e in sorted(by_h.items()):
            gain_phys = float(ref["continuation_physical_constraint_cost_from_branch"] - e["continuation_physical_constraint_cost_from_branch"])
            gain_total = float(ref["continuation_total_cost_from_branch"] - e["continuation_total_cost_from_branch"])
            material = bool(h != PREFIX_H and e.get("branch_reached") and no_regression(e, ref) and (gain_phys >= MATERIAL_GAIN or gain_total >= MATERIAL_GAIN))
            if material:
                positive_counts[mode] = positive_counts.get(mode, 0) + 1
            bdiag = e.get("branch_step_mpc_diag") or {}
            rows.append({
                "state_id": state_id,
                "case": int(e["case"]),
                "branch_step": int(e["branch_step"]),
                "horizon": h,
                "terminal_mode": mode,
                "effective_terminal": e.get("effective_branch_terminal_label"),
                "success": bool(e.get("success")),
                "constraint": bool(e.get("constraint")),
                "steps": int(e.get("steps", 0)),
                "solver_failure_steps": int(e.get("solver_failure_steps", 0)),
                "continuation_physical": float(e["continuation_physical_constraint_cost_from_branch"]),
                "continuation_total": float(e["continuation_total_cost_from_branch"]),
                "gain_vs_per_h_H15_physical": gain_phys,
                "gain_vs_per_h_H15_total": gain_total,
                "material_positive_vs_per_h_H15": material,
                "branch_mpc_value_fn": bdiag.get("info_mpc_value_fn"),
                "branch_mpc_avg_stage_cost": bdiag.get("info_mpc_avg_stage_cost"),
                "branch_objective_opt_f_num": bdiag.get("objective_opt_f_num"),
                "branch_pred_terminal_state": bdiag.get("pred_terminal_state"),
                "decision_sum_s": e["decision_timing_s"]["sum"],
                "terminal_switch_sum_s": e["terminal_switch_timing_s"]["sum"],
                "path": e["path"],
            })
    # Compare terminal modes for identical state+horizon when per_h is present.
    by_state_h: Dict[Tuple[str, int], Dict[str, Mapping[str, Any]]] = {}
    for r in rows:
        by_state_h.setdefault((r["state_id"], int(r["horizon"])), {})[str(r["terminal_mode"])] = r
    for key, modes in sorted(by_state_h.items()):
        base = modes.get("per_h")
        if not base:
            continue
        for mode, r in sorted(modes.items()):
            if mode == "per_h":
                continue
            terminal_sensitivity.append({
                "state_id": key[0],
                "horizon": key[1],
                "mode": mode,
                "physical_delta_vs_per_h": float(r["continuation_physical"] - base["continuation_physical"]),
                "total_delta_vs_per_h": float(r["continuation_total"] - base["continuation_total"]),
                "value_fn_delta_vs_per_h_at_branch": None if r["branch_mpc_value_fn"] is None or base["branch_mpc_value_fn"] is None else float(r["branch_mpc_value_fn"] - base["branch_mpc_value_fn"]),
                "objective_delta_vs_per_h_at_branch": None if r["branch_objective_opt_f_num"] is None or base["branch_objective_opt_f_num"] is None else float(r["branch_objective_opt_f_num"] - base["branch_objective_opt_f_num"]),
                "per_h_material": bool(base["material_positive_vs_per_h_H15"]),
                "mode_material": bool(r["material_positive_vs_per_h_H15"]),
            })
    case5_h30 = next((r for r in rows if r["state_id"] == "positive_case5_step18" and r["horizon"] == 30 and r["terminal_mode"] == "per_h"), None)
    case5_h30_sensitivity = [x for x in terminal_sensitivity if x["state_id"] == "positive_case5_step18" and x["horizon"] == 30]
    flips = [x for x in terminal_sensitivity if bool(x["per_h_material"]) != bool(x["mode_material"])]
    return {
        "comparison_rows": rows,
        "terminal_sensitivity_rows": terminal_sensitivity,
        "positive_counts_by_terminal_mode": positive_counts,
        "material_label_flips_under_terminal_ablation": flips,
        "case5_h30_per_h": case5_h30,
        "case5_h30_terminal_sensitivity": case5_h30_sensitivity,
        "terminal_mode_ablation_count": len(terminal_sensitivity),
        "decision": {
            "retrain_or_refit_now": False,
            "reason": "This is a tiny instrumentation smoke over three development states. It can diagnose terminal/objective artifacts but cannot provide label density for training/refit or validation claims.",
            "next_if_terminal_artifact_strong": "Freeze a per-H/matched-terminal continuation-label protocol or terminal-value refit ablation before selector training.",
            "next_if_terminal_artifact_weak": "Freeze a separately versioned source-supported stress-v1 scenario opportunity protocol to seek denser labels before any multi-H selector refit.",
        },
    }


def freeze_protocol(created: str, backup: Mapping[str, Any]) -> None:
    protocol = {
        "protocol_id": "vehicle_stage2_terminal_objective_smoke_v0_frozen_20260928T1900Z",
        "created_utc": created,
        "classification": "development_IMPROVED_terminal_objective_instrumentation_smoke_not_validation_not_final_test",
        "hypothesis": "Sparse corrected Stage2 labels may be partly caused by learned-terminal mismatch/objective artifacts; measure mpc_value_fn/objective and per-H/H15/H25/zero terminal ablations from deterministic H15 prefixes.",
        "states": STATE_SPECS,
        "run_plan": RUN_PLAN,
        "budget": {"episodes_exact": len(RUN_PLAN), "control_step_upper_bound": len(RUN_PLAN) * MAX_STEPS, "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "input_backup_proof": backup,
        "analysis_rules": {"reference": "per_h terminal H15 for the same state", "material_gain_threshold_abs": MATERIAL_GAIN, "terminal_artifact_flag": "material label or major gain changes between per_h and h15/h25/zero terminal modes at identical state+horizon"},
    }
    write_json(PROTOCOL_JSON, protocol)
    lines = [
        "# Vehicle Stage2 terminal/objective instrumentation smoke v0",
        "",
        f"Frozen UTC: `{created}`. Development-only; no validation64 or sealed test; no training/refit.",
        "",
        "Hypothesis: sparse corrected Stage2 labels may reflect learned terminal-value/objective mismatch rather than absence of horizon opportunity. The diagnostic reproduces H15 prefixes and compares branch horizons with per-H, H15, H25 and zero terminal modes where informative.",
        "",
        f"Episodes: `{len(RUN_PLAN)}`; control-step cap `{len(RUN_PLAN) * MAX_STEPS}`.",
        "",
        "## Run plan",
        "",
        "| exec | state | case | step | role | H | terminal mode |",
        "|---:|---|---:|---:|---|---:|---|",
    ]
    spec = {s["state_id"]: s for s in STATE_SPECS}
    for i, item in enumerate(RUN_PLAN):
        s = spec[item["state_id"]]
        lines.append("| %d | `%s` | %d | %d | `%s` | %d | `%s` |" % (i, item["state_id"], int(s["case"]), int(s["branch_step"]), s["role"], int(item["horizon"]), item["terminal_mode"]))
    PROTOCOL_MD.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    block = f"""<!-- {MARKER} -->
## 2026-09-28 vehicle Stage2 terminal/objective instrumentation smoke v0

UTC: {raw['created_utc']}. Development-only terminal/objective smoke completed: {raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps. No validation64/test access and no training/refit. Per-H positive counts by terminal mode: {a['positive_counts_by_terminal_mode']}; terminal-mode material-label flips: {len(a['material_label_flips_under_terminal_ablation'])}. Case5-step18 H30 per-H branch value/objective recorded in `{rel(OUT / 'summary.md')}` and raw artifacts. Decision remains no retraining/refit from this smoke alone; use results to choose terminal refit vs stress-v1 scenario opportunity mapping. Backup request: `{raw['backup_request']}`.
"""
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        if p.exists():
            old = p.read_text(encoding="utf-8")
            if MARKER not in old:
                p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = [
        "# Vehicle Stage2 terminal/objective instrumentation smoke v0",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only; no training/refit, no validation64 bank, no sealed test.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Headline",
        "",
        f"- Positive counts by terminal mode: `{a['positive_counts_by_terminal_mode']}`.",
        f"- Terminal-mode material-label flips: `{len(a['material_label_flips_under_terminal_ablation'])}`.",
        f"- Case5 step18 H30 per-H: `{a['case5_h30_per_h']}`.",
        f"- Case5 step18 H30 sensitivity: `{a['case5_h30_terminal_sensitivity']}`.",
        "",
        "## Comparison rows",
        "",
        "| state | H | terminal mode | effective terminal | success | constraint | phys gain vs H15 | total gain vs H15 | branch value | objective | material |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in a["comparison_rows"]:
        lines.append("| `%s` | %d | `%s` | `%s` | `%s` | `%s` | %.6g | %.6g | %s | %s | `%s` |" % (
            r["state_id"], int(r["horizon"]), r["terminal_mode"], r["effective_terminal"], r["success"], r["constraint"],
            float(r["gain_vs_per_h_H15_physical"]), float(r["gain_vs_per_h_H15_total"]),
            "NA" if r["branch_mpc_value_fn"] is None else "%.6g" % float(r["branch_mpc_value_fn"]),
            "NA" if r["branch_objective_opt_f_num"] is None else "%.6g" % float(r["branch_objective_opt_f_num"]),
            r["material_positive_vs_per_h_H15"],
        ))
    lines += ["", "## Terminal-mode sensitivity versus per-H terminal", "", "| state | H | mode | phys delta | total delta | value delta | objective delta | flip |", "|---|---:|---|---:|---:|---:|---:|---|"]
    for r in a["terminal_sensitivity_rows"]:
        lines.append("| `%s` | %d | `%s` | %.6g | %.6g | %s | %s | `%s` |" % (
            r["state_id"], int(r["horizon"]), r["mode"], float(r["physical_delta_vs_per_h"]), float(r["total_delta_vs_per_h"]),
            "NA" if r["value_fn_delta_vs_per_h_at_branch"] is None else "%.6g" % float(r["value_fn_delta_vs_per_h_at_branch"]),
            "NA" if r["objective_delta_vs_per_h_at_branch"] is None else "%.6g" % float(r["objective_delta_vs_per_h_at_branch"]),
            bool(r["per_h_material"]) != bool(r["mode_material"]),
        ))
    lines += ["", "## Decision", "", f"- Retrain/refit now: `{a['decision']['retrain_or_refit_now']}`.", f"- Reason: {a['decision']['reason']}", f"- Next if terminal artifact strong: {a['decision']['next_if_terminal_artifact_strong']}", f"- Next if terminal artifact weak: {a['decision']['next_if_terminal_artifact_weak']}", "", f"Backup request: `{raw['backup_request']}`."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-backup-proof", required=True, type=Path)
    ap.add_argument("--i-accept-development-terminal-smoke", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_terminal_smoke:
        raise ContractError("explicit --i-accept-development-terminal-smoke required")
    if OUT.exists() and (OUT / "completed.json").exists():
        verify_done(OUT / "completed.json", check_hashes=True)
        raise SystemExit("terminal/objective smoke already completed; refusing rerun")
    if OUT.exists() and any(p.name not in ("run.lock",) for p in OUT.iterdir()):
        raise ContractError("partial output exists; inspect before rerun: %s" % rel(OUT))
    # Verify inputs and backup covering them.
    done_times = []
    for p in (STAGE2_DONE, PREFIX_DIAG_DONE, SOURCE_AUDIT_DONE, REWARD_AUDIT_DONE):
        d = verify_done(p, check_hashes=False)
        done_times.append(parse_time(d.get("created_utc")))
    min_time = max(t for t in done_times if t is not None)
    backup = verify_backup(args.input_backup_proof, min_time)
    created_start = dt.datetime.now(dt.timezone.utc).isoformat()
    freeze_protocol(created_start, backup)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {"started_utc": created_start, "pid": os.getpid(), "method": "vehicle_stage2_terminal_objective_smoke_v0", "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "training_gradient_steps": 0})
    preflight = stage1_runner.runtime_preflight()
    write_json(OUT / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %r" % preflight)
    stage1_runner.base.v1.latency_verify()
    bank = read_json(STAGE1_BANK)
    selected_cases = bank["selected_cases"]
    stage1_protocol = read_json(STAGE1_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid_for_stage1(stage1_protocol)
    write_json(OUT / "terminal_sources.json", terminal_receipts)
    spec = {s["state_id"]: s for s in STATE_SPECS}
    schedule = []
    for i, x in enumerate(RUN_PLAN):
        s = spec[x["state_id"]]
        item = dict(x)
        item.update({"execution_index": i, "case": int(s["case"]), "branch_step": int(s["branch_step"]), "role": s["role"]})
        schedule.append(item)
    write_json(OUT / "schedule.json", {"episodes": schedule, "prefix_horizon": PREFIX_H, "max_steps": MAX_STEPS})
    episodes = []
    for item in schedule:
        summary = run_one(item, selected_cases[int(item["case"])], terminals, terminal_receipts)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "episodes_done": len(episodes), "episodes_expected": len(schedule), "control_steps_done": int(sum(int(e["steps"]) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "case", "branch_step", "branch_horizon", "terminal_mode", "steps", "success", "termination")}, "historical_validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(OUT / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e["steps"]) for e in episodes))
    if len(episodes) != len(RUN_PLAN) or control_steps > len(RUN_PLAN) * MAX_STEPS:
        raise ContractError("budget violation")
    analysis = analyze(episodes)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_STAGE2_TERMINAL_OBJECTIVE_SMOKE_V0_%s.json" % created.replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created,
        "started_utc": created_start,
        "method": "vehicle_stage2_terminal_objective_smoke_v0",
        "classification": "development_IMPROVED_terminal_objective_instrumentation_smoke_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "input_backup_proof": backup,
        "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON), "md": rel(PROTOCOL_MD), "md_sha256": sha256(PROTOCOL_MD)},
        "inputs": {"stage1_bank": rel(STAGE1_BANK), "stage1_bank_sha256": sha256(STAGE1_BANK), "stage1_protocol": rel(STAGE1_PROTOCOL), "stage1_protocol_sha256": sha256(STAGE1_PROTOCOL), "stage2_raw": rel(STAGE2_RAW), "stage2_raw_sha256": sha256(STAGE2_RAW), "prefix_diagnostic_completed": rel(PREFIX_DIAG_DONE), "source_audit_completed": rel(SOURCE_AUDIT_DONE), "reward_audit_completed": rel(REWARD_AUDIT_DONE)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": {"episodes_exact": len(RUN_PLAN), "control_step_upper_bound": len(RUN_PLAN) * MAX_STEPS, "new_training_episodes": 0, "new_gradient_steps": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "environment_constructions": len(episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "terminal_sources": terminal_receipts,
        "schedule": schedule,
        "episodes": episodes,
        "analysis": analysis,
        "interpretation_limits": ["tiny development smoke only", "states were chosen from development Stage2 evidence", "not validation/model selection", "not final test", "not enough label density for training/refit", "terminal off keeps the same NLP structure by zeroing terminal weights"],
    }
    write_json(req, {"requested_utc": created, "reason": "backup terminal/objective instrumentation smoke before further simulation or scenario/method changes", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "artifacts": [rel(OUT), rel(STATE), rel(Path(__file__).resolve()), rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(req)]})
    raw["backup_request"] = rel(req)
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Vehicle Stage2 terminal/objective smoke state ({created})\n\nCompleted {len(episodes)} episodes / {control_steps} control steps. No validation64/test/training. Positive counts by terminal mode={analysis['positive_counts_by_terminal_mode']}; label flips under terminal ablation={len(analysis['material_label_flips_under_terminal_ablation'])}. Next: inspect terminal sensitivity and choose terminal-refit versus stress-v1 scenario protocol. Backup required before next simulation.\n", encoding="utf-8")
    append_docs(raw)
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE, Path(__file__).resolve(), PROTOCOL_JSON, PROTOCOL_MD, req, args.input_backup_proof]
    write_json(OUT / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created, "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "backup_request": rel(req), "headline": {"positive_counts_by_terminal_mode": analysis["positive_counts_by_terminal_mode"], "terminal_mode_material_label_flips": len(analysis["material_label_flips_under_terminal_ablation"]), "case5_h30_per_h_value": None if analysis["case5_h30_per_h"] is None else analysis["case5_h30_per_h"].get("branch_mpc_value_fn"), "case5_h30_per_h_objective": None if analysis["case5_h30_per_h"] is None else analysis["case5_h30_per_h"].get("branch_objective_opt_f_num"), "next_action": "inspect terminal sensitivity; then freeze terminal-refit or stress-v1 scenario protocol"}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "positive_counts_by_terminal_mode": analysis["positive_counts_by_terminal_mode"], "terminal_label_flips": len(analysis["material_label_flips_under_terminal_ablation"]), "case5_h30_per_h": analysis["case5_h30_per_h"], "case5_h30_terminal_sensitivity": analysis["case5_h30_terminal_sensitivity"], "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {"failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_episodes": 0, "new_gradient_steps": 0, "next_recovery_hint": "Preserve partial output. If failure occurred before any episode, repair the instrumented smoke source with a one-variable version; if after episodes, audit partial traces before rerun."})
        raise
