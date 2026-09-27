#!/usr/bin/env python3
"""Instrumented deterministic replay/ablation for vehicle validation case43.

Development diagnostic only. This intentionally reopens the already-used vehicle
validation bank case43 and runs a small bounded set of deterministic replays to
separate the singleton H35 decision from terminal-source, warm-start and later
policy effects. It never opens/hashes the sealed final test bank and creates no
training updates.
"""

from __future__ import annotations

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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "experiments/bohn2021_reproduction"
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

from latency_tree_protocol import bank_name, verify as verify_latency_registration  # noqa:E402
from latency_tree_policy import choose, constant, features, policy_key  # noqa:E402
from conservative_canonical_reset import make_env  # noqa:E402
from branch_calibration_run import meter, observed_step  # noqa:E402
from branch_calibration_audit import audit_trace  # noqa:E402
from gated_horizon_search import case_metrics  # noqa:E402
from gated_horizon_timing import LoggingTimer  # noqa:E402
from relative_policy_features import context  # noqa:E402
from runtime import imports  # noqa:E402
from run import weights_hash  # noqa:E402

TASK = "vehicle"
CASE_INDEX = 43
MAX_STEPS = 150
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_case43_instrumented_replay_20260927_v1"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
GATE_JSON = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"
LEARNED_SAVED_DIR = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard07/episodes/exec1668_learned_s2_case43"
FIXED_SAVED_DIR = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard00/episodes/exec0185_fixed_seed2_terminal25_controllerH25_case43"
THIS_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_case43_instrumented_replay_v1.py"
DOC_MARKER = "vehicle-case43-instrumented-replay-v1-20260927"


class DiagnosticError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def serial(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=serial, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array(value: Any) -> np.ndarray:
    try:
        raw = value.cat if hasattr(value, "cat") else value
        return np.asarray(raw, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def arr_hash(a: np.ndarray) -> Optional[str]:
    if a.size == 0 or not np.isfinite(a).all():
        return None
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(a.astype(np.float64)).tobytes())
    return h.hexdigest()


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    data = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not data:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    s = sorted(data)
    def pct(p: float) -> float:
        if len(s) == 1:
            return s[0]
        idx = (len(s) - 1) * p / 100.0
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        if lo == hi:
            return s[lo]
        return s[lo] * (hi - idx) + s[hi] * (idx - lo)
    return {"count": len(s), "sum": float(math.fsum(s)), "mean": float(math.fsum(s) / len(s)),
            "median": float(pct(50)), "p95": float(pct(95)), "min": float(s[0]), "max": float(s[-1])}


def state_vec(row: Mapping[str, Any], key: str = "state") -> List[float]:
    s = row.get(key) or {}
    return [float(s.get("theta", math.nan)), float(s.get("x", math.nan)), float(s.get("y", math.nan))]


def input_vec(row: Mapping[str, Any]) -> List[float]:
    u = row.get("input") or {}
    return [float((u.get("u_omega") or [math.nan])[0]), float((u.get("u_s") or [math.nan])[0])]


def max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    vals = [abs(float(x) - float(y)) for x, y in zip(a, b) if math.isfinite(float(x)) and math.isfinite(float(y))]
    return float(max(vals, default=0.0))


def first_divergence(a: List[Mapping[str, Any]], b: List[Mapping[str, Any]], tol: float = 1e-8) -> Dict[str, Any]:
    n = min(len(a), len(b))
    out: Dict[str, Any] = {"comparison_steps": n, "first_pre_state": None, "first_post_state": None, "first_input": None, "first_horizon": None}
    for i in range(n):
        if out["first_pre_state"] is None and max_abs_diff(state_vec(a[i], "previous_state"), state_vec(b[i], "previous_state")) > tol:
            out["first_pre_state"] = i
        if out["first_post_state"] is None and max_abs_diff(state_vec(a[i], "state"), state_vec(b[i], "state")) > tol:
            out["first_post_state"] = i
        if out["first_input"] is None and max_abs_diff(input_vec(a[i]), input_vec(b[i])) > tol:
            out["first_input"] = i
        if out["first_horizon"] is None and int(a[i].get("horizon")) != int(b[i].get("horizon")):
            out["first_horizon"] = i
    out["max_post_state_abs_diff_common_prefix"] = max((max_abs_diff(state_vec(a[i], "state"), state_vec(b[i], "state")) for i in range(n)), default=0.0)
    out["max_input_abs_diff_common_prefix"] = max((max_abs_diff(input_vec(a[i]), input_vec(b[i])) for i in range(n)), default=0.0)
    return out


def install_instrumented_recovery(mpc: Any, folder: Path) -> Dict[str, Any]:
    """Copy conservative recovery but add pre/post warm-start hashes and objective."""
    from casadi import DM
    original = mpc.solve
    state: Dict[str, Any] = {"enabled": False, "events": [], "counts": {"solve_attempts": 0, "solve_completed": 0, "warmup_attempts": 0, "retry_attempts": 0}, "step": None, "case": None}
    attempts = folder / "solver_attempts.json"
    if attempts.exists():
        raise DiagnosticError(f"solver_attempts already exists: {rel(attempts)}")
    write_json(attempts, state["counts"])

    def finite_summary(a: np.ndarray) -> Dict[str, Any]:
        if a.size == 0:
            return {"size": 0, "finite": None, "sha256": None, "mean": None, "max_abs": None}
        finite = bool(np.isfinite(a).all())
        return {"size": int(a.size), "finite": finite, "sha256": arr_hash(a),
                "mean": float(np.mean(a)) if finite else None, "max_abs": float(np.max(np.abs(a))) if finite else None}

    def residual(v: np.ndarray, lo: Any, hi: Any) -> Optional[float]:
        if v.size == 0 or not np.isfinite(v).all():
            return None
        lo_a, hi_a = array(lo), array(hi)
        if lo_a.size != v.size or hi_a.size != v.size:
            return None
        return float(np.maximum.reduce([np.zeros_like(v), lo_a - v, v - hi_a]).max())

    def opt_scalar(name: str) -> Optional[float]:
        try:
            a = array(getattr(mpc, name))
            if a.size and np.isfinite(a).all():
                return float(a.reshape(-1)[0])
        except Exception:
            return None
        return None

    def call(kind: str) -> Dict[str, Any]:
        c = state["counts"]
        c["solve_attempts"] += 1
        if kind == "warmup":
            c["warmup_attempts"] += 1
        if kind.startswith("retry"):
            c["retry_attempts"] += 1
        write_json(attempts, c)
        pre_x = array(getattr(mpc, "opt_x_num", []))
        pre_x_unscaled = array(getattr(mpc, "opt_x_num_unscaled", []))
        start = time.perf_counter()
        original()
        elapsed = time.perf_counter() - start
        c["solve_completed"] += 1
        write_json(attempts, c)
        post_x = array(getattr(mpc, "opt_x_num", []))
        post_x_unscaled = array(getattr(mpc, "opt_x_num_unscaled", []))
        g = array(getattr(mpc, "opt_g_num", []))
        finite = bool((post_x.size == 0 or np.isfinite(post_x).all()) and (g.size == 0 or np.isfinite(g).all()))
        gr = residual(g, mpc.cons_lb, mpc.cons_ub) if g.size else None
        xr = residual(post_x, mpc.lb_opt_x, mpc.ub_opt_x) if post_x.size else None
        stats = dict(getattr(mpc, "solver_stats", {}) or {})
        accepted = finite and bool(stats.get("success", False)) and (gr is None or gr <= 1e-5) and (xr is None or xr <= 1e-5)
        row = {
            "case": state["case"], "step": state["step"], "kind": kind,
            "success": bool(stats.get("success", False)), "return_status": stats.get("return_status"),
            "iterations": int(stats.get("iter_count", -1)) if stats.get("iter_count") is not None else None,
            "finite": finite, "constraint_residual": gr, "bound_residual": xr, "solver_s": float(elapsed), "accepted": bool(accepted),
            "objective_opt_f_num": opt_scalar("opt_f_num"),
            "warm_start_pre_opt_x": finite_summary(pre_x),
            "warm_start_post_opt_x": finite_summary(post_x),
            "warm_start_pre_opt_x_unscaled": finite_summary(pre_x_unscaled),
            "warm_start_post_opt_x_unscaled": finite_summary(post_x_unscaled),
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
            initial_solution = array(mpc.opt_x_num).copy()
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
                    mpc.opt_x_num_unscaled.master = DM(initial_solution * array(mpc.opt_x_scaling))
                except Exception:
                    pass
                for name, value in saved.items():
                    setattr(mpc, name, value)
                mpc.calculate_aux_num()
        state["events"].append({"attempts": records, "recovered": (not records[0]["accepted"] and records[-1]["accepted"]),
                                "final_success": bool(mpc.solver_stats.get("success", False)), "restored_original": not records[-1]["accepted"]})
    mpc.solve = solve
    return state


def safe_value(model: Any, obs: Any) -> Optional[float]:
    try:
        arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if hasattr(model, "value_target") and hasattr(model, "next_observations_ph"):
            return float(model.sess.run(model.value_target, {model.next_observations_ph: arr})[0, 0])
        if hasattr(model, "value_fn") and hasattr(model, "observations_ph"):
            return float(model.sess.run(model.value_fn, {model.observations_ph: arr})[0, 0])
    except Exception:
        return None
    return None


def horizon_counts(trace: List[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in trace:
        h = str(r.get("horizon"))
        counts[h] = counts.get(h, 0) + 1
    return counts


def load_gate() -> Dict[str, Any]:
    gate = read_json(GATE_JSON)
    if gate.get("test_accessed") is not False or gate.get("sealed_test_bank_content_opened") is not False:
        raise DiagnosticError("Gate unexpectedly records test access")
    return gate


def find_arm(gate: Mapping[str, Any], rollout_key: str) -> Dict[str, Any]:
    for arm in gate.get("unique_rollout_arms") or []:
        if arm.get("rollout_key") == rollout_key:
            return dict(arm)
    raise DiagnosticError(f"Missing rollout arm {rollout_key}")


def find_fixed_h25_seed2_arm(gate: Mapping[str, Any]) -> Dict[str, Any]:
    for arm in gate.get("unique_rollout_arms") or []:
        key = str(arm.get("rollout_key"))
        if key.startswith("fixed_seed2") and "terminal25" in key and "controllerH25" in key:
            return dict(arm)
    raise DiagnosticError("Missing fixed seed2 terminal25 controllerH25 arm")


def load_policy(path: Path) -> Dict[str, Any]:
    policy = read_json(path)
    if policy.get("task") != TASK:
        raise DiagnosticError(f"Policy task mismatch: {rel(path)}")
    return policy


def model_paths_for_source(source_rel: str) -> Tuple[Path, Path, Path]:
    folder = ROOT / source_rel
    return folder / "model.zip", folder / "manifest.json", folder / "completed.json"


def load_terminal_model(source_rel: str) -> Tuple[Any, Tuple[Any, Any], Dict[str, Any]]:
    model_zip, manifest_path, completed_path = model_paths_for_source(source_rel)
    _, SAC, _ = imports()
    manifest = read_json(manifest_path)
    done = read_json(completed_path)
    if manifest.get("task") != TASK or done.get("status") != "complete":
        raise DiagnosticError(f"Bad terminal source: {source_rel}")
    model = SAC.load(str(model_zip))
    wh = weights_hash(model)
    if wh != done.get("final_hash"):
        model.sess.close()
        raise DiagnosticError(f"Loaded weights hash mismatch for {source_rel}")
    terminal = model.policy_tf.get_mpc_vfn_weights_and_biases()
    receipt = {"source": source_rel, "model_zip_sha256": sha256(model_zip), "manifest_sha256": sha256(manifest_path),
               "completed_sha256": sha256(completed_path), "weights_hash": wh,
               "manifest": {k: manifest.get(k) for k in ("task", "seed", "fixed_horizon", "steps")}}
    return model, terminal, receipt


def compact_recovery(event: Mapping[str, Any]) -> Dict[str, Any]:
    attempts = []
    for a in event.get("attempts", []):
        attempts.append({k: a.get(k) for k in (
            "kind", "success", "return_status", "iterations", "finite", "constraint_residual", "bound_residual",
            "solver_s", "accepted", "objective_opt_f_num", "warm_start_pre_opt_x", "warm_start_post_opt_x",
            "warm_start_pre_opt_x_unscaled", "warm_start_post_opt_x_unscaled")})
    return {"attempts": attempts, "recovered": event.get("recovered"), "final_success": event.get("final_success"), "restored_original": event.get("restored_original")}


def run_scenario(scenario: Mapping[str, Any], case: Mapping[str, Any], terminal_cache: Dict[str, Tuple[Any, Tuple[Any, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    sid = str(scenario["id"])
    out = OUT_DIR / "episodes" / sid
    out.mkdir(parents=True, exist_ok=False)
    arm = dict(scenario["arm"])
    source = arm["terminal_source"]
    if source not in terminal_cache:
        terminal_cache[source] = load_terminal_model(source)
    model, terminal, receipt = terminal_cache[source]
    env = make_env(TASK, int(arm["seed"]), aligned=True, scaled_obs=True)
    counts = meter(env, out)
    env.set_value_function_weights_and_biases(*terminal)
    controller = env.control_system.controller
    original_get_action = controller.get_action
    measured: List[Dict[str, float]] = []
    trace: List[Dict[str, Any]] = []
    start_wall = time.perf_counter()
    write_json(out / "scenario.json", {k: v for k, v in scenario.items() if k != "policy"})
    with LoggingTimer(out) as logging:
        recovery = install_instrumented_recovery(controller.mpc, out)
        def timed(*args: Any, **kwargs: Any) -> Any:
            before = logging.seconds
            t0 = time.perf_counter()
            value = original_get_action(*args, **kwargs)
            gross = time.perf_counter() - t0
            logged = logging.seconds - before
            measured.append({"controller_gross_s": float(gross), "logging_s": float(logged), "controller_s": float(gross - logged)})
            return value
        controller.get_action = timed
        recovery.update(enabled=False, events=[], case=CASE_INDEX, step=-1)
        reset_t0 = time.perf_counter()
        obs = env.reset(**copy.deepcopy(dict(case)))
        reset_s = time.perf_counter() - reset_t0
        if len(measured) != 1:
            raise DiagnosticError(f"{sid}: expected one reset measurement, got {len(measured)}")
        reset_record = {"reset_gross_s": float(reset_s), **measured.pop()}
        write_json(out / "reset.json", reset_record)
        recovery["enabled"] = True
        previous_initial = False
        previous_final = False
        overrides = {int(k): int(v) for k, v in (scenario.get("overrides") or {}).items()}
        raw_path = out / "trace_compact.jsonl"
        with raw_path.open("x", encoding="utf-8") as stream:
            for step in range(MAX_STEPS):
                recovery["step"] = step
                select_t0 = time.perf_counter()
                ctx = context(env, TASK)
                ctx.update(previous_initial_failure=previous_initial, previous_final_failure=previous_final)
                base_h, decision = choose(scenario["policy"], ctx)
                selected_h = int(overrides.get(step, base_h))
                if step in overrides:
                    decision = {"kind": "forced_override", "base_h": int(base_h), "forced_h": selected_h, "base_decision": decision}
                selection_s = time.perf_counter() - select_t0
                value_before = safe_value(model, obs)
                _, terminated, row = observed_step(env, TASK, selected_h, dict(case), step)
                obs = np.asarray(row["next_observation"], dtype=float)
                value_after = safe_value(model, obs)
                if len(measured) != 1:
                    raise DiagnosticError(f"{sid}: expected one step measurement at step {step}, got {len(measured)}")
                timing = measured.pop()
                timing.update({"selection_s": float(selection_s), "decision_s": float(selection_s + timing["controller_s"]),
                               "decision_gross_s": float(selection_s + timing["controller_gross_s"])})
                event = copy.deepcopy(recovery["events"][-1])
                compact = {
                    "step": step,
                    "horizon": int(row["horizon"]),
                    "base_policy_h": int(base_h),
                    "override_applied": bool(step in overrides),
                    "decision": decision,
                    "previous_state": row.get("previous_state"),
                    "state": row.get("state"),
                    "input": row.get("input"),
                    "observation": row.get("observation"),
                    "next_observation": row.get("next_observation"),
                    "tree_features": features(TASK, ctx),
                    "reward": float(row["reward"]),
                    "performance": float(row["performance"]),
                    "compute": float(row["compute"]),
                    "constraint": float(row["constraint"]),
                    "termination": row.get("termination"),
                    "solver_success": bool(row["solver_success"]),
                    "timing": timing,
                    "terminal_value_estimate_before": value_before,
                    "terminal_value_estimate_after": value_after,
                    "recovery": compact_recovery(event),
                }
                stream.write(json.dumps(compact, default=serial, allow_nan=False) + "\n")
                stream.flush()
                trace.append(compact)
                previous_initial = not bool(event["attempts"][0]["success"])
                previous_final = not bool(row["solver_success"])
                if terminated:
                    break
    if not trace or not trace[-1].get("termination"):
        raise DiagnosticError(f"{sid}: episode did not terminate within {MAX_STEPS}")
    audit_trace(TASK, dict(case), trace)
    metric = case_metrics(TASK, trace)
    decision = [float(r["timing"]["decision_s"]) for r in trace]
    solver_rows = read_jsonl(out / "solver_calls.jsonl")
    summary: Dict[str, Any] = dict(metric)
    summary.update({
        "scenario_id": sid,
        "description": scenario.get("description"),
        "terminal_source": source,
        "terminal_receipt": receipt,
        "policy_key": policy_key(scenario["policy"]),
        "policy_kind": scenario["policy"].get("kind"),
        "overrides": {str(k): v for k, v in overrides.items()},
        "horizon_counts": horizon_counts(trace),
        "override_steps": sorted(overrides),
        "decision_timing_s": values_summary(decision),
        "solver_s": values_summary([r.get("solver_s") for r in solver_rows if r.get("step", -1) >= 0]),
        "solver_iterations": values_summary([r.get("iterations") for r in solver_rows if r.get("step", -1) >= 0 and r.get("iterations") is not None]),
        "solver_retry_rows": int(sum(1 for r in solver_rows if str(r.get("kind", "")).startswith("retry"))),
        "all_solver_accepted": bool(all(r.get("accepted") for r in solver_rows if r.get("step", -1) >= 0)),
        "objective_window_steps0_8": [
            {"step": r["step"], "h": r["horizon"], "objective": ((r.get("recovery") or {}).get("attempts") or [{}])[-1].get("objective_opt_f_num"),
             "pre_x_hash": ((r.get("recovery") or {}).get("attempts") or [{}])[-1].get("warm_start_pre_opt_x", {}).get("sha256"),
             "post_x_hash": ((r.get("recovery") or {}).get("attempts") or [{}])[-1].get("warm_start_post_opt_x", {}).get("sha256"),
             "tv_before": r.get("terminal_value_estimate_before"), "tv_after": r.get("terminal_value_estimate_after")}
            for r in trace[:9]
        ],
        "trace_compact": rel(out / "trace_compact.json"),
        "trace_jsonl": rel(raw_path),
        "solver_calls_jsonl": rel(out / "solver_calls.jsonl"),
        "reset": reset_record,
        "wall_s": float(time.perf_counter() - start_wall),
        "meter_counts": counts,
    })
    write_json(out / "trace_compact.json", trace)
    write_json(out / "summary.json", summary)
    files = [p for p in out.iterdir() if p.is_file() and p.name != "completed.json"]
    write_json(out / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})
    return {"summary": summary, "trace": trace}


def compare_to_saved(trace: List[Mapping[str, Any]], saved: List[Mapping[str, Any]]) -> Dict[str, Any]:
    return first_divergence(trace, saved, tol=1e-7)


def write_summary(path: Path, raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle case43 instrumented replay/ablation v1",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Development diagnostic only: validation bank case43 was intentionally reopened; sealed final test was not opened/hashed; no training updates were run.",
        "",
        "## Scenario outcomes",
        "",
        "| scenario | success | steps | termination | physical+constraint | total | horizons | solver retries | decision mean s |",
        "|---|---:|---:|---|---:|---:|---|---:|---:|",
    ]
    for sid, s in raw["scenario_summaries"].items():
        lines.append(f"| `{sid}` | {s.get('success')} | {s.get('steps')} | {s.get('termination')} | {float(s.get('physical_constraint_cost', 0.0)):.6g} | {float(s.get('total_cost', 0.0)):.6g} | {s.get('horizon_counts')} | {s.get('solver_retry_rows')} | {s.get('decision_timing_s', {}).get('mean')} |")
    lines.extend([
        "",
        "## Key ablation interpretation",
        "",
        f"- Learned replay success: `{raw['scenario_summaries']['learned_policy_replay'].get('success')}`; learned with step3 forced to H25 success: `{raw['scenario_summaries']['learned_force_H25_at_step3'].get('success')}`.",
        f"- Constant H25 with learned terminal success: `{raw['scenario_summaries']['constant_H25_learned_terminal'].get('success')}`; constant H25 with only step3 forced H35 success: `{raw['scenario_summaries']['constant_H25_with_H35_at_step3'].get('success')}`.",
        f"- Registered fixed seed2 terminal25 H25 success: `{raw['scenario_summaries']['registered_fixed_seed2_terminal25_H25'].get('success')}`.",
        f"- Programmatic diagnosis: {raw['programmatic_diagnosis']}",
        "",
        "## Pairwise divergences",
        "",
    ])
    for name, comp in raw["pairwise_divergences"].items():
        lines.append(f"- `{name}`: {comp}")
    lines.extend([
        "",
        "## Objective/warm-start notes",
        "",
        "Each replay logs solver objective `opt_f_num` when available, pre/post NLP warm-start hashes and summaries, solver iterations/residuals, and SAC value-target estimates at realized observations. These are development diagnostics for the already-opened validation case, not fresh independent confirmation evidence.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_once(path: Path, marker: str, body: str) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    token = f"<!-- {marker} -->"
    if token in old:
        return False
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + body.strip() + "\n", encoding="utf-8")
    return True


def main() -> int:
    if (OUT_DIR / "completed.json").exists():
        done = read_json(OUT_DIR / "completed.json")
        if done.get("passed") is True:
            for name, expected in done.get("hashes", {}).items():
                if sha256(ROOT / name) != expected:
                    raise DiagnosticError(f"Existing completed hash mismatch: {name}")
            print(json.dumps({"already_completed": True, "completed": rel(OUT_DIR / "completed.json")}, sort_keys=True))
            return 0
        raise DiagnosticError("Prior completed marker exists but did not pass")
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise DiagnosticError(f"Partial output exists; preserve before retry: {rel(OUT_DIR)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    verify_latency_registration()
    gate = load_gate()
    learned_arm = find_arm(gate, "learned_s2")
    fixed_arm = find_fixed_h25_seed2_arm(gate)
    policy_path = ROOT / next(c["policy_path"] for c in gate["learned_candidates"] if c.get("rollout_key") == "learned_s2" or c.get("seed") == 2)
    learned_policy = load_policy(policy_path)
    validation_bank = bank_name(TASK, "validation")
    bank_sha = sha256(validation_bank)
    gate_bank_sha = gate["splits"]["validation64"].get("recorded_sha256_from_banks_completed")
    if gate_bank_sha and bank_sha != gate_bank_sha:
        raise DiagnosticError("Validation bank hash differs from frozen gate metadata")
    bank = read_json(validation_bank)
    case = bank["cases"][CASE_INDEX]
    saved_learned = read_json(LEARNED_SAVED_DIR / "trace.json")
    saved_fixed = read_json(FIXED_SAVED_DIR / "trace.json")

    scenarios = [
        {"id": "learned_policy_replay", "description": "stored learned_s2 tree and learned terminal, no override", "arm": learned_arm, "policy": learned_policy, "overrides": {}},
        {"id": "learned_force_H25_at_step3", "description": "learned_s2 tree but force the singleton saved H35 step (t=3) to H25", "arm": learned_arm, "policy": learned_policy, "overrides": {3: 25}},
        {"id": "constant_H25_learned_terminal", "description": "constant H25 policy using the learned_s2 terminal source", "arm": learned_arm, "policy": constant(TASK, 25), "overrides": {}},
        {"id": "constant_H25_with_H35_at_step3", "description": "constant H25 policy using learned_s2 terminal, except force H35 at t=3", "arm": learned_arm, "policy": constant(TASK, 25), "overrides": {3: 35}},
        {"id": "registered_fixed_seed2_terminal25_H25", "description": "registered same-seed fixed terminal25/controllerH25 comparator arm", "arm": fixed_arm, "policy": constant(TASK, 25), "overrides": {}},
    ]
    terminal_cache: Dict[str, Tuple[Any, Tuple[Any, Any], Dict[str, Any]]] = {}
    results: Dict[str, Dict[str, Any]] = {}
    try:
        for sc in scenarios:
            result = run_scenario(sc, case, terminal_cache)
            results[str(sc["id"])] = result
    finally:
        for model, _, _ in terminal_cache.values():
            try:
                model.sess.close()
            except Exception:
                pass

    scenario_summaries = {sid: res["summary"] for sid, res in results.items()}
    traces = {sid: res["trace"] for sid, res in results.items()}
    pairwise = {
        "learned_replay_vs_saved_learned": compare_to_saved(traces["learned_policy_replay"], saved_learned),
        "registered_fixed_replay_vs_saved_fixed": compare_to_saved(traces["registered_fixed_seed2_terminal25_H25"], saved_fixed),
        "learned_force_H25_vs_learned_replay": first_divergence(traces["learned_force_H25_at_step3"], traces["learned_policy_replay"], tol=1e-8),
        "constant_H25_step3H35_vs_constant_H25": first_divergence(traces["constant_H25_with_H35_at_step3"], traces["constant_H25_learned_terminal"], tol=1e-8),
        "constant_H25_learned_terminal_vs_registered_fixed": first_divergence(traces["constant_H25_learned_terminal"], traces["registered_fixed_seed2_terminal25_H25"], tol=1e-8),
    }
    learned_failed = not bool(scenario_summaries["learned_policy_replay"].get("success"))
    forced_h25_success = bool(scenario_summaries["learned_force_H25_at_step3"].get("success"))
    constant_h25_success = bool(scenario_summaries["constant_H25_learned_terminal"].get("success"))
    h35_on_h25_success = bool(scenario_summaries["constant_H25_with_H35_at_step3"].get("success"))
    if learned_failed and forced_h25_success and constant_h25_success and h35_on_h25_success:
        diagnosis = "forcing H25 at the singleton H35 rescues the learned replay, while a single H35 perturbation on the constant-H25 path remains successful; H35 is a sensitive branching event for the learned trajectory but is not by itself universally catastrophic."
    elif learned_failed and forced_h25_success and constant_h25_success and not h35_on_h25_success:
        diagnosis = "forcing H25 at the singleton H35 rescues the learned replay, and a single H35 perturbation on the constant-H25 path fails; evidence supports the step3 horizon intervention as a direct cause of this case43 failure under this terminal/source setup."
    elif learned_failed and not forced_h25_success:
        diagnosis = "forcing H25 at step3 did not rescue the learned replay; later state/value/solver/policy effects dominate or the replay differs from saved evidence."
    else:
        diagnosis = "learned replay did not reproduce the saved failure or produced an unexpected combination; inspect raw traces before causal claims."

    raw_path = OUT_DIR / "raw.json"
    summary_path = OUT_DIR / "summary.md"
    episode_csv = OUT_DIR / "episode_summary.csv"
    with episode_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["scenario_id", "success", "steps", "termination", "total_cost", "physical_constraint_cost", "horizon_counts", "solver_retry_rows", "all_solver_accepted", "wall_s"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sid, s in scenario_summaries.items():
            w.writerow({k: json.dumps(s.get(k), sort_keys=True) if k == "horizon_counts" else s.get(k) for k in fields})
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started.isoformat(),
        "method": "IMPROVED_latency_tree_vehicle_case43_instrumented_replay_v1_development_not_original_SAC",
        "passed": True,
        "validation_accessed": True,
        "validation_access_type": "reopened already-used vehicle validation bank to replay case43 for development diagnosis",
        "validation_bank_reopened": True,
        "validation64_bank_content_opened": True,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "test_accessed": False,
        "formal_scientific_evidence_created": False,
        "new_gradient_steps": 0,
        "new_simulations": len(scenarios),
        "new_validation_episodes": len(scenarios),
        "new_control_steps": int(sum(int(s.get("steps", 0)) for s in scenario_summaries.values())),
        "budget_declared": {"episodes": len(scenarios), "control_step_upper_bound": len(scenarios) * MAX_STEPS, "gradient_steps": 0, "sealed_test_episodes": 0},
        "budget_actual": {"episodes": len(scenarios), "control_steps": int(sum(int(s.get("steps", 0)) for s in scenario_summaries.values())), "gradient_steps": 0, "sealed_test_episodes": 0},
        "inputs": {"gate_json": rel(GATE_JSON), "validation_bank": rel(validation_bank), "case_index": CASE_INDEX, "learned_policy": rel(policy_path), "saved_learned_trace": rel(LEARNED_SAVED_DIR / "trace.json"), "saved_fixed_trace": rel(FIXED_SAVED_DIR / "trace.json")},
        "input_hashes": {rel(p): sha256(p) for p in [GATE_JSON, validation_bank, policy_path, LEARNED_SAVED_DIR / "trace.json", FIXED_SAVED_DIR / "trace.json", THIS_SCRIPT]},
        "scenario_summaries": scenario_summaries,
        "pairwise_divergences": pairwise,
        "programmatic_diagnosis": diagnosis,
        "interpretation_limits": [
            "This is a development diagnostic on an already-inspected validation case; it cannot be counted as fresh independent validation or final-test evidence.",
            "Timing is real same-host timing for these replays, but single-case diagnostic timing is not a model-selection timing block.",
            "Terminal-value estimates are SAC value-target evaluations at realized observations; exact predicted-terminal MPC value decomposition may require deeper do-mpc symbolic instrumentation.",
        ],
    }
    write_json(raw_path, raw)
    raw["artifacts"] = {"raw": rel(raw_path), "summary": rel(summary_path), "episode_summary_csv": rel(episode_csv), "completed": rel(OUT_DIR / "completed.json")}
    write_json(raw_path, raw)
    write_summary(summary_path, raw)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%z")
    backup_request = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_CASE43_INSTRUMENTED_REPLAY_V1_{stamp}.json"
    write_json(backup_request, {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "purpose": "backup after vehicle case43 instrumented replay v1 development diagnostic",
                                "validation_accessed": True, "validation_bank_reopened": True, "test_accessed": False, "sealed_test_bank_content_opened": False,
                                "new_simulations": len(scenarios), "new_control_steps": raw["new_control_steps"],
                                "artifacts_to_cover": [rel(OUT_DIR), rel(THIS_SCRIPT)],
                                "pre_completed_hashes": {rel(p): sha256(p) for p in [raw_path, summary_path, episode_csv, THIS_SCRIPT]}})
    doc_body = f"""
## 2026-09-27 vehicle case43 instrumented replay/ablation v1

UTC: {raw['created_utc']}. Development diagnostic intentionally reopened the already-used vehicle validation bank case43 and ran {len(scenarios)} deterministic replay/ablation episodes ({raw['new_control_steps']} new control steps, 0 gradient steps, sealed test closed). This is not fresh independent validation/model-selection evidence.

Outcome: {diagnosis}

Artifacts: `{rel(raw_path)}`, `{rel(summary_path)}`, `{rel(episode_csv)}`, completed marker `{rel(OUT_DIR / 'completed.json')}`. Backup requested at `{rel(backup_request)}`.
"""
    docs_updated: List[str] = []
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md"):
        p = ROOT / doc
        if p.exists() and append_once(p, DOC_MARKER, doc_body):
            docs_updated.append(doc)
    raw["docs_updated"] = docs_updated
    raw["artifacts"]["backup_request"] = rel(backup_request)
    write_json(raw_path, raw)
    write_summary(summary_path, raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [backup_request, THIS_SCRIPT]
    for doc in docs_updated:
        files.append(ROOT / doc)
    write_json(OUT_DIR / "completed.json", {"passed": True, "validation_accessed": True, "validation_bank_reopened": True,
                                             "validation64_bank_content_opened": True, "test_accessed": False,
                                             "sealed_test_bank_content_opened": False, "sealed_test_bank_hashed": False,
                                             "formal_scientific_evidence_created": False,
                                             "new_simulations": len(scenarios), "new_validation_episodes": len(scenarios),
                                             "new_control_steps": raw["new_control_steps"], "new_gradient_steps": 0,
                                             "programmatic_diagnosis": diagnosis,
                                             "backup_request": rel(backup_request),
                                             "hashes": {rel(p): sha256(p) for p in sorted(set(files))}})
    print(json.dumps({"completed": rel(OUT_DIR / "completed.json"), "raw": rel(raw_path), "summary": rel(summary_path),
                      "new_simulations": len(scenarios), "new_control_steps": raw["new_control_steps"],
                      "test_accessed": False, "diagnosis": diagnosis}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            write_json(OUT_DIR / "failure.json", {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "error": repr(exc),
                                                   "validation_accessed": True, "test_accessed": False, "new_gradient_steps": 0})
        except Exception:
            pass
        raise
