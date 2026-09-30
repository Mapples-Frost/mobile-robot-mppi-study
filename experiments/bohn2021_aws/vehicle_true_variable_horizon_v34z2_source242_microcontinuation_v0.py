#!/usr/bin/env python3
"""S-TC2H source242 fixed-H closed-loop microcontinuation.

Temporary GPT-5.5 solo task S-TC2H-source242-fixedH-microcontinuation-v0.
Development-only opened-source242 diagnostic: fixed H=[12,15,35], V15_shared,
no training/refit, no validation64, no sealed/final test. The script refuses to
spend solver/plant resources unless an external backup proof after S-TC2G is
locally visible.
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments" / "bohn2021_aws"
SERVICE_DIR = ROOT / "scripts" / "research_service"
for _p in (SERVICE_DIR, AWS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import execution_contract  # type: ignore  # noqa:E402
import vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0 as gate  # type: ignore  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0"
TASK_ID = "S-TC2H-source242-fixedH-microcontinuation-v0"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}
HORIZONS = [12, 15, 35]
MAX_STEPS_PER_ARM = 60
SOLVER_CAP = 180
PLANT_CAP = 180
TERMINAL_MODE = "V15_shared"
EXPECTED_GATE_SHA256 = "66dbf84e4a7917d824152cae1cac4da77c51efff576cd4775e8cd1be29dbcc97"
TC2_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0_20260930T163737Z/completed.json"
TC2_CSV = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0_20260930T163737Z/cell_metrics.csv"
STC2G_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_h15_initialization_basin_triage_v0_20260930T165402Z/completed.json"
STC2G_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_h15_initialization_basin_triage_v0_20260930T165402Z/raw.json"
STC2G_CSV = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_h15_initialization_basin_triage_v0_20260930T165402Z/cell_metrics.csv"
STC2G_SOURCE = AWS_DIR / "vehicle_true_variable_horizon_v34z2_h15_initialization_basin_triage_v0.py"
GATE_SOURCE = AWS_DIR / "vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0.py"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_S_TC2G_H15_INITIALIZATION_BASIN_TRIAGE_20260930T165402Z.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
DOCS = [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]


class MicroError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Any) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(v: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore
    if isinstance(v, Path):
        return rel(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if np is not None and isinstance(v, (np.integer,)):
        return int(v)
    if np is not None and isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Mapping):
        return {str(k): clean(val) for k, val in v.items() if k != "nlp_expr"}
    if isinstance(v, (list, tuple, set)):
        return [clean(x) for x in v]
    if hasattr(v, "item"):
        try:
            return clean(v.item())
        except Exception:
            pass
    if hasattr(v, "tolist"):
        try:
            return clean(v.tolist())
        except Exception:
            pass
    return v


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def run_git(args: Sequence[str]) -> Dict[str, Any]:
    cmd = ["git", "-C", str(ROOT)] + list(args)
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30, universal_newlines=True)
        return {"ok": True, "stdout": out.strip(), "cmd": cmd}
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc), "cmd": cmd}


def read_api_total_tokens() -> Dict[str, Any]:
    for db in [ROOT / "research.sqlite", ROOT / "research_artifacts" / "research.sqlite", ROOT / "docs" / "research.sqlite"]:
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db))
            try:
                totals: Dict[str, int] = {}
                tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()]
                for table in tables:
                    cols = [r[1] for r in con.execute("pragma table_info(%s)" % table).fetchall()]
                    if "total_tokens" in cols:
                        totals[table] = int(con.execute("select coalesce(sum(total_tokens),0) from %s" % table).fetchone()[0] or 0)
                if totals:
                    return {"available": True, "path": rel(db), "table_sums": totals, "total_tokens": int(sum(totals.values()))}
            finally:
                con.close()
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": "%s: %s" % (type(exc).__name__, exc)}
    return {"available": False, "path": None, "error": "research.sqlite not found in checked repository locations"}


def timing_summary(values: Sequence[Any]) -> Dict[str, Any]:
    xs: List[float] = []
    for v in values:
        try:
            f = float(v)
            if math.isfinite(f):
                xs.append(f)
        except Exception:
            pass
    if not xs:
        return {"n": 0, "mean_s": None, "median_s": None, "p95_s": None, "sum_s": 0.0, "min_s": None, "max_s": None}
    xs.sort()
    if len(xs) == 1:
        p95 = xs[0]
    else:
        pos = 0.95 * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos)); frac = pos - lo
        p95 = xs[lo] * (1 - frac) + xs[hi] * frac
    return {"n": len(xs), "mean_s": float(mean(xs)), "median_s": float(median(xs)), "p95_s": float(p95), "sum_s": float(sum(xs)), "min_s": float(xs[0]), "max_s": float(xs[-1])}


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def verified_backup_candidates(min_time: dt.datetime) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in sorted(BACKUP_DIR.glob("*.json")):
        try:
            obj = read_json(path)
        except Exception:
            continue
        ok_status = obj.get("backup_verified") is True or obj.get("status") == "verified"
        if not ok_status or int(obj.get("remaining_changed_files", -1)) != 0:
            continue
        t = None
        for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
            t = parse_time(obj.get(key))
            if t is not None:
                break
        if t is None or t < min_time:
            continue
        packages = obj.get("packages_this_run") or []
        has_sha = bool(obj.get("asset_sha256") or obj.get("release_asset_sha256") or obj.get("package_sha256") or packages)
        if not has_sha:
            continue
        out.append({"path": rel(path), "sha256": sha256(path), "time": t.isoformat(), "commit": obj.get("commit"), "package_sha256": obj.get("package_sha256"), "packages_this_run": packages})
    return out


def verify_pre_resource_backup_and_priors() -> Dict[str, Any]:
    for p in [TC2_DONE, TC2_CSV, STC2G_DONE, STC2G_RAW, STC2G_CSV, STC2G_SOURCE, GATE_SOURCE, BACKUP_REQUEST]:
        if not p.exists():
            raise MicroError("required prior/backup-request path missing: " + rel(p))
    if sha256(GATE_SOURCE) != EXPECTED_GATE_SHA256:
        raise MicroError("wrapped v34z2 gate source hash changed")
    stc2g_done = read_json(STC2G_DONE)
    done_time = parse_time(stc2g_done.get("created_utc")) or dt.datetime.fromtimestamp(STC2G_DONE.stat().st_mtime, dt.timezone.utc)
    prior_paths = [rel(TC2_DONE), rel(TC2_CSV), rel(STC2G_DONE), rel(STC2G_RAW), rel(STC2G_CSV), rel(STC2G_SOURCE), rel(GATE_SOURCE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", rel(RESPONSE_LOG)]
    status = run_git(["status", "--porcelain", "--"] + prior_paths)
    hashes = {p: sha256(ROOT / p) for p in prior_paths if (ROOT / p).exists() and Path(p).suffix not in ("",)}
    candidates = verified_backup_candidates(done_time)
    head = run_git(["rev-parse", "HEAD"])
    commit_ok = False
    if candidates and head.get("ok"):
        for c in candidates:
            commit = c.get("commit")
            if commit:
                anc = run_git(["merge-base", "--is-ancestor", str(commit), "HEAD"])
                if anc.get("ok") or head.get("stdout") == commit:
                    c["commit_is_current_or_ancestor"] = True
                    commit_ok = True
                else:
                    c["commit_is_current_or_ancestor"] = False
    clean_ok = bool(status.get("ok") and status.get("stdout") == "")
    ok = bool(candidates and clean_ok and commit_ok)
    return {"verified_before_solver_or_plant_steps": ok, "required_min_time": done_time.isoformat(), "backup_candidates": candidates, "current_git_head": head, "prior_paths_git_status_porcelain": status, "prior_path_hashes": hashes, "prior_paths_checked": prior_paths, "S_TC2G_completed_hard_pass": stc2g_done.get("hard_pass"), "S_TC2G_budget_actual": stc2g_done.get("budget_actual")}


def configure_context_source242_goal61(base: Any, env: Any, context: Mapping[str, Any], h: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    original = getattr(base, "extract_goal_xy", None)
    forced_goal = gate.goal_endpoint_61(context["case_snapshot"])

    def _forced_extract_goal_xy(case: Mapping[str, Any], shifted: Mapping[str, Any]) -> Tuple[float, float, str]:
        got = gate.goal_endpoint_61(case)
        return float(got["goal_x"]), float(got["goal_y"]), str(got["source"])

    setattr(base, "extract_goal_xy", _forced_extract_goal_xy)
    try:
        meta = base.configure_context_no_reset(env, context, h)
    finally:
        if original is not None:
            setattr(base, "extract_goal_xy", original)
    meta = dict(meta)
    meta["goal_x"] = float(forced_goal["goal_x"])
    meta["goal_y"] = float(forced_goal["goal_y"])
    meta["goal_source"] = forced_goal["source"]
    meta["goal_endpoint_index"] = int(forced_goal["endpoint_index"])
    repair_meta = {"source242_goal61_identity_verified": True, "goal": forced_goal, "observation_heuristic_used": False}
    return meta, repair_meta


def state_clean(base: Any, value: Any) -> Dict[str, float]:
    try:
        return {str(k): float(v) for k, v in base.state_clean(value).items()}
    except Exception:
        if isinstance(value, Mapping):
            return {str(k): float(value[k]) for k in ("theta", "x", "y") if k in value}
        return {}


def current_state(base: Any, env: Any) -> Dict[str, float]:
    cs = getattr(getattr(env, "control_system", None), "current_state", {})
    return state_clean(base, cs)


def apply_tvp_for_step(base: Any, env: Any, ctrl: Any, context: Mapping[str, Any], h: int, step: int) -> Dict[str, List[float]]:
    shifted = base.shift_case_tvp(context["case_snapshot"], int(context["tvp_start_index"]) + int(step), h)
    scalar = gate.scalarize_tvp(shifted)
    if hasattr(env.control_system, "tvps"):
        for name, values in scalar.items():
            if name in env.control_system.tvps:
                env.control_system.tvps[name].values = copy.deepcopy(values)
    ctrl._tvp_data = copy.deepcopy(scalar)
    return scalar


def action_payload(action: Any, ctrl: Any) -> Any:
    cur = getattr(ctrl, "current_input", None)
    if isinstance(cur, Mapping):
        return {str(k): float(v) for k, v in cur.items()}
    if isinstance(action, Mapping):
        return {str(k): float(v) for k, v in action.items()}
    return action


def parse_step_result(result: Any) -> Tuple[Any, Optional[float], bool, Dict[str, Any]]:
    if isinstance(result, tuple):
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, (None if reward is None else float(reward)), bool(terminated or truncated), dict(info or {})
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, (None if reward is None else float(reward)), bool(done), dict(info or {})
    return result, None, False, {}


def numeric_info_costs(info: Mapping[str, Any], reward: Optional[float]) -> Dict[str, Any]:
    costs = {k: v for k, v in info.items() if isinstance(k, str) and ("cost" in k.lower() or "reward" in k.lower() or "constraint" in k.lower() or "success" in k.lower())}
    physical = None
    for key in ("physical_constraint_cost", "physical_cost", "cost", "stage_cost", "true_cost"):
        if key in info:
            try:
                physical = float(info[key]); break
            except Exception:
                pass
    total = physical
    if total is None and reward is not None:
        total = -float(reward)
    return {"reward": reward, "reported_cost_like_info": clean(costs), "physical_or_stage_cost_best_effort": physical, "total_cost_best_effort": total}


def bool_from_info(info: Mapping[str, Any], names: Sequence[str]) -> Optional[bool]:
    for n in names:
        if n in info:
            return bool(info[n])
    return None


def create_env_with_terminal(base: Any, stage1: Any, h: int, terminal: Tuple[Any, Any]) -> Any:
    maker = None
    if hasattr(base, "case5") and hasattr(base.case5, "make_true_horizon_env"):
        maker = base.case5.make_true_horizon_env
    elif hasattr(stage1, "base") and hasattr(stage1.base, "case5") and hasattr(stage1.base.case5, "make_true_horizon_env"):
        maker = stage1.base.case5.make_true_horizon_env
    if maker is None:
        raise MicroError("could not locate case5.make_true_horizon_env")
    env = maker(int(h))
    env.set_value_function_weights_and_biases(*terminal)
    return env


def run_arm(h: int, context: Mapping[str, Any], terminals: Mapping[int, Any]) -> Dict[str, Any]:
    base = gate.MODULES["base"]
    stage1 = gate.MODULES["stage1_runner"]
    v34u = gate.MODULES["v34u"]
    terminal, term_h, term_note = base.terminal_for_mode(h, TERMINAL_MODE, terminals)
    env = create_env_with_terminal(base, stage1, h, terminal)
    ctrl = env.control_system.controller
    mpc = ctrl.mpc
    gate.CURRENT_ARM_ID = f"ctx=source242_slot0_branch_start|H{h}|term=V15_shared|microcontinuation"
    gate.CURRENT_SOLVE_MODE = "production"
    v34u.install_tta_hmpc_presolve_obj_guard()
    context_meta, goal_meta = configure_context_source242_goal61(base, env, context, h)
    scalar = gate.scalarize_tvp(context_meta["shifted_tvp"])
    context_meta["shifted_tvp"] = scalar
    ctrl.goal_x = float(context_meta["goal_x"]); ctrl.goal_y = float(context_meta["goal_y"]); ctrl._tvp_data = copy.deepcopy(scalar)
    init_meta = gate.set_initial_guess_strict(mpc, env, context_meta, "canonical", h)
    state0 = current_state(base, env)
    context_hash = canonical_hash({"context_id": context.get("context_id"), "state0": state0, "tvp_start_index": context.get("tvp_start_index"), "horizon": h, "goal": goal_meta})

    solve_events: List[Dict[str, Any]] = []
    original_solve = mpc.solve

    def counted_solve(*args: Any, **kwargs: Any) -> Any:
        if gate.RESOURCE_USAGE["solver_calls"] >= SOLVER_CAP:
            raise MicroError("solver cap exhausted in S-TC2H")
        pre = gate.capture_mpc_state(mpc)
        gate.RESOURCE_USAGE["solver_calls"] += 1
        t0 = time.perf_counter()
        exc = None
        try:
            ret = original_solve(*args, **kwargs)
        except Exception as e:
            ret = None; exc = repr(e)
        elapsed = float(time.perf_counter() - t0)
        stats = copy.deepcopy(getattr(mpc, "solver_stats", {}))
        post = gate.capture_mpc_state(mpc)
        ev = {"step_index": len(solve_events), "pre": pre, "post": post, "solver_wall_s": elapsed, "solver_exception": exc, "solver_stats": stats, "return_status": stats.get("return_status"), "success": bool(stats.get("success", False)), "iterations": stats.get("iter_count") or stats.get("iterations"), "objective_opt_f_num": gate.finite_float(getattr(mpc, "opt_f_num", None), None)}
        solve_events.append(ev)
        if exc is not None:
            raise MicroError("mpc.solve failed: " + exc)
        return ret

    mpc.solve = counted_solve
    steps: List[Dict[str, Any]] = []
    stopped = "max_steps_reached"
    first_control_hash = None
    initial_solver_failures = 0
    final_solver_failures = 0
    solver_failure_steps = 0
    success_flag = False
    constraint_flag = False
    for step in range(MAX_STEPS_PER_ARM):
        if gate.RESOURCE_USAGE["plant_steps"] >= PLANT_CAP:
            stopped = "global_plant_cap_reached"
            break
        state_before = current_state(base, env)
        tvp_values = apply_tvp_for_step(base, env, ctrl, context, h, step)
        before_solves = len(solve_events)
        t0 = time.perf_counter()
        try:
            action = ctrl.get_action(state_before, h, tvp_values=copy.deepcopy(tvp_values))
            get_exc = None
        except Exception as exc:
            action = None; get_exc = repr(exc)
        whole = float(time.perf_counter() - t0)
        new_events = solve_events[before_solves:]
        status = new_events[-1].get("return_status") if new_events else None
        if status != "Solve_Succeeded":
            solver_failure_steps += 1
            if step == 0:
                initial_solver_failures += 1
        if get_exc is not None:
            steps.append({"step": step, "state_before": state_before, "tvp_hash": canonical_hash(tvp_values), "controller_get_action_exception": get_exc, "solver_events": clean(new_events), "whole_decision_wall_s": whole})
            stopped = "controller_get_action_exception"
            break
        act = action_payload(action, ctrl)
        if step == 0:
            first_control_hash = canonical_hash(act)
        try:
            step_result = env.step(act)
            obs, reward, done, info = parse_step_result(step_result)
            gate.RESOURCE_USAGE["plant_steps"] += 1
            step_exc = None
        except Exception as exc:
            obs = None; reward = None; done = False; info = {}; step_exc = repr(exc)
        state_after = current_state(base, env)
        costs = numeric_info_costs(info, reward)
        succ = bool_from_info(info, ["success", "is_success", "reached_goal"])
        con = bool_from_info(info, ["constraint", "constraint_violation", "collision", "violated_constraint"])
        if succ is True:
            success_flag = True
        if con is True:
            constraint_flag = True
        row = {"step": step, "horizon": h, "state_before": state_before, "action": clean(act), "first_control_hash": first_control_hash if step == 0 else None, "solver_events": clean(new_events), "solver_status": status, "solver_iterations": new_events[-1].get("iterations") if new_events else None, "solver_wall_s": new_events[-1].get("solver_wall_s") if new_events else None, "whole_decision_wall_s": whole, "tvp_hash": canonical_hash(tvp_values), "state_after": state_after, "reward": reward, "done": done, "info": clean(info), "costs": costs, "env_step_exception": step_exc}
        steps.append(row)
        if step_exc is not None:
            stopped = "env_step_exception"
            break
        if done or success_flag or constraint_flag:
            stopped = "terminated_success_or_constraint_or_done"
            break
    if steps and steps[-1].get("solver_status") != "Solve_Succeeded":
        final_solver_failures += 1
    physical_values = [((s.get("costs") or {}).get("physical_or_stage_cost_best_effort")) for s in steps]
    total_values = [((s.get("costs") or {}).get("total_cost_best_effort")) for s in steps]
    def finite_sum(vals: Sequence[Any]) -> Optional[float]:
        xs = []
        for v in vals:
            try:
                f = float(v)
                if math.isfinite(f): xs.append(f)
            except Exception: pass
        return float(sum(xs)) if xs else None
    return {"horizon": h, "terminal_mode": TERMINAL_MODE, "terminal_source_horizon": int(term_h), "terminal_note": term_note, "context_hash": context_hash, "context_identity": {"context_id": context.get("context_id"), "state_label": context.get("state_label"), "tvp_start_index": context.get("tvp_start_index"), "goal": goal_meta, "state0": state0}, "initialization_meta": clean(init_meta), "steps": steps, "steps_executed": int(sum(1 for s in steps if s.get("env_step_exception") is None and "state_after" in s)), "solver_calls_observed": len(solve_events), "plant_steps_observed": int(sum(1 for s in steps if s.get("env_step_exception") is None and "state_after" in s)), "success": success_flag, "constraint": constraint_flag, "stopped_reason": stopped, "first_control_hash": first_control_hash, "physical_or_stage_cost_sum_best_effort": finite_sum(physical_values), "total_cost_sum_best_effort": finite_sum(total_values), "decision_timing_summary_s": timing_summary([s.get("whole_decision_wall_s") for s in steps]), "solver_timing_summary_s": timing_summary([s.get("solver_wall_s") for s in steps]), "solver_status_counts": {str(k): sum(1 for ev in solve_events if str(ev.get("return_status")) == str(k)) for k in sorted(set(str(ev.get("return_status")) for ev in solve_events))}, "initial_failed_steps": initial_solver_failures, "final_failed_steps": final_solver_failures, "solver_failure_steps": solver_failure_steps}


def write_failure(run_dir: Path, created: dt.datetime, error: str, used: Mapping[str, int], dependency_zero: bool = False) -> int:
    failed = run_dir / "failed.json"
    write_json(failed, {"status": "failed", "task_id": TASK_ID, "created_utc": created.isoformat(), "error": error, "traceback_tail": traceback.format_exc().splitlines()[-18:], "budget_actual": dict(used), "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False})
    try:
        if dependency_zero and all(int(v) == 0 for v in used.values()):
            execution_contract.record_outcome(ROOT, "engineering_failure", dict(used), {"no_scientific_outcome": True, "backup_verified_before_solver_or_plant_steps": False, "failed_json": rel(failed), "error": error}, engineering_error="backup_dependency")
        else:
            ev = {"source242_microcontinuation_completed": False, "prior_T_C2_and_S_TC2G_failures_preserved": TC2_DONE.exists() and STC2G_DONE.exists(), "backup_verified_before_solver_or_plant_steps": False, "all_three_horizon_arms_attempted_or_budget_cap_reported": False, "per_step_cost_safety_solver_timing_traces_persisted": False, "context_source242_goal61_identity_verified": False, "no_training_validation64_or_test_usage": True, "plant_steps_at_most_180": int(used.get("plant_steps", 999)) <= 180, "solver_calls_at_most_180": int(used.get("solver_calls", 999)) <= 180}
            execution_contract.record_outcome(ROOT, "scientific_result", dict(used), ev)
    except Exception:
        pass
    print(json.dumps(clean({"failed": error, "failed_json": rel(failed), "resources": dict(used)}), sort_keys=True), flush=True)
    return 1


def main() -> int:
    created = now_utc(); stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "research_artifacts/aws_diagnostics" / f"{NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = f"vehicle-tc2h-source242-microcontinuation-{stamp}"
    try:
        snapshot = execution_contract.runtime_snapshot(ROOT)
        if (snapshot.get("task") or {}).get("task_id") != TASK_ID:
            raise MicroError("unexpected task_id in structured snapshot")
        backup = verify_pre_resource_backup_and_priors()
        if backup.get("verified_before_solver_or_plant_steps") is not True:
            write_json(run_dir / "backup_dependency_failure.json", backup)
            return write_failure(run_dir, created, "post-S-TC2G external backup proof is not verified before solver/plant resources", dict(ZERO), dependency_zero=True)
        gate.RESOURCE_USAGE = dict(ZERO)
        gate.SOLVER_CAP = SOLVER_CAP
        gate.load_modules(); gate.install_nlpsol_patch()
        context, terminals, load_meta = gate.load_context_and_terminals()
        if context.get("context_id") != "source242_slot0_branch_start":
            raise MicroError("loaded context is not source242_slot0_branch_start")
        arms: List[Dict[str, Any]] = []
        for h in HORIZONS:
            if gate.RESOURCE_USAGE["solver_calls"] >= SOLVER_CAP or gate.RESOURCE_USAGE["plant_steps"] >= PLANT_CAP:
                break
            arm = run_arm(h, context, terminals)
            arms.append(clean(arm))
            progress = {"arms_completed": len(arms), "last_horizon": h, "last_steps": arm.get("steps_executed"), "last_stop": arm.get("stopped_reason"), "resources": dict(gate.RESOURCE_USAGE), "validation64_bank_opened": False, "sealed_test_accessed": False}
            write_json(run_dir / "progress.json", progress)
            print(json.dumps(clean(progress), sort_keys=True), flush=True)
        traces_path = run_dir / "per_step_traces.json"
        write_json(traces_path, {"arms": arms})
        summary_csv = run_dir / "arm_summary.csv"
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            fields = ["horizon", "steps_executed", "solver_calls_observed", "plant_steps_observed", "success", "constraint", "stopped_reason", "physical_or_stage_cost_sum_best_effort", "total_cost_sum_best_effort", "decision_sum_s", "solver_sum_s", "solver_status_counts", "first_control_hash"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for a in arms:
                w.writerow({"horizon": a.get("horizon"), "steps_executed": a.get("steps_executed"), "solver_calls_observed": a.get("solver_calls_observed"), "plant_steps_observed": a.get("plant_steps_observed"), "success": a.get("success"), "constraint": a.get("constraint"), "stopped_reason": a.get("stopped_reason"), "physical_or_stage_cost_sum_best_effort": a.get("physical_or_stage_cost_sum_best_effort"), "total_cost_sum_best_effort": a.get("total_cost_sum_best_effort"), "decision_sum_s": (a.get("decision_timing_summary_s") or {}).get("sum_s"), "solver_sum_s": (a.get("solver_timing_summary_s") or {}).get("sum_s"), "solver_status_counts": json.dumps(a.get("solver_status_counts"), sort_keys=True), "first_control_hash": a.get("first_control_hash")})
        evidence = {"source242_microcontinuation_completed": bool(len(arms) == 3 or gate.RESOURCE_USAGE["plant_steps"] >= PLANT_CAP or gate.RESOURCE_USAGE["solver_calls"] >= SOLVER_CAP), "prior_T_C2_and_S_TC2G_failures_preserved": True, "backup_verified_before_solver_or_plant_steps": True, "all_three_horizon_arms_attempted_or_budget_cap_reported": bool(len(arms) == 3 or gate.RESOURCE_USAGE["plant_steps"] >= PLANT_CAP or gate.RESOURCE_USAGE["solver_calls"] >= SOLVER_CAP), "per_step_cost_safety_solver_timing_traces_persisted": bool(traces_path.exists() and summary_csv.exists() and arms), "context_source242_goal61_identity_verified": all(((a.get("context_identity") or {}).get("goal") or {}).get("goal", {}).get("source") == "case.tvp.trajectory_endpoint[61]" or ((a.get("context_identity") or {}).get("goal") or {}).get("source") == "case.tvp.trajectory_endpoint[61]" for a in arms), "no_training_validation64_or_test_usage": True, "plant_steps_at_most_180": int(gate.RESOURCE_USAGE["plant_steps"]) <= PLANT_CAP, "solver_calls_at_most_180": int(gate.RESOURCE_USAGE["solver_calls"]) <= SOLVER_CAP}
        hard_pass = all(evidence.values())
        raw_path = run_dir / "raw.json"; completed_path = run_dir / "completed.json"; summary_path = run_dir / "summary.md"
        backup_request_after = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_S_TC2H_SOURCE242_MICROCONTINUATION_{stamp}.json"
        state_path = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_s_tc2h_source242_microcontinuation.md"
        raw = {"status": "complete", "task_id": TASK_ID, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "server_api_token_audit": read_api_total_tokens(), "classification": "development_IMPROVED_S_TC2H_source242_microcontinuation_not_validation_not_test", "backup_verification_before_solver_or_plant_steps": backup, "load_meta": load_meta, "arms": arms, "budget_declared": {"solver_calls": SOLVER_CAP, "plant_steps": PLANT_CAP, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}, "budget_actual": dict(gate.RESOURCE_USAGE), "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "new_aws_resources": False, "pass_evidence": evidence, "hard_pass": hard_pass, "limits": ["opened development source242 branch only", "three fixed-H arms only", "not validation64 and not sealed/final test", "temporary GPT-5.5 solo self-review"]}
        write_json(raw_path, raw)
        write_json(backup_request_after, {"request": "backup_after_S_TC2H_source242_microcontinuation", "created_utc": created.isoformat(), "must_cover": [rel(Path(__file__).resolve()), rel(run_dir), rel(backup_request_after), rel(state_path), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", rel(RESPONSE_LOG)], "budget_actual": dict(gate.RESOURCE_USAGE), "validation64_bank_opened": False, "sealed_test_accessed": False})
        lines = ["# S-TC2H source242 fixed-H microcontinuation", "", f"UTC: `{created.isoformat()}`. Development-only IMPROVED evidence; no validation64 or sealed/final test.", "", f"Resources: `{dict(gate.RESOURCE_USAGE)}`. Local hard_pass: `{hard_pass}`.", "", "| H | steps | success | constraint | stop | cost(best effort) | total(best effort) | decision sum s | solver sum s | statuses |", "|---:|---:|---|---|---|---:|---:|---:|---:|---|"]
        for a in arms:
            lines.append(f"| {a.get('horizon')} | {a.get('steps_executed')} | {a.get('success')} | {a.get('constraint')} | `{a.get('stopped_reason')}` | {a.get('physical_or_stage_cost_sum_best_effort')} | {a.get('total_cost_sum_best_effort')} | {(a.get('decision_timing_summary_s') or {}).get('sum_s')} | {(a.get('solver_timing_summary_s') or {}).get('sum_s')} | `{a.get('solver_status_counts')}` |")
        lines += ["", f"Artifacts: `{rel(raw_path)}`, `{rel(traces_path)}`, `{rel(summary_csv)}`, `{rel(completed_path)}`."]
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        block = f"<!-- {marker} -->\n## S-TC2H source242 fixed-H microcontinuation\n\nUTC: {created.isoformat()}. Development-only source242 microcontinuation completed with hard_pass `{hard_pass}` and resources `{dict(gate.RESOURCE_USAGE)}`. Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(traces_path)}`, `{rel(summary_csv)}`. Backup request: `{rel(backup_request_after)}`. Not validation64/final evidence.\n"
        for d in DOCS:
            append_if_missing(d, marker, block)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("# Continue state after S-TC2H\n\n" + block + "\nNext: inspect microcontinuation results, verify backup, then publish next bounded solo plan before further unique science because continue_without_review=false.\n", encoding="utf-8")
        with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([created.isoformat(), NAME, raw["classification"], "source242_branch_deterministic_no_training_seed", "opened_development_source242_slot0_branch_start_microcontinuation_no_validation64_no_sealed_test", gate.RESOURCE_USAGE["plant_steps"], 0, gate.RESOURCE_USAGE["solver_calls"], 0, 0, False, rel(completed_path), marker])
        completed = {"status": "complete", "task_id": TASK_ID, "created_utc": created.isoformat(), "classification": raw["classification"], "summary": rel(summary_path), "raw": rel(raw_path), "per_step_traces": rel(traces_path), "arm_summary_csv": rel(summary_csv), "backup_request": rel(backup_request_after), "state": rel(state_path), "budget_actual": dict(gate.RESOURCE_USAGE), "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "headline": {"hard_pass": hard_pass, "arms": [{"H": a.get("horizon"), "steps": a.get("steps_executed"), "success": a.get("success"), "constraint": a.get("constraint"), "stop": a.get("stopped_reason"), "cost": a.get("physical_or_stage_cost_sum_best_effort"), "total": a.get("total_cost_sum_best_effort"), "decision_sum_s": (a.get("decision_timing_summary_s") or {}).get("sum_s"), "solver_sum_s": (a.get("solver_timing_summary_s") or {}).get("sum_s"), "statuses": a.get("solver_status_counts")} for a in arms]}, "pass_evidence": evidence}
        hash_paths = [Path(__file__).resolve(), GATE_SOURCE, raw_path, summary_path, traces_path, summary_csv, completed_path, backup_request_after, state_path] + DOCS + [ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists() and p != completed_path}
        write_json(completed_path, completed)
        execution_contract.record_outcome(ROOT, "scientific_result", dict(gate.RESOURCE_USAGE), evidence)
        print(json.dumps(clean({"completed": rel(completed_path), "headline": completed["headline"], "pass_evidence": evidence}), sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        used = dict(getattr(gate, "RESOURCE_USAGE", ZERO))
        return write_failure(run_dir, created, "unexpected S-TC2H error: %s: %s" % (type(exc).__name__, exc), used, dependency_zero=all(int(v) == 0 for v in used.values()))


if __name__ == "__main__":
    raise SystemExit(main())
