#!/usr/bin/env python3
"""S-TC2H7 high-level environment-step repair for source242 microcontinuation.

S-TC2H6/v0g proved that the H12/H15/H35 first MPC optimizations solve under the
source242/V15_shared context, but it still passed the low-level control returned
by ``controller.get_action`` into ``LetMPCEnv.step``.  The retained gym-horizon
environment expects the high-level agent action ``mpc_horizon`` and internally
calls ``ControlSystem.step`` -> ``controller.get_action`` -> ``simulator.make_step``.

This wrapper changes only that interface layer: it no longer pre-solves before
``env.step`` and instead calls ``env.step([H])`` while instrumenting the internal
MPC solve.  The source242 state/TVP/goal, H=[12,15,35], V15_shared terminal
weights, canonical initial guess, budgets, and access restrictions are inherited.
No validation64, sealed/final test, training, or refit is used.
"""
from __future__ import annotations

import copy
import datetime as dt
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0g_backup_proof_materialization_rerun as v0g

base = v0g.base
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
NAME = "vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0h_highlevel_env_step_repair"
TASK_ID = "S-TC2H7-source242-high-level-env-step-repair-v0"

V0G_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0g_backup_proof_materialization_rerun_20260930T175410Z/raw.json"
V0G_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0g_backup_proof_materialization_rerun_20260930T175410Z/completed.json"
V0G_RECEIPT = ROOT / "research_artifacts/aws_runs/20260930T175410_221187e7/outcome_receipt.json"
MATERIALIZED_BACKUP_PROOF_181255 = ROOT / "research_artifacts/aws_backup_proofs/BACKUP_VERIFIED_FROM_SUPERVISOR_CONTEXT_20260930T181255Z.json"

_previous_record_outcome = base.execution_contract.record_outcome


def _mtime_utc(path: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)


def _latest_time(values: Sequence[dt.datetime]) -> dt.datetime:
    if not values:
        raise base.MicroError("no timestamps supplied to v0h backup recency gate")
    out = values[0]
    for value in values[1:]:
        if value > out:
            out = value
    return out


def _require_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise base.MicroError("required prior artifact missing for v0h gate: %s -> %s" % (label, base.rel(path)))
    obj = base.read_json(path)
    if not isinstance(obj, dict):
        raise base.MicroError("required prior artifact is not a JSON object: %s" % base.rel(path))
    return obj


def _ready_paths_from_snapshot() -> List[Path]:
    return list(v0g._ready_paths_from_snapshot())


def _commit_gate(candidates: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
    return v0g._commit_gate(candidates)


def _validate_v0g_outcome() -> Dict[str, Any]:
    raw = _require_json(V0G_RAW, "S-TC2H6/v0g raw outcome")
    completed = _require_json(V0G_COMPLETED, "S-TC2H6/v0g completed marker")
    receipt = _require_json(V0G_RECEIPT, "S-TC2H6/v0g execution receipt")
    budget = raw.get("budget_actual") or {}
    if int(budget.get("solver_calls", -1)) != 3 or int(budget.get("plant_steps", -1)) != 0:
        raise base.MicroError("v0g prior is not the expected three-solver/zero-plant interface failure")
    arms = raw.get("arms") or []
    if len(arms) != 3:
        raise base.MicroError("v0g prior did not preserve all three attempted horizon arms")
    stops = [str(a.get("stopped_reason")) for a in arms if isinstance(a, Mapping)]
    if stops != ["env_step_exception", "env_step_exception", "env_step_exception"]:
        raise base.MicroError("v0g prior stop reasons changed from expected env_step_exception triplet")
    evidence = receipt.get("evidence") or {}
    if evidence.get("source242_microcontinuation_produced_plant_transitions") is not False:
        raise base.MicroError("v0g receipt no longer records the expected absence of plant transitions")
    return {
        "raw": base.rel(V0G_RAW),
        "completed": base.rel(V0G_COMPLETED),
        "receipt": base.rel(V0G_RECEIPT),
        "budget_actual": budget,
        "stopped_reasons": stops,
        "receipt_evidence_subset": {
            "source242_microcontinuation_produced_plant_transitions": evidence.get("source242_microcontinuation_produced_plant_transitions"),
            "all_three_horizon_arms_advanced_at_least_one_plant_step": evidence.get("all_three_horizon_arms_advanced_at_least_one_plant_step"),
            "no_env_or_controller_exceptions_in_attempted_arms": evidence.get("no_env_or_controller_exceptions_in_attempted_arms"),
        },
    }


def _validate_latest_backup_proof() -> Dict[str, Any]:
    proof = _require_json(MATERIALIZED_BACKUP_PROOF_181255, "materialized 18:12:55 supervisor-context backup proof")
    if proof.get("status") != "verified" or int(proof.get("remaining_changed_files", -1)) != 0:
        raise base.MicroError("18:12:55 materialized backup proof is not verified/clean")
    if proof.get("commit") != "a3a446b7432af1b7bdb1e4ab84b1463e2d232c62":
        raise base.MicroError("18:12:55 materialized backup proof commit differs from supplied context")
    if proof.get("package_sha256") != "a46a4a03655dee0149ee3c69dfe5ed2711527f7d965e20f47687fb993442717a":
        raise base.MicroError("18:12:55 materialized backup proof package SHA differs from supplied context")
    return {
        "path": base.rel(MATERIALIZED_BACKUP_PROOF_181255),
        "json_time": proof.get("time"),
        "mtime_utc": _mtime_utc(MATERIALIZED_BACKUP_PROOF_181255).isoformat(),
        "commit": proof.get("commit"),
        "package_sha256": proof.get("package_sha256"),
    }


def verify_pre_resource_backup_and_priors_v0h() -> Dict[str, Any]:
    prior = v0g.verify_pre_resource_backup_and_priors_v0g()
    v0g_info = _validate_v0g_outcome()
    backup181255 = _validate_latest_backup_proof()
    required_paths = [
        V0G_RAW,
        V0G_COMPLETED,
        V0G_RECEIPT,
        MATERIALIZED_BACKUP_PROOF_181255,
        SOURCE,
    ] + _ready_paths_from_snapshot()
    for path in required_paths:
        if not path.exists():
            raise base.MicroError("required v0h path missing before backup gate: " + base.rel(path))
    prior_min = base.parse_time(prior.get("required_min_time_v0g") or prior.get("required_min_time")) if isinstance(prior, Mapping) else None
    times: List[dt.datetime] = [prior_min] if prior_min is not None else []
    times.extend(_mtime_utc(p) for p in required_paths)
    min_time = _latest_time(times)
    candidates = base.verified_backup_candidates(min_time)
    commit_ok, head, enriched = _commit_gate(candidates)
    status = base.run_git(["status", "--porcelain", "--"] + [base.rel(p) for p in required_paths])
    clean_ok = bool(status.get("ok") and status.get("stdout") == "")
    ok = bool(prior.get("verified_before_solver_or_plant_steps") and enriched and clean_ok and commit_ok)
    out = dict(prior)
    out.update({
        "verified_before_solver_or_plant_steps": ok,
        "backup_verified_after_v0g_failure_backup181255_v0h_source_and_plan_before_solver_or_plant_steps": ok,
        "required_min_time_v0h": min_time.isoformat(),
        "backup_candidates_v0h": enriched,
        "current_git_head_v0h": head,
        "required_paths_git_status_porcelain_v0h": status,
        "required_path_hashes_v0h": {base.rel(p): base.sha256(p) for p in required_paths if p.exists()},
        "required_paths_checked_v0h": [base.rel(p) for p in required_paths],
        "S_TC2H6_three_solver_zero_plant_interface_failure_preserved": True,
        "materialized_181255_backup_proof_preserved": True,
        "v0g_outcome": v0g_info,
        "materialized_backup_proof_181255": backup181255,
        "v0h_source": {"path": base.rel(SOURCE), "sha256": base.sha256(SOURCE), "mtime_utc": _mtime_utc(SOURCE).isoformat()},
    })
    return out


def _numeric_state_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    try:
        dtheta = math.atan2(math.sin(float(a.get("theta", 0.0)) - float(b.get("theta", 0.0))), math.cos(float(a.get("theta", 0.0)) - float(b.get("theta", 0.0))))
        return float(abs(dtheta) + abs(float(a.get("x", 0.0)) - float(b.get("x", 0.0))) + abs(float(a.get("y", 0.0)) - float(b.get("y", 0.0))))
    except Exception:
        return float("nan")


def _initialize_env_histories(env: Any) -> Dict[str, Any]:
    cs = env.control_system
    try:
        cs.simulator.reset_history()
    except Exception:
        pass
    try:
        cs.simulator.x0 = cs.get_state_vector(cs.current_state)
    except Exception:
        pass
    try:
        cs.simulator._x0.master = cs.get_state_vector(cs.current_state)
    except Exception:
        pass
    cs._preset_process_noise = None
    cs._process_noise_props = {s_name: {} for s_name in getattr(cs, "state_names", [])}
    cs._step_count = 0
    try:
        process0 = np.zeros_like(cs._get_process_noise())
    except Exception:
        process0 = []
    cs.history = {"state": [copy.deepcopy(cs.current_state)], "process_noise": [process0], "tvp": []}
    env.steps_count = 0
    try:
        obs0 = env.get_observation()
        obs0_clean = base.clean(obs0)
    except Exception as exc:
        obs0_clean = {"observation_exception": repr(exc)}
    env.history = {"obs": [obs0_clean], "actions": [], "rewards": []}
    return {
        "control_system_history_initialized": True,
        "env_history_initialized": True,
        "initial_step_count": int(getattr(cs, "_step_count", 0) or 0),
        "initial_env_steps_count": int(getattr(env, "steps_count", 0) or 0),
        "initial_observation": obs0_clean,
    }


def _enhanced_costs(info: Mapping[str, Any], reward: Optional[float]) -> Dict[str, Any]:
    costs = base.numeric_info_costs(info, reward)
    physical_components: List[float] = []
    for key in ("reward/performance", "reward/constraint"):
        if key in info:
            try:
                val = float(np.asarray(info[key]).reshape(-1)[0])
                if math.isfinite(val):
                    physical_components.append(val)
            except Exception:
                pass
    if physical_components:
        costs["physical_control_constraint_cost_from_reward_components_best_effort"] = float(math.fsum(physical_components))
        if costs.get("physical_or_stage_cost_best_effort") is None:
            costs["physical_or_stage_cost_best_effort"] = float(math.fsum(physical_components))
    if "reward/computation" in info:
        try:
            costs["reported_compute_cost_component"] = float(np.asarray(info["reward/computation"]).reshape(-1)[0])
        except Exception:
            pass
    if "mpc_horizon" in info:
        try:
            costs["executed_horizon_from_info"] = int(np.asarray(info["mpc_horizon"]).reshape(-1)[0])
        except Exception:
            pass
    return costs


def run_arm_highlevel(h: int, context: Mapping[str, Any], terminals: Mapping[int, Any]) -> Dict[str, Any]:
    gate = base.gate
    model_base = gate.MODULES["base"]
    stage1 = gate.MODULES["stage1_runner"]
    v34u = gate.MODULES["v34u"]
    terminal, term_h, term_note = base.terminal_for_mode(h, base.TERMINAL_MODE, terminals)
    env = base.create_env_with_terminal(model_base, stage1, h, terminal)
    ctrl = env.control_system.controller
    mpc = ctrl.mpc
    gate.CURRENT_ARM_ID = f"ctx=source242_slot0_branch_start|H{h}|term=V15_shared|microcontinuation|highlevel_env_step"
    gate.CURRENT_SOLVE_MODE = "production"
    v34u.install_tta_hmpc_presolve_obj_guard()
    context_meta, goal_meta = base.configure_context_source242_goal61(model_base, env, context, h)
    scalar = gate.scalarize_tvp(context_meta["shifted_tvp"])
    context_meta["shifted_tvp"] = scalar
    for name, values in scalar.items():
        if hasattr(env.control_system, "tvps") and name in env.control_system.tvps:
            env.control_system.tvps[name].values = copy.deepcopy(values)
    ctrl.goal_x = float(context_meta["goal_x"])
    ctrl.goal_y = float(context_meta["goal_y"])
    ctrl._tvp_data = copy.deepcopy(scalar)
    init_meta = gate.set_initial_guess_strict(mpc, env, context_meta, "canonical", h)
    history_meta = _initialize_env_histories(env)
    state0 = base.current_state(model_base, env)
    context_hash = base.canonical_hash({"context_id": context.get("context_id"), "state0": state0, "tvp_start_index": context.get("tvp_start_index"), "horizon": h, "goal": goal_meta, "interface": "env.step([H])"})

    solve_events: List[Dict[str, Any]] = []
    original_solve = mpc.solve

    def counted_solve(*args: Any, **kwargs: Any) -> Any:
        if gate.RESOURCE_USAGE["solver_calls"] >= base.SOLVER_CAP:
            raise base.MicroError("solver cap exhausted in S-TC2H7")
        pre = gate.capture_mpc_state(mpc)
        gate.RESOURCE_USAGE["solver_calls"] += 1
        t0 = time.perf_counter()
        exc = None
        try:
            ret = original_solve(*args, **kwargs)
        except Exception as e:
            ret = None
            exc = repr(e)
        elapsed = float(time.perf_counter() - t0)
        stats = copy.deepcopy(getattr(mpc, "solver_stats", {}))
        post = gate.capture_mpc_state(mpc)
        ev = {
            "step_index": len(solve_events),
            "pre": pre,
            "post": post,
            "solver_wall_s": elapsed,
            "solver_exception": exc,
            "solver_stats": stats,
            "return_status": stats.get("return_status"),
            "success": bool(stats.get("success", False)),
            "iterations": stats.get("iter_count") or stats.get("iterations"),
            "objective_opt_f_num": gate.finite_float(getattr(mpc, "opt_f_num", None), None),
        }
        solve_events.append(ev)
        if exc is not None:
            raise base.MicroError("mpc.solve failed: " + exc)
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
    plant_steps_in_arm = 0
    for step in range(base.MAX_STEPS_PER_ARM):
        if gate.RESOURCE_USAGE["plant_steps"] >= base.PLANT_CAP:
            stopped = "global_plant_cap_reached"
            break
        state_before = base.current_state(model_base, env)
        before_solves = len(solve_events)
        cs_step_before = int(getattr(env.control_system, "_step_count", 0) or 0)
        high_level_action = np.asarray([float(h)], dtype=float)
        t0 = time.perf_counter()
        try:
            step_result = env.step(high_level_action)
            obs, reward, done, info = base.parse_step_result(step_result)
            step_exc = None
        except Exception as exc:
            obs = None
            reward = None
            done = False
            info = {}
            step_exc = repr(exc)
        whole = float(time.perf_counter() - t0)
        cs_step_after = int(getattr(env.control_system, "_step_count", cs_step_before) or cs_step_before)
        delta_steps = max(0, cs_step_after - cs_step_before)
        if delta_steps > 0:
            gate.RESOURCE_USAGE["plant_steps"] += int(delta_steps)
            plant_steps_in_arm += int(delta_steps)
        new_events = solve_events[before_solves:]
        status = new_events[-1].get("return_status") if new_events else None
        if status != "Solve_Succeeded":
            solver_failure_steps += 1
            if step == 0:
                initial_solver_failures += 1
        act = copy.deepcopy(getattr(ctrl, "current_input", None))
        if step == 0:
            first_control_hash = base.canonical_hash(act)
        state_after = base.current_state(model_base, env)
        costs = _enhanced_costs(info, reward)
        succ = base.bool_from_info(info, ["success", "is_success", "reached_goal"])
        if succ is None and info.get("termination") == "goal":
            succ = True
        con = base.bool_from_info(info, ["constraint", "constraint_violation", "collision", "violated_constraint"])
        if con is None and info.get("termination") == "constraint":
            con = True
        if succ is True:
            success_flag = True
        if con is True:
            constraint_flag = True
        row = {
            "step": step,
            "horizon": h,
            "interface": "LetMPCEnv.step(high_level_mpc_horizon_action)",
            "high_level_action": base.clean(high_level_action),
            "state_before": state_before,
            "state_after": state_after,
            "state_delta_l1_best_effort": _numeric_state_distance(state_before, state_after),
            "action": base.clean(act),
            "first_control_hash": first_control_hash if step == 0 else None,
            "solver_events": base.clean(new_events),
            "solver_status": status,
            "solver_iterations": new_events[-1].get("iterations") if new_events else None,
            "solver_wall_s": new_events[-1].get("solver_wall_s") if new_events else None,
            "whole_decision_wall_s": whole,
            "control_system_step_count_before": cs_step_before,
            "control_system_step_count_after": cs_step_after,
            "plant_step_delta_counted": delta_steps,
            "reward": reward,
            "done": done,
            "info": base.clean(info),
            "costs": costs,
            "env_step_exception": step_exc,
        }
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
        xs: List[float] = []
        for v in vals:
            try:
                f = float(v)
                if math.isfinite(f):
                    xs.append(f)
            except Exception:
                pass
        return float(math.fsum(xs)) if xs else None
    return {
        "horizon": h,
        "terminal_mode": base.TERMINAL_MODE,
        "terminal_source_horizon": int(term_h),
        "terminal_note": term_note,
        "context_hash": context_hash,
        "context_identity": {"context_id": context.get("context_id"), "state_label": context.get("state_label"), "tvp_start_index": context.get("tvp_start_index"), "goal": goal_meta, "state0": state0},
        "initialization_meta": base.clean(init_meta),
        "history_initialization_meta": base.clean(history_meta),
        "steps": steps,
        "steps_executed": int(plant_steps_in_arm),
        "solver_calls_observed": len(solve_events),
        "plant_steps_observed": int(plant_steps_in_arm),
        "success": success_flag,
        "constraint": constraint_flag,
        "stopped_reason": stopped,
        "first_control_hash": first_control_hash,
        "physical_or_stage_cost_sum_best_effort": finite_sum(physical_values),
        "total_cost_sum_best_effort": finite_sum(total_values),
        "decision_timing_summary_s": base.timing_summary([s.get("whole_decision_wall_s") for s in steps]),
        "solver_timing_summary_s": base.timing_summary([s.get("solver_wall_s") for s in steps]),
        "solver_status_counts": {str(k): sum(1 for ev in solve_events if str(ev.get("return_status")) == str(k)) for k in sorted(set(str(ev.get("return_status")) for ev in solve_events))},
        "initial_failed_steps": initial_solver_failures,
        "final_failed_steps": final_solver_failures,
        "solver_failure_steps": solver_failure_steps,
    }


def _strict_microcontinuation_evidence_current() -> Dict[str, Any]:
    diag_root = ROOT / "research_artifacts/aws_diagnostics"
    candidates = []
    for path in diag_root.glob(NAME + "_*/raw.json"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            pass
    if not candidates:
        return {
            "source242_microcontinuation_produced_plant_transitions": False,
            "all_three_horizon_arms_advanced_at_least_one_plant_step": False,
            "no_env_or_controller_exceptions_in_attempted_arms": False,
            "strict_microcontinuation_evidence_source": None,
        }
    candidates.sort()
    raw_path = candidates[-1][1]
    try:
        raw = base.read_json(raw_path)
    except Exception as exc:
        return {
            "source242_microcontinuation_produced_plant_transitions": False,
            "all_three_horizon_arms_advanced_at_least_one_plant_step": False,
            "no_env_or_controller_exceptions_in_attempted_arms": False,
            "strict_microcontinuation_evidence_source": base.rel(raw_path),
            "strict_microcontinuation_evidence_error": "%s: %s" % (type(exc).__name__, exc),
        }
    arms = raw.get("arms") or []
    if not isinstance(arms, list):
        arms = []
    plant_total = int((raw.get("budget_actual") or {}).get("plant_steps", 0) or 0)
    stop_reasons = [str(a.get("stopped_reason")) for a in arms if isinstance(a, Mapping)]
    plant_by_horizon: Dict[str, int] = {}
    for a in arms:
        if isinstance(a, Mapping):
            plant_by_horizon[str(a.get("horizon"))] = int(a.get("plant_steps_observed", 0) or 0)
    exception_free = bool(arms) and all(reason not in ("env_step_exception", "controller_get_action_exception") for reason in stop_reasons)
    all_three_advanced = bool(len(arms) == 3 and all(int(a.get("plant_steps_observed", 0) or 0) > 0 for a in arms if isinstance(a, Mapping)))
    if len(arms) != 3:
        all_three_advanced = False
    return {
        "source242_microcontinuation_produced_plant_transitions": plant_total > 0,
        "all_three_horizon_arms_advanced_at_least_one_plant_step": all_three_advanced,
        "no_env_or_controller_exceptions_in_attempted_arms": exception_free,
        "strict_microcontinuation_evidence_source": base.rel(raw_path),
        "observed_total_plant_steps": plant_total,
        "observed_arm_stop_reasons": stop_reasons,
        "observed_plant_steps_by_horizon": plant_by_horizon,
    }


def _augment_evidence_v0h(evidence: Any) -> Any:
    if not isinstance(evidence, Mapping):
        return evidence
    out = dict(evidence)
    backup_ok = bool(out.get("backup_verified_before_solver_or_plant_steps"))
    out.setdefault("backup_verified_after_v0g_failure_backup181255_v0h_source_and_plan_before_solver_or_plant_steps", backup_ok)
    out.setdefault("S_TC2H6_three_solver_zero_plant_interface_failure_preserved", V0G_RAW.exists() and V0G_COMPLETED.exists() and V0G_RECEIPT.exists())
    out.setdefault("materialized_181255_backup_proof_preserved", MATERIALIZED_BACKUP_PROOF_181255.exists())
    out.setdefault("high_level_env_step_interface_repair_applied", True)
    out.update(_strict_microcontinuation_evidence_current())
    return out


def record_outcome_v0h(root: Any, outcome: str, used: Mapping[str, int], evidence: Any, engineering_error: Optional[str] = None) -> Any:
    if engineering_error == "backup_dependency":
        engineering_error = "dependency"
    return _previous_record_outcome(root, outcome, used, _augment_evidence_v0h(evidence), engineering_error=engineering_error)


# Apply launch/prerequisite/evidence and the only scientific-interface repair.
base.NAME = NAME
base.TASK_ID = TASK_ID
base.__dict__["__file__"] = str(SOURCE)
if hasattr(v0g, "NAME"):
    v0g.NAME = NAME
if hasattr(v0g, "v0f"):
    v0g.v0f.NAME = NAME
base.verify_pre_resource_backup_and_priors = verify_pre_resource_backup_and_priors_v0h
base.run_arm = run_arm_highlevel
base.execution_contract.record_outcome = record_outcome_v0h

if __name__ == "__main__":
    raise SystemExit(base.main())
