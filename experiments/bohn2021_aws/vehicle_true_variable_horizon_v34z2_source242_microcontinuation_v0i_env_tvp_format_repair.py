#!/usr/bin/env python3
"""S-TC2H8 raw-TVP environment format repair for source242 microcontinuation.

The preceding S-TC2H7 high-level ``LetMPCEnv.step([H])`` repair still stopped
before solver/plant resources with ``TypeError("'float' object is not
subscriptable")``.  The new evidence localizes the failure earlier than the
controller solve: the initialized environment could not even compute its
observation, and every arm reported zero solver calls.  Source inspection shows
that ``LetMPCEnv``/``ControlSystem`` expect ``TVP.values`` to remain in the
legacy list-of-dicts format (``{"true": [...], "forecast": [...]}``) so that
``TVP.get_values`` can index ``["true"]``.  S-TC2H7 accidentally wrote the
scalarized controller forecast arrays back into ``env.control_system.tvps``.

This wrapper keeps the S-TC2H7 high-level action repair but restores the raw TVP
storage for the environment before ``env.get_observation`` and ``env.step``.  The
scalarized TVP copy is still used only for strict MPC initial-guess metadata.
No horizons, terminal value, state/goal context, costs, training, validation64,
sealed/final-test access, or acceptance thresholds are changed.
"""
from __future__ import annotations

import copy
import datetime as dt
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0h_highlevel_env_step_repair as v0h

base = v0h.base
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
NAME = "vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0i_env_tvp_format_repair"
TASK_ID = "S-TC2H8-source242-env-tvp-format-repair-v0"

V0H_RECEIPT = ROOT / "research_artifacts/aws_runs/20260930T190047_54183e0e/outcome_receipt.json"
MATERIALIZED_BACKUP_PROOF_185232 = ROOT / "research_artifacts/aws_backup_proofs/BACKUP_VERIFIED_FROM_SUPERVISOR_CONTEXT_20260930T185232Z.json"
MATERIALIZED_BACKUP_PROOF_185834 = ROOT / "research_artifacts/aws_backup_proofs/BACKUP_VERIFIED_FROM_SUPERVISOR_CONTEXT_20260930T185834Z.json"

_previous_record_outcome = base.execution_contract.record_outcome


def _mtime_utc(path: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)


def _latest_time(values: Sequence[dt.datetime]) -> dt.datetime:
    if not values:
        raise base.MicroError("no timestamps supplied to v0i backup recency gate")
    out = values[0]
    for value in values[1:]:
        if value > out:
            out = value
    return out


def _require_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise base.MicroError("required prior artifact missing for v0i gate: %s -> %s" % (label, base.rel(path)))
    obj = base.read_json(path)
    if not isinstance(obj, dict):
        raise base.MicroError("required prior artifact is not a JSON object: %s" % base.rel(path))
    return obj


def _latest_v0h_raw() -> Path:
    diag_root = ROOT / "research_artifacts/aws_diagnostics"
    candidates: List[Tuple[float, Path]] = []
    for path in diag_root.glob("vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0h_highlevel_env_step_repair_*/raw.json"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            pass
    if not candidates:
        raise base.MicroError("missing prior v0h raw output for v0i gate")
    candidates.sort()
    return candidates[-1][1]


def _validate_v0h_zero_solver_tvp_format_failure() -> Dict[str, Any]:
    raw_path = _latest_v0h_raw()
    raw = _require_json(raw_path, "S-TC2H7/v0h raw outcome")
    receipt = _require_json(V0H_RECEIPT, "S-TC2H7/v0h execution receipt")
    budget = raw.get("budget_actual") or {}
    if int(budget.get("solver_calls", -1)) != 0 or int(budget.get("plant_steps", -1)) != 0:
        raise base.MicroError("v0h prior is not the expected zero-solver/zero-plant env TVP-format failure")
    arms = raw.get("arms") or []
    if len(arms) != 3:
        raise base.MicroError("v0h prior did not preserve all three attempted horizon arms")
    stops = [str(a.get("stopped_reason")) for a in arms if isinstance(a, Mapping)]
    if stops != ["env_step_exception", "env_step_exception", "env_step_exception"]:
        raise base.MicroError("v0h prior stop reasons changed from expected env_step_exception triplet")
    exceptions: List[str] = []
    initial_obs_exceptions: List[str] = []
    for arm in arms:
        if isinstance(arm, Mapping):
            steps = arm.get("steps") or []
            if steps and isinstance(steps[0], Mapping):
                exceptions.append(str(steps[0].get("env_step_exception")))
            hist = arm.get("history_initialization_meta") or {}
            init_obs = hist.get("initial_observation") if isinstance(hist, Mapping) else {}
            if isinstance(init_obs, Mapping):
                initial_obs_exceptions.append(str(init_obs.get("observation_exception")))
    if not exceptions or not all("float" in e and "subscriptable" in e for e in exceptions):
        raise base.MicroError("v0h prior does not show the expected float-subscript env_step failure")
    receipt_evidence = receipt.get("evidence") or {}
    if receipt_evidence.get("source242_microcontinuation_produced_plant_transitions") is not False:
        raise base.MicroError("v0h receipt no longer records zero plant transitions")
    return {
        "raw": base.rel(raw_path),
        "receipt": base.rel(V0H_RECEIPT),
        "budget_actual": budget,
        "stopped_reasons": stops,
        "env_step_exceptions": exceptions,
        "initial_observation_exceptions": initial_obs_exceptions,
        "receipt_evidence_subset": {
            "source242_microcontinuation_produced_plant_transitions": receipt_evidence.get("source242_microcontinuation_produced_plant_transitions"),
            "all_three_horizon_arms_advanced_at_least_one_plant_step": receipt_evidence.get("all_three_horizon_arms_advanced_at_least_one_plant_step"),
            "no_env_or_controller_exceptions_in_attempted_arms": receipt_evidence.get("no_env_or_controller_exceptions_in_attempted_arms"),
        },
    }


def _validate_materialized_185232() -> Dict[str, Any]:
    proof = _require_json(MATERIALIZED_BACKUP_PROOF_185232, "materialized 18:52:32 supervisor-context backup proof")
    if proof.get("status") != "verified" or int(proof.get("remaining_changed_files", -1)) != 0:
        raise base.MicroError("18:52:32 materialized backup proof is not verified/clean")
    if proof.get("commit") != "a17cd03e4377ede657bbae1bccf47a9199b4ad63":
        raise base.MicroError("18:52:32 materialized backup proof commit differs from supplied context")
    if proof.get("package_sha256") != "b3fdc210a91a4d5fade2c25f94b1ef2ec989bda273f3b24b895b42a52587b4ee":
        raise base.MicroError("18:52:32 materialized backup proof package SHA differs from supplied context")
    return {"path": base.rel(MATERIALIZED_BACKUP_PROOF_185232), "json_time": proof.get("time"), "mtime_utc": _mtime_utc(MATERIALIZED_BACKUP_PROOF_185232).isoformat(), "commit": proof.get("commit"), "package_sha256": proof.get("package_sha256")}


def _validate_materialized_185834() -> Dict[str, Any]:
    proof = _require_json(MATERIALIZED_BACKUP_PROOF_185834, "materialized 18:58:34 supervisor-context backup proof")
    if proof.get("status") != "verified" or int(proof.get("remaining_changed_files", -1)) != 0:
        raise base.MicroError("18:58:34 materialized backup proof is not verified/clean")
    if proof.get("commit") != "5f02a31503c268cb1a2a0c162fb8fc25f04f09ec":
        raise base.MicroError("18:58:34 materialized backup proof commit differs from supplied context")
    if proof.get("package_sha256") != "7258d6e49549fc9cc7e2f564f2e88841b90ac576f3ae4274a1605fa6bf1a2038":
        raise base.MicroError("18:58:34 materialized backup proof package SHA differs from supplied context")
    return {"path": base.rel(MATERIALIZED_BACKUP_PROOF_185834), "json_time": proof.get("time"), "mtime_utc": _mtime_utc(MATERIALIZED_BACKUP_PROOF_185834).isoformat(), "commit": proof.get("commit"), "package_sha256": proof.get("package_sha256")}


def verify_pre_resource_backup_and_priors_v0i() -> Dict[str, Any]:
    prior = v0h.verify_pre_resource_backup_and_priors_v0h()
    v0h_info = _validate_v0h_zero_solver_tvp_format_failure()
    backup185232 = _validate_materialized_185232()
    backup185834 = _validate_materialized_185834()
    v0h_raw = Path(ROOT / v0h_info["raw"])
    required_paths = [
        SOURCE,
        v0h_raw,
        V0H_RECEIPT,
        MATERIALIZED_BACKUP_PROOF_185232,
        MATERIALIZED_BACKUP_PROOF_185834,
    ] + list(v0h._ready_paths_from_snapshot())
    for path in required_paths:
        if not path.exists():
            raise base.MicroError("required v0i path missing before backup gate: " + base.rel(path))
    prior_min = base.parse_time(prior.get("required_min_time_v0h") or prior.get("required_min_time")) if isinstance(prior, Mapping) else None
    times: List[dt.datetime] = [prior_min] if prior_min is not None else []
    times.extend(_mtime_utc(p) for p in required_paths)
    min_time = _latest_time(times)
    candidates = base.verified_backup_candidates(min_time)
    commit_ok, head, enriched = v0h._commit_gate(candidates)
    status = base.run_git(["status", "--porcelain", "--"] + [base.rel(p) for p in required_paths])
    clean_ok = bool(status.get("ok") and status.get("stdout") == "")
    ok = bool(prior.get("verified_before_solver_or_plant_steps") and enriched and clean_ok and commit_ok)
    out = dict(prior)
    out.update({
        "verified_before_solver_or_plant_steps": ok,
        "backup_verified_after_v0h_failure_v0i_source_and_plan_before_solver_or_plant_steps": ok,
        "required_min_time_v0i": min_time.isoformat(),
        "backup_candidates_v0i": enriched,
        "current_git_head_v0i": head,
        "required_paths_git_status_porcelain_v0i": status,
        "required_path_hashes_v0i": {base.rel(p): base.sha256(p) for p in required_paths if p.exists()},
        "required_paths_checked_v0i": [base.rel(p) for p in required_paths],
        "S_TC2H7_zero_solver_zero_plant_env_tvp_format_failure_preserved": True,
        "materialized_185232_backup_proof_preserved": True,
        "materialized_185834_backup_proof_preserved": True,
        "v0h_outcome": v0h_info,
        "materialized_backup_proof_185232": backup185232,
        "materialized_backup_proof_185834": backup185834,
        "v0i_source": {"path": base.rel(SOURCE), "sha256": base.sha256(SOURCE), "mtime_utc": _mtime_utc(SOURCE).isoformat()},
    })
    return out


def _restore_raw_env_tvps(env: Any, raw_shifted_tvp: Mapping[str, Any]) -> Dict[str, Any]:
    restored: Dict[str, Any] = {}
    if hasattr(env.control_system, "tvps"):
        for name, tvp_obj in env.control_system.tvps.items():
            if name in raw_shifted_tvp:
                values = copy.deepcopy(raw_shifted_tvp[name])
                tvp_obj.values = values
                first = values[0] if isinstance(values, list) and values else None
                restored[name] = {
                    "len": len(values) if hasattr(values, "__len__") else None,
                    "first_type": type(first).__name__,
                    "first_has_true_forecast_keys": bool(isinstance(first, Mapping) and "true" in first and "forecast" in first),
                }
    return restored


def run_arm_raw_tvp_highlevel(h: int, context: Mapping[str, Any], terminals: Mapping[int, Any]) -> Dict[str, Any]:
    gate = base.gate
    model_base = gate.MODULES["base"]
    stage1 = gate.MODULES["stage1_runner"]
    v34u = gate.MODULES["v34u"]
    terminal, term_h, term_note = model_base.terminal_for_mode(h, base.TERMINAL_MODE, terminals)
    env = base.create_env_with_terminal(model_base, stage1, h, terminal)
    ctrl = env.control_system.controller
    mpc = ctrl.mpc
    gate.CURRENT_ARM_ID = f"ctx=source242_slot0_branch_start|H{h}|term=V15_shared|microcontinuation|highlevel_env_step|raw_env_tvp"
    gate.CURRENT_SOLVE_MODE = "production"
    v34u.install_tta_hmpc_presolve_obj_guard()
    context_meta, goal_meta = base.configure_context_source242_goal61(model_base, env, context, h)
    raw_shifted_tvp = copy.deepcopy(context_meta["shifted_tvp"])
    scalar_tvp = gate.scalarize_tvp(raw_shifted_tvp)
    context_meta["shifted_tvp"] = scalar_tvp
    ctrl.goal_x = float(context_meta["goal_x"])
    ctrl.goal_y = float(context_meta["goal_y"])
    ctrl._tvp_data = copy.deepcopy(scalar_tvp)
    init_meta = gate.set_initial_guess_strict(mpc, env, context_meta, "canonical", h)
    raw_tvp_restore_meta = _restore_raw_env_tvps(env, raw_shifted_tvp)
    history_meta = v0h._initialize_env_histories(env)
    state0 = base.current_state(model_base, env)
    context_hash = base.canonical_hash({
        "context_id": context.get("context_id"),
        "state0": state0,
        "tvp_start_index": context.get("tvp_start_index"),
        "horizon": h,
        "goal": goal_meta,
        "interface": "env.step(np.asarray([H])) with raw env TVP storage",
    })

    solve_events: List[Dict[str, Any]] = []
    original_solve = mpc.solve

    def counted_solve(*args: Any, **kwargs: Any) -> Any:
        if gate.RESOURCE_USAGE["solver_calls"] >= base.SOLVER_CAP:
            raise base.MicroError("solver cap exhausted in S-TC2H8")
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
        ev = {"step_index": len(solve_events), "pre": pre, "post": post, "solver_wall_s": elapsed, "solver_exception": exc, "solver_stats": stats, "return_status": stats.get("return_status"), "success": bool(stats.get("success", False)), "iterations": stats.get("iter_count") or stats.get("iterations"), "objective_opt_f_num": gate.finite_float(getattr(mpc, "opt_f_num", None), None)}
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
            _obs, reward, done, info = base.parse_step_result(step_result)
            step_exc = None
            step_traceback = None
        except Exception:
            reward = None
            done = False
            info = {}
            step_exc = traceback.format_exc(limit=12)
            step_traceback = step_exc
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
        costs = v0h._enhanced_costs(info, reward)
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
        steps.append({"step": step, "horizon": h, "interface": "LetMPCEnv.step(np.asarray([H])) with raw_env_tvp_values", "high_level_action": base.clean(high_level_action), "raw_tvp_restore_meta": raw_tvp_restore_meta if step == 0 else None, "state_before": state_before, "state_after": state_after, "state_delta_l1_best_effort": v0h._numeric_state_distance(state_before, state_after), "action": base.clean(act), "first_control_hash": first_control_hash if step == 0 else None, "solver_events": base.clean(new_events), "solver_status": status, "solver_iterations": new_events[-1].get("iterations") if new_events else None, "solver_wall_s": new_events[-1].get("solver_wall_s") if new_events else None, "whole_decision_wall_s": whole, "control_system_step_count_before": cs_step_before, "control_system_step_count_after": cs_step_after, "plant_step_delta_counted": delta_steps, "reward": reward, "done": done, "info": base.clean(info), "costs": costs, "env_step_exception": step_exc, "env_step_traceback": step_traceback})
        if step_exc is not None:
            stopped = "env_step_exception"
            break
        if done or success_flag or constraint_flag:
            stopped = "terminated_success_or_constraint_or_done"
            break
    if steps and steps[-1].get("solver_status") != "Solve_Succeeded":
        final_solver_failures += 1

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

    physical_values = [((s.get("costs") or {}).get("physical_or_stage_cost_best_effort")) for s in steps]
    total_values = [((s.get("costs") or {}).get("total_cost_best_effort")) for s in steps]
    return {"horizon": h, "terminal_mode": base.TERMINAL_MODE, "terminal_source_horizon": int(term_h), "terminal_note": term_note, "context_hash": context_hash, "context_identity": {"context_id": context.get("context_id"), "state_label": context.get("state_label"), "tvp_start_index": context.get("tvp_start_index"), "goal": goal_meta, "state0": state0}, "initialization_meta": base.clean(init_meta), "raw_tvp_restore_meta": base.clean(raw_tvp_restore_meta), "history_initialization_meta": base.clean(history_meta), "steps": steps, "steps_executed": int(plant_steps_in_arm), "solver_calls_observed": len(solve_events), "plant_steps_observed": int(plant_steps_in_arm), "success": success_flag, "constraint": constraint_flag, "stopped_reason": stopped, "first_control_hash": first_control_hash, "physical_or_stage_cost_sum_best_effort": finite_sum(physical_values), "total_cost_sum_best_effort": finite_sum(total_values), "decision_timing_summary_s": base.timing_summary([s.get("whole_decision_wall_s") for s in steps]), "solver_timing_summary_s": base.timing_summary([s.get("solver_wall_s") for s in steps]), "solver_status_counts": {str(k): sum(1 for ev in solve_events if str(ev.get("return_status")) == str(k)) for k in sorted(set(str(ev.get("return_status")) for ev in solve_events))}, "initial_failed_steps": initial_solver_failures, "final_failed_steps": final_solver_failures, "solver_failure_steps": solver_failure_steps}


def _strict_microcontinuation_evidence_current() -> Dict[str, Any]:
    diag_root = ROOT / "research_artifacts/aws_diagnostics"
    candidates: List[Tuple[float, Path]] = []
    for path in diag_root.glob(NAME + "_*/raw.json"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            pass
    if not candidates:
        return {"source242_microcontinuation_produced_plant_transitions": False, "all_three_horizon_arms_advanced_at_least_one_plant_step": False, "no_env_or_controller_exceptions_in_attempted_arms": False, "strict_microcontinuation_evidence_source": None}
    candidates.sort()
    raw_path = candidates[-1][1]
    try:
        raw = base.read_json(raw_path)
    except Exception as exc:
        return {"source242_microcontinuation_produced_plant_transitions": False, "all_three_horizon_arms_advanced_at_least_one_plant_step": False, "no_env_or_controller_exceptions_in_attempted_arms": False, "strict_microcontinuation_evidence_source": base.rel(raw_path), "strict_microcontinuation_evidence_error": "%s: %s" % (type(exc).__name__, exc)}
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
    return {"source242_microcontinuation_produced_plant_transitions": plant_total > 0, "all_three_horizon_arms_advanced_at_least_one_plant_step": all_three_advanced, "no_env_or_controller_exceptions_in_attempted_arms": exception_free, "strict_microcontinuation_evidence_source": base.rel(raw_path), "observed_total_plant_steps": plant_total, "observed_arm_stop_reasons": stop_reasons, "observed_plant_steps_by_horizon": plant_by_horizon}


def _augment_evidence_v0i(evidence: Any) -> Any:
    if not isinstance(evidence, Mapping):
        return evidence
    out = dict(evidence)
    backup_ok = bool(out.get("backup_verified_before_solver_or_plant_steps"))
    out.setdefault("backup_verified_after_v0h_failure_v0i_source_and_plan_before_solver_or_plant_steps", backup_ok)
    out.setdefault("S_TC2H7_zero_solver_zero_plant_env_tvp_format_failure_preserved", V0H_RECEIPT.exists())
    out.setdefault("materialized_185232_backup_proof_preserved", MATERIALIZED_BACKUP_PROOF_185232.exists())
    out.setdefault("materialized_185834_backup_proof_preserved", MATERIALIZED_BACKUP_PROOF_185834.exists())
    out.setdefault("raw_tvp_format_restored_for_LetMPCEnv_step", True)
    out.setdefault("high_level_env_step_interface_repair_applied", True)
    out.update(_strict_microcontinuation_evidence_current())
    return out


def record_outcome_v0i(root: Any, outcome: str, used: Mapping[str, int], evidence: Any, engineering_error: Optional[str] = None) -> Any:
    if engineering_error == "backup_dependency":
        engineering_error = "dependency"
    return _previous_record_outcome(root, outcome, used, _augment_evidence_v0i(evidence), engineering_error=engineering_error)


base.NAME = NAME
base.TASK_ID = TASK_ID
base.__dict__["__file__"] = str(SOURCE)
base.verify_pre_resource_backup_and_priors = verify_pre_resource_backup_and_priors_v0i
base.run_arm = run_arm_raw_tvp_highlevel
base.execution_contract.record_outcome = record_outcome_v0i

if __name__ == "__main__":
    raise SystemExit(base.main())
