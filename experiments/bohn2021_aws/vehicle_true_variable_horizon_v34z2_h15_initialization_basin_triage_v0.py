#!/usr/bin/env python3
"""S-TC2G H15 initialization-basin triage for v34z2 source242.

Temporary GPT-5.5 solo plan task S-TC2G-h15-initialization-basin-triage-v0.
Development-only diagnostic: opened source242 H15 V15_shared, no plant rollout,
no training/refit, no validation64, no sealed/final test. The script solves at
most three production H15 cells initialized by convex blends between the prior
T-C2 canonical zero-control initialization and the prior T-C2 goal-facing
initialization. Alpha=0 and alpha=1 are anchors from T-C2 and are not rerun.
"""
from __future__ import annotations

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

NAME = "vehicle_true_variable_horizon_v34z2_h15_initialization_basin_triage_v0"
TASK_ID = "S-TC2G-h15-initialization-basin-triage-v0"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO_RESOURCES = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}
ALPHAS = [0.25, 0.50, 0.75]
HORIZON = 15
TERMINAL_MODE = "V15_shared"
EXPECTED_GATE_SHA256 = "66dbf84e4a7917d824152cae1cac4da77c51efff576cd4775e8cd1be29dbcc97"
EXPECTED_BACKUP_COMMIT_FROM_SUPERVISOR_CONTEXT = "248d0c4dcd723a406290ac9c9f2070141d85b002"
EXPECTED_BACKUP_TIME_FROM_SUPERVISOR_CONTEXT = "2026-09-30T16:43:07.727241+00:00"
EXPECTED_BACKUP_PACKAGE_SHA256_FROM_SUPERVISOR_CONTEXT = "6d50f2bd3782ec7e02791b987c9d38b845ab5b71cf553bebc21df9896917c140"
PRIOR_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0_20260930T163737Z/completed.json"
PRIOR_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0_20260930T163737Z/raw.json"
PRIOR_CSV = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0_20260930T163737Z/cell_metrics.csv"
GATE_SOURCE = AWS_DIR / "vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0.py"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
DOCS_TO_APPEND = [
    ROOT / "STATUS.md",
    ROOT / "RESEARCH_LOG.md",
    ROOT / "DECISIONS.md",
    ROOT / "RESULTS_AUDIT.md",
    ROOT / "REPRODUCTION_PROTOCOL.md",
    RESPONSE_LOG,
]


class TriageError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Any) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        np = None  # type: ignore
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if np is not None and isinstance(value, (np.integer,)):
        return int(value)
    if np is not None and isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items() if k != "nlp_expr"}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        try:
            return clean(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return clean(value.tolist())
        except Exception:
            pass
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def arr_hash(value: Any) -> Optional[str]:
    return gate.arr_hash(value)


def timing_summary(values: Sequence[Any]) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean_s": None, "median_s": None, "p95_s": None, "min_s": None, "max_s": None}
    vals_sorted = sorted(vals)
    if len(vals_sorted) == 1:
        p95 = vals_sorted[0]
    else:
        pos = 0.95 * (len(vals_sorted) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        p95 = vals_sorted[lo] * (1.0 - (pos - lo)) + vals_sorted[hi] * (pos - lo)
    return {"n": len(vals), "mean_s": float(mean(vals)), "median_s": float(median(vals)), "p95_s": float(p95), "min_s": float(min(vals)), "max_s": float(max(vals))}


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


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


def verify_backup_before_solver() -> Dict[str, Any]:
    """Verify the prior T-C2 evidence is covered by the latest supervisor backup.

    The external release proof itself is supplied by the supervisor context for this
    cycle. Locally, before any solve, we verify that the current git HEAD is the
    backed commit (or has that commit as an ancestor) and that the prior T-C2
    evidence files are clean relative to that repository state. The new triage
    script can be uncommitted; the gate concerns preservation of the prior unique
    solver evidence before spending new solver calls.
    """
    head = run_git(["rev-parse", "HEAD"])
    ancestor = run_git(["merge-base", "--is-ancestor", EXPECTED_BACKUP_COMMIT_FROM_SUPERVISOR_CONTEXT, "HEAD"])
    prior_paths = [rel(PRIOR_DONE), rel(PRIOR_RAW), rel(PRIOR_CSV), rel(GATE_SOURCE)]
    status = run_git(["status", "--porcelain", "--"] + prior_paths)
    present = {p: Path(ROOT / p).exists() for p in prior_paths}
    hashes = {p: sha256(ROOT / p) for p in prior_paths if Path(ROOT / p).exists()}
    ok = bool(
        head.get("ok")
        and status.get("ok")
        and status.get("stdout") == ""
        and all(present.values())
        and hashes.get(rel(GATE_SOURCE)) == EXPECTED_GATE_SHA256
        and (head.get("stdout") == EXPECTED_BACKUP_COMMIT_FROM_SUPERVISOR_CONTEXT or ancestor.get("ok"))
    )
    return {
        "verified_before_solver_calls": ok,
        "verification_source": "supervisor_context_latest_verified_external_backup plus local git cleanliness for prior T-C2 evidence",
        "supervisor_context_backup_time": EXPECTED_BACKUP_TIME_FROM_SUPERVISOR_CONTEXT,
        "supervisor_context_backup_commit": EXPECTED_BACKUP_COMMIT_FROM_SUPERVISOR_CONTEXT,
        "supervisor_context_backup_package_sha256": EXPECTED_BACKUP_PACKAGE_SHA256_FROM_SUPERVISOR_CONTEXT,
        "current_git_head": head,
        "expected_backup_commit_is_ancestor_of_head": ancestor,
        "prior_paths_checked": prior_paths,
        "prior_paths_present": present,
        "prior_paths_git_status_porcelain": status,
        "prior_path_hashes": hashes,
        "new_script_may_be_unbacked_until_post_run_backup_request": True,
    }


def prior_anchor_summary() -> Dict[str, Any]:
    if not PRIOR_DONE.exists() or not PRIOR_RAW.exists() or not PRIOR_CSV.exists():
        raise TriageError("missing prior T-C2 evidence files")
    done = read_json(PRIOR_DONE)
    rows: Dict[str, Dict[str, Any]] = {}
    with PRIOR_CSV.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows[str(row.get("cell_label"))] = dict(row)
    h15c = rows.get("H15_canonical")
    h15g = rows.get("H15_goal_facing")
    if not h15c or not h15g:
        raise TriageError("prior T-C2 CSV lacks H15 anchor cells")
    if str(h15c.get("return_status")) != "Solve_Succeeded" or str(h15g.get("return_status")) != "Infeasible_Problem_Detected":
        raise TriageError("prior T-C2 anchor statuses are not the expected canonical-success/goal-facing-infeasible pair")
    if int((done.get("budget_actual") or {}).get("solver_calls", -1)) != 5:
        raise TriageError("prior T-C2 did not record exactly five solver calls")
    return {
        "prior_completed": rel(PRIOR_DONE),
        "prior_raw": rel(PRIOR_RAW),
        "prior_cell_metrics": rel(PRIOR_CSV),
        "prior_completed_sha256": sha256(PRIOR_DONE),
        "prior_raw_sha256": sha256(PRIOR_RAW),
        "prior_cell_metrics_sha256": sha256(PRIOR_CSV),
        "prior_hard_pass": done.get("hard_pass"),
        "prior_budget_actual": done.get("budget_actual"),
        "canonical_anchor": h15c,
        "goal_facing_anchor": h15g,
        "alpha_0_and_1_not_rerun": True,
    }


def blend_label(alpha: float) -> str:
    return "H15_blend_alpha_%s" % (str(alpha).replace(".", "p"))


def blended_profiles(context_meta: Mapping[str, Any], h: int, alpha: float) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], Dict[str, Any]]:
    state = gate.numeric_state_dict_from_meta(context_meta)
    canonical_states, canonical_controls = gate.zero_guess(state, h)
    goal_states, goal_controls = gate.predicted_unicycle(state, float(context_meta["goal_x"]), float(context_meta["goal_y"]), h, float(gate.finite_float(getattr(context_meta.get("mpc_object", None), "t_step", None), 0.1) or 0.1))
    # The context_meta does not carry the MPC object; use the historical T-C2 rule
    # with t_step=0.1 unless the controller exposes a different value in prepare.
    blended_states: List[Dict[str, float]] = []
    blended_controls: List[Dict[str, float]] = []
    for cs, gs in zip(canonical_states, goal_states):
        blended_states.append({k: float((1.0 - alpha) * cs[k] + alpha * gs[k]) for k in cs})
    for cu, gu in zip(canonical_controls, goal_controls):
        blended_controls.append({k: float((1.0 - alpha) * cu[k] + alpha * gu[k]) for k in cu})
    meta = {
        "alpha": alpha,
        "canonical_profile_hash": canonical_hash({"states": canonical_states, "controls": canonical_controls}),
        "goal_facing_profile_hash": canonical_hash({"states": goal_states, "controls": goal_controls}),
        "blended_profile_hash": canonical_hash({"states": blended_states, "controls": blended_controls}),
        "first_three_canonical_controls": canonical_controls[:3],
        "first_three_goal_facing_controls": goal_controls[:3],
        "first_three_blended_controls": blended_controls[:3],
        "first_three_blended_states": blended_states[:3],
    }
    return blended_states, blended_controls, meta


def set_blended_initial_guess(mpc: Any, env: Any, context_meta: Mapping[str, Any], alpha: float, h: int) -> Dict[str, Any]:
    ctrl = env.control_system.controller
    state_names = list(getattr(ctrl, "state_names", ["theta", "x", "y"]))
    input_names = list(getattr(ctrl, "input_names", ["u_omega", "u_s"]))
    state = gate.numeric_state_dict_from_meta(context_meta)
    dt_s = float(gate.finite_float(getattr(mpc, "t_step", None), 0.1) or 0.1)
    canonical_states, canonical_controls = gate.zero_guess(state, h)
    goal_states, goal_controls = gate.predicted_unicycle(state, float(context_meta["goal_x"]), float(context_meta["goal_y"]), h, dt_s)
    pred_states = [{k: float((1.0 - alpha) * cs[k] + alpha * gs[k]) for k in cs} for cs, gs in zip(canonical_states, goal_states)]
    pred_controls = [{k: float((1.0 - alpha) * cu[k] + alpha * gu[k]) for k in cu} for cu, gu in zip(canonical_controls, goal_controls)]

    obj = getattr(mpc, "opt_x_num", None)
    if obj is None:
        raise TriageError("mpc.opt_x_num unavailable for blended initialization")
    before_hash = arr_hash(obj)
    attempted = 0
    succeeded = 0
    readback_ok = 0
    mismatches: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []
    for label in gate.get_struct_labels(obj):
        parts = gate.parse_label(label)
        if not parts:
            continue
        top = parts[0]
        ints = [int(p) for p in parts[1:] if isinstance(p, int)]
        intended: Optional[float] = None
        kind: Optional[str] = None
        var: Optional[str] = None
        canonical_value: Optional[float] = None
        goal_value: Optional[float] = None
        if top == "_x":
            k = ints[0] if ints else 0
            var = gate.variable_from_parts(parts, state_names)
            if var is not None and 0 <= k < len(pred_states):
                canonical_value = float(canonical_states[k].get(var, state.get(var, 0.0)))
                goal_value = float(goal_states[k].get(var, state.get(var, 0.0)))
                intended = float(pred_states[k].get(var, state.get(var, 0.0)))
                kind = "state"
        elif top == "_u":
            k = ints[0] if ints else 0
            var = gate.variable_from_parts(parts, input_names)
            if var is not None and 0 <= k < len(pred_controls):
                canonical_value = float(canonical_controls[k].get(var, 0.0))
                goal_value = float(goal_controls[k].get(var, 0.0))
                intended = float(pred_controls[k].get(var, 0.0))
                kind = "input"
        elif top == "_eps":
            intended = 0.0
            canonical_value = 0.0
            goal_value = 0.0
            kind = "epsilon"
            var = gate.variable_from_parts(parts, [str(x) for x in parts if isinstance(x, str) and x != "_eps"])
        if intended is None:
            continue
        attempted += 1
        idx = tuple(parts)
        ok = gate.safe_struct_set(obj, idx, intended)
        got = gate.safe_struct_get(obj, idx) if ok else None
        match = bool(got is not None and abs(float(got) - intended) <= 1e-10)
        if ok:
            succeeded += 1
        if match:
            readback_ok += 1
        elif len(mismatches) < 20:
            mismatches.append({"label": label, "intended": intended, "got": got, "set_ok": ok, "kind": kind, "var": var})
        if len(examples) < 18 and kind in ("state", "input"):
            examples.append({"label": label, "kind": kind, "var": var, "canonical": canonical_value, "goal_facing": goal_value, "alpha": alpha, "intended": intended, "readback": got, "match": match})
    try:
        mpc.lam_g_num = 0 * mpc.lam_g_num
    except Exception:
        pass
    after_hash = arr_hash(obj)
    profile = {
        "alpha": alpha,
        "canonical_profile_hash": canonical_hash({"states": canonical_states, "controls": canonical_controls}),
        "goal_facing_profile_hash": canonical_hash({"states": goal_states, "controls": goal_controls}),
        "blended_profile_hash": canonical_hash({"states": pred_states, "controls": pred_controls}),
        "first_three_pred_states": pred_states[:3],
        "first_three_pred_controls": pred_controls[:3],
        "first_three_canonical_controls": canonical_controls[:3],
        "first_three_goal_facing_controls": goal_controls[:3],
    }
    return {
        "initialization": "blend_alpha_%s" % str(alpha),
        "rule": "convex blend of T-C2 canonical zero-control and T-C2 deterministic goal-facing primals; eps initialized to zero",
        "alpha": alpha,
        "state_names": state_names,
        "input_names": input_names,
        "assignments_attempted": attempted,
        "assignments_succeeded": succeeded,
        "numeric_readback_count": readback_ok,
        "numeric_readback_all_successful": bool(attempted > 0 and readback_ok == attempted and not mismatches),
        "mismatches_first20": mismatches,
        "assignment_examples": examples,
        "initial_primal_hash_before": before_hash,
        "initial_primal_hash_after": after_hash,
        "initial_dual_hash_after_zero": arr_hash(getattr(mpc, "lam_g_num", [])) if hasattr(mpc, "lam_g_num") else None,
        "profile": profile,
    }


def prepare_blend_cell(alpha: float, context: Mapping[str, Any], terminals: Mapping[int, Any]) -> Dict[str, Any]:
    base = gate.MODULES["base"]
    v34u = gate.MODULES["v34u"]
    h = HORIZON
    cell = blend_label(alpha)
    row = {"cell_label": cell, "horizon": h, "initialization": "blend_alpha_%s" % str(alpha), "role": "triage_%s" % cell, "solve_mode": "production"}
    arm_id = "ctx=source242_slot0_branch_start|H15|term=V15_shared|init=blend_alpha_%s|role=%s" % (str(alpha), row["role"])
    gate.CURRENT_ARM_ID = arm_id
    gate.CURRENT_SOLVE_MODE = "production"
    cap_before = len(gate.CAPTURES)
    terminal, term_h, term_note = base.terminal_for_mode(h, TERMINAL_MODE, terminals)
    env = base.create_env(h, terminal)
    ctrl = env.control_system.controller
    mpc = ctrl.mpc
    context_meta = base.configure_context_no_reset(env, context, h)
    scalar_tvp = gate.scalarize_tvp(context_meta["shifted_tvp"])
    context_meta["shifted_tvp"] = scalar_tvp
    goal = gate.goal_endpoint_61(context["case_snapshot"])
    context_meta["goal_x"] = goal["goal_x"]
    context_meta["goal_y"] = goal["goal_y"]
    context_meta["goal_source"] = goal["source"]
    context_meta["goal_endpoint_index"] = goal["endpoint_index"]
    ctrl.goal_x = goal["goal_x"]
    ctrl.goal_y = goal["goal_y"]
    ctrl._tvp_data = json.loads(json.dumps(clean(scalar_tvp)))
    v34u.install_tta_hmpc_presolve_obj_guard()
    init_meta = set_blended_initial_guess(mpc, env, context_meta, alpha, h)
    pre = gate.capture_mpc_state(mpc)
    captures = [c for c in gate.CAPTURES[cap_before:] if c.get("arm_id") == gate.CURRENT_ARM_ID]
    setup = {
        "arm_id": arm_id,
        "cell_label": cell,
        "role": row["role"],
        "horizon": h,
        "terminal_mode": TERMINAL_MODE,
        "terminal_source_horizon": int(term_h),
        "terminal_note": term_note,
        "initialization": row["initialization"],
        "alpha": alpha,
        "solve_mode": "production",
        "goal": goal,
        "context_meta_no_shifted_tvp": {k: v for k, v in context_meta.items() if k != "shifted_tvp"},
        "initialization_meta": init_meta,
        "pre_solve_state_after_initialization": pre,
        "captured_nlpsol_meta": clean(captures),
        "captured_nlpsol_count_during_construction": len(captures),
        "production_max_iter_not_forced_to_1": bool(all(c.get("effective_ipopt_max_iter") != 1 for c in captures)),
    }
    return {"env": env, "ctrl": ctrl, "mpc": mpc, "context_meta": context_meta, "setup": setup, "capture_start": cap_before, "arm_id": arm_id, "row": row}


def write_cell_csv(path: Path, cell_results: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "cell_label", "alpha", "return_status", "unified_return_status", "actual_iteration_count", "converged",
        "J_solver", "J_recon_historical_noeps", "J_recon_full", "relative_error_historical_noeps", "relative_error_full",
        "objective_contract_cell_pass", "hard_defect_converged_rel_error_gt_1e_minus_4", "solver_wall_s",
        "whole_decision_get_action_wall_s", "epsilon_dimension_per_stage", "individual_eps_nonzero_count", "epsilon_abs_sum",
        "per_stage_weighted_penalty_sum", "cons_lb_cons_ub_residual", "lb_opt_x_ub_opt_x_residual", "initial_primal_hash_after",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in cell_results:
            ev = r.get("solve_event") or {}
            eps = r.get("epsilon_capture") or {}
            setup = r.get("setup") or {}
            init = setup.get("initialization_meta") or {}
            cons = r.get("constraint_residuals") or {}
            writer.writerow({
                "cell_label": r.get("cell_label"),
                "alpha": setup.get("alpha"),
                "return_status": r.get("return_status"),
                "unified_return_status": r.get("unified_return_status"),
                "actual_iteration_count": r.get("actual_iteration_count"),
                "converged": r.get("converged"),
                "J_solver": r.get("J_solver"),
                "J_recon_historical_noeps": r.get("J_recon_historical_noeps"),
                "J_recon_full": r.get("J_recon_full"),
                "relative_error_historical_noeps": r.get("relative_error_historical_noeps"),
                "relative_error_full": r.get("relative_error_full"),
                "objective_contract_cell_pass": r.get("objective_contract_cell_pass"),
                "hard_defect_converged_rel_error_gt_1e_minus_4": r.get("hard_defect_converged_rel_error_gt_1e_minus_4"),
                "solver_wall_s": ev.get("solver_wall_s"),
                "whole_decision_get_action_wall_s": r.get("controller_get_action_wall_s"),
                "epsilon_dimension_per_stage": eps.get("epsilon_dimension_per_stage"),
                "individual_eps_nonzero_count": eps.get("individual_eps_nonzero_count"),
                "epsilon_abs_sum": eps.get("epsilon_abs_sum"),
                "per_stage_weighted_penalty_sum": eps.get("per_stage_weighted_penalty_sum"),
                "cons_lb_cons_ub_residual": cons.get("cons_lb_cons_ub_residual"),
                "lb_opt_x_ub_opt_x_residual": cons.get("lb_opt_x_ub_opt_x_residual"),
                "initial_primal_hash_after": init.get("initial_primal_hash_after"),
            })


def write_failure(run_dir: Path, created: dt.datetime, error: str, used: Mapping[str, int], dependency_zero: bool = False) -> int:
    failed_path = run_dir / "failed.json"
    payload = {
        "status": "failed",
        "task_id": TASK_ID,
        "created_utc": created.isoformat(),
        "error": error,
        "traceback_tail": traceback.format_exc().splitlines()[-18:],
        "budget_actual": dict(used),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
    }
    write_json(failed_path, payload)
    try:
        if dependency_zero and all(int(v) == 0 for v in used.values()):
            execution_contract.record_outcome(ROOT, "engineering_failure", dict(used), {"no_scientific_outcome": True, "backup_verified_before_solver_calls_or_zero_usage_dependency_failure": True, "failed_json": rel(failed_path), "error": error}, engineering_error="dependency")
        else:
            evidence = {
                "h15_initialization_basin_triage_completed": False,
                "prior_T_C2_failure_preserved": PRIOR_DONE.exists() and PRIOR_RAW.exists() and PRIOR_CSV.exists(),
                "backup_verified_before_solver_calls_or_zero_usage_dependency_failure": bool(dependency_zero and all(int(v) == 0 for v in used.values())),
                "initialization_hashes_and_alpha_grid_reported": False,
                "direct_nlp_and_objective_reconstruction_reported": False,
                "no_plant_training_validation_or_test_usage": True,
                "solver_calls_at_most_three": int(used.get("solver_calls", 999)) <= 3,
            }
            execution_contract.record_outcome(ROOT, "scientific_result", dict(used), evidence)
    except Exception:
        pass
    print(json.dumps(clean({"failed": error, "failed_json": rel(failed_path), "resources": dict(used)}), sort_keys=True), flush=True)
    return 1


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "research_artifacts/aws_diagnostics" / f"{NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = f"vehicle-tc2g-h15-initialization-basin-triage-{stamp}"
    try:
        snapshot = execution_contract.runtime_snapshot(ROOT)
        if snapshot is None:
            raise TriageError("missing structured execution snapshot")
        if (snapshot.get("task") or {}).get("task_id") != TASK_ID:
            raise TriageError("unexpected task_id in snapshot: %r" % ((snapshot.get("task") or {}).get("task_id"),))
        if sha256(GATE_SOURCE) != EXPECTED_GATE_SHA256:
            raise TriageError("wrapped v34z2 gate source hash mismatch")

        prior = prior_anchor_summary()
        backup = verify_backup_before_solver()
        if backup.get("verified_before_solver_calls") is not True:
            write_json(run_dir / "backup_dependency_failure.json", {"created_utc": created.isoformat(), "prior": prior, "backup_verification": backup, "budget_actual": dict(gate.RESOURCE_USAGE)})
            return write_failure(run_dir, created, "post-T-C2 backup verification failed before solver calls", dict(ZERO_RESOURCES), dependency_zero=True)

        gate.RESOURCE_USAGE = dict(ZERO_RESOURCES)
        gate.SOLVER_CAP = 3
        gate.load_modules()
        gate.install_nlpsol_patch()
        context, terminals, load_meta = gate.load_context_and_terminals()

        cell_results: List[Dict[str, Any]] = []
        setup_records: List[Dict[str, Any]] = []
        stop_reason: Optional[str] = None
        for i, alpha in enumerate(ALPHAS):
            prep = prepare_blend_cell(alpha, context, terminals)
            setup = clean(prep["setup"])
            setup_records.append(setup)
            init_meta = setup.get("initialization_meta") or {}
            if int(init_meta.get("assignments_attempted", 0) or 0) <= 0 or int(init_meta.get("assignments_succeeded", 0) or 0) <= 0 or init_meta.get("numeric_readback_all_successful") is not True:
                stop_reason = "inert_or_unreadable_blended_initialization_before_solver_alpha_%s" % alpha
                break
            result = gate.solve_prepared(prep, i)
            cell_results.append(clean(result))
            progress = {"cells_solved": len(cell_results), "new_solver_calls": gate.RESOURCE_USAGE["solver_calls"], "last_alpha": alpha, "last_status": result.get("return_status"), "validation64_bank_opened": False, "sealed_test_accessed": False}
            write_json(run_dir / "progress.json", progress)
            print(json.dumps(clean(progress), sort_keys=True), flush=True)
            if result.get("hard_defect_converged_rel_error_gt_1e_minus_4") is True:
                stop_reason = "stopped_after_converged_hard_defect_relative_error_gt_1e-4_alpha_%s" % alpha
                break
            if gate.RESOURCE_USAGE["solver_calls"] >= 3:
                stop_reason = "solver_call_cap_reached"
                break

        alpha_grid_report = {
            "requested_alpha_grid": ALPHAS,
            "executed_alpha_grid": [((r.get("setup") or {}).get("alpha")) for r in cell_results],
            "do_not_rerun_alpha_0_or_1": True,
            "prior_anchor_alpha_0": prior["canonical_anchor"],
            "prior_anchor_alpha_1": prior["goal_facing_anchor"],
            "initialization_hashes": [
                {
                    "cell_label": s.get("cell_label"),
                    "alpha": s.get("alpha"),
                    "initial_primal_hash_after": ((s.get("initialization_meta") or {}).get("initial_primal_hash_after")),
                    "profile": ((s.get("initialization_meta") or {}).get("profile")),
                }
                for s in setup_records
            ],
            "unique_initial_primal_hashes": len(set([str(((s.get("initialization_meta") or {}).get("initial_primal_hash_after"))) for s in setup_records])),
        }
        direct_and_recon_ok = bool(cell_results) and all((r.get("direct_nlp_f_eval") or {}).get("available") is True and (r.get("objective_reconstruction_full") or {}).get("available") is True for r in cell_results)
        cell_csv = run_dir / "cell_metrics.csv"
        write_cell_csv(cell_csv, cell_results)
        epsilon_path = run_dir / "epsilon_vectors.json"
        write_json(epsilon_path, {"epsilon_vectors": [{"cell_label": r.get("cell_label"), "alpha": (r.get("setup") or {}).get("alpha"), "arm_id": r.get("arm_id"), "epsilon_capture": r.get("epsilon_capture")} for r in cell_results]})

        solver_times = [(r.get("solve_event") or {}).get("solver_wall_s") for r in cell_results]
        whole_times = [r.get("controller_get_action_wall_s") for r in cell_results]
        converged = [r for r in cell_results if r.get("converged") is True]
        statuses = [{"alpha": (r.get("setup") or {}).get("alpha"), "status": r.get("return_status"), "iterations": r.get("actual_iteration_count"), "J_solver": r.get("J_solver"), "rel_full": r.get("relative_error_full"), "eps_abs_sum": (r.get("epsilon_capture") or {}).get("epsilon_abs_sum"), "eps_weighted_sum": (r.get("epsilon_capture") or {}).get("per_stage_weighted_penalty_sum")} for r in cell_results]
        evidence = {
            "h15_initialization_basin_triage_completed": bool(len(cell_results) == 3 or stop_reason is not None),
            "prior_T_C2_failure_preserved": True,
            "backup_verified_before_solver_calls_or_zero_usage_dependency_failure": True,
            "initialization_hashes_and_alpha_grid_reported": bool(alpha_grid_report["executed_alpha_grid"] and len(alpha_grid_report["initialization_hashes"]) >= len(cell_results) and alpha_grid_report["requested_alpha_grid"] == ALPHAS),
            "direct_nlp_and_objective_reconstruction_reported": direct_and_recon_ok,
            "no_plant_training_validation_or_test_usage": True,
            "solver_calls_at_most_three": int(gate.RESOURCE_USAGE.get("solver_calls", 999)) <= 3,
        }
        hard_pass = all(evidence.values())
        raw = {
            "status": "complete",
            "task_id": TASK_ID,
            "snapshot_sha256": snapshot.get("snapshot_sha256"),
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "server_api_token_audit": read_api_total_tokens(),
            "classification": "development_IMPROVED_S_TC2G_h15_initialization_basin_triage_not_validation_not_test",
            "prior_T_C2_anchor_evidence": prior,
            "backup_verification_before_solver_calls": backup,
            "load_meta": load_meta,
            "alpha_grid_report": alpha_grid_report,
            "setup_records": setup_records,
            "cell_results": cell_results,
            "status_summary": statuses,
            "converged_alpha_count": len(converged),
            "solver_timing_summary_s": timing_summary(solver_times),
            "whole_decision_get_action_timing_summary_s": timing_summary(whole_times),
            "stop_reason": stop_reason,
            "budget_declared": {"solver_calls": 3, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0},
            "budget_actual": dict(gate.RESOURCE_USAGE),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "new_aws_resources": False,
            "interpretation_limits": [
                "opened development source242 H15 V15_shared only",
                "initialization-basin diagnostic only; no plant rollout or validation episode",
                "alpha=0 canonical and alpha=1 goal-facing anchors are prior T-C2 evidence and were not rerun",
                "temporary GPT-5.5 solo self-review; no independent Opus/Astra acceptance",
            ],
            "pass_evidence": evidence,
            "hard_pass": hard_pass,
        }
        raw_path = run_dir / "raw.json"
        summary_path = run_dir / "summary.md"
        completed_path = run_dir / "completed.json"
        backup_request = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_S_TC2G_H15_INITIALIZATION_BASIN_TRIAGE_{stamp}.json"
        state_path = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_s_tc2g_h15_initialization_basin_triage.md"
        write_json(raw_path, raw)
        write_json(backup_request, {"request": "backup_after_S_TC2G_h15_initialization_basin_triage", "created_utc": created.isoformat(), "must_cover": [rel(Path(__file__).resolve()), rel(run_dir), rel(backup_request), rel(state_path), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", rel(RESPONSE_LOG)], "solver_calls": gate.RESOURCE_USAGE["solver_calls"], "validation64_bank_opened": False, "sealed_test_accessed": False})

        summary_lines = [
            "# S-TC2G H15 initialization-basin triage",
            "",
            "UTC: `%s`. Task `%s`." % (created.isoformat(), TASK_ID),
            "",
            "Development-only IMPROVED diagnostic on opened source242 H15 V15_shared. No plant rollout, training/refit, validation64, or sealed/final test was used.",
            "",
            "Prior anchors were preserved from T-C2 and not rerun: alpha=0 canonical was `Solve_Succeeded`; alpha=1 goal-facing was `Infeasible_Problem_Detected` with large epsilon/slack penalty.",
            "",
            "Executed blend statuses:",
        ]
        for row in statuses:
            summary_lines.append("- alpha `%s`: status `%s`, iterations `%s`, J `%s`, rel_full `%s`, eps_abs_sum `%s`, eps_weighted_sum `%s`." % (row.get("alpha"), row.get("status"), row.get("iterations"), row.get("J_solver"), row.get("rel_full"), row.get("eps_abs_sum"), row.get("eps_weighted_sum")))
        summary_lines += [
            "",
            "Solver calls: `%s` / 3. Local task hard_pass: `%s`. Stop reason: `%s`." % (gate.RESOURCE_USAGE["solver_calls"], hard_pass, stop_reason),
            "",
            "Evidence: `%s`, `%s`, `%s`, `%s`." % (rel(raw_path), rel(cell_csv), rel(epsilon_path), rel(completed_path)),
        ]
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        doc_block = "\n".join([
            "<!-- %s -->" % marker,
            "## S-TC2G H15 initialization-basin triage",
            "",
            "UTC: %s. Local task hard_pass `%s`; solver calls `%s`/3; plant/training/validation/test all zero. Alpha statuses: `%s`. Evidence: `%s`, `%s`, `%s`. Backup request: `%s`. This is opened-development IMPROVED self-review evidence only, not validation or final-test evidence." % (created.isoformat(), hard_pass, gate.RESOURCE_USAGE["solver_calls"], statuses, rel(summary_path), rel(raw_path), rel(cell_csv), rel(backup_request)),
        ])
        for doc in DOCS_TO_APPEND:
            append_if_missing(doc, marker, doc_block)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("# Continue state after S-TC2G\n\n" + doc_block + "\n\nNext: inspect raw outcomes and publish the next bounded solo plan before further solver/control/training work because this task has continue_without_review=false.\n", encoding="utf-8")
        with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow([created.isoformat(), NAME, raw["classification"], "canonical_solve_no_training_seed", "opened_development_source242_H15_only_no_validation64_no_sealed_test", gate.RESOURCE_USAGE["solver_calls"], 0, 0, 0, 0, False, rel(completed_path), marker])
        completed = {
            "status": "complete",
            "task_id": TASK_ID,
            "created_utc": created.isoformat(),
            "classification": raw["classification"],
            "summary": rel(summary_path),
            "raw": rel(raw_path),
            "cell_metrics_csv": rel(cell_csv),
            "epsilon_vectors": rel(epsilon_path),
            "backup_request": rel(backup_request),
            "state": rel(state_path),
            "budget_actual": dict(gate.RESOURCE_USAGE),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "headline": {"hard_pass": hard_pass, "solver_calls": gate.RESOURCE_USAGE["solver_calls"], "converged_alpha_count": len(converged), "statuses": statuses, "stop_reason": stop_reason},
            "pass_evidence": evidence,
        }
        hash_paths = [Path(__file__).resolve(), GATE_SOURCE, raw_path, summary_path, cell_csv, epsilon_path, completed_path, backup_request, state_path] + DOCS_TO_APPEND + [ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists() and p != completed_path}
        write_json(completed_path, completed)
        execution_contract.record_outcome(ROOT, "scientific_result", dict(gate.RESOURCE_USAGE), evidence)
        print(json.dumps(clean({"completed": rel(completed_path), "headline": completed["headline"], "pass_evidence": evidence, "server_api_token_audit": raw["server_api_token_audit"]}), sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        used = dict(getattr(gate, "RESOURCE_USAGE", ZERO_RESOURCES))
        return write_failure(run_dir, created, "unexpected S-TC2G error: %s: %s" % (type(exc).__name__, exc), used, dependency_zero=all(int(v) == 0 for v in used.values()))


if __name__ == "__main__":
    raise SystemExit(main())
