#!/usr/bin/env python3
"""v34t / T-B5+T-B3 active-plan non-converged objective-contract probe.

Prepared after v34s/T-B4 passed. This wrapper preserves the v34n bounded
low-iteration solver-contract design while binding the two operational repairs
required by the active Opus plan sequence:

* v34o strict previous_input scalarization and fail-loud _u0 setup;
* v34s/v34q authoritative saved trajectory endpoint goal extraction, with
  endpoint index == round(reference.traj_steps)-1.

It also evaluates the 16 v34k previously-passing objective alias candidates at
each solved point, records timing summaries, and keeps the inherited SOLVE_CAP=6.
Run only after a verified backup that postdates the v34s loader-gate completion.
No plant rollout, env.step/reset after construction, validation64, sealed test,
training, or selector refit is allowed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
REPRO_DIR = ROOT / "experiments/bohn2021_reproduction"
for _p in (AWS_DIR, REPRO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vehicle_true_variable_horizon_v34n_nonconverged_objective_contract_probe_v0 as v34n  # noqa:E402
import vehicle_true_variable_horizon_v34o_loader_gate_v0 as v34o  # noqa:E402
import vehicle_true_variable_horizon_v34s_loader_gate_v0 as v34s  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34t_nonconverged_objective_contract_probe_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
# Current activated lead plan supplied by the supervisor/user context.
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T120929Z_f87f2b.md"
OPUS_REPORT_SHA = "ded62e6413a9ab52c0c2f2ed651e592fe6203bbcb0650ed002beff1e9dd6a2a5"
OPUS_REQUEST = "execution-result:20260930T120837_e51ace56"
# Immediate predecessor plan that authorized the v34s loader-gate artifact used below.
PREDECESSOR_OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T115516Z_8330f5.md"
PREDECESSOR_OPUS_REPORT_SHA = "3acada362e46ec94b6a21c9738e076a2c95ca8db7be95369a560969d3f7eb474"
PREDECESSOR_OPUS_REQUEST = "execution-result:20260930T115434_d4836078"
V34M_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34m_residual_attribution_v0_20260930T111306Z/completed.json"
V34N_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34n_nonconverged_objective_contract_probe_v0_20260930T112938Z/failed.json"
V34K_GLOB = str(ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_*/candidate_residuals.csv")
MAIN_CANDIDATE_ID = v34n.FORMULA_SPEC["candidate_id"]


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(v34n.clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    return v34n.sha256(path)


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        t = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def latest_completed(pattern: str, label: str) -> Path:
    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    if not paths:
        raise ContractError(f"{label} completed artifact not found")
    return paths[-1]


def latest_v34s_completed() -> Path:
    return latest_completed(str(ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34s_loader_gate_v0_*/completed.json"), "v34s/T-B4")


def extract_v34n_float_list_call_site(failed: Mapping[str, Any]) -> Dict[str, Any]:
    tb = str(failed.get("traceback") or "")
    out = {
        "error": failed.get("error"),
        "new_solver_calls_recorded": failed.get("new_solver_calls_recorded"),
        "call_site_verified": False,
        "call_site_file": None,
        "call_site_line": None,
        "call_site_symbol": None,
        "repair_binding": "v34n.base.load_contexts = v34o.load_contexts_strict before v34n.run() enters base.load_contexts()",
    }
    for line in tb.splitlines():
        if "vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py" in line and "line 338" in line:
            out.update({
                "call_site_verified": True,
                "call_site_file": "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py",
                "call_site_line": 338,
                "call_site_symbol": "base.load_contexts previous_input dict comprehension float(v)",
            })
            break
    return out


def patched_verify_gates(args: argparse.Namespace) -> Dict[str, Any]:
    for p in [v34n.PLAN_READY, OPUS_REPORT, V34M_DONE, V34N_FAILED]:
        if not p.exists():
            raise ContractError("missing prerequisite " + rel(p))
    ready = read_json(v34n.PLAN_READY)
    if ready.get("request_id") != OPUS_REQUEST or ready.get("report_sha256") != OPUS_REPORT_SHA:
        raise ContractError(f"active PLAN_READY mismatch: request={ready.get('request_id')} sha={ready.get('report_sha256')}; expected request={OPUS_REQUEST} sha={OPUS_REPORT_SHA}")
    if sha256(OPUS_REPORT) != OPUS_REPORT_SHA:
        raise ContractError("active Opus report sha mismatch")
    latest_text = v34n.OPUS_LATEST.read_text(encoding="utf-8", errors="replace") if v34n.OPUS_LATEST.exists() else ""
    if rel(OPUS_REPORT) not in latest_text:
        raise ContractError("Opus LATEST.md does not point to active report")
    v34m = read_json(V34M_DONE)
    if v34m.get("hard_pass") is not True:
        raise ContractError("v34m predecessor did not hard_pass")
    v34n_fail = read_json(V34N_FAILED)
    if int(v34n_fail.get("new_solver_calls_recorded", 0)) != 0:
        raise ContractError("v34n predecessor unexpectedly spent solver calls")
    s_path = latest_v34s_completed()
    s_done = read_json(s_path)
    if s_done.get("hard_pass") is not True or s_done.get("passed") is not True:
        raise ContractError("latest v34s/T-B4 loader gate did not pass")
    b0 = s_done.get("budget_actual") or {}
    for key in ["solver_calls", "plant_steps", "env_step_calls_after_construction", "env_reset_calls_after_construction", "new_training_or_gradient_steps", "selector_refits", "validation64_episodes", "sealed_test_episodes"]:
        if int(b0.get(key, 0)) != 0:
            raise ContractError(f"v34s loader gate budget was not zero for {key}: {b0.get(key)}")
    if (s_done.get("headline") or {}).get("previous_input_hashes_match_expected") is not True:
        raise ContractError("v34s previous_input hash gate was not true")
    bt = parse_time(args.backup_time)
    if bt is None:
        raise ContractError("backup_time not parseable")
    t_v34s = parse_time(s_done.get("created_utc"))
    if t_v34s is not None and bt <= t_v34s:
        raise ContractError(f"backup context must postdate v34s loader gate: backup={bt.isoformat()} v34s={t_v34s.isoformat()}")
    proof = {
        "status": "verified_from_supervisor_context_not_revalidated_by_script",
        "time": bt.isoformat(),
        "commit": args.backup_commit,
        "remaining_changed_files": 0,
        "packages_this_run": [{"sha256": args.backup_package_sha256, "verification": "user_context_verified_backup", "bytes": int(args.backup_package_bytes)}],
        "source": "supervisor_user_context_current_prompt",
        "purpose": "gate v34t/A13c-3 <=6 low-level solver objective-contract probe after v34s loader gate",
    }
    proof_path = v34n.BACKUP_REQUEST.parent / f"backup_proof_{v34n.STAMP}_from_user_context_before_v34t_a13c3_probe.json"
    write_json(proof_path, proof)
    call_site = extract_v34n_float_list_call_site(v34n_fail)
    if call_site.get("call_site_verified") is not True:
        raise ContractError("v34n float(list) call site was not verified from failed.json traceback")
    return {
        "active_lead_plan_ready": rel(v34n.PLAN_READY),
        "active_lead_report": rel(OPUS_REPORT),
        "active_lead_report_sha256": OPUS_REPORT_SHA,
        "active_lead_request": OPUS_REQUEST,
        "predecessor_lead_report_for_v34s": rel(PREDECESSOR_OPUS_REPORT),
        "predecessor_lead_report_sha256_for_v34s": PREDECESSOR_OPUS_REPORT_SHA,
        "predecessor_lead_request_for_v34s": PREDECESSOR_OPUS_REQUEST,
        "v34m_completed": rel(V34M_DONE),
        "v34n_failed_preserved": rel(V34N_FAILED),
        "v34n_float_list_failure_call_site": call_site,
        "v34s_loader_gate_completed": rel(s_path),
        "v34s_headline": s_done.get("headline"),
        "backup_proof": {**proof, "path": rel(proof_path), "sha256": sha256(proof_path)},
        "authorized_budget": {"low_level_solver_attempt_cap": v34n.SOLVE_CAP, "scheduled_cells": len(v34n.CELLS), "plant_steps": 0, "env_step_calls_after_construction": 0, "validation64_episodes": 0, "sealed_test_episodes": 0, "training_or_refit": 0},
    }


def parse_bool(value: Any) -> Optional[bool]:
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def parse_candidate_id(candidate_id: str) -> Optional[Dict[str, Any]]:
    parts: Dict[str, str] = {}
    for token in str(candidate_id).split("|"):
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        parts[k.strip()] = v.strip()
    needed = ["stage", "term", "eps", "r", "discount", "z"]
    if any(k not in parts for k in needed):
        return None
    eps = parse_bool(parts["eps"])
    rterm = parse_bool(parts["r"])
    if eps is None or rterm is None:
        return None
    return {
        "candidate_id": candidate_id,
        "stage_rule": parts["stage"],
        "terminal_rule": parts["term"],
        "include_eps": eps,
        "include_rterm": rterm,
        "discount_rule": parts["discount"],
        "z_rule": parts["z"],
    }


def row_is_previously_passing(row: Mapping[str, Any]) -> bool:
    for key in ["passed_1e_minus_6", "hard_pass", "pass", "passed"]:
        if key in row and row.get(key) not in (None, ""):
            return parse_bool(row.get(key)) is True
    relerr = row.get("relative_error")
    try:
        return relerr is not None and float(relerr) <= 1e-6
    except Exception:
        return False


def latest_v34k_candidates() -> Dict[str, Any]:
    paths = [Path(p) for p in sorted(glob.glob(V34K_GLOB))]
    out: Dict[str, Any] = {"path": None, "sha256": None, "candidate_count": 0, "candidates": [], "selection_rule": "v34k rows with passed_1e_minus_6/pass/hard_pass true or relative_error <= 1e-6"}
    if not paths:
        out["error"] = "v34k candidate_residuals.csv not found"
        return out
    path = paths[-1]
    out["path"] = rel(path)
    out["sha256"] = sha256(path)
    seen = set()
    candidates: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if not row_is_previously_passing(row):
                continue
            cid = row.get("candidate_id") or row.get("formula_id") or ""
            parsed = parse_candidate_id(cid)
            if parsed is None:
                continue
            key = parsed["candidate_id"]
            if key in seen:
                continue
            seen.add(key)
            candidates.append({**parsed, "v34k_row": {k: row.get(k) for k in row.keys() if k in {"rank", "candidate_id", "relative_error", "absolute_error", "abs_error", "hard_pass", "pass", "passed", "passed_1e_minus_6", "total", "solver_objective", "stage_lterm_total", "terminal_discounted"}}})
    out["candidate_count"] = len(candidates)
    out["candidates"] = candidates
    return out


def eval_alias_candidates(mpc: Any, h: int, candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        row: Dict[str, Any] = {"candidate_id": cand.get("candidate_id"), "v34k_row": cand.get("v34k_row")}
        try:
            acc = v34n.kacc.LabelAccessor(mpc)
            comp = v34n.kacc.formula_components(
                mpc,
                acc,
                int(h),
                str(cand["stage_rule"]),
                str(cand["terminal_rule"]),
                bool(cand["include_eps"]),
                bool(cand["include_rterm"]),
                str(cand["discount_rule"]),
                str(cand["z_rule"]),
            )
            row.update({
                "available": True,
                "total": comp.get("total"),
                "solver_objective": comp.get("solver_objective"),
                "relative_error": comp.get("relative_error"),
                "absolute_error": comp.get("absolute_error"),
                "stage_lterm_total": comp.get("stage_lterm_total"),
                "terminal_discounted": comp.get("terminal_discounted"),
                "epsterm_total": comp.get("epsterm_total"),
                "input_regularization_total": comp.get("input_regularization_total"),
                "eps_nonzero_count": comp.get("eps_nonzero_count"),
                "rterm_nonzero_count": comp.get("rterm_nonzero_count"),
            })
        except Exception as exc:
            row.update({"available": False, "error": repr(exc)})
        rows.append(row)
    return rows


def alias_summary(rows: Sequence[Mapping[str, Any]], main_candidate_id: str) -> Dict[str, Any]:
    available = [r for r in rows if r.get("available") and r.get("relative_error") is not None]
    if not available:
        return {"candidate_count": len(rows), "available_count": 0, "main_candidate_id": main_candidate_id}
    ranked = sorted(available, key=lambda r: float(r.get("relative_error")))
    main_rows = [r for r in available if r.get("candidate_id") == main_candidate_id]
    main_rel = None if not main_rows else float(main_rows[0]["relative_error"])
    main_rank = None
    if main_rel is not None:
        for i, r in enumerate(ranked, start=1):
            if r.get("candidate_id") == main_candidate_id:
                main_rank = i
                break
    rels = [float(r.get("relative_error")) for r in available]
    near_ties = [r.get("candidate_id") for r in ranked if float(r.get("relative_error")) <= max(1e-6, (main_rel if main_rel is not None else 0.0) + 1e-12)]
    return {
        "candidate_count": len(rows),
        "available_count": len(available),
        "main_candidate_id": main_candidate_id,
        "main_relative_error": main_rel,
        "main_rank_by_relative_error": main_rank,
        "best_candidate_id": ranked[0].get("candidate_id"),
        "best_relative_error": float(ranked[0].get("relative_error")),
        "worst_relative_error": float(ranked[-1].get("relative_error")),
        "relative_error_span": float(max(rels) - min(rels)) if rels else None,
        "near_tie_candidate_ids_relerr_le_max_1e_minus_6_or_main_plus_1e_minus_12": near_ties,
    }


def timing_stats(values: Iterable[Any]) -> Dict[str, Any]:
    xs: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            xs.append(fv)
    if not xs:
        return {"n": 0, "mean_s": None, "median_s": None, "p95_s": None, "min_s": None, "max_s": None}
    return {
        "n": len(xs),
        "mean_s": float(np.mean(xs)),
        "median_s": float(np.median(xs)),
        "p95_s": float(np.percentile(xs, 95)),
        "min_s": float(np.min(xs)),
        "max_s": float(np.max(xs)),
    }


def write_alias_candidate_csv(raw: Mapping[str, Any]) -> Path:
    path = v34n.RUN_DIR / "alias_16_candidate_metrics.csv"
    fields = [
        "arm_id", "horizon", "initialization", "cell_role", "candidate_id", "available",
        "relative_error", "absolute_error", "total", "solver_objective", "stage_lterm_total",
        "terminal_discounted", "epsterm_total", "input_regularization_total", "is_main_candidate",
        "v34k_rank", "v34k_relative_error", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for arm in raw.get("arms") or []:
            obj = arm.get("objective_reconstruction") or {}
            for r in obj.get("alias_16_candidate_metrics") or []:
                vr = r.get("v34k_row") or {}
                w.writerow({
                    "arm_id": arm.get("arm_id"),
                    "horizon": arm.get("horizon"),
                    "initialization": arm.get("initialization"),
                    "cell_role": arm.get("cell_role"),
                    "candidate_id": r.get("candidate_id"),
                    "available": r.get("available"),
                    "relative_error": r.get("relative_error"),
                    "absolute_error": r.get("absolute_error"),
                    "total": r.get("total"),
                    "solver_objective": r.get("solver_objective"),
                    "stage_lterm_total": r.get("stage_lterm_total"),
                    "terminal_discounted": r.get("terminal_discounted"),
                    "epsterm_total": r.get("epsterm_total"),
                    "input_regularization_total": r.get("input_regularization_total"),
                    "is_main_candidate": r.get("candidate_id") == MAIN_CANDIDATE_ID,
                    "v34k_rank": vr.get("rank"),
                    "v34k_relative_error": vr.get("relative_error"),
                    "error": r.get("error"),
                })
    return path


def install_augmented_formula_analyze_and_outputs() -> None:
    original_formula_try = v34n.formula_try
    original_analyze = v34n.analyze
    original_write_outputs = v34n.write_outputs
    candidate_source = latest_v34k_candidates()

    def formula_try_augmented(mpc: Any, h: int, include_eps: bool = False, include_r: bool = False) -> Dict[str, Any]:
        comp = original_formula_try(mpc, h, include_eps, include_r)
        if include_eps is False and include_r is False:
            rows = eval_alias_candidates(mpc, int(h), candidate_source.get("candidates") or [])
            comp["alias_16_candidate_source"] = candidate_source
            comp["alias_16_candidates_evaluated_at_solved_point"] = True
            comp["alias_16_candidate_metrics"] = rows
            comp["alias_16_summary"] = alias_summary(rows, MAIN_CANDIDATE_ID)
        return comp

    def analyze_augmented(arms: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        analysis = original_analyze(arms)
        h = analysis.get("headline") or {}
        solver_times: List[Any] = []
        action_times: List[Any] = []
        alias_summaries: List[Mapping[str, Any]] = []
        for arm in arms:
            action_times.append(arm.get("controller_get_action_wall_s"))
            for ev in arm.get("solve_events") or []:
                solver_times.append(ev.get("solver_wall_s"))
            obj = arm.get("objective_reconstruction") or {}
            if obj.get("alias_16_candidates_evaluated_at_solved_point"):
                alias_summaries.append(obj.get("alias_16_summary") or {})
        forced_solve_cells = sum(1 for arm in arms if int(arm.get("solve_calls", 0) or 0) >= 1)
        actual_nonconverged_or_near = int(h.get("forced_nonconverged_or_near_offoptimal_cells", 0) or 0)
        alias_all = bool(
            len(alias_summaries) == len(arms)
            and len(arms) == len(v34n.CELLS)
            and all(int(s.get("candidate_count", 0) or 0) == 16 for s in alias_summaries)
            and all(int(s.get("available_count", 0) or 0) == 16 for s in alias_summaries)
        )
        h.update({
            "timing_solver_wall_s": timing_stats(solver_times),
            "timing_whole_decision_get_action_wall_s": timing_stats(action_times),
            "forced_low_iteration_solve_cells": int(forced_solve_cells),
            "forced_low_iteration_solve_gate_all_scheduled_cells": bool(forced_solve_cells == len(v34n.CELLS)),
            "forced_status_gate_pass_at_least_2_nonconverged_or_near_offoptimal": bool(actual_nonconverged_or_near >= 2),
            "alias_group_candidate_source": candidate_source,
            "alias_group_candidate_count_from_v34k_csv": int(candidate_source.get("candidate_count") or 0),
            "alias_group_solved_point_evaluation_available": alias_all,
            "alias_group_cells_with_16_available_candidates": int(sum(1 for s in alias_summaries if int(s.get("candidate_count", 0) or 0) == 16 and int(s.get("available_count", 0) or 0) == 16)),
            "alias_group_main_ranks_by_cell": [s.get("main_rank_by_relative_error") for s in alias_summaries],
            "alias_group_best_candidate_ids_by_cell": [s.get("best_candidate_id") for s in alias_summaries],
            "alias_group_relative_error_spans_by_cell": [s.get("relative_error_span") for s in alias_summaries],
        })
        # Preserve the original objective-contract pass value but make the
        # low-iteration, actual non-convergence/off-optimal status and alias-report
        # prerequisites explicit in hard_pass; these are pre-run gate constants.
        h["G2_formula_pass_before_low_iteration_status_and_alias_gates"] = bool(h.get("G2_pass"))
        h["G2_pass_formula_low_iteration_status_and_alias_gate"] = bool(
            h.get("G2_formula_pass_before_low_iteration_status_and_alias_gates")
            and h.get("forced_low_iteration_solve_gate_all_scheduled_cells")
            and h.get("forced_status_gate_pass_at_least_2_nonconverged_or_near_offoptimal")
            and h.get("alias_group_solved_point_evaluation_available")
        )
        h["G2_pass"] = h["G2_pass_formula_low_iteration_status_and_alias_gate"]
        analysis["headline"] = h
        return analysis

    def write_outputs_augmented(raw: Mapping[str, Any]) -> None:
        original_write_outputs(raw)
        alias_csv = write_alias_candidate_csv(raw)
        h = (raw.get("analysis") or {}).get("headline") or {}
        summary = v34n.RUN_DIR / "summary.md"
        extra = [
            "",
            "## Alias-group separation across v34k's 16 previously passing candidates",
            "",
            f"- Candidate source: `{(h.get('alias_group_candidate_source') or {}).get('path')}`.",
            f"- Candidate count from v34k passing rows: `{h.get('alias_group_candidate_count_from_v34k_csv')}`.",
            f"- Solved-point evaluation available for all cells: `{h.get('alias_group_solved_point_evaluation_available')}`.",
            f"- Cells with 16/16 available candidates: `{h.get('alias_group_cells_with_16_available_candidates')}` / `{h.get('cells_completed')}`.",
            f"- Main-candidate ranks by cell: `{h.get('alias_group_main_ranks_by_cell')}`.",
            f"- Best candidate IDs by cell: `{h.get('alias_group_best_candidate_ids_by_cell')}`.",
            f"- Relative-error spans by cell: `{h.get('alias_group_relative_error_spans_by_cell')}`.",
            f"- CSV: `{rel(alias_csv)}`.",
            "",
        ]
        with summary.open("a", encoding="utf-8") as f:
            f.write("\n".join(extra) + "\n")

    v34n.formula_try = formula_try_augmented  # type: ignore[assignment]
    v34n.analyze = analyze_augmented  # type: ignore[assignment]
    v34n.write_outputs = write_outputs_augmented  # type: ignore[assignment]


def patch_module_identity_and_contracts() -> None:
    v34n.__file__ = __file__
    v34n.NAME = NAME
    v34n.STAMP = STAMP
    v34n.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
    v34n.STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34t_nonconverged_objective_contract_probe.md"
    v34n.BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34T_NONCONVERGED_OBJECTIVE_CONTRACT_PROBE_{STAMP}.json"
    v34n.REQUEST_ID = f"v34t-a13c3-nonconverged-objective-contract-{STAMP}"
    v34n.MARKER = f"vehicle-v34t-a13c3-nonconverged-objective-contract-{STAMP}"
    v34n.OPUS_REPORT = OPUS_REPORT
    v34n.OPUS_REPORT_SHA = OPUS_REPORT_SHA
    v34n.OPUS_REQUEST = OPUS_REQUEST
    v34n.V34M_DONE = V34M_DONE
    v34n.V34M_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34m_residual_attribution_v0_20260930T111306Z/raw.json"
    v34n._CAPTURES.clear()
    v34n._PATCHED_MODULES.clear()
    v34n._CURRENT_ARM_ID = None
    v34n._TOTAL_SOLVER_CALLS = 0
    v34n.base.load_contexts = v34o.load_contexts_strict
    v34n.base.configure_context_no_reset = v34o.configure_context_no_reset_strict
    v34n.base.extract_goal_xy = v34s.strict_authoritative_goal_xy
    v34n.verify_gates = patched_verify_gates  # type: ignore[assignment]
    install_augmented_formula_analyze_and_outputs()


def run(argv: Optional[Sequence[str]] = None) -> int:
    patch_module_identity_and_contracts()
    return v34n.run(argv)


if __name__ == "__main__":
    raise SystemExit(run())
