#!/usr/bin/env python3
"""v34t / T-B3 A13c-3 active-plan non-converged objective-contract probe.

Prepared after v34s/T-B4 passed.  This wrapper preserves the v34n bounded
low-iteration solver-contract design while binding the two operational repairs
required by the current Opus plan 20260930T115516Z_8330f5:

* v34o strict previous_input scalarization and fail-loud _u0 setup;
* v34s/v34q authoritative saved trajectory endpoint goal extraction, with
  endpoint index == round(reference.traj_steps)-1.

It also augments the raw analysis with timing summaries and an alias-group pass
through the candidates recorded by the v34k label-accessor localization CSV.
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
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T115516Z_8330f5.md"
OPUS_REPORT_SHA = "3acada362e46ec94b6a21c9738e076a2c95ca8db7be95369a560969d3f7eb474"
OPUS_REQUEST = "execution-result:20260930T115434_d4836078"
V34M_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34m_residual_attribution_v0_20260930T111306Z/completed.json"
V34N_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34n_nonconverged_objective_contract_probe_v0_20260930T112938Z/failed.json"
V34K_GLOB = str(ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_*/candidate_residuals.csv")


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


def patched_verify_gates(args: argparse.Namespace) -> Dict[str, Any]:
    for p in [v34n.PLAN_READY, OPUS_REPORT, V34M_DONE, V34N_FAILED]:
        if not p.exists():
            raise ContractError("missing prerequisite " + rel(p))
    ready = read_json(v34n.PLAN_READY)
    if ready.get("request_id") != OPUS_REQUEST or ready.get("report_sha256") != OPUS_REPORT_SHA:
        raise ContractError(f"active PLAN_READY mismatch: request={ready.get('request_id')} sha={ready.get('report_sha256')}")
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
    return {
        "active_lead_plan_ready": rel(v34n.PLAN_READY),
        "active_lead_report": rel(OPUS_REPORT),
        "active_lead_report_sha256": OPUS_REPORT_SHA,
        "active_lead_request": OPUS_REQUEST,
        "v34m_completed": rel(V34M_DONE),
        "v34n_failed_preserved": rel(V34N_FAILED),
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


def latest_v34k_candidates() -> Dict[str, Any]:
    paths = [Path(p) for p in sorted(glob.glob(V34K_GLOB))]
    out: Dict[str, Any] = {"path": None, "sha256": None, "candidate_count": 0, "candidates": []}
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
            cid = row.get("candidate_id") or row.get("formula_id") or ""
            parsed = parse_candidate_id(cid)
            if parsed is None:
                continue
            key = parsed["candidate_id"]
            if key in seen:
                continue
            seen.add(key)
            candidates.append({**parsed, "v34k_row": {k: row.get(k) for k in row.keys() if k in {"candidate_id", "relative_error", "abs_error", "hard_pass", "pass", "passed", "total", "solver_objective"}}})
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
                "abs_error": comp.get("abs_error"),
                "stage_lterm_total": comp.get("stage_lterm_total"),
                "terminal_discounted": comp.get("terminal_discounted"),
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
        "near_tie_candidate_ids_relerr_le_max_1e_minus_6_or_main_plus_1e_minus_12": near_ties,
    }


def timing_stats(values: Iterable[Any]) -> Dict[str, Any]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
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


def install_augmented_solve_cell_and_analyze() -> None:
    original_solve_cell = v34n.solve_cell
    original_analyze = v34n.analyze
    candidate_source = latest_v34k_candidates()

    def solve_cell_augmented(row: Mapping[str, Any], context: Mapping[str, Any], terminals: Mapping[int, Any], idx: int) -> Dict[str, Any]:
        arm = original_solve_cell(row, context, terminals, idx)
        try:
            mpc = None
            # original_solve_cell does not return mpc. Re-evaluation of all aliases
            # at solved-point level is therefore only possible inside that function.
            # Keep the v34k candidate list and the original four-branch variants in
            # raw evidence; the direct solver/objective contract remains evaluated
            # by the frozen main formula. This explicit unavailable marker prevents
            # accidental overclaiming and tells the next repair exactly what to move
            # inside solve_cell if the lead insists on per-candidate solved-point CSV.
            del mpc
            arm["alias_16_candidate_source"] = candidate_source
            arm["alias_16_candidates_evaluated_at_solved_point"] = False
            arm["alias_16_unavailable_reason"] = "v34t thin wrapper cannot access mpc after v34n.solve_cell returns; v34n raw still includes main formula plus eps/r/both branch variants. A future source-level copy can move alias evaluation inside solve_cell if required."
        except Exception as exc:
            arm["alias_16_candidate_source"] = candidate_source
            arm["alias_16_candidates_evaluated_at_solved_point"] = False
            arm["alias_16_unavailable_reason"] = repr(exc)
        return arm

    def analyze_augmented(arms: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        analysis = original_analyze(arms)
        h = analysis.get("headline") or {}
        solver_times: List[Any] = []
        action_times: List[Any] = []
        for arm in arms:
            action_times.append(arm.get("controller_get_action_wall_s"))
            for ev in arm.get("solve_events") or []:
                solver_times.append(ev.get("solver_wall_s"))
        forced_low_iteration_cells = sum(1 for arm in arms if int(arm.get("solve_calls", 0) or 0) >= 1)
        h.update({
            "timing_solver_wall_s": timing_stats(solver_times),
            "timing_whole_decision_get_action_wall_s": timing_stats(action_times),
            "forced_low_iteration_solve_cells": int(forced_low_iteration_cells),
            "forced_low_iteration_gate_pass_at_least_2_cells": bool(forced_low_iteration_cells >= 2),
            "alias_group_candidate_source": candidate_source,
            "alias_group_candidate_count_from_v34k_csv": int(candidate_source.get("candidate_count") or 0),
            "alias_group_solved_point_evaluation_available": False,
        })
        # Preserve the original G2 formula pass value but make the low-iteration
        # prerequisite explicit in hard_pass; this is a pre-run gate constant.
        h["G2_pass_formula_and_low_iteration_gate"] = bool(h.get("G2_pass") and h.get("forced_low_iteration_gate_pass_at_least_2_cells"))
        h["G2_pass"] = h["G2_pass_formula_and_low_iteration_gate"]
        analysis["headline"] = h
        return analysis

    v34n.solve_cell = solve_cell_augmented  # type: ignore[assignment]
    v34n.analyze = analyze_augmented  # type: ignore[assignment]


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
    install_augmented_solve_cell_and_analyze()


def run(argv: Optional[Sequence[str]] = None) -> int:
    patch_module_identity_and_contracts()
    return v34n.run(argv)


if __name__ == "__main__":
    raise SystemExit(run())
