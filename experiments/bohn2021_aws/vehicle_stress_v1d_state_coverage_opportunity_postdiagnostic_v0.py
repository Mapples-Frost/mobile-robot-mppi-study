#!/usr/bin/env python3
"""Vehicle stress-v1d state-coverage/opportunity postdiagnostic v0.

Development-only, no-simulation diagnostic.  This script reads already-opened
stress-scenario development artifacts (not validation64 and not sealed test) to
separate three explanations for the failed v1d terminal-stable label gate:

1. the source-supported straight-line vehicle family may have genuinely sparse
   state-dependent horizon opportunity;
2. the v1d trace-selected target sampler may have missed the high-leverage
   states/cases indicated by the episode-level stress-v1 map;
3. previous matched-continuation evidence may have been blocked by a prefix
   artifact rather than by lack of safe non-H15 opportunities.

No controller is trained/refit, no rollout is executed, and no validation/test
bank is opened.  Outputs are diagnostic only and are not formal validation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_stress_v1d_state_coverage_opportunity_postdiagnostic_v0"
STAMP = "20260929T0435Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_{NAME.upper()}_{STAMP}.json"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

STAGE1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/raw.json"
STAGE1_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/summary.md"
STAGE2_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/raw.json"
STAGE2_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/summary.md"
V1D_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/raw.json"
V1D_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/summary.md"
V1D_SELECTED_TARGETS = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/selected_targets.json"
V1D_PARETO_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_pareto_timing_objective_postdiagnostic_v0_20260929T0325Z/raw.json"
OBJ_REPAIR_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_objective_terminal_repair_feasibility_v0_20260929T0415Z/completed.json"
OBJ_REPAIR_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_objective_terminal_repair_feasibility_v0_20260929T0415Z/summary.md"
RANK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_terminal_value_rank_predictivity_postdiagnostic_v0_20260929T0425Z/completed.json"

MATERIAL_GAIN = 3.0
STATE_DISTANCE_TOL = 1e-5


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def clean_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_jsonable(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def unique_sorted(values: Iterable[Any]) -> List[Any]:
    return sorted(set(values))


def count_by(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        key = str(v)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def require_inputs() -> Dict[str, str]:
    required = [
        STAGE1_RAW,
        STAGE1_SUMMARY,
        STAGE2_RAW,
        STAGE2_SUMMARY,
        V1D_RAW,
        V1D_SUMMARY,
        V1D_PARETO_RAW,
        OBJ_REPAIR_DONE,
        OBJ_REPAIR_SUMMARY,
    ]
    optional = [V1D_SELECTED_TARGETS, RANK_DONE]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("required diagnostic inputs are missing: " + ", ".join(missing))
    hashes = {rel(p): sha256(p) for p in required + [p for p in optional if p.exists()]}
    return hashes


def stage1_case_tables(stage1: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    rows = [dict(r) for r in (stage1.get("postdiagnostic_case_table") or [])]
    runner_material = [safe_int(c) for c in (((stage1.get("predeclared_stage1_gate") or {}).get("material_case_ids_from_runner")) or [])]
    phys_ge3_cases = unique_sorted(
        safe_int(r.get("case"))
        for r in (stage1.get("strict_material_vs_H15_rows") or [])
        if safe_float(r.get("physical_gain_vs_H15"), -1e9) >= MATERIAL_GAIN
        and r.get("success") is True
        and r.get("constraint") is False
        and safe_int(r.get("solver_failure_steps"), 999) == 0
    )
    return rows, runner_material, phys_ge3_cases


def v1d_target_rows(v1d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    analysis = v1d.get("analysis") or {}
    rows = []
    for row in analysis.get("state_rows") or []:
        rows.append({
            "state_id": row.get("state_id"),
            "case": safe_int(row.get("case")),
            "group": row.get("selection_group"),
            "role": row.get("target_role"),
            "branch_step": safe_int(row.get("branch_step")),
            "label": row.get("label"),
            "best_any_horizon": safe_int(((row.get("best_by_physical_any_mode") or {}).get("horizon"))),
            "best_any_gain_vs_H15_physical": safe_float(((row.get("best_by_physical_any_mode") or {}).get("gain_vs_H15_physical"))),
            "best_any_terminal_mode": (row.get("best_by_physical_any_mode") or {}).get("terminal_mode"),
            "is_control_state": bool(row.get("is_control_state")),
        })
    return rows


def v1d_selected_payload() -> Dict[str, Any]:
    if not V1D_SELECTED_TARGETS.exists():
        return {"available": False}
    try:
        payload = read_json(V1D_SELECTED_TARGETS)
    except Exception as exc:  # keep diagnostic robust; the raw state rows are enough
        return {"available": False, "read_error": repr(exc)}
    summary: Dict[str, Any] = {"available": True, "top_level_keys": sorted(str(k) for k in payload.keys()) if isinstance(payload, Mapping) else []}
    if isinstance(payload, Mapping):
        for key in ("target_selection_fallbacks", "fallbacks", "selected_group_counts", "candidate_group_counts", "selected_case_counts"):
            if key in payload:
                summary[key] = payload.get(key)
        for key in ("selected_targets", "targets", "high_targets", "control_targets"):
            val = payload.get(key)
            if isinstance(val, list):
                summary[f"{key}_count"] = len(val)
                summary[f"{key}_case_counts"] = count_by(safe_int(x.get("full_case_index", x.get("case", -1))) for x in val if isinstance(x, Mapping))
    return summary


def stage2_prefix_blocked_candidates(stage2: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    blocked: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    for row in ((stage2.get("analysis") or {}).get("state_rows") or []):
        case = safe_int(row.get("case"))
        branch_step = safe_int(row.get("branch_step"))
        role = row.get("selection_role")
        for comp in row.get("comparisons") or []:
            h = safe_int(comp.get("horizon"))
            if h == 15:
                continue
            gain = safe_float(comp.get("gain_vs_H15_physical"), -1e9)
            safe = bool(comp.get("branch_reached")) and bool(comp.get("success")) and comp.get("constraint") is False and bool(comp.get("no_success_constraint_solver_regression_vs_H15"))
            close = safe_float(comp.get("state_distance_vs_H15_branch_state"), 1e9) <= STATE_DISTANCE_TOL
            prefix_ok = bool(comp.get("prefix_clean_sha256_matches_H15"))
            if gain >= MATERIAL_GAIN and safe and close:
                item = {
                    "case": case,
                    "branch_step": branch_step,
                    "role": role,
                    "horizon": h,
                    "gain_vs_H15_physical": gain,
                    "gain_vs_H15_total": safe_float(comp.get("gain_vs_H15_total")),
                    "continuation_physical": safe_float(comp.get("continuation_physical")),
                    "continuation_total": safe_float(comp.get("continuation_total")),
                    "decision_sum_s": safe_float(comp.get("decision_sum_s")),
                    "prefix_clean_sha256_matches_H15": prefix_ok,
                    "state_distance_vs_H15_branch_state": safe_float(comp.get("state_distance_vs_H15_branch_state")),
                    "path": comp.get("path"),
                }
                if prefix_ok:
                    accepted.append(item)
                else:
                    blocked.append(item)
    blocked.sort(key=lambda x: (-safe_float(x.get("gain_vs_H15_physical")), safe_int(x.get("case")), safe_int(x.get("branch_step")), safe_int(x.get("horizon"))))
    accepted.sort(key=lambda x: (-safe_float(x.get("gain_vs_H15_physical")), safe_int(x.get("case")), safe_int(x.get("branch_step")), safe_int(x.get("horizon"))))
    return blocked, accepted


def pareto_case_rows(pareto: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for row in pareto.get("per_state") or []:
        best_strict = row.get("best_strict_compute_pareto")
        best_relaxed = row.get("best_relaxed_compute_pareto")
        rows.append({
            "case": safe_int(row.get("case")),
            "branch_step": safe_int(row.get("branch_step")),
            "state_id": row.get("state_id"),
            "strict_horizon": None if not isinstance(best_strict, Mapping) else safe_int(best_strict.get("horizon")),
            "strict_min_decision_time_gain_s": None if not isinstance(best_strict, Mapping) else safe_float(best_strict.get("min_decision_time_gain_s")),
            "strict_min_physical_gain": None if not isinstance(best_strict, Mapping) else safe_float(best_strict.get("min_physical_gain")),
            "relaxed_horizon": None if not isinstance(best_relaxed, Mapping) else safe_int(best_relaxed.get("horizon")),
            "relaxed_min_decision_time_gain_s": None if not isinstance(best_relaxed, Mapping) else safe_float(best_relaxed.get("min_decision_time_gain_s")),
            "relaxed_min_physical_gain": None if not isinstance(best_relaxed, Mapping) else safe_float(best_relaxed.get("min_physical_gain")),
        })
    return rows


def summarize_case_coverage(stage1_rows: Sequence[Mapping[str, Any]], runner_material: Sequence[int], phys_ge3_cases: Sequence[int], v1d_rows: Sequence[Mapping[str, Any]], blocked: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_case_v1d: Dict[int, List[Mapping[str, Any]]] = {}
    for r in v1d_rows:
        by_case_v1d.setdefault(safe_int(r.get("case")), []).append(r)
    by_case_blocked: Dict[int, List[Mapping[str, Any]]] = {}
    for r in blocked:
        by_case_blocked.setdefault(safe_int(r.get("case")), []).append(r)
    case_rows: List[Dict[str, Any]] = []
    for row in stage1_rows:
        case = safe_int(row.get("case"))
        selected = by_case_v1d.get(case, [])
        blocked_here = by_case_blocked.get(case, [])
        case_rows.append({
            "case": case,
            "group": row.get("group"),
            "source_candidate_index": row.get("source_candidate_index"),
            "theta_r": row.get("theta_r"),
            "traj_steps": row.get("traj_steps"),
            "runner_episode_material": case in set(runner_material),
            "episode_physical_gain_ge_3_case": case in set(phys_ge3_cases),
            "material_strict_horizons_vs_H15": row.get("material_strict_horizons_vs_H15") or [],
            "best_strict_physical_horizon": row.get("best_strict_physical_horizon"),
            "fastest_strict_horizon": row.get("fastest_strict_horizon"),
            "v1d_selected_target_count": len(selected),
            "v1d_selected_branch_steps": [safe_int(x.get("branch_step")) for x in selected],
            "v1d_best_any_gains": [safe_float(x.get("best_any_gain_vs_H15_physical")) for x in selected],
            "stage2_prefix_blocked_material_count": len(blocked_here),
            "stage2_prefix_blocked_branch_steps": unique_sorted(safe_int(x.get("branch_step")) for x in blocked_here),
            "stage2_prefix_blocked_horizons": unique_sorted(safe_int(x.get("horizon")) for x in blocked_here),
            "stage2_prefix_blocked_max_gain": max([safe_float(x.get("gain_vs_H15_physical")) for x in blocked_here], default=0.0),
        })
    return case_rows


def append_docs(block: str, marker: str) -> None:
    for fname in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / fname
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if marker not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def make_markdown(result: Mapping[str, Any]) -> str:
    headline = result["headline"]
    lines: List[str] = []
    lines.append("# Vehicle stress-v1d state-coverage/opportunity postdiagnostic v0")
    lines.append("")
    lines.append(f"UTC: `{result['created_utc']}`. Development-only no-simulation diagnostic; validation64 and sealed test stayed closed.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Classification: `{headline['classification']}`.")
    lines.append(f"- Stage1 runner-material episode cases: `{headline['stage1_runner_material_cases']}`; v1d selected coverage: `{headline['covered_runner_material_count']}/{headline['runner_material_count']}`.")
    lines.append(f"- Stage1 physical-gain>=3 episode cases: `{headline['stage1_physical_ge3_cases']}`; v1d selected coverage: `{headline['covered_physical_ge3_count']}/{headline['physical_ge3_count']}`.")
    lines.append(f"- v1d terminal-stable robust positives remain: `{headline['v1d_robust_positive_state_count']}` / `{headline['v1d_state_count']}`.")
    lines.append(f"- Stage2 prefix-blocked but state-close material candidates: `{headline['stage2_prefix_blocked_material_candidate_count']}` across cases `{headline['stage2_prefix_blocked_material_cases']}`; accepted prefix-clean material candidates: `{headline['stage2_prefix_clean_material_candidate_count']}`.")
    lines.append(f"- Train/refit now: `{headline['train_or_refit_now']}`.")
    lines.append(f"- Recommended next action: `{headline['next_action']}`.")
    lines.append("")
    lines.append("## Case-level coverage table")
    lines.append("")
    lines.append("| case | group | runner material | phys>=3 case | strict material H | v1d targets | v1d branch steps | v1d gains | stage2 blocked material branches/H/maxgain |")
    lines.append("|---:|---|---:|---:|---|---:|---|---|---|")
    for row in result["case_coverage_rows"]:
        if row["runner_episode_material"] or row["episode_physical_gain_ge_3_case"] or row["v1d_selected_target_count"] or row["stage2_prefix_blocked_material_count"]:
            lines.append(
                "| {case} | `{group}` | {runner} | {phys} | `{hs}` | {n} | `{steps}` | `{gains}` | `{branches}` / `{horizons}` / {maxgain:.4g} |".format(
                    case=row["case"],
                    group=row.get("group"),
                    runner=int(bool(row.get("runner_episode_material"))),
                    phys=int(bool(row.get("episode_physical_gain_ge_3_case"))),
                    hs=row.get("material_strict_horizons_vs_H15"),
                    n=row.get("v1d_selected_target_count"),
                    steps=row.get("v1d_selected_branch_steps"),
                    gains=[round(float(x), 6) for x in row.get("v1d_best_any_gains", [])],
                    branches=row.get("stage2_prefix_blocked_branch_steps"),
                    horizons=row.get("stage2_prefix_blocked_horizons"),
                    maxgain=float(row.get("stage2_prefix_blocked_max_gain", 0.0)),
                )
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- SCENARIOS: opportunity in the current source-supported vehicle family is still sparse and concentrated, but absence of adaptive opportunity is not established because v1d selected no states from the two Stage1 cases with episode-level physical gain >= 3.")
    lines.append("- REWARD/TERMINAL/VALUE: objective/terminal repair and raw objective imitation remain unsupported; terminal-stable labels are still required before any selector target is trusted.")
    lines.append("- TRAINING: the failed v1d label gate should not trigger selector/value training; it currently diagnoses state-selection coverage rather than a trainable label bank.")
    lines.append("- COMPARISONS: shorter H or oracle branch labels do not prove speed; any follow-up must keep physical cost, success/safety and repeated measured decision time separate against fixed-H/Pareto baselines.")
    lines.append("")
    lines.append("## Next discriminating intervention")
    lines.append("")
    lines.append("Freeze a compact IMPROVED v1e targeted common-prefix continuation smoke (development only, no validation64/test): rerun common H15-prefix branch comparisons for the missed episode-positive cases, especially Stage2's prefix-blocked case5 branch-step 18 candidate, plus case1/case4 high-trace states and negative/control states.  Acceptance should require >=2 terminal-stable material positive states from at least two cases, retained negatives/controls, and no prefix/state/safety artifacts before any selector/value refit.  If it fails, pivot toward versioned scenario/opportunity redesign or richer dynamics/terminal-value modeling rather than another unchanged label-density sweep.")
    lines.append("")
    lines.append(f"Backup request: `{result['backup_request']}`.")
    return "\n".join(lines) + "\n"


def run() -> int:
    if OUT.exists() and (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"partial output exists; inspect before rerun: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=True)
    input_hashes = require_inputs()
    created = now_utc()

    stage1 = read_json(STAGE1_RAW)
    stage2 = read_json(STAGE2_RAW)
    v1d = read_json(V1D_RAW)
    pareto = read_json(V1D_PARETO_RAW)
    obj_repair = read_json(OBJ_REPAIR_DONE)

    stage1_rows, runner_material, phys_ge3_cases = stage1_case_tables(stage1)
    v1d_rows = v1d_target_rows(v1d)
    selected_payload = v1d_selected_payload()
    blocked, accepted = stage2_prefix_blocked_candidates(stage2)
    pareto_rows = pareto_case_rows(pareto)
    case_coverage = summarize_case_coverage(stage1_rows, runner_material, phys_ge3_cases, v1d_rows, blocked)

    v1d_cases = unique_sorted(safe_int(r.get("case")) for r in v1d_rows)
    covered_runner = sorted(set(runner_material).intersection(v1d_cases))
    missed_runner = sorted(set(runner_material).difference(v1d_cases))
    covered_phys = sorted(set(phys_ge3_cases).intersection(v1d_cases))
    missed_phys = sorted(set(phys_ge3_cases).difference(v1d_cases))
    blocked_cases = unique_sorted(safe_int(r.get("case")) for r in blocked)

    analysis = v1d.get("analysis") or {}
    obj_headline = obj_repair.get("headline") or {}

    coverage_defect_supported = (len(phys_ge3_cases) > 0 and len(covered_phys) < len(phys_ge3_cases)) or (len(runner_material) > 0 and len(covered_runner) <= max(1, len(runner_material) // 2))
    prefix_artifact_supported = len(blocked) > 0 and len(accepted) == 0
    if coverage_defect_supported and prefix_artifact_supported:
        classification = "target_state_coverage_failure_with_prefix_blocked_high_leverage_candidates"
        next_action = "freeze targeted v1e common-prefix continuation smoke before scenario redesign or selector/value refit"
    elif coverage_defect_supported:
        classification = "target_state_coverage_failure_likely"
        next_action = "freeze targeted common-prefix smoke for missed episode-positive cases before selector/value refit"
    else:
        classification = "sparse_opportunity_more_likely_than_v1d_target_miss"
        next_action = "pivot to versioned scenario/opportunity redesign or richer terminal-value representation diagnostic"

    next_protocol_sketch = {
        "protocol_label": "vehicle_stress_v1e_episode_positive_targeted_common_prefix_smoke_v0_to_freeze_next",
        "classification": "IMPROVED development diagnostic, not validation and not final test",
        "do_not_do": [
            "no selector/refit/training before terminal-stable positives exist",
            "no validation64 or sealed-test access",
            "no repeated 186-episode timing confirmation under failed strict gate",
            "no speed claim from shorter H or single-run branch timing",
        ],
        "target_sources": {
            "mandatory_prefix_blocked_replay_candidates_top": blocked[:5],
            "missed_stage1_runner_material_cases": missed_runner,
            "missed_stage1_physical_ge3_cases": missed_phys,
            "retain_negative_or_control_cases_examples": [c for c in v1d_cases if c not in set(runner_material)][:4],
        },
        "candidate_budget_bound": {
            "target_states_suggested": "6-8, selected before rollouts from existing H15 traces/stage2 candidate list",
            "branch_horizons_suggested": [10, 15, 20, 25, 30, 45, 50],
            "terminal_modes_suggested": ["zero_terminal", "h15_common_terminal"],
            "max_new_branch_episodes_if_8_states": 8 * 7 * 2,
            "max_control_steps_if_150_step_cap": 8 * 7 * 2 * 150,
        },
        "acceptance_before_refit": {
            "terminal_stable_material_positive_states_min": 2,
            "distinct_positive_cases_min": 2,
            "must_include_one_of_cases": sorted(set([4, 5]).intersection(set(phys_ge3_cases)) or {5}),
            "retained_negative_or_control_states_min": 2,
            "blocking_prefix_state_safety_artifacts_allowed": 0,
        },
    }

    headline = {
        "classification": classification,
        "stage1_runner_material_cases": runner_material,
        "runner_material_count": len(runner_material),
        "covered_runner_material_cases": covered_runner,
        "covered_runner_material_count": len(covered_runner),
        "missed_runner_material_cases": missed_runner,
        "stage1_physical_ge3_cases": phys_ge3_cases,
        "physical_ge3_count": len(phys_ge3_cases),
        "covered_physical_ge3_cases": covered_phys,
        "covered_physical_ge3_count": len(covered_phys),
        "missed_physical_ge3_cases": missed_phys,
        "v1d_selected_cases": v1d_cases,
        "v1d_selected_case_counts": count_by(r.get("case") for r in v1d_rows),
        "v1d_selected_group_counts": count_by(r.get("group") for r in v1d_rows),
        "v1d_robust_positive_state_count": safe_int(analysis.get("robust_positive_state_count"), 0),
        "v1d_state_count": safe_int(analysis.get("state_count"), len(v1d_rows)),
        "stage2_prefix_blocked_material_candidate_count": len(blocked),
        "stage2_prefix_blocked_material_cases": blocked_cases,
        "stage2_prefix_clean_material_candidate_count": len(accepted),
        "objective_repair_classification": obj_headline.get("classification"),
        "train_or_refit_now": False,
        "next_action": next_action,
    }

    result = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_no_simulation_state_coverage_opportunity_postdiagnostic_no_validation64_no_test",
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "candidate_pool_resets": 0,
        "input_hashes": input_hashes,
        "headline": headline,
        "case_coverage_rows": case_coverage,
        "stage2_prefix_blocked_material_candidates": blocked,
        "stage2_prefix_clean_material_candidates": accepted,
        "v1d_target_rows": v1d_rows,
        "v1d_selected_targets_payload_summary": selected_payload,
        "v1d_pareto_rows": pareto_rows,
        "next_protocol_sketch": next_protocol_sketch,
        "decision_rules": {
            "do_not_train_or_refit_current_labels": True,
            "do_not_claim_absence_from_v1d_zero_labels_alone": coverage_defect_supported,
            "do_not_use_stage2_prefix_blocked_candidates_as_positive_labels": True,
            "targeted_common_prefix_smoke_has_higher_information_value_than_unchanged_label_density_sweep": classification != "sparse_opportunity_more_likely_than_v1d_target_miss",
        },
        "backup_request": rel(BACKUP_REQ),
    }

    write_json(OUT / "raw.json", result)
    summary = make_markdown(result)
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    STATE.write_text(summary, encoding="utf-8")

    marker = f"{NAME}-{STAMP}"
    append_docs("\n".join([
        f"## {marker}",
        "",
        summary.strip(),
    ]), marker)

    write_json(BACKUP_REQ, {
        "request": "backup_after_vehicle_stress_v1d_state_coverage_opportunity_postdiagnostic_v0",
        "created_utc": created.isoformat(),
        "reason": "Preserve no-simulation state-coverage/opportunity diagnostic and next-action decision before any v1e targeted common-prefix rollout.",
        "artifacts": [rel(OUT / "raw.json"), rel(OUT / "summary.md"), rel(STATE)],
    })
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "headline": headline,
        "artifacts": {
            "summary": rel(OUT / "summary.md"),
            "raw": rel(OUT / "raw.json"),
            "state": rel(STATE),
            "backup_request": rel(BACKUP_REQ),
        },
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "classification": classification,
        "missed_physical_ge3_cases": missed_phys,
        "stage2_prefix_blocked_material_candidate_count": len(blocked),
        "train_or_refit_now": False,
        "next_action": next_action,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(BACKUP_REQ),
    }, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-accept-development-no-simulation", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_no_simulation:
        raise RuntimeError("explicit --i-accept-development-no-simulation required")
    try:
        return run()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {
            "passed": False,
            "created_utc": now_utc().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
