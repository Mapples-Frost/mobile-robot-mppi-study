#!/usr/bin/env python3
"""Vehicle stress-v1e targeted common-prefix target preparation v0.

Development-only no-simulation diagnostic.  The preceding cross-bank identity
audit showed that older Stage2-v0b prefix-blocked candidates were not the same
stress-v1 cases that v1d missed.  This script therefore freezes a stress-v1-only
v1e target list/protocol from existing stress-v1 Stage1 H15 traces, before any
new non-H15 branch rollout.

No simulations, no candidate resets, no training/refit, no validation64 bank,
and no sealed-test access occur in this prepare step.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_stress_v1e_targeted_common_prefix_prepare_v0"
STAMP = "20260929T0505Z"
SOURCE = Path(__file__).resolve()
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
PROTOCOL_JSON = ROOT / f"research_artifacts/aws_protocols/{NAME}_frozen_{STAMP}.json"
PROTOCOL_MD = ROOT / f"research_artifacts/aws_protocols/{NAME}_frozen_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1E_TARGETED_COMMON_PREFIX_SMOKE_{STAMP}.json"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

STAGE1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/raw.json"
STAGE1_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/completed.json"
STAGE1_POST_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/raw.json"
STAGE1_POST_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/completed.json"
COVERAGE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_state_coverage_opportunity_postdiagnostic_v0_20260929T0435Z/raw.json"
COVERAGE_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_state_coverage_opportunity_postdiagnostic_v0_20260929T0435Z/completed.json"
CROSSBANK_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_stage2_crossbank_identity_audit_v0_20260929T0445Z/raw.json"
CROSSBANK_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_stage2_crossbank_identity_audit_v0_20260929T0445Z/completed.json"
STAGE1_BANK_PATH = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/bank/vehicle_stress_scenario_opportunity_probe_v1_bank.json"

PREFIX_H = 15
BRANCH_HORIZONS = [10, 15, 20, 25, 30, 45, 50]
TERMINAL_MODES = ["zero_terminal", "h15_common_terminal"]
MAX_STEPS = 150
ORDER_SEED = 2609295101
STATE_DISTANCE_TOL = 1e-5
MATERIAL_GAIN = 3.0
TARGET_COUNT = 12


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_jsonable(v) for v in value]
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


def safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"completed marker did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is not False:
        raise ContractError(f"sealed-test flag not false in {rel(path)}")
    if obj.get("historical_validation64_bank_opened") not in (False, None):
        raise ContractError(f"historical validation64 flag not false in {rel(path)}")
    return obj


def require_inputs() -> Dict[str, str]:
    required = [STAGE1_RAW, STAGE1_COMPLETED, STAGE1_POST_RAW, STAGE1_POST_COMPLETED, COVERAGE_RAW, COVERAGE_COMPLETED, CROSSBANK_RAW, CROSSBANK_COMPLETED]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        raise ContractError("missing required inputs: " + ", ".join(missing))
    for p in [STAGE1_COMPLETED, STAGE1_POST_COMPLETED, COVERAGE_COMPLETED, CROSSBANK_COMPLETED]:
        completed_ok(p)
    hashes = {rel(p): sha256(p) for p in required + [SOURCE]}
    if STAGE1_BANK_PATH.exists():
        # Do not parse the 100MB+ bank in this no-simulation prepare step; the
        # hash preserves provenance for the later rollout script.
        hashes[rel(STAGE1_BANK_PATH)] = sha256(STAGE1_BANK_PATH)
    return hashes


def get_selection_meta(stage1_raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    selection = stage1_raw.get("bank_selection") or {}
    metas = selection.get("selected_metadata") or []
    if len(metas) < 20:
        raise ContractError("stress-v1 Stage1 raw lacks 20 selected metadata rows")
    return metas


def trace_path_for(stage1_raw: Mapping[str, Any], case: int, horizon: int = PREFIX_H) -> Path:
    rows = [e for e in stage1_raw.get("episodes", []) if safe_int(e.get("case")) == case and safe_int(e.get("horizon")) == horizon]
    if len(rows) != 1:
        raise ContractError(f"expected exactly one Stage1 episode for case={case} H{horizon}; got {len(rows)}")
    path = ROOT / str(rows[0].get("path")) / "trace.json"
    if not path.exists():
        raise ContractError(f"missing H{horizon} trace for case {case}: {rel(path)}")
    return path


def observation_component(row: Mapping[str, Any], idx: int) -> float:
    obs = row.get("observation") or []
    try:
        return safe_float(obs[idx], 0.0)
    except Exception:
        return 0.0


def input_omega(row: Mapping[str, Any]) -> float:
    val = (row.get("input") or {}).get("u_omega") if isinstance(row.get("input"), Mapping) else None
    if isinstance(val, list) and val:
        return safe_float(val[0], 0.0)
    return safe_float(val, 0.0)


def solver_iterations(row: Mapping[str, Any]) -> float:
    attempts = ((row.get("recovery") or {}).get("attempts") or [])
    if not attempts:
        return 0.0
    return max(safe_float(a.get("iterations"), 0.0) for a in attempts)


def row_score(row: Mapping[str, Any], step: int, trace_len: int) -> Tuple[float, Dict[str, float]]:
    perf = max(0.0, safe_float(row.get("performance"), 0.0))
    constraint = max(0.0, safe_float(row.get("constraint"), 0.0))
    heading = abs(observation_component(row, 2))
    omega = abs(input_omega(row))
    iters = solver_iterations(row)
    phase = float(step) / float(max(1, trace_len - 1))
    central = 1.0 - min(1.0, abs(phase - 0.5) / 0.5)
    score = 10.0 * math.sqrt(perf) + 2.0 * heading + 0.5 * omega + 0.02 * iters + 0.5 * min(10.0, constraint) + 0.25 * central
    return float(score), {
        "performance_step_cost": float(perf),
        "sqrt_performance": float(math.sqrt(perf)),
        "constraint_step_cost_capped": float(min(10.0, constraint)),
        "abs_heading_proxy_obs2": float(heading),
        "abs_turn_effort_u_omega": float(omega),
        "solver_iteration_proxy": float(iters),
        "phase_fraction": float(phase),
        "centrality": float(central),
    }


def bounds_for_window(trace_len: int, low_frac: float, high_frac: float) -> Tuple[int, int]:
    lo = max(8, int(math.floor(low_frac * trace_len)))
    hi = min(max(8, trace_len - 3), 90, int(math.ceil(high_frac * trace_len)))
    if hi < lo:
        lo = max(1, min(hi, trace_len - 3))
    return lo, hi


def choose_window(trace: Sequence[Mapping[str, Any]], case: int, role: str, window_name: str, low_frac: float, high_frac: float, used_steps: set) -> Dict[str, Any]:
    if len(trace) < 12:
        raise ContractError(f"H15 trace too short for case {case}: {len(trace)}")
    lo, hi = bounds_for_window(len(trace), low_frac, high_frac)
    candidates = []
    for step in range(lo, hi + 1):
        if step in used_steps:
            continue
        row = trace[step]
        score, comps = row_score(row, step, len(trace))
        candidates.append((score, comps["performance_step_cost"], -abs(comps["phase_fraction"] - 0.5), -step, step, comps))
    if not candidates:
        for step in range(8, min(len(trace) - 3, 90) + 1):
            if step in used_steps:
                continue
            row = trace[step]
            score, comps = row_score(row, step, len(trace))
            candidates.append((score, comps["performance_step_cost"], -abs(comps["phase_fraction"] - 0.5), -step, step, comps))
    if not candidates:
        raise ContractError(f"no available H15 trace step for case {case} role {role}")
    _, _, _, _, step, comps = max(candidates)
    used_steps.add(step)
    row = trace[step]
    return {
        "case": int(case),
        "case_role": role,
        "window": window_name,
        "branch_step": int(step),
        "h15_trace_length": int(len(trace)),
        "h15_selection_score": float(row_score(row, step, len(trace))[0]),
        "h15_selection_score_components": comps,
        "h15_reference_previous_state": row.get("previous_state"),
        "h15_reference_state_after_step": row.get("state"),
        "h15_reference_observation": row.get("observation"),
        "h15_reference_step_costs": {
            "performance": row.get("performance"),
            "constraint": row.get("constraint"),
            "compute": row.get("compute"),
            "reward": row.get("reward"),
            "solver_success": row.get("solver_success"),
            "termination": row.get("termination"),
        },
    }


def coverage_case_rows(coverage_raw: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for row in coverage_raw.get("case_coverage_rows") or []:
        case = safe_int(row.get("case"))
        if case >= 0:
            out[case] = row
    return out


def stage1_post_case_rows(post_raw: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    return {safe_int(row.get("case")): row for row in (post_raw.get("postdiagnostic_case_table") or []) if safe_int(row.get("case")) >= 0}


def derive_case_plan(coverage_raw: Mapping[str, Any], post_raw: Mapping[str, Any], metas: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cov = coverage_case_rows(coverage_raw)
    post_cases = stage1_post_case_rows(post_raw)
    physical_cases = [c for c, row in sorted(cov.items()) if bool(row.get("episode_physical_gain_ge_3_case"))]
    if not physical_cases:
        physical_cases = [c for c, row in sorted(post_cases.items()) if row.get("material_strict_horizons_vs_H15") and c in (4, 5)]
    physical_cases = [c for c in physical_cases if c in post_cases][:2]
    if len(physical_cases) < 2:
        raise ContractError(f"expected at least two stress-v1 physical-gain cases; got {physical_cases}")
    runner_extra = [c for c, row in sorted(cov.items()) if bool(row.get("runner_episode_material")) and c not in physical_cases]
    if not runner_extra:
        runner_extra = [c for c in (1, 6) if c in post_cases and c not in physical_cases]
    same_controls_suggested = (((post_raw.get("next_plan") or {}).get("candidate_cases_development_only") or {}).get("same_stratum_controls_suggested") or [0, 2, 3, 7])
    same_controls = [safe_int(c) for c in same_controls_suggested if safe_int(c) in post_cases and safe_int(c) not in physical_cases and safe_int(c) not in runner_extra]
    lower_suggested = (((post_raw.get("next_plan") or {}).get("candidate_cases_development_only") or {}).get("lower_stress_controls_suggested") or [16, 17, 18, 19])
    lower_controls = [safe_int(c) for c in lower_suggested if safe_int(c) in post_cases]
    plan: List[Dict[str, Any]] = []
    for c in physical_cases[:2]:
        plan.extend([
            {"case": c, "role": "missed_physical_gain_ge3", "window": "early", "low_frac": 0.08, "high_frac": 0.28},
            {"case": c, "role": "missed_physical_gain_ge3", "window": "middle", "low_frac": 0.30, "high_frac": 0.55},
            {"case": c, "role": "missed_physical_gain_ge3", "window": "late", "low_frac": 0.56, "high_frac": 0.78},
        ])
    for c in runner_extra[:2]:
        plan.append({"case": c, "role": "runner_material_not_physical_ge3_context", "window": "mid_high_trace", "low_frac": 0.20, "high_frac": 0.70})
    for c in same_controls[:2]:
        plan.append({"case": c, "role": "same_stratum_negative_control", "window": "mid_high_trace", "low_frac": 0.20, "high_frac": 0.70})
    for c in lower_controls[:2]:
        plan.append({"case": c, "role": "lower_stress_control", "window": "mid_high_trace", "low_frac": 0.20, "high_frac": 0.70})
    if len(plan) != TARGET_COUNT:
        raise ContractError(f"target case/window plan has {len(plan)} entries, expected {TARGET_COUNT}; physical={physical_cases}, runner_extra={runner_extra}, same_controls={same_controls}, lower_controls={lower_controls}")
    return plan


def build_targets(stage1_raw: Mapping[str, Any], coverage_raw: Mapping[str, Any], post_raw: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metas = get_selection_meta(stage1_raw)
    cov = coverage_case_rows(coverage_raw)
    post_cases = stage1_post_case_rows(post_raw)
    plan = derive_case_plan(coverage_raw, post_raw, metas)
    by_case_trace: Dict[int, Tuple[Path, List[Mapping[str, Any]]]] = {}
    used_steps: Dict[int, set] = {}
    targets: List[Dict[str, Any]] = []
    for item in plan:
        case = int(item["case"])
        if case not in by_case_trace:
            tp = trace_path_for(stage1_raw, case, PREFIX_H)
            by_case_trace[case] = (tp, read_json(tp))
            used_steps[case] = set()
        trace_path, trace = by_case_trace[case]
        target = choose_window(trace, case, item["role"], item["window"], float(item["low_frac"]), float(item["high_frac"]), used_steps[case])
        meta = dict(metas[case])
        crow = dict(cov.get(case, {}))
        prow = dict(post_cases.get(case, {}))
        target.update({
            "target_index": len(targets),
            "state_id": f"v1e_t{len(targets):02d}_case{case:02d}_cand{safe_int(meta.get('candidate_index')):03d}_b{target['branch_step']:03d}_{target['window']}",
            "source_candidate_index": safe_int(meta.get("candidate_index")),
            "selection_group": meta.get("selection_group") or meta.get("stratum") or prow.get("group"),
            "case_metadata": {k: meta.get(k) for k in ("theta_r", "abs_theta_r", "traj_steps", "min_reference_obstacle_clearance", "stress_flags_count", "stress_v1_score", "selection_reason", "selection_fallbacks")},
            "stage1_material_strict_horizons_vs_H15": prow.get("material_strict_horizons_vs_H15") or crow.get("material_strict_horizons_vs_H15") or [],
            "stage1_best_strict_physical_horizon": prow.get("best_strict_physical_horizon"),
            "stage1_fastest_strict_horizon": prow.get("fastest_strict_horizon"),
            "stage1_runner_episode_material": bool(crow.get("runner_episode_material", False)),
            "stage1_episode_physical_gain_ge_3_case": bool(crow.get("episode_physical_gain_ge_3_case", False)),
            "v1d_selected_target_count": safe_int(crow.get("v1d_selected_target_count"), 0),
            "h15_trace_path": rel(trace_path),
            "h15_trace_sha256": sha256(trace_path),
            "selection_rule": "case chosen from pre-existing stress-v1 Stage1/postdiagnostic/crossbank evidence; branch step chosen from H15 trace only, before any v1e non-H15 branch rollout",
        })
        targets.append(target)
    if len({t["state_id"] for t in targets}) != len(targets):
        raise ContractError("duplicate target state_id")
    return targets, {
        "plan": plan,
        "case_counts": {str(c): sum(1 for t in targets if int(t["case"]) == c) for c in sorted({int(t["case"]) for t in targets})},
        "role_counts": {role: sum(1 for t in targets if t["case_role"] == role) for role in sorted({str(t["case_role"]) for t in targets})},
        "branch_step_by_case": {str(c): [int(t["branch_step"]) for t in targets if int(t["case"]) == c] for c in sorted({int(t["case"]) for t in targets})},
    }


def build_schedule(targets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 0
    for t in targets:
        for mode in TERMINAL_MODES:
            for h in BRANCH_HORIZONS:
                rows.append({
                    "schedule_base_index": idx,
                    "target_index": int(t["target_index"]),
                    "state_id": t["state_id"],
                    "case": int(t["case"]),
                    "source_candidate_index": int(t["source_candidate_index"]),
                    "selection_group": t.get("selection_group"),
                    "case_role": t.get("case_role"),
                    "branch_step": int(t["branch_step"]),
                    "prefix_horizon": PREFIX_H,
                    "horizon": int(h),
                    "terminal_mode": str(mode),
                    "terminal_mode_runner_alias": "h15_terminal" if mode == "h15_common_terminal" else mode,
                })
                idx += 1
    rng = random.Random(ORDER_SEED)
    order = list(range(len(rows)))
    rng.shuffle(order)
    return [dict(rows[i], execution_index=int(j)) for j, i in enumerate(order)]


def write_protocol(created: dt.datetime, input_hashes: Mapping[str, str], targets: Sequence[Mapping[str, Any]], diagnostics: Mapping[str, Any], crossbank: Mapping[str, Any], coverage: Mapping[str, Any], stage1_post: Mapping[str, Any]) -> Mapping[str, Any]:
    schedule = build_schedule(targets)
    protocol = {
        "protocol_id": f"{NAME}_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_stress_v1_only_targeted_common_prefix_prepare_not_validation_not_final_test",
        "hypothesis": "v1d zero-label results were partly caused by target-state coverage failure.  If stress-v1 H15-prefix states sampled from missed episode-positive cases 4/5 still yield no terminal-stable material positives, then sparse within-episode opportunity/terminal-model design is more plausible than merely poor state selection.",
        "provenance_correction": {
            "crossbank_classification": (crossbank.get("headline") or {}).get("classification"),
            "crossbank_decision": (crossbank.get("headline") or {}).get("corrective_decision"),
            "do_not_reuse_stage2_v0b_case_ids_as_stress_v1_targets": True,
        },
        "access_rules": {
            "development_only": True,
            "historical_validation64_bank_opened": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
            "final_test_authorized": False,
            "requires_verified_external_backup_after_prepare_before_any_v1e_smoke_rollout": True,
            "do_not_train_or_refit_from_current_v1d_v1b_labels": True,
        },
        "inputs": {
            "hashes": input_hashes,
            "stage1_bank_path_for_future_rollout": rel(STAGE1_BANK_PATH),
            "stage1_postdiagnostic_decision": stage1_post.get("decision"),
            "coverage_headline": coverage.get("headline"),
            "crossbank_headline": crossbank.get("headline"),
        },
        "target_selection": {
            "targets_exact": len(targets),
            "selection_is_from_H15_trace_only_before_v1e_branch_outcomes": True,
            "case_plan_diagnostics": diagnostics,
            "targets": targets,
        },
        "planned_smoke_after_backup": {
            "prefix_horizon": PREFIX_H,
            "branch_horizons": BRANCH_HORIZONS,
            "terminal_modes": TERMINAL_MODES,
            "runner_terminal_alias": {"h15_common_terminal": "h15_terminal", "zero_terminal": "zero_terminal"},
            "schedule_order_seed": ORDER_SEED,
            "episodes_exact": len(schedule),
            "control_step_upper_bound": len(schedule) * MAX_STEPS,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
            "candidate_pool_resets": 0,
            "schedule": schedule,
        },
        "analysis_rules_frozen_before_rollout": {
            "primary_label": "terminal-stable physical positive: non-H15 horizon improves realised continuation physical cost by >=3 versus H15 from the identical H15 prefix, under both zero_terminal and H15_common_terminal, with no success/constraint/initial/final/solver regression, matching prefix hash and branch-state distance <=1e-5",
            "secondary_metrics": ["continuation total cost", "whole decision timing", "solver-attempt timing", "H distribution only as proxy", "branch MPC objective/value if captured by future runner"],
            "blocking_artifacts": ["missing H15 reference", "missing horizon/mode", "branch not reached", "prefix hash mismatch", "state distance > tolerance", "success/constraint/solver regression", "common-terminal positive non-success"],
            "acceptance_before_any_selector_or_value_refit": ">=2 terminal-stable material positive states across >=2 stress-v1 cases, >=4 retained negative/control states, control false-positive rate <=25%, and zero blocking artifacts",
            "if_gate_fails": "do not train/refit selector from these labels; pivot to richer terminal-value/modeling or documented sparse-opportunity scenario diagnosis rather than another unchanged label-density sweep",
            "if_gate_passes": "after backup, freeze one compact IMPROVED selector/value-refit smoke with strong fixed-H/Pareto baselines; still no validation64/sealed-test use",
        },
    }
    write_json(PROTOCOL_JSON, protocol)
    lines = [
        "# Vehicle stress-v1e targeted common-prefix prepare v0",
        "",
        f"UTC: `{created.isoformat()}`. No simulations, no candidate resets, no training/refit, no validation64 bank, no sealed test.",
        "",
        "## Headline",
        "",
        "- Cross-bank audit confirmed Stage2-v0b prefix-blocked candidates are legacy stress-v0 evidence, not valid stress-v1/v1d targets.",
        "- This prepare step freezes stress-v1-only targets from existing stress-v1 H15 traces for missed episode-positive cases plus controls.",
        f"- Targets: `{len(targets)}`; planned v1e smoke after backup: `{len(schedule)}` episodes / `{len(schedule) * MAX_STEPS}` control-step cap.",
        "- Train/refit now: `False`.",
        "",
        "## Frozen targets",
        "",
        "| target | case | role | group | source cand | branch step | window | score | v1d target count | stage1 material H |",
        "|---:|---:|---|---|---:|---:|---|---:|---:|---|",
    ]
    for t in targets:
        lines.append("| %d | %d | `%s` | `%s` | %d | %d | `%s` | %.6g | %d | `%s` |" % (
            int(t["target_index"]), int(t["case"]), t["case_role"], t.get("selection_group"), int(t["source_candidate_index"]), int(t["branch_step"]), t["window"], float(t["h15_selection_score"]), int(t.get("v1d_selected_target_count", 0)), t.get("stage1_material_strict_horizons_vs_H15"),
        ))
    lines += [
        "",
        "## Four-axis decision update",
        "",
        "- SCENARIOS: v1d missed stress-v1 physical-gain cases 4/5. v1e now targets those exact stress-v1 cases from H15 traces, plus runner-material context and negative/lower-stress controls.",
        "- REWARD/TERMINAL: labels remain terminal-stable realised physical continuation under zero and H15-common terminal; raw objective/per-H/H25 labels remain excluded as selector targets.",
        "- TRAINING: no selector/value refit is justified until the v1e smoke gate passes; this prepare is zero training/gradient/refit.",
        "- COMPARISONS: no adaptive or timing claim. A later method, if any, must face strong same-distribution fixed-H/Pareto baselines and measured timing.",
        "",
        "## Next action after backup",
        "",
        "Implement/run the bounded v1e stress-v1-only common-prefix smoke using the frozen schedule in the protocol. Do not run validation64 or sealed test. Do not reuse Stage2-v0b candidates as stress-v1 targets.",
        "",
        f"Frozen protocol JSON: `{rel(PROTOCOL_JSON)}`.",
        f"Backup request before v1e smoke: `{rel(BACKUP_REQ)}`.",
    ]
    PROTOCOL_MD.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return protocol


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if marker not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def run_prepare() -> int:
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if OUT.exists() and any(OUT.iterdir()):
        raise ContractError(f"partial output exists; inspect first: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=True)
    created = now_utc()
    input_hashes = require_inputs()
    stage1_raw = read_json(STAGE1_RAW)
    stage1_post = read_json(STAGE1_POST_RAW)
    coverage = read_json(COVERAGE_RAW)
    crossbank = read_json(CROSSBANK_RAW)
    headline = crossbank.get("headline") or {}
    if headline.get("classification") != "stage2_prefix_blocked_candidates_are_cross_bank_not_valid_v1d_targets":
        raise ContractError("crossbank audit did not classify Stage2-v0b candidates as invalid stress-v1 targets; inspect before v1e prep")
    if stage1_raw.get("sealed_test_accessed") is not False or stage1_raw.get("historical_validation64_bank_opened") is not False:
        raise ContractError("Stage1 raw access flags invalid")
    targets, diagnostics = build_targets(stage1_raw, coverage, stage1_post)
    protocol = write_protocol(created, input_hashes, targets, diagnostics, crossbank, coverage, stage1_post)
    write_json(BACKUP_REQ, {
        "requested_utc": created.isoformat(),
        "reason": "backup v1e stress-v1-only target-preparation source/protocol/outputs before any common-prefix smoke rollout",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "planned_smoke_episodes_exact": protocol["planned_smoke_after_backup"]["episodes_exact"],
        "planned_smoke_control_step_upper_bound": protocol["planned_smoke_after_backup"]["control_step_upper_bound"],
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(SOURCE), rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(OUT), rel(STATE), rel(BACKUP_REQ)],
    })
    result = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_no_simulation_stress_v1e_targeted_common_prefix_prepare_no_validation64_no_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "input_hashes": input_hashes,
        "protocol_json": rel(PROTOCOL_JSON),
        "protocol_md": rel(PROTOCOL_MD),
        "protocol_sha256": sha256(PROTOCOL_JSON),
        "target_count": len(targets),
        "planned_smoke_episodes_exact": protocol["planned_smoke_after_backup"]["episodes_exact"],
        "planned_smoke_control_step_upper_bound": protocol["planned_smoke_after_backup"]["control_step_upper_bound"],
        "case_plan_diagnostics": diagnostics,
        "targets": targets,
        "headline": {
            "classification": "stress_v1_only_v1e_targets_frozen_after_crossbank_correction",
            "target_cases": sorted({int(t["case"]) for t in targets}),
            "target_count": len(targets),
            "planned_smoke_episodes_exact": protocol["planned_smoke_after_backup"]["episodes_exact"],
            "planned_smoke_control_step_upper_bound": protocol["planned_smoke_after_backup"]["control_step_upper_bound"],
            "train_or_refit_now": False,
            "next_action": "after verified backup, run bounded v1e common-prefix smoke from frozen stress-v1-only targets",
        },
        "backup_request": rel(BACKUP_REQ),
    }
    write_json(OUT / "raw.json", result)
    summary = PROTOCOL_MD.read_text(encoding="utf-8")
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(summary, encoding="utf-8")
    marker = f"{NAME}-{STAMP}"
    append_docs("\n".join([f"## {marker}", "", summary.strip()]), marker)
    files = [SOURCE, PROTOCOL_JSON, PROTOCOL_MD, STATE, BACKUP_REQ, OUT / "raw.json", OUT / "summary.md"] + [STAGE1_RAW, STAGE1_COMPLETED, STAGE1_POST_RAW, STAGE1_POST_COMPLETED, COVERAGE_RAW, COVERAGE_COMPLETED, CROSSBANK_RAW, CROSSBANK_COMPLETED]
    write_json(OUT / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "backup_required_before_smoke": True,
        "backup_request": rel(BACKUP_REQ),
        "headline": result["headline"],
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "protocol_json": rel(PROTOCOL_JSON),
        "target_count": len(targets),
        "target_cases": result["headline"]["target_cases"],
        "planned_smoke_episodes_exact": result["planned_smoke_episodes_exact"],
        "planned_smoke_control_step_upper_bound": result["planned_smoke_control_step_upper_bound"],
        "train_or_refit_now": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(BACKUP_REQ),
    }, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--i-accept-development-no-simulation", action="store_true")
    args = ap.parse_args(argv)
    if not args.prepare_only:
        raise ContractError("only --prepare-only is implemented in this v1e preparation script")
    if not args.i_accept_development_no_simulation:
        raise ContractError("explicit --i-accept-development-no-simulation required")
    try:
        return run_prepare()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {
            "passed": False,
            "created_utc": now_utc().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "candidate_pool_resets": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
            "next_recovery_hint": "Preserve failure. Repair only source/schema/path extraction; do not run simulations or train/refit until prepare succeeds and is backed up.",
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
