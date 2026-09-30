#!/usr/bin/env python3
"""v29 success-aware longer-H feasibility probe after v28.

Development-only IMPROVED diagnostic.  This runner executes the frozen v28
follow-up protocol: identical saved branch states with horizons H12/H15/H25/H35,
separating (a) v27 source242 rows where H12 and H15 both failed, (b) opened-v19
H12-only high-cost rows where H15 was safe, and (c) preselected safe controls.

No validation64 bank or sealed test is opened.  No training/refit/grid-search is
performed.  A verified external backup after the v28 audit is required before
running because this script creates new unique simulation evidence.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_probe_acquisition_v8 as v8  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402
import vehicle_true_variable_horizon_intermediate_h12_boundary_v19 as v19  # noqa:E402

NAME = "vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29"
STAMP = "20260930T0340Z"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0340_after_v29_longer_H_probe.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V29_SUCCESS_AWARE_LONGER_H_PROBE_{STAMP}.json"
MARKER = f"vehicle-success-aware-longer-H-feasibility-v29-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

V28_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_success_aware_scenario_comparison_audit_v28_frozen_success_aware_followup_20260930T0315Z.json"
V28_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_scenario_comparison_audit_v28_20260930T0315Z/completed.json"
V27_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_confirmation_v27_20260930T0250Z/raw.json"
V27_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_confirmation_v27_20260930T0250Z/completed.json"
V27_MANIFEST = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_confirmation_v27_20260930T0250Z/selected_state_manifest.json"
V19_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/raw.json"
V19_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/completed.json"
V19_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_preoutcome_frozen_20260930T0015Z.json"

HORIZONS = [12, 15, 25, 35]
MAX_STEPS = 150
RNG_SEED = 202609300340
CONTROL_STEP_CAP = 6600
MIN_DECISION_SAVING = 0.05

V27_BOTH_FAIL_BASE_IDS = [
    "v27_case09_slot0_early_risk",
    "v27_case09_slot1_mid_late_risk",
]
V19_H12_ONLY_FAILURE_CANDIDATE_INDICES = [12, 13, 14]
# Preselected before any v29 H25/H35 outcome.  These cover lower stress,
# high-heading/high-stress, and global-diverse v27 safe H12-support rows.
V27_SAFE_CONTROL_BASE_IDS = [
    "v27_case00_slot0_early_risk",
    "v27_case00_slot1_mid_late_risk",
    "v27_case08_slot0_early_risk",
    "v27_case08_slot1_mid_late_risk",
    "v27_case11_slot0_early_risk",
    "v27_case11_slot1_mid_late_risk",
]
EXPECTED_STATES = len(V27_BOTH_FAIL_BASE_IDS) + len(V19_H12_ONLY_FAILURE_CANDIDATE_INDICES) + len(V27_SAFE_CONTROL_BASE_IDS)
EXPECTED_EPISODES = EXPECTED_STATES * len(HORIZONS)

class ContractError(RuntimeError):
    pass

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)

def clean(x: Any) -> Any:
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if hasattr(x, "item"):
        return clean(x.item())
    return x

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

def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default

def si(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def pct(x: Any) -> str:
    return f"{100.0 * sf(x):.2f}%"

def metric_sum(e: Mapping[str, Any], name: str) -> float:
    obj = e.get(name)
    return sf(obj.get("sum"), 0.0) if isinstance(obj, Mapping) else 0.0

def is_safe(e: Mapping[str, Any]) -> bool:
    return bool(e.get("success")) and not bool(e.get("constraint")) and si(e.get("solver_failure_steps"), 999) == 0 and si(e.get("initial_failed_steps"), 999) == 0 and si(e.get("final_failed_steps"), 999) == 0

def parse_time(s: Any) -> Optional[dt.datetime]:
    if not isinstance(s, str):
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def completed_ok(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing prerequisite {label}: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True and obj.get("status") not in ("complete", "completed"):
        raise ContractError(f"prerequisite not complete: {label}")
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=true in {label}")
    return obj

def verify_backup(path: Path, v28_time: Optional[dt.datetime]) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("backup proof missing: " + rel(path))
    obj = read_json(path)
    if obj.get("status") != "verified" or obj.get("backup_verified") is not True:
        raise ContractError("backup proof not verified: " + rel(path))
    if obj.get("remaining_changed_files") not in (0, "0"):
        raise ContractError("backup proof has remaining changed files: " + rel(path))
    if not obj.get("commit") or not obj.get("packages_this_run"):
        raise ContractError("backup proof lacks commit/package evidence: " + rel(path))
    if v28_time is not None:
        t = parse_time(obj.get("time"))
        if t is None or t <= v28_time:
            raise ContractError("backup proof does not postdate v28 audit/protocol outputs")
    return obj

def load_protocol_and_inputs() -> Dict[str, Any]:
    v28_done = completed_ok(V28_DONE, "v28 success-aware audit")
    v27_done = completed_ok(V27_DONE, "v27 fixed-H12 confirmation")
    v19_done = completed_ok(V19_DONE, "v19 intermediate H12 boundary")
    protocol = read_json(V28_PROTOCOL)
    if protocol.get("status") != "frozen_planned_not_executed":
        raise ContractError("unexpected v28 protocol status")
    max_budget = protocol.get("max_budget") or {}
    if si(max_budget.get("development_episodes"), -1) != EXPECTED_EPISODES or si(max_budget.get("control_step_cap"), -1) != CONTROL_STEP_CAP:
        raise ContractError("v29 constants do not match frozen v28 budget")
    if list(protocol.get("planned_horizons") or []) != HORIZONS:
        raise ContractError("v29 horizon list does not match frozen v28 protocol")
    return {"v28_done": v28_done, "v27_done": v27_done, "v19_done": v19_done, "protocol": protocol}

def terminal_for_horizon(h: int, terminals: Mapping[int, Any]) -> Tuple[Any, int, str]:
    # H12 has no separately trained terminal in the current grid, so it keeps the
    # established shared-H15 terminal.  H15/H25/H35 use matched fixed-H terminals
    # to test strong fixed-H feasibility where available.
    if h in terminals and h != 12:
        return terminals[h], h, "matched_terminal_available"
    if 15 not in terminals:
        raise ContractError("terminal grid missing H15 fallback")
    return terminals[15], 15, "shared_h15_terminal_fallback"

def load_v27_state_specs() -> List[Dict[str, Any]]:
    raw = read_json(V27_RAW)
    manifest = read_json(V27_MANIFEST)
    cases_by_fc = {si(c.get("fresh_case_index"), -1): c for c in raw.get("selected_cases") or []}
    states_by_id = {str(s.get("base_state_id")): s for s in manifest.get("selected_states") or []}
    out: List[Dict[str, Any]] = []
    for category, ids in [("v27_both_fail_h12_h15", V27_BOTH_FAIL_BASE_IDS), ("v27_safe_fixed_h12_support_control", V27_SAFE_CONTROL_BASE_IDS)]:
        for bid in ids:
            st = states_by_id.get(bid)
            if st is None:
                raise ContractError("missing v27 manifest state " + bid)
            case_meta = cases_by_fc.get(si(st.get("fresh_case_index"), -1))
            if case_meta is None or "case_snapshot_from_candidate_pool" not in case_meta:
                raise ContractError("missing v27 case snapshot for " + bid)
            out.append({
                "category": category,
                "source_campaign": "v27",
                "base_state_id": bid,
                "state_label": bid,
                "source_candidate_index": si(st.get("source_candidate_index"), -1),
                "fresh_case_index": si(st.get("fresh_case_index"), -1),
                "role": st.get("role"),
                "branch_step": si(st.get("branch_step"), -1),
                "branch_previous_state": copy.deepcopy(st.get("branch_previous_state") or {}),
                "initial_observation": copy.deepcopy(st.get("initial_observation_from_h15_trace") or []),
                "case_snapshot": copy.deepcopy(case_meta["case_snapshot_from_candidate_pool"]),
                "selection_rule": "v29 preselected from v27 categories frozen by v28 before any H25/H35 outcome",
            })
    return out

def load_v19_state_specs() -> List[Dict[str, Any]]:
    prepared = v19.prepare_v15_candidates()
    by_idx = {si(c.get("candidate_index"), -1): c for c in prepared}
    out: List[Dict[str, Any]] = []
    for idx in V19_H12_ONLY_FAILURE_CANDIDATE_INDICES:
        cand = by_idx.get(idx)
        if cand is None:
            raise ContractError("missing v19 prepared candidate index " + str(idx))
        out.append({
            "category": "v19_h12_only_failure_h15_safe",
            "source_campaign": "v19",
            "base_state_id": str(cand.get("candidate_id")),
            "state_label": f"v19_c{idx:02d}",
            "candidate_index": idx,
            "source_candidate_index": si(cand.get("source_candidate_index"), -1),
            "role": cand.get("role"),
            "source_key": cand.get("source_key"),
            "offset_from_center": si(cand.get("offset_from_center"), 0),
            "branch_step": si(cand.get("candidate_branch_step"), -1),
            "branch_previous_state": copy.deepcopy(cand.get("branch_previous_state") or {}),
            "initial_observation": copy.deepcopy(cand.get("initial_observation_from_h15_trace") or []),
            "case_snapshot": copy.deepcopy(cand["case_snapshot_from_candidate_pool"]),
            "selection_rule": "v29 preselected from v19 collapsed H12-only high-cost rows: candidate indices 12/13/14",
        })
    return out

def build_state_specs() -> List[Dict[str, Any]]:
    specs = load_v27_state_specs() + load_v19_state_specs()
    if len(specs) != EXPECTED_STATES:
        raise ContractError(f"state count mismatch {len(specs)} != {EXPECTED_STATES}")
    seen = set()
    for s in specs:
        if s["base_state_id"] in seen:
            raise ContractError("duplicate state id " + str(s["base_state_id"]))
        seen.add(s["base_state_id"])
    return specs

def make_schedule(specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    rows: List[Dict[str, Any]] = []
    exe = 0
    # Keep state categories blocked for interpretability, but randomize horizon
    # order within each state to reduce monotonic runtime drift.
    for si_state, spec in enumerate(specs):
        order = list(HORIZONS)
        rng.shuffle(order)
        for h in order:
            rows.append({
                "execution_index": exe,
                "state_index": si_state,
                "category": spec["category"],
                "source_campaign": spec["source_campaign"],
                "base_state_id": spec["base_state_id"],
                "state_label": spec["state_label"],
                "source_candidate_index": spec.get("source_candidate_index"),
                "candidate_index": spec.get("candidate_index"),
                "branch_step": spec["branch_step"],
                "true_mpc_n_horizon": int(h),
                "commanded_horizon": int(h),
                "blocked_randomization_unit": f"v29|state{si_state:02d}|{spec['state_label']}",
            })
            exe += 1
    return rows

def episode_record(summary: Mapping[str, Any], spec: Mapping[str, Any], h: int, terminal_h: int, terminal_profile: str) -> Dict[str, Any]:
    return {
        "category": spec["category"],
        "source_campaign": spec["source_campaign"],
        "base_state_id": spec["base_state_id"],
        "state_label": spec["state_label"],
        "source_candidate_index": spec.get("source_candidate_index"),
        "candidate_index": spec.get("candidate_index"),
        "role": spec.get("role"),
        "horizon": int(h),
        "terminal_source_horizon": int(terminal_h),
        "terminal_profile": terminal_profile,
        "success": bool(summary.get("success")),
        "constraint": bool(summary.get("constraint")),
        "safe_success_no_solver_fail": is_safe(summary),
        "termination": summary.get("termination"),
        "steps": si(summary.get("steps"), 0),
        "hit_step_cap": si(summary.get("steps"), 0) >= MAX_STEPS,
        "solver_failure_steps": si(summary.get("solver_failure_steps"), 0),
        "initial_failed_steps": si(summary.get("initial_failed_steps"), 0),
        "final_failed_steps": si(summary.get("final_failed_steps"), 0),
        "physical_constraint_cost": sf(summary.get("physical_constraint_cost"), 0.0),
        "total_cost": sf(summary.get("total_cost"), 0.0),
        "decision_sum_s": metric_sum(summary, "decision_timing_s"),
        "decision_mean_s": sf((summary.get("decision_timing_s") or {}).get("mean")),
        "decision_median_s": sf((summary.get("decision_timing_s") or {}).get("median")),
        "decision_p95_s": sf((summary.get("decision_timing_s") or {}).get("p95")),
        "solver_sum_s": metric_sum(summary, "solver_attempt_timing_s"),
        "solver_mean_s": sf((summary.get("solver_attempt_timing_s") or {}).get("mean")),
        "solver_median_s": sf((summary.get("solver_attempt_timing_s") or {}).get("median")),
        "solver_p95_s": sf((summary.get("solver_attempt_timing_s") or {}).get("p95")),
        "opt_x_sizes_observed": summary.get("opt_x_sizes_observed"),
        "path": summary.get("path"),
    }

def analyze(episodes: Sequence[Mapping[str, Any]], specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_state_h: Dict[str, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for e in episodes:
        by_state_h[str(e["base_state_id"])][int(e["horizon"])] = e
    state_rows: List[Dict[str, Any]] = []
    horizon_counts: Dict[str, Any] = {}
    for h in HORIZONS:
        hs = [e for e in episodes if int(e["horizon"]) == h]
        horizon_counts[str(h)] = {
            "episodes": len(hs),
            "safe_success_count": sum(1 for e in hs if e["safe_success_no_solver_fail"]),
            "problem_count": sum(1 for e in hs if (not e["safe_success_no_solver_fail"]) or e["hit_step_cap"]),
            "decision_sum_s": math.fsum(sf(e["decision_sum_s"]) for e in hs),
            "solver_sum_s": math.fsum(sf(e["solver_sum_s"]) for e in hs),
            "physical_sum": math.fsum(sf(e["physical_constraint_cost"]) for e in hs),
        }
    category_counts: Dict[str, Counter] = defaultdict(Counter)
    for spec in specs:
        d = by_state_h.get(str(spec["base_state_id"]), {})
        missing_h = [h for h in HORIZONS if h not in d]
        if missing_h:
            raise ContractError("missing horizon outcomes for " + str(spec["base_state_id"]) + ": " + str(missing_h))
        safe_h = [h for h in HORIZONS if d[h]["safe_success_no_solver_fail"] and not d[h]["hit_step_cap"]]
        fastest_safe = min(safe_h, key=lambda h: sf(d[h]["decision_sum_s"])) if safe_h else None
        h15 = d[15]; h12 = d[12]
        tol = max(2.0, 0.05 * abs(sf(h15["physical_constraint_cost"])))
        cat_threshold = max(2.0, 0.25 * abs(sf(h15["physical_constraint_cost"])))
        h12_relative_bad_vs_h15 = bool(h15["safe_success_no_solver_fail"] and ((not h12["safe_success_no_solver_fail"]) or (sf(h12["physical_constraint_cost"]) - sf(h15["physical_constraint_cost"]) > cat_threshold)))
        h12_beneficial_vs_h15 = bool(h15["safe_success_no_solver_fail"] and h12["safe_success_no_solver_fail"] and (sf(h12["physical_constraint_cost"]) - sf(h15["physical_constraint_cost"]) <= tol) and (sf(h12["decision_sum_s"]) < sf(h15["decision_sum_s"])))
        longer_safe = [h for h in (25, 35) if h in safe_h]
        both_h12_h15_problem = bool((not h12["safe_success_no_solver_fail"] or h12["hit_step_cap"]) and (not h15["safe_success_no_solver_fail"] or h15["hit_step_cap"]))
        longer_rescues_h12_h15 = bool(both_h12_h15_problem and longer_safe)
        category_counts[spec["category"]]["states"] += 1
        if fastest_safe is not None:
            category_counts[spec["category"]][f"fastest_safe_H{fastest_safe}"] += 1
        if longer_rescues_h12_h15:
            category_counts[spec["category"]]["longer_rescue_H25_or_H35"] += 1
        if h12_relative_bad_vs_h15:
            category_counts[spec["category"]]["h12_relative_bad_vs_h15"] += 1
        if h12_beneficial_vs_h15:
            category_counts[spec["category"]]["h12_beneficial_vs_h15"] += 1
        if not safe_h:
            category_counts[spec["category"]]["no_safe_horizon"] += 1
        state_rows.append({
            "category": spec["category"],
            "source_campaign": spec["source_campaign"],
            "base_state_id": spec["base_state_id"],
            "state_label": spec["state_label"],
            "source_candidate_index": spec.get("source_candidate_index"),
            "candidate_index": spec.get("candidate_index"),
            "role": spec.get("role"),
            "branch_step": spec.get("branch_step"),
            "safe_horizons": safe_h,
            "fastest_safe_horizon_by_decision_sum": fastest_safe,
            "both_H12_H15_problem": both_h12_h15_problem,
            "longer_safe_horizons": longer_safe,
            "longer_rescues_h12_h15_problem": longer_rescues_h12_h15,
            "h12_relative_bad_vs_h15": h12_relative_bad_vs_h15,
            "h12_beneficial_vs_h15": h12_beneficial_vs_h15,
            "row_tolerance_vs_h15": tol,
            "catastrophic_threshold_vs_h15": cat_threshold,
            "per_horizon": {str(h): d[h] for h in HORIZONS},
        })
    all_states = len(state_rows)
    no_safe = sum(1 for r in state_rows if not r["safe_horizons"])
    bothfail_rescued = sum(1 for r in state_rows if r["category"] == "v27_both_fail_h12_h15" and r["longer_rescues_h12_h15_problem"])
    v19_bad_reproduced = sum(1 for r in state_rows if r["category"] == "v19_h12_only_failure_h15_safe" and r["h12_relative_bad_vs_h15"])
    control_h12_fastest = sum(1 for r in state_rows if r["category"] == "v27_safe_fixed_h12_support_control" and r["fastest_safe_horizon_by_decision_sum"] == 12)
    if bothfail_rescued:
        decision = "Longer fixed-H rescues at least one source242 H12/H15 both-fail row; broaden fixed-H/scenario comparison before any adaptive selector validation."
    elif no_safe:
        decision = "At least one selected state remains unsafe for all tested horizons; treat those rows as scenario infeasibility/hard-case evidence, not timing-success evidence."
    elif v19_bad_reproduced and control_h12_fastest >= 4:
        decision = "v19 H12-only risk reproduces while controls mostly favor H12; next test pre-outcome separability and consider success-aware selector/value refit if fixed longer-H does not dominate."
    elif control_h12_fastest >= 4 and v19_bad_reproduced == 0:
        decision = "Fixed H12 remains dominant on controls and v19 risk did not reproduce under the v29 terminal setup; prioritize scenario/comparison design and avoid selector retraining to force adaptation."
    else:
        decision = "Mixed longer-H feasibility; inspect per-state rows before choosing refit/training versus scenario redesign."
    return {
        "state_rows": state_rows,
        "horizon_counts": horizon_counts,
        "category_counts": {k: dict(v) for k, v in category_counts.items()},
        "headline": {
            "states": all_states,
            "episodes": len(episodes),
            "horizon_counts": {str(h): horizon_counts[str(h)]["episodes"] for h in HORIZONS},
            "absolute_no_safe_horizon_states": no_safe,
            "source242_bothfail_rows_rescued_by_H25_or_H35": bothfail_rescued,
            "v19_h12_only_failure_rows_reproduced": v19_bad_reproduced,
            "safe_control_rows_fastest_H12": control_h12_fastest,
        },
        "decision": decision,
    }

def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["analysis"]["headline"]
    lines = [
        "# v29 success-aware longer-H feasibility probe",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only identical-state H12/H15/H25/H35 probe; validation64 closed, sealed test closed, no training/refit.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` / `{raw['budget_declared']['development_episodes']}` episodes; `{raw['budget_actual']['control_steps']}` / `{raw['budget_declared']['control_step_cap']}` control steps.",
        "",
        "## Headline",
        "",
        f"- States: `{h['states']}`; episodes by horizon `{h['horizon_counts']}`.",
        f"- No-safe-horizon states: `{h['absolute_no_safe_horizon_states']}`.",
        f"- Source242 H12/H15 both-fail rows rescued by H25/H35: `{h['source242_bothfail_rows_rescued_by_H25_or_H35']}`.",
        f"- v19 H12-only failure rows reproduced: `{h['v19_h12_only_failure_rows_reproduced']}` / `{len(V19_H12_ONLY_FAILURE_CANDIDATE_INDICES)}`.",
        f"- Safe controls with fastest safe H12: `{h['safe_control_rows_fastest_H12']}` / `{len(V27_SAFE_CONTROL_BASE_IDS)}`.",
        f"- Decision: {raw['analysis']['decision']}",
        "",
        "## Horizon totals (do not treat failed-row timing as speed evidence)",
        "",
        "| H | episodes | safe successes | problem rows | decision sum s | solver sum s | physical sum |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hh in HORIZONS:
        r = raw["analysis"]["horizon_counts"][str(hh)]
        lines.append(f"| {hh} | {r['episodes']} | {r['safe_success_count']} | {r['problem_count']} | {r['decision_sum_s']:.6g} | {r['solver_sum_s']:.6g} | {r['physical_sum']:.6g} |")
    lines += [
        "",
        "## Per-state success-aware table",
        "",
        "| category | state | safe H | fastest safe H | longer rescue | H12 bad vs H15 | H12 beneficial vs H15 | H12/H15/H25/H35 physical | H12/H15/H25/H35 decision s |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in raw["analysis"]["state_rows"]:
        ph = [sf(r["per_horizon"][str(hh)]["physical_constraint_cost"]) for hh in HORIZONS]
        de = [sf(r["per_horizon"][str(hh)]["decision_sum_s"]) for hh in HORIZONS]
        lines.append("| `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` |" % (r["category"], r["state_label"], r["safe_horizons"], r["fastest_safe_horizon_by_decision_sum"], r["longer_rescues_h12_h15_problem"], r["h12_relative_bad_vs_h15"], r["h12_beneficial_vs_h15"], [round(x, 6) for x in ph], [round(x, 6) for x in de]))
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is opened development/stress-pool evidence, not validation64 or sealed-test evidence. H12 uses the established shared-H15 terminal fallback; H15/H25/H35 use matched fixed-H terminals where available, so longer-H results are feasibility/comparator evidence rather than a pure one-variable horizon-only ablation.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")

def update_docs(raw: Mapping[str, Any]) -> None:
    h = raw["analysis"]["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-30 v29 success-aware longer-H feasibility probe

UTC: {raw['created_utc']}. Development-only identical-state H12/H15/H25/H35 probe completed; validation64 closed, sealed test closed, no training/refit. Budget {raw['budget_actual']['episodes']} episodes / {raw['budget_actual']['control_steps']} control steps. Source242 H12/H15 both-fail rows rescued by H25/H35={h['source242_bothfail_rows_rescued_by_H25_or_H35']}; v19 H12-only failures reproduced={h['v19_h12_only_failure_rows_reproduced']}; no-safe states={h['absolute_no_safe_horizon_states']}; safe controls fastest H12={h['safe_control_rows_fastest_H12']}. Decision: {raw['analysis']['decision']} Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup required before further unique science: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / doc, MARKER, block)
    response = f"""
## Follow-up through v29 success-aware longer-H feasibility probe

Updated by GPT-5.5 executor at `{raw['created_utc']}`. v29 did not access validation64 or sealed test and performed no training/refit.

| linked recommendation(s) | disposition after v29 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A13_both_fail_rows_must_not_count_as_successful_fixed_H12_pass` | accepted; directly tested | v29 ran H12/H15/H25/H35 on source242 both-fail states and records absolute safe/no-safe rows before timing aggregation. Source242 longer-H rescues: {h['source242_bothfail_rows_rescued_by_H25_or_H35']}. | Continue success-sensitive accounting; failed rows are not speed evidence. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; expanded fixed-H comparison | v29 includes H25/H35 matched fixed-H terminals where available plus H12/H15, on identical saved states. | If longer H rescues or dominates, broaden fixed-H comparator/scenario design before any adaptive validation. |
| `A11_training_failure_modes_need_separation` | accepted; conditional | v19 H12-only failures reproduced={h['v19_h12_only_failure_rows_reproduced']}; controls fastest H12={h['safe_control_rows_fastest_H12']}. | Pivot to terminal-risk/value refit only if per-state v29 evidence shows separable adaptive opportunity not dominated by fixed longer H. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; unchanged | v29 is selected opened development/stress-pool evidence only. | Require fresh independent confirmation before validation/final-test claims. |
| `A12_registry_backup_schema_contract` | accepted; active | v29 wrote new source/results/docs/state/registry and backup request `{rel(BACKUP_REQUEST)}`. | Require verified backup before more unique simulations/refits/validation. |
"""
    append_if_missing(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", MARKER, response)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if MARKER not in old:
        with reg.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([raw["created_utc"], NAME, raw["classification"], f"RNG_SEED={RNG_SEED}", "development_success_aware_longer_H_no_validation64_no_test", raw["budget_actual"]["episodes"], raw["budget_actual"]["control_steps"], 0, 0, 0, False, rel(RUN_DIR / "completed.json"), MARKER])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v29 longer-H probe\n\nUTC: {raw['created_utc']}\n\nDecision: {raw['analysis']['decision']}\n\nHeadline: {h}\n\nNext: require verified backup covering v29. Then decide between broader fixed-H/scenario comparison, terminal-risk/value refit, or scenario-redesign/negative adaptive-opportunity conclusion according to v28 decision rules.\n", encoding="utf-8")

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-proof", required=True)
    ap.add_argument("--i-accept-development-v29", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if (RUN_DIR / "completed.json").exists():
            done = read_json(RUN_DIR / "completed.json")
            print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
            return 0
        inputs = load_protocol_and_inputs()
        v28_time = parse_time(inputs["v28_done"].get("created_utc"))
        backup = verify_backup(Path(args.backup_proof), v28_time)
        started = now_utc()
        write_json(RUN_DIR / "run_started.json", {"started_utc": started.isoformat(), "pid": os.getpid(), "method": NAME, "classification": "development_success_aware_longer_H_probe_no_validation_no_test", "backup_proof": rel(Path(args.backup_proof)), "validation64_bank_opened": False, "sealed_test_accessed": False, "new_gradient_steps": 0, "new_refit_grid_evaluations": 0})
        v8.engine.case_runner.SMOKE_DIR = RUN_DIR
        _, stage1_runner, _ = v1d.import_legacy_modules()
        preflight = stage1_runner.runtime_preflight()
        if not preflight.get("passed"):
            raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
        stage1_runner.base.v1.latency_verify()
        terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
        terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
        for h in (15, 25, 35):
            if h not in terminals:
                raise ContractError(f"terminal grid missing H{h}")
        write_json(RUN_DIR / "runtime_preflight.json", preflight)
        write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})
        specs = build_state_specs()
        schedule = make_schedule(specs)
        write_json(RUN_DIR / "selected_state_manifest.json", {"created_utc": now_utc().isoformat(), "states": [{k: v for k, v in s.items() if k != "case_snapshot"} for s in specs], "schedule": schedule, "selection_completed_before_v29_outcomes": True, "validation64_bank_opened": False, "sealed_test_accessed": False})
        spec_by_index = {i: s for i, s in enumerate(specs)}
        episodes: List[Dict[str, Any]] = []
        for row in schedule:
            spec = spec_by_index[int(row["state_index"])]
            h = int(row["true_mpc_n_horizon"])
            terminal, terminal_h, terminal_profile = terminal_for_horizon(h, terminals)
            item = {
                "execution_index": 9000 + int(row["execution_index"]),
                "state_id": f"v29_{int(row['execution_index']):03d}_{spec['state_label']}_H{h}",
                "case": int(row["state_index"]),
                "source_candidate_index": spec.get("source_candidate_index"),
                "branch_step": int(spec["branch_step"]),
                "branch_previous_state": copy.deepcopy(spec["branch_previous_state"]),
                "true_mpc_n_horizon": h,
                "commanded_horizon": h,
                "terminal_mode": terminal_profile,
                "initialization": "v29_success_aware_direct_branch_state_from_frozen_manifest",
            }
            summary = v8.engine.case_runner.run_true_h_episode(item, spec["case_snapshot"], terminal)
            summary.update({"stage": "v29_success_aware_longer_H_feasibility", "category": spec["category"], "source_campaign": spec["source_campaign"], "base_state_id": spec["base_state_id"], "state_label": spec["state_label"], "role": spec.get("role"), "source_candidate_index": spec.get("source_candidate_index"), "candidate_index": spec.get("candidate_index"), "terminal_source_horizon": terminal_h, "terminal_profile": terminal_profile, "blocked_randomization_unit": row["blocked_randomization_unit"]})
            episodes.append(episode_record(summary, spec, h, terminal_h, terminal_profile))
            write_json(RUN_DIR / "progress.json", {"episodes_done": len(episodes), "episodes_expected": EXPECTED_EPISODES, "control_steps_done": int(sum(si(e.get("steps"), 0) for e in episodes)), "last_episode": episodes[-1], "validation64_bank_opened": False, "sealed_test_accessed": False})
            print(json.dumps({"episodes_done": len(episodes), "last_state": spec["state_label"], "last_h": h, "last_safe": episodes[-1]["safe_success_no_solver_fail"], "last_steps": episodes[-1]["steps"]}, sort_keys=True), flush=True)
        control_steps = int(sum(si(e.get("steps"), 0) for e in episodes))
        if len(episodes) != EXPECTED_EPISODES or control_steps > CONTROL_STEP_CAP:
            raise ContractError("budget violation")
        analysis = analyze(episodes, specs)
        created = now_utc()
        input_hashes = {rel(p): sha256(p) for p in [Path(__file__).resolve(), V28_PROTOCOL, V28_DONE, V27_RAW, V27_DONE, V27_MANIFEST, V19_RAW, V19_DONE, V19_PROTOCOL, Path(args.backup_proof)] if p.exists()}
        raw = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_success_aware_longer_H_feasibility_probe_not_validation_not_test",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "mobile_robot_mppi_resumed": False,
            "backup_proof_used": {"path": rel(Path(args.backup_proof)), "sha256": sha256(Path(args.backup_proof)), "commit": backup.get("commit")},
            "frozen_protocol": {"path": rel(V28_PROTOCOL), "sha256": sha256(V28_PROTOCOL)},
            "horizons": HORIZONS,
            "terminal_policy": "H12 uses shared H15 terminal fallback; H15/H25/H35 use matched fixed-H terminal where available",
            "budget_declared": {"development_episodes": EXPECTED_EPISODES, "control_step_cap": CONTROL_STEP_CAP, "training_episodes": 0, "gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "training_episodes": 0, "gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "selected_states": [{k: v for k, v in s.items() if k != "case_snapshot"} for s in specs],
            "schedule": schedule,
            "episodes": episodes,
            "analysis": analysis,
            "input_hashes": input_hashes,
            "runtime_preflight": preflight,
            "terminal_sources": {str(k): v for k, v in terminal_receipts.items()},
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
            "backup_request": rel(BACKUP_REQUEST),
            "interpretation_limits": ["opened development/stress-pool states only", "not validation64", "not sealed test", "not population estimate", "not ORIGINAL SAC", "H25/H35 include matched terminal availability, so this is a feasibility/comparator probe, not a pure horizon-only ablation"],
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        write_json(BACKUP_REQUEST, {"request": "backup_after_v29_success_aware_longer_H_probe", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "new v29 development simulation evidence, docs/state/registry/response-log", "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"], "new_control_steps": control_steps, "new_simulation_episodes": len(episodes), "new_training_or_gradient_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
        update_docs(raw)
        completed = {"status": "complete", "hard_pass": True, "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": raw["budget_actual"], "headline": analysis["headline"], "decision": analysis["decision"], "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "backup_request": rel(BACKUP_REQUEST), "hashes": {}}
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), V28_PROTOCOL, V28_DONE, V27_RAW, V27_DONE, V27_MANIFEST, V19_RAW, V19_DONE, V19_PROTOCOL, STATE, BACKUP_REQUEST, ROOT / "STATUS.md", ROOT / "EXPERIMENT_REGISTRY.csv", ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"]
        completed["hashes"] = {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}
        write_json(RUN_DIR / "completed.json", completed)
        raw["completed_sha256"] = sha256(RUN_DIR / "completed.json")
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": analysis["headline"], "decision": analysis["decision"], "backup_request": rel(BACKUP_REQUEST), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        write_json(RUN_DIR / "failed.json", {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_success_aware_longer_H_feasibility_probe_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
