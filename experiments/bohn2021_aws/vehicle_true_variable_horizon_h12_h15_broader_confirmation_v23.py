#!/usr/bin/env python3
"""v23 broader source-independent H12/H15/selector confirmation.

Development-only IMPROVED experiment after v21/v22.

Goal: broaden the source-independent H12/H15 support/negative check with fixed
H12, fixed H15, and the predeclared v20b/v22 history selector all evaluated on
fresh stress-bank sources.  This is not validation64 and not sealed test.

Protocol highlights:
  * verify a post-v22 external-backup proof before doing new unique science;
  * select fresh source_candidate_index values by metadata only from the stress-v1
    candidate bank, excluding all prior H-outcome sources including v21;
  * run H15 Stage-A traces, choose branch states from H15 traces, and write the
    selected-state manifest before any v23 H12 outcome;
  * instantiate the fixed v20b/v22 history selector from opened v19 rows and make
    all v23 selector choices from Stage-A H15 traces before any Stage-B H12/H15
    branch rollout;
  * run blocked true-H12/H15 branch rollouts with shared H15 terminal and measured
    whole-decision/solver timing; compare fixed H12, fixed H15, oracle, and the
    pre-outcome selector, charging measured selector overhead.
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
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_probe_acquisition_v8 as v8  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402
import vehicle_true_variable_horizon_h12_h15_selector_refit_v20 as v20  # noqa:E402
import vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21 as v21  # noqa:E402
import vehicle_true_variable_horizon_h12_h15_online_overhead_v22 as v22  # noqa:E402

NAME = "vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23"
STAMP = "20260930T0145Z"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0145_after_h12_h15_broader_confirmation_v23.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_H15_BROADER_CONFIRMATION_V23_{STAMP}.json"
MARKER = f"vehicle-true-variable-H-h12-h15-broader-confirmation-v23-{STAMP}"

BANK_PATH = v21.BANK_PATH
BANK_DONE = v21.BANK_DONE
STAGE1_DONE = v21.STAGE1_DONE
V19_DONE = v21.V19_DONE
V19_RAW = v21.V19_RAW
V20B_DONE = v21.V20B_DONE
V20B_RAW = v21.V20B_RAW
V21_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/completed.json"
V21_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/raw.json"
V21_MANIFEST = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/selected_state_manifest.json"
V22_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_20260930T0125Z/completed.json"
V22_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_20260930T0125Z/raw.json"
V22_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_20260930T0125Z/summary.md"
POST_V22_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260930T011757_from_supervisor_context_after_v22.json"

TRUE_HORIZONS = [12, 15]
SHORT_H = 12
REF_H = 15
PRIMARY_TERMINAL_PROFILE = "shared_h15_terminal"
REPEATS = 1
TARGET_CASES = 12
BRANCH_STATES_PER_CASE = 2
MAX_STEPS = 150
RNG_SEED = 202609300145
TOTAL_EPISODES = TARGET_CASES + TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS) * REPEATS
CONTROL_STEP_CAP = TOTAL_EPISODES * MAX_STEPS
MIN_DECISION_SAVING = 0.05

# Broader, still source-independent, metadata-only roles.  Anchors are prior
# morphology references only; anchor source_candidate_index values themselves are
# excluded when previously used.
TARGET_ROLES: List[Dict[str, Any]] = [
    {"group": "fresh_lower_stress_control", "role": "lower_stress_near_case72_C", "mode": "nearest_anchor", "anchor": 72},
    {"group": "fresh_lower_stress_control", "role": "lower_stress_near_case72_D", "mode": "nearest_anchor", "anchor": 72},
    {"group": "fresh_lower_stress_control", "role": "lower_stress_low_stress_diverse2", "mode": "low_stress_diverse", "anchor": 72},
    {"group": "fresh_low_heading_low_clearance_control", "role": "low_clearance_near_case190_B", "mode": "nearest_anchor", "anchor": 190},
    {"group": "fresh_low_heading_low_clearance_control", "role": "low_clearance_near_case134_B", "mode": "nearest_anchor", "anchor": 134},
    {"group": "fresh_low_heading_low_clearance_control", "role": "low_clearance_extreme_remaining", "mode": "extreme_low_clearance", "anchor": 190},
    {"group": "fresh_high_heading_long_or_medium", "role": "high_heading_long_near_case74_B", "mode": "nearest_anchor", "anchor": 74},
    {"group": "fresh_high_heading_long_or_medium", "role": "high_heading_long_near_case108_B", "mode": "nearest_anchor", "anchor": 108},
    {"group": "fresh_high_heading_long_or_medium", "role": "high_heading_long_high_stress_remaining", "mode": "high_stress", "anchor": 108},
    {"group": "fresh_high_heading_short", "role": "high_heading_short_near_case242_B", "mode": "nearest_anchor", "anchor": 242},
    {"group": "fresh_high_heading_short", "role": "high_heading_short_high_stress_remaining", "mode": "high_stress", "anchor": 242},
    {"group": None, "role": "global_median_control", "mode": "global_median", "anchor": None},
]

PRIOR_INDEX_PATHS: List[Path] = list(v21.PRIOR_INDEX_PATHS) + [
    v21.PROTOCOL,
    V21_RAW,
    V21_DONE,
    V21_MANIFEST,
    V22_RAW,
    V22_DONE,
]


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sf(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def si(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def pct(x: Any) -> str:
    return "NA" if x is None else f"{100.0 * sf(x):.2f}%"


def mean(xs: Sequence[float]) -> float:
    return math.fsum(float(x) for x in xs) / len(xs) if xs else 0.0


def stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(math.fsum((float(x) - m) ** 2 for x in xs) / (len(xs) - 1))


def q(xs: Sequence[float], quant: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = quant * (len(vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def completed_ok(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing prerequisite: " + rel(path))
    obj = read_json(path)
    ok = obj.get("passed") is True or obj.get("hard_pass") is True or obj.get("status") in ("complete", "completed")
    if not ok:
        raise ContractError(f"prerequisite did not pass/complete: {label} {rel(path)}")
    for key in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(key) is True:
            raise ContractError(f"forbidden {key}=true in prerequisite {label}")
    return obj


def verify_backup(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("backup proof missing: " + rel(path))
    obj = read_json(path)
    if obj.get("status") != "verified" or obj.get("backup_verified") is not True:
        raise ContractError("backup proof is not verified: " + rel(path))
    if obj.get("remaining_changed_files") not in (0, "0"):
        raise ContractError("backup proof reports remaining changed files: " + rel(path))
    if not obj.get("commit") or not obj.get("packages_this_run"):
        raise ContractError("backup proof lacks commit/package evidence: " + rel(path))
    return obj


def verify_inputs() -> Dict[str, str]:
    required = [SOURCE, BANK_PATH, BANK_DONE, STAGE1_DONE, V19_DONE, V19_RAW, V20B_DONE, V20B_RAW, V21_DONE, V21_RAW, V21_MANIFEST, V22_DONE, V22_RAW, V22_SUMMARY, POST_V22_PROOF]
    for p in required:
        if not p.exists():
            raise ContractError("missing required input: " + rel(p))
    for p, label in [(BANK_DONE, "stress-v1 bank"), (STAGE1_DONE, "stress-v1 stage1"), (V19_DONE, "v19"), (V20B_DONE, "v20b"), (V21_DONE, "v21"), (V22_DONE, "v22")]:
        completed_ok(p, label)
    return {rel(p): sha256(p) for p in required}


def metadata_scales(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance", "stress_v1_score"]:
        vals = [sf(r.get(k), float("nan")) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        out[k] = max(max(vals) - min(vals), 1e-9) if len(vals) > 1 else 1.0
    return out


def metadata_distance(a: Mapping[str, Any], b: Mapping[str, Any], scales: Mapping[str, float]) -> float:
    keys = ["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance", "stress_v1_score"]
    return math.sqrt(sum(((sf(a.get(k)) - sf(b.get(k))) / max(sf(scales.get(k), 1.0), 1e-9)) ** 2 for k in keys) / len(keys))


def collect_source_indices(obj: Any) -> List[int]:
    out: List[int] = []
    if isinstance(obj, Mapping):
        if "source_candidate_index" in obj:
            x = si(obj.get("source_candidate_index"), -1)
            if x >= 0:
                out.append(x)
        for v in obj.values():
            out.extend(collect_source_indices(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(collect_source_indices(v))
    return out


def source_indices_from_paths(paths: Sequence[Path]) -> List[int]:
    vals: List[int] = []
    for p in paths:
        if not p.exists():
            continue
        try:
            vals.extend(collect_source_indices(read_json(p)))
        except Exception:
            pass
    return sorted(set(vals))


def choose_cases_v23(bank: Mapping[str, Any], excluded: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metas = list((bank.get("selection") or {}).get("all_candidate_metadata") or [])
    cases = list(bank.get("candidate_cases") or [])
    if len(metas) != len(cases) or len(metas) < 64:
        raise ContractError("candidate bank dimensions invalid")
    excluded_set = set(int(x) for x in excluded)
    thr = v8.freeze0.thresholds_from_bank(bank, metas)
    enriched: List[Dict[str, Any]] = []
    for row0 in metas:
        row = dict(row0)
        idx = si(row.get("candidate_index"), -1)
        if idx < 0 or idx >= len(cases) or idx in excluded_set:
            continue
        row["strict_group"] = v8.freeze0.classify(row, thr, relaxed=False)
        row["relaxed_group"] = v8.freeze0.classify(row, thr, relaxed=True)
        enriched.append(row)
    if len(enriched) < TARGET_CASES:
        raise ContractError("too few eligible candidate cases after exclusion")
    all_rows_by_idx = {si(r.get("candidate_index"), -1): r for r in metas}
    scales = metadata_scales(metas)
    median_ref = {k: statistics.median([sf(r.get(k)) for r in metas]) for k in ["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance", "stress_v1_score"]}
    used: set = set()
    selected_rows: List[Dict[str, Any]] = []
    fallbacks: List[str] = []
    role_diags: List[Dict[str, Any]] = []

    def pool_for(group: Optional[str]) -> Tuple[str, List[Mapping[str, Any]]]:
        if group is None:
            return "global", [r for r in enriched if si(r.get("candidate_index"), -1) not in used]
        strict = [r for r in enriched if r.get("strict_group") == group and si(r.get("candidate_index"), -1) not in used]
        if strict:
            return "strict", strict
        relaxed = [r for r in enriched if r.get("relaxed_group") == group and si(r.get("candidate_index"), -1) not in used]
        if relaxed:
            return "relaxed", relaxed
        return "global_fallback", [r for r in enriched if si(r.get("candidate_index"), -1) not in used]

    def ordered(pool: Sequence[Mapping[str, Any]], role: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        mode = str(role.get("mode"))
        anchor = all_rows_by_idx.get(si(role.get("anchor"), -1)) if role.get("anchor") is not None else None
        if mode == "nearest_anchor" and anchor is not None:
            return sorted(pool, key=lambda r: (metadata_distance(r, anchor, scales), si(r.get("candidate_index"), 9999)))
        if mode == "high_stress":
            return sorted(pool, key=lambda r: (-sf(r.get("stress_v1_score")), -sf(r.get("abs_theta_r")), -sf(r.get("traj_steps")), si(r.get("candidate_index"), 9999)))
        if mode == "low_stress_diverse" and anchor is not None:
            return sorted(pool, key=lambda r: (sf(r.get("stress_v1_score")), -metadata_distance(r, anchor, scales), sf(r.get("abs_theta_r")), si(r.get("candidate_index"), 9999)))
        if mode == "extreme_low_clearance":
            return sorted(pool, key=lambda r: (sf(r.get("min_reference_obstacle_clearance")), -sf(r.get("stress_v1_score")), si(r.get("candidate_index"), 9999)))
        if mode == "global_median":
            return sorted(pool, key=lambda r: (metadata_distance(r, median_ref, scales), si(r.get("candidate_index"), 9999)))
        return sorted(pool, key=lambda r: (si(r.get("candidate_index"), 9999),))

    for role in TARGET_ROLES:
        sel_mode, pool = pool_for(role.get("group"))
        if not pool:
            raise ContractError("empty candidate pool for role " + str(role["role"]))
        if sel_mode not in ("strict", "global"):
            fallbacks.append(f"{role['role']}_{sel_mode}")
        order = ordered(pool, role)
        chosen = dict(order[0])
        idx = si(chosen.get("candidate_index"), -1)
        if idx in used or idx in excluded_set:
            raise ContractError("selection/exclusion invariant failed")
        used.add(idx)
        chosen["fresh_case_index"] = len(selected_rows)
        chosen["fresh_confirmation_group"] = role.get("group") or "global_control"
        chosen["risk_probe_role"] = str(role["role"])
        chosen["selection_mode"] = sel_mode
        chosen["selection_boundary_mode"] = role.get("mode")
        chosen["selection_anchor_candidate_index"] = role.get("anchor")
        anchor_row = all_rows_by_idx.get(si(role.get("anchor"), -1)) if role.get("anchor") is not None else None
        chosen["selection_anchor_metadata_distance"] = metadata_distance(chosen, anchor_row, scales) if anchor_row else metadata_distance(chosen, median_ref, scales)
        selected_rows.append(chosen)
        role_diags.append({
            "role": role["role"],
            "group": role.get("group"),
            "mode": role.get("mode"),
            "selection_mode": sel_mode,
            "pool_size": len(pool),
            "selected_candidate_index": idx,
            "anchor": role.get("anchor"),
            "distance": chosen["selection_anchor_metadata_distance"],
            "selected_strict_group": chosen.get("strict_group"),
            "selected_relaxed_group": chosen.get("relaxed_group"),
        })

    selected_cases: List[Dict[str, Any]] = []
    for row in selected_rows:
        idx = si(row.get("candidate_index"), -1)
        case_snapshot = cases[idx]
        selected_cases.append({
            "fresh_case_index": int(row["fresh_case_index"]),
            "source_candidate_index": idx,
            "fresh_confirmation_group": row["fresh_confirmation_group"],
            "risk_probe_role": row["risk_probe_role"],
            "selection_mode": row.get("selection_mode"),
            "selection_boundary_mode": row.get("selection_boundary_mode"),
            "selection_anchor_candidate_index": row.get("selection_anchor_candidate_index"),
            "selection_anchor_metadata_distance": row.get("selection_anchor_metadata_distance"),
            "strict_group": row.get("strict_group"),
            "relaxed_group": row.get("relaxed_group"),
            "theta_r": sf(row.get("theta_r")),
            "abs_theta_r": sf(row.get("abs_theta_r")),
            "traj_steps": sf(row.get("traj_steps")),
            "min_reference_obstacle_clearance": sf(row.get("min_reference_obstacle_clearance"), 0.0),
            "stress_v1_score": sf(row.get("stress_v1_score")),
            "case_snapshot_from_candidate_pool": case_snapshot,
            "case_snapshot_sha256": hashlib.sha256(json.dumps(clean(case_snapshot), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
            "selection_rule": "v23 metadata-only broader source-independent H12/H15/selector confirmation; selected before any v23 H12/H15 branch outcome",
        })
    diag = {
        "excluded_source_candidate_indices_count": len(excluded_set),
        "excluded_source_candidate_indices": sorted(excluded_set),
        "eligible_after_exclusion_count": len(enriched),
        "selected_source_candidate_indices": [int(c["source_candidate_index"]) for c in selected_cases],
        "fallbacks": fallbacks,
        "target_roles": TARGET_ROLES,
        "role_diagnostics": role_diags,
        "strict_pool_counts": {str(g): sum(1 for r in enriched if r["strict_group"] == g) for g in sorted({r.get("group") for r in TARGET_ROLES if r.get("group")})},
        "relaxed_pool_counts": {str(g): sum(1 for r in enriched if r["relaxed_group"] == g) for g in sorted({r.get("group") for r in TARGET_ROLES if r.get("group")})},
        "thresholds": thr,
    }
    return selected_cases, diag


def make_stage_a_schedule(selected_cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "stage": "A_h15_trace_scan",
        "execution_index": i,
        "fresh_case_index": int(c["fresh_case_index"]),
        "source_candidate_index": int(c["source_candidate_index"]),
        "true_mpc_n_horizon": REF_H,
        "terminal_profile": "matched_terminal",
        "purpose": "collect H15 trace before v23 selector decisions and before any v23 H12 outcome",
    } for i, c in enumerate(selected_cases)]


def make_stage_b_template(selected_cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    rows: List[Dict[str, Any]] = []
    exe = 0
    for rep in range(REPEATS):
        for c in selected_cases:
            for slot in range(BRANCH_STATES_PER_CASE):
                hs = list(TRUE_HORIZONS)
                rng.shuffle(hs)
                for h in hs:
                    rows.append({
                        "stage": "B_shared_h15_trueH12_H15",
                        "execution_index": exe,
                        "repeat": rep,
                        "fresh_case_index": int(c["fresh_case_index"]),
                        "source_candidate_index": int(c["source_candidate_index"]),
                        "branch_state_slot": slot,
                        "terminal_profile": PRIMARY_TERMINAL_PROFILE,
                        "true_mpc_n_horizon": int(h),
                        "blocked_randomization_unit": f"v23|rep{rep}|case{c['fresh_case_index']}|slot{slot}|shared_h15",
                    })
                    exe += 1
    return rows


def build_protocol(created: dt.datetime, selected_cases: Sequence[Mapping[str, Any]], case_diag: Mapping[str, Any], input_hashes: Mapping[str, str], backup: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_broader_source_independent_H12_H15_selector_confirmation_not_validation_not_test",
        "primary_question": "On broader fresh source-independent stress-bank states, does the v20b/v22 H12/H15 selector add safety/value relative to fixed H12 and fixed H15 after selector overhead, or does fixed H12 dominate?",
        "before_evidence": [
            "v19: fixed H12 had large timing saving but was unsafe on a localized fresh_v11 cluster; oracle H12/H15 had value.",
            "v21: fixed H12 was safe and saved 15.05% on 16 fresh source-independent branch states; selector saved 5.41%, indicating fixed H12 is a strong comparator.",
            "v22: selector micro-overhead mean 0.001956693s/p95 0.002653475s and combined v19/v21 proxy still passed, but evidence remains targeted development only.",
        ],
        "selected_cases": selected_cases,
        "case_selection_diagnostics": case_diag,
        "case_selection_rule": "metadata-only selection from stress-v1 bank; excludes source_candidate_index values seen in prior H-outcome/source acquisitions including v21; no v23 H12/H15 branch outcome is read before selection",
        "stage_A_h15_trace_state_selection": {
            "schedule": make_stage_a_schedule(selected_cases),
            "horizon": REF_H,
            "terminal_profile": "matched_terminal",
            "branch_states_per_case": BRANCH_STATES_PER_CASE,
            "eligible_step_bounds": {"min_step": 8, "max_step": 90, "reserve_terminal_margin_steps": 3},
            "windows": [
                {"slot": 0, "name": "early_risk", "low_fraction": 0.12, "high_fraction": 0.34, "selection": "max_h15_trace_risk_score"},
                {"slot": 1, "name": "mid_late_control", "low_fraction": 0.45, "high_fraction": 0.74, "selection": "max_h15_trace_risk_score_with_min_step_separation_8"},
            ],
            "manifest_rule": "selected_state_manifest is written after all Stage-A H15 traces and before v23 selector choices / Stage-B outcomes",
        },
        "selector_preoutcome_rule": {
            "deployable_config": dict(v22.DEPLOY_CFG),
            "model_fit_source": "opened v19 H12/H15 rows via v20.load_rows; no grid search, no validation64/test, no v23 labels",
            "feature_source": "Stage-A H15 trace prefix only",
            "choice_manifest": rel(RUN_DIR / "preoutcome_selector_choices.json"),
            "overhead_accounting_primary": "charge the measured per-state feature+selector elapsed time from preoutcome choice generation once per branch state",
            "overhead_accounting_sensitivity": "also report v22 mean/p95 microbenchmark charge",
        },
        "stage_B_branch_acquisition": {
            "template": make_stage_b_template(selected_cases),
            "true_horizons": TRUE_HORIZONS,
            "terminal_profile": PRIMARY_TERMINAL_PROFILE,
            "repeats": REPEATS,
            "blocked_randomization": "H12/H15 order randomized within case/slot/repeat using frozen RNG seed",
            "primary_measurements": ["success", "constraint", "solver_failure_steps", "physical_constraint_cost", "total_cost", "whole decision timing", "solver timing", "opt_x sizes"],
        },
        "budget_declared": {
            "stage_A_h15_trace_episodes": TARGET_CASES,
            "stage_B_branch_episodes": TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS) * REPEATS,
            "total_episodes_exact": TOTAL_EPISODES,
            "control_step_upper_bound": CONTROL_STEP_CAP,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_grid_evaluations": 0,
            "selector_choice_evaluations": TARGET_CASES * BRANCH_STATES_PER_CASE,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "decision_rule": {
            "selector_false_positive_or_safety_regression": "pivot to richer terminal-risk/value refit or bounded training; no validation",
            "fixed_H12_safe_and_dominates": "treat fixed H12 as primary simple baseline and reassess scenario/adaptivity opportunity; no adaptive claim",
            "selector_safe_pass5_and_fixed_H12_unsafe": "adaptive safety/value mechanism survives broader development check; plan fresh confirmation with strong fixed-H/per-H baselines before validation",
            "both_weak_or_no_saving": "do not run unchanged sweeps; pivot to risk/value/training or scenario design diagnosis",
        },
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "latest_verified_backup_before_run": {"path": rel(Path(str(backup.get("path", ""))) if backup.get("path") else POST_V22_PROOF), "commit": backup.get("commit"), "package_sha256": (backup.get("packages_this_run") or [{}])[0].get("sha256")},
        "input_hashes": input_hashes,
    }
    write_json(PROTOCOL, protocol)
    return protocol


def select_stage_a_states(stage_a_episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cfg = protocol["stage_A_h15_trace_state_selection"]
    bounds = cfg["eligible_step_bounds"]
    windows = list(cfg["windows"])[:BRANCH_STATES_PER_CASE]
    min_step = si(bounds.get("min_step"), 8)
    max_step_abs = si(bounds.get("max_step"), 90)
    reserve = si(bounds.get("reserve_terminal_margin_steps"), 3)
    selected: List[Dict[str, Any]] = []
    for ep in stage_a_episodes:
        trace_path = ROOT / str(ep.get("path")) / "trace.json"
        trace = read_json(trace_path)
        n = len(trace)
        if n < min_step + reserve + 2:
            raise ContractError("Stage-A trace too short: " + str(ep.get("state_id")))
        chosen_steps: List[int] = []
        for window in windows:
            slot = si(window.get("slot"), len(chosen_steps))
            lo = max(min_step, int(math.floor(n * sf(window.get("low_fraction"), 0.1))))
            hi = min(max_step_abs, n - reserve - 1, int(math.ceil(n * sf(window.get("high_fraction"), 0.8))))
            candidates = [r for r in trace if lo <= si(r.get("step"), -1) <= hi]
            if not candidates:
                candidates = [r for r in trace if min_step <= si(r.get("step"), -1) <= min(max_step_abs, n - reserve - 1)]
            if slot == 1 and chosen_steps:
                separated = [r for r in candidates if all(abs(si(r.get("step"), -1) - s) >= 8 for s in chosen_steps)]
                if separated:
                    candidates = separated
            if not candidates:
                raise ContractError("no eligible Stage-A state for " + str(ep.get("state_id")))
            row = max(candidates, key=lambda r: v21.trace_risk_score(r, n))
            step = si(row.get("step"), -1)
            chosen_steps.append(step)
            base_state_id = f"v23_case{si(ep.get('fresh_case_index'), -1):02d}_slot{slot}_{window.get('name', 'window')}"
            selected.append({
                "base_state_id": base_state_id,
                "fresh_case_index": si(ep.get("fresh_case_index"), -1),
                "source_candidate_index": si(ep.get("source_candidate_index"), -1),
                "fresh_confirmation_group": ep.get("fresh_confirmation_group"),
                "risk_probe_role": ep.get("risk_probe_role"),
                "branch_state_slot": slot,
                "window": window.get("name"),
                "branch_step": step,
                "branch_previous_state": copy.deepcopy(row.get("previous_state") or row.get("state")),
                "initial_observation_from_h15_trace": copy.deepcopy(row.get("observation") or []),
                "h15_trace_episode_path": ep.get("path"),
                "stage_a_trace_risk_score": v21.trace_risk_score(row, n),
                "selection_rule": "v23: branch state selected from H15 trace before selector choices and before Stage-B H12/H15 outcome",
            })
    if len(selected) != len(stage_a_episodes) * BRANCH_STATES_PER_CASE:
        raise ContractError("unexpected selected state count")
    return selected


def obs14(raw: Any) -> List[float]:
    vals = [sf(v) for v in (raw or [])[:14]] if isinstance(raw, list) else []
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def make_selector_features(state: Mapping[str, Any], trace_cache: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, float]:
    cand = {
        "candidate_id": str(state["base_state_id"]),
        "base_state_id": str(state["base_state_id"]),
        "branch_previous_state": state.get("branch_previous_state") or {},
        "candidate_branch_step": si(state.get("branch_step"), -1),
        "offset_from_center": 0,
        "source_candidate_index": si(state.get("source_candidate_index"), -1),
        "initial_observation_from_h15_trace": obs14(state.get("initial_observation_from_h15_trace") or []),
        "h15_trace_episode_path": str(state.get("h15_trace_episode_path") or ""),
    }
    fd = v20.static_features(cand)
    trace = trace_cache.get(str(cand["h15_trace_episode_path"])) or []
    v22.add_history_features_from_trace(fd, trace, si(cand.get("candidate_branch_step"), -1))
    return fd


def make_preoutcome_selector_choices(selected_states: Sequence[Mapping[str, Any]], input_hashes: Dict[str, str]) -> Dict[str, Any]:
    v19_rows, v19_features, v19_diag, v19_hashes = v20.load_rows()
    input_hashes.update(v19_hashes)
    trace_cache: Dict[str, Sequence[Mapping[str, Any]]] = {}
    for st in selected_states:
        ep_path = str(st.get("h15_trace_episode_path") or "")
        trace_path = ROOT / ep_path / "trace.json"
        if not trace_path.exists():
            raise ContractError("missing Stage-A trace for selector features: " + rel(trace_path))
        if ep_path not in trace_cache:
            trace_cache[ep_path] = read_json(trace_path)
            input_hashes[rel(trace_path)] = sha256(trace_path)
    model = v22.train_model(v19_rows, v19_features, v22.DEPLOY_CFG)
    details: List[Dict[str, Any]] = []
    for st in selected_states:
        t0 = time.perf_counter_ns()
        fd = make_selector_features(st, trace_cache)
        t1 = time.perf_counter_ns()
        h, score = v22.predict_one(fd, model)
        t2 = time.perf_counter_ns()
        details.append({
            "base_state_id": st["base_state_id"],
            "source_candidate_index": st.get("source_candidate_index"),
            "risk_probe_role": st.get("risk_probe_role"),
            "selected_h_preoutcome": int(h),
            "feature_time_s": (t1 - t0) / 1e9,
            "selector_time_s": (t2 - t1) / 1e9,
            "feature_plus_selector_time_s": (t2 - t0) / 1e9,
            "score": score,
            "feature_count": len(fd),
        })
    times = [sf(d["feature_plus_selector_time_s"]) for d in details]
    result = {
        "created_utc": now_utc().isoformat(),
        "stage_B_started": False,
        "no_v23_h12_outcome_seen_before_choices": True,
        "deployable_config": dict(v22.DEPLOY_CFG),
        "deployable_config_id": v20.cfg_id(v22.DEPLOY_CFG),
        "fit_source": "v19 opened rows via v20.load_rows; no grid search/refit/validation/test",
        "v19_diag": v19_diag,
        "choices": details,
        "choice_counts": dict(Counter(str(d["selected_h_preoutcome"]) for d in details)),
        "overhead_summary": {"n": len(times), "sum_s": math.fsum(times), "mean_s": mean(times), "median_s": q(times, 0.5), "p95_s": q(times, 0.95), "max_s": max(times) if times else 0.0},
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    write_json(RUN_DIR / "preoutcome_selector_choices.json", result)
    return result


def to_eval_rows(state_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in state_rows:
        out.append({
            "candidate_id": str(r["base_state_id"]),
            "bank_id": "v23",
            "source_key": f"v23/source{si(r.get('source_candidate_index'), -1):03d}/{r['base_state_id']}",
            "base_state_id": r["base_state_id"],
            "role": r.get("risk_probe_role"),
            "h12_physical": sf((r.get("h12") or {}).get("physical")),
            "h15_physical": sf((r.get("h15") or {}).get("physical")),
            "h12_decision_sum_s": sf((r.get("h12") or {}).get("decision_sum_s")),
            "h15_decision_sum_s": sf((r.get("h15") or {}).get("decision_sum_s")),
            "h12_solver_sum_s": sf((r.get("h12") or {}).get("solver_sum_s")),
            "h15_solver_sum_s": sf((r.get("h15") or {}).get("solver_sum_s")),
            "h12_steps": sf((r.get("h12") or {}).get("steps"), 1.0),
            "h15_steps": sf((r.get("h15") or {}).get("steps"), 1.0),
            "phys_delta_h12_minus_h15": sf(r.get("physical_delta_h12_minus_h15")),
            "decision_gain_h12_vs_h15_s": sf(r.get("decision_gain_h12_vs_h15_s")),
            "solver_gain_h12_vs_h15_s": sf(r.get("solver_gain_h12_vs_h15_s")),
            "row_physical_tolerance": sf(r.get("row_tolerance_vs_H15"), 2.0),
            "h12_catastrophic_vs_h15": bool(r.get("h12_catastrophic_vs_h15")),
            "h12_beneficial_vs_h15": bool(r.get("h12_beneficial_vs_h15")),
        })
    return out


def fixed_choices(rows: Sequence[Mapping[str, Any]], h: int) -> Dict[str, int]:
    return {str(r["candidate_id"]): int(h) for r in rows}


def oracle_choices(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {str(r["candidate_id"]): (12 if bool(r.get("h12_beneficial_vs_h15")) else 15) for r in rows}


def selector_choices(choice_manifest: Mapping[str, Any]) -> Dict[str, int]:
    return {str(d["base_state_id"]): int(d["selected_h_preoutcome"]) for d in choice_manifest.get("choices", [])}


def analyze_policies(branch_episodes: Sequence[Mapping[str, Any]], selected_states: Sequence[Mapping[str, Any]], choices: Mapping[str, Any]) -> Dict[str, Any]:
    analysis = v21.analyze(branch_episodes, selected_states)
    rows = to_eval_rows(analysis["state_rows"])
    overhead_mean = sf((choices.get("overhead_summary") or {}).get("mean_s"))
    v22_done = read_json(V22_DONE)
    v22_mean = sf(((v22_done.get("headline") or {}).get("overhead_mean_s")), overhead_mean)
    v22_p95 = sf(((v22_done.get("headline") or {}).get("overhead_p95_s")), overhead_mean)
    sc = selector_choices(choices)
    policies = {
        "fixed_H15": v22.evaluate_policy(rows, fixed_choices(rows, 15)),
        "fixed_H12": v22.evaluate_policy(rows, fixed_choices(rows, 12)),
        "oracle_H12_H15": v22.evaluate_policy(rows, oracle_choices(rows)),
        "selector_preoutcome_actual_overhead": v22.evaluate_policy(rows, sc, overhead_per_branch_call_s=overhead_mean),
        "selector_preoutcome_no_overhead": v22.evaluate_policy(rows, sc, overhead_per_branch_call_s=0.0),
        "selector_preoutcome_v22_mean_overhead": v22.evaluate_policy(rows, sc, overhead_per_branch_call_s=v22_mean),
        "selector_preoutcome_v22_p95_overhead": v22.evaluate_policy(rows, sc, overhead_per_branch_call_s=v22_p95),
    }
    sel = policies["selector_preoutcome_actual_overhead"]
    fx = policies["fixed_H12"]
    oracle = policies["oracle_H12_H15"]
    if sel["catastrophic_false_positive_count"] > 0 or not sel["physical_gate_vs_H15"]:
        decision = "v23 selector produced H12 false positives or failed the physical gate; pivot to richer terminal-risk/value refit or bounded training before any validation."
    elif fx["pass5_zero_cat_physical"] and fx["decision_relative_saving_vs_H15"] >= sel["decision_relative_saving_vs_H15"]:
        decision = "v23 fixed H12 is safe and faster than the selector on this broader fresh batch; fixed H12 must be the primary simple baseline and adaptivity is not yet justified for this stress distribution."
    elif sel["pass5_zero_cat_physical"] and fx["catastrophic_false_positive_count"] > 0:
        decision = "v23 selector preserves safety where fixed H12 fails and retains >=5% overhead-adjusted saving; adaptive H12/H15 mechanism merits fresh confirmation with strong fixed-H/per-H baselines."
    elif sel["pass5_zero_cat_physical"]:
        decision = "v23 selector clears the H15-referenced gate, but fixed-H12 comparison remains close; broaden independent confirmation and keep fixed H12 primary before validation."
    elif oracle["pass5_zero_cat_physical"]:
        decision = "v23 oracle has H12/H15 opportunity but the current selector misses too much saving; pivot to terminal-risk/value refit or selector representation rather than validation."
    else:
        decision = "v23 shows weak H12/H15 opportunity on this broader fresh batch; reassess scenario opportunity/design or fixed-H baselines before further selector work."
    return {"label_analysis": analysis, "evaluation_rows": rows, "policies": policies, "decision": decision, "overhead_charged_s": {"actual_mean": overhead_mean, "v22_mean": v22_mean, "v22_p95": v22_p95}}


def write_summary(raw: Mapping[str, Any]) -> None:
    ag = raw["analysis"]["label_analysis"]["aggregate"]
    pol = raw["analysis"]["policies"]
    sel = pol["selector_preoutcome_actual_overhead"]
    fx = pol["fixed_H12"]
    oracle = pol["oracle_H12_H15"]
    choices = raw["preoutcome_selector_choices"]
    lines = [
        "# Vehicle true-variable-H H12/H15 broader confirmation v23",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only IMPROVED broader source-independent H12/H15/selector confirmation; no validation64, no sealed test, no training, no selector grid refit.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['total_episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`; selector choices `{raw['budget_actual']['selector_choice_evaluations']}`.",
        "",
        "## Headline",
        "",
        f"- States: `{ag['states']}`; H12-beneficial `{ag['beneficial_H12_count']}`; H12-catastrophic/high-cost `{ag['catastrophic_H12_count']}`; H15 unsafe `{ag['h15_unsafe_count']}`.",
        f"- Fixed H12 vs H15: decision saving `{pct(fx['decision_relative_saving_vs_H15'])}`, solver saving `{pct(fx['solver_relative_saving_vs_H15'])}`, bad `{fx['catastrophic_false_positive_count']}`, physical gate `{fx['physical_gate_vs_H15']}`, pass5 `{fx['pass5_zero_cat_physical']}`.",
        f"- Pre-outcome selector: H counts `{sel['chosen_counts']}`, overhead-adjusted decision saving `{pct(sel['decision_relative_saving_vs_H15'])}`, solver saving `{pct(sel['solver_relative_saving_vs_H15'])}`, bad `{sel['catastrophic_false_positive_count']}`, physical gate `{sel['physical_gate_vs_H15']}`, pass5 `{sel['pass5_zero_cat_physical']}`.",
        f"- Oracle H12/H15: H counts `{oracle['chosen_counts']}`, decision saving `{pct(oracle['decision_relative_saving_vs_H15'])}`, bad `{oracle['catastrophic_false_positive_count']}`, pass5 `{oracle['pass5_zero_cat_physical']}`.",
        f"- Measured selector feature+decision overhead during preoutcome choice generation: mean `{choices['overhead_summary']['mean_s']:.9f}` s, p95 `{choices['overhead_summary']['p95_s']:.9f}` s, sum `{choices['overhead_summary']['sum_s']:.9f}` s over `{choices['overhead_summary']['n']}` states.",
        f"- Decision: {raw['analysis']['decision']}",
        "",
        "## Selected fresh source cases (pre-outcome metadata)",
        "",
        "| fresh_case | source_candidate_index | group | role | selection | mode | anchor | distance | theta | traj | clearance | stress |",
        "|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in raw.get("selected_cases") or []:
        lines.append("| %d | %d | `%s` | `%s` | `%s` | `%s` | %s | %.4g | %.4g | %.4g | %.4g | %.4g |" % (
            int(c.get("fresh_case_index", -1)), int(c.get("source_candidate_index", -1)), c.get("fresh_confirmation_group"), c.get("risk_probe_role"), c.get("selection_mode"), c.get("selection_boundary_mode"), str(c.get("selection_anchor_candidate_index")), sf(c.get("selection_anchor_metadata_distance")), sf(c.get("theta_r")), sf(c.get("traj_steps")), sf(c.get("min_reference_obstacle_clearance")), sf(c.get("stress_v1_score"))))
    lines += [
        "",
        "## Per-state labels and selector choices",
        "",
        "| state | role | selector H | H12 ben | H12 cat | physΔ H12-H15 | H12 dec | H15 dec | H12 solver | H15 solver |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    choice_by = {str(d["base_state_id"]): d for d in choices.get("choices", [])}
    for r in raw["analysis"]["label_analysis"]["state_rows"]:
        ch = choice_by.get(str(r["base_state_id"]), {})
        lines.append("| `%s` | `%s` | %s | `%s` | `%s` | %.6g | %.6g | %.6g | %.6g | %.6g |" % (
            r["base_state_id"], r.get("risk_probe_role"), ch.get("selected_h_preoutcome", "NA"), r.get("h12_beneficial_vs_h15"), r.get("h12_catastrophic_vs_h15"), sf(r.get("physical_delta_h12_minus_h15")), sf((r.get("h12") or {}).get("decision_sum_s")), sf((r.get("h15") or {}).get("decision_sum_s")), sf((r.get("h12") or {}).get("solver_sum_s")), sf((r.get("h15") or {}).get("solver_sum_s"))))
    lines += [
        "",
        "## Limits",
        "",
        "This is opened development/stress-pool evidence, not a population estimate, not validation64, not sealed-test evidence, and not ORIGINAL SAC. The selector is a fixed v20b/v22 history-family path instantiated from opened v19 rows; no new RL gradient training or selector grid search occurred. Any final claim still requires fresh independent validation/test and strong fixed-H/per-H terminal baselines.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Protocol: `{rel(PROTOCOL)}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def update_docs(raw: Mapping[str, Any]) -> None:
    pol = raw["analysis"]["policies"]
    sel = pol["selector_preoutcome_actual_overhead"]
    fx = pol["fixed_H12"]
    ag = raw["analysis"]["label_analysis"]["aggregate"]
    block = f"""<!-- {MARKER} -->
## 2026-09-30 vehicle true-variable-H H12/H15 broader confirmation v23

UTC: {raw['created_utc']}. Development-only broader source-independent H12/H15/selector confirmation completed after verified post-v22 backup. No validation64, no sealed test, no training/refit. Budget {raw['budget_actual']['episodes']} episodes / {raw['budget_actual']['control_steps']} control steps plus {raw['budget_actual']['selector_choice_evaluations']} preoutcome selector decisions. States={ag['states']}, H12-beneficial={ag['beneficial_H12_count']}, H12-catastrophic/high-cost={ag['catastrophic_H12_count']}. Fixed H12 save={fx['decision_relative_saving_vs_H15']}, bad={fx['catastrophic_false_positive_count']}, pass5={fx['pass5_zero_cat_physical']}; selector save={sel['decision_relative_saving_vs_H15']}, H counts={sel['chosen_counts']}, bad={sel['catastrophic_false_positive_count']}, pass5={sel['pass5_zero_cat_physical']}. Decision: {raw['analysis']['decision']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup required before further unique science: `{rel(BACKUP_REQUEST)}`.
"""
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        append_if_missing(ROOT / name, MARKER, block)
    response_path = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
    response_block = f"""
## Follow-up through v23 broader H12/H15 confirmation

Updated by GPT-5.5 executor at `{raw['created_utc']}`. Stable Astra IDs are preserved; v23 did not access validation64 or sealed test.

| linked recommendation(s) | disposition after v23 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; further addressed for pre-outcome selector choice overhead on fresh branch states | v23 preoutcome selector manifest measured feature+decision overhead mean {raw['preoutcome_selector_choices']['overhead_summary']['mean_s']:.9f}s and p95 {raw['preoutcome_selector_choices']['overhead_summary']['p95_s']:.9f}s over {raw['preoutcome_selector_choices']['overhead_summary']['n']} states; policy evaluation charged measured overhead. | Still no final speed claim; future confirmation/validation must measure full closed-loop whole-decision timing. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; fixed H12 remains primary comparator | v23 fixed H12 save {pct(fx['decision_relative_saving_vs_H15'])}, bad {fx['catastrophic_false_positive_count']}, pass5 {fx['pass5_zero_cat_physical']}; selector save {pct(sel['decision_relative_saving_vs_H15'])}, bad {sel['catastrophic_false_positive_count']}, pass5 {sel['pass5_zero_cat_physical']}. | Do not compare only against H15. If fixed H12 dominates safely, adaptivity/scenario opportunity must be reassessed; if fixed H12 fails and selector remains safe, plan fresh confirmation with fixed H12 primary. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | v23 selected {len(raw.get('selected_cases') or [])} new source_candidate_index values by metadata only and excluded prior sources, but still used the stress-v1 diagnostic pool. | Treat as broader development confirmation only, not population validation. |
| `A11_training_failure_modes_need_separation` | accepted; conditional | v23 decision: {raw['analysis']['decision']} | If v23 shows false positives or missed oracle opportunity, pivot to terminal-risk/value refit/training; if fixed H12 safely dominates, investigate scenario/comparison design rather than forcing adaptive switching. |
| `A12_registry_backup_schema_contract` | accepted; active | v23 wrote new source/results/docs/state/registry and backup request `{rel(BACKUP_REQUEST)}`. | Require verified external backup covering v23 before further unique science. |
"""
    append_if_missing(response_path, MARKER, response_block)


def update_registry(raw: Mapping[str, Any]) -> None:
    path = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if MARKER in old:
        return
    line = [
        raw["created_utc"],
        NAME,
        raw["classification"],
        f"RNG_SEED={RNG_SEED}",
        "development_stress_pool_source_independent_no_validation64_no_test",
        str(raw["budget_actual"]["episodes"]),
        str(raw["budget_actual"]["control_steps"]),
        str(raw["budget_actual"]["selector_choice_evaluations"]),
        "0",
        "0",
        "False",
        rel(RUN_DIR / "completed.json"),
        MARKER,
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(line)


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-proof", type=str, default=str(POST_V22_PROOF))
    ap.add_argument("--i-accept-development-v23", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_v23:
        raise ContractError("requires --run and --i-accept-development-v23")
    if (RUN_DIR / "completed.json").exists():
        done = completed_ok(RUN_DIR / "completed.json", "v23 existing")
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name not in ("run.lock", "failed.json") for p in RUN_DIR.iterdir()):
        raise ContractError("partial output exists; inspect before rerun: " + rel(RUN_DIR))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    backup = dict(verify_backup(Path(args.backup_proof)))
    backup["path"] = str(Path(args.backup_proof))
    created0 = now_utc()
    input_hashes = verify_inputs()
    input_hashes[rel(Path(args.backup_proof))] = sha256(Path(args.backup_proof))
    bank = read_json(BANK_PATH)
    excluded = set(v8.freeze0.selected_stage1_indices(bank))
    excluded.update(source_indices_from_paths(PRIOR_INDEX_PATHS))
    selected_cases, case_diag = choose_cases_v23(bank, sorted(excluded))
    protocol = build_protocol(created0, selected_cases, case_diag, input_hashes, backup)
    write_json(RUN_DIR / "run_started.json", {"started_utc": created0.isoformat(), "pid": os.getpid(), "method": NAME, "backup_proof": rel(Path(args.backup_proof)), "validation64_bank_opened": False, "sealed_test_accessed": False, "new_gradient_steps": 0, "new_refit_grid_evaluations": 0})

    v8.engine.case_runner.SMOKE_DIR = RUN_DIR
    _, stage1_runner, _ = v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
    stage1_runner.base.v1.latency_verify()
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    if REF_H not in terminals:
        raise ContractError("terminal grid missing H15 shared terminal")
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})

    episodes: List[Dict[str, Any]] = []
    stage_a_eps: List[Dict[str, Any]] = []
    for i, case_meta in enumerate(selected_cases):
        case = case_meta["case_snapshot_from_candidate_pool"]
        item = {
            "execution_index": i,
            "state_id": f"v23_stageA_case{i:02d}_source{int(case_meta['source_candidate_index']):03d}",
            "case": i,
            "source_candidate_index": int(case_meta["source_candidate_index"]),
            "branch_step": 0,
            "branch_previous_state": copy.deepcopy(case.get("state", {"x": 0.0, "y": 0.0, "theta": 0.0})),
            "true_mpc_n_horizon": REF_H,
            "commanded_horizon": REF_H,
            "terminal_mode": "matched_terminal",
            "initialization": "v23_stageA_H15_trace_before_selector_and_H12_outcome",
        }
        summary = v8.engine.case_runner.run_true_h_episode(item, case, terminals[REF_H])
        summary["stage"] = "A_h15_trace_scan"
        summary["fresh_case_index"] = i
        summary["fresh_confirmation_group"] = case_meta.get("fresh_confirmation_group")
        summary["risk_probe_role"] = case_meta.get("risk_probe_role")
        stage_a_eps.append(summary)
        episodes.append(summary)
        write_json(RUN_DIR / "progress.json", {"stage": "A", "episodes_done": len(episodes), "episodes_expected": TOTAL_EPISODES, "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "validation64_bank_opened": False, "sealed_test_accessed": False})

    selected_states = select_stage_a_states(stage_a_eps, protocol)
    role_by_case = {int(c["fresh_case_index"]): c.get("risk_probe_role") for c in selected_cases}
    group_by_case = {int(c["fresh_case_index"]): c.get("fresh_confirmation_group") for c in selected_cases}
    for st in selected_states:
        st["risk_probe_role"] = role_by_case.get(si(st.get("fresh_case_index"), -1))
        st["fresh_confirmation_group"] = group_by_case.get(si(st.get("fresh_case_index"), -1))
    manifest = {"created_utc": now_utc().isoformat(), "stage_A_complete_before_selector_and_stage_B": True, "selected_states": selected_states, "protocol_sha256": sha256(PROTOCOL), "validation64_bank_opened": False, "sealed_test_accessed": False}
    write_json(RUN_DIR / "selected_state_manifest.json", manifest)
    write_json(RUN_DIR / "manifest_completed_before_selector_and_stage_B.json", {"created_utc": now_utc().isoformat(), "selected_state_manifest": rel(RUN_DIR / "selected_state_manifest.json"), "selector_started": False, "stage_B_started": False, "validation64_bank_opened": False, "sealed_test_accessed": False})

    preoutcome_choices = make_preoutcome_selector_choices(selected_states, input_hashes)
    write_json(RUN_DIR / "selector_choices_completed_before_stage_B.json", {"created_utc": now_utc().isoformat(), "preoutcome_selector_choices": rel(RUN_DIR / "preoutcome_selector_choices.json"), "stage_B_started": False, "validation64_bank_opened": False, "sealed_test_accessed": False})

    state_by_case_slot = {(si(s.get("fresh_case_index"), -1), si(s.get("branch_state_slot"), -1)): s for s in selected_states}
    branch_template = protocol["stage_B_branch_acquisition"]["template"]
    branch_episodes: List[Dict[str, Any]] = []
    write_json(RUN_DIR / "stage_B_started.json", {"started_utc": now_utc().isoformat(), "selected_state_manifest_preexisting": True, "selector_choices_preexisting": True, "validation64_bank_opened": False, "sealed_test_accessed": False})
    for j, tmpl in enumerate(branch_template):
        fc = si(tmpl.get("fresh_case_index"), -1)
        slot = si(tmpl.get("branch_state_slot"), -1)
        st = state_by_case_slot.get((fc, slot))
        if st is None:
            raise ContractError("missing selected state for branch template")
        case_meta = selected_cases[fc]
        case = case_meta["case_snapshot_from_candidate_pool"]
        h = si(tmpl.get("true_mpc_n_horizon"), -1)
        rep = si(tmpl.get("repeat"), 0)
        item = {
            "execution_index": 5000 + j,
            "state_id": f"v23_B{j:03d}_{st['base_state_id']}_r{rep}_H{h}",
            "case": fc,
            "source_candidate_index": int(case_meta["source_candidate_index"]),
            "branch_step": int(st["branch_step"]),
            "branch_previous_state": copy.deepcopy(st["branch_previous_state"]),
            "true_mpc_n_horizon": h,
            "commanded_horizon": h,
            "terminal_mode": PRIMARY_TERMINAL_PROFILE,
            "initialization": "v23_stageB_direct_branch_state_from_predeclared_H15_manifest",
        }
        summary = v8.engine.case_runner.run_true_h_episode(item, case, terminals[REF_H])
        summary["stage"] = "B_shared_h15_trueH12_H15"
        summary["template_execution_index"] = int(tmpl.get("execution_index", j))
        summary["repeat"] = rep
        summary["fresh_case_index"] = fc
        summary["base_state_id"] = st["base_state_id"]
        summary["branch_state_slot"] = slot
        summary["terminal_profile"] = PRIMARY_TERMINAL_PROFILE
        summary["terminal_source_horizon"] = REF_H
        summary["fresh_confirmation_group"] = case_meta.get("fresh_confirmation_group")
        summary["risk_probe_role"] = case_meta.get("risk_probe_role")
        branch_episodes.append(summary)
        episodes.append(summary)
        write_json(RUN_DIR / "progress.json", {"stage": "B", "episodes_done": len(episodes), "episodes_expected": TOTAL_EPISODES, "stage_B_done": len(branch_episodes), "stage_B_expected": len(branch_template), "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "validation64_bank_opened": False, "sealed_test_accessed": False})

    control_steps = int(sum(si(e.get("steps")) for e in episodes))
    if len(episodes) != TOTAL_EPISODES or control_steps > CONTROL_STEP_CAP:
        raise ContractError("budget mismatch")
    analysis = analyze_policies(branch_episodes, selected_states, preoutcome_choices)
    created = now_utc()
    raw = {
        "created_utc": created.isoformat(),
        "started_utc": created0.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "classification": protocol["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "mobile_robot_mppi_resumed": False,
        "protocol": {"json": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "backup_proof_used": {"path": rel(Path(args.backup_proof)), "sha256": sha256(Path(args.backup_proof)), "commit": backup.get("commit")},
        "input_hashes": input_hashes,
        "budget_declared": protocol["budget_declared"],
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "selector_choice_evaluations": len(preoutcome_choices.get("choices", [])), "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_grid_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "selected_cases": selected_cases,
        "selected_state_manifest": rel(RUN_DIR / "selected_state_manifest.json"),
        "preoutcome_selector_choices": preoutcome_choices,
        "episodes": episodes,
        "analysis": analysis,
        "backup_request": rel(BACKUP_REQUEST),
        "interpretation_limits": ["development-only stress-pool evidence", "not population estimate", "not validation64", "not sealed test", "not ORIGINAL SAC", "fixed v20b/v22 history selector, no new grid refit/gradient training"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    write_json(BACKUP_REQUEST, {"request": "backup_after_v23_broader_h12_h15_confirmation", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "new v23 development simulation evidence, selector choices, docs/state/registry and response log", "must_cover": [rel(SOURCE), rel(PROTOCOL), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", rel(POST_V22_PROOF)], "episodes": len(episodes), "control_steps": control_steps, "validation64_bank_opened": False, "sealed_test_accessed": False})
    pol = analysis["policies"]
    headline = {
        "states": analysis["label_analysis"]["aggregate"]["states"],
        "beneficial_H12_count": analysis["label_analysis"]["aggregate"]["beneficial_H12_count"],
        "catastrophic_H12_count": analysis["label_analysis"]["aggregate"]["catastrophic_H12_count"],
        "fixed_H12_save": pol["fixed_H12"]["decision_relative_saving_vs_H15"],
        "fixed_H12_bad": pol["fixed_H12"]["catastrophic_false_positive_count"],
        "fixed_H12_pass5": pol["fixed_H12"]["pass5_zero_cat_physical"],
        "selector_save": pol["selector_preoutcome_actual_overhead"]["decision_relative_saving_vs_H15"],
        "selector_bad": pol["selector_preoutcome_actual_overhead"]["catastrophic_false_positive_count"],
        "selector_pass5": pol["selector_preoutcome_actual_overhead"]["pass5_zero_cat_physical"],
        "selector_h_counts": pol["selector_preoutcome_actual_overhead"]["chosen_counts"],
        "decision": analysis["decision"],
    }
    completed = {
        "status": "complete",
        "hard_pass": bool(pol["selector_preoutcome_actual_overhead"]["pass5_zero_cat_physical"] or pol["fixed_H12"]["pass5_zero_cat_physical"]),
        "created_utc": created.isoformat(),
        "classification": protocol["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "budget_actual": raw["budget_actual"],
        "headline": headline,
        "backup_request": rel(BACKUP_REQUEST),
        "hashes": {},
    }
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, BACKUP_REQUEST, Path(args.backup_proof)]
    completed["hashes"] = {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}
    write_json(RUN_DIR / "completed.json", completed)
    raw["completed_sha256"] = sha256(RUN_DIR / "completed.json")
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v23 broader H12/H15 confirmation\n\nUTC: {created.isoformat()}\n\nDecision: {analysis['decision']}\n\nKey results: states={headline['states']}, H12-beneficial={headline['beneficial_H12_count']}, H12-catastrophic={headline['catastrophic_H12_count']}; fixed H12 save={pct(headline['fixed_H12_save'])}, bad={headline['fixed_H12_bad']}, pass5={headline['fixed_H12_pass5']}; selector save={pct(headline['selector_save'])}, H counts={headline['selector_h_counts']}, bad={headline['selector_bad']}, pass5={headline['selector_pass5']}.\n\nNext: require verified external backup covering v23. If fixed H12 dominates safely, reassess scenario/adaptivity opportunity and plan fixed-H12-primary confirmation rather than forcing switching; if selector avoids fixed-H12 negatives with net saving, plan fresh independent confirmation with strong fixed-H/per-H baselines; if selector false positives occur, pivot to terminal-risk/value refit or bounded training. No validation64/sealed test.\n", encoding="utf-8")
    update_docs(raw)
    update_registry(raw)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": headline, "backup_request": rel(BACKUP_REQUEST), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RUN_DIR / "failed.json", {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_broader_source_independent_H12_H15_selector_confirmation_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
