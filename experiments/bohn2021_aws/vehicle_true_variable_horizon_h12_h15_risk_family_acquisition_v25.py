#!/usr/bin/env python3
"""v25 H12-risk-family source-independent acquisition.

Development-only IMPROVED experiment after v24.

v24 established that fresh source-independent rows from v21+v23 are a strong
fixed-H12 result (0 H12 catastrophes, ~16% decision saving), while adaptivity is
only required by the opened v19/v11 lower-stress mid-late negative cluster.  This
script freezes the next discriminating test: select unused stress-bank cases in a
source72-neighborhood by metadata only, use only H15 traces to pick branch states
that resemble the v11/source72 mid-late H12-risk morphology, then evaluate fixed
true H12, fixed true H15, oracle H12/H15, and the fixed v20b/v22 history selector.

Modes:
  --dry-run: freeze protocol/case selection and write a backup request; no MPC
             simulation, no validation64, no sealed test.
  --run:     execute the already frozen protocol (or freeze it if absent), with
             H15 Stage-A traces before selector choices and before any H12 branch
             outcome; then run H12/H15 branches under shared H15 terminal.

No gradient training or selector grid search is performed.
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
import vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23 as v23  # noqa:E402

NAME = "vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25"
STAMP = "20260930T0210Z"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0210_after_h12_risk_family_acquisition_v25.md"
DRYRUN_BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_RISK_FAMILY_ACQUISITION_V25_DRYRUN_{STAMP}.json"
RUN_BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_RISK_FAMILY_ACQUISITION_V25_RUN_{STAMP}.json"
DEFAULT_BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260930T015123_from_supervisor_context_after_v24.json"
MARKER_DRYRUN = f"vehicle-h12-h15-risk-family-acquisition-v25-dryrun-{STAMP}"
MARKER_RUN = f"vehicle-h12-h15-risk-family-acquisition-v25-run-{STAMP}"

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
V23_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/completed.json"
V23_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/raw.json"
V23_MANIFEST = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/selected_state_manifest.json"
V24_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/completed.json"
V24_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/raw.json"
V24_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/summary.md"
V11_MANIFEST = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/selected_state_manifest.json"

TRUE_HORIZONS = [12, 15]
SHORT_H = 12
REF_H = 15
PRIMARY_TERMINAL_PROFILE = "shared_h15_terminal"
TARGET_CASES = 8
BRANCH_STATES_PER_CASE = 2
REPEATS = 1
MAX_STEPS = 150
CONTROL_STEP_CAP = (TARGET_CASES + TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS) * REPEATS) * MAX_STEPS
TOTAL_EPISODES = TARGET_CASES + TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS) * REPEATS
RNG_SEED = 202609300210
MIN_SAVE = 0.05
ANCHOR_SOURCE_CANDIDATE_INDEX = 72
ANCHOR_BASE_STATE_ID = "fresh_case05_slot1_mid_late_control"

# Include prior artifacts only to exclude already-used source_candidate_index values
# and document provenance.  H12/H15 outcomes are not used to choose the new cases.
PRIOR_INDEX_PATHS: List[Path] = list(v23.PRIOR_INDEX_PATHS) + [
    v21.PROTOCOL,
    V21_RAW,
    V21_DONE,
    V21_MANIFEST,
    v23.PROTOCOL,
    V23_RAW,
    V23_DONE,
    V23_MANIFEST,
    V24_RAW,
    V24_DONE,
    V11_MANIFEST,
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
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return math.fsum(vals) / len(vals) if vals else 0.0


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
    required = [SOURCE, BANK_PATH, BANK_DONE, STAGE1_DONE, V19_DONE, V19_RAW, V20B_DONE, V20B_RAW, V21_DONE, V21_RAW, V21_MANIFEST, V22_DONE, V22_RAW, V23_DONE, V23_RAW, V23_MANIFEST, V24_DONE, V24_RAW, V24_SUMMARY, V11_MANIFEST]
    for p in required:
        if not p.exists():
            raise ContractError("missing required input: " + rel(p))
    for p, label in [(BANK_DONE, "stress-v1 bank"), (STAGE1_DONE, "stress-v1 stage1"), (V19_DONE, "v19"), (V20B_DONE, "v20b"), (V21_DONE, "v21"), (V22_DONE, "v22"), (V23_DONE, "v23"), (V24_DONE, "v24")]:
        completed_ok(p, label)
    return {rel(p): sha256(p) for p in required}


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


def metadata_scales(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ["theta_r", "abs_theta_r", "traj_steps", "min_reference_obstacle_clearance", "stress_v1_score"]:
        vals = [sf(r.get(k), float("nan")) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        out[k] = max(max(vals) - min(vals), 1e-9) if len(vals) > 1 else 1.0
    return out


def source72_metadata_distance(row: Mapping[str, Any], anchor: Mapping[str, Any], scales: Mapping[str, float]) -> float:
    # Emphasize signed heading and stress/clearance, since v19/v11 risk was
    # localized to a lower-stress mid-late control morphology rather than simply
    # high absolute heading.
    terms = [
        2.0 * ((sf(row.get("theta_r")) - sf(anchor.get("theta_r"))) / max(sf(scales.get("theta_r"), 1.0), 1e-9)) ** 2,
        1.0 * ((sf(row.get("abs_theta_r")) - sf(anchor.get("abs_theta_r"))) / max(sf(scales.get("abs_theta_r"), 1.0), 1e-9)) ** 2,
        1.0 * ((sf(row.get("traj_steps")) - sf(anchor.get("traj_steps"))) / max(sf(scales.get("traj_steps"), 1.0), 1e-9)) ** 2,
        1.5 * ((sf(row.get("min_reference_obstacle_clearance")) - sf(anchor.get("min_reference_obstacle_clearance"))) / max(sf(scales.get("min_reference_obstacle_clearance"), 1.0), 1e-9)) ** 2,
        1.5 * ((sf(row.get("stress_v1_score")) - sf(anchor.get("stress_v1_score"))) / max(sf(scales.get("stress_v1_score"), 1.0), 1e-9)) ** 2,
    ]
    return math.sqrt(math.fsum(terms) / math.fsum([2.0, 1.0, 1.0, 1.5, 1.5]))


def choose_cases_v25(bank: Mapping[str, Any], excluded: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metas = list((bank.get("selection") or {}).get("all_candidate_metadata") or [])
    cases = list(bank.get("candidate_cases") or [])
    if len(metas) != len(cases) or len(metas) < 64:
        raise ContractError("candidate bank dimensions invalid")
    all_rows_by_idx = {si(r.get("candidate_index"), -1): dict(r) for r in metas}
    if ANCHOR_SOURCE_CANDIDATE_INDEX not in all_rows_by_idx:
        raise ContractError("source72 anchor metadata missing from stress bank")
    anchor = all_rows_by_idx[ANCHOR_SOURCE_CANDIDATE_INDEX]
    excluded_set = set(int(x) for x in excluded)
    thr = v8.freeze0.thresholds_from_bank(bank, metas)
    scales = metadata_scales(metas)
    eligible: List[Dict[str, Any]] = []
    for row0 in metas:
        row = dict(row0)
        idx = si(row.get("candidate_index"), -1)
        if idx < 0 or idx >= len(cases) or idx in excluded_set:
            continue
        row["strict_group"] = v8.freeze0.classify(row, thr, relaxed=False)
        row["relaxed_group"] = v8.freeze0.classify(row, thr, relaxed=True)
        row["source72_metadata_distance"] = source72_metadata_distance(row, anchor, scales)
        # Strictly prefer lower-stress controls, but retain relaxed/global fallbacks
        # if the prior exclusion set exhausts that group.
        if row["strict_group"] == "fresh_lower_stress_control":
            row["source72_pool"] = "strict_lower_stress"
            row["pool_rank"] = 0
        elif row["relaxed_group"] == "fresh_lower_stress_control":
            row["source72_pool"] = "relaxed_lower_stress"
            row["pool_rank"] = 1
        else:
            row["source72_pool"] = "global_nearest_fallback"
            row["pool_rank"] = 2
        eligible.append(row)
    if len(eligible) < TARGET_CASES:
        raise ContractError("too few eligible candidate cases after exclusion")
    ordered = sorted(eligible, key=lambda r: (si(r.get("pool_rank"), 9), sf(r.get("source72_metadata_distance")), si(r.get("candidate_index"), 9999)))
    selected_rows = ordered[:TARGET_CASES]
    selected_cases: List[Dict[str, Any]] = []
    for i, row in enumerate(selected_rows):
        idx = si(row.get("candidate_index"), -1)
        case_snapshot = cases[idx]
        selected_cases.append({
            "fresh_case_index": i,
            "source_candidate_index": idx,
            "fresh_confirmation_group": "source72_h12_risk_family",
            "risk_probe_role": f"source72_neighbor_rank{i:02d}",
            "selection_mode": row.get("source72_pool"),
            "selection_boundary_mode": "nearest_unused_source72_metadata_distance",
            "selection_anchor_candidate_index": ANCHOR_SOURCE_CANDIDATE_INDEX,
            "selection_anchor_metadata_distance": row.get("source72_metadata_distance"),
            "strict_group": row.get("strict_group"),
            "relaxed_group": row.get("relaxed_group"),
            "theta_r": sf(row.get("theta_r")),
            "abs_theta_r": sf(row.get("abs_theta_r")),
            "traj_steps": sf(row.get("traj_steps")),
            "min_reference_obstacle_clearance": sf(row.get("min_reference_obstacle_clearance")),
            "stress_v1_score": sf(row.get("stress_v1_score")),
            "case_snapshot_from_candidate_pool": case_snapshot,
            "case_snapshot_sha256": hashlib.sha256(json.dumps(clean(case_snapshot), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
            "selection_rule": "v25 metadata-only source72-neighborhood H12-risk-family acquisition; source excluded from all prior outcome sources; selected before any v25 H12/H15 outcome",
        })
    diag = {
        "anchor_source_candidate_index": ANCHOR_SOURCE_CANDIDATE_INDEX,
        "anchor_metadata": anchor,
        "excluded_source_candidate_indices_count": len(excluded_set),
        "excluded_source_candidate_indices": sorted(excluded_set),
        "eligible_after_exclusion_count": len(eligible),
        "selected_source_candidate_indices": [int(c["source_candidate_index"]) for c in selected_cases],
        "selected_case_distances": [sf(c.get("selection_anchor_metadata_distance")) for c in selected_cases],
        "pool_counts": dict(Counter(str(r.get("source72_pool")) for r in eligible)),
        "strict_group_counts_after_exclusion": dict(Counter(str(r.get("strict_group")) for r in eligible)),
        "thresholds": thr,
        "selection_never_reads_new_H12_outcomes": True,
    }
    return selected_cases, diag


def load_anchor_morphology() -> Dict[str, Any]:
    manifest = read_json(V11_MANIFEST)
    selected = list(manifest.get("selected_states") or [])
    slot1 = [s for s in selected if si(s.get("source_candidate_index"), -1) == ANCHOR_SOURCE_CANDIDATE_INDEX and str(s.get("base_state_id")) == ANCHOR_BASE_STATE_ID]
    if not slot1:
        # Conservative fallback: same source and slot 1.
        slot1 = [s for s in selected if si(s.get("source_candidate_index"), -1) == ANCHOR_SOURCE_CANDIDATE_INDEX and si(s.get("branch_state_slot"), -1) == 1]
    if not slot1:
        raise ContractError("could not locate source72 mid-late anchor in v11 manifest")
    anchor = dict(slot1[0])
    trace_path = ROOT / str(anchor.get("h15_trace_episode_path")) / "trace.json"
    trace_n = None
    if trace_path.exists():
        trace = read_json(trace_path)
        trace_n = len(trace)
    anchor["anchor_trace_length"] = trace_n
    anchor["catastrophic_branch_steps_from_v19"] = [41, 45, 49]
    anchor["selection_note"] = "H15-trace morphology anchor from v11 source72 slot1; H12 outcomes used only as prior motivation, not for selecting new sources or new branch outcomes"
    return anchor


def obs14(raw: Any) -> List[float]:
    vals = [sf(v) for v in (raw or [])[:14]] if isinstance(raw, list) else []
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def branch_state(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return row.get("previous_state") or row.get("state") or {}


def morphology_distance(row: Mapping[str, Any], n: int, anchor: Mapping[str, Any], target_offset: int) -> float:
    a_state = anchor.get("branch_previous_state") or {}
    r_state = branch_state(row)
    a_obs = obs14(anchor.get("initial_observation_from_h15_trace") or [])
    r_obs = obs14(row.get("observation") or row.get("next_observation") or [])
    # Weight observation components that have historically expressed obstacle/heading proximity.
    obs_components = [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13]
    obs_term = mean([(r_obs[i] - a_obs[i]) ** 2 for i in obs_components]) if obs_components else 0.0
    state_term = (
        ((sf(r_state.get("x")) - sf(a_state.get("x"))) / 12.0) ** 2 +
        ((sf(r_state.get("y")) - sf(a_state.get("y"))) / 10.0) ** 2 +
        ((sf(r_state.get("theta")) - sf(a_state.get("theta"))) / 1.5) ** 2
    ) / 3.0
    anchor_n = sf(anchor.get("anchor_trace_length"), 0.0)
    anchor_step = sf(anchor.get("branch_step"), 45.0) + float(target_offset)
    if anchor_n > 0 and n > 0:
        progress_term = ((sf(row.get("step"), 0.0) / float(n)) - (anchor_step / anchor_n)) ** 2
    else:
        progress_term = ((sf(row.get("step"), 0.0) - anchor_step) / 50.0) ** 2
    risk = v21.trace_risk_score(row, n)
    # Lower is better; a small negative risk bonus avoids selecting trivially
    # similar but dynamically quiet states when morphology distances tie.
    return math.sqrt(obs_term + state_term + progress_term) - 0.01 * risk


def make_stage_a_schedule(selected_cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "stage": "A_h15_trace_scan",
        "execution_index": i,
        "fresh_case_index": int(c["fresh_case_index"]),
        "source_candidate_index": int(c["source_candidate_index"]),
        "true_mpc_n_horizon": REF_H,
        "terminal_profile": "matched_terminal",
        "purpose": "collect H15 trace for source72-neighborhood morphology matching before any v25 H12 outcome",
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
                        "blocked_randomization_unit": f"v25|rep{rep}|case{c['fresh_case_index']}|slot{slot}|shared_h15",
                    })
                    exe += 1
    return rows


def build_protocol(created: dt.datetime, selected_cases: Sequence[Mapping[str, Any]], case_diag: Mapping[str, Any], anchor: Mapping[str, Any], input_hashes: Mapping[str, str], backup: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_H12_H15_source72_risk_family_acquisition_not_validation_not_test",
        "primary_question": "Do source-independent stress-bank neighbors of the v11/source72 mid-late H12-risk morphology reproduce fixed-H12 failures, and does the fixed v20b/v22 H12/H15 selector avoid them while retaining measured compute saving over H15 and fixed H12?",
        "before_evidence": [
            "v19: fixed H12 failed only on the opened fresh_v11/source72 mid-late cluster, while H12/H15 oracle passed.",
            "v21+v23: 40 source-independent stress-pool rows had zero H12 catastrophes and fixed H12 saved 16.22%; selector was slower than fixed H12 on v23.",
            "v24 decision: do not run another broad support sweep; target source72-risk-family reproduction pre-outcome from H15 traces.",
        ],
        "selected_cases": selected_cases,
        "case_selection_diagnostics": case_diag,
        "case_selection_rule": "metadata-only nearest unused source72-neighborhood selection from the stress-v1 bank; source_candidate_index values from prior H-outcome artifacts excluded; no v25 H12/H15 outcome read before selection",
        "anchor_morphology": anchor,
        "stage_A_h15_trace_state_selection": {
            "schedule": make_stage_a_schedule(selected_cases),
            "horizon": REF_H,
            "terminal_profile": "matched_terminal",
            "branch_states_per_case": BRANCH_STATES_PER_CASE,
            "eligible_step_bounds": {"min_step": 8, "max_step": 95, "reserve_terminal_margin_steps": 3},
            "windows": [
                {"slot": 0, "name": "source72_morph_precenter", "low_fraction": 0.38, "high_fraction": 0.66, "target_anchor_step_offset": -4, "selection": "minimum_H15_morphology_distance_to_source72_anchor_with_small_risk_bonus"},
                {"slot": 1, "name": "source72_morph_center", "low_fraction": 0.43, "high_fraction": 0.76, "target_anchor_step_offset": 0, "selection": "minimum_H15_morphology_distance_to_source72_anchor_with_min_step_separation_6"},
            ],
            "manifest_rule": "selected_state_manifest is written after all Stage-A H15 traces and before selector choices and before Stage-B H12/H15 outcomes",
        },
        "selector_preoutcome_rule": {
            "deployable_config": dict(v22.DEPLOY_CFG),
            "model_fit_source": "opened v19 H12/H15 rows via v20.load_rows; no grid search, no validation64/test, no v25 labels",
            "feature_source": "Stage-A H15 trace prefix only",
            "choice_manifest": rel(RUN_DIR / "preoutcome_selector_choices.json"),
            "overhead_accounting_primary": "charge measured feature+selector time during preoutcome choice generation once per branch state",
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
            "fresh_fixed_H12_bad_and_selector_safe_pass5": "adaptive mechanism survives this targeted development check; next plan a fresh independent confirmation with fixed H12 primary and strong fixed/per-H terminal baselines",
            "fresh_fixed_H12_bad_and_selector_bad": "static/history selector is insufficient; inspect features and pivot to terminal-risk/value refit or bounded training before any validation",
            "fresh_fixed_H12_zero_bad_and_pass5": "risk-family neighbors still favor fixed H12; adaptivity not justified for this stress-pool family, pivot to scenario/comparison design and fixed-H12-primary confirmation",
            "zero_bad_but_saving_below_5pct": "physical safety alone is insufficient; broaden opportunity or revise objective/features, no speed claim",
        },
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "latest_verified_backup_before_dryrun_or_run": {"path": rel(Path(str(backup.get("path", ""))) if backup.get("path") else DEFAULT_BACKUP_PROOF), "commit": backup.get("commit"), "package_sha256": (backup.get("packages_this_run") or [{}])[0].get("sha256")},
        "input_hashes": input_hashes,
    }
    write_json(PROTOCOL, protocol)
    return protocol


def freeze_protocol_or_load(backup: Mapping[str, Any], force_rebuild: bool = False) -> Mapping[str, Any]:
    if PROTOCOL.exists() and not force_rebuild:
        return read_json(PROTOCOL)
    created = now_utc()
    input_hashes = verify_inputs()
    bank = read_json(BANK_PATH)
    excluded = set(v8.freeze0.selected_stage1_indices(bank))
    excluded.update(source_indices_from_paths(PRIOR_INDEX_PATHS))
    selected_cases, case_diag = choose_cases_v25(bank, sorted(excluded))
    anchor = load_anchor_morphology()
    input_hashes[rel(V11_MANIFEST)] = sha256(V11_MANIFEST)
    return build_protocol(created, selected_cases, case_diag, anchor, input_hashes, backup)


def select_stage_a_states(stage_a_episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cfg = protocol["stage_A_h15_trace_state_selection"]
    bounds = cfg["eligible_step_bounds"]
    windows = list(cfg["windows"])[:BRANCH_STATES_PER_CASE]
    anchor = protocol["anchor_morphology"]
    min_step = si(bounds.get("min_step"), 8)
    max_step_abs = si(bounds.get("max_step"), 95)
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
            lo = max(min_step, int(math.floor(n * sf(window.get("low_fraction"), 0.40))))
            hi = min(max_step_abs, n - reserve - 1, int(math.ceil(n * sf(window.get("high_fraction"), 0.75))))
            candidates = [r for r in trace if lo <= si(r.get("step"), -1) <= hi]
            if slot == 1 and chosen_steps:
                separated = [r for r in candidates if all(abs(si(r.get("step"), -1) - s) >= 6 for s in chosen_steps)]
                if separated:
                    candidates = separated
            if not candidates:
                candidates = [r for r in trace if min_step <= si(r.get("step"), -1) <= min(max_step_abs, n - reserve - 1)]
            if not candidates:
                raise ContractError("no eligible Stage-A state for " + str(ep.get("state_id")))
            target_offset = si(window.get("target_anchor_step_offset"), 0)
            row = min(candidates, key=lambda r: (morphology_distance(r, n, anchor, target_offset), abs(si(r.get("step"), -1) - (sf(anchor.get("branch_step"), 45.0) + target_offset)), si(r.get("step"), 9999)))
            step = si(row.get("step"), -1)
            chosen_steps.append(step)
            base_state_id = f"v25_case{si(ep.get('fresh_case_index'), -1):02d}_slot{slot}_{window.get('name', 'window')}"
            selected.append({
                "base_state_id": base_state_id,
                "fresh_case_index": si(ep.get("fresh_case_index"), -1),
                "source_candidate_index": si(ep.get("source_candidate_index"), -1),
                "fresh_confirmation_group": ep.get("fresh_confirmation_group"),
                "risk_probe_role": ep.get("risk_probe_role"),
                "branch_state_slot": slot,
                "window": window.get("name"),
                "branch_step": step,
                "branch_previous_state": copy.deepcopy(branch_state(row)),
                "initial_observation_from_h15_trace": copy.deepcopy(row.get("observation") or []),
                "h15_trace_episode_path": ep.get("path"),
                "stage_a_trace_risk_score": v21.trace_risk_score(row, n),
                "source72_morphology_distance": morphology_distance(row, n, anchor, target_offset),
                "target_anchor_step_offset": target_offset,
                "selection_rule": "v25: branch state selected from H15 trace by source72-morphology distance before selector choices and before Stage-B H12/H15 outcome",
            })
    if len(selected) != len(stage_a_episodes) * BRANCH_STATES_PER_CASE:
        raise ContractError("unexpected selected state count")
    return selected


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
        "no_v25_h12_outcome_seen_before_choices": True,
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
            "bank_id": "v25",
            "source_key": f"v25/source{si(r.get('source_candidate_index'), -1):03d}/{r['base_state_id']}",
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


def selector_choices(choice_manifest: Mapping[str, Any]) -> Dict[str, int]:
    return {str(d["base_state_id"]): int(d["selected_h_preoutcome"]) for d in choice_manifest.get("choices", [])}


def fixed_choices(rows: Sequence[Mapping[str, Any]], h: int) -> Dict[str, int]:
    return {str(r["candidate_id"]): int(h) for r in rows}


def oracle_choices(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {str(r["candidate_id"]): (12 if bool(r.get("h12_beneficial_vs_h15")) else 15) for r in rows}


def analyze_policies(branch_episodes: Sequence[Mapping[str, Any]], selected_states: Sequence[Mapping[str, Any]], choices: Mapping[str, Any]) -> Dict[str, Any]:
    label_analysis = v21.analyze(branch_episodes, selected_states)
    rows = to_eval_rows(label_analysis["state_rows"])
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
    if fx["catastrophic_false_positive_count"] > 0 and sel["pass5_zero_cat_physical"] and sel["catastrophic_false_positive_count"] == 0:
        decision = "v25 reproduced fresh fixed-H12 risk while the selector stayed safe/pass5; adaptive H12/H15 mechanism merits source-independent confirmation with fixed H12 as primary baseline."
    elif fx["catastrophic_false_positive_count"] > 0 and (sel["catastrophic_false_positive_count"] > 0 or not sel["physical_gate_vs_H15"]):
        decision = "v25 reproduced fixed-H12 risk and the static/history selector failed or lost physical gate; pivot to richer terminal-risk/value refit or bounded training before validation."
    elif fx["pass5_zero_cat_physical"]:
        decision = "v25 source72-neighborhood risk-family rows still favor safe fixed H12; adaptivity is not justified for this stress-pool family, so prioritize scenario/comparison design or fixed-H12-primary confirmation."
    elif oracle["pass5_zero_cat_physical"] and not sel["pass5_zero_cat_physical"]:
        decision = "v25 has oracle H12/H15 opportunity but current selector misses it; pivot to terminal-risk/value refit or representation/training."
    elif sel["pass5_zero_cat_physical"]:
        decision = "v25 selector passes but fixed-H12 comparison is not clean; broaden confirmation with fixed H12 primary and stronger baselines."
    else:
        decision = "v25 shows weak or unsafe H12/H15 opportunity; do not validate unchanged selector; reassess scenario design, terminal objective, or training/value model."
    return {"label_analysis": label_analysis, "evaluation_rows": rows, "policies": policies, "decision": decision, "overhead_charged_s": {"actual_mean": overhead_mean, "v22_mean": v22_mean, "v22_p95": v22_p95}}


def write_dryrun_summary(obj: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle true-variable-H H12/H15 risk-family acquisition v25 dry-run",
        "",
        f"UTC: `{obj['created_utc']}`. Dry-run/protocol freeze only; no simulations, no validation64, no sealed test, no training/refit.",
        "",
        f"Protocol: `{obj['protocol']['path']}` SHA256 `{obj['protocol']['sha256']}`.",
        f"Selected cases: `{len(obj['selected_cases'])}`. Planned episodes: `{obj['budget_declared']['total_episodes_exact']}`, max control steps `{obj['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Selected source72-neighborhood cases",
        "",
        "| fresh_case | source_candidate_index | selection_mode | distance_to_source72 | theta | traj | clearance | stress |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for c in obj["selected_cases"]:
        lines.append("| %d | %d | `%s` | %.6g | %.4g | %.4g | %.4g | %.4g |" % (int(c["fresh_case_index"]), int(c["source_candidate_index"]), c.get("selection_mode"), sf(c.get("selection_anchor_metadata_distance")), sf(c.get("theta_r")), sf(c.get("traj_steps")), sf(c.get("min_reference_obstacle_clearance")), sf(c.get("stress_v1_score"))))
    lines += [
        "",
        "## Next gate",
        "",
        "A verified external backup covering this source/protocol/dry-run is required before running v25 Stage-A/B simulations. The actual run will use the frozen selected cases/protocol and will write the selected_state_manifest after all H15 Stage-A traces but before selector choices and before any H12 branch outcomes.",
        "",
        f"Backup request: `{obj['backup_request']}`.",
    ]
    (RUN_DIR / "dry_run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_summary(raw: Mapping[str, Any]) -> None:
    ag = raw["analysis"]["label_analysis"]["aggregate"]
    pol = raw["analysis"]["policies"]
    fx = pol["fixed_H12"]
    sel = pol["selector_preoutcome_actual_overhead"]
    oracle = pol["oracle_H12_H15"]
    choices = raw["preoutcome_selector_choices"]
    lines = [
        "# Vehicle true-variable-H H12/H15 risk-family acquisition v25",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only source72-neighborhood H12-risk-family acquisition; no validation64, no sealed test, no training/refit.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['total_episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`; selector choices `{raw['budget_actual']['selector_choice_evaluations']}`.",
        "",
        "## Headline",
        "",
        f"- States: `{ag['states']}`; H12-beneficial `{ag['beneficial_H12_count']}`; H12-catastrophic/high-cost `{ag['catastrophic_H12_count']}`; H15 unsafe `{ag['h15_unsafe_count']}`.",
        f"- Fixed H12 vs H15: decision saving `{pct(fx['decision_relative_saving_vs_H15'])}`, solver saving `{pct(fx['solver_relative_saving_vs_H15'])}`, bad `{fx['catastrophic_false_positive_count']}`, physical gate `{fx['physical_gate_vs_H15']}`, pass5 `{fx['pass5_zero_cat_physical']}`.",
        f"- Pre-outcome selector: H counts `{sel['chosen_counts']}`, overhead-adjusted decision saving `{pct(sel['decision_relative_saving_vs_H15'])}`, solver saving `{pct(sel['solver_relative_saving_vs_H15'])}`, bad `{sel['catastrophic_false_positive_count']}`, physical gate `{sel['physical_gate_vs_H15']}`, pass5 `{sel['pass5_zero_cat_physical']}`.",
        f"- Oracle H12/H15: H counts `{oracle['chosen_counts']}`, decision saving `{pct(oracle['decision_relative_saving_vs_H15'])}`, bad `{oracle['catastrophic_false_positive_count']}`, pass5 `{oracle['pass5_zero_cat_physical']}`.",
        f"- Selector feature+decision overhead during preoutcome choice generation: mean `{choices['overhead_summary']['mean_s']:.9f}` s, p95 `{choices['overhead_summary']['p95_s']:.9f}` s over `{choices['overhead_summary']['n']}` states.",
        f"- Decision: {raw['analysis']['decision']}",
        "",
        "## Per-state labels and selector choices",
        "",
        "| state | source | role | selector H | H12 ben | H12 cat | physΔ H12-H15 | H12 dec | H15 dec | H12 solver | H15 solver |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    choice_by = {str(d["base_state_id"]): d for d in choices.get("choices", [])}
    for r in raw["analysis"]["label_analysis"]["state_rows"]:
        ch = choice_by.get(str(r["base_state_id"]), {})
        lines.append("| `%s` | %d | `%s` | %s | `%s` | `%s` | %.6g | %.6g | %.6g | %.6g | %.6g |" % (
            r["base_state_id"], si(r.get("source_candidate_index"), -1), r.get("risk_probe_role"), ch.get("selected_h_preoutcome", "NA"), r.get("h12_beneficial_vs_h15"), r.get("h12_catastrophic_vs_h15"), sf(r.get("physical_delta_h12_minus_h15")), sf((r.get("h12") or {}).get("decision_sum_s")), sf((r.get("h15") or {}).get("decision_sum_s")), sf((r.get("h12") or {}).get("solver_sum_s")), sf((r.get("h15") or {}).get("solver_sum_s"))))
    lines += [
        "",
        "## Limits",
        "",
        "This is targeted development/stress-pool evidence and not validation64, not sealed-test evidence, not a population estimate, and not ORIGINAL SAC. The selector is the fixed v20b/v22 history-family path and no new grid search or gradient training occurred. Any final claim requires fresh independent validation/test and strong fixed-H/per-H terminal baselines.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Protocol: `{rel(PROTOCOL)}`. Backup request: `{rel(RUN_BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def append_registry(marker: str, created_utc: str, phase: str, episodes: int, control_steps: int, selector_choices: int, artifact: Path) -> None:
    path = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker in old:
        return
    line = [created_utc, NAME, phase, f"RNG_SEED={RNG_SEED}", "development_source72_risk_family_no_validation64_no_test", str(episodes), str(control_steps), str(selector_choices), "0", "0", "False", rel(artifact), marker]
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(line)


def update_docs_dryrun(obj: Mapping[str, Any]) -> None:
    block = f"""<!-- {MARKER_DRYRUN} -->
## 2026-09-30 vehicle H12/H15 risk-family acquisition v25 dry-run/protocol freeze

UTC: {obj['created_utc']}. Dry-run only; no simulations, no validation64, no sealed test, no training/refit. Froze source72-neighborhood metadata-only case selection for v25 after verified post-v24 backup. Selected source_candidate_index values: {obj['case_selection_diagnostics']['selected_source_candidate_indices']}. Planned budget: {obj['budget_declared']['total_episodes_exact']} episodes, max {obj['budget_declared']['control_step_upper_bound']} control steps, {obj['budget_declared']['selector_choice_evaluations']} selector choices. Protocol `{obj['protocol']['path']}` sha256 `{obj['protocol']['sha256']}`; dry-run summary `{rel(RUN_DIR / 'dry_run_summary.md')}`. Backup required before actual Stage-A/B simulation: `{obj['backup_request']}`.
"""
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        append_if_missing(ROOT / name, MARKER_DRYRUN, block)
    response = f"""
## Follow-up through v25 dry-run/protocol freeze

Updated by GPT-5.5 executor at `{obj['created_utc']}`. v25 dry-run did not access validation64/sealed test and ran no simulations.

| linked recommendation(s) | disposition after v25 dry-run | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; baked into v25 protocol | Protocol `{obj['protocol']['path']}` makes fixed H12, fixed H15, oracle H12/H15 and v20b/v22 selector the planned comparisons. | After backup, run v25; report fixed H12 first and do not claim adaptivity unless it beats/avoids a fixed-H12 failure. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; v25 remains targeted development | Dry-run selected {len(obj['selected_cases'])} unused source72-neighborhood cases by metadata only from the stress-v1 bank; selected indices {obj['case_selection_diagnostics']['selected_source_candidate_indices']}. | Interpret v25 as risk-family mechanism acquisition, not population validation. |
| `A11_training_failure_modes_need_separation` | accepted; next discriminating experiment frozen | v24 showed fixed H12 dominates broad fresh rows but fails on opened v19 source72 cluster; v25 tests whether that risk reproduces in unused neighbors before choosing refit/training. | If v25 fixed-H12 risk recurs and selector fails, pivot to terminal-risk/value refit or bounded training; if no risk recurs, prioritize scenario/comparison design and fixed-H12-primary confirmation. |
| `A12_registry_backup_schema_contract` | accepted; active | v25 dry-run wrote new source/protocol/dry-run docs and backup request `{obj['backup_request']}`. | Require verified external backup covering v25 dry-run before actual v25 simulations. |
"""
    append_if_missing(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", MARKER_DRYRUN, response)
    append_registry(MARKER_DRYRUN, obj["created_utc"], "development_v25_dryrun_protocol_freeze_no_sim", 0, 0, 0, RUN_DIR / "dry_run.json")


def update_docs_run(raw: Mapping[str, Any]) -> None:
    pol = raw["analysis"]["policies"]
    fx = pol["fixed_H12"]
    sel = pol["selector_preoutcome_actual_overhead"]
    ag = raw["analysis"]["label_analysis"]["aggregate"]
    block = f"""<!-- {MARKER_RUN} -->
## 2026-09-30 vehicle H12/H15 risk-family acquisition v25 run

UTC: {raw['created_utc']}. Development-only source72-neighborhood H12-risk-family acquisition completed; validation64 closed, sealed test closed, no training/refit. Budget {raw['budget_actual']['episodes']} episodes / {raw['budget_actual']['control_steps']} control steps plus {raw['budget_actual']['selector_choice_evaluations']} preoutcome selector decisions. States={ag['states']}, H12-beneficial={ag['beneficial_H12_count']}, H12-catastrophic/high-cost={ag['catastrophic_H12_count']}. Fixed H12 save={fx['decision_relative_saving_vs_H15']}, bad={fx['catastrophic_false_positive_count']}, pass5={fx['pass5_zero_cat_physical']}; selector save={sel['decision_relative_saving_vs_H15']}, H counts={sel['chosen_counts']}, bad={sel['catastrophic_false_positive_count']}, pass5={sel['pass5_zero_cat_physical']}. Decision: {raw['analysis']['decision']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup required before further unique science: `{rel(RUN_BACKUP_REQUEST)}`.
"""
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        append_if_missing(ROOT / name, MARKER_RUN, block)
    response = f"""
## Follow-up through v25 H12-risk-family acquisition run

Updated by GPT-5.5 executor at `{raw['created_utc']}`. v25 did not access validation64/sealed test.

| linked recommendation(s) | disposition after v25 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; updated by v25 | Fixed H12 save {pct(fx['decision_relative_saving_vs_H15'])}, bad {fx['catastrophic_false_positive_count']}, pass5 {fx['pass5_zero_cat_physical']}; selector save {pct(sel['decision_relative_saving_vs_H15'])}, bad {sel['catastrophic_false_positive_count']}, pass5 {sel['pass5_zero_cat_physical']}. | Keep fixed H12 primary in subsequent confirmation. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | v25 is source-independent but deliberately source72-neighborhood stress-pool development evidence. | No population/validation claim from v25 alone. |
| `A11_training_failure_modes_need_separation` | accepted; decision recorded | v25 decision: {raw['analysis']['decision']} | Follow the decision rule: terminal-risk/value refit or training only if fresh risk/selector failure warrants it; otherwise scenario/comparison/fixed-H confirmation. |
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; partially addressed | v25 charged measured preoutcome feature+selector overhead mean {raw['preoutcome_selector_choices']['overhead_summary']['mean_s']:.9f}s over {raw['preoutcome_selector_choices']['overhead_summary']['n']} states. | Still require full closed-loop whole-decision timing for any final speed claim. |
| `A12_registry_backup_schema_contract` | accepted; active | v25 wrote new simulation/raw/docs/registry and backup request `{rel(RUN_BACKUP_REQUEST)}`. | Require verified external backup before further unique science. |
"""
    append_if_missing(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", MARKER_RUN, response)
    append_registry(MARKER_RUN, raw["created_utc"], raw["classification"], raw["budget_actual"]["episodes"], raw["budget_actual"]["control_steps"], raw["budget_actual"]["selector_choice_evaluations"], RUN_DIR / "completed.json")


def dry_run(args: argparse.Namespace) -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    backup = dict(verify_backup(Path(args.backup_proof)))
    backup["path"] = str(Path(args.backup_proof))
    protocol = freeze_protocol_or_load(backup, force_rebuild=bool(args.force_rebuild_protocol))
    created = now_utc()
    protocol_sha = sha256(PROTOCOL)
    obj = {
        "status": "dry_run_complete",
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "classification": "development_IMPROVED_H12_H15_source72_risk_family_acquisition_dryrun_no_sim_no_validation_no_test",
        "protocol": {"path": rel(PROTOCOL), "sha256": protocol_sha},
        "backup_proof_used": {"path": rel(Path(args.backup_proof)), "sha256": sha256(Path(args.backup_proof)), "commit": backup.get("commit")},
        "selected_cases": protocol["selected_cases"],
        "case_selection_diagnostics": protocol["case_selection_diagnostics"],
        "anchor_morphology": protocol["anchor_morphology"],
        "budget_declared": protocol["budget_declared"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "new_refit_grid_evaluations": 0,
        "backup_request": rel(DRYRUN_BACKUP_REQUEST),
    }
    write_json(RUN_DIR / "dry_run.json", obj)
    write_dryrun_summary(obj)
    write_json(DRYRUN_BACKUP_REQUEST, {
        "request": "backup_after_v25_h12_risk_family_dryrun_protocol_freeze",
        "created_utc": created.isoformat(),
        "backup_required_before_more_unique_science": True,
        "reason": "new v25 source/protocol/dry-run docs/state/registry/response-log before actual v25 simulations",
        "must_cover": [rel(SOURCE), rel(PROTOCOL), rel(RUN_DIR / "dry_run.json"), rel(RUN_DIR / "dry_run_summary.md"), rel(STATE), rel(DRYRUN_BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"],
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })
    update_docs_dryrun(obj)
    files = [SOURCE, PROTOCOL, RUN_DIR / "dry_run.json", RUN_DIR / "dry_run_summary.md", DRYRUN_BACKUP_REQUEST, ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"]
    completed = dict(obj)
    completed["hashes"] = {rel(p): sha256(p) for p in files if p.exists()}
    write_json(RUN_DIR / "dry_run_completed.json", completed)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v25 H12-risk-family dry-run\n\nUTC: {created.isoformat()}\n\nDry-run/protocol freeze complete; no simulations, no validation64, no sealed test. Selected source_candidate_index values: {obj['case_selection_diagnostics']['selected_source_candidate_indices']}. Protocol: `{rel(PROTOCOL)}` sha256 `{protocol_sha}`.\n\nNext: wait for/verify external backup covering v25 source/protocol/dry-run artifacts, then run `experiments/bohn2021_aws/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25.py --run --backup-proof <post-v25-dryrun-proof> --i-accept-development-v25` with the legacy interpreter. If v25 fixed-H12 risk recurs and selector fails, pivot to terminal-risk/value refit/training; if no risk recurs, prioritize fixed-H12-primary scenario/comparison confirmation.\n", encoding="utf-8")
    print(json.dumps({"dry_run_completed": rel(RUN_DIR / "dry_run_completed.json"), "summary": rel(RUN_DIR / "dry_run_summary.md"), "protocol": rel(PROTOCOL), "protocol_sha256": protocol_sha, "selected_source_candidate_indices": obj["case_selection_diagnostics"]["selected_source_candidate_indices"], "backup_request": rel(DRYRUN_BACKUP_REQUEST), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


def run_acquisition(args: argparse.Namespace) -> int:
    if (RUN_DIR / "completed.json").exists():
        done = completed_ok(RUN_DIR / "completed.json", "v25 existing")
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    backup = dict(verify_backup(Path(args.backup_proof)))
    backup["path"] = str(Path(args.backup_proof))
    protocol = freeze_protocol_or_load(backup, force_rebuild=False)
    input_hashes = dict(protocol.get("input_hashes") or {})
    input_hashes[rel(Path(args.backup_proof))] = sha256(Path(args.backup_proof))
    created0 = now_utc()
    write_json(RUN_DIR / "run_started.json", {"started_utc": created0.isoformat(), "pid": os.getpid(), "method": NAME, "backup_proof": rel(Path(args.backup_proof)), "protocol_sha256": sha256(PROTOCOL), "validation64_bank_opened": False, "sealed_test_accessed": False, "new_gradient_steps": 0, "new_refit_grid_evaluations": 0})

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

    selected_cases = list(protocol["selected_cases"])
    episodes: List[Dict[str, Any]] = []
    stage_a_eps: List[Dict[str, Any]] = []
    for i, case_meta in enumerate(selected_cases):
        case = case_meta["case_snapshot_from_candidate_pool"]
        item = {
            "execution_index": i,
            "state_id": f"v25_stageA_case{i:02d}_source{int(case_meta['source_candidate_index']):03d}",
            "case": i,
            "source_candidate_index": int(case_meta["source_candidate_index"]),
            "branch_step": 0,
            "branch_previous_state": copy.deepcopy(case.get("state", {"x": 0.0, "y": 0.0, "theta": 0.0})),
            "true_mpc_n_horizon": REF_H,
            "commanded_horizon": REF_H,
            "terminal_mode": "matched_terminal",
            "initialization": "v25_stageA_H15_trace_before_selector_and_H12_outcome",
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
            "execution_index": 6000 + j,
            "state_id": f"v25_B{j:03d}_{st['base_state_id']}_r{rep}_H{h}",
            "case": fc,
            "source_candidate_index": int(case_meta["source_candidate_index"]),
            "branch_step": int(st["branch_step"]),
            "branch_previous_state": copy.deepcopy(st["branch_previous_state"]),
            "true_mpc_n_horizon": h,
            "commanded_horizon": h,
            "terminal_mode": PRIMARY_TERMINAL_PROFILE,
            "initialization": "v25_stageB_direct_branch_state_from_predeclared_H15_manifest",
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
        "backup_request": rel(RUN_BACKUP_REQUEST),
        "interpretation_limits": ["development-only source72-risk-family stress-pool evidence", "not population estimate", "not validation64", "not sealed test", "not ORIGINAL SAC", "fixed v20b/v22 history selector, no new grid refit/gradient training"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_run_summary(raw)
    write_json(RUN_BACKUP_REQUEST, {"request": "backup_after_v25_h12_risk_family_acquisition_run", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "new v25 development simulation evidence, selector choices, docs/state/registry and response log", "must_cover": [rel(SOURCE), rel(PROTOCOL), rel(RUN_DIR), rel(STATE), rel(RUN_BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"], "episodes": len(episodes), "control_steps": control_steps, "validation64_bank_opened": False, "sealed_test_accessed": False})
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
        "backup_request": rel(RUN_BACKUP_REQUEST),
        "hashes": {},
    }
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, RUN_BACKUP_REQUEST, Path(args.backup_proof)]
    completed["hashes"] = {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}
    write_json(RUN_DIR / "completed.json", completed)
    raw["completed_sha256"] = sha256(RUN_DIR / "completed.json")
    write_json(RUN_DIR / "raw.json", raw)
    write_run_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v25 H12-risk-family acquisition\n\nUTC: {created.isoformat()}\n\nDecision: {analysis['decision']}\n\nKey results: states={headline['states']}, H12-beneficial={headline['beneficial_H12_count']}, H12-catastrophic={headline['catastrophic_H12_count']}; fixed H12 save={pct(headline['fixed_H12_save'])}, bad={headline['fixed_H12_bad']}, pass5={headline['fixed_H12_pass5']}; selector save={pct(headline['selector_save'])}, H counts={headline['selector_h_counts']}, bad={headline['selector_bad']}, pass5={headline['selector_pass5']}.\n\nNext: require verified external backup covering v25. If fixed-H12 risk recurred and selector failed, run terminal-risk/value refit or bounded training; if fixed H12 stayed safe, pivot to scenario/comparison design and fixed-H12-primary confirmation. No validation64/sealed test.\n", encoding="utf-8")
    update_docs_run(raw)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": headline, "backup_request": rel(RUN_BACKUP_REQUEST), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="freeze protocol/case selection only; no simulation")
    mode.add_argument("--run", action="store_true", help="execute the frozen v25 acquisition")
    ap.add_argument("--backup-proof", type=str, default=str(DEFAULT_BACKUP_PROOF))
    ap.add_argument("--force-rebuild-protocol", action="store_true", help="only for dry-run before results; rebuild frozen protocol")
    ap.add_argument("--i-accept-development-v25", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_v25:
        raise ContractError("requires --i-accept-development-v25")
    try:
        if args.dry_run:
            return dry_run(args)
        return run_acquisition(args)
    except Exception as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RUN_DIR / "failed.json", {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_H12_H15_source72_risk_family_acquisition_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
