#!/usr/bin/env python3
"""v25 H12-risk-family source-independent acquisition.

Development-only IMPROVED experiment prepared after v24.  v24 showed that
fresh source-independent stress-pool rows (v21+v23) are mostly a fixed-H12
compute-saving result, whereas H12/H15 adaptivity is only motivated by the
opened v19/v11 lower-stress mid-late cluster where fixed H12 is high-cost.

This runner therefore asks a narrower falsifiable question: do v19/v11-like
H12-risk states reproduce in fresh source-independent source72-neighborhood
cases when branch states are selected from H15 traces before any H12 outcome?

Safeguards:
  * requires a verified external backup proof strictly after v24 completion;
  * no validation64 or sealed final-test access;
  * selects source_candidate_index values from stress-v1 metadata only;
  * excludes all prior H-outcome/source indices available in v21/v23/v24 inputs;
  * writes the H15-trace branch-state manifest and selector choices before Stage B;
  * compares fixed H12, fixed H15, oracle H12/H15 and the fixed v20b/v22 selector;
  * no gradient training and no selector grid refit.
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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_probe_acquisition_v8 as v8  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402
import vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21 as v21  # noqa:E402
import vehicle_true_variable_horizon_h12_h15_online_overhead_v22 as v22  # noqa:E402
import vehicle_true_variable_horizon_h12_h15_selector_refit_v20 as v20  # noqa:E402
import vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23 as v23  # noqa:E402

NAME = "vehicle_true_variable_horizon_h12_risk_family_acquisition_v25"
STAMP = "20260930T0220Z"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0220_after_h12_risk_family_acquisition_v25.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_RISK_FAMILY_ACQUISITION_V25_{STAMP}.json"
MARKER = f"vehicle-h12-risk-family-acquisition-v25-{STAMP}"

BANK_PATH = v21.BANK_PATH
BANK_DONE = v21.BANK_DONE
STAGE1_DONE = v21.STAGE1_DONE
V11_MANIFEST = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/selected_state_manifest.json"
V11_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/summary.md"
V19_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/completed.json"
V19_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/raw.json"
V21_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/completed.json"
V21_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/raw.json"
V21_MANIFEST = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/selected_state_manifest.json"
V22_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_20260930T0125Z/completed.json"
V23_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/completed.json"
V23_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/raw.json"
V23_MANIFEST = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/selected_state_manifest.json"
V24_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/completed.json"
V24_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/raw.json"
V24_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/summary.md"

TRUE_HORIZONS = [12, 15]
SHORT_H = 12
REF_H = 15
PRIMARY_TERMINAL_PROFILE = "shared_h15_terminal"
TARGET_CASES = 10
BRANCH_STATES_PER_CASE = 3
REPEATS = 1
MAX_STEPS = 150
RNG_SEED = 202609300220
TOTAL_EPISODES = TARGET_CASES + TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS) * REPEATS
CONTROL_STEP_CAP = TOTAL_EPISODES * MAX_STEPS
MIN_DECISION_SAVING = 0.05
SOURCE_ANCHOR_INDEX = 72
NEGATIVE_ANCHOR_STATE_ID = "fresh_case05_slot1_mid_late_control"


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


def mean(xs: Sequence[float]) -> float:
    return math.fsum(float(x) for x in xs) / len(xs) if xs else 0.0


def q(xs: Sequence[float], quant: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = quant * (len(vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    return vals[lo] if lo == hi else vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def pct(x: Any) -> str:
    return "NA" if x is None else f"{100.0 * sf(x):.2f}%"


def parse_time(obj: Mapping[str, Any]) -> Optional[dt.datetime]:
    for key in ("time", "time_utc", "created_utc", "timestamp"):
        val = obj.get(key)
        if not val:
            continue
        try:
            s = str(val).replace("Z", "+00:00")
            out = dt.datetime.fromisoformat(s)
            return out if out.tzinfo else out.replace(tzinfo=dt.timezone.utc)
        except Exception:
            pass
    return None


def verify_backup_post_v24(path: Path) -> Mapping[str, Any]:
    proof = v23.verify_backup(path)
    v24 = read_json(V24_DONE)
    proof_time = parse_time(proof)
    v24_time = parse_time(v24)
    if proof_time is None or v24_time is None:
        raise ContractError("cannot verify proof/v24 timestamps")
    if proof_time <= v24_time:
        raise ContractError(f"backup proof {rel(path)} predates or equals v24 completion ({proof_time.isoformat()} <= {v24_time.isoformat()})")
    return proof


def completed_ok(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing prerequisite: " + rel(path))
    obj = read_json(path)
    ok = obj.get("passed") is True or obj.get("hard_pass") is True or obj.get("status") in ("complete", "completed")
    if not ok:
        raise ContractError("prerequisite not complete: " + label)
    for key in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(key) is True:
            raise ContractError(f"forbidden {key}=true in {label}")
    return obj


def verify_inputs() -> Dict[str, str]:
    required = [SOURCE, BANK_PATH, BANK_DONE, STAGE1_DONE, V11_MANIFEST, V11_SUMMARY, V19_DONE, V19_RAW, V21_DONE, V21_RAW, V21_MANIFEST, V22_DONE, V23_DONE, V23_RAW, V23_MANIFEST, V24_DONE, V24_RAW, V24_SUMMARY]
    for path in required:
        if not path.exists():
            raise ContractError("missing required input: " + rel(path))
    for path, label in [(BANK_DONE, "stress bank"), (STAGE1_DONE, "stress stage1"), (V19_DONE, "v19"), (V21_DONE, "v21"), (V22_DONE, "v22"), (V23_DONE, "v23"), (V24_DONE, "v24")]:
        completed_ok(path, label)
    return {rel(p): sha256(p) for p in required}


def metadata_scales(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance", "stress_v1_score"]:
        vals = [sf(r.get(key), float("nan")) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        out[key] = max(max(vals) - min(vals), 1e-9) if len(vals) > 1 else 1.0
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
        for value in obj.values():
            out.extend(collect_source_indices(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(collect_source_indices(value))
    return out


def prior_source_indices(bank: Mapping[str, Any]) -> List[int]:
    paths: List[Path] = list(v21.PRIOR_INDEX_PATHS) + [
        v21.PROTOCOL, V21_RAW, V21_MANIFEST, V22_DONE,
        v23.PROTOCOL, V23_RAW, V23_MANIFEST, V24_RAW, V24_DONE,
    ]
    excluded = set(v8.freeze0.selected_stage1_indices(bank))
    for p in paths:
        if not p.exists():
            continue
        try:
            excluded.update(collect_source_indices(read_json(p)))
        except Exception:
            pass
    return sorted(excluded)


def choose_cases(bank: Mapping[str, Any], excluded: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metas = list((bank.get("selection") or {}).get("all_candidate_metadata") or [])
    cases = list(bank.get("candidate_cases") or [])
    if len(metas) != len(cases) or len(metas) <= SOURCE_ANCHOR_INDEX:
        raise ContractError("invalid stress-v1 candidate bank dimensions")
    excluded_set = set(int(x) for x in excluded)
    thresholds = v8.freeze0.thresholds_from_bank(bank, metas)
    scales = metadata_scales(metas)
    anchor = metas[SOURCE_ANCHOR_INDEX]
    enriched: List[Dict[str, Any]] = []
    for row0 in metas:
        row = dict(row0)
        idx = si(row.get("candidate_index"), -1)
        if idx < 0 or idx >= len(cases) or idx in excluded_set:
            continue
        strict = v8.freeze0.classify(row, thresholds, relaxed=False)
        relaxed = v8.freeze0.classify(row, thresholds, relaxed=True)
        row["strict_group"] = strict
        row["relaxed_group"] = relaxed
        d_anchor = metadata_distance(row, anchor, scales)
        group_penalty = 0.0 if strict == "fresh_lower_stress_control" else (0.05 if relaxed == "fresh_lower_stress_control" else 0.35)
        theta_penalty = abs(sf(row.get("theta_r")) - sf(anchor.get("theta_r"))) / 8.0
        stress_penalty = abs(sf(row.get("stress_v1_score")) - sf(anchor.get("stress_v1_score"))) / max(scales.get("stress_v1_score", 1.0), 1e-9)
        clearance_penalty = abs(sf(row.get("min_reference_obstacle_clearance")) - sf(anchor.get("min_reference_obstacle_clearance"))) / max(scales.get("min_reference_obstacle_clearance", 1.0), 1e-9)
        # Deliberately favor source72-like lower-stress morphology, not any H12 label.
        score = group_penalty + d_anchor + 0.25 * theta_penalty + 0.25 * stress_penalty + 0.15 * clearance_penalty
        row["source72_metadata_distance"] = d_anchor
        row["source72_selection_score"] = score
        enriched.append(row)
    ranked = sorted(enriched, key=lambda r: (sf(r.get("source72_selection_score")), sf(r.get("source72_metadata_distance")), si(r.get("candidate_index"), 9999)))
    if len(ranked) < TARGET_CASES:
        raise ContractError("too few source72-neighborhood candidates after exclusions")
    selected: List[Dict[str, Any]] = []
    for rank, row in enumerate(ranked[:TARGET_CASES]):
        idx = si(row.get("candidate_index"), -1)
        selected.append({
            "fresh_case_index": rank,
            "source_candidate_index": idx,
            "fresh_confirmation_group": "source72_neighborhood_lower_stress_risk_family",
            "risk_probe_role": f"source72_neighborhood_rank{rank:02d}",
            "selection_mode": "metadata_nearest_source72_excluding_prior_sources",
            "selection_anchor_candidate_index": SOURCE_ANCHOR_INDEX,
            "selection_anchor_metadata_distance": row["source72_metadata_distance"],
            "selection_score": row["source72_selection_score"],
            "strict_group": row.get("strict_group"),
            "relaxed_group": row.get("relaxed_group"),
            "theta_r": sf(row.get("theta_r")),
            "abs_theta_r": sf(row.get("abs_theta_r")),
            "traj_steps": sf(row.get("traj_steps")),
            "min_reference_obstacle_clearance": sf(row.get("min_reference_obstacle_clearance")),
            "stress_v1_score": sf(row.get("stress_v1_score")),
            "case_snapshot_from_candidate_pool": cases[idx],
            "case_snapshot_sha256": hashlib.sha256(json.dumps(clean(cases[idx]), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
            "selection_rule": "v25 metadata-only source72-neighborhood selection before any v25 H12/H15 branch outcome",
        })
    diag = {
        "anchor_source_candidate_index": SOURCE_ANCHOR_INDEX,
        "anchor_metadata": clean(anchor),
        "excluded_source_candidate_indices_count": len(excluded_set),
        "excluded_source_candidate_indices": sorted(excluded_set),
        "eligible_after_exclusion_count": len(enriched),
        "selected_source_candidate_indices": [int(c["source_candidate_index"]) for c in selected],
        "top20_candidate_scores": [clean({"candidate_index": si(r.get("candidate_index"), -1), "score": r.get("source72_selection_score"), "distance": r.get("source72_metadata_distance"), "strict_group": r.get("strict_group"), "relaxed_group": r.get("relaxed_group"), "theta_r": r.get("theta_r"), "traj_steps": r.get("traj_steps"), "clearance": r.get("min_reference_obstacle_clearance"), "stress": r.get("stress_v1_score")}) for r in ranked[:20]],
        "thresholds": thresholds,
    }
    return selected, diag


def load_negative_anchor() -> Dict[str, Any]:
    manifest = read_json(V11_MANIFEST)
    states = list(manifest.get("selected_states") or [])
    for st in states:
        if st.get("base_state_id") == NEGATIVE_ANCHOR_STATE_ID:
            return dict(st)
    raise ContractError("missing v11 negative anchor state " + NEGATIVE_ANCHOR_STATE_ID)


def obs14(raw: Any) -> List[float]:
    vals = [sf(v) for v in (raw or [])[:14]] if isinstance(raw, list) else []
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def trace_row_obs(row: Mapping[str, Any]) -> List[float]:
    return obs14(row.get("observation") or row.get("next_observation") or [])


def trace_row_state(row: Mapping[str, Any]) -> Mapping[str, Any]:
    obj = row.get("previous_state") or row.get("state") or {}
    return obj if isinstance(obj, Mapping) else {}


def anchor_feature_distance(row: Mapping[str, Any], anchor: Mapping[str, Any]) -> float:
    st = trace_row_state(row)
    ast = anchor.get("branch_previous_state") or {}
    obs = trace_row_obs(row)
    aobs = obs14(anchor.get("initial_observation_from_h15_trace") or [])
    terms = [
        ((sf(st.get("x")) - sf(ast.get("x"))) / 12.0) ** 2,
        ((sf(st.get("y")) - sf(ast.get("y"))) / 10.0) ** 2,
        ((sf(st.get("theta")) - sf(ast.get("theta"))) / 1.2) ** 2,
        ((si(row.get("step"), 0) - si(anchor.get("branch_step"), 45)) / 24.0) ** 2,
    ]
    for i, scale in [(0, 0.35), (1, 0.35), (2, 0.45), (5, 1.2), (6, 1.2), (8, 1.2), (9, 1.2), (11, 1.2), (12, 1.2), (13, 0.5)]:
        terms.append(((obs[i] - aobs[i]) / scale) ** 2)
    return math.sqrt(math.fsum(terms) / len(terms))


def make_stage_a_schedule(selected_cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "stage": "A_h15_trace_scan",
        "execution_index": i,
        "fresh_case_index": int(c["fresh_case_index"]),
        "source_candidate_index": int(c["source_candidate_index"]),
        "true_mpc_n_horizon": REF_H,
        "terminal_profile": "matched_terminal",
        "purpose": "collect H15 trace for source72-risk-family branch selection before selector and H12 outcomes",
    } for i, c in enumerate(selected_cases)]


def make_stage_b_template(selected_cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    rows: List[Dict[str, Any]] = []
    exe = 0
    for rep in range(REPEATS):
        for c in selected_cases:
            for slot in range(BRANCH_STATES_PER_CASE):
                horizons = list(TRUE_HORIZONS)
                rng.shuffle(horizons)
                for h in horizons:
                    rows.append({
                        "stage": "B_shared_h15_trueH12_H15",
                        "execution_index": exe,
                        "repeat": rep,
                        "fresh_case_index": int(c["fresh_case_index"]),
                        "source_candidate_index": int(c["source_candidate_index"]),
                        "branch_state_slot": slot,
                        "true_mpc_n_horizon": int(h),
                        "terminal_profile": PRIMARY_TERMINAL_PROFILE,
                        "blocked_randomization_unit": f"v25|rep{rep}|case{c['fresh_case_index']}|slot{slot}|shared_h15",
                    })
                    exe += 1
    return rows


def build_protocol(created: dt.datetime, selected_cases: Sequence[Mapping[str, Any]], case_diag: Mapping[str, Any], input_hashes: Mapping[str, str], backup: Mapping[str, Any], anchor: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_source_independent_H12_risk_family_acquisition_not_validation_not_test",
        "primary_question": "Do v19/v11-like lower-stress mid-late H12 high-cost states reproduce in fresh source-independent source72-neighborhood cases, and can the fixed v20b/v22 selector avoid them while retaining measured compute savings versus fixed H15 and fixed H12?",
        "before_evidence": [
            "v19/v24: fixed H12 has measured decision savings but fails only on opened fresh_v11/fresh_case05_slot1_mid_late_control rows; oracle H12/H15 passes.",
            "v21+v23/v24: broader source-independent stress-pool rows show zero H12 catastrophes and fixed H12 pass5 with ~16.22% decision saving, so broad support sweeps mainly strengthen fixed-H12 rather than adaptivity.",
            "v22/v23: selector overhead is small at branch scale, but current static/history selector is conservative and loses much of fixed-H12 savings on safe fresh rows.",
        ],
        "negative_anchor_from_opened_development": {
            "base_state_id": anchor.get("base_state_id"),
            "source_candidate_index": SOURCE_ANCHOR_INDEX,
            "branch_step": anchor.get("branch_step"),
            "branch_previous_state": anchor.get("branch_previous_state"),
            "initial_observation_from_h15_trace": anchor.get("initial_observation_from_h15_trace"),
            "usage": "H15-trace morphology anchor only; no v25 H12 outcome used for selection",
        },
        "selected_cases": selected_cases,
        "case_selection_diagnostics": case_diag,
        "stage_A_h15_trace_state_selection": {
            "schedule": make_stage_a_schedule(selected_cases),
            "horizon": REF_H,
            "terminal_profile": "matched_terminal",
            "branch_states_per_case": BRANCH_STATES_PER_CASE,
            "eligible_step_bounds": {"min_step": 8, "max_step": 90, "reserve_terminal_margin_steps": 3, "low_fraction": 0.38, "high_fraction": 0.82},
            "slot_rule": "For each H15 trace select the row with minimum H15-observable distance to the opened source72/v11 negative anchor, plus nearest -4 and +4 step neighbors when available; manifest is written before selector choices and before Stage-B H12 outcomes.",
        },
        "selector_preoutcome_rule": {"deployable_config": dict(v22.DEPLOY_CFG), "fit_source": "opened v19 rows only via v20.load_rows; no v25 labels, no grid refit", "feature_source": "Stage-A H15 trace prefix only", "overhead_accounting": "charge measured per-state feature+selector time from preoutcome choice generation"},
        "stage_B_branch_acquisition": {"template": make_stage_b_template(selected_cases), "true_horizons": TRUE_HORIZONS, "terminal_profile": PRIMARY_TERMINAL_PROFILE, "repeats": REPEATS, "blocked_randomization": "H12/H15 order randomized within case/slot/repeat using frozen RNG seed"},
        "budget_declared": {"stage_A_h15_trace_episodes": TARGET_CASES, "stage_B_branch_episodes": TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS) * REPEATS, "total_episodes_exact": TOTAL_EPISODES, "control_step_upper_bound": CONTROL_STEP_CAP, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_grid_evaluations": 0, "selector_choice_evaluations": TARGET_CASES * BRANCH_STATES_PER_CASE, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "decision_rule": {
            "fixed_H12_safe_and_dominates": "treat stress-v1 distribution as fixed-H12-favorable; pivot to scenario/comparison design and fixed-H12-primary confirmation, not adaptive validation",
            "fixed_H12_has_fresh_bad_selector_safe_pass5": "adaptive H12/H15 mechanism has source-independent support; plan fresh confirmation with strong fixed-H12/per-H baselines before validation",
            "fixed_H12_has_fresh_bad_selector_bad_or_weak": "static selector/representation inadequate; freeze bounded terminal-risk/value refit or training ablation",
            "no_oracle_value": "scenario opportunity weak; document and pivot to benchmark/scenario design rather than forcing switching",
        },
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "latest_verified_backup_before_run": {"path": rel(Path(str(backup.get("path", ""))) if backup.get("path") else Path(".")), "time": backup.get("time") or backup.get("time_utc"), "commit": backup.get("commit"), "package_sha256": (backup.get("packages_this_run") or [{}])[0].get("sha256")},
        "input_hashes": input_hashes,
    }
    write_json(PROTOCOL, protocol)
    return protocol


def select_stage_a_states(stage_a_episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], anchor: Mapping[str, Any]) -> List[Dict[str, Any]]:
    bounds = (protocol.get("stage_A_h15_trace_state_selection") or {}).get("eligible_step_bounds") or {}
    min_step = si(bounds.get("min_step"), 8)
    max_step_abs = si(bounds.get("max_step"), 90)
    reserve = si(bounds.get("reserve_terminal_margin_steps"), 3)
    low_frac = sf(bounds.get("low_fraction"), 0.38)
    high_frac = sf(bounds.get("high_fraction"), 0.82)
    selected: List[Dict[str, Any]] = []
    for ep in stage_a_episodes:
        trace_path = ROOT / str(ep.get("path")) / "trace.json"
        trace = read_json(trace_path)
        if not isinstance(trace, list):
            raise ContractError("Stage-A trace is not a list: " + str(ep.get("state_id")))
        n = len(trace)
        lo = max(min_step, int(math.floor(n * low_frac)))
        hi = min(max_step_abs, n - reserve - 1, int(math.ceil(n * high_frac)))
        candidates = [r for r in trace if lo <= si(r.get("step"), -1) <= hi]
        if not candidates:
            candidates = [r for r in trace if min_step <= si(r.get("step"), -1) <= min(max_step_abs, n - reserve - 1)]
        if not candidates:
            raise ContractError("no eligible v25 H15-trace rows for " + str(ep.get("state_id")))
        center = min(candidates, key=lambda r: (anchor_feature_distance(r, anchor), abs(si(r.get("step"), 0) - si(anchor.get("branch_step"), 45)), si(r.get("step"), 0)))
        center_step = si(center.get("step"), 0)
        planned = [(-4, "pre_anchor_neighbor"), (0, "nearest_negative_anchor_morphology"), (4, "post_anchor_neighbor")]
        used_steps: set = set()
        for slot, (offset, name) in enumerate(planned):
            target = center_step + offset
            pool = [r for r in candidates if si(r.get("step"), -1) not in used_steps]
            if not pool:
                pool = candidates
            row = min(pool, key=lambda r: (abs(si(r.get("step"), 0) - target), anchor_feature_distance(r, anchor), si(r.get("step"), 0)))
            step = si(row.get("step"), -1)
            used_steps.add(step)
            base_state_id = f"v25_case{si(ep.get('fresh_case_index'), -1):02d}_slot{slot}_{name}"
            selected.append({
                "base_state_id": base_state_id,
                "fresh_case_index": si(ep.get("fresh_case_index"), -1),
                "source_candidate_index": si(ep.get("source_candidate_index"), -1),
                "fresh_confirmation_group": ep.get("fresh_confirmation_group"),
                "risk_probe_role": ep.get("risk_probe_role"),
                "branch_state_slot": slot,
                "window": name,
                "offset_from_anchor_like_center": offset,
                "branch_step": step,
                "branch_previous_state": copy.deepcopy(trace_row_state(row)),
                "initial_observation_from_h15_trace": copy.deepcopy(trace_row_obs(row)),
                "h15_trace_episode_path": ep.get("path"),
                "h15_negative_anchor_distance": anchor_feature_distance(row, anchor),
                "selection_rule": "v25 H12-risk-family: selected from H15 trace before selector choices and before Stage-B H12/H15 outcome",
            })
    if len(selected) != len(stage_a_episodes) * BRANCH_STATES_PER_CASE:
        raise ContractError("unexpected v25 selected-state count")
    return selected


def make_selector_features(state: Mapping[str, Any], trace_cache: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, float]:
    cand = {
        "candidate_id": str(state["base_state_id"]),
        "base_state_id": str(state["base_state_id"]),
        "branch_previous_state": state.get("branch_previous_state") or {},
        "candidate_branch_step": si(state.get("branch_step"), -1),
        "offset_from_center": si(state.get("offset_from_anchor_like_center"), 0),
        "source_candidate_index": si(state.get("source_candidate_index"), -1),
        "initial_observation_from_h15_trace": obs14(state.get("initial_observation_from_h15_trace") or []),
        "h15_trace_episode_path": str(state.get("h15_trace_episode_path") or ""),
    }
    fd = v20.static_features(cand)
    trace = trace_cache.get(str(cand["h15_trace_episode_path"])) or []
    v22.add_history_features_from_trace(fd, trace, si(cand.get("candidate_branch_step"), -1))
    return fd


def make_preoutcome_selector_choices(selected_states: Sequence[Mapping[str, Any]], input_hashes: Dict[str, str]) -> Dict[str, Any]:
    train_rows, train_features, train_diag, train_hashes = v20.load_rows()
    input_hashes.update(train_hashes)
    model = v22.train_model(train_rows, train_features, v22.DEPLOY_CFG)
    trace_cache: Dict[str, Sequence[Mapping[str, Any]]] = {}
    for st in selected_states:
        ep_path = str(st.get("h15_trace_episode_path") or "")
        trace_path = ROOT / ep_path / "trace.json"
        if not trace_path.exists():
            raise ContractError("missing Stage-A trace for selector features: " + rel(trace_path))
        if ep_path not in trace_cache:
            trace_cache[ep_path] = read_json(trace_path)
            input_hashes[rel(trace_path)] = sha256(trace_path)
    details: List[Dict[str, Any]] = []
    import time
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
        "fit_source": "v19 opened rows via v20.load_rows; no v25 labels, no grid refit, no validation/test",
        "v19_diag": train_diag,
        "choices": details,
        "choice_counts": dict(Counter(str(d["selected_h_preoutcome"]) for d in details)),
        "overhead_summary": {"n": len(times), "sum_s": math.fsum(times), "mean_s": mean(times), "median_s": q(times, 0.5), "p95_s": q(times, 0.95), "max_s": max(times) if times else 0.0},
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    write_json(RUN_DIR / "preoutcome_selector_choices.json", result)
    return result


def write_summary(raw: Mapping[str, Any]) -> None:
    ag = raw["analysis"]["label_analysis"]["aggregate"]
    pol = raw["analysis"]["policies"]
    fx = pol["fixed_H12"]
    sel = pol["selector_preoutcome_actual_overhead"]
    oracle = pol["oracle_H12_H15"]
    lines = [
        "# Vehicle true-variable-H H12-risk-family acquisition v25",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only IMPROVED source-independent H12-risk-family acquisition; no validation64, no sealed test, no training, no selector grid refit.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['total_episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`; selector choices `{raw['budget_actual']['selector_choice_evaluations']}`.",
        "",
        "## Headline",
        "",
        f"- States: `{ag['states']}`; H12-beneficial `{ag['beneficial_H12_count']}`; H12-catastrophic/high-cost `{ag['catastrophic_H12_count']}`; H15 unsafe `{ag['h15_unsafe_count']}`.",
        f"- Fixed H12 vs H15: decision saving `{pct(fx['decision_relative_saving_vs_H15'])}`, solver saving `{pct(fx['solver_relative_saving_vs_H15'])}`, bad `{fx['catastrophic_false_positive_count']}`, physical gate `{fx['physical_gate_vs_H15']}`, pass5 `{fx['pass5_zero_cat_physical']}`.",
        f"- Pre-outcome selector: H counts `{sel['chosen_counts']}`, overhead-adjusted decision saving `{pct(sel['decision_relative_saving_vs_H15'])}`, bad `{sel['catastrophic_false_positive_count']}`, physical gate `{sel['physical_gate_vs_H15']}`, pass5 `{sel['pass5_zero_cat_physical']}`.",
        f"- Oracle H12/H15: H counts `{oracle['chosen_counts']}`, decision saving `{pct(oracle['decision_relative_saving_vs_H15'])}`, bad `{oracle['catastrophic_false_positive_count']}`, pass5 `{oracle['pass5_zero_cat_physical']}`.",
        f"- Decision: {raw['analysis']['decision']}",
        "",
        "## Selected source72-neighborhood cases (pre-outcome metadata)",
        "",
        "| case | source_idx | role | strict | relaxed | score | distance | theta | traj | clearance | stress |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in raw.get("selected_cases") or []:
        lines.append("| %d | %d | `%s` | `%s` | `%s` | %.4g | %.4g | %.4g | %.4g | %.4g | %.4g |" % (int(c.get("fresh_case_index", -1)), int(c.get("source_candidate_index", -1)), c.get("risk_probe_role"), c.get("strict_group"), c.get("relaxed_group"), sf(c.get("selection_score")), sf(c.get("selection_anchor_metadata_distance")), sf(c.get("theta_r")), sf(c.get("traj_steps")), sf(c.get("min_reference_obstacle_clearance")), sf(c.get("stress_v1_score"))))
    lines += ["", "## Per-state labels and selector choices", "", "| state | role | slot | selector H | H12 ben | H12 cat | anchor dist | physΔ H12-H15 | H12 dec | H15 dec |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    choice_by = {str(d["base_state_id"]): d for d in raw["preoutcome_selector_choices"].get("choices", [])}
    state_extra = {str(s["base_state_id"]): s for s in raw.get("selected_states", [])}
    for r in raw["analysis"]["label_analysis"]["state_rows"]:
        ch = choice_by.get(str(r["base_state_id"]), {})
        st = state_extra.get(str(r["base_state_id"]), {})
        lines.append("| `%s` | `%s` | `%s` | %s | `%s` | `%s` | %.4g | %.6g | %.6g | %.6g |" % (r["base_state_id"], r.get("risk_probe_role"), r.get("slot_name"), ch.get("selected_h_preoutcome", "NA"), r.get("h12_beneficial_vs_h15"), r.get("h12_catastrophic_vs_h15"), sf(st.get("h15_negative_anchor_distance")), sf(r.get("physical_delta_h12_minus_h15")), sf((r.get("h12") or {}).get("decision_sum_s")), sf((r.get("h15") or {}).get("decision_sum_s"))))
    lines += ["", "## Interpretation limits", "", "This is targeted development evidence around a known opened negative morphology. It is not a population estimate, not validation64, not sealed-test evidence, and not ORIGINAL SAC. Any adaptive claim remains gated on fresh independent validation/test and strong fixed-H/per-H terminal baselines.", "", f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Protocol: `{rel(PROTOCOL)}`. Backup request: `{rel(BACKUP_REQUEST)}`."]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def update_docs(raw: Mapping[str, Any]) -> None:
    ag = raw["analysis"]["label_analysis"]["aggregate"]
    pol = raw["analysis"]["policies"]
    fx = pol["fixed_H12"]; sel = pol["selector_preoutcome_actual_overhead"]; oracle = pol["oracle_H12_H15"]
    block = f"""<!-- {MARKER} -->
## 2026-09-30 vehicle H12-risk-family acquisition v25

UTC: {raw['created_utc']}. Development-only source-independent H12-risk-family acquisition completed after verified post-v24 backup. No validation64, no sealed test, no training/refit. Budget {raw['budget_actual']['episodes']} episodes / {raw['budget_actual']['control_steps']} control steps plus {raw['budget_actual']['selector_choice_evaluations']} preoutcome selector choices. States={ag['states']}, H12-beneficial={ag['beneficial_H12_count']}, H12-catastrophic/high-cost={ag['catastrophic_H12_count']}. Fixed H12 save={fx['decision_relative_saving_vs_H15']}, bad={fx['catastrophic_false_positive_count']}, pass5={fx['pass5_zero_cat_physical']}; selector save={sel['decision_relative_saving_vs_H15']}, H counts={sel['chosen_counts']}, bad={sel['catastrophic_false_positive_count']}, pass5={sel['pass5_zero_cat_physical']}; oracle pass5={oracle['pass5_zero_cat_physical']}. Decision: {raw['analysis']['decision']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup required before further unique science: `{rel(BACKUP_REQUEST)}`.
"""
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        append_if_missing(ROOT / name, MARKER, block)
    response = f"""
## Follow-up through v25 H12-risk-family acquisition

Updated by GPT-5.5 executor at `{raw['created_utc']}`. Stable Astra IDs preserved. v25 did not access validation64 or sealed test.

| linked recommendation(s) | disposition after v25 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; fixed H12 remains first comparator | v25 fixed H12 save {pct(fx['decision_relative_saving_vs_H15'])}, bad {fx['catastrophic_false_positive_count']}, pass5 {fx['pass5_zero_cat_physical']}; selector save {pct(sel['decision_relative_saving_vs_H15'])}, bad {sel['catastrophic_false_positive_count']}, pass5 {sel['pass5_zero_cat_physical']}; oracle pass5 {oracle['pass5_zero_cat_physical']}. | Do not claim adaptivity unless selector handles fresh fixed-H12 failures with net timing value over strong fixed-H baselines. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | v25 selected {len(raw.get('selected_cases') or [])} source72-neighborhood cases by metadata/H15 trace only, but it is deliberately risk-family targeted development evidence. | Treat as mechanism/diagnostic evidence, not population validation. |
| `A11_training_failure_modes_need_separation` | accepted; decision-linked | v25 decision: {raw['analysis']['decision']} | If selector fails fresh negatives, freeze bounded terminal-risk/value refit or training; if fixed H12 dominates, pivot to scenario/comparison design rather than static selector sweeps. |
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; branch-level selector overhead charged | v25 preoutcome selector choices measured feature+selector overhead mean {raw['preoutcome_selector_choices']['overhead_summary']['mean_s']:.9f}s, p95 {raw['preoutcome_selector_choices']['overhead_summary']['p95_s']:.9f}s and charged it once per branch state. | Full closed-loop deployment timing still required for final speed claims. |
| `A12_registry_backup_schema_contract` | accepted; active | v25 wrote new results/docs/state/registry and backup request `{rel(BACKUP_REQUEST)}`. | Require verified external backup covering v25 before additional unique science. |
"""
    append_if_missing(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", MARKER, response)


def update_registry(raw: Mapping[str, Any]) -> None:
    path = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if MARKER in old:
        return
    row = [raw["created_utc"], NAME, raw["classification"], f"RNG_SEED={RNG_SEED}", "development_source72_risk_family_no_validation64_no_test", str(raw["budget_actual"]["episodes"]), str(raw["budget_actual"]["control_steps"]), str(raw["budget_actual"]["selector_choice_evaluations"]), "0", "0", "False", rel(RUN_DIR / "completed.json"), MARKER]
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="execute v25 development acquisition")
    ap.add_argument("--backup-proof", type=str, required=True, help="verified external backup proof strictly after v24 completion")
    ap.add_argument("--i-accept-development-v25", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_v25:
        raise ContractError("requires --run and --i-accept-development-v25")
    if (RUN_DIR / "completed.json").exists():
        done = completed_ok(RUN_DIR / "completed.json", "existing v25")
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name not in ("run.lock", "failed.json") for p in RUN_DIR.iterdir()):
        raise ContractError("partial output exists; inspect before rerun: " + rel(RUN_DIR))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    backup = dict(verify_backup_post_v24(Path(args.backup_proof)))
    backup["path"] = str(Path(args.backup_proof))
    created0 = now_utc()
    input_hashes = verify_inputs()
    input_hashes[rel(Path(args.backup_proof))] = sha256(Path(args.backup_proof))
    bank = read_json(BANK_PATH)
    excluded = prior_source_indices(bank)
    selected_cases, case_diag = choose_cases(bank, excluded)
    anchor = load_negative_anchor()
    protocol = build_protocol(created0, selected_cases, case_diag, input_hashes, backup, anchor)
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
        raise ContractError("terminal grid missing H15 terminal")
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})

    episodes: List[Dict[str, Any]] = []
    stage_a_eps: List[Dict[str, Any]] = []
    for i, case_meta in enumerate(selected_cases):
        case = case_meta["case_snapshot_from_candidate_pool"]
        item = {"execution_index": i, "state_id": f"v25_stageA_case{i:02d}_source{int(case_meta['source_candidate_index']):03d}", "case": i, "source_candidate_index": int(case_meta["source_candidate_index"]), "branch_step": 0, "branch_previous_state": copy.deepcopy(case.get("state", {"x": 0.0, "y": 0.0, "theta": 0.0})), "true_mpc_n_horizon": REF_H, "commanded_horizon": REF_H, "terminal_mode": "matched_terminal", "initialization": "v25_stageA_H15_trace_before_selector_and_H12_outcome"}
        summary = v8.engine.case_runner.run_true_h_episode(item, case, terminals[REF_H])
        summary.update({"stage": "A_h15_trace_scan", "fresh_case_index": i, "fresh_confirmation_group": case_meta.get("fresh_confirmation_group"), "risk_probe_role": case_meta.get("risk_probe_role")})
        stage_a_eps.append(summary); episodes.append(summary)
        write_json(RUN_DIR / "progress.json", {"stage": "A", "episodes_done": len(episodes), "episodes_expected": TOTAL_EPISODES, "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "validation64_bank_opened": False, "sealed_test_accessed": False})

    selected_states = select_stage_a_states(stage_a_eps, protocol, anchor)
    manifest = {"created_utc": now_utc().isoformat(), "stage_A_complete_before_selector_and_stage_B": True, "negative_anchor_state_id": NEGATIVE_ANCHOR_STATE_ID, "selected_states": selected_states, "protocol_sha256": sha256(PROTOCOL), "validation64_bank_opened": False, "sealed_test_accessed": False}
    write_json(RUN_DIR / "selected_state_manifest.json", manifest)
    write_json(RUN_DIR / "manifest_completed_before_selector_and_stage_B.json", {"created_utc": now_utc().isoformat(), "selected_state_manifest": rel(RUN_DIR / "selected_state_manifest.json"), "selector_started": False, "stage_B_started": False, "validation64_bank_opened": False, "sealed_test_accessed": False})
    preoutcome_choices = make_preoutcome_selector_choices(selected_states, input_hashes)
    write_json(RUN_DIR / "selector_choices_completed_before_stage_B.json", {"created_utc": now_utc().isoformat(), "preoutcome_selector_choices": rel(RUN_DIR / "preoutcome_selector_choices.json"), "stage_B_started": False, "validation64_bank_opened": False, "sealed_test_accessed": False})

    state_by_case_slot = {(si(s.get("fresh_case_index"), -1), si(s.get("branch_state_slot"), -1)): s for s in selected_states}
    branch_episodes: List[Dict[str, Any]] = []
    write_json(RUN_DIR / "stage_B_started.json", {"started_utc": now_utc().isoformat(), "selected_state_manifest_preexisting": True, "selector_choices_preexisting": True, "validation64_bank_opened": False, "sealed_test_accessed": False})
    for j, tmpl in enumerate(protocol["stage_B_branch_acquisition"]["template"]):
        fc = si(tmpl.get("fresh_case_index"), -1); slot = si(tmpl.get("branch_state_slot"), -1)
        st = state_by_case_slot.get((fc, slot))
        if st is None:
            raise ContractError("missing selected state for Stage-B template")
        case_meta = selected_cases[fc]
        case = case_meta["case_snapshot_from_candidate_pool"]
        h = si(tmpl.get("true_mpc_n_horizon"), -1); rep = si(tmpl.get("repeat"), 0)
        item = {"execution_index": 6000 + j, "state_id": f"v25_B{j:03d}_{st['base_state_id']}_r{rep}_H{h}", "case": fc, "source_candidate_index": int(case_meta["source_candidate_index"]), "branch_step": int(st["branch_step"]), "branch_previous_state": copy.deepcopy(st["branch_previous_state"]), "true_mpc_n_horizon": h, "commanded_horizon": h, "terminal_mode": PRIMARY_TERMINAL_PROFILE, "initialization": "v25_stageB_direct_branch_state_from_predeclared_H15_manifest"}
        summary = v8.engine.case_runner.run_true_h_episode(item, case, terminals[REF_H])
        summary.update({"stage": "B_shared_h15_trueH12_H15", "template_execution_index": int(tmpl.get("execution_index", j)), "repeat": rep, "fresh_case_index": fc, "base_state_id": st["base_state_id"], "branch_state_slot": slot, "terminal_profile": PRIMARY_TERMINAL_PROFILE, "terminal_source_horizon": REF_H, "fresh_confirmation_group": case_meta.get("fresh_confirmation_group"), "risk_probe_role": case_meta.get("risk_probe_role")})
        branch_episodes.append(summary); episodes.append(summary)
        write_json(RUN_DIR / "progress.json", {"stage": "B", "episodes_done": len(episodes), "episodes_expected": TOTAL_EPISODES, "stage_B_done": len(branch_episodes), "stage_B_expected": len(protocol["stage_B_branch_acquisition"]["template"]), "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "validation64_bank_opened": False, "sealed_test_accessed": False})

    control_steps = int(sum(si(e.get("steps")) for e in episodes))
    if len(episodes) != TOTAL_EPISODES or control_steps > CONTROL_STEP_CAP:
        raise ContractError("budget mismatch")
    analysis = v23.analyze_policies(branch_episodes, selected_states, preoutcome_choices)
    created = now_utc()
    raw = {"created_utc": created.isoformat(), "started_utc": created0.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "method": NAME, "classification": protocol["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "mobile_robot_mppi_resumed": False, "protocol": {"json": rel(PROTOCOL), "sha256": sha256(PROTOCOL)}, "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}, "backup_proof_used": {"path": rel(Path(args.backup_proof)), "sha256": sha256(Path(args.backup_proof)), "commit": backup.get("commit")}, "input_hashes": input_hashes, "budget_declared": protocol["budget_declared"], "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "selector_choice_evaluations": len(preoutcome_choices.get("choices", [])), "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_grid_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "selected_cases": selected_cases, "selected_states": selected_states, "selected_state_manifest": rel(RUN_DIR / "selected_state_manifest.json"), "preoutcome_selector_choices": preoutcome_choices, "episodes": episodes, "analysis": analysis, "backup_request": rel(BACKUP_REQUEST), "interpretation_limits": ["targeted development risk-family evidence", "not population estimate", "not validation64", "not sealed test", "not ORIGINAL SAC", "no new gradient training or selector grid refit"]}
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    write_json(BACKUP_REQUEST, {"request": "backup_after_v25_h12_risk_family_acquisition", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "new v25 development simulation evidence, docs/state/registry/response-log updates", "must_cover": [rel(SOURCE), rel(PROTOCOL), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"], "episodes": len(episodes), "control_steps": control_steps, "validation64_bank_opened": False, "sealed_test_accessed": False})
    pol = analysis["policies"]
    headline = {"states": analysis["label_analysis"]["aggregate"]["states"], "beneficial_H12_count": analysis["label_analysis"]["aggregate"]["beneficial_H12_count"], "catastrophic_H12_count": analysis["label_analysis"]["aggregate"]["catastrophic_H12_count"], "fixed_H12_save": pol["fixed_H12"]["decision_relative_saving_vs_H15"], "fixed_H12_bad": pol["fixed_H12"]["catastrophic_false_positive_count"], "fixed_H12_pass5": pol["fixed_H12"]["pass5_zero_cat_physical"], "selector_save": pol["selector_preoutcome_actual_overhead"]["decision_relative_saving_vs_H15"], "selector_bad": pol["selector_preoutcome_actual_overhead"]["catastrophic_false_positive_count"], "selector_pass5": pol["selector_preoutcome_actual_overhead"]["pass5_zero_cat_physical"], "selector_h_counts": pol["selector_preoutcome_actual_overhead"]["chosen_counts"], "oracle_pass5": pol["oracle_H12_H15"]["pass5_zero_cat_physical"], "decision": analysis["decision"]}
    completed = {"status": "complete", "hard_pass": bool(pol["selector_preoutcome_actual_overhead"]["pass5_zero_cat_physical"] or pol["fixed_H12"]["pass5_zero_cat_physical"]), "created_utc": created.isoformat(), "classification": protocol["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": raw["budget_actual"], "headline": headline, "backup_request": rel(BACKUP_REQUEST), "hashes": {}}
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, BACKUP_REQUEST, Path(args.backup_proof)]
    completed["hashes"] = {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}
    write_json(RUN_DIR / "completed.json", completed)
    raw["completed_sha256"] = sha256(RUN_DIR / "completed.json")
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v25 H12-risk-family acquisition\n\nUTC: {created.isoformat()}\n\nDecision: {analysis['decision']}\n\nKey results: states={headline['states']}, H12-beneficial={headline['beneficial_H12_count']}, H12-catastrophic={headline['catastrophic_H12_count']}; fixed H12 save={pct(headline['fixed_H12_save'])}, bad={headline['fixed_H12_bad']}, pass5={headline['fixed_H12_pass5']}; selector save={pct(headline['selector_save'])}, H counts={headline['selector_h_counts']}, bad={headline['selector_bad']}, pass5={headline['selector_pass5']}; oracle pass5={headline['oracle_pass5']}.\n\nNext: require verified external backup covering v25. If fresh fixed-H12 failures appeared and selector failed, freeze bounded terminal-risk/value refit or training. If no fresh negatives appeared and fixed H12 dominates, prioritize scenario/comparison design and fixed-H12-primary confirmation rather than more static selector sweeps. No validation64/sealed test.\n", encoding="utf-8")
    update_docs(raw)
    update_registry(raw)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": headline, "backup_request": rel(BACKUP_REQUEST), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    try:
        return run()
    except SystemExit:
        raise
    except Exception as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RUN_DIR / "failed.json", {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_source_independent_H12_risk_family_acquisition_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
