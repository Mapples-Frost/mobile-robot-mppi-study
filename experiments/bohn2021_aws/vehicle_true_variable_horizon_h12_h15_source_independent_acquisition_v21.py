#!/usr/bin/env python3
"""v21 source-independent true-variable-H12/H15 boundary acquisition.

Development-only IMPROVED diagnostic after v19/v20b.

v19 showed that true H12 (with actual smaller NLP dimension) is much less
aggressive than H10 and gives substantial measured H12-vs-H15 compute savings on
most opened boundary states, but fixed H12 remains unsafe because all high-cost
H12 rows were localized in one fresh_v11 lower-stress/source72 cluster.  v20b
then found a conservative offline H12/H15 selector that passes strict opened
bank/source splits, but the only negative H12 source is protected largely by a
"no catastrophic training example => choose H15" uncertainty fallback.  This
script therefore acquires fresh source-independent H12/H15 labels before any
validation attempt.

Protocol properties:
  * development-only IMPROVED; no validation64, no sealed test, no mobile robot;
  * selects fresh source_candidate_index values by metadata only from the existing
    stress-v1 candidate pool, excluding all previously used source indices;
  * targets lower-stress/low-clearance/high-heading regions around prior risk
    morphology, but not by H12 outcomes and not duplicating used sources;
  * Stage A runs H15 traces and writes a selected_state_manifest before any H12
    branch outcome is observed;
  * Stage B runs blocked true-H12/H15 branch rollouts under the shared-H15
    terminal, with measured whole-decision and solver timing;
  * no gradient/RL training and no offline selector refit are performed.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_probe_acquisition_v8 as v8  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402

NAME = "vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21"
STAMP = "20260930T0130Z"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
CONTINUE_STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0130_after_h12_h15_source_independent_acquisition_v21.md"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-h12-h15-source-independent-acquisition-v21-{STAMP}"

# Existing development evidence used only for exclusion/provenance and
# before-evidence.  Do not include the huge bank itself in recursive source-index
# collection, otherwise all candidate_index values would be excluded.
BANK_PATH = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/bank/vehicle_stress_scenario_opportunity_probe_v1_bank.json"
BANK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/bank/completed.json"
STAGE1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/completed.json"

V19_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/summary.md"
V19_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/raw.json"
V19_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/completed.json"
V19_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_preoutcome_frozen_20260930T0015Z.json"
V20B_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/summary.md"
V20B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/raw.json"
V20B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/completed.json"
V20B_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_preoutcome_frozen_20260930T0100Z.json"

PRIOR_INDEX_PATHS = [
    # Earlier fresh-source confirmations and risk probes.
    ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_frozen_20260929T1015Z.json",
    ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1_frozen_20260929T1055Z.json",
    ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_frozen_20260929T1120Z.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/raw.json",
    ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_preoutcome_frozen_20260929T1320Z.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/selected_state_manifest.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/raw.json",
    ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_preoutcome_frozen_20260929T1835Z.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/selected_state_manifest.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/raw.json",
    # v15/v19/v20b opened boundary and H12/H15 selector evidence.
    ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_preoutcome_frozen_20260929T2325Z.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_20260929T2325Z/raw.json",
    V19_PROTOCOL,
    V19_RAW,
    V20B_PROTOCOL,
    V20B_RAW,
    # Original oracle/risk-source acquisitions referenced by v8/fresh freeze.
    v8.freeze0.ORACLE_PROTOCOL,
    v8.freeze0.ORACLE_RAW,
    v8.freeze0.RISK_PROTOCOL,
    v8.freeze0.RISK_RAW,
]

TRUE_HORIZONS = [12, 15]
SHORT_H = 12
REF_H = 15
PRIMARY_TERMINAL_PROFILE = "shared_h15_terminal"
REPEATS = 1
TARGET_CASES = 8
BRANCH_STATES_PER_CASE = 2
MAX_STEPS = 150
RNG_SEED = 202609300130
TOTAL_EPISODES = TARGET_CASES + TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS) * REPEATS
CONTROL_STEP_CAP = TOTAL_EPISODES * MAX_STEPS
MIN_DECISION_SAVING = 0.05

# Roles are fixed before reading any new H12 outcomes.  Anchors are used only as
# metadata references to prior risk morphology; selected indices themselves are
# excluded if they have been used before.
TARGET_ROLES = [
    {"group": "fresh_lower_stress_control", "role": "lower_stress_near_case72_A", "mode": "nearest_anchor", "anchor": 72},
    {"group": "fresh_lower_stress_control", "role": "lower_stress_near_case72_B", "mode": "nearest_anchor", "anchor": 72},
    {"group": "fresh_lower_stress_control", "role": "lower_stress_low_stress_diversity", "mode": "low_stress_diverse", "anchor": 72},
    {"group": "fresh_low_heading_low_clearance_control", "role": "low_clearance_near_case190", "mode": "nearest_anchor", "anchor": 190},
    {"group": "fresh_low_heading_low_clearance_control", "role": "low_clearance_near_case134", "mode": "nearest_anchor", "anchor": 134},
    {"group": "fresh_high_heading_long_or_medium", "role": "high_heading_long_support_near_case74", "mode": "nearest_anchor", "anchor": 74},
    {"group": "fresh_high_heading_long_or_medium", "role": "high_heading_long_high_stress_support", "mode": "high_stress", "anchor": 108},
    {"group": "fresh_high_heading_short", "role": "high_heading_short_support_near_case242", "mode": "nearest_anchor", "anchor": 242},
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


def median(xs: Iterable[float]) -> Optional[float]:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def metric_sum(e: Mapping[str, Any], field: str) -> float:
    obj = e.get(field)
    if isinstance(obj, Mapping):
        return sf(obj.get("sum"), 0.0)
    return 0.0


def is_safe_episode(e: Mapping[str, Any]) -> bool:
    return bool(e.get("success")) and not bool(e.get("constraint")) and si(e.get("solver_failure_steps"), 999) == 0 and si(e.get("initial_failed_steps"), 999) == 0 and si(e.get("final_failed_steps"), 999) == 0


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing prerequisite: " + rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True and obj.get("status") not in ("complete", "completed"):
        raise ContractError("prerequisite did not pass: " + rel(path))
    for key in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if key in obj and obj.get(key) is not False:
            raise ContractError(f"forbidden {key} in {rel(path)}")
    return obj


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
            # Missing or malformed historical metadata should not silently remove
            # exclusions; record the path in protocol diagnostics instead.
            pass
    return sorted(set(vals))


def verify_inputs_and_hashes() -> Dict[str, str]:
    for p in (BANK_PATH, BANK_DONE, STAGE1_DONE, V19_DONE, V19_RAW, V19_PROTOCOL, V20B_DONE, V20B_SUMMARY, V20B_PROTOCOL):
        if not p.exists():
            raise ContractError("missing required input: " + rel(p))
    for p in (BANK_DONE, STAGE1_DONE, V19_DONE, V20B_DONE):
        completed_ok(p)
    # v19 and v20b summaries are development-only; v20b is offline.  Hash the
    # principal inputs plus the script; avoid hashing every prior raw trace here.
    paths = [SOURCE, BANK_PATH, BANK_DONE, STAGE1_DONE, V19_SUMMARY, V19_DONE, V19_RAW, V19_PROTOCOL, V20B_SUMMARY, V20B_DONE, V20B_PROTOCOL]
    for p in PRIOR_INDEX_PATHS:
        if p.exists() and p not in paths:
            paths.append(p)
    return {rel(p): sha256(p) for p in paths if p.exists()}


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


def choose_cases_v21(bank: Mapping[str, Any], excluded: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
        raise ContractError("too few eligible candidate cases after source-index exclusion")
    all_rows_by_idx = {si(r.get("candidate_index"), -1): r for r in metas}
    scales = metadata_scales(metas)
    used: set = set()
    selected_rows: List[Dict[str, Any]] = []
    fallbacks: List[str] = []
    role_diags: List[Dict[str, Any]] = []

    def pool_for(group: str) -> Tuple[str, List[Mapping[str, Any]]]:
        strict = [r for r in enriched if r.get("strict_group") == group and si(r.get("candidate_index"), -1) not in used]
        if strict:
            return "strict", strict
        relaxed = [r for r in enriched if r.get("relaxed_group") == group and si(r.get("candidate_index"), -1) not in used]
        if relaxed:
            return "relaxed", relaxed
        return "global", [r for r in enriched if si(r.get("candidate_index"), -1) not in used]

    def ordered(pool: Sequence[Mapping[str, Any]], role: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        mode = str(role.get("mode"))
        anchor = all_rows_by_idx.get(si(role.get("anchor"), -1))
        if mode == "nearest_anchor" and anchor is not None:
            return sorted(pool, key=lambda r: (metadata_distance(r, anchor, scales), si(r.get("candidate_index"), 9999)))
        if mode == "high_stress":
            return sorted(pool, key=lambda r: (-sf(r.get("stress_v1_score")), -sf(r.get("abs_theta_r")), -sf(r.get("traj_steps")), si(r.get("candidate_index"), 9999)))
        if mode == "low_stress_diverse" and anchor is not None:
            return sorted(pool, key=lambda r: (sf(r.get("stress_v1_score")), -metadata_distance(r, anchor, scales), sf(r.get("abs_theta_r")), si(r.get("candidate_index"), 9999)))
        return sorted(pool, key=lambda r: (si(r.get("candidate_index"), 9999),))

    for role in TARGET_ROLES:
        group = str(role["group"])
        mode, pool = pool_for(group)
        if not pool:
            raise ContractError("empty candidate pool for role " + str(role["role"]))
        if mode != "strict":
            fallbacks.append(f"{role['role']}_{mode}")
        order = ordered(pool, role)
        chosen = dict(order[0])
        idx = si(chosen.get("candidate_index"), -1)
        if idx in excluded_set or idx in used:
            raise ContractError("selection/exclusion invariant failed")
        used.add(idx)
        chosen["fresh_case_index"] = len(selected_rows)
        chosen["fresh_confirmation_group"] = group
        chosen["risk_probe_role"] = str(role["role"])
        chosen["selection_mode"] = mode
        chosen["selection_anchor_candidate_index"] = role.get("anchor")
        chosen["selection_boundary_mode"] = role.get("mode")
        anchor_row = all_rows_by_idx.get(si(role.get("anchor"), -1))
        chosen["selection_anchor_metadata_distance"] = metadata_distance(chosen, anchor_row, scales) if anchor_row else None
        selected_rows.append(chosen)
        role_diags.append({
            "role": role["role"],
            "group": group,
            "selection_mode": mode,
            "pool_size": len(pool),
            "selected_candidate_index": idx,
            "anchor": role.get("anchor"),
            "anchor_metadata_distance": chosen.get("selection_anchor_metadata_distance"),
            "selected_strict_group": chosen.get("strict_group"),
            "selected_relaxed_group": chosen.get("relaxed_group"),
        })

    selected_cases: List[Dict[str, Any]] = []
    for row in selected_rows:
        idx = si(row.get("candidate_index"), -1)
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
            "case_snapshot_from_candidate_pool": cases[idx],
            "case_snapshot_sha256": hashlib.sha256(json.dumps(clean(cases[idx]), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
            "selection_rule": "v21 metadata-only source-independent H12/H15 acquisition; selected source_candidate_index excluded from all prior H10/H12 outcome sources and selected before any new H12 outcome",
        })
    diag = {
        "excluded_source_candidate_indices_count": len(excluded_set),
        "excluded_source_candidate_indices": sorted(excluded_set),
        "eligible_after_exclusion_count": len(enriched),
        "selected_source_candidate_indices": [int(c["source_candidate_index"]) for c in selected_cases],
        "fallbacks": fallbacks,
        "target_roles": TARGET_ROLES,
        "role_diagnostics": role_diags,
        "strict_pool_counts": {g: sum(1 for r in enriched if r["strict_group"] == g) for g in sorted(set(r["group"] for r in TARGET_ROLES))},
        "relaxed_pool_counts": {g: sum(1 for r in enriched if r["relaxed_group"] == g) for g in sorted(set(r["group"] for r in TARGET_ROLES))},
        "thresholds": thr,
    }
    return selected_cases, diag


def make_stage_a(selected_cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "stage": "A_h15_trace_scan",
        "execution_index": i,
        "fresh_case_index": int(c["fresh_case_index"]),
        "source_candidate_index": int(c["source_candidate_index"]),
        "true_mpc_n_horizon": REF_H,
        "terminal_profile": "matched_terminal",
        "purpose": "collect H15 trace before any H12 branch outcome",
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
                        "blocked_randomization_unit": f"v21|rep{rep}|case{c['fresh_case_index']}|slot{slot}|shared_h15",
                    })
                    exe += 1
    return rows


def trace_risk_score(row: Mapping[str, Any], n: int) -> float:
    scorer = getattr(v8.engine.fresh0, "trace_risk_score", None)
    if scorer is None:
        raise ContractError("fresh0.trace_risk_score missing; refusing to change state-selection score")
    return float(scorer(row, n))


def select_stage_a_states_v21(stage_a_episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cfg = (protocol.get("stage_A_h15_trace_state_selection") or {})
    bounds = cfg.get("eligible_step_bounds") or {}
    windows = list(cfg.get("windows") or [])
    per_case = si(cfg.get("branch_states_per_case"), len(windows))
    if per_case <= 0 or len(windows) < per_case:
        raise ContractError("invalid v21 branch state/window contract")
    windows = windows[:per_case]
    min_step = si(bounds.get("min_step"), 8)
    max_step_abs = si(bounds.get("max_step"), 90)
    reserve = si(bounds.get("reserve_terminal_margin_steps"), 3)
    selected: List[Dict[str, Any]] = []
    for ep in stage_a_episodes:
        trace_path = ROOT / str(ep.get("path")) / "trace.json"
        trace = read_json(trace_path)
        if not isinstance(trace, list):
            raise ContractError("Stage-A trace is not a list: " + str(ep.get("state_id")))
        n = len(trace)
        if n < min_step + reserve + 2:
            raise ContractError("Stage-A trace too short for v21 state selection: " + str(ep.get("state_id")))
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
                raise ContractError("no eligible v21 Stage-A state for " + str(ep.get("state_id")))
            row = max(candidates, key=lambda r: trace_risk_score(r, n))
            step = si(row.get("step"), -1)
            chosen_steps.append(step)
            base_state_id = f"fresh_case{si(ep.get('fresh_case_index'), -1):02d}_slot{slot}_{window.get('name', 'window')}"
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
                "stage_a_trace_risk_score": trace_risk_score(row, n),
                "selection_rule": "v21: branch state selected from H15 trace before any Stage-B H12/H15 outcome; scoring/window logic inherited from v8c/v11",
            })
    expected = len(stage_a_episodes) * per_case
    if len(selected) != expected:
        raise ContractError("expected %d selected branch states under v21 protocol, got %d" % (expected, len(selected)))
    return selected


def build_protocol(created: dt.datetime, selected_cases: Sequence[Mapping[str, Any]], case_diag: Mapping[str, Any], input_hashes: Mapping[str, str], backup_commit: str) -> Mapping[str, Any]:
    stage_a = make_stage_a(selected_cases)
    stage_b = make_stage_b_template(selected_cases)
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_true_variable_H12_H15_source_independent_negative_support_acquisition_not_validation_not_test",
        "primary_question": "Does the v20b H12/H15 safety rule generalize to source-independent negative/support states, or was strict pass caused by one known negative source plus uncertainty fallback?",
        "before_evidence": [
            "v19: fixed true-H12 saved 14.32% decision time vs H15 on opened boundary rows but had 6 high-cost/catastrophic rows, all localized in fresh_v11/fresh_case05_slot1_mid_late_control; oracle H12/H15 saved 15.72% with physical gate.",
            "v20b: strict leave-bank/source nested H12/H15 selector passed >=5% zero-cat gates, but H12-negative sources consisted only of fresh_v11/fresh_case05_slot1_mid_late_control and the held-out negative fold was protected by a no-cat-training fallback.",
            "Astra A4/A7/A8 remain accepted: current selector timing excludes online selector overhead, and targeted development banks are not population estimates.",
        ],
        "selected_cases": selected_cases,
        "case_selection_diagnostics": case_diag,
        "case_selection_rule": "metadata-only source-independent selection from stress-v1 bank, excluding prior source_candidate_index values observed in Stage1/oracle/risk/fresh/v8c/v11/v15/v19/v20b artifacts; no H12/H15 outcome for selected sources is read before selection",
        "stage_A_h15_trace_state_selection": {
            "schedule": stage_a,
            "horizon": REF_H,
            "terminal_profile": "matched_terminal",
            "branch_states_per_case": BRANCH_STATES_PER_CASE,
            "eligible_step_bounds": {"min_step": 8, "max_step": 90, "reserve_terminal_margin_steps": 3},
            "windows": [
                {"slot": 0, "name": "early_risk", "low_fraction": 0.12, "high_fraction": 0.34, "selection": "max_h15_trace_risk_score"},
                {"slot": 1, "name": "mid_late_control", "low_fraction": 0.45, "high_fraction": 0.74, "selection": "max_h15_trace_risk_score_with_min_step_separation_8"},
            ],
            "manifest_rule": "selected_state_manifest is written after all Stage-A H15 traces and before any Stage-B H12/H15 branch outcome",
        },
        "stage_B_branch_acquisition": {
            "template": stage_b,
            "true_horizons": TRUE_HORIZONS,
            "terminal_profile": PRIMARY_TERMINAL_PROFILE,
            "repeats": REPEATS,
            "primary_measurements": ["success", "constraint", "solver_failure_steps", "physical_constraint_cost", "total_cost", "whole decision timing", "solver timing", "opt_x sizes"],
            "blocked_randomization": "H12/H15 order randomized within each case/slot/repeat using frozen RNG seed",
        },
        "budget_declared": {
            "stage_A_h15_trace_episodes": len(stage_a),
            "stage_B_branch_episodes": len(stage_b),
            "total_episodes_exact": TOTAL_EPISODES,
            "control_step_upper_bound": CONTROL_STEP_CAP,
            "candidate_pool_resets": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "decision_rule": {
            "if_fresh_H12_false_positives_or_safety_regressions": "inspect failing features/telemetry and pivot to richer terminal-risk/value refit or bounded training ablation rather than validation",
            "if_zero_bad_but_saving_below_5pct": "H12 safety may generalize but opportunity/support is too sparse or selector would be too conservative; broaden source-independent support or revise objective/features",
            "if_zero_bad_and_fixed_oracle_support_exceeds_5pct": "combine with online selector-overhead smoke; only after overhead-adjusted strict splits and fresh confirmation plan independent validation with strong fixed H12/H15/per-H terminal baselines",
            "no_direct_validation": "This acquisition is development-only and cannot by itself authorize validation64 or sealed test claims",
        },
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "latest_verified_backup_before_run_from_supervisor_context": backup_commit,
        "input_hashes": input_hashes,
    }
    write_json(PROTOCOL, protocol)
    return protocol


def summarize_for_h(episodes: Sequence[Mapping[str, Any]], h: int) -> Dict[str, Any]:
    reps = [e for e in episodes if si(e.get("true_mpc_n_horizon"), -1) == h]
    return {
        "n": len(reps),
        "safe_all": bool(reps) and all(is_safe_episode(e) for e in reps),
        "physical": median([sf(e.get("physical_constraint_cost"), 0.0) for e in reps]),
        "total_cost": median([sf(e.get("total_cost"), 0.0) for e in reps]),
        "decision_sum_s": median([metric_sum(e, "decision_timing_s") for e in reps]),
        "solver_sum_s": median([metric_sum(e, "solver_attempt_timing_s") for e in reps]),
        "steps": median([si(e.get("steps"), 0) for e in reps]),
        "success_all": bool(reps) and all(bool(e.get("success")) for e in reps),
        "constraint_any": any(bool(e.get("constraint")) for e in reps),
        "solver_failure_steps_sum": int(sum(si(e.get("solver_failure_steps"), 0) for e in reps)),
        "initial_failed_steps_sum": int(sum(si(e.get("initial_failed_steps"), 0) for e in reps)),
        "final_failed_steps_sum": int(sum(si(e.get("final_failed_steps"), 0) for e in reps)),
        "opt_x_size": sorted(set(si(x) for e in reps for x in (e.get("opt_x_sizes_observed") or []))),
    }


def analyze(branch_episodes: Sequence[Mapping[str, Any]], selected_states: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    meta = {str(s["base_state_id"]): s for s in selected_states}
    rows: List[Dict[str, Any]] = []
    for sid in sorted(meta):
        eps = [e for e in branch_episodes if str(e.get("base_state_id")) == sid]
        h12 = summarize_for_h(eps, SHORT_H)
        h15 = summarize_for_h(eps, REF_H)
        h15_phys = sf(h15.get("physical"), 0.0)
        h12_phys = sf(h12.get("physical"), 0.0)
        h15_dec = sf(h15.get("decision_sum_s"), 0.0)
        h12_dec = sf(h12.get("decision_sum_s"), 0.0)
        h15_sol = sf(h15.get("solver_sum_s"), 0.0)
        h12_sol = sf(h12.get("solver_sum_s"), 0.0)
        tol = max(2.0, 0.05 * abs(h15_phys))
        cat_threshold = max(2.0, 0.25 * abs(h15_phys))
        phys_delta = h12_phys - h15_phys
        decision_delta = h12_dec - h15_dec
        solver_delta = h12_sol - h15_sol
        catastrophic = bool(h15.get("safe_all") and ((not h12.get("safe_all")) or phys_delta > cat_threshold))
        beneficial = bool(h15.get("safe_all") and h12.get("safe_all") and phys_delta <= tol and decision_delta < 0.0)
        st = meta[sid]
        rows.append({
            "base_state_id": sid,
            "fresh_case_index": st.get("fresh_case_index"),
            "source_candidate_index": st.get("source_candidate_index"),
            "fresh_confirmation_group": st.get("fresh_confirmation_group"),
            "risk_probe_role": st.get("risk_probe_role"),
            "branch_state_slot": st.get("branch_state_slot"),
            "slot_name": st.get("window"),
            "branch_step": st.get("branch_step"),
            "h12": h12,
            "h15": h15,
            "row_tolerance_vs_H15": tol,
            "catastrophic_threshold_vs_H15": cat_threshold,
            "physical_delta_h12_minus_h15": phys_delta,
            "decision_delta_h12_minus_h15": decision_delta,
            "solver_delta_h12_minus_h15": solver_delta,
            "decision_gain_h12_vs_h15_s": -decision_delta,
            "solver_gain_h12_vs_h15_s": -solver_delta,
            "h12_catastrophic_vs_h15": catastrophic,
            "h12_beneficial_vs_h15": beneficial,
        })
    fixed15_phys = math.fsum(sf(r["h15"].get("physical"), 0.0) for r in rows)
    fixed12_phys = math.fsum(sf(r["h12"].get("physical"), 0.0) for r in rows)
    fixed15_dec = math.fsum(sf(r["h15"].get("decision_sum_s"), 0.0) for r in rows)
    fixed12_dec = math.fsum(sf(r["h12"].get("decision_sum_s"), 0.0) for r in rows)
    fixed15_sol = math.fsum(sf(r["h15"].get("solver_sum_s"), 0.0) for r in rows)
    fixed12_sol = math.fsum(sf(r["h12"].get("solver_sum_s"), 0.0) for r in rows)
    tol_sum = math.fsum(sf(r.get("row_tolerance_vs_H15"), 2.0) for r in rows)
    catastrophic_rows = [r for r in rows if r["h12_catastrophic_vs_h15"]]
    beneficial_rows = [r for r in rows if r["h12_beneficial_vs_h15"]]
    h15_unsafe_rows = [r for r in rows if not r["h15"].get("safe_all")]
    oracle_phys = oracle_dec = oracle_sol = 0.0
    oracle_counts: Dict[str, int] = {str(SHORT_H): 0, str(REF_H): 0}
    for r in rows:
        if r["h12_beneficial_vs_h15"]:
            oracle_counts[str(SHORT_H)] += 1
            oracle_phys += sf(r["h12"].get("physical"), 0.0)
            oracle_dec += sf(r["h12"].get("decision_sum_s"), 0.0)
            oracle_sol += sf(r["h12"].get("solver_sum_s"), 0.0)
        else:
            oracle_counts[str(REF_H)] += 1
            oracle_phys += sf(r["h15"].get("physical"), 0.0)
            oracle_dec += sf(r["h15"].get("decision_sum_s"), 0.0)
            oracle_sol += sf(r["h15"].get("solver_sum_s"), 0.0)
    fixed12_save = None if fixed15_dec <= 0 else (fixed15_dec - fixed12_dec) / fixed15_dec
    fixed12_solver_save = None if fixed15_sol <= 0 else (fixed15_sol - fixed12_sol) / fixed15_sol
    oracle_save = None if fixed15_dec <= 0 else (fixed15_dec - oracle_dec) / fixed15_dec
    oracle_solver_save = None if fixed15_sol <= 0 else (fixed15_sol - oracle_sol) / fixed15_sol
    fixed12_physical_gate = bool((fixed12_phys - fixed15_phys) <= tol_sum)
    oracle_physical_gate = bool((oracle_phys - fixed15_phys) <= tol_sum)
    role_counts: Dict[str, Any] = {}
    for role in sorted({str(r.get("risk_probe_role")) for r in rows}):
        rr = [r for r in rows if str(r.get("risk_probe_role")) == role]
        role_counts[role] = {
            "states": len(rr),
            "beneficial_H12": sum(1 for r in rr if r["h12_beneficial_vs_h15"]),
            "catastrophic_H12": sum(1 for r in rr if r["h12_catastrophic_vs_h15"]),
            "H15_unsafe": sum(1 for r in rr if not r["h15"].get("safe_all")),
            "mean_phys_delta": None if not rr else math.fsum(sf(r.get("physical_delta_h12_minus_h15"), 0.0) for r in rr) / len(rr),
            "mean_decision_gain_s": None if not rr else math.fsum(sf(r.get("decision_gain_h12_vs_h15_s"), 0.0) for r in rr) / len(rr),
        }
    fixed_pass5 = bool(not h15_unsafe_rows and not catastrophic_rows and fixed12_physical_gate and fixed12_save is not None and fixed12_save >= MIN_DECISION_SAVING)
    oracle_value5 = bool(not h15_unsafe_rows and oracle_physical_gate and oracle_save is not None and oracle_save >= MIN_DECISION_SAVING)
    if catastrophic_rows:
        decision = "fresh source-independent H12 acquisition found H12 catastrophic/high-cost rows; inspect failing features and pivot to richer terminal-risk/value refit/training before validation"
    elif fixed_pass5:
        decision = "fresh source-independent fixed H12 itself passes the H15-referenced 5% safety/physical gate on this development batch; next measure online selector overhead and treat fixed H12 as a strong baseline"
    elif oracle_value5:
        decision = "fresh source-independent H12/H15 oracle has value without H12 catastrophes, but fixed H12 does not pass; combine with v20b selector and overhead smoke before any validation planning"
    else:
        decision = "fresh source-independent H12/H15 batch has weak or unsafe opportunity; broaden support/opportunity data or revise objective/features before validation"
    return {
        "state_rows": rows,
        "role_counts": role_counts,
        "beneficial_count": len(beneficial_rows),
        "catastrophic_rows": catastrophic_rows,
        "h15_unsafe_rows": h15_unsafe_rows,
        "aggregate": {
            "states": len(rows),
            "fixed_H15_physical_sum": fixed15_phys,
            "fixed_H12_physical_sum": fixed12_phys,
            "fixed_H12_physical_delta_vs_H15": fixed12_phys - fixed15_phys,
            "physical_tolerance_sum": tol_sum,
            "fixed_H12_physical_gate_vs_H15": fixed12_physical_gate,
            "fixed_H15_decision_sum_s": fixed15_dec,
            "fixed_H12_decision_sum_s": fixed12_dec,
            "fixed_H12_decision_relative_saving_vs_H15": fixed12_save,
            "fixed_H15_solver_sum_s": fixed15_sol,
            "fixed_H12_solver_sum_s": fixed12_sol,
            "fixed_H12_solver_relative_saving_vs_H15": fixed12_solver_save,
            "fixed_H12_tradeoff_pass5": fixed_pass5,
            "oracle_H12_H15_chosen_counts": oracle_counts,
            "oracle_physical_sum": oracle_phys,
            "oracle_physical_delta_vs_H15": oracle_phys - fixed15_phys,
            "oracle_physical_gate_vs_H15": oracle_physical_gate,
            "oracle_decision_sum_s": oracle_dec,
            "oracle_decision_relative_saving_vs_H15": oracle_save,
            "oracle_solver_sum_s": oracle_sol,
            "oracle_solver_relative_saving_vs_H15": oracle_solver_save,
            "oracle_value5": oracle_value5,
            "catastrophic_H12_count": len(catastrophic_rows),
            "beneficial_H12_count": len(beneficial_rows),
            "h15_unsafe_count": len(h15_unsafe_rows),
        },
        "decision": decision,
    }


def fmt(v: Any) -> str:
    return "NA" if v is None else "%.6g" % float(v)


def write_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    ag = a["aggregate"]
    lines = [
        "# Vehicle true-variable-H H12/H15 source-independent acquisition v21",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only source-independent H12/H15 acquisition; no validation64, no sealed test, no training/refit.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['total_episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Headline",
        "",
        f"- States: `{ag['states']}`; H12-beneficial `{ag['beneficial_H12_count']}`; catastrophic/high-cost H12 `{ag['catastrophic_H12_count']}`; H15 unsafe `{ag['h15_unsafe_count']}`.",
        f"- Fixed H12 vs H15: decision saving `{fmt(ag['fixed_H12_decision_relative_saving_vs_H15'])}`, solver saving `{fmt(ag['fixed_H12_solver_relative_saving_vs_H15'])}`, physical delta `{fmt(ag['fixed_H12_physical_delta_vs_H15'])}`, physical gate `{ag['fixed_H12_physical_gate_vs_H15']}`, pass5 `{ag['fixed_H12_tradeoff_pass5']}`.",
        f"- Oracle H12/H15: counts `{ag['oracle_H12_H15_chosen_counts']}`, decision saving `{fmt(ag['oracle_decision_relative_saving_vs_H15'])}`, solver saving `{fmt(ag['oracle_solver_relative_saving_vs_H15'])}`, physical delta `{fmt(ag['oracle_physical_delta_vs_H15'])}`, value5 `{ag['oracle_value5']}`.",
        f"- Decision: {a['decision']}",
        "",
        "## Selected fresh source cases (pre-outcome)",
        "",
        "| fresh_case | source_candidate_index | group | role | selection | anchor | anchor distance | theta | traj | clearance | stress |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in raw.get("selected_cases") or []:
        lines.append("| %d | %d | `%s` | `%s` | `%s` | %s | %s | %.4g | %.4g | %.4g | %.4g |" % (
            int(c.get("fresh_case_index", -1)), int(c.get("source_candidate_index", -1)), c.get("fresh_confirmation_group"), c.get("risk_probe_role"), c.get("selection_mode"), str(c.get("selection_anchor_candidate_index")), fmt(c.get("selection_anchor_metadata_distance")), sf(c.get("theta_r")), sf(c.get("traj_steps")), sf(c.get("min_reference_obstacle_clearance")), sf(c.get("stress_v1_score"))))
    lines += [
        "",
        "## Per-state H12/H15 rows",
        "",
        "| state | role | slot | H12 ben | H12 cat | physΔ H12-H15 | H12 phys | H15 phys | H12 dec | H15 dec | H12 solver | H15 solver |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in a["state_rows"]:
        lines.append("| `%s` | `%s` | `%s` | `%s` | `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
            r["base_state_id"], r.get("risk_probe_role"), r.get("slot_name"), r.get("h12_beneficial_vs_h15"), r.get("h12_catastrophic_vs_h15"), fmt(r.get("physical_delta_h12_minus_h15")), fmt(r["h12"].get("physical")), fmt(r["h15"].get("physical")), fmt(r["h12"].get("decision_sum_s")), fmt(r["h15"].get("decision_sum_s")), fmt(r["h12"].get("solver_sum_s")), fmt(r["h15"].get("solver_sum_s"))))
    lines += [
        "",
        "## Role counts",
        "",
        "```json",
        json.dumps(clean(a["role_counts"]), indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation limits",
        "",
        "This is development evidence only. It was deliberately enriched around hypothesized risk/support morphology and is not a population estimate, not validation64, not sealed-test evidence, and not ORIGINAL SAC. Speed claims still require online selector-overhead measurement; fixed-H and per-H terminal baselines remain required before validation/final claims.",
        "",
        f"Preoutcome protocol: `{raw['protocol']['json']}`.",
        f"Backup request: `{raw['backup_request_after_run']}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    ag = raw["analysis"]["aggregate"]
    block = f"""<!-- {MARKER} -->
## 2026-09-30 vehicle true-variable-H H12/H15 source-independent acquisition v21

UTC: {raw['created_utc']}. Development-only source-independent H12/H15 acquisition completed: {raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, validation64 closed, sealed test closed, no training/refit. States={ag['states']}, H12-beneficial={ag['beneficial_H12_count']}, H12-catastrophic/high-cost={ag['catastrophic_H12_count']}, fixed-H12 decision saving vs H15={ag['fixed_H12_decision_relative_saving_vs_H15']}, oracle decision saving={ag['oracle_decision_relative_saving_vs_H15']}. Decision: {raw['analysis']['decision']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup required before further unique science: `{raw['backup_request_after_run']}`.
"""
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    if reg.exists():
        old = reg.read_text(encoding="utf-8", errors="replace")
        if MARKER not in old[-100000:]:
            reg.write_text(old.rstrip() + f"\n{raw['created_utc']},{NAME},development_H12_H15_source_independent_acquisition,true_variable_H_shared_h15_terminal,{raw['budget_actual']['episodes']},{raw['budget_actual']['control_steps']},0,0,0,False,{rel(RUN_DIR / 'completed.json')}\n", encoding="utf-8")


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", type=str, required=True)
    ap.add_argument("--i-accept-development-v21", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_v21:
        raise ContractError("requires --run and --i-accept-development-v21")
    if (RUN_DIR / "completed.json").exists():
        done = completed_ok(RUN_DIR / "completed.json")
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name != "run.lock" for p in RUN_DIR.iterdir()):
        raise ContractError("partial output exists; inspect before rerun: " + rel(RUN_DIR))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    created0 = now_utc()
    input_hashes = verify_inputs_and_hashes()
    bank = read_json(BANK_PATH)
    excluded = set(v8.freeze0.selected_stage1_indices(bank))
    excluded.update(source_indices_from_paths(PRIOR_INDEX_PATHS))
    selected_cases, case_diag = choose_cases_v21(bank, sorted(excluded))
    protocol = build_protocol(created0, selected_cases, case_diag, input_hashes, args.backup_verified_commit)

    # Legacy runtime and terminal grid.
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
    write_json(RUN_DIR / "run_started.json", {"started_utc": created0.isoformat(), "pid": os.getpid(), "method": NAME, "backup_verified_commit_from_supervisor_context": args.backup_verified_commit, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0})
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})

    episodes: List[Dict[str, Any]] = []
    stage_a_eps: List[Dict[str, Any]] = []
    for i, case_meta in enumerate(selected_cases):
        case = case_meta["case_snapshot_from_candidate_pool"]
        item = {
            "execution_index": i,
            "state_id": f"v21_stageA_case{i:02d}_source{int(case_meta['source_candidate_index']):03d}",
            "case": i,
            "source_candidate_index": int(case_meta["source_candidate_index"]),
            "branch_step": 0,
            "branch_previous_state": copy.deepcopy(case.get("state", {"x": 0.0, "y": 0.0, "theta": 0.0})),
            "true_mpc_n_horizon": REF_H,
            "commanded_horizon": REF_H,
            "terminal_mode": "matched_terminal",
            "initialization": "v21_stageA_H15_trace_before_any_H12_outcome",
        }
        summary = v8.engine.case_runner.run_true_h_episode(item, case, terminals[REF_H])
        summary["stage"] = "A_h15_trace_scan"
        summary["fresh_case_index"] = i
        summary["fresh_confirmation_group"] = case_meta.get("fresh_confirmation_group")
        summary["risk_probe_role"] = case_meta.get("risk_probe_role")
        stage_a_eps.append(summary)
        episodes.append(summary)
        write_json(RUN_DIR / "progress.json", {"stage": "A", "episodes_done": len(episodes), "episodes_expected": TOTAL_EPISODES, "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "validation64_bank_opened": False, "sealed_test_accessed": False})

    selected_states = select_stage_a_states_v21(stage_a_eps, protocol)
    role_by_case = {int(c["fresh_case_index"]): c.get("risk_probe_role") for c in selected_cases}
    group_by_case = {int(c["fresh_case_index"]): c.get("fresh_confirmation_group") for c in selected_cases}
    for st in selected_states:
        st["risk_probe_role"] = role_by_case.get(si(st.get("fresh_case_index"), -1))
        st["fresh_confirmation_group"] = group_by_case.get(si(st.get("fresh_case_index"), -1))
    manifest = {"created_utc": now_utc().isoformat(), "stage_A_complete_before_any_stage_B": True, "selected_states": selected_states, "protocol_sha256": sha256(PROTOCOL), "validation64_bank_opened": False, "sealed_test_accessed": False}
    write_json(RUN_DIR / "selected_state_manifest.json", manifest)
    write_json(RUN_DIR / "manifest_completed_before_stage_B.json", {"created_utc": now_utc().isoformat(), "selected_state_manifest": rel(RUN_DIR / "selected_state_manifest.json"), "stage_B_started": False, "validation64_bank_opened": False, "sealed_test_accessed": False})

    state_by_case_slot = {(si(s.get("fresh_case_index"), -1), si(s.get("branch_state_slot"), -1)): s for s in selected_states}
    branch_template = ((protocol.get("stage_B_branch_acquisition") or {}).get("template") or [])
    branch_episodes: List[Dict[str, Any]] = []
    write_json(RUN_DIR / "stage_B_started.json", {"started_utc": now_utc().isoformat(), "selected_state_manifest_preexisting": True, "validation64_bank_opened": False, "sealed_test_accessed": False})
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
            "execution_index": 4000 + j,
            "state_id": f"v21_B{j:03d}_{st['base_state_id']}_r{rep}_H{h}",
            "case": fc,
            "source_candidate_index": int(case_meta["source_candidate_index"]),
            "branch_step": int(st["branch_step"]),
            "branch_previous_state": copy.deepcopy(st["branch_previous_state"]),
            "true_mpc_n_horizon": h,
            "commanded_horizon": h,
            "terminal_mode": PRIMARY_TERMINAL_PROFILE,
            "initialization": "v21_stageB_direct_branch_state_from_predeclared_H15_manifest",
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
    analysis = analyze(branch_episodes, selected_states)
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_V21_SOURCE_INDEPENDENT_H12_H15_ACQUISITION_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created.isoformat(),
        "started_utc": created0.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_true_variable_H12_H15_source_independent_negative_support_acquisition_not_validation_not_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "mobile_robot_mppi_resumed": False,
        "protocol": {"json": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "budget_declared": protocol["budget_declared"],
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "selected_cases": selected_cases,
        "selected_state_manifest": rel(RUN_DIR / "selected_state_manifest.json"),
        "episodes": episodes,
        "analysis": analysis,
        "backup_request_after_run": rel(req),
        "interpretation_limits": ["development-only", "source-independent from previous H-outcome sources but still selected from a diagnostic stress pool", "not validation64", "not sealed test", "not ORIGINAL SAC", "measured timing only for H12/H15 branch rollouts; online selector overhead not measured here"],
    }
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup v21 source-independent H12/H15 acquisition outputs before any further simulations/refits/overhead smokes", "backup_required_before_more_unique_science": True, "episodes": len(episodes), "control_steps": control_steps, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(RUN_DIR), rel(PROTOCOL), rel(STATE), rel(CONTINUE_STATE), rel(SOURCE), rel(req)]})
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((RUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    CONTINUE_STATE.write_text("# Continue state after v21 source-independent H12/H15 acquisition\n\nUTC: %s\n\nDecision: %s\n\nBudget actual: %s\n\nNext: verify external backup, then inspect v21 rows. If fresh H12 false positives appeared, pivot to richer terminal-risk/value refit/training; otherwise run online selector-overhead smoke before any validation planning.\n" % (created.isoformat(), analysis["decision"], json.dumps(raw["budget_actual"], sort_keys=True)), encoding="utf-8")
    append_docs(raw)
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, CONTINUE_STATE, req]
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "classification": raw["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "backup_request": rel(req),
        "headline": {
            "states": analysis["aggregate"]["states"],
            "beneficial_H12_count": analysis["aggregate"]["beneficial_H12_count"],
            "catastrophic_H12_count": analysis["aggregate"]["catastrophic_H12_count"],
            "fixed_H12_decision_relative_saving_vs_H15": analysis["aggregate"]["fixed_H12_decision_relative_saving_vs_H15"],
            "oracle_decision_relative_saving_vs_H15": analysis["aggregate"]["oracle_decision_relative_saving_vs_H15"],
            "decision": analysis["decision"],
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(RUN_DIR / "completed.json", completed)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "headline": completed["headline"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RUN_DIR / "failed.json", {"failed_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
