#!/usr/bin/env python3
"""Vehicle V1 transient-state continuation probe v0 runner.

Development-only continuation probe frozen by
research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.*.

The probe consumes the no-simulation v0b unique target list, replays each source
fresh-bank case with an H15 prefix to the selected transient/control state, and
then fixes the branch horizon to one of H10/H15/H30/H35.  It performs no
training/refit, opens no historical validation64 bank, and never accesses the
sealed final test.

Execution of rollouts requires an external backup proof after this runner source
and the v0b target-selection artifacts are backed up.  Import/argparse smoke via
--help is intentionally side-effect-free.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

# Reuse the already audited controller/environment mechanics from the fresh
# continuation-label runner.  This runner only changes the target schedule and
# the analysis gate.
import vehicle_v1_fresh_continuation_label_probe_v0_runner as fresh  # noqa:E402

TASK = "vehicle"
PREFIX_H = 15
BRANCH_HORIZONS = [10, 15, 30, 35]
MAX_STEPS = 150
ORDER_SEED = 2609287701
MATERIAL_GAIN_THRESHOLD = 3.0
PREVIOUSLY_MINED_CASES = {7, 10}
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_continuation_probe_v0_20260928"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_transient_state_continuation_probe_v0_20260928.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.json"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.md"
V0B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z/raw.json"
V0B_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z/completed.json"
FRESH_RAW = fresh.OUT_DIR / "raw.json"
FRESH_COMPLETED = fresh.OUT_DIR / "completed.json"
FRESH_BANK = fresh.BANK_PATH
V1_RAW = fresh.V1_RAW
V1_COMPLETED = fresh.V1_DONE
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-v1-transient-state-continuation-probe-v0-runner-20260928"

# Patch the reused fresh-runner globals before any rollout helper is called.
fresh.OUT_DIR = OUT_DIR
fresh.STATE_PATH = STATE_PATH
fresh.PREFIX_H = PREFIX_H
fresh.BRANCH_HORIZONS = BRANCH_HORIZONS
fresh.MAX_STEPS = MAX_STEPS
fresh.IMPROVEMENT_ABS_THRESHOLD = MATERIAL_GAIN_THRESHOLD


class ContractError(RuntimeError):
    """Raised for protocol, access or budget violations."""


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def serial(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=serial) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def source_mtime_utc() -> dt.datetime:
    return dt.datetime.fromtimestamp(Path(__file__).resolve().stat().st_mtime, dt.timezone.utc)


def latest_time(*values: Optional[dt.datetime]) -> Optional[dt.datetime]:
    vals = [v.astimezone(dt.timezone.utc) for v in values if v is not None]
    return max(vals) if vals else None


def verify_completed(path: Path, check_hashes: bool = True) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("missing completed marker: %s" % rel(path))
    done = read_json(path)
    if done.get("passed") is not True:
        raise ContractError("completed marker did not pass: %s" % rel(path))
    if check_hashes:
        for name, expected in (done.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists():
                raise ContractError("completed marker references missing file: %s" % name)
            actual = sha256(p)
            if actual != expected:
                raise ContractError("hash mismatch for completed marker input %s: %s != %s" % (name, actual, expected))
    return done


def verify_backup_proof(path: Path, min_time: Optional[dt.datetime]) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("backup proof does not exist: %s" % rel(path))
    proof = read_json(path)
    verified = proof.get("backup_verified") is True or proof.get("status") == "verified"
    if not verified:
        raise ContractError("backup proof is not verified: %s" % rel(path))
    if int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("backup proof does not record remaining_changed_files=0")
    if not proof.get("commit"):
        raise ContractError("backup proof lacks commit")
    has_asset = bool(
        proof.get("asset_sha256")
        or proof.get("release_asset_sha256")
        or proof.get("package_sha256")
        or proof.get("packages_this_run")
    )
    if not has_asset:
        raise ContractError("backup proof lacks package/release asset SHA")
    proof_time = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp", "recorded_utc"):
        proof_time = parse_time(proof.get(key))
        if proof_time is not None:
            break
    if min_time is not None:
        if proof_time is None:
            raise ContractError("backup proof lacks parseable time")
        if proof_time < min_time:
            raise ContractError("backup proof predates required source/protocol/artifact time: %s < %s" % (proof_time.isoformat(), min_time.isoformat()))
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "commit": proof.get("commit"),
        "time": proof_time.isoformat() if proof_time else None,
        "remaining_changed_files": proof.get("remaining_changed_files"),
        "raw_status": proof.get("status"),
        "backup_verified": proof.get("backup_verified"),
        "packages_this_run": proof.get("packages_this_run"),
        "min_required_time_utc": None if min_time is None else min_time.isoformat(),
    }


def assert_fresh_output_dir() -> None:
    if not OUT_DIR.exists():
        return
    completed = OUT_DIR / "completed.json"
    if completed.exists():
        verify_completed(completed)
        raise SystemExit("transient-state continuation probe already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name not in ("run.lock",)]
    if leftovers:
        raise ContractError("partial transient continuation output exists; inspect/recover first: " + ", ".join(rel(p) for p in leftovers[:20]))


def verify_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for path in (PROTOCOL_JSON, PROTOCOL_MD, V0B_RAW, V0B_COMPLETED, FRESH_RAW, FRESH_COMPLETED, FRESH_BANK, V1_RAW, V1_COMPLETED):
        if not path.exists():
            raise ContractError("required input missing: %s" % rel(path))
    protocol = read_json(PROTOCOL_JSON)
    if protocol.get("protocol_id") != "vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928":
        raise ContractError("unexpected transient-state protocol id")
    if protocol.get("classification") != "development_IMPROVED_transient_state_continuation_diagnostic_not_validation_not_final_test":
        raise ContractError("unexpected transient-state protocol classification")
    design = protocol.get("rollout_design_after_backup") or {}
    if int(design.get("prefix_horizon", -1)) != PREFIX_H:
        raise ContractError("prefix horizon changed")
    if [int(x) for x in design.get("branch_horizons", [])] != BRANCH_HORIZONS:
        raise ContractError("branch horizons changed")
    if int(design.get("max_rollout_episodes", -1)) != 48 or int(design.get("control_step_upper_bound", -1)) != 7200:
        raise ContractError("rollout budget changed")
    if design.get("historical_validation64_bank_opened") is not False or design.get("sealed_test_accessed") is not False:
        raise ContractError("protocol access flags invalid")
    if protocol.get("backup_required_before_rollout") is not True:
        raise ContractError("protocol must require backup before rollout")
    v0b_done = verify_completed(V0B_COMPLETED)
    if v0b_done.get("hard_pass") is not True or v0b_done.get("historical_validation64_bank_opened") is not False or v0b_done.get("sealed_test_accessed") is not False:
        raise ContractError("v0b completed marker access/pass flags invalid")
    if int(v0b_done.get("selected_target_count", -1)) != 12:
        raise ContractError("v0b selected target count changed")
    v0b_raw = read_json(V0B_RAW)
    if v0b_raw.get("historical_validation64_bank_opened") is not False or v0b_raw.get("sealed_test_accessed") is not False:
        raise ContractError("v0b raw access flags invalid")
    targets = v0b_raw.get("selected_targets") or []
    if len(targets) != 12:
        raise ContractError("v0b raw target count changed")
    unique = {(int(t["case"]), str(t["kind"]), int(t["step"])) for t in targets}
    if len(unique) != len(targets):
        raise ContractError("v0b target list contains duplicate case/kind/step entries")
    if any([int(t.get("prefix_horizon", PREFIX_H)) != PREFIX_H for t in targets]):
        raise ContractError("target prefix horizon changed")
    if any([list(map(int, t.get("branch_horizons", []))) != BRANCH_HORIZONS for t in targets]):
        raise ContractError("target branch horizons changed")
    fresh_done = verify_completed(FRESH_COMPLETED, check_hashes=False)
    if fresh_done.get("historical_validation64_bank_opened") is not False or fresh_done.get("sealed_test_accessed") is not False:
        raise ContractError("fresh-probe access flags invalid")
    fixed_done = verify_completed(V1_COMPLETED, check_hashes=False)
    if fixed_done.get("historical_validation64_bank_opened") is not False or fixed_done.get("sealed_test_accessed") is not False:
        raise ContractError("fixed-H V1 access flags invalid")
    bank = read_json(FRESH_BANK)
    if len(bank.get("selected_cases", [])) != 12:
        raise ContractError("fresh bank selected-case count changed")
    for target in targets:
        c = int(target["case"])
        if c < 0 or c >= len(bank["selected_cases"]):
            raise ContractError("target case outside fresh bank: %d" % c)
    return protocol, v0b_raw, bank, read_json(V1_RAW)


def build_schedule(targets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for target in targets:
        for h in BRANCH_HORIZONS:
            rows.append({
                "target_index": int(target["target_index"]),
                "target_kind": str(target["kind"]),
                "case": int(target["case"]),
                "branch_step": int(target["step"]),
                "branch_horizon": int(h),
                "target_transient_score": float(target["transient_score"]),
            })
    if len(rows) != 48:
        raise ContractError("unexpected transient rollout schedule length: %d" % len(rows))
    rng = np.random.RandomState(ORDER_SEED)
    order = rng.permutation(len(rows)).tolist()
    schedule: List[Dict[str, Any]] = []
    for exec_idx, base_idx in enumerate(order):
        row = dict(rows[int(base_idx)])
        row["schedule_base_index"] = int(base_idx)
        row["execution_index"] = int(exec_idx)
        schedule.append(row)
    return schedule


def annotate_episode(summary: Dict[str, Any], item: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(summary)
    out.update({
        "target_index": int(item["target_index"]),
        "target_kind": str(item["target_kind"]),
        "target_transient_score": float(item["target_transient_score"]),
        "target_previous_state_from_selection": target.get("previous_state"),
        "target_h15_trace": target.get("h15_trace"),
        "target_h15_trace_sha256": target.get("h15_trace_sha256"),
        "target_selection_features": {
            "tracking_error_norm": target.get("tracking_error_norm"),
            "heading_error_abs": target.get("heading_error_abs"),
            "performance_step_cost": target.get("performance_step_cost"),
            "obstacle_clearance_proxy": target.get("obstacle_clearance_proxy"),
            "score_components": target.get("score_components"),
        },
    })
    return out


def comparison_is_safe(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    return bool(fresh.no_regression(candidate, reference))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def state_tuple_from_previous_state(row: Mapping[str, Any]) -> Tuple[float, float, float]:
    s = row.get("branch_previous_state") or {}
    return (safe_float(s.get("x"), float("nan")), safe_float(s.get("y"), float("nan")), safe_float(s.get("theta"), float("nan")))


def analyze_transient_labels(episodes: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    target_by_index = {int(t["target_index"]): t for t in targets}
    groups: Dict[int, List[Mapping[str, Any]]] = {}
    for ep in episodes:
        groups.setdefault(int(ep["target_index"]), []).append(ep)
    state_rows: List[Dict[str, Any]] = []
    missing_reference: List[int] = []
    missing_horizons: List[Dict[str, Any]] = []
    prefix_mismatch_count = 0
    state_distance_gt_tol_count = 0
    branch_not_reached_count = 0
    h15_failure_or_constraint_count = 0
    solver_or_initial_regression_count = 0
    positive_state_count = 0
    negative_or_neutral_state_count = 0
    positive_cases: List[int] = []
    positive_outside_mined_cases: List[int] = []
    positive_horizon_counts: Dict[str, int] = {}
    low_control_state_count = 0
    low_control_positive_count = 0
    high_state_count = 0
    high_positive_count = 0
    best_non_h15_total_gain = -float("inf")
    best_non_h15_physical_gain = -float("inf")
    worst_non_h15_total_gain = float("inf")
    large_non_h15_harms = 0
    material_positive_details: List[Dict[str, Any]] = []
    for target_index in sorted(target_by_index):
        target = target_by_index[target_index]
        rows = sorted(groups.get(target_index, []), key=lambda r: int(r.get("branch_horizon", -1)))
        by_h = {int(r["branch_horizon"]): r for r in rows}
        absent = [h for h in BRANCH_HORIZONS if h not in by_h]
        if absent:
            missing_horizons.append({"target_index": target_index, "case": int(target["case"]), "step": int(target["step"]), "missing_horizons": absent})
        if PREFIX_H not in by_h:
            missing_reference.append(target_index)
            continue
        ref = by_h[PREFIX_H]
        if not bool(ref.get("branch_reached")):
            branch_not_reached_count += 1
        if (not bool(ref.get("success"))) or bool(ref.get("constraint")) or int(ref.get("solver_failure_steps", 0)) > 0 or int(ref.get("initial_failed_steps", 0)) > 0:
            h15_failure_or_constraint_count += 1
        ref_state = state_tuple_from_previous_state(ref)
        comparisons: List[Dict[str, Any]] = []
        material_horizons: List[int] = []
        best_physical: Optional[Dict[str, Any]] = None
        best_total: Optional[Dict[str, Any]] = None
        for h in BRANCH_HORIZONS:
            row = by_h.get(h)
            if row is None:
                continue
            branch_reached = bool(row.get("branch_reached"))
            if not branch_reached:
                branch_not_reached_count += 1
            cand_state = state_tuple_from_previous_state(row)
            state_dist = fresh.state_distance(cand_state, ref_state) if branch_reached and bool(ref.get("branch_reached")) else float("inf")
            if branch_reached and h != PREFIX_H and state_dist > 1e-5:
                state_distance_gt_tol_count += 1
            prefix_match = bool(row.get("prefix_clean_sha256") == ref.get("prefix_clean_sha256"))
            if h != PREFIX_H and not prefix_match:
                prefix_mismatch_count += 1
            phys_gain = float(ref["continuation_physical_constraint_cost_from_branch"] - row["continuation_physical_constraint_cost_from_branch"]) if branch_reached and bool(ref.get("branch_reached")) else float("nan")
            total_gain = float(ref["continuation_total_cost_from_branch"] - row["continuation_total_cost_from_branch"]) if branch_reached and bool(ref.get("branch_reached")) else float("nan")
            if h != PREFIX_H and math.isfinite(total_gain):
                best_non_h15_total_gain = max(best_non_h15_total_gain, total_gain)
                worst_non_h15_total_gain = min(worst_non_h15_total_gain, total_gain)
                if total_gain <= -MATERIAL_GAIN_THRESHOLD:
                    large_non_h15_harms += 1
            if h != PREFIX_H and math.isfinite(phys_gain):
                best_non_h15_physical_gain = max(best_non_h15_physical_gain, phys_gain)
            safe = bool(branch_reached and bool(ref.get("branch_reached")) and prefix_match and state_dist <= 1e-5 and comparison_is_safe(row, ref))
            if h != PREFIX_H and branch_reached and not comparison_is_safe(row, ref):
                if int(row.get("solver_failure_steps", 0)) > int(ref.get("solver_failure_steps", 0)) or int(row.get("initial_failed_steps", 0)) > int(ref.get("initial_failed_steps", 0)):
                    solver_or_initial_regression_count += 1
            material = bool(h != PREFIX_H and safe and (phys_gain >= MATERIAL_GAIN_THRESHOLD or total_gain >= MATERIAL_GAIN_THRESHOLD))
            if material:
                material_horizons.append(h)
                positive_horizon_counts[str(h)] = positive_horizon_counts.get(str(h), 0) + 1
            comp = {
                "horizon": int(h),
                "branch_reached": branch_reached,
                "success": bool(row.get("success")),
                "constraint": bool(row.get("constraint")),
                "steps": int(row.get("steps", 0)),
                "continuation_physical": float(row.get("continuation_physical_constraint_cost_from_branch", 0.0)),
                "continuation_total": float(row.get("continuation_total_cost_from_branch", 0.0)),
                "gain_vs_H15_physical": phys_gain,
                "gain_vs_H15_total": total_gain,
                "decision_sum_s": float(row["decision_timing_s"]["sum"]),
                "decision_plus_terminal_switch_sum_s": float(row["decision_plus_terminal_switch_timing_s"]["sum"]),
                "initial_failed_steps": int(row.get("initial_failed_steps", 0)),
                "solver_failure_steps": int(row.get("solver_failure_steps", 0)),
                "state_distance_vs_H15_branch_state": state_dist,
                "prefix_clean_sha256_matches_H15": prefix_match,
                "no_success_constraint_solver_regression_vs_H15": bool(comparison_is_safe(row, ref)),
                "material_positive_vs_H15": material,
                "path": row.get("path"),
            }
            comparisons.append(comp)
            if branch_reached:
                if best_physical is None or (comp["continuation_physical"], comp["decision_sum_s"], h) < (best_physical["continuation_physical"], best_physical["decision_sum_s"], best_physical["horizon"]):
                    best_physical = comp
                if best_total is None or (comp["continuation_total"], comp["decision_sum_s"], h) < (best_total["continuation_total"], best_total["decision_sum_s"], best_total["horizon"]):
                    best_total = comp
        positive = bool(material_horizons)
        if positive:
            positive_state_count += 1
            positive_cases.append(int(target["case"]))
            if int(target["case"]) not in PREVIOUSLY_MINED_CASES:
                positive_outside_mined_cases.append(int(target["case"]))
            for h in material_horizons:
                material_positive_details.append({"target_index": target_index, "case": int(target["case"]), "kind": target["kind"], "step": int(target["step"]), "horizon": int(h)})
        else:
            negative_or_neutral_state_count += 1
        if str(target["kind"]) == "low_transient_control":
            low_control_state_count += 1
            if positive:
                low_control_positive_count += 1
        else:
            high_state_count += 1
            if positive:
                high_positive_count += 1
        state_rows.append({
            "target_index": int(target_index),
            "case": int(target["case"]),
            "kind": str(target["kind"]),
            "branch_step": int(target["step"]),
            "target_transient_score": float(target["transient_score"]),
            "reference_H15": {
                "success": bool(ref.get("success")),
                "constraint": bool(ref.get("constraint")),
                "steps": int(ref.get("steps", 0)),
                "continuation_physical": float(ref.get("continuation_physical_constraint_cost_from_branch", 0.0)),
                "continuation_total": float(ref.get("continuation_total_cost_from_branch", 0.0)),
                "initial_failed_steps": int(ref.get("initial_failed_steps", 0)),
                "solver_failure_steps": int(ref.get("solver_failure_steps", 0)),
                "branch_previous_state": ref.get("branch_previous_state"),
            },
            "best_physical": best_physical,
            "best_total": best_total,
            "material_positive_state": positive,
            "material_positive_horizons": material_horizons,
            "label": "positive_non_H15" if positive else "negative_or_neutral",
            "comparisons": comparisons,
        })
    artifact_flags = {
        "missing_H15_reference_count": len(missing_reference),
        "missing_horizon_groups_count": len(missing_horizons),
        "prefix_mismatch_count_non_H15": int(prefix_mismatch_count),
        "state_distance_gt_1e_minus_5_count_non_H15": int(state_distance_gt_tol_count),
        "branch_not_reached_count": int(branch_not_reached_count),
        "h15_failure_or_constraint_or_solver_issue_count": int(h15_failure_or_constraint_count),
        "solver_or_initial_failure_regression_count_non_H15": int(solver_or_initial_regression_count),
    }
    blocking_artifact_count = int(sum(artifact_flags.values()))
    transient_missed_timing_supported = bool(
        positive_state_count >= 2
        and len(set(positive_outside_mined_cases)) >= 1
        and negative_or_neutral_state_count >= 2
        and len(missing_reference) == 0
    )
    scenario_scarcity_supported = bool(
        positive_state_count < 2
        and blocking_artifact_count == 0
        and negative_or_neutral_state_count >= 2
    )
    if best_non_h15_total_gain == -float("inf"):
        best_non_h15_total_gain = float("nan")
    if best_non_h15_physical_gain == -float("inf"):
        best_non_h15_physical_gain = float("nan")
    if worst_non_h15_total_gain == float("inf"):
        worst_non_h15_total_gain = float("nan")
    return {
        "state_count_total": len(state_rows),
        "positive_state_count": int(positive_state_count),
        "negative_or_neutral_state_count": int(negative_or_neutral_state_count),
        "positive_cases": sorted(set(positive_cases)),
        "positive_cases_outside_previously_mined_case7_case10": sorted(set(positive_outside_mined_cases)),
        "positive_horizon_counts": positive_horizon_counts,
        "low_control_state_count": int(low_control_state_count),
        "low_control_positive_count": int(low_control_positive_count),
        "high_state_count": int(high_state_count),
        "high_positive_count": int(high_positive_count),
        "best_non_H15_total_gain": float(best_non_h15_total_gain),
        "best_non_H15_physical_gain": float(best_non_h15_physical_gain),
        "worst_non_H15_total_gain": float(worst_non_h15_total_gain),
        "large_non_H15_harms_total_gain_le_minus_threshold": int(large_non_h15_harms),
        "material_positive_details": material_positive_details,
        "missing_H15_reference": missing_reference,
        "missing_horizons": missing_horizons,
        "artifact_flags": artifact_flags,
        "blocking_artifact_count": blocking_artifact_count,
        "transient_missed_timing_supported_gate_pass": transient_missed_timing_supported,
        "scenario_scarcity_supported_gate_fail_clean": scenario_scarcity_supported,
        "gate_rule": "pass if >=2 material-positive non-H15 states, at least one outside previously mined case7/case10, and >=2 negative/control states retained; fail-clean if <2 positives and no prefix/state/solver/reference artifacts explain the absence",
        "material_positive_rule": "non-H15 branch horizon from identical H15-prefix state, no success/constraint/solver/initial-failure regression vs H15, prefix hash matches, state distance <=1e-5, and continuation physical or total gain >=3 absolute units",
        "state_rows": state_rows,
    }


def source_hashes() -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(fresh.__file__).resolve(),
        Path(fresh.v1probe.__file__).resolve(),
        Path(fresh.base.__file__).resolve(),
        Path(fresh.base.smoke_base.__file__).resolve(),
        Path(fresh.base.v1.__file__).resolve(),
        PROTOCOL_JSON,
        PROTOCOL_MD,
        V0B_RAW,
        V0B_COMPLETED,
        FRESH_RAW,
        FRESH_COMPLETED,
        FRESH_BANK,
        V1_RAW,
        V1_COMPLETED,
        ROOT / "experiments/bohn2021_reproduction/conservative_canonical_reset.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_solver_recovery.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_run.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_audit.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_search.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_timing.py",
        ROOT / "experiments/bohn2021_reproduction/run.py",
        ROOT / "experiments/bohn2021_reproduction/runtime.py",
    ]
    return {rel(p): sha256(p) for p in paths if p.exists()}


def write_backup_request(raw: Mapping[str, Any]) -> str:
    stamp = str(raw["created_utc"]).replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_V1_TRANSIENT_STATE_CONTINUATION_PROBE_V0_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup transient-state continuation probe before selector/refit/stress-scenario follow-up",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": raw["budget_actual"]["episodes"],
        "control_steps": raw["budget_actual"]["control_steps"],
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(PROTOCOL_MD), rel(PROTOCOL_JSON), rel(V0B_RAW), rel(V0B_COMPLETED)],
    })
    return rel(path)


def write_summary(raw: Mapping[str, Any]) -> None:
    analysis = raw["analysis"]
    lines = [
        "# Vehicle V1 transient-state continuation probe v0",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Development-only identical-prefix continuation probe from H15-selected transient/control states. No training/refit, no historical validation64-bank access, no sealed-test access.",
        "",
        "## Access and budget",
        "",
        f"- Backup proof: `{raw['backup_proof']['path']}` commit `{raw['backup_proof']['commit']}`.",
        f"- Episodes: `{raw['budget_actual']['episodes']}` / frozen max `{raw['budget_declared']['max_rollout_episodes']}`.",
        f"- Control steps: `{raw['budget_actual']['control_steps']}` / frozen upper bound `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- Training episodes / gradient steps: `{raw['budget_actual']['new_training_episodes']}` / `{raw['budget_actual']['new_gradient_steps']}`.",
        f"- historical_validation64_bank_opened: `{raw['historical_validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Gate result",
        "",
        f"- positive states: `{analysis['positive_state_count']}` / `{analysis['state_count_total']}`; negative/neutral states: `{analysis['negative_or_neutral_state_count']}`.",
        f"- positive cases: `{analysis['positive_cases']}`; positive cases outside previously mined case7/case10: `{analysis['positive_cases_outside_previously_mined_case7_case10']}`.",
        f"- positive horizon counts: `{analysis['positive_horizon_counts']}`.",
        f"- high states positive: `{analysis['high_positive_count']}` / `{analysis['high_state_count']}`; low/control positive: `{analysis['low_control_positive_count']}` / `{analysis['low_control_state_count']}`.",
        f"- best non-H15 total gain: `{analysis['best_non_H15_total_gain']}`; best non-H15 physical gain: `{analysis['best_non_H15_physical_gain']}`; large non-H15 harms: `{analysis['large_non_H15_harms_total_gain_le_minus_threshold']}`.",
        f"- artifact flags: `{analysis['artifact_flags']}`; blocking_artifact_count: `{analysis['blocking_artifact_count']}`.",
        f"- transient_missed_timing_supported_gate_pass: `{analysis['transient_missed_timing_supported_gate_pass']}`.",
        f"- scenario_scarcity_supported_gate_fail_clean: `{analysis['scenario_scarcity_supported_gate_fail_clean']}`.",
        f"- Gate rule: {analysis['gate_rule']}",
        "",
        "## Target labels",
        "",
        "| target | kind | case | step | score | label | material H | H15 cont total | best total H | best total gain | best phys H | best phys gain |",
        "|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["state_rows"]:
        ref = row["reference_H15"]
        bp = row.get("best_physical") or {}
        bt = row.get("best_total") or {}
        lines.append("| %d | `%s` | %d | %d | %.6g | `%s` | `%s` | %.6g | %s | %.6g | %s | %.6g |" % (
            int(row["target_index"]),
            row["kind"],
            int(row["case"]),
            int(row["branch_step"]),
            float(row["target_transient_score"]),
            row["label"],
            row["material_positive_horizons"],
            float(ref["continuation_total"]),
            str(bt.get("horizon")),
            float(bt.get("gain_vs_H15_total", 0.0)),
            str(bp.get("horizon")),
            float(bp.get("gain_vs_H15_physical", 0.0)),
        ))
    lines += [
        "",
        "## Decision rule for next work",
        "",
    ]
    if analysis["transient_missed_timing_supported_gate_pass"]:
        lines.append("Gate pass: freeze a compact supervised/refit selector using transient-state features, then confirm on a separate fresh development bank with strong fixed-H references before any validation/test claim.")
    elif analysis["scenario_scarcity_supported_gate_fail_clean"]:
        lines.append("Clean gate fail: the canonical fresh bank still lacks enough reusable state-dependent horizon labels at high-transient H15 states; next work should freeze a versioned source-supported stress-scenario opportunity protocol before retraining.")
    else:
        lines.append("Inconclusive gate: artifacts or missing comparisons require a targeted one-variable repair/diagnostic before selector refit or stress-scenario redesign.")
    lines.append("")
    lines.append(f"Backup request before further simulations: `{raw['backup_request']}`.")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    analysis = raw["analysis"]
    if analysis["transient_missed_timing_supported_gate_pass"]:
        next_action = "freeze compact selector/refit confirmation on a separate fresh development bank"
    elif analysis["scenario_scarcity_supported_gate_fail_clean"]:
        next_action = "freeze versioned stress-scenario opportunity protocol before retraining"
    else:
        next_action = "repair/diagnose rollout artifacts before interpreting opportunity scarcity"
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 transient-state continuation probe v0\n\n"
        f"UTC: {raw['created_utc']}. Development-only H15-prefix transient/control continuation probe completed: "
        f"{raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps. No validation64/test/training access. "
        f"Positive states={analysis['positive_state_count']}/{analysis['state_count_total']}; positive cases outside mined case7/case10={analysis['positive_cases_outside_previously_mined_case7_case10']}; "
        f"gate_pass={analysis['transient_missed_timing_supported_gate_pass']}; clean_scenario_scarcity_fail={analysis['scenario_scarcity_supported_gate_fail_clean']}; next={next_action}. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        if p.exists():
            old = p.read_text(encoding="utf-8")
            if MARKER not in old:
                p.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup-proof", required=True, type=Path, help="verified external backup proof after this runner source and v0b target artifacts are backed up")
    ap.add_argument("--i-accept-transient-state-development-rollout", action="store_true", help="explicit acknowledgement: development continuation probe only, no validation64/test access")
    args = ap.parse_args(argv)
    if not args.i_accept_transient_state_development_rollout:
        raise ContractError("explicit --i-accept-transient-state-development-rollout is required")
    assert_fresh_output_dir()
    protocol, v0b_raw, bank, fixed_v1_raw = verify_inputs()
    min_backup_time = latest_time(
        parse_time(protocol.get("created_utc")),
        parse_time(v0b_raw.get("created_utc")),
        source_mtime_utc(),
    )
    backup = verify_backup_proof(args.backup_proof, min_backup_time)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "vehicle_v1_transient_state_continuation_probe_v0",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "training_gradient_steps": 0,
    })
    preflight = fresh.base.runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError(str(preflight.get("diagnosis", "runtime preflight failed")) + " " + str(preflight.get("exception", "")))
    fresh.v1.latency_verify()
    selected_cases = bank["selected_cases"]
    selected_meta = bank["selection"]["selected_metadata"]
    terminals, terminal_receipts = fresh.base.load_terminal_grid(fixed_v1_raw["protocol_full"])
    write_json(OUT_DIR / "terminal_sources.json", terminal_receipts)
    targets = list(v0b_raw["selected_targets"])
    target_by_index = {int(t["target_index"]): t for t in targets}
    schedule = build_schedule(targets)
    write_json(OUT_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "branch_horizons": BRANCH_HORIZONS, "prefix_horizon": PREFIX_H, "source_targets": targets})
    episodes: List[Dict[str, Any]] = []
    for item in schedule:
        case_id = int(item["case"])
        target = target_by_index[int(item["target_index"])]
        summary = fresh.run_episode(item, selected_cases[case_id], selected_meta[case_id], terminals, terminal_receipts)
        summary = annotate_episode(summary, item, target)
        # Rewrite the episode summary after annotation so raw episode files carry the target lineage too.
        if summary.get("path"):
            write_json(ROOT / str(summary["path"]) / "summary.json", summary)
        episodes.append(summary)
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(int(e["steps"]) for e in episodes)),
            "last_episode": {k: summary[k] for k in ("execution_index", "target_index", "target_kind", "case", "branch_step", "branch_horizon", "steps", "success", "termination", "branch_reached")},
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    design = protocol["rollout_design_after_backup"]
    if len(episodes) != 48 or len(episodes) > int(design["max_rollout_episodes"]) or control_steps > int(design["control_step_upper_bound"]):
        raise ContractError("transient-state continuation budget violation")
    analysis = analyze_transient_labels(episodes, targets)
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "vehicle_v1_transient_state_continuation_probe_v0_identical_H15_prefix_branch_fixed_H",
        "classification": "development_IMPROVED_transient_state_continuation_diagnostic_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "backup_proof": backup,
        "protocol": {"path": rel(PROTOCOL_MD), "sha256": sha256(PROTOCOL_MD), "json_path": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)},
        "protocol_full": protocol,
        "v0b_target_source": {"raw": rel(V0B_RAW), "raw_sha256": sha256(V0B_RAW), "completed": rel(V0B_COMPLETED), "completed_sha256": sha256(V0B_COMPLETED)},
        "fresh_probe_source": {"raw": rel(FRESH_RAW), "raw_sha256": sha256(FRESH_RAW), "completed": rel(FRESH_COMPLETED), "completed_sha256": sha256(FRESH_COMPLETED), "bank": rel(FRESH_BANK), "bank_sha256": sha256(FRESH_BANK)},
        "fixed_H_v1_source": {"raw": rel(V1_RAW), "raw_sha256": sha256(V1_RAW), "completed": rel(V1_COMPLETED), "completed_sha256": sha256(V1_COMPLETED)},
        "source_hashes": source_hashes(),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": {"max_rollout_episodes": int(design["max_rollout_episodes"]), "control_step_upper_bound": int(design["control_step_upper_bound"]), "new_training_episodes": 0, "new_gradient_steps": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "environment_constructions": len(episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "targets": targets,
        "schedule": schedule,
        "terminal_sources": terminal_receipts,
        "episodes": episodes,
        "analysis": analysis,
        "interpretation_limits": [
            "development diagnostic only",
            "targets selected from H15 traces of the already-opened fresh development probe",
            "non-H15 outcomes were ignored during target selection but are development-known after this run",
            "not online adaptive selector validation",
            "not new training/refit",
            "not ORIGINAL SAC",
            "not final test",
            "actual timing from AWS run; terminal-switch overhead reported separately",
        ],
    }
    raw["backup_request"] = write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if analysis["transient_missed_timing_supported_gate_pass"]:
        next_action = "freeze compact selector/refit confirmation on separate fresh development bank"
    elif analysis["scenario_scarcity_supported_gate_fail_clean"]:
        next_action = "freeze versioned stress-scenario opportunity protocol before retraining"
    else:
        next_action = "repair/diagnose rollout artifacts before interpreting labels"
    STATE_PATH.write_text(
        f"# Vehicle V1 transient-state continuation probe v0 state ({raw['created_utc']})\n\n"
        f"Completed {len(episodes)} episodes / {control_steps} control steps. No validation64/test/training access. "
        f"Positive states={analysis['positive_state_count']}/{analysis['state_count_total']}; gate_pass={analysis['transient_missed_timing_supported_gate_pass']}; "
        f"clean_scenario_scarcity_fail={analysis['scenario_scarcity_supported_gate_fail_clean']}. Next: {next_action}. Backup required before further simulations.\n",
        encoding="utf-8",
    )
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [ROOT / raw["backup_request"], STATE_PATH, Path(__file__).resolve(), PROTOCOL_MD, PROTOCOL_JSON, V0B_RAW, V0B_COMPLETED, FRESH_COMPLETED, V1_COMPLETED]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_request": raw["backup_request"],
        "headline": {
            "positive_state_count": analysis["positive_state_count"],
            "negative_or_neutral_state_count": analysis["negative_or_neutral_state_count"],
            "positive_cases": analysis["positive_cases"],
            "positive_cases_outside_previously_mined_case7_case10": analysis["positive_cases_outside_previously_mined_case7_case10"],
            "transient_missed_timing_supported_gate_pass": analysis["transient_missed_timing_supported_gate_pass"],
            "scenario_scarcity_supported_gate_fail_clean": analysis["scenario_scarcity_supported_gate_fail_clean"],
            "next_action": next_action,
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "episodes": len(episodes),
        "control_steps": control_steps,
        "positive_state_count": analysis["positive_state_count"],
        "negative_or_neutral_state_count": analysis["negative_or_neutral_state_count"],
        "positive_cases": analysis["positive_cases"],
        "positive_cases_outside_previously_mined_case7_case10": analysis["positive_cases_outside_previously_mined_case7_case10"],
        "transient_missed_timing_supported_gate_pass": analysis["transient_missed_timing_supported_gate_pass"],
        "scenario_scarcity_supported_gate_fail_clean": analysis["scenario_scarcity_supported_gate_fail_clean"],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": raw["backup_request"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(OUT_DIR / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "next_recovery_hint": "Preserve partial directory, audit failure, and create a versioned one-variable repair. Use legacy interpreter and a verified backup proof after this runner source is externally backed up.",
        })
        raise
