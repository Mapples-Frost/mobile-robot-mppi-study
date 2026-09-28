#!/usr/bin/env python3
"""Vehicle stress-v1b matched-continuation rollout runner v0.

Development-only IMPROVED diagnostic after the stress-v1 Stage1 fixed-H map and
v1b prepare/dry-run.  The prepare artifacts froze 20 H15-prefix target states
from positive high-heading cases, same-stratum controls, and lower-stress
controls.  This runner provides:

* --dry-run: no simulation; verifies the frozen v1b protocol, post-prepare
  backup proof, legacy runtime import, terminal metadata and budget/access
  gates, then writes a backup request for the real rollout.
* --run-rollout: after a new external backup covers this runner and dry-run
  outputs, executes 20 targets x 7 branch horizons = 140 identical-prefix
  continuation episodes.

It never opens validation64 or sealed final-test banks and performs no training,
refit, or gradient updates.  The rollout is development label evidence only.
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
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_stress_scenario_opportunity_probe_v1_runner as stage1_runner  # noqa:E402
import vehicle_v1_fresh_continuation_label_probe_v0_runner as fresh  # noqa:E402

TASK = "vehicle"
STAMP = "20260928T2215Z"
PREFIX_H = 15
BRANCH_HORIZONS = [10, 15, 25, 30, 35, 45, 50]
MAX_STEPS = 150
ORDER_SEED = 2609289501
MATERIAL_GAIN_THRESHOLD = 3.0
STATE_DISTANCE_TOLERANCE = 1e-5

PREPARE_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_prepare_20260928T2210Z_schema_repair"
PREPARE_COMPLETED = PREPARE_DIR / "completed.json"
PREPARE_RAW = PREPARE_DIR / "raw.json"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1b_matched_continuation_frozen_20260928T2210Z_schema_repair.json"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1b_matched_continuation_frozen_20260928T2210Z_schema_repair.md"
STAGE1_BANK = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/bank/vehicle_stress_scenario_opportunity_probe_v1_bank.json"
STAGE1_PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.json"
STAGE1_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/completed.json"
POST_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/completed.json"

DRYRUN_DIR = ROOT / ("research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_rollout_v0_dryrun_%s" % STAMP)
RUN_DIR = ROOT / ("research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_rollout_v0_%s" % STAMP)
DRYRUN_STATE = ROOT / ("research_artifacts/aws_state/vehicle_stress_v1b_matched_continuation_rollout_v0_dryrun_%s.md" % STAMP)
RUN_STATE = ROOT / ("research_artifacts/aws_state/vehicle_stress_v1b_matched_continuation_rollout_v0_run_%s.md" % STAMP)
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP_BEFORE_ROLLOUT = BACKUP_DIR / ("REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1B_MATCHED_CONTINUATION_ROLLOUT_V0_%s.json" % STAMP)
MARKER_DRYRUN = "vehicle-stress-v1b-matched-continuation-rollout-v0-dryrun-%s" % STAMP
MARKER_RUN = "vehicle-stress-v1b-matched-continuation-rollout-v0-run-%s" % STAMP


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


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


def clean_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_jsonable(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return clean_jsonable(value.tolist())
    if hasattr(value, "item"):
        return clean_jsonable(value.item())
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


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def completed_passed(path: Path, check_hashes: bool = False) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("missing completed marker: %s" % rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError("completed marker did not pass: %s" % rel(path))
    if obj.get("sealed_test_accessed") is not False:
        raise ContractError("sealed-test flag not false in %s" % rel(path))
    if obj.get("historical_validation64_bank_opened") not in (False, None):
        raise ContractError("historical validation64 flag not false in %s" % rel(path))
    if check_hashes:
        for name, expected in (obj.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists():
                raise ContractError("completed hash references missing file: %s" % name)
            if sha256(p) != expected:
                raise ContractError("completed hash mismatch for %s" % name)
    return obj


def verify_backup_proof(path: Path, min_time: Optional[dt.datetime], purpose: str) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("backup proof path does not exist: %s" % rel(path))
    proof = read_json(path)
    if not (proof.get("backup_verified") is True or proof.get("status") == "verified"):
        raise ContractError("backup proof is not verified")
    if int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("backup proof does not record remaining_changed_files=0")
    if not proof.get("commit"):
        raise ContractError("backup proof lacks commit")
    packages = proof.get("packages_this_run") or []
    if not (proof.get("asset_sha256") or proof.get("release_asset_sha256") or proof.get("package_sha256") or packages):
        raise ContractError("backup proof lacks release/package SHA")
    for pkg in packages:
        if not pkg.get("sha256") or not pkg.get("bytes") or not pkg.get("verification"):
            raise ContractError("backup package entry lacks sha256/bytes/verification")
    proof_time = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        proof_time = parse_time(proof.get(key))
        if proof_time is not None:
            break
    if min_time is not None:
        if proof_time is None:
            raise ContractError("backup proof lacks parseable time for %s" % purpose)
        if proof_time < min_time:
            raise ContractError("backup proof predates required artifact time for %s" % purpose)
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "time": proof_time.isoformat() if proof_time else None,
        "min_required_time_utc": None if min_time is None else min_time.isoformat(),
        "commit": proof.get("commit"),
        "remaining_changed_files": proof.get("remaining_changed_files"),
        "packages_this_run": packages,
        "purpose": purpose,
    }


def source_hashes(extra: Sequence[Path] = ()) -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(fresh.__file__).resolve(),
        Path(stage1_runner.__file__).resolve(),
        PROTOCOL_JSON,
        PROTOCOL_MD,
        PREPARE_COMPLETED,
        PREPARE_RAW,
        STAGE1_BANK,
        STAGE1_PROTOCOL_JSON,
        STAGE1_COMPLETED,
        POST_COMPLETED,
    ] + list(extra)
    return {rel(p): sha256(p) for p in paths if p.exists()}


def verify_inputs(check_prepare_hashes: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for p in (PREPARE_COMPLETED, PREPARE_RAW, PROTOCOL_JSON, PROTOCOL_MD, STAGE1_BANK, STAGE1_PROTOCOL_JSON, STAGE1_COMPLETED, POST_COMPLETED):
        if not p.exists():
            raise ContractError("required v1b rollout input missing: %s" % rel(p))
    prepare_done = completed_passed(PREPARE_COMPLETED, check_hashes=check_prepare_hashes)
    stage1_done = completed_passed(STAGE1_COMPLETED, check_hashes=False)
    post_done = completed_passed(POST_COMPLETED, check_hashes=False)
    protocol = read_json(PROTOCOL_JSON)
    prepare_raw = read_json(PREPARE_RAW)
    bank = read_json(STAGE1_BANK)
    if prepare_done.get("stage1_gate_preserved_failed") is not True:
        raise ContractError("v1b prepare did not preserve failed Stage1 diversity gate")
    if int(prepare_done.get("target_count", -1)) != 20:
        raise ContractError("v1b prepare target count mismatch")
    budget = protocol.get("budget") or {}
    if int(budget.get("rollout_episodes_exact", -1)) != 140 or int(budget.get("control_step_upper_bound", -1)) != 21000:
        raise ContractError("v1b frozen rollout budget mismatch")
    if [int(x) for x in budget.get("branch_horizons", [])] != BRANCH_HORIZONS:
        raise ContractError("v1b branch horizon list changed")
    if int(budget.get("prefix_horizon", -1)) != PREFIX_H:
        raise ContractError("v1b prefix horizon changed")
    if int(budget.get("new_training_episodes", -1)) != 0 or int(budget.get("new_gradient_steps", -1)) != 0:
        raise ContractError("v1b protocol unexpectedly permits training")
    if canonical_sha(protocol.get("targets")) != canonical_sha(prepare_raw.get("protocol", {}).get("targets")):
        raise ContractError("v1b protocol targets do not match prepare raw")
    if stage1_done.get("historical_validation64_bank_opened") is not False or post_done.get("historical_validation64_bank_opened") is not False:
        raise ContractError("v1b source unexpectedly opened validation64")
    selected_cases = bank.get("selected_cases") or []
    selected_meta = (bank.get("selection") or {}).get("selected_metadata") or []
    if len(selected_cases) <= max(int(t["case"]) for t in protocol["targets"]):
        raise ContractError("Stage1 bank selected_cases too short for v1b targets")
    if len(selected_meta) <= max(int(t["case"]) for t in protocol["targets"]):
        raise ContractError("Stage1 bank selected_metadata too short for v1b targets")
    return prepare_done, prepare_raw, protocol, bank


def build_schedule(targets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in targets:
        for h in BRANCH_HORIZONS:
            rows.append({
                "target_index": int(t["target_index"]),
                "case": int(t["case"]),
                "branch_step": int(t["branch_step"]),
                "branch_horizon": int(h),
                "selection_role": t.get("selection_role"),
                "window_id": t.get("window_id"),
            })
    rng = random.Random(ORDER_SEED)
    order = list(range(len(rows)))
    rng.shuffle(order)
    return [dict(rows[i], execution_index=j, schedule_base_index=i) for j, i in enumerate(order)]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def state_tuple(row: Mapping[str, Any], key: str = "branch_previous_state") -> Tuple[float, float, float]:
    state = row.get(key) or {}
    return (safe_float(state.get("x"), float("nan")), safe_float(state.get("y"), float("nan")), safe_float(state.get("theta"), float("nan")))


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def state_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    if not all(math.isfinite(x) for x in a + b):
        return float("inf")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5 * abs(angle_diff(a[2], b[2])))


def no_regression(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    if bool(reference.get("success")) and not bool(candidate.get("success")):
        return False
    if bool(candidate.get("constraint")) and not bool(reference.get("constraint")):
        return False
    for key in ("initial_failed_steps", "final_failed_steps", "solver_failure_steps"):
        if int(candidate.get(key, 0)) > int(reference.get(key, 0)):
            return False
    return True


def value_summary(values: Iterable[float]) -> Dict[str, Any]:
    xs = sorted([float(x) for x in values if math.isfinite(float(x))])
    if not xs:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "max": None}
    n = len(xs)
    med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    return {"count": n, "sum": float(sum(xs)), "mean": float(sum(xs) / n), "median": float(med), "max": float(xs[-1])}


def analyze_v1b(episodes: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[int, List[Mapping[str, Any]]] = {}
    target_by_index = {int(t["target_index"]): t for t in targets}
    for ep in episodes:
        groups.setdefault(int(ep["target_index"]), []).append(ep)
    artifact_flags = {
        "missing_H15_reference": 0,
        "missing_horizon": 0,
        "reference_branch_not_reached": 0,
        "candidate_branch_not_reached": 0,
        "prefix_mismatch_non_H15": 0,
        "state_distance_gt_tol_non_H15": 0,
        "safety_solver_regression_non_H15": 0,
    }
    state_rows: List[Dict[str, Any]] = []
    positive_cases: List[int] = []
    positive_horizon_counts: Dict[str, int] = {}
    best_non_h15_total_gain = -float("inf")
    best_non_h15_physical_gain = -float("inf")
    large_harms = 0
    total_decision_by_h: Dict[str, float] = {str(h): 0.0 for h in BRANCH_HORIZONS}
    total_decision_plus_switch_by_h: Dict[str, float] = {str(h): 0.0 for h in BRANCH_HORIZONS}
    total_solver_by_h: Dict[str, float] = {str(h): 0.0 for h in BRANCH_HORIZONS}
    steps_by_h: Dict[str, int] = {str(h): 0 for h in BRANCH_HORIZONS}
    for tid in sorted(target_by_index):
        target = target_by_index[tid]
        rows = sorted(groups.get(tid, []), key=lambda r: int(r.get("branch_horizon", -1)))
        by_h = {int(r["branch_horizon"]): r for r in rows}
        for h in BRANCH_HORIZONS:
            if h not in by_h:
                artifact_flags["missing_horizon"] += 1
        ref = by_h.get(PREFIX_H)
        if ref is None:
            artifact_flags["missing_H15_reference"] += 1
            state_rows.append({"target_index": tid, "case": int(target["case"]), "selection_role": target.get("selection_role"), "branch_step": int(target["branch_step"]), "label": "missing_H15_reference", "comparisons": []})
            continue
        if not bool(ref.get("branch_reached")):
            artifact_flags["reference_branch_not_reached"] += 1
        ref_state = state_tuple(ref)
        material_horizons: List[int] = []
        comparisons: List[Dict[str, Any]] = []
        best_physical = None
        best_total = None
        for h in BRANCH_HORIZONS:
            row = by_h.get(h)
            if row is None:
                continue
            hs = str(h)
            total_decision_by_h[hs] += safe_float((row.get("decision_timing_s") or {}).get("sum"), 0.0)
            total_decision_plus_switch_by_h[hs] += safe_float((row.get("decision_plus_terminal_switch_timing_s") or {}).get("sum"), 0.0)
            total_solver_by_h[hs] += safe_float((row.get("solver_attempt_timing_s") or {}).get("sum"), 0.0)
            steps_by_h[hs] += int(row.get("steps", 0))
            branch_reached = bool(row.get("branch_reached"))
            if not branch_reached:
                artifact_flags["candidate_branch_not_reached"] += 1
            cand_state = state_tuple(row)
            dist = state_distance(cand_state, ref_state) if branch_reached and bool(ref.get("branch_reached")) else float("inf")
            prefix_match = bool(row.get("prefix_clean_sha256") == ref.get("prefix_clean_sha256"))
            if h != PREFIX_H and not prefix_match:
                artifact_flags["prefix_mismatch_non_H15"] += 1
            if h != PREFIX_H and branch_reached and dist > STATE_DISTANCE_TOLERANCE:
                artifact_flags["state_distance_gt_tol_non_H15"] += 1
            phys_gain = float(ref["continuation_physical_constraint_cost_from_branch"] - row["continuation_physical_constraint_cost_from_branch"]) if branch_reached and bool(ref.get("branch_reached")) else float("nan")
            total_gain = float(ref["continuation_total_cost_from_branch"] - row["continuation_total_cost_from_branch"]) if branch_reached and bool(ref.get("branch_reached")) else float("nan")
            safety_ok = no_regression(row, ref)
            if h != PREFIX_H and not safety_ok:
                artifact_flags["safety_solver_regression_non_H15"] += 1
            if h != PREFIX_H and math.isfinite(total_gain):
                best_non_h15_total_gain = max(best_non_h15_total_gain, total_gain)
                if total_gain <= -MATERIAL_GAIN_THRESHOLD:
                    large_harms += 1
            if h != PREFIX_H and math.isfinite(phys_gain):
                best_non_h15_physical_gain = max(best_non_h15_physical_gain, phys_gain)
            material = bool(h != PREFIX_H and branch_reached and bool(ref.get("branch_reached")) and prefix_match and dist <= STATE_DISTANCE_TOLERANCE and safety_ok and (phys_gain >= MATERIAL_GAIN_THRESHOLD or total_gain >= MATERIAL_GAIN_THRESHOLD))
            if material:
                material_horizons.append(h)
                positive_horizon_counts[str(h)] = positive_horizon_counts.get(str(h), 0) + 1
            comp = {
                "horizon": h,
                "branch_reached": branch_reached,
                "success": bool(row.get("success")),
                "constraint": bool(row.get("constraint")),
                "steps": int(row.get("steps", 0)),
                "continuation_physical": safe_float(row.get("continuation_physical_constraint_cost_from_branch"), 0.0),
                "continuation_total": safe_float(row.get("continuation_total_cost_from_branch"), 0.0),
                "gain_vs_H15_physical": phys_gain,
                "gain_vs_H15_total": total_gain,
                "decision_sum_s": safe_float((row.get("decision_timing_s") or {}).get("sum"), 0.0),
                "decision_plus_terminal_switch_sum_s": safe_float((row.get("decision_plus_terminal_switch_timing_s") or {}).get("sum"), 0.0),
                "solver_attempt_sum_s": safe_float((row.get("solver_attempt_timing_s") or {}).get("sum"), 0.0),
                "state_distance_vs_H15_branch_state": dist,
                "prefix_clean_sha256_matches_H15": prefix_match,
                "no_success_constraint_solver_regression_vs_H15": safety_ok,
                "material_positive_vs_H15": material,
                "initial_failed_steps": int(row.get("initial_failed_steps", 0)),
                "final_failed_steps": int(row.get("final_failed_steps", 0)),
                "solver_failure_steps": int(row.get("solver_failure_steps", 0)),
                "path": row.get("path"),
            }
            comparisons.append(comp)
            if branch_reached:
                if best_physical is None or (comp["continuation_physical"], comp["decision_plus_terminal_switch_sum_s"], h) < (best_physical["continuation_physical"], best_physical["decision_plus_terminal_switch_sum_s"], best_physical["horizon"]):
                    best_physical = comp
                if best_total is None or (comp["continuation_total"], comp["decision_plus_terminal_switch_sum_s"], h) < (best_total["continuation_total"], best_total["decision_plus_terminal_switch_sum_s"], best_total["horizon"]):
                    best_total = comp
        if material_horizons:
            positive_cases.append(int(target["case"]))
        state_rows.append({
            "target_index": tid,
            "case": int(target["case"]),
            "selection_role": target.get("selection_role"),
            "window_id": target.get("window_id"),
            "branch_step": int(target["branch_step"]),
            "stage1_material_positive_case": bool(target.get("stage1_material_positive_case")),
            "material_positive_state": bool(material_horizons),
            "material_positive_horizons": material_horizons,
            "label": "positive_non_H15" if material_horizons else "negative_or_neutral",
            "reference_H15": {
                "success": bool(ref.get("success")),
                "constraint": bool(ref.get("constraint")),
                "steps": int(ref.get("steps", 0)),
                "continuation_physical": safe_float(ref.get("continuation_physical_constraint_cost_from_branch"), 0.0),
                "continuation_total": safe_float(ref.get("continuation_total_cost_from_branch"), 0.0),
                "branch_previous_state": ref.get("branch_previous_state"),
                "initial_failed_steps": int(ref.get("initial_failed_steps", 0)),
                "final_failed_steps": int(ref.get("final_failed_steps", 0)),
                "solver_failure_steps": int(ref.get("solver_failure_steps", 0)),
            },
            "best_physical": best_physical,
            "best_total": best_total,
            "comparisons": comparisons,
        })
    positive_state_count = int(sum(1 for r in state_rows if r.get("material_positive_state")))
    negative_neutral_state_count = int(sum(1 for r in state_rows if r.get("label") == "negative_or_neutral"))
    negative_control_states = int(sum(1 for r in state_rows if r.get("label") == "negative_or_neutral" and r.get("selection_role") in ("same_stratum_stage1_nonmaterial_control", "lower_stress_control")))
    blocking_artifacts = int(artifact_flags["missing_H15_reference"] + artifact_flags["missing_horizon"] + artifact_flags["reference_branch_not_reached"] + artifact_flags["prefix_mismatch_non_H15"] + artifact_flags["state_distance_gt_tol_non_H15"])
    distinct_positive_cases = sorted(set(positive_cases))
    gate = bool(len(distinct_positive_cases) >= 3 and positive_state_count >= 6 and negative_control_states >= 2 and blocking_artifacts == 0)
    return {
        "state_count": len(state_rows),
        "positive_state_count": positive_state_count,
        "negative_neutral_state_count": negative_neutral_state_count,
        "positive_cases": distinct_positive_cases,
        "positive_horizon_counts": positive_horizon_counts,
        "negative_control_state_count": negative_control_states,
        "artifact_flags": artifact_flags,
        "blocking_artifact_count": blocking_artifacts,
        "best_non_H15_total_gain": None if best_non_h15_total_gain == -float("inf") else float(best_non_h15_total_gain),
        "best_non_H15_physical_gain": None if best_non_h15_physical_gain == -float("inf") else float(best_non_h15_physical_gain),
        "large_non_H15_harms": int(large_harms),
        "timing_totals_by_horizon_s": {
            str(h): {
                "steps": steps_by_h[str(h)],
                "decision_sum_s": total_decision_by_h[str(h)],
                "decision_plus_terminal_switch_sum_s": total_decision_plus_switch_by_h[str(h)],
                "solver_attempt_sum_s": total_solver_by_h[str(h)],
            }
            for h in BRANCH_HORIZONS
        },
        "training_refit_label_gate_pass_development_only": gate,
        "gate_rule": ">=3 distinct cases and >=6 matched states with material non-H15 positives, >=2 retained negative/control states, and no missing/prefix/state-distance/reference artifacts",
        "state_rows": state_rows,
    }


def write_dryrun_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle stress-v1b matched-continuation rollout v0 dry-run",
        "",
        "UTC: `%s`. No simulations, no training/refit, no validation64-bank access, no sealed-test access." % raw["created_utc"],
        "",
        "## Verified design",
        "",
        "- Frozen v1b prepare: `%s`, target_count `%d`." % (raw["prepare"]["completed"], raw["target_count"]),
        "- Backup proof used for prepare artifacts: `%s` commit `%s`." % (raw["backup_proof"]["path"], raw["backup_proof"]["commit"]),
        "- Prefix H%d; branch horizons `%s`; planned episodes `%d`; control-step cap `%d`." % (PREFIX_H, BRANCH_HORIZONS, raw["planned_rollout_episodes"], raw["planned_control_step_upper_bound"]),
        "- Runtime preflight passed: `%s`; terminal metadata horizons verified: `%s`." % (raw["runtime_preflight"].get("passed"), sorted(raw["terminal_metadata"].keys(), key=int)),
        "",
        "## Next gate",
        "",
        "The 140-episode rollout remains blocked until external backup covers this new runner source, dry-run outputs, state/docs and backup request.",
        "",
        "Backup request before rollout: `%s`." % raw["backup_request_before_rollout"],
    ]
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    (DRYRUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = [
        "# Vehicle stress-v1b matched-continuation rollout v0",
        "",
        "UTC: `%s`. Development-only identical-prefix continuation diagnostic; no training/refit, no validation64-bank access, no sealed-test access." % raw["created_utc"],
        "",
        "## Budget/access",
        "",
        "- Backup proof: `%s` commit `%s`." % (raw["backup_proof"]["path"], raw["backup_proof"]["commit"]),
        "- Episodes: `%d` / `%d`; control steps: `%d` / cap `%d`." % (raw["budget_actual"]["episodes"], raw["budget_declared"]["rollout_episodes_exact"], raw["budget_actual"]["control_steps"], raw["budget_declared"]["control_step_upper_bound"]),
        "- historical_validation64_bank_opened: `%s`; sealed_test_accessed: `%s`." % (raw["historical_validation64_bank_opened"], raw["sealed_test_accessed"]),
        "",
        "## Label result",
        "",
        "- Positive matched states: `%d` across cases `%s`; negative/neutral states: `%d`." % (a["positive_state_count"], a["positive_cases"], a["negative_neutral_state_count"]),
        "- Positive horizon counts: `%s`; negative/control states: `%d`." % (a["positive_horizon_counts"], a["negative_control_state_count"]),
        "- Blocking artifact count: `%d`; flags: `%s`." % (a["blocking_artifact_count"], a["artifact_flags"]),
        "- Best non-H15 total gain: `%s`; best physical gain: `%s`; large harms: `%d`." % (a["best_non_H15_total_gain"], a["best_non_H15_physical_gain"], a["large_non_H15_harms"]),
        "- training_refit_label_gate_pass_development_only: `%s`." % a["training_refit_label_gate_pass_development_only"],
        "",
        "## Per-target labels",
        "",
        "| target | case | role | window | branch step | label | material H | H15 cont phys | H15 cont total | best phys H/gain | best total H/gain |",
        "|---:|---:|---|---|---:|---|---|---:|---:|---|---|",
    ]
    for r in a["state_rows"]:
        ref = r.get("reference_H15") or {}
        bp = r.get("best_physical") or {}
        bt = r.get("best_total") or {}
        lines.append("| %d | %d | `%s` | `%s` | %d | `%s` | `%s` | %.6g | %.6g | H%s/%.6g | H%s/%.6g |" % (
            int(r["target_index"]), int(r["case"]), r.get("selection_role"), r.get("window_id"), int(r["branch_step"]), r.get("label"), r.get("material_positive_horizons"),
            float(ref.get("continuation_physical", 0.0)), float(ref.get("continuation_total", 0.0)), str(bp.get("horizon")), float(bp.get("gain_vs_H15_physical", 0.0)), str(bt.get("horizon")), float(bt.get("gain_vs_H15_total", 0.0)),
        ))
    lines += ["", "## Decision", ""]
    if a["training_refit_label_gate_pass_development_only"]:
        lines.append("The predeclared development label gate passed. Next freeze a compact IMPROVED selector/refit smoke, then use fresh confirmation and fair same-distribution fixed-H baselines; this v1b evidence remains development-only.")
    else:
        lines.append("The predeclared development label gate failed. Do not train on sparse labels; preserve the negative evidence and pivot to the next evidence-ranked scenario/reward/terminal/modeling diagnostic.")
    lines.append("")
    lines.append("Backup request after rollout: `%s`." % raw["backup_request_after_rollout"])
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if marker not in text:
                path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def run_dry_run(args: argparse.Namespace) -> int:
    if (DRYRUN_DIR / "completed.json").exists():
        completed_passed(DRYRUN_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"dry_run_already_completed": rel(DRYRUN_DIR / "completed.json"), "historical_validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    prepare_done, prepare_raw, protocol, bank = verify_inputs(check_prepare_hashes=True)
    min_time = parse_time(prepare_done.get("created_utc"))
    backup = verify_backup_proof(args.backup_proof, min_time, "v1b_prepare_outputs_before_rollout_runner_dryrun")
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %s" % preflight)
    stage1_protocol = read_json(STAGE1_PROTOCOL_JSON)
    terminal_meta = stage1_runner.verify_terminal_metadata(stage1_protocol["terminal_grid_readiness_reused_from_v1"])
    missing = [str(h) for h in BRANCH_HORIZONS if str(h) not in terminal_meta]
    if missing:
        raise ContractError("terminal metadata missing branch horizons: %s" % missing)
    schedule = build_schedule(protocol["targets"])
    budget = protocol["budget"]
    if len(schedule) != int(budget["rollout_episodes_exact"]):
        raise ContractError("dry-run schedule length mismatch")
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(REQUEST_BACKUP_BEFORE_ROLLOUT, {
        "requested_utc": created,
        "reason": "backup v1b rollout runner source and dry-run outputs before the 140-episode matched-continuation rollout",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "planned_rollout_episodes": int(budget["rollout_episodes_exact"]),
        "planned_control_step_upper_bound": int(budget["control_step_upper_bound"]),
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(Path(__file__).resolve()), rel(DRYRUN_DIR), rel(DRYRUN_STATE), rel(REQUEST_BACKUP_BEFORE_ROLLOUT), rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(PREPARE_DIR), rel(args.backup_proof)],
    })
    raw = {
        "created_utc": created,
        "method": "vehicle_stress_v1b_matched_continuation_rollout_v0_dryrun_no_simulation",
        "classification": "development_diagnostic_runner_smoke_no_simulation_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_proof": backup,
        "prepare": {"completed": rel(PREPARE_COMPLETED), "completed_sha256": sha256(PREPARE_COMPLETED), "raw": rel(PREPARE_RAW), "raw_sha256": sha256(PREPARE_RAW)},
        "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON), "md": rel(PROTOCOL_MD), "md_sha256": sha256(PROTOCOL_MD)},
        "target_count": len(protocol["targets"]),
        "target_case_counts": {str(c): sum(1 for t in protocol["targets"] if int(t["case"]) == c) for c in sorted(set(int(t["case"]) for t in protocol["targets"]))},
        "planned_rollout_episodes": int(budget["rollout_episodes_exact"]),
        "planned_control_step_upper_bound": int(budget["control_step_upper_bound"]),
        "schedule_preview_first10": schedule[:10],
        "runtime_preflight": preflight,
        "terminal_metadata": {str(h): terminal_meta[str(h)] for h in BRANCH_HORIZONS},
        "source_hashes": source_hashes(extra=[args.backup_proof, REQUEST_BACKUP_BEFORE_ROLLOUT]),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "backup_request_before_rollout": rel(REQUEST_BACKUP_BEFORE_ROLLOUT),
        "interpretation_limits": ["dry-run only", "no simulations", "not validation/model selection", "not final test"],
    }
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DRYRUN_DIR / "raw.json", raw)
    write_dryrun_summary(raw)
    DRYRUN_STATE.parent.mkdir(parents=True, exist_ok=True)
    DRYRUN_STATE.write_text(
        "# Vehicle stress-v1b matched-continuation rollout dry-run (%s)\n\nNo-simulation dry-run passed: target_count=%d, planned_episodes=%d, control-step cap=%d. No validation64/test/training. Rollout remains blocked until external backup covers this runner/dry-run/request.\n" % (created, len(protocol["targets"]), int(budget["rollout_episodes_exact"]), int(budget["control_step_upper_bound"])),
        encoding="utf-8",
    )
    block = """<!-- %s -->
## 2026-09-28 vehicle stress-v1b matched-continuation rollout v0 dry-run

UTC: %s. No-simulation runner dry-run passed after the verified post-prepare backup proof. It verified v1b prepare/protocol hashes, legacy runtime import, H10/H15/H25/H30/H35/H45/H50 terminal metadata and the 140-episode/21000-step budget. No validation64-bank or sealed-test access and no training/refit. Rollout is blocked until external backup covers `%s`, `%s`, state/docs and this runner source.
""" % (MARKER_DRYRUN, created, rel(REQUEST_BACKUP_BEFORE_ROLLOUT), rel(DRYRUN_DIR / "completed.json"))
    append_docs(block, MARKER_DRYRUN)
    files = [p for p in DRYRUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [DRYRUN_STATE, REQUEST_BACKUP_BEFORE_ROLLOUT, Path(__file__).resolve(), PROTOCOL_JSON, PROTOCOL_MD, PREPARE_COMPLETED, args.backup_proof]
    write_json(DRYRUN_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "target_count": len(protocol["targets"]),
        "planned_rollout_episodes": int(budget["rollout_episodes_exact"]),
        "planned_control_step_upper_bound": int(budget["control_step_upper_bound"]),
        "backup_required_before_rollout": True,
        "backup_request": rel(REQUEST_BACKUP_BEFORE_ROLLOUT),
        "next_after_backup": "run --run-rollout with a verified backup proof postdating this dry-run and runner source",
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(DRYRUN_DIR / "completed.json"),
        "summary": rel(DRYRUN_DIR / "summary.md"),
        "planned_rollout_episodes": int(budget["rollout_episodes_exact"]),
        "planned_control_step_upper_bound": int(budget["control_step_upper_bound"]),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "backup_request": rel(REQUEST_BACKUP_BEFORE_ROLLOUT),
        "next": "await_verified_external_backup_before_v1b_rollout",
    }, sort_keys=True), flush=True)
    return 0


def run_rollout(args: argparse.Namespace) -> int:
    if not (DRYRUN_DIR / "completed.json").exists():
        raise ContractError("rollout requires completed dry-run marker")
    dry_done = completed_passed(DRYRUN_DIR / "completed.json", check_hashes=True)
    if (RUN_DIR / "completed.json").exists():
        completed_passed(RUN_DIR / "completed.json", check_hashes=True)
        raise SystemExit("v1b rollout already completed and verified; refusing rerun")
    if RUN_DIR.exists():
        leftovers = [p for p in RUN_DIR.iterdir() if p.name != "run.lock"]
        if leftovers:
            raise ContractError("partial v1b rollout output exists; inspect before rerun: " + ", ".join(rel(p) for p in leftovers[:20]))
    prepare_done, prepare_raw, protocol, bank = verify_inputs(check_prepare_hashes=True)
    min_time = max(parse_time(dry_done.get("created_utc")) or dt.datetime.now(dt.timezone.utc), source_mtime_utc())
    backup = verify_backup_proof(args.backup_proof, min_time, "v1b_rollout_after_runner_dryrun")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(RUN_DIR / "run_started.json", {"started_utc": started, "pid": os.getpid(), "method": "vehicle_stress_v1b_matched_continuation_rollout_v0", "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "training_gradient_steps": 0})
    preflight = stage1_runner.runtime_preflight()
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %s" % preflight)
    fresh.v1.latency_verify()
    stage1_protocol = read_json(STAGE1_PROTOCOL_JSON)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(stage1_protocol["terminal_grid_readiness_reused_from_v1"])
    write_json(RUN_DIR / "terminal_sources.json", {str(h): terminal_receipts[str(h)] for h in BRANCH_HORIZONS})
    fresh.OUT_DIR = RUN_DIR
    fresh.STATE_PATH = RUN_STATE
    fresh.PREFIX_H = PREFIX_H
    fresh.BRANCH_HORIZONS = list(BRANCH_HORIZONS)
    fresh.MAX_STEPS = MAX_STEPS
    fresh.IMPROVEMENT_ABS_THRESHOLD = MATERIAL_GAIN_THRESHOLD
    selected_cases = bank["selected_cases"]
    selected_meta = (bank.get("selection") or {}).get("selected_metadata") or []
    target_by_index = {int(t["target_index"]): t for t in protocol["targets"]}
    schedule = build_schedule(protocol["targets"])
    write_json(RUN_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "branch_horizons": BRANCH_HORIZONS, "prefix_horizon": PREFIX_H})
    episodes: List[Dict[str, Any]] = []
    for item in schedule:
        target = target_by_index[int(item["target_index"])]
        case_id = int(item["case"])
        summary = fresh.run_episode(item, selected_cases[case_id], selected_meta[case_id], terminals, terminal_receipts)
        summary.update({
            "target_index": int(item["target_index"]),
            "selection_role": target.get("selection_role"),
            "window_id": target.get("window_id"),
            "stage1_material_positive_case": bool(target.get("stage1_material_positive_case")),
            "target_h15_only_selection_features": target.get("h15_only_selection_features"),
        })
        write_json(ROOT / summary["path"] / "summary.json", summary)
        episodes.append(summary)
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(int(e.get("steps", 0)) for e in episodes)),
            "last_episode": {k: summary.get(k) for k in ("execution_index", "target_index", "case", "branch_step", "branch_horizon", "steps", "success", "termination", "branch_reached")},
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    budget = protocol["budget"]
    if len(episodes) != int(budget["rollout_episodes_exact"]) or control_steps > int(budget["control_step_upper_bound"]):
        raise ContractError("v1b rollout budget violation")
    analysis = analyze_v1b(episodes, protocol["targets"])
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    backup_request = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1B_MATCHED_CONTINUATION_ROLLOUT_V0_%s.json" % created.replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(backup_request, {"requested_utc": created, "reason": "backup v1b matched-continuation rollout before selector/refit or further simulations", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "artifacts": [rel(RUN_DIR), rel(RUN_STATE), rel(Path(__file__).resolve()), rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(backup_request)]})
    raw = {
        "created_utc": created,
        "started_utc": started,
        "method": "vehicle_stress_v1b_matched_continuation_rollout_v0_identical_H15_prefix_branch_fixed_H",
        "classification": "development_IMPROVED_matched_continuation_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "backup_proof": backup,
        "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON), "md": rel(PROTOCOL_MD), "md_sha256": sha256(PROTOCOL_MD)},
        "prepare": {"completed": rel(PREPARE_COMPLETED), "completed_sha256": sha256(PREPARE_COMPLETED)},
        "dry_run": {"completed": rel(DRYRUN_DIR / "completed.json"), "completed_sha256": sha256(DRYRUN_DIR / "completed.json")},
        "source_hashes": source_hashes(extra=[args.backup_proof]),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": budget,
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "environment_constructions": len(episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "schedule": schedule,
        "terminal_sources": {str(h): terminal_receipts[str(h)] for h in BRANCH_HORIZONS},
        "episodes": episodes,
        "analysis": analysis,
        "backup_request_after_rollout": rel(backup_request),
        "interpretation_limits": ["development diagnostic only", "matched continuations from replayed H15 prefix", "not online adaptive validation", "not new training/refit", "not final test"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_run_summary(raw)
    RUN_STATE.parent.mkdir(parents=True, exist_ok=True)
    RUN_STATE.write_text("# Vehicle stress-v1b matched-continuation rollout state (%s)\n\nCompleted %d episodes / %d control steps. Positive states=%d across cases=%s; negative/neutral=%d; label gate=%s. No validation64/test/training. Backup required before next simulation.\n" % (created, len(episodes), control_steps, analysis["positive_state_count"], analysis["positive_cases"], analysis["negative_neutral_state_count"], analysis["training_refit_label_gate_pass_development_only"]), encoding="utf-8")
    block = """<!-- %s -->
## 2026-09-28 vehicle stress-v1b matched-continuation rollout v0

UTC: %s. Development-only matched-continuation rollout completed: %d episodes, %d control steps. Positive states=%d across cases=%s, negative/neutral=%d, label gate=%s, blocking artifacts=%d. No training, no validation64-bank access, no sealed-test access. Artifacts: `%s`, `%s`, `%s`.
""" % (MARKER_RUN, created, len(episodes), control_steps, analysis["positive_state_count"], analysis["positive_cases"], analysis["negative_neutral_state_count"], analysis["training_refit_label_gate_pass_development_only"], analysis["blocking_artifact_count"], rel(RUN_DIR / "summary.md"), rel(RUN_DIR / "raw.json"), rel(RUN_DIR / "completed.json"))
    append_docs(block, MARKER_RUN)
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [RUN_STATE, backup_request, Path(__file__).resolve(), PROTOCOL_JSON, PROTOCOL_MD, args.backup_proof]
    write_json(RUN_DIR / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created, "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "episodes": len(episodes), "control_steps": control_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "backup_request": rel(backup_request), "headline": {"positive_state_count": analysis["positive_state_count"], "negative_neutral_state_count": analysis["negative_neutral_state_count"], "positive_cases": analysis["positive_cases"], "training_refit_label_gate_pass_development_only": analysis["training_refit_label_gate_pass_development_only"], "blocking_artifact_count": analysis["blocking_artifact_count"], "next_action": "if gate passed freeze selector/refit smoke; otherwise pivot to next evidence-ranked scenario/reward/terminal/modeling diagnostic"}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "positive_state_count": analysis["positive_state_count"], "positive_cases": analysis["positive_cases"], "training_refit_label_gate_pass_development_only": analysis["training_refit_label_gate_pass_development_only"], "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(backup_request)}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup-proof", required=True, type=Path, help="verified external backup proof")
    ap.add_argument("--dry-run", action="store_true", help="no-simulation runner/protocol/runtime/terminal diagnostic")
    ap.add_argument("--run-rollout", action="store_true", help="run 140 matched-continuation episodes after post-dryrun backup")
    ap.add_argument("--i-accept-v1b-development-diagnostic", action="store_true", help="acknowledge development-only, no validation64/test access")
    args = ap.parse_args(argv)
    if not args.i_accept_v1b_development_diagnostic:
        raise ContractError("explicit --i-accept-v1b-development-diagnostic is required")
    if args.dry_run == args.run_rollout:
        raise ContractError("choose exactly one of --dry-run or --run-rollout")
    if args.dry_run:
        return run_dry_run(args)
    return run_rollout(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = DRYRUN_DIR if "--dry-run" in sys.argv else RUN_DIR
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
            "new_rollouts": 0 if "--dry-run" in sys.argv else None,
            "new_control_steps": 0 if "--dry-run" in sys.argv else None,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "next_recovery_hint": "Preserve partial output. If dry-run failed, repair runner/protocol checks only; if rollout failed, audit partial episodes before any rerun.",
        })
        raise
