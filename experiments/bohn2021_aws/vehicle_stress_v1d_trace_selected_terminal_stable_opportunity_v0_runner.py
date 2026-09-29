#!/usr/bin/env python3
"""Vehicle stress-v1d trace-selected terminal-stable opportunity runner v0.

Frozen protocol:
  research_artifacts/aws_protocols/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_frozen_20260929T0210Z.json

Purpose: development-only diagnostic to distinguish whether the v1c zero-label
result is a true absence of state-dependent horizon opportunity or an artifact
of the fixed-fraction branch-state selector.  v1d reuses the already frozen
fresh v1c case bank, runs H15-only traces to choose branch states by realised
H15 trace difficulty before any non-H15 branch rollout, then compares candidate
branch horizons from common H15-prefix states using terminal-stable labels only
(zero terminal and H15-common terminal).

Access contract: no historical validation64 bank, no validation64 bank, no
sealed test, no selector/refit, no gradient training.  Smoke rollouts require a
verified external backup after this source and dry-run.
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

import vehicle_stress_v1c_terminal_stable_label_density_probe_v0_runner as v1c  # noqa:E402

TASK = "vehicle"
STAMP = "20260929T0210Z"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_frozen_20260929T0210Z.json"
SOURCE = Path(__file__).resolve()

DRYRUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_dryrun_{STAMP}"
SMOKE_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_{STAMP}"
STATE_DRYRUN = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_dryrun_{STAMP}.md"
STATE_SMOKE = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP_BEFORE_SMOKE = BACKUP_DIR / f"REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1D_TRACE_SELECTED_TERMINAL_STABLE_OPPORTUNITY_V0_SMOKE_{STAMP}.json"
MARKER_DRYRUN = f"vehicle-stress-v1d-trace-selected-terminal-stable-opportunity-v0-dryrun-{STAMP}"
MARKER_SMOKE = f"vehicle-stress-v1d-trace-selected-terminal-stable-opportunity-v0-smoke-{STAMP}"

V1C_REANALYSIS_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_smoke_reanalysis_v0_20260929T0205Z/completed.json"
V1C_REANALYSIS_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_smoke_reanalysis_v0_20260929T0205Z/raw.json"
V1C_REANALYSIS_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_smoke_reanalysis_v0_20260929T0205Z/summary.md"
V1C_SMOKE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_smoke_20260929T0055Z_hash_repair/completed.json"
V1C_BANK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_bank_20260929T0055Z_hash_repair/completed.json"
V1C_BANK_ACTUAL = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_bank_20260929T0055Z_hash_repair/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_bank.json"

PREFIX_H = 15
BRANCH_HORIZONS = [10, 15, 25, 30, 35, 45, 50]
TERMINAL_MODES = ["zero_terminal", "h15_common_terminal"]
MAX_STEPS = 150
ORDER_SEED = 2609292113
MATERIAL_GAIN = 3.0
STATE_DISTANCE_TOL = 1e-5
PHYSICAL_PREFIX_TOL = 1e-9
SCAN_EPISODES = 20
TARGETS_EXACT = 12
BRANCH_EPISODES = TARGETS_EXACT * len(BRANCH_HORIZONS) * len(TERMINAL_MODES)
SMOKE_EPISODES_EXACT_UPPER = SCAN_EPISODES + BRANCH_EPISODES
SMOKE_CONTROL_STEP_UPPER = SMOKE_EPISODES_EXACT_UPPER * MAX_STEPS
HIGH_GROUPS = {"fresh_high_heading_long_or_medium", "fresh_high_heading_short"}
CONTROL_GROUPS = {"fresh_low_heading_low_clearance_control", "fresh_lower_stress_control"}


class ContractError(RuntimeError):
    """Protocol/access/budget violation."""


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


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
    return v1c.sha256(path)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def parse_time(value: Any) -> Optional[dt.datetime]:
    return v1c.parse_time(value)


def completed_passed(path: Path, check_hashes: bool = False) -> Mapping[str, Any]:
    return v1c.completed_passed(path, check_hashes=check_hashes)


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if marker not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def source_mtime_utc() -> dt.datetime:
    return dt.datetime.fromtimestamp(SOURCE.stat().st_mtime, dt.timezone.utc)


def verify_protocol_and_inputs() -> Dict[str, Any]:
    required = [PROTOCOL_JSON, V1C_REANALYSIS_DONE, V1C_REANALYSIS_RAW, V1C_REANALYSIS_SUMMARY, V1C_SMOKE_DONE, V1C_BANK_DONE]
    for p in required:
        if not p.exists():
            raise ContractError(f"required input missing: {rel(p)}")
    protocol = read_json(PROTOCOL_JSON)
    if protocol.get("protocol_id") != "vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_frozen_20260929T0210Z":
        raise ContractError("unexpected v1d protocol id")
    if protocol.get("classification") != "development_IMPROVED_trace_selected_terminal_stable_opportunity_probe_not_validation_not_final_test":
        raise ContractError("unexpected v1d protocol classification")
    access = protocol.get("access_rules") or {}
    for key in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if access.get(key) is not False:
            raise ContractError(f"protocol access flag must be false: {key}")
    if access.get("requires_verified_external_backup_after_this_protocol_and_runner_before_any_simulation") is not True:
        raise ContractError("protocol must require post-runner backup before simulation")
    design = protocol.get("diagnostic_design") or {}
    phase_a = design.get("phase_A_h15_trace_scan") or {}
    phase_b = design.get("phase_B_common_prefix_branch_rollout") or {}
    if int(phase_a.get("episodes_exact", -1)) != SCAN_EPISODES:
        raise ContractError("v1d phase-A episode budget changed")
    if int(phase_b.get("branch_targets_exact", -1)) != TARGETS_EXACT or int(phase_b.get("branch_episodes_exact", -1)) != BRANCH_EPISODES:
        raise ContractError("v1d phase-B branch budget changed")
    if [int(x) for x in phase_b.get("branch_horizons", [])] != BRANCH_HORIZONS:
        raise ContractError("v1d branch horizon grid changed")
    if [str(x) for x in phase_b.get("post_branch_terminal_modes", [])] != TERMINAL_MODES:
        raise ContractError("v1d terminal modes changed")
    if int(phase_b.get("new_training_episodes", -1)) != 0 or int(phase_b.get("new_gradient_steps", -1)) != 0 or int(phase_b.get("new_refit_steps", -1)) != 0:
        raise ContractError("v1d unexpectedly permits training/refit")
    label = protocol.get("primary_label_rule") or {}
    if float(label.get("material_gain_threshold_physical", -1)) != MATERIAL_GAIN:
        raise ContractError("v1d material threshold changed")
    if float(label.get("common_prefix_state_distance_tolerance", -1)) != STATE_DISTANCE_TOL:
        raise ContractError("v1d state-distance tolerance changed")
    re_done = completed_passed(V1C_REANALYSIS_DONE, check_hashes=False)
    if re_done.get("hard_pass") is not True:
        raise ContractError("v1c reanalysis did not hard-pass")
    headline = re_done.get("headline") or {}
    if headline.get("smoke_pass_to_full_repaired") is not False or headline.get("train_or_refit_now") is not False:
        raise ContractError("v1c reanalysis headline no longer justifies v1d diagnostic")
    smoke_done = completed_passed(V1C_SMOKE_DONE, check_hashes=False)
    bank_done = completed_passed(V1C_BANK_DONE, check_hashes=False)
    case_source = protocol.get("case_source") or {}
    protocol_bank_path = case_source.get("primary_case_bank")
    candidates: List[Path] = []
    if protocol_bank_path:
        candidates.append(ROOT / str(protocol_bank_path))
    candidates.append(V1C_BANK_ACTUAL)
    bank_path = next((p for p in candidates if p.exists()), None)
    if bank_path is None:
        raise ContractError("no v1c bank path exists among protocol/actual candidates")
    path_notice = None
    if protocol_bank_path and (ROOT / str(protocol_bank_path)).resolve() != bank_path.resolve():
        path_notice = {
            "type": "protocol_bank_filename_alias_repair_no_simulation",
            "protocol_primary_case_bank": protocol_bank_path,
            "actual_verified_bank": rel(bank_path),
            "reason": "The frozen protocol text points to the v0 bank filename inside the v0c bank directory; the verified v0c bank completed marker identifies the existing v0c bank file.  Cases/selection are unchanged.",
        }
    return {
        "protocol": protocol,
        "protocol_sha256": sha256(PROTOCOL_JSON),
        "v1c_reanalysis_completed": re_done,
        "v1c_reanalysis_completed_sha256": sha256(V1C_REANALYSIS_DONE),
        "v1c_smoke_completed": smoke_done,
        "v1c_smoke_completed_sha256": sha256(V1C_SMOKE_DONE),
        "v1c_bank_completed": bank_done,
        "v1c_bank_completed_sha256": sha256(V1C_BANK_DONE),
        "v1c_bank_path": bank_path,
        "v1c_bank_sha256_if_small_marker": bank_done.get("hashes", {}).get(rel(bank_path)),
        "path_notice": path_notice,
    }


def verify_backup_proof(path: Path, min_time: dt.datetime, purpose: str) -> Mapping[str, Any]:
    return v1c.verify_backup_proof(path, min_time, purpose)


def finite_summary(vals: Iterable[float]) -> Dict[str, Any]:
    return v1c.finite_summary(vals)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def state_tuple(obj: Any) -> Tuple[float, float, float]:
    return v1c.state_tuple(obj)


def state_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return v1c.state_distance(a, b)


def no_regression(candidate: Mapping[str, Any], ref: Mapping[str, Any]) -> bool:
    return v1c.no_regression(candidate, ref)


def finalize_episode_completed(ep_path: Path) -> None:
    files = [p for p in ep_path.iterdir() if p.is_file() and p.name != "completed.json"]
    write_json(ep_path / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})


def physical_prefix_rows(trace: Sequence[Mapping[str, Any]], branch_step: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in list(trace)[: max(0, min(branch_step, len(trace)))]:
        rows.append({
            "step": int(row.get("step", -1)),
            "horizon": int(row.get("horizon", -1)),
            "previous_state": row.get("previous_state"),
            "state": row.get("state"),
            "input": row.get("input"),
            "performance": safe_float(row.get("performance"), 0.0),
            "constraint": safe_float(row.get("constraint"), 0.0),
            "reward": safe_float(row.get("reward"), 0.0),
            "solver_success": bool(row.get("solver_success")),
        })
    return rows


def annotate_physical_prefix(summary: Dict[str, Any]) -> Dict[str, Any]:
    ep_path = ROOT / str(summary["path"])
    trace = read_json(ep_path / "trace.json")
    branch_step = int(summary.get("branch_step", 0))
    prefix = physical_prefix_rows(trace, branch_step)
    summary["physical_prefix_equivalence_sha256"] = canonical_sha(prefix)
    summary["physical_prefix_length"] = len(prefix)
    summary["physical_prefix_last_state"] = prefix[-1]["state"] if prefix else None
    if summary.get("terminal_mode") == "h15_common_terminal":
        summary["terminal_receipt_effective"] = summary.get("terminal_receipt_effective") if isinstance(summary.get("terminal_receipt_effective"), Mapping) and summary.get("terminal_receipt_effective", {}).get("weights_hash") else None
    write_json(ep_path / "summary.json", summary)
    finalize_episode_completed(ep_path)
    return summary


def patch_terminal_mode(base_smoke: Any) -> None:
    if getattr(base_smoke, "_v1d_h15_common_patched", False):
        return
    original_terminal_weights_for = base_smoke.terminal_weights_for

    def terminal_weights_for_v1d(mode: str, h: int, terminals: Mapping[int, Tuple[Any, Any]]) -> Tuple[Any, Any, str]:
        if mode == "h15_common_terminal":
            return terminals[PREFIX_H][0], terminals[PREFIX_H][1], "H15_common"
        return original_terminal_weights_for(mode, h, terminals)

    base_smoke.terminal_weights_for = terminal_weights_for_v1d
    base_smoke._v1d_h15_common_patched = True


def import_legacy_modules() -> Tuple[Any, Any, Any]:
    base_smoke, stage1_runner, fixed_base = v1c.import_legacy_modules()
    patch_terminal_mode(base_smoke)
    return base_smoke, stage1_runner, fixed_base


def update_h15_common_receipt(summary: Dict[str, Any], terminal_receipts: Mapping[str, Any]) -> Dict[str, Any]:
    if summary.get("terminal_mode") == "h15_common_terminal":
        summary["terminal_receipt_effective"] = terminal_receipts.get("15")
        summary["effective_branch_terminal_label"] = "H15_common"
    return summary


def load_v1c_bank(bank_path: Path) -> Dict[str, Any]:
    bank = read_json(bank_path)
    if bank.get("task") != TASK:
        raise ContractError("v1c bank task mismatch")
    if len(bank.get("selected_cases_full") or []) != SCAN_EPISODES:
        raise ContractError("v1c bank does not contain 20 selected full cases")
    sel = bank.get("selection") or {}
    if len(sel.get("selected_full_metadata") or []) != SCAN_EPISODES:
        raise ContractError("v1c bank selection metadata length mismatch")
    return bank


def trace_value(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    return safe_float(row.get(key), default)


def observation_component(row: Mapping[str, Any], idx: int, default: float = 0.0) -> float:
    obs = row.get("observation") or []
    try:
        return safe_float(obs[idx], default)
    except Exception:
        return default


def input_omega(row: Mapping[str, Any]) -> float:
    inp = row.get("input") or {}
    val = inp.get("u_omega") if isinstance(inp, Mapping) else None
    if isinstance(val, list) and val:
        return safe_float(val[0], 0.0)
    return safe_float(val, 0.0)


def solver_iterations(row: Mapping[str, Any]) -> float:
    attempts = ((row.get("recovery") or {}).get("attempts") or [])
    if not attempts:
        return 0.0
    return safe_float(attempts[-1].get("iterations"), 0.0)


def near_obstacle_flags(trace: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
    # The current logged trace has a constraint-cost proxy but not a direct
    # obstacle-distance series.  Use only positive constraint variation if it is
    # present; otherwise do not manufacture a near-obstacle signal.
    vals = [max(0.0, safe_float(r.get("constraint"), 0.0)) for r in trace]
    positives = [v for v in vals if v > 0.0]
    if not positives:
        return {int(r.get("step", i)): 0 for i, r in enumerate(trace)}
    threshold = sorted(positives)[max(0, int(math.floor(0.75 * (len(positives) - 1))))]
    return {int(r.get("step", i)): int(vals[i] >= threshold and vals[i] > 0.0) for i, r in enumerate(trace)}


def score_row(row: Mapping[str, Any], near_flag: int) -> Tuple[float, Dict[str, float]]:
    performance = max(0.0, trace_value(row, "performance", 0.0))
    heading_proxy = abs(observation_component(row, 2, 0.0))
    turn_effort = abs(input_omega(row))
    iters = solver_iterations(row)
    score = 10.0 * math.sqrt(performance) + 2.0 * heading_proxy + 0.5 * turn_effort + 0.02 * iters + 1.0 * float(near_flag)
    return float(score), {
        "performance_step_cost": float(performance),
        "sqrt_performance": float(math.sqrt(performance)),
        "abs_heading_proxy": float(heading_proxy),
        "abs_turn_effort_u_omega": float(turn_effort),
        "solver_iteration_proxy": float(iters),
        "near_obstacle_indicator": float(near_flag),
    }


def best_candidate_in_window(trace: Sequence[Mapping[str, Any]], full_case_index: int, meta: Mapping[str, Any], window_name: str, start: int, stop: int, near_flags: Mapping[int, int], scan_path: str) -> Optional[Dict[str, Any]]:
    upper = min(stop, len(trace) - 3, 90)
    if upper < start:
        return None
    best: Optional[Tuple[float, float, int, Dict[str, Any]]] = None
    for step in range(start, upper + 1):
        row = trace[step]
        near = int(near_flags.get(step, 0))
        score, comp = score_row(row, near)
        perf = comp["performance_step_cost"]
        item = {
            "full_case_index": int(full_case_index),
            "source_candidate_index": int(meta.get("candidate_index", -1)),
            "selection_group": meta.get("selection_group") or meta.get("stratum"),
            "branch_step": int(step),
            "window": window_name,
            "score": score,
            "score_components": comp,
            "tie_break_performance_step_cost": perf,
            "scan_trace_path": scan_path,
            "branch_previous_state_from_scan": row.get("previous_state"),
            "scan_step_horizon": int(row.get("horizon", -1)),
        }
        key = (score, perf, -step)
        if best is None or key > best[:3]:
            best = (score, perf, -step, item)
    return None if best is None else best[3]


def score_scan_trace(summary: Mapping[str, Any], meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
    ep_path = ROOT / str(summary["path"])
    trace = read_json(ep_path / "trace.json")
    near_flags = near_obstacle_flags(trace)
    candidates: List[Dict[str, Any]] = []
    for name, start, stop in (("early", 8, 32), ("mid", 33, 70)):
        cand = best_candidate_in_window(trace, int(summary["case"]), meta, name, start, stop, near_flags, str(summary["path"]))
        if cand is not None:
            candidates.append(cand)
    if not candidates and len(trace) > 10:
        step = max(8, min(len(trace) - 3, 90))
        row = trace[step]
        score, comp = score_row(row, int(near_flags.get(step, 0)))
        candidates.append({
            "full_case_index": int(summary["case"]),
            "source_candidate_index": int(meta.get("candidate_index", -1)),
            "selection_group": meta.get("selection_group") or meta.get("stratum"),
            "branch_step": int(step),
            "window": "fallback_single_eligible",
            "score": score,
            "score_components": comp,
            "tie_break_performance_step_cost": comp["performance_step_cost"],
            "scan_trace_path": str(summary["path"]),
            "branch_previous_state_from_scan": row.get("previous_state"),
            "scan_step_horizon": int(row.get("horizon", -1)),
            "selection_fallback": "trace_too_short_for_declared_windows",
        })
    return candidates


def select_with_case_cap(candidates: Sequence[Mapping[str, Any]], k: int, max_per_case: int, high: bool) -> Tuple[List[Dict[str, Any]], List[str]]:
    selected: List[Dict[str, Any]] = []
    counts: Dict[int, int] = {}
    fallbacks: List[str] = []
    if high:
        # First ensure at least 4 cases when available, then fill by score.
        by_case: Dict[int, List[Mapping[str, Any]]] = {}
        for c in candidates:
            by_case.setdefault(int(c["full_case_index"]), []).append(c)
        case_best = []
        for case, rows in by_case.items():
            best = max(rows, key=lambda r: (float(r["score"]), float(r.get("tie_break_performance_step_cost", 0.0)), -int(r["branch_step"])))
            case_best.append(best)
        for row in sorted(case_best, key=lambda r: (-float(r["score"]), int(r["full_case_index"]), int(r["branch_step"]))):
            if len(selected) >= min(4, len(by_case), k):
                break
            selected.append(copy.deepcopy(dict(row)))
            counts[int(row["full_case_index"])] = counts.get(int(row["full_case_index"]), 0) + 1
    if not high:
        scores = sorted(float(c["score"]) for c in candidates)
        med = scores[len(scores) // 2] if scores else 0.0
        order = sorted(candidates, key=lambda r: (abs(float(r["score"]) - med), counts.get(int(r["full_case_index"]), 0), int(r["full_case_index"]), int(r["branch_step"])))
    else:
        order = sorted(candidates, key=lambda r: (-float(r["score"]), int(r["full_case_index"]), int(r["branch_step"])))
    existing = {(int(r["full_case_index"]), int(r["branch_step"])) for r in selected}
    for row in order:
        if len(selected) >= k:
            break
        case = int(row["full_case_index"])
        key = (case, int(row["branch_step"]))
        if key in existing:
            continue
        if counts.get(case, 0) >= max_per_case:
            continue
        selected.append(copy.deepcopy(dict(row)))
        existing.add(key)
        counts[case] = counts.get(case, 0) + 1
    if len(selected) < k:
        fallbacks.append("relaxed_case_cap_to_meet_target_count")
        for row in order:
            if len(selected) >= k:
                break
            key = (int(row["full_case_index"]), int(row["branch_step"]))
            if key in existing:
                continue
            selected.append(copy.deepcopy(dict(row)))
            existing.add(key)
    if len(selected) != k:
        raise ContractError(f"could not select {k} targets from {len(candidates)} candidates")
    return selected, fallbacks


def select_targets(scan_summaries: Sequence[Mapping[str, Any]], selection_meta: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    all_candidates: List[Dict[str, Any]] = []
    for summary in scan_summaries:
        meta = selection_meta[int(summary["case"])]
        all_candidates.extend(score_scan_trace(summary, meta))
    high_candidates = [c for c in all_candidates if str(c.get("selection_group")) in HIGH_GROUPS]
    control_candidates = [c for c in all_candidates if str(c.get("selection_group")) in CONTROL_GROUPS]
    high, fb_high = select_with_case_cap(high_candidates, 8, 2, high=True)
    control, fb_control = select_with_case_cap(control_candidates, 4, 2, high=False)
    targets = high + control
    if len(targets) != TARGETS_EXACT:
        raise ContractError("target count mismatch")
    seen = set()
    for i, t in enumerate(targets):
        group_kind = "high_trace_selected" if i < len(high) else "control_trace_selected"
        t["target_index"] = int(i)
        t["target_role"] = group_kind
        t["state_id"] = "target%02d_case%02d_cand%03d_b%03d_%s" % (i, int(t["full_case_index"]), int(t["source_candidate_index"]), int(t["branch_step"]), str(t.get("window", "w")))
        key = (int(t["full_case_index"]), int(t["branch_step"]), str(t["state_id"]))
        if key in seen:
            raise ContractError("duplicate selected target")
        seen.add(key)
    cases_high = sorted(set(int(t["full_case_index"]) for t in high))
    if len(cases_high) < 4 and len(set(int(c["full_case_index"]) for c in high_candidates)) >= 4:
        fb_high.append("high_target_at_least_4_cases_not_met_despite_available_cases")
    return {
        "all_trace_candidates": all_candidates,
        "selected_targets": targets,
        "selected_high_case_count": len(cases_high),
        "selected_control_case_count": len(set(int(t["full_case_index"]) for t in control)),
        "selection_fallbacks": fb_high + fb_control,
        "selection_rule": "8 highest-score high-heading trace candidates with <=2 targets/case and >=4 cases when possible; 4 control targets by median H15 score with <=2/case; all selected before non-H15 branch rollout",
    }


def build_branch_schedule(targets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_index = 0
    for target in targets:
        for mode in TERMINAL_MODES:
            for h in BRANCH_HORIZONS:
                rows.append({
                    "schedule_base_index": int(base_index),
                    "state_id": str(target["state_id"]),
                    "case": int(target["full_case_index"]),
                    "source_candidate_index": int(target["source_candidate_index"]),
                    "selection_group": str(target.get("selection_group")),
                    "role": str(target.get("target_role")),
                    "target_index": int(target["target_index"]),
                    "branch_step": int(target["branch_step"]),
                    "horizon": int(h),
                    "terminal_mode": str(mode),
                    "prefix_horizon": PREFIX_H,
                    "run_kind": "smoke",
                    "score": float(target["score"]),
                    "score_components": target.get("score_components"),
                    "scan_trace_path": target.get("scan_trace_path"),
                })
                base_index += 1
    rng = random.Random(ORDER_SEED)
    order = list(range(len(rows)))
    rng.shuffle(order)
    return [dict(rows[i], execution_index=int(j + 1000)) for j, i in enumerate(order)]


def analyze_branch_episodes(branch_episodes: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_target_mode: Dict[Tuple[str, str], Dict[int, Mapping[str, Any]]] = {}
    for e in branch_episodes:
        by_target_mode.setdefault((str(e["state_id"]), str(e["terminal_mode"])), {})[int(e["branch_horizon"])] = e
    target_lookup = {str(t["state_id"]): t for t in targets}
    artifact_flags = {
        "missing_H15_reference": 0,
        "missing_horizon_or_terminal_mode": 0,
        "branch_not_reached": 0,
        "state_distance_gt_tol": 0,
        "physical_prefix_mismatch": 0,
        "safety_solver_regression": 0,
        "positive_with_common_terminal_non_success": 0,
    }
    state_rows: List[Dict[str, Any]] = []
    robust_positive_cases: List[int] = []
    robust_positive_horizon_counts: Dict[str, int] = {}
    control_positive_states = 0
    control_state_count = 0
    timing_by_h_mode: Dict[str, List[float]] = {}
    for target in targets:
        sid = str(target["state_id"])
        group = str(target.get("selection_group"))
        case_id = int(target["full_case_index"])
        is_control = group in CONTROL_GROUPS or str(target.get("target_role")) == "control_trace_selected"
        if is_control:
            control_state_count += 1
        mode_results: Dict[str, Any] = {}
        robust_sets: List[set] = []
        all_comparisons: List[Dict[str, Any]] = []
        for mode in TERMINAL_MODES:
            by_h = by_target_mode.get((sid, mode), {})
            ref = by_h.get(PREFIX_H)
            if ref is None:
                artifact_flags["missing_H15_reference"] += 1
                robust_sets.append(set())
                mode_results[mode] = {"missing_H15_reference": True, "material_horizons": [], "comparisons": []}
                continue
            if not bool(ref.get("branch_reached")):
                artifact_flags["branch_not_reached"] += 1
            ref_state = state_tuple(ref.get("branch_previous_state"))
            ref_prefix_hash = ref.get("physical_prefix_equivalence_sha256")
            material_h: List[int] = []
            comps: List[Dict[str, Any]] = []
            for h in BRANCH_HORIZONS:
                e = by_h.get(h)
                if e is None:
                    artifact_flags["missing_horizon_or_terminal_mode"] += 1
                    continue
                timing_by_h_mode.setdefault(f"H{h}_{mode}", []).append(safe_float((e.get("decision_timing_s") or {}).get("sum"), 0.0))
                if not bool(e.get("branch_reached")):
                    artifact_flags["branch_not_reached"] += 1
                cand_state = state_tuple(e.get("branch_previous_state"))
                dist = state_distance(cand_state, ref_state)
                prefix_hash = e.get("physical_prefix_equivalence_sha256")
                prefix_match = bool(prefix_hash == ref_prefix_hash)
                if h != PREFIX_H and dist > STATE_DISTANCE_TOL:
                    artifact_flags["state_distance_gt_tol"] += 1
                if h != PREFIX_H and not prefix_match:
                    artifact_flags["physical_prefix_mismatch"] += 1
                phys_gain = float(ref["continuation_physical_constraint_cost_from_branch"] - e["continuation_physical_constraint_cost_from_branch"])
                total_gain = float(ref["continuation_total_cost_from_branch"] - e["continuation_total_cost_from_branch"])
                safety_ok = no_regression(e, ref)
                if h != PREFIX_H and not safety_ok:
                    artifact_flags["safety_solver_regression"] += 1
                material = bool(h != PREFIX_H and bool(e.get("branch_reached")) and bool(ref.get("branch_reached")) and dist <= STATE_DISTANCE_TOL and prefix_match and safety_ok and phys_gain >= MATERIAL_GAIN)
                if material:
                    material_h.append(h)
                bdiag = e.get("branch_step_mpc_diag") or {}
                comp = {
                    "state_id": sid,
                    "case": case_id,
                    "selection_group": group,
                    "target_role": target.get("target_role"),
                    "terminal_mode": mode,
                    "horizon": int(h),
                    "success": bool(e.get("success")),
                    "constraint": bool(e.get("constraint")),
                    "steps": int(e.get("steps", 0)),
                    "initial_failed_steps": int(e.get("initial_failed_steps", 0)),
                    "final_failed_steps": int(e.get("final_failed_steps", 0)),
                    "solver_failure_steps": int(e.get("solver_failure_steps", 0)),
                    "branch_reached": bool(e.get("branch_reached")),
                    "continuation_physical": safe_float(e.get("continuation_physical_constraint_cost_from_branch"), 0.0),
                    "continuation_total": safe_float(e.get("continuation_total_cost_from_branch"), 0.0),
                    "gain_vs_H15_physical": phys_gain,
                    "gain_vs_H15_total": total_gain,
                    "state_distance_vs_H15": dist,
                    "physical_prefix_matches_H15": prefix_match,
                    "no_success_constraint_solver_regression_vs_H15": safety_ok,
                    "material_positive_terminal_mode": material,
                    "branch_mpc_value_fn": bdiag.get("info_mpc_value_fn"),
                    "branch_objective_opt_f_num": bdiag.get("objective_opt_f_num"),
                    "decision_sum_s": safe_float((e.get("decision_timing_s") or {}).get("sum"), 0.0),
                    "solver_attempt_sum_s": safe_float((e.get("solver_attempt_timing_s") or {}).get("sum"), 0.0),
                    "path": e.get("path"),
                }
                comps.append(comp)
                all_comparisons.append(comp)
            mode_results[mode] = {
                "reference_H15": {
                    "success": bool(ref.get("success")),
                    "constraint": bool(ref.get("constraint")),
                    "continuation_physical": safe_float(ref.get("continuation_physical_constraint_cost_from_branch"), 0.0),
                    "continuation_total": safe_float(ref.get("continuation_total_cost_from_branch"), 0.0),
                    "steps": int(ref.get("steps", 0)),
                    "path": ref.get("path"),
                },
                "material_horizons": sorted(material_h),
                "comparisons": comps,
            }
            robust_sets.append(set(material_h))
        robust_h = sorted(set.intersection(*robust_sets) if robust_sets else set())
        robust_positive = bool(robust_h)
        if robust_positive:
            robust_positive_cases.append(case_id)
            if is_control:
                control_positive_states += 1
            for h in robust_h:
                robust_positive_horizon_counts[str(h)] = robust_positive_horizon_counts.get(str(h), 0) + 1
            for comp in all_comparisons:
                if comp["horizon"] in robust_h and comp["terminal_mode"] == "h15_common_terminal" and not comp["success"]:
                    artifact_flags["positive_with_common_terminal_non_success"] += 1
        best_by_phys = min(all_comparisons, key=lambda r: (r["continuation_physical"], r["decision_sum_s"], r["horizon"])) if all_comparisons else None
        state_rows.append({
            "state_id": sid,
            "case": case_id,
            "source_candidate_index": int(target.get("source_candidate_index", -1)),
            "selection_group": group,
            "target_role": target.get("target_role"),
            "branch_step": int(target["branch_step"]),
            "score": float(target["score"]),
            "score_components": target.get("score_components"),
            "is_control_state": bool(is_control),
            "robust_positive_state": robust_positive,
            "robust_positive_horizons": robust_h,
            "label": "robust_positive_non_H15" if robust_positive else "negative_or_neutral",
            "best_by_physical_any_mode": best_by_phys,
            "mode_results": mode_results,
        })
    positive_count = sum(1 for r in state_rows if r["robust_positive_state"])
    negative_count = sum(1 for r in state_rows if not r["robust_positive_state"])
    distinct_cases = sorted(set(robust_positive_cases))
    blocking_artifacts = int(artifact_flags["missing_H15_reference"] + artifact_flags["missing_horizon_or_terminal_mode"] + artifact_flags["branch_not_reached"] + artifact_flags["state_distance_gt_tol"] + artifact_flags["physical_prefix_mismatch"] + artifact_flags["positive_with_common_terminal_non_success"])
    control_fpr = float(control_positive_states) / float(control_state_count) if control_state_count else 0.0
    smoke_gate = bool(positive_count >= 2 and len(distinct_cases) >= 2 and negative_count >= 4 and control_positive_states <= 1 and blocking_artifacts == 0)
    return {
        "state_count": len(state_rows),
        "robust_positive_state_count": int(positive_count),
        "negative_or_neutral_state_count": int(negative_count),
        "robust_positive_cases": distinct_cases,
        "robust_positive_horizon_counts": robust_positive_horizon_counts,
        "positive_groups": sorted(set(str(r["selection_group"]) for r in state_rows if r["robust_positive_state"])),
        "control_state_count": int(control_state_count),
        "control_positive_state_count": int(control_positive_states),
        "control_false_positive_rate": control_fpr,
        "artifact_flags": artifact_flags,
        "blocking_artifact_count": blocking_artifacts,
        "smoke_pass_to_next_label_or_refit_design": smoke_gate,
        "timing_decision_sum_by_horizon_mode_s": {k: finite_summary(v) for k, v in sorted(timing_by_h_mode.items())},
        "state_rows": state_rows,
        "decision": {
            "train_or_refit_now": False,
            "if_smoke_fails": "do not train/refit; preserve negative evidence and prioritize terminal/modeling or documented sparse-opportunity diagnosis",
            "if_smoke_passes": "development evidence only; after backup freeze fuller trace-selected label-density or compact selector/value-refit smoke with fresh confirmation",
        },
    }


def write_dryrun_summary(raw: Mapping[str, Any]) -> None:
    notice = raw.get("path_notice")
    lines = [
        "# Vehicle stress-v1d trace-selected terminal-stable opportunity dry-run",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations, no candidate resets, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Verified inputs",
        "",
        f"- Protocol: `{raw['protocol']['json']}` sha256 `{raw['protocol']['json_sha256']}`.",
        f"- v1c repaired smoke reanalysis: `{raw['v1c_reanalysis']['completed']}` sha256 `{raw['v1c_reanalysis']['completed_sha256']}`.",
        f"- v1c bank completed: `{raw['v1c_bank']['completed']}` sha256 `{raw['v1c_bank']['completed_sha256']}`.",
    ]
    if notice:
        lines += ["", "## Non-scientific path notice", "", f"- `{notice}`"]
    lines += [
        "",
        "## Frozen next simulation budget (not yet run)",
        "",
        f"- Phase A H15 scan: `{SCAN_EPISODES}` episodes / `{SCAN_EPISODES * MAX_STEPS}` control-step cap.",
        f"- Phase B branch rollout: `{BRANCH_EPISODES}` episodes / `{BRANCH_EPISODES * MAX_STEPS}` control-step cap.",
        f"- Total smoke upper bound: `{SMOKE_EPISODES_EXACT_UPPER}` episodes / `{SMOKE_CONTROL_STEP_UPPER}` control-step cap.",
        f"- Branch horizons: `{BRANCH_HORIZONS}`; terminal modes: `{TERMINAL_MODES}`; prefix H `{PREFIX_H}`.",
        "",
        "## Evidence-ranked decision",
        "",
        "1. v1c fixed-fraction fresh states remained 0/12 robust-positive after physical-prefix repair; trace-selected fresh states are now the most informative discriminator.",
        "2. Terminal-source artifacts remain material, so v1d keeps per-H/H25 labels excluded and uses zero/H15-common realised continuation labels.",
        "3. No selector/refit is justified before a stable label-density gate passes.",
        "",
        f"Backup request before smoke: `{raw['backup_request_before_smoke']}`.",
    ]
    (DRYRUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_smoke_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    target_sel = raw["target_selection"]
    lines = [
        "# Vehicle stress-v1d trace-selected terminal-stable opportunity smoke",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only; no training/refit, no validation64 bank, no sealed test.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / cap `{raw['budget_declared']['episodes_upper_bound']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Headline",
        "",
        f"- Robust-positive states: `{a['robust_positive_state_count']}` across cases `{a['robust_positive_cases']}`.",
        f"- Negative/neutral states: `{a['negative_or_neutral_state_count']}`.",
        f"- Robust positive horizons: `{a['robust_positive_horizon_counts']}`.",
        f"- Control positive states: `{a['control_positive_state_count']}/{a['control_state_count']}`.",
        f"- Blocking artifacts: `{a['blocking_artifact_count']}`; flags `{a['artifact_flags']}`.",
        f"- Smoke pass to next label/refit design: `{a['smoke_pass_to_next_label_or_refit_design']}`.",
        f"- Target selection fallbacks: `{target_sel.get('selection_fallbacks')}`.",
        "",
        "## Selected targets",
        "",
        "| target | case | group | role | branch | score | components |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for t in target_sel["selected_targets"]:
        lines.append("| `%s` | %d | `%s` | `%s` | %d | %.6g | `%s` |" % (t["state_id"], int(t["full_case_index"]), t.get("selection_group"), t.get("target_role"), int(t["branch_step"]), float(t["score"]), t.get("score_components")))
    lines += ["", "## Per-state labels", "", "| state | case | group | role | branch | label | robust H | best phys mode/H/cost |", "|---|---:|---|---|---:|---|---|---|"]
    for row in a["state_rows"]:
        best = row.get("best_by_physical_any_mode") or {}
        lines.append("| `%s` | %d | `%s` | `%s` | %d | `%s` | `%s` | `%s`/H%s/%.6g |" % (row["state_id"], int(row["case"]), row.get("selection_group"), row.get("target_role"), int(row["branch_step"]), row.get("label"), row.get("robust_positive_horizons"), best.get("terminal_mode"), str(best.get("horizon")), float(best.get("continuation_physical", 0.0))))
    lines += [
        "",
        "## Four-axis evidence update",
        "",
        "### SCENARIOS",
        f"- verified: H15-trace-selected fresh states produced robust positives={a['robust_positive_state_count']} / {a['state_count']} across cases={a['robust_positive_cases']}.",
        "- hypothesis: if this remains sparse, adaptive opportunity in the current source-supported vehicle family is likely narrow or terminal/model dependent rather than broadly learnable.",
        "- missing: independent confirmation and fair fixed-H timing comparisons remain unavailable.",
        "- discriminator: obey the smoke gate; do not train/refit if it fails.",
        "",
        "### REWARD / TERMINAL",
        "- verified: labels use realised physical continuation under zero_terminal and H15-common terminal; per-H/H25 are excluded.",
        "- hypothesis: terminal-source-stable labels are safer but may suppress terminal-artifact positives.",
        "- missing: terminal-model/refit diagnostic if labels stay sparse.",
        "- discriminator: compare zero vs H15-common material horizons in raw state rows.",
        "",
        "### TRAINING",
        "- verified: new_training_episodes=0, new_gradient_steps=0, new_refit_steps=0.",
        "- hypothesis: selector training is downstream of label density and should not proceed from a failed smoke.",
        "- missing: compact IMPROVED selector/value-refit only if a stable-label gate passes.",
        "- discriminator: freeze refit after, not before, stable labels.",
        "",
        "### COMPARISONS",
        "- verified: no adaptive superiority claim; timing was recorded but not used for acceptance.",
        "- hypothesis: strong fixed-H/Pareto baselines may absorb any gains.",
        "- missing: fair fixed-H retuning and measured timing under any revised method.",
        "- discriminator: later paired validation only after method/scenario protocol is frozen.",
        "",
        "## Decision",
        f"- Train/refit now: `{a['decision']['train_or_refit_now']}`.",
        f"- Backup request: `{raw['backup_request_after_run']}`.",
    ]
    (SMOKE_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dry_run() -> int:
    if (DRYRUN_DIR / "completed.json").exists():
        completed_passed(DRYRUN_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(DRYRUN_DIR / "completed.json"), "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if DRYRUN_DIR.exists() and any(p.name != "run.lock" for p in DRYRUN_DIR.iterdir()):
        raise ContractError(f"partial dry-run output exists; inspect first: {rel(DRYRUN_DIR)}")
    inputs = verify_protocol_and_inputs()
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DRYRUN_DIR / "run_started.json", {"started_utc": created, "pid": os.getpid(), "method": "vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_dryrun", "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0})
    write_json(REQUEST_BACKUP_BEFORE_SMOKE, {
        "requested_utc": created,
        "reason": "backup v1d trace-selected terminal-stable opportunity runner/dry-run/protocol before any v1d smoke simulation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "planned_smoke_episodes_upper_bound": SMOKE_EPISODES_EXACT_UPPER,
        "planned_smoke_control_step_upper_bound": SMOKE_CONTROL_STEP_UPPER,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(SOURCE), rel(PROTOCOL_JSON), rel(V1C_REANALYSIS_DONE), rel(V1C_SMOKE_DONE), rel(V1C_BANK_DONE), rel(DRYRUN_DIR), rel(STATE_DRYRUN), rel(REQUEST_BACKUP_BEFORE_SMOKE)],
    })
    raw = {
        "created_utc": created,
        "method": "vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_dryrun_no_simulation",
        "classification": "development_protocol_readiness_no_simulation_trace_selected_terminal_stable_opportunity_probe",
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
        "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": inputs["protocol_sha256"], "full": inputs["protocol"]},
        "v1c_reanalysis": {"completed": rel(V1C_REANALYSIS_DONE), "completed_sha256": inputs["v1c_reanalysis_completed_sha256"], "raw": rel(V1C_REANALYSIS_RAW), "raw_sha256": sha256(V1C_REANALYSIS_RAW), "summary": rel(V1C_REANALYSIS_SUMMARY), "summary_sha256": sha256(V1C_REANALYSIS_SUMMARY), "headline": inputs["v1c_reanalysis_completed"].get("headline")},
        "v1c_smoke": {"completed": rel(V1C_SMOKE_DONE), "completed_sha256": inputs["v1c_smoke_completed_sha256"]},
        "v1c_bank": {"completed": rel(V1C_BANK_DONE), "completed_sha256": inputs["v1c_bank_completed_sha256"], "bank_path": rel(inputs["v1c_bank_path"]), "bank_sha256_from_completed_marker": inputs.get("v1c_bank_sha256_if_small_marker")},
        "path_notice": inputs.get("path_notice"),
        "planned_smoke": {"phase_A_scan_episodes": SCAN_EPISODES, "phase_B_branch_episodes": BRANCH_EPISODES, "episodes_upper_bound": SMOKE_EPISODES_EXACT_UPPER, "control_step_upper_bound": SMOKE_CONTROL_STEP_UPPER, "branch_horizons": BRANCH_HORIZONS, "terminal_modes": TERMINAL_MODES, "order_seed": ORDER_SEED},
        "source_hashes": {rel(SOURCE): sha256(SOURCE), rel(PROTOCOL_JSON): sha256(PROTOCOL_JSON), rel(V1C_REANALYSIS_DONE): sha256(V1C_REANALYSIS_DONE), rel(V1C_SMOKE_DONE): sha256(V1C_SMOKE_DONE), rel(V1C_BANK_DONE): sha256(V1C_BANK_DONE)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "backup_request_before_smoke": rel(REQUEST_BACKUP_BEFORE_SMOKE),
        "next": "after verified external backup covering this dry-run/source/protocol, run --run-smoke under legacy interpreter with explicit development acceptance",
    }
    write_json(DRYRUN_DIR / "raw.json", raw)
    write_dryrun_summary(raw)
    STATE_DRYRUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_DRYRUN.write_text(
        f"# Vehicle stress-v1d trace-selected terminal-stable opportunity dry-run\n\n"
        f"UTC: {created}. No simulations/control steps/candidate resets/training/refit/validation/test access. "
        f"Verified v1d protocol and v1c repaired-smoke inputs. Smoke remains blocked until external backup covers `{rel(REQUEST_BACKUP_BEFORE_SMOKE)}` plus source/protocol/dry-run outputs.\n",
        encoding="utf-8",
    )
    append_docs(f"""<!-- {MARKER_DRYRUN} -->
## 2026-09-29 vehicle stress-v1d trace-selected terminal-stable opportunity dry-run

UTC: {created}. No-simulation dry-run completed for the trace-selected terminal-stable opportunity probe. Verified frozen protocol `{rel(PROTOCOL_JSON)}`, v1c repaired smoke reanalysis `{rel(V1C_REANALYSIS_DONE)}`, and v1c bank marker `{rel(V1C_BANK_DONE)}`. No candidate resets, rollouts, training/refit, validation64-bank access or sealed-test access. Planned smoke upper bound is {SMOKE_EPISODES_EXACT_UPPER} episodes/{SMOKE_CONTROL_STEP_UPPER} control steps (20 H15 scan episodes plus 168 common-prefix branch episodes). Smoke is blocked until verified external backup covers the new runner/dry-run/protocol and request `{rel(REQUEST_BACKUP_BEFORE_SMOKE)}`.
""", MARKER_DRYRUN)
    files = [p for p in DRYRUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_DRYRUN, REQUEST_BACKUP_BEFORE_SMOKE, SOURCE, PROTOCOL_JSON, V1C_REANALYSIS_DONE, V1C_REANALYSIS_RAW, V1C_REANALYSIS_SUMMARY, V1C_SMOKE_DONE, V1C_BANK_DONE]
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
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "planned_smoke_episodes_upper_bound": SMOKE_EPISODES_EXACT_UPPER,
        "planned_smoke_control_step_upper_bound": SMOKE_CONTROL_STEP_UPPER,
        "backup_required_before_smoke": True,
        "backup_request": rel(REQUEST_BACKUP_BEFORE_SMOKE),
        "headline": {
            "train_or_refit_now": False,
            "smoke_blocked_until_backup": True,
            "next_action": "after verified external backup, run v1d smoke under legacy interpreter; if smoke gate fails, do not train/refit",
            "v1c_repaired_robust_positive_state_count": (inputs["v1c_reanalysis_completed"].get("headline") or {}).get("repaired_robust_positive_state_count"),
            "v1c_smoke_pass_to_full_repaired": (inputs["v1c_reanalysis_completed"].get("headline") or {}).get("smoke_pass_to_full_repaired"),
            "path_notice_present": inputs.get("path_notice") is not None,
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(DRYRUN_DIR / "completed.json"),
        "summary": rel(DRYRUN_DIR / "summary.md"),
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
        "planned_smoke_episodes_upper_bound": SMOKE_EPISODES_EXACT_UPPER,
        "planned_smoke_control_step_upper_bound": SMOKE_CONTROL_STEP_UPPER,
        "backup_request": rel(REQUEST_BACKUP_BEFORE_SMOKE),
    }, sort_keys=True), flush=True)
    return 0


def run_smoke(backup_proof: Path) -> int:
    if not (DRYRUN_DIR / "completed.json").exists():
        raise ContractError("smoke requires completed v1d dry-run marker")
    dry_done = completed_passed(DRYRUN_DIR / "completed.json", check_hashes=True)
    if (SMOKE_DIR / "completed.json").exists():
        completed_passed(SMOKE_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(SMOKE_DIR / "completed.json"), "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if SMOKE_DIR.exists() and any(p.name != "run.lock" for p in SMOKE_DIR.iterdir()):
        raise ContractError(f"partial smoke output exists; inspect first: {rel(SMOKE_DIR)}")
    inputs = verify_protocol_and_inputs()
    min_candidates = [source_mtime_utc(), parse_time(dry_done.get("created_utc")), parse_time(inputs["v1c_reanalysis_completed"].get("created_utc"))]
    min_time = max(t for t in min_candidates if t is not None)
    backup = verify_backup_proof(backup_proof, min_time, "vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_smoke")
    base_smoke, stage1_runner, _ = import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError(f"legacy runtime preflight failed: {preflight}")
    stage1_runner.base.v1.latency_verify()
    bank = load_v1c_bank(inputs["v1c_bank_path"])
    selected_cases_full = bank["selected_cases_full"]
    selection_meta = bank["selection"]["selected_full_metadata"]
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(SMOKE_DIR / "run_started.json", {"started_utc": started, "pid": os.getpid(), "method": "vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke", "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0})
    write_json(SMOKE_DIR / "runtime_preflight.json", preflight)
    write_json(SMOKE_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})
    base_smoke.OUT = SMOKE_DIR
    base_smoke.PREFIX_H = PREFIX_H
    base_smoke.MAX_STEPS = MAX_STEPS
    base_smoke.MATERIAL_GAIN = MATERIAL_GAIN
    scan_summaries: List[Dict[str, Any]] = []
    for case_idx in range(SCAN_EPISODES):
        meta = selection_meta[case_idx]
        item = {
            "execution_index": int(case_idx),
            "state_id": "scan_case%02d_cand%03d" % (case_idx, int(meta.get("candidate_index", -1))),
            "case": int(case_idx),
            "branch_step": 0,
            "horizon": PREFIX_H,
            "terminal_mode": "h15_common_terminal",
            "role": "phase_A_h15_trace_scan",
        }
        summary = base_smoke.run_one(item, selected_cases_full[case_idx], terminals, terminal_receipts)
        summary = update_h15_common_receipt(summary, terminal_receipts)
        summary["phase"] = "phase_A_h15_trace_scan"
        summary["selection_group"] = meta.get("selection_group") or meta.get("stratum")
        write_json(ROOT / summary["path"] / "summary.json", summary)
        finalize_episode_completed(ROOT / summary["path"])
        scan_summaries.append(summary)
        progress = {"pid": os.getpid(), "phase": "scan", "scan_done": len(scan_summaries), "scan_expected": SCAN_EPISODES, "control_steps_done": int(sum(int(e.get("steps", 0)) for e in scan_summaries)), "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "case", "steps", "success", "termination")}, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(SMOKE_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    target_selection = select_targets(scan_summaries, selection_meta)
    write_json(SMOKE_DIR / "selected_targets.json", target_selection)
    branch_schedule = build_branch_schedule(target_selection["selected_targets"])
    if len(branch_schedule) != BRANCH_EPISODES:
        raise ContractError("branch schedule length mismatch")
    write_json(SMOKE_DIR / "branch_schedule.json", {"episodes": branch_schedule, "order_seed": ORDER_SEED, "branch_horizons": BRANCH_HORIZONS, "terminal_modes": TERMINAL_MODES, "prefix_horizon": PREFIX_H, "max_steps": MAX_STEPS})
    branch_episodes: List[Dict[str, Any]] = []
    for item in branch_schedule:
        summary = base_smoke.run_one(item, selected_cases_full[int(item["case"])], terminals, terminal_receipts)
        summary = update_h15_common_receipt(summary, terminal_receipts)
        summary.update({
            "phase": "phase_B_common_prefix_branch_rollout",
            "source_candidate_index": int(item["source_candidate_index"]),
            "selection_group": item.get("selection_group"),
            "target_index": int(item["target_index"]),
            "target_role": item.get("role"),
            "score": float(item["score"]),
            "score_components": item.get("score_components"),
            "scan_trace_path": item.get("scan_trace_path"),
            "protocol_terminal_label_family": "terminal_stable_zero_or_H15_common_trace_selected_v1d",
        })
        summary = annotate_physical_prefix(summary)
        branch_episodes.append(summary)
        control_steps_done = int(sum(int(e.get("steps", 0)) for e in scan_summaries) + sum(int(e.get("steps", 0)) for e in branch_episodes))
        progress = {"pid": os.getpid(), "phase": "branch", "scan_done": len(scan_summaries), "branch_done": len(branch_episodes), "branch_expected": len(branch_schedule), "control_steps_done": control_steps_done, "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "case", "branch_step", "branch_horizon", "terminal_mode", "steps", "success", "termination", "branch_reached")}, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(SMOKE_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    total_episodes = len(scan_summaries) + len(branch_episodes)
    control_steps = int(sum(int(e.get("steps", 0)) for e in scan_summaries) + sum(int(e.get("steps", 0)) for e in branch_episodes))
    if total_episodes != SMOKE_EPISODES_EXACT_UPPER or control_steps > SMOKE_CONTROL_STEP_UPPER:
        raise ContractError("v1d smoke budget violation")
    analysis = analyze_branch_episodes(branch_episodes, target_selection["selected_targets"])
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1D_TRACE_SELECTED_TERMINAL_STABLE_OPPORTUNITY_V0_SMOKE_%s.json" % created.replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created, "reason": "backup v1d trace-selected terminal-stable opportunity smoke outputs before further simulation/training/refit", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "episodes": total_episodes, "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "artifacts": [rel(SMOKE_DIR), rel(STATE_SMOKE), rel(SOURCE), rel(PROTOCOL_JSON), rel(req)]})
    raw = {
        "created_utc": created,
        "started_utc": started,
        "method": "vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke",
        "classification": "development_IMPROVED_trace_selected_terminal_stable_opportunity_probe_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "backup_proof": backup,
        "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)},
        "inputs": {"v1c_reanalysis_completed": rel(V1C_REANALYSIS_DONE), "v1c_reanalysis_completed_sha256": sha256(V1C_REANALYSIS_DONE), "v1c_bank": rel(inputs["v1c_bank_path"]), "v1c_bank_completed": rel(V1C_BANK_DONE), "v1c_bank_completed_sha256": sha256(V1C_BANK_DONE), "dryrun_completed": rel(DRYRUN_DIR / "completed.json"), "dryrun_completed_sha256": sha256(DRYRUN_DIR / "completed.json")},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": {"phase_A_scan_episodes": SCAN_EPISODES, "phase_B_branch_episodes": BRANCH_EPISODES, "episodes_upper_bound": SMOKE_EPISODES_EXACT_UPPER, "control_step_upper_bound": SMOKE_CONTROL_STEP_UPPER, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"episodes": total_episodes, "scan_episodes": len(scan_summaries), "branch_episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "environment_constructions": total_episodes, "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in scan_summaries + branch_episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "scan_episodes": scan_summaries,
        "target_selection": target_selection,
        "branch_schedule": branch_schedule,
        "branch_episodes": branch_episodes,
        "analysis": analysis,
        "backup_request_after_run": rel(req),
        "interpretation_limits": ["development diagnostic only", "fresh cases inherited from v1c bank", "branch targets selected by H15 trace before non-H15 rollout", "not validation/model selection", "not training/refit", "not ORIGINAL SAC", "not final test"],
    }
    write_json(SMOKE_DIR / "raw.json", raw)
    write_smoke_summary(raw)
    STATE_SMOKE.parent.mkdir(parents=True, exist_ok=True)
    STATE_SMOKE.write_text(
        f"# Vehicle stress-v1d trace-selected terminal-stable opportunity smoke\n\n"
        f"UTC: {created}. Completed {total_episodes} episodes / {control_steps} control steps. Robust positives={analysis['robust_positive_state_count']} across cases={analysis['robust_positive_cases']}; negative/neutral={analysis['negative_or_neutral_state_count']}; smoke gate={analysis['smoke_pass_to_next_label_or_refit_design']}. No validation64/test/training. Backup required before further simulation/refit.\n",
        encoding="utf-8",
    )
    append_docs(f"""<!-- {MARKER_SMOKE} -->
## 2026-09-29 vehicle stress-v1d trace-selected terminal-stable opportunity smoke

UTC: {created}. Development-only trace-selected terminal-stable opportunity smoke completed: {total_episodes} episodes, {control_steps} control steps, candidate resets=0. Robust-positive states={analysis['robust_positive_state_count']} across cases={analysis['robust_positive_cases']}; negative/neutral states={analysis['negative_or_neutral_state_count']}; smoke gate={analysis['smoke_pass_to_next_label_or_refit_design']}; blocking artifacts={analysis['blocking_artifact_count']}. No validation64-bank or sealed-test access, no training/refit. Artifacts: `{rel(SMOKE_DIR / 'summary.md')}`, `{rel(SMOKE_DIR / 'raw.json')}`, `{rel(SMOKE_DIR / 'completed.json')}`.
""", MARKER_SMOKE)
    files = [p for p in SMOKE_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_SMOKE, req, SOURCE, PROTOCOL_JSON, backup_proof, V1C_REANALYSIS_DONE, V1C_SMOKE_DONE, V1C_BANK_DONE, DRYRUN_DIR / "completed.json"]
    write_json(SMOKE_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "episodes": total_episodes,
        "scan_episodes": len(scan_summaries),
        "branch_episodes": len(branch_episodes),
        "control_steps": control_steps,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "backup_request": rel(req),
        "headline": {"robust_positive_state_count": analysis["robust_positive_state_count"], "negative_or_neutral_state_count": analysis["negative_or_neutral_state_count"], "robust_positive_cases": analysis["robust_positive_cases"], "smoke_pass_to_next_label_or_refit_design": analysis["smoke_pass_to_next_label_or_refit_design"], "blocking_artifact_count": analysis["blocking_artifact_count"], "train_or_refit_now": False, "next_action": "if smoke gate passes, freeze next label/refit design after backup; if fails, do not train/refit and pivot to terminal/modeling or sparse-opportunity diagnosis"},
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({"completed": rel(SMOKE_DIR / "completed.json"), "summary": rel(SMOKE_DIR / "summary.md"), "episodes": total_episodes, "control_steps": control_steps, "candidate_pool_resets": 0, "robust_positive_state_count": analysis["robust_positive_state_count"], "negative_or_neutral_state_count": analysis["negative_or_neutral_state_count"], "robust_positive_cases": analysis["robust_positive_cases"], "smoke_pass_to_next_label_or_refit_design": analysis["smoke_pass_to_next_label_or_refit_design"], "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no-simulation protocol/source readiness check")
    ap.add_argument("--run-smoke", action="store_true", help="run frozen v1d smoke after verified backup")
    ap.add_argument("--backup-proof", type=Path, default=None)
    ap.add_argument("--i-accept-development-terminal-stable-opportunity-probe", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_terminal_stable_opportunity_probe:
        raise ContractError("explicit --i-accept-development-terminal-stable-opportunity-probe required")
    if bool(args.dry_run) == bool(args.run_smoke):
        raise ContractError("exactly one of --dry-run or --run-smoke is required")
    if args.dry_run:
        return run_dry_run()
    if args.backup_proof is None:
        raise ContractError("--run-smoke requires --backup-proof")
    return run_smoke(args.backup_proof)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = DRYRUN_DIR if "--dry-run" in sys.argv else SMOKE_DIR if "--run-smoke" in sys.argv else ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_failure_unknown_{STAMP}"
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
            "new_refit_steps": 0,
            "next_recovery_hint": "Preserve partial output. If dry-run failed, repair source/protocol checks only; if smoke failed, audit partial scan/branch outputs before any rerun.",
        })
        raise
