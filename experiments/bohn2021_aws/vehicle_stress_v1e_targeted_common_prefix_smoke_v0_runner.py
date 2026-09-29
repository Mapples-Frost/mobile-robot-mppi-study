#!/usr/bin/env python3
"""Vehicle stress-v1e targeted common-prefix smoke runner v0.

Development-only IMPROVED diagnostic following the frozen prepare protocol:
  research_artifacts/aws_protocols/vehicle_stress_v1e_targeted_common_prefix_prepare_v0_frozen_20260929T0505Z.json

This runner is intentionally separated from the prepare step.  --dry-run performs
no simulations and checks that the stress-v1-only frozen schedule is executable
from the stress-v1 Stage1 bank/traces.  --run-smoke is blocked unless a verified
external backup proof postdates the prepare, this runner and dry-run artifacts.

No validation64 bank, no sealed test, no training/refit/gradient updates.
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
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

# Reuse the already-audited terminal-mode patch, prefix annotation and label
# helpers from v1d. Importing this module does not run simulations.
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402

NAME = "vehicle_stress_v1e_targeted_common_prefix_smoke_v0"
STAMP = "20260929T0515Z"
SOURCE = Path(__file__).resolve()
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

PREPARE_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_targeted_common_prefix_prepare_v0_20260929T0505Z"
PREPARE_DONE = PREPARE_DIR / "completed.json"
PREPARE_RAW = PREPARE_DIR / "raw.json"
PREPARE_SUMMARY = PREPARE_DIR / "summary.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1e_targeted_common_prefix_prepare_v0_frozen_20260929T0505Z.json"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1e_targeted_common_prefix_prepare_v0_frozen_20260929T0505Z.md"
STAGE1_BANK = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/bank/vehicle_stress_scenario_opportunity_probe_v1_bank.json"
STAGE1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/completed.json"
STAGE1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/raw.json"

DRYRUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_dryrun_{STAMP}"
SMOKE_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_run_{STAMP}"
STATE_DRYRUN = ROOT / f"research_artifacts/aws_state/{NAME}_dryrun_{STAMP}.md"
STATE_SMOKE = ROOT / f"research_artifacts/aws_state/{NAME}_run_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP_BEFORE_RUN = BACKUP_DIR / f"REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1E_TARGETED_COMMON_PREFIX_SMOKE_V0_RUN_{STAMP}.json"
MARKER_DRYRUN = f"vehicle-stress-v1e-targeted-common-prefix-smoke-v0-dryrun-{STAMP}"
MARKER_RUN = f"vehicle-stress-v1e-targeted-common-prefix-smoke-v0-run-{STAMP}"

PREFIX_H = 15
BRANCH_HORIZONS = [10, 15, 20, 25, 30, 45, 50]
TERMINAL_MODES = ["zero_terminal", "h15_common_terminal"]
MAX_STEPS = 150
TARGET_COUNT = 12
EPISODES_EXACT = TARGET_COUNT * len(BRANCH_HORIZONS) * len(TERMINAL_MODES)
CONTROL_STEP_UPPER = EPISODES_EXACT * MAX_STEPS
MATERIAL_GAIN = 3.0
STATE_DISTANCE_TOL = 1e-5


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


def parse_time(value: Any) -> Optional[dt.datetime]:
    return v1d.parse_time(value)


def source_mtime_utc() -> dt.datetime:
    return dt.datetime.fromtimestamp(SOURCE.stat().st_mtime, dt.timezone.utc)


def completed_ok(path: Path, check_hashes: bool = False) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"completed marker did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is not False:
        raise ContractError(f"sealed-test flag not false in {rel(path)}")
    if obj.get("historical_validation64_bank_opened") not in (False, None):
        raise ContractError(f"historical validation64 flag not false in {rel(path)}")
    if obj.get("validation64_bank_opened") not in (False, None):
        raise ContractError(f"validation64 flag not false in {rel(path)}")
    if check_hashes:
        for name, expected in (obj.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists() or sha256(p) != expected:
                raise ContractError(f"hash mismatch from {rel(path)}: {name}")
    return obj


def verify_inputs(load_bank: bool) -> Dict[str, Any]:
    required = [PREPARE_DONE, PREPARE_RAW, PREPARE_SUMMARY, PROTOCOL_JSON, PROTOCOL_MD, STAGE1_BANK, STAGE1_DONE, STAGE1_RAW]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        raise ContractError("missing required inputs: " + ", ".join(missing))
    prepare_done = completed_ok(PREPARE_DONE, check_hashes=False)
    stage1_done = completed_ok(STAGE1_DONE, check_hashes=False)
    protocol = read_json(PROTOCOL_JSON)
    prepare_raw = read_json(PREPARE_RAW)
    if protocol.get("protocol_id") != "vehicle_stress_v1e_targeted_common_prefix_prepare_v0_frozen_20260929T0505Z":
        raise ContractError("unexpected v1e protocol id")
    access = protocol.get("access_rules") or {}
    for key in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if access.get(key) is not False:
            raise ContractError(f"protocol access flag must be false: {key}")
    smoke = protocol.get("planned_smoke_after_backup") or {}
    if int(smoke.get("prefix_horizon", -1)) != PREFIX_H:
        raise ContractError("prefix horizon changed")
    if [int(x) for x in smoke.get("branch_horizons", [])] != BRANCH_HORIZONS:
        raise ContractError("branch horizons changed")
    if [str(x) for x in smoke.get("terminal_modes", [])] != TERMINAL_MODES:
        raise ContractError("terminal modes changed")
    schedule = smoke.get("schedule") or []
    targets = ((protocol.get("target_selection") or {}).get("targets") or [])
    if len(targets) != TARGET_COUNT or len(schedule) != EPISODES_EXACT:
        raise ContractError(f"unexpected target/schedule size: targets={len(targets)} schedule={len(schedule)}")
    expected_keys = {(int(t["target_index"]), str(mode), int(h)) for t in targets for mode in TERMINAL_MODES for h in BRANCH_HORIZONS}
    actual_keys = {(int(r["target_index"]), str(r["terminal_mode"]), int(r["horizon"])) for r in schedule}
    if actual_keys != expected_keys:
        raise ContractError("schedule does not cover exactly target x terminal_mode x horizon")
    if int(smoke.get("episodes_exact", -1)) != EPISODES_EXACT or int(smoke.get("control_step_upper_bound", -1)) != CONTROL_STEP_UPPER:
        raise ContractError("smoke budget changed")
    if int(smoke.get("new_training_episodes", -1)) != 0 or int(smoke.get("new_gradient_steps", -1)) != 0 or int(smoke.get("new_refit_steps", -1)) != 0:
        raise ContractError("protocol unexpectedly permits training/refit")
    target_by_idx = {int(t["target_index"]): t for t in targets}
    for row in schedule:
        t = target_by_idx[int(row["target_index"])]
        if int(row["case"]) != int(t["case"]) or int(row["branch_step"]) != int(t["branch_step"]):
            raise ContractError("schedule target metadata mismatch")
    bank_info: Dict[str, Any] = {"loaded": False}
    if load_bank:
        bank = read_json(STAGE1_BANK)
        selected_cases = bank.get("selected_cases_full") or bank.get("selected_cases")
        selection = bank.get("selection") or {}
        selected_meta = selection.get("selected_full_metadata") or selection.get("selected_metadata")
        if not isinstance(selected_cases, list) or len(selected_cases) < 20:
            raise ContractError("stress-v1 bank selected_cases missing or too short")
        if not isinstance(selected_meta, list) or len(selected_meta) < 20:
            raise ContractError("stress-v1 bank selected_metadata missing or too short")
        for t in targets:
            case = int(t["case"])
            if case < 0 or case >= len(selected_cases):
                raise ContractError(f"target case index out of bank range: {case}")
            if int(selected_meta[case].get("candidate_index", -999)) != int(t["source_candidate_index"]):
                raise ContractError(f"target candidate mismatch for case {case}")
        bank_info = {"loaded": True, "selected_cases": len(selected_cases), "selected_metadata": len(selected_meta), "bank_sha256": sha256(STAGE1_BANK)}
    hashes = {rel(p): sha256(p) for p in [SOURCE, PREPARE_DONE, PREPARE_RAW, PROTOCOL_JSON, PROTOCOL_MD, STAGE1_DONE, STAGE1_RAW]}
    # bank hash may already have been computed during load; avoid a second pass.
    hashes[rel(STAGE1_BANK)] = bank_info.get("bank_sha256") or (prepare_raw.get("input_hashes") or {}).get(rel(STAGE1_BANK)) or "not_rehashed_in_dryrun"
    return {"prepare_done": prepare_done, "stage1_done": stage1_done, "protocol": protocol, "prepare_raw": prepare_raw, "targets": targets, "schedule": schedule, "bank_info": bank_info, "hashes": hashes}


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if marker not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_dryrun_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle stress-v1e targeted common-prefix smoke v0 dry-run",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations, no candidate resets, no training/refit, no validation64 bank, no sealed test.",
        "",
        "## Verified executable design",
        "",
        f"- Frozen prepare protocol: `{raw['protocol_json']}` sha256 `{raw['protocol_sha256']}`.",
        f"- Targets: `{raw['target_count']}`; schedule episodes: `{raw['planned_smoke_episodes_exact']}`; control-step cap: `{raw['planned_smoke_control_step_upper_bound']}`.",
        f"- Branch horizons: `{BRANCH_HORIZONS}`; terminal modes: `{TERMINAL_MODES}`; prefix H `{PREFIX_H}`.",
        f"- Stress-v1 bank loaded: `{raw['bank_info']['loaded']}`; selected cases `{raw['bank_info'].get('selected_cases')}`.",
        "",
        "## Decision",
        "",
        "The simulation inputs are schema-consistent, but the smoke rollout remains blocked until a verified external backup covers the new runner, dry-run outputs and backup request. No selector/value refit is justified before the frozen smoke gate is run and passes.",
        "",
        f"Backup request before smoke: `{raw['backup_request_before_run']}`.",
    ]
    (DRYRUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dry_run() -> int:
    if (DRYRUN_DIR / "completed.json").exists():
        done = completed_ok(DRYRUN_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(DRYRUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if DRYRUN_DIR.exists() and any(p.name != "run.lock" for p in DRYRUN_DIR.iterdir()):
        raise ContractError(f"partial dry-run output exists; inspect first: {rel(DRYRUN_DIR)}")
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    created_dt = now_utc()
    inputs = verify_inputs(load_bank=True)
    write_json(REQUEST_BACKUP_BEFORE_RUN, {
        "requested_utc": created_dt.isoformat(),
        "reason": "backup v1e targeted common-prefix smoke runner/dry-run/protocol/prepare artifacts before any v1e rollout",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "planned_smoke_episodes_exact": EPISODES_EXACT,
        "planned_smoke_control_step_upper_bound": CONTROL_STEP_UPPER,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(SOURCE), rel(DRYRUN_DIR), rel(STATE_DRYRUN), rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(PREPARE_DIR), rel(REQUEST_BACKUP_BEFORE_RUN)],
    })
    raw = {
        "created_utc": created_dt.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": f"{NAME}_dryrun",
        "classification": "development_no_simulation_v1e_common_prefix_smoke_readiness",
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
        "protocol_json": rel(PROTOCOL_JSON),
        "protocol_sha256": sha256(PROTOCOL_JSON),
        "prepare_completed": rel(PREPARE_DONE),
        "prepare_completed_sha256": sha256(PREPARE_DONE),
        "target_count": len(inputs["targets"]),
        "target_cases": sorted({int(t["case"]) for t in inputs["targets"]}),
        "planned_smoke_episodes_exact": EPISODES_EXACT,
        "planned_smoke_control_step_upper_bound": CONTROL_STEP_UPPER,
        "bank_info": inputs["bank_info"],
        "source_hashes": inputs["hashes"],
        "backup_request_before_run": rel(REQUEST_BACKUP_BEFORE_RUN),
        "next_action": "after verified external backup, run --run-smoke with this frozen schedule; if gate fails, pivot to terminal/modeling or sparse-opportunity scenario diagnosis rather than selector/refit",
    }
    write_json(DRYRUN_DIR / "raw.json", raw)
    write_dryrun_summary(raw)
    STATE_DRYRUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_DRYRUN.write_text((DRYRUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER_DRYRUN} -->
## 2026-09-29 vehicle stress-v1e targeted common-prefix smoke v0 dry-run

UTC: {created_dt.isoformat()}. No-simulation readiness diagnostic completed for the stress-v1-only v1e common-prefix smoke. Verified the frozen prepare protocol, 12 targets, 168 scheduled episodes, stress-v1 bank target/candidate consistency and closed validation/test access. Rollout remains blocked until verified external backup covers `{rel(REQUEST_BACKUP_BEFORE_RUN)}` plus this runner/dry-run/protocol/prepare artifacts.
""", MARKER_DRYRUN)
    files = [p for p in DRYRUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, STATE_DRYRUN, REQUEST_BACKUP_BEFORE_RUN, PROTOCOL_JSON, PROTOCOL_MD, PREPARE_DONE, PREPARE_RAW, STAGE1_DONE]
    write_json(DRYRUN_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created_dt.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds(),
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
        "planned_smoke_episodes_exact": EPISODES_EXACT,
        "planned_smoke_control_step_upper_bound": CONTROL_STEP_UPPER,
        "backup_required_before_smoke": True,
        "backup_request": rel(REQUEST_BACKUP_BEFORE_RUN),
        "headline": {"v1e_smoke_inputs_schema_consistent": True, "target_count": len(inputs["targets"]), "planned_smoke_episodes_exact": EPISODES_EXACT, "train_or_refit_now": False, "smoke_blocked_until_verified_backup": True},
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(DRYRUN_DIR / "completed.json"),
        "summary": rel(DRYRUN_DIR / "summary.md"),
        "target_count": len(inputs["targets"]),
        "target_cases": raw["target_cases"],
        "planned_smoke_episodes_exact": EPISODES_EXACT,
        "planned_smoke_control_step_upper_bound": CONTROL_STEP_UPPER,
        "train_or_refit_now": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(REQUEST_BACKUP_BEFORE_RUN),
    }, sort_keys=True), flush=True)
    return 0


def finite_summary(vals: Iterable[float]) -> Dict[str, Any]:
    return v1d.finite_summary(vals)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def is_control_target(target: Mapping[str, Any]) -> bool:
    role = str(target.get("case_role") or target.get("role") or "")
    return "negative_control" in role or "lower_stress_control" in role


def analyze_episodes(branch_episodes: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_target_mode: Dict[Tuple[str, str], Dict[int, Mapping[str, Any]]] = {}
    for e in branch_episodes:
        by_target_mode.setdefault((str(e["state_id"]), str(e["terminal_mode"])), {})[int(e["branch_horizon"])] = e
    target_by_sid = {str(t["state_id"]): t for t in targets}
    artifact_flags = {"missing_H15_reference": 0, "missing_horizon_or_terminal_mode": 0, "branch_not_reached": 0, "state_distance_gt_tol": 0, "physical_prefix_mismatch": 0, "safety_solver_regression": 0, "positive_with_common_terminal_non_success": 0}
    state_rows: List[Dict[str, Any]] = []
    robust_positive_cases: List[int] = []
    robust_positive_horizon_counts: Dict[str, int] = {}
    timing_by_h_mode: Dict[str, List[float]] = {}
    control_state_count = 0
    control_positive_states = 0
    for sid, target in target_by_sid.items():
        case_id = int(target["case"])
        is_control = is_control_target(target)
        if is_control:
            control_state_count += 1
        robust_sets: List[set] = []
        all_comps: List[Dict[str, Any]] = []
        mode_results: Dict[str, Any] = {}
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
            ref_state = v1d.state_tuple(ref.get("branch_previous_state"))
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
                cand_state = v1d.state_tuple(e.get("branch_previous_state"))
                dist = v1d.state_distance(cand_state, ref_state)
                prefix_match = bool(e.get("physical_prefix_equivalence_sha256") == ref_prefix_hash)
                if h != PREFIX_H and dist > STATE_DISTANCE_TOL:
                    artifact_flags["state_distance_gt_tol"] += 1
                if h != PREFIX_H and not prefix_match:
                    artifact_flags["physical_prefix_mismatch"] += 1
                phys_gain = float(ref["continuation_physical_constraint_cost_from_branch"] - e["continuation_physical_constraint_cost_from_branch"])
                total_gain = float(ref["continuation_total_cost_from_branch"] - e["continuation_total_cost_from_branch"])
                safety_ok = v1d.no_regression(e, ref)
                if h != PREFIX_H and not safety_ok:
                    artifact_flags["safety_solver_regression"] += 1
                material = bool(h != PREFIX_H and bool(e.get("branch_reached")) and bool(ref.get("branch_reached")) and dist <= STATE_DISTANCE_TOL and prefix_match and safety_ok and phys_gain >= MATERIAL_GAIN)
                if material:
                    material_h.append(h)
                bdiag = e.get("branch_step_mpc_diag") or {}
                comp = {"state_id": sid, "case": case_id, "case_role": target.get("case_role"), "terminal_mode": mode, "horizon": int(h), "success": bool(e.get("success")), "constraint": bool(e.get("constraint")), "steps": int(e.get("steps", 0)), "initial_failed_steps": int(e.get("initial_failed_steps", 0)), "final_failed_steps": int(e.get("final_failed_steps", 0)), "solver_failure_steps": int(e.get("solver_failure_steps", 0)), "branch_reached": bool(e.get("branch_reached")), "continuation_physical": safe_float(e.get("continuation_physical_constraint_cost_from_branch"), 0.0), "continuation_total": safe_float(e.get("continuation_total_cost_from_branch"), 0.0), "gain_vs_H15_physical": phys_gain, "gain_vs_H15_total": total_gain, "state_distance_vs_H15": dist, "physical_prefix_matches_H15": prefix_match, "no_success_constraint_solver_regression_vs_H15": safety_ok, "material_positive_terminal_mode": material, "branch_mpc_value_fn": bdiag.get("info_mpc_value_fn"), "branch_objective_opt_f_num": bdiag.get("objective_opt_f_num"), "decision_sum_s": safe_float((e.get("decision_timing_s") or {}).get("sum"), 0.0), "solver_attempt_sum_s": safe_float((e.get("solver_attempt_timing_s") or {}).get("sum"), 0.0), "path": e.get("path")}
                comps.append(comp)
                all_comps.append(comp)
            mode_results[mode] = {"reference_H15": {"success": bool(ref.get("success")), "constraint": bool(ref.get("constraint")), "continuation_physical": safe_float(ref.get("continuation_physical_constraint_cost_from_branch"), 0.0), "continuation_total": safe_float(ref.get("continuation_total_cost_from_branch"), 0.0), "steps": int(ref.get("steps", 0)), "path": ref.get("path")}, "material_horizons": sorted(material_h), "comparisons": comps}
            robust_sets.append(set(material_h))
        robust_h = sorted(set.intersection(*robust_sets) if robust_sets else set())
        robust_positive = bool(robust_h)
        if robust_positive:
            robust_positive_cases.append(case_id)
            if is_control:
                control_positive_states += 1
            for h in robust_h:
                robust_positive_horizon_counts[str(h)] = robust_positive_horizon_counts.get(str(h), 0) + 1
            for comp in all_comps:
                if comp["horizon"] in robust_h and comp["terminal_mode"] == "h15_common_terminal" and not comp["success"]:
                    artifact_flags["positive_with_common_terminal_non_success"] += 1
        best_by_phys = min(all_comps, key=lambda r: (r["continuation_physical"], r["decision_sum_s"], r["horizon"])) if all_comps else None
        state_rows.append({"state_id": sid, "target_index": int(target["target_index"]), "case": case_id, "source_candidate_index": int(target.get("source_candidate_index", -1)), "case_role": target.get("case_role"), "selection_group": target.get("selection_group"), "branch_step": int(target["branch_step"]), "is_control_state": is_control, "robust_positive_state": robust_positive, "robust_positive_horizons": robust_h, "label": "robust_positive_non_H15" if robust_positive else "negative_or_neutral", "best_by_physical_any_mode": best_by_phys, "mode_results": mode_results})
    positive_count = sum(1 for r in state_rows if r["robust_positive_state"])
    negative_count = sum(1 for r in state_rows if not r["robust_positive_state"])
    distinct_cases = sorted(set(robust_positive_cases))
    blocking_artifacts = int(sum(artifact_flags.values()))
    control_fpr = float(control_positive_states) / float(control_state_count) if control_state_count else 0.0
    gate = bool(positive_count >= 2 and len(distinct_cases) >= 2 and negative_count >= 4 and control_fpr <= 0.25 and blocking_artifacts == 0)
    return {"state_count": len(state_rows), "robust_positive_state_count": positive_count, "negative_or_neutral_state_count": negative_count, "robust_positive_cases": distinct_cases, "robust_positive_horizon_counts": robust_positive_horizon_counts, "control_state_count": control_state_count, "control_positive_state_count": control_positive_states, "control_false_positive_rate": control_fpr, "artifact_flags": artifact_flags, "blocking_artifact_count": blocking_artifacts, "smoke_pass_to_selector_or_value_refit_design": gate, "timing_decision_sum_by_horizon_mode_s": {k: finite_summary(v) for k, v in sorted(timing_by_h_mode.items())}, "state_rows": state_rows, "decision": {"train_or_refit_now": False, "if_gate_passes": "development evidence only; after backup freeze compact IMPROVED selector/value-refit smoke with strong fixed-H/Pareto baselines", "if_gate_fails": "do not train/refit selector from these labels; pivot to terminal/modeling or documented sparse-opportunity scenario diagnosis"}}


def write_smoke_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = [
        "# Vehicle stress-v1e targeted common-prefix smoke v0",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only; no training/refit, no validation64 bank, no sealed test.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Headline",
        "",
        f"- Robust-positive states: `{a['robust_positive_state_count']}` across cases `{a['robust_positive_cases']}`.",
        f"- Negative/neutral states: `{a['negative_or_neutral_state_count']}`.",
        f"- Robust positive horizons: `{a['robust_positive_horizon_counts']}`.",
        f"- Control positive states: `{a['control_positive_state_count']}/{a['control_state_count']}` (FPR `{a['control_false_positive_rate']:.3f}`).",
        f"- Blocking artifacts: `{a['blocking_artifact_count']}`; flags `{a['artifact_flags']}`.",
        f"- Smoke pass to selector/value-refit design: `{a['smoke_pass_to_selector_or_value_refit_design']}`.",
        "",
        "## Per-state labels",
        "",
        "| state | case | role | branch | label | robust H | best phys mode/H/cost |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for row in a["state_rows"]:
        best = row.get("best_by_physical_any_mode") or {}
        lines.append("| `%s` | %d | `%s` | %d | `%s` | `%s` | `%s`/H%s/%.6g |" % (row["state_id"], int(row["case"]), row.get("case_role"), int(row["branch_step"]), row.get("label"), row.get("robust_positive_horizons"), best.get("terminal_mode"), str(best.get("horizon")), float(best.get("continuation_physical", 0.0))))
    lines += [
        "",
        "## Decision",
        "",
        f"- Train/refit now: `{a['decision']['train_or_refit_now']}`.",
        f"- Backup request: `{raw['backup_request_after_run']}`.",
    ]
    (SMOKE_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(backup_proof: Path) -> int:
    if not (DRYRUN_DIR / "completed.json").exists():
        raise ContractError("smoke requires completed v1e dry-run marker")
    dry_done = completed_ok(DRYRUN_DIR / "completed.json", check_hashes=True)
    if (SMOKE_DIR / "completed.json").exists():
        done = completed_ok(SMOKE_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(SMOKE_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if SMOKE_DIR.exists() and any(p.name != "run.lock" for p in SMOKE_DIR.iterdir()):
        raise ContractError(f"partial smoke output exists; inspect first: {rel(SMOKE_DIR)}")
    inputs = verify_inputs(load_bank=True)
    min_time = max(t for t in [source_mtime_utc(), parse_time(dry_done.get("created_utc")), parse_time(inputs["prepare_done"].get("created_utc"))] if t is not None)
    backup = v1d.verify_backup_proof(backup_proof, min_time, "vehicle_stress_v1e_targeted_common_prefix_smoke_v0")
    bank = read_json(STAGE1_BANK)
    selected_cases = bank.get("selected_cases_full") or bank.get("selected_cases")
    base_smoke, stage1_runner, _ = v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError(f"legacy runtime preflight failed: {preflight}")
    stage1_runner.base.v1.latency_verify()
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    started = now_utc().isoformat()
    write_json(SMOKE_DIR / "run_started.json", {"started_utc": started, "pid": os.getpid(), "method": f"{NAME}_run", "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0})
    write_json(SMOKE_DIR / "runtime_preflight.json", preflight)
    write_json(SMOKE_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})
    base_smoke.OUT = SMOKE_DIR
    base_smoke.PREFIX_H = PREFIX_H
    base_smoke.MAX_STEPS = MAX_STEPS
    base_smoke.MATERIAL_GAIN = MATERIAL_GAIN
    branch_episodes: List[Dict[str, Any]] = []
    target_by_idx = {int(t["target_index"]): t for t in inputs["targets"]}
    for item in inputs["schedule"]:
        case_id = int(item["case"])
        summary = base_smoke.run_one(item, selected_cases[case_id], terminals, terminal_receipts)
        summary = v1d.update_h15_common_receipt(summary, terminal_receipts)
        target = target_by_idx[int(item["target_index"])]
        summary.update({"phase": "v1e_targeted_common_prefix_branch_rollout", "source_candidate_index": int(item["source_candidate_index"]), "selection_group": item.get("selection_group"), "target_index": int(item["target_index"]), "case_role": target.get("case_role"), "protocol_terminal_label_family": "stress_v1e_zero_or_H15_common_terminal_stable"})
        summary = v1d.annotate_physical_prefix(summary)
        branch_episodes.append(summary)
        control_steps_done = int(sum(int(e.get("steps", 0)) for e in branch_episodes))
        progress = {"pid": os.getpid(), "episodes_done": len(branch_episodes), "episodes_expected": EPISODES_EXACT, "control_steps_done": control_steps_done, "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "case", "branch_step", "branch_horizon", "terminal_mode", "steps", "success", "termination", "branch_reached")}, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(SMOKE_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in branch_episodes))
    if len(branch_episodes) != EPISODES_EXACT or control_steps > CONTROL_STEP_UPPER:
        raise ContractError("v1e smoke budget violation")
    analysis = analyze_episodes(branch_episodes, inputs["targets"])
    created_dt = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1E_TARGETED_COMMON_PREFIX_SMOKE_V0_RUN_%s.json" % created_dt.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created_dt.isoformat(), "reason": "backup v1e targeted common-prefix smoke outputs before further simulation/training/refit", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "artifacts": [rel(SMOKE_DIR), rel(STATE_SMOKE), rel(SOURCE), rel(PROTOCOL_JSON), rel(req)]})
    raw = {"created_utc": created_dt.isoformat(), "started_utc": started, "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds(), "method": f"{NAME}_run", "classification": "development_IMPROVED_stress_v1e_targeted_common_prefix_smoke_not_validation_not_final_test", "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "backup_proof": backup, "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)}, "inputs": {"dryrun_completed": rel(DRYRUN_DIR / "completed.json"), "prepare_completed": rel(PREPARE_DONE), "stage1_bank": rel(STAGE1_BANK)}, "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}}, "runtime_preflight": preflight, "budget_declared": {"episodes_exact": EPISODES_EXACT, "control_step_upper_bound": CONTROL_STEP_UPPER, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "budget_actual": {"episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "environment_constructions": len(branch_episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in branch_episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "targets": inputs["targets"], "schedule": inputs["schedule"], "branch_episodes": branch_episodes, "analysis": analysis, "backup_request_after_run": rel(req), "interpretation_limits": ["development diagnostic only", "stress-v1 targeted states selected after development evidence", "not validation/model selection", "not training/refit", "not ORIGINAL SAC", "not final test"]}
    write_json(SMOKE_DIR / "raw.json", raw)
    write_smoke_summary(raw)
    STATE_SMOKE.parent.mkdir(parents=True, exist_ok=True)
    STATE_SMOKE.write_text((SMOKE_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER_RUN} -->
## 2026-09-29 vehicle stress-v1e targeted common-prefix smoke v0 run

UTC: {created_dt.isoformat()}. Development-only v1e targeted common-prefix smoke completed: {len(branch_episodes)} episodes, {control_steps} control steps. Robust-positive states={analysis['robust_positive_state_count']} across cases={analysis['robust_positive_cases']}; gate={analysis['smoke_pass_to_selector_or_value_refit_design']}; blocking artifacts={analysis['blocking_artifact_count']}. No validation64-bank or sealed-test access, no training/refit. Artifacts: `{rel(SMOKE_DIR / 'summary.md')}`, `{rel(SMOKE_DIR / 'raw.json')}`, `{rel(SMOKE_DIR / 'completed.json')}`.
""", MARKER_RUN)
    files = [p for p in SMOKE_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, STATE_SMOKE, req, PROTOCOL_JSON, PREPARE_DONE, DRYRUN_DIR / "completed.json", backup_proof]
    write_json(SMOKE_DIR / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created_dt.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds(), "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "backup_request": rel(req), "headline": {"robust_positive_state_count": analysis["robust_positive_state_count"], "negative_or_neutral_state_count": analysis["negative_or_neutral_state_count"], "robust_positive_cases": analysis["robust_positive_cases"], "smoke_pass_to_selector_or_value_refit_design": analysis["smoke_pass_to_selector_or_value_refit_design"], "blocking_artifact_count": analysis["blocking_artifact_count"], "train_or_refit_now": False, "next_action": "if gate passes, freeze compact selector/value-refit design after backup; if fails, do not train/refit and pivot to terminal/modeling or sparse-opportunity diagnosis"}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(SMOKE_DIR / "completed.json"), "summary": rel(SMOKE_DIR / "summary.md"), "episodes": len(branch_episodes), "control_steps": control_steps, "robust_positive_state_count": analysis["robust_positive_state_count"], "negative_or_neutral_state_count": analysis["negative_or_neutral_state_count"], "robust_positive_cases": analysis["robust_positive_cases"], "smoke_pass_to_selector_or_value_refit_design": analysis["smoke_pass_to_selector_or_value_refit_design"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-smoke", action="store_true")
    ap.add_argument("--backup-proof", type=Path, default=None)
    ap.add_argument("--i-accept-development-v1e-common-prefix-smoke", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_v1e_common_prefix_smoke:
        raise ContractError("explicit --i-accept-development-v1e-common-prefix-smoke required")
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
        target = DRYRUN_DIR if "--dry-run" in sys.argv else SMOKE_DIR if "--run-smoke" in sys.argv else ROOT / f"research_artifacts/aws_diagnostics/{NAME}_failure_unknown_{STAMP}"
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "failure.json", {"failed_utc": now_utc().isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_rollouts": 0 if "--dry-run" in sys.argv else None, "new_control_steps": 0 if "--dry-run" in sys.argv else None, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "next_recovery_hint": "Preserve partial output. If dry-run failed, repair schema/source only; if smoke failed, audit partial branch outputs before rerun."})
        raise
