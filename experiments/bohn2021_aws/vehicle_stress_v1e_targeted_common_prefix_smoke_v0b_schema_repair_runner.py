#!/usr/bin/env python3
"""Vehicle stress-v1e targeted common-prefix smoke runner v0b schema repair.

This is a one-variable implementation/schema repair for the v1e smoke runner.
The frozen v1e prepare protocol stored target role metadata as ``case_role``;
the inherited Stage2 branch runner expects ``item['role']``.  v0 failed with
KeyError('role') before completing scientific labels.  v0b copies each frozen
schedule item and adds a role alias before calling the inherited runner.

No scenario, horizon grid, terminal mode, label rule, gate, training/refit budget,
validation64 access, or sealed-test access is changed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_stress_v1e_targeted_common_prefix_smoke_v0_runner as v0  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402

NAME = "vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair"
STAMP = "20260929T0545Z"
SOURCE = Path(__file__).resolve()
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

AMENDMENT_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_amendment_20260929T0545Z.json"
PARENT_FAILURE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_targeted_common_prefix_smoke_v0_run_20260929T0515Z/failure.json"
PARENT_PARTIAL_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_targeted_common_prefix_smoke_v0_run_20260929T0515Z"
DRYRUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_dryrun_{STAMP}"
SMOKE_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_run_{STAMP}"
STATE_DRYRUN = ROOT / f"research_artifacts/aws_state/{NAME}_dryrun_{STAMP}.md"
STATE_SMOKE = ROOT / f"research_artifacts/aws_state/{NAME}_run_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP_BEFORE_RUN = BACKUP_DIR / f"REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1E_TARGETED_COMMON_PREFIX_SMOKE_V0B_SCHEMA_REPAIR_RUN_{STAMP}.json"
MARKER_DRYRUN = f"vehicle-stress-v1e-targeted-common-prefix-smoke-v0b-schema-repair-dryrun-{STAMP}"
MARKER_RUN = f"vehicle-stress-v1e-targeted-common-prefix-smoke-v0b-schema-repair-run-{STAMP}"

PREFIX_H = v0.PREFIX_H
BRANCH_HORIZONS = list(v0.BRANCH_HORIZONS)
TERMINAL_MODES = list(v0.TERMINAL_MODES)
MAX_STEPS = v0.MAX_STEPS
TARGET_COUNT = v0.TARGET_COUNT
EPISODES_EXACT = v0.EPISODES_EXACT
CONTROL_STEP_UPPER = v0.CONTROL_STEP_UPPER
MATERIAL_GAIN = v0.MATERIAL_GAIN


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    return v0.rel(path)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def read_json(path: Path) -> Any:
    return v0.read_json(path)


def write_json(path: Path, value: Any) -> None:
    v0.write_json(path, value)


def sha256(path: Path) -> str:
    return v0.sha256(path)


def parse_time(value: Any) -> Optional[dt.datetime]:
    return v0.parse_time(value)


def completed_ok(path: Path, check_hashes: bool = False) -> Mapping[str, Any]:
    return v0.completed_ok(path, check_hashes=check_hashes)


def source_mtime_utc() -> dt.datetime:
    return dt.datetime.fromtimestamp(SOURCE.stat().st_mtime, dt.timezone.utc)


def file_mtime_utc(path: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)


def verify_amendment() -> Mapping[str, Any]:
    if not AMENDMENT_JSON.exists():
        raise ContractError(f"missing schema-repair amendment: {rel(AMENDMENT_JSON)}")
    obj = read_json(AMENDMENT_JSON)
    if obj.get("protocol_id") != "vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_amendment_20260929T0545Z":
        raise ContractError("unexpected v0b schema-repair amendment id")
    access = obj.get("access_rules") or {}
    for key in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if access.get(key) is not False:
            raise ContractError(f"amendment access flag must be false: {key}")
    budgets = obj.get("budgets") or {}
    if int(budgets.get("new_training_episodes", -1)) != 0 or int(budgets.get("new_gradient_steps", -1)) != 0 or int(budgets.get("new_refit_steps", -1)) != 0:
        raise ContractError("amendment unexpectedly permits training/refit")
    if int(budgets.get("planned_smoke_episodes_exact", -1)) != EPISODES_EXACT:
        raise ContractError("amendment changed v1e smoke episode count")
    if int(budgets.get("planned_smoke_control_step_upper_bound", -1)) != CONTROL_STEP_UPPER:
        raise ContractError("amendment changed v1e smoke control-step cap")
    return obj


def audit_parent_failure() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "parent_failure_path": rel(PARENT_FAILURE),
        "parent_failure_exists": PARENT_FAILURE.exists(),
        "schema_error_confirmed": False,
        "partial_episode_dirs": 0,
        "partial_trace_jsonl_files": 0,
        "partial_trace_jsonl_rows": 0,
        "partial_completed_markers": 0,
        "note": "metadata/trace-file audit only; no new simulation",
    }
    if PARENT_FAILURE.exists():
        failure = read_json(PARENT_FAILURE)
        out["parent_failure_sha256"] = sha256(PARENT_FAILURE)
        out["exception"] = failure.get("exception")
        out["schema_error_confirmed"] = "KeyError('role')" in str(failure.get("exception")) or "KeyError: 'role'" in str(failure.get("traceback"))
    ep_root = PARENT_PARTIAL_DIR / "episodes"
    if ep_root.exists():
        dirs = [p for p in sorted(ep_root.iterdir()) if p.is_dir()]
        out["partial_episode_dirs"] = len(dirs)
        for ep in dirs:
            if (ep / "completed.json").exists():
                out["partial_completed_markers"] += 1
            trace = ep / "trace.jsonl"
            if trace.exists():
                out["partial_trace_jsonl_files"] += 1
                try:
                    with trace.open("r", encoding="utf-8") as stream:
                        out["partial_trace_jsonl_rows"] += sum(1 for _ in stream)
                except Exception as exc:
                    out.setdefault("trace_count_errors", []).append({"path": rel(trace), "exception": repr(exc)})
    return out


def verify_inputs(load_bank: bool = True) -> Dict[str, Any]:
    inputs = dict(v0.verify_inputs(load_bank=load_bank))
    amendment = verify_amendment()
    parent_dry = completed_ok(v0.DRYRUN_DIR / "completed.json", check_hashes=True)
    parent_failure = audit_parent_failure()
    if not parent_failure.get("schema_error_confirmed"):
        raise ContractError("parent v0 failure does not confirm KeyError('role'); refusing schema-repair continuation")
    inputs["schema_repair_amendment"] = amendment
    inputs["parent_v0_dryrun_completed"] = parent_dry
    inputs["parent_v0_failure_audit"] = parent_failure
    return inputs


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if marker not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_dryrun_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle stress-v1e targeted common-prefix smoke v0b schema-repair dry-run",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations, no candidate resets, no training/refit, no validation64 bank, no sealed test.",
        "",
        "## Schema repair",
        "",
        "- Parent v0 smoke failed with `KeyError('role')` before completed scientific labels.",
        "- v0b adds only `role = case_role` in the per-schedule item passed to the inherited Stage2 runner.",
        "- Frozen v1e targets, schedule, horizon grid, terminal modes, label rule and gate are unchanged.",
        "",
        "## Verified executable design",
        "",
        f"- Amendment: `{raw['amendment_json']}` sha256 `{raw['amendment_sha256']}`.",
        f"- Parent prepare protocol: `{raw['parent_protocol_json']}` sha256 `{raw['parent_protocol_sha256']}`.",
        f"- Targets: `{raw['target_count']}`; schedule episodes: `{raw['planned_smoke_episodes_exact']}`; control-step cap: `{raw['planned_smoke_control_step_upper_bound']}`.",
        f"- Branch horizons: `{BRANCH_HORIZONS}`; terminal modes: `{TERMINAL_MODES}`; prefix H `{PREFIX_H}`.",
        f"- Parent partial v0 audit: `{raw['parent_v0_failure_audit']}`.",
        "",
        "## Decision",
        "",
        "Run the bounded v0b smoke only after verified external backup of this repair/dry-run/amendment. The failed v0 partial episode remains implementation-failure evidence only and is not counted as a completed label bank.",
        "",
        f"Backup request before v0b smoke: `{raw['backup_request_before_run']}`.",
    ]
    (DRYRUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dry_run() -> int:
    done_path = DRYRUN_DIR / "completed.json"
    if done_path.exists():
        done = completed_ok(done_path, check_hashes=True)
        print(json.dumps({"already_completed": rel(done_path), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if DRYRUN_DIR.exists() and any(p.name != "run.lock" for p in DRYRUN_DIR.iterdir()):
        raise ContractError(f"partial dry-run output exists; inspect first: {rel(DRYRUN_DIR)}")
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    created_dt = now_utc()
    inputs = verify_inputs(load_bank=True)
    parent_protocol = read_json(v0.PROTOCOL_JSON)
    parent_targets = inputs["targets"]
    target_roles = sorted({str(t.get("case_role")) for t in parent_targets})
    missing_case_role = [int(t.get("target_index", -1)) for t in parent_targets if not t.get("case_role")]
    if missing_case_role:
        raise ContractError(f"targets missing case_role, cannot alias role: {missing_case_role}")
    write_json(REQUEST_BACKUP_BEFORE_RUN, {
        "requested_utc": created_dt.isoformat(),
        "reason": "backup v1e v0b schema-repair source/dry-run/amendment before rerunning targeted common-prefix smoke",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "planned_smoke_episodes_exact": EPISODES_EXACT,
        "planned_smoke_control_step_upper_bound": CONTROL_STEP_UPPER,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(SOURCE), rel(AMENDMENT_JSON), rel(DRYRUN_DIR), rel(STATE_DRYRUN), rel(v0.PROTOCOL_JSON), rel(v0.PREPARE_DIR), rel(v0.DRYRUN_DIR), rel(PARENT_FAILURE), rel(REQUEST_BACKUP_BEFORE_RUN)],
    })
    raw = {
        "created_utc": created_dt.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": f"{NAME}_dryrun",
        "classification": "development_no_simulation_schema_repair_readiness_not_validation_not_final_test",
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
        "schema_repair": "add role alias from case_role for inherited Stage2 run_one",
        "amendment_json": rel(AMENDMENT_JSON),
        "amendment_sha256": sha256(AMENDMENT_JSON),
        "parent_protocol_json": rel(v0.PROTOCOL_JSON),
        "parent_protocol_sha256": sha256(v0.PROTOCOL_JSON),
        "parent_protocol_id": parent_protocol.get("protocol_id"),
        "parent_prepare_completed": rel(v0.PREPARE_DONE),
        "parent_dryrun_completed": rel(v0.DRYRUN_DIR / "completed.json"),
        "parent_v0_failure_audit": inputs["parent_v0_failure_audit"],
        "target_count": len(parent_targets),
        "target_cases": sorted({int(t["case"]) for t in parent_targets}),
        "target_roles": target_roles,
        "planned_smoke_episodes_exact": EPISODES_EXACT,
        "planned_smoke_control_step_upper_bound": CONTROL_STEP_UPPER,
        "bank_info": inputs["bank_info"],
        "source_hashes": {rel(SOURCE): sha256(SOURCE), rel(AMENDMENT_JSON): sha256(AMENDMENT_JSON), rel(v0.SOURCE): sha256(v0.SOURCE), rel(v0.PROTOCOL_JSON): sha256(v0.PROTOCOL_JSON), rel(v0.DRYRUN_DIR / "completed.json"): sha256(v0.DRYRUN_DIR / "completed.json")},
        "backup_request_before_run": rel(REQUEST_BACKUP_BEFORE_RUN),
        "next_action": "after verified external backup, run --run-smoke using role=case_role alias; if gate fails, do not refit from labels",
    }
    write_json(DRYRUN_DIR / "raw.json", raw)
    write_dryrun_summary(raw)
    STATE_DRYRUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_DRYRUN.write_text((DRYRUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER_DRYRUN} -->
## 2026-09-29 vehicle stress-v1e targeted common-prefix smoke v0b schema-repair dry-run

UTC: {created_dt.isoformat()}. No-simulation schema repair readiness completed. Parent v0 failure is confirmed as `KeyError('role')`; v0b changes only the schedule-item role alias (`role = case_role`) before calling the inherited branch runner. Frozen v1e targets/schedule/gate remain unchanged: {EPISODES_EXACT} planned episodes, control-step cap {CONTROL_STEP_UPPER}, no training/refit, no validation64 or sealed-test access. Run is blocked until verified external backup covers `{rel(REQUEST_BACKUP_BEFORE_RUN)}` and v0b source/amendment/dry-run artifacts.
""", MARKER_DRYRUN)
    files = [p for p in DRYRUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, AMENDMENT_JSON, STATE_DRYRUN, REQUEST_BACKUP_BEFORE_RUN, v0.PROTOCOL_JSON, v0.PREPARE_DONE, v0.PREPARE_RAW, v0.DRYRUN_DIR / "completed.json", PARENT_FAILURE]
    write_json(done_path, {
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
        "headline": {"schema_repair_ready": True, "parent_schema_error_confirmed": True, "target_count": len(parent_targets), "planned_smoke_episodes_exact": EPISODES_EXACT, "train_or_refit_now": False, "smoke_blocked_until_verified_backup": True},
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(done_path),
        "summary": rel(DRYRUN_DIR / "summary.md"),
        "schema_repair_ready": True,
        "target_count": len(parent_targets),
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
    return v0.finite_summary(vals)


def write_smoke_summary(raw: Mapping[str, Any]) -> None:
    # Reuse the parent summary writer after redirecting its output directory.
    old_dir = v0.SMOKE_DIR
    try:
        v0.SMOKE_DIR = SMOKE_DIR
        v0.write_smoke_summary(raw)
    finally:
        v0.SMOKE_DIR = old_dir
    p = SMOKE_DIR / "summary.md"
    text = p.read_text(encoding="utf-8")
    prefix = "# Vehicle stress-v1e targeted common-prefix smoke v0b schema-repair\n\nSchema repair: the frozen v1e schedule field `case_role` was aliased to inherited runner field `role`; all scientific design variables are unchanged from the parent v1e protocol.\n\n"
    if not text.startswith("# Vehicle stress-v1e targeted common-prefix smoke v0b"):
        p.write_text(prefix + text, encoding="utf-8")


def run_smoke(backup_proof: Path) -> int:
    dry_path = DRYRUN_DIR / "completed.json"
    if not dry_path.exists():
        raise ContractError("v0b smoke requires completed v0b dry-run marker")
    dry_done = completed_ok(dry_path, check_hashes=True)
    done_path = SMOKE_DIR / "completed.json"
    if done_path.exists():
        done = completed_ok(done_path, check_hashes=True)
        print(json.dumps({"already_completed": rel(done_path), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if SMOKE_DIR.exists() and any(p.name != "run.lock" for p in SMOKE_DIR.iterdir()):
        raise ContractError(f"partial v0b smoke output exists; inspect first: {rel(SMOKE_DIR)}")
    inputs = verify_inputs(load_bank=True)
    min_times = [source_mtime_utc(), file_mtime_utc(AMENDMENT_JSON), parse_time(dry_done.get("created_utc")), parse_time(inputs["prepare_done"].get("created_utc")), parse_time(inputs["parent_v0_dryrun_completed"].get("created_utc"))]
    min_time = max(t for t in min_times if t is not None)
    backup = v1d.verify_backup_proof(backup_proof, min_time, NAME)
    bank = read_json(v0.STAGE1_BANK)
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
    write_json(SMOKE_DIR / "run_started.json", {"started_utc": started, "pid": os.getpid(), "method": f"{NAME}_run", "schema_repair": "role_alias_from_case_role", "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0})
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
        target = target_by_idx[int(item["target_index"])]
        run_item = dict(item)
        role_alias = str(run_item.get("role") or run_item.get("case_role") or target.get("case_role") or "unspecified_v1e_role")
        run_item["role"] = role_alias
        summary = base_smoke.run_one(run_item, selected_cases[case_id], terminals, terminal_receipts)
        summary = v1d.update_h15_common_receipt(summary, terminal_receipts)
        summary.update({
            "phase": "v1e_targeted_common_prefix_branch_rollout_v0b_schema_repair",
            "source_candidate_index": int(item["source_candidate_index"]),
            "selection_group": item.get("selection_group"),
            "target_index": int(item["target_index"]),
            "case_role": target.get("case_role"),
            "schema_repair_role_alias": role_alias,
            "protocol_terminal_label_family": "stress_v1e_zero_or_H15_common_terminal_stable",
        })
        summary = v1d.annotate_physical_prefix(summary)
        branch_episodes.append(summary)
        control_steps_done = int(sum(int(e.get("steps", 0)) for e in branch_episodes))
        progress = {"pid": os.getpid(), "episodes_done": len(branch_episodes), "episodes_expected": EPISODES_EXACT, "control_steps_done": control_steps_done, "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "case", "branch_step", "branch_horizon", "terminal_mode", "steps", "success", "termination", "branch_reached")}, "schema_repair_role_alias": role_alias, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(SMOKE_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in branch_episodes))
    if len(branch_episodes) != EPISODES_EXACT or control_steps > CONTROL_STEP_UPPER:
        raise ContractError("v1e v0b smoke budget violation")
    analysis = v0.analyze_episodes(branch_episodes, inputs["targets"])
    created_dt = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1E_TARGETED_COMMON_PREFIX_SMOKE_V0B_SCHEMA_REPAIR_RUN_%s.json" % created_dt.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created_dt.isoformat(), "reason": "backup v1e v0b schema-repair targeted common-prefix smoke outputs before further simulation/training/refit", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "artifacts": [rel(SMOKE_DIR), rel(STATE_SMOKE), rel(SOURCE), rel(AMENDMENT_JSON), rel(v0.PROTOCOL_JSON), rel(req)]})
    raw = {
        "created_utc": created_dt.isoformat(),
        "started_utc": started,
        "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": f"{NAME}_run",
        "classification": "development_IMPROVED_stress_v1e_targeted_common_prefix_smoke_schema_repair_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "schema_repair": "role_alias_from_case_role_only",
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "backup_proof": backup,
        "protocol": {"parent_json": rel(v0.PROTOCOL_JSON), "parent_json_sha256": sha256(v0.PROTOCOL_JSON), "amendment_json": rel(AMENDMENT_JSON), "amendment_sha256": sha256(AMENDMENT_JSON)},
        "inputs": {"v0b_dryrun_completed": rel(dry_path), "parent_v0_dryrun_completed": rel(v0.DRYRUN_DIR / "completed.json"), "parent_v0_failure_audit": inputs["parent_v0_failure_audit"], "prepare_completed": rel(v0.PREPARE_DONE), "stage1_bank": rel(v0.STAGE1_BANK)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": {"episodes_exact": EPISODES_EXACT, "control_step_upper_bound": CONTROL_STEP_UPPER, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "environment_constructions": len(branch_episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in branch_episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0, "prior_failed_v0_partial_trace_rows_not_counted": inputs["parent_v0_failure_audit"].get("partial_trace_jsonl_rows")},
        "targets": inputs["targets"],
        "schedule": inputs["schedule"],
        "branch_episodes": branch_episodes,
        "analysis": analysis,
        "backup_request_after_run": rel(req),
        "interpretation_limits": ["development diagnostic only", "schema repair after v0 implementation failure", "stress-v1 targeted states selected after development evidence", "not validation/model selection", "not training/refit", "not ORIGINAL SAC", "not final test"],
    }
    write_json(SMOKE_DIR / "raw.json", raw)
    write_smoke_summary(raw)
    STATE_SMOKE.parent.mkdir(parents=True, exist_ok=True)
    STATE_SMOKE.write_text((SMOKE_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER_RUN} -->
## 2026-09-29 vehicle stress-v1e targeted common-prefix smoke v0b schema-repair run

UTC: {created_dt.isoformat()}. Development-only v1e targeted common-prefix smoke completed after one-variable role-alias schema repair: {len(branch_episodes)} episodes, {control_steps} control steps. Robust-positive states={analysis['robust_positive_state_count']} across cases={analysis['robust_positive_cases']}; gate={analysis['smoke_pass_to_selector_or_value_refit_design']}; blocking artifacts={analysis['blocking_artifact_count']}. No validation64-bank or sealed-test access, no training/refit. Artifacts: `{rel(SMOKE_DIR / 'summary.md')}`, `{rel(SMOKE_DIR / 'raw.json')}`, `{rel(SMOKE_DIR / 'completed.json')}`.
""", MARKER_RUN)
    files = [p for p in SMOKE_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, AMENDMENT_JSON, STATE_SMOKE, req, v0.PROTOCOL_JSON, v0.PREPARE_DONE, dry_path, backup_proof]
    write_json(done_path, {"passed": True, "hard_pass": True, "created_utc": created_dt.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_SUPERVISOR_EVENT).total_seconds(), "formal_scientific_evidence": False, "schema_repair": "role_alias_from_case_role_only", "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "backup_request": rel(req), "headline": {"robust_positive_state_count": analysis["robust_positive_state_count"], "negative_or_neutral_state_count": analysis["negative_or_neutral_state_count"], "robust_positive_cases": analysis["robust_positive_cases"], "smoke_pass_to_selector_or_value_refit_design": analysis["smoke_pass_to_selector_or_value_refit_design"], "blocking_artifact_count": analysis["blocking_artifact_count"], "train_or_refit_now": False, "next_action": "if gate passes, freeze compact selector/value-refit design after backup; if fails, do not train/refit and pivot to terminal/modeling or sparse-opportunity diagnosis"}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(done_path), "summary": rel(SMOKE_DIR / "summary.md"), "episodes": len(branch_episodes), "control_steps": control_steps, "robust_positive_state_count": analysis["robust_positive_state_count"], "negative_or_neutral_state_count": analysis["negative_or_neutral_state_count"], "robust_positive_cases": analysis["robust_positive_cases"], "smoke_pass_to_selector_or_value_refit_design": analysis["smoke_pass_to_selector_or_value_refit_design"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-smoke", action="store_true")
    ap.add_argument("--backup-proof", type=Path, default=None)
    ap.add_argument("--i-accept-development-v1e-common-prefix-smoke-v0b-schema-repair", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_v1e_common_prefix_smoke_v0b_schema_repair:
        raise ContractError("explicit --i-accept-development-v1e-common-prefix-smoke-v0b-schema-repair required")
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
        write_json(target / "failure.json", {"failed_utc": now_utc().isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_rollouts": 0 if "--dry-run" in sys.argv else None, "new_control_steps": 0 if "--dry-run" in sys.argv else None, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "next_recovery_hint": "Preserve partial output. If dry-run failed, repair source/provenance only; if smoke failed, audit partial branch outputs before rerun."})
        raise
