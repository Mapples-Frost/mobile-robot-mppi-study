#!/usr/bin/env python3
"""Schema-repair wrapper for the true variable-H case5 smoke.

The first v0 run after backup failed before any branch rollout because the
frozen protocol intentionally stores branch state in `targets`, while the v0
runtime passed only a compact `schedule` item into reset_env_to_branch().  This
v0b wrapper does not change the scientific protocol, target states, horizon
arms, terminal mode, or budgets.  It patches the runtime call so each schedule
item is merged with its matching target metadata before branch reset.

Run contract: development-only IMPROVED implementation smoke, no validation64
bank, no sealed test, no training/refit/candidate generation.  A verified
external backup *after this v0b source and the v0 failure artifacts* is required
before --run-smoke-v0b will execute simulations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_case5_smoke_v0_runner as v0  # noqa:E402

NAME_V0B = "vehicle_true_variable_horizon_case5_smoke_v0b_schema_repair"
STAMP_V0B = "20260929T0645Z"
SOURCE_V0B = Path(__file__).resolve()
SMOKE_DIR_V0B = ROOT / f"research_artifacts/aws_diagnostics/{NAME_V0B}_run_{STAMP_V0B}"
STATE_SMOKE_V0B = ROOT / f"research_artifacts/aws_state/{NAME_V0B}_run_{STAMP_V0B}.md"
MARKER_SMOKE_V0B = f"vehicle-true-variable-horizon-case5-smoke-v0b-schema-repair-run-{STAMP_V0B}"
V0_FAILURE = v0.SMOKE_DIR / "failure.json"
REQUEST_BACKUP_BEFORE_V0B = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_TRUE_VARIABLE_HORIZON_CASE5_SMOKE_V0B_SCHEMA_REPAIR_RUN_{STAMP_V0B}.json"


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(v0.clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify_v0_failure_and_protocol() -> Dict[str, Any]:
    dry_done = v0.completed_ok(v0.DRYRUN_DIR / "completed.json", check_hashes=True)
    if not V0_FAILURE.exists():
        raise ContractError(f"expected v0 failure marker is missing: {rel(V0_FAILURE)}")
    failure = v0.read_json(V0_FAILURE)
    if "branch_previous_state" not in str(failure.get("exception")) + str(failure.get("traceback")):
        raise ContractError("v0 failure is not the expected missing branch_previous_state schema bug")
    if failure.get("validation64_bank_opened") is not False or failure.get("sealed_test_accessed") is not False:
        raise ContractError("v0 failure access flags are invalid")
    protocol = v0.read_json(v0.PROTOCOL_JSON)
    targets = protocol.get("targets") or []
    schedule = protocol.get("schedule") or []
    target_by_state = {str(t.get("state_id")): t for t in targets}
    if len(target_by_state) != len(v0.EXPECTED_TARGET_STATE_IDS):
        raise ContractError("unexpected target count in frozen v0 protocol")
    for sid in v0.EXPECTED_TARGET_STATE_IDS:
        t = target_by_state.get(sid)
        if not isinstance(t, Mapping) or "branch_previous_state" not in t:
            raise ContractError(f"target lacks branch_previous_state: {sid}")
    if len(schedule) != v0.EPISODES_EXACT:
        raise ContractError("unexpected schedule length in frozen v0 protocol")
    if any("branch_previous_state" in item for item in schedule):
        raise ContractError("schedule unexpectedly already contains branch_previous_state; v0b patch assumptions changed")
    return {"dry_done": dry_done, "failure": failure, "protocol": protocol, "target_by_state": target_by_state}


def merged_item(item: Mapping[str, Any], target_by_state: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    sid = str(item.get("state_id"))
    if sid not in target_by_state:
        raise ContractError(f"schedule item has no matching target: {sid}")
    merged = dict(target_by_state[sid])
    merged.update(dict(item))
    if "branch_previous_state" not in merged:
        raise ContractError(f"merged schedule item still lacks branch_previous_state: {sid}")
    return merged


def install_patch(target_by_state: Mapping[str, Mapping[str, Any]]) -> None:
    original_run_true_h_episode = v0.run_true_h_episode

    def run_true_h_episode_v0b(item: Mapping[str, Any], case: Mapping[str, Any], h15_terminal: Any) -> Dict[str, Any]:
        return original_run_true_h_episode(merged_item(item, target_by_state), case, h15_terminal)

    v0.run_true_h_episode = run_true_h_episode_v0b
    v0.NAME = NAME_V0B
    v0.SOURCE = SOURCE_V0B
    v0.SMOKE_DIR = SMOKE_DIR_V0B
    v0.STATE_SMOKE = STATE_SMOKE_V0B
    v0.MARKER_SMOKE = MARKER_SMOKE_V0B


def run_smoke_v0b(backup_proof: Path) -> int:
    info = verify_v0_failure_and_protocol()
    min_time_candidates = [
        v0.file_mtime_utc(SOURCE_V0B),
        v0.file_mtime_utc(v0.PROTOCOL_JSON),
        v0.file_mtime_utc(V0_FAILURE),
        v0.parse_time(info["dry_done"].get("created_utc")),
    ]
    min_time = max(t for t in min_time_candidates if t is not None)
    # Enforce backup coverage of the v0b repair before any simulation.  The
    # underlying v0.run_smoke also verifies the proof after globals are patched.
    v0.v1d.verify_backup_proof(backup_proof, min_time, NAME_V0B)
    install_patch(info["target_by_state"])
    return v0.run_smoke(backup_proof)


def write_backup_request() -> None:
    write_json(REQUEST_BACKUP_BEFORE_V0B, {
        "requested_utc": now_utc().isoformat(),
        "reason": "backup v0 failed true-variable-H smoke evidence and v0b schema-repair source before any rerun simulation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "planned_development_branch_episodes_exact": v0.EPISODES_EXACT,
        "planned_development_control_step_upper_bound": v0.CONTROL_STEP_UPPER,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [
            rel(SOURCE_V0B),
            rel(V0_FAILURE),
            rel(v0.SMOKE_DIR),
            rel(v0.PROTOCOL_JSON),
            rel(v0.DRYRUN_DIR),
            rel(REQUEST_BACKUP_BEFORE_V0B),
        ],
    })


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-backup-request", action="store_true", help="metadata-only request; no simulation")
    ap.add_argument("--run-smoke-v0b", action="store_true")
    ap.add_argument("--backup-proof", type=Path, default=None)
    ap.add_argument("--i-accept-development-true-variable-horizon-case5-smoke-v0b", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_true_variable_horizon_case5_smoke_v0b:
        raise ContractError("explicit --i-accept-development-true-variable-horizon-case5-smoke-v0b required")
    if bool(args.write_backup_request) == bool(args.run_smoke_v0b):
        raise ContractError("exactly one of --write-backup-request or --run-smoke-v0b is required")
    if args.write_backup_request:
        verify_v0_failure_and_protocol()
        write_backup_request()
        print(json.dumps({"backup_request": rel(REQUEST_BACKUP_BEFORE_V0B), "new_rollouts": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if args.backup_proof is None:
        raise ContractError("--run-smoke-v0b requires --backup-proof")
    return run_smoke_v0b(args.backup_proof)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        SMOKE_DIR_V0B.mkdir(parents=True, exist_ok=True)
        write_json(SMOKE_DIR_V0B / "failure.json", {
            "failed_utc": now_utc().isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
            "candidate_pool_resets": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
            "next_recovery_hint": "Preserve partial output; audit before rerun.  If failure is before branch steps, repair only the implementation wrapper and re-backup before simulation.",
        })
        raise
