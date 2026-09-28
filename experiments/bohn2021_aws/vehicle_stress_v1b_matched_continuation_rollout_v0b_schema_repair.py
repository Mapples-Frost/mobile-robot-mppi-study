#!/usr/bin/env python3
"""Schema-repair wrapper for vehicle stress-v1b matched-continuation rollout v0.

The first v0 dry-run failed before any simulation because it read the stress-v1
Stage1 protocol for ``terminal_grid_readiness_reused_from_v1``.  That field is
intentionally stored in the terminal-source stress-v0 protocol, while the v1
protocol references the terminal source indirectly.  This wrapper preserves the
v0 source and failed dry-run evidence, injects the missing terminal-readiness
object only when reading the v1 Stage1 protocol, and writes v0b-specific output
folders/backup requests.

No validation64 or sealed final-test banks are opened.  Dry-run performs no
simulation.  Rollout still requires a verified post-dry-run external backup.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_stress_v1b_matched_continuation_rollout_v0_runner as base  # noqa:E402

ORIGINAL_V0_SOURCE = Path(base.__file__).resolve()
WRAPPER_SOURCE = Path(__file__).resolve()
STAMP_V0B = "20260928T2220Z_SCHEMA_REPAIR"

# Route all newly produced artifacts away from the failed v0 dry-run directory.
base.STAMP = STAMP_V0B
base.DRYRUN_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_rollout_v0b_dryrun_20260928T2220Z_schema_repair"
base.RUN_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_rollout_v0b_20260928T2220Z_schema_repair"
base.DRYRUN_STATE = ROOT / "research_artifacts/aws_state/vehicle_stress_v1b_matched_continuation_rollout_v0b_dryrun_20260928T2220Z_schema_repair.md"
base.RUN_STATE = ROOT / "research_artifacts/aws_state/vehicle_stress_v1b_matched_continuation_rollout_v0b_run_20260928T2220Z_schema_repair.md"
base.REQUEST_BACKUP_BEFORE_ROLLOUT = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1B_MATCHED_CONTINUATION_ROLLOUT_V0B_20260928T2220Z_SCHEMA_REPAIR.json"
base.MARKER_DRYRUN = "vehicle-stress-v1b-matched-continuation-rollout-v0b-dryrun-20260928T2220Z-schema-repair"
base.MARKER_RUN = "vehicle-stress-v1b-matched-continuation-rollout-v0b-run-20260928T2220Z-schema-repair"

# Make Path(__file__) references inside imported v0 functions identify this
# executable wrapper in backup-request artifact lists and completed hashes.
base.__file__ = str(WRAPPER_SOURCE)

_ORIGINAL_READ_JSON = base.read_json
_ORIGINAL_SOURCE_HASHES = base.source_hashes
_TERMINAL_SOURCE_PROTOCOL = base.stage1_runner.TERMINAL_SOURCE_PROTOCOL


def _read_json_schema_repair(path: Path) -> Any:
    obj = _ORIGINAL_READ_JSON(path)
    try:
        resolved = Path(path).resolve()
    except Exception:
        resolved = Path(str(path))
    if resolved == base.STAGE1_PROTOCOL_JSON.resolve() and isinstance(obj, dict) and "terminal_grid_readiness_reused_from_v1" not in obj:
        terminal_protocol = _ORIGINAL_READ_JSON(_TERMINAL_SOURCE_PROTOCOL)
        terminal_grid = terminal_protocol.get("terminal_grid_readiness_reused_from_v1")
        if not terminal_grid:
            raise base.ContractError("terminal-source protocol lacks terminal_grid_readiness_reused_from_v1: %s" % base.rel(_TERMINAL_SOURCE_PROTOCOL))
        repaired = dict(obj)
        repaired["terminal_grid_readiness_reused_from_v1"] = terminal_grid
        repaired["_schema_repair_terminal_grid_source"] = base.rel(_TERMINAL_SOURCE_PROTOCOL)
        repaired["_schema_repair_reason"] = "stress-v1 protocol references terminal source indirectly; v0 runner expected inline terminal_grid_readiness_reused_from_v1"
        return repaired
    return obj


def _source_hashes_schema_repair(extra: Sequence[Path] = ()):
    merged = dict(_ORIGINAL_SOURCE_HASHES(extra=extra))
    for p in (WRAPPER_SOURCE, ORIGINAL_V0_SOURCE, _TERMINAL_SOURCE_PROTOCOL):
        if p.exists():
            merged[base.rel(p)] = base.sha256(p)
    return merged


base.read_json = _read_json_schema_repair
base.source_hashes = _source_hashes_schema_repair


def _annotate_completed(mode: str) -> None:
    completed_path = (base.DRYRUN_DIR if mode == "dry" else base.RUN_DIR) / "completed.json"
    if not completed_path.exists():
        return
    obj = _ORIGINAL_READ_JSON(completed_path)
    obj["schema_repair_wrapper"] = {
        "wrapper_source": base.rel(WRAPPER_SOURCE),
        "wrapper_sha256": base.sha256(WRAPPER_SOURCE),
        "original_v0_source": base.rel(ORIGINAL_V0_SOURCE),
        "original_v0_sha256": base.sha256(ORIGINAL_V0_SOURCE),
        "terminal_source_protocol": base.rel(_TERMINAL_SOURCE_PROTOCOL),
        "terminal_source_protocol_sha256": base.sha256(_TERMINAL_SOURCE_PROTOCOL),
        "repair": "Use terminal_grid_readiness_reused_from_v1 from the terminal-source stress-v0 protocol when v1 Stage1 protocol lacks the inline field.",
    }
    hashes = dict(obj.get("hashes") or {})
    hashes[base.rel(WRAPPER_SOURCE)] = base.sha256(WRAPPER_SOURCE)
    hashes[base.rel(ORIGINAL_V0_SOURCE)] = base.sha256(ORIGINAL_V0_SOURCE)
    hashes[base.rel(_TERMINAL_SOURCE_PROTOCOL)] = base.sha256(_TERMINAL_SOURCE_PROTOCOL)
    obj["hashes"] = hashes
    base.write_json(completed_path, obj)


def main(argv=None) -> int:
    mode = "dry" if "--dry-run" in (argv if argv is not None else sys.argv[1:]) else "run"
    rc = base.main(argv)
    if rc == 0:
        _annotate_completed(mode)
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = base.DRYRUN_DIR if "--dry-run" in sys.argv else base.RUN_DIR
        target.mkdir(parents=True, exist_ok=True)
        base.write_json(target / "failure.json", {
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
            "schema_repair_wrapper": base.rel(WRAPPER_SOURCE),
            "original_v0_source": base.rel(ORIGINAL_V0_SOURCE),
            "next_recovery_hint": "Preserve partial output. If dry-run failed, inspect failure.json and repair only the wrapper/source integration; if rollout failed, audit partial episodes before any rerun.",
        })
        raise
