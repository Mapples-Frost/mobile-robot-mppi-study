#!/usr/bin/env python3
"""Hash-consistent v0c wrapper for vehicle stress-v1c label-density probe.

This supersedes only the engineering wrapper/marker handling of v0b.  The v0b
smoke failed before any simulation because the v0b wrapper annotated raw.json
after base completed.json had already recorded raw.json's pre-annotation hash.
The scientific protocol, case-selection rules, horizon grid, terminal modes,
label gates and validation/test/training access rules are unchanged.

Repairs relative to v0b:
  * route outputs to distinct v0c directories;
  * keep the v0b schedule-schema repair (role=selection_group);
  * keep the h15_common_terminal receipt annotation repair;
  * after wrapper annotations, rebuild completed.json hashes from final files so
    later completed_passed(..., check_hashes=True) verifies the final artifact
    bytes instead of a transient pre-annotation raw.json.

No validation64 bank or sealed test is opened; no training/refit/gradient update
is performed.  Dry-run remains no-simulation.  Smoke/full still require a
verified external backup proof newer than the v0c dry-run and source.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_stress_v1c_terminal_stable_label_density_probe_v0_runner as base  # noqa:E402

WRAPPER_SOURCE = Path(__file__).resolve()
ORIGINAL_SOURCE = Path(base.__file__).resolve()
PREVIOUS_V0B_WRAPPER = ROOT / "experiments/bohn2021_aws/vehicle_stress_v1c_terminal_stable_label_density_probe_v0b_schema_repair.py"
STAMP_V0C = "20260929T0055Z_hash_repair"

# Preserve v0/v0b artifacts and route all v0c outputs separately.
base.DRYRUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_dryrun_{STAMP_V0C}"
base.SMOKE_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_smoke_{STAMP_V0C}"
base.FULL_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_full_{STAMP_V0C}"
base.BANK_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_bank_{STAMP_V0C}"
base.BANK_PATH = base.BANK_DIR / "vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_bank.json"
base.BANK_COMPLETED = base.BANK_DIR / "completed.json"
base.STATE_DRYRUN = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_dryrun_{STAMP_V0C}.md"
base.STATE_SMOKE = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_smoke_{STAMP_V0C}.md"
base.STATE_FULL = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_full_{STAMP_V0C}.md"
base.REQUEST_BACKUP_BEFORE_SMOKE = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1C_TERMINAL_STABLE_LABEL_DENSITY_PROBE_V0C_SMOKE_{STAMP_V0C}.json"
base.MARKER_DRYRUN = f"vehicle-stress-v1c-terminal-stable-label-density-probe-v0c-dryrun-{STAMP_V0C}"
base.MARKER_SMOKE = f"vehicle-stress-v1c-terminal-stable-label-density-probe-v0c-smoke-{STAMP_V0C}"
base.MARKER_FULL = f"vehicle-stress-v1c-terminal-stable-label-density-probe-v0c-full-{STAMP_V0C}"

_ORIGINAL_BUILD_SCHEDULE = base.build_schedule
_ORIGINAL_IMPORT_LEGACY_MODULES = base.import_legacy_modules
_ORIGINAL_SOURCE_MTIME_UTC = base.source_mtime_utc
base.__file__ = str(WRAPPER_SOURCE)


def _rel(path: Path) -> str:
    return base.rel(path)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _write_json(path: Path, value: Any) -> None:
    base.write_json(path, value)


def _source_mtime_utc() -> dt.datetime:
    times = [dt.datetime.fromtimestamp(WRAPPER_SOURCE.stat().st_mtime, dt.timezone.utc)]
    if ORIGINAL_SOURCE.exists():
        times.append(dt.datetime.fromtimestamp(ORIGINAL_SOURCE.stat().st_mtime, dt.timezone.utc))
    if PREVIOUS_V0B_WRAPPER.exists():
        times.append(dt.datetime.fromtimestamp(PREVIOUS_V0B_WRAPPER.stat().st_mtime, dt.timezone.utc))
    try:
        times.append(_ORIGINAL_SOURCE_MTIME_UTC())
    except Exception:
        pass
    return max(times)


base.source_mtime_utc = _source_mtime_utc


def _build_schedule_with_role(selection: Any, run_kind: str) -> Any:
    schedule = _ORIGINAL_BUILD_SCHEDULE(selection, run_kind)
    for item in schedule:
        item.setdefault("role", item.get("selection_group", "fresh_v1c_terminal_stable_state"))
    return schedule


base.build_schedule = _build_schedule_with_role


def _import_legacy_modules_with_receipt_repair() -> Any:
    base_smoke, stage1_runner, fixed_base = _ORIGINAL_IMPORT_LEGACY_MODULES()
    if not getattr(base_smoke, "_v1c_v0c_run_one_patched", False):
        original_run_one = base_smoke.run_one

        def run_one_v1c_v0c(item: Any, case: Any, terminals: Any, terminal_receipts: Any) -> Any:
            item = dict(item)
            item.setdefault("role", item.get("selection_group", "fresh_v1c_terminal_stable_state"))
            summary = original_run_one(item, case, terminals, terminal_receipts)
            if item.get("terminal_mode") == "h15_common_terminal":
                summary["terminal_receipt_effective"] = terminal_receipts.get("15")
                summary["effective_branch_terminal_label"] = "H15_common"
            summary.setdefault("state_role", item.get("role"))
            return summary

        base_smoke.run_one = run_one_v1c_v0c
        base_smoke._v1c_v0c_run_one_patched = True
    return base_smoke, stage1_runner, fixed_base


base.import_legacy_modules = _import_legacy_modules_with_receipt_repair


def _wrapper_info(run_kind: str) -> dict:
    info = {
        "wrapper_source": _rel(WRAPPER_SOURCE),
        "wrapper_sha256": base.sha256(WRAPPER_SOURCE),
        "original_v0_source": _rel(ORIGINAL_SOURCE),
        "original_v0_sha256": base.sha256(ORIGINAL_SOURCE) if ORIGINAL_SOURCE.exists() else None,
        "previous_v0b_wrapper": _rel(PREVIOUS_V0B_WRAPPER),
        "previous_v0b_wrapper_sha256": base.sha256(PREVIOUS_V0B_WRAPPER) if PREVIOUS_V0B_WRAPPER.exists() else None,
        "preserved_v0_dryrun_completed": "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0_dryrun_20260929T0040Z/completed.json",
        "preserved_v0b_dryrun_completed": "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0b_dryrun_20260929T0045Z_schema_repair/completed.json",
        "preserved_v0b_smoke_failure": "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0b_smoke_20260929T0045Z_schema_repair/failure.json",
        "repair": "v0c preserves v0b schedule/terminal annotation fixes and additionally recomputes completed.json hashes after wrapper annotations so final artifacts pass hash verification.",
        "run_kind": run_kind,
        "validation64_bank_opened": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
    }
    return info


def _annotate_json(path: Path, run_kind: str) -> None:
    if not path.exists():
        return
    obj = _read_json(path)
    obj["v0c_hash_repair_wrapper"] = _wrapper_info(run_kind)
    _write_json(path, obj)


def _finalize_completed_hashes(out_dir: Path, run_kind: str) -> None:
    completed = out_dir / "completed.json"
    if not completed.exists():
        return
    obj = _read_json(completed)
    # Include all final output files except completed.json itself, plus sources and
    # critical frozen inputs.  This replaces transient pre-annotation hashes.
    files = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "completed.json"]
    extra = [WRAPPER_SOURCE, ORIGINAL_SOURCE, PREVIOUS_V0B_WRAPPER, base.PROTOCOL_JSON, base.V0B_DONE, base.V0B_RAW, base.V0B_SUMMARY]
    for p in extra:
        if p.exists():
            files.append(p)
    # Retain referenced external backup proof hash for simulation modes when
    # base.completed already included it.
    for value in list((obj.get("hashes") or {}).keys()):
        p = ROOT / value
        if p.exists():
            files.append(p)
    obj["hashes"] = {_rel(p): base.sha256(p) for p in sorted(set(files)) if p.exists()}
    obj["v0c_hash_repair_wrapper"] = _wrapper_info(run_kind)
    _write_json(completed, obj)
    # Verify immediately using the base checker that blocked v0b smoke.
    base.completed_passed(completed, check_hashes=True)


def _annotate_outputs(run_kind: str) -> None:
    out_dir = base.DRYRUN_DIR if run_kind == "dryrun" else base.SMOKE_DIR if run_kind == "smoke" else base.FULL_DIR
    note = out_dir / "v0c_hash_repair_wrapper_note.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# v0c hash-repair wrapper note\n\n"
        f"Run kind: `{run_kind}`. This wrapper preserves the frozen v1c protocol and v0b schedule/terminal annotation repair. "
        "It additionally rebuilds completed.json hashes after raw/summary annotation so future check_hashes=True validation checks final artifact bytes. "
        "No case selection rule, label rule, horizon grid, terminal mode, validation/test access, or training/refit budget is changed.\n",
        encoding="utf-8",
    )
    for name in ("raw.json", "completed.json"):
        _annotate_json(out_dir / name, run_kind)
    _finalize_completed_hashes(out_dir, run_kind)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    run_kind = "dryrun" if "--dry-run" in args else "smoke" if "--run-smoke" in args else "full" if "--run-full" in args else "unknown"
    rc = base.main(args)
    if rc == 0 and run_kind in ("dryrun", "smoke", "full"):
        _annotate_outputs(run_kind)
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = base.DRYRUN_DIR if "--dry-run" in sys.argv else base.SMOKE_DIR if "--run-smoke" in sys.argv else base.FULL_DIR if "--run-full" in sys.argv else ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_failure_unknown_{STAMP_V0C}"
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "failure.json", {
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
            "v0c_hash_repair_wrapper": _wrapper_info("dryrun" if "--dry-run" in sys.argv else "smoke" if "--run-smoke" in sys.argv else "full" if "--run-full" in sys.argv else "unknown"),
            "next_recovery_hint": "Preserve partial output. If dry-run failed, repair wrapper/source logic only; if simulation failed, audit partial episodes before any rerun.",
        })
        raise
