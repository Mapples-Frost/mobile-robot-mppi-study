#!/usr/bin/env python3
"""Schema/runtime repair wrapper for vehicle stress-v1c label-density probe.

The v0 dry-run completed successfully and is preserved, but static inspection of
the v0 rollout path found an impending schedule-schema mismatch: the reusable
`vehicle_stage2_terminal_objective_smoke_v0.run_one` helper expects each item to
contain a `role` field, while the v0 v1c schedule only emitted the metadata
`selection_group`.  This wrapper makes the minimal no-science-change repair:

* route dry-run/smoke/full outputs to distinct v0b directories;
* add `role=selection_group` to every generated schedule item before simulation;
* repair the human-readable terminal receipt for `h15_common_terminal` to point
  to the H15 terminal source instead of the helper's fallback zero-like label;
* require the same development-only acceptance and backup gates as v0.

No validation64 bank or sealed test is opened; no training/refit/gradient update
is performed.  The frozen v1c protocol remains unchanged.
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
STAMP_V0B = "20260929T0045Z_schema_repair"

# Preserve v0 dry-run artifacts and route all v0b outputs separately.
base.DRYRUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0b_dryrun_{STAMP_V0B}"
base.SMOKE_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0b_smoke_{STAMP_V0B}"
base.FULL_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0b_full_{STAMP_V0B}"
base.BANK_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0b_bank_{STAMP_V0B}"
base.BANK_PATH = base.BANK_DIR / "vehicle_stress_v1c_terminal_stable_label_density_probe_v0b_bank.json"
base.BANK_COMPLETED = base.BANK_DIR / "completed.json"
base.STATE_DRYRUN = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1c_terminal_stable_label_density_probe_v0b_dryrun_{STAMP_V0B}.md"
base.STATE_SMOKE = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1c_terminal_stable_label_density_probe_v0b_smoke_{STAMP_V0B}.md"
base.STATE_FULL = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1c_terminal_stable_label_density_probe_v0b_full_{STAMP_V0B}.md"
base.REQUEST_BACKUP_BEFORE_SMOKE = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1C_TERMINAL_STABLE_LABEL_DENSITY_PROBE_V0B_SMOKE_{STAMP_V0B}.json"
base.MARKER_DRYRUN = f"vehicle-stress-v1c-terminal-stable-label-density-probe-v0b-dryrun-{STAMP_V0B}"
base.MARKER_SMOKE = f"vehicle-stress-v1c-terminal-stable-label-density-probe-v0b-smoke-{STAMP_V0B}"
base.MARKER_FULL = f"vehicle-stress-v1c-terminal-stable-label-density-probe-v0b-full-{STAMP_V0B}"

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
    if not getattr(base_smoke, "_v1c_v0b_run_one_patched", False):
        original_run_one = base_smoke.run_one

        def run_one_v1c_v0b(item: Any, case: Any, terminals: Any, terminal_receipts: Any) -> Any:
            item = dict(item)
            item.setdefault("role", item.get("selection_group", "fresh_v1c_terminal_stable_state"))
            summary = original_run_one(item, case, terminals, terminal_receipts)
            if item.get("terminal_mode") == "h15_common_terminal":
                summary["terminal_receipt_effective"] = terminal_receipts.get("15")
                summary["effective_branch_terminal_label"] = "H15_common"
            summary.setdefault("state_role", item.get("role"))
            return summary

        base_smoke.run_one = run_one_v1c_v0b
        base_smoke._v1c_v0b_run_one_patched = True
    return base_smoke, stage1_runner, fixed_base


base.import_legacy_modules = _import_legacy_modules_with_receipt_repair


def _annotate_json(path: Path, run_kind: str) -> None:
    if not path.exists():
        return
    obj = _read_json(path)
    info = {
        "wrapper_source": _rel(WRAPPER_SOURCE),
        "wrapper_sha256": base.sha256(WRAPPER_SOURCE),
        "original_v0_source": _rel(ORIGINAL_SOURCE),
        "original_v0_sha256": base.sha256(ORIGINAL_SOURCE) if ORIGINAL_SOURCE.exists() else None,
        "preserved_v0_dryrun_completed": "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0_dryrun_20260929T0040Z/completed.json",
        "preserved_v0_dryrun_source_sha256": "264a7a5b07932b0f3eecf4614921ecca6b72824aa7b9116e78e984314f8b8086",
        "repair": "Add role=selection_group to schedule items required by the reused terminal/objective branch runner, and repair h15_common_terminal terminal receipt metadata to H15.",
        "run_kind": run_kind,
        "validation64_bank_opened": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
    }
    obj["v0b_schema_repair_wrapper"] = info
    hashes = dict(obj.get("hashes") or {})
    for p in (WRAPPER_SOURCE, ORIGINAL_SOURCE, ROOT / info["preserved_v0_dryrun_completed"]):
        if p.exists():
            hashes[_rel(p)] = base.sha256(p)
    obj["hashes"] = hashes
    _write_json(path, obj)


def _annotate_outputs(run_kind: str) -> None:
    out_dir = base.DRYRUN_DIR if run_kind == "dryrun" else base.SMOKE_DIR if run_kind == "smoke" else base.FULL_DIR
    note = out_dir / "v0b_schema_repair_wrapper_note.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# v0b schema-repair wrapper note\n\n"
        f"Run kind: `{run_kind}`. This wrapper preserves the completed v0 dry-run and makes one engineering repair before any v1c simulation: "
        "schedule items now include `role=selection_group` for compatibility with the reused branch-runner helper. "
        "It also corrects the h15_common_terminal receipt annotation to the H15 terminal source. No scientific protocol, case selection rule, label rule, horizon grid, terminal target modes, validation/test access, or training/refit budget is changed.\n",
        encoding="utf-8",
    )
    for name in ("raw.json", "completed.json"):
        _annotate_json(out_dir / name, run_kind)
    completed = out_dir / "completed.json"
    if completed.exists():
        obj = _read_json(completed)
        hashes = dict(obj.get("hashes") or {})
        hashes[_rel(note)] = base.sha256(note)
        obj["hashes"] = hashes
        _write_json(completed, obj)


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
        target = base.DRYRUN_DIR if "--dry-run" in sys.argv else base.SMOKE_DIR if "--run-smoke" in sys.argv else base.FULL_DIR if "--run-full" in sys.argv else ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0b_failure_unknown_{STAMP_V0B}"
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
            "v0b_schema_repair_wrapper": {
                "wrapper_source": _rel(WRAPPER_SOURCE),
                "wrapper_sha256": base.sha256(WRAPPER_SOURCE),
                "original_v0_source": _rel(ORIGINAL_SOURCE),
                "original_v0_sha256": base.sha256(ORIGINAL_SOURCE) if ORIGINAL_SOURCE.exists() else None,
                "repair": "Add role=selection_group to schedule items and repair h15_common_terminal receipt metadata.",
            },
            "next_recovery_hint": "Preserve partial output. If dry-run failed, repair wrapper/source logic only; if simulation failed, audit partial episodes before any rerun.",
        })
        raise
