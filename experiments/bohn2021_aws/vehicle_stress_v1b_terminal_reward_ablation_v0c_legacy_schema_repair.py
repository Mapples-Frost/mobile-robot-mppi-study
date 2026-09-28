#!/usr/bin/env python3
"""Legacy + schema-repair retry for vehicle stress-v1b terminal/reward ablation.

The frozen v0 terminal/reward-source ablation protocol remains unchanged.  Two
failed pre-rollout attempts are preserved:

* v0 smoke under the modern interpreter failed before simulation because TF1 is
  available only in the legacy runtime.
* v0b legacy retry passed the TF1 preflight but failed before simulation because
  the stress-v1 Stage1 protocol stores terminal readiness indirectly; the runner
  expected an inline ``terminal_grid_readiness_reused_from_v1`` field.

This v0c wrapper makes the smallest engineering repair needed to execute the
already-frozen development smoke: it routes outputs to distinct v0c directories,
uses the legacy interpreter, and injects the terminal-grid readiness object from
the registered terminal-source protocol only when reading the stress-v1 Stage1
protocol that lacks the inline field.  It performs no dry-run/protocol change,
opens no validation64 or sealed-test bank, and performs no training/refit or
gradient updates.  A verified external backup proof postdating this wrapper is
required by the inherited source-mtime gate before any rollout can execute.
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

import vehicle_stress_v1b_terminal_reward_ablation_v0 as base  # noqa:E402

WRAPPER_SOURCE = Path(__file__).resolve()
ORIGINAL_V0_SOURCE = Path(base.__file__).resolve()
FAILED_MODERN_SMOKE_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_smoke_20260928T2320Z"
FAILED_V0B_LEGACY_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0b_smoke_20260928T2335Z_legacy_retry"
STAMP_V0C = "20260928T2340Z_legacy_schema_repair"

# Route only simulation outputs away from preserved v0/v0b failed pre-rollout
# directories.  The frozen dry-run/protocol paths remain unchanged and are
# verified by the imported base module.
base.SMOKE_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_smoke_{STAMP_V0C}"
base.FULL_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_full_{STAMP_V0C}"
base.STATE_SMOKE = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1b_terminal_reward_ablation_v0c_smoke_{STAMP_V0C}.md"
base.STATE_FULL = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1b_terminal_reward_ablation_v0c_full_{STAMP_V0C}.md"
base.MARKER_RUN = f"vehicle-stress-v1b-terminal-reward-ablation-v0c-legacy-schema-repair-run-{STAMP_V0C}"

# Make Path(__file__) references inside base point at this wrapper for backup
# min-time checks, backup requests and completed hashes.  Original source hashes
# are added explicitly in annotations.
base.__file__ = str(WRAPPER_SOURCE)

_ORIGINAL_READ_JSON = base.read_json
_ORIGINAL_SOURCE_MTIME_UTC = base.source_mtime_utc
_TERMINAL_SOURCE_PROTOCOL = base.stage1_runner.TERMINAL_SOURCE_PROTOCOL


def _source_mtime_utc_with_wrapper() -> dt.datetime:
    original = dt.datetime.fromtimestamp(ORIGINAL_V0_SOURCE.stat().st_mtime, dt.timezone.utc)
    wrapper = dt.datetime.fromtimestamp(WRAPPER_SOURCE.stat().st_mtime, dt.timezone.utc)
    try:
        base_original = _ORIGINAL_SOURCE_MTIME_UTC()
        if base_original > original:
            original = base_original
    except Exception:
        pass
    return max(original, wrapper)


base.source_mtime_utc = _source_mtime_utc_with_wrapper


def _rel(path: Path) -> str:
    return base.rel(path)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _write_json(path: Path, value: Any) -> None:
    base.write_json(path, value)


def _terminal_grid_readiness() -> Any:
    terminal_protocol = _ORIGINAL_READ_JSON(_TERMINAL_SOURCE_PROTOCOL)
    terminal_grid = terminal_protocol.get("terminal_grid_readiness_reused_from_v1")
    if not terminal_grid:
        raise base.ContractError(
            "terminal-source protocol lacks terminal_grid_readiness_reused_from_v1: %s"
            % _rel(_TERMINAL_SOURCE_PROTOCOL)
        )
    return terminal_grid


def _read_json_schema_repair(path: Path) -> Any:
    obj = _ORIGINAL_READ_JSON(path)
    try:
        resolved = Path(path).resolve()
    except Exception:
        resolved = Path(str(path))
    if (
        resolved == base.STAGE1_PROTOCOL.resolve()
        and isinstance(obj, dict)
        and "terminal_grid_readiness_reused_from_v1" not in obj
    ):
        repaired = dict(obj)
        repaired["terminal_grid_readiness_reused_from_v1"] = _terminal_grid_readiness()
        repaired["_schema_repair_terminal_grid_source"] = _rel(_TERMINAL_SOURCE_PROTOCOL)
        repaired["_schema_repair_reason"] = (
            "stress-v1 Stage1 protocol references terminal readiness indirectly; "
            "terminal/reward ablation v0 expected inline terminal_grid_readiness_reused_from_v1"
        )
        return repaired
    return obj


base.read_json = _read_json_schema_repair


def _annotate_json(path: Path, run_kind: str) -> None:
    if not path.exists():
        return
    obj = _read_json(path)
    wrapper_info = {
        "wrapper_source": _rel(WRAPPER_SOURCE),
        "wrapper_sha256": base.sha256(WRAPPER_SOURCE),
        "original_v0_source": _rel(ORIGINAL_V0_SOURCE),
        "original_v0_sha256": base.sha256(ORIGINAL_V0_SOURCE),
        "terminal_source_protocol": _rel(_TERMINAL_SOURCE_PROTOCOL),
        "terminal_source_protocol_sha256": base.sha256(_TERMINAL_SOURCE_PROTOCOL),
        "preserved_failed_modern_preflight_dir": _rel(FAILED_MODERN_SMOKE_DIR),
        "preserved_failed_modern_preflight_failure_json": _rel(FAILED_MODERN_SMOKE_DIR / "failure.json"),
        "preserved_failed_v0b_legacy_schema_dir": _rel(FAILED_V0B_LEGACY_DIR),
        "preserved_failed_v0b_legacy_schema_failure_json": _rel(FAILED_V0B_LEGACY_DIR / "failure.json"),
        "repair": "Use terminal_grid_readiness_reused_from_v1 from the terminal-source stress-v0 protocol when the stress-v1 Stage1 protocol lacks the inline field.",
        "run_kind": run_kind,
        "output_dir": _rel(base.SMOKE_DIR if run_kind == "smoke" else base.FULL_DIR),
        "requires_interpreter": "legacy",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
    }
    obj["legacy_schema_repair_wrapper"] = wrapper_info
    hashes = dict(obj.get("hashes") or {})
    for p in (
        WRAPPER_SOURCE,
        ORIGINAL_V0_SOURCE,
        _TERMINAL_SOURCE_PROTOCOL,
        FAILED_MODERN_SMOKE_DIR / "failure.json",
        FAILED_MODERN_SMOKE_DIR / "runtime_preflight.json",
        FAILED_MODERN_SMOKE_DIR / "run_started.json",
        FAILED_V0B_LEGACY_DIR / "failure.json",
        FAILED_V0B_LEGACY_DIR / "runtime_preflight.json",
        FAILED_V0B_LEGACY_DIR / "run_started.json",
    ):
        if p.exists():
            hashes[_rel(p)] = base.sha256(p)
    obj["hashes"] = hashes
    _write_json(path, obj)


def _annotate_outputs(run_kind: str) -> None:
    out_dir = base.SMOKE_DIR if run_kind == "smoke" else base.FULL_DIR
    note_path = out_dir / "legacy_schema_repair_wrapper_note.md"
    note_path.write_text(
        "# v0c legacy schema-repair wrapper note\n\n"
        f"Run kind: `{run_kind}`. This wrapper preserves failed pre-rollout attempts at "
        f"`{_rel(FAILED_MODERN_SMOKE_DIR)}` and `{_rel(FAILED_V0B_LEGACY_DIR)}`, then executes the unchanged "
        "frozen terminal/reward-source ablation protocol under the legacy/TF1 interpreter. The only schema repair is "
        f"loading `terminal_grid_readiness_reused_from_v1` from `{_rel(_TERMINAL_SOURCE_PROTOCOL)}` when the stress-v1 "
        "Stage1 protocol lacks the inline field. No validation64 bank, sealed test, training/refit or gradient update is used.\n",
        encoding="utf-8",
    )
    _annotate_json(out_dir / "raw.json", run_kind)
    _annotate_json(out_dir / "completed.json", run_kind)
    # Add the note hash after it exists.
    completed = out_dir / "completed.json"
    if completed.exists():
        obj = _read_json(completed)
        hashes = dict(obj.get("hashes") or {})
        hashes[_rel(note_path)] = base.sha256(note_path)
        obj["hashes"] = hashes
        _write_json(completed, obj)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--dry-run" in args:
        raise base.ContractError(
            "v0c legacy schema repair does not create a new scientific dry-run; "
            "use the already frozen v0 dry-run/protocol and run --run-smoke or --run-full only."
        )
    run_kind = "smoke" if "--run-smoke" in args else "full" if "--run-full" in args else "unknown"
    rc = base.main(args)
    if rc == 0 and run_kind in ("smoke", "full"):
        _annotate_outputs(run_kind)
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = (
            base.SMOKE_DIR
            if "--run-smoke" in sys.argv
            else base.FULL_DIR
            if "--run-full" in sys.argv
            else ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_failure_unknown_mode_20260928T2340Z_legacy_schema_repair"
        )
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "legacy_schema_repair_wrapper": {
                "wrapper_source": _rel(WRAPPER_SOURCE),
                "wrapper_sha256": base.sha256(WRAPPER_SOURCE),
                "original_v0_source": _rel(ORIGINAL_V0_SOURCE),
                "original_v0_sha256": base.sha256(ORIGINAL_V0_SOURCE),
                "terminal_source_protocol": _rel(_TERMINAL_SOURCE_PROTOCOL),
                "terminal_source_protocol_sha256": base.sha256(_TERMINAL_SOURCE_PROTOCOL),
                "preserved_failed_modern_preflight_dir": _rel(FAILED_MODERN_SMOKE_DIR),
                "preserved_failed_v0b_legacy_schema_dir": _rel(FAILED_V0B_LEGACY_DIR),
                "requires_interpreter": "legacy",
            },
            "next_recovery_hint": "Preserve this failure. If no episodes were written, fix only the wrapper/runtime/schema issue in a new version after backup; if episodes exist, audit partial outputs before any rerun.",
        })
        raise
