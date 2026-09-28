#!/usr/bin/env python3
"""Legacy-interpreter retry wrapper for vehicle stress-v1b terminal/reward ablation.

The frozen v0 smoke protocol was attempted once under the modern Python 3.12
interpreter and failed before any rollout because TensorFlow is only available
in the legacy/TF1 runtime.  The failed directory is preserved as evidence.  This
wrapper makes no scientific/protocol change: it reuses the frozen v0 protocol,
dry-run marker, state targets, horizons, terminal modes and analysis rules, but
routes outputs to v0b legacy-retry directories and requires a verified backup
proof that postdates this wrapper source before any rollout can execute.

No validation64 bank or sealed test is opened.  No training/refit/gradient
updates are performed.  Run with the research tool using interpreter="legacy".
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
STAMP_V0B = "20260928T2335Z_legacy_retry"

# Route only the simulation outputs away from the preserved failed v0 smoke.
# The frozen dry-run/protocol paths remain unchanged and are verified by base.
base.SMOKE_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0b_smoke_{STAMP_V0B}"
base.FULL_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0b_full_{STAMP_V0B}"
base.STATE_SMOKE = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1b_terminal_reward_ablation_v0b_smoke_{STAMP_V0B}.md"
base.STATE_FULL = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1b_terminal_reward_ablation_v0b_full_{STAMP_V0B}.md"
base.MARKER_RUN = f"vehicle-stress-v1b-terminal-reward-ablation-v0b-legacy-retry-run-{STAMP_V0B}"

# Make Path(__file__) references in base.run_ablation identify this executable
# wrapper for backup min-time checks, backup requests and completed hashes.  The
# original v0 source is added back explicitly in annotations/hashes below.
base.__file__ = str(WRAPPER_SOURCE)

_ORIGINAL_SOURCE_MTIME_UTC = base.source_mtime_utc


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


def _annotate_json(path: Path, run_kind: str) -> None:
    if not path.exists():
        return
    obj = _read_json(path)
    wrapper_info = {
        "wrapper_source": _rel(WRAPPER_SOURCE),
        "wrapper_sha256": base.sha256(WRAPPER_SOURCE),
        "original_v0_source": _rel(ORIGINAL_V0_SOURCE),
        "original_v0_sha256": base.sha256(ORIGINAL_V0_SOURCE),
        "preserved_failed_modern_preflight_dir": _rel(FAILED_MODERN_SMOKE_DIR),
        "preserved_failed_modern_preflight_failure_json": _rel(FAILED_MODERN_SMOKE_DIR / "failure.json"),
        "reason": "v0 smoke was mistakenly run with the modern interpreter and failed pre-rollout on missing TensorFlow; v0b reroutes outputs and must be executed with legacy/TF1 while preserving the frozen v0 protocol unchanged.",
        "run_kind": run_kind,
        "output_dir": _rel(base.SMOKE_DIR if run_kind == "smoke" else base.FULL_DIR),
        "requires_interpreter": "legacy",
    }
    obj["legacy_retry_wrapper"] = wrapper_info
    hashes = dict(obj.get("hashes") or {})
    for p in (
        WRAPPER_SOURCE,
        ORIGINAL_V0_SOURCE,
        FAILED_MODERN_SMOKE_DIR / "failure.json",
        FAILED_MODERN_SMOKE_DIR / "runtime_preflight.json",
        FAILED_MODERN_SMOKE_DIR / "run_started.json",
    ):
        if p.exists():
            hashes[_rel(p)] = base.sha256(p)
    obj["hashes"] = hashes
    _write_json(path, obj)


def _annotate_outputs(run_kind: str) -> None:
    out_dir = base.SMOKE_DIR if run_kind == "smoke" else base.FULL_DIR
    _annotate_json(out_dir / "raw.json", run_kind)
    _annotate_json(out_dir / "completed.json", run_kind)
    note_path = out_dir / "legacy_retry_wrapper_note.md"
    note_path.write_text(
        "# v0b legacy retry wrapper note\n\n"
        f"Run kind: `{run_kind}`. This wrapper preserves the failed modern-interpreter v0 smoke attempt at "
        f"`{_rel(FAILED_MODERN_SMOKE_DIR)}` and reroutes the unchanged frozen terminal/reward-source ablation "
        "protocol to a v0b output directory for execution under the legacy/TF1 interpreter. No validation64 bank, "
        "sealed test, training/refit or gradient update is used.\n",
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--dry-run" in args:
        raise base.ContractError("v0b legacy retry does not create a new scientific dry-run; use the already frozen v0 dry-run/protocol and run --run-smoke or --run-full only.")
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
        target = base.SMOKE_DIR if "--run-smoke" in sys.argv else base.FULL_DIR if "--run-full" in sys.argv else ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0b_failure_unknown_mode_20260928T2335Z_legacy_retry"
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
            "legacy_retry_wrapper": {
                "wrapper_source": _rel(WRAPPER_SOURCE),
                "wrapper_sha256": base.sha256(WRAPPER_SOURCE),
                "original_v0_source": _rel(ORIGINAL_V0_SOURCE),
                "original_v0_sha256": base.sha256(ORIGINAL_V0_SOURCE),
                "preserved_failed_modern_preflight_dir": _rel(FAILED_MODERN_SMOKE_DIR),
                "requires_interpreter": "legacy",
            },
            "next_recovery_hint": "Preserve this failure. If no episodes were written, fix the wrapper/runtime issue in a new version after backup; if episodes exist, audit partial outputs before any rerun.",
        })
        raise
