#!/usr/bin/env python3
"""Audit and archive the failed shard02 modern-interpreter attempt.

The 2026-09-26 shard02 attempt was accidentally launched with the modern
interpreter.  The frozen formal runner opened the validation64 bank and wrote
run_started/schedule metadata, then failed before any terminal model load,
episode directory, simulation step, raw result, or completed marker because
TensorFlow is not installed in the modern interpreter.  This recovery script is
metadata-only with respect to validation outcomes: it does not open the
validation bank, does not open/hash the sealed test bank, and runs no
simulations or training.

It verifies the failed partial contents, checks that the legacy Python/TF1
runtime imports successfully, archives the empty partial shard directory under a
failure-specific name so the frozen original runner can later create shard02
cleanly, writes a backup request, and appends durable notes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "experiments/bohn2021_reproduction"
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

FORMAL_ROOT = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926"
FAILED_DIR = FORMAL_ROOT / "shard02"
ARCHIVE_DIR = FORMAL_ROOT / "failed_shard02_20260926T161356_modern_tf_missing"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_shard02_modern_failure_recovery_20260926"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REGISTRY = ROOT / "research_artifacts/aws_runs/20260926T161356_fb71c8d7/registry.json"
STDERR = ROOT / "research_artifacts/aws_runs/20260926T161356_fb71c8d7/stderr.log"
STDOUT = ROOT / "research_artifacts/aws_runs/20260926T161356_fb71c8d7/stdout.log"
PRE_SHARD_BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260926T161238_after_shard01_audit_and_addendum.json"
RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
GATE_JSON = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"

EXPECTED_RUNNER_SHA = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
EXPECTED_GATE_SHA = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"
EXPECTED_BACKUP_PROOF_SHA = "ab714bf57fc88a35e7deb6899cac670254740fe8cef1a5c55642c1264460958a"
EXPECTED_EXPERIMENT_ID = "20260926T161356_fb71c8d7"
DOC_MARKER = "vehicle-validation64-shard02-modern-runtime-failure-recovery-20260926"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def root_path(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_once(path: Path, marker: str, body: str) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    token = "<!-- %s -->" % marker
    if token in old:
        return False
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + body.strip() + "\n", encoding="utf-8")
    return True


def require(condition: bool, failures: List[str], message: str) -> None:
    if not condition:
        failures.append(message)


def tree_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.rglob("*") if p.is_file()])


def hash_files(paths: List[Path]) -> Dict[str, str]:
    return {rel(p): sha256(p) for p in sorted(paths)}


def verify_legacy_runtime() -> Dict[str, Any]:
    """Import the legacy runtime stack without creating environments or rollouts."""
    result: Dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pid": os.getpid(),
        "legacy_executable_expected": "bohn2021-python37" in sys.executable,
        "tensorflow_imported": False,
        "stable_baselines_imported": False,
        "runtime_imports_ok": False,
    }
    try:
        tf_spec = importlib.util.find_spec("tensorflow")
        result["tensorflow_find_spec"] = bool(tf_spec)
        sb_spec = importlib.util.find_spec("stable_baselines")
        result["stable_baselines_find_spec"] = bool(sb_spec)
        from runtime import imports  # type: ignore

        tf, SAC, EvalCallback = imports()
        result["tensorflow_imported"] = True
        result["tensorflow_version"] = getattr(tf, "__version__", None)
        result["tensorflow_file"] = getattr(tf, "__file__", None)
        result["stable_baselines_imported"] = True
        result["sac_repr"] = repr(SAC)
        result["eval_callback_repr"] = repr(EvalCallback)
        result["runtime_imports_ok"] = True
    except Exception as exc:  # pragma: no cover - diagnostic path
        result["runtime_imports_ok"] = False
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
    return result


def inspect_partial_dir(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    failures: List[str] = []
    files = tree_files(path)
    rel_names = sorted(rel(p) for p in files)
    expected_names = sorted([rel(path / "run_started.json"), rel(path / "schedule.json")])
    require(path.exists(), failures, "failed shard02 directory is missing before archive")
    require(rel_names == expected_names, failures, "failed shard02 directory contains unexpected files: %s" % rel_names)
    require(not (path / "episodes").exists(), failures, "failed shard02 contains an episodes directory")
    require(not (path / "raw.json").exists(), failures, "failed shard02 contains raw.json")
    require(not (path / "summary.md").exists(), failures, "failed shard02 contains summary.md")
    require(not (path / "completed.json").exists(), failures, "failed shard02 contains completed.json")
    require(not (path / "progress.json").exists(), failures, "failed shard02 contains progress.json")

    run_started: Dict[str, Any] = {}
    schedule: Dict[str, Any] = {}
    if (path / "run_started.json").exists():
        run_started = read_json(path / "run_started.json")
        require(run_started.get("validation_accessed") is True, failures, "run_started validation_accessed is not true")
        require(run_started.get("validation64_bank_content_opened") is True, failures, "run_started validation64_bank_content_opened is not true")
        require(run_started.get("test_accessed") is False, failures, "run_started test_accessed is not false")
        require(run_started.get("sealed_test_bank_content_opened") is False, failures, "run_started sealed_test_bank_content_opened is not false")
        require(run_started.get("shard_index") == 2, failures, "run_started shard_index is not 2")
    if (path / "schedule.json").exists():
        schedule = read_json(path / "schedule.json")
        rows = schedule.get("rows") or []
        require(schedule.get("shard_index") == 2, failures, "schedule shard_index is not 2")
        require(len(rows) == 224, failures, "schedule row count is not 224")
        if rows:
            require(int(rows[0].get("execution_index")) == 448, failures, "schedule starts at unexpected execution index")
            require(int(rows[-1].get("execution_index")) == 671, failures, "schedule ends at unexpected execution index")
        require(schedule.get("validation_accessed") is True, failures, "schedule validation_accessed is not true")
        require(schedule.get("test_accessed") is False, failures, "schedule test_accessed is not false")
    return {
        "path": rel(path),
        "exists": path.exists(),
        "file_count": len(files),
        "files": rel_names,
        "hashes": hash_files(files),
        "run_started": run_started,
        "schedule_summary": {
            "shard_index": schedule.get("shard_index"),
            "rows": len(schedule.get("rows") or []),
            "first_execution_index": (schedule.get("rows") or [{}])[0].get("execution_index") if schedule.get("rows") else None,
            "last_execution_index": (schedule.get("rows") or [{}])[-1].get("execution_index") if schedule.get("rows") else None,
            "validation_accessed": schedule.get("validation_accessed"),
            "test_accessed": schedule.get("test_accessed"),
            "unique_rollout_arm_count_in_shard": schedule.get("unique_rollout_arm_count_in_shard"),
        },
    }, failures


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    failures: List[str] = []

    required = [REGISTRY, STDERR, STDOUT, PRE_SHARD_BACKUP_PROOF, RUNNER, GATE_JSON]
    missing = [rel(p) for p in required if not p.exists()]
    require(not missing, failures, "missing required files: %s" % missing)

    registry: Dict[str, Any] = read_json(REGISTRY) if REGISTRY.exists() else {}
    stderr_text = STDERR.read_text(encoding="utf-8", errors="replace") if STDERR.exists() else ""
    stdout_text = STDOUT.read_text(encoding="utf-8", errors="replace") if STDOUT.exists() else ""
    pre_proof = read_json(PRE_SHARD_BACKUP_PROOF) if PRE_SHARD_BACKUP_PROOF.exists() else {}

    require(registry.get("experiment_id") == EXPECTED_EXPERIMENT_ID, failures, "failed-run registry experiment id mismatch")
    require(registry.get("exit_status") == 1, failures, "failed-run registry exit status was not 1")
    require(registry.get("status") == "failed", failures, "failed-run registry status was not failed")
    require(registry.get("script_sha256") == EXPECTED_RUNNER_SHA, failures, "failed-run registry runner sha mismatch")
    require("ModuleNotFoundError" in stderr_text and "tensorflow" in stderr_text, failures, "stderr does not show the expected missing TensorFlow error")
    require(stdout_text == "", failures, "stdout was expected to be empty")
    require(sha256(RUNNER) == EXPECTED_RUNNER_SHA, failures, "current frozen runner sha mismatch")
    require(sha256(GATE_JSON) == EXPECTED_GATE_SHA, failures, "current frozen gate sha mismatch")
    require(sha256(PRE_SHARD_BACKUP_PROOF) == EXPECTED_BACKUP_PROOF_SHA, failures, "pre-shard backup proof sha mismatch")
    require(pre_proof.get("backup_verified") is True and pre_proof.get("remaining_changed_files") == 0, failures, "pre-shard backup proof was not verified/clean")
    require(pre_proof.get("runner_sha256") == EXPECTED_RUNNER_SHA and pre_proof.get("gate_sha256") == EXPECTED_GATE_SHA, failures, "pre-shard backup proof runner/gate sha mismatch")

    # Verify/record partial before moving.  If the archive already exists from a
    # prior successful recovery and the original path is absent, use the archive
    # as the source and avoid a second move.
    moved = False
    archive_preexisting = ARCHIVE_DIR.exists() and not FAILED_DIR.exists()
    partial_source = ARCHIVE_DIR if archive_preexisting else FAILED_DIR
    partial_before, partial_failures = inspect_partial_dir(partial_source)
    failures.extend(partial_failures)

    if not failures and FAILED_DIR.exists():
        require(not ARCHIVE_DIR.exists(), failures, "archive directory already exists while original failed shard02 also exists")
    if not failures and FAILED_DIR.exists():
        shutil.move(str(FAILED_DIR), str(ARCHIVE_DIR))
        moved = True

    archive_files = tree_files(ARCHIVE_DIR)
    archive_marker = ARCHIVE_DIR / "failure_recovery_marker.json"
    if not failures and ARCHIVE_DIR.exists():
        marker_payload = {
            "created_utc": now,
            "purpose": "Preserve the failed modern-interpreter shard02 attempt after verifying it created no episodes/control steps; original shard02 path freed for a legacy-interpreter retry after backup.",
            "failed_experiment_id": EXPECTED_EXPERIMENT_ID,
            "moved_from": rel(FAILED_DIR),
            "archive_dir": rel(ARCHIVE_DIR),
            "moved_this_run": moved,
            "validation_accessed_by_failed_run": True,
            "validation_bank_reopened_by_this_recovery": False,
            "test_accessed": False,
            "sealed_test_bank_content_opened": False,
            "new_simulations_by_failed_run": 0,
            "new_control_steps_by_failed_run": 0,
            "new_gradient_steps_by_failed_run": 0,
            "failure_reason": "wrong interpreter: modern Python lacks TensorFlow; legacy bohn2021 Python/TF1 is required for formal runner terminal model loading",
        }
        write_json(archive_marker, marker_payload)
        archive_files = tree_files(ARCHIVE_DIR)

    require(not FAILED_DIR.exists(), failures, "original shard02 path still exists after recovery; runner retry remains blocked")
    require(ARCHIVE_DIR.exists(), failures, "archive directory missing after recovery")

    legacy_runtime = verify_legacy_runtime()
    require(legacy_runtime.get("runtime_imports_ok") is True, failures, "legacy runtime imports did not pass")
    require(legacy_runtime.get("legacy_executable_expected") is True, failures, "script was not run under expected bohn2021-python37 legacy executable")

    backup_request_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_SHARD02_MODERN_FAILURE_RECOVERY_%s.json" % dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S"))
    raw_path = OUT_DIR / "raw.json"
    summary_path = OUT_DIR / "summary.md"
    completed_path = OUT_DIR / "completed.json"

    raw: Dict[str, Any] = {
        "created_utc": now,
        "passed": not failures,
        "failures": failures,
        "purpose": "Audit and archive empty failed shard02 attempt caused by accidental modern-interpreter launch; verify legacy TF runtime before retry.",
        "validation_accessed": True,
        "validation_accessed_by_failed_run": True,
        "validation_bank_reopened_by_this_recovery": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed_by_this_recovery": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created_by_this_recovery": False,
        "failed_run": {
            "experiment_id": EXPECTED_EXPERIMENT_ID,
            "registry": rel(REGISTRY),
            "registry_exit_status": registry.get("exit_status"),
            "registry_status": registry.get("status"),
            "registry_environment_python": (registry.get("environment") or {}).get("python"),
            "runtime_seconds": registry.get("runtime_seconds"),
            "peak_process_rss_kb": registry.get("peak_process_rss_kb"),
            "cloudwatch_status": (registry.get("cloudwatch") or {}).get("status"),
            "stderr_error_tail": stderr_text[-800:],
        },
        "partial_before_archive": partial_before,
        "archive": {
            "path": rel(ARCHIVE_DIR),
            "moved_this_run": moved,
            "archive_preexisting": archive_preexisting,
            "file_count": len(archive_files),
            "hashes": hash_files(archive_files),
        },
        "legacy_runtime": legacy_runtime,
        "pre_shard_backup_proof": {
            "path": rel(PRE_SHARD_BACKUP_PROOF),
            "sha256": sha256(PRE_SHARD_BACKUP_PROOF) if PRE_SHARD_BACKUP_PROOF.exists() else None,
            "commit": pre_proof.get("commit"),
            "asset_sha256": pre_proof.get("asset_sha256"),
            "backup_verified": pre_proof.get("backup_verified"),
            "remaining_changed_files": pre_proof.get("remaining_changed_files"),
        },
        "retry_plan": {
            "allowed_only_after_external_backup": True,
            "reason": "failed run and recovery artifacts changed repository state and validation bank was opened by the failed run",
            "next_command": "vehicle_validation64_shard_runner.py --shard 2 --backup-proof <post-recovery-proof> --i-accept-validation-access",
            "required_interpreter": "legacy",
            "do_not_repeat_failed_config": "Do not run shard02 with the modern interpreter again; TensorFlow is absent there.",
        },
        "interpretation_limits": [
            "This recovery is not validation model-selection evidence; no episode/control outcome was produced.",
            "The failed runner opened the validation64 bank before failing, so validation_accessed remains true for budget accounting.",
            "No sealed final test content was opened or hashed.",
            "A verified external backup covering the failure archive, recovery outputs, docs, registry and backup request is required before the legacy shard02 retry.",
        ],
        "backup_required_before_more_formal_validation": True,
        "backup_request_path": rel(backup_request_path),
    }
    write_json(raw_path, raw)

    lines = [
        "# Vehicle validation64 shard02 modern-runtime failure recovery",
        "",
        "Created UTC: `%s`." % now,
        "",
        "This metadata recovery audited the failed shard02 attempt `20260926T161356_fb71c8d7`. The failed run used the modern interpreter, opened the validation64 bank, wrote only `run_started.json` and `schedule.json`, then failed before terminal model loading because TensorFlow is unavailable in the modern environment.",
        "",
        "- recovery passed: `%s`" % raw["passed"],
        "- archived partial directory: `%s`" % rel(ARCHIVE_DIR),
        "- moved this run: `%s`" % moved,
        "- partial files before archive: `%s`" % partial_before.get("files"),
        "- new simulations/control steps/gradient steps: `0 / 0 / 0`",
        "- validation bank reopened by recovery: `false`",
        "- sealed test accessed/hashed: `false / false`",
        "- legacy runtime import OK: `%s`" % legacy_runtime.get("runtime_imports_ok"),
        "- legacy executable: `%s`" % legacy_runtime.get("python_executable"),
        "- TensorFlow version: `%s`" % legacy_runtime.get("tensorflow_version"),
        "",
        "Next formal shard02 retry is blocked until an external backup proof covers this recovery/archive/docs/registry. The retry must use the legacy interpreter; do not repeat the modern-interpreter configuration.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    backup_request = {
        "status": "external_backup_requested_after_shard02_modern_runtime_failure_recovery",
        "backup_verified": False,
        "created_utc": now,
        "reason": "Preserve failed modern-interpreter shard02 attempt, archive marker, recovery diagnostics, docs and registry before any legacy shard02 retry.",
        "validation_accessed_by_failed_run": True,
        "validation_bank_reopened_by_recovery": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations_by_failed_run": 0,
        "new_control_steps_by_failed_run": 0,
        "new_gradient_steps_by_failed_run": 0,
        "required_before_shard02_retry": True,
        "artifacts_to_cover": sorted(list(raw["archive"]["hashes"].keys()) + [
            rel(raw_path), rel(summary_path), rel(completed_path), rel(REGISTRY), rel(STDERR), rel(STDOUT),
            rel(PRE_SHARD_BACKUP_PROOF), rel(RUNNER), rel(GATE_JSON), "STATUS.md", "RESEARCH_LOG.md",
            "RESULTS_AUDIT.md", "DECISIONS.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv",
        ]),
        "next_action_after_verified_backup": "Retry vehicle validation64 shard02 with legacy interpreter and the frozen original runner using a post-recovery backup proof; sealed test remains closed.",
    }
    write_json(backup_request_path, backup_request)
    raw["backup_request_sha256"] = sha256(backup_request_path)
    write_json(raw_path, raw)

    doc_body = """
## 2026-09-26 vehicle validation64 shard02 modern-runtime failure recovery

UTC: {now}. The planned shard02 run `20260926T161356_fb71c8d7` failed before any episode/control step because it was accidentally launched with the modern interpreter, where TensorFlow is unavailable (`ModuleNotFoundError: No module named 'tensorflow'`). The failed run had already opened the validation64 bank and wrote only `run_started.json` plus `schedule.json`; no `episodes/`, `progress.json`, `raw.json`, `summary.md`, or `completed.json` existed. Recovery archived the empty partial directory to `{archive}` so the frozen runner can later create `shard02` cleanly. Recovery itself reopened no validation bank content, opened no sealed test content, and ran 0 simulations / 0 control steps / 0 gradient steps. Legacy runtime import under `{python}` passed with TensorFlow `{tf_version}`. A verified external backup covering the failure archive, recovery artifacts, docs, registry and backup request is required before retrying shard02 with the legacy interpreter.
""".format(
        now=now,
        archive=rel(ARCHIVE_DIR),
        python=legacy_runtime.get("python_executable"),
        tf_version=legacy_runtime.get("tensorflow_version"),
    ).strip()
    docs_updated = []
    for name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists() and append_once(path, DOC_MARKER, doc_body):
            docs_updated.append(name)

    completed = {
        "passed": raw["passed"],
        "validation_accessed": True,
        "validation_accessed_by_failed_run": True,
        "validation_bank_reopened_by_this_recovery": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed_by_this_recovery": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created_by_this_recovery": False,
        "failed_experiment_id": EXPECTED_EXPERIMENT_ID,
        "failed_run_exit_status": registry.get("exit_status"),
        "failure_reason": "modern interpreter missing TensorFlow before terminal model load",
        "legacy_runtime_imports_ok": legacy_runtime.get("runtime_imports_ok"),
        "archived_partial_dir": rel(ARCHIVE_DIR),
        "original_shard02_path_free_for_retry": not FAILED_DIR.exists(),
        "backup_required_before_more_formal_validation": True,
        "backup_request_path": rel(backup_request_path),
        "docs_updated": docs_updated,
        "hashes": {
            rel(raw_path): sha256(raw_path),
            rel(summary_path): sha256(summary_path),
            rel(backup_request_path): sha256(backup_request_path),
        },
    }
    write_json(completed_path, completed)

    print(json.dumps({
        "passed": raw["passed"],
        "recovery_raw": rel(raw_path),
        "recovery_summary": rel(summary_path),
        "recovery_completed": rel(completed_path),
        "backup_request": rel(backup_request_path),
        "validation_accessed_by_failed_run": True,
        "validation_bank_reopened_by_recovery": False,
        "test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "legacy_runtime_imports_ok": legacy_runtime.get("runtime_imports_ok"),
        "archived_partial_dir": rel(ARCHIVE_DIR),
        "original_shard02_path_free_for_retry": not FAILED_DIR.exists(),
        "backup_required_before_more_formal_validation": True,
    }, sort_keys=True))
    return 0 if raw["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
