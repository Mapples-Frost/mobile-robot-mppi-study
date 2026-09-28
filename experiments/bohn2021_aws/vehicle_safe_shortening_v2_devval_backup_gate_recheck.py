#!/usr/bin/env python3
"""Metadata-only backup gate recheck before v2 transition-hold devval shard00.

This script performs no rollout, no training, no validation/development bank
creation or reopening, and no sealed-test access.  It inspects only repository
metadata and backup-proof JSON files to decide whether the frozen v2
transition-hold development-validation shard00 may start.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKUP_DIR = REPO_ROOT / "research_artifacts" / "aws_backup_proofs"
DIAG_ROOT = REPO_ROOT / "research_artifacts" / "aws_diagnostics"

# Preflight completed at 2026-09-28T00:46:17.422154Z; require the external
# proof to be strictly after that, and in practice after run_experiment has
# finalized registry/stdout/stderr/cloudwatch for the preflight.  We use a
# conservative post-preflight threshold.
REQUIRED_AFTER_UTC = "2026-09-28T00:46:20+00:00"

TRANSITION_PROTOCOL = "research_artifacts/aws_protocols/vehicle_safe_shortening_v2_transition_hold_protocol_20260928.md"
TRANSITION_PROTOCOL_SHA256 = "cba016c6ca3615de2a1c2b770e24475d4f06784b6060d97ac30e50f509bae65e"
DEVVAL_PROTOCOL = "research_artifacts/aws_protocols/vehicle_safe_shortening_v2_transition_hold_devval_protocol_20260928.md"
DEVVAL_PROTOCOL_SHA256 = "57556418b6caee02fd28a949aa92ceeba705361c4a80c183e12b1d67e80c293a"
SMOKE_SCRIPT = "experiments/bohn2021_aws/vehicle_safe_shortening_v2_transition_hold_smoke.py"
SMOKE_SCRIPT_SHA256 = "f394fd2b53fa39e7ecfa5944d67c469b620a4a9214ad97f81c994949458bbaac"
DEVVAL_RUNNER = "experiments/bohn2021_aws/vehicle_safe_shortening_v2_transition_hold_devval_shard_runner.py"
DEVVAL_RUNNER_SHA256 = "0f35aa0432f4d0d7f57df74c8dc5aee5ae33d0a848f191ea3696e196291da229"
PREFLIGHT_SCRIPT = "experiments/bohn2021_aws/vehicle_safe_shortening_v2_devval_preflight.py"
PREFLIGHT_SCRIPT_SHA256 = "05b9d530ad4347674f9e21d086d9ac3c54bce4c3c25448593255647041d4dcce"
SMOKE_DIR = "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_smoke_20260928"
PREFLIGHT_DIR = "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_devval_preflight_20260928T004500Z"
PREFLIGHT_RUN_ID = "20260928T004617_70cc8a85"
SMOKE_RUN_ID = "20260928T003231_beca23eb"
DEVVAL_ROOT = "research_artifacts/aws_development_validation/vehicle_safe_shortening_v2_transition_hold_devval_20260928_v1"
FRESH_BANK_PATH = "research_artifacts/aws_development_validation/vehicle_safe_shortening_v2_transition_hold_devval_20260928_v1/fresh_devval_bank"

LATEST_SUPERVISOR_BACKUP_FROM_CONTEXT = {
    "time": "2026-09-28T00:45:40.181961+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "c4e6be03e8c856eab0d19eeb9bc2aab98c02e55e",
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "asset": "20260928T004536_373f4112.tar.gz",
    "asset_sha256": "9286dbaf2217bf69f6399a0c16430aacd7832efe7f2110ff32b4b187156a286c",
    "why_inadequate": "It predates the v2 devval preflight run 20260928T004617_70cc8a85 and therefore cannot cover the preflight outputs, finalized run registry/logs, or the preflight backup request.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def label_from_time(ts: str) -> str:
    return ts.replace("-", "").replace(":", "").replace("+00:00", "Z").replace("Z", "Z")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def proof_time(data: Dict[str, Any]) -> Optional[str]:
    for key in ("proof_created_utc", "proof_recorded_utc", "time", "created_utc", "reported_time_utc"):
        v = data.get(key)
        if isinstance(v, str):
            return v
    return None


def filename_timestamp(path: Path) -> Optional[str]:
    m = re.search(r"backup_proof_(\d{8}T\d{6})", path.name)
    return m.group(1) if m else None


def coverage_flags(data: Dict[str, Any]) -> Dict[str, bool]:
    text = json.dumps(data, sort_keys=True)
    return {
        "github_asset_verified": "github_server_sha256" in text or "download_sha256" in text or "asset_sha256" in text,
        "transition_protocol_hash": TRANSITION_PROTOCOL_SHA256 in text or TRANSITION_PROTOCOL in text,
        "devval_protocol_hash": DEVVAL_PROTOCOL_SHA256 in text or DEVVAL_PROTOCOL in text,
        "smoke_script_hash": SMOKE_SCRIPT_SHA256 in text or SMOKE_SCRIPT in text,
        "devval_runner_hash": DEVVAL_RUNNER_SHA256 in text or DEVVAL_RUNNER in text,
        "preflight_script_hash": PREFLIGHT_SCRIPT_SHA256 in text or PREFLIGHT_SCRIPT in text,
        "smoke_outputs": SMOKE_DIR in text or SMOKE_RUN_ID in text,
        "preflight_outputs": PREFLIGHT_DIR in text or PREFLIGHT_RUN_ID in text,
        "docs_registry": "STATUS.md" in text and "EXPERIMENT_REGISTRY.csv" in text,
    }


def inspect_backup_proofs() -> Dict[str, Any]:
    required_dt = parse_time(REQUIRED_AFTER_UTC)
    all_records: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    adequate: List[Dict[str, Any]] = []
    for path in sorted(BACKUP_DIR.glob("backup_proof_*.json")):
        data = read_json(path)
        rec: Dict[str, Any] = {
            "path": rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "json_ok": data is not None,
            "filename_timestamp": filename_timestamp(path),
        }
        if data is not None:
            t_str = proof_time(data)
            t_dt = parse_time(t_str)
            flags = coverage_flags(data)
            after_required = bool(required_dt and t_dt and t_dt > required_dt)
            filename_late = bool(rec["filename_timestamp"] and rec["filename_timestamp"] >= "20260928T004620")
            candidate = after_required or filename_late
            is_adequate = bool(
                data.get("backup_verified") is True
                and data.get("remaining_changed_files") == 0
                and after_required
                and flags["github_asset_verified"]
                and flags["transition_protocol_hash"]
                and flags["devval_protocol_hash"]
                and flags["smoke_script_hash"]
                and flags["devval_runner_hash"]
                and flags["preflight_script_hash"]
                and flags["smoke_outputs"]
                and flags["preflight_outputs"]
            )
            rec.update({
                "proof_time": t_str,
                "backup_verified": data.get("backup_verified"),
                "remaining_changed_files": data.get("remaining_changed_files"),
                "after_required_time": after_required,
                "candidate_post_preflight_time_or_filename": candidate,
                "coverage_flags": flags,
                "adequate_for_v2_devval_shard00": is_adequate,
            })
            if candidate:
                candidates.append(rec)
            if is_adequate:
                adequate.append(rec)
        all_records.append(rec)
    return {
        "required_after_utc": REQUIRED_AFTER_UTC,
        "all_backup_proof_count": len(all_records),
        "backup_proof_20260928_glob": [rel(p) for p in sorted(BACKUP_DIR.glob("backup_proof_20260928*.json"))],
        "all_backup_proofs": all_records,
        "post_preflight_candidate_count": len(candidates),
        "post_preflight_candidates": candidates,
        "adequate_backup_proofs_found": len(adequate),
        "adequate_backup_proofs": adequate,
        "latest_backup_proof_file": all_records[-1] if all_records else None,
    }


def key_hash_checks() -> List[Dict[str, Any]]:
    expected = [
        (TRANSITION_PROTOCOL, TRANSITION_PROTOCOL_SHA256),
        (DEVVAL_PROTOCOL, DEVVAL_PROTOCOL_SHA256),
        (SMOKE_SCRIPT, SMOKE_SCRIPT_SHA256),
        (DEVVAL_RUNNER, DEVVAL_RUNNER_SHA256),
        (PREFLIGHT_SCRIPT, PREFLIGHT_SCRIPT_SHA256),
    ]
    checks: List[Dict[str, Any]] = []
    for name, want in expected:
        p = REPO_ROOT / name
        got = sha256_file(p)
        checks.append({
            "path": name,
            "exists": p.exists(),
            "expected_sha256": want,
            "actual_sha256": got,
            "matches": got == want,
        })
    for name in [
        f"{SMOKE_DIR}/summary.md",
        f"{SMOKE_DIR}/raw.json",
        f"{SMOKE_DIR}/completed.json",
        f"{PREFLIGHT_DIR}/summary.md",
        f"{PREFLIGHT_DIR}/raw.json",
        f"{PREFLIGHT_DIR}/completed.json",
        f"research_artifacts/aws_runs/{SMOKE_RUN_ID}/registry.json",
        f"research_artifacts/aws_runs/{SMOKE_RUN_ID}/stdout.log",
        f"research_artifacts/aws_runs/{SMOKE_RUN_ID}/stderr.log",
        f"research_artifacts/aws_runs/{SMOKE_RUN_ID}/cloudwatch_snapshot.json",
        f"research_artifacts/aws_runs/{PREFLIGHT_RUN_ID}/registry.json",
        f"research_artifacts/aws_runs/{PREFLIGHT_RUN_ID}/stdout.log",
        f"research_artifacts/aws_runs/{PREFLIGHT_RUN_ID}/stderr.log",
        f"research_artifacts/aws_runs/{PREFLIGHT_RUN_ID}/cloudwatch_snapshot.json",
        "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V2_TRANSITION_HOLD_SMOKE_20260928T003940.json",
        "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V2_DEVVAL_PREFLIGHT_20260928T004617.json",
    ]:
        p = REPO_ROOT / name
        checks.append({"path": name, "exists": p.exists(), "sha256": sha256_file(p)})
    return checks


def check_no_partial_devval() -> Dict[str, Any]:
    root = REPO_ROOT / DEVVAL_ROOT
    bank = REPO_ROOT / FRESH_BANK_PATH
    if not root.exists():
        entries: List[str] = []
    else:
        entries = [rel(p) for p in sorted(root.iterdir())[:200]]
    return {
        "devval_root": DEVVAL_ROOT,
        "devval_root_exists": root.exists(),
        "devval_root_entries_first_200": entries,
        "fresh_bank_path": FRESH_BANK_PATH,
        "fresh_bank_exists": bank.exists(),
        "shard00_exists": (root / "shard00").exists(),
    }


def append_once(path: Path, marker: str, text: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old + "\n" + text.strip() + "\n", encoding="utf-8")


def append_registry(timestamp: str, status: str, record_path: str) -> None:
    # Do not duplicate run_experiment's row. This manual row records the
    # scientific gate decision; run_experiment records execution metadata.
    csv_path = REPO_ROOT / "EXPERIMENT_REGISTRY.csv"
    row = {
        "experiment_id": "",
        "timestamp": timestamp,
        "method": "IMPROVED_vehicle_safe_shortening_v2_devval_backup_gate_recheck_metadata_only",
        "seed": "metadata_no_rng",
        "split": "metadata_only_no_devval_bank_no_validation64_no_sealed_test",
        "commit_sha": "",
        "status": status,
        "exit_status": "",
        "runtime_seconds": "",
        "peak_process_rss_kb": "",
        "record": record_path,
    }
    existing = csv_path.read_text(encoding="utf-8") if csv_path.exists() else ""
    if record_path in existing:
        return
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not existing.strip():
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    created = utc_now()
    label = label_from_time(created)
    diag_dir = DIAG_ROOT / f"vehicle_safe_shortening_v2_devval_backup_gate_recheck_{label}"
    diag_dir.mkdir(parents=True, exist_ok=False)

    proof_inspection = inspect_backup_proofs()
    hash_checks = key_hash_checks()
    missing_or_mismatch = [c for c in hash_checks if c.get("exists") is False or ("matches" in c and c.get("matches") is not True)]
    partial_devval = check_no_partial_devval()
    shard00_blocked = proof_inspection["adequate_backup_proofs_found"] == 0

    request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V2_DEVVAL_BACKUP_GATE_RECHECK_{label}.json"
    must_cover = [
        TRANSITION_PROTOCOL,
        DEVVAL_PROTOCOL,
        SMOKE_SCRIPT,
        DEVVAL_RUNNER,
        PREFLIGHT_SCRIPT,
        SMOKE_DIR,
        PREFLIGHT_DIR,
        f"research_artifacts/aws_runs/{SMOKE_RUN_ID}/registry.json",
        f"research_artifacts/aws_runs/{SMOKE_RUN_ID}/stdout.log",
        f"research_artifacts/aws_runs/{SMOKE_RUN_ID}/stderr.log",
        f"research_artifacts/aws_runs/{SMOKE_RUN_ID}/cloudwatch_snapshot.json",
        f"research_artifacts/aws_runs/{PREFLIGHT_RUN_ID}/registry.json",
        f"research_artifacts/aws_runs/{PREFLIGHT_RUN_ID}/stdout.log",
        f"research_artifacts/aws_runs/{PREFLIGHT_RUN_ID}/stderr.log",
        f"research_artifacts/aws_runs/{PREFLIGHT_RUN_ID}/cloudwatch_snapshot.json",
        "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V2_TRANSITION_HOLD_SMOKE_20260928T003940.json",
        "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V2_DEVVAL_PREFLIGHT_20260928T004617.json",
        rel(request_path),
        "STATUS.md",
        "RESEARCH_LOG.md",
        "DECISIONS.md",
        "RESULTS_AUDIT.md",
        "REPRODUCTION_PROTOCOL.md",
        "EXPERIMENT_REGISTRY.csv",
        "this gate recheck raw/summary/completed artifacts",
        "this run_experiment registry/stdout/stderr/cloudwatch snapshot after finalization",
    ]
    request = {
        "created_utc": created,
        "status": "external_backup_requested_before_v2_devval_shard00" if shard00_blocked else "adequate_backup_found",
        "required_before_v2_devval_shard00": shard00_blocked,
        "backup_verified": False,
        "required_after_utc": REQUIRED_AFTER_UTC,
        "latest_supervisor_backup_from_context": LATEST_SUPERVISOR_BACKUP_FROM_CONTEXT,
        "validation64_bank_opened_by_this_recheck": False,
        "fresh_devval_bank_generated_by_this_recheck": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "reason": "The frozen v2 transition-hold devval campaign may not start until an external backup after preflight finalization covers the v2 smoke, protocols, shard runner, preflight artifacts, backup requests, docs/registry and run logs. Repository-local proof search found no adequate proof at this recheck." if shard00_blocked else "A repository-local adequate proof was found; shard00 may proceed if other protocol gates remain satisfied.",
        "must_cover_before_shard00": must_cover,
        "next_action_after_verified_backup": {
            "script": DEVVAL_RUNNER,
            "args": ["--shard", "0"],
            "interpreter": "legacy",
            "method": "IMPROVED_vehicle_safe_shortening_v2_transition_hold_devval_shard00",
            "split": "fresh_v2_devval_shard00_cases_4_of_64_no_validation64_no_sealed_test",
            "budget": {
                "new_rollout_episodes_planned": 172,
                "new_control_steps_planned_max": 25800,
                "fresh_devval_bank_resets_if_absent": 64,
                "new_training_episodes": 0,
                "new_gradient_steps": 0,
                "sealed_test_episodes": 0,
            },
        },
    }
    write_json(request_path, request)

    raw = {
        "created_utc": created,
        "purpose": "metadata-only backup gate recheck before v2 transition-hold fresh development-validation shard00",
        "method_classification": "IMPROVED safe-shortening v2 transition-hold, not ORIGINAL SAC",
        "validation64_bank_opened": False,
        "fresh_devval_bank_generated": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "required_after_utc": REQUIRED_AFTER_UTC,
        "latest_supervisor_backup_from_context": LATEST_SUPERVISOR_BACKUP_FROM_CONTEXT,
        "proof_inspection": proof_inspection,
        "hash_checks": hash_checks,
        "missing_or_mismatch": missing_or_mismatch,
        "partial_devval_inventory": partial_devval,
        "v2_devval_shard00_blocked": shard00_blocked,
        "backup_request_path": rel(request_path),
    }
    raw_path = diag_dir / "raw.json"
    write_json(raw_path, raw)

    summary_lines = [
        "# Vehicle safe-shortening v2 devval backup gate recheck",
        "",
        f"UTC: `{created}`.",
        "",
        "Metadata-only gate check before the frozen IMPROVED v2 transition-hold development-validation shard00. No rollout/control steps, no training, no fresh devval bank generation, no historical validation64 bank reopen, and no sealed final-test access/hash occurred.",
        "",
        "## Result",
        "",
        f"- Required proof time: after `{REQUIRED_AFTER_UTC}`.",
        f"- Repository-local `backup_proof_20260928*.json` files: `{proof_inspection['backup_proof_20260928_glob']}`.",
        f"- Post-preflight candidate backup proofs found: `{proof_inspection['post_preflight_candidate_count']}`.",
        f"- Adequate backup proofs found: `{proof_inspection['adequate_backup_proofs_found']}`.",
        f"- Latest supervisor backup from context: `{LATEST_SUPERVISOR_BACKUP_FROM_CONTEXT['time']}`; inadequate because it predates the preflight run/finalized outputs.",
        f"- Fresh v2 devval root exists: `{partial_devval['devval_root_exists']}`; shard00 exists: `{partial_devval['shard00_exists']}`; fresh bank exists: `{partial_devval['fresh_bank_exists']}`.",
        f"- v2 devval shard00 blocked: `{shard00_blocked}`.",
        "",
        "## Key hash/source checks",
        "",
    ]
    for c in hash_checks:
        if "matches" in c:
            summary_lines.append(f"- `{c['path']}` exists={c['exists']} matches_expected_sha256={c['matches']}")
        else:
            summary_lines.append(f"- `{c['path']}` exists={c['exists']} sha256={c.get('sha256')}")
    summary_lines.extend([
        "",
        "## Backup request",
        "",
        f"- `{rel(request_path)}`",
        "",
        "Next action if and only if a verified external backup proof after this gate recheck and after this run_experiment finalizes is available: run one v2 devval shard00 with the legacy interpreter:",
        "",
        "```text",
        "experiments/bohn2021_aws/vehicle_safe_shortening_v2_transition_hold_devval_shard_runner.py --shard 0",
        "```",
        "",
        "Declared shard00 budget: 172 rollout episodes, maximum 25,800 control steps, up to 64 fresh devval bank reset snapshots if the bank is still absent, 0 training/gradient steps, and 0 sealed-test episodes/control steps.",
    ])
    summary_path = diag_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    completed = {
        "created_utc": created,
        "passed": not missing_or_mismatch,
        "adequate_backup_proofs_found": proof_inspection["adequate_backup_proofs_found"],
        "v2_devval_shard00_blocked": shard00_blocked,
        "validation64_bank_opened": False,
        "fresh_devval_bank_generated": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "raw_path": rel(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "summary_path": rel(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "backup_request_path": rel(request_path),
        "backup_request_sha256": sha256_file(request_path),
    }
    completed_path = diag_dir / "completed.json"
    write_json(completed_path, completed)
    completed["completed_path"] = rel(completed_path)
    completed["completed_sha256"] = sha256_file(completed_path)
    write_json(completed_path, completed)
    completed["completed_sha256"] = sha256_file(completed_path)

    marker = f"<!-- vehicle-safe-shortening-v2-devval-backup-gate-recheck-{label} -->"
    doc = f"""
{marker}
## 2026-09-28 vehicle safe-shortening v2 devval backup gate recheck

UTC: {created}. Metadata-only gate check before frozen IMPROVED v2 transition-hold devval shard00. No rollout/control steps, no training, no fresh devval bank generation, no historical validation64 bank reopen, and no sealed-test access/hash occurred. Required proof time was after `{REQUIRED_AFTER_UTC}` and after preflight/run-log finalization. Repository-local post-preflight candidate proofs found: {proof_inspection['post_preflight_candidate_count']}; adequate proofs found: {proof_inspection['adequate_backup_proofs_found']}. Latest supervisor-context verified backup remains `{LATEST_SUPERVISOR_BACKUP_FROM_CONTEXT['time']}`, which predates the v2 preflight and is therefore inadequate. Fresh v2 devval root exists={partial_devval['devval_root_exists']}, shard00 exists={partial_devval['shard00_exists']}, fresh bank exists={partial_devval['fresh_bank_exists']}. Shard00 blocked={shard00_blocked}. Artifacts: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(completed_path)}`. Backup request: `{rel(request_path)}`. Next after verified backup: run exactly one legacy shard00 (`{DEVVAL_RUNNER} --shard 0`) with 172 episodes/max 25,800 control steps and sealed test closed.
""".strip()
    for doc_name in ["STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md"]:
        append_once(REPO_ROOT / doc_name, marker, doc)
    decision_marker = f"<!-- decision-v2-devval-backup-gate-{label} -->"
    decision_doc = f"""
{decision_marker}
## 2026-09-28 decision: hold v2 devval until post-preflight backup proof

Evidence: metadata-only gate recheck at {created} found {proof_inspection['adequate_backup_proofs_found']} adequate repository-local backup proofs after `{REQUIRED_AFTER_UTC}`. The latest supervisor-context verified backup at `{LATEST_SUPERVISOR_BACKUP_FROM_CONTEXT['time']}` predates the v2 devval preflight and cannot cover its outputs/logs. Decision: do not start v2 devval shard00 until a verified external backup after this recheck covers the v2 protocols/source, smoke, preflight, docs/registry, backup requests, this gate recheck, and this run's finalized logs. This is a storage/recoverability gate only; it does not alter the frozen v2 controller, schedule, selection rules, or acceptance criteria.
""".strip()
    append_once(REPO_ROOT / "DECISIONS.md", decision_marker, decision_doc)
    protocol_marker = f"<!-- protocol-v2-devval-backup-gate-state-{label} -->"
    protocol_doc = f"""
{protocol_marker}
## 2026-09-28 v2 transition-hold devval backup gate state

The frozen v2 transition-hold development-validation protocol remains unchanged. This gate recheck created no rollout, no bank generation, no training, and no sealed-test access. Shard00 remains blocked until an external backup proof after `{created}` (and after this run_experiment finalizes) covers the v2 smoke/protocols/runner/preflight artifacts and run logs plus docs/registry and backup requests. Once satisfied, the next protocol action is exactly shard00 with legacy interpreter and the preregistered shard00 budget: 172 episodes, max 25,800 control steps, 0 training/gradient steps, sealed test closed.
""".strip()
    append_once(REPO_ROOT / "REPRODUCTION_PROTOCOL.md", protocol_marker, protocol_doc)
    append_registry(created, "backup_gate_blocked" if shard00_blocked else "backup_gate_passed", rel(completed_path))

    print(json.dumps({
        "created_utc": created,
        "diag_dir": rel(diag_dir),
        "completed_path": rel(completed_path),
        "v2_devval_shard00_blocked": shard00_blocked,
        "adequate_backup_proofs_found": proof_inspection["adequate_backup_proofs_found"],
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "test_accessed": False,
        "backup_request_path": rel(request_path),
    }, indent=2, sort_keys=True))
    return 0 if not missing_or_mismatch else 2


if __name__ == "__main__":
    raise SystemExit(main())
