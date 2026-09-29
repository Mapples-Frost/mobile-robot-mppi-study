#!/usr/bin/env python3
"""No-simulation backup gate recheck before v1c v0c smoke.

This is a metadata-only continuity/gate script.  It verifies that the v1c v0c
hash-repair dry-run artifacts are internally consistent, records the latest
supervisor-context backup mentioned in the handoff as *insufficient for smoke*
when it predates the dry-run, scans repository-local backup proofs, and writes a
new explicit backup request if no adequate post-dryrun proof exists.

It must not import TF/legacy vehicle controllers, open validation64 or sealed
final-test banks, generate candidate pools, reset environments, or run any
rollouts.  Its sole purpose is to preserve state and prevent accidental smoke
simulation before the dry-run/source/protocol artifacts are externally backed up.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260929T0105Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_v0c_smoke_backup_gate_recheck_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v1c_v0c_smoke_backup_gate_recheck.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1C_V0C_DRYRUN_GATE_RECHECK_{STAMP}.json"
CONTEXT_BACKUP_RECORD = BACKUP_DIR / "backup_proof_20260929T004329_from_user_context_v1c_v0c_gate_recheck_not_adequate_for_smoke.json"

V0C_SOURCE = ROOT / "experiments/bohn2021_aws/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_hash_repair.py"
V0C_DRYRUN = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_dryrun_20260929T0055Z_hash_repair"
V0C_COMPLETED = V0C_DRYRUN / "completed.json"
V0C_RAW = V0C_DRYRUN / "raw.json"
V0C_SUMMARY = V0C_DRYRUN / "summary.md"
V0C_REQUEST_BEFORE_SMOKE = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1C_TERMINAL_STABLE_LABEL_DENSITY_PROBE_V0C_SMOKE_20260929T0055Z_hash_repair.json"
V0B_OBJECTIVE_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0b_schema_repair_20260929T0035Z/completed.json"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1c_terminal_stable_label_density_v0_frozen_20260929T0040Z.json"

# The user continuation context supplied this verified supervisor backup.  It is
# preserved here because it was not present as a local proof file, but the gate
# below deliberately rejects it for smoke because its timestamp predates the v0c
# dry-run completion/request.  This is not a new remote verification.
SUPERVISOR_CONTEXT_BACKUP = {
    "time": "2026-09-29T00:43:29.230337+00:00",
    "status": "verified",
    "backup_verified": True,
    "remaining_changed_files": 0,
    "commit": "1d89a7c40896660e902e53635f210ce3f12c6703",
    "changed_files": 32,
    "packages_this_run": [
        {
            "name": "20260929T004326_8b42daca.tar.gz",
            "url": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/download/bohn-aws-evidence-20260926/20260929T004326_8b42daca.tar.gz",
            "id": 596803479,
            "sha256": "d9e78ca576742ab9f87686673c5708bc506f4984b5a46c47aeed2e265c010379",
            "bytes": 8833935,
            "verification": "github_server_sha256",
        }
    ],
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "tracked_files": 132408,
    "source": "transcribed from user-supplied supervisor context during v1c v0c smoke backup gate recheck",
    "coverage_note": "Rejected for v1c v0c smoke because it predates the v0c dry-run completion/request at 2026-09-29T00:45:25Z. It remains useful as evidence that earlier artifacts were backed up.",
    "historical_validation64_bank_opened": False,
    "validation64_bank_opened": False,
    "sealed_test_accessed": False,
    "sealed_test_bank_opened": False,
    "new_rollouts": 0,
    "new_control_steps": 0,
    "new_training_episodes": 0,
    "new_gradient_steps": 0,
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_mtime(path: Path) -> Optional[dt.datetime]:
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
    except FileNotFoundError:
        return None


def max_time(values: Iterable[Optional[dt.datetime]]) -> dt.datetime:
    xs = [v for v in values if v is not None]
    if not xs:
        return dt.datetime.now(dt.timezone.utc)
    return max(xs)


def verify_completed_marker(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    if not path.exists():
        raise RuntimeError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        errors.append("completed marker lacks passed/hard_pass true")
    for flag in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if obj.get(flag) is not False:
            errors.append(f"{flag} is not false")
    for name, expected in sorted((obj.get("hashes") or {}).items()):
        p = ROOT / name
        if not p.exists():
            errors.append(f"hash reference missing: {name}")
            continue
        got = sha256(p)
        if got != expected:
            errors.append(f"hash mismatch for {name}: got {got}, expected {expected}")
    return obj, errors


def proof_package_ok(proof: Mapping[str, Any]) -> bool:
    packages = proof.get("packages_this_run") or []
    if proof.get("asset_sha256") or proof.get("release_asset_sha256") or proof.get("package_sha256"):
        return True
    if not isinstance(packages, list) or not packages:
        return False
    for pkg in packages:
        if not isinstance(pkg, Mapping):
            return False
        if not pkg.get("sha256") or not pkg.get("bytes") or not pkg.get("verification"):
            return False
    return True


def proof_time(proof: Mapping[str, Any]) -> Optional[dt.datetime]:
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        parsed = parse_time(proof.get(key))
        if parsed is not None:
            return parsed
    return None


def classify_proof(path: Path, min_required: dt.datetime) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": rel(path), "readable": False, "adequate_for_v1c_v0c_smoke": False}
    try:
        proof = read_json(path)
        out["readable"] = True
    except Exception as exc:
        out["error"] = repr(exc)
        return out
    t = proof_time(proof)
    verified = bool(proof.get("backup_verified") is True or proof.get("status") == "verified")
    try:
        remaining_ok = int(proof.get("remaining_changed_files", -1)) == 0
    except Exception:
        remaining_ok = False
    package_ok = proof_package_ok(proof)
    has_commit = bool(proof.get("commit"))
    time_ok = bool(t is not None and t >= min_required)
    out.update({
        "sha256": sha256(path),
        "time": None if t is None else t.isoformat(),
        "verified": verified,
        "remaining_changed_files": proof.get("remaining_changed_files"),
        "remaining_ok": remaining_ok,
        "package_ok": package_ok,
        "has_commit": has_commit,
        "commit": proof.get("commit"),
        "time_ok": time_ok,
        "min_required_utc": min_required.isoformat(),
        "adequate_for_v1c_v0c_smoke": bool(verified and remaining_ok and package_ok and has_commit and time_ok),
        "why_not": [],
    })
    if not verified:
        out["why_not"].append("not_verified")
    if not remaining_ok:
        out["why_not"].append("remaining_changed_files_not_zero")
    if not package_ok:
        out["why_not"].append("missing_release_or_package_sha")
    if not has_commit:
        out["why_not"].append("missing_commit")
    if not time_ok:
        out["why_not"].append("proof_time_predates_required_v0c_dryrun_or_unparseable")
    return out


def list_backup_proofs(min_required: dt.datetime) -> List[Dict[str, Any]]:
    proofs = []
    for p in sorted(BACKUP_DIR.glob("backup_proof_*.json")):
        proofs.append(classify_proof(p, min_required))
    proofs.sort(key=lambda r: (r.get("time") or "", r.get("path") or ""))
    return proofs


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if marker not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    latest = raw.get("latest_proof") or {}
    context = raw.get("supervisor_context_backup") or {}
    lines = [
        "# Vehicle stress-v1c v0c smoke backup gate recheck",
        "",
        f"UTC: `{raw['created_utc']}`. No simulation, no candidate resets, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Dry-run/source integrity",
        "",
        f"- v0c dry-run completed: `{raw['v0c_dryrun_completed']}`.",
        f"- Dry-run hash audit passed: `{raw['v0c_dryrun_hash_audit_passed']}`; errors: `{raw['v0c_dryrun_hash_audit_errors']}`.",
        f"- Required backup time for smoke: `>= {raw['min_required_backup_time_utc']}`.",
        "",
        "## Backup gate",
        "",
        f"- Adequate local backup proofs found: `{len(raw['adequate_proofs'])}`.",
        f"- Latest local proof: `{latest.get('path')}` at `{latest.get('time')}`; adequate=`{latest.get('adequate_for_v1c_v0c_smoke')}`; reasons=`{latest.get('why_not')}`.",
        f"- Supervisor-context backup at `{context.get('time')}` was transcribed but rejected for smoke because it predates the v0c dry-run/request; adequate=`{context.get('adequate_for_v1c_v0c_smoke')}`.",
        f"- Smoke may run now: `{raw['smoke_may_run_now']}`.",
        "",
        "## Decision",
        "",
        "Because no adequate post-v0c-dryrun proof exists, the v1c smoke remains blocked. The next iteration should first check for a verified external backup after the required time; if present, run the v0c smoke under the legacy interpreter. Do not train/refit or open validation/test banks.",
        "",
        f"Backup request: `{raw['backup_request_after_gate_recheck']}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    created = now_utc()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {
        "started_utc": created,
        "pid": os.getpid(),
        "method": "vehicle_stress_v1c_v0c_smoke_backup_gate_recheck_no_simulation",
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
    })

    # Preserve the latest supervisor context backup object but do not let it pass
    # the gate if it predates the dry-run.
    if not CONTEXT_BACKUP_RECORD.exists():
        write_json(CONTEXT_BACKUP_RECORD, SUPERVISOR_CONTEXT_BACKUP)

    required_paths = [V0C_SOURCE, V0C_COMPLETED, V0C_RAW, V0C_SUMMARY, V0C_REQUEST_BEFORE_SMOKE, V0B_OBJECTIVE_COMPLETED, PROTOCOL]
    missing = [rel(p) for p in required_paths if not p.exists()]
    if missing:
        write_json(OUT / "failure.json", {
            "failed_utc": now_utc(),
            "error": "required files missing",
            "missing": missing,
            "historical_validation64_bank_opened": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
        })
        return 2

    v0c_done, v0c_errors = verify_completed_marker(V0C_COMPLETED)
    request_before = read_json(V0C_REQUEST_BEFORE_SMOKE)
    protocol = read_json(PROTOCOL)
    context_proof_classified = classify_proof(CONTEXT_BACKUP_RECORD, dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    created_times = [
        parse_time(v0c_done.get("created_utc")),
        parse_time(request_before.get("requested_utc")),
        parse_time(protocol.get("created_utc")),
        file_mtime(V0C_SOURCE),
        file_mtime(V0C_COMPLETED),
        file_mtime(V0C_RAW),
        file_mtime(V0C_SUMMARY),
        file_mtime(V0C_REQUEST_BEFORE_SMOKE),
    ]
    min_required = max_time(created_times)
    proofs = list_backup_proofs(min_required)
    adequate = [p for p in proofs if p.get("adequate_for_v1c_v0c_smoke")]
    latest = proofs[-1] if proofs else None
    smoke_may_run = bool(not v0c_errors and adequate)
    selected_proof = adequate[-1] if adequate else None

    write_json(REQUEST, {
        "requested_utc": created,
        "reason": "backup v1c v0c hash-repair dry-run, source, protocol, v0b corrected objective diagnostic and this no-simulation gate recheck before any v1c smoke rollout",
        "backup_required_before_more_simulations": True,
        "min_required_backup_time_for_smoke_utc": min_required.isoformat(),
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [
            rel(V0C_SOURCE),
            rel(PROTOCOL),
            rel(V0B_OBJECTIVE_COMPLETED),
            rel(V0C_DRYRUN),
            rel(V0C_REQUEST_BEFORE_SMOKE),
            rel(OUT),
            rel(STATE),
            rel(REQUEST),
        ],
    })

    raw: Dict[str, Any] = {
        "created_utc": created,
        "method": "vehicle_stress_v1c_v0c_smoke_backup_gate_recheck_no_simulation",
        "classification": "metadata_only_backup_gate_no_validation_no_test_no_simulation",
        "v0c_source": rel(V0C_SOURCE),
        "v0c_source_sha256": sha256(V0C_SOURCE),
        "v0c_dryrun_completed": rel(V0C_COMPLETED),
        "v0c_dryrun_completed_sha256": sha256(V0C_COMPLETED),
        "v0c_dryrun_hash_audit_passed": not v0c_errors,
        "v0c_dryrun_hash_audit_errors": v0c_errors,
        "v0c_dryrun_created_utc": v0c_done.get("created_utc"),
        "protocol": rel(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "v0b_objective_completed": rel(V0B_OBJECTIVE_COMPLETED),
        "v0b_objective_completed_sha256": sha256(V0B_OBJECTIVE_COMPLETED),
        "request_before_smoke": rel(V0C_REQUEST_BEFORE_SMOKE),
        "request_before_smoke_sha256": sha256(V0C_REQUEST_BEFORE_SMOKE),
        "min_required_backup_time_utc": min_required.isoformat(),
        "proofs_scanned": proofs,
        "adequate_proofs": adequate,
        "latest_proof": latest,
        "selected_backup_proof_for_smoke": selected_proof,
        "supervisor_context_backup_transcribed": rel(CONTEXT_BACKUP_RECORD),
        "supervisor_context_backup": dict(context_proof_classified, adequate_for_v1c_v0c_smoke=False, why_not=["proof_time_predates_v0c_dryrun_request"]),
        "smoke_may_run_now": smoke_may_run,
        "next_action": "run v1c v0c smoke under legacy interpreter only after an adequate backup proof appears; otherwise remain blocked and do not simulate",
        "backup_request_after_gate_recheck": rel(REQUEST),
        "budgets_actual": {
            "new_rollouts": 0,
            "new_control_steps": 0,
            "candidate_pool_resets": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "access_flags": {
            "historical_validation64_bank_opened": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
        },
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        "# v1c v0c smoke backup gate recheck\n\n"
        f"UTC: {created}. No simulation/training/validation/test. v0c dry-run hash audit passed={not v0c_errors}. "
        f"Required backup time for smoke: {min_required.isoformat()}. Adequate proofs found={len(adequate)}. "
        f"Smoke may run now={smoke_may_run}. Next: await verified backup then run v0c smoke with legacy interpreter, or recheck gate if still absent.\n",
        encoding="utf-8",
    )
    marker = "vehicle-stress-v1c-v0c-smoke-backup-gate-recheck-20260929T0105Z"
    doc_block = f"""<!-- {marker} -->
## 2026-09-29 vehicle stress-v1c v0c smoke backup gate recheck

UTC: {created}. Metadata-only backup gate recheck completed with no simulations, no candidate resets, no training/refit, no validation64-bank access and no sealed-test access. v0c dry-run hash audit passed={not v0c_errors}. Required backup time for smoke is >= `{min_required.isoformat()}`. Adequate local proof count={len(adequate)}; latest proof `{None if latest is None else latest.get('path')}` at `{None if latest is None else latest.get('time')}` is adequate={False if latest is None else latest.get('adequate_for_v1c_v0c_smoke')}. The supervisor-context backup at 2026-09-29T00:43:29Z was transcribed for audit but rejected for v1c v0c smoke because it predates the dry-run/request at 00:45:25Z. Smoke remains blocked until a verified external backup appears after the required time. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`. Backup request: `{rel(REQUEST)}`.
"""
    append_docs(doc_block, marker)

    files = [
        OUT / "run_started.json",
        OUT / "raw.json",
        OUT / "summary.md",
        STATE,
        REQUEST,
        CONTEXT_BACKUP_RECORD,
        V0C_SOURCE,
        V0C_COMPLETED,
        V0C_RAW,
        V0C_SUMMARY,
        V0C_REQUEST_BEFORE_SMOKE,
        V0B_OBJECTIVE_COMPLETED,
        PROTOCOL,
    ]
    write_json(OUT / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "smoke_may_run_now": smoke_may_run,
        "adequate_backup_proofs": len(adequate),
        "selected_backup_proof_for_smoke": None if selected_proof is None else selected_proof.get("path"),
        "min_required_backup_time_utc": min_required.isoformat(),
        "v0c_dryrun_hash_audit_passed": not v0c_errors,
        "v0c_dryrun_hash_audit_errors": v0c_errors,
        "backup_request_after_gate_recheck": rel(REQUEST),
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "next_action": raw["next_action"],
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "smoke_may_run_now": smoke_may_run,
        "adequate_backup_proofs": len(adequate),
        "min_required_backup_time_utc": min_required.isoformat(),
        "backup_request_after_gate_recheck": rel(REQUEST),
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
