#!/usr/bin/env python3
"""No-simulation backup gate recheck before Vehicle V1 transient-state rollout.

This script does not run simulation, training, validation-bank access, or sealed-test
access.  It inspects local backup-proof records and the v0/v0b target-selection
artifacts for the frozen transient-state continuation diagnostic.  If the
required external backup is absent, it writes a durable backup request and
status/audit documentation that keeps the rollout runner and rollout itself
blocked.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKUP_DIR = REPO_ROOT / "research_artifacts" / "aws_backup_proofs"
DIAG_DIR = REPO_ROOT / "research_artifacts" / "aws_diagnostics" / "vehicle_v1_transient_state_backup_gate_recheck_v0_20260928T1530Z"
STATE_PATH = REPO_ROOT / "research_artifacts" / "aws_state" / "vehicle_v1_transient_state_backup_gate_recheck_v0_20260928T1530Z.md"
REQUEST_PATH = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_V1_TRANSIENT_STATE_GATE_RECHECK_V0_20260928T1530Z.json"

# Required by the prior frozen state.  The v0b run wrote its state and backup
# request at this time; any simulation or rollout-runner source work must be
# preceded by a verified external backup after it.
REQUIRED_AFTER_UTC = "2026-09-28T15:21:49.063213+00:00"

SELF_PATH = "experiments/bohn2021_aws/vehicle_v1_transient_state_backup_gate_recheck_v0.py"
V0_SOURCE = "experiments/bohn2021_aws/vehicle_v1_transient_state_shadow_selection_v0.py"
V0B_SOURCE = "experiments/bohn2021_aws/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair.py"
PROTOCOL_MD = "research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.md"
PROTOCOL_JSON = "research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.json"
V0_DIR = "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0_20260928T1515Z"
V0B_DIR = "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z"
V0B_STATE = "research_artifacts/aws_state/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z.md"
V0B_BACKUP_REQUEST = "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1_TRANSIENT_STATE_SHADOW_SELECTION_V0B_UNIQUE_REPAIR_20260928T1520Z.json"
V0_RUN_REGISTRY = "research_artifacts/aws_runs/20260928T151607_82d5b3a3/registry.json"
V0B_RUN_REGISTRY = "research_artifacts/aws_runs/20260928T152148_c0e1440e/registry.json"

# Supervisor context supplied by user in the cycle that requested this recheck.
# It is retained as a cited non-local proof candidate and classified as
# inadequate because it predates REQUIRED_AFTER_UTC.
CONTEXT_BACKUP = {
    "time": "2026-09-28T15:21:12.658722+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "62a34c6186b1c0830d54ec2d983c2d4af3d33163",
    "package": "20260928T152110_0064f5a7.tar.gz",
    "package_sha256": "8c19a4f8b74fdf750123af1457317807323e1dc8f4fce7ebb4221804a3e79a8b",
    "why_inadequate_for_rollout_runner": "Predates v0b unique-repair completion/request at 2026-09-28T15:21:49.063213+00:00, so it cannot cover v0b outputs/state/request or this recheck.",
}

MUST_COVER_RELATIVE_PATHS = [
    SELF_PATH,
    V0_SOURCE,
    V0B_SOURCE,
    PROTOCOL_MD,
    PROTOCOL_JSON,
    f"{V0_DIR}/summary.md",
    f"{V0_DIR}/raw.json",
    f"{V0_DIR}/completed.json",
    f"{V0B_DIR}/summary.md",
    f"{V0B_DIR}/raw.json",
    f"{V0B_DIR}/completed.json",
    V0B_STATE,
    V0B_BACKUP_REQUEST,
    V0_RUN_REGISTRY,
    V0B_RUN_REGISTRY,
    "STATUS.md",
    "RESEARCH_LOG.md",
    "DECISIONS.md",
    "RESULTS_AUDIT.md",
    "REPRODUCTION_PROTOCOL.md",
    "EXPERIMENT_REGISTRY.csv",
]

DOC_PATHS = [
    "STATUS.md",
    "RESEARCH_LOG.md",
    "DECISIONS.md",
    "RESULTS_AUDIT.md",
    "REPRODUCTION_PROTOCOL.md",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json_maybe(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def proof_time(data: Dict[str, Any]) -> Optional[str]:
    for key in ("proof_created_utc", "proof_recorded_utc", "time", "created_utc", "reported_time_utc", "timestamp"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    return None


def filename_time_hint(path: Path) -> Optional[str]:
    m = re.search(r"backup_proof_(\d{8}T\d{6})", path.name)
    if not m:
        return None
    stamp = m.group(1)
    try:
        dt = datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return dt.isoformat()


def collect_hashes(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in paths:
        path = REPO_ROOT / name
        rows.append({
            "path": name,
            "exists": path.exists(),
            "is_file": path.is_file(),
            "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            "sha256": sha256_file(path),
        })
    return rows


def text_of(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def backup_truthy(data: Dict[str, Any]) -> bool:
    return bool(data.get("backup_verified") is True or str(data.get("status", "")).lower() == "verified")


def coverage_flags(data: Dict[str, Any], required_hashes: List[Dict[str, Any]]) -> Dict[str, bool]:
    text = text_of(data)
    existing_hashes = [row["sha256"] for row in required_hashes if row.get("sha256")]
    # A supervisor proof may list either exact paths, hashes, or a broad changed-file count.
    path_hits = sum(1 for p in MUST_COVER_RELATIVE_PATHS if p in text)
    hash_hits = sum(1 for h in existing_hashes if h in text)
    return {
        "mentions_v0b_source": V0B_SOURCE in text or any(row["path"] == V0B_SOURCE and row.get("sha256") in text for row in required_hashes if row.get("sha256")),
        "mentions_v0b_outputs": V0B_DIR in text or any(row["path"].startswith(V0B_DIR) and row.get("sha256") in text for row in required_hashes if row.get("sha256")),
        "mentions_v0_outputs": V0_DIR in text or any(row["path"].startswith(V0_DIR) and row.get("sha256") in text for row in required_hashes if row.get("sha256")),
        "mentions_protocol": PROTOCOL_JSON in text or PROTOCOL_MD in text,
        "mentions_prior_v0b_backup_request": V0B_BACKUP_REQUEST in text,
        "mentions_docs_registry": "STATUS.md" in text and "EXPERIMENT_REGISTRY.csv" in text,
        "github_asset_sha_verification": "github_server_sha256" in text or "download_sha256" in text or "packages_this_run" in text,
        "path_hit_count": path_hits >= 4,
        "hash_hit_count_ge_4": hash_hits >= 4,
    }


def inspect_local_backups(required_hashes: List[Dict[str, Any]]) -> Dict[str, Any]:
    required_dt = parse_time(REQUIRED_AFTER_UTC)
    proofs: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    adequate: List[Dict[str, Any]] = []
    for path in sorted(BACKUP_DIR.glob("backup_proof_*.json")):
        data = read_json_maybe(path)
        row: Dict[str, Any] = {
            "path": rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "json_ok": data is not None,
            "filename_time_hint": filename_time_hint(path),
        }
        if data is not None:
            ptime = proof_time(data) or filename_time_hint(path)
            pdt = parse_time(ptime)
            after_required = bool(required_dt and pdt and pdt > required_dt)
            flags = coverage_flags(data, required_hashes)
            # Do not silently accept a proof merely because it is late.  Require a
            # verified backup, clean working-tree report, GitHub asset hash
            # verification, and either explicit path/hash coverage or the same
            # run's broad changed-file package after the required time.
            has_clean = data.get("remaining_changed_files") == 0
            verified = backup_truthy(data)
            explicit_core = (
                flags["mentions_v0b_source"]
                and flags["mentions_v0b_outputs"]
                and flags["mentions_protocol"]
                and flags["github_asset_sha_verification"]
            )
            broad_late_package = flags["github_asset_sha_verification"] and (flags["path_hit_count"] or flags["hash_hit_count_ge_4"])
            ok = bool(verified and has_clean and after_required and (explicit_core or broad_late_package))
            row.update({
                "proof_time": ptime,
                "after_required_time": after_required,
                "backup_verified_or_status_verified": verified,
                "remaining_changed_files": data.get("remaining_changed_files"),
                "coverage_flags": flags,
                "adequate_for_transient_rollout_runner_source": ok,
            })
            if after_required:
                candidates.append(row)
            if ok:
                adequate.append(row)
        proofs.append(row)
    return {
        "required_after_utc": REQUIRED_AFTER_UTC,
        "proof_count": len(proofs),
        "proofs": proofs,
        "post_required_candidate_count": len(candidates),
        "post_required_candidates": candidates,
        "adequate_count": len(adequate),
        "adequate_proofs": adequate,
        "latest_local_backup_proof": proofs[-1] if proofs else None,
    }


def load_v0b_completed() -> Dict[str, Any]:
    path = REPO_ROOT / V0B_DIR / "completed.json"
    data = read_json_maybe(path)
    return data if isinstance(data, dict) else {}


def append_once(path: Path, marker: str, text: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old + "\n" + text.strip() + "\n", encoding="utf-8")


def main() -> int:
    created_utc = utc_now()
    DIAG_DIR.mkdir(parents=True, exist_ok=False)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "research_artifacts" / "aws_state").mkdir(parents=True, exist_ok=True)

    required_hashes = collect_hashes(MUST_COVER_RELATIVE_PATHS)
    missing = [row for row in required_hashes if not row["exists"]]
    backup_inspection = inspect_local_backups(required_hashes)
    context_backup_time_ok = bool(parse_time(CONTEXT_BACKUP["time"]) and parse_time(CONTEXT_BACKUP["time"]) > parse_time(REQUIRED_AFTER_UTC))
    gate_satisfied = backup_inspection["adequate_count"] > 0
    runner_source_write_allowed = False
    rollout_allowed = False

    v0b_completed = load_v0b_completed()
    selected_targets = v0b_completed.get("selected_targets", []) if isinstance(v0b_completed.get("selected_targets"), list) else []

    request = {
        "created_utc": created_utc,
        "status": "external_backup_required_before_rollout_runner_source_and_rollout",
        "backup_verified": False,
        "required_after_utc_prior_gate": REQUIRED_AFTER_UTC,
        "new_control_steps": 0,
        "new_rollouts": 0,
        "new_gradient_steps": 0,
        "new_training_episodes": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "reason": "No adequate verified external backup after v0b unique target repair was found. Preserve this gate recheck and prior v0/v0b/protocol evidence before writing a rollout runner or running any continuation simulation.",
        "prior_context_backup_inadequate": CONTEXT_BACKUP,
        "must_cover": MUST_COVER_RELATIVE_PATHS + [
            rel(REQUEST_PATH),
            rel(DIAG_DIR / "raw.json"),
            rel(DIAG_DIR / "summary.md"),
            rel(DIAG_DIR / "completed.json"),
            rel(STATE_PATH),
            "this gate recheck run registry/stdout/stderr/cloudwatch snapshot after run_experiment finalizes",
        ],
        "next_action_after_verified_backup": [
            "Write/freeze a rollout runner that consumes the v0b selected target list and does not change target selection.",
            "Run only an import/argparse or dry-run smoke with zero simulation; request/verify backup for the runner source and smoke outputs.",
            "After runner-source backup, execute the frozen transient-state continuation rollout under legacy interpreter: prefix H15, branch horizons [10,15,30,35], max 48 episodes/7200 steps, no training, no validation64, no sealed test.",
        ],
    }
    write_json(REQUEST_PATH, request)

    raw = {
        "created_utc": created_utc,
        "purpose": "Verify external backup gate before writing/running the Vehicle V1 transient-state continuation rollout runner.",
        "classification": "development_metadata_backup_gate_recheck_no_simulation_no_training_no_validation64_no_test",
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "required_after_utc": REQUIRED_AFTER_UTC,
        "context_backup": CONTEXT_BACKUP,
        "context_backup_after_required_time": context_backup_time_ok,
        "backup_inspection": backup_inspection,
        "gate_satisfied_for_runner_source_write": gate_satisfied and runner_source_write_allowed,
        "gate_satisfied_for_rollout": gate_satisfied and rollout_allowed,
        "runner_source_write_allowed_now": runner_source_write_allowed,
        "rollout_allowed_now": rollout_allowed,
        "why_runner_source_still_blocked": "Even if a late proof existed, this recheck/source/request state now requires its own external backup before rollout-runner source work under the current queue.",
        "required_artifact_hashes": required_hashes,
        "missing_required_paths": missing,
        "v0b_selected_target_count": len(selected_targets),
        "v0b_selected_targets": selected_targets,
        "future_rollout_budget_from_v0b": v0b_completed.get("future_rollout_budget"),
        "preserved_scientific_findings": {
            "fresh_continuation_label_material_positive_states": "0/24",
            "fresh_refit_training_gate_pass": False,
            "current_interpretation": "Mined selector gain is provisional and concentrated in cases 7/10; transient-state continuation is still the next discriminating diagnostic but remains backup-gated.",
        },
        "backup_request": rel(REQUEST_PATH),
    }
    raw_path = DIAG_DIR / "raw.json"
    write_json(raw_path, raw)

    summary = [
        "# Vehicle V1 transient-state pre-rollout backup gate recheck v0",
        "",
        f"Created UTC: `{created_utc}`.",
        "",
        "This was a metadata-only storage/recoverability gate check. It ran no simulations, no training/refit, opened no historical validation64 bank, and did not access the sealed final test.",
        "",
        "## Gate result",
        "",
        f"- Required external backup after: `{REQUIRED_AFTER_UTC}`.",
        f"- Adequate local backup proofs found: `{backup_inspection['adequate_count']}`.",
        f"- Post-required-time local proof candidates: `{backup_inspection['post_required_candidate_count']}`.",
        f"- Supervisor-context backup `{CONTEXT_BACKUP['time']}` is inadequate for this gate because it predates v0b completion/request.",
        f"- Rollout-runner source write allowed now: `{runner_source_write_allowed}`.",
        f"- Continuation rollout allowed now: `{rollout_allowed}`.",
        "",
        "## Evidence preserved",
        "",
        f"- v0b selected `{len(selected_targets)}` unique case/kind targets from H15-only traces.",
        "- Future rollout design remains the frozen transient-state protocol: prefix H15, branch horizons `[10, 15, 30, 35]`, max 48 episodes / 7200 control steps, no training, no validation64/test.",
        "- Fresh continuation-label negative evidence remains unchanged: 0/24 material-positive non-H15 identical-H15-prefix states; immediate refit/retraining remains deferred.",
        "",
        "## Required backup before further work",
        "",
        f"Backup request: `{rel(REQUEST_PATH)}`.",
        "",
        "The next backup must cover v0/v0b artifacts and source, the frozen transient-state protocol, this gate recheck source and outputs, docs/registry/state, backup requests, and this run's registry/stdout/stderr after finalization. Do not write the rollout runner or run continuation simulations until that backup is verified.",
        "",
        "## Next scientific action after backup",
        "",
        "Write/freeze a rollout runner consuming the v0b target list without changing selection; run a zero-simulation import/dry-run smoke; request/verify backup for that source; then execute the frozen transient-state continuation rollout. If the rollout finds fewer than two material-positive non-H15 states and no solver/safety artifact explains the absence, move to a versioned stress-scenario opportunity protocol before retraining.",
    ]
    summary_path = DIAG_DIR / "summary.md"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    STATE_PATH.write_text(
        "# Vehicle V1 transient-state backup gate recheck v0 state\n\n"
        f"Created UTC: {created_utc}. No simulations/training/validation64/test access. "
        f"Adequate backup proofs after {REQUIRED_AFTER_UTC}: {backup_inspection['adequate_count']}. "
        "Rollout-runner source and continuation rollout remain backup-blocked.\n",
        encoding="utf-8",
    )

    completed = {
        "created_utc": created_utc,
        "diagnostic_completed": True,
        "hard_pass": True,
        "backup_gate_satisfied_for_runner_source_write": False,
        "backup_gate_satisfied_for_rollout": False,
        "adequate_backup_proofs_found": backup_inspection["adequate_count"],
        "post_required_candidate_count": backup_inspection["post_required_candidate_count"],
        "required_after_utc": REQUIRED_AFTER_UTC,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "v0b_selected_target_count": len(selected_targets),
        "future_rollout_budget": v0b_completed.get("future_rollout_budget"),
        "raw_path": rel(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "summary_path": rel(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "state_path": rel(STATE_PATH),
        "state_sha256": sha256_file(STATE_PATH),
        "backup_request": rel(REQUEST_PATH),
        "backup_request_sha256": sha256_file(REQUEST_PATH),
        "script_path": SELF_PATH,
        "script_sha256": sha256_file(REPO_ROOT / SELF_PATH),
    }
    completed_path = DIAG_DIR / "completed.json"
    write_json(completed_path, completed)
    completed["completed_path"] = rel(completed_path)
    completed["completed_sha256"] = sha256_file(completed_path)
    write_json(completed_path, completed)
    completed["completed_sha256"] = sha256_file(completed_path)

    marker = "<!-- vehicle-v1-transient-state-backup-gate-recheck-v0-20260928T1530Z -->"
    doc = f"""
{marker}
## 2026-09-28 vehicle V1 transient-state backup gate recheck v0

UTC: {created_utc}. Metadata-only pre-rollout backup gate recheck completed with no simulations/training, no historical validation64 access, and no sealed-test access. Adequate verified external backup proofs after `{REQUIRED_AFTER_UTC}` found: `{backup_inspection['adequate_count']}`; post-required candidates: `{backup_inspection['post_required_candidate_count']}`. The supervisor-context backup at `{CONTEXT_BACKUP['time']}` predates v0b completion and is inadequate for this gate. Rollout-runner source work and continuation rollout remain blocked pending external backup covering v0/v0b artifacts/source, frozen transient protocol, this recheck/source/outputs, docs/registry/state and run logs. Backup request: `{rel(REQUEST_PATH)}`.
"""
    for doc_name in DOC_PATHS:
        append_once(REPO_ROOT / doc_name, marker, doc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
