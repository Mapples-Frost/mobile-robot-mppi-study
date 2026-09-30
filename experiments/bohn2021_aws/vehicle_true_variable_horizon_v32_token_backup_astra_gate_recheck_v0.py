#!/usr/bin/env python3
"""Metadata-only backup/Astra gate recheck after v32 and API-token audit.

This script intentionally performs no simulations, control rollouts, training,
selector refits, validation-bank reads, or sealed-test reads.  It only inspects
existing handoff metadata, backup request/proof metadata, and aggregate API-token
usage from the supervisor SQLite usage JSON field.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FIRST_SUPERVISOR_EVENT = datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
CURRENT_REQUEST_EXPECTED = "v32-h12-supported-default-h35-diagnostic-20260930T051611Z"

# This claim is copied from the current supervisor/user context.  It is a
# materialized external-backup claim, not a new backup operation by this script.
SUPERVISOR_CONTEXT_BACKUP = {
    "time": "2026-09-30T05:23:56.296257+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "76e9cc71bca2ce0b4dee1c7d07a2610829884c3d",
    "changed_files": 50,
    "packages_this_run": [
        {
            "name": "20260930T052352_545f396a.tar.gz",
            "url": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/download/bohn-aws-evidence-20260926/20260930T052352_545f396a.tar.gz",
            "id": 600112111,
            "sha256": "2db0fd58b9068be7cf5d7711d05151d50c3b2444d1cb26f43b506fb2b591ed59",
            "bytes": 16592908,
            "verification": "github_server_sha256",
        }
    ],
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "tracked_files": 149059,
}

REQUESTS_TO_CHECK = [
    "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_H12_SUPPORTED_DEFAULT_H35_DIAGNOSTIC_20260930T051611Z.json",
    "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_NEXT_REVIEW_REFRESH_20260930T051840Z.json",
    "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_GATE_PREFLIGHT_STATE_20260930T052039Z.json",
    "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_API_TOKEN_USAGE_AUDIT_SOURCE_20260930T0522Z.json",
    "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_API_TOKEN_USAGE_AUDIT_20260930T052434Z.json",
]

V32_SUMMARY = "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v32_h12_supported_default_h35_diagnostic_v0_20260930T051611Z/summary.md"
API_TOKEN_SUMMARY = "research_artifacts/aws_diagnostics/api_token_usage_audit_v0_20260930T052434Z/summary.md"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat()


def parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def fmt_elapsed(delta) -> str:
    total = delta.total_seconds()
    days = int(total // 86400)
    rem = total - days * 86400
    hours = int(rem // 3600)
    rem -= hours * 3600
    minutes = int(rem // 60)
    seconds = rem - minutes * 60
    return f"{days}d {hours}h {minutes}m {seconds:.3f}s"


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
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def file_info(root: Path, rel: str) -> Dict[str, Any]:
    p = root / rel
    out: Dict[str, Any] = {"path": rel, "exists": p.exists(), "is_file": p.is_file()}
    if p.exists():
        st = p.stat()
        out.update(
            {
                "bytes": st.st_size,
                "mtime_utc": iso(datetime.fromtimestamp(st.st_mtime, timezone.utc)),
                "sha256": sha256_file(p) if p.is_file() else None,
            }
        )
    return out


def materialize_supervisor_backup(root: Path, now: datetime) -> Path:
    out_path = root / "research_artifacts/aws_backup_proofs/backup_proof_20260930T052356_from_user_context_after_v32_pre_token_audit.json"
    obj = {
        "proof_file_created_by": "GPT-5.5 executor from explicit current user/supervisor context; no backup operation was performed by this script",
        "proof_file_created_utc": iso(now),
        "backup": SUPERVISOR_CONTEXT_BACKUP,
        "backup_claim_valid": True,
        "interpretation": (
            "This materializes the prompt-supplied verified backup at 2026-09-30T05:23:56Z. "
            "It can only cover files whose local mtimes are not later than that backup time; "
            "the subsequent 05:24 API-token audit outputs still require a later backup."
        ),
        "access_flags": {
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
        },
        "budgets_actual": {
            "new_simulation_episodes": 0,
            "new_control_steps": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
    }
    # Do not rewrite if the same proof was already materialized; preserve old file.
    if not out_path.exists():
        write_json(out_path, obj)
    return out_path


def backup_candidate_from_obj(path_label: str, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    backup = obj.get("backup") if isinstance(obj.get("backup"), dict) else obj
    time_s = backup.get("time") or obj.get("time") or backup.get("timestamp") or obj.get("timestamp")
    dt = parse_iso(str(time_s)) if time_s is not None else None
    if not dt:
        return None
    status = backup.get("status") or obj.get("status")
    remaining = backup.get("remaining_changed_files", obj.get("remaining_changed_files"))
    packages = backup.get("packages_this_run") or obj.get("packages_this_run") or []
    package_sha256 = None
    package_ok = False
    if isinstance(packages, list) and packages:
        first = packages[0]
        if isinstance(first, dict):
            package_sha256 = first.get("sha256")
            package_ok = bool(first.get("sha256")) and first.get("verification") in ("github_server_sha256", "download_sha256", "sha256")
    return {
        "path": path_label,
        "time": iso(dt),
        "status": status,
        "remaining_changed_files": remaining,
        "commit": backup.get("commit") or obj.get("commit"),
        "package_sha256": package_sha256,
        "package_ok": package_ok,
    }


def load_backup_candidates(root: Path, materialized: Path) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for rel_or_abs in glob.glob(str(root / "research_artifacts/aws_backup_proofs/backup_proof_*.json")):
        p = Path(rel_or_abs)
        obj = read_json(p)
        if not obj:
            continue
        rel = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
        cand = backup_candidate_from_obj(rel, obj)
        if cand:
            candidates.append(cand)
    # Ensure the prompt-supplied context is represented even if file parsing failed.
    extra = backup_candidate_from_obj(str(materialized.relative_to(root)), {"backup": SUPERVISOR_CONTEXT_BACKUP})
    if extra and not any(c.get("time") == extra.get("time") and c.get("commit") == extra.get("commit") for c in candidates):
        candidates.append(extra)
    candidates.sort(key=lambda x: x.get("time", ""), reverse=True)
    return candidates


def check_request_against_candidate(root: Path, request_rel: str, cand: Dict[str, Any]) -> Dict[str, Any]:
    req_path = root / request_rel
    req = read_json(req_path) or {}
    must_cover = req.get("must_cover") if isinstance(req.get("must_cover"), list) else []
    dt = parse_iso(str(cand.get("time", "")))
    infos = [file_info(root, str(x)) for x in must_cover]
    missing = [i["path"] for i in infos if not i.get("exists") or not i.get("is_file")]
    after = []
    if dt:
        for i in infos:
            mdt = parse_iso(str(i.get("mtime_utc", "")))
            if mdt and mdt > dt:
                after.append({"path": i["path"], "mtime_utc": i["mtime_utc"]})
    status_ok = cand.get("status") == "verified"
    remaining_ok = cand.get("remaining_changed_files") == 0
    package_ok = bool(cand.get("package_sha256"))
    clear = bool(dt and status_ok and remaining_ok and package_ok and not missing and not after)
    return {
        "request": request_rel,
        "request_exists": req_path.exists(),
        "requested_utc": req.get("requested_utc"),
        "must_cover_count": len(must_cover),
        "candidate": cand,
        "missing_count": len(missing),
        "missing": missing,
        "files_after_backup_count": len(after),
        "files_after_backup_sample": after[:20],
        "clear": clear,
    }


def best_backup_check(root: Path, request_rel: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks = [check_request_against_candidate(root, request_rel, c) for c in candidates]
    clear_checks = [c for c in checks if c.get("clear")]
    if clear_checks:
        # Candidates are sorted newest first; keep newest clear.
        best = clear_checks[0]
    elif checks:
        best = checks[0]
    else:
        best = {"request": request_rel, "request_exists": (root / request_rel).exists(), "clear": False, "reason": "no_backup_candidates"}
    return {"best": best, "checked_candidates": len(checks)}


def audit_token_usage() -> Dict[str, Any]:
    db = Path("/data/openai-agent/state/research.sqlite")
    if not db.exists():
        return {"available": False, "reason": "sqlite_not_found", "path": str(db), "selected_text": "unknown"}
    try:
        conn = sqlite3.connect(str(db))
        try:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(calls)").fetchall()]
            if "usage" not in cols:
                return {"available": False, "reason": "calls_usage_column_absent", "path": str(db), "selected_text": "unknown"}
            total = 0
            rows = 0
            parsed = 0
            for (usage,) in conn.execute("SELECT usage FROM calls"):
                rows += 1
                if usage is None:
                    continue
                try:
                    obj = json.loads(usage) if isinstance(usage, str) else usage
                except Exception:
                    continue
                if isinstance(obj, dict) and isinstance(obj.get("total_tokens"), (int, float)):
                    total += int(obj["total_tokens"])
                    parsed += 1
            return {
                "available": True,
                "path": str(db),
                "source": "calls.usage.total_tokens",
                "basis": "json_field_sum",
                "row_count": rows,
                "json_rows_with_total_tokens": parsed,
                "total_tokens": total,
                "selected_text": f"{total:,} ({total/1e6:.3f}M) from calls.usage.total_tokens",
            }
        finally:
            conn.close()
    except Exception as e:
        return {"available": False, "reason": f"sqlite_error:{type(e).__name__}:{e}", "path": str(db), "selected_text": "unknown"}


def audit_astra(root: Path) -> Dict[str, Any]:
    req_path = root / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
    req = read_json(req_path) or {}
    current_id = req.get("request_id")
    ready_path = root / "docs/bohn2021_takeover/astra_reviews/ANALYSIS_READY.json"
    latest_path = root / "docs/bohn2021_takeover/astra_reviews/LATEST.md"
    ready = read_json(ready_path) if ready_path.exists() else None
    ids: List[str] = []
    if isinstance(ready, dict):
        for k in ("request_id", "for_request_id", "covers_request_id", "current_request_id", "supersedes_request_id"):
            v = ready.get(k)
            if isinstance(v, str):
                ids.append(v)
            elif isinstance(v, list):
                ids.extend([x for x in v if isinstance(x, str)])
    latest_text = ""
    if latest_path.exists():
        try:
            latest_text = latest_path.read_text(encoding="utf-8")[:2000]
        except Exception:
            latest_text = ""
    return {
        "next_review_request_path": str(req_path.relative_to(root)),
        "current_request_id": current_id,
        "expected_request_id": CURRENT_REQUEST_EXPECTED,
        "current_is_expected": current_id == CURRENT_REQUEST_EXPECTED,
        "analysis_ready_exists": ready_path.exists(),
        "analysis_ready_path": str(ready_path.relative_to(root)),
        "analysis_ready_ids": ids,
        "analysis_ready_matches_current": bool(current_id and current_id in ids),
        "latest_md_path": str(latest_path.relative_to(root)),
        "latest_md_mentions_old_20260929_report": "20260929T153837Z" in latest_text,
    }


def summarize_v32_and_token(root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, rel in [("v32_summary", V32_SUMMARY), ("api_token_summary", API_TOKEN_SUMMARY)]:
        p = root / rel
        info = file_info(root, rel)
        text = ""
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8")[:4000]
            except Exception as e:
                text = f"<read_error {type(e).__name__}: {e}>"
        out[key] = {"file": info, "excerpt": text}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default=None, help="Stable output suffix, e.g. 20260930T0530Z")
    args = parser.parse_args()

    root = repo_root()
    now = utc_now()
    stamp = args.stamp or now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / f"research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v32_token_backup_astra_gate_recheck_v0_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    run_started_path = out_dir / "run_started.json"
    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    completed_path = out_dir / "completed.json"
    state_path = root / f"research_artifacts/aws_state/continue_state_{stamp}_after_v32_token_backup_astra_gate_recheck.md"
    backup_request_path = root / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_TOKEN_BACKUP_ASTRA_GATE_RECHECK_{stamp}.json"

    access_flags = {
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "validation64_episodes": 0,
        "sealed_test_accessed": False,
        "sealed_test_episodes": 0,
    }
    write_json(run_started_path, {"created_utc": iso(now), "classification": "metadata_only_no_sim_no_validation_no_test_no_training_no_refit", **access_flags})

    token = audit_token_usage()
    elapsed_text = fmt_elapsed(now - FIRST_SUPERVISOR_EVENT)
    materialized_proof = materialize_supervisor_backup(root, now)
    candidates = load_backup_candidates(root, materialized_proof)
    backup_checks = {rel: best_backup_check(root, rel, candidates) for rel in REQUESTS_TO_CHECK}
    astra = audit_astra(root)
    evidence = summarize_v32_and_token(root)

    token_audit_request = "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_API_TOKEN_USAGE_AUDIT_20260930T052434Z.json"
    latest_token_backup_clear = bool(backup_checks.get(token_audit_request, {}).get("best", {}).get("clear"))
    astra_ready_for_current = bool(astra.get("analysis_ready_matches_current"))

    high_level = {
        "created_utc": iso(now),
        "classification": "metadata_only_v32_token_backup_astra_gate_recheck_no_sim_no_validation_no_test_no_training_no_refit",
        "service_lifetime_elapsed": elapsed_text,
        "server_api_total_tokens": token.get("selected_text", "unknown"),
        "desktop_conversation_tokens_excluded": True,
        "materialized_supervisor_backup_proof": str(materialized_proof.relative_to(root)),
        "backup_latest_token_audit_request_clear": latest_token_backup_clear,
        "astra_ready_for_current_v32_request": astra_ready_for_current,
        "unique_science_gate_clear": bool(latest_token_backup_clear and astra_ready_for_current),
        "access_flags": access_flags,
        "decision": "Do not start fresh labels/refit/training/scenario/validation/final-test work unless both current backup and matching Astra gates are clear.",
    }

    raw = {
        **high_level,
        "token_audit": token,
        "astra": astra,
        "backup_candidates_count": len(candidates),
        "latest_backup_candidate": candidates[0] if candidates else None,
        "backup_checks": backup_checks,
        "existing_evidence_excerpts": evidence,
    }
    write_json(raw_path, raw)

    # Human-readable summary and durable state.
    lines: List[str] = []
    lines.append("# v32/token backup + Astra gate recheck\n")
    lines.append(f"UTC: `{iso(now)}`. Metadata-only operational check; no simulations, no control steps, no selector refit, no training, no validation64 bank access and no sealed-test access.\n")
    lines.append("## Required status-line values")
    lines.append(f"- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `{elapsed_text}`.")
    lines.append(f"- Cumulative server API total_tokens from research.sqlite: `{token.get('selected_text', 'unknown')}`; desktop conversation tokens excluded.\n")
    lines.append("## Gate findings")
    lines.append(f"- Materialized prompt-supplied verified backup proof: `{materialized_proof.relative_to(root)}`.")
    lines.append(f"- Current Astra request: `{astra.get('current_request_id')}`; ANALYSIS_READY exists=`{astra.get('analysis_ready_exists')}`, matches current=`{astra.get('analysis_ready_matches_current')}`.")
    for rel in REQUESTS_TO_CHECK:
        best = backup_checks[rel]["best"]
        lines.append(
            f"- Backup request `{rel}`: clear=`{best.get('clear')}`; "
            f"best_candidate_time=`{best.get('candidate', {}).get('time')}`; "
            f"missing={best.get('missing_count')}; files_after_backup={best.get('files_after_backup_count')}."
        )
    lines.append("\n## Current decision")
    if latest_token_backup_clear and astra_ready_for_current:
        lines.append("- Both token-audit backup and current Astra gates appear clear; next executor cycle should read the Astra report and implement its selected plan before starting work.")
    else:
        lines.append("- Unique scientific simulation/refit/training/validation/final-test work remains blocked: "
                     f"token-audit backup clear=`{latest_token_backup_clear}`, Astra current report ready=`{astra_ready_for_current}`.")
        lines.append("- Continue only reversible integrity/preparation until a later verified backup covers this recheck/token-audit output and a matching/superseding Astra report is available.")
    lines.append("\n## Preserved scientific constraints")
    lines.append("- v29/v30b/v31/v32 remain opened-development diagnostics only, not validation/test/reproduction/deployed-speed evidence.")
    lines.append("- v32 suggests an H12-vs-non-H12 signal but leaves the single H15-family residual unresolved; Astra remains scientific lead for branch selection.")
    lines.append("- Sealed final test remains unopened and unauthorized.\n")
    summary = "\n".join(lines)
    summary_path.write_text(summary, encoding="utf-8")
    state_path.write_text(summary, encoding="utf-8")

    # Append concise entries to mutable coordination logs.
    log_entry = f"""

<!-- vehicle_true_variable_horizon_v32_token_backup_astra_gate_recheck_v0-{stamp} -->
## v32/token backup + Astra gate recheck

Updated by GPT-5.5 executor at `{iso(now)}`. Metadata-only; no simulations, no control steps, no selector refits, no training, no validation64 access and no sealed-test access.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| Astra v32 direction request | pending | NEXT_REVIEW_REQUEST `{astra.get('current_request_id')}`; ANALYSIS_READY exists=`{astra.get('analysis_ready_exists')}`, matches/supersedes current=`{astra.get('analysis_ready_matches_current')}`; `LATEST.md` still old-report marker=`{astra.get('latest_md_mentions_old_20260929_report')}`. | Await/read a matching or superseding Astra report before choosing acquisition/refit/training/scenario branch. |
| `A12_registry_backup_schema_contract` | accepted; current latest prompt-supplied backup materialized, but latest token-audit outputs still not covered | Materialized proof `{materialized_proof.relative_to(root)}` from backup time `2026-09-30T05:23:56.296257+00:00`, commit `76e9cc71bca2ce0b4dee1c7d07a2610829884c3d`, package SHA256 `2db0fd58b9068be7cf5d7711d05151d50c3b2444d1cb26f43b506fb2b591ed59`. Token-audit request clear=`{latest_token_backup_clear}`. | New request `{backup_request_path.relative_to(root)}` must be externally backed up before unique scientific work. |
| v32 opened-development signal | preserved; interpretation deferred to Astra | v32 summary path `{V32_SUMMARY}`; API token audit summary `{API_TOKEN_SUMMARY}`; no new scientific labels or rollouts in this recheck. | Do not claim validation/test/reproduction/deployable-selector success; next branch waits for Astra. |
"""
    append_text(root / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", log_entry)
    append_text(root / "RESEARCH_LOG.md", "\n" + summary + "\n")
    append_text(root / "STATUS.md", "\n" + summary + "\n")

    must_cover = [
        str(Path(__file__).relative_to(root)),
        str(run_started_path.relative_to(root)),
        str(summary_path.relative_to(root)),
        str(raw_path.relative_to(root)),
        str(completed_path.relative_to(root)),
        str(backup_request_path.relative_to(root)),
        str(state_path.relative_to(root)),
        str(materialized_proof.relative_to(root)),
        "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md",
        "STATUS.md",
        "RESEARCH_LOG.md",
        "EXPERIMENT_REGISTRY.csv",
    ]
    backup_request = {
        "requested_utc": iso(now),
        "reason": "Back up metadata-only v32/token backup and Astra gate recheck outputs, including materialized supervisor backup proof and log updates.",
        "must_cover": must_cover,
        **access_flags,
    }
    write_json(backup_request_path, backup_request)

    completed = {
        **high_level,
        "status": "complete",
        "summary": str(summary_path.relative_to(root)),
        "raw": str(raw_path.relative_to(root)),
        "continue_state": str(state_path.relative_to(root)),
        "backup_request": str(backup_request_path.relative_to(root)),
        "hard_pass": True,
        "hashes": {rel: sha256_file(root / rel) for rel in must_cover if (root / rel).exists() and (root / rel).is_file()},
    }
    write_json(completed_path, completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
