#!/usr/bin/env python3
"""Current S-TC2H8 backup-status recheck after Asset-422/timeout failures.

Temporary solo GPT-5.5 operational diagnostic. This script reads only sanitized
canonical supervisor state under /data/openai-agent/state and public GitHub
release metadata. It does not import MPC/controller code, run simulations, train,
refit, open validation64, or access sealed/final tests. Its purpose is to decide
whether the backup gate that blocks the S-TC2H8 v0i microcontinuation is still
blocked after the latest supervisor backup attempt, and to persist exact evidence
for the next iteration.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

CANON_BASE = Path("/data/openai-agent")
ROOT = CANON_BASE / "mobile-robot-mppi-study"
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[2]
STATE = CANON_BASE / "state"
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
try:
    import execution_contract  # type: ignore
except Exception:
    execution_contract = None  # type: ignore

NAME = "backup_failure_status_capture_s_tc2h8_current_status_v0c"
TASK_ID = "S-BACKUP-RECHECK-STC2H8-current-status-v0c"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "research_artifacts/aws_diagnostics" / f"{NAME}_{STAMP}"
STATE_MD = ROOT / "research_artifacts/aws_state" / f"continue_state_{STAMP}_after_s_tc2h8_backup_current_status_v0c.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_RETRY_AFTER_S_TC2H8_CURRENT_STATUS_V0C_{STAMP}.json"
PROOF_IF_VERIFIED = BACKUP_DIR / f"BACKUP_VERIFIED_FROM_STATUS_AFTER_S_TC2H8_CURRENT_STATUS_V0C_{STAMP}.json"
BACKUP_STATUS = STATE / "backup_status.json"
BACKUP_RECEIPTS = STATE / "backup_receipts.jsonl"
BACKUP_INDEX = STATE / "backup_index.sqlite"
BACKUP_STAGING = CANON_BASE / "backup_staging"
RESEARCH_DB = STATE / "research.sqlite"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
REPO = "Mapples-Frost/mobile-robot-mppi-study"
DEFAULT_TAG = "bohn-aws-evidence-20260926"
ZERO = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}

RELEVANT_REL_PATHS = [
    "docs/bohn2021_takeover/solo_gpt55/PLAN_READY.json",
    "docs/bohn2021_takeover/solo_gpt55/solo_c4683d64c130f4c67a0a08fc.execution_plan.json",
    "docs/bohn2021_takeover/solo_gpt55/solo_c4683d64c130f4c67a0a08fc.md",
    "experiments/bohn2021_aws/backup_failure_status_capture_s_tc2h8_asset422_v0.py",
    "experiments/bohn2021_aws/backup_failure_status_capture_s_tc2h8_asset422_v0b_absolute_state.py",
    "experiments/bohn2021_aws/backup_failure_status_capture_s_tc2h8_current_status_v0c.py",
    "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0i_env_tvp_format_repair.py",
    "research_artifacts/aws_runs/20260930T210609_251bf3c6/outcome_receipt.json",
    "research_artifacts/aws_runs/20260930T210609_251bf3c6/registry.json",
    "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0i_env_tvp_format_repair_20260930T210609Z/failed.json",
    "research_artifacts/aws_runs/20260930T211255_3de72618/outcome_receipt.json",
    "research_artifacts/aws_runs/20260930T211255_3de72618/registry.json",
    "research_artifacts/aws_diagnostics/backup_failure_status_capture_s_tc2h8_asset422_v0_20260930T211255Z/raw.json",
    "research_artifacts/aws_diagnostics/backup_failure_status_capture_s_tc2h8_asset422_v0_20260930T211255Z/summary.md",
    "research_artifacts/aws_runs/20260930T211709_e7e6e1e5/outcome_receipt.json",
    "research_artifacts/aws_runs/20260930T211709_e7e6e1e5/registry.json",
    "research_artifacts/aws_diagnostics/backup_failure_status_capture_s_tc2h8_asset422_v0b_absolute_state_20260930T211710Z/raw.json",
    "research_artifacts/aws_diagnostics/backup_failure_status_capture_s_tc2h8_asset422_v0b_absolute_state_20260930T211710Z/summary.md",
    "research_artifacts/aws_diagnostics/backup_failure_status_capture_s_tc2h8_asset422_v0b_absolute_state_20260930T211710Z/completed.json",
    "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_REPAIR_AFTER_S_TC2H8_ASSET422_V0B_20260930T211710Z.json",
    "research_artifacts/aws_state/continue_state_20260930T211710Z_after_s_tc2h8_backup_422_status_capture_v0b.md",
]
DOCS = [
    ROOT / "STATUS.md",
    ROOT / "RESEARCH_LOG.md",
    ROOT / "DECISIONS.md",
    ROOT / "RESULTS_AUDIT.md",
    ROOT / "REPRODUCTION_PROTOCOL.md",
    ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md",
]
SECRET_KEY_RE = re.compile(r"(token|secret|password|credential|authorization|bearer|github\.token|key_material)", re.I)
LONG_HEX_RE = re.compile(r"\b[a-fA-F0-9]{96,}\b")


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        try:
            return path.resolve().relative_to(CANON_BASE.resolve()).as_posix()
        except Exception:
            return str(path)


def backup_key_for_repo_rel(repo_rel: str) -> str:
    return "mobile-robot-mppi-study/" + repo_rel.lstrip("/")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, val in value.items():
            ks = str(key)
            if SECRET_KEY_RE.search(ks) and ks.lower() not in {"sha256", "asset_sha256", "package_sha256"}:
                out[ks] = "[REDACTED]"
            else:
                out[ks] = sanitize(val)
        return out
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        redacted = LONG_HEX_RE.sub("[REDACTED_HEX]", value)
        redacted = redacted.replace(str(CANON_BASE / ".secrets"), "[REDACTED_SECRET_PATH]")
        return redacted
    return value


def safe_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return sanitize(read_json(path)), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:500]}"


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def path_info(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "bytes": None, "sha256": None, "mtime_utc": None, "kind": "missing"}
    stat = path.stat()
    return {"path": rel(path), "exists": True, "bytes": stat.st_size if path.is_file() else None, "sha256": sha256(path) if path.is_file() else None, "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(), "kind": "file" if path.is_file() else "dir"}


def required_min_time(paths: Iterable[Path]) -> dt.datetime:
    times: List[dt.datetime] = []
    for path in list(paths) + [Path(__file__), ROOT / "docs/bohn2021_takeover/solo_gpt55/PLAN_READY.json"]:
        if path.exists():
            times.append(dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc))
    return max(times) if times else now_utc()


def package_metadata_ok(status: Mapping[str, Any]) -> bool:
    packages = status.get("packages_this_run") or []
    if isinstance(packages, list) and packages:
        return all(isinstance(item, Mapping) and bool(item.get("sha256")) and bool(item.get("bytes")) and bool(item.get("verification")) for item in packages)
    return bool(status.get("package_sha256") or status.get("asset_sha256") or status.get("release_asset_sha256"))


def classify_status(status: Optional[Mapping[str, Any]], min_time: dt.datetime) -> Dict[str, Any]:
    if not isinstance(status, Mapping):
        return {"status_file_readable": False, "adequate_backup_for_s_tc2h8": False, "why_not": ["backup_status_missing_or_unreadable"], "min_required_backup_time_utc": min_time.isoformat(), "current_failure_class": "status_missing"}
    t = parse_time(status.get("time") or status.get("created_utc") or status.get("verified_utc"))
    verified = status.get("status") == "verified" or status.get("backup_verified") is True
    try:
        remaining_ok = int(status.get("remaining_changed_files", -1)) == 0
    except Exception:
        remaining_ok = False
    has_commit = bool(status.get("commit"))
    packages_ok = package_metadata_ok(status)
    time_ok = bool(t is not None and t >= min_time)
    message = str(status.get("message", ""))
    error_type = str(status.get("error_type", ""))
    http_422 = "422" in message or "HTTP 422" in message
    timeout = "timeout" in message.lower() or "timeout" in error_type.lower() or "timed out" in message.lower()
    if verified and remaining_ok and has_commit and packages_ok and time_ok:
        failure_class = "verified_current_backup"
    elif http_422:
        failure_class = "github_asset_upload_http_422"
    elif timeout:
        failure_class = "backup_write_timeout"
    elif status.get("status") == "partial":
        failure_class = "partial_backup_remaining_changes"
    else:
        failure_class = "backup_failed_or_incomplete_other"
    why: List[str] = []
    if not verified:
        why.append(f"status_is_{status.get('status')!r}_not_verified")
    if not remaining_ok:
        why.append("remaining_changed_files_not_zero_or_missing")
    if not has_commit:
        why.append("missing_commit")
    if not packages_ok:
        why.append("missing_verified_package_metadata")
    if not time_ok:
        why.append("status_time_predates_s_tc2h8_required_artifacts_or_unparseable")
    return {"status_file_readable": True, "status": status.get("status"), "error_type": status.get("error_type"), "message": status.get("message"), "status_time": None if t is None else t.isoformat(), "remaining_changed_files": status.get("remaining_changed_files"), "commit": status.get("commit"), "package_count": len(status.get("packages_this_run") or []), "has_verified_package_metadata": packages_ok, "time_ok": time_ok, "verified_flag_ok": bool(verified), "min_required_backup_time_utc": min_time.isoformat(), "adequate_backup_for_s_tc2h8": bool(verified and remaining_ok and has_commit and packages_ok and time_ok), "why_not": why, "http_422_observed": http_422, "timeout_observed": timeout, "current_failure_class": failure_class}


def receipts_tail(limit: int = 15) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not BACKUP_RECEIPTS.exists():
        return [], "missing"
    try:
        lines = BACKUP_RECEIPTS.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return [], f"{type(exc).__name__}: {str(exc)[:500]}"
    out: List[Dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            obj = sanitize(json.loads(line))
            out.append({"time": obj.get("time"), "asset": obj.get("asset"), "manifest": obj.get("manifest")})
        except Exception:
            out.append({"unparsed_prefix": line[:200]})
    return out, None


def backup_index_probe(repo_rels: Iterable[str]) -> Dict[str, Any]:
    if not BACKUP_INDEX.exists():
        return {"exists": False, "probed": True}
    try:
        con = sqlite3.connect(str(BACKUP_INDEX))
        row_count = con.execute("select count(*) from files").fetchone()[0]
        hits = []
        for repo_rel in repo_rels:
            candidate_keys = [backup_key_for_repo_rel(repo_rel), repo_rel]
            rows = []
            for key in candidate_keys:
                row = con.execute("select size, mtime, sha, asset from files where path=?", (key,)).fetchone()
                if row is not None:
                    rows.append({"lookup_key": key, "size": row[0], "mtime_ns": row[1], "sha256": row[2], "asset": row[3]})
            hits.append({"repo_rel": repo_rel, "indexed": bool(rows), "hits": rows})
        con.close()
        return {"exists": True, "probed": True, "row_count": row_count, "relevant_artifacts": hits}
    except Exception as exc:
        return {"exists": True, "probed": True, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def staging_probe() -> Dict[str, Any]:
    if not BACKUP_STAGING.exists():
        return {"exists": False, "path": rel(BACKUP_STAGING)}
    try:
        entries = []
        for path in sorted(BACKUP_STAGING.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:40]:
            entries.append(path_info(path))
        return {"exists": True, "path": rel(BACKUP_STAGING), "recent_entries": entries}
    except Exception as exc:
        return {"exists": True, "path": rel(BACKUP_STAGING), "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def release_probe(tag: str) -> Dict[str, Any]:
    base = "https://api.github.com/repos/" + REPO
    headers = {"User-Agent": "bohn-research-backup-current-status"}
    out: Dict[str, Any] = {"attempted": True, "repo": REPO, "tag": tag}
    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/releases/tags/" + tag, headers=headers), timeout=45) as response:
            release = json.loads(response.read().decode("utf-8"))
        rid = release.get("id")
        assets: List[Dict[str, Any]] = []
        if rid is not None:
            for page in range(1, 31):
                url = f"{base}/releases/{rid}/assets?per_page=100&page={page}"
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                    batch = json.loads(response.read().decode("utf-8"))
                if not batch:
                    break
                for item in batch:
                    assets.append({"name": item.get("name"), "size": item.get("size"), "state": item.get("state"), "created_at": item.get("created_at"), "updated_at": item.get("updated_at"), "digest": item.get("digest")})
                if len(batch) < 100:
                    break
        names: Dict[str, int] = {}
        for item in assets:
            name = str(item.get("name"))
            names[name] = names.get(name, 0) + 1
        duplicates = [name for name, count in names.items() if count > 1]
        out.update({"ok": True, "release_id": rid, "html_url": release.get("html_url"), "asset_count_returned": len(assets), "asset_count_probe_cap": 3000, "asset_limit_warning_ge_900": len(assets) >= 900, "duplicate_names_returned": duplicates[:20], "last_assets_sample": assets[-15:]})
    except Exception as exc:
        out.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
    return out


def git_metadata() -> Dict[str, Any]:
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0")
    commands = {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status_relevant": ["git", "status", "--porcelain", "--"] + RELEVANT_REL_PATHS,
        "status_short_prefix": ["git", "status", "--porcelain"],
    }
    out: Dict[str, Any] = {}
    for name, cmd in commands.items():
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=30)
            out[name] = {"returncode": proc.returncode, "stdout_prefix": sanitize(proc.stdout[:8000]), "stderr_prefix": sanitize(proc.stderr[:1000])}
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return out


def api_token_total() -> Dict[str, Any]:
    if not RESEARCH_DB.exists():
        return {"available": False, "path": rel(RESEARCH_DB), "error": "state research.sqlite missing"}
    try:
        con = sqlite3.connect(str(RESEARCH_DB))
        rows = con.execute("select usage from calls").fetchall()
        total = 0
        for (usage_text,) in rows:
            try:
                total += int((json.loads(usage_text or "{}") or {}).get("total_tokens", 0) or 0)
            except Exception:
                pass
        con.close()
        return {"available": True, "path": rel(RESEARCH_DB), "call_rows": len(rows), "total_tokens": total}
    except Exception as exc:
        return {"available": False, "path": rel(RESEARCH_DB), "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    d = raw["backup_gate_decision"]
    release = raw["public_release_probe_default_tag"]
    lines = [
        "# S-TC2H8 current backup status recheck v0c",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata-only; no solver, plant, training/refit, validation64, or sealed/final-test usage.",
        f"Canonical supervisor state path: `{raw['supervisor_state_path']}`.",
        "",
        "## Current backup gate decision",
        "",
        f"- Adequate verified backup for S-TC2H8: `{d['adequate_backup_for_s_tc2h8']}`.",
        f"- Current failure class: `{d.get('current_failure_class')}`.",
        f"- Reasons if blocked: `{d['why_not']}`.",
        f"- Status: `{d.get('status')}`; error type: `{d.get('error_type')}`; message: `{d.get('message')}`.",
        f"- HTTP 422 observed: `{d.get('http_422_observed')}`; timeout observed: `{d.get('timeout_observed')}`.",
        f"- Minimum required backup time: `{d['min_required_backup_time_utc']}`.",
        "",
        "## Public release probe",
        "",
        f"- Default tag probe ok: `{release.get('ok')}`; assets returned: `{release.get('asset_count_returned')}`; near/at asset cap: `{release.get('asset_limit_warning_ge_900')}`.",
        "",
        "## Next action",
        "",
        str(raw["next_action"]),
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def record_receipt(evidence: Mapping[str, Any]) -> None:
    if execution_contract is None or not os.environ.get("BOHN_EXECUTION_SNAPSHOT"):
        return
    execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), dict(evidence))


def main() -> int:
    created_dt = now_utc()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {"started_utc": created_dt.isoformat(), "method": NAME, "task_id": TASK_ID, "classification": "metadata_only_backup_status_recheck", "resources": ZERO, "validation64_bank_opened": False, "sealed_test_accessed": False, "no_secrets_read": True})

    relevant_paths = [ROOT / p for p in RELEVANT_REL_PATHS]
    status_obj, status_error = safe_json(BACKUP_STATUS)
    receipts, receipts_error = receipts_tail()
    min_time = required_min_time(relevant_paths)
    decision = classify_status(status_obj if isinstance(status_obj, Mapping) else None, min_time)
    default_release = release_probe(DEFAULT_TAG)
    index_probe = backup_index_probe(RELEVANT_REL_PATHS)
    staging = staging_probe()
    git_meta = git_metadata()
    token_total = api_token_total()

    if decision["adequate_backup_for_s_tc2h8"] and isinstance(status_obj, Mapping):
        proof = dict(status_obj)
        proof.update({
            "backup_verified": True,
            "created_utc": created_dt.isoformat(),
            "source": "sanitized copy from canonical supervisor backup_status after S-TC2H8 current-status v0c recheck",
            "covers_s_tc2h8_required_artifacts": True,
            "min_required_backup_time_utc": min_time.isoformat(),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "resources": ZERO,
        })
        write_json(PROOF_IF_VERIFIED, sanitize(proof))
        copied_proof = rel(PROOF_IF_VERIFIED)
        retry_request = None
        next_action = "Backup is adequate. Run the already-authored S-TC2H8 v0i microcontinuation under the frozen legacy/diagnostic resource bounds."
    else:
        if decision.get("current_failure_class") == "backup_write_timeout":
            suggested = "Retry or repair the external backup transport with smaller/resumable packages or a longer write timeout; preserve staging packages and do not delete evidence."
        elif decision.get("current_failure_class") == "github_asset_upload_http_422":
            suggested = "Rotate the backup release tag or shard release assets because the default tag appears to be at/near GitHub asset limits; preserve all prior assets/manifests."
        else:
            suggested = "Repair or retry the external backup until backup_status is verified with remaining_changed_files=0, commit, and package verification metadata."
        write_json(REQUEST, {
            "requested_utc": created_dt.isoformat(),
            "request": "retry_or_repair_external_backup_after_s_tc2h8_current_status_v0c",
            "backup_required_before_nonzero_solver_or_plant_resources": True,
            "backup_gate_decision": decision,
            "suggested_supervisor_repair": suggested,
            "must_cover_before_next_nonzero_resources": RELEVANT_REL_PATHS + [rel(OUT), rel(STATE_MD), rel(REQUEST)],
            "required_next_backup_properties": {
                "status_or_backup_verified": "verified true",
                "remaining_changed_files": 0,
                "must_include_commit": True,
                "must_include_external_release_asset_or_package_sha256": True,
                "must_postdate_required_artifacts_and_this_request": True,
                "must_not_contain_secrets": True,
            },
            "next_scientific_action_after_valid_backup": "Run S-TC2H8 v0i exactly once with frozen task_id S-TC2H8-source242-env-tvp-format-repair-v0, legacy interpreter, diagnostic split, H=[12,15,35], V15_shared, no training/refit, no validation64, no sealed/final test.",
            "resources": ZERO,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        })
        copied_proof = None
        retry_request = rel(REQUEST)
        next_action = "Nonzero S-TC2H8 resources remain blocked by backup. Use this current-status evidence to repair/retry backup; do not run solver/plant work until a verified proof postdating v0c exists."

    raw: Dict[str, Any] = {
        "created_utc": created_dt.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_EVENT).total_seconds(),
        "server_api_token_audit": token_total,
        "method": NAME,
        "task_id": TASK_ID,
        "classification": "metadata_only_current_supervisor_backup_status_recheck_no_simulation_no_validation_no_test",
        "resources": ZERO,
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
        "supervisor_state_path": rel(STATE),
        "no_secrets_read": True,
        "backup_status_file": path_info(BACKUP_STATUS),
        "backup_status_read_error": status_error,
        "backup_status_sanitized": status_obj,
        "backup_receipts_file": path_info(BACKUP_RECEIPTS),
        "backup_receipts_read_error": receipts_error,
        "backup_receipts_tail_sanitized": receipts,
        "backup_index_file": path_info(BACKUP_INDEX),
        "backup_index_probe": index_probe,
        "backup_staging_probe": staging,
        "relevant_artifacts": [path_info(p) for p in relevant_paths],
        "backup_gate_decision": decision,
        "public_release_probe_default_tag": default_release,
        "git_metadata_sanitized": git_meta,
        "copied_verified_proof": copied_proof,
        "retry_backup_request": retry_request,
        "next_action": next_action,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE_MD.parent.mkdir(parents=True, exist_ok=True)
    STATE_MD.write_text(
        "# Continue state after S-TC2H8 current backup status recheck v0c\n\n"
        + f"UTC: {created_dt.isoformat()}\n"
        + f"Elapsed since first supervisor event: {(created_dt - FIRST_EVENT).total_seconds()/3600.0:.2f} h.\n"
        + f"Server API total tokens (from state DB, if available): {token_total.get('total_tokens')}\n"
        + "No solver/plant/training/validation/test resources were used. "
        + f"Adequate backup={decision['adequate_backup_for_s_tc2h8']}; class={decision.get('current_failure_class')}; reasons={decision['why_not']}.\n"
        + f"Summary: `{rel(OUT / 'summary.md')}`. Raw: `{rel(OUT / 'raw.json')}`. Next: {next_action}\n",
        encoding="utf-8",
    )

    marker = f"s-tc2h8-current-backup-status-v0c-{STAMP}"
    block = f"""<!-- {marker} -->
## S-TC2H8 current backup status recheck v0c

UTC: {created_dt.isoformat()}. Metadata-only current backup-state recheck completed in temporary solo GPT-5.5 mode; no solver, plant, training/refit, validation64, or sealed/final-test resources were used. Current backup adequate for S-TC2H8=`{decision['adequate_backup_for_s_tc2h8']}`; class=`{decision.get('current_failure_class')}`; reasons=`{decision['why_not']}`. Evidence: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`. Next action: {next_action}
"""
    for doc in DOCS:
        append_if_missing(doc, marker, block)

    completed_files = [
        OUT / "run_started.json", OUT / "raw.json", OUT / "summary.md", STATE_MD, Path(__file__).resolve(),
        ROOT / "experiments/bohn2021_aws/backup_failure_status_capture_s_tc2h8_current_status_v0c.py",
    ]
    if REQUEST.exists():
        completed_files.append(REQUEST)
    if PROOF_IF_VERIFIED.exists():
        completed_files.append(PROOF_IF_VERIFIED)
    completed_files.extend([p for p in DOCS if p.exists()])
    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created_dt.isoformat(),
        "task_id": TASK_ID,
        "classification": raw["classification"],
        "backup_gate_decision": decision,
        "copied_verified_proof": copied_proof,
        "retry_backup_request": retry_request,
        "summary": rel(OUT / "summary.md"),
        "raw": rel(OUT / "raw.json"),
        "state": rel(STATE_MD),
        "resources": ZERO,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {rel(p): sha256(p) for p in sorted(set(completed_files)) if p.exists() and p.is_file()},
    }
    write_json(OUT / "completed.json", completed)
    evidence = {
        "current_backup_status_rechecked": True,
        "canonical_supervisor_state_used": True,
        "no_solver_plant_training_validation_test_usage": True,
        "backup_status_read_attempted": True,
        "backup_receipts_tail_captured": True,
        "backup_index_probe_completed": True,
        "public_release_probe_attempted": True,
        "latest_backup_failure_timeout_or_current_status_classified": True,
        "s_tc2h8_backup_gate_decision_recorded": True,
        "artifacts_persisted": True,
        "no_secrets_read": True,
        "no_validation64_or_test_usage": True,
    }
    record_receipt(evidence)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "adequate_backup_for_s_tc2h8": decision["adequate_backup_for_s_tc2h8"], "current_failure_class": decision.get("current_failure_class"), "why_not": decision["why_not"], "copied_verified_proof": copied_proof, "retry_backup_request": retry_request, "resources": ZERO, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
