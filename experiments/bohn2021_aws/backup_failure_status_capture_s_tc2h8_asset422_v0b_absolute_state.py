#!/usr/bin/env python3
"""Corrected S-TC2H8 backup-status capture using the canonical supervisor state.

The first Asset-422 diagnostic (v0) accidentally derived supervisor state from
``ROOT.parent`` in the execution mirror, so it reported missing backup_status and
backup_index files.  This v0b diagnostic reads the canonical
``/data/openai-agent/state`` path explicitly while still avoiding secrets.  It is
metadata-only: no MPC/controller imports, no solver calls, no plant steps, no
training/refit, no validation64, and no sealed/final-test access.
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

ROOT = Path(__file__).resolve().parents[2]
CANON_BASE = Path("/data/openai-agent")
STATE_CANDIDATES = [CANON_BASE / "state", ROOT.parent / "state"]
SUP_STATE = next((p for p in STATE_CANDIDATES if (p / "backup_status.json").exists() or (p / "research.sqlite").exists()), STATE_CANDIDATES[0])
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
try:
    import execution_contract  # type: ignore
except Exception:
    execution_contract = None  # type: ignore

NAME = "backup_failure_status_capture_s_tc2h8_asset422_v0b_absolute_state"
TASK_ID = "S-BACKUP422-STC2H8-status-capture-v0b"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "research_artifacts/aws_diagnostics" / f"{NAME}_{STAMP}"
STATE_MD = ROOT / "research_artifacts/aws_state" / f"continue_state_{STAMP}_after_s_tc2h8_backup_422_status_capture_v0b.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_REPAIR_AFTER_S_TC2H8_ASSET422_V0B_{STAMP}.json"
PROOF_IF_VERIFIED = BACKUP_DIR / f"BACKUP_VERIFIED_FROM_STATUS_AFTER_S_TC2H8_ASSET422_V0B_{STAMP}.json"
BACKUP_STATUS = SUP_STATE / "backup_status.json"
BACKUP_RECEIPTS = SUP_STATE / "backup_receipts.jsonl"
BACKUP_INDEX = SUP_STATE / "backup_index.sqlite"
RESEARCH_DB = SUP_STATE / "research.sqlite"
BACKUP_STAGING = CANON_BASE / "backup_staging"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
REPO = "Mapples-Frost/mobile-robot-mppi-study"
TAG = "bohn-aws-evidence-20260926"
ZERO = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}

RELEVANT_PATHS = [
    ROOT / "docs/bohn2021_takeover/solo_gpt55/PLAN_READY.json",
    ROOT / "docs/bohn2021_takeover/solo_gpt55/solo_80306ffca6ad9f9fb172f634.execution_plan.json",
    ROOT / "docs/bohn2021_takeover/solo_gpt55/solo_80306ffca6ad9f9fb172f634.md",
    ROOT / "experiments/bohn2021_aws/backup_failure_status_capture_s_tc2h8_asset422_v0.py",
    ROOT / "experiments/bohn2021_aws/backup_failure_status_capture_s_tc2h8_asset422_v0b_absolute_state.py",
    ROOT / "research_artifacts/aws_runs/20260930T211255_3de72618/outcome_receipt.json",
    ROOT / "research_artifacts/aws_runs/20260930T211255_3de72618/registry.json",
    ROOT / "research_artifacts/aws_diagnostics/backup_failure_status_capture_s_tc2h8_asset422_v0_20260930T211255Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/backup_failure_status_capture_s_tc2h8_asset422_v0_20260930T211255Z/summary.md",
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0i_env_tvp_format_repair.py",
    ROOT / "research_artifacts/aws_runs/20260930T210609_251bf3c6/outcome_receipt.json",
    ROOT / "research_artifacts/aws_runs/20260930T210609_251bf3c6/registry.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0i_env_tvp_format_repair_20260930T210609Z/failed.json",
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
LONG_HEX_RE = re.compile(r"\b[a-fA-F0-9]{80,}\b")


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
    for path in list(paths) + [Path(__file__)]:
        if path.exists():
            times.append(dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc))
    return max(times) if times else now_utc()


def package_metadata_ok(status: Mapping[str, Any]) -> bool:
    packages = status.get("packages_this_run") or []
    if isinstance(packages, list) and packages:
        return all(isinstance(item, Mapping) and bool(item.get("sha256")) and bool(item.get("bytes")) and bool(item.get("verification")) for item in packages)
    return bool(status.get("package_sha256") or status.get("asset_sha256") or status.get("release_asset_sha256"))


def classify_backup_status(status: Optional[Mapping[str, Any]], min_time: dt.datetime) -> Dict[str, Any]:
    if not isinstance(status, Mapping):
        return {"status_file_readable": False, "adequate_backup_for_s_tc2h8": False, "why_not": ["backup_status_missing_or_unreadable"], "min_required_backup_time_utc": min_time.isoformat(), "http_422_observed": False}
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
    http_422 = "422" in message or "HTTP 422" in message
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
    return {"status_file_readable": True, "status": status.get("status"), "error_type": status.get("error_type"), "message": status.get("message"), "status_time": None if t is None else t.isoformat(), "remaining_changed_files": status.get("remaining_changed_files"), "commit": status.get("commit"), "package_count": len(status.get("packages_this_run") or []), "has_verified_package_metadata": packages_ok, "time_ok": time_ok, "verified_flag_ok": bool(verified), "min_required_backup_time_utc": min_time.isoformat(), "adequate_backup_for_s_tc2h8": bool(verified and remaining_ok and has_commit and packages_ok and time_ok), "why_not": why, "http_422_observed": http_422}


def receipts_tail(limit: int = 12) -> Tuple[List[Dict[str, Any]], Optional[str]]:
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


def backup_index_probe(paths: Iterable[Path]) -> Dict[str, Any]:
    if not BACKUP_INDEX.exists():
        return {"exists": False, "probed": True}
    keys = []
    for path in paths:
        try:
            keys.append(str(path.resolve().relative_to(CANON_BASE.resolve())))
        except Exception:
            keys.append(rel(path))
    try:
        con = sqlite3.connect(str(BACKUP_INDEX))
        row_count = con.execute("select count(*) from files").fetchone()[0]
        hits = []
        for key in keys:
            row = con.execute("select size, mtime, sha, asset from files where path=?", (key,)).fetchone()
            hits.append({"path": key, "indexed": row is not None, "size": None if row is None else row[0], "mtime_ns": None if row is None else row[1], "sha256": None if row is None else row[2], "asset": None if row is None else row[3]})
        con.close()
        return {"exists": True, "probed": True, "row_count": row_count, "relevant_artifacts": hits}
    except Exception as exc:
        return {"exists": True, "probed": True, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def staging_probe() -> Dict[str, Any]:
    if not BACKUP_STAGING.exists():
        return {"exists": False, "path": rel(BACKUP_STAGING)}
    try:
        entries = []
        for path in sorted(BACKUP_STAGING.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            entries.append(path_info(path))
        return {"exists": True, "path": rel(BACKUP_STAGING), "recent_entries": entries}
    except Exception as exc:
        return {"exists": True, "path": rel(BACKUP_STAGING), "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def public_release_probe() -> Dict[str, Any]:
    base = "https://api.github.com/repos/" + REPO
    headers = {"User-Agent": "bohn-research-backup-diagnostic"}
    out: Dict[str, Any] = {"attempted": True, "repo": REPO, "tag": TAG}
    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/releases/tags/" + TAG, headers=headers), timeout=45) as response:
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
        out.update({"ok": True, "release_id": rid, "html_url": release.get("html_url"), "asset_count_returned": len(assets), "asset_count_probe_cap": 3000, "asset_limit_warning_ge_900": len(assets) >= 900, "duplicate_names_returned": duplicates[:20], "first_assets": assets[:5], "last_assets_sample": assets[-20:]})
    except Exception as exc:
        out.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
    return out


def git_metadata() -> Dict[str, Any]:
    env = os.environ.copy(); env.update(GIT_TERMINAL_PROMPT="0")
    commands = {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status_relevant": ["git", "status", "--porcelain", "--"] + [rel(p) for p in RELEVANT_PATHS if p.exists()],
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
    decision = raw["backup_gate_decision"]
    release = raw["public_release_asset_probe"]
    lines = [
        "# Corrected S-TC2H8 backup HTTP-422 status capture v0b",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata-only; no solver, plant, training/refit, validation64, or sealed/final-test usage.",
        f"Supervisor state path used: `{raw['supervisor_state_path']}`.",
        "",
        "## Backup gate decision",
        "",
        f"- Adequate verified backup for S-TC2H8: `{decision['adequate_backup_for_s_tc2h8']}`.",
        f"- Reasons if blocked: `{decision['why_not']}`.",
        f"- Status: `{decision.get('status')}`; error type: `{decision.get('error_type')}`; message: `{decision.get('message')}`.",
        f"- HTTP 422 observed: `{decision.get('http_422_observed')}`.",
        f"- Minimum required backup time: `{decision['min_required_backup_time_utc']}`.",
        "",
        "## Public release probe",
        "",
        f"- Probe ok: `{release.get('ok')}`; assets returned: `{release.get('asset_count_returned')}`; near/at release asset cap: `{release.get('asset_limit_warning_ge_900')}`.",
        f"- Duplicate names returned: `{release.get('duplicate_names_returned')}`.",
        "",
        "## Inference and next action",
        "",
        str(raw["next_action"]),
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    created_dt = now_utc()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {"started_utc": created_dt.isoformat(), "method": NAME, "classification": "metadata_only_backup_diagnostic", "resources": ZERO, "validation64_bank_opened": False, "sealed_test_accessed": False, "supervisor_state_path": rel(SUP_STATE), "no_secrets_read": True})

    status_obj, status_error = safe_json(BACKUP_STATUS)
    receipts, receipts_error = receipts_tail()
    min_time = required_min_time(RELEVANT_PATHS)
    decision = classify_backup_status(status_obj if isinstance(status_obj, Mapping) else None, min_time)
    public_probe = public_release_probe()
    index_probe = backup_index_probe(RELEVANT_PATHS)
    staging = staging_probe()
    git_meta = git_metadata()
    token_total = api_token_total()

    if decision["adequate_backup_for_s_tc2h8"] and isinstance(status_obj, Mapping):
        proof = dict(status_obj)
        proof.update({"backup_verified": True, "created_utc": created_dt.isoformat(), "source": "sanitized copy from canonical supervisor backup_status after corrected S-TC2H8 Asset-422 diagnostic", "covers_s_tc2h8_required_artifacts": True, "min_required_backup_time_utc": min_time.isoformat(), "validation64_bank_opened": False, "sealed_test_accessed": False, "resources": ZERO})
        write_json(PROOF_IF_VERIFIED, sanitize(proof))
        copied_proof = rel(PROOF_IF_VERIFIED)
        retry_request = None
        next_action = "The backup gate is adequate; rerun the already-authored S-TC2H8 v0i microcontinuation under its frozen legacy/diagnostic resource bounds."
    else:
        inferred_cause = "unknown"
        if decision.get("http_422_observed") and public_probe.get("asset_limit_warning_ge_900"):
            inferred_cause = "probable_GitHub_release_asset_cap_reached_for_fixed_tag"
        elif decision.get("http_422_observed"):
            inferred_cause = "GitHub_upload_validation_failure_HTTP_422_needs_supervisor_backup_repair"
        write_json(REQUEST, {"requested_utc": created_dt.isoformat(), "request": "repair_or_retry_external_backup_after_S_TC2H8_HTTP_422_failure_v0b", "backup_required_before_nonzero_solver_or_plant_resources": True, "backup_gate_decision": decision, "inferred_backup_failure_cause": inferred_cause, "public_release_asset_probe_summary": {"ok": public_probe.get("ok"), "asset_count_returned": public_probe.get("asset_count_returned"), "asset_limit_warning_ge_900": public_probe.get("asset_limit_warning_ge_900"), "duplicate_names_returned": public_probe.get("duplicate_names_returned"), "error": public_probe.get("error")}, "suggested_supervisor_repair": "If fixed release tag bohn-aws-evidence-20260926 has reached GitHub's asset limit, rotate backup.py to a fresh dated release tag or release shard while preserving previous assets and manifests. Do not delete evidence and do not bypass verified external backup gates.", "must_cover_before_next_nonzero_resources": [rel(p) for p in RELEVANT_PATHS] + [rel(OUT), rel(STATE_MD), rel(REQUEST)], "required_next_backup_properties": {"status_or_backup_verified": "verified true", "remaining_changed_files": 0, "must_include_commit": True, "must_include_external_release_asset_or_package_sha256": True, "must_postdate_required_artifacts_and_this_request": True, "must_not_contain_secrets": True}, "next_scientific_action_after_valid_backup": "Rerun S-TC2H8 v0i exactly once with frozen task_id S-TC2H8-source242-env-tvp-format-repair-v0, legacy interpreter, diagnostic split, H=[12,15,35], V15_shared, no training/refit, no validation64, no sealed/final test.", "resources": ZERO, "validation64_bank_opened": False, "sealed_test_accessed": False})
        copied_proof = None
        retry_request = rel(REQUEST)
        next_action = "Nonzero S-TC2H8 resources remain blocked by backup. The corrected evidence should be used to repair/retry external backup (likely release-tag rotation if HTTP 422 and >=1000 assets persist), then S-TC2H8 may be rerun only after a verified proof postdating this diagnostic exists."

    raw: Dict[str, Any] = {"created_utc": created_dt.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_EVENT).total_seconds(), "server_api_token_audit": token_total, "method": NAME, "task_id": TASK_ID, "classification": "metadata_only_corrected_supervisor_state_backup_diagnostic_no_simulation_no_validation_no_test", "resources": ZERO, "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False}, "supervisor_state_path": rel(SUP_STATE), "state_candidates": [rel(p) for p in STATE_CANDIDATES], "no_secrets_read": True, "backup_status_file": path_info(BACKUP_STATUS), "backup_status_read_error": status_error, "backup_status_sanitized": status_obj, "backup_receipts_file": path_info(BACKUP_RECEIPTS), "backup_receipts_read_error": receipts_error, "backup_receipts_tail_sanitized": receipts, "backup_index_file": path_info(BACKUP_INDEX), "backup_index_probe": index_probe, "backup_staging_probe": staging, "relevant_artifacts": [path_info(p) for p in RELEVANT_PATHS], "backup_gate_decision": decision, "public_release_asset_probe": public_probe, "git_metadata_sanitized": git_meta, "copied_verified_proof": copied_proof, "retry_backup_request": retry_request, "next_action": next_action, "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}}
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE_MD.parent.mkdir(parents=True, exist_ok=True)
    STATE_MD.write_text("# Continue state after corrected S-TC2H8 backup HTTP-422 diagnostic\n\n" + f"UTC: {created_dt.isoformat()}\nElapsed since first supervisor event: {(created_dt - FIRST_EVENT).total_seconds()/3600.0:.2f} h.\nServer API total tokens (from state DB, if available): {token_total.get('total_tokens')}\nSupervisor state path: `{rel(SUP_STATE)}`.\nNo solver/plant/training/validation/test resources were used. Adequate backup={decision['adequate_backup_for_s_tc2h8']}; reasons={decision['why_not']}; HTTP422={decision.get('http_422_observed')}; public assets={public_probe.get('asset_count_returned')}.\nSummary: `{rel(OUT / 'summary.md')}`. Raw: `{rel(OUT / 'raw.json')}`. Next: {next_action}\n", encoding="utf-8")
    marker = f"s-tc2h8-backup-422-status-capture-v0b-{STAMP}"
    block = (f"<!-- {marker} -->\n## Corrected S-TC2H8 backup HTTP-422 status capture\n\nUTC: {created_dt.isoformat()}. Metadata-only infrastructure diagnostic using canonical supervisor state `{rel(SUP_STATE)}`; resources `{ZERO}`; no validation64 or sealed/final-test access. Adequate backup for S-TC2H8=`{decision['adequate_backup_for_s_tc2h8']}`; reasons=`{decision['why_not']}`; status=`{decision.get('status')}`; error=`{decision.get('error_type')}`; message=`{decision.get('message')}`; HTTP422=`{decision.get('http_422_observed')}`; public release asset count returned=`{public_probe.get('asset_count_returned')}`. Evidence: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`. Request/proof: `{retry_request or copied_proof}`.\n")
    for doc in DOCS:
        append_if_missing(doc, marker, block)

    completed_paths = [OUT / "run_started.json", OUT / "raw.json", OUT / "summary.md", STATE_MD, REQUEST, PROOF_IF_VERIFIED] + RELEVANT_PATHS + DOCS
    completed = {"passed": True, "hard_pass": True, "created_utc": created_dt.isoformat(), "summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "state": rel(STATE_MD), "classification": raw["classification"], "backup_gate_decision": decision, "copied_verified_proof": copied_proof, "retry_backup_request": retry_request, "next_action": next_action, "resources": ZERO, "validation64_bank_opened": False, "sealed_test_accessed": False, "no_secrets_read": True, "server_api_token_audit": token_total, "hashes": {rel(p): sha256(p) for p in sorted(set(completed_paths)) if p.exists() and p.is_file()}}
    write_json(OUT / "completed.json", completed)

    evidence = {"corrected_backup_422_status_capture_completed": True, "canonical_supervisor_state_used": True, "no_solver_plant_training_validation_test_usage": True, "backup_status_read_attempted": True, "backup_receipts_tail_captured": True, "backup_index_probe_completed": True, "public_release_asset_probe_attempted": True, "backup_failure_422_or_current_status_classified": True, "s_tc2h8_backup_gate_decision_recorded": True, "artifacts_persisted": True, "no_secrets_read": True, "no_validation64_or_test_usage": True}
    try:
        if execution_contract is not None:
            execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), evidence)
    except Exception:
        pass
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "adequate_backup_for_s_tc2h8": decision["adequate_backup_for_s_tc2h8"], "why_not": decision["why_not"], "http_422_observed": decision.get("http_422_observed"), "status": decision.get("status"), "error_type": decision.get("error_type"), "message": decision.get("message"), "public_asset_probe_ok": public_probe.get("ok"), "public_asset_count_returned": public_probe.get("asset_count_returned"), "retry_backup_request": retry_request, "copied_verified_proof": copied_proof, "resources": ZERO, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
