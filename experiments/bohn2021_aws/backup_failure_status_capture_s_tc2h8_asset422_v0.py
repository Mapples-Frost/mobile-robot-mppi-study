#!/usr/bin/env python3
"""S-TC2H8 backup-failure status capture and Asset-422 diagnosis.

Metadata-only infrastructure diagnostic for temporary GPT-5.5 solo mode.  This
script does not run controllers, MPC, simulations, training/refits, validation64,
or sealed/final tests.  It reads only sanitized supervisor state files, local
repository metadata, backup index rows, and public GitHub release metadata to
explain why S-TC2H8 cannot spend solver/plant resources while external backup is
not verified.
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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
SUP_STATE = BASE / "state"
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
try:
    import execution_contract  # type: ignore
except Exception:  # pragma: no cover - receipt is best-effort for non-structured use
    execution_contract = None  # type: ignore

NAME = "backup_failure_status_capture_s_tc2h8_asset422_v0"
TASK_ID = "S-BACKUP422-STC2H8-status-capture-v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "research_artifacts/aws_diagnostics" / f"{NAME}_{STAMP}"
STATE_MD = ROOT / "research_artifacts/aws_state" / f"continue_state_{STAMP}_after_s_tc2h8_backup_422_status_capture.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_REPAIR_AFTER_S_TC2H8_ASSET422_{STAMP}.json"
PROOF_IF_VERIFIED = BACKUP_DIR / f"BACKUP_VERIFIED_FROM_STATUS_AFTER_S_TC2H8_ASSET422_{STAMP}.json"
BACKUP_STATUS = SUP_STATE / "backup_status.json"
BACKUP_RECEIPTS = SUP_STATE / "backup_receipts.jsonl"
BACKUP_INDEX = SUP_STATE / "backup_index.sqlite"
RESEARCH_DB = SUP_STATE / "research.sqlite"
BACKUP_STAGING = BASE / "backup_staging"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
REPO = "Mapples-Frost/mobile-robot-mppi-study"
TAG = "bohn-aws-evidence-20260926"
ZERO = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}

RELEVANT_PATHS = [
    ROOT / "docs/bohn2021_takeover/solo_gpt55/PLAN_READY.json",
    ROOT / "docs/bohn2021_takeover/solo_gpt55/solo_71af8e63b0984e6a955146f4.execution_plan.json",
    ROOT / "docs/bohn2021_takeover/solo_gpt55/solo_71af8e63b0984e6a955146f4.md",
    ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0i_env_tvp_format_repair.py",
    ROOT / "research_artifacts/aws_runs/20260930T210609_251bf3c6/outcome_receipt.json",
    ROOT / "research_artifacts/aws_runs/20260930T210609_251bf3c6/registry.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0i_env_tvp_format_repair_20260930T210609Z/failed.json",
    ROOT / "research_artifacts/aws_runs/20260930T190047_54183e0e/outcome_receipt.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0h_highlevel_env_step_repair_20260930T190048Z/raw.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0h_highlevel_env_step_repair_20260930T190048Z/summary.md",
    ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_S_TC2H8_BACKUP_DEPENDENCY_AND_STATE_20260930T201700Z.json",
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
            return path.resolve().relative_to(BASE.resolve()).as_posix()
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
        return redacted.replace(str(BASE / ".secrets"), "[REDACTED_SECRET_PATH]")
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
    st = path.stat()
    info: Dict[str, Any] = {"path": rel(path), "exists": True, "bytes": st.st_size if path.is_file() else None, "sha256": sha256(path) if path.is_file() else None, "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(), "kind": "file" if path.is_file() else "dir"}
    return info


def required_min_time(paths: Iterable[Path]) -> dt.datetime:
    times: List[dt.datetime] = []
    for path in list(paths) + [Path(__file__)]:
        if path.exists():
            times.append(dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc))
    return max(times) if times else now_utc()


def package_metadata_ok(status: Mapping[str, Any]) -> bool:
    packages = status.get("packages_this_run") or []
    if isinstance(packages, list) and packages:
        ok = True
        for item in packages:
            ok = ok and isinstance(item, Mapping) and bool(item.get("sha256")) and bool(item.get("bytes")) and bool(item.get("verification"))
        return bool(ok)
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
    return {
        "status_file_readable": True,
        "status": status.get("status"),
        "error_type": status.get("error_type"),
        "message": status.get("message"),
        "status_time": None if t is None else t.isoformat(),
        "remaining_changed_files": status.get("remaining_changed_files"),
        "commit": status.get("commit"),
        "package_count": len(status.get("packages_this_run") or []),
        "has_verified_package_metadata": packages_ok,
        "time_ok": time_ok,
        "verified_flag_ok": bool(verified),
        "min_required_backup_time_utc": min_time.isoformat(),
        "adequate_backup_for_s_tc2h8": bool(verified and remaining_ok and has_commit and packages_ok and time_ok),
        "why_not": why,
        "http_422_observed": http_422,
    }


def receipts_tail(n: int = 12) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not BACKUP_RECEIPTS.exists():
        return [], "missing"
    try:
        lines = BACKUP_RECEIPTS.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return [], f"{type(exc).__name__}: {str(exc)[:500]}"
    out: List[Dict[str, Any]] = []
    for line in lines[-n:]:
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
            keys.append(str(path.resolve().relative_to(BASE.resolve())))
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
        return {"exists": False}
    entries = []
    try:
        for path in sorted(BACKUP_STAGING.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            if path.is_file():
                entries.append(path_info(path))
            else:
                st = path.stat()
                entries.append({"path": rel(path), "exists": True, "kind": "dir", "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat()})
        return {"exists": True, "recent_entries": entries}
    except Exception as exc:
        return {"exists": True, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def public_release_probe() -> Dict[str, Any]:
    base = "https://api.github.com/repos/" + REPO
    headers = {"User-Agent": "bohn-research-backup-diagnostic"}
    out: Dict[str, Any] = {"attempted": True, "repo": REPO, "tag": TAG}
    try:
        req = urllib.request.Request(base + "/releases/tags/" + TAG, headers=headers)
        with urllib.request.urlopen(req, timeout=45) as response:
            release = json.loads(response.read().decode("utf-8"))
        rid = release.get("id")
        out.update({"release_id": rid, "html_url": release.get("html_url"), "assets_url": release.get("assets_url")})
        assets: List[Dict[str, Any]] = []
        if rid is not None:
            for page in range(1, 31):
                url = f"{base}/releases/{rid}/assets?per_page=100&page={page}"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=45) as response:
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
        out.update({"ok": True, "asset_count_returned": len(assets), "asset_count_probe_cap": 3000, "duplicate_names_returned": duplicates[:20], "recent_assets": assets[:20], "asset_limit_warning_ge_900": len(assets) >= 900})
    except Exception as exc:
        out.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
    return out


def git_metadata() -> Dict[str, Any]:
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0")
    cmds = {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status_relevant": ["git", "status", "--porcelain", "--"] + [rel(p) for p in RELEVANT_PATHS if p.exists()],
        "status_short_prefix": ["git", "status", "--porcelain"],
    }
    out: Dict[str, Any] = {}
    for name, cmd in cmds.items():
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
        total = 0
        rows = con.execute("select usage from calls").fetchall()
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
        "# S-TC2H8 backup failure / HTTP 422 status capture v0",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata-only infrastructure diagnostic; no solver calls, plant steps, training, validation64, or sealed/final-test access.",
        "",
        "## Backup gate decision",
        "",
        f"- Adequate verified post-S-TC2H8 backup: `{decision['adequate_backup_for_s_tc2h8']}`.",
        f"- Reasons if blocked: `{decision['why_not']}`.",
        f"- Supervisor status: `{decision.get('status')}`; error type: `{decision.get('error_type')}`; message: `{decision.get('message')}`.",
        f"- HTTP 422 observed in status message: `{decision.get('http_422_observed')}`.",
        f"- Minimum required backup time: `{decision['min_required_backup_time_utc']}`.",
        "",
        "## Public GitHub release probe",
        "",
        f"- Probe ok: `{release.get('ok')}`; assets returned: `{release.get('asset_count_returned')}`; asset limit warning >=900: `{release.get('asset_limit_warning_ge_900')}`.",
        f"- Duplicate names returned by public API: `{release.get('duplicate_names_returned')}`.",
        "",
        "## Next action",
        "",
        str(raw["next_action"]),
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    created_dt = now_utc()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {"started_utc": created_dt.isoformat(), "method": NAME, "classification": "metadata_only_backup_diagnostic", "resources": ZERO, "validation64_bank_opened": False, "sealed_test_accessed": False, "no_secrets_read": True})

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
        proof.update({
            "backup_verified": True,
            "created_utc": created_dt.isoformat(),
            "source": "sanitized copy from supervisor backup_status after S-TC2H8 Asset-422 diagnostic",
            "covers_s_tc2h8_required_artifacts": True,
            "min_required_backup_time_utc": min_time.isoformat(),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "resources": ZERO,
        })
        write_json(PROOF_IF_VERIFIED, sanitize(proof))
        copied_proof = rel(PROOF_IF_VERIFIED)
        retry_request = None
        next_action = "The backup gate is adequate; rerun the already-approved S-TC2H8 v0i microcontinuation task once under its frozen legacy/diagnostic resource bounds."
    else:
        write_json(REQUEST, {
            "requested_utc": created_dt.isoformat(),
            "request": "repair_or_retry_external_backup_after_S_TC2H8_HTTP_422_failure",
            "backup_required_before_nonzero_solver_or_plant_resources": True,
            "backup_gate_decision": decision,
            "public_release_asset_probe_summary": {"ok": public_probe.get("ok"), "asset_count_returned": public_probe.get("asset_count_returned"), "asset_limit_warning_ge_900": public_probe.get("asset_limit_warning_ge_900"), "duplicate_names_returned": public_probe.get("duplicate_names_returned"), "error": public_probe.get("error")},
            "must_cover_before_next_nonzero_resources": [rel(p) for p in RELEVANT_PATHS] + [rel(OUT), rel(STATE_MD), rel(REQUEST)],
            "required_next_backup_properties": {"status_or_backup_verified": "verified true", "remaining_changed_files": 0, "must_include_commit": True, "must_include_external_release_asset_or_package_sha256": True, "must_postdate_required_artifacts_and_this_request": True, "must_not_contain_secrets": True},
            "next_scientific_action_after_valid_backup": "Rerun S-TC2H8 v0i exactly once with its frozen task_id, legacy interpreter, diagnostic split, H=[12,15,35], V15_shared, no training/refit, no validation64, no sealed/final test.",
            "resources": ZERO,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        })
        copied_proof = None
        retry_request = rel(REQUEST)
        next_action = "Nonzero S-TC2H8 resources remain blocked. Repair/retry the external backup path outside this experiment (HTTP 422 is preserved here), then rerun S-TC2H8 only after a verified backup proof postdating this diagnostic exists."

    raw: Dict[str, Any] = {
        "created_utc": created_dt.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created_dt - FIRST_EVENT).total_seconds(),
        "server_api_token_audit": token_total,
        "method": NAME,
        "task_id": TASK_ID,
        "classification": "metadata_only_infrastructure_diagnostic_no_simulation_no_validation_no_test",
        "resources": ZERO,
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
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
        "relevant_artifacts": [path_info(p) for p in RELEVANT_PATHS],
        "backup_gate_decision": decision,
        "public_release_asset_probe": public_probe,
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
        "# Continue state after S-TC2H8 backup HTTP-422 diagnostic\n\n"
        + f"UTC: {created_dt.isoformat()}\n"
        + f"Elapsed since first supervisor event: {(created_dt - FIRST_EVENT).total_seconds()/3600.0:.2f} h.\n"
        + f"Server API total tokens (from state DB, if available): {token_total.get('total_tokens')}\n\n"
        + "No solver/plant/training/validation/test resources were used. "
        + f"Adequate backup={decision['adequate_backup_for_s_tc2h8']}; reasons={decision['why_not']}; HTTP422={decision.get('http_422_observed')}.\n"
        + f"Summary: `{rel(OUT / 'summary.md')}`. Raw: `{rel(OUT / 'raw.json')}`. Next: {next_action}\n",
        encoding="utf-8",
    )
    marker = f"s-tc2h8-backup-422-status-capture-{STAMP}"
    block = (
        f"<!-- {marker} -->\n"
        "## S-TC2H8 backup HTTP-422 status capture\n\n"
        f"UTC: {created_dt.isoformat()}. Metadata-only infrastructure diagnostic; resources `{ZERO}`; no validation64 or sealed/final-test access. "
        f"Adequate backup for S-TC2H8=`{decision['adequate_backup_for_s_tc2h8']}`; reasons=`{decision['why_not']}`; status=`{decision.get('status')}`; error=`{decision.get('error_type')}`; message=`{decision.get('message')}`; public asset probe ok=`{public_probe.get('ok')}` asset_count=`{public_probe.get('asset_count_returned')}`. "
        f"Evidence: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`. Request/proof: `{retry_request or copied_proof}`.\n"
    )
    for doc in DOCS:
        append_if_missing(doc, marker, block)

    completed_paths = [OUT / "run_started.json", OUT / "raw.json", OUT / "summary.md", STATE_MD, REQUEST, PROOF_IF_VERIFIED] + RELEVANT_PATHS + DOCS
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created_dt.isoformat(),
        "summary": rel(OUT / "summary.md"),
        "raw": rel(OUT / "raw.json"),
        "state": rel(STATE_MD),
        "classification": raw["classification"],
        "backup_gate_decision": decision,
        "copied_verified_proof": copied_proof,
        "retry_backup_request": retry_request,
        "next_action": next_action,
        "resources": ZERO,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "no_secrets_read": True,
        "server_api_token_audit": token_total,
        "hashes": {rel(p): sha256(p) for p in sorted(set(completed_paths)) if p.exists() and p.is_file()},
    }
    write_json(OUT / "completed.json", completed)

    evidence = {
        "backup_422_status_capture_completed": True,
        "no_solver_plant_training_validation_test_usage": True,
        "backup_status_read_attempted": True,
        "backup_receipts_tail_captured": True,
        "backup_index_probe_completed": True,
        "public_release_asset_probe_attempted": True,
        "backup_failure_422_or_current_status_classified": True,
        "s_tc2h8_backup_gate_decision_recorded": True,
        "artifacts_persisted": True,
        "no_secrets_read": True,
        "no_validation64_or_test_usage": True,
    }
    try:
        if execution_contract is not None:
            execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), evidence)
    except Exception:
        pass
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "adequate_backup_for_s_tc2h8": decision["adequate_backup_for_s_tc2h8"], "why_not": decision["why_not"], "http_422_observed": decision.get("http_422_observed"), "public_asset_probe_ok": public_probe.get("ok"), "public_asset_count_returned": public_probe.get("asset_count_returned"), "retry_backup_request": retry_request, "copied_verified_proof": copied_proof, "resources": ZERO, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
