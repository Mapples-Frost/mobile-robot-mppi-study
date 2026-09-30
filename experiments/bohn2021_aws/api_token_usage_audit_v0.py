#!/usr/bin/env python3
"""Aggregate server API token usage from research.sqlite without logging prompts.

Operational audit only. It inspects SQLite schema and aggregates numeric token
fields or JSON usage objects. It must not open validation/test data, run
simulations, train/refit anything, or print prompt/response contents.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
NAME = "api_token_usage_audit_v0"
OUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
STATE_DIR = ROOT / "research_artifacts/aws_state"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
SQLITE_CANDIDATES = [
    BASE / "state" / "research.sqlite",
    ROOT / "research.sqlite",
    Path("/data/openai-agent/state/research.sqlite"),
    Path("/data/openai-agent/research.sqlite"),
]
TOKEN_KEYS = {
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "audio_tokens",
}
TOTAL_KEYS = {"total_tokens", "tokens_total", "usage_total_tokens", "server_total_tokens"}
JSON_COLUMN_HINTS = ("usage", "response", "result", "raw", "metadata", "record", "json")
MAX_JSON_ROWS_PER_COLUMN = 20000


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_elapsed(seconds: float) -> str:
    days = int(seconds // 86400)
    rem = seconds - days * 86400
    hours = int(rem // 3600)
    rem -= hours * 3600
    minutes = int(rem // 60)
    sec = rem - minutes * 60
    return f"{days}d {hours}h {minutes}m {sec:.3f}s"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def append_registry_once(row_id: str, row: str) -> None:
    path = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if row_id not in old:
        path.write_text(old.rstrip() + "\n" + row.strip() + "\n", encoding="utf-8")


def safe_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def numeric_or_none(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def walk_token_fields(obj: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = str(k)
            lname = key.lower()
            path = f"{prefix}.{key}" if prefix else key
            val = numeric_or_none(v)
            if val is not None and (lname in TOKEN_KEYS or lname in TOTAL_KEYS or lname.endswith("_tokens")):
                out[path] = out.get(path, 0.0) + val
            elif isinstance(v, (Mapping, list)):
                for sub, subval in walk_token_fields(v, path).items():
                    out[sub] = out.get(sub, 0.0) + subval
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            # Do not expose arbitrary list positions in final summary except path prefix.
            for sub, subval in walk_token_fields(v, f"{prefix}[]" if prefix else "[]").items():
                out[sub] = out.get(sub, 0.0) + subval
    return out


def merge_sum(dst: Dict[str, float], src: Mapping[str, float]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + float(v)


def find_sqlite() -> Optional[Path]:
    seen: set[str] = set()
    for p in SQLITE_CANDIDATES:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        if p.exists():
            return p
    return None


def audit_sqlite(db_path: Path) -> Dict[str, Any]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [str(r[0]) for r in cur.fetchall()]
    table_summaries: Dict[str, Any] = {}
    global_numeric: Dict[str, float] = {}
    global_json: Dict[str, float] = {}
    best_total_candidates: List[Dict[str, Any]] = []

    for table in tables:
        qt = safe_ident(table)
        cur.execute(f"PRAGMA table_info({qt})")
        cols_info = cur.fetchall()
        cols = [str(r[1]) for r in cols_info]
        col_types = {str(r[1]): str(r[2]) for r in cols_info}
        try:
            cur.execute(f"SELECT COUNT(*) FROM {qt}")
            row_count = int(cur.fetchone()[0] or 0)
        except Exception as exc:
            table_summaries[table] = {"columns": cols, "column_types": col_types, "row_count_error": str(exc)}
            continue

        table_numeric: Dict[str, float] = {}
        numeric_columns = [c for c in cols if c.lower() in TOKEN_KEYS or c.lower() in TOTAL_KEYS or c.lower().endswith("_tokens")]
        for col in numeric_columns:
            try:
                cur.execute(f"SELECT SUM(CASE WHEN {safe_ident(col)} IS NULL THEN 0 ELSE {safe_ident(col)} END) FROM {qt}")
                val = cur.fetchone()[0]
                if val is not None:
                    key = f"{table}.{col}"
                    table_numeric[key] = float(val)
                    global_numeric[key] = global_numeric.get(key, 0.0) + float(val)
                    if col.lower() in TOTAL_KEYS:
                        best_total_candidates.append({"source": key, "total_tokens": int(float(val)), "basis": "numeric_column_sum", "row_count": row_count})
            except Exception as exc:
                table_numeric[f"{table}.{col}__error"] = 0.0

        table_json: Dict[str, float] = {}
        json_columns = [c for c in cols if any(h in c.lower() for h in JSON_COLUMN_HINTS)]
        for col in json_columns:
            scanned = 0
            parsed = 0
            found_any = False
            try:
                cur.execute(f"SELECT {safe_ident(col)} FROM {qt} WHERE {safe_ident(col)} IS NOT NULL LIMIT {MAX_JSON_ROWS_PER_COLUMN}")
                for (value,) in cur.fetchall():
                    scanned += 1
                    if not isinstance(value, str):
                        continue
                    s = value.strip()
                    if not (s.startswith("{") or s.startswith("[")):
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    parsed += 1
                    extracted = walk_token_fields(obj)
                    if extracted:
                        found_any = True
                        prefixed = {f"{table}.{col}.{k}": v for k, v in extracted.items()}
                        merge_sum(table_json, prefixed)
                        merge_sum(global_json, prefixed)
                if found_any:
                    # Prefer fields explicitly named total_tokens.
                    for key, val in table_json.items():
                        if key.lower().endswith("total_tokens"):
                            best_total_candidates.append({"source": key, "total_tokens": int(val), "basis": "json_field_sum", "row_count": row_count, "rows_scanned": scanned, "json_rows_parsed": parsed})
            except Exception:
                pass

        table_summaries[table] = {
            "row_count": row_count,
            "columns": cols,
            "column_types": col_types,
            "numeric_token_sums": table_numeric,
            "json_token_sums": table_json,
            "json_columns_scanned": json_columns,
        }

    conn.close()
    # Avoid double-counting multiple aliases. Report candidates and select the
    # largest exact total_tokens-like aggregate as the conservative cumulative
    # server total if any exists.
    selected_total = None
    if best_total_candidates:
        selected_total = sorted(best_total_candidates, key=lambda x: x.get("total_tokens", 0), reverse=True)[0]
    return {
        "sqlite_path": str(db_path),
        "sqlite_sha256": sha256(db_path),
        "sqlite_bytes": db_path.stat().st_size,
        "sqlite_mtime_utc": dt.datetime.fromtimestamp(db_path.stat().st_mtime, dt.timezone.utc).isoformat(),
        "tables": table_summaries,
        "global_numeric_token_sums": global_numeric,
        "global_json_token_sums": global_json,
        "total_token_candidates": best_total_candidates,
        "selected_cumulative_total_tokens": selected_total,
    }


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"{NAME}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "run_started.json", {
        "started_utc": created.isoformat(),
        "classification": "metadata_only_api_token_usage_audit_no_sim_no_validation_no_test_no_training_no_refit",
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    elapsed_text = fmt_elapsed((created - FIRST_SUPERVISOR_EVENT).total_seconds())
    db = find_sqlite()
    audit: Dict[str, Any]
    if db is None:
        audit = {"available": False, "error": "research.sqlite not found", "checked_paths": [str(p) for p in SQLITE_CANDIDATES]}
    else:
        try:
            audit = {"available": True, "result": audit_sqlite(db)}
        except Exception as exc:
            audit = {"available": False, "sqlite_path": str(db), "error": f"{type(exc).__name__}: {str(exc)[:500]}"}

    selected = None
    if audit.get("available") and isinstance(audit.get("result"), Mapping):
        selected = audit["result"].get("selected_cumulative_total_tokens")
    if selected:
        token_text = f"{selected['total_tokens']:,} ({selected['total_tokens'] / 1_000_000:.3f}M) from {selected['source']}"
    elif audit.get("available"):
        token_text = "unknown (SQLite present but no aggregate total_tokens field found; schema summarized without prompts)"
    else:
        token_text = f"unknown ({audit.get('error')})"

    backup_request = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_API_TOKEN_USAGE_AUDIT_{stamp}.json"
    continue_path = STATE_DIR / f"continue_state_{stamp}_after_api_token_usage_audit.md"
    planned_cover = [
        rel(Path(__file__).resolve()),
        rel(out_dir / "run_started.json"),
        rel(out_dir / "summary.md"),
        rel(out_dir / "raw.json"),
        rel(out_dir / "completed.json"),
        rel(backup_request),
        rel(continue_path),
        "STATUS.md",
        "RESEARCH_LOG.md",
        "EXPERIMENT_REGISTRY.csv",
    ]
    write_json(backup_request, {
        "requested_utc": created.isoformat(),
        "reason": "Back up metadata-only API token usage audit outputs.",
        "must_cover": planned_cover,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    raw = {
        "created_utc": created.isoformat(),
        "classification": "metadata_only_api_token_usage_audit_no_sim_no_validation_no_test_no_training_no_refit",
        "elapsed_since_first_supervisor_event_text": elapsed_text,
        "api_token_audit": audit,
        "selected_status_line_token_text": token_text,
        "budgets_actual": {
            "new_simulation_episodes": 0,
            "new_control_steps": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "backup_request": rel(backup_request),
    }
    write_json(out_dir / "raw.json", raw)

    if audit.get("available") and isinstance(audit.get("result"), Mapping):
        result = audit["result"]
        table_count = len(result.get("tables", {}))
        numeric_sources = len(result.get("global_numeric_token_sums", {}))
        json_sources = len(result.get("global_json_token_sums", {}))
        sqlite_line = f"SQLite `{result.get('sqlite_path')}` present with {table_count} tables; numeric token sources {numeric_sources}, JSON token sources {json_sources}."
    else:
        sqlite_line = f"SQLite token audit unavailable: {audit.get('error')}"

    summary = f"""# API token usage audit

UTC: `{created.isoformat()}`. Metadata-only operational audit; no simulations, no control steps, no selector refit, no training, no validation64 bank access and no sealed-test access. Prompt/response contents were not logged.

## Required status-line values
- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `{elapsed_text}`.
- Cumulative server API total_tokens from research.sqlite: `{token_text}`; desktop conversation tokens excluded.

## SQLite audit
- {sqlite_line}
- Selected cumulative total token source: `{selected}`.
- Full schema/aggregate-only output: `{rel(out_dir / 'raw.json')}`.
- New backup request: `{rel(backup_request)}`.
"""
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    continue_path.write_text(summary, encoding="utf-8")

    marker = f"<!-- {NAME}-{stamp} -->"
    doc_block = f"""## 2026-09-30 API token usage audit

UTC: {created.isoformat()}. Metadata-only operational audit; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `{elapsed_text}`. Server API total_tokens status: `{token_text}`. Details: `{rel(out_dir / 'raw.json')}`. Backup request: `{rel(backup_request)}`.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md"]:
        append_once(doc, marker, doc_block)

    registry_id = f"{stamp}_{NAME}"
    append_registry_once(
        registry_id,
        f"{registry_id},{created.isoformat()},metadata_only_api_token_usage_audit_no_science,none,no_validation_no_test,unknown,complete,0,0,0,{rel(out_dir / 'raw.json')}"
    )

    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "classification": raw["classification"],
        "summary": rel(out_dir / "summary.md"),
        "raw": rel(out_dir / "raw.json"),
        "backup_request": rel(backup_request),
        "continue_state": rel(continue_path),
        "selected_status_line_token_text": token_text,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {},
    }
    write_json(out_dir / "completed.json", completed)
    hash_paths = [
        Path(__file__).resolve(),
        out_dir / "run_started.json",
        out_dir / "summary.md",
        out_dir / "raw.json",
        out_dir / "completed.json",
        backup_request,
        continue_path,
        ROOT / "STATUS.md",
        ROOT / "RESEARCH_LOG.md",
        ROOT / "EXPERIMENT_REGISTRY.csv",
    ]
    completed["hashes"] = {rel(p): sha256(p) for p in hash_paths}
    write_json(out_dir / "completed.json", completed)
    raw["completed_hash"] = sha256(out_dir / "completed.json")
    write_json(out_dir / "raw.json", raw)

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
