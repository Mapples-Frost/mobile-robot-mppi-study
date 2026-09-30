#!/usr/bin/env python3
"""v33 bookkeeping preflight before the fixed 24-call objective-vs-basin solver probe.

Astra 20260930T072558Z Task 1: close the evidence/registry bookkeeping gap
without re-running terminal identity or any plant/solver/training work.

Development-only metadata diagnostic:
  * 0 plant steps, 0 solver calls, 0 rollout episodes, 0 training/refit
  * no validation64 access and no sealed-test access
  * joins v33 schedule -> raw episode -> trace files for all 72 cells
  * verifies per-trace line counts/step sequences/horizon identity and 4013 total
  * writes actual-path supplemental receipts, immutable reviewer bindings, terminal
    bias/parameter values, response-log addendum, state, and backup request
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - only used for bias extraction
    np = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_v33_bookkeeping_preflight_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
D33 = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0_20260930T054827Z"
I33 = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_terminal_identity_evidence_audit_v0b_20260930T072504Z"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
BACKUP_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V33_BOOKKEEPING_PREFLIGHT_{STAMP}.json"
STATE_FILE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v33_bookkeeping_preflight.md"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
ANALYSIS_READY = ROOT / "docs/bohn2021_takeover/astra_reviews/ANALYSIS_READY.json"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-true-variable-horizon-v33-bookkeeping-preflight-v0-{STAMP}"

EXPECTED_EPISODES = 72
EXPECTED_TRACE_LINES = 4013
EXPECTED_HORIZONS = {12, 15, 25, 35}
EXPECTED_TERMINALS = {"zero", "V15_shared", "V35_shared"}
TERMINAL_PARAM_SOURCES = {
    "V15": ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/results/paper_defaults/vehicle_fixed_h15",
    "V35": ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/results/paper_exact_grid_2026-09-23/vehicle_fixed_h35",
}
TERMINAL_INPUTS = ["theta", "x", "y", "goal_x", "goal_y"]


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(x: Any) -> Any:
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if hasattr(x, "item"):
        return clean(x.item())
    return x


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_time(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        out = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def best_effort_sqlite_tokens() -> Dict[str, Any]:
    candidates = [
        ROOT / "research.sqlite",
        ROOT / "research_artifacts/research.sqlite",
        Path("/data/openai-agent/state/research.sqlite"),
        Path("/data/openai-agent/research.sqlite"),
    ]
    for db in candidates:
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db))
            cur = con.cursor()
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            sums = []
            for table in tables:
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                for col in cols:
                    if col == "total_tokens" or col.endswith("_total_tokens"):
                        val = cur.execute(f"SELECT SUM({col}) FROM {table}").fetchone()[0]
                        if val is not None:
                            sums.append({"table": table, "column": col, "sum": int(val)})
            con.close()
            if sums:
                return {"path": str(db), "sums": sums, "best_effort_total_tokens_sum": int(sum(x["sum"] for x in sums))}
            return {"path": str(db), "sums": [], "best_effort_total_tokens_sum": None, "note": "sqlite found but no total_tokens-like column"}
        except Exception as exc:
            return {"path": str(db), "error": repr(exc), "best_effort_total_tokens_sum": None}
    return {"path": None, "best_effort_total_tokens_sum": None, "note": "no sqlite candidate found"}


def snapshot_file(src: Path, dst_name: str) -> Dict[str, Any]:
    out = RUN_DIR / dst_name
    if src.exists():
        data = src.read_bytes()
        out.write_bytes(data)
        return {"source": rel(src), "snapshot": rel(out), "bytes": len(data), "sha256": sha256(src)}
    return {"source": rel(src), "snapshot": None, "exists": False}


def load_terminal_parameters() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    receipts: List[Dict[str, Any]] = []
    if np is None:
        return [], [{"error": "numpy unavailable; terminal bias extraction skipped"}]
    for name, folder in TERMINAL_PARAM_SOURCES.items():
        w_path = folder / "terminal_cnnvf_weights.npy"
        b_path = folder / "terminal_cnnvf_biases.npy"
        rec = {"terminal": name, "folder": rel(folder), "weights_path": rel(w_path), "biases_path": rel(b_path)}
        if not (w_path.exists() and b_path.exists()):
            rec["error"] = "missing weights or biases"
            receipts.append(rec)
            continue
        weights = np.load(str(w_path), allow_pickle=False)
        biases = np.load(str(b_path), allow_pickle=False)
        rec.update({
            "weights_sha256": sha256(w_path),
            "biases_sha256": sha256(b_path),
            "weights_shape": list(weights.shape),
            "biases_shape": list(biases.shape),
            "biases_flat_F": [float(x) for x in np.asarray(biases).flatten(order="F")],
        })
        receipts.append(rec)
        arr = np.asarray(weights, dtype=float)
        bias_flat = np.asarray(biases, dtype=float).flatten(order="F")
        bias = float(bias_flat[0]) if bias_flat.size else 0.0
        if arr.ndim == 2 and arr.shape[0] >= len(TERMINAL_INPUTS) and arr.shape[1] >= 2:
            for i, inp in enumerate(TERMINAL_INPUTS):
                rows.append({
                    "terminal": name,
                    "input": inp,
                    "linear_coeff": float(arr[i, 0]),
                    "quadratic_coeff": float(arr[i, 1]),
                    "second_derivative": float(2.0 * arr[i, 1]),
                    "bias": bias,
                })
        else:
            flat = arr.flatten(order="F")
            for i, val in enumerate(flat):
                rows.append({"terminal": name, "input": f"flat_F_{i}", "linear_coeff": float(val), "quadratic_coeff": None, "second_derivative": None, "bias": bias})
    return rows, receipts


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: clean(row.get(k)) for k in fieldnames})


def line_join_check(raw: Mapping[str, Any]) -> Dict[str, Any]:
    schedule = raw.get("schedule") or []
    episodes = raw.get("episodes") or []
    if len(schedule) != EXPECTED_EPISODES or len(episodes) != EXPECTED_EPISODES:
        raise RuntimeError(f"schedule/episode length mismatch: {len(schedule)} / {len(episodes)}")
    rows: List[Dict[str, Any]] = []
    problems: List[Dict[str, Any]] = []
    seen_cells = set()
    total_lines = 0
    for i, (sch, ep) in enumerate(zip(schedule, episodes)):
        state = str(ep.get("state_label"))
        h = int(ep.get("horizon"))
        term = str(ep.get("terminal_mode"))
        cell = (state, h, term)
        sch_cell = (str(sch.get("state_label")), int(sch.get("horizon")), str(sch.get("terminal_mode")))
        if cell in seen_cells:
            problems.append({"index": i, "problem": "duplicate cell", "cell": cell})
        seen_cells.add(cell)
        if cell != sch_cell:
            problems.append({"index": i, "problem": "schedule/raw mismatch", "schedule_cell": sch_cell, "episode_cell": cell})
        if h not in EXPECTED_HORIZONS or term not in EXPECTED_TERMINALS:
            problems.append({"index": i, "problem": "unexpected H or terminal", "cell": cell})
        trace_diag = ep.get("trace_diagnostics") or {}
        trace_path = ROOT / str(trace_diag.get("trace_path") or "")
        if not trace_path.exists():
            alt = ROOT / str(ep.get("path", "")) / "trace.jsonl"
            trace_path = alt
        summary_path = ROOT / str(ep.get("path", "")) / "summary.json"
        branch_reset_path = ROOT / str(ep.get("path", "")) / "branch_reset.json"
        line_count = 0
        seq_ok = True
        horizon_ok = True
        true_h_ok = True
        first_step = None
        last_step = None
        first_previous_state = None
        first_input = None
        if not trace_path.exists():
            problems.append({"index": i, "problem": "missing trace", "trace_path": rel(trace_path), "cell": cell})
        else:
            with trace_path.open("r", encoding="utf-8") as f:
                for expected_step, line in enumerate(f):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    step = item.get("step")
                    if first_step is None:
                        first_step = step
                        first_previous_state = item.get("previous_state")
                        first_input = item.get("input")
                    last_step = step
                    if step != line_count:
                        seq_ok = False
                    if int(item.get("horizon")) != h:
                        horizon_ok = False
                    if int(item.get("true_mpc_n_horizon")) != h:
                        true_h_ok = False
                    line_count += 1
        total_lines += line_count
        steps = int(ep.get("steps"))
        line_steps_ok = (line_count == steps)
        if not line_steps_ok or not seq_ok or not horizon_ok or not true_h_ok:
            problems.append({
                "index": i,
                "problem": "trace content mismatch",
                "cell": cell,
                "line_count": line_count,
                "steps": steps,
                "seq_ok": seq_ok,
                "horizon_ok": horizon_ok,
                "true_h_ok": true_h_ok,
            })
        rows.append({
            "execution_index": i,
            "v33_execution_index": 33000 + i,
            "state_label": state,
            "horizon": h,
            "terminal_mode": term,
            "raw_steps": steps,
            "trace_line_count": line_count,
            "line_count_matches_steps": line_steps_ok,
            "step_sequence_ok": seq_ok,
            "horizon_ok": horizon_ok,
            "true_horizon_ok": true_h_ok,
            "first_step": first_step,
            "last_step": last_step,
            "trace_path": rel(trace_path),
            "trace_sha256": sha256(trace_path) if trace_path.exists() else None,
            "summary_path": rel(summary_path),
            "summary_sha256": sha256(summary_path) if summary_path.exists() else None,
            "branch_reset_path": rel(branch_reset_path),
            "branch_reset_sha256": sha256(branch_reset_path) if branch_reset_path.exists() else None,
            "first_previous_state_json": json.dumps(clean(first_previous_state), sort_keys=True),
            "first_input_json": json.dumps(clean(first_input), sort_keys=True),
        })
    if total_lines != EXPECTED_TRACE_LINES:
        problems.append({"problem": "total trace line count mismatch", "total_lines": total_lines, "expected": EXPECTED_TRACE_LINES})
    return {
        "schedule_count": len(schedule),
        "episode_count": len(episodes),
        "unique_cell_count": len(seen_cells),
        "trace_total_lines": total_lines,
        "rows": rows,
        "problems": problems,
        "passed": len(problems) == 0,
    }


def actual_path_receipts(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    receipts: List[Dict[str, Any]] = []
    for p in paths:
        if p.is_dir():
            files = sorted([x for x in p.rglob("*") if x.is_file()])
            receipts.append({"path": rel(p), "type": "dir", "exists": True, "file_count": len(files), "bytes": int(sum(x.stat().st_size for x in files))})
        elif p.exists():
            receipts.append({"path": rel(p), "type": "file", "exists": True, "bytes": p.stat().st_size, "sha256": sha256(p)})
        else:
            receipts.append({"path": rel(p), "exists": False})
    return receipts


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-time", default=None)
    ap.add_argument("--backup-commit", default=None)
    ap.add_argument("--backup-package-sha256", default=None)
    ap.add_argument("--backup-package-bytes", type=int, default=0)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        created = now_utc()
        raw_path = D33 / "raw.json"
        if not raw_path.exists():
            raise RuntimeError(f"missing D33 raw: {raw_path}")
        d33_raw = read_json(raw_path)
        join = line_join_check(d33_raw)
        trace_csv = RUN_DIR / "v33_schedule_raw_trace_join.csv"
        write_csv(trace_csv, join["rows"], [
            "execution_index", "v33_execution_index", "state_label", "horizon", "terminal_mode", "raw_steps", "trace_line_count",
            "line_count_matches_steps", "step_sequence_ok", "horizon_ok", "true_horizon_ok", "first_step", "last_step",
            "trace_path", "trace_sha256", "summary_path", "summary_sha256", "branch_reset_path", "branch_reset_sha256",
            "first_previous_state_json", "first_input_json",
        ])
        param_rows, param_receipts = load_terminal_parameters()
        param_csv = RUN_DIR / "terminal_parameters_with_bias.csv"
        write_csv(param_csv, param_rows, ["terminal", "input", "linear_coeff", "quadratic_coeff", "second_derivative", "bias"])
        request_snap = snapshot_file(NEXT_REVIEW_REQUEST, "NEXT_REVIEW_REQUEST_snapshot.json")
        ready_snap = snapshot_file(ANALYSIS_READY, "ANALYSIS_READY_snapshot.json")
        ready = read_json(ANALYSIS_READY) if ANALYSIS_READY.exists() else {}
        report_path = ROOT / str(ready.get("report", "")) if isinstance(ready, dict) and ready.get("report") else None
        report_snap = snapshot_file(report_path, "Astra_report_snapshot.md") if report_path else {"source": None, "exists": False}
        prebackup_time = parse_time(args.backup_time)
        analysis_completed = parse_time(ready.get("completed")) if isinstance(ready, dict) else None
        prebackup = {
            "provided": bool(args.backup_time and args.backup_commit and args.backup_package_sha256),
            "time": prebackup_time.isoformat() if prebackup_time else None,
            "commit": args.backup_commit,
            "package_sha256": args.backup_package_sha256,
            "package_bytes": args.backup_package_bytes,
            "postdates_i33_completed": bool(prebackup_time and prebackup_time > parse_time(read_json(I33 / "completed.json").get("created_utc"))),
            "postdates_latest_analysis_ready_completed": bool(prebackup_time and analysis_completed and prebackup_time > analysis_completed),
            "note": "false for latest Astra report is not a v33 data mismatch; new backup request below covers this preflight and current report/log changes",
        }
        receipts = actual_path_receipts([
            Path(__file__).resolve(), D33, I33, D33 / "raw.json", D33 / "completed.json", I33 / "raw.json", I33 / "completed.json",
            I33 / "terminal_coefficients.csv", trace_csv, param_csv, NEXT_REVIEW_REQUEST, ANALYSIS_READY, report_path if report_path else RUN_DIR / "missing_report",
        ])
        gate = {
            "passed": bool(join["passed"] and join["schedule_count"] == EXPECTED_EPISODES and join["episode_count"] == EXPECTED_EPISODES and join["unique_cell_count"] == EXPECTED_EPISODES and join["trace_total_lines"] == EXPECTED_TRACE_LINES),
            "schedule_raw_trace_join_passed": join["passed"],
            "schedule_count": join["schedule_count"],
            "episode_count": join["episode_count"],
            "unique_cell_count": join["unique_cell_count"],
            "trace_total_lines": join["trace_total_lines"],
            "expected_trace_total_lines": EXPECTED_TRACE_LINES,
            "plant_steps": 0,
            "solver_calls": 0,
            "rollout_episodes": 0,
            "gradient_steps": 0,
            "selector_refits": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        sqlite_usage = best_effort_sqlite_tokens()
        raw = {
            "created_utc": created.isoformat(),
            "method": NAME,
            "classification": "development_IMPROVED_zero_rollout_bookkeeping_preflight_not_validation_not_test",
            "hypothesis": "Astra A12 residual gap is bookkeeping/registry incompleteness, not a v33 data/trace mismatch.",
            "gate": gate,
            "join_summary": {k: v for k, v in join.items() if k != "rows"},
            "join_table_csv": rel(trace_csv),
            "terminal_parameters_with_bias_csv": rel(param_csv),
            "terminal_parameter_receipts": param_receipts,
            "immutable_bindings": {"next_review_request": request_snap, "analysis_ready": ready_snap, "astra_report": report_snap},
            "actual_path_receipts": receipts,
            "pre_run_backup_context": prebackup,
            "sqlite_usage_best_effort": sqlite_usage,
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "limitations": [
                "metadata/file integrity only; no solver or plant calls",
                "does not independently recompute all dynamics or objectives",
                "does not access validation64 or sealed/final test",
                "new preflight artifacts still require external backup before unique solver science",
            ],
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
        }
        write_json(RUN_DIR / "raw.json", raw)
        receipt_path = RUN_DIR / "supplemental_actual_path_receipts.json"
        write_json(receipt_path, {"created_utc": created.isoformat(), "receipts": receipts, "registry_placeholder_note": "Original run registry retained <STAMP> placeholder paths; this supplemental receipt records actual existing artifacts without mutating historical registry."})
        # Summary and docs.
        summary_lines = [
            "# v33 bookkeeping preflight before objective-vs-basin solver probe",
            "",
            f"created_utc: `{created.isoformat()}`",
            "",
            "## Gate",
            f"- passed: `{gate['passed']}`",
            f"- schedule_count / episode_count / unique_cell_count: `{gate['schedule_count']}` / `{gate['episode_count']}` / `{gate['unique_cell_count']}`",
            f"- trace_total_lines: `{gate['trace_total_lines']}` (expected `{EXPECTED_TRACE_LINES}`)",
            f"- mismatch/problem count: `{len(join['problems'])}`",
            "- budget: 0 rollout episodes, 0 plant steps, 0 solver calls, 0 gradient/refit steps, validation64 closed, sealed test closed",
            "",
            "## Outputs",
            f"- join table: `{rel(trace_csv)}`",
            f"- terminal parameters with bias: `{rel(param_csv)}`",
            f"- actual-path receipts: `{rel(receipt_path)}`",
            f"- raw: `{rel(RUN_DIR / 'raw.json')}`",
            "",
            "## Backup context",
            f"- provided pre-run backup: `{prebackup}`",
            f"- sqlite total_tokens best effort: `{sqlite_usage}`",
            "",
            "## Decision",
            "If this gate remains passed after backup, Astra 20260930T072558Z authorizes proceeding to the fixed 24-call objective-vs-basin solver probe; no repeat identity audit or v33 72-rollout rerun is indicated.",
        ]
        if join["problems"]:
            summary_lines += ["", "## Problems", "```json", json.dumps(clean(join["problems"][:20]), indent=2, sort_keys=True), "```"]
        (RUN_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        block = f"""
<!-- {MARKER} -->
## v33 bookkeeping preflight before objective-vs-basin solver probe

Updated by GPT-5.5 executor at `{created.isoformat()}`. Zero-rollout metadata/file-integrity diagnostic requested by Astra report `20260930T072558Z`: 0 plant steps, 0 solver calls, 0 training/refit, validation64 closed, sealed test closed.

| linked recommendation(s) | disposition | verified evidence | action / next step |
|---|---|---|---|
| `A12_registry_backup_schema_contract` | accepted and executed | `{rel(RUN_DIR / 'summary.md')}` and `{rel(RUN_DIR / 'raw.json')}`: schedule/raw/trace join passed=`{gate['passed']}`, counts `{gate['schedule_count']}`/`{gate['episode_count']}`/`{gate['unique_cell_count']}`, trace lines `{gate['trace_total_lines']}`; mismatch count `{len(join['problems'])}`. | Supplemental actual-path receipt `{rel(receipt_path)}` and terminal bias table `{rel(param_csv)}` written. Historical registry placeholders are retained, not overwritten. |
| `A11_training_failure_modes_need_separation`, `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | ready for next frozen discriminator if backup clears | This preflight adds no new science and does not alter v33 interpretation; it confirms no schedule/trace mismatch blocker was found. | After external backup covers this preflight/report/log state, proceed to Astra's fixed 24-call objective-vs-basin solver probe; do not repeat terminal identity audit. |
"""
        append_if_missing(RESPONSE_LOG, MARKER, block)
        for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md"]:
            append_if_missing(ROOT / doc, MARKER, f"<!-- {MARKER} -->\n## v33 bookkeeping preflight\n\nUTC `{created.isoformat()}`: gate passed `{gate['passed']}`; schedule/raw/trace counts `{gate['schedule_count']}`/`{gate['episode_count']}`/`{gate['unique_cell_count']}`, trace lines `{gate['trace_total_lines']}`; zero rollout/solver/training, validation64 and sealed test closed. Artifacts `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`. Backup required: `{rel(BACKUP_REQUEST)}`.")
        write_json(BACKUP_REQUEST, {
            "request": "backup_after_v33_bookkeeping_preflight",
            "created_utc": created.isoformat(),
            "backup_required_before_24_call_solver_probe_or_other_unique_science": True,
            "must_cover": [
                rel(Path(__file__).resolve()), rel(RUN_DIR), rel(BACKUP_REQUEST), rel(STATE_FILE), rel(RESPONSE_LOG),
                "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "EXPERIMENT_REGISTRY.csv",
            ],
            "budget_actual": gate,
            "sealed_test_accessed": False,
            "validation64_bank_opened": False,
        })
        state_txt = f"# Continue state after v33 bookkeeping preflight\n\nUTC: {created.isoformat()}\n\nGate passed: {gate['passed']}\n\nCounts: schedule={gate['schedule_count']} episodes={gate['episode_count']} unique_cells={gate['unique_cell_count']} trace_lines={gate['trace_total_lines']} expected={EXPECTED_TRACE_LINES}\n\nArtifacts: {rel(RUN_DIR / 'summary.md')}, {rel(RUN_DIR / 'raw.json')}, {rel(trace_csv)}, {rel(param_csv)}\n\nBackup required before unique solver science: {rel(BACKUP_REQUEST)}\n\nNext if backup clears and no newer Astra contrary report: implement/run fixed 24-call objective-vs-basin solver probe exactly as frozen in Astra 20260930T072558Z.\n"
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(state_txt, encoding="utf-8")
        reg = ROOT / "EXPERIMENT_REGISTRY.csv"
        with reg.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([created.isoformat(), NAME, raw["classification"], "zero_rollout_A12_preflight", "opened_v33_metadata_only_no_validation_no_test", 0, 0, 0, 0, 0, False, rel(RUN_DIR / "completed.json"), MARKER])
        files = [Path(__file__).resolve(), RUN_DIR / "raw.json", RUN_DIR / "summary.md", trace_csv, param_csv, receipt_path, BACKUP_REQUEST, STATE_FILE, RESPONSE_LOG, reg]
        completed = {
            "status": "complete",
            "passed": gate["passed"],
            "created_utc": created.isoformat(),
            "classification": raw["classification"],
            "budget_actual": gate,
            "summary": rel(RUN_DIR / "summary.md"),
            "raw": rel(RUN_DIR / "raw.json"),
            "join_table_csv": rel(trace_csv),
            "terminal_parameters_with_bias_csv": rel(param_csv),
            "backup_request": rel(BACKUP_REQUEST),
            "sqlite_usage_best_effort": sqlite_usage,
            "elapsed_since_first_supervisor_event_seconds": raw["elapsed_since_first_supervisor_event_seconds"],
            "hashes": {rel(p): sha256(p) for p in files if p.exists()},
        }
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({
            "completed": rel(RUN_DIR / "completed.json"),
            "summary": rel(RUN_DIR / "summary.md"),
            "passed": gate["passed"],
            "trace_total_lines": gate["trace_total_lines"],
            "mismatch_count": len(join["problems"]),
            "plant_steps": 0,
            "solver_calls": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "elapsed_since_first_supervisor_event_seconds": raw["elapsed_since_first_supervisor_event_seconds"],
            "sqlite_total_tokens_best_effort": sqlite_usage.get("best_effort_total_tokens_sum"),
            "backup_request": rel(BACKUP_REQUEST),
        }, sort_keys=True), flush=True)
        return 0 if gate["passed"] else 2
    except Exception as exc:
        write_json(RUN_DIR / "failed.json", {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "plant_steps": 0, "solver_calls": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "plant_steps": 0, "solver_calls": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
