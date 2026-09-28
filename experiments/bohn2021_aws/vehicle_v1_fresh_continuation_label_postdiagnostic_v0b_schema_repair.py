#!/usr/bin/env python3
"""Schema repair for Vehicle V1 fresh continuation-label postdiagnostic v0.

The v0 postdiagnostic completed its read-only analysis and wrote raw/summary/
state/protocol/backup-request artifacts, but then failed while appending docs
because the doc appender used the stale key name
``best_non_H15_total_gain_candidates`` instead of
``max_non_H15_total_gain_candidates``.  This script performs no simulations and
no training.  It validates the partial v0 outputs, writes a versioned v0b
completion/audit wrapper, appends documentation with the correct key, and keeps
the transient-state continuation protocol frozen for execution only after an
external backup.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T1505Z"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0b_schema_repair_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_v1_fresh_continuation_label_postdiagnostic_v0b_schema_repair_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
BACKUP_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_V1_FRESH_CONTINUATION_LABEL_POSTDIAGNOSTIC_V0B_SCHEMA_REPAIR_{STAMP}.json"

FAILED_V0_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0_20260928T1500Z"
FAILED_V0_RAW = FAILED_V0_DIR / "raw.json"
FAILED_V0_SUMMARY = FAILED_V0_DIR / "summary.md"
FAILED_V0_STATE = ROOT / "research_artifacts/aws_state/vehicle_v1_fresh_continuation_label_postdiagnostic_v0_20260928T1500Z.md"
FAILED_V0_BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1_FRESH_CONTINUATION_LABEL_POSTDIAGNOSTIC_V0_20260928T1500Z.json"
V0_RUN_REGISTRY = ROOT / "research_artifacts/aws_runs/20260928T145806_b791973a/registry.json"
FRESH_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928/completed.json"
FRESH_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928/summary.md"
NEXT_PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.json"
NEXT_PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.md"
MARKER = "vehicle-v1-fresh-continuation-label-postdiagnostic-v0b-schema-repair-20260928T1505Z"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(rel(path))


def get_nested(obj: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def append_once(path: Path, marker: str, block: str) -> None:
    if not path.exists():
        return
    old = path.read_text(encoding="utf-8")
    if marker not in old:
        path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def validate_v0(raw: Dict[str, Any], fresh_done: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if raw.get("historical_validation64_bank_opened") is not False:
        issues.append("v0 raw historical_validation64_bank_opened is not false")
    if raw.get("sealed_test_accessed") is not False:
        issues.append("v0 raw sealed_test_accessed is not false")
    if raw.get("new_rollouts") != 0 or raw.get("new_control_steps") != 0:
        issues.append("v0 raw reports nonzero rollout/control budget")
    if raw.get("new_training_episodes") != 0 or raw.get("new_gradient_steps") != 0:
        issues.append("v0 raw reports nonzero training/gradient budget")
    if fresh_done.get("passed") is not True or fresh_done.get("hard_pass") is not True:
        issues.append("fresh-probe completed marker is not passed/hard_pass")
    if fresh_done.get("sealed_test_accessed") is not False:
        issues.append("fresh-probe completed marker sealed_test_accessed is not false")
    if fresh_done.get("historical_validation64_bank_opened") not in (None, False):
        issues.append("fresh-probe completed marker historical_validation64_bank_opened is not false")
    fp = raw.get("fresh_probe_headline", {})
    if fp.get("positive_state_count") != 0 or fp.get("informative_state_count") != 24:
        issues.append("unexpected fresh-probe headline; expected 0/24 positives")
    if get_nested(raw, ["diagnostic_stats", "max_non_H15_total_gain_candidates"], None) is None:
        issues.append("v0 raw lacks corrected max_non_H15_total_gain_candidates key")
    if get_nested(raw, ["diagnostic_stats", "horizon_stats"], None) is None:
        issues.append("v0 raw lacks horizon_stats")
    if not NEXT_PROTOCOL_JSON.exists() or not NEXT_PROTOCOL_MD.exists():
        issues.append("transient-state protocol artifacts are missing")
    return issues


def write_summary(out: Dict[str, Any]) -> None:
    fp = out["fresh_probe_headline"]
    best_total = out["best_non_h15_total_candidate"]
    hstats = out["diagnostic_stats"].get("horizon_stats", {})
    lines = [
        "# Vehicle V1 fresh continuation-label postdiagnostic v0b schema repair",
        "",
        f"Created UTC: `{out['created_utc']}`.",
        "",
        "This is a no-simulation/no-training completion repair for the failed v0 postdiagnostic. The predecessor v0 wrote valid raw/summary/protocol artifacts, then failed only during documentation append because of a stale key name. This v0b wrapper preserves that failure and completes the audit with the corrected key name.",
        "",
        "## Access and budget",
        "",
        "- New rollouts/control steps/training episodes/gradient steps: `0 / 0 / 0 / 0`.",
        "- historical_validation64_bank_opened: `False`; sealed_test_accessed: `False`.",
        f"- Failed predecessor raw: `{out['failed_v0_artifacts']['raw']}` sha256 `{out['failed_v0_artifacts']['raw_sha256']}`.",
        f"- Failed predecessor summary: `{out['failed_v0_artifacts']['summary']}` sha256 `{out['failed_v0_artifacts']['summary_sha256']}`.",
        "",
        "## Restated findings",
        "",
        f"- Fresh continuation-label probe: `{fp['positive_state_count']}` material-positive non-H15 states out of `{fp['informative_state_count']}` informative H15-prefix states; refit/training gate `{fp['fresh_refit_training_gate_pass']}`.",
        f"- Best non-H15 total-gain candidate: `{best_total}`.",
        f"- Large non-H15 harms (gain <= -3 physical or total): `{out['diagnostic_stats'].get('large_harm_count')}`.",
        "",
        "| H | states | material positives | large harms | max physical gain | max total gain | mean physical gain | mean total gain |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in sorted(hstats, key=lambda x: int(x)):
        row = hstats[h]
        pg = row.get("physical_gain_vs_H15", {})
        tg = row.get("total_gain_vs_H15", {})
        lines.append(
            "| %s | %s | %s | %s | %.6g | %.6g | %.6g | %.6g |" % (
                h,
                row.get("states"),
                row.get("material_count"),
                row.get("large_harm_count_vs_H15_threshold3"),
                float(pg.get("max") or 0.0),
                float(tg.get("max") or 0.0),
                float(pg.get("mean") or 0.0),
                float(tg.get("mean") or 0.0),
            )
        )
    lines += [
        "",
        "## Decision preserved",
        "",
        "The mined nearest-state selector remains development-only and over-concentrated in earlier mined cases. Fresh generic branch labels do not support immediate refit/retraining. The next frozen action remains the transient-state continuation probe, because it directly tests whether the generic branch schedule missed high-error/near-obstacle opportunities before moving to stress-scenario redesign.",
        "",
        f"- Frozen next protocol: `{out['next_protocol']['md']}` / `{out['next_protocol']['json']}`.",
        f"- Backup request before any rollout: `{out['backup_request']}`.",
        "",
        "## Failure accounting",
        "",
        "The v0 run is retained as a failed no-simulation diagnostic with exit status 1. The failure changed documentation/completion only; it did not alter scientific outcomes or consume simulations. This v0b repair is the authoritative completion marker for the postdiagnostic stage.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    for path in [FAILED_V0_RAW, FAILED_V0_SUMMARY, FAILED_V0_BACKUP_REQUEST, FAILED_V0_STATE, V0_RUN_REGISTRY, FRESH_DONE, FRESH_SUMMARY, NEXT_PROTOCOL_JSON, NEXT_PROTOCOL_MD]:
        require(path)
    raw = read_json(FAILED_V0_RAW)
    fresh_done = read_json(FRESH_DONE)
    issues = validate_v0(raw, fresh_done)
    if issues:
        write_json(OUT_DIR / "failed_validation.json", {"created_utc": created, "issues": issues})
        raise RuntimeError("; ".join(issues))

    candidates = raw["diagnostic_stats"].get("max_non_H15_total_gain_candidates", [])
    best_total = candidates[0] if candidates else None
    out = {
        "created_utc": created,
        "method": "vehicle_v1_fresh_continuation_label_postdiagnostic_v0b_schema_repair_no_simulation",
        "classification": "schema_repair_no_simulation_no_training_not_validation_not_final_test",
        "predecessor_failure": {
            "run_registry": rel(V0_RUN_REGISTRY),
            "exit_status": 1,
            "cause": "append_docs used stale key best_non_H15_total_gain_candidates; raw/summary/protocol/back-request/state had already been written",
            "scientific_budget_consumed_by_failed_v0": {"new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0},
        },
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "fresh_probe_headline": raw["fresh_probe_headline"],
        "diagnostic_stats": raw["diagnostic_stats"],
        "best_non_h15_total_candidate": best_total,
        "decision": raw["decision"],
        "next_protocol": {
            "json": rel(NEXT_PROTOCOL_JSON),
            "json_sha256": sha256(NEXT_PROTOCOL_JSON),
            "md": rel(NEXT_PROTOCOL_MD),
            "md_sha256": sha256(NEXT_PROTOCOL_MD),
        },
        "failed_v0_artifacts": {
            "raw": rel(FAILED_V0_RAW),
            "raw_sha256": sha256(FAILED_V0_RAW),
            "summary": rel(FAILED_V0_SUMMARY),
            "summary_sha256": sha256(FAILED_V0_SUMMARY),
            "state": rel(FAILED_V0_STATE),
            "state_sha256": sha256(FAILED_V0_STATE),
            "backup_request": rel(FAILED_V0_BACKUP_REQUEST),
            "backup_request_sha256": sha256(FAILED_V0_BACKUP_REQUEST),
        },
        "inputs": {
            rel(FRESH_DONE): sha256(FRESH_DONE),
            rel(FRESH_SUMMARY): sha256(FRESH_SUMMARY),
            rel(V0_RUN_REGISTRY): sha256(V0_RUN_REGISTRY),
            rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        },
    }
    write_json(OUT_DIR / "raw.json", out)
    write_summary(out)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    write_json(BACKUP_REQUEST, {
        "requested_utc": created,
        "reason": "backup v0 failed no-simulation diagnostic artifacts plus v0b completion repair and frozen transient-state protocol before any further rollout/simulation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "artifacts": [
            rel(OUT_DIR),
            rel(STATE_PATH),
            rel(BACKUP_REQUEST),
            rel(FAILED_V0_RAW),
            rel(FAILED_V0_SUMMARY),
            rel(FAILED_V0_BACKUP_REQUEST),
            rel(FAILED_V0_STATE),
            rel(V0_RUN_REGISTRY),
            rel(NEXT_PROTOCOL_JSON),
            rel(NEXT_PROTOCOL_MD),
            rel(Path(__file__).resolve()),
        ],
    })
    out["backup_request"] = rel(BACKUP_REQUEST)
    # Re-write raw with backup request populated.
    write_json(OUT_DIR / "raw.json", out)

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 fresh continuation-label postdiagnostic v0b schema repair state ({created})\n\n"
        "No simulations/training/validation64/test access. v0 failed only during doc append after writing raw/summary/protocol. "
        f"Fresh labels remain 0/{raw['fresh_probe_headline']['informative_state_count']} material positives; next frozen action is {rel(NEXT_PROTOCOL_MD)} after external backup. "
        f"Backup request: {rel(BACKUP_REQUEST)}.\n",
        encoding="utf-8",
    )
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 fresh continuation-label postdiagnostic v0b schema repair\n\n"
        f"UTC: {created}. No simulations, no training/refit, no historical validation64, no sealed test. "
        "The preceding v0 postdiagnostic failed only in a documentation append due to a stale key name; v0 raw/summary/protocol are preserved as failed-run evidence. "
        f"Fresh continuation labels remain {raw['fresh_probe_headline']['positive_state_count']}/{raw['fresh_probe_headline']['informative_state_count']} material positives, so immediate selector refit/retraining is deferred. "
        f"Next frozen diagnostic: `{rel(NEXT_PROTOCOL_MD)}`; execute only after a verified external backup covers v0/v0b outputs and the protocol. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / name, MARKER, block)

    files_for_hash = [
        OUT_DIR / "summary.md",
        OUT_DIR / "raw.json",
        STATE_PATH,
        BACKUP_REQUEST,
        FAILED_V0_RAW,
        FAILED_V0_SUMMARY,
        FAILED_V0_STATE,
        FAILED_V0_BACKUP_REQUEST,
        NEXT_PROTOCOL_JSON,
        NEXT_PROTOCOL_MD,
        V0_RUN_REGISTRY,
        Path(__file__).resolve(),
    ]
    completed = {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_request": rel(BACKUP_REQUEST),
        "headline": {
            "predecessor_exit_status": 1,
            "fresh_positive_states": raw["fresh_probe_headline"]["positive_state_count"],
            "fresh_informative_states": raw["fresh_probe_headline"]["informative_state_count"],
            "fresh_refit_training_gate_pass": raw["fresh_probe_headline"]["fresh_refit_training_gate_pass"],
            "best_non_h15_total_gain": best_total.get("total_gain") if isinstance(best_total, dict) else None,
            "large_harm_count": raw["diagnostic_stats"].get("large_harm_count"),
            "next_protocol": {"json": rel(NEXT_PROTOCOL_JSON), "md": rel(NEXT_PROTOCOL_MD)},
            "backup_required_before_more_simulations": True,
        },
        "hashes": {rel(p): sha256(p) for p in files_for_hash if p.exists()},
    }
    write_json(OUT_DIR / "completed.json", completed)
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "fresh_positive_states": raw["fresh_probe_headline"]["positive_state_count"],
        "fresh_informative_states": raw["fresh_probe_headline"]["informative_state_count"],
        "best_non_h15_total_gain": best_total.get("total_gain") if isinstance(best_total, dict) else None,
        "large_harm_count": raw["diagnostic_stats"].get("large_harm_count"),
        "next_protocol_json": rel(NEXT_PROTOCOL_JSON),
        "backup_request": rel(BACKUP_REQUEST),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
