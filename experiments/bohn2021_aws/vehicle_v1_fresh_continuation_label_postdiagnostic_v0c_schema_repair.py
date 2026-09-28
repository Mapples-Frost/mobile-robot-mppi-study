#!/usr/bin/env python3
"""Robust no-simulation completion repair for Vehicle V1 fresh-label postdiagnostic.

Context preserved:
- v0 completed the substantive read-only postdiagnostic and froze the transient-state
  continuation protocol, but exited 1 while appending documentation because it
  referenced a stale raw key name.
- v0b attempted a schema repair, validated the scientific outputs and wrote a raw
  wrapper, but exited 1 before summary/completed because the summary writer read
  ``out['backup_request']`` before that field was populated.

This v0c script performs no rollouts, no training/refit, no validation64-bank
access, and no sealed-test access. It validates the existing v0/v0b artifacts,
accounts for both failed no-simulation repairs, writes a completed marker, and
leaves the frozen transient-state continuation probe unchanged. Execute it only
as a bounded no-simulation diagnostic.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T1510Z"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0c_schema_repair_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_v1_fresh_continuation_label_postdiagnostic_v0c_schema_repair_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
BACKUP_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_V1_FRESH_CONTINUATION_LABEL_POSTDIAGNOSTIC_V0C_SCHEMA_REPAIR_{STAMP}.json"

FAILED_V0_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0_20260928T1500Z"
FAILED_V0_RAW = FAILED_V0_DIR / "raw.json"
FAILED_V0_SUMMARY = FAILED_V0_DIR / "summary.md"
FAILED_V0_STATE = ROOT / "research_artifacts/aws_state/vehicle_v1_fresh_continuation_label_postdiagnostic_v0_20260928T1500Z.md"
FAILED_V0_BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1_FRESH_CONTINUATION_LABEL_POSTDIAGNOSTIC_V0_20260928T1500Z.json"
FAILED_V0_RUN_REGISTRY = ROOT / "research_artifacts/aws_runs/20260928T145806_b791973a/registry.json"

FAILED_V0B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0b_schema_repair_20260928T1505Z/raw.json"
FAILED_V0B_RUN_REGISTRY = ROOT / "research_artifacts/aws_runs/20260928T150458_c0c16c13/registry.json"
FAILED_V0B_SOURCE = ROOT / "experiments/bohn2021_aws/vehicle_v1_fresh_continuation_label_postdiagnostic_v0b_schema_repair.py"

FRESH_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928/completed.json"
FRESH_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928/summary.md"
NEXT_PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.json"
NEXT_PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.md"
SOURCE_PATH = Path(__file__).resolve()
MARKER = "vehicle-v1-fresh-continuation-label-postdiagnostic-v0c-schema-repair-20260928T1510Z"


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


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def append_once(path: Path, marker: str, block: str) -> None:
    if not path.exists():
        return
    old = path.read_text(encoding="utf-8")
    if marker not in old:
        path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def validate_inputs(raw: Dict[str, Any], fresh_done: Dict[str, Any], v0b_raw: Optional[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    if raw.get("historical_validation64_bank_opened") is not False:
        issues.append("v0 raw historical_validation64_bank_opened is not False")
    if raw.get("sealed_test_accessed") is not False:
        issues.append("v0 raw sealed_test_accessed is not False")
    for key in ["new_rollouts", "new_control_steps", "new_training_episodes", "new_gradient_steps"]:
        if raw.get(key) != 0:
            issues.append(f"v0 raw reports {key}={raw.get(key)!r}, expected 0")
    if fresh_done.get("passed") is not True or fresh_done.get("hard_pass") is not True:
        issues.append("fresh-probe completed marker is not passed/hard_pass")
    if fresh_done.get("sealed_test_accessed") is not False:
        issues.append("fresh-probe completed marker sealed_test_accessed is not False")
    if fresh_done.get("historical_validation64_bank_opened") not in (None, False):
        issues.append("fresh-probe completed marker historical_validation64_bank_opened is not False/None")
    fp = raw.get("fresh_probe_headline", {})
    if fp.get("positive_state_count") != 0:
        issues.append(f"unexpected positive_state_count={fp.get('positive_state_count')!r}; expected 0")
    if fp.get("informative_state_count") != 24:
        issues.append(f"unexpected informative_state_count={fp.get('informative_state_count')!r}; expected 24")
    if fp.get("fresh_refit_training_gate_pass") is not False:
        issues.append("fresh_refit_training_gate_pass is not False")
    if get_nested(raw, ["diagnostic_stats", "max_non_H15_total_gain_candidates"], None) is None:
        issues.append("v0 raw lacks corrected max_non_H15_total_gain_candidates key")
    if get_nested(raw, ["diagnostic_stats", "horizon_stats"], None) is None:
        issues.append("v0 raw lacks horizon_stats")
    for protocol in [NEXT_PROTOCOL_JSON, NEXT_PROTOCOL_MD]:
        if not protocol.exists():
            issues.append(f"missing next protocol artifact {rel(protocol)}")
    if v0b_raw is not None:
        if v0b_raw.get("historical_validation64_bank_opened") is not False:
            issues.append("v0b raw historical_validation64_bank_opened is not False")
        if v0b_raw.get("sealed_test_accessed") is not False:
            issues.append("v0b raw sealed_test_accessed is not False")
        if get_nested(v0b_raw, ["fresh_probe_headline", "positive_state_count"], None) != 0:
            issues.append("v0b raw does not preserve 0 positive states")
    return issues


def make_summary(out: Dict[str, Any]) -> str:
    fp = out["fresh_probe_headline"]
    hstats = out["diagnostic_stats"].get("horizon_stats", {})
    best_total = out.get("best_non_h15_total_candidate")
    lines = [
        "# Vehicle V1 fresh continuation-label postdiagnostic v0c schema repair",
        "",
        f"Created UTC: `{out['created_utc']}`.",
        "",
        "This is a no-simulation/no-training completion repair. It preserves two failed no-simulation documentation/completion attempts and completes the postdiagnostic audit without altering the frozen transient-state continuation protocol.",
        "",
        "## Access and budget",
        "",
        "- New rollouts/control steps/training episodes/gradient steps: `0 / 0 / 0 / 0`.",
        "- historical_validation64_bank_opened: `False`; sealed_test_accessed: `False`.",
        f"- v0 failed run registry: `{out['failure_accounting']['v0']['run_registry']}`.",
        f"- v0b failed run registry: `{out['failure_accounting']['v0b']['run_registry']}`.",
        "",
        "## Restated scientific findings",
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
        pg = row.get("physical_gain_vs_H15", {}) or {}
        tg = row.get("total_gain_vs_H15", {}) or {}
        lines.append(
            "| %s | %s | %s | %s | %.6g | %.6g | %.6g | %.6g |" % (
                h,
                row.get("states"),
                row.get("material_count"),
                row.get("large_harm_count_vs_H15_threshold3"),
                safe_float(pg.get("max")),
                safe_float(tg.get("max")),
                safe_float(pg.get("mean")),
                safe_float(tg.get("mean")),
            )
        )
    lines += [
        "",
        "## Decision preserved",
        "",
        "The mined nearest-state selector remains development-only and over-concentrated in earlier mined cases. The fresh generic branch labels do not justify immediate refit/retraining or unchanged validation. The next frozen action remains the transient-state continuation probe, which tests whether generic branch scheduling missed high-error/near-obstacle opportunities before moving to stress-scenario redesign.",
        "",
        f"- Frozen next protocol: `{out['next_protocol']['md']}` / `{out['next_protocol']['json']}`.",
        f"- Backup request before any rollout: `{out['backup_request']}`.",
        "",
        "## Failure accounting",
        "",
        "- v0 exit status 1: documentation append referenced stale key `best_non_H15_total_gain_candidates`; scientific raw/summary/protocol were already written.",
        "- v0b exit status 1: summary writer referenced `backup_request` before populating it; validation raw was already written.",
        "- v0c is the authoritative completion marker for this postdiagnostic stage. Both earlier failures are retained as evidence and consumed no simulation/training budget.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    required = [
        FAILED_V0_RAW,
        FAILED_V0_SUMMARY,
        FAILED_V0_STATE,
        FAILED_V0_BACKUP_REQUEST,
        FAILED_V0_RUN_REGISTRY,
        FAILED_V0B_RUN_REGISTRY,
        FAILED_V0B_SOURCE,
        FRESH_DONE,
        FRESH_SUMMARY,
        NEXT_PROTOCOL_JSON,
        NEXT_PROTOCOL_MD,
    ]
    for path in required:
        require(path)
    raw = read_json(FAILED_V0_RAW)
    fresh_done = read_json(FRESH_DONE)
    v0b_raw = read_json(FAILED_V0B_RAW) if FAILED_V0B_RAW.exists() else None
    issues = validate_inputs(raw, fresh_done, v0b_raw)
    if issues:
        write_json(OUT_DIR / "failed_validation.json", {"created_utc": created, "issues": issues})
        raise RuntimeError("; ".join(issues))

    candidates = raw["diagnostic_stats"].get("max_non_H15_total_gain_candidates", [])
    best_total = candidates[0] if candidates else None
    source_hash = sha256(SOURCE_PATH)
    out: Dict[str, Any] = {
        "created_utc": created,
        "method": "vehicle_v1_fresh_continuation_label_postdiagnostic_v0c_schema_repair_no_simulation",
        "classification": "schema_repair_no_simulation_no_training_not_validation_not_final_test",
        "formal_scientific_evidence": False,
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
        "backup_request": rel(BACKUP_REQUEST),
        "next_protocol": {
            "json": rel(NEXT_PROTOCOL_JSON),
            "json_sha256": sha256(NEXT_PROTOCOL_JSON),
            "md": rel(NEXT_PROTOCOL_MD),
            "md_sha256": sha256(NEXT_PROTOCOL_MD),
        },
        "failure_accounting": {
            "v0": {
                "run_registry": rel(FAILED_V0_RUN_REGISTRY),
                "run_registry_sha256": sha256(FAILED_V0_RUN_REGISTRY),
                "exit_status": 1,
                "cause": "append_docs used stale key best_non_H15_total_gain_candidates after writing raw/summary/protocol/state/backup request",
                "budget": {"new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0},
            },
            "v0b": {
                "run_registry": rel(FAILED_V0B_RUN_REGISTRY),
                "run_registry_sha256": sha256(FAILED_V0B_RUN_REGISTRY),
                "raw": rel(FAILED_V0B_RAW) if FAILED_V0B_RAW.exists() else None,
                "raw_sha256": sha256(FAILED_V0B_RAW) if FAILED_V0B_RAW.exists() else None,
                "exit_status": 1,
                "cause": "summary writer referenced backup_request before it was populated; no completed marker written",
                "budget": {"new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0},
            },
        },
        "input_hashes": {
            rel(FAILED_V0_RAW): sha256(FAILED_V0_RAW),
            rel(FAILED_V0_SUMMARY): sha256(FAILED_V0_SUMMARY),
            rel(FAILED_V0_STATE): sha256(FAILED_V0_STATE),
            rel(FAILED_V0_BACKUP_REQUEST): sha256(FAILED_V0_BACKUP_REQUEST),
            rel(FAILED_V0B_SOURCE): sha256(FAILED_V0B_SOURCE),
            rel(FRESH_DONE): sha256(FRESH_DONE),
            rel(FRESH_SUMMARY): sha256(FRESH_SUMMARY),
            rel(SOURCE_PATH): source_hash,
        },
    }

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    write_json(BACKUP_REQUEST, {
        "requested_utc": created,
        "reason": "backup failed v0/v0b no-simulation diagnostics plus v0c completion repair and frozen transient-state protocol before any further rollout/simulation",
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
            rel(FAILED_V0_STATE),
            rel(FAILED_V0_BACKUP_REQUEST),
            rel(FAILED_V0_RUN_REGISTRY),
            rel(FAILED_V0B_RAW),
            rel(FAILED_V0B_RUN_REGISTRY),
            rel(FAILED_V0B_SOURCE),
            rel(NEXT_PROTOCOL_JSON),
            rel(NEXT_PROTOCOL_MD),
            rel(SOURCE_PATH),
        ],
    })

    write_json(OUT_DIR / "raw.json", out)
    (OUT_DIR / "summary.md").write_text(make_summary(out), encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 fresh continuation-label postdiagnostic v0c schema repair state ({created})\n\n"
        "No simulations/training/validation64/test access. v0 and v0b failures are preserved as documentation/completion errors only. "
        f"Fresh labels remain 0/{raw['fresh_probe_headline']['informative_state_count']} material-positive non-H15 states; immediate refit/retraining remains deferred. "
        f"Next frozen action is {rel(NEXT_PROTOCOL_MD)} after verified external backup. Backup request: {rel(BACKUP_REQUEST)}.\n",
        encoding="utf-8",
    )

    completed_paths = [
        OUT_DIR / "summary.md",
        OUT_DIR / "raw.json",
        STATE_PATH,
        BACKUP_REQUEST,
        FAILED_V0_RAW,
        FAILED_V0_SUMMARY,
        FAILED_V0_STATE,
        FAILED_V0_BACKUP_REQUEST,
        FAILED_V0_RUN_REGISTRY,
        FAILED_V0B_SOURCE,
        FAILED_V0B_RUN_REGISTRY,
        NEXT_PROTOCOL_JSON,
        NEXT_PROTOCOL_MD,
        SOURCE_PATH,
    ]
    if FAILED_V0B_RAW.exists():
        completed_paths.append(FAILED_V0B_RAW)
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
            "predecessor_failures": ["v0", "v0b"],
            "fresh_positive_states": raw["fresh_probe_headline"]["positive_state_count"],
            "fresh_informative_states": raw["fresh_probe_headline"]["informative_state_count"],
            "fresh_refit_training_gate_pass": raw["fresh_probe_headline"]["fresh_refit_training_gate_pass"],
            "best_non_h15_total_gain": best_total.get("total_gain") if isinstance(best_total, dict) else None,
            "large_harm_count": raw["diagnostic_stats"].get("large_harm_count"),
            "next_protocol": {"json": rel(NEXT_PROTOCOL_JSON), "md": rel(NEXT_PROTOCOL_MD)},
            "backup_required_before_more_simulations": True,
        },
        "hashes": {rel(p): sha256(p) for p in completed_paths if p.exists()},
    }
    write_json(OUT_DIR / "completed.json", completed)

    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 fresh continuation-label postdiagnostic v0c schema repair\n\n"
        f"UTC: {created}. No simulations, no training/refit, no historical validation64, no sealed test. "
        "The preceding v0 and v0b postdiagnostic repair attempts failed only in documentation/completion code after validating/writing partial artifacts. "
        f"Fresh continuation labels remain {raw['fresh_probe_headline']['positive_state_count']}/{raw['fresh_probe_headline']['informative_state_count']} material positives; immediate selector refit/retraining remains deferred. "
        f"Next frozen diagnostic: `{rel(NEXT_PROTOCOL_MD)}`; execute only after a verified external backup covers v0/v0b/v0c outputs and the protocol. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / name, MARKER, block)

    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "predecessor_failures": ["v0", "v0b"],
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
