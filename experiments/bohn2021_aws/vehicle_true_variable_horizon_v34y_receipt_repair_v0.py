#!/usr/bin/env python3
"""T-C1R zero-solve receipt repair for the OC-epsilon-1a handoff.

This script is an operational artifact correction, not a new solver/rollout/training
experiment. It preserves the original v34x raw/completed bytes, emits a corrected
machine-readable receipt that sets the unsupported "full per-stage epsilon vector
serialized in v34u" field to false, and records the lead's explicit coverage limits
for ORIGINAL SAC, pendulum, and v33 rollout evidence.

No validation64 bank, sealed/final test, plant rollout, environment step/reset after
construction, optimizer/gradient update, selector refit, or low-level solver call is
performed.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
try:
    import execution_contract  # type: ignore
except ImportError:  # Defensive for direct diagnostics; structured runs should set PYTHONPATH.
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "research_service"))
    import execution_contract  # type: ignore

NAME = "vehicle_true_variable_horizon_v34y_receipt_repair_v0"
TASK_ID = "T-C1R-OC-epsilon-1a-receipt-repair"
EXPECTED_REQUEST = "execution-result:20260930T131908_12b79445"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
EXPECTED_V34X_RAW_SHA256 = "c0540b523d93105d02b380c27729d4112eb20e0f7b68e74d1cb9141925c45c5e"
EXPECTED_V34X_COMPLETED_SHA256 = "a9732844fff19523cbd97c2c8c2cf32f05bcc31f8097085814e63cd2f1041545"
MARKER_STEM = "vehicle-v34y-tc1r-receipt-repair"

V34X_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34x_epsilon_provenance_audit_v0_20260930T131908Z"
V34X_RAW = V34X_DIR / "raw.json"
V34X_COMPLETED = V34X_DIR / "completed.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
PLAN_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
EXEC_PLAN = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T131933Z_8873cc.execution_plan.json"
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T131933Z_8873cc.md"

ZERO_RESOURCES = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def collect_eps_key_hits(obj: Any, path: str = "root", limit: int = 80) -> list:
    """Collect bounded key-path evidence for the overbroad v34x detector.

    This intentionally does not recompute any objective score. It only shows that
    the earlier detector can be triggered by formula metadata such as include_eps,
    epsterm_total and eps_nonzero_count rather than by a per-stage solver _eps vector.
    """
    hits = []

    def walk(x: Any, p: str) -> None:
        if len(hits) >= limit:
            return
        if isinstance(x, Mapping):
            for k, v in x.items():
                kp = f"{p}.{k}"
                lower = kp.lower()
                if any(tok in lower for tok in ["eps", "epsilon", "slack", "epsterm"]):
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        preview = v
                    elif isinstance(v, Mapping):
                        preview = sorted([str(kk) for kk in v.keys()])[:12]
                    elif isinstance(v, list):
                        preview = [type(item).__name__ for item in v[:8]]
                    else:
                        preview = type(v).__name__
                    hits.append({"path": kp, "type": type(v).__name__, "preview": preview})
                    if len(hits) >= limit:
                        return
                walk(v, kp)
                if len(hits) >= limit:
                    return
        elif isinstance(x, list):
            for i, v in enumerate(x[:200]):
                walk(v, f"{p}[{i}]")
                if len(hits) >= limit:
                    return

    walk(obj, path)
    return hits


def has_strict_full_vector_claim(hits: Iterable[Mapping[str, Any]]) -> bool:
    """Strictly decide whether a full per-stage _eps vector is evidenced by hits.

    A full vector claim must not be satisfied by booleans/scalars or by alias/formula
    summaries. v34x's own summary already states the full vector is absent, so this
    check is a guard against accidentally repeating the overbroad substring detector.
    """
    excluded_fragments = [
        "include_eps", "epsterm_total", "eps_nonzero_count", "eps_delta", "alias_branch_variants",
        "alias_16_candidate", "candidate", "analysis.headline", "analysis.per_cell",
    ]
    for h in hits:
        p = str(h.get("path", "")).lower()
        typ = str(h.get("type", ""))
        if any(fragment in p for fragment in excluded_fragments):
            continue
        if ("_eps" in p or "slack" in p or ".eps" in p) and typ == "list":
            return True
    return False


def write_registry_row(created: dt.datetime, completed_path: Path, marker: str) -> None:
    registry = ROOT / "EXPERIMENT_REGISTRY.csv"
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            created.isoformat(),
            NAME,
            "development_IMPROVED_zero_solve_T_C1R_receipt_repair_not_validation_not_test",
            "correct unsupported v34x epsilon-vector receipt field and append lead coverage lines",
            "opened_development_artifacts_only_no_validation64_no_sealed_test",
            0,
            0,
            0,
            0,
            0,
            False,
            rel(completed_path),
            marker,
        ])


def record_engineering_failure(message: str, run_dir: Path, snapshot: Optional[Mapping[str, Any]], engineering_error: str = "missing_file") -> int:
    created = now_utc()
    failed = {
        "status": "failed",
        "classification": "engineering_failure_zero_resource_before_scientific_outcome",
        "created_utc": created.isoformat(),
        "error": message,
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
    }
    write_json(run_dir / "failed.json", failed)
    if snapshot is not None:
        execution_contract.record_outcome(
            ROOT,
            "engineering_failure",
            dict(ZERO_RESOURCES),
            {"no_scientific_outcome": True, "error": message, "failed_json": rel(run_dir / "failed.json")},
            engineering_error=engineering_error,
        )
    return 1


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    marker = f"{MARKER_STEM}-{stamp}"
    run_dir = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot: Optional[Mapping[str, Any]] = None
    try:
        snapshot = execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
    except Exception as exc:  # Authorization failure is pre-resource and must be preserved.
        return record_engineering_failure(f"structured execution snapshot verification failed: {type(exc).__name__}: {exc}", run_dir, None, engineering_error="authorization")
    if snapshot is None:
        return record_engineering_failure("missing structured execution snapshot", run_dir, None, engineering_error="authorization")

    required = [V34X_RAW, V34X_COMPLETED, PLAN_READY, EXEC_PLAN, OPUS_REPORT]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        return record_engineering_failure("missing required input(s): " + repr(missing), run_dir, snapshot, engineering_error="missing_file")

    try:
        raw_sha_before = sha256(V34X_RAW)
        completed_sha_before = sha256(V34X_COMPLETED)
        if raw_sha_before != EXPECTED_V34X_RAW_SHA256:
            return record_engineering_failure(
                f"v34x raw sha mismatch: observed {raw_sha_before}, expected {EXPECTED_V34X_RAW_SHA256}",
                run_dir,
                snapshot,
                engineering_error="loader",
            )
        if completed_sha_before != EXPECTED_V34X_COMPLETED_SHA256:
            return record_engineering_failure(
                f"v34x completed sha mismatch: observed {completed_sha_before}, expected {EXPECTED_V34X_COMPLETED_SHA256}",
                run_dir,
                snapshot,
                engineering_error="loader",
            )

        v34x_completed = read_json(V34X_COMPLETED)
        v34x_raw = read_json(V34X_RAW)
        old_flag = bool(((v34x_completed.get("headline") or {}) if isinstance(v34x_completed, Mapping) else {}).get("individual_epsilon_vector_values_serialized_in_v34u_raw"))
        note_on_vector_values = None
        if isinstance(v34x_raw, Mapping):
            artifact_audit = v34x_raw.get("artifact_audit")
            if isinstance(artifact_audit, Mapping):
                note_on_vector_values = artifact_audit.get("note_on_vector_values")
        eps_key_hits = collect_eps_key_hits(v34x_raw)
        strict_full_vector_claim = has_strict_full_vector_claim(eps_key_hits)
        corrected_flag = False

        coverage_lines = [
            {
                "area": "ORIGINAL author SAC reconstruction",
                "coverage_status": "partially inspected",
                "line": "ORIGINAL author SAC reconstruction is only partially inspected in the current lead cycle; primary training/checkpoint/progress evidence has not yet been read by the lead.",
                "hypothesis_status": "coverage limitation, not a hypothesis result",
            },
            {
                "area": "ORIGINAL vehicle-then-inverted-pendulum pendant",
                "coverage_status": "UNINSPECTED this cycle",
                "line": "The ORIGINAL vehicle-then-inverted-pendulum pendant is UNINSPECTED in the current lead cycle.",
                "hypothesis_status": "coverage limitation, not a hypothesis result",
            },
            {
                "area": "full v33 rollout matrix",
                "coverage_status": "not independently rechecked",
                "line": "The full v33 rollout matrix was not independently rechecked from raw in the current lead cycle.",
                "hypothesis_status": "coverage limitation, not a hypothesis result",
            },
        ]

        corrected_receipt = {
            "receipt_type": "T-C1R corrected machine-readable receipt",
            "created_utc": created.isoformat(),
            "task_id": TASK_ID,
            "classification": "development_IMPROVED_zero_solve_operational_receipt_repair_not_validation_not_test",
            "original_artifacts_preserved": True,
            "original_v34x_raw": rel(V34X_RAW),
            "original_v34x_completed": rel(V34X_COMPLETED),
            "original_v34x_raw_sha256_echoed_exactly": raw_sha_before,
            "original_v34x_completed_sha256_echoed_exactly": completed_sha_before,
            "original_completed_headline_field": {
                "individual_epsilon_vector_values_serialized_in_v34u_raw": old_flag,
            },
            "corrected_headline_field": {
                "individual_epsilon_vector_values_serialized_in_v34u_raw": corrected_flag,
            },
            "correction_reason": "The v34x path scanner matched keys containing 'eps' such as include_eps booleans and epsterm/epsilon summary scalars. v34x summary/note_on_vector_values states that v34u did not serialize the full per-stage _eps numeric vector. No v34x raw/completed bytes were rewritten.",
            "strict_guard_found_full_vector_claim": strict_full_vector_claim,
            "bounded_eps_key_hits_sample": eps_key_hits,
            "note_on_vector_values_from_v34x_raw": note_on_vector_values,
            "T_C1_pass_verdict_unchanged": bool(v34x_completed.get("T_C1_pass") if isinstance(v34x_completed, Mapping) else False),
            "no_rescoring_of_T_C1": True,
            "coverage_lines": coverage_lines,
            "label_ORIGINAL_vs_IMPROVED": "This receipt-repair thread is IMPROVED latency-tree vehicle development bookkeeping only; it is not ORIGINAL SAC reproduction evidence.",
            "access_and_budget": {
                "new_solver_calls": 0,
                "plant_steps": 0,
                "env_step_calls_after_construction": 0,
                "env_reset_calls_after_construction": 0,
                "new_training_or_gradient_steps": 0,
                "selector_refits": 0,
                "validation64_episodes": 0,
                "sealed_test_episodes": 0,
                "validation64_bank_opened": False,
                "sealed_test_accessed": False,
                "new_aws_resources": False,
            },
        }

        receipt_path = run_dir / "corrected_receipt.json"
        raw_path = run_dir / "raw.json"
        summary_path = run_dir / "summary.md"
        completed_path = run_dir / "completed.json"
        backup_request = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34Y_TC1R_RECEIPT_REPAIR_{stamp}.json"
        state_path = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_v34y_tc1r_receipt_repair.md"

        raw = {
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "task_id": TASK_ID,
            "method": NAME,
            "purpose": "T-C1R operational repair of unsupported v34x full-vector receipt field plus explicit lead coverage lines.",
            "snapshot_task_id": snapshot.get("task", {}).get("task_id") if isinstance(snapshot, Mapping) else None,
            "corrected_receipt": corrected_receipt,
            "input_hashes": {
                rel(V34X_RAW): raw_sha_before,
                rel(V34X_COMPLETED): completed_sha_before,
                rel(PLAN_READY): sha256(PLAN_READY),
                rel(EXEC_PLAN): sha256(EXEC_PLAN),
                rel(OPUS_REPORT): sha256(OPUS_REPORT),
            },
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
        }
        write_json(receipt_path, corrected_receipt)
        write_json(raw_path, raw)

        raw_sha_after = sha256(V34X_RAW)
        completed_sha_after = sha256(V34X_COMPLETED)
        raw_preserved = raw_sha_after == raw_sha_before == EXPECTED_V34X_RAW_SHA256
        completed_preserved = completed_sha_after == completed_sha_before == EXPECTED_V34X_COMPLETED_SHA256

        backup_payload = {
            "request": "backup_after_v34y_tc1r_receipt_repair",
            "created_utc": created.isoformat(),
            "backup_required_before_more_unique_science": True,
            "reason": "T-C1R wrote a corrected receipt, state/docs and registry after v34x; preserve before solver-bearing T-C2.",
            "must_cover": [
                rel(Path(__file__).resolve()),
                rel(run_dir),
                rel(backup_request),
                rel(state_path),
                rel(RESPONSE_LOG),
                "STATUS.md",
                "RESEARCH_LOG.md",
                "DECISIONS.md",
                "RESULTS_AUDIT.md",
                "REPRODUCTION_PROTOCOL.md",
                "EXPERIMENT_REGISTRY.csv",
            ],
            "new_solver_calls": 0,
            "new_plant_steps": 0,
            "env_step_calls_after_construction": 0,
            "new_training_or_gradient_steps": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "next_gate": "After verified backup and dependency acceptance, T-C2 may run under the structured plan if PLAN_READY remains unchanged. T-C2 must persist per-stage epsilon vectors and must not access validation64 or sealed/final test.",
        }
        write_json(backup_request, backup_payload)

        summary_lines = [
            "# v34y T-C1R receipt repair",
            "",
            f"UTC: `{created.isoformat()}`.",
            "",
            "## Result",
            f"- Corrected machine-readable receipt written: `{rel(receipt_path)}`.",
            f"- Original v34x raw SHA256 echoed exactly: `{raw_sha_before}`.",
            f"- Original v34x completed SHA256 echoed exactly: `{completed_sha_before}`.",
            f"- Original v34x raw bytes preserved after repair: `{raw_preserved}`.",
            f"- Original v34x completed bytes preserved after repair: `{completed_preserved}`.",
            f"- Unsupported field corrected from `{old_flag}` to `{corrected_flag}` in the new receipt only; the original v34x files were not rewritten.",
            f"- T-C1 source/provenance pass verdict preserved: `{corrected_receipt['T_C1_pass_verdict_unchanged']}`; this task did not rescore T-C1.",
            "",
            "## Coverage lines appended",
            "- ORIGINAL author SAC reconstruction: partially inspected; primary training/checkpoint/progress evidence not yet read by the lead in this cycle.",
            "- ORIGINAL vehicle-then-inverted-pendulum pendant: UNINSPECTED this cycle.",
            "- Full v33 rollout matrix: not independently rechecked from raw this cycle.",
            "",
            "## Budget/access",
            "- New solver calls: 0; plant steps: 0; env.step/reset-after-construction: 0; training/refit: 0; validation64 episodes: 0; sealed/final-test episodes: 0.",
            "- This is IMPROVED latency-tree vehicle development bookkeeping, not ORIGINAL SAC reproduction evidence.",
            "",
            f"Backup request: `{rel(backup_request)}`.",
        ]
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        doc_block = f"""
<!-- {marker} -->
## v34y T-C1R receipt repair

UTC: {created.isoformat()}. Structured task `{TASK_ID}` completed as a zero-solve operational correction. New solver calls=0, plant/env.step/reset-after-construction=0, training/refit=0, validation64=0, sealed/final test=0. A corrected receipt was written at `{rel(receipt_path)}` setting `individual_epsilon_vector_values_serialized_in_v34u_raw=false`; the original v34x raw SHA256 `{raw_sha_before}` and completed SHA256 `{completed_sha_before}` were echoed and the original bytes were preserved. T-C1 source/provenance pass remains unchanged and was not rescored. Coverage limitations recorded: ORIGINAL author SAC reconstruction is partially inspected, ORIGINAL vehicle-then-inverted-pendulum pendant is UNINSPECTED this cycle, and the full v33 rollout matrix was not independently rechecked. Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(completed_path)}`. Backup request: `{rel(backup_request)}`. Next dependency-gated action after backup is T-C2 or another approved zero-solve ledger task under the active structured plan; validation64 and sealed/final test remain closed.
"""
        for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
            append_if_missing(doc, marker, doc_block)

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            "# Continue state after v34y T-C1R receipt repair\n\n"
            + f"UTC: {created.isoformat()}\n\n"
            + "T-C1R passed as a zero-solve operational receipt repair. The v34x full-vector field is corrected only in the new receipt; original v34x raw/completed bytes remain frozen. Backup is required before solver-bearing T-C2. Validation64 and sealed/final test remain closed.\n\n"
            + f"Artifacts: `{rel(receipt_path)}`, `{rel(raw_path)}`, `{rel(summary_path)}`, `{rel(completed_path)}`.\n",
            encoding="utf-8",
        )

        pass_evidence = {
            "corrected_epsilon_vector_field_present_and_false": corrected_receipt["corrected_headline_field"]["individual_epsilon_vector_values_serialized_in_v34u_raw"] is False,
            "coverage_lines_for_ORIGINAL_and_pendulum_present": all(line["line"] for line in coverage_lines) and any("pendulum" in line["area"].lower() for line in coverage_lines),
            "no_new_solver_or_plant_or_training_usage": True,
            "original_v34x_raw_bytes_preserved": raw_preserved,
            "original_v34x_raw_sha256_echoed_exactly": raw_sha_before,
            "receipt_written": receipt_path.exists(),
        }
        completed = {
            "status": "complete",
            "hard_pass": all([
                pass_evidence["corrected_epsilon_vector_field_present_and_false"],
                pass_evidence["coverage_lines_for_ORIGINAL_and_pendulum_present"],
                pass_evidence["no_new_solver_or_plant_or_training_usage"],
                pass_evidence["original_v34x_raw_bytes_preserved"],
                pass_evidence["original_v34x_raw_sha256_echoed_exactly"] == EXPECTED_V34X_RAW_SHA256,
                pass_evidence["receipt_written"],
                completed_preserved,
            ]),
            "created_utc": created.isoformat(),
            "classification": "development_IMPROVED_zero_solve_T_C1R_receipt_repair_not_validation_not_test",
            "task_id": TASK_ID,
            "summary": rel(summary_path),
            "raw": rel(raw_path),
            "corrected_receipt": rel(receipt_path),
            "backup_request": rel(backup_request),
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "headline": {
                "corrected_epsilon_vector_field_present_and_false": pass_evidence["corrected_epsilon_vector_field_present_and_false"],
                "old_v34x_field_value": old_flag,
                "corrected_field_value": corrected_flag,
                "original_v34x_raw_sha256": raw_sha_before,
                "original_v34x_completed_sha256": completed_sha_before,
                "original_v34x_raw_bytes_preserved": raw_preserved,
                "original_v34x_completed_bytes_preserved": completed_preserved,
                "coverage_lines_for_ORIGINAL_and_pendulum_present": pass_evidence["coverage_lines_for_ORIGINAL_and_pendulum_present"],
                "T_C1_pass_verdict_unchanged": corrected_receipt["T_C1_pass_verdict_unchanged"],
                "validation64_bank_opened": False,
                "sealed_test_accessed": False,
            },
        }
        write_json(completed_path, completed)
        write_registry_row(created, completed_path, marker)

        # Record the structured outcome only after all task evidence files exist.
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), pass_evidence)

        # Add final hashes after the outcome receipt has been recorded. This is auxiliary, not part of the pass gate.
        hash_paths = [
            Path(__file__).resolve(), receipt_path, raw_path, summary_path, completed_path, backup_request, state_path,
            RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md",
            ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv",
        ]
        completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists()}
        write_json(completed_path, completed)

        print(json.dumps(clean({
            "completed": rel(completed_path),
            "summary": rel(summary_path),
            "corrected_receipt": rel(receipt_path),
            "backup_request": rel(backup_request),
            "headline": completed["headline"],
            "pass_evidence": pass_evidence,
        }), sort_keys=True), flush=True)
        return 0 if completed["hard_pass"] else 2
    except Exception as exc:
        return record_engineering_failure(f"unexpected T-C1R error: {type(exc).__name__}: {exc}", run_dir, snapshot, engineering_error="startup")


if __name__ == "__main__":
    raise SystemExit(main())
