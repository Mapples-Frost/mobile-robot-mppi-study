#!/usr/bin/env python3
"""Structured receipt repair for vehicle_current_gated_horizon_training_audit_v1.

The original zero-resource training/search provenance audit completed and wrote
raw/summary/completed artifacts, but it was not instrumented with
execution_contract.record_outcome, so the scheduler could not mark the approved
structured task as accepted.  This repair reads only those existing artifacts and
run logs, verifies the intended pass-condition evidence, writes a compact repair
artifact set, and emits a structured outcome receipt.  It does not open a
validation bank/generator, run controllers, run solvers, step the plant, train,
refit, or access any sealed/final test content.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
import execution_contract  # type: ignore

TASK_ID = "S-TRAINING-CURRENT-GATED-HORIZON-AUDIT-v0b-receipt-repair"
NAME = "vehicle_current_gated_horizon_training_audit_v1_receipt_repair_solo_v0"
ZERO = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}
SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_current_gated_horizon_training_audit_v1_receipt_repair_solo_v0.py"
PRIOR_DIAG = ROOT / "research_artifacts/aws_diagnostics/vehicle_current_gated_horizon_training_audit_v1_20261001T001446+0000"
PRIOR_RAW = PRIOR_DIAG / "raw.json"
PRIOR_SUMMARY = PRIOR_DIAG / "summary.md"
PRIOR_COMPLETED = PRIOR_DIAG / "completed.json"
PRIOR_CANDIDATE_TABLE = PRIOR_DIAG / "candidate_table.csv"
PRIOR_BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_CURRENT_GATED_HORIZON_TRAINING_AUDIT_V1_20261001T001446+0000.json"
PRIOR_RUN = ROOT / "research_artifacts/aws_runs/20261001T001445_3d582b25"
PRIOR_REGISTRY = PRIOR_RUN / "registry.json"
PRIOR_STDOUT = PRIOR_RUN / "stdout.log"
PRIOR_STDERR = PRIOR_RUN / "stderr.log"
PRIOR_RECEIPT = PRIOR_RUN / "outcome_receipt.json"
OUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
DOC_MARKER = "vehicle-current-gated-horizon-training-audit-v1-receipt-repair-solo-v0"

EXPECTED_EVIDENCE = {
    "zero_solver_plant_training_validation_test_resources": True,
    "validation_bank_content_opened": False,
    "sealed_or_final_test_accessed": False,
    "prior_audit_artifacts_inspected": True,
    "prior_completed_passed_true": True,
    "prior_run_missing_outcome_receipt_confirmed": True,
    "current_gated_policy_artifacts_inspected_or_missing_recorded": True,
    "seed_level_horizon_collapse_or_adaptivity_recorded": True,
    "training_selection_provenance_recorded": True,
    "missing_training_or_selection_evidence_recorded": True,
    "outputs_persisted": True,
    "backup_blocker_preserved": True,
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
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    token = "<!-- %s -->" % marker
    if token in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + block.strip() + "\n", encoding="utf-8")


def compact_seed_record(raw: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for seed, rec in sorted((raw.get("seed_summaries") or {}).items(), key=lambda kv: int(kv[0])):
        selected = rec.get("stored_current_policy") or {}
        train_cov = rec.get("training_trace_coverage_selected_policy") or {}
        dev_cov = rec.get("already_opened_development_trace_coverage_v2_shard00") or {}
        budget = rec.get("budget_from_fit_summaries") or {}
        counts = rec.get("candidate_counts") or {}
        chosen = rec.get("chosen_by_recomputed_training_rule") or {}
        out[str(seed)] = {
            "stored_current_policy": selected,
            "selected_matches_fit_completed": bool(rec.get("selected_matches_stored_policy")),
            "selection_rule_recompute_matches_fit": bool(rec.get("selection_rule_recompute_matches_fit")),
            "candidate_counts": counts,
            "finite_search_budget": budget,
            "chosen_by_recomputed_training_rule": chosen,
            "training_selected_horizon_counts": train_cov.get("horizon_counts"),
            "training_selected_short_step_fraction": train_cov.get("short_step_fraction"),
            "already_opened_v2_shard00_horizon_counts": dev_cov.get("horizon_counts"),
            "already_opened_v2_shard00_raw_horizon_counts": dev_cov.get("raw_horizon_counts"),
            "already_opened_v2_shard00_short_step_fraction": dev_cov.get("short_step_fraction"),
            "rejection_reason_counts": rec.get("rejection_reason_counts"),
        }
    return out


def build_evidence(prior_completed: Mapping[str, Any], raw: Mapping[str, Any]) -> Dict[str, Any]:
    flags = raw.get("access_flags") or {}
    seeds = raw.get("seed_summaries") or {}
    evidence = dict(EXPECTED_EVIDENCE)
    evidence["zero_solver_plant_training_validation_test_resources"] = (
        flags.get("new_control_steps") == 0
        and flags.get("new_gradient_steps") == 0
        and flags.get("new_rollout_episodes") == 0
        and flags.get("new_training_episodes") == 0
    )
    evidence["validation_bank_content_opened"] = bool(flags.get("historical_validation64_bank_opened"))
    evidence["sealed_or_final_test_accessed"] = bool(flags.get("sealed_test_accessed") or flags.get("final_test_authorization_requested"))
    evidence["prior_audit_artifacts_inspected"] = all(p.exists() for p in (PRIOR_RAW, PRIOR_SUMMARY, PRIOR_COMPLETED, PRIOR_CANDIDATE_TABLE, PRIOR_REGISTRY, PRIOR_STDOUT, PRIOR_STDERR))
    evidence["prior_completed_passed_true"] = prior_completed.get("passed") is True
    evidence["prior_run_missing_outcome_receipt_confirmed"] = not PRIOR_RECEIPT.exists()
    evidence["current_gated_policy_artifacts_inspected_or_missing_recorded"] = set(seeds.keys()) == {"0", "1", "2"}
    evidence["seed_level_horizon_collapse_or_adaptivity_recorded"] = all(
        isinstance((seeds.get(str(seed)) or {}).get("training_trace_coverage_selected_policy"), Mapping)
        and isinstance((seeds.get(str(seed)) or {}).get("already_opened_development_trace_coverage_v2_shard00"), Mapping)
        for seed in (0, 1, 2)
    )
    evidence["training_selection_provenance_recorded"] = all(
        (seeds.get(str(seed)) or {}).get("selected_matches_stored_policy") is True
        and (seeds.get(str(seed)) or {}).get("selection_rule_recompute_matches_fit") is True
        and ((seeds.get(str(seed)) or {}).get("budget_from_fit_summaries") or {}).get("new_gradient_steps") == 0
        for seed in (0, 1, 2)
    )
    evidence["missing_training_or_selection_evidence_recorded"] = bool(raw.get("missing_evidence")) and bool(raw.get("hypotheses"))
    evidence["outputs_persisted"] = all(p.exists() for p in (PRIOR_RAW, PRIOR_SUMMARY, PRIOR_COMPLETED, PRIOR_CANDIDATE_TABLE))
    evidence["backup_blocker_preserved"] = PRIOR_BACKUP_REQUEST.exists()
    return evidence


def mismatches(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    bad: Dict[str, Any] = {}
    for key, expected in EXPECTED_EVIDENCE.items():
        actual = evidence.get(key, "<missing>")
        if actual != expected or type(actual) is not type(expected):
            bad[key] = {"expected": expected, "actual": actual}
    return bad


def make_summary(raw: Mapping[str, Any]) -> str:
    compact = raw["compact_seed_record"]
    lines = [
        "# Current gated-horizon training audit receipt repair (solo v0)",
        "",
        "This zero-resource repair verifies the previously completed training/search provenance audit and emits the structured outcome receipt that the original audit omitted. It reads only existing audit artifacts and run logs.",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        f"Corrected pass predicate: `{raw['corrected_pass']}`.",
        "",
        "## Why the repair was needed",
        "",
        "The prior audit completed with `completed.passed=true`, wrote raw/summary/candidate-table artifacts, and exited status 0, but did not create `outcome_receipt.json`. The structured scheduler therefore marked scientific acceptance as unverified. This repair does not change the scientific data; it only supplies a receipt after re-checking the evidence.",
        "",
        "## Preserved compact findings",
        "",
    ]
    for seed in ("0", "1", "2"):
        rec = compact.get(seed, {})
        lines.append(
            "- seed {seed}: policy `{policy}`, train H counts `{train_counts}`, train short fraction `{train_short}`, already-opened v2 shard00 H counts `{dev_counts}`, raw H counts `{raw_counts}`, shard00 short fraction `{dev_short}`, gradient steps `{grad_steps}`.".format(
                seed=seed,
                policy=rec.get("stored_current_policy", {}).get("id"),
                train_counts=rec.get("training_selected_horizon_counts"),
                train_short=rec.get("training_selected_short_step_fraction"),
                dev_counts=rec.get("already_opened_v2_shard00_horizon_counts"),
                raw_counts=rec.get("already_opened_v2_shard00_raw_horizon_counts"),
                dev_short=rec.get("already_opened_v2_shard00_short_step_fraction"),
                grad_steps=(rec.get("finite_search_budget") or {}).get("new_gradient_steps"),
            )
        )
    lines.extend([
        "",
        "## Safety/resource status",
        "",
        "No validation bank/generator content was opened by this repair. No controller, plant, solver, training/refit, new validation episode, sealed test or final test was run. All structured resource counters are zero.",
        "",
        "## Evidence mismatches",
        "",
        json.dumps(raw["evidence_mismatches"], indent=2, sort_keys=True),
        "",
        "## Next implication",
        "",
        "The current reused gated-horizon selector remains a negative development path unless a substantive training/selection change is made. Nonzero controller-path measurements remain gated by external-backup recoverability; while backup is blocked, only zero-resource preparation/repair should proceed.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    snapshot = execution_contract.runtime_snapshot(ROOT)
    if snapshot is None:
        raise RuntimeError("No structured execution snapshot is available")
    observed_task = snapshot.get("task", {}).get("task_id")
    if observed_task != TASK_ID:
        raise RuntimeError("Unexpected structured task_id: %r" % (observed_task,))

    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / (NAME + "_" + stamp)
    out_dir.mkdir(parents=True, exist_ok=False)
    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    completed_path = out_dir / "completed.json"
    backup_request_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_CURRENT_GATED_AUDIT_RECEIPT_REPAIR_SOLO_V0_" + stamp + ".json")

    required_inputs = (PRIOR_RAW, PRIOR_SUMMARY, PRIOR_COMPLETED, PRIOR_CANDIDATE_TABLE, PRIOR_REGISTRY, PRIOR_STDOUT, PRIOR_STDERR)
    missing_inputs = [rel(p) for p in required_inputs if not p.exists()]
    if missing_inputs:
        failure = {
            "created_utc": created.isoformat(),
            "task_id": TASK_ID,
            "resources": dict(ZERO),
            "missing_inputs": missing_inputs,
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
            "no_scientific_outcome": True,
        }
        write_json(raw_path, failure)
        evidence = {
            "no_scientific_outcome": True,
            "zero_solver_plant_training_validation_test_resources": True,
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
            "missing_inputs": missing_inputs,
        }
        execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO), evidence, engineering_error="missing_file")
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1

    prior_raw = read_json(PRIOR_RAW)
    prior_completed = read_json(PRIOR_COMPLETED)
    registry = read_json(PRIOR_REGISTRY)
    stdout_tail = read_text(PRIOR_STDOUT)[-4000:]
    stderr_tail = read_text(PRIOR_STDERR)[-4000:]
    evidence = build_evidence(prior_completed, prior_raw)
    bad = mismatches(evidence)
    corrected_pass = not bad

    raw = {
        "created_utc": created.isoformat(),
        "task_id": TASK_ID,
        "authority_mode": "temporary_user_authorized_solo_self_review",
        "analysis_type": "zero_resource_structured_receipt_repair_for_prior_training_search_audit",
        "resources": dict(ZERO),
        "script": rel(SCRIPT),
        "script_sha256": sha256(SCRIPT),
        "input_artifacts": {
            rel(PRIOR_RAW): sha256(PRIOR_RAW),
            rel(PRIOR_SUMMARY): sha256(PRIOR_SUMMARY),
            rel(PRIOR_COMPLETED): sha256(PRIOR_COMPLETED),
            rel(PRIOR_CANDIDATE_TABLE): sha256(PRIOR_CANDIDATE_TABLE),
            rel(PRIOR_REGISTRY): sha256(PRIOR_REGISTRY),
            rel(PRIOR_STDOUT): sha256(PRIOR_STDOUT),
            rel(PRIOR_STDERR): sha256(PRIOR_STDERR),
        },
        "prior_completed_passed": prior_completed.get("passed"),
        "prior_run_exit_status": registry.get("exit_status"),
        "prior_run_runtime_seconds": registry.get("runtime_seconds"),
        "prior_outcome_receipt_exists": PRIOR_RECEIPT.exists(),
        "prior_stdout_tail": stdout_tail,
        "prior_stderr_tail": stderr_tail,
        "prior_access_flags": prior_raw.get("access_flags"),
        "compact_seed_record": compact_seed_record(prior_raw),
        "verified_causes": prior_raw.get("verified_causes"),
        "hypotheses": prior_raw.get("hypotheses"),
        "missing_evidence": prior_raw.get("missing_evidence"),
        "evidence": evidence,
        "evidence_mismatches": bad,
        "corrected_pass": corrected_pass,
        "split_safety": {
            "existing_audit_artifacts_read": True,
            "validation_bank_content_opened": False,
            "new_validation_episodes": 0,
            "sealed_or_final_test_accessed": False,
            "controller_or_solver_executed": False,
            "training_or_refit_executed": False,
        },
    }
    write_json(raw_path, raw)
    write_text(summary_path, make_summary(raw))

    backup_request = {
        "created_utc": created.isoformat(),
        "request": "external_backup_after_current_gated_horizon_training_audit_receipt_repair_solo_v0",
        "reason": "Preserve structured receipt repair for the current gated-horizon training/search audit before any future nonzero controller-path work.",
        "backup_blocker_context": "Supervisor context still reports external backup failure/timeout and earlier GitHub asset-cap HTTP 422; this request records desired backup but does not claim recoverability.",
        "resources": dict(ZERO),
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "artifacts_to_backup": [rel(SCRIPT), rel(raw_path), rel(summary_path), rel(completed_path), rel(PRIOR_RAW), rel(PRIOR_SUMMARY), rel(PRIOR_COMPLETED), rel(PRIOR_CANDIDATE_TABLE)],
    }
    write_json(backup_request_path, backup_request)

    completed = {
        "created_utc": created.isoformat(),
        "task_id": TASK_ID,
        "passed": corrected_pass,
        "raw": rel(raw_path),
        "raw_sha256": sha256(raw_path),
        "summary": rel(summary_path),
        "summary_sha256": sha256(summary_path),
        "completed": rel(completed_path),
        "backup_request": rel(backup_request_path),
        "backup_request_sha256": sha256(backup_request_path),
        "resources": dict(ZERO),
        "evidence": evidence,
        "evidence_mismatches": bad,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
    }
    write_json(completed_path, completed)
    completed["completed_sha256"] = sha256(completed_path)
    write_json(completed_path, completed)

    log_block = """
### Current gated-horizon training audit receipt repair solo v0 ({created})

- Repaired the structured receipt omission for `S-TRAINING-CURRENT-GATED-HORIZON-AUDIT-v0`. The prior audit exited status 0 and wrote `completed.passed=true`, but no `outcome_receipt.json`, so scheduler acceptance was unverified.
- Corrected pass predicate: `{passed}`. Evidence mismatches: `{mismatches}`.
- Preserved audit findings: current vehicle policies are finite-search IMPROVED gated selectors, not ORIGINAL SAC and not the older latency tree; seed0 uses `h20_p1_g5`, seed1 uses `h15_p2_g5`, seed2 uses `h10_p0_g5`; all three have `new_gradient_steps=0`; selection used mean raw cost with physical-cost gates and no measured wall-clock objective.
- No validation bank/generator, controller, plant, solver, training/refit, new validation episode, sealed test or final test was accessed. Resources are zero.
- Artifacts: `{raw}`, `{summary}`, `{completed}`.
""".format(
        created=created.isoformat(),
        passed=corrected_pass,
        mismatches=bad,
        raw=rel(raw_path),
        summary=rel(summary_path),
        completed=rel(completed_path),
    )
    append_once(ROOT / "RESEARCH_LOG.md", DOC_MARKER + "-research-log", log_block)
    append_once(ROOT / "STATUS.md", DOC_MARKER + "-status", log_block)
    append_once(ROOT / "DECISIONS.md", DOC_MARKER + "-decisions", log_block)
    append_once(ROOT / "RESULTS_AUDIT.md", DOC_MARKER + "-results-audit", log_block)

    if corrected_pass:
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), evidence)
    else:
        failure_evidence = dict(evidence)
        failure_evidence["no_scientific_outcome"] = True
        execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO), failure_evidence, engineering_error="shape")
    print(json.dumps(completed, indent=2, sort_keys=True))
    return 0 if corrected_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        try:
            evidence = {
                "no_scientific_outcome": True,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "traceback_tail": traceback.format_exc()[-4000:],
                "zero_solver_plant_training_validation_test_resources": True,
                "validation_bank_content_opened": False,
                "sealed_or_final_test_accessed": False,
            }
            execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO), evidence, engineering_error="startup")
        except Exception:
            pass
        raise
