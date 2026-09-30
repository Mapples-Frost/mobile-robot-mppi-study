#!/usr/bin/env python3
"""Corrected pass-predicate receipt repair for the learned-policy collapse diagnostic.

The v0b receipt-repair script wrote the right evidence values but used
``all(evidence.values())`` for its local completed.passed flag. That is wrong for
pass conditions whose expected value is False (for example
validation_bank_content_opened=False and sealed_or_final_test_accessed=False), so
it exited with status 2 even though the structured receipt itself was valid and
all safety/resource claims were zero. This wrapper performs a zero-resource
repair by reading only the already-created v0b receipt-repair artifacts, checking
those evidence values against their expected truth values, and writing a clean
structured receipt for the ledger.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
import execution_contract  # type: ignore

TASK_ID = "S-VAL64-learned-collapse-h35-pattern-diagnostic-v0c-pass-predicate-repair"
NAME = "vehicle_learned_policy_collapse_diagnostic_v3_receipt_repair_solo_v0c"
ZERO = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}
SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_learned_policy_collapse_diagnostic_v3_receipt_repair_solo_v0c.py"
V0B_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_v3_receipt_repair_solo_v0_20260930T231002Z"
V0B_RAW = V0B_DIR / "raw.json"
V0B_SUMMARY = V0B_DIR / "summary.md"
V0B_COMPLETED = V0B_DIR / "completed.json"
V0B_RUN_RECEIPT = ROOT / "research_artifacts/aws_runs/20260930T231001_7a710056/outcome_receipt.json"
OUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
DOC_MARKER = "vehicle-learned-collapse-v3-receipt-repair-solo-v0c"

EXPECTED_EVIDENCE = {
    "zero_solver_plant_training_validation_test_resources": True,
    "existing_validation_outputs_read_only": True,
    "validation_bank_content_opened": False,
    "sealed_or_final_test_accessed": False,
    "learned_constant_h25_collapse_quantified": True,
    "learned_s2_h35_usage_and_failure_pattern_quantified": True,
    "case43_failure_context_preserved": True,
    "next_nonzero_experiment_implication_recorded": True,
    "outputs_persisted": True,
    "backup_request_written_or_existing_backup_blocker_recorded": True,
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


def pass_expected(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    mismatches = {}
    for key, expected in EXPECTED_EVIDENCE.items():
        actual = evidence.get(key, "<missing>")
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    return mismatches


def make_summary(raw: Mapping[str, Any]) -> str:
    v0b_raw = raw["v0b_raw"]
    conclusions = v0b_raw["collapse_conclusions"]
    delta = v0b_raw["learned_s2_delta_vs_h25"]
    case43 = v0b_raw["case43_focus"]
    lines = [
        "# Learned-policy collapse receipt pass-predicate repair (solo v0c)",
        "",
        "This zero-resource task repairs only the v0b local pass predicate. The prior v0b artifact already contained the required evidence values, but `all(evidence.values())` incorrectly treated expected-false safety flags as failures.",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        f"Corrected pass predicate: `{raw['corrected_pass']}`.",
        "",
        "## Evidence preserved",
        "",
        f"- learned_s0 classification: `{conclusions['learned_s0']['classification']}`, non-H25 steps `{conclusions['learned_s0']['actual_non25_steps']}`, mismatches `{conclusions['learned_s0']['policy_trace_mismatch_count']}`.",
        f"- learned_s1 classification: `{conclusions['learned_s1']['classification']}`, non-H25 steps `{conclusions['learned_s1']['actual_non25_steps']}`, mismatches `{conclusions['learned_s1']['policy_trace_mismatch_count']}`.",
        f"- learned_s2 classification: `{conclusions['learned_s2']['classification']}`, actual/predicted non-H25 steps `{conclusions['learned_s2']['actual_non25_steps']}/{conclusions['learned_s2']['predicted_non25_steps']}`, mismatches `{conclusions['learned_s2']['policy_trace_mismatch_count']}`.",
        f"- learned_s2 delta versus matched fixed H25: physical mean delta `{delta.get('physical_constraint_cost_mean_episode_delta_learned_minus_comparator')}`, success-rate delta `{delta.get('success_rate_delta_learned_minus_comparator')}`, decision-time ratio `{delta.get('decision_mean_s_per_step_ratio')}`.",
        f"- Case43 learned_s2 remains preserved as a failure: `{case43.get('learned_s2')}`.",
        "",
        "## Safety/resource status",
        "",
        "No validation bank/generator content was opened. No controller, plant, solver, training/refit, validation episode, sealed test or final test was run. Resources are all zero.",
        "",
        "## Next implication",
        "",
        v0b_raw["next_nonzero_experiment_implication"],
    ]
    if raw["evidence_mismatches"]:
        lines += ["", "## Mismatches", "", json.dumps(raw["evidence_mismatches"], indent=2, sort_keys=True)]
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
    backup_request_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_LEARNED_COLLAPSE_RECEIPT_REPAIR_SOLO_V0C_" + stamp + ".json")

    missing = [rel(p) for p in (V0B_RAW, V0B_SUMMARY, V0B_COMPLETED, V0B_RUN_RECEIPT) if not p.exists()]
    if missing:
        failure_raw = {
            "created_utc": created.isoformat(),
            "task_id": TASK_ID,
            "resources": dict(ZERO),
            "missing_inputs": missing,
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
            "no_scientific_outcome": True,
        }
        write_json(raw_path, failure_raw)
        evidence = {
            "no_scientific_outcome": True,
            "missing_inputs": missing,
            "zero_solver_plant_training_validation_test_resources": True,
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
        }
        execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO), evidence, engineering_error="missing_input_artifact")
        print(json.dumps(failure_raw, indent=2, sort_keys=True))
        return 1

    v0b_raw = read_json(V0B_RAW)
    v0b_completed = read_json(V0B_COMPLETED)
    v0b_receipt = read_json(V0B_RUN_RECEIPT)
    prior_evidence = v0b_completed.get("evidence") or v0b_receipt.get("evidence") or {}
    evidence_mismatches = pass_expected(prior_evidence)
    corrected_pass = not evidence_mismatches

    evidence = dict(EXPECTED_EVIDENCE)
    evidence["prior_v0b_completed_passed_false_due_expected_false_flags"] = (v0b_completed.get("passed") is False)
    evidence["corrected_pass_predicate_applied"] = True

    raw = {
        "created_utc": created.isoformat(),
        "task_id": TASK_ID,
        "authority_mode": "temporary_user_authorized_solo_self_review",
        "analysis_type": "zero_resource_pass_predicate_repair_for_prior_receipt_repair",
        "resources": dict(ZERO),
        "script": rel(SCRIPT),
        "script_sha256": sha256(SCRIPT),
        "input_artifacts": {
            rel(V0B_RAW): sha256(V0B_RAW),
            rel(V0B_SUMMARY): sha256(V0B_SUMMARY),
            rel(V0B_COMPLETED): sha256(V0B_COMPLETED),
            rel(V0B_RUN_RECEIPT): sha256(V0B_RUN_RECEIPT),
        },
        "prior_v0b_completed_passed": v0b_completed.get("passed"),
        "prior_v0b_receipt_outcome": v0b_receipt.get("outcome"),
        "prior_evidence": prior_evidence,
        "expected_evidence": EXPECTED_EVIDENCE,
        "evidence_mismatches": evidence_mismatches,
        "corrected_pass": corrected_pass,
        "v0b_raw": v0b_raw,
        "split_safety": {
            "existing_validation_output_artifacts_read": True,
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
        "request": "external_backup_after_learned_collapse_receipt_repair_solo_v0c",
        "reason": "Preserve corrected structured pass predicate for the zero-resource validation-output synthesis before any future nonzero source242/controller-path measurement.",
        "backup_blocker_context": "Supervisor backup remains reported as failed/HTTP-422 or timeout in current context; this request does not claim backup success.",
        "resources": dict(ZERO),
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "artifacts_to_backup": [rel(SCRIPT), rel(raw_path), rel(summary_path), rel(completed_path)],
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
        "backup_request": rel(backup_request_path),
        "backup_request_sha256": sha256(backup_request_path),
        "resources": dict(ZERO),
        "evidence": evidence,
        "evidence_mismatches": evidence_mismatches,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
    }
    write_json(completed_path, completed)

    log_block = """
### Learned-policy collapse receipt pass-predicate repair solo v0c ({created})

- Ran a zero-resource repair for the v0b local `completed.passed` predicate. The v0b evidence values matched the plan, but `all(evidence.values())` incorrectly failed expected-false safety flags (`validation_bank_content_opened=False`, `sealed_or_final_test_accessed=False`).
- Corrected pass predicate: `{passed}`. Evidence mismatches: `{mismatches}`.
- Preserved scientific finding: s0/s1 are structurally H25-only; s2 uses H35 on 409 stored validation steps with no policy/trace mismatches but is worse than matched fixed H25 and preserves the case43 failure. This is IMPROVED development/validation-output evidence only, not ORIGINAL or final-test evidence.
- No validation bank/generator, solver, plant, training/refit, new validation episode, sealed test or final test was accessed. Resources are zero.
- Artifacts: `{raw}`, `{summary}`, `{completed}`.
""".format(
        created=created.isoformat(),
        passed=corrected_pass,
        mismatches=evidence_mismatches,
        raw=rel(raw_path),
        summary=rel(summary_path),
        completed=rel(completed_path),
    )
    append_once(ROOT / "RESEARCH_LOG.md", DOC_MARKER + "-research-log", log_block)
    append_once(ROOT / "STATUS.md", DOC_MARKER + "-status", log_block)
    append_once(ROOT / "DECISIONS.md", DOC_MARKER + "-decisions", log_block)

    if corrected_pass:
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), evidence)
    else:
        failure_evidence = dict(evidence)
        failure_evidence["no_scientific_outcome"] = True
        execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO), failure_evidence, engineering_error="prior_evidence_mismatch")
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
