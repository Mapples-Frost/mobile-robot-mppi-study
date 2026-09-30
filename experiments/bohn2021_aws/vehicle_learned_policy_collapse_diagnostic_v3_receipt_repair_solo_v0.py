#!/usr/bin/env python3
"""Receipt repair for vehicle learned-policy collapse diagnostic v3.

The previous approved S-VAL64 learned-collapse task launched under structured
coordination but the pre-existing v3 diagnostic script returned early because its
2026-09-27 completed.json already existed. That early-return path printed the old
completion marker but did not call execution_contract.record_outcome, so the
scheduler recorded a missing receipt. This wrapper reads only the already-created
v3 collapse diagnostic artifacts and the already-created solo all-shard summary
outputs, writes a current receipt-repair artifact set, and records the structured
outcome receipt. It performs no MPC/control/plant/solver/training/refit work and
opens no validation bank or sealed/final test content.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
import execution_contract  # type: ignore

TASK_ID = "S-VAL64-learned-collapse-h35-pattern-diagnostic-v0b-receipt-repair"
NAME = "vehicle_learned_policy_collapse_diagnostic_v3_receipt_repair_solo_v0"
ZERO = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}
SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_learned_policy_collapse_diagnostic_v3_receipt_repair_solo_v0.py"
PRIOR_RUN = ROOT / "research_artifacts/aws_runs/20260930T230144_d2f598e3/registry.json"
PRIOR_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v3_full_validation64/completed.json"
PRIOR_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v3_full_validation64/raw.json"
PRIOR_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v3_full_validation64/summary.md"
PAIRWISE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_all_shards_pairwise_summary_solo_v0_20260930T225821Z/raw.json"
PAIRWISE_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_all_shards_pairwise_summary_solo_v0_20260930T225821Z/summary.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
OUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
DOC_MARKER = "vehicle-learned-collapse-v3-receipt-repair-solo-v0"


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


def first_record(records: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(records, list):
        for item in records:
            if isinstance(item, Mapping) and "rollout_key" in item:
                return item
    return None


def require_file(path: Path, failures: list) -> None:
    if not path.exists():
        failures.append("missing required input artifact: %s" % rel(path))


def build_summary(raw: Mapping[str, Any]) -> str:
    c = raw["collapse_conclusions"]
    case = raw["case43_focus"]
    lines = [
        "# Vehicle learned-policy collapse/H35-pattern receipt repair (solo v0)",
        "",
        "This is a zero-resource structured-receipt repair for the approved learned-collapse diagnostic. It reads already-created output artifacts only, not the validation bank/generator, and does not run controllers, plant, solvers, training or refits.",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "## Preserved findings from existing artifacts",
        "",
        f"- `learned_s0`: `{c['learned_s0']['classification']}`; non-H25 steps `{c['learned_s0']['actual_non25_steps']}`; policy/trace mismatches `{c['learned_s0']['policy_trace_mismatch_count']}`.",
        f"- `learned_s1`: `{c['learned_s1']['classification']}`; non-H25 steps `{c['learned_s1']['actual_non25_steps']}`; policy/trace mismatches `{c['learned_s1']['policy_trace_mismatch_count']}`.",
        f"- `learned_s2`: `{c['learned_s2']['classification']}`; actual/predicted non-H25 steps `{c['learned_s2']['actual_non25_steps']}/{c['learned_s2']['predicted_non25_steps']}`; unique horizons `{c['learned_s2']['unique_leaf_horizons']}`; policy/trace mismatches `{c['learned_s2']['policy_trace_mismatch_count']}`.",
        f"- Pairwise validation summary preserved: learned_s2 physical mean `{raw['learned_s2_pairwise'].get('physical_constraint_cost_mean_episode')}` versus matched fixed H25 physical mean `{raw['learned_s2_matched_fixed_h25'].get('physical_constraint_cost_mean_episode')}`; learned_s2 decision-time ratio versus H25 `{raw['learned_s2_delta_vs_h25'].get('decision_mean_s_per_step_ratio')}`.",
        f"- Case43 learned_s2 focus: `{case.get('learned_s2')}`.",
        f"- Case43 fixed seed2 H25 focus: `{case.get('fixed_seed2_terminal25_controllerH25')}`.",
        f"- Case43 fixed seed2 H35 focus: `{case.get('fixed_seed2_terminal25_controllerH35')}`.",
        "",
        "## Next nonzero implication",
        "",
        raw["next_nonzero_experiment_implication"],
        "",
        "## Coordination note",
        "",
        "The 20260930T230144_d2f598e3 launch is preserved as an operational failure: the scientific artifact already existed and passed, but the script's idempotent early-exit path did not write the structured receipt. This wrapper records the receipt under a new solo plan without consuming any solver/plant/training/validation/test resources.",
    ]
    return "\n".join(lines) + "\n"


def evidence_from_inputs(completed: Mapping[str, Any], pairwise: Mapping[str, Any], outputs_persisted: bool, backup_written: bool) -> Dict[str, Any]:
    conclusions = completed.get("conclusions") or {}
    s0 = conclusions.get("learned_s0") or {}
    s1 = conclusions.get("learned_s1") or {}
    s2 = conclusions.get("learned_s2") or {}
    comparisons = pairwise.get("learned_vs_matched_fixed") or {}
    s2_comp = comparisons.get("learned_s2") or {}
    case43 = pairwise.get("case43_records", {}).get("focus_records", {}) if isinstance(pairwise.get("case43_records"), Mapping) else {}
    learned_s2_case = first_record(case43.get("learned_s2"))
    fixed25_case = first_record(case43.get("fixed_seed2_terminal25_controllerH25"))
    fixed35_case = first_record(case43.get("fixed_seed2_terminal25_controllerH35"))
    constant_collapse = (
        s0.get("classification") == "extracted_policy_structurally_constant_H25"
        and s1.get("classification") == "extracted_policy_structurally_constant_H25"
        and s0.get("actual_non25_steps") == 0
        and s1.get("actual_non25_steps") == 0
        and s0.get("policy_trace_mismatch_count") == 0
        and s1.get("policy_trace_mismatch_count") == 0
    )
    s2_h35 = (
        s2.get("classification") == "adaptive_horizon_used_and_trace_matches_policy"
        and s2.get("actual_non25_steps") == 409
        and s2.get("predicted_non25_steps") == 409
        and s2.get("policy_trace_mismatch_count") == 0
    )
    case43_preserved = (
        isinstance(learned_s2_case, Mapping)
        and learned_s2_case.get("episode_failure") is True
        and learned_s2_case.get("horizon_counts") == {"25": 149, "35": 1}
        and isinstance(fixed25_case, Mapping)
        and fixed25_case.get("success") is True
        and isinstance(fixed35_case, Mapping)
        and fixed35_case.get("success") is True
    )
    implication_recorded = bool(s2_comp.get("deltas_vs_fixed_H25")) and s2_h35 and constant_collapse
    return {
        "zero_solver_plant_training_validation_test_resources": True,
        "existing_validation_outputs_read_only": True,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "learned_constant_h25_collapse_quantified": bool(constant_collapse),
        "learned_s2_h35_usage_and_failure_pattern_quantified": bool(s2_h35),
        "case43_failure_context_preserved": bool(case43_preserved),
        "next_nonzero_experiment_implication_recorded": bool(implication_recorded),
        "outputs_persisted": bool(outputs_persisted),
        "backup_request_written_or_existing_backup_blocker_recorded": bool(backup_written),
    }


def main() -> int:
    snapshot = execution_contract.runtime_snapshot(ROOT)
    if snapshot is None:
        raise RuntimeError("No structured execution snapshot is available")
    if snapshot.get("task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("Unexpected structured task_id: %r" % (snapshot.get("task", {}).get("task_id"),))

    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / (NAME + "_" + stamp)
    out_dir.mkdir(parents=True, exist_ok=False)
    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    completed_path = out_dir / "completed.json"
    backup_request_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_LEARNED_COLLAPSE_RECEIPT_REPAIR_SOLO_V0_" + stamp + ".json")

    failures = []
    for path in (PRIOR_RUN, PRIOR_COMPLETED, PRIOR_RAW, PRIOR_SUMMARY, PAIRWISE_RAW, PAIRWISE_SUMMARY):
        require_file(path, failures)
    if failures:
        fail_raw = {
            "created_utc": created.isoformat(),
            "task_id": TASK_ID,
            "resources": dict(ZERO),
            "failures": failures,
            "no_validation_bank_content_opened": True,
            "sealed_or_final_test_accessed": False,
            "no_scientific_outcome": True,
        }
        write_json(raw_path, fail_raw)
        evidence = {
            "no_scientific_outcome": True,
            "missing_inputs": failures,
            "zero_solver_plant_training_validation_test_resources": True,
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
        }
        execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO), evidence, engineering_error="missing_file")
        print(json.dumps(fail_raw, indent=2, sort_keys=True))
        return 1

    completed = read_json(PRIOR_COMPLETED)
    prior_raw = read_json(PRIOR_RAW)
    pairwise = read_json(PAIRWISE_RAW)
    conclusions = completed.get("conclusions") or {}
    comparisons = pairwise.get("learned_vs_matched_fixed") or {}
    s2_comp = comparisons.get("learned_s2") or {}
    s2_learned = s2_comp.get("learned") or ((pairwise.get("learned_rollouts") or {}).get("learned_s2") or {})
    s2_fixed25 = s2_comp.get("matched_fixed_H25") or {}
    s2_delta = s2_comp.get("deltas_vs_fixed_H25") or {}
    case43_focus = pairwise.get("case43_records", {}).get("focus_records", {}) if isinstance(pairwise.get("case43_records"), Mapping) else {}

    next_implication = (
        "Do not allocate more nonzero validation rollouts to the current frozen latency-tree selector as a candidate success path: "
        "learned_s0 and learned_s1 are structurally H25-only, and learned_s2's only adaptive behavior is a H25/H35 tree that is slightly slower on average and dominated by the preserved case43 failure. "
        "Once external backup recoverability is restored, the next nonzero work should continue the already prepared source242 true-variable-horizon microcontinuation/controller-path measurement (or a direct successor repair if that task still fails pre-solve) rather than another unchanged latency-tree validation sweep. "
        "If recoverability remains blocked, only zero-resource preparation/receipt repair should continue; no sealed/final test is authorized."
    )

    raw = {
        "created_utc": created.isoformat(),
        "task_id": TASK_ID,
        "authority_mode": "temporary_user_authorized_solo_self_review",
        "analysis_type": "structured_receipt_repair_over_existing_validation_output_artifacts_only",
        "resources": dict(ZERO),
        "script": rel(SCRIPT),
        "script_sha256": sha256(SCRIPT),
        "prior_failed_or_unverified_structured_attempt": {
            "experiment_id": "20260930T230144_d2f598e3",
            "registry": rel(PRIOR_RUN),
            "failure_mode": "script returned from pre-existing completed.json and did not write BOHN_OUTCOME_RECEIPT",
            "resources_consumed": dict(ZERO),
        },
        "input_artifacts": {
            rel(PRIOR_COMPLETED): sha256(PRIOR_COMPLETED),
            rel(PRIOR_RAW): sha256(PRIOR_RAW),
            rel(PRIOR_SUMMARY): sha256(PRIOR_SUMMARY),
            rel(PAIRWISE_RAW): sha256(PAIRWISE_RAW),
            rel(PAIRWISE_SUMMARY): sha256(PAIRWISE_SUMMARY),
        },
        "prior_collapse_raw_created_utc": prior_raw.get("created_utc"),
        "prior_collapse_passed": completed.get("passed"),
        "collapse_conclusions": conclusions,
        "learned_s2_pairwise": s2_learned,
        "learned_s2_matched_fixed_h25": s2_fixed25,
        "learned_s2_delta_vs_h25": s2_delta,
        "case43_focus": {
            "learned_s2": first_record(case43_focus.get("learned_s2")),
            "learned_s0": first_record(case43_focus.get("learned_s0")),
            "learned_s1": first_record(case43_focus.get("learned_s1")),
            "fixed_seed2_terminal25_controllerH25": first_record(case43_focus.get("fixed_seed2_terminal25_controllerH25")),
            "fixed_seed2_terminal25_controllerH35": first_record(case43_focus.get("fixed_seed2_terminal25_controllerH35")),
        },
        "split_safety": {
            "existing_validation_output_artifacts_read": True,
            "validation_bank_content_opened": False,
            "new_validation_episodes": 0,
            "sealed_or_final_test_accessed": False,
            "controller_or_solver_executed": False,
            "training_or_refit_executed": False,
        },
        "next_nonzero_experiment_implication": next_implication,
        "failures": [],
    }
    write_json(raw_path, raw)
    write_text(summary_path, build_summary(raw))
    backup_request = {
        "created_utc": created.isoformat(),
        "request": "external_backup_after_learned_collapse_receipt_repair_solo_v0",
        "reason": "Preserve structured receipt repair and negative validation-output synthesis before any future nonzero source242/controller-path measurement.",
        "backup_blocker_context": "Supervisor backup currently reports HTTP 422/failed status; this request does not claim backup success.",
        "resources": dict(ZERO),
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "artifacts_to_backup": [rel(SCRIPT), rel(raw_path), rel(summary_path), rel(completed_path)],
    }
    write_json(backup_request_path, backup_request)

    evidence = evidence_from_inputs(completed, pairwise, outputs_persisted=False, backup_written=False)
    completed_marker = {
        "created_utc": created.isoformat(),
        "task_id": TASK_ID,
        "passed": all(evidence.values()),
        "raw": rel(raw_path),
        "raw_sha256": sha256(raw_path),
        "summary": rel(summary_path),
        "summary_sha256": sha256(summary_path),
        "backup_request": rel(backup_request_path),
        "backup_request_sha256": sha256(backup_request_path),
        "resources": dict(ZERO),
        "evidence": None,
        "sealed_or_final_test_accessed": False,
        "validation_bank_content_opened": False,
    }
    write_json(completed_path, completed_marker)
    evidence = evidence_from_inputs(completed, pairwise, outputs_persisted=True, backup_written=True)
    completed_marker["passed"] = all(evidence.values())
    completed_marker["evidence"] = evidence
    write_json(completed_path, completed_marker)

    log_block = """
### Learned-policy collapse/H35-pattern receipt repair solo v0 ({created})

- Zero-resource structured receipt repair for the approved collapse diagnostic after run `20260930T230144_d2f598e3` exited from pre-existing completed artifacts without writing an outcome receipt.
- Inputs read: `{prior_completed}`, `{prior_raw}`, `{pairwise_raw}`. Validation bank/generator content was not opened; no solver/plant/training/refit/new validation episodes/sealed test.
- Preserved conclusions: s0/s1 are structurally H25-only with zero policy/trace mismatches; s2 uses H35 on 409/5014 stored steps with zero policy/trace mismatches, but all-shard validation shows physical mean 628.572 vs matched fixed H25 18.918 and case43 failure with horizon counts {{25:149, 35:1}} while fixed H25/H35 comparators succeed.
- Decision: current latency-tree selector is negative development evidence; next nonzero work, after backup recoverability, should continue source242 true-variable-horizon microcontinuation/controller-path measurement or its direct repair rather than another unchanged latency-tree validation sweep.
- Artifacts: `{raw}`, `{summary}`, `{completed}`.
""".format(
        created=created.isoformat(),
        prior_completed=rel(PRIOR_COMPLETED),
        prior_raw=rel(PRIOR_RAW),
        pairwise_raw=rel(PAIRWISE_RAW),
        raw=rel(raw_path),
        summary=rel(summary_path),
        completed=rel(completed_path),
    )
    append_once(ROOT / "RESEARCH_LOG.md", DOC_MARKER + "-research-log", log_block)
    append_once(ROOT / "STATUS.md", DOC_MARKER + "-status", log_block)
    append_once(ROOT / "DECISIONS.md", DOC_MARKER + "-decisions", log_block)

    execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), evidence)
    print(json.dumps(completed_marker, indent=2, sort_keys=True))
    return 0 if completed_marker["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        # Best-effort structured receipt for unexpected zero-resource startup/shape errors.
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
