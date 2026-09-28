#!/usr/bin/env python3
"""Schema-repaired v1b matched-continuation prepare/dry-run.

This wrapper repairs the first v1b prepare failure without changing the
scientific design.  The failed source expected `runner_material_cases` at the
postdiagnostic raw top level; the actual saved postdiagnostic records the same
predeclared material cases under `predeclared_stage1_gate.material_case_ids_from_runner`
and in `strict_material_vs_H15_rows`.  No simulations, training, validation64 or
sealed-test access are performed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_stress_v1b_matched_continuation_prepare as base  # noqa:E402

STAMP = "20260928T2210Z"
METHOD = "vehicle_stress_v1b_matched_continuation_prepare_v2_schema_repair"

# Version the outputs; leave the failed v1 source/run registry as negative evidence.
base.STAMP = STAMP
base.METHOD = METHOD
base.OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_prepare_{STAMP}_schema_repair"
base.PROTOCOL_JSON = ROOT / f"research_artifacts/aws_protocols/vehicle_stress_v1b_matched_continuation_frozen_{STAMP}_schema_repair.json"
base.PROTOCOL_MD = ROOT / f"research_artifacts/aws_protocols/vehicle_stress_v1b_matched_continuation_frozen_{STAMP}_schema_repair.md"
base.STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1b_matched_continuation_prepare_{STAMP}_schema_repair.md"
base.REQUEST_BACKUP = base.BACKUP_DIR / f"REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1B_MATCHED_CONTINUATION_ROLLOUT_{STAMP}_SCHEMA_REPAIR.json"
base.MARKER = f"vehicle-stress-v1b-matched-continuation-prepare-{STAMP}-schema-repair"


def _extract_material_cases(post_raw: Dict[str, Any]) -> list:
    """Return the predeclared Stage1 material cases across observed schemas."""
    candidates = []
    for key in ("runner_material_cases", "material_non_H15_positive_cases"):
        val = post_raw.get(key)
        if val:
            candidates.append(val)
    nested = post_raw.get("predeclared_stage1_gate") or {}
    for key in ("material_case_ids_from_runner", "material_non_H15_positive_cases"):
        val = nested.get(key)
        if val:
            candidates.append(val)
    if not candidates:
        rows = post_raw.get("strict_material_vs_H15_rows") or []
        if rows:
            candidates.append(sorted({int(r["case"]) for r in rows if "case" in r}))
    if not candidates:
        return []
    return sorted({int(x) for xs in candidates for x in xs})


def repaired_verify_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for p in (base.STAGE1_RAW, base.STAGE1_COMPLETED, base.STAGE1_SUMMARY, base.STAGE1_BANK, base.POST_RAW, base.POST_COMPLETED, base.POST_SUMMARY, base.STAGE1_PROTOCOL_JSON, base.STAGE1_PROTOCOL_MD):
        if not p.exists():
            raise base.ContractError("required input missing: %s" % base.rel(p))
    stage1_done = base.completed_passed(base.STAGE1_COMPLETED)
    post_done = base.completed_passed(base.POST_COMPLETED)
    if stage1_done.get("historical_validation64_bank_opened") is not False:
        raise base.ContractError("Stage1 unexpectedly opened historical validation64 bank")
    if post_done.get("historical_validation64_bank_opened") is not False:
        raise base.ContractError("postdiagnostic unexpectedly opened historical validation64 bank")
    stage1_raw = base.read_json(base.STAGE1_RAW)
    post_raw = base.read_json(base.POST_RAW)
    bank = base.read_json(base.STAGE1_BANK)
    protocol = base.read_json(base.STAGE1_PROTOCOL_JSON)
    if int(stage1_done.get("episodes", -1)) != 200 or int(stage1_done.get("control_steps", -1)) <= 0:
        raise base.ContractError("Stage1 budget markers unexpected")
    if post_done.get("predeclared_stage1_gate_failed") is not True:
        raise base.ContractError("postdiagnostic did not preserve the failed Stage1 gate")
    selected_meta = (bank.get("selection") or {}).get("selected_metadata") or []
    if len(selected_meta) < max(base.TARGET_CASES) + 1:
        raise base.ContractError("bank selected_metadata too short for v1b target cases")
    material_cases = _extract_material_cases(post_raw)
    if material_cases != base.TARGET_CASES_POSITIVE:
        raise base.ContractError("unexpected material cases in postdiagnostic after schema repair: %s" % material_cases)
    nested_gate = post_raw.get("predeclared_stage1_gate") or {}
    trigger = post_raw.get("predeclared_stage2_trigger_candidate", nested_gate.get("stage2_continuation_trigger_candidate_development_only", False))
    if bool(trigger) is not False:
        raise base.ContractError("predeclared Stage2 gate unexpectedly passed; v1b should remain postdiagnostic")
    # Preserve a normalized copy for downstream raw snapshots without editing inputs.
    post_raw = dict(post_raw)
    post_raw["runner_material_cases"] = material_cases
    post_raw["predeclared_stage2_trigger_candidate"] = False
    return stage1_raw, stage1_done, post_raw, post_done, bank, protocol


base.verify_inputs = repaired_verify_inputs

if __name__ == "__main__":
    raise SystemExit(base.main())
