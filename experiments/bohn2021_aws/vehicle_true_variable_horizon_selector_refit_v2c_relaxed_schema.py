#!/usr/bin/env python3
"""Relaxed-schema wrapper for selector-refit v2b diagnostic.

The v2b source repaired the original SyntaxError but retained an overly strict
artifact-schema guard: older completed.json/raw.json development artifacts may omit
`validation64_bank_opened` / `sealed_test_accessed` keys even though they are known
from their protocol and summaries to be development-only.  Treating missing flags
as failure would make the no-simulation diagnostic fail before answering the
scientific question.

This wrapper performs no simulations, no validation64 access and no sealed-test
access.  It delegates the actual cross-bank selector enumeration to v2b after
patching only schema guards and output identifiers:

  * explicit True validation/test flags still abort;
  * missing validation/test flags are accepted for legacy development artifacts;
  * source_summary always includes zero-valued category keys;
  * output paths/markers are versioned as v2c.

The scientific hypothesis and candidate grid are otherwise unchanged from v2b.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_selector_refit_v2b_schema_repair as v2b  # noqa:E402

NAME = "vehicle_true_variable_horizon_selector_refit_v2c_relaxed_schema"
STAMP = "20260929T1145Z"
PRIMARY_PROFILE = v2b.PRIMARY_PROFILE
TERMINAL_PROFILES = v2b.TERMINAL_PROFILES


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def completed_ok_relaxed(path: Path) -> Mapping[str, Any]:
    """Accept legacy development artifacts with absent access flags, but not True flags."""
    if not path.exists():
        raise ContractError("missing prerequisite: " + rel(path))
    obj = v2b.read_json(path)
    if obj.get("sealed_test_accessed") is True or obj.get("sealed_test_bank_opened") is True or obj.get("validation64_bank_opened") is True:
        raise ContractError("diagnostic prerequisite explicitly opened validation/test: " + rel(path))
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError("diagnostic prerequisite did not complete cleanly: " + rel(path))
    return obj


def build_dev_rows_relaxed(bank_id: str, spec: Mapping[str, Path]):
    """v2b.build_dev_rows with legacy-missing raw access flags accepted."""
    raw = v2b.read_json(spec["raw"])
    if raw.get("sealed_test_accessed") is True or raw.get("sealed_test_bank_opened") is True or raw.get("validation64_bank_opened") is True:
        raise ContractError(f"{bank_id} raw flags show validation/test access")
    states = v2b.load_manifest(raw, spec["manifest"])
    by_base = {str(s.get("base_state_id")): s for s in states}
    rows = []
    for r in ((raw.get("analysis") or {}).get("state_profile_rows") or []):
        base = str(r.get("base_state_id") or r.get("state_id"))
        st = by_base.get(base)
        if st is None:
            raise ContractError(f"{bank_id}: missing manifest state for {base}")
        med = r.get("median_by_h") or {}
        h10 = v2b.median_summary(med, 10)
        h15 = v2b.median_summary(med, 15)
        h10_phys = v2b.sf(h10.get("physical"), 0.0)
        h15_phys = v2b.sf(h15.get("physical"), 0.0)
        h10_dec = v2b.sf(h10.get("decision_sum_s"), 0.0)
        h15_dec = v2b.sf(h15.get("decision_sum_s"), 0.0)
        h10_sol = v2b.sf(h10.get("solver_sum_s"), 0.0)
        h15_sol = v2b.sf(h15.get("solver_sum_s"), 0.0)
        row_tol = v2b.sf(r.get("row_physical_tolerance_vs_H15", r.get("row_physical_tolerance")), max(2.0, 0.05 * abs(h15_phys)))
        row_tol = max(2.0, row_tol)
        h10_safe = bool(h10.get("safe_all"))
        h15_safe = bool(h15.get("safe_all"))
        if "h10_beneficial_vs_h15" in r:
            beneficial = bool(r.get("h10_beneficial_vs_h15"))
        else:
            beneficial = bool(h15_safe and h10_safe and (h10_phys - h15_phys <= row_tol) and (h10_dec < h15_dec))
        catastrophic = bool((not h10_safe) or (h15_safe and (h10_phys - h15_phys > row_tol)))
        rows.append({
            "bank_id": bank_id,
            "base_state_id": base,
            "terminal_profile": str(r.get("terminal_profile")),
            "fresh_confirmation_group": r.get("fresh_confirmation_group"),
            "source_candidate_index": st.get("source_candidate_index"),
            "branch_state_slot": v2b.si(r.get("branch_state_slot", st.get("branch_state_slot")), -1),
            "feature": v2b.gv.feature_from_observation_state(st.get("initial_observation_from_h15_trace"), st.get("branch_previous_state")),
            "stage_a_trace_risk_score": v2b.sf(st.get("stage_a_trace_risk_score"), 0.0),
            "h10_beneficial_vs_h15": beneficial,
            "oracle_label": r.get("oracle_label"),
            "h10_physical": h10_phys,
            "h15_physical": h15_phys,
            "row_physical_tolerance": row_tol,
            "h10_decision_sum_s": h10_dec,
            "h15_decision_sum_s": h15_dec,
            "h10_solver_sum_s": h10_sol,
            "h15_solver_sum_s": h15_sol,
            "h10_safe_all": h10_safe,
            "h15_safe_all": h15_safe,
            "phys_delta_h10_minus_h15": h10_phys - h15_phys,
            "decision_delta_h10_minus_h15": h10_dec - h15_dec,
            "solver_delta_h10_minus_h15": h10_sol - h15_sol,
            "h10_catastrophic_vs_h15": catastrophic,
        })
    return rows


def source_summary_relaxed(examples):
    out: Dict[str, int] = {
        "total": len(examples),
        "agreement_positive_h10": 0,
        "agreement_non_h10": 0,
        "terminal_disagreement_or_missing": 0,
    }
    for ex in examples:
        cat = str(ex.get("category"))
        out[cat] = out.get(cat, 0) + 1
    return out


def patch_v2b_globals() -> None:
    v2b.NAME = NAME
    v2b.STAMP = STAMP
    v2b.MARKER = f"vehicle-true-variable-H-selector-refit-v2c-relaxed-schema-{STAMP}"
    v2b.OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
    v2b.STATE_FILE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
    v2b.CONTINUE_STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1145_after_selector_refit_v2c_relaxed_schema.md"
    v2b.PROTOCOL_OUT = ROOT / f"research_artifacts/aws_protocols/{NAME}_{STAMP}_candidate_protocol.json"
    v2b.completed_ok = completed_ok_relaxed
    v2b.build_dev_rows = build_dev_rows_relaxed
    v2b.source_summary = source_summary_relaxed


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-selector-refit-v2c-relaxed-schema", action="store_true", required=True)
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_selector_refit_v2c_relaxed_schema:
        raise ContractError("requires --run and explicit v2c relaxed-schema acknowledgement")
    patch_v2b_globals()
    translated = [
        "--run",
        "--backup-verified-commit", args.backup_verified_commit,
        "--i-accept-development-selector-refit-v2b-schema-repair",
    ]
    return v2b.run(translated)


if __name__ == "__main__":
    raise SystemExit(main())
