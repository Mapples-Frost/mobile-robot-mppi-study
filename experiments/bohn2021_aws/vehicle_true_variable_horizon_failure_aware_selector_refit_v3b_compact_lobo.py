#!/usr/bin/env python3
"""Compact bounded wrapper for failure-aware selector refit v3 LOBO.

The base v3 source preserves the full design.  This wrapper applies two bounded
engineering constraints before execution:

* compact evaluation records (drop per-row details for every candidate; retain
  false-positive/catastrophic rows and top-candidate holdout summaries), and
* a deployable-feature-focused candidate subset to keep the no-simulation LOBO
  diagnostic within t3a.medium memory/time while still testing the discriminating
  hypothesis after the v2 catastrophic false positives.

No simulations, no validation64 access, no sealed-test access and no gradient
training are performed.  Results remain opened-development evidence only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_failure_aware_selector_refit_v3_lobo as v3  # noqa:E402

NAME = "vehicle_true_variable_horizon_failure_aware_selector_refit_v3b_compact_lobo"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
MARKER = f"vehicle-true-variable-H-failure-aware-selector-refit-v3b-compact-lobo-{STAMP}"


def compact_evaluate_model(model: Mapping[str, Any], rows, bank_id: str, profile: str):
    out = _ORIG_EVALUATE_MODEL(model, rows, bank_id, profile)
    # Keep the failure rows that determine gates, but do not store every per-state
    # diagnostic for every enumerated candidate.
    out.pop("details", None)
    return out


def bounded_candidate_specs():
    specs = _ORIG_CANDIDATE_SPECS()
    keep_reps = {
        "raw_abs_l2",
        "pose_goal_obs_l2",
        "obstacle_slice_l2",
        "deploy_obs_state_step_risk_std",
        "deploy_pose_state_step_risk_std",
        "deploy_obs_state_noabs_std",
    }
    keep_label_modes = {
        "primary_only",
        "primary_plus_matched_cat_veto",
        "terminal_consensus_positive",
        "low_regret_positive_primary",
    }
    keep_radius_q = {0.5, 0.75}
    keep_radius_scale = {0.75, 1.0, 1.25, 1.5}
    bounded = []
    for s in specs:
        if s.get("representation") not in keep_reps:
            continue
        if s.get("label_mode") not in keep_label_modes:
            continue
        if float(s.get("positive_radius_quantile")) not in keep_radius_q:
            continue
        if float(s.get("positive_radius_scale")) not in keep_radius_scale:
            continue
        # Retain all negative/veto pool choices and margins because the v2 failure
        # is exactly a false-positive/veto problem.
        bounded.append(dict(s))
    return bounded


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--backup-verified-commit", required=True)
    parser.add_argument("--i-accept-development-failure-aware-selector-refit-v3b-compact-lobo", action="store_true", required=True)
    args = parser.parse_args(argv)
    if not args.i_accept_development_failure_aware_selector_refit_v3b_compact_lobo:
        raise RuntimeError("explicit v3b compact LOBO acknowledgement required")

    # Patch output identity and bounded implementation hooks.
    v3.NAME = NAME
    v3.STAMP = STAMP
    v3.MARKER = MARKER
    v3.OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
    v3.STATE_FILE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_failure_aware_selector_refit_v3b_compact_lobo.md"
    v3.evaluate_model = compact_evaluate_model
    v3.candidate_specs = bounded_candidate_specs

    translated = [
        "--run",
        "--backup-verified-commit", args.backup_verified_commit,
        "--i-accept-development-failure-aware-selector-refit-v3-lobo",
    ]
    rc = v3.main(translated)
    print(json.dumps({
        "wrapper": NAME,
        "stamp": STAMP,
        "bounded_candidate_specs": len(bounded_candidate_specs()),
        "delegated_return_code": rc,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, indent=2, sort_keys=True))
    return rc


_ORIG_EVALUATE_MODEL = v3.evaluate_model
_ORIG_CANDIDATE_SPECS = v3.candidate_specs

if __name__ == "__main__":
    raise SystemExit(main())
