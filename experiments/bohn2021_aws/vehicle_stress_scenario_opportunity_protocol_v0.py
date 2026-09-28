#!/usr/bin/env python3
"""Freeze Vehicle V1 stress-scenario opportunity protocol v0.

No-simulation/no-training diagnostic.  This converts the latest canonical
negative/weak-opportunity evidence into a versioned, source-supported stress
scenario opportunity protocol before any new rollout or retraining.  It does not
open historical validation64 or sealed final-test data.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T1625Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v0_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/vehicle_stress_scenario_opportunity_protocol_v0_{STAMP}.md"
PROTO_MD = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.md"
PROTO_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.json"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_OPPORTUNITY_PROTOCOL_V0_{STAMP}.json"
MARKER = "vehicle-stress-scenario-opportunity-protocol-v0-20260928T1625Z"

INPUTS = {
    "capability_completed": ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/completed.json",
    "v1_fixed_h_post_completed": ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_postdiagnostic_20260928T1150Z/completed.json",
    "fresh_probe_completed": ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928/completed.json",
    "transient_rollout_completed": ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_continuation_probe_v0_20260928/completed.json",
    "prefix_postdiag_completed": ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_prefix_artifact_postdiagnostic_v0_20260928T1620Z/completed.json",
    "prefix_postdiag_raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_prefix_artifact_postdiagnostic_v0_20260928T1620Z/raw.json",
    "v1_fixed_h_protocol": ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v1_frozen_20260928.json",
}


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def verify_completed(name: str, p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise RuntimeError(f"missing required input {name}: {rel(p)}")
    obj = read_json(p)
    if obj.get("passed") is not True:
        raise RuntimeError(f"input did not pass: {name} {rel(p)}")
    if obj.get("historical_validation64_bank_opened") is not False:
        raise RuntimeError(f"input unexpectedly opened validation64: {name}")
    if obj.get("sealed_test_accessed") is not False:
        raise RuntimeError(f"input unexpectedly accessed sealed test: {name}")
    return obj


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        if p.exists():
            old = p.read_text(encoding="utf-8")
            if MARKER not in old:
                p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    completed = {name: verify_completed(name, path) for name, path in INPUTS.items() if name.endswith("completed")}
    v1_protocol = read_json(INPUTS["v1_fixed_h_protocol"])
    prefix_raw = read_json(INPUTS["prefix_postdiag_raw"])
    if prefix_raw.get("historical_validation64_bank_opened") is not False or prefix_raw.get("sealed_test_accessed") is not False:
        raise RuntimeError("prefix postdiagnostic raw access flags invalid")
    term = v1_protocol.get("terminal_grid_readiness") or {}
    if term.get("available_all_required") is not True:
        raise RuntimeError("fixed-H terminal grid is not ready for stress opportunity protocol")

    evidence = {
        "canonical_fresh_continuation": {
            "material_positive_states": "0/24",
            "best_non_H15_total_gain": "about 0.560688 (< +3 material threshold)",
            "large_non_H15_harms": 35,
            "interpretation": "fresh canonical identical-state labels provide no usable positive class for immediate refit",
        },
        "transient_continuation_after_prefix_repair": {
            "original_prefix_mismatches": prefix_raw["analysis"].get("original_prefix_mismatch_count_non_H15"),
            "repaired_prefix_mismatches": prefix_raw["analysis"].get("repaired_prefix_mismatch_count_non_H15"),
            "positive_states": prefix_raw["analysis"].get("positive_state_count_after_repair"),
            "best_non_H15_total_gain": prefix_raw["analysis"].get("best_non_H15_total_gain_after_repair"),
            "best_non_H15_physical_gain": prefix_raw["analysis"].get("best_non_H15_physical_gain_after_repair"),
            "large_harms": prefix_raw["analysis"].get("large_non_H15_harms_total_gain_le_minus_threshold"),
            "note": "one non-H15 branch had solver/safety regression; it is retained as negative/adverse evidence and does not create positive label density",
        },
        "fixed_H_v1_postdiagnostic": completed["v1_fixed_h_post_completed"].get("headline", {}),
        "scenario_capability": {
            "supported_now": ["theta_r heading", "traj_steps/goal distance", "built-in straight-line obstacle positions/radii/noise", "initial pose x/y/theta"],
            "excluded_in_v0": ["curved paths", "direct initial speed", "robust uncertainty", "plant process noise", "hand-picked favorable outcomes"],
        },
    }

    protocol: Dict[str, Any] = {
        "protocol_id": "vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928",
        "created_utc": created,
        "classification": "development_diagnostic_IMPROVED_scenario_design_not_original_SAC_not_model_selection_not_final_test",
        "motivation": "Canonical fresh/transient identical-state labels were essentially all H15-dominant, while V1 fixed-H case-level oracle opportunity was material but not predictable from simple metadata.  Before retraining, test whether source-supported stress regimes create reusable state-dependent horizon labels.",
        "access_rules": {
            "development_only": True,
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "final_test_authorized": False,
            "requires_verified_backup_before_runner_source_or_rollout": True,
            "do_not_modify_current_frozen_validation_campaign": True,
            "do_not_use_for_final_model_selection_without_fresh_confirmation": True,
        },
        "hypotheses": [
            {
                "id": "H_scenario_sparse_canonical",
                "rank": 1,
                "claim": "Paper-aligned canonical straight-line Vehicle V1 has too few reusable non-H15 identical-state positives for training/refit, but source-supported stress strata may reveal denser adaptive opportunity.",
                "falsifier": "stress map and continuation labels remain <2 material positive states and best fixed-H dominates without safety/timing tradeoff",
            },
            {
                "id": "H_training_after_labels",
                "rank": 2,
                "claim": "Retraining/refit becomes informative only if stress or repaired labels produce enough positive and negative state labels with clear safety/timing objectives.",
                "falsifier": "state labels remain mostly negative/neutral, implying training would chase noise or H15 dominance",
            },
            {
                "id": "H_objective_comparison_guard",
                "rank": 3,
                "claim": "Any apparent adaptive gain must survive strong same-distribution fixed-H tuning and separate physical, total, safety and actual timing accounting.",
                "falsifier": "gain exists only by unsafe shortcuts, early termination, solver artifacts, or synthetic H penalty unrelated to measured runtime",
            },
        ],
        "scenario_generator_v0": {
            "name": "natural_stress_straight_line_vehicle_v0",
            "fidelity": "same registered TTAHMPC straight-line environment and dynamics; distribution shifts only through source-supported reset/reference/obstacle metadata selection",
            "candidate_pool_resets": 128,
            "rng_seed": 2609288801,
            "selected_cases": 12,
            "selection_rule_pre_outcome": "Generate 128 source-supported candidate resets without control rollout.  Compute only metadata: abs(theta_r), traj_steps/goal distance, built-in obstacle clearance/radius/noise summaries.  Select 8 stress cases from high heading/low natural clearance/extreme length strata using deterministic farthest-diversity, plus 4 control cases from lower-stress strata.  Selection is before any horizon outcome; retain all selected cases.",
            "stress_factors_used": ["high |theta_r| within built-in ±45 degrees", "short and long traj_steps extremes", "low natural obstacle clearance from built-in obstacle generator", "obstacle forecast noise already implemented by TTAHMPC"],
            "not_used_in_v0": ["curved trajectories", "initial speed changes", "process noise/disturbance", "robust MPC uncertainty", "manual outcome-based obstacle placement"],
        },
        "stage1_fixed_H_stress_map": {
            "purpose": "measure whether the revised source-supported stress distribution has material case-level control/compute horizon tradeoff against strong fixed-H baselines",
            "horizons": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "terminal_value": "independent seed0 fixed-H terminal for each H using the verified full grid from V1; abort rather than silently fallback if unavailable",
            "episodes_exact": 120,
            "control_step_upper_bound": 18000,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "actual_timing_recorded": True,
            "postdiagnostic_rules": {
                "primary_references": ["best same-bank all-success/no-constraint fixed-H by total", "best same-bank all-success/no-constraint fixed-H by physical", "fastest safe fixed-H", "strict solver-risk fixed-H"],
                "material_physical_oracle": "oracle best physical must improve physical+constraint by >=5% or >=5 absolute units vs same-bank best fixed reference while total is not >5% worse and safety/solver does not regress",
                "material_total_oracle": "oracle best total must improve total by >=3% or >=3 absolute units vs same-bank best fixed reference without safety/solver regression",
                "predictability_gate": "a frozen LOOCV metadata/state selector must beat the strongest fixed-H risk reference before selector/refit training is launched",
            },
        },
        "stage2_identical_state_continuation_if_stage1_material": {
            "purpose": "distinguish episode-level fixed-H variation from true within-episode adaptive switching opportunity",
            "trigger": "run only if Stage1 finds material oracle opportunity or clear multi-H non-dominated tradeoffs not explained by solver artifacts",
            "state_selection_pre_outcome": "from saved fixed-H traces select up to 12 states balanced across high heading-error, near-obstacle/high-curvature transient proxy, late/steady controls and cases; choose states using metadata and H15/Hbest reference trace statistics, not non-H15 branch outcomes",
            "branch_horizons": [10, 15, 25, 30, 35, 45],
            "prefix_reference": "same saved prefix state from H15 or same-bank best fixed-H reference; prefix comparability must drop only future-arm metadata such as decision.branch_horizon",
            "budget_upper_bound": {"episodes": 72, "control_steps": 10800},
            "positive_label_gate_for_training": ">=2 material non-reference positives with at least one in a stress case and >=2 negative/control states retained, no missing-reference/prefix/state-distance/solver/safety artifact explaining the labels",
        },
        "reward_and_reporting": {
            "report_separately": ["physical/control cost", "total cost including h_penalty", "success", "constraint violations", "initial and final solver failures", "episode length/termination", "whole-decision wall time", "solver-attempt wall time"],
            "forbidden_interpretations": ["shorter H as runtime proxy without measured timing", "unsafe or early-termination shortcuts as improvement", "post-hoc objective reweighting", "switch-count success criterion"],
        },
        "decision_after_protocol": {
            "next_executable_after_backup": "write/import-smoke a Stage1 stress fixed-H runner, then execute 120-episode fixed-H stress map under legacy interpreter",
            "if_stage1_and_stage2_positive": "freeze a compact IMPROVED supervised selector/value-refit smoke with fair same-distribution fixed-H baselines and fresh confirmation bank",
            "if_negative": "preserve canonical and stress negative evidence; diagnose terminal-value/reward/modeling limitations or freeze a stronger separately labeled scenario protocol only if source evidence warrants it; do not retrain on sparse negative labels",
        },
        "input_evidence": evidence,
        "terminal_grid_readiness_reused_from_v1": term,
    }
    protocol["protocol_sha256_without_self"] = canonical_hash({k: v for k, v in protocol.items() if k != "protocol_sha256_without_self"})

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(PROTO_JSON, protocol)
    md = f"""# Vehicle stress-scenario opportunity probe v0 frozen protocol

Frozen UTC: `{created}`. Development-only IMPROVED diagnostic; no historical validation64 access; sealed final test remains closed.

## Why this protocol is now the next informative action

Canonical Vehicle V1 diagnostics now show weak reusable state-level labels: fresh continuation had `0/24` material positives and repaired transient continuation had `0/12` positives after prefix mismatches were reduced from `{evidence['transient_continuation_after_prefix_repair']['original_prefix_mismatches']}` to `{evidence['transient_continuation_after_prefix_repair']['repaired_prefix_mismatches']}`.  V1 fixed-H mapping nevertheless found material episode-level oracle opportunity but poor metadata predictability.  Therefore broad retraining on canonical labels is premature; a versioned stress-opportunity diagnostic has higher information value.

## Scenario design

Use only source-supported straight-line Vehicle/TTAHMPC factors in v0: high heading within ±45°, extreme trajectory length, and naturally generated low obstacle clearance/noise.  Do **not** introduce curved paths, direct initial speed, process noise, robust MPC, or outcome-picked obstacles in this v0 protocol.

Stage 1: generate 128 candidate resets, select 12 cases by metadata before outcomes (8 stress + 4 controls), then run the full fixed-H grid H5..H50: 120 episodes, <=18,000 control steps, zero training.

Stage 2 (conditional): if Stage 1 shows material opportunity, run identical-state continuation comparisons from up to 12 preselected stress/control states using branch horizons [10,15,25,30,35,45], <=72 episodes/10,800 steps.

## Gates

Proceed to selector/refit only if stress results provide material, safe, measurable labels and a simple out-of-sample selector beats the strongest same-distribution fixed-H references.  Otherwise preserve the negative evidence and diagnose terminal/reward/model limitations before retraining.

## Backup and next executable

A verified external backup is required before writing/running the Stage1 runner or any further simulation.  Backup request: `{rel(BACKUP_REQ)}`.
"""
    PROTO_MD.parent.mkdir(parents=True, exist_ok=True)
    PROTO_MD.write_text(md, encoding="utf-8")

    raw = {
        "created_utc": created,
        "method": "vehicle_stress_scenario_opportunity_protocol_v0_no_simulation_freeze",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "protocol": rel(PROTO_JSON),
        "protocol_md": rel(PROTO_MD),
        "input_hashes": {rel(p): sha256(p) for p in INPUTS.values() if p.exists()},
        "decision": protocol["decision_after_protocol"],
        "hypotheses": protocol["hypotheses"],
    }
    write_json(OUT / "raw.json", raw)
    summary = f"""# Vehicle stress-scenario opportunity protocol v0 preflight

UTC: `{created}`. No simulations, no training/refit, no validation64-bank access, no sealed-test access.

Frozen protocol: `{rel(PROTO_MD)}` / `{rel(PROTO_JSON)}`.

Decision: run a source-supported natural-stress fixed-H opportunity map after backup, not immediate retraining. Stage1 budget is 120 episodes / 18,000 control steps; Stage2 identical-state continuation is conditional on Stage1 materiality.

Key preserved evidence: canonical fresh continuation 0/24 positives; repaired transient continuation 0/12 positives, best total gain `{evidence['transient_continuation_after_prefix_repair']['best_non_H15_total_gain']}`; V1 fixed-H oracle materiality exists but metadata predictability gate is false.

Backup request: `{rel(BACKUP_REQ)}`.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        f"# Vehicle stress-scenario opportunity protocol v0 state ({created})\n\n"
        "No-simulation protocol freeze complete. Next: wait for/verify backup, then write/import-smoke Stage1 stress fixed-H runner; do not retrain until stress fixed-H and conditional identical-state labels pass materiality/predictability gates.\n",
        encoding="utf-8",
    )
    write_json(BACKUP_REQ, {
        "requested_utc": created,
        "reason": "backup no-simulation stress-scenario protocol before Stage1 runner source/smoke or any further simulations",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT), rel(STATE), rel(PROTO_MD), rel(PROTO_JSON), rel(BACKUP_REQ), rel(Path(__file__).resolve())],
    })
    block = f"""<!-- {MARKER} -->
## 2026-09-28 vehicle stress-scenario opportunity protocol v0

UTC: {created}. No-simulation protocol freeze completed after canonical fresh/transient continuation produced no material non-H15 label density. Frozen a source-supported natural-stress Vehicle V1 diagnostic: 128 candidate resets, 12 selected cases, Stage1 fixed-H grid H5..H50 with 120 episodes/18,000-step cap, and conditional identical-state continuation before any selector/refit. No validation64 or sealed-test access; no training. Next requires external backup before runner source/smoke or simulation. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(PROTO_MD)}`, `{rel(PROTO_JSON)}`.
"""
    append_docs(block)
    hashes = {rel(p): sha256(p) for p in [OUT / "raw.json", OUT / "summary.md", STATE, PROTO_MD, PROTO_JSON, BACKUP_REQ, Path(__file__).resolve()] if p.exists()}
    write_json(OUT / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_required_before_more_simulations": True,
        "backup_request": rel(BACKUP_REQ),
        "frozen_protocol": rel(PROTO_JSON),
        "headline": {
            "next_action": "after backup, write/import-smoke Stage1 stress fixed-H runner, then execute 120-episode stress opportunity map",
            "stage1_episodes": 120,
            "stage1_control_step_upper_bound": 18000,
            "stage2_conditional": True,
            "retraining_now": False,
        },
        "hashes": hashes,
    })
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "frozen_protocol": rel(PROTO_JSON),
        "stage1_episodes": 120,
        "stage1_control_step_upper_bound": 18000,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(BACKUP_REQ),
        "next_action": "backup_then_stage1_runner_source_and_smoke",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
