#!/usr/bin/env python3
"""Vehicle fixed-H opportunity probe V1 metadata preflight.

No-rollout/no-training/no-validation/no-test preflight.  It freezes an enlarged
fixed-H opportunity map after V0 showed nonconstant but very small oracle gains.
The subsequent rollout remains backup-blocked until external backup covers this
preflight and the V1 runner source.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T1045Z"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_preflight_20260928T1045Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_fixed_h_opportunity_probe_v1_preflight_20260928T1045Z.md"
BACKUP_PATH = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_FIXED_H_OPPORTUNITY_PROBE_V1_PREFLIGHT_20260928T1045Z.json"
FROZEN_MD = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v1_frozen_20260928.md"
FROZEN_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v1_frozen_20260928.json"
CAP_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/completed.json"
CAP_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/raw.json"
CAP_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/summary.md"
V0_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/completed.json"
V0_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/raw.json"
V0_POST_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_postdiagnostic_20260928T1030Z/completed.json"
V0_POST_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_postdiagnostic_20260928T1030Z/raw.json"
THREE_LAYER = ROOT / "research_artifacts/aws_state/vehicle_fixed_h_opportunity_three_layer_diagnosis_20260928T1035Z.md"
VALIDATION_GATE = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"
V1_RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_fixed_h_opportunity_probe_v1_runner.py"
MARKER = "vehicle-fixed-h-opportunity-probe-v1-preflight-20260928T1045Z"


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def verify_completed(path: Path) -> Dict[str, Any]:
    done = read_json(path)
    if done.get("passed") is not True:
        raise RuntimeError("Completed marker did not pass: %s" % rel(path))
    if done.get("historical_validation64_bank_opened") is not False or done.get("sealed_test_accessed") is not False:
        raise RuntimeError("Access flags are invalid in %s" % rel(path))
    for name, expected in done.get("hashes", {}).items():
        p = ROOT / name
        if not p.exists() or sha256(p) != expected:
            raise RuntimeError("Hash mismatch/missing in completed marker %s for %s" % (rel(path), name))
    return done


def terminal_grid_readiness(gate: Mapping[str, Any]) -> Dict[str, Any]:
    entries = (gate.get("fixed_comparators") or {}).get("independent_terminal_full_grid_seed0") or []
    required = list(range(5, 51, 5))
    by_h: Dict[int, Dict[str, Any]] = {}
    missing = []
    bad = []
    for row in entries:
        h = int(row.get("h"))
        by_h[h] = row
    for h in required:
        row = by_h.get(h)
        if row is None:
            missing.append(h)
            continue
        if not bool(row.get("available_complete")) or not bool(row.get("available_for_rollout")):
            bad.append({"h": h, "reason": "not available_complete/for_rollout"})
            continue
        for key in ("model.zip", "manifest.json", "completed.json"):
            p = ROOT / row[key]["path"]
            if not p.exists():
                bad.append({"h": h, "reason": "missing " + key, "path": row[key]["path"]})
            elif sha256(p) != row[key]["sha256"]:
                bad.append({"h": h, "reason": "sha mismatch " + key, "path": row[key]["path"]})
    return {
        "required_horizons": required,
        "available_all_required": (not missing and not bad),
        "missing_horizons": missing,
        "bad_entries": bad,
        "terminal_sources": {str(h): {"path": by_h[h].get("path"), "model_zip_sha256": by_h[h]["model.zip"]["sha256"]} for h in required if h in by_h and h not in missing},
    }


def build_protocol(created: str, term: Mapping[str, Any]) -> Dict[str, Any]:
    horizons = list(range(5, 51, 5))
    protocol = {
        "protocol_id": "vehicle_fixed_h_opportunity_probe_v1_frozen_20260928",
        "classification": "diagnostic_development_IMPROVED_context_fixed_H_opportunity_probe_not_model_selection_not_final_test",
        "created_utc": created,
        "motivating_evidence": [
            "V0 fixed-H probe found nonconstant per-case best physical H but very small oracle best-physical improvement vs H15: 0.747 physical+constraint over 8 cases, and oracle best-total improvement only 0.850 total.",
            "V0 LOOCV metadata selectors were worse than H15, so adaptive training/refit is not yet warranted on the canonical source-supported distribution.",
            "Capability audit showed only straight-line heading/length/obstacle factors are source-supported without a stronger scenario redesign. V1 tests whether the weak V0 margin is a small-bank artifact using a larger source-supported bank.",
        ],
        "split": "vehicle_fixed_h_opportunity_probe_v1_fresh64_select16_no_validation64_no_test",
        "access_rules": {
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "development_only": True,
            "requires_verified_backup_after_preflight_before_rollout": True,
            "do_not_modify_current_frozen_validation_campaign": True,
            "do_not_use_for_final_model_selection_or_final_test_gate": True,
        },
        "scenario_generator": {
            "candidate_pool_resets": 64,
            "selected_case_count": 16,
            "rng_seed": 2609287001,
            "selection_rule": "generate 64 canonical source-supported TTAHMPC cases; compute metadata without control rollout; deterministically select 16 using the V0 stratified/farthest-diversity rule over low/high |theta_r|, short/long traj_steps and low/high natural obstacle clearance, preserving both easy and hard naturally generated cases",
            "source_supported_factors_used": ["theta_r", "traj_steps", "built_in_3_obstacle_positions_radii_noise", "fixed initial pose from registered config"],
            "explicitly_excluded_factors": ["curved trajectories", "direct initial speed", "robust MPC uncertainty", "plant process noise/disturbances", "hand-placed only-favorable obstacles"],
            "original_task_fidelity": "paper-aligned straight-line vehicle distribution, but enlarged fresh development diagnostic bank; not a redesigned stress benchmark",
        },
        "rollout_design": {
            "horizons": horizons,
            "terminal_value": "independent seed0 fixed-H terminal for each H; abort/re-protocol if any H terminal is unavailable",
            "arms": [{"family": "fixed_H_independent_terminal_seed0_grid", "h": h, "terminal_source": term.get("terminal_sources", {}).get(str(h), {})} for h in horizons],
            "episodes_exact": 16 * len(horizons),
            "control_step_upper_bound": 16 * len(horizons) * 150,
            "max_steps_per_episode": 150,
            "legacy_interpreter_required": True,
            "single_thread_runtime_required": True,
            "actual_decision_and_solver_wall_time_recorded": True,
        },
        "metrics": [
            "physical_constraint_cost", "total_cost", "performance_cost", "h_penalty", "success", "termination", "constraints", "initial_failed_steps", "solver_failure_steps", "retries", "steps", "decision_timing_s", "solver_attempt_timing_s", "construction_s", "reset_gross_s", "deadline_exceed_steps",
        ],
        "materiality_thresholds": {
            "primary_fixed_H_reference": "best all-success no-constraint aggregate fixed H selected within the same V1 bank, expected to be H15 if V0 generalizes but not fixed a priori",
            "material_oracle_physical_opportunity": "same-bank oracle best-physical must improve physical+constraint cost by >=5 percent or >=5 absolute units versus the V1 aggregate physical reference while not increasing total cost by >5 percent, after safety/solver filtering",
            "material_oracle_total_opportunity": "same-bank oracle best-total must improve total cost by >=3 percent or >=3 absolute units versus the V1 aggregate total reference without any solver/constraint regression",
            "predictability_gate_for_adaptive_refit": "a frozen/simple LOOCV metadata selector or continuation-state selector must beat the same-bank aggregate reference on risk score; otherwise do not launch selector/refit training solely from oracle hindsight",
            "weak_opportunity_decision": "if thresholds fail, document weak canonical-distribution adaptive opportunity and move to versioned scenario-redesign/continuation diagnostics rather than another unchanged adaptive campaign",
        },
        "budget": {"new_training_episodes": 0, "new_gradient_steps": 0, "fresh_bank_candidate_resets": 64, "rollout_episodes_exact": 160, "control_step_upper_bound": 24000, "sealed_test_episodes": 0},
        "hash_inputs": {},
        "terminal_grid_readiness": term,
        "protocol_sha256_without_self": None,
    }
    protocol["protocol_sha256_without_self"] = canonical_hash(protocol)
    return protocol


def write_markdown(protocol: Mapping[str, Any], term: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle fixed-H opportunity probe V1 frozen protocol",
        "",
        f"Frozen UTC: `{protocol['created_utc']}`.",
        "",
        "## Status and access",
        "",
        "Development diagnostic only. This does not authorize final-test access, does not read the historical validation64 bank, and does not alter frozen validation campaigns. Rollout is blocked until external backup covers this preflight and the V1 runner source.",
        "",
        "## Frozen hypothesis",
        "",
        "V0's nonconstant per-case fixed-H optima may be a small-bank artifact with too little material oracle margin to justify adaptive training. V1 expands the source-supported straight-line vehicle bank to decide whether material state/scenario-dependent horizon opportunity exists before spending another adaptive-selector budget.",
        "",
        "## Scenario and exclusions",
        "",
        "- Generate 64 fresh candidate cases from the registered source-supported TTAHMPC straight-line generator.",
        "- Deterministically select 16 cases using the same metadata-only stratum/diversity rule as V0 over heading magnitude, trajectory length and natural obstacle clearance.",
        "- Preserve both favorable and unfavorable naturally generated cases; no outcome-based case selection.",
        "- Exclude curved paths, direct initial speed, robust uncertainty and plant process noise unless a separate versioned scenario-redesign protocol is frozen.",
        "",
        "## Rollout design",
        "",
        f"- Split: `{protocol['split']}`.",
        f"- Horizons: `{protocol['rollout_design']['horizons']}`.",
        "- Terminal values: independent seed0 fixed-H terminal for each H; no fallback if missing.",
        f"- Episodes/control bound: `{protocol['rollout_design']['episodes_exact']}` / `{protocol['rollout_design']['control_step_upper_bound']}`.",
        "- Runtime: legacy Python/TF1, single-thread, actual decision/solver timing recorded.",
        "",
        "## Materiality and next-decision rules",
        "",
        "```json",
        json.dumps(protocol["materiality_thresholds"], indent=2, sort_keys=True),
        "```",
        "",
        "If V1 fails materiality, do not launch another adaptive validation campaign on the canonical distribution merely because oracle labels vary. Move to a versioned scenario-redesign or continuation/value diagnostic. If V1 passes materiality and a predictor beats the aggregate reference, freeze one IMPROVED selector/refit smoke with fair fixed-H baselines.",
        "",
        "## Terminal readiness",
        "",
        f"- available_all_required: `{term['available_all_required']}`; missing: `{term['missing_horizons']}`; bad_entries: `{term['bad_entries']}`.",
    ]
    FROZEN_MD.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## Vehicle fixed-H opportunity probe V1 preflight\n\n"
        f"UTC: {raw['created_utc']}. Metadata-only preflight froze an enlarged fixed-H opportunity map: "
        f"160 planned fixed-H episodes, <=24000 control steps, 64 fresh candidate-bank resets, 16 selected source-supported cases, full H grid 5..50. "
        f"Terminal grid ready={raw['terminal_grid_readiness']['available_all_required']}. No simulations/training/validation64/test access occurred. "
        "External backup is required before the rollout. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'completed.json')}`, `{rel(FROZEN_MD)}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    # Verify all inputs and their completed hashes before freezing V1.
    cap_done = verify_completed(CAP_COMPLETED)
    v0_done = verify_completed(V0_COMPLETED)
    v0_post_done = verify_completed(V0_POST_COMPLETED)
    v0_post_raw = read_json(V0_POST_RAW)
    if v0_post_raw.get("diagnostic_flags", {}).get("oracle_material_development_only") is not False:
        raise RuntimeError("V0 postdiagnostic does not support the V1 small-bank-artifact hypothesis")
    if not V1_RUNNER.exists():
        raise RuntimeError("V1 runner source missing; write runner before preflight so backup request covers it")
    gate = read_json(VALIDATION_GATE)
    term = terminal_grid_readiness(gate)
    protocol = build_protocol(created, term)
    protocol["hash_inputs"] = {rel(p): sha256(p) for p in [CAP_COMPLETED, CAP_RAW, CAP_SUMMARY, V0_COMPLETED, V0_RAW, V0_POST_COMPLETED, V0_POST_RAW, THREE_LAYER, VALIDATION_GATE, V1_RUNNER, Path(__file__).resolve()] if p.exists()}
    protocol["capability_completed_headline"] = cap_done.get("headline")
    protocol["v0_completed_headline"] = v0_done.get("headline")
    protocol["v0_post_headline"] = v0_post_done.get("headline")
    protocol["protocol_sha256_without_self"] = canonical_hash({k: v for k, v in protocol.items() if k != "protocol_sha256_without_self"})
    write_markdown(protocol, term)
    write_json(FROZEN_JSON, protocol)

    hard_pass = bool(term["available_all_required"])
    decision = {
        "headline": "Fixed-H opportunity probe V1 is protocol-ready but rollout remains backup-blocked.",
        "rollout_allowed_now": False,
        "rollout_blocker": "external backup must cover V1 preflight/protocol and runner source before any new simulations",
        "next_after_backup": "run_vehicle_fixed_h_opportunity_probe_v1_runner_legacy_fresh64_select16_no_validation64_no_test",
        "do_not_next": "do_not_launch_adaptive_training_or_long_validation_until_V1_materiality_postdiagnostic",
    }
    raw: Dict[str, Any] = {
        "created_utc": created,
        "method": "vehicle_fixed_h_opportunity_probe_v1_metadata_preflight",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "input_hashes": protocol["hash_inputs"],
        "frozen_protocol": rel(FROZEN_MD),
        "frozen_protocol_json": rel(FROZEN_JSON),
        "terminal_grid_readiness": term,
        "budget_freeze": protocol["budget"],
        "materiality_thresholds": protocol["materiality_thresholds"],
        "decision": decision,
        "hard_pass": hard_pass,
    }
    lines = [
        "# Vehicle fixed-H opportunity probe V1 preflight",
        "",
        f"UTC: `{created}`. Metadata-only; no simulations, no training, no validation64-bank access, no sealed-test access.",
        "",
        "## Frozen design",
        "",
        f"- Split: `{protocol['split']}`.",
        f"- Fresh candidate cases/resets: `{protocol['scenario_generator']['candidate_pool_resets']}`; selected cases: `{protocol['scenario_generator']['selected_case_count']}`.",
        f"- Horizons: `{protocol['rollout_design']['horizons']}`.",
        f"- Episodes/control-step upper bound: `{protocol['rollout_design']['episodes_exact']}` / `{protocol['rollout_design']['control_step_upper_bound']}`.",
        "- Terminal policy: independent seed0 fixed-H terminal for each H; no silent fallback allowed.",
        "",
        "## Frozen materiality thresholds",
        "",
        "```json",
        json.dumps(protocol["materiality_thresholds"], indent=2, sort_keys=True),
        "```",
        "",
        "## Readiness",
        "",
        f"- terminal_grid_available_all_required: `{term['available_all_required']}`.",
        f"- missing_horizons: `{term['missing_horizons']}`.",
        f"- bad_entries: `{term['bad_entries']}`.",
        f"- hard_pass: `{hard_pass}`.",
        "",
        "## Decision",
        "",
        f"{decision['headline']} Rollout allowed now: `{decision['rollout_allowed_now']}` because `{decision['rollout_blocker']}`.",
        "",
        f"Frozen protocol: `{rel(FROZEN_MD)}`; JSON: `{rel(FROZEN_JSON)}`.",
        f"Backup request: `{rel(BACKUP_PATH)}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT_DIR / "raw.json", raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle fixed-H opportunity probe V1 preflight state ({created})\n\n"
        f"Decision: {decision['headline']}\n\n"
        f"No simulations/training/validation64/test access. Terminal grid ready={term['available_all_required']}. "
        "Backup is required before any rollout. Next after backup: run the legacy V1 fixed-H opportunity runner under the frozen protocol.\n",
        encoding="utf-8",
    )
    write_json(BACKUP_PATH, {
        "requested_utc": created,
        "reason": "backup fixed-H opportunity probe V1 preflight/protocol/runner before enlarged fresh diagnostic rollout",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(BACKUP_PATH), rel(FROZEN_MD), rel(FROZEN_JSON), rel(V1_RUNNER), rel(Path(__file__).resolve())],
    })
    append_docs(raw)
    files = [OUT_DIR / "summary.md", OUT_DIR / "raw.json", STATE_PATH, BACKUP_PATH, FROZEN_MD, FROZEN_JSON, V1_RUNNER, Path(__file__).resolve(), CAP_COMPLETED, V0_COMPLETED, V0_POST_COMPLETED, THREE_LAYER, VALIDATION_GATE]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": hard_pass,
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "backup_request": rel(BACKUP_PATH),
        "frozen_protocol": rel(FROZEN_MD),
        "frozen_protocol_json": rel(FROZEN_JSON),
        "next_after_backup": decision["next_after_backup"],
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
        "headline": decision["headline"],
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "hard_pass": hard_pass,
        "new_rollouts": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "frozen_protocol": rel(FROZEN_MD),
        "backup_request": rel(BACKUP_PATH),
        "decision": decision["headline"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
