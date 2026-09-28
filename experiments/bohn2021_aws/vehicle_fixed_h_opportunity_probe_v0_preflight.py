#!/usr/bin/env python3
"""Vehicle fixed-H opportunity probe V0 metadata preflight.

This is a no-rollout/no-training/no-validation/no-test preflight after the
scenario-opportunity capability audit.  It freezes a bounded fixed-H opportunity
probe design but does not execute simulations.  The subsequent rollout remains
blocked until external backup covers the capability audit and this preflight.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T1000Z"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_preflight_20260928T1000Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_fixed_h_opportunity_probe_v0_preflight_20260928T1000Z.md"
BACKUP_PATH = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_FIXED_H_OPPORTUNITY_PROBE_V0_PREFLIGHT_20260928T1000Z.json"
FROZEN_MD = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v0_frozen_20260928.md"
FROZEN_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v0_frozen_20260928.json"
CAP_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/completed.json"
CAP_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/raw.json"
CAP_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/summary.md"
DRAFT_MD = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v0_after_capability_audit_20260928.md"
VALIDATION_GATE = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"
MARKER = "vehicle-fixed-h-opportunity-probe-v0-preflight-20260928T1000Z"


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
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_capability_inputs() -> Dict[str, Any]:
    cap_done = read_json(CAP_COMPLETED)
    if cap_done.get("passed") is not True or cap_done.get("hard_pass") is not True:
        raise RuntimeError("Capability diagnostic did not pass")
    if cap_done.get("historical_validation64_bank_opened") is not False or cap_done.get("sealed_test_accessed") is not False:
        raise RuntimeError("Capability diagnostic access flags are invalid")
    cap_raw = read_json(CAP_RAW)
    if cap_raw.get("new_rollouts") != 0 or cap_raw.get("new_control_steps") != 0:
        raise RuntimeError("Capability diagnostic unexpectedly records rollouts")
    return {"completed": cap_done, "raw": cap_raw}


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


def build_protocol(created: str, cap: Mapping[str, Any], term: Mapping[str, Any]) -> Dict[str, Any]:
    horizons = list(range(5, 51, 5))
    protocol = {
        "protocol_id": "vehicle_fixed_h_opportunity_probe_v0_frozen_20260928",
        "classification": "diagnostic_development_IMPROVED_context_fixed_H_opportunity_probe_not_model_selection_not_final_test",
        "created_utc": created,
        "motivating_evidence": [
            "V2b smoke was all-success/equal-step with near-zero physical deltas and mixed actual timing.",
            "Capability audit verified only straight-line heading/length variation plus built-in three-obstacle generator; no curved-path, initial-speed, robust-uncertainty, or registered process-noise factors.",
            "Before another adaptive validation campaign, fixed-H Pareto opportunity must be measured from identical fresh states with actual wall time."
        ],
        "split": "vehicle_fixed_h_opportunity_probe_v0_fresh_no_validation64_no_test",
        "access_rules": {
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "development_only": True,
            "requires_verified_backup_after_preflight_before_rollout": True,
            "do_not_modify_current_frozen_validation_campaign": True
        },
        "scenario_generator": {
            "candidate_pool_resets": 24,
            "selected_case_count": 8,
            "rng_seed": 2609286001,
            "selection_rule": "generate 24 canonical source-supported TTAHMPC cases; compute metadata without control rollout; deterministically select 8 covering low/high |theta_r|, short/long traj_steps, and low/high natural nearest obstacle clearance while preserving easy and hard strata",
            "source_supported_factors_used": ["theta_r", "traj_steps", "built_in_3_obstacle_positions_radii_noise", "fixed initial pose from registered config"],
            "explicitly_excluded_factors": ["curved trajectories", "direct initial speed", "robust MPC uncertainty", "plant process noise/disturbances", "hand-placed only-favorable obstacles"]
        },
        "rollout_design": {
            "horizons": horizons,
            "terminal_value": "independent seed0 fixed-H terminal for each H when available; if any H terminal is unavailable the rollout must stop and be re-protocolled rather than silently falling back",
            "arms": [{"family": "fixed_H_independent_terminal_seed0_grid", "h": h, "terminal_source": term.get("terminal_sources", {}).get(str(h), {})} for h in horizons],
            "episodes_exact": 8 * len(horizons),
            "control_step_upper_bound": 8 * len(horizons) * 150,
            "max_steps_per_episode": 150,
            "legacy_interpreter_required": True,
            "single_thread_runtime_required": True
        },
        "metrics": [
            "physical_constraint_cost", "total_cost", "performance_cost", "h_penalty", "success", "termination", "constraints", "initial_failed_steps", "solver_failure_steps", "retries", "steps", "decision_timing_s", "solver_attempt_timing_s", "construction_s", "reset_gross_s", "deadline_exceed_steps"
        ],
        "primary_diagnostic_acceptance": {
            "opportunity_exists_if": "at least two selected scenario strata have different nondominated fixed-H optima or materially different cost-vs-time slopes after safety/failure filtering",
            "weak_opportunity_if": "one H or a monotone neighboring-H set dominates across strata after actual timing and safety are considered",
            "no_adaptive_win_forced": True
        },
        "budget": {"new_training_episodes": 0, "new_gradient_steps": 0, "fresh_bank_candidate_resets": 24, "rollout_episodes_exact": 80, "control_step_upper_bound": 12000, "sealed_test_episodes": 0},
        "capability_audit_hashes": {rel(CAP_COMPLETED): sha256(CAP_COMPLETED), rel(CAP_RAW): sha256(CAP_RAW), rel(CAP_SUMMARY): sha256(CAP_SUMMARY)},
        "terminal_grid_readiness": term,
        "protocol_sha256_without_self": None,
    }
    protocol["protocol_sha256_without_self"] = canonical_hash(protocol)
    return protocol


def write_markdown(protocol: Mapping[str, Any], term: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle fixed-H opportunity probe V0 frozen protocol",
        "",
        f"Frozen UTC: `{protocol['created_utc']}`.",
        "",
        "## Status and access",
        "",
        "This is a frozen development diagnostic protocol only. It does not authorize final-test access and does not alter the frozen historical validation campaigns. The rollout must not start until external backup covers the capability audit and this preflight.",
        "",
        "## Hypothesis",
        "",
        "Weak adaptive results may reflect absence of useful state-dependent horizon opportunity in the current straight-line vehicle scenario distribution. The probe maps the fixed-H cost/timing Pareto surface across source-supported scenario strata before spending another long adaptive validation budget.",
        "",
        "## Scenario and exclusions",
        "",
        "- Generate 24 fresh candidate cases from the registered source-supported TTAHMPC straight-line generator, then deterministically select 8 covering heading magnitude, path length and natural obstacle clearance strata.",
        "- Preserve both easy and hard cases; do not hand-pick only favorable cases.",
        "- Exclude curved paths, direct initial speed, robust uncertainty and plant process-noise stress unless a separate environment amendment is written.",
        "",
        "## Rollout design",
        "",
        f"- Split: `{protocol['split']}`.",
        f"- Horizons: `{protocol['rollout_design']['horizons']}` (full 5..50 grid including H45; this corrects the earlier draft's accidental omission of H45).",
        "- Terminal values: independent seed0 fixed-H terminal for each H; abort/re-protocol if any terminal source is missing.",
        f"- Episodes/control bound: `{protocol['rollout_design']['episodes_exact']}` / `{protocol['rollout_design']['control_step_upper_bound']}`.",
        "- Runtime: legacy Python/TF1, single-thread, actual decision and solver wall time recorded.",
        "",
        "## Terminal readiness",
        "",
        f"- available_all_required: `{term['available_all_required']}`; missing: `{term['missing_horizons']}`; bad_entries: `{term['bad_entries']}`.",
        "",
        "## Acceptance",
        "",
        "Opportunity exists only if at least two scenario strata show different nondominated fixed-H optima or materially different cost/time slopes. If one fixed H dominates everywhere, document weak opportunity rather than forcing switching.",
        "",
        "## Interpretation",
        "",
        "This future probe will be development evidence only. It cannot support a final success claim, cannot tune sealed-test outcomes, and must be followed by fair adaptive/fixed validation if it finds opportunity.",
    ]
    FROZEN_MD.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## Vehicle fixed-H opportunity probe V0 preflight\n\n"
        f"UTC: {raw['created_utc']}. Metadata-only preflight froze the next bounded fixed-H opportunity-probe design: "
        f"80 planned fixed-H episodes, <=12000 control steps, 24 fresh candidate-bank resets, full H grid 5..50, independent seed0 terminal per H. "
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
    cap = validate_capability_inputs()
    gate = read_json(VALIDATION_GATE)
    term = terminal_grid_readiness(gate)
    protocol = build_protocol(created, cap, term)
    write_markdown(protocol, term)
    write_json(FROZEN_JSON, protocol)

    hard_pass = bool(term["available_all_required"])
    decision = {
        "headline": "Fixed-H opportunity probe V0 is protocol-ready but rollout remains backup-blocked.",
        "rollout_allowed_now": False,
        "rollout_blocker": "external backup must cover capability audit and this preflight before any new simulations",
        "next_after_backup": "write_or_run_vehicle_fixed_h_opportunity_probe_v0_runner_legacy_fresh_bank_no_validation64_no_test",
        "do_not_next": "do_not_resume_unchanged_long_adaptive_validation_before_fixed_H_opportunity_probe",
    }
    raw: Dict[str, Any] = {
        "created_utc": created,
        "method": "vehicle_fixed_h_opportunity_probe_v0_metadata_preflight",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "input_hashes": {rel(p): sha256(p) for p in [CAP_COMPLETED, CAP_RAW, CAP_SUMMARY, DRAFT_MD, VALIDATION_GATE, Path(__file__).resolve()] if p.exists()},
        "frozen_protocol": rel(FROZEN_MD),
        "frozen_protocol_json": rel(FROZEN_JSON),
        "terminal_grid_readiness": term,
        "budget_freeze": protocol["budget"],
        "decision": decision,
        "hard_pass": hard_pass,
    }
    lines = [
        "# Vehicle fixed-H opportunity probe V0 preflight",
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
    state = (
        f"# Vehicle fixed-H opportunity probe V0 preflight state ({created})\n\n"
        f"Decision: {decision['headline']}\n\n"
        f"No simulations/training/validation64/test access. Terminal grid ready={term['available_all_required']}. "
        "Backup is required before any rollout. Next after backup: write/run the legacy fixed-H opportunity probe runner under the frozen protocol.\n"
    )
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(state, encoding="utf-8")
    write_json(BACKUP_PATH, {
        "requested_utc": created,
        "reason": "backup fixed-H opportunity probe V0 preflight/protocol before any fresh diagnostic rollout",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(BACKUP_PATH), rel(FROZEN_MD), rel(FROZEN_JSON), rel(Path(__file__).resolve())],
    })
    append_docs(raw)
    files = [OUT_DIR / "summary.md", OUT_DIR / "raw.json", STATE_PATH, BACKUP_PATH, FROZEN_MD, FROZEN_JSON, Path(__file__).resolve(), CAP_COMPLETED, CAP_RAW, VALIDATION_GATE]
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
