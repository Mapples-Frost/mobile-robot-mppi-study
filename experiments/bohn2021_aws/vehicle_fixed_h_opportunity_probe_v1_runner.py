#!/usr/bin/env python3
"""Vehicle fixed-H opportunity probe V1 runner.

Development-only enlarged source-supported fixed-H opportunity map frozen by
research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v1_frozen_20260928.*.

V1 deliberately remains a fixed-H diagnostic: it runs no adaptive policy, no
training/refit, no historical validation64-bank rollouts, and no sealed-test
access.  It expands V0 from 24->64 candidate resets and 8->16 selected cases to
check whether V0's weak oracle margin was a small-bank artifact before launching
another adaptive-horizon training/validation campaign.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_fixed_h_opportunity_probe_v0_runner as base  # noqa:E402

# Make helper functions that record Path(__file__) include this V1 runner rather
# than the V0 module when they are called from our copied main.
base.__file__ = __file__

TASK = "vehicle"
HORIZONS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
CANDIDATE_RESETS = 64
SELECTED_CASES = 16
BANK_RNG = 2609287001
ORDER_SEED = 2609287101
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928"
BANK_DIR = OUT_DIR / "bank"
BANK_PATH = BANK_DIR / "vehicle_fixed_h_opportunity_probe_v1_bank.json"
BANK_COMPLETED = BANK_DIR / "completed.json"
FROZEN_MD = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v1_frozen_20260928.md"
FROZEN_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v1_frozen_20260928.json"
PREFLIGHT_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_preflight_20260928T1045Z/completed.json"
PREFLIGHT_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_preflight_20260928T1045Z/raw.json"
CAPABILITY_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/completed.json"
CAPABILITY_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/raw.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_fixed_h_opportunity_probe_v1_runner_state_20260928.md"
MARKER = "vehicle-fixed-h-opportunity-probe-v1-runner-20260928"

# Patch V0 helper globals used by bank generation, rollout, analysis and backup
# verification.  The helper code is intentionally reused to avoid changing the
# environment/controller mechanics between V0 and V1; only the diagnostic bank
# size, seeds, paths and protocol identity are changed.
base.TASK = TASK
base.HORIZONS = HORIZONS
base.CANDIDATE_RESETS = CANDIDATE_RESETS
base.SELECTED_CASES = SELECTED_CASES
base.BANK_RNG = BANK_RNG
base.ORDER_SEED = ORDER_SEED
base.OUT_DIR = OUT_DIR
base.BANK_DIR = BANK_DIR
base.BANK_PATH = BANK_PATH
base.BANK_COMPLETED = BANK_COMPLETED
base.FROZEN_MD = FROZEN_MD
base.FROZEN_JSON = FROZEN_JSON
base.PREFLIGHT_COMPLETED = PREFLIGHT_COMPLETED
base.PREFLIGHT_RAW = PREFLIGHT_RAW
base.CAPABILITY_COMPLETED = CAPABILITY_COMPLETED
base.CAPABILITY_RAW = CAPABILITY_RAW
base.BACKUP_DIR = BACKUP_DIR
base.STATE_PATH = STATE_PATH
base.MARKER = MARKER

ContractError = base.ContractError
rel = base.rel
sha256 = base.sha256
read_json = base.read_json
write_json = base.write_json
serial = base.serial


def verify_protocol_inputs() -> Dict[str, Any]:
    """V1-specific frozen-protocol verification."""
    for path in (FROZEN_MD, FROZEN_JSON, PREFLIGHT_COMPLETED, PREFLIGHT_RAW, CAPABILITY_COMPLETED, CAPABILITY_RAW):
        if not path.exists():
            raise ContractError("Required frozen input missing: %s" % rel(path))
    pre = base.verify_completed_marker(PREFLIGHT_COMPLETED)
    cap = base.verify_completed_marker(CAPABILITY_COMPLETED)
    if pre.get("hard_pass") is not True or pre.get("historical_validation64_bank_opened") is not False or pre.get("sealed_test_accessed") is not False:
        raise ContractError("V1 preflight access/readiness flags are invalid")
    if cap.get("hard_pass") is not True or cap.get("historical_validation64_bank_opened") is not False or cap.get("sealed_test_accessed") is not False:
        raise ContractError("Capability diagnostic access/readiness flags are invalid")
    protocol = read_json(FROZEN_JSON)
    if protocol.get("protocol_id") != "vehicle_fixed_h_opportunity_probe_v1_frozen_20260928":
        raise ContractError("Unexpected V1 frozen protocol id")
    if protocol.get("split") != "vehicle_fixed_h_opportunity_probe_v1_fresh64_select16_no_validation64_no_test":
        raise ContractError("Unexpected V1 frozen split")
    if list(protocol["rollout_design"]["horizons"]) != HORIZONS:
        raise ContractError("Frozen V1 horizon grid changed")
    if int(protocol["rollout_design"]["episodes_exact"]) != SELECTED_CASES * len(HORIZONS):
        raise ContractError("Frozen V1 episode count mismatch")
    if int(protocol["scenario_generator"]["candidate_pool_resets"]) != CANDIDATE_RESETS:
        raise ContractError("Frozen V1 candidate-pool size mismatch")
    if int(protocol["scenario_generator"]["selected_case_count"]) != SELECTED_CASES:
        raise ContractError("Frozen V1 selected-case count mismatch")
    term = protocol.get("terminal_grid_readiness") or {}
    if term.get("available_all_required") is not True:
        raise ContractError("Frozen V1 terminal readiness is not true")
    return {
        "protocol": protocol,
        "hashes": {rel(p): sha256(p) for p in (FROZEN_MD, FROZEN_JSON, PREFLIGHT_COMPLETED, PREFLIGHT_RAW, CAPABILITY_COMPLETED, CAPABILITY_RAW)},
    }


def write_backup_request(raw: Mapping[str, Any]) -> str:
    stamp = raw["created_utc"].replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_FIXED_H_OPPORTUNITY_PROBE_V1_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup enlarged fixed-H opportunity probe V1 before any adaptive selector/training diagnostic or further validation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": raw["budget_actual"]["episodes"],
        "control_steps": raw["budget_actual"]["control_steps"],
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(FROZEN_MD), rel(FROZEN_JSON)],
    })
    return rel(path)


def write_summary(raw: Mapping[str, Any]) -> None:
    opp = raw["opportunity_analysis"]
    thresholds = (raw.get("protocol_full") or {}).get("materiality_thresholds", {})
    lines = [
        "# Vehicle fixed-H opportunity probe V1",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Development diagnostic only. Enlarged fresh source-supported vehicle bank; fixed-H grid only; no adaptive policy, no training/refit, no historical validation64-bank access, no sealed-test access.",
        "",
        "## Access and budget",
        "",
        f"- Backup proof: `{raw['backup_proof']['path']}` commit `{raw['backup_proof']['commit']}`.",
        f"- Candidate-bank resets: `{raw['budget_actual']['fresh_bank_candidate_resets']}` / declared `{raw['budget_declared']['fresh_bank_candidate_resets']}`.",
        f"- Rollout episodes: `{raw['budget_actual']['episodes']}` / declared `{raw['budget_declared']['rollout_episodes_exact']}`.",
        f"- Control steps: `{raw['budget_actual']['control_steps']}` / upper bound `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- historical_validation64_bank_opened: `{raw['historical_validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Frozen materiality thresholds",
        "",
        "```json",
        json.dumps(thresholds, indent=2, sort_keys=True),
        "```",
        "",
        "## Selected source-supported cases",
        "",
        "| selected case | source candidate | stratum | theta_r | traj_steps | min reference obstacle clearance |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for i, meta in enumerate(raw["bank_selection"]["selected_metadata"]):
        lines.append("| %d | %d | `%s` | %.6g | %d | %.6g |" % (
            i,
            int(meta["candidate_index"]),
            meta.get("stratum"),
            float(meta.get("theta_r", float("nan"))),
            int(meta.get("traj_steps", 0)),
            float(meta.get("min_reference_obstacle_clearance", float("nan"))),
        ))
    lines += [
        "",
        "## Aggregate fixed-H grid",
        "",
        "| H | episodes | success | constraints | init-fail steps | final-fail steps | physical+constraint | total | decision mean s/step | decision total s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in HORIZONS:
        a = opp["by_horizon"][str(h)]
        lines.append("| %d | %d | %d | %d | %d | %d | %.6g | %.6g | %.6g | %.6g |" % (
            h,
            a["episodes"],
            a["success_count"],
            a["constraint_count"],
            a["initial_failed_steps"],
            a["solver_failure_steps"],
            a["physical_constraint_cost_sum"],
            a["total_cost_sum"],
            a["decision_mean_s_per_step"],
            a["decision_total_s"],
        ))
    lines += [
        "",
        "## Case/stratum opportunity map",
        "",
        "| case | stratum | best physical H | best total H | fastest H | nondominated H set |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for case_id in sorted(opp["by_case"], key=lambda x: int(x)):
        row = opp["by_case"][case_id]
        lines.append("| %s | `%s` | %d | %d | %d | `%s` |" % (
            case_id,
            row["stratum"],
            row["best_physical_horizon"],
            row["best_total_horizon"],
            row["fastest_horizon"],
            row["nondominated_horizons_cost_vs_decision_total_s"],
        ))
    lines += [
        "",
        "## Development interpretation",
        "",
        f"- Safe horizons succeeding without constraints on all selected cases: `{opp['safe_horizons_success_no_constraint_all_cases']}`.",
        f"- Overall strongest total-cost H: `{opp['overall_strongest_total_horizon']}`; strongest physical-cost H: `{opp['overall_strongest_physical_horizon']}`; fastest safe H: `{opp['overall_fastest_safe_horizon']}`.",
        f"- Distinct per-case best physical H: `{opp['distinct_case_best_physical_horizons']}`; distinct fastest H: `{opp['distinct_case_fastest_horizons']}`.",
        f"- protocol_opportunity_flag_development_only: `{opp['protocol_opportunity_flag_development_only']}`; weak_single_fixed_H_pattern_flag_development_only: `{opp['weak_single_fixed_H_pattern_flag_development_only']}`.",
        "",
        "A separate metadata postdiagnostic should now apply the frozen V1 materiality rules before any adaptive selector/refit is launched.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    opp = raw["opportunity_analysis"]
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle fixed-H opportunity probe V1\n\n"
        f"UTC: {raw['created_utc']}. Enlarged fresh development-only fixed-H opportunity probe completed: "
        f"{raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, "
        f"candidate resets={raw['budget_actual']['fresh_bank_candidate_resets']}. "
        f"No validation64 or sealed-test access. Opportunity flag={opp['protocol_opportunity_flag_development_only']}; "
        f"weak single-H pattern flag={opp['weak_single_fixed_H_pattern_flag_development_only']}; "
        f"safe horizons={opp['safe_horizons_success_no_constraint_all_cases']}; strongest total H={opp['overall_strongest_total_horizon']}. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def source_hashes() -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(base.smoke_base.__file__).resolve(),
        Path(base.v1.__file__).resolve(),
        FROZEN_MD,
        FROZEN_JSON,
        PREFLIGHT_COMPLETED,
        PREFLIGHT_RAW,
        CAPABILITY_COMPLETED,
        CAPABILITY_RAW,
        ROOT / "experiments/bohn2021_reproduction/conservative_canonical_reset.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_solver_recovery.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_run.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_audit.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_search.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_timing.py",
        ROOT / "experiments/bohn2021_reproduction/run.py",
        ROOT / "experiments/bohn2021_reproduction/runtime.py",
    ]
    return {rel(p): sha256(p) for p in paths if p.exists()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup-proof", required=True, type=Path, help="verified external backup proof after the V1 preflight and runner source are backed up")
    ap.add_argument("--i-accept-fresh-development-rollout", action="store_true", help="explicit acknowledgement: V1 development diagnostic only, no validation64/test access")
    args = ap.parse_args(argv)
    if not args.i_accept_fresh_development_rollout:
        raise ContractError("Explicit --i-accept-fresh-development-rollout is required")
    base.assert_fresh_output_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "vehicle_fixed_h_opportunity_probe_v1_regime_map_fixed_H_grid_development_diagnostic",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "training_gradient_steps": 0,
    })
    protocol_inputs = verify_protocol_inputs()
    backup = base.verify_backup_proof(args.backup_proof)
    preflight = base.runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError(preflight["diagnosis"] + " " + preflight.get("exception", ""))
    base.v1.latency_verify()
    bank_info = base.generate_bank_if_needed()
    bank = read_json(BANK_PATH)
    selected_cases = bank["selected_cases"]
    selected_meta = bank["selection"]["selected_metadata"]
    terminals, terminal_receipts = base.load_terminal_grid(protocol_inputs["protocol"])
    write_json(OUT_DIR / "terminal_sources.json", terminal_receipts)
    schedule = base.randomized_schedule(len(selected_cases))
    write_json(OUT_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "horizons": HORIZONS})
    episodes = []
    for item in schedule:
        case_id = int(item["selected_case_index"])
        h = int(item["horizon"])
        summary = base.run_episode(item, selected_cases[case_id], selected_meta[case_id], terminals[h], terminal_receipts[str(h)])
        episodes.append(summary)
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(int(e["steps"]) for e in episodes)),
            "last_episode": {k: summary[k] for k in ("execution_index", "case", "horizon", "steps", "success", "termination")},
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e["steps"]) for e in episodes))
    declared = protocol_inputs["protocol"]["budget"]
    if len(episodes) != declared["rollout_episodes_exact"] or control_steps > declared["control_step_upper_bound"]:
        raise ContractError("V1 fixed-H opportunity probe budget violation")
    analysis = base.analyze_opportunity(episodes, selected_meta)
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "vehicle_fixed_h_opportunity_probe_v1_regime_map_fixed_H_grid_development_diagnostic_not_model_selection_not_final_test",
        "classification": "diagnostic_development_fixed_H_opportunity_probe_v1",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "split": protocol_inputs["protocol"]["split"],
        "backup_proof": backup,
        "protocol": {"path": rel(FROZEN_MD), "sha256": sha256(FROZEN_MD), "json_path": rel(FROZEN_JSON), "json_sha256": sha256(FROZEN_JSON)},
        "protocol_full": protocol_inputs["protocol"],
        "protocol_inputs": protocol_inputs["hashes"],
        "bank": bank_info,
        "bank_selection": bank["selection"],
        "source_hashes": source_hashes(),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": declared,
        "budget_actual": {
            "fresh_bank_candidate_resets": CANDIDATE_RESETS if bank_info.get("created_now") else 0,
            "episodes": len(episodes),
            "control_steps": control_steps,
            "environment_constructions": len(episodes),
            "episode_resets": int(sum(int(e["resets_metered"]) for e in episodes)),
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "historical_validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "schedule": schedule,
        "terminal_sources": terminal_receipts,
        "episodes": episodes,
        "opportunity_analysis": analysis,
        "interpretation_limits": [
            "development diagnostic only",
            "fixed-H grid only",
            "not adaptive model selection",
            "not ORIGINAL SAC",
            "not final test",
            "actual timing from this AWS run only",
            "scenario bank generated from source-supported straight-line vehicle factors",
            "V1 enlarged bank tests V0 small-bank artifact hypothesis before selector/refit",
        ],
    }
    raw["backup_request"] = write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    state_text = (
        f"# Vehicle fixed-H opportunity probe V1 state ({raw['created_utc']})\n\n"
        f"Completed {len(episodes)} fixed-H episodes / {control_steps} control steps on enlarged fresh development bank. "
        f"No validation64 or sealed-test access. Opportunity flag={analysis['protocol_opportunity_flag_development_only']}; "
        f"weak single-H pattern={analysis['weak_single_fixed_H_pattern_flag_development_only']}. "
        "Backup required before postdiagnostic simulation or method-selection work.\n"
    )
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(state_text, encoding="utf-8")
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [ROOT / raw["backup_request"], STATE_PATH, Path(__file__).resolve(), FROZEN_MD, FROZEN_JSON, PREFLIGHT_COMPLETED, CAPABILITY_COMPLETED]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "backup_request": raw["backup_request"],
        "headline": {
            "opportunity_flag_development_only": analysis["protocol_opportunity_flag_development_only"],
            "weak_single_fixed_H_pattern_flag_development_only": analysis["weak_single_fixed_H_pattern_flag_development_only"],
            "overall_strongest_total_horizon": analysis["overall_strongest_total_horizon"],
            "safe_horizons": analysis["safe_horizons_success_no_constraint_all_cases"],
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "episodes": len(episodes),
        "control_steps": control_steps,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "opportunity_flag_development_only": analysis["protocol_opportunity_flag_development_only"],
        "weak_single_fixed_H_pattern_flag_development_only": analysis["weak_single_fixed_H_pattern_flag_development_only"],
        "backup_request": raw["backup_request"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(OUT_DIR / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "next_recovery_hint": "Preserve this partial directory, audit failure, and create a versioned one-variable repair. Use legacy interpreter and a verified backup proof.",
        })
        raise
