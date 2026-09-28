#!/usr/bin/env python3
"""Vehicle V1 state-continuation selector broad development confirmation v0.

Development-only online confirmation frozen by
research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_frozen_20260928.*

This runner reuses the frozen nearest-positive-state H30 latch selector from the
v0b smoke, and compares it against fixed H15/H30/H10 on all 16 V1 development
cases.  It performs no training or gradient updates, opens no historical
validation64 bank, and never accesses the sealed final test.

A verified external backup proof after the shadow scan/protocol/this runner
source is required before any rollout simulation is allowed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_v1_state_continuation_selector_smoke_v0b_runner as smoke  # noqa:E402

OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_20260928"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_20260928.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_frozen_20260928.json"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_frozen_20260928.md"
SHADOW_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_selector_shadow_scan_v0_20260928T1310Z/raw.json"
SHADOW_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_selector_shadow_scan_v0_20260928T1310Z/completed.json"
SMOKE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_smoke_v0b_20260928/raw.json"
SMOKE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_smoke_v0b_20260928/completed.json"
V1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/raw.json"
V1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/completed.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
ORDER_SEED = 2609287501
EXPECTED_CASES = list(range(16))
EXPECTED_ARMS = ["selector_latch_H30", "fixed_H15", "fixed_H30", "fixed_H10"]
EXPECTED_SHADOW_TRIGGER_CASES = [7, 10]
MARKER = "vehicle-v1-state-continuation-selector-broad-dev-confirmation-v0-20260928"

# Reuse the smoke runner's environment, metering and rollout mechanics but patch
# its global paths so helper functions write into this protocol's own directory.
smoke.OUT_DIR = OUT_DIR
smoke.STATE_PATH = STATE_PATH
smoke.PROTOCOL_JSON = PROTOCOL_JSON
smoke.PROTOCOL_MD = PROTOCOL_MD
smoke.ORDER_SEED = ORDER_SEED
smoke.MARKER = MARKER

ContractError = smoke.ContractError
rel = smoke.rel
read_json = smoke.read_json
write_json = smoke.write_json
sha256 = smoke.sha256
parse_time = smoke.parse_time
serial = smoke.serial


def assert_fresh_output_dir() -> None:
    if not OUT_DIR.exists():
        return
    completed = OUT_DIR / "completed.json"
    if completed.exists():
        smoke.verify_completed_marker(completed)
        raise SystemExit("broad development confirmation already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    if leftovers:
        raise ContractError("partial broad-confirmation output exists; inspect/recover first: " + ", ".join(rel(p) for p in leftovers[:20]))


def latest_time(*values: Optional[dt.datetime]) -> Optional[dt.datetime]:
    vals = [v.astimezone(dt.timezone.utc) for v in values if v is not None]
    return max(vals) if vals else None


def source_mtime_utc() -> dt.datetime:
    return dt.datetime.fromtimestamp(Path(__file__).resolve().stat().st_mtime, dt.timezone.utc)


def verify_broad_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for path in (PROTOCOL_JSON, PROTOCOL_MD, SHADOW_RAW, SHADOW_DONE, SMOKE_RAW, SMOKE_DONE, V1_RAW, V1_DONE):
        if not path.exists():
            raise ContractError("required input missing: %s" % rel(path))
    protocol = read_json(PROTOCOL_JSON)
    if protocol.get("protocol_id") != "vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_frozen_20260928":
        raise ContractError("unexpected broad-confirmation protocol id")
    if protocol.get("classification") != "development_IMPROVED_selector_broad_confirmation_not_validation_not_final_test":
        raise ContractError("unexpected broad-confirmation classification")
    access = protocol.get("access_rules") or {}
    if access.get("development_only") is not True or access.get("historical_validation64_bank_opened") is not False or access.get("sealed_test_accessed") is not False:
        raise ContractError("broad-confirmation access flags are invalid")
    if access.get("requires_verified_external_backup_after_shadow_scan_before_rollout") is not True:
        raise ContractError("broad-confirmation protocol must require verified backup before rollout")
    budget = protocol.get("split_and_budget") or {}
    if [int(x) for x in (budget.get("rollout_cases") or [])] != EXPECTED_CASES:
        raise ContractError("broad-confirmation rollout cases changed")
    if list(budget.get("arms") or []) != EXPECTED_ARMS:
        raise ContractError("broad-confirmation arms changed")
    if int(budget.get("max_episodes", -1)) != 64 or int(budget.get("control_step_upper_bound", -1)) != 9600:
        raise ContractError("unexpected broad-confirmation rollout budget")
    if int(budget.get("new_training_episodes", -1)) != 0 or int(budget.get("new_gradient_steps", -1)) != 0:
        raise ContractError("broad-confirmation unexpectedly permits training")
    if int(budget.get("historical_validation64_episodes", -1)) != 0 or int(budget.get("sealed_test_episodes", -1)) != 0:
        raise ContractError("broad-confirmation unexpectedly permits validation64/test episodes")
    selector = protocol.get("selector") or {}
    if int(selector.get("default_horizon", -1)) != 15 or int(selector.get("trigger_horizon", -1)) != 30:
        raise ContractError("unexpected selector horizons")
    if int(selector.get("training_episodes", -1)) != 0 or int(selector.get("gradient_updates", -1)) != 0:
        raise ContractError("selector must be zero-training/zero-gradient for this confirmation")
    shadow_done = smoke.verify_completed_marker(SHADOW_DONE)
    smoke_done = smoke.verify_completed_marker(SMOKE_DONE)
    v1_done = smoke.verify_completed_marker(V1_DONE, check_hashes=False)
    shadow = read_json(SHADOW_RAW)
    smoke_raw = read_json(SMOKE_RAW)
    v1_raw = read_json(V1_RAW)
    for label, obj in (("shadow_done", shadow_done), ("smoke_done", smoke_done), ("v1_done", v1_done), ("shadow_raw", shadow), ("smoke_raw", smoke_raw), ("v1_raw", v1_raw)):
        if obj.get("historical_validation64_bank_opened") not in (False, None):
            raise ContractError(label + " unexpectedly opened validation64")
        if obj.get("sealed_test_accessed") not in (False, None):
            raise ContractError(label + " unexpectedly accessed sealed test")
    if shadow_done.get("hard_pass") is not True or shadow_done.get("headline", {}).get("shadow_gate_pass") is not True:
        raise ContractError("shadow scan gate did not pass")
    if [int(x) for x in shadow.get("triggered_cases", [])] != EXPECTED_SHADOW_TRIGGER_CASES:
        raise ContractError("shadow trigger cases changed or were not sparse positives")
    if list(shadow.get("extra_trigger_cases") or []) != []:
        raise ContractError("shadow scan reported extra trigger cases")
    if smoke_done.get("headline", {}).get("selector_smoke_expansion_gate_pass") is not True:
        raise ContractError("selector-smoke expansion gate did not pass")
    return protocol, shadow, smoke_raw, v1_raw


def verify_backup_after_source(args_backup_proof: Path, protocol: Mapping[str, Any], shadow: Mapping[str, Any]) -> Dict[str, Any]:
    min_time = latest_time(parse_time(protocol.get("created_utc")), parse_time(shadow.get("created_utc")), source_mtime_utc())
    return smoke.verify_backup_proof(args_backup_proof, min_time)


def by_case_rows(episodes: Sequence[Mapping[str, Any]], case: int) -> Dict[str, Mapping[str, Any]]:
    rows = {str(e["arm_id"]): e for e in episodes if int(e["case"]) == int(case)}
    if set(rows) != set(EXPECTED_ARMS):
        raise ContractError("missing arm(s) for case %d: %s" % (case, sorted(rows)))
    return rows


def safe_bool(value: Any) -> bool:
    return bool(value)


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def safe_float(value: Any) -> float:
    try:
        out = float(value or 0.0)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return 0.0


def analyze_broad(episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], shadow: Mapping[str, Any]) -> Dict[str, Any]:
    by_arm = {arm: smoke.aggregate([e for e in episodes if str(e.get("arm_id")) == arm]) for arm in EXPECTED_ARMS}
    per_case: Dict[str, Any] = {}
    failure_where_h15_succeeds: List[int] = []
    selector_constraint_cases: List[int] = []
    solver_increase_cases: List[int] = []
    initial_increase_cases: List[int] = []
    triggered_cases: List[int] = []
    trigger_steps: Dict[str, Optional[int]] = {}
    for case in EXPECTED_CASES:
        rows = by_case_rows(episodes, case)
        sel = rows["selector_latch_H30"]
        h15 = rows["fixed_H15"]
        h30 = rows["fixed_H30"]
        h10 = rows["fixed_H10"]
        if safe_bool(sel.get("selector_triggered")):
            triggered_cases.append(case)
        trigger_steps[str(case)] = None if sel.get("selector_first_trigger_step") is None else int(sel.get("selector_first_trigger_step"))
        if safe_bool(h15.get("success")) and not safe_bool(sel.get("success")):
            failure_where_h15_succeeds.append(case)
        if safe_bool(sel.get("constraint")):
            selector_constraint_cases.append(case)
        if safe_int(sel.get("solver_failure_steps")) > safe_int(h15.get("solver_failure_steps")):
            solver_increase_cases.append(case)
        if safe_int(sel.get("initial_failed_steps")) > safe_int(h15.get("initial_failed_steps")):
            initial_increase_cases.append(case)
        phys_delta_h15 = safe_float(sel.get("physical_constraint_cost")) - safe_float(h15.get("physical_constraint_cost"))
        total_delta_h15 = safe_float(sel.get("total_cost")) - safe_float(h15.get("total_cost"))
        per_case[str(case)] = {
            "selector_triggered": safe_bool(sel.get("selector_triggered")),
            "selector_first_trigger_step": trigger_steps[str(case)],
            "selector_horizon_counts": sel.get("horizon_counts"),
            "selector_vs_H15_physical_delta": phys_delta_h15,
            "selector_vs_H15_total_delta": total_delta_h15,
            "selector_vs_H30_physical_delta": safe_float(sel.get("physical_constraint_cost")) - safe_float(h30.get("physical_constraint_cost")),
            "selector_vs_H30_total_delta": safe_float(sel.get("total_cost")) - safe_float(h30.get("total_cost")),
            "selector_vs_H10_physical_delta": safe_float(sel.get("physical_constraint_cost")) - safe_float(h10.get("physical_constraint_cost")),
            "selector_vs_H10_total_delta": safe_float(sel.get("total_cost")) - safe_float(h10.get("total_cost")),
            "selector_vs_H15_decision_time_ratio": (safe_float(sel.get("decision_timing_s", {}).get("sum")) / safe_float(h15.get("decision_timing_s", {}).get("sum"))) if safe_float(h15.get("decision_timing_s", {}).get("sum")) > 0 else None,
            "success": {arm: safe_bool(row.get("success")) for arm, row in rows.items()},
            "termination": {arm: row.get("termination") for arm, row in rows.items()},
            "constraint": {arm: safe_bool(row.get("constraint")) for arm, row in rows.items()},
            "initial_failed_steps": {arm: safe_int(row.get("initial_failed_steps")) for arm, row in rows.items()},
            "solver_failure_steps": {arm: safe_int(row.get("solver_failure_steps")) for arm, row in rows.items()},
            "costs": {arm: {"physical_constraint_cost": safe_float(row.get("physical_constraint_cost")), "total_cost": safe_float(row.get("total_cost"))} for arm, row in rows.items()},
            "timing_sums": {arm: {"decision": safe_float(row.get("decision_timing_s", {}).get("sum")), "decision_plus_terminal_switch": safe_float(row.get("decision_plus_terminal_switch_timing_s", {}).get("sum")), "solver_attempt": safe_float(row.get("solver_attempt_timing_s", {}).get("sum")), "selection": safe_float(row.get("selection_timing_s", {}).get("sum")), "terminal_switch": safe_float(row.get("terminal_switch_timing_s", {}).get("sum"))} for arm, row in rows.items()},
        }
    triggered_cases = sorted(triggered_cases)
    expected_triggers = [int(x) for x in shadow.get("triggered_cases", EXPECTED_SHADOW_TRIGGER_CASES)]
    extra_trigger_cases = [c for c in triggered_cases if c not in expected_triggers]
    missing_expected_trigger_cases = [c for c in expected_triggers if c not in triggered_cases]
    safety_pass = bool(
        by_arm["selector_latch_H30"]["success_count"] == len(EXPECTED_CASES)
        and by_arm["selector_latch_H30"]["constraint_count"] == 0
        and not failure_where_h15_succeeds
        and not solver_increase_cases
    )
    secondary_initial_no_increase = bool(not initial_increase_cases)
    benefit_physical_abs_vs_H15 = safe_float(by_arm["fixed_H15"]["physical_constraint_cost_sum"]) - safe_float(by_arm["selector_latch_H30"]["physical_constraint_cost_sum"])
    benefit_total_abs_vs_H15 = safe_float(by_arm["fixed_H15"]["total_cost_sum"]) - safe_float(by_arm["selector_latch_H30"]["total_cost_sum"])
    benefit_pass = bool(benefit_physical_abs_vs_H15 >= 20.0 and benefit_total_abs_vs_H15 >= 20.0)
    trigger_consistency_pass = bool(not extra_trigger_cases and not missing_expected_trigger_cases)
    protocol_acceptance_pass = bool(safety_pass and benefit_pass)
    return {
        "by_arm": by_arm,
        "per_case": per_case,
        "triggered_cases": triggered_cases,
        "expected_trigger_cases_from_shadow": expected_triggers,
        "extra_trigger_cases": extra_trigger_cases,
        "missing_expected_trigger_cases": missing_expected_trigger_cases,
        "trigger_consistency_pass_secondary": trigger_consistency_pass,
        "failure_where_H15_succeeds_cases": failure_where_h15_succeeds,
        "selector_constraint_cases": selector_constraint_cases,
        "solver_failure_step_increase_cases_vs_H15": solver_increase_cases,
        "initial_failed_step_increase_cases_vs_H15_secondary": initial_increase_cases,
        "safety_pass_protocol": safety_pass,
        "secondary_initial_failed_step_no_increase": secondary_initial_no_increase,
        "benefit_physical_abs_vs_H15": benefit_physical_abs_vs_H15,
        "benefit_total_abs_vs_H15": benefit_total_abs_vs_H15,
        "benefit_pass_protocol": benefit_pass,
        "protocol_acceptance_pass": protocol_acceptance_pass,
        "secondary_acceptance_with_trigger_consistency_and_initial": bool(protocol_acceptance_pass and trigger_consistency_pass and secondary_initial_no_increase),
        "aggregate_deltas": {
            "selector_minus_H15_physical": safe_float(by_arm["selector_latch_H30"]["physical_constraint_cost_sum"]) - safe_float(by_arm["fixed_H15"]["physical_constraint_cost_sum"]),
            "selector_minus_H15_total": safe_float(by_arm["selector_latch_H30"]["total_cost_sum"]) - safe_float(by_arm["fixed_H15"]["total_cost_sum"]),
            "selector_minus_H30_physical": safe_float(by_arm["selector_latch_H30"]["physical_constraint_cost_sum"]) - safe_float(by_arm["fixed_H30"]["physical_constraint_cost_sum"]),
            "selector_minus_H30_total": safe_float(by_arm["selector_latch_H30"]["total_cost_sum"]) - safe_float(by_arm["fixed_H30"]["total_cost_sum"]),
            "selector_minus_H10_physical": safe_float(by_arm["selector_latch_H30"]["physical_constraint_cost_sum"]) - safe_float(by_arm["fixed_H10"]["physical_constraint_cost_sum"]),
            "selector_minus_H10_total": safe_float(by_arm["selector_latch_H30"]["total_cost_sum"]) - safe_float(by_arm["fixed_H10"]["total_cost_sum"]),
        },
        "acceptance_rule_frozen": protocol.get("acceptance_for_next_refit_or_training"),
        "interpretation": "Development confirmation only. Passing supports a broader versioned refit/training design; it does not validate the final method and is not ORIGINAL SAC evidence.",
    }


def write_backup_request(raw: Mapping[str, Any]) -> str:
    stamp = str(raw["created_utc"]).replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_V1_STATE_CONTINUATION_SELECTOR_BROAD_DEV_CONFIRMATION_V0_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup broad development selector confirmation before any refit/training or validation work",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": raw["budget_actual"]["episodes"],
        "control_steps": raw["budget_actual"]["control_steps"],
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(PROTOCOL_MD), rel(PROTOCOL_JSON)],
    })
    return rel(path)


def write_summary(raw: Mapping[str, Any]) -> None:
    analysis = raw["analysis"]
    lines = [
        "# Vehicle V1 state-continuation selector broad development confirmation v0",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Development-only IMPROVED selector confirmation: frozen nearest-state H30 latch compared with fixed H15/H30/H10 on all 16 V1 development cases. No training/gradient updates, no validation64-bank access, no sealed-test access.",
        "",
        "## Access and budget",
        "",
        f"- Backup proof: `{raw['backup_proof']['path']}` commit `{raw['backup_proof']['commit']}`.",
        f"- Episodes: `{raw['budget_actual']['episodes']}` / frozen max `{raw['budget_declared']['max_episodes']}`.",
        f"- Control steps: `{raw['budget_actual']['control_steps']}` / upper bound `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- historical_validation64_bank_opened: `{raw['historical_validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Arm aggregates",
        "",
        "| arm | episodes | steps | success | constraints | init-fail steps | solver-fail steps | physical+constraint | total | decision mean s/step | decision+switch mean s/step | solver total s | selection total s | terminal switch total s | horizons | selector triggered eps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for arm in EXPECTED_ARMS:
        a = analysis["by_arm"][arm]
        lines.append("| `%s` | %d | %d | %d | %d | %d | %d | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g | `%s` | %d |" % (
            arm,
            a["episodes"],
            a["steps"],
            a["success_count"],
            a["constraint_count"],
            a["initial_failed_steps"],
            a["solver_failure_steps"],
            a["physical_constraint_cost_sum"],
            a["total_cost_sum"],
            a["decision_mean_s_per_step"],
            a["decision_plus_terminal_switch_mean_s_per_step"],
            a["solver_attempt_total_s"],
            a["selection_total_s"],
            a["terminal_switch_total_s"],
            a["horizon_counts"],
            a["selector_triggered_episodes"],
        ))
    lines += [
        "",
        "## Frozen acceptance result",
        "",
        f"- safety_pass_protocol: `{analysis['safety_pass_protocol']}`; benefit_pass_protocol: `{analysis['benefit_pass_protocol']}`; protocol_acceptance_pass: `{analysis['protocol_acceptance_pass']}`.",
        f"- Benefit vs H15: physical `{analysis['benefit_physical_abs_vs_H15']:.6g}` absolute units; total `{analysis['benefit_total_abs_vs_H15']:.6g}` absolute units (threshold 20 each).",
        f"- Triggered cases: `{analysis['triggered_cases']}`; expected from shadow: `{analysis['expected_trigger_cases_from_shadow']}`; extra: `{analysis['extra_trigger_cases']}`; missing expected: `{analysis['missing_expected_trigger_cases']}`.",
        f"- Secondary initial-failed-step no-increase: `{analysis['secondary_initial_failed_step_no_increase']}`; trigger consistency: `{analysis['trigger_consistency_pass_secondary']}`.",
        "",
        "## Aggregate deltas (selector minus comparator; negative is better for cost)",
        "",
        "```json",
        json.dumps(analysis["aggregate_deltas"], indent=2, sort_keys=True),
        "```",
        "",
        "## Per-case selector deltas",
        "",
        "| case | triggered | first trigger | H counts | selector-H15 physical | selector-H15 total | selector-H30 physical | selector-H30 total | safety flags |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for case in EXPECTED_CASES:
        row = analysis["per_case"][str(case)]
        safety_flags = {
            "success": row["success"],
            "constraint": row["constraint"],
            "initial_failed_steps": row["initial_failed_steps"],
            "solver_failure_steps": row["solver_failure_steps"],
        }
        lines.append("| %d | `%s` | %s | `%s` | %.6g | %.6g | %.6g | %.6g | `%s` |" % (
            case,
            str(row["selector_triggered"]),
            str(row["selector_first_trigger_step"]),
            row["selector_horizon_counts"],
            row["selector_vs_H15_physical_delta"],
            row["selector_vs_H15_total_delta"],
            row["selector_vs_H30_physical_delta"],
            row["selector_vs_H30_total_delta"],
            safety_flags,
        ))
    lines += [
        "",
        "## Evidence update",
        "",
        "| axis | verified finding from this run | remaining uncertainty | next discriminating experiment if gate passes/fails |",
        "|---|---|---|---|",
        "| scenarios | Online all-16 results test whether sparse continuation-labelled H30 opportunities survive closed-loop selector execution on the V1 development bank. | V1 remains a development bank derived after earlier diagnostics; generalization to fresh confirmation scenarios is unknown. | If pass, freeze a fresh refit/training/confirmation split; if fail, localize failing cases with controlled continuation from the observed trigger or near-trigger states. |",
        "| reward/objective | Physical+constraint and total-cost acceptance use measured episode outcomes, not only h-penalty. | Terminal-value and objective mismatch may still distort broader training. | If pass, design a value/refit objective using continuation labels and matched fixed-H baselines; if fail, inspect objective/terminal mismatches in failed cases. |",
        "| training/selection | This is a zero-gradient state-label selector, not ORIGINAL SAC; success would implicate historical training/selection/policy-class failure more than absence of opportunity. | Nearest-state latch may not scale or may overfit sparse prototypes. | Refit a compact classifier/regressor or finite-search policy on independently generated continuation labels across >=3 seeds. |",
        "| comparisons | Fixed H15/H30/H10 are rerun as paired online baselines with actual timing on the same cases. | Broader H grid and independent terminal tuning still matter for formal claims. | Future validation must include strong fixed-H grid and matched/independent terminal baselines. |",
        "",
        f"Backup request before further simulations: `{raw['backup_request']}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    analysis = raw["analysis"]
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 state-continuation selector broad development confirmation v0\n\n"
        f"UTC: {raw['created_utc']}. Development-only broad selector confirmation completed: "
        f"{raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, no validation64/test access. "
        f"Protocol acceptance={analysis['protocol_acceptance_pass']}; safety={analysis['safety_pass_protocol']}; "
        f"benefit_vs_H15=(physical {analysis['benefit_physical_abs_vs_H15']:.6g}, total {analysis['benefit_total_abs_vs_H15']:.6g}); "
        f"triggered_cases={analysis['triggered_cases']}. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
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
        Path(smoke.__file__).resolve(),
        Path(smoke.v1probe.__file__).resolve(),
        Path(smoke.base.__file__).resolve(),
        Path(smoke.base.smoke_base.__file__).resolve(),
        Path(smoke.base.v1.__file__).resolve(),
        PROTOCOL_JSON,
        PROTOCOL_MD,
        SHADOW_RAW,
        SHADOW_DONE,
        SMOKE_RAW,
        SMOKE_DONE,
        V1_RAW,
        V1_DONE,
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
    ap.add_argument("--backup-proof", required=True, type=Path, help="verified external backup proof after shadow scan/protocol/this runner source")
    ap.add_argument("--i-accept-broad-development-confirmation-rollout", action="store_true", help="explicit acknowledgement: development diagnostic only, no validation64/test access")
    args = ap.parse_args(argv)
    if not args.i_accept_broad_development_confirmation_rollout:
        raise ContractError("explicit --i-accept-broad-development-confirmation-rollout is required")
    protocol, shadow, smoke_raw, v1_raw = verify_broad_inputs()
    backup = verify_backup_after_source(args.backup_proof, protocol, shadow)
    assert_fresh_output_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {"started_utc": started, "pid": os.getpid(), "method": "vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0", "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "training_gradient_steps": 0})
    preflight = smoke.base.runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError(str(preflight.get("diagnosis", "runtime preflight failed")) + " " + str(preflight.get("exception", "")))
    smoke.v1.latency_verify()
    bank_path = ROOT / str(v1_raw["bank"]["path"])
    bank_done = bank_path.parent / "completed.json"
    smoke.verify_completed_marker(bank_done)
    bank = read_json(bank_path)
    selected_cases = bank["selected_cases"]
    selected_meta = bank["selection"]["selected_metadata"]
    if len(selected_cases) <= max(EXPECTED_CASES):
        raise ContractError("protocol references unavailable V1 selected case")
    terminals, terminal_receipts = smoke.base.load_terminal_grid(v1_raw["protocol_full"])
    write_json(OUT_DIR / "terminal_sources.json", terminal_receipts)
    arms = smoke.build_arms(protocol)
    if [str(a["arm_id"]) for a in arms] != EXPECTED_ARMS:
        raise ContractError("build_arms returned unexpected arm order")
    schedule = smoke.randomized_schedule(EXPECTED_CASES, arms)
    write_json(OUT_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "arms": arms, "rollout_cases": EXPECTED_CASES})
    episodes: List[Dict[str, Any]] = []
    selector_protocol = protocol["selector"]
    arm_by_index = {i: arm for i, arm in enumerate(arms)}
    for item in schedule:
        arm = arm_by_index[int(item["arm_index"])]
        case_id = int(item["case"])
        summary = smoke.run_episode(item, arm, selected_cases[case_id], selected_meta[case_id], terminals, terminal_receipts, selector_protocol)
        episodes.append(summary)
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(int(e["steps"]) for e in episodes)),
            "last_episode": {k: summary[k] for k in ("execution_index", "case", "arm_id", "steps", "success", "termination")},
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    budget = protocol["split_and_budget"]
    if len(episodes) != int(budget["max_episodes"]) or control_steps > int(budget["control_step_upper_bound"]):
        raise ContractError("broad-confirmation budget violation")
    analysis = analyze_broad(episodes, protocol, shadow)
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_nearest_state_H30_latch",
        "classification": "development_IMPROVED_selector_broad_confirmation_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "backup_proof": backup,
        "protocol": {"path": rel(PROTOCOL_MD), "sha256": sha256(PROTOCOL_MD), "json_path": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)},
        "shadow_scan": {"raw": rel(SHADOW_RAW), "raw_sha256": sha256(SHADOW_RAW), "completed": rel(SHADOW_DONE), "completed_sha256": sha256(SHADOW_DONE), "headline": {"shadow_gate_pass": True, "triggered_cases": shadow.get("triggered_cases")}},
        "selector_smoke": {"raw": rel(SMOKE_RAW), "raw_sha256": sha256(SMOKE_RAW), "completed": rel(SMOKE_DONE), "completed_sha256": sha256(SMOKE_DONE), "headline": (smoke_raw.get("analysis") or {})},
        "v1_source": {"raw": rel(V1_RAW), "raw_sha256": sha256(V1_RAW), "completed": rel(V1_DONE), "completed_sha256": sha256(V1_DONE)},
        "bank": {"path": rel(bank_path), "sha256": sha256(bank_path)},
        "source_hashes": source_hashes(),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": {"max_episodes": int(budget["max_episodes"]), "control_step_upper_bound": int(budget["control_step_upper_bound"]), "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "environment_constructions": len(episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "schedule": schedule,
        "terminal_sources": terminal_receipts,
        "episodes": episodes,
        "analysis": analysis,
        "interpretation_limits": [
            "development confirmation only",
            "selector prototypes/radius derived from already-opened development controlled-continuation traces",
            "not validation or final-test evidence",
            "not ORIGINAL SAC and not gradient RL training",
            "actual timing from AWS run; selection/terminal-switch overhead reported separately",
            "passing should trigger a versioned broader refit/training protocol before formal validation",
        ],
    }
    raw["backup_request"] = write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 state-continuation selector broad development confirmation v0 ({raw['created_utc']})\n\n"
        f"Completed {len(episodes)} episodes / {control_steps} control steps. No validation64/test access. "
        f"Protocol acceptance={analysis['protocol_acceptance_pass']}; safety={analysis['safety_pass_protocol']}; "
        f"benefit_vs_H15=(physical {analysis['benefit_physical_abs_vs_H15']:.6g}, total {analysis['benefit_total_abs_vs_H15']:.6g}); "
        f"triggered_cases={analysis['triggered_cases']}. Backup required before further simulation/refit.\n",
        encoding="utf-8",
    )
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [ROOT / raw["backup_request"], STATE_PATH, Path(__file__).resolve(), PROTOCOL_MD, PROTOCOL_JSON, SHADOW_DONE, SHADOW_RAW, SMOKE_DONE, SMOKE_RAW]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_request": raw["backup_request"],
        "headline": {"protocol_acceptance_pass": analysis["protocol_acceptance_pass"], "safety_pass_protocol": analysis["safety_pass_protocol"], "benefit_pass_protocol": analysis["benefit_pass_protocol"], "benefit_physical_abs_vs_H15": analysis["benefit_physical_abs_vs_H15"], "benefit_total_abs_vs_H15": analysis["benefit_total_abs_vs_H15"], "triggered_cases": analysis["triggered_cases"]},
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({"completed": rel(OUT_DIR / "completed.json"), "summary": rel(OUT_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "protocol_acceptance_pass": analysis["protocol_acceptance_pass"], "safety_pass_protocol": analysis["safety_pass_protocol"], "benefit_pass_protocol": analysis["benefit_pass_protocol"], "benefit_physical_abs_vs_H15": analysis["benefit_physical_abs_vs_H15"], "benefit_total_abs_vs_H15": analysis["benefit_total_abs_vs_H15"], "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": raw["backup_request"]}, sort_keys=True), flush=True)
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
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "next_recovery_hint": "Preserve partial directory, audit failure, and do not broaden changes without a new hypothesis. Use legacy interpreter and a verified backup proof after the shadow scan/protocol/runner source.",
        })
        raise
