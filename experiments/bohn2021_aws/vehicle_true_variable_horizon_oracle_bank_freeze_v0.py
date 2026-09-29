#!/usr/bin/env python3
"""Freeze a larger development-only true-variable-H oracle bank protocol.

This metadata-only freezer follows the broader true-H postdiagnostic.  It does
not run rollouts, open validation64/final-test banks, train, refit, or reset any
candidate pool.  Its purpose is to turn the already inspected design into a
versioned, executable protocol for the next *simulation* runner, which remains
blocked until a verified external backup covers this freezer and its outputs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_oracle_bank_freeze_v0"
STAMP = "20260929T0725Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/vehicle_true_variable_horizon_oracle_bank_v0_frozen_{STAMP}.json"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_TRUE_VARIABLE_HORIZON_ORACLE_BANK_V0_RUN_{STAMP}.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

SOURCE = Path(__file__).resolve()
DESIGN_NOTE = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_oracle_bank_v0_design_20260929T0715Z.md"
POSTDIAG_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_broader_block_postdiagnostic_v0_20260929T0710Z/raw.json"
POSTDIAG_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_broader_block_postdiagnostic_v0_20260929T0710Z/completed.json"
BROADER_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_broader_block_v0_frozen_20260929T0650Z.json"
BROADER_RUN_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_broader_block_v0_run_20260929T0650Z/completed.json"
BROADER_RUN_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_broader_block_v0_run_20260929T0650Z/raw.json"
V1D_SELECTED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/selected_targets.json"
V1D_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/completed.json"
V1E_CASE5_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_run_20260929T0605Z/raw.json"
V1E_CASE5_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_run_20260929T0605Z/completed.json"

TRUE_HORIZONS = [10, 15, 25]
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
REPEATS = 2
MAX_PRIMARY_STATES = 16
MAX_BRANCH_STEPS = 150
RNG_SEED = 202609290725


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def si(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def sf(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def parse_completed(path: Path) -> Mapping[str, Any]:
    obj = read_json(path)
    if obj.get("validation64_bank_opened") is not False:
        raise SystemExit(f"validation64 flag not closed in {rel(path)}")
    if obj.get("sealed_test_accessed") is not False:
        raise SystemExit(f"sealed test flag not closed in {rel(path)}")
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise SystemExit(f"required completed marker did not pass: {rel(path)}")
    return obj


def state_dict_from_v1d(t: Mapping[str, Any]) -> Dict[str, float]:
    s = t.get("branch_previous_state_from_scan") or {}
    return {"x": sf(s.get("x")), "y": sf(s.get("y")), "theta": sf(s.get("theta"))}


def state_dict_from_broader(t: Mapping[str, Any]) -> Dict[str, float]:
    s = t.get("branch_previous_state") or {}
    return {"x": sf(s.get("x")), "y": sf(s.get("y")), "theta": sf(s.get("theta"))}


def target_from_v1d(t: Mapping[str, Any], order: int) -> Dict[str, Any]:
    role = str(t.get("target_role") or "v1d_trace_selected")
    return {
        "state_id": str(t.get("state_id")),
        "source_stratum": "stress_v1d_trace_selected_primary",
        "selection_order": order,
        "previously_used_in_true_h_broader_block": False,
        "development_exposed_state": True,
        "case": si(t.get("full_case_index")),
        "source_candidate_index": si(t.get("source_candidate_index")),
        "branch_step": si(t.get("branch_step")),
        "branch_previous_state": state_dict_from_v1d(t),
        "scan_step_horizon": si(t.get("scan_step_horizon")),
        "scan_trace_path": t.get("scan_trace_path"),
        "selection_group": t.get("selection_group"),
        "target_role": role,
        "window": t.get("window"),
        "score": sf(t.get("score")),
        "score_components": t.get("score_components") or {},
        "noncontrol_for_primary_gate": "control" not in role.lower(),
        "inclusion_reason": "all 12 preselected stress-v1d trace states are included by deterministic rule, controls retained",
    }


def target_from_broader(t: Mapping[str, Any], order: int) -> Dict[str, Any]:
    role = str(t.get("role") or t.get("case_role") or "broader_block_anchor")
    return {
        "state_id": str(t.get("state_id")),
        "source_stratum": "true_h_broader_block_anchor_previously_used",
        "selection_order": order,
        "previously_used_in_true_h_broader_block": True,
        "development_exposed_state": True,
        "case": si(t.get("case")),
        "source_candidate_index": si(t.get("source_candidate_index")),
        "branch_step": si(t.get("branch_step")),
        "branch_previous_state": state_dict_from_broader(t),
        "scan_step_horizon": None,
        "scan_trace_path": t.get("source_path") or t.get("source_record"),
        "selection_group": t.get("selection_group"),
        "target_role": role,
        "window": t.get("case_role"),
        "score": None,
        "score_components": {},
        "noncontrol_for_primary_gate": ("control" not in role.lower() and "negative" not in role.lower()),
        "inclusion_reason": "calibration/continuity anchor from completed true-H broader block; not fresh confirmation evidence",
    }


def make_schedule(targets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    schedule: List[Dict[str, Any]] = []
    execution_index = 0
    for repeat in range(REPEATS):
        for target in targets:
            for profile in TERMINAL_PROFILES:
                horizons = list(TRUE_HORIZONS)
                rng.shuffle(horizons)
                for h in horizons:
                    schedule.append({
                        "execution_index": execution_index,
                        "repeat": repeat,
                        "state_id": target["state_id"],
                        "case": target["case"],
                        "branch_step": target["branch_step"],
                        "true_mpc_n_horizon": h,
                        "terminal_profile": profile,
                        "source_stratum": target["source_stratum"],
                        "blocked_randomization_unit": f"repeat{repeat}|{target['state_id']}|{profile}",
                    })
                    execution_index += 1
    return schedule


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    marker = f"vehicle-true-variable-horizon-oracle-bank-freeze-v0-{STAMP}"
    block = f"""<!-- {marker} -->
## 2026-09-29 vehicle true variable-H oracle bank freeze v0

UTC: {raw['created_utc']}. Metadata-only/no-simulation protocol freeze completed for a larger source-supported true-H oracle-label/value-modeling bank. Planned development branch budget is {raw['planned_development_branch_episodes_exact']} episodes / {raw['planned_development_control_step_upper_bound']} control-step cap, horizons {TRUE_HORIZONS}, terminal profiles {TERMINAL_PROFILES}, repeats {REPEATS}, selected states {raw['selected_state_count']} ({raw['selected_noncontrol_count']} non-control for primary gate). Validation64 and sealed test remain closed; training/refit/candidate resets are zero. More simulation/training/refit is blocked until an external verified backup covers `{raw['backup_request']}` and this freezer output. Protocol: `{raw['protocol']}`.
"""
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / name, marker, block)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    if reg.exists():
        old = reg.read_text(encoding="utf-8", errors="replace")
        row = f"{raw['created_utc']},{NAME},metadata_no_simulation_oracle_bank_freeze,no_validation_no_test,0,0,0,0,0,False,{raw['completed']}\n"
        if NAME not in old[-50000:]:
            reg.write_text(old.rstrip() + "\n" + row, encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle true variable-H oracle bank freeze v0",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata-only protocol freeze; no simulations/training/refit, no validation64 bank, no sealed test.",
        "",
        "## Frozen next diagnostic",
        "",
        f"- Selected states: `{raw['selected_state_count']}` total; non-control for the primary gate: `{raw['selected_noncontrol_count']}`.",
        f"- Strata counts: `{raw['selected_strata_counts']}`.",
        f"- Planned arms: horizons `{TRUE_HORIZONS}`, terminal profiles `{TERMINAL_PROFILES}`, repeats `{REPEATS}`.",
        f"- Planned development branch episodes: `{raw['planned_development_branch_episodes_exact']}`; control-step cap: `{raw['planned_development_control_step_upper_bound']}`.",
        "- Labels after the future runner are fastest safe horizon within near-best physical tolerance over H10/H15/H25, using measured whole-decision time (not H alone).",
        "",
        "## Why this diagnostic",
        "",
        "The completed true-H broader block showed real measured compute savings and a mixed near-best oracle, but also large H10 physical regressions in middle/late case5 states. This frozen bank tests whether the oracle control/compute tradeoff survives across a larger, source-supported development state set before any selector/value-model refit.",
        "",
        "## Decision",
        "",
        "Run no simulations until a verified external backup covers this protocol and output. If the future bank has non-degenerate near-best labels and aggregate oracle-vs-H25 timing savings without physical/safety regression, proceed to an IMPROVED selector/value-modeling experiment; otherwise pivot to terminal-value/modeling/objective/scenario diagnostics rather than another sparse-label sweep.",
        "",
        f"Protocol: `{raw['protocol']}`. Backup request: `{raw['backup_request']}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--i-accept-no-simulation-oracle-bank-freeze-v0", action="store_true")
    args = parser.parse_args(argv)
    if not args.freeze or not args.i_accept_no_simulation_oracle_bank_freeze_v0:
        raise SystemExit("requires --freeze and explicit no-simulation oracle-bank acknowledgement")

    required = [DESIGN_NOTE, POSTDIAG_RAW, POSTDIAG_COMPLETED, BROADER_PROTOCOL, BROADER_RUN_COMPLETED, BROADER_RUN_RAW, V1D_SELECTED, V1D_COMPLETED, V1E_CASE5_RAW, V1E_CASE5_COMPLETED]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing required development inputs: " + ", ".join(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    created = now_utc()
    postdiag_done = parse_completed(POSTDIAG_COMPLETED)
    broader_done = parse_completed(BROADER_RUN_COMPLETED)
    v1d_done = parse_completed(V1D_COMPLETED)
    v1e_done = parse_completed(V1E_CASE5_COMPLETED)
    postdiag = read_json(POSTDIAG_RAW)
    broader_protocol = read_json(BROADER_PROTOCOL)
    v1d = read_json(V1D_SELECTED)

    if postdiag.get("near_best_oracle_useful_vs_fixed_H25_noncontrol") is not True:
        raise SystemExit("postdiagnostic did not support oracle-bank freeze")
    if (broader_done.get("headline") or {}).get("implementation_pass") is not True or (broader_done.get("headline") or {}).get("compute_pass") is not True:
        raise SystemExit("broader true-H implementation/compute gate did not pass")

    selected_targets: List[Dict[str, Any]] = []
    seen = set()
    for t in v1d.get("selected_targets") or []:
        if len(selected_targets) >= 12:
            break
        target = target_from_v1d(t, len(selected_targets))
        if target["state_id"] in seen:
            continue
        seen.add(target["state_id"])
        selected_targets.append(target)
    broad_targets = (broader_protocol.get("target_selection") or {}).get("targets") or []
    for t in broad_targets:
        if len(selected_targets) >= MAX_PRIMARY_STATES:
            break
        target = target_from_broader(t, len(selected_targets))
        if target["state_id"] in seen:
            continue
        seen.add(target["state_id"])
        selected_targets.append(target)

    excluded = []
    for t in broad_targets:
        sid = str(t.get("state_id"))
        if sid in seen and not any(x["state_id"] == sid and x["source_stratum"].startswith("true_h_broader") for x in selected_targets):
            excluded.append({"state_id": sid, "reason": "duplicate_or_cap"})
    if len(selected_targets) != MAX_PRIMARY_STATES:
        raise SystemExit(f"expected {MAX_PRIMARY_STATES} deterministic states but selected {len(selected_targets)}")

    schedule = make_schedule(selected_targets)
    planned_episodes = len(schedule)
    planned_steps = planned_episodes * MAX_BRANCH_STEPS
    strata_counts: Dict[str, int] = {}
    for t in selected_targets:
        strata_counts[t["source_stratum"]] = strata_counts.get(t["source_stratum"], 0) + 1
    noncontrol_count = sum(1 for t in selected_targets if t.get("noncontrol_for_primary_gate"))
    protocol = {
        "protocol_id": f"vehicle_true_variable_horizon_oracle_bank_v0_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_true_variable_horizon_oracle_bank_freeze_no_simulation",
        "method_label": "IMPROVED",
        "hypothesis": "A larger source-supported development bank will determine whether true per-H MPC admits learnable state-dependent H10/H15/H25 choices: the fastest safe near-best-physical oracle should remain mixed and save measured decision time versus fixed H25 without aggregate physical or safety regression if selector/value-model training is justified.",
        "before_evidence": {
            "postdiagnostic_oracle_label_counts": postdiag.get("oracle_label_counts"),
            "postdiagnostic_noncontrol_oracle_vs_H25": (postdiag.get("comparisons") or {}).get("noncontrol", {}).get("oracle_vs_fixed_H25"),
            "broader_run_headline": broader_done.get("headline"),
            "v1d_smoke_headline": v1d_done.get("headline"),
            "v1e_case5_headline": v1e_done.get("headline"),
            "interpretation": "global H10 is physically unsafe as a policy, but the mixed near-best oracle may expose a control/compute Pareto opportunity worth modeling",
        },
        "target_selection": {
            "rule": "include all 12 stress-v1d selected targets first, retaining controls; then include the four completed true-H broader-block anchors as calibration/continuity states; cap at 16 before any outcome from this future bank",
            "max_primary_states": MAX_PRIMARY_STATES,
            "selected_count": len(selected_targets),
            "selected_noncontrol_for_primary_gate": noncontrol_count,
            "strata_counts": strata_counts,
            "excluded_or_duplicate_candidates": excluded,
            "targets": selected_targets,
        },
        "arms": {"true_horizons": TRUE_HORIZONS, "terminal_profiles": TERMINAL_PROFILES, "repeats": REPEATS, "randomization_seed": RNG_SEED, "blocked_randomization": "shuffle horizon order inside each state/profile/repeat block", "schedule": schedule},
        "budget_declared": {"development_branch_episodes_exact": planned_episodes, "development_control_step_upper_bound": planned_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "candidate_pool_resets": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "future_label_definition": {"safe_horizon": "success true, no constraint, no initial/final/solver failure steps in all repeats", "near_best_physical_set": "safe horizons with median physical cost <= best safe physical cost + max(2.0, 0.05*abs(best safe physical cost))", "oracle_label": "fastest near-best horizon by median whole-decision time; tie by solver time then shorter H", "timing_claim": "measured whole-decision and solver timing only; do not infer from H alone"},
        "gate_after_future_runner": {"oracle_vs_fixed_H25_primary_noncontrol_physical": "aggregate oracle physical delta vs fixed H25 within max(2.0*N, 0.05*abs(fixedH25 physical sum))", "oracle_vs_fixed_H25_primary_noncontrol_timing": ">=15% aggregate/median measured whole-decision saving", "safety": "no safety or solver-regression vs H25", "label_non_degeneracy_for_selector_identifiability": "at least two horizons selected and at least 20% of non-control labels not H25", "if_pass": "freeze bounded IMPROVED selector/value-modeling/refit with >=3 independent seeds and fair fixed-H baselines", "if_fail": "pivot to terminal-value/modeling/objective or scenario-opportunity diagnosis; do not repeat sparse-label sweeps"},
        "access_rules": {"development_only": True, "validation64_bank_opened": False, "sealed_test_accessed": False, "requires_verified_external_backup_before_future_run": True},
    }
    write_json(PROTOCOL, protocol)
    write_json(BACKUP_REQ, {"requested_utc": created.isoformat(), "reason": "backup oracle-bank freezer/source/protocol before the next 192-episode development true-H oracle simulation bank", "backup_required_before_more_simulations": True, "required_after_artifacts_created_utc": created.isoformat(), "planned_development_branch_episodes_exact": planned_episodes, "planned_development_control_step_upper_bound": planned_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "candidate_pool_resets": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(SOURCE), rel(PROTOCOL), rel(OUT), rel(STATE), rel(BACKUP_REQ), rel(DESIGN_NOTE), rel(POSTDIAG_RAW), rel(POSTDIAG_COMPLETED), rel(BROADER_RUN_RAW), rel(BROADER_RUN_COMPLETED), rel(V1D_SELECTED), rel(V1D_COMPLETED), rel(V1E_CASE5_RAW), rel(V1E_CASE5_COMPLETED)]})

    raw = {"created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "method": NAME, "classification": "metadata_no_simulation_oracle_bank_protocol_freeze", "formal_scientific_evidence": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "candidate_pool_resets": 0, "selected_state_count": len(selected_targets), "selected_noncontrol_count": noncontrol_count, "selected_strata_counts": strata_counts, "selected_state_ids": [t["state_id"] for t in selected_targets], "planned_development_branch_episodes_exact": planned_episodes, "planned_development_control_step_upper_bound": planned_steps, "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQ), "completed": rel(OUT / "completed.json"), "next_action_after_backup": "implement/run oracle-bank runner using this frozen protocol, no validation64/test; then decide selector/value-model training versus terminal/modeling/scenario pivot"}
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((OUT / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(raw)
    files = [SOURCE, PROTOCOL, BACKUP_REQ, OUT / "raw.json", OUT / "summary.md", STATE, DESIGN_NOTE, POSTDIAG_RAW, POSTDIAG_COMPLETED, BROADER_PROTOCOL, BROADER_RUN_COMPLETED, BROADER_RUN_RAW, V1D_SELECTED, V1D_COMPLETED, V1E_CASE5_RAW, V1E_CASE5_COMPLETED]
    completed = {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "classification": "metadata_no_simulation_oracle_bank_protocol_freeze", "formal_scientific_evidence": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "candidate_pool_resets": 0, "selected_state_count": len(selected_targets), "selected_noncontrol_count": noncontrol_count, "planned_development_branch_episodes_exact": planned_episodes, "planned_development_control_step_upper_bound": planned_steps, "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQ), "headline": {"train_or_refit_now": False, "oracle_bank_protocol_frozen": True, "selected_state_count": len(selected_targets), "planned_episodes": planned_episodes, "requires_backup_before_simulation": True}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}}
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQ), "selected_state_count": len(selected_targets), "selected_noncontrol_count": noncontrol_count, "planned_episodes": planned_episodes, "control_step_cap": planned_steps, "new_rollouts": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "next_action_after_backup": raw["next_action_after_backup"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
