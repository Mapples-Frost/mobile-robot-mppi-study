#!/usr/bin/env python3
"""Run v2 unused fresh-source confirmation for the v2c top selector.

Development-only IMPROVED diagnostic.  This runner consumes the frozen v2
fresh-source confirmation protocol, first runs Stage-A H15 traces, writes the
selected_state_manifest before any H10 branch outcome, then runs blocked true
H10/H15 branch comparisons under shared-H15-terminal (primary) and matched-
terminal (secondary diagnostic) profiles.

No validation64 bank, no sealed test, no training and no selector refit are
performed.  The primary selector is the v2c top source-only raw_abs_l2
positive-support guard with terminal-disagreement veto:

  source_only::v2b_raw_abs_l2_prs1.25_nm1.5_vm0.75_agreement_only_disagreement_only

The variant was selected using already-opened development banks (fresh_v0/v1),
so this v2 block is still development confirmation, not final validation.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

# Reuse the audited v1 execution/evaluation utilities, but do not reuse v1
# protocols, output directories or labels.  The v2-specific source_guard_model
# patch below reads the v2 protocol key.
import vehicle_true_variable_horizon_fresh_source_confirmation_v1_runner as engine  # noqa:E402

NAME = "vehicle_true_variable_horizon_fresh_source_confirmation_v2"
STAMP = "20260929T1135Z"
SOURCE = Path(__file__).resolve()
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_frozen_20260929T1120Z.json"
FREEZE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_20260929T1120Z/completed.json"
V2C_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_selector_refit_v2c_relaxed_schema_20260929T1145Z/completed.json"
ORACLE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/completed.json"
RISK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/completed.json"

RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_run_{STAMP}"
STATE_RUN = ROOT / f"research_artifacts/aws_state/{NAME}_run_{STAMP}.md"
CONTINUE_STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1135_after_fresh_source_confirmation_v2_run.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER_RUN = f"vehicle-true-variable-H-fresh-source-confirmation-v2-run-{STAMP}"

TRUE_HORIZONS = [10, 15]
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
PRIMARY_TERMINAL_PROFILE = "shared_h15_terminal"
PRIMARY_MODEL_KEY = "source_only::v2b_raw_abs_l2_prs1.25_nm1.5_vm0.75_agreement_only_disagreement_only"
PRIMARY_VARIANT_ID = "v2b_raw_abs_l2_prs1.25_nm1.5_vm0.75_agreement_only_disagreement_only"
MIN_DECISION_SAVING_WEAK = 0.05
MIN_DECISION_SAVING_STRONG = 0.10


class ContractError(RuntimeError):
    pass


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


def write_json(path: Path, value: Any) -> None:
    engine.write_json(path, value)


def sha256(path: Path) -> str:
    return engine.sha256(path)


def si(value: Any, default: int = 0) -> int:
    return engine.si(value, default)


def completed_ok(path: Path, check_hashes: bool = False) -> Mapping[str, Any]:
    obj = engine.completed_ok(path, check_hashes=check_hashes)
    if obj.get("validation64_bank_opened") is not False or obj.get("sealed_test_accessed") is not False:
        raise ContractError("validation/test access flags invalid in " + rel(path))
    return obj


def patch_engine_for_v2() -> None:
    """Patch only global identifiers and v2 protocol lookup in the reused engine."""
    engine.NAME = NAME
    engine.STAMP = STAMP
    engine.PROTOCOL = PROTOCOL
    engine.FREEZE_DONE = FREEZE_DONE
    engine.GUARD_DONE = V2C_DONE
    engine.ORACLE_DONE = ORACLE_DONE
    engine.RISK_DONE = RISK_DONE
    engine.RUN_DIR = RUN_DIR
    engine.STATE_RUN = STATE_RUN
    engine.CONTINUE_STATE = CONTINUE_STATE
    engine.MARKER_RUN = MARKER_RUN
    engine.PRIMARY_TERMINAL_PROFILE = PRIMARY_TERMINAL_PROFILE
    engine.PRIMARY_VARIANT = PRIMARY_VARIANT_ID
    engine.MIN_DECISION_SAVING_WEAK = MIN_DECISION_SAVING_WEAK
    engine.MIN_DECISION_SAVING_STRONG = MIN_DECISION_SAVING_STRONG

    def source_guard_model_v2(protocol: Mapping[str, Any]):
        primary = ((protocol.get("source_trained_selector_confirmation") or {}).get("primary_policy_frozen_before_fresh_v2_outcomes") or {})
        if primary.get("policy_id") != PRIMARY_MODEL_KEY:
            raise ContractError("v2 primary policy key mismatch inside protocol")
        spec = dict(primary.get("config") or {})
        if spec.get("variant_id") != PRIMARY_VARIANT_ID:
            raise ContractError("v2 primary variant_id mismatch inside protocol")
        examples = engine.gv.load_source_examples(engine.gv.ORACLE_RAW, "oracle_bank") + engine.gv.load_source_examples(engine.gv.RISK_RAW, "risk_anchor")
        model = engine.gv.fit_predictor(examples, spec)
        return model, examples

    engine.source_guard_model = source_guard_model_v2


def verify_inputs() -> Mapping[str, Any]:
    patch_engine_for_v2()
    for p in (PROTOCOL, FREEZE_DONE, V2C_DONE, ORACLE_DONE, RISK_DONE):
        if not p.exists():
            raise ContractError("required input missing: " + rel(p))
    completed_ok(FREEZE_DONE, check_hashes=False)
    completed_ok(V2C_DONE, check_hashes=False)
    completed_ok(ORACLE_DONE, check_hashes=False)
    completed_ok(RISK_DONE, check_hashes=False)
    protocol = read_json(PROTOCOL)
    if protocol.get("protocol_id") != "vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_frozen_20260929T1120Z":
        raise ContractError("unexpected v2 protocol id")
    access = protocol.get("access_rules") or {}
    if access.get("validation64_bank_opened") is not False or access.get("sealed_test_accessed") is not False:
        raise ContractError("v2 protocol access flags invalid")
    primary = ((protocol.get("source_trained_selector_confirmation") or {}).get("primary_policy_frozen_before_fresh_v2_outcomes") or {})
    if primary.get("policy_id") != PRIMARY_MODEL_KEY:
        raise ContractError("v2 protocol primary policy_id mismatch")
    spec = primary.get("config") or {}
    for k, v in {
        "variant_id": PRIMARY_VARIANT_ID,
        "mode": "raw_abs_l2",
        "min_positive_support": 1,
        "positive_radius_scale": 1.25,
        "negative_margin": 1.5,
        "veto_margin": 0.75,
        "negative_pool": "agreement_only",
        "veto_pool": "disagreement_only",
    }.items():
        if spec.get(k) != v:
            raise ContractError(f"v2 primary spec mismatch for {k}: {spec.get(k)!r} != {v!r}")
    budget = protocol.get("budget_declared") or {}
    if int(budget.get("total_episodes_exact", -1)) != 136 or int(budget.get("stage_b_branch_episodes", -1)) != 128:
        raise ContractError("unexpected v2 budget")
    selected = ((protocol.get("case_source") or {}).get("selected_fresh_cases") or [])
    if len(selected) != 8:
        raise ContractError("expected exactly 8 v2 fresh cases")
    template = ((protocol.get("stage_B_branch_confirmation") or {}).get("branch_arm_template_after_state_selection") or [])
    if len(template) != 128:
        raise ContractError("expected exactly 128 v2 Stage-B branch arms")
    return protocol


def terminal_for(profile: str, h: int, terminals: Mapping[int, Tuple[Any, Any]]) -> Tuple[Any, Any]:
    return engine.terminal_for(profile, h, terminals)


def decision_from_primary_gate(gate: Mapping[str, Any]) -> str:
    if gate.get("primary_strong_10pct"):
        return "v2 independent fresh-source confirmation strongly supports the v2c source-only shared-H15 true-H10/H15 selector; next freeze validation64 with fair fixed true-H baselines and actual overhead accounting, after backup"
    if gate.get("primary_pass_5pct"):
        return "v2 independent fresh-source confirmation gives only weak measured-decision saving; inspect timing/false-negative pattern and run targeted representation/value/objective calibration before validation64"
    if gate.get("catastrophic_false_positive_rows") or gate.get("unsafe_rows") or not gate.get("physical_gate"):
        return "v2 independent fresh-source confirmation failed safety/physical gates; block selector rollout and prioritize value/objective/representation refit or training ablation"
    if not gate.get("nonconstant_horizons"):
        return "v2 independent fresh-source guard collapsed to a constant horizon; block validation and prioritize representation/value/training diagnosis rather than more unchanged label sweeps"
    return "v2 independent fresh-source confirmation failed compute-saving gate; compare oracle opportunity and timing noise before any validation rollout"


def analyze_v2(branch_episodes: Sequence[Mapping[str, Any]], selected_states: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    patch_engine_for_v2()
    analysis = dict(engine.analyze(branch_episodes, selected_states, protocol))
    analysis["decision"] = decision_from_primary_gate(analysis["primary_gate"])
    return analysis


def write_summary(raw: Mapping[str, Any]) -> None:
    analysis = raw["analysis"]
    gate = analysis["primary_gate"]
    lines: List[str] = [
        "# Vehicle true-variable-H fresh-source confirmation v2 run",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only unused fresh-source confirmation for `{PRIMARY_MODEL_KEY}`; no validation64 bank, no sealed test, no training/refit.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['total_episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Primary shared-H15-terminal gate",
        "",
        f"- Groups: `{gate['groups']}`; H10 predictions `{gate['h10_predictions']}`, H15 abstentions `{gate['h15_predictions']}`, nonconstant `{gate['nonconstant_horizons']}`.",
        f"- Confusion vs fresh H10-beneficial labels: `{gate['confusion_vs_fresh_h10_beneficial_label']}`.",
        f"- Physical gate: `{gate['physical_gate']}`; policy comparison `{gate['comparison_policy_vs_fixed_H15']}`; tolerance sum `{gate['physical_tolerance_sum_vs_fixed_H15']}`.",
        f"- Decision saving gates: >=5% `{gate['decision_saving_gate_5pct']}`, >=10% `{gate['decision_saving_gate_10pct_strong']}`; solver saving `{gate['solver_saving_reported_not_primary']}`.",
        f"- Unsafe H10 rows: `{len(gate['unsafe_rows'])}`; catastrophic false positives: `{len(gate['catastrophic_false_positive_rows'])}`.",
        f"- Weak pass: `{gate['primary_pass_5pct']}`; strong pass: `{gate['primary_strong_10pct']}`.",
        "",
        "## Oracle opportunity and matched-terminal diagnostic",
        "",
        f"- Shared-H15 oracle comparison: `{gate['comparison_oracle_vs_fixed_H15']}`; chosen counts `{gate['oracle_chosen_counts']}`.",
        f"- Matched-terminal policy gate: `{analysis['matched_terminal_gate']}`.",
        f"- Terminal-profile oracle disagreements: `{len(analysis['terminal_profile_disagreements'])}` / 16 states.",
        "",
        "## Per-state shared-H15 rows",
        "",
        "| state | group | pred | oracle | H10 beneficial | H10 phys | H15 phys | H10 dec | H15 dec |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    def fmt(v: Any) -> str:
        return "NA" if v is None else "%.6g" % float(v)
    for row in analysis["state_profile_rows"]:
        if row.get("terminal_profile") != PRIMARY_TERMINAL_PROFILE:
            continue
        m10 = row["median_by_h"]["10"]
        m15 = row["median_by_h"]["15"]
        lines.append("| `%s` | `%s` | %s | %s | `%s` | %s | %s | %s | %s |" % (
            row["base_state_id"], row.get("fresh_confirmation_group"), row.get("primary_policy_prediction"), row.get("oracle_label"), bool(row.get("h10_beneficial_vs_h15")),
            fmt(m10.get("physical")), fmt(m15.get("physical")), fmt(m10.get("decision_sum_s")), fmt(m15.get("decision_sum_s")),
        ))
    if gate["catastrophic_false_positive_rows"] or gate["unsafe_rows"]:
        lines += ["", "## Failure anatomy", ""]
        for row in gate["catastrophic_false_positive_rows"]:
            lines.append(f"- Catastrophic FP `{row['base_state_id']}`: phys delta {row.get('physical_delta_h10_minus_h15')}, decision delta {row.get('decision_delta_h10_minus_h15')}, tol {row.get('row_tolerance')}, threshold {row.get('catastrophic_threshold')}." )
        for row in gate["unsafe_rows"]:
            lines.append(f"- Unsafe H10 `{row['base_state_id']}`: {row.get('reason')}.")
    lines += [
        "",
        "## Decision",
        "",
        str(analysis["decision"]),
        "",
        "Interpretation limits: development-only unused fresh-source confirmation; the selector variant was chosen after opened fresh_v0/fresh_v1 outcomes and remains IMPROVED, not ORIGINAL SAC. Timing claims use measured whole-decision and solver timings, not nominal H. Final validation/test remain unopened.",
        "",
        f"Backup request after run: `{raw['backup_request_after_run']}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER_RUN not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    if reg.exists():
        old = reg.read_text(encoding="utf-8", errors="replace")
        if MARKER_RUN not in old[-50000:]:
            reg.write_text(old.rstrip() + f"\n{now_utc().isoformat()},{NAME},development_fresh_source_confirmation_v2,development_no_validation_no_test,136,0,0,0,0,False,{rel(RUN_DIR / 'completed.json')}\n", encoding="utf-8")


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--backup-verified-commit", type=str, required=True)
    parser.add_argument("--i-accept-development-fresh-source-confirmation-v2", action="store_true")
    args = parser.parse_args(argv)
    if not args.run or not args.i_accept_development_fresh_source_confirmation_v2:
        raise ContractError("requires --run and explicit v2 fresh-source confirmation acknowledgement")
    patch_engine_for_v2()
    if (RUN_DIR / "completed.json").exists():
        done = completed_ok(RUN_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name != "run.lock" for p in RUN_DIR.iterdir()):
        raise ContractError("partial run output exists; inspect before rerun: " + rel(RUN_DIR))

    protocol = verify_inputs()
    selected_cases = (protocol.get("case_source") or {}).get("selected_fresh_cases") or []
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    engine.case_runner.SMOKE_DIR = RUN_DIR

    _, stage1_runner, _ = engine.v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
    stage1_runner.base.v1.latency_verify()
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    for h in TRUE_HORIZONS:
        if h not in terminals:
            raise ContractError("terminal grid missing H%d" % h)

    started = now_utc()
    write_json(RUN_DIR / "run_started.json", {"started_utc": started.isoformat(), "pid": os.getpid(), "method": NAME, "primary_model_key": PRIMARY_MODEL_KEY, "backup_verified_commit_from_supervisor_context": args.backup_verified_commit, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "candidate_pool_resets": 0})
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})

    episodes: List[Dict[str, Any]] = []
    stage_a_eps: List[Dict[str, Any]] = []
    for i, case_meta in enumerate(selected_cases):
        case = case_meta["case_snapshot_from_candidate_pool"]
        item = {
            "execution_index": i,
            "state_id": f"v2_stageA_fresh_case{i:02d}_source{int(case_meta['source_candidate_index']):03d}",
            "case": i,
            "source_candidate_index": int(case_meta["source_candidate_index"]),
            "branch_step": 0,
            "branch_previous_state": copy.deepcopy(case.get("state", {"x": 0.0, "y": 0.0, "theta": 0.0})),
            "true_mpc_n_horizon": 15,
            "commanded_horizon": 15,
            "terminal_mode": "matched_terminal",
            "initialization": "fresh_source_v2_stageA_trueH15_trace_from_initial_state",
        }
        summary = engine.case_runner.run_true_h_episode(item, case, terminals[15])
        summary["stage"] = "A_h15_trace_scan"
        summary["fresh_case_index"] = i
        summary["fresh_confirmation_group"] = case_meta.get("fresh_confirmation_group")
        stage_a_eps.append(summary)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "stage": "A", "episodes_done": len(episodes), "episodes_expected": 136, "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("state_id", "true_mpc_n_horizon", "steps", "success", "termination")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    selected_states = engine.fresh0.select_stage_a_states(stage_a_eps, protocol)
    manifest = {"created_utc": now_utc().isoformat(), "stage_A_complete_before_any_stage_B": True, "stage_A_episode_count": len(stage_a_eps), "selected_states": selected_states, "sha256_protocol": sha256(PROTOCOL), "validation64_bank_opened": False, "sealed_test_accessed": False}
    write_json(RUN_DIR / "selected_state_manifest.json", manifest)
    write_json(RUN_DIR / "manifest_completed_before_stage_B.json", {"created_utc": now_utc().isoformat(), "selected_state_manifest": rel(RUN_DIR / "selected_state_manifest.json"), "selected_state_count": len(selected_states), "stage_B_started": False, "validation64_bank_opened": False, "sealed_test_accessed": False})

    state_by_case_slot = {(si(s["fresh_case_index"], -1), si(s["branch_state_slot"], -1)): s for s in selected_states}
    branch_template = ((protocol.get("stage_B_branch_confirmation") or {}).get("branch_arm_template_after_state_selection") or [])
    if len(branch_template) != 128:
        raise ContractError("unexpected Stage-B template length")
    write_json(RUN_DIR / "stage_B_started.json", {"started_utc": now_utc().isoformat(), "selected_state_manifest_preexisting": True, "validation64_bank_opened": False, "sealed_test_accessed": False})
    branch_episodes: List[Dict[str, Any]] = []
    for j, tmpl in enumerate(branch_template):
        fc = si(tmpl.get("fresh_case_index"), -1)
        slot = si(tmpl.get("branch_state_slot"), -1)
        st = state_by_case_slot.get((fc, slot))
        if st is None:
            raise ContractError("missing selected state for Stage-B template")
        case_meta = selected_cases[fc]
        case = case_meta["case_snapshot_from_candidate_pool"]
        h = si(tmpl.get("true_mpc_n_horizon"), -1)
        profile = str(tmpl.get("terminal_profile"))
        repeat = si(tmpl.get("repeat"), 0)
        unique_state_id = f"v2_B{j:03d}_{st['base_state_id']}_{profile}_r{repeat}_H{h}"
        item = {
            "execution_index": 3000 + j,
            "state_id": unique_state_id,
            "case": fc,
            "source_candidate_index": int(case_meta["source_candidate_index"]),
            "branch_step": int(st["branch_step"]),
            "branch_previous_state": copy.deepcopy(st["branch_previous_state"]),
            "true_mpc_n_horizon": h,
            "commanded_horizon": h,
            "terminal_mode": profile,
            "initialization": "fresh_source_v2_stageB_direct_branch_state_from_predeclared_H15_trace_manifest",
        }
        summary = engine.case_runner.run_true_h_episode(item, case, terminal_for(profile, h, terminals))
        summary["stage"] = "B_blocked_branch_trueH10_H15"
        summary["template_execution_index"] = int(tmpl.get("execution_index", j))
        summary["repeat"] = repeat
        summary["fresh_case_index"] = fc
        summary["base_state_id"] = st["base_state_id"]
        summary["branch_state_slot"] = slot
        summary["terminal_profile"] = profile
        summary["fresh_confirmation_group"] = case_meta.get("fresh_confirmation_group")
        summary["terminal_source_horizon"] = 15 if profile == "shared_h15_terminal" else h
        summary["terminal_receipt_effective"] = terminal_receipts.get(str(summary["terminal_source_horizon"])) or terminal_receipts.get(summary["terminal_source_horizon"])
        branch_episodes.append(summary)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "stage": "B", "episodes_done": len(episodes), "episodes_expected": 136, "stage_B_done": len(branch_episodes), "stage_B_expected": 128, "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("base_state_id", "repeat", "terminal_profile", "true_mpc_n_horizon", "steps", "success", "termination")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    declared = protocol.get("budget_declared") or {}
    control_steps = int(sum(si(e.get("steps")) for e in episodes))
    if len(stage_a_eps) != int(declared.get("stage_a_h15_trace_episodes", -1)) or len(branch_episodes) != int(declared.get("stage_b_branch_episodes", -1)) or len(episodes) != int(declared.get("total_episodes_exact", -1)):
        raise ContractError("episode budget mismatch")
    if control_steps > int(declared.get("control_step_upper_bound", -1)):
        raise ContractError("control-step budget exceeded")

    analysis = analyze_v2(branch_episodes, selected_states, protocol)
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_FRESH_SOURCE_CONFIRMATION_V2_RUN_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup v2 unused fresh-source confirmation runner/source/raw outputs before validation planning or further simulation", "backup_required_before_more_simulations": True, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(SOURCE), rel(RUN_DIR), rel(STATE_RUN), rel(CONTINUE_STATE), rel(PROTOCOL), rel(req)]})
    raw = {
        "created_utc": created.isoformat(),
        "started_utc": started.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_v2_source_only_shared_h15_true_variable_H_fresh_source_confirmation_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "primary_model_key": PRIMARY_MODEL_KEY,
        "primary_variant_id": PRIMARY_VARIANT_ID,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "protocol": {"json": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": declared,
        "budget_actual": {"episodes": len(episodes), "stage_A_h15_trace_episodes": len(stage_a_eps), "stage_B_branch_episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "environment_constructions": len(episodes), "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "stage_A_episodes": stage_a_eps,
        "selected_state_manifest": manifest,
        "branch_episodes": branch_episodes,
        "analysis": analysis,
        "backup_request_after_run": rel(req),
        "interpretation_limits": ["development-only", "unused fresh source but not validation/test", "variant selected after fresh_v0/fresh_v1 development outcomes", "not ORIGINAL SAC", "primary scope is shared-H15 terminal", "does not infer speed from H alone; measured decision/solver timing reported"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_RUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_RUN.write_text((RUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    CONTINUE_STATE.write_text(f"""# Continue state after fresh-source confirmation v2 run

UTC: {created.isoformat()}
Elapsed since first supervisor event: {(created - FIRST_SUPERVISOR_EVENT).total_seconds()/3600.0:.2f} h.

Completed: `{NAME}` development run for `{PRIMARY_MODEL_KEY}`: {len(episodes)} episodes, {control_steps} control steps. No validation64, no sealed test, no training/refit.

Primary gate: {analysis['primary_gate']}
Decision: {analysis['decision']}
Summary: `{rel(RUN_DIR / 'summary.md')}`
Raw: `{rel(RUN_DIR / 'raw.json')}`
Completed: `{rel(RUN_DIR / 'completed.json')}`
Backup request: `{rel(req)}`

Next action: backup this run before any additional simulation/training/refit. If strong pass, freeze validation64 with fair fixed true-H baselines and overhead accounting. If fail or weak-only, prioritize value/objective/representation/terminal-value refit or training ablation instead of another unchanged label-density sweep.
""", encoding="utf-8")
    append_docs(f"""<!-- {MARKER_RUN} -->
## 2026-09-29 vehicle true-variable-H fresh-source confirmation v2 run

UTC: {created.isoformat()}. Development-only unused fresh-source confirmation completed for `{PRIMARY_MODEL_KEY}`: {len(episodes)} episodes ({len(stage_a_eps)} H15 traces + {len(branch_episodes)} blocked H10/H15 branches), {control_steps} control steps. No validation64, no sealed test, no training/refit. Primary gate: {analysis['primary_gate']}. Decision: {analysis['decision']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`.
""")
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, FREEZE_DONE, V2C_DONE, ORACLE_DONE, RISK_DONE, STATE_RUN, CONTINUE_STATE, req]
    completed = {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(), "classification": raw["classification"], "formal_scientific_evidence": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "backup_request": rel(req), "headline": analysis["primary_gate"], "decision": analysis["decision"], "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}}
    write_json(RUN_DIR / "completed.json", completed)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "headline": analysis["primary_gate"], "decision": analysis["decision"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except SystemExit:
        raise
    except BaseException as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RUN_DIR / "failure.json", {"failed_utc": now_utc().isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "next_recovery_hint": "Preserve partial outputs. Audit completed episode directories, selected_state_manifest, progress.json and hashes before any rerun; if source repair is needed, version it and re-backup before additional simulation."})
        raise
