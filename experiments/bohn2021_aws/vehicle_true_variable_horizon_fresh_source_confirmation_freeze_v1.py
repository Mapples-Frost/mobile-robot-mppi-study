#!/usr/bin/env python3
"""Freeze independent fresh-source confirmation for the guard/veto H10/H15 selector.

Development-only IMPROVED protocol freeze.  No simulation, no training/refit,
no validation64, and no sealed-test access.

Why this exists: the v0 fresh-source confirmation found real true-H10/H15
state-level measured-compute opportunity, but the frozen source-trained terminal-
agreement-only guard produced one physical-regression false positive.  A later
no-simulation diagnostic, necessarily outcome-informed on that v0 fresh block,
found that a simple disagreement-veto/raw_abs_l2 guard removed the false positive
and exceeded the 10% measured-decision-saving development gate.  This script
freezes a new independent fresh-source confirmation for that preselected variant,
using different stress-v1 candidate-source cases before any new H10 outcome is
observed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, List

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
import sys
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0 as base  # noqa:E402

NAME = "vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1"
STAMP = "20260929T1055Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
CONTINUE_STATE = ROOT / "research_artifacts/aws_state/continue_state_20260929T1055_after_fresh_source_confirmation_freeze_v1.md"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_frozen_{STAMP}.json"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_FRESH_SOURCE_CONFIRMATION_FREEZE_V1_{STAMP}.json"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-true-variable-H-fresh-source-confirmation-freeze-v1-{STAMP}"

FRESH_V0_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_frozen_20260929T1015Z.json"
FRESH_V0_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/completed.json"
FRESH_V0_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/raw.json"
GUARD_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_guard_veto_diagnostic_v0_20260929T1035Z/completed.json"
GUARD_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_guard_veto_diagnostic_v0_20260929T1035Z/raw.json"

TRUE_HORIZONS = [10, 15]
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
STAGE_A_SCAN_H = 15
TARGET_CASES = 8
BRANCH_STATES_PER_CASE = 2
REPEATS = 2
MAX_STEPS = 150
RNG_SEED = 202609291055
PRIMARY_VARIANT = "disagreementVeto_s1_raw_abs_l2_m1.5_v1.0"
GROUP_QUOTAS = base.GROUP_QUOTAS


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


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
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


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


def completed_ok(path: Path) -> Mapping[str, Any]:
    obj = base.completed_ok(path)
    if obj.get("validation64_bank_opened") is not False or obj.get("sealed_test_accessed") is not False:
        raise ContractError("access flags invalid in " + rel(path))
    return obj


def previous_fresh_source_indices() -> List[int]:
    out = []
    if FRESH_V0_PROTOCOL.exists():
        protocol = read_json(FRESH_V0_PROTOCOL)
        for c in ((protocol.get("case_source") or {}).get("selected_fresh_cases") or []):
            try:
                out.append(int(c.get("source_candidate_index")))
            except Exception:
                pass
    if FRESH_V0_RAW.exists():
        manifest = (read_json(FRESH_V0_RAW).get("selected_state_manifest") or {})
        for s in (manifest.get("selected_states") or []):
            try:
                out.append(int(s.get("source_candidate_index")))
            except Exception:
                pass
    return sorted(set(x for x in out if x >= 0))


def make_branch_arm_template(selected_cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    rows: List[Dict[str, Any]] = []
    exe = 0
    for rep in range(REPEATS):
        for case in selected_cases:
            for state_slot in range(BRANCH_STATES_PER_CASE):
                for profile in TERMINAL_PROFILES:
                    hs = list(TRUE_HORIZONS)
                    rng.shuffle(hs)
                    for h in hs:
                        rows.append({
                            "stage": "B_blocked_branch_trueH10_H15",
                            "execution_index": exe,
                            "repeat": rep,
                            "fresh_case_index": int(case["fresh_case_index"]),
                            "source_candidate_index": int(case["source_candidate_index"]),
                            "branch_state_slot": state_slot,
                            "terminal_profile": profile,
                            "true_mpc_n_horizon": h,
                            "blocked_randomization_unit": f"v1|repeat{rep}|fresh_case{case['fresh_case_index']}|slot{state_slot}|{profile}",
                        })
                        exe += 1
    return rows


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    if reg.exists():
        old = reg.read_text(encoding="utf-8", errors="replace")
        if MARKER not in old[-50000:]:
            reg.write_text(old.rstrip() + f"\n{now_utc().isoformat()},{NAME},metadata_no_simulation_guard_veto_fresh_source_confirmation_freeze,development_no_validation_no_test,0,0,0,0,0,False,{rel(OUT / 'completed.json')}\n", encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    diag = raw["case_selection_diagnostics"]
    lines = [
        "# Vehicle true-variable-H fresh-source confirmation freeze v1",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata-only/no-simulation freeze. No training/refit, no validation64, no sealed test.",
        "",
        "## Frozen question",
        "",
        "Confirm, on independent fresh-source branch states, the outcome-informed guard/veto variant selected after the v0 fresh-source failure. This is still development confirmation, not formal validation or final test.",
        "",
        "## Primary variant frozen before new outcomes",
        "",
        f"- `{PRIMARY_VARIANT}`",
        "- Feature: online-observable raw observation plus absolute values, L2-normalized (`raw_abs_l2`).",
        "- Training labels: existing oracle-bank + risk-anchor only; agreement H10 are positives, agreement non-H10 are negatives, terminal-disagreement/missing states are veto anchors. No fitting or threshold tuning on v1 fresh outcomes.",
        "- Config: min_positive_support=1, positive_radius_quantile=0.5, positive_radius_scale=1.25, negative_margin=1.5, veto_margin=1.0.",
        "",
        "## Independent fresh source selection",
        "",
        f"- Excluded count: `{diag['excluded_source_candidate_indices_count']}` including Stage1 selected, oracle/risk sources, and v0 fresh-source cases `{raw['v0_fresh_source_indices_excluded']}`.",
        f"- Selected source_candidate_index values: `{diag['selected_source_candidate_indices']}`.",
        f"- Group counts: `{diag['selected_group_counts']}`.",
        "",
        "| fresh_case | source_candidate_index | group | theta_r | traj_steps | clearance | stress_score |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for c in raw["selected_cases"]:
        lines.append("| %d | %d | `%s` | %.6g | %.6g | %.6g | %.6g |" % (int(c["fresh_case_index"]), int(c["source_candidate_index"]), c["fresh_confirmation_group"], base.sf(c.get("theta_r")), base.sf(c.get("traj_steps")), base.sf(c.get("min_reference_obstacle_clearance")), base.sf(c.get("stress_v1_score"))))
    lines += [
        "",
        "## Frozen future runner budget/gates",
        "",
        f"- Stage A H15 trace scans: `{raw['budget_declared']['stage_a_h15_trace_episodes']}`.",
        f"- Stage B blocked H10/H15 branch episodes: `{raw['budget_declared']['stage_b_branch_episodes']}`.",
        f"- Total budget: `{raw['budget_declared']['total_episodes_exact']}` episodes / cap `{raw['budget_declared']['control_step_upper_bound']}` control steps.",
        "- Primary profile: shared_h15_terminal; matched_terminal remains a secondary diagnostic.",
        "- Success gates: manifest before Stage B; H15 baseline safe; nonconstant H; zero unsafe/catastrophic H10 choices; aggregate physical delta within tolerance; measured whole-decision saving >=10% for strong confirmation (>=5% weak only); solver timing reported separately.",
        "",
        "## Decision",
        "",
        "After verified backup of this source/protocol, run the paired development-only v1 runner. A pass permits drafting a validation-freeze plan; a fail shifts priority to value/objective/representation refit or training ablation rather than another label-density sweep.",
        "",
        f"Protocol: `{raw['protocol']}`. Backup request: `{raw['backup_request']}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_continue_state(raw: Mapping[str, Any]) -> None:
    text = f"""# Continue state after fresh-source confirmation freeze v1

UTC: {raw['created_utc']}
Elapsed since first supervisor event: {raw['elapsed_since_first_supervisor_event_seconds']/3600.0:.2f} h.

Completed: `{NAME}` metadata-only/no-simulation freeze for the outcome-informed guard/veto variant `{PRIMARY_VARIANT}`. No simulations, no control steps, no training/refit, no validation64, no sealed test.

Independent source guard: selected stress-v1 candidate IDs {raw['case_selection_diagnostics']['selected_source_candidate_indices']}; excluded v0 fresh candidate IDs {raw['v0_fresh_source_indices_excluded']} plus prior Stage1/oracle/risk sources.

Future runner budget: {raw['budget_declared']['total_episodes_exact']} episodes ({raw['budget_declared']['stage_a_h15_trace_episodes']} H15 traces + {raw['budget_declared']['stage_b_branch_episodes']} blocked branch episodes), cap {raw['budget_declared']['control_step_upper_bound']} control steps; no training/refit.

Next action: verify external backup covering `{rel(SOURCE)}`, `{raw['protocol']}`, and this freeze's artifacts, then run the v1 development confirmation runner. Do not open validation64 or sealed test. If v1 fails, move to value/objective/representation refit or training ablation; do not repeat unchanged label-density sweeps.

Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`, `{raw['protocol']}`.
Backup request: `{raw['backup_request']}`.
"""
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((OUT / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    CONTINUE_STATE.write_text(text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--i-accept-no-simulation-fresh-source-confirmation-freeze-v1", action="store_true")
    args = parser.parse_args(argv)
    if not args.freeze or not args.i_accept_no_simulation_fresh_source_confirmation_freeze_v1:
        raise ContractError("requires --freeze and explicit no-simulation v1 fresh-source freeze acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    created = now_utc()
    # Base prerequisites cover bank/stage1/oracle/risk/transfer artifacts and access flags.
    input_hashes = base.require_inputs()
    completed_ok(FRESH_V0_DONE)
    completed_ok(GUARD_DONE)
    guard_done = read_json(GUARD_DONE)
    if int(guard_done.get("passing_variant_count_10pct", 0)) <= 0:
        raise ContractError("guard/veto diagnostic did not provide a strong passing variant")
    guard_raw = read_json(GUARD_RAW)
    passing10 = guard_raw.get("passing_variants_shared_h15_10pct") or []
    if not any(p.get("variant_id") == PRIMARY_VARIANT for p in passing10):
        raise ContractError("primary variant not present among 10% passing guard/veto candidates")

    bank = read_json(base.BANK_PATH)
    oracle_protocol = read_json(base.ORACLE_PROTOCOL)
    oracle_raw = read_json(base.ORACLE_RAW)
    risk_protocol = read_json(base.RISK_PROTOCOL)
    risk_raw = read_json(base.RISK_RAW)
    fresh_v0_indices = previous_fresh_source_indices()
    stage1_selected = base.selected_stage1_indices(bank)
    prior_source_used = base.used_candidate_indices(oracle_protocol, oracle_raw, risk_protocol, risk_raw)
    excluded = sorted(set(stage1_selected) | set(prior_source_used) | set(fresh_v0_indices))
    selected_cases, case_diag = base.build_case_selection(bank, excluded)
    selected_idx = set(case_diag["selected_source_candidate_indices"])
    if selected_idx & set(fresh_v0_indices):
        raise ContractError("v1 selected a v0 fresh-source candidate")

    stage_a = base.make_stage_a_schedule(selected_cases)
    branch_template = make_branch_arm_template(selected_cases)
    stage_a_eps = len(stage_a)
    stage_b_eps = len(branch_template)
    total_eps = stage_a_eps + stage_b_eps
    control_cap = total_eps * MAX_STEPS
    if total_eps != 136 or control_cap != 20400:
        raise ContractError("unexpected budget")

    primary_policy = {
        "policy_id": PRIMARY_VARIANT,
        "terminal_profile": "shared_h15_terminal",
        "selection_origin": "chosen after v0 fresh-source outcome-informed guard/veto diagnostic; this v1 protocol is the required independent fresh-source confirmation before validation",
        "training_sources": ["oracle_bank", "risk_anchor"],
        "fresh_v1_fit_allowed": False,
        "fresh_v1_threshold_tuning_allowed": False,
        "model_family": "positive_support_knn_with_terminal_disagreement_veto",
        "feature_set": "online_observable_raw_abs_l2_no_metadata",
        "label_strategy": {
            "positive": "terminal-profile agreement and oracle_label==10",
            "negative": "terminal-profile agreement and oracle_label!=10",
            "veto": "terminal-profile disagreement or missing label",
        },
        "config": {
            "variant_id": PRIMARY_VARIANT,
            "mode": "raw_abs_l2",
            "min_positive_support": 1,
            "positive_radius_quantile": 0.5,
            "positive_radius_scale": 1.25,
            "negative_margin": 1.5,
            "negative_pool": "agreement_only",
            "veto_pool": "disagreement_only",
            "veto_margin": 1.0,
        },
        "primary_gate": True,
    }
    protocol = {
        "protocol_id": f"{NAME}_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_guard_veto_fresh_source_confirmation_freeze_no_simulation_not_validation_not_test",
        "method_label": "IMPROVED",
        "hypothesis": "The v0 failure was driven by an under-constrained terminal-agreement-only support guard rather than absence of state-dependent H10/H15 opportunity. A fixed disagreement-veto raw_abs_l2 source-trained guard should retain measured decision-time savings on independent fresh-source states while eliminating unsafe/catastrophic H10 false positives vs fixed true H15.",
        "before_evidence": {
            "fresh_v0_primary_gate": read_json(FRESH_V0_DONE).get("headline"),
            "fresh_v0_decision": read_json(FRESH_V0_DONE).get("decision"),
            "fresh_v0_summary": rel(ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/summary.md"),
            "guard_veto_diagnostic_decision": guard_raw.get("decision"),
            "primary_variant_v0_development_result": next((p for p in passing10 if p.get("variant_id") == PRIMARY_VARIANT), None),
        },
        "case_source": {
            "bank_path": rel(base.BANK_PATH),
            "candidate_pool_created_before_this_freeze": True,
            "candidate_pool_horizon_outcomes_available_only_for_previous_excluded_indices": True,
            "excluded_indices_rule": "exclude stress-v1 Stage1 selected_indices, every source_candidate_index in oracle-bank or risk-anchor protocols/runs, and all v0 fresh-source confirmation candidate IDs",
            "stage1_selected_indices_excluded": stage1_selected,
            "oracle_or_risk_source_candidate_indices_excluded": prior_source_used,
            "v0_fresh_source_indices_excluded": fresh_v0_indices,
            "all_excluded_source_candidate_indices": excluded,
            "selected_fresh_cases": selected_cases,
            "case_selection_diagnostics": case_diag,
        },
        "stage_A_h15_trace_state_selection": {
            "schedule": stage_a,
            "horizon": STAGE_A_SCAN_H,
            "terminal_profile": "matched_terminal",
            "state_selection_time_order": "run all Stage-A H15 traces, write selected_state_manifest, then and only then run any Stage-B H10/H15 branch comparison",
            "branch_states_per_case": BRANCH_STATES_PER_CASE,
            "eligible_step_bounds": {"min_step": 8, "max_step": 90, "reserve_terminal_margin_steps": 3},
            "windows": [
                {"slot": 0, "name": "early_mid", "low_fraction": 0.12, "high_fraction": 0.36, "selection": "max_h15_trace_risk_score"},
                {"slot": 1, "name": "mid_late", "low_fraction": 0.38, "high_fraction": 0.75, "selection": "max_h15_trace_risk_score_with_min_step_separation_8"},
            ],
            "risk_score_formula": "same as v0 fresh-source confirmation; derived solely from H15 trace before Stage-B",
        },
        "stage_B_branch_confirmation": {
            "branch_arm_template_after_state_selection": branch_template,
            "true_horizons": TRUE_HORIZONS,
            "terminal_profiles": TERMINAL_PROFILES,
            "repeats": REPEATS,
            "blocked_randomization_seed": RNG_SEED,
            "measured_timing_required": ["whole_decision_time", "solver_attempt_time"],
            "baseline": "fixed true H15 within the same terminal profile and fresh state group",
        },
        "source_trained_selector_confirmation": {
            "primary_policy_frozen_before_fresh_v1_outcomes": primary_policy,
            "primary_success_gate": {
                "scope": "shared_h15_terminal fresh branch groups only",
                "manifest": "selected_state_manifest must be written before any Stage-B branch outcome",
                "safety": "zero H10 chosen groups with H10 unsafe, constraint, initial/final solver failure, or solver_failure_steps>0 when H15 is safe",
                "physical": "aggregate predicted-policy physical delta vs fixed true H15 <= sum per-row tolerances and no catastrophic per-group physical regression > max(2, 0.25*abs(H15 physical))",
                "compute": ">=10% measured whole-decision saving vs fixed true H15 for strong pass; >=5% is weak development-only pass; solver timing reported separately",
                "nontriviality": "at least one H10 prediction and at least one H15 abstention; all-H10/all-H15 is not a strong adaptive-selector pass",
            },
            "secondary_diagnostics": ["matched_terminal profile", "oracle opportunity", "terminal-profile disagreement rate", "timing medians and paired groups", "false-positive anatomy if any"],
        },
        "budget_declared": {
            "stage_a_h15_trace_episodes": stage_a_eps,
            "stage_b_branch_episodes": stage_b_eps,
            "total_episodes_exact": total_eps,
            "control_step_upper_bound": control_cap,
            "candidate_pool_resets": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "decision_after_future_runner": {
            "if_strong_passes": "freeze a validation64 protocol with fair fixed true-H baselines and explicit overhead measurement; still no sealed test until multi-seed/validation gates are met",
            "if_weak_only": "inspect timing uncertainty and false-negative pattern; likely run value/objective/representation calibration before validation",
            "if_false_positive_or_no_saving": "block selector rollout; prioritize bounded value-refit/training/objective ablation or scenario-design diagnosis",
        },
        "access_rules": {"development_only": True, "validation64_bank_opened": False, "sealed_test_accessed": False, "requires_verified_external_backup_before_future_runner_or_simulation": True},
    }
    write_json(PROTOCOL, protocol)
    write_json(BACKUP_REQ, {
        "requested_utc": created.isoformat(),
        "reason": "backup v1 guard/veto fresh-source confirmation freeze/source/protocol before any v1 H15-trace or H10/H15 branch simulations",
        "backup_required_before_more_simulations": True,
        "backup_required_before_training_or_refit": True,
        "future_total_episodes_exact": total_eps,
        "future_control_step_upper_bound": control_cap,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(SOURCE), rel(PROTOCOL), rel(OUT), rel(STATE), rel(CONTINUE_STATE), rel(BACKUP_REQ)],
    })
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "classification": protocol["classification"],
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "input_hashes": {**input_hashes, rel(SOURCE): sha256(SOURCE), rel(FRESH_V0_DONE): sha256(FRESH_V0_DONE), rel(FRESH_V0_RAW): sha256(FRESH_V0_RAW), rel(GUARD_DONE): sha256(GUARD_DONE), rel(GUARD_RAW): sha256(GUARD_RAW)},
        "primary_variant": PRIMARY_VARIANT,
        "v0_fresh_source_indices_excluded": fresh_v0_indices,
        "selected_cases": selected_cases,
        "case_selection_diagnostics": case_diag,
        "budget_declared": protocol["budget_declared"],
        "protocol": rel(PROTOCOL),
        "backup_request": rel(BACKUP_REQ),
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    write_continue_state(raw)
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H fresh-source confirmation freeze v1

UTC: {created.isoformat()}. Metadata-only/no-simulation protocol freeze completed for `{PRIMARY_VARIANT}`. It excludes previous Stage1/oracle/risk sources and all v0 fresh-source candidate IDs {fresh_v0_indices}; selected independent candidate IDs {case_diag['selected_source_candidate_indices']}. Future runner budget is {total_eps} episodes / {control_cap} control-step cap. No validation64/test/training/refit. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(PROTOCOL)}`.
""")
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, CONTINUE_STATE, BACKUP_REQ, FRESH_V0_DONE, GUARD_DONE, GUARD_RAW, base.BANK_DONE, base.STAGE1_DONE, base.ORACLE_DONE, base.RISK_DONE]
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "classification": raw["classification"],
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "future_total_episodes_exact": total_eps,
        "future_control_step_upper_bound": control_cap,
        "headline": {"fresh_source_protocol_frozen": True, "primary_variant": PRIMARY_VARIANT, "selected_case_count": len(selected_cases), "selected_group_counts": case_diag["selected_group_counts"], "requires_backup_before_future_runner": True},
        "backup_request": rel(BACKUP_REQ),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "protocol": rel(PROTOCOL), "headline": completed["headline"], "future_total_episodes_exact": total_eps, "future_control_step_upper_bound": control_cap, "new_simulations": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(BACKUP_REQ)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {"failed_utc": now_utc().isoformat(), "exception": repr(exc), "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulations": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0})
        raise
