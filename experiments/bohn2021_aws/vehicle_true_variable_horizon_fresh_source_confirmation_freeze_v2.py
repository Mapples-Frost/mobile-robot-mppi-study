#!/usr/bin/env python3
"""Freeze a v2 independent fresh-source confirmation for the v2c top selector.

Metadata-only protocol freeze: no simulation, no selector refit, no gradient
training, no validation64 access, and no sealed-test access.

Why: selector-refit v2c found a source-only raw_abs_l2 guard that strongly
passes the shared-H15-terminal primary gate on both already-opened development
banks (fresh_v0 and fresh_v1) but fails matched-terminal robustness.  This
script freezes an unused fresh-source confirmation block before any new H10
branch outcomes are observed.  The primary method is deliberately scoped to a
shared-H15 terminal/value profile; matched-terminal remains a diagnostic, not a
validation claim.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0 as base  # noqa:E402

NAME = "vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2"
STAMP = "20260929T1120Z"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-true-variable-H-fresh-source-confirmation-freeze-v2-{STAMP}"

OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
CONTINUE_STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1120_after_fresh_source_confirmation_freeze_v2.md"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_frozen_{STAMP}.json"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_FRESH_SOURCE_CONFIRMATION_FREEZE_V2_{STAMP}.json"

V2C_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_selector_refit_v2c_relaxed_schema_20260929T1145Z"
V2C_DONE = V2C_DIR / "completed.json"
V2C_RAW = V2C_DIR / "raw.json"
V2C_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_selector_refit_v2c_relaxed_schema_20260929T1145Z_candidate_protocol.json"
V0_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_frozen_20260929T1015Z.json"
V1_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1_frozen_20260929T1055Z.json"
V0_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/raw.json"
V1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/raw.json"
V0_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/completed.json"
V1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/completed.json"

PRIMARY_MODEL_KEY = "source_only::v2b_raw_abs_l2_prs1.25_nm1.5_vm0.75_agreement_only_disagreement_only"
PRIMARY_SPEC = {
    "variant_id": "v2b_raw_abs_l2_prs1.25_nm1.5_vm0.75_agreement_only_disagreement_only",
    "mode": "raw_abs_l2",
    "min_positive_support": 1,
    "positive_radius_quantile": 0.5,
    "positive_radius_scale": 1.25,
    "negative_margin": 1.5,
    "negative_pool": "agreement_only",
    "veto_margin": 0.75,
    "veto_pool": "disagreement_only",
}

TRUE_HORIZONS = [10, 15]
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
PRIMARY_TERMINAL_PROFILE = "shared_h15_terminal"
STAGE_A_SCAN_H = 15
TARGET_CASES = 8
BRANCH_STATES_PER_CASE = 2
REPEATS = 2
MAX_STEPS = 150
RNG_SEED = 202609291120


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


def completed_ok_relaxed(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing prerequisite: " + rel(path))
    obj = read_json(path)
    if obj.get("sealed_test_accessed") is True or obj.get("sealed_test_bank_opened") is True or obj.get("validation64_bank_opened") is True:
        raise ContractError("prerequisite explicitly opened validation/test: " + rel(path))
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError("prerequisite did not complete cleanly: " + rel(path))
    return obj


def iter_source_indices_from_obj(obj: Any) -> Iterable[int]:
    if isinstance(obj, Mapping):
        for key in ("source_candidate_index", "candidate_index"):
            if key in obj:
                try:
                    yield int(obj[key])
                except Exception:
                    pass
        for value in obj.values():
            yield from iter_source_indices_from_obj(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_source_indices_from_obj(item)


def source_indices_from_paths(paths: Sequence[Path]) -> List[int]:
    out: List[int] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            out.extend(iter_source_indices_from_obj(read_json(path)))
        except Exception:
            continue
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
                            "blocked_randomization_unit": f"v2|repeat{rep}|fresh_case{case['fresh_case_index']}|slot{state_slot}|{profile}",
                        })
                        exe += 1
    return rows


def verify_v2c() -> Dict[str, Any]:
    done = completed_ok_relaxed(V2C_DONE)
    completed_ok_relaxed(V0_DONE)
    completed_ok_relaxed(V1_DONE)
    if not V2C_RAW.exists() or not V2C_PROTOCOL.exists():
        raise ContractError("missing v2c raw/protocol artifact")
    protocol = read_json(V2C_PROTOCOL)
    best = protocol.get("best_candidate") or {}
    if best.get("model_key") != PRIMARY_MODEL_KEY:
        raise ContractError("v2c best candidate changed or not as expected")
    if best.get("crossbank_shared_strong_pass") is not True or int(best.get("primary_bad_count_total", -1)) != 0:
        raise ContractError("v2c best candidate does not satisfy the shared-H15 strong/no-primary-bad prerequisites")
    if best.get("validation64_bank_opened") is True or best.get("sealed_test_accessed") is True:
        raise ContractError("unexpected validation/test flags in v2c protocol best candidate")
    raw = read_json(V2C_RAW)
    if raw.get("validation64_bank_opened") is True or raw.get("sealed_test_accessed") is True:
        raise ContractError("v2c raw shows validation/test access")
    return {"completed": done, "candidate_protocol": protocol, "raw_decision": raw.get("decision"), "bank_label_summary": raw.get("bank_label_summary")}


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
            reg.write_text(old.rstrip() + f"\n{now_utc().isoformat()},{NAME},metadata_no_simulation_v2_fresh_source_confirmation_freeze,development_no_validation_no_test,0,0,0,0,0,False,{rel(OUT / 'completed.json')}\n", encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    diag = raw["case_selection_diagnostics"]
    lines = [
        "# Vehicle true-variable-H fresh-source confirmation freeze v2",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata-only/no-simulation freeze. No selector refit, no training, no validation64, no sealed test.",
        "",
        "## Frozen hypothesis",
        "",
        raw["hypothesis"],
        "",
        "## Frozen primary candidate",
        "",
        f"- Model key: `{PRIMARY_MODEL_KEY}`",
        f"- Spec: `{PRIMARY_SPEC}`",
        "- Primary deployment profile: `shared_h15_terminal` (matched-terminal is secondary diagnostic only because v2c showed catastrophic matched-terminal rows).",
        "- Training data: source-only oracle + risk-anchor labels; no fresh_v0/fresh_v1 labels are used to fit the model, although v2c used opened development banks to select this exact variant.",
        "",
        "## Fresh-source exclusion/selection rule",
        "",
        f"- Excluded source_candidate_index count: `{diag['excluded_source_candidate_indices_count']}`.",
        f"- Excluded prior fresh_v0/v1 indices: `{raw['prior_fresh_source_indices_excluded']}`.",
        f"- Selected new indices: `{diag['selected_source_candidate_indices']}`.",
        f"- Selected group counts: `{diag['selected_group_counts']}`.",
        "",
        "| fresh_case | source_candidate_index | group | theta_r | traj_steps | clearance | stress_score |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for c in raw["selected_cases"]:
        lines.append("| %d | %d | `%s` | %.6g | %.6g | %.6g | %.6g |" % (int(c["fresh_case_index"]), int(c["source_candidate_index"]), c["fresh_confirmation_group"], base.sf(c.get("theta_r")), base.sf(c.get("traj_steps")), base.sf(c.get("min_reference_obstacle_clearance")), base.sf(c.get("stress_v1_score"))))
    lines += [
        "",
        "## Future runner budget and gates",
        "",
        f"- Stage A H15 traces: `{raw['budget_declared']['stage_a_h15_trace_episodes']}`; Stage B blocked branches: `{raw['budget_declared']['stage_b_branch_episodes']}`; total episodes: `{raw['budget_declared']['total_episodes_exact']}`; control-step cap: `{raw['budget_declared']['control_step_upper_bound']}`.",
        "- Manifest from Stage-A H15 traces must be written before any Stage-B H10 branch outcome.",
        "- Primary strong pass requires fixed true H15 safe, nonconstant H10/H15 decisions, zero unsafe/catastrophic selected H10 rows, aggregate physical delta within tolerance, and >=10% measured whole-decision saving vs fixed true H15. >=5% is weak only. Solver timing is secondary and reported separately.",
        "",
        "## Decision",
        "",
        "After verified external backup of this freeze/source/protocol, run the v2 fresh-source confirmation runner. Do not open validation64 or sealed final test. If v2 fails or is weak-only, move to terminal-value/objective/representation or bounded training/refit ablation rather than another unchanged selector sweep.",
        "",
        f"Protocol: `{rel(PROTOCOL)}`. Backup request: `{rel(BACKUP_REQ)}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--i-accept-no-simulation-fresh-source-confirmation-freeze-v2", action="store_true")
    args = parser.parse_args(argv)
    if not args.freeze or not args.i_accept_no_simulation_fresh_source_confirmation_freeze_v2:
        raise ContractError("requires --freeze and explicit no-simulation v2 fresh-source freeze acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    created = now_utc()
    input_hashes = base.require_inputs()
    v2c = verify_v2c()
    bank = read_json(base.BANK_PATH)
    oracle_protocol = read_json(base.ORACLE_PROTOCOL)
    oracle_raw = read_json(base.ORACLE_RAW)
    risk_protocol = read_json(base.RISK_PROTOCOL)
    risk_raw = read_json(base.RISK_RAW)

    stage1_selected = base.selected_stage1_indices(bank)
    oracle_risk_used = base.used_candidate_indices(oracle_protocol, oracle_raw, risk_protocol, risk_raw)
    prior_fresh = source_indices_from_paths([V0_PROTOCOL, V1_PROTOCOL, V0_RAW, V1_RAW])
    excluded = sorted(set(stage1_selected) | set(oracle_risk_used) | set(prior_fresh))
    selected_cases, case_diag = base.build_case_selection(bank, excluded)
    selected_idx = set(case_diag["selected_source_candidate_indices"])
    if selected_idx & set(prior_fresh):
        raise ContractError("v2 selected a prior fresh-source candidate")
    if selected_idx & set(oracle_risk_used):
        raise ContractError("v2 selected an oracle/risk source candidate")
    if len(selected_cases) != TARGET_CASES:
        raise ContractError("unexpected selected case count")

    stage_a = base.make_stage_a_schedule(selected_cases)
    branch_template = make_branch_arm_template(selected_cases)
    total_eps = len(stage_a) + len(branch_template)
    control_cap = total_eps * MAX_STEPS
    if total_eps != 136 or len(branch_template) != 128 or control_cap != 20400:
        raise ContractError("unexpected v2 confirmation budget")

    hypothesis = (
        "The v2c top source-only raw_abs_l2 guard identifies shared-H15-terminal states where H10 preserves physical/safety performance "
        "while improving measured decision time. Because v2c selection used opened development banks, an unused fresh-source block is required before any validation64 plan."
    )
    primary_policy = {
        "policy_id": PRIMARY_MODEL_KEY,
        "method_label": "IMPROVED",
        "model_family": "positive_support_knn_with_terminal_disagreement_veto",
        "feature_set": "online_observable_raw_abs_l2_no_metadata",
        "terminal_profile_primary": PRIMARY_TERMINAL_PROFILE,
        "training_sources": ["oracle_bank", "risk_anchor"],
        "development_selection_sources": ["fresh_v0", "fresh_v1", "selector_refit_v2c_crossbank"],
        "fresh_v2_fit_allowed": False,
        "fresh_v2_threshold_tuning_allowed": False,
        "config": PRIMARY_SPEC,
        "known_limitation_before_v2": "matched-terminal profile produced catastrophic false positives in v2c; matched-terminal is secondary diagnostic only and not validated by this freeze",
    }
    protocol = {
        "protocol_id": f"{NAME}_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_fresh_source_confirmation_v2_freeze_no_simulation_not_validation_not_test",
        "method_label": "IMPROVED",
        "hypothesis": hypothesis,
        "before_evidence": {
            "selector_refit_v2c_completed": rel(V2C_DONE),
            "selector_refit_v2c_protocol": rel(V2C_PROTOCOL),
            "v2c_headline": v2c["completed"].get("headline"),
            "v2c_decision": v2c["completed"].get("decision"),
            "fresh_v0_completed": rel(V0_DONE),
            "fresh_v1_completed": rel(V1_DONE),
            "interpretation": "Primary shared-H15 cross-bank strong pass but matched-terminal robustness concern; v2 is independent fresh-source confirmation before validation64.",
        },
        "case_source": {
            "bank_path": rel(base.BANK_PATH),
            "candidate_pool_created_before_this_freeze": True,
            "excluded_indices_rule": "exclude stress-v1 Stage1 selected indices, oracle/risk source indices, and every source_candidate_index appearing in fresh_v0 or fresh_v1 protocols/raw manifests",
            "stage1_selected_indices_excluded": stage1_selected,
            "oracle_or_risk_source_candidate_indices_excluded": oracle_risk_used,
            "prior_fresh_source_indices_excluded": prior_fresh,
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
            "risk_score_formula": "same as v0/v1 fresh-source confirmation; derived solely from H15 trace before Stage-B",
        },
        "stage_B_branch_confirmation": {
            "branch_arm_template_after_state_selection": branch_template,
            "true_horizons": TRUE_HORIZONS,
            "terminal_profiles": TERMINAL_PROFILES,
            "primary_terminal_profile": PRIMARY_TERMINAL_PROFILE,
            "repeats": REPEATS,
            "blocked_randomization_seed": RNG_SEED,
            "measured_timing_required": ["whole_decision_time", "solver_attempt_time"],
            "baseline": "fixed true H15 within the same terminal profile and fresh state group",
        },
        "source_trained_selector_confirmation": {
            "primary_policy_frozen_before_fresh_v2_outcomes": primary_policy,
            "primary_success_gate": {
                "scope": "shared_h15_terminal fresh branch groups only",
                "manifest": "selected_state_manifest must be written before any Stage-B branch outcome",
                "safety": "zero H10 chosen groups with H10 unsafe/constraint/initial or final solver failure/solver_failure_steps>0 when H15 is safe",
                "physical": "aggregate predicted-policy physical delta vs fixed true H15 <= sum per-row tolerances and no catastrophic per-group physical regression > max(2, 0.25*abs(H15 physical))",
                "compute": ">=10% measured whole-decision saving vs fixed true H15 for strong pass; >=5% is weak development-only pass; solver timing reported separately",
                "nontriviality": "at least one H10 prediction and at least one H15 abstention; all-H10/all-H15 is not a strong adaptive-selector pass",
            },
            "secondary_diagnostics": ["matched_terminal profile", "oracle opportunity", "terminal-profile disagreement rate", "timing medians and paired groups", "false-positive anatomy if any"],
        },
        "budget_declared": {
            "stage_a_h15_trace_episodes": len(stage_a),
            "stage_b_branch_episodes": len(branch_template),
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
            "if_strong_passes": "freeze validation64 protocol with fair fixed true-H baselines, shared-H15 terminal scope, and actual overhead measurement; do not open sealed test",
            "if_weak_only": "inspect timing/false-negative pattern and run targeted representation/value/objective calibration before validation",
            "if_false_positive_or_no_saving": "block selector rollout and prioritize bounded value-refit/training/objective ablation or scenario-diagnosis rather than repeating label-density sweeps",
        },
        "access_rules": {"development_only": True, "validation64_bank_opened": False, "sealed_test_accessed": False, "requires_verified_external_backup_before_future_runner_or_simulation": True},
    }
    write_json(PROTOCOL, protocol)
    write_json(BACKUP_REQ, {
        "requested_utc": created.isoformat(),
        "reason": "backup v2 fresh-source confirmation freeze/source/protocol before any v2 H15-trace or H10/H15 branch simulations",
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
        "classification": "development_IMPROVED_metadata_fresh_source_confirmation_v2_freeze_no_simulation_no_validation_no_test",
        "hypothesis": hypothesis,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "primary_model_key": PRIMARY_MODEL_KEY,
        "primary_spec": PRIMARY_SPEC,
        "primary_terminal_profile": PRIMARY_TERMINAL_PROFILE,
        "v2c_prerequisite_summary": {"headline": v2c["completed"].get("headline"), "decision": v2c["completed"].get("decision"), "bank_label_summary": v2c.get("bank_label_summary")},
        "stage1_selected_indices_excluded": stage1_selected,
        "oracle_or_risk_source_candidate_indices_excluded": oracle_risk_used,
        "prior_fresh_source_indices_excluded": prior_fresh,
        "all_excluded_source_candidate_indices": excluded,
        "selected_cases": selected_cases,
        "case_selection_diagnostics": case_diag,
        "stage_A_schedule": stage_a,
        "stage_B_branch_template": branch_template,
        "budget_declared": protocol["budget_declared"],
        "protocol": rel(PROTOCOL),
        "backup_request": rel(BACKUP_REQ),
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((OUT / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    CONTINUE_STATE.write_text(f"""# Continue state after fresh-source confirmation freeze v2

UTC: {created.isoformat()}
Elapsed since first supervisor event: {(created - FIRST_EVENT).total_seconds()/3600.0:.2f} h.

Completed `{NAME}` metadata-only freeze for the v2c top candidate `{PRIMARY_MODEL_KEY}`. No simulation, no selector refit, no training, no validation64, no sealed test.

Primary scope: shared-H15 terminal. Matched-terminal is secondary diagnostic only because v2c showed 5 matched-terminal catastrophic false-positive rows.

Selected unused fresh-source candidate IDs: {case_diag['selected_source_candidate_indices']}. Excluded prior fresh/source/Stage1 candidate IDs: {len(excluded)} total, including fresh_v0/v1 IDs {prior_fresh}.

Future run budget: {total_eps} episodes = {len(stage_a)} Stage-A H15 traces + {len(branch_template)} Stage-B branch episodes; cap {control_cap} control steps; no training/refit.

Next action: obtain verified external backup covering this freeze and the v2c outputs, then run a versioned v2 fresh-source confirmation runner for this protocol. Do not open validation64 or sealed test. If v2 fails/weak, move to terminal-value/objective/representation or bounded training/refit ablation.

Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`, `{rel(PROTOCOL)}`.
Backup request: `{rel(BACKUP_REQ)}`.
""", encoding="utf-8")
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H fresh-source confirmation freeze v2

UTC: {created.isoformat()}. Metadata-only/no-simulation freeze for v2c top candidate `{PRIMARY_MODEL_KEY}` under shared-H15 terminal primary scope. Selected unused fresh-source candidate IDs {case_diag['selected_source_candidate_indices']}; total future budget {total_eps} episodes / cap {control_cap} control steps; no validation64/sealed test/training/refit. Matched-terminal remains secondary because v2c showed robustness failures. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`, protocol `{rel(PROTOCOL)}`.
""")
    files = [SOURCE, PROTOCOL, OUT / "raw.json", OUT / "summary.md", STATE, CONTINUE_STATE, BACKUP_REQ, V2C_DONE, V2C_PROTOCOL, V0_DONE, V1_DONE]
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "classification": raw["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "future_episodes_exact": total_eps,
        "future_control_step_upper_bound": control_cap,
        "headline": {"primary_model_key": PRIMARY_MODEL_KEY, "primary_terminal_profile": PRIMARY_TERMINAL_PROFILE, "selected_source_candidate_indices": case_diag["selected_source_candidate_indices"], "prior_fresh_source_indices_excluded": prior_fresh, "requires_backup_before_runner": True},
        "decision": "freeze complete; obtain external backup, then run v2 independent fresh-source confirmation before any validation64/test access",
        "backup_request": rel(BACKUP_REQ),
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": completed["headline"], "new_simulations": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
