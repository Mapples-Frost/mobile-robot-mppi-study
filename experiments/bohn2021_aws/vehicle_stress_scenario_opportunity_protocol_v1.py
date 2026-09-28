#!/usr/bin/env python3
"""Freeze Vehicle stress-scenario opportunity protocol v1.

This is a bounded metadata-only diagnostic after the Stage2 terminal-mode broad
scan.  It uses already-created development artifacts to (i) quantify the sparse
feature pattern behind the one robust stress-v0 positive state and (ii) freeze a
source-supported stress-v1 opportunity-map protocol.  It runs no vehicle control
rollouts, opens no historical validation64 bank, opens no sealed final test, and
performs no training/refit/gradient update.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T2025Z"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v1_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_stress_scenario_opportunity_protocol_v1_{STAMP}.md"
PROTO_JSON = ROOT / f"research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_{STAMP}.json"
PROTO_MD = ROOT / f"research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_OPPORTUNITY_PROTOCOL_V1_{STAMP}.json"
MARKER = f"vehicle-stress-scenario-opportunity-protocol-v1-{STAMP}"

INPUTS = {
    "broad_summary": ROOT / "research_artifacts/aws_diagnostics/vehicle_stage2_terminal_mode_broad_scan_v0_20260928T1915Z/summary.md",
    "broad_completed": ROOT / "research_artifacts/aws_diagnostics/vehicle_stage2_terminal_mode_broad_scan_v0_20260928T1915Z/completed.json",
    "stage1_summary": ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_postdiagnostic_20260928T1730Z/summary.md",
    "stage1_post_raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_postdiagnostic_20260928T1730Z/raw.json",
    "stage1_raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_20260928/raw.json",
    "capability_summary": ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/summary.md",
    "reward_timing_summary": ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_reward_timing_terminal_audit_v0_20260928T1820Z/summary.md",
    "terminal_source_summary": ROOT / "research_artifacts/aws_diagnostics/vehicle_terminal_objective_source_audit_v0_20260928T1830Z/summary.md",
}


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")).hexdigest()


def verify_completed(path: Path) -> Dict[str, Any]:
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise RuntimeError(f"completed marker did not pass: {rel(path)}")
    if obj.get("historical_validation64_bank_opened") is True or obj.get("sealed_test_accessed") is True:
        raise RuntimeError(f"input has forbidden access flag: {rel(path)}")
    # Hash lists in large completed markers are evidence, but some earlier tools
    # hash many per-episode files.  Verify only local existence here to keep this
    # no-simulation diagnostic bounded; full hashes remain in the source marker.
    return obj


def must_exist(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"missing required input: {rel(path)}")


def parse_backticked_counts(text: str, label: str) -> Optional[List[int]]:
    pat = re.compile(re.escape(label) + r"[^`]*`([0-9]+)`\s*/\s*`([0-9]+)`\s*/\s*`([0-9]+)`")
    m = pat.search(text)
    if not m:
        return None
    return [int(m.group(1)), int(m.group(2)), int(m.group(3))]


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def mean(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def compute_stage1_feature_diagnostic(stage1_raw: Mapping[str, Any], stage1_post: Mapping[str, Any]) -> Dict[str, Any]:
    selected = (stage1_raw.get("bank_selection") or {}).get("selected_metadata") or []
    by_case_meta = {int(m.get("selected_case_index", i)): dict(m) for i, m in enumerate(selected)}
    diagnostics = stage1_post.get("case_diagnostics") or []
    rows: List[Dict[str, Any]] = []
    for d in diagnostics:
        case = int(d.get("case"))
        meta = by_case_meta.get(case, {})
        ref = d.get("ref_H15") or {}
        best_total = d.get("best_branch_total") or {}
        best_phys = d.get("best_branch_physical") or {}
        row = {
            "case": case,
            "candidate_index": int(d.get("candidate_index", meta.get("candidate_index", -1))),
            "group": d.get("group", meta.get("selection_group")),
            "material_non_H15_positive": bool(d.get("material_non_H15_positive")),
            "theta_r": safe_float(meta.get("theta_r")),
            "abs_theta_r": abs(safe_float(meta.get("theta_r"))) if math.isfinite(safe_float(meta.get("theta_r"))) else float("nan"),
            "traj_steps": int(meta.get("traj_steps", 0) or 0),
            "min_reference_obstacle_clearance": safe_float(meta.get("min_reference_obstacle_clearance")),
            "stress_flags_count": int(d.get("stress_flags_count", meta.get("stress_flags_count", 0) or 0)),
            "best_total_horizon": int(best_total.get("horizon", 15)),
            "best_total_gain_vs_H15": safe_float(best_total.get("gain_vs_H15"), 0.0),
            "best_physical_horizon": int(best_phys.get("horizon", 15)),
            "best_physical_gain_vs_H15": safe_float(best_phys.get("gain_vs_H15"), 0.0),
            "h15_physical": safe_float(ref.get("physical_constraint_cost"), 0.0),
            "h15_total": safe_float(ref.get("total_cost"), 0.0),
        }
        rows.append(row)
    positives = [r for r in rows if r["material_non_H15_positive"]]
    nonpositives = [r for r in rows if not r["material_non_H15_positive"]]
    high_heading_long = [r for r in rows if r["abs_theta_r"] >= 0.55 and r["traj_steps"] >= 88]
    low_clearance = [r for r in rows if math.isfinite(r["min_reference_obstacle_clearance"]) and r["min_reference_obstacle_clearance"] <= -0.7]
    top_nonmaterial_phys = sorted(nonpositives, key=lambda r: r["best_physical_gain_vs_H15"], reverse=True)[:5]
    return {
        "rows": rows,
        "positive_cases": [r["case"] for r in positives],
        "positive_count": len(positives),
        "selected_case_count": len(rows),
        "positive_metadata": positives,
        "nonpositive_summary": {
            "count": len(nonpositives),
            "mean_abs_theta_r": mean(r["abs_theta_r"] for r in nonpositives),
            "mean_traj_steps": mean(r["traj_steps"] for r in nonpositives),
            "mean_clearance": mean(r["min_reference_obstacle_clearance"] for r in nonpositives),
        },
        "high_heading_long_cases": [r["case"] for r in high_heading_long],
        "high_heading_long_positive_cases": [r["case"] for r in high_heading_long if r["material_non_H15_positive"]],
        "low_clearance_cases": [r["case"] for r in low_clearance],
        "low_clearance_positive_cases": [r["case"] for r in low_clearance if r["material_non_H15_positive"]],
        "top_nonmaterial_physical_gains": top_nonmaterial_phys,
        "interpretation": "Exploratory only: N=12 stress-v0 cases, one material positive. The signature supports testing high-heading/long-transient strata, but it is not enough to train or to claim a predictive rule.",
    }


def build_protocol(created: str, broad_counts: Mapping[str, Any], feature_diag: Mapping[str, Any], hashes: Mapping[str, str]) -> Dict[str, Any]:
    protocol: Dict[str, Any] = {
        "protocol_id": f"vehicle_stress_scenario_opportunity_probe_v1_frozen_{STAMP}",
        "created_utc": created,
        "classification": "development_IMPROVED_source_supported_stress_scenario_opportunity_diagnostic_not_original_SAC_not_validation_not_final_test",
        "status": "frozen_protocol_no_simulation; rollout_requires_external_backup_and_separate_runner_dryrun",
        "motivation": {
            "broad_scan_result": "Stage2 terminal-mode broad scan found material positives and terminal-mode flips in only one state/case, so immediate selector/terminal refit would overfit one mined state.",
            "scenario_hypothesis": "Stress-v0 episode map showed a real but localized case5/early-transient opportunity.  Source-supported high-heading/long-distance and obstacle-proximity strata should be tested on fresh metadata-selected cases before any training/refit.",
            "why_not_training_now": "Current matched labels are too sparse and terminal-sensitive.  Training/refit is gated on denser, multicase matched-state labels after stress-v1 Stage1/Stage2.",
        },
        "access_rules": {
            "development_only": True,
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "final_test_authorized": False,
            "do_not_modify_current_frozen_validation_campaign": True,
            "do_not_use_for_final_model_selection_without_fresh_confirmation": True,
            "requires_verified_backup_before_runner_source_or_any_rollout": True,
        },
        "input_hashes": dict(hashes),
        "evidence_snapshot": {
            "stage2_broad_counts": dict(broad_counts),
            "stage1_feature_diagnostic": feature_diag,
            "leading_hypothesis": "scenario/opportunity design is the current leading bottleneck; reward/timing and terminal-source artifacts are important audit constraints but do not yet justify refit/training on stress-v0 labels.",
        },
        "scenario_generator_v1": {
            "name": "source_supported_high_heading_long_transient_vehicle_v1",
            "original_task_fidelity": "same registered straight-line TTAHMPC vehicle dynamics, obstacle generator and reset/reference mechanisms; development diagnostic distribution shift only through pre-outcome metadata stratification",
            "candidate_pool_resets": 256,
            "rng_seed": 2609289301,
            "selected_cases": 20,
            "selection_rule_pre_outcome": "Generate 256 candidate resets without control rollout.  Compute only source-supported metadata: theta_r/sign/abs(theta_r), traj_steps/goal-distance, natural obstacle clearance and clearance step from built-in TVPs, and obstacle-noise metadata.  Select fixed case IDs before any horizon scoring: 10 high-|theta_r| long/medium-long transient cases balanced by sign and clearance, 4 high-|theta_r| short cases to test the length interaction, 2 low-heading low-clearance obstacle-stress cases, and 4 lower-stress controls.  Use deterministic farthest-diversity within each stratum; if a stratum is underfilled, relax only the least central threshold and record the fallback.  Do not select by H outcome.",
            "stress_factors_used": [
                "high |theta_r| within built-in +/-45 degree straight-line reference sampling",
                "long and medium-long traj_steps / goal distance",
                "low or moderate natural obstacle clearance from the built-in three-obstacle generator",
                "balanced theta_r sign to avoid fitting the one negative-heading positive case only",
                "control strata with lower heading and mixed length/clearance"
            ],
            "explicitly_excluded_factors": [
                "curved paths",
                "direct initial speed changes",
                "plant process noise/disturbance injection",
                "robust MPC uncertainty/n_robust",
                "manual outcome-based obstacle placement",
                "any validation64 or sealed-test case"
            ],
        },
        "stage1_fixed_H_opportunity_map": {
            "purpose": "test whether source-supported stress-v1 metadata strata create denser multicase fixed-H control/timing tradeoffs against strong constant-H baselines",
            "horizons": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "terminal_value": "independent seed0 fixed-H terminal for each H; abort rather than silently fallback if any source is unavailable",
            "episodes_exact": 200,
            "control_step_upper_bound": 30000,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "actual_timing_recorded": True,
            "report_separately": ["physical/control cost", "total cost including synthetic h_penalty", "success", "constraint/solver failures", "episode length", "whole-decision wall time", "solver-attempt wall time"],
            "stage1_materiality_gate_for_stage2": {
                "case_diversity": ">=3 distinct selected cases with safe material non-H15 fixed-H improvement versus that case's H15 or strongest same-bank reference; not all from one metadata stratum",
                "material_gain": "physical or total continuation/episode gain >=3 absolute units or >=5 percent, with no success/constraint/solver regression",
                "aggregate_guard": "same-bank oracle must improve over best all-success no-constraint fixed-H reference by >=5 absolute physical or >=3 absolute total, or show a non-dominated physical-vs-measured-time frontier with >=3 horizons across cases",
                "h5_handling": "H5 retained as a strong-baseline/safety arm but cannot count as a positive adaptive label if it fails or terminates early",
            },
        },
        "stage2_matched_continuation_if_stage1_passes": {
            "freeze_separately_before_rollout": True,
            "purpose": "distinguish episode-level fixed-H variation from true within-episode adaptive switching opportunity using identical prefixes/states",
            "state_selection_rule_pre_branch_outcome": "Select states from H15 and/or same-bank best-reference traces by metadata only: early high heading-error/transient windows, near-obstacle/low-clearance windows, post-transient controls, and low-stress controls.  Do not choose states using non-H15 branch outcomes.",
            "branch_horizons_initial": [10, 15, 25, 30, 35, 45, 50],
            "terminal_modes_to_audit": ["per_h", "h15_terminal", "zero_terminal"],
            "budget_upper_bound_initial": {"states": 16, "episodes": 16 * (1 + 6 * 2), "control_steps": 16 * (1 + 6 * 2) * 150},
            "label_gate_for_selector_refit": ">=3 distinct cases and >=6 states with material non-reference positives under a predeclared terminal-mode rule, plus retained negative/control states; no prefix/hash/solver/safety artifact explains the labels",
        },
        "training_or_refit_gate_after_stage2": {
            "if_pass": "freeze one IMPROVED selector/refit intervention including longer horizons supported by labels (e.g. H10/H15/H25/H30/H35/H45/H50), with physical/timing Pareto objective and fair same-distribution fixed-H baselines",
            "if_fail": "do not train on sparse labels; preserve negative evidence and report weak adaptive opportunity for current vehicle scenario family or design a stronger source-supported scenario protocol only with a new hypothesis",
            "do_not_do": ["reward switching for appearance", "best-seed-only selection", "post-hoc scalar reweighting", "weakening fixed-H baselines", "using sealed final test for development"],
        },
        "four_axis_decision_table": {
            "SCENARIOS": {
                "verified": "stress-v0 and terminal broad scan found real but single-case/single-state opportunity; canonical/fresh probes were sparse",
                "hypothesis": "opportunity may require high-heading long-transient source-supported strata and is underrepresented in v0/canonical banks",
                "missing": "fresh stress-v1 fixed-H map and then matched-state continuation if multicase episode-level diversity appears",
                "next_discriminator": "stage1 stress-v1 fixed-H map after backup and runner dry-run"
            },
            "REWARD": {
                "verified": "synthetic h_penalty is not measured runtime; positives are physical-driven locally; timing is noisy/nonmonotone",
                "hypothesis": "scalar total cost can misstate compute-performance tradeoff",
                "missing": "paired measured-time data on any future selector/baseline",
                "next_discriminator": "stress-v1 reports physical and measured timing separately; no scalar acceleration claim"
            },
            "TRAINING": {
                "verified": "current policies are finite search/reselection with zero gradient updates and short-only class; labels are too sparse for refit",
                "hypothesis": "training/refit helps only after denser multicase labels exist",
                "missing": "matched multicase labels including longer horizons",
                "next_discriminator": "no training now; Stage1/Stage2 gates decide whether to freeze an IMPROVED selector/refit"
            },
            "COMPARISONS": {
                "verified": "strong same-bank fixed-H grids with H-specific terminals exist; terminal modes can flip local labels",
                "hypothesis": "apparent adaptive opportunity may be absorbed by tuned fixed-H/terminal choices",
                "missing": "same-distribution fixed-H baselines and terminal-mode controls on stress-v1",
                "next_discriminator": "full H5..H50 Stage1 grid and terminal-mode-audited Stage2 before selector claims"
            }
        },
        "budget": {
            "this_protocol_freeze_new_rollouts": 0,
            "this_protocol_freeze_control_steps": 0,
            "this_protocol_freeze_training_episodes": 0,
            "this_protocol_freeze_gradient_steps": 0,
            "future_stage1_rollout_episodes_exact": 200,
            "future_stage1_control_step_upper_bound": 30000,
            "future_validation64_episodes": 0,
            "future_sealed_test_episodes": 0,
        },
    }
    protocol["protocol_sha256_without_self"] = canonical_hash({k: v for k, v in protocol.items() if k != "protocol_sha256_without_self"})
    return protocol


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    for p in INPUTS.values():
        must_exist(p)
    broad_done = verify_completed(INPUTS["broad_completed"])
    broad_text = INPUTS["broad_summary"].read_text(encoding="utf-8")
    stage1_text = INPUTS["stage1_summary"].read_text(encoding="utf-8")
    stage1_post = read_json(INPUTS["stage1_post_raw"])
    stage1_raw = read_json(INPUTS["stage1_raw"])
    if stage1_post.get("historical_validation64_bank_opened") is not False or stage1_post.get("sealed_test_accessed") is not False:
        raise RuntimeError("stage1 postdiagnostic access flags invalid")
    if stage1_raw.get("historical_validation64_bank_opened") is not False or stage1_raw.get("sealed_test_accessed") is not False:
        raise RuntimeError("stage1 raw access flags invalid")

    broad_counts = {
        "material_positive_rows_states_cases": parse_backticked_counts(broad_text, "Material positive rows/states/cases:") or [],
        "terminal_label_flips_rows_states_cases": parse_backticked_counts(broad_text, "Terminal label flips rows/states/cases:") or [],
        "dense_terminal_artifact_gate_pass": "Dense terminal/artifact gate pass: `True`" in broad_text,
        "local_only_case5_step18": "target00_case5_step18" in broad_text and "Material positive cases: `[5]`" in broad_text,
        "episodes": broad_done.get("episodes"),
        "control_steps": broad_done.get("control_steps"),
    }
    feature_diag = compute_stage1_feature_diagnostic(stage1_raw, stage1_post)
    hashes = {rel(p): sha256(p) for p in INPUTS.values()}
    protocol = build_protocol(created, broad_counts, feature_diag, hashes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(PROTO_JSON, protocol)
    md_lines = [
        "# Vehicle stress-scenario opportunity probe v1 frozen protocol",
        "",
        f"Frozen UTC: `{created}`. Development-only IMPROVED scenario diagnostic; no validation64-bank access and no sealed-test access.",
        "",
        "## Why v1, and why not training yet",
        "",
        "The completed Stage2 terminal-mode broad scan found material positives/flips in only one case/state, so a selector or terminal refit would mostly learn one mined development event. Stress-v1 instead tests a source-supported scenario-opportunity hypothesis on fresh metadata-selected cases before any training/refit.",
        "",
        "## Exploratory metadata signal from stress-v0",
        "",
        f"- Material non-H15 positive cases in stress-v0 Stage1/Stage2 evidence: `{feature_diag['positive_cases']}` out of `{feature_diag['selected_case_count']}` selected cases.",
        f"- High-heading long cases in stress-v0: `{feature_diag['high_heading_long_cases']}`; positive subset: `{feature_diag['high_heading_long_positive_cases']}`.",
        f"- Low-clearance cases in stress-v0: `{feature_diag['low_clearance_cases']}`; positive subset: `{feature_diag['low_clearance_positive_cases']}`.",
        "- This is hypothesis-generating only, not a predictive rule or training label set.",
        "",
        "## Frozen stress-v1 design",
        "",
        f"- Candidate pool: `{protocol['scenario_generator_v1']['candidate_pool_resets']}` fresh metadata-only resets; selected cases: `{protocol['scenario_generator_v1']['selected_cases']}`.",
        "- Selection emphasizes high-|theta_r| long/medium-long transients, includes high-|theta_r| short controls, obstacle-only stress controls, and lower-stress controls; all case IDs are fixed before any H scoring.",
        f"- Stage1 H grid: `{protocol['stage1_fixed_H_opportunity_map']['horizons']}`; episodes/cap: `{protocol['stage1_fixed_H_opportunity_map']['episodes_exact']}` / `{protocol['stage1_fixed_H_opportunity_map']['control_step_upper_bound']}`.",
        "- Stage2 matched continuation is conditional and must be frozen separately.",
        "",
        "## Gates",
        "",
        "```json",
        json.dumps({
            "stage1_gate": protocol["stage1_fixed_H_opportunity_map"]["stage1_materiality_gate_for_stage2"],
            "stage2_gate": protocol["stage2_matched_continuation_if_stage1_passes"]["label_gate_for_selector_refit"],
            "training_gate": protocol["training_or_refit_gate_after_stage2"],
        }, indent=2, sort_keys=True),
        "```",
        "",
        "## Backup",
        "",
        f"Runner source/dry-run or any rollout is blocked until a verified external backup covers this protocol freeze and docs. Backup request: `{rel(BACKUP_REQ)}`.",
    ]
    PROTO_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    raw = {
        "created_utc": created,
        "method": "vehicle_stress_scenario_opportunity_protocol_v1_metadata_diagnostic_freeze",
        "classification": "development_metadata_diagnostic_no_rollouts_no_training_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "input_hashes": hashes,
        "broad_counts": broad_counts,
        "stage1_feature_diagnostic": feature_diag,
        "frozen_protocol": rel(PROTO_JSON),
        "frozen_protocol_md": rel(PROTO_MD),
        "decision": "backup_then_write_dryrun_stress_v1_runner; no selector_refit_or_training_until_stress_v1_stage1_and_conditional_stage2_pass_multicase_label_gates",
    }
    write_json(OUT_DIR / "raw.json", raw)
    summary = "\n".join([
        "# Vehicle stress-scenario opportunity protocol v1 metadata diagnostic",
        "",
        f"UTC: `{created}`. No simulations/control steps/training; validation64 and sealed test remained closed.",
        "",
        "## Result",
        "",
        "Frozen stress-v1 protocol because broad terminal-mode labels remain local-only and current stress-v0 labels are too sparse for selector/refit.",
        "",
        f"- Broad scan material positive rows/states/cases: `{broad_counts['material_positive_rows_states_cases']}`; dense terminal/artifact gate: `{broad_counts['dense_terminal_artifact_gate_pass']}`.",
        f"- Stress-v0 material positive cases: `{feature_diag['positive_cases']}` / `{feature_diag['selected_case_count']}`.",
        f"- Positive metadata snapshot: `{feature_diag['positive_metadata']}`.",
        "- Next after backup: write/import-smoke a stress-v1 runner, then execute a 200-episode fixed-H map only after the runner/source backup gate is satisfied.",
        "",
        f"Protocol: `{rel(PROTO_MD)}` / `{rel(PROTO_JSON)}`.",
        f"Backup request: `{rel(BACKUP_REQ)}`.",
    ]) + "\n"
    (OUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle stress-v1 protocol state ({created})\n\n"
        "Metadata diagnostic and protocol freeze completed with zero rollouts/training and no validation64/test access. "
        "Next: require verified backup, then write/dry-run a stress-v1 fixed-H runner. Do not train/refit until stress-v1 Stage1 plus separately frozen matched Stage2 produce multicase robust labels.\n",
        encoding="utf-8",
    )
    write_json(BACKUP_REQ, {
        "requested_utc": created,
        "reason": "backup stress-v1 metadata diagnostic/protocol freeze before any runner source or further simulation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(PROTO_JSON), rel(PROTO_MD), rel(BACKUP_REQ), rel(Path(__file__).resolve())],
    })
    block = f"""<!-- {MARKER} -->
## 2026-09-28 vehicle stress-scenario opportunity protocol v1

UTC: {created}. Metadata-only diagnostic/protocol freeze completed after broad terminal-mode scan showed positives/flips local to case5 step18. No rollouts, control steps, training, validation64-bank access or sealed-test access. Frozen stress-v1 uses 256 metadata-only candidate resets, 20 selected source-supported cases, full fixed-H grid H5..H50 (200 episodes / 30,000-step cap) and a separate matched-continuation gate before any selector/refit. Immediate decision: no training/refit on current sparse labels; require backup before runner source/smoke or simulation. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(PROTO_MD)}`, `{rel(PROTO_JSON)}`.
"""
    append_docs(block)
    files = [OUT_DIR / "raw.json", OUT_DIR / "summary.md", STATE_PATH, PROTO_JSON, PROTO_MD, BACKUP_REQ, Path(__file__).resolve()] + list(INPUTS.values())
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
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
        "frozen_protocol_md": rel(PROTO_MD),
        "headline": {
            "decision": raw["decision"],
            "future_stage1_episodes_exact": protocol["stage1_fixed_H_opportunity_map"]["episodes_exact"],
            "future_stage1_control_step_upper_bound": protocol["stage1_fixed_H_opportunity_map"]["control_step_upper_bound"],
            "training_or_refit_now": False,
            "broad_dense_terminal_artifact_gate_pass": broad_counts["dense_terminal_artifact_gate_pass"],
            "stress_v0_positive_cases": feature_diag["positive_cases"],
        },
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "frozen_protocol": rel(PROTO_JSON),
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(BACKUP_REQ),
        "next": "backup_then_stress_v1_runner_dryrun_then_200_episode_fixed_H_map",
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
