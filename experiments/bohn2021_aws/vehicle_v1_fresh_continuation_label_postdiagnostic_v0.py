#!/usr/bin/env python3
"""Postdiagnostic after Vehicle V1 fresh continuation-label probe v0.

No-simulation diagnostic.  Reads the completed fresh continuation-label probe and
prior development summaries, quantifies why the fresh label/refit gate failed,
and freezes the next bounded transient-state continuation diagnostic.  It does
not construct environments, run rollouts, train/refit models, open the historical
validation64 bank, or access the sealed final test.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T1500Z"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_v1_fresh_continuation_label_postdiagnostic_v0_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
BACKUP_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_V1_FRESH_CONTINUATION_LABEL_POSTDIAGNOSTIC_V0_{STAMP}.json"
NEXT_PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.json"
NEXT_PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.md"

FRESH_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928/raw.json"
FRESH_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928/completed.json"
FRESH_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928/summary.md"
V1_POST_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_postdiagnostic_20260928T1150Z/completed.json"
BROAD_POST_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_selector_broad_postdiagnostic_v0_20260928T1345Z/completed.json"
BROAD_POST_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_selector_broad_postdiagnostic_v0_20260928T1345Z/raw.json"
CONTROLLED_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928/completed.json"
CONTROLLED_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928/raw.json"
MARKER = "vehicle-v1-fresh-continuation-label-postdiagnostic-v0-20260928T1500Z"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def require_completed(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True or obj.get("hard_pass") is not True:
        raise RuntimeError(f"completed marker is not hard-pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is not False:
        raise RuntimeError(f"sealed test flag is not false in {rel(path)}")
    # Historical-validation64 may be absent on some old markers, but if present it must be false.
    if obj.get("historical_validation64_bank_opened") not in (None, False):
        raise RuntimeError(f"historical validation64 flag is not false in {rel(path)}")
    return obj


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def values(xs: Iterable[float]) -> Dict[str, Any]:
    data = [float(x) for x in xs if math.isfinite(float(x))]
    if not data:
        return {"count": 0, "min": None, "max": None, "mean": None, "sum": 0.0}
    return {
        "count": len(data),
        "min": min(data),
        "max": max(data),
        "mean": sum(data) / len(data),
        "sum": sum(data),
    }


def comparison_stats(state_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_h: Dict[str, Dict[str, Any]] = {}
    best_phys_counts: Dict[str, int] = {}
    best_total_counts: Dict[str, int] = {}
    non_h15_candidates: List[Dict[str, Any]] = []
    large_harms: List[Dict[str, Any]] = []
    prefix_hash_mismatches_state_identical = 0
    prefix_hash_matches = 0
    state_distance_values: List[float] = []
    for row in state_rows:
        if not row.get("informative"):
            continue
        bp = row.get("best_physical") or {}
        bt = row.get("best_total") or {}
        if bp.get("horizon") is not None:
            best_phys_counts[str(bp["horizon"])] = best_phys_counts.get(str(bp["horizon"]), 0) + 1
        if bt.get("horizon") is not None:
            best_total_counts[str(bt["horizon"])] = best_total_counts.get(str(bt["horizon"]), 0) + 1
        for comp in row.get("comparisons", []):
            h = str(comp.get("horizon"))
            d = by_h.setdefault(h, {"physical_gains": [], "total_gains": [], "material_count": 0, "success_count": 0, "constraint_count": 0, "solver_failure_steps": 0, "initial_failed_steps": 0, "large_harm_count": 0})
            pg = safe_float(comp.get("gain_vs_H15_physical"))
            tg = safe_float(comp.get("gain_vs_H15_total"))
            d["physical_gains"].append(pg)
            d["total_gains"].append(tg)
            d["material_count"] += int(bool(comp.get("material_positive_vs_H15")))
            d["success_count"] += int(bool(comp.get("success")))
            d["constraint_count"] += int(bool(comp.get("constraint")))
            d["solver_failure_steps"] += int(comp.get("solver_failure_steps", 0) or 0)
            d["initial_failed_steps"] += int(comp.get("initial_failed_steps", 0) or 0)
            if int(comp.get("horizon", 15)) != 15:
                non_h15_candidates.append({
                    "case": int(row.get("case")),
                    "branch_step": int(row.get("branch_step")),
                    "horizon": int(comp.get("horizon")),
                    "physical_gain": pg,
                    "total_gain": tg,
                    "success": bool(comp.get("success")),
                    "safe": bool(comp.get("no_success_constraint_solver_regression_vs_H15")),
                    "path": comp.get("path"),
                })
                if pg <= -3.0 or tg <= -3.0:
                    d["large_harm_count"] += 1
                    large_harms.append({
                        "case": int(row.get("case")),
                        "branch_step": int(row.get("branch_step")),
                        "horizon": int(comp.get("horizon")),
                        "physical_gain": pg,
                        "total_gain": tg,
                    })
                dist = safe_float(comp.get("state_distance_vs_H15_branch_state"))
                if math.isfinite(dist):
                    state_distance_values.append(dist)
                    if dist == 0.0 and comp.get("prefix_clean_sha256_matches_H15") is False:
                        prefix_hash_mismatches_state_identical += 1
                    if comp.get("prefix_clean_sha256_matches_H15") is True:
                        prefix_hash_matches += 1
    horizon_stats: Dict[str, Any] = {}
    for h, d in sorted(by_h.items(), key=lambda kv: int(kv[0])):
        horizon_stats[h] = {
            "states": len(d["physical_gains"]),
            "success_count": d["success_count"],
            "constraint_count": d["constraint_count"],
            "initial_failed_steps": d["initial_failed_steps"],
            "solver_failure_steps": d["solver_failure_steps"],
            "material_count": d["material_count"],
            "large_harm_count_vs_H15_threshold3": d["large_harm_count"],
            "physical_gain_vs_H15": values(d["physical_gains"]),
            "total_gain_vs_H15": values(d["total_gains"]),
        }
    best_non_h15_by_total = sorted(non_h15_candidates, key=lambda r: (r["total_gain"], r["physical_gain"]), reverse=True)[:8]
    best_non_h15_by_physical = sorted(non_h15_candidates, key=lambda r: (r["physical_gain"], r["total_gain"]), reverse=True)[:8]
    return {
        "horizon_stats": horizon_stats,
        "best_physical_horizon_counts": best_phys_counts,
        "best_total_horizon_counts": best_total_counts,
        "max_non_H15_total_gain_candidates": best_non_h15_by_total,
        "max_non_H15_physical_gain_candidates": best_non_h15_by_physical,
        "large_harm_count": len(large_harms),
        "large_harm_examples": large_harms[:12],
        "prefix_hash_mismatches_state_identical_count": prefix_hash_mismatches_state_identical,
        "prefix_hash_matches_count": prefix_hash_matches,
        "branch_state_distance_vs_H15": values(state_distance_values),
    }


def metadata_stats(raw: Mapping[str, Any], state_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = raw.get("bank_selection", {}).get("selected_metadata", [])
    branch_ratios: List[float] = []
    for row in state_rows:
        if not row.get("informative"):
            continue
        case = int(row["case"])
        if 0 <= case < len(selected):
            traj = safe_float(selected[case].get("traj_steps"))
            if traj > 0:
                branch_ratios.append(float(row["branch_step"]) / traj)
    return {
        "selected_case_count": len(selected),
        "theta_r": values(safe_float(m.get("theta_r")) for m in selected),
        "abs_theta_r": values(abs(safe_float(m.get("theta_r"))) for m in selected),
        "traj_steps": values(safe_float(m.get("traj_steps")) for m in selected),
        "min_reference_obstacle_clearance": values(safe_float(m.get("min_reference_obstacle_clearance")) for m in selected),
        "branch_step_ratio_to_traj_steps": values(branch_ratios),
        "strata_counts": {stratum: sum(1 for m in selected if m.get("stratum") == stratum) for stratum in sorted({m.get("stratum") for m in selected})},
    }


def freeze_next_protocol(created: str, raw: Mapping[str, Any], stats: Mapping[str, Any]) -> None:
    protocol = {
        "protocol_id": "vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928",
        "created_utc": created,
        "classification": "development_IMPROVED_transient_state_continuation_diagnostic_not_validation_not_final_test",
        "motivation": [
            "The hand-mined V1 nearest-state selector only gained on previously mined cases 7 and 10.",
            "The fresh metadata-balanced label probe found zero material non-H15 positives across 24 generic early/mid H15-prefix branch states.",
            "The next missing evidence is whether generic branch-step selection simply missed high-error/near-obstacle transients, or whether canonical V1 opportunity is truly too sparse for broad refit/training.",
        ],
        "inputs_allowed_for_state_selection": {
            "fresh_probe_bank": rel(FRESH_RAW),
            "allowed_trace_fields": [
                "H15-only traces already produced in the fresh probe",
                "state/reference/obstacle-distance/tracking-error/time-step metadata",
                "success/safety of the H15 reference arm",
            ],
            "forbidden_for_state_selection": [
                "non-H15 branch outcomes from the fresh probe",
                "historical validation64 outcomes",
                "sealed final test outcomes",
            ],
            "note": "Non-H15 fresh-probe outcomes are already development-known, so this remains diagnostic development work only; the selection algorithm must nevertheless ignore them to isolate missed-transient timing from hindsight label mining.",
        },
        "rollout_design_after_backup": {
            "source_cases": "reuse the 12 completed fresh-probe development cases; no new candidate resets in v0",
            "state_selection_rule": "from one H15 reference trace per case, compute a transient score from tracking-error magnitude, heading-error magnitude and proximity-to-forecast/constraint obstacles; select up to 8 high-transient states plus 4 low-transient controls, max one high and one low state per case where feasible; de-duplicate branch steps within +/-3 control steps; do not inspect non-H15 outcomes when selecting",
            "prefix_horizon": 15,
            "branch_horizons": [10, 15, 30, 35],
            "max_selected_states": 12,
            "max_rollout_episodes": 48,
            "control_step_upper_bound": 7200,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
        },
        "gate_for_next_action": {
            "transient_missed_timing_supported": "at least 2 material-positive non-H15 states, with at least one not in the previously mined V1 case7/case10 coordinates and at least 2 negative/control states retained",
            "scenario_scarcity_supported": "fewer than 2 material-positive states and no safety/solver artifact explaining the absence",
            "if_pass": "freeze a compact supervised/refit selector using transient-state features and confirm on a separate fresh development bank with strong fixed-H10/H15/H30/H35/full-grid references",
            "if_fail": "freeze a versioned source-supported stress scenario protocol (heading/length/obstacle-proximity strata) instead of retraining on sparse canonical labels",
        },
        "backup_required_before_rollout": True,
        "current_postdiagnostic_inputs": {
            rel(FRESH_RAW): sha256(FRESH_RAW),
            rel(FRESH_DONE): sha256(FRESH_DONE),
            rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        },
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
    }
    write_json(NEXT_PROTOCOL_JSON, protocol)
    lines = [
        "# Vehicle V1 transient-state continuation probe v0",
        "",
        f"Frozen UTC: `{created}`.",
        "",
        "Development-only diagnostic; not validation and not final test. Execute only after external backup covers this protocol, the fresh-probe outputs and this postdiagnostic.",
        "",
        "## Motivation",
        "",
    ]
    for item in protocol["motivation"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Frozen rollout design",
        "",
        f"- Source cases: {protocol['rollout_design_after_backup']['source_cases']}.",
        f"- State-selection rule: {protocol['rollout_design_after_backup']['state_selection_rule']}",
        "- Selection must ignore non-H15 branch outcomes and all validation/test outcomes.",
        f"- Prefix horizon: `15`; branch horizons: `{protocol['rollout_design_after_backup']['branch_horizons']}`.",
        f"- Budget ceiling: `{protocol['rollout_design_after_backup']['max_rollout_episodes']}` episodes / `{protocol['rollout_design_after_backup']['control_step_upper_bound']}` control steps; training/gradient `0/0`.",
        "",
        "## Gate",
        "",
        f"- Pass/transient-missed-timing support: {protocol['gate_for_next_action']['transient_missed_timing_supported']}.",
        f"- Fail/scenario-scarcity support: {protocol['gate_for_next_action']['scenario_scarcity_supported']}.",
        f"- If pass: {protocol['gate_for_next_action']['if_pass']}.",
        f"- If fail: {protocol['gate_for_next_action']['if_fail']}.",
        "",
        "## Access",
        "",
        "No historical validation64 bank and no sealed final test. This is development-only evidence.",
    ]
    NEXT_PROTOCOL_MD.parent.mkdir(parents=True, exist_ok=True)
    NEXT_PROTOCOL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 fresh continuation-label postdiagnostic v0\n\n"
        f"UTC: {raw['created_utc']}. No-simulation postdiagnostic of the completed fresh label probe. "
        f"Fresh probe had {raw['fresh_probe_headline']['positive_state_count']} material positive states across {raw['fresh_probe_headline']['informative_state_count']} informative states; "
        f"best non-H15 total gain was {raw['diagnostic_stats']['best_non_H15_total_gain_candidates'][0]['total_gain'] if raw['diagnostic_stats']['best_non_H15_total_gain_candidates'] else 'NA'} and no refit/training gate passed. "
        f"Next protocol frozen: `{rel(NEXT_PROTOCOL_MD)}` / `{rel(NEXT_PROTOCOL_JSON)}`; backup required before any rollout. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    fresh = raw["fresh_probe_headline"]
    stats = raw["diagnostic_stats"]
    lines = [
        "# Vehicle V1 fresh continuation-label postdiagnostic v0",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "No simulations, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Fresh-label result restated",
        "",
        f"- Fresh probe access: historical_validation64_bank_opened=`{raw['historical_validation64_bank_opened']}`, sealed_test_accessed=`{raw['sealed_test_accessed']}`.",
        f"- Informative states: `{fresh['informative_state_count']}`; material positives: `{fresh['positive_state_count']}` across cases `{fresh['positive_cases']}`; negative/neutral: `{fresh['negative_neutral_state_count']}`.",
        f"- Fresh refit/training gate: `{fresh['fresh_refit_training_gate_pass']}`.",
        "",
        "## Horizon-gain diagnostics from identical H15-prefix states",
        "",
        "| H | states | material positives | large harms vs H15 | max phys gain | max total gain | mean phys gain | mean total gain | solver-fail steps |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h, hs in stats["horizon_stats"].items():
        pg = hs["physical_gain_vs_H15"]
        tg = hs["total_gain_vs_H15"]
        lines.append("| %s | %d | %d | %d | %.6g | %.6g | %.6g | %.6g | %d |" % (
            h,
            hs["states"],
            hs["material_count"],
            hs["large_harm_count_vs_H15_threshold3"],
            pg["max"] if pg["max"] is not None else float("nan"),
            tg["max"] if tg["max"] is not None else float("nan"),
            pg["mean"] if pg["mean"] is not None else float("nan"),
            tg["mean"] if tg["mean"] is not None else float("nan"),
            hs["solver_failure_steps"],
        ))
    lines += [
        "",
        f"- Best physical horizon counts over fresh branch states: `{stats['best_physical_horizon_counts']}`.",
        f"- Best total horizon counts over fresh branch states: `{stats['best_total_horizon_counts']}`.",
        f"- Largest non-H15 total-gain candidates: `{stats['max_non_H15_total_gain_candidates'][:5]}`.",
        f"- Large non-H15 harms (gain <= -3 physical or total) count: `{stats['large_harm_count']}`; examples `{stats['large_harm_examples'][:5]}`.",
        f"- Branch-state distance vs H15 reference: `{stats['branch_state_distance_vs_H15']}`.",
        f"- Prefix hash mismatches with zero branch-state distance: `{stats['prefix_hash_mismatches_state_identical_count']}`. This is treated as instrumentation/metadata-hash mismatch, not evidence of different prefix states.",
        "",
        "## Evidence table",
        "",
        "| axis | verified finding | competing hypothesis / uncertainty | next discriminating action |",
        "|---|---|---|---|",
        "| scenarios | Fresh metadata-balanced canonical cases produced 0/24 material non-H15 continuation positives, while previous mined V1 states had 3 H30-positive points concentrated in cases 7/10. | Either canonical opportunity is sparse/case-specific, or the generic 15%/35% branch steps missed transient/high-risk states. | Run the frozen transient-state continuation probe selecting branch states from H15 trace transients, not from non-H15 outcomes. |",
        "| reward/objective | No positive disappears because of synthetic horizon penalty: physical and total gains are both far below the +3 material threshold; H10 gains are small and H30/H35 often harm physical cost. | Objective/terminal values may still bias historical learned policies, but fresh realized-continuation labels do not justify refit yet. | Keep realized physical, total, success/safety and timing separate in the transient probe; do not train on sparse/near-zero labels. |",
        "| training/selection | The nearest-state latch's development gain remains a hand-mined prototype; fresh labels are all negative/neutral, so supervised/refit training would be class-imbalanced and likely overfit. | A richer transient-feature selector may still be viable if positives exist at high-error/near-obstacle states. | Defer gradient/refit until transient-state or stress-scenario labels produce enough positive/negative coverage. |",
        "| comparisons | H15 remains the relevant strong cost comparator; H10 can be slightly better in total on some easy states but only by <1 unit in this fresh probe; no speed claim is established. | Fixed-H baselines may dominate canonical distribution; stress distribution might change the Pareto map. | Any later online selector must be paired against H10/H15/H30/H35/full-grid references with actual timing. |",
        "",
        "## Decision",
        "",
        "The fresh result weakens the earlier mined-selector evidence and argues against launching a broad selector/refit or validation campaign now. The next highest-information experiment is a bounded transient-state continuation probe: it tests whether the fresh branch schedule missed the relevant state-dependent tradeoff before moving to a versioned stress-scenario redesign.",
        "",
        f"Frozen next protocol: `{rel(NEXT_PROTOCOL_MD)}` / `{rel(NEXT_PROTOCOL_JSON)}`.",
        f"Backup request before any rollout: `{rel(BACKUP_REQUEST)}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    fresh_done = require_completed(FRESH_DONE)
    raw = read_json(FRESH_RAW)
    if raw.get("historical_validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise RuntimeError("fresh probe access flags are not closed")
    state_rows = raw.get("analysis", {}).get("state_rows", [])
    stats = comparison_stats(state_rows)
    meta = metadata_stats(raw, state_rows)
    v1_done = require_completed(V1_POST_DONE) if V1_POST_DONE.exists() else {}
    broad_done = require_completed(BROAD_POST_DONE) if BROAD_POST_DONE.exists() else {}
    controlled_done = require_completed(CONTROLLED_DONE) if CONTROLLED_DONE.exists() else {}
    freeze_next_protocol(created, raw, stats)
    fresh_headline = {
        "informative_state_count": int(raw["analysis"]["informative_state_count"]),
        "positive_state_count": int(raw["analysis"]["positive_state_count"]),
        "negative_neutral_state_count": int(raw["analysis"]["negative_neutral_state_count"]),
        "positive_cases": raw["analysis"].get("positive_cases", []),
        "fresh_refit_training_gate_pass": bool(raw["analysis"].get("fresh_refit_training_gate_pass")),
        "episodes": int(raw["budget_actual"]["episodes"]),
        "control_steps": int(raw["budget_actual"]["control_steps"]),
        "fresh_candidate_resets": int(raw["budget_actual"].get("fresh_bank_candidate_resets", 0)),
    }
    decision = {
        "ranked_hypotheses_after_fresh_probe": [
            {"rank": 1, "hypothesis": "Canonical-source V1 adaptive opportunity is sparse and concentrated in particular obstacle/heading transients; generic fresh branch states are mostly H15-dominant or near-neutral.", "evidence": "0/24 fresh material positives; prior material positives only in mined cases 7/10; H30/H35 often large harms."},
            {"rank": 2, "hypothesis": "The fresh branch schedule missed high-transient states despite using 15%/35% trajectory steps.", "evidence": "Not yet directly tested; branch states were chosen from reset traj_steps, not from observed tracking/obstacle transient scores."},
            {"rank": 3, "hypothesis": "Historical training/value/objective mismatch caused near-constant policies, but retraining now would be premature because fresh labels contain no positive class.", "evidence": "Mined latch can exploit exact known states; fresh label gate failed."},
            {"rank": 4, "hypothesis": "Implementation/prefix mismatch explains the fresh failure.", "evidence": "Branch-state distances vs H15 are zero; prefix-clean hash mismatch is attributable to metadata-hash design, not state divergence; all episodes succeeded without solver-failure steps."},
        ],
        "next_action": "after verified backup, run vehicle_v1_transient_state_continuation_probe_v0; if that also lacks positives, freeze a versioned source-supported stress-scenario opportunity protocol rather than retraining or validating the mined selector",
        "why_not_retrain_now": "fresh labels have zero material positive states, so any selector/value refit would be class-imbalanced and would mostly learn constant H15/H10 near-neutral choices",
        "why_not_more_unchanged_validation": "broad selector gain is concentrated in previously mined development cases and fresh generalization probe failed",
    }
    out = {
        "created_utc": created,
        "method": "vehicle_v1_fresh_continuation_label_postdiagnostic_v0_no_simulation",
        "formal_scientific_evidence": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "inputs": {
            rel(FRESH_RAW): sha256(FRESH_RAW),
            rel(FRESH_DONE): sha256(FRESH_DONE),
            rel(FRESH_SUMMARY): sha256(FRESH_SUMMARY) if FRESH_SUMMARY.exists() else None,
            rel(V1_POST_DONE): sha256(V1_POST_DONE) if V1_POST_DONE.exists() else None,
            rel(BROAD_POST_DONE): sha256(BROAD_POST_DONE) if BROAD_POST_DONE.exists() else None,
            rel(BROAD_POST_RAW): sha256(BROAD_POST_RAW) if BROAD_POST_RAW.exists() else None,
            rel(CONTROLLED_DONE): sha256(CONTROLLED_DONE) if CONTROLLED_DONE.exists() else None,
            rel(CONTROLLED_RAW): sha256(CONTROLLED_RAW) if CONTROLLED_RAW.exists() else None,
            rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        },
        "fresh_completed_headline": fresh_done.get("headline", {}),
        "v1_postdiagnostic_headline": v1_done.get("headline", {}),
        "broad_selector_postdiagnostic_headline": broad_done.get("headline", {}),
        "controlled_continuation_headline": controlled_done.get("headline", {}),
        "fresh_probe_headline": fresh_headline,
        "fresh_metadata_stats": meta,
        "diagnostic_stats": stats,
        "decision": decision,
        "next_protocol": {"json": rel(NEXT_PROTOCOL_JSON), "json_sha256": sha256(NEXT_PROTOCOL_JSON), "md": rel(NEXT_PROTOCOL_MD), "md_sha256": sha256(NEXT_PROTOCOL_MD)},
    }
    write_summary(out)
    write_json(OUT_DIR / "raw.json", out)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    write_json(BACKUP_REQUEST, {
        "requested_utc": created,
        "reason": "backup fresh-label postdiagnostic and frozen transient-state continuation protocol before any further rollout/simulation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(NEXT_PROTOCOL_MD), rel(NEXT_PROTOCOL_JSON), rel(BACKUP_REQUEST), rel(Path(__file__).resolve())],
    })
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 fresh continuation-label postdiagnostic v0 state ({created})\n\n"
        f"No simulations/training/validation64/test access. Fresh probe: {fresh_headline['positive_state_count']}/{fresh_headline['informative_state_count']} material positives, gate={fresh_headline['fresh_refit_training_gate_pass']}. "
        f"Next: after verified backup, execute {rel(NEXT_PROTOCOL_MD)} transient-state continuation probe; if it fails, move to versioned stress-scenario opportunity design. Backup required before rollout: {rel(BACKUP_REQUEST)}.\n",
        encoding="utf-8",
    )
    append_docs(out)
    files = [OUT_DIR / "summary.md", OUT_DIR / "raw.json", STATE_PATH, NEXT_PROTOCOL_MD, NEXT_PROTOCOL_JSON, BACKUP_REQUEST, FRESH_RAW, FRESH_DONE, Path(__file__).resolve()]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_request": rel(BACKUP_REQUEST),
        "headline": {
            "fresh_positive_states": fresh_headline["positive_state_count"],
            "fresh_informative_states": fresh_headline["informative_state_count"],
            "fresh_refit_training_gate_pass": fresh_headline["fresh_refit_training_gate_pass"],
            "next_protocol": {"json": rel(NEXT_PROTOCOL_JSON), "md": rel(NEXT_PROTOCOL_MD)},
            "next_action": decision["next_action"],
        },
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "fresh_positive_states": fresh_headline["positive_state_count"],
        "fresh_informative_states": fresh_headline["informative_state_count"],
        "fresh_refit_training_gate_pass": fresh_headline["fresh_refit_training_gate_pass"],
        "max_non_H15_total_gain": stats["max_non_H15_total_gain_candidates"][0]["total_gain"] if stats["max_non_H15_total_gain_candidates"] else None,
        "large_harm_count": stats["large_harm_count"],
        "next_protocol_json": rel(NEXT_PROTOCOL_JSON),
        "backup_request": rel(BACKUP_REQUEST),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
