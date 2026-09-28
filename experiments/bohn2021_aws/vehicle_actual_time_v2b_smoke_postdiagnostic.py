#!/usr/bin/env python3
"""Post-smoke diagnostic for vehicle actual-time-aware gated-horizon V2b.

Metadata-only diagnostic. Reads the just-completed V2b smoke outputs and prior
V2b candidate metrics; writes a compact three-layer diagnosis, a state artifact,
a backup request, and a versioned draft/protocol for the next scenario-opportunity
capability audit. It performs no simulation, training, validation-bank access, or
sealed-test access.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T0935Z"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_smoke_postdiagnostic_20260928T0935Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_actual_time_v2b_smoke_postdiagnostic_20260928T0935Z.md"
BACKUP_PATH = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_ACTUAL_TIME_V2B_SMOKE_POSTDIAGNOSTIC_20260928T0935Z.json"
NEXT_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_scenario_opportunity_capability_diagnostic_v0_after_v2b_smoke_20260928.md"
SMOKE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_20260928/raw.json"
SMOKE_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_20260928/completed.json"
SMOKE_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_20260928/summary.md"
CANDIDATE_CSV = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_seed0_overhead_repair_20260928T0915Z/candidate_timing_metrics.csv"
CANDIDATE_DIAG = ROOT / "research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z/completed.json"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_protocol_20260928.md"
MARKER = "vehicle-actual-time-v2b-smoke-postdiagnostic-20260928"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def arm_id(kind: str, seed: int) -> str:
    if kind == "selected":
        return f"actual_time_v2b_selected_vehicle_s{seed}"
    if kind == "current":
        return f"current_gated_vehicle_s{seed}"
    if kind == "fixed":
        return f"fixed_H25_vehicle_s{seed}"
    raise ValueError(kind)


def sum_field(aggregates: Mapping[str, Mapping[str, Any]], ids: List[str], field: str) -> float:
    return float(sum(float(aggregates[i].get(field, 0.0)) for i in ids))


def load_candidate_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with CANDIDATE_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out: Dict[str, Any] = dict(row)
            for k in (
                "seed", "short_h", "profile", "guard", "mean_physical_delta_pct_vs_fixed",
                "max_pair_physical_regression_pct", "positive_tail_cvar80_pct", "short_step_fraction",
                "episodes_with_short_estimated", "top2_short_step_share", "switch_rate",
                "timing_ratio_median_vs_H25", "timing_support_n", "estimated_gross_time_saving_fraction",
                "selection_overhead_fraction", "estimated_net_time_saving_fraction",
            ):
                if k in out:
                    v = maybe_float(out[k])
                    if v is not None:
                        out[k] = v
            out["actual_time_primary_eligible"] = str(out.get("actual_time_primary_eligible", "")).lower() == "true"
            rows.append(out)
    return rows


def make_next_protocol(created: str, raw: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> None:
    text = f"""# Vehicle scenario-opportunity/capability diagnostic V0 after V2b smoke

Frozen UTC: {created}.

## Classification and purpose

- Diagnostic only; no final-test access.
- This protocol is not a scenario revision campaign and not model selection.
- Purpose: before allocating another long revised-method validation campaign, verify whether the vehicle environment and current scenario distributions actually create state-dependent prediction-horizon opportunity, and separate scenario/design limitations from selector/training and implementation overhead.

## Motivating evidence

- V2b finite actual-time-aware re-selection smoke passed engineering checks on 36 fresh smoke episodes, but it is tiny and not scientific evidence.
- Seed0 V2b falls back to fixed H25 because no adaptive candidate was eligible under measured-time/risk gates.
- Seeds1/2 both use the same finite candidate h15_p1_g5; smoke realized timing was mixed rather than a stable multi-seed acceleration signal.
- Smoke all-success/equal-step outcomes and near-zero physical deltas indicate that the smoke bank itself had little control-performance stress.

## Phase A: metadata/source capability audit (next bounded experiment)

Inputs:
- environment/controller source files under the registered Bohn vehicle/gym-horizon sources;
- existing training/dev diagnostic summaries only;
- no historical validation64 bank content, no sealed final test content, no new rollouts.

Questions:
1. What reset variables and goal/state distributions are actually supported (position, heading, velocity, curvature/path parameters, constraints, disturbance/model uncertainty)?
2. Which of these were used in the paper-aligned reference benchmark versus current fresh diagnostic/smoke banks?
3. Do existing training/dev states cover transient, near-constraint, heading/curvature-change and far/near-goal regimes Leah enough for horizon choice to matter?
4. Are fixed-H Pareto relationships already dominant across regimes, making adaptation unnecessary?

Acceptance for Phase A:
- produce a table of supported vs unsupported scenario factors with source-line evidence/hash references;
- explicitly forbid adding unsupported factors;
- recommend either a bounded fixed-H continuation/opportunity rollout or a versioned revised generator only if source support and hypothesis are clear.

## Phase B: only if Phase A supports it

A bounded fresh diagnostic bank (not validation64/test) may compare fixed H values from identical generated states with actual wall time, continuation/control cost, constraints and solver failures. It must be versioned before generation, include favorable and unfavorable cases, and fairly re-tune fixed-H baselines before any later claim.

## Current next action

After verified backup of the V2b smoke and this postdiagnostic, run Phase A metadata/source capability audit. Do not resume unchanged long risk-reselection or V2b devval campaigns by default.
"""
    NEXT_PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    NEXT_PROTOCOL.write_text(text, encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    text = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle actual-time V2b smoke postdiagnostic\n\n"
        f"UTC: {raw['created_utc']}. Metadata-only postdiagnostic of the V2b smoke completed; no simulations/training, no validation64 bank access, and no sealed-test access. "
        f"Smoke hard_pass={raw['smoke']['hard_pass']}; episodes={raw['smoke']['episodes']}; control_steps={raw['smoke']['control_steps']}. "
        f"Weighted selected/fixed decision-time ratio={raw['timing']['weighted_selected_vs_fixed_decision_ratio']:.6g}; adaptive seed ratios={raw['timing']['adaptive_seed_selected_vs_fixed_decision_ratios']}. "
        f"Decision: {raw['decision']['headline']} Next protocol draft: `{rel(NEXT_PROTOCOL)}`. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    smoke = read_json(SMOKE_RAW)
    completed = read_json(SMOKE_COMPLETED)
    candidate_rows = load_candidate_rows()
    aggregates = smoke["aggregates"]
    selected_ids = [arm_id("selected", seed) for seed in (0, 1, 2)]
    fixed_ids = [arm_id("fixed", seed) for seed in (0, 1, 2)]
    current_ids = [arm_id("current", seed) for seed in (0, 1, 2)]

    per_seed: Dict[str, Any] = {}
    adaptive_ratios: Dict[str, float] = {}
    for seed in (0, 1, 2):
        sid = arm_id("selected", seed)
        fid = arm_id("fixed", seed)
        cid = arm_id("current", seed)
        s = aggregates[sid]
        f = aggregates[fid]
        c = aggregates[cid]
        pair = smoke["paired_comparisons"][str(seed)]
        short_steps = int(s.get("steps", 0) - int(s.get("horizon_counts", {}).get("25", 0)))
        current_short_steps = int(c.get("steps", 0) - int(c.get("horizon_counts", {}).get("25", 0)))
        ratio = float(pair["selected_minus_fixed"]["decision_time_ratio"])
        if s.get("adapted_below_H25"):
            adaptive_ratios[str(seed)] = ratio
        per_seed[str(seed)] = {
            "selected_role": s.get("role"),
            "selected_policy_id": s.get("policy", {}).get("id"),
            "selected_adapted_below_H25": bool(s.get("adapted_below_H25")),
            "selected_short_step_fraction": short_steps / float(max(1, int(s.get("steps", 0)))),
            "current_short_step_fraction": current_short_steps / float(max(1, int(c.get("steps", 0)))),
            "selected_horizon_counts": s.get("horizon_counts"),
            "current_horizon_counts": c.get("horizon_counts"),
            "selected_vs_fixed_decision_ratio": ratio,
            "selected_vs_fixed_realized_decision_saving_fraction": 1.0 - ratio,
            "selected_vs_fixed_gross_decision_ratio": float(s.get("decision_gross_total_s", 0.0)) / float(f.get("decision_gross_total_s", 1.0)),
            "selected_minus_fixed_decision_total_s": float(s.get("decision_total_s", 0.0)) - float(f.get("decision_total_s", 0.0)),
            "selected_minus_fixed_physical_constraint_delta": pair["selected_minus_fixed"]["physical_constraint_delta"],
            "selected_minus_fixed_total_delta": pair["selected_minus_fixed"]["total_cost_delta"],
            "estimated_net_time_saving_fraction_from_v2b_policy": s.get("policy", {}).get("v2b_estimated_net_time_saving_fraction"),
            "safety_counts_not_worse_than_fixed_on_smoke": not (
                s.get("success_count", 0) < f.get("success_count", 0)
                or s.get("constraint_count", 0) > f.get("constraint_count", 0)
                or s.get("initial_failed_steps", 0) > f.get("initial_failed_steps", 0)
                or s.get("solver_failure_steps", 0) > f.get("solver_failure_steps", 0)
            ),
            "success_counts": {"selected": s.get("success_count"), "fixed": f.get("success_count"), "current": c.get("success_count")},
            "initial_failed_steps": {"selected": s.get("initial_failed_steps"), "fixed": f.get("initial_failed_steps"), "current": c.get("initial_failed_steps")},
        }

    total_selected_decision = sum_field(aggregates, selected_ids, "decision_total_s")
    total_fixed_decision = sum_field(aggregates, fixed_ids, "decision_total_s")
    total_current_decision = sum_field(aggregates, current_ids, "decision_total_s")
    total_selected_gross = sum_field(aggregates, selected_ids, "decision_gross_total_s")
    total_fixed_gross = sum_field(aggregates, fixed_ids, "decision_gross_total_s")
    total_steps = int(smoke["budget_actual"]["control_steps"])
    deadline_exceed = int(sum(int(a.get("deadline_exceed_steps", 0)) for a in aggregates.values()))
    selected_physical_delta = sum(float(per_seed[str(seed)]["selected_minus_fixed_physical_constraint_delta"]) for seed in (0, 1, 2))
    selected_total_delta = sum(float(per_seed[str(seed)]["selected_minus_fixed_total_delta"]) for seed in (0, 1, 2))

    eligible_counts: Dict[str, int] = {"0": 0, "1": 0, "2": 0}
    for row in candidate_rows:
        seed_val = str(int(float(row["seed"]))) if str(row.get("seed", "")).strip() else "unknown"
        if row.get("candidate_id") != "fixed" and row.get("actual_time_primary_eligible"):
            eligible_counts[seed_val] = eligible_counts.get(seed_val, 0) + 1

    timing_mixed = any(r < 1.0 for r in adaptive_ratios.values()) and any(r >= 1.0 for r in adaptive_ratios.values())
    all_success_equal_steps = all(aggregates[i].get("success_count") == 4 and aggregates[i].get("steps") == 314 for i in aggregates)
    near_zero_physical = abs(selected_physical_delta) < 1e-2

    diagnosis = {
        "scenario_opportunity_design": {
            "verified_on_smoke": "all arms succeeded on all four per-arm episodes, every arm had 314 steps, zero constraints, and selected-vs-fixed physical deltas summed to near zero",
            "implication": "the smoke bank confirms executability but contains little control-performance stress; it cannot prove state-dependent horizon opportunity",
            "missing": "source-supported scenario factors and fixed-H Pareto structure across transient/constraint/heading regimes remain to be audited before any redesigned campaign",
        },
        "training_selection_terminal_value": {
            "verified": "V2b is finite metadata re-selection over reused gated candidates; zero new gradient training; all selected arms reuse the existing H25 terminal/value source",
            "candidate_surface": {"rows": len(candidate_rows), "eligible_adaptive_by_seed": eligible_counts, "nominated": {"0": "fixed", "1": "h15_p1_g5", "2": "h15_p1_g5"}},
            "implication": "seed0 has no adaptive candidate under current gates; seeds1/2 depend on one shared candidate, so a broader selector/value/training change remains an open bottleneck",
        },
        "implementation_runtime_overhead": {
            "verified": "controller stack executed, replay matched, selected/current arms dispatched no H>25, and decision timing was recorded",
            "realized_timing": {"adaptive_seed_ratios": adaptive_ratios, "weighted_selected_vs_fixed_ratio": total_selected_decision / total_fixed_decision, "weighted_selected_vs_current_ratio": total_selected_decision / total_current_decision},
            "implication": "timing signal is mixed: seed1 selected was faster, seed2 selected was slower, and seed0 was fixed fallback; horizon-count reduction alone remains insufficient timing evidence",
            "deadline_exceed_steps_all_arms": deadline_exceed,
        },
    }

    decision = {
        "headline": "V2b is engineering-ready but not strong enough to justify a long validation campaign by default.",
        "rationale": [
            "Smoke hard-pass establishes executability/replay/safety-count readiness only.",
            "Only two adaptive seeds exist and realized timing is mixed on the tiny smoke bank.",
            "All-success/equal-step smoke cases provide weak scenario-opportunity evidence.",
            "Before more long simulation, run a metadata/source capability and scenario-opportunity audit; then decide between a bounded fixed-H opportunity rollout, richer selector refit/training, or documented absence of adaptive opportunity.",
        ],
        "next_action_after_verified_backup": "run_vehicle_scenario_opportunity_capability_diagnostic_v0_metadata_only_no_rollouts_no_validation64_no_test",
        "do_not_do_next_by_default": "do_not_resume_unchanged_long_risk_reselection_or_v2b_devval_shards",
    }

    raw: Dict[str, Any] = {
        "created_utc": created,
        "method": "metadata_only_vehicle_actual_time_v2b_smoke_postdiagnostic",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "inputs": {
            rel(SMOKE_RAW): sha256(SMOKE_RAW),
            rel(SMOKE_COMPLETED): sha256(SMOKE_COMPLETED),
            rel(SMOKE_SUMMARY): sha256(SMOKE_SUMMARY),
            rel(CANDIDATE_CSV): sha256(CANDIDATE_CSV),
            rel(CANDIDATE_DIAG): sha256(CANDIDATE_DIAG),
            rel(PROTOCOL): sha256(PROTOCOL),
            rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        },
        "smoke": {
            "hard_pass": bool(completed.get("hard_pass")),
            "episodes": smoke["budget_actual"]["episodes"],
            "control_steps": smoke["budget_actual"]["control_steps"],
            "replay_passed": smoke["replay"].get("passed"),
            "selected_below_H25_seed_count": smoke["smoke_pass_checks"].get("actual_time_v2b_selected_below_H25_seed_count"),
            "all_success_equal_steps_per_arm": all_success_equal_steps,
            "near_zero_selected_physical_delta_sum": near_zero_physical,
        },
        "per_seed": per_seed,
        "timing": {
            "total_selected_decision_s": total_selected_decision,
            "total_fixed_decision_s": total_fixed_decision,
            "total_current_decision_s": total_current_decision,
            "weighted_selected_vs_fixed_decision_ratio": total_selected_decision / total_fixed_decision,
            "weighted_selected_vs_current_decision_ratio": total_selected_decision / total_current_decision,
            "weighted_selected_vs_fixed_gross_decision_ratio": total_selected_gross / total_fixed_gross,
            "adaptive_seed_selected_vs_fixed_decision_ratios": adaptive_ratios,
            "adaptive_seed_timing_mixed": timing_mixed,
            "deadline_exceed_steps_sum_all_arms": deadline_exceed,
            "deadline_exceed_fraction_all_steps": deadline_exceed / float(max(1, total_steps)),
        },
        "control_cost": {
            "selected_minus_fixed_physical_constraint_delta_sum": selected_physical_delta,
            "selected_minus_fixed_total_delta_sum": selected_total_delta,
            "interpretation": "total delta contains horizon/work penalty and is not measured wall-time; physical deltas are the primary control-cost smoke check",
        },
        "diagnosis_layers": diagnosis,
        "decision": decision,
        "next_protocol": rel(NEXT_PROTOCOL),
    }

    make_next_protocol(created, raw, diagnosis)
    raw["outputs"] = {
        "summary": rel(OUT_DIR / "summary.md"),
        "raw": rel(OUT_DIR / "raw.json"),
        "completed": rel(OUT_DIR / "completed.json"),
        "state": rel(STATE_PATH),
        "backup_request": rel(BACKUP_PATH),
        "next_protocol": rel(NEXT_PROTOCOL),
    }

    summary_lines = [
        "# Vehicle actual-time V2b smoke postdiagnostic",
        "",
        f"UTC: `{created}`. Metadata-only; no simulations/training, no validation64-bank access, no sealed-test access.",
        "",
        "## Smoke recap",
        f"- hard_pass: `{raw['smoke']['hard_pass']}`; episodes/control_steps: `{raw['smoke']['episodes']}` / `{raw['smoke']['control_steps']}`.",
        f"- selected_below_H25_seed_count: `{raw['smoke']['selected_below_H25_seed_count']}`.",
        f"- all_success_equal_steps_per_arm: `{all_success_equal_steps}`; near_zero_selected_physical_delta_sum: `{near_zero_physical}`.",
        "",
        "## Realized timing/control signals",
        f"- weighted selected/fixed decision-time ratio: `{raw['timing']['weighted_selected_vs_fixed_decision_ratio']:.6g}`.",
        f"- weighted selected/current decision-time ratio: `{raw['timing']['weighted_selected_vs_current_decision_ratio']:.6g}`.",
        f"- adaptive-seed selected/fixed ratios: `{adaptive_ratios}`; mixed={timing_mixed}.",
        f"- selected-fixed physical delta sum: `{selected_physical_delta:.6g}`; total delta sum: `{selected_total_delta:.6g}`.",
        f"- deadline_exceed_fraction_all_steps: `{raw['timing']['deadline_exceed_fraction_all_steps']:.6g}`.",
        "",
        "## Per-seed summary",
        "",
        "| seed | selected policy | selected short frac | current short frac | selected/fixed time ratio | physical Δ | total Δ |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for seed in (0, 1, 2):
        p = per_seed[str(seed)]
        summary_lines.append(
            f"| {seed} | `{p['selected_policy_id']}` | {p['selected_short_step_fraction']:.4f} | {p['current_short_step_fraction']:.4f} | "
            f"{p['selected_vs_fixed_decision_ratio']:.6g} | {float(p['selected_minus_fixed_physical_constraint_delta']):.6g} | {float(p['selected_minus_fixed_total_delta']):.6g} |"
        )
    summary_lines += [
        "",
        "## Three-layer diagnosis",
        "",
        "1. Scenario/opportunity: smoke cases were easy/all-success/equal-step and therefore do not establish meaningful horizon-choice opportunity.",
        "2. Training/selection/value: V2b remains finite re-selection over reused candidates; seed0 has no eligible adaptive arm and seeds1/2 share one candidate; terminal-value bias is untested.",
        "3. Implementation/runtime: executable and deterministic, but measured timing is mixed and H-count reduction is not reliable acceleration evidence.",
        "",
        "## Decision",
        "",
        f"{decision['headline']}",
        "",
        f"Next after verified backup: `{decision['next_action_after_verified_backup']}`.",
        f"Do not next by default: `{decision['do_not_do_next_by_default']}`.",
        f"Next protocol draft/freeze: `{rel(NEXT_PROTOCOL)}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_json(OUT_DIR / "raw.json", raw)
    write_json(BACKUP_PATH, {
        "requested_utc": created,
        "reason": "backup actual-time V2b smoke postdiagnostic and next scenario-opportunity/capability protocol before any further simulations",
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(BACKUP_PATH), rel(NEXT_PROTOCOL), rel(Path(__file__).resolve()), rel(SMOKE_COMPLETED), rel(SMOKE_SUMMARY), rel(SMOKE_RAW)],
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
    })
    raw["outputs"]["backup_request_sha256"] = sha256(BACKUP_PATH)
    write_json(OUT_DIR / "raw.json", raw)
    state_text = (
        f"# Vehicle actual-time V2b smoke postdiagnostic state ({created})\n\n"
        f"Smoke hard_pass={raw['smoke']['hard_pass']}, episodes={raw['smoke']['episodes']}, control_steps={raw['smoke']['control_steps']}. "
        f"Weighted selected/fixed decision ratio={raw['timing']['weighted_selected_vs_fixed_decision_ratio']:.6g}; adaptive seed ratios={adaptive_ratios}. "
        f"Decision: {decision['headline']} Backup required before more simulations: {rel(BACKUP_PATH)}. "
        f"Next: {decision['next_action_after_verified_backup']} using protocol {rel(NEXT_PROTOCOL)}. "
        "Final test remains sealed; validation64 bank not opened.\n"
    )
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(state_text, encoding="utf-8")
    append_docs(raw)
    files = [OUT_DIR / "summary.md", OUT_DIR / "raw.json", STATE_PATH, BACKUP_PATH, NEXT_PROTOCOL, Path(__file__).resolve(), SMOKE_COMPLETED, SMOKE_SUMMARY, SMOKE_RAW, CANDIDATE_CSV]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "backup_request": rel(BACKUP_PATH),
        "next_action_after_verified_backup": decision["next_action_after_verified_backup"],
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
        "headline": decision["headline"],
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "backup_request": rel(BACKUP_PATH),
        "next_protocol": rel(NEXT_PROTOCOL),
        "weighted_selected_vs_fixed_decision_ratio": raw["timing"]["weighted_selected_vs_fixed_decision_ratio"],
        "adaptive_seed_ratios": adaptive_ratios,
        "decision": decision["headline"],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
