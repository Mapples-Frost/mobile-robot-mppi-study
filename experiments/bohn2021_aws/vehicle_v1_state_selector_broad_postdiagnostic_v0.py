#!/usr/bin/env python3
"""No-simulation postdiagnostic after the V1 state-selector broad confirmation.

This script consumes only already-opened development diagnostics and writes a
ranked causal interpretation plus the next frozen development protocol.  It does
not construct an environment, run a rollout, train/refit a model, open the
historical validation64 bank, or access the sealed final test.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_selector_broad_postdiagnostic_v0_20260928T1345Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_state_selector_broad_postdiagnostic_v0_20260928T1345Z.md"
NEXT_PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_fresh_continuation_label_probe_v0_frozen_20260928.json"
NEXT_PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_fresh_continuation_label_probe_v0_frozen_20260928.md"
BACKUP_REQUEST_DIR = ROOT / "research_artifacts/aws_backup_proofs"
BROAD_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_20260928/raw.json"
BROAD_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_20260928/completed.json"
CONTROLLED_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928/raw.json"
CONTROLLED_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928/completed.json"
V1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/raw.json"
V1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/completed.json"
SHADOW_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_selector_shadow_scan_v0_20260928T1310Z/raw.json"
SHADOW_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_selector_shadow_scan_v0_20260928T1310Z/completed.json"
MARKER = "vehicle-v1-state-selector-broad-postdiagnostic-v0-20260928"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def safe_float(x: Any) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else 0.0
    except Exception:
        return 0.0


def sum_cases(per_case: Mapping[str, Any], cases: Iterable[int], key: str) -> float:
    return float(math.fsum(safe_float(per_case[str(c)][key]) for c in cases))


def verify_inputs() -> Dict[str, Any]:
    for p in (BROAD_RAW, BROAD_DONE, CONTROLLED_RAW, CONTROLLED_DONE, V1_RAW, V1_DONE, SHADOW_RAW, SHADOW_DONE):
        require(p.exists(), "missing required input %s" % rel(p))
    broad_done = read_json(BROAD_DONE)
    controlled_done = read_json(CONTROLLED_DONE)
    v1_done = read_json(V1_DONE)
    shadow_done = read_json(SHADOW_DONE)
    for label, obj in (("broad_done", broad_done), ("controlled_done", controlled_done), ("v1_done", v1_done), ("shadow_done", shadow_done)):
        require(obj.get("passed") is True, "%s did not pass" % label)
        require(obj.get("historical_validation64_bank_opened") is False, "%s opened validation64" % label)
        require(obj.get("sealed_test_accessed") is False, "%s accessed sealed test" % label)
    broad = read_json(BROAD_RAW)
    controlled = read_json(CONTROLLED_RAW)
    v1 = read_json(V1_RAW)
    shadow = read_json(SHADOW_RAW)
    for label, obj in (("broad", broad), ("controlled", controlled), ("v1", v1), ("shadow", shadow)):
        require(obj.get("historical_validation64_bank_opened") is False, "%s opened validation64" % label)
        require(obj.get("sealed_test_accessed") is False, "%s accessed sealed test" % label)
    require((broad.get("analysis") or {}).get("protocol_acceptance_pass") is True, "broad selector confirmation did not pass frozen protocol")
    return dict(broad=broad, controlled=controlled, v1=v1, shadow=shadow, done=dict(broad=broad_done, controlled=controlled_done, v1=v1_done, shadow=shadow_done))


def build_next_protocol(now: str, raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "protocol_id": "vehicle_v1_fresh_continuation_label_probe_v0_frozen_20260928",
        "created_utc": now,
        "classification": "development_IMPROVED_fresh_state_continuation_label_probe_not_validation_not_final_test",
        "motivation": [
            "The V1 nearest-state latch passed all-16 development confirmation, but all benefit came from two previously labelled cases, so generalization/training cannot be inferred.",
            "A fresh continuation-label probe is the smallest next rollout that distinguishes real reusable state-dependent horizon structure from overfit exact-state prototypes before launching a broader refit/training campaign."
        ],
        "access_rules": {
            "development_only": True,
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "requires_verified_external_backup_after_broad_postdiagnostic_before_rollout": True
        },
        "generator_and_split": {
            "task": "vehicle",
            "source_support": "same source-supported vehicle reset factors and obstacle/reference capabilities used by vehicle_fixed_h_opportunity_probe_v1; no scenario parameters are changed for this probe",
            "fresh_candidate_pool_resets": 48,
            "candidate_pool_seed": 2609287601,
            "selected_case_count": 12,
            "selection_seed": 2609287602,
            "selection_rule_frozen_before_outcomes": "balanced metadata strata by abs(theta_r), nominal trajectory length and reference-obstacle clearance, using reset metadata only; no fixed-H/adaptive outcomes may influence case selection",
            "validation64_or_test_overlap_allowed": False
        },
        "rollout_design": {
            "phase": "controlled_continuation_label_mining_only",
            "prefix_horizon": 15,
            "branch_horizons": [10, 15, 30, 35],
            "branch_step_rule": "for each selected case after an H15 prefix, use two deterministic branch steps based only on generated reference length: floor(0.15*traj_steps) and floor(0.35*traj_steps), clipped to [8, 90] and de-duplicated; if the H15 episode would terminate before a branch, record the skipped branch as noninformative rather than replacing it",
            "max_rollout_episodes": 96,
            "control_step_upper_bound": 14400,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "online_selector_rollouts": 0,
            "comparators": "Within each identical prefix state compare fixed branch H10/H15/H30/H35; H15 is the continuation reference"
        },
        "label_and_gate_rules": {
            "material_positive_state": "a branch state is positive for a non-H15 horizon if it succeeds, has no constraint/solver regression versus H15 from the identical prefix state, and improves continuation physical or total cost by >=3 absolute units versus H15",
            "negative_or_neutral_state": "no non-H15 branch horizon meeting the material-positive rule; keep all negatives and near misses",
            "gate_for_next_refit": "at least 4 material-positive states spanning at least 3 selected cases, plus at least 4 negative/neutral states, with no evidence that positives are all one exact case-specific coordinate cluster",
            "if_gate_passes": "freeze a compact refit/training experiment (e.g. kNN/radius/prototype classifier or small supervised horizon-value model) on these labels and confirm online on a separate fresh development bank with fixed H10/H15/H30/H35/full-grid baselines",
            "if_gate_fails": "document sparse canonical-distribution adaptive opportunity and prioritize scenario-design or objective/terminal-value diagnosis instead of validating another overfit selector"
        },
        "budget_accounting": {
            "counts_as_development_simulation": True,
            "formal_validation_evidence": False,
            "expected_environment_constructions_upper_bound": 96,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "sealed_test_episodes": 0
        },
        "inputs_from_current_postdiagnostic": {
            "postdiagnostic_raw": rel(OUT_DIR / "raw.json"),
            "postdiagnostic_summary": rel(OUT_DIR / "summary.md")
        }
    }


def write_next_protocol_files(protocol: Mapping[str, Any]) -> None:
    write_json(NEXT_PROTOCOL_JSON, protocol)
    lines = [
        "# Vehicle V1 fresh continuation-label probe v0",
        "",
        f"Frozen UTC: `{protocol['created_utc']}`.",
        "",
        "Development-only next rollout protocol. It is not validation and does not access the sealed test or historical validation64 bank.",
        "",
        "## Motivation",
        "",
    ]
    for item in protocol["motivation"]:
        lines.append(f"- {item}")
    g = protocol["generator_and_split"]
    r = protocol["rollout_design"]
    l = protocol["label_and_gate_rules"]
    lines += [
        "",
        "## Frozen split and budget",
        "",
        f"- Candidate pool resets: `{g['fresh_candidate_pool_resets']}` with seed `{g['candidate_pool_seed']}`.",
        f"- Selected cases: `{g['selected_case_count']}` with selection seed `{g['selection_seed']}`.",
        f"- Selection rule: {g['selection_rule_frozen_before_outcomes']}.",
        f"- Prefix horizon: `H{r['prefix_horizon']}`; branch horizons: `{r['branch_horizons']}`.",
        f"- Branch step rule: {r['branch_step_rule']}.",
        f"- Max rollout episodes/control steps: `{r['max_rollout_episodes']}` / `{r['control_step_upper_bound']}`.",
        "- Training episodes / gradient steps: `0 / 0`.",
        "",
        "## Gate for next refit/training",
        "",
        f"- Positive label: {l['material_positive_state']}.",
        f"- Negative/neutral label: {l['negative_or_neutral_state']}.",
        f"- Gate: {l['gate_for_next_refit']}.",
        f"- If pass: {l['if_gate_passes']}.",
        f"- If fail: {l['if_gate_fails']}.",
        "",
        "A verified external backup covering this protocol, the broad-confirmation outputs, and this postdiagnostic is required before rollout.",
    ]
    NEXT_PROTOCOL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_backup_request(now: str) -> str:
    stamp = now.replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_REQUEST_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_V1_STATE_SELECTOR_BROAD_POSTDIAGNOSTIC_V0_%s.json" % stamp)
    write_json(path, {
        "requested_utc": now,
        "reason": "backup broad selector confirmation postdiagnostic and next fresh continuation-label protocol before any further simulation/refit",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [
            rel(OUT_DIR), rel(STATE_PATH), rel(NEXT_PROTOCOL_JSON), rel(NEXT_PROTOCOL_MD), rel(Path(__file__).resolve()),
            rel(BROAD_RAW), rel(BROAD_DONE)
        ]
    })
    return rel(path)


def append_docs(raw: Mapping[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 state-selector broad postdiagnostic v0\n\n"
        f"UTC: {raw['created_utc']}. No-simulation postdiagnostic completed after broad selector confirmation. "
        f"Broad selector acceptance={raw['broad_gate']['protocol_acceptance_pass']}; benefit_vs_H15=(physical {raw['broad_gate']['benefit_physical_abs_vs_H15']:.6g}, total {raw['broad_gate']['benefit_total_abs_vs_H15']:.6g}); "
        f"triggered_cases={raw['broad_gate']['triggered_cases']}. Next frozen protocol: `{rel(NEXT_PROTOCOL_MD)}` / `{rel(NEXT_PROTOCOL_JSON)}`. "
        f"No validation64/test access and no new rollout/training. Backup required before further simulation.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = verify_inputs()
    broad = inputs["broad"]
    v1 = inputs["v1"]
    controlled = inputs["controlled"]
    shadow = inputs["shadow"]
    analysis = broad["analysis"]
    by_arm = analysis["by_arm"]
    per_case = analysis["per_case"]
    triggered_cases = [int(c) for c in analysis["triggered_cases"]]
    nontriggered_cases = [c for c in range(16) if c not in triggered_cases]
    selector_steps = int(by_arm["selector_latch_H30"]["steps"])
    selector_h30_steps = int(by_arm["selector_latch_H30"].get("horizon_counts", {}).get("30", 0))
    triggered_phys_gain = -sum_cases(per_case, triggered_cases, "selector_vs_H15_physical_delta")
    triggered_total_gain = -sum_cases(per_case, triggered_cases, "selector_vs_H15_total_delta")
    nontrigger_phys_gain = -sum_cases(per_case, nontriggered_cases, "selector_vs_H15_physical_delta")
    nontrigger_total_gain = -sum_cases(per_case, nontriggered_cases, "selector_vs_H15_total_delta")
    selector_decision_sum = safe_float(by_arm["selector_latch_H30"]["decision_total_s"])
    h15_decision_sum = safe_float(by_arm["fixed_H15"]["decision_total_s"])
    selector_solver_sum = safe_float(by_arm["selector_latch_H30"]["solver_attempt_total_s"])
    h15_solver_sum = safe_float(by_arm["fixed_H15"]["solver_attempt_total_s"])
    controlled_rows = (controlled.get("analysis") or {}).get("state_rows") or []
    controlled_positive = [r for r in controlled_rows if r.get("confirmed_material_noninitial_state")]
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    causal_table = [
        {
            "axis": "scenarios",
            "verified_finding": "V1 fixed-H grid had case-dependent best horizons, and identical-prefix continuation found 3 material noninitial H30-positive states across cases 7 and 10. The online latch transferred those states exactly and passed all-16 development confirmation.",
            "competing_hypothesis_or_uncertainty": "Opportunity may be sparse and overfit to exact development coordinates; fresh cases may have no reusable state-local H30 regions.",
            "discriminating_experiment": "Fresh continuation-label probe v0: generate a fresh metadata-balanced development bank and compare H10/H15/H30/H35 from identical H15-prefix branch states before any online selector validation."
        },
        {
            "axis": "reward/objective",
            "verified_finding": "Broad selector benefit vs H15 was physical+constraint +38.7155 and total +36.9005 absolute units; the gain is not only a synthetic horizon penalty. All arms succeeded without constraints or solver-failure steps.",
            "competing_hypothesis_or_uncertainty": "Historical training objectives/terminal values may still mis-rank transitions; current latch uses hand-mined labels and does not prove reward/value correctness.",
            "discriminating_experiment": "In the fresh probe, label branch states by realized continuation physical and total costs under identical prefixes; later fit using those labels and compare against matched fixed-H baselines."
        },
        {
            "axis": "training/selection",
            "verified_finding": "A zero-gradient state-local latch exploited cases that historical gated policies missed, strengthening the diagnosis of policy-class/state-label/objective/coverage mismatch rather than absent adaptive opportunity.",
            "competing_hypothesis_or_uncertainty": "Nearest exact-state prototypes are not a scalable trained policy and may not generalize; value/terminal mismatch remains plausible.",
            "discriminating_experiment": "If the fresh probe yields enough positive and negative labels, freeze a compact refit/training experiment across at least three independent seeds with explicit gradient/refit budget accounting."
        },
        {
            "axis": "comparisons",
            "verified_finding": "Broad confirmation reran paired H15/H30/H10 with actual timing. Selector beat H15/H30/H10 in aggregate cost on V1; selection overhead was 0.1536s total and terminal-switch overhead 8.0e-5s, but whole-decision timing differences are tiny and noisy.",
            "competing_hypothesis_or_uncertainty": "No speed claim is established; broader H grid and independently tuned fixed-H/terminal baselines remain necessary for validation.",
            "discriminating_experiment": "Use fresh label/refit development first; any later validation must include full fixed-H grid and measured whole-decision/solver timing in randomized paired blocks."
        }
    ]
    next_protocol = build_next_protocol(now, dict())
    write_next_protocol_files(next_protocol)
    raw: Dict[str, Any] = {
        "created_utc": now,
        "classification": "no_simulation_postdiagnostic_development_only",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "input_hashes": {rel(p): sha256(p) for p in (BROAD_RAW, BROAD_DONE, CONTROLLED_RAW, CONTROLLED_DONE, V1_RAW, V1_DONE, SHADOW_RAW, SHADOW_DONE)},
        "broad_gate": {
            "protocol_acceptance_pass": bool(analysis["protocol_acceptance_pass"]),
            "safety_pass_protocol": bool(analysis["safety_pass_protocol"]),
            "benefit_pass_protocol": bool(analysis["benefit_pass_protocol"]),
            "benefit_physical_abs_vs_H15": safe_float(analysis["benefit_physical_abs_vs_H15"]),
            "benefit_total_abs_vs_H15": safe_float(analysis["benefit_total_abs_vs_H15"]),
            "triggered_cases": triggered_cases,
            "extra_trigger_cases": analysis.get("extra_trigger_cases"),
            "missing_expected_trigger_cases": analysis.get("missing_expected_trigger_cases")
        },
        "decomposition": {
            "selector_steps": selector_steps,
            "selector_H30_steps": selector_h30_steps,
            "selector_H30_step_fraction": selector_h30_steps / float(selector_steps),
            "triggered_cases_physical_gain_vs_H15": triggered_phys_gain,
            "triggered_cases_total_gain_vs_H15": triggered_total_gain,
            "nontrigger_cases_physical_gain_vs_H15": nontrigger_phys_gain,
            "nontrigger_cases_total_gain_vs_H15": nontrigger_total_gain,
            "selector_minus_H15_decision_sum_s": selector_decision_sum - h15_decision_sum,
            "selector_minus_H15_solver_attempt_sum_s": selector_solver_sum - h15_solver_sum,
            "selector_selection_total_s": safe_float(by_arm["selector_latch_H30"].get("selection_total_s")),
            "selector_terminal_switch_total_s": safe_float(by_arm["selector_latch_H30"].get("terminal_switch_total_s")),
            "selector_minus_H30_physical": safe_float(analysis["aggregate_deltas"]["selector_minus_H30_physical"]),
            "selector_minus_H30_total": safe_float(analysis["aggregate_deltas"]["selector_minus_H30_total"]),
            "v1_overall_strongest_total_horizon": (v1.get("opportunity_analysis") or {}).get("overall_strongest_total_horizon"),
            "v1_safe_horizons": (v1.get("opportunity_analysis") or {}).get("safe_horizons_success_no_constraint_all_cases"),
            "controlled_positive_state_count": len(controlled_positive),
            "shadow_triggered_cases": shadow.get("triggered_cases")
        },
        "causal_table": causal_table,
        "decision": {
            "ranked_next_action": "After external backup, run the fresh continuation-label probe v0 rather than another unchanged selector validation shard.",
            "why_higher_information_value": "The current selector has passed a development transfer check but all gains are from previously mined states; fresh identical-prefix labels directly test reusable state-dependent opportunity and produce data needed for a real refit/training intervention.",
            "why_not_more_validation_now": "Running current exact-coordinate prototypes on validation/fresh confirmation would mostly test overfitting or non-triggering, not the causal training/label bottleneck. Validation64 and sealed test remain untouched.",
            "why_not_immediate_gradient_training": "There are only three positive continuation labels from two cases; a small fresh label probe is needed before spending training budget or claiming a scalable policy."
        },
        "next_protocol": {"json": rel(NEXT_PROTOCOL_JSON), "md": rel(NEXT_PROTOCOL_MD), "json_sha256": sha256(NEXT_PROTOCOL_JSON), "md_sha256": sha256(NEXT_PROTOCOL_MD)},
        "source": {"path": rel(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}
    }
    raw["backup_request"] = write_backup_request(now)
    write_json(OUT_DIR / "raw.json", raw)
    lines = [
        "# Vehicle V1 state-selector broad postdiagnostic v0",
        "",
        f"Created UTC: `{now}`.",
        "",
        "No-simulation postdiagnostic after the passed all-16 development selector confirmation. No environment construction, no rollouts, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Result",
        "",
        f"- Broad selector protocol acceptance: `{raw['broad_gate']['protocol_acceptance_pass']}` (safety `{raw['broad_gate']['safety_pass_protocol']}`, benefit `{raw['broad_gate']['benefit_pass_protocol']}`).",
        f"- Benefit vs paired H15: physical `{raw['broad_gate']['benefit_physical_abs_vs_H15']:.6g}`, total `{raw['broad_gate']['benefit_total_abs_vs_H15']:.6g}`.",
        f"- Triggered cases: `{triggered_cases}`; H30 steps `{selector_h30_steps}/{selector_steps}` = `{raw['decomposition']['selector_H30_step_fraction']:.3%}`.",
        f"- Gain decomposition vs H15: triggered cases physical `{triggered_phys_gain:.6g}` / total `{triggered_total_gain:.6g}`; nontrigger cases physical `{nontrigger_phys_gain:.6g}` / total `{nontrigger_total_gain:.6g}`.",
        f"- Timing deltas vs H15: decision sum `{raw['decomposition']['selector_minus_H15_decision_sum_s']:.6g}s`, solver-attempt sum `{raw['decomposition']['selector_minus_H15_solver_attempt_sum_s']:.6g}s`; selection overhead `{raw['decomposition']['selector_selection_total_s']:.6g}s`, terminal-switch overhead `{raw['decomposition']['selector_terminal_switch_total_s']:.6g}s`.",
        "",
        "## Evidence table",
        "",
        "| axis | verified finding | uncertainty / competing hypothesis | discriminating experiment |",
        "|---|---|---|---|",
    ]
    for row in causal_table:
        lines.append("| %s | %s | %s | %s |" % (row["axis"], row["verified_finding"], row["competing_hypothesis_or_uncertainty"], row["discriminating_experiment"]))
    lines += [
        "",
        "## Decision",
        "",
        f"Next action after verified backup: **{raw['decision']['ranked_next_action']}**",
        "",
        raw["decision"]["why_higher_information_value"],
        "",
        f"Frozen next protocol: `{rel(NEXT_PROTOCOL_MD)}` / `{rel(NEXT_PROTOCOL_JSON)}`.",
        f"Backup request: `{raw['backup_request']}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 state-selector broad postdiagnostic v0 ({now})\n\n"
        f"No-simulation postdiagnostic completed. Broad selector passed; benefit_vs_H15 physical={raw['broad_gate']['benefit_physical_abs_vs_H15']:.6g}, total={raw['broad_gate']['benefit_total_abs_vs_H15']:.6g}; triggered_cases={triggered_cases}. "
        f"Next after backup: fresh continuation-label probe v0 (`{rel(NEXT_PROTOCOL_MD)}`). Backup required before further simulation/refit.\n",
        encoding="utf-8"
    )
    append_docs(raw)
    completed_files = [OUT_DIR / "raw.json", OUT_DIR / "summary.md", STATE_PATH, NEXT_PROTOCOL_JSON, NEXT_PROTOCOL_MD, ROOT / raw["backup_request"], Path(__file__).resolve()]
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
        "headline": {
            "broad_protocol_acceptance": raw["broad_gate"]["protocol_acceptance_pass"],
            "benefit_physical_abs_vs_H15": raw["broad_gate"]["benefit_physical_abs_vs_H15"],
            "benefit_total_abs_vs_H15": raw["broad_gate"]["benefit_total_abs_vs_H15"],
            "triggered_cases": triggered_cases,
            "next_protocol": raw["next_protocol"]
        },
        "backup_request": raw["backup_request"],
        "hashes": {rel(p): sha256(p) for p in completed_files if p.exists()}
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "new_rollouts": 0,
        "new_control_steps": 0,
        "broad_protocol_acceptance": raw["broad_gate"]["protocol_acceptance_pass"],
        "next_protocol": raw["next_protocol"],
        "backup_request": raw["backup_request"],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
