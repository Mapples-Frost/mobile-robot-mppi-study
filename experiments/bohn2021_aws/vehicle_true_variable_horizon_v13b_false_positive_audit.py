#!/usr/bin/env python3
"""Metadata-only audit of v13b history-feature refit outputs.

This script reads the already-produced development-only v13b raw.json and writes a
compact evidence report identifying nested false positives and ambiguity patterns.
It performs no MPC simulation, no selector refit/search, no training, no
validation64 access, and no sealed-test access.
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

RAW_PATH = Path(
    "research_artifacts/aws_diagnostics/"
    "vehicle_true_variable_horizon_history_feature_refit_v13b_fast_20260929T2150Z/"
    "raw.json"
)
SUMMARY_PATH = Path(
    "research_artifacts/aws_diagnostics/"
    "vehicle_true_variable_horizon_v13b_false_positive_audit_20260929T2210Z/summary.md"
)
JSON_PATH = Path(
    "research_artifacts/aws_diagnostics/"
    "vehicle_true_variable_horizon_v13b_false_positive_audit_20260929T2210Z/audit.json"
)
COMPLETED_PATH = Path(
    "research_artifacts/aws_diagnostics/"
    "vehicle_true_variable_horizon_v13b_false_positive_audit_20260929T2210Z/completed.json"
)
BACKUP_REQUEST_PATH = Path(
    "research_artifacts/aws_backup_proofs/"
    "REQUEST_BACKUP_AFTER_V13B_FALSE_POSITIVE_AUDIT_20260929T2210Z.json"
)
STATE_PATH = Path(
    "research_artifacts/aws_state/continue_state_20260929T2210_after_v13b_false_positive_audit.md"
)
PROTOCOL_PATH = Path(
    "research_artifacts/aws_protocols/vehicle_true_variable_horizon_calibrated_risk_value_v14_plan_prebackup_20260929T2210Z.json"
)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def iter_candidate_dicts(obj: Any, path: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], Dict[str, Any]]]:
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from iter_candidate_dicts(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_candidate_dicts(v, path + (str(i),))


def collect_rows_with_keys(obj: Any, required: Tuple[str, ...]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen_ids = set()
    for path, d in iter_candidate_dicts(obj):
        if all(k in d for k in required):
            # Deduplicate exact dictionary object/content by common row identity and path.
            row_id = (
                d.get("bank_id"),
                d.get("base_state_id"),
                d.get("id"),
                d.get("selected_h"),
                d.get("catastrophic"),
                d.get("label_positive"),
                d.get("decision_gain_s"),
                d.get("phys_delta"),
                path[-4:] if len(path) >= 4 else path,
            )
            if row_id not in seen_ids:
                out.append({"_path": "/".join(path), **d})
                seen_ids.add(row_id)
    return out


def main() -> None:
    utc = datetime.now(timezone.utc).isoformat()
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPLETED_PATH.parent.mkdir(parents=True, exist_ok=True)
    BACKUP_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = json.loads(RAW_PATH.read_text())
    top_keys = sorted(raw.keys())
    headline = raw.get("headline", {})

    # Locate nested evaluation objects without assuming exact top-level key names.
    nested_objs: List[Tuple[str, Dict[str, Any]]] = []
    for path, d in iter_candidate_dicts(raw):
        if "catastrophic_false_positive_rows" in d and "chosen_counts" in d and "details" in d:
            nested_objs.append(("/".join(path), d))

    outer_reports: List[Dict[str, Any]] = []
    total_selected_h10 = 0
    total_false_positive = 0
    selected_h10_rows: List[Dict[str, Any]] = []
    false_positive_rows: List[Dict[str, Any]] = []
    selected_by_group = Counter()
    false_by_group = Counter()
    selected_by_window = Counter()
    false_by_window = Counter()
    outer_family_counter = Counter()

    for path, obj in nested_objs:
        # Keep only outer-bank-like objects: a details list of rows with bank/base_state IDs.
        details = obj.get("details") or []
        if not isinstance(details, list) or not details:
            continue
        if not all(isinstance(r, dict) and "bank_id" in r and "base_state_id" in r for r in details[: min(3, len(details))]):
            continue
        counts = obj.get("chosen_counts", {}) or {}
        if not isinstance(counts, dict):
            counts = {}
        h10 = int(counts.get("10", 0) or counts.get(10, 0) or 0)
        fp = obj.get("catastrophic_false_positive_rows") or []
        if not isinstance(fp, list):
            fp = []
        # Path often includes config and outer bank; extract rough labels.
        parts = path.split("/")
        family = None
        cfg = None
        bank = None
        for p in parts:
            if p.startswith("v13b_"):
                cfg = p
                if "history_with_risk" in p:
                    family = "history_with_risk"
                elif "history_no_risk" in p:
                    family = "history_no_risk"
                elif "base_no_history" in p:
                    family = "base_no_history"
            if p.startswith("fresh_v"):
                bank = p
        if family:
            outer_family_counter[family] += 1
        chosen_rows = [r for r in details if r.get("selected_h") == 10]
        total_selected_h10 += len(chosen_rows)
        total_false_positive += len(fp)
        for r in chosen_rows:
            selected_by_group[str(r.get("group"))] += 1
            selected_by_window[str(r.get("window"))] += 1
            selected_h10_rows.append({"outer_path": path, "outer_bank_guess": bank, "config_guess": cfg, **r})
        for r in fp:
            false_by_group[str(r.get("group"))] += 1
            false_by_window[str(r.get("window"))] += 1
            false_positive_rows.append({"outer_path": path, "outer_bank_guess": bank, "config_guess": cfg, **r})
        outer_reports.append(
            {
                "path": path,
                "outer_bank_guess": bank,
                "config_guess": cfg,
                "family_guess": family,
                "chosen_counts": counts,
                "h10_details_count": len(chosen_rows),
                "false_positive_count": len(fp),
                "decision_relative_saving_vs_fixed_H15": obj.get("decision_relative_saving_vs_fixed_H15"),
                "physical_gate": obj.get("physical_gate"),
                "physical_delta_vs_fixed_H15": obj.get("physical_delta_vs_fixed_H15"),
                "confusion": obj.get("confusion"),
                "false_positive_rows": fp,
            }
        )

    # Prefer a compact unique list if nested objects include multiple duplicate views.
    unique_fp = {}
    for r in false_positive_rows:
        key = (r.get("bank_id"), r.get("base_state_id"), r.get("outer_bank_guess"), r.get("config_guess"))
        unique_fp[key] = r
    false_positive_rows_unique = list(unique_fp.values())

    unique_h10 = {}
    for r in selected_h10_rows:
        key = (r.get("bank_id"), r.get("base_state_id"), r.get("outer_bank_guess"), r.get("config_guess"))
        unique_h10[key] = r
    selected_h10_rows_unique = list(unique_h10.values())

    ambiguity = raw.get("ambiguity", {})
    ambiguity_compact = {}
    for family, item in sorted(ambiguity.items()):
        if isinstance(item, dict):
            ambiguity_compact[family] = {
                "feature_count": item.get("feature_count"),
                "positive_with_cat_closer_than_pos": item.get("positive_with_cat_closer_than_pos"),
                "cat_with_pos_closer_than_cat": item.get("cat_with_pos_closer_than_cat"),
                "closest_cat_to_positive_top3": (item.get("closest_cat_to_positive") or [])[:3],
                "closest_positive_to_cat_top3": (item.get("closest_positive_to_cat") or [])[:3],
            }

    # Search any top-global table-like entries if present.
    zero_bad_global = []
    all_config_like = []
    for path, d in iter_candidate_dicts(raw):
        name = d.get("config") or d.get("config_id") or d.get("name")
        if isinstance(name, str) and name.startswith("v13b_"):
            all_config_like.append((path, d))
            bad = d.get("bad", d.get("catastrophic_false_positive_count"))
            avg = d.get("avg_save", d.get("avg_decision_saving_vs_fixed_H15", d.get("decision_relative_saving_vs_fixed_H15")))
            min_save = d.get("min_save", d.get("min_decision_saving_vs_fixed_H15"))
            h10 = d.get("h10", d.get("h10_count"))
            if bad == 0 and avg is not None:
                zero_bad_global.append(
                    {"path": "/".join(path), "config": name, "bad": bad, "avg_save": avg, "min_save": min_save, "h10": h10}
                )
    zero_bad_global = sorted(zero_bad_global, key=lambda x: (-(x.get("avg_save") or 0), x.get("config") or ""))[:20]

    decision = {
        "verified": [
            "v13b loaded all 68 opened development rows with H15-prefix traces and evaluated 24,305 compact support-selector calls.",
            "No global or nested configuration met the >=5% decision-time saving with zero catastrophic H10 false positives gate.",
            "Nested opened-bank deployment chose H10 rarely and still produced a catastrophic H10 false positive, so validation rollout is not justified.",
            "Feature-space ambiguity persists even after adding H15-prefix history; some catastrophic rows are closer to positive rows than to other catastrophic rows.",
        ],
        "hypotheses": [
            "Catastrophic H10 risk is not separable by the current deterministic support-threshold features; calibrated uncertainty or explicit continuation-value estimates are needed.",
            "The current opened-bank labels may undersample decision boundaries, causing conservative selectors to abstain and less conservative selectors to produce false positives.",
            "Actual adaptive opportunity remains present (v11/v12) but requires better deployable risk/value representation rather than more simple feature sweeps.",
        ],
        "next_experiment": "After backup verification, freeze and run a small development-only v14 calibrated risk/value refit using uncertainty-aware abstention and LOBO selection; if calibration remains data-limited, acquire targeted boundary labels under a predeclared MPC simulation budget before validation64.",
    }

    audit = {
        "utc": utc,
        "classification": "metadata_only_existing_v13b_raw_audit_no_simulation_no_refit_no_training_no_validation64_no_sealed_test",
        "raw_path": str(RAW_PATH),
        "raw_sha256": sha256_path(RAW_PATH),
        "top_keys": top_keys,
        "headline": headline,
        "nested_object_count_detected": len(nested_objs),
        "outer_reports_detected": outer_reports,
        "selected_h10_rows_unique": selected_h10_rows_unique,
        "false_positive_rows_unique": false_positive_rows_unique,
        "selected_h10_by_group": dict(selected_by_group),
        "false_positive_by_group": dict(false_by_group),
        "selected_h10_by_window": dict(selected_by_window),
        "false_positive_by_window": dict(false_by_window),
        "outer_family_counter": dict(outer_family_counter),
        "ambiguity_compact": ambiguity_compact,
        "zero_bad_global_config_like_top20": zero_bad_global,
        "decision": decision,
        "budgets": {
            "development_mpc_simulation_episodes": 0,
            "development_control_steps": 0,
            "training_episodes": 0,
            "gradient_steps": 0,
            "selector_refit_evaluations": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
    }

    JSON_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    # v14 plan is intentionally a pre-backup protocol plan, not executable outcome evidence.
    v14_plan = {
        "created_utc": utc,
        "protocol_id": "vehicle_true_variable_horizon_calibrated_risk_value_v14_plan_prebackup_20260929T2210Z",
        "status": "planned_not_run_backup_required_before_execution",
        "classification": "development_IMPROVED_calibrated_risk_terminal_value_refit_plan",
        "blocked_by_backup": "v13b and this metadata audit require verified external backup before any v14 refit/simulation is run",
        "hypothesis": "A deployable selector that estimates catastrophic-H10 risk and H10-vs-H15 continuation value with calibrated uncertainty can abstain on ambiguous boundary states while preserving at least 5% measured decision-time saving on opened development LOBO splits.",
        "before_evidence": [
            "v12 aligned oracle: 30.8% branch decision saving with physical gate on opened rows, so adaptive opportunity exists.",
            "v13b: nested saving 1.5% with 1 catastrophic false positive; zero global/nested pass5 configurations, feature ambiguity persists.",
            "v13b false-positive audit identifies selected catastrophic boundary rows and overlapping positive/catastrophic neighborhoods.",
        ],
        "candidate_method": {
            "inputs": "deployable instantaneous and H15-prefix history features only; no bank ID, case ID, outcome labels, or validation/test data",
            "models": [
                "bagged/calibrated catastrophic-H10 classifier with leave-bank-out calibration",
                "robust H10-vs-H15 decision-time saving regressor",
                "robust H10-vs-H15 physical-delta regressor or upper-confidence bound guard",
            ],
            "selection_rule": "choose H10 only when upper confidence catastrophic risk <= threshold, upper confidence physical delta within tolerance, and lower confidence decision saving exceeds overhead-aware threshold; else H15",
            "uncertainty": "bootstrap/bank-jackknife intervals plus distance-to-support abstention; no forced switching",
        },
        "split": "nested leave-opened-development-bank-out over fresh_v0/fresh_v1/fresh_v2/fresh_v8c/fresh_v11 only; validation64 and sealed test forbidden",
        "initial_budget_caps": {
            "development_mpc_simulation_episodes": 0,
            "development_control_steps": 0,
            "training_episodes": 0,
            "gradient_steps": 0,
            "selector_refit_evaluations_max": 50000,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "gate": {
            "weak_pass": "nested aggregate zero catastrophic H10 false positives, physical gate true, and >=5% measured decision saving vs fixed H15 on opened development rows",
            "strong_pass": "same with >=10% saving",
            "if_fail": "do not run validation64; either targeted boundary acquisition with explicit MPC budget or reconsider learning objective/terminal value from raw trajectories",
        },
        "fairness_notes": [
            "This remains opened-development evidence only and cannot support final claims.",
            "Any later validation requires a frozen overhead implementation and strong fixed-H comparison with actual timing.",
        ],
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
    }
    PROTOCOL_PATH.write_text(json.dumps(v14_plan, indent=2, sort_keys=True) + "\n")

    def pct(x: Any) -> str:
        return "n/a" if x is None else f"{100.0 * float(x):.2f}%"

    lines = []
    lines.append("# v13b false-positive and ambiguity audit\n")
    lines.append(f"UTC `{utc}`. Metadata-only audit of existing v13b raw output; no MPC simulation, no selector refit/search, no training, no validation64 access, and no sealed-test access.\n")
    lines.append("## Verified headline\n")
    lines.append(f"- Raw: `{RAW_PATH}` SHA256 `{audit['raw_sha256']}`.\n")
    lines.append(f"- v13b headline rows `{headline.get('rows')}`, positives `{headline.get('positive_rows')}`, catastrophic rows `{headline.get('catastrophic_rows')}`, selector evaluations `{headline.get('selector_refit_evaluations')}`.\n")
    lines.append(f"- Global pass5/pass10: `{headline.get('global_pass5_count')}` / `{headline.get('global_pass10_count')}`; nested save `{pct(headline.get('nested_save'))}`, nested bad `{headline.get('nested_bad')}`, nested H10 `{headline.get('nested_h10')}`.\n")
    lines.append("\n## Nested false-positive rows detected\n\n")
    if false_positive_rows_unique:
        lines.append("| outer bank/config guess | row | group | window | phys_delta | decision_gain_s | label_positive | catastrophic |\n")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|\n")
        for r in false_positive_rows_unique:
            row = f"{r.get('bank_id')}/{r.get('base_state_id')}"
            outer = f"{r.get('outer_bank_guess')} / {r.get('config_guess')}"
            lines.append(
                f"| `{outer}` | `{row}` | `{r.get('group')}` | `{r.get('window')}` | {float(r.get('phys_delta', 0)):.6g} | {float(r.get('decision_gain_s', 0)):.6g} | `{r.get('label_positive')}` | `{r.get('catastrophic')}` |\n"
            )
    else:
        lines.append("No false-positive rows were detected by the generic parser; rely on v13b completed/summary headline if this conflicts.\n")
    lines.append("\n## Selected H10 pattern\n\n")
    lines.append(f"- Unique selected-H10 rows detected: `{len(selected_h10_rows_unique)}`.\n")
    lines.append(f"- Selected-H10 by group: `{dict(selected_by_group)}`.\n")
    lines.append(f"- Selected-H10 by window: `{dict(selected_by_window)}`.\n")
    lines.append(f"- False-positive by group: `{dict(false_by_group)}`.\n")
    lines.append(f"- False-positive by window: `{dict(false_by_window)}`.\n")
    lines.append("\n## Ambiguity evidence\n\n")
    for family, item in ambiguity_compact.items():
        lines.append(f"- `{family}`: features `{item.get('feature_count')}`, positive-with-closer-cat `{item.get('positive_with_cat_closer_than_pos')}`, catastrophic-with-closer-positive `{item.get('cat_with_pos_closer_than_cat')}`.\n")
    lines.append("\n## Decision\n\n")
    for s in decision["verified"]:
        lines.append(f"- Verified: {s}\n")
    for s in decision["hypotheses"]:
        lines.append(f"- Hypothesis: {s}\n")
    lines.append(f"- Next: {decision['next_experiment']}\n")
    lines.append("\n## Backup and next-run gate\n\n")
    lines.append("v13b artifacts and this audit are not yet covered by a repository-visible verified external backup. No v14 refit/simulation/validation should run until backup verification covers the listed paths.\n")
    SUMMARY_PATH.write_text("".join(lines))

    completed = {
        "status": "complete",
        "utc": utc,
        "classification": audit["classification"],
        "summary": str(SUMMARY_PATH),
        "audit_json": str(JSON_PATH),
        "v14_plan": str(PROTOCOL_PATH),
        "raw_path": str(RAW_PATH),
        "raw_sha256": audit["raw_sha256"],
        "budgets": audit["budgets"],
        "access_flags": audit["access_flags"],
        "headline": {
            "nested_save": headline.get("nested_save"),
            "nested_bad": headline.get("nested_bad"),
            "global_pass5_count": headline.get("global_pass5_count"),
            "global_pass10_count": headline.get("global_pass10_count"),
            "false_positive_rows_detected": len(false_positive_rows_unique),
            "selected_h10_rows_detected": len(selected_h10_rows_unique),
        },
    }
    COMPLETED_PATH.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")

    backup_req = {
        "created_utc": utc,
        "reason": "backup after v13b false-positive audit and v14 prebackup plan before any v14 science",
        "paths": [
            str(SUMMARY_PATH.parent),
            str(PROTOCOL_PATH),
            str(BACKUP_REQUEST_PATH),
            str(STATE_PATH),
            "experiments/bohn2021_aws/vehicle_true_variable_horizon_v13b_false_positive_audit.py",
            "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_feature_refit_v13b_fast_20260929T2150Z",
            "research_artifacts/aws_protocols/vehicle_true_variable_horizon_history_feature_refit_v13b_fast_preoutcome_frozen_20260929T2150Z.json",
            "STATUS.md",
            "RESEARCH_LOG.md",
            "DECISIONS.md",
            "RESULTS_AUDIT.md",
            "REPRODUCTION_PROTOCOL.md",
            "EXPERIMENT_REGISTRY.csv",
        ],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    BACKUP_REQUEST_PATH.write_text(json.dumps(backup_req, indent=2, sort_keys=True) + "\n")

    state = {
        "utc": utc,
        "phase": "after_v13b_false_positive_audit_backup_blocked",
        "current_backup_status": "not verified after v13b/v13b-audit; run external backup before v14 refit/simulation/validation",
        "audit_summary": str(SUMMARY_PATH),
        "audit_json": str(JSON_PATH),
        "v14_plan": str(PROTOCOL_PATH),
        "backup_request": str(BACKUP_REQUEST_PATH),
        "decision": decision,
        "budget_actual_this_iteration": audit["budgets"],
        "access_flags": audit["access_flags"],
    }
    STATE_PATH.write_text("# Continue state after v13b false-positive audit\n\n" + json.dumps(state, indent=2, sort_keys=True) + "\n")

    print(json.dumps(completed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
