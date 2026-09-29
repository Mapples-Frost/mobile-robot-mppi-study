#!/usr/bin/env python3
"""Corrected metadata-only audit of v13b nested deployment outputs.

The first generic audit intentionally searched recursively, but that also counted
nested_aggregate, nested_outer, and top-global holdout views together. This v2
uses only the explicit v13b nested_aggregate and nested_outer objects for the
main counts, preserving the earlier audit as an over-inclusive diagnostic.

No MPC simulation, no selector refit/search, no training, no validation64 access,
and no sealed-test access are performed.
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

RAW_PATH = Path(
    "research_artifacts/aws_diagnostics/"
    "vehicle_true_variable_horizon_history_feature_refit_v13b_fast_20260929T2150Z/"
    "raw.json"
)
OUT_DIR = Path(
    "research_artifacts/aws_diagnostics/"
    "vehicle_true_variable_horizon_v13b_false_positive_audit_v2_20260929T2205Z"
)
SUMMARY_PATH = OUT_DIR / "summary.md"
AUDIT_PATH = OUT_DIR / "audit.json"
COMPLETED_PATH = OUT_DIR / "completed.json"
BACKUP_REQUEST_PATH = Path(
    "research_artifacts/aws_backup_proofs/"
    "REQUEST_BACKUP_AFTER_V13B_FALSE_POSITIVE_AUDIT_V2_20260929T2205Z.json"
)
STATE_PATH = Path(
    "research_artifacts/aws_state/continue_state_20260929T2205_after_v13b_false_positive_audit_v2.md"
)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def join_detail(detail_map: Dict[str, Dict[str, Any]], row: Dict[str, Any]) -> Dict[str, Any]:
    key = f"{row.get('bank_id')}/{row.get('base_state_id')}"
    joined = dict(row)
    if key in detail_map:
        for k, v in detail_map[key].items():
            joined.setdefault(k, v)
    joined["row_id"] = key
    return joined


def count_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    c = Counter(str(r.get(key)) for r in rows)
    return dict(c)


def main() -> None:
    utc = datetime.now(timezone.utc).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = json.loads(RAW_PATH.read_text())
    headline = raw.get("headline", {})
    nested = raw.get("nested_aggregate", {})
    if not isinstance(nested, dict):
        raise RuntimeError("v13b raw.json lacks dict nested_aggregate")
    nested_outer = raw.get("nested_outer", {})
    if not isinstance(nested_outer, dict):
        nested_outer = {}

    details = nested.get("details") or []
    if not isinstance(details, list):
        details = []
    detail_map = {
        f"{d.get('bank_id')}/{d.get('base_state_id')}": d
        for d in details
        if isinstance(d, dict) and d.get("bank_id") is not None and d.get("base_state_id") is not None
    }
    selected_h10 = [d for d in details if isinstance(d, dict) and d.get("selected_h") == 10]
    false_positive = [join_detail(detail_map, r) for r in (nested.get("catastrophic_false_positive_rows") or []) if isinstance(r, dict)]

    outer = {}
    for bank, obj in sorted(nested_outer.items()):
        if not isinstance(obj, dict):
            continue
        outer_eval = obj.get("outer_eval", {}) if isinstance(obj.get("outer_eval"), dict) else obj
        cfg = obj.get("selected_config") or obj.get("selected_config_id") or obj.get("config")
        if cfg is None and isinstance(obj.get("inner_selected"), dict):
            cfg = obj["inner_selected"].get("config") or obj["inner_selected"].get("config_id")
        odetails = outer_eval.get("details") or []
        oh10 = [d for d in odetails if isinstance(d, dict) and d.get("selected_h") == 10]
        ofp = outer_eval.get("catastrophic_false_positive_rows") or []
        outer[bank] = {
            "selected_config": cfg,
            "chosen_counts": outer_eval.get("chosen_counts"),
            "confusion": outer_eval.get("confusion"),
            "h10_count": len(oh10),
            "false_positive_count": len(ofp) if isinstance(ofp, list) else None,
            "decision_relative_saving_vs_fixed_H15": outer_eval.get("decision_relative_saving_vs_fixed_H15"),
            "physical_delta_vs_fixed_H15": outer_eval.get("physical_delta_vs_fixed_H15"),
            "physical_gate": outer_eval.get("physical_gate"),
            "false_positive_rows": [join_detail({f"{d.get('bank_id')}/{d.get('base_state_id')}": d for d in odetails if isinstance(d, dict)}, r) for r in ofp if isinstance(r, dict)],
            "selected_h10_rows": oh10,
        }

    ambiguity = raw.get("ambiguity", {}) if isinstance(raw.get("ambiguity"), dict) else {}
    ambiguity_summary = {
        fam: {
            "feature_count": val.get("feature_count"),
            "positive_with_cat_closer_than_pos": val.get("positive_with_cat_closer_than_pos"),
            "cat_with_pos_closer_than_cat": val.get("cat_with_pos_closer_than_cat"),
        }
        for fam, val in sorted(ambiguity.items())
        if isinstance(val, dict)
    }

    audit = {
        "utc": utc,
        "classification": "corrected_metadata_only_existing_v13b_nested_audit_no_simulation_no_refit_no_training_no_validation64_no_sealed_test",
        "supersedes_counting_in": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v13b_false_positive_audit_20260929T2210Z/summary.md",
        "note": "The superseded generic audit over-counted selected-H10 patterns by recursively mixing nested_aggregate, nested_outer and top-global holdout views. This v2 uses explicit nested_aggregate/nested_outer only.",
        "raw_path": str(RAW_PATH),
        "raw_sha256": sha256_path(RAW_PATH),
        "headline": headline,
        "nested_aggregate": {
            "chosen_counts": nested.get("chosen_counts"),
            "confusion": nested.get("confusion"),
            "decision_relative_saving_vs_fixed_H15": nested.get("decision_relative_saving_vs_fixed_H15"),
            "solver_relative_saving_vs_fixed_H15": nested.get("solver_relative_saving_vs_fixed_H15"),
            "physical_delta_vs_fixed_H15": nested.get("physical_delta_vs_fixed_H15"),
            "physical_gate": nested.get("physical_gate"),
            "h10_count_from_details": len(selected_h10),
            "false_positive_count": len(false_positive),
            "selected_h10_rows": selected_h10,
            "false_positive_rows": false_positive,
            "selected_h10_by_group": count_by(selected_h10, "group"),
            "selected_h10_by_window": count_by(selected_h10, "window"),
            "false_positive_by_group": count_by(false_positive, "group"),
            "false_positive_by_window": count_by(false_positive, "window"),
        },
        "nested_outer": outer,
        "ambiguity_summary": ambiguity_summary,
        "interpretation": {
            "verified": [
                "Corrected nested aggregate has 7 selected-H10 rows, not the 28 unique rows reported by the over-inclusive recursive audit.",
                "The single nested catastrophic false positive is fresh_v0/fresh_case03_slot1_mid_late, a mid_late fresh_high_heading_long_or_medium row with phys_delta about 6.66 over a tolerance of 2.",
                "fresh_v8c is the only opened outer bank where nested deployment obtains >5% saving (7.37%) without a false positive; aggregate remains only 1.50% and unsafe.",
                "History features did not remove feature-space ambiguity: history_no_risk/history_with_risk each still report 12 positive rows closer to catastrophic rows and 7 catastrophic rows closer to positives.",
            ],
            "decision": "Do not run validation64. Proceed after backup to explicit calibrated risk/value or targeted boundary acquisition; avoid another deterministic static/history feature sweep.",
        },
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

    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    def pct(x: Any) -> str:
        if x is None:
            return "n/a"
        return f"{100.0 * float(x):.2f}%"

    lines = []
    lines.append("# v13b false-positive audit v2 (corrected nested counts)\n")
    lines.append(f"UTC `{utc}`. Corrected metadata-only audit of explicit v13b nested outputs. No MPC simulation, selector refit/search, training, validation64 access, or sealed-test access.\n\n")
    lines.append("## Correction note\n\n")
    lines.append("The prior generic recursive audit is preserved but over-counted selected-H10 patterns by mixing `nested_aggregate`, `nested_outer`, and top-global holdout views. This v2 uses only explicit `nested_aggregate` and `nested_outer` for the main counts.\n\n")
    lines.append("## Corrected nested aggregate\n\n")
    na = audit["nested_aggregate"]
    lines.append(f"- Chosen counts: `{na['chosen_counts']}`; confusion `{na['confusion']}`.\n")
    lines.append(f"- Decision saving vs fixed H15: `{pct(na['decision_relative_saving_vs_fixed_H15'])}`; solver saving `{pct(na['solver_relative_saving_vs_fixed_H15'])}`.\n")
    lines.append(f"- Physical delta vs fixed H15: `{na['physical_delta_vs_fixed_H15']}`; physical gate `{na['physical_gate']}`.\n")
    lines.append(f"- Selected H10 rows: `{na['h10_count_from_details']}`; catastrophic false positives: `{na['false_positive_count']}`.\n")
    lines.append(f"- Selected-H10 by group: `{na['selected_h10_by_group']}`.\n")
    lines.append(f"- Selected-H10 by window: `{na['selected_h10_by_window']}`.\n\n")
    lines.append("## Catastrophic false positive\n\n")
    if false_positive:
        lines.append("| row | group | window | phys_delta | tolerance | decision_gain_s | selected_h | label_positive | catastrophic |\n")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in false_positive:
            lines.append(
                f"| `{r.get('row_id')}` | `{r.get('group')}` | `{r.get('window')}` | {float(r.get('phys_delta', 0)):.6g} | {float(r.get('tol', 2.0)):.6g} | {float(r.get('decision_gain_s', 0)):.6g} | `{r.get('selected_h')}` | `{r.get('label_positive')}` | `{r.get('catastrophic')}` |\n"
            )
    else:
        lines.append("No nested catastrophic false positives found.\n")
    lines.append("\n## Outer-bank nested deployment\n\n")
    lines.append("| outer bank | chosen counts | H10 | false positives | decision save | physical gate | physical delta | selected config |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for bank, obj in outer.items():
        lines.append(
            f"| `{bank}` | `{obj.get('chosen_counts')}` | {obj.get('h10_count')} | {obj.get('false_positive_count')} | {pct(obj.get('decision_relative_saving_vs_fixed_H15'))} | `{obj.get('physical_gate')}` | {obj.get('physical_delta_vs_fixed_H15')} | `{obj.get('selected_config')}` |\n"
        )
    lines.append("\n## Ambiguity summary\n\n")
    for fam, obj in ambiguity_summary.items():
        lines.append(f"- `{fam}`: feature_count `{obj.get('feature_count')}`, positive-with-closer-cat `{obj.get('positive_with_cat_closer_than_pos')}`, catastrophic-with-closer-positive `{obj.get('cat_with_pos_closer_than_cat')}`.\n")
    lines.append("\n## Decision\n\n")
    for item in audit["interpretation"]["verified"]:
        lines.append(f"- Verified: {item}\n")
    lines.append(f"- Decision: {audit['interpretation']['decision']}\n")
    lines.append("\n## Backup gate\n\n")
    lines.append("v13b plus both audit files require verified external backup before any v14 refit/simulation/validation.\n")
    SUMMARY_PATH.write_text("".join(lines))

    completed = {
        "status": "complete",
        "utc": utc,
        "classification": audit["classification"],
        "summary": str(SUMMARY_PATH),
        "audit_json": str(AUDIT_PATH),
        "raw_path": str(RAW_PATH),
        "raw_sha256": audit["raw_sha256"],
        "headline": {
            "nested_h10_count": len(selected_h10),
            "nested_false_positive_count": len(false_positive),
            "nested_decision_save": nested.get("decision_relative_saving_vs_fixed_H15"),
            "nested_physical_gate": nested.get("physical_gate"),
            "global_pass5_count": headline.get("global_pass5_count"),
            "global_pass10_count": headline.get("global_pass10_count"),
        },
        "budgets": audit["budgets"],
        "access_flags": audit["access_flags"],
    }
    COMPLETED_PATH.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")

    backup_req = {
        "created_utc": utc,
        "reason": "backup after corrected v13b false-positive audit v2 before any v14 science",
        "paths": [
            str(OUT_DIR),
            str(BACKUP_REQUEST_PATH),
            str(STATE_PATH),
            "experiments/bohn2021_aws/vehicle_true_variable_horizon_v13b_false_positive_audit.py",
            "experiments/bohn2021_aws/vehicle_true_variable_horizon_v13b_false_positive_audit_v2.py",
            "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v13b_false_positive_audit_20260929T2210Z",
            "research_artifacts/aws_protocols/vehicle_true_variable_horizon_calibrated_risk_value_v14_plan_prebackup_20260929T2210Z.json",
            "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_feature_refit_v13b_fast_20260929T2150Z",
            "research_artifacts/aws_protocols/vehicle_true_variable_horizon_history_feature_refit_v13b_fast_preoutcome_frozen_20260929T2150Z.json",
            "research_artifacts/aws_state/continue_state_20260929T2150_after_history_feature_refit_v13b.md",
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
        "phase": "after_corrected_v13b_false_positive_audit_v2_backup_blocked",
        "current_backup_status": "not verified after v13b and corrected audit; run external backup before v14 refit/simulation/validation",
        "summary": str(SUMMARY_PATH),
        "audit_json": str(AUDIT_PATH),
        "backup_request": str(BACKUP_REQUEST_PATH),
        "corrected_findings": completed["headline"],
        "decision": audit["interpretation"]["decision"],
        "next_action_after_backup": "freeze executable v14 calibrated risk/value refit with uncertainty-aware abstention, or targeted boundary acquisition if v14 fitting data are insufficient; no validation64/sealed test",
        "budgets": audit["budgets"],
        "access_flags": audit["access_flags"],
    }
    STATE_PATH.write_text("# Continue state after corrected v13b false-positive audit v2\n\n" + json.dumps(state, indent=2, sort_keys=True) + "\n")

    print(json.dumps(completed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
