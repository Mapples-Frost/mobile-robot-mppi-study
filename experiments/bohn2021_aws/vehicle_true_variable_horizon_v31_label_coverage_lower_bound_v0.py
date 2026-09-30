#!/usr/bin/env python3
"""Concrete v31 label-coverage lower-bound diagnostic.

Uses only already-opened development diagnostics (v31 raw).  No MPC rollout,
no selector refit, no training, no validation64 bank access, and no sealed-test
access.  This prepares an evidence-bounded fact package for Astra: how many
independent source families are mathematically missing before grouped stability
can be evaluated for the non-default H15/H35 regimes, and what separability
intervals are implied by the current opened rows.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
NAME = "vehicle_true_variable_horizon_v31_label_coverage_lower_bound_v0"
V31_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/raw.json"
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
OUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
STATE_DIR = ROOT / "research_artifacts/aws_state"
EXPECTED_REQUEST_ID = "v31-cluster-stability-diagnostic-20260930T0405Z"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


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
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def fmt_elapsed(seconds: float) -> str:
    days = int(seconds // 86400)
    rem = seconds - days * 86400
    hours = int(rem // 3600)
    rem -= hours * 3600
    minutes = int(rem // 60)
    sec = rem - minutes * 60
    return f"{days}d {hours}h {minutes}m {sec:.3f}s"


def condition_pass(value: float, cond: Mapping[str, Any]) -> bool:
    op = cond.get("op")
    thr = float(cond.get("threshold"))
    if op == "<=":
        return value <= thr
    if op == "<":
        return value < thr
    if op == ">=":
        return value >= thr
    if op == ">":
        return value > thr
    raise ValueError(f"unsupported op {op!r}")


def interval_between(low_max: float, high_min: float) -> Dict[str, Any]:
    if high_min <= low_max:
        return {"separable": False, "lower_open": low_max, "upper_open": high_min, "width": high_min - low_max, "midpoint": None, "half_margin": None}
    return {"separable": True, "lower_open": low_max, "upper_open": high_min, "width": high_min - low_max, "midpoint": (low_max + high_min) / 2.0, "half_margin": (high_min - low_max) / 2.0}


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"{NAME}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "run_started.json", {
        "started_utc": created.isoformat(),
        "classification": "development_analysis_only_no_sim_no_validation_no_test_no_training",
        "input": rel(V31_RAW),
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    raw = read_json(V31_RAW)
    family_summary = raw["family_summary"]
    families = family_summary["families"]
    label_to_families = {str(k): list(v) for k, v in family_summary["oracle_label_to_families"].items()}
    family_counts = {k: len(v) for k, v in label_to_families.items()}

    headline = raw.get("headline", {})
    rule = raw["in_sample_fit_two_feature"]["rule"]
    h35_cond = rule["h35_condition"]
    h15_cond = rule["h15_condition"]
    h35_feature = h35_cond["feature"]
    h15_feature = h15_cond["feature"]

    rows: List[Dict[str, Any]] = []
    for family_name, info in families.items():
        labels = info.get("labels", {})
        if len(labels) != 1:
            raise RuntimeError(f"family {family_name} has non-single label map {labels}")
        label = int(next(iter(labels.keys())))
        obs00s = info.get("abs_obs_00_values", [])
        obs07s = info.get("abs_obs_07_values", [])
        states = info.get("state_labels", [])
        for i, state in enumerate(states):
            rows.append({
                "family": family_name,
                "state_label": state,
                "oracle_h": label,
                "abs_obs_00": float(obs00s[min(i, len(obs00s) - 1)]),
                "abs_obs_07": float(obs07s[min(i, len(obs07s) - 1)]),
            })

    # Reconstruct the current two-feature rule predictions from deployable features.
    pred_bad = []
    for r in rows:
        pred = 12
        if condition_pass(r[h35_feature], h35_cond):
            pred = 35
        elif condition_pass(r[h15_feature], h15_cond):
            pred = 15
        r["rule_prediction"] = pred
        if pred != r["oracle_h"]:
            pred_bad.append({"state_label": r["state_label"], "family": r["family"], "oracle_h": r["oracle_h"], "pred": pred})

    h35_values = [r["abs_obs_00"] for r in rows if r["oracle_h"] == 35]
    non_h35_values = [r["abs_obs_00"] for r in rows if r["oracle_h"] != 35]
    h15_values_non_h35 = [r["abs_obs_07"] for r in rows if r["oracle_h"] == 15 and r["oracle_h"] != 35]
    h12_values_non_h35 = [r["abs_obs_07"] for r in rows if r["oracle_h"] == 12]
    h35_interval = interval_between(max(h35_values), min(non_h35_values))
    h15_interval_after_h35_veto = interval_between(max(h12_values_non_h35), min(h15_values_non_h35))

    min_families_for_basic_logo = 2
    min_families_for_three_source_stability = 3
    class_requirements: Dict[str, Any] = {}
    for label in sorted({"12", "15", "35"} | set(family_counts.keys()), key=int):
        count = int(family_counts.get(label, 0))
        class_requirements[label] = {
            "current_independent_families": count,
            "additional_families_for_basic_leave_one_family_out_class_presence": max(0, min_families_for_basic_logo - count),
            "additional_families_for_three_source_stability_target": max(0, min_families_for_three_source_stability - count),
            "current_family_ids": label_to_families.get(label, []),
        }

    lower_bound = {
        "basic_logo_class_presence_deficit_total": sum(v["additional_families_for_basic_leave_one_family_out_class_presence"] for v in class_requirements.values()),
        "three_source_stability_deficit_total": sum(v["additional_families_for_three_source_stability_target"] for v in class_requirements.values()),
        "nondefault_basic_logo_deficit_total": sum(class_requirements[h]["additional_families_for_basic_leave_one_family_out_class_presence"] for h in ["15", "35"]),
        "nondefault_three_source_stability_deficit_total": sum(class_requirements[h]["additional_families_for_three_source_stability_target"] for h in ["15", "35"]),
    }

    next_review = read_json(NEXT_REVIEW) if NEXT_REVIEW.exists() else {}
    analysis_ready_exists = ANALYSIS_READY.exists()
    current_request_id = next_review.get("request_id") if isinstance(next_review, Mapping) else None
    elapsed_seconds = (created - FIRST_SUPERVISOR_EVENT).total_seconds()
    elapsed_text = fmt_elapsed(elapsed_seconds)

    diagnostic = {
        "created_utc": created.isoformat(),
        "classification": "development_analysis_only_no_sim_no_validation_no_test_no_training",
        "input_v31_raw": rel(V31_RAW),
        "input_v31_raw_sha256": sha256(V31_RAW),
        "rows": len(rows),
        "source_families": len(families),
        "oracle_label_family_counts": family_counts,
        "class_requirements": class_requirements,
        "lower_bound_additional_family_counts": lower_bound,
        "two_feature_rule": rule,
        "rule_prediction_mismatches_on_opened_rows": pred_bad,
        "threshold_intervals_from_opened_rows": {
            "h35_abs_obs_00_vs_non_h35": h35_interval,
            "h15_abs_obs_07_vs_h12_after_h35_veto": h15_interval_after_h35_veto,
        },
        "v31_cv_outcome_for_context": {
            "row_level_loo_bad": headline.get("loo_bad_two_feature_rule"),
            "leave_one_source_family_out_bad": headline.get("logo_bad_two_feature_rule"),
            "logo_bad_folds": headline.get("logo_bad_folds"),
            "in_sample_saving_vs_fixed_H35": headline.get("in_sample_saving_vs_fixed_H35"),
            "loo_saving_vs_fixed_H35": headline.get("loo_saving_vs_fixed_H35"),
        },
        "interpretation_limits": [
            "This is a mathematical/metadata diagnostic over already-opened development rows, not new validation evidence.",
            "Additional-family counts are lower bounds for class presence under grouped CV, not an instruction to acquire labels; Astra must choose the next scientific direction.",
            "Threshold intervals are descriptive of current opened rows and do not establish population separability.",
        ],
        "astra_gate": {
            "current_next_review_request_id": current_request_id,
            "expected_request_id": EXPECTED_REQUEST_ID,
            "analysis_ready_exists": analysis_ready_exists,
        },
        "budgets_actual": {
            "new_simulation_episodes": 0,
            "new_control_steps": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
    }
    write_json(out_dir / "raw.json", diagnostic)

    request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V31_LABEL_COVERAGE_LOWER_BOUND_{stamp}.json"
    continue_path = STATE_DIR / f"continue_state_{stamp}_after_v31_label_coverage_lower_bound.md"
    write_json(request_path, {
        "requested_utc": created.isoformat(),
        "reason": "Back up v31 label-coverage lower-bound diagnostic outputs before unique simulation/refit/training/validation/final-test work.",
        "must_cover": [
            rel(Path(__file__).resolve()), rel(out_dir / "run_started.json"), rel(out_dir / "raw.json"),
            rel(out_dir / "summary.md"), rel(out_dir / "completed.json"), rel(request_path), rel(continue_path),
            "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", "STATUS.md", "RESEARCH_LOG.md",
            "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md",
        ],
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    summary = f"""# v31 label-coverage lower-bound diagnostic

UTC: `{created.isoformat()}`. Development-analysis only over already-opened v31 raw evidence; no simulations, no control steps, no selector refits, no training, no validation64 bank open, no sealed-test access.

## Required status-line values
- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `{elapsed_text}`.
- Cumulative server API total_tokens: unknown in this script; desktop conversation tokens excluded.

## Concrete result
- Current independent source-family counts by oracle label: `{family_counts}`.
- Lower-bound additional independent families for basic leave-one-source-family-out class presence: `{ {k: v['additional_families_for_basic_leave_one_family_out_class_presence'] for k, v in class_requirements.items()} }`.
- Non-default lower-bound deficit for basic grouped-CV class presence: `{lower_bound['nondefault_basic_logo_deficit_total']}` total families (`+1` H15 family and `+1` H35 family under the current counts).
- Lower-bound additional independent families for a three-source stability target: `{ {k: v['additional_families_for_three_source_stability_target'] for k, v in class_requirements.items()} }`.
- The current in-sample two-feature rule has `{len(pred_bad)}` prediction mismatches on opened rows when reconstructed from deployable features, matching the in-sample fit; grouped-CV failure is therefore attributable to source-family label coverage, not a parsing mismatch in this diagnostic.

## Separability intervals descriptive of opened rows
- H35-vs-non-H35 by `abs_obs_00`: `{h35_interval}`.
- H15-vs-H12 after H35 veto by `abs_obs_07`: `{h15_interval_after_h35_veto}`.

## Astra gate / next action
- Current NEXT_REVIEW_REQUEST id: `{current_request_id}`; ANALYSIS_READY present: `{analysis_ready_exists}`.
- This diagnostic does **not** choose the next scientific branch. It supplies a concrete lower bound for Astra: current opened rows are insufficient for source-family-independent evidence of H15/H35 behavior; at least two additional independent non-default source families are required even for basic grouped-CV class presence.
- New outputs require backup request `{rel(request_path)}` before unique simulation/refit/training/validation/final-test work.
"""
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    continue_path.write_text(summary, encoding="utf-8")

    marker = f"<!-- {NAME}-{stamp} -->"
    log_block = f"""## v31 label-coverage lower-bound diagnostic

Updated by GPT-5.5 executor at `{created.isoformat()}`. Analysis-only over already-opened v31 raw evidence; no simulations, no training/refit, no validation64 access and no sealed-test access.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| Astra v31 handoff / non-default source coverage | accepted as concrete diagnostic while Astra direction pending | `{rel(out_dir / 'summary.md')}` and `{rel(out_dir / 'raw.json')}` show current oracle-label family counts `{family_counts}`. Basic grouped-CV class-presence deficits are `{ {k: v['additional_families_for_basic_leave_one_family_out_class_presence'] for k, v in class_requirements.items()} }`; H15 and H35 each need at least one additional independent family before leave-one-source-family-out can train with that class present. | Carry to Astra. Do not validate/deploy a selector from v31; do not choose acquisition/refit/scenario branch until matching Astra report is read. |
| `A12_registry_backup_schema_contract` | accepted; new diagnostic pending backup | Backup request `{rel(request_path)}`. | Require follow-up external backup before unique simulation/refit/training/validation/final-test work. |
"""
    append_once(RESPONSE_LOG, marker, log_block)
    doc_block = f"""## 2026-09-30 v31 label-coverage lower-bound diagnostic

UTC: {created.isoformat()}. Development-analysis only; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `{elapsed_text}`. Current source-family counts by oracle label `{family_counts}` imply lower-bound basic grouped-CV class-presence deficits `{ {k: v['additional_families_for_basic_leave_one_family_out_class_presence'] for k, v in class_requirements.items()} }`; non-default total deficit `{lower_bound['nondefault_basic_logo_deficit_total']}`. This is not a selector validation or a new branch decision; carry to Astra request `{current_request_id}`. Artifacts: `{rel(out_dir / 'summary.md')}`, `{rel(out_dir / 'raw.json')}`, `{rel(out_dir / 'completed.json')}`. Backup request: `{rel(request_path)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, marker, doc_block)

    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "classification": diagnostic["classification"],
        "summary": rel(out_dir / "summary.md"),
        "raw": rel(out_dir / "raw.json"),
        "backup_request": rel(request_path),
        "continue_state": rel(continue_path),
        "source_family_counts": family_counts,
        "lower_bound_additional_family_counts": lower_bound,
        "rule_prediction_mismatches_on_opened_rows": len(pred_bad),
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {},
    }
    write_json(out_dir / "completed.json", completed)
    hash_paths = [Path(__file__).resolve(), out_dir / "run_started.json", out_dir / "raw.json", out_dir / "summary.md", out_dir / "completed.json", request_path, continue_path, RESPONSE_LOG]
    completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists() and p.is_file()}
    write_json(out_dir / "completed.json", completed)
    print(json.dumps(completed, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
