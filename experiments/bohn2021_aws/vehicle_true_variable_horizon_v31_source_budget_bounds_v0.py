#!/usr/bin/env python3
"""v31 source-coverage budget/identifiability bounds diagnostic.

Concrete, reversible, analysis-only preparation for Astra.  Uses already-opened
v31/v29 development artifacts to quantify what the current source-family
coverage can and cannot identify, and the *minimum* additional labelled-family
rollout budgets that would be required if Astra chooses fresh source-independent
triage-label acquisition.  This script does not choose that branch, does not run
MPC, does not refit a selector/value model, and does not open validation/test.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
NAME = "vehicle_true_variable_horizon_v31_source_budget_bounds_v0"
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
OUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
STATE_DIR = ROOT / "research_artifacts/aws_state"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
V31_LOWER_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_label_coverage_lower_bound_v0_20260930T044251Z/raw.json"
V31_CLUSTER_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/raw.json"
V29_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/completed.json"
V29_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/summary.md"
EXPECTED_PREV_REQUEST = "v31-label-coverage-lower-bound-20260930T044251Z"
NEW_REQUEST_ID_PREFIX = "v31-source-coverage-budget-bounds"


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


def recursive_numbers(obj: Any, key_fragments: Iterable[str]) -> List[Tuple[str, float]]:
    frags = [s.lower() for s in key_fragments]
    out: List[Tuple[str, float]] = []
    def walk(x: Any, path: str) -> None:
        if isinstance(x, Mapping):
            for k, v in x.items():
                ks = str(k).lower()
                p = f"{path}.{k}" if path else str(k)
                if any(f in ks for f in frags) and isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append((p, float(v)))
                walk(v, p)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
    walk(obj, "")
    return out


def parse_budget_from_v29() -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "episodes": None,
        "control_steps": None,
        "mean_control_steps_per_episode": None,
        "source": None,
        "fallback_used": False,
    }
    if V29_COMPLETED.exists():
        data = read_json(V29_COMPLETED)
        ep_candidates = recursive_numbers(data, ["episode"])
        step_candidates = recursive_numbers(data, ["control_step", "control steps", "steps"])
        # Prefer exact top-level or semantically direct actual budget fields when present.
        for p, v in ep_candidates:
            if p.lower().endswith("new_simulation_episodes") or p.lower().endswith("episodes") or p.lower().endswith("episode_count"):
                if v > 0:
                    evidence["episodes"] = int(v)
                    evidence["episodes_path"] = p
                    break
        for p, v in step_candidates:
            pl = p.lower()
            if "control" in pl and v > 0:
                evidence["control_steps"] = int(v)
                evidence["control_steps_path"] = p
                break
        evidence["source"] = rel(V29_COMPLETED)
    if (evidence["episodes"] is None or evidence["control_steps"] is None) and V29_SUMMARY.exists():
        text = V29_SUMMARY.read_text(encoding="utf-8", errors="replace")
        m_ep = re.search(r"(\d+)\s+episodes", text, flags=re.I)
        m_steps = re.search(r"(\d+)\s+control steps", text, flags=re.I)
        if evidence["episodes"] is None and m_ep:
            evidence["episodes"] = int(m_ep.group(1))
            evidence["episodes_path"] = rel(V29_SUMMARY) + ":regex episodes"
        if evidence["control_steps"] is None and m_steps:
            evidence["control_steps"] = int(m_steps.group(1))
            evidence["control_steps_path"] = rel(V29_SUMMARY) + ":regex control steps"
        evidence["source"] = (evidence.get("source") or "") + ";" + rel(V29_SUMMARY)
    # Supervisor context and prior summaries explicitly record 44 episodes / 2557 steps.
    if evidence["episodes"] is None or evidence["control_steps"] is None:
        evidence["episodes"] = 44
        evidence["control_steps"] = 2557
        evidence["source"] = "fallback_from_supervisor_context_and_v29_summary: 44 episodes, 2557 control steps"
        evidence["fallback_used"] = True
    evidence["mean_control_steps_per_episode"] = float(evidence["control_steps"]) / float(evidence["episodes"])
    return evidence


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"{NAME}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "run_started.json", {
        "started_utc": created.isoformat(),
        "classification": "development_analysis_only_no_sim_no_validation_no_test_no_training_no_refit",
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    lower = read_json(V31_LOWER_RAW)
    cluster = read_json(V31_CLUSTER_RAW)
    counts = {str(k): int(v) for k, v in lower["oracle_label_family_counts"].items()}
    family_ids = {k: v["current_family_ids"] for k, v in lower["class_requirements"].items()}
    threshold_intervals = lower.get("threshold_intervals_from_opened_rows", {})
    v29_budget = parse_budget_from_v29()
    mean_steps = float(v29_budget["mean_control_steps_per_episode"])

    targets = {
        "basic_logo_class_presence": 2,
        "three_source_stability_target": 3,
    }
    horizon_sets = {
        "triage_H12_H15_H35": [12, 15, 35],
        "triage_plus_reference_H12_H15_H25_H35": [12, 15, 25, 35],
    }
    budget_bounds: Dict[str, Any] = {}
    for target_name, min_families_per_label in targets.items():
        add_by_label = {label: max(0, min_families_per_label - counts.get(label, 0)) for label in ["12", "15", "35"]}
        add_nondefault = add_by_label["15"] + add_by_label["35"]
        # Same-family extra rows do not increase source-family count; include explicit structural note.
        per_hset = {}
        for hset_name, hs in horizon_sets.items():
            min_episodes = add_nondefault * len(hs)
            per_hset[hset_name] = {
                "horizons_per_new_family_state": hs,
                "minimum_new_independent_nondefault_family_states": add_nondefault,
                "minimum_rollout_episodes": min_episodes,
                "estimated_control_steps_from_v29_mean": int(math.ceil(min_episodes * mean_steps)),
                "estimate_basis_mean_steps_per_episode": mean_steps,
            }
        budget_bounds[target_name] = {
            "minimum_families_per_label": min_families_per_label,
            "additional_independent_families_by_label": add_by_label,
            "additional_nondefault_families_total": add_nondefault,
            "rollout_budget_lower_bounds": per_hset,
        }

    impossibility_notes = [
        "With H15 and H35 each represented by one independent source family, any leave-one-source-family-out fold that holds out that family has zero empirical examples of that label in training.",
        "Adding more rows from the same H15/H35 source family cannot repair grouped source-family class presence; at least one new independent H15-like family and one new independent H35-like family are required for basic class-presence under LOGO.",
        "This is a lower-bound identifiability/budget statement for supervised empirical selectors, not proof that acquisition is the best next scientific branch and not a validation result.",
    ]
    same_family_extra_rows_effect = {
        "current_nondefault_family_counts": {"15": counts.get("15", 0), "35": counts.get("35", 0)},
        "same_family_rows_needed_to_change_independent_family_count": "impossible_by_definition",
        "minimum_new_source_families_for_any_basic_grouped_cv_nondefault_class_presence": budget_bounds["basic_logo_class_presence"]["additional_nondefault_families_total"],
    }

    prev_request = read_json(NEXT_REVIEW) if NEXT_REVIEW.exists() else {}
    prev_request_id = prev_request.get("request_id") if isinstance(prev_request, Mapping) else None
    analysis_ready_exists = ANALYSIS_READY.exists()
    elapsed = fmt_elapsed((created - FIRST_SUPERVISOR_EVENT).total_seconds())

    diagnostic = {
        "created_utc": created.isoformat(),
        "classification": "development_analysis_only_no_sim_no_validation_no_test_no_training_no_refit",
        "inputs": {
            "v31_lower_bound_raw": rel(V31_LOWER_RAW),
            "v31_lower_bound_raw_sha256": sha256(V31_LOWER_RAW),
            "v31_cluster_raw": rel(V31_CLUSTER_RAW),
            "v31_cluster_raw_sha256": sha256(V31_CLUSTER_RAW),
            "v29_budget_source": v29_budget,
        },
        "current_oracle_label_independent_source_family_counts": counts,
        "current_family_ids_by_label": family_ids,
        "threshold_intervals_descriptive_only": threshold_intervals,
        "budget_bounds_if_astra_selects_fresh_source_label_acquisition": budget_bounds,
        "same_family_extra_rows_do_not_repair_grouped_cv": same_family_extra_rows_effect,
        "structural_identifiability_notes": impossibility_notes,
        "not_a_branch_decision": True,
        "not_validation_or_test_evidence": True,
        "astra_gate": {
            "previous_request_id": prev_request_id,
            "expected_previous_request_id": EXPECTED_PREV_REQUEST,
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

    request_id = f"{NEW_REQUEST_ID_PREFIX}-{stamp}"
    next_review_obj = {
        "created": created.isoformat(),
        "evidence_paths": [
            rel(out_dir / "summary.md"), rel(out_dir / "raw.json"), rel(out_dir / "completed.json"),
            rel(V31_LOWER_RAW), rel(V31_CLUSTER_RAW),
            "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md",
        ],
        "experiment_id": NAME,
        "question": "Astra should incorporate this source-coverage budget/identifiability diagnostic into the v29/v30b/v31 direction decision. It quantifies that current independent source-family counts are H12=3, H15=1, H35=1; same-family extra rows cannot repair grouped-CV class presence; if Astra chooses fresh source-independent triage-label acquisition, the lower bound is +1 H15-like and +1 H35-like independent family for basic LOGO class presence (6 H12/H15/H35 rollout episodes or 8 if H25 is included), or +2 each for a three-source stability target. Decide whether acquisition, terminal-risk/value refit/training, or scenario/comparison redesign is the next scientific action. GPT-5.5 should not treat this as branch selection or validation.",
        "request_id": request_id,
        "supersedes_request_id": prev_request_id,
        "status": "analysis_requested",
        "trigger": "source_coverage_budget_bounds_after_v31_label_coverage_lower_bound",
        "access_and_budget_note": diagnostic["budgets_actual"] | {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "operational_note": "Post-diagnostic outputs and this refreshed request require external backup before unique simulation/refit/training/validation/final-test work."
    }
    # Summary first because NEXT_REVIEW cites it.
    basic = budget_bounds["basic_logo_class_presence"]
    three = budget_bounds["three_source_stability_target"]
    summary = f"""# v31 source-coverage budget/identifiability bounds

UTC: `{created.isoformat()}`. Analysis-only over already-opened development artifacts; no simulations, no control steps, no selector refit, no training, no validation64 bank access and no sealed-test access.

## Required status-line values
- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `{elapsed}`.
- Cumulative server API total_tokens: unknown to this script; desktop conversation tokens excluded.

## Concrete diagnostic result
- Current independent source-family counts by oracle label: `{counts}`.
- Same-family extra rows cannot change independent source-family counts; therefore they cannot fix the grouped-CV missing-class condition for H15/H35.
- If Astra chooses fresh source-independent triage-label acquisition, the **minimum lower bound** for basic leave-one-source-family-out class presence is `{basic['additional_independent_families_by_label']}` = `{basic['additional_nondefault_families_total']}` additional non-default source families.
  - H12/H15/H35 triage lower bound: `{basic['rollout_budget_lower_bounds']['triage_H12_H15_H35']['minimum_rollout_episodes']}` rollout episodes; estimated `{basic['rollout_budget_lower_bounds']['triage_H12_H15_H35']['estimated_control_steps_from_v29_mean']}` control steps using v29 mean `{mean_steps:.3f}` steps/episode.
  - Including H25 reference lower bound: `{basic['rollout_budget_lower_bounds']['triage_plus_reference_H12_H15_H25_H35']['minimum_rollout_episodes']}` rollout episodes; estimated `{basic['rollout_budget_lower_bounds']['triage_plus_reference_H12_H15_H25_H35']['estimated_control_steps_from_v29_mean']}` control steps.
- For a three-source stability target, the lower bound is `{three['additional_independent_families_by_label']}` = `{three['additional_nondefault_families_total']}` additional non-default source families.
  - H12/H15/H35 triage lower bound: `{three['rollout_budget_lower_bounds']['triage_H12_H15_H35']['minimum_rollout_episodes']}` rollout episodes; estimated `{three['rollout_budget_lower_bounds']['triage_H12_H15_H35']['estimated_control_steps_from_v29_mean']}` control steps.
  - Including H25 reference lower bound: `{three['rollout_budget_lower_bounds']['triage_plus_reference_H12_H15_H25_H35']['minimum_rollout_episodes']}` rollout episodes; estimated `{three['rollout_budget_lower_bounds']['triage_plus_reference_H12_H15_H25_H35']['estimated_control_steps_from_v29_mean']}` control steps.

## Limits / gate
- This is not a branch decision, not a selector validation, and not final-test evidence.
- It only tells Astra that if source-label acquisition is selected, same-family densification is structurally insufficient and the above independent-family rollout counts are lower bounds.
- Refreshed Astra request id: `{request_id}`; ANALYSIS_READY present at run time: `{analysis_ready_exists}`.
"""
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_json(NEXT_REVIEW, next_review_obj)

    backup_request = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V31_SOURCE_BUDGET_BOUNDS_{stamp}.json"
    continue_path = STATE_DIR / f"continue_state_{stamp}_after_v31_source_budget_bounds.md"
    write_json(backup_request, {
        "requested_utc": created.isoformat(),
        "reason": "Back up v31 source-coverage budget bounds diagnostic and refreshed Astra request before unique simulation/refit/training/validation/final-test work.",
        "must_cover": [
            rel(Path(__file__).resolve()), rel(out_dir / "run_started.json"), rel(out_dir / "summary.md"), rel(out_dir / "raw.json"), rel(out_dir / "completed.json"),
            rel(NEXT_REVIEW), rel(RESPONSE_LOG), rel(backup_request), rel(continue_path), "EXPERIMENT_REGISTRY.csv", "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"
        ],
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })
    continue_path.write_text(summary + "\n\nNext: wait for matching/superseding Astra ANALYSIS_READY and external backup before unique science.\n", encoding="utf-8")

    marker = f"<!-- {NAME}-{stamp} -->"
    log_block = f"""## v31 source-coverage budget/identifiability bounds

Updated by GPT-5.5 executor at `{created.isoformat()}`. Analysis-only; no simulations, no training/refit, no validation64 access and no sealed-test access.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| Astra v31 handoff / non-default source coverage | accepted as bounded preparatory diagnostic while Astra direction pending | `{rel(out_dir / 'summary.md')}` and `{rel(out_dir / 'raw.json')}` show current family counts `{counts}`. Same-family extra rows cannot repair LOGO missing-class structure. If Astra selects fresh source-label acquisition, basic grouped-CV class-presence lower bound is +1 H15-like and +1 H35-like independent family = 6 H12/H15/H35 rollout episodes or 8 including H25; three-source target is +2 each = 12 or 16 rollout episodes. | Refreshed NEXT_REVIEW_REQUEST `{request_id}`. Do not start acquisition/refit/scenario branch until matching/superseding Astra report is read and backup gate is satisfied. |
| `A12_registry_backup_schema_contract` | accepted; new diagnostic pending backup | Backup request `{rel(backup_request)}`. | Require follow-up external backup before unique simulation/refit/training/validation/final-test work. |
"""
    append_once(RESPONSE_LOG, marker, log_block)
    doc_block = f"""## 2026-09-30 v31 source-coverage budget/identifiability bounds

UTC: {created.isoformat()}. Analysis-only; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `{elapsed}`. Current source-family counts `{counts}`. Same-family densification cannot fix LOGO missing-class coverage. If Astra chooses fresh source-independent labels, basic grouped-CV lower bound is `{basic['additional_independent_families_by_label']}` with `{basic['rollout_budget_lower_bounds']['triage_H12_H15_H35']['minimum_rollout_episodes']}` H12/H15/H35 rollout episodes (`{basic['rollout_budget_lower_bounds']['triage_plus_reference_H12_H15_H25_H35']['minimum_rollout_episodes']}` including H25); three-source target lower bound is `{three['additional_independent_families_by_label']}` with `{three['rollout_budget_lower_bounds']['triage_H12_H15_H35']['minimum_rollout_episodes']}` or `{three['rollout_budget_lower_bounds']['triage_plus_reference_H12_H15_H25_H35']['minimum_rollout_episodes']}` episodes. Not a branch decision or validation. Artifacts: `{rel(out_dir / 'summary.md')}`, `{rel(out_dir / 'raw.json')}`, `{rel(out_dir / 'completed.json')}`. Backup request: `{rel(backup_request)}`. Astra request `{request_id}`.
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
        "backup_request": rel(backup_request),
        "continue_state": rel(continue_path),
        "refreshed_astra_request": rel(NEXT_REVIEW),
        "request_id": request_id,
        "current_counts": counts,
        "basic_logo_min_additional_nondefault_families": basic["additional_nondefault_families_total"],
        "three_source_min_additional_nondefault_families": three["additional_nondefault_families_total"],
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {},
    }
    write_json(out_dir / "completed.json", completed)
    hash_paths = [Path(__file__).resolve(), out_dir / "run_started.json", out_dir / "summary.md", out_dir / "raw.json", out_dir / "completed.json", NEXT_REVIEW, RESPONSE_LOG, backup_request, continue_path]
    completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists() and p.is_file()}
    write_json(out_dir / "completed.json", completed)
    print(json.dumps(completed, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
