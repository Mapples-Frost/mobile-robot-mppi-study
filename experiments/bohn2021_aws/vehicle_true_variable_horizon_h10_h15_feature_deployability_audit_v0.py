#!/usr/bin/env python3
"""Offline deployability/leakage audit for the H10/H15 residual-CV result.

Development-only.  No simulation, no training/refit, no validation64 read, and no
sealed-test read.

Motivation: v0b found two apparent H10-vs-H15 CV passes against fixed true H15,
but both were driven by `cat:selection_group`.  Before spending rollout budget on
selector-overhead timing, this diagnostic checks whether those pass features are
online/state-observable or instead source/protocol metadata that would leak
scenario-bank identity.  A non-deployable metadata pass must not be treated as a
learned adaptive MPC-horizon policy.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_h10_h15_feature_deployability_audit_v0"
STAMP = "20260929T0850Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
V0B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair_20260929T0845Z/raw.json"
V0B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair_20260929T0845Z/completed.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-h10-h15-feature-deployability-audit-v0-{STAMP}"


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def clean(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return rel(obj)
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    return obj


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sf(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        y = float(value)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing prerequisite completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"prerequisite did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is True or obj.get("validation64_bank_opened") is True:
        raise ContractError(f"unexpected validation/test access in prerequisite marker: {rel(path)}")
    return obj


def collect_policy_result(dataset: str, policy: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    cv = ((payload.get("cv") or {}).get(policy) or {})
    ls = cv.get("leave_state_out") or {}
    lc = cv.get("leave_case_out") or {}
    return {
        "dataset": dataset,
        "policy": policy,
        "leave_state_core5": bool(ls.get("core_pass_5pct")),
        "leave_case_core5": bool(lc.get("core_pass_5pct")),
        "leave_state_core10": bool(ls.get("core_pass_10pct")),
        "leave_case_core10": bool(lc.get("core_pass_10pct")),
        "passes_both_5pct": bool(ls.get("core_pass_5pct") and lc.get("core_pass_5pct")),
        "passes_both_10pct": bool(ls.get("core_pass_10pct") and lc.get("core_pass_10pct")),
        "leave_state": {
            "horizon_counts": ls.get("horizon_counts"),
            "physical_delta_vs_H15": sf(ls.get("physical_delta_vs_H15")),
            "physical_tolerance_vs_H15": sf(ls.get("physical_tolerance_vs_H15")),
            "decision_relative_saving_vs_H15": sf(ls.get("decision_relative_saving_vs_H15")),
            "solver_relative_saving_vs_H15": sf(ls.get("solver_relative_saving_vs_H15")),
            "unsafe_chosen_groups": ls.get("unsafe_chosen_groups"),
            "oracle_regret_physical_sum": sf(ls.get("oracle_regret_physical_sum")),
        },
        "leave_case": {
            "horizon_counts": lc.get("horizon_counts"),
            "physical_delta_vs_H15": sf(lc.get("physical_delta_vs_H15")),
            "physical_tolerance_vs_H15": sf(lc.get("physical_tolerance_vs_H15")),
            "decision_relative_saving_vs_H15": sf(lc.get("decision_relative_saving_vs_H15")),
            "solver_relative_saving_vs_H15": sf(lc.get("solver_relative_saving_vs_H15")),
            "unsafe_chosen_groups": lc.get("unsafe_chosen_groups"),
            "oracle_regret_physical_sum": sf(lc.get("oracle_regret_physical_sum")),
        },
        "models_preview": {
            "leave_state": (ls.get("models_preview") or [])[:3],
            "leave_case": (lc.get("models_preview") or [])[:3],
        },
    }


def extract_rule_values(policy_result: Mapping[str, Any]) -> List[str]:
    vals = set()
    for side in ("leave_state", "leave_case"):
        for item in ((policy_result.get("models_preview") or {}).get(side) or []):
            rules = (((item.get("model") or {}).get("rules")) or {})
            vals.update(str(k) for k in rules.keys())
    return sorted(vals)


def feature_status(policy: str, rule_values: Iterable[str]) -> Dict[str, Any]:
    values = list(rule_values)
    if policy == "obs_stump":
        return {
            "online_state_observable": True,
            "allowed_for_deployable_selector_claim": True,
            "risk": "simple state-observation proxy; capacity is tiny but feature source is deployable",
            "rule_values": values,
        }
    if policy == "cat:selection_group":
        suspicious_tokens = ["stress_v", "true_h", "anchor", "control", "primary", "lower_stress", "high_heading"]
        suspicious = sorted({v for v in values if any(tok in v for tok in suspicious_tokens)})
        return {
            "online_state_observable": False,
            "allowed_for_deployable_selector_claim": False,
            "risk": "uses scenario-bank/protocol stratum labels rather than sensor/state features; a pass can encode data-origin leakage and is valid only as a stratified diagnostic",
            "rule_values": values,
            "suspicious_values": suspicious,
        }
    if policy == "cat:role_family":
        return {
            "online_state_observable": False,
            "allowed_for_deployable_selector_claim": False,
            "risk": "uses mining/target role metadata (risk_anchor, control, previously_used, etc.), unavailable to a deployed controller",
            "rule_values": values,
        }
    if policy == "cat:terminal_profile":
        return {
            "online_state_observable": False,
            "allowed_for_deployable_selector_claim": False,
            "risk": "uses experimental terminal-profile arm; not a state-dependent controller input and also implicated in label flips",
            "rule_values": values,
        }
    if policy == "cat:window":
        return {
            "online_state_observable": False,
            "allowed_for_deployable_selector_claim": False,
            "risk": "uses hand-labelled branch-window metadata (early/middle/late/missing), not a frozen online feature mapping; can be revisited only with an explicit time/state-derived rule",
            "rule_values": values,
        }
    return {"online_state_observable": False, "allowed_for_deployable_selector_claim": False, "risk": "unknown policy feature", "rule_values": values}


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    done_path = OUT_DIR / "completed.json"
    if done_path.exists():
        done = read_json(done_path)
        print(json.dumps({"already_completed": rel(done_path), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0

    completed_ok(V0B_DONE)
    v0b = read_json(V0B_RAW)
    if v0b.get("validation64_bank_opened") is True or v0b.get("sealed_test_accessed") is True:
        raise ContractError("v0b unexpectedly indicates validation/test access")

    policies = ["cat:selection_group", "cat:role_family", "cat:terminal_profile", "cat:window", "obs_stump"]
    dataset_reports: Dict[str, Any] = {}
    all_policy_rows: List[Dict[str, Any]] = []
    for dataset, payload in sorted((v0b.get("datasets") or {}).items()):
        rows = []
        for policy in policies:
            if policy not in (payload.get("cv") or {}):
                continue
            r = collect_policy_result(dataset, policy, payload)
            r["feature_status"] = feature_status(policy, extract_rule_values(r))
            rows.append(r)
            all_policy_rows.append(r)
        dataset_reports[dataset] = {
            "n": payload.get("n"),
            "state_count": payload.get("state_count"),
            "case_count": payload.get("case_count"),
            "label_counts": payload.get("label_counts"),
            "policy_rows": rows,
        }

    original_passes = v0b.get("deployable_cv_passes") or []
    valid_online_passes = [
        r for r in all_policy_rows
        if r.get("passes_both_5pct") and (r.get("feature_status") or {}).get("allowed_for_deployable_selector_claim")
    ]
    metadata_passes = [
        r for r in all_policy_rows
        if r.get("passes_both_5pct") and not (r.get("feature_status") or {}).get("allowed_for_deployable_selector_claim")
    ]
    selection_group_passes = [r for r in metadata_passes if r.get("policy") == "cat:selection_group"]
    obs_rows = [r for r in all_policy_rows if r.get("policy") == "obs_stump"]

    leakage_blocks_selector_rollout = bool(original_passes) and not bool(valid_online_passes)
    if leakage_blocks_selector_rollout:
        decision = (
            "block the proposed selector-overhead rollout based on cat:selection_group; "
            "the only v0b passes use non-online scenario/protocol metadata.  Next bounded action should be a state-observable H10/H15 model-CV/refit or terminal-value calibration diagnostic, not a rollout of a metadata oracle."
        )
    else:
        decision = (
            "a state-observable pass exists; after backup freeze a tiny selector-overhead rollout using only the passing online features and fixed true H10/H15/H25 baselines"
        )

    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_H10_H15_FEATURE_DEPLOYABILITY_AUDIT_V0_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_offline_no_simulation_feature_deployability_leakage_audit_not_validation_not_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "input_v0b_raw": rel(V0B_RAW),
        "input_v0b_completed": rel(V0B_DONE),
        "v0b_original_deployable_cv_passes": original_passes,
        "valid_online_deployable_passes": valid_online_passes,
        "metadata_feature_passes": metadata_passes,
        "selection_group_passes": selection_group_passes,
        "obs_stump_rows": obs_rows,
        "dataset_reports": dataset_reports,
        "leakage_blocks_selector_rollout": leakage_blocks_selector_rollout,
        "decision": decision,
        "next_recommended_diagnostic": {
            "name": "vehicle_true_variable_horizon_state_observable_h10_h15_model_cv_v0",
            "budget": "offline only; 0 simulations; count model fits/refit trials explicitly",
            "allowed_features": ["initial_observation_at_branch", "time/step fraction only if derived online, not window labels", "no source/selection_group/role/terminal_profile"],
            "baselines": ["fixed true H15", "fixed true H10", "oracle H10/H15 upper bound"],
            "gates": ["leave-state-out and leave-case-out physical within tolerance vs H15", ">=5% measured decision saving vs H15", "zero unsafe chosen groups", "nonconstant horizon choices", "no metadata features"],
        },
        "backup_request_after_diagnostic": rel(req),
    }
    write_json(OUT_DIR / "raw.json", raw)

    def fmt_metric(row: Mapping[str, Any], side: str, key: str) -> str:
        val = ((row.get(side) or {}).get(key))
        if isinstance(val, float):
            return f"{val:.6g}"
        return str(val)

    lines = [
        "# Vehicle true-variable-H H10/H15 feature-deployability audit v0",
        "",
        f"UTC: `{created.isoformat()}`. Offline/no-simulation audit of v0b; validation64 and sealed test remain closed.",
        "",
        "## Headline",
        "",
        f"- v0b original deployable-proxy passes: `{len(original_passes)}`.",
        f"- State/online-observable passes after leakage audit: `{len(valid_online_passes)}`.",
        f"- Metadata-feature passes: `{len(metadata_passes)}`; selection_group passes: `{len(selection_group_passes)}`.",
        f"- Leakage blocks selector rollout: `{leakage_blocks_selector_rollout}`.",
        f"- Decision: {decision}",
        "",
        "## Policy feature classification",
        "",
        "| dataset | policy | pass 5% both folds | state-observable? | allowed deployable claim? | LS dec save | LC dec save | LS physΔ/tol | LC physΔ/tol | feature risk |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in all_policy_rows:
        fs = row.get("feature_status") or {}
        lines.append("| `%s` | `%s` | `%s` | `%s` | `%s` | %s | %s | %s/%s | %s/%s | %s |" % (
            row.get("dataset"), row.get("policy"), row.get("passes_both_5pct"),
            fs.get("online_state_observable"), fs.get("allowed_for_deployable_selector_claim"),
            fmt_metric(row, "leave_state", "decision_relative_saving_vs_H15"),
            fmt_metric(row, "leave_case", "decision_relative_saving_vs_H15"),
            fmt_metric(row, "leave_state", "physical_delta_vs_H15"), fmt_metric(row, "leave_state", "physical_tolerance_vs_H15"),
            fmt_metric(row, "leave_case", "physical_delta_vs_H15"), fmt_metric(row, "leave_case", "physical_tolerance_vs_H15"),
            str(fs.get("risk", ""))[:160].replace("|", "/"),
        ))
    lines += [
        "",
        "## Interpretation",
        "",
        "The positive v0b CV result is a useful stratified clue that scenario family affects the H10/H15 tradeoff, but `selection_group` is a bank/protocol label (e.g. stress-v1d target group versus broader-block anchor versus controls), not a deployable state feature.  It must not be promoted to a learned adaptive MPC-horizon controller without an additional state-observable mapping and fresh rollout evidence including selector overhead.",
        "",
        "## Next concrete action",
        "",
        "After backup, run an offline state-observable H10/H15 model-CV/refit diagnostic using only initial observations or explicitly online-derived quantities, with fixed true H15 as the primary comparator.  If no online-feature model passes both leave-state and leave-case gates, pivot to terminal-value/objective calibration or scenario design rather than spending rollout budget on a metadata selector.",
        "",
        f"Backup request after this audit: `{rel(req)}`.",
    ]
    summary_path = OUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(req, {
        "requested_utc": created.isoformat(),
        "reason": "backup feature-deployability/leakage audit before any selector rollout, simulation, training or refit",
        "backup_required_before_more_simulations": True,
        "backup_required_before_training_or_refit": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(req)],
    })
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H H10/H15 feature-deployability audit v0

UTC: {created.isoformat()}. Offline/no-simulation audit of the v0b H10/H15 residual-CV result. Validation64 and sealed test stayed closed; no training/refit. Original v0b deployable-proxy passes={len(original_passes)}, but state/online-observable passes after feature audit={len(valid_online_passes)}; metadata-feature passes={len(metadata_passes)}. Leakage blocks selector rollout={leakage_blocks_selector_rollout}. Decision: {decision}. Artifacts: `{rel(summary_path)}`, `{rel(OUT_DIR / 'raw.json')}`.
""")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE_PATH, req, V0B_RAW, V0B_DONE]
    done = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "classification": raw["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "headline": {
            "original_v0b_deployable_proxy_passes": len(original_passes),
            "valid_online_deployable_passes": len(valid_online_passes),
            "metadata_feature_passes": len(metadata_passes),
            "leakage_blocks_selector_rollout": leakage_blocks_selector_rollout,
            "decision": decision,
        },
        "backup_request": rel(req),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(done_path, done)
    print(json.dumps({"completed": rel(done_path), "summary": rel(summary_path), "headline": done["headline"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
