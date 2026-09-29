#!/usr/bin/env python3
"""Offline terminal-consistency audit for true-variable-H H10/H15 evidence.

Development-only. No simulation, no rollout, no training/refit, no validation64
read, and no sealed-test read.

Why this exists: the latest state-observable H10/H15 nested-CV diagnostic found
one pass only on the pooled risk-anchor subset, while no terminal-fixed dataset
passed.  Earlier diagnostics showed H labels flipping across terminal profiles.
This audit asks whether a terminal-consistent target (e.g. choose H10 only when
both available terminal profiles agree) would still offer a useful measured
control/compute tradeoff versus fixed true H15, and how damaging it is to apply
labels learned under one terminal profile to the other.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_terminal_consistency_audit_v0"
STAMP = "20260929T0915Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
V0B_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair.py"
V0B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair_20260929T0845Z/completed.json"
FEATURE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h10_h15_feature_deployability_audit_v0_20260929T0850Z/completed.json"
STATE_CV_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_state_observable_h10_h15_model_cv_v0b_bounded_20260929T0905Z/completed.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-terminal-consistency-audit-v0-{STAMP}"
MIN_SAVE = 0.05
STRONG_SAVE = 0.10


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


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


def sf(value: Any, default: float = 0.0) -> float:
    try:
        y = float(value)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def quantile(values: Sequence[float], q: float, default: Optional[float] = None) -> Optional[float]:
    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not xs:
        return default
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def finite(values: Iterable[Any]) -> Dict[str, Any]:
    xs: List[float] = []
    for v in values:
        try:
            y = float(v)
        except Exception:
            continue
        if math.isfinite(y):
            xs.append(y)
    xs.sort()
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    return {
        "n": len(xs),
        "min": xs[0],
        "median": quantile(xs, 0.5),
        "mean": math.fsum(xs) / len(xs),
        "p95": quantile(xs, 0.95),
        "max": xs[-1],
        "sum": math.fsum(xs),
    }


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"prerequisite did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is True or obj.get("validation64_bank_opened") is True:
        raise ContractError(f"unexpected validation/test access flag in {rel(path)}")
    return obj


def load_v0b() -> Any:
    spec = importlib.util.spec_from_file_location("h10h15_v0b_schema_repair", str(V0B_SCRIPT))
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot import {rel(V0B_SCRIPT)}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def by_state(samples: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for row in samples:
        out.setdefault(str(row["state_key"]), []).append(row)
    return out


def profile_labels(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for r in rows:
        labels[str(r.get("terminal_profile") or "missing")] = int(r.get("label", 15))
    return labels


def label_for_profile(rows: Sequence[Mapping[str, Any]], profile: str, default: int = 15) -> int:
    labels = profile_labels(rows)
    return int(labels.get(profile, default))


def eval_named(v0b: Any, rows: Sequence[Mapping[str, Any]], preds: Sequence[int]) -> Dict[str, Any]:
    ev = v0b.eval_policy(list(rows), list(preds), require_nonconstant=True)
    ev["safe_compute_tradeoff_5pct"] = bool(ev.get("core_pass_5pct"))
    ev["safe_compute_tradeoff_10pct"] = bool(ev.get("core_pass_10pct"))
    return ev


def policy_predictions(rows: Sequence[Mapping[str, Any]], states: Mapping[str, Sequence[Mapping[str, Any]]], policy: str) -> List[int]:
    preds: List[int] = []
    for row in rows:
        sk = str(row["state_key"])
        state_rows = list(states.get(sk) or [row])
        labels = profile_labels(state_rows)
        vals = set(labels.values())
        if policy == "fixed_H15":
            preds.append(15)
        elif policy == "fixed_H10":
            preds.append(10)
        elif policy == "row_oracle":
            preds.append(int(row.get("label", 15)))
        elif policy == "matched_profile_lookup":
            preds.append(label_for_profile(state_rows, "matched_terminal", 15))
        elif policy == "shared_h15_profile_lookup":
            preds.append(label_for_profile(state_rows, "shared_h15_terminal", 15))
        elif policy == "opposite_profile_lookup":
            profile = str(row.get("terminal_profile") or "missing")
            other = "shared_h15_terminal" if profile == "matched_terminal" else "matched_terminal"
            preds.append(label_for_profile(state_rows, other, int(row.get("label", 15))))
        elif policy == "terminal_agreement_only":
            # Choose H10 only when every observed terminal profile for this state
            # says H10; otherwise fall back to H15. This is not deployable by
            # itself, but it tests whether terminal-consistent labels could be a
            # safer target for a later state-observable/value-calibrated method.
            preds.append(10 if vals and vals == {10} else 15)
        elif policy == "terminal_disagreement_to_H15_else_label":
            if len(vals) == 1:
                preds.append(int(next(iter(vals))))
            else:
                preds.append(15)
        elif policy == "any_H10_optimistic":
            preds.append(10 if 10 in vals else 15)
        else:
            raise ValueError(policy)
    return preds


def summarize_terminal_effects(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    states = by_state(rows)
    paired = []
    for sk, state_rows in sorted(states.items()):
        prof = {str(r.get("terminal_profile")): r for r in state_rows}
        if "matched_terminal" not in prof or "shared_h15_terminal" not in prof:
            continue
        m = prof["matched_terminal"]
        s = prof["shared_h15_terminal"]
        labels = {"matched_terminal": int(m["label"]), "shared_h15_terminal": int(s["label"])}
        h10_delta = sf(s["h10"]["physical"]) - sf(m["h10"]["physical"])
        h15_delta = sf(s["h15"]["physical"]) - sf(m["h15"]["physical"])
        matched_label = int(m["label"])
        shared_label = int(s["label"])
        # Physical regret if a label selected under one terminal profile is
        # applied to the other profile's measured outcomes.
        shared_chosen_by_matched = s["h10"] if matched_label == 10 else s["h15"]
        shared_oracle = s["h10"] if shared_label == 10 else s["h15"]
        matched_chosen_by_shared = m["h10"] if shared_label == 10 else m["h15"]
        matched_oracle = m["h10"] if matched_label == 10 else m["h15"]
        paired.append({
            "state_key": sk,
            "state_id": m.get("state_id"),
            "source": m.get("source"),
            "case": m.get("case"),
            "role_family": m.get("role_family"),
            "selection_group": m.get("selection_group"),
            "window": m.get("window"),
            "labels": labels,
            "label_flip": matched_label != shared_label,
            "stable_H10": matched_label == shared_label == 10,
            "stable_H15": matched_label == shared_label == 15,
            "h10_terminal_physical_delta_shared_minus_matched": h10_delta,
            "h15_terminal_physical_delta_shared_minus_matched": h15_delta,
            "abs_h10_terminal_physical_delta": abs(h10_delta),
            "abs_h15_terminal_physical_delta": abs(h15_delta),
            "matched_label_on_shared_physical_regret": sf(shared_chosen_by_matched["physical"]) - sf(shared_oracle["physical"]),
            "shared_label_on_matched_physical_regret": sf(matched_chosen_by_shared["physical"]) - sf(matched_oracle["physical"]),
            "h10_physical_matched": sf(m["h10"]["physical"]),
            "h10_physical_shared": sf(s["h10"]["physical"]),
            "h15_physical_matched": sf(m["h15"]["physical"]),
            "h15_physical_shared": sf(s["h15"]["physical"]),
            "h10_decision_matched": sf(m["h10"]["decision"]),
            "h10_decision_shared": sf(s["h10"]["decision"]),
            "h15_decision_matched": sf(m["h15"]["decision"]),
            "h15_decision_shared": sf(s["h15"]["decision"]),
        })
    flip_rows = [p for p in paired if p["label_flip"]]
    stable_rows = [p for p in paired if not p["label_flip"]]
    return {
        "paired_state_count": len(paired),
        "label_flip_count": len(flip_rows),
        "label_flip_rate": len(flip_rows) / float(len(paired) or 1),
        "stable_H10_count": sum(1 for p in paired if p["stable_H10"]),
        "stable_H15_count": sum(1 for p in paired if p["stable_H15"]),
        "h10_terminal_physical_delta_summary_all": finite(p["h10_terminal_physical_delta_shared_minus_matched"] for p in paired),
        "abs_h10_terminal_physical_delta_summary_flips": finite(p["abs_h10_terminal_physical_delta"] for p in flip_rows),
        "abs_h10_terminal_physical_delta_summary_stable": finite(p["abs_h10_terminal_physical_delta"] for p in stable_rows),
        "matched_label_on_shared_regret_summary": finite(p["matched_label_on_shared_physical_regret"] for p in paired),
        "shared_label_on_matched_regret_summary": finite(p["shared_label_on_matched_physical_regret"] for p in paired),
        "max_mismatch_regret": max([p["matched_label_on_shared_physical_regret"] for p in paired] + [p["shared_label_on_matched_physical_regret"] for p in paired] + [0.0]),
        "paired_states": paired,
    }


def analyze_subset(v0b: Any, name: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    states = by_state(rows)
    policies = [
        "fixed_H15",
        "fixed_H10",
        "row_oracle",
        "matched_profile_lookup",
        "shared_h15_profile_lookup",
        "opposite_profile_lookup",
        "terminal_agreement_only",
        "terminal_disagreement_to_H15_else_label",
        "any_H10_optimistic",
    ]
    evals = {p: eval_named(v0b, rows, policy_predictions(rows, states, p)) for p in policies}
    labels = {"10": sum(1 for r in rows if int(r.get("label", 15)) == 10), "15": sum(1 for r in rows if int(r.get("label", 15)) == 15)}
    return {
        "dataset": name,
        "n": len(rows),
        "state_count": len(states),
        "case_count": len({r.get("case") for r in rows}),
        "label_counts": labels,
        "terminal_effects": summarize_terminal_effects(rows),
        "policy_evaluations_vs_fixed_H15": evals,
    }


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
    v0b_done = completed_ok(V0B_DONE)
    feature_done = completed_ok(FEATURE_DONE)
    state_cv_done = completed_ok(STATE_CV_DONE)
    v0b = load_v0b()
    samples, _aux = v0b.load_samples()
    variants, flips = v0b.variants(samples)
    subsets = {
        "all_profiles": list(samples),
        "risk_anchor_source_all_profiles": [r for r in samples if r.get("source") == "risk_anchor"],
        "oracle_bank_all_profiles": [r for r in samples if r.get("source") == "oracle_bank"],
        "terminal_agreement_states_all_profiles": variants.get("terminal_agreement_states_all_profiles", []),
        "matched_terminal_only": variants.get("matched_terminal_only", []),
        "shared_h15_terminal_only": variants.get("shared_h15_terminal_only", []),
    }
    results = {name: analyze_subset(v0b, name, rows) for name, rows in subsets.items() if rows}
    key = results.get("all_profiles", {})
    risk = results.get("risk_anchor_source_all_profiles", {})
    def ev(dataset: Mapping[str, Any], policy: str) -> Mapping[str, Any]:
        return ((dataset.get("policy_evaluations_vs_fixed_H15") or {}).get(policy) or {})
    robust_all = ev(key, "terminal_agreement_only")
    robust_risk = ev(risk, "terminal_agreement_only")
    matched_all = ev(key, "matched_profile_lookup")
    shared_all = ev(key, "shared_h15_profile_lookup")
    opposite_all = ev(key, "opposite_profile_lookup")
    terminal_mismatch_evidence = {
        "all_label_flip_rate": ((key.get("terminal_effects") or {}).get("label_flip_rate")),
        "risk_label_flip_rate": ((risk.get("terminal_effects") or {}).get("label_flip_rate")),
        "opposite_profile_physical_delta_vs_H15": opposite_all.get("physical_delta_vs_H15"),
        "opposite_profile_tolerance_vs_H15": opposite_all.get("physical_tolerance_vs_H15"),
        "max_profile_mismatch_regret_all": ((key.get("terminal_effects") or {}).get("max_mismatch_regret")),
    }
    robust_target_has_value = bool(robust_all.get("core_pass_5pct") or robust_risk.get("core_pass_5pct"))
    mismatch_bad = bool(
        sf(terminal_mismatch_evidence.get("all_label_flip_rate")) >= 0.25
        or sf(terminal_mismatch_evidence.get("risk_label_flip_rate")) >= 0.25
        or sf(opposite_all.get("physical_delta_vs_H15")) > sf(opposite_all.get("physical_tolerance_vs_H15"), 1e9)
    )
    if robust_target_has_value and mismatch_bad:
        decision = "terminal-consistent labels retain some measured compute value, but cross-profile label transfer is unsafe; next intervention should calibrate terminal/objective targets before any learned selector rollout"
    elif robust_target_has_value:
        decision = "terminal-consistent labels may be a useful safe target; next run should test a small state-observable robust-label classifier/value calibration before rollout"
    elif mismatch_bad:
        decision = "terminal-profile mismatch dominates and robust labels do not preserve enough compute value; prioritize terminal-value/objective repair or scenario/modeling audit over selector refit"
    else:
        decision = "terminal consistency alone is not decisive; next diagnostic should inspect scenario/state coverage and value/objective scaling before retraining"
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_TERMINAL_CONSISTENCY_AUDIT_V0_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_offline_no_simulation_terminal_consistency_audit_not_validation_not_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "inputs": {"v0b_completed": rel(V0B_DONE), "feature_audit_completed": rel(FEATURE_DONE), "state_observable_cv_completed": rel(STATE_CV_DONE), "v0b_script": rel(V0B_SCRIPT)},
        "prerequisite_headlines": {"v0b": v0b_done.get("headline"), "feature_audit": feature_done.get("headline"), "state_observable_cv": state_cv_done.get("headline")},
        "terminal_label_flipped_states_from_v0b": len(flips),
        "terminal_mismatch_evidence": terminal_mismatch_evidence,
        "robust_target_has_value": robust_target_has_value,
        "mismatch_bad": mismatch_bad,
        "decision": decision,
        "results": results,
        "backup_request_after_diagnostic": rel(req),
    }
    write_json(OUT_DIR / "raw.json", raw)
    def fmt(v: Any) -> str:
        return f"{v:.6g}" if isinstance(v, float) else str(v)
    lines = [
        "# Vehicle true-variable-H terminal-consistency audit v0",
        "",
        f"UTC: `{created.isoformat()}`. Offline/no-simulation diagnostic; validation64 and sealed test remain closed.",
        "",
        "## Headline",
        "",
        f"- Terminal label-flipped states inherited from v0b: `{len(flips)}`.",
        f"- Terminal mismatch evidence: `{terminal_mismatch_evidence}`.",
        f"- Robust terminal-agreement target has value: `{robust_target_has_value}`.",
        f"- Mismatch bad: `{mismatch_bad}`.",
        f"- Decision: {decision}",
        "",
        "## Policy diagnostics versus fixed true H15",
        "",
        "`terminal_agreement_only` chooses H10 only when all observed terminal profiles for a state agree on H10; otherwise H15. `opposite_profile_lookup` quantifies applying labels learned under the other terminal profile.",
        "",
        "| dataset | policy | H counts | physΔ/tol | decision saving | solver saving | unsafe | pass5 | pass10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    show_policies = ["row_oracle", "terminal_agreement_only", "terminal_disagreement_to_H15_else_label", "matched_profile_lookup", "shared_h15_profile_lookup", "opposite_profile_lookup", "fixed_H10"]
    for dname in ("all_profiles", "risk_anchor_source_all_profiles", "oracle_bank_all_profiles", "terminal_agreement_states_all_profiles"):
        if dname not in results:
            continue
        pe = results[dname]["policy_evaluations_vs_fixed_H15"]
        for pol in show_policies:
            if pol not in pe:
                continue
            x = pe[pol]
            lines.append("| `%s` | `%s` | `%s` | %s/%s | %s | %s | %s | `%s` | `%s` |" % (
                dname,
                pol,
                x.get("horizon_counts"),
                fmt(x.get("physical_delta_vs_H15")),
                fmt(x.get("physical_tolerance_vs_H15")),
                fmt(x.get("decision_relative_saving_vs_H15")),
                fmt(x.get("solver_relative_saving_vs_H15")),
                fmt(x.get("unsafe_chosen_groups")),
                bool(x.get("core_pass_5pct")),
                bool(x.get("core_pass_10pct")),
            ))
    lines += [
        "",
        "## Interpretation rule frozen before execution",
        "",
        "If terminal-agreement labels keep useful measured decision-time savings without physical/safety loss, terminal/objective calibration becomes the next target before any rollout. If applying labels across terminal profiles creates large regret or robust labels lose all value, do not refit the supervised selector; instead repair terminal-value/objective/scenario design first.",
        "",
        f"Backup request after this diagnostic: `{rel(req)}`.",
    ]
    summary = OUT_DIR / "summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(req, {
        "requested_utc": created.isoformat(),
        "reason": "backup terminal-consistency audit before terminal-value/objective calibration, simulations, training or refit",
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
## 2026-09-29 vehicle true-variable-H terminal-consistency audit v0

UTC: {created.isoformat()}. Offline/no-simulation audit of terminal-profile consistency after the state-observable H10/H15 nested-CV result; validation64 and sealed test stayed closed. Robust target has value={robust_target_has_value}; mismatch bad={mismatch_bad}. Decision: {decision}. Artifacts: `{rel(summary)}`, `{rel(OUT_DIR / 'raw.json')}`.
""")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), V0B_SCRIPT, V0B_DONE, FEATURE_DONE, STATE_CV_DONE, STATE_PATH, req]
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
        "headline": {"terminal_mismatch_evidence": terminal_mismatch_evidence, "robust_target_has_value": robust_target_has_value, "mismatch_bad": mismatch_bad, "decision": decision},
        "backup_request": rel(req),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(done_path, done)
    print(json.dumps({"completed": rel(done_path), "summary": rel(summary), "headline": done["headline"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
