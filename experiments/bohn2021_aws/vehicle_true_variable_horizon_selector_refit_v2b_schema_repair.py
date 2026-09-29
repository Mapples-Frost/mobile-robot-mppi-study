#!/usr/bin/env python3
"""Schema/syntax-repaired development-only selector-refit diagnostic.

This is a replacement for the failed v2 source
`vehicle_true_variable_horizon_selector_refit_v2_diagnostic.py` whose first
execution failed at import time with a SyntaxError before any science or output.

No simulations, no validation64 access and no sealed-test access are performed.
The script uses already-opened development banks (fresh_v0 and fresh_v1) plus
oracle/risk source banks to test whether a simple cost-aware nearest-neighbour
selector calibration generalizes across independent development banks.  Any
positive result remains development evidence and requires another frozen unused
fresh-source confirmation before validation/test.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_guard_veto_diagnostic_v0 as gv  # noqa:E402

NAME = "vehicle_true_variable_horizon_selector_refit_v2b_schema_repair"
STAMP = "20260929T1135Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
PRIMARY_PROFILE = "shared_h15_terminal"
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
MARKER = f"vehicle-true-variable-H-selector-refit-v2b-schema-repair-{STAMP}"

V0_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z"
V1_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z"
BANKS = {
    "fresh_v0": {"raw": V0_DIR / "raw.json", "completed": V0_DIR / "completed.json", "manifest": V0_DIR / "selected_state_manifest.json"},
    "fresh_v1": {"raw": V1_DIR / "raw.json", "completed": V1_DIR / "completed.json", "manifest": V1_DIR / "selected_state_manifest.json"},
}
SOURCE_INPUTS = {
    "oracle": {"raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/raw.json", "completed": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/completed.json"},
    "risk": {"raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/raw.json", "completed": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/completed.json"},
}

OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_FILE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
CONTINUE_STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1135_after_selector_refit_v2b_schema_repair.md"
PROTOCOL_OUT = ROOT / f"research_artifacts/aws_protocols/{NAME}_{STAMP}_candidate_protocol.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"


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
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sf(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def si(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing prerequisite: " + rel(path))
    obj = read_json(path)
    if obj.get("sealed_test_accessed") is not False or obj.get("validation64_bank_opened") is not False:
        raise ContractError("diagnostic prerequisite has validation/test flags set: " + rel(path))
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError("diagnostic prerequisite did not complete cleanly: " + rel(path))
    return obj


def median_summary(medians: Mapping[str, Any], horizon: int) -> Mapping[str, Any]:
    return medians.get(str(horizon)) or medians.get(horizon) or {}


def load_manifest(raw: Mapping[str, Any], fallback_path: Path) -> List[Mapping[str, Any]]:
    manifest = raw.get("selected_state_manifest")
    if isinstance(manifest, Mapping) and isinstance(manifest.get("selected_states"), list):
        return list(manifest.get("selected_states") or [])
    if isinstance(manifest, list):
        return list(manifest)
    if fallback_path.exists():
        fb = read_json(fallback_path)
        if isinstance(fb, Mapping):
            return list(fb.get("selected_states") or [])
        if isinstance(fb, list):
            return list(fb)
    raise ContractError("cannot locate selected_state_manifest for " + rel(fallback_path))


def build_dev_rows(bank_id: str, spec: Mapping[str, Path]) -> List[Dict[str, Any]]:
    raw = read_json(spec["raw"])
    if raw.get("sealed_test_accessed") is not False or raw.get("validation64_bank_opened") is not False:
        raise ContractError(f"{bank_id} raw flags show validation/test access")
    states = load_manifest(raw, spec["manifest"])
    by_base = {str(s.get("base_state_id")): s for s in states}
    rows: List[Dict[str, Any]] = []
    for r in ((raw.get("analysis") or {}).get("state_profile_rows") or []):
        base = str(r.get("base_state_id") or r.get("state_id"))
        st = by_base.get(base)
        if st is None:
            raise ContractError(f"{bank_id}: missing manifest state for {base}")
        med = r.get("median_by_h") or {}
        h10 = median_summary(med, 10)
        h15 = median_summary(med, 15)
        h10_phys = sf(h10.get("physical"), 0.0)
        h15_phys = sf(h15.get("physical"), 0.0)
        h10_dec = sf(h10.get("decision_sum_s"), 0.0)
        h15_dec = sf(h15.get("decision_sum_s"), 0.0)
        h10_sol = sf(h10.get("solver_sum_s"), 0.0)
        h15_sol = sf(h15.get("solver_sum_s"), 0.0)
        row_tol = sf(r.get("row_physical_tolerance_vs_H15"), max(2.0, 0.05 * abs(h15_phys)))
        row_tol = max(2.0, row_tol)
        h10_safe = bool(h10.get("safe_all"))
        h15_safe = bool(h15.get("safe_all"))
        if "h10_beneficial_vs_h15" in r:
            beneficial = bool(r.get("h10_beneficial_vs_h15"))
        else:
            beneficial = bool(h15_safe and h10_safe and (h10_phys - h15_phys <= row_tol) and (h10_dec < h15_dec))
        catastrophic = bool((not h10_safe) or (h15_safe and (h10_phys - h15_phys > row_tol)))
        rows.append({
            "bank_id": bank_id,
            "base_state_id": base,
            "terminal_profile": str(r.get("terminal_profile")),
            "fresh_confirmation_group": r.get("fresh_confirmation_group"),
            "source_candidate_index": st.get("source_candidate_index"),
            "branch_state_slot": si(r.get("branch_state_slot", st.get("branch_state_slot")), -1),
            "feature": gv.feature_from_observation_state(st.get("initial_observation_from_h15_trace"), st.get("branch_previous_state")),
            "stage_a_trace_risk_score": sf(st.get("stage_a_trace_risk_score"), 0.0),
            "h10_beneficial_vs_h15": beneficial,
            "oracle_label": r.get("oracle_label"),
            "h10_physical": h10_phys,
            "h15_physical": h15_phys,
            "row_physical_tolerance": row_tol,
            "h10_decision_sum_s": h10_dec,
            "h15_decision_sum_s": h15_dec,
            "h10_solver_sum_s": h10_sol,
            "h15_solver_sum_s": h15_sol,
            "h10_safe_all": h10_safe,
            "h15_safe_all": h15_safe,
            "phys_delta_h10_minus_h15": h10_phys - h15_phys,
            "decision_delta_h10_minus_h15": h10_dec - h15_dec,
            "solver_delta_h10_minus_h15": h10_sol - h15_sol,
            "h10_catastrophic_vs_h15": catastrophic,
        })
    return rows


def category_from_row(row: Mapping[str, Any]) -> str:
    if bool(row.get("h10_beneficial_vs_h15")):
        return "agreement_positive_h10"
    if bool(row.get("h10_catastrophic_vs_h15")):
        return "terminal_disagreement_or_missing"
    return "agreement_non_h10"


def dev_examples(rows: Sequence[Mapping[str, Any]], label_mode: str) -> List[Dict[str, Any]]:
    by_base: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_base.setdefault(str(row.get("base_state_id")), {})[str(row.get("terminal_profile"))] = row
    out: List[Dict[str, Any]] = []
    for base, profs in sorted(by_base.items()):
        primary = profs.get(PRIMARY_PROFILE)
        matched = profs.get("matched_terminal")
        if primary is None:
            continue
        if label_mode == "primary_only":
            cat = category_from_row(primary)
        elif label_mode == "both_profiles_agree":
            if matched is None:
                cat = category_from_row(primary)
            elif bool(primary.get("h10_beneficial_vs_h15")) and bool(matched.get("h10_beneficial_vs_h15")):
                cat = "agreement_positive_h10"
            elif bool(primary.get("h10_beneficial_vs_h15")) != bool(matched.get("h10_beneficial_vs_h15")):
                cat = "terminal_disagreement_or_missing"
            elif bool(primary.get("h10_catastrophic_vs_h15")) or bool(matched.get("h10_catastrophic_vs_h15")):
                cat = "terminal_disagreement_or_missing"
            else:
                cat = "agreement_non_h10"
        elif label_mode == "primary_plus_matched_veto":
            if bool(primary.get("h10_catastrophic_vs_h15")) or (matched is not None and bool(matched.get("h10_catastrophic_vs_h15"))):
                cat = "terminal_disagreement_or_missing"
            elif bool(primary.get("h10_beneficial_vs_h15")):
                cat = "agreement_positive_h10"
            else:
                cat = "agreement_non_h10"
        elif label_mode == "strict_noncat_primary_positive":
            if bool(primary.get("h10_beneficial_vs_h15")) and not bool(primary.get("h10_catastrophic_vs_h15")) and not (matched is not None and bool(matched.get("h10_catastrophic_vs_h15"))):
                cat = "agreement_positive_h10"
            elif bool(primary.get("h10_catastrophic_vs_h15")) or (matched is not None and bool(matched.get("h10_catastrophic_vs_h15"))):
                cat = "terminal_disagreement_or_missing"
            else:
                cat = "agreement_non_h10"
        else:
            raise ContractError("unknown label_mode " + label_mode)
        out.append({
            "source": "development_" + str(primary.get("bank_id")),
            "state_id": base,
            "labels_by_profile": {p: (None if profs[p].get("oracle_label") is None else int(profs[p].get("oracle_label"))) for p in profs},
            "category": cat,
            "binary_h10": 1 if cat == "agreement_positive_h10" else 0,
            "feature": list(primary.get("feature") or []),
            "fresh_confirmation_group": primary.get("fresh_confirmation_group"),
            "label_mode": label_mode,
        })
    return out


def candidate_specs() -> List[Dict[str, Any]]:
    pools = [
        ("agreement_only", "disagreement_only"),
        ("agreement_only", "all_nonpositive_including_disagreement"),
        ("all_nonpositive_including_disagreement", "none"),
    ]
    specs: List[Dict[str, Any]] = []
    for mode in ["raw_abs_l2", "obstacle_slice_l2", "pose_goal_obs_l2"]:
        for positive_radius_scale in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
            for negative_margin in [1.0, 1.25, 1.5, 1.75, 2.0]:
                for veto_margin in [0.75, 1.0, 1.25, 1.5]:
                    for negative_pool, veto_pool in pools:
                        specs.append({
                            "variant_id": f"v2b_{mode}_prs{positive_radius_scale:g}_nm{negative_margin:g}_vm{veto_margin:g}_{negative_pool}_{veto_pool}",
                            "mode": mode,
                            "min_positive_support": 1,
                            "positive_radius_quantile": 0.5,
                            "positive_radius_scale": positive_radius_scale,
                            "negative_margin": negative_margin,
                            "veto_margin": veto_margin,
                            "negative_pool": negative_pool,
                            "veto_pool": veto_pool,
                        })
    return specs


def summarize_gate(gate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "profile": gate.get("profile"),
        "groups": gate.get("groups"),
        "chosen_counts": gate.get("chosen_counts"),
        "confusion": gate.get("confusion"),
        "physical_delta_vs_fixed_H15": gate.get("physical_delta_vs_fixed_H15"),
        "physical_tolerance_sum": gate.get("physical_tolerance_sum"),
        "physical_gate": gate.get("physical_gate"),
        "decision_relative_saving_vs_fixed_H15": gate.get("decision_relative_saving_vs_fixed_H15"),
        "solver_relative_saving_vs_fixed_H15": gate.get("solver_relative_saving_vs_fixed_H15"),
        "pass_5pct_no_cat_fp": gate.get("pass_5pct_no_cat_fp"),
        "pass_10pct_no_cat_fp": gate.get("pass_10pct_no_cat_fp"),
        "no_catastrophic_fp": gate.get("no_catastrophic_fp"),
        "false_positive_count": len(gate.get("false_positive_rows") or []),
        "catastrophic_false_positive_count": len(gate.get("catastrophic_false_positive_rows") or []),
        "unsafe_count": len(gate.get("unsafe_rows") or []),
        "false_positive_rows": gate.get("false_positive_rows") or [],
        "catastrophic_false_positive_rows": gate.get("catastrophic_false_positive_rows") or [],
        "unsafe_rows": gate.get("unsafe_rows") or [],
    }


def evaluate_model(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "primary_shared_h15": summarize_gate(gv.evaluate(model, rows, PRIMARY_PROFILE)),
        "matched_terminal": summarize_gate(gv.evaluate(model, rows, "matched_terminal")),
    }


def score_pair(primary_a: Mapping[str, Any], primary_b: Mapping[str, Any], matched_a: Mapping[str, Any], matched_b: Mapping[str, Any]) -> Tuple[Any, ...]:
    shared_strong = bool(primary_a.get("pass_10pct_no_cat_fp")) and bool(primary_b.get("pass_10pct_no_cat_fp"))
    shared_weak = bool(primary_a.get("pass_5pct_no_cat_fp")) and bool(primary_b.get("pass_5pct_no_cat_fp"))
    shared_bad = si(primary_a.get("catastrophic_false_positive_count")) + si(primary_b.get("catastrophic_false_positive_count")) + si(primary_a.get("unsafe_count")) + si(primary_b.get("unsafe_count"))
    matched_bad = si(matched_a.get("catastrophic_false_positive_count")) + si(matched_b.get("catastrophic_false_positive_count")) + si(matched_a.get("unsafe_count")) + si(matched_b.get("unsafe_count"))
    min_decision = min(sf(primary_a.get("decision_relative_saving_vs_fixed_H15"), -999.0), sf(primary_b.get("decision_relative_saving_vs_fixed_H15"), -999.0))
    avg_decision = 0.5 * (sf(primary_a.get("decision_relative_saving_vs_fixed_H15"), 0.0) + sf(primary_b.get("decision_relative_saving_vs_fixed_H15"), 0.0))
    h10_total = si((primary_a.get("chosen_counts") or {}).get("10")) + si((primary_b.get("chosen_counts") or {}).get("10"))
    phys_max = max(sf(primary_a.get("physical_delta_vs_fixed_H15"), 0.0), sf(primary_b.get("physical_delta_vs_fixed_H15"), 0.0))
    return (shared_strong, shared_weak, -shared_bad, -matched_bad, min_decision, avg_decision, h10_total, -phys_max)


def aggregate_crossbank(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for row in results:
        key = str(row["model_key"])
        rec = by_key.setdefault(key, {"model_key": key, "spec": row["spec"], "label_mode": row["label_mode"], "holdouts": {}})
        rec["holdouts"][str(row["test_bank"])] = row
    ranked: List[Dict[str, Any]] = []
    for key, obj in by_key.items():
        holds = obj["holdouts"]
        if "fresh_v0" not in holds or "fresh_v1" not in holds:
            continue
        a = holds["fresh_v0"]["evaluation"]["primary_shared_h15"]
        b = holds["fresh_v1"]["evaluation"]["primary_shared_h15"]
        ma = holds["fresh_v0"]["evaluation"]["matched_terminal"]
        mb = holds["fresh_v1"]["evaluation"]["matched_terminal"]
        score = score_pair(a, b, ma, mb)
        ranked.append({
            "model_key": key,
            "spec": obj["spec"],
            "label_mode": obj["label_mode"],
            "crossbank_shared_strong_pass": bool(score[0]),
            "crossbank_shared_weak_pass": bool(score[1]),
            "primary_bad_count_total": -si(score[2]),
            "matched_bad_count_total": -si(score[3]),
            "min_primary_decision_saving": score[4],
            "avg_primary_decision_saving": score[5],
            "primary_h10_predictions_total": score[6],
            "max_primary_physical_delta": -float(score[7]),
            "holdouts": {"fresh_v0": holds["fresh_v0"]["evaluation"], "fresh_v1": holds["fresh_v1"]["evaluation"]},
            "sort_score": score,
        })
    return sorted(ranked, key=lambda x: x["sort_score"], reverse=True)


def source_summary(examples: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {"total": len(examples)}
    for ex in examples:
        cat = str(ex.get("category"))
        out[cat] = out.get(cat, 0) + 1
    return out


def bank_label_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for profile in TERMINAL_PROFILES:
        rr = [r for r in rows if r.get("terminal_profile") == profile]
        h15_dec_sum = sum(sf(r.get("h15_decision_sum_s"), 0.0) for r in rr)
        out[profile] = {
            "groups": len(rr),
            "h10_beneficial": sum(1 for r in rr if r.get("h10_beneficial_vs_h15")),
            "h10_catastrophic": sum(1 for r in rr if r.get("h10_catastrophic_vs_h15")),
            "mean_decision_saving_if_all_h10": (sum(sf(r.get("h15_decision_sum_s"), 0.0) - sf(r.get("h10_decision_sum_s"), 0.0) for r in rr) / h15_dec_sum) if h15_dec_sum > 0 else None,
            "physical_delta_if_all_h10": sum(sf(r.get("h10_physical"), 0.0) - sf(r.get("h15_physical"), 0.0) for r in rr),
        }
    return out


def write_summary(raw: Mapping[str, Any]) -> None:
    best = raw["crossbank_ranked"][0] if raw["crossbank_ranked"] else None
    lines: List[str] = [
        "# Vehicle true-variable-H selector refit v2b schema repair diagnostic",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata/refit enumeration only: no simulations, no validation64, no sealed test.",
        "",
        "## Why this was run",
        "Fresh-source v1 was safe and nonconstant but weak: 5/16 H10 predictions, no shared-H15 false positives, physical delta +0.228 within tolerance, measured decision saving 8.21% (<10% strong gate). Oracle opportunity remained large (14/16 H10, 24.46% decision saving). This repaired diagnostic tests whether a small cost-aware selector refit can recover recall across independent development banks.",
        "",
        "## Development label summaries",
        "",
        f"- Source examples: `{raw['source_summary']}`.",
    ]
    for bank_id, summary in raw["bank_label_summary"].items():
        lines.append(f"- `{bank_id}`: `{summary}`")
    lines += ["", "## Top cross-bank selector candidates", "", "| rank | model_key | strong | weak | primary bad | matched bad | min dec save | avg dec save | H10 total | max phys Δ |", "|---:|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for i, row in enumerate(raw["crossbank_ranked"][:15], 1):
        lines.append("| %d | `%s` | `%s` | `%s` | %d | %d | %.3f | %.3f | %d | %.4g |" % (
            i,
            row["model_key"],
            row["crossbank_shared_strong_pass"],
            row["crossbank_shared_weak_pass"],
            int(row["primary_bad_count_total"]),
            int(row["matched_bad_count_total"]),
            sf(row["min_primary_decision_saving"], 0.0),
            sf(row["avg_primary_decision_saving"], 0.0),
            int(row["primary_h10_predictions_total"]),
            sf(row["max_primary_physical_delta"], 0.0),
        ))
    if best:
        lines += [
            "",
            "## Best candidate details",
            "",
            f"Best model key: `{best['model_key']}`.",
            f"Spec: `{best['spec']}`.",
            f"Fresh_v0 holdout primary: `{best['holdouts']['fresh_v0']['primary_shared_h15']}`.",
            f"Fresh_v1 holdout primary: `{best['holdouts']['fresh_v1']['primary_shared_h15']}`.",
            f"Matched-terminal robustness: v0 `{best['holdouts']['fresh_v0']['matched_terminal']}`, v1 `{best['holdouts']['fresh_v1']['matched_terminal']}`.",
        ]
    lines += ["", "## Decision", "", str(raw["decision"]), "", f"Candidate protocol: `{rel(PROTOCOL_OUT)}`.", f"Raw: `{rel(OUT_DIR / 'raw.json')}`; completed: `{rel(OUT_DIR / 'completed.json')}`."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    if reg.exists():
        old = reg.read_text(encoding="utf-8", errors="replace")
        if MARKER not in old[-50000:]:
            reg.write_text(old.rstrip() + f"\n{now_utc().isoformat()},{NAME},development_selector_refit_v2b_schema_repair,metadata_refit_no_validation_no_test,0,0,0,0,0,False,{rel(OUT_DIR / 'completed.json')}\n", encoding="utf-8")


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--backup-verified-commit", type=str, required=True)
    parser.add_argument("--i-accept-development-selector-refit-v2b-schema-repair", action="store_true")
    args = parser.parse_args(argv)
    if not args.run or not args.i_accept_development_selector_refit_v2b_schema_repair:
        raise ContractError("requires --run and explicit selector-refit v2b schema-repair acknowledgement")
    if (OUT_DIR / "completed.json").exists():
        done = read_json(OUT_DIR / "completed.json")
        print(json.dumps({"already_completed": rel(OUT_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in BANKS.values():
        completed_ok(spec["completed"])
    for spec in SOURCE_INPUTS.values():
        completed_ok(spec["completed"])

    created_start = now_utc()
    source_examples = gv.load_source_examples(SOURCE_INPUTS["oracle"]["raw"], "oracle_bank") + gv.load_source_examples(SOURCE_INPUTS["risk"]["raw"], "risk_anchor")
    bank_rows = {bank_id: build_dev_rows(bank_id, spec) for bank_id, spec in BANKS.items()}
    specs = candidate_specs()
    label_modes = ["source_only", "primary_only", "both_profiles_agree", "primary_plus_matched_veto", "strict_noncat_primary_positive"]

    all_results: List[Dict[str, Any]] = []
    for train_bank, test_bank in [("fresh_v0", "fresh_v1"), ("fresh_v1", "fresh_v0")]:
        for label_mode in label_modes:
            if label_mode == "source_only":
                train_examples = list(source_examples)
                train_source = "oracle+risk_source_only"
            else:
                train_examples = list(source_examples) + dev_examples(bank_rows[train_bank], label_mode)
                train_source = "oracle+risk+" + train_bank
            for spec in specs:
                model = gv.fit_predictor(train_examples, spec)
                ev = evaluate_model(model, bank_rows[test_bank])
                all_results.append({
                    "model_key": label_mode + "::" + spec["variant_id"],
                    "label_mode": label_mode,
                    "train_source": train_source,
                    "train_bank": train_bank,
                    "test_bank": test_bank,
                    "spec": spec,
                    "model_counts": model.get("counts"),
                    "model_radius": model.get("radius"),
                    "evaluation": ev,
                })

    ranked = aggregate_crossbank(all_results)
    best = ranked[0] if ranked else None
    if best is None:
        decision = "No selector-refit candidates were produced; inspect parser/source assumptions before any simulation."
    elif best["crossbank_shared_strong_pass"] and best["matched_bad_count_total"] == 0:
        decision = "A simple refit has cross-bank strong shared-H15 performance and no matched-terminal catastrophic rows; freeze an independent v2 fresh-source confirmation before validation64."
    elif best["crossbank_shared_strong_pass"]:
        decision = "A simple refit reaches cross-bank strong shared-H15 performance but has matched-terminal robustness concerns; either constrain claims/profile to shared-H15 terminal or prioritize terminal-value calibration before validation."
    elif best["crossbank_shared_weak_pass"]:
        decision = "Best simple refit is still weak across banks; do not validate yet. Prefer a targeted value/objective/representation training or terminal-value refit ablation."
    else:
        decision = "Simple selector refit does not cross-bank generalize; shift from selector thresholding to value/objective/representation/training ablation."

    protocol = {
        "protocol_id": NAME + "_candidate_protocol_" + STAMP,
        "created_utc": now_utc().isoformat(),
        "classification": "development_IMPROVED_selector_refit_candidate_not_validation_not_test",
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "best_candidate": None if best is None else {k: best[k] for k in ("model_key", "spec", "label_mode", "crossbank_shared_strong_pass", "crossbank_shared_weak_pass", "primary_bad_count_total", "matched_bad_count_total", "min_primary_decision_saving", "avg_primary_decision_saving", "primary_h10_predictions_total", "max_primary_physical_delta")},
        "use_restriction": "Not permission to open validation/test. A promising result still requires a separate fresh-source confirmation freeze on unused cases before validation64.",
    }
    write_json(PROTOCOL_OUT, protocol)
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_SELECTOR_REFIT_V2B_SCHEMA_REPAIR_%s.json" % now_utc().isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": now_utc().isoformat(), "reason": "backup selector-refit v2b diagnostic outputs and candidate protocol before any new simulations/validation/training", "backup_required_before_more_simulations": True, "backup_required_before_training_or_refit": True, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(OUT_DIR), rel(PROTOCOL_OUT), rel(STATE_FILE), rel(CONTINUE_STATE), rel(req)]})

    raw = {
        "created_utc": now_utc().isoformat(),
        "started_utc": created_start.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (now_utc() - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_metadata_selector_refit_no_simulation_no_validation_no_test",
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "candidate_selector_refits_enumerated": len(all_results),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "source_summary": source_summary(source_examples),
        "bank_label_summary": {bank_id: bank_label_summary(rows) for bank_id, rows in bank_rows.items()},
        "candidate_spec_count": len(specs),
        "label_modes": label_modes,
        "all_result_count": len(all_results),
        "crossbank_ranked": ranked[:200],
        "decision": decision,
        "candidate_protocol": rel(PROTOCOL_OUT),
        "backup_request_after_run": rel(req),
    }
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_FILE.write_text((OUT_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    CONTINUE_STATE.write_text(f"""# Continue state after selector-refit v2b schema-repair diagnostic

UTC: {raw['created_utc']}
Elapsed since first supervisor event: {raw['elapsed_since_first_supervisor_event_seconds']/3600.0:.2f} h.

Completed `{NAME}` with 0 simulations, 0 control steps, 0 gradient steps, no validation64 and no sealed test. This repaired the syntax failure in the v2 source while preserving that failed run as evidence.

Decision: {decision}
Best candidate: {None if best is None else best['model_key']}
Summary: `{rel(OUT_DIR / 'summary.md')}`
Raw: `{rel(OUT_DIR / 'raw.json')}`
Completed: `{rel(OUT_DIR / 'completed.json')}`
Candidate protocol: `{rel(PROTOCOL_OUT)}`
Backup request: `{rel(req)}`

Next action: obtain external backup for these outputs. If the best candidate is strong and robust, freeze an independent unused fresh-source confirmation. Otherwise move to a targeted value/objective/representation or terminal-value training/refit ablation before validation64.
""", encoding="utf-8")
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H selector-refit v2b schema-repair diagnostic

UTC: {raw['created_utc']}. Metadata/refit-only diagnostic completed with 0 simulations/control steps, no validation64, no sealed test, no gradient training. It cross-trained simple selector variants on fresh_v0/fresh_v1 development banks and evaluated on the opposite bank. Decision: {decision}. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`, candidate protocol `{rel(PROTOCOL_OUT)}`.
""")

    files = [SOURCE_INPUTS["oracle"]["raw"], SOURCE_INPUTS["risk"]["raw"], BANKS["fresh_v0"]["raw"], BANKS["fresh_v1"]["raw"], Path(__file__).resolve(), OUT_DIR / "raw.json", OUT_DIR / "summary.md", STATE_FILE, CONTINUE_STATE, PROTOCOL_OUT, req]
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": raw["created_utc"],
        "elapsed_since_first_supervisor_event_seconds": raw["elapsed_since_first_supervisor_event_seconds"],
        "classification": raw["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "candidate_selector_refits_enumerated": len(all_results),
        "headline": None if best is None else {k: best[k] for k in ("model_key", "crossbank_shared_strong_pass", "crossbank_shared_weak_pass", "primary_bad_count_total", "matched_bad_count_total", "min_primary_decision_saving", "avg_primary_decision_saving", "primary_h10_predictions_total")},
        "decision": decision,
        "backup_request": rel(req),
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    }
    write_json(OUT_DIR / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT_DIR / "completed.json"), "summary": rel(OUT_DIR / "summary.md"), "headline": completed["headline"], "decision": decision, "new_simulations": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
