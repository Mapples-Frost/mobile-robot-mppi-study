#!/usr/bin/env python3
"""Vehicle latency-tree training/selection collapse diagnostic v1.

Metadata-only diagnostic for why learned_s0 and learned_s1 collapsed to H25
and why learned_s2 is the only switching current candidate.  Reads existing
training/selection metadata (fit.json, policy.json, completed.json,
selection_registration.json, generation selection.json files, and selection
completed markers).  It does not reopen validation banks, does not read sealed
test artifacts, and runs no rollout/control/training.

This is development/diagnostic evidence for an IMPROVED method revision; it is
not final-test evidence and not an ORIGINAL SAC reproduction result.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/train"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_training_selection_collapse_diagnostic_20260927_v1"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_training_selection_collapse_diagnostic_v1.py"
DOC_MARKER = "vehicle-training-selection-collapse-diagnostic-v1-20260927"
SEEDS = [0, 1, 2]
H25 = 25


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def values_summary(values: Iterable[Any]) -> Dict[str, Any]:
    xs = sorted(v for v in (as_float(x) for x in values) if v is not None)
    if not xs:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    n = len(xs)
    med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    return {"count": n, "mean": float(math.fsum(xs) / n), "median": float(med), "min": float(xs[0]), "max": float(xs[-1])}


def classify_policy(policy: Mapping[str, Any]) -> Dict[str, Any]:
    kind = policy.get("kind")
    if kind == "constant":
        h = int(policy.get("horizon"))
        leaves = [h]
    else:
        leaves = [int(x) for x in (policy.get("leaves") or [])]
    unique = sorted(set(leaves))
    return {
        "kind": kind,
        "leaves": leaves,
        "unique_horizons": unique,
        "is_structurally_constant": len(unique) == 1,
        "is_constant_H25": unique == [H25],
        "is_adaptive": len(unique) > 1,
        "has_shorter_than_H25": any(h < H25 for h in unique),
        "has_longer_than_H25": any(h > H25 for h in unique),
    }


def generation_from_id(cid: str) -> Optional[int]:
    m = re.match(r"g(\d+)_c\d+", cid or "")
    return int(m.group(1)) if m else None


def candidate_record(seed: int, c: Mapping[str, Any]) -> Dict[str, Any]:
    rank = c.get("rank") or {}
    policy_class = classify_policy(c.get("policy") or {})
    obj = as_float(rank.get("objective"))
    cost = as_float(rank.get("cost_change"))
    phys = as_float(rank.get("physical_change"))
    tr = as_float(rank.get("time_ratio"))
    objective_formula_residual = None
    if obj is not None and cost is not None and tr is not None:
        objective_formula_residual = obj - (cost + 0.5 * (tr - 1.0))
    rec: Dict[str, Any] = {
        "seed": seed,
        "id": str(c.get("id")),
        "generation": generation_from_id(str(c.get("id"))),
        "eligible": bool(rank.get("eligible")),
        "violations": int(rank.get("violations", 0) or 0),
        "objective": obj,
        "cost_change": cost,
        "physical_change": phys,
        "time_ratio": tr,
        "objective_minus_cost_plus_half_time_residual": objective_formula_residual,
        "folder": str(c.get("folder")),
    }
    rec.update(policy_class)
    return rec


def rank_key(rec: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (0 if rec.get("eligible") else 1,
            int(rec.get("violations") or 0),
            rec.get("objective") if rec.get("objective") is not None else float("inf"),
            rec.get("cost_change") if rec.get("cost_change") is not None else float("inf"),
            rec.get("time_ratio") if rec.get("time_ratio") is not None else float("inf"),
            str(rec.get("id")))


def best_of(records: Sequence[Mapping[str, Any]], predicate) -> Optional[Dict[str, Any]]:
    xs = [r for r in records if predicate(r)]
    if not xs:
        return None
    return dict(sorted(xs, key=rank_key)[0])


def count_by(records: Sequence[Mapping[str, Any]], predicate) -> int:
    return sum(1 for r in records if predicate(r))


def summarize_generation(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for gen in sorted({r.get("generation") for r in records if r.get("generation") is not None}):
        rs = [r for r in records if r.get("generation") == gen]
        out[str(gen)] = {
            "candidates": len(rs),
            "eligible": count_by(rs, lambda r: r.get("eligible")),
            "adaptive": count_by(rs, lambda r: r.get("is_adaptive")),
            "eligible_adaptive": count_by(rs, lambda r: r.get("eligible") and r.get("is_adaptive")),
            "constant_H25": count_by(rs, lambda r: r.get("is_constant_H25")),
            "eligible_constant_H25": count_by(rs, lambda r: r.get("eligible") and r.get("is_constant_H25")),
            "best_eligible_overall": best_of(rs, lambda r: r.get("eligible")),
            "best_eligible_adaptive": best_of(rs, lambda r: r.get("eligible") and r.get("is_adaptive")),
            "best_eligible_constant_H25": best_of(rs, lambda r: r.get("eligible") and r.get("is_constant_H25")),
        }
    return out


def small_json_summary(path: Path) -> Dict[str, Any]:
    """Summarize a small JSON without assuming schema; avoid heavy trace/raw reads."""
    try:
        obj = read_json(path)
    except Exception as e:
        return {"path": rel(path), "sha256": sha256(path) if path.exists() else None, "read_error": str(e)}
    summary: Dict[str, Any] = {"path": rel(path), "sha256": sha256(path)}
    if isinstance(obj, Mapping):
        summary["top_level_keys"] = sorted(obj.keys())
        for k in ["passed", "selected", "winner", "rollout_key", "success", "steps", "total_cost", "physical_constraint_cost", "objective", "cost_change", "time_ratio", "rank"]:
            if k in obj:
                summary[k] = obj[k]
        # Compactly capture obvious scalar metrics nested one level deep.
        scalar_hits: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                if k not in summary:
                    scalar_hits[k] = v
            elif isinstance(v, Mapping):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float, str, bool)) or vv is None:
                        name = f"{k}.{kk}"
                        if any(tok in name.lower() for tok in ["objective", "cost", "time", "success", "failure", "eligible", "violation", "selected", "winner", "mean", "sum"]):
                            scalar_hits[name] = vv
        if scalar_hits:
            summary["scalar_metric_hints"] = scalar_hits
    else:
        summary["json_type"] = type(obj).__name__
    return summary


def inspect_selection_metadata(sdir: Path) -> Dict[str, Any]:
    selection_dir = sdir / "selection"
    result: Dict[str, Any] = {
        "selection_registration": small_json_summary(sdir / "selection_registration.json") if (sdir / "selection_registration.json").exists() else None,
        "selection_completed_markers": [],
        "generation_selection_json": [],
    }
    if selection_dir.exists():
        for p in sorted(selection_dir.glob("*/completed.json")):
            name = p.parent.name
            m = re.match(r"r(\d+)_(.+)", name)
            rec = small_json_summary(p)
            rec["selection_dir_name"] = name
            rec["repeat"] = int(m.group(1)) if m else None
            rec["candidate_id_from_dir"] = m.group(2) if m else name
            result["selection_completed_markers"].append(rec)
    for p in sorted(sdir.glob("generation*/selection.json")):
        rec = small_json_summary(p)
        rec["generation_dir"] = p.parent.name
        result["generation_selection_json"].append(rec)
    return result


def diagnose_seed(seed: int) -> Dict[str, Any]:
    sdir = TRAIN_ROOT / f"vehicle_s{seed}"
    fit_path = sdir / "fit.json"
    completed_path = sdir / "completed.json"
    policy_path = sdir / "policy.json"
    thresholds_path = sdir / "thresholds.json"
    fit = read_json(fit_path)
    completed = read_json(completed_path)
    final_policy = read_json(policy_path)
    records = [candidate_record(seed, c) for c in (fit.get("all_candidates") or [])]
    selected_id = str(completed.get("selected", fit.get("selected")))
    selected_candidate = next((r for r in records if r["id"] == selected_id), None)
    eligible_sorted = sorted([r for r in records if r.get("eligible")], key=rank_key)
    all_sorted = sorted(records, key=rank_key)
    selected_eligible_rank = None
    selected_all_rank = None
    if selected_candidate is not None:
        for i, r in enumerate(eligible_sorted, start=1):
            if r["id"] == selected_id:
                selected_eligible_rank = i
                break
        for i, r in enumerate(all_sorted, start=1):
            if r["id"] == selected_id:
                selected_all_rank = i
                break
    best_eligible_overall = best_of(records, lambda r: r.get("eligible"))
    best_eligible_adaptive = best_of(records, lambda r: r.get("eligible") and r.get("is_adaptive"))
    best_eligible_constant = best_of(records, lambda r: r.get("eligible") and r.get("is_constant_H25"))
    final_class = classify_policy(final_policy)
    if selected_id == "fixed":
        collapse_classification = "selection_chose_fixed_H25_baseline"
    elif final_class["is_constant_H25"]:
        collapse_classification = "selection_chose_structurally_constant_H25_tree"
    elif final_class["is_adaptive"]:
        collapse_classification = "selection_chose_adaptive_tree"
    else:
        collapse_classification = "selection_chose_other"

    formula_residuals = [r.get("objective_minus_cost_plus_half_time_residual") for r in records]
    threshold_summary = None
    if thresholds_path.exists():
        th = read_json(thresholds_path)
        threshold_summary = {"path": rel(thresholds_path), "sha256": sha256(thresholds_path), "keys": sorted(th.keys()) if isinstance(th, Mapping) else None}
        if isinstance(th, Mapping):
            for k, v in th.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    threshold_summary[k] = v
                elif isinstance(v, Mapping):
                    threshold_summary[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str, bool)) or vv is None}
    diagnosis_points: List[str] = []
    if final_class["is_constant_H25"]:
        diagnosis_points.append("final extracted policy is structurally constant H25, so validation H25-only behavior is explained before runtime")
    if selected_id == "fixed":
        diagnosis_points.append("final selection selected the fixed H25 baseline, not a learned/adaptive tree")
    if best_eligible_adaptive and best_eligible_overall:
        if best_eligible_adaptive["id"] != best_eligible_overall["id"]:
            diagnosis_points.append("best eligible training-objective candidate is not adaptive; adaptive candidates were dominated or later lost selection")
        else:
            diagnosis_points.append("an adaptive candidate had the best in-fit training objective; later selection/noise/generalization likely overrode it")
    else:
        diagnosis_points.append("no eligible adaptive candidate was found in fit.json")
    if values_summary(formula_residuals)["max"] is not None and max(abs(x or 0.0) for x in formula_residuals if x is not None) < 1e-9:
        diagnosis_points.append("rank objective exactly matches cost_change + 0.5*(time_ratio-1), confirming a linear cost/timing objective in fit metadata")
    return {
        "seed": seed,
        "paths": {
            "fit": rel(fit_path), "fit_sha256": sha256(fit_path),
            "completed": rel(completed_path), "completed_sha256": sha256(completed_path),
            "policy": rel(policy_path), "policy_sha256": sha256(policy_path),
        },
        "completed_selected": selected_id,
        "fit_selected": fit.get("selected"),
        "final_policy": final_policy,
        "final_policy_classification": final_class,
        "collapse_classification": collapse_classification,
        "candidate_counts": {
            "total": len(records),
            "eligible": count_by(records, lambda r: r.get("eligible")),
            "adaptive": count_by(records, lambda r: r.get("is_adaptive")),
            "eligible_adaptive": count_by(records, lambda r: r.get("eligible") and r.get("is_adaptive")),
            "constant_H25": count_by(records, lambda r: r.get("is_constant_H25")),
            "eligible_constant_H25": count_by(records, lambda r: r.get("eligible") and r.get("is_constant_H25")),
            "has_shorter_than_H25": count_by(records, lambda r: r.get("has_shorter_than_H25")),
            "eligible_has_shorter_than_H25": count_by(records, lambda r: r.get("eligible") and r.get("has_shorter_than_H25")),
            "has_longer_than_H25": count_by(records, lambda r: r.get("has_longer_than_H25")),
            "eligible_has_longer_than_H25": count_by(records, lambda r: r.get("eligible") and r.get("has_longer_than_H25")),
        },
        "best_eligible_overall": best_eligible_overall,
        "best_eligible_adaptive": best_eligible_adaptive,
        "best_eligible_constant_H25": best_eligible_constant,
        "selected_candidate_in_fit": selected_candidate,
        "selected_eligible_rank_in_fit_objective": selected_eligible_rank,
        "selected_all_rank_in_fit_objective": selected_all_rank,
        "top_12_eligible_by_fit_objective": [dict(r) for r in eligible_sorted[:12]],
        "generation_summaries": summarize_generation(records),
        "rank_component_summaries": {
            "all_objective": values_summary(r.get("objective") for r in records),
            "eligible_objective": values_summary(r.get("objective") for r in records if r.get("eligible")),
            "eligible_adaptive_objective": values_summary(r.get("objective") for r in records if r.get("eligible") and r.get("is_adaptive")),
            "eligible_constant_H25_objective": values_summary(r.get("objective") for r in records if r.get("eligible") and r.get("is_constant_H25")),
            "eligible_time_ratio": values_summary(r.get("time_ratio") for r in records if r.get("eligible")),
            "eligible_adaptive_time_ratio": values_summary(r.get("time_ratio") for r in records if r.get("eligible") and r.get("is_adaptive")),
            "eligible_constant_H25_time_ratio": values_summary(r.get("time_ratio") for r in records if r.get("eligible") and r.get("is_constant_H25")),
            "objective_formula_residual": values_summary(formula_residuals),
        },
        "thresholds_metadata": threshold_summary,
        "selection_metadata": inspect_selection_metadata(sdir),
        "diagnosis_points": diagnosis_points,
        "candidate_records": records,
    }


def write_candidate_csv(path: Path, seed_summaries: Mapping[str, Any]) -> None:
    cols = ["seed", "id", "generation", "eligible", "violations", "objective", "cost_change", "physical_change", "time_ratio", "is_adaptive", "is_constant_H25", "has_shorter_than_H25", "has_longer_than_H25", "unique_horizons", "leaves", "objective_minus_cost_plus_half_time_residual", "folder"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for seed in sorted(seed_summaries, key=int):
            for r in seed_summaries[seed]["candidate_records"]:
                row = {c: r.get(c) for c in cols}
                row["unique_horizons"] = json.dumps(row["unique_horizons"])
                row["leaves"] = json.dumps(row["leaves"])
                w.writerow(row)


def write_summary(path: Path, raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle latency-tree training/selection collapse diagnostic v1",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Metadata-only diagnostic over existing vehicle training/selection artifacts. No validation bank reopen, no sealed-test access, no new simulations/control/training.",
        "",
        "## Per-seed findings",
        "",
    ]
    for seed in ["0", "1", "2"]:
        s = raw["seed_summaries"][seed]
        counts = s["candidate_counts"]
        lines += [
            f"### vehicle_s{seed}",
            f"- Final selected: `{s['completed_selected']}`; collapse classification: `{s['collapse_classification']}`; final horizons `{s['final_policy_classification']['unique_horizons']}`.",
            f"- Candidates total/eligible/adaptive/eligible_adaptive/constantH25/eligible_constantH25: `{counts['total']}/{counts['eligible']}/{counts['adaptive']}/{counts['eligible_adaptive']}/{counts['constant_H25']}/{counts['eligible_constant_H25']}`.",
            f"- Best eligible in-fit objective: `{None if s['best_eligible_overall'] is None else s['best_eligible_overall']['id']}`; best eligible adaptive: `{None if s['best_eligible_adaptive'] is None else s['best_eligible_adaptive']['id']}`; best eligible constant H25: `{None if s['best_eligible_constant_H25'] is None else s['best_eligible_constant_H25']['id']}`.",
            f"- Selected in-fit eligible rank: `{s['selected_eligible_rank_in_fit_objective']}`; all-rank: `{s['selected_all_rank_in_fit_objective']}` (None means selected fixed or not present in all_candidates).",
            f"- Diagnosis points: `{s['diagnosis_points']}`.",
            "",
        ]
    lines += [
        "## Cross-seed interpretation",
        "",
        raw["interpretation"],
        "",
        "## Artifacts",
        "",
        f"- Raw: `{raw['artifacts']['raw']}`",
        f"- Candidate table: `{raw['artifacts']['candidate_table_csv']}`",
        f"- Completed marker: `{raw['artifacts']['completed']}`",
        f"- Backup request: `{raw['artifacts']['backup_request']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_once(path: Path, marker: str, body: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    token = f"<!-- {marker} -->"
    if token in old:
        return
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + body.strip() + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    stamp = created.replace("-", "").replace(":", "").replace("+00:00", "Z")
    failures: List[str] = []
    seed_summaries: Dict[str, Any] = {}
    for seed in SEEDS:
        try:
            seed_summaries[str(seed)] = diagnose_seed(seed)
        except Exception as e:
            failures.append(f"vehicle_s{seed} diagnostic failed: {e}")

    classifications = {seed: s.get("collapse_classification") for seed, s in seed_summaries.items()}
    interpretation = (
        "Current evidence supports a training/selection-collapse diagnosis rather than a runtime-dispatch bug: "
        "the final stored policies for s0/s1 are fixed-H25 by structure (s1 explicitly selects the fixed baseline), and the fit metadata objective is a linear cost/timing score cost_change + 0.5*(time_ratio-1) under eligibility gates. "
        "This objective/gating combination frequently rewards near-H25 policies and penalizes risky shorter/adaptive candidates through violations or cost increases; where an adaptive candidate appears strong in the in-fit objective, the later selection race metadata must be treated as noisy development evidence rather than proof of robust generalization. "
        "A versioned IMPROVED revision should therefore target policy extraction/selection objective and risk handling, not merely runtime horizon lookup."
    )

    raw_path = OUT_DIR / "raw.json"
    summary_path = OUT_DIR / "summary.md"
    candidate_csv = OUT_DIR / "candidate_table.csv"
    completed_path = OUT_DIR / "completed.json"
    backup_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_TRAINING_SELECTION_COLLAPSE_DIAGNOSTIC_V1_{stamp}.json"

    raw: Dict[str, Any] = {
        "created_utc": created,
        "script": rel(SCRIPT),
        "script_sha256": sha256(SCRIPT),
        "method": "IMPROVED_latency_tree_vehicle_training_selection_collapse_diagnostic_v1_metadata_only_not_original_SAC",
        "formal_scientific_evidence_created": False,
        "evidence_type": "development_training_selection_metadata_diagnostic",
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "validation_accessed": False,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "seed_summaries": seed_summaries,
        "cross_seed_classifications": classifications,
        "interpretation": interpretation,
        "failures": failures,
    }
    write_candidate_csv(candidate_csv, seed_summaries)
    raw["artifacts"] = {
        "raw": rel(raw_path),
        "summary": rel(summary_path),
        "candidate_table_csv": rel(candidate_csv),
        "completed": rel(completed_path),
        "backup_request": rel(backup_path),
    }
    write_json(raw_path, raw)
    write_summary(summary_path, raw)

    backup_request = {
        "created_utc": created,
        "request": "external_backup_after_vehicle_training_selection_collapse_diagnostic_v1",
        "reason": "Preserve metadata-only training/selection collapse diagnostic before method-revision design or further non-formal case43 replays.",
        "formal_scientific_evidence_created": False,
        "validation_accessed": False,
        "validation_bank_reopened": False,
        "sealed_test_access": "none; sealed final test remained closed, unauthorized, unopened, and unhashed",
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "artifacts_to_backup": [rel(SCRIPT), rel(raw_path), rel(summary_path), rel(candidate_csv), rel(completed_path)],
    }
    write_json(backup_path, backup_request)
    passed = len(failures) == 0
    completed = {
        "created_utc": created,
        "passed": passed,
        "raw": rel(raw_path), "raw_sha256": sha256(raw_path),
        "summary": rel(summary_path), "summary_sha256": sha256(summary_path),
        "candidate_table_csv": rel(candidate_csv), "candidate_table_csv_sha256": sha256(candidate_csv),
        "backup_request": rel(backup_path), "backup_request_sha256": sha256(backup_path),
        "formal_scientific_evidence_created": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "validation_accessed": False,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "headline": {
            "cross_seed_classifications": classifications,
            "selected": {seed: s.get("completed_selected") for seed, s in seed_summaries.items()},
            "final_horizons": {seed: s.get("final_policy_classification", {}).get("unique_horizons") for seed, s in seed_summaries.items()},
            "eligible_adaptive_counts": {seed: s.get("candidate_counts", {}).get("eligible_adaptive") for seed, s in seed_summaries.items()},
            "best_eligible_adaptive": {seed: (None if s.get("best_eligible_adaptive") is None else s.get("best_eligible_adaptive", {}).get("id")) for seed, s in seed_summaries.items()},
        },
        "failures": failures,
    }
    write_json(completed_path, completed)

    doc_body = f"""
### Vehicle training/selection collapse diagnostic v1 ({created})

- Metadata-only development diagnostic over existing vehicle_s0/s1/s2 training/selection artifacts; no validation access, no validation-bank reopen, no sealed-test access/hash, no simulations/control steps/gradient steps.
- Artifacts: `{rel(raw_path)}`, `{rel(summary_path)}`, candidate table `{rel(candidate_csv)}`, completed marker `{rel(completed_path)}`. Backup requested at `{rel(backup_path)}`.
- Headline: `{completed['headline']}`.
- Interpretation: {interpretation}
- Next action: after backup proof for aggregate+this diagnostic, freeze a versioned IMPROVED revision plan targeting selection objective/risk handling and optionally run an instrumented non-formal case43 one-variable replay; do not open final test.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md"]:
        append_once(ROOT / doc, DOC_MARKER + "-" + doc, doc_body)
    append_once(ROOT / "DECISIONS.md", DOC_MARKER + "-decision", f"""
### Decision: current vehicle latency-tree failure is a selection/objective problem, not a dispatch bug ({created})

Before evidence: validation64 aggregate showed learned_s0/s1 were H25-only and learned_s2 adaptive but unsafe on case43; v3 policy diagnostic showed trace/policy consistency.

Diagnostic change: no algorithm/data/controller change. Read existing training/selection metadata only to classify candidate pools, final selected policies, eligibility/objective components, and selection metadata presence.

Decision: current frozen IMPROVED latency-tree candidate is insufficient for final test. The next method revision should modify the learned horizon selection/extraction objective and risk/robustness treatment, while preserving the fixed-H grid and independent validation protocol. Do not tune on or open sealed test.

Evidence: `{rel(completed_path)}` headline `{completed['headline']}`.
""")

    print(json.dumps(completed, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
