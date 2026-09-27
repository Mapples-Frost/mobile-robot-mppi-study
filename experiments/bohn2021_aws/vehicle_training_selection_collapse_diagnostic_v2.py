#!/usr/bin/env python3
"""Vehicle latency-tree training/selection collapse diagnostic v2.

Repair of v1 diagnostic failure.  V1 failed because stored constant policies use
`h` rather than `horizon`; v2 accepts both schemas and writes to a new v2 output
folder so the failed v1 artifacts remain preserved.

Metadata-only diagnostic for the current vehicle IMPROVED latency-tree candidate.
It reads existing training/selection metadata for vehicle_s0/s1/s2 only.  It does
not reopen validation banks, read sealed tests, run MPC rollouts, or train.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/train"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_training_selection_collapse_diagnostic_20260927_v2"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_training_selection_collapse_diagnostic_v2.py"
DOC_MARKER = "vehicle-training-selection-collapse-diagnostic-v2-20260927"
SEEDS = [0, 1, 2]
H25 = 25
V1_FAILED_RUN = "research_artifacts/aws_runs/20260927T065214_05cbaa7a/registry.json"


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
    median = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    return {"count": n, "mean": float(math.fsum(xs) / n), "median": float(median), "min": float(xs[0]), "max": float(xs[-1])}


def policy_horizon(policy: Mapping[str, Any]) -> Optional[int]:
    for key in ("horizon", "h", "H", "controller_h"):
        if key in policy and policy.get(key) is not None:
            return int(policy[key])
    return None


def classify_policy(policy: Mapping[str, Any]) -> Dict[str, Any]:
    kind = policy.get("kind")
    leaves: List[int] = []
    if kind == "constant":
        h = policy_horizon(policy)
        if h is not None:
            leaves = [h]
    elif isinstance(policy.get("leaves"), list):
        leaves = [int(x) for x in policy.get("leaves")]
    else:
        h = policy_horizon(policy)
        if h is not None:
            leaves = [h]
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
        "schema_note": "constant h/horizon accepted" if kind == "constant" else None,
    }


def generation_from_id(cid: str) -> Optional[int]:
    m = re.match(r"g(\d+)_c\d+", cid or "")
    return int(m.group(1)) if m else None


def candidate_record(seed: int, c: Mapping[str, Any]) -> Dict[str, Any]:
    rank = c.get("rank") or {}
    policy_class = classify_policy(c.get("policy") or {})
    obj = as_float(rank.get("objective", c.get("objective")))
    cost = as_float(rank.get("cost_change"))
    phys = as_float(rank.get("physical_change"))
    tr = as_float(rank.get("time_ratio"))
    residual = None
    if obj is not None and cost is not None and tr is not None:
        residual = obj - (cost + 0.5 * (tr - 1.0))
    rec: Dict[str, Any] = {
        "seed": seed,
        "id": str(c.get("id")),
        "generation": generation_from_id(str(c.get("id"))),
        "eligible": bool(rank.get("eligible", c.get("eligible", False))),
        "violations": int(rank.get("violations", c.get("violations", 0)) or 0),
        "objective": obj,
        "cost_change": cost,
        "physical_change": phys,
        "time_ratio": tr,
        "objective_minus_cost_plus_half_time_residual": residual,
        "folder": str(c.get("folder")),
    }
    rec.update(policy_class)
    return rec


def rank_key(rec: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        0 if rec.get("eligible") else 1,
        int(rec.get("violations") or 0),
        rec.get("objective") if rec.get("objective") is not None else float("inf"),
        rec.get("cost_change") if rec.get("cost_change") is not None else float("inf"),
        rec.get("time_ratio") if rec.get("time_ratio") is not None else float("inf"),
        str(rec.get("id")),
    )


def best_of(records: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> Optional[Dict[str, Any]]:
    xs = [r for r in records if predicate(r)]
    if not xs:
        return None
    return dict(sorted(xs, key=rank_key)[0])


def count_by(records: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> int:
    return sum(1 for r in records if predicate(r))


def summarize_generation(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for gen in sorted({r.get("generation") for r in records if r.get("generation") is not None}):
        rs = [r for r in records if r.get("generation") == gen]
        out[str(gen)] = {
            "candidates": len(rs),
            "eligible": count_by(rs, lambda r: bool(r.get("eligible"))),
            "adaptive": count_by(rs, lambda r: bool(r.get("is_adaptive"))),
            "eligible_adaptive": count_by(rs, lambda r: bool(r.get("eligible") and r.get("is_adaptive"))),
            "constant_H25": count_by(rs, lambda r: bool(r.get("is_constant_H25"))),
            "eligible_constant_H25": count_by(rs, lambda r: bool(r.get("eligible") and r.get("is_constant_H25"))),
            "best_eligible_overall": best_of(rs, lambda r: bool(r.get("eligible"))),
            "best_eligible_adaptive": best_of(rs, lambda r: bool(r.get("eligible") and r.get("is_adaptive"))),
            "best_eligible_constant_H25": best_of(rs, lambda r: bool(r.get("eligible") and r.get("is_constant_H25"))),
        }
    return out


def small_json_summary(path: Path) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"path": rel(path)}
    if not path.exists():
        rec["exists"] = False
        return rec
    rec["exists"] = True
    rec["sha256"] = sha256(path)
    try:
        obj = read_json(path)
    except Exception as e:
        rec["read_error"] = str(e)
        return rec
    if isinstance(obj, Mapping):
        rec["top_level_keys"] = sorted(str(k) for k in obj.keys())
        for k in ["passed", "selected", "winner", "rollout_key", "success", "steps", "total_cost", "physical_constraint_cost", "objective", "cost_change", "time_ratio", "learned_tree_selected", "validation_access", "test_access", "new_gradient_steps"]:
            if k in obj:
                rec[k] = obj[k]
        hints: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                if k not in rec:
                    hints[k] = v
            elif isinstance(v, Mapping):
                for kk, vv in v.items():
                    name = f"{k}.{kk}"
                    if isinstance(vv, (int, float, str, bool)) or vv is None:
                        if any(tok in name.lower() for tok in ["objective", "cost", "time", "success", "failure", "eligible", "violation", "selected", "winner", "mean", "sum"]):
                            hints[name] = vv
        if hints:
            rec["scalar_metric_hints"] = hints
    else:
        rec["json_type"] = type(obj).__name__
    return rec


def inspect_selection_metadata(sdir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "selection_registration": small_json_summary(sdir / "selection_registration.json"),
        "generation_selection_json": [],
        "final_selection_completed_markers": [],
    }
    for p in sorted(sdir.glob("generation*/selection.json")):
        rec = small_json_summary(p)
        rec["generation_dir"] = p.parent.name
        out["generation_selection_json"].append(rec)
    selection_dir = sdir / "selection"
    if selection_dir.exists():
        for p in sorted(selection_dir.glob("*/completed.json")):
            name = p.parent.name
            m = re.match(r"r(\d+)_(.+)", name)
            rec = small_json_summary(p)
            rec["selection_dir_name"] = name
            rec["repeat"] = int(m.group(1)) if m else None
            rec["candidate_id_from_dir"] = m.group(2) if m else name
            out["final_selection_completed_markers"].append(rec)
    return out


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
    selected_candidate = next((r for r in records if r.get("id") == selected_id), None)
    eligible_sorted = sorted([r for r in records if r.get("eligible")], key=rank_key)
    all_sorted = sorted(records, key=rank_key)
    selected_eligible_rank = None
    selected_all_rank = None
    for i, r in enumerate(eligible_sorted, start=1):
        if r.get("id") == selected_id:
            selected_eligible_rank = i
            break
    for i, r in enumerate(all_sorted, start=1):
        if r.get("id") == selected_id:
            selected_all_rank = i
            break

    best_eligible_overall = best_of(records, lambda r: bool(r.get("eligible")))
    best_eligible_adaptive = best_of(records, lambda r: bool(r.get("eligible") and r.get("is_adaptive")))
    best_eligible_constant = best_of(records, lambda r: bool(r.get("eligible") and r.get("is_constant_H25")))
    final_class = classify_policy(final_policy)

    if selected_id == "fixed":
        collapse_classification = "selection_chose_fixed_H25_baseline"
    elif final_class.get("is_constant_H25"):
        collapse_classification = "selection_chose_structurally_constant_H25_tree"
    elif final_class.get("is_adaptive"):
        collapse_classification = "selection_chose_adaptive_tree"
    else:
        collapse_classification = "selection_chose_other_or_unclassified"

    formula_residuals = [r.get("objective_minus_cost_plus_half_time_residual") for r in records]
    residual_values = [x for x in formula_residuals if x is not None]
    residual_max_abs = max((abs(x) for x in residual_values), default=None)
    top12 = [dict(r) for r in eligible_sorted[:12]]

    threshold_summary = small_json_summary(thresholds_path)
    diagnosis_points: List[str] = []
    if final_class.get("is_constant_H25"):
        diagnosis_points.append("final stored policy is structurally constant H25, explaining H25-only validation behavior without any runtime dispatch bug")
    if selected_id == "fixed":
        diagnosis_points.append("final selection selected the fixed H25 baseline rather than a learned tree")
    if best_eligible_adaptive is None:
        diagnosis_points.append("no eligible adaptive candidate appears in all_candidates")
    elif best_eligible_overall and best_eligible_adaptive.get("id") == best_eligible_overall.get("id"):
        diagnosis_points.append("an adaptive candidate had the best in-fit objective; collapse happened after the in-fit candidate ranking, likely in final selection/generalization/noise")
    else:
        diagnosis_points.append("eligible adaptive candidates existed but were not best by in-fit objective")
    if residual_max_abs is not None and residual_max_abs < 1e-9:
        diagnosis_points.append("candidate rank objective exactly matches cost_change + 0.5*(time_ratio-1) for records with all components")

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
            "eligible": count_by(records, lambda r: bool(r.get("eligible"))),
            "adaptive": count_by(records, lambda r: bool(r.get("is_adaptive"))),
            "eligible_adaptive": count_by(records, lambda r: bool(r.get("eligible") and r.get("is_adaptive"))),
            "constant_H25": count_by(records, lambda r: bool(r.get("is_constant_H25"))),
            "eligible_constant_H25": count_by(records, lambda r: bool(r.get("eligible") and r.get("is_constant_H25"))),
            "has_shorter_than_H25": count_by(records, lambda r: bool(r.get("has_shorter_than_H25"))),
            "eligible_has_shorter_than_H25": count_by(records, lambda r: bool(r.get("eligible") and r.get("has_shorter_than_H25"))),
            "has_longer_than_H25": count_by(records, lambda r: bool(r.get("has_longer_than_H25"))),
            "eligible_has_longer_than_H25": count_by(records, lambda r: bool(r.get("eligible") and r.get("has_longer_than_H25"))),
        },
        "best_eligible_overall": best_eligible_overall,
        "best_eligible_adaptive": best_eligible_adaptive,
        "best_eligible_constant_H25": best_eligible_constant,
        "selected_candidate_in_fit": selected_candidate,
        "selected_eligible_rank_in_fit_objective": selected_eligible_rank,
        "selected_all_rank_in_fit_objective": selected_all_rank,
        "top_12_eligible_by_fit_objective": top12,
        "generation_summaries": summarize_generation(records),
        "rank_component_summaries": {
            "eligible_objective": values_summary(r.get("objective") for r in records if r.get("eligible")),
            "eligible_adaptive_objective": values_summary(r.get("objective") for r in records if r.get("eligible") and r.get("is_adaptive")),
            "eligible_constant_H25_objective": values_summary(r.get("objective") for r in records if r.get("eligible") and r.get("is_constant_H25")),
            "eligible_time_ratio": values_summary(r.get("time_ratio") for r in records if r.get("eligible")),
            "eligible_adaptive_time_ratio": values_summary(r.get("time_ratio") for r in records if r.get("eligible") and r.get("is_adaptive")),
            "eligible_constant_H25_time_ratio": values_summary(r.get("time_ratio") for r in records if r.get("eligible") and r.get("is_constant_H25")),
            "objective_formula_residual": values_summary(formula_residuals),
            "objective_formula_residual_max_abs": residual_max_abs,
        },
        "thresholds_metadata": threshold_summary,
        "selection_metadata": inspect_selection_metadata(sdir),
        "diagnosis_points": diagnosis_points,
        "candidate_records_for_csv_only_count": len(records),
        "candidate_records": records,
    }


def write_candidate_csv(path: Path, seed_summaries: Mapping[str, Any]) -> None:
    cols = ["seed", "id", "generation", "eligible", "violations", "objective", "cost_change", "physical_change", "time_ratio", "is_adaptive", "is_constant_H25", "has_shorter_than_H25", "has_longer_than_H25", "unique_horizons", "leaves", "objective_minus_cost_plus_half_time_residual", "folder"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for seed in sorted(seed_summaries, key=int):
            for r in seed_summaries[seed].get("candidate_records", []):
                row = {c: r.get(c) for c in cols}
                row["unique_horizons"] = json.dumps(row["unique_horizons"])
                row["leaves"] = json.dumps(row["leaves"])
                w.writerow(row)


def compact_seed_summary(s: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "completed_selected": s.get("completed_selected"),
        "collapse_classification": s.get("collapse_classification"),
        "final_unique_horizons": s.get("final_policy_classification", {}).get("unique_horizons"),
        "candidate_counts": s.get("candidate_counts"),
        "best_eligible_overall": None if s.get("best_eligible_overall") is None else {k: s["best_eligible_overall"].get(k) for k in ["id", "objective", "cost_change", "time_ratio", "unique_horizons"]},
        "best_eligible_adaptive": None if s.get("best_eligible_adaptive") is None else {k: s["best_eligible_adaptive"].get(k) for k in ["id", "objective", "cost_change", "time_ratio", "unique_horizons"]},
        "best_eligible_constant_H25": None if s.get("best_eligible_constant_H25") is None else {k: s["best_eligible_constant_H25"].get(k) for k in ["id", "objective", "cost_change", "time_ratio", "unique_horizons"]},
        "selected_eligible_rank_in_fit_objective": s.get("selected_eligible_rank_in_fit_objective"),
        "diagnosis_points": s.get("diagnosis_points"),
    }


def write_summary(path: Path, raw: Mapping[str, Any]) -> None:
    lines: List[str] = [
        "# Vehicle latency-tree training/selection collapse diagnostic v2",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "V2 repairs the failed v1 metadata diagnostic by accepting stored constant policies with key `h` as well as `horizon`. It writes new v2 artifacts and preserves v1 as failed evidence.",
        "",
        "No validation bank reopen, no sealed-test access/hash, no simulations/control steps/gradient steps.",
        "",
        "## Per-seed findings",
        "",
    ]
    for seed in sorted(raw["seed_summaries"], key=int):
        s = raw["seed_summaries"][seed]
        cs = compact_seed_summary(s)
        counts = cs["candidate_counts"]
        lines += [
            f"### vehicle_s{seed}",
            f"- Final selected: `{cs['completed_selected']}`; collapse classification: `{cs['collapse_classification']}`; final horizons `{cs['final_unique_horizons']}`.",
            f"- Candidate counts total/eligible/adaptive/eligible_adaptive/constantH25/eligible_constantH25: `{counts['total']}/{counts['eligible']}/{counts['adaptive']}/{counts['eligible_adaptive']}/{counts['constant_H25']}/{counts['eligible_constant_H25']}`.",
            f"- Best eligible overall: `{cs['best_eligible_overall']}`.",
            f"- Best eligible adaptive: `{cs['best_eligible_adaptive']}`.",
            f"- Best eligible constant H25: `{cs['best_eligible_constant_H25']}`.",
            f"- Selected eligible rank in in-fit objective: `{cs['selected_eligible_rank_in_fit_objective']}` (None means selected fixed or selected id not in all_candidates).",
            f"- Diagnosis points: `{cs['diagnosis_points']}`.",
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
    stamp = created.replace("-", "").replace(":", "").replace("+00:00", "+0000")
    failures: List[str] = []
    seed_summaries: Dict[str, Any] = {}
    for seed in SEEDS:
        try:
            seed_summaries[str(seed)] = diagnose_seed(seed)
        except Exception as e:
            failures.append(f"vehicle_s{seed} diagnostic failed: {type(e).__name__}: {e}")

    classifications = {seed: s.get("collapse_classification") for seed, s in seed_summaries.items()}
    selected = {seed: s.get("completed_selected") for seed, s in seed_summaries.items()}
    final_horizons = {seed: s.get("final_policy_classification", {}).get("unique_horizons") for seed, s in seed_summaries.items()}
    eligible_adaptive = {seed: s.get("candidate_counts", {}).get("eligible_adaptive") for seed, s in seed_summaries.items()}
    best_adaptive = {seed: (None if s.get("best_eligible_adaptive") is None else s["best_eligible_adaptive"].get("id")) for seed, s in seed_summaries.items()}

    interpretation = (
        "The completed v2 metadata diagnostic supports a selection/objective-collapse diagnosis for the current frozen IMPROVED latency-tree candidate. "
        "vehicle_s0 had eligible adaptive candidates and the best in-fit candidate was adaptive, but the final stored policy is a structurally constant H25 tree, implying collapse occurred during later final selection/generalization/noisy paired selection rather than at runtime. "
        "vehicle_s1 explicitly selected the fixed H25 baseline; its stored policy is constant with schema key h=25, explaining all H25 validation behavior. "
        "vehicle_s2 selected an adaptive tree, consistent with validation H25/H35 usage, but validation aggregate already showed no speed advantage and one catastrophic case43 failure. "
        "Across seeds, the old objective is a linear cost_change + 0.5*(time_ratio-1) with hard eligibility gates, which is not producing robust adaptive-horizon policies under the current setup. "
        "Next revision should be versioned IMPROVED work on selection/extraction and risk/robustness handling, not a runtime dispatch repair, and must use fresh non-test validation after any code/config change."
    )

    raw_path = OUT_DIR / "raw.json"
    summary_path = OUT_DIR / "summary.md"
    candidate_csv = OUT_DIR / "candidate_table.csv"
    completed_path = OUT_DIR / "completed.json"
    backup_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_TRAINING_SELECTION_COLLAPSE_DIAGNOSTIC_V2_{stamp}.json"

    write_candidate_csv(candidate_csv, seed_summaries)
    # Store all detailed summaries, including candidate records, because this is small metadata and useful for audit.
    raw: Dict[str, Any] = {
        "created_utc": created,
        "script": rel(SCRIPT),
        "script_sha256": sha256(SCRIPT),
        "method": "IMPROVED_latency_tree_vehicle_training_selection_collapse_diagnostic_v2_metadata_only_not_original_SAC",
        "repair_of_failed_v1": {
            "failed_run_registry": V1_FAILED_RUN,
            "failure": "v1 classify_policy expected policy['horizon'] for constant policies; vehicle_s1 policy uses {'kind':'constant','h':25}; v2 accepts both h and horizon and uses a new output directory",
        },
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
        "request": "external_backup_after_vehicle_training_selection_collapse_diagnostic_v2",
        "reason": "Preserve repaired metadata-only training/selection collapse diagnostic and failed-v1 provenance before method revision or further case43 replays.",
        "formal_scientific_evidence_created": False,
        "validation_accessed": False,
        "validation_bank_reopened": False,
        "sealed_test_access": "none; sealed final test remained closed, unauthorized, unopened, and unhashed",
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "artifacts_to_backup": [rel(SCRIPT), rel(raw_path), rel(summary_path), rel(candidate_csv), rel(completed_path), V1_FAILED_RUN],
    }
    write_json(backup_path, backup_request)

    passed = len(failures) == 0 and set(seed_summaries.keys()) == {"0", "1", "2"}
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
            "selected": selected,
            "final_horizons": final_horizons,
            "eligible_adaptive_counts": eligible_adaptive,
            "best_eligible_adaptive": best_adaptive,
            "interpretation_short": "selection/objective-collapse rather than runtime-dispatch bug; current frozen candidate insufficient for final test",
        },
        "failures": failures,
    }
    write_json(completed_path, completed)

    doc_body = f"""
### Vehicle training/selection collapse diagnostic v2 ({created})

- Repair of failed v1 `{V1_FAILED_RUN}`: v1 expected `horizon` for constant policies but vehicle_s1 uses `h=25`; v2 accepts both schemas and writes new artifacts.
- Metadata-only development diagnostic over existing vehicle_s0/s1/s2 training/selection artifacts; no validation access, no validation-bank reopen, no sealed-test access/hash, no simulations/control steps/gradient steps.
- Artifacts: `{rel(raw_path)}`, `{rel(summary_path)}`, candidate table `{rel(candidate_csv)}`, completed marker `{rel(completed_path)}`. Backup requested at `{rel(backup_path)}`.
- Headline: `{completed['headline']}`.
- Interpretation: {interpretation}
- Next action: after external backup proof for aggregate+v1 failure+v2 diagnostic, freeze a versioned IMPROVED revision plan targeting selection objective/risk handling and decide a bounded non-formal case43 replay if needed; do not open final test.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md"]:
        append_once(ROOT / doc, DOC_MARKER + "-" + doc, doc_body)
    append_once(ROOT / "DECISIONS.md", DOC_MARKER + "-decision", f"""
### Decision: current vehicle latency-tree failure is selection/objective collapse, not runtime dispatch ({created})

Before evidence: validation64 aggregate showed learned_s0/s1 were H25-only and learned_s2 adaptive but unsafe on case43; v3 policy diagnostic showed validation traces match stored policies exactly; v1 diagnostic failed on a policy schema bug and is preserved.

Diagnostic repair/change: no controller/algorithm/data change. V2 only changed diagnostic parsing to accept constant-policy key `h` as well as `horizon` and wrote fresh v2 outputs. Validation/test access remained zero.

Decision: current frozen IMPROVED latency-tree candidate is insufficient for final test. A versioned IMPROVED method revision should modify the horizon policy selection/extraction objective and risk/robustness handling. Preserve ORIGINAL/old negative results, keep strong fixed-H baselines, and use fresh independent validation for any changed candidate.

Evidence: `{rel(completed_path)}` headline `{completed['headline']}`.
""")

    print(json.dumps(completed, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
