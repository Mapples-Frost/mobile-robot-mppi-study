#!/usr/bin/env python3
"""T-C3 zero-solve matched-parameter ledger for v34u solver-entry records.

This script implements the active Opus structured task
`T-C3-matched-parameter-ledger` from plan 20260930T133239Z_8fd7da.
It reads only already-opened development artifacts and does not construct an
environment, call a solver, step a plant, train/refit, open validation64, or open
sealed/final test data.

The task outcome is intentionally conservative. If the existing artifact only
contains opt_p hashes and not full numeric opt_p vectors, the script records that
insufficiency as the scientific result instead of fabricating a per-entry ledger.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
try:
    import execution_contract  # type: ignore
except ImportError:
    sys.path.insert(0, str(ROOT / "scripts" / "research_service"))
    import execution_contract  # type: ignore

NAME = "vehicle_true_variable_horizon_matched_parameter_ledger_v0"
TASK_ID = "T-C3-matched-parameter-ledger"
EXPECTED_REQUEST = "execution-result:20260930T131908_12b79445"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
V34U_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z/raw.json"
V34U_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z/completed.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"

EXPECTED_OPT_P_HASHES = {
    "H12_canonical": "7032ac0c47903ccfe27a25accc556b22331ba32247bf1a8b7a95f996cd1fdd5a",
    "H15_canonical": "ca19ce63ea7aa807e66c33f9871022b24e1e7bd1ac0ee3292a82cc0025b52502",
    "H35_canonical": "9878b3d2235205772b48cf79a3e30d8eedd4e625feb2dc3d758130faa0ea651d",
    "H15_goal_facing": "fbe488a0e5749ee25682b7f6a963294112b041c31e0340de4214480cab638352",
}
ZERO_RESOURCES = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}
CLASSIFICATIONS = {"nuisance_obstacle_forecast_or_noise", "objective_relevant", "structural"}
NUMERIC_TOL = 1e-12


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


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def traverse(obj: Any, path: str = "root") -> Iterable[Tuple[str, Any]]:
    yield path, obj
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from traverse(v, "%s.%s" % (path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from traverse(v, "%s[%d]" % (path, i))


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def flatten_numeric(value: Any, cap: int = 1000000) -> Optional[List[float]]:
    out: List[float] = []
    stack = [value]
    while stack:
        x = stack.pop(0)
        if is_number(x):
            out.append(float(x))
            if len(out) > cap:
                return None
        elif isinstance(x, (list, tuple)):
            stack = list(x) + stack
        else:
            return None
    return out if out else None


def arm_label(arm: Mapping[str, Any]) -> str:
    horizon = int(arm.get("horizon", 0) or 0)
    init = str(arm.get("initialization", ""))
    role = str(arm.get("role", arm.get("cell_role", "")))
    if horizon == 15 and ("goal" in init or "goal" in role or "alias_separation" in role):
        return "H15_goal_facing"
    if horizon == 12:
        return "H12_canonical"
    if horizon == 15:
        return "H15_canonical"
    if horizon == 35:
        return "H35_canonical"
    return "H%d_%s" % (horizon, init or role or "unknown")


def find_arms(raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    arms = raw.get("arms")
    if not isinstance(arms, list):
        raise RuntimeError("v34u raw does not contain an arms list")
    labelled: Dict[str, Mapping[str, Any]] = {}
    for arm in arms:
        if isinstance(arm, Mapping):
            label = arm_label(arm)
            if label in EXPECTED_OPT_P_HASHES and label not in labelled:
                labelled[label] = arm
    missing = [k for k in EXPECTED_OPT_P_HASHES if k not in labelled]
    if missing:
        raise RuntimeError("missing required v34u arms: " + repr(missing))
    return [labelled[k] for k in ["H12_canonical", "H15_canonical", "H35_canonical", "H15_goal_facing"]]


def find_opt_p_hash_paths(arm: Mapping[str, Any]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for path, value in traverse(arm, "arm"):
        if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower()):
            if "opt_p_hash" in path.lower() or value in set(EXPECTED_OPT_P_HASHES.values()):
                hits.append({"path": path, "sha256": value})
    return hits


def expected_p_size(arm: Mapping[str, Any]) -> Optional[int]:
    for path, value in traverse(arm, "arm"):
        if isinstance(value, str) and path.endswith("captured_nlp_meta.p"):
            text = value.strip()
            if text.startswith("(") and "," in text:
                try:
                    return int(text[1:].split(",", 1)[0].strip())
                except Exception:
                    pass
    return None


def score_vector_path(path: str, n: int, p_size: Optional[int]) -> int:
    lower = path.lower()
    score = n
    if p_size is not None and n == p_size:
        score += 5000000
    for token, weight in [
        ("opt_p_num", 3000000),
        ("opt_p", 2500000),
        ("p_num", 2200000),
        ("_p", 1200000),
        ("parameter", 1000000),
        ("tvp", 500000),
        ("solver", 250000),
    ]:
        if token in lower:
            score += weight
    for bad, penalty in [
        ("stage_first", 2000000),
        ("candidate", 1500000),
        ("objective", 1200000),
        ("terminal", 1000000),
        ("first_control", 1000000),
        ("iterations", 1000000),
        ("hash", 1000000),
        ("cost", 750000),
        ("time", 750000),
        ("wall", 750000),
    ]:
        if bad in lower:
            score -= penalty
    return score


def find_parameter_vector(arm: Mapping[str, Any]) -> Dict[str, Any]:
    p_size = expected_p_size(arm)
    candidates: List[Dict[str, Any]] = []
    for path, value in traverse(arm, "arm"):
        if not isinstance(value, list):
            continue
        flat = flatten_numeric(value)
        if flat is None or len(flat) < 10:
            continue
        lower = path.lower()
        if not any(tok in lower for tok in ["opt_p", "p_num", "parameter", "tvp", "solver", "_p"]):
            continue
        score = score_vector_path(path, len(flat), p_size)
        candidates.append({"path": path, "count": len(flat), "score": score, "values": flat})
    candidates.sort(key=lambda c: (c["score"], c["count"]), reverse=True)
    if candidates and candidates[0]["score"] > 0:
        chosen = candidates[0]
        return {
            "available": True,
            "path": chosen["path"],
            "count": chosen["count"],
            "expected_p_size": p_size,
            "score": chosen["score"],
            "values": chosen["values"],
            "candidate_summaries": [{"path": c["path"], "count": c["count"], "score": c["score"]} for c in candidates[:20]],
        }
    return {
        "available": False,
        "reason": "No full numeric opt_p vector candidate was serialized under this arm; only hashes and metadata may be present.",
        "expected_p_size": p_size,
        "candidate_summaries": [{"path": c["path"], "count": c["count"], "score": c["score"]} for c in candidates[:20]],
    }


def label_for_index(index: int, fallback_prefix: str = "opt_p") -> str:
    return "%s[%d]" % (fallback_prefix, index)


def classify_parameter(label: str, path: str, delta: Optional[float], dimension_mismatch: bool = False) -> Tuple[str, str]:
    text = (label + " " + path).lower()
    if dimension_mismatch:
        return "structural", "Vector length or n_horizon differs; this is a structural solver-entry mismatch."
    if any(tok in text for tok in ["obj_", "obstacle", "obs", "noise", "forecast", "r_obstacle", "distance"]):
        return "nuisance_obstacle_forecast_or_noise", "Obstacle/TVP/noise-related entry; it blocks pure initialization-basin attribution even if expected across horizons."
    if any(tok in text for tok in ["goal", "reference", "trajectory", "target", "terminal", "weight", "coefficient", "hend"]):
        return "objective_relevant", "Goal/reference/terminal/weight entry can alter the objective or terminal target."
    if any(tok in text for tok in ["previous_input", "u_prev", "u0", "bound", "lb", "ub", "scale", "scaling", "n_horizon", "horizon"]):
        return "structural", "Bounds, previous input, scaling, or horizon parameter is a structural solver-entry component."
    if delta is not None and abs(delta) > NUMERIC_TOL:
        return "objective_relevant", "Unlabelled numeric opt_p difference; conservatively treated as objective/constraint relevant."
    return "structural", "No material numeric difference."


def pairwise(labels: Sequence[str]) -> Iterable[Tuple[str, str]]:
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            yield labels[i], labels[j]


def write_failure(run_dir: Path, message: str, snapshot: Optional[Mapping[str, Any]], engineering_error: str = "loader") -> int:
    failed = run_dir / "failed.json"
    write_json(failed, {
        "status": "failed",
        "classification": "engineering_failure_zero_resource_before_scientific_outcome",
        "created_utc": now_utc().isoformat(),
        "error": message,
        "traceback_tail": traceback.format_exc().splitlines()[-8:],
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
    })
    if snapshot is not None:
        execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO_RESOURCES), {"no_scientific_outcome": True, "error": message, "failed_json": rel(failed)}, engineering_error=engineering_error)
    print(json.dumps({"failed": message, "failed_json": rel(failed), "resources": ZERO_RESOURCES}, sort_keys=True), flush=True)
    return 1


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "research_artifacts/aws_diagnostics/%s_%s" % (NAME, stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = "vehicle-tc3-matched-parameter-ledger-%s" % stamp

    try:
        snapshot = execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
    except Exception as exc:
        return write_failure(run_dir, "structured execution snapshot verification failed: %s: %s" % (type(exc).__name__, exc), None, "authorization")
    if snapshot is None:
        return write_failure(run_dir, "missing structured execution snapshot", None, "authorization")

    try:
        if not V34U_RAW.exists() or not V34U_COMPLETED.exists():
            return write_failure(run_dir, "missing v34u raw/completed input", snapshot, "missing_file")

        raw_artifact = read_json(V34U_RAW)
        arms = find_arms(raw_artifact)
        labels = [arm_label(a) for a in arms]
        arm_by_label = dict(zip(labels, arms))
        hashes_by_label: Dict[str, List[Dict[str, Any]]] = {label: find_opt_p_hash_paths(arm_by_label[label]) for label in labels}
        vector_by_label: Dict[str, Dict[str, Any]] = {label: find_parameter_vector(arm_by_label[label]) for label in labels}

        arm_summary_path = run_dir / "opt_p_arm_summary.csv"
        with arm_summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["arm_label", "arm_id", "horizon", "initialization", "expected_hash", "expected_hash_found", "hash_paths", "vector_available", "vector_path", "vector_count", "expected_p_size", "vector_reason"])
            writer.writeheader()
            for label in labels:
                vec = vector_by_label[label]
                hashes = hashes_by_label[label]
                expected_hash = EXPECTED_OPT_P_HASHES[label]
                writer.writerow({
                    "arm_label": label,
                    "arm_id": arm_by_label[label].get("arm_id"),
                    "horizon": arm_by_label[label].get("horizon"),
                    "initialization": arm_by_label[label].get("initialization"),
                    "expected_hash": expected_hash,
                    "expected_hash_found": any(h.get("sha256") == expected_hash for h in hashes),
                    "hash_paths": ";".join("%s=%s" % (h.get("path"), h.get("sha256")) for h in hashes),
                    "vector_available": vec.get("available"),
                    "vector_path": vec.get("path"),
                    "vector_count": vec.get("count"),
                    "expected_p_size": vec.get("expected_p_size"),
                    "vector_reason": vec.get("reason"),
                })

        diff_rows: List[Dict[str, Any]] = []
        pair_statements: Dict[str, Dict[str, Any]] = {}
        all_values_available_for_required_pairs = True
        for a, b in pairwise(labels):
            va = vector_by_label[a]
            vb = vector_by_label[b]
            pair = "%s__vs__%s" % (a, b)
            expected_hash_a = EXPECTED_OPT_P_HASHES[a]
            expected_hash_b = EXPECTED_OPT_P_HASHES[b]
            hashes_differ = expected_hash_a != expected_hash_b
            if not (va.get("available") and vb.get("available")):
                all_values_available_for_required_pairs = False
                cls, reason = classify_parameter("opt_p_hash_only", "hash-only", None, dimension_mismatch=False)
                diff_rows.append({
                    "pair": pair,
                    "index": "NA",
                    "parameter_label": "opt_p_vector_unavailable_hash_only",
                    "arm_a": a,
                    "arm_b": b,
                    "arm_a_path": ";".join(h.get("path", "") for h in hashes_by_label[a]),
                    "arm_b_path": ";".join(h.get("path", "") for h in hashes_by_label[b]),
                    "arm_a_value": "",
                    "arm_b_value": "",
                    "delta_b_minus_a": "",
                    "abs_delta": "",
                    "relative_delta": "",
                    "classification_primary": "objective_relevant" if hashes_differ else cls,
                    "classification_reason": "Only opt_p hashes are recorded, not full numeric vectors; hashes differ=%s, so per-entry attribution is not possible from this artifact." % hashes_differ,
                })
                pair_statements[pair] = {
                    "matched_parameter_comparison_possible": False,
                    "reason": "Full numeric opt_p vectors are not serialized for one or both arms; hash-only evidence cannot support an initialization-only basin attribution.",
                    "hashes_differ": hashes_differ,
                }
                continue
            values_a = list(va["values"])
            values_b = list(vb["values"])
            if len(values_a) != len(values_b):
                cls, reason = classify_parameter("opt_p_dimension", "%s vs %s" % (va.get("path"), vb.get("path")), None, dimension_mismatch=True)
                diff_rows.append({
                    "pair": pair,
                    "index": "dimension",
                    "parameter_label": "opt_p_vector_dimension",
                    "arm_a": a,
                    "arm_b": b,
                    "arm_a_path": va.get("path"),
                    "arm_b_path": vb.get("path"),
                    "arm_a_value": len(values_a),
                    "arm_b_value": len(values_b),
                    "delta_b_minus_a": len(values_b) - len(values_a),
                    "abs_delta": abs(len(values_b) - len(values_a)),
                    "relative_delta": "",
                    "classification_primary": cls,
                    "classification_reason": reason,
                })
                pair_statements[pair] = {"matched_parameter_comparison_possible": False, "reason": "Numeric vectors have different lengths; this is a structural mismatch.", "hashes_differ": hashes_differ}
                continue
            diff_count = 0
            for i, (x, y) in enumerate(zip(values_a, values_b)):
                delta = y - x
                if abs(delta) <= NUMERIC_TOL:
                    continue
                diff_count += 1
                label = label_for_index(i)
                cls, reason = classify_parameter(label, "%s | %s" % (va.get("path"), vb.get("path")), delta)
                diff_rows.append({
                    "pair": pair,
                    "index": i,
                    "parameter_label": label,
                    "arm_a": a,
                    "arm_b": b,
                    "arm_a_path": va.get("path"),
                    "arm_b_path": vb.get("path"),
                    "arm_a_value": x,
                    "arm_b_value": y,
                    "delta_b_minus_a": delta,
                    "abs_delta": abs(delta),
                    "relative_delta": abs(delta) / max(abs(x), abs(y), 1.0),
                    "classification_primary": cls,
                    "classification_reason": reason,
                })
            pair_statements[pair] = {
                "matched_parameter_comparison_possible": diff_count == 0 and not hashes_differ,
                "reason": "Full vectors compared; differing entries=%d; hashes_differ=%s." % (diff_count, hashes_differ),
                "hashes_differ": hashes_differ,
                "differing_entry_count": diff_count,
            }

        diff_csv = run_dir / "parameter_diff_table.csv"
        fields = ["pair", "index", "parameter_label", "arm_a", "arm_b", "arm_a_path", "arm_b_path", "arm_a_value", "arm_b_value", "delta_b_minus_a", "abs_delta", "relative_delta", "classification_primary", "classification_reason"]
        with diff_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in diff_rows:
                writer.writerow({k: clean(row.get(k)) for k in fields})

        h15_pair_key = "H15_canonical__vs__H15_goal_facing"
        h15_statement = pair_statements.get(h15_pair_key, {"matched_parameter_comparison_possible": False, "reason": "H15 pair not evaluated"})
        any_pair_possible = any(v.get("matched_parameter_comparison_possible") is True for v in pair_statements.values())
        all_known_hashes_found = all(any(h.get("sha256") == EXPECTED_OPT_P_HASHES[label] for h in hashes_by_label[label]) for label in labels)
        every_row_labelled = all(str(r.get("classification_primary")) in CLASSIFICATIONS for r in diff_rows)
        per_entry_values_available = all_values_available_for_required_pairs and all("hash_only" not in str(r.get("parameter_label", "")) for r in diff_rows)

        within_cell_statement = (
            "Within-cell objective reconstruction is not invalidated by cross-arm opt_p differences: each cell's reconstruction compares the solver objective and reconstructed objective within that arm's own recorded opt_x/opt_p context. The ledger only governs cross-arm causal attribution, especially any initialization-only basin claim."
        )
        cross_arm_statement = (
            "Cross-arm initialization-only basin attribution is admissible only after the same context, horizon, terminal mode, previous input, goal/reference, structural parameters, and obstacle/TVP forecast entries are shown identical, with only initialization/warm-start state intentionally varied."
        )

        result = {
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "task_id": TASK_ID,
            "classification": "development_IMPROVED_zero_solve_T_C3_matched_parameter_ledger_not_validation_not_test",
            "input_hashes": {rel(V34U_RAW): sha256(V34U_RAW), rel(V34U_COMPLETED): sha256(V34U_COMPLETED)},
            "arm_labels": labels,
            "expected_opt_p_hashes": EXPECTED_OPT_P_HASHES,
            "all_known_hashes_found": all_known_hashes_found,
            "hashes_by_label": hashes_by_label,
            "vector_sources_by_label": {label: {k: v for k, v in vector_by_label[label].items() if k != "values"} for label in labels},
            "diff_count_rows": len(diff_rows),
            "classification_counts": {cls: sum(1 for r in diff_rows if r.get("classification_primary") == cls) for cls in sorted(CLASSIFICATIONS)},
            "pair_statements": pair_statements,
            "h15_canonical_vs_goal_facing_matched_parameter_comparison_possible": h15_statement.get("matched_parameter_comparison_possible") is True,
            "h15_canonical_vs_goal_facing_statement": h15_statement,
            "any_pair_among_four_matched_parameter_comparison_possible": any_pair_possible,
            "matched_parameter_comparison_possible_statement": {
                "H15_canonical_vs_goal_facing": h15_statement,
                "any_pair_among_four": any_pair_possible,
            },
            "within_cell_validity_statement": within_cell_statement,
            "cross_arm_attribution_limit_statement": cross_arm_statement,
            "data_sufficiency_for_per_entry_ledger": per_entry_values_available,
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
        }
        raw_path = run_dir / "raw.json"
        write_json(raw_path, result)
        summary_path = run_dir / "summary.md"
        summary_path.write_text(
            "# T-C3 matched-parameter ledger\n\n"
            + "UTC: `%s`. Structured zero-solve task `%s`.\n\n" % (created.isoformat(), TASK_ID)
            + "## Result\n"
            + "- Known opt_p hashes found for all four arms: `%s`.\n" % all_known_hashes_found
            + "- Full numeric opt_p vectors available for per-entry comparison: `%s`.\n" % per_entry_values_available
            + "- Diff/classification rows written: `%d`; classification counts: `%s`.\n" % (len(diff_rows), result["classification_counts"])
            + "- H15 canonical vs goal_facing matched-parameter comparison possible: `%s`.\n" % result["h15_canonical_vs_goal_facing_matched_parameter_comparison_possible"]
            + "- Any pair among four matched-parameter comparison possible: `%s`.\n\n" % any_pair_possible
            + "## Interpretation boundary\n"
            + within_cell_statement + "\n\n"
            + cross_arm_statement + "\n\n"
            + "If full opt_p vectors are absent, this task intentionally fails the automatic gate and returns the insufficiency to the lead rather than fabricating a labelled per-entry attribution.\n\n"
            + "Budget/access: solver=0, plant=0, training=0, validation64=0, sealed/final test=0.\n\n"
            + "Artifacts: `%s`, `%s`, `%s`.\n" % (rel(raw_path), rel(arm_summary_path), rel(diff_csv)),
            encoding="utf-8",
        )

        backup_request = ROOT / "research_artifacts/aws_backup_proofs" / ("REQUEST_BACKUP_AFTER_T_C3_MATCHED_PARAMETER_LEDGER_%s.json" % stamp)
        write_json(backup_request, {
            "request": "backup_after_t_c3_matched_parameter_ledger",
            "created_utc": created.isoformat(),
            "backup_required_before_more_unique_science": True,
            "must_cover": [rel(Path(__file__).resolve()), rel(run_dir), rel(backup_request), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
            "new_solver_calls": 0,
            "new_plant_steps": 0,
            "new_training_or_gradient_steps": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        })
        state_path = ROOT / "research_artifacts/aws_state" / ("continue_state_%s_after_t_c3_matched_parameter_ledger.md" % stamp)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        doc_block = """
<!-- {marker} -->
## T-C3 matched-parameter ledger

UTC: {created}. Structured zero-solve task `{task}`. Known opt_p hashes found for all four arms={hashes_found}. Full numeric opt_p vectors available={vectors_available}. H15 canonical vs goal_facing matched comparison possible={h15_possible}; any pair among four possible={any_possible}. Within-cell reconstruction remains separate from cross-arm causal attribution. Evidence: `{summary}`, `{raw}`, `{arm_csv}`, `{diff_csv}`. Backup request: `{backup}`. Solver=0, plant=0, training=0, validation64=0, sealed/final test=0.
""".format(
            marker=marker,
            created=created.isoformat(),
            task=TASK_ID,
            hashes_found=all_known_hashes_found,
            vectors_available=per_entry_values_available,
            h15_possible=result["h15_canonical_vs_goal_facing_matched_parameter_comparison_possible"],
            any_possible=any_pair_possible,
            summary=rel(summary_path),
            raw=rel(raw_path),
            arm_csv=rel(arm_summary_path),
            diff_csv=rel(diff_csv),
            backup=rel(backup_request),
        )
        for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
            append_if_missing(doc, marker, doc_block)
        state_path.write_text("# Continue state after T-C3 matched-parameter ledger\n\n" + doc_block + "\nIf task gates failed because full opt_p vectors are absent, return this actual insufficiency to Opus before T-C2.\n", encoding="utf-8")
        with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([created.isoformat(), NAME, result["classification"], "not_applicable_no_training_seed", "opened_development_artifacts_only_no_validation64_no_sealed_test", 0, 0, 0, 0, 0, False, rel(run_dir / "completed.json"), marker])

        pass_evidence = {
            "each_difference_labelled_nuisance_or_objective_relevant_or_structural": every_row_labelled,
            "explicit_statement_of_whether_matched_basin_comparison_is_possible": True,
            "no_solver_plant_training_validation_or_test_usage": True,
            "parameter_diff_table_written_with_paths_and_values": bool(diff_csv.exists() and arm_summary_path.exists() and per_entry_values_available),
            "within_cell_validity_stated_separately_from_cross_arm_attribution": True,
        }
        completed = {
            "status": "complete",
            "hard_pass": all(pass_evidence.values()),
            "created_utc": created.isoformat(),
            "classification": result["classification"],
            "task_id": TASK_ID,
            "summary": rel(summary_path),
            "raw": rel(raw_path),
            "arm_summary_csv": rel(arm_summary_path),
            "parameter_diff_table_csv": rel(diff_csv),
            "backup_request": rel(backup_request),
            "state": rel(state_path),
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "headline": {
                "all_known_hashes_found": all_known_hashes_found,
                "full_numeric_opt_p_vectors_available": per_entry_values_available,
                "diff_rows": len(diff_rows),
                "classification_counts": result["classification_counts"],
                "H15_canonical_vs_goal_facing_matched_parameter_comparison_possible": result["h15_canonical_vs_goal_facing_matched_parameter_comparison_possible"],
                "any_pair_among_four_matched_parameter_comparison_possible": any_pair_possible,
            },
            "pass_evidence": pass_evidence,
        }
        completed_path = run_dir / "completed.json"
        write_json(completed_path, completed)
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), pass_evidence)
        hash_paths = [Path(__file__).resolve(), raw_path, summary_path, arm_summary_path, diff_csv, completed_path, backup_request, state_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists()}
        write_json(completed_path, completed)
        print(json.dumps(clean({"completed": rel(completed_path), "summary": rel(summary_path), "headline": completed["headline"], "pass_evidence": pass_evidence, "backup_request": rel(backup_request)}), sort_keys=True), flush=True)
        return 0 if completed["hard_pass"] else 2
    except Exception as exc:
        return write_failure(run_dir, "unexpected T-C3 error: %s: %s" % (type(exc).__name__, exc), snapshot, "loader")


if __name__ == "__main__":
    raise SystemExit(main())
