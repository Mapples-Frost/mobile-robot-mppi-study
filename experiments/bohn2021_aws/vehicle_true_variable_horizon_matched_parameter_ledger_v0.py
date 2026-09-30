#!/usr/bin/env python3
"""T-C3 zero-solve matched-parameter ledger for v34u H15 arms.

Lead task T-C3 asks whether the canonical and goal-facing H15 arms in the
source242/V15_shared diagnostic were solved on identical solver-entry parameters.
This script uses only already-opened development artifacts. It performs no solver
call, no plant rollout, no env.step/reset after construction, no training/refit, no
validation64 access, and no sealed/final-test access.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
try:
    import execution_contract  # type: ignore
except ImportError:
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "research_service"))
    import execution_contract  # type: ignore

NAME = "vehicle_true_variable_horizon_matched_parameter_ledger_v0"
TASK_ID = "T-C3-matched-parameter-ledger"
EXPECTED_REQUEST = "execution-result:20260930T131908_12b79445"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
V34U_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z/raw.json"
V34U_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z/completed.json"
SOURCE_GYM_CONTROLLER = ROOT / "sources/gym-horizon/gym_let_mpc/controllers.py"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
KNOWN_HASH_CANONICAL = "7032ac0c47903ccfe27a25accc556b22331ba32247bf1a8b7a95f996cd1fdd5a"
KNOWN_HASH_GOAL_FACING = "fbe488a0e5749ee25682b7f6a963294112b041c31e0340de4214480cab638352"
ZERO_RESOURCES = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}


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


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def flatten_numbers(value: Any, cap: int = 200000) -> Optional[List[float]]:
    out: List[float] = []
    stack = [value]
    while stack:
        x = stack.pop(0)
        if is_number(x):
            out.append(float(x))
            if len(out) > cap:
                return None
        elif isinstance(x, list):
            stack = list(x) + stack
        elif isinstance(x, tuple):
            stack = list(x) + stack
        else:
            return None
    return out if out else None


def traverse(obj: Any, path: str = "root") -> Iterable[Tuple[str, Any]]:
    yield path, obj
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from traverse(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from traverse(v, f"{path}[{i}]")


def find_arms(raw: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    arms = raw.get("arms")
    if not isinstance(arms, list):
        raise RuntimeError("v34u raw does not contain an arms list")
    canonical = None
    goal = None
    for arm in arms:
        if not isinstance(arm, Mapping):
            continue
        arm_id = str(arm.get("arm_id", ""))
        role = str(arm.get("role", ""))
        init = str(arm.get("initialization", ""))
        horizon = int(arm.get("horizon", 0) or 0)
        if horizon == 15 and "term=V15_shared" in arm_id and ("init=canonical" in arm_id or init == "canonical" or role == "canonical_H15"):
            canonical = arm
        if horizon == 15 and "term=V15_shared" in arm_id and ("init=goal_facing" in arm_id or init == "goal_facing" or "goal" in role):
            goal = arm
    if canonical is None or goal is None:
        raise RuntimeError("could not locate both H15 canonical and H15 goal_facing arms")
    return canonical, goal


def find_hashes(arm: Mapping[str, Any]) -> List[Dict[str, Any]]:
    hits = []
    for p, v in traverse(arm):
        if isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v.lower()):
            if "opt_p" in p.lower() or "hash" in p.lower() or v in (KNOWN_HASH_CANONICAL, KNOWN_HASH_GOAL_FACING):
                hits.append({"path": p, "sha256": v})
    return hits


def score_numeric_path(path: str, n: int) -> int:
    lower = path.lower()
    score = n
    for token, weight in [
        ("opt_p_num", 1000000), ("opt_p", 900000), ("p_num", 800000), ("parameter", 500000),
        ("solver", 300000), ("tvp", 200000), ("_p", 100000),
    ]:
        if token in lower:
            score += weight
    for bad in ["objective", "stage", "terminal", "cost", "time", "wall", "hash", "relative_error", "absolute_error", "candidate"]:
        if bad in lower:
            score -= 200000
    return score


def find_parameter_vector(arm: Mapping[str, Any]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for p, v in traverse(arm):
        if isinstance(v, (list, tuple)):
            flat = flatten_numbers(v)
            if flat is None or len(flat) < 10:
                continue
            lower = p.lower()
            if any(tok in lower for tok in ["opt_p", "p_num", "parameter", "solver", "tvp", "_p"]):
                candidates.append({"path": p, "count": len(flat), "score": score_numeric_path(p, len(flat)), "values": flat})
    if not candidates:
        return {"available": False, "reason": "no numeric parameter-vector candidate found under this arm"}
    candidates.sort(key=lambda x: (x["score"], x["count"]), reverse=True)
    chosen = candidates[0]
    return {
        "available": True,
        "path": chosen["path"],
        "count": chosen["count"],
        "score": chosen["score"],
        "values": chosen["values"],
        "candidate_summaries": [{"path": c["path"], "count": c["count"], "score": c["score"]} for c in candidates[:20]],
    }


def find_labels(arm: Mapping[str, Any], expected_len: int) -> List[str]:
    string_candidates: List[Tuple[str, List[str]]] = []
    for p, v in traverse(arm):
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            lower = p.lower()
            if any(tok in lower for tok in ["label", "name", "opt_p", "parameter", "p_num"]):
                string_candidates.append((p, list(v)))
    for p, labels in string_candidates:
        if len(labels) == expected_len:
            return labels
    return [f"opt_p[{i}]" for i in range(expected_len)]


def classify_label(label: str, delta: float) -> Tuple[str, str]:
    lower = label.lower()
    if any(tok in lower for tok in ["obstacle", "obj", "obs", "noise", "tvp", "forecast", "distance", "radius", "r_obstacle"]):
        return "nuisance", "obstacle/TVP/noise-related solver-entry value; it can affect constraints/objective and blocks pure initialization-basin attribution"
    if any(tok in lower for tok in ["reference", "goal", "trajectory", "target", "setpoint", "x_ref", "y_ref"]):
        return "objective-relevant", "reference/goal trajectory value affects objective or terminal target"
    if any(tok in lower for tok in ["previous_input", "u_prev", "rterm", "input", "control"]):
        return "objective-relevant", "previous/control input value can affect move penalty or warm-start objective terms"
    if any(tok in lower for tok in ["bound", "lower", "upper", "lb", "ub", "scale", "scaling", "mask"]):
        return "structural", "bounds/scaling/structural parameter"
    if abs(delta) > 0:
        return "objective-relevant", "unlabeled solver-entry numeric difference; conservatively objective/constraint relevant"
    return "nuisance", "no numerical difference"


def source_excerpt(path: Path, start: int = 858, end: int = 865) -> Dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": rel(path)}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for no in range(start, min(end, len(lines)) + 1):
        out.append({"line": no, "text": lines[no - 1]})
    return {"available": True, "path": rel(path), "sha256": sha256(path), "lines": out}


def write_failure(run_dir: Path, message: str, snapshot: Optional[Mapping[str, Any]], engineering_error: str = "loader") -> int:
    write_json(run_dir / "failed.json", {
        "status": "failed",
        "classification": "engineering_failure_zero_resource_before_scientific_outcome",
        "created_utc": now_utc().isoformat(),
        "error": message,
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
    })
    if snapshot is not None:
        execution_contract.record_outcome(
            ROOT,
            "engineering_failure",
            dict(ZERO_RESOURCES),
            {"no_scientific_outcome": True, "error": message, "failed_json": rel(run_dir / "failed.json")},
            engineering_error=engineering_error,
        )
    return 1


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = f"vehicle-tc3-matched-parameter-ledger-{stamp}"

    try:
        snapshot = execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
    except Exception as exc:
        return write_failure(run_dir, f"structured execution snapshot verification failed: {type(exc).__name__}: {exc}", None, engineering_error="authorization")
    if snapshot is None:
        return write_failure(run_dir, "missing structured execution snapshot", None, engineering_error="authorization")
    if not V34U_RAW.exists() or not V34U_COMPLETED.exists():
        return write_failure(run_dir, "missing v34u raw/completed input", snapshot, engineering_error="missing_file")

    try:
        raw = read_json(V34U_RAW)
        canonical, goal = find_arms(raw)
        can_hashes = find_hashes(canonical)
        goal_hashes = find_hashes(goal)
        can_vec = find_parameter_vector(canonical)
        goal_vec = find_parameter_vector(goal)
        can_hash_match = any(h["sha256"] == KNOWN_HASH_CANONICAL for h in can_hashes)
        goal_hash_match = any(h["sha256"] == KNOWN_HASH_GOAL_FACING for h in goal_hashes)

        diff_rows: List[Dict[str, Any]] = []
        exhaustive = False
        if can_vec.get("available") and goal_vec.get("available") and can_vec.get("count") == goal_vec.get("count"):
            exhaustive = True
            values_a = list(can_vec["values"])
            values_b = list(goal_vec["values"])
            labels = find_labels(canonical, len(values_a))
            for i, (a, b) in enumerate(zip(values_a, values_b)):
                delta = b - a
                if abs(delta) <= 1e-12:
                    continue
                label = labels[i] if i < len(labels) else f"opt_p[{i}]"
                cls, reason = classify_label(label, delta)
                diff_rows.append({
                    "index": i,
                    "parameter_label": label,
                    "canonical_value": a,
                    "goal_facing_value": b,
                    "delta_goal_minus_canonical": delta,
                    "abs_delta": abs(delta),
                    "relative_delta": abs(delta) / max(abs(a), abs(b), 1.0),
                    "classification_primary": cls,
                    "classification_reason": reason,
                    "canonical_path": can_vec.get("path"),
                    "goal_facing_path": goal_vec.get("path"),
                })

        classifications_ok = bool(exhaustive and all(r["classification_primary"] in {"nuisance", "objective-relevant", "structural"} for r in diff_rows)) or bool(exhaustive and not diff_rows)
        nuisance_present = any(r["classification_primary"] == "nuisance" for r in diff_rows)
        hashes_differ = bool(can_hash_match and goal_hash_match and KNOWN_HASH_CANONICAL != KNOWN_HASH_GOAL_FACING)
        matched_possible = bool(exhaustive and not diff_rows and not hashes_differ)
        if not exhaustive:
            matched_statement = "BLOCKED: parameter hashes differ or inputs are not exhaustively enumerable from the existing artifact; a matched-parameter basin claim is not admissible."
        elif diff_rows:
            matched_statement = "BLOCKED: canonical and goal_facing H15 solver-entry parameter vectors differ; between-arm behaviour cannot be attributed to initialization alone."
        else:
            matched_statement = "ADMISSIBLE_FROM_THIS_LEDGER: exhaustive parameter-vector enumeration found no numeric difference."

        csv_path = run_dir / "parameter_diff_table.csv"
        fields = ["index", "parameter_label", "canonical_value", "goal_facing_value", "delta_goal_minus_canonical", "abs_delta", "relative_delta", "classification_primary", "classification_reason", "canonical_path", "goal_facing_path"]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in diff_rows:
                writer.writerow(row)

        backup_request = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_T_C3_MATCHED_PARAMETER_LEDGER_{stamp}.json"
        state_path = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_t_c3_matched_parameter_ledger.md"
        result = {
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "task_id": TASK_ID,
            "classification": "development_IMPROVED_zero_solve_T_C3_matched_parameter_ledger_not_validation_not_test",
            "input_hashes": {rel(V34U_RAW): sha256(V34U_RAW), rel(V34U_COMPLETED): sha256(V34U_COMPLETED)},
            "arm_ids": {"canonical": canonical.get("arm_id"), "goal_facing": goal.get("arm_id")},
            "opt_p_hashes": {"canonical": can_hashes, "goal_facing": goal_hashes, "canonical_known_hash_found": can_hash_match, "goal_facing_known_hash_found": goal_hash_match, "hashes_differ": hashes_differ},
            "parameter_vector_sources": {"canonical": {k: v for k, v in can_vec.items() if k != "values"}, "goal_facing": {k: v for k, v in goal_vec.items() if k != "values"}},
            "exhaustive_numeric_parameter_vector_comparison": exhaustive,
            "diff_count": len(diff_rows),
            "classification_counts": {c: sum(1 for r in diff_rows if r["classification_primary"] == c) for c in ["nuisance", "objective-relevant", "structural"]},
            "matched_parameter_basin_comparison_possible": matched_possible,
            "matched_parameter_statement": matched_statement,
            "reset_noise_seed_source_excerpt": source_excerpt(SOURCE_GYM_CONTROLLER),
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
        }
        write_json(run_dir / "raw.json", result)
        write_json(backup_request, {
            "request": "backup_after_t_c3_matched_parameter_ledger",
            "created_utc": created.isoformat(),
            "backup_required_before_more_unique_science": True,
            "must_cover": [rel(Path(__file__).resolve()), rel(run_dir), rel(backup_request), rel(state_path), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
            "new_solver_calls": 0,
            "new_plant_steps": 0,
            "new_training_or_gradient_steps": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        })

        summary_path = run_dir / "summary.md"
        summary_path.write_text(
            "# T-C3 matched-parameter ledger\n\n"
            + f"UTC: `{created.isoformat()}`. Zero-solve existing-artifact diagnostic.\n\n"
            + "## Result\n"
            + f"- Known canonical opt_p hash found: `{can_hash_match}`; known goal-facing opt_p hash found: `{goal_hash_match}`; hashes differ: `{hashes_differ}`.\n"
            + f"- Exhaustive numeric parameter-vector comparison: `{exhaustive}`; differing numeric entries: `{len(diff_rows)}`.\n"
            + f"- Classification counts: `{result['classification_counts']}`.\n"
            + f"- Matched-parameter basin comparison possible: `{matched_possible}`.\n"
            + f"- Statement: {matched_statement}\n\n"
            + "Budget/access: solver=0, plant=0, training=0, validation64=0, sealed/final test=0.\n\n"
            + f"Diff CSV: `{rel(csv_path)}`. Raw: `{rel(run_dir/'raw.json')}`. Backup request: `{rel(backup_request)}`.\n",
            encoding="utf-8",
        )

        doc_block = f"""
<!-- {marker} -->
## T-C3 matched-parameter ledger

UTC: {created.isoformat()}. Zero-solve structured task `{TASK_ID}` over existing v34u artifacts. Solver=0, plant/env.step/reset=0, training/refit=0, validation64=0, sealed/final test=0. Known H15 opt_p hash matches: canonical={can_hash_match}, goal_facing={goal_hash_match}; hashes differ={hashes_differ}. Exhaustive numeric vector comparison={exhaustive}; diff_count={len(diff_rows)}; classification_counts={result['classification_counts']}. Matched-parameter basin comparison possible={matched_possible}. Statement: {matched_statement} Evidence: `{rel(summary_path)}`, `{rel(csv_path)}`, `{rel(run_dir/'raw.json')}`, `{rel(run_dir/'completed.json')}`. Backup request: `{rel(backup_request)}`.
"""
        for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
            append_if_missing(doc, marker, doc_block)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            "# Continue state after T-C3 matched-parameter ledger\n\n"
            + f"UTC: {created.isoformat()}\n\n{matched_statement}\n\n"
            + f"Artifacts: `{rel(summary_path)}`, `{rel(csv_path)}`, `{rel(run_dir/'raw.json')}`.\n",
            encoding="utf-8",
        )

        pass_evidence = {
            "each_difference_labelled_nuisance_or_objective_relevant_or_structural": classifications_ok,
            "explicit_statement_of_whether_matched_basin_comparison_is_possible": True,
            "no_solver_plant_training_validation_or_test_usage": True,
            "parameter_diff_table_written_with_paths_and_values": csv_path.exists() and (exhaustive or hashes_differ),
        }
        completed = {
            "status": "complete",
            "hard_pass": all(pass_evidence.values()),
            "created_utc": created.isoformat(),
            "classification": result["classification"],
            "task_id": TASK_ID,
            "summary": rel(summary_path),
            "raw": rel(run_dir / "raw.json"),
            "parameter_diff_table_csv": rel(csv_path),
            "backup_request": rel(backup_request),
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "headline": {
                "known_hashes_differ": hashes_differ,
                "diff_count": len(diff_rows),
                "exhaustive_numeric_parameter_vector_comparison": exhaustive,
                "matched_parameter_basin_comparison_possible": matched_possible,
                "classification_counts": result["classification_counts"],
            },
        }
        write_json(run_dir / "completed.json", completed)
        with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([created.isoformat(), NAME, result["classification"], "not_applicable_no_training_seed", "opened_development_artifacts_only_no_validation64_no_sealed_test", 0, 0, 0, 0, 0, False, rel(run_dir / "completed.json"), marker])
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), pass_evidence)
        hash_paths = [Path(__file__).resolve(), run_dir / "raw.json", summary_path, csv_path, run_dir / "completed.json", backup_request, state_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists()}
        write_json(run_dir / "completed.json", completed)
        print(json.dumps(clean({"completed": rel(run_dir / "completed.json"), "summary": rel(summary_path), "pass_evidence": pass_evidence, "headline": completed["headline"]}), sort_keys=True), flush=True)
        return 0 if completed["hard_pass"] else 2
    except Exception as exc:
        return write_failure(run_dir, f"unexpected T-C3 error: {type(exc).__name__}: {exc}", snapshot, engineering_error="loader")


if __name__ == "__main__":
    raise SystemExit(main())
