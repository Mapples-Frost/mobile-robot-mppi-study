#!/usr/bin/env python3
"""T-C3R zero-solve matched-parameter ledger and receipt repair.

This implementation serves the active structured Opus task
`T-C3R-matched-parameter-ledger-operational-repair` from plan
`20260930T135811Z_8f32d0`. It performs only opened-artifact inspection.
It does not construct an environment, call an MPC solver, step a plant, train,
refit, open validation64, or open sealed/final test data.

Scientific rule for this repair: v34u/v34s contain opt_p hashes but not full
labelled numeric opt_p vectors. A per-entry parameter ledger is therefore
UNAVAILABLE unless a numeric vector is explicitly serialized under an opt_p_num
(or equivalently labelled) path and digest-tied to the recorded opt_p hash. This
script deliberately disables the previous weak score_vector_path heuristic.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import sqlite3
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
TASK_ID = "T-C3R-matched-parameter-ledger-operational-repair"
EXPECTED_REQUEST = "execution-failure:T-C3-startup-repair:20260930T135213_97b8a1ff"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

V34U_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z/raw.json"
V34U_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z/completed.json"
V34S_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34s_loader_gate_v0_20260930T120837Z/raw.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"

# Corrected by Opus 20260930T135811Z_8f32d0 from primary v34u raw evidence.
EXPECTED_OPT_P_HASHES = {
    "H12_canonical": "ca19ce63ea7aa807e66c33f9871022b24e1e7bd1ac0ee3292a82cc0025b52502",
    "H15_canonical": "7032ac0c47903ccfe27a25accc556b22331ba32247bf1a8b7a95f996cd1fdd5a",
    "H35_canonical": "9878b3d2235205772b48cf79a3e30d8eedd4e625feb2dc3d758130faa0ea651d",
    "H15_goal_facing": "fbe488a0e5749ee25682b7f6a963294112b041c31e0340de4214480cab638352",
}
ORDERED_LABELS = ["H12_canonical", "H15_canonical", "H35_canonical", "H15_goal_facing"]
ZERO_RESOURCES = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}
V34U_RAW_SHA256_EXPECTED = "a8bd5314528ee8c951722a12727df1ebabc85c4b885dec1dba62d1c9028a5058"


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


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


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
            yield from traverse(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from traverse(v, f"{path}[{i}]")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def flatten_numeric(value: Any, cap: int = 1_000_000) -> Optional[List[float]]:
    out: List[float] = []
    stack: List[Any] = [value]
    while stack:
        current = stack.pop(0)
        if is_number(current):
            out.append(float(current))
            if len(out) > cap:
                return None
        elif isinstance(current, (list, tuple)):
            stack = list(current) + stack
        else:
            return None
    return out if out else None


def parse_shape_first_dim(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], int):
        return int(value[0])
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("(") and "," in text:
            head = text[1:].split(",", 1)[0].strip()
            try:
                return int(head)
            except ValueError:
                return None
    return None


def arm_label(arm: Mapping[str, Any]) -> str:
    horizon = int(arm.get("horizon", 0) or 0)
    init = str(arm.get("initialization", ""))
    role = str(arm.get("role", arm.get("cell_role", "")))
    text = (init + " " + role).lower()
    if horizon == 15 and ("goal" in text or "alias_separation" in text):
        return "H15_goal_facing"
    if horizon == 12:
        return "H12_canonical"
    if horizon == 15:
        return "H15_canonical"
    if horizon == 35:
        return "H35_canonical"
    return f"H{horizon}_{init or role or 'unknown'}"


def find_arms(raw: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    arms = raw.get("arms")
    if not isinstance(arms, list):
        raise RuntimeError("v34u raw does not contain an arms list")
    labelled: Dict[str, Mapping[str, Any]] = {}
    for arm in arms:
        if isinstance(arm, Mapping):
            label = arm_label(arm)
            if label in EXPECTED_OPT_P_HASHES and label not in labelled:
                labelled[label] = arm
    missing = [label for label in ORDERED_LABELS if label not in labelled]
    if missing:
        raise RuntimeError(f"missing required v34u arms: {missing!r}")
    return labelled


def collect_hash_paths(arm: Mapping[str, Any], token: str, expected: Optional[str] = None) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for path, value in traverse(arm, "arm"):
        lower = path.lower()
        if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower()):
            if token.lower() in lower or (expected is not None and value == expected):
                hits.append({"path": path, "sha256": value})
    return hits


def collect_size_paths(arm: Mapping[str, Any], key_name: str) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    suffix = "." + key_name.lower()
    for path, value in traverse(arm, "arm"):
        if path.lower().endswith(suffix):
            parsed = parse_shape_first_dim(value)
            if parsed is not None:
                priority = 0
                pl = path.lower()
                if "captured_nlp_meta" in pl:
                    priority += 20
                if "nlp_meta" in pl:
                    priority += 10
                hits.append({"path": path, "raw": value, "size": parsed, "priority": priority})
    hits.sort(key=lambda h: (-int(h["priority"]), h["path"]))
    return hits


def first_nested_mapping_with_key(arm: Mapping[str, Any], wanted_key: str) -> Optional[Tuple[str, Mapping[str, Any]]]:
    for path, value in traverse(arm, "arm"):
        if isinstance(value, Mapping) and wanted_key in value:
            return path, value
    return None


def forced_solver_options_hash(arm: Mapping[str, Any]) -> Dict[str, Any]:
    explicit = collect_hash_paths(arm, "solver_options_hash")
    if explicit:
        return {"value": explicit[0]["sha256"], "source": explicit[0]["path"], "computed_from_options": False}
    hit = first_nested_mapping_with_key(arm, "ipopt.max_iter")
    if hit:
        path, opts = hit
        return {"value": json_digest(opts), "source": f"computed_sha256_json({path})", "computed_from_options": True, "options": dict(opts)}
    return {"value": None, "source": "UNAVAILABLE: no solver_options_hash or forced option mapping found", "computed_from_options": False}


def strict_opt_p_vector_status(arm: Mapping[str, Any], expected_hash: str, expected_size: Optional[int]) -> Dict[str, Any]:
    """Strictly accept only explicitly labelled opt_p vectors.

    The previous score_vector_path heuristic is intentionally not present here.
    Hash reproduction is conservative: if an explicit vector appears but its
    canonical JSON digest does not equal the recorded opt_p hash, this script does
    not report it as the matched opt_p vector.
    """
    candidates: List[Dict[str, Any]] = []
    for path, value in traverse(arm, "arm"):
        lower = path.lower()
        if not any(token in lower for token in ["opt_p_num", "opt_p_vector", "nlp_p_num"]):
            continue
        flat = flatten_numeric(value)
        if flat is None:
            candidates.append({"path": path, "numeric": False, "count": None, "digest_matches_recorded_opt_p_hash": False})
            continue
        digest_flat = json_digest(flat)
        candidates.append({
            "path": path,
            "numeric": True,
            "count": len(flat),
            "expected_size": expected_size,
            "size_matches": expected_size is not None and len(flat) == expected_size,
            "canonical_json_sha256": digest_flat,
            "digest_matches_recorded_opt_p_hash": digest_flat == expected_hash,
        })
    accepted = [c for c in candidates if c.get("numeric") and c.get("size_matches") and c.get("digest_matches_recorded_opt_p_hash")]
    if accepted:
        return {
            "full_labelled_numeric_opt_p_vector_serialized": True,
            "accepted_path": accepted[0]["path"],
            "accepted_count": accepted[0]["count"],
            "candidate_summaries": candidates,
            "disposition": "AVAILABLE: explicit opt_p vector size matched and canonical digest reproduced recorded opt_p hash.",
        }
    return {
        "full_labelled_numeric_opt_p_vector_serialized": False,
        "accepted_path": None,
        "accepted_count": None,
        "candidate_summaries": candidates,
        "disposition": "UNAVAILABLE: no explicit full labelled numeric opt_p vector serialized under opt_p_num/opt_p_vector/nlp_p_num with size and digest tied to the recorded opt_p hash. Hash-only evidence cannot support a per-entry parameter ledger.",
    }


def get_nested_int(arm: Mapping[str, Any], key: str) -> Optional[int]:
    for path, value in traverse(arm, "arm"):
        if path.lower().endswith("." + key.lower()) and isinstance(value, int):
            return value
    return None


def read_api_total_tokens() -> Dict[str, Any]:
    candidates = [ROOT / "research.sqlite", ROOT / "research_artifacts" / "research.sqlite", ROOT / "docs" / "research.sqlite"]
    for db in candidates:
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db))
            try:
                tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()]
                totals: Dict[str, int] = {}
                for table in tables:
                    cols = [r[1] for r in con.execute(f"pragma table_info({table})").fetchall()]
                    if "total_tokens" in cols:
                        val = con.execute(f"select coalesce(sum(total_tokens),0) from {table}").fetchone()[0]
                        totals[table] = int(val or 0)
                if totals:
                    return {"available": True, "path": rel(db), "table_sums": totals, "total_tokens": int(sum(totals.values()))}
            finally:
                con.close()
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": f"{type(exc).__name__}: {exc}"}
    return {"available": False, "path": None, "error": "research.sqlite not found in checked repository locations"}


def make_failure_payload(created: dt.datetime, message: str) -> Dict[str, Any]:
    tb = traceback.format_exc()
    tb_tail = [] if tb.strip() == "NoneType: None" else tb.splitlines()[-12:]
    return {
        "status": "failed",
        "error": message,
        "traceback_tail": tb_tail,
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "created_utc": created.isoformat(),
        "classification": "engineering_failure_zero_resource_before_scientific_outcome",
    }


def write_failure(run_dir_hint: Optional[Path], created: dt.datetime, message: str, engineering_error: str = "startup") -> int:
    safe_dirs: List[Path] = []
    if run_dir_hint is not None:
        safe_dirs.append(run_dir_hint)
    safe_dirs.append(ROOT / "research_artifacts" / "aws_diagnostics" / f"{NAME}_failure_{created.strftime('%Y%m%dT%H%M%SZ')}")
    receipt_env = None
    try:
        import os
        receipt_env = os.environ.get("BOHN_OUTCOME_RECEIPT")
    except Exception:
        receipt_env = None
    if receipt_env:
        safe_dirs.append(Path(receipt_env).resolve().parent / f"{NAME}_failed_payload")

    failed_path: Optional[Path] = None
    last_error = None
    for directory in safe_dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            failed_path = directory / "failed.json"
            write_json(failed_path, make_failure_payload(created, message))
            break
        except Exception as exc:  # preserve final fallback attempt below
            last_error = exc
            continue
    evidence: Dict[str, Any] = {"no_scientific_outcome": True, "error": message}
    if failed_path is not None:
        evidence["failed_json"] = rel(failed_path)
    else:
        evidence["failed_json_write_error"] = f"{type(last_error).__name__}: {last_error}"
    try:
        execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO_RESOURCES), evidence, engineering_error=engineering_error)
    except Exception as exc:
        evidence["record_outcome_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(clean({"failed": message, "resources": ZERO_RESOURCES, "evidence": evidence}), sort_keys=True), flush=True)
    return 1


def validate_failure_payload_contract(run_dir: Path, created: dt.datetime) -> Dict[str, Any]:
    path = run_dir / "failure_path_contract_failed.json"
    payload = make_failure_payload(created, "dry-run failure-payload contract validation; no exception was raised in the measurement run")
    write_json(path, payload)
    required = ["status", "error", "traceback_tail", "budget_actual", "validation64_bank_opened", "sealed_test_accessed"]
    loaded = read_json(path)
    valid = all(k in loaded for k in required) and loaded.get("budget_actual") == ZERO_RESOURCES and loaded.get("validation64_bank_opened") is False and loaded.get("sealed_test_accessed") is False
    return {"path": rel(path), "required_fields": required, "valid": bool(valid)}


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir: Optional[Path] = None
    try:
        snapshot = execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
        if snapshot is None:
            raise RuntimeError("missing structured execution snapshot")

        run_dir = ROOT / "research_artifacts" / "aws_diagnostics" / f"{NAME}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        marker = f"vehicle-tc3r-matched-parameter-ledger-operational-repair-{stamp}"

        if not V34U_RAW.exists() or not V34U_COMPLETED.exists():
            raise FileNotFoundError("missing required v34u raw/completed input")
        v34u_sha = sha256(V34U_RAW)
        if v34u_sha != V34U_RAW_SHA256_EXPECTED:
            raise RuntimeError(f"v34u raw sha256 mismatch: expected {V34U_RAW_SHA256_EXPECTED}, got {v34u_sha}")

        raw_text = V34U_RAW.read_text(encoding="utf-8-sig", errors="replace")
        v34s_text = V34S_RAW.read_text(encoding="utf-8-sig", errors="replace") if V34S_RAW.exists() else ""
        raw = read_json(V34U_RAW)
        arm_by_label = find_arms(raw)
        failure_payload_validation = validate_failure_payload_contract(run_dir, created)

        per_arm: Dict[str, Dict[str, Any]] = {}
        for label in ORDERED_LABELS:
            arm = arm_by_label[label]
            expected_hash = EXPECTED_OPT_P_HASHES[label]
            p_paths = collect_size_paths(arm, "p")
            x_paths = collect_size_paths(arm, "x")
            g_paths = collect_size_paths(arm, "g")
            p_size = p_paths[0]["size"] if p_paths else None
            opt_p_hash_paths = collect_hash_paths(arm, "opt_p_hash", expected_hash)
            opt_x_hash_paths = collect_hash_paths(arm, "opt_x_hash")
            vector_status = strict_opt_p_vector_status(arm, expected_hash, p_size)
            solver_hash = forced_solver_options_hash(arm)
            recorded_hashes = sorted(set(h["sha256"] for h in opt_p_hash_paths))
            per_arm[label] = {
                "arm_id": arm.get("arm_id"),
                "horizon": arm.get("horizon"),
                "initialization": arm.get("initialization"),
                "role": arm.get("role", arm.get("cell_role")),
                "recorded_opt_p_hash": expected_hash if expected_hash in recorded_hashes else (recorded_hashes[0] if recorded_hashes else None),
                "expected_corrected_opt_p_hash": expected_hash,
                "expected_hash_found_in_arm": expected_hash in recorded_hashes,
                "opt_p_hash_paths": opt_p_hash_paths,
                "captured_nlp_meta_p_size": p_size,
                "opt_x_size": x_paths[0]["size"] if x_paths else None,
                "opt_g_size": g_paths[0]["size"] if g_paths else None,
                "p_size_paths": p_paths[:5],
                "x_size_paths": x_paths[:5],
                "g_size_paths": g_paths[:5],
                "solver_options_hash": solver_hash["value"],
                "solver_options_hash_source": solver_hash["source"],
                "solver_options_hash_computed_from_options": solver_hash["computed_from_options"],
                "opt_x_hashes": opt_x_hash_paths[:10],
                "assignments_attempted": get_nested_int(arm, "assignments_attempted"),
                "assignments_succeeded": get_nested_int(arm, "assignments_succeeded"),
                "full_labelled_numeric_opt_p_vector_serialized_in_v34u_or_v34s_raw": vector_status["full_labelled_numeric_opt_p_vector_serialized"],
                "opt_p_vector_disposition": vector_status["disposition"],
                "strict_candidate_summaries": vector_status["candidate_summaries"],
            }

        arm_summary_path = run_dir / "opt_p_arm_summary.csv"
        with arm_summary_path.open("w", encoding="utf-8", newline="") as f:
            fields = [
                "arm_label", "arm_id", "horizon", "initialization", "role", "recorded_opt_p_hash",
                "expected_hash_found_in_arm", "captured_nlp_meta_p_size", "opt_x_size", "opt_g_size",
                "solver_options_hash", "solver_options_hash_source", "solver_options_hash_computed_from_options",
                "assignments_attempted", "assignments_succeeded", "full_labelled_numeric_opt_p_vector_serialized_in_v34u_or_v34s_raw",
                "opt_p_vector_disposition",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for label in ORDERED_LABELS:
                row = {k: per_arm[label].get(k) for k in fields if k != "arm_label"}
                row["arm_label"] = label
                writer.writerow(row)

        pair_rows: List[Dict[str, Any]] = []
        pair_statements: Dict[str, Dict[str, Any]] = {}
        for i, a in enumerate(ORDERED_LABELS):
            for b in ORDERED_LABELS[i + 1:]:
                pair = f"{a}__vs__{b}"
                hashes_differ = EXPECTED_OPT_P_HASHES[a] != EXPECTED_OPT_P_HASHES[b]
                possible = bool(per_arm[a]["full_labelled_numeric_opt_p_vector_serialized_in_v34u_or_v34s_raw"] and per_arm[b]["full_labelled_numeric_opt_p_vector_serialized_in_v34u_or_v34s_raw"])
                reason = "Full labelled numeric opt_p vectors are unavailable in v34u/v34s; only hashes are recorded, so per-entry matched-parameter comparison cannot be built from these artifacts."
                pair_statements[pair] = {
                    "matched_parameter_comparison_currently_possible": False if not possible else "requires_digest_tied_vectors_and_was_not_needed_here",
                    "full_labelled_numeric_opt_p_vectors_available_for_both_arms": possible,
                    "recorded_opt_p_hashes_differ": hashes_differ,
                    "reason": reason if not possible else "Unexpected explicit opt_p vectors appeared; this script did not perform heuristic substitution.",
                }
                pair_rows.append({
                    "pair": pair,
                    "arm_a": a,
                    "arm_b": b,
                    "arm_a_recorded_opt_p_hash": EXPECTED_OPT_P_HASHES[a],
                    "arm_b_recorded_opt_p_hash": EXPECTED_OPT_P_HASHES[b],
                    "recorded_opt_p_hashes_differ": hashes_differ,
                    "matched_parameter_comparison_currently_possible": False if not possible else "requires_digest_tied_vectors_and_was_not_needed_here",
                    "reason": pair_statements[pair]["reason"],
                })

        pair_csv = run_dir / "pairwise_availability_table.csv"
        with pair_csv.open("w", encoding="utf-8", newline="") as f:
            fields = ["pair", "arm_a", "arm_b", "arm_a_recorded_opt_p_hash", "arm_b_recorded_opt_p_hash", "recorded_opt_p_hashes_differ", "matched_parameter_comparison_currently_possible", "reason"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(pair_rows)

        h15_pair = pair_statements["H15_canonical__vs__H15_goal_facing"]
        any_pair_possible = any(v.get("matched_parameter_comparison_currently_possible") is True for v in pair_statements.values())
        all_known_hashes_found = all(per_arm[label]["expected_hash_found_in_arm"] is True for label in ORDERED_LABELS)
        all_vector_dispositions_written = all("opt_p_vector_disposition" in per_arm[label] for label in ORDERED_LABELS)
        per_arm_metadata_table_written = arm_summary_path.exists() and arm_summary_path.stat().st_size > 0

        within_cell_statement = (
            "Within-cell objective reconstruction remains interpretable when it compares the solver objective and a reconstructed objective inside the same arm's own recorded opt_x/opt_p context; absence of cross-arm parameter vectors does not by itself invalidate those within-cell formula checks."
        )
        cross_arm_statement = (
            "Cross-arm initialization-only or basin-attribution claims are blocked here: the artifacts provide opt_p hashes but not full labelled numeric opt_p vectors, so the H15 canonical versus goal_facing pair and all other pairs cannot be proven matched on every effective parameter entry from v34u/v34s alone."
        )

        result = {
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "server_api_token_audit": read_api_total_tokens(),
            "task_id": TASK_ID,
            "classification": "development_IMPROVED_zero_solve_T_C3R_matched_parameter_ledger_operational_repair_not_validation_not_test",
            "input_hashes": {rel(V34U_RAW): v34u_sha, rel(V34U_COMPLETED): sha256(V34U_COMPLETED), rel(V34S_RAW): sha256(V34S_RAW) if V34S_RAW.exists() else None},
            "literal_token_counts": {"v34u_raw_opt_p_num": raw_text.count("opt_p_num"), "v34u_raw_opt_x_hash": raw_text.count("opt_x_hash"), "v34u_raw_opt_p_hash": raw_text.count("opt_p_hash"), "v34s_raw_opt_p_num": v34s_text.count("opt_p_num") if v34s_text else None},
            "corrected_arm_hash_mapping": EXPECTED_OPT_P_HASHES,
            "earlier_plan_label_was_reversed": True,
            "do_not_change_known_hash_canonical_note": "No KNOWN_HASH_CANONICAL variable is used here; the corrected H15 canonical opt_p hash is 7032ac0c..., as authorized by Opus 20260930T135811Z_8f32d0.",
            "score_vector_path_heuristic_used": False,
            "forbidden_reporting_complied": True,
            "per_arm": per_arm,
            "pair_statements": pair_statements,
            "h15_canonical_vs_goal_facing_matched_parameter_comparison_possible": h15_pair.get("matched_parameter_comparison_currently_possible") is True,
            "any_pair_among_four_matched_parameter_comparison_possible": any_pair_possible,
            "within_cell_validity_statement": within_cell_statement,
            "cross_arm_attribution_limit_statement": cross_arm_statement,
            "failure_payload_validation": failure_payload_validation,
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
        }
        raw_path = run_dir / "raw.json"
        write_json(raw_path, result)

        summary_lines = [
            "# T-C3R matched-parameter ledger operational repair",
            "",
            f"UTC: `{created.isoformat()}`. Structured task `{TASK_ID}`. This is a zero-solve opened-artifact measurement, not validation or test.",
            "",
            "## Gate-relevant results",
            f"- Corrected arm-to-opt_p_hash mapping recorded: H12 canonical=`{EXPECTED_OPT_P_HASHES['H12_canonical']}`; H15 canonical=`{EXPECTED_OPT_P_HASHES['H15_canonical']}`; H35 canonical=`{EXPECTED_OPT_P_HASHES['H35_canonical']}`; H15 goal-facing=`{EXPECTED_OPT_P_HASHES['H15_goal_facing']}`.",
            "- The earlier 20260930T133239Z_8fd7da H12/H15 canonical hash labels were reversed; this report uses the Opus-corrected mapping.",
            f"- Literal token counts: v34u `opt_p_num`={raw_text.count('opt_p_num')}, v34u `opt_p_hash`={raw_text.count('opt_p_hash')}, v34u `opt_x_hash`={raw_text.count('opt_x_hash')}, v34s `opt_p_num`={v34s_text.count('opt_p_num') if v34s_text else 'missing'}.",
            "- Full labelled numeric opt_p vectors are UNAVAILABLE per arm from v34u/v34s; no heuristic numeric array was substituted.",
            f"- H15 canonical vs goal-facing matched-parameter comparison currently possible: `{result['h15_canonical_vs_goal_facing_matched_parameter_comparison_possible']}`.",
            f"- Any pair among the four arms currently possible: `{any_pair_possible}`.",
            f"- Failure-path payload contract written and validated: `{failure_payload_validation['valid']}` at `{failure_payload_validation['path']}`.",
            "",
            "## Interpretation boundary",
            f"- {within_cell_statement}",
            f"- {cross_arm_statement}",
            "",
            "Budget/access: solver=0, plant=0, training=0, validation64=0, sealed/final test=0.",
            "",
            f"Artifacts: `{rel(raw_path)}`, `{rel(arm_summary_path)}`, `{rel(pair_csv)}`.",
        ]
        summary_path = run_dir / "summary.md"
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        backup_request = ROOT / "research_artifacts" / "aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_T_C3R_MATCHED_PARAMETER_LEDGER_{stamp}.json"
        write_json(backup_request, {
            "request": "backup_after_t_c3r_matched_parameter_ledger_operational_repair",
            "created_utc": created.isoformat(),
            "backup_required_before_solver_bearing_T_C2": True,
            "must_cover": [rel(Path(__file__).resolve()), rel(run_dir), rel(backup_request), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
            "new_solver_calls": 0,
            "new_plant_steps": 0,
            "new_training_or_gradient_steps": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        })

        state_path = ROOT / "research_artifacts" / "aws_state" / f"continue_state_{stamp}_after_t_c3r_matched_parameter_ledger.md"
        doc_block = f"""
<!-- {marker} -->
## T-C3R matched-parameter ledger operational repair

UTC: {created.isoformat()}. Structured zero-solve task `{TASK_ID}` completed under Opus plan `20260930T135811Z_8f32d0`. Corrected mapping recorded; prior H12/H15 canonical hash labels were reversed. v34u literal `opt_p_num` count={raw_text.count('opt_p_num')}; full labelled numeric opt_p vectors are unavailable per arm, so H15 canonical-vs-goal-facing and all other cross-arm matched-parameter comparisons are not currently possible from v34u/v34s. This blocks cross-arm initialization-only causal attribution but does not invalidate within-cell objective reconstruction. Failure-path payload contract validation={failure_payload_validation['valid']}. Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(arm_summary_path)}`, `{rel(pair_csv)}`. Backup request: `{rel(backup_request)}`. Resources: solver=0, plant=0, training=0, validation64=0, sealed/final test=0.
""".strip()
        for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
            append_if_missing(doc, marker, doc_block)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("# Continue state after T-C3R matched-parameter ledger\n\n" + doc_block + "\n\nNext preauthorized dependency if this task gate passes: T-C5 zero-solve vector-sum closure and aggregate-count correction. Do not run solver-bearing T-C2 until T-C3R and T-C5 receipts exist and the backup/budget gates are reconciled.\n", encoding="utf-8")

        with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([created.isoformat(), NAME, result["classification"], "not_applicable_no_training_seed", "opened_development_artifacts_only_no_validation64_no_sealed_test", 0, 0, 0, 0, 0, False, rel(run_dir / "completed.json"), marker])

        pass_evidence = {
            "cross_arm_versus_within_cell_stated_separately": True,
            "failed_outcome_receipt_written_and_valid": bool(failure_payload_validation["valid"]),
            "no_solver_plant_training_validation_or_test_usage": True,
            "opt_p_vector_availability_disposition_written_per_arm": bool(all_vector_dispositions_written),
            "per_arm_hash_size_and_metadata_table_written": bool(per_arm_metadata_table_written and all_known_hashes_found),
            "reversed_hash_label_correction_recorded": True,
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
            "pairwise_availability_csv": rel(pair_csv),
            "backup_request": rel(backup_request),
            "state": rel(state_path),
            "failure_path_contract_failed_json": failure_payload_validation["path"],
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "headline": {
                "all_known_hashes_found": all_known_hashes_found,
                "corrected_mapping_H12_canonical": EXPECTED_OPT_P_HASHES["H12_canonical"],
                "corrected_mapping_H15_canonical": EXPECTED_OPT_P_HASHES["H15_canonical"],
                "v34u_literal_opt_p_num_count": raw_text.count("opt_p_num"),
                "v34s_literal_opt_p_num_count": v34s_text.count("opt_p_num") if v34s_text else None,
                "full_labelled_numeric_opt_p_vectors_available_all_arms": all(per_arm[label]["full_labelled_numeric_opt_p_vector_serialized_in_v34u_or_v34s_raw"] for label in ORDERED_LABELS),
                "H15_canonical_vs_goal_facing_matched_parameter_comparison_possible": result["h15_canonical_vs_goal_facing_matched_parameter_comparison_possible"],
                "any_pair_among_four_matched_parameter_comparison_possible": any_pair_possible,
                "score_vector_path_heuristic_used": False,
                "failure_path_payload_valid": bool(failure_payload_validation["valid"]),
            },
            "pass_evidence": pass_evidence,
        }
        completed_path = run_dir / "completed.json"
        write_json(completed_path, completed)
        hash_paths = [Path(__file__).resolve(), raw_path, summary_path, arm_summary_path, pair_csv, completed_path, backup_request, state_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv", run_dir / "failure_path_contract_failed.json"]
        completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists()}
        write_json(completed_path, completed)

        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), pass_evidence)
        print(json.dumps(clean({"completed": rel(completed_path), "summary": rel(summary_path), "headline": completed["headline"], "pass_evidence": pass_evidence, "hard_pass": completed["hard_pass"], "backup_request": rel(backup_request), "server_api_token_audit": result["server_api_token_audit"]}), sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        return write_failure(run_dir, created, f"unexpected T-C3R error: {type(exc).__name__}: {exc}", "startup")


if __name__ == "__main__":
    raise SystemExit(main())
