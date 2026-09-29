#!/usr/bin/env python3
"""v15 local boundary acquisition v0b with protocol-aware case recovery.

This is a minimal versioned repair of
`vehicle_true_variable_horizon_v15_boundary_acquisition_v0.py`.  The v0 runner
failed before any MPC episode because fresh_v0/fresh_v1/fresh_v2 run raw outputs
do not carry `case_snapshot_from_candidate_pool` for selected source indices.
The metadata-only v15 protocol-aware recovery diagnostic v0b showed that the
same case/TVP snapshots are recoverable from the frozen fresh-source protocols.

This wrapper preserves the v0 24-candidate plan, schedule, budgets, terminal
profile, labels and acceptance rules.  It only patches case lookup to try:

1. v0 raw-direct lookup (unchanged path, used for fresh_v8c/fresh_v11), then
2. frozen fresh-source protocols for fresh_v0/fresh_v1/fresh_v2.

No validation64 bank access and no sealed-test access are permitted.  Running
this script performs development-only MPC continuations only when invoked with
`--run --i-accept-development-v15-boundary-acquisition` and a verified backup
commit argument.  It is IMPROVED development evidence, not validation/test.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_v15_boundary_acquisition_v0 as v0  # noqa:E402

NAME = "vehicle_true_variable_horizon_v15_boundary_acquisition_v0b"
STAMP = "20260929T2325Z"
SOURCE = Path(__file__).resolve()
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T2325_after_v15_boundary_acquisition_v0b.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V15_BOUNDARY_ACQUISITION_V0B_{STAMP}.json"
MARKER = f"vehicle-true-variable-H-v15-boundary-acquisition-v0b-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

BASE_V0_SOURCE = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v15_boundary_acquisition_v0.py"
RECOVERY_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_protocol_case_snapshot_recovery_v0b_20260929T2315Z/completed.json"
RECOVERY_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_protocol_case_snapshot_recovery_v0b_20260929T2315Z/raw.json"
RECOVERY_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_protocol_case_snapshot_recovery_v0b_20260929T2315Z/summary.md"

FRESH_PROTOCOL_BY_BANK: Dict[str, Path] = {
    "fresh_v0": ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_frozen_20260929T1015Z.json",
    "fresh_v1": ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1_frozen_20260929T1055Z.json",
    "fresh_v2": ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_frozen_20260929T1120Z.json",
}

_ORIGINAL_FIND_CASE = v0.smoke.find_case_snapshot
_ORIGINAL_BUILD_PROTOCOL = v0.build_protocol
CASE_LOOKUP_CACHE: Dict[str, Dict[int, Mapping[str, Any]]] = {}
CASE_LOOKUP_LOG: Dict[str, Dict[str, Any]] = {}
EXPECTED_SHA_BY_BANK_SOURCE: Dict[str, str] = {}


class ContractError(RuntimeError):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if hasattr(x, "item"):
        return clean(x.item())
    return x


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
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def si(x: Any, default: int = -999999) -> int:
    try:
        return int(x)
    except Exception:
        return default


def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def has_case(obj: Any) -> bool:
    return isinstance(obj, Mapping) and isinstance(obj.get("tvp"), Mapping) and bool(obj.get("tvp"))


def completed_dev_only(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing prerequisite: " + rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True and obj.get("status") not in ("complete", "completed"):
        raise ContractError("prerequisite not complete: " + rel(path))
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=True in {label}")
    return obj


def _index_protocol_cases(protocol: Any) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}

    def rec(obj: Any) -> None:
        if isinstance(obj, Mapping):
            idx = si(obj.get("source_candidate_index"), -1)
            cs = obj.get("case_snapshot_from_candidate_pool")
            if idx >= 0 and has_case(cs):
                out.setdefault(idx, cs)
            for v in obj.values():
                if isinstance(v, (Mapping, list)):
                    rec(v)
        elif isinstance(obj, list):
            for v in obj:
                if isinstance(v, (Mapping, list)):
                    rec(v)

    rec(protocol)
    return out


def _protocol_cases_for_bank(bank_id: str) -> Dict[int, Mapping[str, Any]]:
    if bank_id in CASE_LOOKUP_CACHE:
        return CASE_LOOKUP_CACHE[bank_id]
    path = FRESH_PROTOCOL_BY_BANK.get(bank_id)
    if path is None:
        CASE_LOOKUP_CACHE[bank_id] = {}
        return CASE_LOOKUP_CACHE[bank_id]
    if not path.exists():
        raise ContractError("missing frozen fresh-source protocol for " + bank_id + ": " + rel(path))
    cases = _index_protocol_cases(read_json(path))
    CASE_LOOKUP_CACHE[bank_id] = cases
    return cases


def _load_expected_case_hashes() -> None:
    EXPECTED_SHA_BY_BANK_SOURCE.clear()
    raw = read_json(RECOVERY_RAW)
    rows = list(raw.get("candidate_recovery") or [])
    if len(rows) != 24:
        raise ContractError("protocol recovery raw does not contain 24 candidate_recovery rows")
    for row in rows:
        if row.get("recoverable") is not True:
            raise ContractError("protocol recovery row is not recoverable: " + repr(row))
        key = f"{row.get('bank_id')}:{int(row.get('source_candidate_index'))}"
        sha = str(row.get("case_sha256"))
        old = EXPECTED_SHA_BY_BANK_SOURCE.get(key)
        if old is not None and old != sha:
            raise ContractError("conflicting expected case sha for " + key)
        EXPECTED_SHA_BY_BANK_SOURCE[key] = sha


def robust_find_case_snapshot(bank_id: str, source_idx: int, raw_by_bank: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """v0-compatible case lookup with a frozen-protocol fallback."""
    key = f"{bank_id}:{int(source_idx)}"
    original_error: Optional[BaseException] = None
    try:
        case = _ORIGINAL_FIND_CASE(bank_id, source_idx, raw_by_bank)
        route = "raw_direct_case_snapshot"
    except BaseException as exc:  # preserve v0 behavior unless a frozen protocol gives the exact case.
        original_error = exc
        cases = _protocol_cases_for_bank(bank_id)
        case = cases.get(int(source_idx))
        route = "freeze_protocol_selected_fresh_case"
        if case is None:
            raise original_error
    if not has_case(case):
        raise ContractError(f"case snapshot lacks tvp for {bank_id} source_candidate_index={source_idx}")
    observed_sha = canonical_sha(case)
    expected_sha = EXPECTED_SHA_BY_BANK_SOURCE.get(key)
    if expected_sha is not None and observed_sha != expected_sha:
        raise ContractError(f"case sha mismatch for {key}: observed {observed_sha} expected {expected_sha}")
    CASE_LOOKUP_LOG[key] = {
        "bank_id": bank_id,
        "source_candidate_index": int(source_idx),
        "route": route,
        "case_sha256": observed_sha,
        "expected_sha256": expected_sha,
        "tvp_key_count": len(case.get("tvp", {})),
    }
    return copy.deepcopy(case)


def check_recovery_prerequisites() -> Mapping[str, Any]:
    done = completed_dev_only(RECOVERY_DONE, "v15 protocol-aware recovery v0b")
    raw = read_json(RECOVERY_RAW)
    if done.get("headline", {}).get("recoverable_candidate_count") != 24:
        raise ContractError("recovery completed.json does not show 24 recoverable candidates")
    if done.get("headline", {}).get("missing_candidate_count") != 0:
        raise ContractError("recovery completed.json shows missing candidates")
    h = raw.get("headline") or {}
    if h.get("recoverable_candidate_count") != 24 or h.get("missing_candidate_count") != 0:
        raise ContractError("recovery raw headline does not show complete recovery")
    _load_expected_case_hashes()
    return done


def build_protocol_v0b(created: dt.datetime, prepared: Sequence[Mapping[str, Any]], backup_commit: str, input_hashes: Mapping[str, str]) -> Mapping[str, Any]:
    enriched_hashes = dict(input_hashes)
    for p in [SOURCE, BASE_V0_SOURCE, RECOVERY_DONE, RECOVERY_RAW, RECOVERY_SUMMARY] + list(FRESH_PROTOCOL_BY_BANK.values()):
        if p.exists():
            enriched_hashes[rel(p)] = sha256(p)
    proto = _ORIGINAL_BUILD_PROTOCOL(created, prepared, backup_commit, enriched_hashes)
    route_counts = Counter(v.get("route") for v in CASE_LOOKUP_LOG.values())
    proto["protocol_id"] = f"{NAME}_preoutcome_frozen_{STAMP}"
    proto["classification"] = "development_IMPROVED_true_variable_H_local_boundary_acquisition_v0b_protocol_case_recovery_preoutcome_not_validation_not_test"
    proto["source_repair"] = {
        "kind": "case_snapshot_lookup_only",
        "base_failed_runner": rel(BASE_V0_SOURCE),
        "failure_fixed": "fresh_v0/fresh_v1/fresh_v2 selected source case snapshots are loaded from frozen fresh-source protocols when run raw artifacts omit case_snapshot_from_candidate_pool",
        "no_candidate_reselection": True,
        "candidate_count_preserved": len(prepared),
        "case_lookup_route_counts_before_simulation": dict(route_counts),
        "case_lookup_log": CASE_LOOKUP_LOG,
        "recovery_evidence": {"completed": rel(RECOVERY_DONE), "raw": rel(RECOVERY_RAW), "summary": rel(RECOVERY_SUMMARY)},
    }
    proto["access_flags"] = {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False}
    write_json(PROTOCOL, proto)
    return proto


def write_summary_v0b(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    ag = a["aggregate"]
    lines = [
        "# Vehicle true-variable-H v15 local boundary acquisition v0b",
        "",
        f"UTC `{raw['created_utc']}`. Development-only 96-episode local boundary acquisition from opened H15-prefix states; no validation64, no sealed test, no selector refit/search, no gradient training.",
        "",
        "v0b changes only the case-snapshot recovery path: raw direct lookup first, then frozen fresh-source protocols for fresh_v0/fresh_v1/fresh_v2. The 24 candidates, H10/H15 x2 schedule, terminal profile, budget and decision gates are inherited from v0.",
        "",
        "## Budget/access",
        "",
        f"- Episodes/control steps: `{raw['budget_actual']['development_mpc_simulation_episodes']}` / `{raw['budget_declared']['development_mpc_simulation_episodes']}` episodes; `{raw['budget_actual']['development_control_steps']}` / `{raw['budget_declared']['development_control_step_upper_bound']}` control steps.",
        f"- validation64_bank_opened: `{raw['validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Aggregate",
        "",
        f"- Pair rows `{ag['pair_count']}` / `{ag['expected_pair_count']}`; all pairs present `{ag['all_pairs_present']}`; all H15 safe `{ag['all_h15_safe']}`.",
        f"- Catastrophic H10 rows `{ag['catastrophic_h10_rows']}`; beneficial H10 rows `{ag['beneficial_h10_rows']}`; label-data-sufficient-for-refit `{ag['label_data_sufficient_for_refit']}`.",
        f"- Fixed H10-vs-H15 decision saving over acquired pairs `{100.0 * sf(ag['fixed_H10_decision_relative_saving_vs_H15']):.2f}%`; solver saving `{100.0 * sf(ag['fixed_H10_solver_relative_saving_vs_H15']):.2f}%`; physical sums H10/H15 `{ag['fixed_H10_physical_sum']:.6g}` / `{ag['fixed_H15_physical_sum']:.6g}`.",
        f"- Role counts: `{ag['role_counts']}`.",
        f"- Decision: {a['decision']}",
        "",
        "## Case lookup repair evidence",
        "",
        f"- Recovery prerequisite: `{rel(RECOVERY_DONE)}`.",
        f"- Runtime lookup routes: `{dict(Counter(v.get('route') for v in CASE_LOOKUP_LOG.values()))}`.",
        "",
        "## Candidate-level label summary",
        "",
        "| idx | role | source | off | step | cat reps | ben reps | H15 safe | mean physΔ | mean decision gain s |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in a["candidate_rows"]:
        if r.get("missing"):
            lines.append(f"| {r.get('candidate_index')} | MISSING | `{r.get('candidate_id')}` | | | | | | | |")
            continue
        lines.append(
            "| %d | `%s` | `%s` | %d | %d | %d | %d | `%s` | %.6g | %.6g |" % (
                si(r.get("candidate_index"), 0), r.get("role"), r.get("source_key"), si(r.get("offset_from_center"), 0), si(r.get("branch_step"), 0),
                si(r.get("catastrophic_h10_repeats"), 0), si(r.get("beneficial_h10_repeats"), 0), r.get("all_h15_safe"),
                sf(r.get("mean_phys_delta_h10_minus_h15")), sf(r.get("mean_decision_gain_h10_vs_h15_s"))
            )
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is opened-development boundary-label acquisition, not validation or final-test evidence. It deliberately enriches local ambiguous cases already implicated by v14; it cannot by itself establish generalization. If used for refit, the method must be versioned as IMPROVED and confirmed on unused development sources before any validation64 rollout.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`; completed: `{rel(RUN_DIR / 'completed.json')}`; backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs_v0b(raw: Mapping[str, Any]) -> None:
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    ag = raw["analysis"]["aggregate"]
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v15 local boundary acquisition v0b

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only local boundary acquisition on 24 saved H15-prefix candidates x H10/H15 x2 repeats; no validation64/sealed-test access, no selector refit/search, no gradient training. v0b repairs only case-snapshot lookup using frozen fresh-source protocols where raw outputs omit selected-source snapshots. Budget actual: {raw['budget_actual']['development_mpc_simulation_episodes']} episodes, {raw['budget_actual']['development_control_steps']} control steps. Aggregate: pairs={ag['pair_count']}/{ag['expected_pair_count']}, all_h15_safe={ag['all_h15_safe']}, catastrophic_h10={ag['catastrophic_h10_rows']}, beneficial_h10={ag['beneficial_h10_rows']}, decision_saving_H10_vs_H15={ag['fixed_H10_decision_relative_saving_vs_H15']}, label_data_sufficient_for_refit={ag['label_data_sufficient_for_refit']}. Decision: {raw['analysis']['decision']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    tail = reg.read_text(encoding="utf-8", errors="replace")[-120000:] if reg.exists() else ""
    if MARKER not in tail:
        with reg.open("a", encoding="utf-8") as f:
            f.write(f"{STAMP},{NAME},development_v15_local_boundary_acquisition_v0b_protocol_case_recovery,opened_dev_false_positive_boundary_candidates,{raw['budget_actual']['development_mpc_simulation_episodes']},0,0,0,0,False,{rel(RUN_DIR / 'completed.json')},{MARKER}\n")


def patch_base_runner() -> None:
    v0.NAME = NAME
    v0.STAMP = STAMP
    v0.SOURCE = SOURCE
    v0.RUN_DIR = RUN_DIR
    v0.PROTOCOL = PROTOCOL
    v0.STATE = STATE
    v0.BACKUP_REQUEST = BACKUP_REQUEST
    v0.MARKER = MARKER
    v0.smoke.find_case_snapshot = robust_find_case_snapshot
    v0.build_protocol = build_protocol_v0b
    v0.write_summary = write_summary_v0b
    v0.append_docs = append_docs_v0b


def main(argv: Optional[Sequence[str]] = None) -> int:
    patch_base_runner()
    check_recovery_prerequisites()
    return int(v0.main(argv))


if __name__ == "__main__":
    patch_base_runner()
    try:
        check_recovery_prerequisites()
        raise SystemExit(v0.main())
    except Exception as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        failure = {
            "failed_utc": now().isoformat(),
            "error": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "budget_declared": {"development_mpc_simulation_episodes": v0.EXPECTED_EPISODES, "development_control_step_upper_bound": v0.CONTROL_STEP_CAP},
            "source_repair": "v0b protocol-aware case lookup",
            "next_recovery_hint": "Preserve partial outputs. Do not rerun v0 unchanged. If this v0b source fails before simulation, repair in a new versioned file and back up before running science.",
        }
        write_json(RUN_DIR / "failure.json", failure)
        raise
