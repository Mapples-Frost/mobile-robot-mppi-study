#!/usr/bin/env python3
"""v15 protocol-aware case-snapshot recovery diagnostic.

The v15 full boundary-acquisition v0 failed before any MPC episode because its
case-snapshot helper looked only in run raw artifacts.  The first recovery
scan confirmed that v8c/v11 raw artifacts contain direct snapshots but that
fresh_v0/v1/v2 raw artifacts do not.  This metadata-only diagnostic checks the
corresponding frozen fresh-source protocols, which should contain the original
selected_fresh_cases with case_snapshot_from_candidate_pool.

No MPC simulation, no selector refit/search, no training, no validation64 bank
access and no sealed-test access.  It only determines whether the already
predeclared 24 v15 candidates can be recovered without case reselection.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_v15_protocol_case_snapshot_recovery_v0b"
STAMP = "20260929T2315Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T2315_after_v15_protocol_case_snapshot_recovery_v0b.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V15_PROTOCOL_CASE_SNAPSHOT_RECOVERY_V0B_{STAMP}.json"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-true-variable-H-v15-protocol-case-snapshot-recovery-v0b-{STAMP}"

AUDIT = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/audit.json"
AUDIT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/completed.json"
PREFLIGHT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_candidate_trace_preflight_v0b_20260929T2235Z/completed.json"
SMOKE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_smoke_v0_20260929T2245Z/completed.json"
PRIOR_RECOVERY_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_case_snapshot_recovery_diagnostic_v0_20260929T2305Z/completed.json"
PRIOR_RECOVERY_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_case_snapshot_recovery_diagnostic_v0_20260929T2305Z/raw.json"
FAILED_FULL = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0_20260929T2255Z/failure.json"

BANKS: Dict[str, Dict[str, Path]] = {
    "fresh_v0": {
        "raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/raw.json",
        "done": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/completed.json",
        "freeze_protocol": ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_frozen_20260929T1015Z.json",
    },
    "fresh_v1": {
        "raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/raw.json",
        "done": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/completed.json",
        "freeze_protocol": ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1_frozen_20260929T1055Z.json",
    },
    "fresh_v2": {
        "raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/raw.json",
        "done": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/completed.json",
        "freeze_protocol": ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_frozen_20260929T1120Z.json",
    },
    "fresh_v8c": {
        "raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/raw.json",
        "done": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/completed.json",
        "freeze_protocol": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8.py",
    },
    "fresh_v11": {
        "raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/raw.json",
        "done": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/completed.json",
        "freeze_protocol": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_boundary_acquisition_v11.py",
    },
}


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


def has_case(obj: Any) -> bool:
    return isinstance(obj, Mapping) and isinstance(obj.get("tvp"), Mapping) and bool(obj.get("tvp"))


def si(x: Any, default: int = -999999) -> int:
    try:
        return int(x)
    except Exception:
        return default


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


def source_index_from_trace(path: str) -> Optional[int]:
    m = re.search(r"source(\d+)", path)
    return int(m.group(1)) if m else None


def index_raw_cases(raw: Any) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}

    def consider(obj: Any, path: str) -> None:
        if not isinstance(obj, Mapping):
            return
        idx = si(obj.get("source_candidate_index"), -1)
        cs = obj.get("case_snapshot_from_candidate_pool")
        if idx >= 0 and has_case(cs):
            out.setdefault(idx, {"source_candidate_index": idx, "route": "raw_direct_case_snapshot", "path": path + ".case_snapshot_from_candidate_pool", "case_sha256": canonical_sha(cs), "top_keys": sorted(str(k) for k in cs.keys()), "tvp_key_count": len(cs.get("tvp", {}))})

    def rec(obj: Any, path: str) -> None:
        if isinstance(obj, Mapping):
            consider(obj, path)
            cc = obj.get("candidate_cases")
            if isinstance(cc, list):
                for i, c in enumerate(cc):
                    if has_case(c):
                        out.setdefault(i, {"source_candidate_index": i, "route": "raw_candidate_cases_index", "path": path + f".candidate_cases[{i}]", "case_sha256": canonical_sha(c), "top_keys": sorted(str(k) for k in c.keys()), "tvp_key_count": len(c.get("tvp", {}))})
            for k, v in obj.items():
                if isinstance(v, (Mapping, list)):
                    rec(v, path + "." + str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, (Mapping, list)):
                    rec(v, path + f"[{i}]")
    rec(raw, "$raw")
    return out


def index_protocol_cases(protocol: Any) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}

    def rec(obj: Any, path: str) -> None:
        if isinstance(obj, Mapping):
            idx = si(obj.get("source_candidate_index"), -1)
            cs = obj.get("case_snapshot_from_candidate_pool")
            if idx >= 0 and has_case(cs):
                out.setdefault(idx, {"source_candidate_index": idx, "route": "freeze_protocol_selected_fresh_case", "path": path + ".case_snapshot_from_candidate_pool", "case_sha256": canonical_sha(cs), "top_keys": sorted(str(k) for k in cs.keys()), "tvp_key_count": len(cs.get("tvp", {}))})
            # The fresh-source protocols also contain full candidate_pool snapshots under selected_fresh_cases.
            for k, v in obj.items():
                if isinstance(v, (Mapping, list)):
                    rec(v, path + "." + str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, (Mapping, list)):
                    rec(v, path + f"[{i}]")
    rec(protocol, "$protocol")
    return out


def append_docs(raw: Mapping[str, Any]) -> None:
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    h = raw["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-29 v15 protocol-aware case-snapshot recovery v0b

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Metadata-only recovery diagnostic after the v15 acquisition v0 pre-simulation failure and the raw-only recovery miss; no MPC simulation, no selector refit/search, no training, no validation64/sealed-test access. Candidate recovery: {h['recoverable_candidate_count']}/{h['candidate_count']} recoverable, missing={h['missing_candidate_count']}; routes={h['route_counts']}. Decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'completed.json')}`.
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
            f.write(f"{STAMP},{NAME},metadata_protocol_case_snapshot_recovery,opened_development_v15_candidates,0,0,0,0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# Vehicle true-variable-H v15 protocol-aware case-snapshot recovery v0b",
        "",
        f"UTC `{raw['created_utc']}`. Metadata-only; no MPC simulation, no selector refit/search, no training, no validation64, no sealed test.",
        "",
        "## Headline",
        "",
        f"- Candidate states checked: `{h['candidate_count']}` across banks `{h['banks']}`.",
        f"- Recoverable case/TVP snapshots: `{h['recoverable_candidate_count']}/{h['candidate_count']}`; missing `{h['missing_candidate_count']}`.",
        f"- Recovery routes: `{h['route_counts']}`.",
        f"- Failed v15 v0 cause: `{h['failed_attempt_message']}`.",
        f"- Decision: {raw['decision']}",
        "",
        "## Recovery by bank",
        "",
        "| bank | needed source ids | recovered | missing | routes |",
        "|---|---|---:|---:|---|",
    ]
    for bank, b in raw["banks"].items():
        lines.append(f"| `{bank}` | `{b['needed_source_indices']}` | {b['recoverable_candidate_count']} | {b['missing_candidate_count']} | `{b['route_counts']}` |")
    lines += ["", "## Candidate routes", "", "| idx | bank | source idx | role | source key | step | route | case sha256 |", "|---:|---|---:|---|---|---:|---|---|"]
    for c in raw["candidate_recovery"]:
        lines.append(f"| {c['candidate_index']} | `{c['bank_id']}` | {c['source_candidate_index']} | `{c['role']}` | `{c['source_key']}` | {c['candidate_branch_step']} | `{c.get('route')}` | `{c.get('case_sha256')}` |")
    if raw["missing_candidates"]:
        lines += ["", "## Missing candidates", "", "The following candidates remain unrecoverable and would require a reduced protocol or deterministic case reconstruction:"]
        for c in raw["missing_candidates"]:
            lines.append(f"- candidate {c['candidate_index']} `{c['bank_id']}` source {c['source_candidate_index']} `{c['source_key']}`")
    else:
        lines += ["", "No candidates are missing. A v15 acquisition v0b can preserve the predeclared 24 candidates and repair only the source-case lookup by falling back to frozen fresh-source protocols for fresh_v0/v1/v2."]
    lines += ["", "## Interpretation limits", "", "This is not performance evidence. It establishes that the earlier failure was an implementation recovery-path omission: fresh-source raw outputs omitted case snapshots, while their frozen protocols retain them. The next simulation runner must be versioned and must not reselect cases or alter acceptance gates."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    created = now()
    for label, path in (("v14 audit", AUDIT_DONE), ("v15 preflight", PREFLIGHT_DONE), ("v15 smoke", SMOKE_DONE), ("prior raw recovery", PRIOR_RECOVERY_DONE)):
        completed_dev_only(path, label)
    for bank, paths in BANKS.items():
        completed_dev_only(paths["done"], bank)
    audit = read_json(AUDIT)
    candidates = list(((audit.get("v15_candidate_plan") or {}).get("candidate_branch_states") or []))
    if len(candidates) != 24:
        raise ContractError(f"expected 24 candidates, got {len(candidates)}")
    write_json(PROTOCOL, {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_metadata_protocol_case_snapshot_recovery_no_simulation_no_validation_no_test",
        "hypothesis": "Fresh-source case snapshots missing from run raw artifacts are recoverable from the corresponding frozen fresh-source protocols, allowing v15 boundary acquisition to preserve its predeclared 24 candidates without reselection.",
        "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "inputs": {"audit": rel(AUDIT), "prior_recovery_raw": rel(PRIOR_RECOVERY_RAW), "failed_full_attempt": rel(FAILED_FULL)},
    })

    raw_indices: Dict[str, Dict[int, Dict[str, Any]]] = {}
    protocol_indices: Dict[str, Dict[int, Dict[str, Any]]] = {}
    needed_by_bank: Dict[str, List[int]] = defaultdict(list)
    for c in candidates:
        bank = str(c.get("bank_id"))
        src = source_index_from_trace(str(c.get("h15_trace_episode_path")))
        if src is None:
            src = si(c.get("source_candidate_index"), -1)
        needed_by_bank[bank].append(int(src))
    needed_by_bank = {k: sorted(set(v)) for k, v in needed_by_bank.items()}

    for bank, paths in BANKS.items():
        raw_indices[bank] = index_raw_cases(read_json(paths["raw"]))
        protocol_indices[bank] = {}
        p = paths.get("freeze_protocol")
        if p and p.exists() and p.suffix == ".json":
            protocol_indices[bank] = index_protocol_cases(read_json(p))

    candidate_recovery: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        bank = str(c.get("bank_id"))
        src = source_index_from_trace(str(c.get("h15_trace_episode_path")))
        if src is None:
            src = si(c.get("source_candidate_index"), -1)
        route_obj = raw_indices.get(bank, {}).get(src)
        if route_obj is None:
            route_obj = protocol_indices.get(bank, {}).get(src)
        candidate_recovery.append({
            "candidate_index": i,
            "bank_id": bank,
            "source_candidate_index": int(src),
            "role": c.get("role"),
            "source_key": c.get("source_key"),
            "base_state_id": c.get("base_state_id"),
            "candidate_branch_step": si(c.get("candidate_branch_step"), -1),
            "h15_trace_episode_path": c.get("h15_trace_episode_path"),
            "recoverable": route_obj is not None,
            "route": None if route_obj is None else route_obj.get("route"),
            "case_path": None if route_obj is None else route_obj.get("path"),
            "case_sha256": None if route_obj is None else route_obj.get("case_sha256"),
            "tvp_key_count": None if route_obj is None else route_obj.get("tvp_key_count"),
        })
    missing = [c for c in candidate_recovery if not c["recoverable"]]
    route_counts = Counter(str(c.get("route") or "missing") for c in candidate_recovery)
    banks: Dict[str, Any] = {}
    for bank, needed in needed_by_bank.items():
        rows = [c for c in candidate_recovery if c["bank_id"] == bank]
        banks[bank] = {
            "needed_source_indices": needed,
            "candidate_count": len(rows),
            "recoverable_candidate_count": sum(1 for r in rows if r["recoverable"]),
            "missing_candidate_count": sum(1 for r in rows if not r["recoverable"]),
            "route_counts": dict(Counter(str(r.get("route") or "missing") for r in rows)),
            "indexed_raw_source_count": len(raw_indices.get(bank, {})),
            "indexed_protocol_source_count": len(protocol_indices.get(bank, {})),
        }
    failed_msg = None
    if FAILED_FULL.exists():
        try:
            failed_msg = read_json(FAILED_FULL).get("message")
        except Exception:
            failed_msg = "failed marker unparsable"
    full_ok = len(missing) == 0
    decision = (
        "All 24 predeclared v15 candidates are recoverable without case reselection; write a v0b acquisition runner that falls back to frozen fresh-source protocols for fresh_v0/v1/v2 case snapshots, then run the 96-episode acquisition only after these recovery artifacts and source repair are externally backed up."
        if full_ok else
        "Some v15 candidates remain unrecoverable; do not run full acquisition. Freeze either deterministic case reconstruction or a reduced-candidate protocol with explicit limitations."
    )
    raw = {
        "created_utc": now().isoformat(),
        "method": NAME,
        "classification": "development_metadata_protocol_case_snapshot_recovery_no_simulation_no_validation_no_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "protocol": rel(PROTOCOL),
        "budget_actual": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "headline": {"candidate_count": len(candidates), "banks": sorted(needed_by_bank), "recoverable_candidate_count": sum(1 for c in candidate_recovery if c["recoverable"]), "missing_candidate_count": len(missing), "route_counts": dict(route_counts), "failed_attempt_message": failed_msg},
        "needed_by_bank": needed_by_bank,
        "banks": banks,
        "candidate_recovery": candidate_recovery,
        "missing_candidates": missing,
        "decision": decision,
        "input_hashes": {rel(p): sha256(p) for p in [SOURCE, AUDIT, AUDIT_DONE, PREFLIGHT_DONE, SMOKE_DONE, PRIOR_RECOVERY_DONE, PRIOR_RECOVERY_RAW, FAILED_FULL, PROTOCOL] + [v for d in BANKS.values() for v in d.values()] if p.exists()},
        "interpretation_limits": ["metadata-only", "opened development artifacts only", "not adaptive-H performance evidence", "no case reselection", "no validation64/test access"],
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    write_json(BACKUP_REQUEST, {"requested_utc": now().isoformat(), "reason": "backup v15 protocol-aware recovery diagnostic before source repair and before any v15 acquisition simulation", "backup_required_before_more_science": True, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(SOURCE), rel(PROTOCOL), rel(OUT), rel(STATE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"]})
    append_docs(raw)
    STATE.write_text(f"""# Continue state after v15 protocol-aware case-snapshot recovery v0b

UTC: {raw['created_utc']}
Elapsed since first supervisor event: {(now() - FIRST_EVENT).total_seconds()/3600.0:.2f} h.

Metadata-only diagnostic completed. No simulation, no selector refit/search, no training, no validation64, no sealed test.

Headline: {raw['headline']}
Decision: {decision}
Summary: `{rel(OUT / 'summary.md')}`
Raw: `{rel(OUT / 'raw.json')}`
Completed: `{rel(OUT / 'completed.json')}`
Backup request: `{rel(BACKUP_REQUEST)}`

Next action: external backup. If backup verifies, write/run `vehicle_true_variable_horizon_v15_boundary_acquisition_v0b.py` with robust case lookup: raw direct first, then frozen protocol selected_fresh_cases for fresh_v0/v1/v2. Preserve the existing 24-candidate plan and v0 acceptance gates; do not rerun v0 unchanged and do not open validation64/sealed test.
""", encoding="utf-8")
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, BACKUP_REQUEST]
    completed = {"passed": True, "hard_pass": True, "created_utc": raw["created_utc"], "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "budget_actual": raw["budget_actual"], "headline": raw["headline"], "decision": decision, "summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQUEST), "hashes": {rel(p): sha256(p) for p in sorted(set(files), key=lambda q: rel(q)) if p.exists()}}
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": raw["headline"], "decision": decision, "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(BACKUP_REQUEST)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {"failed_utc": now().isoformat(), "error": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0}})
        raise
