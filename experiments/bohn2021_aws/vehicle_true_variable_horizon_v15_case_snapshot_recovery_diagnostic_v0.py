#!/usr/bin/env python3
"""Metadata-only v15 case-snapshot recovery diagnostic.

The first full v15 boundary-acquisition attempt failed before any MPC episode
because fresh_v1 source_candidate_index=171 could not be resolved to a
case_snapshot_from_candidate_pool by the narrow smoke helper.  This diagnostic
uses only opened development artifacts and source text to determine whether all
24 predeclared v15 candidate branch states have recoverable case/TVP snapshots,
and if so which repair route should be used in a versioned acquisition runner.

No MPC simulation, no selector refit/search, no training, no validation64 bank
access, and no sealed-test access.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_v15_case_snapshot_recovery_diagnostic_v0"
STAMP = "20260929T2305Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T2305_after_v15_case_snapshot_recovery_diagnostic_v0.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V15_CASE_SNAPSHOT_RECOVERY_DIAGNOSTIC_V0_{STAMP}.json"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-true-variable-H-v15-case-snapshot-recovery-diagnostic-v0-{STAMP}"

AUDIT = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/audit.json"
AUDIT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/completed.json"
PREFLIGHT = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_candidate_trace_preflight_v0b_20260929T2235Z/preflight.json"
PREFLIGHT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_candidate_trace_preflight_v0b_20260929T2235Z/completed.json"
SMOKE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_smoke_v0_20260929T2245Z/completed.json"
FAILED_FULL = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0_20260929T2255Z/failure.json"

BANKS = {
    "fresh_v0": {
        "raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/raw.json",
        "done": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/completed.json",
        "manifest": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/selected_state_manifest.json",
        "runner": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_v0_runner.py",
        "freeze": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0.py",
    },
    "fresh_v1": {
        "raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/raw.json",
        "done": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/completed.json",
        "manifest": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/selected_state_manifest.json",
        "runner": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_v1_runner.py",
        "freeze": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1.py",
    },
    "fresh_v2": {
        "raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/raw.json",
        "done": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/completed.json",
        "manifest": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/selected_state_manifest.json",
        "runner": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_v2_runner.py",
        "freeze": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2.py",
    },
    "fresh_v8c": {
        "raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/raw.json",
        "done": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/completed.json",
        "manifest": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/selected_state_manifest.json",
        "runner": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count.py",
        "freeze": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8.py",
    },
    "fresh_v11": {
        "raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/raw.json",
        "done": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/completed.json",
        "manifest": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/selected_state_manifest.json",
        "runner": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_boundary_acquisition_v11.py",
        "freeze": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8.py",
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


def si(x: Any, default: int = -999999) -> int:
    try:
        return int(x)
    except Exception:
        return default


def extract_source_idx(trace_path: str) -> Optional[int]:
    m = re.search(r"source(\d+)", trace_path)
    return int(m.group(1)) if m else None


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def has_tvp_case(obj: Any) -> bool:
    return isinstance(obj, Mapping) and isinstance(obj.get("tvp"), Mapping) and bool(obj.get("tvp"))


def scan_raw(raw: Any, needed: Sequence[int]) -> Dict[str, Any]:
    needed_set = set(int(x) for x in needed)
    direct: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    candidate_case_lists: List[Dict[str, Any]] = []
    metadata_lists: List[Dict[str, Any]] = []
    source_index_mentions: Dict[int, int] = defaultdict(int)
    paths_with_case_snapshot_key = 0

    def rec(obj: Any, path: str, depth: int = 0) -> None:
        nonlocal paths_with_case_snapshot_key
        if isinstance(obj, Mapping):
            if "source_candidate_index" in obj:
                idx = si(obj.get("source_candidate_index"))
                if idx in needed_set:
                    source_index_mentions[idx] += 1
                    cs = obj.get("case_snapshot_from_candidate_pool")
                    if has_tvp_case(cs):
                        direct[idx].append({"path": path + ".case_snapshot_from_candidate_pool", "sha256": canonical_sha(cs), "tvp_keys": sorted(str(k) for k in cs.get("tvp", {}).keys()), "top_keys": sorted(str(k) for k in cs.keys())[:20]})
            if "case_snapshot_from_candidate_pool" in obj:
                paths_with_case_snapshot_key += 1
            cc = obj.get("candidate_cases")
            if isinstance(cc, list):
                candidate_case_lists.append({"path": path + ".candidate_cases", "length": len(cc), "tvp_case_count": sum(1 for c in cc if has_tvp_case(c)), "needed_indices_in_range": [idx for idx in sorted(needed_set) if 0 <= idx < len(cc)], "needed_indices_with_tvp": [idx for idx in sorted(needed_set) if 0 <= idx < len(cc) and has_tvp_case(cc[idx])], "needed_index_shas": {str(idx): canonical_sha(cc[idx]) for idx in sorted(needed_set) if 0 <= idx < len(cc) and has_tvp_case(cc[idx])}})
            meta = obj.get("all_candidate_metadata")
            if isinstance(meta, list):
                metadata_lists.append({"path": path + ".all_candidate_metadata", "length": len(meta), "needed_candidate_indices_present": [idx for idx in sorted(needed_set) if any(si(m.get("candidate_index")) == idx for m in meta if isinstance(m, Mapping))]})
            # Keep scan bounded but complete for JSON tree; path strings are compact only.
            for k, v in obj.items():
                if isinstance(v, (Mapping, list)):
                    rec(v, path + "." + str(k), depth + 1)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, (Mapping, list)):
                    rec(v, path + f"[{i}]", depth + 1)
    rec(raw, "$")
    return {
        "direct_case_snapshots_by_source": {str(k): v for k, v in sorted(direct.items())},
        "candidate_case_lists": candidate_case_lists,
        "metadata_lists": metadata_lists,
        "source_index_mentions": {str(k): int(v) for k, v in sorted(source_index_mentions.items())},
        "paths_with_case_snapshot_key": paths_with_case_snapshot_key,
    }


def scan_manifest(path: Path, needed: Sequence[int]) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    obj = read_json(path)
    selected = obj.get("selected_states") or obj.get("selected_cases") or []
    rows = []
    for r in selected if isinstance(selected, list) else []:
        if not isinstance(r, Mapping):
            continue
        idx = si(r.get("source_candidate_index"))
        if idx in set(needed):
            rows.append({"source_candidate_index": idx, "base_state_id": r.get("base_state_id"), "has_case_snapshot": has_tvp_case(r.get("case_snapshot_from_candidate_pool")), "has_tvp_key": isinstance((r.get("case_snapshot_from_candidate_pool") or {}).get("tvp"), Mapping), "trace": r.get("h15_trace_episode_path"), "branch_step": r.get("branch_step")})
    return {"exists": True, "path": rel(path), "selected_rows_for_needed_sources": rows, "sha256": sha256(path)}


def scan_source_text(paths: Sequence[Path]) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {}
    for p in paths:
        if not p.exists():
            evidence[rel(p)] = {"exists": False}
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        strings = sorted(set(s for s in re.findall(r"research_artifacts/[^\"'\s)]+", txt) if "validation64" not in s.lower() and "sealed" not in s.lower()))[:60]
        constants = []
        for line in txt.splitlines():
            if any(tok in line for tok in ("STAGE1", "BANK", "candidate_cases", "selected_cases", "choose_cases", "case_snapshot")):
                if "validation64" not in line.lower() and "sealed" not in line.lower():
                    constants.append(line.strip()[:220])
        evidence[rel(p)] = {"exists": True, "sha256": sha256(p), "artifact_string_samples": strings, "relevant_line_samples": constants[:120]}
    return evidence


def append_docs(raw: Mapping[str, Any]) -> None:
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    h = raw["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-29 v15 case-snapshot recovery diagnostic v0

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Metadata-only diagnostic after the v15 full-acquisition preparation failure; no MPC simulation, no selector refit/search, no training, no validation64/sealed-test access. Required candidates={h['candidate_count']}; direct/raw case recovery full={h['full_case_recovery_possible']}; recoverable candidates={h['recoverable_candidate_count']}/{h['candidate_count']}; missing candidates={h['missing_candidate_count']}. Decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'completed.json')}`.
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
            f.write(f"{STAMP},{NAME},metadata_case_snapshot_recovery_preflight,opened_development_v15_candidates,0,0,0,0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# Vehicle true-variable-H v15 case-snapshot recovery diagnostic v0",
        "",
        f"UTC `{raw['created_utc']}`. Metadata-only diagnostic; no MPC simulation, no selector refit/search, no training, no validation64, no sealed test.",
        "",
        "## Headline",
        "",
        f"- Candidate states checked: `{h['candidate_count']}` across banks `{h['banks']}`.",
        f"- Recoverable case/TVP snapshots: `{h['recoverable_candidate_count']}/{h['candidate_count']}`; missing `{h['missing_candidate_count']}`.",
        f"- Full recovery possible from already-opened artifacts: `{h['full_case_recovery_possible']}`.",
        f"- Failed v15 attempt cause confirmed: `{h['failed_attempt_message']}`.",
        f"- Decision: {raw['decision']}",
        "",
        "## Recovery by bank",
        "",
        "| bank | needed source ids | recoverable candidates | missing candidates | raw case routes | manifest matched rows |",
        "|---|---|---:|---:|---|---:|",
    ]
    for bank, b in raw["banks"].items():
        lines.append(f"| `{bank}` | `{b['needed_source_indices']}` | {b['recoverable_candidate_count']} | {b['missing_candidate_count']} | `{b['route_counts']}` | {b['manifest_match_count']} |")
    lines += [
        "",
        "## Missing candidates",
        "",
    ]
    missing = [c for c in raw["candidate_recovery"] if not c["recoverable"]]
    if missing:
        lines.append("| idx | bank | source idx | role | source key | trace | reason |")
        lines.append("|---:|---|---:|---|---|---|---|")
        for c in missing:
            lines.append(f"| {c['candidate_index']} | `{c['bank_id']}` | {c['source_candidate_index']} | `{c['role']}` | `{c['source_key']}` | `{c['h15_trace_episode_path']}` | `{c['reason']}` |")
    else:
        lines.append("No missing candidates; a v15 acquisition v0b can use the reported robust recovery routes.")
    lines += [
        "",
        "## Interpretation",
        "",
        "This diagnostic does not judge adaptive-H performance. It only determines whether the predeclared full boundary acquisition can be repaired without regenerating or selecting new cases. If recovery is incomplete, the next safe action is a versioned source-case recovery repair or a reduced acquisition using only recoverable candidates, not a blind rerun of the failed v0 script.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    created = now()
    completed_dev_only(AUDIT_DONE, "v14 false-positive audit")
    completed_dev_only(PREFLIGHT_DONE, "v15 candidate preflight")
    completed_dev_only(SMOKE_DONE, "v15 boundary-centre smoke")
    for bank, paths in BANKS.items():
        completed_dev_only(paths["done"], f"{bank} opened development bank")
    audit = read_json(AUDIT)
    candidates = list(((audit.get("v15_candidate_plan") or {}).get("candidate_branch_states") or []))
    if len(candidates) != 24:
        raise ContractError(f"expected 24 candidates, got {len(candidates)}")
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_metadata_case_snapshot_recovery_no_simulation_no_validation_no_test",
        "hypothesis": "The failed v15 full acquisition is likely a recovery-path implementation issue, not a scientific failure: fresh-source raw artifacts may encode case/TVP snapshots differently from v8c/v11. A metadata-only scan can identify whether all 24 predeclared candidates can be recovered from already-opened development artifacts before writing a v0b acquisition runner.",
        "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "inputs": {"audit": rel(AUDIT), "preflight": rel(PREFLIGHT), "smoke_done": rel(SMOKE_DONE), "failed_full_attempt": rel(FAILED_FULL)},
    }
    write_json(PROTOCOL, protocol)

    needed_by_bank: Dict[str, List[int]] = defaultdict(list)
    for c in candidates:
        bank = str(c.get("bank_id"))
        idx = extract_source_idx(str(c.get("h15_trace_episode_path")))
        if idx is None:
            idx = si(c.get("source_candidate_index"))
        needed_by_bank[bank].append(int(idx))
    needed_by_bank = {b: sorted(set(v)) for b, v in needed_by_bank.items()}

    scans: Dict[str, Any] = {}
    candidate_recovery: List[Dict[str, Any]] = []
    source_text_paths: List[Path] = []
    for bank, needed in needed_by_bank.items():
        paths = BANKS[bank]
        raw = read_json(paths["raw"])
        scan = scan_raw(raw, needed)
        manifest = scan_manifest(paths["manifest"], needed)
        scans[bank] = {"raw_scan": scan, "manifest_scan": manifest}
        source_text_paths.extend([paths["runner"], paths["freeze"]])
    source_text_scan = scan_source_text(sorted(set(source_text_paths + [SOURCE, ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v15_boundary_acquisition_v0.py"])))

    for i, c in enumerate(candidates):
        bank = str(c.get("bank_id"))
        trace = str(c.get("h15_trace_episode_path"))
        source_idx = extract_source_idx(trace)
        if source_idx is None:
            source_idx = si(c.get("source_candidate_index"))
        scan = scans[bank]["raw_scan"]
        route = None
        case_sha = None
        reason = "no_case_snapshot_or_candidate_cases_route_found"
        direct = scan["direct_case_snapshots_by_source"].get(str(source_idx)) or []
        if direct:
            route = "direct_case_snapshot_from_raw_object"
            case_sha = direct[0].get("sha256")
            reason = "direct case_snapshot_from_candidate_pool found in raw"
        else:
            for cc in scan.get("candidate_case_lists") or []:
                if source_idx in cc.get("needed_indices_with_tvp", []):
                    route = "raw_candidate_cases_index_lookup"
                    case_sha = (cc.get("needed_index_shas") or {}).get(str(source_idx))
                    reason = f"candidate_cases list {cc.get('path')} contains TVP case at source index"
                    break
        candidate_recovery.append({
            "candidate_index": i,
            "bank_id": bank,
            "source_candidate_index": source_idx,
            "role": c.get("role"),
            "source_key": c.get("source_key"),
            "base_state_id": c.get("base_state_id"),
            "candidate_branch_step": c.get("candidate_branch_step"),
            "h15_trace_episode_path": trace,
            "recoverable": route is not None,
            "route": route,
            "case_snapshot_sha256": case_sha,
            "reason": reason,
        })

    bank_summaries: Dict[str, Any] = {}
    for bank, needed in needed_by_bank.items():
        rows = [c for c in candidate_recovery if c["bank_id"] == bank]
        route_counts = Counter(c.get("route") or "missing" for c in rows)
        manifest_rows = scans[bank]["manifest_scan"].get("selected_rows_for_needed_sources") or []
        bank_summaries[bank] = {
            "needed_source_indices": needed,
            "candidate_count": len(rows),
            "recoverable_candidate_count": sum(1 for r in rows if r["recoverable"]),
            "missing_candidate_count": sum(1 for r in rows if not r["recoverable"]),
            "route_counts": dict(route_counts),
            "raw_scan_digest": scans[bank]["raw_scan"],
            "manifest_match_count": len(manifest_rows),
            "manifest_rows": manifest_rows,
        }

    failed_msg = None
    if FAILED_FULL.exists():
        try:
            failed_msg = read_json(FAILED_FULL).get("message")
        except Exception:
            failed_msg = "failed attempt marker exists but could not parse"
    recoverable = sum(1 for c in candidate_recovery if c["recoverable"])
    missing = len(candidate_recovery) - recoverable
    full_ok = missing == 0 and len(candidate_recovery) == len(candidates)
    if full_ok:
        decision = "All v15 candidates have recoverable case/TVP snapshots from already-opened artifacts; next write a v0b acquisition runner with robust case lookup and rerun the 96-episode boundary acquisition after backup."
    else:
        decision = "Not all v15 candidates have recoverable case/TVP snapshots via raw/manifest scan; do not rerun v0. Next inspect source runner case-bank construction and either repair deterministic source-case reconstruction or freeze a reduced recoverable-candidate acquisition."
    raw = {
        "created_utc": now().isoformat(),
        "method": NAME,
        "classification": "development_metadata_case_snapshot_recovery_no_simulation_no_validation_no_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "protocol": rel(PROTOCOL),
        "budget_actual": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "headline": {"candidate_count": len(candidates), "banks": sorted(needed_by_bank), "recoverable_candidate_count": recoverable, "missing_candidate_count": missing, "full_case_recovery_possible": full_ok, "failed_attempt_message": failed_msg},
        "needed_by_bank": needed_by_bank,
        "banks": bank_summaries,
        "candidate_recovery": candidate_recovery,
        "source_text_scan": source_text_scan,
        "decision": decision,
        "input_hashes": {rel(p): sha256(p) for p in [SOURCE, AUDIT, AUDIT_DONE, PREFLIGHT, PREFLIGHT_DONE, SMOKE_DONE, FAILED_FULL, PROTOCOL] if p.exists()},
        "interpretation_limits": ["metadata-only", "opened development artifacts only", "not adaptive-H performance evidence", "no case reselection or validation/test access"],
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    write_json(BACKUP_REQUEST, {"requested_utc": now().isoformat(), "reason": "backup v15 case-snapshot recovery diagnostic and failed v15 v0 marker before source repair", "backup_required_before_more_science": True, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(SOURCE), rel(PROTOCOL), rel(OUT), rel(STATE), rel(BACKUP_REQUEST), rel(FAILED_FULL), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"]})
    append_docs(raw)
    STATE.write_text(f"""# Continue state after v15 case-snapshot recovery diagnostic v0

UTC: {raw['created_utc']}
Elapsed since first supervisor event: {(now() - FIRST_EVENT).total_seconds()/3600.0:.2f} h.

Metadata-only diagnostic completed after the v15 acquisition v0 pre-simulation failure. No MPC simulation, no selector refit/search, no training, no validation64, no sealed test.

Headline: {raw['headline']}
Decision: {decision}
Summary: `{rel(OUT / 'summary.md')}`
Raw: `{rel(OUT / 'raw.json')}`
Completed: `{rel(OUT / 'completed.json')}`
Backup request: `{rel(BACKUP_REQUEST)}`

Next action: backup these artifacts. If full_case_recovery_possible is true, write a versioned v15 acquisition v0b using the reported robust case lookup. If false, inspect the source runner case-bank construction lines in raw.source_text_scan and freeze either deterministic source-case reconstruction or a reduced recoverable-candidate acquisition. Do not rerun v0 unchanged, do not open validation64/sealed test.
""", encoding="utf-8")
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, BACKUP_REQUEST, FAILED_FULL]
    done = {"passed": True, "hard_pass": True, "created_utc": raw["created_utc"], "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "budget_actual": raw["budget_actual"], "headline": raw["headline"], "decision": decision, "summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQUEST), "hashes": {rel(p): sha256(p) for p in sorted(set(files), key=lambda q: rel(q)) if p.exists()}}
    write_json(OUT / "completed.json", done)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": raw["headline"], "decision": decision, "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(BACKUP_REQUEST)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {"failed_utc": now().isoformat(), "error": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0}})
        raise
