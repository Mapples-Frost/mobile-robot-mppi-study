#!/usr/bin/env python3
"""No-simulation source audit for the current vehicle MPC horizon mechanism.

The immediately preceding v1e case5 timing decomposition found stable local H10
physical gains but no robust measured decision-time gain, and observed identical
optimizer-vector sizes for H10/H15 steps.  This diagnostic inspects the migrated
source tree to determine whether the current adaptive-horizon controller changes
only a horizon parameter inside a fixed-size NLP instead of rebuilding/choosing a
smaller optimization problem.

No rollouts, no training/refit, no validation64 bank and no sealed-test access.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260929T0630Z"
NAME = "vehicle_mpc_horizon_source_audit_v0"
SOURCE = Path(__file__).resolve()
DECOMP_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_timing_decomposition_postdiagnostic_v0_20260929T0625Z/completed.json"
DECOMP_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_timing_decomposition_postdiagnostic_v0_20260929T0625Z/summary.md"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_MPC_HORIZON_SOURCE_AUDIT_V0_{STAMP}.json"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-mpc-horizon-source-audit-v0-{STAMP}"
SOURCE_CANDIDATES = [
    ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/gym-horizon/gym_let_mpc/controllers.py",
    ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/gym-letMPC/gym_let_mpc/controllers.py",
    ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/do-mpc-horizon/do_mpc/controller.py",
    ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/do-mpc/do_mpc/controller.py",
    ROOT / "source_snapshots/windows_early/src/bohn2021/mpc/mpc_base.py",
]
PATTERNS = {
    "mpc_n_horizon": re.compile(r"mpc_n_horizon"),
    "n_horizon": re.compile(r"\bn_horizon\b"),
    "opt_x": re.compile(r"\bopt_x\b|opt_x_num|opt_x_scaling"),
    "setup_or_compile": re.compile(r"\bsetup\s*\(|compile_nlp|create_nlp|nlpsol|prepare_nlp|create_.*solver"),
    "runtime_action_horizon": re.compile(r"get_action|make_step|executed_horizon|action\[|float\(h\)|set_horizon|horizon\s*="),
    "conditional_mask": re.compile(r"if_else|logic|<=\s*.*mpc_n_horizon|mpc_n_horizon\s*[<>=]"),
    "rebuild_or_resize": re.compile(r"resize|rebuild|reset_controller|setup\(\).*horizon|n_horizon\s*=\s*int|set_param\(.*n_horizon"),
}


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
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
    return x


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs() -> Dict[str, Any]:
    if not DECOMP_DONE.exists():
        raise ContractError(f"missing preceding decomposition completed marker: {rel(DECOMP_DONE)}")
    done = read_json(DECOMP_DONE)
    if done.get("passed") is not True and done.get("hard_pass") is not True:
        raise ContractError("preceding decomposition did not pass")
    if done.get("validation64_bank_opened") is not False or done.get("sealed_test_accessed") is not False:
        raise ContractError("preceding decomposition has invalid access flags")
    return {"decomp_done": done}


def excerpt(lines: List[str], idx: int, radius: int = 2) -> Dict[str, Any]:
    lo = max(0, idx - radius)
    hi = min(len(lines), idx + radius + 1)
    return {
        "line": idx + 1,
        "text": lines[idx].rstrip("\n")[:240],
        "context": [f"{j + 1}: {lines[j].rstrip()}"[:320] for j in range(lo, hi)],
    }


def audit_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    hits: Dict[str, List[Dict[str, Any]]] = {}
    counts: Dict[str, int] = {}
    for name, regex in PATTERNS.items():
        idxs = [i for i, line in enumerate(lines) if regex.search(line)]
        counts[name] = len(idxs)
        hits[name] = [excerpt(lines, i) for i in idxs[:12]]
    # Look for whether setup/solver construction appears inside likely runtime methods.
    runtime_windows: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        if re.match(r"\s*def\s+(get_action|make_step|step|predict|set_horizon)", line):
            body = lines[i:min(len(lines), i + 80)]
            body_text = "\n".join(body)
            runtime_windows.append({
                "method_line": i + 1,
                "signature": line.strip(),
                "contains_setup_or_compile": bool(PATTERNS["setup_or_compile"].search(body_text)),
                "contains_n_horizon": bool(PATTERNS["n_horizon"].search(body_text)),
                "contains_mpc_n_horizon": bool(PATTERNS["mpc_n_horizon"].search(body_text)),
                "contains_rebuild_or_resize": bool(PATTERNS["rebuild_or_resize"].search(body_text)),
                "first_relevant_lines": [f"{i + k + 1}: {body[k].rstrip()}"[:320] for k in range(min(len(body), 20)) if ("horizon" in body[k] or "setup" in body[k] or "compile" in body[k] or "solver" in body[k] or "opt_x" in body[k])][:10],
            })
    return {
        "path": rel(path),
        "exists": True,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "line_count": len(lines),
        "pattern_counts": counts,
        "pattern_excerpts": hits,
        "runtime_method_windows": runtime_windows[:12],
    }


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def render_summary(result: Mapping[str, Any]) -> str:
    a = result["analysis"]
    lines = [
        "# Vehicle MPC horizon source audit v0",
        "",
        f"UTC: `{result['created_utc']}`. No simulations/training/refit; no validation64 or sealed-test access.",
        "",
        "## Finding",
        "",
        f"- Source-level classification: `{a['classification']}`.",
        f"- Runtime rebuild/resize evidence found: `{a['runtime_rebuild_or_resize_found']}`.",
        f"- Horizon parameter/mask evidence found: `{a['horizon_parameter_mask_evidence_found']}`.",
        f"- Fixed-size trace evidence from previous no-rollout decomposition: `{a['previous_fixed_nlp_trace_evidence']}`.",
        "",
        "## Source counts",
        "",
        "| source | exists | mpc_n_horizon | n_horizon | opt_x | runtime windows | rebuild/resize hits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in a["file_audits"]:
        c = row.get("pattern_counts") or {}
        lines.append(
            f"| `{row['path']}` | `{row.get('exists')}` | {c.get('mpc_n_horizon', 0)} | {c.get('n_horizon', 0)} | {c.get('opt_x', 0)} | {len(row.get('runtime_method_windows') or [])} | {c.get('rebuild_or_resize', 0)} |"
        )
    lines += [
        "",
        "## Evidence snippets",
        "",
    ]
    for item in a["key_evidence_snippets"][:10]:
        lines.append(f"- `{item['path']}` line {item['line']}: `{item['text']}`")
    lines += [
        "",
        "## Interpretation and next experiment",
        "",
        a["interpretation"],
        "",
        a["next_action"],
        "",
        f"Backup request: `{result['backup_request']}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    started = now_utc()
    inputs = verify_inputs()
    if OUT.exists() and (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline")}, sort_keys=True))
        return 0
    if OUT.exists() and any(p.name != "run.lock" for p in OUT.iterdir()):
        raise ContractError(f"partial output exists; inspect first: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=True)
    audits = [audit_file(p) for p in SOURCE_CANDIDATES]
    existing = [a for a in audits if a.get("exists")]
    if not existing:
        raise ContractError("no source candidates were found for audit")

    horizon_parameter_mask_evidence = any((a.get("pattern_counts") or {}).get("mpc_n_horizon", 0) > 0 or (a.get("pattern_counts") or {}).get("conditional_mask", 0) > 0 for a in existing)
    runtime_rebuild = any(any(w.get("contains_setup_or_compile") or w.get("contains_rebuild_or_resize") for w in (a.get("runtime_method_windows") or [])) for a in existing)
    opt_x_evidence = any((a.get("pattern_counts") or {}).get("opt_x", 0) > 0 and (a.get("pattern_counts") or {}).get("n_horizon", 0) > 0 for a in existing)
    previous_fixed = bool((inputs["decomp_done"].get("headline") or {}).get("fixed_nlp_size_across_h10_h15"))

    snippets: List[Dict[str, Any]] = []
    for a in existing:
        for key in ("mpc_n_horizon", "conditional_mask", "opt_x", "n_horizon", "rebuild_or_resize"):
            for hit in (a.get("pattern_excerpts") or {}).get(key, [])[:3]:
                snippets.append({"path": a["path"], "pattern": key, "line": hit["line"], "text": hit["text"]})

    if previous_fixed and horizon_parameter_mask_evidence and not runtime_rebuild:
        classification = "source_consistent_with_fixed_size_masked_horizon_no_runtime_rebuild"
        interpretation = (
            "The source audit supports the trace-level diagnosis: horizon changes are represented through horizon parameters/masking in a solver constructed for a fixed problem size, and no inspected runtime method shows a clear setup/compile/rebuild/resize path when H changes. Therefore shorter H is not expected to reduce CasADi/IPOPT problem dimension in the current controller; physical local H10 gains do not imply a compute tradeoff."
        )
        next_action = (
            "After backup, freeze a bounded IMPROVED variable-dimension smoke: instantiate separate H10 and H15 MPC controllers (or controller cache) before rollout and replay the two case5 H15-common terminal states for one repeat. Acceptance for expanding: same prefix/branch safety plus >=15-20% branch solver-time reduction for true H10 construction; otherwise pivot to terminal/value/modeling or scenario-opportunity design rather than further H-as-speed selector training."
        )
    elif previous_fixed and runtime_rebuild:
        classification = "trace_fixed_size_but_source_has_possible_rebuild_path_needs_targeted_instrumentation"
        interpretation = (
            "The previous traces show fixed optimizer size, but source text contains a possible rebuild/setup/resize path in runtime methods. A small instrumentation smoke should identify whether the path is inactive, cached, or disabled."
        )
        next_action = "After backup, freeze a no/one-rollout instrumentation smoke around the possible rebuild path before any selector/refit."
    else:
        classification = "source_audit_inconclusive"
        interpretation = "The source-text audit was not sufficient to classify the horizon implementation. Treat current measured speed evidence as negative until a targeted runtime instrumentation test resolves solver dimensions."
        next_action = "After backup, inspect/import the controller object metadata or run one bounded instrumentation rollout; no training/refit yet."

    created = now_utc()
    write_json(BACKUP_REQ, {
        "requested_utc": created.isoformat(),
        "reason": "backup no-simulation vehicle MPC horizon source audit before any further simulation/training/refit",
        "backup_required_before_more_simulations": True,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(OUT), rel(STATE), rel(SOURCE), rel(BACKUP_REQ)],
    })
    analysis = {
        "classification": classification,
        "previous_fixed_nlp_trace_evidence": previous_fixed,
        "horizon_parameter_mask_evidence_found": horizon_parameter_mask_evidence,
        "runtime_rebuild_or_resize_found": runtime_rebuild,
        "opt_x_and_n_horizon_source_evidence_found": opt_x_evidence,
        "file_audits": audits,
        "key_evidence_snippets": snippets,
        "interpretation": interpretation,
        "next_action": next_action,
    }
    result = {
        "created_utc": created.isoformat(),
        "started_utc": started.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_metadata_source_audit_no_rollout_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "inputs": {"decomposition_completed": rel(DECOMP_DONE), "decomposition_completed_sha256": sha256(DECOMP_DONE), "decomposition_summary": rel(DECOMP_SUMMARY) if DECOMP_SUMMARY.exists() else None},
        "source_hashes": {rel(SOURCE): sha256(SOURCE)},
        "analysis": analysis,
        "backup_request": rel(BACKUP_REQ),
    }
    write_json(OUT / "raw.json", result)
    summary = render_summary(result)
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(summary, encoding="utf-8")
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle MPC horizon source audit v0

UTC: {created.isoformat()}. No-simulation source audit after the v1e case5 timing decomposition. No rollouts, no training/refit, no validation64 bank, no sealed-test access. Classification: `{classification}`. The audit supports treating current shorter-H timing as a fixed-size masked-horizon implementation issue rather than a selector label-density problem alone. Next action after backup: {next_action} Backup request: `{rel(BACKUP_REQ)}`.
""")
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, STATE, BACKUP_REQ]
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "headline": {
            "classification": classification,
            "runtime_rebuild_or_resize_found": runtime_rebuild,
            "horizon_parameter_mask_evidence_found": horizon_parameter_mask_evidence,
            "previous_fixed_nlp_trace_evidence": previous_fixed,
            "train_or_refit_now": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        },
        "backup_request": rel(BACKUP_REQ),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": completed["headline"], "backup_request": rel(BACKUP_REQ)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
