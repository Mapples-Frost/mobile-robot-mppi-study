#!/usr/bin/env python3
"""v15 candidate trace preflight, repaired after v0 syntax failure.

Development-only bounded diagnostic.  It reads the completed v14 false-positive
neighbor audit and checks whether the proposed H15-prefix trace paths exist and
contain parseable sequence-like data covering the requested branch steps.  It
performs no MPC simulation, no selector refit/search, no gradient training, no
validation64 access, and no sealed-test access.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_v15_candidate_trace_preflight_v0b"
STAMP = "20260929T2235Z"
AUDIT = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/audit.json"
DONE_AUDIT = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/completed.json"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T2235_after_v15_candidate_trace_preflight_v0b.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-v15-candidate-trace-preflight-v0b-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")


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
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
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
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_dev_only(obj: Mapping[str, Any], label: str) -> None:
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=True in {label}")


def suffix_counts(files: Sequence[Path]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for p in files:
        s = p.suffix.lower() or "<no_suffix>"
        d[s] = d.get(s, 0) + 1
    return dict(sorted(d.items()))


def item_kind(item: Any) -> str:
    if isinstance(item, Mapping):
        keys = {str(k).lower() for k in item.keys()}
        if keys & {"state", "states", "observation", "obs", "x", "y", "theta"}:
            return "dict_state_like"
        if keys & {"step", "t", "time", "control", "action", "horizon"}:
            return "dict_step_like"
        return "dict"
    if isinstance(item, list):
        nums = sum(1 for v in item[:20] if isinstance(v, (int, float)))
        return f"numeric_list_{nums}" if nums else "list"
    if isinstance(item, (int, float)):
        return "number"
    return type(item).__name__


def inspect_json_obj(obj: Any, needed_max_step: int, source: str) -> List[Dict[str, Any]]:
    seqs: List[Dict[str, Any]] = []
    seen_ids = set()

    def rec(x: Any, path: str, depth: int) -> None:
        if depth > 7 or len(seqs) > 120:
            return
        xid = id(x)
        if xid in seen_ids:
            return
        seen_ids.add(xid)
        if isinstance(x, list):
            ln = len(x)
            first = None
            for v in x[: min(5, ln)]:
                if v is not None:
                    first = v
                    break
            if ln > needed_max_step:
                seqs.append({
                    "source": source,
                    "json_path": path,
                    "length": ln,
                    "covers_needed_max_step": True,
                    "item_kind": item_kind(first),
                    "sample_keys": sorted(list(first.keys()))[:20] if isinstance(first, Mapping) else None,
                })
            for i, v in enumerate(x[:3]):
                if isinstance(v, (dict, list)):
                    rec(v, f"{path}[{i}]", depth + 1)
        elif isinstance(x, Mapping):
            preferred = []
            rest = []
            for k, v in x.items():
                lk = str(k).lower()
                if any(tok in lk for tok in ("trace", "traj", "state", "obs", "step", "episode", "history", "record")):
                    preferred.append((k, v))
                else:
                    rest.append((k, v))
            for k, v in preferred + rest[:20]:
                if isinstance(v, (dict, list)):
                    rec(v, f"{path}.{k}", depth + 1)

    rec(obj, "$", 0)
    seqs.sort(key=lambda r: (0 if "state" in str(r.get("item_kind")) else 1, -int(r.get("length", 0)), str(r.get("source"))))
    return seqs[:20]


def inspect_csv_file(path: Path, needed_max_step: int) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = 0
            for count, _ in enumerate(reader, 1):
                if count > needed_max_step + 5:
                    break
        return {"source": rel(path), "header": header[:30], "rows_counted_lower_bound": count, "covers_needed_max_step": count > needed_max_step}
    except Exception as exc:
        return {"source": rel(path), "error": type(exc).__name__, "message": str(exc)[:200]}


def inspect_numpy_file(path: Path, needed_max_step: int) -> Dict[str, Any]:
    try:
        import numpy as np  # type: ignore
        if path.suffix.lower() == ".npz":
            out: Dict[str, Any] = {"source": rel(path), "arrays": []}
            with np.load(path, allow_pickle=False) as z:
                for k in z.files[:30]:
                    shape = tuple(int(v) for v in z[k].shape)
                    out["arrays"].append({"name": k, "shape": shape, "covers_needed_max_step": bool(shape and shape[0] > needed_max_step)})
            return out
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = tuple(int(v) for v in arr.shape)
        return {"source": rel(path), "shape": shape, "covers_needed_max_step": bool(shape and shape[0] > needed_max_step)}
    except Exception as exc:
        return {"source": rel(path), "error": type(exc).__name__, "message": str(exc)[:200]}


def inspect_trace(trace_rel: str, steps: Sequence[int]) -> Dict[str, Any]:
    p = ROOT / trace_rel
    needed = max(int(s) for s in steps) if steps else -1
    out: Dict[str, Any] = {
        "trace_path": trace_rel,
        "exists": p.exists(),
        "is_dir": p.is_dir(),
        "needed_steps": sorted({int(s) for s in steps}),
        "needed_max_step": needed,
    }
    if not p.exists():
        out.update({"files_sample": [], "suffix_counts": {}, "candidate_step_coverage": False, "issues": ["trace path missing"]})
        return out
    files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
    out["file_count"] = len(files)
    out["suffix_counts"] = suffix_counts(files)
    out["files_sample"] = [rel(q) for q in files[:60]]
    json_seqs: List[Dict[str, Any]] = []
    csv_infos: List[Dict[str, Any]] = []
    np_infos: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, str]] = []
    for q in files[:250]:
        try:
            if q.stat().st_size > 25 * 1024 * 1024:
                continue
        except OSError:
            continue
        suf = q.suffix.lower()
        if suf in (".json", ".jsn"):
            try:
                json_seqs.extend(inspect_json_obj(read_json(q), needed, rel(q)))
            except Exception as exc:
                parse_errors.append({"source": rel(q), "error": type(exc).__name__, "message": str(exc)[:160]})
        elif suf == ".csv":
            csv_infos.append(inspect_csv_file(q, needed))
        elif suf in (".npy", ".npz"):
            np_infos.append(inspect_numpy_file(q, needed))
    cover_json = any(bool(s.get("covers_needed_max_step")) for s in json_seqs)
    cover_csv = any(bool(s.get("covers_needed_max_step")) for s in csv_infos)
    cover_np = False
    for info in np_infos:
        if info.get("covers_needed_max_step"):
            cover_np = True
        arrays = info.get("arrays")
        if isinstance(arrays, list) and any(bool(a.get("covers_needed_max_step")) for a in arrays if isinstance(a, Mapping)):
            cover_np = True
    issues = []
    if not files:
        issues.append("trace path contains no files")
    if not (cover_json or cover_csv or cover_np):
        issues.append("no parsed structured file visibly covers max requested branch step")
    out.update({
        "candidate_step_coverage": bool(cover_json or cover_csv or cover_np),
        "json_sequences": json_seqs[:12],
        "csv_files": csv_infos[:8],
        "numpy_files": np_infos[:8],
        "parse_errors_sample": parse_errors[:8],
        "issues": issues,
    })
    return out


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# Vehicle true-variable-H v15 candidate trace preflight v0b",
        "",
        f"UTC `{raw['created_utc']}`. Development-only structural preflight; no MPC simulation, no selector refit/search, no gradient training, no validation64, no sealed test.",
        "",
        "## Headline",
        "",
        f"- Candidate branch states: `{h['candidate_branch_states']}` across `{h['unique_trace_count']}` unique H15-prefix traces.",
        f"- Existing trace paths: `{h['existing_trace_count']}/{h['unique_trace_count']}`.",
        f"- Trace paths with parsed step coverage: `{h['trace_step_coverage_count']}/{h['unique_trace_count']}`.",
        f"- Candidate states blocked by missing/opaque traces: `{h['blocked_candidate_count']}`.",
        f"- Suggested minimal smoke if implemented next: `{h['suggested_smoke_episodes']}` development continuation episodes (H10/H15 x1 on the two FP centers), cap `{h['suggested_smoke_control_step_cap']}` control steps.",
        f"- Decision: {raw['decision']}",
        "",
        "## Trace coverage",
        "",
        "| trace | candidates | exists | parsed step coverage | suffixes | issues |",
        "|---|---:|---:|---:|---|---|",
    ]
    for tr in raw["trace_preflight"]:
        lines.append(f"| `{tr['trace_path']}` | {len(tr['needed_steps'])} | `{tr['exists']}` | `{tr['candidate_step_coverage']}` | `{tr.get('suffix_counts')}` | `{'; '.join(tr.get('issues') or [])}` |")
    lines += ["", "## Next experiment design implication", "", raw["interpretation"]]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    h = raw["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v15 candidate trace preflight v0b

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only structural preflight; no MPC simulation, no selector refit/search, no gradient training, no validation64 or sealed test. Candidate branch states={h['candidate_branch_states']} across {h['unique_trace_count']} traces; existing trace paths={h['existing_trace_count']}/{h['unique_trace_count']}; parsed step coverage={h['trace_step_coverage_count']}/{h['unique_trace_count']}; blocked candidates={h['blocked_candidate_count']}. Decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'preflight.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    tail = reg.read_text(encoding="utf-8", errors="replace")[-120000:] if reg.exists() else ""
    if MARKER not in tail:
        with reg.open("a", encoding="utf-8") as f:
            f.write(f"{STAMP},{NAME},development_trace_preflight,v14_fp_candidates,0,0,0,0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    for p in (AUDIT, DONE_AUDIT):
        if not p.exists():
            raise ContractError("missing prerequisite " + rel(p))
    audit = read_json(AUDIT)
    done = read_json(DONE_AUDIT)
    assert_dev_only(audit, "v14 fp audit")
    assert_dev_only(done, "v14 fp audit completed")
    candidates = list(((audit.get("v15_candidate_plan") or {}).get("candidate_branch_states") or []))
    if not candidates:
        raise ContractError("no v15_candidate_plan.candidate_branch_states found")
    unique_traces = sorted({str(c.get("h15_trace_episode_path")) for c in candidates if c.get("h15_trace_episode_path")})
    by_trace: Dict[str, List[int]] = {t: [] for t in unique_traces}
    for c in candidates:
        t = str(c.get("h15_trace_episode_path"))
        if t in by_trace:
            by_trace[t].append(int(c.get("candidate_branch_step")))
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": now().isoformat(),
        "classification": "development_metadata_structural_preflight_no_simulation_no_refit",
        "hypothesis": "The v14 false-positive neighbor audit has enough saved H15-prefix trace information to support a bounded v15 paired H10/H15 continuation acquisition. Structural trace/step availability should be verified before spending MPC simulation budget.",
        "inputs": {"audit": rel(AUDIT), "audit_completed": rel(DONE_AUDIT)},
        "declared_budget": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "candidate_count": len(candidates),
        "unique_traces": unique_traces,
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "latest_verified_backup_before_preflight": args.backup_verified_commit,
        "input_hashes": {rel(AUDIT): sha256(AUDIT), rel(DONE_AUDIT): sha256(DONE_AUDIT), rel(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
    }
    write_json(PROTOCOL, protocol)
    trace_results = [inspect_trace(t, by_trace[t]) for t in unique_traces]
    good_traces = {tr["trace_path"] for tr in trace_results if tr.get("exists") and tr.get("candidate_step_coverage")}
    blocked = [c for c in candidates if str(c.get("h15_trace_episode_path")) not in good_traces]
    fp_centers = [c for c in candidates if c.get("role") == "false_positive_center" and int(c.get("offset_from_center", 999)) == 0]
    suggested_smoke_episodes = len(fp_centers) * 2
    all_ok = len(good_traces) == len(unique_traces) and not blocked
    if all_ok:
        decision = "Inputs are structurally sufficient for a very small v15 smoke: implement/freeze paired H10/H15 continuation on the two false-positive center states first, then expand only if replay fidelity checks pass."
    else:
        decision = "Do not run v15 MPC yet; first repair missing/opaque trace extraction for blocked candidate states or reduce the candidate set to structurally verified traces."
    raw = {
        "created_utc": now().isoformat(),
        "classification": protocol["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "protocol": rel(PROTOCOL),
        "input_hashes": {**protocol["input_hashes"], rel(PROTOCOL): sha256(PROTOCOL)},
        "headline": {
            "candidate_branch_states": len(candidates),
            "unique_trace_count": len(unique_traces),
            "existing_trace_count": sum(1 for tr in trace_results if tr.get("exists")),
            "trace_step_coverage_count": len(good_traces),
            "blocked_candidate_count": len(blocked),
            "suggested_smoke_episodes": suggested_smoke_episodes,
            "suggested_smoke_control_step_cap": suggested_smoke_episodes * 150,
        },
        "candidate_role_counts": {r: sum(1 for c in candidates if c.get("role") == r) for r in sorted({str(c.get("role")) for c in candidates})},
        "trace_preflight": trace_results,
        "blocked_candidates": blocked,
        "decision": decision,
        "interpretation": "This preflight is not evidence of adaptive-H performance. It only checks whether the candidate boundary-acquisition states from the v14 false-positive audit are recoverable enough to justify spending MPC simulation budget. If the next runner is written, keep it development-only, smoke first on the two false-positive centers (4 episodes, 600-step cap), log replay fidelity and actual decision/solver times, and only then consider the full candidate plan.",
        "budget_actual": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "metadata_candidate_states_checked": len(candidates), "validation64_episodes": 0, "sealed_test_episodes": 0},
        "platform": {"python": sys.version, "platform": platform.platform()},
    }
    return raw


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-v15-preflight", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_v15_preflight:
        raise ContractError("requires --run and explicit development preflight acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run(args)
    write_json(OUT / "preflight.json", raw)
    write_summary(raw)
    req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V15_CANDIDATE_TRACE_PREFLIGHT_V0B_{STAMP}.json"
    write_json(req, {"created_utc": raw["created_utc"], "reason": "backup after repaired v15 candidate trace preflight before any v15 MPC simulation source/run", "paths": [rel(OUT), rel(PROTOCOL), rel(STATE), rel(Path(__file__).resolve()), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "validation64_bank_opened": False, "sealed_test_accessed": False})
    done = {
        "passed": True,
        "status": "complete",
        "marker": MARKER,
        "summary": rel(OUT / "summary.md"),
        "preflight": rel(OUT / "preflight.json"),
        "protocol": rel(PROTOCOL),
        "backup_request": rel(req),
        "headline": raw["headline"],
        "decision": raw["decision"],
        "budget_actual": raw["budget_actual"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()), rel(PROTOCOL): sha256(PROTOCOL), rel(OUT / "preflight.json"): sha256(OUT / "preflight.json"), rel(OUT / "summary.md"): sha256(OUT / "summary.md"), rel(req): sha256(req)},
    }
    write_json(OUT / "completed.json", done)
    append_docs(raw)
    state_payload = {
        "utc": raw["created_utc"],
        "headline": raw["headline"],
        "decision": raw["decision"],
        "budget_actual": raw["budget_actual"],
        "artifacts": {"summary": rel(OUT / "summary.md"), "preflight": rel(OUT / "preflight.json"), "completed": rel(OUT / "completed.json"), "protocol": rel(PROTOCOL), "backup_request": rel(req)},
        "failed_predecessor": {"source": "experiments/bohn2021_aws/vehicle_true_variable_horizon_v15_candidate_trace_preflight_v0.py", "failure": "syntax error before execution; registry recorded by run_experiment; superseded by v0b, not overwritten"},
        "current_backup_status": "not verified after v0 syntax failure and v0b preflight; request supervisor backup before any MPC simulation if possible",
        "next_action": "If backup is verified and trace coverage is complete, write/freeze a v15 smoke runner for paired H10/H15 continuations on the two false-positive center states only (4 episodes, 600-step cap); otherwise inspect/repair trace extraction.",
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("# Continue state after v15 candidate trace preflight v0b\n\n" + json.dumps(clean(state_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failed.json", {"passed": False, "status": "failed", "error": type(exc).__name__, "message": str(exc), "validation64_bank_opened": False, "sealed_test_accessed": False})
        raise
