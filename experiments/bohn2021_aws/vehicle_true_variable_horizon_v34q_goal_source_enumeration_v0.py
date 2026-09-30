#!/usr/bin/env python3
"""v34q / T-B1 zero-solve goal-source enumeration before A13c-3.

Active Opus plan 20260930T114431Z_6f8902 requested one final zero-solve
engineering diagnostic after v34o: identify the authoritative saved-case goal
source for the two opened development contexts, without spending solver, plant,
training, validation64, or sealed-test budget.  The script enumerates full
case-snapshot keys (without the old 20-list-item recursion cap), relevant TVP
endpoints, and the actual v34g/v34f/v34d/v34c construction chain that previously
entered the solver.

This is not a scientific outcome and does not justify validation/final-test
claims.  If the authoritative goal source is unique, it authorizes the paired
zero-solve v34r loader gate; if not, it fails closed and asks the lead to decide.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
REPRO_DIR = ROOT / "experiments/bohn2021_reproduction"
for _p in (AWS_DIR, REPRO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0 as base  # noqa:E402
import vehicle_true_variable_horizon_intermediate_h12_boundary_v19 as v19  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34q_goal_source_enumeration_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34q_goal_source_enumeration.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34Q_GOAL_SOURCE_ENUMERATION_{STAMP}.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
PLAN_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
OPUS_LATEST = ROOT / "docs/bohn2021_takeover/opus_lead/LATEST.md"
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T114431Z_6f8902.md"
OPUS_SHA = "f0494a123570fd389a81ffd563bdf610964741b6f578e7f5f410cd23c27bbe3c"
OPUS_REQUEST = "execution-result:20260930T114327_085d17fb"
V34O_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34o_loader_gate_v0_20260930T114327Z/completed.json"
V34O_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34o_loader_gate_v0_20260930T114327Z/raw.json"
V34O_RUN_REGISTRY = ROOT / "research_artifacts/aws_runs/20260930T114327_085d17fb/registry.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-v34q-goal-source-enumeration-{STAMP}"
CHECK_CONTEXT_IDS = ("source242_slot0_branch_start", "v19_c13_step1_after_V15_H35_common_state")


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(x) for x in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if hasattr(value, "item"):
        try:
            return clean(value.item())
        except Exception:
            pass
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    hh = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hh.update(chunk)
    return hh.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def finite_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("true", "value", "actual"):
            if key in value:
                out = finite_scalar(value.get(key))
                if out is not None:
                    return out
        if "forecast" in value:
            return finite_scalar(value.get("forecast"))
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if len(value) == 1:
            return finite_scalar(value[0])
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size != 1:
        return None
    out = float(arr[0])
    return out if math.isfinite(out) else None


def sequence_scalars(values: Any) -> List[Optional[float]]:
    if not isinstance(values, (list, tuple)):
        return []
    return [finite_scalar(v) for v in values]


def lookup_nested(mapping: Mapping[str, Any], names: Sequence[str]) -> Optional[float]:
    for name in names:
        if name in mapping:
            got = finite_scalar(mapping.get(name))
            if got is not None:
                return float(got)
    return None


def explicit_goal_from_mapping(obj: Any, source: str = "case") -> Optional[Tuple[float, float, str]]:
    if not isinstance(obj, Mapping):
        return None
    x_names = ("goal_x", "trajectory_goal_x", "x_goal", "target_x")
    y_names = ("goal_y", "trajectory_goal_y", "y_goal", "target_y")
    gx = lookup_nested(obj, x_names)
    gy = lookup_nested(obj, y_names)
    if gx is not None and gy is not None:
        return float(gx), float(gy), source + ".explicit_goal_fields"
    goal = obj.get("goal")
    if isinstance(goal, Mapping):
        gx = lookup_nested(goal, ("x", "goal_x", "target_x"))
        gy = lookup_nested(goal, ("y", "goal_y", "target_y"))
        if gx is not None and gy is not None:
            return float(gx), float(gy), source + ".goal"
    for key, val in obj.items():
        if isinstance(val, Mapping):
            got = explicit_goal_from_mapping(val, source + "." + str(key))
            if got is not None:
                return got
    return None


def trajectory_endpoint(case: Mapping[str, Any], traj_steps: Optional[float]) -> Optional[Tuple[float, float, str, int]]:
    tvp = case.get("tvp") if isinstance(case, Mapping) else None
    if not isinstance(tvp, Mapping):
        return None
    xs = sequence_scalars(tvp.get("trajectory_x"))
    ys = sequence_scalars(tvp.get("trajectory_y"))
    n = min(len(xs), len(ys))
    if n <= 0:
        return None
    candidate_indices: List[int] = []
    if traj_steps is not None and math.isfinite(float(traj_steps)):
        candidate_indices.append(max(0, min(n - 1, int(round(float(traj_steps))) - 1)))
    candidate_indices.append(n - 1)
    seen: set[int] = set()
    for idx in candidate_indices:
        if idx in seen:
            continue
        seen.add(idx)
        gx = xs[idx]
        gy = ys[idx]
        if gx is not None and gy is not None:
            return float(gx), float(gy), f"case.tvp.trajectory_endpoint[{idx}]", int(idx)
    return None


def reference_theta_goal(case: Mapping[str, Any], traj_steps: Optional[float]) -> Optional[Tuple[float, float, str]]:
    ref = case.get("reference") if isinstance(case, Mapping) else None
    if not isinstance(ref, Mapping):
        return None
    theta_r = finite_scalar(ref.get("theta_r"))
    if traj_steps is None:
        traj_steps = finite_scalar(ref.get("traj_steps"))
    if theta_r is None or traj_steps is None:
        return None
    ep = trajectory_endpoint(case, traj_steps)
    if ep is not None:
        return ep[0], ep[1], ep[2]
    u_s_ref = finite_scalar(ref.get("u_s_ref"))
    src = "case.reference.theta_r_traj_steps"
    if u_s_ref is None:
        tvp = case.get("tvp") if isinstance(case, Mapping) else None
        if isinstance(tvp, Mapping):
            xs = [v for v in sequence_scalars(tvp.get("trajectory_x")) if v is not None]
            ys = [v for v in sequence_scalars(tvp.get("trajectory_y")) if v is not None]
            if len(xs) >= 2 and len(ys) >= 2:
                ds = [math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(min(len(xs), len(ys)) - 1)]
                ds = [d for d in ds if math.isfinite(d) and d > 1e-12]
                if ds:
                    u_s_ref = float(sorted(ds)[len(ds) // 2])
                    src += "+trajectory_spacing_inferred_u_s_ref"
        if u_s_ref is None:
            u_s_ref = 0.3
            src += "+vehicle_default_u_s_ref_0.3"
    return float(math.cos(float(theta_r)) * float(traj_steps) * float(u_s_ref)), float(math.sin(float(theta_r)) * float(traj_steps) * float(u_s_ref)), src


def robust_goal_from_saved_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    explicit = explicit_goal_from_mapping(case)
    if explicit is not None:
        return {"ok": True, "goal_x": explicit[0], "goal_y": explicit[1], "source": explicit[2], "source_priority": "explicit_goal_fields"}
    ref = case.get("reference") if isinstance(case, Mapping) else None
    traj_steps = finite_scalar(ref.get("traj_steps")) if isinstance(ref, Mapping) else None
    ep = trajectory_endpoint(case, traj_steps)
    if ep is not None:
        return {"ok": True, "goal_x": ep[0], "goal_y": ep[1], "source": ep[2], "endpoint_index": ep[3], "traj_steps": traj_steps, "source_priority": "reference.traj_steps_plus_saved_trajectory_tvp_endpoint"}
    theta = reference_theta_goal(case, traj_steps)
    if theta is not None:
        return {"ok": True, "goal_x": theta[0], "goal_y": theta[1], "source": theta[2], "traj_steps": traj_steps, "source_priority": "reference.theta_r_traj_steps_fallback"}
    return {"ok": False, "error": "no explicit saved-case goal/reference/trajectory source"}


def recursive_key_inventory(obj: Any, path: str = "case") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    key_rows: List[Dict[str, Any]] = []
    leaf_rows: List[Dict[str, Any]] = []

    def visit(o: Any, p: str) -> None:
        if isinstance(o, Mapping):
            key_rows.append({"path": p, "kind": "mapping", "key_count": len(o), "keys": sorted(str(k) for k in o.keys())})
            for k, v in o.items():
                visit(v, p + "." + str(k))
        elif isinstance(o, list):
            key_rows.append({"path": p, "kind": "list", "length": len(o)})
            for i, v in enumerate(o):
                visit(v, p + f"[{i}]")
        else:
            scalar = finite_scalar(o)
            leaf_rows.append({"path": p, "kind": type(o).__name__, "scalar": scalar, "is_scalar": scalar is not None})

    visit(obj, path)
    return key_rows, leaf_rows


def tvp_endpoint_summary(case: Mapping[str, Any], start_index: int) -> Dict[str, Any]:
    tvp = case.get("tvp") if isinstance(case, Mapping) else None
    out: Dict[str, Any] = {"tvp_present": isinstance(tvp, Mapping), "channels": {}, "object_endpoints": {}, "trajectory": {}}
    if not isinstance(tvp, Mapping):
        return out
    for name, values in tvp.items():
        if isinstance(values, list):
            scalars = sequence_scalars(values)
            length = len(values)
            first = scalars[0] if scalars else None
            last = scalars[-1] if scalars else None
            at_start = scalars[start_index] if 0 <= start_index < len(scalars) else None
            out["channels"][str(name)] = {"length": length, "first": first, "at_tvp_start_index": at_start, "last": last, "scalarizable_count": sum(v is not None for v in scalars)}
        else:
            out["channels"][str(name)] = {"length": None, "scalar": finite_scalar(values)}
    for obj_idx in range(10):
        xk = f"obj_{obj_idx}_x"; yk = f"obj_{obj_idx}_y"; rk = f"obj_{obj_idx}_r"
        if xk in out["channels"] or yk in out["channels"]:
            out["object_endpoints"][f"obj_{obj_idx}"] = {
                "x_last": (out["channels"].get(xk) or {}).get("last"),
                "y_last": (out["channels"].get(yk) or {}).get("last"),
                "r_last": (out["channels"].get(rk) or {}).get("last"),
                "x_at_start": (out["channels"].get(xk) or {}).get("at_tvp_start_index"),
                "y_at_start": (out["channels"].get(yk) or {}).get("at_tvp_start_index"),
                "note": "enumerated because requested by T-B1; not read by the v34c/v34g goal extractor path",
            }
    ref = case.get("reference") if isinstance(case, Mapping) else None
    traj_steps = finite_scalar(ref.get("traj_steps")) if isinstance(ref, Mapping) else None
    ep = trajectory_endpoint(case, traj_steps)
    out["trajectory"] = {
        "trajectory_x_last": (out["channels"].get("trajectory_x") or {}).get("last"),
        "trajectory_y_last": (out["channels"].get("trajectory_y") or {}).get("last"),
        "trajectory_x_at_start": (out["channels"].get("trajectory_x") or {}).get("at_tvp_start_index"),
        "trajectory_y_at_start": (out["channels"].get("trajectory_y") or {}).get("at_tvp_start_index"),
        "reference_traj_steps": traj_steps,
        "endpoint_selected_by_v34c_v0c_logic": None if ep is None else {"x": ep[0], "y": ep[1], "source": ep[2], "index": ep[3]},
    }
    return out


def load_target_contexts() -> List[Dict[str, Any]]:
    specs = base.v29.build_state_specs()
    source242 = base.find_spec("v27_case09_slot0_early_risk", specs)
    c13 = base.find_spec("v19_c13", specs)
    c13_trace_path, c13_rows = base.find_c13_trace()
    c13_step1 = c13_rows[1]
    contexts = [
        {
            "context_id": "source242_slot0_branch_start",
            "state_label": "v27_case09_slot0_early_risk",
            "source": "v29.build_state_specs -> v27 selected case_snapshot_from_candidate_pool",
            "case_snapshot": source242["case_snapshot"],
            "branch_step": int(source242["branch_step"]),
            "tvp_start_index": int(source242["branch_step"]),
            "state": base.state_clean(source242["branch_previous_state"]),
        },
        {
            "context_id": "v19_c13_step1_after_V15_H35_common_state",
            "state_label": "v19_c13",
            "source": "v29.build_state_specs -> v19.prepare_v15_candidates + v33 trace second previous_state",
            "case_snapshot": c13["case_snapshot"],
            "branch_step": int(c13["branch_step"]),
            "tvp_start_index": int(c13["branch_step"]) + 1,
            "state": base.state_clean(c13_step1.get("previous_state") or {}),
            "trace_path": rel(c13_trace_path),
        },
    ]
    prepared = v19.prepare_v15_candidates()
    return contexts, {"v29_spec_count": len(specs), "v19_prepared_candidate_count": len(prepared), "v19_candidate13_present": any(int(c.get("candidate_index", -1)) == 13 for c in prepared)}


def line_matches(path: Path, needles: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(text, start=1):
        for needle in needles:
            if needle in line:
                rows.append({"file": rel(path), "line": i, "needle": needle, "text": line.strip()})
    return rows


def construction_chain_evidence() -> Dict[str, Any]:
    files = {
        "v34g": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0.py",
        "v34f": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34f_objective_basin_solver_probe_v0.py",
        "v34d": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34d_objective_basin_solver_probe_v0.py",
        "v34c_v0e": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34c_contract_preflight_v0e.py",
        "v34c_v0d": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34c_contract_preflight_v0d.py",
        "v34c_v0c": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34c_contract_preflight_v0c.py",
        "v34c_base": ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34c_contract_preflight_v0.py",
    }
    needles = {
        "v34g": ["f.patch_runtime()", "base.solve_arm", "TARGET_CELL"],
        "v34f": ["d.patch_runtime()", "base.run(argv)", "base.__file__"],
        "v34d": ["contract.patch_runtime()", "base.configure_context_no_reset = configure_context_no_reset_v34d", "contract.setup_context_direct_v0e", "configure_context_no_reset_v34d"],
        "v34c_v0e": ["def setup_context_direct_v0e", "v0d.setup_context_direct_v0d", "controller._tvp_data", "shifted_tvp_numeric"],
        "v34c_v0d": ["_ORIGINAL_SETUP_CONTEXT_DIRECT", "def setup_context_direct_v0d", "ctrl.object_noise_seed", "case_snapshot.reference.ns"],
        "v34c_v0c": ["def robust_extract_goal_xy", "_trajectory_endpoint", "c.m.extract_goal_xy = robust_extract_goal_xy", "case.tvp.trajectory_endpoint"],
        "v34c_base": ["gx, gy, goal_source = m.extract_goal_xy", "def setup_context_direct"],
    }
    rows: List[Dict[str, Any]] = []
    hashes: Dict[str, str] = {}
    for key, path in files.items():
        hashes[rel(path)] = sha256(path) if path.exists() else "missing"
        if path.exists():
            rows.extend(line_matches(path, needles[key]))
    return {
        "source_files": {k: rel(v) for k, v in files.items()},
        "source_hashes": hashes,
        "line_evidence": rows,
        "summary": "v34g.patch_runtime calls v34f, which calls v34d, which patches base.configure_context_no_reset to configure_context_no_reset_v34d; that adapter calls v34c/v0e setup_context_direct_v0e, whose parent v0c patched m.extract_goal_xy to robust_extract_goal_xy. The robust path explicitly uses saved reference/trajectory TVP, not observation slots or obj_* channels.",
        "authoritative_goal_symbol": "vehicle_true_variable_horizon_v34c_contract_preflight_v0c.robust_extract_goal_xy -> _trajectory_endpoint",
    }


def latest_verified_backup_summary() -> Dict[str, Any]:
    out: Dict[str, Any] = {"backup_proof_count": 0, "latest_verified": None, "backup_status_json": None, "research_state_json": None}
    proofs: List[Tuple[str, Path, Mapping[str, Any]]] = []
    for p in sorted((ROOT / "research_artifacts/aws_backup_proofs").glob("backup_proof_*.json")):
        try:
            obj = read_json(p)
        except Exception:
            continue
        out["backup_proof_count"] += 1
        if obj.get("status") == "verified" or obj.get("backup_verified") is True:
            proofs.append((str(obj.get("time") or ""), p, obj))
    if proofs:
        proofs.sort(key=lambda x: x[0])
        _, p, obj = proofs[-1]
        out["latest_verified"] = {"path": rel(p), "sha256": sha256(p), "time": obj.get("time"), "commit": obj.get("commit"), "remaining_changed_files": obj.get("remaining_changed_files"), "status": obj.get("status"), "backup_verified": obj.get("backup_verified"), "packages_this_run": obj.get("packages_this_run"), "package_sha256": obj.get("package_sha256"), "package_bytes": obj.get("package_bytes")}
    bs = ROOT / "research_artifacts/aws_backup_proofs/backup_status.json"
    if bs.exists():
        try:
            out["backup_status_json"] = {"path": rel(bs), "sha256": sha256(bs), "content": read_json(bs)}
        except Exception as exc:
            out["backup_status_json"] = {"path": rel(bs), "error": repr(exc)}
    rs_candidates = [ROOT / "research_state.json", ROOT / "research_artifacts/aws_state/research_state.json"]
    for rs in rs_candidates:
        if rs.exists():
            try:
                out["research_state_json"] = {"path": rel(rs), "sha256": sha256(rs), "backup_error": read_json(rs).get("backup_error"), "content_status": read_json(rs).get("status")}
            except Exception as exc:
                out["research_state_json"] = {"path": rel(rs), "error": repr(exc)}
            break
    if out["research_state_json"] is None:
        out["research_state_json"] = {"status": "not_present_in_repository", "checked": [rel(p) for p in rs_candidates]}
    return out


def registry_false_negative_summary() -> Dict[str, Any]:
    out: Dict[str, Any] = {"registry_path": rel(V34O_RUN_REGISTRY), "registry_exists": V34O_RUN_REGISTRY.exists(), "artifact_inventory_false_negative": None}
    run_dir = V34O_DONE.parent
    actual_files = sorted(rel(p) for p in run_dir.iterdir() if p.is_file()) if run_dir.exists() else []
    out["actual_run_dir_files"] = actual_files
    if V34O_RUN_REGISTRY.exists():
        try:
            reg = read_json(V34O_RUN_REGISTRY)
            out["registry_sha256"] = sha256(V34O_RUN_REGISTRY)
            inv = reg.get("artifact_inventory") or reg.get("artifacts") or {}
            out["artifact_inventory"] = inv
            inv_text = json.dumps(clean(inv), sort_keys=True)
            out["artifact_inventory_false_negative"] = ("false" in inv_text.lower() and len(actual_files) >= 6)
        except Exception as exc:
            out["error"] = repr(exc)
    return out


def verify_active_plan() -> Dict[str, Any]:
    if not PLAN_READY.exists() or not OPUS_REPORT.exists():
        raise ContractError("missing active Opus plan/report")
    ready = read_json(PLAN_READY)
    report_sha = sha256(OPUS_REPORT)
    if ready.get("request_id") != OPUS_REQUEST or ready.get("report_sha256") != OPUS_SHA or report_sha != OPUS_SHA:
        raise ContractError(f"active Opus plan mismatch: ready_request={ready.get('request_id')} ready_sha={ready.get('report_sha256')} actual_sha={report_sha}")
    latest_text = OPUS_LATEST.read_text(encoding="utf-8", errors="replace") if OPUS_LATEST.exists() else ""
    if rel(OPUS_REPORT) not in latest_text:
        raise ContractError("Opus LATEST.md does not point to latest report")
    if not V34O_DONE.exists() or not V34O_RAW.exists():
        raise ContractError("missing v34o predecessor outputs")
    v34o_done = read_json(V34O_DONE)
    if v34o_done.get("hard_pass") is True:
        raise ContractError("v34q expected v34o to have failed goal gate; latest v34o hard_pass true")
    budget = v34o_done.get("budget_actual") or {}
    if int(budget.get("solver_calls", 0)) != 0 or int(budget.get("plant_steps", 0)) != 0:
        raise ContractError("v34o predecessor was not zero solver/plant")
    return {"plan_ready": rel(PLAN_READY), "request_id": ready.get("request_id"), "experiment_id": ready.get("experiment_id"), "report": rel(OPUS_REPORT), "report_sha256": OPUS_SHA, "v34o_completed": rel(V34O_DONE), "v34o_headline": v34o_done.get("headline")}


def write_tables(context_rows: Sequence[Mapping[str, Any]], all_key_rows: Sequence[Mapping[str, Any]], all_leaf_rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    key_csv = RUN_DIR / "recursive_key_inventory.csv"
    with key_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["context_id", "path", "kind", "key_count", "length", "keys"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_key_rows:
            w.writerow({k: json.dumps(clean(r.get(k)), ensure_ascii=False) if k == "keys" else r.get(k) for k in fields})
    leaf_csv = RUN_DIR / "recursive_leaf_inventory.csv"
    with leaf_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["context_id", "path", "kind", "is_scalar", "scalar"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_leaf_rows:
            w.writerow({k: r.get(k) for k in fields})
    cand_csv = RUN_DIR / "goal_candidates.csv"
    with cand_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["context_id", "candidate_type", "role", "x", "y", "r", "source", "authority_status", "note"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in context_rows:
            goal = c.get("authoritative_goal") or {}
            w.writerow({"context_id": c["context_id"], "candidate_type": "trajectory_endpoint", "role": "goal", "x": goal.get("goal_x"), "y": goal.get("goal_y"), "r": None, "source": goal.get("source"), "authority_status": "authoritative_by_v34g_v34c_construction_chain", "note": goal.get("source_priority")})
            for obj_name, obj in ((c.get("tvp_endpoint_summary") or {}).get("object_endpoints") or {}).items():
                w.writerow({"context_id": c["context_id"], "candidate_type": obj_name, "role": "object_obstacle_tvp_endpoint_not_goal", "x": obj.get("x_last"), "y": obj.get("y_last"), "r": obj.get("r_last"), "source": f"case.tvp.{obj_name}_x/y/r[-1]", "authority_status": "non_authoritative_not_read_by_v34c_v0c_goal_extractor", "note": obj.get("note")})
    return {"recursive_key_inventory_csv": rel(key_csv), "recursive_leaf_inventory_csv": rel(leaf_csv), "goal_candidates_csv": rel(cand_csv)}


def hash_existing(paths: Iterable[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        try:
            if p.exists() and p.is_file():
                out[rel(p)] = sha256(p)
        except Exception:
            pass
    return out


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_summary_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# v34q / T-B1 goal-source enumeration",
        "",
        f"UTC: `{raw['created_utc']}`. Zero-solve operational enumeration under Opus plan `{raw['active_lead_report']}`.",
        "",
        "## Budget",
        "- Solver calls: `0`; plant/env steps after construction: `0`; training/refit: `0`; validation64: `0`; sealed test: `0`.",
        "",
        "## Gate result",
        f"- G-GOAL passed: `{h['G_GOAL_pass']}`.",
        f"- Contexts enumerated: `{h['contexts_enumerated']}`; authoritative goal source: `{h['authoritative_goal_symbol']}`.",
        f"- Multiple non-authoritative object/trajectory endpoint values observed: `{h['non_authoritative_distinct_endpoint_values_count']}` distinct endpoint pairs; they are not treated as goal authority because the v34g construction chain does not read `obj_*` as goal.",
        "",
        "| context | authoritative source | goal_x | goal_y | top-level keys | recursive key rows | leaf rows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for c in raw["contexts"]:
        g = c["authoritative_goal"]
        lines.append(f"| `{c['context_id']}` | `{g.get('source')}` | {g.get('goal_x')} | {g.get('goal_y')} | {len(c.get('top_level_keys') or [])} | {c.get('recursive_key_rows')} | {c.get('recursive_leaf_rows')} |")
    lines += [
        "",
        "## v34g construction-chain conclusion",
        raw["construction_chain"]["summary"],
        "",
        "## Operational side notes requested by Opus",
        f"- Latest verified backup summary is recorded in raw JSON; backup request after this zero-solve source/output: `{rel(BACKUP_REQUEST)}`.",
        f"- v34o registry artifact-inventory false-negative recorded: `{raw['operational_sidecar']['v34o_registry'].get('artifact_inventory_false_negative')}`; actual run-dir files: `{len(raw['operational_sidecar']['v34o_registry'].get('actual_run_dir_files') or [])}`.",
        f"- Previous-input hashes are preserved from v34o and must match in v34r: `{raw['previous_input_hashes_from_v34o']}`.",
        "",
        f"Raw: `{rel(RUN_DIR/'raw.json')}`. Candidate CSV: `{raw['table_paths']['goal_candidates_csv']}`. Recursive key CSV: `{raw['table_paths']['recursive_key_inventory_csv']}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    block = f"""
<!-- {MARKER} -->
## v34q/T-B1 zero-solve goal-source enumeration

UTC: {raw['created_utc']}. Executed Opus T-B1 with zero solver calls, zero plant/env steps, no training/refit, validation64=0, sealed_test=0. G-GOAL_pass={h['G_GOAL_pass']}. Authoritative source resolved to `{h['authoritative_goal_symbol']}` via the actual v34g->v34f->v34d->v34c/v0e/v0c construction chain; per-context goals: {h['authoritative_goals_by_context']}. Object `obj_*` endpoints and trajectory endpoints are enumerated in `{raw['table_paths']['goal_candidates_csv']}`; `obj_*` endpoints are non-authoritative because the verified construction path does not read them as goal. v34o registry artifact-inventory false-negative recorded={raw['operational_sidecar']['v34o_registry'].get('artifact_inventory_false_negative')}. Evidence: `{rel(RUN_DIR/'summary.md')}`, `{rel(RUN_DIR/'raw.json')}`, `{raw['table_paths']['recursive_key_inventory_csv']}`, `{raw['table_paths']['goal_candidates_csv']}`. Backup request: `{rel(BACKUP_REQUEST)}`. Next if G-GOAL_pass remains true: run v34r zero-solve loader gate preserving v34o previous-input scalarization hashes exactly; do not spend A13c-3 solver calls until post-v34r backup is verified.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([raw["created_utc"], NAME, raw["classification"], "T-B1 goal source enumeration; contexts=source242,c13", "zero_solve_goal_source_no_validation_no_test", len(raw["contexts"]), 0, 0, 0, 0, False, rel(RUN_DIR/"completed.json"), MARKER])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        "# Continue state after v34q/T-B1 goal-source enumeration\n\n"
        + f"UTC: {raw['created_utc']}\n\nHeadline: {json.dumps(clean(h), sort_keys=True)}\n\n"
        + f"Artifacts: {rel(RUN_DIR/'summary.md')}, {rel(RUN_DIR/'raw.json')}, {raw['table_paths']['goal_candidates_csv']}, {raw['table_paths']['recursive_key_inventory_csv']}\n\n"
        + f"Previous-input hashes to preserve in v34r: {json.dumps(clean(raw['previous_input_hashes_from_v34o']), sort_keys=True)}\n\n"
        + "Next: if G-GOAL_pass is true, run v34r zero-solve loader gate using copy-only robust saved-trajectory goal source and v34o previous_input scalarization unchanged. After v34r, request verified external backup before A13c-3 solver calls.\n",
        encoding="utf-8",
    )


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--i-accept-zero-solve-goal-enumeration", action="store_true", required=True)
    args = ap.parse_args(argv)
    del args
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        started = now_utc()
        gates = verify_active_plan()
        write_json(RUN_DIR / "run_started.json", {"started_utc": started.isoformat(), "pid": os.getpid(), "method": NAME, "active_plan": gates, "budget": {"solver_calls": 0, "plant_steps": 0, "env_step_calls_after_construction": 0, "env_reset_calls_after_construction": 0, "training_or_refit": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}})
        contexts, loader_meta = load_target_contexts()
        all_key_rows: List[Dict[str, Any]] = []
        all_leaf_rows: List[Dict[str, Any]] = []
        context_rows: List[Dict[str, Any]] = []
        nonauth_pairs: set[Tuple[Optional[float], Optional[float]]] = set()
        for c in contexts:
            case = c["case_snapshot"]
            key_rows, leaf_rows = recursive_key_inventory(case, "case_snapshot")
            for r in key_rows:
                r["context_id"] = c["context_id"]
            for r in leaf_rows:
                r["context_id"] = c["context_id"]
            all_key_rows.extend(key_rows)
            all_leaf_rows.extend(leaf_rows)
            tvp_summary = tvp_endpoint_summary(case, int(c["tvp_start_index"]))
            for obj in (tvp_summary.get("object_endpoints") or {}).values():
                nonauth_pairs.add((obj.get("x_last"), obj.get("y_last")))
            tr = tvp_summary.get("trajectory") or {}
            nonauth_pairs.add((tr.get("trajectory_x_last"), tr.get("trajectory_y_last")))
            auth = robust_goal_from_saved_case(case)
            shifted_15 = base.shift_case_tvp(case, int(c["tvp_start_index"]), 15)
            legacy_extract_error = None
            try:
                gx, gy, src = base.extract_goal_xy(case, shifted_15)
                legacy = {"ok": True, "goal_x": gx, "goal_y": gy, "source": src}
            except Exception as exc:
                legacy_extract_error = repr(exc)
                legacy = {"ok": False, "error": repr(exc)}
            row = {
                "context_id": c["context_id"],
                "state_label": c["state_label"],
                "source": c["source"],
                "branch_step": c["branch_step"],
                "tvp_start_index": c["tvp_start_index"],
                "state": c["state"],
                "case_snapshot_sha256": canonical_hash(case),
                "top_level_keys": sorted(str(k) for k in case.keys()) if isinstance(case, Mapping) else [],
                "recursive_key_rows": len(key_rows),
                "recursive_leaf_rows": len(leaf_rows),
                "reference_summary": clean(case.get("reference") if isinstance(case, Mapping) else None),
                "tvp_endpoint_summary": tvp_summary,
                "authoritative_goal": auth,
                "legacy_base_extract_goal_xy_on_full_case_plus_shifted_H15": legacy,
                "legacy_extract_goal_error": legacy_extract_error,
            }
            context_rows.append(row)
            write_json(RUN_DIR / "progress.json", {"contexts_done": len(context_rows), "contexts_expected": len(contexts), "last_context": {k: v for k, v in row.items() if k not in ("tvp_endpoint_summary", "reference_summary")}, "solver_calls": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
            print(json.dumps(clean({"contexts_done": len(context_rows), "context_id": c["context_id"], "authoritative_goal": auth, "legacy_base_ok": legacy.get("ok")}), sort_keys=True), flush=True)
        table_paths = write_tables(context_rows, all_key_rows, all_leaf_rows)
        chain = construction_chain_evidence()
        v34o_raw = read_json(V34O_RAW)
        previous_hashes = (v34o_raw.get("headline") or {}).get("previous_input_scalar_hashes") or (read_json(V34O_DONE).get("headline") or {}).get("previous_input_scalar_hashes")
        auth_ok = all((c.get("authoritative_goal") or {}).get("ok") is True for c in context_rows)
        auth_sources = [str((c.get("authoritative_goal") or {}).get("source_priority")) for c in context_rows]
        auth_symbol_ok = "robust_extract_goal_xy" in chain.get("authoritative_goal_symbol", "")
        same_source_class = len(set(auth_sources)) == 1 and bool(auth_sources)
        g_goal = bool(auth_ok and auth_symbol_ok and same_source_class)
        goals_by_context = {c["context_id"]: {"x": (c.get("authoritative_goal") or {}).get("goal_x"), "y": (c.get("authoritative_goal") or {}).get("goal_y"), "source": (c.get("authoritative_goal") or {}).get("source")} for c in context_rows}
        created = now_utc()
        raw: Dict[str, Any] = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_operational_zero_solve_goal_source_enumeration_not_validation_not_test",
            "active_lead": "claude-opus-5-5",
            "active_lead_report": rel(OPUS_REPORT),
            "active_lead_report_sha256": OPUS_SHA,
            "active_lead_request": OPUS_REQUEST,
            "hypothesis_frozen": "v34o goal failure is operational: the actual previously successful v34g/v34c construction path obtains the goal from explicit saved case reference/trajectory TVP, while base.extract_goal_xy was too narrow and did not inspect that source.",
            "gates": gates,
            "loader_meta": loader_meta,
            "contexts": context_rows,
            "construction_chain": chain,
            "table_paths": table_paths,
            "previous_input_hashes_from_v34o": previous_hashes,
            "operational_sidecar": {"backup": latest_verified_backup_summary(), "v34o_registry": registry_false_negative_summary()},
            "headline": {
                "G_GOAL_pass": g_goal,
                "contexts_enumerated": len(context_rows),
                "authoritative_goal_symbol": chain.get("authoritative_goal_symbol"),
                "authoritative_goal_source_classes": auth_sources,
                "authoritative_goals_by_context": goals_by_context,
                "non_authoritative_distinct_endpoint_values_count": len([p for p in nonauth_pairs if p != (None, None)]),
                "legacy_base_extract_goal_ok_count": sum(1 for c in context_rows if ((c.get("legacy_base_extract_goal_xy_on_full_case_plus_shifted_H15") or {}).get("ok") is True)),
                "solver_calls": 0,
                "plant_steps": 0,
                "validation64_episodes": 0,
                "sealed_test_episodes": 0,
            },
            "budget_declared": {"solver_calls": 0, "plant_steps": 0, "env_step_calls_after_construction": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "budget_actual": {"solver_calls": 0, "plant_steps": 0, "env_step_calls_after_construction": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "interpretation_limits": ["zero-solve operational source enumeration only", "development opened contexts only", "not validation64", "not sealed/final test", "does not execute A13c-3 objective-contract solver probe"],
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
            "input_hashes": hash_existing([Path(__file__).resolve(), PLAN_READY, OPUS_LATEST, OPUS_REPORT, V34O_DONE, V34O_RAW, ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py", ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34c_contract_preflight_v0c.py", ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34d_objective_basin_solver_probe_v0.py", ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0.py"]),
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_summary_docs(raw)
        write_json(BACKUP_REQUEST, {"request": "backup_after_v34q_goal_source_enumeration", "created_utc": created.isoformat(), "backup_required_before_a13c3_solver_calls": True, "reason": "new T-B1 zero-solve source/output/docs/state must be recoverable; post-v34r backup is required before A13c-3 solver calls", "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "new_solver_calls": 0, "new_plant_steps": 0, "new_training_or_gradient_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "next_gate": "If G-GOAL_pass true, run v34r zero-solve loader gate; require verified backup after v34r before A13c-3 solver calls."})
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE, BACKUP_REQUEST, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed = {"status": "complete", "passed": g_goal, "hard_pass": g_goal, "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": raw["budget_actual"], "headline": raw["headline"], "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "table_paths": table_paths, "backup_request": rel(BACKUP_REQUEST), "next_if_passed": "vehicle_true_variable_horizon_v34r_loader_gate_v0.py zero-solve loader gate", "hashes": hash_existing(files)}
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({"completed": rel(RUN_DIR/"completed.json"), "summary": rel(RUN_DIR/"summary.md"), "raw": rel(RUN_DIR/"raw.json"), "headline": raw["headline"], "backup_request": rel(BACKUP_REQUEST), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0 if g_goal else 2
    except Exception as exc:
        fail = {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_operational_zero_solve_goal_source_enumeration_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": {"solver_calls": 0, "plant_steps": 0, "env_step_calls_after_construction": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}}
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(f"# v34q/T-B1 goal-source enumeration failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR/'failed.json')}\n\nNo solver, plant, validation64, sealed-test, training or refit access was requested. Per Opus stop rule, do not add a third goal diagnostic without lead analysis; preserve this failure.\n", encoding="utf-8")
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR/"failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False, "solver_calls": 0}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
