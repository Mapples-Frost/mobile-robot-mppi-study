#!/usr/bin/env python3
"""v34o / T-A61 zero-solve loader gate after v34n parser failure.

Active Opus plan 20260930T113021Z_20dfa3 authorizes this operational gate after
v34n failed before any solver call on a legacy base.load_contexts() assumption.
This script implements T-A60 as an extension-file monkey patch (not by editing
frozen v34 sources):

* strict scalar normalization for trace previous_input channels using the
  existing finite-array-to-scalar semantics, recorded with per-channel rules;
* no silent defaulting of missing previous_input channels in context
  configuration; missing/non-finite channels fail loud;
* load both source242_slot0 and v19_c13 contexts, then check shifted TVP suffixes
  for H={12,15,35};
* instantiate no-reset controller contexts only to fingerprint _u0 shape/norm.

Budget is zero solves, zero plant/env steps, zero training/refit, zero
validation64, and zero sealed-test access.  Passing this gate does not itself
create scientific evidence; it only allows the already approved A13c-3 solver
probe to run after external backup verification.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
REPRO_DIR = ROOT / "experiments/bohn2021_reproduction"
for _p in (AWS_DIR, REPRO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0 as base  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34o_loader_gate_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34o_loader_gate.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
BACKUP_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V34O_LOADER_GATE_{STAMP}.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
PLAN_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
OPUS_LATEST = ROOT / "docs/bohn2021_takeover/opus_lead/LATEST.md"
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T113021Z_20dfa3.md"
OPUS_REPORT_SHA = "251993db94a04f38d22747754de141a97fd39ea7008ebd48b901ec382747d959"
OPUS_REQUEST = "execution-result:20260930T112938_862dec3e"
V34M_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34m_residual_attribution_v0_20260930T111306Z/completed.json"
V34N_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34n_nonconverged_objective_contract_probe_v0_20260930T112938Z/failed.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
REQUIRED_INPUT_CHANNELS = ("u_omega", "u_s")
CHECK_HORIZONS = (12, 15, 35)
MARKER = f"vehicle-v34o-loader-gate-{STAMP}"


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(v: Any) -> Any:
    if isinstance(v, Path):
        return rel(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Mapping):
        return {str(k): clean(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [clean(x) for x in v]
    if hasattr(v, "tolist"):
        return clean(v.tolist())
    if hasattr(v, "item"):
        return clean(v.item())
    return v


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


def arr(value: Any) -> np.ndarray:
    try:
        if hasattr(value, "full"):
            return np.asarray(value.full(), dtype=float).reshape(-1)
        if hasattr(value, "cat"):
            return np.asarray(value.cat, dtype=float).reshape(-1)
        if hasattr(value, "master"):
            return np.asarray(value.master, dtype=float).reshape(-1)
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def arr_hash(value: Any) -> Optional[str]:
    a = arr(value)
    if a.size == 0:
        return None
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def scalarize(value: Any, channel: str, source: str) -> Tuple[float, Dict[str, Any]]:
    """Strict finite scalar coercion: singleton list/array is OK; >1 values fail."""
    try:
        a = arr(value)
    except Exception as exc:  # arr() is defensive, but keep provenance explicit.
        raise ContractError(f"{source}.{channel} cannot be converted to numeric array: {exc!r}")
    if int(a.size) != 1:
        raise ContractError(f"{source}.{channel} expected exactly one finite scalar after unwrap, got size={a.size} raw_type={type(value).__name__}")
    out = float(a.reshape(-1)[0])
    if not math.isfinite(out):
        raise ContractError(f"{source}.{channel} scalar is not finite: {out}")
    if isinstance(value, list):
        rule = "singleton_list_unwrap" if len(value) == 1 else "list_array_singleton_flatten"
    elif hasattr(value, "tolist") or hasattr(value, "full") or hasattr(value, "cat"):
        rule = "array_like_singleton_flatten"
    else:
        rule = "direct_scalar_float"
    return out, {"channel": channel, "rule": rule, "raw_type": type(value).__name__, "flat_size": int(a.size), "scalar": out, "raw_preview": clean(value)}


def normalize_previous_input(raw: Any, required_names: Sequence[str], source: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise ContractError(f"{source} previous_input must be a mapping, got {type(raw).__name__}")
    missing = [name for name in required_names if name not in raw]
    if missing:
        raise ContractError(f"{source} previous_input missing required channels {missing}; refusing silent zero default")
    scalars: Dict[str, float] = {}
    rules: Dict[str, Any] = {}
    for name in required_names:
        scalar, meta = scalarize(raw[name], str(name), source)
        scalars[str(name)] = scalar
        rules[str(name)] = meta
    extras = sorted(str(k) for k in raw.keys() if str(k) not in set(required_names))
    return scalars, {"source": source, "required_channels": list(required_names), "missing_channels": [], "extra_channels": extras, "channel_rules": rules, "hash": canonical_hash(scalars)}


def load_contexts_strict() -> List[Dict[str, Any]]:
    specs = base.v29.build_state_specs()
    source242 = base.find_spec("v27_case09_slot0_early_risk", specs)
    c13 = base.find_spec("v19_c13", specs)
    c13_trace_path, c13_rows = base.find_c13_trace()
    c13_step0 = c13_rows[0]
    c13_step1 = c13_rows[1]
    src_prev, src_norm = normalize_previous_input({"u_omega": 0.0, "u_s": 0.0}, REQUIRED_INPUT_CHANNELS, "source242_branch_reset_zero_input")
    c13_prev, c13_norm = normalize_previous_input(c13_step0.get("input") or {}, REQUIRED_INPUT_CHANNELS, rel(c13_trace_path) + ":row0.input")
    contexts: List[Dict[str, Any]] = [
        {
            "context_id": "source242_slot0_branch_start",
            "state_label": "v27_case09_slot0_early_risk",
            "source": "v29/v33 selected branch state, original branch start",
            "case_snapshot": copy.deepcopy(source242["case_snapshot"]),
            "branch_step": int(source242["branch_step"]),
            "tvp_start_index": int(source242["branch_step"]),
            "state": base.state_clean(source242["branch_previous_state"]),
            "previous_input": src_prev,
            "previous_input_source": "Astra/Opus-specified branch-reset zero-input semantics",
            "previous_input_normalization": src_norm,
            "raw_previous_input": {"u_omega": 0.0, "u_s": 0.0},
            "horizons": list(CHECK_HORIZONS),
        },
        {
            "context_id": "v19_c13_step1_after_V15_H35_common_state",
            "state_label": "v19_c13",
            "source": rel(c13_trace_path) + ": second trace row previous_state",
            "case_snapshot": copy.deepcopy(c13["case_snapshot"]),
            "branch_step": int(c13["branch_step"]),
            "tvp_start_index": int(c13["branch_step"]) + 1,
            "state": base.state_clean(c13_step1.get("previous_state") or {}),
            "previous_input": c13_prev,
            "previous_input_source": rel(c13_trace_path) + ": first trace row input",
            "previous_input_normalization": c13_norm,
            "raw_previous_input": c13_step0.get("input") or {},
            "horizons": list(CHECK_HORIZONS),
        },
    ]
    expected = {"theta": 0.08060330210484318, "x": 13.74497830467862, "y": 2.8552860589337556}
    if base.state_distance(contexts[1]["state"], expected) > 1e-6:
        raise ContractError("c13 reconstructed state does not match previously specified fixed state")
    return contexts


def input_vector_strict(env: Any, ctrl: Any, previous_input: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    names = list(getattr(ctrl, "input_names", []) or [])
    if not names and hasattr(ctrl, "current_input"):
        names = list(ctrl.current_input.keys())
    if not names:
        names = list(REQUIRED_INPUT_CHANNELS)
    scalars, norm = normalize_previous_input(previous_input, names, "configure_context_no_reset.previous_input")
    if hasattr(env.control_system, "get_input_vector"):
        try:
            vec = env.control_system.get_input_vector(dict(scalars))
            return vec, {**norm, "vector_source": "env.control_system.get_input_vector", "input_order": names}
        except Exception as exc:
            raise ContractError("strict get_input_vector failed after channel validation: " + repr(exc))
    vec = np.asarray([float(scalars[name]) for name in names], dtype=float).reshape((-1, 1))
    return vec, {**norm, "vector_source": "manual_controller_input_names", "input_order": names}


def configure_context_no_reset_strict(env: Any, context: Mapping[str, Any], h: int) -> Dict[str, Any]:
    shifted = base.shift_case_tvp(context["case_snapshot"], int(context["tvp_start_index"]), h)
    gx, gy, goal_source = base.extract_goal_xy(context["case_snapshot"], shifted)
    ctrl = env.control_system.controller
    state = base.state_clean(context["state"])
    for name, values in shifted.items():
        if hasattr(env.control_system, "tvps") and name in env.control_system.tvps:
            env.control_system.tvps[name].values = copy.deepcopy(values)
    try:
        env.control_system._step_count = 0
    except Exception:
        pass
    try:
        env.control_system.current_state.update(copy.deepcopy(state))
    except Exception:
        env.control_system.current_state = copy.deepcopy(state)
    try:
        ctrl.reset(env.control_system.current_state, reference=None, constraint=None, tvp=None)
    except Exception as exc:
        raise ContractError("controller.reset failed in no-env-reset context reconstruction: " + repr(exc))
    for attr in ("goal_x",):
        try:
            setattr(ctrl, attr, gx)
        except Exception:
            pass
        try:
            setattr(env, "trajectory_goal_x", gx)
        except Exception:
            pass
    for attr in ("goal_y",):
        try:
            setattr(ctrl, attr, gy)
        except Exception:
            pass
        try:
            setattr(env, "trajectory_goal_y", gy)
        except Exception:
            pass
    try:
        env.control_system.current_state.update(copy.deepcopy(state))
    except Exception:
        pass
    names = list(getattr(ctrl, "input_names", []) or (list(ctrl.current_input.keys()) if hasattr(ctrl, "current_input") else []))
    if not names:
        names = list(REQUIRED_INPUT_CHANNELS)
    prev_input, prev_norm = normalize_previous_input(context.get("previous_input") or {}, names, f"{context.get('context_id')}.previous_input")
    if hasattr(ctrl, "current_input"):
        for name in list(ctrl.current_input.keys()):
            if name not in prev_input:
                raise ContractError(f"controller current_input channel {name} missing from normalized previous_input")
            ctrl.current_input[name] = float(prev_input[name])
    try:
        ctrl.mpc._x0.master = base.state_vector_from_dict(env, state)
    except Exception:
        pass
    u0_vec, u0_meta = input_vector_strict(env, ctrl, prev_input)
    try:
        ctrl.mpc._u0.master = u0_vec
    except Exception as exc:
        raise ContractError("failed to assign strict _u0.master: " + repr(exc))
    if hasattr(ctrl, "current_reference"):
        for ref_name in list(ctrl.current_reference.keys()):
            if ref_name in shifted and shifted[ref_name]:
                ctrl.current_reference[ref_name] = shifted[ref_name][0]
    ctrl._tvp_data = copy.deepcopy(shifted)
    dist = base.state_distance(state, base.state_clean(env.control_system.current_state))
    if dist > base.STATE_DISTANCE_TOL:
        raise ContractError(f"context state not preserved after direct controller reset: distance {dist}")
    u0_arr = arr(getattr(ctrl.mpc, "_u0", []))
    return {
        "shifted_tvp": shifted,
        "goal_x": gx,
        "goal_y": gy,
        "goal_source": goal_source,
        "state_after_config": base.state_clean(env.control_system.current_state),
        "state_distance_after_config": dist,
        "previous_input_applied": copy.deepcopy(ctrl.current_input) if hasattr(ctrl, "current_input") else copy.deepcopy(prev_input),
        "previous_input_normalization": prev_norm,
        "u0_vector_meta": {**u0_meta, "shape": list(np.asarray(u0_vec).shape), "flat_size": int(arr(u0_vec).size), "norm2": float(np.linalg.norm(arr(u0_vec))) if arr(u0_vec).size else 0.0, "hash": arr_hash(u0_vec)},
        "u0_master_fingerprint": {"flat_size": int(u0_arr.size), "norm2": float(np.linalg.norm(u0_arr)) if u0_arr.size else 0.0, "hash": arr_hash(getattr(ctrl.mpc, "_u0", []))},
        "tvp_hash": canonical_hash(shifted),
        "tvp_lengths": {k: len(v) for k, v in shifted.items()},
    }


def apply_operational_patch() -> Dict[str, Any]:
    """Patch the imported v34 base module in-process; do not mutate frozen source."""
    base.load_contexts = load_contexts_strict  # type: ignore[assignment]
    base.configure_context_no_reset = configure_context_no_reset_strict  # type: ignore[assignment]
    return {
        "patch_scope": "in_process_monkey_patch_only_extension_file",
        "patched_symbols": ["base.load_contexts", "base.configure_context_no_reset"],
        "source_file": rel(Path(__file__).resolve()),
        "base_file": rel(Path(base.__file__).resolve()),
    }


def verify_plan_and_backup(args: argparse.Namespace) -> Dict[str, Any]:
    for p in [PLAN_READY, OPUS_REPORT, V34M_DONE, V34N_FAILED]:
        if not p.exists():
            raise ContractError("missing prerequisite " + rel(p))
    ready = read_json(PLAN_READY)
    if ready.get("request_id") != OPUS_REQUEST or ready.get("report_sha256") != OPUS_REPORT_SHA:
        raise ContractError(f"active PLAN_READY mismatch: request={ready.get('request_id')} sha={ready.get('report_sha256')}")
    if sha256(OPUS_REPORT) != OPUS_REPORT_SHA:
        raise ContractError("active Opus report sha mismatch")
    latest = OPUS_LATEST.read_text(encoding="utf-8", errors="replace") if OPUS_LATEST.exists() else ""
    if rel(OPUS_REPORT) not in latest:
        raise ContractError("Opus LATEST.md does not point to active report")
    v34m = read_json(V34M_DONE)
    if v34m.get("hard_pass") is not True:
        raise ContractError("v34m predecessor did not hard_pass")
    v34n_fail = read_json(V34N_FAILED)
    if int(v34n_fail.get("new_solver_calls_recorded", 0)) != 0:
        raise ContractError("v34n predecessor unexpectedly spent solver calls")
    proof = {
        "status": "verified_from_supervisor_context_not_revalidated_by_script",
        "time": args.backup_time,
        "commit": args.backup_commit,
        "package_sha256": args.backup_package_sha256,
        "package_bytes": int(args.backup_package_bytes),
        "source": "supervisor_user_context_current_prompt",
        "purpose": "input gate for v34o zero-solve loader repair after v34n failed before solver",
    }
    proof_path = BACKUP_DIR / f"backup_proof_{STAMP}_from_user_context_before_v34o_loader_gate.json"
    write_json(proof_path, proof)
    return {
        "active_lead_plan_ready": rel(PLAN_READY),
        "active_lead_report": rel(OPUS_REPORT),
        "active_lead_report_sha256": OPUS_REPORT_SHA,
        "active_lead_request": OPUS_REQUEST,
        "v34m_completed": rel(V34M_DONE),
        "v34n_failed": rel(V34N_FAILED),
        "v34n_failure_preserved_zero_solver": True,
        "backup_proof": {**proof, "path": rel(proof_path), "sha256": sha256(proof_path)},
    }


def install_zero_solve_guard(mpc: Any, record: Dict[str, Any]) -> None:
    def forbidden_solve(*_args: Any, **_kwargs: Any) -> Any:
        record["solver_call_attempts"] = int(record.get("solver_call_attempts", 0)) + 1
        raise ContractError("mpc.solve is forbidden in v34o zero-solve loader gate")
    mpc.solve = forbidden_solve


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


def write_summary_and_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    rows = raw["context_horizon_checks"]
    lines = [
        "# v34o / T-A61 zero-solve loader gate",
        "",
        f"UTC: `{raw['created_utc']}`. Operational repair gate under Opus plan `{raw['active_lead_report']}`.",
        "",
        "## Budget",
        "- Solver calls: `0`; plant steps: `0`; env.step after construction: `0`; env.reset after construction: `0`; training/refit: `0`; validation64: `0`; sealed test: `0`.",
        "",
        "## Headline",
        f"- passed: `{h['passed']}`; contexts loaded: `{h['contexts_loaded']}`; context/H checks: `{h['context_horizon_checks']}`; exceptions: `{h['exception_count']}`.",
        f"- Previous-input scalar hashes: `{h['previous_input_scalar_hashes']}`.",
        f"- _u0 norm range: `{h['u0_norm_min']}` to `{h['u0_norm_max']}`.",
        "",
        "## Context/H checks",
        "",
        "| context | H | tvp min len | previous input | _u0 shape | _u0 norm | solve attempts | ok |",
        "|---|---:|---:|---|---|---:|---:|---|",
    ]
    for r in rows:
        lines.append(f"| `{r['context_id']}` | {r['horizon']} | {r.get('tvp_min_len')} | `{r.get('previous_input')}` | `{r.get('u0_shape')}` | {r.get('u0_norm2')} | {r.get('solver_call_attempts')} | `{r.get('ok')}` |")
    lines += [
        "",
        "This gate fixes only the loader/configuration contract. It is not a solver-result, validation, selector, timing, or final-test result. If externally backed up, the approved next action is v34p/A13c-3 (<=6 low-level solver attempts) without another scientific audit.",
        f"Raw: `{rel(RUN_DIR/'raw.json')}`; completed: `{rel(RUN_DIR/'completed.json')}`; backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (RUN_DIR / "previous_input_normalization.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["context_id", "channel", "scalar", "rule", "raw_type", "flat_size", "raw_preview"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in raw["contexts_no_case_snapshot"]:
            norm = c.get("previous_input_normalization") or {}
            for ch, meta in (norm.get("channel_rules") or {}).items():
                w.writerow({"context_id": c.get("context_id"), "channel": ch, "scalar": meta.get("scalar"), "rule": meta.get("rule"), "raw_type": meta.get("raw_type"), "flat_size": meta.get("flat_size"), "raw_preview": json.dumps(clean(meta.get("raw_preview")), ensure_ascii=False)})
    block = f"""
<!-- {MARKER} -->
## v34o/T-A61 zero-solve loader gate

UTC: {raw['created_utc']}. Implemented T-A60 operational repair in a new extension file (no mutation of frozen v34 source): strict singleton scalarization for `previous_input`, and fail-loud required-channel checks before `_u0` assignment. Ran T-A61 loader gate on source242 and v19_c13 for H={{12,15,35}} with zero solver calls, zero plant/env steps, zero training/refit, validation64=0, sealed_test=0. passed={h['passed']}; context/H checks={h['context_horizon_checks']}; exception_count={h['exception_count']}; u0_norm_range=[{h['u0_norm_min']}, {h['u0_norm_max']}]. Evidence: `{rel(RUN_DIR/'summary.md')}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(RUN_DIR/'completed.json')}`, `{rel(RUN_DIR/'previous_input_normalization.csv')}`. Backup request: `{rel(BACKUP_REQUEST)}`. Next after verified backup: run v34p/A13c-3 <=6 solver attempts exactly as active Opus plan directs; no duplicate scientific audit needed if this gate remains passed.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([raw["created_utc"], NAME, raw["classification"], "T-A60/T-A61; H=12,15,35; contexts=source242,c13", "zero_solve_loader_gate_no_validation_no_test", h["context_horizon_checks"], 0, 0, 0, 0, False, rel(RUN_DIR/"completed.json"), MARKER])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        "# Continue state after v34o/T-A61 zero-solve loader gate\n\n"
        + f"UTC: {raw['created_utc']}\n\nHeadline: {json.dumps(clean(h), sort_keys=True)}\n\n"
        + f"Artifacts: {rel(RUN_DIR/'summary.md')}, {rel(RUN_DIR/'raw.json')}, {rel(RUN_DIR/'completed.json')}, {rel(RUN_DIR/'previous_input_normalization.csv')}\n\n"
        + f"Backup request: {rel(BACKUP_REQUEST)}\n\n"
        + "Next: obtain/verify external backup covering v34o and v34p sources plus artifacts. If backup is verified, run v34p/A13c-3 <=6 low-level solver attempts. Do not use validation64 or sealed test.\n",
        encoding="utf-8",
    )


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-time", required=True)
    ap.add_argument("--backup-commit", required=True)
    ap.add_argument("--backup-package-sha256", required=True)
    ap.add_argument("--backup-package-bytes", type=int, required=True)
    ap.add_argument("--i-accept-zero-solve-loader-gate", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        started = now_utc()
        gates = verify_plan_and_backup(args)
        patch_info = apply_operational_patch()
        write_json(RUN_DIR / "run_started.json", {"started_utc": started.isoformat(), "pid": os.getpid(), "method": NAME, "gates": gates, "patch_info": patch_info, "budget": {"solver_calls": 0, "plant_steps": 0, "env_step_calls_after_construction": 0, "env_reset_calls_after_construction": 0, "training_or_refit": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}})
        _, stage1_runner, _ = base.v1d.import_legacy_modules()
        preflight = stage1_runner.runtime_preflight()
        if not preflight.get("passed"):
            raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
        stage1_runner.base.v1.latency_verify()
        term_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
        terminals, terminal_receipts = stage1_runner.load_terminal_grid(term_protocol["terminal_grid_readiness_reused_from_v1"])
        if 15 not in terminals:
            raise ContractError("terminal grid missing H15")
        contexts = load_contexts_strict()
        context_summaries: List[Dict[str, Any]] = []
        checks: List[Dict[str, Any]] = []
        exceptions: List[Dict[str, Any]] = []
        for c in contexts:
            c_public = {k: v for k, v in c.items() if k != "case_snapshot"}
            context_summaries.append(c_public)
            for h in CHECK_HORIZONS:
                rec: Dict[str, Any] = {"context_id": c["context_id"], "state_label": c["state_label"], "horizon": int(h), "solver_call_attempts": 0, "ok": False}
                try:
                    shifted = base.shift_case_tvp(c["case_snapshot"], int(c["tvp_start_index"]), int(h))
                    rec["tvp_lengths"] = {k: len(v) for k, v in shifted.items()}
                    rec["tvp_min_len"] = min(rec["tvp_lengths"].values()) if rec["tvp_lengths"] else None
                    rec["tvp_hash"] = canonical_hash(shifted)
                    terminal, _, _ = base.terminal_for_mode(15, "V15_shared", terminals)
                    env = base.create_env(int(h), terminal)
                    ctrl = env.control_system.controller
                    install_zero_solve_guard(ctrl.mpc, rec)
                    meta = configure_context_no_reset_strict(env, c, int(h))
                    rec["previous_input"] = clean(meta["previous_input_applied"])
                    rec["previous_input_hash"] = canonical_hash(meta["previous_input_applied"])
                    rec["previous_input_normalization"] = meta["previous_input_normalization"]
                    rec["u0_shape"] = meta["u0_vector_meta"].get("shape")
                    rec["u0_norm2"] = meta["u0_vector_meta"].get("norm2")
                    rec["u0_hash"] = meta["u0_vector_meta"].get("hash")
                    rec["u0_master_fingerprint"] = meta.get("u0_master_fingerprint")
                    rec["state_distance_after_config"] = meta.get("state_distance_after_config")
                    rec["goal_source"] = meta.get("goal_source")
                    rec["goal_x"] = meta.get("goal_x")
                    rec["goal_y"] = meta.get("goal_y")
                    rec["ok"] = True
                except Exception as exc:
                    rec["ok"] = False
                    rec["error"] = repr(exc)
                    rec["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
                    exceptions.append(rec)
                checks.append(rec)
                write_json(RUN_DIR / "progress.json", {"checks_done": len(checks), "checks_expected": len(contexts) * len(CHECK_HORIZONS), "last_check": rec, "solver_calls": sum(int(x.get("solver_call_attempts", 0)) for x in checks), "validation64_bank_opened": False, "sealed_test_accessed": False})
                print(json.dumps(clean({"checks_done": len(checks), "context": c["context_id"], "H": h, "ok": rec["ok"], "solver_call_attempts": rec.get("solver_call_attempts", 0), "u0_norm2": rec.get("u0_norm2")}), sort_keys=True), flush=True)
        total_solve_attempts = int(sum(int(x.get("solver_call_attempts", 0)) for x in checks))
        u0_norms = [float(x["u0_norm2"]) for x in checks if x.get("u0_norm2") is not None]
        passed = bool(len(contexts) == 2 and len(checks) == 6 and all(x.get("ok") for x in checks) and total_solve_attempts == 0)
        headline = {
            "passed": passed,
            "contexts_loaded": len(contexts),
            "context_horizon_checks": len(checks),
            "exception_count": len(exceptions),
            "solver_calls": total_solve_attempts,
            "plant_steps": 0,
            "env_step_calls_after_construction": 0,
            "env_reset_calls_after_construction": 0,
            "training_or_refit": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
            "previous_input_scalar_hashes": {c["context_id"]: c.get("previous_input_normalization", {}).get("hash") for c in context_summaries},
            "u0_norm_min": min(u0_norms) if u0_norms else None,
            "u0_norm_max": max(u0_norms) if u0_norms else None,
        }
        created = now_utc()
        raw: Dict[str, Any] = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_operational_zero_solve_loader_gate_not_validation_not_test",
            "active_lead": "claude-opus-5-5",
            "active_lead_report": rel(OPUS_REPORT),
            "active_lead_report_sha256": OPUS_REPORT_SHA,
            "active_lead_request": OPUS_REQUEST,
            "hypothesis_frozen": "The v34n failure was an operational previous_input scalarization/defaulting defect; strict singleton scalarization and fail-loud channel checks can load both contexts and configure _u0 for H=12/15/35 without solver/plant access.",
            "gates": gates,
            "patch_info": patch_info,
            "runtime_preflight": preflight,
            "terminal_receipts": {str(k): v for k, v in terminal_receipts.items()},
            "contexts_no_case_snapshot": context_summaries,
            "context_horizon_checks": checks,
            "exceptions": exceptions,
            "headline": headline,
            "budget_declared": {"solver_calls": 0, "plant_steps": 0, "env_step_calls_after_construction": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "budget_actual": {"solver_calls": total_solve_attempts, "plant_steps": 0, "env_step_calls_after_construction": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "input_hashes": hash_existing([Path(__file__).resolve(), ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34p_nonconverged_objective_contract_probe_v0.py", Path(base.__file__).resolve(), PLAN_READY, OPUS_LATEST, OPUS_REPORT, V34M_DONE, V34N_FAILED]),
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
            "interpretation_limits": ["operational loader/configuration gate only", "no low-level solves", "no plant rollouts", "not validation64", "not sealed/final test", "does not test objective reconstruction G2"],
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_summary_and_docs(raw)
        write_json(BACKUP_REQUEST, {"request": "backup_after_v34o_loader_gate", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "T-A60/T-A61 source, zero-solve artifacts and state must be externally recoverable before v34p/A13c-3 solver calls", "must_cover": [rel(Path(__file__).resolve()), "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34p_nonconverged_objective_contract_probe_v0.py", rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "new_solver_calls": 0, "new_plant_steps": 0, "env_step_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "next_gate": "If backup verified and v34o passed, run v34p/A13c-3 <=6 solver calls under active Opus plan; no intervening scientific audit needed."})
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34p_nonconverged_objective_contract_probe_v0.py", STATE, BACKUP_REQUEST, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed = {"status": "complete", "passed": passed, "hard_pass": passed, "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": raw["budget_actual"], "headline": headline, "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "previous_input_normalization_csv": rel(RUN_DIR / "previous_input_normalization.csv"), "backup_request": rel(BACKUP_REQUEST), "next_if_backup_verified": "vehicle_true_variable_horizon_v34p_nonconverged_objective_contract_probe_v0.py", "hashes": hash_existing(files)}
        write_json(RUN_DIR / "completed.json", completed)
        print(json.dumps({"completed": rel(RUN_DIR/"completed.json"), "summary": rel(RUN_DIR/"summary.md"), "raw": rel(RUN_DIR/"raw.json"), "headline": headline, "backup_request": rel(BACKUP_REQUEST), "next_if_backup_verified": "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34p_nonconverged_objective_contract_probe_v0.py"}, sort_keys=True), flush=True)
        return 0 if passed else 2
    except Exception as exc:
        fail = {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_operational_zero_solve_loader_gate_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": {"solver_calls": 0, "plant_steps": 0, "env_step_calls_after_construction": 0, "env_reset_calls_after_construction": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}}
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(f"# v34o/T-A61 loader gate failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR/'failed.json')}\n\nNo solver, plant, validation64, sealed-test, training or refit access was requested. Preserve this operational failure and repair the exact missing primitive before A13c-3.\n", encoding="utf-8")
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR/"failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False, "solver_calls": 0}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
