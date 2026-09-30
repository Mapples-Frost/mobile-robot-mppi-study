#!/usr/bin/env python3
"""v34 fixed-context objective-vs-basin solver probe.

Astra-directed development diagnostic after the v33 bookkeeping preflight.  This
script performs at most 24 low-level MPC solves and no plant steps, no selector
training/refit, no validation64, and no sealed-test access.

Frozen design:
  contexts: source242 slot0 branch start; v19_c13 step1 previous_state from the
            V15/H35 v33 trace.
  horizons: source242 H15/H35; c13 H12/H35.
  terminals: zero, V15_shared, V35_shared.
  initializations: canonical, deterministic_goal_facing.
  budget: 2 * 2 * 3 * 2 = 24 solver calls; all failures count.

The implementation intentionally avoids env.reset and env.step after controller
construction.  It calls controller.get_action once per arm, with a patched
mpc.solve counter/capture hook.  If context reconstruction or objective
reconstruction is not safe, it fails closed or records the exact unavailable
primitive rather than adding unplanned rollouts/retries.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
REPRO_DIR = ROOT / "experiments/bohn2021_reproduction"
for _p in (AWS_DIR, REPRO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29 as v29  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402
import vehicle_true_variable_horizon_case5_smoke_v0_runner as case5  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34_objective_basin_solver_probe.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
BACKUP_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V34_OBJECTIVE_BASIN_SOLVER_PROBE_{STAMP}.json"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
ANALYSIS_READY = ROOT / "docs/bohn2021_takeover/astra_reviews/ANALYSIS_READY.json"
ASTRA_REPORT = ROOT / "docs/bohn2021_takeover/astra_reviews/20260930T072558Z.md"
TASK1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_bookkeeping_preflight_v0_20260930T074544Z/completed.json"
TASK1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_bookkeeping_preflight_v0_20260930T074544Z/raw.json"
TASK1_TERMINAL_CSV = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_bookkeeping_preflight_v0_20260930T074544Z/terminal_parameters_with_bias.csv"
D33_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0_20260930T054827Z"
D33_DONE = D33_DIR / "completed.json"
D33_MANIFEST = D33_DIR / "selected_state_manifest.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
CURRENT_ASTRA_REQUEST = "execution-result:20260930T072503_5188c5c7"
CURRENT_ASTRA_SHA = "c2cdc17ade124ba25d7d0ede80c3f6452c06bfcadaf1a957f87c564e3d2173b2"
REQUEST_ID = f"v34-objective-basin-solver-probe-{STAMP}"
MARKER = f"vehicle-v34-objective-basin-solver-probe-{STAMP}"

TERMINAL_MODES = ["zero", "V15_shared", "V35_shared"]
INIT_MODES = ["canonical", "goal_facing"]
MAX_SOLVES = 24
ACCEPT_RESIDUAL_TOL = 1e-5
RECON_REL_TOL = 1e-6
BASIN_REL_TOL = 1e-4
STATE_DISTANCE_TOL = 1e-9
RNG_SEED = 202609300845

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
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if hasattr(x, "item"):
        return clean(x.item())
    return x


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def arr(value: Any) -> np.ndarray:
    try:
        if hasattr(value, "full"):
            return np.asarray(value.full(), dtype=float).reshape(-1)
        if hasattr(value, "cat"):
            return np.asarray(value.cat, dtype=float).reshape(-1)
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def arr_hash(value: Any) -> str:
    a = arr(value)
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(np.asarray(value).reshape(-1)[0])
        return out if math.isfinite(out) else default
    except Exception:
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    out = finite_float(value, None)
    return default if out is None else float(out)


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        t = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def completed_ok(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing prerequisite {label}: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True and obj.get("status") not in ("complete", "completed"):
        raise ContractError(f"prerequisite not complete: {label}")
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=true in {label}")
    return obj


def verify_astra_and_gates(args: argparse.Namespace) -> Dict[str, Any]:
    ready = read_json(ANALYSIS_READY)
    if ready.get("request_id") != CURRENT_ASTRA_REQUEST:
        raise ContractError(f"Astra ready request mismatch: {ready.get('request_id')} != {CURRENT_ASTRA_REQUEST}")
    if not ASTRA_REPORT.exists() or sha256(ASTRA_REPORT) != CURRENT_ASTRA_SHA:
        raise ContractError("Astra report sha mismatch or missing")
    t1 = completed_ok(TASK1_DONE, "Task1 v33 bookkeeping preflight")
    t1_budget = t1.get("budget_actual") or {}
    if int(t1_budget.get("solver_calls", 0)) != 0 or int(t1_budget.get("plant_steps", 0)) != 0:
        raise ContractError("Task1 bookkeeping preflight did not have zero solver/plant budget")
    d33 = completed_ok(D33_DONE, "v33 terminal-H cross probe")
    if not (args.backup_time and args.backup_commit and args.backup_package_sha256):
        raise ContractError("verified backup context args are required before v34 unique solver evidence")
    bt = parse_time(args.backup_time)
    if bt is None:
        raise ContractError("backup_time not parseable")
    task1_time = parse_time(t1.get("created_utc"))
    if task1_time is not None and bt <= task1_time:
        raise ContractError("backup context does not postdate Task1 bookkeeping preflight")
    proof_path = BACKUP_DIR / f"backup_proof_{STAMP}_from_user_context_before_v34_objective_basin_probe.json"
    proof = {
        "status": "verified", "backup_verified": True, "time": bt.isoformat(),
        "commit": args.backup_commit, "remaining_changed_files": 0,
        "packages_this_run": [{"sha256": args.backup_package_sha256, "verification": "user_context_verified_backup", "bytes": args.backup_package_bytes}],
        "source": "supervisor_user_context_current_prompt",
        "purpose": "gate v34 fixed 24-call objective-vs-basin solver probe after Task1",
    }
    write_json(proof_path, proof)
    return {
        "analysis_ready": ready,
        "astra_report": rel(ASTRA_REPORT),
        "astra_report_sha256": CURRENT_ASTRA_SHA,
        "task1_completed": rel(TASK1_DONE),
        "task1_headline": t1.get("headline"),
        "v33_completed": rel(D33_DONE),
        "v33_headline": d33.get("headline"),
        "backup_proof": {**proof, "path": rel(proof_path), "sha256": sha256(proof_path)},
    }


def zero_like_terminal(pair: Tuple[Any, Any]) -> Tuple[List[Any], List[Any]]:
    weights, biases = pair
    return [np.zeros_like(np.asarray(w)) for w in weights], [np.zeros_like(np.asarray(b)) for b in biases]


def terminal_for_mode(h: int, mode: str, terminals: Mapping[int, Tuple[Any, Any]]) -> Tuple[Tuple[Any, Any], int, str]:
    if mode == "V15_shared":
        return terminals[15], 15, "V15_shared_all_H"
    if mode == "V35_shared":
        return terminals[35], 35, "V35_shared_all_H"
    if mode == "zero":
        base = h if h in terminals else 15
        return zero_like_terminal(terminals[base]), base, f"zero_like_V{base}"
    raise ContractError("unknown terminal mode " + mode)


def state_clean(state: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ("theta", "x", "y"):
        value = state.get(key)
        if isinstance(value, list) and value:
            value = value[0]
        out[key] = float(value)
    return out


def state_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    aa = state_clean(a); bb = state_clean(b)
    dtheta = math.atan2(math.sin(aa["theta"] - bb["theta"]), math.cos(aa["theta"] - bb["theta"]))
    return float(abs(dtheta) + abs(aa["x"] - bb["x"]) + abs(aa["y"] - bb["y"]))


def load_trace_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_spec(label: str, specs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    matches = [s for s in specs if str(s.get("state_label")) == label]
    if len(matches) != 1:
        raise ContractError(f"expected one spec for {label}, got {len(matches)}")
    return matches[0]


def find_c13_trace() -> Tuple[Path, List[Dict[str, Any]]]:
    matches = sorted(glob.glob(str(D33_DIR / "episodes" / "*v19_c13_V15_shared_H35_trueH35" / "trace.jsonl")))
    if len(matches) != 1:
        raise ContractError("could not uniquely locate v19_c13 V15/H35 v33 trace")
    p = Path(matches[0])
    rows = load_trace_jsonl(p)
    if len(rows) < 2:
        raise ContractError("c13 V15/H35 trace has fewer than two rows")
    return p, rows


def load_contexts() -> List[Dict[str, Any]]:
    specs = v29.build_state_specs()
    source242 = find_spec("v27_case09_slot0_early_risk", specs)
    c13 = find_spec("v19_c13", specs)
    c13_trace_path, c13_rows = find_c13_trace()
    c13_step0 = c13_rows[0]
    c13_step1 = c13_rows[1]
    contexts = [
        {
            "context_id": "source242_slot0_branch_start",
            "state_label": "v27_case09_slot0_early_risk",
            "source": "v29/v33 selected branch state, original branch start",
            "case_snapshot": copy.deepcopy(source242["case_snapshot"]),
            "branch_step": int(source242["branch_step"]),
            "tvp_start_index": int(source242["branch_step"]),
            "state": state_clean(source242["branch_previous_state"]),
            "previous_input": {"u_omega": 0.0, "u_s": 0.0},
            "previous_input_source": "Astra-specified v33 branch-reset zero-input semantics",
            "horizons": [15, 35],
        },
        {
            "context_id": "v19_c13_step1_after_V15_H35_common_state",
            "state_label": "v19_c13",
            "source": rel(c13_trace_path) + ": second trace row previous_state",
            "case_snapshot": copy.deepcopy(c13["case_snapshot"]),
            "branch_step": int(c13["branch_step"]),
            "tvp_start_index": int(c13["branch_step"]) + 1,
            "state": state_clean(c13_step1.get("previous_state") or {}),
            "previous_input": {k: float(v) for k, v in (c13_step0.get("input") or {}).items()},
            "previous_input_source": rel(c13_trace_path) + ": first trace row input",
            "horizons": [12, 35],
        },
    ]
    expected = {"theta": 0.08060330210484318, "x": 13.74497830467862, "y": 2.8552860589337556}
    if state_distance(contexts[1]["state"], expected) > 1e-6:
        raise ContractError("c13 reconstructed state does not match Astra-specified fixed state")
    return contexts


def shift_case_tvp(case: Mapping[str, Any], start: int, h: int) -> Dict[str, Any]:
    tvp = case.get("tvp") or {}
    if not isinstance(tvp, Mapping) or not tvp:
        raise ContractError("case snapshot lacks tvp")
    out: Dict[str, Any] = {}
    min_len = h + 1
    for name, values in tvp.items():
        if not isinstance(values, list) or len(values) <= start:
            raise ContractError(f"tvp {name} too short for start {start}")
        suffix = copy.deepcopy(values[start:])
        if len(suffix) < min_len:
            raise ContractError(f"tvp {name} suffix length {len(suffix)} < H+1={min_len}")
        out[str(name)] = suffix
    return out


def recursive_goal_find(obj: Any, x_names: Sequence[str], y_names: Sequence[str]) -> Tuple[Optional[float], Optional[float], str]:
    gx: Optional[float] = None; gy: Optional[float] = None; src = ""
    def visit(o: Any, path: str = "root") -> None:
        nonlocal gx, gy, src
        if gx is not None and gy is not None:
            return
        if isinstance(o, Mapping):
            keys = {str(k): k for k in o.keys()}
            for xn in x_names:
                if xn in keys and gx is None:
                    gx = finite_float(o[keys[xn]], None); src = path + "." + xn
            for yn in y_names:
                if yn in keys and gy is None:
                    gy = finite_float(o[keys[yn]], None); src = path + "." + yn
            if "goal" in keys and isinstance(o[keys["goal"]], Mapping):
                g = o[keys["goal"]]
                if gx is None:
                    gx = finite_float(g.get("x"), None)
                if gy is None:
                    gy = finite_float(g.get("y"), None)
                src = path + ".goal"
            for k, v in o.items():
                visit(v, path + "." + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o[:20]):
                visit(v, path + f"[{i}]")
    visit(obj)
    return gx, gy, src or "recursive_case_search"


def extract_goal_xy(case: Mapping[str, Any], shifted_tvp: Mapping[str, Any]) -> Tuple[float, float, str]:
    x_names = ["goal_x", "trajectory_goal_x", "x_goal", "target_x"]
    y_names = ["goal_y", "trajectory_goal_y", "y_goal", "target_y"]
    gx, gy, src = recursive_goal_find(case, x_names, y_names)
    if gx is not None and gy is not None:
        return float(gx), float(gy), src
    def first_tvp(names: Sequence[str]) -> Optional[float]:
        for exact in names:
            if exact in shifted_tvp:
                return finite_float((shifted_tvp[exact] or [None])[0], None)
        for k, vals in shifted_tvp.items():
            lk = k.lower()
            if any(n in lk for n in names):
                return finite_float((vals or [None])[0], None)
        return None
    gx = first_tvp(x_names); gy = first_tvp(y_names)
    if gx is not None and gy is not None:
        return float(gx), float(gy), "shifted_case_tvp_goal_fields"
    raise ContractError("could not reconstruct goal_x/goal_y from case/tvp; refusing to use observation heuristic")


def make_schedule(contexts: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    rows: List[Dict[str, Any]] = []
    idx = 0
    for c in contexts:
        cells = [(h, t, init) for h in c["horizons"] for t in TERMINAL_MODES for init in INIT_MODES]
        rng.shuffle(cells)
        for h, t, init in cells:
            rows.append({
                "execution_index": idx,
                "context_id": c["context_id"],
                "state_label": c["state_label"],
                "horizon": int(h),
                "terminal_mode": t,
                "initialization": init,
                "blocked_randomization_unit": f"v34|{c['context_id']}|H{h}",
            })
            idx += 1
    if len(rows) != MAX_SOLVES:
        raise ContractError(f"schedule length {len(rows)} != {MAX_SOLVES}")
    return rows


def get_struct_labels(obj: Any) -> List[str]:
    try:
        return [str(x) for x in obj.labels()]
    except Exception:
        return []


def parse_label(label: str) -> List[str]:
    s = label.strip().strip("[]")
    parts = []
    for p in s.split(","):
        p = p.strip().strip("'\"")
        if p:
            parts.append(p)
    return parts


def safe_struct_get(obj: Any, *idx: Any) -> Any:
    try:
        return obj[idx]
    except Exception:
        try:
            return obj.__getitem__(idx)
        except Exception:
            return None


def safe_struct_set(obj: Any, idx: Tuple[Any, ...], value: Any) -> bool:
    try:
        obj[idx] = value
        return True
    except Exception:
        try:
            obj.__setitem__(idx, value)
            return True
        except Exception:
            return False


def state_vector_from_dict(env: Any, state: Mapping[str, float]) -> Any:
    if hasattr(env.control_system, "get_state_vector"):
        return env.control_system.get_state_vector(dict(state))
    names = list(getattr(env.control_system.controller, "state_names", ["x", "y", "theta"]))
    return np.asarray([float(state[n]) for n in names], dtype=float).reshape((-1, 1))


def input_vector_from_dict(env: Any, inputs: Mapping[str, float]) -> Any:
    if hasattr(env.control_system, "get_input_vector"):
        try:
            return env.control_system.get_input_vector(dict(inputs))
        except Exception:
            pass
    names = list(getattr(env.control_system.controller, "input_names", ["u_omega", "u_s"]))
    return np.asarray([float(inputs.get(n, 0.0)) for n in names], dtype=float).reshape((-1, 1))


def configure_context_no_reset(env: Any, context: Mapping[str, Any], h: int) -> Dict[str, Any]:
    shifted = shift_case_tvp(context["case_snapshot"], int(context["tvp_start_index"]), h)
    gx, gy, goal_source = extract_goal_xy(context["case_snapshot"], shifted)
    ctrl = env.control_system.controller
    state = state_clean(context["state"])
    prev_input = {str(k): float(v) for k, v in (context.get("previous_input") or {}).items()}
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
    # Reset controller bookkeeping without env.reset/env.step.  Existing direct
    # branch code uses controller.reset; if this unexpectedly solves, the solve
    # hook below will catch the budget violation.
    try:
        ctrl.reset(env.control_system.current_state, reference=None, constraint=None, tvp=None)
    except Exception as exc:
        raise ContractError("controller.reset failed in no-env-reset context reconstruction: " + repr(exc))
    for attr in ("goal_x",):
        try: setattr(ctrl, attr, gx)
        except Exception: pass
        try: setattr(env, "trajectory_goal_x", gx)
        except Exception: pass
    for attr in ("goal_y",):
        try: setattr(ctrl, attr, gy)
        except Exception: pass
        try: setattr(env, "trajectory_goal_y", gy)
        except Exception: pass
    try:
        env.control_system.current_state.update(copy.deepcopy(state))
    except Exception:
        pass
    if hasattr(ctrl, "current_input"):
        for name in list(ctrl.current_input.keys()):
            ctrl.current_input[name] = float(prev_input.get(name, 0.0))
    try:
        ctrl.mpc._x0.master = state_vector_from_dict(env, state)
    except Exception:
        pass
    try:
        ctrl.mpc._u0.master = input_vector_from_dict(env, ctrl.current_input if hasattr(ctrl, "current_input") else prev_input)
    except Exception:
        pass
    if hasattr(ctrl, "current_reference"):
        for ref_name in list(ctrl.current_reference.keys()):
            if ref_name in shifted and shifted[ref_name]:
                ctrl.current_reference[ref_name] = shifted[ref_name][0]
    ctrl._tvp_data = copy.deepcopy(shifted)
    dist = state_distance(state, state_clean(env.control_system.current_state))
    if dist > STATE_DISTANCE_TOL:
        raise ContractError(f"context state not preserved after direct controller reset: distance {dist}")
    return {
        "shifted_tvp": shifted,
        "goal_x": gx,
        "goal_y": gy,
        "goal_source": goal_source,
        "state_after_config": state_clean(env.control_system.current_state),
        "state_distance_after_config": dist,
        "previous_input_applied": copy.deepcopy(ctrl.current_input) if hasattr(ctrl, "current_input") else prev_input,
        "tvp_hash": canonical_hash(shifted),
        "tvp_lengths": {k: len(v) for k, v in shifted.items()},
    }


def find_discount(mpc: Any) -> Tuple[float, str]:
    for name in ("discount_factor", "gamma", "discount", "vf_discount", "value_discount"):
        if hasattr(mpc, name):
            val = finite_float(getattr(mpc, name), None)
            if val is not None:
                return float(val), "mpc." + name
    try:
        settings = getattr(mpc, "settings", {})
        if isinstance(settings, Mapping):
            for name in ("discount_factor", "gamma", "discount"):
                if name in settings:
                    val = finite_float(settings[name], None)
                    if val is not None:
                        return float(val), "mpc.settings." + name
    except Exception:
        pass
    return 1.0, "default_1.0_not_found"


def predicted_unicycle(state: Mapping[str, float], goal_x: float, goal_y: float, h: int, dt_s: float) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    theta = float(state["theta"]); x = float(state["x"]); y = float(state["y"])
    states = [{"theta": theta, "x": x, "y": y}]
    controls: List[Dict[str, float]] = []
    for _ in range(h):
        desired = math.atan2(goal_y - y, goal_x - x)
        err = math.atan2(math.sin(desired - theta), math.cos(desired - theta))
        omega = max(-4.0, min(4.0, err / max(dt_s, 1e-9)))
        speed = max(0.0, min(5.0, 5.0 * max(0.0, math.cos(err))))
        controls.append({"u_omega": omega, "u_s": speed})
        theta = theta + omega * dt_s
        x = x + math.cos(theta) * speed * dt_s
        y = y + math.sin(theta) * speed * dt_s
        states.append({"theta": theta, "x": x, "y": y})
    return states, controls


def zero_guess(state: Mapping[str, float], h: int) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    return [dict(state_clean(state)) for _ in range(h + 1)], [{"u_omega": 0.0, "u_s": 0.0} for _ in range(h)]


def set_initial_guess(mpc: Any, env: Any, context_meta: Mapping[str, Any], init_mode: str, h: int) -> Dict[str, Any]:
    ctrl = env.control_system.controller
    state_names = list(getattr(ctrl, "state_names", ["x", "y", "theta"]))
    input_names = list(getattr(ctrl, "input_names", ["u_omega", "u_s"]))
    state = state_clean(context_meta["state_after_config"])
    if init_mode == "canonical":
        pred_states, pred_controls = zero_guess(state, h)
        rule = "repeat_initial_state_zero_control_zero_slack"
    elif init_mode == "goal_facing":
        dt_s = as_float(getattr(mpc, "t_step", None), 0.1)
        pred_states, pred_controls = predicted_unicycle(state, float(context_meta["goal_x"]), float(context_meta["goal_y"]), h, dt_s)
        rule = "deterministic_unicycle_goal_facing_clip_omega_pm4_speed_0_5"
    else:
        raise ContractError("unknown init mode " + init_mode)
    assigned = 0; attempted = 0; examples: List[str] = []
    for obj_name in ("opt_x_num",):
        obj = getattr(mpc, obj_name, None)
        if obj is None:
            continue
        for label in get_struct_labels(obj):
            parts = parse_label(label)
            if not parts:
                continue
            top = parts[0]
            try:
                nums = [int(p) for p in parts[1:] if str(p).lstrip("-").isdigit()]
            except Exception:
                nums = []
            if top == "_x":
                k = nums[0] if nums else 0
                var = parts[-1]
                if var in state_names and 0 <= k < len(pred_states):
                    attempted += 1
                    val = float(pred_states[k].get(var, state.get(var, 0.0)))
                    if safe_struct_set(obj, tuple(parts), val):
                        assigned += 1
                        if len(examples) < 5: examples.append(label)
            elif top == "_u":
                k = nums[0] if nums else 0
                var = parts[-1]
                if var in input_names and 0 <= k < len(pred_controls):
                    attempted += 1
                    val = float(pred_controls[k].get(var, 0.0))
                    if safe_struct_set(obj, tuple(parts), val):
                        assigned += 1
                        if len(examples) < 5: examples.append(label)
            elif top == "_z":
                attempted += 1
                if safe_struct_set(obj, tuple(parts), 0.0):
                    assigned += 1
                    if len(examples) < 5: examples.append(label)
    # Use identical zero dual guess across arms and initializations.
    lam_before = arr_hash(getattr(mpc, "lam_g_num", [])) if hasattr(mpc, "lam_g_num") else None
    try:
        mpc.lam_g_num = 0 * mpc.lam_g_num
    except Exception:
        pass
    return {
        "initialization": init_mode,
        "rule": rule,
        "state_names": state_names,
        "input_names": input_names,
        "assignments_attempted": attempted,
        "assignments_succeeded": assigned,
        "assignment_examples": examples,
        "initial_primal_hash": arr_hash(getattr(mpc, "opt_x_num", [])),
        "initial_dual_hash_before_zero": lam_before,
        "initial_dual_hash_after_zero": arr_hash(getattr(mpc, "lam_g_num", [])) if hasattr(mpc, "lam_g_num") else None,
        "first_three_pred_states": pred_states[:3],
        "first_three_pred_controls": pred_controls[:3],
    }


def residual_vs_bounds(values: Any, lb: Any, ub: Any) -> Optional[float]:
    x = arr(values); lo = arr(lb); hi = arr(ub)
    if x.size == 0 or lo.size != x.size or hi.size != x.size:
        return None
    return float(max(0.0, np.max(lo - x), np.max(x - hi)))


def capture_mpc_state(mpc: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "opt_x_hash": arr_hash(getattr(mpc, "opt_x_num", [])),
        "opt_x_unscaled_hash": arr_hash(getattr(mpc, "opt_x_num_unscaled", [])) if hasattr(mpc, "opt_x_num_unscaled") else None,
        "opt_p_hash": arr_hash(getattr(mpc, "opt_p_num", [])) if hasattr(mpc, "opt_p_num") else None,
        "lam_g_hash": arr_hash(getattr(mpc, "lam_g_num", [])) if hasattr(mpc, "lam_g_num") else None,
        "opt_x_size": int(arr(getattr(mpc, "opt_x_num", [])).size),
        "opt_g_size": int(arr(getattr(mpc, "opt_g_num", [])).size) if hasattr(mpc, "opt_g_num") else None,
        "bounds_hashes": {
            "opt_x_lb": arr_hash(getattr(mpc, "opt_x_lb", [])) if hasattr(mpc, "opt_x_lb") else None,
            "opt_x_ub": arr_hash(getattr(mpc, "opt_x_ub", [])) if hasattr(mpc, "opt_x_ub") else None,
            "opt_g_lb": arr_hash(getattr(mpc, "opt_g_lb", [])) if hasattr(mpc, "opt_g_lb") else None,
            "opt_g_ub": arr_hash(getattr(mpc, "opt_g_ub", [])) if hasattr(mpc, "opt_g_ub") else None,
        },
    }
    try:
        out["solver_options_hash"] = canonical_hash(getattr(mpc, "nlpsol_opts", {}))
    except Exception:
        out["solver_options_hash"] = None
    return out


def labeled_struct_values(obj: Any, max_items: int = 5000) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    labels = get_struct_labels(obj)
    for label in labels[:max_items]:
        parts = parse_label(label)
        try:
            out[label] = finite_float(obj[tuple(parts)], None)
        except Exception:
            pass
    return out


def p_value_map(mpc: Any) -> Dict[str, Any]:
    p = getattr(mpc, "opt_p_num", None)
    if p is None:
        return {}
    out = labeled_struct_values(p, max_items=2000)
    return out


def extract_goal_from_p(pmap: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    out = {"goal_x": None, "goal_y": None, "n_horizon": None}
    for label, val in pmap.items():
        parts = parse_label(label)
        name = parts[-1] if parts else label
        if name in out and val is not None:
            out[name] = float(val)
    return out


def extract_terminal_state(mpc: Any, h: int, state_names: Sequence[str]) -> Dict[str, Any]:
    source = getattr(mpc, "opt_x_num_unscaled", getattr(mpc, "opt_x_num", None))
    values: Dict[str, Optional[float]] = {n: None for n in state_names}
    raw_vec: List[float] = []
    try:
        xh = source["_x", h, 0, 0]
        raw_vec = arr(xh).tolist()
        if len(raw_vec) == len(state_names):
            for n, v in zip(state_names, raw_vec):
                values[n] = float(v)
    except Exception:
        pass
    if any(v is None for v in values.values()):
        for label in get_struct_labels(source):
            parts = parse_label(label)
            if len(parts) >= 4 and parts[0] == "_x":
                try:
                    if int(parts[1]) == h and parts[-1] in values:
                        values[parts[-1]] = finite_float(source[tuple(parts)], None)
                except Exception:
                    pass
    return {"state": values, "raw_vector": raw_vec, "source": "opt_x_num_unscaled" if hasattr(mpc, "opt_x_num_unscaled") else "opt_x_num"}


def call_casadi_scalar(fn: Any, candidates: Sequence[Tuple[Any, ...]]) -> Tuple[Optional[float], Dict[str, Any]]:
    errors: List[str] = []
    try:
        n_in = int(fn.n_in())
    except Exception:
        n_in = None
    for args in candidates:
        if n_in is not None and len(args) != n_in:
            continue
        try:
            val = fn(*args)
            f = finite_float(val, None)
            if f is not None:
                return float(f), {"ok": True, "n_args": len(args), "n_in": n_in}
        except Exception as exc:
            if len(errors) < 5:
                errors.append(f"{len(args)} args: {type(exc).__name__}: {exc}")
    return None, {"ok": False, "n_in": n_in, "errors": errors}


def reconstruct_stage_and_terminal(mpc: Any, h: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": False,
        "stage_lterm_total": None,
        "terminal_value_current_vf": None,
        "gamma": None,
        "gamma_source": None,
        "gamma_pow_H_terminal": None,
        "solver_objective": finite_float(getattr(mpc, "opt_f_num", None), None),
        "reconstructed_total_current_objective": None,
        "relative_error_vs_solver": None,
        "failure_reasons": [],
        "component_limits": {
            "slack_separate_component_available": False,
            "input_regularization_separate_component_available": False,
            "stage_lterm_may_include_slack_or_other_terms": True,
        },
    }
    gamma, gsrc = find_discount(mpc)
    out["gamma"] = gamma; out["gamma_source"] = gsrc
    try:
        p0 = mpc.opt_p_num["_p", 0]
    except Exception as exc:
        out["failure_reasons"].append("p0 unavailable: " + repr(exc)); p0 = None
    stage_values: List[float] = []
    if p0 is not None and hasattr(mpc, "lterm_fun"):
        for k in range(h):
            try:
                xk = mpc.opt_p_num["_x0"] if k == 0 else mpc.opt_x_num_unscaled["_x", k, 0, 0]
                uk = mpc.opt_x_num_unscaled["_u", k, 0]
                try:
                    zk = mpc.opt_x_num_unscaled["_z", k + 1, 0, -1]
                except Exception:
                    zk = mpc.opt_x_num_unscaled["_z", k, 0, -1]
                tvpk = mpc.opt_p_num["_tvp", k]
                val, meta = call_casadi_scalar(mpc.lterm_fun, [(xk, uk, zk, tvpk, p0), (xk, uk, tvpk, p0), (xk, uk, p0)])
                if val is None:
                    out["failure_reasons"].append(f"lterm k={k} unavailable: {meta}")
                    break
                stage_values.append((gamma ** k) * float(val))
            except Exception as exc:
                out["failure_reasons"].append(f"lterm k={k} exception: {repr(exc)}")
                break
    else:
        out["failure_reasons"].append("lterm_fun or p0 unavailable")
    if len(stage_values) == h:
        out["stage_lterm_total"] = float(math.fsum(stage_values))
    try:
        xh = mpc.opt_x_num_unscaled["_x", h, 0, 0]
        tvph = mpc.opt_p_num["_tvp", h]
        val, meta = call_casadi_scalar(mpc.vf_fun, [(xh, p0), (xh, tvph, p0), (arr(xh), arr(p0)), (arr(xh), arr(tvph), arr(p0)), (xh,), (arr(xh),)])
        out["terminal_vf_call_meta"] = meta
        if val is not None:
            out["terminal_value_current_vf"] = float(val)
            out["gamma_pow_H_terminal"] = float((gamma ** h) * val)
    except Exception as exc:
        out["failure_reasons"].append("vf current exception: " + repr(exc))
    if out["stage_lterm_total"] is not None and out["gamma_pow_H_terminal"] is not None:
        total = float(out["stage_lterm_total"] + out["gamma_pow_H_terminal"])
        out["reconstructed_total_current_objective"] = total
        solver_obj = out["solver_objective"]
        if solver_obj is not None:
            out["relative_error_vs_solver"] = abs(total - solver_obj) / max(1.0, abs(solver_obj))
        out["available"] = True
    return out


def parse_terminal_coeff_csv() -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    if not TASK1_TERMINAL_CSV.exists():
        return {}
    with TASK1_TERMINAL_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = {k.lower(): k for k in row.keys() if k is not None}
            hval = None
            for hn in ("terminal_source_horizon", "source_horizon", "horizon", "terminal_horizon", "h"):
                if hn in keys:
                    hval = row[keys[hn]]; break
            var = None
            for vn in ("variable", "input", "input_name", "name", "feature"):
                if vn in keys:
                    var = row[keys[vn]]; break
            if hval is None or var is None:
                continue
            mode = "V" + str(int(float(hval))) + "_shared"
            def gv(names: Sequence[str]) -> Optional[float]:
                for n in names:
                    if n in keys:
                        return finite_float(row[keys[n]], None)
                return None
            linear = gv(["linear", "linear_coefficient", "coef_linear", "a"])
            quadratic = gv(["quadratic", "quadratic_coefficient", "coef_quadratic", "q"])
            bias = gv(["bias", "intercept", "b"])
            out[mode][str(var)] = {"linear": 0.0 if linear is None else float(linear), "quadratic": 0.0 if quadratic is None else float(quadratic), "bias": 0.0 if bias is None else float(bias)}
    return {k: dict(v) for k, v in out.items()}


def coefficient_components(mode: str, terminal_state: Mapping[str, Any], p_goals: Mapping[str, Any], coeffs: Mapping[str, Mapping[str, Mapping[str, float]]]) -> Dict[str, Any]:
    if mode == "zero":
        return {"available": True, "components": {"bias": 0.0, "theta": 0.0, "x": 0.0, "y": 0.0, "goal_x": 0.0, "goal_y": 0.0}, "sum": 0.0, "source": "zero_terminal"}
    c = coeffs.get(mode) or {}
    if not c:
        return {"available": False, "reason": "no parsed coefficients for " + mode}
    values: Dict[str, Optional[float]] = {
        "theta": terminal_state.get("theta"),
        "x": terminal_state.get("x"),
        "y": terminal_state.get("y"),
        "goal_x": p_goals.get("goal_x"),
        "goal_y": p_goals.get("goal_y"),
    }
    comps: Dict[str, float] = {}
    bias_once = 0.0
    for var, par in c.items():
        lname = var.strip().lower()
        z = values.get(lname)
        if lname in ("bias", "intercept"):
            bias_once += float(par.get("bias", 0.0) + par.get("linear", 0.0))
            continue
        if z is None:
            continue
        comps[lname] = float(par.get("linear", 0.0) * float(z) + par.get("quadratic", 0.0) * float(z) * float(z))
        if abs(par.get("bias", 0.0)) > 0 and bias_once == 0.0:
            bias_once = float(par.get("bias", 0.0))
    comps["bias"] = bias_once
    return {"available": bool(comps), "components": comps, "sum": float(math.fsum(comps.values())), "source": rel(TASK1_TERMINAL_CSV)}


def create_env(h: int, terminal: Tuple[Any, Any]) -> Any:
    env = case5.make_true_horizon_env(int(h))
    env.set_value_function_weights_and_biases(*terminal)
    def forbidden_reset(*_args: Any, **_kwargs: Any) -> Any:
        raise ContractError("env.reset is forbidden in v34 solver probe")
    def forbidden_step(*_args: Any, **_kwargs: Any) -> Any:
        raise ContractError("env.step is forbidden in v34 solver probe")
    env.reset = forbidden_reset
    env.step = forbidden_step
    return env


def solve_arm(row: Mapping[str, Any], context: Mapping[str, Any], terminals: Mapping[int, Tuple[Any, Any]], coeffs: Mapping[str, Any]) -> Dict[str, Any]:
    h = int(row["horizon"]); mode = str(row["terminal_mode"]); init_mode = str(row["initialization"])
    terminal, term_h, term_note = terminal_for_mode(h, mode, terminals)
    env = create_env(h, terminal)
    ctrl = env.control_system.controller
    mpc = ctrl.mpc
    context_meta = configure_context_no_reset(env, context, h)
    init_meta = set_initial_guess(mpc, env, context_meta, init_mode, h)
    solve_events: List[Dict[str, Any]] = []
    original_solve = mpc.solve
    def counted_solve(*args: Any, **kwargs: Any) -> Any:
        if len(solve_events) >= 1:
            raise ContractError("unexpected second mpc.solve call inside a single v34 arm")
        pre = capture_mpc_state(mpc)
        t0 = time.perf_counter()
        try:
            ret = original_solve(*args, **kwargs)
            ok_exc = None
        except Exception as exc:
            ret = None
            ok_exc = repr(exc)
        elapsed = time.perf_counter() - t0
        post = capture_mpc_state(mpc)
        stats = copy.deepcopy(getattr(mpc, "solver_stats", {}))
        event = {
            "pre": pre,
            "post": post,
            "solver_wall_s": float(elapsed),
            "solver_exception": ok_exc,
            "solver_stats": stats,
            "return_status": stats.get("return_status"),
            "success": bool(stats.get("success", False)),
            "iterations": stats.get("iter_count") or stats.get("iterations"),
            "objective_opt_f_num": finite_float(getattr(mpc, "opt_f_num", None), None),
        }
        solve_events.append(event)
        if ok_exc is not None:
            raise ContractError("mpc.solve failed: " + ok_exc)
        return ret
    mpc.solve = counted_solve
    started = time.perf_counter()
    action = ctrl.get_action(state_clean(context["state"]), h, tvp_values=copy.deepcopy(context_meta["shifted_tvp"]))
    wall_s = float(time.perf_counter() - started)
    if len(solve_events) != 1:
        raise ContractError(f"expected exactly one solve, got {len(solve_events)}")
    ev = solve_events[0]
    constraint_resid = residual_vs_bounds(getattr(mpc, "opt_g_num", []), getattr(mpc, "opt_g_lb", []), getattr(mpc, "opt_g_ub", []))
    bound_resid = residual_vs_bounds(getattr(mpc, "opt_x_num", []), getattr(mpc, "opt_x_lb", []), getattr(mpc, "opt_x_ub", []))
    ev["constraint_residual"] = constraint_resid
    ev["bound_residual"] = bound_resid
    accepted = bool(ev.get("success")) and (constraint_resid is None or constraint_resid <= ACCEPT_RESIDUAL_TOL) and (bound_resid is None or bound_resid <= ACCEPT_RESIDUAL_TOL) and ev.get("objective_opt_f_num") is not None
    state_names = list(getattr(ctrl, "state_names", ["x", "y", "theta"]))
    terminal_state_obj = extract_terminal_state(mpc, h, state_names)
    pmap = p_value_map(mpc)
    pgoals = extract_goal_from_p(pmap)
    recon = reconstruct_stage_and_terminal(mpc, h)
    term_components: Dict[str, Any] = {}
    for m in TERMINAL_MODES:
        term_components[m] = coefficient_components(m, terminal_state_obj["state"], pgoals, coeffs)
    first_control = clean(action)
    if hasattr(ctrl, "current_input"):
        first_control = copy.deepcopy(ctrl.current_input)
    return {
        "arm_id": f"ctx={row['context_id']}|H{h}|term={mode}|init={init_mode}",
        "execution_index": int(row["execution_index"]),
        "context_id": row["context_id"],
        "state_label": row["state_label"],
        "horizon": h,
        "terminal_mode": mode,
        "terminal_source_horizon": int(term_h),
        "terminal_note": term_note,
        "initialization": init_mode,
        "solve_calls": 1,
        "plant_steps": 0,
        "env_reset_calls": 0,
        "controller_get_action_wall_s": wall_s,
        "accepted": accepted,
        "first_control": first_control,
        "terminal_state": terminal_state_obj,
        "context_meta": {k: v for k, v in context_meta.items() if k != "shifted_tvp"},
        "initialization_meta": init_meta,
        "solver_event": ev,
        "p_values": pmap,
        "p_goal_summary": pgoals,
        "objective_reconstruction_current": recon,
        "terminal_coeff_components_best_effort": term_components,
        "hashes": {
            "opt_p_labeled_hash": canonical_hash(pmap),
            "terminal_state_hash": canonical_hash(terminal_state_obj),
            "first_control_hash": canonical_hash(first_control),
        },
    }


def build_vf_evaluator_cache(terminals: Mapping[int, Tuple[Any, Any]]) -> Dict[Tuple[int, str], Any]:
    cache: Dict[Tuple[int, str], Any] = {}
    for h in (12, 15, 35):
        for mode in TERMINAL_MODES:
            if mode == "zero":
                cache[(h, mode)] = None
            else:
                terminal, _, _ = terminal_for_mode(h, mode, terminals)
                env = create_env(h, terminal)
                cache[(h, mode)] = env.control_system.controller.mpc.vf_fun
    return cache


def cross_evaluate(arms: Sequence[Mapping[str, Any]], vf_cache: Mapping[Tuple[int, str], Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for arm in arms:
        h = int(arm["horizon"])
        recon = arm.get("objective_reconstruction_current") or {}
        stage_total = recon.get("stage_lterm_total")
        gamma = recon.get("gamma") if recon.get("gamma") is not None else 1.0
        terminal_state = arm.get("terminal_state") or {}
        x_vec = terminal_state.get("raw_vector") or []
        # Fall back to state dict order theta,x,y if raw vector unavailable.  The
        # call meta below makes any failed signature explicit.
        if not x_vec:
            st = terminal_state.get("state") or {}
            x_vec = [st.get("x"), st.get("y"), st.get("theta")]
        pmap = arm.get("p_values") or {}
        p_arr_values = [v for _, v in sorted(pmap.items()) if v is not None]
        row: Dict[str, Any] = {"arm_id": arm["arm_id"], "context_id": arm["context_id"], "horizon": h, "candidate_terminal_mode": arm["terminal_mode"], "candidate_initialization": arm["initialization"], "accepted": arm["accepted"], "stage_lterm_total": stage_total, "objectives": {}}
        for mode in TERMINAL_MODES:
            if mode == "zero":
                vf_val = 0.0; meta = {"ok": True, "source": "zero"}
            else:
                fn = vf_cache.get((h, mode))
                if fn is None:
                    vf_val = None; meta = {"ok": False, "reason": "missing vf cache"}
                else:
                    # Candidate p array is only a fallback; successful exact calls
                    # are recorded by n_args/n_in metadata.  No solver call occurs.
                    val, meta = call_casadi_scalar(fn, [(np.asarray(x_vec, dtype=float).reshape((-1, 1)), np.asarray(p_arr_values, dtype=float).reshape((-1, 1))), (np.asarray(x_vec, dtype=float).reshape(-1), np.asarray(p_arr_values, dtype=float).reshape(-1)), (np.asarray(x_vec, dtype=float).reshape((-1, 1)),), (np.asarray(x_vec, dtype=float).reshape(-1),)])
                    vf_val = val
            gamma_terminal = None if vf_val is None else float((float(gamma) ** h) * float(vf_val))
            centered_coeff = (arm.get("terminal_coeff_components_best_effort") or {}).get(mode)
            row["objectives"][mode] = {
                "vf_value": vf_val,
                "gamma_pow_H_terminal": gamma_terminal,
                "cross_objective_stage_plus_terminal": None if stage_total is None or gamma_terminal is None else float(stage_total + gamma_terminal),
                "vf_call_meta": meta,
                "centered_coeff_components_best_effort": centered_coeff,
            }
        out.append(row)
    return out


def analyze(arms: Sequence[Mapping[str, Any]], cross: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_ctx_h: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for arm in arms:
        by_ctx_h[(str(arm["context_id"]), int(arm["horizon"]))].append(arm)
    cross_by_arm = {r["arm_id"]: r for r in cross}
    groups: List[Dict[str, Any]] = []
    basin_support = 0
    ranking_reversal = 0
    unavailable_cross = 0
    recon_errors: List[float] = []
    recon_unavailable = 0
    for arm in arms:
        rec = arm.get("objective_reconstruction_current") or {}
        if rec.get("relative_error_vs_solver") is not None:
            recon_errors.append(float(rec["relative_error_vs_solver"]))
        else:
            recon_unavailable += 1
    for (ctx, h), rows in sorted(by_ctx_h.items()):
        group: Dict[str, Any] = {"context_id": ctx, "horizon": h, "accepted_count": sum(1 for r in rows if r.get("accepted")), "terminal_init_rows": [], "basin_pairs": [], "objective_best_candidates": {}}
        for terminal in TERMINAL_MODES:
            pair = [r for r in rows if r["terminal_mode"] == terminal]
            by_init = {r["initialization"]: r for r in pair}
            can = by_init.get("canonical"); alt = by_init.get("goal_facing")
            pair_row = {"terminal_mode": terminal, "canonical_arm": can["arm_id"] if can else None, "goal_facing_arm": alt["arm_id"] if alt else None, "basin_significant_same_objective": False, "delta_canonical_minus_goal_facing_same_J": None}
            if can is not None and alt is not None and can.get("accepted") and alt.get("accepted"):
                cobj = ((cross_by_arm.get(can["arm_id"]) or {}).get("objectives") or {}).get(terminal, {}).get("cross_objective_stage_plus_terminal")
                aobj = ((cross_by_arm.get(alt["arm_id"]) or {}).get("objectives") or {}).get(terminal, {}).get("cross_objective_stage_plus_terminal")
                if cobj is None or aobj is None:
                    unavailable_cross += 1
                else:
                    delta = float(cobj - aobj)
                    pair_row["delta_canonical_minus_goal_facing_same_J"] = delta
                    if delta > BASIN_REL_TOL * max(1.0, abs(float(cobj))):
                        pair_row["basin_significant_same_objective"] = True
                        basin_support += 1
            group["basin_pairs"].append(pair_row)
        for objective in TERMINAL_MODES:
            scored: List[Tuple[float, str, str, str]] = []
            missing = 0
            for r in rows:
                if not r.get("accepted"):
                    continue
                val = ((cross_by_arm.get(r["arm_id"]) or {}).get("objectives") or {}).get(objective, {}).get("cross_objective_stage_plus_terminal")
                if val is None:
                    missing += 1
                    continue
                scored.append((float(val), str(r["terminal_mode"]), str(r["initialization"]), str(r["arm_id"])))
            scored.sort()
            if missing:
                unavailable_cross += missing
            group["objective_best_candidates"][objective] = None if not scored else {"J": scored[0][0], "candidate_terminal_mode": scored[0][1], "candidate_initialization": scored[0][2], "arm_id": scored[0][3], "ranked": [{"J": s[0], "terminal": s[1], "init": s[2], "arm_id": s[3]} for s in scored]}
        best_terms = [v["candidate_terminal_mode"] for v in group["objective_best_candidates"].values() if isinstance(v, Mapping)]
        if len(set(best_terms)) > 1:
            ranking_reversal += 1
        groups.append(group)
    solver_status_counts: Dict[str, int] = {}
    for arm in arms:
        st = str((arm.get("solver_event") or {}).get("return_status"))
        solver_status_counts[st] = solver_status_counts.get(st, 0) + 1
    return {
        "headline": {
            "solver_calls": int(sum(int(a.get("solve_calls", 0)) for a in arms)),
            "plant_steps": int(sum(int(a.get("plant_steps", 0)) for a in arms)),
            "env_reset_calls": int(sum(int(a.get("env_reset_calls", 0)) for a in arms)),
            "arms": len(arms),
            "accepted_arms": sum(1 for a in arms if a.get("accepted")),
            "solver_status_counts": solver_status_counts,
            "objective_reconstruction_available_arms": len([a for a in arms if (a.get("objective_reconstruction_current") or {}).get("available")]),
            "objective_reconstruction_max_rel_error": max(recon_errors) if recon_errors else None,
            "objective_reconstruction_unavailable_arms": recon_unavailable,
            "basin_support_pairs": basin_support,
            "objective_ranking_reversal_groups": ranking_reversal,
            "cross_scoring_missing_cells_count": unavailable_cross,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        },
        "groups": groups,
        "executor_caution": "If objective reconstruction is unavailable or exceeds tolerance, mechanism attribution must stop at engineering localization; do not use v34 as validation/final evidence.",
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["analysis"]["headline"]
    lines = [
        "# v34 objective-vs-basin fixed-context solver probe",
        "",
        f"UTC: `{raw['created_utc']}`. Astra Task2 development diagnostic; no env.reset/env.step after controller construction, no plant steps, no training/refit, no validation64, no sealed test.",
        "",
        "## Budget/accounting",
        "",
        f"- Solver calls: `{h['solver_calls']}` / `{raw['budget_declared']['solver_call_cap']}`.",
        f"- Arms: `{h['arms']}`; accepted arms: `{h['accepted_arms']}`; solver statuses: `{h['solver_status_counts']}`.",
        f"- Plant steps: `{h['plant_steps']}`; env.reset calls: `{h['env_reset_calls']}`; training/refit: `0`.",
        f"- validation64 opened: `{h['validation64_bank_opened']}`; sealed test accessed: `{h['sealed_test_accessed']}`.",
        "",
        "## Mechanism readout (executor numeric, development-only)",
        "",
        f"- Objective reconstruction available arms: `{h['objective_reconstruction_available_arms']}`; max relative error: `{h['objective_reconstruction_max_rel_error']}`; unavailable arms: `{h['objective_reconstruction_unavailable_arms']}`.",
        f"- Same-objective basin-support pairs (goal-facing J lower than canonical by > {BASIN_REL_TOL} relative): `{h['basin_support_pairs']}`.",
        f"- Context/H groups whose best candidate terminal differs across common objectives: `{h['objective_ranking_reversal_groups']}`.",
        f"- Missing cross-scoring cells: `{h['cross_scoring_missing_cells_count']}`.",
        "",
        "## Per context/H group",
        "",
        "| context | H | accepted | basin pairs | best under zero/V15/V35 objectives |",
        "|---|---:|---:|---|---|",
    ]
    for g in raw["analysis"]["groups"]:
        best = {m: None if not isinstance(g["objective_best_candidates"].get(m), Mapping) else {"terminal": g["objective_best_candidates"][m]["candidate_terminal_mode"], "init": g["objective_best_candidates"][m]["candidate_initialization"], "J": round(float(g["objective_best_candidates"][m]["J"]), 6)} for m in TERMINAL_MODES}
        pairs = [{"term": p["terminal_mode"], "delta": p["delta_canonical_minus_goal_facing_same_J"], "sig": p["basin_significant_same_objective"]} for p in g["basin_pairs"]]
        lines.append(f"| `{g['context_id']}` | {g['horizon']} | {g['accepted_count']} | `{pairs}` | `{best}` |")
    lines += [
        "",
        "## Limits",
        "",
        "This is a targeted opened-development-state optimization diagnostic. It is not population validation, not validation64, not final test, and not an adaptive closed-loop selector/timing claim. Objective decomposition is recorded as best-effort and gated by the reconstruction error in raw.json.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_docs(raw: Mapping[str, Any]) -> None:
    h = raw["analysis"]["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-30 v34 objective-vs-basin fixed-context solver probe

UTC: {raw['created_utc']}. Executed Astra Task2 as a development-only fixed-context low-level solver diagnostic: 2 contexts × 2 horizons × 3 terminal contracts × 2 initializations = {h['solver_calls']} solve attempts, with plant_steps={h['plant_steps']}, env_reset_calls={h['env_reset_calls']}, training/refit=0, validation64=false, sealed_test=false. Accepted arms={h['accepted_arms']}; objective reconstruction available arms={h['objective_reconstruction_available_arms']} with max rel error={h['objective_reconstruction_max_rel_error']}; basin-support pairs={h['basin_support_pairs']}; objective ranking reversal groups={h['objective_ranking_reversal_groups']}; missing cross-scoring cells={h['cross_scoring_missing_cells_count']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`. New Astra request `{REQUEST_ID}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / doc, MARKER, block)
    response = f"""
<!-- {MARKER} -->
## v34 objective-vs-basin solver-probe response to Astra report

Updated by GPT-5.5 executor at `{raw['created_utc']}` for Astra report `{rel(ASTRA_REPORT)}` (`{CURRENT_ASTRA_SHA}`) and stable recommendations `A11_training_failure_modes_need_separation` / `A6_strong_fixed_H_and_terminal_opportunity_not_closed`. Task1 was already passed and externally backed up, so this run executed the authorized fixed 24-call Task2. No validation64, no sealed test, no selector training/refit.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A11_training_failure_modes_need_separation`, `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted and executed | `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'summary.md')}` | Fixed-context objective-vs-basin probe completed with headline `{h}`. Await Astra interpretation before selector search, new-source acquisition, validation64, or training changes. |
| `A12_registry_backup_schema_contract` | accepted/continued | pre-run proof from supervisor context and post-run backup request `{rel(BACKUP_REQUEST)}` | Requires external backup covering v34 code/raw/docs before further unique scientific simulation/training. |
| `A4`, `A7`, `A8` | still open | v34 has no adaptive online selector and only two targeted opened contexts | Carry forward; do not treat v34 as population or timing-generalization evidence. |
"""
    append_if_missing(RESPONSE_LOG, MARKER, response)
    req = {
        "request_id": REQUEST_ID,
        "created": raw["created_utc"],
        "status": "analysis_requested",
        "trigger": "v34 objective-vs-basin fixed-context solver probe completed",
        "experiment_id": NAME,
        "covered_prior_report": rel(ASTRA_REPORT),
        "question": "Analyze the v34 24-call fixed-context solver probe. Decide whether the next scientific action should target terminal objective/value representation, canonical warm-start/basin recovery, context reconstruction/objective-decomposition repair, or another bounded action. Preserve that v34 is development-only and does not authorize validation64 or sealed-test access.",
        "evidence_paths": [rel(RUN_DIR / "summary.md"), rel(RUN_DIR / "raw.json"), rel(RUN_DIR / "completed.json"), rel(ASTRA_REPORT), rel(RESPONSE_LOG)],
        "budget_actual": raw["budget_actual"],
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "plant_steps": 0, "training_or_refit": 0},
        "operational_note": f"Post-run backup required via {rel(BACKUP_REQUEST)} before further unique science.",
    }
    write_json(NEXT_REVIEW_REQUEST, req)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    tail = reg.read_text(encoding="utf-8", errors="replace")[-100000:] if reg.exists() else ""
    if MARKER not in tail:
        with reg.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([raw["created_utc"], NAME, raw["classification"], f"RNG_SEED={RNG_SEED}", "opened_development_fixed_context_objective_basin_solver_probe_no_validation_no_test", h["arms"], 0, h["solver_calls"], 0, 0, False, rel(RUN_DIR / "completed.json"), MARKER])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v34 objective-vs-basin solver probe\n\nUTC: {raw['created_utc']}\n\nHeadline: {h}\n\nArtifacts: {rel(RUN_DIR / 'summary.md')}, {rel(RUN_DIR / 'raw.json')}, {rel(RUN_DIR / 'completed.json')}\n\nAstra request pending: {REQUEST_ID}\n\nNext: verify external backup for {rel(BACKUP_REQUEST)}; read matching Astra analysis before starting selector training/new source acquisition/validation64/final-test work.\n", encoding="utf-8")


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-time", required=True)
    ap.add_argument("--backup-commit", required=True)
    ap.add_argument("--backup-package-sha256", required=True)
    ap.add_argument("--backup-package-bytes", type=int, default=0)
    ap.add_argument("--i-accept-v34-fixed-24-solve-budget", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        gates = verify_astra_and_gates(args)
        started = now_utc()
        write_json(RUN_DIR / "run_started.json", {"started_utc": started.isoformat(), "pid": os.getpid(), "method": NAME, "gates": gates, "budget_cap_solver_calls": MAX_SOLVES, "plant_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
        _, stage1_runner, _ = v1d.import_legacy_modules()
        preflight = stage1_runner.runtime_preflight()
        if not preflight.get("passed"):
            raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
        stage1_runner.base.v1.latency_verify()
        term_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
        terminals, terminal_receipts = stage1_runner.load_terminal_grid(term_protocol["terminal_grid_readiness_reused_from_v1"])
        for h in (15, 35):
            if h not in terminals:
                raise ContractError(f"terminal grid missing H{h}")
        contexts = load_contexts()
        schedule = make_schedule(contexts)
        write_json(RUN_DIR / "frozen_schedule.json", {"created_utc": now_utc().isoformat(), "contexts": [{k: v for k, v in c.items() if k != "case_snapshot"} for c in contexts], "schedule": schedule, "solver_call_cap": MAX_SOLVES, "plant_steps": 0, "env_reset_calls": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
        coeffs = parse_terminal_coeff_csv()
        write_json(RUN_DIR / "parsed_terminal_coefficients.json", coeffs)
        context_by_id = {c["context_id"]: c for c in contexts}
        arms: List[Dict[str, Any]] = []
        for row in schedule:
            if len(arms) >= MAX_SOLVES:
                raise ContractError("solver call cap would be exceeded")
            arm = solve_arm(row, context_by_id[str(row["context_id"])], terminals, coeffs)
            arms.append(arm)
            solves = int(sum(int(a.get("solve_calls", 0)) for a in arms))
            if solves > MAX_SOLVES:
                raise ContractError("solver call cap exceeded")
            write_json(RUN_DIR / "progress.json", {"arms_done": len(arms), "arms_expected": MAX_SOLVES, "solver_calls": solves, "plant_steps": 0, "env_reset_calls": 0, "last_arm": arm, "validation64_bank_opened": False, "sealed_test_accessed": False, "training_or_refit": 0})
            print(json.dumps({"arms_done": len(arms), "solver_calls": solves, "arm": arm["arm_id"], "accepted": arm["accepted"], "status": arm["solver_event"].get("return_status"), "obj": arm["solver_event"].get("objective_opt_f_num")}, sort_keys=True), flush=True)
        vf_cache = build_vf_evaluator_cache(terminals)
        cross = cross_evaluate(arms, vf_cache)
        analysis = analyze(arms, cross)
        created = now_utc()
        raw = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_fixed_context_objective_basin_solver_probe_not_validation_not_test",
            "hypothesis_frozen": "Shared fixed context/H with terminal contracts and two deterministic initializations separates terminal objective preference from canonical warm-start/basin effects before selector training or validation.",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "plant_steps": 0,
            "env_reset_calls": 0,
            "budget_declared": {"solver_call_cap": MAX_SOLVES, "plant_steps": 0, "env_reset_calls": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "budget_actual": {"arms": len(arms), "solver_calls": int(sum(int(a.get("solve_calls", 0)) for a in arms)), "plant_steps": 0, "env_reset_calls": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
            "gates": gates,
            "runtime_preflight": preflight,
            "terminal_receipts": {str(k): v for k, v in terminal_receipts.items()},
            "contexts": [{k: v for k, v in c.items() if k != "case_snapshot"} for c in contexts],
            "schedule": schedule,
            "arms": arms,
            "cross_evaluation": cross,
            "analysis": analysis,
            "input_hashes": {rel(p): sha256(p) for p in [Path(__file__).resolve(), ANALYSIS_READY, ASTRA_REPORT, TASK1_DONE, TASK1_RAW, TASK1_TERMINAL_CSV, D33_DONE, D33_MANIFEST] if p.exists()},
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
            "interpretation_limits": ["opened development contexts only", "not validation64", "not sealed test", "no closed-loop adaptive selector", "no plant continuation", "objective decomposition best-effort and gated by reconstruction error"],
            "next_astra_request_id": REQUEST_ID,
            "backup_request_after_run": rel(BACKUP_REQUEST),
        }
        if raw["budget_actual"]["solver_calls"] != MAX_SOLVES:
            raise ContractError("did not execute exactly fixed 24 solver calls")
        if raw["budget_actual"]["plant_steps"] != 0 or raw["budget_actual"]["env_reset_calls"] != 0:
            raise ContractError("forbidden plant/reset call count nonzero")
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        write_json(BACKUP_REQUEST, {"request": "backup_after_v34_objective_basin_solver_probe", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "new v34 low-level solver diagnostic and Astra handoff", "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(NEXT_REVIEW_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "new_solver_calls": raw["budget_actual"]["solver_calls"], "new_plant_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
        update_docs(raw)
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE, BACKUP_REQUEST, NEXT_REVIEW_REQUEST, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed = {"status": "complete", "passed": True, "hard_pass": True, "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": raw["budget_actual"], "headline": analysis["headline"], "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "backup_request": rel(BACKUP_REQUEST), "next_astra_request_id": REQUEST_ID, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}}
        write_json(RUN_DIR / "completed.json", completed)
        raw["completed_sha256"] = sha256(RUN_DIR / "completed.json")
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "headline": analysis["headline"], "backup_request": rel(BACKUP_REQUEST), "next_astra_request_id": REQUEST_ID}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        fail = {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_fixed_context_objective_basin_solver_probe_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_caps": {"solver_call_cap": MAX_SOLVES, "plant_steps": 0, "env_reset_calls": 0}}
        write_json(RUN_DIR / "failed.json", fail)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(f"# v34 objective-basin solver probe failed\n\nUTC: {fail['created_utc']}\n\nError: {fail['error']}\n\nArtifact: {rel(RUN_DIR / 'failed.json')}\n\nNo validation64 or sealed test access was requested by this script. Inspect exact missing primitive/defect before retrying; do not rerun v33 rollouts.\n", encoding="utf-8")
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
