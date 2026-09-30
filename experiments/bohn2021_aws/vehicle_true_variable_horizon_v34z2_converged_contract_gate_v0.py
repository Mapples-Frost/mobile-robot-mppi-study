#!/usr/bin/env python3
"""T-C2 converged objective-contract gate with prospective epsilon capture.

Structured task: T-C2-OC-epsilon-1b-converged-contract-gate from Opus plan
20260930T141043Z_49ac6c. Development artifacts only: opened source242 context,
no plant rollout, no training/refit, no validation64, no sealed/final test.
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts" / "research_service",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import execution_contract  # type: ignore  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0"
TASK_ID = "T-C2-OC-epsilon-1b-converged-contract-gate"
EXPECTED_REQUEST = "execution-failure:T-C3-startup-repair:20260930T135213_97b8a1ff"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO_RESOURCES = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}
RESOURCE_USAGE = dict(ZERO_RESOURCES)
GAMMA = 0.97
SOFT_OBSTACLE_PENALTY = 1000.0
EPS_MATERIAL = 1e-8
CONVERGED_REL_MAX = 1e-6
H35_NONCONV_REL_MAX = 1e-5
HARD_DEFECT_REL = 1e-4
SOLVER_CAP = 6
SCHEDULED_SOLVES = 5
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
TC3R_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_matched_parameter_ledger_v0_20260930T141339Z/completed.json"
TC5_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z_converged_contract_gate_v0_20260930T141917Z/completed.json"
V34Z_T_C5_SOURCE = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34z_converged_contract_gate_v0.py"

PRODUCTION_CELLS = [
    {"cell_label": "H12_canonical", "horizon": 12, "initialization": "canonical", "role": "production_H12_canonical", "solve_mode": "production"},
    {"cell_label": "H15_canonical", "horizon": 15, "initialization": "canonical", "role": "production_H15_canonical", "solve_mode": "production"},
    {"cell_label": "H35_canonical", "horizon": 35, "initialization": "canonical", "role": "production_H35_canonical", "solve_mode": "production"},
    {"cell_label": "H15_goal_facing", "horizon": 15, "initialization": "goal_facing", "role": "production_H15_goal_facing", "solve_mode": "production"},
]
SOLVE_ORDER = [
    {"cell_label": "H15_canonical", "horizon": 15, "initialization": "canonical", "role": "production_H15_canonical", "solve_mode": "production"},
    {"cell_label": "H15_goal_facing", "horizon": 15, "initialization": "goal_facing", "role": "production_H15_goal_facing", "solve_mode": "production"},
    {"cell_label": "H12_canonical", "horizon": 12, "initialization": "canonical", "role": "production_H12_canonical", "solve_mode": "production"},
    {"cell_label": "H35_canonical", "horizon": 35, "initialization": "canonical", "role": "production_H35_canonical", "solve_mode": "production"},
    {"cell_label": "H12_truncated_control", "horizon": 12, "initialization": "canonical", "role": "truncated_control_H12_maxiter1", "solve_mode": "truncated_maxiter1"},
]

CAPTURES: List[Dict[str, Any]] = []
PATCHED_MODULES: List[str] = []
CURRENT_ARM_ID: Optional[str] = None
CURRENT_SOLVE_MODE: Optional[str] = None
MODULES: Dict[str, Any] = {}


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
    try:
        import numpy as np  # local for numpy scalar conversion
    except Exception:  # pragma: no cover
        np = None  # type: ignore
    if isinstance(v, Path):
        return rel(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if np is not None and isinstance(v, (np.integer,)):
        return int(v)
    if np is not None and isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Mapping):
        return {str(k): clean(val) for k, val in v.items() if k != "nlp_expr"}
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def arr(value: Any):
    import numpy as np
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
    import numpy as np
    a = arr(value)
    if int(a.size) == 0:
        return None
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    a = arr(value)
    if int(a.size) < 1:
        return default
    try:
        out = float(a.reshape(-1)[0])
    except Exception:
        return default
    return out if math.isfinite(out) else default


def cat_arg(obj: Any) -> Any:
    import numpy as np
    if hasattr(obj, "cat"):
        return obj.cat
    if hasattr(obj, "master"):
        return obj.master
    return np.asarray(arr(obj), dtype=float).reshape((-1, 1))


def parse_label(label: str) -> List[Any]:
    s = str(label).strip().strip("[]")
    out: List[Any] = []
    for p in s.split(","):
        p = p.strip().strip("'\"")
        if p == "":
            continue
        if p.lstrip("-").isdigit():
            out.append(int(p))
        else:
            out.append(p)
    return out


def get_struct_labels(obj: Any) -> List[str]:
    try:
        return [str(x) for x in obj.labels()]
    except Exception:
        return []


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


def safe_struct_get(obj: Any, idx: Tuple[Any, ...]) -> Optional[float]:
    try:
        return finite_float(obj[idx], None)
    except Exception:
        try:
            return finite_float(obj.__getitem__(idx), None)
        except Exception:
            return None


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def timing_summary(values: Sequence[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean_s": None, "median_s": None, "p95_s": None, "min_s": None, "max_s": None}
    vals_sorted = sorted(vals)
    if len(vals_sorted) == 1:
        p95 = vals_sorted[0]
    else:
        pos = 0.95 * (len(vals_sorted) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos)); frac = pos - lo
        p95 = vals_sorted[lo] * (1 - frac) + vals_sorted[hi] * frac
    return {"n": len(vals), "mean_s": float(mean(vals)), "median_s": float(median(vals)), "p95_s": float(p95), "min_s": float(min(vals)), "max_s": float(max(vals))}


def read_api_total_tokens() -> Dict[str, Any]:
    for db in [ROOT / "research.sqlite", ROOT / "research_artifacts" / "research.sqlite", ROOT / "docs" / "research.sqlite"]:
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db))
            try:
                totals: Dict[str, int] = {}
                tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()]
                for table in tables:
                    cols = [r[1] for r in con.execute(f"pragma table_info({table})").fetchall()]
                    if "total_tokens" in cols:
                        totals[table] = int(con.execute(f"select coalesce(sum(total_tokens),0) from {table}").fetchone()[0] or 0)
                if totals:
                    return {"available": True, "path": rel(db), "table_sums": totals, "total_tokens": int(sum(totals.values()))}
            finally:
                con.close()
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": f"{type(exc).__name__}: {exc}"}
    return {"available": False, "path": None, "error": "research.sqlite not found in checked repository locations"}


def load_modules() -> None:
    if MODULES:
        return
    for _p in (ROOT / "experiments/bohn2021_aws", ROOT / "experiments/bohn2021_reproduction"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import numpy as np  # noqa:F401
    import casadi as ca  # type: ignore
    import vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0 as base  # type: ignore
    import vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0 as kacc  # type: ignore
    import vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0 as v34u  # type: ignore
    import vehicle_true_variable_horizon_v34q_goal_source_enumeration_v0 as v34q  # type: ignore
    import vehicle_stress_scenario_opportunity_probe_v1_runner as stage1_runner  # type: ignore
    MODULES.update(np=np, ca=ca, base=base, kacc=kacc, v34u=v34u, v34q=v34q, stage1_runner=stage1_runner)


def exercise_failure_path(run_dir: Path) -> Dict[str, Any]:
    bad = run_dir / "invalid_authorization_snapshot_for_failure_path.json"
    write_json(bad, {"schema_version": execution_contract.VERSION, "intentionally_invalid": True, "reason": "T-C2 P3 failure-path exercise"})
    try:
        execution_contract.verify_snapshot(ROOT, bad, expected_request=EXPECTED_REQUEST)
    except Exception as exc:
        payload = {
            "status": "failed",
            "error": f"invalid authorization rejected as expected: {type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "synthetic_success_receipt_written": False,
        }
        out = run_dir / "failure_path_invalid_authorization_exercised_failed.json"
        write_json(out, payload)
        return {"exercised": True, "invalid_authorization_remained_failure": True, "path": rel(out), "error_type": type(exc).__name__}
    return {"exercised": False, "invalid_authorization_remained_failure": False, "path": rel(bad), "error": "invalid snapshot unexpectedly verified"}


def verify_prior_and_preconditions(run_dir: Path) -> Dict[str, Any]:
    for p in [TC3R_DONE, TC5_DONE, V34Z_T_C5_SOURCE]:
        if not p.exists():
            raise ContractError("missing prerequisite/precondition path: " + rel(p))
    tc3 = read_json(TC3R_DONE); tc5 = read_json(TC5_DONE)
    if tc3.get("hard_pass") is not True or (tc3.get("budget_actual") or {}).get("solver_calls") != 0:
        raise ContractError("T-C3R receipt is not a zero-resource hard pass")
    if tc5.get("hard_pass") is not True or (tc5.get("budget_actual") or {}).get("solver_calls") != 0:
        raise ContractError("T-C5 receipt is not a zero-resource hard pass")
    failure_exercise = exercise_failure_path(run_dir)
    if failure_exercise.get("invalid_authorization_remained_failure") is not True:
        raise ContractError("P3 failure-path exercise did not preserve invalid authorization as failure")
    return {
        "new_script_not_t_c5_source": Path(__file__).name == "vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0.py" and Path(__file__).resolve() != V34Z_T_C5_SOURCE.resolve(),
        "t_c5_source_preserved_path": rel(V34Z_T_C5_SOURCE),
        "t_c5_source_sha256": sha256(V34Z_T_C5_SOURCE),
        "digest_encoding_must_be_float64_tobytes_not_json": True,
        "digest_encoding_function": "hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()",
        "json_digest_for_parameter_vectors_forbidden": True,
        "failure_path_exercise": failure_exercise,
        "T_C3R": {"path": rel(TC3R_DONE), "hard_pass": tc3.get("hard_pass"), "budget_actual": tc3.get("budget_actual"), "sha256": sha256(TC3R_DONE)},
        "T_C5": {"path": rel(TC5_DONE), "hard_pass": tc5.get("hard_pass"), "budget_actual": tc5.get("budget_actual"), "sha256": sha256(TC5_DONE)},
    }


def install_nlpsol_patch() -> None:
    ca = MODULES["ca"]
    import do_mpc.controller as controller_mod  # type: ignore
    try:
        import do_mpc.optimizer as optimizer_mod  # type: ignore
    except Exception:
        optimizer_mod = None  # type: ignore
    if getattr(controller_mod, "_v34z2_nlpsol_patched", False):
        return
    original = getattr(controller_mod, "nlpsol", None) or getattr(ca, "nlpsol", None)
    if original is None:
        raise ContractError("no nlpsol callable found for v34z2 patch")

    def wrapped_nlpsol(name: Any, solver: Any, nlp: Mapping[str, Any], opts: Optional[Mapping[str, Any]] = None) -> Any:
        opts_in = dict(opts or {})
        opts_out = dict(opts_in)
        if CURRENT_SOLVE_MODE == "truncated_maxiter1":
            opts_out.update({"ipopt.max_iter": 1, "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False})
        else:
            opts_out.setdefault("ipopt.print_level", 0)
            opts_out.setdefault("ipopt.sb", "yes")
            opts_out.setdefault("print_time", False)
        cap = {
            "arm_id": CURRENT_ARM_ID,
            "solve_mode": CURRENT_SOLVE_MODE,
            "solver_name": str(name),
            "solver_kind": str(solver),
            "opts_in_subset": {k: opts_in.get(k) for k in sorted(set(list(opts_in.keys()) + ["ipopt.max_iter", "ipopt.tol", "ipopt.acceptable_tol", "ipopt.print_level", "ipopt.sb", "print_time"]))},
            "opts_out_subset": {k: opts_out.get(k) for k in sorted(set(list(opts_out.keys()) + ["ipopt.max_iter", "ipopt.tol", "ipopt.acceptable_tol", "ipopt.print_level", "ipopt.sb", "print_time"]))},
            "effective_ipopt_max_iter": opts_out.get("ipopt.max_iter"),
            "production_max_iter_not_forced_to_1": bool(CURRENT_SOLVE_MODE == "production" and opts_out.get("ipopt.max_iter") != 1),
            "truncated_max_iter_forced_to_1": bool(CURRENT_SOLVE_MODE == "truncated_maxiter1" and opts_out.get("ipopt.max_iter") == 1),
            "nlp_meta": {k: str(getattr(v, "shape", None)) for k, v in (nlp or {}).items() if k in ("x", "f", "g", "p")},
            "nlp_expr": nlp,
        }
        CAPTURES.append(cap)
        return original(name, solver, nlp, opts_out)

    setattr(controller_mod, "_v34z2_original_nlpsol", original)
    setattr(controller_mod, "nlpsol", wrapped_nlpsol)
    setattr(ca, "nlpsol", wrapped_nlpsol)
    if optimizer_mod is not None:
        setattr(optimizer_mod, "nlpsol", wrapped_nlpsol)
    setattr(controller_mod, "_v34z2_nlpsol_patched", True)
    PATCHED_MODULES.extend(["casadi.nlpsol", "do_mpc.controller.nlpsol"] + (["do_mpc.optimizer.nlpsol"] if optimizer_mod is not None else []))


def scalarize_one(value: Any) -> float:
    import numpy as np
    if isinstance(value, Mapping):
        for k in ("true", "value", "forecast", "mean", "x", "y", "r"):
            if k in value:
                try:
                    return scalarize_one(value[k])
                except Exception:
                    pass
        for v in value.values():
            try:
                return scalarize_one(v)
            except Exception:
                pass
        raise ContractError("mapping TVP element has no scalar numeric value")
    if isinstance(value, (list, tuple)):
        if not value:
            raise ContractError("empty sequence TVP element")
        return scalarize_one(value[0])
    if hasattr(value, "shape") or hasattr(value, "tolist"):
        a = np.asarray(value, dtype=float).reshape(-1)
        if a.size < 1:
            raise ContractError("empty array TVP element")
        out = float(a[0])
    else:
        out = float(value)
    if not math.isfinite(out):
        raise ContractError("non-finite TVP scalar")
    return out


def scalarize_tvp(tvp: Mapping[str, Any]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for k, seq in tvp.items():
        if not isinstance(seq, list):
            seq = [seq]
        out[str(k)] = [scalarize_one(x) for x in seq]
    return out


def goal_endpoint_61(case: Mapping[str, Any]) -> Dict[str, Any]:
    v34q = MODULES["v34q"]
    got = v34q.robust_goal_from_saved_case(case)
    if got.get("ok") is not True:
        raise ContractError("authoritative goal extraction failed: " + repr(got))
    if str(got.get("source")) != "case.tvp.trajectory_endpoint[61]" or int(got.get("endpoint_index")) != 61:
        raise ContractError("goal provenance is not frozen case.tvp.trajectory_endpoint[61]: " + repr(got))
    return {"goal_x": float(got["goal_x"]), "goal_y": float(got["goal_y"]), "source": got.get("source"), "endpoint_index": int(got.get("endpoint_index")), "raw": clean(got)}


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
    return [dict(state) for _ in range(h + 1)], [{"u_omega": 0.0, "u_s": 0.0} for _ in range(h)]


def variable_from_parts(parts: Sequence[Any], names: Sequence[str]) -> Optional[str]:
    name_set = set(str(n) for n in names)
    for p in reversed(parts):
        if isinstance(p, str) and p in name_set:
            return p
    return None


def numeric_state_dict_from_meta(meta: Mapping[str, Any]) -> Dict[str, float]:
    st = meta.get("state_after_config") or {}
    return {"theta": float(st["theta"]), "x": float(st["x"]), "y": float(st["y"])}


def set_initial_guess_strict(mpc: Any, env: Any, context_meta: Mapping[str, Any], init_mode: str, h: int) -> Dict[str, Any]:
    ctrl = env.control_system.controller
    state_names = list(getattr(ctrl, "state_names", ["theta", "x", "y"]))
    input_names = list(getattr(ctrl, "input_names", ["u_omega", "u_s"]))
    state = numeric_state_dict_from_meta(context_meta)
    if init_mode == "canonical":
        pred_states, pred_controls = zero_guess(state, h)
        rule = "repeat_initial_state_zero_control_zero_slack"
    elif init_mode == "goal_facing":
        dt_s = float(finite_float(getattr(mpc, "t_step", None), 0.1) or 0.1)
        pred_states, pred_controls = predicted_unicycle(state, float(context_meta["goal_x"]), float(context_meta["goal_y"]), h, dt_s)
        rule = "deterministic_unicycle_goal_facing_clip_omega_pm4_speed_0_5"
    else:
        raise ContractError("unknown initialization mode " + init_mode)

    before_hash = arr_hash(getattr(mpc, "opt_x_num", []))
    attempted = 0; succeeded = 0; readback_ok = 0; mismatches: List[Dict[str, Any]] = []; examples: List[Dict[str, Any]] = []
    obj = getattr(mpc, "opt_x_num", None)
    if obj is None:
        raise ContractError("mpc.opt_x_num unavailable for initialization")
    for label in get_struct_labels(obj):
        parts = parse_label(label)
        if not parts:
            continue
        top = parts[0]
        ints = [int(p) for p in parts[1:] if isinstance(p, int)]
        intended: Optional[float] = None
        kind: Optional[str] = None
        var: Optional[str] = None
        if top == "_x":
            k = ints[0] if ints else 0
            var = variable_from_parts(parts, state_names)
            if var is not None and 0 <= k < len(pred_states):
                intended = float(pred_states[k].get(var, state.get(var, 0.0)))
                kind = "state"
        elif top == "_u":
            k = ints[0] if ints else 0
            var = variable_from_parts(parts, input_names)
            if var is not None and 0 <= k < len(pred_controls):
                intended = float(pred_controls[k].get(var, 0.0))
                kind = "input"
        elif top == "_eps":
            intended = 0.0
            kind = "epsilon"
            var = variable_from_parts(parts, [str(x) for x in parts if isinstance(x, str) and x != "_eps"])
        if intended is None:
            continue
        attempted += 1
        idx = tuple(parts)
        ok = safe_struct_set(obj, idx, intended)
        got = safe_struct_get(obj, idx) if ok else None
        match = bool(got is not None and abs(float(got) - intended) <= 1e-10)
        if ok:
            succeeded += 1
        if match:
            readback_ok += 1
        elif len(mismatches) < 20:
            mismatches.append({"label": label, "intended": intended, "got": got, "set_ok": ok, "kind": kind, "var": var})
        if len(examples) < 12 and kind in ("state", "input"):
            examples.append({"label": label, "kind": kind, "var": var, "intended": intended, "readback": got, "match": match})
    try:
        mpc.lam_g_num = 0 * mpc.lam_g_num
    except Exception:
        pass
    after_hash = arr_hash(getattr(mpc, "opt_x_num", []))
    return {
        "initialization": init_mode,
        "rule": rule,
        "state_names": state_names,
        "input_names": input_names,
        "assignments_attempted": attempted,
        "assignments_succeeded": succeeded,
        "numeric_readback_count": readback_ok,
        "numeric_readback_all_successful": bool(attempted > 0 and readback_ok == attempted and not mismatches),
        "mismatches_first20": mismatches,
        "assignment_examples": examples,
        "initial_primal_hash_before": before_hash,
        "initial_primal_hash_after": after_hash,
        "initial_dual_hash_after_zero": arr_hash(getattr(mpc, "lam_g_num", [])) if hasattr(mpc, "lam_g_num") else None,
        "first_three_pred_states": pred_states[:3],
        "first_three_pred_controls": pred_controls[:3],
    }


def capture_mpc_state(mpc: Any) -> Dict[str, Any]:
    out = {
        "opt_x_hash": arr_hash(getattr(mpc, "opt_x_num", [])),
        "opt_x_unscaled_hash": arr_hash(getattr(mpc, "opt_x_num_unscaled", [])) if hasattr(mpc, "opt_x_num_unscaled") else None,
        "opt_p_hash": arr_hash(getattr(mpc, "opt_p_num", [])) if hasattr(mpc, "opt_p_num") else None,
        "lam_g_hash": arr_hash(getattr(mpc, "lam_g_num", [])) if hasattr(mpc, "lam_g_num") else None,
        "lam_x_hash": arr_hash(getattr(mpc, "lam_x_num", [])) if hasattr(mpc, "lam_x_num") else None,
        "opt_x_size": int(arr(getattr(mpc, "opt_x_num", [])).size),
        "opt_p_size": int(arr(getattr(mpc, "opt_p_num", [])).size) if hasattr(mpc, "opt_p_num") else None,
        "opt_g_size": int(arr(getattr(mpc, "opt_g_num", [])).size) if hasattr(mpc, "opt_g_num") else None,
        "bounds_hashes": {
            "lb_opt_x": arr_hash(getattr(mpc, "lb_opt_x", [])) if hasattr(mpc, "lb_opt_x") else None,
            "ub_opt_x": arr_hash(getattr(mpc, "ub_opt_x", [])) if hasattr(mpc, "ub_opt_x") else None,
            "cons_lb": arr_hash(getattr(mpc, "cons_lb", [])) if hasattr(mpc, "cons_lb") else None,
            "cons_ub": arr_hash(getattr(mpc, "cons_ub", [])) if hasattr(mpc, "cons_ub") else None,
            "opt_x_lb": arr_hash(getattr(mpc, "opt_x_lb", [])) if hasattr(mpc, "opt_x_lb") else None,
            "opt_x_ub": arr_hash(getattr(mpc, "opt_x_ub", [])) if hasattr(mpc, "opt_x_ub") else None,
            "opt_g_lb": arr_hash(getattr(mpc, "opt_g_lb", [])) if hasattr(mpc, "opt_g_lb") else None,
            "opt_g_ub": arr_hash(getattr(mpc, "opt_g_ub", [])) if hasattr(mpc, "opt_g_ub") else None,
        },
        "scaling_hashes": {
            "opt_x_scaling": arr_hash(getattr(mpc, "opt_x_scaling", [])) if hasattr(mpc, "opt_x_scaling") else None,
            "opt_p_scaling": arr_hash(getattr(mpc, "opt_p_scaling", [])) if hasattr(mpc, "opt_p_scaling") else None,
        },
    }
    try:
        out["solver_options_hash"] = canonical_hash(getattr(mpc, "nlpsol_opts", {}))
        out["solver_options_subset"] = {k: getattr(mpc, "nlpsol_opts", {}).get(k) for k in sorted(set(getattr(mpc, "nlpsol_opts", {}).keys()) | {"ipopt.max_iter", "ipopt.tol", "ipopt.acceptable_tol"})}
    except Exception:
        out["solver_options_hash"] = None
        out["solver_options_subset"] = {}
    return out


def p_value_map(mpc: Any) -> Dict[str, Any]:
    p = getattr(mpc, "opt_p_num", None)
    if p is None:
        return {}
    out: Dict[str, Any] = {}
    for label in get_struct_labels(p)[:5000]:
        parts = parse_label(label)
        try:
            out[label] = safe_struct_get(p, tuple(parts))
        except Exception:
            pass
    return out


def parameter_readback(mpc: Any) -> Dict[str, Any]:
    pmap = p_value_map(mpc)
    out: Dict[str, Any] = {"p_value_count": len(pmap), "goal_x": None, "goal_y": None, "n_horizon": None, "source": "opt_p_num_labels"}
    for label, val in pmap.items():
        parts = parse_label(label)
        name = next((p for p in reversed(parts) if isinstance(p, str)), str(label))
        if name in ("goal_x", "goal_y", "n_horizon") and val is not None:
            out[name] = float(val)
    out["goal_and_horizon_readback_available"] = bool(out["goal_x"] is not None and out["goal_y"] is not None and out["n_horizon"] is not None)
    return out


def residual_vs_bounds(values: Any, lb: Any, ub: Any) -> Optional[float]:
    import numpy as np
    x = arr(values); lo = arr(lb); hi = arr(ub)
    if x.size == 0 or lo.size != x.size or hi.size != x.size:
        return None
    return float(max(0.0, np.max(lo - x), np.max(x - hi)))


def prepare_cell(row: Mapping[str, Any], context: Mapping[str, Any], terminals: Mapping[int, Any], for_presolve_only: bool) -> Dict[str, Any]:
    global CURRENT_ARM_ID, CURRENT_SOLVE_MODE
    base = MODULES["base"]
    v34u = MODULES["v34u"]
    h = int(row["horizon"])
    arm_id = f"ctx=source242_slot0_branch_start|H{h}|term=V15_shared|init={row['initialization']}|role={row['role']}"
    CURRENT_ARM_ID = arm_id + ("|preflight" if for_presolve_only else "")
    CURRENT_SOLVE_MODE = str(row["solve_mode"])
    cap_before = len(CAPTURES)
    terminal, term_h, term_note = base.terminal_for_mode(h, "V15_shared", terminals)
    env = base.create_env(h, terminal)
    ctrl = env.control_system.controller
    mpc = ctrl.mpc
    context_meta = base.configure_context_no_reset(env, context, h)
    scalar_tvp = scalarize_tvp(context_meta["shifted_tvp"])
    context_meta["shifted_tvp"] = scalar_tvp
    goal = goal_endpoint_61(context["case_snapshot"])
    context_meta["goal_x"] = goal["goal_x"]
    context_meta["goal_y"] = goal["goal_y"]
    context_meta["goal_source"] = goal["source"]
    context_meta["goal_endpoint_index"] = goal["endpoint_index"]
    ctrl.goal_x = goal["goal_x"]
    ctrl.goal_y = goal["goal_y"]
    ctrl._tvp_data = copy.deepcopy(scalar_tvp)
    # Install the object-channel guard used in the earlier operational repair.
    v34u.install_tta_hmpc_presolve_obj_guard()
    init_meta = set_initial_guess_strict(mpc, env, context_meta, str(row["initialization"]), h)
    pre = capture_mpc_state(mpc)
    captures = [c for c in CAPTURES[cap_before:] if c.get("arm_id") == CURRENT_ARM_ID]
    setup = {
        "arm_id": arm_id,
        "cell_label": row["cell_label"],
        "role": row["role"],
        "horizon": h,
        "terminal_mode": "V15_shared",
        "terminal_source_horizon": int(term_h),
        "terminal_note": term_note,
        "initialization": row["initialization"],
        "solve_mode": row["solve_mode"],
        "goal": goal,
        "context_meta_no_shifted_tvp": {k: v for k, v in context_meta.items() if k != "shifted_tvp"},
        "initialization_meta": init_meta,
        "pre_solve_state_after_initialization": pre,
        "captured_nlpsol_meta": clean(captures),
        "captured_nlpsol_count_during_construction": len(captures),
        "production_max_iter_not_forced_to_1": bool(row["solve_mode"] == "production" and all(c.get("effective_ipopt_max_iter") != 1 for c in captures)),
        "truncated_max_iter_forced_to_1": bool(row["solve_mode"] == "truncated_maxiter1" and any(c.get("effective_ipopt_max_iter") == 1 for c in captures)),
    }
    return {"env": env, "ctrl": ctrl, "mpc": mpc, "context_meta": context_meta, "setup": setup, "capture_start": cap_before, "arm_id": arm_id, "row": dict(row)}


def validate_setup_gate(setups: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_label = {s["cell_label"]: s for s in setups}
    assignment_failures = [s["cell_label"] for s in setups if int((s.get("initialization_meta") or {}).get("assignments_attempted", 0) or 0) == 0 or int((s.get("initialization_meta") or {}).get("assignments_succeeded", 0) or 0) == 0]
    readback_failures = [s["cell_label"] for s in setups if (s.get("initialization_meta") or {}).get("numeric_readback_all_successful") is not True]
    h15c = by_label.get("H15_canonical"); h15g = by_label.get("H15_goal_facing")
    h15_identical_pre_state = None
    h15_identical_p = None
    h15_distinct_numeric = None
    if h15c and h15g:
        pre_c = h15c["pre_solve_state_after_initialization"]; pre_g = h15g["pre_solve_state_after_initialization"]
        h15_identical_pre_state = pre_c.get("opt_x_hash") == pre_g.get("opt_x_hash")
        h15_identical_p = pre_c.get("opt_p_hash") == pre_g.get("opt_p_hash")
        c_controls = ((h15c.get("initialization_meta") or {}).get("first_three_pred_controls") or [])
        g_controls = ((h15g.get("initialization_meta") or {}).get("first_three_pred_controls") or [])
        h15_distinct_numeric = clean(c_controls) != clean(g_controls)
    inert = bool(assignment_failures or readback_failures or (h15_identical_pre_state and h15_identical_p) or not h15_distinct_numeric)
    return {
        "evaluated": True,
        "assignment_failures": assignment_failures,
        "numeric_readback_failures": readback_failures,
        "H15_preflight_opt_x_hash_identical": h15_identical_pre_state,
        "H15_preflight_opt_p_hash_identical": h15_identical_p,
        "H15_preflight_numeric_profiles_distinct": h15_distinct_numeric,
        "inert_intervention_failure": inert,
        "verdict": "INERT_INTERVENTION_FAILURE" if inert else "PASS_PRELAUNCH_SETUP_GATE",
    }


def direct_nlp_eval(mpc: Any, capture: Optional[Mapping[str, Any]], idx: int) -> Dict[str, Any]:
    ca = MODULES["ca"]
    if not capture or not isinstance(capture.get("nlp_expr"), Mapping):
        return {"available": False, "reason": "no captured nlp expression"}
    try:
        nlp = capture["nlp_expr"]
        f_fn = ca.Function(f"v34z2_f_{idx}", [nlp.get("x"), nlp.get("p")], [nlp.get("f")])
        g_fn = ca.Function(f"v34z2_g_{idx}", [nlp.get("x"), nlp.get("p")], [nlp.get("g")])
        x_val = cat_arg(mpc.opt_x_num)
        p_val = cat_arg(mpc.opt_p_num)
        f_val = finite_float(f_fn(x_val, p_val), None)
        g_val = arr(g_fn(x_val, p_val))
        saved_g = arr(getattr(mpc, "opt_g_num", []))
        solver = finite_float(getattr(mpc, "opt_f_num", None), None)
        abs_err = None if f_val is None or solver is None else float(abs(float(f_val) - float(solver)))
        rel_err = None if abs_err is None or solver is None else float(abs_err / max(1.0, abs(float(solver))))
        return {"available": True, "direct_f_value": f_val, "solver_opt_f_num": solver, "direct_f_abs_error_vs_solver": abs_err, "direct_f_rel_error_vs_solver": rel_err, "g_eval_size": int(g_val.size), "saved_opt_g_size": int(saved_g.size), "g_eval_minus_saved_opt_g_linf": None if g_val.size == 0 or g_val.size != saved_g.size else float(max(abs(g_val - saved_g))), "captured_nlp_meta": clean(capture.get("nlp_meta"))}
    except Exception as exc:
        return {"available": False, "reason": "direct_nlp_eval_failed", "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-8:]}


def formula_try(mpc: Any, h: int, include_eps: bool, include_rterm: bool) -> Dict[str, Any]:
    kacc = MODULES["kacc"]
    try:
        acc = kacc.LabelAccessor(mpc)
        comp = kacc.formula_components(mpc, acc, int(h), "x0_then_last_node", "author_last_node", bool(include_eps), bool(include_rterm), "n_horizon_parameter_or_H", "same_k_last")
        comp["available"] = True
        comp["candidate_id"] = f"stage=x0_then_last_node|term=author_last_node|eps={bool(include_eps)}|r={bool(include_rterm)}|discount=n_horizon_parameter_or_H|z=same_k_last"
        return comp
    except Exception as exc:
        return {"available": False, "include_eps": bool(include_eps), "include_rterm": bool(include_rterm), "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-8:]}


def eps_vectors_and_margins(mpc: Any, ctrl: Any, h: int) -> Dict[str, Any]:
    kacc = MODULES["kacc"]
    acc = kacc.LabelAccessor(mpc)
    eps_labels = list(getattr(acc, "eps_vars", []))
    rows: List[Dict[str, Any]] = []
    for k in range(h):
        vec = arr(acc.eps(k, 0))
        for j, val in enumerate(vec.tolist()):
            rows.append({"k": k, "scenario": 0, "epsilon_index": j, "epsilon_label": eps_labels[j] if j < len(eps_labels) else f"eps[{j}]", "epsilon_value": float(val), "gamma_power": float(GAMMA ** k), "weighted_penalty_contribution": float((GAMMA ** k) * SOFT_OBSTACLE_PENALTY * float(val)), "dimension": "scalar"})
    margin_rows: List[Dict[str, Any]] = []
    state_vars = list(getattr(acc, "state_vars", []))
    n_obj = int(getattr(ctrl, "n_objects", 0) or 0)
    def tvp_scalar(k: int, name: str) -> Optional[float]:
        try:
            return acc._scalar(acc.p_records, ("_tvp", int(k), name, 0), "_tvp", (int(k),), name)  # type: ignore[attr-defined]
        except Exception:
            return None
    for k in range(h):
        try:
            xvec = arr(acc.state(k, 0, -1))
            sdict = {name: float(xvec[i]) for i, name in enumerate(state_vars) if i < len(xvec)}
        except Exception:
            sdict = {}
        for obj_i in range(n_obj):
            ox = tvp_scalar(k, f"obj_{obj_i}_x"); oy = tvp_scalar(k, f"obj_{obj_i}_y"); rr = tvp_scalar(k, f"obj_{obj_i}_r")
            if ox is None or oy is None or rr is None or "x" not in sdict or "y" not in sdict:
                continue
            d = math.sqrt((float(ox) - sdict["x"]) ** 2 + (float(oy) - sdict["y"]) ** 2)
            margin_rows.append({"k": k, "scenario": 0, "object_index": obj_i, "node_rule": "stage_last_node", "x": sdict.get("x"), "y": sdict.get("y"), "obj_x": ox, "obj_y": oy, "obj_r": rr, "distance": d, "soft_margin_1p5r_minus_d": float(1.5 * float(rr) - d), "hard_clearance_d_minus_r": float(d - float(rr))})
    values = [abs(float(r["epsilon_value"])) for r in rows]
    return {"epsilon_labels": eps_labels, "epsilon_dimension_per_stage": len(eps_labels), "per_stage_epsilon_vector": rows, "individual_eps_nonzero_count": int(sum(v > EPS_MATERIAL for v in values)), "epsilon_abs_sum": float(sum(values)), "epsilon_materiality_threshold": EPS_MATERIAL, "epsilon_materially_nonzero_at_convergence": bool(any(v > EPS_MATERIAL for v in values)), "per_stage_weighted_penalty_sum": float(math.fsum(float(r["weighted_penalty_contribution"]) for r in rows)), "soft_margin_and_hard_clearance": margin_rows, "constraint_node_statement": "Geometry is reported at the labelled stage last-node state used by the objective accessor; direct NLP g/bound residuals are reported separately as the authoritative solver constraint residuals."}


def solve_prepared(prep: Mapping[str, Any], idx: int) -> Dict[str, Any]:
    global RESOURCE_USAGE, CURRENT_ARM_ID, CURRENT_SOLVE_MODE
    base = MODULES["base"]
    row = prep["row"]; mpc = prep["mpc"]; ctrl = prep["ctrl"]; context_meta = prep["context_meta"]
    h = int(row["horizon"]); arm_id = prep["arm_id"]
    CURRENT_ARM_ID = arm_id
    CURRENT_SOLVE_MODE = str(row["solve_mode"])
    events: List[Dict[str, Any]] = []
    original_solve = mpc.solve
    def counted_solve(*args: Any, **kwargs: Any) -> Any:
        if len(events) >= 1:
            raise ContractError("second low-level solver call inside one cell was blocked")
        if RESOURCE_USAGE["solver_calls"] >= SOLVER_CAP:
            raise ContractError("T-C2 solver cap exhausted")
        pre = capture_mpc_state(mpc)
        pre_param = parameter_readback(mpc)
        RESOURCE_USAGE["solver_calls"] += 1
        t0 = time.perf_counter()
        try:
            ret = original_solve(*args, **kwargs)
            exc = None
        except Exception as e:
            ret = None; exc = repr(e)
        elapsed = float(time.perf_counter() - t0)
        post = capture_mpc_state(mpc)
        stats = copy.deepcopy(getattr(mpc, "solver_stats", {}))
        event = {"pre": pre, "pre_parameter_readback": pre_param, "post": post, "solver_wall_s": elapsed, "solver_exception": exc, "solver_stats": stats, "return_status": stats.get("return_status"), "unified_return_status": stats.get("return_status") or ("exception" if exc else "unknown"), "success": bool(stats.get("success", False)), "iterations": stats.get("iter_count") or stats.get("iterations"), "objective_opt_f_num": finite_float(getattr(mpc, "opt_f_num", None), None)}
        events.append(event)
        if exc is not None:
            raise ContractError("mpc.solve failed: " + exc)
        return ret
    mpc.solve = counted_solve
    start = time.perf_counter()
    action = None
    get_exc = None
    try:
        action = ctrl.get_action(base.state_clean({"theta": context_meta["state_after_config"]["theta"], "x": context_meta["state_after_config"]["x"], "y": context_meta["state_after_config"]["y"]}), h, tvp_values=copy.deepcopy(context_meta["shifted_tvp"]))
    except Exception as exc:
        get_exc = repr(exc)
    whole = float(time.perf_counter() - start)
    if get_exc is not None:
        raise ContractError("controller.get_action failed: " + get_exc)
    if len(events) != 1:
        raise ContractError(f"expected one solve event for {arm_id}, got {len(events)}")
    capture = next((c for c in reversed(CAPTURES) if c.get("arm_id") == arm_id), None)
    noeps = formula_try(mpc, h, False, False)
    full = formula_try(mpc, h, True, True)
    dnlp = direct_nlp_eval(mpc, capture, idx)
    eps = eps_vectors_and_margins(mpc, ctrl, h)
    ev = events[0]
    converged = bool(ev.get("success") is True and str(ev.get("return_status")) == "Solve_Succeeded")
    rel_full = full.get("relative_error")
    rel_noeps = noeps.get("relative_error")
    if str(row["cell_label"]) == "H35_canonical" and not converged:
        threshold = H35_NONCONV_REL_MAX
        threshold_reason = "H35 non-converged allowance was documented before solving"
    else:
        threshold = CONVERGED_REL_MAX
        threshold_reason = "production converged threshold"
    contract_pass = bool(rel_full is not None and float(rel_full) <= threshold and (converged or str(row["cell_label"]) == "H35_canonical"))
    hard_defect = bool(converged and rel_full is not None and float(rel_full) > HARD_DEFECT_REL)
    constraints = {"cons_lb_cons_ub_residual": residual_vs_bounds(getattr(mpc, "opt_g_num", []), getattr(mpc, "cons_lb", []), getattr(mpc, "cons_ub", [])), "opt_g_lb_ub_residual": residual_vs_bounds(getattr(mpc, "opt_g_num", []), getattr(mpc, "opt_g_lb", []), getattr(mpc, "opt_g_ub", [])), "lb_opt_x_ub_opt_x_residual": residual_vs_bounds(getattr(mpc, "opt_x_num", []), getattr(mpc, "lb_opt_x", []), getattr(mpc, "ub_opt_x", [])), "opt_x_lb_ub_residual": residual_vs_bounds(getattr(mpc, "opt_x_num", []), getattr(mpc, "opt_x_lb", []), getattr(mpc, "opt_x_ub", []))}
    return {"arm_id": arm_id, "cell_label": row["cell_label"], "role": row["role"], "horizon": h, "initialization": row["initialization"], "solve_mode": row["solve_mode"], "setup": prep["setup"], "solve_event": ev, "controller_get_action_wall_s": whole, "first_control": clean(action), "return_status": ev.get("return_status"), "unified_return_status": ev.get("unified_return_status"), "actual_iteration_count": ev.get("iterations"), "converged": converged, "J_solver": ev.get("objective_opt_f_num"), "J_recon_historical_noeps": noeps.get("total"), "J_recon_full": full.get("total"), "relative_error_historical_noeps": rel_noeps, "relative_error_full": rel_full, "objective_reconstruction_historical_noeps": noeps, "objective_reconstruction_full": full, "direct_nlp_f_eval": dnlp, "epsilon_capture": eps, "constraint_residuals": constraints, "effective_tolerances_and_solver_options": {"nlpsol_capture": clean(capture), "mpc_solver_options_subset": ev.get("pre", {}).get("solver_options_subset")}, "acceptance_threshold": threshold, "acceptance_threshold_reason": threshold_reason, "objective_contract_cell_pass": contract_pass, "hard_defect_converged_rel_error_gt_1e_minus_4": hard_defect}


def load_context_and_terminals() -> Tuple[Mapping[str, Any], Mapping[int, Any], Dict[str, Any]]:
    base = MODULES["base"]
    stage1 = MODULES["stage1_runner"]
    try:
        base.v1.latency_verify()
    except Exception:
        pass
    term_protocol = read_json(stage1.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1.load_terminal_grid(term_protocol["terminal_grid_readiness_reused_from_v1"])

    # T-C2R2 source242 loader repair. The opened source242 context is built
    # from the same primitive state specification and literals used by the base
    # source242 entry, without invoking the broader context list builder.
    specs = base.v29.build_state_specs()
    spec = base.find_spec("v27_case09_slot0_early_risk", specs)
    context = {
        "context_id": "source242_slot0_branch_start",
        "state_label": "v27_case09_slot0_early_risk",
        "source": "v29/v33 selected branch state, original branch start",
        "case_snapshot": copy.deepcopy(spec["case_snapshot"]),
        "branch_step": int(spec["branch_step"]),
        "tvp_start_index": int(spec["branch_step"]),
        "state": base.state_clean(spec["branch_previous_state"]),
        "previous_input": {"u_omega": 0.0, "u_s": 0.0},
        "previous_input_source": "Astra-specified v33 branch-reset zero-input semantics",
        "horizons": [15, 35],
    }
    if 15 not in terminals:
        raise ContractError("terminal grid lacks V15")
    return context, terminals, {
        "terminal_protocol": rel(stage1.TERMINAL_SOURCE_PROTOCOL),
        "terminal_receipts": clean({str(k): v for k, v in terminal_receipts.items()}),
        "context_id": "source242_slot0_branch_start",
        "context_construction": {
            "source242_context_built_without_calling_base_load_contexts": True,
            "base_load_contexts_called": False,
            "reference_primitives": [
                "base.v29.build_state_specs()",
                "base.find_spec('v27_case09_slot0_early_risk', specs)",
                "base.state_clean(spec['branch_previous_state'])",
            ],
            "mirrors_base_load_contexts_source_lines": "vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py:317-329",
            "not_determined": "Full base.load_contexts output is not constructed here because the unrelated c13 entry is the observed loader defect; equality is established for the source242 fields by using the same primitives and literals as the base source242 entry.",
        },
    }
def write_failure(run_dir: Optional[Path], created: dt.datetime, error: str, used: Mapping[str, int], evidence: Optional[Mapping[str, Any]] = None, engineering: bool = False) -> int:
    if run_dir is None:
        run_dir = ROOT / "research_artifacts" / "aws_diagnostics" / f"{NAME}_failure_{created.strftime('%Y%m%dT%H%M%SZ')}"
    payload = {"status": "failed", "created_utc": now_utc().isoformat(), "error": error, "traceback_tail": traceback.format_exc().splitlines()[-12:], "budget_actual": dict(used), "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False}
    write_json(run_dir / "failed.json", payload)
    try:
        if engineering and all(int(v) == 0 for v in used.values()):
            execution_contract.record_outcome(ROOT, "engineering_failure", dict(used), {"no_scientific_outcome": True, "error": error, "failed_json": rel(run_dir / "failed.json")}, engineering_error="loader")
        else:
            fail_evidence = {"converged_cells_reported_with_relative_errors": False, "inert_intervention_gate_evaluated_and_reported": bool(evidence and evidence.get("inert_intervention_gate_evaluated_and_reported")), "no_plant_training_validation_or_test_usage": True, "numeric_readback_reported_for_intended_primals": False, "per_stage_epsilon_vector_persisted_with_label_and_dimension": False, "solver_and_whole_decision_timing_reported_with_n": False}
            execution_contract.record_outcome(ROOT, "scientific_result", dict(used), fail_evidence)
    except Exception:
        pass
    print(json.dumps(clean({"failed": error, "failed_json": rel(run_dir / "failed.json"), "resources": used}), sort_keys=True), flush=True)
    return 1


def write_csvs(run_dir: Path, cell_results: Sequence[Mapping[str, Any]], setup_records: Sequence[Mapping[str, Any]], epsilon_path: Path) -> Path:
    cell_csv = run_dir / "cell_metrics.csv"
    fields = ["cell_label", "role", "horizon", "initialization", "solve_mode", "return_status", "unified_return_status", "actual_iteration_count", "converged", "J_solver", "J_recon_historical_noeps", "J_recon_full", "relative_error_historical_noeps", "relative_error_full", "acceptance_threshold", "objective_contract_cell_pass", "hard_defect_converged_rel_error_gt_1e_minus_4", "solver_wall_s", "whole_decision_get_action_wall_s", "epsilon_dimension_per_stage", "individual_eps_nonzero_count", "epsilon_abs_sum", "epsilon_materially_nonzero_at_convergence"]
    with cell_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in cell_results:
            ev = r.get("solve_event") or {}; eps = r.get("epsilon_capture") or {}
            w.writerow({"cell_label": r.get("cell_label"), "role": r.get("role"), "horizon": r.get("horizon"), "initialization": r.get("initialization"), "solve_mode": r.get("solve_mode"), "return_status": r.get("return_status"), "unified_return_status": r.get("unified_return_status"), "actual_iteration_count": r.get("actual_iteration_count"), "converged": r.get("converged"), "J_solver": r.get("J_solver"), "J_recon_historical_noeps": r.get("J_recon_historical_noeps"), "J_recon_full": r.get("J_recon_full"), "relative_error_historical_noeps": r.get("relative_error_historical_noeps"), "relative_error_full": r.get("relative_error_full"), "acceptance_threshold": r.get("acceptance_threshold"), "objective_contract_cell_pass": r.get("objective_contract_cell_pass"), "hard_defect_converged_rel_error_gt_1e_minus_4": r.get("hard_defect_converged_rel_error_gt_1e_minus_4"), "solver_wall_s": ev.get("solver_wall_s"), "whole_decision_get_action_wall_s": r.get("controller_get_action_wall_s"), "epsilon_dimension_per_stage": eps.get("epsilon_dimension_per_stage"), "individual_eps_nonzero_count": eps.get("individual_eps_nonzero_count"), "epsilon_abs_sum": eps.get("epsilon_abs_sum"), "epsilon_materially_nonzero_at_convergence": eps.get("epsilon_materially_nonzero_at_convergence")})
    setup_csv = run_dir / "setup_assertions.csv"
    sfields = ["cell_label", "role", "horizon", "initialization", "solve_mode", "assignments_attempted", "assignments_succeeded", "numeric_readback_count", "numeric_readback_all_successful", "pre_opt_x_hash", "pre_opt_p_hash", "production_max_iter_not_forced_to_1", "truncated_max_iter_forced_to_1"]
    with setup_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        for s in setup_records:
            im = s.get("initialization_meta") or {}; pre = s.get("pre_solve_state_after_initialization") or {}
            w.writerow({"cell_label": s.get("cell_label"), "role": s.get("role"), "horizon": s.get("horizon"), "initialization": s.get("initialization"), "solve_mode": s.get("solve_mode"), "assignments_attempted": im.get("assignments_attempted"), "assignments_succeeded": im.get("assignments_succeeded"), "numeric_readback_count": im.get("numeric_readback_count"), "numeric_readback_all_successful": im.get("numeric_readback_all_successful"), "pre_opt_x_hash": pre.get("opt_x_hash"), "pre_opt_p_hash": pre.get("opt_p_hash"), "production_max_iter_not_forced_to_1": s.get("production_max_iter_not_forced_to_1"), "truncated_max_iter_forced_to_1": s.get("truncated_max_iter_forced_to_1")})
    eps_payload = {"epsilon_vectors": [{"cell_label": r.get("cell_label"), "arm_id": r.get("arm_id"), "horizon": r.get("horizon"), "epsilon_capture": r.get("epsilon_capture")} for r in cell_results]}
    write_json(epsilon_path, eps_payload)
    return cell_csv


def main() -> int:
    created = now_utc(); stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir: Optional[Path] = None
    try:
        execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
        run_dir = ROOT / "research_artifacts" / "aws_diagnostics" / f"{NAME}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        marker = f"vehicle-tc2-oc-epsilon-1b-converged-contract-gate-{stamp}"
        preconditions = verify_prior_and_preconditions(run_dir)
        if preconditions["new_script_not_t_c5_source"] is not True:
            raise ContractError("P1 failed: v34z2 script identity not distinct from T-C5 source")
        load_modules()
        install_nlpsol_patch()
        context, terminals, load_meta = load_context_and_terminals()
        h35_allowance = {"documented_before_solver_calls": True, "utc": now_utc().isoformat(), "rule": "H35 may use relative error <=1e-5 only if H35 is non-converged; other production cells require convergence and <=1e-6."}

        # Pre-launch setup gate over all planned arms, before any low-level solver call.
        setup_records: List[Dict[str, Any]] = []
        for row in SOLVE_ORDER:
            prep = prepare_cell(row, context, terminals, for_presolve_only=True)
            setup_records.append(clean(prep["setup"]))
            del prep
            gc.collect()
        setup_gate = validate_setup_gate([s for s in setup_records if s.get("solve_mode") == "production"])
        if setup_gate["inert_intervention_failure"]:
            raw = {"created_utc": created.isoformat(), "task_id": TASK_ID, "classification": "development_IMPROVED_T_C2_inert_intervention_failure_not_validation_not_test", "preconditions": preconditions, "load_meta": load_meta, "h35_nonconverged_allowance_pre_run": h35_allowance, "prelaunch_setup_records": setup_records, "inert_intervention_gate": setup_gate, "budget_actual": dict(RESOURCE_USAGE), "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False}
            raw_path = run_dir / "raw.json"; write_json(raw_path, raw)
            summary_path = run_dir / "summary.md"; summary_path.write_text("# T-C2 inert-intervention failure\n\nThe pre-launch setup gate failed before any solver call. No initialization-sensitivity claim is made.\n", encoding="utf-8")
            pass_evidence = {"converged_cells_reported_with_relative_errors": False, "inert_intervention_gate_evaluated_and_reported": True, "no_plant_training_validation_or_test_usage": True, "numeric_readback_reported_for_intended_primals": False, "per_stage_epsilon_vector_persisted_with_label_and_dimension": False, "solver_and_whole_decision_timing_reported_with_n": False}
            completed_path = run_dir / "completed.json"
            completed = {"status": "complete", "hard_pass": False, "task_id": TASK_ID, "created_utc": created.isoformat(), "classification": raw["classification"], "summary": rel(summary_path), "raw": rel(raw_path), "budget_actual": dict(RESOURCE_USAGE), "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "headline": {"T_C2_pass": False, "inert_intervention_failure": True, "new_solver_calls": 0}, "pass_evidence": pass_evidence}
            write_json(completed_path, completed)
            execution_contract.record_outcome(ROOT, "scientific_result", dict(RESOURCE_USAGE), pass_evidence)
            print(json.dumps(clean({"completed": rel(completed_path), "headline": completed["headline"], "pass_evidence": pass_evidence}), sort_keys=True), flush=True)
            return 0

        cell_results: List[Dict[str, Any]] = []
        actual_h15_pre: Dict[str, Dict[str, Any]] = {}
        for idx, row in enumerate(SOLVE_ORDER):
            prep = prepare_cell(row, context, terminals, for_presolve_only=False)
            s = prep["setup"]
            im = s.get("initialization_meta") or {}
            if int(im.get("assignments_attempted", 0) or 0) == 0 or int(im.get("assignments_succeeded", 0) or 0) == 0 or im.get("numeric_readback_all_successful") is not True:
                raise ContractError("actual setup became inert before solve for " + str(row["cell_label"]))
            result = solve_prepared(prep, idx)
            cell_results.append(clean(result))
            if row["cell_label"] in ("H15_canonical", "H15_goal_facing"):
                actual_h15_pre[row["cell_label"]] = (result.get("solve_event") or {}).get("pre") or {}
                if set(actual_h15_pre) == {"H15_canonical", "H15_goal_facing"}:
                    c = actual_h15_pre["H15_canonical"]; g = actual_h15_pre["H15_goal_facing"]
                    if c.get("opt_x_hash") == g.get("opt_x_hash") and c.get("opt_p_hash") == g.get("opt_p_hash"):
                        break
            progress = {"cells_solved": len(cell_results), "new_solver_calls": RESOURCE_USAGE["solver_calls"], "last_cell": row["cell_label"], "validation64_bank_opened": False, "sealed_test_accessed": False}
            write_json(run_dir / "progress.json", progress)
            print(json.dumps(clean(progress), sort_keys=True), flush=True)
            del prep
            gc.collect()
            if any(r.get("hard_defect_converged_rel_error_gt_1e_minus_4") for r in cell_results):
                break
        h15_actual_inert = False
        if set(actual_h15_pre) == {"H15_canonical", "H15_goal_facing"}:
            h15_actual_inert = bool(actual_h15_pre["H15_canonical"].get("opt_x_hash") == actual_h15_pre["H15_goal_facing"].get("opt_x_hash") and actual_h15_pre["H15_canonical"].get("opt_p_hash") == actual_h15_pre["H15_goal_facing"].get("opt_p_hash"))

        prod = [r for r in cell_results if r.get("solve_mode") == "production"]
        prod_by_label = {r.get("cell_label"): r for r in prod}
        all_prod_present = all(label in prod_by_label for label in ["H12_canonical", "H15_canonical", "H35_canonical", "H15_goal_facing"])
        no_hard = not any(r.get("hard_defect_converged_rel_error_gt_1e_minus_4") for r in prod)
        prod_contract_pass = bool(all_prod_present and no_hard and all(r.get("objective_contract_cell_pass") is True for r in prod))
        vectors_ok = bool(cell_results and all((r.get("epsilon_capture") or {}).get("epsilon_dimension_per_stage", 0) is not None and len((r.get("epsilon_capture") or {}).get("per_stage_epsilon_vector", [])) >= int(r.get("horizon", 0)) for r in cell_results))
        timing_ok = bool(cell_results and timing_summary([(r.get("solve_event") or {}).get("solver_wall_s") for r in cell_results]).get("n") == len(cell_results) and timing_summary([r.get("controller_get_action_wall_s") for r in cell_results]).get("n") == len(cell_results))
        numeric_ok = bool(setup_gate.get("inert_intervention_failure") is False and all((s.get("initialization_meta") or {}).get("numeric_readback_all_successful") is True for s in setup_records))
        pass_evidence = {"converged_cells_reported_with_relative_errors": bool(prod_contract_pass and not h15_actual_inert), "inert_intervention_gate_evaluated_and_reported": True, "no_plant_training_validation_or_test_usage": True, "numeric_readback_reported_for_intended_primals": numeric_ok, "per_stage_epsilon_vector_persisted_with_label_and_dimension": vectors_ok, "solver_and_whole_decision_timing_reported_with_n": timing_ok}
        hard_pass = all(pass_evidence.values())
        epsilon_path = run_dir / "epsilon_vectors.json"
        cell_csv = write_csvs(run_dir, cell_results, setup_records, epsilon_path)
        solver_times = [(r.get("solve_event") or {}).get("solver_wall_s") for r in cell_results]
        whole_times = [r.get("controller_get_action_wall_s") for r in cell_results]
        raw = {"created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "server_api_token_audit": read_api_total_tokens(), "task_id": TASK_ID, "classification": "development_IMPROVED_T_C2_converged_contract_gate_not_validation_not_test", "preconditions": preconditions, "load_meta": load_meta, "h35_nonconverged_allowance_pre_run": h35_allowance, "patched_modules": PATCHED_MODULES, "cells_frozen": SOLVE_ORDER, "prelaunch_setup_records": setup_records, "inert_intervention_gate": {**setup_gate, "H15_actual_solver_entry_identical_opt_x_and_opt_p": h15_actual_inert, "actual_h15_solver_entry_pre": actual_h15_pre}, "cell_results": cell_results, "timing_solver_wall_s": timing_summary(solver_times), "timing_whole_decision_get_action_wall_s": timing_summary(whole_times), "truncated_cell_timing_not_deployable_estimate": True, "objective_contract_verdict": {"all_production_cells_present": all_prod_present, "production_contract_pass": prod_contract_pass, "hard_defect": not no_hard, "T_C2_pass": hard_pass}, "budget_declared": {"solver_calls_cap_ceiling": SOLVER_CAP, "solver_calls_scheduled": SCHEDULED_SOLVES, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}, "budget_actual": dict(RESOURCE_USAGE), "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "interpretation_limits": ["opened development source242_slot0_branch_start only", "objective-contract diagnostic only", "not validation64", "not sealed/final test", "no plant rollout", "does not close terminal-treatment opportunity A6", "truncated max_iter=1 timing is not deployable"]}
        raw_path = run_dir / "raw.json"; write_json(raw_path, raw)
        summary_path = run_dir / "summary.md"
        summary_path.write_text(
            "# T-C2 OC-epsilon-1b converged contract gate\n\n"
            f"UTC: `{created.isoformat()}`. Task `{TASK_ID}`.\n\n"
            f"T-C2 local hard_pass: `{hard_pass}`. New solver calls: `{RESOURCE_USAGE['solver_calls']}` / cap `{SOLVER_CAP}`.\n\n"
            f"Inert-intervention gate: `{raw['inert_intervention_gate']}`.\n\n"
            f"Production objective contract pass: `{prod_contract_pass}`. Timing summaries are in raw; the H12 max_iter=1 control timing is explicitly not deployable.\n\n"
            f"Artifacts: `{rel(raw_path)}`, `{rel(cell_csv)}`, `{rel(epsilon_path)}`.\n",
            encoding="utf-8",
        )
        backup_request = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_T_C2_OC_EPSILON_1B_CONVERGED_CONTRACT_GATE_{stamp}.json"
        write_json(backup_request, {"request": "backup_after_t_c2_oc_epsilon_1b_converged_contract_gate", "created_utc": created.isoformat(), "must_cover": [rel(Path(__file__).resolve()), rel(run_dir), rel(backup_request), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", rel(RESPONSE_LOG)], "new_solver_calls": RESOURCE_USAGE["solver_calls"], "validation64_bank_opened": False, "sealed_test_accessed": False})
        state_path = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_t_c2_oc_epsilon_1b_converged_contract_gate.md"
        doc_block = f"""
<!-- {marker} -->
## T-C2 OC-epsilon-1b converged contract gate

UTC: {created.isoformat()}. Local task hard_pass `{hard_pass}`; new solver calls `{RESOURCE_USAGE['solver_calls']}`; plant/training/validation64/sealed-test all zero. Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(cell_csv)}`, `{rel(epsilon_path)}`. The H12 max_iter=1 timing is explicitly not deployable. Backup request: `{rel(backup_request)}`.
""".strip()
        for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
            append_if_missing(doc, marker, doc_block)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("# Continue state after T-C2\n\n" + doc_block + "\n\nNext: if T-C2 gates passed and backup is verified, continue with the new Opus plan's zero-resource T-C4 ledger; otherwise return raw evidence to the lead for diagnosis.\n", encoding="utf-8")
        with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([created.isoformat(), NAME, raw["classification"], "canonical_solve_no_training_seed", "opened_development_source242_slot0_branch_start_only_no_validation64_no_sealed_test", RESOURCE_USAGE["solver_calls"], 0, 0, 0, 0, False, rel(run_dir / "completed.json"), marker])
        completed_path = run_dir / "completed.json"
        completed = {"status": "complete", "hard_pass": hard_pass, "created_utc": created.isoformat(), "classification": raw["classification"], "task_id": TASK_ID, "summary": rel(summary_path), "raw": rel(raw_path), "cell_metrics_csv": rel(cell_csv), "epsilon_vectors": rel(epsilon_path), "backup_request": rel(backup_request), "state": rel(state_path), "budget_actual": dict(RESOURCE_USAGE), "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "headline": {"T_C2_pass": hard_pass, "production_contract_pass": prod_contract_pass, "new_solver_calls": RESOURCE_USAGE["solver_calls"], "h15_actual_inert": h15_actual_inert, "all_production_cells_present": all_prod_present, "solver_timing_n": timing_summary(solver_times).get("n"), "whole_decision_timing_n": timing_summary(whole_times).get("n")}, "pass_evidence": pass_evidence}
        hash_paths = [Path(__file__).resolve(), raw_path, summary_path, cell_csv, epsilon_path, backup_request, state_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists()}
        write_json(completed_path, completed)
        execution_contract.record_outcome(ROOT, "scientific_result", dict(RESOURCE_USAGE), pass_evidence)
        print(json.dumps(clean({"completed": rel(completed_path), "summary": rel(summary_path), "headline": completed["headline"], "pass_evidence": pass_evidence, "backup_request": rel(backup_request), "server_api_token_audit": raw["server_api_token_audit"]}), sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        return write_failure(run_dir, created, f"unexpected T-C2 error: {type(exc).__name__}: {exc}", RESOURCE_USAGE, engineering=all(v == 0 for v in RESOURCE_USAGE.values()))


if __name__ == "__main__":
    raise SystemExit(main())
