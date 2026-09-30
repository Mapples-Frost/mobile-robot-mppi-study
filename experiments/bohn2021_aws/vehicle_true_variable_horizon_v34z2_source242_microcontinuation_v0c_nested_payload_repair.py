#!/usr/bin/env python3
"""S-TC2H2 nested-action-payload repair wrapper for source242 microcontinuation.

The S-TC2H v0 run passed a name-keyed control mapping to env.step and failed
with KeyError(0).  The v0b wrapper changed that to a flat [u_omega, u_s]
sequence, but the legacy low-level plant path then failed with
TypeError("'float' object is not subscriptable"), implying it indexes controls
as action[0][0], action[1][0] (or equivalent).  This wrapper keeps the same
source242 state, horizons, terminal mode, objective, initialization, split and
budgets, and only adapts the control payload to a 2x1 numeric numpy array before
env.step.  It also requires a verified external backup after the v0b failed
receipt before any solver/plant resources are spent.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

import vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0b_action_payload_repair as v0b

base = v0b.base
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
PREV_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0b_action_payload_repair_20260930T171033Z/completed.json"
NAME = "vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0c_nested_payload_repair"


def _scalar(value: Any) -> Optional[float]:
    try:
        if hasattr(value, "full"):
            arr = value.full()
            out = float(arr.reshape(-1)[0]) if getattr(arr, "size", 0) else None
        elif hasattr(value, "cat"):
            out = float(value.cat[0]) if len(value.cat) else None
        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float).reshape(-1)
            out = float(arr[0]) if arr.size else None
        else:
            out = float(value)
        return out if out is not None and math.isfinite(out) else None
    except Exception:
        return None


def _flat_sequence(value: Any) -> Optional[List[float]]:
    try:
        if hasattr(value, "full"):
            arr = np.asarray(value.full(), dtype=float).reshape(-1)
        elif hasattr(value, "cat"):
            arr = np.asarray(value.cat, dtype=float).reshape(-1)
        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float).reshape(-1)
        else:
            return None
        return [float(x) for x in arr] if arr.size >= 2 else None
    except Exception:
        return None


def _nested_from_values(vals: List[float]) -> np.ndarray:
    # Legacy plant code has now shown both positional and subscripted control
    # access.  A shape-(2,1) array supports action[0][0] and action[1][0].
    return np.asarray([[float(vals[0])], [float(vals[1])]], dtype=float)


def nested_action_payload(action: Any, ctrl: Any) -> Any:
    seq = _flat_sequence(action)
    if seq is not None:
        return _nested_from_values(seq)
    if isinstance(action, Mapping):
        vals = [_scalar(action.get("u_omega")), _scalar(action.get("u_s"))]
        if all(v is not None for v in vals):
            return _nested_from_values([float(vals[0]), float(vals[1])])
    cur = getattr(ctrl, "current_input", None)
    seq = _flat_sequence(cur)
    if seq is not None:
        return _nested_from_values(seq)
    if isinstance(cur, Mapping):
        vals = [_scalar(cur.get("u_omega")), _scalar(cur.get("u_s"))]
        if all(v is not None for v in vals):
            return _nested_from_values([float(vals[0]), float(vals[1])])
    return action


_original_verify = base.verify_pre_resource_backup_and_priors


def verify_pre_resource_backup_and_priors_v0c() -> dict:
    out = _original_verify()
    if not PREV_COMPLETED.exists():
        raise base.MicroError("previous v0b completed marker missing before v0c repair")
    prev = base.read_json(PREV_COMPLETED)
    if prev.get("status") != "complete" or (prev.get("budget_actual") or {}).get("solver_calls") != 3:
        raise base.MicroError("previous v0b evidence marker is not the expected preserved 3-solver failed-result receipt")
    prev_time = base.parse_time(prev.get("created_utc")) or dt.datetime.fromtimestamp(PREV_COMPLETED.stat().st_mtime, dt.timezone.utc)
    candidates = base.verified_backup_candidates(prev_time)
    if not candidates:
        raise base.MicroError("no verified external backup after v0b failure/completion before v0c solver/plant resources")
    out["post_v0b_backup_verified_before_solver_or_plant_steps"] = True
    out["post_v0b_backup_candidates"] = candidates
    out["previous_v0b_completed"] = {"path": base.rel(PREV_COMPLETED), "sha256": base.sha256(PREV_COMPLETED), "created_utc": prev.get("created_utc"), "budget_actual": prev.get("budget_actual")}
    return out


base.action_payload = nested_action_payload
base.verify_pre_resource_backup_and_priors = verify_pre_resource_backup_and_priors_v0c
base.NAME = NAME
base.__dict__["__file__"] = str(SOURCE)

if __name__ == "__main__":
    raise SystemExit(base.main())
