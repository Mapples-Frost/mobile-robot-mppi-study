#!/usr/bin/env python3
"""S-TC2H action-payload repair wrapper.

The first S-TC2H run solved all three first MPC decisions but every env.step
raised KeyError(0) because the runner passed a name-keyed control mapping to a
legacy environment path that indexes the action as action[0], action[1]. This
wrapper keeps the same authorized task and prior source intact, monkey-patches
only the action payload adapter to pass an ordered [u_omega, u_s] vector to the
environment, and then delegates to the archived v0 runner. No validation64 or
sealed/final test access is added.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, List, Optional

import vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0 as base


def _scalar(value: Any) -> Optional[float]:
    try:
        if hasattr(value, "full"):
            arr = value.full()
            return float(arr[0]) if getattr(arr, "size", 0) else None
        if hasattr(value, "cat"):
            arr = value.cat
            return float(arr[0]) if len(arr) else None
        if isinstance(value, (list, tuple)):
            return _scalar(value[0]) if value else None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _flat_sequence(value: Any) -> Optional[List[float]]:
    try:
        if hasattr(value, "full"):
            arr = value.full()
            return [float(x) for x in arr.reshape(-1)]
        if hasattr(value, "cat"):
            return [float(x) for x in value.cat]
        if isinstance(value, (list, tuple)):
            return [float(x) for x in value]
    except Exception:
        return None
    return None


def repaired_action_payload(action: Any, ctrl: Any) -> Any:
    """Return a legacy-env compatible ordered control vector.

    Prefer explicit action objects from the controller when they are already a
    sequence/DMStruct. For name-keyed mappings, use the vehicle controller order
    [u_omega, u_s]. The previous v0 returned {'u_omega':..., 'u_s':...}, which
    made env.step fail with KeyError(0) before plant advancement.
    """
    seq = _flat_sequence(action)
    if seq is not None and len(seq) >= 2:
        return [seq[0], seq[1]]
    if isinstance(action, Mapping):
        vals = []
        for key in ("u_omega", "u_s"):
            vals.append(_scalar(action.get(key)))
        if all(v is not None for v in vals):
            return [float(vals[0]), float(vals[1])]
    cur = getattr(ctrl, "current_input", None)
    seq = _flat_sequence(cur)
    if seq is not None and len(seq) >= 2:
        return [seq[0], seq[1]]
    if isinstance(cur, Mapping):
        vals = []
        for key in ("u_omega", "u_s"):
            vals.append(_scalar(cur.get(key)))
        if all(v is not None for v in vals):
            return [float(vals[0]), float(vals[1])]
    return action


base.action_payload = repaired_action_payload
base.NAME = "vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0b_action_payload_repair"

if __name__ == "__main__":
    raise SystemExit(base.main())
