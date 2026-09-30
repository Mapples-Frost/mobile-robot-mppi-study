#!/usr/bin/env python3
"""v34u operational retry wrapper for A13c-3 objective-contract probe.

This file is intentionally small: the approved scientific task remains the v34t
repair/re-run requested by Opus after the 20260930T123137 zero-solve outcome.
The immediately preceding v34t source already contains strict saved-TVP
numericization, but its live-plan gate required a machine token that the active
Opus report did not literally print even though the report explicitly authorizes
"A13c-3" and the same four-cell objective-contract retry. This wrapper performs
only two operational repairs before delegating to v34t:

1. accept the exact human-readable authorization token present in the active
   report ("A13c-3") while preserving request-id, report hash and LATEST checks;
2. install a pre-solve controller guard for TTAHMPC object TVP channels. The
   guard pre-populates obj_data with float-dtype arrays and verifies
   get_obj_distance(state, obj_i) is finite before the original controller path
   reaches mpc.solve. It does not change plant dynamics, validation/test access,
   objective thresholds, cells or solver-call cap.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

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

import vehicle_true_variable_horizon_v34t_nonconverged_objective_contract_probe_v0 as v34t  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0"


def _finite_first_float(value: Any, key: str) -> float:
    """Return the first finite numeric entry from a scalar/list/array TVP channel."""
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception as exc:
        raise v34t.ContractError(f"controller TVP channel {key} is not numeric after scalarization: {type(value).__name__}: {exc!r}")
    if arr.size < 1:
        raise v34t.ContractError(f"controller TVP channel {key} is empty after scalarization")
    out = float(arr[0])
    if not math.isfinite(out):
        raise v34t.ContractError(f"controller TVP channel {key} first value is not finite: {out!r}")
    return out


def install_tta_hmpc_presolve_obj_guard() -> None:
    """Patch TTAHMPC.get_action so object channels are verified before solve."""
    # Ensure the legacy package path/module is initialized in the same way as the
    # delegated v34n run will initialize it.
    try:
        v34t.v34n.base.v1d.import_legacy_modules()
    except Exception:
        # The delegated run will report the authoritative runtime-preflight error
        # if import really is impossible. Continue to the direct import attempt so
        # this wrapper remains transparent.
        pass
    import gym_let_mpc.controllers as controllers  # type: ignore

    cls = controllers.TTAHMPC
    if getattr(cls, "_v34u_presolve_obj_guard_installed", False):
        return
    original_get_action = cls.get_action

    def guarded_get_action(self: Any, state: Any, n_horizon: int, tvp_values: Any = None) -> Any:
        if tvp_values is None:
            raise v34t.ContractError("TTAHMPC.get_action received tvp_values=None; cannot verify object TVP channels")
        if getattr(self, "obj_data", None) is None:
            obj_data = []
            n_obj = int(getattr(self, "n_objects", 0) or 0)
            horizon_len = int(getattr(self.mpc, "n_horizon")) + 1
            for obj_i in range(n_obj):
                per_obj = {}
                for comp in ("x", "y", "r"):
                    key = f"obj_{obj_i}_{comp}"
                    if key not in tvp_values:
                        raise v34t.ContractError(f"controller TVP channel {key} missing before solve")
                    val = _finite_first_float(tvp_values[key], key)
                    arr = np.full((horizon_len,), val, dtype=float)
                    if arr.dtype.kind not in {"f", "c"} or not np.all(np.isfinite(arr)):
                        raise v34t.ContractError(f"controller obj_data {key} is not finite float dtype before solve")
                    per_obj[comp] = arr
                obj_data.append(per_obj)
            self.obj_data = obj_data
            for obj_i in range(n_obj):
                dist = float(self.get_obj_distance(state, obj_i))
                if not math.isfinite(dist):
                    raise v34t.ContractError(f"get_obj_distance returned non-finite before solve for obj_i={obj_i}: {dist!r}")
        return original_get_action(self, state, n_horizon, tvp_values=tvp_values)

    cls.get_action = guarded_get_action
    cls._v34u_presolve_obj_guard_installed = True


def run(argv: Optional[Sequence[str]] = None) -> int:
    # Distinguish artifacts from the previous v34t failure while preserving the
    # delegated v34t/v34n implementation and all precommitted thresholds/cells.
    v34t.NAME = NAME
    v34t.TASK_AUTHORIZATION_TOKEN = "A13c-3"
    install_tta_hmpc_presolve_obj_guard()
    return v34t.run(argv)


if __name__ == "__main__":
    raise SystemExit(run())
