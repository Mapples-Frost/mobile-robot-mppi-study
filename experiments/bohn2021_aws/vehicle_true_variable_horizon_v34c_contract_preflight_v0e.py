#!/usr/bin/env python3
"""v34c/v0e zero-solve contract preflight wrapper: saved-TVP scalarization repair.

Narrow operational repair for Astra 20260930T084456Z Task1.  The v0d wrapper
repaired typed opt_x initialization and restored the saved object-noise seed, but
then failed before the controlled no-solve intercept in ``TTAHMPC.get_action``:

    TypeError: unsupported operand type(s) for -: 'float' and 'dict'

Cause: the direct context reconstruction passed saved TVP records of the form
``{"true": [...], "forecast": [...]}`` directly as controller forecast values.
The controller's online path normally receives numeric arrays from
``TVP.get_values``; for object TVPs it then constructs ``obj_data`` from the
numeric first element and applies its own deterministic forecast perturbation
using ``object_noise_seed``.  Passing saved dictionaries therefore made
``obj_data[obj_i][comp][0]`` a dict instead of a scalar.

This wrapper keeps the frozen contexts, terminal modes, horizons,
initializations, 0-solve budget, no-validation/no-test access, v0d typed-index
repair and v0c explicit-goal repair unchanged.  It only converts the copied
saved TVP suffix used by ``ctrl.get_action`` into strict finite numeric sequences
before the no-solve intercept.  Object TVPs use the saved true component as the
base because TTAHMPC recomputes the forecast-noise profile from the restored
``reference.ns`` seed; non-object TVPs use true plus forecast if both are present.
"""
from __future__ import annotations

import copy
import datetime as dt
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_v34c_contract_preflight_v0d as v0d  # noqa:E402

c = v0d.c


def _component_sum(value: Any) -> Optional[float]:
    """Finite scalar sum for TVP component lists/scalars; None if not scalarizable."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("true", "value", "actual"):
            if key in value:
                out = _component_sum(value.get(key))
                if out is not None:
                    return out
        if "forecast" in value:
            return _component_sum(value.get("forecast"))
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        vals: List[float] = []
        for item in value:
            out = _component_sum(item)
            if out is None:
                return None
            vals.append(float(out))
        total = float(math.fsum(vals))
        return total if math.isfinite(total) else None
    try:
        a = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if a.size != 1:
        return None
    out = float(a[0])
    return out if math.isfinite(out) else None


def _saved_tvp_entry_to_numeric(name: str, entry: Any) -> float:
    """Convert one saved TVP entry to the numeric value expected by get_action."""
    if isinstance(entry, Mapping):
        true_val = _component_sum(entry.get("true")) if "true" in entry else None
        forecast_val = _component_sum(entry.get("forecast")) if "forecast" in entry else None
        if true_val is not None:
            # Object TVP forecasts are reconstructed inside TTAHMPC.get_action
            # from object_noise_seed; feeding saved forecast dictionaries here
            # double-counts/changes the controller path and caused v0d failure.
            if str(name).startswith("obj_"):
                return float(true_val)
            if forecast_val is not None:
                return float(true_val + forecast_val)
            return float(true_val)
        if forecast_val is not None:
            return float(forecast_val)
    out = _component_sum(entry)
    if out is None:
        raise c.ContractError(f"TVP {name} entry is not a finite scalar: {entry!r}")
    return float(out)


def _numeric_tvp_sequence(name: str, values: Any, h: int) -> List[float]:
    if not isinstance(values, (list, tuple)):
        scalar = _saved_tvp_entry_to_numeric(name, values)
        return [scalar for _ in range(h + 1)]
    if len(values) < h + 1:
        raise c.ContractError(f"TVP {name} length {len(values)} < H+1={h + 1} after saved suffix shift")
    out: List[float] = []
    for i, entry in enumerate(values[: h + 1]):
        val = _saved_tvp_entry_to_numeric(name, entry)
        if not math.isfinite(val):
            raise c.ContractError(f"TVP {name}[{i}] converted to non-finite value {val!r}")
        out.append(float(val))
    return out


def _numericize_shifted_tvp(raw_shifted: Mapping[str, Any], h: int) -> Dict[str, List[float]]:
    numeric: Dict[str, List[float]] = {}
    for name, values in raw_shifted.items():
        numeric[str(name)] = _numeric_tvp_sequence(str(name), values, h)
    if "hend" not in numeric:
        # AHMPC.get_action mutates this vector at indices >= n_horizon.  In the
        # normal controller path it is a constant-zero TVP; provide the same
        # explicit numeric default if absent from saved case TVPs.
        numeric["hend"] = [0.0 for _ in range(h + 1)]
    return numeric


def setup_context_direct_v0e(env: Any, context: Mapping[str, Any], h: int) -> Dict[str, Any]:
    meta = v0d.setup_context_direct_v0d(env, context, h)
    raw_shifted = copy.deepcopy(meta.get("shifted_tvp") or {})
    if not isinstance(raw_shifted, Mapping) or not raw_shifted:
        raise c.ContractError("v0e expected non-empty shifted_tvp mapping from v0d direct setup")
    numeric = _numericize_shifted_tvp(raw_shifted, int(h))
    # Make the controller bookkeeping match the numeric values used by the
    # upcoming ctrl.get_action call; env.control_system.tvps retains the saved
    # raw values as a reconstruction receipt and is not consumed before solve.
    env.control_system.controller._tvp_data = copy.deepcopy(numeric)
    meta["shifted_tvp_raw_schema"] = "saved_case_TVP_records_true_forecast"
    meta["shifted_tvp_raw_hash"] = c.canonical_hash(raw_shifted)
    meta["shifted_tvp_numeric_hash"] = c.canonical_hash(numeric)
    meta["shifted_tvp_numeric_lengths"] = {k: len(v) for k, v in numeric.items()}
    meta["shifted_tvp_numeric_first3"] = {k: v[:3] for k, v in numeric.items()}
    meta["shifted_tvp_numeric_conversion_note"] = (
        "object TVPs use saved true scalar base and restored reference.ns for "
        "TTAHMPC forecast perturbations; non-object TVPs use true+forecast when "
        "a saved forecast component exists"
    )
    meta["shifted_tvp"] = numeric
    return meta


def patch_runtime() -> None:
    v0d.patch_runtime()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = "vehicle_true_variable_horizon_v34c_contract_preflight_v0e"
    c.NAME = name
    c.STAMP = stamp
    c.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{name}_{stamp}"
    c.ARRAY_DIR = c.RUN_DIR / "arrays"
    c.BACKUP_REQ = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34C_CONTRACT_PREFLIGHT_V0E_{stamp}.json"
    c.STATE = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_v34c_contract_preflight_v0e.md"
    c.MARKER = f"vehicle-v34c-contract-preflight-v0e-{stamp}"
    c.setup_context_direct = setup_context_direct_v0e


def main() -> int:
    patch_runtime()
    return int(c.run())


if __name__ == "__main__":
    raise SystemExit(main())
