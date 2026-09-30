#!/usr/bin/env python3
"""v34c contract preflight wrapper with explicit trajectory-goal reconstruction.

Operational repair for the v34c zero-solve contract preflight after the v0b
attempt failed before any preflight cell with:

    could not reconstruct goal_x/goal_y from case/tvp; refusing to use
    observation heuristic

The missing primitive is source-specific goal recovery for saved TTAHMPC vehicle
cases.  This wrapper keeps the Astra 20260930T084456Z gate and the v34c
zero-solve contract unchanged, but patches the imported goal extractor to use
explicit saved-case reference / trajectory TVP data only:

* direct explicit goal fields if present;
* `reference.theta_r` and `reference.traj_steps` combined with the saved
  `trajectory_x`/`trajectory_y` endpoint at `traj_steps-1` when available;
* as a last explicit-reference fallback, `cos(theta_r)*traj_steps*u_s_ref`, with
  `u_s_ref` inferred from the saved trajectory endpoint or the vehicle default
  `3*t_step = 0.3`.

It deliberately does NOT use observation slots 3/4 as goals and does not alter
terminals, horizons, initializations, budgets, validation/test access, or the
24-call solver-probe design.  It still performs 0 NLP solves and 0 plant steps.
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_v34c_contract_preflight_v0 as c  # noqa:E402

CURRENT_REQUEST = "execution-result:20260930T084409_7dd79362"
CURRENT_REPORT = ROOT / "docs/bohn2021_takeover/astra_reviews/20260930T084456Z.md"
CURRENT_SHA = "9888c40d0cb52640ee18cd548e001698236aab2f1414e615f361c04abfa40ba0"

_ORIGINAL_EXTRACT_GOAL_XY = c.m.extract_goal_xy


def _finite_scalar(value: Any) -> Optional[float]:
    """Return a finite scalar from nested JSON/numpy objects, else None."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        # Saved TVP entries use {"true": [v], "forecast": [...]}; explicit
        # scalar/value fields appear in some metadata.  Prefer true values for
        # actual goal reconstruction, never forecast-only unless no true/value is
        # present and the caller explicitly passes such a value.
        for key in ("true", "value", "actual"):
            if key in value:
                out = _finite_scalar(value.get(key))
                if out is not None:
                    return out
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if len(value) == 1:
            return _finite_scalar(value[0])
        # Not a scalar; caller should decide which sequence element is intended.
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size != 1:
        return None
    out = float(arr[0])
    return out if math.isfinite(out) else None


def _sequence_scalars(values: Any) -> Sequence[Optional[float]]:
    if not isinstance(values, (list, tuple)):
        return []
    return [_finite_scalar(v) for v in values]


def _lookup_nested(mapping: Mapping[str, Any], names: Sequence[str]) -> Optional[float]:
    for name in names:
        if name in mapping:
            val = _finite_scalar(mapping.get(name))
            if val is not None:
                return val
    return None


def _explicit_goal_from_mapping(obj: Any, source: str = "case") -> Optional[Tuple[float, float, str]]:
    if not isinstance(obj, Mapping):
        return None
    x_names = ("goal_x", "trajectory_goal_x", "x_goal", "target_x")
    y_names = ("goal_y", "trajectory_goal_y", "y_goal", "target_y")
    gx = _lookup_nested(obj, x_names)
    gy = _lookup_nested(obj, y_names)
    if gx is not None and gy is not None:
        return float(gx), float(gy), source + ".explicit_goal_fields"
    goal = obj.get("goal")
    if isinstance(goal, Mapping):
        gx = _lookup_nested(goal, ("x", "goal_x", "target_x"))
        gy = _lookup_nested(goal, ("y", "goal_y", "target_y"))
        if gx is not None and gy is not None:
            return float(gx), float(gy), source + ".goal"
    for key, value in obj.items():
        if isinstance(value, Mapping):
            got = _explicit_goal_from_mapping(value, source + "." + str(key))
            if got is not None:
                return got
    return None


def _trajectory_endpoint(case: Mapping[str, Any], traj_steps: Optional[float]) -> Optional[Tuple[float, float, str]]:
    tvp = case.get("tvp") if isinstance(case, Mapping) else None
    if not isinstance(tvp, Mapping):
        return None
    xs = _sequence_scalars(tvp.get("trajectory_x"))
    ys = _sequence_scalars(tvp.get("trajectory_y"))
    if not xs or not ys:
        return None
    n = min(len(xs), len(ys))
    if n <= 0:
        return None
    candidate_indices = []
    if traj_steps is not None and math.isfinite(float(traj_steps)):
        candidate_indices.append(max(0, min(n - 1, int(round(float(traj_steps))) - 1)))
    # Also try the final available true trajectory point.  This remains explicit
    # saved TVP evidence, not an observation heuristic.
    candidate_indices.append(n - 1)
    for idx in candidate_indices:
        gx = xs[idx]
        gy = ys[idx]
        if gx is not None and gy is not None:
            return float(gx), float(gy), f"case.tvp.trajectory_endpoint[{idx}]"
    return None


def _reference_theta_goal(case: Mapping[str, Any], traj_steps: Optional[float]) -> Optional[Tuple[float, float, str]]:
    ref = case.get("reference") if isinstance(case, Mapping) else None
    if not isinstance(ref, Mapping):
        return None
    theta_r = _finite_scalar(ref.get("theta_r"))
    if traj_steps is None:
        traj_steps = _finite_scalar(ref.get("traj_steps"))
    if theta_r is None or traj_steps is None:
        return None
    # Prefer to infer u_s_ref from the saved endpoint when possible; otherwise
    # use the vehicle implementation default TTAHMPC.u_s_ref = 3*t_step and the
    # paper/default vehicle t_step=0.1.  The source string exposes this fallback.
    endpoint = _trajectory_endpoint(case, traj_steps)
    if endpoint is not None:
        return endpoint
    u_s_ref = _finite_scalar(ref.get("u_s_ref"))
    src = "case.reference.theta_r_traj_steps"
    if u_s_ref is None:
        # If the full trajectory is present but the requested endpoint is not,
        # an approximate per-step spacing can still identify u_s_ref.  Otherwise
        # use the vehicle config default created in TTAHMPC.__init__.
        tvp = case.get("tvp") if isinstance(case, Mapping) else None
        if isinstance(tvp, Mapping):
            xs = [v for v in _sequence_scalars(tvp.get("trajectory_x")) if v is not None]
            ys = [v for v in _sequence_scalars(tvp.get("trajectory_y")) if v is not None]
            if len(xs) >= 2 and len(ys) >= 2:
                # Median step length is robust to repeated/appended constants.
                ds = [math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(min(len(xs), len(ys)) - 1)]
                ds = [d for d in ds if math.isfinite(d) and d > 1e-12]
                if ds:
                    u_s_ref = float(sorted(ds)[len(ds) // 2])
                    src += "+trajectory_spacing_inferred_u_s_ref"
        if u_s_ref is None:
            u_s_ref = 0.3
            src += "+vehicle_default_u_s_ref_0.3"
    gx = math.cos(float(theta_r)) * float(traj_steps) * float(u_s_ref)
    gy = math.sin(float(theta_r)) * float(traj_steps) * float(u_s_ref)
    return float(gx), float(gy), src


def robust_extract_goal_xy(case: Mapping[str, Any], shifted_tvp: Mapping[str, Any]) -> Tuple[float, float, str]:
    try:
        return _ORIGINAL_EXTRACT_GOAL_XY(case, shifted_tvp)
    except Exception as original_exc:
        original_msg = repr(original_exc)
    explicit = _explicit_goal_from_mapping(case)
    if explicit is not None:
        return explicit
    ref = case.get("reference") if isinstance(case, Mapping) else None
    traj_steps = _finite_scalar(ref.get("traj_steps")) if isinstance(ref, Mapping) else None
    endpoint = _trajectory_endpoint(case, traj_steps)
    if endpoint is not None:
        return endpoint
    theta_goal = _reference_theta_goal(case, traj_steps)
    if theta_goal is not None:
        return theta_goal
    raise c.ContractError(
        "could not reconstruct explicit goal_x/goal_y from saved case reference "
        "or trajectory TVP; observation-slot heuristic remains forbidden; "
        f"original_extract_error={original_msg}"
    )


def patch_runtime() -> None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = "vehicle_true_variable_horizon_v34c_contract_preflight_v0c"
    c.NAME = name
    c.STAMP = stamp
    c.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{name}_{stamp}"
    c.ARRAY_DIR = c.RUN_DIR / "arrays"
    c.BACKUP_REQ = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34C_CONTRACT_PREFLIGHT_V0C_{stamp}.json"
    c.STATE = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_v34c_contract_preflight_v0c.md"
    c.MARKER = f"vehicle-v34c-contract-preflight-v0c-{stamp}"

    # Current authoritative Astra analysis gate.
    c.m.ASTRA_REPORT = CURRENT_REPORT
    c.m.CURRENT_ASTRA_REQUEST = CURRENT_REQUEST
    c.m.CURRENT_ASTRA_SHA = CURRENT_SHA

    # Source-specific operational repair: explicit goal extraction from saved
    # case reference/TVP only.  No observation-derived goal fallback is added.
    c.m.extract_goal_xy = robust_extract_goal_xy


def main() -> int:
    patch_runtime()
    return int(c.run())


if __name__ == "__main__":
    raise SystemExit(main())
