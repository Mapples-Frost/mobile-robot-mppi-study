#!/usr/bin/env python3
"""v34s / T-B4 active-plan zero-solve loader gate.

Operational repair wrapper for the prepared v34r loader gate.  It preserves the
v34r/v34o implementation path, but updates the active Opus-plan identity to the
current report 20260930T115516Z_8330f5 and makes the v34q goal-source binding
fail loud on the exact endpoint provenance required by T-B4:

* source must be reference.traj_steps + saved trajectory TVP endpoint;
* provenance must be exactly case.tvp.trajectory_endpoint[i];
* i must equal round(reference.traj_steps)-1, so the silent n-1 fallback cannot
  pass unnoticed;
* previous_input scalarization remains the v34o strict path and its hashes must
  match the frozen values.

Budget: 0 solver calls, 0 plant/env steps, 0 training/refit, validation64=0,
sealed/final test=0.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
REPRO_DIR = ROOT / "experiments/bohn2021_reproduction"
for _p in (AWS_DIR, REPRO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vehicle_true_variable_horizon_v34q_goal_source_enumeration_v0 as v34q  # noqa:E402
import vehicle_true_variable_horizon_v34r_loader_gate_v0 as v34r  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34s_loader_gate_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T115516Z_8330f5.md"
OPUS_SHA = "3acada362e46ec94b6a21c9738e076a2c95ca8db7be95369a560969d3f7eb474"
OPUS_REQUEST = "execution-result:20260930T115434_d4836078"
EXPECTED_SOURCE_PRIORITY = "reference.traj_steps_plus_saved_trajectory_tvp_endpoint"
SOURCE_RE = re.compile(r"^case\.tvp\.trajectory_endpoint\[(\d+)\]$")


def _finite_traj_steps(value: Any) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise v34r.ContractError(f"v34s goal-source contract: traj_steps is not numeric: {value!r}; {exc!r}")
    if not math.isfinite(out):
        raise v34r.ContractError(f"v34s goal-source contract: traj_steps is not finite: {out!r}")
    return out


def _trajectory_length(case: Mapping[str, Any]) -> Optional[int]:
    tvp = case.get("tvp") if isinstance(case, Mapping) else None
    if not isinstance(tvp, Mapping):
        return None
    xs = v34q.sequence_scalars(tvp.get("trajectory_x"))
    ys = v34q.sequence_scalars(tvp.get("trajectory_y"))
    if not xs or not ys:
        return None
    return min(len(xs), len(ys))


def strict_authoritative_goal_xy(case: Mapping[str, Any], shifted_tvp: Mapping[str, Any]) -> Tuple[float, float, str]:
    """Return only the v34q-authorized saved trajectory endpoint goal.

    shifted_tvp is intentionally not used for goal authority; the saved full
    case's reference.traj_steps and trajectory arrays are the provenance checked
    by v34q and required by the active Opus T-B4 gate.
    """
    del shifted_tvp
    got = v34q.robust_goal_from_saved_case(case)
    if got.get("ok") is not True:
        raise v34r.ContractError("v34s authoritative saved-case goal extraction failed: " + repr(got))
    priority = str(got.get("source_priority"))
    if priority != EXPECTED_SOURCE_PRIORITY:
        raise v34r.ContractError(f"v34s goal-source contract: unexpected source_priority={priority!r}; expected {EXPECTED_SOURCE_PRIORITY!r}; got={got!r}")
    source = str(got.get("source"))
    m = SOURCE_RE.match(source)
    if not m:
        raise v34r.ContractError(f"v34s goal-source contract: provenance is not case.tvp.trajectory_endpoint[i]: {source!r}; got={got!r}")
    source_idx = int(m.group(1))
    endpoint_idx = got.get("endpoint_index")
    if endpoint_idx is None or int(endpoint_idx) != source_idx:
        raise v34r.ContractError(f"v34s goal-source contract: source index {source_idx} != endpoint_index {endpoint_idx!r}; got={got!r}")
    traj_steps = _finite_traj_steps(got.get("traj_steps"))
    expected_idx = int(round(traj_steps)) - 1
    if source_idx != expected_idx:
        raise v34r.ContractError(f"v34s goal-source contract: endpoint index {source_idx} != round(traj_steps)-1 {expected_idx}; got={got!r}")
    n = _trajectory_length(case)
    if n is None or source_idx < 0 or source_idx >= n:
        raise v34r.ContractError(f"v34s goal-source contract: endpoint index {source_idx} out of saved trajectory range n={n}; got={got!r}")
    return float(got["goal_x"]), float(got["goal_y"]), source


def patch_v34r_for_current_active_plan() -> None:
    v34r.__file__ = __file__
    v34r.NAME = NAME
    v34r.STAMP = STAMP
    v34r.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
    v34r.STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34s_loader_gate.md"
    v34r.BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34S_LOADER_GATE_{STAMP}.json"
    v34r.OPUS_REPORT = OPUS_REPORT
    v34r.OPUS_SHA = OPUS_SHA
    v34r.OPUS_REQUEST = OPUS_REQUEST
    v34r.MARKER = f"vehicle-v34s-loader-gate-{STAMP}"
    v34r.robust_extract_goal_xy_v34r = strict_authoritative_goal_xy  # type: ignore[assignment]


def run(argv: Optional[Sequence[str]] = None) -> int:
    patch_v34r_for_current_active_plan()
    return v34r.run(argv)


if __name__ == "__main__":
    raise SystemExit(run())
