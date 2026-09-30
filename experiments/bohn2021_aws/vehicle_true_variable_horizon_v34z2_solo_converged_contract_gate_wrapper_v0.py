#!/usr/bin/env python3
"""Solo-compatible wrapper for the v34z2 T-C2 solver-bearing gate.

This wrapper is intentionally thin. It verifies the structured execution
snapshot for the current temporary GPT-5.5 solo plan, binds the imported frozen
v34z2 gate to that request/task ID, applies the already preflighted source242
in-memory goal extractor repair, and delegates to the original solver-bearing
implementation. It must not access validation64 or sealed/final test data.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments" / "bohn2021_aws"
SERVICE_DIR = ROOT / "scripts" / "research_service"
for _p in (SERVICE_DIR, AWS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import execution_contract  # type: ignore  # noqa: E402
import vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0 as gate  # type: ignore  # noqa: E402

TASK_ID = "S-TC2-OC-epsilon-1b-converged-contract-gate-wrapper-v0"
GATE_SOURCE = AWS_DIR / "vehicle_true_variable_horizon_v34z2_converged_contract_gate_v0.py"
EXPECTED_GATE_SHA256 = "66dbf84e4a7917d824152cae1cac4da77c51efff576cd4775e8cd1be29dbcc97"


class WrapperContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return value.resolve().relative_to(ROOT.resolve()).as_posix()
        except Exception:
            return str(value)
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        try:
            return clean(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return clean(value.tolist())
        except Exception:
            pass
    return value


def install_source242_goal_patch() -> None:
    """Patch base.extract_goal_xy to the source242 v34q endpoint[61] rule.

    S-TC2C2 already verified this exact in-memory repair path with zero solver
    resources. The patch is installed before gate.main() so the gate's own
    load_modules() call becomes a no-op and all subsequent prepare_cell calls use
    the patched base module.
    """
    gate.load_modules()
    base = gate.MODULES["base"]
    v34q = gate.MODULES["v34q"]
    original = getattr(base, "extract_goal_xy", None)
    events = []

    def patched_extract_goal_xy(case: Mapping[str, Any], shifted_tvp: Mapping[str, Any]) -> Any:
        got = v34q.robust_goal_from_saved_case(case)
        events.append({"ok": got.get("ok") is True, "raw": clean(got)})
        if got.get("ok") is not True:
            raise WrapperContractError("source242 robust goal extraction failed: %r" % (got,))
        if str(got.get("source")) != "case.tvp.trajectory_endpoint[61]" or int(got.get("endpoint_index")) != 61:
            raise WrapperContractError("unexpected source242 goal provenance: %r" % (got,))
        return float(got["goal_x"]), float(got["goal_y"]), str(got["source"])

    setattr(base, "extract_goal_xy", patched_extract_goal_xy)
    setattr(base, "_solo_tc2_original_extract_goal_xy_repr", repr(original))
    setattr(base, "_solo_tc2_goal_patch_events", events)
    setattr(base, "_solo_tc2_goal_patch_rule", "v34q.robust_goal_from_saved_case with required case.tvp.trajectory_endpoint[61]")


def main() -> int:
    snapshot = execution_contract.runtime_snapshot(ROOT)
    if snapshot is None:
        raise WrapperContractError("missing structured execution snapshot")
    ready = snapshot.get("ready") or {}
    request_id = ready.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise WrapperContractError("current solo request_id is unavailable in execution snapshot")
    gate_sha = sha256(GATE_SOURCE)
    if gate_sha != EXPECTED_GATE_SHA256:
        raise WrapperContractError("v34z2 gate source hash mismatch: %s" % gate_sha)
    if snapshot.get("task", {}).get("task_id") != TASK_ID:
        raise WrapperContractError("wrapper launched under unexpected task_id: %r" % (snapshot.get("task", {}).get("task_id"),))

    gate.EXPECTED_REQUEST = request_id
    gate.TASK_ID = TASK_ID
    install_source242_goal_patch()
    return int(gate.main())


if __name__ == "__main__":
    raise SystemExit(main())
