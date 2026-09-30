#!/usr/bin/env python3
"""v34b wrapper repair for the Astra-directed objective-vs-basin solver probe.

This version preserves the frozen v34 design and patches only the observed
pre-solve bookkeeping defect from the first attempt: v33 trace inputs are stored
as one-element lists, so c13 previous_input must be scalarized before direct
controller-context reconstruction. No v33 rollout, training/refit, validation64,
or sealed-test access is added.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

import vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0 as m

STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
NAME = "vehicle_true_variable_horizon_v34b_objective_basin_solver_probe_v0"

# Version the repaired run/artifacts without modifying the failed v34 evidence.
m.NAME = NAME
m.STAMP = STAMP
m.RUN_DIR = m.ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
m.STATE = m.ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34b_objective_basin_solver_probe.md"
m.BACKUP_REQUEST = m.BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V34B_OBJECTIVE_BASIN_SOLVER_PROBE_{STAMP}.json"
m.REQUEST_ID = f"v34b-objective-basin-solver-probe-{STAMP}"
m.MARKER = f"vehicle-v34b-objective-basin-solver-probe-{STAMP}"


def scalarize(value: Any, label: str) -> float:
    """Return a finite scalar from scalar or singleton vector/list values."""
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception as exc:  # pragma: no cover - legacy failure path only
        raise m.ContractError(f"could not scalarize {label}: {value!r}: {exc!r}")
    if arr.size != 1:
        raise m.ContractError(f"expected singleton scalar for {label}, got shape {arr.shape} value {value!r}")
    out = float(arr[0])
    if not math.isfinite(out):
        raise m.ContractError(f"nonfinite scalar for {label}: {value!r}")
    return out


def patched_load_contexts() -> list[dict[str, Any]]:
    specs = m.v29.build_state_specs()
    source242 = m.find_spec("v27_case09_slot0_early_risk", specs)
    c13 = m.find_spec("v19_c13", specs)
    c13_trace_path, c13_rows = m.find_c13_trace()
    c13_step0 = c13_rows[0]
    c13_step1 = c13_rows[1]
    c13_input = c13_step0.get("input") or {}
    contexts = [
        {
            "context_id": "source242_slot0_branch_start",
            "state_label": "v27_case09_slot0_early_risk",
            "source": "v29/v33 selected branch state, original branch start",
            "case_snapshot": copy.deepcopy(source242["case_snapshot"]),
            "branch_step": int(source242["branch_step"]),
            "tvp_start_index": int(source242["branch_step"]),
            "state": m.state_clean(source242["branch_previous_state"]),
            "previous_input": {"u_omega": 0.0, "u_s": 0.0},
            "previous_input_source": "Astra-specified v33 branch-reset zero-input semantics",
            "horizons": [15, 35],
        },
        {
            "context_id": "v19_c13_step1_after_V15_H35_common_state",
            "state_label": "v19_c13",
            "source": m.rel(c13_trace_path) + ": second trace row previous_state",
            "case_snapshot": copy.deepcopy(c13["case_snapshot"]),
            "branch_step": int(c13["branch_step"]),
            "tvp_start_index": int(c13["branch_step"]) + 1,
            "state": m.state_clean(c13_step1.get("previous_state") or {}),
            "previous_input": {str(k): scalarize(v, f"c13_step0.input.{k}") for k, v in c13_input.items()},
            "previous_input_source": m.rel(c13_trace_path) + ": first trace row input (singleton-list scalarized in v34b)",
            "horizons": [12, 35],
        },
    ]
    expected = {"theta": 0.08060330210484318, "x": 13.74497830467862, "y": 2.8552860589337556}
    if m.state_distance(contexts[1]["state"], expected) > 1e-6:
        raise m.ContractError("c13 reconstructed state does not match Astra-specified fixed state")
    return contexts


m.load_contexts = patched_load_contexts


def augment_provenance(exit_code: int) -> None:
    """Record that this run used the v34b wrapper scalarization repair."""
    try:
        m.RUN_DIR.mkdir(parents=True, exist_ok=True)
        wrapper = Path(__file__).resolve()
        base = Path(m.__file__).resolve()
        provenance = {
            "status": "completed" if exit_code == 0 else "failed",
            "created_utc": m.now_utc().isoformat(),
            "repair_scope": "single pre-solve scalarization repair for singleton-list previous_input values in v33 trace rows",
            "base_script": m.rel(base),
            "base_script_sha256": m.sha256(base),
            "wrapper_script": m.rel(wrapper),
            "wrapper_script_sha256": m.sha256(wrapper),
            "no_design_change": True,
            "solver_budget_change": 0,
            "plant_steps_added": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        prov_path = m.RUN_DIR / "v34b_wrapper_provenance.json"
        m.write_json(prov_path, provenance)
        for name in ("raw.json", "completed.json", "failed.json"):
            p = m.RUN_DIR / name
            if not p.exists():
                continue
            obj = m.read_json(p)
            obj["v34b_wrapper_repair_provenance"] = {**provenance, "path": m.rel(prov_path)}
            if name == "completed.json":
                hashes = dict(obj.get("hashes") or {})
                hashes[m.rel(prov_path)] = m.sha256(prov_path)
                hashes[m.rel(wrapper)] = m.sha256(wrapper)
                hashes[m.rel(base)] = m.sha256(base)
                obj["hashes"] = hashes
            m.write_json(p, obj)
    except Exception as exc:  # avoid masking primary result
        print(json.dumps({"v34b_provenance_augmentation_failed": repr(exc)}), flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    code = m.run(argv)
    augment_provenance(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
