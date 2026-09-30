#!/usr/bin/env python3
"""v34n v0b operational repair wrapper for A13c-3.

The first v34n attempt failed before any solver call because it reused
base.load_contexts(), which reconstructs an unrelated c13 context and assumes a
scalar input where the preserved trace now stores list-valued inputs.  Opus
A13c-3 explicitly requires only the already-opened source242/slot0/V15_shared
cell, so this wrapper preserves the v34n protocol and monkey-patches the context
loader to construct only that authorized source242 context.  It does not change
the objective formula, cells, gates, or budgets.
"""
from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import vehicle_true_variable_horizon_v34n_nonconverged_objective_contract_probe_v0 as v34n

ROOT = v34n.ROOT
NAME = "vehicle_true_variable_horizon_v34n_nonconverged_objective_contract_probe_v0b"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_source242_only_contexts() -> List[Dict[str, Any]]:
    """Return only the A13c-3-authorized source242 context.

    This avoids touching the unrelated v19_c13 trace loader that caused the v0
    operational failure.  The returned context is byte-for-byte equivalent in the
    used fields to base.load_contexts()[0] before that function reaches the c13
    conversion.
    """
    specs = v34n.base.v29.build_state_specs()
    source242 = v34n.base.find_spec("v27_case09_slot0_early_risk", specs)
    return [
        {
            "context_id": "source242_slot0_branch_start",
            "state_label": "v27_case09_slot0_early_risk",
            "source": "v29/v33 selected branch state, original branch start; v0b source242-only repair avoids unrelated c13 loader",
            "case_snapshot": copy.deepcopy(source242["case_snapshot"]),
            "branch_step": int(source242["branch_step"]),
            "tvp_start_index": int(source242["branch_step"]),
            "state": v34n.base.state_clean(source242["branch_previous_state"]),
            "previous_input": {"u_omega": 0.0, "u_s": 0.0},
            "previous_input_source": "Astra/Opus-specified v33 branch-reset zero-input semantics; unchanged from base source242 context",
            "horizons": [12, 15, 35],
        }
    ]


def patch_module_identity() -> None:
    # Preserve the original implementation while giving this repaired run unique
    # artifact paths and source provenance.
    v34n.__file__ = __file__
    v34n.NAME = NAME
    v34n.STAMP = STAMP
    v34n.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
    v34n.STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34n_v0b_nonconverged_objective_contract_probe.md"
    v34n.BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34N_V0B_NONCONVERGED_OBJECTIVE_CONTRACT_PROBE_{STAMP}.json"
    v34n.REQUEST_ID = f"v34n-v0b-a13c3-nonconverged-objective-contract-{STAMP}"
    v34n.MARKER = f"vehicle-v34n-v0b-a13c3-nonconverged-objective-contract-{STAMP}"
    v34n.base.load_contexts = load_source242_only_contexts


def run(argv: Optional[Sequence[str]] = None) -> int:
    patch_module_identity()
    return v34n.run(argv)


if __name__ == "__main__":
    raise SystemExit(run())
