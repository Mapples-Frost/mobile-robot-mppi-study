#!/usr/bin/env python3
"""S-TC2H8 v0j task-ID binding repair wrapper.

The diagnostic-split launch of v0i reached the script and consumed zero solver,
plant, training, validation, or test resources, but stopped before the intended
backup dependency gate because the structured plan task ID intentionally included
the diagnostic-launch suffix while v0i's base.main() still required the older
internal task ID.  This wrapper leaves the v0i raw-TVP and high-level
LetMPCEnv.step repairs unchanged, but rebinds the module-level task identity used
by base.main(), failure artifacts, and the pre-resource backup recency gate to
this structured repair task before entering base.main().

If external backup is still unavailable, the inherited v0i gate must stop before
any solver or plant resource and write the ordinary zero-resource dependency
receipt.  If a verified backup proof appears before launch, this wrapper runs the
same bounded development-only H12/H15/H35 source242 microcontinuation.
"""
from __future__ import annotations

from pathlib import Path

import vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0i_env_tvp_format_repair as v0i

SOURCE = Path(__file__).resolve()
NAME = "vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0j_task_id_binding_repair"
TASK_ID = "S-TC2H8-source242-env-tvp-format-repair-v0j-task-id-binding-repair"


def _apply_task_binding() -> None:
    """Rebind all globals that v0i/base.main() use for structured identity."""
    v0i.SOURCE = SOURCE
    v0i.NAME = NAME
    v0i.TASK_ID = TASK_ID
    v0i.base.NAME = NAME
    v0i.base.TASK_ID = TASK_ID
    v0i.base.__dict__["NAME"] = NAME
    v0i.base.__dict__["TASK_ID"] = TASK_ID
    v0i.base.__dict__["__file__"] = str(SOURCE)


if __name__ == "__main__":
    _apply_task_binding()
    raise SystemExit(v0i.base.main())
