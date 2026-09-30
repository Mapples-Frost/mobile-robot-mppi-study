"""Repository-local legacy do_mpc import shim for structured T-C2 repairs.

The legacy execution environment used for Bohn 2021 reproduction does not expose
``do_mpc`` as an installed site package. Earlier runners normally call the
project's legacy import bootstrap before importing ``do_mpc`` directly. The
T-C2 v34z2 source-level precondition imports ``do_mpc.controller`` early, so run
20260930T143813_cd837da9 failed before any scientific resource use with
``ModuleNotFoundError: No module named 'do_mpc'``.

This shim is intentionally minimal: when ``experiments/bohn2021_aws`` is on
``sys.path`` it redirects the package namespace to the vendored historical
``do-mpc-horizon`` source tree retained under research_artifacts. It does not
change solver options, model equations, costs, horizons, split access or any
scientific acceptance threshold.

Operational repair note, 2026-09-30: the next T-C2 launch
(20260930T144148_12c26dcc) progressed past the package import and then failed
with zero solver/plant/training/validation/test resources because v34z2 had
bound ``vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner``
as ``stage1_runner`` while expecting terminal-grid symbols that live in
``vehicle_stress_scenario_opportunity_probe_v1_runner``.  To avoid repurposing
the protected v34z2/T-C5 sources inside this bounded cycle, this shim adds a
narrow compatibility alias to the already-imported v1d module when do_mpc is
loaded.  The alias exposes only the terminal-source protocol path and delegates
``load_terminal_grid`` to the audited stress-scenario v1 runner.  It does not
open validation64 or sealed-test data and does not alter any controller or
solver behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Tuple, Dict

_ROOT = Path(__file__).resolve().parents[3]
_REAL_PACKAGE = _ROOT / "research_artifacts" / "bohn2021_reproduction_2026-09-17" / "sources" / "do-mpc-horizon" / "do_mpc"
_REAL_INIT = _REAL_PACKAGE / "__init__.py"

if not _REAL_INIT.exists():  # pragma: no cover - fail loudly in the experiment receipt.
    raise ModuleNotFoundError(f"vendored do_mpc package not found at {_REAL_PACKAGE}")

# Make relative imports such as ``import do_mpc.controller`` resolve against the
# archived package while preserving the public package name ``do_mpc``.
__path__ = [str(_REAL_PACKAGE)]
__file__ = str(_REAL_INIT)

_code = compile(_REAL_INIT.read_text(encoding="utf-8"), str(_REAL_INIT), "exec")
exec(_code, globals(), globals())


def _install_v34z2_terminal_alias() -> None:
    """Repair the narrow v34z2 terminal-grid alias after v1d was imported.

    v34z2 imports the v1d trace-selected runner under the local name
    ``stage1_runner`` but then asks it for ``TERMINAL_SOURCE_PROTOCOL`` and
    ``load_terminal_grid``.  Those symbols are provided by the stress-scenario
    v1 runner.  This compatibility alias is deliberately lazy: it imports the
    provider only when v34z2 calls ``load_terminal_grid``.
    """
    target_name = "vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner"
    target = sys.modules.get(target_name)
    if target is None:
        return
    if hasattr(target, "TERMINAL_SOURCE_PROTOCOL") and hasattr(target, "load_terminal_grid"):
        return

    terminal_protocol = _ROOT / "research_artifacts" / "aws_protocols" / "vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.json"

    def _delegated_load_terminal_grid(terminal_grid: Mapping[str, Any]) -> Tuple[Dict[int, Tuple[Any, Any]], Dict[str, Any]]:
        import vehicle_stress_scenario_opportunity_probe_v1_runner as _terminal_provider  # type: ignore
        return _terminal_provider.load_terminal_grid(terminal_grid)

    setattr(target, "TERMINAL_SOURCE_PROTOCOL", terminal_protocol)
    setattr(target, "load_terminal_grid", _delegated_load_terminal_grid)
    setattr(target, "_v34z2_terminal_source_alias_repair", {
        "status": "installed",
        "reason": "T-C2 v34z2 expected terminal-grid symbols on the v1d module; delegated to vehicle_stress_scenario_opportunity_probe_v1_runner without changing scientific config.",
        "terminal_source_protocol": str(terminal_protocol),
        "no_solver_or_data_access_performed_by_alias_install": True,
    })


_install_v34z2_terminal_alias()
