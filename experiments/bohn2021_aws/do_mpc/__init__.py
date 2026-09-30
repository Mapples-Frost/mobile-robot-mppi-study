"""Repository-local legacy do_mpc import shim for structured T-C2 repairs.

The legacy execution environment used for Bohn 2021 reproduction does not expose
``do_mpc`` as an installed site package. Earlier runners normally call the
project's legacy import bootstrap before importing ``do_mpc`` directly. The
T-C2 v34z2 source-level precondition imports ``do_mpc.controller`` earlier, so
run 20260930T143813_cd837da9 failed before any scientific resource use with
``ModuleNotFoundError: No module named 'do_mpc'``.

This shim is intentionally minimal: when ``experiments/bohn2021_aws`` is on
``sys.path`` it redirects the package namespace to the vendored historical
``do-mpc-horizon`` source tree retained under research_artifacts. It does not
change solver options, model equations, costs, horizons, split access or any
scientific acceptance threshold.
"""
from __future__ import annotations

from pathlib import Path

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
