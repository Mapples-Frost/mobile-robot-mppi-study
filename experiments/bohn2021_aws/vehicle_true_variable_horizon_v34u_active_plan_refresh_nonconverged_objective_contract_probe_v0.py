#!/usr/bin/env python3
"""v34u operational wrapper for active-plan A13c-3/T-B3.

This wrapper delegates the scientific design to v34t/v34n, while applying only
engineering repairs required by the current Opus plan:

* use the live PLAN_READY request supplied at launch;
* relay the canonical authorization token A13c3_off_solution_objective_contract;
* preserve the v34s/v34q saved-trajectory goal source and v34o previous_input
  path via the delegated v34t module;
* pre-solve scalarize saved object-TVPs so TTAHMPC obstacle code receives finite
  numeric arrays instead of dict-valued {true/forecast} cells;
* bind the forced low-iteration nlpsol hook at the constructor lookup site used
  by do_mpc.controller (and casadi/do_mpc.optimizer as secondary aliases), then
  fail before get_action/solver calls if construction capture or max_iter=1
  evidence is absent.

No plant rollout, validation64, sealed test, training, refit, threshold change,
cell change, or solver-call cap change is made here.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

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
CANONICAL_TASK_AUTHORIZATION_TOKEN = "A13c3_off_solution_objective_contract"


def _finite_first_float(value: Any, key: str) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception as exc:
        raise v34t.ContractError(
            f"controller TVP channel {key} is not numeric after scalarization: "
            f"{type(value).__name__}: {exc!r}"
        )
    if arr.size < 1:
        raise v34t.ContractError(f"controller TVP channel {key} is empty after scalarization")
    out = float(arr[0])
    if not math.isfinite(out):
        raise v34t.ContractError(f"controller TVP channel {key} first value is not finite: {out!r}")
    return out


def install_tta_hmpc_presolve_obj_guard() -> None:
    """Patch TTAHMPC.get_action so object channels are finite before solve."""
    try:
        v34t.v34n.base.v1d.import_legacy_modules()
    except Exception:
        # Delegated runtime preflight remains authoritative if import truly fails.
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


def _install_bound_nlpsol_patch() -> Dict[str, Any]:
    """Install the forced nlpsol wrapper at the actual do_mpc constructor site."""
    import casadi as ca  # type: ignore
    try:
        import do_mpc.controller as controller_mod  # type: ignore
    except Exception as exc:
        raise v34t.v34n.ContractError("could not import do_mpc.controller for nlpsol binding gate: " + repr(exc))
    try:
        import do_mpc.optimizer as optimizer_mod  # type: ignore
    except Exception:
        optimizer_mod = None  # type: ignore

    if getattr(controller_mod, "_v34u_bound_nlpsol_patched", False):
        return getattr(controller_mod, "_v34u_bound_nlpsol_patch_info", {})

    original = getattr(controller_mod, "nlpsol", None)
    if original is None:
        original = getattr(ca, "nlpsol", None)
    if original is None:
        raise v34t.v34n.ContractError("no nlpsol callable found in do_mpc.controller or casadi")

    def wrapped_nlpsol(name: Any, solver: Any, nlp: Mapping[str, Any], opts: Optional[Mapping[str, Any]] = None) -> Any:
        opts_in = dict(opts or {})
        opts_out = dict(opts_in)
        opts_out.update(v34t.v34n.FORCED_NLPSOL_OPTIONS)
        cap = {
            "arm_id": v34t.v34n._CURRENT_ARM_ID,
            "solver_name": str(name),
            "solver_kind": str(solver),
            "opts_in_keys": sorted(str(k) for k in opts_in.keys()),
            "forced_options": dict(v34t.v34n.FORCED_NLPSOL_OPTIONS),
            "opts_out_subset": {k: opts_out.get(k) for k in sorted(v34t.v34n.FORCED_NLPSOL_OPTIONS.keys())},
            "effective_ipopt_max_iter_equals_1_from_constructor_opts": opts_out.get("ipopt.max_iter") == 1,
            "constructor_lookup_site": "do_mpc.controller.nlpsol plus casadi.nlpsol aliases patched by v34u",
            "nlp_meta": {k: str(getattr(val, "shape", None)) for k, val in (nlp or {}).items() if k in ("x", "f", "g", "p")},
            "nlp_expr": nlp,
        }
        v34t.v34n._CAPTURES.append(cap)
        return original(name, solver, nlp, opts_out)

    setattr(controller_mod, "_v34u_original_nlpsol", original)
    setattr(controller_mod, "nlpsol", wrapped_nlpsol)
    setattr(ca, "nlpsol", wrapped_nlpsol)
    if optimizer_mod is not None:
        setattr(optimizer_mod, "nlpsol", wrapped_nlpsol)
        setattr(optimizer_mod, "_v34u_original_nlpsol", original)
    setattr(controller_mod, "_v34u_bound_nlpsol_patched", True)
    info = {
        "patch_label": "v34u_bound_constructor_lookup_patch",
        "casadi_has_nlpsol": hasattr(ca, "nlpsol"),
        "do_mpc_controller_has_nlpsol": hasattr(controller_mod, "nlpsol"),
        "do_mpc_optimizer_has_nlpsol": bool(optimizer_mod is not None and hasattr(optimizer_mod, "nlpsol")),
        "casadi_nlpsol_is_wrapper": getattr(ca, "nlpsol", None) is wrapped_nlpsol,
        "do_mpc_controller_nlpsol_is_wrapper": getattr(controller_mod, "nlpsol", None) is wrapped_nlpsol,
        "do_mpc_optimizer_nlpsol_is_wrapper": bool(optimizer_mod is not None and getattr(optimizer_mod, "nlpsol", None) is wrapped_nlpsol),
        "forced_options": dict(v34t.v34n.FORCED_NLPSOL_OPTIONS),
    }
    setattr(controller_mod, "_v34u_bound_nlpsol_patch_info", info)
    for label in ["casadi.nlpsol", "do_mpc.controller.nlpsol"] + (["do_mpc.optimizer.nlpsol"] if optimizer_mod is not None else []):
        if label not in v34t.v34n._PATCHED_MODULES:
            v34t.v34n._PATCHED_MODULES.append(label)
    return info


def _install_constructor_gate() -> None:
    """Patch v34n after v34t identity setup to enforce T-B3-A before solves."""
    v34n = v34t.v34n
    if getattr(v34n, "_v34u_constructor_gate_installed", False):
        return
    original_create_env = v34n.base.create_env
    original_solve_cell = v34n.solve_cell
    original_analyze = v34n.analyze

    def patch_nlpsol_constructor_bound() -> None:
        _install_bound_nlpsol_patch()

    def create_env_with_constructor_gate(h: int, terminal: Any) -> Any:
        before = len(v34n._CAPTURES)
        env = original_create_env(h, terminal)
        new_caps = v34n._CAPTURES[before:]
        import casadi as ca  # type: ignore
        import do_mpc.controller as controller_mod  # type: ignore
        try:
            import do_mpc.optimizer as optimizer_mod  # type: ignore
        except Exception:
            optimizer_mod = None  # type: ignore
        patch_info = getattr(controller_mod, "_v34u_bound_nlpsol_patch_info", {})
        mpc = env.control_system.controller.mpc
        opts_obj = getattr(mpc, "nlpsol_opts", None)
        try:
            opts_dict = dict(opts_obj or {})
        except Exception:
            opts_dict = {"unserializable_repr": repr(opts_obj)}
        gate = {
            "T_B3_A_constructor_binding_gate": True,
            "captures_added_during_create_env": len(new_caps),
            "casadi_nlpsol_is_controller_global": getattr(ca, "nlpsol", None) is getattr(controller_mod, "nlpsol", None),
            "controller_nlpsol_is_wrapper": bool(patch_info.get("do_mpc_controller_nlpsol_is_wrapper")),
            "casadi_nlpsol_is_wrapper": bool(patch_info.get("casadi_nlpsol_is_wrapper")),
            "optimizer_nlpsol_is_wrapper": bool(patch_info.get("do_mpc_optimizer_nlpsol_is_wrapper")),
            "constructor_capture_has_effective_max_iter_1": any(c.get("effective_ipopt_max_iter_equals_1_from_constructor_opts") is True for c in new_caps),
            "mpc_nlpsol_opts_subset": {k: opts_dict.get(k) for k in sorted(v34n.FORCED_NLPSOL_OPTIONS.keys())},
            "mpc_nlpsol_opts_has_ipopt_max_iter_1": opts_dict.get("ipopt.max_iter") == 1,
            "patch_info": patch_info,
        }
        setattr(mpc, "_v34u_constructor_binding_gate", gate)
        if len(new_caps) < 1:
            raise v34n.ContractError("T-B3-A constructor binding gate failed: captured_nlpsol_count_during_construction=0; stopping before get_action/solver call")
        if gate["constructor_capture_has_effective_max_iter_1"] is not True:
            raise v34n.ContractError("T-B3-A constructor binding gate failed: constructor capture lacks effective ipopt.max_iter=1")
        return env

    def solve_cell_with_gate_record(row: Mapping[str, Any], context: Mapping[str, Any], terminals: Mapping[int, Any], idx: int) -> Dict[str, Any]:
        arm = original_solve_cell(row, context, terminals, idx)
        # Add an explicit top-level summary for easy inspection. The full capture
        # metadata remains in captured_nlpsol_meta and direct_nlp_eval.
        captures = [c for c in v34n._CAPTURES if c.get("arm_id") == arm.get("arm_id")]
        arm["T_B3_A_constructor_binding"] = {
            "captured_nlpsol_count_during_construction": arm.get("captured_nlpsol_count_during_construction"),
            "captures_for_arm": len(captures),
            "capture_effective_max_iter_1_count": sum(1 for c in captures if c.get("effective_ipopt_max_iter_equals_1_from_constructor_opts") is True),
            "constructor_lookup_site": "do_mpc.controller.nlpsol/casadi.nlpsol patched before create_env",
            "passed": bool((arm.get("captured_nlpsol_count_during_construction") or 0) >= 1 and any(c.get("effective_ipopt_max_iter_equals_1_from_constructor_opts") is True for c in captures)),
        }
        if arm.get("T_B3_A_constructor_binding", {}).get("passed") is not True and int(arm.get("solve_calls", 0) or 0) == 0:
            raise v34n.ContractError("T-B3-A constructor binding gate failed before any solver call; raw arm evidence: " + repr(arm.get("T_B3_A_constructor_binding")))
        return arm

    def analyze_with_constructor_gate(arms: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        analysis = original_analyze(arms)
        h = analysis.get("headline") or {}
        gates = [((a.get("T_B3_A_constructor_binding") or {}).get("passed") is True) for a in arms]
        h["T_B3_A_constructor_binding_cells_passed"] = int(sum(1 for x in gates if x))
        h["T_B3_A_constructor_binding_gate_all_cells"] = bool(len(gates) == len(v34n.CELLS) and all(gates))
        h["T_B3_A_effective_max_iter_1_capture_count"] = int(sum(1 for c in v34n._CAPTURES if c.get("effective_ipopt_max_iter_equals_1_from_constructor_opts") is True))
        h["G2_pass_before_T_B3_A_constructor_gate"] = bool(h.get("G2_pass"))
        h["G2_pass"] = bool(h.get("G2_pass") and h.get("T_B3_A_constructor_binding_gate_all_cells"))
        analysis["headline"] = h
        return analysis

    v34n.patch_nlpsol_constructor = patch_nlpsol_constructor_bound  # type: ignore[assignment]
    v34n.base.create_env = create_env_with_constructor_gate  # type: ignore[assignment]
    v34n.solve_cell = solve_cell_with_gate_record  # type: ignore[assignment]
    v34n.analyze = analyze_with_constructor_gate  # type: ignore[assignment]
    v34n._v34u_constructor_gate_installed = True


def _wrap_v34t_patch_module_identity() -> None:
    if getattr(v34t, "_v34u_patch_module_identity_wrapped", False):
        return
    original = v34t.patch_module_identity_and_contracts

    def patched_patch_module_identity_and_contracts() -> None:
        original()
        _install_constructor_gate()

    v34t.patch_module_identity_and_contracts = patched_patch_module_identity_and_contracts  # type: ignore[assignment]
    v34t._v34u_patch_module_identity_wrapped = True


def run(argv: Optional[Sequence[str]] = None) -> int:
    v34t.NAME = NAME
    v34t.TASK_AUTHORIZATION_TOKEN = CANONICAL_TASK_AUTHORIZATION_TOKEN
    install_tta_hmpc_presolve_obj_guard()
    _wrap_v34t_patch_module_identity()
    return v34t.run(argv)


if __name__ == "__main__":
    raise SystemExit(run())
