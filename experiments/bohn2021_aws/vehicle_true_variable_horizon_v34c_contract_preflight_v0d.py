#!/usr/bin/env python3
"""v34c/v0d zero-solve contract preflight wrapper: typed-struct index repair.

This is a narrow operational repair for the Astra 20260930T084456Z Task1
zero-solve preflight.  The previous v0c attempt failed before any controlled
solve intercept with:

    typed initialization failed: attempted=267 succeeded=0 failed=267

The failure is a CasADi/do-mpc structured-data indexing mismatch: labels such as
``[_x,0,0,0,theta,0]`` were parsed as strings and then written back verbatim,
whereas the structure expects integer repeat indices and/or the scalar trailing
index to be omitted.  This wrapper keeps the frozen contexts, terminals,
horizons, initialization rules, budgets and no-validation/no-test access exactly
unchanged, but patches the imported preflight with:

* robust label-to-typed-index conversion (integer repeat indices, optional scalar
  suffix removal, flat-master fallback);
* scaled write/readback verification against physical target values using the
  actual opt_x_scaling structure when available;
* opt_x_num_unscaled synchronization after the typed initial guess so the lterm
  availability check sees the same prepared trajectory;
* saved-case object-noise seed restoration from reference.ns before
  TTAHMPC.get_action constructs forecast perturbations.

It still performs 0 lower-level NLP solves, 0 plant steps, 0 training/refit, no
validation64 and no sealed-test access.  If any required primitive remains
unavailable, it fails closed before the 24-call solver probe.
"""
from __future__ import annotations

import copy
import datetime as dt
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_v34c_contract_preflight_v0c as v0c  # noqa:E402

c = v0c.c


def _token_to_index(token: Any) -> Any:
    """Convert CasADi label repeat indices to ints while preserving names."""
    if isinstance(token, int):
        return token
    s = str(token).strip()
    if s.lstrip("-").isdigit():
        try:
            return int(s)
        except Exception:
            return token
    return s


def _candidate_indices(parts: Sequence[Any], var: Optional[str] = None) -> List[Tuple[Any, ...]]:
    """Candidate DMStruct indices for a parsed label.

    do-mpc/CasADi labels often include a scalar component suffix (e.g.
    ``theta,0``).  The full typed path may or may not accept this suffix.  Try
    the complete converted label first, then the no-trailing-scalar form, then
    the path up to the semantic variable name.
    """
    conv = [_token_to_index(p) for p in parts]
    out: List[Tuple[Any, ...]] = []

    def add(seq: Sequence[Any]) -> None:
        tup = tuple(seq)
        if tup not in out:
            out.append(tup)

    add(conv)
    if conv and isinstance(conv[-1], int):
        add(conv[:-1])
    if var is not None and var in conv:
        vi = conv.index(var)
        add(conv[: vi + 1])
        if vi + 1 < len(conv) and isinstance(conv[-1], int):
            add(conv[: vi + 1] + conv[-1:])
    return out


def _safe_get_any(obj: Any, parts: Sequence[Any], var: Optional[str] = None, flat_index: Optional[int] = None) -> Any:
    for idx in _candidate_indices(parts, var=var):
        try:
            return obj[idx]
        except Exception:
            try:
                return obj.__getitem__(idx)
            except Exception:
                pass
    if flat_index is not None:
        try:
            return obj.cat[int(flat_index)]
        except Exception:
            try:
                return obj.master[int(flat_index)]
            except Exception:
                return None
    return None


def _safe_set_any(obj: Any, parts: Sequence[Any], value: float, var: Optional[str] = None, flat_index: Optional[int] = None) -> Tuple[bool, str]:
    errors: List[str] = []
    for idx in _candidate_indices(parts, var=var):
        try:
            obj[idx] = value
            return True, repr(idx)
        except Exception as exc:
            if len(errors) < 3:
                errors.append(repr(exc))
        try:
            obj.__setitem__(idx, value)
            return True, repr(idx)
        except Exception as exc:
            if len(errors) < 3:
                errors.append(repr(exc))
    if flat_index is not None:
        try:
            obj.master[int(flat_index)] = value
            return True, f"master[{int(flat_index)}]"
        except Exception as exc:
            errors.append("master:" + repr(exc))
    return False, "; ".join(errors[-3:])


def _safe_scalar(value: Any) -> Optional[float]:
    try:
        a = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        try:
            if hasattr(value, "full"):
                a = np.asarray(value.full(), dtype=float).reshape(-1)
            elif hasattr(value, "cat"):
                a = np.asarray(value.cat, dtype=float).reshape(-1)
            else:
                return None
        except Exception:
            return None
    if a.size != 1:
        return None
    out = float(a[0])
    return out if math.isfinite(out) else None


def _scale_for_label(mpc: Any, parts: Sequence[Any], var: Optional[str], flat_index: int) -> float:
    scale_obj = getattr(mpc, "opt_x_scaling", None)
    if scale_obj is not None:
        val = _safe_scalar(_safe_get_any(scale_obj, parts, var=var, flat_index=flat_index))
        if val is not None and math.isfinite(val) and abs(val) > 0.0:
            return float(val)
    return 1.0


def _sync_unscaled_from_scaled(mpc: Any) -> Optional[str]:
    try:
        mpc.opt_x_num_unscaled.master = mpc.opt_x_num.cat * mpc.opt_x_scaling.cat
        return "opt_x_num_unscaled.master = opt_x_num.cat * opt_x_scaling.cat"
    except Exception as exc:
        return "sync_failed:" + repr(exc)


def typed_set_initial_guess_v0d(mpc: Any, env: Any, context_meta: Mapping[str, Any], init: str, h: int) -> Dict[str, Any]:
    """Strict typed initialization with CasADi index conversion and readback."""
    ctrl = env.control_system.controller
    state_names = list(getattr(ctrl, "state_names", ["theta", "x", "y"]))
    input_names = list(getattr(ctrl, "input_names", ["u_omega", "u_s"]))
    state = c.m.state_clean(context_meta["state_after_direct_setup"])
    if init == "canonical":
        pred_states, pred_controls = c.m.zero_guess(state, h)
        rule = "repeat_initial_state_zero_control_zero_algebraic"
    elif init == "goal_facing":
        dt_s = c.m.as_float(getattr(mpc, "t_step", None), 0.1)
        pred_states, pred_controls = c.m.predicted_unicycle(state, float(context_meta["goal_x"]), float(context_meta["goal_y"]), h, dt_s)
        rule = "deterministic_unicycle_goal_facing_clip_omega_pm4_speed_0_5"
    else:
        raise c.ContractError("unknown init " + init)

    obj = getattr(mpc, "opt_x_num", None)
    if obj is None:
        raise c.ContractError("mpc.opt_x_num missing for typed initialization")

    attempted = succeeded = failed = 0
    examples: List[Dict[str, Any]] = []
    failure_examples: List[Dict[str, Any]] = []
    labels = c.get_struct_labels(obj)
    for flat_i, label in enumerate(labels):
        parts = c.parse_label(label)
        if not parts:
            continue
        top = parts[0]
        val: Optional[float] = None
        var: Optional[str] = None
        if top == "_x":
            var = c.label_var(parts, state_names)
            if var is None:
                continue
            k = c.first_int_after_top(parts)
            if 0 <= k < len(pred_states):
                val = float(pred_states[k].get(var, state.get(var, 0.0)))
        elif top == "_u":
            var = c.label_var(parts, input_names)
            if var is None:
                continue
            k = c.first_int_after_top(parts)
            if 0 <= k < len(pred_controls):
                val = float(pred_controls[k].get(var, 0.0))
        elif top in ("_z", "_eps"):
            val = 0.0
        if val is None:
            continue

        attempted += 1
        scale = _scale_for_label(mpc, parts, var, flat_i)
        target_scaled = float(val) / float(scale)
        ok, write_path = _safe_set_any(obj, parts, target_scaled, var=var, flat_index=flat_i)
        rb_scaled = _safe_scalar(_safe_get_any(obj, parts, var=var, flat_index=flat_i)) if ok else None
        rb_unscaled = None if rb_scaled is None else rb_scaled * scale
        good = bool(ok and rb_unscaled is not None and abs(rb_unscaled - float(val)) <= 1e-8 * max(1.0, abs(float(val))))
        if good:
            succeeded += 1
        else:
            failed += 1
            if len(failure_examples) < 12:
                failure_examples.append({
                    "label": label,
                    "parts": parts,
                    "var": var,
                    "target_physical": val,
                    "scale": scale,
                    "target_scaled": target_scaled,
                    "write_ok": ok,
                    "write_path_or_error": write_path,
                    "readback_scaled": rb_scaled,
                    "readback_unscaled": rb_unscaled,
                })
        if len(examples) < 10:
            examples.append({
                "label": label,
                "parts": parts,
                "var": var,
                "target_physical": val,
                "scale": scale,
                "target_scaled": target_scaled,
                "write_ok": ok,
                "write_path": write_path,
                "readback_scaled": rb_scaled,
                "readback_unscaled": rb_unscaled,
            })

    if attempted <= 0 or succeeded <= 0 or failed > 0:
        raise c.ContractError(
            "typed initialization failed after index repair: "
            f"attempted={attempted} succeeded={succeeded} failed={failed} "
            f"examples={failure_examples[:4]}"
        )

    unscaled_sync = _sync_unscaled_from_scaled(mpc)
    try:
        mpc.lam_g_num = 0 * mpc.lam_g_num
    except Exception:
        pass

    return {
        "initialization": init,
        "rule": rule,
        "state_names": state_names,
        "input_names": input_names,
        "assignments_attempted": attempted,
        "assignments_succeeded": succeeded,
        "assignments_failed": failed,
        "label_count": len(labels),
        "assignment_examples": examples,
        "failure_examples": failure_examples,
        "unscaled_sync": unscaled_sync,
        "initial_primal_hash": c.arr_hash(getattr(mpc, "opt_x_num", [])),
        "initial_primal_unscaled_hash": c.arr_hash(getattr(mpc, "opt_x_num_unscaled", [])) if hasattr(mpc, "opt_x_num_unscaled") else None,
        "initial_dual_hash_after_zero": c.arr_hash(getattr(mpc, "lam_g_num", [])) if hasattr(mpc, "lam_g_num") else None,
        "first_three_pred_states": pred_states[:3],
        "first_three_pred_controls": pred_controls[:3],
    }


_ORIGINAL_SETUP_CONTEXT_DIRECT = c.setup_context_direct


def setup_context_direct_v0d(env: Any, context: Mapping[str, Any], h: int) -> Dict[str, Any]:
    """Restore saved object-noise seed for forecast construction, then direct setup."""
    meta = _ORIGINAL_SETUP_CONTEXT_DIRECT(env, context, h)
    ctrl = env.control_system.controller
    ref = (context.get("case_snapshot") or {}).get("reference") if isinstance(context.get("case_snapshot"), Mapping) else None
    ns = copy.deepcopy(ref.get("ns")) if isinstance(ref, Mapping) and "ns" in ref else None
    n_objects = int(getattr(ctrl, "n_objects", 0))
    if n_objects > 0:
        if ns is None:
            raise c.ContractError("saved reference.ns object-noise seed missing for TTAHMPC forecast reconstruction")
        try:
            if len(ns) != n_objects:
                raise c.ContractError(f"reference.ns object count {len(ns)} != controller.n_objects {n_objects}")
            for row in ns:
                if len(row) != 3:
                    raise c.ContractError("reference.ns entries must have x/y/r components")
        except TypeError as exc:
            raise c.ContractError("reference.ns is not a nested sequence: " + repr(exc))
        ctrl.object_noise_seed = ns
        # Force TTAHMPC.get_action to rebuild obj_data from the provided shifted
        # TVPs under the restored noise seed, just as reset(reference={'ns': ...})
        # would do without redrawing random perturbations.
        ctrl.obj_data = None
        meta["object_noise_seed_source"] = "case_snapshot.reference.ns"
        meta["object_noise_seed_shape"] = [len(ns), len(ns[0]) if ns else 0]
        meta["object_noise_seed_hash"] = c.canonical_hash(ns)
    else:
        meta["object_noise_seed_source"] = "not_needed_n_objects_0"
    return meta


def patch_runtime() -> None:
    # Reuse v0c's explicit goal reconstruction patch and current Astra gate.
    v0c.patch_runtime()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = "vehicle_true_variable_horizon_v34c_contract_preflight_v0d"
    c.NAME = name
    c.STAMP = stamp
    c.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{name}_{stamp}"
    c.ARRAY_DIR = c.RUN_DIR / "arrays"
    c.BACKUP_REQ = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34C_CONTRACT_PREFLIGHT_V0D_{stamp}.json"
    c.STATE = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_v34c_contract_preflight_v0d.md"
    c.MARKER = f"vehicle-v34c-contract-preflight-v0d-{stamp}"
    c.typed_set_initial_guess = typed_set_initial_guess_v0d
    c.setup_context_direct = setup_context_direct_v0d


def main() -> int:
    patch_runtime()
    return int(c.run())


if __name__ == "__main__":
    raise SystemExit(main())
