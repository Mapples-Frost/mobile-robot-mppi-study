#!/usr/bin/env python3
"""v34d thin wrapper: run the fixed 24-call objective-vs-basin probe using
v34c/v0e preflight-verified contract primitives.

This is a narrow implementation repair, not a scientific-design change.  The
frozen Task2 design remains exactly 2 opened development contexts x 2 horizons x
3 terminal contracts x 2 deterministic initializations = 24 low-level solver
attempts, with 0 plant steps, no automatic retry/extra guesses, no selector
search/refit/training, no validation64 and no sealed/final test access.

Why a wrapper is required: the original v34 probe still used the older unsafe
direct context reconstruction / typed initial guess / raw saved TVP path.  The
Astra-approved v34c/v0e preflight hard-passed after verifying the repaired
contract: explicit goal reconstruction, singleton scalarization, typed CasADi
initialization/readback, saved object-noise seed restoration, finite numeric TVP
conversion for TTAHMPC.get_action, four-argument vf_fun availability, lterm
availability, and no-solve invariants.  This wrapper imports v34 unchanged and
monkey-patches only those operational primitives plus vf4 cross-evaluation so the
24-call diagnostic spends its budget on the intended objective-vs-basin question.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0 as base
import vehicle_true_variable_horizon_v34c_contract_preflight_v0e as contract
import vehicle_true_variable_horizon_v34c_contract_preflight_v0d as contract_d

ROOT = Path(__file__).resolve().parents[2]
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
NAME = "vehicle_true_variable_horizon_v34d_objective_basin_solver_probe_v0"
CURRENT_REQUEST = "execution-result:20260930T084409_7dd79362"
CURRENT_REPORT = ROOT / "docs/bohn2021_takeover/astra_reviews/20260930T084456Z.md"
CURRENT_SHA = "9888c40d0cb52640ee18cd548e001698236aab2f1414e615f361c04abfa40ba0"
V0E_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34c_contract_preflight_v0e_20260930T093751Z/completed.json"
V0E_BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260930T093751Z_from_user_context_before_v34_objective_basin_probe.json"


def _finite(value: Any) -> Optional[float]:
    try:
        if hasattr(value, "full"):
            a = np.asarray(value.full(), dtype=float).reshape(-1)
        elif hasattr(value, "cat"):
            a = np.asarray(value.cat, dtype=float).reshape(-1)
        else:
            a = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if a.size != 1:
        return None
    out = float(a[0])
    return out if math.isfinite(out) else None


def _as_vec(value: Any) -> np.ndarray:
    try:
        if hasattr(value, "full"):
            return np.asarray(value.full(), dtype=float).reshape((-1, 1))
        if hasattr(value, "cat"):
            return np.asarray(value.cat, dtype=float).reshape((-1, 1))
        return np.asarray(value, dtype=float).reshape((-1, 1))
    except Exception:
        return np.asarray([], dtype=float).reshape((0, 1))


def _call_vf4(mpc: Any, x_value: Any) -> Tuple[Optional[float], Dict[str, Any]]:
    """Evaluate current MPC terminal value with the verified 4-argument API."""
    meta: Dict[str, Any] = {"api": "vf_fun(x, p_excluding_n_horizon, weights_num, biases_num)"}
    try:
        p_keep, p_names = contract.c.p_excluding_n_horizon(mpc)
        p_arr = np.asarray(p_keep, dtype=float).reshape((-1, 1))
        x_arr = _as_vec(x_value)
        meta.update({
            "ok": False,
            "vf_fun_n_in": int(mpc.vf_fun.n_in()) if hasattr(mpc.vf_fun, "n_in") else None,
            "x_size": int(x_arr.size),
            "p_size": int(p_arr.size),
            "p_names_excluding_n_horizon": list(p_names),
        })
        val = mpc.vf_fun(x_arr, p_arr, mpc.vf.weights_num, mpc.vf.biases_num)
        out = _finite(val)
        if out is None:
            meta["error"] = "vf_fun returned non-scalar/non-finite value"
            return None, meta
        meta["ok"] = True
        return float(out), meta
    except Exception as exc:
        meta.update({"ok": False, "error": repr(exc)})
        return None, meta


def configure_context_no_reset_v34d(env: Any, context_row: Mapping[str, Any], h: int) -> Dict[str, Any]:
    """Adapter from v34c/v0e direct setup metadata to v34 solver-probe fields."""
    meta = contract.setup_context_direct_v0e(env, context_row, int(h))
    if "state_after_config" not in meta and "state_after_direct_setup" in meta:
        meta["state_after_config"] = meta["state_after_direct_setup"]
    if "state_distance_after_config" not in meta and "state_distance_after_direct_setup" in meta:
        meta["state_distance_after_config"] = meta["state_distance_after_direct_setup"]
    if "tvp_hash" not in meta and "shifted_tvp_numeric_hash" in meta:
        meta["tvp_hash"] = meta["shifted_tvp_numeric_hash"]
    meta["v34d_contract_adapter"] = "v34c/v0e explicit-goal + object-seed + typed-init-compatible numeric TVP path"
    return meta


def set_initial_guess_v34d(mpc: Any, env: Any, context_meta: Mapping[str, Any], init: str, h: int) -> Dict[str, Any]:
    return contract_d.typed_set_initial_guess_v0d(mpc, env, context_meta, init, int(h))


def verify_gates_with_v0e(args: Any) -> Dict[str, Any]:
    gates = ORIGINAL_VERIFY(args)
    v0e = base.read_json(V0E_DONE)
    if v0e.get("hard_pass") is not True:
        raise base.ContractError("v34c/v0e contract preflight did not hard-pass")
    b = v0e.get("budget_actual") or {}
    if int(b.get("lower_level_solver_calls", 0)) != 0 or int(b.get("plant_steps", 0)) != 0 or int(b.get("training_or_refit", 0)) != 0:
        raise base.ContractError("v34c/v0e contract preflight budget was not zero-solve/zero-plant/zero-training")
    proof = base.read_json(V0E_BACKUP_PROOF)
    if proof.get("status") != "verified" or proof.get("backup_verified") is not True:
        raise base.ContractError("missing verified supervisor backup proof after v34c/v0e preflight")
    pkgs = proof.get("packages_this_run") or []
    if args.backup_commit != proof.get("commit"):
        raise base.ContractError("backup commit arg does not match v0e backup proof")
    if not any((p or {}).get("sha256") == args.backup_package_sha256 for p in pkgs):
        raise base.ContractError("backup package sha arg does not match v0e backup proof")
    gates["v34c_v0e_contract_preflight_gate"] = {
        "completed": base.rel(V0E_DONE),
        "hard_pass": True,
        "budget_actual": b,
        "backup_proof": base.rel(V0E_BACKUP_PROOF),
        "backup_proof_status": proof.get("status"),
        "backup_proof_time": proof.get("time"),
        "backup_proof_source": proof.get("source"),
        "note": "external-backup verification is taken from the supervisor-context proof available to this API executor",
    }
    return gates


def reconstruct_stage_and_terminal_v34d(mpc: Any, h: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": False,
        "stage_lterm_total": None,
        "terminal_value_current_vf": None,
        "gamma": None,
        "gamma_source": None,
        "gamma_pow_H_terminal": None,
        "solver_objective": _finite(getattr(mpc, "opt_f_num", None)),
        "reconstructed_total_current_objective": None,
        "relative_error_vs_solver": None,
        "failure_reasons": [],
        "component_limits": {
            "slack_separate_component_available": False,
            "input_regularization_separate_component_available": False,
            "stage_lterm_may_include_slack_or_other_terms": True,
        },
        "vf4_contract": True,
    }
    gamma, gsrc = base.find_discount(mpc)
    out["gamma"] = gamma
    out["gamma_source"] = gsrc
    try:
        p0 = mpc.opt_p_num["_p", 0]
    except Exception as exc:
        p0 = None
        out["failure_reasons"].append("p0 unavailable: " + repr(exc))
    stage_vals = []
    if p0 is not None and hasattr(mpc, "lterm_fun"):
        for k in range(int(h)):
            try:
                xk = mpc.opt_p_num["_x0"] if k == 0 else mpc.opt_x_num_unscaled["_x", k, 0, 0]
                uk = mpc.opt_x_num_unscaled["_u", k, 0]
                try:
                    zk = mpc.opt_x_num_unscaled["_z", k + 1, 0, -1]
                except Exception:
                    zk = mpc.opt_x_num_unscaled["_z", k, 0, -1]
                tvpk = mpc.opt_p_num["_tvp", k]
                val = mpc.lterm_fun(xk, uk, zk, tvpk, p0)
                f = _finite(val)
                if f is None:
                    out["failure_reasons"].append(f"lterm k={k} nonfinite/non-scalar")
                    break
                stage_vals.append((float(gamma) ** k) * float(f))
            except Exception as exc:
                out["failure_reasons"].append(f"lterm k={k} exception: {repr(exc)}")
                break
    else:
        out["failure_reasons"].append("lterm_fun or p0 unavailable")
    if len(stage_vals) == int(h):
        out["stage_lterm_total"] = float(math.fsum(stage_vals))
    try:
        xh = mpc.opt_x_num_unscaled["_x", int(h), 0, 0]
        out["terminal_x_vector"] = _as_vec(xh).reshape(-1).tolist()
        p_keep, p_names = contract.c.p_excluding_n_horizon(mpc)
        out["p_excluding_n_horizon_values"] = np.asarray(p_keep, dtype=float).reshape(-1).tolist()
        out["p_names_excluding_n_horizon"] = list(p_names)
        vf_val, vf_meta = _call_vf4(mpc, xh)
        out["terminal_vf_call_meta"] = vf_meta
        if vf_val is not None:
            out["terminal_value_current_vf"] = float(vf_val)
            out["gamma_pow_H_terminal"] = float((float(gamma) ** int(h)) * float(vf_val))
    except Exception as exc:
        out["failure_reasons"].append("vf4 current exception: " + repr(exc))
    if out["stage_lterm_total"] is not None and out["gamma_pow_H_terminal"] is not None:
        total = float(out["stage_lterm_total"] + out["gamma_pow_H_terminal"])
        out["reconstructed_total_current_objective"] = total
        if out["solver_objective"] is not None:
            out["relative_error_vs_solver"] = abs(total - float(out["solver_objective"])) / max(1.0, abs(float(out["solver_objective"])))
        out["available"] = True
    return out


def build_vf_evaluator_cache_v34d(terminals: Mapping[int, Tuple[Any, Any]]) -> Dict[Tuple[int, str], Any]:
    cache: Dict[Tuple[int, str], Any] = {}
    for h in (12, 15, 35):
        for mode in base.TERMINAL_MODES:
            if mode == "zero":
                cache[(h, mode)] = {"mode": "zero", "horizon": h}
            else:
                terminal, _, note = base.terminal_for_mode(h, mode, terminals)
                env = base.create_env(h, terminal)
                mpc = env.control_system.controller.mpc
                cache[(h, mode)] = {
                    "mode": mode,
                    "horizon": h,
                    "terminal_note": note,
                    "vf_fun": mpc.vf_fun,
                    "weights_num": mpc.vf.weights_num,
                    "biases_num": mpc.vf.biases_num,
                    "vf_fun_n_in": int(mpc.vf_fun.n_in()) if hasattr(mpc.vf_fun, "n_in") else None,
                }
    return cache


def _cross_vf4(entry: Mapping[str, Any], x_vec: Sequence[float], p_vec: Sequence[float]) -> Tuple[Optional[float], Dict[str, Any]]:
    if entry.get("mode") == "zero":
        return 0.0, {"ok": True, "source": "zero_terminal", "api": "constant_zero"}
    try:
        x_arr = np.asarray(x_vec, dtype=float).reshape((-1, 1))
        p_arr = np.asarray(p_vec, dtype=float).reshape((-1, 1))
        val = entry["vf_fun"](x_arr, p_arr, entry["weights_num"], entry["biases_num"])
        out = _finite(val)
        if out is None:
            return None, {"ok": False, "error": "non-scalar/non-finite", "api": "vf4", "vf_fun_n_in": entry.get("vf_fun_n_in")}
        return float(out), {"ok": True, "api": "vf4", "vf_fun_n_in": entry.get("vf_fun_n_in"), "x_size": int(x_arr.size), "p_size": int(p_arr.size)}
    except Exception as exc:
        return None, {"ok": False, "api": "vf4", "error": repr(exc), "vf_fun_n_in": entry.get("vf_fun_n_in")}


def cross_evaluate_v34d(arms: Sequence[Mapping[str, Any]], vf_cache: Mapping[Tuple[int, str], Any]) -> list[Dict[str, Any]]:
    rows = []
    for arm in arms:
        h = int(arm["horizon"])
        recon = arm.get("objective_reconstruction_current") or {}
        stage_total = recon.get("stage_lterm_total")
        gamma = float(recon.get("gamma") if recon.get("gamma") is not None else 1.0)
        x_vec = recon.get("terminal_x_vector") or (arm.get("terminal_state") or {}).get("raw_vector") or []
        p_vec = recon.get("p_excluding_n_horizon_values") or []
        row: Dict[str, Any] = {
            "arm_id": arm["arm_id"],
            "context_id": arm["context_id"],
            "horizon": h,
            "candidate_terminal_mode": arm["terminal_mode"],
            "candidate_initialization": arm["initialization"],
            "accepted": arm["accepted"],
            "stage_lterm_total": stage_total,
            "p_names_excluding_n_horizon": recon.get("p_names_excluding_n_horizon"),
            "objectives": {},
        }
        for mode in base.TERMINAL_MODES:
            vf_val, meta = _cross_vf4(vf_cache.get((h, mode), {}), x_vec, p_vec)
            gamma_terminal = None if vf_val is None else float((gamma ** h) * float(vf_val))
            row["objectives"][mode] = {
                "vf_value": vf_val,
                "gamma_pow_H_terminal": gamma_terminal,
                "cross_objective_stage_plus_terminal": None if stage_total is None or gamma_terminal is None else float(float(stage_total) + float(gamma_terminal)),
                "vf_call_meta": meta,
                "centered_coeff_components_best_effort": (arm.get("terminal_coeff_components_best_effort") or {}).get(mode),
            }
        rows.append(row)
    return rows


def patch_runtime() -> None:
    # Activate all v34c/v0e repairs and current Astra report gate.
    contract.patch_runtime()
    base.NAME = NAME
    base.STAMP = STAMP
    base.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
    base.STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34d_objective_basin_solver_probe.md"
    base.BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34D_OBJECTIVE_BASIN_SOLVER_PROBE_{STAMP}.json"
    base.REQUEST_ID = f"v34d-objective-basin-solver-probe-{STAMP}"
    base.MARKER = f"vehicle-v34d-objective-basin-solver-probe-{STAMP}"
    base.CURRENT_ASTRA_REQUEST = CURRENT_REQUEST
    base.ASTRA_REPORT = CURRENT_REPORT
    base.CURRENT_ASTRA_SHA = CURRENT_SHA
    base.load_contexts = contract.c.load_contexts_strict
    base.configure_context_no_reset = configure_context_no_reset_v34d
    base.set_initial_guess = set_initial_guess_v34d
    base.verify_astra_and_gates = verify_gates_with_v0e
    base.reconstruct_stage_and_terminal = reconstruct_stage_and_terminal_v34d
    base.build_vf_evaluator_cache = build_vf_evaluator_cache_v34d
    base.cross_evaluate = cross_evaluate_v34d


def augment_provenance(exit_code: int) -> None:
    try:
        wrapper = Path(__file__).resolve()
        provenance = {
            "status": "completed" if exit_code == 0 else "failed",
            "created_utc": base.now_utc().isoformat(),
            "wrapper_script": base.rel(wrapper),
            "wrapper_script_sha256": base.sha256(wrapper),
            "base_script": base.rel(Path(base.__file__).resolve()),
            "base_script_sha256": base.sha256(Path(base.__file__).resolve()),
            "contract_preflight_wrapper": base.rel(Path(contract.__file__).resolve()),
            "contract_preflight_wrapper_sha256": base.sha256(Path(contract.__file__).resolve()),
            "repair_scope": "apply v34c/v0e contract primitives and vf4 cross-evaluation to the fixed 24-call v34 Task2 design",
            "scientific_design_changed": False,
            "solver_budget_cap": base.MAX_SOLVES,
            "plant_steps_added": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        prov_path = base.RUN_DIR / "v34d_wrapper_provenance.json"
        base.write_json(prov_path, provenance)
        for name in ("raw.json", "completed.json", "failed.json"):
            p = base.RUN_DIR / name
            if not p.exists():
                continue
            obj = base.read_json(p)
            obj["v34d_wrapper_repair_provenance"] = {**provenance, "path": base.rel(prov_path)}
            if name == "completed.json":
                hashes = dict(obj.get("hashes") or {})
                for hp in (prov_path, wrapper, Path(base.__file__).resolve(), Path(contract.__file__).resolve(), Path(contract_d.__file__).resolve()):
                    if hp.exists():
                        hashes[base.rel(hp)] = base.sha256(hp)
                obj["hashes"] = hashes
            base.write_json(p, obj)
        if base.BACKUP_REQUEST.exists():
            req = base.read_json(base.BACKUP_REQUEST)
            must = list(req.get("must_cover") or [])
            for add in [base.rel(wrapper), base.rel(prov_path)]:
                if add not in must:
                    must.append(add)
            req["must_cover"] = must
            req["v34d_wrapper_repair_provenance"] = {"path": base.rel(prov_path), "sha256": base.sha256(prov_path)}
            base.write_json(base.BACKUP_REQUEST, req)
    except Exception as exc:
        print(json.dumps({"v34d_provenance_augmentation_failed": repr(exc)}), flush=True)


ORIGINAL_VERIFY = base.verify_astra_and_gates


def main(argv: Optional[Sequence[str]] = None) -> int:
    patch_runtime()
    code = base.run(argv)
    augment_provenance(code)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
