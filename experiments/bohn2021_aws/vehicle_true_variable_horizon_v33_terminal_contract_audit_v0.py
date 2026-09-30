#!/usr/bin/env python3
"""v33 terminal-contract zero-rollout audit and evidence-chain closure.

Development-only metadata/numeric audit requested after Astra review of v33.
It performs no plant rollouts, no MPC solver calls, no validation64 access, no
sealed-test access, no training and no selector refit.

Checks implemented:
  * preserve v33 artifacts and write supplemental final hashes (non-self-
    referential receipts rather than editing raw/completed files);
  * verify V15/V35 terminal weight files and shapes against the source contract;
  * evaluate the polynomial terminal-value contract with NumPy and a CasADi
    expression using state order theta,x,y and parameter order goal_x,goal_y;
  * include theta +/- 2*pi variants for v33 branch states;
  * append RESPONSE_LOG and request external backup after the new artifacts.

The audit intentionally does not execute the 24 fixed-context solver-call probe;
that is the next step only if this zero-rollout gate and backup pass.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_v33_terminal_contract_audit_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / "research_artifacts/aws_diagnostics" / f"{NAME}_{STAMP}"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
V33_GLOB = "vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0_*"
STATE_ORDER = ["theta", "x", "y"]
PARAM_ORDER = ["goal_x", "goal_y"]
THETA_VARIANTS = [0.0, 2.0 * math.pi, -2.0 * math.pi]
MAX_NUMERIC_DIFF_TOL = 1e-8


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def clean(x: Any) -> Any:
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if hasattr(x, "item"):
        try:
            return clean(x.item())
        except Exception:
            pass
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(clean(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def latest_v33_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = ROOT / explicit if not Path(explicit).is_absolute() else Path(explicit)
        if not p.exists():
            raise RuntimeError(f"explicit v33 dir does not exist: {p}")
        return p
    dirs = sorted((ROOT / "research_artifacts/aws_diagnostics").glob(V33_GLOB), key=lambda p: p.name)
    if not dirs:
        raise RuntimeError("no v33 artifact directory found")
    return dirs[-1]


def file_receipt(path: Path) -> Dict[str, Any]:
    return {"path": rel(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None, "sha256": sha256(path) if path.exists() else None}


def np_hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a.astype(np.float64)).tobytes()).hexdigest()


def load_terminal(folder: Path) -> Dict[str, Any]:
    w_path = folder / "terminal_cnnvf_weights.npy"
    b_path = folder / "terminal_cnnvf_biases.npy"
    if not w_path.exists() or not b_path.exists():
        raise RuntimeError(f"terminal files missing in {folder}")
    w = np.load(str(w_path))
    b = np.load(str(b_path))
    w2 = np.asarray(w, dtype=float).reshape(-1, 1)
    b2 = np.asarray(b, dtype=float).reshape(-1, 1)
    return {
        "folder": rel(folder),
        "weights_path": rel(w_path),
        "biases_path": rel(b_path),
        "weights": w2,
        "biases": b2,
        "weights_shape": list(w.shape),
        "biases_shape": list(b.shape),
        "weights_flat_shape": list(w2.shape),
        "biases_flat_shape": list(b2.shape),
        "weights_sha256_file": sha256(w_path),
        "biases_sha256_file": sha256(b_path),
        "weights_sha256_numeric_float64": np_hash(w2),
        "biases_sha256_numeric_float64": np_hash(b2),
        "finite": bool(np.isfinite(w2).all() and np.isfinite(b2).all()),
        "weight_abs_max": float(np.max(np.abs(w2))) if w2.size else None,
        "bias_abs_max": float(np.max(np.abs(b2))) if b2.size else None,
    }


def direct_poly_value(weights: np.ndarray, biases: np.ndarray, state_vec: Sequence[float], param_vec: Sequence[float]) -> float:
    x = np.asarray(list(state_vec) + list(param_vec), dtype=float).reshape(1, -1)
    feat = np.concatenate([x, x ** 2], axis=1)
    return float((feat @ weights.reshape(-1, 1) + biases.reshape(1, 1))[0, 0])


def casadi_poly_value(weights: np.ndarray, biases: np.ndarray, state_vec: Sequence[float], param_vec: Sequence[float]) -> Tuple[Optional[float], Optional[str]]:
    try:
        import casadi as ca  # type: ignore
        state = ca.SX.sym("state", 3)
        params = ca.SX.sym("params", 2)
        inp = ca.vertcat(state, params).T
        hidden = ca.horzcat(inp, ca.power(inp, 2))
        out = hidden @ ca.DM(weights.reshape(-1, 1)) + ca.DM(biases.reshape(1, 1))
        fn = ca.Function("v33_terminal_contract_eval", [state, params], [out])
        val = fn(np.asarray(state_vec, dtype=float).reshape(3, 1), np.asarray(param_vec, dtype=float).reshape(2, 1))
        return float(np.asarray(val).reshape(-1)[0]), None
    except Exception as exc:
        return None, repr(exc)


def extract_branch_states(v33_dir: Path) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    ep_root = v33_dir / "episodes"
    for br in sorted(ep_root.glob("*/branch_reset.json")):
        try:
            data = read_json(br)
        except Exception:
            continue
        state = data.get("branch_state_target") or data.get("branch_state_after_direct_reset")
        if not isinstance(state, Mapping):
            continue
        ep = br.parent.name
        # Folder names include ..._<state_label>_<terminal_mode>_Hxx_trueHxx.  Prefer
        # the manifest/raw state labels later, but this robust parse is enough for
        # sample identity in a zero-rollout numeric audit.
        label = None
        for token in ["v27_case09_slot0_early_risk", "v27_case09_slot1_mid_late_risk", "v19_c12", "v19_c13", "v27_case00_slot1_mid_late_risk", "v27_case08_slot1_mid_late_risk"]:
            if token in ep:
                label = token
                break
        if label is None:
            label = ep
        out.setdefault(label, {"state_label": label, "state": dict(state), "branch_reset_path": rel(br), "sample_episode": ep})
    return [out[k] for k in sorted(out)]


def extract_goal_candidates(v33_dir: Path, branch_states: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Best-effort metadata-only goal/parameter extraction.

    v33 traces do not expose mpc.opt_p_num; branch_reset also stores lengths but
    not actual shifted TVP arrays.  This audit therefore evaluates the terminal
    contract on declared state order with a neutral goal vector and, when a first
    observation is available, a heuristic immediate-trajectory parameter vector.
    The limitation is explicit in raw/summary and blocks any conclusion about
    the physical correctness of the learned value function itself.
    """
    candidates: Dict[str, Any] = {"zero_goal": [0.0, 0.0]}
    # Heuristic: observation slots 3,4 correspond to scaled trajectory_x/y in the
    # vehicle config.  We include them only as an additional finite vector, not as
    # an asserted deployment goal.
    for bs in branch_states:
        br = ROOT / str(bs["branch_reset_path"])
        try:
            data = read_json(br)
            obs = data.get("initial_observation_at_branch") or []
            if len(obs) >= 5:
                candidates[f"obs_slots_3_4_for_{bs['state_label']}"] = [float(obs[3]), float(obs[4])]
        except Exception:
            pass
    return candidates


def append_response_log(block: str, marker: str) -> None:
    RESPONSE_LOG.parent.mkdir(parents=True, exist_ok=True)
    old = RESPONSE_LOG.read_text(encoding="utf-8", errors="replace") if RESPONSE_LOG.exists() else ""
    if marker not in old:
        RESPONSE_LOG.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def make_summary(raw: Mapping[str, Any]) -> str:
    gate = raw["gate"]
    lines = [
        f"# {NAME}",
        "",
        f"created_utc: `{raw['created_utc']}`",
        f"v33_dir: `{raw['v33_dir']}`",
        "",
        "## Gate",
        f"- passed: `{gate['passed']}`",
        f"- max_numpy_vs_casadi_abs_diff: `{gate['max_numpy_vs_casadi_abs_diff']}`",
        f"- terminal_shape_contract_ok: `{gate['terminal_shape_contract_ok']}`",
        f"- no_rollouts_solver_training_validation_test: `{gate['no_rollouts_solver_training_validation_test']}`",
        "",
        "## Important limitation",
        "v33 artifacts do not expose the deployed `goal_x,goal_y` parameter vector at each solver call. The audit therefore verifies the polynomial/CasADi terminal evaluation contract, file hashes, shape/order assumptions and theta wrapping sensitivity on branch states with neutral and heuristic parameter vectors; it does not prove that a learned terminal function is physically accurate.",
        "",
        "## Terminal files",
    ]
    for name, info in raw["terminal_audit"].items():
        lines.append(f"- {name}: weights_shape={info['weights_flat_shape']} biases_shape={info['biases_flat_shape']} finite={info['finite']} file_sha={info['weights_sha256_file'][:12]}...")
    lines.extend([
        "",
        "## Theta-wrap sensitivity",
    ])
    for name, stats in raw["theta_wrap_sensitivity_by_terminal"].items():
        lines.append(f"- {name}: max_abs_delta_vs_base={stats['max_abs_delta_vs_base']:.6g}, samples={stats['samples']}")
    lines.extend([
        "",
        "## Evidence-chain closure",
        f"- supplemental receipt: `{raw['supplemental_final_hash_receipts_path']}`",
        f"- response log updated: `{raw['response_log_updated']}`",
        f"- backup request: `{raw['backup_request_path']}`",
        "",
        "## Next pre-authorized step if backup remains verified",
        "Implement the separate 24 fixed-context solver-call probe to distinguish accepted NLP objective preference from optimization/warm-start basin effects. Do not start selector training, validation64 or sealed test from this audit alone.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v33-dir", default=None)
    ap.add_argument("--backup-time", default=None)
    ap.add_argument("--backup-commit", default=None)
    ap.add_argument("--backup-package-sha256", default=None)
    ap.add_argument("--i-accept-zero-rollout-terminal-contract-audit", action="store_true")
    args = ap.parse_args()
    if not args.i_accept_zero_rollout_terminal_contract_audit:
        raise RuntimeError("explicit zero-rollout terminal-contract audit flag required")

    RUN_DIR.mkdir(parents=True, exist_ok=False)
    v33_dir = latest_v33_dir(args.v33_dir)
    terminal_sources = read_json(v33_dir / "terminal_sources.json")
    receipts = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "supplemental non-self-referential final hashes; original v33 files are preserved unchanged",
        "note": "This closes the v33 evidence chain after known write-order/self-reference ambiguity without editing raw.json or completed.json.",
        "v33_dir": rel(v33_dir),
        "files": {},
    }
    for p in [v33_dir / "raw.json", v33_dir / "completed.json", v33_dir / "summary.md", v33_dir / "selected_state_manifest.json", v33_dir / "terminal_sources.json"]:
        receipts["files"][rel(p)] = file_receipt(p)
    for trace in sorted((v33_dir / "episodes").glob("*/trace.jsonl")):
        receipts["files"][rel(trace)] = file_receipt(trace)

    terminal_audit: Dict[str, Any] = {}
    numeric_rows: List[Dict[str, Any]] = []
    max_diff = 0.0
    casadi_errors: List[str] = []
    branch_states = extract_branch_states(v33_dir)
    goal_candidates = extract_goal_candidates(v33_dir, branch_states)
    for h_name in ("15", "35"):
        src = terminal_sources[h_name]
        term = load_terminal(ROOT / src["folder"])
        w = term.pop("weights")
        b = term.pop("biases")
        expected_features = 2 * (len(STATE_ORDER) + len(PARAM_ORDER))
        term["expected_poly_feature_count"] = expected_features
        term["shape_contract_ok"] = bool(w.shape == (expected_features, 1) and b.shape == (1, 1) and term["finite"])
        term["source_manifest_weights_hash"] = src.get("weights_hash")
        term["source_model_zip_sha256"] = src.get("model_zip_sha256")
        terminal_audit[f"V{h_name}"] = term
        for bs in branch_states:
            s = bs["state"]
            base_state = [float(s[k]) for k in STATE_ORDER]
            for param_label, param_vec in goal_candidates.items():
                # Keep per-state heuristic parameters attached to their own state;
                # the neutral goal is global.
                if param_label.startswith("obs_slots_3_4_for_") and not param_label.endswith(str(bs["state_label"])):
                    continue
                base_val: Optional[float] = None
                for delta in THETA_VARIANTS:
                    state_vec = list(base_state)
                    state_vec[0] += delta
                    np_val = direct_poly_value(w, b, state_vec, param_vec)
                    ca_val, ca_err = casadi_poly_value(w, b, state_vec, param_vec)
                    if ca_err:
                        casadi_errors.append(ca_err)
                    diff = None if ca_val is None else abs(np_val - ca_val)
                    if diff is not None:
                        max_diff = max(max_diff, float(diff))
                    if delta == 0.0:
                        base_val = np_val
                    numeric_rows.append({
                        "terminal": f"V{h_name}",
                        "state_label": bs["state_label"],
                        "state_order": STATE_ORDER,
                        "param_order": PARAM_ORDER,
                        "param_label": param_label,
                        "theta_delta": delta,
                        "input_state_vec": state_vec,
                        "input_param_vec": list(map(float, param_vec)),
                        "numpy_value": np_val,
                        "casadi_value": ca_val,
                        "abs_diff": diff,
                        "delta_vs_base_same_param": None if base_val is None else np_val - base_val,
                    })

    theta_stats: Dict[str, Any] = {}
    for term in ["V15", "V35"]:
        vals = [abs(float(r["delta_vs_base_same_param"])) for r in numeric_rows if r["terminal"] == term and r["delta_vs_base_same_param"] is not None and float(r["theta_delta"]) != 0.0]
        theta_stats[term] = {"samples": len(vals), "max_abs_delta_vs_base": max(vals) if vals else 0.0, "mean_abs_delta_vs_base": float(sum(vals) / len(vals)) if vals else 0.0}

    shape_ok = all(v["shape_contract_ok"] for v in terminal_audit.values())
    casadi_ok = not casadi_errors and max_diff <= MAX_NUMERIC_DIFF_TOL
    gate = {
        "passed": bool(shape_ok and casadi_ok),
        "terminal_shape_contract_ok": bool(shape_ok),
        "numpy_casadi_contract_ok": bool(casadi_ok),
        "max_numpy_vs_casadi_abs_diff": max_diff,
        "tolerance": MAX_NUMERIC_DIFF_TOL,
        "casadi_error_count": len(casadi_errors),
        "no_rollouts_solver_training_validation_test": True,
        "rollout_episodes": 0,
        "plant_steps": 0,
        "solver_calls": 0,
        "gradient_steps": 0,
        "selector_refits": 0,
        "validation64_episodes": 0,
        "sealed_test_episodes": 0,
    }

    receipt_path = RUN_DIR / "supplemental_final_hash_receipts.json"
    write_json(receipt_path, receipts)
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": STAMP,
        "method": "IMPROVED zero-rollout terminal-contract numeric audit and evidence-chain closure; not ORIGINAL SAC; not validation/test",
        "v33_dir": rel(v33_dir),
        "backup_context": {"time": args.backup_time, "commit": args.backup_commit, "package_sha256": args.backup_package_sha256},
        "state_order": STATE_ORDER,
        "parameter_order": PARAM_ORDER,
        "branch_state_count": len(branch_states),
        "branch_states": branch_states,
        "parameter_candidates_used": goal_candidates,
        "parameter_limitation": "v33 did not record deployed goal_x/goal_y opt_p values; zero_goal and heuristic observation-slot parameter vectors are audit inputs, not asserted true deployment parameters.",
        "terminal_audit": terminal_audit,
        "numeric_rows": numeric_rows,
        "theta_wrap_sensitivity_by_terminal": theta_stats,
        "gate": gate,
        "casadi_errors_unique": sorted(set(casadi_errors))[:5],
        "supplemental_final_hash_receipts_path": rel(receipt_path),
    }
    raw_path = RUN_DIR / "raw.json"
    write_json(raw_path, raw)
    raw["raw_sha256"] = sha256(raw_path)

    backup_request = {
        "request_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "Backup required after v33 terminal-contract audit before further unique science/solver-call probe.",
        "experiment": NAME,
        "artifact_dir": rel(RUN_DIR),
        "required_paths": [rel(RUN_DIR), rel(RESPONSE_LOG), rel(NEXT_REVIEW_REQUEST)],
        "no_secrets": True,
    }
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V33_TERMINAL_CONTRACT_AUDIT_{STAMP}.json"
    write_json(backup_path, backup_request)
    raw["backup_request_path"] = rel(backup_path)

    marker = f"v33-terminal-contract-audit-{STAMP}"
    response_block = f"""
## Follow-up through v33 terminal-contract audit (`{marker}`)

Updated by GPT-5.5 executor at `{dt.datetime.now(dt.timezone.utc).isoformat()}`. This was a zero-rollout audit: 0 plant steps, 0 solver calls, 0 training/refit, 0 validation64 and 0 sealed-test episodes.

| linked recommendation(s) | disposition after audit | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` / latest Astra v33 direction | accepted; evidence chain closed, terminal numeric contract checked | Audit artifacts `{rel(raw_path)}` and `{rel(receipt_path)}`. Gate passed={gate['passed']}; terminal_shape_contract_ok={gate['terminal_shape_contract_ok']}; max NumPy-vs-CasADi diff={gate['max_numpy_vs_casadi_abs_diff']:.3g}. | Do not rerun v33. After verified backup, proceed only to the pre-authorized 24 fixed-context solver-call probe if no fresh Astra objection supersedes it. |
| `A12_registry_backup_schema_contract` | accepted; active | Backup request `{rel(backup_path)}` written for the audit artifacts and response log. | Require verified external backup before additional unique science. |
"""
    append_response_log(response_block, marker)
    raw["response_log_updated"] = rel(RESPONSE_LOG)

    review_request = {
        "request_id": f"v33-terminal-contract-audit-{STAMP}",
        "experiment_id": STAMP,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "question": "Audit evidence chain and terminal polynomial/CasADi contract for v33. If gate passes and backup is verified, confirm whether executor should proceed with Astra's 24 fixed-context solver-call probe separating objective preference from optimization basin.",
        "evidence_paths": [rel(raw_path), rel(receipt_path), rel(RUN_DIR / "summary.md"), rel(backup_path), rel(RESPONSE_LOG)],
        "access_budget": {"rollout_episodes": 0, "plant_steps": 0, "solver_calls": 0, "gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "current_gate": gate,
        "limitations": [raw["parameter_limitation"], "No objective stage/slack/gamma^H decomposition and no alternate-initialization solver calls in this audit."],
    }
    write_json(NEXT_REVIEW_REQUEST, review_request)
    raw["next_review_request_path"] = rel(NEXT_REVIEW_REQUEST)

    # Rewrite final raw after adding paths, then summary and completed markers.
    write_json(raw_path, raw)
    summary = make_summary(raw)
    summary_path = RUN_DIR / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    completed = {
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "passed": bool(gate["passed"]),
        "artifact_dir": rel(RUN_DIR),
        "raw_path": rel(raw_path),
        "summary_path": rel(summary_path),
        "supplemental_final_hash_receipts_path": rel(receipt_path),
        "backup_request_path": rel(backup_path),
        "next_review_request_path": rel(NEXT_REVIEW_REQUEST),
        "hashes": {rel(raw_path): sha256(raw_path), rel(summary_path): sha256(summary_path), rel(receipt_path): sha256(receipt_path), rel(backup_path): sha256(backup_path), rel(NEXT_REVIEW_REQUEST): sha256(NEXT_REVIEW_REQUEST)},
        "budgets": gate,
    }
    completed_path = RUN_DIR / "completed.json"
    write_json(completed_path, completed)
    print(json.dumps({"passed": gate["passed"], "artifact_dir": rel(RUN_DIR), "summary": rel(summary_path), "raw": rel(raw_path), "completed": rel(completed_path), "backup_request": rel(backup_path), "max_diff": max_diff, "branch_state_count": len(branch_states)}, sort_keys=True))


if __name__ == "__main__":
    main()
