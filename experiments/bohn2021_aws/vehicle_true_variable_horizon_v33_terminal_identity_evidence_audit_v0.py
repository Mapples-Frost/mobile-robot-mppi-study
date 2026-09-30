#!/usr/bin/env python3
"""v33 real-loader terminal identity and evidence-chain audit.

Astra follow-up after the v33 terminal x H probe and the narrow polynomial
contract audit.  This diagnostic performs no rollouts, no plant steps, no MPC
solver calls, no validation64 access, no sealed-test access, no training and no
selector refit.

It uses the same terminal-loading path used by v33 (stage1_runner.load_terminal_grid)
to compare in-memory terminal tensors against the exported .npy files, writes a
signed coefficient table, quantifies theta wrap/curvature on the six opened v33
branch states, and verifies v33 schedule/trace-count evidence assertions.  It
does not claim learned terminal accuracy and it does not execute the later
24-call solver-basin probe.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

NAME = "vehicle_true_variable_horizon_v33_terminal_identity_evidence_audit_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / "research_artifacts/aws_diagnostics" / f"{NAME}_{STAMP}"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
BACKUP_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V33_TERMINAL_IDENTITY_EVIDENCE_AUDIT_{STAMP}.json"
STATE = ROOT / "research_artifacts/aws_state" / f"continue_state_{STAMP}_after_v33_terminal_identity_evidence_audit.md"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
V33_GLOB = "vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0_*"
TARGET_LABELS = [
    "v27_case09_slot0_early_risk",
    "v27_case09_slot1_mid_late_risk",
    "v19_c12",
    "v19_c13",
    "v27_case00_slot1_mid_late_risk",
    "v27_case08_slot1_mid_late_risk",
]
STATE_ORDER = ["theta", "x", "y"]
PARAM_ORDER = ["goal_x", "goal_y"]
FEATURES = STATE_ORDER + PARAM_ORDER
THETA_DELTAS = [0.0, 2.0 * math.pi, -2.0 * math.pi]
TENSOR_TOL = 1e-12


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


def bytes_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr.astype(np.float64)).tobytes()).hexdigest()


def latest_v33_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        p = p if p.is_absolute() else ROOT / p
        if not p.exists():
            raise RuntimeError(f"explicit v33 dir missing: {p}")
        return p
    dirs = sorted((ROOT / "research_artifacts/aws_diagnostics").glob(V33_GLOB), key=lambda p: p.name)
    if not dirs:
        raise RuntimeError("no v33 directory found")
    return dirs[-1]


def flatten_any(obj: Any) -> np.ndarray:
    if isinstance(obj, (list, tuple)):
        parts = [np.asarray(x, dtype=np.float64).ravel(order="F") for x in obj]
        return np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)
    return np.asarray(obj, dtype=np.float64).ravel(order="F")


def file_receipt(path: Path) -> Dict[str, Any]:
    return {"path": rel(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None, "sha256": sha256(path) if path.exists() else None}


def import_terminal_loader() -> Tuple[Any, Any]:
    import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # type: ignore
    _base_smoke, stage1_runner, _fixed_base = v1d.import_legacy_modules()
    return v1d, stage1_runner


def load_with_real_v33_path(stage1_runner: Any) -> Tuple[Mapping[int, Any], Mapping[str, Any], Dict[str, Any]]:
    proto_path = Path(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    proto = read_json(proto_path)
    terminals, receipts = stage1_runner.load_terminal_grid(proto["terminal_grid_readiness_reused_from_v1"])
    return terminals, receipts, {"path": rel(proto_path), "sha256": sha256(proto_path)}


def compare_terminal(h: int, term_pair: Any, receipt: Mapping[str, Any], coeff_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    folder = ROOT / str(receipt.get("folder"))
    w_path = folder / "terminal_cnnvf_weights.npy"
    b_path = folder / "terminal_cnnvf_biases.npy"
    if not w_path.exists() or not b_path.exists():
        raise RuntimeError(f"terminal npy files missing for H{h}: {folder}")
    file_w = np.load(str(w_path))
    file_b = np.load(str(b_path))
    try:
        loader_w_obj, loader_b_obj = term_pair
    except Exception as exc:
        raise RuntimeError(f"unexpected terminal pair for H{h}: {type(term_pair)}") from exc
    loader_w = flatten_any(loader_w_obj)
    loader_b = flatten_any(loader_b_obj)
    file_w_f = np.asarray(file_w, dtype=np.float64).ravel(order="F")
    file_b_f = np.asarray(file_b, dtype=np.float64).ravel(order="F")
    file_w_c = np.asarray(file_w, dtype=np.float64).ravel(order="C")
    file_b_c = np.asarray(file_b, dtype=np.float64).ravel(order="C")

    def maxdiff(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        if a.shape != b.shape:
            return None
        return float(np.max(np.abs(a - b))) if a.size else 0.0

    dw_f = maxdiff(loader_w, file_w_f); db_f = maxdiff(loader_b, file_b_f)
    dw_c = maxdiff(loader_w, file_w_c); db_c = maxdiff(loader_b, file_b_c)
    if file_w_f.size == 2 * len(FEATURES):
        lin = file_w_f[: len(FEATURES)]
        quad = file_w_f[len(FEATURES) :]
        for i, name in enumerate(FEATURES):
            coeff_rows.append({
                "terminal": f"V{h}",
                "feature": name,
                "linear_coeff": float(lin[i]),
                "quadratic_coeff": float(quad[i]),
                "curvature_second_derivative_if_applicable": float(2.0 * quad[i]),
            })
    return {
        "horizon": h,
        "folder": rel(folder),
        "loader_weight_shape_flat": list(loader_w.shape),
        "loader_bias_shape_flat": list(loader_b.shape),
        "file_weight_shape": list(np.asarray(file_w).shape),
        "file_bias_shape": list(np.asarray(file_b).shape),
        "file_weight_sha256": sha256(w_path),
        "file_bias_sha256": sha256(b_path),
        "loader_weight_numeric_sha256": bytes_hash(loader_w),
        "loader_bias_numeric_sha256": bytes_hash(loader_b),
        "file_weight_numeric_sha256_F": bytes_hash(file_w_f),
        "file_bias_numeric_sha256_F": bytes_hash(file_b_f),
        "receipt_weights_hash": receipt.get("weights_hash"),
        "receipt_model_zip_sha256": receipt.get("model_zip_sha256"),
        "max_abs_diff_loader_vs_file_order_F": dw_f,
        "max_abs_diff_loader_vs_file_order_C": dw_c,
        "max_abs_diff_loader_bias_vs_file_order_F": db_f,
        "max_abs_diff_loader_bias_vs_file_order_C": db_c,
        "loader_matches_file_F": bool(dw_f is not None and db_f is not None and dw_f <= TENSOR_TOL and db_f <= TENSOR_TOL),
        "loader_matches_file_C": bool(dw_c is not None and db_c is not None and dw_c <= TENSOR_TOL and db_c <= TENSOR_TOL),
        "finite": bool(np.isfinite(loader_w).all() and np.isfinite(loader_b).all() and np.isfinite(file_w_f).all() and np.isfinite(file_b_f).all()),
        "theta_linear_coeff": float(file_w_f[0]) if file_w_f.size >= 1 else None,
        "theta_quadratic_coeff": float(file_w_f[len(FEATURES)]) if file_w_f.size >= 2 * len(FEATURES) else None,
        "theta_second_derivative": float(2.0 * file_w_f[len(FEATURES)]) if file_w_f.size >= 2 * len(FEATURES) else None,
        "bias_scalar": float(file_b_f[0]) if file_b_f.size else None,
    }


def extract_branch_states(v33_dir: Path) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for br in sorted((v33_dir / "episodes").glob("*/branch_reset.json")):
        try:
            data = read_json(br)
        except Exception:
            continue
        state = data.get("branch_state_target") or data.get("branch_state_after_direct_reset")
        if not isinstance(state, Mapping):
            continue
        ep_name = br.parent.name
        label = next((x for x in TARGET_LABELS if x in ep_name), ep_name)
        obs = data.get("initial_observation_at_branch") or []
        out.setdefault(label, {"state_label": label, "state": dict(state), "branch_reset_path": rel(br), "sample_episode": ep_name, "initial_observation_at_branch": obs})
    return [out[k] for k in sorted(out)]


def poly_value_grad(weights_flat: np.ndarray, bias_flat: np.ndarray, state_vec: Sequence[float], param_vec: Sequence[float]) -> Dict[str, Any]:
    z = np.asarray(list(state_vec) + list(param_vec), dtype=np.float64)
    n = len(z)
    lin = weights_flat[:n]
    quad = weights_flat[n:2*n]
    value = float(bias_flat[0] + np.dot(lin, z) + np.dot(quad, z * z))
    grad = lin + 2.0 * quad * z
    return {"value": value, "gradient_by_input": {FEATURES[i]: float(grad[i]) for i in range(n)}, "theta_second_derivative": float(2.0 * quad[0])}


def terminal_eval_rows(h: int, receipt: Mapping[str, Any], branch_states: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    folder = ROOT / str(receipt.get("folder"))
    w = np.asarray(np.load(str(folder / "terminal_cnnvf_weights.npy")), dtype=np.float64).ravel(order="F")
    b = np.asarray(np.load(str(folder / "terminal_cnnvf_biases.npy")), dtype=np.float64).ravel(order="F")
    rows: List[Dict[str, Any]] = []
    deltas_abs: List[float] = []
    for bs in branch_states:
        st = bs["state"]
        base_state = [float(st[k]) for k in STATE_ORDER]
        params = {"zero_goal_not_deployment": [0.0, 0.0]}
        obs = bs.get("initial_observation_at_branch") or []
        if isinstance(obs, list) and len(obs) >= 5:
            params["obs_slots_3_4_heuristic_not_deployment"] = [float(obs[3]), float(obs[4])]
        for param_label, param_vec in params.items():
            base_val = None
            for dtheta in THETA_DELTAS:
                sv = list(base_state); sv[0] += dtheta
                ev = poly_value_grad(w, b, sv, param_vec)
                if dtheta == 0.0:
                    base_val = ev["value"]
                else:
                    deltas_abs.append(abs(ev["value"] - float(base_val)))
                rows.append({
                    "terminal": f"V{h}", "state_label": bs["state_label"], "state_order": STATE_ORDER,
                    "param_order": PARAM_ORDER, "param_label": param_label,
                    "theta_delta": dtheta, "state_vec": sv, "param_vec": param_vec,
                    "value": ev["value"], "delta_vs_base_same_param": None if base_val is None else ev["value"] - float(base_val),
                    "gradient_by_input": ev["gradient_by_input"], "theta_second_derivative": ev["theta_second_derivative"],
                })
    return rows, {"samples": len(deltas_abs), "max_abs_delta_vs_base": max(deltas_abs) if deltas_abs else 0.0, "mean_abs_delta_vs_base": float(sum(deltas_abs) / len(deltas_abs)) if deltas_abs else 0.0}


def verify_v33_evidence(v33_dir: Path) -> Dict[str, Any]:
    raw_path = v33_dir / "raw.json"
    completed_path = v33_dir / "completed.json"
    raw = read_json(raw_path)
    schedule = raw.get("schedule") or []
    episodes = raw.get("episodes") or []
    trace_files = sorted((v33_dir / "episodes").glob("*/trace.jsonl"))
    trace_line_counts: Dict[str, int] = {}
    for tf in trace_files:
        with tf.open("r", encoding="utf-8") as f:
            trace_line_counts[rel(tf)] = sum(1 for line in f if line.strip())
    schedule_cells = {(str(r.get("state_label")), int(r.get("horizon")), str(r.get("terminal_mode"))) for r in schedule}
    expected_cells = {(s, h, m) for s in TARGET_LABELS for h in [12, 15, 25, 35] for m in ["zero", "V15_shared", "V35_shared"]}
    trace_total = int(sum(trace_line_counts.values()))
    budget_steps = int((raw.get("budget_actual") or {}).get("control_steps", -1))
    return {
        "v33_dir": rel(v33_dir),
        "raw_receipt": file_receipt(raw_path),
        "completed_receipt": file_receipt(completed_path),
        "summary_receipt": file_receipt(v33_dir / "summary.md"),
        "schedule_rows": len(schedule),
        "episode_rows_in_raw": len(episodes),
        "episode_dirs_with_trace_jsonl": len(trace_files),
        "trace_line_total": trace_total,
        "budget_control_steps_recorded": budget_steps,
        "schedule_unique_cells": len(schedule_cells),
        "expected_cells": len(expected_cells),
        "missing_cells": sorted([f"{a}|H{b}|{c}" for (a, b, c) in expected_cells - schedule_cells]),
        "extra_cells": sorted([f"{a}|H{b}|{c}" for (a, b, c) in schedule_cells - expected_cells]),
        "schedule_trace_count_gate": bool(len(schedule) == 72 and len(episodes) == 72 and len(trace_files) == 72 and trace_total == budget_steps == 4013 and len(schedule_cells) == 72 and not (expected_cells - schedule_cells) and not (schedule_cells - expected_cells)),
        "selected_state_count": len(raw.get("selected_states") or []),
        "aggregate_from_raw": (raw.get("analysis") or {}).get("aggregate"),
        "source242_by_terminal_mode_from_raw": (raw.get("analysis") or {}).get("source242_by_terminal_mode"),
        "v19_c13_H35_terminal_effect_from_raw": (raw.get("analysis") or {}).get("v19_c13_H35_terminal_effect"),
    }


def write_coeff_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["terminal", "feature", "linear_coeff", "quadratic_coeff", "curvature_second_derivative_if_applicable"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def append_response(marker: str, raw_path: Path, summary_path: Path, passed: bool, gate: Mapping[str, Any]) -> None:
    block = f"""
## Follow-up through v33 terminal identity/evidence audit (`{marker}`)

Updated by GPT-5.5 executor at `{dt.datetime.now(dt.timezone.utc).isoformat()}`. Zero-rollout/zero-solver audit: 0 plant steps, 0 solver calls, 0 training/refit, 0 validation64 and 0 sealed-test episodes.

| linked recommendation(s) | disposition after audit | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A11_training_failure_modes_need_separation` | accepted; real terminal loader/file identity tested | `{rel(raw_path)}` and `{rel(summary_path)}`. loader_identity_passed={gate.get('loader_identity_passed')}; max loader-vs-file diff={gate.get('max_loader_file_diff')}. | If backup is verified and Astra raises no contrary evidence, remaining pre-authorized mechanism step is the 24 fixed-context solver-call probe; full TF/checkpoint graph equality remains separately noted if unavailable. |
| `A12_registry_backup_schema_contract` | accepted; v33 schedule/trace assertions checked | v33 schedule_trace_count_gate={gate.get('schedule_trace_count_gate')}; trace lines match control steps if true. | Backup request written; require verified external backup before solver-call probe. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; coefficient/terminal confound preserved | Coefficient table and theta-wrap stats in audit artifacts. | Do not treat old H labels as pure horizon labels; keep H35/zero and H35/V15 as strong comparators. |
"""
    old = RESPONSE_LOG.read_text(encoding="utf-8", errors="replace") if RESPONSE_LOG.exists() else ""
    if marker not in old:
        RESPONSE_LOG.parent.mkdir(parents=True, exist_ok=True)
        RESPONSE_LOG.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_summary(path: Path, raw: Mapping[str, Any]) -> None:
    gate = raw["gate"]
    lines = [
        f"# {NAME}", "",
        f"created_utc: `{raw['created_utc']}`", f"v33_dir: `{raw['v33_dir']}`", "",
        "## Gate", f"- passed: `{gate['passed']}`", f"- loader_identity_passed: `{gate['loader_identity_passed']}`", f"- schedule_trace_count_gate: `{gate['schedule_trace_count_gate']}`", f"- max_loader_file_diff: `{gate['max_loader_file_diff']}`", "",
        "## Terminal identity", "",
    ]
    for h, info in raw["terminal_identity"].items():
        lines.append(f"- {h}: matches_file_F={info['loader_matches_file_F']} theta_quad={info['theta_quadratic_coeff']} theta_second_derivative={info['theta_second_derivative']} file_sha={str(info['file_weight_sha256'])[:12]}...")
    lines += ["", "## Theta-wrap sensitivity on six opened branch states", ""]
    for h, st in raw["theta_wrap_sensitivity"].items():
        lines.append(f"- {h}: max_abs_delta_vs_base={st['max_abs_delta_vs_base']:.6g}, mean={st['mean_abs_delta_vs_base']:.6g}, samples={st['samples']}")
    lines += ["", "## v33 evidence assertions", "", f"- schedule rows/raw episodes/trace files: `{raw['v33_evidence']['schedule_rows']}` / `{raw['v33_evidence']['episode_rows_in_raw']}` / `{raw['v33_evidence']['episode_dirs_with_trace_jsonl']}`", f"- trace_line_total vs recorded control steps: `{raw['v33_evidence']['trace_line_total']}` vs `{raw['v33_evidence']['budget_control_steps_recorded']}`", "", "## Limitations", ""]
    for lim in raw["limitations"]:
        lines.append(f"- {lim}")
    lines += ["", f"Coefficient table: `{raw['coefficient_table_csv']}`", f"Backup request: `{raw['backup_request_path']}`", "", "Next if backup/Astra permit: implement the already specified 24 fixed-context solver-call objective-vs-basin probe; do not start selector training or validation64 from this audit alone."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v33-dir", default=None)
    ap.add_argument("--backup-time", default=None)
    ap.add_argument("--backup-commit", default=None)
    ap.add_argument("--backup-package-sha256", default=None)
    ap.add_argument("--i-accept-zero-rollout-terminal-identity-audit", action="store_true", required=True)
    args = ap.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    try:
        v33_dir = latest_v33_dir(args.v33_dir)
        _v1d, stage1_runner = import_terminal_loader()
        terminals, receipts, terminal_protocol = load_with_real_v33_path(stage1_runner)
        coeff_rows: List[Dict[str, Any]] = []
        terminal_identity: Dict[str, Any] = {}
        for h in (15, 35):
            terminal_identity[f"V{h}"] = compare_terminal(h, terminals[h], receipts[str(h)] if str(h) in receipts else receipts[h], coeff_rows)
        coeff_path = RUN_DIR / "terminal_coefficients.csv"
        write_coeff_csv(coeff_path, coeff_rows)
        branch_states = extract_branch_states(v33_dir)
        eval_rows: List[Dict[str, Any]] = []
        theta_stats: Dict[str, Any] = {}
        for h in (15, 35):
            rows, stats = terminal_eval_rows(h, receipts[str(h)] if str(h) in receipts else receipts[h], branch_states)
            eval_rows.extend(rows); theta_stats[f"V{h}"] = stats
        v33_evidence = verify_v33_evidence(v33_dir)
        maxdiffs: List[float] = []
        for info in terminal_identity.values():
            for k in ("max_abs_diff_loader_vs_file_order_F", "max_abs_diff_loader_bias_vs_file_order_F"):
                if info[k] is not None:
                    maxdiffs.append(float(info[k]))
        gate = {
            "passed": bool(all(x["loader_matches_file_F"] for x in terminal_identity.values()) and v33_evidence["schedule_trace_count_gate"]),
            "loader_identity_passed": bool(all(x["loader_matches_file_F"] for x in terminal_identity.values())),
            "schedule_trace_count_gate": bool(v33_evidence["schedule_trace_count_gate"]),
            "max_loader_file_diff": max(maxdiffs) if maxdiffs else None,
            "full_tf_checkpoint_graph_evaluated": False,
            "deployed_mpc_vf_fun_evaluated": False,
            "rollout_episodes": 0, "plant_steps": 0, "solver_calls": 0, "gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0,
        }
        raw = {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": NAME,
            "classification": "development_IMPROVED_zero_rollout_terminal_identity_evidence_audit_not_validation_not_test",
            "v33_dir": rel(v33_dir),
            "pre_run_backup_context": {"time": args.backup_time, "commit": args.backup_commit, "package_sha256": args.backup_package_sha256, "note": "context may predate this audit; a new backup request is written"},
            "terminal_source_protocol": terminal_protocol,
            "terminal_identity": terminal_identity,
            "coefficient_table_csv": rel(coeff_path),
            "branch_states": branch_states,
            "terminal_value_gradient_rows": eval_rows,
            "theta_wrap_sensitivity": theta_stats,
            "v33_evidence": v33_evidence,
            "gate": gate,
            "limitations": [
                "opened v33 development states only; not validation64 and not sealed test",
                "real v33 terminal loader path was compared to exported .npy tensors; full stable-baselines TF graph and live mpc.vf_fun evaluation were not executed in this zero-rollout script",
                "goal_x/goal_y deployed opt_p values remain unavailable in v33 traces; observation slots are reported only as heuristics, not true goals",
                "theta-wrap values are function-property diagnostics on branch states, not proof of closed-loop causality",
                "no objective decomposition and no alternate-initialization solver calls in this audit",
            ],
            "backup_request_path": rel(BACKUP_REQUEST),
        }
        raw_path = RUN_DIR / "raw.json"
        summary_path = RUN_DIR / "summary.md"
        write_json(raw_path, raw)
        write_summary(summary_path, raw)
        write_json(BACKUP_REQUEST, {"request": "backup_after_v33_terminal_identity_evidence_audit", "created_utc": raw["created_utc"], "backup_required_before_solver_call_probe_or_other_unique_science": True, "reason": "new zero-rollout terminal identity/evidence audit artifacts and handoff", "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(BACKUP_REQUEST), rel(RESPONSE_LOG), rel(NEXT_REVIEW_REQUEST), rel(STATE)], "budgets": gate, "validation64_bank_opened": False, "sealed_test_accessed": False})
        marker = f"v33-terminal-identity-evidence-audit-{STAMP}"
        append_response(marker, raw_path, summary_path, gate["passed"], gate)
        review_req = {"request_id": marker, "experiment_id": STAMP, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "question": "Review v33 real-loader terminal identity/evidence audit. If loader identity and schedule/trace assertions suffice despite no TF/mpc.vf_fun evaluation, confirm whether to proceed to the 24 fixed-context solver-call objective-vs-basin probe after backup.", "evidence_paths": [rel(summary_path), rel(raw_path), rel(RUN_DIR / "completed.json"), rel(coeff_path), rel(BACKUP_REQUEST), rel(RESPONSE_LOG)], "access_budget": {"rollout_episodes": 0, "plant_steps": 0, "solver_calls": 0, "gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "gate": gate, "limitations": raw["limitations"]}
        write_json(NEXT_REVIEW_REQUEST, review_req)
        STATE.write_text(f"# Continue state after v33 terminal identity/evidence audit\n\nUTC: {raw['created_utc']}\n\nGate: {gate}\n\nArtifacts: {rel(summary_path)}, {rel(raw_path)}, {rel(RUN_DIR / 'completed.json')}\n\nBackup request: {rel(BACKUP_REQUEST)}\n\nNext: wait/read Astra analysis for request {marker}; after verified backup and absent contrary Astra evidence, implement the pre-authorized 24 fixed-context solver-call objective-vs-basin probe.\n", encoding="utf-8")
        completed = {"status": "complete", "passed": bool(gate["passed"]), "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "classification": raw["classification"], "artifact_dir": rel(RUN_DIR), "summary": rel(summary_path), "raw": rel(raw_path), "coefficient_table_csv": rel(coeff_path), "backup_request": rel(BACKUP_REQUEST), "next_review_request": rel(NEXT_REVIEW_REQUEST), "budgets": gate, "hashes": {}}
        files = [Path(__file__).resolve(), raw_path, summary_path, coeff_path, BACKUP_REQUEST, RESPONSE_LOG, NEXT_REVIEW_REQUEST, STATE]
        completed["hashes"] = {rel(p): sha256(p) for p in files if p.exists()}
        completed_path = RUN_DIR / "completed.json"
        write_json(completed_path, completed)
        print(json.dumps({"passed": gate["passed"], "artifact_dir": rel(RUN_DIR), "summary": rel(summary_path), "raw": rel(raw_path), "completed": rel(completed_path), "backup_request": rel(BACKUP_REQUEST), "loader_identity_passed": gate["loader_identity_passed"], "schedule_trace_count_gate": gate["schedule_trace_count_gate"], "max_loader_file_diff": gate["max_loader_file_diff"], "theta_wrap_V35_max": theta_stats.get("V35", {}).get("max_abs_delta_vs_base"), "solver_calls": 0, "plant_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        write_json(RUN_DIR / "failed.json", {"status": "failed", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "solver_calls": 0, "plant_steps": 0})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "solver_calls": 0, "plant_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
