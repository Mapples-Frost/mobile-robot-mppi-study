#!/usr/bin/env python3
"""v33 terminal identity/evaluator evidence audit v0b.

Astra A11/A12 follow-up after the repeated narrow contract audit.  This is a
zero-rollout, zero-solver, zero-training development diagnostic.  It closes the
real terminal evidence chain as far as available local artifacts permit:

* actual v33 loader path (SAC.load -> weights_hash -> get_mpc_vfn_weights...)
  vs exported terminal_cnnvf_*.npy, using flatten(order='F');
* independently loaded stable-baselines TF checkpoint evaluator value/gradient;
* real gym_horizon casadiNNVF value and actual deployed mpc.vf_fun value;
* actual MPC _x/_p labels, excluding n_horizon from terminal parameters;
* v33 72-cell schedule/trace-line/4013-step assertions and fixed-H terminal
  safety/quality/time table.

No env.reset(), no env.step(), no MPC solve(), no validation64/test access.
"""
from __future__ import annotations

import argparse, csv, datetime as dt, hashlib, json, math, os, sqlite3, sys, traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
REPRO = ROOT / "experiments/bohn2021_reproduction"
for p in (AWS_DIR, REPRO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

NAME = "vehicle_true_variable_horizon_v33_terminal_identity_evidence_audit_v0b"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / "research_artifacts/aws_diagnostics" / f"{NAME}_{STAMP}"
BACKUP_REQ = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V33_TERMINAL_IDENTITY_EVIDENCE_AUDIT_V0B_{STAMP}.json"
STATE = ROOT / "research_artifacts/aws_state" / f"continue_state_{STAMP}_after_v33_terminal_identity_evidence_audit_v0b.md"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ROOT / "docs/bohn2021_takeover/astra_reviews/ANALYSIS_READY.json"
REPORT_064906 = ROOT / "docs/bohn2021_takeover/astra_reviews/20260930T064906Z.md"
REPORT_063724 = ROOT / "docs/bohn2021_takeover/astra_reviews/20260930T063724Z.md"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
V33_GLOB = "vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0_*"
TARGET_LABELS = ["v27_case09_slot0_early_risk","v27_case09_slot1_mid_late_risk","v19_c12","v19_c13","v27_case00_slot1_mid_late_risk","v27_case08_slot1_mid_late_risk"]
TERMINAL_HS = [15, 35]
THETA_DELTAS = [0.0, 2.0 * math.pi, -2.0 * math.pi]
TOL_TENSOR = 1e-12
TOL_REL = 1e-5


def rel(p: Path) -> str:
    try: return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception: return str(p)

def clean(x: Any) -> Any:
    if isinstance(x, Path): return rel(x)
    if isinstance(x, (dt.datetime, dt.date)): return x.isoformat()
    if isinstance(x, float): return x if math.isfinite(x) else None
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if hasattr(x, "item"):
        try: return clean(x.item())
        except Exception: pass
    if isinstance(x, Mapping): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)): return [clean(v) for v in x]
    return x

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f: return json.load(f)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def bytes_hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()

def flatten_any(obj: Any) -> np.ndarray:
    if isinstance(obj, (list, tuple)):
        return np.concatenate([np.asarray(x, dtype=np.float64).ravel(order="F") for x in obj]) if obj else np.asarray([], dtype=np.float64)
    return np.asarray(obj, dtype=np.float64).ravel(order="F")

def maxdiff(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.shape != b.shape: return None
    return float(np.max(np.abs(a-b))) if a.size else 0.0

def parse_label(label: Any) -> str:
    s = str(label).strip().strip("[]")
    return s.split(",")[0].strip().strip("'").strip('"')

def latest_v33_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit); p = p if p.is_absolute() else ROOT / p
        if not p.exists(): raise RuntimeError(f"missing explicit v33 dir {p}")
        return p
    dirs = sorted((ROOT / "research_artifacts/aws_diagnostics").glob(V33_GLOB), key=lambda p: p.name)
    if not dirs: raise RuntimeError("no v33 diagnostic directory found")
    return dirs[-1]

def file_receipt(p: Path) -> Dict[str, Any]:
    return {"path": rel(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None, "sha256": sha256(p) if p.exists() else None}

def sqlite_usage() -> Dict[str, Any]:
    candidates = [ROOT / "research.sqlite", ROOT / "research_state.sqlite", ROOT.parent / "research.sqlite"]
    for db in candidates:
        if not db.exists(): continue
        try:
            con = sqlite3.connect(str(db)); cur = con.cursor()
            tables = [r[0] for r in cur.execute("select name from sqlite_master where type='table'")]
            sums = []
            for t in tables:
                cols = [r[1] for r in cur.execute(f"pragma table_info('{t}')")]
                if "total_tokens" in cols:
                    val = cur.execute(f"select coalesce(sum(total_tokens),0) from '{t}'").fetchone()[0]
                    sums.append({"table": t, "sum_total_tokens": int(val or 0)})
            con.close()
            return {"path": rel(db), "tables_with_total_tokens": sums, "best_effort_total_tokens_sum": int(sum(x["sum_total_tokens"] for x in sums)) if sums else None}
        except Exception as exc:
            return {"path": rel(db), "error": repr(exc), "best_effort_total_tokens_sum": None}
    return {"path": None, "best_effort_total_tokens_sum": None}

def import_stage_loader() -> Tuple[Any, Any, Any]:
    import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # type: ignore
    return v1d.import_legacy_modules()

def load_terminals(stage1_runner: Any) -> Tuple[Mapping[int, Any], Mapping[str, Any], Dict[str, Any]]:
    proto_path = Path(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    proto = read_json(proto_path)
    terminals, receipts = stage1_runner.load_terminal_grid(proto["terminal_grid_readiness_reused_from_v1"])
    return terminals, receipts, {"path": rel(proto_path), "sha256": sha256(proto_path)}

def extract_branch_states(v33_dir: Path) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for br in sorted((v33_dir / "episodes").glob("*/branch_reset.json")):
        data = read_json(br)
        state = data.get("branch_state_target") or data.get("branch_state_after_direct_reset")
        if not isinstance(state, Mapping): continue
        ep_name = br.parent.name
        label = next((x for x in TARGET_LABELS if x in ep_name), ep_name)
        out.setdefault(label, {"state_label": label, "state": dict(state), "branch_reset_path": rel(br), "sample_episode": ep_name, "initial_observation_at_branch": data.get("initial_observation_at_branch") or []})
    return [out[k] for k in sorted(out)]

def terminal_files(receipt: Mapping[str, Any]) -> Tuple[Path, Path, Path]:
    folder = ROOT / str(receipt.get("folder"))
    return folder, folder / "terminal_cnnvf_weights.npy", folder / "terminal_cnnvf_biases.npy"

def coeff_rows_from_flat(term: str, w: np.ndarray, input_names: Sequence[str]) -> List[Dict[str, Any]]:
    n = len(input_names); rows = []
    if w.size >= 2*n:
        for i, name in enumerate(input_names):
            rows.append({"terminal": term, "input": name, "linear_coeff": float(w[i]), "quadratic_coeff": float(w[n+i]), "second_derivative": float(2*w[n+i])})
    return rows

def np_poly(w: np.ndarray, b: np.ndarray, z: Sequence[float], nx: int) -> Tuple[float, List[float]]:
    z = np.asarray(z, dtype=np.float64).ravel(); n = z.size
    lin = w[:n]; quad = w[n:2*n]
    val = float((b[0] if b.size else 0.0) + np.dot(lin, z) + np.dot(quad, z*z))
    grad = (lin[:nx] + 2.0 * quad[:nx] * z[:nx]).astype(float).tolist()
    return val, grad

def relerr(a: float, b: float) -> float:
    return abs(a-b)/max(1.0, abs(a), abs(b))

def grad_relerr(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=float); bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape: return float("inf")
    return float(np.max(np.abs(aa-bb))/max(1.0, float(np.max(np.abs(aa))), float(np.max(np.abs(bb)))))

def fd_grad(fn, x: Sequence[float], p: Sequence[float], eps: float = 1e-6) -> List[float]:
    x = np.asarray(x, dtype=float).ravel(); out = []
    for i in range(x.size):
        xp = x.copy(); xm = x.copy(); xp[i] += eps; xm[i] -= eps
        out.append(float((fn(xp, p) - fn(xm, p))/(2*eps)))
    return out

def make_samples(branch_states: Sequence[Mapping[str, Any]], x_names: Sequence[str], p_names: Sequence[str]) -> List[Dict[str, Any]]:
    samples = []
    for bs in branch_states:
        st = bs["state"]
        base_x = [float(st[n]) for n in x_names]
        contexts = {"zero_context_reconstructed_not_original": [0.0]*len(p_names)}
        obs = bs.get("initial_observation_at_branch") or []
        if len(p_names) == 2 and isinstance(obs, list) and len(obs) >= 5:
            contexts["observation_slots_3_4_heuristic_not_original"] = [float(obs[3]), float(obs[4])]
        for ctx, pvec in contexts.items():
            for dth in THETA_DELTAS:
                x = list(base_x)
                if "theta" in x_names: x[list(x_names).index("theta")] += dth
                samples.append({"state_label": bs["state_label"], "context": ctx, "theta_delta": dth, "x": x, "p": pvec})
    return samples

def tf_eval_for_h(h: int, receipt: Mapping[str, Any], samples: Sequence[Mapping[str, Any]], nx: int) -> Dict[str, Any]:
    from runtime import imports  # type: ignore
    from run import weights_hash  # type: ignore
    import tensorflow as tf  # type: ignore
    _, SAC, _ = imports()
    folder, _wpath, _bpath = terminal_files(receipt)
    model_zip, completed_path, manifest_path = folder / "model.zip", folder / "completed.json", folder / "manifest.json"
    done = read_json(completed_path); manifest = read_json(manifest_path)
    model = SAC.load(str(model_zip))
    rows: List[Dict[str, Any]] = []
    try:
        wh = weights_hash(model)
        weights, biases = model.policy_tf.get_mpc_vfn_weights_and_biases()
        wf, bf = flatten_any(weights), flatten_any(biases)
        value_t = getattr(model.policy_tf, "mpc_value_fn", None)
        ph = getattr(model.policy_tf, "mpc_state_ph", None)
        if value_t is None or ph is None:
            raise RuntimeError("policy_tf lacks mpc_value_fn or mpc_state_ph")
        grad_t = tf.gradients(value_t, ph)[0]
        for s in samples:
            z = np.asarray(list(s["x"]) + list(s["p"]), dtype=np.float32).reshape(1, -1)
            val, grad = model.sess.run([value_t, grad_t], {ph: z})
            rows.append({"state_label": s["state_label"], "context": s["context"], "theta_delta": s["theta_delta"], "value": float(np.asarray(val).reshape(-1)[0]), "grad_state": np.asarray(grad, dtype=float).reshape(-1)[:nx].tolist()})
        return {"ok": True, "weights_hash_loaded_model": wh, "completed_final_hash": done.get("final_hash"), "manifest": {"task": manifest.get("task"), "fixed_horizon": manifest.get("fixed_horizon"), "seed": manifest.get("seed"), "steps": manifest.get("steps")}, "weight_flat": wf.tolist(), "bias_flat": bf.tolist(), "weight_numeric_sha256": bytes_hash(wf), "bias_numeric_sha256": bytes_hash(bf), "eval_rows": rows}
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
    finally:
        try: model.sess.close()
        except Exception: pass

def casadi_direct_eval(term_pair: Any, samples: Sequence[Mapping[str, Any]], nx: int) -> Dict[str, Any]:
    import casadi as ca  # type: ignore
    from gym_let_mpc.utils import casadiNNVF  # type: ignore
    np0 = len(samples[0]["p"]) if samples else 0
    vf = casadiNNVF(layers=[], type="poly")
    xs = ca.SX.sym("state", nx); ps = ca.SX.sym("parameters", np0)
    vf.create_function(xs, ps); vf.set_weights_and_biases(*term_pair)
    def f(x, p): return float(vf.eval_VF(np.asarray(x).reshape(nx,1), np.asarray(p).reshape(np0,1), vf.weights_num, vf.biases_num))
    rows=[]
    for s in samples:
        rows.append({"state_label": s["state_label"], "context": s["context"], "theta_delta": s["theta_delta"], "value": f(s["x"], s["p"]), "grad_state_fd": fd_grad(f, s["x"], s["p"])})
    return {"ok": True, "weights_num_shape": list(np.asarray(vf.weights_num).shape), "biases_num_shape": list(np.asarray(vf.biases_num).shape), "eval_rows": rows}

def mpc_vf_eval_all(terminals: Mapping[int, Any], samples: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from runtime import make_env  # type: ignore
    env = make_env("vehicle", 0, aligned=True, scaled_obs=True)
    mpc = env.control_system.controller.mpc
    x_labels_raw = list(mpc.model._x.labels()); p_labels_raw = list(mpc.model._p.labels())
    x_names = [parse_label(x) for x in x_labels_raw]
    p_names_all = [parse_label(x) for x in p_labels_raw]
    p_keep = [i for i, n in enumerate(p_names_all) if "n_horizon" not in n]
    p_names = [p_names_all[i] for i in p_keep]
    meta = {"x_labels_raw": x_labels_raw, "x_names": x_names, "p_labels_raw": p_labels_raw, "p_names_all": p_names_all, "p_keep_indices_excluding_n_horizon": p_keep, "p_names_excluding_n_horizon": p_names, "use_nn_vf": bool(getattr(mpc, "use_nn_vf", False)), "env_constructed_no_reset_no_step": True}
    out: Dict[str, Any] = {}
    for h in TERMINAL_HS:
        env.set_value_function_weights_and_biases(*terminals[h])
        def f(x, p):
            return float(mpc.vf_fun(np.asarray(x, dtype=float).reshape(len(x_names),1), np.asarray(p, dtype=float).reshape(len(p_names),1), mpc.vf.weights_num, mpc.vf.biases_num))
        rows=[]
        for s in samples:
            rows.append({"state_label": s["state_label"], "context": s["context"], "theta_delta": s["theta_delta"], "value": f(s["x"], s["p"]), "grad_state_fd": fd_grad(f, s["x"], s["p"])})
        out[f"V{h}"] = {"ok": True, "eval_rows": rows, "vf_weights_num_shape": list(np.asarray(mpc.vf.weights_num).shape), "vf_biases_num_shape": list(np.asarray(mpc.vf.biases_num).shape)}
    return meta, out

def verify_v33(v33_dir: Path) -> Dict[str, Any]:
    raw_path = v33_dir / "raw.json"; done_path = v33_dir / "completed.json"
    raw = read_json(raw_path); episodes = raw.get("episodes") or []; schedule = raw.get("schedule") or []
    traces = sorted((v33_dir/"episodes").glob("*/trace.jsonl")); line_counts={}
    for t in traces:
        with t.open("r", encoding="utf-8") as f: line_counts[rel(t)] = sum(1 for line in f if line.strip())
    cells = {(str(r.get("state_label")), int(r.get("horizon")), str(r.get("terminal_mode"))) for r in schedule}
    expected = {(s,h,m) for s in TARGET_LABELS for h in [12,15,25,35] for m in ["zero","V15_shared","V35_shared"]}
    by = {}
    for e in episodes:
        key = f"H{int(e.get('horizon',-1))}|{e.get('terminal_mode')}"; by.setdefault(key, []).append(e)
    table = {}
    for key, rows in sorted(by.items()):
        def val(e: Mapping[str,Any], *names: str) -> float:
            for n in names:
                x=e.get(n)
                if isinstance(x, Mapping) and x.get("sum") is not None: return float(x.get("sum"))
                try:
                    if x is not None: return float(x)
                except Exception: pass
            return 0.0
        table[key] = {"episodes": len(rows), "safe_success_no_solver_fail_count": sum(1 for e in rows if bool(e.get("safe_success_no_solver_fail"))), "success_count": sum(1 for e in rows if bool(e.get("success"))), "step_sum": int(sum(int(e.get("steps",0)) for e in rows)), "physical_sum": float(math.fsum(val(e,"physical_constraint_cost") for e in rows)), "decision_sum_s": float(math.fsum(val(e,"decision_sum_s","decision_timing_s") for e in rows)), "solver_sum_s": float(math.fsum(val(e,"solver_sum_s","solver_attempt_timing_s") for e in rows))}
    trace_total = int(sum(line_counts.values())); budget_steps = int((raw.get("budget_actual") or {}).get("control_steps", -1))
    return {"v33_dir": rel(v33_dir), "raw_receipt": file_receipt(raw_path), "completed_receipt": file_receipt(done_path), "summary_receipt": file_receipt(v33_dir/"summary.md"), "schedule_rows": len(schedule), "episode_rows_in_raw": len(episodes), "trace_files": len(traces), "trace_line_total": trace_total, "budget_control_steps_recorded": budget_steps, "schedule_unique_cells": len(cells), "expected_cells": len(expected), "missing_cells": sorted([f"{a}|H{b}|{c}" for a,b,c in expected-cells]), "extra_cells": sorted([f"{a}|H{b}|{c}" for a,b,c in cells-expected]), "schedule_trace_count_gate": bool(len(schedule)==72 and len(episodes)==72 and len(traces)==72 and trace_total==budget_steps==4013 and len(cells)==72 and not (expected-cells) and not (cells-expected)), "fixed_H_terminal_table": table, "aggregate_from_raw": (raw.get("analysis") or {}).get("aggregate")}

def write_coeff_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["terminal","input","linear_coeff","quadratic_coeff","second_derivative"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in fields})

def append_response(marker: str, raw_path: Path, summary_path: Path, gate: Mapping[str, Any]) -> None:
    old = RESPONSE_LOG.read_text(encoding="utf-8", errors="replace") if RESPONSE_LOG.exists() else ""
    if marker in old: return
    block = f"""
## Follow-up through extended v33 terminal identity/evaluator audit (`{marker}`)

Updated by GPT-5.5 executor at `{dt.datetime.now(dt.timezone.utc).isoformat()}`. Zero rollout/solver/training/refit: plant steps=0, solver calls=0, validation64=0, sealed test=0.

| linked recommendation(s) | disposition after audit | verified evidence | next step |
|---|---|---|---|
| `A11_training_failure_modes_need_separation` | accepted; real loader/export/TF/casadi/mpc evaluator identity tested | `{rel(raw_path)}`, `{rel(summary_path)}`; evaluator_identity_passed={gate.get('evaluator_identity_passed')}; max_value_relerr={gate.get('max_value_relerr')}; max_grad_relerr={gate.get('max_grad_relerr')} | If backup is verified and Astra raises no contrary evidence, execute the fixed 24-call objective-vs-basin solver probe. |
| `A12_registry_backup_schema_contract` | accepted; v33 72-cell/trace/step assertions retested | schedule_trace_count_gate={gate.get('schedule_trace_count_gate')} | Backup request written; no further unique science before external backup verification. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; fixed-H terminal table retained | coefficient table and H/terminal table in artifacts | Preserve H35/zero and H35/V15 comparators; do not learn from old pure-H labels. |
"""
    RESPONSE_LOG.parent.mkdir(parents=True, exist_ok=True)
    RESPONSE_LOG.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")

def write_summary(raw: Mapping[str, Any]) -> None:
    gate = raw["gate"]
    lines = [f"# {NAME}", "", f"created_utc: `{raw['created_utc']}`", f"v33_dir: `{raw['v33_evidence']['v33_dir']}`", "", "## Gate", f"- passed: `{gate['passed']}`", f"- loader_export_identity_passed: `{gate['loader_export_identity_passed']}`", f"- tf_loader_identity_passed: `{gate['tf_loader_identity_passed']}`", f"- evaluator_identity_passed: `{gate['evaluator_identity_passed']}`", f"- schedule_trace_count_gate: `{gate['schedule_trace_count_gate']}`", f"- max_value_relerr: `{gate['max_value_relerr']}`", f"- max_grad_relerr: `{gate['max_grad_relerr']}`", "", "## Terminal identities"]
    for h, info in raw["terminal_identity"].items():
        lines.append(f"- {h}: loader_vs_export_F={info.get('max_abs_diff_loader_vs_file_order_F')}; tf_vs_loader={info.get('max_abs_diff_tf_vs_loader')}; theta_second_derivative={info.get('theta_second_derivative')}")
    lines += ["", "## v33 fixed-H terminal table"]
    for key, row in raw["v33_evidence"]["fixed_H_terminal_table"].items():
        lines.append(f"- {key}: safe={row['safe_success_no_solver_fail_count']}/{row['episodes']} physical_sum={row['physical_sum']:.6g} decision_sum_s={row['decision_sum_s']:.6g}")
    lines += ["", "## Limitations"] + [f"- {x}" for x in raw["limitations"]]
    lines += ["", f"Coefficient table: `{raw['coefficient_table_csv']}`", f"Backup request: `{raw['backup_request_path']}`"]
    (RUN_DIR/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

def main(argv: Optional[Sequence[str]]=None) -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--v33-dir", default=None); ap.add_argument("--backup-time", default=None); ap.add_argument("--backup-commit", default=None); ap.add_argument("--backup-package-sha256", default=None); ap.add_argument("--i-accept-zero-rollout-terminal-identity-audit-v0b", action="store_true", required=True)
    args=ap.parse_args(argv); RUN_DIR.mkdir(parents=True, exist_ok=False)
    try:
        started = dt.datetime.now(dt.timezone.utc); usage = sqlite_usage(); v33_dir=latest_v33_dir(args.v33_dir)
        _base, stage1, _fixed = import_stage_loader(); terminals, receipts, terminal_protocol = load_terminals(stage1)
        branch_states = extract_branch_states(v33_dir)
        mpc_meta_probe, _ = mpc_vf_eval_all(terminals, [])
        x_names = mpc_meta_probe["x_names"]; p_names = mpc_meta_probe["p_names_excluding_n_horizon"]; input_names = list(x_names)+list(p_names)
        samples = make_samples(branch_states, x_names, p_names)
        mpc_meta, mpc_rows_by_h = mpc_vf_eval_all(terminals, samples)
        terminal_identity: Dict[str, Any] = {}; coeff_rows: List[Dict[str, Any]]=[]; all_value_err=[]; all_grad_err=[]; evaluator_rows: List[Dict[str, Any]]=[]
        for h in TERMINAL_HS:
            key=f"V{h}"; receipt = receipts[str(h)] if str(h) in receipts else receipts[h]
            folder,wpath,bpath = terminal_files(receipt); file_w=np.asarray(np.load(str(wpath)), dtype=np.float64); file_b=np.asarray(np.load(str(bpath)), dtype=np.float64)
            loader_w=flatten_any(terminals[h][0]); loader_b=flatten_any(terminals[h][1]); file_wf=file_w.ravel(order="F"); file_bf=file_b.ravel(order="F")
            tfres = tf_eval_for_h(h, receipt, samples, len(x_names)); casres = casadi_direct_eval(terminals[h], samples, len(x_names))
            tfw = np.asarray(tfres.get("weight_flat", []), dtype=float); tfb=np.asarray(tfres.get("bias_flat", []), dtype=float)
            coeff_rows.extend(coeff_rows_from_flat(key, file_wf, input_names))
            terminal_identity[key] = {"folder": rel(folder), "model_zip_sha256": receipt.get("model_zip_sha256"), "weights_hash_receipt": receipt.get("weights_hash"), "file_weight_sha256": sha256(wpath), "file_bias_sha256": sha256(bpath), "loader_weight_numeric_sha256": bytes_hash(loader_w), "tf_weight_numeric_sha256": tfres.get("weight_numeric_sha256"), "max_abs_diff_loader_vs_file_order_F": maxdiff(loader_w,file_wf), "max_abs_diff_loader_bias_vs_file_order_F": maxdiff(loader_b,file_bf), "max_abs_diff_tf_vs_loader": maxdiff(tfw, loader_w) if tfres.get("ok") else None, "max_abs_diff_tf_bias_vs_loader": maxdiff(tfb, loader_b) if tfres.get("ok") else None, "tf_eval_ok": bool(tfres.get("ok")), "tf_eval_error": tfres.get("error"), "tf_weights_hash_loaded_model": tfres.get("weights_hash_loaded_model"), "tf_completed_final_hash": tfres.get("completed_final_hash"), "theta_second_derivative": float(2*file_wf[len(input_names)]) if file_wf.size >= 2*len(input_names) and "theta" in input_names else None}
            tf_by = {(r["state_label"],r["context"],float(r["theta_delta"])): r for r in tfres.get("eval_rows", [])}
            ca_by = {(r["state_label"],r["context"],float(r["theta_delta"])): r for r in casres.get("eval_rows", [])}
            mp_by = {(r["state_label"],r["context"],float(r["theta_delta"])): r for r in mpc_rows_by_h[key].get("eval_rows", [])}
            for s in samples:
                sk=(s["state_label"],s["context"],float(s["theta_delta"])); nv,ng=np_poly(file_wf,file_bf,list(s["x"])+list(s["p"]),len(x_names)); tr=tf_by.get(sk); cr=ca_by.get(sk); mr=mp_by.get(sk)
                row={"terminal":key, "state_label":s["state_label"], "context":s["context"], "theta_delta":s["theta_delta"], "numpy_value":nv, "numpy_grad_state":ng, "tf_value": None if tr is None else tr["value"], "tf_grad_state": None if tr is None else tr["grad_state"], "casadiNNVF_value": None if cr is None else cr["value"], "casadiNNVF_grad_state_fd": None if cr is None else cr["grad_state_fd"], "mpc_vf_fun_value": None if mr is None else mr["value"], "mpc_vf_fun_grad_state_fd": None if mr is None else mr["grad_state_fd"]}
                vals=[row[k] for k in ("tf_value","casadiNNVF_value","mpc_vf_fun_value") if row[k] is not None]
                grads=[row[k] for k in ("tf_grad_state","casadiNNVF_grad_state_fd","mpc_vf_fun_grad_state_fd") if row[k] is not None]
                for v in vals: all_value_err.append(relerr(float(v), nv))
                for g in grads: all_grad_err.append(grad_relerr(g, ng))
                evaluator_rows.append(row)
        coeff_path=RUN_DIR/"terminal_coefficients.csv"; write_coeff_csv(coeff_path, coeff_rows)
        v33_ev=verify_v33(v33_dir)
        gate={"loader_export_identity_passed": all((terminal_identity[f"V{h}"]["max_abs_diff_loader_vs_file_order_F"] is not None and terminal_identity[f"V{h}"]["max_abs_diff_loader_vs_file_order_F"] <= TOL_TENSOR and terminal_identity[f"V{h}"]["max_abs_diff_loader_bias_vs_file_order_F"] is not None and terminal_identity[f"V{h}"]["max_abs_diff_loader_bias_vs_file_order_F"] <= TOL_TENSOR) for h in TERMINAL_HS), "tf_loader_identity_passed": all(bool(terminal_identity[f"V{h}"]["tf_eval_ok"]) and terminal_identity[f"V{h}"]["max_abs_diff_tf_vs_loader"] is not None and terminal_identity[f"V{h}"]["max_abs_diff_tf_vs_loader"] <= TOL_TENSOR and terminal_identity[f"V{h}"]["max_abs_diff_tf_bias_vs_loader"] is not None and terminal_identity[f"V{h}"]["max_abs_diff_tf_bias_vs_loader"] <= TOL_TENSOR for h in TERMINAL_HS), "evaluator_identity_passed": bool(all_value_err and all_grad_err and max(all_value_err) <= TOL_REL and max(all_grad_err) <= TOL_REL), "max_value_relerr": max(all_value_err) if all_value_err else None, "max_grad_relerr": max(all_grad_err) if all_grad_err else None, "schedule_trace_count_gate": bool(v33_ev["schedule_trace_count_gate"]), "rollout_episodes":0,"plant_steps":0,"solver_calls":0,"gradient_steps":0,"selector_refits":0,"validation64_episodes":0,"sealed_test_episodes":0}
        gate["passed"] = bool(gate["loader_export_identity_passed"] and gate["tf_loader_identity_passed"] and gate["evaluator_identity_passed"] and gate["schedule_trace_count_gate"])
        raw={"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "started_utc": started.isoformat(), "elapsed_since_first_supervisor_event_seconds": (dt.datetime.now(dt.timezone.utc)-FIRST_EVENT).total_seconds(), "sqlite_usage_best_effort": usage, "method": NAME, "classification": "development_IMPROVED_zero_rollout_real_terminal_identity_evaluator_audit_not_validation_not_test", "bound_astra_reports": {"analysis_ready": file_receipt(ANALYSIS_READY), "report_064906": file_receipt(REPORT_064906), "report_063724": file_receipt(REPORT_063724)}, "pre_run_backup_context": {"time": args.backup_time, "commit": args.backup_commit, "package_sha256": args.backup_package_sha256}, "terminal_source_protocol": terminal_protocol, "mpc_mapping": mpc_meta, "branch_states": branch_states, "sample_count": len(samples), "terminal_identity": terminal_identity, "coefficient_table_csv": rel(coeff_path), "evaluator_rows": evaluator_rows, "v33_evidence": v33_ev, "gate": gate, "budget_actual": gate, "limitations": ["opened v33 development states only; validation64 and sealed test not accessed", "v33 traces do not preserve original deployed opt_p goal values at branch; zero/observation-slot contexts are reconstructed and explicitly not original", "mpc.vf_fun was evaluated without env.reset/env.step/solve; this tests deployed evaluator function identity, not closed-loop causality", "finite-difference gradients used for CasADi/mpc.vf_fun state gradients", "no objective decomposition or alternate-initialization solver calls in this audit"], "backup_request_path": rel(BACKUP_REQ)}
        raw_path=RUN_DIR/"raw.json"; summary_path=RUN_DIR/"summary.md"; write_json(raw_path, raw); write_summary(raw)
        write_json(BACKUP_REQ, {"request":"backup_after_v33_terminal_identity_evidence_audit_v0b", "created_utc": raw["created_utc"], "backup_required_before_solver_call_probe_or_other_unique_science": True, "must_cover":[rel(Path(__file__).resolve()), rel(RUN_DIR), rel(BACKUP_REQ), rel(RESPONSE_LOG), rel(NEXT_REVIEW_REQUEST), rel(STATE)], "budget_actual": gate, "validation64_bank_opened": False, "sealed_test_accessed": False})
        marker=f"v33-terminal-identity-evaluator-audit-v0b-{STAMP}"; append_response(marker, raw_path, summary_path, gate)
        review={"request_id": marker, "experiment_id": STAMP, "created_utc": raw["created_utc"], "question":"Review extended v33 terminal identity/evaluator audit. If real loader/export/TF/casadi/mpc evaluator identity and v33 evidence assertions pass, confirm direct move after backup to the already specified 24-call objective-vs-basin solver probe; if not, identify the specific implementation defect.", "evidence_paths":[rel(summary_path), rel(raw_path), rel(RUN_DIR/"completed.json"), rel(coeff_path), rel(BACKUP_REQ), rel(RESPONSE_LOG)], "access_budget": gate, "gate": gate}
        write_json(NEXT_REVIEW_REQUEST, review)
        STATE.write_text(f"# Continue state after v33 terminal identity/evaluator audit v0b\n\nUTC: {raw['created_utc']}\nGate: {gate}\nArtifacts: {rel(summary_path)}, {rel(raw_path)}, {rel(RUN_DIR/'completed.json')}\nBackup request: {rel(BACKUP_REQ)}\nNext: verify external backup; check Astra response for {marker}; if gate remains accepted, execute fixed 24-call objective-vs-basin solver probe.\n", encoding="utf-8")
        completed={"status":"complete", "passed": bool(gate["passed"]), "created_utc": raw["created_utc"], "classification": raw["classification"], "artifact_dir": rel(RUN_DIR), "summary": rel(summary_path), "raw": rel(raw_path), "coefficient_table_csv": rel(coeff_path), "backup_request": rel(BACKUP_REQ), "next_review_request": rel(NEXT_REVIEW_REQUEST), "budgets": gate, "hashes": {}}
        files=[Path(__file__).resolve(), raw_path, summary_path, coeff_path, BACKUP_REQ, RESPONSE_LOG, NEXT_REVIEW_REQUEST, STATE]
        completed["hashes"]={rel(p):sha256(p) for p in files if p.exists()}; write_json(RUN_DIR/"completed.json", completed)
        print(json.dumps({"completed": rel(RUN_DIR/"completed.json"), "summary": rel(summary_path), "raw": rel(raw_path), "passed": gate["passed"], "max_value_relerr": gate["max_value_relerr"], "max_grad_relerr": gate["max_grad_relerr"], "sqlite_total_tokens_best_effort": usage.get("best_effort_total_tokens_sum"), "elapsed_since_first_supervisor_event_seconds": raw["elapsed_since_first_supervisor_event_seconds"], "backup_request": rel(BACKUP_REQ), "validation64_bank_opened": False, "sealed_test_accessed": False, "solver_calls":0, "plant_steps":0}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        write_json(RUN_DIR/"failed.json", {"status":"failed", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "solver_calls":0, "plant_steps":0})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR/"failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False, "solver_calls":0, "plant_steps":0}, sort_keys=True), flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
