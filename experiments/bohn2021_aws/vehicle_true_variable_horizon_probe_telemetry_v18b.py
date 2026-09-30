#!/usr/bin/env python3
"""v18b repaired H10 probe-telemetry risk/value diagnostic.

This is a source-preserving repair wrapper around the unexecuted v18 diagnostic.
Repairs before first execution:
  * charge probe solver overhead with first-solver time rather than first-decision time;
  * carry separate extra_probe_decision_s and extra_probe_solver_s through nested aggregation;
  * declare an offline model-evaluation cap consistent with the planned full grid;
  * avoid appending a malformed extra row to EXPERIMENT_REGISTRY.csv (run_experiment
    records the canonical registry row).

Development-only IMPROVED diagnostic: uses existing v15 branch traces, performs no
new MPC simulation, no validation64 access, no sealed-test access, and no gradient
training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_probe_telemetry_v18.py"
spec = importlib.util.spec_from_file_location("vehicle_true_variable_horizon_probe_telemetry_v18_base", str(BASE_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import v18 base")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)  # type: ignore[union-attr]

NAME = "vehicle_true_variable_horizon_probe_telemetry_v18b"
STAMP = "20260930T0015Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / "research_artifacts/aws_state/continue_state_20260930T0015_after_probe_telemetry_v18b.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_PROBE_TELEMETRY_V18B_{STAMP}.json"
SOURCE = Path(__file__).resolve()
MARKER = f"vehicle-true-variable-H-probe-telemetry-v18b-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

# Repoint imported-base globals so reused reporting/freezing helpers write v18b artifacts.
base.NAME = NAME
base.STAMP = STAMP
base.OUT = OUT
base.PROTOCOL = PROTOCOL
base.STATE = STATE
base.BACKUP_REQUEST = BACKUP_REQUEST
base.SOURCE = SOURCE
base.MARKER = MARKER
base.FIRST_EVENT = FIRST_EVENT


def rel(p: Path) -> str:
    return base.rel(p)


def sf(x: Any, default: float = 0.0) -> float:
    return base.sf(x, default)


def si(x: Any, default: int = 0) -> int:
    return base.si(x, default)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows_v18b() -> Tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, Any], dict[str, str]]:
    rows, fdicts, diag, hashes = base.load_rows()
    hashes[rel(BASE_PATH)] = sha256(BASE_PATH)
    hashes[rel(SOURCE)] = sha256(SOURCE)
    for r in rows:
        fd = fdicts[base.row_key(r)]
        r["h10_first_solver_s"] = sf(fd.get("h10_first_solver_s"))
        r["h15_first_solver_s"] = sf(fd.get("h15_first_solver_s"))
    diag = dict(diag)
    diag["v18b_repair"] = "probe solver overhead charged with first-solver time; nested aggregation carries decision/solver overhead separately"
    return rows, fdicts, diag, hashes


def evaluate_v18b(rows: Sequence[Mapping[str, Any]], choices: Mapping[str, int], overhead_mode: str) -> Dict[str, Any]:
    fixed_phys = math.fsum(sf(r["h15_physical"]) for r in rows)
    fixed_dec = math.fsum(sf(r["h15_decision_sum_s"]) for r in rows)
    fixed_sol = math.fsum(sf(r["h15_solver_sum_s"]) for r in rows)
    tol_sum = math.fsum(sf(r["row_physical_tolerance"]) for r in rows)
    pol_phys = pol_dec = pol_sol = 0.0
    counts: Counter[str] = Counter()
    conf: Counter[str] = Counter()
    bad = []
    details = []
    extra_probe_dec = 0.0
    extra_probe_sol = 0.0
    for r in rows:
        key = base.row_key(r)
        h = int(choices.get(key, 15))
        counts[str(h)] += 1
        pos = bool(r["h10_beneficial_vs_h15"])
        cat = bool(r["h10_catastrophic_vs_h15"])
        extra_d = 0.0
        extra_s = 0.0
        if overhead_mode == "h10_probe" and h == 15:
            extra_d += sf(r.get("h10_first_decision_s"))
            extra_s += sf(r.get("h10_first_solver_s"))
        elif overhead_mode == "dual_probe":
            if h == 10:
                extra_d += sf(r.get("h15_first_decision_s"))
                extra_s += sf(r.get("h15_first_solver_s"))
            else:
                extra_d += sf(r.get("h10_first_decision_s"))
                extra_s += sf(r.get("h10_first_solver_s"))
        extra_probe_dec += extra_d
        extra_probe_sol += extra_s
        if h == 10:
            pol_phys += sf(r["h10_physical"])
            pol_dec += sf(r["h10_decision_sum_s"]) + extra_d
            pol_sol += sf(r["h10_solver_sum_s"]) + extra_s
            conf["TP" if pos else "FP"] += 1
            if cat:
                bad.append({
                    "candidate_id": key,
                    "bank_id": r["bank_id"],
                    "source_key": r["source_key"],
                    "role": r["role"],
                    "offset": r["offset_from_center"],
                    "phys_delta": r["phys_delta_h10_minus_h15"],
                    "decision_gain_s": r["decision_gain_h10_vs_h15_s"],
                })
        else:
            pol_phys += sf(r["h15_physical"])
            pol_dec += sf(r["h15_decision_sum_s"]) + extra_d
            pol_sol += sf(r["h15_solver_sum_s"]) + extra_s
            conf["FN" if pos else "TN"] += 1
        details.append({
            "candidate_id": key,
            "bank_id": r["bank_id"],
            "source_key": r["source_key"],
            "selected_h": h,
            "label_positive": pos,
            "catastrophic": cat,
            "phys_delta": r["phys_delta_h10_minus_h15"],
            "decision_gain_s": r["decision_gain_h10_vs_h15_s"],
            "extra_probe_decision_s": extra_d,
            "extra_probe_solver_s": extra_s,
        })
    dec_save = (fixed_dec - pol_dec) / fixed_dec if fixed_dec > 0 else 0.0
    sol_save = (fixed_sol - pol_sol) / fixed_sol if fixed_sol > 0 else 0.0
    phys_delta = pol_phys - fixed_phys
    return {
        "rows": len(rows),
        "overhead_mode": overhead_mode,
        "chosen_counts": dict(counts),
        "confusion": dict(conf),
        "policy_physical_sum": pol_phys,
        "fixed_H15_physical_sum": fixed_phys,
        "physical_delta_vs_fixed_H15": phys_delta,
        "physical_tolerance_sum": tol_sum,
        "physical_gate": phys_delta <= tol_sum,
        "policy_decision_sum_s": pol_dec,
        "fixed_H15_decision_sum_s": fixed_dec,
        "extra_probe_decision_sum_s": extra_probe_dec,
        "decision_relative_saving_vs_fixed_H15": dec_save,
        "policy_solver_sum_s": pol_sol,
        "fixed_H15_solver_sum_s": fixed_sol,
        "extra_probe_solver_sum_s": extra_probe_sol,
        "solver_relative_saving_vs_fixed_H15": sol_save,
        "catastrophic_false_positive_rows": bad,
        "pass_5pct_no_cat_fp": bool(dec_save >= 0.05 and phys_delta <= tol_sum and not bad),
        "pass_10pct_no_cat_fp": bool(dec_save >= 0.10 and phys_delta <= tol_sum and not bad),
        "details": details,
    }


def nested_v18b(rows: Sequence[Mapping[str, Any]], fdicts: Mapping[str, Mapping[str, float]], cfgs: Sequence[Mapping[str, Any]], group_field: str) -> Tuple[Dict[str, Any], int]:
    groups = sorted({str(r[group_field]) for r in rows})
    outer: Dict[str, Any] = {}
    evals = 0
    for og in groups:
        train_outer = [r for r in rows if str(r[group_field]) != og]
        ev_outer = [r for r in rows if str(r[group_field]) == og]
        inner_groups = sorted({str(r[group_field]) for r in train_outer})
        best_cfg = None
        best_inner = None
        for cfg in cfgs:
            hold = {}
            for ig in inner_groups:
                tr = [r for r in train_outer if str(r[group_field]) != ig]
                evr = [r for r in train_outer if str(r[group_field]) == ig]
                hold[ig] = base.eval_cfg(tr, evr, fdicts, cfg, include_scores=False)
                evals += 1
            mode = base.overhead_for_family(str(cfg["family"]))
            inn = base.aggregate(train_outer, hold, mode)
            if best_inner is None or base.rank(base.summary(inn)) < base.rank(base.summary(best_inner)):
                best_inner = inn
                best_cfg = dict(cfg)
        assert best_cfg is not None and best_inner is not None
        ev = base.eval_cfg(train_outer, ev_outer, fdicts, best_cfg, include_scores=True)
        evals += 1
        outer[og] = {
            "selected_config_id": base.cfg_id(best_cfg),
            "selected_config": best_cfg,
            "inner_eval": base.summary(best_inner),
            "outer_eval": base.compact_eval(ev),
        }
    fixed_phys = math.fsum(sf(r["h15_physical"]) for r in rows)
    fixed_dec = math.fsum(sf(r["h15_decision_sum_s"]) for r in rows)
    fixed_sol = math.fsum(sf(r["h15_solver_sum_s"]) for r in rows)
    tol_sum = math.fsum(sf(r["row_physical_tolerance"]) for r in rows)
    pol_phys = pol_dec = pol_sol = 0.0
    bad = []
    counts: Counter[str] = Counter()
    conf: Counter[str] = Counter()
    detail = []
    details_by_id = {d["candidate_id"]: d for obj in outer.values() for d in (obj["outer_eval"].get("details") or [])}
    for r in rows:
        d = details_by_id.get(base.row_key(r), {"selected_h": 15, "extra_probe_decision_s": 0.0, "extra_probe_solver_s": 0.0})
        h = int(d.get("selected_h", 15))
        counts[str(h)] += 1
        extra_d = sf(d.get("extra_probe_decision_s"))
        extra_s = sf(d.get("extra_probe_solver_s"))
        pos = bool(r["h10_beneficial_vs_h15"])
        cat = bool(r["h10_catastrophic_vs_h15"])
        if h == 10:
            pol_phys += sf(r["h10_physical"])
            pol_dec += sf(r["h10_decision_sum_s"]) + extra_d
            pol_sol += sf(r["h10_solver_sum_s"]) + extra_s
            conf["TP" if pos else "FP"] += 1
            if cat:
                bad.append({
                    "candidate_id": base.row_key(r),
                    "bank_id": r["bank_id"],
                    "source_key": r["source_key"],
                    "phys_delta": r["phys_delta_h10_minus_h15"],
                    "decision_gain_s": r["decision_gain_h10_vs_h15_s"],
                })
        else:
            pol_phys += sf(r["h15_physical"])
            pol_dec += sf(r["h15_decision_sum_s"]) + extra_d
            pol_sol += sf(r["h15_solver_sum_s"]) + extra_s
            conf["FN" if pos else "TN"] += 1
        detail.append({
            "candidate_id": base.row_key(r),
            "bank_id": r["bank_id"],
            "source_key": r["source_key"],
            "selected_h": h,
            "extra_probe_decision_s": extra_d,
            "extra_probe_solver_s": extra_s,
            "label_positive": pos,
            "catastrophic": cat,
        })
    dec_save = (fixed_dec - pol_dec) / fixed_dec if fixed_dec else 0.0
    sol_save = (fixed_sol - pol_sol) / fixed_sol if fixed_sol else 0.0
    phys_delta = pol_phys - fixed_phys
    agg = {
        "rows": len(rows),
        "chosen_counts": dict(counts),
        "confusion": dict(conf),
        "policy_physical_sum": pol_phys,
        "fixed_H15_physical_sum": fixed_phys,
        "physical_delta_vs_fixed_H15": phys_delta,
        "physical_tolerance_sum": tol_sum,
        "physical_gate": phys_delta <= tol_sum,
        "policy_decision_sum_s": pol_dec,
        "fixed_H15_decision_sum_s": fixed_dec,
        "decision_relative_saving_vs_fixed_H15": dec_save,
        "policy_solver_sum_s": pol_sol,
        "fixed_H15_solver_sum_s": fixed_sol,
        "solver_relative_saving_vs_fixed_H15": sol_save,
        "catastrophic_false_positive_rows": bad,
        "pass_5pct_no_cat_fp": bool(dec_save >= 0.05 and phys_delta <= tol_sum and not bad),
        "pass_10pct_no_cat_fp": bool(dec_save >= 0.10 and phys_delta <= tol_sum and not bad),
        "details": detail,
    }
    return {"group_field": group_field, "groups": groups, "outer": outer, "aggregate": agg}, evals


def freeze_protocol_v18b(created: dt.datetime, diag: Mapping[str, Any], cfgs: Sequence[Mapping[str, Any]], hashes: Mapping[str, str], backup_commit: str) -> None:
    base.write_json(PROTOCOL, {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_probe_telemetry_risk_value_no_sim_no_validation_no_test",
        "source_repair_from_v18": {
            "reason": "pre-execution repair of solver-overhead accounting, nested solver aggregation, and offline-evaluation cap; v18 source remains archived and unexecuted",
            "base_source": rel(BASE_PATH),
            "base_source_sha256": sha256(BASE_PATH),
            "v18b_source": rel(SOURCE),
            "v18b_source_sha256": sha256(SOURCE),
        },
        "hypothesis": "H10 candidate-solve terminal/objective telemetry may contain bank/source-invariant risk information missing from static/history observations; if strict nested leave-bank and leave-source gates pass after charging probe overhead, a bounded online overhead smoke is justified. If only global diagnostics pass or nested gates fail, telemetry is informative in-sample but not source/bank invariant enough; if nested selections collapse to no H10, the veto is safe but not useful.",
        "inputs": {
            "v15_boundary_raw": rel(base.V15_RAW),
            "v15_boundary_completed": rel(base.V15_DONE),
            "v17_completed": rel(base.V17_DONE),
        },
        "data_diag": dict(diag),
        "candidate_config_count": len(cfgs),
        "families": ["h10_probe_objective_only", "h10_probe_plus_state", "dual_probe_objective"],
        "strict_splits": ["leave_bank", "leave_source_key"],
        "overhead_semantics": "H10-probe rejected rows add first H10 decision time to decision cost and first H10 solver time to solver cost; dual-probe rows add the unused first solve's decision/solver time to the selected arm.",
        "decision_gate": {"zero_catastrophic_H10_false_positives": True, "physical_gate": True, "minimum_overhead_charged_decision_saving_vs_fixed_H15": 0.05},
        "budget_declared": {
            "development_mpc_simulation_episodes": 0,
            "development_control_steps": 0,
            "candidate_configs": len(cfgs),
            "offline_model_evaluations_cap": 150000,
            "training_episodes": 0,
            "gradient_steps": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "latest_verified_backup_before_run_from_supervisor_context": backup_commit,
        "input_hashes": dict(hashes),
    })


def append_docs_v18b(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    elapsed_h = (base.now() - FIRST_EVENT).total_seconds() / 3600.0
    block = f"""<!-- {MARKER} -->
## 2026-09-30 vehicle true-variable-H v18b H10 probe-telemetry diagnostic

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED offline diagnostic over existing v15 true-H branch traces; no MPC simulation, no validation64/sealed-test access, no gradient training. v18b repairs the unexecuted v18 overhead accounting by charging first-solver time separately from first-decision time. rows={h['rows']}; configs={h['config_count']}; offline_model_evaluations={h['offline_model_evaluations']}; leave_bank_save={h['leave_bank_save']:.6f}; leave_bank_bad={h['leave_bank_bad']}; leave_bank_h10={h['leave_bank_h10']}; leave_source_save={h['leave_source_save']:.6f}; leave_source_bad={h['leave_source_bad']}; leave_source_h10={h['leave_source_h10']}; best_global={h['best_global_config']} save={h['best_global_save']:.6f} bad={h['best_global_bad']} H10={h['best_global_h10']}. Decision: {raw['decision']}. Artifacts: `{rel(OUT/'summary.md')}`, `{rel(OUT/'raw.json')}`, `{rel(OUT/'completed.json')}`. Canonical experiment registry row is written by run_experiment, not by this script.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    base.load_rows = load_rows_v18b
    base.evaluate = evaluate_v18b
    base.nested = nested_v18b
    base.freeze_protocol = freeze_protocol_v18b
    base.append_docs = append_docs_v18b
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    args = ap.parse_args(argv)
    if not args.run:
        print("pass --run", file=sys.stderr)
        return 2
    return int(base.main(["--run", "--backup-verified-commit", args.backup_verified_commit]))


if __name__ == "__main__":
    raise SystemExit(main())
