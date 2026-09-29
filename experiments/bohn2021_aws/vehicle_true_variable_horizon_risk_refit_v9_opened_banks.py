#!/usr/bin/env python3
"""v9 opened-bank risk-aware selector/refit smoke for true variable H10/H15.

Development-only IMPROVED diagnostic after v8c.  This script consumes only
already-opened development branch-state banks (fresh v0/v1/v2 plus the new v8c
risk-probe bank) and performs a bounded finite selector refit/evaluation.  It
runs no MPC simulation, no gradient training, no validation64, and no sealed
final test.

Hypothesis: v8c demonstrated real measured compute opportunity but catastrophic
fixed-H10 risk.  If a conservative cost-sensitive tree over deployable online
features can leave-one-bank-out (including held-out v8c) select H10 only on safe
states, then the next concrete experiment should be a small closed-loop/branch
confirmation with actual selector overhead.  If only scenario/window metadata
works, or all deployable variants retain catastrophic false positives/collapse,
then the bottleneck is risk representation/value learning rather than label
thresholding; next action should be richer risk/value refit/training or targeted
risk data acquisition.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_tree_representation_v6 as v6  # noqa:E402

NAME = "vehicle_true_variable_horizon_risk_refit_v9_opened_banks"
STAMP = "20260929T1345Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1345_after_risk_refit_v9.md"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-risk-refit-v9-opened-banks-{STAMP}"
V8C_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z"
SOURCE = Path(__file__).resolve()

class ContractError(RuntimeError):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def sf(x: Any, default: float = 0.0) -> float:
    return v6.sf(x, default)


def si(x: Any, default: int = 0) -> int:
    return v6.si(x, default)


def clean(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    return x


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def obs14(st: Mapping[str, Any]) -> List[float]:
    raw = st.get("initial_observation_from_h15_trace") or st.get("initial_observation_at_branch") or []
    vals = [sf(x) for x in raw[:14]] if isinstance(raw, list) else []
    return (vals + [0.0] * 14)[:14]


def load_v8c_rows() -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    raw_path = V8C_DIR / "raw.json"
    done_path = V8C_DIR / "completed.json"
    man_path = V8C_DIR / "selected_state_manifest.json"
    for p in (raw_path, done_path, man_path):
        if not p.exists():
            raise ContractError("missing v8c input: " + rel(p))
    raw = read_json(raw_path)
    done = read_json(done_path)
    man = read_json(man_path)
    for obj_name, obj in (("raw", raw), ("completed", done), ("manifest", man)):
        if obj.get("validation64_bank_opened") is not False or obj.get("sealed_test_accessed") is not False:
            raise ContractError(f"forbidden validation/test access flag in v8c {obj_name}")
    if done.get("passed") is not True:
        raise ContractError("v8c completion did not pass")
    by_base = {str(s.get("base_state_id")): s for s in (man.get("selected_states") or [])}
    rows: List[Dict[str, Any]] = []
    for r in ((raw.get("analysis") or {}).get("state_rows") or []):
        sid = str(r.get("base_state_id"))
        st = by_base.get(sid, {})
        h10 = r.get("h10") or {}
        h15 = r.get("h15") or {}
        prev = st.get("branch_previous_state") if isinstance(st.get("branch_previous_state"), Mapping) else {}
        group = str(r.get("risk_probe_role") or st.get("fresh_confirmation_group") or "v8c_unknown")
        window = str(r.get("slot_name") or st.get("window") or "unknown")
        h10_phys = sf(h10.get("physical")); h15_phys = sf(h15.get("physical"))
        h10_dec = sf(h10.get("decision_sum_s")); h15_dec = sf(h15.get("decision_sum_s"))
        h10_sol = sf(h10.get("solver_sum_s")); h15_sol = sf(h15.get("solver_sum_s"))
        rows.append({
            "bank_id": "fresh_v8c", "base_state_id": sid, "group": group, "window": window,
            "branch_state_slot": si(r.get("branch_state_slot", st.get("branch_state_slot")), 0),
            "branch_step": si(r.get("branch_step", st.get("branch_step")), -1),
            "obs14": obs14(st), "prev_x": sf(prev.get("x")), "prev_y": sf(prev.get("y")), "prev_theta": sf(prev.get("theta")),
            "stage_a_trace_risk_score": sf(st.get("stage_a_trace_risk_score"), 0.0),
            "h10_physical": h10_phys, "h15_physical": h15_phys,
            "h10_decision_sum_s": h10_dec, "h15_decision_sum_s": h15_dec,
            "h10_solver_sum_s": h10_sol, "h15_solver_sum_s": h15_sol,
            "h10_safe_all": bool(h10.get("safe_all")), "h15_safe_all": bool(h15.get("safe_all")),
            "row_physical_tolerance": max(2.0, sf(r.get("row_tolerance_vs_H15"), 2.0)),
            "phys_delta_h10_minus_h15": sf(r.get("physical_delta_h10_minus_h15", h10_phys - h15_phys)),
            "decision_gain_h10_vs_h15_s": sf(h15_dec - h10_dec),
            "solver_gain_h10_vs_h15_s": sf(h15_sol - h10_sol),
            "h10_beneficial_vs_h15": bool(r.get("h10_beneficial_vs_h15")),
            "h10_catastrophic_vs_h15": bool(r.get("h10_catastrophic_vs_h15")),
        })
    if len(rows) != 8:
        raise ContractError(f"expected 8 v8c rows, got {len(rows)}")
    return rows, {rel(raw_path): sha256(raw_path), rel(done_path): sha256(done_path), rel(man_path): sha256(man_path)}


def oracle_eval(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    choices = {v6.row_key(r): (10 if r["h10_beneficial_vs_h15"] else 15) for r in rows}
    return v6.eval_choices(rows, choices)


def fit_grid(rows: List[Dict[str, Any]], all_groups: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    fdicts = {v6.row_key(r): v6.feature_dict(r, all_groups) for r in rows}
    banks = sorted(set(str(r["bank_id"]) for r in rows))
    by_bank = {b: [r for r in rows if str(r["bank_id"]) == b] for b in banks}
    families = ["deploy_obs_pose_step_no_risk", "deploy_obs_pose_step_risk", "step_window_only", "scenario_group_window_only", "deploy_plus_group_window"]
    variants: List[Dict[str, Any]] = []
    for fam in families:
        names = v6.feature_names(fam, all_groups)
        for depth in [1, 2, 3, 4]:
            for min_leaf in [2, 3, 4, 5]:
                for min_pos in [1, 2, 3]:
                    for max_cat in [0]:
                        for min_gain in [0.0, 0.25, 0.5, 1.0, 1.5]:
                            hold = {}; total_bad = 0; phys_fail = 0; total_h10 = 0; min_save = 1e9; avg_save = 0.0; pass5 = True; pass10 = True
                            trees = {}
                            for hb in banks:
                                train = [r for r in rows if str(r["bank_id"]) != hb]
                                tree = v6.build_tree(train, fdicts, names, depth, min_leaf, min_pos, max_cat, min_gain)
                                choices = {v6.row_key(r): v6.predict(tree, fdicts[v6.row_key(r)]) for r in by_bank[hb]}
                                ev = v6.eval_choices(by_bank[hb], choices)
                                hold[hb] = ev; trees[hb] = v6.compact_tree(tree)
                                bad = len(ev["catastrophic_false_positive_rows"])
                                total_bad += bad
                                phys_fail += 0 if ev["physical_gate"] else 1
                                total_h10 += int(ev["chosen_counts"].get("10", 0))
                                save = sf(ev["decision_relative_saving_vs_fixed_H15"])
                                min_save = min(min_save, save); avg_save += save
                                pass5 = pass5 and bool(ev["pass_5pct_no_cat_fp"])
                                pass10 = pass10 and bool(ev["pass_10pct_no_cat_fp"])
                            avg_save /= len(banks)
                            spec = {"family": fam, "max_depth": depth, "min_leaf": min_leaf, "min_pos_leaf": min_pos, "max_train_cat_leaf": max_cat, "min_leaf_gain_s": min_gain}
                            variants.append({"variant_id": f"v9_{fam}_d{depth}_ml{min_leaf}_mp{min_pos}_mc{max_cat}_mg{min_gain:g}", "spec": spec, "holdout_evaluations": hold, "trees_by_holdout": trees, "all_holdouts_pass5": pass5, "all_holdouts_pass10": pass10, "total_catastrophic_fp": total_bad, "physical_gate_fail_count": phys_fail, "min_decision_saving": min_save, "avg_decision_saving": avg_save, "total_h10": total_h10, "feature_count": len(names)})
    return variants, fdicts


def rank_key(c: Mapping[str, Any]) -> Tuple[Any, ...]:
    pure = c["spec"]["family"] in ("deploy_obs_pose_step_no_risk", "deploy_obs_pose_step_risk")
    v8c = (c.get("holdout_evaluations") or {}).get("fresh_v8c") or {}
    v8c_bad = len(v8c.get("catastrophic_false_positive_rows") or [])
    v8c_save = sf(v8c.get("decision_relative_saving_vs_fixed_H15"), -9.0)
    return (not bool(c["all_holdouts_pass10"]), not bool(c["all_holdouts_pass5"]), int(c["total_catastrophic_fp"]), int(v8c_bad), int(c["physical_gate_fail_count"]), not pure, -sf(c["min_decision_saving"]), -v8c_save, -sf(c["avg_decision_saving"]), -int(c["total_h10"]))


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]; top = raw["top_variants"]
    lines = ["# Vehicle true-variable-H risk-aware selector/refit v9 opened-bank smoke", "", f"UTC: `{raw['created_utc']}`. Development-only finite selector refit over opened banks v0/v1/v2/v8c; no MPC simulation, no validation64, no sealed test, no gradient training.", "", "## Evidence and objective", "", "v8c showed measured compute opportunity from identical branch states but catastrophic fixed-H10 modes. v9 tests whether a conservative cost-sensitive tree policy can learn the control/compute tradeoff without treating sparse positive labels as a universal blocker.", "", "## Headline", "", f"- Rows: `{h['rows']}` across banks `{h['banks']}`; positives `{h['positive_rows']}`; catastrophic H10 rows `{h['catastrophic_rows']}`.", f"- Oracle opened-bank decision saving vs fixed H15: `{100.0*sf(h['oracle_decision_saving']):.2f}%`; oracle physical gate `{h['oracle_physical_gate']}`.", f"- Variants fit: `{h['variant_count']}`; pure-deploy pass10 `{h['pure_deploy_pass10']}`, pass5 `{h['pure_deploy_pass5']}`; all-family pass10 `{h['all_family_pass10']}`.", f"- Best deployable: `{h['best_deploy_variant']}` min save `{100.0*sf(h['best_deploy_min_save']):.2f}%`, bad `{h['best_deploy_bad']}`, v8c holdout save `{100.0*sf(h['best_deploy_v8c_save']):.2f}%`, v8c bad `{h['best_deploy_v8c_bad']}`.", f"- Best overall: `{h['best_variant']}` family `{h['best_family']}` min save `{100.0*sf(h['best_min_save']):.2f}%`, bad `{h['best_bad']}`.", "", "## Top variants", "", "| rank | family | pass10 | pass5 | bad | v8c bad | phys fails | min save | v8c save | avg save | H10 total | variant |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for i, v in enumerate(top[:15], 1):
        ev8 = (v.get("holdout_evaluations") or {}).get("fresh_v8c") or {}
        lines.append(f"| {i} | `{v['spec']['family']}` | `{v['all_holdouts_pass10']}` | `{v['all_holdouts_pass5']}` | {v['total_catastrophic_fp']} | {len(ev8.get('catastrophic_false_positive_rows') or [])} | {v['physical_gate_fail_count']} | {100.0*sf(v['min_decision_saving']):.2f}% | {100.0*sf(ev8.get('decision_relative_saving_vs_fixed_H15')):.2f}% | {100.0*sf(v['avg_decision_saving']):.2f}% | {v['total_h10']} | `{v['variant_id']}` |")
    lines += ["", "## Best deployable holdouts", "", "| held-out bank | H counts | confusion | bad | physical gate | decision save | pass10 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for b, ev in raw["best_deployable_variant"]["holdout_evaluations"].items():
        lines.append(f"| `{b}` | `{ev['chosen_counts']}` | `{ev['confusion']}` | {len(ev['catastrophic_false_positive_rows'])} | `{ev['physical_gate']}` | {100.0*sf(ev['decision_relative_saving_vs_fixed_H15']):.2f}% | `{ev['pass_10pct_no_cat_fp']}` |")
    lines += ["", "## Decision", "", raw["decision"], "", "Interpretation: opened-development refit evidence only. It can justify the next bounded fresh confirmation or risk/value retraining design, but it is not validation64 or final-test evidence and does not prove speed without later actual end-to-end overhead measurement."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-risk-refit-v9", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_risk_refit_v9:
        raise ContractError("requires --run and explicit v9 development refit acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    created0 = now()
    source_rows, hashes, old_groups = v6.load_rows()
    v8c_rows, v8_hashes = load_v8c_rows()
    rows = source_rows + v8c_rows
    all_groups = sorted(set(old_groups) | {str(r["group"]) for r in rows} | {f"window::{r['window']}" for r in rows} | {f"gw::{r['group']}::{r['window']}" for r in rows})
    hashes.update(v8_hashes); hashes[rel(SOURCE)] = sha256(SOURCE); hashes[rel(Path(v6.__file__))] = sha256(Path(v6.__file__))
    protocol = {"protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}", "created_utc": created0.isoformat(), "classification": "development_IMPROVED_opened_bank_selector_refit_no_simulation_no_validation_no_test", "hypothesis": "A conservative cost-sensitive deployable tree can use v8c risk anchors to separate H10-safe compute-saving states from catastrophic states better than prior label-density sweeps.", "inputs": sorted(hashes), "split": "leave_one_opened_development_bank_out over fresh_v0/fresh_v1/fresh_v2/fresh_v8c", "objective": "no catastrophic H10 false positives and physical gate before measured decision-time saving; fixed-H15 comparator; oracle reported separately", "budget_declared": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "new_training_episodes": 0, "gradient_steps": 0, "selector_refit_fits_upper_bound": 1200, "validation64_episodes": 0, "sealed_test_episodes": 0}, "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False}}
    write_json(PROTOCOL, protocol)
    variants, _ = fit_grid(rows, all_groups)
    ranked = sorted(variants, key=rank_key)
    deploy = [v for v in variants if v["spec"]["family"] in ("deploy_obs_pose_step_no_risk", "deploy_obs_pose_step_risk")]
    best = ranked[0]; best_deploy = sorted(deploy, key=rank_key)[0]
    oracle = oracle_eval(rows)
    ev8_best_dep = best_deploy["holdout_evaluations"].get("fresh_v8c", {})
    headline = {"rows": len(rows), "banks": sorted(set(str(r["bank_id"]) for r in rows)), "positive_rows": sum(1 for r in rows if r["h10_beneficial_vs_h15"]), "catastrophic_rows": sum(1 for r in rows if r["h10_catastrophic_vs_h15"]), "oracle_decision_saving": oracle["decision_relative_saving_vs_fixed_H15"], "oracle_physical_gate": oracle["physical_gate"], "variant_count": len(variants), "pure_deploy_pass10": sum(1 for v in deploy if v["all_holdouts_pass10"]), "pure_deploy_pass5": sum(1 for v in deploy if v["all_holdouts_pass5"]), "all_family_pass10": sum(1 for v in variants if v["all_holdouts_pass10"]), "best_deploy_variant": best_deploy["variant_id"], "best_deploy_min_save": best_deploy["min_decision_saving"], "best_deploy_bad": best_deploy["total_catastrophic_fp"], "best_deploy_v8c_save": ev8_best_dep.get("decision_relative_saving_vs_fixed_H15"), "best_deploy_v8c_bad": len(ev8_best_dep.get("catastrophic_false_positive_rows") or []), "best_variant": best["variant_id"], "best_family": best["spec"]["family"], "best_min_save": best["min_decision_saving"], "best_bad": best["total_catastrophic_fp"]}
    if headline["pure_deploy_pass10"] > 0:
        decision = "Pure deployable cost-sensitive selector passes 10% LOBO gate including v8c; next freeze an unused fresh-source confirmation with actual selector overhead and fair fixed-H baselines before validation64."
    elif headline["pure_deploy_pass5"] > 0:
        decision = "Pure deployable selector only reaches weak LOBO tradeoff; next run a small overhead-aware confirmation only if its v8c heldout has no catastrophic FP, otherwise improve risk/value representation."
    elif headline["all_family_pass10"] > 0:
        decision = "Scenario/window or hybrid information can solve opened-bank risk while pure deployable features fail; next bounded intervention should add/learn deployable risk-state/value features rather than more threshold sweeps."
    else:
        decision = "No opened-bank tree selector meets weak/strong safety-compute gates with v8c included; prioritize richer risk/value representation training/refit or targeted risk-data acquisition, not validation rollout."
    raw = {"created_utc": now().isoformat(), "backup_verified_commit_from_supervisor_context": args.backup_verified_commit, "classification": protocol["classification"], "protocol": rel(PROTOCOL), "access_flags": protocol["access_flags"], "budget_actual": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "new_training_episodes": 0, "gradient_steps": 0, "selector_refit_fits": len(variants) * len(headline["banks"]), "validation64_episodes": 0, "sealed_test_episodes": 0}, "headline": headline, "oracle_opened_bank": oracle, "top_variants": ranked[:20], "best_deployable_variant": best_deploy, "best_overall_variant": best, "bank_label_summary": {b: {"rows": sum(1 for r in rows if r["bank_id"] == b), "positive": sum(1 for r in rows if r["bank_id"] == b and r["h10_beneficial_vs_h15"]), "catastrophic": sum(1 for r in rows if r["bank_id"] == b and r["h10_catastrophic_vs_h15"])} for b in headline["banks"]}, "hashes": hashes, "decision": decision, "platform": {"python": sys.version, "platform": platform.platform()}}
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_RISK_REFIT_V9_OPENED_BANKS_{STAMP}.json"
    write_json(req, {"created_utc": raw["created_utc"], "reason": "backup after v9 opened-bank selector refit smoke before any further simulation/training", "paths": [rel(OUT), rel(PROTOCOL), rel(STATE), rel(SOURCE)], "validation64_bank_opened": False, "sealed_test_accessed": False})
    done = {"passed": True, "status": "complete", "marker": MARKER, "summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "backup_request": rel(req), "headline": headline, "decision": decision, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_mpc_simulation_episodes": 0, "new_control_steps": 0, "gradient_steps": 0, "selector_refit_fits": raw["budget_actual"]["selector_refit_fits"], "hashes": {rel(SOURCE): sha256(SOURCE), rel(PROTOCOL): sha256(PROTOCOL), rel(OUT / "raw.json"): sha256(OUT / "raw.json"), rel(OUT / "summary.md"): sha256(OUT / "summary.md")}}
    write_json(OUT / "completed.json", done)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("# Continue state after v9 risk-aware selector/refit smoke\n\n" + json.dumps(clean({"utc": raw["created_utc"], "headline": headline, "decision": decision, "budgets": raw["budget_actual"], "artifacts": {"summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "completed": rel(OUT / "completed.json")}, "next_action": "Verify backup, then follow v9 decision: overhead-aware unused fresh confirmation if deployable selector passed, otherwise implement bounded richer risk/value representation or targeted risk-data acquisition. No validation64/sealed test."}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = f"\n- {raw['created_utc']} `{NAME}`: opened-bank risk-aware finite selector refit including v8c; pure_deploy_pass10={headline['pure_deploy_pass10']}, pure_deploy_pass5={headline['pure_deploy_pass5']}, best_deploy={headline['best_deploy_variant']} min_save={headline['best_deploy_min_save']:.4f} bad={headline['best_deploy_bad']} v8c_bad={headline['best_deploy_v8c_bad']}; no simulation/validation/test; artifacts `{rel(OUT / 'summary.md')}`.\n"
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + audit, encoding="utf-8")
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8") as f:
        f.write(f"{STAMP},{NAME},development_risk_selector_refit,no_simulation_opened_banks_v0_v1_v2_v8c,0,0,{raw['budget_actual']['selector_refit_fits']},0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": headline, "decision": decision, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "completed.json", {"passed": False, "status": "failed", "error": type(exc).__name__, "message": str(exc), "validation64_bank_opened": False, "sealed_test_accessed": False})
        raise
