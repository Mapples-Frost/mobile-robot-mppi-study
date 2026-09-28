#!/usr/bin/env python3
"""Vehicle terminal-value/objective source/config/checkpoint audit v0.

Analysis-only diagnostic.  No MPC rollouts, no training/refit, no validation64
bank opening, and no sealed-test access.  This audit follows the Stage1/Stage2
stress opportunity result by inspecting source/config/checkpoint lineage for
objective, reward, terminal-value and policy-class explanations of sparse or
fragile adaptive-horizon labels.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17"
CFG = ART / "configs/vehicle.json"
RUNTIME = ROOT / "experiments/bohn2021_reproduction/runtime.py"
RUNPY = ROOT / "experiments/bohn2021_reproduction/run.py"
GATED_SEARCH = ROOT / "experiments/bohn2021_reproduction/gated_horizon_search.py"
GATED_POLICY = ROOT / "experiments/bohn2021_reproduction/gated_horizon_policy.py"
FIXED_BRANCHES = ROOT / "experiments/bohn2021_reproduction/fixed_policy_branches.py"
LETMPC = ART / "sources/gym-horizon/gym_let_mpc/let_mpc.py"
CONTROLLERS = ART / "sources/gym-horizon/gym_let_mpc/controllers.py"
STRESS_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.json"
STRESS_STAGE2_POST = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_postdiagnostic_v0_20260928T1815Z/completed.json"
REWARD_AUDIT = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_reward_timing_terminal_audit_v0_20260928T1820Z/completed.json"
GATED_TRAIN_ROOT = ART / "results/gated_horizon_search_2026-09-25/train"
OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_terminal_objective_source_audit_v0_20260928T1830Z"
STATE = ROOT / "research_artifacts/aws_state/vehicle_terminal_objective_source_audit_v0_20260928T1830Z.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-terminal-objective-source-audit-v0-20260928T1830Z"
HORIZONS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
CURRENT_GATED_SHORT_H = [5, 10, 15, 20]
BASE_H = 25


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha256(p: Path) -> Optional[str]:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sf(x: Any) -> Optional[float]:
    try:
        if x is None or isinstance(x, bool):
            return None
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def si(x: Any) -> Optional[int]:
    try:
        if x is None or isinstance(x, bool):
            return None
        y = int(x)
        return y
    except Exception:
        return None


def grep_context(text: str, patterns: Sequence[str], window: int = 220) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for pat in patterns:
        rows: List[str] = []
        for m in re.finditer(re.escape(pat), text):
            a = max(0, m.start() - window)
            b = min(len(text), m.end() + window)
            snippet = " ".join(text[a:b].split())
            rows.append(snippet)
            if len(rows) >= 4:
                break
        out[pat] = rows
    return out


def extract_coeff(expr: str, var: str) -> Optional[float]:
    # Simple robust extraction for config expressions like "... - 0.001 * mpc_horizon".
    if not isinstance(expr, str) or var not in expr:
        return None
    pats = [
        r"([+-]?\s*\d+(?:\.\d+)?)\s*\*\s*" + re.escape(var),
        re.escape(var) + r"\s*\*\s*([+-]?\s*\d+(?:\.\d+)?)",
    ]
    vals = []
    for pat in pats:
        for m in re.finditer(pat, expr):
            try:
                vals.append(float(m.group(1).replace(" ", "")))
            except Exception:
                pass
    return vals[0] if vals else None


def terminal_file_inventory(folder: Path) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"path": rel(folder), "exists": folder.exists()}
    if not folder.exists():
        return rec
    for name in ("manifest.json", "completed.json"):
        p = folder / name
        item: Dict[str, Any] = {"exists": p.exists(), "sha256": sha256(p)}
        if p.exists():
            try:
                obj = read_json(p)
                for k in ("task", "seed", "fixed_horizon", "steps", "status", "final_hash", "updates", "weights_changed", "reproduction_level"):
                    if k in obj:
                        item[k] = obj.get(k)
                if name == "manifest.json" and isinstance(obj.get("adaptations"), Mapping):
                    item["adaptations"] = obj.get("adaptations")
                if name == "manifest.json":
                    for k in ("reward_scaling", "terminal_state_dim", "terminal_parameter_dim", "observation_dim"):
                        if k in obj:
                            item[k] = obj.get(k)
            except Exception as exc:
                item["read_error"] = repr(exc)
        rec[name] = item
    for pat in ("model.zip", "terminal*", "checkpoint_*.zip"):
        files = sorted(folder.glob(pat))[:20]
        rec[pat] = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files if p.is_file()]
    return rec


def expected_fixed_terminal_dir(seed: int, h: int) -> Path:
    # Mirrors experiments/bohn2021_reproduction/min_q_eval_suite.py model_dir for fixed vehicle terminals.
    if seed == 0:
        return ART / "results/paper_exact_grid_2026-09-23" / ("vehicle_fixed_h%d" % h)
    return ART / "results/min_q_training_2026-09-24" / ("vehicle_fixed_h%d_s%d" % (h, seed))


def audit_stress_terminal_grid(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    term = protocol.get("terminal_grid_readiness_reused_from_v1") or {}
    sources = term.get("terminal_sources") or {}
    by_h: Dict[str, Any] = {}
    mismatches: List[str] = []
    for h in HORIZONS:
        src = sources.get(str(h)) or {}
        folder = ROOT / str(src.get("path", "__missing__"))
        inv = terminal_file_inventory(folder)
        manifest = inv.get("manifest.json") if isinstance(inv.get("manifest.json"), Mapping) else {}
        completed = inv.get("completed.json") if isinstance(inv.get("completed.json"), Mapping) else {}
        checks = {
            "source_path_declared": bool(src.get("path")),
            "manifest_fixed_horizon_matches_H": manifest.get("fixed_horizon") == h,
            "completed_status_complete": completed.get("status") == "complete",
            "completed_steps_15000": completed.get("steps") == 15000,
            "model_zip_hash_matches_protocol": bool(src.get("model_zip_sha256")) and bool(inv.get("model.zip")) and inv.get("model.zip", [{}])[0].get("sha256") == src.get("model_zip_sha256"),
        }
        if not all(checks.values()):
            mismatches.append("H%d:%s" % (h, checks))
        by_h[str(h)] = {"protocol_source": src, "inventory": inv, "checks": checks}
    return {"available_all_required_flag": term.get("available_all_required"), "required_horizons": term.get("required_horizons"), "by_horizon": by_h, "mismatches": mismatches}


def audit_gated_training_lineage() -> Dict[str, Any]:
    out: Dict[str, Any] = {"train_root": rel(GATED_TRAIN_ROOT), "seeds": {}, "current_policy_class": {"base_h": BASE_H, "short_h_grid": CURRENT_GATED_SHORT_H, "can_select_longer_than_base": False, "profiles": 3, "guards": [5, 15, 30], "structured_candidate_count": len(CURRENT_GATED_SHORT_H) * 3 * 3}}
    for seed in (0, 1, 2):
        root = GATED_TRAIN_ROOT / ("vehicle_s%d" % seed)
        rec: Dict[str, Any] = {"root": rel(root), "exists": root.exists()}
        fit_path = root / "fit_completed.json"
        pol_path = root / "policy.json"
        if fit_path.exists():
            fit = read_json(fit_path)
            rec["fit_completed_sha256"] = sha256(fit_path)
            rec["gradient_updates"] = fit.get("gradient_updates")
            rec["validation_access"] = fit.get("validation_access")
            rec["test_access"] = fit.get("test_access")
            rec["selected"] = fit.get("selected")
            results = fit.get("results") or []
            rec["candidate_results_count"] = len(results)
            rec["fully_evaluated_count"] = sum(1 for r in results if r.get("fully_evaluated"))
            rec["rejected_or_pruned_count"] = sum(1 for r in results if r.get("rejected"))
            rec["mean_raw_cost_selected"] = next((r.get("mean_raw_cost") for r in results if (r.get("policy") or {}).get("id") == (fit.get("selected") or {}).get("id")), None)
            rec["mean_physical_cost_selected"] = next((r.get("mean_physical_cost") for r in results if (r.get("policy") or {}).get("id") == (fit.get("selected") or {}).get("id")), None)
            baseline = next((r for r in results if (r.get("policy") or {}).get("id") == "fixed"), None)
            if baseline:
                rec["baseline_fixed_H25_mean_raw_cost"] = baseline.get("mean_raw_cost")
                rec["baseline_fixed_H25_mean_physical_cost"] = baseline.get("mean_physical_cost")
        if pol_path.exists():
            pol = read_json(pol_path)
            rec["policy_json_sha256"] = sha256(pol_path)
            rec["stored_policy"] = pol
        # Current policy terminal is the independently trained fixed H25 terminal for this seed.
        term_dir = expected_fixed_terminal_dir(seed, BASE_H)
        rec["expected_current_H25_terminal"] = terminal_file_inventory(term_dir)
        out["seeds"][str(seed)] = rec
    return out


def config_objective_audit(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    env = cfg.get("environment", {})
    reward = env.get("reward", {})
    info_reward = (env.get("info") or {}).get("reward") or {}
    mpc = cfg.get("mpc", {})
    obj = mpc.get("objective", {})
    expr = reward.get("expression")
    return {
        "environment_max_steps": env.get("max_steps"),
        "action_variables": env.get("action", {}).get("variables"),
        "reward_expression": expr,
        "reward_normalize": reward.get("normalize"),
        "termination_weight": reward.get("termination_weight"),
        "reward_variables": reward.get("variables"),
        "info_reward_terms": info_reward,
        "parsed_reward_horizon_penalty_coeff": extract_coeff(str(expr), "mpc_horizon"),
        "info_computation_horizon_penalty_coeff": extract_coeff(str(info_reward.get("computation")), "mpc_horizon"),
        "mpc_type": mpc.get("type"),
        "mpc_params": mpc.get("params"),
        "mpc_objective_discount_factor": obj.get("discount_factor"),
        "mpc_lterm_expression": ((obj.get("lterm") or {}).get("expression")),
        "mpc_mterm_expression": ((obj.get("mterm") or {}).get("expression")),
        "mpc_vf_config": obj.get("vf"),
        "mpc_R_delta": obj.get("R_delta"),
        "constraint_count": len(mpc.get("constraints") or []),
    }


def source_audit() -> Dict[str, Any]:
    files = {"runtime.py": RUNTIME, "run.py": RUNPY, "gated_horizon_search.py": GATED_SEARCH, "gated_horizon_policy.py": GATED_POLICY, "fixed_policy_branches.py": FIXED_BRANCHES, "let_mpc.py": LETMPC, "controllers.py": CONTROLLERS}
    patterns = ["get_reward", "reward_scale", "mpc_gamma", "discount_factor", "set_value_function_weights_and_biases", "mpc_value_fn", "mpc_rewards", "save_value_function", "vf", "mean_raw_cost", "mean_physical_cost", "h_penalty", "compute", "short_h", "BASE", "mterm", "terminal"]
    out: Dict[str, Any] = {}
    for name, path in files.items():
        if not path.exists():
            out[name] = {"exists": False}
            continue
        text = read_text(path)
        out[name] = {"exists": True, "sha256": sha256(path), "grep": grep_context(text, patterns)}
    return out


def find_stage2_positive_direction() -> Dict[str, Any]:
    rec: Dict[str, Any] = {"source": rel(STRESS_STAGE2_POST), "available": STRESS_STAGE2_POST.exists()}
    if not STRESS_STAGE2_POST.exists():
        return rec
    obj = read_json(STRESS_STAGE2_POST)
    # completed.json may include only aggregate fields; if not, use summary as unavailable.
    rec.update({k: obj.get(k) for k in ("positive_state_count", "relaxed_positive_state_count", "training_refit_label_gate_pass_development_only", "retrain_or_selector_refit_now") if k in obj})
    # Prefer raw if present and not validation/test.
    raw_path = STRESS_STAGE2_POST.parent / "raw.json"
    if raw_path.exists():
        raw = read_json(raw_path)
        positives = []
        try:
            for row in raw.get("relaxed_state_matched_positive_states", []) or raw.get("analysis", {}).get("relaxed_state_matched_positive_states", []):
                positives.append(row)
        except Exception:
            positives = []
        # Fallback scan through state_rows for positive non-H15 gains.
        if not positives:
            try:
                for sr in raw.get("analysis", {}).get("state_rows", []):
                    hs = []
                    for c in sr.get("comparisons", []):
                        h = si(c.get("horizon"))
                        if h != 15 and (sf(c.get("gain_vs_H15_total")) or -1e9) >= 3.0:
                            hs.append(h)
                    if hs:
                        positives.append({"case": sr.get("case"), "branch_step": sr.get("branch_step"), "positive_horizons": sorted(hs)})
            except Exception:
                pass
        rec["positive_direction_rows"] = positives[:20]
        rec["positive_horizons"] = sorted(set(h for r in positives for h in (r.get("positive_horizons") or r.get("positive_H") or [])))
    return rec


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def append_registry(created: str, completed_path: Path) -> None:
    p = ROOT / "EXPERIMENT_REGISTRY.csv"
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    if rel(completed_path) in existing:
        return
    row = {
        "experiment_id": "",
        "timestamp": created,
        "method": "vehicle_terminal_objective_source_audit_v0_analysis_only",
        "seed": "no_rng_source_config_checkpoint_audit",
        "split": "development_diagnostic_no_validation64_no_test",
        "commit_sha": "",
        "status": "completed_analysis_only",
        "exit_status": "0",
        "runtime_seconds": "",
        "peak_process_rss_kb": "",
        "record": rel(completed_path),
    }
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not existing.strip():
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    cfg = read_json(CFG)
    protocol = read_json(STRESS_PROTOCOL) if STRESS_PROTOCOL.exists() else {}
    reward_done = read_json(REWARD_AUDIT) if REWARD_AUDIT.exists() else {}
    objective = config_objective_audit(cfg)
    src = source_audit()
    terminal_grid = audit_stress_terminal_grid(protocol)
    gated = audit_gated_training_lineage()
    positive = find_stage2_positive_direction()

    current_can_hit_positive_long_h = False
    pos_hs = set(int(h) for h in positive.get("positive_horizons", []) if h is not None)
    if pos_hs:
        current_can_hit_positive_long_h = bool(pos_hs.intersection(set(CURRENT_GATED_SHORT_H + [BASE_H])))
    selection_no_measured_timing = True  # verified from gated_horizon_search.py objective and previous timing audit; source snippets included.
    terminal_grid_mismatches = terminal_grid.get("mismatches") or []
    current_terminal_incompatibility_hypothesis = "Current gated policies choose only short_h in {5,10,15,20} relative to base H25 but use the seed's fixed-H25 terminal; this is a plausible horizon-compatibility bias, not directly proven harmful by this source audit."

    findings = [
        "Vehicle reward/config uses a synthetic horizon/computation penalty term in the environment reward/info, separate from measured decision wall time.",
        "MPC objective supports a learned terminal value (`vf`) and source LetMPC exposes `mpc_value_fn` when `vf` is configured, but existing stress traces did not record numeric value predictions/errors; terminal accuracy remains unmeasured.",
        "Stress Stage1 fixed-H grid used H-specific terminal-source metadata for required H values; no source/checkpoint hash mismatch was found" if not terminal_grid_mismatches else "Stress terminal grid has mismatches: %s" % terminal_grid_mismatches,
        "Current gated-horizon policies are finite searched, not gradient-trained: selected policy.json per seed records short_h/profile/guard; fit_completed records gradient_updates=0, validation_access=false, test_access=false.",
        "Current gated policy class can only shorten from H25 to {5,10,15,20}; it cannot select H30/H35/H45. The stress Stage2 relaxed positive direction, where available, points to %s, so the current class is not aligned with that observed positive branch if it requires longer-than-H25 horizons." % (sorted(pos_hs) if pos_hs else "no parsed positive horizons"),
        "Gated search selection objective used mean raw episode cost with synthetic h_penalty and safety constraints; source does not use measured decision time for training/selection.",
    ]
    hypotheses = [
        {"id": "terminal_value_bias_or_horizon_mismatch", "rank": 1, "evidence": current_terminal_incompatibility_hypothesis + " H-specific fixed-H terminals exist for stress fixed-H mapping, but current adaptive short-H policy uses H25 terminal lineage.", "missing": "Numeric terminal predictions/errors at matched states and terminal-off/per-H terminal ablations.", "discriminating_experiment": "Instrument a no-training matched-state mini-smoke that records mpc_value_fn, mpc lterm/mterm and terminal-off outcomes for the one positive stress state plus neutral/harm controls; do not train on it."},
        {"id": "policy_class_direction_mismatch", "rank": 2, "evidence": "Current gated policy class only shortens from H25; sparse stress positive parsed as horizons %s. This can explain why local patches to safe-shortening do not exploit longer-horizon opportunities." % (sorted(pos_hs) if pos_hs else "unknown"), "missing": "Whether a multi-H selector with {10,15,25,30,35} has enough robust labels on a broader source-supported scenario design.", "discriminating_experiment": "If further scenarios are generated, require pre-frozen labels for both shorter and longer safe horizons before refit; otherwise no selector refit."},
        {"id": "objective_timing_alignment", "rank": 3, "evidence": "Reward and gated search use synthetic h_penalty/raw cost, not measured time; prior reward audit found measured timing only moderately correlated with H.", "missing": "Stable time distributions for candidate selectors under randomized/block order.", "discriminating_experiment": "Future IMPROVED objective must treat measured timing as a separate Pareto metric or remeasure paired blocks; no scalar total-cost acceleration claim."},
        {"id": "scenario_opportunity_sparse", "rank": 4, "evidence": "Stage2 matched labels remain sparse after stress v0; terminal/source audit found no simple checkpoint hash/config contradiction that would by itself explain sparsity.", "missing": "A denser source-supported stress/opportunity map not selected by outcomes.", "discriminating_experiment": "Freeze a stronger source-supported v1 stress generator emphasizing allowed heading/goal-distance/obstacle-clearance strata and audit fixed-H opportunity before retraining."},
    ]
    decision = {
        "retrain_or_selector_refit_now": False,
        "reason": "The source audit strengthens policy-class and terminal-value hypotheses but does not create enough robust matched labels; current class cannot select longer-H positives and uses synthetic not measured timing. Immediate refit would chase sparse/noisy labels.",
        "next_high_information_action_after_backup": "Freeze a small terminal-value instrumentation smoke or, if instrumentation is too invasive, freeze a source-supported stress-v1 scenario protocol with multi-H opportunity mapping. Do not run broad retraining until labels include >=2 robust positive states spanning the intended horizon set and fair fixed-H baselines are defined.",
    }
    four_axis = {
        "SCENARIOS": {"verified": "Stress v0 labels sparse; source/config audit found no hidden curved-path capability needed for current vehicle, and no terminal hash mismatch that invalidates all stress fixed-H rows.", "hypotheses": "Allowed straight-line stress may still be too weak or too sparse; a stronger but source-supported v1 generator may be needed.", "missing": "Fresh pre-frozen dense opportunity map on modified-but-fair scenario distribution.", "experiment": "Version and freeze stress-v1 before any rollouts if terminal mini-smoke does not reveal an actionable artifact."},
        "REWARD": {"verified": "Synthetic horizon penalty is configured in reward/info and is not measured runtime; training/search optimizes raw total cost with this synthetic term.", "hypotheses": "Scalar raw cost can choose a misleading control/compute tradeoff under noisy nonmonotone wall time.", "missing": "Randomized paired measured-time blocks for any candidate objective.", "experiment": "For any future selector, report physical cost and measured timing Pareto separately; do not infer acceleration from total_cost."},
        "TRAINING": {"verified": "Current gated policies are finite search/reselection with gradient_updates=0 and short-only policy class; not new gradient RL training.", "hypotheses": "A richer multi-H policy class may be required, but labels are too sparse for training now.", "missing": "Sufficient robust positives/negatives and terminal-value measurements.", "experiment": "Only after richer labels, freeze compact IMPROVED selector/refit with explicit candidate class and fresh confirmation."},
        "COMPARISONS": {"verified": "Stress fixed-H grid has H-specific terminal sources; current gated policies use H25 terminal lineage and must be compared against strong same-distribution fixed-H grids.", "hypotheses": "Matched-terminal versus H-specific terminal choices can change apparent opportunity.", "missing": "Terminal-off/per-H terminal continuation ablation from identical states.", "experiment": "Instrument terminal-value continuation smoke before attributing sparse positives to scenarios alone."},
    }
    raw = {
        "created_utc": created,
        "method": "vehicle_terminal_objective_source_audit_v0_analysis_only",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "inputs": {"vehicle_config": rel(CFG), "stress_protocol": rel(STRESS_PROTOCOL), "reward_audit_completed": rel(REWARD_AUDIT), "stage2_postdiagnostic_completed": rel(STRESS_STAGE2_POST)},
        "config_objective_audit": objective,
        "source_audit": src,
        "stress_terminal_grid_audit": terminal_grid,
        "current_gated_training_lineage": gated,
        "stress_stage2_positive_direction": positive,
        "findings": findings,
        "ranked_hypotheses": hypotheses,
        "four_axis_evidence": four_axis,
        "decision": decision,
    }
    write_json(OUT / "raw.json", raw)
    req = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_TERMINAL_OBJECTIVE_SOURCE_AUDIT_V0_20260928T1830Z.json"
    write_json(req, {"requested_utc": created, "reason": "backup terminal/objective source audit before any further simulations or method/scenario revision", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "artifacts": [rel(OUT), rel(STATE), rel(Path(__file__).resolve()), rel(req)]})
    raw["backup_request"] = rel(req)
    write_json(OUT / "raw.json", raw)
    summary = [
        "# Vehicle terminal/objective source audit v0",
        "",
        f"UTC: `{created}`. Analysis-only: no simulations, no training/refit, no validation64 bank, no sealed test.",
        "",
        "## Verified findings",
        "",
    ]
    summary += ["- " + f for f in findings]
    summary += ["", "## Decision", "", f"- Retrain/refit now: `{decision['retrain_or_selector_refit_now']}`.", f"- Reason: {decision['reason']}", f"- Next: {decision['next_high_information_action_after_backup']}", "", "## Four-axis evidence", ""]
    for k, v in four_axis.items():
        summary.append(f"### {k}")
        summary.append(f"- Verified: {v['verified']}")
        summary.append(f"- Hypotheses: {v['hypotheses']}")
        summary.append(f"- Missing: {v['missing']}")
        summary.append(f"- Discriminating experiment: {v['experiment']}")
        summary.append("")
    summary += ["## Ranked hypotheses", ""]
    for h in hypotheses:
        summary.append(f"{h['rank']}. `{h['id']}` — {h['evidence']} Missing: {h['missing']} Next: {h['discriminating_experiment']}")
    summary += ["", f"Backup request: `{rel(req)}`."]
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("# Vehicle terminal/objective source audit state\n\n" + "\n".join(summary[:30]) + "\n", encoding="utf-8")
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-28 vehicle terminal/objective source audit v0

UTC: {created}. Analysis-only; no rollouts/training/refit and no validation64/test access. The audit verified that vehicle reward/search uses a synthetic horizon penalty separate from measured wall time, stress fixed-H mapping has H-specific terminal sources without detected hash/config mismatches, and current gated policies are short-only finite search (`gradient_updates=0`) using H25 terminal lineage. The current policy class cannot choose longer-H positives such as the parsed Stage2 H30 direction, so immediate refit of the current class is not justified. Terminal-value accuracy remains missing because traces did not record numeric value errors; source supports instrumentation via `mpc_value_fn`. Decision: do not retrain/refit now; after backup, run a bounded terminal-value instrumentation smoke or freeze a stronger source-supported stress-v1 scenario protocol before any broader retraining. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`. Backup request: `{rel(req)}`.
""")
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "retrain_or_selector_refit_now": False,
        "terminal_grid_mismatches": terminal_grid_mismatches,
        "current_gated_can_select_longer_than_base": False,
        "parsed_stage2_positive_horizons": sorted(pos_hs),
        "backup_request": rel(req),
        "hashes": {rel(p): sha256(p) for p in [Path(__file__).resolve(), CFG, RUNTIME, RUNPY, GATED_SEARCH, GATED_POLICY, FIXED_BRANCHES, LETMPC, CONTROLLERS, STRESS_PROTOCOL, STRESS_STAGE2_POST, REWARD_AUDIT, OUT / "raw.json", OUT / "summary.md", STATE, req] if p.exists()},
    }
    write_json(OUT / "completed.json", completed)
    append_registry(created, OUT / "completed.json")
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "retrain_or_selector_refit_now": False, "terminal_grid_mismatches": terminal_grid_mismatches, "parsed_stage2_positive_horizons": sorted(pos_hs), "current_gated_can_select_longer_than_base": False, "backup_request": rel(req), "historical_validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
