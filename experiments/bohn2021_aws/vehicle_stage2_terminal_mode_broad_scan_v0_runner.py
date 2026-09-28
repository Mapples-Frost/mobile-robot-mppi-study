#!/usr/bin/env python3
"""Vehicle Stage2 terminal-mode broad scan v0.

Development-only diagnostic frozen by
`research_artifacts/aws_protocols/vehicle_stage2_terminal_mode_broad_scan_v0_plan_20260928T1915Z.json`.

This reuses the terminal/objective smoke instrumentation to scan the same 12
Stage2 target states under per-H, H15 and zero terminal modes for H10/H25/H30/H35
plus the per-H H15 reference. It is not validation, not final test, and performs
no training/refit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_stage2_terminal_objective_smoke_v0 as smoke  # noqa:E402
import vehicle_stress_scenario_opportunity_probe_v0_runner as stage1_runner  # noqa:E402

OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_stage2_terminal_mode_broad_scan_v0_20260928T1915Z"
STATE = ROOT / "research_artifacts/aws_state/vehicle_stage2_terminal_mode_broad_scan_v0_20260928T1915Z.md"
PLAN = ROOT / "research_artifacts/aws_protocols/vehicle_stage2_terminal_mode_broad_scan_v0_plan_20260928T1915Z.json"
TERMINAL_SMOKE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stage2_terminal_objective_smoke_v0_20260928T1900Z/completed.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-stage2-terminal-mode-broad-scan-v0-20260928T1915Z"
SOURCE_WRITTEN_UTC = "2026-09-28T19:16:30+00:00"


class ContractError(RuntimeError):
    pass


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    return smoke.write_json(p, obj)


def sha256(p: Path) -> str:
    return smoke.sha256(p)


def parse_time(v: Any) -> Optional[dt.datetime]:
    return smoke.parse_time(v)


def build_schedule(plan: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
    gen = plan["run_plan_generator"]
    hs = list(gen["for_each_state"][1]["for_each_horizon"])
    modes = list(gen["for_each_state"][1]["for_each_terminal_mode"])
    schedule = []
    idx = 0
    for spec in plan["state_specs"]:
        base = {"state_id": spec["state_id"], "case": int(spec["case"]), "branch_step": int(spec["branch_step"]), "role": spec["role"]}
        item = dict(base)
        item.update({"execution_index": idx, "horizon": 15, "terminal_mode": "per_h"})
        schedule.append(item)
        idx += 1
        for h in hs:
            for mode in modes:
                item = dict(base)
                item.update({"execution_index": idx, "horizon": int(h), "terminal_mode": str(mode)})
                schedule.append(item)
                idx += 1
    expected = int(gen["episode_count_exact"])
    if len(schedule) != expected:
        raise ContractError("schedule length %d != expected %d" % (len(schedule), expected))
    return schedule


def summarize_broad(episodes: Sequence[Mapping[str, Any]], analysis: Mapping[str, Any], plan: Mapping[str, Any]) -> Dict[str, Any]:
    state_to_case = {s["state_id"]: int(s["case"]) for s in plan["state_specs"]}
    rows = list(analysis.get("comparison_rows") or [])
    flips = []
    for x in analysis.get("material_label_flips_under_terminal_ablation") or []:
        y = dict(x)
        y["case"] = state_to_case.get(y.get("state_id"))
        flips.append(y)
    material_rows = [r for r in rows if r.get("material_positive_vs_per_h_H15")]
    material_states = sorted({str(r["state_id"]) for r in material_rows})
    material_cases = sorted({int(r["case"]) for r in material_rows if r.get("case") is not None})
    flip_states = sorted({str(r["state_id"]) for r in flips})
    flip_cases = sorted({int(r["case"]) for r in flips if r.get("case") is not None})
    positives_by_mode: Dict[str, int] = {}
    positives_by_horizon: Dict[str, int] = {}
    positives_by_case: Dict[str, int] = {}
    for r in material_rows:
        positives_by_mode[str(r["terminal_mode"])] = positives_by_mode.get(str(r["terminal_mode"]), 0) + 1
        positives_by_horizon[str(r["horizon"])] = positives_by_horizon.get(str(r["horizon"]), 0) + 1
        positives_by_case[str(r["case"])] = positives_by_case.get(str(r["case"]), 0) + 1
    # Sort top wins/harms by physical gain to preserve unfavorable evidence.
    top_wins = sorted(rows, key=lambda r: float(r.get("gain_vs_per_h_H15_physical", 0.0)), reverse=True)[:12]
    top_harms = sorted(rows, key=lambda r: float(r.get("gain_vs_per_h_H15_physical", 0.0)))[:12]
    if len(set(material_cases) | set(flip_cases)) >= 2:
        next_decision = "freeze_terminal_mode_or_terminal_refit_label_protocol_before_selector_training"
        gate = True
    else:
        next_decision = "do_not_refit_selector; freeze_source_supported_stress_v1_opportunity_protocol"
        gate = False
    return {
        "material_positive_row_count": len(material_rows),
        "material_positive_state_count": len(material_states),
        "material_positive_case_count": len(material_cases),
        "material_positive_states": material_states,
        "material_positive_cases": material_cases,
        "material_positive_by_mode": positives_by_mode,
        "material_positive_by_horizon": positives_by_horizon,
        "material_positive_by_case": positives_by_case,
        "terminal_label_flip_count": len(flips),
        "terminal_label_flip_state_count": len(flip_states),
        "terminal_label_flip_case_count": len(flip_cases),
        "terminal_label_flip_states": flip_states,
        "terminal_label_flip_cases": flip_cases,
        "terminal_label_flips_enriched": flips,
        "top_physical_wins": top_wins,
        "top_physical_harms": top_harms,
        "predeclared_dense_artifact_gate_pass": gate,
        "predeclared_next_decision": next_decision,
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    b = raw["broad_summary"]
    lines = [
        "# Vehicle Stage2 terminal-mode broad scan v0",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only; no training/refit, no validation64 bank, no sealed test.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Predeclared decision",
        "",
        f"- Dense terminal/artifact gate pass: `{b['predeclared_dense_artifact_gate_pass']}`.",
        f"- Next decision: `{b['predeclared_next_decision']}`.",
        "",
        "## Counts",
        "",
        f"- Material positive rows/states/cases: `{b['material_positive_row_count']}` / `{b['material_positive_state_count']}` / `{b['material_positive_case_count']}`.",
        f"- Material positive cases: `{b['material_positive_cases']}`; states: `{b['material_positive_states']}`.",
        f"- Positives by mode: `{b['material_positive_by_mode']}`.",
        f"- Positives by horizon: `{b['material_positive_by_horizon']}`.",
        f"- Terminal label flips rows/states/cases: `{b['terminal_label_flip_count']}` / `{b['terminal_label_flip_state_count']}` / `{b['terminal_label_flip_case_count']}`.",
        f"- Terminal label flip cases: `{b['terminal_label_flip_cases']}`; states: `{b['terminal_label_flip_states']}`.",
        "",
        "## Top physical wins",
        "",
        "| case | state | H | terminal mode | phys gain | total gain | success | constraint | material |",
        "|---:|---|---:|---|---:|---:|---|---|---|",
    ]
    for r in b["top_physical_wins"]:
        lines.append("| %s | `%s` | %s | `%s` | %.6g | %.6g | `%s` | `%s` | `%s` |" % (r.get("case"), r.get("state_id"), r.get("horizon"), r.get("terminal_mode"), float(r.get("gain_vs_per_h_H15_physical", 0.0)), float(r.get("gain_vs_per_h_H15_total", 0.0)), r.get("success"), r.get("constraint"), r.get("material_positive_vs_per_h_H15")))
    lines += ["", "## Top physical harms", "", "| case | state | H | terminal mode | phys gain | total gain | success | constraint | material |", "|---:|---|---:|---|---:|---:|---|---|---|"]
    for r in b["top_physical_harms"]:
        lines.append("| %s | `%s` | %s | `%s` | %.6g | %.6g | `%s` | `%s` | `%s` |" % (r.get("case"), r.get("state_id"), r.get("horizon"), r.get("terminal_mode"), float(r.get("gain_vs_per_h_H15_physical", 0.0)), float(r.get("gain_vs_per_h_H15_total", 0.0)), r.get("success"), r.get("constraint"), r.get("material_positive_vs_per_h_H15")))
    lines += ["", f"Backup request: `{raw['backup_request']}`."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    b = raw["broad_summary"]
    block = f"""<!-- {MARKER} -->
## 2026-09-28 vehicle Stage2 terminal-mode broad scan v0

UTC: {raw['created_utc']}. Development-only broad terminal-mode scan completed: {raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps. No validation64/test access and no training/refit. Material positives rows/states/cases={b['material_positive_row_count']}/{b['material_positive_state_count']}/{b['material_positive_case_count']}; terminal label flips rows/states/cases={b['terminal_label_flip_count']}/{b['terminal_label_flip_state_count']}/{b['terminal_label_flip_case_count']}. Predeclared gate={b['predeclared_dense_artifact_gate_pass']}; next={b['predeclared_next_decision']}. Summary: `{rel(OUT / 'summary.md')}`. Backup request: `{raw['backup_request']}`.
"""
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        if p.exists():
            old = p.read_text(encoding="utf-8")
            if MARKER not in old:
                p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-backup-proof", required=True, type=Path)
    ap.add_argument("--i-accept-development-terminal-broad-scan", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_terminal_broad_scan:
        raise ContractError("explicit --i-accept-development-terminal-broad-scan required")
    if OUT.exists() and (OUT / "completed.json").exists():
        smoke.verify_done(OUT / "completed.json", check_hashes=True)
        raise SystemExit("broad scan already completed; refusing rerun")
    if OUT.exists() and any(p.name not in ("run.lock",) for p in OUT.iterdir()):
        raise ContractError("partial output exists; inspect before rerun: %s" % rel(OUT))
    terminal_done = smoke.verify_done(TERMINAL_SMOKE_DONE, check_hashes=False)
    plan = read_json(PLAN)
    min_times = [parse_time(terminal_done.get("created_utc")), parse_time(plan.get("created_utc")), parse_time(SOURCE_WRITTEN_UTC)]
    min_time = max(t for t in min_times if t is not None)
    backup = smoke.verify_backup(args.input_backup_proof, min_time)
    created_start = dt.datetime.now(dt.timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {"started_utc": created_start, "pid": os.getpid(), "method": "vehicle_stage2_terminal_mode_broad_scan_v0", "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "training_gradient_steps": 0, "plan": rel(PLAN)})
    preflight = stage1_runner.runtime_preflight()
    write_json(OUT / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %r" % preflight)
    stage1_runner.base.v1.latency_verify()
    bank = read_json(smoke.STAGE1_BANK)
    selected_cases = bank["selected_cases"]
    stage1_protocol = read_json(smoke.STAGE1_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid_for_stage1(stage1_protocol)
    write_json(OUT / "terminal_sources.json", terminal_receipts)
    schedule = list(build_schedule(plan))
    write_json(OUT / "schedule.json", {"episodes": schedule, "prefix_horizon": int(plan["run_plan_generator"]["prefix_horizon"]), "max_steps": int(plan["run_plan_generator"]["max_steps_per_episode"]), "plan": rel(PLAN), "plan_sha256": sha256(PLAN)})
    # Repoint the imported smoke instrumentation to this output directory and plan.
    smoke.OUT = OUT
    smoke.MAX_STEPS = int(plan["run_plan_generator"]["max_steps_per_episode"])
    episodes = []
    for item in schedule:
        summary = smoke.run_one(item, selected_cases[int(item["case"])], terminals, terminal_receipts)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "episodes_done": len(episodes), "episodes_expected": len(schedule), "control_steps_done": int(sum(int(e["steps"]) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "case", "branch_step", "branch_horizon", "terminal_mode", "steps", "success", "termination")}, "historical_validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(OUT / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e["steps"]) for e in episodes))
    cap = int(plan["run_plan_generator"]["control_step_cap"])
    if len(episodes) != len(schedule) or control_steps > cap:
        raise ContractError("budget violation")
    analysis = smoke.analyze(episodes)
    broad = summarize_broad(episodes, analysis, plan)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_STAGE2_TERMINAL_MODE_BROAD_SCAN_V0_%s.json" % created.replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created,
        "started_utc": created_start,
        "method": "vehicle_stage2_terminal_mode_broad_scan_v0",
        "classification": "development_IMPROVED_terminal_mode_opportunity_mapping_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "input_backup_proof": backup,
        "plan": {"path": rel(PLAN), "sha256": sha256(PLAN), "content": plan},
        "inputs": {"terminal_smoke_completed": rel(TERMINAL_SMOKE_DONE), "terminal_smoke_completed_sha256": sha256(TERMINAL_SMOKE_DONE), "stage1_bank": rel(smoke.STAGE1_BANK), "stage1_bank_sha256": sha256(smoke.STAGE1_BANK), "stage1_protocol": rel(smoke.STAGE1_PROTOCOL), "stage1_protocol_sha256": sha256(smoke.STAGE1_PROTOCOL)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": {"episodes_exact": len(schedule), "control_step_upper_bound": cap, "new_training_episodes": 0, "new_gradient_steps": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "environment_constructions": len(episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "terminal_sources": terminal_receipts,
        "schedule": schedule,
        "episodes": episodes,
        "analysis": analysis,
        "broad_summary": broad,
        "interpretation_limits": ["development-only scan over states selected before this protocol from prior development evidence", "not validation/model selection", "not final test", "not sufficient alone for reproduction claim", "preserve harms and failures", "terminal off keeps the same NLP structure by zeroing terminal weights"],
    }
    write_json(req, {"requested_utc": created, "reason": "backup broad terminal-mode scan before terminal-refit/scenario-v1/source changes", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "artifacts": [rel(OUT), rel(STATE), rel(Path(__file__).resolve()), rel(PLAN), rel(req)]})
    raw["backup_request"] = rel(req)
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("# Vehicle Stage2 terminal-mode broad scan state (%s)\n\nCompleted %d episodes / %d control steps. No validation64/test/training. Material positives rows/states/cases=%d/%d/%d; terminal flips rows/states/cases=%d/%d/%d. Predeclared gate=%s; next=%s. Backup required before next simulation or source change.\n" % (created, len(episodes), control_steps, broad["material_positive_row_count"], broad["material_positive_state_count"], broad["material_positive_case_count"], broad["terminal_label_flip_count"], broad["terminal_label_flip_state_count"], broad["terminal_label_flip_case_count"], broad["predeclared_dense_artifact_gate_pass"], broad["predeclared_next_decision"]), encoding="utf-8")
    append_docs(raw)
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE, Path(__file__).resolve(), PLAN, req, args.input_backup_proof]
    write_json(OUT / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created, "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "backup_request": rel(req), "headline": {"material_positive_row_count": broad["material_positive_row_count"], "material_positive_state_count": broad["material_positive_state_count"], "material_positive_case_count": broad["material_positive_case_count"], "terminal_label_flip_count": broad["terminal_label_flip_count"], "terminal_label_flip_state_count": broad["terminal_label_flip_state_count"], "terminal_label_flip_case_count": broad["terminal_label_flip_case_count"], "predeclared_dense_artifact_gate_pass": broad["predeclared_dense_artifact_gate_pass"], "next_action": broad["predeclared_next_decision"]}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "broad_summary": broad, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {"failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_episodes": 0, "new_gradient_steps": 0, "next_recovery_hint": "Preserve partial output. Audit completed episode summaries before deciding whether a source repair/rerun is valid."})
        raise
