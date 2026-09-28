#!/usr/bin/env python3
"""Vehicle scenario opportunity / environment capability diagnostic V0.

Metadata/source-only audit after the V2b actual-time smoke.  It reads registered
source/configuration files and already-produced non-test diagnostic metadata, but
performs no simulation, training, validation-bank replay, or sealed-test access.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T0950Z"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z.md"
BACKUP_PATH = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SCENARIO_OPPORTUNITY_CAPABILITY_DIAGNOSTIC_V0_20260928T0950Z.json"
NEXT_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v0_after_capability_audit_20260928.md"

SRC_ROOT = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/sources/gym-horizon/gym_let_mpc"
LET_MPC = SRC_ROOT / "let_mpc.py"
SIMULATOR = SRC_ROOT / "simulator.py"
CONTROLLERS = SRC_ROOT / "controllers.py"
VEHICLE_CONFIG = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/configs/vehicle.json"
POSTDIAG_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_smoke_postdiagnostic_20260928T0935Z/completed.json"
POSTDIAG_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_smoke_postdiagnostic_20260928T0935Z/raw.json"
MARKER = "vehicle-scenario-opportunity-capability-diagnostic-v0-20260928T0950Z"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def source_hits(path: Path, needles: Iterable[str], context: int = 0) -> List[Dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits: List[Dict[str, Any]] = []
    for needle in needles:
        for idx, line in enumerate(lines, start=1):
            if needle in line:
                start = max(1, idx - context)
                end = min(len(lines), idx + context)
                hits.append({
                    "file": rel(path),
                    "line": idx,
                    "needle": needle,
                    "excerpt": "\n".join(f"{j}: {lines[j-1]}" for j in range(start, end + 1)),
                })
                break
    return hits


def config_summary(config: Mapping[str, Any]) -> Dict[str, Any]:
    env = config["environment"]
    mpc = config["mpc"]
    plant_model = config["plant"]["model"]
    randomize = env.get("randomize", {})
    states = plant_model.get("states", {})
    process_noise_states = {name: props.get("W") for name, props in states.items() if "W" in props}
    return {
        "max_steps": env.get("max_steps"),
        "environment_end_on_constraint_violation_config_value": env.get("end_on_constraint_violation"),
        "randomize_state": randomize.get("state", {}),
        "randomize_reference": randomize.get("reference", {}),
        "randomize_constraints": randomize.get("constraints", {}),
        "randomize_model": randomize.get("model", {}),
        "plant_states": sorted(plant_model.get("states", {}).keys()),
        "plant_inputs": sorted(plant_model.get("inputs", {}).keys()),
        "process_noise_states_in_vehicle_config": process_noise_states,
        "mpc_type": mpc.get("type"),
        "mpc_horizon_max": mpc.get("params", {}).get("n_horizon"),
        "mpc_n_objects": mpc.get("params", {}).get("n_objects"),
        "input_constraints": [c for c in mpc.get("constraints", []) if c.get("type") == "_u"],
        "reward_expression": env.get("reward", {}).get("expression"),
        "info_reward_terms": env.get("info", {}).get("reward", {}),
    }


def make_next_protocol(created: str) -> None:
    text = f"""# Vehicle fixed-H opportunity probe V0 after capability audit

Frozen draft UTC: {created}.

## Status

Draft only; do not execute until the postdiagnostic/capability-audit artifacts are externally backed up and this protocol is explicitly frozen in a later cycle.

## Purpose

Determine whether the source-supported vehicle task contains state-dependent prediction-horizon opportunity before allocating another long adaptive-controller validation campaign.

## Inputs and exclusions

- No sealed final test.
- Do not reopen the historical validation64 bank for this probe.
- Use a fresh diagnostic bank only, generated from source-supported vehicle factors.
- Do not add unsupported curvature, initial-speed, or robust-uncertainty factors.

## Source-supported factors from V0 audit

Allowed without source modification: initial x/y/theta, theta_r, traj_steps, obstacle radii/centres/noise seeds through reset/tvp/reference controls, input constraints, and model overrides already accepted by reset.  Highest-fidelity subset: canonical straight-line path with theta_r and traj_steps variation plus the built-in 3-obstacle generator.

Excluded unless a separate method/environment amendment is written: curved trajectory benchmarks, direct initial-speed state, robust MPC uncertainty, and stochastic process-noise stress in the current vehicle config.

## Candidate bounded design

- Fresh diagnostic split: `vehicle_fixed_h_opportunity_probe_v0_fresh_no_validation64_no_test`.
- Proposed cases: 8 to 12 generated cases, stratified after generation by path length, heading change, and nearest obstacle clearance; include easy and hard strata instead of hand-picking only favorable cases.
- Horizons: fixed H in {{5, 10, 15, 20, 25, 30, 35, 40, 50}} using the same terminal/value source rules already used for fixed-H diagnostics.
- Metrics: physical+constraint cost, total synthetic reward cost separately, success, constraints, initial/final solver failures, steps, actual decision/solver wall time, deadline exceedance, and reset/construction timing.
- Budget ceiling: <= 120 episodes and <= 18,000 control steps for the first probe.
- Acceptance for adaptive-opportunity existence: at least two distinct scenario strata show different fixed-H Pareto optima or materially different cost/time slopes; otherwise document strong fixed-H dominance or weak opportunity and prioritize training/selection only if counterfactual opportunity exists.

## Interpretation

This probe, if run, is development evidence only.  It cannot support a final success claim and cannot be used to tune sealed-test outcomes.
"""
    NEXT_PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    NEXT_PROTOCOL.write_text(text, encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## Vehicle scenario-opportunity/capability diagnostic V0\n\n"
        f"UTC: {raw['created_utc']}. Metadata/source-only audit; no simulations, training, validation64 bank access, or sealed-test access. "
        "Finding: the vehicle environment supports straight-line goal/path length and heading variation plus three obstacle constraints/noisy forecasts, but the current registered vehicle config fixes initial x/y/theta, lacks plant process noise/model randomization, has no direct initial-speed state, and has no native curved-path generator. "
        f"Decision: {raw['decision']['headline']} Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    config = read_json(VEHICLE_CONFIG)
    cfg = config_summary(config)
    post_completed = read_json(POSTDIAG_COMPLETED)
    post_raw = read_json(POSTDIAG_RAW)

    evidence = []
    evidence += source_hits(LET_MPC, [
        "def reset(self, state=None, reference=None, constraint=None, model=None, process_noise=None, tvp=None):",
        "sampled_state = self.sample_state()",
        "self.theta_r = self.np_random.uniform(sampled_state[\"theta\"] - np.radians(45), sampled_state[\"theta\"] + np.radians(45))",
        "self.traj_steps = self.np_random.randint(60, 100)",
        "self.trajectory_goal_x = np.cos(self.theta_r) * self.traj_steps * self.control_system.controller.u_s_ref",
        "np.linspace(sampled_state[\"x\"], self.trajectory_goal_x, self.traj_steps)",
        "obj_feasible_points = [traj_i for traj_i in range(self.trajectory[\"n_steps\"])",
        "sample_state(self):",
        "sample_reference(self):",
        "sample_model(self):",
        "create_dataset(self, n_scenarios):",
        "np.linalg.norm(np.array([self.trajectory_goal_x, self.trajectory_goal_y])",
    ])
    evidence += source_hits(SIMULATOR, [
        "def reset(self, state=None, reference=None, constraint=None, model=None, process_noise=None, tvp=None):",
        "def _get_process_noise(self):",
        "if \"W\" in state:",
        "if \"uncertainty\" in config:",
    ])
    evidence += source_hits(CONTROLLERS, [
        "class TTAHMPC(AHMPC):",
        "self.n_objects = mpc_config[\"params\"].pop(\"n_objects\", 0)",
        "self.obj_kw = data = {\"r\": {\"u_low\": 0.5, \"u_high\": 1",
        "mpc_config[\"constraints\"].append({",
        "forecast_noise[self._n_noiseless:n_horizon] = np.linspace(0, self.object_noise_seed",
        "self.u_s_ref = 3 * self.mpc_config[\"params\"][\"t_step\"]",
        "def get_obj_distance(self, state, obj_i):",
    ])

    capabilities = [
        {
            "factor": "initial pose x/y/theta",
            "support": "supported by reset(state=...) and environment.randomize.state",
            "current_reference_use": "registered vehicle config fixes x=0, y=0, theta=0 via uniform low=high=0",
            "fidelity_note": "varying these is a scenario-distribution change but source-supported",
        },
        {
            "factor": "path heading theta_r",
            "support": "supported as reset reference theta_r or sampled uniformly within initial theta ±45 degrees",
            "current_reference_use": "implicit TTAHMPC reset sampling; straight-line path only",
            "fidelity_note": "safe to stratify within ±45 degrees before considering broader shifts",
        },
        {
            "factor": "goal distance / episode phase via traj_steps",
            "support": "supported as reset reference traj_steps or sampled randint(60,100)",
            "current_reference_use": "straight-line trajectory length roughly traj_steps*u_s_ref with u_s_ref=3*t_step",
            "fidelity_note": "source-supported; near/far splits can be diagnostic without changing dynamics",
        },
        {
            "factor": "obstacle proximity and radius",
            "support": "three TTAHMPC objects; radii/positions stored as TVPs, auto-generated around the straight trajectory and overrideable via tvp",
            "current_reference_use": "n_objects=3; obstacle constraints added by TTAHMPC; hard termination configured by source append",
            "fidelity_note": "stratify naturally generated clearance first; direct obstacle placement is a stronger diagnostic shift",
        },
        {
            "factor": "forecast/model uncertainty of obstacles",
            "support": "object_noise_seed and OU forecast noise are implemented for obstacle TVPs",
            "current_reference_use": "used by TTAHMPC forecast construction; true obstacle centres remain fixed within an episode",
            "fidelity_note": "source-supported as part of current controller forecast model",
        },
        {
            "factor": "curved/path-shape changes",
            "support": "not natively generated: reset constructs trajectory_x/y with np.linspace straight line; low-level tvp override would bypass normal goal/path bookkeeping",
            "current_reference_use": "not used",
            "fidelity_note": "do not add to a benchmark without a separate environment/protocol amendment and verification",
        },
        {
            "factor": "initial speed as plant state",
            "support": "not supported in current vehicle model; states are x, y, theta and inputs are u_s, u_omega",
            "current_reference_use": "previous input after reset warmup may be observed but speed is not an initial state",
            "fidelity_note": "do not treat initial speed as a reset factor without changing model/controller semantics",
        },
        {
            "factor": "plant process noise / disturbances",
            "support": "ControlSystem supports state W noise only if configured; registered vehicle states have no W entries",
            "current_reference_use": "none in the registered vehicle config",
            "fidelity_note": "adding W would be a distribution/dynamics shift requiring a separate protocol",
        },
        {
            "factor": "robust MPC uncertainty",
            "support": "explicit NotImplemented in simulator/MPC initialization for uncertainty/n_robust>0",
            "current_reference_use": "n_robust=0",
            "fidelity_note": "unsupported; exclude from immediate scenario redesign",
        },
    ]

    opportunity_diagnosis = {
        "observed_v2b_smoke": {
            "hard_pass": post_completed.get("hard_pass"),
            "episodes": post_raw.get("smoke", {}).get("episodes"),
            "control_steps": post_raw.get("smoke", {}).get("control_steps"),
            "all_success_equal_steps_per_arm": post_raw.get("smoke", {}).get("all_success_equal_steps_per_arm"),
            "near_zero_selected_physical_delta_sum": post_raw.get("smoke", {}).get("near_zero_selected_physical_delta_sum"),
            "weighted_selected_vs_fixed_decision_ratio": post_raw.get("timing", {}).get("weighted_selected_vs_fixed_decision_ratio"),
            "adaptive_seed_ratios": post_raw.get("timing", {}).get("adaptive_seed_selected_vs_fixed_decision_ratios"),
        },
        "source_level_interpretation": [
            "The canonical vehicle generator is mostly straight-line tracking with obstacle avoidance: horizon choice may be consequential near obstacle/heading transients but not during long easy steady tracking segments.",
            "The registered config fixes initial pose and does not vary plant parameters/noise, which can reduce state-dependent horizon-opportunity diversity.",
            "Because V2b smoke was all-success/equal-step and physical deltas were nearly zero, the next informative experiment is a fixed-H opportunity probe across source-supported strata, not another long adaptive validation shard by default.",
        ],
    }

    decision = {
        "headline": "Run a bounded fixed-H opportunity probe only after backup; do not redesign scenarios or resume long adaptive validation until fixed-H Pareto opportunity is measured.",
        "verified_causes": [
            "Vehicle source supports straight-line heading/length variation, obstacle constraints, and obstacle forecast noise.",
            "Registered vehicle config fixes initial x/y/theta and lacks model/process-noise randomization.",
            "Curved paths, direct initial speed, and robust uncertainty are not supported by the current paper-aligned vehicle model without a separate amendment.",
        ],
        "missing_evidence": [
            "No regime-stratified fixed-H Pareto map from identical states with actual timing has been produced yet.",
            "No evidence yet distinguishes whether weak adaptation is caused primarily by easy scenario distribution or by finite selector/value limitations when real opportunity exists.",
        ],
        "next_action_after_backup": "freeze_then_run_vehicle_fixed_h_opportunity_probe_v0_fresh_diagnostic_bank_no_validation64_no_test",
        "do_not_do_next_by_default": "do_not_resume_unchanged_risk_reselection_or_v2b_long_devval_shards",
    }

    make_next_protocol(created)
    raw: Dict[str, Any] = {
        "created_utc": created,
        "method": "metadata_source_only_vehicle_scenario_opportunity_capability_diagnostic_v0",
        "formal_scientific_evidence": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "inputs": {
            rel(VEHICLE_CONFIG): sha256(VEHICLE_CONFIG),
            rel(LET_MPC): sha256(LET_MPC),
            rel(SIMULATOR): sha256(SIMULATOR),
            rel(CONTROLLERS): sha256(CONTROLLERS),
            rel(POSTDIAG_COMPLETED): sha256(POSTDIAG_COMPLETED),
            rel(POSTDIAG_RAW): sha256(POSTDIAG_RAW),
            rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        },
        "vehicle_config_summary": cfg,
        "source_evidence": evidence,
        "capability_table": capabilities,
        "opportunity_diagnosis": opportunity_diagnosis,
        "decision": decision,
        "next_protocol": rel(NEXT_PROTOCOL),
    }

    lines = [
        "# Vehicle scenario-opportunity/capability diagnostic V0",
        "",
        f"UTC: `{created}`. Metadata/source-only; no simulations, no training, no validation64-bank access, no sealed-test access.",
        "",
        "## Why this diagnostic was run",
        "",
        "The V2b actual-time smoke passed engineering checks but had all-success/equal-step episodes, near-zero physical deltas, seed0 fixed fallback, and mixed measured timing for the two adaptive seeds. Before another long adaptive validation run, this audit checks which scenario factors are actually supported by the registered vehicle environment.",
        "",
        "## Capability table",
        "",
        "| factor | source support | current reference use | fidelity / revision note |",
        "|---|---|---|---|",
    ]
    for row in capabilities:
        lines.append(f"| {row['factor']} | {row['support']} | {row['current_reference_use']} | {row['fidelity_note']} |")
    lines += [
        "",
        "## Registered vehicle config summary",
        "",
        f"- max_steps: `{cfg['max_steps']}`; MPC type/horizon: `{cfg['mpc_type']}` / `{cfg['mpc_horizon_max']}`; n_objects: `{cfg['mpc_n_objects']}`.",
        f"- plant states: `{cfg['plant_states']}`; plant inputs: `{cfg['plant_inputs']}`.",
        f"- randomize.state: `{cfg['randomize_state']}`.",
        f"- randomize.reference/model/constraints: `{cfg['randomize_reference']}` / `{cfg['randomize_model']}` / `{cfg['randomize_constraints']}`.",
        f"- process-noise states in vehicle config: `{cfg['process_noise_states_in_vehicle_config']}`.",
        f"- reward: `{cfg['reward_expression']}`; info reward terms: `{cfg['info_reward_terms']}`.",
        "",
        "## Source-backed interpretation",
        "",
        "- Current high-fidelity vehicle benchmark is straight-line tracking with three obstacle constraints and noisy obstacle forecasts, not curved-path following.",
        "- Horizon choice is most plausibly consequential during early heading transients and obstacle-proximity regimes; steady straight tracking can make strong fixed H25/H30 dominate.",
        "- Initial speed and robust uncertainty should not be added as factors in the next probe because they are not supported by the current vehicle state/config without changing model semantics.",
        "- The immediate missing evidence is a regime-stratified fixed-H Pareto map from identical fresh states with actual wall time and continuation/control cost.",
        "",
        "## Source evidence excerpts",
        "",
    ]
    for hit in evidence[:40]:
        lines.append(f"- `{hit['file']}:{hit['line']}` contains `{hit['needle']}`")
    lines += [
        "",
        "## Decision",
        "",
        f"{decision['headline']}",
        "",
        f"Next draft protocol: `{rel(NEXT_PROTOCOL)}`.",
        f"Backup request: `{rel(BACKUP_PATH)}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT_DIR / "raw.json", raw)
    write_json(BACKUP_PATH, {
        "requested_utc": created,
        "reason": "backup scenario-opportunity capability audit and fixed-H opportunity-probe draft before any further simulations",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(BACKUP_PATH), rel(NEXT_PROTOCOL), rel(Path(__file__).resolve())],
    })
    state_text = (
        f"# Vehicle scenario-opportunity/capability diagnostic V0 state ({created})\n\n"
        f"Decision: {decision['headline']}\n\n"
        "No simulations/training/validation64/test access. Backup is required before more simulations. "
        f"Next: {decision['next_action_after_backup']} using draft protocol {rel(NEXT_PROTOCOL)}.\n"
    )
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(state_text, encoding="utf-8")
    append_docs(raw)
    files = [OUT_DIR / "summary.md", OUT_DIR / "raw.json", STATE_PATH, BACKUP_PATH, NEXT_PROTOCOL, Path(__file__).resolve(), VEHICLE_CONFIG, LET_MPC, SIMULATOR, CONTROLLERS]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "backup_request": rel(BACKUP_PATH),
        "next_action_after_backup": decision["next_action_after_backup"],
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
        "headline": decision["headline"],
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "backup_request": rel(BACKUP_PATH),
        "next_protocol": rel(NEXT_PROTOCOL),
        "decision": decision["headline"],
        "new_rollouts": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
