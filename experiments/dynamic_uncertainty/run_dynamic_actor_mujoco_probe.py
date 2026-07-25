"""Run paired development-seed MuJoCo probes for a dynamic Actor checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.run_rl_hss_combined_safety_probe import (
    _summary,
    configure_combined_job,
)
from experiments.dynamic_uncertainty.run_rl_hss_exact_fallback_probe import (
    STAGE5_PROTOCOL,
)
from experiments.dynamic_uncertainty.run_rl_hss_stage4 import (
    _complete,
    _json,
    _mapping,
    _resolve,
    _write_json,
)
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_SOURCE = (
    "results/research_platform/rl/gate1_direct_control_multidomain_screen_l175/"
    "checkpoints/best.pt"
)
DEFAULT_CANDIDATE = (
    "research_artifacts/dynamic_actor_correction_development_v2/"
    "checkpoints/best.pt"
)

ARM_ORDERS = {
    "source_first": ("source_actor", "dynamic_actor"),
    "candidate_first": ("dynamic_actor", "source_actor"),
}


def _resolve_arm_order(value):
    try:
        return ARM_ORDERS[str(value)]
    except KeyError as exc:
        raise ValueError("unknown dynamic Actor arm order") from exc


def _extended_summary(metrics):
    result = _summary(metrics)
    result.update({
        "guided_minus_gaussian_cost_min_mean": float(metrics.get(
            "paper_guided_minus_gaussian_cost_min_mean", 0.0
        )),
        "guided_minus_gaussian_cost_mean_mean": float(metrics.get(
            "paper_guided_minus_gaussian_cost_mean_mean", 0.0
        )),
        "guided_cost_observed_fraction": float(metrics.get(
            "paper_guided_cost_observed_fraction", 0.0
        )),
        "proposal_disadvantage_fraction": float(metrics.get(
            "reliability_proposal_advantage_disadvantage_fraction", 0.0
        )),
        "same_cycle_filter_enabled_fraction": float(metrics.get(
            "paper_same_cycle_guided_cost_filter_enabled_fraction", 0.0
        )),
        "same_cycle_filter_active_fraction": float(metrics.get(
            "paper_same_cycle_guided_cost_filter_active_fraction", 0.0
        )),
        "same_cycle_filter_iterations_total": int(metrics.get(
            "paper_same_cycle_guided_cost_filter_iterations_total", 0
        )),
        "same_cycle_filtered_candidates_total": int(metrics.get(
            "paper_same_cycle_guided_filtered_candidates_total", 0
        )),
        "pareto_forward_commit_steps": int(metrics.get(
            "probabilistic_obstacle_pareto_forward_commit_steps", 0
        )),
        "low_risk_forward_commit_steps": int(metrics.get(
            "probabilistic_obstacle_low_risk_forward_commit_steps", 0
        )),
        "planner_temporal_escape_active_steps": int(metrics.get(
            "planner_temporal_escape_active_steps", 0
        )),
        "traversal_window_safe_steps": int(metrics.get(
            "probabilistic_obstacle_traversal_window_safe_steps", 0
        )),
        "traversal_forecast_sufficient_steps": int(metrics.get(
            "probabilistic_obstacle_traversal_forecast_sufficient_steps", 0
        )),
        "traversal_commit_active_steps": int(metrics.get(
            "probabilistic_obstacle_traversal_commit_active_steps", 0
        )),
        "traversal_commit_started_steps": int(metrics.get(
            "probabilistic_obstacle_traversal_commit_started_steps", 0
        )),
        "traversal_commit_completed_steps": int(metrics.get(
            "probabilistic_obstacle_traversal_commit_completed_steps", 0
        )),
        "traversal_commit_cancelled_steps": int(metrics.get(
            "probabilistic_obstacle_traversal_commit_cancelled_steps", 0
        )),
        "traversal_commit_cancelled_by_temporal_closing_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_closing_steps",
                0,
            )
        ),
        "traversal_commit_cancelled_by_temporal_exit_deadline_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_commit_cancelled_by_temporal_exit_deadline_guard_steps",
                0,
            )
        ),
        "traversal_exit_deadline_retreat_escape_active_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_transaction_active_steps",
                0,
            )
        ),
        "traversal_exit_deadline_retreat_escape_latched_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_latched_steps",
                0,
            )
        ),
        "traversal_exit_deadline_retreat_escape_reused_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_reused_steps",
                0,
            )
        ),
        "traversal_commit_admission_exit_deadline_rejected_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_commit_admission_exit_deadline_rejected_steps",
                0,
            )
        ),
        "traversal_commit_admission_exit_deadline_hold_requested_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_requested_steps",
                0,
            )
        ),
        "traversal_commit_admission_exit_deadline_hold_hard_risk_override_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_commit_admission_exit_deadline_hold_hard_risk_override_steps",
                0,
            )
        ),
        "traversal_commit_admission_exit_deadline_forward_lattice_filtered_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_commit_admission_exit_deadline_forward_lattice_filtered_steps",
                0,
            )
        ),
        "traversal_retreat_requested_steps": int(metrics.get(
            "probabilistic_obstacle_traversal_retreat_requested_steps", 0
        )),
        "traversal_retreat_completed_steps": int(metrics.get(
            "probabilistic_obstacle_traversal_retreat_completed_steps", 0
        )),
        "traversal_rearm_pending_steps": int(metrics.get(
            "probabilistic_obstacle_traversal_rearm_pending_steps", 0
        )),
        "traversal_rearm_no_crossing_safe_streak_max": int(metrics.get(
            "probabilistic_obstacle_traversal_rearm_no_crossing_safe_streak_max",
            0,
        )),
        "traversal_rearm_released_by_no_crossing_clearance_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_rearm_released_by_no_crossing_clearance_steps",
                0,
            )
        ),
        "traversal_rearm_no_crossing_certified_handoff_active_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_rearm_no_crossing_certified_handoff_active_steps",
                0,
            )
        ),
        "traversal_rearm_no_crossing_certified_handoff_safe_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_rearm_no_crossing_certified_handoff_safe_steps",
                0,
            )
        ),
        "traversal_rearm_hold_overridden_by_hard_risk_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_rearm_hold_overridden_by_hard_risk_steps",
                0,
            )
        ),
        "traversal_temporal_retreat_raw_lattice_requested_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_temporal_retreat_raw_lattice_requested_steps",
                0,
            )
        ),
        "traversal_temporal_retreat_post_intent_forward_lattice_filtered_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_temporal_retreat_post_intent_forward_lattice_filtered_steps",
                0,
            )
        ),
        "traversal_temporal_midpoint_retreat_forward_lattice_filtered_steps": int(
            metrics.get(
                "probabilistic_obstacle_traversal_temporal_midpoint_retreat_forward_lattice_filtered_steps",
                0,
            )
        ),
        "traversal_candidate_selected_steps": int(metrics.get(
            "probabilistic_obstacle_traversal_candidate_selected_steps", 0
        )),
        "traversal_maximum_probability_max": float(metrics.get(
            "probabilistic_obstacle_traversal_maximum_probability_max", 0.0
        )),
        "traversal_probability_mass_min": float(metrics.get(
            "probabilistic_obstacle_traversal_probability_mass_min", 0.0
        )),
        "traversal_required_steps_max": int(metrics.get(
            "probabilistic_obstacle_traversal_required_steps_max", 0
        )),
        "path_progress_m_max": float(metrics.get("path_progress_m_max", 0.0)),
        "path_progress_ratio_max": float(
            metrics.get("path_progress_ratio_max", 0.0)
        ),
    })
    return result


def run(
    seed, candidate, output_dir, source=DEFAULT_SOURCE,
    base_config=None,
    profile_components=False,
    proposal_gate_mode="episode_latched_veto",
    proposal_only=False,
    arm_order="source_first",
    completion_handover_full_fallback_distance=0.0,
    completion_handover_full_rl_distance=0.0,
    vetted_reactive_escape=False,
    dynamic_escape_min_probability_mass_relative_improvement=0.0,
    probabilistic_emergency_trigger_ttc_s=0.0,
    probabilistic_emergency_intent_hold_steps=0,
    probabilistic_emergency_pareto_forward_commit=False,
    probabilistic_emergency_forward_risk_ceiling=0.0,
    probabilistic_emergency_forward_mass_ceiling=0.0,
    probabilistic_emergency_rearm_ttc_s=0.0,
    probabilistic_emergency_rearm_clear_steps=1,
):
    seed = int(seed)
    if not 730100001 <= seed <= 730100300:
        raise ValueError("dynamic Actor probe seed must remain development-only")
    candidate = Path(candidate)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    source = Path(source)
    if not source.is_absolute():
        source = ROOT / source
    if not candidate.is_file() or not source.is_file():
        raise FileNotFoundError("dynamic Actor probe checkpoint is missing")
    output = Path(output_dir).resolve()
    if ROOT not in output.parents:
        raise ValueError("dynamic Actor probe output must remain in repository")
    protocol = _mapping(STAGE5_PROTOCOL)
    base_config_path = (
        _resolve(protocol["base_config"])
        if base_config is None
        else _resolve(base_config)
    )
    base = load_yaml(base_config_path)
    stage3 = _mapping(_resolve(protocol["stage3_protocol"]))
    stage4 = _mapping(_resolve(protocol["stage4_protocol"]))
    checkpoints = {
        "source_actor": source,
        "dynamic_actor": candidate,
    }
    arm_sequence = _resolve_arm_order(arm_order)
    handover_low = float(completion_handover_full_fallback_distance)
    handover_high = float(completion_handover_full_rl_distance)
    if handover_low < 0.0 or handover_high < 0.0:
        raise ValueError("completion handover distances must be non-negative")
    if (handover_low > 0.0 or handover_high > 0.0) and not (
        handover_high > handover_low > 0.0
    ):
        raise ValueError(
            "completion handover requires 0 < fallback distance < RL distance"
        )
    mass_relative_improvement = float(
        dynamic_escape_min_probability_mass_relative_improvement
    )
    if not 0.0 <= mass_relative_improvement <= 1.0:
        raise ValueError(
            "dynamic escape relative mass improvement must be in [0, 1]"
        )
    emergency_trigger_ttc = float(probabilistic_emergency_trigger_ttc_s)
    if emergency_trigger_ttc < 0.0:
        raise ValueError(
            "probabilistic emergency trigger TTC must be non-negative"
        )
    emergency_intent_hold_steps = int(
        probabilistic_emergency_intent_hold_steps
    )
    if emergency_intent_hold_steps < 0:
        raise ValueError(
            "probabilistic emergency intent hold must be non-negative"
        )
    emergency_forward_risk_ceiling = float(
        probabilistic_emergency_forward_risk_ceiling
    )
    emergency_forward_mass_ceiling = float(
        probabilistic_emergency_forward_mass_ceiling
    )
    if (
        not 0.0 <= emergency_forward_risk_ceiling <= 1.0
        or emergency_forward_mass_ceiling < 0.0
        or (emergency_forward_risk_ceiling > 0.0)
        != (emergency_forward_mass_ceiling > 0.0)
    ):
        raise ValueError(
            "probabilistic emergency forward risk ceilings are invalid"
        )
    emergency_rearm_ttc = float(probabilistic_emergency_rearm_ttc_s)
    if emergency_rearm_ttc < 0.0:
        raise ValueError(
            "probabilistic emergency rearm TTC must be non-negative"
        )
    emergency_rearm_clear_steps = int(
        probabilistic_emergency_rearm_clear_steps
    )
    if emergency_rearm_clear_steps < 1:
        raise ValueError(
            "probabilistic emergency rearm clear steps must be positive"
        )
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in arm_sequence:
        checkpoint = checkpoints[arm]
        run_dir = output / "runs" / arm
        if not _complete(run_dir):
            config = configure_combined_job(
                base, stage3, stage4, "combined_veto", seed
            )
            config["planner"]["profile_components"] = bool(
                profile_components
            )
            if proposal_gate_mode in ("shadow", "same_cycle_filter"):
                config["planner"]["paper_rl_driven"][
                    "proposal_advantage_gate"
                ] = {
                    "enabled": True,
                    "mode": "shadow",
                    "relative_disadvantage_margin": 0.0,
                    "consecutive_disadvantages": 3,
                }
                config["planner"]["paper_rl_driven"][
                    "standard_fallback_on_advantage_veto"
                ] = False
                if proposal_gate_mode == "same_cycle_filter":
                    config["planner"]["paper_rl_driven"][
                        "same_cycle_guided_cost_filter"
                    ] = True
                    config["planner"]["paper_rl_driven"][
                        "same_cycle_guided_relative_margin"
                    ] = 0.0
            elif proposal_gate_mode != "episode_latched_veto":
                raise ValueError("unknown dynamic Actor proposal gate mode")
            if proposal_only:
                config["planner"]["paper_rl_driven"][
                    "terminal_value_weight"
                ] = 0.0
            config["planner"]["paper_rl_driven"].update({
                "completion_handover_full_fallback_distance": handover_low,
                "completion_handover_full_rl_distance": handover_high,
            })
            config["planner"][
                "probabilistic_obstacle_emergency_candidate_trigger_ttc_s"
            ] = emergency_trigger_ttc
            config["planner"][
                "probabilistic_obstacle_emergency_candidate_intent_hold_steps"
            ] = emergency_intent_hold_steps
            config["planner"][
                "probabilistic_obstacle_emergency_candidate_pareto_forward_commit_enabled"
            ] = bool(probabilistic_emergency_pareto_forward_commit)
            config["planner"][
                "probabilistic_obstacle_emergency_candidate_forward_risk_ceiling"
            ] = emergency_forward_risk_ceiling
            config["planner"][
                "probabilistic_obstacle_emergency_candidate_forward_mass_ceiling"
            ] = emergency_forward_mass_ceiling
            config["planner"][
                "probabilistic_obstacle_emergency_candidate_rearm_ttc_s"
            ] = emergency_rearm_ttc
            config["planner"][
                "probabilistic_obstacle_emergency_candidate_rearm_clear_steps"
            ] = emergency_rearm_clear_steps
            config["perception"]["scan_guard"].update({
                "dynamic_escape_use_vetted_planner_control": bool(
                    vetted_reactive_escape
                ),
                "dynamic_escape_min_probability_mass_relative_improvement": (
                    mass_relative_improvement
                ),
            })
            config["rl"]["checkpoint"] = str(checkpoint)
            config["rl"]["policy_id"] = arm
            config["experiment"]["name"] += "__%s" % arm
            config["experiment"]["dynamic_actor_probe"] = True
            config["experiment"]["sealed_seeds_opened"] = False
            print(json.dumps({
                "stage": "mujoco_probe",
                "seed": seed,
                "arm": arm,
                "checkpoint": str(checkpoint),
            }, sort_keys=True), flush=True)
            ExperimentRunner(
                config, ROOT, output_dir=run_dir, headless=True
            ).run()
        results[arm] = _extended_summary(_json(run_dir / "metrics.json"))
    source_result = results["source_actor"]
    candidate_result = results["dynamic_actor"]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "scope": "paired_dynamic_actor_development_probe",
        "proposal_gate_mode": proposal_gate_mode,
        "proposal_only": bool(proposal_only),
        "arm_order": str(arm_order),
        "arm_sequence": list(arm_sequence),
        "completion_handover_full_fallback_distance": handover_low,
        "completion_handover_full_rl_distance": handover_high,
        "vetted_reactive_escape": bool(vetted_reactive_escape),
        "dynamic_escape_min_probability_mass_relative_improvement": (
            mass_relative_improvement
        ),
        "probabilistic_emergency_trigger_ttc_s": emergency_trigger_ttc,
        "probabilistic_emergency_intent_hold_steps": (
            emergency_intent_hold_steps
        ),
        "probabilistic_emergency_pareto_forward_commit": bool(
            probabilistic_emergency_pareto_forward_commit
        ),
        "probabilistic_emergency_forward_risk_ceiling": (
            emergency_forward_risk_ceiling
        ),
        "probabilistic_emergency_forward_mass_ceiling": (
            emergency_forward_mass_ceiling
        ),
        "probabilistic_emergency_rearm_ttc_s": emergency_rearm_ttc,
        "probabilistic_emergency_rearm_clear_steps": (
            emergency_rearm_clear_steps
        ),
        "episode_seed": seed,
        "sealed_seeds_opened": False,
        "base_config": str(base_config_path.relative_to(ROOT)),
        "profile_components": bool(profile_components),
        "source_checkpoint": str(source.relative_to(ROOT)),
        "candidate_checkpoint": str(candidate.relative_to(ROOT)),
        "results": results,
        "paired_delta_dynamic_minus_source": {
            "success": int(candidate_result["success"]) - int(
                source_result["success"]
            ),
            "collision": int(candidate_result["collision"]) - int(
                source_result["collision"]
            ),
            "steps": candidate_result["steps"] - source_result["steps"],
            "final_goal_distance_m": (
                candidate_result["final_goal_distance_m"]
                - source_result["final_goal_distance_m"]
            ),
            "guided_minus_gaussian_cost_min_mean": (
                candidate_result["guided_minus_gaussian_cost_min_mean"]
                - source_result["guided_minus_gaussian_cost_min_mean"]
            ),
        },
    }
    _write_json(output / "result.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--base-config")
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--proposal-gate-mode",
        choices=("episode_latched_veto", "shadow", "same_cycle_filter"),
        default="episode_latched_veto",
    )
    parser.add_argument("--proposal-only", action="store_true")
    parser.add_argument(
        "--arm-order",
        choices=tuple(ARM_ORDERS),
        default="source_first",
    )
    parser.add_argument(
        "--completion-handover-full-fallback-distance",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--completion-handover-full-rl-distance",
        type=float,
        default=0.0,
    )
    parser.add_argument("--vetted-reactive-escape", action="store_true")
    parser.add_argument(
        "--dynamic-escape-min-probability-mass-relative-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--probabilistic-emergency-trigger-ttc-s",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--probabilistic-emergency-intent-hold-steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--probabilistic-emergency-pareto-forward-commit",
        action="store_true",
    )
    parser.add_argument(
        "--probabilistic-emergency-forward-risk-ceiling",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--probabilistic-emergency-forward-mass-ceiling",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--probabilistic-emergency-rearm-ttc-s",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--probabilistic-emergency-rearm-clear-steps",
        type=int,
        default=1,
    )
    args = parser.parse_args(argv)
    run(
        args.seed,
        args.candidate,
        args.output_dir,
        source=args.source,
        base_config=args.base_config,
        profile_components=bool(args.profile_components),
        proposal_gate_mode=args.proposal_gate_mode,
        proposal_only=bool(args.proposal_only),
        arm_order=args.arm_order,
        completion_handover_full_fallback_distance=(
            args.completion_handover_full_fallback_distance
        ),
        completion_handover_full_rl_distance=(
            args.completion_handover_full_rl_distance
        ),
        vetted_reactive_escape=bool(args.vetted_reactive_escape),
        dynamic_escape_min_probability_mass_relative_improvement=(
            args.dynamic_escape_min_probability_mass_relative_improvement
        ),
        probabilistic_emergency_trigger_ttc_s=(
            args.probabilistic_emergency_trigger_ttc_s
        ),
        probabilistic_emergency_intent_hold_steps=(
            args.probabilistic_emergency_intent_hold_steps
        ),
        probabilistic_emergency_pareto_forward_commit=bool(
            args.probabilistic_emergency_pareto_forward_commit
        ),
        probabilistic_emergency_forward_risk_ceiling=(
            args.probabilistic_emergency_forward_risk_ceiling
        ),
        probabilistic_emergency_forward_mass_ceiling=(
            args.probabilistic_emergency_forward_mass_ceiling
        ),
        probabilistic_emergency_rearm_ttc_s=(
            args.probabilistic_emergency_rearm_ttc_s
        ),
        probabilistic_emergency_rearm_clear_steps=(
            args.probabilistic_emergency_rearm_clear_steps
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
