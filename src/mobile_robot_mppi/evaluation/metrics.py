import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class EpisodeMetrics:
    goal_x: float
    goal_y: float
    goal_tolerance: float
    control_dt: float = 0.0
    reference_points: Optional[Sequence[Sequence[float]]] = None
    records: List[Dict[str, float]] = field(default_factory=list)

    def _cross_track_errors(self, values):
        if self.reference_points is None:
            return None
        points = np.asarray(self.reference_points, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[0] < 2
            or points.shape[1] < 2
            or not np.isfinite(points[:, :2]).all()
        ):
            raise ValueError(
                "reference_points must contain at least two finite XY points"
            )
        query = np.asarray(
            [(row["x"], row["y"]) for row in values], dtype=np.float64
        )
        starts = points[:-1, :2]
        segments = points[1:, :2] - starts
        squared_lengths = np.sum(segments * segments, axis=1)
        valid = squared_lengths > 1e-16
        if not np.any(valid):
            raise ValueError("reference_points must contain a nonzero segment")
        starts = starts[valid]
        segments = segments[valid]
        squared_lengths = squared_lengths[valid]
        relative = query[:, None, :] - starts[None, :, :]
        fraction = np.sum(relative * segments[None, :, :], axis=2)
        fraction = np.clip(fraction / squared_lengths[None, :], 0.0, 1.0)
        projections = starts[None, :, :] + fraction[:, :, None] * segments[None, :, :]
        distance = np.linalg.norm(query[:, None, :] - projections, axis=2)
        return np.min(distance, axis=1)

    def update(self, truth, safety_decision, planner_diagnostics, applied_control=None):
        distance = math.hypot(truth.pose.x - self.goal_x, truth.pose.y - self.goal_y)
        applied = applied_control or safety_decision.executed_control
        prior = dict(planner_diagnostics.get("prior", {}))
        truth_metadata = dict(getattr(truth, "metadata", {}) or {})
        safety_diagnostics = dict(
            getattr(safety_decision, "diagnostics", {}) or {}
        )
        self.records.append({
            "time": truth.timestamp,
            "x": truth.pose.x,
            "y": truth.pose.y,
            "theta": truth.pose.theta,
            "v": truth.twist.v,
            "omega": truth.twist.omega,
            "goal_distance": distance,
            "collision": float(truth.collision),
            "clearance": truth.minimum_clearance,
            "slip_ratio": truth.slip_ratio,
            "dynamic_obstacle_count": int(
                truth_metadata.get("dynamic_obstacle_count", 0)
            ),
            "nearest_dynamic_obstacle_center_distance": (
                float(truth_metadata["nearest_dynamic_obstacle_center_distance"])
                if truth_metadata.get(
                    "nearest_dynamic_obstacle_center_distance"
                ) is not None
                else float("inf")
            ),
            "proposed_v": safety_decision.proposed_control.v,
            "proposed_omega": safety_decision.proposed_control.omega,
            "executed_v": safety_decision.executed_control.v,
            "executed_omega": safety_decision.executed_control.omega,
            "applied_v": applied.v,
            "applied_omega": applied.omega,
            "safety_override": float(safety_decision.overridden),
            "safety_reason": str(safety_decision.reason),
            "temporal_scan_valid": float(
                safety_diagnostics.get("temporal_scan_valid", False)
            ),
            "temporal_scan_closing_rate_mps": float(
                safety_diagnostics.get(
                    "temporal_scan_closing_rate_mps", 0.0
                )
            ),
            "temporal_scan_ttc_s": float(
                safety_diagnostics.get(
                    "temporal_scan_ttc_s", float("inf")
                )
            ),
            "temporal_scan_risk_alpha": float(
                safety_diagnostics.get(
                    "temporal_scan_risk_alpha", 0.0
                )
            ),
            "temporal_scan_support_beams": int(
                safety_diagnostics.get(
                    "temporal_scan_support_beams", 0
                )
            ),
            "temporal_scan_rejected_jump_fraction": float(
                safety_diagnostics.get(
                    "temporal_scan_rejected_jump_fraction", 0.0
                )
            ),
            "temporal_scan_held": float(
                safety_diagnostics.get("temporal_scan_held", False)
            ),
            "reference_id": str(planner_diagnostics.get("reference_id", "unknown")),
            "target_phase": str(planner_diagnostics.get("target_phase", "unknown")),
            "target_x": float(planner_diagnostics.get("target_x", self.goal_x)),
            "target_y": float(planner_diagnostics.get("target_y", self.goal_y)),
            "terminal_heading_gate_active": float(
                planner_diagnostics.get("terminal_heading_gate_active", False)
            ),
            "terminal_bearing_error": float(
                planner_diagnostics.get("terminal_bearing_error", 0.0)
            ),
            "terminal_translation_scale": float(
                planner_diagnostics.get("terminal_translation_scale", 1.0)
            ),
            "terminal_alignment_active": float(
                planner_diagnostics.get("terminal_alignment_active", False)
            ),
            "terminal_alignment_omega": float(
                planner_diagnostics.get("terminal_alignment_omega", 0.0)
            ),
            "planner_compute_ms": float(planner_diagnostics.get("compute_ms", 0.0)),
            "profile_planner_state_reference_ms": float(
                planner_diagnostics.get(
                    "profile_planner_state_reference_ms", 0.0
                )
            ),
            "profile_planner_prior_ms": float(
                planner_diagnostics.get("profile_planner_prior_ms", 0.0)
            ),
            "profile_mppi_sampling_ms": float(
                planner_diagnostics.get("profile_mppi_sampling_ms", 0.0)
            ),
            "profile_mppi_batch_rollout_ms": float(
                planner_diagnostics.get(
                    "profile_mppi_batch_rollout_ms", 0.0
                )
            ),
            "profile_mppi_cost_ms": float(
                planner_diagnostics.get("profile_mppi_cost_ms", 0.0)
            ),
            "profile_mppi_weighting_update_ms": float(
                planner_diagnostics.get(
                    "profile_mppi_weighting_update_ms", 0.0
                )
            ),
            "profile_mppi_final_rollout_ms": float(
                planner_diagnostics.get(
                    "profile_mppi_final_rollout_ms", 0.0
                )
            ),
            "profile_mppi_solve_total_ms": float(
                planner_diagnostics.get(
                    "profile_mppi_solve_total_ms", 0.0
                )
            ),
            "planner_cost_min": float(planner_diagnostics.get("cost_min", 0.0)),
            "planner_cost_mean": float(planner_diagnostics.get("cost_mean", 0.0)),
            "effective_sample_size": float(
                planner_diagnostics.get("effective_sample_size", 0.0)
            ),
            "sample_saturation_fraction": float(
                planner_diagnostics.get("sample_saturation_fraction", 0.0)
            ),
            "residual_support_confidence_mean": float(
                planner_diagnostics.get("residual_support_confidence_mean", 1.0)
            ),
            "residual_support_confidence_min": float(
                planner_diagnostics.get("residual_support_confidence_min", 1.0)
            ),
            "residual_support_reduced_fraction": float(
                planner_diagnostics.get("residual_support_reduced_fraction", 0.0)
            ),
            "residual_support_disabled_fraction": float(
                planner_diagnostics.get("residual_support_disabled_fraction", 0.0)
            ),
            "residual_reliability_enabled": float(
                planner_diagnostics.get("residual_reliability_enabled", False)
            ),
            "residual_reliability_alpha": float(
                planner_diagnostics.get("residual_reliability_alpha", 0.0)
            ),
            "residual_reliability_evidence_alpha": float(
                planner_diagnostics.get(
                    "residual_reliability_evidence_alpha", 0.0
                )
            ),
            "residual_reliability_context_alpha": float(
                planner_diagnostics.get(
                    "residual_reliability_context_alpha", 0.0
                )
            ),
            "residual_reliability_context_value": float(
                planner_diagnostics.get(
                    "residual_reliability_context_value", 0.0
                )
            ),
            "residual_reliability_samples": int(
                planner_diagnostics.get("residual_reliability_samples", 0)
            ),
            "residual_reliability_mean_improvement": float(
                planner_diagnostics.get(
                    "residual_reliability_mean_improvement", 0.0
                )
            ),
            "residual_reliability_lcb": float(
                planner_diagnostics.get("residual_reliability_lcb", 0.0)
            ),
            "residual_reliability_last_relative_improvement": float(
                planner_diagnostics.get(
                    "residual_reliability_last_relative_improvement", 0.0
                )
            ),
            "residual_reliability_nominal_error": float(
                planner_diagnostics.get("residual_reliability_nominal_error", 0.0)
            ),
            "residual_reliability_residual_error": float(
                planner_diagnostics.get("residual_reliability_residual_error", 0.0)
            ),
            "optimizer": str(planner_diagnostics.get("optimizer", "standard")),
            "rl_driven_total_rollouts": int(
                planner_diagnostics.get("rl_driven_total_rollouts", 0)
            ),
            "paper_total_rollouts": int(
                planner_diagnostics.get("paper_total_rollouts", 0)
            ),
            "paper_candidates_per_iteration": int(
                planner_diagnostics.get(
                    "paper_candidates_per_iteration", 0
                )
            ),
            "paper_guided_unique_sequences": int(
                planner_diagnostics.get(
                    "paper_guided_unique_sequences", 0
                )
            ),
            "paper_guided_reuses": int(
                planner_diagnostics.get("paper_guided_reuses", 0)
            ),
            "reliability_hss_enabled": float(
                planner_diagnostics.get("reliability_hss_enabled", False)
            ),
            "reliability_level": str(
                planner_diagnostics.get("reliability_level", "disabled")
            ),
            "reliability_authority": float(
                planner_diagnostics.get("reliability_authority", 1.0)
            ),
            "reliability_proposal_authority": float(
                planner_diagnostics.get(
                    "reliability_proposal_authority", 1.0
                )
            ),
            "reliability_proposal_fallback_fraction": float(
                planner_diagnostics.get(
                    "reliability_proposal_fallback_fraction", 0.0
                )
            ),
            "reliability_dynamics_confidence": float(
                planner_diagnostics.get("dynamics_confidence", 1.0)
            ),
            "reliability_dynamics_routing_mode": str(
                planner_diagnostics.get(
                    "dynamics_routing_mode", "trust_weighted"
                )
            ),
            "reliability_model_routing_factor": float(
                planner_diagnostics.get("model_routing_factor", 1.0)
            ),
            "reliability_actor_confidence": float(
                planner_diagnostics.get("actor_confidence", 1.0)
            ),
            "reliability_actor_competence_confidence": float(
                planner_diagnostics.get(
                    "actor_competence_confidence", 1.0
                )
            ),
            "reliability_actor_competence_raw_confidence": float(
                planner_diagnostics.get(
                    "actor_competence_raw_confidence", 1.0
                )
            ),
            "reliability_actor_competence_mapped_confidence": float(
                planner_diagnostics.get(
                    "actor_competence_mapped_confidence", 1.0
                )
            ),
            "reliability_actor_competence_guided_yield": float(
                planner_diagnostics.get(
                    "actor_competence_guided_yield", 0.0
                )
            ),
            "reliability_actor_competence_gaussian_yield": float(
                planner_diagnostics.get(
                    "actor_competence_gaussian_yield", 0.0
                )
            ),
            "reliability_ensemble_disagreement_max": float(
                planner_diagnostics.get(
                    "ensemble_disagreement_max", 0.0
                )
            ),
            "reliability_innovation_error_ema": float(
                planner_diagnostics.get("innovation_error_ema", 0.0)
            ),
            "reliability_guided_fraction_applied": float(
                planner_diagnostics.get(
                    "reliability_guided_fraction_applied", 0.30
                )
            ),
            "reliability_guided_fraction_next": float(
                planner_diagnostics.get(
                    "reliability_guided_fraction_next", 0.30
                )
            ),
            "reliability_guided_fraction_raw_applied": float(
                planner_diagnostics.get(
                    "reliability_guided_fraction_raw_applied", 0.30
                )
            ),
            "reliability_guided_fraction_raw_next": float(
                planner_diagnostics.get(
                    "reliability_guided_fraction_raw_next", 0.30
                )
            ),
            "terminal_guidance_floor_enabled": float(
                planner_diagnostics.get(
                    "terminal_guidance_floor_enabled", False
                )
            ),
            "terminal_guidance_floor_active": float(
                planner_diagnostics.get(
                    "terminal_guidance_floor_active", False
                )
            ),
            "terminal_guidance_terminal_phase": float(
                planner_diagnostics.get(
                    "terminal_guidance_terminal_phase", False
                )
            ),
            "terminal_guidance_distance": float(
                planner_diagnostics.get(
                    "terminal_guidance_distance", float("inf")
                )
            ),
            "terminal_guidance_radius": float(
                planner_diagnostics.get("terminal_guidance_radius", 0.0)
            ),
            "terminal_guided_fraction_floor": float(
                planner_diagnostics.get(
                    "terminal_guided_fraction_floor", 0.0
                )
            ),
            "completion_handover_enabled": float(
                planner_diagnostics.get(
                    "completion_handover_enabled", False
                )
            ),
            "completion_handover_authority": float(
                planner_diagnostics.get(
                    "completion_handover_authority", 1.0
                )
            ),
            "completion_handover_distance": float(
                planner_diagnostics.get(
                    "completion_handover_distance", float("inf")
                )
            ),
            "terminal_value_completion_authority": float(
                planner_diagnostics.get(
                    "terminal_value_completion_authority", 1.0
                )
            ),
            "paper_guided_elite_count": int(
                planner_diagnostics.get("paper_guided_elite_count", 0)
            ),
            "paper_gaussian_elite_count": int(
                planner_diagnostics.get("paper_gaussian_elite_count", 0)
            ),
            "rl_source_samples": int(
                planner_diagnostics.get("rl_source_samples", 0)
            ),
            "shifted_source_samples": int(
                planner_diagnostics.get("shifted_source_samples", 0)
            ),
            "base_source_samples": int(
                planner_diagnostics.get("base_source_samples", 0)
            ),
            "rl_elite_count": int(
                planner_diagnostics.get("rl_elite_count", 0)
            ),
            "shifted_elite_count": int(
                planner_diagnostics.get("shifted_elite_count", 0)
            ),
            "base_elite_count": int(
                planner_diagnostics.get("base_elite_count", 0)
            ),
            "rl_elite_fraction": float(
                planner_diagnostics.get("rl_elite_fraction", 0.0)
            ),
            "covariance_scale_mean": float(
                planner_diagnostics.get("covariance_scale_mean", 1.0)
            ),
            "terminal_value_enabled": float(
                planner_diagnostics.get("terminal_value_enabled", False)
            ),
            "terminal_value_weight": float(
                planner_diagnostics.get("terminal_value_weight", 0.0)
            ),
            "terminal_value_conservative_enabled": float(
                planner_diagnostics.get(
                    "terminal_value_conservative_enabled", False
                )
            ),
            "terminal_value_authority_mean": float(
                planner_diagnostics.get(
                    "terminal_value_authority_mean", 1.0
                )
            ),
            "terminal_value_authority_min": float(
                planner_diagnostics.get(
                    "terminal_value_authority_min", 1.0
                )
            ),
            "terminal_value_authority_max": float(
                planner_diagnostics.get(
                    "terminal_value_authority_max", 1.0
                )
            ),
            "terminal_value_dynamics_confidence_mean": float(
                planner_diagnostics.get(
                    "terminal_value_dynamics_confidence_mean", 1.0
                )
            ),
            "terminal_value_critic_confidence_mean": float(
                planner_diagnostics.get(
                    "terminal_value_critic_confidence_mean", 1.0
                )
            ),
            "terminal_value_uncertainty_mean": float(
                planner_diagnostics.get(
                    "terminal_value_uncertainty_mean", 0.0
                )
            ),
            "terminal_value_uncertainty_cost_mean": float(
                planner_diagnostics.get(
                    "terminal_value_uncertainty_cost_mean", 0.0
                )
            ),
            "terminal_q_mean": float(
                planner_diagnostics.get("terminal_q_mean", 0.0)
            ),
            "terminal_q_disagreement_mean": float(
                planner_diagnostics.get(
                    "terminal_q_disagreement_mean", 0.0
                )
            ),
            "residual_policy_context_enabled": float(
                planner_diagnostics.get(
                    "residual_policy_context_enabled", False
                )
            ),
            "residual_policy_predicted_abs_mean": float(
                planner_diagnostics.get(
                    "residual_policy_predicted_abs_mean", 0.0
                )
            ),
            "residual_policy_predicted_abs_max": float(
                planner_diagnostics.get(
                    "residual_policy_predicted_abs_max", 0.0
                )
            ),
            "residual_policy_innovation_abs_mean": float(
                planner_diagnostics.get(
                    "residual_policy_innovation_abs_mean", 0.0
                )
            ),
            "residual_policy_innovation_abs_max": float(
                planner_diagnostics.get(
                    "residual_policy_innovation_abs_max", 0.0
                )
            ),
            "residual_policy_disagreement_mean": float(
                planner_diagnostics.get(
                    "residual_policy_disagreement_mean", 0.0
                )
            ),
            "residual_policy_support_mean": float(
                planner_diagnostics.get(
                    "residual_policy_support_mean", 0.0
                )
            ),
            "residual_policy_innovation_valid_fraction": float(
                planner_diagnostics.get(
                    "residual_policy_innovation_valid_fraction", 0.0
                )
            ),
            "residual_policy_authority_enabled": float(
                planner_diagnostics.get(
                    "residual_policy_authority_enabled", False
                )
            ),
            "residual_policy_authority_mean": float(
                planner_diagnostics.get(
                    "residual_policy_authority_mean", 0.0
                )
            ),
            "residual_policy_authority_max": float(
                planner_diagnostics.get(
                    "residual_policy_authority_max", 0.0
                )
            ),
            "anytime_selected_samples": int(
                planner_diagnostics.get("anytime_selected_samples", 0)
            ),
            "anytime_add_samples": float(
                planner_diagnostics.get("anytime_add_samples", False)
            ),
            "anytime_predicted_advantage": float(
                planner_diagnostics.get("anytime_predicted_advantage", 0.0)
            ),
            "anytime_confidence_width": float(
                planner_diagnostics.get("anytime_confidence_width", 0.0)
            ),
            "anytime_score": float(
                planner_diagnostics.get("anytime_score", 0.0)
            ),
            "anytime_decision_refreshed": float(
                planner_diagnostics.get("anytime_decision_refreshed", False)
            ),
            "prior_type": str(prior.get("type", "unknown")),
            "rl_gate_mode": str(prior.get("gate_mode", "disabled")),
            "rl_gate_alpha": float(prior.get("gate_alpha", 0.0)),
            "rl_policy_mode": str(prior.get("policy_mode", "direct")),
            "rl_correction_gate_alpha": float(
                prior.get("correction_gate_alpha", 0.0)
            ),
            "rl_base_action_abs_mean": float(
                prior.get("base_action_abs_mean", 0.0)
            ),
            "rl_unit_correction_abs_mean": float(
                prior.get("unit_correction_abs_mean", 0.0)
            ),
            "rl_applied_correction_abs_mean": float(
                prior.get("applied_correction_abs_mean", 0.0)
            ),
            "rl_applied_correction_abs_max": float(
                prior.get("applied_correction_abs_max", 0.0)
            ),
            "rl_raw_applied_correction_abs_mean": float(
                prior.get("raw_applied_correction_abs_mean", 0.0)
            ),
            "rl_correction_advantage_gate_mode": str(
                prior.get("correction_advantage_gate_mode", "none")
            ),
            "rl_correction_advantage_critic_source": str(
                prior.get("correction_advantage_critic_source", "online")
            ),
            "rl_correction_advantage_threshold": float(
                prior.get("correction_advantage_threshold", 0.0)
            ),
            "rl_correction_advantage_uncertainty_multiplier": float(
                prior.get(
                    "correction_advantage_uncertainty_multiplier", 1.0
                )
            ),
            "rl_correction_advantage_gate_alpha": float(
                prior.get("correction_advantage_gate_alpha", 1.0)
            ),
            "rl_correction_support_gate_enabled": float(
                prior.get("correction_support_gate_enabled", False)
            ),
            "rl_correction_support_confidence": float(
                prior.get("correction_support_confidence", 1.0)
            ),
            "rl_correction_effective_gate_alpha": float(
                prior.get("correction_effective_gate_alpha", 1.0)
            ),
            "rl_unselected_critic_diagnostics_computed": float(
                prior.get(
                    "unselected_critic_diagnostics_computed", False
                )
            ),
            "rl_selected_consensus_lcb": float(
                prior.get("selected_consensus_lcb", 0.0)
            ),
            "rl_online_q1_base": float(prior.get("online_q1_base", 0.0)),
            "rl_online_q2_base": float(prior.get("online_q2_base", 0.0)),
            "rl_online_q1_candidate": float(
                prior.get("online_q1_candidate", 0.0)
            ),
            "rl_online_q2_candidate": float(
                prior.get("online_q2_candidate", 0.0)
            ),
            "rl_target_q1_base": float(prior.get("target_q1_base", 0.0)),
            "rl_target_q2_base": float(prior.get("target_q2_base", 0.0)),
            "rl_target_q1_candidate": float(
                prior.get("target_q1_candidate", 0.0)
            ),
            "rl_target_q2_candidate": float(
                prior.get("target_q2_candidate", 0.0)
            ),
            "rl_online_advantage_q1": float(
                prior.get("online_advantage_q1", 0.0)
            ),
            "rl_online_advantage_q2": float(
                prior.get("online_advantage_q2", 0.0)
            ),
            "rl_online_conservative_advantage": float(
                prior.get("online_conservative_advantage", 0.0)
            ),
            "rl_target_advantage_q1": float(
                prior.get("target_advantage_q1", 0.0)
            ),
            "rl_target_advantage_q2": float(
                prior.get("target_advantage_q2", 0.0)
            ),
            "rl_target_conservative_advantage": float(
                prior.get("target_conservative_advantage", 0.0)
            ),
            "rl_ood_score": float(prior.get("ood_score", 0.0)),
            "rl_scene_complexity_score": float(
                prior.get("scene_complexity_score", 0.0)
            ),
            "rl_scene_complexity_front_proximity": float(
                prior.get("scene_complexity_front_proximity", 0.0)
            ),
            "rl_scene_complexity_constriction": float(
                prior.get("scene_complexity_constriction", 0.0)
            ),
            "rl_scene_complexity_density": float(
                prior.get("scene_complexity_density", 0.0)
            ),
            "rl_scene_complexity_front_clearance_m": float(
                prior.get("scene_complexity_front_clearance_m", 0.0)
            ),
            "rl_scene_complexity_left_clearance_m": float(
                prior.get("scene_complexity_left_clearance_m", 0.0)
            ),
            "rl_scene_complexity_right_clearance_m": float(
                prior.get("scene_complexity_right_clearance_m", 0.0)
            ),
            "rl_scene_complexity_near_obstacle_fraction": float(
                prior.get("scene_complexity_near_obstacle_fraction", 0.0)
            ),
            "rl_scene_complexity_scan_valid": float(
                prior.get("scene_complexity_scan_valid", False)
            ),
            "rl_temporal_closing_gate_alpha": float(
                prior.get("temporal_closing_gate_alpha", 0.0)
            ),
            "rl_temporal_closing_rate_mps": float(
                prior.get("temporal_closing_rate_mps", 0.0)
            ),
            "rl_temporal_closing_held": float(
                prior.get("temporal_closing_held", False)
            ),
            "rl_hazard_activation": float(
                prior.get("hazard_activation", 0.0)
            ),
            "rl_competence_confidence": float(
                prior.get("competence_confidence", 0.0)
            ),
            "rl_baseline_progress_m": float(
                prior.get("baseline_progress_m", 0.0)
            ),
            "rl_baseline_progress_window_s": float(
                prior.get("baseline_progress_window_s", 0.0)
            ),
            "rl_baseline_progress_gate_ready": float(
                prior.get("baseline_progress_gate_ready", False)
            ),
            "rl_baseline_stagnation_activation": float(
                prior.get("baseline_stagnation_activation", 0.0)
            ),
            "rl_baseline_stagnation_held": float(
                prior.get("baseline_stagnation_held", False)
            ),
            "rl_learned_inference_skipped": float(
                prior.get("learned_inference_skipped", False)
            ),
            "profile_prior_fallback_ms": float(
                prior.get("profile_prior_fallback_ms", 0.0)
            ),
            "profile_prior_encode_normalize_ms": float(
                prior.get("profile_prior_encode_normalize_ms", 0.0)
            ),
            "profile_prior_gate_features_ms": float(
                prior.get("profile_prior_gate_features_ms", 0.0)
            ),
            "profile_prior_actor_ms": float(
                prior.get("profile_prior_actor_ms", 0.0)
            ),
            "profile_prior_advantage_ms": float(
                prior.get("profile_prior_advantage_ms", 0.0)
            ),
            "profile_prior_decoder_ms": float(
                prior.get("profile_prior_decoder_ms", 0.0)
            ),
            "profile_prior_outer_gate_ms": float(
                prior.get("profile_prior_outer_gate_ms", 0.0)
            ),
            "profile_prior_total_ms": float(
                prior.get("profile_prior_total_ms", 0.0)
            ),
            "rl_critic_disagreement": float(
                prior.get("critic_disagreement", 0.0)
            ),
            "rl_exploration_activation": float(
                prior.get("exploration_activation", 0.0)
            ),
            "rl_exploration_latch_alpha": float(
                prior.get("exploration_latch_alpha", 0.0)
            ),
            "rl_subgoal_distance": float(
                prior.get("subgoal_distance", 0.0)
            ),
            "rl_subgoal_bearing": float(
                prior.get("subgoal_bearing", 0.0)
            ),
            "rl_subgoal_x_body": float(
                prior.get("subgoal_x_body", 0.0)
            ),
            "rl_subgoal_y_body": float(
                prior.get("subgoal_y_body", 0.0)
            ),
        })

    def summary(self, termination_reason=None):
        if not self.records:
            return {
                "steps": 0,
                "success": False,
                "termination_reason": termination_reason or "no_steps",
            }
        values = self.records
        path_length = sum(
            math.hypot(values[index]["x"] - values[index - 1]["x"], values[index]["y"] - values[index - 1]["y"])
            for index in range(1, len(values))
        )
        controls = np.asarray([(row["executed_v"], row["executed_omega"]) for row in values])
        jerk = np.diff(controls, axis=0) if len(values) > 1 else np.zeros((0, 2))
        applied_controls = np.asarray(
            [(row["applied_v"], row["applied_omega"]) for row in values]
        )
        applied_jerk = (
            np.diff(applied_controls, axis=0)
            if len(values) > 1
            else np.zeros((0, 2))
        )
        compute = np.asarray([row["planner_compute_ms"] for row in values])
        effective_samples = np.asarray([row["effective_sample_size"] for row in values])
        finite_clearance = [row["clearance"] for row in values if math.isfinite(row["clearance"])]
        finite_dynamic_distance = [
            row["nearest_dynamic_obstacle_center_distance"]
            for row in values
            if math.isfinite(row["nearest_dynamic_obstacle_center_distance"])
        ]
        cross_track = self._cross_track_errors(values)
        collision = bool(any(row["collision"] for row in values))
        success = bool(values[-1]["goal_distance"] <= self.goal_tolerance and not collision)
        deadline_ms = 1000.0 * self.control_dt if self.control_dt > 0.0 else None
        deadline_misses = int(np.sum(compute > deadline_ms)) if deadline_ms is not None else 0
        stuck_steps = sum(
            abs(row["v"]) < 0.02 and row["goal_distance"] > self.goal_tolerance
            for row in values
        )
        spin_steps = sum(abs(row["v"]) < 0.03 and abs(row["omega"]) > 0.5 for row in values)
        safety_reason_counts = {}
        for row in values:
            if row["safety_override"]:
                reason = row["safety_reason"]
                safety_reason_counts[reason] = safety_reason_counts.get(reason, 0) + 1
        return {
            "steps": len(values),
            "success": success,
            "termination_reason": termination_reason or ("goal_reached" if success else "unknown"),
            "time_to_goal_s": float(values[-1]["time"]) if success else None,
            "collision": collision,
            "final_goal_distance": values[-1]["goal_distance"],
            "trajectory_length": path_length,
            "cross_track_rmse": (
                None
                if cross_track is None
                else float(np.sqrt(np.mean(np.square(cross_track))))
            ),
            "cross_track_mean": (
                None if cross_track is None else float(np.mean(cross_track))
            ),
            "cross_track_max": (
                None if cross_track is None else float(np.max(cross_track))
            ),
            "minimum_clearance": min(finite_clearance) if finite_clearance else None,
            "dynamic_obstacle_count": int(max(
                row.get("dynamic_obstacle_count", 0) for row in values
            )),
            "minimum_dynamic_obstacle_center_distance": (
                min(finite_dynamic_distance) if finite_dynamic_distance else None
            ),
            "mean_abs_omega": float(np.mean(np.abs(controls[:, 1]))),
            "control_jerk": float(np.mean(np.linalg.norm(jerk, axis=1))) if jerk.size else 0.0,
            "applied_control_jerk": (
                float(np.mean(np.linalg.norm(applied_jerk, axis=1)))
                if applied_jerk.size
                else 0.0
            ),
            "stuck_steps": int(stuck_steps),
            "spin_steps": int(spin_steps),
            "mean_slip_ratio": float(np.mean([row["slip_ratio"] for row in values])),
            "safety_interventions": int(sum(row["safety_override"] for row in values)),
            "safety_reason_counts": dict(sorted(safety_reason_counts.items())),
            "temporal_scan_valid_fraction": float(np.mean([
                row.get("temporal_scan_valid", 0.0) for row in values
            ])),
            "temporal_scan_risk_alpha_mean": float(np.mean([
                row.get("temporal_scan_risk_alpha", 0.0) for row in values
            ])),
            "temporal_scan_risk_alpha_max": float(np.max([
                row.get("temporal_scan_risk_alpha", 0.0) for row in values
            ])),
            "temporal_scan_closing_rate_mps_max": float(np.max([
                row.get("temporal_scan_closing_rate_mps", 0.0)
                for row in values
            ])),
            "temporal_scan_rejected_jump_fraction_max": float(np.max([
                row.get("temporal_scan_rejected_jump_fraction", 0.0)
                for row in values
            ])),
            "temporal_safety_interventions": int(sum(
                row.get("safety_override", 0.0)
                and str(row.get("safety_reason", "")).startswith("temporal_")
                for row in values
            )),
            "planner_compute_ms_mean": float(compute.mean()),
            "planner_compute_ms_p50": float(np.percentile(compute, 50)),
            "planner_compute_ms_p95": float(np.percentile(compute, 95)),
            "planner_compute_ms_p99": float(np.percentile(compute, 99)),
            "planner_compute_ms_max": float(compute.max()),
            "residual_support_confidence_mean": float(np.mean([
                row.get("residual_support_confidence_mean", 1.0) for row in values
            ])),
            "residual_support_confidence_min": float(np.min([
                row.get("residual_support_confidence_min", 1.0) for row in values
            ])),
            "residual_support_reduced_fraction_mean": float(np.mean([
                row.get("residual_support_reduced_fraction", 0.0) for row in values
            ])),
            "residual_support_disabled_fraction_mean": float(np.mean([
                row.get("residual_support_disabled_fraction", 0.0) for row in values
            ])),
            "profile_planner_state_reference_ms_mean": float(np.mean([
                row.get("profile_planner_state_reference_ms", 0.0)
                for row in values
            ])),
            "profile_planner_prior_ms_mean": float(np.mean([
                row.get("profile_planner_prior_ms", 0.0)
                for row in values
            ])),
            "profile_mppi_sampling_ms_mean": float(np.mean([
                row.get("profile_mppi_sampling_ms", 0.0)
                for row in values
            ])),
            "profile_mppi_batch_rollout_ms_mean": float(np.mean([
                row.get("profile_mppi_batch_rollout_ms", 0.0)
                for row in values
            ])),
            "profile_mppi_cost_ms_mean": float(np.mean([
                row.get("profile_mppi_cost_ms", 0.0)
                for row in values
            ])),
            "profile_mppi_weighting_update_ms_mean": float(np.mean([
                row.get("profile_mppi_weighting_update_ms", 0.0)
                for row in values
            ])),
            "profile_mppi_final_rollout_ms_mean": float(np.mean([
                row.get("profile_mppi_final_rollout_ms", 0.0)
                for row in values
            ])),
            "profile_mppi_solve_total_ms_mean": float(np.mean([
                row.get("profile_mppi_solve_total_ms", 0.0)
                for row in values
            ])),
            "planner_deadline_ms": deadline_ms,
            "planner_deadline_misses": deadline_misses,
            "planner_deadline_miss_rate": (
                float(deadline_misses / len(values)) if deadline_ms is not None else None
            ),
            "effective_sample_size_mean": float(effective_samples.mean()),
            "effective_sample_size_min": float(effective_samples.min()),
            "sample_saturation_fraction_mean": float(
                np.mean([row["sample_saturation_fraction"] for row in values])
            ),
            "optimizer": str(values[-1].get("optimizer", "standard")),
            "rl_driven_total_rollouts_mean": float(np.mean([
                row.get("rl_driven_total_rollouts", 0) for row in values
            ])),
            "paper_total_rollouts_mean": float(np.mean([
                row.get("paper_total_rollouts", 0) for row in values
            ])),
            "paper_candidates_per_iteration_mean": float(np.mean([
                row.get("paper_candidates_per_iteration", 0)
                for row in values
            ])),
            "paper_guided_unique_sequences_mean": float(np.mean([
                row.get("paper_guided_unique_sequences", 0)
                for row in values
            ])),
            "paper_guided_reuses_total": int(sum(
                row.get("paper_guided_reuses", 0) for row in values
            )),
            "reliability_hss_enabled_fraction": float(np.mean([
                row.get("reliability_hss_enabled", 0.0)
                for row in values
            ])),
            "reliability_authority_mean": float(np.mean([
                row.get("reliability_authority", 1.0)
                for row in values
            ])),
            "reliability_authority_min": float(np.min([
                row.get("reliability_authority", 1.0)
                for row in values
            ])),
            "reliability_authority_max": float(np.max([
                row.get("reliability_authority", 1.0)
                for row in values
            ])),
            "reliability_proposal_authority_mean": float(np.mean([
                row.get("reliability_proposal_authority", 1.0)
                for row in values
            ])),
            "reliability_proposal_authority_min": float(np.min([
                row.get("reliability_proposal_authority", 1.0)
                for row in values
            ])),
            "reliability_proposal_authority_max": float(np.max([
                row.get("reliability_proposal_authority", 1.0)
                for row in values
            ])),
            "reliability_proposal_fallback_fraction_mean": float(
                np.mean([
                    row.get(
                        "reliability_proposal_fallback_fraction", 0.0
                    )
                    for row in values
                ])
            ),
            "reliability_dynamics_confidence_mean": float(np.mean([
                row.get("reliability_dynamics_confidence", 1.0)
                for row in values
            ])),
            "reliability_dynamics_confidence_min": float(np.min([
                row.get("reliability_dynamics_confidence", 1.0)
                for row in values
            ])),
            "reliability_dynamics_confidence_max": float(np.max([
                row.get("reliability_dynamics_confidence", 1.0)
                for row in values
            ])),
            "reliability_dynamics_routing_mode": str(
                values[-1].get(
                    "reliability_dynamics_routing_mode",
                    "trust_weighted",
                )
            ),
            "reliability_model_routing_factor_mean": float(np.mean([
                row.get("reliability_model_routing_factor", 1.0)
                for row in values
            ])),
            "reliability_model_routing_factor_min": float(np.min([
                row.get("reliability_model_routing_factor", 1.0)
                for row in values
            ])),
            "reliability_model_routing_factor_max": float(np.max([
                row.get("reliability_model_routing_factor", 1.0)
                for row in values
            ])),
            "reliability_actor_support_confidence_mean": float(
                np.mean([
                    row.get("reliability_actor_confidence", 1.0)
                    for row in values
                ])
            ),
            "reliability_actor_support_confidence_min": float(
                np.min([
                    row.get("reliability_actor_confidence", 1.0)
                    for row in values
                ])
            ),
            "reliability_actor_support_confidence_max": float(
                np.max([
                    row.get("reliability_actor_confidence", 1.0)
                    for row in values
                ])
            ),
            "reliability_actor_competence_mean": float(np.mean([
                row.get(
                    "reliability_actor_competence_confidence", 1.0
                )
                for row in values
            ])),
            "reliability_actor_competence_min": float(np.min([
                row.get(
                    "reliability_actor_competence_confidence", 1.0
                )
                for row in values
            ])),
            "reliability_actor_competence_max": float(np.max([
                row.get(
                    "reliability_actor_competence_confidence", 1.0
                )
                for row in values
            ])),
            "reliability_actor_competence_raw_mean": float(np.mean([
                row.get(
                    "reliability_actor_competence_raw_confidence", 1.0
                )
                for row in values
            ])),
            "reliability_actor_competence_mapped_mean": float(np.mean([
                row.get(
                    "reliability_actor_competence_mapped_confidence", 1.0
                )
                for row in values
            ])),
            "reliability_actor_competence_guided_yield_mean": float(
                np.mean([
                    row.get(
                        "reliability_actor_competence_guided_yield",
                        0.0,
                    )
                    for row in values
                ])
            ),
            "reliability_actor_competence_gaussian_yield_mean": float(
                np.mean([
                    row.get(
                        "reliability_actor_competence_gaussian_yield",
                        0.0,
                    )
                    for row in values
                ])
            ),
            "reliability_low_fraction": float(np.mean([
                row.get("reliability_level", "disabled") == "low"
                for row in values
            ])),
            "reliability_medium_fraction": float(np.mean([
                row.get("reliability_level", "disabled") == "medium"
                for row in values
            ])),
            "reliability_high_fraction": float(np.mean([
                row.get("reliability_level", "disabled") == "high"
                for row in values
            ])),
            "reliability_guided_fraction_applied_mean": float(np.mean([
                row.get("reliability_guided_fraction_applied", 0.30)
                for row in values
            ])),
            "reliability_guided_fraction_next_mean": float(np.mean([
                row.get("reliability_guided_fraction_next", 0.30)
                for row in values
            ])),
            "reliability_guided_fraction_raw_applied_mean": float(np.mean([
                row.get("reliability_guided_fraction_raw_applied", 0.30)
                for row in values
            ])),
            "reliability_guided_fraction_raw_applied_min": float(np.min([
                row.get("reliability_guided_fraction_raw_applied", 0.30)
                for row in values
            ])),
            "reliability_guided_fraction_raw_next_mean": float(np.mean([
                row.get("reliability_guided_fraction_raw_next", 0.30)
                for row in values
            ])),
            "reliability_guided_fraction_raw_next_min": float(np.min([
                row.get("reliability_guided_fraction_raw_next", 0.30)
                for row in values
            ])),
            "terminal_guidance_floor_active_fraction": float(np.mean([
                row.get("terminal_guidance_floor_active", 0.0)
                for row in values
            ])),
            "terminal_guidance_tracking_active_fraction": float(np.mean([
                row.get("terminal_guidance_floor_active", 0.0)
                for row in values
                if row.get("target_phase") == "tracking"
            ])) if any(
                row.get("target_phase") == "tracking" for row in values
            ) else 0.0,
            "terminal_guidance_terminal_active_fraction": float(np.mean([
                row.get("terminal_guidance_floor_active", 0.0)
                for row in values
                if row.get("target_phase") in (
                    "terminal_approach", "terminal"
                )
            ])) if any(
                row.get("target_phase") in (
                    "terminal_approach", "terminal"
                )
                for row in values
            ) else 0.0,
            "completion_handover_enabled_fraction": float(np.mean([
                row.get("completion_handover_enabled", 0.0)
                for row in values
            ])),
            "completion_handover_authority_mean": float(np.mean([
                row.get("completion_handover_authority", 1.0)
                for row in values
            ])),
            "completion_handover_authority_min": float(np.min([
                row.get("completion_handover_authority", 1.0)
                for row in values
            ])),
            "terminal_value_completion_authority_mean": float(np.mean([
                row.get("terminal_value_completion_authority", 1.0)
                for row in values
            ])),
            "terminal_value_completion_authority_min": float(np.min([
                row.get("terminal_value_completion_authority", 1.0)
                for row in values
            ])),
            "paper_guided_elite_count_total": int(sum(
                row.get("paper_guided_elite_count", 0)
                for row in values
            )),
            "paper_gaussian_elite_count_total": int(sum(
                row.get("paper_gaussian_elite_count", 0)
                for row in values
            )),
            "rl_elite_fraction_mean": float(np.mean([
                row.get("rl_elite_fraction", 0.0) for row in values
            ])),
            "rl_elite_count_total": int(sum(
                row.get("rl_elite_count", 0) for row in values
            )),
            "shifted_elite_count_total": int(sum(
                row.get("shifted_elite_count", 0) for row in values
            )),
            "base_elite_count_total": int(sum(
                row.get("base_elite_count", 0) for row in values
            )),
            "covariance_scale_mean": float(np.mean([
                row.get("covariance_scale_mean", 1.0) for row in values
            ])),
            "terminal_value_enabled_fraction": float(np.mean([
                row.get("terminal_value_enabled", 0.0) for row in values
            ])),
            "terminal_value_conservative_enabled_fraction": float(
                np.mean([
                    row.get(
                        "terminal_value_conservative_enabled", 0.0
                    )
                    for row in values
                ])
            ),
            "terminal_value_authority_mean": float(np.mean([
                row.get("terminal_value_authority_mean", 1.0)
                for row in values
            ])),
            "terminal_value_authority_min": float(np.min([
                row.get("terminal_value_authority_min", 1.0)
                for row in values
            ])),
            "terminal_value_authority_max": float(np.max([
                row.get("terminal_value_authority_max", 1.0)
                for row in values
            ])),
            "terminal_value_dynamics_confidence_mean": float(np.mean([
                row.get(
                    "terminal_value_dynamics_confidence_mean", 1.0
                )
                for row in values
            ])),
            "terminal_value_critic_confidence_mean": float(np.mean([
                row.get(
                    "terminal_value_critic_confidence_mean", 1.0
                )
                for row in values
            ])),
            "terminal_value_uncertainty_mean": float(np.mean([
                row.get("terminal_value_uncertainty_mean", 0.0)
                for row in values
            ])),
            "terminal_value_uncertainty_cost_mean": float(np.mean([
                row.get("terminal_value_uncertainty_cost_mean", 0.0)
                for row in values
            ])),
            "terminal_q_mean": float(np.mean([
                row.get("terminal_q_mean", 0.0) for row in values
            ])),
            "terminal_q_disagreement_mean": float(np.mean([
                row.get("terminal_q_disagreement_mean", 0.0)
                for row in values
            ])),
            "residual_policy_context_enabled_fraction": float(np.mean([
                row.get("residual_policy_context_enabled", 0.0)
                for row in values
            ])),
            "residual_policy_predicted_abs_mean": float(np.mean([
                row.get("residual_policy_predicted_abs_mean", 0.0)
                for row in values
            ])),
            "residual_policy_predicted_abs_max": float(np.max([
                row.get("residual_policy_predicted_abs_max", 0.0)
                for row in values
            ])),
            "residual_policy_innovation_abs_mean": float(np.mean([
                row.get("residual_policy_innovation_abs_mean", 0.0)
                for row in values
            ])),
            "residual_policy_innovation_abs_max": float(np.max([
                row.get("residual_policy_innovation_abs_max", 0.0)
                for row in values
            ])),
            "residual_policy_disagreement_mean": float(np.mean([
                row.get("residual_policy_disagreement_mean", 0.0)
                for row in values
            ])),
            "residual_policy_support_mean": float(np.mean([
                row.get("residual_policy_support_mean", 0.0)
                for row in values
            ])),
            "residual_policy_innovation_valid_fraction": float(np.mean([
                row.get("residual_policy_innovation_valid_fraction", 0.0)
                for row in values
            ])),
            "residual_policy_authority_enabled_fraction": float(np.mean([
                row.get("residual_policy_authority_enabled", 0.0)
                for row in values
            ])),
            "residual_policy_authority_mean": float(np.mean([
                row.get("residual_policy_authority_mean", 0.0)
                for row in values
            ])),
            "residual_policy_authority_max": float(np.max([
                row.get("residual_policy_authority_max", 0.0)
                for row in values
            ])),
            "anytime_mean_samples": float(np.mean([
                row.get("anytime_selected_samples", 0) for row in values
            ])),
            "anytime_add_fraction": float(np.mean([
                row.get("anytime_add_samples", 0.0) for row in values
            ])),
            "anytime_predicted_advantage_mean": float(np.mean([
                row.get("anytime_predicted_advantage", 0.0)
                for row in values
            ])),
            "anytime_confidence_width_mean": float(np.mean([
                row.get("anytime_confidence_width", 0.0) for row in values
            ])),
            "anytime_decision_refresh_fraction": float(np.mean([
                row.get("anytime_decision_refreshed", 0.0) for row in values
            ])),
            "rl_gate_alpha_mean": float(
                np.mean([row.get("rl_gate_alpha", 0.0) for row in values])
            ),
            "rl_gate_fallback_steps": int(
                sum(
                    row.get("prior_type") == "rl_sac"
                    and row.get("rl_gate_alpha", 0.0) <= 1e-12
                    for row in values
                )
            ),
            "rl_policy_mode": str(values[-1].get("rl_policy_mode", "direct")),
            "rl_correction_gate_alpha_mean": float(np.mean([
                row.get("rl_correction_gate_alpha", 0.0) for row in values
            ])),
            "rl_base_action_abs_mean": float(np.mean([
                row.get("rl_base_action_abs_mean", 0.0) for row in values
            ])),
            "rl_unit_correction_abs_mean": float(np.mean([
                row.get("rl_unit_correction_abs_mean", 0.0) for row in values
            ])),
            "rl_applied_correction_abs_mean": float(np.mean([
                row.get("rl_applied_correction_abs_mean", 0.0) for row in values
            ])),
            "rl_applied_correction_abs_max": float(np.max([
                row.get("rl_applied_correction_abs_max", 0.0) for row in values
            ])),
            "rl_raw_applied_correction_abs_mean": float(np.mean([
                row.get("rl_raw_applied_correction_abs_mean", 0.0)
                for row in values
            ])),
            "rl_correction_advantage_gate_mode": str(values[-1].get(
                "rl_correction_advantage_gate_mode", "none"
            )),
            "rl_correction_advantage_critic_source": str(values[-1].get(
                "rl_correction_advantage_critic_source", "online"
            )),
            "rl_correction_advantage_threshold": float(values[-1].get(
                "rl_correction_advantage_threshold", 0.0
            )),
            "rl_correction_advantage_uncertainty_multiplier": float(
                values[-1].get(
                    "rl_correction_advantage_uncertainty_multiplier", 1.0
                )
            ),
            "rl_correction_advantage_gate_alpha_mean": float(np.mean([
                row.get("rl_correction_advantage_gate_alpha", 1.0)
                for row in values
            ])),
            "rl_correction_support_gate_enabled": bool(
                values[-1].get(
                    "rl_correction_support_gate_enabled", False
                )
            ),
            "rl_correction_support_confidence_mean": float(np.mean([
                row.get("rl_correction_support_confidence", 1.0)
                for row in values
            ])),
            "rl_correction_support_confidence_min": float(np.min([
                row.get("rl_correction_support_confidence", 1.0)
                for row in values
            ])),
            "rl_correction_effective_gate_alpha_mean": float(np.mean([
                row.get("rl_correction_effective_gate_alpha", 1.0)
                for row in values
            ])),
            "rl_unselected_critic_diagnostics_computed_fraction": float(
                np.mean([
                    row.get(
                        "rl_unselected_critic_diagnostics_computed", 0.0
                    )
                    for row in values
                ])
            ),
            "rl_selected_consensus_lcb_mean": float(np.mean([
                row.get("rl_selected_consensus_lcb", 0.0)
                for row in values
            ])),
            "rl_online_conservative_advantage_mean": float(np.mean([
                row.get("rl_online_conservative_advantage", 0.0)
                for row in values
            ])),
            "rl_online_conservative_advantage_min": float(np.min([
                row.get("rl_online_conservative_advantage", 0.0)
                for row in values
            ])),
            "rl_online_conservative_advantage_max": float(np.max([
                row.get("rl_online_conservative_advantage", 0.0)
                for row in values
            ])),
            "rl_online_positive_advantage_fraction": float(np.mean([
                row.get("rl_online_conservative_advantage", 0.0) >= 0.0
                for row in values
            ])),
            "rl_target_conservative_advantage_mean": float(np.mean([
                row.get("rl_target_conservative_advantage", 0.0)
                for row in values
            ])),
            "rl_target_conservative_advantage_min": float(np.min([
                row.get("rl_target_conservative_advantage", 0.0)
                for row in values
            ])),
            "rl_target_conservative_advantage_max": float(np.max([
                row.get("rl_target_conservative_advantage", 0.0)
                for row in values
            ])),
            "rl_target_positive_advantage_fraction": float(np.mean([
                row.get("rl_target_conservative_advantage", 0.0) >= 0.0
                for row in values
            ])),
            "rl_ood_score_mean": float(
                np.mean([row.get("rl_ood_score", 0.0) for row in values])
            ),
            "rl_ood_score_max": float(
                np.max([row.get("rl_ood_score", 0.0) for row in values])
            ),
            "rl_scene_complexity_score_mean": float(np.mean([
                row.get("rl_scene_complexity_score", 0.0) for row in values
            ])),
            "rl_scene_complexity_score_max": float(np.max([
                row.get("rl_scene_complexity_score", 0.0) for row in values
            ])),
            "rl_scene_complexity_front_proximity_mean": float(np.mean([
                row.get("rl_scene_complexity_front_proximity", 0.0)
                for row in values
            ])),
            "rl_scene_complexity_constriction_mean": float(np.mean([
                row.get("rl_scene_complexity_constriction", 0.0)
                for row in values
            ])),
            "rl_scene_complexity_density_mean": float(np.mean([
                row.get("rl_scene_complexity_density", 0.0)
                for row in values
            ])),
            "rl_scene_complexity_scan_valid_fraction": float(np.mean([
                row.get("rl_scene_complexity_scan_valid", 0.0)
                for row in values
            ])),
            "rl_temporal_closing_gate_alpha_mean": float(np.mean([
                row.get("rl_temporal_closing_gate_alpha", 0.0)
                for row in values
            ])),
            "rl_temporal_closing_gate_alpha_max": float(np.max([
                row.get("rl_temporal_closing_gate_alpha", 0.0)
                for row in values
            ])),
            "rl_temporal_closing_rate_mps_max": float(np.max([
                row.get("rl_temporal_closing_rate_mps", 0.0)
                for row in values
            ])),
            "rl_temporal_closing_held_fraction": float(np.mean([
                row.get("rl_temporal_closing_held", 0.0)
                for row in values
            ])),
            "rl_hazard_activation_mean": float(np.mean([
                row.get("rl_hazard_activation", 0.0) for row in values
            ])),
            "rl_competence_confidence_mean": float(np.mean([
                row.get("rl_competence_confidence", 0.0) for row in values
            ])),
            "rl_baseline_progress_m_mean": float(np.mean([
                row.get("rl_baseline_progress_m", 0.0) for row in values
            ])),
            "rl_baseline_progress_gate_ready_fraction": float(np.mean([
                row.get("rl_baseline_progress_gate_ready", 0.0)
                for row in values
            ])),
            "rl_baseline_stagnation_activation_mean": float(np.mean([
                row.get("rl_baseline_stagnation_activation", 0.0)
                for row in values
            ])),
            "rl_baseline_stagnation_activation_max": float(np.max([
                row.get("rl_baseline_stagnation_activation", 0.0)
                for row in values
            ])),
            "rl_baseline_stagnation_held_fraction": float(np.mean([
                row.get("rl_baseline_stagnation_held", 0.0)
                for row in values
            ])),
            "rl_gate_active_fraction": float(np.mean([
                row.get("rl_gate_alpha", 0.0) > 0.05 for row in values
            ])),
            "rl_learned_inference_skip_fraction": float(np.mean([
                row.get("rl_learned_inference_skipped", 0.0)
                for row in values
            ])),
            "profile_prior_fallback_ms_mean": float(np.mean([
                row.get("profile_prior_fallback_ms", 0.0)
                for row in values
            ])),
            "profile_prior_encode_normalize_ms_mean": float(np.mean([
                row.get("profile_prior_encode_normalize_ms", 0.0)
                for row in values
            ])),
            "profile_prior_gate_features_ms_mean": float(np.mean([
                row.get("profile_prior_gate_features_ms", 0.0)
                for row in values
            ])),
            "profile_prior_actor_ms_mean": float(np.mean([
                row.get("profile_prior_actor_ms", 0.0)
                for row in values
            ])),
            "profile_prior_advantage_ms_mean": float(np.mean([
                row.get("profile_prior_advantage_ms", 0.0)
                for row in values
            ])),
            "profile_prior_decoder_ms_mean": float(np.mean([
                row.get("profile_prior_decoder_ms", 0.0)
                for row in values
            ])),
            "profile_prior_outer_gate_ms_mean": float(np.mean([
                row.get("profile_prior_outer_gate_ms", 0.0)
                for row in values
            ])),
            "profile_prior_total_ms_mean": float(np.mean([
                row.get("profile_prior_total_ms", 0.0)
                for row in values
            ])),
            "rl_critic_disagreement_mean": float(np.mean([
                row.get("rl_critic_disagreement", 0.0) for row in values
            ])),
            "rl_critic_disagreement_max": float(np.max([
                row.get("rl_critic_disagreement", 0.0) for row in values
            ])),
            "rl_exploration_activation_mean": float(np.mean([
                row.get("rl_exploration_activation", 0.0) for row in values
            ])),
            "rl_exploration_latch_alpha_mean": float(np.mean([
                row.get("rl_exploration_latch_alpha", 0.0) for row in values
            ])),
            "rl_subgoal_distance_mean": float(
                np.mean([row.get("rl_subgoal_distance", 0.0) for row in values])
            ),
            "rl_subgoal_abs_bearing_mean": float(
                np.mean([
                    abs(row.get("rl_subgoal_bearing", 0.0)) for row in values
                ])
            ),
        }
