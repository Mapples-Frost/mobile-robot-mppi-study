"""Resolve the current Full Proposed stack for map-free Pi 5 shadow use."""

from copy import deepcopy
from pathlib import Path

from experiments.dynamic_uncertainty.complex_full_method import (
    build_complex_full_config,
)


def _weight(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError("required deployment weight missing: %s" % path)
    return str(path)


def _physical_front_envelope(max_v_mps):
    """Front ranges for a continuous clearance-to-speed governor.

    The hard stop is deliberately a close, fixed last line of defence.  The
    measured loop latency is handled by reducing forward speed continuously
    before that line, rather than turning the full maximum-speed stopping
    distance into an abrupt stop threshold.
    """
    speed = float(max_v_mps)
    if not 0.0 < speed <= 0.50:
        raise ValueError("physical forward speed limit must be in (0, 0.50]")
    # 0.50 s covers the observed 0.318 s p95 loop plus transport/scheduling
    # margin.  0.80 m/s^2 is deliberately below an unmeasured emergency
    # braking claim; a physical braking test may tighten it later.
    reaction_s = 0.50
    assumed_deceleration_mps2 = 0.80
    base_clearance_m = 0.35
    hard = 0.50
    soft = hard
    slow = max(
        hard,
        base_clearance_m
        + speed * reaction_s
        + speed * speed / (2.0 * assumed_deceleration_mps2),
    )
    return {
        "hard_stop_distance_m": float(hard),
        "soft_block_distance_m": float(soft),
        "slow_distance_m": float(slow),
        "reaction_time_s": reaction_s,
        "assumed_deceleration_mps2": assumed_deceleration_mps2,
        "base_clearance_m": base_clearance_m,
    }


_PI5_ALGORITHM_FEATURES = (
    "actor_guidance",
    "hss_reliability",
    "residual_learning",
    "change_aware_prediction",
    "probabilistic_risk",
    "ar1_sampling",
    "forward_passage",
)


def resolve_pi5_algorithm_features(
    *,
    full_proposed=False,
    traditional_mppi=False,
    enable_actor_guidance=False,
    enable_hss_reliability=False,
    enable_residual_learning=False,
    enable_change_aware_prediction=False,
    enable_probabilistic_risk=False,
    enable_ar1_sampling=False,
    enable_forward_passage=False,
    disable_residual_learning=False,
):
    """Resolve explicit, auditable real-robot algorithm ablations.

    With no positive feature switch the result is conventional MPPI.  The
    Full Proposed preset enables every enhancement, while the legacy residual
    disable switch may still remove only ICODE/residual shielding from it.
    """

    if bool(full_proposed) and bool(traditional_mppi):
        raise ValueError("Full Proposed and traditional MPPI are exclusive")
    explicit = {
        "actor_guidance": bool(enable_actor_guidance),
        "hss_reliability": bool(enable_hss_reliability),
        "residual_learning": bool(enable_residual_learning),
        "change_aware_prediction": bool(enable_change_aware_prediction),
        "probabilistic_risk": bool(enable_probabilistic_risk),
        "ar1_sampling": bool(enable_ar1_sampling),
        "forward_passage": bool(enable_forward_passage),
    }
    if bool(traditional_mppi) and any(explicit.values()):
        raise ValueError(
            "traditional MPPI cannot be combined with enhancement switches"
        )
    if bool(enable_residual_learning) and bool(disable_residual_learning):
        raise ValueError(
            "residual learning cannot be both enabled and disabled"
        )
    features = {
        name: bool(full_proposed) or explicit[name]
        for name in _PI5_ALGORITHM_FEATURES
    }
    if bool(disable_residual_learning):
        features["residual_learning"] = False
    if features["hss_reliability"] and not features["actor_guidance"]:
        raise ValueError("HSS reliability requires Actor guidance")
    if features["probabilistic_risk"] and not features[
        "change_aware_prediction"
    ]:
        raise ValueError(
            "probabilistic risk requires change-aware obstacle prediction"
        )
    if features["forward_passage"] and not features["probabilistic_risk"]:
        raise ValueError("Forward Passage requires probabilistic risk")
    return features


def apply_pi5_algorithm_features(config, features):
    """Apply a resolved ablation profile without weakening common safety."""

    unknown = sorted(set(features) - set(_PI5_ALGORITHM_FEATURES))
    missing = sorted(set(_PI5_ALGORITHM_FEATURES) - set(features))
    if unknown or missing:
        raise ValueError(
            "algorithm feature contract mismatch: unknown=%s missing=%s"
            % (unknown, missing)
        )
    enabled = {name: bool(features[name]) for name in _PI5_ALGORITHM_FEATURES}
    if enabled["hss_reliability"] and not enabled["actor_guidance"]:
        raise ValueError("HSS reliability requires Actor guidance")
    if enabled["probabilistic_risk"] and not enabled[
        "change_aware_prediction"
    ]:
        raise ValueError(
            "probabilistic risk requires change-aware obstacle prediction"
        )
    if enabled["forward_passage"] and not enabled["probabilistic_risk"]:
        raise ValueError("Forward Passage requires probabilistic risk")

    planner = config["planner"]
    paper = planner["paper_rl_driven"]
    shield = planner["residual_safety_shield"]
    if not enabled["actor_guidance"]:
        # Preserve the Full treatment's total rollout budget: Paper RL uses
        # K=300 for two iterations, whereas standard MPPI has one iteration.
        planner["num_samples"] = int(planner["num_samples"]) * int(
            paper.get("iterations", 1)
        )
        planner["optimizer"] = "standard"
        planner["sampling_prior"] = "goal_warm_start"
        config["rl"]["enabled"] = False
        shield["rl_hss_integration"] = ""

    if not enabled["hss_reliability"]:
        paper["reliability"]["enabled"] = False
        paper["reliability_sidecar"] = {}

    if not enabled["residual_learning"]:
        planner["prediction_mode"] = "nominal"
        planner["residual_device_rollout_enabled"] = False
        planner["residual_cuda_graph_enabled"] = False
        shield["enabled"] = False

    tracker = config["perception"]["dynamic_obstacle_tracker"]
    tracker["enabled"] = bool(enabled["change_aware_prediction"])
    if enabled["change_aware_prediction"]:
        tracker["predictor_mode"] = "change_aware"

    if not enabled["probabilistic_risk"]:
        # Disable the entire predictive decision family, not just its running
        # cost, so a risk-off ablation cannot retain an emergency candidate,
        # recovery turn, speed governor, or reference-authority side effect.
        for key in tuple(planner):
            if key.startswith("probabilistic_") and key.endswith("_enabled"):
                planner[key] = False
        scan_guard = config["perception"]["scan_guard"]
        for key in (
            "dynamic_escape_enabled",
            "dynamic_escape_reactive_enabled",
            "dynamic_recovery_enabled",
            "dynamic_recovery_translation_enabled",
            "dynamic_escape_probability_mass_enabled",
            "dynamic_escape_hold_enabled",
            "dynamic_escape_use_vetted_planner_control",
        ):
            scan_guard[key] = False

    planner["noise_basis"] = (
        "ar1:2.0" if enabled["ar1_sampling"] else "iid"
    )
    mode = (
        "full_proposed"
        if all(enabled.values())
        else "traditional_mppi"
        if not any(enabled.values())
        else "custom_ablation"
    )
    config["real_robot_deployment"].update({
        "algorithm_profile": mode,
        "algorithm_features": dict(enabled),
        # These common layers deliberately remain active in every profile.
        "common_safety_features": {
            "local_scan_obstacle_geometry": True,
            "physical_action_limits": True,
            "near_body_hard_stop": True,
            "temporal_ttc_fallback": True,
            "remote_controller_can_interlocks": True,
        },
    })
    return config


def build_pi5_full_config(
    project_root,
    seed=20260803,
    goal_x=3.0,
    goal_y=0.0,
    max_v_mps=None,
    max_reverse_v_mps=None,
    max_omega_radps=None,
):
    """Return B11 Full Proposed with only physical-runtime adaptations.

    The Chapter 2 geometry is used solely to resolve the latest shared B11
    algorithm block.  All simulator scene truth is discarded immediately.
    The resulting controller sees only the real scan and causal forecasts.
    """

    root = Path(project_root).resolve()
    config = deepcopy(build_complex_full_config(
        "chapter2_admissible", int(seed), arm="B11_full_proposed"
    ))
    config["experiment"].update({
        "name": "pi5_scout_mapfree_full_proposed_shadow",
        "seed": int(seed),
        "initial_state": [0.0, 0.0, 0.0, 0.0, 0.0],
        "max_steps": 1000000,
        "real_robot_shadow": True,
        "complex_map": "none_mapfree_real_robot",
        "complex_map_source": None,
    })
    # A two-point polyline gives traversal logic metric progress/tangent
    # geometry without claiming that a static global map exists.
    config["task"] = {
        "type": "polyline",
        "points": [[0.0, 0.0], [float(goal_x), float(goal_y)]],
        "position": [float(goal_x), float(goal_y)],
        "position_tolerance": 0.30,
        "lookahead_distance": 1.40,
        "terminal_approach_distance": 1.15,
        "projection_backtrack_distance": 1.35,
        "projection_forward_distance": 2.65,
        "corridor_half_width": 0.90,
        "footprint_radius": 0.25,
    }
    config["scene"] = {"obstacles": []}

    physical_limits = None
    supplied_limits = (
        max_v_mps,
        max_reverse_v_mps,
        max_omega_radps,
    )
    if any(value is not None for value in supplied_limits):
        if not all(value is not None for value in supplied_limits):
            raise ValueError("all physical action limits must be supplied together")
        max_v_mps = float(max_v_mps)
        max_reverse_v_mps = float(max_reverse_v_mps)
        max_omega_radps = float(max_omega_radps)
        if not 0.0 < max_reverse_v_mps <= 0.30:
            raise ValueError("physical reverse speed limit must be in (0, 0.30]")
        if not 0.0 < max_omega_radps <= 0.60:
            raise ValueError("physical angular speed limit must be in (0, 0.60]")
        envelope = _physical_front_envelope(max_v_mps)
        names = tuple(config["action_space"]["names"])
        v_index = names.index("v_cmd")
        omega_index = names.index("omega_cmd")
        config["action_space"]["lower"][v_index] = -max_reverse_v_mps
        config["action_space"]["upper"][v_index] = max_v_mps
        config["action_space"]["lower"][omega_index] = -max_omega_radps
        config["action_space"]["upper"][omega_index] = max_omega_radps
        scan_guard = config["perception"]["scan_guard"]
        scan_guard.update({
            "front_stop_distance": envelope["hard_stop_distance_m"],
            "hard_stop_distance": envelope["hard_stop_distance_m"],
            "front_soft_block_distance": envelope["soft_block_distance_m"],
            "front_soft_block_max_speed": 0.0,
            "front_slow_distance": envelope["slow_distance_m"],
            # The legacy linear slow scale has a non-zero floor.  Use a
            # physically interpretable clearance-to-speed cap instead.
            "front_slow_min_scale": 1.0,
            "physical_front_speed_governor_enabled": True,
            "physical_front_speed_governor_clearance_m": (
                envelope["base_clearance_m"]
            ),
            "physical_front_speed_governor_reaction_s": (
                envelope["reaction_time_s"]
            ),
            "physical_front_speed_governor_deceleration_mps2": (
                envelope["assumed_deceleration_mps2"]
            ),
            # Keep the protected front/side sector fail-closed, while a
            # rear-only return cannot stop continued forward translation.
            # Reverse commands still stop for rear obstacles.
            "directional_motion_guard_enabled": True,
            "directional_forward_protected_half_angle_deg": 100.0,
            "directional_motion_minimum_speed_mps": 0.02,
            # A person approaching from the rear is the one directional case
            # where waiting makes the interaction worse.  With at least
            # 0.90 m freshly observed front clearance, leave continuously in
            # forward gear even if this MPPI sample proposed reverse.  Mixed
            # or front evidence never enters this branch.
            "rear_pass_through_force_forward_enabled": True,
            "rear_pass_through_min_front_clearance_m": 0.90,
            "rear_pass_through_min_forward_speed_mps": min(
                0.35, max_v_mps
            ),
            "rear_pass_through_max_omega_radps": min(
                0.30, max_omega_radps
            ),
            "rear_pass_through_min_turn_omega_radps": min(
                0.15, max_omega_radps
            ),
            # Hold the first causal rear-side steering sign through up to
            # three missing/side-switching scans.  A person cannot physically
            # cross behind the chassis in this 0.3 s interval, whereas Livox
            # leg fragments can alternate sides on consecutive packets.
            "rear_pass_through_direction_release_steps": 4,
            # Once CA-IMM predicts a crossing, select and commit an avoidance
            # side while there is still room to pass.  The 041158 trace first
            # entered active escape at TTC=1.21 s, which was visibly late.
            "dynamic_escape_trigger_ttc_s": 3.00,
            # Real point-cloud fragmentation made independently selected
            # emergency candidates alternate full forward/reverse and left/
            # right commands.  A front-half-plane crossing instead receives
            # one short forecast-vetted forward arc; near-body hard stop still
            # preempts the commitment immediately.
            "dynamic_escape_uncertainty_fusion_enabled": True,
            # Respect the physical reverse envelope when the probabilistic
            # planner or the bounded close-range transaction selects reverse.
            "dynamic_escape_reverse_speed": max_reverse_v_mps,
            # Four full-yaw cycles establish a visible passage side.  Further
            # live-threat cycles coast on the tangent instead of continuing a
            # saturated circle, so the detour cannot wind up into a large arc.
            "dynamic_escape_direction_commit_steps": 4,
            # A frontal approach produced two false left/right reversals from
            # fragmented leg-cluster velocity estimates (085438 cycles 46 and
            # 53: only 0.13--0.14 m/s lateral).  Keep the selected side unless
            # a genuinely lateral crossing/reversal supplies stronger motion.
            "dynamic_escape_direction_refresh_minimum_lateral_speed_mps": 0.35,
            "dynamic_escape_coast_direction_lock_enabled": True,
            # Once the finite full-yaw prefix has selected a passage side,
            # continue tracking the forecast-relative heading with bounded yaw
            # instead of freezing omega at zero on a stale tangent.
            "dynamic_escape_coast_turn_gain": 0.55,
            "dynamic_escape_coast_max_omega_radps": min(
                0.30, max_omega_radps
            ),
            # A continuous human encounter may consume the short geometric
            # arc only once.  Forecast fragmentation no longer restarts six
            # saturated turn steps until five genuinely clear cycles re-arm
            # the mechanism.
            "dynamic_escape_geometric_single_commit_enabled": True,
            "dynamic_escape_geometric_rearm_clear_steps": 3,
            # A person inside the 0.50 m forward hard-stop envelope still
            # forbids forward translation.  Establish a side in place, then
            # permit six short reverse commands only with positively observed
            # rear headroom.  At 0.30 m/s and 10 Hz this transaction travels at
            # most 0.18 m before it must re-arm on fresh evidence.
            "dynamic_escape_hard_stop_enabled": True,
            "dynamic_escape_hard_stop_turn_steps": 3,
            "dynamic_escape_hard_stop_reverse_steps": 6,
            "dynamic_escape_hard_stop_reverse_speed": max_reverse_v_mps,
            "dynamic_escape_hard_stop_min_rear_range": 0.80,
            "dynamic_escape_hard_stop_rear_sector_deg": 120.0,
            # A close crowd swaps the selected leg/person track frequently.
            # Do not interpret that identity change as permission to restart
            # an already bounded turn transaction.
            "dynamic_escape_hard_stop_direction_refresh_enabled": False,
            # If the rear was blocked during the finite transaction and later
            # opens, take that exit once without replaying the turn phase.
            "dynamic_escape_hard_stop_rear_clear_retry_enabled": True,
            "dynamic_escape_hard_stop_rear_clear_retry_steps": 6,
            # Include any stop-only temporal turn immediately preceding the
            # close-range transaction in one 0.9 s yaw budget.  Once consumed,
            # wait or translate instead of continuing to rotate in a crowd.
            "dynamic_escape_max_zero_translation_turn_steps": 9,
        })
        local_layer = config["perception"]["local_obstacle_layer"]
        local_layer.update({
            # Unknown/static scan returns remain hard footprint constraints
            # even when the dynamic tracker has not yet produced a forecast.
            # Sixteen nearest circles cover a person or compact obstacle while
            # keeping the real-time candidate-clearance pass bounded.
            "local_obstacle_hard_filter_enabled": True,
            "local_obstacle_hard_filter_max_obstacles": 16,
        })
        temporal_guard = config["perception"]["temporal_scan_guard"]
        temporal_guard.update({
            # Independent last-mile protection when a moving person has not
            # yet satisfied the multi-frame identity/forecast gates.  This
            # reacts to measured closing TTC, not distance alone.
            "enabled": True,
            "safety_enabled": True,
            "safety_hard_stop_ttc_s": 0.80,
            "safety_slow_ttc_s": 2.00,
            "safety_slow_scale": 0.25,
            "ego_motion_compensation_enabled": True,
            "safety_continuous_slowdown_enabled": True,
        })
        physical_limits = {
            "max_v_mps": max_v_mps,
            "max_reverse_v_mps": max_reverse_v_mps,
            "max_omega_radps": max_omega_radps,
            "front_safety_envelope": envelope,
        }

    # make_components currently constructs a plant/sensor pair even though the
    # physical runtime never steps it.  A dependency-light kinematic instance
    # prevents any MuJoCo import or simulator ground truth from entering the
    # physical control loop.
    config["plant"]["backend"] = "legacy_kinematic"

    planner = config["planner"]
    planner.update({
        "device": "cpu",
        "checkpoint": _weight(
            root,
            "weights/icode_stage2_task_aware.pt",
        ),
        "residual_device_rollout_enabled": True,
        "residual_cuda_graph_enabled": False,
        "profile_components": True,
        # The perception facade injects the current causal local scan into the
        # existing hard static-feasibility path.  This prevents the weighted
        # MPPI update from averaging individually safe samples into a command
        # that crosses an unclassified person or static obstacle.
        "known_static_map_candidate_filter_enabled": True,
        # The same per-cycle causal scan geometry supplies the exact footprint
        # cost required by the hard candidate filter below.  Cache keys include
        # obstacle content, so geometry from an earlier scan cannot leak into
        # the current solve.
        "known_static_map_cost_enabled": True,
        "static_astar_replan_enabled": False,
        "path_boundary_enabled": False,
        "path_boundary_candidate_filter_enabled": False,
        "probabilistic_obstacle_missing_forecast_action": "scan_only",
        "probabilistic_obstacle_emergency_candidate_trigger_ttc_s": 3.00,
        "probabilistic_obstacle_emergency_candidate_trigger_distance_m": 1.20,
        "probabilistic_obstacle_emergency_candidate_rearm_ttc_s": 3.40,
        # Real differential drive cannot instantly realize a counterflow vector
        # that lies behind it.  Significant lateral human motion therefore
        # selects a forward-biased direction on the opposite body side; direct
        # approaches (small lateral component) continue to use live side
        # clearance in the physical arbiter.
        "probabilistic_obstacle_forward_lateral_countermotion_enabled": True,
        "probabilistic_obstacle_forward_lateral_countermotion_weight": 1.20,
        "probabilistic_obstacle_forward_lateral_minimum_speed_mps": 0.10,
        "probabilistic_obstacle_forward_lateral_minimum_fraction": 0.25,
        "probabilistic_obstacle_forward_lateral_reversal_confirm_steps": 2,
        "safety_recovery_translation_stop_threshold": 0.02,
        # The base complex builder predates the closed-loop noise-basis Gate.
        # All later successful Full deployments froze the accepted treatment.
        "noise_basis": "ar1:2.0",
    })
    # Preserve the frozen Full Proposed stochastic/search contract.
    if str(planner.get("noise_basis")) != "ar1:2.0":
        raise RuntimeError("latest Full Proposed must resolve noise_basis=ar1:2.0")
    iterations = int(planner["paper_rl_driven"].get("iterations", 1))
    if int(planner["num_samples"]) * iterations != 600:
        raise RuntimeError("latest Full Proposed must preserve 600 rollouts")

    config["rl"].update({
        "device": "cpu",
        "checkpoint": _weight(root, "weights/actor_full_proposed.pt"),
        # The frozen Actor was trained on wider physical bounds.  Its outputs
        # remain frozen, then every proposal and rollout is clipped into the
        # narrower controller action space before dynamics or cost evaluation.
        "allow_controller_action_subset": bool(physical_limits is not None),
        # The requested physical envelope is narrower than the checkpoint on
        # reverse/yaw but wider on forward speed.  Keep the frozen Actor's
        # decoding bounds and let MPPI use the complete physical envelope.
        "allow_actor_action_subspace": bool(physical_limits is not None),
    })
    sidecar = planner["paper_rl_driven"]["reliability_sidecar"]
    sidecar["device"] = "cpu"
    # The Actor executes many small autoregressive batches.  The global
    # two-thread setting installed while loading HSS made those tiny MLP calls
    # slower through thread-launch overhead (measured 16.5 ms network time per
    # cycle at two threads).  One thread also keeps the three-member HSS pass
    # below 1 ms on the deployment CPU.
    sidecar["torch_num_threads"] = 1
    sidecar["checkpoints"] = [
        _weight(root, "weights/hss_member%d.pt" % index)
        for index in (1, 2, 3)
    ]
    planner["residual_torch_num_threads"] = 2

    tracker = config["perception"]["dynamic_obstacle_tracker"]
    tracker.update({
        "maximum_tracks": 3,
        # The point adapter transforms measurements into base_link before
        # creating LaserScan, so no second sensor translation is permitted.
        "sensor_forward_offset_m": 0.0,
        "known_static_filter_enabled": False,
        "known_static_track_rejection_enabled": False,
        # MotionBootstrap already restricts track creation to measured motion,
        # and MaplessStaticDynamicFilter independently decides which CA-IMM
        # forecasts may reach MPPI.  Requiring the low-level tracker to
        # re-confirm instantaneous speed on every scan created 1--3 frame
        # forecast holes when a person slowed or the lidar cluster fragmented.
        # Disable only this redundant real-robot gate; the mapless static
        # rejection layer remains the forecast publication authority.
        "motion_confirmation_enabled": False,
    })
    config["real_robot_deployment"] = {
        "contract": "pi5_scout_mapfree_full_proposed_shadow_v1",
        "algorithm": "B11_full_proposed",
        "publish_enabled": False,
        "can_mode": "zero_only",
        "hardware": {
            "computer": "raspberry_pi_5_aarch64",
            "lidar": "Livox Mid-360",
            "base": "AgileX SCOUT MINI",
            "can_interface": "can0",
        },
        "perception_contract": {
            "raw_scan_for_geometry_and_safety": True,
            "packet_timestamp_deskew_for_dynamic_tracking": True,
            "mapless_motion_bootstrap_for_dynamic_tracking": True,
            "mapless_static_dynamic_forecast_filter": True,
            "closing_ttc_safety_fallback": True,
            "simulator_truth_used": False,
            "static_map_used": False,
        },
    }
    if physical_limits is not None:
        config["real_robot_deployment"]["physical_action_limits"] = (
            physical_limits
        )
    return config


__all__ = [
    "build_pi5_full_config",
    "_physical_front_envelope",
    "resolve_pi5_algorithm_features",
    "apply_pi5_algorithm_features",
]
