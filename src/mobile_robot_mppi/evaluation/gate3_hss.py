"""Preregistered paired design and decision logic for Gate 3 HSS."""

import numpy as np


def paired_hss_schedule(seeds, domains, scenes, schedule_seed):
    """Randomize fixed/adaptive order inside every repeated-measures block."""

    arms = ("fixed", "adaptive")
    rng = np.random.RandomState(int(schedule_seed))
    jobs = []
    for scene in scenes:
        for domain in domains:
            for seed in seeds:
                block = "%s__%s__seed_%d" % (
                    scene["name"], domain["name"], int(seed)
                )
                for position, arm_index in enumerate(rng.permutation(2)):
                    jobs.append({
                        "arm": arms[int(arm_index)],
                        "scene": str(scene["name"]),
                        "physics_domain": str(domain["name"]),
                        "seed": int(seed),
                        "block": block,
                        "run_order_within_block": int(position),
                    })
    return jobs


def gate3_development_decision(
    comparison,
    adaptive_rows,
    goal_worsening_fraction=0.02,
    jerk_worsening_fraction=0.01,
    compute_worsening_fraction=0.10,
    tolerance=1e-12,
):
    """Apply the frozen Gate 3B equal-budget engineering criteria."""

    metrics = comparison["metrics"]
    success = metrics["success"]
    collision = metrics["collision"]
    goal = metrics["final_goal_distance"]
    jerk = metrics["control_jerk"]
    stuck = metrics["stuck_steps"]
    compute = metrics["planner_compute_ms_mean"]
    rollouts = metrics["paper_total_rollouts_mean"]
    success_ok = success["favorable_effect"] >= -tolerance
    collision_ok = collision["favorable_effect"] >= -tolerance
    goal_ok = goal["favorable_effect"] >= (
        -float(goal_worsening_fraction)
        * max(abs(float(goal["control_mean"])), tolerance)
    )
    jerk_ok = jerk["favorable_effect"] >= (
        -float(jerk_worsening_fraction)
        * max(abs(float(jerk["control_mean"])), tolerance)
    )
    compute_ok = float(compute["aligned_mean"]) <= (
        (1.0 + float(compute_worsening_fraction))
        * max(float(compute["control_mean"]), tolerance)
    )
    rollout_ok = abs(
        float(rollouts["aligned_mean"]) - float(rollouts["control_mean"])
    ) <= tolerance
    primary = any(
        metrics[name]["favorable_effect"] > tolerance
        for name in ("success", "final_goal_distance", "stuck_steps")
    )
    low_fraction = float(np.mean([
        float(row.get("reliability_low_fraction", 0.0))
        for row in adaptive_rows
    ]))
    nonlow_fraction = float(np.mean([
        float(row.get("reliability_medium_fraction", 0.0))
        + float(row.get("reliability_high_fraction", 0.0))
        for row in adaptive_rows
    ]))
    authority_exercised = (
        low_fraction > tolerance and nonlow_fraction > tolerance
    )
    passed = all((
        primary,
        success_ok,
        collision_ok,
        goal_ok,
        jerk_ok,
        compute_ok,
        rollout_ok,
        authority_exercised,
    ))
    return {
        "positive_primary_outcome": bool(primary),
        "success_not_worse": bool(success_ok),
        "collision_not_worse": bool(collision_ok),
        "goal_within_two_percent_margin": bool(goal_ok),
        "jerk_within_one_percent_margin": bool(jerk_ok),
        "serial_compute_within_ten_percent_margin": bool(compute_ok),
        "equal_model_rollout_budget": bool(rollout_ok),
        "low_and_nonlow_authority_exercised": bool(
            authority_exercised
        ),
        "adaptive_low_step_fraction": low_fraction,
        "adaptive_nonlow_step_fraction": nonlow_fraction,
        "closed_loop_development_passed": bool(passed),
    }

