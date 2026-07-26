"""Training-only upper-bound teacher for complex mixed environments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from experiments.dynamic_uncertainty.complex_full_method import (
    ROOT,
    build_complex_full_config,
)


DEFAULT_PROTOCOL = (
    ROOT / "configs/research/complex_actor_teacher_probe_v1.yaml"
)


def load_teacher_protocol(path=DEFAULT_PROTOCOL):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("complex Actor teacher protocol must be a mapping")
    if value.get("protocol") != "complex_actor_teacher_upper_bound_v1":
        raise ValueError("complex Actor teacher protocol mismatch")
    return value


def build_complex_actor_teacher_config(
    map_name,
    seed,
    *,
    protocol_path=DEFAULT_PROTOCOL,
    maximum_steps=None,
):
    """Build a causal, higher-compute teacher without changing risk limits."""

    full = build_complex_full_config(map_name, seed)
    result = deepcopy(full)
    protocol = load_teacher_protocol(protocol_path)
    teacher = dict(protocol["teacher"])
    planner = result["planner"]
    frozen_risk_contract = {
        key: deepcopy(value)
        for key, value in planner.items()
        if key.startswith("probabilistic_obstacle_")
    }
    frozen_static_contract = {
        key: deepcopy(value)
        for key, value in planner.items()
        if key.startswith("known_static_map_")
    }

    planner.update({
        "optimizer": str(teacher["optimizer"]),
        "sampling_prior": str(teacher["sampling_prior"]),
        "horizon": int(teacher["horizon"]),
        "num_samples": int(teacher["num_samples"]),
    })
    # The standard teacher must not instantiate or consult the deployed
    # Actor. ICODE and the matched residual/nominal safety shield remain
    # active because they are configured independently of the sampling prior.
    result["rl"]["enabled"] = False
    planner["residual_safety_shield"]["rl_hss_integration"] = ""
    result["experiment"].update({
        "name": "complex_actor_teacher__%s__seed%d" % (
            map_name, int(seed)
        ),
        "max_steps": int(
            maximum_steps
            if maximum_steps is not None
            else teacher["maximum_steps"]
        ),
        "complex_actor_teacher_probe": True,
        "formal_experiment_started": False,
    })
    result["complex_actor_teacher_contract"] = {
        "deployment_method_changed": False,
        "training_only": True,
        "future_truth_used": bool(teacher["future_truth_used"]),
        "dynamic_obstacles_in_astar": bool(
            teacher["dynamic_obstacles_in_astar"]
        ),
        "probability_thresholds_changed": bool(
            teacher["probability_thresholds_changed"]
        ),
        "safety_contract_changed": bool(
            teacher["safety_contract_changed"]
        ),
        "teacher_optimizer": planner["optimizer"],
        "teacher_sampling_prior": planner["sampling_prior"],
        "teacher_horizon": planner["horizon"],
        "teacher_num_samples": planner["num_samples"],
        "actor_enabled": bool(result["rl"]["enabled"]),
        "source_full_rollouts_per_decision": int(
            full["planner"]["num_samples"]
            * full["planner"]["paper_rl_driven"]["iterations"]
        ),
    }

    if planner["optimizer"] != "standard":
        raise ValueError("complex Actor teacher must use standard MPPI")
    if planner["sampling_prior"] != "goal_warm_start":
        raise ValueError("complex Actor teacher cannot use the Actor")
    if result["rl"]["enabled"]:
        raise ValueError("complex Actor teacher must disable Actor inference")
    if {
        key: planner[key] for key in frozen_risk_contract
    } != frozen_risk_contract:
        raise ValueError("complex Actor teacher changed probability risk")
    if {
        key: planner[key] for key in frozen_static_contract
    } != frozen_static_contract:
        raise ValueError("complex Actor teacher changed static geometry")
    if result["complex_actor_teacher_contract"]["future_truth_used"]:
        raise ValueError("complex Actor teacher must remain causal")
    if result["complex_actor_teacher_contract"][
        "dynamic_obstacles_in_astar"
    ]:
        raise ValueError("dynamic obstacles cannot enter static A*")
    return result


__all__ = [
    "DEFAULT_PROTOCOL",
    "build_complex_actor_teacher_config",
    "load_teacher_protocol",
]
