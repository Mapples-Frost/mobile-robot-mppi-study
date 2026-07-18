#!/usr/bin/env python3
"""Run one model block of the preregistered L29 cross-layer factorial."""

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import deep_merge, git_sha, load_yaml
from mobile_robot_mppi.core.spaces import action_spec_from_config, state_spec_from_config
from mobile_robot_mppi.planning.mppi import MppiConfig
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior
from mobile_robot_mppi.rl.prior import TorchSACPrior
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner
from experiments.rl.evaluate_rl_sampling_prior import _apply_scene_config


CONDITION_SPECS = {
    "traditional_nominal": {"policy": "traditional", "residual": "nominal"},
    "traditional_mlp": {"policy": "traditional", "residual": "mlp"},
    "traditional_icode": {"policy": "traditional", "residual": "icode"},
    "traditional_icode_reliability_gate": {
        "policy": "traditional",
        "residual": "icode",
        "residual_reliability_gate": True,
    },
    "traditional_icode_oracle_domain_gate": {
        "policy": "traditional",
        "residual": "icode",
        "residual_oracle_domain_gate": True,
        "residual_factor": "icode_oracle_domain_gate",
    },
    "traditional_icode_support_gate": {
        "policy": "traditional",
        "residual": "icode",
        "residual_support_gate": True,
    },
    "traditional_icode_dynamic_mask": {
        "policy": "traditional",
        "residual": "icode",
        "residual_component_mask": True,
    },
    "traditional_icode_dynamic_mask_support_gate": {
        "policy": "traditional",
        "residual": "icode",
        "residual_component_mask": True,
        "residual_support_gate": True,
    },
    "lcb_nominal": {"policy": "lcb", "residual": "nominal"},
    "lcb_icode": {"policy": "lcb", "residual": "icode"},
    "gated_lcb_nominal": {"policy": "gated_lcb", "residual": "nominal"},
    "gated_lcb_icode": {"policy": "gated_lcb", "residual": "icode"},
    "progress_gated_lcb_icode": {
        "policy": "progress_gated_lcb",
        "residual": "icode",
        "gate_mode": "progress_complexity",
    },
    "support_gated_lcb_icode": {
        "policy": "support_gated_lcb",
        "residual": "icode",
        "gate_mode": "complexity",
        "correction_support_gate_enabled": True,
    },
    "frozen_bc_icode": {
        "policy": "frozen_bc",
        "residual": "icode",
        "gate_mode": "none",
        "correction_advantage_gate_mode": "base",
    },
    "complexity_bc_icode": {
        "policy": "complexity_bc",
        "residual": "icode",
        "gate_mode": "complexity",
        "correction_advantage_gate_mode": "base",
    },
    "temporal_gated_lcb_nominal": {
        "policy": "temporal_gated_lcb", "residual": "nominal"
    },
    "temporal_gated_lcb_icode": {
        "policy": "temporal_gated_lcb", "residual": "icode"
    },
    "traditional_icode_no_temporal_safety": {
        "policy": "traditional",
        "residual": "icode",
        "temporal_scan_enabled": False,
        "temporal_safety_enabled": False,
    },
    "traditional_icode_temporal_safety": {
        "policy": "traditional",
        "residual": "icode",
        "temporal_scan_enabled": True,
        "temporal_safety_enabled": True,
    },
    "legacy_temporal_gated_lcb_icode": {
        "policy": "temporal_gated_lcb",
        "residual": "icode",
        "gate_mode": "complexity",
        "temporal_closing_source": "legacy_sector_minimum",
        "temporal_scan_enabled": False,
        "temporal_safety_enabled": False,
    },
    "robust_temporal_gated_lcb_icode": {
        "policy": "temporal_gated_lcb",
        "residual": "icode",
        "gate_mode": "complexity",
        "temporal_closing_source": "perception_scan_flow",
        "temporal_scan_enabled": True,
        "temporal_safety_enabled": False,
    },
    "robust_temporal_gated_lcb_icode_safety": {
        "policy": "temporal_gated_lcb",
        "residual": "icode",
        "gate_mode": "complexity",
        "temporal_closing_source": "perception_scan_flow",
        "temporal_scan_enabled": True,
        "temporal_safety_enabled": True,
    },
    "competence_gated_lcb_icode_safety": {
        "policy": "temporal_gated_lcb",
        "residual": "icode",
        "gate_mode": "complexity_confidence",
        "temporal_closing_source": "perception_scan_flow",
        "temporal_scan_enabled": True,
        "temporal_safety_enabled": True,
    },
    "bounded_rl_nominal": {
        "policy": "bounded_rl",
        "residual": "nominal",
        "gate_mode": "fixed",
        "fixed_alpha": 0.25,
        "correction_advantage_gate_mode": "none",
        "temporal_scan_enabled": True,
        "temporal_safety_enabled": True,
    },
    "bounded_rl_icode": {
        "policy": "bounded_rl",
        "residual": "icode",
        "gate_mode": "fixed",
        "fixed_alpha": 0.25,
        "correction_advantage_gate_mode": "none",
        "temporal_scan_enabled": True,
        "temporal_safety_enabled": True,
    },
}

L29_CONDITIONS = (
    "traditional_nominal",
    "traditional_icode",
    "lcb_nominal",
    "lcb_icode",
    "gated_lcb_nominal",
    "gated_lcb_icode",
)

PREVIOUS_PROTECTED_CONFIGS = (
    "configs/rl/sac_mppi_scene_complexity_l25.yaml",
    "configs/rl/sac_mppi_sample_efficiency_l26.yaml",
    "configs/rl/sac_mppi_zero_complexity_fastpath_l27.yaml",
    "configs/rl/sac_mppi_inference_profile_l28.yaml",
)

STEP_FIELDS = (
    "time",
    "x",
    "y",
    "theta",
    "v",
    "omega",
    "goal_distance",
    "collision",
    "clearance",
    "dynamic_obstacle_count",
    "nearest_dynamic_obstacle_center_distance",
    "executed_v",
    "executed_omega",
    "applied_v",
    "applied_omega",
    "safety_override",
    "safety_reason",
    "temporal_scan_valid",
    "temporal_scan_closing_rate_mps",
    "temporal_scan_ttc_s",
    "temporal_scan_risk_alpha",
    "temporal_scan_support_beams",
    "temporal_scan_rejected_jump_fraction",
    "temporal_scan_held",
    "target_phase",
    "target_x",
    "target_y",
    "terminal_heading_gate_active",
    "terminal_bearing_error",
    "terminal_translation_scale",
    "terminal_alignment_active",
    "terminal_alignment_omega",
    "planner_compute_ms",
    "residual_support_confidence_mean",
    "residual_support_confidence_min",
    "residual_support_reduced_fraction",
    "residual_support_disabled_fraction",
    "residual_reliability_enabled",
    "residual_reliability_alpha",
    "residual_reliability_evidence_alpha",
    "residual_reliability_context_alpha",
    "residual_reliability_context_value",
    "residual_reliability_samples",
    "residual_reliability_mean_improvement",
    "residual_reliability_lcb",
    "residual_reliability_last_relative_improvement",
    "residual_reliability_nominal_error",
    "residual_reliability_residual_error",
    "rl_gate_alpha",
    "rl_correction_advantage_gate_alpha",
    "rl_correction_support_gate_enabled",
    "rl_correction_support_confidence",
    "rl_correction_effective_gate_alpha",
    "rl_scene_complexity_score",
    "rl_temporal_closing_gate_alpha",
    "rl_temporal_closing_rate_mps",
    "rl_temporal_closing_held",
    "rl_hazard_activation",
    "rl_competence_confidence",
    "rl_baseline_progress_m",
    "rl_baseline_progress_window_s",
    "rl_baseline_progress_gate_ready",
    "rl_baseline_stagnation_activation",
    "rl_baseline_stagnation_held",
    "rl_learned_inference_skipped",
)


def _resolved_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty L29 table")
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _scene_entries(design):
    entries = []
    for raw in design["scenes"]:
        path = _resolved_path(raw["path"])
        config = load_yaml(path)
        entries.append({
            "name": str(config["scene"]["name"]),
            "role": str(raw["role"]),
            "path": path,
        })
    names = [item["name"] for item in entries]
    if len(names) != len(set(names)):
        raise ValueError("L29 scene names must be unique")
    return entries


def _domain_entries(design):
    entries = []
    for raw in design["physics_domains"]:
        entries.append({
            "name": str(raw["name"]),
            "role": str(raw["role"]),
            "plant_override": copy.deepcopy(dict(raw.get("plant_override", {}))),
            "sensor_override": copy.deepcopy(dict(raw.get("sensor_override", {}))),
        })
    names = [item["name"] for item in entries]
    if len(names) != len(set(names)):
        raise ValueError("L29 physics-domain names must be unique")
    return entries


def _condition_config(
    base,
    scene_path,
    domain,
    condition,
    rl_checkpoint,
    icode_checkpoint,
    episode_seed,
    mlp_checkpoint=None,
):
    if condition not in CONDITION_SPECS:
        raise ValueError("unknown L29 condition %s" % condition)
    spec = CONDITION_SPECS[condition]
    factorial_design = base.get("rl", {}).get("cross_layer_factorial", {})
    support_design = factorial_design.get(
        "support_gate", {}
    )
    component_mask = factorial_design.get(
        "residual_component_mask"
    )
    if spec.get("residual_support_gate", False):
        spec = dict(spec)
        spec["residual_support_soft_z"] = float(support_design.get("soft_z", 3.0))
        spec["residual_support_hard_z"] = float(support_design.get("hard_z", 5.0))
    config = copy.deepcopy(_apply_scene_config(base, scene_path))
    config["plant"] = deep_merge(
        config["plant"], domain.get("plant_override", {})
    )
    if bool(
        base.get("rl", {}).get("cross_layer_factorial", {}).get(
            "planner_known_command_delay", False
        )
    ):
        config["planner"]["command_delay_s"] = float(
            config["plant"].get("actuator", {}).get("command_delay", 0.0)
        )
    config["experiment"]["seed"] = int(episode_seed)
    config["experiment"]["name"] = "l29_%s_%s_seed_%d" % (
        condition,
        domain["name"],
        int(episode_seed),
    )
    config.setdefault("memory", {})["enable"] = False
    config["planner"]["num_samples"] = int(
        base["planner"].get("num_samples", 100)
    )
    sensor_overrides = dict(config.get("rl", {}).get("sensor_overrides", {}))
    if sensor_overrides:
        config.setdefault("sensors", {}).update(sensor_overrides)
    config["sensors"] = deep_merge(
        config.get("sensors", {}), domain.get("sensor_override", {})
    )

    if spec["residual"] in ("icode", "mlp"):
        residual_kind = str(spec["residual"])
        selected_checkpoint = (
            icode_checkpoint if residual_kind == "icode" else mlp_checkpoint
        )
        if selected_checkpoint is None:
            raise ValueError("%s requires a %s checkpoint" % (
                condition, residual_kind
            ))
        config["planner"]["prediction_mode"] = "%s_residual" % residual_kind
        config["planner"]["checkpoint"] = str(Path(selected_checkpoint).resolve())
        config["planner"]["residual_support_gate"] = {
            "enabled": bool(spec.get("residual_support_gate", False)),
            "soft_z": float(spec.get("residual_support_soft_z", 3.0)),
            "hard_z": float(spec.get("residual_support_hard_z", 5.0)),
        }
        reliability_design = dict(factorial_design.get(
            "residual_reliability_gate", {}
        ))
        reliability_design["enabled"] = bool(
            spec.get("residual_reliability_gate", False)
        )
        config["planner"]["residual_reliability_gate"] = reliability_design
        if spec.get("residual_component_mask", False):
            if component_mask is None:
                state_names = state_spec_from_config(
                    config["state_space"]
                ).names
                component_mask = [
                    0.0 if name in ("x", "y", "theta") else 1.0
                    for name in state_names
                ]
                if not component_mask or not any(component_mask):
                    raise ValueError(
                        "%s requires an explicit residual component mask"
                        % condition
                    )
            config["planner"]["residual_component_mask"] = list(component_mask)
        else:
            config["planner"].pop("residual_component_mask", None)
        active_oracle_domains = tuple(
            str(value) for value in factorial_design.get(
                "oracle_residual_active_domains", ()
            )
        )
        if (
            spec.get("residual_oracle_domain_gate", False)
            and active_oracle_domains
            and str(domain["name"]) not in active_oracle_domains
        ):
            config["planner"]["prediction_mode"] = "nominal"
            config["planner"].pop("checkpoint", None)
            config["planner"].pop("residual_support_gate", None)
            config["planner"].pop("residual_component_mask", None)
            config["planner"].pop("residual_reliability_gate", None)
    else:
        config["planner"]["prediction_mode"] = "nominal"
        config["planner"].pop("checkpoint", None)
        config["planner"].pop("residual_support_gate", None)
        config["planner"].pop("residual_component_mask", None)
        config["planner"].pop("residual_reliability_gate", None)

    gate = config.setdefault("rl", {}).setdefault("gate", {})
    temporal_scan = config.setdefault("perception", {}).setdefault(
        "temporal_scan_guard", {}
    )
    shared_design = base.get("rl", {}).get("cross_layer_factorial", {})
    temporal_scan["enabled"] = bool(shared_design.get(
        "shared_temporal_scan_enabled",
        spec.get("temporal_scan_enabled", False),
    ))
    temporal_scan["safety_enabled"] = bool(shared_design.get(
        "shared_temporal_safety_enabled",
        spec.get("temporal_safety_enabled", False),
    ))
    if spec["policy"] == "traditional":
        config["planner"]["sampling_prior"] = "goal_warm_start"
        config["rl"]["enabled"] = False
        config["rl"]["checkpoint"] = None
    else:
        config["planner"]["sampling_prior"] = "rl"
        config["rl"]["enabled"] = True
        config["rl"]["checkpoint"] = str(Path(rl_checkpoint).resolve())
        gate["mode"] = spec.get(
            "gate_mode",
            "none" if spec["policy"] == "lcb" else "complexity",
        )
        if "fixed_alpha" in spec:
            gate["fixed_alpha"] = float(spec["fixed_alpha"])
        gate["temporal_closing_enabled"] = bool(
            spec["policy"] == "temporal_gated_lcb"
        )
        gate["temporal_closing_source"] = str(
            spec.get(
                "temporal_closing_source", "legacy_sector_minimum"
            )
        )
        gate["correction_advantage_gate_mode"] = str(
            spec.get("correction_advantage_gate_mode", "lcb")
        )
        gate["correction_advantage_critic_source"] = "target"
        gate["correction_advantage_threshold"] = 0.0
        gate["correction_advantage_uncertainty_multiplier"] = 2.0
        gate["correction_support_gate_enabled"] = bool(
            spec.get("correction_support_gate_enabled", False)
        )
    config["factorial"] = {
        "condition": condition,
        "policy_factor": spec["policy"],
        "residual_factor": spec.get("residual_factor", spec["residual"]),
        "physics_domain": domain["name"],
        "physics_role": domain["role"],
    }
    return config


def _make_prior(config, checkpoint):
    action_spec = action_spec_from_config(config["action_space"])
    planner = dict(config["planner"])
    planner.setdefault("dt", config["experiment"]["control_dt"])
    planner.setdefault("seed", config["experiment"].get("seed", 0))
    mppi = MppiConfig.from_mapping(planner, action_spec.dimension)
    fallback = GoalWarmStartPrior(
        planner.get("prior_v_gain", 0.8),
        planner.get("prior_yaw_gain", 1.2),
        planner.get("prior_translation_heading_gate_rad"),
        planner.get("prior_translation_heading_gate_terminal_only", False),
    )
    return TorchSACPrior.from_checkpoint(
        checkpoint,
        action_spec,
        mppi.noise_sigma,
        device="cpu",
        gate_config=config["rl"].get("gate", {}),
        fallback_prior=fallback,
        policy_id=str(config["rl"].get("policy_id", "l29")),
    )


def _parse_seeds(text, default):
    values = default if text is None else [
        int(value) for value in text.split(",") if value.strip()
    ]
    values = [int(value) for value in values]
    if not values or len(values) != len(set(values)):
        raise ValueError("L29 episode seeds must be nonempty and unique")
    return values


def _parse_conditions(text, configured):
    values = configured if text is None else [
        value.strip() for value in text.split(",") if value.strip()
    ]
    values = [str(value) for value in values]
    if not values or len(values) != len(set(values)):
        raise ValueError("L29 conditions must be nonempty and unique")
    unknown = sorted(set(values) - set(CONDITION_SPECS))
    if unknown:
        raise ValueError("unknown L29 conditions: %s" % unknown)
    unavailable = sorted(set(values) - set(configured))
    if unavailable:
        raise ValueError(
            "requested L29 conditions are absent from config: %s"
            % unavailable
        )
    return values


def _prior_instance_key(condition):
    """Keep stateful observation histories isolated across conditions."""

    spec = CONDITION_SPECS[str(condition)]
    return None if spec["policy"] == "traditional" else str(condition)


def _protected_previous_seeds(extra_configs=()):
    protected = set()

    def visit(value, parent_key=""):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, str(key))
        elif isinstance(value, (list, tuple)) and "episode_seeds" in parent_key:
            # A preregistered sealed list is intentionally unused until its
            # explicit confirmation run.  Once used, the confirmation config
            # also records it as development_episode_seeds, which then makes
            # it protected for all later experiments.
            if parent_key == "sealed_confirmation_episode_seeds":
                return
            for item in value:
                if isinstance(item, int):
                    protected.add(int(item))

    for value in tuple(PREVIOUS_PROTECTED_CONFIGS) + tuple(extra_configs):
        visit(load_yaml(_resolved_path(value)))
    return protected


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rl-checkpoint", required=True)
    parser.add_argument("--icode-checkpoint", required=True)
    parser.add_argument("--mlp-checkpoint")
    parser.add_argument("--model-block", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-seeds")
    parser.add_argument("--conditions")
    parser.add_argument("--allow-sealed-confirmation", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    base = load_yaml(_resolved_path(args.config))
    design = base["rl"]["cross_layer_factorial"]
    blocks = list(design["model_blocks"])
    if not 0 <= args.model_block < len(blocks):
        raise ValueError("model block index is out of range")
    block = blocks[args.model_block]
    seeds = _parse_seeds(
        args.episode_seeds, design["development_episode_seeds"]
    )
    sealed = {int(value) for value in design["sealed_confirmation_episode_seeds"]}
    if sealed.intersection(seeds) and not args.allow_sealed_confirmation:
        raise ValueError("sealed L29 confirmation seeds require explicit approval")
    previous_protected = _protected_previous_seeds(
        design.get("protected_config_paths", ())
    )
    overlap = previous_protected.intersection(seeds)
    if overlap:
        raise ValueError(
            "L29 episode seeds overlap protected L25--L28 seeds: %s"
            % sorted(overlap)
        )
    configured_conditions = [str(value) for value in design["conditions"]]
    conditions = _parse_conditions(args.conditions, configured_conditions)

    rl_checkpoint = _resolved_path(args.rl_checkpoint)
    configured_condition_checkpoints = dict(
        block.get("condition_checkpoints", {})
    )
    condition_checkpoints = {
        condition: _resolved_path(
            configured_condition_checkpoints.get(
                condition, rl_checkpoint
            )
        )
        for condition in conditions
    }
    for condition, checkpoint in condition_checkpoints.items():
        if not checkpoint.exists():
            raise FileNotFoundError(
                "%s checkpoint does not exist: %s"
                % (condition, checkpoint)
            )
    icode_checkpoint = _resolved_path(args.icode_checkpoint)
    mlp_checkpoint = (
        None if args.mlp_checkpoint is None else _resolved_path(args.mlp_checkpoint)
    )
    if "traditional_mlp" in conditions and mlp_checkpoint is None:
        raise ValueError("traditional_mlp requires --mlp-checkpoint")
    scenes = _scene_entries(design)
    domains = _domain_entries(design)
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    schedule = [
        {
            "scene": scene,
            "domain": domain,
            "episode_seed": seed,
            "condition": condition,
        }
        for scene in scenes
        for domain in domains
        for seed in seeds
        for condition in conditions
    ]
    schedule_seed = int(design["schedule_seed"]) + int(args.model_block)
    random.Random(schedule_seed).shuffle(schedule)
    schedule_rows = [{
        "run_order": index,
        "model_block": int(args.model_block),
        "rl_training_seed": int(block["rl_seed"]),
        "icode_training_seed": int(block["icode_seed"]),
        "mlp_training_seed": int(block.get("mlp_seed", -1)),
        "scene": item["scene"]["name"],
        "scene_role": item["scene"]["role"],
        "physics_domain": item["domain"]["name"],
        "physics_role": item["domain"]["role"],
        "episode_seed": int(item["episode_seed"]),
        "condition": item["condition"],
    } for index, item in enumerate(schedule)]
    _write_csv(output / "condition_schedule.csv", schedule_rows)

    representative_scene = scenes[0]["path"]
    representative_domain = domains[0]
    priors = {}
    for condition in conditions:
        key = _prior_instance_key(condition)
        if key is None:
            continue
        config = _condition_config(
            base,
            representative_scene,
            representative_domain,
            condition,
            condition_checkpoints[condition],
            icode_checkpoint,
            seeds[0],
            mlp_checkpoint,
        )
        priors[key] = _make_prior(
            config, condition_checkpoints[condition]
        )

    episodes = []
    steps = []
    for index, item in enumerate(schedule):
        scene = item["scene"]
        domain = item["domain"]
        condition = item["condition"]
        spec = CONDITION_SPECS[condition]
        seed = int(item["episode_seed"])
        config = _condition_config(
            base,
            scene["path"],
            domain,
            condition,
            condition_checkpoints[condition],
            icode_checkpoint,
            seed,
            mlp_checkpoint,
        )
        if args.max_steps is not None:
            if args.max_steps <= 0:
                raise ValueError("--max-steps must be positive")
            config["experiment"]["max_steps"] = int(args.max_steps)
        run_dir = (
            output / "runs" / scene["name"] / domain["name"]
            / condition / ("seed_%d" % seed)
        )
        metrics_path = run_dir / "metrics.json"
        trajectory_path = run_dir / "trajectory.csv"
        if metrics_path.exists() and trajectory_path.exists():
            episode = json.loads(metrics_path.read_text(encoding="utf-8"))
            episode.pop("metadata", None)
            episode.pop("provenance", None)
        else:
            result = ExperimentRunner(
                config,
                ROOT,
                output_dir=run_dir,
                headless=True,
                rl_policy=priors.get(_prior_instance_key(condition)),
            ).run()
            episode = dict(result.summary)
        episode.update(schedule_rows[index])
        episode["rl_checkpoint"] = (
            "none"
            if spec["policy"] == "traditional"
            else str(condition_checkpoints[condition])
        )
        episode["icode_checkpoint"] = (
            str(icode_checkpoint) if spec["residual"] == "icode" else "none"
        )
        episode["mlp_checkpoint"] = (
            str(mlp_checkpoint) if spec["residual"] == "mlp" else "none"
        )
        episode["residual_checkpoint"] = (
            str(icode_checkpoint) if spec["residual"] == "icode" else
            str(mlp_checkpoint) if spec["residual"] == "mlp" else "none"
        )
        episodes.append(episode)

        with trajectory_path.open("r", newline="", encoding="utf-8") as handle:
            for step_index, raw in enumerate(csv.DictReader(handle)):
                row = dict(schedule_rows[index])
                row["step"] = step_index
                row.update({name: raw[name] for name in STEP_FIELDS})
                steps.append(row)

    _write_csv(output / "episodes.csv", episodes)
    _write_csv(output / "factorial_steps.csv", steps)
    metadata = {
        "model_block": int(args.model_block),
        "rl_training_seed": int(block["rl_seed"]),
        "icode_training_seed": int(block["icode_seed"]),
        "mlp_training_seed": int(block.get("mlp_seed", -1)),
        "episode_seeds": seeds,
        "sealed_confirmation_seeds_used": sorted(sealed.intersection(seeds)),
        "previous_protected_seeds_used": sorted(
            previous_protected.intersection(seeds)
        ),
        "schedule_seed": schedule_seed,
        "conditions": conditions,
        "scenes": [
            {"name": item["name"], "role": item["role"], "path": str(item["path"])}
            for item in scenes
        ],
        "physics_domains": domains,
        "rl_checkpoint": str(rl_checkpoint),
        "rl_checkpoint_sha256": _sha256(rl_checkpoint),
        "condition_checkpoints": {
            condition: {
                "path": str(condition_checkpoints[condition]),
                "sha256": _sha256(condition_checkpoints[condition]),
            }
            for condition in conditions
            if CONDITION_SPECS[condition]["policy"] != "traditional"
        },
        "icode_checkpoint": str(icode_checkpoint),
        "icode_checkpoint_sha256": _sha256(icode_checkpoint),
        "mlp_checkpoint": None if mlp_checkpoint is None else str(mlp_checkpoint),
        "mlp_checkpoint_sha256": (
            None if mlp_checkpoint is None else _sha256(mlp_checkpoint)
        ),
        "run_git_sha": git_sha(ROOT),
        "episodes": len(episodes),
        "steps": len(steps),
        "interpretation_guard": (
            "episode seeds are repeated measures within model blocks; control "
            "steps are technical measurements, not independent replicates"
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
