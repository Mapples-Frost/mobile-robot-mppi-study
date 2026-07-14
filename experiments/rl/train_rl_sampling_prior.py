#!/usr/bin/env python3
"""Train SAC to propose MPPI sampling priors across configured MuJoCo scenes."""

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import deep_merge, load_yaml
from mobile_robot_mppi.rl.trainer import SACTrainer


def _resolve_config(path):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def _apply_training_seed(config, seed):
    """Apply one explicit seed to training, resets, and recorded config."""

    if seed is None:
        return config
    seed = int(seed)
    if not 0 <= seed <= 2 ** 32 - 1:
        raise ValueError("--seed must be in [0, 2**32 - 1]")
    config.setdefault("experiment", {})["seed"] = seed
    config.setdefault("rl", {}).setdefault("training", {})["seed"] = seed
    return config


def _scene_configs(base, names):
    if not names:
        return [copy.deepcopy(base)]
    result = []
    for name in names:
        scene = load_yaml(_resolve_config(name))
        # The selected scene owns plant/perception/geometry.  The main RL file
        # owns the controlled planner setup and every learning hyperparameter.
        scene = deep_merge(scene, {
            "planner": base["planner"],
            "rl": base["rl"],
        })
        result.append(scene)
    return result


def _physics_domains(configs, domain_file, roles):
    if not domain_file:
        return configs
    specification = load_yaml(_resolve_config(domain_file))
    selected_roles = set(str(role) for role in (roles or ("seen",)))
    domains = [
        domain for domain in specification.get("physics_domains", {}).get("domains", ())
        if str(domain.get("role", "seen")) in selected_roles
    ]
    if not domains:
        raise ValueError("no physics domains matched roles: %s" % sorted(selected_roles))
    expanded = []
    for config in configs:
        for domain in domains:
            item = deep_merge(
                config,
                {"plant": domain.get("plant_override", {})},
            )
            domain_name = str(domain.get("name", "domain"))
            scene_name = str(item.get("scene", {}).get("name", "scene"))
            item.setdefault("scene", {})["name"] = "%s__%s" % (scene_name, domain_name)
            item.setdefault("experiment", {})["physics_domain"] = domain_name
            item["experiment"]["physics_domain_role"] = str(domain.get("role", "seen"))
            expanded.append(item)
    return expanded


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs/rl/sac_mppi_prior.yaml")
    )
    parser.add_argument("--output-dir")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume")
    checkpoint_group.add_argument(
        "--initialize-actor-from",
        help=(
            "load only actor weights and fitted observation normalizer; "
            "critics, optimizers, replay and counters remain fresh"
        ),
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        help="override the training/reset/validation seed in the config",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    _apply_training_seed(config, args.seed)
    training = config["rl"].setdefault("training", {})
    if args.steps is not None:
        training["total_steps"] = int(args.steps)
    if args.device is not None:
        training["device"] = args.device
    if args.smoke:
        training.update({
            "total_steps": min(int(training.get("total_steps", 60)), 60),
            "warmup_steps": 8,
            "update_after": 8,
            "batch_size": 8,
            "replay_capacity": 256,
            "updates_per_step": 1,
            "evaluation_interval": 30,
            "evaluation_episodes": 1,
            "checkpoint_interval": 30,
            "save_replay_buffer": True,
            "scene_configs": training.get("scene_configs", [])[:1],
            "validation_scene_configs": training.get(
                "validation_scene_configs", []
            )[:1],
            "physics_domain_config": None,
            "curriculum": {"enabled": False},
            "initial_state_curriculum": {"enabled": False},
        })
        config["experiment"]["max_steps"] = min(
            int(config["experiment"]["max_steps"]), 15
        )
        config["planner"]["horizon"] = min(int(config["planner"]["horizon"]), 8)
        config["planner"]["num_samples"] = min(
            int(config["planner"]["num_samples"]), 32
        )
    train_configs = _scene_configs(config, training.get("scene_configs", []))
    validation_configs = _scene_configs(
        config, training.get("validation_scene_configs", [])
    )
    train_configs = _physics_domains(
        train_configs,
        training.get("physics_domain_config"),
        training.get("training_domain_roles", ("seen",)),
    )
    validation_configs = _physics_domains(
        validation_configs,
        training.get("physics_domain_config"),
        training.get("validation_domain_roles", ("seen", "unseen")),
    )
    for item in train_configs + validation_configs:
        if args.smoke:
            item["experiment"]["max_steps"] = config["experiment"]["max_steps"]
            item["planner"]["horizon"] = config["planner"]["horizon"]
            item["planner"]["num_samples"] = config["planner"]["num_samples"]
    output = Path(
        args.output_dir
        or config["experiment"].get(
            "output_dir", ROOT / "results/research_platform/rl_training"
        )
    ).resolve()
    trainer = SACTrainer(
        train_configs,
        validation_configs,
        ROOT,
        output,
        config,
    )
    if args.resume:
        trainer.resume(_resolve_config(args.resume))
    elif args.initialize_actor_from:
        trainer.initialize_actor_from(_resolve_config(args.initialize_actor_from))
    result = trainer.run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
