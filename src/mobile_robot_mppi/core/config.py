"""Validated, reproducible YAML configuration loading without framework lock-in."""

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


class ConfigError(ValueError):
    pass


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError("top-level YAML value must be a mapping")
    merged = {}
    includes = data.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    for include in includes:
        include_path = (config_path.parent / include).resolve()
        merged = deep_merge(merged, load_yaml(include_path))
    data = deep_merge(merged, data)
    data["_config_path"] = str(config_path)
    validate_experiment_config(data)
    return data


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    for section in ("experiment", "task", "state_space", "action_space", "plant", "planner"):
        if section not in config:
            raise ConfigError("missing required config section: %s" % section)
    dt = float(config["experiment"].get("control_dt", 0.0))
    if dt <= 0.0:
        raise ConfigError("experiment.control_dt must be positive")
    if int(config["experiment"].get("max_steps", 0)) <= 0:
        raise ConfigError("experiment.max_steps must be positive")
    if int(config["planner"].get("horizon", 0)) <= 0:
        raise ConfigError("planner.horizon must be positive")
    if int(config["planner"].get("num_samples", 0)) <= 0:
        raise ConfigError("planner.num_samples must be positive")
    prior = str(config["planner"].get("sampling_prior", "goal_warm_start"))
    if prior == "rl" and not bool(config.get("rl", {}).get("enabled", False)):
        raise ConfigError("planner.sampling_prior=rl requires rl.enabled=true")


def canonical_json(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def git_sha(root=None) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
        return output.decode("ascii").strip()
    except Exception:
        return "unknown"
