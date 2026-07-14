"""Versioned, provenance-rich SAC checkpoint I/O."""

from datetime import datetime, timezone
from pathlib import Path

import torch

from mobile_robot_mppi.core.config import git_sha


CHECKPOINT_FORMAT = "mobile_robot_mppi.sac_prior.v1"


def save_sac_checkpoint(
    path,
    agent,
    normalizer,
    encoder_config,
    parameterization_config,
    action_spec,
    resolved_config,
    project_root,
    training_state,
    replay_buffer=None,
    include_replay=True,
):
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(project_root),
        "agent": agent.state_dict(),
        "normalizer": normalizer.state_dict(),
        "encoder_config": encoder_config.to_dict(),
        "parameterization_config": parameterization_config.to_dict(),
        "action_spec": {
            "names": tuple(action_spec.names),
            "lower": action_spec.lower.copy(),
            "upper": action_spec.upper.copy(),
        },
        "resolved_config": dict(resolved_config),
        "training_state": dict(training_state),
        "replay_buffer": (
            None
            if replay_buffer is None
            else replay_buffer.state_dict(include_data=bool(include_replay))
        ),
    }
    torch.save(payload, str(destination))
    return destination


def load_sac_checkpoint(path, map_location="cpu"):
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError("RL checkpoint does not exist: %s" % source)
    try:
        payload = torch.load(str(source), map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(str(source), map_location=map_location)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported or corrupt RL checkpoint format")
    required = (
        "agent",
        "normalizer",
        "encoder_config",
        "parameterization_config",
        "action_spec",
        "training_state",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError("RL checkpoint is missing: %s" % ", ".join(missing))
    return payload
