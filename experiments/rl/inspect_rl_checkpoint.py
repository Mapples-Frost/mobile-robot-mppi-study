#!/usr/bin/env python3
"""Print non-tensor metadata from an RL sampling-prior checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Object of type %s is not JSON serializable" % type(value).__name__)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args(argv)
    payload = load_sac_checkpoint(args.checkpoint, map_location="cpu")
    agent = payload["agent"]
    summary = {
        "format": payload["format"],
        "created_utc": payload.get("created_utc"),
        "git_sha": payload.get("git_sha"),
        "observation_dim": agent["observation_dim"],
        "policy_action_dim": agent["action_dim"],
        "sac_config": agent["config"],
        "encoder_config": payload["encoder_config"],
        "parameterization_config": payload["parameterization_config"],
        "action_spec": {
            "names": list(payload["action_spec"]["names"]),
            "lower": payload["action_spec"]["lower"].tolist(),
            "upper": payload["action_spec"]["upper"].tolist(),
        },
        "normalizer_samples": payload["normalizer"]["count"],
        "training_state": {
            key: value for key, value in payload["training_state"].items()
            if not key.endswith("rng_state")
        },
        "replay_saved": bool(
            payload.get("replay_buffer") is not None
            and "observations" in payload["replay_buffer"]
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
