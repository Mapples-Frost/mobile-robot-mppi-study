"""Run a two-arm fresh-development probe of exact MPPI fallback.

This is a bounded engineering probe, not a confirmatory experiment.  It uses
one unsealed development seed, runs RL/HSS-off followed by the active proposal-
advantage veto with exact standard-MPPI fallback, and stops after a collision.
Historical Stage 4/5 cells and artifacts are never resumed or modified.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.run_rl_hss_stage4 import (
    _complete,
    _json,
    _mapping,
    _resolve,
    _write_json,
    configure_job as configure_stage4_job,
)
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


STAGE5_PROTOCOL = (
    ROOT
    / "configs"
    / "research"
    / "dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development.yaml"
)
DEFAULT_EPISODE_SEED = 730100070
ARMS = ("rl_hss_off", "exact_fallback_veto")


def _default_output(episode_seed):
    return (
        ROOT
        / "research_artifacts"
        / (
            "dynamic_uncertainty_rl_hss_exact_fallback_probe_seed%d"
            % int(episode_seed)
        )
    )


def configure_probe_job(base, stage3, stage4, arm, episode_seed):
    config = configure_stage4_job(
        base,
        {
            "condition": "nominal",
            "episode_seed": int(episode_seed),
            "model_block": -1,
            "rl_hss_enabled": arm != "rl_hss_off",
        },
        stage4,
        stage3,
    )
    config = deepcopy(config)
    if arm == "exact_fallback_veto":
        paper = config["planner"]["paper_rl_driven"]
        paper["proposal_advantage_gate"] = {
            "enabled": True,
            "mode": "episode_latched_veto",
            "relative_disadvantage_margin": 0.0,
            "consecutive_disadvantages": 3,
        }
        paper["standard_fallback_on_advantage_veto"] = True
    config["experiment"]["name"] += "__exact_fallback_probe__%s" % arm
    config["experiment"]["exact_fallback_probe"] = True
    config["experiment"]["sealed_seeds_opened"] = False
    return config


def _summary(metrics):
    return {
        "success": bool(metrics.get("success", False)),
        "collision": bool(metrics.get("collision", False)),
        "steps": int(metrics.get("steps", 0)),
        "final_goal_distance_m": float(
            metrics.get("final_goal_distance", float("inf"))
        ),
        "planner_p95_ms": float(
            metrics.get("planner_compute_ms_p95", float("inf"))
        ),
        "proposal_advantage_latch_step": int(
            metrics.get("reliability_proposal_advantage_latch_step", -1)
        ),
        "standard_fallback_fraction": float(
            metrics.get("paper_standard_fallback_active_fraction", 0.0)
        ),
        "paper_total_rollouts_mean": float(
            metrics.get("paper_total_rollouts_mean", 0.0)
        ),
    }


def run(*, execute=False, episode_seed=DEFAULT_EPISODE_SEED, output_dir=None):
    episode_seed = int(episode_seed)
    if not 730100001 <= episode_seed <= 730100300:
        raise ValueError("probe seed must remain in the development split")
    output_dir = Path(
        output_dir or _default_output(episode_seed)
    ).resolve()
    if ROOT not in output_dir.parents:
        raise ValueError("probe output must remain inside the repository")
    schedule = [
        {"run_order": index, "arm": arm, "episode_seed": episode_seed}
        for index, arm in enumerate(ARMS)
    ]
    if not execute:
        return {
            "status": "preflight_only",
            "sealed_seeds_opened": False,
            "schedule": schedule,
        }

    protocol = _mapping(STAGE5_PROTOCOL)
    base = load_yaml(_resolve(protocol["base_config"]))
    stage3 = _mapping(_resolve(protocol["stage3_protocol"]))
    stage4 = _mapping(_resolve(protocol["stage4_protocol"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "schedule.json", {
        "schema_version": 1,
        "scope": "bounded_engineering_probe",
        "sealed_seeds_opened": False,
        "jobs": schedule,
    })

    results = {}
    stopped = None
    for job in schedule:
        arm = job["arm"]
        run_dir = output_dir / "runs" / arm
        if not _complete(run_dir):
            config = configure_probe_job(
                base, stage3, stage4, arm, episode_seed
            )
            print("[%d/%d] %s" % (
                job["run_order"] + 1, len(schedule), arm
            ), flush=True)
            ExperimentRunner(
                config, ROOT, output_dir=run_dir, headless=True
            ).run()
        metrics = _json(run_dir / "metrics.json")
        results[arm] = _summary(metrics)
        if bool(metrics.get("collision", False)):
            stopped = {"reason": "first_collision", "arm": arm}
            break

    payload = {
        "schema_version": 1,
        "status": "stopped_on_collision" if stopped else (
            "complete" if len(results) == len(schedule) else "incomplete"
        ),
        "scope": "bounded_engineering_probe",
        "episode_seed": episode_seed,
        "sealed_seeds_opened": False,
        "results": results,
    }
    if stopped:
        payload["early_stop"] = stopped
    _write_json(output_dir / "result.json", payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--episode-seed", type=int, default=DEFAULT_EPISODE_SEED
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    result = run(
        execute=bool(args.execute),
        episode_seed=args.episode_seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
