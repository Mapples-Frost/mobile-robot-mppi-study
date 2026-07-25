"""Run a post-hoc MuJoCo probe of mass-aware stopping feasibility.

This bounded mechanism probe replays one development obstacle seed under two
RL/HSS shadow treatments.  The treatments are identical except that the
second opts into accumulated-probability ordering when motion and stopping
have the same saturated maximum collision probability.  It is explicitly
post-hoc and never writes into the historical Stage 4 or Stage 5 artifacts.
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
    _sha256,
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
DEFAULT_EPISODE_SEED = 730100064
ARMS = ("legacy_stop_order", "mass_aware_stop_order")


def _default_output(episode_seed):
    return (
        ROOT
        / "research_artifacts"
        / (
            "dynamic_uncertainty_stopping_feasibility_posthoc_seed%d"
            % int(episode_seed)
        )
    )


def configure_probe_job(base, stage3, stage4, stage5, arm, episode_seed):
    if arm not in ARMS:
        raise ValueError("unknown stopping-feasibility arm: %s" % arm)
    config = configure_stage4_job(
        base,
        {
            "condition": "nominal",
            "episode_seed": int(episode_seed),
            "model_block": -1,
            "rl_hss_enabled": True,
        },
        stage4,
        stage3,
    )
    config = deepcopy(config)
    config["planner"]["paper_rl_driven"]["proposal_advantage_gate"] = (
        deepcopy(stage5["proposal_advantage_gate"]["shadow"])
    )
    mass_aware = arm == "mass_aware_stop_order"
    config["planner"][
        "probabilistic_obstacle_stopping_feasibility_enabled"
    ] = mass_aware
    config["perception"]["scan_guard"][
        "dynamic_escape_probability_mass_enabled"
    ] = mass_aware
    config["experiment"]["name"] += (
        "__stopping_feasibility_posthoc__%s" % arm
    )
    config["experiment"]["stopping_feasibility_posthoc"] = True
    config["experiment"]["stopping_feasibility_arm"] = arm
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
        "minimum_clearance_m": (
            None
            if metrics.get("minimum_clearance") is None
            else float(metrics["minimum_clearance"])
        ),
        "planner_p95_ms": float(
            metrics.get("planner_compute_ms_p95", float("inf"))
        ),
        "active_fallback_steps": int(
            metrics.get("probabilistic_obstacle_active_fallback_steps", 0)
        ),
        "mass_fallback_steps": int(
            metrics.get("probabilistic_obstacle_mass_fallback_steps", 0)
        ),
        "mass_escape_steps": int(
            metrics.get(
                "dynamic_escape_probability_mass_fallback_steps", 0
            )
        ),
        "dynamic_escape_steps": int(
            metrics.get("dynamic_escape_allowed_steps", 0)
        ),
        "stopping_feasibility_enabled_fraction": float(
            metrics.get(
                "probabilistic_obstacle_stopping_feasibility_enabled_fraction",
                0.0,
            )
        ),
    }


def run(*, execute=False, episode_seed=DEFAULT_EPISODE_SEED, output_dir=None):
    episode_seed = int(episode_seed)
    if not 730100001 <= episode_seed <= 730100300:
        raise ValueError("probe seed must remain in the development split")
    output_dir = Path(output_dir or _default_output(episode_seed)).resolve()
    if ROOT not in output_dir.parents:
        raise ValueError("probe output must remain inside the repository")
    schedule = [
        {"run_order": index, "arm": arm, "episode_seed": episode_seed}
        for index, arm in enumerate(ARMS)
    ]
    if not execute:
        return {
            "status": "preflight_only",
            "scope": "posthoc_mechanism_probe",
            "sealed_seeds_opened": False,
            "schedule": schedule,
        }

    stage5 = _mapping(STAGE5_PROTOCOL)
    base = load_yaml(_resolve(stage5["base_config"]))
    stage3 = _mapping(_resolve(stage5["stage3_protocol"]))
    stage4 = _mapping(_resolve(stage5["stage4_protocol"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "schedule.json", {
        "schema_version": 1,
        "scope": "posthoc_mechanism_probe",
        "posthoc_after_observed_collision": True,
        "sealed_seeds_opened": False,
        "source_sha256": {
            "mppi.py": _sha256(
                ROOT / "src" / "mobile_robot_mppi" / "planning" / "mppi.py"
            ),
            "arbiter.py": _sha256(
                ROOT / "src" / "mobile_robot_mppi" / "safety" / "arbiter.py"
            ),
            "metrics.py": _sha256(
                ROOT
                / "src"
                / "mobile_robot_mppi"
                / "evaluation"
                / "metrics.py"
            ),
        },
        "jobs": schedule,
    })

    results = {}
    for job in schedule:
        arm = job["arm"]
        run_dir = output_dir / "runs" / arm
        if not _complete(run_dir):
            config = configure_probe_job(
                base, stage3, stage4, stage5, arm, episode_seed
            )
            print("[%d/%d] %s" % (
                job["run_order"] + 1, len(schedule), arm
            ), flush=True)
            ExperimentRunner(
                config, ROOT, output_dir=run_dir, headless=True
            ).run()
        results[arm] = _summary(_json(run_dir / "metrics.json"))

    legacy = results["legacy_stop_order"]
    mass_aware = results["mass_aware_stop_order"]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "scope": "posthoc_mechanism_probe",
        "posthoc_after_observed_collision": True,
        "episode_seed": episode_seed,
        "sealed_seeds_opened": False,
        "only_treatment_difference": [
            "planner.probabilistic_obstacle_stopping_feasibility_enabled",
            "perception.scan_guard.dynamic_escape_probability_mass_enabled",
        ],
        "results": results,
        "paired_deltas_mass_aware_minus_legacy": {
            "collision": int(mass_aware["collision"])
            - int(legacy["collision"]),
            "success": int(mass_aware["success"])
            - int(legacy["success"]),
            "steps": mass_aware["steps"] - legacy["steps"],
            "final_goal_distance_m": (
                mass_aware["final_goal_distance_m"]
                - legacy["final_goal_distance_m"]
            ),
            "planner_p95_ms": (
                mass_aware["planner_p95_ms"] - legacy["planner_p95_ms"]
            ),
        },
    }
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
