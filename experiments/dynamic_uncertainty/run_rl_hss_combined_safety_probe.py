"""Run a bounded MuJoCo probe of the combined RL/HSS safety fixes.

The combined treatment enables proposal-advantage veto with exact standard
MPPI fallback and mass-aware stopping feasibility.  Historical Stage 4/5
cells and artifacts are never resumed or modified, and sealed seeds remain
closed.
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

from experiments.dynamic_uncertainty.run_rl_hss_exact_fallback_probe import (
    STAGE5_PROTOCOL,
    configure_probe_job as configure_exact_fallback_job,
)
from experiments.dynamic_uncertainty.run_rl_hss_stage4 import (
    _complete,
    _json,
    _mapping,
    _resolve,
    _sha256,
    _write_json,
)
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_EPISODE_SEED = 730100064
HISTORICAL_STAGE5_SEEDS = frozenset((730100064, 730100066, 730100068))


def _default_output(episode_seed):
    return (
        ROOT
        / "research_artifacts"
        / (
            "dynamic_uncertainty_rl_hss_combined_safety_probe_seed%d"
            % int(episode_seed)
        )
    )


def configure_combined_job(base, stage3, stage4, arm, episode_seed):
    if arm == "rl_hss_off":
        return configure_exact_fallback_job(
            base, stage3, stage4, arm, episode_seed
        )
    if arm != "combined_veto":
        raise ValueError("unknown combined-safety arm: %s" % arm)
    config = configure_exact_fallback_job(
        base, stage3, stage4, "exact_fallback_veto", episode_seed
    )
    config = deepcopy(config)
    config["planner"][
        "probabilistic_obstacle_stopping_feasibility_enabled"
    ] = True
    emergency_prefix_steps = int(config["planner"].get(
        "probabilistic_obstacle_emergency_candidate_prefix_steps", 3
    ))
    config["planner"].update({
        "probabilistic_obstacle_emergency_candidates_enabled": True,
        "probabilistic_obstacle_emergency_candidate_prefix_steps": (
            emergency_prefix_steps
        ),
    })
    config["perception"]["scan_guard"][
        "dynamic_escape_probability_mass_enabled"
    ] = True
    config["perception"]["scan_guard"].update({
        "dynamic_escape_hold_enabled": True,
        "dynamic_escape_hold_steps": 3,
        "dynamic_escape_hold_min_probability": 0.20,
    })
    config["experiment"]["name"] += "__combined_safety"
    config["experiment"]["combined_safety_probe"] = True
    config["experiment"]["combined_safety_arm"] = arm
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
        "proposal_advantage_latch_step": int(
            metrics.get("reliability_proposal_advantage_latch_step", -1)
        ),
        "standard_fallback_fraction": float(
            metrics.get("paper_standard_fallback_active_fraction", 0.0)
        ),
        "paper_total_rollouts_mean": float(
            metrics.get("paper_total_rollouts_mean", 0.0)
        ),
        "mass_fallback_steps": int(
            metrics.get("probabilistic_obstacle_mass_fallback_steps", 0)
        ),
        "mass_escape_steps": int(
            metrics.get(
                "dynamic_escape_probability_mass_fallback_steps", 0
            )
        ),
        "dynamic_escape_held_steps": int(
            metrics.get("dynamic_escape_held_steps", 0)
        ),
        "emergency_candidate_selected_steps": int(
            metrics.get(
                "probabilistic_obstacle_emergency_candidate_selected_steps",
                0,
            )
        ),
        "temporal_emergency_triggered_steps": int(
            metrics.get(
                "probabilistic_obstacle_temporal_emergency_triggered_steps",
                0,
            )
        ),
        "temporal_emergency_vetted_steps": int(
            metrics.get(
                "probabilistic_obstacle_temporal_emergency_vetted_steps",
                0,
            )
        ),
        "stopping_feasibility_enabled_fraction": float(
            metrics.get(
                "probabilistic_obstacle_stopping_feasibility_enabled_fraction",
                0.0,
            )
        ),
    }


def run(
    *,
    execute=False,
    episode_seed=DEFAULT_EPISODE_SEED,
    paired=False,
    output_dir=None,
):
    episode_seed = int(episode_seed)
    if not 730100001 <= episode_seed <= 730100300:
        raise ValueError("probe seed must remain in the development split")
    if paired and episode_seed in HISTORICAL_STAGE5_SEEDS:
        raise ValueError(
            "paired mode cannot fill a stopped historical Stage 5 seed"
        )
    arms = ("rl_hss_off", "combined_veto") if paired else (
        "combined_veto",
    )
    output_dir = Path(output_dir or _default_output(episode_seed)).resolve()
    if ROOT not in output_dir.parents:
        raise ValueError("probe output must remain inside the repository")
    schedule = [
        {"run_order": index, "arm": arm, "episode_seed": episode_seed}
        for index, arm in enumerate(arms)
    ]
    if not execute:
        return {
            "status": "preflight_only",
            "scope": "bounded_combined_safety_probe",
            "paired": bool(paired),
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
        "scope": "bounded_combined_safety_probe",
        "paired": bool(paired),
        "historical_stage5_cell_resumed": False,
        "sealed_seeds_opened": False,
        "source_sha256": {
            "mppi.py": _sha256(
                ROOT / "src" / "mobile_robot_mppi" / "planning" / "mppi.py"
            ),
            "rl_driven_mppi.py": _sha256(
                ROOT
                / "src"
                / "mobile_robot_mppi"
                / "planning"
                / "rl_driven_mppi.py"
            ),
            "reliability.py": _sha256(
                ROOT / "src" / "mobile_robot_mppi" / "rl" / "reliability.py"
            ),
            "arbiter.py": _sha256(
                ROOT / "src" / "mobile_robot_mppi" / "safety" / "arbiter.py"
            ),
        },
        "jobs": schedule,
    })

    results = {}
    for job in schedule:
        arm = job["arm"]
        run_dir = output_dir / "runs" / arm
        if not _complete(run_dir):
            config = configure_combined_job(
                base, stage3, stage4, arm, episode_seed
            )
            print("[%d/%d] %s" % (
                job["run_order"] + 1, len(schedule), arm
            ), flush=True)
            ExperimentRunner(
                config, ROOT, output_dir=run_dir, headless=True
            ).run()
        results[arm] = _summary(_json(run_dir / "metrics.json"))

    payload = {
        "schema_version": 1,
        "status": "complete",
        "scope": "bounded_combined_safety_probe",
        "episode_seed": episode_seed,
        "paired": bool(paired),
        "historical_stage5_cell_resumed": False,
        "sealed_seeds_opened": False,
        "results": results,
    }
    if paired:
        off = results["rl_hss_off"]
        combined = results["combined_veto"]
        payload["paired_deltas_combined_minus_off"] = {
            "collision": int(combined["collision"]) - int(off["collision"]),
            "success": int(combined["success"]) - int(off["success"]),
            "steps": combined["steps"] - off["steps"],
            "final_goal_distance_m": (
                combined["final_goal_distance_m"]
                - off["final_goal_distance_m"]
            ),
            "planner_p95_ms": (
                combined["planner_p95_ms"] - off["planner_p95_ms"]
            ),
        }
    _write_json(output_dir / "result.json", payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument(
        "--episode-seed", type=int, default=DEFAULT_EPISODE_SEED
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    result = run(
        execute=bool(args.execute),
        episode_seed=args.episode_seed,
        paired=bool(args.paired),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
