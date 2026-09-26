"""Independently audit and summarize the registered min-Q comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from min_q_eval_suite import JOBS, model_dir
from min_q_protocol import OUT, COUNTS, BASELINE_H, digest, protocol
from min_q_train_suite import verify_freeze
from paper_grid_audit_report import physics
from runtime import ART, ROOT
from run import write


def read(path):
    return json.loads(path.read_text())


def close(x, y, label):
    assert math.isfinite(x) and math.isfinite(y), label
    assert math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-6), (label, x, y)


def audit_one(split, job):
    task, method, seed = job
    source = model_dir(*job)
    spec, trained = read(source / "manifest.json"), read(source / "completed.json")
    author = ART / "results/paper_defaults" / ("%s_rl_s%d" % (task, seed))
    author_spec = read(author / "manifest.json")
    assert spec["initial_hash"] == author_spec["initial_hash"], (job, "initial weights")
    assert spec["config_sha256"] == author_spec["config_sha256"], (job, "task configuration")
    assert spec["adaptations"] == author_spec["adaptations"], (job, "training settings")
    assert trained["steps"] == 15000 and trained["updates"] == 14745
    if method == "min_q":
        assert spec["method_extension"]["name"] == "actor_min_q"
        assert trained["reproduction_level"].startswith("method extension")
    else:
        assert "method_extension" not in spec
    folder = OUT / "evaluations" / split / task / (method + "_s%d" % seed)
    completed, summary = read(folder / "completed.json"), read(folder / "summary.json")
    bank_path = OUT / (task + "_" + split + "_bank.json")
    cases = read(bank_path)["cases"]
    assert completed["bank"] == str(bank_path) and completed["model_dir"] == str(source)
    assert completed["frozen"] and completed["terminal_value"]
    assert completed["model_hash"] == trained["final_hash"]
    if source.parent == OUT:
        assert spec["test_bank_sha256"] == digest(OUT / (task + "_validation_bank.json"))
    assert len(cases) == len(summary["episodes"]) == COUNTS[split]
    assert summary["terminal_value"]
    total_steps, failures, input_excess_steps, max_input_excess = 0, 0, 0, 0.
    unlabeled_terminal_violations = 0
    for j, (episode, case) in enumerate(zip(summary["episodes"], cases)):
        trace = read(folder / ("trace_%02d.json" % j))
        assert episode["episode"] == j and len(trace) == episode["steps"] and trace
        assert len(trace) <= (150 if task == "vehicle" else 100)
        assert episode["solver_failure_steps"] == sum(not step["solver_success"] for step in trace)
        sums = {"performance": 0., "compute": 0., "constraint": 0.}
        for t, step in enumerate(trace):
            assert 1 <= step["horizon"] <= 50
            if method == "fixed":
                assert step["horizon"] == BASELINE_H[task]
            physical, violated, input_excess = physics(task, step, case, t)
            close(physical, step["performance"], (job, split, j, t, "physical"))
            close(step["compute"], step["horizon"] * (.001 if task == "vehicle" else .003),
                  (job, split, j, t, "H cost"))
            remaining = (150 if task == "vehicle" else 100) - t - 1
            close(step["constraint"], (2 if task == "vehicle" else 10) * remaining if violated else 0.,
                  (job, split, j, t, "constraint"))
            if violated:
                assert t == len(trace) - 1
                if episode["termination"] != "constraint":
                    assert task == "pendulum" and remaining == 0 and episode["termination"] == "steps"
                    unlabeled_terminal_violations += 1
            input_excess_steps += int(input_excess > 1e-5)
            max_input_excess = max(max_input_excess, input_excess)
            close(-step["reward"], step["performance"] + step["compute"] + step["constraint"],
                  (job, split, j, t, "reward"))
            for key in sums:
                sums[key] += step[key]
        termination = episode["termination"]
        assert termination in ("constraint", "goal", "steps")
        if termination == "constraint":
            assert violated and len(trace) < (150 if task == "vehicle" else 100)
        elif termination == "steps":
            assert len(trace) == (150 if task == "vehicle" else 100)
        else:
            assert task == "vehicle" and not violated
            goal_index = case["reference"]["traj_steps"] - 1
            goal_x = case["tvp"]["trajectory_x"][goal_index]["true"][0]
            goal_y = case["tvp"]["trajectory_y"][goal_index]["true"][0]
            state = trace[-1]["state"]
            assert math.hypot(state["x"] - goal_x, state["y"] - goal_y) <= 0.5 + 1e-8
        for key, summary_key in (("performance", "performance_cost"),
                                 ("compute", "computation_cost"), ("constraint", "constraint_cost")):
            close(sums[key], episode[summary_key], (job, split, j, key))
        close(sum(sums.values()), episode["total_cost"], (job, split, j, "total"))
        close(mean(step["horizon"] for step in trace), episode["mean_horizon"],
              (job, split, j, "mean H"))
        total_steps += len(trace)
        failures += episode["solver_failure_steps"]
    close(mean(e["total_cost"] for e in summary["episodes"]), summary["mean_total_cost"],
          (job, split, "mean total"))
    assert summary["constraint_episodes"] == sum(e["termination"] == "constraint" for e in summary["episodes"])
    assert summary["goal_episodes"] == sum(e["termination"] == "goal" for e in summary["episodes"])
    return {"task": task, "method": method, "seed": seed, "split": split,
            "model_sha256": digest(source / "model.zip"), "bank_sha256": digest(bank_path),
            "train_steps": trained["steps"], "train_updates": trained["updates"],
            "mean_total_cost": summary["mean_total_cost"],
            "mean_physical_cost": mean(e["performance_cost"] for e in summary["episodes"]),
            "mean_H_cost": mean(e["computation_cost"] for e in summary["episodes"]),
            "mean_constraint_cost": mean(e["constraint_cost"] for e in summary["episodes"]),
            "constraint_episodes": summary["constraint_episodes"],
            "goal_episodes": summary["goal_episodes"], "solver_failure_steps": failures,
            "evaluated_steps": total_steps, "input_bound_excess_steps": input_excess_steps,
            "max_input_bound_excess": max_input_excess,
            "unlabeled_terminal_violations": unlabeled_terminal_violations,
            "scene_costs": [e["total_cost"] for e in summary["episodes"]],
            "mean_H": mean(e["mean_horizon"] for e in summary["episodes"])}


def comparison(rows, task):
    subset = [r for r in rows if r["task"] == task]
    lookup = {(r["method"], r["seed"]): r for r in subset}
    paired = []
    for seed in range(3):
        modified, author, fixed = (lookup[(method, seed)] for method in ("min_q", "author_rl", "fixed"))
        paired.append({"seed": seed, "min_q_minus_author": modified["mean_total_cost"] - author["mean_total_cost"],
                       "min_q_minus_fixed": modified["mean_total_cost"] - fixed["mean_total_cost"],
                       "scene_min_q_minus_author": [x-y for x, y in zip(modified["scene_costs"], author["scene_costs"])],
                       "scene_min_q_minus_fixed": [x-y for x, y in zip(modified["scene_costs"], fixed["scene_costs"])]})
    costs = {method: [lookup[(method, seed)]["mean_total_cost"] for seed in range(3)]
             for method in ("author_rl", "min_q", "fixed")}
    strict_cost = max(costs["min_q"]) < min(costs["fixed"])
    no_more_constraints = all(lookup[("min_q", seed)]["constraint_episodes"] <=
                              lookup[("fixed", seed)]["constraint_episodes"] for seed in range(3))
    no_more_solver_failures = all(lookup[("min_q", seed)]["solver_failure_steps"] <=
                                  lookup[("fixed", seed)]["solver_failure_steps"] for seed in range(3))
    return {"task": task, "fixed_H": BASELINE_H[task], "seed_mean_costs": costs,
            "mean_cost_across_seeds": {k: mean(v) for k, v in costs.items()},
            "sd_cost_across_seeds": {k: stdev(v) for k, v in costs.items()},
            "paired": paired, "strict_cost_advantage": strict_cost,
            "no_more_constraints_by_seed": no_more_constraints,
            "no_more_solver_failures_by_seed": no_more_solver_failures,
            "registered_robust_advantage": strict_cost and no_more_constraints and no_more_solver_failures}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("validation", "test"), required=True)
    args = ap.parse_args()
    verify_freeze()
    assert read(OUT / (args.split + "_eval_complete.json"))["complete"]
    if args.split == "test":
        assert read(OUT / "validation_audit_passed.json")["passed"]
    rows = [audit_one(args.split, job) for job in JOBS]
    comparisons = [comparison(rows, task) for task in ("vehicle", "pendulum")]
    result = {"passed": True, "split": args.split, "models": len(rows),
              "episode_conditions": len(rows) * COUNTS[args.split],
              "evaluated_steps": sum(r["evaluated_steps"] for r in rows),
              "new_training_steps": 10 * 15000,
              "new_training_updates": sum(read(model_dir(task, method, seed) / "completed.json")["updates"]
                                          for task, method, seed in JOBS if method == "min_q" or (method == "fixed" and seed > 0)),
              "inherited_fixed_search_steps": 20 * 15000,
              "inherited_author_rl_steps": 6 * 15000,
              "note": "Fixed selection used ten H values at seed0 per task; additional fixed seeds1/2 each received 15000 steps. Totals exclude interrupted historical work. H is a compute proxy, not latency.",
              "rows": rows, "comparisons": comparisons,
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(OUT / (args.split + "_audit.json"), result)
    write(OUT / (args.split + "_audit_passed.json"), {"passed": True,
          "audit_sha256": digest(OUT / (args.split + "_audit.json"))})
    print(json.dumps({"passed": True, "split": args.split,
                      "episode_conditions": result["episode_conditions"],
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
