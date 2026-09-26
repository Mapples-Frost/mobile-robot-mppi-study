"""Pre-register and audit a matched fixed-H terminal-value transplant."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from min_q_eval_suite import model_dir
from min_q_train_suite import verify_freeze as verify_prior
from paper_grid_audit_report import physics
from run import evaluate, snapshot, weights_hash, write
from runtime import ART, ROOT, imports, make_env


OUT = ART / "results/terminal_transplant_2026-09-24"
SEEDS = {"validation": 26092471, "test": 26092472}
COUNTS = {"validation": 10, "test": 20}
TASKS = ("vehicle", "pendulum")
VARIANTS = ("min_q_own", "min_q_fixed_value", "fixed")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol():
    return {
        "label": "Mechanism/method extension, not original Bohn method reproduction.",
        "hypothesis": "A terminal value trained with the selected fixed H may improve the existing min-Q horizon policy without changing its learned actor or critic.",
        "single_intervention": "At inference only, replace each min-Q model's terminal polynomial with the matched-seed fixed-H model's polynomial. The min-Q policy weights and horizon action remain unchanged.",
        "arms": {
            "min_q_own": "Existing min-Q actor and own terminal value.",
            "min_q_fixed_value": "Same min-Q actor, fixed-H matched-seed terminal value.",
            "fixed": "Selected fixed-H actor with its own terminal value; shares donor and reset warmup with the transplant arm."
        },
        "tasks": list(TASKS), "training_seeds": [0, 1, 2],
        "fixed_H": {"vehicle": 25, "pendulum": 30},
        "training_budget": "No new training. Reuse six min-Q and six selected fixed-H models, each trained 15000 steps. The inherited ten-H seed0 search per task cost 300000 training steps and fixed seeds1/2 cost 60000 more; min-Q training cost 90000 steps. Do not erase prior unsuccessful/interrupted work from project accounting.",
        "splits": {name: {"seed": SEEDS[name], "episodes_per_task": COUNTS[name]} for name in SEEDS},
        "selection": "H25/H30 were selected on the historical validation bank before min-Q training. Keep all three final checkpoints and all scenes; no new H, seed, model or threshold selection.",
        "primary": "Matched-scene mean total physical + H-proxy + constraint cost, lower is better.",
        "secondary": ["constraint stops", "goal arrivals", "solver-failed steps", "H choices", "common post-warmup initial state"],
        "validation_gate": "Open the new test bank only if the transplant lowers mean cost by at least 2% versus min-Q's own value in at least two of three seeds on each task, creates no extra constraint stops or solver-failed steps by seed, and the across-seed mean is below the selected fixed-H arm on each task. Otherwise stop with a negative validation result.",
        "test_claim": "If the gate passes, report all seeds and require transplanted cost below all matched fixed-H seed means on both tasks, with no excess constraint stops or solver-failed steps. This test is independent only until opened once; do not tune on it thereafter.",
        "limits": "Terminal transplant changes the MPC reset warmup and subsequent trajectory. A matched donor in the fixed arm controls this starting-state effect; it does not prove that terminal learning caused any performance change. H cost is a proxy, not measured speed. Exposed historical holdout and min-Q test are excluded from model or parameter selection."
    }


def verify():
    verify_prior()
    assert read(OUT / "protocol.json") == protocol()
    hashes = read(OUT / "bank_hashes.json")
    assert len(hashes) == 5
    for path, expected in hashes.items():
        assert digest(ROOT / path) == expected, path


def prepare():
    verify_prior()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        assert read(path) == protocol(), "Protocol already frozen"
    else:
        write(path, protocol())
    for task in TASKS:
        for split in SEEDS:
            bank_path = OUT / (task + "_" + split + "_bank.json")
            if bank_path.exists():
                bank = read(bank_path)
                assert bank["seed"] == SEEDS[split] and len(bank["cases"]) == COUNTS[split]
                continue
            env = make_env(task, SEEDS[split], aligned=True, scaled_obs=True)
            np.random.seed(SEEDS[split])
            cases = []
            for _ in range(COUNTS[split]):
                env.reset()
                cases.append(snapshot(env))
            write(bank_path, {"task": task, "split": split, "seed": SEEDS[split],
                              "generator": "make_env.reset + run.snapshot", "cases": cases})
    paths = [path] + sorted(OUT.glob("*_bank.json"))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    hash_path = OUT / "bank_hashes.json"
    if hash_path.exists():
        assert read(hash_path) == hashes, "Bank or protocol hash changed"
    else:
        write(hash_path, hashes)
    print(json.dumps({"prepared": True, "bank_hashes": hashes}))


def evaluate_split(split):
    verify()
    if split == "test":
        assert read(OUT / "validation_gate.json")["passed"], "Validation gate failed"
    _, SAC, _ = imports()
    for task in TASKS:
        bank_path = OUT / (task + "_" + split + "_bank.json")
        cases = read(bank_path)["cases"]
        for seed in range(3):
            min_dir = model_dir(task, "min_q", seed)
            fixed_dir = model_dir(task, "fixed", seed)
            min_env = make_env(task, 923, aligned=True, scaled_obs=True)
            fixed_env = make_env(task, 923, fixed_horizon=protocol()["fixed_H"][task],
                                 aligned=True, scaled_obs=True)
            min_model = SAC.load(str(min_dir / "model.zip"), env=min_env)
            fixed_model = SAC.load(str(fixed_dir / "model.zip"), env=fixed_env)
            min_hash, fixed_hash = weights_hash(min_model), weights_hash(fixed_model)
            assert min_hash == read(min_dir / "completed.json")["final_hash"]
            assert fixed_hash == read(fixed_dir / "completed.json")["final_hash"]
            donor_weights = fixed_model.policy_tf.get_mpc_vfn_weights_and_biases()
            jobs = (("min_q_own", min_model, min_env, min_dir, None, min_hash),
                    ("min_q_fixed_value", min_model, min_env, min_dir, donor_weights, fixed_hash),
                    ("fixed", fixed_model, fixed_env, fixed_dir, None, fixed_hash))
            for variant, model, env, source, terminal, donor_hash in jobs:
                dest = OUT / "evaluations" / split / task / (variant + "_s%d" % seed)
                if (dest / "completed.json").exists():
                    done = read(dest / "completed.json")
                    assert done["model_hash"] == weights_hash(model) and done["bank_hash"] == digest(bank_path)
                    continue
                if dest.exists() and any(dest.iterdir()):
                    raise RuntimeError("Inspect incomplete evaluation: " + str(dest))
                dest.mkdir(parents=True, exist_ok=True)
                evaluate(model, env, cases, dest, read(source / "manifest.json")["fixed_horizon"],
                         True, terminal_weights=terminal)
                assert weights_hash(model) == (min_hash if variant != "fixed" else fixed_hash)
                write(dest / "completed.json", {"variant": variant, "task": task, "seed": seed,
                      "model_dir": str(source), "model_hash": weights_hash(model),
                      "terminal_donor_hash": donor_hash, "bank_hash": digest(bank_path),
                      "frozen": True, "episodes": len(cases)})
                print(json.dumps({"split": split, "task": task, "seed": seed,
                                  "variant": variant, "complete": True}), flush=True)


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-6), (a, b)


def audit_shared_warmup(task, seed, cases):
    _, SAC, _ = imports()
    min_env = make_env(task, 923, aligned=True, scaled_obs=True)
    fixed_env = make_env(task, 923, fixed_horizon=protocol()["fixed_H"][task],
                         aligned=True, scaled_obs=True)
    donor_dir = model_dir(task, "fixed", seed)
    donor = SAC.load(str(donor_dir / "model.zip"), env=fixed_env)
    assert weights_hash(donor) == read(donor_dir / "completed.json")["final_hash"]
    weights = donor.policy_tf.get_mpc_vfn_weights_and_biases()
    min_env.set_value_function_weights_and_biases(*weights)
    fixed_env.set_value_function_weights_and_biases(*weights)
    for case in cases:
        min_env.reset(**case)
        fixed_env.reset(**case)
        a = min_env.control_system.current_state
        b = fixed_env.control_system.current_state
        assert a.keys() == b.keys()
        for key in a:
            close(a[key], b[key])
    return len(cases)


def audit_split(split):
    verify()
    rows = []
    warmup_pairs = 0
    for task in TASKS:
        bank_path = OUT / (task + "_" + split + "_bank.json")
        cases = read(bank_path)["cases"]
        for seed in range(3):
            for variant in VARIANTS:
                dest = OUT / "evaluations" / split / task / (variant + "_s%d" % seed)
                done, summary = read(dest / "completed.json"), read(dest / "summary.json")
                source = model_dir(task, "fixed" if variant == "fixed" else "min_q", seed)
                donor = model_dir(task, "fixed" if variant != "min_q_own" else "min_q", seed)
                assert done["model_dir"] == str(source) and done["frozen"]
                assert done["model_hash"] == read(source / "completed.json")["final_hash"]
                assert done["terminal_donor_hash"] == read(donor / "completed.json")["final_hash"]
                assert done["bank_hash"] == digest(bank_path)
                assert len(summary["episodes"]) == done["episodes"] == len(cases) == COUNTS[split]
                assert summary["terminal_value"]
                failures = 0
                for j, (episode, case) in enumerate(zip(summary["episodes"], cases)):
                    trace = read(dest / ("trace_%02d.json" % j))
                    assert trace and episode["episode"] == j and len(trace) == episode["steps"]
                    assert len(trace) <= (150 if task == "vehicle" else 100)
                    sums = {"performance": 0., "compute": 0., "constraint": 0.}
                    for t, step in enumerate(trace):
                        assert 1 <= step["horizon"] <= 50
                        if variant == "fixed":
                            assert step["horizon"] == protocol()["fixed_H"][task]
                        physical, violated, excess = physics(task, step, case, t)
                        close(physical, step["performance"])
                        close(step["compute"], step["horizon"] * (.001 if task == "vehicle" else .003))
                        remaining = (150 if task == "vehicle" else 100) - t - 1
                        close(step["constraint"], (2 if task == "vehicle" else 10) * remaining if violated else 0.)
                        assert excess <= 1e-5
                        if violated:
                            assert t == len(trace) - 1
                        close(-step["reward"], sum(step[k] for k in sums))
                        for key in sums:
                            sums[key] += step[key]
                    term = episode["termination"]
                    assert term in ("constraint", "goal", "steps")
                    if term == "constraint":
                        assert violated and len(trace) < (150 if task == "vehicle" else 100)
                    elif term == "steps":
                        assert len(trace) == (150 if task == "vehicle" else 100)
                    else:
                        assert task == "vehicle" and not violated
                        ix = case["reference"]["traj_steps"] - 1
                        gx = case["tvp"]["trajectory_x"][ix]["true"][0]
                        gy = case["tvp"]["trajectory_y"][ix]["true"][0]
                        state = trace[-1]["state"]
                        assert math.hypot(state["x"] - gx, state["y"] - gy) <= .5 + 1e-8
                    for key, label in (("performance", "performance_cost"),
                                       ("compute", "computation_cost"), ("constraint", "constraint_cost")):
                        close(sums[key], episode[label])
                    close(sum(sums.values()), episode["total_cost"])
                    assert episode["solver_failure_steps"] == sum(not step["solver_success"] for step in trace)
                    failures += episode["solver_failure_steps"]
                close(mean(e["total_cost"] for e in summary["episodes"]), summary["mean_total_cost"])
                assert summary["constraint_episodes"] == sum(e["termination"] == "constraint" for e in summary["episodes"])
                assert summary["goal_episodes"] == sum(e["termination"] == "goal" for e in summary["episodes"])
                rows.append({"task": task, "seed": seed, "variant": variant,
                             "mean_cost": summary["mean_total_cost"],
                             "scene_costs": [e["total_cost"] for e in summary["episodes"]],
                             "constraint_stops": summary["constraint_episodes"],
                             "goals": summary["goal_episodes"], "solver_failures": failures})
            warmup_pairs += audit_shared_warmup(task, seed, cases)
    result = {"passed": True, "split": split, "episode_conditions": len(rows) * COUNTS[split],
              "warmup_pairs": warmup_pairs, "rows": rows, "script_sha256": digest(Path(__file__))}
    write(OUT / (split + "_audit.json"), result)
    if split == "validation":
        lookup = {(r["task"], r["seed"], r["variant"]): r for r in rows}
        task_gates = {}
        for task in TASKS:
            improvements, constraints, failures = [], [], []
            for seed in range(3):
                own = lookup[(task, seed, "min_q_own")]
                trans = lookup[(task, seed, "min_q_fixed_value")]
                improvements.append(trans["mean_cost"] <= .98 * own["mean_cost"])
                constraints.append(trans["constraint_stops"] <= own["constraint_stops"])
                failures.append(trans["solver_failures"] <= own["solver_failures"])
            below_fixed = mean(lookup[(task, s, "min_q_fixed_value")]["mean_cost"] for s in range(3)) < mean(
                lookup[(task, s, "fixed")]["mean_cost"] for s in range(3))
            task_gates[task] = {"improved_seeds": sum(improvements),
                                "constraints_ok": all(constraints), "solver_ok": all(failures),
                                "below_fixed_mean": below_fixed,
                                "passed": sum(improvements) >= 2 and all(constraints) and all(failures) and below_fixed}
        write(OUT / "validation_gate.json", {"passed": all(g["passed"] for g in task_gates.values()),
                                                "tasks": task_gates, "audit_hash": digest(OUT / "validation_audit.json")})
    print(json.dumps({"split": split, "passed": True, "rows": len(rows),
                      "validation_gate": read(OUT / "validation_gate.json") if split == "validation" else None}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "evaluate", "audit"))
    parser.add_argument("--split", choices=tuple(SEEDS))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        assert args.split
        if args.phase == "evaluate":
            evaluate_split(args.split)
        else:
            audit_split(args.split)


if __name__ == "__main__":
    main()
