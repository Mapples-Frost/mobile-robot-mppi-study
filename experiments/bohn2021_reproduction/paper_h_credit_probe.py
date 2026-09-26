"""Matched-state horizon-credit diagnosis on the current paper configuration."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from min_q_eval_suite import model_dir
from min_q_protocol import OUT as SOURCE, BASELINE_H
from paper_grid_audit_report import physics
from run import weights_hash, write
from runtime import ART, imports, make_env


OUT = ART / "results/paper_h_credit_2026-09-24"
CASES = (0, 3)
ANCHORS = (20, 60)
METHODS = ("author_rl", "min_q")
TASKS = ("vehicle", "pendulum")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol():
    return {
        "label": "Post-hoc mechanism diagnosis, not a method comparison or strict reproduction claim.",
        "tasks": list(TASKS), "methods": list(METHODS), "training_seeds": [0, 1, 2],
        "validation_cases": list(CASES), "prefix_steps": list(ANCHORS),
        "bank": "Existing min_q_training_2026-09-24 validation bank only; do not read test.",
        "branches": "Actor, argmax Q1, argmax min(Q1,Q2), and validation-selected fixed H. Change only the first H at an identical replayed state, then restore the same deterministic actor and terminal function.",
        "q_query": "Query the continuous SAC critic at the 50 integer-H bin centers; do not assume its within-bin slope represents a physical distinction.",
        "metric": "Remaining undiscounted and gamma=.97 discounted physical + H-proxy + constraint costs; termination and solver failures separately.",
        "scope": "All frozen final models, no training, deployment, checkpoint selection, holdout evaluation or test tuning. Sampled validation states are correlated within scenes.",
        "fixed_H": BASELINE_H,
    }


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        assert read(path) == protocol(), "Frozen probe protocol changed"
    else:
        write(path, protocol())
    hashes = {}
    for task in TASKS:
        bank = SOURCE / (task + "_validation_bank.json")
        hashes[str(bank)] = digest(bank)
        for method in METHODS:
            for seed in range(3):
                source = model_dir(task, method, seed)
                done = read(source / "completed.json")
                assert done["status"] == "complete" and done["steps"] == 15000
                hashes[str(source / "model.zip")] = digest(source / "model.zip")
    frozen = OUT / "inputs_sha256.json"
    if frozen.exists():
        assert read(frozen) == hashes, "Frozen inputs changed"
    else:
        write(frozen, hashes)


def step(env, task, h, case, t):
    _, reward, done, info = env.step(np.array([float(h)]))
    row = {"state": copy.deepcopy(env.control_system.current_state),
           "input": copy.deepcopy(env.control_system.controller.current_input),
           "horizon": int(info["executed_horizon"]), "reward": float(reward),
           "performance": float(info["reward/performance"]),
           "compute": float(info["reward/computation"]),
           "constraint": float(info["reward/constraint"]),
           "solver_success": bool(info["solver_success"]),
           "termination": info.get("termination")}
    assert row["horizon"] == h
    physical, violated, excess = physics(task, row, case, t)
    assert np.isclose(physical, row["performance"], atol=1e-7)
    assert np.isclose(row["compute"], h * (.001 if task == "vehicle" else .003))
    max_steps = 150 if task == "vehicle" else 100
    penalty = (2 if task == "vehicle" else 10) * (max_steps - t - 1) if violated else 0.
    assert np.isclose(row["constraint"], penalty)
    assert np.isclose(-reward, physical + row["compute"] + penalty)
    assert excess <= 1e-5
    return done, row


def same_row(actual, saved):
    for key in ("horizon", "state", "solver_success"):
        assert actual[key] == saved[key], key
    for key in ("reward", "performance", "compute", "constraint"):
        assert np.isclose(actual[key], saved[key], rtol=1e-8, atol=1e-7), key


def worker(task, method, seed):
    prepare()
    dest = OUT / ("%s_%s_s%d" % (task, method, seed))
    if (dest / "completed.json").exists():
        assert read(dest / "completed.json")["input_hashes_sha256"] == digest(OUT / "inputs_sha256.json")
        return
    dest.mkdir(exist_ok=True)
    source = model_dir(task, method, seed)
    bank = read(SOURCE / (task + "_validation_bank.json"))["cases"]
    env = make_env(task, 923, aligned=True, scaled_obs=True)
    _, SAC, _ = imports()
    model = SAC.load(str(source / "model.zip"), env=env)
    before = weights_hash(model)
    assert before == read(source / "completed.json")["final_hash"]
    terminal = model.policy_tf.get_mpc_vfn_weights_and_biases()
    env.set_value_function_weights_and_biases(*terminal)
    centers = np.linspace(-1., 1., 50, dtype=np.float32)

    def prefix(case_id, anchor, saved):
        obs = env.reset(**copy.deepcopy(bank[case_id]))
        for t in range(anchor):
            h = saved[t]["horizon"]
            done, row = step(env, task, h, bank[case_id], t)
            assert not done
            same_row(row, saved[t])
            obs = env.get_observation()
        return obs

    def branch(case_id, anchor, first_h, saved):
        obs = prefix(case_id, anchor, saved)
        start = copy.deepcopy(env.control_system.current_state)
        rows = []
        max_steps = 150 if task == "vehicle" else 100
        for t in range(anchor, max_steps):
            h = first_h if t == anchor else int(np.clip(np.rint(model.predict(obs, deterministic=True)[0][0]), 1, 50))
            done, row = step(env, task, h, bank[case_id], t)
            rows.append(row)
            if first_h == saved[anchor]["horizon"]:
                same_row(row, saved[t])
            if done:
                break
            obs = env.get_observation()
        assert done
        costs = [sum(r[k] for k in ("performance", "compute", "constraint")) for r in rows]
        return {"start_state": start, "first_h": first_h, "trace": rows,
                "total_cost": float(sum(costs)),
                "discounted_cost": float(sum(.97**i * c for i, c in enumerate(costs))),
                "termination": rows[-1]["termination"],
                "solver_failure_steps": sum(not r["solver_success"] for r in rows)}

    summaries = []
    for case_id in CASES:
        saved = read(SOURCE / "evaluations" / "validation" / task /
                     (method + "_s%d" % seed) / ("trace_%02d.json" % case_id))
        for anchor in ANCHORS:
            assert anchor < len(saved)
            name = "case%02d_t%03d" % (case_id, anchor)
            out = dest / (name + ".json")
            if out.exists():
                summaries.append(read(out)["summary"])
                continue
            obs = prefix(case_id, anchor, saved)
            actor_h = int(np.clip(np.rint(model.predict(obs, deterministic=True)[0][0]), 1, 50))
            assert actor_h == saved[anchor]["horizon"]
            values = model.sess.run([model.step_ops[4], model.step_ops[5]], {
                model.observations_ph: np.repeat(obs[None, :], 50, axis=0),
                model.actions_ph: centers.reshape(-1, 1)})
            q1, q2 = [v.ravel() for v in values]
            picks = {"actor": actor_h, "q1": int(np.argmax(q1)) + 1,
                     "min_q": int(np.argmax(np.minimum(q1, q2))) + 1,
                     "fixed": BASELINE_H[task]}
            results = {str(h): branch(case_id, anchor, h, saved) for h in sorted(set(picks.values()))}
            summary = {"case": case_id, "anchor": anchor, "picks": picks,
                       "q1": q1.tolist(), "q2": q2.tolist(),
                       "branches": {h: {k: v for k, v in row.items() if k not in ("trace", "start_state")}
                                    for h, row in results.items()},
                       "prefix_replay_verified": True, "actor_suffix_replay_verified": True}
            write(out, {"summary": summary, "branches": results})
            summaries.append(summary)
            print(json.dumps({"model": dest.name, "case": case_id, "anchor": anchor,
                              "picks": picks, "costs": {h: r["total_cost"] for h, r in results.items()}}), flush=True)
    assert weights_hash(model) == before
    write(dest / "completed.json", {"task": task, "method": method, "seed": seed,
                                     "model_hash": before, "input_hashes_sha256": digest(OUT / "inputs_sha256.json"),
                                     "anchors": len(summaries), "summaries": summaries, "frozen": True})
    model.sess.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--seed", choices=range(3), type=int, required=True)
    args = ap.parse_args()
    worker(args.task, args.method, args.seed)


if __name__ == "__main__":
    main()
