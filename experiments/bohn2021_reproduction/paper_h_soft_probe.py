"""Shared-noise soft-return diagnosis for the current paper configuration.

This is a validation-only post-hoc probe. It does not train, select a checkpoint,
or open a test bank. The first H is intervened; later actions are sampled from
the frozen actor with common standardized Gaussian noise across first-H arms.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from min_q_eval_suite import model_dir
from min_q_protocol import OUT as SOURCE, BASELINE_H
from paper_grid_audit_report import physics
from run import weights_hash, write
from runtime import ART, ROOT, imports, make_env

OUT = ART / "results/paper_h_soft_2026-09-24"
TASKS = ("vehicle", "pendulum")
METHODS = ("author_rl", "min_q")
CASES = (0, 3)
ANCHORS = (20, 60)
REPEATS = 4
GAMMA = .97
SCALES = {"vehicle": .3, "pendulum": .6}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol():
    return {
        "label": "Validation-only soft-return mechanism diagnosis; method extension, not strict reproduction.",
        "tasks": list(TASKS), "methods": list(METHODS), "training_seeds": [0, 1, 2],
        "cases": list(CASES), "anchors": list(ANCHORS), "repeats": REPEATS,
        "bank": "Existing min_q_training_2026-09-24 validation bank and saved validation prefixes only.",
        "first_actions": "Actor deterministic H, argmax Q1 H, argmax min(Q1,Q2) H, selected fixed H.",
        "continuation": "After the forced first H, sample the frozen actor as tanh(mu + std*z). The same z sequence is used for every first-H arm at each state anchor.",
        "returns": "Report physical finite discounted cost, reward/scale - alpha log pi, entropy contribution, and target-V tail for time-limit truncation separately.",
        "forced_action_logp": "Zero at the forced first action: soft-Q includes entropy from subsequent sampled actions only. Use the author's float32 Gaussian likelihood and tanh Jacobian without inverse-tanh clipping.",
        "controls": "Exact saved actor prefixes, frozen model/terminal weights, no training or checkpoint selection, no test evaluation.",
        "limits": "Four repeats and two scenes per task are mechanism evidence only; common standardized noise does not make branches independent; learned target V is not ground truth. No new confirmation or policy selection.",
        "fixed_H": BASELINE_H, "gamma": GAMMA, "reward_scales": SCALES,
    }


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "protocol.json"
    if p.exists():
        assert read(p) == protocol(), "Frozen soft probe protocol changed"
    else:
        write(p, protocol())
    paths = [Path(__file__), Path(__file__).with_name('runtime.py'), Path(__file__).with_name('run.py'),
             Path(__file__).with_name('paper_grid_audit_report.py')]
    paths.extend((ART / 'sources').glob('*/**/*.py'))
    hashes = {str(p): digest(p) for p in paths}
    for task in TASKS:
        bank = SOURCE / (task + "_validation_bank.json")
        hashes[str(bank)] = digest(bank)
        for method in METHODS:
            for seed in range(3):
                source = model_dir(task, method, seed)
                done = read(source / "completed.json")
                assert done["status"] == "complete" and done["steps"] == 15000
                hashes[str(source / "model.zip")] = digest(source / "model.zip")
                for name in ('manifest.json', 'completed.json'):
                    hashes[str(source / name)] = digest(source / name)
                for c in CASES:
                    p = SOURCE / 'evaluations/validation' / task / (method + '_s%d' % seed) / ('trace_%02d.json' % c)
                    hashes[str(p)] = digest(p)
        p = ART / 'configs' / (task + '.json')
        hashes[str(p)] = digest(p)
    h = OUT / "inputs_sha256.json"
    if h.exists():
        assert read(h) == hashes, "Frozen soft probe input changed"
    else:
        write(h, hashes)


def horizon(action):
    a = np.array([action], dtype=np.float32)
    return int(np.clip(np.rint(1 + (a + 1) * 24.5)[0], 1, 50))


def check_step(env, task, h, case, t):
    obs, reward, done, info = env.step(np.array([float(h)]))
    row = {"state": copy.deepcopy(env.control_system.current_state),
           "input": copy.deepcopy(env.control_system.controller.current_input),
           "horizon": int(info["executed_horizon"]), "reward": float(reward),
           "performance": float(info["reward/performance"]),
           "compute": float(info["reward/computation"]),
           "constraint": float(info["reward/constraint"]),
           "solver_success": bool(info["solver_success"]), "termination": info.get("termination")}
    assert row["horizon"] == int(h)
    physical, violated, excess = physics(task, row, case, t)
    assert np.isclose(row["performance"], physical, atol=1e-7)
    assert np.isclose(row["compute"], h * (.001 if task == "vehicle" else .003))
    horizon_len = 150 if task == "vehicle" else 100
    penalty = (2 if task == "vehicle" else 10) * (horizon_len - t - 1) if violated else 0.
    assert np.isclose(row["constraint"], penalty)
    assert np.isclose(-reward, physical + row["compute"] + penalty)
    assert excess <= 1e-5
    return obs, done, row


def worker(task, method, seed):
    assert read(OUT / 'protocol.json') == protocol()
    dest = OUT / ("%s_%s_s%d" % (task, method, seed))
    dest.mkdir(exist_ok=True)
    lock = (dest / 'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (dest / "completed.json").exists():
        assert read(dest / "completed.json")["input_hashes_sha256"] == digest(OUT / "inputs_sha256.json")
        return
    source = model_dir(task, method, seed)
    bank = read(SOURCE / (task + "_validation_bank.json"))["cases"]
    env = make_env(task, 923, aligned=True, scaled_obs=True)
    _, SAC, _ = imports()
    model = SAC.load(str(source / "model.zip"), env=env)
    before = weights_hash(model)
    assert before == read(source / "completed.json")["final_hash"]
    terminal = model.policy_tf.get_mpc_vfn_weights_and_biases()
    env.set_value_function_weights_and_biases(*terminal)
    alpha = float(model.ent_coef)
    assert alpha == 1. and model.gamma == GAMMA and model.time_aware
    scale = SCALES[task]
    spec = read(source / 'manifest.json')
    assert spec['adaptations']['ent_coef'] == '1.0' and spec['adaptations']['aligned'] and spec['adaptations']['scaled_obs']
    import tensorflow as tf
    from stable_baselines.sac.policies import gaussian_likelihood, apply_squashing_func
    with model.graph.as_default():
        z_ph = tf.placeholder(tf.float32, [None, 1], name='probe_common_noise')
        p = model.policy_tf
        log_std = p.std.op.inputs[0]
        latent = p.act_mu + p.std * z_ph
        gaussian_lp = gaussian_likelihood(latent, p.act_mu, log_std)
        _, sampled_action, sampled_lp = apply_squashing_func(p.act_mu, latent, gaussian_lp)
    def sample(obs, z):
        values = model.sess.run([p.act_mu, log_std, p.std, latent, sampled_action, sampled_lp],
                               {model.observations_ph: obs[None, :], z_ph: [[float(z)]]})
        mu, ls, std, lat, act, lp = [float(v.ravel()[0]) for v in values]
        return horizon(act), lp, {'mu': mu, 'log_std': ls, 'std': std, 'latent': lat,
                                 'raw_action': act, 'z': float(np.float32(z)), 'observation': obs.tolist()}
    centers = np.linspace(-1., 1., 50, dtype=np.float32)

    def prefix(case_id, anchor, saved):
        obs = env.reset(**copy.deepcopy(bank[case_id]))
        for t in range(anchor):
            obs, done, row = check_step(env, task, saved[t]["horizon"], bank[case_id], t)
            assert not done
            assert row["state"] == saved[t]["state"] and row["horizon"] == saved[t]["horizon"]
        return obs

    def actor_stats(obs):
        policy = model.policy_tf
        mu, std = model.sess.run([policy.act_mu, policy.std], {model.observations_ph: obs[None, :]})
        return float(mu[0, 0]), float(std[0, 0])

    def q_values(obs):
        values = model.sess.run([model.step_ops[4], model.step_ops[5]], {
            model.observations_ph: np.repeat(obs[None, :], 50, axis=0),
            model.actions_ph: centers.reshape(-1, 1)})
        q1, q2 = [x.ravel() for x in values]
        return q1, q2

    def rollout(case_id, anchor, first_h, saved, z):
        obs = prefix(case_id, anchor, saved)
        start_state = copy.deepcopy(env.control_system.current_state)
        initial_obs = obs.tolist()
        rows, costs, logps, entropy = [], [], [], []
        for k in range((150 if task == "vehicle" else 100) - anchor):
            t = anchor + k
            if k == 0:
                h, lp, sampling = int(first_h), 0., None
            else:
                h, lp, sampling = sample(obs, z[k])
            obs, done, row = check_step(env, task, h, bank[case_id], t)
            row["log_probability"] = lp
            row['sampling'] = sampling
            rows.append(row)
            c = row["performance"] + row["compute"] + row["constraint"]
            costs.append(c); logps.append(lp); entropy.append(-alpha * lp)
            if done:
                break
        assert done
        finite_soft = sum(GAMMA ** k * (-cost / scale - alpha * lp) for k, (cost, lp) in enumerate(zip(costs, logps)))
        finite_entropy = sum(GAMMA ** k * e for k, e in enumerate(entropy))
        target_tail = 0.
        if rows[-1]["termination"] == "steps":
            target_tail = float(model.sess.run(model.value_target, {model.next_observations_ph: obs[None, :]})[0, 0])
        return {"first_h": first_h, "first_log_probability": 0., "trace": rows,
                "start_state": start_state, "initial_observation": initial_obs,
                "final_observation": obs.tolist(), "target_v": target_tail,
                "steps": len(rows), "total_cost": float(sum(costs)),
                "discounted_cost": float(sum(GAMMA ** k * c for k, c in enumerate(costs))),
                "finite_soft_return": float(finite_soft), "entropy_contribution": float(finite_entropy),
                "discounted_target_v_tail": float(GAMMA ** len(rows) * target_tail),
                "soft_return_with_target_tail": float(finite_soft + GAMMA ** len(rows) * target_tail),
                "termination": rows[-1]["termination"],
                "solver_failure_steps": sum(not r["solver_success"] for r in rows)}

    summaries = []
    for case_id in CASES:
        saved = read(SOURCE / "evaluations" / "validation" / task /
                     (method + "_s%d" % seed) / ("trace_%02d.json" % case_id))
        for anchor in ANCHORS:
            out = dest / ("case%02d_t%03d.json" % (case_id, anchor))
            if out.exists():
                summaries.append(read(out)["summary"]); continue
            obs = prefix(case_id, anchor, saved)
            mu, std = actor_stats(obs)
            q1, q2 = q_values(obs)
            actor_h = int(np.clip(np.rint(model.predict(obs, deterministic=True)[0][0]), 1, 50))
            assert actor_h == saved[anchor]["horizon"]
            picks = {"actor": actor_h, "q1": int(np.argmax(q1)) + 1,
                     "min_q": int(np.argmax(np.minimum(q1, q2))) + 1,
                     "fixed": BASELINE_H[task]}
            branches = {}
            for repeat in range(REPEATS):
                z = np.random.RandomState(26092490 + case_id * 100 + anchor + repeat).normal(size=200)
                for h in sorted(set(picks.values())):
                    result = rollout(case_id, anchor, h, saved, z)
                    assert result['initial_observation'] == obs.tolist()
                    result['noise_seed'] = 26092490 + case_id * 100 + anchor + repeat
                    branches.setdefault(str(h), []).append(result)
            summary = {"case": case_id, "anchor": anchor, "picks": picks,
                       "actor_mu": mu, "actor_std": std, "alpha": alpha, "reward_scale": scale,
                       "q1": q1.tolist(), "q2": q2.tolist(),
                       "branches": {h: [{k: v for k, v in r.items() if k != "trace"} for r in rs]
                                    for h, rs in branches.items()},
                       "prefix_replay_verified": True, "shared_noise_seeds": [26092490 + case_id * 100 + anchor + r for r in range(REPEATS)]}
            write(out, {"summary": summary, "branches": branches})
            summaries.append(summary)
            print(json.dumps({"model": dest.name, "case": case_id, "anchor": anchor,
                              "picks": picks, "mean_soft": {h: np.mean([x["finite_soft_return"] for x in rs]) for h, rs in branches.items()}}), flush=True)
    assert weights_hash(model) == before
    write(dest / "completed.json", {"task": task, "method": method, "seed": seed,
                                     "model_hash": before, "input_hashes_sha256": digest(OUT / "inputs_sha256.json"),
                                     "anchors": len(summaries), "repeats": REPEATS, "summaries": summaries, "frozen": True,
                                     "loaded_reward_scale": model.reward_scale, "actual_training_reward_scale": scale})
    model.sess.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--seed", type=int, choices=range(3))
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    if args.all:
        OUT.mkdir(parents=True, exist_ok=True)
        lock = (OUT / 'suite.lock').open('a')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare()
        jobs = [(t, m, s) for t in TASKS for m in METHODS for s in range(3)]
        def launch(job):
            task, method, seed = job
            with (OUT / ('%s_%s_s%d.log' % job)).open('a') as log:
                run = subprocess.run([sys.executable, '-u', __file__, '--task', task,
                                      '--method', method, '--seed', str(seed)], cwd=str(ROOT),
                                     stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps({'job': job, 'exit_code': run.returncode}), flush=True)
            return run.returncode
        with ThreadPoolExecutor(max_workers=3) as pool:
            codes = list(pool.map(launch, jobs))
        assert all(c == 0 for c in codes), codes
        write(OUT / 'completed.json', {'models': len(jobs), 'complete': True})
    else:
        assert args.task and args.method and args.seed is not None
        if not (OUT / 'inputs_sha256.json').exists():
            prepare()
        worker(args.task, args.method, args.seed)


if __name__ == "__main__":
    main()
