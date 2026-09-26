"""Matched-state actor/critic diagnosis with deterministic and soft continuations."""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import ndtr

from runtime import ART, ROOT, imports
from run import write, weights_hash
from optimized_runtime import install_terminal
from mechanism_probe import checked_step
import recoverable_runtime as distribution

OUT = ART / 'results/horizon_credit_probe'
GROUPS = ['recoverable_distribution', 'prior_refinement']
CASES = [0, 3]
TIMES = [24, 74]
REPEATS = 4
SCALE = .6


def read(path):
    return json.loads(path.read_text())


def horizon(action):
    return int(np.clip(np.rint(1 + (float(action) + 1) * 24.5), 1, 50))


def worker(group, seed):
    source = ART / 'results' / group
    dest = OUT / ('%s_s%d' % (group, seed))
    dest.mkdir(parents=True, exist_ok=True)
    distribution.OUT = source
    distribution.install_distribution()
    install_terminal('pendulum')
    _, SAC, _ = imports()
    folder = source / ('pendulum_rl_s%d' % seed)
    model = SAC.load(str(folder / 'model.zip'))
    before = weights_hash(model)
    assert before == read(folder / 'completed.json')['final_hash']
    assert model.gamma == .97 and model.ent_coef == .01 and model.time_aware
    env = distribution.make_env('pendulum', 26091820 + seed)
    env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases())
    cases = read(source / 'validation_bank.json')['cases']
    fixed = read(source / 'selected_fixed.json')['h']
    centers = np.linspace(-1, 1, 50, dtype=np.float32)
    dense = np.linspace(-1, 1, 1001, dtype=np.float32)

    def q(obs, actions):
        batch = np.repeat(obs[None, :], len(actions), axis=0)
        values = model.sess.run([model.step_ops[4], model.step_ops[5]], {
            model.observations_ph: batch, model.actions_ph: np.asarray(actions, dtype=np.float32).reshape(-1, 1)})
        return [v.ravel() for v in values]

    def policy(obs, noise=None):
        mu, std = model.sess.run([model.policy_tf.act_mu, model.policy_tf.std], {model.observations_ph: obs[None, :]})
        mu, std = float(mu[0, 0]), float(std[0, 0])
        latent = mu if noise is None else mu + std * noise
        action = float(np.tanh(latent))
        # Use the author's epsilon conventions for likelihood and tanh Jacobian.
        logp = -.5 * (((latent - mu) / (std + 1e-6))**2 + 2 * np.log(std) + np.log(2 * np.pi))
        logp -= np.log(1 - action**2 + 1e-6)
        return horizon(action), float(logp), action, mu, std

    def reset_prefix(case_id, t):
        obs = env.reset(**copy.deepcopy(cases[case_id]))
        trace = read(folder / 'eval_value' / ('trace_%02d.json' % case_id))
        assert t < len(trace)
        for i in range(t):
            obs, done, row = checked_step(env, 'pendulum', trace[i]['horizon'])
            assert not done and row['state'] == trace[i]['state']
        return obs, trace

    def rollout(case_id, t, first_h, noise):
        obs, original = reset_prefix(case_id, t)
        start = copy.deepcopy(env.control_system.current_state)
        initial_obs = obs.tolist()
        rows, soft, entropy = [], 0., 0.
        for k in range(100 - t):
            if k == 0:
                h, logp = first_h, 0.
            elif noise is None:
                h = int(np.clip(np.rint(model.predict(obs, deterministic=True)[0][0]), 1, 50))
                logp = 0.
            else:
                h, logp, _, _, _ = policy(obs, noise[k])
            obs, done, row = checked_step(env, 'pendulum', h)
            row['continuation_log_probability'] = logp
            rows.append(row)
            soft += .97**k * (-row['cost'] / SCALE - model.ent_coef * logp)
            entropy += .97**k * (-model.ent_coef * logp)
            if done:
                break
        assert done
        tail = 0.
        if rows[-1]['termination'] == 'steps':
            tail = float(model.sess.run(model.value_target, {model.next_observations_ph: obs[None, :]})[0, 0])
        return {'start_state': start, 'initial_observation': initial_obs, 'trace': rows,
                'steps': len(rows), 'termination': rows[-1]['termination'],
                'cost': sum(r['cost'] for r in rows),
                'discounted_cost': sum(.97**k * r['cost'] for k, r in enumerate(rows)),
                'soft_return_finite': soft, 'entropy_contribution': entropy,
                'discounted_target_tail': .97**len(rows) * tail,
                'soft_return_with_target_tail': soft + .97**len(rows) * tail}

    results = []
    for case_id in CASES:
        for t in TIMES:
            obs, original = reset_prefix(case_id, t)
            actor_h, _, raw, mu, std = policy(obs)
            assert actor_h == int(np.clip(np.rint(model.predict(obs, deterministic=True)[0][0]), 1, 50))
            assert actor_h == original[t]['horizon']
            q1, q2 = q(obs, centers)
            dq1, dq2 = q(obs, dense)
            qa1, qa2 = q(obs, [raw])
            q1_h, min_h = int(np.argmax(q1)) + 1, int(np.argmax(np.minimum(q1, q2))) + 1
            picks = {'actor': actor_h, 'q1': q1_h, 'min_q': min_h, 'fixed': fixed}
            edges = 2 * (np.arange(1.5, 50, 1) - 1) / 49 - 1
            cdf = ndtr((np.arctanh(edges) - mu) / std)
            mass = np.diff(np.r_[0., cdf, 1.])
            same_bin = (np.rint(1 + (dense + 1) * 24.5) == actor_h)
            header = {'case': case_id, 't': t, 'picks': picks, 'raw_actor_action': raw,
                      'actor_mu': mu, 'actor_std': std, 'actor_h_mass': float(mass[actor_h - 1]),
                      'horizon_probability': mass.tolist(), 'q1': q1.tolist(), 'q2': q2.tolist(),
                      'actor_q1_raw': float(qa1[0]), 'actor_q2_raw': float(qa2[0]),
                      'q1_dense_best_h': horizon(dense[np.argmax(dq1)]),
                      'q1_extraction_gap': float(dq1.max() - qa1[0]),
                      'q1_same_actor_bin_range': float(np.ptp(dq1[same_bin]))}
            anchor = dest / ('case%d_t%d' % (case_id, t))
            anchor.mkdir(exist_ok=True)
            deterministic = {}
            for h in sorted(set(picks.values())):
                result = rollout(case_id, t, h, None)
                assert result['initial_observation'] == obs.tolist()
                if h == actor_h:
                    keys = ['state', 'input', 'horizon', 'cost']
                    assert [{k: r[k] for k in keys} for r in result['trace']] == [{k: r[k] for k in keys} for r in original[t:]]
                write(anchor / ('det_h%d.json' % h), result)
                deterministic[str(h)] = {k: v for k, v in result.items() if k not in ['trace', 'initial_observation', 'start_state']}
            stochastic = {}
            for h in sorted({actor_h, q1_h}):
                samples = []
                for repeat in range(REPEATS):
                    noise_seed = 26091830 + case_id * 1000 + t * 10 + repeat
                    noise = np.random.RandomState(noise_seed).normal(size=100)
                    result = rollout(case_id, t, h, noise)
                    assert result['initial_observation'] == obs.tolist()
                    result['common_noise_seed'] = noise_seed
                    write(anchor / ('stoch_h%d_r%d.json' % (h, repeat)), result)
                    samples.append({k: v for k, v in result.items() if k not in ['trace', 'initial_observation', 'start_state']})
                stochastic[str(h)] = samples
            result = {**header, 'deterministic': deterministic, 'stochastic': stochastic}
            write(anchor / 'summary.json', result)
            results.append(result)
            print(json.dumps({'group': group, 'seed': seed, 'case': case_id, 't': t,
                              'picks': picks, 'det_costs': {h: v['discounted_cost'] for h, v in deterministic.items()}}), flush=True)
    assert weights_hash(model) == before
    write(dest / 'summary.json', {'complete': True, 'frozen': True, 'group': group, 'seed': seed,
                                  'weights_sha256': before, 'loaded_reward_scale': model.reward_scale,
                                  'actual_training_reward_scale': SCALE, 'results': results})
    model.sess.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=GROUPS)
    parser.add_argument('--seed', type=int, choices=range(3))
    args = parser.parse_args()
    if args.group is not None:
        assert args.seed is not None
        worker(args.group, args.seed)
        return
    OUT.mkdir(exist_ok=True)
    protocol = {'groups': GROUPS, 'seeds': [0, 1, 2], 'validation_cases': CASES, 'times': TIMES,
                'stochastic_repeats': REPEATS, 'first_actions': 'actor, Q1-grid argmax, min(Q1,Q2)-grid argmax, validation-selected fixed H',
                'stochastic_first_actions': 'actor and Q1-grid argmax, four shared-noise continuations',
                'scope': 'Post-hoc mechanistic diagnosis on validation only; no training, new deployment policy or holdout improvement claim.',
                'controls': 'Exact original prefixes and deterministic actor suffixes; frozen terminal and all model weights; identical observation.',
                'soft_return': 'First reward, then reward minus alpha log pi; physical reward divided by .6; gamma .97. Report finite and time-limit target-V tail separately.',
                'limits': 'Four stochastic samples are a small diagnostic sample. Tail uses learned target V, so is not independent ground truth. Continuous Q is queried at bin centers and a dense action grid.',
                'script_sha256': hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    path = OUT / 'protocol.json'
    if path.exists():
        assert read(path) == protocol, 'Existing experiment protocol differs'
    else:
        write(path, protocol)
    def launch(job):
        group, seed = job
        completed = OUT / ('%s_s%d' % job) / 'summary.json'
        if completed.exists():
            assert read(completed)['complete']
            return
        with open(OUT / ('%s_s%d.log' % job), 'a') as log:
            result = subprocess.run([sys.executable, '-u', __file__, '--group', group, '--seed', str(seed)],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        assert result.returncode == 0, job
        print(json.dumps({'completed': group, 'seed': seed}), flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(launch, [(g, s) for g in GROUPS for s in range(3)]))
    write(OUT / 'completed.json', {'complete': True, 'models': 6, 'anchors': 24})


if __name__ == '__main__':
    main()
