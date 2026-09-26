"""Collect paired branch returns, calibrate only twin Q, and evaluate controls."""
import argparse
import copy
import fcntl
import json
import subprocess
import sys
import os
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from branch_calibration_protocol import (OUT, TASKS, HORIZONS, ANCHORS, REPEATS,
    COUNTS, BASE_H, ARMS, bank_path, protocol, verify, model_dir)
from runtime import ROOT, imports, make_env
from run import write, weights_hash, evaluate, snapshot
from paper_h_soft_probe import read, digest, check_step, horizon, SCALES, GAMMA
from paper_h_soft_report import audit_branch, verify_inference


def meter(env, dest):
    """Count attempted calls durably before each call, including interrupted work."""
    path = dest / ('attempt_%d_%d.json' % (time.time_ns(), os.getpid()))
    counts = {'pid': os.getpid(), 'reset_calls': 0, 'step_calls': 0,
              'scope': 'Explicit calls after environment construction; resets each include one H50 warmup.'}
    for name in ('reset', 'step'):
        original = getattr(env, name)
        def counted(*args, _name=name, _call=original, **kwargs):
            counts[_name + '_calls'] += 1
            write(path, counts)
            return _call(*args, **kwargs)
        setattr(env, name, counted)
    return counts


def observed_step(env, task, h, case, t):
    before = env.get_observation().tolist()
    state = copy.deepcopy(env.control_system.current_state)
    obs, done, row = check_step(env, task, h, case, t)
    row.update(observation=before, previous_state=state, next_observation=obs.tolist())
    return obs, done, row


def policy_ops(model):
    import tensorflow as tf
    from stable_baselines.sac.policies import gaussian_likelihood, apply_squashing_func
    with model.graph.as_default():
        z = tf.placeholder(tf.float32, [None, 1], name='calibration_common_noise')
        p = model.policy_tf
        ls = p.std.op.inputs[0]
        latent = p.act_mu + p.std * z
        _, a, lp = apply_squashing_func(p.act_mu, latent, gaussian_likelihood(latent, p.act_mu, ls))
    def sample(obs, noise):
        values = model.sess.run([p.act_mu, ls, p.std, latent, a, lp],
                               {model.observations_ph: obs[None, :], z: [[float(noise)]]})
        mu, log_std, std, lat, act, logp = [float(v.ravel()[0]) for v in values]
        return horizon(act), logp, {'mu': mu, 'log_std': log_std, 'std': std,
            'latent': lat, 'raw_action': act, 'z': float(np.float32(noise)), 'observation': obs.tolist()}
    return sample


def actor_h(model, obs):
    return int(np.clip(np.rint(model.predict(obs, deterministic=True)[0][0]), 1, 50))


def collect(model, env, task, seed, cases, dest, smoke=False):
    sample = policy_ops(model)
    original_hash = weights_hash(model)
    groups, skipped, files = [], [], {}
    budget = dict(source_steps=0, prefix_steps=0, suffix_steps=0, resets=0, rollouts=0, skipped_anchors=0)
    for case_id, case in enumerate(cases):
        source_path = dest / ('source_%02d.json' % case_id)
        if source_path.exists():
            original = read(source_path)
        else:
            obs = env.reset(**copy.deepcopy(case))
            original = []
            for t in range(env.max_steps):
                obs, done, row = observed_step(env, task, actor_h(model, obs), case, t)
                original.append(row)
                if done: break
            assert done
            write(source_path, original)
        budget['source_steps'] += len(original)
        budget['resets'] += 1
        files[str(source_path)] = digest(source_path)
        def prefix(anchor):
            obs = env.reset(**copy.deepcopy(case))
            for t in range(anchor):
                obs, done, row = check_step(env, task, original[t]['horizon'], case, t)
                assert not done and row['state'] == original[t]['state']
                assert row['solver_success'] == original[t]['solver_success']
            return obs
        for anchor in ((20,) if smoke else ANCHORS):
            if anchor >= len(original):
                skipped.append({'case': case_id, 'anchor': anchor, 'steps': len(original),
                                'termination': original[-1]['termination']})
                budget['skipped_anchors'] += 1
                continue
            path = dest / ('case%02d_t%03d.json' % (case_id, anchor))
            if not path.exists():
                obs = prefix(anchor)
                h_actor = actor_h(model, obs)
                assert h_actor == original[anchor]['horizon']
                initial = obs.tolist()
                candidates = sorted(set(((10, 25) if smoke else HORIZONS) + (h_actor,)))
                branches = {str(h): [] for h in candidates}
                for repeat in range(REPEATS):
                    noise_seed = 2609250000 + TASKS.index(task) * 100000 + seed * 10000 + case_id * 100 + anchor + repeat
                    noise = np.random.RandomState(noise_seed).normal(size=200)
                    for h in candidates:
                        obs = prefix(anchor)
                        assert obs.tolist() == initial
                        start_state = copy.deepcopy(env.control_system.current_state)
                        trace = []
                        for k in range(env.max_steps - anchor):
                            chosen, lp, sampled = (h, 0., None) if k == 0 else sample(obs, noise[k])
                            obs, done, row = observed_step(env, task, chosen, case, anchor + k)
                            row.update(log_probability=lp, sampling=sampled)
                            trace.append(row)
                            if done: break
                        assert done
                        costs = [-r['reward'] for r in trace]
                        dc = sum(GAMMA**k * c for k, c in enumerate(costs))
                        ent = sum(GAMMA**k * -r['log_probability'] for k, r in enumerate(trace))
                        soft = -dc / SCALES[task] + ent
                        tail = 0.
                        if trace[-1]['termination'] == 'steps':
                            tail = float(model.sess.run(model.value_target, {model.next_observations_ph: obs[None, :]})[0, 0])
                        branches[str(h)].append({'first_h': h, 'first_log_probability': 0.,
                            'initial_observation': initial, 'start_state': start_state, 'trace': trace,
                            'steps': len(trace), 'noise_seed': noise_seed, 'final_observation': obs.tolist(),
                            'total_cost': sum(costs), 'discounted_cost': dc, 'entropy_contribution': ent,
                            'finite_soft_return': soft, 'target_v': tail,
                            'discounted_target_v_tail': GAMMA**len(trace) * tail,
                            'soft_return_with_target_tail': soft + GAMMA**len(trace) * tail,
                            'termination': trace[-1]['termination'],
                            'solver_failure_steps': sum(not r['solver_success'] for r in trace)})
                write(path, {'case': case_id, 'anchor': anchor, 'actor_h': h_actor,
                             'candidates': candidates, 'observation': initial, 'branches': branches})
            data = read(path)
            assert data['case'] == case_id and data['anchor'] == anchor
            assert data['actor_h'] == original[anchor]['horizon']
            assert data['candidates'] == sorted(set(((10, 25) if smoke else HORIZONS) + (data['actor_h'],)))
            samples, tails, labels = [], [], []
            budget['prefix_steps'] += anchor
            budget['resets'] += 1
            for h in data['candidates']:
                runs = data['branches'][str(h)]
                assert len(runs) == REPEATS
                for r, b in enumerate(runs):
                    assert b['initial_observation'] == data['observation']
                    assert b['noise_seed'] == 2609250000 + TASKS.index(task) * 100000 + seed * 10000 + case_id * 100 + anchor + r
                    samples.extend(audit_branch(task, case, original, anchor, h, r, b))
                    if b['termination'] == 'steps': tails.append(b)
                    budget['prefix_steps'] += anchor
                    budget['resets'] += 1
                    budget['suffix_steps'] += len(b['trace'])
                    budget['rollouts'] += 1
                labels.append(float(np.mean([b['soft_return_with_target_tail'] for b in runs])))
            verify_inference(model, samples, tails)
            groups.append({'case': case_id, 'anchor': anchor, 'observation': data['observation'],
                           'horizons': data['candidates'], 'targets': labels, 'source_hash': digest(path)})
            files[str(path)] = digest(path)
            print(json.dumps({'task': task, 'seed': seed, 'case': case_id, 'anchor': anchor,
                              'audited_rollouts': budget['rollouts']}), flush=True)
    assert weights_hash(model) == original_hash
    dataset = {'groups': groups, 'skipped': skipped, 'budget': budget, 'hashes': files,
               'bank_hash': digest(bank_path(task, 'train', seed)) if not smoke else None,
               'original_hash': original_hash, 'data_audit_passed': True}
    write(dest / 'dataset.json', dataset)
    return dataset


def fit(model, data, dest, smoke=False):
    import tensorflow as tf
    groups = data['groups']
    assert len(groups) >= (1 if smoke else 4), 'Insufficient anchors; preserve all failures and stop'
    before = model.get_parameters()
    obs, actions, targets, indices = [], [], [], []
    for group_id, g in enumerate(groups):
        n = len(g['horizons'])
        obs.extend([g['observation']] * n)
        actions.extend([[(h - 1) / 24.5 - 1] for h in g['horizons']])
        targets.extend(np.array(g['targets']) - np.mean(g['targets']))
        indices.extend([group_id] * n)
    with model.graph.as_default():
        existing = set(tf.global_variables())
        y = tf.placeholder(tf.float32, [None], name='branch_centered_target')
        ids = tf.placeholder(tf.int32, [None], name='branch_state_index')
        losses = []
        for q in model.step_ops[4:6]:
            flat = tf.reshape(q, [-1])
            centered = flat - tf.gather(tf.unsorted_segment_mean(flat, ids, len(groups)), ids)
            residual = tf.losses.huber_loss(y, centered, reduction=tf.losses.Reduction.NONE)
            losses.append(tf.reduce_mean(tf.unsorted_segment_mean(residual, ids, len(groups))))
        loss = sum(losses) / 2
        variables = [v for v in tf.trainable_variables() if v.name.startswith(('model/values_fn/qf1/', 'model/values_fn/qf2/'))]
        assert len(variables) == 12
        train = tf.train.AdamOptimizer(learning_rate=protocol()['fit']['learning_rate'], name='branch_adam').minimize(loss, var_list=variables)
        model.sess.run(tf.variables_initializer([v for v in tf.global_variables() if v not in existing]))
    feed = {model.observations_ph: np.asarray(obs, np.float32), model.actions_ph: np.asarray(actions, np.float32),
            y: np.asarray(targets, np.float32), ids: np.asarray(indices, np.int32)}
    initial_loss = float(model.sess.run(loss, feed))
    losses = [initial_loss]
    steps = 10 if smoke else protocol()['fit']['steps']
    for i in range(steps):
        model.sess.run(train, feed)
        if (i + 1) % 10 == 0:
            losses.append(float(model.sess.run(loss, feed)))
    assert np.isfinite(losses).all()
    assert losses[-1] < initial_loss, 'Technical fit did not reduce registered training objective'
    after = model.get_parameters()
    changed = [k for k in before if not np.array_equal(before[k], after[k])]
    assert changed and all(k.startswith(('model/values_fn/qf1/', 'model/values_fn/qf2/')) for k in changed)
    model.save(str(dest / 'model'))
    _, SAC, _ = imports()
    loaded = SAC.load(str(dest / 'model.zip'))
    assert weights_hash(loaded) == weights_hash(model)
    np.testing.assert_array_equal(loaded.predict(np.asarray(obs[:2]), deterministic=True)[0],
                                  model.predict(np.asarray(obs[:2]), deterministic=True)[0])
    loaded.sess.close()
    write(dest / 'fit.json', {'steps': steps, 'groups': len(groups), 'labels': len(targets),
        'initial_loss': initial_loss, 'final_loss': losses[-1], 'loss_every_10': losses,
        'changed_parameters': changed, 'non_q_parameters_exact': True, 'reload_exact': True,
        'dataset_hash': digest(dest / 'dataset.json'), 'final_hash': weights_hash(model)})


def train_job(task, seed, smoke=False):
    verify()
    dest = OUT / ('smoke' if smoke else '%s_s%d' % (task, seed))
    dest.mkdir(exist_ok=True)
    lock = (dest / 'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (dest / 'completed.json').exists():
        from branch_calibration_audit import audit_fit
        audit_fit(task, seed, smoke)
        return
    env = make_env(task, seed, aligned=True, scaled_obs=True)
    meter(env, dest)
    _, SAC, _ = imports()
    source = model_dir(task, 'min_q', seed)
    model = SAC.load(str(source / 'model.zip'), env=env)
    assert weights_hash(model) == read(source / 'completed.json')['final_hash']
    assert model.ent_coef == 1. and model.gamma == GAMMA
    env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases())
    if smoke:
        np.random.seed(2609241000)
        env.seed(2609241000)
        env.reset()
        cases = [snapshot(env)]
        write(dest / 'smoke_bank.json', {'seed': 2609241000, 'cases': cases})
    else:
        cases = read(bank_path(task, 'train', seed))['cases']
    data = collect(model, env, task, seed, cases, dest, smoke)
    fit(model, data, dest, smoke)
    write(dest / 'completed.json', {'task': task, 'seed': seed, 'smoke': smoke,
        'model_hash': weights_hash(model), 'input_hash': digest(OUT / 'inputs_sha256.json'),
        'fit_hash': digest(dest / 'fit.json'), 'dataset_hash': digest(dest / 'dataset.json')})
    model.sess.close()


class Greedy:
    def __init__(self, model):
        self.model, self.policy_tf = model, model.policy_tf

    def predict(self, obs, deterministic=True):
        hs = sorted(set(HORIZONS + (actor_h(self.model, obs),)))
        q1, q2 = self.model.sess.run(self.model.step_ops[4:6], {
            self.model.observations_ph: np.repeat(obs[None, :], len(hs), axis=0),
            self.model.actions_ph: np.array([[(h - 1) / 24.5 - 1] for h in hs], np.float32)})
        return np.array([hs[int(np.argmax(np.minimum(q1, q2)))]], dtype=float), None


def evaluate_logged(model, env, cases, dest, fixed_h, task):
    env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases())
    episodes = []
    for j, case in enumerate(cases):
        obs = env.reset(**copy.deepcopy(case))
        trace = []
        for t in range(env.max_steps):
            start = time.perf_counter()
            h = fixed_h if fixed_h is not None else int(np.rint(model.predict(obs, deterministic=True)[0][0]))
            obs, done, row = observed_step(env, task, h, case, t)
            row['elapsed_s'] = time.perf_counter() - start
            trace.append(row)
            if done: break
        assert done
        write(dest / ('trace_%02d.json' % j), trace)
        episodes.append({'episode': j, 'steps': len(trace), 'termination': trace[-1]['termination'],
            'total_cost': sum(-r['reward'] for r in trace),
            'performance_cost': sum(r['performance'] for r in trace),
            'computation_cost': sum(r['compute'] for r in trace),
            'constraint_cost': sum(r['constraint'] for r in trace),
            'mean_horizon': float(np.mean([r['horizon'] for r in trace])),
            'solver_failure_steps': sum(not r['solver_success'] for r in trace)})
    write(dest / 'summary.json', {'episodes': episodes, 'mean_total_cost': float(np.mean([e['total_cost'] for e in episodes])),
        'constraint_episodes': sum(e['termination'] == 'constraint' for e in episodes),
        'goal_episodes': sum(e['termination'] == 'goal' for e in episodes), 'terminal_value': True})


def eval_job(task, seed, split):
    verify()
    from branch_calibration_audit import verify_artifact_hashes
    verify_artifact_hashes(OUT / 'training_audit.json')
    if split == 'test':
        gate = OUT / 'validation_gate.json'
        assert read(gate)['passed']
        verify_artifact_hashes(gate)
    _, SAC, _ = imports()
    for arm in np.random.RandomState(2609260000 + TASKS.index(task) * 100 + seed).permutation(ARMS):
        dest = OUT / 'evaluations' / split / task / ('%s_s%d' % (arm, seed))
        dest.mkdir(parents=True, exist_ok=True)
        lock = (dest / 'run.lock').open('a')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (dest / 'completed.json').exists():
            verify_artifact_hashes(dest / 'completed.json')
            continue
        assert not list(dest.glob('trace_*.json')), 'Inspect partial evaluation before retry'
        fixed_h = BASE_H[task] if arm == 'fixed' else None
        source = OUT / ('%s_s%d' % (task, seed)) if arm == 'calibrated_greedy' else model_dir(task, 'fixed' if arm == 'fixed' else 'min_q', seed)
        env = make_env(task, 923, fixed_horizon=fixed_h, aligned=True, scaled_obs=True)
        meter(env, dest)
        model = SAC.load(str(source / 'model.zip'), env=env)
        before = weights_hash(model)
        assert before == read(source / 'completed.json')['model_hash' if arm == 'calibrated_greedy' else 'final_hash']
        bank = bank_path(task, split)
        evaluate_logged(Greedy(model) if arm.endswith('greedy') else model, env, read(bank)['cases'], dest, fixed_h, task)
        assert weights_hash(model) == before
        write(dest / 'completed.json', {'task': task, 'seed': seed, 'arm': arm, 'split': split,
            'model_path': str(source / 'model.zip'), 'model_hash': before,
            'model_file_hash': digest(source / 'model.zip'), 'bank_hash': digest(bank), 'frozen': True,
            'hashes': {str(p): digest(p) for p in list(dest.glob('trace_*.json')) + [dest / 'summary.json', source / 'model.zip', bank]}})
        model.sess.close()
        print(json.dumps({'evaluated': str(dest.relative_to(OUT))}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('smoke', 'train', 'evaluate'))
    ap.add_argument('--task', choices=TASKS)
    ap.add_argument('--seed', type=int, choices=range(3))
    ap.add_argument('--split', choices=('validation', 'test'), default='validation')
    args = ap.parse_args()
    if args.phase == 'smoke': train_job('vehicle', 0, smoke=True); return
    if args.task:
        assert args.seed is not None
        if args.phase == 'train': train_job(args.task, args.seed)
        else: eval_job(args.task, args.seed, args.split)
        return
    verify()
    assert read(OUT / 'smoke/completed.json')['smoke']
    lock = (OUT / 'suite.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    def launch(job):
        task, seed = job
        with (OUT / ('%s_%s_%s_s%d.log' % (args.phase, args.split, task, seed))).open('a') as log:
            p = subprocess.run([sys.executable, '-u', __file__, args.phase, '--task', task,
                                '--seed', str(seed), '--split', args.split], cwd=str(ROOT),
                               stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps({'phase': args.phase, 'task': task, 'seed': seed, 'exit': p.returncode}), flush=True)
        return p.returncode
    with ThreadPoolExecutor(max_workers=protocol()['execution']['workers']) as pool:
        codes = list(pool.map(launch, [(t, s) for t in TASKS for s in range(3)]))
    assert all(c == 0 for c in codes), codes
    write(OUT / (args.phase + '_' + args.split + '_complete.json'), {'passed': True, 'jobs': 6})


if __name__ == '__main__':
    try: main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / ('failure_%d_%d.json' % (time.time_ns(), os.getpid())),
              {'argv': sys.argv, 'traceback': traceback.format_exc()})
        raise
