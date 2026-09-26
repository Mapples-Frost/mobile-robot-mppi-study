"""Recompute soft-return traces and verify frozen policy/target inference."""
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from paper_h_soft_probe import (OUT, TASKS, METHODS, CASES, ANCHORS, REPEATS,
                                SOURCE, GAMMA, SCALES, read, digest, protocol, model_dir)
from paper_grid_audit_report import physics
from runtime import ART, imports
from run import write, weights_hash


def close(a, b, atol=1e-6):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-7, abs_tol=atol), (a, b)


def audit_branch(task, case, original, anchor, h, repeat, branch):
    trace = branch['trace']
    limit = 150 if task == 'vehicle' else 100
    assert trace and len(trace) == branch['steps'] <= limit - anchor
    assert branch['start_state'] == original[anchor - 1]['state']
    assert trace[0]['horizon'] == branch['first_h'] == h
    assert branch['first_log_probability'] == trace[0]['log_probability'] == 0.
    assert trace[0]['sampling'] is None
    z = np.random.RandomState(branch['noise_seed']).normal(size=200).astype(np.float32)
    costs, entropy, samples = [], [], []
    for k, row in enumerate(trace):
        physical, violated, excess = physics(task, row, case, anchor + k)
        compute = row['horizon'] * (.001 if task == 'vehicle' else .003)
        penalty = (2 if task == 'vehicle' else 10) * (limit - anchor - k - 1) if violated else 0.
        close(physical, row['performance'])
        close(compute, row['compute'])
        close(penalty, row['constraint'])
        close(physical + compute + penalty, -row['reward'])
        assert excess <= 1e-5 and 1 <= row['horizon'] <= 50
        if k:
            s = row['sampling']
            assert s['z'] == float(z[k])
            close(s['std'], math.exp(s['log_std']))
            close(s['latent'], float(np.float32(s['mu']) + np.float32(s['std']) * z[k]))
            close(s['raw_action'], math.tanh(s['latent']))
            a = np.float32(s['raw_action'])
            mapped = int(np.clip(np.rint(np.float32(1) + (a + np.float32(1)) * np.float32(24.5)), 1, 50))
            assert row['horizon'] == mapped
            lp = -.5 * (((s['latent'] - s['mu']) / (s['std'] + 1e-6))**2
                        + 2 * s['log_std'] + math.log(2 * math.pi))
            # Float32 subtraction is part of the author's Jacobian computation.
            jac = float(np.float32(1) - np.float32(a * a) + np.float32(1e-6))
            lp -= math.log(jac)
            close(lp, row['log_probability'], atol=4e-5)
            samples.append(s)
        if k < len(trace) - 1:
            assert not violated and row['termination'] is None
        costs.append(physical + compute + penalty)
        entropy.append(-row['log_probability'])
    term = branch['termination']
    assert term == trace[-1]['termination']
    if term == 'steps':
        assert anchor + len(trace) == limit
    elif term == 'constraint':
        assert violated
    else:
        assert term == 'goal' and task == 'vehicle' and not violated
        end = case['reference']['traj_steps'] - 1
        s = trace[-1]['state']
        assert math.hypot(s['x'] - case['tvp']['trajectory_x'][end]['true'][0],
                          s['y'] - case['tvp']['trajectory_y'][end]['true'][0]) <= .5 + 1e-8
    dc = sum(GAMMA**k * c for k, c in enumerate(costs))
    ent = sum(GAMMA**k * e for k, e in enumerate(entropy))
    soft = -dc / SCALES[task] + ent
    close(sum(costs), branch['total_cost'])
    close(dc, branch['discounted_cost'])
    close(ent, branch['entropy_contribution'])
    close(soft, branch['finite_soft_return'])
    close(branch['discounted_target_v_tail'], GAMMA**len(trace) * branch['target_v'])
    close(branch['soft_return_with_target_tail'], soft + branch['discounted_target_v_tail'])
    if term != 'steps':
        assert branch['target_v'] == 0.
    assert branch['solver_failure_steps'] == sum(not r['solver_success'] for r in trace)
    return samples


def verify_inference(model, samples, tails):
    for start in range(0, len(samples), 512):
        batch = samples[start:start + 512]
        obs = np.array([s['observation'] for s in batch], dtype=np.float32)
        mu, std = model.sess.run([model.policy_tf.act_mu, model.policy_tf.std],
                                 {model.observations_ph: obs})
        np.testing.assert_allclose(mu.ravel(), [s['mu'] for s in batch], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(std.ravel(), [s['std'] for s in batch], atol=2e-5, rtol=2e-5)
    if tails:
        obs = np.array([b['final_observation'] for b in tails], dtype=np.float32)
        v = model.sess.run(model.value_target, {model.next_observations_ph: obs}).ravel()
        np.testing.assert_allclose(v, [b['target_v'] for b in tails], atol=1e-4, rtol=2e-5)


def main():
    assert read(OUT / 'protocol.json') == protocol()
    for path, expected in read(OUT / 'inputs_sha256.json').items():
        assert digest(Path(path)) == expected, path
    input_hash = digest(OUT / 'inputs_sha256.json')
    _, SAC, _ = imports()
    rows, contrasts, hashes = [], [], {}
    outcomes = {t: {'rollouts': 0, 'goals': 0, 'constraints': 0, 'time_limits': 0,
                    'solver_failed_steps': 0, 'maximum_cost': float('-inf')} for t in TASKS}
    branch_count = suffix_steps = prefix_steps = target_queries = sampled_actions = 0
    for task in TASKS:
        cases = read(SOURCE / (task + '_validation_bank.json'))['cases']
        for method in METHODS:
            for seed in range(3):
                name = '%s_%s_s%d' % (task, method, seed)
                folder = OUT / name
                done = read(folder / 'completed.json')
                assert done['frozen'] and done['anchors'] == 4 and done['repeats'] == REPEATS
                assert done['input_hashes_sha256'] == input_hash
                model = SAC.load(str(model_dir(task, method, seed) / 'model.zip'))
                before = weights_hash(model)
                assert before == done['model_hash'] == read(model_dir(task, method, seed) / 'completed.json')['final_hash']
                assert model.gamma == GAMMA and model.ent_coef == 1. and model.time_aware
                assert {(s['case'], s['anchor']) for s in done['summaries']} == {(c, a) for c in CASES for a in ANCHORS}
                inference_samples, tails, model_contrasts = [], [], []
                for s in done['summaries']:
                    case_id, anchor = s['case'], s['anchor']
                    path = folder / ('case%02d_t%03d.json' % (case_id, anchor))
                    hashes[str(path)] = digest(path)
                    data = read(path)
                    assert data['summary'] == s and s['alpha'] == 1. and s['reward_scale'] == SCALES[task]
                    old = read(ART / 'results/paper_h_credit_2026-09-24' / name / path.name)['summary']
                    assert s['picks'] == old['picks']
                    np.testing.assert_allclose(s['q1'], old['q1'], rtol=1e-6, atol=1e-6)
                    np.testing.assert_allclose(s['q2'], old['q2'], rtol=1e-6, atol=1e-6)
                    original = read(SOURCE / 'evaluations/validation' / task / (method + '_s%d' % seed) / ('trace_%02d.json' % case_id))
                    picks = s['picks']
                    assert set(data['branches']) == set(str(h) for h in picks.values())
                    reference_obs = None
                    prefix_steps += anchor
                    for h, branches in data['branches'].items():
                        assert len(branches) == REPEATS
                        for r, b in enumerate(branches):
                            assert b['noise_seed'] == s['shared_noise_seeds'][r] == 26092490 + case_id * 100 + anchor + r
                            if reference_obs is None:
                                reference_obs = b['initial_observation']
                            assert b['initial_observation'] == reference_obs
                            inference_samples.extend(audit_branch(task, cases[case_id], original, anchor, int(h), r, b))
                            assert s['branches'][h][r] == {k: v for k, v in b.items() if k != 'trace'}
                            branch_count += 1
                            suffix_steps += b['steps']
                            prefix_steps += anchor
                            outcome = outcomes[task]
                            outcome['rollouts'] += 1
                            outcome['goals'] += int(b['termination'] == 'goal')
                            outcome['constraints'] += int(b['termination'] == 'constraint')
                            outcome['time_limits'] += int(b['termination'] == 'steps')
                            outcome['solver_failed_steps'] += b['solver_failure_steps']
                            if b['total_cost'] > outcome['maximum_cost']:
                                outcome['maximum_cost'] = b['total_cost']
                                outcome['maximum_cost_key'] = [name, case_id, anchor, int(h), r]
                            if b['termination'] == 'steps':
                                tails.append(b)
                    initial = np.asarray(reference_obs, dtype=np.float32)
                    centers = np.linspace(-1., 1., 50, dtype=np.float32).reshape(-1, 1)
                    q1, q2 = model.sess.run([model.step_ops[4], model.step_ops[5]], {
                        model.observations_ph: np.repeat(initial[None, :], 50, axis=0), model.actions_ph: centers})
                    np.testing.assert_allclose(q1.ravel(), s['q1'], atol=1e-6, rtol=1e-6)
                    np.testing.assert_allclose(q2.ravel(), s['q2'], atol=1e-6, rtol=1e-6)
                    actor_h = int(np.clip(np.rint(model.predict(initial, deterministic=True)[0][0]), 1, 50))
                    assert actor_h == picks['actor']
                    assert picks['q1'] == int(np.argmax(q1)) + 1
                    assert picks['min_q'] == int(np.argmax(np.minimum(q1, q2))) + 1
                    actor = data['branches'][str(picks['actor'])]
                    for arm in ('q1', 'min_q', 'fixed'):
                        other = data['branches'][str(picks[arm])]
                        deltas = {key: [b[key] - a[key] for a, b in zip(actor, other)]
                                  for key in ('finite_soft_return', 'soft_return_with_target_tail',
                                              'entropy_contribution', 'discounted_target_v_tail', 'discounted_cost')}
                        row = {'task': task, 'method': method, 'seed': seed, 'case': case_id,
                               'anchor': anchor, 'arm': arm, 'actor_h': picks['actor'], 'other_h': picks[arm],
                               'deltas_other_minus_actor': deltas,
                               'means': {key: mean(v) for key, v in deltas.items()},
                               'q1_predicted_delta': s['q1'][picks[arm] - 1] - s['q1'][picks['actor'] - 1],
                               'min_q_predicted_delta': min(s['q1'][picks[arm] - 1], s['q2'][picks[arm] - 1]) - min(s['q1'][picks['actor'] - 1], s['q2'][picks['actor'] - 1]),
                               'actor_solver_failures': [b['solver_failure_steps'] for b in actor],
                               'other_solver_failures': [b['solver_failure_steps'] for b in other],
                               'actor_terminations': [b['termination'] for b in actor],
                               'other_terminations': [b['termination'] for b in other]}
                        contrasts.append(row)
                        model_contrasts.append(row)
                verify_inference(model, inference_samples, tails)
                target_queries += len(tails)
                sampled_actions += len(inference_samples)
                assert weights_hash(model) == before
                model.sess.close()
                for arm in ('q1', 'min_q', 'fixed'):
                    cs = [c for c in model_contrasts if c['arm'] == arm]
                    rows.append({'model': name, 'arm': arm,
                                 'different_h': sum(c['other_h'] != c['actor_h'] for c in cs),
                                 'mean_finite_delta': mean(c['means']['finite_soft_return'] for c in cs),
                                 'mean_tail_delta': mean(c['means']['soft_return_with_target_tail'] for c in cs),
                                 'finite_positive_anchors': sum(c['means']['finite_soft_return'] > 1e-6 for c in cs),
                                 'tail_positive_anchors': sum(c['means']['soft_return_with_target_tail'] > 1e-6 for c in cs),
                                 'all_repeats_negative_finite_and_tail': sum(
                                     all(x < -1e-6 for x in c['deltas_other_minus_actor']['finite_soft_return']) and
                                     all(x < -1e-6 for x in c['deltas_other_minus_actor']['soft_return_with_target_tail']) for c in cs)})
                print(json.dumps({'audited': name, 'sampled_actions': len(inference_samples), 'target_values': len(tails)}), flush=True)
    assert len(contrasts) == 144 and len(rows) == 36
    budget = {'models': 12, 'anchors': 48, 'repeats': REPEATS, 'logical_policy_conditions': 48 * 4 * REPEATS,
              'actual_rollouts': branch_count, 'suffix_steps': suffix_steps, 'replayed_prefix_steps': prefix_steps,
              'explicit_reset_warmup_steps': 48 + branch_count, 'training_steps': 0,
              'budget_scope': 'Explicit rollout/prefix resets and retained simulations. Environment-construction overhead and audit graph inference are separate and not converted to H-proxy savings.',
              'inference_samples_verified': sampled_actions, 'target_values_verified': target_queries}
    write(OUT / 'audit.json', {'passed': True, 'budget': budget, 'trace_hashes': hashes,
                              'input_hash': input_hash, 'audit_script_sha256': digest(Path(__file__)),
                              'rows': rows, 'contrasts': contrasts, 'outcomes': outcomes})
    md = ['# Paper H Soft-Return Diagnosis', '',
          'Validation-only post-hoc diagnosis. Higher soft return is better. Each difference is alternative minus actor; first action has no entropy term.', '',
          '| Model | First-H alternative | Different H /4 | Finite mean delta | With tail mean delta | Positive finite /4 | Positive with tail /4 | All four repeats negative in both /4 |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        md.append('| {model} | {arm} | {different_h} | {mean_finite_delta:.4f} | {mean_tail_delta:.4f} | {finite_positive_anchors} | {tail_positive_anchors} | {all_repeats_negative_finite_and_tail} |'.format(**r))
    md += ['', 'Audit budget: `' + json.dumps(budget, sort_keys=True) + '`.', '',
           'All retained outcomes: `' + json.dumps(outcomes, sort_keys=True) + '`.', '',
           'The audit recomputed physical costs, constraints, entropy and discounted returns; verified paired starts, standardized noise, rounded actions, model inputs and all saved actor/target-V inferences. No test scenes or training were used.', '',
           'Four shared-noise continuations are a small descriptive Monte Carlo sample, not proof of expected soft-Q ordering. The target-V tail is learned, not independent truth. Q values are queried at integer-H bin centers. Multiple anchors and policies share two scenarios per task and cannot be counted as independent replications. Q-inference disagreement cannot alone establish actor optimization failure.']
    (OUT / 'report.md').write_text('\n'.join(md) + '\n')
    print(json.dumps({'passed': True, 'budget': budget}), flush=True)


if __name__ == '__main__':
    main()
