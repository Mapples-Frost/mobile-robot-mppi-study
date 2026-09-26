"""Independent trace, frozen-parameter, regression and evaluation gate audit."""
import argparse
import copy
import math
from pathlib import Path
import numpy as np

from branch_calibration_protocol import (OUT, TASKS, HORIZONS, ANCHORS, REPEATS,
    COUNTS, ARMS, BASE_H, bank_path, protocol, verify, model_dir)
from paper_h_soft_probe import read, digest, GAMMA
from paper_h_soft_report import close, audit_branch, verify_inference
from paper_grid_audit_report import physics
from runtime import ART, ROOT, imports, make_env
from run import weights_hash, write

Q_PREFIXES = ('model/values_fn/qf1/', 'model/values_fn/qf2/')


def verify_artifact_hashes(path):
    data = read(path)
    assert data.get('passed', data.get('frozen', False))
    assert data['hashes']
    for filename, expected in data['hashes'].items():
        assert digest(Path(filename)) == expected, filename
    return data


def observation(task, state, case, before_step):
    # reset has already applied one warmup; after step t, TVP index is t+2.
    variables = read(ART / 'configs' / (task + '.json'))['environment']['observation']['variables']
    values = np.asarray([state[v['name']] if v['type'] == 'state'
                         else case['tvp'][v['name']][before_step + 1]['true'][0]
                         for v in variables], dtype=float)
    if task == 'vehicle':
        xy = values[:2].copy()
        values[3:5] = (values[3:5] - xy) / 5.
        for j in range(3): values[5+3*j:7+3*j] = (values[5+3*j:7+3*j] - xy) / 10.
        values[:2] /= 30.
        values[2] /= np.pi
    else:
        values /= np.array([1.5, 5., np.pi/2, 10., 1.])
    return values


def audit_trace(task, case, trace, offset=0):
    limit = 150 if task == 'vehicle' else 100
    assert trace and len(trace) + offset <= limit
    for k, row in enumerate(trace):
        t = offset + k
        np.testing.assert_allclose(row['observation'], observation(task, row['previous_state'], case, t), atol=1e-12)
        np.testing.assert_allclose(row['next_observation'], observation(task, row['state'], case, t+1), atol=1e-12)
        if k: assert row['previous_state'] == trace[k-1]['state']
        cost, violation, excess = physics(task, row, case, t)
        close(cost, row['performance'])
        close(row['compute'], row['horizon'] * (.001 if task == 'vehicle' else .003))
        close(row['constraint'], (2 if task == 'vehicle' else 10) * (limit-t-1) if violation else 0.)
        close(-row['reward'], cost + row['compute'] + row['constraint'])
        assert excess <= 1e-5 and row['horizon'] in range(1, 51)
        goal = False
        if task == 'vehicle':
            end = case['reference']['traj_steps'] - 1
            goal = math.hypot(row['state']['x'] - case['tvp']['trajectory_x'][end]['true'][0],
                              row['state']['y'] - case['tvp']['trajectory_y'][end]['true'][0]) <= .5
        if k < len(trace)-1:
            assert not violation and not goal and row['termination'] is None
        else:
            term = row['termination']
            assert term in ('goal', 'constraint', 'steps')
            if term == 'constraint': assert violation
            elif term == 'goal': assert goal and not violation
            else: assert t+1 == limit and not violation
    return {'steps': len(trace), 'total_cost': sum(-r['reward'] for r in trace),
            'solver_failure_steps': sum(not r['solver_success'] for r in trace)}


def actor_choices(model, obs):
    return np.clip(np.rint(model.predict(np.asarray(obs), deterministic=True)[0]).ravel(), 1, 50).astype(int)


def q_values(model, obs, hs):
    return model.sess.run(model.step_ops[4:6], {
        model.observations_ph: np.repeat(np.asarray(obs)[None, :], len(hs), axis=0),
        model.actions_ph: np.asarray([[(h-1)/24.5-1] for h in hs], np.float32)})


def centered_loss(model, groups):
    losses = []
    for g in groups:
        target = np.asarray(g['targets']) - np.mean(g['targets'])
        for q in q_values(model, g['observation'], g['horizons']):
            delta = abs(q.ravel() - np.mean(q) - target)
            losses.append(float(np.mean(np.where(delta <= 1., .5*delta**2, delta-.5))))
    return float(np.mean(losses))


def audit_fit(task, seed, smoke=False, replay=False):
    verify()
    folder = OUT / ('smoke' if smoke else '%s_s%d' % (task, seed))
    done, fit, data = [read(folder / (n + '.json')) for n in ('completed', 'fit', 'dataset')]
    assert done['task'] == task and done['seed'] == seed and done['smoke'] == smoke
    assert done['input_hash'] == digest(OUT / 'inputs_sha256.json')
    assert done['fit_hash'] == digest(folder / 'fit.json')
    assert done['dataset_hash'] == fit['dataset_hash'] == digest(folder / 'dataset.json')
    for path, expected in data['hashes'].items(): assert digest(Path(path)) == expected
    cases = read(folder / 'smoke_bank.json' if smoke else bank_path(task, 'train', seed))['cases']
    assert len(cases) == (1 if smoke else COUNTS['train'])
    if not smoke: assert data['bank_hash'] == digest(bank_path(task, 'train', seed))
    _, SAC, _ = imports()
    source = SAC.load(str(model_dir(task, 'min_q', seed) / 'model.zip'))
    fitted = SAC.load(str(folder / 'model.zip'))
    assert source.gamma == fitted.gamma == GAMMA and source.ent_coef == fitted.ent_coef == 1.
    assert source.time_aware and fitted.time_aware
    assert weights_hash(source) == data['original_hash'] == read(model_dir(task, 'min_q', seed) / 'completed.json')['final_hash']
    assert weights_hash(fitted) == done['model_hash'] == fit['final_hash']
    old, new = source.get_parameters(), fitted.get_parameters()
    assert set(old) == set(new)
    changed = [k for k in old if not np.array_equal(old[k], new[k])]
    assert set(changed) == set(fit['changed_parameters']) and changed
    assert all(k.startswith(Q_PREFIXES) for k in changed)
    assert fit['steps'] == (10 if smoke else protocol()['fit']['steps'])
    groups = {(g['case'], g['anchor']): g for g in data['groups']}
    assert len(groups) == len(data['groups']) == fit['groups'] >= (1 if smoke else 4)
    expected_groups, expected_skips = set(), []
    budget = dict(source_steps=0, prefix_steps=0, suffix_steps=0, resets=0, rollouts=0, skipped_anchors=0)
    samples, tails = [], []
    if replay:
        from branch_calibration_run import meter, observed_step
        env = make_env(task, seed, aligned=True, scaled_obs=True)
        replay_dir = folder / 'audit_replay'
        replay_dir.mkdir(exist_ok=True)
        meter(env, replay_dir)
        env.set_value_function_weights_and_biases(*source.policy_tf.get_mpc_vfn_weights_and_biases())
    for c, case in enumerate(cases):
        original = read(folder / ('source_%02d.json' % c))
        audit_trace(task, case, original)
        np.testing.assert_array_equal(actor_choices(source, [r['observation'] for r in original]), [r['horizon'] for r in original])
        if replay:
            env.reset(**copy.deepcopy(case))
            for t, row in enumerate(original):
                _, _, actual = observed_step(env, task, row['horizon'], case, t)
                assert actual == row, (task, seed, c, t, 'source replay')
        budget['source_steps'] += len(original)
        budget['resets'] += 1
        for anchor in ((20,) if smoke else ANCHORS):
            if anchor >= len(original):
                expected_skips.append({'case': c, 'anchor': anchor, 'steps': len(original), 'termination': original[-1]['termination']})
                budget['skipped_anchors'] += 1
                continue
            expected_groups.add((c, anchor))
            g = groups[c, anchor]
            path = folder / ('case%02d_t%03d.json' % (c, anchor))
            bdata = read(path)
            assert g['source_hash'] == digest(path)
            assert g['observation'] == bdata['observation'] == original[anchor]['observation']
            hs = sorted(set(((10, 25) if smoke else HORIZONS) + (original[anchor]['horizon'],)))
            assert g['horizons'] == bdata['candidates'] == hs
            assert set(bdata['branches']) == set(map(str, hs))
            labels = []
            budget['prefix_steps'] += anchor
            budget['resets'] += 1
            for h in hs:
                runs = bdata['branches'][str(h)]
                assert len(runs) == REPEATS
                for r, b in enumerate(runs):
                    assert b['noise_seed'] == 2609250000 + TASKS.index(task)*100000 + seed*10000 + c*100 + anchor + r
                    assert b['initial_observation'] == b['trace'][0]['observation'] == g['observation']
                    assert b['final_observation'] == b['trace'][-1]['next_observation']
                    audit_trace(task, case, b['trace'], anchor)
                    for row in b['trace'][1:]: assert row['sampling']['observation'] == row['observation']
                    samples.extend(audit_branch(task, case, original, anchor, h, r, b))
                    if b['termination'] == 'steps': tails.append(b)
                    budget['prefix_steps'] += anchor
                    budget['resets'] += 1
                    budget['suffix_steps'] += b['steps']
                    budget['rollouts'] += 1
                labels.append(np.mean([b['soft_return_with_target_tail'] for b in runs]))
            np.testing.assert_allclose(labels, g['targets'], atol=1e-10)
    assert set(groups) == expected_groups and data['skipped'] == expected_skips
    assert data['budget'] == budget
    verify_inference(source, samples, tails)
    for model, key in ((source, 'initial_loss'), (fitted, 'final_loss')):
        np.testing.assert_allclose(centered_loss(model, data['groups']), fit[key], atol=2e-4, rtol=2e-5)
    assert fit['labels'] == sum(len(g['horizons']) for g in groups.values())
    source.sess.close(); fitted.sess.close()
    hashes = {str(p): digest(p) for p in folder.glob('*.json') if not p.name.startswith('attempt_')}
    hashes[str(folder / 'model.zip')] = digest(folder / 'model.zip')
    attempts = [read(p) for p in folder.glob('attempt_*.json')]
    attempted = {k: sum(a[k] for a in attempts) for k in ('reset_calls', 'step_calls')}
    assert attempted['step_calls'] >= budget['source_steps'] + budget['prefix_steps'] + budget['suffix_steps']
    assert attempted['reset_calls'] >= budget['resets']
    return {'task': task, 'seed': seed, 'smoke': smoke, 'passed': True, 'budget': budget,
            'attempted_calls': attempted, 'source_replay': replay, 'initial_loss': fit['initial_loss'],
            'final_loss': fit['final_loss'], 'hashes': hashes}


def audit_evaluation(task, seed, arm, split):
    folder = OUT / 'evaluations' / split / task / ('%s_s%d' % (arm, seed))
    done = verify_artifact_hashes(folder / 'completed.json')
    assert (done['task'], done['seed'], done['arm'], done['split']) == (task, seed, arm, split)
    bank = bank_path(task, split)
    assert done['bank_hash'] == digest(bank)
    source = OUT / ('%s_s%d' % (task, seed)) if arm == 'calibrated_greedy' else model_dir(task, 'fixed' if arm == 'fixed' else 'min_q', seed)
    assert done['model_path'] == str(source / 'model.zip')
    assert done['model_file_hash'] == digest(source / 'model.zip')
    _, SAC, _ = imports()
    model = SAC.load(str(source / 'model.zip'))
    assert done['model_hash'] == weights_hash(model)
    cases = read(bank)['cases']
    summary = read(folder / 'summary.json')
    assert len(cases) == len(summary['episodes']) == COUNTS[split] and summary['terminal_value']
    assert len(list(folder.glob('trace_*.json'))) == len(cases)
    scene_costs, total_steps = [], 0
    for j, (case, e) in enumerate(zip(cases, summary['episodes'])):
        trace = read(folder / ('trace_%02d.json' % j))
        stats = audit_trace(task, case, trace)
        assert e['episode'] == j and e['termination'] == trace[-1]['termination']
        for key, val in stats.items(): close(e[key], val)
        for rk, ek in [('performance', 'performance_cost'), ('compute', 'computation_cost'), ('constraint', 'constraint_cost')]:
            close(e[ek], sum(r[rk] for r in trace))
        close(e['mean_horizon'], np.mean([r['horizon'] for r in trace]))
        choices = actor_choices(model, [r['observation'] for r in trace])
        for h_actor, row in zip(choices, trace):
            if arm == 'fixed': expected = BASE_H[task]
            elif arm == 'actor': expected = h_actor
            else:
                hs = sorted(set(HORIZONS + (int(h_actor),)))
                q1, q2 = q_values(model, row['observation'], hs)
                expected = hs[int(np.argmax(np.minimum(q1, q2)))]
            assert row['horizon'] == expected
        scene_costs.append(stats['total_cost']); total_steps += len(trace)
    model.sess.close()
    close(summary['mean_total_cost'], np.mean(scene_costs))
    assert summary['constraint_episodes'] == sum(e['termination'] == 'constraint' for e in summary['episodes'])
    assert summary['goal_episodes'] == sum(e['termination'] == 'goal' for e in summary['episodes'])
    return {'task': task, 'seed': seed, 'arm': arm, 'cost': float(np.mean(scene_costs)),
            'scene_costs': scene_costs, 'steps': total_steps,
            'constraints': summary['constraint_episodes'], 'goals': summary['goal_episodes'],
            'solver_failures': sum(e['solver_failure_steps'] for e in summary['episodes']),
            'hashes': dict(done['hashes'], **{str(folder / 'completed.json'): digest(folder / 'completed.json')})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('smoke', 'training', 'validation', 'test'))
    ap.add_argument('--replay', action='store_true')
    args = ap.parse_args()
    verify()
    if args.phase in ('smoke', 'training'):
        jobs = [('vehicle', 0)] if args.phase == 'smoke' else [(t, s) for t in TASKS for s in range(3)]
        rows = []
        for t, s in jobs:
            rows.append(audit_fit(t, s, args.phase == 'smoke', args.replay))
            print('%s s%d fit audited' % (t, s), flush=True)
        hashes = {p: h for row in rows for p, h in row['hashes'].items()}
        write(OUT / (args.phase + '_audit.json'), {'passed': True, 'rows': rows, 'hashes': hashes})
        return
    verify_artifact_hashes(OUT / 'training_audit.json')
    rows = [audit_evaluation(t, s, a, args.phase) for t in TASKS for s in range(3) for a in ARMS]
    checks, differences = {}, []
    for t in TASKS:
        lookup = {(r['seed'], r['arm']): r for r in rows if r['task'] == t}
        safety = True
        for s in range(3):
            c = lookup[s, 'calibrated_greedy']
            for a in ('actor', 'raw_greedy', 'fixed'):
                other = lookup[s, a]
                differences.append({'task': t, 'seed': s, 'comparator': a,
                    'mean_cost_difference': c['cost'] - other['cost'],
                    'paired_scene_differences': [x-y for x,y in zip(c['scene_costs'], other['scene_costs'])]})
                if args.phase == 'test' or a != 'fixed':
                    safety &= c['constraints'] <= other['constraints'] and c['solver_failures'] <= other['solver_failures']
        checks[t + '_safety'] = bool(safety)
        for a in ('actor', 'raw_greedy'):
            wins = sum(lookup[s, 'calibrated_greedy']['cost'] < lookup[s, a]['cost'] for s in range(3))
            checks[t + '_beats_' + a] = wins >= (3 if args.phase == 'test' else 2)
        checks[t + '_beats_fixed'] = (all(lookup[s, 'calibrated_greedy']['cost'] < lookup[s, 'fixed']['cost'] for s in range(3))
            if args.phase == 'test' else np.mean([lookup[s, 'calibrated_greedy']['cost'] for s in range(3)]) < np.mean([lookup[s, 'fixed']['cost'] for s in range(3)]))
    hashes = {p: h for row in rows for p,h in row['hashes'].items()}
    hashes[str(OUT / 'training_audit.json')] = digest(OUT / 'training_audit.json')
    result = {'passed': all(checks.values()), 'data_audit_passed': True,
              'checks': {k: bool(v) for k,v in checks.items()}, 'rows': rows, 'differences': differences, 'hashes': hashes}
    write(OUT / (args.phase + '_gate.json'), result)
    lines = ['# Bøhn 2021 分支回报校准扩展', '',
        '数据审计通过；%s gate：%s。此为方法扩展，H 成本是代理，不代表实测加速。' % (args.phase, result['passed']), '',
        '|任务|种子|方法|成本|约束回合|到达回合|求解失败步|', '|---|---:|---|---:|---:|---:|---:|']
    for r in rows: lines.append('|%s|%d|%s|%.6f|%d|%d|%d|' % (r['task'], r['seed'], r['arm'], r['cost'], r['constraints'], r['goals'], r['solver_failures']))
    lines += ['', '完整逐场景配对差、检查项和输入散列保存在相应 gate.json。',
        '固定 H 继承 300,000 搜索训练步和 60,000 补充种子训练步；author RL/min-Q 各 90,000 步。校准的额外仿真和拟合预算见 training_audit.json，另保留 smoke 和全部 attempt 记录。',
        '仅三个训练种子；场景和分支噪声不是独立训练重复。两次续跑的标签含学习 target-V 尾项；冻结原 actor 的回报不保证重复贪心部署有效。',
        '数据审计通过不等于核心结论复现。验证不通过时禁止开启测试；Goal 保持 active。']
    (ROOT / 'docs/reports/bohn2021_branch_calibration_2026-09-24.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print({k: v for k,v in result.items() if k not in ('rows','differences','hashes')}, flush=True)


if __name__ == '__main__': main()
