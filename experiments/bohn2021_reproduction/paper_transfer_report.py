"""Independent trace accounting and seed-level transfer gate."""
import argparse
import math
import sys
from pathlib import Path
from statistics import mean
from paper_transfer_common import (OUT, PRESERVE, TEACHER, LEARNED, DOMAINS,
                                   read, write, verify, metrics)


def close(a, b, tol=1e-7):
    assert math.isclose(a, b, abs_tol=tol, rel_tol=tol), (a, b)


def expected_obs(state, refs, t, T):
    p = state['pos']
    return [p/1.5, state['v']/5, state['theta']/(math.pi/2), state['omega']/10,
            refs[t+1]] + [(refs[t+1+k]-p)/1.5 for k in range(1, 51)] + [(T-t)/T]


def audit_split(split, starts):
    import numpy as np
    from sac_preserve_inference import HorizonPolicy, encode
    from teacher_common import HS, infer
    bank = read(OUT/(split+'_bank.json'))['domains']
    totals = dict(episodes=0, steps=0, reset_warmups=0)
    groups = {}
    for completed in sorted((OUT/'evaluation'/split).glob('*/*/completed.json')):
        folder = completed.parent
        domain, name = folder.parent.name, folder.name
        eps = read(completed)['episodes']
        assert [e['scene'] for e in eps] == list(range(len(bank[domain])))
        actor = HorizonPolicy(PRESERVE/'export'/(name+'.json')) if name in LEARNED else None
        teachers = [read(TEACHER/'models'/('round1_s%d.json' % s)) for s in range(3)] if name=='value_teacher' else None
        for e in eps:
            d = read(folder/('scene_%02d.json' % e['scene']))
            scene = bank[domain][e['scene']]
            T = scene['steps']
            refs = [v['true'][0] for v in scene['case']['tvp']['pos_r']]
            rows = d['trace']
            assert d['summary'] == e and rows and rows[-1]['done']
            key = (split, domain, e['scene'])
            if key in starts:
                assert starts[key] == d['initial'], key
            starts[key] = d['initial']
            previous = d['initial']
            for t, row in enumerate(rows, 1):
                assert row['step'] == t and row['obs'] == previous
                s = row['state']; u = row['input']['u1']; h = row['horizon']
                assert 1 <= h <= 50 and abs(u) <= 5+1e-5
                perf = (.4*s['v']**2 + .05*s['v']*s['omega']*math.cos(s['theta'])
                        + s['omega']**2/120 - .4905*math.cos(s['theta'])
                        + 10*(s['pos']-refs[t+1])**2 + .1*u*u)
                fail = abs(s['pos']) > 1.5 or abs(s['theta']) > math.pi/2
                penalty = 10*(T-t) if fail else 0
                close(row['performance'], perf)
                close(row['constraint'], penalty)
                close(row['compute'], .003*h)
                close(row['cost'], perf+penalty+.003*h)
                for x, y in zip(row['next_obs'], expected_obs(s, refs, t, T)):
                    close(x, y, 3e-7)
                assert row['done'] == (fail or t==T)
                assert (row['termination']=='constraint') == fail
                if t < len(rows):
                    assert not row['done']
                if actor:
                    assert actor.predict(row['obs']) == h
                elif teachers:
                    hs = HS[int(np.argmin(np.mean([infer(m, encode(row['obs'])) for m in teachers], axis=0)))]
                    assert hs == h
                elif name.startswith('fixed_'):
                    assert h == int(name.split('_')[1])
                previous = row['next_obs']
            recomputed = metrics(rows, T)
            for k, v in recomputed.items():
                close(e[k], v)
            close(e['adjusted_cost'], e['physical_cost']+e['H_cost']+e['failure_penalty']+e['unexecuted_offset'])
            totals['episodes'] += 1
            totals['steps'] += len(rows)
            totals['reset_warmups'] += 1
        groups.setdefault(domain, {})[name] = {
            'mean_cost': mean(e['adjusted_cost'] for e in eps),
            'mean_raw_cost': mean(e['raw_cost'] for e in eps),
            'physical_failures': sum(e['physical_failure'] for e in eps),
            'solver_failures': sum(e['solver_failures'] for e in eps),
            'steps': sum(e['steps'] for e in eps),
            'scene_costs': [e['adjusted_cost'] for e in eps],
            'mean_H': mean(e['mean_H'] for e in eps)}
    return groups, totals


def main(smoke=False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'icra_scene_design'))
    from sweep_guard import require_no_live_sweep
    require_no_live_sweep('paper transfer audit')
    verify()
    starts = {}
    if smoke:
        _, totals = audit_split('smoke', starts)
        assert totals['episodes'] == 6
        write(OUT/'smoke_audit.json', dict(passed=True, **totals))
        print('Smoke audited:', totals)
        return
    validation, vt = audit_split('validation', starts)
    holdout, ht = audit_split('holdout', starts)
    assert vt['episodes']==440 and ht['episodes']==720
    selected = read(OUT/'selection.json')
    for domain in DOMAINS:
        expected = set(LEARNED + ['rule_baseline', 'value_teacher', selected[domain]['name']])
        assert set(holdout[domain]) == expected
    contrasts = []
    for domain, g in holdout.items():
        for arm in ['free', 'rule', 'value']:
            for control in [selected[domain]['name'], 'rule_baseline', 'value_teacher']:
                seed_deltas = [g['%s_s%d' % (arm, s)]['mean_cost']-g[control]['mean_cost'] for s in range(3)]
                contrasts.append({'domain': domain, 'arm': arm, 'control': control,
                                  'seed_cost_deltas': seed_deltas, 'mean_delta': mean(seed_deltas)})
    domain = 'redraw_broad_100'
    g = holdout[domain]
    fixed = g[selected[domain]['name']]
    gates = []
    for s in range(3):
        value = g['value_s%d' % s]
        gates.append(value['mean_cost'] < fixed['mean_cost'] and
                     value['physical_failures'] <= fixed['physical_failures'] and
                     value['solver_failures']/value['steps'] <= fixed['solver_failures']/fixed['steps'])
    attempts = []
    for p in (OUT/'evaluation').glob('*/*/*/attempt_*.json'):
        a = read(p)
        if a['status'] != 'complete':
            attempts.append({'file': str(p.relative_to(OUT)), **a})
    smoke_totals = read(OUT/'smoke_audit.json')
    budget = {'formal': {'validation': vt, 'holdout': ht}, 'smoke': smoke_totals,
              'interrupted_attempts': attempts, 'new_training_updates': 0,
              'inherited_models': '9 SAC actors and 3 branch-value teachers; historical budget is separate.',
              'note': 'Interrupted attempts give lower bounds; unflushed physical work remains unknown.'}
    write(OUT/'analysis.json', {'validation': validation, 'holdout': holdout, 'contrasts': contrasts})
    gate = {'domain': domain, 'passed_by_seed': gates, 'passed': all(gates),
            'next': 'register_joint_terminal_study' if all(gates) else 'diagnose_failed_transfer_factor_then_register_retraining'}
    write(OUT/'gate.json', gate)
    write(OUT/'audit.json', dict(passed=True, budget=budget))
    lines = ['# Bøhn 论文任务迁移诊断', '',
             '冻结策略，无新增训练。原论文精确配置仍缺失；100步分布采用已有重建假设。', '',
             '|任务分布|方法|平均成本|物理失败|求解失败/步|', '|---|---|---:|---:|---:|']
    for domain in DOMAINS:
        for name, v in sorted(holdout[domain].items()):
            lines.append('|%s|%s|%.4f|%d|%d/%d|' % (domain, name, v['mean_cost'], v['physical_failures'], v['solver_failures'], v['steps']))
    lines += ['', '三个value种子的预登记迁移门槛：'+str(gates), '',
              '下一步：'+gate['next'], '',
              '训练种子是重复单位；场景为配对评价。不同回合长度的绝对成本不直接比较；成本平移依赖长度。',
              '固定H仅在对应验证集按可靠性优先规则选定。学习模型不挑seed、不挑检查点。所有失败保留。',
              '本轮保持解析终端与七H候选，尚未恢复作者联合终端学习，也不是原论文完整复现。', '']
    (OUT/'report.md').write_text('\n'.join(lines))
    write(OUT/'completed.json', {'formal_episodes': 1160, 'audit_passed': True, 'gate': gate})
    print(json_dump(gate))


def json_dump(value):
    import json
    return json.dumps(value)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    main(p.parse_args().smoke)
