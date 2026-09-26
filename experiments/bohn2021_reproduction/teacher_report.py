"""Independent physical/label audit and all-seed results for teacher-value pilot."""
import hashlib
import json
from pathlib import Path
import numpy as np
from teacher_common import OUT, ROOT, HS, STEPS, SCALE, GAMMA, OFFSET, BRANCH, read, write, verify, infer
from riccati_terminal_probe import prior


def feature_check(c):
    s, refs = c['state'], np.asarray(c['refs'])
    diff = refs[1:]-refs[0]
    changes = np.flatnonzero(abs(diff)>1e-8)
    first = changes[0] if len(changes) else None
    expected = [(s['pos']-refs[0])/1.5, s['v']/2, s['theta']/.5, s['omega']/3,
                s['pos']/1.5, refs[0]/1.5, (STEPS-c['elapsed'])/STEPS,
                (first+1)/50 if first is not None else 1.1,
                diff[first]/1.5 if first is not None else 0.]
    expected += (diff.reshape(10, 5).mean(1)/1.5).tolist()
    np.testing.assert_allclose(c['features'], expected, rtol=2e-7, atol=2e-7)


def teacher_h(c):
    s, refs = c['state'], c['refs']
    unsettled = (abs(s['pos']-refs[0])>.03 or abs(s['v'])>.08 or
                 abs(s['theta'])>.04 or abs(s['omega'])>.12)
    upcoming = any(abs(v-refs[0])>1e-9 for v in refs[1:31])
    return 30 if unsettled or upcoming else 5


def audit_trace(trace, initial, scene, branch_h=None, model=None, fixed=None):
    previous = initial
    for i, r in enumerate(trace):
        c, after = r['before'], r['after']
        assert c['state'] == previous['state']
        assert c['elapsed'] == initial['elapsed']+i
        assert c['clock'] == initial['clock']+i
        assert after['clock'] == c['clock']+1 and after['elapsed'] == c['elapsed']+1
        assert after['state'] == r['state']
        feature_check(c)
        feature_check(after)
        for context in [c, after]:
            for k, value in enumerate(context['refs']):
                clock = context['clock']+k
                if branch_h is not None and clock > initial['clock']+50:
                    expected = initial['refs'][-1]
                else:
                    expected = scene['case']['tvp']['pos_r'][clock]['true'][0]
                assert value == expected
        if branch_h is not None:
            h = branch_h if i == 0 else teacher_h(c)
        elif model is not None:
            h = HS[int(np.argmin(infer(model, c['features'])))]
        elif fixed is not None:
            h = fixed
        else:
            h = teacher_h(c)
        assert h == r['horizon']
        s, u, ref = r['state'], r['input']['u1'], after['refs'][0]
        assert abs(u) <= 5+1e-5
        v, omega, theta = s['v'], s['omega'], s['theta']
        physical = (.4*v*v + .05*v*omega*np.cos(theta) +
                    (2/3)*.2*.25**2*omega*omega - OFFSET*np.cos(theta) +
                    10*(s['pos']-ref)**2+.1*u*u)
        violation = abs(s['pos'])>1.5 or abs(theta)>np.pi/2
        remaining = STEPS-after['elapsed']
        penalty = 10*remaining if violation else 0.
        shaped = (physical+OFFSET+.003*h+penalty+(OFFSET*remaining if violation else 0))/SCALE
        assert physical+OFFSET >= -1e-9
        np.testing.assert_allclose([r['performance'], r['compute'], r['constraint'], r['cost'], r['shaped_cost']],
            [physical, .003*h, penalty, physical+.003*h+penalty, shaped], rtol=1e-9, atol=1e-8)
        if violation:
            assert i == len(trace)-1 and r['termination'] == 'constraint'
        previous = after


def aggregate(rows):
    keys = ['raw_cost', 'adjusted_cost', 'discounted_shaped_cost', 'physical_cost',
            'H_cost', 'constraint_cost', 'mean_H', 'tracking_rmse']
    return dict({k: float(np.mean([r[k] for r in rows])) for k in keys},
        episodes=len(rows), failures=sum(r['termination']=='constraint' for r in rows),
        solver_failures=sum(r['solver_failures'] for r in rows))


def main():
    verify()
    assert (OUT/'completed.json').exists()
    _, _, details = prior()
    P = np.asarray(details['P'])
    banks = {s: read(OUT/(s+'_bank.json'))['scenes'] for s in ['train', 'validation', 'aggregation', 'holdout']}
    selection = read(OUT/'selection.json')
    counts = {'label_anchors': {}, 'branch_rollouts': 0, 'branch_transitions': 0,
              'prefix_episodes': 0, 'prefix_transitions': 0, 'evaluation_episodes': 0,
              'evaluation_transitions': 0, 'smoke_extra_transitions': 61}
    hashes = {}
    starts = {}
    rows_by_split = {}
    # Audit every prefix/evaluation trajectory, including failures and duplicates.
    for category in ['labels', 'evaluation']:
        for path in sorted((OUT/category).glob('*/*/scene_*/trajectory.json')):
            split, arm, folder = path.parts[-4:-1]
            index = int(folder.split('_')[1])
            scene = banks[split][index]
            ep = read(path)
            model = read(OUT/'models'/(arm+'.json')) if arm.startswith('round') else None
            fixed = int(arm.split('_')[1]) if arm.startswith('fixed_') else None
            audit_trace(ep['trace'], ep['initial'], scene, model=model, fixed=fixed)
            key = (split, index)
            if key in starts:
                assert starts[key] == ep['initial']
            starts[key] = ep['initial']
            summary = read(path.parent/'completed.json')
            trace = ep['trace']
            raw = sum(r['cost'] for r in trace)
            shaped = sum(r['shaped_cost'] for r in trace)
            np.testing.assert_allclose([summary['raw_cost'], summary['adjusted_cost'],
                summary['discounted_shaped_cost'], summary['physical_cost'], summary['H_cost'], summary['constraint_cost']],
                [raw, shaped*SCALE, sum(GAMMA**i*r['shaped_cost'] for i,r in enumerate(trace)),
                 sum(r['performance']+OFFSET for r in trace), sum(r['compute'] for r in trace),
                 sum(r['constraint'] for r in trace)], atol=1e-7)
            assert np.isclose(raw+OFFSET*STEPS, shaped*SCALE)
            assert summary['steps'] == len(trace)
            assert len(trace)==STEPS or trace[-1]['termination']=='constraint'
            assert summary['solver_failures']==sum(not r['solver_success'] for r in trace)
            np.testing.assert_allclose([summary['mean_H'], summary['tracking_rmse']],
                [np.mean([r['horizon'] for r in trace]),
                 np.sqrt(np.mean([(r['state']['pos']-r['after']['refs'][0])**2 for r in trace]))], atol=1e-8)
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            name = 'prefix' if category=='labels' else 'evaluation'
            counts[name+'_episodes'] += 1
            counts[name+'_transitions'] += len(trace)
            if category=='evaluation':
                rows_by_split.setdefault(split, {}).setdefault(arm, []).append(summary)
            else:
                anchors = read(path.parent/'anchors.json')
                counts['label_anchors'][split] = counts['label_anchors'].get(split, 0)+len(anchors)
                assert [a['elapsed'] for a in anchors] == [t for t in scene['anchors'] if t < len(trace)]
                for a in anchors:
                    assert a['context'] == trace[a['elapsed']]['before']
                    costs=[]
                    for h in HS:
                        p = path.parent/('branch_t%03d_h%02d.json'%(a['elapsed'],h))
                        result=read(p)
                        assert result['initial']==a['context']
                        audit_trace(result['trace'], result['initial'], scene, branch_h=h)
                        tail_context=result['trace'][-1]['after']
                        s=tail_context['state']
                        z=np.array([s['omega'],s['pos']-tail_context['refs'][0],s['theta'],s['v']])
                        rem=STEPS-tail_context['elapsed']
                        tail=0. if result['trace'][-1]['termination'] else (z@P@z+.015*(1-GAMMA**rem)/(1-GAMMA))/SCALE
                        assert np.isclose(result['tail'],tail,atol=1e-8)
                        cost=sum(GAMMA**i*r['shaped_cost'] for i,r in enumerate(result['trace']))+GAMMA**len(result['trace'])*tail
                        assert np.isclose(result['discounted_cost'],cost,atol=1e-8)
                        costs.append(cost)
                        counts['branch_rollouts']+=1
                        counts['branch_transitions']+=len(result['trace'])
                        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
                    np.testing.assert_allclose(a['costs'],costs,atol=1e-8)
                    np.testing.assert_allclose(a['relative_costs'],np.asarray(costs)-costs[HS.index(30)],atol=1e-8)
    expected_validation={r['arm'] for r in selection['fixed_validation']} | set(selection['students']) | {'switch_5_30'}
    expected_holdout=set(selection['students']) | {'switch_5_30', selection['fixed'], 'fixed_30'}
    for split,expected in [('validation',expected_validation),('holdout',expected_holdout)]:
        assert set(rows_by_split[split])==expected
        for arm,rows in rows_by_split[split].items():
            assert sorted(r['scene'] for r in rows)==list(range(len(banks[split])))
    for arm,expected in selection['model_hashes'].items():
        assert hashlib.sha256((OUT/'models'/(arm+'.json')).read_bytes()).hexdigest()==expected
    fixed_candidates={arm:aggregate(rows)['adjusted_cost'] for arm,rows in rows_by_split['validation'].items() if arm.startswith('fixed_')}
    assert min(fixed_candidates,key=lambda a:(fixed_candidates[a],a))==selection['fixed']
    results={split:{arm:{'summary':aggregate(rows),'episodes':rows} for arm,rows in arms.items()} for split,arms in rows_by_split.items()}
    training={arm:read(OUT/'models'/(arm+'_metrics.json')) for arm in selection['students']}
    import torch
    for seed in range(3):
        initial_round = torch.load(OUT/'models'/('round0_s%d.pt'%seed), map_location='cpu', weights_only=False)
        augmented_round = torch.load(OUT/'models'/('round1_s%d.pt'%seed), map_location='cpu', weights_only=False)
        assert initial_round['updates'] == augmented_round['updates'] == 3000
        for name, tensor in initial_round['initial'].items():
            assert torch.equal(tensor,augmented_round['initial'][name]), (seed,name)
    # Recompute final exported model diagnostics from the actual labelled validation states.
    labels=[]
    for p in sorted((OUT/'labels/validation/switch_5_30').glob('scene_*/anchors.json')):
        labels.extend(read(p))
    for arm,report in training.items():
        model=read(OUT/'models'/(arm+'.json'))
        predictions=np.asarray([infer(model,a['context']['features']) for a in labels])
        y=np.asarray([a['relative_costs'] for a in labels])
        regret=y[np.arange(len(y)),predictions.argmin(1)]-y.min(1)
        np.testing.assert_allclose([np.mean(abs(predictions-y)),np.mean(regret)],
            [report['validation']['relative_cost_mae'],report['validation']['mean_regret']],atol=1e-5)
    phase_diagnostics = {}
    for arm in list(training)+['switch_5_30','fixed_30']:
        model = read(OUT/'models'/(arm+'.json')) if arm in training else None
        grouped = {}
        for a in labels:
            c = a['context']
            future_change = any(abs(v-c['refs'][0])>1e-9 for v in c['refs'][1:])
            phase = 'preview' if future_change else ('recovery' if teacher_h(c)==30 else 'settled')
            h = HS[int(np.argmin(infer(model,c['features'])))] if model else (30 if arm=='fixed_30' else teacher_h(c))
            regret = a['relative_costs'][HS.index(h)]-min(a['relative_costs'])
            grouped.setdefault(phase,[]).append(regret)
        phase_diagnostics[arm] = {p:{'anchors':len(v),'mean_regret':float(np.mean(v)),
                                    'max_regret':float(np.max(v))} for p,v in grouped.items()}
    data_coverage = {}
    for split, arm in [('train','switch_5_30'),('validation','switch_5_30'),('aggregation','round0_s0')]:
        records=[]
        for path in sorted((OUT/'labels'/split/arm).glob('scene_*/anchors.json')):
            records.extend(read(path))
        winners = [HS[int(np.argmin(a['relative_costs']))] for a in records]
        phases = []
        for a in records:
            c=a['context']
            phases.append('preview' if any(abs(v-c['refs'][0])>1e-9 for v in c['refs'][1:])
                          else ('recovery' if teacher_h(c)==30 else 'settled'))
        data_coverage[split]={'anchors':len(records),
            'teacher_best_H_counts':{str(h):winners.count(h) for h in HS},
            'phase_counts':{p:phases.count(p) for p in ['preview','recovery','settled']}}
    analysis={'results':results,'training':training,'selection':selection,'budget':counts,
              'validation_label_phase_diagnostics':phase_diagnostics,'data_coverage':data_coverage}
    counts['reset_warmup_steps'] = counts['prefix_episodes']+counts['evaluation_episodes']+1
    counts['physical_steps_including_reset_smoke'] = sum(counts[k] for k in
        ['branch_transitions','prefix_transitions','evaluation_transitions','smoke_extra_transitions','reset_warmup_steps'])
    write(OUT/'analysis.json',analysis)
    write(OUT/'audit.json',{'passed':True,'budget':counts,'hashes':hashes,
        'checks':['registered sources','physical reward and failure padding','same-state forks',
                  'no reference leakage beyond original preview','teacher continuation policy',
                  'approximate tail and discounted labels','exported model actions',
                  'matched starts','matched per-seed round initialization','all seeds/scenes','validation-only fixed selection'],
        'report_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    dest=OUT.parents[1]/'report/teacher_value'
    dest.mkdir(parents=True,exist_ok=True)
    lines=['# Teacher-data horizon-value pilot','',
        'All seeds and both supervised rounds are reported. Lower cost is better. No SAC updates or globally optimal teacher claim.','',
        'Reward = -(original physical cost + 0.4905 + H cost + failure cost + failure-only remaining offsets) / 0.6. Adjusted episode cost equals old raw cost + 294.3 for every episode. Discounted targets use gamma .97; full 600-step performance is also reported.','',
        'Branch targets use 60 simulated steps, the frozen H5/H30 continuation rule, constant extrapolation beyond the originally visible 50-step reference, and an approximate Riccati tail. This approximates a teacher policy value, not Q*.','',
        '## Holdout','', '| Arm | Adjusted cost | Physical cost | H cost | Mean H | Failures | Solver failures |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in sorted(results['holdout'].items()):
        s=r['summary']
        lines.append('| %s | %.4f | %.4f | %.4f | %.2f | %d/%d | %d |'%(arm,s['adjusted_cost'],s['physical_cost'],s['H_cost'],s['mean_H'],s['failures'],s['episodes'],s['solver_failures']))
    lines += ['', '## Critic validation on unseen labelled scenes','',
              '| Arm | Training anchors | Relative-cost MAE | Decision regret | Within .02 of teacher best |',
              '|---|---:|---:|---:|---:|']
    for arm,r in sorted(training.items()):
        v=r['validation']
        lines.append('| %s | %d | %.6f | %.6f | %.3f |'%(arm,r['training_anchors'],v['relative_cost_mae'],v['mean_regret'],v['near_optimal_fraction_002']))
    lines += ['', 'Teacher-label action regret by phase (not actual deployment cost):', '',
              '| Arm | Preview | Recovery | Settled |', '|---|---:|---:|---:|']
    for arm, groups in phase_diagnostics.items():
        lines.append('| %s | '%arm+' | '.join('%.6f'%groups[p]['mean_regret'] if p in groups else 'n/a'
                       for p in ['preview','recovery','settled'])+' |')
    lines += ['', '## Validation complete episodes','', '| Arm | Adjusted cost | Failures |','|---|---:|---:|']
    for arm,r in sorted(results['validation'].items()):
        s=r['summary']
        lines.append('| %s | %.4f | %d/%d |'%(arm,s['adjusted_cost'],s['failures'],s['episodes']))
    lines += ['', '## Budget and limitations','', '```json',json.dumps(counts,indent=2),'```','',
        'The fork integration smoke reused its first seven branches; only its repeated 60-step branch and one parent step are additional. The formal training-scene prefix is rerun after smoke. All physical simulation is counted, not just neural updates.','',
        'Architecture, scene distribution and supervised objective differ from previous SAC; this cannot isolate a data-only improvement. Three seeds are learning repeats; holdout scenes are paired, not independent learning seeds. Computation remains an H proxy. Physical Riccati tail is a local infinite-horizon approximation; only the computation tail is explicitly cut to remaining steps.','',
        '## Every paired holdout scene','', '| Scene | '+ ' | '.join(sorted(results['holdout']))+' |',
        '|---|'+'---:|'*len(results['holdout'])]
    for i in range(12):
        values=[next(r for r in results['holdout'][a]['episodes'] if r['scene']==i)['adjusted_cost'] for a in sorted(results['holdout'])]
        lines.append('| %d | '%i+' | '.join('%.4f'%v for v in values)+' |')
    lines += ['', '![All-seed teacher-data results](overview.png)', '']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    plot(results, training, selection, dest)
    print(json.dumps({'holdout':{a:r['summary'] for a,r in results['holdout'].items()},
                     'critic':{a:r['validation'] for a,r in training.items()},'budget':counts},indent=2))


def plot(results, training, selection, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors=['#4878A8','#C05B42','#57916B']
    fixed=results['holdout'][selection['fixed']]['summary']['adjusted_cost']
    rule=results['holdout']['switch_5_30']['summary']['adjusted_cost']
    for seed in range(3):
        costs=[results['holdout']['round%d_s%d'%(r,seed)]['summary']['adjusted_cost'] for r in [0,1]]
        axes[0,0].plot([0,1],costs,'o-',color=colors[seed],label='Seed %d'%seed)
        regret=[training['round%d_s%d'%(r,seed)]['validation']['mean_regret'] for r in [0,1]]
        axes[0,1].plot([0,1],regret,'o-',color=colors[seed],label='Seed %d'%seed)
    axes[0,0].axhline(fixed,color='0.25',ls='--',label=selection['fixed'])
    axes[0,0].axhline(rule,color='0.5',ls=':',label='H5/H30 rule')
    for ax in axes[0]:
        ax.set_xticks([0,1],['Initial teacher data','+ Student states'])
        ax.grid(alpha=.2)
    axes[0,0].set(ylabel='Offset-adjusted total cost',title='A. Holdout cost: all initialization seeds')
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(ylabel='Mean teacher-label decision regret',title='B. Unseen labelled validation states')
    axes[0,1].legend(fontsize=8)
    base=sorted(results['holdout'][selection['fixed']]['episodes'],key=lambda r:r['scene'])
    for seed in range(3):
        student=sorted(results['holdout']['round1_s%d'%seed]['episodes'],key=lambda r:r['scene'])
        axes[1,0].scatter(np.arange(12)+1,[a['adjusted_cost']-b['adjusted_cost'] for a,b in zip(base,student)],
                          color=colors[seed],label='Seed %d'%seed,s=24)
    axes[1,0].axhline(0,color='0.4',lw=1)
    axes[1,0].set(xlabel='Holdout scene',ylabel='Fixed cost minus student cost',title='C. Paired savings (positive favors student)')
    axes[1,0].legend(fontsize=8)
    # Prespecified scene0 and seed0 for visual mechanism, not selected by outcome.
    for arm,color in [('round1_s0',colors[0]),('switch_5_30','0.35')]:
        ep=read(OUT/'evaluation/holdout'/arm/'scene_00/trajectory.json')
        axes[1,1].step([r['before']['clock']*.04 for r in ep['trace']],
                       [r['horizon'] for r in ep['trace']],where='post',label=arm,color=color)
    axes[1,1].set(xlabel='Time (s)',ylabel='Executed horizon',ylim=(0,52),title='D. Prespecified holdout scene0 / seed0')
    axes[1,1].legend(fontsize=8)
    for ax in axes[1]:
        ax.grid(alpha=.2)
    fig.savefig(dest/'overview.png',dpi=170)
    fig.savefig(dest/'overview.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
