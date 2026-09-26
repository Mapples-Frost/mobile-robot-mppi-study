"""Reconstruct whole-episode ranks, CE evolution and frozen training selection.

Independent rank and probability formulas; no production rank/update imports.
Uses no simulator, no validation or test outcomes.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from latency_tree_protocol import OUT, REG, TASKS, read, write, sha, verify
from latency_tree_policy import HS, BASE, FEATURES, initialize, sample, policy_key


def equal(a,b):
    if isinstance(b,dict):
        assert a.keys()==b.keys()
        for key in b:equal(a[key],b[key])
    elif isinstance(b,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(b,float):
        assert math.isfinite(a) and math.isfinite(b)
        assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(a,b)
    else:assert a==b,(a,b)


def independent_rank(episodes,reference):
    assert [x['case'] for x in episodes]==[x['case'] for x in reference]
    n=len(episodes);a_steps=sum(x['steps'] for x in episodes);b_steps=sum(x['steps'] for x in reference)
    means=lambda xs,k:math.fsum(x[k] for x in xs)/n
    b_cost=means(reference,'total_cost');b_physical=means(reference,'physical_constraint_cost')
    assert abs(b_cost)>1e-12 and abs(b_physical)>1e-12
    cost=(means(episodes,'total_cost')-b_cost)/abs(b_cost)
    physical=(means(episodes,'physical_constraint_cost')-b_physical)/abs(b_physical)
    ratio=(math.fsum(x['decision_total_s'] for x in episodes)/a_steps)/(math.fsum(x['decision_total_s'] for x in reference)/b_steps)
    violations=sum(int(x['success']<y['success'])+int(x['constraint']>y['constraint'])+
                   int(x['physical_constraint_cost']>y['physical_constraint_cost']+.05*abs(y['physical_constraint_cost'])+1e-8)
                   for x,y in zip(episodes,reference))
    for key in ('initial_failed_steps','solver_failure_steps'):
        violations+=int(sum(x[key] for x in episodes)*b_steps>sum(x[key] for x in reference)*a_steps)
    violations+=int(cost>.02)+int(physical>.02)
    return dict(eligible=violations==0,violations=violations,objective=cost+.5*(ratio-1.),
                cost_change=cost,physical_change=physical,time_ratio=ratio)


def probability_update(old,elites):
    new=copy.deepcopy(old)
    def change(p,draws):
        if not draws:return list(p)
        mixed=[.9*(.4*value+.6*draws.count(i)/len(draws))+.1/len(p) for i,value in enumerate(p)]
        return [v/sum(mixed) for v in mixed]
    for node in range(3):
        new['feature'][node]=change(old['feature'][node],[e['nodes'][node][0] for e in elites])
        for f in range(len(old['feature'][node])):
            new['cut'][node][f]=change(old['cut'][node][f],[e['nodes'][node][1] for e in elites if e['nodes'][node][0]==f])
    for leaf in range(4):new['action'][leaf]=change(old['action'][leaf],[e['actions'][leaf] for e in elites])
    return new


def synthetic():
    from latency_tree_policy import rank,update
    rng=np.random.RandomState(61);base=[]
    for i in range(5):
        base.append(dict(case=i,steps=100,success=True,constraint=False,initial_failed_steps=8,solver_failure_steps=4,
            total_cost=10.+i,physical_constraint_cost=5.+i,decision_total_s=8.+i))
    checked=0
    for j in range(40):
        candidate=copy.deepcopy(base)
        for row in candidate:
            row['total_cost']+=float(rng.normal());row['physical_constraint_cost']+=float(rng.normal())
            row['steps']=int(rng.randint(50,121));row['initial_failed_steps']=int(rng.randint(0,13))
            row['solver_failure_steps']=int(rng.randint(0,8));row['decision_total_s']*=float(rng.uniform(.6,1.4))
        equal(rank(candidate,base),independent_rank(candidate,base));checked+=1
    for task in TASKS:
        cuts=[[0.,1.] for f in FEATURES[task]];dist=initialize(task,cuts)
        for j in range(5):
            elites=[sample(task,cuts,dist,rng)[1] for i in range(3)]
            equal(update(dist,elites),probability_update(dist,elites));dist=update(dist,elites);checked+=1
    result=dict(passed=True,checks=checked,source_sha256=sha(Path(__file__)),simulations=0,test_accessed=False)
    if REG.exists():
        verify();target=OUT/'learning_checks.json'
        if target.exists():assert read(target)==result
        else:write(target,result)
    print(json.dumps(result,indent=2))


def main():
    verify();checks=read(OUT/'learning_checks.json');assert checks['passed'] and checks['source_sha256']==sha(Path(__file__))
    audit=read(OUT/'audit_train.json');assert audit['passed']
    for p,h in audit['hashes'].items():assert sha(Path(p))==h
    hashes={str(OUT/'audit_train.json'):sha(OUT/'audit_train.json'),str(Path(__file__).resolve()):sha(Path(__file__)),str(OUT/'learning_checks.json'):sha(OUT/'learning_checks.json')};rows=[]
    for task in TASKS:
        for seed in range(3):
            folder=OUT/'train'/('%s_s%d'%(task,seed));done=read(folder/'completed.json');assert done['passed']
            for p,h in done['hashes'].items():assert sha(Path(p))==h
            hashes[str(folder/'completed.json')]=sha(folder/'completed.json')
            saved_cuts=read(folder/'thresholds.json');data=[]
            for cid in range(12):data.extend(r['tree_features'] for r in read(folder/'threshold_reference'/('r0_trace_%02d.json'%cid)))
            array=np.asarray(data);cuts=[]
            for col,name in enumerate(FEATURES[task]):
                cuts.append([.5] if name.startswith('previous_') else sorted(set(float(x) for x in np.quantile(array[:,col],[.1,.25,.5,.75,.9]))))
            equal(cuts,saved_cuts['cuts']);distribution=initialize(task,cuts)
            rng=np.random.RandomState(2609265000+TASKS.index(task)*100+seed);records=[]
            for generation in range(4):
                dest=folder/('generation%d'%generation);reg=read(dest/'registration.json');equal(reg['distribution'],distribution)
                expected_order=np.random.RandomState(2609266000+TASKS.index(task)*1000+seed*10+generation).permutation(13).tolist()
                assert reg['order']==expected_order
                generated=[]
                for i in range(12):
                    p,g=sample(task,cuts,distribution,rng);generated.append(dict(id='g%d_c%02d'%(generation,i),policy=p,genome=g))
                assert generated==reg['proposals']
                baseline=read(dest/'fixed/summary.json')['episodes'];current=[]
                for proposal in generated:
                    summary=read(dest/proposal['id']/'summary.json');assert summary['policy']==proposal['policy']
                    score=independent_rank(summary['episodes'],baseline)
                    current.append(dict(proposal,rank=score,folder=str(dest/proposal['id'])))
                result=read(dest/'selection.json');equal(result['all_candidates'],current);records.extend(current)
                ordered=sorted(current,key=lambda r:(r['rank']['violations'],r['rank']['objective'],r['id']))
                assert result['elite_ids']==[r['id'] for r in ordered[:3]]
                calculated=probability_update(distribution,[r['genome'] for r in ordered[:3]])
                equal(calculated,result['next_distribution'])
                # Replay RNG with the actual saved full-precision probability vector.
                distribution=result['next_distribution']
            eligible=sorted([r for r in records if r['rank']['eligible']],key=lambda r:(r['rank']['objective'],r['id']))
            finalists=[];seen=set()
            for item in eligible:
                key=policy_key(item['policy'])
                if key not in seen:finalists.append(item);seen.add(key)
                if len(finalists)==4:break
            reg=read(folder/'selection_registration.json');equal(reg['finalists'],finalists)
            expected_ids=['fixed']+[f['id'] for f in finalists]+(['certificate'] if task=='pendulum' else [])
            assert [r['id'] for r in reg['arms']]==expected_ids
            expected_order=[]
            for repeat in range(2):
                for i in np.random.RandomState(2609267000+TASKS.index(task)*1000+seed*10+repeat).permutation(len(expected_ids)):
                    expected_order.append(dict(repeat=repeat,arm=expected_ids[int(i)]))
            assert reg['order']==expected_order
            scores=[]
            for arm in reg['arms']:
                if arm['id']=='certificate':continue
                ranks=[independent_rank(read(folder/'selection'/('r%d_%s'%(r,arm['id']))/'summary.json')['episodes'],
                                        read(folder/'selection'/('r%d_fixed'%r)/'summary.json')['episodes']) for r in range(2)]
                scores.append(dict(id=arm['id'],policy=arm['policy'],eligible=all(r['eligible'] for r in ranks),objective=sum(r['objective'] for r in ranks)/2,ranks=ranks))
            fit=read(folder/'fit.json');equal(fit['all_candidates'],records);equal(fit['selection'],scores)
            winner=min([r for r in scores if r['eligible']],key=lambda r:(r['objective'],r['id']!='fixed',r['id']))
            assert fit['selected']==winner['id'] and read(folder/'policy.json')==winner['policy']
            rows.append(dict(task=task,seed=seed,all_candidates=48,eligible_candidates=len(eligible),finalists=len(finalists),selected=winner['id'],tree_selected=winner['policy']['kind']=='tree'))
    result=dict(passed=True,groups=rows,candidates=288,simulations=0,test_accessed=False,core_effect_evaluated=False,hashes=hashes)
    target=OUT/'learning_audit.json'
    if target.exists():assert read(target)==result
    else:write(target,result)
    print(json.dumps({k:v for k,v in result.items() if k!='hashes'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    if parser.parse_args().check:synthetic()
    else:main()
