"""Recompute every measured time/budget and verify replay independently of timing writer."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from gated_horizon_search import OUT
from gated_horizon_timing import register
from paper_h_soft_probe import read,digest
from run import write


def clean(trace):
    trace=copy.deepcopy(trace)
    for r in trace:
        r.pop('timing',None)
        for a in r['recovery']['attempts']:a.pop('solver_s')
    return trace


def main(smoke=False):
    assert not smoke,'Test timing audit has no smoke mode'
    register();phase='test';root=OUT/('timing_'+phase)
    complete=read(root/'completed.json');assert complete['passed'] and complete['smoke']==smoke
    for p,h in complete['hashes'].items():assert digest(Path(p))==h
    folders=sorted(p for p in root.glob('r*') if p.is_dir());assert len(folders)==complete['conditions']
    groups=[];hashes={str(root/'completed.json'):digest(root/'completed.json')}
    for folder in folders:
        done=read(folder/'completed.json');assert done['passed'] and done['exact_replay']
        for p,h in done['hashes'].items():assert digest(Path(p))==h
        references=[Path(p).parent for p in done['hashes'] if p.endswith('/completed.json') and str(root) not in p]
        assert len(references)==1;reference=references[0]
        summary=read(folder/'summary.json');rows=summary['episodes'];n=64
        assert len(rows)==n and [e['case'] for e in rows]==list(range(n))
        calls=[json.loads(line) for line in (folder/'solver_calls.jsonl').read_text().splitlines()];counts=read(folder/'solver_attempts.json')
        assert counts==summary['solver_counts']
        assert len(calls)==counts['solve_attempts']==counts['solve_completed']
        assert sum(c['kind']=='warmup' for c in calls)==counts['warmup_attempts']==n
        assert sum(c['kind'].startswith('retry') for c in calls)==counts['retry_attempts']
        assert summary['logging_operations']==1+3*len(calls)
        cursor=0;steps=0;logged=0.
        for e in rows:
            cid=e['case'];trace=read(folder/('r0_trace_%02d.json'%cid));old=read(reference/('r0_trace_%02d.json'%cid))
            assert clean(trace)==clean(old) and e['steps']==len(trace)
            assert e['cost']==sum(-r['reward'] for r in trace)
            reset=read(folder/('reset_%02d.json'%cid));assert reset==e['reset']
            assert reset['reset_gross_s']>=reset['controller_gross_s']>reset['logging_s']>0
            np.testing.assert_allclose(reset['controller_s'],reset['controller_gross_s']-reset['logging_s'],rtol=1e-12,atol=1e-12)
            assert calls[cursor]['kind']=='warmup' and calls[cursor]['case']==cid and calls[cursor]['step']==-1;cursor+=1
            logged+=reset['logging_s'];times=[]
            for t,r in enumerate(trace):
                a=r['recovery']['attempts'];assert calls[cursor:cursor+len(a)]==a;cursor+=len(a)
                assert all(x['case']==cid and x['step']==t for x in a)
                tm=r['timing'];assert all(np.isfinite(v) and v>0 for v in tm.values())
                np.testing.assert_allclose(tm['controller_s'],tm['controller_gross_s']-tm['logging_s'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(tm['decision_s'],tm['selection_s']+tm['controller_s'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(tm['decision_gross_s'],tm['decision_s']+tm['logging_s'],rtol=1e-12,atol=1e-12)
                assert tm['controller_s']>=sum(x['solver_s'] for x in a)-1e-12
                times.append(tm['decision_s']);logged+=tm['logging_s']
            assert e['deadline_exceed_steps']==sum(t>(.1 if summary['task']=='vehicle' else .04) for t in times)
            expected=dict(decision_total_s=np.sum(times),decision_mean_s=np.mean(times),decision_median_s=np.median(times),decision_p95_s=np.percentile(times,95),
                decision_gross_total_s=sum(r['timing']['decision_gross_s'] for r in trace),logging_total_s=sum(r['timing']['logging_s'] for r in trace))
            for k,v in expected.items():np.testing.assert_allclose(e[k],v,rtol=1e-12,atol=1e-12)
            steps+=len(trace)
        assert cursor==len(calls) and summary['logging_total_s']>logged
        attempts=[read(p) for p in folder.glob('attempt_*.json')]
        assert sum(a['step_calls'] for a in attempts)==steps
        assert sum(a['reset_calls'] for a in attempts)==n
        groups.append(dict(folder=str(folder),episodes=n,steps=steps,solver_counts=counts,logging_operations=summary['logging_operations']))
        hashes[str(folder/'completed.json')]=digest(folder/'completed.json')
    result=dict(passed=True,phase=phase,conditions=len(groups),episodes=sum(g['episodes'] for g in groups),steps=sum(g['steps'] for g in groups),groups=groups,hashes=hashes,source_hash=digest(Path(__file__)),
        scope='Every trace replay, raw solve log, latency decomposition, aggregate percentile and durable attempt budget checked. Smoke may run concurrently and gives no speed evidence. Does not eliminate host scheduling noise.')
    write(OUT/('audit_timing_'+phase+'.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in ('groups','hashes')},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.smoke)
