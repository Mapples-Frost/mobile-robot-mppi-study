"""Serial measured closed-loop tree learning, durable records, sealed evaluation.

This entry point intentionally implements training only. Validation/test rollout
requires a separate frozen evaluator after all training and independent audits.
"""
import argparse
import copy
import fcntl
import json
import os
import platform
import sys
import time
from pathlib import Path
from latency_tree_protocol import ROOT, ART, OUT, REG, TASKS, read, write, sha, verify, model_dir, bank_name, bank_specs, spec
from latency_tree_policy import BASE, FEATURES, constant, choose, features, thresholds, initialize, sample, update, policy_key, rank
from runtime import imports
from conservative_canonical_reset import make_env
from conservative_iteration_timing import idle
from branch_calibration_run import meter, observed_step
from relative_policy_features import context
from gated_horizon_timing import LoggingTimer
from gated_horizon_search import case_metrics
from run import snapshot, weights_hash, serial
import conservative_solver_recovery as recovery_module
import numpy as np


def verify_done(path):
    done=read(path)
    assert done['passed']
    for name, digest in done['hashes'].items():
        assert sha(Path(name)) == digest, name
    return done


def clean(trace):
    data=copy.deepcopy(trace)
    for row in data:
        row.pop('timing', None)
        for attempt in row['recovery']['attempts']:
            attempt.pop('solver_s', None)
    return data


def acquire(folder):
    folder.mkdir(parents=True, exist_ok=True)
    lock=(folder/'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    return lock


def bank_generation():
    verify(); lock=acquire(OUT/'banks')
    if (OUT/'banks/completed.json').exists():
        verify_done(OUT/'banks/completed.json'); return
    for task in TASKS:
        dest=OUT/'bank_generation'/task; dest.mkdir(parents=True,exist_ok=True)
        jobs=[j for j in bank_specs() if j['task']==task]
        if all(bank_name(task,j['split'],j['seed']).exists() for j in jobs):
            continue
        assert not list(dest.glob('attempt_*.json')), 'Partial bank generation requires preserved recovery audit'
        env=make_env(task,0,aligned=True,scaled_obs=True); counts=meter(env,dest)
        recovery=recovery_module.install(env.control_system.controller.mpc,dest)
        for j in jobs:
            path=bank_name(task,j['split'],j['seed']); assert not path.exists()
            env.seed(j['rng']); np.random.seed(j['rng']); cases=[]
            for cid in range(j['cases']):
                recovery.update(enabled=False,case=cid,step=-1)
                env.reset(); cases.append(snapshot(env))
            write(path,dict(task=task,split=j['split'],seed=j['seed'],rng=j['rng'],cases=cases,test_sealed=j['split']=='test'))
        assert counts['reset_calls']==sum(j['cases'] for j in jobs) and counts['step_calls']==0
        assert recovery['counts']['solve_attempts']==recovery['counts']['solve_completed']==counts['reset_calls']
        write(dest/'completed.json',dict(passed=True,resets=counts['reset_calls'],constructor_calls_unmeasured=True,
            hashes={str(p):sha(p) for p in list(dest.glob('attempt_*.json'))+[dest/'solver_attempts.json',dest/'solver_calls.jsonl']}))
    paths=list((OUT/'banks').glob('*_bank.json'))+list((OUT/'bank_generation').glob('*/completed.json'))
    assert len(paths)==20
    write(OUT/'banks/completed.json',dict(passed=True,cases=556,hashes={str(p):sha(p) for p in paths}))
    print('All18 fresh scenario banks generated:556 cases; no scored outcomes',flush=True)


def split_audit():
    from conservative_split_audit import fingerprint
    verify(); verify_done(OUT/'banks/completed.json')
    result_path=OUT/'split_audit.json'
    if result_path.exists():
        verify_done(result_path); return
    candidates=set()
    for root in (ART/'configs',ART/'results'):
        for pattern in ('*bank*.json','*validation.json','*test.json','*holdout.json'):
            candidates.update(root.rglob(pattern))
    history={}; files=[]; unsupported=[]
    for path in sorted(candidates):
        if OUT in path.parents: continue
        value=read(path)
        if not isinstance(value,dict) or not isinstance(value.get('cases'),list): continue
        files.append(path)
        for cid,case in enumerate(value['cases']):
            key=fingerprint(case)
            if key is None: unsupported.append(dict(path=str(path),case=cid))
            else: history.setdefault(key,[]).append(dict(path=str(path),case=cid))
    current={}; overlap=[]; rows=[]
    for job in bank_specs():
        path=bank_name(job['task'],job['split'],job['seed']); value=read(path)
        assert len(value['cases'])==job['cases'] and value['rng']==job['rng']
        for cid,case in enumerate(value['cases']):
            key=fingerprint(case); assert key
            row=dict(path=str(path),case=cid,task=key[0],fingerprint=key[1],split=job['split'])
            if key in current or key in history: overlap.append(row)
            current[key]=row; rows.append(row)
    value=dict(passed=not overlap and not unsupported,current_cases=len(rows),historical_unique_cases=len(history),
               historical_banks=len(files),overlaps=overlap,unsupported=unsupported,current=rows,
               scope='Exact saved reset-input identity only; no test outcomes, no proof of statistical independence.',
               hashes={str(p):sha(p) for p in files+[OUT/'banks/completed.json',Path(__file__).resolve()]})
    write(result_path,value); assert value['passed']; print(json.dumps({k:v for k,v in value.items() if k not in ('hashes','current')},indent=2))


def load_terminal(task,seed):
    _,SAC,_=imports(); folder=model_dir(task,seed); model=SAC.load(str(folder/'model.zip'))
    try:
        assert weights_hash(model)==read(folder/'completed.json')['final_hash']
        return model.policy_tf.get_mpc_vfn_weights_and_biases()
    finally: model.sess.close()


def condition(task,seed,split,policy,dest,repeats=1):
    assert split in ('smoke','fit','select'), 'This runner cannot access validation/test outcomes'
    idle(); lock=acquire(dest); completion=dest/'completed.json'
    if completion.exists():
        verify_done(completion); return read(dest/'summary.json')
    assert not (dest/'solver_attempts.json').exists() and not list(dest.glob('attempt_*.json')), 'Interrupted condition: preserve and audit before recovery'
    bank=bank_name(task,split,seed if split in ('fit','select') else None)
    assert sha(bank)==read(OUT/'banks/completed.json')['hashes'][str(bank)]
    write(dest/'policy.json',policy)
    env=make_env(task,seed,aligned=True,scaled_obs=True); counts=meter(env,dest)
    env.set_value_function_weights_and_biases(*load_terminal(task,seed))
    controller=env.control_system.controller; original=controller.get_action; measured=[]; episodes=[]
    write(dest/'environment.json',dict(pid=os.getpid(),python=sys.version,platform=platform.platform(),started=time.time(),
          thread_environment={k:v for k,v in os.environ.items() if k.endswith('NUM_THREADS') or k.startswith('TF_NUM_')},
          constructor_calls_unmeasured=True,training_time_not_independent_effect=True))
    with LoggingTimer(dest) as logging:
        recovery=recovery_module.install(controller.mpc,logging)
        def timed(*args,**kwargs):
            before=logging.seconds; start=time.perf_counter(); value=original(*args,**kwargs); gross=time.perf_counter()-start
            logged=logging.seconds-before; assert 0<=logged<gross
            measured.append(dict(controller_gross_s=gross,logging_s=logged,controller_s=gross-logged)); return value
        controller.get_action=timed
        for repeat in range(repeats):
            for cid,case in enumerate(read(bank)['cases']):
                recovery.update(enabled=False,events=[],case=cid,step=-1)
                start=time.perf_counter(); env.reset(**copy.deepcopy(case)); reset_s=time.perf_counter()-start
                assert len(measured)==1; reset=dict(reset_gross_s=reset_s,**measured.pop()); recovery['enabled']=True
                trace=[]; previous_initial=previous_final=False
                rawpath=dest/('r%d_trace_%02d.jsonl'%(repeat,cid))
                with rawpath.open('x') as stream:
                    for t in range(env.max_steps):
                        recovery['step']=t
                        start=time.perf_counter()
                        if policy['kind'] in ('tree','combined'):
                            ctx=context(env,task)
                            ctx.update(previous_initial_failure=previous_initial,previous_final_failure=previous_final)
                            h,decision=choose(policy,ctx)
                        else:
                            h,decision=choose(policy,{'state':env.control_system.current_state})
                            ctx=None
                        selection_s=time.perf_counter()-start
                        if ctx is None:
                            ctx=context(env,task); ctx.update(previous_initial_failure=previous_initial,previous_final_failure=previous_final)
                        _,terminated,row=observed_step(env,task,h,case,t); assert len(measured)==1
                        timing=measured.pop(); timing.update(selection_s=selection_s,decision_s=selection_s+timing['controller_s'],decision_gross_s=selection_s+timing['controller_gross_s'])
                        row.update(policy_context=ctx,tree_features=features(task,ctx),decision=decision,recovery=recovery['events'][-1],timing=timing)
                        stream.write(json.dumps(row,default=serial,allow_nan=False)+'\n'); stream.flush(); trace.append(row)
                        previous_initial=not row['recovery']['attempts'][0]['success']; previous_final=not row['solver_success']
                        if terminated: break
                assert terminated
                write(dest/('r%d_trace_%02d.json'%(repeat,cid)),trace)
                write(dest/('r%d_reset_%02d.json'%(repeat,cid)),reset)
                if repeat: assert clean(trace)==clean(read(dest/('r0_trace_%02d.json'%cid)))
                values=np.asarray([r['timing']['decision_s'] for r in trace]); ep=case_metrics(task,trace)
                ep.update(case=cid,repeat=repeat,decision_total_s=float(values.sum()),decision_mean_s=float(values.mean()),
                          gross_total_s=sum(r['timing']['decision_gross_s'] for r in trace),logging_total_s=sum(r['timing']['logging_s'] for r in trace),
                          decision_p95_s=float(np.percentile(values,95)),deadline_exceed_steps=int(sum(values>(.1 if task=='vehicle' else .04))))
                episodes.append(ep)
                write(dest/'progress.json',dict(pid=os.getpid(),episodes=len(episodes),expected=len(read(bank)['cases'])*repeats,steps=counts['step_calls']))
        assert logging.operations==1+3*recovery['counts']['solve_completed']
        summary=dict(task=task,seed=seed,split=split,policy=policy,episodes=episodes,repeats=repeats,
                     steps=counts['step_calls'],resets=counts['reset_calls'],solver_counts=recovery['counts'],logging_operations=logging.operations,
                     bank=str(bank),test_accessed=False)
        write(dest/'summary.json',summary)
    files=[p for p in dest.iterdir() if p.is_file() and p.name not in ('run.lock','completed.json','progress.json')]
    files += [REG,bank,model_dir(task,seed)/'model.zip']
    write(completion,dict(passed=True,hashes={str(p):sha(p) for p in files},training_only=True,record_audit_pending=True))
    print(json.dumps(dict(condition=dest.name,task=task,seed=seed,episodes=len(episodes),steps=summary['steps'])),flush=True)
    return summary


def smoke():
    verify(); verify_done(OUT/'split_audit.json'); lock=acquire(OUT/'smoke')
    for task in TASKS:
        tree=dict(kind='tree',task=task,nodes=[dict(feature=0,threshold=v) for v in (.3,.1,.8)],leaves=[5,BASE[task],35,50])
        for seed in range(3):
            for name,policy in (('fixed',constant(task)),('tree',tree)):
                condition(task,seed,'smoke',policy,OUT/'smoke'/('%s_s%d_%s'%(task,seed,name)),repeats=2)
    paths=list((OUT/'smoke').glob('*/completed.json')); assert len(paths)==12
    write(OUT/'smoke/completed.json',dict(passed=True,conditions=12,episodes=48,hashes={str(p):sha(p) for p in paths}))


def train_job(task,seed):
    dest=OUT/'train'/('%s_s%d'%(task,seed)); lock=acquire(dest)
    if (dest/'completed.json').exists(): verify_done(dest/'completed.json'); return
    assert not (dest/'started.json').exists(), 'Partial training requires explicit recovery; never silently resample'
    write(dest/'started.json',dict(pid=os.getpid(),started=time.time(),task=task,seed=seed))
    initial=condition(task,seed,'fit',constant(task),dest/'threshold_reference')
    episodes_features=[[r['tree_features'] for r in read(dest/'threshold_reference'/('r0_trace_%02d.json'%cid))] for cid in range(12)]
    cuts=thresholds(task,episodes_features); distribution=initialize(task,cuts)
    write(dest/'thresholds.json',dict(feature_names=FEATURES[task],cuts=cuts,source=sha(dest/'threshold_reference/completed.json')))
    rng=np.random.RandomState(2609265000+TASKS.index(task)*100+seed); records=[]
    for generation in range(4):
        folder=dest/('generation%d'%generation); folder.mkdir()
        proposals=[]
        for j in range(12):
            policy,genome=sample(task,cuts,distribution,rng)
            proposals.append(dict(id='g%d_c%02d'%(generation,j),policy=policy,genome=genome))
        order=np.random.RandomState(2609266000+TASKS.index(task)*1000+seed*10+generation).permutation(13).tolist()
        write(folder/'registration.json',dict(distribution=distribution,proposals=proposals,order=order))
        summaries={}
        for index in order:
            name='fixed' if index==12 else proposals[index]['id']
            policy=constant(task) if index==12 else proposals[index]['policy']
            summaries[name]=condition(task,seed,'fit',policy,folder/name)
        current=[]
        for p in proposals:
            score=rank(summaries[p['id']]['episodes'],summaries['fixed']['episodes'])
            item=dict(p,rank=score,folder=str(folder/p['id'])); records.append(item); current.append(item)
        ordered=sorted(current,key=lambda r:(r['rank']['violations'],r['rank']['objective'],r['id']))
        distribution=update(distribution,[r['genome'] for r in ordered[:3]])
        write(folder/'selection.json',dict(all_candidates=current,elite_ids=[r['id'] for r in ordered[:3]],next_distribution=distribution))
    eligible=sorted([r for r in records if r['rank']['eligible']],key=lambda r:(r['rank']['objective'],r['id']))
    selected=[]; seen=set()
    for item in eligible:
        key=policy_key(item['policy'])
        if key not in seen: selected.append(item); seen.add(key)
        if len(selected)==4: break
    arms=[dict(id='fixed',policy=constant(task))]+selected
    if task=='pendulum': arms.append(dict(id='certificate',policy=dict(kind='certificate',task=task)))
    order=[]
    for repeat in range(2):
        for i in np.random.RandomState(2609267000+TASKS.index(task)*1000+seed*10+repeat).permutation(len(arms)):
            order.append(dict(repeat=repeat,arm=arms[int(i)]['id']))
    write(dest/'selection_registration.json',dict(finalists=selected,arms=arms,order=order))
    results={}
    for item in order:
        arm=next(a for a in arms if a['id']==item['arm'])
        results[(item['repeat'],arm['id'])]=condition(task,seed,'select',arm['policy'],dest/'selection'/('r%d_%s'%(item['repeat'],arm['id'])))
        if item['repeat']:
            for cid in range(16):
                suffix='r0_trace_%02d.json'%cid
                assert clean(read(dest/'selection'/('r1_'+arm['id'])/suffix))==clean(read(dest/'selection'/('r0_'+arm['id'])/suffix))
    scores=[]
    for arm in arms:
        if arm['id']=='certificate': continue
        ranks=[rank(results[(r,arm['id'])]['episodes'],results[(r,'fixed')]['episodes']) for r in range(2)]
        score=dict(id=arm['id'],policy=arm['policy'],eligible=all(r['eligible'] for r in ranks),
                   objective=sum(r['objective'] for r in ranks)/2,ranks=ranks)
        scores.append(score)
    winner=min([s for s in scores if s['eligible']],key=lambda r:(r['objective'],r['id']!='fixed',r['id']))
    write(dest/'policy.json',winner['policy']); write(dest/'fit.json',dict(task=task,seed=seed,all_candidates=records,selection=scores,selected=winner['id'],
           learned_tree_selected=winner['policy']['kind']=='tree',new_gradient_steps=0,validation_access=False,test_access=False))
    paths=list(dest.rglob('completed.json'))+list(dest.glob('*.json'))+list(dest.glob('generation*/registration.json'))+list(dest.glob('generation*/selection.json'))
    write(dest/'completed.json',dict(passed=True,hashes={str(p):sha(p) for p in sorted(set(paths))},training_only=True,selected=winner['id']))
    print(json.dumps(dict(task=task,seed=seed,training_complete=True,selected=winner['id'])),flush=True)


def suite():
    verify(); verify_done(OUT/'audit_smoke.json'); verify_done(OUT/'split_audit.json'); idle(); lock=acquire(OUT/'train')
    status=OUT/'status.json'
    if status.exists():
        old=read(status)
        if old.get('complete'):
            for p,h in old['hashes'].items(): assert sha(Path(p))==h
            print('Existing full training verified'); return
        raise RuntimeError('Existing incomplete controller state requires explicit recovery audit')
    state=dict(pid=os.getpid(),active=True,complete=False,stage='serial_measured_training',completed=[],started=time.time(),test_accessed=False)
    write(status,state)
    try:
        for task in TASKS:
            for seed in range(3):
                train_job(task,seed); path=OUT/'train'/('%s_s%d'%(task,seed))/'completed.json'
                state['completed'].append(str(path)); write(status,state)
        state.update(active=False,complete=True,ended=time.time(),hashes={p:sha(Path(p)) for p in state['completed']}); write(status,state)
    except BaseException as exc:
        state.update(active=False,complete=False,ended=time.time(),exception=repr(exc));write(status,state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('banks','splits','smoke','suite'),required=True)
    args=parser.parse_args()
    {'banks':bank_generation,'splits':split_audit,'smoke':smoke,'suite':suite}[args.mode]()
