"""Five-step discrete policy iteration, with full deterministic branch returns.

Original task and learned fixed-H terminals are retained. This is a supervised
policy-iteration extension, not an implementation of the paper's SAC algorithm.
"""
import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from runtime import ART, ROOT, imports
from conservative_canonical_reset import make_env
import numpy as np
from run import snapshot, weights_hash, write
from min_q_eval_suite import model_dir
from paper_h_soft_probe import digest, read
from branch_calibration_run import meter, observed_step
from branch_calibration_audit import audit_trace
from fixed_policy_branches import BASE, TASKS, metrics
from relative_policy_features import context

OUT = ART / 'results/conservative_iteration_2026-09-24'
HS = (5, 10, 20, 25, 30, 40, 50)
ANCHORS = {'vehicle': (0, 10, 25, 40, 60, 80, 100, 125), 'pendulum': (0, 10, 25, 40, 60, 80)}
BLOCK = 5


def protocol():
    return dict(method='Conservative ensemble discrete policy iteration; method extension',
        rationale='One-step pilot failed its across-seed 1% mechanism gate. Repeated five-step choices test cumulative horizon savings; dense causal features and policy-aligned continuation address observed sparse coverage and continuation mismatch.',
        tasks=TASKS, seeds=[0,1,2], horizons=HS, block=BLOCK, anchors=ANCHORS,
        terminal='Each seed reuses its own 15000-step independently trained fixed H25/H30 terminal, frozen. No new terminal fitting.',
        rounds=2, train_cases_per_seed_per_round=8,
        train_seed='2609244000 + task_index*100 + seed*10 + round',
        smoke_seed='2609243998 + task_index; one case, anchors0/10, H10 and fixed H',
        splits=dict(validation_seed_base=2609245000, validation_cases=24,
                    test_seed_base=2609246000, test_cases=48,
                    split_task_offset=100, shared_across_training_seeds=True),
        collection='Round0 source/continuation fixed H. Round1 source/continuation frozen round0 policy. First candidate is held five steps or until termination, then continuation policy recomputes H at each five-step boundary. Exact prefix replay retains warm starts. Force source choice to reproduce source suffix exactly.',
        labels='Finite undiscounted total cost differences to first-block fixed-H candidate; signed log1p(diff/0.1). No clipping, entropy, learned tail or future physical-state features. Full training returns are labels only.',
        safety_label='Candidate must not reduce source-policy goal/survival success, add constraint termination, or add solver-failure steps.',
        learner=dict(ensemble=3, hidden=[64,64], activation='ReLU', optimizer='Adam',
                     learning_rate=.0003, updates_per_member_per_round=3000, batch_size=32,
                     regression='SmoothL1 on relative transformed cost, fixed-H output constrained zero',
                     safety='BCE auxiliary head, equal weight to mean regression loss',
                     bootstrap='Resample complete training scenes independently per member; shared within a scene',
                     round1='Fresh same-seed initialization, fit only round1 labels to avoid mixing continuation policies'),
        extraction='At each block, use prior policy choice as fallback. A new H must be predicted cheaper than fallback by at least 0.1 transformed cost units by all 3 members, safety probability >=0.8 by all members, and within training support. Then choose smallest mean transformed cost; ties shorter H. Support radius is 95th percentile of leave-one-scene-out nearest-neighbor distance in training-normalized features. No guarantee of real safety is claimed.',
        baseline='Retain independently trained selected H25/H30 at all three seeds as primary fixed baselines. Re-evaluate all ten independently trained seed0 H5..50 candidates on new validation; if a different common H is selected, give it independent 15k-step seed1/2 terminal training before confirmation. Also validate all seven fixed H with the adaptive terminal to guard against a merely better constant action.',
        selection='Report rounds0 and1, select one common round per task by minimum across-seed mean total cost among rounds with no per-seed success decrease, constraint increase or solver-failure rate increase versus primary fixed and <=2% mean physical+constraint cost increase per seed. No winning-seed or scene selection. If no round qualifies, retain negative results and do not open test.',
        confirmation='Fresh sealed 48-scene test only after complete validation, fixed baseline selection/training and model hashes freeze. All three seeds must preserve success, constraint count and solver-failure rate and have <=2% physical+constraint cost increase versus selected fixed comparator. Core gain requires either >=3% total-cost reduction in every seed with paired scenario-bootstrap upper cost-difference bound below zero, or >=10% actual mean decision-time reduction in every seed with cost noninferiority and bootstrap upper mean timing ratio <1. At least one criterion in each task; all results and failures reported.',
        bootstrap='10000 paired scenario resamples, training seeds retained together; 95% percentile interval conditional on the three fitted seeds. Also report every seed separately; no inference about a large seed population.',
        timing='After model selection, two serial seeded random-order timing repetitions over all confirmation scenes and all selected arms; policy+controller latency, median/p95, deadline exceedance, episode totals and reset separately. No training/rollout workers concurrent. Short H and parallel training times do not establish acceleration.',
        metrics=['paper_total_cost','physical_cost','constraint_cost','success','constraint_episodes',
                 'solver_failure_steps','solver_failure_rate','H_penalty','mean_H','measured_decision_time'],
        budget='Durable explicit reset/step attempts; bank generation, constructors, prefix/source/suffix, model fitting and all extra evaluations separate. No credit for early failure. Inherited fixed search300k + selected extra seeds60k; previous method history remains separate.',
        limits='Branch labels use sampled training futures; source-prefix and next-policy state distributions still differ. Ensembles/support checks are heuristics. Five-step held H and supervised cost/safety learning depart from original method. Original masked-50 NLP remains in use.',
        workers=2)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    p = json.loads(json.dumps(protocol()))
    path = OUT / 'protocol.json'
    if path.exists(): assert read(path) == p
    else: write(path, p)
    names = ['conservative_iteration.py','conservative_canonical_reset.py','relative_policy_features.py','runtime.py','run.py',
             'fixed_policy_branches.py','branch_calibration_run.py','branch_calibration_audit.py',
             'paper_h_soft_probe.py','paper_grid_audit_report.py','min_q_eval_suite.py']
    paths = [ROOT/'experiments/bohn2021_reproduction'/n for n in names]
    paths += list((ART/'sources').glob('*/**/*.py'))
    paths += [path]
    for task in TASKS:
        paths.append(ART/'configs'/(task+'.json'))
        for seed in range(3):
            source=model_dir(task,'fixed',seed)
            paths += [source/n for n in ('model.zip','manifest.json','completed.json')]
    hashes={str(p):digest(p) for p in paths}
    path=OUT/'collection_inputs_sha256.json'
    if path.exists(): assert read(path)==hashes
    else: write(path,hashes)


def verify():
    assert read(OUT/'protocol.json')==json.loads(json.dumps(protocol()))
    for p,h in read(OUT/'collection_inputs_sha256.json').items(): assert digest(Path(p))==h,p


def prior_model(task, seed, round_id):
    if not round_id: return None
    path=OUT/('%s_s%d_r0'%(task,seed))/'policy.json'
    done=read(path.parent/'fit_completed.json')
    assert digest(path)==done['policy_hash']
    return read(path)


def action(task, ctx, model):
    if model is None: return BASE[task]
    from conservative_policy_model import choose
    return choose(model,ctx['features'])


def collect(task,seed,round_id,smoke=False):
    verify()
    dest=OUT/('smoke_'+task if smoke else '%s_s%d_r%d'%(task,seed,round_id))
    dest.mkdir(exist_ok=True)
    lock=(dest/'run.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (dest/'collection_completed.json').exists():
        for p,h in read(dest/'collection_completed.json')['hashes'].items(): assert digest(Path(p))==h
        return
    write(dest/'running.json',dict(pid=os.getpid(),started=time.time()))
    env=make_env(task,seed,aligned=True,scaled_obs=True)
    meter(env,dest)
    _,SAC,_=imports()
    source=model_dir(task,'fixed',seed)
    model=SAC.load(str(source/'model.zip'))
    assert weights_hash(model)==read(source/'completed.json')['final_hash']
    env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases())
    model.sess.close()
    prior=prior_model(task,seed,round_id)
    bank=dest/'train_bank.json'
    if not bank.exists():
        scene_seed=2609243998+TASKS.index(task) if smoke else 2609244000+TASKS.index(task)*100+seed*10+round_id
        env.seed(scene_seed);np.random.seed(scene_seed)
        cases=[]
        for _ in range(1 if smoke else 8):
            env.reset();cases.append(snapshot(env))
        write(bank,dict(seed=scene_seed,split='smoke' if smoke else 'train',cases=cases))
    cases=read(bank)['cases'];groups=[];skips=[];hashes={str(bank):digest(bank)}
    def step(h,case,t):
        ctx=context(env,task)
        _,done,row=observed_step(env,task,h,case,t)
        row['policy_context']=ctx
        return done,row
    for cid,case in enumerate(cases):
        path=dest/('source_%02d.json'%cid)
        if not path.exists():
            env.reset(**copy.deepcopy(case));trace=[]
            for t in range(env.max_steps):
                if t%BLOCK==0: h=action(task,context(env,task),prior)
                done,row=step(h,case,t);trace.append(row)
                if done:break
            assert done
            audit_trace(task,case,trace);write(path,trace)
        original=read(path);audit_trace(task,case,original)
        hashes[str(path)]=digest(path)
        for anchor in ((0,10) if smoke else ANCHORS[task]):
            if anchor>=len(original):
                skips.append(dict(case=cid,anchor=anchor,termination=original[-1]['termination']));continue
            candidates=sorted(set((10,BASE[task]) if smoke else HS))
            branches={}
            for h in candidates:
                path=dest/('case%02d_t%03d_h%02d.json'%(cid,anchor,h))
                def rollout():
                    env.reset(**copy.deepcopy(case))
                    for t in range(anchor):
                        done,row=step(original[t]['horizon'],case,t)
                        assert not done and row==original[t],('prefix',cid,anchor,h,t)
                    assert context(env,task)==original[anchor]['policy_context']
                    trace=[];selected=h
                    for t in range(anchor,env.max_steps):
                        if t>=anchor+BLOCK and t%BLOCK==0: selected=action(task,context(env,task),prior)
                        done,row=step(selected,case,t);trace.append(row)
                        if done:break
                    assert done
                    audit_trace(task,case,trace,offset=anchor)
                    return trace
                if not path.exists():
                    trace=rollout()
                    if smoke:assert rollout()==trace,'Branch repeat differs'
                    write(path,dict(case=cid,anchor=anchor,h=h,trace=trace,metrics=metrics(task,trace)))
                b=read(path);audit_trace(task,case,b['trace'],offset=anchor)
                assert metrics(task,b['trace'])==b['metrics']
                if h==original[anchor]['horizon']:assert b['trace']==original[anchor:],'Source-choice branch must reproduce source suffix'
                hashes[str(path)]=digest(path);branches[str(h)]=b['metrics']
            groups.append(dict(case=cid,anchor=anchor,context=original[anchor]['policy_context'],
                source_h=original[anchor]['horizon'],branches=branches))
            print(json.dumps(dict(task=task,seed=seed,round=round_id,case=cid,anchor=anchor,
                source_h=original[anchor]['horizon'],costs={k:v['total_cost'] for k,v in branches.items()})),flush=True)
    attempts=[read(p) for p in dest.glob('attempt_*.json')]
    write(dest/'collection_completed.json',dict(task=task,seed=seed,round=round_id,smoke=smoke,
        audit_passed=True,groups=groups,skips=skips,hashes=hashes,
        inputs_hash=digest(OUT/'collection_inputs_sha256.json'),
        prior_policy_hash=digest(OUT/('%s_s%d_r0'%(task,seed))/'policy.json') if prior else None,
        explicit_step_calls=sum(a['step_calls'] for a in attempts),
        explicit_reset_calls=sum(a['reset_calls'] for a in attempts),environment_constructions=len(attempts)))


def suite(round_id,smoke):
    verify()
    if not smoke:
        for task in TASKS:assert read(OUT/('smoke_'+task)/'collection_completed.json')['audit_passed']
    def launch(pair):
        task,seed=pair
        log=OUT/('%s_%s_s%d_r%d_%d.log'%('smoke' if smoke else 'collect',task,seed,round_id,time.time_ns()))
        cmd=[sys.executable,'-u',__file__,'--mode','collect','--task',task,'--seed',str(seed),'--round',str(round_id)]
        if smoke:cmd.append('--smoke')
        with log.open('w') as stream:r=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=28800)
        return dict(task=task,seed=seed,exit_code=r.returncode,log=str(log))
    status=dict(pid=os.getpid(),active=True,finished=[],round=round_id,smoke=smoke)
    path=OUT/('smoke_status.json' if smoke else 'round%d_status.json'%round_id)
    write(path,status)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[(t,s) for t in TASKS for s in ([0] if smoke else range(3))]
        for f in as_completed([pool.submit(launch,j) for j in jobs]):
            status['finished'].append(f.result());write(path,status)
    status.update(active=False,complete=all(j['exit_code']==0 for j in status['finished']))
    write(path,status)
    assert status['complete'],'Inspect failures before resuming'


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['freeze','collect','suite'],required=True)
    ap.add_argument('--task',choices=TASKS);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--round',type=int,default=0,choices=[0,1]);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.mode=='freeze':freeze()
    elif args.mode=='collect':collect(args.task,args.seed,args.round,args.smoke)
    else:suite(args.round,args.smoke)
