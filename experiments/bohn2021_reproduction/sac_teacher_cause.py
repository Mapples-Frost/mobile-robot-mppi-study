"""Frozen interventions and full-return diagnostics of teacher-replay degradation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sac_teacher_study import (OUT as STUDY, ROOT, read, write, sha, verify as verify_study,
    environment, reset, transition, terminal_check, imports, weights_hash)
from teacher_common import make_bank
from plateau_screen import action as rule_action

OUT = STUDY.parent/'sac_teacher_cause'
GRID = np.asarray([1,2,3,5,10,20,25,30,40,50])
SCRIPT = Path(__file__).resolve()


def models():
    result={}
    for arm in ['plain','teacher']:
        for seed in range(3):
            labels=['step_21000'] if arm=='plain' else ['pretrained','step_05000','step_10000','step_15000']
            for label in labels:
                result['%s_s%d_%s'%(arm,seed,label)] = STUDY/'models'/('%s_s%d'%(arm,seed))/(label+'.zip')
    return result


def finals():
    return [name for name in models() if name.endswith(('step_21000','step_15000'))]


def prepare():
    verify_study()
    OUT.mkdir(exist_ok=True)
    if (OUT/'hashes.json').exists():
        verify();return
    protocol={
        'purpose':'Post-result mechanism diagnosis, no retraining or new performance claim.',
        'bank':[26091991,4], 'training_seeds':[0,1,2],
        'frozen':['all original models','terminal','reward','original study sources'],
        'interventions':{'det':'Original deterministic actor',
            'stoch0':'Original squashed Gaussian actor with external reproducible normal stream, replica0',
            'stoch1':'Same stochastic policy, independent replica1',
            'min5':'Execute max(5, deterministic rounded H); no retraining',
            'q1':'Execute maximum Q1 on registered integer grid, smallest H tie break'},
        'grid':GRID.tolist(),
        'history':'All three teacher pretrained/5k/10k checkpoints evaluated deterministically; final endpoints included above. No checkpoint selection.',
        'controls':['fixed25','rule'],
        'branches':'Scene0 rule-prefix common states at elapsed40, first switch-15, +15, +70. All6 final models; first action from actor, grid-Q1 or rule; then current stochastic actor to actual episode termination. Two common-random-number replicas per candidate. No approximate tail.',
        'branch_target':'Q(s,a)=r0+sum gamma^t*(rt-alpha*log pi(at|st)) for t>=1; original scaled reward, gamma.97, alpha.01. Compare paired candidate differences, report Monte Carlo variation. Conditional on one exogenous scene future, not population true-Q proof.',
        'information':'Policy only reads existing50-step preview; evaluation simulator uses actual future. Branch future is diagnostic conditioning, not a new deployable oracle.',
        'budget':'164 full evaluation episodes <=98400 transitions;144 full prefix+branch episodes <=86400 transitions; one warmup per reset; wiring smoke separate. No SAC updates.',
        'limits':['4 diagnostic scenes are paired cases, not independent training seeds.',
                  'Stochastic two-replica results are descriptive, not high-precision MC calibration.',
                  'min5 changes both compute penalty and physical control; isolates executed horizon restriction, not solver alone.',
                  'Q1-greedy changes policy; improvement alone does not establish accurate Q.',
                  'Checkpoint changes locate timing, not causal attribution to actor versus critic optimization.'],
        'resources':'At most two MPC workers; original CPU single-thread settings.'}
    write(OUT/'protocol.json',protocol)
    write(OUT/'bank.json',make_bank(26091991,4))
    paths=[SCRIPT,OUT/'protocol.json',OUT/'bank.json']+list(models().values())
    write(OUT/'hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    verify_study()
    for name,digest in read(OUT/'hashes.json').items():assert sha(ROOT/name)==digest,name


class Policy:
    def __init__(self,name):
        self.name=name
        _,SAC,_=imports()
        path=models()[name]
        self.model=SAC.load(str(path));self.model.reward_scale=.6
        self.before=weights_hash(self.model)
        assert self.before==read(path.with_name(path.stem+'_metadata.json'))['weights_hash']
        self.q1=self.model.graph.get_tensor_by_name('model/values_fn/qf1/qf1/BiasAdd:0')
        self.q2=self.model.graph.get_tensor_by_name('model/values_fn/qf2/qf2/BiasAdd:0')

    def q(self,obs,actions):
        actions=np.asarray(actions,dtype=np.float32)
        feed={self.model.observations_ph:np.repeat(np.asarray(obs,np.float32)[None],len(actions),axis=0),
              self.model.actions_ph:((actions-1)/24.5-1).reshape(-1,1)}
        q1,q2=self.model.sess.run([self.q1,self.q2],feed)
        return q1.ravel(),q2.ravel()

    def choose(self,obs,mode,rng):
        from stable_baselines.common.math_util import unscale_action
        p=self.model.policy_tf
        det,mu,std=self.model.sess.run([p.deterministic_policy,p.act_mu,p.std],{p.obs_ph:np.asarray(obs,np.float32)[None]})
        mu,std=np.float32(mu.ravel()[0]),np.float32(std.ravel()[0])
        norm=np.float32(det.ravel()[0]);logp=None
        if mode.startswith('stoch'):
            z=np.float32(mu+std*np.float32(rng.normal()))
            norm=np.float32(np.tanh(z))
            logp=float(-.5*(((z-mu)/(std+1e-6))**2+2*np.log(std)+np.log(2*np.pi))-np.log(1-norm**2+1e-6))
        a=float(unscale_action(self.model.action_space,np.array([norm],np.float32))[0])
        suggested=int(np.clip(np.rint(a),1,50))
        if mode=='q1':a=float(GRID[self.q(obs,GRID)[0].argmax()])
        elif mode=='min5':a=float(max(5,suggested))
        return a,{'actor_mean':float(mu),'actor_std':float(std),'actor_H':suggested,'logp':logp}

    def close(self):
        assert self.before==weights_hash(self.model)
        self.model.sess.close()


def step(env,obs,a,t,decision=None):
    obs,done,row=transition(env,obs,a,(a-1)/24.5-1,'evaluation',0,t)
    row['decision']=decision or {}
    return obs,done,row


def summary(trace):
    return {'steps':len(trace),'raw_cost':sum(r['cost'] for r in trace),
        'adjusted_cost':sum(r['cost'] for r in trace)+294.3,
        'physical_cost':sum(r['performance']+.4905 for r in trace),
        'H_cost':sum(r['compute'] for r in trace),'termination':trace[-1]['termination'],
        'mean_H':float(np.mean([r['horizon'] for r in trace])),
        'solver_failures':sum(not r['solver_success'] for r in trace),
        'discounted_cost':sum(.97**t*r['cost'] for t,r in enumerate(trace))}


def evaluate(name,mode):
    verify()
    folder=OUT/'evaluation'/(name+'__'+mode)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():return
    bank=read(OUT/'bank.json')['scenes'];env=environment(26091992)
    policy=Policy(name) if name not in ['rule','fixed25'] else None
    results=[]
    for scene in bank:
        path=folder/('scene_%02d.json'%scene['id'])
        if path.exists():results.append(read(path)['summary']);continue
        obs=reset(env,bank,scene['id']);initial=obs.tolist();trace=[]
        replica=int(mode[-1]) if mode.startswith('stoch') else 0
        rng=np.random.RandomState(26091993+100*scene['id']+replica)
        while True:
            if policy:a,decision=policy.choose(obs,mode,rng)
            else:a,decision=(25,{}) if name=='fixed25' else (rule_action(env,'switch_5_30')[0],{})
            obs,done,row=step(env,obs,a,len(trace)+1,decision);trace.append(row)
            if done:break
        terminal_check(env)
        result=dict(summary(trace),scene=scene['id'],name=name,mode=mode)
        write(path,{'initial':initial,'summary':result,'trace':trace})
        results.append(result)
        print('evaluation',name,mode,'scene',scene['id'],round(result['adjusted_cost'],3),flush=True)
    if policy:policy.close()
    write(folder/'completed.json',{'episodes':results})


def branches(name):
    verify()
    folder=OUT/'branches'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():return
    bank=read(OUT/'bank.json')['scenes'];scene=bank[0]
    base=read(OUT/'evaluation/rule__det/scene_00.json')
    anchors=[40,scene['switches'][0]-15,scene['switches'][0]+15,scene['switches'][0]+70]
    env=environment(26091992);policy=Policy(name);results=[]
    for anchor in anchors:
        common=np.asarray(base['trace'][anchor]['obs'],np.float32)
        actor,_=policy.choose(common,'det',np.random.RandomState(0))
        qgrid=policy.q(common,GRID)[0]
        actions={'actor':actor,'q1':float(GRID[qgrid.argmax()]),'rule':float(base['trace'][anchor]['horizon'])}
        for source,a in actions.items():
            q1,q2=policy.q(common,[a])
            for replica in range(2):
                path=folder/('t%03d_%s_r%d.json'%(anchor,source,replica))
                if path.exists():results.append(read(path)['summary']);continue
                obs=reset(env,bank,0)
                np.testing.assert_array_equal(obs,base['initial'])
                for t in range(anchor):
                    expected=base['trace'][t]
                    obs,done,row=step(env,obs,expected['unscaled_action'],t+1)
                    assert not done
                    for key in ['state','input','cost','horizon','next_obs']:assert row[key]==expected[key],key
                np.testing.assert_array_equal(obs,common)
                trace=[];rng=np.random.RandomState(26092000+anchor*10+replica)
                while True:
                    if not trace:chosen,decision=a,{'logp':None}
                    else:chosen,decision=policy.choose(obs,'stoch',rng)
                    obs,done,row=step(env,obs,chosen,anchor+len(trace)+1,decision)
                    trace.append(row)
                    if done:break
                terminal_check(env)
                soft=sum(.97**t*(-r['cost']/.6-(.01*r['decision']['logp'] if t else 0.)) for t,r in enumerate(trace))
                result={'name':name,'anchor':anchor,'candidate':source,'first_action':a,'first_H':trace[0]['horizon'],
                    'replica':replica,'predicted_Q1':float(q1[0]),'predicted_Q2':float(q2[0]),
                    'soft_return':soft,'scaled_discounted_return':sum(.97**t*(-r['cost']/.6) for t,r in enumerate(trace)),
                    'branch_steps':len(trace),'prefix_steps':anchor,'reset_warmups':1,
                    'solver_failures':sum(not r['solver_success'] for r in trace),'termination':trace[-1]['termination']}
                write(path,{'initial':common.tolist(),'summary':result,'trace':trace});results.append(result)
        print('branches',name,'anchor',anchor,'done',flush=True)
    policy.close();write(folder/'completed.json',{'branches':results})


def smoke():
    OUT.mkdir(exist_ok=True)
    verify_study();bank=make_bank(26091991,1)['scenes'];env=environment(26091992)
    policy=Policy('teacher_s0_pretrained');checks=[]
    from sac_teacher_report import audit_row
    for mode in ['det','stoch0','min5','q1']:
        obs=reset(env,bank,0);previous=None
        rng=np.random.RandomState(42)
        for t in range(4):
            a,decision=policy.choose(obs,mode,rng)
            obs,done,row=step(env,obs,a,t+1,decision)
            audit_row(row,bank[0],t+1,previous);previous=row
            checks.append({'mode':mode,'H':row['horizon'],'logp':decision['logp']})
    # Check the external Gaussian density against the author's exact TF formula.
    import tensorflow as tf
    from stable_baselines.sac.policies import gaussian_likelihood,apply_squashing_func
    rng=np.random.RandomState(42)
    mu=rng.normal(size=(100,1)).astype(np.float32)
    logstd=rng.uniform(-3,1,size=(100,1)).astype(np.float32)
    std=np.exp(logstd);z=mu+std*rng.normal(size=(100,1)).astype(np.float32);norm=np.tanh(z)
    own=(-.5*(((z-mu)/(std+1e-6))**2+2*np.log(std)+np.log(2*np.pi))-np.log(1-norm**2+1e-6)).ravel()
    with policy.model.graph.as_default():
        logp=apply_squashing_func(tf.constant(mu),tf.constant(z),gaussian_likelihood(tf.constant(z),tf.constant(mu),tf.constant(logstd)))[2]
    ref=policy.model.sess.run(logp)
    np.testing.assert_allclose(own,ref,atol=2e-3,rtol=1e-3)
    policy.close()
    write(OUT/'smoke.json',{'checks':checks,'physical_steps':20,'density_max_error':float(np.max(abs(own-ref))),'reward_and_observation_verified':True})
    print('smoke passed',flush=True)


def jobs():
    result=[['--evaluate',name,'--mode',mode] for name in finals() for mode in ['det','stoch0','stoch1','min5','q1']]
    result += [['--evaluate',name,'--mode','det'] for name in models() if name not in finals()]
    result += [['--evaluate',name,'--mode','det'] for name in ['rule','fixed25']]
    np.random.RandomState(26091994).shuffle(result)
    return result


def launch(jobs_to_run):
    def run(args):
        name='_'.join(args).replace('--','')
        folder=OUT/'logs';folder.mkdir(exist_ok=True)
        with (folder/(name+'.log')).open('a') as stream:
            p=subprocess.run([sys.executable,'-u',str(SCRIPT)]+args,stdout=stream,stderr=subprocess.STDOUT,timeout=18000)
        assert p.returncode==0,(args,p.returncode)
        print('complete',args,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs_to_run))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--evaluate');ap.add_argument('--mode',default='det');ap.add_argument('--branches')
    args=ap.parse_args()
    if args.smoke:smoke()
    elif args.prepare:prepare()
    elif args.evaluate:evaluate(args.evaluate,args.mode)
    elif args.branches:branches(args.branches)
    else:
        prepare();launch(jobs());launch([['--branches',name] for name in finals()])
        verify();write(OUT/'completed.json',{'evaluation_jobs':len(jobs()),'branch_jobs':len(finals()),'no_training':True})
