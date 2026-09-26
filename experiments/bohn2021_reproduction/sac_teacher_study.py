"""Matched-budget author SAC with causal teacher replay; frozen terminal pilot."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from runtime import ART, ROOT, imports
from run import write, weights_hash
from teacher_common import make_bank

OUT = ART/'results/sac_teacher'
SCRIPT = Path(__file__).resolve()
SEEDS = [0, 1, 2]
SPLITS = {'validation': (26091951, 4), 'holdout': (26091952, 12)}
GRID = [1]+list(range(5, 51, 5))


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'hashes.json').exists():
        verify()
        return
    protocol = {
        'question': 'Can author SAC learn useful horizon allocation on the demonstrated-headroom plateau task, and does causal teacher replay help at equal simulation/update budget?',
        'scope': 'Frozen terminal mechanistic extension, not full paper reproduction or joint-terminal experiment.',
        'seeds': SEEDS, 'plain_steps': 21000, 'plain_checkpoints': [15000, 21000],
        'teacher_collection_steps': 6000, 'teacher_pretraining_updates': 6000, 'teacher_online_steps': 15000,
        'updates': 'Both final arms 20901 updates: online updates begin at transition100; teacher pretraining6000 + online14901; plain20901.',
        'teacher': 'Frozen causal H5/H30 rule with probability0.8; uniform continuous H1..50 with probability0.2. Same real dynamics and original reward. Store actual continuous action and rounded executed H. No Q-label regression or actor imitation.',
        'data': 'Each seed has independently generated common scene list for plain online and teacher online; separate teacher-collection scene list. All actions and actual state transitions saved. Counts include failures; no rejection sampling.',
        'algorithm': 'Pinned TF1 author SAC: target V, Q1 actor objective, twin Q, original entropy-regularized updates; no modern SAC replacement.',
        'observation': 'Existing56D physical state/current ref/full50-step preview/remaining time. Same in all SAC arms.',
        'hyperparameters': {'gamma': .97, 'reward_divisor': .6, 'alpha': .01, 'lr': .0003, 'tau': .005, 'batch': 256, 'buffer': 50000, 'actor': [32,32], 'q_v': [256,256]},
        'reward': 'Original raw reward -(physical energy/tracking/control cost + .003*H + 10*remaining on physical failure), divided by.6 exactly once. No shift, clipping or teacher bonus.',
        'endpoint': '600-step finite horizon and physical failures both terminal, no bootstrap. This differs from earlier author time-aware timeout bootstrapping.',
        'terminal': 'Identical frozen float32 analytic Riccati terminal in every arm; common zero-terminal H50 reset warmup. No auxiliary terminal optimizer.',
        'selection': 'All final seeds reported. Plain15000 is secondary online-budget comparison. Select fixed H on4 validation scenes: coarse grid then every integer +/-4 of winner. No learned checkpoint selection.',
        'test': SPLITS, 'fixed_grid': GRID,
        'metrics': ['raw cost', 'raw cost+294.3 offset-adjusted cost', 'physical cost', 'H proxy cost', 'constraint and solver failures', 'discounted original cost', 'phase horizon', 'actor/Q ranking diagnostic'],
        'unit': 'Training seed is replicate; identical test scenes are paired evaluations. No significance claim with3 seeds.',
        'gate': 'After validation, do not add another algorithm/hyperparameter variant using holdout feedback. Joint terminal is a later study; this run ends after matched SAC comparison.',
        'resources': 'At most2 active MPC processes, CPU single thread per process. Every checkpoint includes model plus optimizer/RNG/replay state.',
        'limitations': ['Teacher quality is an empirical assumption, not globally optimal control.', 'Teacher replay changes state/action distribution and update timing; equal-budget comparison does not isolate every component.', 'H penalty is compute proxy, not demonstrated wall-clock saving.', 'New task and preview/terminal extensions differ from paper.']}
    write(OUT/'protocol.json', protocol)
    for split, (seed,n) in SPLITS.items():
        write(OUT/(split+'_bank.json'), make_bank(seed,n))
    for seed in SEEDS:
        write(OUT/('online_s%d_bank.json'%seed), make_bank(26091960+seed,100))
        write(OUT/('teacher_s%d_bank.json'%seed), make_bank(26091970+seed,100))
    sources = [SCRIPT]+[SCRIPT.parent/n for n in ['runtime.py','run.py','optimized_runtime.py','forecast_runtime.py','riccati_terminal_probe.py','mechanism_probe.py','teacher_common.py','plateau_screen.py']]
    sources += [ART/'configs/pendulum.json',OUT/'protocol.json']+list(OUT.glob('*_bank.json'))
    sources += [ART/'sources/stable-baselines-horizon/stable_baselines'/n for n in ['sac/sac.py','sac/policies.py','common/buffers.py']]
    write(OUT/'hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in sources})


def verify():
    for p,h in read(OUT/'hashes.json').items():
        assert sha(ROOT/p)==h, p


def environment(seed):
    from optimized_runtime import install_terminal
    from forecast_runtime import make_env
    from riccati_terminal_probe import prior
    install_terminal('pendulum')
    env = make_env('pendulum',seed)
    env.max_steps=600
    env.config['environment']['max_steps']=600
    w,b,_=prior()
    env.set_value_function_weights_and_biases([w.astype(np.float32)],[b.astype(np.float32)])
    return env


def terminal_check(env):
    from riccati_terminal_probe import prior
    w,b,_=prior()
    vf=env.control_system.controller.mpc.vf
    np.testing.assert_array_equal(np.asarray(vf.weights_num).ravel(),w.astype(np.float32).ravel())
    np.testing.assert_array_equal(np.asarray(vf.biases_num).ravel(),b.astype(np.float32).ravel())


def model_new(env, seed):
    _,SAC,Policy=imports()
    return SAC(Policy,env,gamma=.97,learning_rate=3e-4,buffer_size=50000,
        learning_starts=100,batch_size=256,train_freq=1,gradient_steps=1,tau=.005,
        reward_scale=.6,ent_coef=.01,time_aware=False,seed=seed,n_cpu_tf_sess=1,verbose=0,
        policy_kwargs={'layers':{'pi':[32,32],'qf':[256,256],'vf':[256,256],'mpc':[]},
        'mpc_state_dim':4,'mpc_parameter_dim':1,'mpc_gamma':.97,
        'use_mpc_value_fn':True,'train_mpc_value_fn':False,'use_mpc_vf_target':False,'mpc_vf_type':'poly'})


def reset(env, bank, index):
    assert index<len(bank), 'Registered scene list exhausted'
    return env.reset(**copy.deepcopy(bank[index]['case']))


def transition(env, obs, unscaled, action, phase, episode, t):
    from mechanism_probe import checked_step
    h=int(np.clip(np.rint(unscaled),1,50))
    new,done,row=checked_step(env,'pendulum',h)
    row.update(env.optimized_solver_info)
    row.update(obs=np.asarray(obs).tolist(),next_obs=new.tolist(),action=float(action),
               unscaled_action=float(unscaled),done=bool(done),episode=episode,phase=phase,step=t)
    return new,done,row


def checkpoint(model, dest, label, metadata):
    import tensorflow as tf
    model.save(str(dest/label))
    with model.graph.as_default():
        variables=tf.global_variables()
    state={'variables':dict(zip([v.name for v in variables],model.sess.run(variables))),
        'replay':model.replay_buffer,'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
        'metadata':metadata,'note':'TF state and host RNG preserved; full in-progress MPC state is in trajectory, not restartable from this pickle alone.'}
    with (dest/(label+'_state.pkl')).open('wb') as f:
        pickle.dump(state,f,protocol=4)
    write(dest/(label+'_metadata.json'),dict(metadata,weights_hash=weights_hash(model)))


def train(arm,seed,smoke=False):
    verify()
    dest=OUT/('smoke' if smoke else 'models')/('%s_s%d'%(arm,seed))
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completed.json').exists(): return
    env=environment(seed)
    model=model_new(env,seed)
    initial_hash=weights_hash(model)
    from stable_baselines.common.math_util import scale_action, unscale_action
    from plateau_screen import action as rule
    online=read(OUT/('online_s%d_bank.json'%seed))['scenes']
    teacher=read(OUT/('teacher_s%d_bank.json'%seed))['scenes']
    rng=np.random.RandomState(26091980+seed)
    started=time.monotonic()
    updates=0
    trace_path=dest/'transitions.jsonl'
    assert not trace_path.exists(), 'Partial run preserved; explicit recovery required'
    manifest={'arm':arm,'seed':seed,'initial_hash':initial_hash,'smoke':smoke,'protocol_sha':sha(OUT/'protocol.json')}
    write(dest/'manifest.json',manifest)
    losses=[]
    def update():
        nonlocal updates
        value=model._train_step(updates,None,3e-4)
        model.sess.run(model.target_update_op)
        updates+=1
        assert np.all(np.isfinite(value)), value
        if updates%100==0: losses.append({'update':updates,'values':[float(v) for v in value]})
    def add(row):
        model.replay_buffer.add(np.asarray(row['obs'],np.float32),np.array([row['action']],np.float32),
            -row['cost']/.6,np.asarray(row['next_obs'],np.float32),float(row['done']))
    with trace_path.open('w') as log:
        if arm=='teacher':
            obs=reset(env,teacher,0);episode=0
            for t in range(256 if smoke else 6000):
                a=float(rng.uniform(1,50)) if rng.rand()<.2 else float(rule(env,'switch_5_30')[0])
                normalized=float(scale_action(env.action_space,np.array([a]))[0])
                obs,done,row=transition(env,obs,a,normalized,'teacher',episode,t+1)
                add(row); log.write(json.dumps(row)+'\n')
                if done and t+1<(256 if smoke else 6000):
                    terminal_check(env);episode+=1;obs=reset(env,teacher,episode)
            log.flush()
            for _ in range(10 if smoke else 6000):
                update()
                if updates%1000==0:
                    write(dest/'progress.json',{'phase':'pretrain','updates':updates,'elapsed_s':time.monotonic()-started})
                    print('pretrain',arm,seed,updates,flush=True)
            checkpoint(model,dest,'pretrained',{'updates':updates,'teacher_steps':256 if smoke else 6000})
        # Reset host RNGs for online collection, while retaining trained network/TF RNG.
        np.random.seed(seed);random.seed(seed);env.action_space.seed(seed)
        obs=reset(env,online,0);episode=0
        count=300 if smoke else (21000 if arm=='plain' else 15000)
        for t in range(count):
            if t<100:
                a=float(env.action_space.sample()[0])
                normalized=float(scale_action(env.action_space,np.array([a]))[0])
            else:
                normalized=float(model.policy_tf.step(obs[None],deterministic=False).ravel()[0])
                a=float(unscale_action(env.action_space,np.array([normalized]))[0])
            obs,done,row=transition(env,obs,a,normalized,'online',episode,t+1)
            add(row);log.write(json.dumps(row)+'\n')
            model.num_timesteps=t+1
            if t+1>=100: update()
            if (t+1)%500==0:
                terminal_check(env);log.flush()
                write(dest/'progress.json',{'phase':'online','steps':t+1,'target':count,'updates':updates,'episodes':episode,'elapsed_s':time.monotonic()-started})
                write(dest/'losses.json',losses)
                print('online',arm,seed,t+1,updates,round(time.monotonic()-started,1),flush=True)
            if (t+1)%5000==0 or t+1==count:
                checkpoint(model,dest,'step_%05d'%(t+1),{'updates':updates,'online_steps':t+1,'teacher_steps':(256 if smoke else 6000) if arm=='teacher' else 0})
            if done and t+1<count:
                episode+=1;obs=reset(env,online,episode)
    terminal_check(env)
    write(dest/'losses.json',losses)
    write(dest/'completed.json',dict(manifest,updates=updates,online_steps=count,teacher_steps=(256 if smoke else 6000) if arm=='teacher' else 0,
        resets=episode+1,elapsed_s=time.monotonic()-started,final_hash=weights_hash(model),terminal_frozen=True))
    model.sess.close()


def evaluate(split,arm):
    verify()
    dest=OUT/'evaluation'/split/arm;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists(): return
    bank=read(OUT/(split+'_bank.json'))['scenes']
    model=None
    if arm.startswith(('plain','teacher')):
        group,seed_s,step_s=arm.split('_');seed=int(seed_s[1:]);step=int(step_s)
        env=environment(seed)
        _,SAC,_=imports()
        model=SAC.load(str(OUT/'models'/('%s_s%d'%(group,seed))/('step_%05d.zip'%step)))
        model.reward_scale=.6
        before=weights_hash(model)
    else:
        env=environment(26091990)
    from plateau_screen import action as rule
    summaries=[]
    for scene in bank:
        path=dest/('scene_%02d.json'%scene['id'])
        if path.exists():
            summaries.append(read(path)['summary']);continue
        obs=reset(env,bank,scene['id']);initial=obs.tolist();trace=[]
        while True:
            h=int(np.clip(np.rint(model.predict(obs,deterministic=True)[0][0]),1,50)) if model else rule(env,arm)[0]
            obs,done,row=transition(env,obs,h,2*(h-1)/49.-1,'evaluation',scene['id'],len(trace)+1)
            trace.append(row)
            if done:break
        terminal_check(env)
        total=sum(r['cost'] for r in trace)
        summary={'scene':scene['id'],'arm':arm,'steps':len(trace),'raw_cost':total,'adjusted_cost':total+294.3,
            'physical_cost':sum(r['performance']+.4905 for r in trace),'H_cost':sum(r['compute'] for r in trace),
            'constraint_cost':sum(r['constraint'] for r in trace),'termination':trace[-1]['termination'],
            'mean_H':float(np.mean([r['horizon'] for r in trace])), 'solver_failures':sum(not r['solver_success'] for r in trace),
            'discounted_cost':sum(.97**t*r['cost'] for t,r in enumerate(trace)),
            'rmse':float(np.sqrt(np.mean([(r['state']['pos']-r['next_obs'][4])**2 for r in trace])))}
        # Current reference is raw in obs[4] (normalization divisor=1).
        summaries.append(summary);write(path,{'initial':initial,'summary':summary,'trace':trace})
    if model:
        assert before==weights_hash(model);model.sess.close()
    write(dest/'summary.json',{'arm':arm,'episodes':summaries,'mean_adjusted_cost':float(np.mean([r['adjusted_cost'] for r in summaries]))})
    print(split,arm,'done',flush=True)


def launch(jobs):
    def job(args):
        name='_'.join(args).replace('--','')
        logs=OUT/'logs';logs.mkdir(exist_ok=True)
        with (logs/(name+'.log')).open('a') as log:
            p=subprocess.run([sys.executable,'-u',str(SCRIPT)]+args,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
        assert p.returncode==0,(args,p.returncode)
        print('complete',args,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(job,jobs))


def suite():
    prepare()
    jobs=[['--train',arm,'--seed',str(seed)] for seed in SEEDS for arm in ['plain','teacher']]
    np.random.RandomState(26091950).shuffle(jobs)
    launch(jobs)
    learned=['%s_s%d_%d'%(arm,seed,steps) for arm,steps in [('plain',15000),('plain',21000),('teacher',15000)] for seed in SEEDS]
    def eval_jobs(split,arms):launch([['--evaluate',arm,'--split',split] for arm in arms])
    eval_jobs('validation',['fixed_%d'%h for h in GRID]+['switch_5_30']+learned)
    def fixed_rows():return [read(p) for p in (OUT/'evaluation/validation').glob('fixed_*/summary.json')]
    best=min(fixed_rows(),key=lambda x:(x['mean_adjusted_cost'],int(x['arm'].split('_')[1])))
    h=int(best['arm'].split('_')[1])
    refined=[i for i in range(max(1,h-4),min(50,h+4)+1) if i not in GRID]
    write(OUT/'refinement.json',{'coarse_best':h,'extra':refined})
    eval_jobs('validation',['fixed_%d'%i for i in refined])
    best=min(fixed_rows(),key=lambda x:(x['mean_adjusted_cost'],int(x['arm'].split('_')[1])))
    write(OUT/'selection.json',{'fixed':best['arm'],'validation_cost':best['mean_adjusted_cost'],'learned_selection':'none'})
    # All choices finalized before opening any holdout outcomes.
    eval_jobs('holdout',list(dict.fromkeys([best['arm'],'fixed_30','switch_5_30']+learned)))
    verify()
    write(OUT/'completed.json',{'all_seeds':SEEDS,'selected_fixed':best['arm'],'holdout_arms':list(dict.fromkeys([best['arm'],'fixed_30','switch_5_30']+learned))})


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--train',choices=['plain','teacher'])
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--evaluate')
    ap.add_argument('--split',default='validation',choices=['validation','holdout'])
    args=ap.parse_args()
    if args.prepare:prepare()
    elif args.train:train(args.train,args.seed,args.smoke)
    elif args.evaluate:evaluate(args.split,args.evaluate)
    else:suite()
