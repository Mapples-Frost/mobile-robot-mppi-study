"""Original author SAC + do-mpc, reconstructed configs, explicit runtime fixes."""
import argparse
import fcntl
import copy
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from runtime import ART, ROOT, imports, make_env, install_correct_nstep
import numpy as np


def serial(obj):
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,np.generic): return obj.item()
    raise TypeError(type(obj).__name__)


def write(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,default=serial,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def weights_hash(model):
    h = hashlib.sha256()
    for k,v in sorted(model.get_parameters().items()):
        h.update(k.encode()); h.update(v.tobytes())
    return h.hexdigest()


def snapshot(env):
    """Capture actual randomized reset arguments; JSON avoids untrusted pickle."""
    c = env.control_system
    # history starts at the state BEFORE the upstream reset's warmup action.
    out = {'state':copy.deepcopy(c.history['state'][0]),'reference':{},'tvp':{}}
    for name,tvp in c.tvps.items():
        required = env.max_steps+52
        tvp.generate_values(max(0,required-len(tvp.values)))
        out['tvp'][name] = copy.deepcopy(tvp.values[:required])
    if env.config['mpc']['type']=='TTAHMPC':
        out['reference'] = {'theta_r':env.theta_r,'traj_steps':env.traj_steps,'ns':copy.deepcopy(c.controller.object_noise_seed)}
    return out


def bank(task):
    path = ART/'configs'/(task+'_test_bank.json')
    if path.exists(): return json.loads(path.read_text())
    env = make_env(task, 20210917)
    np.random.seed(20210917)
    cases = []
    for _ in range(10):
        env.reset()
        cases.append(snapshot(env))
    data = {'provenance':'New fixed test set: original paper test files unavailable.', 'seed':20210917,'cases':cases}
    write(path,data)
    return data


def evaluate(model, env, cases, out, fixed_horizon, use_value, terminal_weights=None):
    rows = []
    # Setting zero weights removes only terminal value, keeping the NLP identical.
    w,b = model.policy_tf.get_mpc_vfn_weights_and_biases() if terminal_weights is None else terminal_weights
    if not use_value:
        w,b = [np.zeros_like(a) for a in w], [np.zeros_like(a) for a in b]
    env.set_value_function_weights_and_biases(w,b)
    for j,case in enumerate(cases):
        obs = env.reset(**copy.deepcopy(case))
        done = False
        trace, perf, compute, violation = [],0.,0.,0.
        while not done:
            before = time.perf_counter()
            a = model.predict(obs,deterministic=True)[0] if fixed_horizon is None else np.array([fixed_horizon],dtype=float)
            obs,r,done,info = env.step(a)
            elapsed = time.perf_counter()-before
            perf += info['reward/performance']; compute += info['reward/computation']; violation += info['reward/constraint']
            trace.append({'state':copy.deepcopy(env.control_system.current_state),'horizon':info['executed_horizon'],
                'reward':r,'performance':info['reward/performance'],'compute':info['reward/computation'],
                'constraint':info['reward/constraint'],'elapsed_s':elapsed,'solver_success':info['solver_success'],
                'input':copy.deepcopy(env.control_system.controller.current_input)})
        row = {'episode':j,'steps':len(trace),'termination':info.get('termination'),
               'performance_cost':perf,'computation_cost':compute,'constraint_cost':violation,'total_cost':perf+compute+violation,
               'mean_horizon':float(np.mean([r['horizon'] for r in trace])),
               'solver_failure_steps':sum(not r['solver_success'] for r in trace),
               'mean_step_s':float(np.mean([r['elapsed_s'] for r in trace]))}
        rows.append(row)
        write(out/('trace_%02d.json'%j),trace)
    write(out/'summary.json',{'episodes':rows,'mean_total_cost':float(np.mean([r['total_cost'] for r in rows])),
        'constraint_episodes':sum(r['termination']=='constraint' for r in rows),
        'goal_episodes':sum(r['termination']=='goal' for r in rows),'terminal_value':use_value})
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',choices=['pendulum','vehicle'],required=True)
    ap.add_argument('--steps',type=int,default=15000)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--fixed-horizon',type=int)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--eval-episodes',type=int,default=10)
    ap.add_argument('--aligned',action='store_true')
    ap.add_argument('--scaled-obs',action='store_true')
    ap.add_argument('--ent-coef',default='auto')
    ap.add_argument('--no-online-value',action='store_true')
    ap.add_argument('--test-bank',type=Path)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--buffer-size',type=int,default=50000)
    args=ap.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    # A second launcher waits and then reuses the completed result, never trains
    # the same job concurrently. File lifetime intentionally spans this call.
    run_lock=open(out/'run.lock','a')
    fcntl.flock(run_lock.fileno(),fcntl.LOCK_EX)
    if (out/'completed.json').exists():
        print('Already complete:',out);return
    start=time.perf_counter()
    write(out/'running.json',{'pid':os.getpid(),'task':args.task,'steps':args.steps,'seed':args.seed,'fixed_horizon':args.fixed_horizon})
    Env,SAC,Policy=imports()
    install_correct_nstep()
    test = json.loads(args.test_bank.read_text()) if args.test_bank else bank(args.task)
    np.random.seed(args.seed)
    env=make_env(args.task,args.seed,args.fixed_horizon,aligned=args.aligned,scaled_obs=args.scaled_obs)
    n_state=env.control_system.controller.mpc.model.n_x
    n_param=env.control_system.controller.mpc.model.n_p-1
    model=SAC(Policy,env,gamma=.97,learning_rate=3e-4,buffer_size=args.buffer_size,
        learning_starts=100,batch_size=args.batch_size,train_freq=1,gradient_steps=1,tau=.005,
        reward_scale=.6 if args.task=='pendulum' else .3,ent_coef='auto' if args.ent_coef=='auto' else float(args.ent_coef),time_aware=True,
        seed=args.seed,n_cpu_tf_sess=1,verbose=0,
        policy_kwargs={'layers':{'pi':[32,32],'qf':[256,256],'vf':[256,256],'mpc':[]},
            'mpc_state_dim':n_state,'mpc_parameter_dim':n_param,'mpc_gamma':.97,
            'use_mpc_value_fn':True,'train_mpc_value_fn':True,'use_mpc_vf_target':False,'mpc_vf_type':'poly'})
    initial_hash=weights_hash(model)
    updates=[0]
    original_train_step=model._train_step
    def counted_train_step(*a,**kw):
        value=original_train_step(*a,**kw)
        updates[0]+=1
        if updates[0]==1:
            write(out/'first_update.json',{'cycle':model.num_timesteps,'weights_changed':weights_hash(model)!=initial_hash})
        return value
    model._train_step=counted_train_step
    spec={'task':args.task,'seed':args.seed,'steps':args.steps,'fixed_horizon':args.fixed_horizon,
          'test_bank_sha256':hashlib.sha256((args.test_bank or ART/'configs'/(args.task+'_test_bank.json')).read_bytes()).hexdigest(),
          'adaptations':{'aligned':args.aligned,'scaled_obs':args.scaled_obs,'ent_coef':args.ent_coef,'no_online_value':args.no_online_value,
                         'batch_size':args.batch_size,'buffer_size':args.buffer_size},
          'config_sha256':hashlib.sha256((ART/'configs'/(args.task+'.json')).read_bytes()).hexdigest(),
          'initial_hash':initial_hash,'observation_dim':env.observation_space.shape,'terminal_state_dim':n_state,'terminal_parameter_dim':n_param,
          'reward_scaling':'Author SAC divides reward by 0.6 (pendulum) or 0.3 (vehicle).',
          'fidelity':'Author core algorithms; missing original configs/test sets reconstructed; audited fixes applied.'}
    write(out/'manifest.json',spec)
    episode_rows=[]; current={'steps':0,'total_cost':0.,'solver_failures':0}
    def callback(loc,glob):
        info=model.info[0]
        current['steps']+=1;current['total_cost']+=sum(info.get('reward/'+k,0.) for k in ['performance','computation','constraint'])
        current['solver_failures']+=int(not info['solver_success'])
        if info.get('termination'):
            episode_rows.append(dict(current,termination=info['termination']))
            current.update(steps=0,total_cost=0.,solver_failures=0)
        w,b=model.policy_tf.get_mpc_vfn_weights_and_biases()
        if args.no_online_value:w,b=[np.zeros_like(a) for a in w],[np.zeros_like(a) for a in b]
        env.set_value_function_weights_and_biases(w,b)
        if model.num_timesteps%100==0:
            progress={'steps':model.num_timesteps,'target':args.steps,'updates_before_current':updates[0],
                      'episodes':len(episode_rows),'elapsed_s':time.perf_counter()-start,
                      'last_episode':episode_rows[-1] if episode_rows else None}
            write(out/'progress.json',progress)
            print(json.dumps(progress),flush=True)
        if model.num_timesteps%2500==0:
            model.save(str(out/('checkpoint_%05d'%model.num_timesteps)))
            write(out/'training_episodes.json',episode_rows)
        return True
    model.learn(args.steps,callback=callback)
    model.save(str(out/'model'))
    env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases())
    env.save_value_function(str(out),'terminal')
    final_hash=weights_hash(model)
    write(out/'training_episodes.json',episode_rows)
    before=weights_hash(model)
    for enabled in [True,False]:
        ev=out/('eval_value' if enabled else 'eval_no_value');ev.mkdir(exist_ok=True)
        evaluate(model,env,test['cases'][:args.eval_episodes],ev,args.fixed_horizon,enabled)
    assert weights_hash(model)==before, 'Evaluation changed model weights'
    result={'status':'complete','steps':args.steps,'elapsed_s':time.perf_counter()-start,
        'initial_hash':initial_hash,'final_hash':final_hash,'weights_changed':initial_hash!=final_hash,
        'train_episodes':len(episode_rows),'updates':updates[0],'evaluation_frozen':True,'test_episodes':args.eval_episodes,
        'reproduction_level':'core author code with reconstructed experiment configuration'}
    write(out/'completed.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        raise
