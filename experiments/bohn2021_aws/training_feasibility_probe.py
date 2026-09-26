"""Bounded AWS feasibility probe, NOT a formal reproduction experiment."""
import pathlib,sys,time,json,resource,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/bohn2021_reproduction'))
from runtime import imports,make_env,install_correct_nstep
from run import weights_hash,serial
OUT=ROOT/'research_artifacts/aws_diagnostics/training_feasibility_20260926';OUT.mkdir(parents=True,exist_ok=True)
Env,SAC,Policy=imports();install_correct_nstep();results=[]
def stats(values):
 return dict(n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p95=float(np.percentile(values,95))) if values else dict(n=0)
for task,h in [('vehicle',25),('pendulum',30)]:
 folder=OUT/task;folder.mkdir(exist_ok=True)
 if (folder/'result.json').exists():results.append(json.loads((folder/'result.json').read_text()));continue
 seed=2609268801;np.random.seed(seed);start=time.perf_counter();env=make_env(task,seed,fixed_horizon=h,aligned=True,scaled_obs=True)
 model=SAC(Policy,env,gamma=.97,learning_rate=3e-4,buffer_size=2000,learning_starts=100,batch_size=64,train_freq=1,gradient_steps=1,tau=.005,reward_scale=.6 if task=='pendulum' else .3,ent_coef='auto',time_aware=True,seed=seed,n_cpu_tf_sess=1,verbose=0,policy_kwargs={'layers':{'pi':[32,32],'qf':[256,256],'vf':[256,256],'mpc':[]},'mpc_state_dim':env.control_system.controller.mpc.model.n_x,'mpc_parameter_dim':env.control_system.controller.mpc.model.n_p-1,'mpc_gamma':.97,'use_mpc_value_fn':True,'train_mpc_value_fn':True,'use_mpc_vf_target':False,'mpc_vf_type':'poly'})
 initialization=time.perf_counter()-start;before=weights_hash(model);rows=[];updates=[];reset_times=[]
 original_step=env.step;original_update=model._train_step;original_reset=env.reset
 def step(action):
  t=time.perf_counter();out=original_step(action);elapsed=time.perf_counter()-t;obs,reward,done,info=out
  rows.append(dict(step=len(rows)+1,elapsed_s=elapsed,horizon=int(info['executed_horizon']),reward=float(reward),solver_success=bool(info['solver_success']),termination=info.get('termination'),finite=bool(np.all(np.isfinite(obs)) and np.isfinite(reward))))
  return out
 def reset(*a,**kw):
  t=time.perf_counter();out=original_reset(*a,**kw);reset_times.append(time.perf_counter()-t);return out
 def update(*a,**kw):
  t=time.perf_counter();out=original_update(*a,**kw);updates.append(time.perf_counter()-t);return out
 env.step=step;env.reset=reset;model._train_step=update
 def callback(loc,glob):
  env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases())
  if len(rows)%50==0:
   p=dict(task=task,steps=len(rows),updates=len(updates),elapsed_s=time.perf_counter()-training_start)
   (folder/'progress.json').write_text(json.dumps(p));print(json.dumps(p),flush=True)
  return True
 training_start=time.perf_counter();model.learn(200,callback=callback);elapsed=time.perf_counter()-training_start
 model.save(str(folder/'probe_model'));after=weights_hash(model)
 result=dict(task=task,seed=seed,fixed_horizon=h,steps=len(rows),updates=len(updates),weights_changed=before!=after,all_parameters_finite=all(np.all(np.isfinite(v)) for v in model.get_parameters().values()),all_steps_finite=all(r['finite'] for r in rows),solver_failure_steps=sum(not r['solver_success'] for r in rows),episode_terminations=[r['termination'] for r in rows if r['termination']],initialization_s=initialization,training_wall_s=elapsed,steps_per_second=len(rows)/elapsed,control_step_seconds=stats([r['elapsed_s'] for r in rows]),gradient_update_seconds=stats(updates),reset_seconds=stats(reset_times),first50_step_seconds=stats([r['elapsed_s'] for r in rows[:50]]),last50_step_seconds=stats([r['elapsed_s'] for r in rows[-50:]]),process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,training_only=True,validation_accessed=False,test_accessed=False,scope='200-step engineering feasibility only; no convergence or long-duration CPU-credit claim')
 (folder/'steps.json').write_text(json.dumps(rows,indent=2));(folder/'result.json').write_text(json.dumps(result,indent=2,default=serial));results.append(result);print(json.dumps(result,default=serial),flush=True)
 assert len(rows)==200 and len(updates)>0 and result['weights_changed'] and result['all_parameters_finite'] and result['all_steps_finite']
 model.sess.close()
 if hasattr(env,'close'):env.close()
(OUT/'summary.json').write_text(json.dumps(results,indent=2,default=serial))
