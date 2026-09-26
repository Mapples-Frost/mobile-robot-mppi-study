"""Frozen causal diagnostics. No retraining, test tuning or author-source edits."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from runtime import ART, ROOT, imports, make_env
from run import snapshot, write, weights_hash
import numpy as np

OUT = ART / 'results/mechanism_probe'
HORIZONS = list(range(5, 51, 5))
BRANCH_HORIZONS = [5, 10, 15, 25, 40]
ANCHORS = [0, 20, 40, 60]


def configure_value(env, model=None, mode='zero'):
    mpc = env.control_system.controller.mpc
    nx, nd = mpc.model.n_x, mpc.model.n_x + mpc.model.n_p - 1
    if model is None:
        w, b = [np.zeros((2 * nd, 1))], [np.zeros(1)]
    else:
        w, b = model.policy_tf.get_mpc_vfn_weights_and_biases()
    if mode == 'zero':
        w, b = [np.zeros_like(v) for v in w], [np.zeros_like(v) for v in b]
    elif mode == 'no_position':
        assert nx == 3 and nd == 5
        w[0][[1, 2, nd + 1, nd + 2]] = 0
    elif mode == 'no_heading':
        assert nx == 3 and nd == 5
        w[0][[0, nd]] = 0
    env.set_value_function_weights_and_biases(w, b)
    return w, b


def fresh(task, case, model=None, mode='zero', noise_scale=1.):
    env = make_env(task, 26091703, aligned=True, scaled_obs=True)
    # Author reset applies one physical H50 control. All interventions must
    # share that initial state and solver warm start before their effects begin.
    configure_value(env)
    case = copy.deepcopy(case)
    if task == 'vehicle':
        case['reference']['ns'] = (np.asarray(case['reference']['ns']) * noise_scale).tolist()
    obs = env.reset(**case)
    configure_value(env, model, mode)
    return env, obs


def checked_step(env, task, horizon):
    obs, reward, done, info = env.step(np.array([float(horizon)]))
    s = copy.deepcopy(env.control_system.current_state)
    u = {k: float(np.asarray(v).ravel()[0]) for k, v in env.control_system.controller.current_input.items()}
    def ref(name):
        return float(np.asarray(env.control_system.tvps[name].get_values(env.control_system._step_count)).ravel()[0])
    if task == 'vehicle':
        perf = (s['x']-ref('trajectory_x'))**2+(s['y']-ref('trajectory_y'))**2
        assert -1e-5 <= u['u_s'] <= 5+1e-5 and abs(u['u_omega']) <= 4+1e-5
        violation = any(np.hypot(s['x']-ref('obj_%d_x'%j), s['y']-ref('obj_%d_y'%j)) <= ref('obj_%d_r'%j) for j in range(3))
    else:
        perf = .4*s['v']**2 + .05*s['v']*s['omega']*np.cos(s['theta']) + (2/3)*.2*.25**2*s['omega']**2 - .2*9.81*.25*np.cos(s['theta']) + 10*(s['pos']-ref('pos_r'))**2 + .1*u['u1']**2
        assert abs(u['u1']) <= 5+1e-5
        violation = abs(s['pos']) > 1.5 or abs(s['theta']) > np.pi/2
    compute = horizon * (.001 if task == 'vehicle' else .003)
    penalty = (2 if task == 'vehicle' else 10)*(env.max_steps-env.steps_count) if violation else 0.
    assert np.isclose(perf, info['reward/performance'], atol=1e-8)
    assert np.isclose(compute, info['reward/computation'])
    assert np.isclose(penalty, info['reward/constraint'])
    assert np.isclose(-reward, perf+compute+penalty)
    if violation:
        assert done and info['termination'] == 'constraint'
    return obs, done, {'state': s, 'input': u, 'horizon': horizon, 'cost': -reward,
        'performance': perf, 'compute': compute, 'constraint': penalty,
        'solver_success': info['solver_success'], 'termination': info.get('termination'),
        'terminal_input': info['data']['mpc_state'].tolist(),
        'stage_cost': info['data']['mpc_rewards']}


def rollout(task, case, horizon=None, model=None, mode='zero', noise_scale=1.):
    env, obs = fresh(task, case, model, mode, noise_scale)
    initial = copy.deepcopy(env.control_system.current_state)
    trace = []
    while True:
        h = horizon if horizon is not None else int(np.clip(np.rint(model.predict(obs, deterministic=True)[0][0]), 1, 50))
        obs, done, row = checked_step(env, task, h)
        trace.append(row)
        if done:
            break
    costs = np.array([r['cost'] for r in trace])
    return {'initial_state': initial, 'trace': trace, 'total_cost': float(costs.sum()),
        'discounted_cost': float(costs @ (.97**np.arange(len(costs)))),
        'termination': trace[-1]['termination'], 'steps': len(trace)}


def worker(task, case_id):
    start = time.monotonic()
    folder = OUT / ('%s_case%02d' % (task, case_id))
    folder.mkdir(exist_ok=True)
    if (folder/'completed.json').exists():
        return
    case = json.loads((OUT/(task+'_bank.json')).read_text())['cases'][case_id]
    _, SAC, _ = imports()
    models = [SAC.load(str(ART/'results/paper_defaults'/('%s_rl_s%d'%(task, seed))/'model.zip')) for seed in range(3)]
    hashes = [weights_hash(m) for m in models]
    sweeps = {}
    modes = ['zero', 'learned', 'no_position', 'no_heading'] if task == 'vehicle' else ['zero', 'learned']
    jobs = [('h%d'%h, h, None, 'zero', 1.) for h in HORIZONS]
    jobs += [('rl%d_%s'%(seed, mode), None, m, mode, 1.) for seed, m in enumerate(models) for mode in modes]
    if task == 'vehicle':
        jobs += [('exact_forecast_h%d'%h, h, None, 'zero', 0.) for h in HORIZONS]
    for name, h, model, mode, noise in jobs:
        path = folder/(name+'.json')
        if path.exists():
            result = json.loads(path.read_text())
        else:
            result = rollout(task, case, h, model, mode, noise)
            write(path, result)
        sweeps[name] = {k: v for k, v in result.items() if k != 'trace'}
        write(folder/'progress.json', {'phase': 'sweep', 'finished': len(sweeps), 'target':len(jobs), 'elapsed_s':time.monotonic()-start})
    # Full remaining-episode returns from identical H10/H30 prefix states.
    # Only the first horizon changes; all subsequent decisions use the base H.
    # This uses known future test information and is diagnostic, not deployable.
    base_h = 10 if task == 'vehicle' else 30
    base = json.loads((folder/('h%d.json'%base_h)).read_text())
    branches = []
    for anchor in ANCHORS:
        if anchor >= base['steps']:
            continue
        probe_env, obs = fresh(task, case)
        for t in range(anchor):
            obs, done, row = checked_step(probe_env, task, base_h)
            assert not done
            assert row['state'] == base['trace'][t]['state'], 'Prefix replay is not exact'
        choices = [int(np.clip(np.rint(m.predict(obs, deterministic=True)[0][0]), 1, 50)) for m in models]
        candidates = sorted(set((BRANCH_HORIZONS if task == 'vehicle' else [10,20,30,40,50]) + [base_h] + choices))
        row = {'anchor':anchor, 'actor_horizons':choices, 'state':copy.deepcopy(probe_env.control_system.current_state), 'candidates':[]}
        for h in candidates:
            path = folder/('branch_t%03d_h%02d.json'%(anchor,h))
            if path.exists():
                result=json.loads(path.read_text())
            else:
                env, obs = fresh(task, case)
                for t in range(anchor):
                    obs, done, trace_row = checked_step(env, task, base_h)
                    assert not done
                    assert trace_row['state'] == base['trace'][t]['state'], 'Candidate prefix is not exact'
                assert env.control_system.current_state == row['state']
                trace=[]
                while True:
                    obs, done, trace_row=checked_step(env, task, h if not trace else base_h)
                    trace.append(trace_row)
                    if done:break
                costs=np.array([r['cost'] for r in trace])
                result={'horizon':h,'trace':trace,'total_cost':float(costs.sum()),
                    'discounted_cost':float(costs @ (.97**np.arange(len(costs)))),
                    'termination':trace[-1]['termination']}
                if h==base_h:
                    assert trace==base['trace'][anchor:], 'Baseline suffix is not exact'
                write(path,result)
            row['candidates'].append({k:v for k,v in result.items() if k!='trace'})
        branches.append(row)
        write(folder/'branches.json',branches)
        write(folder/'progress.json',{'phase':'branch','anchors_finished':len(branches),'elapsed_s':time.monotonic()-start})
    assert hashes == [weights_hash(m) for m in models]
    write(folder/'completed.json',{'task':task,'case':case_id,'sweeps':sweeps,'branches':branches,
        'weights_frozen':True,'exact_prefix_verified':True,'costs_independently_verified':True,
        'elapsed_s':time.monotonic()-start})
    for model in models:model.sess.close()


def prepare():
    OUT.mkdir(exist_ok=True)
    protocol = {'research_question':'Does the performance gap arise from terminal feedback, absent horizon headroom, or learned horizon decisions?',
        'mode':'run','provenance':'Frozen paper_defaults models; new diagnostic scenes; no training or checkpoint selection.',
        'seed':26091703,'scenes_per_task':10,'fixed_horizons':HORIZONS,'anchors':ANCHORS,
        'terminal_interventions':['zero','learned','vehicle: remove x/y coefficients only','vehicle: remove theta coefficients only'],
        'reset_control':'All arms use zero terminal for reset warmup; interventions start after reset. Paired initial state is identical.',
        'forecast_intervention':'Vehicle: zero stored obstacle forecast perturbations; preserve actual geometry and trajectory.',
        'branch_intervention':'Common prefix generated by zero-terminal H10 vehicle / H30 pendulum. Change H for one decision, then return to baseline until termination.',
        'branch_candidates':{'vehicle':BRANCH_HORIZONS,'pendulum':[10,20,30,40,50]},
        'additional_candidates':'Exact H choices of all 3 frozen RL actors at each shared state.',
        'outcomes':['total undiscounted episode cost','discounted .97 return','termination','solver status'],
        'limitations':['Branch best uses future information and is not an implementable learned policy.',
            'Single-decision branches do not bound the value of arbitrary adaptive sequences.',
            'Zero-terminal evaluation changes the environment seen during joint training.',
            '10 independent scenes per task; no statistical significance claim.',
            'Fresh MPC environment per episode excludes solver history between episodes.'],
        'hard_timeout_seconds_per_case':7200}
    write(OUT/'protocol.json',protocol)
    for task in ['vehicle','pendulum']:
        p=OUT/(task+'_bank.json')
        if p.exists():continue
        env=make_env(task,26091703,aligned=True,scaled_obs=True)
        np.random.seed(26091703)
        cases=[]
        for _ in range(10):
            env.reset();cases.append(snapshot(env))
        write(p,{'seed':26091703,'purpose':'New diagnostic bank, not used for fitting or checkpoint selection','cases':cases})
    write(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),ROOT/'experiments/bohn2021_reproduction/runtime.py',ROOT/'experiments/bohn2021_reproduction/run.py']})


def launch(tasks, count, workers):
    jobs=[(task,i) for task in tasks for i in range(count)]
    def run_job(job):
        task,i=job
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',task,'--case',str(i)]
        with open(OUT/('%s_case%02d.log'%(task,i)),'a') as log:
            p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
        result={'task':task,'case':i,'exit_code':p.returncode}
        print(json.dumps(result),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:results=list(pool.map(run_job,jobs))
    write(OUT/('launch_'+'_'.join(tasks)+'.json'),{'jobs':results,'complete':all(r['exit_code']==0 for r in results)})
    assert all(r['exit_code']==0 for r in results),results


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--worker',choices=['vehicle','pendulum'])
    ap.add_argument('--case',type=int,default=0)
    ap.add_argument('--tasks',nargs='+',default=['vehicle','pendulum'])
    ap.add_argument('--count',type=int,default=10)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    if args.worker:worker(args.worker,args.case)
    else:
        prepare();launch(args.tasks,args.count,args.workers)
