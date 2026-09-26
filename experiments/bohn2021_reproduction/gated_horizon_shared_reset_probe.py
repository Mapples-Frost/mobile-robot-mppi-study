"""Smoke-check shared starts across all independent fixed terminals before new validation."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from gated_horizon_search import OUT,TASKS,freeze
from gated_horizon_policy import BASE
from gated_horizon_evaluate import grid_model
from gated_horizon_shared_reset import install as install_shared
from conservative_canonical_reset import make_env
from conservative_solver_recovery import install as install_recovery,array
from conservative_fixed_log_audit import integrate
from branch_calibration_run import meter,observed_step
from runtime import imports
from paper_h_soft_probe import read,digest
from run import write,serial,weights_hash

DEST=OUT/'shared_initialization_amendment/probe'


def register():
    freeze();DEST.mkdir(exist_ok=True)
    files=[Path(__file__).resolve(),Path(__file__).with_name('gated_horizon_shared_reset.py').resolve(),OUT/'inputs_sha256.json',OUT/'shared_initialization_amendment/decision.json']
    for task in TASKS:
        files.append(OUT/'banks'/(task+'_smoke_bank.json'))
        for h in range(5,51,5):files += [grid_model(task,h,0)/n for n in ('model.zip','manifest.json','completed.json')]
    value=dict(hashes={str(p):digest(p) for p in files},seeds=[0],horizons=list(range(5,51,5)),cases=2,
        purpose='Existing smoke scenarios only. Compare post-reset plant state, previous input, full primal guess and warmup terminal parameters exactly across10 independently trained terminal models. Verify restoration and use of own terminal for scored step. Primary full trajectories must reproduce previous fixed smoke exactly.',
        test_access=False,validation_access=False)
    p=DEST/'registration.json'
    if p.exists():assert read(p)==value
    else:write(p,value)


def record():
    register();_,SAC,_=imports();normalize=lambda x:json.loads(json.dumps(x,default=serial))
    for task in TASKS:
        cases=read(OUT/'banks'/(task+'_smoke_bank.json'))['cases']
        for h in [BASE[task]]+[h for h in range(5,51,5) if h!=BASE[task]]:
            folder=DEST/('%s_h%d'%(task,h));folder.mkdir(exist_ok=True)
            if (folder/'completed.json').exists():
                for p,v in read(folder/'completed.json')['hashes'].items():assert digest(Path(p))==v
                continue
            assert not (folder/'solver_attempts.json').exists(),'Inspect interrupted initialization probe first'
            source=grid_model(task,h,0);model=SAC.load(str(source/'model.zip'))
            assert weights_hash(model)==read(source/'completed.json')['final_hash'];own=model.policy_tf.get_mpc_vfn_weights_and_biases();model.sess.close()
            env=make_env(task,0,aligned=True,scaled_obs=True);meter(env,folder);env.set_value_function_weights_and_biases(*own)
            mpc=env.control_system.controller.mpc;own_weights=array(mpc.vf.weights_num).copy();own_biases=array(mpc.vf.biases_num).copy()
            descriptor=install_shared(env,task,0,own,independent_terminal=h!=BASE[task]);recovery=install_recovery(mpc,folder);hashes={}
            for cid,case in enumerate(cases):
                recovery.update(enabled=False,events=[],case=cid,step=-1);env.reset(**copy.deepcopy(case))
                warmup=dict(state=normalize(env.control_system.current_state),input=normalize(env.control_system.controller.current_input),pre_state=case['state'])
                arrays=dict(warmup_primal=array(mpc.opt_x_num).copy(),warmup_weights=array(mpc.opt_p_num['_vf_weights']).copy(),warmup_biases=array(mpc.opt_p_num['_vf_biases']).copy(),
                    own_weights=own_weights,own_biases=own_biases,restored_weights=array(mpc.vf.weights_num).copy(),restored_biases=array(mpc.vf.biases_num).copy())
                np.testing.assert_array_equal(arrays['own_weights'],arrays['restored_weights']);np.testing.assert_array_equal(arrays['own_biases'],arrays['restored_biases'])
                if h!=BASE[task]:
                    ref=DEST/('%s_h%d'%(task,BASE[task]));assert warmup==read(ref/('case_%02d.json'%cid))['warmup']
                    with np.load(ref/('case_%02d.npz'%cid),allow_pickle=False) as saved:
                        for k in ('warmup_primal','warmup_weights','warmup_biases'):np.testing.assert_array_equal(arrays[k],saved[k])
                recovery['enabled']=True;trace=[]
                for t in range(env.max_steps if h==BASE[task] else 1):
                    recovery['step']=t;_,done,row=observed_step(env,task,h,case,t)
                    np.testing.assert_array_equal(array(mpc.opt_p_num['_vf_weights']),own_weights);np.testing.assert_array_equal(array(mpc.opt_p_num['_vf_biases']),own_biases)
                    if h==BASE[task]:
                        old=read(OUT/'smoke'/('%s_s0'%task)/'fixed'/('r0_trace_%02d.json'%cid))[t]
                        assert row=={k:old[k] for k in row},'Primary scored trajectory changed'
                    row['recovery']=recovery['events'][-1];trace.append(row)
                    if done:break
                if h==BASE[task]:assert done
                p=folder/('case_%02d.json'%cid);write(p,dict(warmup=warmup,trace=trace,primary_full_replay=h==BASE[task]));hashes[str(p)]=digest(p)
                p=folder/('case_%02d.npz'%cid);np.savez(p,**arrays);hashes[str(p)]=digest(p)
            for p in [folder/'solver_attempts.json',folder/'solver_calls.jsonl']+list(folder.glob('attempt_*.json')):hashes[str(p)]=digest(p)
            write(folder/'completed.json',dict(passed=True,descriptor=descriptor,hashes=hashes));print(folder.name+' shared-start and own-terminal checks passed',flush=True)


def audit():
    register();rows=[];hashes={};maximum=0.;recorded_steps=0;integrations=0
    for task in TASKS:
        for h in range(5,51,5):
            folder=DEST/('%s_h%d'%(task,h));done=read(folder/'completed.json');assert done['passed']
            for p,v in done['hashes'].items():assert digest(Path(p))==v
            counts=read(folder/'solver_attempts.json');calls=[json.loads(l) for l in (folder/'solver_calls.jsonl').read_text().splitlines()]
            assert len(calls)==counts['solve_attempts']==counts['solve_completed'];assert counts['warmup_attempts']==2
            assert sum(c['kind'].startswith('retry') for c in calls)==counts['retry_attempts']
            cursor=0;steps=0
            for cid in range(2):
                d=read(folder/('case_%02d.json'%cid));ref=read(DEST/('%s_h%d'%(task,BASE[task]))/('case_%02d.json'%cid));assert d['warmup']==ref['warmup']
                with np.load(folder/('case_%02d.npz'%cid),allow_pickle=False) as a,np.load(DEST/('%s_h%d'%(task,BASE[task]))/('case_%02d.npz'%cid),allow_pickle=False) as b:
                    for k in ('warmup_primal','warmup_weights','warmup_biases'):np.testing.assert_array_equal(a[k],b[k])
                    for suffix in ('weights','biases'):np.testing.assert_array_equal(a['own_'+suffix],a['restored_'+suffix])
                assert calls[cursor]['kind']=='warmup' and calls[cursor]['case']==cid;cursor+=1
                transitions=[(d['warmup']['pre_state'],d['warmup']['input'],d['warmup']['state'])]
                for r in d['trace']:
                    attempts=r['recovery']['attempts'];assert calls[cursor:cursor+len(attempts)]==attempts;cursor+=len(attempts)
                    transitions.append((r['previous_state'],r['input'],r['state']))
                for before,control,after in transitions:
                    for key,value in integrate(task,before,control).items():
                        np.testing.assert_allclose(after[key],value,rtol=1e-7,atol=1e-7);maximum=max(maximum,abs(after[key]-value))
                    integrations+=1
                steps+=len(d['trace'])
            assert cursor==len(calls)
            attempts=[read(p) for p in folder.glob('attempt_*.json')];assert sum(a['step_calls'] for a in attempts)==steps and sum(a['reset_calls'] for a in attempts)==2
            recorded_steps+=steps;rows.append(dict(task=task,h=h,steps=steps,counts=counts));hashes[str(folder/'completed.json')]=digest(folder/'completed.json')
    write(DEST/'audit.json',dict(passed=True,conditions=20,resets=40,steps=recorded_steps,offline_integrations=integrations,maximum_dynamics_error=maximum,groups=rows,hashes=hashes,
        scope='Existing smoke cases only; common warmup state/input/primal exactly verified, own terminal restored and used, original primary full trajectories preserved, reset and scored plant transitions independently integrated. Not validation/test effects.'))
    print(json.dumps(dict(passed=True,conditions=20,resets=40,steps=recorded_steps,offline_integrations=integrations,maximum_dynamics_error=maximum),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['record','audit'],required=True);a=ap.parse_args()
    if a.mode=='record':record()
    else:audit()
