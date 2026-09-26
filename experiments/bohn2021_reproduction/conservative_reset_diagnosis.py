"""Diagnose exact initial-context disagreement without changing collection rules."""
import copy
import json
from conservative_iteration import OUT, verify
from runtime import make_env, imports
from min_q_eval_suite import model_dir
from run import weights_hash, write, serial
from paper_h_soft_probe import read, digest
from relative_policy_features import context
from branch_calibration_run import meter
from branch_calibration_run import observed_step


def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return sum((differences(a[k],b[k],path+'/'+k) for k in a),[])
    if isinstance(a,list) and isinstance(b,list):
        return sum((differences(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))),[])
    return [] if a==b else [dict(path=path,actual=a,expected=b)]


def main():
    verify();folder=OUT/'vehicle_s1_r1';case=read(folder/'train_bank.json')['cases'][4]
    expected=read(folder/'source_04.json')[0]['policy_context']
    dest=OUT/'reset_diagnosis';dest.mkdir(exist_ok=True)
    env=make_env('vehicle',1,aligned=True,scaled_obs=True);meter(env,dest)
    _,SAC,_=imports();source=model_dir('vehicle','fixed',1);model=SAC.load(str(source/'model.zip'))
    assert weights_hash(model)==read(source/'completed.json')['final_hash']
    env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases());model.sess.close()
    results=[]
    for repeat in range(2):
        env.reset(**copy.deepcopy(case));actual=context(env,'vehicle')
        results.append(dict(repeat=repeat,actual=actual,differences=differences(actual,expected)))
    trace=read(folder/'source_04.json');replayed=[]
    for t,row in enumerate(trace):
        ctx=context(env,'vehicle')
        _,done,actual=observed_step(env,'vehicle',row['horizon'],case,t)
        actual['policy_context']=ctx
        assert actual==row,('source_replay',t,differences(actual,row)[:10])
        replayed.append(actual)
    assert done
    env.reset(**copy.deepcopy(case));actual=context(env,'vehicle')
    results.append(dict(repeat='after_source_replay',actual=actual,differences=differences(actual,expected)))
    env.control_system.controller.mpc.opt_x_num['_eps']=0
    env.reset(**copy.deepcopy(case));actual=context(env,'vehicle')
    results.append(dict(repeat='after_zero_slack',actual=actual,differences=differences(actual,expected)))
    write(dest/'source_replay.json',replayed)
    write(dest/'context_comparison.json',dict(task='vehicle',seed=1,round=1,case=4,anchor=0,
        expected=expected,results=results,source_trace_hash=digest(folder/'source_04.json'),
        scenario_bank_hash=digest(folder/'train_bank.json'),script_hash=digest(__import__('pathlib').Path(__file__))))
    print(json.dumps([dict(repeat=r['repeat'],differences=r['differences']) for r in results],indent=2,default=serial))


if __name__=='__main__':main()
