"""Deterministic endpoint-H replay across every available task/seed/round."""
import copy
import gc
import json
from pathlib import Path
from conservative_iteration import OUT,TASKS,BASE,BLOCK,verify,prior_model,action
from conservative_canonical_reset import make_env
from runtime import imports
from min_q_eval_suite import model_dir
from paper_h_soft_probe import read,digest
from run import write,weights_hash
from branch_calibration_run import meter,observed_step
from relative_policy_features import context
from conservative_reset_diagnosis import differences


def main():
    verify();dest=OUT/'reset_diagnosis/replay';dest.mkdir(exist_ok=True)
    results=[]
    for task in TASKS:
        for seed in range(3):
            folder=dest/('%s_s%d'%(task,seed));folder.mkdir(exist_ok=True)
            env=make_env(task,seed,aligned=True,scaled_obs=True);meter(env,folder)
            _,SAC,_=imports();source=model_dir(task,'fixed',seed);model=SAC.load(str(source/'model.zip'))
            assert weights_hash(model)==read(source/'completed.json')['final_hash']
            env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases());model.sess.close()
            for round_id in range(2):
                original=OUT/('%s_s%d_r%d'%(task,seed,round_id))
                for h in (5,50):
                    path=original/('case00_t000_h%02d.json'%h)
                    if not path.exists():continue
                    case=read(original/'train_bank.json')['cases'][0];expected=read(path)['trace']
                    prior=prior_model(task,seed,round_id)
                    env.reset(**copy.deepcopy(case));actual=[]
                    for t,row in enumerate(expected):
                        ctx=context(env,task)
                        selected=h if t<BLOCK else action(task,ctx,prior) if t%BLOCK==0 else selected
                        _,done,new=observed_step(env,task,selected,case,t);new['policy_context']=ctx;actual.append(new)
                        if done:break
                    same=actual==expected
                    record=dict(task=task,seed=seed,round=round_id,h=h,steps=len(actual),exact=same,
                        original_hash=digest(path),first_difference=None if same else next((dict(t=t,differences=differences(a,b)[:10]) for t,(a,b) in enumerate(zip(actual,expected)) if a!=b),dict(lengths=[len(actual),len(expected)])))
                    results.append(record);write(folder/('r%d_h%d_trace.json'%(round_id,h)),actual)
                    write(dest/'progress.json',dict(complete=False,results=results));print(json.dumps({k:v for k,v in record.items() if k!='first_difference'}),flush=True)
            if task=='vehicle' and seed==1:
                original=OUT/'vehicle_s1_r1';case=read(original/'train_bank.json')['cases'][4]
                expected=read(original/'source_04.json');prior=prior_model(task,seed,1)
                for repeat in range(2):
                    env.reset(**copy.deepcopy(case));actual=[]
                    for t,row in enumerate(expected):
                        ctx=context(env,task)
                        if t%BLOCK==0:selected=action(task,ctx,prior)
                        _,done,new=observed_step(env,task,selected,case,t);new['policy_context']=ctx;actual.append(new)
                        if done:break
                    same=actual==expected;results.append(dict(task=task,seed=seed,case=4,source_repeat=repeat,steps=len(actual),exact=same))
                    write(folder/('failed_source_repeat%d.json'%repeat),actual)
            del model,env;gc.collect()
    write(dest/'completed.json',dict(complete=True,all_exact=all(r['exact'] for r in results),results=results,
        source_hash=digest(Path(__file__)),adapter_hash=digest(Path(__file__).with_name('conservative_canonical_reset.py')),
        scope='Deterministic case0/anchor0 extreme H5/H50 from every available task/seed/round plus twice-repeated original failing source. Other alternative branches not replayed by this diagnostic.'))
    assert all(r['exact'] for r in results),'Preserve mismatch records; do not resume collection'


if __name__=='__main__':main()
