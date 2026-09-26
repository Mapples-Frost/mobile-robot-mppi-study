"""Check every saved training initial context against zero-slack reset."""
import copy
import gc
import json
from pathlib import Path
from conservative_iteration import OUT,TASKS,verify
from runtime import make_env,imports
from min_q_eval_suite import model_dir
from paper_h_soft_probe import read,digest
from run import write,weights_hash
from branch_calibration_run import meter
from relative_policy_features import context
from conservative_reset_diagnosis import differences


def main():
    verify();dest=OUT/'reset_diagnosis/canonical_probe';dest.mkdir(exist_ok=True)
    results=[]
    for task in TASKS:
        for seed in range(3):
            attempt=dest/('%s_s%d'%(task,seed));attempt.mkdir(exist_ok=True)
            env=make_env(task,seed,aligned=True,scaled_obs=True);meter(env,attempt)
            _,SAC,_=imports();source=model_dir(task,'fixed',seed);model=SAC.load(str(source/'model.zip'))
            assert weights_hash(model)==read(source/'completed.json')['final_hash']
            env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases());model.sess.close()
            folders=[OUT/('%s_s%d_r%d'%(task,seed,r)) for r in range(2)]
            if seed==0:folders.append(OUT/('smoke_'+task))
            for folder in folders:
                if not (folder/'train_bank.json').exists():continue
                cases=read(folder/'train_bank.json')['cases']
                for path in sorted(folder.glob('source_*.json')):
                    cid=int(path.stem.split('_')[1]);expected=read(path)[0]['policy_context']
                    env.control_system.controller.mpc.opt_x_num['_eps']=0
                    env.reset(**copy.deepcopy(cases[cid]));actual=context(env,task)
                    result=dict(task=task,seed=seed,folder=folder.name,case=cid,
                        source_hash=digest(path),exact=actual==expected,differences=differences(actual,expected))
                    results.append(result)
                    write(dest/'progress.json',dict(results=results,complete=False))
                    print(json.dumps({k:result[k] for k in ('task','seed','folder','case','exact')}),flush=True)
            del model,env;gc.collect()
    write(dest/'completed.json',dict(complete=True,results=results,all_exact=all(r['exact'] for r in results),
        source_hash=digest(Path(__file__)),scope='Initial saved training contexts only; no validation/test and no newly trained parameters.'))


if __name__=='__main__':main()
