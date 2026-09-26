"""Post-training validation-only causal diagnostics; not a new candidate policy."""
import json
from pathlib import Path
import numpy as np
from runtime import ART, imports
from run import write, weights_hash
from optimized_runtime import install_terminal, make_env
from optimized_evaluate import evaluate

OUT=ART/'results/optimized'


def main():
    install_terminal('pendulum');_,SAC,_=imports()
    cases=json.loads((ART/'configs/pendulum_validation_bank.json').read_text())['cases']
    dest=OUT/'pendulum_interventions';dest.mkdir(exist_ok=True)
    write(dest/'protocol.json',{'scope':'Diagnostic original validation bank only. No policy training or holdout tuning.',
        'arms':['same RL terminal, force H30','same RL terminal, clamp H>=20','same RL terminal, clamp H>=25'],
        'seeds':[0,1,2],'cases':10,'warning':'Floors are diagnostic interventions, not claims of newly reproduced paper method.'})
    results=[]
    for seed in range(3):
        source=OUT/('pendulum_rl_s%d'%seed)
        model=SAC.load(str(source/'model.zip'));before=weights_hash(model);predict=model.predict
        for mode,floor,fixed in [('force30',None,30),('floor20',20,None),('floor25',25,None)]:
            folder=dest/('s%d_%s'%(seed,mode));folder.mkdir(exist_ok=True)
            if not (folder/'completed.json').exists():
                def policy(obs,deterministic=True):
                    action,state=predict(obs,deterministic=deterministic)
                    return (np.maximum(action,floor) if floor is not None else action),state
                model.predict=policy
                env=make_env('pendulum',100+seed)
                evaluate(model,env,cases,folder,fixed,True)
                assert weights_hash(model)==before
                write(folder/'completed.json',{'frozen':True,'weights_sha256':before})
            s=json.loads((folder/'summary.json').read_text())
            row={'seed':seed,'mode':mode,'cost':s['mean_total_cost'],'constraints':s['constraint_episodes']}
            results.append(row);print(json.dumps(row),flush=True)
        model.sess.close()
    write(dest/'summary.json',{'results':results,'purpose':'Diagnosis only; all original benchmark arms retained.'})


if __name__=='__main__':main()
