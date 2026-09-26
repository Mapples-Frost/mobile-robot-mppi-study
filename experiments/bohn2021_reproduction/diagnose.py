"""Frozen actor/value crossover; no training or checkpoint selection."""
import argparse
import json
import numpy as np
from runtime import ART, imports, make_env
from run import evaluate, write, weights_hash


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,required=True)
    args=ap.parse_args()
    _,SAC,_=imports()
    src=ART/'results/full'/('vehicle_rl_s%d'%args.seed)
    out=ART/'results/diagnosis'/('crossover_s%d'%args.seed)
    out.mkdir(parents=True,exist_ok=True)
    cases=json.loads((ART/'configs/vehicle_test_bank.json').read_text())['cases']
    models={}
    for name,filename in [('early','checkpoint_05000.zip'),('late','model.zip')]:
        models[name]=SAC.load(str(src/filename))
    values={name:m.policy_tf.get_mpc_vfn_weights_and_biases() for name,m in models.items()}
    coefs={name:{'weights':[w.tolist() for w in wb[0]],'biases':[b.tolist() for b in wb[1]]} for name,wb in values.items()}
    write(out/'coefficients.json',coefs)
    rows=[]
    for actor in ['early','late']:
        for value in ['early','late','zero']:
            dest=out/(actor+'_actor_'+value+'_value');dest.mkdir(exist_ok=True)
            if not (dest/'summary.json').exists():
                env=make_env('vehicle',9100+args.seed)
                before=weights_hash(models[actor])
                evaluate(models[actor],env,cases,dest,None,value!='zero',values.get(value))
                assert before==weights_hash(models[actor])
            summary=json.loads((dest/'summary.json').read_text())
            row={'actor':actor,'value':value,'mean_total_cost':summary['mean_total_cost'],
                 'goals':summary['goal_episodes'],'collisions':summary['constraint_episodes']}
            rows.append(row);print(json.dumps(row),flush=True)
    write(out/'completed.json',{'seed':args.seed,'early_step':5000,'late_step':15000,'rows':rows,
        'purpose':'Causal intervention on frozen checkpoints; not test-based checkpoint selection.'})
    for m in models.values():m.sess.close()


if __name__=='__main__':main()
