"""Evaluate saved RL checkpoints on the frozen bank, without further learning."""
import argparse
import json
from runtime import ART,imports,make_env
from run import evaluate,weights_hash,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    folder=ART/'results/full'/(args.task+'_rl_s'+str(args.seed))
    assert (folder/'completed.json').exists(),'Training must finish before post hoc checkpoint evaluation'
    _,SAC,_=imports();cases=json.loads((ART/'configs'/(args.task+'_test_bank.json')).read_text())['cases']
    out=folder/'learning_curve';out.mkdir(exist_ok=True)
    for step in range(2500,15000,2500):
        dest=out/str(step);dest.mkdir(exist_ok=True)
        if (dest/'completed.json').exists():continue
        env=make_env(args.task,9100+args.seed)
        model=SAC.load(str(folder/('checkpoint_%05d.zip'%step)),env=env)
        before=weights_hash(model)
        evaluate(model,env,cases,dest,None,True)
        assert before==weights_hash(model)
        write(dest/'completed.json',{'nominal_training_step':step,'updates':step-100,
             'weights_frozen':True,'checkpoint_note':'Callback checkpoint is saved before the current-step update; final evaluation uses model.zip after all updates.'})
        model.sess.close()
    write(out/'completed.json',{'steps':[2500,5000,7500,10000,12500,15000],
        'final_point':'../eval_value/summary.json','test_episodes_per_point':10,
        'evaluation_only':True,'no_training_or_model_selection':True})
    print(json.dumps({'task':args.task,'seed':args.seed,'curve':'complete'}))


if __name__=='__main__':main()
