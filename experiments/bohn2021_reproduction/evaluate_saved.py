"""Evaluate a saved model using its manifest's observation/runtime options."""
import argparse
import fcntl
import json
from pathlib import Path
from runtime import imports,make_env
from run import evaluate,weights_hash,write

ap=argparse.ArgumentParser()
ap.add_argument('--model-dir',type=Path,required=True)
ap.add_argument('--bank',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--aligned',action='store_true')
ap.add_argument('--no-value',action='store_true')
args=ap.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
evaluation_lock=open(args.out/'evaluation.lock','a')
fcntl.flock(evaluation_lock.fileno(),fcntl.LOCK_EX)
if not (args.out/'completed.json').exists():
    spec=json.loads((args.model_dir/'manifest.json').read_text())
    adaptation=spec.get('adaptations',{})
    env=make_env(spec['task'],923,fixed_horizon=spec['fixed_horizon'],
        aligned=adaptation.get('aligned',False) or args.aligned,scaled_obs=adaptation.get('scaled_obs',False))
    _,SAC,_=imports()
    m=SAC.load(str(args.model_dir/'model.zip'),env=env)
    before=weights_hash(m)
    cases=json.loads(args.bank.read_text())['cases']
    evaluate(m,env,cases,args.out,spec['fixed_horizon'],not args.no_value)
    assert before==weights_hash(m)
    write(args.out/'completed.json',{'model_dir':str(args.model_dir),'bank':str(args.bank),
        'model_hash':before,'frozen':True,'aligned_override':args.aligned,'terminal_value':not args.no_value})
