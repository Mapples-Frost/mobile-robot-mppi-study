"""Reload a saved policy and check that a frozen test case can be replayed."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
from runtime import ART, imports, make_env
from run import evaluate, weights_hash, write
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--bank',type=Path)
    p.add_argument('--evaluation-dir',default='eval_value')
    p.add_argument('--audit-out',type=Path)
    args=p.parse_args()
    spec=json.loads((args.run/'manifest.json').read_text())
    task=spec['task'];fixed=spec['fixed_horizon']
    _,SAC,_=imports()
    adaptation=spec.get('adaptations',{})
    kw={'aligned':adaptation.get('aligned',False),'scaled_obs':adaptation.get('scaled_obs',False)}
    env=make_env(task,9188,fixed,**kw)
    model=SAC.load(str(args.run/'model.zip'),env=env)
    before=weights_hash(model)
    case=json.loads((args.bank or ART/'configs'/(task+'_test_bank.json')).read_text())['cases'][0]
    trajectories=[];costs=[]
    with tempfile.TemporaryDirectory(prefix='bohn-replay-') as temp:
        for i in range(2):
            out=Path(temp)/str(i);out.mkdir()
            # Fresh environment excludes accidental solver history from prior episodes.
            fresh=make_env(task,9188,fixed,**kw)
            result=evaluate(model,fresh,[copy.deepcopy(case)],out,fixed,True)
            traces=json.loads((out/'trace_00.json').read_text())
            trajectories.append([[*sorted(t['state'].items()),t['horizon']] for t in traces])
            costs.append(result[0]['total_cost'])
    assert trajectories[0]==trajectories[1], 'Frozen physical trajectories differ'
    assert costs[0]==costs[1], 'Frozen costs differ'
    assert weights_hash(model)==before, 'Replay changed saved policy'
    original=json.loads((args.run/args.evaluation_dir/'summary.json').read_text())['episodes'][0]['total_cost']
    original_trace=json.loads((args.run/args.evaluation_dir/'trace_00.json').read_text())
    original_states=[[*sorted(t['state'].items()),t['horizon']] for t in original_trace]
    assert trajectories[0]==original_states, 'Reloaded trace differs from original saved evaluation'
    assert costs[0]==original, 'Reloaded cost differs from original saved evaluation'
    result={'status':'passed','run':str(args.run),'task':task,'steps':len(trajectories[0]),
            'same_trajectory_exact':True,'same_cost_exact':True,'policy_unchanged':True,
            'matches_original_evaluation_exact':True,
            'cost':costs[0],'scope':'First frozen case, two fresh environments, reloaded saved model.'}
    write(args.audit_out or ART/'results'/('replay_check_'+task+'.json'),result)
    print(json.dumps(result))


if __name__=='__main__':main()
