"""Compact live process and durable-output snapshot; never restarts an experiment."""
import json
from pathlib import Path
from gated_horizon_search import OUT
from paper_h_soft_probe import read


def command(pid):
    try:return [x for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0') if x]
    except (OSError,UnicodeError):return []


def main():
    controllers={}
    expected={'status.json':'gated_horizon_search.py','posttrain_status.json':'gated_horizon_posttrain.py',
        'baseline_completion_status.json':'gated_horizon_complete_baselines.py','validation_finish_status.json':'gated_horizon_validation_finish.py',
        'validation_status.json':'gated_horizon_evaluate.py','test_status.json':'gated_horizon_evaluate.py'}
    for name,script in expected.items():
        p=OUT/name
        if not p.exists():continue
        state=read(p);args=command(state['pid']);alive=any(Path(a).name==script for a in args[1:])
        controllers[name]=dict(pid=state['pid'],live_matching_process=alive,active=state['active'],complete=state.get('complete'),stage=state.get('stage'),exception=state.get('exception'))
    training={}
    for folder in sorted((OUT/'train').glob('*')):
        current=[]
        for p in folder.glob('*/*attempt_*.json'):
            if (p.parent/'completed.json').exists():continue
            a=read(p);args=command(a['pid'])
            current.append(dict(candidate=p.parent.name,pid=a['pid'],live_matching_worker=any(Path(x).name=='gated_horizon_search.py' for x in args[1:]) and 'train' in args,
                attempted_steps=a['step_calls'],resets=a['reset_calls']))
        fit=folder/'fit_completed.json'
        training[folder.name]=dict(completed_candidates=len(list(folder.glob('*/completed.json'))),current=current,fit_complete=fit.exists(),selected=read(fit)['selected'] if fit.exists() else None)
    print(json.dumps(dict(controllers=controllers,training=training,validation_started=(OUT/'evaluations/validation').exists(),
        confirmation_frozen=(OUT/'confirmation_registration.json').exists(),test_outcomes_present=(OUT/'evaluations/test').exists()),indent=2))


if __name__=='__main__':main()
