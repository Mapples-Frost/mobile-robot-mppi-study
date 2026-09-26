"""Read-only validation progress; no efficacy metrics and no writes/restarts.

Do not run during formal serial timing: even a monitor Python process is a host
conflict under the registered idle check. Use native file reads at that stage.
"""
import json
from pathlib import Path
from gated_horizon_search import OUT


def command(pid):
    try:return (Path('/proc')/str(pid)/'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
    except FileNotFoundError:return []


def children(pid):
    # subprocesses launched by ThreadPoolExecutor belong to the spawning Linux
    # thread, so the main thread's children file alone can incorrectly be empty.
    result=set()
    for path in (Path('/proc')/str(pid)/'task').glob('*/children'):
        try:result.update(int(p) for p in path.read_text().split())
        except FileNotFoundError:pass
    return sorted(result)


def main():
    errors=[]
    def read(path):
        try:return json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError) as exc:
            errors.append(dict(path=str(path),error=repr(exc)));return None
    controllers={};suite=None;workers=[]
    for name,script in [('validation_status.json','gated_horizon_evaluate.py'),('posttrain_status.json','gated_horizon_posttrain.py'),
            ('baseline_completion_status.json','gated_horizon_complete_baselines.py'),('validation_finish_status.json','gated_horizon_validation_finish.py')]:
        state=read(OUT/name)
        if state is None:continue
        pid=state['pid'];cmd=command(pid);live=any(Path(p).name==script for p in cmd)
        controllers[name]=dict(pid=pid,live_matching_process=live,active=state['active'],complete=state.get('complete'),stage=state.get('stage'),exception=state.get('exception'))
        if name=='validation_status.json':
            suite=state
            if live:
                for child in children(pid):
                    args=command(child)
                    if not any(Path(p).name==script for p in args):continue
                    def arg(flag):return args[args.index(flag)+1] if flag in args and args.index(flag)+1<len(args) else None
                    workers.append(dict(pid=child,task=arg('--task'),family=arg('--family'),seed=arg('--seed'),h=arg('--h'),mode=arg('--mode'),live_matching_process=True))
    registration=read(OUT/'evaluation_registration.json');expected=len(registration['jobs']) if registration else None
    completed=suite.get('completed',[]) if suite else []
    successful=[r for r in completed if r['exit_code']==0];failures=[r for r in completed if r['exit_code']!=0]
    conditions=[];saved_episodes=0;completed_folders=0
    for folder in sorted((OUT/'evaluations/validation').glob('*/*')):
        if not folder.is_dir():continue
        episodes=len(list(folder.glob('r0_trace_*.json')));saved_episodes+=episodes
        if (folder/'completed.json').exists():completed_folders+=1;continue
        meters=[]
        for p in folder.glob('attempt_*.json'):
            meter=read(p)
            if meter is not None:meters.append(dict(pid=meter['pid'],step_calls=meter['step_calls'],reset_calls=meter['reset_calls'],
                live_matching_process=any(Path(v).name=='gated_horizon_evaluate.py' for v in command(meter['pid']))))
        conditions.append(dict(condition=str(folder.relative_to(OUT/'evaluations/validation')),saved_episodes=episodes,meters=meters))
    print(json.dumps(dict(controllers=controllers,registered_initial_conditions=expected,reported_successful_conditions=len(successful),
        reported_failed_conditions=failures,completed_result_folders=completed_folders,saved_validation_episodes=saved_episodes,
        live_validation_children=workers,incomplete_conditions=conditions,observation_errors=errors,
        validation_audit_exists=(OUT/'audit_validation.json').exists(),confirmation_frozen=(OUT/'confirmation_registration.json').exists(),
        test_outcomes_present=(OUT/'evaluations/test').exists(),
        scope='Non-atomic live operational snapshot only; no score inspection, integrity/effect audit, mutation or restart. Process disappearance during a completed transition is not itself a failure.'),indent=2))


if __name__=='__main__':main()
