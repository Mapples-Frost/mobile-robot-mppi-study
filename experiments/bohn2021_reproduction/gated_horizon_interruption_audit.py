"""Preserve externally interrupted attempts before resuming immutable candidate search."""
import json
import time
from pathlib import Path
import numpy as np
from gated_horizon_search import OUT,freeze
from gated_horizon_policy import decide,candidates
from gated_horizon_audit import audit_candidate
from conservative_iteration_audit import audit_context
from branch_calibration_audit import audit_trace
from conservative_fixed_log_audit import integrate
from paper_h_soft_probe import read,digest
from run import write


def main():
    freeze();active=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=p.read_bytes().decode().split('\0')
        except (OSError,UnicodeError):continue
        if args and 'python' in Path(args[0]).name and any(Path(a).name in ('gated_horizon_search.py','gated_horizon_posttrain.py','gated_horizon_complete_baselines.py') for a in args[1:]):active.append((p.parent.name,args))
    assert not active,active
    destination=OUT/('interruption_%d'%time.time_ns());destination.mkdir()
    complete=[];incomplete=[];steps=0;max_error=0.
    for root in sorted((OUT/'train').glob('*')):
        task,seed_text=root.name.rsplit('_s',1);seed=int(seed_text);cases=read(OUT/'banks'/('%s_train_s%d_bank.json'%(task,seed)))['cases']
        baseline=read(root/'fixed/completed.json')['result']['episodes']
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            if (folder/'completed.json').exists():
                a=audit_candidate(folder,task,cases,None if folder.name=='fixed' else baseline);a.update(task=task,seed=seed);complete.append(a);continue
            policy=next(p for p in candidates(task) if p['id']==folder.name)
            traces=sorted(folder.glob('r0_trace_*.json'));assert [p.name for p in traces]==['r0_trace_%02d.json'%i for i in range(len(traces))]
            calls=[json.loads(line) for line in (folder/'solver_calls.jsonl').read_text().splitlines()];counts=read(folder/'solver_attempts.json')
            assert counts['solve_attempts']>=counts['solve_completed']>=len(calls)
            attempts=[read(p) for p in folder.glob('attempt_*.json')];cursor=0;recorded=0
            for cid,path in enumerate(traces):
                trace=read(path);audit_trace(task,cases[cid],trace);audit_context(task,cases[cid],trace,0)
                assert calls[cursor]['kind']=='warmup' and calls[cursor]['case']==cid;cursor+=1
                for row in trace:
                    h,_=decide(policy,row['policy_context']);assert h==row['horizon']
                    a=row['recovery']['attempts'];assert calls[cursor:cursor+len(a)]==a;cursor+=len(a)
                    for key,value in integrate(task,row['previous_state'],row['input']).items():
                        np.testing.assert_allclose(row['state'][key],value,rtol=1e-7,atol=1e-7);max_error=max(max_error,abs(row['state'][key]-value))
                recorded+=len(trace)
            step_attempts=sum(a['step_calls'] for a in attempts);reset_attempts=sum(a['reset_calls'] for a in attempts)
            assert step_attempts>=recorded and reset_attempts>=len(traces)
            record=dict(task=task,seed=seed,policy=policy,original_path=str(folder),saved_full_episodes=len(traces),saved_steps=recorded,
                step_attempts=step_attempts,reset_attempts=reset_attempts,solver_counts=counts,logged_completed_solves=len(calls),
                attempted_steps_without_saved_full_trace=step_attempts-recorded,raw_solves_beyond_saved_prefix=len(calls)-cursor,
                unresolved_raw_solve_attempts=counts['solve_attempts']-len(calls))
            # Move only this verified candidate subtree inside the same study root. No deletion.
            target=destination/root.name/folder.name;target.parent.mkdir(exist_ok=True)
            folder.resolve().relative_to(OUT.resolve());target.resolve().relative_to(OUT.resolve());assert not target.exists()
            folder.rename(target);record['archive_path']=str(target)
            record['hashes']={str(p):digest(p) for p in target.rglob('*') if p.is_file()}
            incomplete.append(record);steps+=recorded
    for name in ('status.json','posttrain_status.json','baseline_completion_status.json'):
        p=OUT/name
        if p.exists():
            d=read(p);assert not (Path('/proc')/str(d['pid'])/'cmdline').exists(),(name,d['pid'])
            p.rename(destination/name)
    value=dict(passed=True,completed_candidates_retained=complete,incomplete_candidates_preserved=incomplete,
        interrupted_step_attempts=sum(a['step_attempts'] for a in incomplete),interrupted_reset_attempts=sum(a['reset_attempts'] for a in incomplete),
        interrupted_raw_solve_attempts=sum(a['solver_counts']['solve_attempts'] for a in incomplete),saved_steps_independently_audited=steps,maximum_dynamics_error=max_error,
        reason='Original process handles absent; external termination cause unknown. Status files were stale. No surviving registered workers. Completed candidates reused; incomplete candidates restart from beginning under unchanged frozen code. All interrupted work remains charged, including repeated saved episodes and unresolved solve attempts.',
        source_hash=digest(Path(__file__)))
    write(destination/'audit.json',value);print(json.dumps({k:v for k,v in value.items() if k not in ('completed_candidates_retained','incomplete_candidates_preserved')},indent=2));print(str(destination))


if __name__=='__main__':main()
