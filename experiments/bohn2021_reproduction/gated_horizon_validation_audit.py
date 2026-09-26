"""Independent trace/dynamics audit of every validation arm, no test access."""
import json
from pathlib import Path
from gated_horizon_search import OUT
from gated_horizon_evaluate import jobs,arm_name,freeze_policies
from gated_horizon_audit import audit_candidate
from paper_h_soft_probe import read,digest
from run import write
from gated_horizon_amendment import audit_starts


def main():
    freeze_policies();assert read(OUT/'validation_status.json')['complete'];groups=[];hashes={}
    for task,family,seed,h in jobs():
        folder=OUT/'evaluations/validation'/task/arm_name(family,seed,h)
        bank=OUT/'banks'/(task+'_validation_bank.json');assert digest(bank)==read(OUT/'banks/hashes.json')[str(bank)]
        a=audit_candidate(folder,task,read(bank)['cases'],fixed_h=None if family=='adaptive' else h)
        a.update(task=task,family=family,seed=seed,h=h);groups.append(a)
        summary=read(folder/'summary.json');done=read(folder/'completed.json')
        assert summary['episodes']==done['result']['episodes'] and summary['solver_counts']==a['counts']
        hashes[str(folder/'completed.json')]=digest(folder/'completed.json')
    result=dict(passed=True,conditions=len(groups),episodes=sum(g['episodes'] for g in groups),steps=sum(g['steps'] for g in groups),
        maximum_dynamics_error=max(g['maximum_dynamics_error'] for g in groups),groups=groups,hashes=hashes,source_hash=digest(Path(__file__)),test_access=False)
    result['shared_initialization']=audit_starts(jobs(),'validation')
    write(OUT/'audit_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','hashes')},indent=2))


if __name__=='__main__':main()
