"""Verify a narrowly scoped, pre-validation initialization amendment.

Original registrations remain byte-for-byte immutable. Only source hashes named
in the auditable receipt may differ; all settings, jobs and criteria stay exact.
"""
from pathlib import Path
from gated_horizon_search import OUT
from paper_h_soft_probe import read,digest

ROOT=OUT/'shared_initialization_amendment'


def verify():
    assert not (ROOT/'pending.json').exists(),'Shared initialization amendment not audited and finalized'
    receipt=read(ROOT/'receipt.json');assert receipt['passed'] and receipt['before_new_validation']
    for p,h in receipt['hashes'].items():assert digest(Path(p))==h,('Amendment artifact changed',p)
    for p,change in receipt['sources'].items():
        assert digest(Path(p))==change['after']
        assert digest(ROOT/'original_sources'/Path(p).name)==change['before']
    return ROOT/'receipt.json'


def registration(path,value):
    verify();original=read(path);receipt=read(ROOT/'receipt.json')
    assert str(path) in receipt['hashes']
    assert {k:v for k,v in original.items() if k!='hashes'}=={k:v for k,v in value.items() if k!='hashes'}
    old_hashes={str(Path(p).resolve()):h for p,h in original['hashes'].items()}
    new_hashes={str(Path(p).resolve()):h for p,h in value['hashes'].items()}
    assert len(old_hashes)==len(original['hashes']) and len(new_hashes)==len(value['hashes'])
    assert old_hashes.keys()==new_hashes.keys()
    for p,before in old_hashes.items():
        after=new_hashes[p]
        if before!=after:assert receipt['sources'][str(Path(p).resolve())]==dict(before=before,after=after),(p,before,after)


def audit_starts(groups,split):
    """Require every scored initial plant state/input to equal its primary arm."""
    from gated_horizon_evaluate import arm_name
    from gated_horizon_policy import BASE
    count=0
    for task,family,seed,h in groups:
        folder=OUT/'evaluations'/split/task/arm_name(family,seed,h)
        primary=OUT/'evaluations'/split/task/arm_name('primary',seed,BASE[task])
        # Test may omit primary when another H wins; compare against any same-seed
        # included arm instead, because every arm uses the same reset adapter.
        if not primary.exists():
            reference=next(j for j in groups if j[0]==task and j[2]==seed)
            primary=OUT/'evaluations'/split/task/arm_name(reference[1],seed,reference[3])
        starts=read(folder/'initial_states.json');assert starts==read(primary/'initial_states.json'),(task,family,seed,h)
        trace=read(folder/'r0_trace_00.json');assert trace[0]['previous_state']==starts[0]['state']
        for cid,start in enumerate(starts):
            assert read(folder/('r0_trace_%02d.json'%cid))[0]['previous_state']==start['state']
        count+=len(starts)
    return dict(passed=True,episodes=count,scope='Exact common scored initial state and previous input within every task/seed; all arms retain own terminal during scoring.')
