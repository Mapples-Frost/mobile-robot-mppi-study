"""Independent branch semantics, feature information, and budget audit."""
import argparse
from pathlib import Path
import numpy as np
from conservative_iteration import OUT, TASKS, HS, ANCHORS, BLOCK, verify, prior_model, action
from relative_policy_features import encode
from branch_calibration_audit import audit_trace
from fixed_policy_branches import metrics
from paper_h_soft_probe import read, digest
from run import write
from conservative_canonical_reset import verify_collection_provenance


def audit_context(task,case,trace,offset):
    for j,row in enumerate(trace):
        t=offset+j;ctx=row['policy_context'];clock=t+1
        assert ctx['clock']==clock and ctx['elapsed']==t
        assert ctx['state']==row['previous_state']
        if j:assert ctx['previous_input']==trace[j-1]['input'] and ctx['previous_h']==trace[j-1]['horizon']
        for name,values in ctx['previews'].items():
            assert len(values)==51
            # Registered task TVPs are forecast-aware; use same documented ramp.
            # TTAHMPC creates obstacle forecast components with starts_at=5
            # (absolute TVP clock, not relative preview offset).
            starts_at=5 if name.startswith('obj_') else 0
            expected=[sum(case['tvp'][name][clock+k]['true'])+
                      (sum(case['tvp'][name][clock+k]['forecast']) if clock+k>=starts_at else 0)*min(51,2*k)/51 for k in range(51)]
            np.testing.assert_allclose(values,expected,rtol=0,atol=1e-12)
        assert ctx['noise']==(case['reference']['ns'] if task=='vehicle' else [])
        remaining=((150 if task=='vehicle' else 100)-t)/(150 if task=='vehicle' else 100)
        expected=encode(task,ctx['state'],ctx['previews'],remaining,ctx['previous_h'],ctx['previous_input'],ctx['noise'])
        np.testing.assert_array_equal(expected,ctx['features'])


def audit_job(task,seed,round_id,smoke=False):
    verify()
    dest=OUT/('smoke_'+task if smoke else '%s_s%d_r%d'%(task,seed,round_id))
    done=read(dest/'collection_completed.json')
    assert done['audit_passed'] and (done['task'],done['seed'],done['round'],done['smoke'])==(task,seed,round_id,smoke)
    verify_collection_provenance(dest,done,digest(OUT/'collection_inputs_sha256.json'))
    for p,h in done['hashes'].items():assert digest(Path(p))==h
    cases=read(dest/'train_bank.json')['cases']
    assert len(cases)==(1 if smoke else 8)
    prior=prior_model(task,seed,round_id)
    steps=0;resets=len(cases);groups=[];skips=[];branches=0
    for cid,case in enumerate(cases):
        source=read(dest/('source_%02d.json'%cid))
        audit_trace(task,case,source);audit_context(task,case,source,0)
        steps+=len(source);resets+=1
        for t,row in enumerate(source):
            if t%BLOCK==0:h=action(task,row['policy_context'],prior)
            assert row['horizon']==h
        candidates=sorted(set((10,25 if task=='vehicle' else 30) if smoke else HS))
        for anchor in ((0,10) if smoke else ANCHORS[task]):
            if anchor>=len(source):
                skips.append(dict(case=cid,anchor=anchor,termination=source[-1]['termination']));continue
            m={}
            for h in candidates:
                path=dest/('case%02d_t%03d_h%02d.json'%(cid,anchor,h))
                b=read(path);tr=b['trace']
                assert (b['case'],b['anchor'],b['h'])==(cid,anchor,h)
                audit_trace(task,case,tr,offset=anchor);audit_context(task,case,tr,anchor)
                assert tr[0]['policy_context']==source[anchor]['policy_context']
                for j,row in enumerate(tr):
                    t=anchor+j
                    if j<BLOCK:selected=h
                    elif t%BLOCK==0:selected=action(task,row['policy_context'],prior)
                    assert row['horizon']==selected
                if h==source[anchor]['horizon']:assert tr==source[anchor:]
                assert b['metrics']==metrics(task,tr)
                m[str(h)]=b['metrics'];branches+=1
                repeats=2 if smoke else 1
                steps+=(anchor+len(tr))*repeats;resets+=repeats
            groups.append(dict(case=cid,anchor=anchor,context=source[anchor]['policy_context'],source_h=source[anchor]['horizon'],branches=m))
    assert groups==done['groups'] and skips==done['skips']
    attempts=[read(p) for p in dest.glob('attempt_*.json')]
    actual_steps=sum(a['step_calls'] for a in attempts);actual_resets=sum(a['reset_calls'] for a in attempts)
    assert actual_steps==done['explicit_step_calls']>=steps
    assert actual_resets==done['explicit_reset_calls']>=resets
    result=dict(passed=True,task=task,seed=seed,round=round_id,smoke=smoke,
        groups=len(groups),branches=branches,skipped_anchors=len(skips),
        reconstructed_step_calls=steps,actual_step_calls=actual_steps,extra_steps=actual_steps-steps,
        reconstructed_reset_calls=resets,actual_reset_calls=actual_resets,extra_resets=actual_resets-resets,
        source_choice_exact=True,causal_feature_audit=True,
        hashes={str(dest/'collection_completed.json'):digest(dest/'collection_completed.json'),str(Path(__file__)):digest(Path(__file__))})
    write(dest/'collection_audit.json',result)
    print(result)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=TASKS,required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--round',type=int,default=0);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();audit_job(a.task,a.seed,a.round,a.smoke)
