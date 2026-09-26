"""Finalize the pre-validation shared-start amendment without replacing registrations."""
import copy
import time
from pathlib import Path
from gated_horizon_search import OUT,freeze
from gated_horizon_amendment import ROOT,verify,registration
from paper_h_soft_probe import read,digest
from run import write


def main():
    freeze();assert not (OUT/'evaluations').exists(),'No amendment after new validation starts'
    assert not (OUT/'confirmation_registration.json').exists()
    audit=read(ROOT/'probe/audit.json');assert audit['passed'] and audit['conditions']==20 and audit['resets']==40
    for p,h in audit['hashes'].items():
        assert digest(Path(p))==h
        for q,v in read(Path(p))['hashes'].items():assert digest(Path(q))==v
    source=Path(__file__).resolve().parent
    changed=('evaluate','timing','validation_finish','validation_audit','posttrain','complete_baselines','confirmation')
    sources={str(source/('gated_horizon_'+n+'.py')):dict(before=digest(ROOT/'original_sources'/('gated_horizon_'+n+'.py')),after=digest(source/('gated_horizon_'+n+'.py'))) for n in changed}
    for old in (ROOT/'original_sources').glob('*.py'):
        if str(source/old.name) not in sources:assert digest(old)==digest(source/old.name),('Unexpected source edit',old.name)
    paths=[ROOT/n for n in ('decision.json','probe/registration.json','probe/audit.json','orchestration_transition.json')]
    paths += [OUT/'reset_interface_review.json']
    for old in (ROOT/'original_registrations').glob('*.json'):
        assert digest(old)==digest(OUT/old.name),('Original registration changed',old.name)
        paths += [old,OUT/old.name]
    paths += [source/('gated_horizon_'+n+'.py') for n in ('amendment','shared_reset','shared_reset_probe','finalize_amendment')]
    value=dict(passed=True,before_new_validation=True,time=time.time(),sources=sources,hashes={str(p):digest(p) for p in paths},
        reason='Exposed historical validation revealed terminal-dependent unscored H50 reset actions and unequal scored initial states for independent fixed-H baselines. Equalize reset only using the task/seed primary fixed terminal, then restore each own independently trained terminal before all scoring.',
        unchanged='Training code, fitted models, scenario banks, search budgets, baseline opportunity, metrics, selection and all-seed acceptance criteria. Original registration files retained byte-for-byte; the exact source differences alone are authorized here.',
        smoke=dict(conditions=20,resets=40,steps=audit['steps'],maximum_dynamics_error=audit['maximum_dynamics_error']),
        limitation='Common initialization uses the primary task/seed terminal and is conditional on that initializer. Smoke covers both tasks, every seed0 independent H, two existing scenarios each; formal audits require exact common starts for every included task/seed/scenario. No new validation or test outcome used.')
    receipt=ROOT/'receipt.json';assert not receipt.exists(),'Inspect existing receipt rather than rewrite'
    write(receipt,value)
    pending=ROOT/'pending.json';assert pending.exists();pending.rename(ROOT/'completed_pending_request.json')
    try:
        verify()
        from gated_horizon_evaluate import register as evaluation
        from gated_horizon_timing import register as timing
        from gated_horizon_posttrain import registration as posttrain
        from gated_horizon_complete_baselines import register as baseline
        from gated_horizon_validation_finish import registration as finish
        from gated_horizon_report import register as effect
        from gated_horizon_confirmation import register as confirmation
        for fn in (evaluation,timing,posttrain,baseline,effect,finish,confirmation):fn()
        # Fail closed on an unregistered settings change and an arbitrary hash.
        original=read(OUT/'evaluation_registration.json')
        for mutation in ('settings','hash'):
            bad=copy.deepcopy(original)
            if mutation=='settings':bad['workers']=99
            else:bad['hashes'][next(iter(bad['hashes']))]='0'*64
            try:registration(OUT/'evaluation_registration.json',bad)
            except (AssertionError,KeyError):pass
            else:raise AssertionError('Unregistered mutation accepted: '+mutation)
        write(ROOT/'finalization_checks.json',dict(passed=True,registrations_checked=7,unregistered_mutations_rejected=2,receipt_hash=digest(receipt),validation_started=False,test_opened=False))
    except BaseException:
        (ROOT/'completed_pending_request.json').rename(pending)
        raise
    print('Shared initialization amendment verified; original registrations preserved; no validation/test outcomes accessed.')


if __name__=='__main__':main()
