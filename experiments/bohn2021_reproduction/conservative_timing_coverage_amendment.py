"""Pre-validation registration of descriptive latency coverage for failed rounds."""
import time
from pathlib import Path
from runtime import ROOT,ART
from paper_h_soft_probe import read,digest
from run import write

OUT=ART/'results/conservative_iteration_2026-09-24'


def main():
    dest=OUT/'timing_coverage_amendment';receipt=dest/'amendment.json'
    assert not receipt.exists() and not (OUT/'evaluations').exists()
    source=ROOT/'experiments/bohn2021_reproduction/conservative_iteration_timing.py'
    old=digest(dest/source.name);new=digest(source);changes=[]
    for name in ('timing_registration.json','confirmation_analysis_registration.json'):
        p=OUT/name;assert p.read_bytes()==(dest/name).read_bytes()
        data=read(p);before=digest(p)
        for key,h in data['hashes'].items():
            if Path(key).resolve()==source:
                assert h==old;data['hashes'][key]=new
            else:assert digest(Path(key))==h
        if name=='timing_registration.json':
            data['diagnostic_validation']='Optional separate all-candidate output: both trained rounds and both nominated fixed comparators, all seeds, including ineligible rounds. Descriptive only; never changes policy selection or test gate.'
        write(p,data);changes.append(dict(path=str(p),before=before,after=digest(p)))
    write(receipt,dict(time=time.time(),source_before=old,source_after=new,registrations=changes,
        reason='User requires actual compute evaluation even when learned policies fail safety/control eligibility.',
        unchanged='Existing selected validation/test job rules, latency boundaries, two repetitions, per-step raw recording, exact replay, and all effect/selection criteria.',
        added='--all-candidates is validation-only; both rounds and both nominated fixed comparators at all3 seeds. Separate timing_validation_all_candidates directory; descriptive, not a new policy selection path.',
        validation_outcomes_read=False,test_outcomes_read=False,source_hash=digest(Path(__file__))))


if __name__=='__main__':main()
