"""Control-flow smoke only: no scenario outcomes, clocks or simulated metrics."""
from unittest.mock import patch
from pathlib import Path
import conservative_iteration_timing as timing
from conservative_iteration import OUT,TASKS,BASE
from paper_h_soft_probe import digest
from run import write


def main():
    synthetic=dict(selection_eligible=False,selections={t:dict(round=None,independent_h=BASE[t],matched_h=BASE[t]) for t in TASKS})
    with patch.object(timing,'read',return_value=synthetic):
        try:timing.jobs('validation')
        except AssertionError:rejected=True
        else:rejected=False
        assert rejected,'The ordinary effect-gated path must still reject ineligible policies'
        shared=timing.all_candidate_jobs();assert len(shared)==18
        for t in TASKS:
            for s in range(3):
                assert (t,'round0',s,0) in shared and (t,'round1',s,0) in shared and (t,'primary',s,BASE[t]) in shared
        synthetic['selections']['vehicle'].update(independent_h=15,matched_h=10)
        synthetic['selections']['pendulum'].update(independent_h=35,matched_h=20)
        separate=timing.all_candidate_jobs();assert len(separate)==24
        assert len(set(separate))==24
    with patch.object(timing,'verify'),patch.object(timing,'register'),patch.object(timing,'idle'):
        try:timing.run('test',all_candidates=True)
        except AssertionError:rejected_test=True
        else:rejected_test=False
    assert rejected_test,'Diagnostic timing must never unlock test'
    write(OUT/'timing_coverage_amendment/control_flow_smoke.json',dict(passed=True,
        ordinary_ineligible_path_rejected=True,diagnostic_test_rejected=True,
        deduplicated_shared_fixed_conditions=len(shared),distinct_fixed_conditions=len(separate),
        scope='Synthetic selection metadata only; no control/timing results generated or validation/test outcomes read.',
        source_hash=digest(Path(__file__))))
    print('Timing coverage smoke passed: default gate intact, all candidates covered, test rejected')


if __name__=='__main__':main()
