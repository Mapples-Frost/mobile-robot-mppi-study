"""Ensure plotting cannot silently omit seeds or merge distinct comparator results."""
import copy
from pathlib import Path
from gated_horizon_plot import deduplicate,OUT,TASKS
from paper_h_soft_probe import digest
from run import write


def main():
    rows=[dict(task=t,seed=s,comparator=c,adaptive=dict(path='%s_adaptive_s%d'%(t,s)),fixed=dict(path='%s_primary_s%d'%(t,s)),statistic=1.)
          for t in TASKS for s in range(3) for c in ('independent','matched')]
    assert len(deduplicate(rows))==6;checks=['same_physical_comparator_labels_merge_without_extra_samples']
    different=copy.deepcopy(rows)
    for r in different:
        if r['comparator']=='matched':r['fixed']['path']+='different'
    assert len(deduplicate(different))==12;checks.append('different_models_all_retained')
    for label,data in [('missing_seed_label',rows[:-1]),('duplicate_label_replacing_a_seed',rows[:-1]+[rows[0]]),('inconsistent_statistics',copy.deepcopy(rows))]:
        if label=='inconsistent_statistics':data[1]['statistic']=2.
        try:deduplicate(data)
        except AssertionError:pass
        else:raise AssertionError(label+' should fail')
        checks.append(label+'_rejected')
    write(OUT/'plot_input_checks.json',dict(passed=True,checks=checks,synthetic_only=True,plot_source_hash=digest(Path(__file__).with_name('gated_horizon_plot.py')),check_source_hash=digest(Path(__file__))))
    print('Passed5 figure-input safeguards; no scientific figure or result generated')


if __name__=='__main__':main()
