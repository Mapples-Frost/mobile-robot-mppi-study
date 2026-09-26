"""Adversarial checks of all-seed gates, signed costs and paired timing inference."""
import copy
from pathlib import Path
import numpy as np
from gated_horizon_report import assess,combine,interval,ratio,OUT
from paper_h_soft_probe import digest
from run import write


def main():
    b=dict(total_cost=100.,physical_constraint_cost=90.,success=32,constraints=0,initial_failure_rate=0.,final_failure_rate=0.,switches=0,
        episodes=[dict(case=i,total_cost=100.) for i in range(32)])
    def cost(value,physical=90.):
        a=copy.deepcopy(b);a.update(total_cost=value,physical_constraint_cost=physical,switches=1)
        for e in a['episodes']:e['total_cost']=value
        return a
    time=dict(mean_ratio=.8,lower=.75,upper=.85,repeat_ratios=[.79,.81])
    good=assess(cost(96.),b,time);assert combine([good]*3)['passed']
    checks=['constant4percent_cost_gain']
    for key,value in [('success',31),('constraints',1),('initial_failure_rate',.01),('final_failure_rate',.01),('switches',0),('physical_constraint_cost',92.)]:
        a=cost(96.);a[key]=value;bad=assess(a,b,time)
        assert not combine([good,bad,good])['passed'];checks.append('single_seed_'+key)
    absent=assess(cost(96.),b,None);assert not combine([absent]*3)['passed'];checks.append('cost_route_requires_timing')
    slowrepeat=dict(time,repeat_ratios=[.6,1.01]);r=assess(cost(100.),b,slowrepeat)
    assert not combine([r]*3)['passed'];checks.append('repeat_direction_disagreement')
    r=assess(cost(103.),b,time);assert not r['time_gain'];checks.append('speed_with_inferior_total_cost')
    neg=copy.deepcopy(b);neg.update(total_cost=-100.,physical_constraint_cost=-90.)
    for e in neg['episodes']:e['total_cost']=-100.
    a=cost(-104.,-89.);r=assess(a,neg,time);assert r['control_noninferior'] and r['cost_gain'];checks.append('negative_cost_tolerance')
    a=cost(-97.,-89.);r=assess(a,neg,time);assert not r['total_noninferior'] and not r['cost_gain'];checks.append('negative_cost_degradation')
    def times(scale):return [[dict(case=i,steps=i+1,decision_total_s=scale*(i+1)) for i in range(32)] for _ in range(2)]
    r=ratio(times(.8),times(1.));np.testing.assert_allclose([r[k] for k in ('mean_ratio','lower','upper')]+r['repeat_ratios'],.8,rtol=1e-14,atol=1e-14);checks.append('paired_weighted_latency_ratio')
    ci=interval(np.full((3,32),-4.));assert ci==dict(mean=-4.,lower=-4.,upper=-4.);checks.append('scene_block_cost_interval')
    write(OUT/'effect_checks.json',dict(passed=True,checks=checks,source_hash=digest(Path(__file__)),report_hash=digest(Path(__file__).with_name('gated_horizon_report.py'))))
    print('Passed %d gate/inference checks'%len(checks))


if __name__=='__main__':main()
