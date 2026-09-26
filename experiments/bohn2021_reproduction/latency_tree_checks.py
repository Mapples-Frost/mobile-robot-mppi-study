"""Synthetic policy, CE and safety tests; never imports a simulator."""
import copy
import json
from pathlib import Path
import numpy as np
from latency_tree_policy import HS, BASE, FEATURES, features, choose, constant, thresholds, initialize, sample, update, rank, policy_key
from latency_tree_protocol import OUT, REG, read, write, sha, verify


def main():
    checks=[]
    def record(name, condition):
        assert condition, name
        checks.append(name)
    pctx=dict(state=dict(pos=.4,v=-.5,theta=-.6,omega=-1.),previews=dict(pos_r=[.1]*51),
              previous_input=dict(u1=[[-2.]]),elapsed=20,previous_initial_failure=True,previous_final_failure=False)
    pf=features('pendulum',pctx)
    record('pendulum_feature_formula',np.allclose(pf,[.6,1.,.5,.3,2.,0.,0.,.8,1.,0.],rtol=0,atol=1e-14))
    mirrored=copy.deepcopy(pctx);mirrored['state'].update(theta=.6,omega=1.)
    record('pendulum_sign_symmetry',features('pendulum',mirrored)==pf)
    preview=dict(trajectory_x=[k*.2 for k in range(51)],trajectory_y=[0.]*51)
    for j in range(3):
        preview.update({'obj_%d_x'%j:[float(20+j)]*51,'obj_%d_y'%j:[2.]*51,'obj_%d_r'%j:[.5]*51})
    vctx=dict(state=dict(x=0.,y=0.,theta=0.),previews=preview,previous_input=dict(u_s=[[3.]],u_omega=[[-.5]]),noise=[[0.,0.,0.]]*3,elapsed=0)
    vf=features('vehicle',vctx)
    record('vehicle_feature_dimension_and_heading',len(vf)==13 and vf[:5]==[0.,0.,0.,0.,0.] and vf[8:]==[3.,.5,1.,0.,0.])
    altered=copy.deepcopy(vctx);altered['state']['x']=10.;altered['state']['y']=-7.
    for name in altered['previews']:
        if name.endswith('_x'):altered['previews'][name]=[v+10. for v in preview[name]]
        if name.endswith('_y'):altered['previews'][name]=[v-7. for v in preview[name]]
    record('vehicle_translation_invariance',np.allclose(features('vehicle',altered),vf,rtol=0,atol=1e-13))
    for task,ctx in (('pendulum',pctx),('vehicle',vctx)):
        record(task+'_constant_all_horizons',all(choose(constant(task,h),{})[0]==h for h in HS))
        value=features(task,ctx)[0]
        tree=dict(kind='tree',task=task,nodes=[dict(feature=0,threshold=value)]*3,leaves=[5,10,15,20])
        record(task+'_tree_boundary_uses_left',choose(tree,ctx)[0]==5)
        tree['nodes']=[dict(feature=0,threshold=value-1.)]*3
        record(task+'_tree_right_route',choose(tree,ctx)[0]==20)
        rows=[[features(task,ctx),features(task,ctx)]]
        cuts=thresholds(task,rows); dist=initialize(task,cuts)
        record(task+'_degenerate_thresholds_finite',all(c and all(np.isfinite(c)) for c in cuts))
        r1=np.random.RandomState(11);r2=np.random.RandomState(11)
        samples=[sample(task,cuts,dist,r1) for _ in range(50)]
        record(task+'_seeded_search_exact',samples==[sample(task,cuts,dist,r2) for _ in range(50)])
        changed=update(dist,[g for p,g in samples[:3]])
        probabilities=changed['feature']+changed['action']+[p for group in changed['cut'] for p in group]
        record(task+'_updated_probabilities',all(abs(sum(p)-1)<1e-12 and min(p)>0 for p in probabilities))
        record(task+'_distribution_learns',changed!=dist)
        record(task+'_source_distribution_unchanged',dist==initialize(task,cuts))
        record(task+'_policy_json_exact',all(policy_key(p)==policy_key(json.loads(json.dumps(p))) for p,g in samples))
    b=dict(case=0,steps=100,success=True,constraint=False,initial_failed_steps=10,solver_failure_steps=5,
           total_cost=100.,physical_constraint_cost=100.,decision_total_s=10.)
    record('identical_reference_eligible',rank([b],[b])['eligible'])
    a=dict(b,steps=50,initial_failed_steps=6,solver_failure_steps=3)
    record('failure_rates_not_counts',not rank([a],[b])['eligible'])
    for field,value in (('success',False),('constraint',True),('initial_failed_steps',11),('solver_failure_steps',6),
                        ('total_cost',102.01),('physical_constraint_cost',102.01)):
        record('reject_'+field,not rank([dict(b,**{field:value})],[b])['eligible'])
    negative=dict(b,total_cost=-100.,physical_constraint_cost=-100.)
    record('negative_costs_use_absolute_denominator',rank([dict(negative,total_cost=-103.)],[negative])['cost_change']==-.03)
    record('boundary_two_percent_accepted',rank([dict(b,total_cost=102.,physical_constraint_cost=102.)],[b])['eligible'])
    record('latency_changes_training_objective',rank([dict(b,decision_total_s=5.)],[b])['objective']==-.25)
    failed=False
    try:rank([dict(b,total_cost=0.)],[dict(b,total_cost=0.)])
    except AssertionError:failed=True
    record('zero_cost_reference_fails_closed',failed)
    failed=False
    try:rank([dict(b,case=1)],[b])
    except AssertionError:failed=True
    record('unpaired_scenarios_rejected',failed)
    bad=copy.deepcopy(pctx);bad['state']['theta']=float('nan');failed=False
    try:features('pendulum',bad)
    except AssertionError:failed=True
    record('nonfinite_features_rejected',failed)
    result=dict(passed=True,checks=checks,count=len(checks),simulations=0,test_accessed=False,
                source_hashes={str(Path(__file__).resolve()):sha(Path(__file__)),str(Path(__file__).with_name('latency_tree_policy.py').resolve()):sha(Path(__file__).with_name('latency_tree_policy.py'))})
    if REG.exists():
        verify();path=OUT/'synthetic_checks.json'
        if path.exists():assert read(path)==result
        else:write(path,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
