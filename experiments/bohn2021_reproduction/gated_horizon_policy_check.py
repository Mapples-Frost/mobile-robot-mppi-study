"""Boundary/causality unit checks for the finite gate class; no simulations."""
import copy
from pathlib import Path
from gated_horizon_policy import candidates,decide,BASE
from gated_horizon_search import OUT,freeze
from paper_h_soft_probe import digest
from run import write


def main():
    freeze();checks=[]
    for task in ('vehicle','pendulum'):
        if task=='vehicle':
            preview=dict(trajectory_x=[.3*i for i in range(51)],trajectory_y=[0.]*51)
            for j in range(3):
                for k,v in [('x',100.+10*j),('y',100.),('r',.5)]:preview['obj_%d_%s'%(j,k)]=[v]*51
            context=dict(elapsed=5,state=dict(x=0.,y=0.,theta=0.),previews=preview,previous_input=dict(u_s=[3.],u_omega=[0.]),noise=[[0,0,0]]*3)
        else:context=dict(elapsed=5,state=dict(pos=0.,v=0.,theta=0.,omega=0.),previews=dict(pos_r=[0.]*51),previous_input=dict(u1=[0.]))
        for policy in candidates(task):
            assert decide(policy,context)[0]==policy['short_h'];checks.append((task,policy['id'],'steady_short'))
            early=copy.deepcopy(context);early['elapsed']=4;assert decide(policy,early)[0]==BASE[task];checks.append((task,policy['id'],'initial_fixed'))
            unsafe=copy.deepcopy(context)
            if task=='vehicle':unsafe['previews']['obj_0_x']=[0.]*51;unsafe['previews']['obj_0_y']=[0.]*51
            else:unsafe['state']['theta']=.5
            assert decide(policy,unsafe)[0]==BASE[task];checks.append((task,policy['id'],'hazard_fixed'))
            if task=='pendulum':
                imminent=copy.deepcopy(context);imminent['previews']['pos_r'][policy['guard']]=.5
                assert decide(policy,imminent)[0]==BASE[task]
                outside=copy.deepcopy(context);outside['previews']['pos_r'][policy['guard']+1]=.5
                assert decide(policy,outside)[0]==policy['short_h']
                checks += [(task,policy['id'],'reference_inside_guard'),(task,policy['id'],'outside_chosen_guard')]
            else:
                yaw=copy.deepcopy(context);yaw['previous_input']['u_omega']=[2.]
                assert decide(policy,yaw)[0]==BASE[task]
                offtrack=copy.deepcopy(context);offtrack['state']['y']=2.
                assert decide(policy,offtrack)[0]==BASE[task]
                checks += [(task,policy['id'],'turning_fixed'),(task,policy['id'],'offtrack_fixed')]
    write(OUT/'policy_checks.json',dict(passed=True,checks=checks,count=len(checks),source_hash=digest(Path(__file__)),
        scope='Synthetic boundary controls only. No actual safety guarantee, efficacy or simulation evidence.'))
    print(dict(passed=True,checks=len(checks)))


if __name__=='__main__':main()
