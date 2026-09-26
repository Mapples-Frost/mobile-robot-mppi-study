"""Independent numerical checks of high-impact reproduction semantics."""
from runtime import ART, imports, make_env, install_correct_nstep
from run import write
import numpy as np
import copy


def main():
    imports()
    from stable_baselines.common.buffers import ReplayBuffer
    b=ReplayBuffer(10,extra_data_names=('bootstrap',))
    for i,(r,done) in enumerate([(1.,False),(2.,True),(100.,False)]):
        b.add(np.array([i]),np.array([0.]),r,np.array([i+1]),done,bootstrap=not done)
    old=b._encode_sample([0],n_step=32,gamma=.5)
    install_correct_nstep()
    new=b._encode_sample([0],n_step=32,gamma=.5)
    assert new[2][0]==2 and new[3][0,0]==2 and new[4][0]==1 and new[5]['n_step'][0]==2
    # A starting terminal must not reach into the following episode.
    end=b._encode_sample([1],n_step=32,gamma=.5)
    assert end[2][0]==2 and end[5]['n_step'][0]==1 and end[4][0]==1
    env=make_env('pendulum',9187)
    env.reset()
    _,reward,_,info=env.step(np.array([25.]))
    s=env.control_system.current_state
    u=float(env.control_system.controller.current_input['u1'][0])
    ref=env.control_system.tvps['pos_r'].get_values(env.steps_count)
    expected=.5*.8*s['v']**2+.2*.25*s['v']*s['omega']*np.cos(s['theta'])+(2/3)*.2*.25**2*s['omega']**2-.2*9.81*.25*np.cos(s['theta'])+10*(s['pos']-ref)**2+.1*u*u
    assert abs(info['reward/performance']-expected)<1e-9
    assert abs(reward+expected+.003*25+info['reward/constraint'])<1e-9
    v=make_env('vehicle',9187)
    v.reset()
    ctrl=v.control_system.controller
    state=copy.deepcopy(v.control_system.current_state)
    # Verify the upstream function is center distance, not signed clearance.
    state['x']=ctrl.obj_data[0]['x'][0]+.25*ctrl.obj_data[0]['r'][0]
    state['y']=ctrl.obj_data[0]['y'][0]
    dist=ctrl.get_obj_distance(state,0)
    assert 0 < dist < ctrl.obj_data[0]['r'][0]
    report={'status':'passed','checks':['32-step return includes terminal transition','n-step count and bootstrap mask','no cross-episode return leakage','pendulum reward equals independently evaluated physical cost','constraint and horizon cost decomposition','upstream collision distance semantics'],
        'upstream_nstep_example':{'reward':float(old[2][0]),'n_step':int(old[5]['n_step'][0]),'terminal':float(old[4][0])},
        'corrected_nstep_example':{'reward':2.,'n_step':2,'terminal':1.},
        'collision_example':{'center_distance':float(dist),'radius':float(ctrl.obj_data[0]['r'][0])}}
    write(ART/'results'/'validation.json',report)
    print(report)


if __name__=='__main__':main()
