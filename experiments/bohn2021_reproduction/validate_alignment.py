"""Numerical assertions for time/label alignment, not performance assertions."""
import copy
import json
import numpy as np
from runtime import ART,make_env
from run import write

rows=[]
for task in ['vehicle','pendulum']:
    case=json.loads((ART/'configs'/(task+'_test_bank.json')).read_text())['cases'][0]
    if task=='pendulum':
        case['state']={'pos':0.,'v':0.,'theta':0.,'omega':0.}
        case['tvp']['pos_r']=[{'true':[0. if t<25 else .3],'forecast':[]} for t in range(160)]
    env=make_env(task,13,aligned=True)
    obs=env.reset(**copy.deepcopy(case))
    for t in range(32):
        before=env.control_system.get_state_vector(env.control_system.current_state).copy()
        obs,r,d,info=env.step(np.array([25.]))
        mpc=env.control_system.controller.mpc
        current_ref_index=env.control_system._step_count
        for j,var in enumerate(env.config['environment']['observation']['variables']):
            if var['type']=='tvp':
                ref=float(np.asarray(env.control_system.tvps[var['name']].get_values(current_ref_index)).ravel()[0])
                assert np.isclose(obs[j],ref)
        assert np.allclose(info['data']['mpc_state'][:len(before)],before)
        expected=float(mpc.lterm_fun(before,mpc.opt_x_num_unscaled['_u',0,0],
                       mpc.opt_x_num_unscaled['_z',1,0,-1],mpc.opt_p_num['_tvp',0],mpc.opt_p_num['_p',0]))
        assert np.isclose(info['data']['mpc_rewards'],expected)
        assert np.isclose(-r,sum(info['reward/'+k] for k in ['performance','computation','constraint']))
        rows.append({'task':task,'step':t,'simulator_step':current_ref_index,'bellman_cost':expected})
        assert not d,'Unexpected termination in the alignment test'
write(ART/'results/diagnosis/alignment_audit.json',{'passed':True,'steps':rows})
print('Alignment checks passed')
