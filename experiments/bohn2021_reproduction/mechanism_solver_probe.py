"""Post-hoc local-solution diagnosis of a difficult scene, not a benchmark arm."""
import copy
import hashlib
import json
import time
import numpy as np
from mechanism_probe import OUT, fresh, checked_step
from run import write


def initialize(mpc, state, h, mode):
    if mode=='warm':return
    import casadi as ca
    mpc.opt_x_num.master=ca.DM.zeros(*mpc.opt_x_num.shape)
    theta,x,y=state['theta'],state['x'],state['y']
    npoints=len(mpc.opt_x_num['_x',0,0])
    for k in range(51):
        for j in range(npoints):mpc.opt_x_num['_x',k,0,j]=np.array([theta,x,y])
        if k<50:
            speed=0. if mode=='cold' or k>=h else 3.
            turn=0. if mode=='cold' or k>=min(h,8) else (2. if mode=='left' else -2.)
            mpc.opt_x_num['_u',k,0]=np.array([turn,speed])
            x+=.1*speed*np.cos(theta+.05*turn)
            y+=.1*speed*np.sin(theta+.05*turn)
            theta+=.1*turn


def main():
    folder=OUT/'solver_case06';folder.mkdir(exist_ok=True)
    write(folder/'protocol.json',{'post_hoc':True,'case':6,'task':'vehicle','base_h':10,
        'anchors':[20,30,40],'initial_guesses':['warm','cold','left','right'],
        'reason':'First diagnostic bank showed H10 cost 4574 vs H5 cost 4.76, with solver success at every step.',
        'intervention':'Change only NLP primal initial guess at one decision; identical parameters/objective/constraints. Then return to H10.',
        'scope':'Can demonstrate multiple local solutions at the same state. Not evidence of a general fix or paper replication.'})
    case=json.loads((OUT/'vehicle_bank.json').read_text())['cases'][6]
    base=json.loads((OUT/'vehicle_case06/h10.json').read_text())
    rows=[]
    for anchor in [20,30,40]:
        parameter_hashes=[]
        for mode in ['warm','cold','left','right']:
            start=time.monotonic()
            env,obs=fresh('vehicle',case)
            mpc=env.control_system.controller.mpc
            assert mpc.model._u.labels()==['[u_omega,0]','[u_s,0]']
            for t in range(anchor):
                obs,done,r=checked_step(env,'vehicle',10)
                assert not done and r['state']==base['trace'][t]['state']
            initialize(mpc,env.control_system.current_state,10,mode)
            obs,done,r=checked_step(env,'vehicle',10)
            params=np.array(mpc.opt_p_num.cat,dtype=np.float64)
            parameter_hashes.append(hashlib.sha256(params.tobytes()).hexdigest())
            g=np.array(mpc.opt_g_num).ravel()
            low=np.array(mpc.cons_lb).ravel();high=np.array(mpc.cons_ub).ravel()
            infeas=float(max(0,np.max(low-g),np.max(g-high)))
            local={'objective':float(mpc.opt_f_num),'constraint_max_violation':infeas,
                'solver_success':r['solver_success'],'input':r['input'],
                'iterations':mpc.solver_stats.get('iter_count')}
            trace=[r]
            while not done:
                obs,done,r=checked_step(env,'vehicle',10);trace.append(r)
            costs=np.array([r['cost'] for r in trace])
            row={'anchor':anchor,'mode':mode,'local':local,'suffix_cost':float(costs.sum()),
                'discounted_suffix_cost':float(costs @ (.97**np.arange(len(costs)))),
                'termination':trace[-1]['termination'],'elapsed_s':time.monotonic()-start}
            if mode=='warm':assert trace==base['trace'][anchor:]
            write(folder/('%d_%s.json'%(anchor,mode)),dict(row,trace=trace))
            rows.append(row);print(json.dumps(row),flush=True)
        assert len(set(parameter_hashes))==1,'NLP parameters changed across initial guesses'
    write(folder/'completed.json',{'rows':rows,'same_nlp_parameters_verified':True,
        'same_prefix_verified':True,'costs_independently_verified':True})


if __name__=='__main__':main()
