"""Audit the actual terminal inputs, gradients and reset confounding."""
import copy
import json
import numpy as np
from runtime import ART, imports, make_env
from run import write, weights_hash
from mechanism_probe import OUT, configure_value


def main():
    _, SAC, _ = imports()
    import casadi as ca
    env=make_env('vehicle',26091703,aligned=True,scaled_obs=True)
    cases=json.loads((ART/'configs/vehicle_holdout_bank.json').read_text())['cases']
    mpc=env.control_system.controller.mpc
    labels=mpc.model._x.labels()+[l for l in mpc.model._p.labels() if 'n_horizon' not in l]
    assert labels==['[theta,0]','[x,0]','[y,0]','[goal_x,0]','[goal_y,0]'],labels
    rows=[]
    for seed in range(3):
        model=SAC.load(str(ART/'results/paper_defaults'/('vehicle_rl_s%d'%seed)/'model.zip'))
        before=weights_hash(model)
        w,b=model.policy_tf.get_mpc_vfn_weights_and_biases()
        linear=w[0][:5,0].astype(float);quadratic=w[0][5:,0].astype(float)
        z=np.array([[0.,5.,1.,20.,0.],[0.,5.,1.,-20.,10.],[0.,15.,-2.,30.,-3.]],dtype=np.float32)
        tf=model.sess.run(model.mpc_value_fn,feed_dict={model.mpc_state_ph:z}).ravel()
        formula=z@linear+(z.astype(float)**2)@quadratic+b[0][0]
        assert np.allclose(tf,formula,rtol=1e-5,atol=1e-4)
        state=ca.SX.sym('state',3);goal=ca.SX.sym('goal',2)
        val=mpc.vf_fun(state,goal,ca.DM(w[0]),ca.DM(b[0]))
        gradient=ca.Function('grad',[state,goal],[ca.gradient(val,state)])
        mixed=ca.jacobian(ca.gradient(val,state),goal)
        assert mixed.is_zero()
        grads=[np.array(gradient(v[:3],v[3:])).ravel() for v in z]
        assert np.array_equal(grads[0],grads[1])
        init=[]
        for i,case in enumerate(cases):
            configure_value(env)
            env.reset(**copy.deepcopy(case))
            zero=copy.deepcopy(env.control_system.current_state)
            configure_value(env,model,'learned')
            env.reset(**copy.deepcopy(case))
            learned=copy.deepcopy(env.control_system.current_state)
            init.append({'case':i,'zero':zero,'learned':learned,
                'position_delta':float(np.hypot(zero['x']-learned['x'],zero['y']-learned['y'])),
                'heading_delta':abs(zero['theta']-learned['theta'])})
        rows.append({'seed':seed,'linear':linear.tolist(),'quadratic':quadratic.tolist(),'bias':b[0].tolist(),
            'state_hessian_eigenvalues':(2*quadratic[:3]).tolist(),
            'input_examples':z.tolist(),'state_gradients':np.array(grads).tolist(),
            'mixed_state_goal_derivative_identically_zero':True,
            'translation_gradient_delta':(grads[2]-grads[0]).tolist(),
            'tf_numpy_max_abs_difference':float(np.max(np.abs(tf-formula))),
            'old_holdout_reset_differences':init,
            'max_reset_position_delta':max(r['position_delta'] for r in init),
            'max_reset_heading_delta':max(r['heading_delta'] for r in init)})
        assert before==weights_hash(model)
        model.sess.close()
    result={'labels':labels,'actor_observation_scaling_reaches_terminal_input':False,
        'terminal_missing_inputs':['obstacle geometry','current trajectory reference','episode time'],
        'relative_coordinate_code':'Found only commented-out goal_distance lines in recovered policies.py:351,356.',
        'claims_scope':'Recovered implementation and reconstructed configs; original external preprocessing unavailable.',
        'rows':rows}
    write(OUT/'terminal_structure_audit.json',result)
    print(json.dumps([{'seed':r['seed'],'state_hessian':r['state_hessian_eigenvalues'],
        'max_reset_position_delta':r['max_reset_position_delta'],'max_reset_heading_delta':r['max_reset_heading_delta']} for r in rows]),flush=True)


if __name__=='__main__':main()
