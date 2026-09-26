"""Numerical feature/gradient, reset, branch, constraints and training smoke checks."""
import copy
import json
import sys
import numpy as np
from runtime import ART,imports
from optimized_runtime import install_terminal,make_env
from run import write

task=sys.argv[1]
install_terminal(task)
_,SAC,Policy=imports()
import tensorflow as tf
import casadi as ca
env=make_env(task,901)
mpc=env.control_system.controller.mpc
model=SAC(Policy,env,batch_size=16,learning_starts=16,buffer_size=1000,n_cpu_tf_sess=1,seed=0,
    policy_kwargs={'layers':{'pi':[32,32],'qf':[32,32],'vf':[32,32],'mpc':[]},
        'mpc_state_dim':mpc.model.n_x,'mpc_parameter_dim':mpc.model.n_p-1,
        'mpc_gamma':.97,'mpc_vf_type':'poly','train_mpc_value_fn':True,'use_mpc_value_fn':True})
w,b=model.policy_tf.get_mpc_vfn_weights_and_biases()
nx=mpc.model.n_x;nd=nx+mpc.model.n_p-1
state=ca.SX.sym('x',nx);p=ca.SX.sym('p',nd-nx)
value=mpc.vf_fun(state,p,ca.DM(w[0]),ca.DM(b[0]))
fun=ca.Function('value_grad_hessian',[state,p],[value,ca.gradient(value,state),ca.hessian(value,state)[0]])
with model.graph.as_default():grad=tf.gradients(model.mpc_value_fn,model.mpc_state_ph)[0]
samples=np.random.RandomState(83).uniform(-2,2,(20,nd)).astype(np.float32)
tv,tg=model.sess.run([model.mpc_value_fn,grad],{model.mpc_state_ph:samples})
for i,z in enumerate(samples):
    cv,cg,ch=fun(z[:nx],z[nx:])
    assert np.allclose(float(cv),tv[i,0],atol=1e-6)
    assert np.allclose(np.array(cg).ravel(),tg[i,:nx],atol=1e-6)
    assert np.linalg.eigvalsh(np.array(ch)).min()>=-1e-9
cases=json.loads((ART/'configs'/(task+'_validation_bank.json')).read_text())['cases']
case=cases[0]
env.set_value_function_weights_and_biases(w,b)
obs=env.reset(**copy.deepcopy(case));s0=copy.deepcopy(env.control_system.current_state)
assert obs.shape==env.observation_space.shape
env.set_value_function_weights_and_biases([v*3 for v in w],b)
env.reset(**copy.deepcopy(case));assert env.control_system.current_state==s0
env.set_value_function_weights_and_biases(w,b)
rows=[]
for i in range(32):
    h=[5,10,25,40][i%4] if task=='vehicle' else [20,30,40][i%3]
    before=copy.deepcopy(env.control_system.current_state)
    obs,r,done,info=env.step(np.array([h]))
    assert np.isfinite(obs).all() and np.isfinite(r)
    assert np.isclose(-r,sum(info['reward/'+k] for k in ['performance','computation','constraint']))
    assert info['solver_calls']==(3 if task=='vehicle' else 1)
    assert info['data']['mpc_state'].shape==(nd,)
    rows.append({'h':h,'cost':-r,'solver_calls':info['solver_calls'],'residual':info['max_constraint_residual']})
    if done:env.reset(**copy.deepcopy(case))
out=ART/'results/optimized';out.mkdir(exist_ok=True)
write(out/('validation_'+task+'.json'),{'passed':True,'state_labels':mpc.model._x.labels(),
    'parameter_labels':mpc.model._p.labels(),'tf_casadi_value_gradient_match':True,'psd_hessian':True,
    'same_reset_state_despite_terminal_weights':True,'observation_dim':obs.size,'rows':rows})
print('VALIDATED',task,flush=True)
