"""Cross-check trained terminal values in TensorFlow, CasADi and NumPy."""
import json
import numpy as np
from runtime import ART,imports
from run import write

_,SAC,_=imports()
import casadi
from gym_let_mpc.utils import casadiNNVF
rows=[]
for task in ['pendulum','vehicle']:
    folder=ART/'results/full'/(task+'_rl_s0')
    spec=json.loads((folder/'manifest.json').read_text())
    model=SAC.load(str(folder/'model.zip'))
    w,b=model.policy_tf.get_mpc_vfn_weights_and_biases()
    nx,np_=spec['terminal_state_dim'],spec['terminal_parameter_dim']
    value=casadiNNVF(layers=[],type='poly')
    value.create_function(casadi.SX.sym('state',nx),casadi.SX.sym('parameters',np_))
    value.set_weights_and_biases(w,b)
    rng=np.random.RandomState(9127)
    samples=rng.uniform(-2,2,size=(20,nx+np_)).astype(np.float32)
    if task=='vehicle':samples[:,nx:]=rng.uniform(0,25,size=(20,np_))
    tf_values=model.sess.run(model.mpc_value_fn,feed_dict={model.mpc_state_ph:samples}).ravel()
    cas_values=np.array([float(value.eval_VF(s[:nx],s[nx:],value.weights_num,value.biases_num)) for s in samples])
    ref=np.concatenate([samples.astype(float),samples.astype(float)**2],axis=1)@w[0].astype(float)+b[0]
    assert np.allclose(tf_values,cas_values,rtol=3e-6,atol=1e-4)
    assert np.allclose(ref.ravel(),cas_values,rtol=1e-12,atol=1e-10)
    rows.append({'task':task,'samples':20,'tensorflow_casadi_max_abs_difference':float(np.max(np.abs(tf_values-cas_values))),
                 'casadi_numpy_max_abs_difference':float(np.max(np.abs(cas_values-ref.ravel()))),'passed':True})
    model.sess.close()
write(ART/'results/terminal_value_audit.json',{'status':'passed','comparisons':rows})
print(json.dumps(rows))
