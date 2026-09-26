"""One intervention: project negative state curvature to zero in frozen H5 V."""
import json
import numpy as np
from runtime import ART,imports,make_env
from run import evaluate,write,weights_hash

_,SAC,_=imports()
src=ART/'results/full/vehicle_fixed_h5'
out=ART/'results/diagnosis/terminal_convex_projection';out.mkdir(exist_ok=True)
model=SAC.load(str(src/'model.zip'))
before=weights_hash(model)
w,b=model.policy_tf.get_mpc_vfn_weights_and_biases()
n=w[0].shape[0]//2
original=w[0].copy()
w[0][n:n+3]=np.maximum(w[0][n:n+3],0)
assert np.all(w[0][n:n+3]>=0)
env=make_env('vehicle',17)
cases=json.loads((ART/'configs/vehicle_test_bank.json').read_text())['cases']
evaluate(model,env,cases,out,5,True,(w,b))
assert weights_hash(model)==before
write(out/'intervention.json',{'model_frozen':True,'original_kernel':original.tolist(),'projected_kernel':w[0].tolist(),
    'note':'Single post-training intervention, no retraining or test selection. Not the author original model.'})
print((out/'summary.json').read_text())
