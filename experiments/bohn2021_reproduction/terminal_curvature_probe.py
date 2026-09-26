"""Compare learned terminal local curvature with model-derived Riccati curvature."""
import json
import numpy as np
from runtime import ART,imports
from optimized_runtime import install_terminal
from riccati_terminal_probe import prior
from run import write

install_terminal('pendulum');_,SAC,_=imports();pw,pb,details=prior()
P=np.array(details['P']);S_inv=np.diag([.1,1/1.5,2/np.pi,.2]);rows=[]
for group,names in [('optimized',['pendulum_rl_s%d'%s for s in range(3)]),
                    ('forecast_refinement',['pendulum_forecast_s%d'%s for s in range(3)]),
                    ('recoverable_distribution',['pendulum_rl_s%d'%s for s in range(3)])]:
    for name in names:
        model=SAC.load(str(ART/'results'/group/name/'model.zip'))
        w,b=model.policy_tf.get_mpc_vfn_weights_and_biases();L=w[0][:16].reshape(4,4);Q=S_inv@L@L.T@S_inv
        row={'group':group,'model':name,'raw_quadratic_diagonal':Q.diagonal().tolist(),
            'riccati_diagonal_ratio':(Q.diagonal()/P.diagonal()).tolist(),
            'theta_curvature_ratio':float(Q[2,2]/P[2,2]),'bias':float(b[0][0])}
        rows.append(row);model.sess.close()
write(ART/'results/optimized/terminal_curvature_probe.json',{'scope':'Riccati matrix is a local stationary-reference comparator, not exact constrained stochastic value truth.',
    'riccati_diagonal':P.diagonal().tolist(),'rows':rows})
print(json.dumps(rows,indent=2),flush=True)
