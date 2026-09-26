"""Validate a model-derived local terminal prior before any new training."""
import json
import numpy as np
from scipy.linalg import expm,solve_discrete_are
from runtime import ART,imports
from run import write
from optimized_runtime import install_terminal,make_env
from optimized_evaluate import evaluate


def prior():
    # Raw state order: omega, position error, theta, velocity.
    m,M,l,g=.2,.8,.25,9.81
    A=np.zeros((4,4));A[1,3]=1;A[2,0]=1
    A[0,2]=M*g/((4/3)*M*l-m*l);A[3,2]=m*g/(m-(4/3)*M)
    B=np.array([[-1/((4/3)*M*l-m*l)],[0],[0],[-(4/3)/(m-(4/3)*M)]])
    augmented=np.zeros((5,5));augmented[:4,:4]=A;augmented[:4,4:]=B
    discrete=expm(.04*augmented);Ad,Bd=discrete[:4,:4],discrete[:4,4:]
    Q=np.diag([(2/3)*m*l*l,10,m*g*l/2,M/2]);Q[0,3]=Q[3,0]=m*l/2
    R=np.array([[.1]]);rho=.97
    P=solve_discrete_are(np.sqrt(rho)*Ad,np.sqrt(rho)*Bd,Q,R)
    K=np.linalg.solve(R+rho*Bd.T@P@Bd,rho*Bd.T@P@Ad)
    residual=P-(Q+rho*Ad.T@P@Ad-rho*Ad.T@P@Bd@K)
    assert np.max(abs(residual))<1e-7
    S=np.diag([10.,1.5,np.pi/2,5.]);L=np.linalg.cholesky(S@P@S)
    w=np.r_[L.ravel(),np.zeros(4)].reshape(-1,1);b=np.array([-m*g*l/(1-rho)])
    return w,b,{'P':P.tolist(),'K':K.tolist(),'dare_residual':float(np.max(abs(residual))),
        'local_closed_loop_eigenvalue_magnitudes':abs(np.linalg.eigvals(Ad-Bd@K)).tolist(),
        'scope':'Local stationary-reference unconstrained linearization; not exact value for changing references or constraints.'}


def main():
    dest=ART/'results/riccati_terminal_probe';dest.mkdir(exist_ok=True)
    w,b,details=prior();write(dest/'prior.json',details)
    install_terminal('pendulum');_,SAC,_=imports()
    model=SAC.load(str(ART/'results/optimized/pendulum_rl_s0/model.zip'))
    cases=json.loads((ART/'configs/pendulum_validation_bank.json').read_text())['cases']
    rows=[]
    for h in [5,10,15,20,25,30]:
        folder=dest/('h%d'%h);folder.mkdir(exist_ok=True)
        if not (folder/'summary.json').exists():
            env=make_env('pendulum',831)
            evaluate(model,env,cases,folder,h,True,terminal_weights=([w],[b]))
        s=json.loads((folder/'summary.json').read_text());row={'h':h,'cost':s['mean_total_cost'],'constraints':s['constraint_episodes']}
        rows.append(row);print(json.dumps(row),flush=True)
    write(dest/'summary.json',{'source':'Analytical local prior, not learned terminal','validation_only':True,'rows':rows})


if __name__=='__main__':main()
