"""Final saved-model replay plus terminal gradient checks after training."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
import numpy as np
from runtime import ART,imports
from optimized_runtime import install_terminal,make_env
from optimized_evaluate import evaluate
from run import write,weights_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['vehicle','pendulum'],required=True)
    ap.add_argument('--bank',default='holdout');args=ap.parse_args();task=args.task
    install_terminal(task);_,SAC,_=imports()
    source=ART/'results/optimized'/(task+'_rl_s0')
    model=SAC.load(str(source/'model.zip'));before=weights_hash(model)
    bank=ART/'results/optimized'/(task+'_holdout_bank.json') if args.bank=='holdout' else ART/'configs'/(task+'_validation_bank.json')
    case=json.loads(bank.read_text())['cases'][0]
    original=source/('holdout_value' if args.bank=='holdout' else 'eval_value')/'trace_00.json'
    trace=json.loads(original.read_text())
    with tempfile.TemporaryDirectory(prefix='optimized-replay-') as tmp:
        env=make_env(task,979)
        evaluate(model,env,[case],Path(tmp),None,True)
        replay=json.loads((Path(tmp)/'trace_00.json').read_text())
    keys=['state','input','horizon','cost','performance','compute','constraint','solver_calls','selected_objective','candidate_objectives']
    assert [{k:r[k] for k in keys} for r in trace]==[{k:r[k] for k in keys} for r in replay]
    assert before==weights_hash(model)
    import casadi as ca
    import tensorflow as tf
    mpc=env.control_system.controller.mpc;nx=mpc.model.n_x;nd=nx+mpc.model.n_p-1
    w,b=model.policy_tf.get_mpc_vfn_weights_and_biases()
    x=ca.SX.sym('x',nx);p=ca.SX.sym('p',nd-nx)
    value=mpc.vf_fun(x,p,ca.DM(w[0]),ca.DM(b[0]))
    f=ca.Function('trained_value',[x,p],[value,ca.gradient(value,x),ca.hessian(value,x)[0]])
    with model.graph.as_default():g=tf.gradients(model.mpc_value_fn,model.mpc_state_ph)[0]
    samples=np.array([r['terminal_input'] for r in trace],dtype=np.float32)
    tv,tg=model.sess.run([model.mpc_value_fn,g],{model.mpc_state_ph:samples})
    max_v,max_g,min_eig=0.,0.,float('inf')
    for i,z in enumerate(samples):
        cv,cg,ch=f(z[:nx],z[nx:]);cg=np.array(cg).ravel()
        assert np.allclose(float(cv),tv[i,0],rtol=1e-5,atol=1e-4)
        assert np.allclose(cg,tg[i,:nx],rtol=1e-5,atol=1e-4)
        eig=np.linalg.eigvalsh(np.array(ch)).min();assert eig>=-1e-8
        max_v=max(max_v,abs(float(cv)-tv[i,0]));max_g=max(max_g,np.max(abs(cg-tg[i,:nx])));min_eig=min(min_eig,eig)
    result={'passed':True,'task':task,'bank':args.bank,'first_case_steps':len(trace),
        'physical_trajectory_and_control_and_h_and_cost_exact':True,'all_candidate_objectives_exact':True,
        'frozen_model_hash':before,'trained_terminal_psd':True,'min_state_hessian_eigenvalue':float(min_eig),
        'max_tf_casadi_value_difference':float(max_v),'max_tf_casadi_gradient_difference':float(max_g)}
    write(ART/'results/optimized'/('replay_%s_%s.json'%(task,args.bank)),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
