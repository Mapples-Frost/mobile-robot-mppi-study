"""Freeze policy and terminal; contrast critic action rankings with causal rollouts."""
import copy
import json
import numpy as np
from runtime import ART, imports
from run import write, weights_hash
from optimized_runtime import install_terminal, make_env
from mechanism_probe import checked_step

OUT=ART/'results/optimized/pendulum_critic_probe'


def main():
    OUT.mkdir(exist_ok=True)
    protocol={'seed':0,'cases':[0,3,7],'times':[0,25,50,75],'horizons':[10,15,20,25,30,40,50],
        'scope':'Post-hoc validation diagnosis only; one H intervention then original deterministic policy to termination.',
        'caveat':'SAC Q includes entropy and training-distribution effects; deterministic finite episode returns are diagnostic rankings, not exact soft-Q calibration labels.'}
    write(OUT/'protocol.json',protocol)
    install_terminal('pendulum');_,SAC,_=imports()
    model=SAC.load(str(ART/'results/optimized/pendulum_rl_s0/model.zip'));before=weights_hash(model)
    env=make_env('pendulum',811);env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases())
    cases=json.loads((ART/'configs/pendulum_validation_bank.json').read_text())['cases']
    def action(obs):return int(np.clip(np.rint(model.predict(obs,deterministic=True)[0][0]),1,50))
    results=[]
    for case_id in protocol['cases']:
        source=ART/'results/optimized/pendulum_rl_s0/eval_value'/('trace_%02d.json'%case_id)
        trace=json.loads(source.read_text())
        for t in protocol['times']:
            path=OUT/('case%d_t%d.json'%(case_id,t))
            if path.exists():results.append(json.loads(path.read_text()));continue
            branches=[];query=None
            for h in protocol['horizons']:
                obs=env.reset(**copy.deepcopy(cases[case_id]))
                for i in range(t):
                    obs,done,row=checked_step(env,'pendulum',trace[i]['horizon'])
                    assert not done and row['state']==trace[i]['state']
                if query is None:
                    original_h=action(obs)
                    observations=np.repeat(obs[None,:],len(protocol['horizons']),axis=0)
                    actions=(2*(np.array(protocol['horizons'])-1)/49.-1).astype(np.float32)[:,None]
                    q1,q2=model.sess.run([model.step_ops[4],model.step_ops[5]],
                        {model.observations_ph:observations,model.actions_ph:actions})
                    query={'actor_h':original_h,'q1':q1.ravel().tolist(),'q2':q2.ravel().tolist()}
                cost=0.;steps=0;first=True
                while True:
                    selected=h if first else action(obs);first=False
                    obs,done,row=checked_step(env,'pendulum',selected)
                    cost+=.97**steps*row['cost'];steps+=1
                    if done:break
                branches.append({'h':h,'discounted_cost':cost,'steps':steps,'termination':row['termination']})
            row={'case':case_id,'t':t,**query,'branches':branches,
                 'best_measured_h':min(branches,key=lambda b:b['discounted_cost'])['h'],
                 'critic_q1_best_h':protocol['horizons'][int(np.argmax(query['q1']))]}
            write(path,row);results.append(row)
            print(json.dumps({'case':case_id,'t':t,'actor_h':row['actor_h'],
                'best_measured_h':row['best_measured_h'],'critic_q1_best_h':row['critic_q1_best_h']}),flush=True)
    assert weights_hash(model)==before
    write(OUT/'summary.json',{'frozen':True,'weights_sha256':before,'results':results,'protocol':protocol})


if __name__=='__main__':main()
