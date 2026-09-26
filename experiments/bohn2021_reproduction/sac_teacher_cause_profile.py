"""Read-only checkpoint and replay support diagnostics on fixed recorded states."""
import json
import numpy as np
from sac_teacher_cause import OUT,STUDY,GRID,Policy,models,read,write,verify


def rows(path):
    with path.open() as stream:return [json.loads(line) for line in stream]


def coverage(part):
    obs=np.asarray([r['obs'] for r in part]);hs=np.asarray([r['horizon'] for r in part])
    error=obs[:,0]*1.5-obs[:,4]
    changes=np.max(abs(obs[:,5:55]-obs[:,5:6]),axis=1)>1e-6
    return {'rows':len(part),'mean_abs_position_error':float(np.mean(abs(error))),
        'position_error_p95':float(np.percentile(abs(error),95)),
        'error_above_02_fraction':float(np.mean(abs(error)>.2)),
        'visible_change_fraction':float(np.mean(changes)),
        'H_histogram':{str(h):int(sum(hs==h)) for h in range(1,51)},
        'H5_or_30_fraction':float(np.mean(np.isin(hs,[5,30]))),
        'H_below_5_fraction':float(np.mean(hs<5)),
        'solver_failure_fraction':float(np.mean([not r['solver_success'] for r in part]))}


def main():
    verify()
    from optimized_runtime import install_terminal
    install_terminal('pendulum')
    replay={};sample={};profiles=[];sources={}
    for seed in range(3):
        for arm in ['plain','teacher']:
            data=rows(STUDY/'models'/('%s_s%d'%(arm,seed))/'transitions.jsonl')
            for phase in ['teacher','online']:
                part=[r for r in data if r['phase']==phase]
                if not part:continue
                replay['%s_s%d_%s'%(arm,seed,phase)]=coverage(part)
                if arm=='teacher' and phase=='teacher':
                    # Uniform time subsample fixed before inspecting checkpoint values.
                    idx=np.arange(0,len(part),10)
                    sample[seed]=[part[i] for i in idx]
                    sources[str(seed)]={'source':'models/teacher_s%d/transitions.jsonl'%seed,'phase':'teacher','indices':idx.tolist()}
    for name in models():
        seed=int(name.split('_')[1][1:]);data=sample[seed]
        obs=np.asarray([r['obs'] for r in data],np.float32)
        next_obs=np.asarray([r['next_obs'] for r in data],np.float32)
        actions=np.asarray([r['action'] for r in data],np.float32).reshape(-1,1)
        rewards=-np.asarray([r['cost'] for r in data])/.6
        dones=np.asarray([r['done'] for r in data])
        policy=Policy(name);model=policy.model;p=model.policy_tf
        det,mu,std=model.sess.run([p.deterministic_policy,p.act_mu,p.std],{p.obs_ph:obs})
        feed={model.observations_ph:obs,model.actions_ph:actions,model.next_observations_ph:next_obs}
        q1,q2,target=model.sess.run([policy.q1,policy.q2,model.value_target],feed)
        residual=q1.ravel()-(rewards+.97*(1-dones)*target.ravel())
        gridfeed={model.observations_ph:np.repeat(obs,len(GRID),axis=0),
            model.actions_ph:np.tile((GRID-1)/24.5-1,len(obs)).astype(np.float32).reshape(-1,1)}
        grid1,grid2=model.sess.run([policy.q1,policy.q2],gridfeed)
        grid1=grid1.reshape(len(obs),-1);grid2=grid2.reshape(len(obs),-1)
        qactor=model.sess.run(policy.q1,{model.observations_ph:obs,model.actions_ph:det}).ravel()
        actor_h=np.clip(np.rint(1+(det.ravel()+1)*24.5),1,50).astype(int)
        best_h=GRID[grid1.argmax(1)]
        row={'name':name,'sample_n':len(obs),'sample_source':sources[str(seed)],
            'actor_H_histogram':{str(h):int(sum(actor_h==h)) for h in range(1,51)},
            'mean_actor_H':float(actor_h.mean()),'Q1_greedy_mean_H':float(best_h.mean()),
            'latent_std_median':float(np.median(std)),'latent_std_p95':float(np.percentile(std,95)),
            'mean_behavior_Q1':float(q1.mean()),'mean_actor_Q1':float(qactor.mean()),
            'actor_minus_behavior_Q1_mean':float((qactor-q1.ravel()).mean()),
            'TD_RMSE_teacher_rows':float(np.sqrt(np.mean(residual**2))),
            'TD_MAE_teacher_rows':float(np.mean(abs(residual))),
            'Q1_grid_span_median':float(np.median(np.ptp(grid1,axis=1))),
            'Q1_Q2_disagreement_mean':float(np.mean(abs(grid1-grid2))),
            'actor_H':actor_h.tolist(),'Q1_grid':grid1.tolist(),'Q2_grid':grid2.tolist(),
            'Q1_behavior':q1.ravel().tolist(),'Q1_actor':qactor.tolist(),'mu':mu.ravel().tolist(),'std':std.ravel().tolist()}
        profiles.append(row);policy.close()
        print(name,'H',round(row['mean_actor_H'],2),'TD',round(row['TD_RMSE_teacher_rows'],3),'Qgap',round(row['Q1_grid_span_median'],3),flush=True)
    write(OUT/'checkpoint_profile.json',{'scope':'Same600 recorded teacher states per seed, all fixed checkpoint evaluations; TD residual is self-consistency, not true value accuracy. No optimizer updates.',
        'replay_coverage':replay,'profiles':profiles})


if __name__=='__main__':main()
