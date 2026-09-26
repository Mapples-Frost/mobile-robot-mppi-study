"""Post-hoc frozen-network input sensitivity; no simulated transitions."""
import numpy as np
from sac_teacher_cause import OUT, GRID, Policy, models, read, write, verify, sha
from pathlib import Path


def main():
    verify()
    from optimized_runtime import install_terminal
    install_terminal('pendulum')
    samples=[]
    for scene in read(OUT/'bank.json')['scenes']:
        rows=read(OUT/'evaluation/rule__det'/('scene_%02d.json'%scene['id']))['trace']
        for switch in scene['switches']:
            for lead in [40,25,10]:
                # Before executing row t, obs contains the reference at clock t.
                row=rows[switch-lead-1]
                assert row['step']==switch-lead
                obs=np.asarray(row['obs'],np.float32)
                held=obs.copy()
                held[5:55]=(obs[4]-obs[0]*1.5)/1.5
                samples.append({'scene':scene['id'],'switch':switch,'lead':lead,
                    'obs':obs.tolist(),'held_preview_obs':held.tolist()})
    outputs=[]
    for name in models():
        policy=Policy(name);m=policy.model;p=m.policy_tf
        original=np.asarray([r['obs'] for r in samples],np.float32)
        held=np.asarray([r['held_preview_obs'] for r in samples],np.float32)
        decisions=[]
        for observations in [original,held]:
            norm=m.sess.run(p.deterministic_policy,{p.obs_ph:observations}).ravel()
            decisions.append(np.clip(np.rint(1+(norm+1)*24.5),1,50).astype(int))
        q1,q2=m.sess.run([policy.q1,policy.q2],{
            m.observations_ph:np.repeat(original,len(GRID),axis=0),
            m.actions_ph:np.tile((GRID-1)/24.5-1,len(original)).astype(np.float32).reshape(-1,1)})
        q1=q1.reshape(len(original),-1);q2=q2.reshape(len(original),-1)
        by_lead={}
        for lead in [40,25,10]:
            mask=np.asarray([r['lead']==lead for r in samples]);real=decisions[0][mask];flat=decisions[1][mask]
            by_lead[str(lead)]={'n':int(sum(mask)),'actual_preview_mean_H':float(real.mean()),
                'held_preview_mean_H':float(flat.mean()),'actual_minus_held_mean_H':float((real-flat).mean()),
                'actual_H':real.tolist(),'held_H':flat.tolist(),
                'actual_preview_Q1_H30_minus_H5':float((q1[mask,7]-q1[mask,3]).mean()),
                'actual_preview_Q1_argmax_mean_H':float(GRID[q1[mask].argmax(axis=1)].mean())}
        outputs.append({'name':name,'by_lead':by_lead,'Q1_grid':q1.tolist(),'Q2_grid':q2.tolist()});policy.close()
        print(name,by_lead['25']['actual_preview_mean_H'],by_lead['25']['held_preview_mean_H'],flush=True)
    write(OUT/'preview_sensitivity.json',{'scope':'Post-hoc network input intervention on 24 rule states. Hold physical state, current reference and time fixed; replace only future reference by current reference. Not a physical outcome or training causal ablation.',
        'SAC_updates':0,'physical_steps':0,'source_sha256':sha(Path(__file__)),
        'samples':samples,'models':outputs})


if __name__=='__main__':main()
