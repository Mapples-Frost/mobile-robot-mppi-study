"""Frozen diagnostic on historical validation anchors; never selects models."""
import argparse
import numpy as np
from sac_teacher_study import OUT, ART, SEEDS, imports, read, write, weights_hash, verify, sha


def observations(context):
    s=context['state'];refs=context['refs']
    return np.asarray([s['pos']/1.5,s['v']/5,s['theta']/(np.pi/2),s['omega']/10,refs[0]]+
        [(r-s['pos'])/1.5 for r in refs[1:]]+[(600-context['elapsed'])/600],np.float32)


def main(smoke=False):
    verify()
    from optimized_runtime import install_terminal
    install_terminal('pendulum');_,SAC,_=imports()
    anchors=[]
    old=ART/'results/teacher_value/labels/validation'
    paths=sorted(old.glob('*/scene_*/anchors.json'))
    for p in paths:anchors.extend(read(p))
    assert len(anchors)==48
    obs=np.stack([observations(a['context']) for a in anchors])
    hs=np.array([5,10,20,25,30,40,50])
    costs=np.array([a['costs'] for a in anchors])
    targets=costs-costs.min(1,keepdims=True)
    candidates=[('smoke',OUT/'smoke/teacher_s0/step_00300.zip')] if smoke else [
        ('%s_s%d_%d'%(arm,seed,step),OUT/'models'/('%s_s%d'%(arm,seed))/('step_%05d.zip'%step))
        for arm,step in [('plain',15000),('plain',21000),('teacher',15000)] for seed in SEEDS]
    results=[]
    for name,path in candidates:
        model=SAC.load(str(path));model.reward_scale=.6;before=weights_hash(model)
        assert before==read(path.with_name(path.stem+'_metadata.json'))['weights_hash']
        q1=model.graph.get_tensor_by_name('model/values_fn/qf1/qf1/BiasAdd:0')
        q2=model.graph.get_tensor_by_name('model/values_fn/qf2/qf2/BiasAdd:0')
        feed={model.observations_ph:np.repeat(obs,len(hs),axis=0),
              model.actions_ph:np.tile(2*(hs-1)/49.-1,len(obs)).reshape(-1,1)}
        v1,v2=model.sess.run([q1,q2],feed)
        v1=v1.reshape(len(obs),len(hs));v2=v2.reshape(len(obs),len(hs))
        pick1=v1.argmax(1);pickmin=np.minimum(v1,v2).argmax(1)
        actor=model.predict(obs,deterministic=True)[0].ravel()
        actor_h=np.clip(np.rint(actor),1,50).astype(int)
        nearest=np.abs(hs[None,:]-actor_h[:,None]).argmin(1)
        row={'arm':name,'anchors':len(obs),'reloaded_weights_match_saved':True,'model_sha256':sha(path),'actor_H':actor_h.tolist(),
            'Q1_selected_H':hs[pick1].tolist(),'minQ_selected_H':hs[pickmin].tolist(),
            'teacher_best_H':hs[costs.argmin(1)].tolist(),
            'Q1_policy_specific_regret':float(targets[np.arange(len(obs)),pick1].mean()),
            'minQ_policy_specific_regret':float(targets[np.arange(len(obs)),pickmin].mean()),
            'nearest_candidate_to_actor_regret':float(targets[np.arange(len(obs)),nearest].mean()),
            'nearest_actor_Q1_disagreements':int(sum(nearest!=pick1)),
            'Q1':v1.tolist(),'Q2':v2.tolist()}
        assert weights_hash(model)==before;model.sess.close();results.append(row)
    result={'scope':'Historical48 validation anchors only; frozen diagnostic, no tuning. Branch labels execute candidate once then H5/H30 rule with approximate tail and no entropy. They are NOT SAC soft-Q targets. Regret is policy-specific and actor uses nearest candidate, not actual continuous-action branch evaluation.',
        'source_paths':[str(p) for p in paths],'source_sha256':{str(p):sha(p) for p in paths},'results':results}
    write(OUT/('smoke_diagnostic.json' if smoke else 'critic_diagnostic.json'),result)
    print([{k:r[k] for k in ['arm','Q1_policy_specific_regret','minQ_policy_specific_regret','nearest_actor_Q1_disagreements']} for r in results])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();main(args.smoke)
