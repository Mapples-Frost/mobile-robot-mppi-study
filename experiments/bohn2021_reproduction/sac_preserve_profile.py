"""Read-only fixed teacher-state checkpoint profile; no training or simulation."""
import json
import numpy as np
import torch
from sac_preserve_common import OUT,OLD,ARMS,HS,read,write,state_features,rule_index,sha
from sac_preserve_agent import load,ValuePrior
from pathlib import Path


def main():
    value=ValuePrior();profiles=[]
    for seed in range(3):
        records=[]
        for line in (OLD/'models'/('teacher_s%d'%seed)/'transitions.jsonl').open():
            r=json.loads(line)
            if r['phase']=='teacher':records.append(r)
        records=records[::20];obs=np.asarray([r['obs'] for r in records]);x=torch.tensor(np.asarray([state_features(o) for o in obs]))
        rule=np.asarray([rule_index(o) for o in obs]);held=obs.copy()
        for row in held:row[5:55]=(row[4]-row[0]*1.5)/1.5
        flat=torch.tensor(np.asarray([state_features(o) for o in held]));visible=np.asarray([abs(r[8])>1e-6 for r in x.numpy()])
        paths=[OUT/'pretrained'/('s%d.pt'%seed)]+[OUT/'models'/('%s_s%d'%(arm,seed))/('step_%05d.pt'%t) for arm in ARMS for t in [5000,10000,15000]]
        for path in paths:
            if not path.exists():continue
            a=load(path)
            with torch.no_grad():
                prob=a.distribution(x)[0].numpy();chosen=prob.argmax(1);flat_chosen=a.actor(flat).argmax(1).numpy()
                costs=value.costs(x).numpy();q=torch.minimum(a.q1(x),a.q2(x)).numpy()
            profiles.append({'checkpoint':str(path.relative_to(OUT)),'seed':seed,'fixed_teacher_states':len(records),
                'rule_agreement':float(np.mean(chosen==rule)),
                'mean_rule_probability':float(np.mean(prob[np.arange(len(rule)),rule])),
                'mean_H':float(np.mean(np.asarray(HS)[chosen])),
                'visible_change_states':int(visible.sum()),
                'visible_change_mean_H':float(np.asarray(HS)[chosen[visible]].mean()),
                'held_preview_mean_H_on_visible_states':float(np.asarray(HS)[flat_chosen[visible]].mean()),
                'frozen_teacher_regret':float(np.mean(costs[np.arange(len(rule)),chosen]-costs.min(1))),
                'actor_vs_Q_argmax_disagreement':float(np.mean(chosen!=q.argmax(1)))})
    write(OUT/'preservation_profile.json',{'source_sha256':sha(Path(__file__)),
        'scope':'Every20th of6000 teacher transitions per seed, same300 observations across checkpoints. Frozen teacher regret is approximate teacher-continuation reference, not true SAC-Q error.',
        'profiles':profiles})
    print('profiled',len(profiles),'checkpoints')


if __name__=='__main__':main()
