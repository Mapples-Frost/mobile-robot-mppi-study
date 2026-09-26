"""Independent switch/bridge and optimizer isolation checks before formal runs."""
import json
import numpy as np
import torch
from sac_preserve_agent import Bridge,Agent,teacher_data,tensor_batch,digest,rule_prior
from sac_preserve_common import OUT,HS,read,write,state_features,rule_index,sha
from sac_teacher_report import audit_row
from pathlib import Path


def main():
    scene=read(OUT/'train_s0_bank.json')['scenes'][0];env=Bridge(0)
    result=env.call('reset',case=scene['case']);initial=result['obs'];trace=[];prev=None
    try:
        while True:
            env.check(result)
            result=env.call('step',horizon=HS[rule_index(result['obs'])])
            row=result['row'];audit_row(row,scene,len(trace)+1,prev);prev=row;trace.append(row)
            if result['done']:break
        env.check(result)
    finally:env.close()
    assert len(trace)==600 and not any(not r['solver_success'] for r in trace)
    write(OUT/'smoke_switch_trace.json',{'initial':initial,'trace':trace})
    a=Agent(40);batch,ri=tensor_batch(teacher_data(0).sample(np.random.RandomState(0),256))
    ahash=digest(a.actor);qhash=digest(a.q1);a.update(batch,critic_only=True)
    assert digest(a.actor)==ahash and digest(a.q1)!=qhash
    x=batch[0];prior=rule_prior(ri);before=-(prior*a.distribution(x)[1]).sum(-1).mean().item()
    for _ in range(20):
        _,lp=a.distribution(x);loss=-(prior*lp).sum(-1).mean()
        a.ao.zero_grad();loss.backward();a.ao.step()
    after=-(prior*a.distribution(x)[1]).sum(-1).mean().item();assert after<before
    write(OUT/'checks.json',{'passed':True,'full_switch_physical_steps':601,'rule_and_features_equivalent_every_step':True,
        'teacher_prior_ce_before':before,'teacher_prior_ce_after':after,'critic_only_actor_unchanged':True,
        'unit_critic_updates':1,'unit_actor_updates':20,'source_sha256':sha(Path(__file__))})
    print('Full switch bridge and optimizer checks passed',flush=True)


if __name__=='__main__':main()
