"""Legacy process bridge: same frozen environment, rule checked from full state."""
import json
import sys
from sac_teacher_study import environment,transition,terminal_check
from plateau_screen import action
from run import serial
from teacher_worker import context


def main():
    env=None;obs=None;t=0
    for line in sys.stdin:
        r=json.loads(line);cmd=r['command']
        if cmd=='close':break
        if cmd=='init':env=environment(r['seed']);result={'ok':True}
        elif cmd=='reset':
            obs=env.reset(**r['case']);t=0
            result={'obs':obs,'features':context(env)['features'],'rule':action(env,'switch_5_30')[0]}
        elif cmd=='step':
            h=r['horizon'];assert isinstance(h,int) and 1<=h<=50
            t+=1;obs,done,row=transition(env,obs,h,(h-1)/24.5-1,'online',r.get('episode',0),t)
            terminal_check(env)
            result={'obs':obs,'done':done,'row':row,'features':context(env)['features'],'rule':action(env,'switch_5_30')[0]}
        else:raise ValueError(cmd)
        print('@RPC '+json.dumps(result,default=serial,allow_nan=False),flush=True)


def old_evaluate(name):
    import numpy as np
    from sac_preserve_common import OUT,read,write,verify,metric
    from sac_teacher_cause import Policy
    verify();folder=OUT/'evaluation/holdout'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():return
    arm,seed=name.split('_');label='step_21000' if arm=='plain' else 'step_15000'
    # Build the environment BEFORE loading the policy. environment() calls
    # install_terminal('pendulum'), which installs the MPC terminal value
    # function; the saved policies use mpc_vf_type='poly', whose kernel is the
    # concatenation of the MPC feature vector and its square. Loading first
    # builds that kernel at half width and SAC.load dies feeding (20,1) into a
    # (10,1) placeholder. Ordering only -- no model, weight or protocol change.
    env=environment(260919190);policy=Policy(name+'_'+label);episodes=[]
    for scene in read(OUT/'holdout_bank.json')['scenes']:
        path=folder/('scene_%02d.json'%scene['id'])
        if path.exists():episodes.append(read(path)['summary']);continue
        obs=env.reset(**scene['case']);initial=obs.tolist();trace=[]
        while True:
            a,decision=policy.choose(obs,'det',np.random.RandomState(0))
            obs,done,row=transition(env,obs,a,(a-1)/24.5-1,'evaluation',0,len(trace)+1)
            trace.append(row)
            if done:break
        terminal_check(env);summary=dict(metric(trace),scene=scene['id'],name=name)
        write(path,{'initial':initial,'summary':summary,'trace':trace});episodes.append(summary)
        print('old eval',name,scene['id'],round(summary['adjusted_cost'],3),flush=True)
    policy.close();write(folder/'completed.json',{'episodes':episodes})


if __name__=='__main__':
    if len(sys.argv)>1:old_evaluate(sys.argv[1])
    else:main()
