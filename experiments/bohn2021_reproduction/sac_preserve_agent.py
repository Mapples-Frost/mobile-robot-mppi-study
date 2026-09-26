"""Exact-expectation categorical SAC with separate frozen policy priors."""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sac_preserve_common import OUT,OLD,TEACHER,ROOT,HS,LEGACY,ARMS,read,write,verify,state_features,rule_index,metric,sha

torch.set_num_threads(1)


def net(sizes):
    layers=[]
    for i,(a,b) in enumerate(zip(sizes[:-1],sizes[1:])):
        layers.append(nn.Linear(a,b))
        if i<len(sizes)-2:layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def digest(module):
    h=hashlib.sha256()
    for k,v in sorted(module.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Agent(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed)
        self.actor=net([19,64,64,7]);self.q1=net([19,256,256,7]);self.q2=net([19,256,256,7])
        self.t1=copy.deepcopy(self.q1);self.t2=copy.deepcopy(self.q2)
        for p in list(self.t1.parameters())+list(self.t2.parameters()):p.requires_grad_(False)
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        self.qo=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=3e-4)

    def distribution(self,x):
        lp=self.actor(x).log_softmax(-1);return lp.exp(),lp

    def action(self,obs,rng=None):
        with torch.no_grad():p=self.distribution(torch.tensor(state_features(obs)[None]))[0][0].numpy()
        return int(p.argmax() if rng is None else rng.choice(7,p=p/p.sum()))

    def update(self,batch,prior=None,weight=0.,critic_only=False):
        x,a,r,n,d=batch
        with torch.no_grad():
            p,lp=self.distribution(n)
            target=r+.97*(1-d)*(p*(torch.minimum(self.t1(n),self.t2(n))-.01*lp)).sum(-1)
        q1=self.q1(x).gather(1,a[:,None]).squeeze(1);q2=self.q2(x).gather(1,a[:,None]).squeeze(1)
        qloss=nn.functional.smooth_l1_loss(q1,target)+nn.functional.smooth_l1_loss(q2,target)
        self.qo.zero_grad();qloss.backward()
        qnorm=nn.utils.clip_grad_norm_(list(self.q1.parameters())+list(self.q2.parameters()),10.)
        self.qo.step()
        aloss=ce=anorm=torch.tensor(0.)
        if not critic_only:
            p,lp=self.distribution(x)
            with torch.no_grad():q=torch.minimum(self.q1(x),self.q2(x))
            base=(p*(.01*lp-q)).sum(-1).mean()
            ce=-(prior.detach()*lp).sum(-1).mean() if prior is not None else torch.tensor(0.)
            aloss=base+weight*ce
            self.ao.zero_grad();aloss.backward();anorm=nn.utils.clip_grad_norm_(self.actor.parameters(),10.);self.ao.step()
        with torch.no_grad():
            for targetnet,online in [(self.t1,self.q1),(self.t2,self.q2)]:
                for t,s in zip(targetnet.parameters(),online.parameters()):t.mul_(.995).add_(s,alpha=.005)
        values=[float(v) for v in [qloss,aloss,ce,qnorm,anorm,target.abs().max()]]
        assert np.isfinite(values).all();return values


class ValuePrior:
    def __init__(self):
        self.nets=[]
        for seed in range(3):
            model=read(TEACHER/'models'/('round1_s%d.json'%seed));n=net([19,64,64,7])
            for layer,saved in zip([l for l in n if isinstance(l,nn.Linear)],model['layers']):
                layer.weight.data.copy_(torch.tensor(saved['weight']));layer.bias.data.copy_(torch.tensor(saved['bias']))
            n.requires_grad_(False);self.nets.append(n)
        self.known=torch.tensor(.003*(np.asarray(HS)-30)/.6,dtype=torch.float32)

    def costs(self,x):
        with torch.no_grad():
            values=[]
            for n in self.nets:
                v=n(x);values.append(v-v[:,HS.index(30):HS.index(30)+1]+self.known)
            return torch.stack(values).mean(0)

    def __call__(self,x):return (-self.costs(x)/.1).softmax(-1)


def rule_prior(indices):
    p=torch.full((len(indices),7),.01);p[torch.arange(len(indices)),indices]=.94;return p


class Replay:
    def __init__(self):self.rows=[]
    def add(self,row):
        self.rows.append((state_features(row['obs']),HS.index(row['horizon']),-row['cost']/.6,
                          state_features(row['next_obs']),float(row['done']),rule_index(row['obs'])))
    def sample(self,rng,n):return [self.rows[i] for i in rng.randint(0,len(self.rows),n)]


def tensor_batch(rows):
    columns=list(zip(*rows))
    tensors=[torch.tensor(np.asarray(v),dtype=torch.long if i in [1,5] else torch.float32) for i,v in enumerate(columns)]
    return tensors[:5],tensors[5]


def teacher_data(seed):
    replay=Replay();all_rows=0
    for line in (OLD/'models'/('teacher_s%d'%seed)/'transitions.jsonl').open():
        r=json.loads(line)
        if r['phase']!='teacher':continue
        all_rows+=1
        if r['horizon'] in HS:replay.add(r)
    assert all_rows==6000
    return replay


class Bridge:
    def __init__(self,seed):
        self.p=subprocess.Popen([LEGACY,'-u',str(Path(__file__).with_name('sac_preserve_worker.py'))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        self.call('init',seed=seed)
    def call(self,command,**kwargs):
        self.p.stdin.write(json.dumps(dict(command=command,**kwargs))+'\n');self.p.stdin.flush()
        while True:
            line=self.p.stdout.readline()
            if not line:raise RuntimeError('Legacy worker exited %s'%self.p.poll())
            if line.startswith('@RPC '):return json.loads(line[5:])
    def check(self,result):
        np.testing.assert_allclose(state_features(result['obs']),result['features'],atol=2e-6,rtol=2e-6)
        assert HS[rule_index(result['obs'])]==result['rule']
    def close(self):
        if self.p.poll() is None:self.p.stdin.write('{"command":"close"}\n');self.p.stdin.flush();self.p.wait(timeout=30)
        self.p.stdin.close();self.p.stdout.close()


def save(agent,path,metadata,replay=None,rngs=None):
    path.parent.mkdir(exist_ok=True,parents=True)
    obj={'model':agent.state_dict(),'actor_optim':agent.ao.state_dict(),'critic_optim':agent.qo.state_dict(),
         'metadata':metadata,'torch_rng':torch.get_rng_state(),'replay':replay,
         'rngs':{k:r.get_state() for k,r in (rngs or {}).items()}}
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
    write(path.with_suffix('.json'),dict(metadata,model_hash=digest(agent),file_sha256=sha(path)))


def load(path):
    s=torch.load(path,weights_only=False,map_location='cpu');a=Agent(s['metadata']['seed'])
    a.load_state_dict(s['model']);a.ao.load_state_dict(s['actor_optim']);a.qo.load_state_dict(s['critic_optim'])
    assert digest(a)==read(path.with_suffix('.json'))['model_hash'];return a


def pretrain(seed,smoke=False):
    verify();folder=OUT/('smoke' if smoke else 'pretrained');folder.mkdir(exist_ok=True)
    path=folder/('s%d.pt'%seed)
    if path.exists():return
    a=Agent(seed);initial=digest(a);replay=teacher_data(seed);rng=np.random.RandomState(260919160+seed)
    steps=8 if smoke else 1000;losses=[]
    q_before=digest(a.q1)
    for i in range(steps):
        batch,ri=tensor_batch(replay.sample(rng,256));_,lp=a.distribution(batch[0]);loss=-(rule_prior(ri)*lp).sum(-1).mean()
        a.ao.zero_grad();loss.backward();nn.utils.clip_grad_norm_(a.actor.parameters(),10.);a.ao.step()
        if (i+1)%100==0:losses.append({'phase':'actor','update':i+1,'loss':float(loss)})
    assert q_before==digest(a.q1);actor_before=digest(a.actor)
    for i in range(steps):
        batch,_=tensor_batch(replay.sample(rng,256));v=a.update(batch,critic_only=True)
        if (i+1)%100==0:losses.append({'phase':'critic','update':i+1,'values':v})
    assert actor_before==digest(a.actor)
    save(a,path,{'seed':seed,'initial_hash':initial,'imitation_updates':steps,'critic_updates':steps,'teacher_rows_retained':len(replay.rows)},rngs={'replay':rng})
    write(folder/('s%d_losses.json'%seed),losses)
    print('pretrained',seed,len(replay.rows),digest(a),flush=True)


def train(arm,seed,smoke=False):
    verify();source=OUT/('smoke' if smoke else 'pretrained')/('s%d.pt'%seed)
    folder=OUT/('smoke_models' if smoke else 'models')/('%s_s%d'%(arm,seed));folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():return
    assert not (folder/'transitions.jsonl').exists(),'Partial run preserved; explicit recovery required'
    agent=load(source);start_hash=digest(agent);teacher=teacher_data(seed);online=Replay();value=ValuePrior()
    rr=np.random.RandomState(260919170+seed);ar=np.random.RandomState(260919180+seed)
    bank=read(OUT/('train_s%d_bank.json'%seed))['scenes'];env=Bridge(seed)
    steps=120 if smoke else 15000;episode=0;updates=0;losses=[];started=time.monotonic()
    write(folder/'manifest.json',{'arm':arm,'seed':seed,'initial_hash':start_hash,'source':str(source.relative_to(ROOT)),
        'teacher_rows_retained':len(teacher.rows),'target':steps,'smoke':smoke})
    try:
        result=env.call('reset',case=bank[episode]['case']);env.check(result);obs=result['obs']
        with (folder/'transitions.jsonl').open('w') as stream:
            for t in range(1,steps+1):
                idx=agent.action(obs,ar)
                result=env.call('step',horizon=HS[idx],episode=episode);env.check(result)
                row=result['row'];row['online_step']=t;row['action_index']=idx
                online.add(row);stream.write(json.dumps(row)+'\n');obs=result['obs']
                if t>=100:
                    batch,ri=tensor_batch(teacher.sample(rr,102)+online.sample(rr,154))
                    prior=None if arm=='free' else rule_prior(ri) if arm=='rule' else value(batch[0])
                    weight=0 if arm=='free' else 2-1.5*t/15000
                    values=agent.update(batch,prior,weight);updates+=1
                    if updates%100==0:losses.append({'update':updates,'online_step':t,'values':values,'prior_weight':weight})
                if t%500==0 or t==steps:
                    stream.flush();write(folder/'progress.json',{'online_step':t,'updates':updates,'episodes':episode+1,'elapsed_s':time.monotonic()-started})
                    write(folder/'losses.json',losses);print('train',arm,seed,t,round(time.monotonic()-started,1),flush=True)
                if t%5000==0 or t==steps:
                    save(agent,folder/('step_%05d.pt'%t),{'seed':seed,'arm':arm,'online_step':t,'updates':updates,'initial_hash':start_hash},replay=online.rows,rngs={'replay':rr,'action':ar})
                if result['done'] and t<steps:
                    episode+=1;result=env.call('reset',case=bank[episode]['case']);env.check(result);obs=result['obs']
        write(folder/'completed.json',{'arm':arm,'seed':seed,'steps':steps,'updates':updates,'resets':episode+1,'initial_hash':start_hash,'final_hash':digest(agent)})
    finally:env.close()


def evaluate(name,split):
    verify();folder=OUT/'evaluation'/split/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():return
    a=None;value=None
    if name.startswith('pretrained_'):a=load(OUT/'pretrained'/('s%s.pt'%name.split('_s')[1]))
    # Match the arm_seed_step shape, not just the arm prefix: the reference
    # conditions 'rule' and 'value_teacher' also begin with an arm name and
    # must not be routed to the checkpoint loader.
    elif len(name.split('_'))==3 and name.split('_')[0] in ARMS:
        arm,seed,step=name.split('_');a=load(OUT/'models'/(arm+'_'+seed)/('step_%05d.pt'%int(step)))
    elif name=='value_teacher':value=ValuePrior()
    before=digest(a) if a else None;env=Bridge(260919190);episodes=[]
    try:
        for scene in read(OUT/(split+'_bank.json'))['scenes']:
            path=folder/('scene_%02d.json'%scene['id'])
            if path.exists():episodes.append(read(path)['summary']);continue
            result=env.call('reset',case=scene['case']);env.check(result);obs=result['obs'];initial=obs;trace=[]
            while True:
                if a:h=HS[a.action(obs)]
                elif value:h=HS[int(value.costs(torch.tensor(state_features(obs)[None])).argmin())]
                elif name=='rule':h=HS[rule_index(obs)]
                elif name=='fixed25':h=25
                else:raise ValueError(name)
                result=env.call('step',horizon=h);env.check(result);trace.append(result['row']);obs=result['obs']
                if result['done']:break
            summary=dict(metric(trace),scene=scene['id'],name=name)
            write(path,{'initial':initial,'summary':summary,'trace':trace});episodes.append(summary)
            print('eval',name,split,scene['id'],round(summary['adjusted_cost'],3),flush=True)
        if a:assert before==digest(a)
        write(folder/'completed.json',{'episodes':episodes,'model_hash':before})
    finally:env.close()


def smoke():
    pretrain(0,True);train('value',0,True)
    replay=teacher_data(0);batch,ri=tensor_batch(replay.sample(np.random.RandomState(42),16));a=Agent(42)
    with torch.no_grad():
        p,lp=a.distribution(batch[3]);v=(p*(torch.minimum(a.t1(batch[3]),a.t2(batch[3]))-.01*lp)).sum(-1)
        loop=torch.stack([sum(p[i,k]*(min(a.t1(batch[3])[i,k],a.t2(batch[3])[i,k])-.01*lp[i,k]) for k in range(7)) for i in range(16)])
        torch.testing.assert_close(v,loop)
        y=batch[2]+.97*(1-torch.ones_like(batch[4]))*v;torch.testing.assert_close(y,batch[2])
    prior=ValuePrior();from teacher_common import infer
    expected=np.mean([np.asarray([infer(read(TEACHER/'models'/('round1_s%d.json'%s)),x) for x in batch[0].numpy()]) for s in range(3)],axis=0)
    np.testing.assert_allclose(prior.costs(batch[0]).numpy(),expected,atol=3e-5,rtol=3e-5)
    path=OUT/'smoke_models/value_s0/step_00120.pt';loaded=load(path)
    assert digest(loaded)==read(path.with_suffix('.json'))['model_hash']
    write(OUT/'smoke.json',{'passed':True,'transitions':120,'resets':read(OUT/'smoke_models/value_s0/completed.json')['resets'],
        'actor_warmup':8,'critic_warmup':8,'joint_updates':21,'target_formula_and_terminal_mask':True,'prior_export_equivalence':True,'reload':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pretrain',type=int);p.add_argument('--train',choices=ARMS);p.add_argument('--seed',type=int);p.add_argument('--evaluate');p.add_argument('--split');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.smoke:smoke()
    elif args.pretrain is not None:pretrain(args.pretrain)
    elif args.train:train(args.train,args.seed)
    else:evaluate(args.evaluate,args.split)
