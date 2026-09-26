"""Portable NumPy inference and CPU Torch fitting for conservative iteration."""
import argparse
import json
from pathlib import Path
import numpy as np
from conservative_iteration import OUT, HS, BASE, TASKS, protocol, verify, prior_model
from paper_h_soft_probe import read, digest
from run import write


def predict(model, features):
    x=(np.asarray(features,dtype=np.float32)-np.asarray(model['mean'],np.float32))/np.asarray(model['scale'],np.float32)
    costs=[];safety=[]
    for member in model['members']:
        y=x.copy()
        for j,layer in enumerate(member):
            y=y@np.asarray(layer['weight'],np.float32).T+np.asarray(layer['bias'],np.float32)
            if j<len(member)-1:y=np.maximum(y,0)
        q=y[:len(HS)]-y[HS.index(model['base_h'])]
        costs.append(q)
        z=np.clip(y[len(HS):],-60,60)
        safety.append(1/(1+np.exp(-z)))
    support=np.asarray(model['support'],np.float32)
    distance=float(np.sqrt(np.mean((support-x)**2,axis=1)).min())
    return np.asarray(costs),np.asarray(safety),distance


def choose(model, features):
    previous=choose(model['prior'],features) if model['prior'] else model['base_h']
    q,safe,distance=predict(model,features)
    if distance>model['support_radius']:return previous
    ref=HS.index(previous)
    candidates=[i for i in range(len(HS)) if np.all(q[:,i]<q[:,ref]-.1) and np.all(safe[:,i]>=.8)]
    if not candidates:return previous
    best=min(candidates,key=lambda i:(float(q[:,i].mean()),HS[i]))
    return HS[best]


def register():
    verify()
    files=[Path(__file__),Path(__file__).with_name('conservative_iteration.py'),
           Path(__file__).with_name('relative_policy_features.py'),OUT/'protocol.json']
    hashes={str(p):digest(p) for p in files}
    path=OUT/'learner_registration.json'
    if path.exists():
        # Compare file identities, independent of absolute/relative CLI spelling.
        canonical=lambda mapping:{str(Path(p).resolve()):h for p,h in mapping.items()}
        assert canonical(read(path)['hashes'])==canonical(hashes),'Learner implementation changed'
    else:write(path,dict(hashes=hashes,torch='2.4.1+cpu',numpy=np.__version__,
        normalization='Training-only feature mean and standard deviation floored at0.05',
        support='RMS Euclidean distance in normalized feature space; leave-one-scene-out 95th percentile',
        seed_rule='training_seed*100 + member_id; same initialization/bootstrap per round',
        updates=3000,checkpoint='Final only; no validation loss or checkpoint selection'))


def fit(task,seed,round_id,smoke=False):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    register()
    dest=OUT/('smoke_'+task if smoke else '%s_s%d_r%d'%(task,seed,round_id))
    data=read(dest/'collection_completed.json')
    assert data['audit_passed'] and data['smoke']==smoke
    for path,h in data['hashes'].items():assert digest(Path(path))==h
    output=dest/('smoke_policy.json' if smoke else 'policy.json')
    completed=dest/('smoke_fit_completed.json' if smoke else 'fit_completed.json')
    if completed.exists():
        done=read(completed)
        assert done['policy_hash']==digest(output) and done['dataset_hash']==digest(dest/'collection_completed.json')
        return
    groups=data['groups']
    assert groups
    x=np.array([g['context']['features'] for g in groups],np.float32)
    ids=np.array([g['case'] for g in groups])
    scenes=np.unique(ids)
    assert smoke or len(scenes)>=2,'Cannot estimate support without multiple scenes'
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),.05)
    z=(x-mean)/scale
    distances=np.sqrt(np.mean((z[:,None,:]-z[None,:,:])**2,axis=2))
    distances[ids[:,None]==ids[None,:]]=np.inf
    radius=float(np.percentile(distances.min(axis=1),95)) if not smoke else 1e6
    y=[];safe=[]
    for g in groups:
        base=g['branches'][str(BASE[task])]
        source=g['branches'][str(g['source_h'])]
        yr=[];sr=[]
        for h in HS:
            # Smoke has two actions; unobserved actions are fixed-zero safe=false,
            # strictly technical fit/reload testing, never a deployment candidate.
            b=g['branches'].get(str(h))
            if b is None:
                assert smoke;yr.append(0.);sr.append(0.);continue
            d=(b['total_cost']-base['total_cost'])/.1
            yr.append(np.sign(d)*np.log1p(abs(d)))
            sr.append(float(b['success']>=source['success'] and b['constraint']<=source['constraint']
                      and b['solver_failure_steps']<=source['solver_failure_steps']))
        y.append(yr);safe.append(sr)
    y=np.asarray(y,np.float32);safe=np.asarray(safe,np.float32)
    tx=torch.from_numpy(z);ty=torch.from_numpy(y);ts=torch.from_numpy(safe)
    model=dict(task=task,seed=seed,round=round_id,base_h=BASE[task],horizons=list(HS),
        mean=mean.tolist(),scale=scale.tolist(),support=z.tolist(),support_radius=radius,
        prior=prior_model(task,seed,round_id),members=[])
    logs=[];torch_predictions=[]
    for member in range(3):
        fit_seed=seed*100+member
        torch.manual_seed(fit_seed);rng=np.random.RandomState(fit_seed)
        network=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2*len(HS)))
        optimizer=torch.optim.Adam(network.parameters(),lr=.0003)
        scene_sample=rng.choice(scenes,size=len(scenes),replace=True)
        pool=np.concatenate([np.flatnonzero(ids==cid) for cid in scene_sample])
        def loss(indices):
            out=network(tx[indices]);q=out[:,:len(HS)]-out[:,HS.index(BASE[task]):HS.index(BASE[task])+1]
            return nn.functional.smooth_l1_loss(q,ty[indices])+nn.functional.binary_cross_entropy_with_logits(out[:,len(HS):],ts[indices])
        initial=float(loss(np.arange(len(x))).detach())
        history=[]
        updates=20 if smoke else 3000
        for i in range(updates):
            indices=rng.choice(pool,size=32,replace=True)
            optimizer.zero_grad();value=loss(indices);value.backward();optimizer.step()
            if (i+1)%100==0 or i+1==updates:history.append(dict(update=i+1,all_data_loss=float(loss(np.arange(len(x))).detach())))
        assert np.isfinite([initial]+[v['all_data_loss'] for v in history]).all()
        layers=[]
        for layer in network:
            if isinstance(layer,nn.Linear):layers.append(dict(weight=layer.weight.detach().numpy().tolist(),bias=layer.bias.detach().numpy().tolist()))
        model['members'].append(layers)
        torch_predictions.append(network(tx).detach().numpy())
        logs.append(dict(member=member,seed=fit_seed,bootstrap_scenes=scene_sample.tolist(),
            initial_loss=initial,history=history,updates=updates))
    write(output,model)
    loaded=read(output)
    choices=[]
    for k,features in enumerate(x):
        q,s,d=predict(loaded,features)
        for j in range(3):
            expected=torch_predictions[j][k]
            expected_q=expected[:len(HS)]-expected[HS.index(BASE[task])]
            np.testing.assert_allclose(q[j],expected_q,atol=5e-5,rtol=5e-5)
            np.testing.assert_allclose(s[j],1/(1+np.exp(-np.clip(expected[len(HS):],-60,60))),atol=5e-5,rtol=5e-5)
        choices.append(choose(loaded,features))
    diagnostics=[]
    for g,h in zip(groups,choices):
        assert str(h) in g['branches'] or smoke
        if str(h) not in g['branches']:continue
        ref=g['branches'][str(g['source_h'])];b=g['branches'][str(h)]
        diagnostics.append(dict(case=g['case'],anchor=g['anchor'],source_h=g['source_h'],selected_h=h,
            actual_label_cost_difference=b['total_cost']-ref['total_cost'],
            actual_safe=b['success']>=ref['success'] and b['constraint']<=ref['constraint'] and b['solver_failure_steps']<=ref['solver_failure_steps']))
    write(completed,dict(task=task,seed=seed,round=round_id,smoke=smoke,
        policy_hash=digest(output),dataset_hash=digest(dest/'collection_completed.json'),
        learner_registration_hash=digest(OUT/'learner_registration.json'),groups=len(groups),
        scenes=len(scenes),members=logs,inference_reload_passed=True,training_diagnostics=diagnostics,
        training_updates=sum(l['updates'] for l in logs),environment_steps=0))
    print(json.dumps(dict(task=task,seed=seed,round=round_id,groups=len(groups),
        selected_changes=sum(d['selected_h']!=d['source_h'] for d in diagnostics),policy=str(output))))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=TASKS);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--round',type=int,default=0);ap.add_argument('--smoke',action='store_true');ap.add_argument('--register',action='store_true')
    a=ap.parse_args()
    if a.register:register()
    else:fit(a.task,a.seed,a.round,a.smoke)
