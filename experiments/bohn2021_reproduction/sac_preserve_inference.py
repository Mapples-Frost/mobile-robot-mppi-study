"""NumPy-only deployment inference for exported teacher-preserving SAC actors.

Input: original 56-dimensional causal pendulum observation. Output: integer H.
No privileged future, critic, teacher model or training runtime is needed.
"""
import json
import numpy as np


def encode(observation):
    o=np.asarray(observation,dtype=np.float32)
    if o.shape!=(56,) or not np.isfinite(o).all():raise ValueError('Expected finite56D observation')
    pos=float(o[0])*1.5;ref=float(o[4]);remaining=float(o[55])
    refs=o[5:55].astype(float)*1.5+pos
    refs[abs(refs-ref)<1e-6]=ref
    delta=refs-ref;changes=np.flatnonzero(abs(delta)>1e-8)
    first=int(changes[0]) if len(changes) else None
    values=[(pos-ref)/1.5,float(o[1])*5/2,float(o[2])*(np.pi/2)/.5,
            float(o[3])*10/3,pos/1.5,ref/1.5,remaining,
            (first+1)/50 if first is not None else 1.1,delta[first]/1.5 if first is not None else 0.]
    values+=(delta.reshape(10,5).mean(axis=1)/1.5).tolist()
    return np.asarray(values,dtype=np.float32)


class HorizonPolicy:
    def __init__(self,path):
        with open(path) as stream:self.model=json.load(stream)
        assert self.model['format']=='sac-preserve-actor-v1'
        self.horizons=np.asarray(self.model['horizons'],dtype=int)
        self.layers=[(np.asarray(x['weight'],np.float32),np.asarray(x['bias'],np.float32)) for x in self.model['layers']]

    def probabilities(self,observation):
        x=encode(observation)
        for i,(w,b) in enumerate(self.layers):
            x=x@w.T+b
            if i<len(self.layers)-1:x=np.maximum(x,0)
        p=np.exp(x-x.max());return p/p.sum()

    def predict(self,observation):
        return int(self.horizons[int(self.probabilities(observation).argmax())])
