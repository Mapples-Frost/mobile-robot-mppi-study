"""Export all final actors and independently verify NumPy inference."""
import json
import numpy as np
import torch
from torch import nn
from sac_preserve_common import OUT,OLD,ARMS,HS,read,write,state_features,sha
from sac_preserve_agent import load,digest
from sac_preserve_inference import HorizonPolicy,encode
from pathlib import Path


def main():
    dest=OUT/'export';dest.mkdir(exist_ok=True);checks=[]
    for seed in range(3):
        obs=[]
        with (OLD/'models'/('teacher_s%d'%seed)/'transitions.jsonl').open() as stream:
            for i,line in enumerate(stream):
                r=json.loads(line)
                if r['phase']=='teacher' and i%20==0:obs.append(r['obs'])
        for o in obs:np.testing.assert_allclose(encode(o),state_features(o),atol=2e-6,rtol=2e-6)
        for arm in ARMS:
            name='%s_s%d'%(arm,seed);path=OUT/'models'/name/'step_15000.pt'
            if not path.exists():continue
            a=load(path);model={'format':'sac-preserve-actor-v1','arm':arm,'seed':seed,
                'source_model_hash':digest(a),'horizons':HS,'observation':'Original56D causal preview, converted to19D internally',
                'terminal_requirement':'Same frozen float32 Riccati MPC terminal; not validated for other plants or controllers.',
                'layers':[{'weight':l.weight.detach().numpy().tolist(),'bias':l.bias.detach().numpy().tolist()} for l in a.actor if isinstance(l,nn.Linear)]}
            output=dest/(name+'.json');write(output,model);policy=HorizonPolicy(output)
            with torch.no_grad():expected=a.distribution(torch.tensor(np.asarray([state_features(o) for o in obs])))[0].numpy()
            actual=np.asarray([policy.probabilities(o) for o in obs])
            np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
            np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
            checks.append({'model':name,'observations':len(obs),'max_probability_error':float(abs(actual-expected).max()),'sha256':sha(output)})
    write(OUT/'export_audit.json',{'models':checks,'inference_sha256':sha(Path(__file__).with_name('sac_preserve_inference.py'))})
    print('Exported and verified',len(checks),'actors')


if __name__=='__main__':main()
