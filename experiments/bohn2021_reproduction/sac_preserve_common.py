"""Registered teacher-preserving discrete SAC optimization, shared definitions."""
import hashlib
import json
from pathlib import Path
import numpy as np
from runtime import ROOT, ART
from run import write
from teacher_common import make_bank, features, infer, HS

OUT=ART/'results/sac_preserve'
OLD=ART/'results/sac_teacher'
TEACHER=ART/'results/teacher_value'
ARMS=['free','rule','value']
LEGACY='/home/mapples/.local/share/bohn2021-python37/bin/python'
MODERN=str(ROOT/'.venv/bin/python')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def state_features(obs):
    o=np.asarray(obs);pos=o[0]*1.5
    state={'pos':pos,'v':o[1]*5,'theta':o[2]*np.pi/2,'omega':o[3]*10}
    refs=np.r_[o[4],o[5:55]*1.5+pos]
    refs[abs(refs-refs[0])<1e-6]=refs[0]
    return features(state,refs,o[55]*600)


def rule_index(obs):
    o=np.asarray(obs);pos=o[0]*1.5
    upcoming=np.max(abs(o[5:35]*1.5+pos-o[4]))>1e-6
    unsettled=abs(pos-o[4])>.03 or abs(o[1]*5)>.08 or abs(o[2]*np.pi/2)>.04 or abs(o[3]*10)>.12
    return HS.index(30 if upcoming or unsettled else 5)


def prepare():
    from sac_teacher_study import verify as old_verify
    from teacher_common import verify as teacher_verify
    old_verify();teacher_verify();OUT.mkdir(exist_ok=True)
    if (OUT/'hashes.json').exists():verify();return
    protocol={'purpose':'Optimization after diagnosed lost anticipation. Three matched arms isolate online policy-preservation priors, not individual bundled representation changes.',
        'arms':{'free':'Discrete SAC after common rule imitation and critic warmup; no online prior',
                'rule':'Same plus smoothed causal-rule cross entropy',
                'value':'Same plus frozen ensemble branch-value soft policy cross entropy'},
        'seeds':[0,1,2],'online_steps':15000,'checkpoints':[0,5000,10000,15000],
        'shared':'19 causal preview features; seven integer H5,10,20,25,30,40,50; actor64x64, twinQ256x256; exact categorical SAC expectation, min twin targetQ; gamma.97 alpha.01 tau.005 lr3e-4; reward original/.6; finite terminals; frozen original Riccati MPC terminal.',
        'updates':'1000 actor-only imitation of causal rule on saved teacher states;1000 critic-only updates with actor fixed;14901 joint online updates from transition100. No checkpoint selection.',
        'replay':'Reuse each paired seed old teacher6000; keep only recorded integer H in candidate set, report retained count.102 teacher+154 online samples per batch256, with replacement; independent RNG streams for environment actions and replay. Same common initialization and warmup per seed.',
        'loss':'TwinQ SmoothL1(beta1), actor expected alpha log pi-minQ plus lambda*cross entropy; gradient norm cap10. lambda rule/value=2-(1.5*online_step/15000), ending.5; free lambda0. Priors stop-gradient, never inserted as SAC Q targets.',
        'rule_prior':'Probability.94 at causal H5/H30 choice, .01 at each other candidate. Shared warmup uses same target.',
        'value_prior':'Equal average of all three previously frozen round1 branch-value models; softmax(-relative_cost/.1). Uses only causal19D features; inherited approximate teacher-continuation labels, not true current-policy Q.',
        'test':{'validation':[260919131,4],'holdout':[260919132,12]},
        'training_banks':[260919140,260919141,260919142],
        'evaluation':'All9 new final policies on validation and holdout; all9 pretrain/5k/10k checkpoints on validation. Old3 plain/3 teacher final, rule, fixed25 and frozen value ensemble on holdout. Fixed25 is a locked historical reference, not claimed newly optimal. Evaluate no-cost ensemble as comparator to expose inherited teacher benefit.',
        'success':'Report all seeds; primary value-arm mean holdout cost vs new free arm and old plain. Reliability counts separately; no superiority to fixed/rule unless measured. No algorithm/hyperparameter changes after holdout.',
        'accounting':'135000 online transitions for9 models plus warmups/evaluations/smoke. Reused18000 teacher transitions and217730 historical branch-study physical steps disclosed separately as inherited data cost, not free samples or equal total budget vs old plain.',
        'limits':['Bundled discrete action/support/features/loss/warmup differ from author SAC.',
                  'Offline critic-only before imitation would value random continuation; shared rule imitation precedes critic-only warmup intentionally.',
                  'Preservation prior remains nonzero; success is a hybrid teacher-guided policy, not unconstrained SAC discovery.',
                  '3 training seeds; scenes are paired evaluations, not independent training replicates.',
                  'No reward/gamma change, solver retry or joint terminal training.'],
        'resources':'At most2 concurrent legacy MPC workers; no subagents.'}
    write(OUT/'protocol.json',protocol)
    for split,(seed,n) in protocol['test'].items():write(OUT/(split+'_bank.json'),make_bank(seed,n))
    for seed in range(3):write(OUT/('train_s%d_bank.json'%seed),make_bank(260919140+seed,100))
    paths=[Path(__file__)]+[Path(__file__).with_name(n) for n in ['sac_preserve_agent.py','sac_preserve_worker.py','sac_preserve_suite.py']]
    paths += [OUT/'protocol.json']+list(OUT.glob('*bank.json'))
    paths += [TEACHER/'models'/('round1_s%d.json'%s) for s in range(3)]
    paths += [OLD/'models'/('teacher_s%d'%s)/'transitions.jsonl' for s in range(3)]
    write(OUT/'hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    from sac_teacher_study import verify as old_verify
    from teacher_common import verify as teacher_verify
    old_verify();teacher_verify()
    for name,digest in read(OUT/'hashes.json').items():assert sha(ROOT/name)==digest,name


def metric(trace):
    return {'steps':len(trace),'adjusted_cost':sum(r['cost'] for r in trace)+294.3,
        'raw_cost':sum(r['cost'] for r in trace),'physical_cost':sum(r['performance']+.4905 for r in trace),
        'H_cost':sum(r['compute'] for r in trace),'termination':trace[-1]['termination'],
        'solver_failures':sum(not r['solver_success'] for r in trace),'mean_H':float(np.mean([r['horizon'] for r in trace]))}
