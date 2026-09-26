"""Give any newly nominated independent H the registered terminal-training opportunity."""
import argparse
import fcntl
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from gated_horizon_search import OUT,TASKS,freeze
from gated_horizon_policy import BASE
from gated_horizon_evaluate import grid_model,arm_name
from gated_horizon_audit import audit_candidate
from paper_h_soft_probe import read,digest
from runtime import ROOT,ART
from run import write
from gated_horizon_amendment import registration as amended_registration,audit_starts

SCRIPTS=ROOT/'experiments/bohn2021_reproduction'
LEGACY='/home/mapples/.local/share/bohn2021-python37/bin/python'
MODERN=str(ROOT/'.venv/bin/python')
SETTINGS=dict(aligned=True,scaled_obs=True,ent_coef='1.0',no_online_value=False,batch_size=256,buffer_size=1000000)


def register():
    freeze()
    old=SCRIPTS/'conservative_fixed_train.py';new=SCRIPTS/'gated_horizon_fixed_train.py'
    expected=old.read_text().replace('from conservative_iteration import OUT,verify','from gated_horizon_search import OUT,freeze as verify').replace("(task+'_validation.json')","(task+'_validation_bank.json')")
    assert new.read_text()==expected,'Only experiment root/validation filename may differ from audited fixed instrumentation'
    paths=[Path(__file__).resolve(),old,new,SCRIPTS/'conservative_fixed_log_audit.py',SCRIPTS/'run.py',OUT/'protocol.json',OUT/'evaluation_registration.json']
    for task in TASKS:
        p=ART/'results/conservative_iteration_2026-09-24/extra_fixed_smoke'/('%s_h%d_s0'%(task,BASE[task]))/'instrumentation_completed.json'
        d=read(p);assert d['passed'] and d['smoke']
        for q,h in d['hashes'].items():assert digest(Path(q))==h
        paths.append(p)
    value=dict(hashes={str(p):digest(p) for p in paths},settings=SETTINGS,training_steps=15000,seeds=[1,2],workers=2,
        criterion='Only new fixed H nominated by full independent seed0 grid; all pending seeds trained to final checkpoint before efficacy comparison. No repeated already complete training.',
        instrumentation='Reuse previously smoke-audited logging source with only root and validation filename substitutions, checked bytewise. Independent dynamics/cost audit of every new train and final validation transition.',
        extra_budget='Training wrapper emits32 validation episodes with terminal and32 without; no checkpoint selection from these. Registered recovery validation adds another32 episodes per new fixed model.',test_access=False)
    p=OUT/'baseline_completion_registration.json'
    if p.exists():amended_registration(p,value)
    else:assert not (OUT/'evaluations').exists();write(p,value)


def command(j):
    task,seed,h=j['task'],j['seed'],j['h'];assert task in TASKS and seed in (1,2) and h in range(5,51,5) and h!=BASE[task] and j['training_steps']==15000
    return [LEGACY,'-u',str(SCRIPTS/'gated_horizon_fixed_train.py'),'--task',task,'--seed',str(seed),'--fixed-horizon',str(h),'--steps','15000','--out',str(grid_model(task,h,seed)),
        '--test-bank',str(OUT/'banks'/(task+'_validation_bank.json')),'--eval-episodes','32','--aligned','--scaled-obs','--ent-coef','1.0','--batch-size','256','--buffer-size','1000000']


def run(wait):
    register();lock=(OUT/'baseline_completion.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/'baseline_completion_status.json';assert not path.exists(),'Inspect existing completion controller before restart'
    state=dict(pid=os.getpid(),active=True,stage='waiting_for_posttrain',started=time.time(),completed=[]);write(path,state)
    def execute(label,cmd):
        log=OUT/('%s_%d.log'%(label,time.time_ns()))
        with log.open('w') as stream:r=subprocess.run(cmd,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=86400)
        assert r.returncode==0,(label,str(log),r.returncode)
        return dict(stage=label,log=str(log),exit_code=0)
    try:
        deadline=time.monotonic()+172800
        while True:
            s=read(OUT/'posttrain_status.json')
            if not s['active']:assert s.get('complete'),s;break
            assert wait,'Posttrain still running'
            p=Path('/proc')/str(s['pid'])/'cmdline';assert p.exists() and b'gated_horizon_posttrain.py' in p.read_bytes()
            assert time.monotonic()<deadline;time.sleep(15)
        selected=read(OUT/'baseline_selection.json')
        for p,h in selected['hashes'].items():assert digest(Path(p))==h
        pending=selected['pending_independent_baselines'];state.update(stage='independent_training',pending=pending,selection_hash=digest(OUT/'baseline_selection.json'));write(path,state)
        def train(j):
            result=execute('fixed_%s_h%d_s%d'%(j['task'],j['h'],j['seed']),command(j))
            folder=grid_model(j['task'],j['h'],j['seed']);m=read(folder/'manifest.json');d=read(folder/'completed.json')
            assert m['adaptations']==SETTINGS and m['fixed_horizon']==j['h'] and m['seed']==j['seed']
            assert d['status']=='complete' and d['steps']==15000 and d['weights_changed'] and d['evaluation_frozen']
            audit=execute('audit_fixed_%s_h%d_s%d'%(j['task'],j['h'],j['seed']),[MODERN,str(SCRIPTS/'conservative_fixed_log_audit.py'),'--folder',str(folder)])
            return dict(job=j,training=result,audit=audit)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for f in as_completed([pool.submit(train,j) for j in pending]):state['completed'].append(f.result());write(path,state)
        state.update(stage='extra_validation');write(path,state)
        for j in pending:
            result=execute('eval_fixed_%s_h%d_s%d'%(j['task'],j['h'],j['seed']),[LEGACY,'-u',str(SCRIPTS/'gated_horizon_evaluate.py'),'--mode','job','--task',j['task'],'--family','grid','--seed',str(j['seed']),'--h',str(j['h'])])
            state['completed'].append(result);write(path,state)
        # Preserve the initial complete grid audit and extend it with each new arm.
        original=OUT/'audit_validation_initial.json'
        if not original.exists():write(original,read(OUT/'audit_validation.json'))
        audit=read(original);assert audit['passed']
        for p,h in audit['hashes'].items():assert digest(Path(p))==h
        for j in pending:
            folder=OUT/'evaluations/validation'/j['task']/arm_name('grid',j['seed'],j['h'])
            bank=read(OUT/'banks'/(j['task']+'_validation_bank.json'))['cases']
            a=audit_candidate(folder,j['task'],bank,fixed_h=j['h']);a.update(task=j['task'],family='grid',seed=j['seed'],h=j['h'])
            audit['groups'].append(a);audit['hashes'][str(folder/'completed.json')]=digest(folder/'completed.json')
        audit.update(conditions=len(audit['groups']),episodes=sum(g['episodes'] for g in audit['groups']),steps=sum(g['steps'] for g in audit['groups']),
            maximum_dynamics_error=max(g['maximum_dynamics_error'] for g in audit['groups']),initial_audit_hash=digest(original),supplement_source_hash=digest(Path(__file__)))
        audit['shared_initialization']=audit_starts([(g['task'],g['family'],g['seed'],g['h']) for g in audit['groups']],'validation')
        write(OUT/'audit_validation.json',audit)
        state['completed'].append(execute('refresh_baseline_selection',[MODERN,str(SCRIPTS/'gated_horizon_select.py')]))
        assert not read(OUT/'baseline_selection.json')['pending_independent_baselines']
        state.update(active=False,complete=True,ended=time.time(),test_opened=False);write(path,state)
    except BaseException as exc:
        state.update(active=False,complete=False,exception=repr(exc),ended=time.time());write(path,state);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');ap.add_argument('--wait',action='store_true');a=ap.parse_args()
    if a.register:register()
    else:run(a.wait)
