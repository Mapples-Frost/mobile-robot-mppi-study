"""Post-hoc first-divergence block vs repeated-policy mechanism experiment."""
import copy
import csv
import fcntl
import json
import os
import time
from pathlib import Path
from runtime import ART,imports
from conservative_recovery_diagnosis import OUT as RECOVERY,PREVIOUS,register as verify_recovery,TASKS,BASE,BLOCK
from conservative_solver_recovery import install
from conservative_canonical_reset import make_env
from conservative_policy_model import choose,predict
from conservative_iteration_evaluate import arm_name
from conservative_iteration_audit import audit_context
from branch_calibration_run import meter,observed_step
from branch_calibration_audit import audit_trace
from fixed_policy_branches import metrics
from min_q_eval_suite import model_dir
from relative_policy_features import context
from paper_h_soft_probe import read,digest
from run import write,weights_hash

OUT=ART/'results/first_block_probe_2026-09-25'


def without_recovery(trace):return [{k:v for k,v in row.items() if k!='recovery'} for row in trace]


def main():
    verify_recovery();assert read(RECOVERY/'audit_formal/audit.json')['passed']
    OUT.mkdir(exist_ok=True);lock=(OUT/'run.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    pairs=list(csv.DictReader((RECOVERY/'delivery/all_paired_episodes.csv').open()))
    selected=[]
    for task in TASKS:
        for seed in range(3):
            for family in ('round0','round1'):
                group=[r for r in pairs if (r['task'],int(r['seed']),r['family'])==(task,seed,family)]
                assert len(group)==24
                worst=min(group,key=lambda r:(-float(r['difference']),int(r['case'])))
                assert float(worst['difference'])>0
                selected.append(dict(task=task,seed=seed,family=family,case=int(worst['case']),full_policy_excess=float(worst['difference'])))
    sources=[Path(__file__).resolve(),Path(__file__).with_name('conservative_solver_recovery.py').resolve(),
        RECOVERY/'delivery/report.json',RECOVERY/'audit_formal/audit.json']
    protocol=dict(scope='Post-hoc conditional mechanism only, exposed scenarios. No new validation/test or policy fit.',
        selection='Largest recovered-policy minus recovered-fixed total cost among all24 cases for each task/seed/round; ties lower case index.',
        jobs=selected,intervention='Replay shared fixed prefix to first H disagreement; hold adaptive choice for one5-step block then fixed H for rest of episode. Identical bounded solver recovery for all arms.',
        controls='Full fixed replay must equal archived fixed trace, including every state/control/cost/termination; one-block prefix must equal fixed prefix. Full learned-policy outcome already saved.',
        measures=['total_cost','success','constraint','solver_failures','initial_failures','retries','extra_steps'],
        interpretation='One-block excess and full-policy excess reported separately. No claim that first intervention explains all downstream loss or all selected cases represent population.',
        budget='12 conditions,24 new full rollouts including prefix; no new model updates. Durable solver and environment attempts.',
        test_access=False,hashes={str(p):digest(p) for p in sources})
    p=OUT/'protocol.json'
    if p.exists():assert read(p)==protocol
    else:write(p,protocol)
    status=dict(pid=os.getpid(),active=True,completed=[],started=time.time());write(OUT/'status.json',status)
    results=[]
    try:
        _,SAC,_=imports()
        for item in selected:
            task,seed,family,cid=[item[k] for k in ('task','seed','family','case')]
            dest=OUT/('%s_s%d_%s'%(task,seed,family));dest.mkdir(exist_ok=True)
            if (dest/'completed.json').exists():
                d=read(dest/'completed.json')
                for name,h in d['hashes'].items():assert digest(Path(name))==h
                results.append(d['result']);status['completed'].append(item);write(OUT/'status.json',status);continue
            assert not (dest/'solver_attempts.json').exists(),'Inspect interrupted condition before repeating'
            base_path=RECOVERY/'formal'/task/arm_name('primary',seed,BASE[task])/('repeat0_trace_%02d.json'%cid)
            adaptive_path=RECOVERY/'formal'/task/arm_name(family,seed,0)/('repeat0_trace_%02d.json'%cid)
            baseline=without_recovery(read(base_path));adaptive=without_recovery(read(adaptive_path))
            anchor=next(i for i,(a,b) in enumerate(zip(adaptive,baseline)) if a['horizon']!=b['horizon'])
            assert anchor%BLOCK==0 and adaptive[:anchor]==baseline[:anchor]
            assert adaptive[anchor]['policy_context']==baseline[anchor]['policy_context']
            h_adaptive=adaptive[anchor]['horizon']
            policy_path=PREVIOUS/('%s_s%d_r%d'%(task,seed,int(family[-1])))/'policy.json';policy=read(policy_path)
            features=baseline[anchor]['policy_context']['features'];assert choose(policy,features)==h_adaptive
            predictions=[];layer=policy
            while layer:
                q,s,d=predict(layer,features)
                predictions.append(dict(round=layer['round'],chosen=choose(layer,features),q=q.tolist(),safety=s.tolist(),distance=d,support_radius=layer['support_radius']))
                layer=layer['prior']
            bank=PREVIOUS/'banks'/(task+'_validation.json');case=read(bank)['cases'][cid]
            env=make_env(task,seed,aligned=True,scaled_obs=True);meter(env,dest)
            source=model_dir(task,'fixed',seed);model=SAC.load(str(source/'model.zip'))
            assert weights_hash(model)==read(source/'completed.json')['final_hash']
            env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases());model.sess.close()
            state=install(env.control_system.controller.mpc,dest);branch_results={};hashes={str(p):digest(p) for p in (base_path,adaptive_path,policy_path,bank,source/'model.zip',OUT/'protocol.json')}
            for mode in ('fixed_replay','one_block'):
                state.update(enabled=False,events=[],case=cid,step=-1);env.reset(**copy.deepcopy(case));state['enabled']=True;trace=[]
                for t in range(env.max_steps):
                    state['step']=t;ctx=context(env,task)
                    h=h_adaptive if mode=='one_block' and anchor<=t<anchor+BLOCK else BASE[task]
                    _,done,row=observed_step(env,task,h,case,t);row['policy_context']=ctx
                    if mode=='fixed_replay' or t<anchor:assert row==baseline[t],(item,mode,t,'prefix replay')
                    row['recovery']=state['events'][-1];trace.append(row)
                    if done:break
                assert done
                if mode=='fixed_replay':assert without_recovery(trace)==baseline
                audit_trace(task,case,trace);audit_context(task,case,trace,0)
                path=dest/(mode+'.json');write(path,trace);hashes[str(path)]=digest(path)
                branch_results[mode]=dict(metrics(task,trace),initial_failed_steps=sum(not r['recovery']['attempts'][0]['success'] for r in trace),
                    retries=sum(len(r['recovery']['attempts'])-1 for r in trace))
            result=dict(item,anchor=anchor,first_adaptive_h=h_adaptive,predictions=predictions,branches=branch_results,
                one_block_excess=branch_results['one_block']['total_cost']-branch_results['fixed_replay']['total_cost'],
                full_policy_metrics=metrics(task,adaptive))
            for path in (dest/'solver_attempts.json',dest/'solver_calls.jsonl'):hashes[str(path)]=digest(path)
            write(dest/'completed.json',dict(passed=True,result=result,hashes=hashes));results.append(result)
            status['completed'].append(item);write(OUT/'status.json',status)
            print(json.dumps({k:v for k,v in result.items() if k not in ('predictions','branches','full_policy_metrics')}),flush=True)
        write(OUT/'results.json',dict(results=results,test_access=False,independent_efficacy=False))
        status.update(active=False,complete=True,ended=time.time());write(OUT/'status.json',status)
    except BaseException as exc:
        status.update(active=False,complete=False,exception=repr(exc),ended=time.time());write(OUT/'status.json',status);raise


if __name__=='__main__':main()
