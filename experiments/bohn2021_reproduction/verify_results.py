"""Check saved scientific outputs independently of the training process."""
import argparse
import hashlib
import json
import math
from runtime import ART
from run import write


def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--require-complete',action='store_true');args=ap.parse_args()
    full=ART/'results/full';checks=[];errors=[]
    for p in sorted(full.glob('*/completed.json')):
        try:
            folder=p.parent;c=read(p);m=read(folder/'manifest.json');task=m['task']
            assert c['steps']==15000 and c['updates']==14901
            assert c['weights_changed'] and c['evaluation_frozen'] and c['test_episodes']==10
            assert read(folder/'first_update.json')['weights_changed']
            for key,suffix in [('config_sha256','.json'),('test_bank_sha256','_test_bank.json')]:
                assert m[key]==hashlib.sha256((ART/'configs'/(task+suffix)).read_bytes()).hexdigest()
            evaluations=[folder/'eval_value',folder/'eval_no_value']
            if m['fixed_horizon'] is None:
                curve_dirs=[q.parent for q in (folder/'learning_curve').glob('*/completed.json')]
                if args.require_complete:assert len(curve_dirs)==5, 'Missing checkpoint evaluations'
                for q in curve_dirs:assert read(q/'completed.json')['weights_frozen']
                evaluations+=curve_dirs
            for ev in evaluations:
                s=read(ev/'summary.json');eps=s['episodes'];assert len(eps)==10
                for ep in eps:
                    t=read(ev/('trace_%02d.json'%ep['episode']))
                    assert len(t)==ep['steps'] and 0<len(t)<=(100 if task=='pendulum' else 150)
                    assert math.isclose(sum(x['performance']+x['compute']+x['constraint'] for x in t),ep['total_cost'],rel_tol=1e-10,abs_tol=1e-8)
                    assert ep['solver_failure_steps']==sum(not x['solver_success'] for x in t)
                    assert ep['termination'] in ['steps','constraint','goal']
                    for x in t:
                        assert x['horizon'] in range(1,51)
                        if m['fixed_horizon'] is not None:assert x['horizon']==m['fixed_horizon']
                        assert math.isclose(x['compute'],(.003 if task=='pendulum' else .001)*x['horizon'],abs_tol=1e-12)
                        assert math.isclose(-x['reward'],x['performance']+x['compute']+x['constraint'],rel_tol=1e-10,abs_tol=1e-8)
                assert math.isclose(s['mean_total_cost'],sum(e['total_cost'] for e in eps)/10,rel_tol=1e-10,abs_tol=1e-8)
                assert s['constraint_episodes']==sum(e['termination']=='constraint' for e in eps)
                assert s['goal_episodes']==sum(e['termination']=='goal' for e in eps)
            checks.append(folder.name)
        except Exception as e:errors.append({'run':p.parent.name,'error':repr(e)})
    if args.require_complete and len(checks)!=26:errors.append({'suite':'Expected 26 complete, verified runs; found %d'%len(checks)})
    result={'status':'passed' if not errors else 'failed','verified_runs':checks,'errors':errors,
            'checks':['15000 transitions and 14901 updates','weights changed and frozen evaluation',
                      'configuration and common test-bank hashes','20 final test traces per run and completed intermediate checkpoint traces',
                      'independent cost recomputation','executed horizon and proxy cost','termination and solver-failure counts']}
    write(ART/'results/results_audit.json',result)
    print(json.dumps({'status':result['status'],'verified_runs':len(checks),'errors':errors}))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
