"""Evidence-based continuation snapshot; never opens held-out outcomes."""
import json
import time
from pathlib import Path
from conservative_iteration import OUT, TASKS, verify
from paper_h_soft_probe import read, digest
from run import write


def main():
    verify();processes={};conditions=[]
    for name in ('pipeline_status.json','posttrain_status.json','baseline_completion_status.json','validation_finish_status.json','round0_status.json','round1_status.json','validation_status.json','test_status.json'):
        path=OUT/name
        if not path.exists():continue
        d=read(path);pid=d.get('pid')
        proc=Path('/proc')/str(pid)/'cmdline'
        command=proc.read_bytes().decode().replace('\0',' ').strip() if proc.exists() else None
        processes[name]=dict(pid=pid,recorded_active=d.get('active'),
            live_experiment_process=bool(command and 'conservative_' in command),command=command,
            stage=d.get('current',d.get('stage')),complete=d.get('complete'),failed=d.get('failed'))
    for round_id in range(2):
        for task in TASKS:
            for seed in range(3):
                folder=OUT/('%s_s%d_r%d'%(task,seed,round_id))
                row=dict(task=task,seed=seed,round=round_id,collection_complete=False,fit_complete=False,audit_passed=False)
                cp=folder/'collection_completed.json';fp=folder/'fit_completed.json';ap=folder/'collection_audit.json'
                if cp.exists():
                    d=read(cp);row.update(collection_complete=True,groups=len(d['groups']),steps=d['explicit_step_calls'],resets=d['explicit_reset_calls'])
                if fp.exists():
                    d=read(fp);assert digest(folder/'policy.json')==d['policy_hash']
                    assert digest(cp)==d['dataset_hash']
                    row.update(fit_complete=True,training_updates=d['training_updates'],policy_hash=d['policy_hash'])
                if ap.exists():
                    d=read(ap)
                    for p,h in d['hashes'].items():assert digest(Path(p))==h
                    row['audit_passed']=d['passed']
                if folder.exists() and not row['collection_complete']:
                    row['persisted_sources']=len(list(folder.glob('source_*.json')))
                    row['persisted_branches']=len(list(folder.glob('case*_t*_h*.json')))
                    attempts=[read(p) for p in folder.glob('attempt_*.json')]
                    row['attempted_steps']=sum(a['step_calls'] for a in attempts)
                conditions.append(row)
    status=dict(recorded_unix=time.time(),goal_complete=False,processes=processes,conditions=conditions,
        evaluation_directory_exists=(OUT/'evaluations').exists(),
        test_outcome_directory_exists=(OUT/'evaluations/test').exists(),
        scope='Process identity plus hashes/completion/audit files; directory existence does not read test outcomes.',
        next_steps=['All12 formal datasets/models are complete and audited; never retrain completed models just to resume.',
          'Wait for posttrain independent12-dataset audit, banks,72-condition validation and baseline nomination.',
          'baseline_completion_status.json tracks the active waiting runner: nominated fixed H gets missing independent15k seeds1/2 and re-evaluation/selection/report automatically. Audit new fixed logs before final claims.',
          'After banks exist run conservative_split_audit.py --require-eval; after baseline completion run conservative_validation_inventory.py so added fixed seeds appear in complete tables.',
          'validation_finish_status.json tracks the active continuation: after prerequisites exit it audits extra fixed logs, inventories complete validation, reports both rounds, then runs serial timing. Do not launch duplicate timing. Avoid experiment Python tools while serial timing is active; inspect JSON directly instead.',
          'If no round is eligible, measure both rounds versus nominated fixed comparators with conservative_iteration_timing.py --split validation --all-candidates after all workers exit. Output is descriptive and separate; never alter the effect gate to admit failed policies.',
          'Unseal48-case test only after positive validation gate and confirmation freeze. Failed validation requires diagnosis, not test peeking.'])
    write(OUT/'continuation_status_2026-09-25.json',status)
    print(json.dumps(dict(processes=processes,completed_collections=sum(r['collection_complete'] for r in conditions),
        completed_fits=sum(r['fit_complete'] for r in conditions),audits_passed=sum(r['audit_passed'] for r in conditions),
        incomplete=[r for r in conditions if not r['fit_complete']]),indent=2))


if __name__=='__main__':main()
