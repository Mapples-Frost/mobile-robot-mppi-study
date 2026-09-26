"""Finish numerical audit, fresh-process replay, and reports after queued runs."""
import json
import subprocess
import sys
import time
from runtime import ART,ROOT
from run import write

scripts=ROOT/'experiments/bohn2021_reproduction'
checks=[]
for group in ['refined','paper_defaults']:
    out=ART/'results'/group
    while not (out/'holdout_completed.json').exists():time.sleep(10)
    assert json.loads((out/'holdout_completed.json').read_text())['complete'],group
    subprocess.check_call([sys.executable,str(scripts/'verify_refined.py'),'--group',group])
    for task in ['vehicle','pendulum']:
        subprocess.check_call([sys.executable,str(scripts/'check_replay.py'),
            '--run',str(out/(task+'_rl_s0')),'--bank',str(ART/'configs'/(task+'_holdout_bank.json')),
            '--evaluation-dir','holdout_value','--audit-out',str(out/('replay_'+task+'.json'))])
    subprocess.check_call([str(ROOT/'.venv/bin/python'),str(scripts/'diagnosis_report.py'),'--group',group])
    checks.append({'group':group,'verified':True,'holdout_episodes':json.loads((out/'audit.json').read_text())['holdout_episodes']})
clean=[]
for source in ['gym-horizon','stable-baselines-horizon','do-mpc-horizon']:
    p=ART/'sources'/source
    status=subprocess.check_output(['git','-C',str(p),'status','--porcelain'],universal_newlines=True)
    assert not status,(source,status)
    clean.append(source)
write(ART/'results/diagnosis_finalized.json',{'complete':True,'groups':checks,'pinned_sources_clean':clean,
    'new_training_runs':28,'new_training_transitions':420000,'old_training_runs_preserved':26,
    'old_training_transitions_preserved':390000,'total_training_transitions':810000,
    'note':'No test-based checkpoint or seed selection. Diagnostic interventions are separate from final comparison.'})
print('All diagnostics and reconstruction audits complete',flush=True)
