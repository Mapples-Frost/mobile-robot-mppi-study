"""Metadata-only historical filename search, recording failures explicitly."""
import json
import subprocess
from runtime import ART

rows=[]
for name in ['gym-letMPC','stable-baselines','do-mpc','rlmpcopt']:
    command=['git','-C',str(ART/'sources'/name),'log','--all','--no-renames','--name-only','--format=']
    try:
        r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40,encoding='utf-8')
        paths=sorted(set(r.stdout.splitlines())-{''})
        (ART/'search'/(name+'_history_paths_verified.txt')).write_text('\n'.join(paths)+'\n')
        rows.append({'repo':name,'command':command,'exit_code':r.returncode,'complete':r.returncode==0,
                     'paths':len(paths),'stderr':r.stderr,
                     'candidate_paths':[x for x in paths if any(s in x.lower() for s in ['horizon','unicycle','cart_pendulum','test_set','checkpoint'])]})
    except subprocess.TimeoutExpired:
        rows.append({'repo':name,'command':command,'complete':False,'error':'40 second timeout; no completeness claim'})
(ART/'search/history_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
