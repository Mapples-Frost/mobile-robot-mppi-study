"""Archive the pinned upstream trees with their original tracked licenses."""
import hashlib
import json
import subprocess
from runtime import ART

dest=ART/'sources/archives';dest.mkdir(exist_ok=True)
rows=[]
for name in ['gym-horizon','stable-baselines-horizon','do-mpc-horizon']:
    folder=ART/'sources'/name
    sha=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
    path=dest/(name+'-'+sha[:12]+'.zip')
    subprocess.run(['git','-C',str(folder),'archive','--format=zip','--prefix='+name+'/',
                    '--output='+str(path),sha],check=True,timeout=60)
    rows.append({'repo':name,'commit':sha,'file':path.name,'bytes':path.stat().st_size,
                 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(dest/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
