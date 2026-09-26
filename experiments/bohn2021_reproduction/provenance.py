"""Record pinned code, local implementation hashes, and search coverage."""
import hashlib
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from runtime import ROOT,ART

def main():
    repos={}
    for name in ['gym-letMPC','gym-horizon','stable-baselines','stable-baselines-horizon','do-mpc','do-mpc-horizon','rlmpcopt','author-site']:
        p=ART/'sources'/name
        def git(*args):return subprocess.check_output(['git','-C',str(p),*args],text=True).strip()
        repos[name]={'commit':git('rev-parse','HEAD'),'remote':git('remote','get-url','origin'),
                     'tracked_changes':git('status','--porcelain','--untracked-files=no')}
    files={}
    for folder in [ROOT/'experiments/bohn2021_reproduction',ART/'configs']:
        for p in folder.glob('*'):
            if p.is_file():files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    data={'accessed_utc':datetime.now(timezone.utc).isoformat(),'repos':repos,'files':files,
        'search_scope':{'author_public_repositories':18,'gym_forks_checked':5,'rlmpc_forks_checked':2,
            'histories':'Fetched branches and pre-2021-03-01 author commits inspected. Initial WSL history scans encountered network timeouts. Windows Git subsequently completed all-fetched-ref path scans for stable-baselines and do-mpc (exit 0); gym and rlmpcopt completed under WSL. See search/history_audit_windows.json and history_audit.json.',
            'missing':['lmpc-horizon experiment launcher','cart_pendulum_horizon.json','unicycle_ca_horizon.json','original 10-episode test files','original SAC checkpoints'],
            'limits':['GitHub code search requires authentication (401).','Two guessed lmpc-horizon repository URLs return 404, which does not distinguish private/deleted/nonexistent.',
                'Software Heritage endpoint returned anti-bot HTML, not archive metadata.','Google returned a JavaScript interstitial; Bing results were irrelevant and excluded.',
                'grep.app query for cart_pendulum_horizon returned HTTP 429.',
                'No Zenodo match for exact arXiv identifier; this is not proof of global absence.']}}
    (ART/'sources'/'provenance.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(repos,indent=2))

if __name__=='__main__':main()
