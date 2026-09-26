"""Record actual local runtimes and pinned author checkout state without simulations."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from runtime import ART,ROOT
from gated_horizon_search import OUT
from paper_h_soft_probe import digest
from run import write


def main(label):
    dest=OUT/'environment';dest.mkdir(exist_ok=True)
    path=dest/(label+'.json');assert not path.exists(),'Keep original environment snapshot; compare separately if environment changes'
    packages=subprocess.check_output([sys.executable,'-m','pip','list','--format=json','--disable-pip-version-check'],text=True)
    sources={}
    for name in ('gym-horizon','do-mpc-horizon','stable-baselines-horizon'):
        folder=ART/'sources'/name
        head=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(folder),'status','--porcelain','--untracked-files=no'],text=True)
        sources[name]=dict(path=str(folder),commit=head,tracked_worktree_status=dirty)
    executable=Path(sys.executable).resolve()
    value=dict(label=label,created=time.time(),executable=sys.executable,resolved_executable=str(executable),executable_sha256=digest(executable),
        python=sys.version,platform=platform.platform(),uname=list(platform.uname()),packages=json.loads(packages),
        numerical_thread_environment={k:v for k,v in os.environ.items() if k.endswith('NUM_THREADS') or k.startswith('TF_NUM_')},
        author_sources=sources,workspace=str(ROOT),cpuinfo=Path('/proc/cpuinfo').read_text(),os_release=Path('/etc/os-release').read_text(),
        source_hash=digest(Path(__file__)),scope='Current installed distributions and actual interpreter; not a claim that pip alone can reconstruct the legacy binary environment. Pinned source archives and existing runtime installation scripts remain required.')
    write(path,value)
    print(json.dumps(dict(label=label,python=platform.python_version(),packages=len(value['packages']),author_sources=sources),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--label',choices=['legacy','audit'],required=True);a=ap.parse_args();main(a.label)
