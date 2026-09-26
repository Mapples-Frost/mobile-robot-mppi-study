"""Stop only this study's verified processes before a numerical reset amendment."""
import json
import os
import signal
import time
from pathlib import Path
from conservative_iteration import OUT,ROOT
from paper_h_soft_probe import read
from run import write


def main():
    dest=OUT/'reset_diagnosis/stop_snapshot.json';assert not dest.exists()
    permitted={'conservative_iteration_pipeline.py','conservative_iteration.py',
               'conservative_posttrain.py','conservative_complete_baselines.py'}
    found=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().decode().split('\0')
            cwd=(p.parent/'cwd').resolve()
        except (OSError,UnicodeError):continue
        if cwd!=ROOT or not args or 'python' not in Path(args[0]).name:continue
        if any(Path(a).name in permitted for a in args[1:]):
            found.append(dict(pid=int(p.parent.name),args=args))
    assert found
    statuses={p.name:read(p) for p in OUT.glob('*status.json')}
    write(dest,dict(time=time.time(),processes=found,statuses=statuses,
        reason='Verified cross-episode soft-slack initialization leak; preserve files and prevent further data collection before correction.'))
    # Stop scheduling first; then terminate only the inspected study processes.
    for p in found:os.kill(p['pid'],signal.SIGSTOP)
    for p in found:
        os.kill(p['pid'],signal.SIGTERM);os.kill(p['pid'],signal.SIGCONT)
    time.sleep(1)
    for name,state in statuses.items():
        if state.get('pid') in {p['pid'] for p in found}:
            state.update(active=False,complete=False,interrupted=True,
                         interruption_reason='reset_slack_audit',ended=time.time())
            write(OUT/name,state)
    print(json.dumps(dict(terminated=[p['pid'] for p in found],snapshot=str(dest))))


if __name__=='__main__':main()
