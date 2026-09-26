"""Refresh reports when results change; verify and finalize after both queues exit."""
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17'
full=ART/'results/full';previous=None
while True:
    finished=sorted(str(p) for p in full.glob('*/completed.json'))
    curves=sorted(str(p) for p in full.glob('*/learning_curve/*/completed.json'))
    state=(tuple(finished),tuple(curves))
    if state!=previous:
        subprocess.run([sys.executable,str(ROOT/'experiments/bohn2021_reproduction/report.py')],check=True,cwd=str(ROOT))
        previous=state
        print(json.dumps({'completed_runs':len(finished),'intermediate_checkpoints':len(curves)}),flush=True)
    if (full/'suite_completed.json').exists() and (full/'curve_suite_completed.json').exists():
        subprocess.run([sys.executable,str(ROOT/'experiments/bohn2021_reproduction/verify_results.py'),'--require-complete'],check=True,cwd=str(ROOT))
        subprocess.run([sys.executable,str(ROOT/'experiments/bohn2021_reproduction/audit_constraints.py')],check=True,cwd=str(ROOT))
        subprocess.run([sys.executable,str(ROOT/'experiments/bohn2021_reproduction/provenance.py')],check=True,cwd=str(ROOT))
        break
    time.sleep(30)
print('Final reports and result audit complete.',flush=True)
