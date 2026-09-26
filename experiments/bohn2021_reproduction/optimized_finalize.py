"""Finish the declared run with audits, replay, figures, and a completion record."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from runtime import ART, ROOT
from run import write

OUT = ART / 'results/optimized'
SCRIPTS = ROOT / 'experiments/bohn2021_reproduction'
LEGACY = Path('/home/mapples/.local/share/bohn2021-python37/bin/python')
MODERN = ROOT / '.venv/bin/python'


def main():
    started = time.time()
    while not (OUT / 'holdout_completed.json').exists():
        if time.time() - started > 24 * 3600:
            raise TimeoutError('Declared experiment did not finish within 24 hours.')
        time.sleep(30)
    assert json.loads((OUT / 'holdout_completed.json').read_text())['complete']
    subprocess.run([str(MODERN), str(SCRIPTS / 'optimized_report.py')], check=True)
    subprocess.run([str(MODERN), str(SCRIPTS / 'optimized_plot.py')], check=True)
    for task in ['pendulum', 'vehicle']:
        subprocess.run([str(LEGACY), '-u', str(SCRIPTS / 'optimized_replay.py'),
                        '--task', task, '--bank', 'holdout'], check=True)
    repos = {}
    for name in ['gym-horizon', 'do-mpc-horizon', 'stable-baselines-horizon']:
        repo = ART / 'sources' / name
        status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True)
        assert not status.strip(), (name, status)
        repos[name] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    files = [OUT / 'audit.json', OUT / 'summary.json', OUT / 'protocol.json',
             OUT / 'replay_pendulum_holdout.json', OUT / 'replay_vehicle_holdout.json',
             ART / 'report/optimized/report.md', ART / 'report/optimized/comparison.png']
    result = {'complete': True, 'training_runs': 30, 'training_transitions': 450000,
              'holdout_episode_evaluations': 1200, 'independent_holdout_scenes': 40,
              'author_repositories_clean': repos,
              'sha256': {str(p.relative_to(ART)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'reproduction_level': 'Hypothesis reconstruction with disclosed extensions; numerical conclusions are in summary.json.'}
    write(OUT / 'finalized.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(OUT / 'finalization_failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
