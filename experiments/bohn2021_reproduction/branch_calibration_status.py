"""Compact file/process status; never calls a simulator or opens a test bank."""
import json
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/branch_calibration_2026-09-24'


def read(p): return json.loads(p.read_text())


def main():
    rows = []
    for task in ('vehicle', 'pendulum'):
        for seed in range(3):
            d = OUT / ('%s_s%d' % (task, seed))
            attempts = [read(p) for p in d.glob('attempt_*.json')]
            row = {'task': task, 'seed': seed, 'fit_complete': (d/'completed.json').exists(),
                   'source_episodes': len(list(d.glob('source_*.json'))),
                   'anchor_groups': len(list(d.glob('case*_t*.json'))),
                   'attempted_steps': sum(a['step_calls'] for a in attempts),
                   'live_pids': [a['pid'] for a in attempts if Path('/proc/%d' % a['pid']).exists()]}
            if (d/'fit.json').exists():
                f = read(d/'fit.json'); row['loss'] = [f['initial_loss'], f['final_loss']]
            rows.append(row)
    state = {'unix': time.time(), 'training': rows,
             'training_audit': (OUT/'training_audit.json').exists(),
             'validation_completed_arms': len(list((OUT/'evaluations/validation').glob('*/*/completed.json'))),
             'validation_gate': read(OUT/'validation_gate.json')['passed'] if (OUT/'validation_gate.json').exists() else None,
             'test_evaluations_exist': (OUT/'evaluations/test').exists(),
             'serial_timing_completed_conditions': len(list((OUT/'serial_timing').glob('r*/completed.json'))),
             'failures': [p.name for p in OUT.glob('failure_*.json')]}
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
