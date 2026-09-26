"""Registered numerical-audit-only amendment; preserve original frozen sources.

Use train --task ... --seed ... for an interrupted job after the old suite exits.
Use audit --phase training to recheck all six runs with the corrected auditor.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from branch_calibration_protocol import OUT, verify
from paper_h_soft_probe import read, digest
from run import write
from branch_calibration_numeric import audit_branch
import branch_calibration_run as runner
import branch_calibration_audit as auditor


def register():
    verify()
    paths = [Path(__file__), Path(__file__).with_name('branch_calibration_numeric.py'),
             OUT/'numeric_failure_diagnosis.json', OUT/'inputs_sha256.json']
    data = {'scope': 'Numerical audit correction only; no change to sampler, data, returns, fit, policy, seeds, selection or acceptance criteria.',
        'reason': 'Vehicle seed0 halted on float32 log-Jacobian reassociation. Two problematic saved outputs exactly replay under the original frozen network.',
        'change': 'Independent NumPy denominator is float32(1+EPS)-float32(a*a), matching TF1 constant folding. Existing 4e-5 log-density tolerance unchanged.',
        'recovery': 'Reuse every existing source/branch file; record failure and attempted-call overhead. Only missing branches and unexecuted fit run. Reaudit all models, including those completed with original auditor.',
        'original_sources_unchanged': True, 'hashes': {str(p): digest(p) for p in paths}}
    path = OUT/'numeric_amendment.json'
    if path.exists(): assert {k:v for k,v in read(path).items() if k != 'registered_unix'} == data
    else: write(path, dict(data, registered_unix=time.time()))
    runner.audit_branch = audit_branch
    auditor.audit_branch = audit_branch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('register','train','audit'))
    ap.add_argument('--task', choices=('vehicle','pendulum'))
    ap.add_argument('--seed', type=int, choices=range(3))
    ap.add_argument('--phase', choices=('smoke','training','validation','test'))
    args = ap.parse_args()
    register()
    if args.mode == 'train':
        assert args.task is not None and args.seed is not None
        runner.train_job(args.task, args.seed)
        folder = OUT/('%s_s%d' % (args.task,args.seed))
        write(folder/'numeric_amendment_used.json', {'amendment_hash': digest(OUT/'numeric_amendment.json'),
            'completed_hash': digest(folder/'completed.json')})
    elif args.mode == 'audit':
        assert args.phase is not None
        sys.argv = [auditor.__file__, args.phase]
        auditor.main()
        report = OUT/(args.phase + ('_audit.json' if args.phase in ('smoke','training') else '_gate.json'))
        write(OUT/('numeric_%s_audit_provenance.json' % args.phase), {
            'amendment_hash': digest(OUT/'numeric_amendment.json'), 'audit_hash': digest(report), 'original_inputs_verified': True})


if __name__ == '__main__':
    try: main()
    except Exception:
        write(OUT/('numeric_failure_%d.json' % time.time_ns()), {'argv':sys.argv,'traceback':traceback.format_exc()})
        raise
