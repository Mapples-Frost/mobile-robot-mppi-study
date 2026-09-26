"""Preserve shutdown runs and audit original-seed restarts without editing training."""
import argparse
from datetime import datetime, timezone
import json
import numpy as np
from sac_teacher_study import OUT, read, write, verify, sha


def inventory(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def prepare():
    verify()
    record = OUT/'recovery.json'
    if record.exists():
        return read(record)
    paused = read(OUT/'paused_shutdown.json')
    archive = OUT/'interrupted_shutdown'
    archive.mkdir(exist_ok=True)
    records = {}
    for name in ['plain_s0', 'teacher_s1']:
        src = OUT/'models'/name
        dst = archive/name
        assert src.resolve().parent == (OUT/'models').resolve()
        assert dst.resolve().parent == archive.resolve()
        assert not (src/'completed.json').exists()
        assert src.exists() and not dst.exists()
        files = inventory(src)
        with (src/'transitions.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == paused['saved_trace_counts'][name]['saved_complete_transitions']
        resets = len({(r['phase'], r['episode']) for r in rows})
        records[name] = {'files': files, 'saved_transitions': len(rows),
                         'observed_reset_warmups': resets,
                         'last_phase': rows[-1]['phase'], 'last_step': rows[-1]['step']}
        src.rename(dst)
        assert inventory(dst) == files
    completed = {name: inventory(OUT/'models'/name) for name in ['plain_s2', 'teacher_s2']}
    result = {'timestamp': datetime.now(timezone.utc).isoformat(),
              'method': 'Restart unfinished models from original seed; reuse completed seed2 pair.',
              'exact_continuation_claimed': False,
              'budget_caveat': 'Interrupted counts are observed lower bounds; unflushed transitions at shutdown are unknown.',
              'interrupted': records, 'completed_reused': completed}
    write(record, result)
    return result


def audit():
    verify()
    recovery = read(OUT/'recovery.json')
    for name, files in recovery['completed_reused'].items():
        for relative, digest in files.items():
            assert sha(OUT/'models'/name/relative) == digest
    results = {}
    for name, rec in recovery['interrupted'].items():
        old = OUT/'interrupted_shutdown'/name
        new = OUT/'models'/name
        assert (new/'completed.json').exists(), name
        assert inventory(old) == rec['files']
        assert read(old/'manifest.json') == read(new/'manifest.json')
        checkpoints = {}
        for p in old.glob('*_metadata.json'):
            before, after = read(p), read(new/p.name)
            checkpoints[p.name] = {'same_weights': before['weights_hash'] == after['weights_hash'],
                                   'old_hash': before['weights_hash'], 'new_hash': after['weights_hash']}
        matching = 0
        compared = 0
        max_state_delta = 0.
        with (old/'transitions.jsonl').open() as a, (new/'transitions.jsonl').open() as b:
            for first, second in zip(a, b):
                compared += 1
                x, y = json.loads(first), json.loads(second)
                same = all(x[k] == y[k] for k in ['obs', 'next_obs', 'action', 'horizon', 'cost', 'done', 'phase', 'episode', 'step'])
                matching += int(same)
                max_state_delta = max(max_state_delta, float(np.max(np.abs(np.asarray(x['next_obs'])-y['next_obs']))))
        assert compared == rec['saved_transitions']
        results[name] = {'checkpoints': checkpoints, 'identical_transition_rows': matching,
                         'compared_rows': compared, 'max_next_obs_absolute_delta': max_state_delta}
    result = {'archives_unchanged': True, 'completed_seed2_unchanged': True,
              'restarted_from_same_initialization': True, 'comparisons': results}
    write(OUT/'recovery_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit() if args.audit else prepare(), indent=2))
