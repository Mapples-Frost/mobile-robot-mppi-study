"""Overlap the already-declared per-task replications with other task training."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from optimized_suite import OUT, train
from run import write


def main():
    pending = {'pendulum', 'vehicle'}
    futures = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        while pending:
            if time.time() - started > 24 * 3600:
                raise TimeoutError('Fixed-H grid did not complete within 24 hours.')
            for task in sorted(pending):
                folders = [OUT / ('%s_fixed_h%d_s0' % (task, h)) for h in range(5, 51, 5)]
                if not all((f / 'completed.json').exists() for f in folders):
                    continue
                candidates = [(json.loads((f / 'eval_value/summary.json').read_text())['mean_total_cost'], h)
                              for f, h in zip(folders, range(5, 51, 5))]
                cost, horizon = min(candidates)
                write(OUT / (task + '_replication_selection.json'),
                      {'h': horizon, 'validation_cost': cost, 'all_candidates': candidates,
                       'scope': 'Identical declared selection, scheduled independently per task; no holdout read.'})
                print(json.dumps({'task': task, 'selected_fixed_h': horizon}), flush=True)
                futures.extend(pool.submit(train, (task, seed, horizon)) for seed in [1, 2])
                pending.remove(task)
            if pending:
                time.sleep(30)
        results = [f.result() for f in futures]
    assert all(r['exit_code'] == 0 for r in results)


if __name__ == '__main__':
    main()
