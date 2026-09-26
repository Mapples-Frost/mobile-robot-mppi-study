"""Post-hoc intervention on every timeout of the declared vehicle condition."""
import copy
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from runtime import ART, imports
from run import write, weights_hash
from optimized_runtime import install_terminal, make_env
from mechanism_probe import checked_step

SOURCE = ART / 'results/vehicle_entropy'
OUT = ART / 'results/vehicle_stall_probe'
NAME = 'vehicle_alpha0p1_s1'


def read(path):
    return json.loads(path.read_text())


def worker(case_id):
    install_terminal('vehicle')
    _, SAC, _ = imports()
    model = SAC.load(str(SOURCE / NAME / 'model.zip'))
    before = weights_hash(model)
    assert before == read(SOURCE / NAME / 'completed.json')['final_hash']
    case = read(SOURCE / 'holdout_bank.json')['cases'][case_id]
    original = read(SOURCE / 'evaluations' / NAME / 'value' / ('trace_%02d.json' % case_id))
    assert len(original) == 150 and original[-1]['termination'] == 'steps'
    assert all(r['horizon'] == 1 for r in original[-20:])
    env = make_env('vehicle', 26091850)
    env.set_value_function_weights_and_biases(*model.policy_tf.get_mpc_vfn_weights_and_biases())
    dest = OUT / ('case%02d' % case_id)
    dest.mkdir(exist_ok=True)
    results = {}
    for mode in ['original', 'force10']:
        obs = env.reset(**copy.deepcopy(case))
        for i in range(130):
            obs, done, row = checked_step(env, 'vehicle', original[i]['horizon'])
            assert not done and row['state'] == original[i]['state']
        start = copy.deepcopy(env.control_system.current_state)
        q1, q2 = model.sess.run([model.step_ops[4], model.step_ops[5]], {
            model.observations_ph: np.repeat(obs[None, :], 50, axis=0),
            model.actions_ph: np.linspace(-1, 1, 50, dtype=np.float32)[:, None]})
        assert int(np.rint(model.predict(obs, deterministic=True)[0][0])) == 1
        rows = []
        while True:
            h = 10 if mode == 'force10' else int(np.clip(np.rint(model.predict(obs, deterministic=True)[0][0]), 1, 50))
            obs, done, row = checked_step(env, 'vehicle', h)
            row.update(env.optimized_solver_info)
            rows.append(row)
            if done:
                break
        if mode == 'original':
            keys = ['state', 'input', 'horizon', 'cost']
            assert [{k: r[k] for k in keys} for r in rows] == [{k: r[k] for k in keys} for r in original[130:]]
        result = {'start_state': start, 'cost': sum(r['cost'] for r in rows), 'steps': len(rows),
                  'termination': rows[-1]['termination'], 'solver_failures': sum(not r['solver_success'] for r in rows),
                  'mean_speed': float(np.mean([r['input']['u_s'] for r in rows])),
                  'q1_best_h': int(np.argmax(q1)) + 1, 'min_q_best_h': int(np.argmax(np.minimum(q1, q2))) + 1,
                  'q1': q1.ravel().tolist(), 'q2': q2.ravel().tolist(), 'trace': rows}
        write(dest / (mode + '.json'), result)
        results[mode] = {k: v for k, v in result.items() if k not in ['trace', 'q1', 'q2']}
    assert results['original']['start_state'] == results['force10']['start_state']
    assert weights_hash(model) == before
    write(dest / 'summary.json', {'case': case_id, 'frozen': True, 'original_suffix_exact': True,
                                 'weights_sha256': before, 'results': results})
    print(json.dumps({'case': case_id, 'results': results}), flush=True)
    model.sess.close()


def main():
    if len(sys.argv) > 1:
        worker(int(sys.argv[1]))
        return
    OUT.mkdir(exist_ok=True)
    episodes = read(SOURCE / 'evaluations' / NAME / 'value/summary.json')['episodes']
    cases = [r['episode'] for r in episodes if r['termination'] == 'steps']
    assert len(cases) == 6
    write(OUT / 'protocol.json', {'condition': NAME, 'cases': cases, 'anchor': 130,
                                  'intervention': 'Keep learned terminal, replay first130steps exactly; force H10 for remaining at most20steps.',
                                  'scope': 'All six observed timeouts, selected post-hoc; mechanism diagnosis, not fresh holdout evaluation or deployment claim.'})
    def launch(case):
        with open(OUT / ('case%d.log' % case), 'a') as log:
            result = subprocess.run([sys.executable, '-u', __file__, str(case)], stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        assert result.returncode == 0, case
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(launch, cases))
    write(OUT / 'summary.json', {'complete': True, 'results': [read(OUT / ('case%02d' % i) / 'summary.json') for i in cases]})


if __name__ == '__main__':
    main()
