"""Post-fit descriptive ranking/noise diagnosis. No simulation or policy tuning."""
import csv
from pathlib import Path
import numpy as np
from branch_calibration_protocol import OUT, TASKS, BASE_H, model_dir, verify
from branch_calibration_audit import q_values, verify_artifact_hashes
from paper_h_soft_probe import read, digest
from runtime import imports
from run import weights_hash, write


def main():
    verify()
    verify_artifact_hashes(OUT / 'training_audit.json')
    complete = OUT / 'training_ranking_diagnosis_completed.json'
    if complete.exists():
        verify_artifact_hashes(complete)
        print('Existing training ranking diagnosis verified; no repeated network queries.', flush=True)
        return
    _, SAC, _ = imports()
    rows, summary, hashes = [], [], {}
    for task in TASKS:
        for seed in range(3):
            folder = OUT / ('%s_s%d' % (task, seed))
            models = [SAC.load(str(model_dir(task, 'min_q', seed) / 'model.zip')),
                      SAC.load(str(folder / 'model.zip'))]
            before = [weights_hash(m) for m in models]
            data = read(folder / 'dataset.json')
            model_rows = []
            for g in data['groups']:
                branch_path = folder / ('case%02d_t%03d.json' % (g['case'], g['anchor']))
                branch_data = read(branch_path)
                branches = branch_data['branches']
                hs = g['horizons']
                labels = np.asarray(g['targets'])
                repeats = np.asarray([[b['soft_return_with_target_tail'] for b in branches[str(h)]] for h in hs])
                assert repeats.shape == (len(hs), 2)
                np.testing.assert_allclose(labels, repeats.mean(axis=1))
                picks, errors = [], []
                for model in models:
                    q1, q2 = [q.ravel() for q in q_values(model, g['observation'], hs)]
                    q = np.minimum(q1, q2)
                    picks.append(int(np.argmax(q)))
                    errors.append(float(np.sqrt(np.mean(((q-q.mean())-(labels-labels.mean()))**2))))
                winner = int(np.argmax(labels))
                a, b = picks
                actor_index = hs.index(branch_data['actor_h'])
                fixed_index = hs.index(BASE_H[task])
                advantage = repeats[b] - repeats[a]
                row = {'task': task, 'seed': seed, 'case': g['case'], 'anchor': g['anchor'],
                    'label_best_h': hs[winner], 'raw_h': hs[a], 'calibrated_h': hs[b],
                    'raw_label_regret': float(labels[winner]-labels[a]),
                    'calibrated_label_regret': float(labels[winner]-labels[b]),
                    'actor_first_action_label_regret': float(labels[winner]-labels[actor_index]),
                    'fixed_first_action_label_regret': float(labels[winner]-labels[fixed_index]),
                    'raw_centered_rmse': errors[0], 'calibrated_centered_rmse': errors[1],
                    'repeat0_best_h': hs[int(np.argmax(repeats[:, 0]))],
                    'repeat1_best_h': hs[int(np.argmax(repeats[:, 1]))],
                    'paired_label_advantage_repeat0': float(advantage[0]),
                    'paired_label_advantage_repeat1': float(advantage[1]),
                    'paired_advantage_changes_sign': bool(advantage[0] * advantage[1] < 0),
                    'mean_abs_between_repeat_centered_label_difference': float(np.mean(abs(
                        (repeats[:,0]-repeats[:,0].mean()) - (repeats[:,1]-repeats[:,1].mean())))),
                    'median_abs_between_repeat_centered_label_difference': float(np.median(abs(
                        (repeats[:,0]-repeats[:,0].mean()) - (repeats[:,1]-repeats[:,1].mean())))),
                    'return_range': float(np.ptp(labels))}
                model_rows.append(row); rows.append(row)
                hashes[str(branch_path)] = digest(branch_path)
            summary.append({'task': task, 'seed': seed, 'groups': len(model_rows),
                'raw_mean_label_regret': float(np.mean([r['raw_label_regret'] for r in model_rows])),
                'calibrated_mean_label_regret': float(np.mean([r['calibrated_label_regret'] for r in model_rows])),
                'actor_mean_first_action_label_regret': float(np.mean([r['actor_first_action_label_regret'] for r in model_rows])),
                'fixed_mean_first_action_label_regret': float(np.mean([r['fixed_first_action_label_regret'] for r in model_rows])),
                'mean_abs_centered_between_repeat_difference': float(np.mean([r['mean_abs_between_repeat_centered_label_difference'] for r in model_rows])),
                'raw_picks': {str(h): sum(r['raw_h']==h for r in model_rows) for h in sorted(set(r['raw_h'] for r in model_rows))},
                'calibrated_picks': {str(h): sum(r['calibrated_h']==h for r in model_rows) for h in sorted(set(r['calibrated_h'] for r in model_rows))},
                'replicate_best_h_disagreements': sum(r['repeat0_best_h'] != r['repeat1_best_h'] for r in model_rows),
                'paired_advantage_sign_disagreements': sum(r['paired_advantage_changes_sign'] for r in model_rows)})
            assert before == [weights_hash(m) for m in models]
            for model in models: model.sess.close()
    write(OUT / 'training_ranking_diagnosis.json', {'scope': 'Post-fit training-only descriptive diagnostic; no generalization evidence or retuning.',
        'limits': 'Two noisy branch draws under original actor continuation; label regret is not true optimal-policy regret. Fixed/actor label contrasts intervene on the first action only, not complete fixed/actor controllers. Centered between-repeat differences describe noise with two samples, not standard errors or confidence bounds. Correlated states are not independent training seeds.',
        'summary': summary, 'rows': rows, 'hashes': hashes})
    with (OUT / 'training_ranking_diagnosis.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    paths = [Path(__file__), OUT/'training_audit.json', OUT/'training_ranking_diagnosis.json', OUT/'training_ranking_diagnosis.csv']
    write(complete, {'passed':True, 'hashes':{str(p):digest(p) for p in paths}})
    print(summary, flush=True)


if __name__ == '__main__': main()
