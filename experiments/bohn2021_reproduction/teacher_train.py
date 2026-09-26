"""Supervised relative-cost critics; final checkpoints only, all seeds kept."""
import argparse
import copy
import time
import numpy as np
import torch
from torch import nn
from teacher_common import OUT, HS, SCALE, SEEDS, read, write, infer, verify

torch.set_num_threads(1)


def data(split, arm):
    records = []
    for path in sorted((OUT/'labels'/split/arm).glob('scene_*/anchors.json')):
        assert (path.parent/'completed.json').exists()
        for a in read(path):
            records.append(dict(a, scene=path.parent.name, split=split))
    return records


def tensors(records):
    x = torch.tensor([r['context']['features'] for r in records], dtype=torch.float32)
    y = torch.tensor([r['relative_costs'] for r in records], dtype=torch.float32)
    return x, y


def metrics(pred, y):
    chosen = pred.argmin(1)
    regret = y[np.arange(len(y)), chosen] - y.min(1)
    return {'relative_cost_mae': float(np.mean(abs(pred-y))),
            'mean_regret': float(np.mean(regret)), 'max_regret': float(np.max(regret)),
            'near_optimal_fraction_002': float(np.mean(regret <= .02)),
            'chosen_H_counts': {str(h): int(sum(chosen == i)) for i, h in enumerate(HS)}}


def train(round_id):
    verify()
    records = data('train', 'switch_5_30')
    if round_id == 1:
        records += data('aggregation', 'round0_s0')
    validation = data('validation', 'switch_5_30')
    assert records and validation
    x, y = tensors(records)
    xv, yv = tensors(validation)
    known = torch.tensor(.003*(np.asarray(HS)-30)/SCALE, dtype=torch.float32)
    target = y-known
    dest = OUT/'models'
    dest.mkdir(exist_ok=True)
    for seed in SEEDS:
        arm = 'round%d_s%d'%(round_id, seed)
        if (dest/(arm+'_metrics.json')).exists():
            continue
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        net = nn.Sequential(nn.Linear(19, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, len(HS)))
        initial = copy.deepcopy(net.state_dict())
        optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
        history = []
        started = time.monotonic()
        for update in range(3000):
            ix = rng.randint(0, len(records), size=64)
            raw = net(x[ix])
            pred = raw - raw[:, HS.index(30):HS.index(30)+1]
            regression = nn.functional.smooth_l1_loss(pred, target[ix])
            costs = pred + known
            delta = y[ix, :, None] - y[ix, None, :]
            predicted_delta = costs[:, :, None] - costs[:, None, :]
            weights = torch.clamp(abs(delta), max=1.) * (abs(delta) > .005)
            ranking = (nn.functional.softplus(-torch.sign(delta)*predicted_delta/.1)*weights).sum()/weights.sum().clamp_min(1.)
            loss = regression+.1*ranking
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.)
            optimizer.step()
            if (update+1)%500 == 0:
                history.append({'update': update+1, 'loss': float(loss), 'regression': float(regression), 'ranking': float(ranking)})
        net.eval()
        model = {'arm': arm, 'seed': seed, 'round': round_id, 'horizons': HS,
                 'layers': [{'weight': layer.weight.detach().numpy().tolist(),
                             'bias': layer.bias.detach().numpy().tolist()} for layer in net if isinstance(layer, nn.Linear)]}
        write(dest/(arm+'.json'), model)
        torch.save({'state_dict': net.state_dict(), 'initial': initial, 'optimizer': optimizer.state_dict(),
                    'updates': 3000, 'seed': seed}, dest/(arm+'.pt'))
        with torch.no_grad():
            def forward(z):
                p = net(z)
                return (p-p[:, HS.index(30):HS.index(30)+1]+known).numpy()
            train_pred, val_pred = forward(x), forward(xv)
        reloaded = read(dest/(arm+'.json'))
        exported = np.asarray([infer(reloaded, value) for value in xv.numpy()])
        np.testing.assert_allclose(exported, val_pred, atol=2e-5, rtol=2e-5)
        assert np.array_equal(exported.argmin(1), val_pred.argmin(1))
        report = {'arm': arm, 'training_anchors': len(records), 'validation_anchors': len(validation),
            'updates': 3000, 'train': metrics(train_pred, y.numpy()),
            'validation': metrics(val_pred, yv.numpy()), 'history': history,
            'export_reload_verified': True, 'elapsed_s': time.monotonic()-started}
        write(dest/(arm+'_metrics.json'), report)
        print(str(report), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, required=True)
    train(ap.parse_args().round)
