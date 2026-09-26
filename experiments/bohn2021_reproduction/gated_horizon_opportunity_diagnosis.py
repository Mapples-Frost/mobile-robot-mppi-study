"""Post-hoc, non-deployable per-scene oracle across the already exposed grid.

No policy is trained or selected, no new simulation is run, and no test outcome
is read. The oracle uses future whole-episode outcomes and is only a diagnostic;
it is neither an upper bound for arbitrary adaptive policies nor validation of
a deployable selector. Full grids and all failed scenes remain visible.
"""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25'
DEST = RUN / 'opportunity_diagnosis'
REG = RUN / 'opportunity_diagnosis_registration.json'
BASE = {'vehicle': 25, 'pendulum': 30}
HS = tuple(range(5, 51, 5))
PREFIX = '/home/mapples/projects/mobile-robot-mppi-study/'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def resolve(name):
    p = Path(name[len(PREFIX):]) if name.startswith(PREFIX) else Path(name)
    return p if p.is_absolute() else ROOT / p


def relative(p):
    return p.resolve().relative_to(ROOT.resolve()).as_posix()


def main():
    status = read(RUN / 'validation_finish_status.json')
    assert status['complete'] and not status['active']
    assert not (RUN / 'evaluations/test').exists()
    assert not (RUN / 'confirmation_registration.json').exists()
    sources = [Path(__file__), RUN / 'baseline_selection_review/report.json',
               RUN / 'effect_gate_review/report.json', RUN / 'validation_diagnosis/report.json']
    inputs = {relative(p): sha(p) for p in sources}
    if REG.exists():
        registration = read(REG)
        assert registration['input_hashes'] == inputs
    else:
        registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(), input_hashes=inputs,
                            post_hoc=True, previously_seen='Full validation effects, all-H summary table, and cost/time diagnosis.',
                            questions=['What outcome-informed per-episode H opportunities exist in the full ten-H grids?',
                                       'Can any recorded fixed H rescue a reference-failed pendulum scene?',
                                       'Does the largest vehicle cost difference precede the first short-H intervention?'],
                            oracle='At each already exposed case choose lowest whole-episode total cost; ties smaller H. Also describe safety-only and safety plus per-case physical-cost NI2% filters. Reference included. This is not a deployable selector or general adaptive-policy bound.',
                            prohibitions='No new simulation, training, model selection, acceptance test, data deletion, or sealed-test access.')
        REG.write_text(json.dumps(registration, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    hashes = dict(inputs)
    hashes[relative(REG)] = sha(REG)
    for source in sources[1:]:
        report = read(source)
        assert report['passed']
        for name, value in report['hashes'].items():
            assert sha(resolve(name)) == value
            hashes[relative(resolve(name))] = value
    baseline = read(sources[1])
    summaries = {}

    def arm(task, family, seed, h):
        key = (task, family, seed, h)
        if key not in summaries:
            folder = RUN / 'evaluations/validation' / task / ('%s_h%d_s%d' % (family, h, seed))
            done = read(folder / 'completed.json')
            assert done['passed']
            hashes[relative(folder / 'completed.json')] = sha(folder / 'completed.json')
            for name, value in done['hashes'].items():
                assert sha(resolve(name)) == value
                hashes[relative(resolve(name))] = value
            summary = read(folder / 'summary.json')
            assert [r['case'] for r in summary['episodes']] == list(range(32))
            summaries[key] = summary['episodes']
        return summaries[key]

    full_rows, scene_rows, totals, rescues = [], [], [], []
    for task, base_h in BASE.items():
        nomination = baseline['nominations'][task]
        for family in ('matched', 'independent'):
            for seed in (range(3) if family == 'matched' else (0,)):
                selected_h = nomination['matched_h' if family == 'matched' else 'independent_h']
                grid = {h: arm(task, 'primary' if h == base_h else ('matched' if family == 'matched' else 'grid'), seed, h) for h in HS}
                reference = grid[selected_h]
                for h, episodes in grid.items():
                    for e in episodes:
                        full_rows.append(dict(task=task, terminal_family=family, seed=seed, h=h,
                                              case=e['case'], total_cost=e['total_cost'], physical_constraint_cost=e['physical_constraint_cost'],
                                              success=e['success'], constraint=e['constraint'], steps=e['steps'],
                                              initial_failures=e['initial_failed_steps'], final_failures=e['solver_failure_steps']))
                for cid, b in enumerate(reference):
                    improved_success = [h for h, episodes in grid.items() if episodes[cid]['success'] and not b['success']]
                    rescues.append(dict(task=task, terminal_family=family, seed=seed, case=cid,
                                        selected_h=selected_h, reference_success=b['success'],
                                        success_hs=';'.join(map(str, improved_success)), rescued=bool(improved_success)))
                for mode in ('unconstrained', 'safety_only', 'safety_and_physical_ni2'):
                    picked = []
                    for cid, b in enumerate(reference):
                        candidates = []
                        for h, episodes in grid.items():
                            a = episodes[cid]
                            safe = a['success'] >= b['success'] and a['constraint'] <= b['constraint'] and all(
                                a[k] / a['steps'] <= b[k] / b['steps'] for k in ('initial_failed_steps', 'solver_failure_steps'))
                            physical = a['physical_constraint_cost'] <= b['physical_constraint_cost'] + .02 * abs(b['physical_constraint_cost'])
                            if mode == 'unconstrained' or (safe and (mode == 'safety_only' or physical)):
                                candidates.append((a['total_cost'], h, a))
                        _, h, a = min(candidates, key=lambda r: (r[0], r[1]))
                        picked.append((h, a))
                        scene_rows.append(dict(task=task, terminal_family=family, seed=seed, mode=mode,
                                               case=cid, reference_h=selected_h, oracle_h=h, reference_success=b['success'], oracle_success=a['success'],
                                               reference_cost=b['total_cost'], oracle_cost=a['total_cost'],
                                               cost_delta=a['total_cost'] - b['total_cost'], eligible_h_count=len(candidates)))
                    bc = sum(r['total_cost'] for r in reference)
                    ac = sum(r['total_cost'] for _, r in picked)
                    totals.append(dict(task=task, terminal_family=family, seed=seed, mode=mode,
                                       reference_h=selected_h, cost_change_percent=100 * (ac - bc) / abs(bc),
                                       reference_mean_cost=bc / 32, oracle_mean_cost=ac / 32,
                                       reference_success=sum(r['success'] for r in reference),
                                       oracle_success=sum(r['success'] for _, r in picked),
                                       choice_counts=dict(sorted(Counter(h for h, _ in picked).items()))))
    assert len(full_rows) == 2560 and len(scene_rows) == 768 and len(totals) == 24 and len(rescues) == 256
    # Follow the one largest vehicle effect identified by the separate diagnosis.
    traces = {}
    for family, h in (('adaptive', 0), ('primary', 25), ('matched', 30)):
        arm('vehicle', family, 2, h)
        p = RUN / 'evaluations/validation/vehicle' / ('%s_h%d_s2' % (family, h)) / 'r0_trace_24.json'
        traces[family] = read(p)
        hashes[relative(p)] = sha(p)
    a, b, c = (traces[k] for k in ('adaptive', 'primary', 'matched'))
    short_steps = [i for i, r in enumerate(a) if r['horizon'] < 25]
    first_short = min(short_steps) if short_steps else len(a)
    fields = ('state', 'input', 'performance', 'compute', 'constraint', 'horizon', 'termination', 'solver_success')
    identical_prefix = all({k: x[k] for k in fields} == {k: y[k] for k in fields} for x, y in zip(a[:first_short], b[:first_short]))
    total = lambda trace: sum(r['performance'] + r['compute'] + r['constraint'] for r in trace)
    vehicle_case = dict(task='vehicle', seed=2, case=24, post_hoc_selection='Largest absolute cost difference from existing complete diagnosis.',
                        short_step_indices_zero_based=short_steps, first_short_step_zero_based=first_short,
                        adaptive_steps=len(a), primary_steps=len(b), matched_steps=len(c),
                        exact_primary_prefix_until_short=identical_prefix,
                        adaptive_primary_total_delta=total(a)-total(b), adaptive_matched_total_delta=total(a)-total(c),
                        adaptive_cost_before_first_short=total(a[:first_short]), primary_cost_before_first_short=total(b[:first_short]),
                        matched_cost_over_same_step_count=total(c[:first_short]))
    for name, value in hashes.items():
        assert sha(resolve(name)) == value
    report = dict(passed=True, post_hoc=True, full_grid_rows=len(full_rows), oracle_rows=len(scene_rows), summaries=totals,
                  pendulum_reference_failed=sum(not r['reference_success'] for r in rescues if r['task']=='pendulum'),
                  pendulum_reference_failed_rescued=sum(r['rescued'] for r in rescues if r['task']=='pendulum'),
                  vehicle_case24=vehicle_case, hashes=hashes, new_simulation_steps=0, test_accessed=False,
                  no_deployable_policy=True, validation_effect_passed=False,
                  caveats='Outcome-informed episode oracle is post-hoc and not deployable. Its filtered envelope is not an upper bound on arbitrary adaptive strategies; per-case filters are descriptive, not replacement acceptance criteria. Independent all-H grid is seed0 only. No selection from this report may reuse these validation cases as independent confirmation.')
    if DEST.exists():
        old = read(DEST / 'report.json')
        assert {k:v for k,v in old.items() if k != 'delivery_hashes'} == report
        for name, value in old['delivery_hashes'].items():
            assert sha(DEST / name) == value
    else:
        DEST.mkdir()
        for name, rows in [('all_grid_episodes.csv', full_rows), ('all_oracle_cases.csv', scene_rows), ('all_rescue_cases.csv', rescues)]:
            with (DEST / name).open('w', newline='', encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        report['delivery_hashes'] = {p.name:sha(p) for p in DEST.glob('*.csv')}
        (DEST / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('passed','summaries','pendulum_reference_failed','pendulum_reference_failed_rescued','vehicle_case24')}, indent=2))


if __name__ == '__main__':
    main()
