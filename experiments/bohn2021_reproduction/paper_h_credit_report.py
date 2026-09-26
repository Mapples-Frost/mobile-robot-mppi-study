"""Independent audit and report for paper_h_credit_probe outputs."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from paper_h_credit_probe import METHODS, TASKS, OUT, SOURCE, CASES, ANCHORS, read, digest, protocol, model_dir
from paper_grid_audit_report import physics


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-6), (a, b)


def audit_branches(task, folder, summary, case, original):
    path = folder / ('case%02d_t%03d.json' % (summary['case'], summary['anchor']))
    data = read(path)
    assert data['summary'] == summary
    anchor, picks = summary['anchor'], summary['picks']
    assert set(data['branches']) == set(str(h) for h in picks.values())
    assert picks['q1'] == max(range(50), key=lambda i: summary['q1'][i]) + 1
    assert picks['min_q'] == max(range(50), key=lambda i: min(summary['q1'][i], summary['q2'][i])) + 1
    assert picks['fixed'] == protocol()['fixed_H'][task]
    max_steps = 150 if task == 'vehicle' else 100
    checked = 0
    for h, branch in data['branches'].items():
        trace = branch['trace']
        assert trace and trace[0]['horizon'] == int(h) == branch['first_h']
        assert branch['start_state'] == original[anchor - 1]['state']
        assert len(trace) <= max_steps - anchor
        costs = []
        for k, step in enumerate(trace):
            t = anchor + k
            assert 1 <= step['horizon'] <= 50
            physical, violated, excess = physics(task, step, case, t)
            compute = step['horizon'] * (.001 if task == 'vehicle' else .003)
            penalty = (2 if task == 'vehicle' else 10) * (max_steps - t - 1) if violated else 0.
            for observed, expected in ((step['performance'], physical), (step['compute'], compute),
                                       (step['constraint'], penalty), (-step['reward'], physical + compute + penalty)):
                close(observed, expected)
            assert excess <= 1e-5
            if violated:
                assert k == len(trace) - 1 and step['termination'] == 'constraint'
            if k < len(trace) - 1:
                assert step['termination'] is None
            costs.append(physical + compute + penalty)
        close(branch['total_cost'], sum(costs))
        close(branch['discounted_cost'], sum(.97**k * c for k, c in enumerate(costs)))
        assert branch['solver_failure_steps'] == sum(not s['solver_success'] for s in trace)
        assert branch['termination'] == trace[-1]['termination']
        if branch['termination'] == 'steps':
            assert anchor + len(trace) == max_steps
        elif branch['termination'] == 'constraint':
            assert violated
        else:
            assert branch['termination'] == 'goal' and task == 'vehicle' and not violated
            end = case['reference']['traj_steps'] - 1
            s = trace[-1]['state']
            assert math.hypot(s['x'] - case['tvp']['trajectory_x'][end]['true'][0],
                              s['y'] - case['tvp']['trajectory_y'][end]['true'][0]) <= .5 + 1e-8
        assert summary['branches'][h] == {k: v for k, v in branch.items() if k not in ('trace', 'start_state')}
        if int(h) == picks['actor']:
            assert len(trace) == len(original) - anchor
            for actual, saved in zip(trace, original[anchor:]):
                assert actual['state'] == saved['state'] and actual['horizon'] == saved['horizon']
                assert actual['input'] == saved['input'] and actual['solver_success'] == saved['solver_success']
                close(actual['reward'], saved['reward'])
        checked += len(trace)
    return len(data['branches']), checked, str(path), digest(path)


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main():
    prepare = OUT / "protocol.json"
    assert read(prepare) == protocol()
    input_hash = digest(OUT / "inputs_sha256.json")
    for path, expected in read(OUT / 'inputs_sha256.json').items():
        assert digest(Path(path)) == expected, path
    rows = []
    actual_branches, trace_steps = 0, 0
    trace_hashes = {}
    for task in TASKS:
        cases = read(SOURCE / (task + '_validation_bank.json'))['cases']
        for method in METHODS:
            for seed in range(3):
                folder = OUT / ("%s_%s_s%d" % (task, method, seed))
                done = read(folder / "completed.json")
                assert done["frozen"] and done["input_hashes_sha256"] == input_hash
                assert done['model_hash'] == read(model_dir(task, method, seed) / 'completed.json')['final_hash']
                summaries = done["summaries"]
                assert len(summaries) == 4
                assert {(s['case'], s['anchor']) for s in summaries} == {(c, a) for c in CASES for a in ANCHORS}
                disagree_q1 = disagree_min = 0
                q1_better = min_better = fixed_better = 0
                q1_delta, min_delta, fixed_delta = [], [], []
                solver_failures = {"actor": 0, "q1": 0, "min_q": 0, "fixed": 0}
                for summary in summaries:
                    original = read(SOURCE / 'evaluations/validation' / task /
                                    (method + '_s%d' % seed) / ('trace_%02d.json' % summary['case']))
                    count, steps, path, sha = audit_branches(task, folder, summary, cases[summary['case']], original)
                    actual_branches += count
                    trace_steps += steps
                    trace_hashes[path] = sha
                    picks = summary["picks"]
                    if picks["q1"] != picks["actor"]: disagree_q1 += 1
                    if picks["min_q"] != picks["actor"]: disagree_min += 1
                    branches = summary["branches"]
                    actor = branches[str(picks["actor"])]
                    q1 = branches[str(picks["q1"])]
                    min_q = branches[str(picks["min_q"])]
                    fixed = branches[str(picks["fixed"])]
                    for key, branch in (("actor", actor), ("q1", q1), ("min_q", min_q), ("fixed", fixed)):
                        solver_failures[key] += branch["solver_failure_steps"]
                    q1_delta.append(q1["total_cost"] - actor["total_cost"])
                    min_delta.append(min_q["total_cost"] - actor["total_cost"])
                    fixed_delta.append(fixed["total_cost"] - actor["total_cost"])
                    q1_better += q1["total_cost"] < actor["total_cost"]
                    min_better += min_q["total_cost"] < actor["total_cost"]
                    fixed_better += fixed["total_cost"] < actor["total_cost"]
                    for branch in branches.values():
                        assert branch["termination"] in ("goal", "constraint", "steps")
                        assert branch["solver_failure_steps"] >= 0
                rows.append({
                    "task": task, "method": method, "seed": seed,
                    "anchors": len(summaries), "actor_q1_disagreements": disagree_q1,
                    "actor_min_q_disagreements": disagree_min,
                    "q1_better_than_actor": q1_better,
                    "min_q_better_than_actor": min_better,
                    "fixed_better_than_actor": fixed_better,
                    "mean_q1_minus_actor": mean(q1_delta),
                    "mean_min_q_minus_actor": mean(min_delta),
                    "mean_fixed_minus_actor": mean(fixed_delta),
                    "solver_failures": solver_failures,
                    "model_hash": done["model_hash"],
                })
    assert len(rows) == 12 and all(r["anchors"] == 4 for r in rows)
    result = {"passed": True, "models": len(rows), "anchors": sum(r["anchors"] for r in rows),
              "logical_policy_conditions": 12 * 4 * 4, "unique_rollouts": actual_branches,
              "audited_suffix_steps": trace_steps, "trace_hashes": trace_hashes,
              "input_hash": input_hash, "rows": rows,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(OUT / "audit.json", result)
    md = ["# Paper-configuration H credit diagnosis", "", "This is a frozen validation-only mechanism diagnosis, not a new performance claim or strict reproduction.", "",
          "| task | method | seed | actor/Q1 disagreement | actor/min-Q disagreement | Q1 wins | min-Q wins | mean (Q1-actor) | mean (min-Q-actor) | mean (fixed-actor) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append("| {task} | {method} | {seed} | {actor_q1_disagreements}/4 | {actor_min_q_disagreements}/4 | {q1_better_than_actor}/4 | {min_q_better_than_actor}/4 | {mean_q1_minus_actor:.4f} | {mean_min_q_minus_actor:.4f} | {mean_fixed_minus_actor:.4f} |".format(**r))
    md += ["", "Independent audit: %d unique rollouts, %d suffix steps, 48 shared starting states. Recomputed physical, H and constraint costs, discounted totals, termination and solver counts; checked raw input hashes and actor suffixes against saved validation trajectories. Equal first H choices share a rollout; 192 logical conditions are not 192 independently executed trajectories." % (actual_branches, trace_steps), "", "Lower cost is better. The table reports undiscounted finite deterministic suffix costs. SAC critics estimate stochastic entropy-regularized discounted returns with possible timeout bootstrap: these comparisons are not ground-truth soft-Q calibration. Actor/critic disagreement alone does not prove actor optimization failure. Two scenes per task with correlated anchors do not support a general success rate. This evidence can motivate a separately registered training intervention, but is not a confirmation or a reason to open a gated test set."]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"passed": True, "models": len(rows), "anchors": sum(r["anchors"] for r in rows)}))


if __name__ == "__main__":
    main()
