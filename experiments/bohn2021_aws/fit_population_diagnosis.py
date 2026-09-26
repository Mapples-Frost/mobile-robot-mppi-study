"""Training-only behavioral diversity audit; no simulation/test access."""
import pathlib,json,statistics,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
BASE=ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/train'
records=[]
for seed in range(3):
 folder=BASE/('vehicle_s%d'%seed);fit=json.loads((folder/'fit.json').read_text());rows=[]
 for item in fit['all_candidates']:
  p=pathlib.Path(item['folder']);summary=p/'summary.json';done=json.loads((p/'completed.json').read_text())
  assert hashlib.sha256(summary.read_bytes()).hexdigest()==done['hashes'][str(summary)]
  data=json.loads(summary.read_text());eps=data['episodes'];switches=sum(e['switches'] for e in eps)
  rows.append(dict(candidate=item['id'],eligible=item['rank']['eligible'],rank=item['rank'],episodes=len(eps),steps=sum(e['steps'] for e in eps),within_episode_switches=switches,constant_H25_behavior=switches==0 and all(abs(e['mean_horizon']-25)<1e-12 for e in eps),nominal_leaf_actions=sorted(set(item['policy']['leaves'])),success=sum(e['success'] for e in eps),constraints=sum(e['constraint'] for e in eps),solver_failures=sum(e['solver_failure_steps'] for e in eps)))
 selected=json.loads((folder/'selection_registration.json').read_text())['finalists']
 record=dict(seed=seed,fit_candidates=len(rows),eligible=sum(r['eligible'] for r in rows),eligible_constant_H25=sum(r['eligible'] and r['constant_H25_behavior'] for r in rows),eligible_with_switches=sum(r['eligible'] and r['within_episode_switches']>0 for r in rows),finalist_ids=[x['id'] for x in selected],selected_policy=fit['selected'],candidates=rows)
 records.append(record)
out=ROOT/'research_artifacts/aws_diagnostics/fit_population_diagnosis';out.mkdir(parents=True,exist_ok=True)
result=dict(method='IMPROVED latency-tree retrospective training diagnosis',scope='Saved fit/select training metadata only, hashes checked, no new simulations, no validation/test reads',all_seeds=records)
(out/'raw.json').write_text(json.dumps(result,indent=2))
for r in records:print(json.dumps({k:v for k,v in r.items() if k!='candidates'}))
