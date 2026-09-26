"""Audit existing vehicle seed1 selection; no new simulation or held-out outcomes."""
import pathlib,json,hashlib,collections,statistics,time
ROOT=pathlib.Path(__file__).resolve().parents[2];base=ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/train/vehicle_s1';fit=json.loads((base/'fit.json').read_text());rows=[]
for p in sorted((base/'selection').iterdir()):
 if not p.is_dir():continue
 done=json.loads((p/'completed.json').read_text());errors=[]
 for name,h in done['hashes'].items():
  q=pathlib.Path(name)
  # Inputs located in sources may still be transferring. This experiment audits raw condition files only.
  if q.parent!=p:continue
  if hashlib.sha256(q.read_bytes()).hexdigest()!=h:errors.append(name)
 assert not errors,errors
 counts=collections.Counter();leaves=collections.Counter();times=[];switches=0;cases=0
 for trace in sorted(p.glob('r0_trace_*.json')):
  data=json.loads(trace.read_text());hs=[r['horizon'] for r in data];counts.update(hs);switches+=sum(a!=b for a,b in zip(hs,hs[1:]));cases+=1
  for row in data:
   if 'leaf' in row['decision']:leaves[row['decision']['leaf']]+=1
 summary=json.loads((p/'summary.json').read_text());eps=summary['episodes']
 assert sum(counts.values())==summary['steps']
 rows.append(dict(condition=p.name,raw_hashes_verified=True,cases=cases,steps=sum(counts.values()),horizon_counts=dict(counts),leaf_counts=dict(leaves),within_episode_switches=switches,mean_episode_total_cost=statistics.mean(x['total_cost'] for x in eps),mean_decision_seconds=sum(x['decision_total_s'] for x in eps)/sum(x['steps'] for x in eps),success=sum(x['success'] for x in eps),constraints=sum(x['constraint'] for x in eps),solver_failures=sum(x['solver_failure_steps'] for x in eps)))
result=dict(scope='Existing training-only selection, WSL timings; no new simulation, validation or test reads',selected=fit['selected'],learned_tree_selected=fit['learned_tree_selected'],fit_candidates=len(fit['all_candidates']),eligible_fit_candidates=sum(r['rank']['eligible'] for r in fit['all_candidates']),conditions=rows,conclusion='Vehicle seed1 selected constant H25. This saved policy cannot satisfy the preregistered per-seed within-episode adaptation gate. That is a gate incompatibility, not a full final-test result. Most selected candidates may be behaviorally constant; raw occupancy below resolves this.',next_hypothesis='Inspect whether safe fit candidates actually vary H, and whether fit-to-selection loss reflects distribution shift, inactive leaves, or time noise. Preserve old negative selection; no test opening.')
out=ROOT/'docs/bohn2021_takeover/vehicle_seed1_raw_diagnosis.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
