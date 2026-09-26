"""Snapshot all new-study attempts, including archived interruptions and extra fixed training."""
import json
import time
from collections import defaultdict
from pathlib import Path
from gated_horizon_search import OUT
from paper_h_soft_probe import read,digest
from run import write


def main():
    groups=defaultdict(lambda:dict(environment_meters=0,explicit_step_attempts=0,explicit_reset_attempts=0,raw_solve_attempts=0,raw_solve_completed=0,warmup_attempts=0,retry_attempts=0,training_instrumentation_steps=0,training_instrumentation_resets=0))
    files={};incomplete=[]
    for p in OUT.rglob('*.json'):
        if p.name.startswith('attempt_'):
            d=read(p)
            if 'step_calls' not in d:continue
            g=groups[p.relative_to(OUT).parts[0]];g['environment_meters']+=1;g['explicit_step_attempts']+=d['step_calls'];g['explicit_reset_attempts']+=d['reset_calls'];files[str(p)]=digest(p)
        elif p.name=='solver_attempts.json':
            d=read(p);g=groups[p.relative_to(OUT).parts[0]]
            for src,dst in [('solve_attempts','raw_solve_attempts'),('solve_completed','raw_solve_completed'),('warmup_attempts','warmup_attempts'),('retry_attempts','retry_attempts')]:g[dst]+=d[src]
            if d['solve_attempts']!=d['solve_completed']:incomplete.append(dict(path=str(p),counts=d))
            files[str(p)]=digest(p)
        elif p.name.startswith('instrumentation_attempt_'):
            d=read(p);g=groups[p.relative_to(OUT).parts[0]];g['training_instrumentation_steps']+=d['step_attempts'];g['training_instrumentation_resets']+=d['reset_attempts'];files[str(p)]=digest(p)
    totals={k:sum(g[k] for g in groups.values()) for k in next(iter(groups.values()))}
    audits=[]
    for p in OUT.glob('audit*.json'):
        d=read(p)
        if d.get('passed'):audits.append(dict(path=str(p),phase=d.get('phase'),recorded_steps=d.get('recorded_steps',d.get('steps')),hash=digest(p)))
    result=dict(snapshot_time=time.time(),groups=dict(groups),totals=totals,unresolved_or_current_solve_attempts=incomplete,hashes=files,audits=audits,
        inherited_terminal_training=dict(independent_seed0_full10_H_grid_both_tasks=300000,selected_primary_seeds1_and2_both_tasks=60000),
        caveats=['Live snapshots are not atomic across processes. Completion requires terminal process states and exact reconciliation.',
                 'Environment meters begin after construction; construction-internal calls are not included. Number of meters is not a complete constructor audit.',
                 'Bank creation has reset meters but no raw solver wrapper; its reset warmup calls are not in raw solve totals.',
                 'Fixed-training instrumentation counts train and final evaluation transitions together; per-model independent audit separates them. Its raw NLP attempts are not instrumented by the recovery wrapper.',
                 'Archived incomplete candidates are charged in full. Some work will be repeated under unchanged code; no archived attempts discarded.',
                 'Numerical audit integrations are additional offline computation, not controller simulation; individual audit reports give their coverage.'],source_hash=digest(Path(__file__)))
    dest=OUT/'budget';dest.mkdir(exist_ok=True);write(dest/'snapshot.json',result)
    print(json.dumps(dict(totals=totals,groups=dict(groups)),indent=2))


if __name__=='__main__':main()
