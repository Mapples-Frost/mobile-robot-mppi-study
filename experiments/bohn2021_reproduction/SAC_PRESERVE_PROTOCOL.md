# Teacher-preserving SAC optimization

Registered before new training or evaluation; full definitions are in
`results/sac_preserve/protocol.json`, with source/data/model hashes.

Three arms share the same rule-imitation initialization and critic-only warmup,
then run 15,000 online transitions for each of three paired seeds:

- `free`: categorical SAC without an online policy prior.
- `rule`: SAC plus cross entropy against a smoothed causal H5/H30 rule.
- `value`: SAC plus cross entropy against the frozen ensemble of all three prior
  round1 relative-branch-value teachers. These values are NOT SAC-Q targets.

All arms use seven H candidates (5,10,20,25,30,40,50), 19 causal preview features,
original reward/.6, gamma .97, alpha .01, the same frozen Riccati terminal,
SmoothL1 twin-Q regression, and gradient norm clipping. We do not attribute any
comparison with old author SAC to a single bundled change. The free arm isolates
ongoing preservation within this shared new implementation.

The earlier suggestion of reverse KL to a deterministic teacher is replaced by
well-defined forward cross entropy. The earlier 64.2% replay calculation is not
evidence that this share caused overfitting. Here the online replay proportion is
fixed at 102 teacher and 154 online rows per batch. Prior weight decreases from
2 to .5; it does not vanish, so this is a hybrid policy, not unconstrained SAC.

Use old teacher6000 per seed only where the executed integer H is in the new
action set. Every retained index and actual action/reward remains recoverable;
excluded transitions are disclosed, not relabelled as another action. The prior
branch models were trained on their old train/aggregation data; their previously
seen holdout is not reused. Report inherited 18,000 teacher transitions and the
historical branch study's 217,730 physical steps separately from new work. Never
claim equal total sample budget against the older plain SAC.

New validation seed260919131 (4 scenes), holdout260919132 (12 scenes), and online
seeds260919140..142. Validate pretrain/5k/10k/final checkpoints; test every final
model, all six old final models, fixed25, causal rule and frozen value ensemble.
The latter reveals whether online SAC improves on or merely inherits its prior.
Fixed25 is a locked historical reference, not newly selected global best H.
No checkpoint/seed selection; no tuning after holdout. All failures retained.
Seed is the replication unit; scenes are paired cases. Report cost, physical/H
components, physical and solver failures, and preview-switch response. At most
two active MPC workers. Unit/bridge smoke and inherited data budgets are separate.

## Running and reviewing

Run from the repository root under WSL Ubuntu-20.04; `.venv/bin/python` is the
existing modern PyTorch runtime. The scripts launch the pinned Python3.7 MPC
runtime through a JSON pipe. No author repository files are changed.

```bash
.venv/bin/python experiments/bohn2021_reproduction/sac_preserve_suite.py --prepare
.venv/bin/python experiments/bohn2021_reproduction/sac_preserve_agent.py --smoke
.venv/bin/python experiments/bohn2021_reproduction/sac_preserve_check.py
.venv/bin/python -u experiments/bohn2021_reproduction/sac_preserve_suite.py
.venv/bin/python experiments/bohn2021_reproduction/sac_preserve_profile.py
.venv/bin/python experiments/bohn2021_reproduction/sac_preserve_export.py
.venv/bin/python experiments/bohn2021_reproduction/sac_preserve_report.py
```

The suite skips completed jobs but deliberately refuses to overwrite partial
training. If interrupted, preserve the partial directory and account for its
steps before a documented recovery; saved checkpoints include optimizer, replay
and RNG state but not full MPC process state. Do not claim exact mid-episode
resumption from the policy checkpoint alone.

Primary review artifacts: `audit.json`, `comparison.json`, `analysis.json`,
`preservation_profile.json`, all `models/*/completed.json`, and
`report/sac_preserve/report.md`. A lower cost without matching physical and solver
reliability does not pass a reliability claim.
