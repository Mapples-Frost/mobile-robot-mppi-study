# Paper-task transfer gate, 2026-09-23

The user prioritizes continuing Bøhn replication/optimization before redesigning
the mobile-robot experiment to identify where joint K/H allocation helps.
The existing successful `sac_preserve` extension keeps a teacher policy prior,
seven H candidates and a frozen Riccati terminal. It is not the author algorithm.

First test its transfer before adding moving terminal targets. This is a
diagnostic evaluation with **zero new training updates**, not an optimization
result. The next training intervention depends on the diagnosed factor.

At 600 steps, cross sparse plateau versus reference redraw probability .04 with
near-equilibrium versus the broad reconstructed initial-state distribution.
Add a 100-step redraw/broad condition with exactly the same reference prefix and
initial state as its 600-step pair. These settings come from local reconstruction
assumptions; the original author's exact experiment files remain unavailable.

All nine final free/rule/value policies, the causal rule, and the frozen value
ensemble are retained. Fixed H in {1,5,10,...,50} is selected per domain on four
new validation scenes by physical failure count, solver failure fraction, mean
raw cost, then H. Each domain then uses twelve independent holdout scenes shared
by every policy. No learned checkpoint or seed selection.

- Validation: 22 conditions × 5 domains × 4 scenes = 440 episodes.
- Holdout: 12 conditions × 5 domains × 12 scenes = 720 episodes.
- Separate smoke: 3 policies × 2 domains × 1 scene = 6 episodes.
- Seed is the learning replication unit. Factor contrasts use paired scenes;
  five domains are not pooled into one headline result.
- Primary comparison: value arm against the validation-selected fixed H in
  redraw/broad/100. Also report the margin against the frozen value teacher.
- Progression gate: every value seed has lower mean cost than fixed H, with
  no increase in physical failures or solver failure fraction. Passing permits
  a separately registered joint-terminal study; it does not prove replication.
- If it fails, diagnose distribution transfer, register fresh training and new
  validation/test seeds, then optimize. Never tune against this holdout.

Reward, plant, bounds and solver stay fixed. Both durations use the actual
remaining fraction (T-t)/T in the existing observation interface; the changed
time scale is disclosed. Adjusted cost = raw cost + .4905*T. After a failure,
report physical, H, failure penalty and unexecuted constant offset separately.
Do not compare absolute costs across durations or interpret H as CPU latency.

The source/model/bank hashes and machine-readable protocol are frozen by
`paper_transfer_suite.py --prepare`. Existing Bøhn study files are not rewritten.
Fresh MPC state per episode and shared reset warmup enforce paired starts.
Saved trajectories are independently checked for reward, action inference,
preview, termination, summaries and paired starts before completion.

Only two single-threaded MPC processes may run. The suite waits for all
measured-delay workers/drivers and `/tmp/kh_backfill.sh`; no simulator or numerical
runtime is imported during the wait. Smoke must pass before formal evaluations.
Episode files are atomic, incomplete attempts are retained and counted as lower
bounds. A suite lock prevents duplicate launch.

Run under WSL from the repository root:

```bash
python3 experiments/bohn2021_reproduction/paper_transfer_suite.py --check
python3 experiments/bohn2021_reproduction/paper_transfer_suite.py --prepare
python3 -u experiments/bohn2021_reproduction/paper_transfer_suite.py --wait-for-idle
```

The worker and full audit use the existing pinned Python 3.7 MPC runtime. Main
artifacts: `results/paper_transfer_2026-09-23/status.json`, `gate.json`, `audit.json`,
`report.md`. Until smoke executes, the runtime bridge is prepared but unverified.
