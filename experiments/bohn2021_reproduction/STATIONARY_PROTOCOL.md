# Stationary Terminal and Categorical Horizon Optimization

This is an explicitly modified method study following the 2026-09-18 credit
diagnosis. It does not claim to recover the paper's missing experiment setup.

## Registered Comparisons

1. Author continuous SAC with frozen initial Riccati terminal versus all three
   existing joint-learning final models from `prior_refinement`. The frozen
   wrapper omits only the terminal optimizer execution. Graph construction,
   both replay draws, initialization, reward and horizon policy are retained.
   Initial model and configuration hashes must match for each seed.
2. Tianshou 0.5.1 continuous versus categorical SAC, both using the identical
   legacy MPC worker and frozen Riccati terminal. Tianshou implements the core
   SAC updates; no custom SAC update or DQN substitution is used.

Every learned arm uses seeds 0, 1, 2, 15,000 transitions, 14,745 updates,
gamma .97, alpha .01, Adam .0003, tau .005, batch 256, actor 32x32, critic
256x256, and physical rewards divided by .6. Modern arms share the first 100
uniform integer-H exploration actions. Time-limit bootstrap remains enabled;
physical constraint termination stops bootstrap. The first training action
retains the legacy zero-terminal callback timing. Evaluation always uses the
full frozen prior, with the common zero-terminal H50 reset warmup.

The frozen wrapper asserts terminal equality after every update. The modern
worker asserts it after every step. Checkpoints are retained every 2,500 steps;
only final checkpoints enter the comparisons.

Legacy checkpoints are evaluated through the frozen evaluation wrapper, with
reward_scale=.6 explicitly restored. The author save format omits this scalar,
and loading a frozen checkpoint through plain SAC then calling learn would
restore its terminal optimizer. Replay state is not saved; this protocol
supports complete seeded training runs, not checkpoint-based training resume.

## Selection and Evaluation

Validation: 10 new scenes, seed 26091820. Holdout: 30 new scenes, seed 26091821.
All arms share these banks. Fixed analytic-terminal H candidates are
1, 5, 10, ..., 50, selected by validation mean total cost with smaller-H ties.
This fixed baseline needs zero training transitions and has no independent
training seeds. The prior joint-terminal fixed H30 models remain a separate
three-seed reference.

The joint-discrete follow-up gate uses validation only: no constraints in any
discrete seed, and at least 2% lower mean total cost than both the matched
modern continuous arm and validation-selected analytic fixed H. Failing the
gate means no joint learning restoration or holdout-driven temperature tuning.

Report each seed, constraint terminations, total/performance/H proxy costs,
mean H and solver failures. Thirty scenes are repeated evaluations, not thirty
independent training replicates. No significance claim is planned from n=3.
Saved trajectories are independently audited against the physical equations,
input bounds, reference timeline and common reset states.

## Interpretation Limits

- Frozen versus joint changes the whole terminal update trajectory, including
  the opportunity to learn a better terminal. A worse frozen result cannot
  alone disprove an adverse effect of nonstationarity.
- Modern SAC uses twin-Q targets and a minQ actor objective; the legacy author
  code uses a separate target V network and Q1 actor objective. The matched
  modern continuous arm controls the framework change for the discrete study.
- Equal numeric alpha does not equate continuous differential entropy with
  categorical entropy. The second comparison tests standard method packages,
  including action representation and Q outputs, not rounding in isolation.
- The Riccati prior is a local stationary-reference linearization. It is not
  an exact constrained value function for changing references.
- H penalty is a computation proxy. Concurrent wall times are not a realtime
  speed benchmark.

## Resource Recovery

Combined concurrency caused host memory pressure and WSL command timeouts.
The four incomplete modern attempts are retained under
`results/categorical_frozen/interrupted_memory`; their extra transitions are
reported separately from the nine completed formal models and three smoke
tests. No interrupted result is used for seed or checkpoint selection.

`categorical_resource_run.py` limits physical replay allocation to the smaller
of the configured capacity and the run's total steps. No transition is evicted
before this run ends. A paired buffer test verifies identical sample indices,
stored transitions and unfinished-episode indexing through the final insert.
The recovery launcher runs at most three jobs after legacy evaluations finish.
Original registered scripts remain unchanged; wrapper hashes and the reason
for this execution-only amendment are saved in `resource_amendment.json`.

## Reproduction

Run from the WSL repository root. Python 3.7 hosts the pinned author sources;
Python 3.8 plus torch 2.4.1+cpu hosts Tianshou. Exact isolated dependency
versions are saved in `results/categorical_frozen/dependencies.json`.

```bash
/home/mapples/.local/share/bohn2021-python37/bin/python experiments/bohn2021_reproduction/stationary_suite.py
/home/mapples/.local/share/bohn2021-python37/bin/python experiments/bohn2021_reproduction/stationary_reload_verify.py
.venv/bin/python experiments/bohn2021_reproduction/categorical_recover.py
.venv/bin/python experiments/bohn2021_reproduction/stationary_report.py
```

Machine-readable registrations are `results/stationary_terminal/protocol.json`
and `results/categorical_frozen/protocol.json`. Their source hash inventories
are checked again by the final report script.
