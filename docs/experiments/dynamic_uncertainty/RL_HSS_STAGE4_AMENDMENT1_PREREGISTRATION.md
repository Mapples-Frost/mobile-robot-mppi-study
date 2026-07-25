# Dynamic-Uncertainty Stage 4 Amendment 1 Preregistration

Date: 2026-07-24  
Status: frozen after an implementation-only failed launch and before any
completed Stage 4 episode.

## Trigger and retained failure

The original Stage 4 protocol was formally launched with protocol SHA-256
`111babd78f9b1b9d2c9d0b545df184951419c3e9ec996aeb772d2052f808fa5c`.
The first randomized cell was
`nominal::seed730100006::rl_hss_on`. Its first planning call raised:

`ValueError: probabilistic obstacle risk is enabled but forecasts are absent`

The process stopped before a control command was executed and before an episode
produced `metrics.json` or `trajectory.csv`. Completed Stage 4 episodes were
therefore `0/24`; no outcome was observed, no cell was selected, and sealed
seeds remained closed. The original output directory, manifest, resolved
configuration, schedule, stdout and stderr are retained unchanged.

## Root cause

The Paper RL-MPPI optimizer was previously qualified only in environments that
did not exercise the dynamic-obstacle probabilistic-risk path. Its custom
optimizer called the shared trajectory cost without forwarding the observation
forecast. It also did not carry the standard optimizer's hard candidate filter,
final active-avoidance/stop-is-safest selection, missing-forecast stop, or
probability/tracker diagnostics.

The preflight tests constructed real Stage 4 controllers and verified hashes,
budgets, state isolation and causal HSS updates, but did not execute one Paper
`plan()` call with the probabilistic forecast contract. This test omission is
retained as part of the failure record.

## Single bounded amendment

Amendment 1 changes only the Paper optimizer's integration with the already
frozen probability-risk mechanism:

1. extract the configured forecast tuple from the current observation;
2. delegate a missing forecast to the existing deterministic fail-closed stop;
3. include the unchanged probabilistic trajectory cost;
4. reserve one braking sequence inside the existing candidate budget;
5. apply the unchanged probability hard filter to each Paper iteration;
6. apply the unchanged final active-avoidance, minimum-risk and
   stop-is-safest action guard;
7. emit probability-risk and dynamic-tracker diagnostics.

The action guard selects only from the final Paper iteration's already-budgeted
candidates. It adds no rollout, privileged observation, controller authority or
safety bypass. RL-on remains `300 x 2 = 600` candidates per controller decision.

## Frozen implementation binding

Amendment 1 preflight verifies byte-level SHA-256 hashes for the shared MPPI
risk code, Paper optimizer, controller factory, Actor adapter and Stage 4
runner. The binding is recorded in
`configs/research/dynamic_uncertainty_rl_hss_stage4_amendment1.yaml` under
`implementation_contract: paper_probabilistic_risk_parity_v1`.

Before this document was frozen:

- two new tests executed Paper `plan()` directly;
- missing forecast returned the frozen zero-command fail-closed result;
- a synthetic mixed safe/unsafe batch retained risk candidate filtering and
  selected safe active motion;
- the Stage 4 focused suite passed `9/9` tests;
- the Paper RL, generic RL, probability-risk, residual-shield and runtime
  regression group passed `61/61` tests.

No MuJoCo Stage 4 episode was run during amendment implementation or testing.

## Unchanged scientific protocol

The following remain byte-for-byte or value-for-value identical to the
original preregistration:

- three obstacle seeds and randomized 24-cell schedule;
- complete episode as the independent unit;
- nominal plus all three Stage 3 residual blocks;
- RL/HSS off/on factors and common random numbers;
- Amendment 17 environment, Change-Aware IMM and Risk V1 thresholds;
- Actor, HSS sidecar and residual checkpoint hashes;
- MPPI horizon, costs, action limits and total candidate budget;
- matched residual shield, parallel planning and final scan guard;
- first-collision early stop and sealed-seed prohibition.

Because no original episode completed, Amendment 1 restarts the same 24-cell
schedule from the beginning in the new directory
`research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development`.
The failed original directory is excluded from all treatment analysis.

## Execution and interpretation boundary

The amended runner still defaults to preflight. Formal execution requires both
the Amendment 1 protocol and explicit `--execute`. A protocol, implementation
hash or schedule mismatch fails closed before creating or resuming an episode.

This amendment repairs treatment integrity only. It does not predict, improve
or relabel any Stage 4 closed-loop outcome.
