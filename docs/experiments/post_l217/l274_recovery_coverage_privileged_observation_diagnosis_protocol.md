# L274 Recovery Coverage and Privileged-Observation Diagnosis

## Question

L273 showed that the current Critic function class can almost perfectly fit
the L263 action grid and generalize to L263 validation scenes, yet fails on the
independent L268 recovery-state distribution. Increasing width or adding a
degree-3 action basis did not help. L274 separates two remaining explanations:

1. L263 did not contain enough recovery-distribution states;
2. the current 69D observation omits path-local information needed to rank
   actions in those states.

This is a Critic-only oracle-fit diagnostic. Privileged values and oracle H40
returns never enter an Actor, replay buffer, MPPI, or deployable observation.

## Leakage-safe three-fold cross-fitting

The 36 L268 states contain six states from each of six training scenes. Within
each scene, states are sorted by severity, side, and chain id, then assigned by
index modulo three. Each fold holds out exactly two states per scene (12 total)
and trains on the other 24. Each recovery state is evaluated exactly once and
never by a model that fitted its returns. L263's 18 training-role states remain
the common source data. The final three maps, L258, and sealed seeds are
forbidden.

## Frozen 2 x 2 design

The four arms isolate recovery-state coverage and privileged observation:

| Arm | Training states | State features |
| --- | --- | --- |
| `source_69d` | L263 only | frozen 69D |
| `source_privileged75d` | L263 only | 69D + six path-local diagnostics |
| `recovery_augmented_69d` | L263 + recovery train fold | frozen 69D |
| `recovery_augmented_privileged75d` | L263 + recovery train fold | 69D + six diagnostics |

The six diagnostics are signed and absolute cross-track error, heading error,
curvature, progress fraction, and remaining fraction. They are intentionally
privileged probes, normalized using training-fold statistics only. Their use
does not authorize adding them to the formal method.

All arms use the baseline 256x256, 25-quantile Critic, raw normalized 2D action,
3,000 updates, batch 256, identical optimizer settings, and three paired model
seeds. Returns are standardized within state so the task is action ranking, not
state-value offset regression.

## Frozen decision

Cross-fitted predictions are aggregated once per paired seed. An intervention
must reach 0.65 recovery-vs-forward accuracy, improve paired median accuracy
and Spearman by at least 0.10, improve at least four of six scenes, avoid a
scene decrease greater than 0.166667, and improve consistently in at least two
of three seed blocks.

- A privileged-only pass identifies missing observation information.
- A recovery-augmentation pass identifies state-distribution coverage.
- If only their combination passes, both mechanisms matter.
- If none passes, the next diagnosis is temporal history or return-target
  dynamics; Actor training remains prohibited.

No threshold, seed, fold, checkpoint, or feature may be selected after outcomes
are viewed. All failures and raw predictions are retained.

