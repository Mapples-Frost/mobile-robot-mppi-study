# L281 Component-Separated Recovery-Retention Anchor Gate

## Causal question

L280 localized recovery/source anchor conflict to the linear-speed mean output
and found neither broad SAC-vs-anchor conflict nor anchor dominance.  L281 asks
whether separating the two anchor roles only at that conflicted component
preserves recovery behavior without blocking useful online SAC adaptation.

## Single intervention

L281 is paired exactly to L279 and reuses its three training seeds, L276
initial checkpoints, six training scenes, three independent validation scenes,
6,000-step budget, replay semantics, optimizer settings, and immutable L279
train-only anchor dataset.  The only change is source-kind-aware mean-loss
routing:

- recovery rows supervise normalized linear and angular action means;
- source-distillation rows supervise only the angular action mean;
- both source kinds retain the frozen log-standard-deviation anchor;
- the online SAC loss remains unchanged.

The mean-loss denominator remains the original batch-size times action
dimension; active terms are not renormalized after masking.  Thus the
intervention deletes only the preregistered source/linear-speed contribution
and does not amplify the remaining recovery or angular gradients.

This is the minimal deployable realization of the L280-authorized separated
anchor intervention: no new observation, router, parameter, or inference-time
branch is introduced.  It removes only the source gradient from the conflicted
linear-speed mean component while retaining the recovery gradient there.

Each anchor batch contains exactly 128 recovery and 128 source rows.  Sampling
RNG state, source-kind labels, and component weights are checkpointed and must
resume exactly.  The intervention is active only when Actor updates begin at
step 3,000.

## Frozen evaluation and Gate

Only step 6,000 is compared with the paired L279 and L277 controls.  The same
L268 validation/test recovery chains and the same L277 independent validation
Gate are evaluated after all three seeds finish.  The Gate requires:

- complete native-Windows CUDA provenance and exact resume behavior;
- held-out recovery RMSE no worse than initialization and at least 20% better
  than paired L277;
- held-out recovery return/reentry and collision/boundary safety pass the L279
  thresholds;
- each seed has no collision increase, success loss, or completion regression
  beyond 0.02;
- median completion change is nonnegative;
- median CTE improves by at least 0.05 or median goal distance by at least 0.10;
- at least two of three seeds and two of three validation scenes improve in the
  preregistered directions.

No threshold, seed, checkpoint, component mask, or mixture may be selected
from results.  L258, sealed seeds, and final Hairpin/S-Chicane/Infinity are
forbidden.  A negative Gate stops expansion and preserves all raw artifacts.
A positive Gate authorizes only preregistration of a larger independent-seed
validation; it never authorizes final-map evaluation directly.
