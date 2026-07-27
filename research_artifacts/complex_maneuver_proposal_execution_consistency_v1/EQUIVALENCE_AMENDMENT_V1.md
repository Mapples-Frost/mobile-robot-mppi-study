# Phase Zero equivalence amendment v1

## Decision

The Phase Zero equivalence oracle is changed from the historical one-off
trajectory to a concurrent rerun of the original `af691f1` source under the
same current runtime used for the diagnostic implementation.

This amendment does **not** introduce a numerical tolerance. After excluding
the same nine frozen timing columns, every other common trajectory field must
remain exactly equal as a string at every row. Outcomes and the `0/600`
rollout contract must also match.

## Why an amendment is necessary

The historical Chapter 1 Actor-off artifact could not be reproduced by its own
source commit under the current runtime. The verified `af691f1` rerun instead
matched the isolated diagnostic build exactly on all 593 non-timing common
columns. Both runs ended with the same 211-step collision and the same rollout
counts.

Therefore, the earlier mismatch cannot identify a Phase Zero diagnostic side
effect. It identifies a runtime-level reproducibility boundary in the
historical artifact. Keeping that artifact as the bitwise oracle would make
the gate impossible to satisfy even for its own original source.

## Frozen comparison

- Unit: the same Chapter 1 map/seed/arm/runtime block.
- Reference: detached `af691f1` source with verified import path and
  materialized LFS checkpoints.
- Candidate: the isolated Phase Zero diagnostic implementation.
- Exclusions: only the nine named timing columns in the amendment YAML.
- Equality: exact row count, exact row alignment, and exact string equality
  for every remaining common column.
- Additional contracts: identical success, collision, termination reason and
  steps; per-cycle rollouts may only be 0 or 600.

This is an implementation-equivalence audit, not an independent performance
replicate and not evidence of task improvement.

## Scope and authorization

The concurrent Actor-off comparison passes the amended oracle. This authorizes
only the Actor-on, margin-zero equivalence audit after the amendment tests
pass. It does not authorize margin 0.02, retraining, data expansion, a paid
server, or a formal experiment.

If Actor-on margin zero differs, all evidence is preserved and the workflow
stops before any nonzero margin. Risk, ICODE, MPPI costs, Safety, maps, the
Actor checkpoint, and the 600-rollout budget remain frozen.

## Historical artifact

The original trajectory remains sealed and cited as a reproducibility
limitation. It is neither overwritten nor silently discarded; it is simply no
longer used as the impossible bitwise oracle.
