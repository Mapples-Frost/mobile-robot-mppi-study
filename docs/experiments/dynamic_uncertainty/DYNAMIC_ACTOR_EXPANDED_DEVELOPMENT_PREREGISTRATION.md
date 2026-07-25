# Dynamic Actor V5 Amendment 6 Expanded Development Preregistration

## Status

Frozen before execution on 2026-07-24. This is a single-dynamic-obstacle development replication, not a sealed or confirmatory test.

## Question

Does the retained V5 Amendment 6 update-250 Actor remain noninferior in safety and completion, while preserving at least one strict efficiency or completion improvement, when compared with the V3 source Actor on a larger set of previously unused development seeds?

## Design

- Independent unit: one complete paired episode seed.
- Pairs: 24, split into three eight-pair batches.
- Seeds: 730100204--730100227, with execution order frozen in the YAML protocol.
- Both arms use common random numbers, proposal-only integration, the same same-cycle guided-cost filter, the same safety stack, and 600 rollouts per controller decision.
- The only intended treatment difference is the Actor checkpoint.
- Source: V3 roll-in Actor.
- Candidate: V5 Amendment 6 update 250.
- Each batch contains four source-first and four candidate-first pairs; order was generated with schedule seed 730199024.
- No checkpoint selection, hyperparameter adjustment or seed replacement is allowed during the matrix.
- Collisions do not stop the simulator matrix; they fail the applicable safety gate and remain in the complete outcome record.
- Sealed seeds remain closed.

## Frozen gates

The protocol passes only if all of the following hold:

1. all 24 pairs complete;
2. candidate-only new collisions equal zero;
3. lost source successes equal zero;
4. pooled candidate success is not lower than pooled source success;
5. pooled candidate total steps are not higher than source total steps;
6. pooled candidate mean final distance is not higher than source mean final distance;
7. at least one pooled success, step or final-distance endpoint improves strictly;
8. at least two of the three batches are jointly nonregressive in success, collision, total steps and mean final distance;
9. all artifact/checkpoint/config bindings match their frozen SHA-256 values;
10. the same-cycle filter is enabled and exercised in both arms.

No p-value threshold is used as a substitute for the above paired engineering gates. The result is interpreted as expanded development evidence only.

## Outputs

- `research_artifacts/dynamic_actor_v5a6_u000250_samecycle_expanded_development/protocol_snapshot.json`
- one immutable pair directory per scheduled seed;
- `progress.json`;
- `summary.json`;
- `gate.json`.

The exact schedule, bindings and machine-readable gate are frozen in `configs/research/dynamic_actor_v5a6_samecycle_expanded_development.yaml`.
