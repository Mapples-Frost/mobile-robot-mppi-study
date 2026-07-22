# L275 Recovery Commitment and Continuation-Semantics Diagnosis

## Question

L269 found that a complete recorded recovery chain beats the frozen Actor in
91.7% of held-out recovery states, while a single recovery action followed by
the same Actor is advantageous in only 55.6% at H40. L274 then showed that
recovery-state coverage plus static privileged path fields is not sufficient
to repair held-out action ranking. L275 tests the remaining causal explanation:

> recovery is an extended action sequence whose benefit is destroyed when
> control is returned to the current Actor too early.

This is a diagnostic-only experiment. It does not train an Actor or Critic and
does not modify reward, observation, ICODE, MPPI, fusion, or safety semantics.

## Frozen states and continuations

The 36 L268 validation and test recovery chains are hash-pinned. For every
state, MuJoCo is reset with the frozen L268 reset contract. The evaluation
horizon is the complete accepted-chain length. The baseline applies the frozen
source Actor throughout. Each intervention applies the recorded recovery
sequence for a fixed commitment of 1, 5, 10, 20, or 40 steps and then follows
the same frozen Actor. A `full_chain` arm applies the complete recorded chain.

Every arm therefore shares the same state, horizon, reward, dynamics, safety
layer, and terminal semantics. Only the number of consecutive recorded
recovery actions before Actor hand-back changes. The recorded chain is replayed
exactly; its reset observation and reward prefix must match the pinned shard
within tolerance.

## Outputs

For every state and commitment, record discounted return, cross-track change,
path-progress change, re-entry, collision, safety override, reset error, and
recorded-prefix reward error. Report:

- recovery advantage over the all-Actor baseline;
- fraction of positive advantages at each commitment;
- median advantage and per-scene positive fractions;
- each state's first positive commitment;
- monotonicity of the commitment-response curve.

## Frozen decision

`continuation_policy_mismatch` requires all of the following:

1. full-chain recovery advantage is positive in at least 0.75 of states;
2. the full-chain positive fraction exceeds the one-step fraction by at least
   0.20;
3. median full-chain advantage exceeds median one-step advantage;
4. at least 0.25 of states first become positive only after five or more
   committed recovery steps;
5. at least four of six scenes have a full-chain majority-positive result;
6. all resets, recorded prefixes, and numeric outputs pass integrity checks.

If this Gate passes, the next intervention must explicitly address the
Actor--Critic recovery coordination/option-continuation mismatch and must first
pass a Critic-only or imitation-initialization preregistration. L275 does not
authorize Actor training. If it fails, the next unique diagnosis is a
temporal-history direct-return cross-fit. L258, final Hairpin/S-Chicane/Infinity
maps, sealed seeds, and outcome-based selection are forbidden.
