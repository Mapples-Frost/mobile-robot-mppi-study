# Gate 3A2 preregistration: graded independent reliability stress test

Date: 2026-07-18

## Why this addendum exists

Gate 3A was executed exactly as frozen in
`174_gate3_reliability_hss_prereg_2026-07-18.md` and **did not pass**. The
validation and test splits showed strong negative episode-level association
between authority and rollout error, but the four-episode unseen split
contained only the combined severe OOD domain. All four episodes were assigned
low authority, so that split could not test ordering between two or more
discrete authority levels.

This failure is retained. Gate 3A2 does not change the selected thresholds,
ensemble, Actor, critic, or offline dataset. It repairs the evaluation design
by collecting a new mixed-severity stress set with enough independent episodes
to ask whether the already-frozen score ranks light, moderate and severe model
error.

## Locked data design

- source environment snapshot: frozen L175 Actor run;
- task families: clean single obstacle and narrow corridor;
- physics domains: nominal, high mass, low friction, long delay and combined
  unseen;
- three independent episodes per scene-domain cell;
- 30 episodes total, maximum 120 steps each;
- new seed base `22835000`;
- frozen deterministic Actor with the same 0.03 action noise and 0.05 random
  action probability used by L183;
- no model training, threshold search or online adaptation.

Episode is the independent unit. Overlapping ten-step windows are
within-episode observations only.

## Locked method

The three L192 value-aligned ICODE members, L175 Actor normalizer and the exact
candidate selected on L183 validation in L193 are reused without modification.
The protected Gate 3A2 data are collected only after this document and its
runner are committed.

The offline Actor-support term uses recorded post-transition observations as a
documented proxy. Closed-loop runtime will use hypothetical observations along
the Actor mean rollout. No simulator domain label is available to the score;
domain is used only after execution for stratified analysis.

## Gate

Gate 3A2 passes only if all conditions hold:

1. episode-level Spearman association between authority and ten-step rollout
   error is at most -0.30;
2. at least two authority bins are occupied, with at least two independent
   episodes in every occupied bin;
3. low authority contains at least two episodes and non-low authority contains
   at least two episodes;
4. mean low-authority error is greater than mean non-low-authority error;
5. combined-unseen mean authority is lower than nominal mean authority;
6. combined-unseen mean rollout error is greater than nominal mean error.

No p-value from overlapping windows will be reported. Failure again stops
adaptive HSS closed-loop work. Passage permits only the preregistered
development seeds 32--34; confirmation seeds 35--39 remain sealed.

