# Teacher-Replay SAC Degradation: Frozen Diagnosis

Registered machine-readable design: `results/sac_teacher_cause/protocol.json`.
The diagnosis follows observation of the previous study's holdout result and is
exploratory. It does not replace that result or tune a new policy on its holdout.

## Questions and Discriminating Tests

1. Does degradation already exist after offline pretraining, or emerge online?
   Evaluate all three pretrained, 5k, 10k and 15k teacher checkpoints on identical
   four new scenes. No best-checkpoint selection or new optimizer updates.
2. Is deterministic action extraction the main problem? Compare frozen actor
   means with two reproducible draws from the original squashed Gaussian policy.
3. Are executed very short horizons essential to the observed loss? Replace only
   the final rounded horizon by max(H,5). This changes both physical behavior and
   the compute proxy; it does not isolate the solver implementation alone.
4. Is directly extracting the highest learned Q1 sufficient? Execute its maximum
   on the fixed grid [1,2,3,5,10,20,25,30,40,50], for all six final models.
5. Are the learned candidate advantages consistent with realized soft returns?
   At four common rule-prefix states in new scene0, execute the actor, Q1-grid,
   or rule first action, then the corresponding model's current stochastic actor
   until actual episode termination. Keep two common-random-number repeats.
   Include continuation entropy, original reward scaling and no approximate tail.

The fifth test conditions on one realized exogenous reference sequence. A policy
still sees only the original 50-step preview. These are conditional Monte Carlo
estimates, not exact population soft-Q targets. Two draws quantify only limited
simulation variability; report their paired differences, including disagreement.

## Controls and Accounting

- New scene bank seed26091991, four scenes, fixed before intervention outcomes.
- All three training seeds, identical frozen terminal, original reward and model
  hashes; no modifications to the previous experiment's source or results.
- FixedH25 and causal H5/H30 rule as common controls. Models are not selected.
- Six final models times five modes, nine intermediate teacher checkpoints and
  two controls: 41 conditions, 164 episodes, at most98400 evaluation transitions.
- Six final models times four anchors times three candidate actions times two
  repeats: 144 branches. Exact rule-prefix reconstruction plus continuation uses
  at most86400 transitions. Reset warmups and the20-step smoke are separate.
- At most two active MPC workers. Every failure and partial outcome is retained.
- Original teacher data are profiled read-only: every tenth recorded teacher
  transition gives600 fixed samples per seed. Compare checkpoint Bellman residual,
  candidate value span, actor distribution and recorded-action value.

Interpretation distinguishes physical outcome interventions, training-stage
associations, replay support and value calibration. No training-factor ablation
has yet isolated actor updates from critic updates or data-distribution changes.
