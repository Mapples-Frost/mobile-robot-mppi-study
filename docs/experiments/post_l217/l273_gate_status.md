# L273 Gate Status

- Status: **diagnostic negative for capacity and action encoding; Actor training
  remains unauthorized**.
- Integrity: 3 paired seeds x 4 arms completed, 24/24 checkpoints present, all
  values finite.
- The current 256x256 raw-action Critic fit L263 training action rankings
  (`0.9849` mean Spearman) and generalized to L263 validation states
  (`0.6675`), but failed on independent L268 recovery states (`0.0593`
  Spearman; `0.4722` recovery-vs-forward accuracy).
- Wider networks and fixed degree-3 action features did not improve external
  recovery ranking and caused scene regressions.
- Conclusion: ordinary network capacity and raw 2D action encoding are not the
  primary bottleneck. The remaining ambiguity is recovery-state distribution
  coverage versus missing state information/target generalization.
- Next authorized action: a state-level cross-fitted, Critic-only diagnosis of
  recovery data coverage and privileged path-local information. No Actor,
  reward, controller, or safety change is authorized.

Raw output is retained at
`results/research_platform/rl/l273_critic_capacity_action_representation_diagnosis`.

