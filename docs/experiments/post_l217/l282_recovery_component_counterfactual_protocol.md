# L282 Recovery Action-Component Counterfactual Diagnosis

## Question

L281 restored independent validation performance but failed the recovery
scene-return coverage threshold even though teacher-action RMSE improved.  L282
asks whether the remaining sequence-level loss is caused primarily by linear
velocity, angular velocity, or their temporal coupling.

## Frozen design

L282 performs no optimizer step.  It uses all 18 L268 **test** recovery chains
from all six L261 training scenes and the frozen L281 step-6k checkpoints for
all three paired seeds.  Validation and train recovery chains are excluded.
For each chain, reset the exact saved MuJoCo state and execute the accepted
chain horizon under four deterministic arms:

1. `teacher_both`: recorded recovery velocity and steering;
2. `actor_both`: current Actor velocity and steering;
3. `teacher_v_actor_omega`: recorded velocity with Actor steering;
4. `actor_v_teacher_omega`: Actor velocity with recorded steering.

At every step the Actor observes the actual hybrid-rollout state.  The same
reward, termination, physics, normalizer, and reset tolerance are used for all
arms.  Report discounted return, reentry, collision, boundary violation,
progress, CTE, and goal-distance change.  Expected coverage is exactly
3 seeds × 18 chains × 4 arms = 216 rollouts.

## Frozen attribution

For states where `teacher_both` outperforms `actor_both`, define the recovered
loss fraction of each hybrid as its return gain over `actor_both`, divided by
the teacher-versus-Actor return gap.  Values are reported without clipping.

- `velocity_component_bottleneck`: teacher velocity recovers at least 50% of
  the loss in at least four of six scenes for at least two of three seeds,
  while teacher steering does not satisfy that rule;
- `steering_component_bottleneck`: the symmetric steering result;
- `both_components_contribute`: both hybrids satisfy the rule;
- `coupled_sequence_bottleneck`: neither hybrid satisfies the rule but the
  teacher gap is positive in at least four scenes for at least two seeds;
- otherwise `component_attribution_unresolved`.

Reset error must not exceed 1e-6; all outputs must be finite; recorded teacher
prefix reward must reproduce within 1e-5.  No threshold or subset may be
changed after results.  L258, sealed seeds, and final
Hairpin/S-Chicane/Infinity are forbidden.  L282 never authorizes Actor
training or final-map evaluation directly; it only authorizes preregistration
of the matching minimal intervention.
