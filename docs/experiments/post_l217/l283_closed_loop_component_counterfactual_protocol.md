# L283 Closed-Loop Recovery Component Counterfactual Diagnosis

## Question

L282 identified a coupled sequence bottleneck, but its teacher component was
the action recorded on the original teacher trajectory.  Once a hybrid arm
deviates, that recorded component may be inappropriate for the new state.
L283 asks whether the apparent coupling persists when the same frozen
geometric teacher recomputes its action at every actual hybrid-rollout state.

## Frozen design

L283 is evaluation-only.  It reuses all 18 L268 test recovery chains, the
three L281 step-6k Actors, the exact L267 teacher configuration, and the four
L282 arms.  For every reset and every arm, construct a fresh teacher, set its
tracker to the chain's frozen starting progress, and query it from the actual
MuJoCo pose at every step.  The chain-specific frozen velocity/angular
perturbation is then applied and clipped exactly as in L267.

The `teacher_both` arm must reproduce every recorded teacher action and reward
within 1e-5.  Expected coverage remains exactly 3 seeds x 18 chains x 4 arms =
216 rollouts.  Actor observations always come from the actual hybrid state.

## Frozen attribution

Use the same scene-then-seed coverage rule as L282: a component is attributed
when its closed-loop hybrid recovers at least 50% of the positive full-teacher
return gap in at least four of six scenes for at least two of three seeds.
Both, coupled, and unresolved decisions retain their L282 definitions.

If a single component is attributed, only its matching intervention may be
preregistered.  If coupling remains, only a joint closed-loop recovery
roll-in/imitation intervention may be preregistered.  L283 never authorizes
training or final Hairpin/S-Chicane/Infinity evaluation directly.  L258,
sealed seeds, threshold changes, state selection, and deletion of failures are
forbidden.
