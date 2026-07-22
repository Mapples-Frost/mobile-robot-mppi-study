# L284 Closed-Loop Roll-In Joint Anchor Paired SAC Gate

## Causal intervention

L283 confirmed that the residual recovery failure is jointly temporal rather
than a single velocity or steering component.  L284 changes one thing relative
to L281: the recovery half of the Actor anchor is replaced by geometric-teacher
labels queried on states visited by the three frozen L281 step-6k Actors.
Velocity and steering labels remain joint.  The source-distillation half is
copied byte-for-value from L279, and all SAC settings remain L281-identical.

## Frozen roll-in dataset

Use only the 72 L268 train recovery starts across the six L261 train scenes.
For each start, roll out all three L281 Actors for the accepted-chain horizon.
At every actual Actor-visited MuJoCo state, query the same closed-loop L267
geometric teacher and store the 69D observation plus the joint two-dimensional
teacher action.  Sample exactly 128 recovery rows per original chain, allocated
43/43/42 across the three roll-in Actors in sorted-seed order, yielding 9216
recovery rows.  Copy exactly 9216 L279 source-distillation rows.  Validation
and test chains never enter roll-in generation or Actor updates.

## Paired SAC Gate

Run paired seeds 20263311/12/13 from the same corresponding L276 3k Actor
initializations, for exactly 6000 Windows CUDA steps.  Actor remains frozen to
step 3000; thereafter every anchor batch is 128 roll-in recovery plus 128
source-distillation rows.  Recovery weights are [1,1], source weights [0,1],
and log-std anchoring is unchanged.  Reward, 69D observation, network,
Critics/alpha, replay semantics, optimizer, ICODE, MPPI, HSS, fusion, and
safety chain are frozen.

Evaluate once after all seeds on the same L268 held-out validation/test
recovery chains and the same three L277 validation scenes.  The Gate retains
all L281 engineering, safety, completion, CTE, goal-distance, and recovery
thresholds and additionally requires median held-out recovery return to improve
over paired L281 and at least four of six recovery scenes to be nonnegative.

No seed, checkpoint, chain, threshold, or scene may be selected after outcomes.
L258, final Hairpin/S-Chicane/Infinity, and sealed seeds are forbidden.  A
positive L284 Gate only authorizes preregistration of larger independent-seed
validation; a negative Gate stops expansion.
