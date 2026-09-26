# Prospective observable-context SAC development round

Prepared2026-09-08 after four40k-cycle training arms completed, while initial
development validation is in progress. No outcome of this new round exists.
The first round remains an archived exploratory comparator; no old result is
replaced. User requests continued experiments along joint K/H with ICODE.

## Purpose and changes

Test whether better common MPPI tuning and more stable learning recover useful
joint allocation. Apply all common backbone settings to every fixed and learned
method. Use the completed shared-backbone factorial and budget-check stages to
choose parameters. Nominal paired results remain visible, but the new learned
arms use ICODE as explicitly requested. ICODE is not assumed superior.

The actor observes50features: the original49 plus normalized remaining task time.
Innovation uses sensed state transitions and an applied-command estimate derived
from issued commands, measured command-readiness timestamps and fixed calibrated
delay40ms. Actual hidden actuator targets/queue are excluded. No scene ID, future
control, post-planning diagnostic or unknown plant parameter enters the actor.

Q and all physical task definitions remain unchanged. Reward uses the same
observed distance-progress, time and failure components as the first round,
with frozen positive compute price0.05 per measured second. This reduces the
initial round's price0.20, and is an explicit resource-preference change, not a
natural physical optimum claim. Report task Q without compute alongside all
reward components and original stage cost. Do not compare shaped return across
prices as if it had a common scale.

SAC settings: hidden128/128, gamma.995 per five-cycle decision, tau.005,
actor_lr1e-4, critic_lr3e-4, alpha_lr1e-4, initial_alpha.02, learned entropy,
two gradient updates per decision, batch128, replay100k. Replay sampling balances
the24family/speed groups using training labels only; no label reaches the actor.
Learning starts after500decisions. Raw tanh actions are saved unchanged.

Initial deterministic actor mean equals the shared fixed reference budget,
independent of observation. Hidden layers retain seeded initialization; final
layer weights zero, logstd=-.7 for each active coordinate. Exploration is
stochastic from the first decision and remains trainable. This is not behavior
cloning or a pretrained method. Apply the same corresponding initialization to
K-only,H-only and joint arms. No optional robust/BC modes are enabled.
For an extreme integer budget, use normalized mean+/-.97 within its rounding
bin to preserve that integer choice without nearly-zero tanh gradients.

Initial development arms: joint full, H-only, K-only, joint masked, one seed
9091201, each60,000physical cycles plus the complete final episode. The masked
arm still computes identical context and masks dynamics values/availability.
Every arm uses the same independent scenario-order RNG; episode environment
seed=9093000+episode_index. Resume includes optimizer,replay,Torch RNG,schedule.
Save every15,000cycles. All16checkpoints may be evaluated on new development
seeds9091202/9091203,24contexts. No final909030xseed is touched by selection.

## Selection and comparisons

Pre-execution amendment: source inspection identified a physical-domain mismatch
between the current9kgplant and ICODE's13kgpretraining plant. Follow
`icode_sac_domain_screen.md` before training. That amendment supersedes the
inactive-axis pair selection below: retain the shared planner settings selected
on9kg,screen both9kgand13kg,then choose the fixed pair within13kg. The new training
and all its comparators use13kgas designated before screen outcomes. Every
resolved spec records the complete plant override; this round is an in-domain
ICODE study, not an unknown-physics robustness claim.

Before launching training, write a resolved immutable spec for each arm. Choose
common planner configuration from the completed shared-backbone budget check:
among ICODE method cells with collision count no higher than the incumbent
K64/H24, rank by success descending, collisions ascending, Q ascending,
compute ascending. Include the incumbent if no tuned candidate qualifies.
The selected cell supplies both common parameters and fixed K/H reference.
This is exploratory selection, not noninferiority evidence.

Development evaluation must retain cheap fixed K16/H20, medium K64/H24, selected
fixed, high fixed K256/H24, all four learned arms and a simple causal progress
feedback comparator. Both nominal and ICODE fixed models use the selected common
parameters. Match scene geometry,seeds,measurement pipeline and safety. Time
context construction,history update,policy inference,mapping,MPPI,safety/readiness.
Optimization occurs offboard between steps and is reported separately from
deployment inference. Timing runs are serial.

Prospective process-hygiene amendment after initial validation completed: the
initial evaluator accumulated approximately5.9GBRSS/1.1GBswap at exit despite
closing environments. New round records per-episode RSS/swap and collects closed
cyclic objects outside the decision timer. Its validation uses one serial worker
process per complete paired family/speed/seed context, preserving the frozen
randomized schedule and all methods. Process startup/checkpoint loading is
excluded from online inference, as in the initial evaluator. This isolates
native allocation accumulation; monitor remaining warmup effects within each
context. Initial timing data remain archived with this memory limitation.

If joint training remains weak, separately version a curriculum using the
tested axis-transfer helper; count every source training interaction and replay
sample budget. Do not silently warm-start only the proposed arm and compare it
to lower-budget baselines. Curriculum is not part of this default round.

Any promising selected policy must later be replicated across initialization
seeds and tested on untouched paired seeds with uncertainty intervals before a
final benefit claim. Further development is allowed even if this round fails,
with a new recorded hypothesis and dataset split.
