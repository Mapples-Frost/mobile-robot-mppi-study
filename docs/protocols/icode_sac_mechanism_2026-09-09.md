# Prospective mechanism interventions, 2026-09-09

Prepared while independent-initialization training is finishing, before its
validation outcomes. User requests continued experiments on joint K/H and ICODE.
No final 909030x seeds are used. This is exploratory, not a paper acceptance gate.

## Questions and interventions

1. Does changing each coordinate within an episode improve task/resource outcomes?
   Evaluate the full-context joint final checkpoint from each initialization
   9091301/9091302 normally and with K held, H held, or both held. The held value
   is selected at episode start from the current observation, not hindsight.
   Continue computing the original actor every five cycles, charging its actual
   overhead in all treatments. This isolates action suppression under matched
   policy computation, not the most efficient deployable static controller.
2. How much does the active ICODE predictor contribute conditional on a frozen
   policy? Evaluate both full and independently trained masked joint final
   checkpoints using ICODE and nominal planning. The shadow ICODE context and
   sensing pipeline remain common. Nominal transfer has not been retrained:
   this identifies conditional model replacement, not superiority against an
   equally trained nominal SAC controller. A separately trained nominal arm is
   still required if this becomes a central paper claim.

All eight learned policies use the final >=60k-cycle checkpoint rule from the
independent round. Additional K/H holding interventions apply only to the two
full-context ICODE policies (six treatments). Fixed comparators use both models
at 256/24 and at 160/29 and 128/27: the latter two are rounded episode-equal mean
budgets from the previously selected observable full/masked development policies,
respectively (159.128/28.630 and 131.752/27.006). They are strong retrospective
development choices, now frozen before this fresh evaluation. Six fixed methods
give 20 methods total.

## Design and outputs

Use all 12 families and two speeds, paired fresh environment seeds 9195201 and
9195202, randomized order using schedule RNG 9195300. Total 960 episodes. Match
13kg plant, planner tuning, sensing, safety, task Q, actor hold and physical
readiness accounting. A single worker executes each complete context; timing
runs are serial. Start only after independent validation and its reports exit.
Archive raw traces, episode summaries, resolved configs, checkpoints hashes,
source snapshots, process RSS/swap, seeded schedule and training/evaluation seed
intersection audit. Failed stages remain visible and stop the queue.

Report every method and initialization, task Q and components, success/collision,
stage cost, measured computation, and deadline misses. Paired contrasts are full
versus each hold intervention and masked per initialization, and ICODE versus
nominal for each frozen actor and fixed pair. Use existing descriptive two-way
family/environment-seed bootstrap plus conservative binary discordance intervals.
Do not select an initialization, omit unsuccessful families, infer significance
from these many uncorrected exploratory contrasts, or count shared fixed runs
multiple times as independent replication. No positive result is guaranteed.
