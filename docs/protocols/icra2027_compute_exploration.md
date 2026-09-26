# ICRA 2027 adaptive MPPI computation: broad development campaign

Date 2026-09-08. New user authorization: pursue many methods and new scenes for
the same K/H allocation idea. Parent ec56db89349aa4a802d4a468e5e1453fae01e89e.
All prior NO-GO reports and artifacts remain immutable. This is exploratory
development, not a revision of v2, and no search winner is a confirmatory claim.

## Questions and useful alternatives

Q1: Can context-dependent K/H improve task quality across heterogeneous geometry,
motion and model conditions? Q2: Can it save substantial actual computation at
non-inferior task quality? Q3: Do physical deadline/resource limitations change
the answer? Q4: Does causal model evidence add predictive value after scene and
speed are controlled? A null in one question does not stop all other exploration.
SAC is one candidate optimizer; a simple causal selector is an essential comparator.
No promise of acceptance, novelty or positive results substitutes for experiments.

## Stage 1 broad geometry screen

Twelve parameterized point-goal families, two speed caps (.35/.65 m/s), eight
K/H pairs [(16,8),(16,20),(64,12),(64,24),(128,16),(128,32),(256,20),(256,40)],
one seed 9080101: 192 complete episodes. Static obstacle bank is fixed before
outcomes; families include open, offset block, small/large blocker, two staggered
cylinders, open/wide doorway, offset wall, sparse slalom, dynamic crossing and
mixed open+obstacle. Exact geometry recorded by scenes.py/config YAML.
Use a 4.5 m goal along world x, nominal model .18/.12 s, existing torque PI plant,
0.04 s actuator transport, current ground-truth sensor isolation, no future truth.
Maximum 300 cycles / 30 s, tolerance .30 m, safety prefix >=8. Baseline v2 planner
temperature2.5/noise[.12,.5]/original obstacle objective unchanged.
Actual command-readiness latency physically advances the plant, no compute price.

Record success/collision/timeout, duration, remaining goal distance, geometric
clearance, trajectory, stage objective, command, actual CPU latency and physical
latency, ESS/entropy/U_MC, previous-only causal context. Compact raw cycle traces
retain sufficient fields for reproducibility; omit large rollout arrays during
broad screen, keep occasional planned sequences/trajectories.

Task-quality score for NEW exploratory comparison (not v2 rescoring):
Q = duration / max_duration + 2 * final_distance / initial_distance
    + 3 * timeout + 10 * collision.
Always report its components separately; safety/success are primary outcomes.
This avoids the old summed laser-point obstacle stage cost dominating comparisons
between fundamentally different scenes. Raw v2-consistent stage objective remains
recorded, so neither metric is silently substituted. No K/H or latency enters Q.

## Stage 2 baseline and mechanism robustness

Screen alternative backbones on a declared subset before their outcomes: temperature
10 (tests importance-weight collapse), residual frozen checkpoint (not assumed
more reliable), and physical resource envelope emulation. All alternative results
are labelled separately; no controller gets an exclusive stronger base model.
Resource emulation multiplies measured planning readiness by an explicit factor;
it is a simulation study, not measured runtime on a slower embedded machine.
Prefer exploring resource efficiency on actual timing first; no fictitious delay
claim from multiplying a number without advancing physics.

## Stage 3 repeat and allocation confirmation

Confirm the complete nominal geometry grid using 3 fresh seeds 9080201–9080203,
not just successful cells. Randomize static cylinder/wall offsets by modest
per-family perturbations shared across budget treatments; mirror y with seed.
All-failure contexts remain in full-bank summaries. A clearly labelled serviceable
subset may be analyzed but cannot replace full-bank results.
Select budgets on development data, evaluate on unseen seeds/geometries. Cross-evaluate
best fixed, task-only context selector, and minimum-compute selector constrained
to no extra collisions, success non-inferiority and <=5% task-quality regret on
selection data. Report held-out constraints, not only training feasibility.
Evaluate joint K/H against K-only and H-only allocation on the same information
and common backbone. Inspect full task/compute Pareto sets.

## Learning and generalization

If geometry-dependent efficient budgets appear learnable, train causal simple
feature selectors and compare on untouched geometry/seed split. No simulator scene
ID in deployable policy; scene-label oracle is diagnostic only. At decision t only
sensed goal/scan/motion and previous planner statistics are legal. If a stateful
SAC approach is justified, freeze its training schedule separately, use continuous
unrounded replay actions and integer execution, compare common SAC-H/K/KH and
fixed/heuristic. Do not call supervised classification SAC.
Exploratory search can continue after a null by recording a new round; never
delete failures, cherry-pick test seeds, redefine completed metrics, or call an
exploration-selected scene a held-out benchmark.

## Execution and archival

Serial timing experiments, CPU/BLAS/Torch threads1. Use resumable jobs with frozen
schedule and per-episode unique keys; finished traces must never be overwritten.
Each round creates an immutable manifest (source/config hashes, commit, seeds,
hardware, settings), logs failures and summaries. Timed tasks must not overlap
other experiments. Local stage commits only; no push.
Deliver an evidence-ranked report with all completed rounds, promising directions,
negative findings, resource savings and remaining publication gaps. An ICRA claim
requires independent holdouts, strong common-backbone baselines, credible timing,
and eventually hardware evidence; exploratory positives are candidates for that work.
