# Independent physical K/H study v2 — 2026-09-07

New protocol / new scientific question. Parent commit 0c6dfdab49ba18921401b9b9141684fa96a6b1e0.
The v1 physical_tradeoff_2026-09-07 results, scoring, thresholds and NO-GO are immutable.
The authoritative user request is archived in the v2 artifact root. No push.

## Frozen before closed-loop outcomes

- Question: does real computation delay, finite sampling and finite foresight create
  context-dependent task-optimal K/H with enough cross-evaluated headroom to justify SAC?
- Nominal dynamic-unicycle rollout (.18 s velocity, .12 s yaw constants), RK4;
  no checkpoint used. This isolates the nominal controller already established in the
  benchmark without assuming ICODE is better in every regime. Hash the dynamics source
  and all resolved configurations. Do not switch model after viewing outcomes.
- Existing benchmark geometry: Easy clean_goal_3_3 and Medium clean_single_obstacle
  (cylinder at [1.5,1.5], radius .28 m, on the start-to-goal diagonal), goal [3,3],
  start [0,0,0,0,0], tolerance .30 m. The existing Medium requires an early detour;
  no new positive-control geometry. 360 cycles of .1 s; terminate on collision/success.
- Torque PI MuJoCo plant and existing .04 s actuator delay retained. Ground-truth
  current pose/twist supplied as explicitly isolated sensors; no future truth access.
  Existing zero-noise mechanism sensors retained. Seeds vary planner sampling; they
  are paired environment/planner seed labels, NOT independent randomized scenes.
- Low/high speed envelopes .25/.65 m/s; yaw cap 1.25 rad/s, slew 1.1/3.5.
  Same goal-warm-start prior, temperature 2.5, noise [.12,.50], safety prefix 8.
- Scene/planner context only, no ICODE ensemble or reliability checkpoint overhead.
- Measured decision latency includes context, budget selection/mapping, optional
  compensation, planner, safety arbitration and controller notification. It ends
  immediately before plant stepping. Simulation, sensor synthesis and retrospective
  diagnostics are outside this boundary. Wrapper advances real plant under old command
  during latency; tau>=Tc drops current command; no sleep or clipping of logged tau.
- Primary J = sum of executed running cost + remaining-step collision failure cost;
  lambda_compute=0. Actual truth pose, observed world obstacle representation, original
  MPPI running coefficients and time-weighted actuator effort. No terminal/potential
  shaping or added stale penalty. Report completion and distance alongside J.
- Offline H-step model error: propagate nominal model from sensed state under the
  actually applied actuator command segments, compare with future truth after H cycles.
  This isolates model error under executed controls, not divergence caused by replanning.
  Also retain chosen control sequences/trajectories and factual trajectory.
- Serial CPU, Torch/BLAS threads=1. Randomize treatment order with recorded seed.

## A: H before everything else

Before outcomes, profile K={16,64,128,256}, H={8,12,16,20,30,40} on initial states
of both scenes (high cap), 12 calls/cell after warmup, serial randomized schedule.
K128 is admitted if all its cells have P95<80 ms, P99<100 ms. Prior v1 repeated
sampling evidence supports K128 as the middle choice (K256 did not improve aggregate
first-action variance over K128). If not admitted, stop for an engineering revision,
do not pick K using closed-loop outcomes. No K384/512 extension in this v2 grid.

Once admitted, freeze K128 and run six H values in all four scene/speed contexts
with seed 7092801. Effective H sensitivity means within a context >=5% spread in J
AND either >=1 s completion-duration spread, >=.15 m final-distance spread, or a
success/collision difference. A single run is screening only, not a physical optimum.
If no context passes, STOP. If any passes, add seeds 7092802,7092803,7092804.
Confirm the screened best-versus-worst H pair (fixed per context from seed1): same
J ordering in >=3/4 seeds and >=5% mean difference, with the corresponding task
outcome signal. Report all pairs and uncertainty, not just the winning context.
If none confirms, STOP. Classify short-H deficiency, long-H downside, monotonicity,
and flatness separately; a price cannot manufacture a natural interior optimum.

## B: speed x delay

After A confirms, run delays {0,.02,.05,.08} seconds with both speed envelopes and
four paired seeds, same scenes/start/reference. Two distinct estimands:
(1) closed-loop fixed K128/Href, same planner algorithm/noise seeds (commands can
respond to changed state); (2) exact normalized command replay, using the zero-delay
high-speed controller command profile normalized by [.65,1.25], replayed for both
speed caps and all delays with no policy/safety recomputation. Replay is an isolation
diagnostic, not the safety-qualified closed-loop result; collision still terminates.
Use a common maximum-duration reference command bank, padding terminal commands with
zero, and log path deviation from each speed's own zero-delay reference.
Meaningful task effect: >=5% paired J increase and >=1 s duration or >=.15 m final
distance change, or success/collision degradation; high-speed interaction compares
within-speed normalized treatment changes. Drift alone does not qualify.

## C then D

Href for K sweep: H minimizing four-context mean J in confirmed A; ties within 1%
choose smaller H. C compares compensation off/on at K128 and all six H, same four
contexts/four seeds. Expected delay is the frozen initial profile median, pooled
by K/H across scene (a K/H-only lookup). Planning never sees the current measured tau.
Compensation predicts command-ready state; the planner retains .04 s actuator-delay
semantics. Evaluate whether high-H disadvantage shrinks, disappears or remains.
D runs K={16,64,128,256} at Href, same contexts/seeds, no compensation primary.
Use passive U_MC, ESS, command variance, latency, staleness; supplement with 32 independent
sampling seeds at three preregistered progress checkpoints (0,1,2 m from start-to-goal
projection, first reached or mark unavailable) from K128/Href trajectories.
K saturation and higher compute alone are not physical degradation.

## E and oracle; no premature learning

E requires confirmed useful A sensitivity plus D task sensitivity/sampling-quality
signal. A/B/D mechanisms must be interpretable, not all treatment failures.
The request's final stopping sequence explicitly allows dropping staleness as a
core claim when B is null; therefore B null narrows the claim but does not alone
block a landscape supported by A/D. It does not authorize a delay-based SAC claim.
No confirmed natural long-H downside means at most CONDITIONAL GO to inspect the
landscape, never a claim of a proven physical interior optimum.

Before E, freeze four H from A (always min/max; remaining two nearest 1/3 and 2/3
indices of the sorted candidate list, i.e. 12 and 20), all four K; positive price
lambda=.1 per tau/Tc (small compared with initial goal stage cost18) fixed now.
Four paired repeat seeds; compute zero-price and priced J separately. Select global
and per-context optima on three seeds and evaluate on held-out fourth, rotate folds.
Report paired held-out difference, bootstrap over seed blocks, selection frequencies,
noise and expected actor/context/switching overhead. These labels are an offline
selection benchmark, not an online optimal dynamic policy upper bound.
Before computing oracle, freeze net GO threshold=max(10%,2*development relative
repeat-noise estimate), with exact development estimate/formula recorded separately.
Require positive held-out benefit in >=3/4 folds, bootstrap 95% lower bound>0,
and stable context-different choices. No context optimum or insufficient headroom => STOP.

Only passing A/E/oracle authorizes SAC T0/T1/T2/T3 as specified in the archived user
request. Freeze a separate SAC training protocol before implementation/training.
No PPO substitution; compare common-backbone Bohn-style SAC-H, fixed, oracle and
fitted interpretable heuristic. Reliability remains an independent later extension.

## Archive and reporting

Each stage writes protocol, commit/source/config hashes, hardware, seeds/order,
cycle-level raw JSONL.gz, episode summaries, figures and a decision. Outcomes never
overwrite the freeze. Save failed attempts and amendments. Run targeted adapter
tests and artifact recomputation, stage only v2 files, local commits.
Final docs/reports/physical_tradeoff_v2_final.md answers all 15 questions and marks
unrun phases/figures explicitly. GO/CONDITIONAL GO/NO-GO must follow observed gates.
