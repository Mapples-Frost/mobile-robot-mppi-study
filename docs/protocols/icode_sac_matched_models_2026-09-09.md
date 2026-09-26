# Matched model and information training factorial

## Material Passport

Mode: prospective run. Status: protocol before new training or evaluation.
Question: under equal training budgets, does active ICODE prediction and access
to model-context features improve joint K/H allocation relative to nominal?
User authorizes continued development. All observations from earlier rounds are
development evidence; no hidden failure exclusion or final-test reuse is allowed.

## Design

Full 2x2 factorial: active planning model {ICODE, nominal} x dynamics context
{full, masked}. Train each combination from two new initializations 9096101 and
9096102, eight groups total. Every group receives 60,000 physical cycles plus
the complete final episode, the same 128/128 SAC, reward price .05, raw action
mapping, initial mean 128/32, logstd -.7, scene-balanced replay and 2 updates per
decision. Use the same corresponding architecture and no curriculum/warm-start.
In each initialization, all four treatments share the shuffled context RNG seed
and environment seed base. Resulting trajectories/episode lengths may diverge.
Training environment bases are 9290000 and 9390000, separated by 100,000.
At most 60,000 episodes per arm can occur before reaching 60,000 physical cycles,
so ranges do not overlap. Evaluation uses fresh 9490101--9490104 and keeps the
original reserved final seeds 9090301--9090304 sealed.

Common physics: 13kg calibrated plant from observable_specs/selection.json.
Common planner temperature2.5, noise[.2,.9], obstacle influence.7. Keep ideal
pose/twist and simulated scan; no hardware or noisy odometry claims. Every
treatment retains the frozen shadow ICODE context acquisition and its cost.
Mask only dynamics values/availability (17:33); all other inputs remain common.
Thus nominal-full is nominal planning with access to ICODE-derived context;
nominal-masked still pays acquisition cost. Do not claim the cheapest pure
nominal implementation has been benchmarked by this factorial.

## Execution and evaluation

Training order is seeded using 9096000 and frozen before execution. Serial
CPU timing, BLAS/Torch1thread, complete-episode recovery with stored RNG/replay.
Record every raw trace, resolved config, source/checkpoint hash and RSS/swap.
Use only the final checkpoint >=60k cycles for each group, no outcome selection.
Re-evaluate all eight learned methods concurrently in the same serial randomized
block schedule along with nominal/ICODE fixed pairs128/27,160/29,256/24. Those
pairs were selected earlier on development and are fixed here, not retuned.
14 methods x12families x2speeds x4environment seeds =1,344 paired episodes.
One complete paired context per child process; order RNG9490000.

## Analysis and interpretation

Primary outputs are success/collision counts, taskQ, total measured computation,
deadline misses and original stage cost, each initialization separately. Planned
contrasts: ICODE minus nominal within each mask/initialization; full minus masked
within each model/initialization; information-by-model difference of differences;
and every learned method versus each same-model fixed candidate. Keep the
complete family/speed table, including failures. Do not pick a winning seed.
Use descriptive family/environment-seed paired bootstrap (speeds kept together)
and conservative paired binary intervals. Only two training seeds provide weak
initialization uncertainty; do not pool repeated baselines as independent runs.
No best-method p-value, significance/noninferiority or real-world claim.

Mechanism criterion for further development: seek consistent directions across
initializations and strong fixed baselines, then inspect failure contexts.
Mixed/negative results remain recorded. Natural costs include explicit compute
price .05; model-specific compute affects the reward and observed trajectory,
so this is a system comparison under equal resource preference, not a pure
prediction-accuracy isolation. The prior model-swap interventions complement it.
