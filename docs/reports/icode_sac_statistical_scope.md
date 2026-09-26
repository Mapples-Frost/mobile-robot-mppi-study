# Initial ICODE/SAC evidence: statistical scope

## Material Passport

Mode: validate. Verification status: ANALYZED, with separately passed stored
trace/source arithmetic audits. No independent physical replay has yet verified
the complete study. Data: `research_artifacts/icode_sac_compute_2026-09-08/`.
Prepared2026-09-09 during prospective13kgdomain-study preparation.

The first48paired evaluations per method cover12scene families,2speed levels
and2environment seeds, with one SAC initialization per arm. These are development
data, used to choose checkpoints and plan additional experiments. Reported95%
intervals are descriptive; they are neither simultaneous intervals nor adjusted
for checkpoint/model selection. No confirmatory significance or noninferiority
decision is made.

## Scope review: 11/11 items checked

| Issue | Evidence and interpretation |
|---|---|
| Simpson reversal | Every method uses the same48contexts. Selected joint full is worse than fixed ICODE128/32in both speed strata: fast17vs21successes,Q2.41vs1.06;moderate13vs15,Q3.31vs2.87. No speed-aggregation reversal for this contrast. Family heterogeneity remains visible in context summaries. |
| Ecological inference | Episode-context oracle choices do not establish cycle-level reliability effects or benefits of within-episode switching. No such inference is made. |
| Selection-induced association | Checkpoints and shared parameters are outcome-selected on development data. Their apparent benefits need independent initialization and held-out confirmation. |
| Collider conditioning | Main outcome tables include all episodes, not only successes. Collision-qualified tuning selection conditions on an outcome and is a development selection rule, not evidence of causal safety equivalence. |
| Base-rate neglect | Success and collision counts have explicit denominators24or48; predictive diagnostic probabilities/PPV are not claimed. Zero collisions is not zero collision risk. |
| Regression to mean | Candidate tuning was selected on seed9091101and evaluated against the incumbent on seed9091102. This helps diagnose repeatability, but subsequent choice on the second seed makes it development data too. |
| Survivorship | Closed-loop tables retain success,collision andtimeout episodes. Prediction analysis requires complete32cyclewindows; short episodes and late terminal portions are outside that prediction estimand. New domain prediction explicitly lists short-episode exclusions. |
| Multiple comparisons | All16SACcheckpoints and all fixed candidates are retained. Many contrasts are exploratory; no best-checkpoint p-value is presented as confirmatory. |
| Researcher choices | Protocols/source snapshots retain original and prospective rounds. Physics,compute price,features,optimization andprocess hygiene all change in the new round, so between-round differences cannot identify the effect of a single change. |
| Correlation versus cause | Innovation/disagreement correlations with later prediction error are descriptive and temporally dependent. Masked closed-loop training is a separate intervention; an offline masking probe is distribution-shift sensitivity, not proof of causal information value. |
| Reverse causality | Online features are built before current planning; past planner diagnostics are allowed. Nevertheless past control affects current model-error indicators, so their observational association with chosen H does not identify a one-way reliability-to-budget mechanism. |

The first evaluator accumulated about5.9GBresident memory and1.1GBswap at
process exit. Its treatment order was randomized within paired contexts, but
initial measured-latency comparisons retain this resource limitation. New
validation uses serial process isolation by complete paired context and records
RSS/swap. Current sensors remain ideal simulated pose/twist; hardware and noisy
odometry generalization are untested.

The conservative binary discordance interval avoids the false zero-width
bootstrap interval when both methods have identical observed outcomes. It still
assumes independent paired episodes. The two-way family/seed bootstrap keeps
speed pairs together, but only two seeds provide weak uncertainty information.
Training-control cycles and overlapping prediction windows are not independent
replicates. The next substantive evidence must come from the actual prospective
experiments, not stronger wording of these intervals.
