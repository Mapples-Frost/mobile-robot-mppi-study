# Matched K/H extension, frozen before its first outcome

2026-09-08. Motivation: the original eight pairs cannot isolate the two axes.
First geometry discovery (192 episodes) was already observed; the 576-episode
nominal confirmation was in progress when this design was declared. This is a
new exploratory extension, not a preregistered external benchmark.

Train a complete Cartesian grid K={16,64,128,256}, H={8,16,24,40}, all twelve
families, both speeds, perturbed seeds 9080401/9080402: 768 episodes.
Preserve the same plant, controller, task Q, safety and actual command-readiness
timing as campaign.py. All sixteen treatments share the same seeded context.
No residual or temperature selection in this nominal comparison.

Fit kernel arm-outcome regression from initial goal/scan features and speed cap.
Use five nearest standardized feature vectors, inverse distance weights with
0.05 floor; predict success/collision/Q/compute for each of sixteen actions.
No simulator family ID, geometry, current-plan statistics, or test outcomes.
Pick cheapest predicted action within 2 percentage points of predicted maximum
success, no extra predicted collision risk, and <=5% predicted Q regret relative
to the lexicographic success/collision/Q choice. These are point-estimate
selection rules, not statistical guarantees.

Compare fixed task-optimal, fixed efficient, learned K-only, H-only and K/H on
fresh seeds 9080501–9080504 in every family/speed (480 episodes). K-only fixes H
at the development best-global pair; H-only fixes its K. They share the same
features, predictor, training episodes, action grid, tolerances and timing.
Budget chosen once at the initial observation; this tests episode-level causal
allocation and does not claim within-episode adaptation or SAC.
Block by paired family/speed/seed and randomize method order inside each block.
Record all contexts including failures. No holdout-driven selector retuning.

Report success, collisions, Q, duration, remaining distance, stage cost, compute
total and per-cycle latency. For comparisons use paired differences with
descriptive bootstrap intervals, preserving treatment pairs; additionally
cluster by family to expose dependence on the twelve chosen geometry families.
No sequential p-value claims or significance-based stopping. NI margins are 2pp
success / 5% Q and no extra collisions; point-estimate feasibility does not
establish non-inferiority when uncertainty crosses a margin. Actual measured
compute savings target 20%, reported separately from task performance. All
figures and reported totals come from saved episode summaries and raw traces.
