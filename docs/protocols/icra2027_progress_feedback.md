# Prospective progress-feedback exploration

2026-09-08. Declared after the first 1392 episodes and complete 768-episode grid;
the 480-episode kernel holdout is running and its final outcomes are not used here.
New motivation: static scene-winner selection did not transfer reliably, and the
first kNN lost success despite compute savings. Test feedback that observes actual
task progress, as well as obstacle proximity, on new seeds. This is an exploratory
method round, not a confirmatory claim selected from the preceding successes.

Six methods: fixed K256/H24 (full-grid training best), fixed K64/H24 (earlier
training best), previous scan/ESS rule, and matched K-only/H-only/KH progress
feedback. All 12 families, two speeds, four fresh perturbed seeds 9080601–9080604:
576 complete episodes. Same nominal controller, physical command-readiness,
100ms cycle, safety, Q and recorded metrics. Randomized method order within each
family/speed/seed; serial execution. Do not retune after outcomes.

Feedback rule at each timed policy call uses only current observed goal distance,
current scan-sector minimum, and its own past goal-distance observations.
Low level K16/H8 in open space; switch to K64/H24 when clearance <1.8m; leave
that state only when clearance >2.2m. If goal-distance reduction over the last
15 intervals (1.5s) is <0.08m while goal distance >0.5m, use K256/H24 for the
next 20 decisions (2s), then reevaluate. No training, scene ID, truth, future
obstacle trajectory, or current-plan diagnostics. It is a deliberately simple
closed-loop comparator, not SAC. K-only always H24; H-only always K256. All other
rules and sensed information matched. Starting from zero motion is permitted;
the 1.5s history prevents an immediate false stagnation trigger at reset.

Analyze success, collisions, Q, duration, final distance, compute total, mean/P95
latency and budget occupancy. Report K/H/KH relative to fixed and to each other,
with paired descriptive intervals and family-cluster sensitivity; no p-value
search. Margins remain 2pp success / 5% Q / no extra collision, with desired 20%
compute saving, and uncertainty must be reported. Keep all negative contexts.
The rule cannot guarantee recovery from a controller local minimum or distinguish
temporary safe yielding from stagnation; observe and report this failure mode.
