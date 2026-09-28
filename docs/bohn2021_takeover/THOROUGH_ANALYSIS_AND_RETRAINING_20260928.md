# User-authorized research steering

Recorded UTC: 2026-09-28T10:17:54.560545+00:00

## Priority user steering: thorough causal analysis and retraining when warranted

User instruction (2026-09-28): fully analyze the causes, and retrain when necessary. This explicitly authorizes bounded retraining within the existing scientific scope without further confirmation.

After the current bounded fixed-H opportunity probe completes, inspect its raw outcomes alongside existing training records. Produce a concise evidence table separating verified findings, competing hypotheses, missing evidence, and the experiment that distinguishes them. Cover scenario-dependent horizon opportunity; current policy/search-class limitations and state coverage; training budget/convergence/exploration/objective scaling; terminal-value accuracy and horizon mismatch; transition continuation costs; solver and policy overhead and timing noise. Episode-level fixed-H differences are suggestive, not proof of state-dependent switching benefit: use controlled continuation comparisons where needed.

Choose the most informative bounded intervention. If evidence points to insufficient learning, terminal-value bias, poor representation, or an inadequate policy class, actually run a targeted training/value-refit ablation after freezing its hypothesis, budget, data and comparison criteria. Do not substitute repeated metadata audits or reselection of the same checkpoints for needed retraining. If retraining is deferred, record the specific evidence and the concrete next experiment that will resolve that decision; do not indefinitely defer it with more summaries. Conversely do not retrain merely to spend tokens or repeat completed expensive runs without a new hypothesis.

Start with smoke and a bounded diagnostic seed; retain failures and training curves/checkpoints, then extend promising changes to at least three independent training seeds with fair fixed-H tuning and disclosed total budgets. Distinguish real gradient updates from finite search/reselection. Label method changes IMPROVED, preserve ORIGINAL/reference results, and keep sealed final test untouched. Scenario redesign remains authorized with versioned protocols and fair baselines. Preserve the currently running experiment and do not restart the service to deliver these instructions. Continue autonomously after each bounded result.
