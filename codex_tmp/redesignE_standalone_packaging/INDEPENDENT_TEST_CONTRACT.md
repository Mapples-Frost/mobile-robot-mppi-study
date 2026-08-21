# Redesign E Independent Test Contract

## Identity

- Test ID: `redesignE_independent_test_v1`
- Map: `chapter3_redesignE`
- Design: four-arm, matched-case standalone test
- Display aliases: `795100001–795100050`
- Sample size: 50 cases per arm; 200 episodes total

Each display alias occurs exactly once in every arm. The original RNG seed,
case ID and arm identity are stored in each episode's `case_identity.json`.

## Arms

- B00: learning off, probability off
- B01: learning off, probability on
- B10: learning on, probability off
- B11: learning on, probability on

Resolved configurations and expanded physical-scenario identity are audited
from the bundled episode evidence. All four arms must share the same physical
scenario for each matched case.

## Endpoints

- Primary binary endpoint: `metrics.success AND NOT metrics.collision`
- Safety endpoints: collision and minimum clearance
- Efficiency endpoints: completion steps, final goal distance and planner timing

## Statistical contract

- Binary matched comparisons: two-sided exact McNemar test
- Binary effect: paired risk difference with 95% confidence interval
- Continuous matched comparisons: paired t-test when paired differences pass
  the predeclared normality rule; otherwise Wilcoxon signed-rank
- Continuous uncertainty: paired bootstrap confidence interval
- Effect sizes and confidence intervals are reported with p-values

## Scope and reproducibility

This package is analyzed as one self-contained matched-case test. Its report,
tables and tests are generated exclusively from the bundled 200 episode records.
No other experiment's summary, combined estimate or analysis directory is an
input to the reproducible analysis command.
