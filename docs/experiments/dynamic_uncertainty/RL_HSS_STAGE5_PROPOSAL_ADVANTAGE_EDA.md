# Stage 5 proposal-advantage threshold-selection EDA

## Scope and provenance

This is a retrospective analysis of the ten completed Stage 4 Amendment 1
RL/HSS-on trajectory CSV files.  It is development evidence used to select the
Stage 5 mechanism-probe rule.  It is not an independent confirmation dataset,
and no held-out or sealed seed was opened.

The reproducible analysis entry point is
`experiments/dynamic_uncertainty/analyze_stage4_proposal_advantage.py`.

## Data structure and quality

- files: 10 complete RL/HSS-on episode trajectories;
- rows: one row per completed 100 ms control decision;
- comparable decisions: 3,395 rows with both guided and Gaussian costs;
- missingness: expected structural missingness after HSS assigned zero guided
  candidates; those rows were excluded from cost-source comparison;
- comparison: best guided cost minus best Gaussian cost evaluated by the same
  controller cost, state, observation forecast, and control decision;
- no duplicate episode cells and no partial trajectory was included.

## Descriptive results

Of 3,395 comparable decisions, 3,378 (`99.499%`) had

```text
best guided cost - best Gaussian cost > 0,
```

so the best Actor-guided candidate was worse than the best current-Gaussian
candidate under the controller's own objective.  The pooled difference had:

| quantile | raw cost difference | relative disadvantage |
|---:|---:|---:|
| 5% | 6.2643 | 0.01041 |
| 25% | 17.5629 | 0.02853 |
| 50% | 27.6681 | 0.06184 |
| 75% | 57.5356 | 0.29612 |
| 95% | 236.2705 | 8.57678 |

All 10 episodes began with at least three consecutive disadvantages; all 10
also began with at least five.  Episode-level disadvantage fractions ranged
from `0.9645` to `1.0000`.

## Stage 5 rule selected before prospective execution

Stage 5 uses an episode-latched veto after three consecutive comparable
decisions with positive guided-minus-Gaussian minimum cost.  The comparison is
normalized only for diagnostics:

```text
relative Actor advantage
  = (best Gaussian cost - best guided cost)
    / max(abs(best Gaussian cost), 1).
```

The frozen margin is zero and the patience is three decisions.  The rule is
causal: a completed decision updates authority for the next decision.  Once
latched, it cannot recover without new guided evidence, so it remains at zero
until episode reset.  A separate shadow arm computes the same decision but does
not apply it.

## Interpretation limits

The threshold and patience were selected using Stage 4 development outcomes.
They therefore cannot be evaluated as confirmatory on the same seeds.  Stage 5
uses fresh development seeds and is explicitly a small mechanism probe.  Its
three episodes per arm can establish wiring, safety invariants, and a strong
directional failure or recovery; it cannot estimate a publication-level effect
size or generalize to the sealed evaluation distribution.
